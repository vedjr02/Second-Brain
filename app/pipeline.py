"""Phase 1-5 pipeline: classify text, ingest photos, voice notes and video.

Text routing (per the build spec):
- question -> retrieve top-k chunks above the threshold, answer strictly from
  them via the LLM
- note / bookmark -> embed and store the raw text as a memory chunk
- reminder -> embed/store the text as a note AND persist a reminder row after
  a dedicated date-parsing call; confirm back exactly what was understood so
  a bad date-parse is caught immediately (runtime rule 4)
- other -> reply that nothing was stored

Photo ingestion (Phase 3): the raw image is never stored anywhere — Telegram
remains the blob store. The message row keeps only the file_id pointer, the
file is fetched ONCE for local OCR (pytesseract), and a vision-model call
adds a one-line description only when OCR finds no usable text and a vision
model is configured. The combined text is stored on the message row and
embedded as a memory chunk.
"""

import asyncio
import io
import logging
import os
import tempfile
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytesseract  # type: ignore[import-untyped]
from PIL import Image
from telegram import Bot, Message, Update
from telegram.ext import ContextTypes

from . import memory, reels, voice
from .embeddings import embed_texts, split_for_embedding
from .llm import Classification, LLMVisionUnavailableError, ReminderParseError
from .llm import get_llm
from .settings import Settings

logger = logging.getLogger(__name__)

_OTHER_REPLY = (
    "Got it — I only store notes, links, reminders, and answer questions for now."
)
_SORRY_REPLY = (
    "I couldn't reach my language model just now, so I haven't filed that "
    "properly — send it again in a moment."
)
_REMINDER_BAD_TIME_REPLY = (
    "I saved that as a note, but I couldn't work out WHEN to remind you. "
    "Send it again with a day and time (e.g. 'remind me about the dentist "
    "Sunday 9am') and I'll set the reminder."
)

# Telegram's standard Bot API file-size cap: larger media can't be fetched.
_TELEGRAM_FILE_SIZE_CAP = 20 * 1024 * 1024
_PHOTO_MIME = "image/jpeg"
_OCR_HINT_REPLY = (
    "I kept the image, but couldn't read it right now (text extraction is "
    "unavailable). Try again later — it's not searchable yet."
)
_BOOKMARK_FALLBACK_REPLY = (
    "I couldn't download that video (the platform may block me), so I saved "
    "the link as a plain bookmark — searchable by its link and caption, but "
    "I never saw the video itself."
)
_BOOKMARK_ONLY_REPLY = (
    "I got the video but couldn't make out anything in it (no speech, no "
    "on-screen text), so I saved it as a plain bookmark — I never understood "
    "its contents."
)
_STORED_WITHOUT_TEXT_REPLY = (
    "Saved — but I couldn't read any text or describe that image, so it isn't "
    "searchable yet."
)


# The last media message each chat sent, so a text that follows it can be
# understood as its caption ("remember I want to post this tomorrow").
_LAST_MEDIA: dict[int, tuple[int, str, datetime]] = {}


def note_media_message(chat_id: int, message_id: int, label: str) -> None:
    """Record that this chat just sent a photo/voice/video."""
    _LAST_MEDIA[chat_id] = (message_id, label, datetime.now(UTC))


def recent_media(chat_id: int, settings: Settings) -> tuple[int, str] | None:
    """The media this chat sent moments ago, if it is still fresh."""
    entry = _LAST_MEDIA.get(chat_id)
    if entry is None:
        return None
    message_id, label, when = entry
    age = (datetime.now(UTC) - when).total_seconds()
    if age > settings.media_link_window_seconds:
        del _LAST_MEDIA[chat_id]
        return None
    return message_id, label


def clear_recent_media(chat_id: int) -> None:
    _LAST_MEDIA.pop(chat_id, None)


def utc_now_iso() -> str:
    """Current UTC time as ISO 8601, passed to the LLM calls as their clock."""
    return datetime.now(UTC).isoformat()


async def handle_text_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text_override: str | None = None,
) -> None:
    """Classify the text and apply the routing rules, replying in Telegram.

    `text_override` carries the combined text of a burst of messages (see
    grouping.py); the update itself is still the last message of that burst,
    which is what gets replied to.
    """
    message = update.effective_message
    if message is None:
        return
    raw = text_override if text_override is not None else message.text
    if raw is None or not raw.strip():
        return
    settings: Settings = context.application.bot_data["settings"]
    chat_id = message.chat_id
    text = raw.strip()

    llm = get_llm(settings)
    try:
        classification = await asyncio.to_thread(
            llm.classify, text, utc_now_iso()
        )
    except Exception:
        logger.exception("classification failed for chat %s", chat_id)
        await message.reply_text(_SORRY_REPLY)
        return

    message_id = memory.save_message(
        chat_id=chat_id,
        telegram_message_id=message.message_id,
        direction="in",
        raw_type="text",
        file_id=None,
        raw_content_text=text,
    )
    memory.mark_message_processed(message_id, classification.message_type)

    if classification.message_type == "question":
        await _answer_question(message, settings, chat_id, text)
    elif classification.message_type in ("note", "bookmark"):
        if reels.is_video_link(text):
            await _handle_video_link(
                message, context, message_id, text, classification
            )
        else:
            linked = await _store_note_async(
                message_id, text, classification, settings, chat_id
            )
            await message.reply_text(
                f"Saved — added to the {linked} you just sent." if linked
                else "Saved."
            )
    elif classification.message_type == "reminder":
        await _handle_reminder(
            message, settings, message_id, text, classification
        )
    else:
        await message.reply_text(_OTHER_REPLY)


async def _answer_question(
    message: Message, settings: Settings, chat_id: int, text: str
) -> None:
    """Retrieve similar chunks and answer strictly from them."""
    llm = get_llm(settings)
    query_vec = (await asyncio.to_thread(embed_texts, [text]))[0]
    results = memory.search_memory(
        query_vec,
        chat_id=chat_id,
        top_k=settings.search_top_k,
        threshold=settings.similarity_threshold,
        query_text=text,
    )
    chunks = [chunk for chunk, _similarity in results]
    answer = await asyncio.to_thread(llm.answer, text, chunks)
    await message.reply_text(answer)


async def _handle_reminder(
    message: Message,
    settings: Settings,
    message_id: int,
    text: str,
    classification: Classification,
) -> None:
    """Resolve the reminder time, store note + reminder, confirm back (rule 4).

    A dedicated parsing call turns the request into an exact due_at (relative
    phrases resolved against the real current time and the user's timezone).
    If no plausible time can be parsed, the note is still saved and the user
    is told plainly — nothing silently misfires later.
    """
    llm = get_llm(settings)
    try:
        spec = await asyncio.to_thread(
            llm.parse_reminder,
            text,
            utc_now_iso(),
            settings.user_display_timezone,
        )
    except ReminderParseError:
        logger.warning(
            "reminder without usable due time (message %s) - stored as note only",
            message_id,
        )
        await _store_note_async(
            message_id, text, classification, settings, message.chat_id
        )
        await message.reply_text(_REMINDER_BAD_TIME_REPLY)
        return
    except Exception:
        logger.exception("reminder parsing failed for chat %s", message.chat_id)
        await message.reply_text(_SORRY_REPLY)
        return

    due_at = _coerce_utc(datetime.fromisoformat(spec.due_at_iso), settings)
    await _store_note_async(
        message_id, text, classification, settings, message.chat_id
    )
    memory.save_reminder(message_id, spec.what, due_at)
    await message.reply_text(
        _reminder_confirmation(spec.what, due_at, settings.user_display_timezone)
    )


def _coerce_utc(due_at: datetime, settings: Settings) -> datetime:
    """The LLM is told to return an offset; attach the display tz if it didn't."""
    if due_at.tzinfo is None:
        return due_at.replace(tzinfo=ZoneInfo(settings.user_display_timezone))
    return due_at


def _reminder_confirmation(what: str, due_at: datetime, tz_name: str) -> str:
    """Echo exactly what was understood, in the user's local time (rule 4)."""
    local = due_at.astimezone(ZoneInfo(tz_name))
    when = local.strftime("%A %d %B at %H:%M %Z")
    return f"Got it — reminding you about {what} on {when}."


async def handle_photo_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Ingest a photo: pointer-first, then fetch once + OCR (+ vision).

    Honesty rules: if the fetch/OCR pipeline fails, the pointer is kept but
    the user is told the image is not searchable; if the image simply contains
    no text, a one-line vision-model description is used instead; and if even
    that yields nothing, the user is told it is stored but not searchable.
    """
    message = update.effective_message
    if message is None or not message.photo:
        return
    settings: Settings = context.application.bot_data["settings"]
    photo = message.photo[-1]  # Telegram sends the largest size last
    file_id = photo.file_id
    chat_id = message.chat_id

    if (photo.file_size or 0) > _TELEGRAM_FILE_SIZE_CAP:
        logger.warning(
            "photo too large to fetch: chat_id=%s size=%s", chat_id, photo.file_size
        )
        await message.reply_text(
            "That image is too large for me to read (over the 20 MB Bot API "
            "cap). Send a smaller version and I'll remember that instead."
        )
        return

    message_id = memory.save_message(
        chat_id=chat_id,
        telegram_message_id=message.message_id,
        direction="in",
        raw_type="photo",
        file_id=file_id,
        raw_content_text=None,  # enriched after OCR below
    )
    note_media_message(chat_id, message_id, "photo")

    try:
        photo_bytes = await _fetch_photo_bytes(context.bot, file_id)
    except Exception:
        logger.exception("photo fetch failed for chat %s", chat_id)
        await message.reply_text(_OCR_HINT_REPLY)
        return
    await _extract_and_store_image(message, message_id, photo_bytes, settings)


async def _fetch_photo_bytes(bot: Bot, file_id: str) -> bytes:
    """Fetch the image bytes from Telegram exactly once (pointer-first rule)."""
    tg_file = await bot.get_file(file_id)
    buffer = io.BytesIO()
    await tg_file.download_to_memory(buffer)
    return buffer.getvalue()


def _ocr_image_bytes(data: bytes) -> str:
    """Local OCR via pytesseract; whitespace-collapsed text ('' when none)."""
    with Image.open(io.BytesIO(data)) as image:
        text = pytesseract.image_to_string(image)
    return " ".join(text.split())


def _combine_photo_text(ocr_text: str, vision_desc: str) -> str:
    """Join the OCR text and vision description into one chunk text."""
    parts = [part for part in (ocr_text, vision_desc) if part]
    return " ".join(parts)


# --- shared media ingestion (voice notes, audio, video, document images) -----


def _is_telegram_file_too_big(file_size: int | None) -> bool:
    return (file_size or 0) > _TELEGRAM_FILE_SIZE_CAP


async def _handle_video_link(
    message: Message,
    context: ContextTypes.DEFAULT_TYPE,
    message_id: int,
    text: str,
    classification: Classification,
) -> None:
    """Phase 5: reel/video link -> yt-dlp -> transcript+OCR -> summary.

    Rule 5 fallback: if the download or processing fails for any reason, the
    link (+ caption) is stored as a lightweight bookmark and the bot says so —
    never pretending the content was understood when it wasn't.
    """
    settings: Settings = context.application.bot_data["settings"]
    url = reels.extract_url(text)
    caption = reels.strip_url(text)
    await message.reply_text(
        "⏳ Downloading and processing that video — this can take a minute."
    )
    with tempfile.TemporaryDirectory(prefix="reel-") as tmp:
        try:
            combined = await asyncio.to_thread(
                reels.process_video_link, settings, url, caption, tmp
            )
        except Exception:
            logger.exception(
                "reel processing failed for chat %s; storing as bookmark",
                message.chat_id,
            )
            await _store_note_async(message_id, text, classification)
            await message.reply_text(_BOOKMARK_FALLBACK_REPLY)
            return
    if not combined:
        # Nothing was actually understood — rule 5: save it as a bookmark and
        # say so, never claim the video was watched.
        await _store_note_async(message_id, text, classification)
        await message.reply_text(_BOOKMARK_ONLY_REPLY)
        return
    await _store_media_summary(
        message, message_id, f"{combined}\nLink: {url}", "video"
    )


async def _download_telegram_file(bot: Bot, file_id: str) -> str:
    """Download a Telegram file to a temp path; caller removes it."""
    tg_file = await bot.get_file(file_id)
    fd, path = tempfile.mkstemp(suffix=".media")
    os.close(fd)
    try:
        await tg_file.download_to_drive(custom_path=path)
    except Exception:
        os.unlink(path)
        raise
    return path


async def _ingest_transcribed_media(
    message: Message,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    raw_type: str,
    file_id: str,
    file_size: int | None,
    media_label: str,
) -> None:
    """Pointer-first save, transcribe locally, route transcript as text.

    The transcript is treated exactly like an incoming text message (Phase 4
    spec): classified, stored, answered — the only difference is the reply
    prefix, which quotes what was heard so a bad transcription is caught.
    """
    settings: Settings = context.application.bot_data["settings"]
    if _is_telegram_file_too_big(file_size):
        logger.warning(
            "%s too large to fetch: chat_id=%s size=%s",
            media_label,
            message.chat_id,
            file_size,
        )
        await message.reply_text(
            f"That {media_label} is too large for me to read (over the 20 MB "
            "Bot API cap). Send a smaller one and I'll remember that instead."
        )
        return

    message_id = memory.save_message(
        chat_id=message.chat_id,
        telegram_message_id=message.message_id,
        direction="in",
        raw_type=raw_type,
        file_id=file_id,
        raw_content_text=None,
    )
    note_media_message(message.chat_id, message_id, media_label)

    try:
        path = await _download_telegram_file(context.bot, file_id)
    except Exception:
        logger.exception("%s download failed for chat %s", media_label, message.chat_id)
        await message.reply_text(
            f"I kept the {media_label}, but couldn't download it just now. "
            "Try again later — it's not searchable yet."
        )
        return

    try:
        transcript = await asyncio.to_thread(voice.transcribe, path)
    except Exception:
        logger.exception("transcription failed for chat %s", message.chat_id)
        await message.reply_text(
            f"I kept the {media_label}, but couldn't transcribe it right now "
            "(speech-to-text is unavailable). Try again later."
        )
        return
    finally:
        os.unlink(path)

    if not transcript:
        await message.reply_text(
            f"I kept the {media_label}, but couldn't hear any words in it — "
            "so it isn't searchable yet."
        )
        return

    memory.update_message_content(message_id, transcript)
    await message.reply_text(f"🎙 I heard: “{transcript}”")
    await _route_text(message, context, message_id, transcript)


async def _route_text(
    message: Message,
    context: ContextTypes.DEFAULT_TYPE,
    message_id: int,
    text: str,
) -> None:
    """Run the Phase 1 routing on already-stored text (no re-saving)."""
    settings: Settings = context.application.bot_data["settings"]
    llm = get_llm(settings)
    try:
        classification = await asyncio.to_thread(
            llm.classify, text, utc_now_iso()
        )
    except Exception:
        logger.exception("classification of transcript failed", extra={"chat": message.chat_id})
        await message.reply_text(_SORRY_REPLY)
        return
    memory.mark_message_processed(message_id, classification.message_type)

    if classification.message_type == "question":
        await _answer_question(message, settings, message.chat_id, text)
    elif classification.message_type in ("note", "bookmark"):
        await _store_note_async(message_id, text, classification)
        await message.reply_text("Saved.")
    elif classification.message_type == "reminder":
        await _handle_reminder(message, settings, message_id, text, classification)
    else:
        await message.reply_text(_OTHER_REPLY)


async def handle_voice_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Voice notes, audio files, and circular video notes -> transcript -> text."""
    message = update.effective_message
    if message is None:
        return
    media = message.voice or message.audio or message.video_note
    if media is None:
        return
    await _ingest_transcribed_media(
        message,
        context,
        raw_type="voice" if (message.voice or message.video_note) else "audio",
        file_id=media.file_id,
        file_size=media.file_size,
        media_label="voice note" if (message.voice or message.video_note) else "audio file",
    )


async def handle_document_image_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Images sent as files (no compression) -> same pipeline as photos."""
    message = update.effective_message
    if message is None or message.document is None:
        return
    document = message.document
    if (document.file_size or 0) > _TELEGRAM_FILE_SIZE_CAP:
        await message.reply_text(
            "That image is too large for me to read (over the 20 MB Bot API "
            "cap). Send a smaller version and I'll remember that instead."
        )
        return
    message_id = memory.save_message(
        chat_id=message.chat_id,
        telegram_message_id=message.message_id,
        direction="in",
        raw_type="photo",
        file_id=document.file_id,
        raw_content_text=None,
    )
    note_media_message(message.chat_id, message_id, "photo")
    try:
        photo_bytes = await _fetch_photo_bytes(context.bot, document.file_id)
    except Exception:
        logger.exception("document image download failed for chat %s", message.chat_id)
        await message.reply_text(_OCR_HINT_REPLY)
        return
    settings: Settings = context.application.bot_data["settings"]
    await _extract_and_store_image(message, message_id, photo_bytes, settings)


async def _extract_and_store_image(
    message: Message, message_id: int, photo_bytes: bytes, settings: Settings
) -> None:
    """OCR + optional vision on already-downloaded image bytes, then store."""
    llm = get_llm(settings)
    ocr_text = ""
    vision_desc = ""
    try:
        ocr_text = await asyncio.to_thread(_ocr_image_bytes, photo_bytes)
    except Exception:
        logger.warning("OCR unavailable for chat %s", message.chat_id, exc_info=True)
    if not ocr_text:
        try:
            vision_desc = await asyncio.to_thread(
                llm.describe_image, photo_bytes, _PHOTO_MIME
            )
        except LLMVisionUnavailableError:
            logger.info("no vision model configured; image kept OCR-only")
        except Exception:
            logger.exception("vision description failed for chat %s", message.chat_id)

    combined = _combine_photo_text(ocr_text, vision_desc)
    if not combined:
        await message.reply_text(_STORED_WITHOUT_TEXT_REPLY)
        return
    await asyncio.to_thread(_store_extracted_text, message_id, combined)
    if ocr_text:
        await message.reply_text("Saved — text in the image is searchable.")
    else:
        await message.reply_text("Saved — I described what I saw; it's searchable.")


async def handle_telegram_video_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Uploaded video files: transcribe audio, OCR keyframes, summarize."""
    message = update.effective_message
    if message is None or message.video is None:
        return
    settings: Settings = context.application.bot_data["settings"]
    video = message.video
    if _is_telegram_file_too_big(video.file_size):
        await message.reply_text(
            "That video is too large for me to read (over the 20 MB Bot API "
            "cap). Send a link instead (Instagram/TikTok/YouTube) and I'll "
            "process it from the source."
        )
        return
    message_id = memory.save_message(
        chat_id=message.chat_id,
        telegram_message_id=message.message_id,
        direction="in",
        raw_type="video",
        file_id=video.file_id,
        raw_content_text=None,
    )
    note_media_message(message.chat_id, message_id, "video")
    await message.reply_text("⏳ Working on that video — this can take a minute.")
    try:
        path = await _download_telegram_file(context.bot, video.file_id)
    except Exception:
        logger.exception("video download failed for chat %s", message.chat_id)
        await message.reply_text(
            "I kept the video, but couldn't download it just now. Try again "
            "later — it isn't searchable yet."
        )
        return
    try:
        combined = await asyncio.to_thread(
            reels.process_video_file, settings, path, message.caption or ""
        )
    except Exception:
        logger.exception("video processing failed for chat %s", message.chat_id)
        await message.reply_text(
            "I kept the video, but couldn't process it right now — it isn't "
            "searchable yet. Try again later."
        )
        return
    finally:
        os.unlink(path)
    await _store_media_summary(message, message_id, combined, "video")


async def _store_media_summary(
    message: Message, message_id: int, combined: str, label: str
) -> None:
    """Store a processed video/reel summary and confirm honestly."""
    if not combined:
        await message.reply_text(
            f"I kept the {label}, but couldn't extract anything readable from "
            "it — so it isn't searchable yet."
        )
        return
    await asyncio.to_thread(_store_extracted_text, message_id, combined)
    await message.reply_text("Saved — the video is summarized and searchable.")


def _store_extracted_text(message_id: int, combined: str) -> None:
    """Persist extracted media text on the message row and as memory chunks.

    A reel transcript or a page of OCR'd text is often long enough to need
    splitting, same as a long note.
    """
    memory.update_message_content(message_id, combined)
    pieces = split_for_embedding(combined) or [combined]
    for piece, vec in zip(pieces, embed_texts(pieces)):
        memory.insert_memory_chunk(
            source_message_id=message_id,
            chunk_text=piece,
            embedding=vec,
            tags=[],
        )


async def _store_note_async(
    message_id: int,
    text: str,
    classification: Classification,
    settings: Settings | None = None,
    chat_id: int | None = None,
) -> str | None:
    """Store a note off the event loop (embedding is CPU-bound).

    When the chat sent a photo/voice/video moments ago, the note is filed
    against THAT message instead of this one: the text is almost always about
    the media ("remember I want to post this tomorrow"), and filing it there
    keeps the file_id pointer and the words in one memory. Returns the media
    label when that happened, so the reply can say so.
    """
    linked: str | None = None
    target = message_id
    if settings is not None and chat_id is not None:
        media = recent_media(chat_id, settings)
        if media is not None:
            target, linked = media
            clear_recent_media(chat_id)
    await asyncio.to_thread(_store_note, target, text, classification)
    return linked


def _store_note(
    message_id: int,
    text: str,
    classification: Classification,
) -> None:
    """Embed the note text and persist it as a memory chunk.

    The stored chunk text carries the save date (runtime rule 2) so answers
    can say roughly when something was saved; the embedding is computed from
    the clean text so the date suffix never skews retrieval.
    """
    pieces = split_for_embedding(text) or [text]
    vectors = embed_texts(pieces)
    saved_day = memory.get_message_saved_day(message_id)
    for piece, vec in zip(pieces, vectors):
        chunk_text = f"{piece} (saved {saved_day})" if saved_day else piece
        memory.insert_memory_chunk(
            source_message_id=message_id,
            chunk_text=chunk_text,
            embedding=vec,
            tags=classification.topics,
        )


def format_reminder_message(
    what: str, due_at: datetime, created_at: datetime, tz_name: str
) -> str:
    """The Telegram text a fired reminder gets (shared with the worker/tests)."""
    tz = ZoneInfo(tz_name)
    due_local = due_at.astimezone(tz)
    set_day = created_at.astimezone(tz).strftime("%A %d %B")
    if due_local.date() == datetime.now(tz).date():
        when = "today"
    else:
        when = due_local.strftime("%A %d %B")
    return (
        f"⏰ Reminder: {what}\n"
        f"(due {when}, set on {set_day})"
    )
