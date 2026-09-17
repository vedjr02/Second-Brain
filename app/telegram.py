"""Telegram Bot API wiring (webhook mode) and Phase 1-5 handler registration.

The Application runs with `updater=None`: Telegram pushes updates to our
FastAPI webhook endpoint, which calls `application.process_update` directly.
No polling loop, no persistent connection (webhook mode per the build spec).

Text goes to the classification pipeline (Phase 1/2), photos to the OCR+vision
ingestion pipeline (Phase 3), voice/audio/video notes to local transcription
(Phase 4), and videos + reel links to the yt-dlp pipeline (Phase 5). Anything
else (stickers, contacts, polls) gets a plain "not supported" reply.
"""

import asyncio
import logging
from io import BytesIO
from zoneinfo import ZoneInfo

from telegram import Document, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from . import backup, grouping, memory, pipeline
from .embeddings import embed_texts
from .settings import Settings

logger = logging.getLogger(__name__)

_START_REPLY = (
    "Hi! I'm your second brain.\n"
    "Send me notes, links, reels, photos, voice notes or reminders and "
    "I'll remember them.\n"
    "Ask me about anything you've saved and I'll answer from your memory.\n\n"
    "Commands: /recent, /find, /forget, /status, /backup, /help"
)

_HELP_REPLY = (
    "What I do:\n"
    "• *Notes* — just tell me something ('the spare key is under the mat') "
    "and I'll keep it.\n"
    "• *Questions* — ask normally ('where's the spare key?'). I answer only "
    "from what you actually saved, and say so when I have nothing.\n"
    "• *Reminders* — 'remind me about the dentist Sunday 9am'. I echo back "
    "what I understood so a bad date is caught straight away.\n"
    "• *Photos* — I read any text in them (and describe them if a vision "
    "model is configured).\n"
    "• *Voice notes* — transcribed locally, then treated like a text message.\n"
    "• *Reels and video links* — downloaded, transcribed, on-screen text "
    "read, then summarised. If a platform blocks me I save the link as a "
    "bookmark and tell you I never saw the video.\n\n"
    "Commands:\n"
    "/recent — the last things I saved\n"
    "/forget <number> — delete one of them (numbers come from /recent)\n"
    "/find <words> — search your memories without spending a model call\n"
    "/reminders — what is still going to fire\n"
    "/cancel <number> — call one of them off\n"
    "/status — how much I'm holding, and when I last backed up\n"
    "/export — every memory as a plain text file\n"
    "/backup — snapshot the database to this chat right now\n"
    "/chatid — show this chat's id (for OWNER_CHAT_ID)"
)

_NOT_OWNER_REPLY = (
    "This is someone else's personal second brain — it only answers to its "
    "owner."
)

_MEDIA_REPLY = (
    "I can't do anything with that kind of message. Send me text, links, "
    "photos, voice notes, or videos and I'll remember them."
)


def _is_owner(update: Update, settings: Settings) -> bool:
    """Single-user tool: when OWNER_CHAT_ID is set, only that chat is served.

    Left open when the id is unset so a fresh install is usable before you
    know your own chat id (/chatid tells you).
    """
    if not settings.owner_chat_id:
        return True
    chat = update.effective_chat
    return chat is not None and chat.id == settings.owner_chat_id


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is not None:
        await message.reply_text(_START_REPLY)


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is not None:
        await message.reply_text(_HELP_REPLY)


async def handle_chatid(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Report this chat's id — what OWNER_CHAT_ID needs to be set to."""
    message = update.effective_message
    if message is None:
        return
    await message.reply_text(
        f"This chat's id is {message.chat_id}.\n"
        "Set OWNER_CHAT_ID to it so I only answer to you and can back my "
        "database up to this chat."
    )


async def handle_recent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List the most recent memories, numbered for /forget."""
    message = update.effective_message
    if message is None:
        return
    chunks = await asyncio.to_thread(memory.recent_chunks, message.chat_id, 10)
    if not chunks:
        await message.reply_text("I haven't saved anything yet.")
        return
    lines = [
        f"{chunk.id}. {_shorten(chunk.chunk_text)}"
        f" — {chunk.created_at.strftime('%d %b')}"
        for chunk in chunks
    ]
    await message.reply_text(
        "Most recent first:\n" + "\n".join(lines) + "\n\nDelete one with /forget <number>."
    )


def _shorten(text: str, limit: int = 120) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


async def handle_forget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Delete one memory by the number shown in /recent."""
    message = update.effective_message
    if message is None:
        return
    args = context.args or []
    if len(args) != 1 or not args[0].lstrip("-").isdigit():
        await message.reply_text(
            "Use /forget <number>, with a number from /recent."
        )
        return
    deleted = await asyncio.to_thread(
        memory.delete_chunk, message.chat_id, int(args[0])
    )
    if deleted is None:
        await message.reply_text(
            "I couldn't find a memory with that number — check /recent."
        )
        return
    await message.reply_text(f"Forgotten: {_shorten(deleted)}")


async def handle_find(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Raw search: show matching memories without spending an LLM call.

    Asking a question costs a model call and gives a written answer. /find is
    the cheap version — it lists what is actually stored, which is also how
    you check whether something was saved at all.
    """
    message = update.effective_message
    if message is None:
        return
    query = " ".join(context.args or []).strip()
    if not query:
        await message.reply_text("Use /find <words> to search your memories.")
        return
    settings: Settings = context.application.bot_data["settings"]
    vector = (await asyncio.to_thread(embed_texts, [query]))[0]
    results = await asyncio.to_thread(
        memory.search_memory,
        vector,
        message.chat_id,
        10,
        settings.similarity_threshold,
        query,
    )
    if not results:
        await message.reply_text(f"Nothing saved matches “{query}”.")
        return
    lines = [f"• {_shorten(text)}" for text, _score in results]
    await message.reply_text(f"Matches for “{query}”:\n" + "\n".join(lines))


async def handle_reminders(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """List what is still going to fire, in the user's own timezone."""
    message = update.effective_message
    if message is None:
        return
    settings: Settings = context.application.bot_data["settings"]
    pending = await asyncio.to_thread(memory.pending_reminders, message.chat_id)
    if not pending:
        await message.reply_text("No reminders waiting.")
        return
    tz = ZoneInfo(settings.user_display_timezone)
    lines = [
        f"{item.id}. {_shorten(item.reminder_text, 80)}"
        f" — {item.due_at.astimezone(tz).strftime('%a %d %b %H:%M')}"
        for item in pending
    ]
    await message.reply_text(
        "Coming up:\n" + "\n".join(lines) + "\n\nCancel one with /cancel <number>."
    )


async def handle_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Cancel a pending reminder by the number shown in /reminders."""
    message = update.effective_message
    if message is None:
        return
    args = context.args or []
    if len(args) != 1 or not args[0].lstrip("-").isdigit():
        await message.reply_text(
            "Use /cancel <number>, with a number from /reminders."
        )
        return
    cancelled = await asyncio.to_thread(
        memory.cancel_reminder, message.chat_id, int(args[0])
    )
    if cancelled is None:
        await message.reply_text(
            "No waiting reminder with that number — check /reminders."
        )
        return
    await message.reply_text(f"Cancelled: {_shorten(cancelled, 80)}")


async def handle_export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send every memory back as a plain text file.

    A second brain you cannot get your notes out of is a trap. This is plain
    readable text, not a database file — it needs no tool to open.
    """
    message = update.effective_message
    if message is None:
        return
    chunks = await asyncio.to_thread(memory.all_chunks, message.chat_id)
    if not chunks:
        await message.reply_text("Nothing saved yet, so nothing to export.")
        return
    body = "\n\n".join(
        f"[{chunk.created_at.strftime('%Y-%m-%d')}] {chunk.chunk_text}"
        for chunk in chunks
    )
    payload = BytesIO(body.encode("utf-8"))
    payload.name = "second-brain-export.txt"
    await message.reply_document(
        document=payload,
        filename="second-brain-export.txt",
        caption=f"{len(chunks)} memories, oldest first.",
    )


async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """How much is stored, and whether the backup safety net is armed."""
    message = update.effective_message
    if message is None:
        return
    settings: Settings = context.application.bot_data["settings"]
    saved = await asyncio.to_thread(memory.count_chunks, message.chat_id)
    pending = await asyncio.to_thread(
        memory.count_pending_reminders, message.chat_id
    )
    if settings.owner_chat_id:
        last = await asyncio.to_thread(backup.db.get_meta, "last_backup_at")
        backup_line = (
            f"Last backup: {last[:16].replace('T', ' ')} UTC"
            if last
            else "Last backup: none yet"
        )
    else:
        backup_line = (
            "Backups: OFF — set OWNER_CHAT_ID (see /chatid) or a redeploy "
            "loses everything."
        )
    await message.reply_text(
        f"{saved} memories saved.\n"
        f"{pending} reminder(s) waiting.\n"
        f"{backup_line}"
    )


async def handle_backup(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Force a snapshot to this chat right now."""
    message = update.effective_message
    if message is None:
        return
    settings: Settings = context.application.bot_data["settings"]
    if not settings.owner_chat_id:
        await message.reply_text(
            "Backups are off: set OWNER_CHAT_ID to this chat's id (/chatid) "
            "and restart me."
        )
        return
    if await backup.back_up(settings, context.bot):
        await message.reply_text(
            "Backed up and pinned. If my disk is ever wiped I'll restore "
            "from that pinned file on my next boot."
        )
    else:
        await message.reply_text(
            "The backup didn't go through — check my logs. Nothing was lost."
        )


def _is_image_document(document: Document) -> bool:
    """Images sent as files (not compressed photos) still carry an image mime."""
    return bool(document.mime_type and document.mime_type.startswith("image/"))


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route each supported message type to its pipeline."""
    message = update.effective_message
    if message is None:
        return
    settings: Settings = context.application.bot_data["settings"]
    if not _is_owner(update, settings):
        logger.warning("ignoring message from non-owner chat %s", message.chat_id)
        await message.reply_text(_NOT_OWNER_REPLY)
        return
    if message.text:
        # Buffered, not handled now: the next message may be the other half
        # of the same thought (see grouping.py).
        await grouping.submit_text(update, context)
    elif message.photo:
        await pipeline.handle_photo_message(update, context)
    elif message.document and _is_image_document(message.document):
        await pipeline.handle_document_image_message(update, context)
    elif message.voice or message.audio or message.video_note:
        await pipeline.handle_voice_message(update, context)
    elif message.video:
        # Telegram video uploads are Phase 5; yt-dlp links take that phase's
        # download path, this one reuses the Telegram fetch path.
        await pipeline.handle_telegram_video_message(update, context)
    else:
        logger.info(
            "unsupported media ignored: chat_id=%s message_id=%s type=%s",
            message.chat_id,
            message.message_id,
            type(message).__name__,
        )
        await message.reply_text(_MEDIA_REPLY)


def register_handlers(application: Application) -> None:
    """Register the update handlers shared by webhook and polling modes."""
    application.add_handler(CommandHandler("start", handle_start))
    application.add_handler(CommandHandler("help", handle_help))
    application.add_handler(CommandHandler("chatid", handle_chatid))
    application.add_handler(CommandHandler("recent", handle_recent))
    application.add_handler(CommandHandler("forget", handle_forget))
    application.add_handler(CommandHandler("find", handle_find))
    application.add_handler(CommandHandler("reminders", handle_reminders))
    application.add_handler(CommandHandler("cancel", handle_cancel))
    application.add_handler(CommandHandler("export", handle_export))
    application.add_handler(CommandHandler("status", handle_status))
    application.add_handler(CommandHandler("backup", handle_backup))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    application.add_handler(MessageHandler(~filters.TEXT, handle_message))


def build_application(settings: Settings, polling: bool = False) -> Application:
    """Build the PTB application; webhook mode has no polling updater."""
    builder = ApplicationBuilder().token(settings.telegram_bot_token)
    if not polling:
        builder = builder.updater(None)
    application = builder.build()
    application.bot_data["settings"] = settings
    register_handlers(application)
    return application
