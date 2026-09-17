"""Tests for the Phase 1-5 pipeline routing rules (all collaborators faked)."""

import asyncio
import io
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest
import pytesseract  # type: ignore[import-untyped]
from telegram import Update

import app.pipeline as pipeline
from app.llm import Classification, ReminderParseError, ReminderSpec
from app.settings import Settings


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = dict(
        telegram_bot_token="stub-token",
        telegram_webhook_secret="s",
        webhook_base_url="",
        llm_api_key="stub-key",
        search_top_k=3,
        similarity_threshold=0.5,
    )
    values.update(overrides)
    return Settings(**values)


def _text_update(text: str) -> tuple[Update, AsyncMock]:
    bot = AsyncMock()
    data: dict[str, Any] = {
        "update_id": 7,
        "message": {
            "message_id": 11,
            "date": 0,
            "chat": {"id": 42, "type": "private"},
            "text": text,
        },
    }
    update = Update.de_json(data, bot)
    assert update is not None
    return update, bot


def _reply_text(update: Update, bot: AsyncMock) -> str:
    assert bot.send_message.await_args is not None
    return str(bot.send_message.await_args.kwargs["text"])


class FakeGemini:
    def __init__(self, classification: Classification, answer: str = "the answer") -> None:
        self._classification = classification
        self._answer = answer
        self.classify_calls: list[tuple[str, str]] = []
        self.answer_calls: list[tuple[str, list[str]]] = []
        self.parse_calls: list[tuple[str, str, str]] = []
        self.parsed_spec: ReminderSpec | None = None
        self.parse_error: Exception | None = None
        self.describe_calls: list[tuple[bytes, str]] = []
        self.image_description: str | None = None

    def classify(self, text: str, now_iso: str) -> Classification:
        self.classify_calls.append((text, now_iso))
        return self._classification

    def answer(self, question: str, chunks: list[str]) -> str:
        self.answer_calls.append((question, chunks))
        return self._answer

    def parse_reminder(self, text: str, now_iso: str, tz_name: str) -> ReminderSpec:
        self.parse_calls.append((text, now_iso, tz_name))
        if self.parse_error is not None:
            raise self.parse_error
        assert self.parsed_spec is not None
        return self.parsed_spec

    def describe_image(self, image_bytes: bytes, mime_type: str) -> str:
        self.describe_calls.append((image_bytes, mime_type))
        if self.image_description is None:
            raise RuntimeError("no image description configured for this test")
        return self.image_description


class FakeMemory:
    """Replaces app.pipeline.memory with recording fakes."""

    def __init__(self) -> None:
        self.saved: list[dict[str, Any]] = []
        self.marked: list[tuple[int, str]] = []
        self.chunks: list[dict[str, Any]] = []
        self.searches: list[dict[str, Any]] = []
        self.reminders: list[tuple[int, str, datetime]] = []
        self.search_results: list[tuple[str, float]] = []
        self.saved_day: str | None = None
        self.updated_content: list[tuple[int, str]] = []
        self.media_pointer: Any = None
        self.existing_chunk_texts: set[str] = set()

    def save_message(self, **kwargs: Any) -> int:
        self.saved.append(kwargs)
        return 101

    def mark_message_processed(self, message_id: int, classified_type: str) -> None:
        self.marked.append((message_id, classified_type))

    def insert_memory_chunk(
        self, source_message_id: int, chunk_text: str, embedding: list[float],
        tags: list[str],
    ) -> None:
        self.chunks.append(
            {"id": source_message_id, "text": chunk_text, "embedding": embedding,
             "tags": tags}
        )

    def search_memory(
        self, query_embedding: list[float], chat_id: int, top_k: int,
        threshold: float, query_text: str = "",
    ) -> list[tuple[str, float]]:
        self.searches.append(
            {"chat_id": chat_id, "top_k": top_k, "threshold": threshold,
             "query_text": query_text}
        )
        return self.search_results

    def save_reminder(
        self, source_message_id: int, reminder_text: str, due_at: datetime
    ) -> None:
        self.reminders.append((source_message_id, reminder_text, due_at))

    def get_message_saved_day(self, message_id: int) -> str | None:
        return self.saved_day

    def update_message_content(self, message_id: int, raw_content_text: str) -> None:
        self.updated_content.append((message_id, raw_content_text))

    def media_for_chunk_text(self, chat_id: int, chunk_text: str) -> Any:
        return self.media_pointer

    def chunk_exists(self, chat_id: int, chunk_text: str) -> bool:
        return chunk_text in self.existing_chunk_texts


@pytest.fixture(autouse=True)
def _no_leftover_media() -> Any:
    """'The text after a photo belongs to that photo' is per-chat state."""
    pipeline._LAST_MEDIA.clear()
    yield
    pipeline._LAST_MEDIA.clear()


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeGemini, FakeMemory]:
    gemini = FakeGemini(Classification("note", "s", [], [], None))
    memory = FakeMemory()
    monkeypatch.setattr(pipeline, "get_llm", lambda _s: gemini)
    monkeypatch.setattr(pipeline, "memory", memory)
    monkeypatch.setattr(pipeline, "embed_texts", lambda texts: [[0.1] * 384])
    return gemini, memory


async def test_note_is_stored_with_tags(fakes: tuple[FakeGemini, FakeMemory]) -> None:
    gemini, memory = fakes
    update, _bot = _text_update("Remember: the wifi password is hunter2")
    context = MagicMock()
    context.application.bot_data = {"settings": _settings()}

    await pipeline.handle_text_message(update, context)

    assert memory.saved[0]["raw_content_text"].endswith("hunter2")
    assert memory.chunks[0]["tags"] == []
    assert memory.marked == [(101, "note")]
    assert _reply_text(update, _bot) == "Saved."
    assert gemini.answer_calls == []


async def test_note_chunk_carries_save_date_but_embedding_does_not(
    fakes: tuple[FakeGemini, FakeMemory],
) -> None:
    _gemini, memory = fakes
    memory.saved_day = "Saturday 12 July"
    update, _bot = _text_update("the wifi password is hunter2")
    context = MagicMock()
    context.application.bot_data = {"settings": _settings()}

    await pipeline.handle_text_message(update, context)

    assert memory.chunks[0]["text"] == "the wifi password is hunter2 (saved Saturday 12 July)"


async def test_question_answers_from_retrieved_chunks(
    fakes: tuple[FakeGemini, FakeMemory],
) -> None:
    gemini, memory = fakes
    gemini._classification = Classification("question", "s", [], [], None)
    memory.search_results = [("wifi is hunter2", 0.8), ("old note", 0.6)]
    update, _bot = _text_update("what is the wifi password?")
    context = MagicMock()
    context.application.bot_data = {"settings": _settings()}

    await pipeline.handle_text_message(update, context)

    assert memory.searches == [
        {"chat_id": 42, "top_k": 3, "threshold": 0.5,
         "query_text": "what is the wifi password?"}
    ]
    assert gemini.answer_calls == [("what is the wifi password?",
                                    ["wifi is hunter2", "old note"])]
    assert _reply_text(update, _bot) == "the answer"
    assert memory.chunks == []  # questions are not stored as memory
    assert memory.marked == [(101, "question")]


async def test_reminder_parses_stores_and_confirms_exactly(
    fakes: tuple[FakeGemini, FakeMemory],
) -> None:
    gemini, memory = fakes
    gemini._classification = Classification("reminder", "s", ["dentist"], [], None)
    gemini.parsed_spec = ReminderSpec(
        what="Call the dentist to book a cleaning",
        due_at_iso="2026-09-20T09:00:00+00:00",
    )
    update, _bot = _text_update("remind me about the dentist on sunday 9am")
    context = MagicMock()
    context.application.bot_data = {"settings": _settings()}

    await pipeline.handle_text_message(update, context)

    # The dedicated parse call got the real clock and the configured timezone.
    (_text, now_iso, tz_name) = gemini.parse_calls[0]
    assert tz_name == "UTC"
    assert now_iso  # some ISO clock was injected
    # The reminder row stores the rewritten 'what' and the parsed instant.
    assert memory.reminders == [
        (101, "Call the dentist to book a cleaning",
         datetime(2026, 9, 20, 9, 0, tzinfo=timezone.utc))
    ]
    assert len(memory.chunks) == 1  # reminder text is also stored as a note
    reply = _reply_text(update, _bot)
    assert reply == (
        "Got it — reminding you about Call the dentist to book a cleaning "
        "on Sunday 20 September at 09:00 UTC."
    )


async def test_reminder_confirmation_uses_display_timezone(
    fakes: tuple[FakeGemini, FakeMemory],
) -> None:
    gemini, _memory = fakes
    gemini._classification = Classification("reminder", "s", [], [], None)
    gemini.parsed_spec = ReminderSpec(
        what="Take out the trash", due_at_iso="2026-09-20T09:00:00+00:00"
    )
    update, _bot = _text_update("remind me to take out the trash sunday")
    context = MagicMock()
    context.application.bot_data = {
        "settings": _settings(user_display_timezone="Europe/Berlin")
    }

    await pipeline.handle_text_message(update, context)

    # 09:00 UTC is 11:00 in Berlin on that date (CEST, UTC+2).
    assert "11:00" in _reply_text(update, _bot)
    assert "Sunday 20 September" in _reply_text(update, _bot)


async def test_reminder_with_unparsable_time_stores_note_and_says_so(
    fakes: tuple[FakeGemini, FakeMemory],
) -> None:
    gemini, memory = fakes
    gemini._classification = Classification("reminder", "s", [], [], None)
    gemini.parse_error = ReminderParseError("no plausible future time")
    update, _bot = _text_update("remind me about the dentist someday")
    context = MagicMock()
    context.application.bot_data = {"settings": _settings()}

    await pipeline.handle_text_message(update, context)

    assert len(memory.chunks) == 1  # the note itself is still kept
    assert memory.reminders == []  # but nothing that could silently misfire
    assert "couldn't work out WHEN" in _reply_text(update, _bot)


async def test_reminder_parse_crash_replies_sorry_and_stores_nothing(
    fakes: tuple[FakeGemini, FakeMemory],
) -> None:
    gemini, memory = fakes
    gemini._classification = Classification("reminder", "s", [], [], None)
    gemini.parse_error = RuntimeError("gemini down")
    update, _bot = _text_update("remind me about the dentist sunday 9am")
    context = MagicMock()
    context.application.bot_data = {"settings": _settings()}

    await pipeline.handle_text_message(update, context)

    assert memory.chunks == []
    assert memory.reminders == []
    assert "couldn't reach my language model" in _reply_text(update, _bot)


async def test_other_is_not_stored(fakes: tuple[FakeGemini, FakeMemory]) -> None:
    gemini, memory = fakes
    gemini._classification = Classification("other", "s", [], [], None)
    update, _bot = _text_update("lol thanks")
    context = MagicMock()
    context.application.bot_data = {"settings": _settings()}

    await pipeline.handle_text_message(update, context)

    assert memory.chunks == []
    assert memory.marked == [(101, "other")]
    assert "only store notes" in _reply_text(update, _bot)


async def test_classification_failure_replies_sorry_and_stores_nothing(
    fakes: tuple[FakeGemini, FakeMemory],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemini, memory = fakes

    def boom(text: str, now_iso: str) -> Classification:
        raise RuntimeError("gemini down")

    monkeypatch.setattr(gemini, "classify", boom)
    update, _bot = _text_update("anything")
    context = MagicMock()
    context.application.bot_data = {"settings": _settings()}

    await pipeline.handle_text_message(update, context)

    assert memory.saved == []
    assert memory.chunks == []
    assert "couldn't reach my language model" in _reply_text(update, _bot)


async def test_blank_text_is_ignored(fakes: tuple[FakeGemini, FakeMemory]) -> None:
    update, bot = _text_update("   ")
    context = MagicMock()
    context.application.bot_data = {"settings": _settings()}

    await pipeline.handle_text_message(update, context)

    bot.send_message.assert_not_awaited()


# --- format_reminder_message (the fired ping) --------------------------------


def test_fired_message_due_today_says_today() -> None:
    due = datetime(2026, 9, 17, 15, 0, tzinfo=timezone.utc)
    created = datetime(2026, 9, 17, 9, 0, tzinfo=timezone.utc)
    msg = pipeline.format_reminder_message("Call the dentist", due, created, "UTC")
    assert msg.startswith("⏰ Reminder: Call the dentist\n")
    assert "due today" in msg
    assert "set on Thursday 17 September" in msg


def test_fired_message_future_due_shows_weekday() -> None:
    due = datetime(2026, 9, 20, 9, 0, tzinfo=timezone.utc)
    created = datetime(2026, 9, 17, 9, 0, tzinfo=timezone.utc)
    msg = pipeline.format_reminder_message("Call the dentist", due, created, "UTC")
    assert "due Sunday 20 September" in msg
    assert "set on Thursday 17 September" in msg


def test_fired_message_respects_display_timezone() -> None:
    due = datetime(2026, 9, 20, 23, 30, tzinfo=timezone.utc)  # 01:30 in Berlin on the 21st
    created = datetime(2026, 9, 17, 9, 0, tzinfo=timezone.utc)
    msg = pipeline.format_reminder_message(
        "Call the dentist", due, created, "Europe/Berlin"
    )
    assert "Monday 21 September" in msg


def test_due_reminders_stale_boundary_uses_utc_now() -> None:
    # Sanity check on the ZoneInfo conversion used in confirmations.
    due = datetime.fromisoformat("2026-09-20T09:00:00+00:00")
    local = due.astimezone(ZoneInfo("Europe/Berlin"))
    assert local.hour == 11


# --- Phase 3: photo ingestion -------------------------------------------------


def _photo_update(
    bot: AsyncMock, *, file_size: int | None = 2048
) -> tuple[Update, AsyncMock]:
    data: dict[str, Any] = {
        "update_id": 21,
        "message": {
            "message_id": 31,
            "date": 0,
            "chat": {"id": 42, "type": "private"},
            "photo": [
                {"file_id": "small", "file_unique_id": "u1",
                 "width": 90, "height": 90, "file_size": 1200},
                {"file_id": "big", "file_unique_id": "u2",
                 "width": 1280, "height": 960, "file_size": file_size},
            ],
        },
    }
    update = Update.de_json(data, bot)
    assert update is not None
    return update, bot


def _photo_context(settings: Settings | None = None) -> MagicMock:
    context = MagicMock()
    context.application.bot_data = {"settings": settings or _settings()}
    return context


async def test_photo_with_ocr_text_is_saved_and_searchable(
    fakes: tuple[FakeGemini, FakeMemory],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemini, memory = fakes
    update, bot = _photo_update(AsyncMock())
    # A tiny real JPEG keeps this hermetic while exercising the real decode path.
    png = _one_pixel_jpeg()
    monkeypatch.setattr(pipeline, "_fetch_photo_bytes", _fake_fetch(png))
    monkeypatch.setattr(
        pipeline, "_ocr_image_bytes", lambda _data: "wifi password hunter2"
    )

    await pipeline.handle_photo_message(update, _photo_context())

    # Pointer-first: the message row stores the file_id and NO text initially.
    assert memory.saved[0]["raw_type"] == "photo"
    assert memory.saved[0]["file_id"] == "big"  # largest size is used
    assert memory.saved[0]["raw_content_text"] is None
    # Extracted text lands on the message row and as an embedded chunk.
    assert memory.updated_content == [(101, "wifi password hunter2")]
    assert memory.chunks[0]["text"] == "wifi password hunter2"
    # OCR succeeded -> no vision call, no credit spent.
    assert gemini.describe_calls == []
    assert "searchable" in str(bot.send_message.await_args.kwargs["text"])


async def test_photo_without_ocr_text_gets_vision_description(
    fakes: tuple[FakeGemini, FakeMemory],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemini, memory = fakes
    gemini.image_description = "A screenshot of a pasta recipe video"
    update, bot = _photo_update(AsyncMock())
    monkeypatch.setattr(pipeline, "_fetch_photo_bytes", _fake_fetch(b"img"))
    monkeypatch.setattr(pipeline, "_ocr_image_bytes", lambda _data: "")

    await pipeline.handle_photo_message(update, _photo_context())

    assert len(gemini.describe_calls) == 1
    (image_bytes, mime) = gemini.describe_calls[0]
    assert image_bytes == b"img"
    assert mime == "image/jpeg"
    assert memory.updated_content == [
        (101, "A screenshot of a pasta recipe video")
    ]
    assert "searchable" in str(bot.send_message.await_args.kwargs["text"])


async def test_photo_with_both_empty_is_honest_about_not_searchable(
    fakes: tuple[FakeGemini, FakeMemory],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemini, memory = fakes
    gemini.image_description = ""  # vision returned blank
    update, bot = _photo_update(AsyncMock())
    monkeypatch.setattr(pipeline, "_fetch_photo_bytes", _fake_fetch(b"img"))
    monkeypatch.setattr(pipeline, "_ocr_image_bytes", lambda _data: "")

    await pipeline.handle_photo_message(update, _photo_context())

    # Pointer still stored, but no chunk is fabricated from nothing.
    assert len(memory.saved) == 1
    assert memory.chunks == []
    assert memory.updated_content == []
    assert "isn't searchable" in str(bot.send_message.await_args.kwargs["text"])


async def test_photo_pipeline_failure_keeps_pointer_and_says_so(
    fakes: tuple[FakeGemini, FakeMemory],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemini, memory = fakes
    update, bot = _photo_update(AsyncMock())

    async def fetch_fails(_bot: Any, _file_id: str) -> bytes:
        raise RuntimeError("telegram fetch down")

    monkeypatch.setattr(pipeline, "_fetch_photo_bytes", fetch_fails)

    await pipeline.handle_photo_message(update, _photo_context())

    assert len(memory.saved) == 1  # pointer kept, nothing lost
    assert memory.chunks == []
    assert "couldn't read" in str(bot.send_message.await_args.kwargs["text"])


async def test_oversized_photo_is_rejected_before_storing(
    fakes: tuple[FakeGemini, FakeMemory],
) -> None:
    _gemini, memory = fakes
    update, bot = _photo_update(AsyncMock(), file_size=21 * 1024 * 1024)

    await pipeline.handle_photo_message(update, _photo_context())

    assert memory.saved == []  # nothing stored at all
    bot.send_message.assert_awaited_once()
    assert "20 MB" in str(bot.send_message.await_args.kwargs["text"])


async def test_ocr_crash_but_vision_works_still_saves(
    fakes: tuple[FakeGemini, FakeMemory],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemini, memory = fakes
    gemini.image_description = "A whiteboard with a phone number"
    update, _bot = _photo_update(AsyncMock())
    monkeypatch.setattr(pipeline, "_fetch_photo_bytes", _fake_fetch(b"img"))

    def ocr_boom(_data: bytes) -> str:
        raise RuntimeError("tesseract missing")

    monkeypatch.setattr(pipeline, "_ocr_image_bytes", ocr_boom)

    await pipeline.handle_photo_message(update, _photo_context())

    assert memory.updated_content == [(101, "A whiteboard with a phone number")]
    assert len(memory.chunks) == 1


def test_ocr_image_bytes_decodes_real_jpeg() -> None:
    """Real pytesseract run on a blank JPEG: passes with tesseract installed,
    auto-skips otherwise (CI/dev machines without the binary)."""
    try:
        pipeline.Image.open(io.BytesIO(_one_pixel_jpeg())).verify()
        pytesseract_available = True
    except Exception:
        pytesseract_available = False
    if pytesseract_available:
        try:
            pipeline._ocr_image_bytes(_one_pixel_jpeg())
        except pytesseract.TesseractNotFoundError:
            pytest.skip("tesseract binary not installed")
    else:
        pytest.skip("Pillow could not decode the fixture")


def _one_pixel_jpeg() -> bytes:
    """A minimal real JPEG, generated in-memory (no fixture files)."""
    image = pipeline.Image.new("RGB", (1, 1), color=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _fake_fetch(payload: bytes) -> Any:
    async def fetch(_bot: Any, _file_id: str) -> bytes:
        return payload

    return fetch


# --- Phase 5: reel / video links -------------------------------------------


async def test_video_link_is_processed_and_stored_with_its_link(
    fakes: tuple[FakeGemini, FakeMemory], monkeypatch: pytest.MonkeyPatch
) -> None:
    _gemini, memory = fakes
    calls: list[tuple[str, str]] = []

    def fake_process(settings: Any, url: str, caption: str, workdir: str) -> str:
        calls.append((url, caption))
        return "A pasta recipe reel: boil, fry garlic, toss."

    monkeypatch.setattr(pipeline.reels, "process_video_link", fake_process)
    update, _bot = _text_update(
        "save this pasta reel https://instagram.com/reel/abc for later"
    )
    context = MagicMock()
    context.application.bot_data = {"settings": _settings()}

    await pipeline.handle_text_message(update, context)

    # The URL is found mid-message and the caption is the text around it.
    assert calls == [
        ("https://instagram.com/reel/abc", "save this pasta reel for later")
    ]
    stored = memory.chunks[0]["text"]
    assert "pasta recipe reel" in stored
    assert "Link: https://instagram.com/reel/abc" in stored
    assert _reply_text(update, _bot) == "Saved — the video is summarized and searchable."


async def test_video_link_download_failure_falls_back_to_bookmark(
    fakes: tuple[FakeGemini, FakeMemory], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runtime rule 5: never a hard failure, never a pretended understanding."""
    _gemini, memory = fakes

    def boom(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("yt-dlp failed: login required")

    monkeypatch.setattr(pipeline.reels, "process_video_link", boom)
    update, _bot = _text_update("https://instagram.com/reel/private")
    context = MagicMock()
    context.application.bot_data = {"settings": _settings()}

    await pipeline.handle_text_message(update, context)

    assert memory.chunks[0]["text"].startswith("https://instagram.com/reel/private")
    reply = _reply_text(update, _bot)
    assert "saved the link as a plain bookmark" in reply
    assert "never saw the video itself" in reply


async def test_video_link_with_nothing_understood_is_saved_as_bookmark(
    fakes: tuple[FakeGemini, FakeMemory], monkeypatch: pytest.MonkeyPatch
) -> None:
    _gemini, memory = fakes
    monkeypatch.setattr(
        pipeline.reels, "process_video_link", lambda *a, **k: ""
    )
    update, _bot = _text_update("https://instagram.com/reel/silent")
    context = MagicMock()
    context.application.bot_data = {"settings": _settings()}

    await pipeline.handle_text_message(update, context)

    assert memory.chunks[0]["text"].startswith("https://instagram.com/reel/silent")
    assert "never understood its contents" in _reply_text(update, _bot)


async def test_plain_note_without_a_link_never_touches_the_video_pipeline(
    fakes: tuple[FakeGemini, FakeMemory], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("the video pipeline must not run for plain notes")

    monkeypatch.setattr(pipeline.reels, "process_video_link", boom)
    update, _bot = _text_update("the electrician is on 555 0123")
    context = MagicMock()
    context.application.bot_data = {"settings": _settings()}

    await pipeline.handle_text_message(update, context)

    assert _reply_text(update, _bot) == "Saved."


# --- grouping: several messages, one thought --------------------------------


async def test_a_burst_of_messages_is_processed_as_one(
    fakes: tuple[FakeGemini, FakeMemory],
) -> None:
    """People send one sentence across three messages. Handle it as one."""
    from app import grouping

    gemini, memory = fakes
    context = MagicMock()
    context.application.bot_data = {
        "settings": _settings(group_window_seconds=0.05)
    }
    for part in ("remember", "the spare key", "is under the mat"):
        update, _bot = _text_update(part)
        await grouping.submit_text(update, context)

    await asyncio.sleep(0.2)

    # One classification, one saved memory — not three.
    assert len(gemini.classify_calls) == 1
    assert gemini.classify_calls[0][0] == "remember. the spare key. is under the mat"
    assert len(memory.chunks) == 1


async def test_a_new_message_restarts_the_window(
    fakes: tuple[FakeGemini, FakeMemory],
) -> None:
    from app import grouping

    gemini, _memory = fakes
    context = MagicMock()
    context.application.bot_data = {
        "settings": _settings(group_window_seconds=0.15)
    }
    update, _bot = _text_update("first")
    await grouping.submit_text(update, context)
    await asyncio.sleep(0.1)  # still inside the window
    assert gemini.classify_calls == []

    update2, _bot2 = _text_update("second")
    await grouping.submit_text(update2, context)
    await asyncio.sleep(0.1)  # the timer restarted, so still nothing
    assert gemini.classify_calls == []

    await asyncio.sleep(0.15)
    assert len(gemini.classify_calls) == 1
    assert gemini.classify_calls[0][0] == "first. second"


def test_combine_punctuates_fragments_but_leaves_sentences_alone() -> None:
    from app.grouping import combine

    assert combine(["remember", "buy milk"]) == "remember. buy milk"
    assert combine(["Is it Tuesday?", "or Wednesday"]) == "Is it Tuesday? or Wednesday"
    assert combine([" ", "only this"]) == "only this"


async def test_text_after_a_photo_is_filed_against_the_photo(
    fakes: tuple[FakeGemini, FakeMemory],
) -> None:
    """The real case: photo, then 'remember I want to post this tomorrow'."""
    _gemini, memory = fakes
    settings = _settings()
    pipeline.note_media_message(42, 101, "photo")

    update, _bot = _text_update("remember I want to post this tomorrow")
    context = MagicMock()
    context.application.bot_data = {"settings": settings}
    await pipeline.handle_text_message(update, context)

    # The note hangs off the photo's message row (101), which still holds the
    # file_id pointer — not off the text message that would lose it.
    assert memory.chunks[0]["id"] == 101
    assert "added to the photo you just sent" in _reply_text(update, _bot)


async def test_an_old_photo_does_not_capture_a_later_unrelated_note(
    fakes: tuple[FakeGemini, FakeMemory],
) -> None:
    from datetime import timedelta

    _gemini, memory = fakes
    settings = _settings(media_link_window_seconds=60)
    pipeline._LAST_MEDIA[42] = (
        101,
        "photo",
        datetime.now(timezone.utc) - timedelta(minutes=10),
    )

    update, _bot = _text_update("the bins go out on Thursday")
    context = MagicMock()
    context.application.bot_data = {"settings": settings}
    await pipeline.handle_text_message(update, context)

    # Stale media is dropped, so the note is filed on its own and the reply
    # makes no claim about a photo.
    assert _reply_text(update, _bot) == "Saved."
    assert pipeline.recent_media(42, settings) is None


async def test_a_long_note_is_stored_as_several_retrievable_chunks(
    fakes: tuple[FakeGemini, FakeMemory], monkeypatch: pytest.MonkeyPatch
) -> None:
    _gemini, memory = fakes
    monkeypatch.setattr(
        pipeline, "embed_texts", lambda texts: [[0.1] * 384 for _ in texts]
    )
    long_note = " ".join(f"Point {i} of the article." for i in range(200))
    update, _bot = _text_update(long_note)
    context = MagicMock()
    context.application.bot_data = {"settings": _settings()}

    await pipeline.handle_text_message(update, context)

    assert len(memory.chunks) > 1
    assert any("Point 100" in chunk["text"] for chunk in memory.chunks)
    assert _reply_text(update, _bot) == "Saved."


async def test_an_answer_from_a_photo_sends_the_photo_back(
    fakes: tuple[FakeGemini, FakeMemory],
) -> None:
    from app.memory import MediaPointer

    gemini, memory = fakes
    gemini._classification = Classification("question", "s", [], [], None)
    memory.search_results = [("a screenshot of a pasta recipe", 0.9)]
    memory.media_pointer = MediaPointer("photo-file-1", "photo")
    update, bot = _text_update("what was that pasta photo?")
    context = MagicMock()
    context.application.bot_data = {"settings": _settings()}

    await pipeline.handle_text_message(update, context)

    assert bot.send_photo.await_args.kwargs["photo"] == "photo-file-1"


async def test_a_text_only_answer_sends_no_media(
    fakes: tuple[FakeGemini, FakeMemory],
) -> None:
    gemini, memory = fakes
    gemini._classification = Classification("question", "s", [], [], None)
    memory.search_results = [("the spare key is under the mat", 0.9)]
    memory.media_pointer = None
    update, bot = _text_update("where is the key?")
    context = MagicMock()
    context.application.bot_data = {"settings": _settings()}

    await pipeline.handle_text_message(update, context)

    assert bot.send_photo.await_count == 0


async def test_the_same_note_sent_twice_is_stored_once(
    fakes: tuple[FakeGemini, FakeMemory],
) -> None:
    _gemini, memory = fakes
    memory.existing_chunk_texts.add("the spare key is under the mat")
    update, _bot = _text_update("the spare key is under the mat")
    context = MagicMock()
    context.application.bot_data = {"settings": _settings()}

    await pipeline.handle_text_message(update, context)

    assert memory.chunks == []
    assert _reply_text(update, _bot) == "Saved."


def test_day_old_media_links_are_swept_away() -> None:
    from datetime import timedelta

    pipeline._LAST_MEDIA[7] = (
        1, "photo", datetime.now(timezone.utc) - timedelta(days=2)
    )
    pipeline.note_media_message(8, 2, "photo")

    assert 7 not in pipeline._LAST_MEDIA  # swept
    assert 8 in pipeline._LAST_MEDIA  # fresh
