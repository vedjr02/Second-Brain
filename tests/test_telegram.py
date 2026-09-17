"""Tests for app.telegram: message routing and application construction."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import Update

import app.pipeline as pipeline_module
import app.telegram as telegram_module
from app.settings import Settings
from app.telegram import build_application, handle_message, handle_start


def _settings() -> Settings:
    return Settings(
        telegram_bot_token="stub-token",
        telegram_webhook_secret="s",
        webhook_base_url="",
        llm_api_key="stub-key",
    )


def _context(settings: Settings | None = None) -> MagicMock:
    context = MagicMock()
    context.application.bot_data = {"settings": settings or _settings()}
    return context


def _update_from_data(data: dict[str, Any], bot: Any) -> Update:
    update = Update.de_json(data, bot)
    assert update is not None
    return update


def _text_update(text: str, bot: Any = None) -> Update:
    data: dict[str, Any] = {
        "update_id": 7,
        "message": {
            "message_id": 11,
            "date": 0,
            "chat": {"id": 42, "type": "private"},
            "text": text,
        },
    }
    return _update_from_data(data, bot)


async def test_handle_message_routes_text_to_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Update] = []

    async def fake_handle_text(update: Update, context: Any) -> None:
        calls.append(update)

    monkeypatch.setattr(pipeline_module, "handle_text_message", fake_handle_text)
    update = _text_update("hello second brain")
    await handle_message(update, _context())
    assert calls == [update]


async def test_handle_message_routes_photos_to_photo_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text_calls: list[Update] = []
    photo_calls: list[Update] = []

    async def fake_handle_text(update: Update, context: Any) -> None:
        text_calls.append(update)

    async def fake_handle_photo(update: Update, context: Any) -> None:
        photo_calls.append(update)

    monkeypatch.setattr(pipeline_module, "handle_text_message", fake_handle_text)
    monkeypatch.setattr(pipeline_module, "handle_photo_message", fake_handle_photo)
    bot = AsyncMock()
    update = _update_from_data(
        {
            "update_id": 9,
            "message": {
                "message_id": 12,
                "date": 0,
                "chat": {"id": 42, "type": "private"},
                "photo": [
                    {"file_id": "abc", "file_unique_id": "u1", "width": 1, "height": 1}
                ],
            },
        },
        bot,
    )
    await handle_message(update, _context())
    assert text_calls == []  # photos never reach the text pipeline
    assert photo_calls == [update]
    bot.send_message.assert_not_awaited()  # the photo pipeline replies itself


async def test_handle_message_routes_voice_to_voice_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voice_calls: list[Update] = []

    async def fake_handle_voice(update: Update, context: Any) -> None:
        voice_calls.append(update)

    monkeypatch.setattr(pipeline_module, "handle_voice_message", fake_handle_voice)
    bot = AsyncMock()
    update = _update_from_data(
        {
            "update_id": 10,
            "message": {
                "message_id": 13,
                "date": 0,
                "chat": {"id": 42, "type": "private"},
                "voice": {"file_id": "v1", "file_unique_id": "uv1", "duration": 2},
            },
        },
        bot,
    )
    await handle_message(update, _context())
    assert voice_calls == [update]  # voice now goes to the Phase 4 pipeline
    bot.send_message.assert_not_awaited()


async def test_handle_message_replies_to_unsupported_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Update] = []

    async def fake_handle_text(update: Update, context: Any) -> None:
        calls.append(update)

    async def fake_handle_photo(update: Update, context: Any) -> None:
        calls.append(update)

    async def fake_handle_voice(update: Update, context: Any) -> None:
        calls.append(update)

    monkeypatch.setattr(pipeline_module, "handle_text_message", fake_handle_text)
    monkeypatch.setattr(pipeline_module, "handle_photo_message", fake_handle_photo)
    monkeypatch.setattr(pipeline_module, "handle_voice_message", fake_handle_voice)
    bot = AsyncMock()
    update = _update_from_data(
        {
            "update_id": 10,
            "message": {
                "message_id": 13,
                "date": 0,
                "chat": {"id": 42, "type": "private"},
                "sticker": {"file_id": "s1", "file_unique_id": "us1", "width": 1,
                            "height": 1, "is_animated": False, "is_video": False,
                            "type": "regular"},
            },
        },
        bot,
    )
    await handle_message(update, _context())
    assert calls == []  # stickers never reach a pipeline
    (_args, kwargs) = bot.send_message.await_args
    assert kwargs["chat_id"] == 42
    assert "can't do anything with that kind of message" in kwargs["text"]


async def test_handle_start_replies_with_intro() -> None:
    bot = AsyncMock()
    update = _text_update("/start", bot)
    await handle_start(update, MagicMock())
    (_args, kwargs) = bot.send_message.await_args
    assert kwargs["chat_id"] == 42
    assert "second brain" in kwargs["text"]


def test_build_application_registers_handlers() -> None:
    settings = _settings()
    application = build_application(settings)
    group_zero = application.handlers[0]
    # 7 commands (/start /help /chatid /recent /forget /status /backup)
    # + the text handler + the non-text handler
    assert len(group_zero) == 9
    assert application.bot_data["settings"] is settings
    assert application.updater is None  # webhook mode: no polling updater


# --- owner gate and chat commands ------------------------------------------


async def test_messages_from_other_chats_are_refused_when_owner_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public bot username means strangers can write to your brain."""
    calls: list[Update] = []

    async def fake_handle_text(update: Update, context: Any) -> None:
        calls.append(update)

    monkeypatch.setattr(pipeline_module, "handle_text_message", fake_handle_text)
    bot = AsyncMock()
    update = _text_update("hello", bot)
    settings = Settings(
        telegram_bot_token="t",
        telegram_webhook_secret="s",
        llm_api_key="k",
        owner_chat_id=999,  # the update above comes from chat 42
    )

    await handle_message(update, _context(settings))

    assert calls == []
    assert "only answers to its owner" in bot.send_message.await_args.kwargs["text"]


async def test_owner_messages_still_route_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Update] = []

    async def fake_handle_text(update: Update, context: Any) -> None:
        calls.append(update)

    monkeypatch.setattr(pipeline_module, "handle_text_message", fake_handle_text)
    update = _text_update("hello")
    settings = Settings(
        telegram_bot_token="t",
        telegram_webhook_secret="s",
        llm_api_key="k",
        owner_chat_id=42,
    )

    await handle_message(update, _context(settings))

    assert calls == [update]


async def test_recent_lists_saved_memories_with_their_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import datetime, timezone

    from app.memory import SavedChunk

    monkeypatch.setattr(
        telegram_module.memory,
        "recent_chunks",
        lambda chat_id, limit: [
            SavedChunk(7, "the spare key is under the mat", datetime(2026, 9, 1, tzinfo=timezone.utc))
        ],
    )
    bot = AsyncMock()
    update = _text_update("/recent", bot)

    await telegram_module.handle_recent(update, _context())

    text = bot.send_message.await_args.kwargs["text"]
    assert "7. the spare key is under the mat" in text
    assert "/forget" in text


async def test_forget_deletes_by_number_and_confirms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted: list[tuple[int, int]] = []

    def fake_delete(chat_id: int, chunk_id: int) -> str:
        deleted.append((chat_id, chunk_id))
        return "the spare key is under the mat"

    bot = AsyncMock()
    update = _text_update("/forget 7", bot)
    context = _context()
    context.args = ["7"]
    context.bot = bot
    monkeypatch.setattr(telegram_module.memory, "delete_chunk", fake_delete)
    await telegram_module.handle_forget(update, context)

    assert deleted == [(42, 7)]
    assert "Forgotten" in bot.send_message.await_args.kwargs["text"]


async def test_forget_without_a_number_explains_itself() -> None:
    bot = AsyncMock()
    update = _text_update("/forget", bot)
    context = _context()
    context.args = []

    await telegram_module.handle_forget(update, context)

    assert "/forget <number>" in bot.send_message.await_args.kwargs["text"]


async def test_status_warns_loudly_when_backups_are_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telegram_module.memory, "count_chunks", lambda chat_id: 12)
    monkeypatch.setattr(
        telegram_module.memory, "count_pending_reminders", lambda chat_id: 1
    )
    bot = AsyncMock()
    update = _text_update("/status", bot)

    await telegram_module.handle_status(update, _context())  # owner_chat_id unset

    text = bot.send_message.await_args.kwargs["text"]
    assert "12 memories saved" in text
    assert "Backups: OFF" in text
