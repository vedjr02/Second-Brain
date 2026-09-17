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
    await handle_message(update, MagicMock())
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
    await handle_message(update, MagicMock())
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
    await handle_message(update, MagicMock())
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
    await handle_message(update, MagicMock())
    assert calls == []  # stickers never reach a pipeline
    (_args, kwargs) = bot.send_message.await_args
    assert kwargs["chat_id"] == 42
    assert "coming in later phases" in kwargs["text"]


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
    assert len(group_zero) == 3  # /start + text handler + non-text handler
    assert application.bot_data["settings"] is settings
    assert application.updater is None  # webhook mode: no polling updater
