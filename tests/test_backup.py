"""Tests for the Telegram-as-durable-storage backup/restore safety net."""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.backup as backup
from app.settings import Settings


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = dict(
        telegram_bot_token="t",
        telegram_webhook_secret="s",
        llm_api_key="k",
        owner_chat_id=42,
    )
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def fake_db(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    stub = MagicMock()
    stub.get_meta.return_value = None
    stub.is_empty.return_value = True
    # The real snapshot writes a file; back_up then opens it to upload.
    stub.snapshot_to_file.side_effect = lambda path: open(path, "wb").write(b"db")
    monkeypatch.setattr(backup, "db", stub)
    return stub


def test_backups_are_off_without_an_owner_chat(fake_db: MagicMock) -> None:
    assert backup.due_for_backup(_settings(owner_chat_id=0)) is False


def test_first_backup_is_always_due(fake_db: MagicMock) -> None:
    assert backup.due_for_backup(_settings()) is True


def test_backup_waits_out_the_interval(fake_db: MagicMock) -> None:
    now = datetime.now(UTC)
    fake_db.get_meta.return_value = (now - timedelta(hours=1)).isoformat()
    assert backup.due_for_backup(_settings(backup_interval_hours=6), now) is False

    fake_db.get_meta.return_value = (now - timedelta(hours=7)).isoformat()
    assert backup.due_for_backup(_settings(backup_interval_hours=6), now) is True


def test_a_corrupt_timestamp_forces_a_backup(fake_db: MagicMock) -> None:
    fake_db.get_meta.return_value = "not-a-date"
    assert backup.due_for_backup(_settings()) is True


async def test_back_up_sends_and_pins_the_snapshot(fake_db: MagicMock) -> None:
    bot = AsyncMock()
    bot.send_document.return_value = MagicMock(message_id=555)

    assert await backup.back_up(_settings(), bot) is True

    sent = bot.send_document.await_args.kwargs
    assert sent["chat_id"] == 42
    assert sent["filename"].startswith("second-brain-backup")
    # Pinning is what makes the file findable again after a total wipe.
    assert bot.pin_chat_message.await_args.kwargs == {
        "chat_id": 42,
        "message_id": 555,
        "disable_notification": True,
    }
    assert fake_db.set_meta.call_args.args[0] == "last_backup_at"


async def test_a_failed_backup_is_not_recorded_so_it_retries(
    fake_db: MagicMock,
) -> None:
    from telegram.error import TelegramError

    bot = AsyncMock()
    bot.send_document.side_effect = TelegramError("network down")

    assert await backup.back_up(_settings(), bot) is False
    assert fake_db.set_meta.call_count == 0


async def test_restore_pulls_the_pinned_backup_into_an_empty_database(
    fake_db: MagicMock, tmp_path: Any
) -> None:
    target = tmp_path / "second_brain.db"
    bot = AsyncMock()
    pinned = MagicMock()
    pinned.document.file_name = "second-brain-backup-20260917T120000Z.db"
    pinned.document.file_id = "file-123"
    bot.get_chat.return_value = MagicMock(pinned_message=pinned)

    restored = await backup.restore_if_empty(
        _settings(sqlite_path=str(target)), bot
    )

    assert restored is True
    assert bot.get_file.await_args.args == ("file-123",)
    fake_db.setup_schema.assert_called_once()


async def test_restore_never_touches_a_database_that_has_data(
    fake_db: MagicMock,
) -> None:
    fake_db.is_empty.return_value = False
    bot = AsyncMock()

    assert await backup.restore_if_empty(_settings(), bot) is False
    assert bot.get_chat.await_count == 0


async def test_restore_ignores_a_pinned_message_that_is_not_a_backup(
    fake_db: MagicMock,
) -> None:
    bot = AsyncMock()
    pinned = MagicMock()
    pinned.document.file_name = "holiday-photos.zip"
    bot.get_chat.return_value = MagicMock(pinned_message=pinned)

    assert await backup.restore_if_empty(_settings(), bot) is False
    assert bot.get_file.await_count == 0
