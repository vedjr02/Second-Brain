"""Tests for the Phase 2 reminder-firing worker (app.reminders).

Database access is faked (in-memory queue) or backed by a throwaway SQLite
file; the Telegram bot is an AsyncMock.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from telegram.error import TelegramError

import app.memory as memory_module
import app.reminders as reminders
from app.memory import DueReminder
from app.settings import Settings


def _settings() -> Settings:
    return Settings(
        telegram_bot_token="stub-token",
        telegram_webhook_secret="s",
        webhook_base_url="",
        llm_api_key="stub-key",
        user_display_timezone="UTC",
    )


def _due(id: int = 1, chat_id: int = 42) -> DueReminder:
    return DueReminder(
        id=id,
        chat_id=chat_id,
        reminder_text="Call the dentist to book a cleaning",
        due_at=datetime(2026, 9, 20, 9, 0, tzinfo=UTC),
        created_at=datetime(2026, 9, 17, 9, 0, tzinfo=UTC),
    )


class FakeMemory:
    """Replaces app.reminders.memory with a claimable in-memory queue."""

    def __init__(self, due: list[DueReminder]) -> None:
        self.due = due
        self.claimed: list[int] = []
        self.released: list[int] = []
        self.recorded: list[tuple[int, int, str]] = []
        self.claim_will_succeed = True

    def fetch_due_reminders(self, now: datetime) -> list[DueReminder]:
        return self.due

    def claim_reminder(self, reminder_id: int) -> bool:
        self.claimed.append(reminder_id)
        return self.claim_will_succeed

    def release_reminder(self, reminder_id: int) -> None:
        self.released.append(reminder_id)

    def record_reminder_sent(
        self, message_id: int, chat_id: int, fired_text: str
    ) -> None:
        self.recorded.append((message_id, chat_id, fired_text))


@pytest.fixture
def fake_memory(monkeypatch: pytest.MonkeyPatch) -> FakeMemory:
    mem = FakeMemory([])
    monkeypatch.setattr(reminders, "memory", mem)
    return mem


async def test_no_due_reminders_sends_nothing(
    fake_memory: FakeMemory,
) -> None:
    bot = AsyncMock()
    sent = await reminders.fire_due_reminders(_settings(), bot)
    assert sent == 0
    bot.send_message.assert_not_awaited()


async def test_due_reminder_is_claimed_then_sent_and_recorded(
    fake_memory: FakeMemory,
) -> None:
    fake_memory.due = [_due()]
    bot = AsyncMock()
    bot.send_message.return_value.message_id = 777

    sent = await reminders.fire_due_reminders(_settings(), bot)

    assert sent == 1
    assert fake_memory.claimed == [1]  # claimed BEFORE the send
    (_args, kwargs) = bot.send_message.await_args
    assert kwargs["chat_id"] == 42
    assert kwargs["text"].startswith("⏰ Reminder: Call the dentist")
    assert "due today" in kwargs["text"] or "due Sunday 20 September" in kwargs["text"]
    assert fake_memory.recorded == [(777, 42, kwargs["text"])]
    assert fake_memory.released == []


async def test_send_failure_releases_the_claim(fake_memory: FakeMemory) -> None:
    fake_memory.due = [_due()]
    bot = AsyncMock()
    bot.send_message.side_effect = TelegramError("telegram down")

    sent = await reminders.fire_due_reminders(_settings(), bot)

    assert sent == 0
    assert fake_memory.claimed == [1]
    assert fake_memory.released == [1]  # retried on the next run
    assert fake_memory.recorded == []


async def test_lost_claim_race_sends_nothing(fake_memory: FakeMemory) -> None:
    fake_memory.due = [_due()]
    fake_memory.claim_will_succeed = False  # another run claimed it first
    bot = AsyncMock()

    sent = await reminders.fire_due_reminders(_settings(), bot)

    assert sent == 0
    bot.send_message.assert_not_awaited()
    assert fake_memory.released == []


async def test_one_failure_does_not_block_other_reminders(
    fake_memory: FakeMemory,
) -> None:
    first, second = _due(id=1), _due(id=2, chat_id=43)
    fake_memory.due = [first, second]
    bot = AsyncMock()

    async def send_one_fails(**kwargs: Any) -> Any:
        if kwargs["chat_id"] == 42:
            raise TelegramError("blocked chat")
        return type("Sent", (), {"message_id": 888})()

    bot.send_message.side_effect = send_one_fails

    sent = await reminders.fire_due_reminders(_settings(), bot)

    assert sent == 1  # the second reminder still went out
    assert fake_memory.released == [1]
    assert [r[1] for r in fake_memory.recorded] == [43]  # chat 42 failed, 43 sent


def test_fetch_claim_release_roundtrip_sqlite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Real SQLite: due fetch joins chat_id; claim is atomic and one-shot."""
    import app.db as db

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "stub-token")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "stub-secret")
    monkeypatch.setenv("LLM_API_KEY", "stub-key")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "test.db"))
    db.close_connection()
    try:
        db.setup_schema()
        with db.tx() as conn:
            conn.execute(
                "INSERT INTO messages (chat_id, telegram_message_id, direction,"
                " raw_type) VALUES (42, 1, 'in', 'text')"
            )
            conn.execute(
                "INSERT INTO reminders (source_message_id, reminder_text, due_at)"
                " VALUES (1, 'Call the dentist', ?)",
                (
                    (datetime.now(UTC) - timedelta(minutes=5)).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                ),
            )

        now = datetime.now(UTC)
        due = memory_module.fetch_due_reminders(now)
        assert len(due) == 1
        assert due[0].chat_id == 42
        assert due[0].reminder_text == "Call the dentist"

        # Claim is one-shot: a second claim loses (overlapping cron runs).
        assert memory_module.claim_reminder(due[0].id) is True
        assert memory_module.claim_reminder(due[0].id) is False
        assert memory_module.fetch_due_reminders(now) == []

        # Release puts it back in the due list (failed send -> retry).
        memory_module.release_reminder(due[0].id)
        assert len(memory_module.fetch_due_reminders(now)) == 1

        # The outgoing ping is recorded as a message row.
        memory_module.record_reminder_sent(777, 42, "ping text")
        with db.query() as conn:
            row = conn.execute(
                "SELECT raw_content_text FROM messages WHERE direction = 'out'"
            ).fetchone()
        assert row is not None and row["raw_content_text"] == "ping text"
    finally:
        db.close_connection()
