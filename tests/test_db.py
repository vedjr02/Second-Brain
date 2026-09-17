"""Unit tests for the SQLite layer (app.db) — real temp-file database.

Real connectivity semantics are exercised here with a throwaway database file,
so transactions, probing, and schema creation are all pinned without any
external service.
"""

import sqlite3
from pathlib import Path
from typing import Iterator

import pytest

import app.db as db
from app.settings import load_settings


@pytest.fixture
def temp_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[None]:
    """Point settings at a throwaway database file and reset the singleton."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "stub-token")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "stub-secret")
    monkeypatch.setenv("LLM_API_KEY", "stub-key")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "test.db"))
    db.close_connection()
    yield
    db.close_connection()


def test_schema_and_probe_roundtrip(temp_db: None) -> None:
    db.setup_schema()
    result = db.test_connection()
    assert result["probe_insert_read_delete_ok"] is True
    assert result["probe_id"] >= 1
    assert result["sqlite_version"]


def test_schema_is_idempotent(temp_db: None) -> None:
    db.setup_schema()
    db.setup_schema()  # must not raise


def test_query_sees_committed_writes(temp_db: None) -> None:
    db.setup_schema()
    with db.tx() as conn:
        conn.execute(
            "INSERT INTO messages (chat_id, telegram_message_id, direction, raw_type,"
            " raw_content_text) VALUES (42, 1, 'in', 'text', 'hello')"
        )
    with db.query() as conn:
        row = conn.execute(
            "SELECT raw_content_text FROM messages WHERE chat_id = 42"
        ).fetchone()
    assert row is not None and row["raw_content_text"] == "hello"


def test_tx_rolls_back_on_exception(temp_db: None) -> None:
    db.setup_schema()
    with pytest.raises(RuntimeError, match="boom"):
        with db.tx() as conn:
            conn.execute(
                "INSERT INTO messages (chat_id, telegram_message_id, direction,"
                " raw_type) VALUES (1, 1, 'in', 'text')"
            )
            raise RuntimeError("boom")
    with db.query() as conn:
        count = conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()
    assert count is not None and count["n"] == 0


def test_foreign_keys_are_enforced(temp_db: None) -> None:
    db.setup_schema()
    with pytest.raises(sqlite3.IntegrityError):
        with db.tx() as conn:
            conn.execute(
                "INSERT INTO memory_chunks (source_message_id, chunk_text)"
                " VALUES (999, 'orphan')"
            )


def test_missing_env_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_WEBHOOK_SECRET", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        load_settings()
