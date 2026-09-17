"""SQLite access: connection management, schema setup, connection probe.

Replaces the original Neon/pgvector design with a local SQLite file (no
external database service to sign up for). Structured metadata and extracted
text only — never raw media files (see the storage strategy in the build
spec). Embeddings are stored as little-endian float32 BLOBs and compared with
cosine similarity in Python (fine at personal scale).

Concurrency: one connection per thread (check_same_thread=False) guarded by a
process-wide RLock, since pipeline work runs via asyncio.to_thread. WAL mode
keeps readers and the writer from blocking each other.
"""

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator
from contextlib import contextmanager

from .settings import load_settings

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_CONNECTION: sqlite3.Connection | None = None

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id INTEGER NOT NULL,
  telegram_message_id INTEGER NOT NULL,
  direction TEXT NOT NULL CHECK (direction IN ('in', 'out')),
  raw_type TEXT NOT NULL,
  file_id TEXT,
  raw_content_text TEXT,
  received_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  classified_type TEXT,
  processed_at TEXT,
  UNIQUE (chat_id, telegram_message_id)
);

CREATE TABLE IF NOT EXISTS memory_chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_message_id INTEGER NOT NULL REFERENCES messages(id),
  chunk_text TEXT NOT NULL,
  embedding BLOB,
  tags TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS reminders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_message_id INTEGER NOT NULL REFERENCES messages(id),
  reminder_text TEXT NOT NULL,
  due_at TEXT NOT NULL,
  fired INTEGER NOT NULL DEFAULT 0,
  fired_at TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_reminders_due
  ON reminders (fired, due_at);
"""


def get_connection() -> sqlite3.Connection:
    """The process-wide SQLite connection, created on first use."""
    global _CONNECTION
    with _LOCK:
        if _CONNECTION is None:
            db_path = Path(load_settings().sqlite_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            _CONNECTION = sqlite3.connect(
                db_path, check_same_thread=False, isolation_level=None
            )
            _CONNECTION.row_factory = sqlite3.Row
            _CONNECTION.execute("PRAGMA journal_mode=WAL")
            _CONNECTION.execute("PRAGMA foreign_keys=ON")
            _CONNECTION.execute("PRAGMA busy_timeout=5000")
        return _CONNECTION


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    """Serialized access to the shared connection with a write transaction."""
    with _LOCK:
        conn = get_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except Exception:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")


@contextmanager
def query() -> Iterator[sqlite3.Connection]:
    """Serialized read access to the shared connection (no transaction)."""
    with _LOCK:
        yield get_connection()


def close_connection() -> None:
    global _CONNECTION
    with _LOCK:
        if _CONNECTION is not None:
            _CONNECTION.close()
            _CONNECTION = None


def setup_schema() -> None:
    """Create the three spec tables (idempotent)."""
    with _LOCK:
        get_connection().executescript(SCHEMA_SQL)


def test_connection() -> dict[str, Any]:
    """Probe, proving the database file works end to end.

    1. Verifies SQLite is live and writable.
    2. Inserts a probe row into the real `messages` table, reads it back, and
       deletes it — all in one transaction, so nothing lingers on failure.
    """
    import time

    with tx() as conn:
        version_row = conn.execute("SELECT sqlite_version()").fetchone()
        if version_row is None:
            raise RuntimeError("sqlite_version() returned no result")

        unique_marker = f"probe-{time.time_ns()}"
        cursor = conn.execute(
            """
            INSERT INTO messages (chat_id, telegram_message_id, direction,
                                  raw_type, raw_content_text)
            VALUES (0, ?, 'in', 'system', ?)
            """,
            (-time.time_ns(), unique_marker),
        )
        probe_row_id = cursor.lastrowid
        if probe_row_id is None:
            raise RuntimeError("probe insert returned no id")

        read_back = conn.execute(
            "SELECT raw_content_text FROM messages WHERE id = ?",
            (probe_row_id,),
        ).fetchone()
        conn.execute("DELETE FROM messages WHERE id = ?", (probe_row_id,))

    return {
        "sqlite_version": str(version_row[0]),
        "probe_insert_read_delete_ok": (
            read_back is not None and read_back["raw_content_text"] == unique_marker
        ),
        "probe_id": int(probe_row_id),
    }
