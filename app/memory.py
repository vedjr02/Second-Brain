"""Memory storage and retrieval: messages, memory_chunks, reminders, search.

SQLite port of the original pgvector design: embeddings live in the rows as
little-endian float32 BLOBs and cosine similarity is computed in Python —
simple and dependency-free, plenty fast for a personal memory (thousands of
chunks search in well under a millisecond per query).

All DB access is serialized through the shared connection guarded by the
module-level lock in db.py (pipeline work runs via asyncio.to_thread).
"""

import logging
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime, timezone

from . import db

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DueReminder:
    """A reminder row joined with its source message, ready to fire."""

    id: int
    chat_id: int
    reminder_text: str
    due_at: datetime
    created_at: datetime


def save_message(
    chat_id: int,
    telegram_message_id: int,
    direction: str,
    raw_type: str,
    file_id: str | None,
    raw_content_text: str | None,
) -> int:
    """Insert the incoming message row and return its id.

    Uses ON CONFLICT DO UPDATE so re-delivered Telegram updates update the
    existing row in place and return its id.
    """
    with db.tx() as conn:
        conn.execute(
            """
            INSERT INTO messages (chat_id, telegram_message_id, direction,
                                  raw_type, file_id, raw_content_text)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (chat_id, telegram_message_id) DO UPDATE
              SET raw_content_text = excluded.raw_content_text
            """,
            (chat_id, telegram_message_id, direction, raw_type, file_id,
             raw_content_text),
        )
        row = conn.execute(
            "SELECT id FROM messages WHERE chat_id = ? AND telegram_message_id = ?",
            (chat_id, telegram_message_id),
        ).fetchone()
    assert row is not None
    return int(row["id"])


def mark_message_processed(message_id: int, classified_type: str) -> None:
    """Stamp classification result and processed_at on the message row."""
    with db.tx() as conn:
        conn.execute(
            """
            UPDATE messages
               SET classified_type = ?,
                   processed_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
             WHERE id = ?
            """,
            (classified_type, message_id),
        )


def update_message_content(message_id: int, raw_content_text: str) -> None:
    """Store the extracted media text (OCR + description) on the message row.

    Media rows are saved pointer-first (file_id, no text), then enriched once
    OCR/vision completes. Always executed after save_message, so the row exists.
    """
    with db.tx() as conn:
        conn.execute(
            """
            UPDATE messages
               SET raw_content_text = ?
             WHERE id = ?
            """,
            (raw_content_text, message_id),
        )


def get_message_saved_day(message_id: int) -> str | None:
    """Human-readable save date for a message ('Saturday 12 July').

    Embedded into chunk text so grounded answers can mention roughly when a
    memory was saved (runtime rule 2).
    """
    with db.query() as conn:
        row = conn.execute(
            "SELECT received_at FROM messages WHERE id = ?",
            (message_id,),
        ).fetchone()
    if row is None or row["received_at"] is None:
        return None
    received = _parse_iso(str(row["received_at"]))
    return received.strftime("%A %d %B").replace(" 0", " ")


def insert_memory_chunk(
    source_message_id: int, chunk_text: str, embedding: list[float], tags: list[str]
) -> None:
    """Store a text chunk with its 384-dim embedding (float32 BLOB)."""
    with db.tx() as conn:
        conn.execute(
            """
            INSERT INTO memory_chunks (source_message_id, chunk_text, embedding, tags)
            VALUES (?, ?, ?, ?)
            """,
            (source_message_id, chunk_text, _pack_vector(embedding),
             _dump_tags(tags)),
        )


def search_memory(
    query_embedding: list[float],
    chat_id: int,
    top_k: int,
    threshold: float,
    query_text: str = "",
) -> list[tuple[str, float]]:
    """Top-k chunks for this chat, ranked by meaning AND exact wording.

    Embeddings alone are weak at exactly what a second brain is for: a phone
    number, a name, a wifi password, a product nobody paraphrases. Those are
    rare tokens the model has no useful vector for, but they match literally.
    So each chunk gets two scores — cosine similarity (0..1) and the fraction
    of the question's distinctive words it literally contains — and a chunk is
    retrieved when EITHER is convincing.

    Returns (chunk_text, score), highest first, filtered to this chat.
    """
    q = _pack_vector(query_embedding)
    terms = _keywords(query_text)
    with db.query() as conn:
        rows = conn.execute(
            """
            SELECT mc.chunk_text, mc.embedding
              FROM memory_chunks mc
              JOIN messages m ON m.id = mc.source_message_id
             WHERE m.chat_id = ?
               AND mc.embedding IS NOT NULL
            """,
            (chat_id,),
        ).fetchall()
    scored: list[tuple[str, float]] = []
    for row in rows:
        text = str(row["chunk_text"])
        similarity = _cosine(q, row["embedding"])
        overlap = _keyword_overlap(terms, text)
        # Keyword hits lift a chunk's score but never outrank a genuinely
        # closer meaning match; a strong literal match (most of the asked
        # words present) is retrieved even when the vectors disagree.
        score = min(1.0, similarity + _KEYWORD_WEIGHT * overlap)
        if similarity >= threshold or overlap >= _KEYWORD_RESCUE:
            scored.append((text, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


# How much a full literal word overlap can lift a chunk's score, and how much
# overlap on its own is enough to retrieve a chunk the vectors would have
# missed. Half is deliberately generous: asking "what was the electrician's
# number?" about a note reading "Dave the electrician: 07700 900123" matches
# only one of the two asked words, and that has to be enough. Over-retrieval
# is cheap here — the answer is grounded, so an irrelevant excerpt just gets
# ignored, while a missed one makes the bot claim it never knew.
_KEYWORD_WEIGHT = 0.3
_KEYWORD_RESCUE = 0.5

# Words carrying no retrieval signal — questions are mostly made of these.
_STOPWORDS = frozenset(
    """a an and are as at be but by can did do does for from had has have how i
    if in is it its me my of on or so than that the their them then there these
    they this to was we were what when where which who whom why will with you
    your about again all am any because been before being could would should
    just like get got tell show find remember saved save please""".split()
)


def _keywords(text: str) -> set[str]:
    """Distinctive words of a question: lowercased, stopwords dropped.

    Numbers are always kept however short — '9am', '555', a house number or a
    year is exactly the kind of token that makes a memory findable.
    """
    words = "".join(c.lower() if c.isalnum() else " " for c in text).split()
    return {
        word
        for word in words
        if word not in _STOPWORDS and (len(word) > 2 or word.isdigit())
    }


def _keyword_overlap(terms: set[str], chunk_text: str) -> float:
    """Fraction of the question's distinctive words present in the chunk."""
    if not terms:
        return 0.0
    chunk_words = _keywords(chunk_text)
    hits = sum(
        1
        for term in terms
        if term in chunk_words or any(term in word for word in chunk_words)
    )
    return hits / len(terms)


def save_reminder(
    source_message_id: int, reminder_text: str, due_at: datetime
) -> None:
    """Insert a reminder row for the scheduler to fire."""
    with db.tx() as conn:
        conn.execute(
            """
            INSERT INTO reminders (source_message_id, reminder_text, due_at)
            VALUES (?, ?, ?)
            """,
            (source_message_id, reminder_text, _format_iso(due_at)),
        )


def fetch_due_reminders(now: datetime) -> list[DueReminder]:
    """Unfired reminders with due_at <= now, joined with the owner's chat_id."""
    with db.query() as conn:
        rows = conn.execute(
            """
            SELECT r.id, m.chat_id, r.reminder_text, r.due_at, r.created_at
              FROM reminders r
              JOIN messages m ON m.id = r.source_message_id
             WHERE r.fired = 0
               AND r.due_at <= ?
             ORDER BY r.due_at
            """,
            (_format_iso(now),),
        ).fetchall()
    return [
        DueReminder(
            id=int(row["id"]),
            chat_id=int(row["chat_id"]),
            reminder_text=str(row["reminder_text"]),
            due_at=_parse_iso(str(row["due_at"])),
            created_at=_parse_iso(str(row["created_at"])),
        )
        for row in rows
    ]


def claim_reminder(reminder_id: int) -> bool:
    """Atomically claim a reminder for firing; True iff this caller won it.

    The claim (fired=1) is taken BEFORE the Telegram send so overlapping
    cron runs can never both send the same reminder. If the send then fails,
    release_reminder() gives the claim back for the next run to retry.
    """
    with db.tx() as conn:
        cursor = conn.execute(
            """
            UPDATE reminders
               SET fired = 1, fired_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
             WHERE id = ? AND fired = 0
            """,
            (reminder_id,),
        )
    return cursor.rowcount == 1


def release_reminder(reminder_id: int) -> None:
    """Give a claim back (fired=0) so the next check retries the send."""
    with db.tx() as conn:
        conn.execute(
            """
            UPDATE reminders
               SET fired = 0, fired_at = NULL
             WHERE id = ?
            """,
            (reminder_id,),
        )


def record_reminder_sent(
    message_id: int, chat_id: int, fired_text: str
) -> None:
    """Store the outgoing notification as a message row (direction 'out').

    Bookkeeping only: the ping has already been delivered when this runs.
    The messages table is the audit log of what was pushed to the user.
    """
    with db.tx() as conn:
        conn.execute(
            """
            INSERT INTO messages (chat_id, telegram_message_id, direction,
                                  raw_type, raw_content_text)
            VALUES (?, ?, 'out', 'reminder', ?)
            """,
            (chat_id, message_id, fired_text),
        )


def _pack_vector(embedding: list[float]) -> bytes:
    """384 floats -> little-endian float32 BLOB."""
    return struct.pack(f"<{len(embedding)}f", *embedding)


def _unpack_vector(blob: bytes) -> list[float]:
    """float32 BLOB -> floats."""
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def _cosine(a_packed: bytes, b_packed: bytes) -> float:
    """Cosine similarity between two packed float32 vectors."""
    a = _unpack_vector(a_packed)
    b = _unpack_vector(b_packed)
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _dump_tags(tags: list[str]) -> str:
    import json

    return json.dumps(tags)


def _format_iso(dt: datetime) -> str:
    """UTC ISO 8601 with Z suffix — the canonical stored timestamp format."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(value: str) -> datetime:
    """Parse a stored timestamp; Z suffix becomes UTC."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass(frozen=True)
class SavedChunk:
    """A stored memory chunk with the day it was saved."""

    id: int
    chunk_text: str
    created_at: datetime


def recent_chunks(chat_id: int, limit: int) -> list[SavedChunk]:
    """The most recently saved memories for this chat, newest first."""
    with db.query() as conn:
        rows = conn.execute(
            """
            SELECT mc.id, mc.chunk_text, mc.created_at
              FROM memory_chunks mc
              JOIN messages m ON m.id = mc.source_message_id
             WHERE m.chat_id = ?
             ORDER BY mc.id DESC
             LIMIT ?
            """,
            (chat_id, limit),
        ).fetchall()
    return [
        SavedChunk(
            id=int(row["id"]),
            chunk_text=str(row["chunk_text"]),
            created_at=_parse_iso(str(row["created_at"])),
        )
        for row in rows
    ]


def delete_chunk(chat_id: int, chunk_id: int) -> str | None:
    """Delete one memory chunk owned by this chat; returns its text.

    Scoped by chat_id so a chunk id from someone else's chat can never be
    deleted. Returns None when there was nothing to delete.
    """
    with db.tx() as conn:
        row = conn.execute(
            """
            SELECT mc.id, mc.chunk_text
              FROM memory_chunks mc
              JOIN messages m ON m.id = mc.source_message_id
             WHERE mc.id = ? AND m.chat_id = ?
            """,
            (chunk_id, chat_id),
        ).fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM memory_chunks WHERE id = ?", (chunk_id,))
    return str(row["chunk_text"])


def count_chunks(chat_id: int) -> int:
    """How many memories this chat has saved (for /status)."""
    with db.query() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n
              FROM memory_chunks mc
              JOIN messages m ON m.id = mc.source_message_id
             WHERE m.chat_id = ?
            """,
            (chat_id,),
        ).fetchone()
    return 0 if row is None else int(row["n"])


def count_pending_reminders(chat_id: int) -> int:
    """Unfired reminders still waiting for this chat."""
    with db.query() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n
              FROM reminders r
              JOIN messages m ON m.id = r.source_message_id
             WHERE m.chat_id = ? AND r.fired = 0
            """,
            (chat_id,),
        ).fetchone()
    return 0 if row is None else int(row["n"])
