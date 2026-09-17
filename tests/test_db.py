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


# --- hybrid retrieval: meaning plus exact wording ---------------------------


def test_keywords_drop_question_noise_but_keep_numbers() -> None:
    from app.memory import _keywords

    assert _keywords("what was the electrician's number again?") == {
        "electrician",
        "number",
    }
    assert "555" in _keywords("call 555 at 9am")


def test_keyword_overlap_scores_the_fraction_of_asked_words_present() -> None:
    from app.memory import _keyword_overlap, _keywords

    terms = _keywords("what is the wifi password")
    assert _keyword_overlap(terms, "the wifi password is hunter2") == 1.0
    assert _keyword_overlap(terms, "the wifi is flaky in the kitchen") == 0.5
    assert _keyword_overlap(terms, "pasta recipe from a reel") == 0.0


def test_rare_literal_match_is_retrieved_even_when_the_vectors_disagree(
    temp_db: None,
) -> None:
    """The whole point of a second brain: exact tokens must be findable."""
    from app import memory

    db.setup_schema()

    message_id = memory.save_message(
        chat_id=1, telegram_message_id=1, direction="in", raw_type="text",
        file_id=None, raw_content_text="Dave the electrician: 07700 900123",
    )
    memory.insert_memory_chunk(
        source_message_id=message_id,
        chunk_text="Dave the electrician: 07700 900123",
        embedding=[0.0] * 384,  # a vector that matches nothing at all
        tags=[],
    )

    results = memory.search_memory(
        [0.0] * 384, chat_id=1, top_k=5, threshold=0.9,
        query_text="what was the electrician's number?",
    )

    assert results and "900123" in results[0][0]


def test_unrelated_memories_are_not_dragged_in_by_keywords(
    temp_db: None,
) -> None:
    from app import memory

    db.setup_schema()

    message_id = memory.save_message(
        chat_id=1, telegram_message_id=1, direction="in", raw_type="text",
        file_id=None, raw_content_text="pasta recipe: garlic, chilli, oil",
    )
    memory.insert_memory_chunk(
        source_message_id=message_id,
        chunk_text="pasta recipe: garlic, chilli, oil",
        embedding=[0.0] * 384,
        tags=[],
    )

    assert memory.search_memory(
        [0.0] * 384, chat_id=1, top_k=5, threshold=0.9,
        query_text="what was the electrician's number?",
    ) == []


# --- long text is split so its middle stays retrievable ---------------------


def test_short_text_is_not_split() -> None:
    from app.embeddings import split_for_embedding

    assert split_for_embedding("the spare key is under the mat") == [
        "the spare key is under the mat"
    ]
    assert split_for_embedding("   ") == []


def test_long_text_splits_into_overlapping_chunks() -> None:
    from app.embeddings import split_for_embedding

    text = " ".join(f"Fact number {i} about the thing." for i in range(120))
    chunks = split_for_embedding(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 760 for chunk in chunks)
    # Nothing is lost: a fact from the middle still appears somewhere.
    assert any("Fact number 60" in chunk for chunk in chunks)
    # Consecutive chunks overlap, so a fact on a boundary survives whole.
    assert chunks[0].split()[-3] in chunks[1]


# --- timezone chosen from chat beats the env var ----------------------------


def test_stored_timezone_overrides_the_env_var(temp_db: None) -> None:
    from app.settings import Settings, effective_timezone, set_stored_timezone

    db.setup_schema()
    settings = Settings(
        telegram_bot_token="t", telegram_webhook_secret="s", llm_api_key="k",
        user_display_timezone="UTC",
    )
    assert effective_timezone(settings) == "UTC"

    set_stored_timezone("Asia/Kolkata")
    assert effective_timezone(settings) == "Asia/Kolkata"


def test_an_invalid_timezone_is_refused_not_stored(temp_db: None) -> None:
    from app.settings import set_stored_timezone

    db.setup_schema()
    with pytest.raises(RuntimeError, match="not a valid IANA timezone"):
        set_stored_timezone("Mars/Olympus")
