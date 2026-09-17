"""Typed application settings loaded from the environment."""

import os
from dataclasses import dataclass
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv


def _validated_timezone(name: str) -> str:
    """Reject unknown timezones at boot rather than at first reminder."""
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, KeyError) as exc:
        raise RuntimeError(
            f"USER_DISPLAY_TIMEZONE={name!r} is not a valid IANA timezone "
            "(examples: Europe/Berlin, America/New_York)"
        ) from exc
    return name

load_dotenv()

# all-MiniLM-L6-v2 via fastembed: 384 dims, matches the embedding BLOB format.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMS = 384


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    telegram_webhook_secret: str
    # Local SQLite database file (created + migrated on boot).
    sqlite_path: str = "second_brain.db"
    webhook_base_url: str = ""
    # Unset until load_settings() fills it; boot fails loudly without the env var.
    llm_api_key: str = ""
    # Empty = auto-pick the cheapest capable model from the account's model list.
    llm_model: str = ""
    # Optional vision-capable model for image descriptions (e.g. a Kimi VL
    # model). Empty disables the vision call; photos then rely on OCR only.
    llm_vision_model: str = ""
    llm_base_url: str = "https://api.moonshot.ai/v1"
    similarity_threshold: float = 0.35
    search_top_k: int = 5
    # Used to resolve relative reminder times and to echo "what was understood"
    # back in the user's local time (Phase 2).
    user_display_timezone: str = "UTC"
    # Your own chat with the bot. Enables two things: restricting the bot to
    # you, and backing the database up to that chat (see backup.py). Zero
    # means "not configured" — both features stay off.
    owner_chat_id: int = 0
    # How often the database is snapshotted to Telegram.
    backup_interval_hours: int = 6
    # Shared secret the GitHub Actions cron must present as a Bearer token when
    # calling POST /check-reminders. Empty disables the endpoint (404).
    reminder_check_secret: str = ""


def load_settings() -> Settings:
    """Read required env vars, failing loudly if any is missing or empty."""
    missing = [
        name
        for name in (
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_WEBHOOK_SECRET",
            "LLM_API_KEY",
        )
        if not os.getenv(name)
    ]
    if missing:
        hint = (
            " (you still have a GEMINI_API_KEY line — replace it with "
            "LLM_API_KEY=<your Kimi key from platform.moonshot.ai>)"
            if os.getenv("GEMINI_API_KEY")
            else ""
        )
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}{hint}"
        )
    return Settings(
        telegram_bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
        telegram_webhook_secret=os.environ["TELEGRAM_WEBHOOK_SECRET"],
        sqlite_path=os.getenv("SQLITE_PATH", "second_brain.db"),
        webhook_base_url=os.getenv("WEBHOOK_BASE_URL", "").rstrip("/"),
        llm_api_key=os.environ["LLM_API_KEY"],
        llm_model=os.getenv("LLM_MODEL", ""),
        llm_vision_model=os.getenv("LLM_VISION_MODEL", ""),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.moonshot.ai/v1"),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.35")),
        search_top_k=int(os.getenv("SEARCH_TOP_K", "5")),
        user_display_timezone=_validated_timezone(
            os.getenv("USER_DISPLAY_TIMEZONE", "UTC")
        ),
        owner_chat_id=int(os.getenv("OWNER_CHAT_ID", "0") or 0),
        backup_interval_hours=int(os.getenv("BACKUP_INTERVAL_HOURS", "6")),
        reminder_check_secret=os.getenv("REMINDER_CHECK_SECRET", ""),
    )
