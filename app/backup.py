"""Free, durable persistence: the database backs itself up to Telegram.

The build spec already treats Telegram as the blob store for media. The same
trick solves the one thing a free host cannot give us — a disk that survives a
redeploy. Every few hours the bot sends a consistent snapshot of the SQLite
file to the owner's own chat as a document and pins it. Telegram keeps it
forever, for free.

Recovery is the interesting half: after a wipe there is no database left to
remember where the backup went. So the backup message is *pinned*, and
`getChat` returns the pinned message (with its document file_id) to anyone who
asks — which is how a completely blank instance finds its own last snapshot at
boot and restores it before serving a single message.

Requires OWNER_CHAT_ID (your own chat with the bot). Without it the whole
feature is off, and the bot says so at boot rather than pretending.
"""

import logging
import os
import tempfile
from datetime import UTC, datetime, timedelta

from telegram import Bot
from telegram.error import TelegramError

from . import db
from .settings import Settings

logger = logging.getLogger(__name__)

_LAST_BACKUP_KEY = "last_backup_at"
_BACKUP_PREFIX = "second-brain-backup"
_CAPTION = (
    "🧠 Second brain backup — {when} UTC\n"
    "Keep this pinned: it is how the bot restores itself if its disk is wiped."
)


def _backup_filename(now: datetime) -> str:
    return f"{_BACKUP_PREFIX}-{now.strftime('%Y%m%dT%H%M%SZ')}.db"


def is_backup_document(file_name: str | None) -> bool:
    """Whether a pinned document looks like one of our own snapshots."""
    return bool(file_name and file_name.startswith(_BACKUP_PREFIX))


def due_for_backup(settings: Settings, now: datetime | None = None) -> bool:
    """True when enough time has passed since the last successful backup."""
    if not settings.owner_chat_id:
        return False
    now = now or datetime.now(UTC)
    last = db.get_meta(_LAST_BACKUP_KEY)
    if last is None:
        return True
    try:
        previous = datetime.fromisoformat(last)
    except ValueError:
        return True
    return now - previous >= timedelta(hours=settings.backup_interval_hours)


async def back_up(settings: Settings, bot: Bot) -> bool:
    """Send a snapshot to the owner's chat and pin it. True when it landed.

    Never raises: a failed backup must not take down the message that
    triggered it. The timestamp is only recorded on success, so a failure is
    retried at the next opportunity rather than skipped for hours.
    """
    if not settings.owner_chat_id:
        return False
    now = datetime.now(UTC)
    with tempfile.TemporaryDirectory(prefix="sb-backup-") as tmp:
        path = os.path.join(tmp, _backup_filename(now))
        try:
            db.snapshot_to_file(path)
            with open(path, "rb") as handle:
                sent = await bot.send_document(
                    chat_id=settings.owner_chat_id,
                    document=handle,
                    filename=os.path.basename(path),
                    caption=_CAPTION.format(when=now.strftime("%Y-%m-%d %H:%M")),
                    disable_notification=True,
                )
            await bot.pin_chat_message(
                chat_id=settings.owner_chat_id,
                message_id=sent.message_id,
                disable_notification=True,
            )
        except (TelegramError, OSError):
            logger.exception("database backup to Telegram failed")
            return False
    db.set_meta(_LAST_BACKUP_KEY, now.isoformat())
    logger.info("database backed up to Telegram (%s)", os.path.basename(path))
    return True


async def maybe_back_up(settings: Settings, bot: Bot) -> bool:
    """Back up only if one is due — safe to call on every cron tick."""
    if not due_for_backup(settings):
        return False
    return await back_up(settings, bot)


async def restore_if_empty(settings: Settings, bot: Bot) -> bool:
    """Pull the pinned snapshot back down when this instance has no data.

    Only ever runs against an empty database, so it can never clobber live
    memories — the worst case is that it finds nothing and we start fresh.
    """
    if not settings.owner_chat_id:
        logger.info(
            "OWNER_CHAT_ID not set - automatic backup/restore is disabled "
            "(send /start to the bot to learn your chat id)"
        )
        return False
    if not db.is_empty():
        return False
    try:
        chat = await bot.get_chat(settings.owner_chat_id)
        pinned = chat.pinned_message
        document = pinned.document if pinned is not None else None
        if document is None or not is_backup_document(document.file_name):
            logger.info("no pinned backup found - starting with an empty brain")
            return False
        tg_file = await bot.get_file(document.file_id)
        target = settings.sqlite_path
        db.close_connection()
        await tg_file.download_to_drive(custom_path=target)
    except (TelegramError, OSError):
        logger.exception("restoring the database from Telegram failed")
        return False
    db.setup_schema()
    logger.info("database restored from the pinned Telegram backup")
    return True
