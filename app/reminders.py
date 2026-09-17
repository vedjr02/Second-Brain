"""Reminder firing worker (Phase 2).

Webhooks are passive: nothing fires on a timer by itself. A free GitHub
Actions scheduled workflow hits POST /check-reminders every minute (main.py),
which calls fire_due_reminders() here.

Safety properties:
- A reminder is claimed (fired=1) atomically in SQLite BEFORE sending,
  so two overlapping cron runs can never both send the same reminder.
- If the Telegram send fails, the claim is released so the next run retries.
- The outgoing ping is stored as a message row (direction 'out'), keeping a
  record of what was actually pushed to the user.
"""

import asyncio
import logging
from datetime import datetime, timezone

from telegram import Bot
from telegram.error import TelegramError

from . import memory, pipeline
from .settings import Settings

logger = logging.getLogger(__name__)


async def fire_due_reminders(
    settings: Settings, bot: Bot, now: datetime | None = None
) -> int:
    """Send every due, not-yet-fired reminder; returns how many were sent."""
    if now is None:
        now = datetime.now(timezone.utc)
    due = await asyncio.to_thread(memory.fetch_due_reminders, now)
    sent_count = 0
    for reminder in due:
        if not await _fire_one(settings, bot, reminder):
            continue
        sent_count += 1
    if due:
        logger.info("reminder check: %d due, %d sent", len(due), sent_count)
    return sent_count


async def _fire_one(settings: Settings, bot: Bot, reminder: memory.DueReminder) -> bool:
    """Claim, send, and record a single reminder; release the claim on failure."""
    claimed = await asyncio.to_thread(memory.claim_reminder, reminder.id)
    if not claimed:
        logger.info("reminder %s already claimed by another run", reminder.id)
        return False

    text = pipeline.format_reminder_message(
        what=reminder.reminder_text,
        due_at=reminder.due_at,
        created_at=reminder.created_at,
        tz_name=settings.user_display_timezone,
    )
    try:
        sent = await bot.send_message(chat_id=reminder.chat_id, text=text)
    except TelegramError:
        logger.exception("telegram send failed for reminder %s", reminder.id)
        await asyncio.to_thread(memory.release_reminder, reminder.id)
        return False

    try:
        await asyncio.to_thread(
            memory.record_reminder_sent,
            sent.message_id,
            reminder.chat_id,
            text,
        )
    except Exception:
        # The ping is already out; recording is bookkeeping, not delivery.
        logger.exception(
            "could not record outgoing reminder message (reminder %s)",
            reminder.id,
        )
    return True
