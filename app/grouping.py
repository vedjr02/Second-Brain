"""Group rapid-fire messages into one thought before processing them.

People do not write one message per idea. They send "remember this", then the
photo, then "I want to post it tomorrow" — three updates that only mean
something together. Handling each one alone gets all three wrong: the first is
noise, the second has no context, the third refers to a "this" the classifier
cannot see.

So text is debounced per chat: each new message restarts a short timer, and
only when the user actually stops typing is everything they sent joined into a
single message and run through the normal pipeline. The last message of the
burst is the one replied to, so the answer lands under what they just wrote.
"""

import asyncio
import logging
from dataclasses import dataclass, field

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from . import pipeline
from .settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class _Burst:
    """Text collected from one chat while the user is still typing."""

    parts: list[str] = field(default_factory=list)
    timer: asyncio.Task[None] | None = None
    update: Update | None = None
    context: ContextTypes.DEFAULT_TYPE | None = None


_BURSTS: dict[int, _Burst] = {}


def combine(parts: list[str]) -> str:
    """Join a burst into one message.

    Sentence-ending punctuation is left alone; a fragment gets a full stop so
    the classifier reads "post this tomorrow" as its own clause rather than
    running it into the previous line.
    """
    cleaned = [part.strip() for part in parts if part.strip()]
    joined: list[str] = []
    for index, part in enumerate(cleaned):
        last = index == len(cleaned) - 1
        if not last and part[-1] not in ".!?,;:":
            part = f"{part}."
        joined.append(part)
    return " ".join(joined)


async def submit_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add a message to the chat's burst and restart its timer."""
    message = update.effective_message
    if message is None or not message.text:
        return
    settings: Settings = context.application.bot_data["settings"]
    window = settings.group_window_seconds
    if window <= 0:
        await pipeline.handle_text_message(update, context)
        return

    chat_id = message.chat_id
    burst = _BURSTS.setdefault(chat_id, _Burst())
    burst.parts.append(message.text.strip())
    burst.update = update
    burst.context = context
    if burst.timer is not None:
        burst.timer.cancel()
    burst.timer = asyncio.create_task(_flush_after(chat_id, window))


async def _flush_after(chat_id: int, window: float) -> None:
    try:
        await asyncio.sleep(window)
    except asyncio.CancelledError:
        return  # another message arrived; a new timer owns this burst
    await flush(chat_id)


async def flush(chat_id: int) -> None:
    """Process everything buffered for this chat as one message."""
    burst = _BURSTS.pop(chat_id, None)
    if burst is None or burst.update is None or burst.context is None:
        return
    text = combine(burst.parts)
    if not text:
        return
    if len(burst.parts) > 1:
        logger.info(
            "grouped %d messages from chat %s into one", len(burst.parts), chat_id
        )
    message = burst.update.effective_message
    if message is not None:
        try:
            await message.chat.send_action(ChatAction.TYPING)
        except Exception:  # a missing typing indicator must never block work
            logger.debug("could not send typing action", exc_info=True)
    await pipeline.handle_text_message(
        burst.update, burst.context, text_override=text
    )


async def flush_all() -> None:
    """Flush every pending burst (used at shutdown so nothing is lost)."""
    for chat_id in list(_BURSTS):
        burst = _BURSTS.get(chat_id)
        if burst is not None and burst.timer is not None:
            burst.timer.cancel()
        await flush(chat_id)
