"""Telegram Bot API wiring (webhook mode) and Phase 1-3 handler registration.

The Application runs with `updater=None`: Telegram pushes updates to our
FastAPI webhook endpoint, which calls `application.process_update` directly.
No polling loop, no persistent connection (webhook mode per the build spec).

Text goes to the classification pipeline (Phase 1/2), photos to the OCR+vision
ingestion pipeline (Phase 3); other media types get a coming-soon reply.
"""

import logging

from telegram import Document, Update, VideoNote, Voice
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from . import pipeline
from .settings import Settings

logger = logging.getLogger(__name__)

_START_REPLY = (
    "Hi! I'm your second brain.\n"
    "Send me notes, links, or reminders and I'll remember them.\n"
    "Ask me about anything you've saved and I'll answer from your memory."
)

_MEDIA_REPLY = (
    "I can't process that type of content yet (voice notes and video are "
    "coming in later phases). Send text or photos for now."
)


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is not None:
        await message.reply_text(_START_REPLY)


def _is_image_document(document: Document) -> bool:
    """Images sent as files (not compressed photos) still carry an image mime."""
    return bool(document.mime_type and document.mime_type.startswith("image/"))


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route each supported message type to its pipeline."""
    message = update.effective_message
    if message is None:
        return
    if message.text:
        await pipeline.handle_text_message(update, context)
    elif message.photo:
        await pipeline.handle_photo_message(update, context)
    elif message.document and _is_image_document(message.document):
        await pipeline.handle_document_image_message(update, context)
    elif message.voice or message.audio or message.video_note:
        await pipeline.handle_voice_message(update, context)
    elif message.video:
        # Telegram video uploads are Phase 5; yt-dlp links take that phase's
        # download path, this one reuses the Telegram fetch path.
        await pipeline.handle_telegram_video_message(update, context)
    else:
        logger.info(
            "unsupported media ignored: chat_id=%s message_id=%s type=%s",
            message.chat_id,
            message.message_id,
            type(message).__name__,
        )
        await message.reply_text(_MEDIA_REPLY)


def register_handlers(application: Application) -> None:
    """Register the update handlers shared by webhook and polling modes."""
    application.add_handler(CommandHandler("start", handle_start))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    application.add_handler(MessageHandler(~filters.TEXT, handle_message))


def build_application(settings: Settings, polling: bool = False) -> Application:
    """Build the PTB application; webhook mode has no polling updater."""
    builder = ApplicationBuilder().token(settings.telegram_bot_token)
    if not polling:
        builder = builder.updater(None)
    application = builder.build()
    application.bot_data["settings"] = settings
    register_handlers(application)
    return application
