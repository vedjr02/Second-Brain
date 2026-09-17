"""Local launcher: run the bot with long polling — no tunnel, no deploy.

Telegram has no public URL to call in polling mode, so there is no webhook;
instead the process fetches updates itself every second. Perfect for local
testing and requires nothing but this command:

    .venv/bin/python -m app.local

For production (Render + GitHub Actions cron) the webhook entrypoint
`python -m app` remains the way to run.
"""

import asyncio
import logging

from . import reminders
from .db import setup_schema
from .settings import load_settings
from .telegram import build_application

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


async def run() -> None:
    settings = load_settings()
    application = build_application(settings, polling=True)
    updater = application.updater
    assert updater is not None  # polling=True builds with an updater
    await application.initialize()
    await application.start()
    await updater.start_polling(drop_pending_updates=True)
    setup_schema()
    logger.info("database ready")

    # Local-only convenience: check due reminders every 30s (the production
    # deployment uses the GitHub Actions cron hitting /check-reminders instead).
    async def reminder_loop() -> None:
        while True:
            try:
                await reminders.fire_due_reminders(settings, application.bot)
            except Exception:
                logger.exception("local reminder check failed")
            await asyncio.sleep(30)

    reminder_task = asyncio.create_task(reminder_loop())

    me = await application.bot.get_me()
    logger.info("bot is LIVE via polling: @%s — send it a message! (Ctrl+C to stop)", me.username)

    stop = asyncio.Event()
    try:
        await stop.wait()  # run until Ctrl+C
    finally:
        reminder_task.cancel()
        await updater.stop()
        await application.stop()
        await application.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("stopped")
