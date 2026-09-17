"""FastAPI entrypoint: lifespan wiring and routes.

Routes:
- GET  /healthz            : liveness probe
- POST /telegram/webhook   : Telegram pushes updates here (secret header)
- POST /check-reminders    : GitHub Actions cron pings this every minute to
                             fire due reminders (Bearer secret); webhooks are
                             passive and cannot act on a timer by themselves
"""

import hmac
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from telegram import Bot, Update

from . import reminders
from .db import setup_schema
from .settings import Settings, load_settings
from .telegram import build_application

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    application = build_application(settings)
    await application.initialize()
    await application.start()
    app.state.settings = settings
    app.state.ptb = application

    setup_schema()
    logger.info("database schema ready (pgvector + tables)")

    if settings.webhook_base_url:
        url = f"{settings.webhook_base_url}/telegram/webhook"
        await application.bot.set_webhook(
            url=url,
            secret_token=settings.telegram_webhook_secret,
            allowed_updates=["message"],
        )
        logger.info("webhook registered: %s", url)
    else:
        logger.info(
            "WEBHOOK_BASE_URL not set - register the webhook manually "
            "for local testing (see README)"
        )
    yield
    await application.stop()
    await application.shutdown()


app = FastAPI(title="second-brain-telegram", lifespan=lifespan)


def get_bot() -> Bot:
    """The PTB Bot instance created during startup (used by the cron route)."""
    return app.state.ptb.bot


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request) -> dict[str, bool]:
    settings: Settings = app.state.settings
    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not hmac.compare_digest(
        secret.encode(), settings.telegram_webhook_secret.encode()
    ):
        logger.warning("webhook call rejected: bad secret token")
        raise HTTPException(status_code=403, detail="invalid secret token")

    data = await request.json()
    update = Update.de_json(data, app.state.ptb.bot)
    if update is None:
        raise HTTPException(status_code=400, detail="unparseable update")
    await app.state.ptb.process_update(update)
    return {"ok": True}


@app.post("/check-reminders")
async def check_reminders(request: Request) -> dict[str, int | bool]:
    """Fire due reminders. Called by a GitHub Actions cron every minute.

    Auth: Authorization: Bearer <REMINDER_CHECK_SECRET>. If the secret is not
    configured the endpoint is disabled entirely (404) rather than open.
    """
    settings: Settings = app.state.settings
    if not settings.reminder_check_secret:
        raise HTTPException(status_code=404, detail="not found")

    auth = request.headers.get("Authorization", "")
    expected = f"Bearer {settings.reminder_check_secret}"
    if not hmac.compare_digest(auth.encode(), expected.encode()):
        logger.warning("reminder check rejected: bad bearer token")
        raise HTTPException(status_code=403, detail="invalid bearer token")

    sent = await reminders.fire_due_reminders(settings, get_bot())
    return {"ok": True, "sent": sent}
