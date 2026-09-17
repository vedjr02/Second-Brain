"""Tests for the Phase 0 FastAPI endpoints (no network, no Postgres)."""

from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.settings import Settings

WEBHOOK_SECRET = "test-secret"


class StubBot:
    def __init__(self) -> None:
        self.webhook_calls: list[dict[str, Any]] = []

    async def set_webhook(self, **kwargs: Any) -> None:
        self.webhook_calls.append(kwargs)


class StubApplication:
    """Stands in for the PTB Application so no Telegram calls happen."""

    def __init__(self, bot: StubBot) -> None:
        self.bot = bot
        self.processed: list[Any] = []

    async def initialize(self) -> None:
        return None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def process_update(self, update: Any) -> None:
        self.processed.append(update)


class Stubs:
    def __init__(self, client: TestClient, ptb: StubApplication, bot: StubBot) -> None:
        self.client = client
        self.ptb = ptb
        self.bot = bot


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> Iterator[Stubs]:
    """Patch settings, DB setup, and PTB construction; run the lifespan."""
    bot = StubBot()
    stub_app = StubApplication(bot)
    monkeypatch.setattr(
        main_module,
        "load_settings",
        lambda: Settings(
            telegram_bot_token="stub-token",
            telegram_webhook_secret=WEBHOOK_SECRET,
            webhook_base_url="https://example.onrender.com",
            llm_api_key="stub-key",
        ),
    )
    monkeypatch.setattr(main_module, "setup_schema", lambda: None)
    monkeypatch.setattr(main_module, "build_application", lambda _s: stub_app)

    with TestClient(app) as client:  # entering the context manager runs lifespan
        yield Stubs(client, stub_app, bot)


def test_healthz(stubs: Stubs) -> None:
    resp = stubs.client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_webhook_registers_at_boot_when_base_url_set(stubs: Stubs) -> None:
    assert len(stubs.bot.webhook_calls) == 1
    call = stubs.bot.webhook_calls[0]
    assert call["url"] == "https://example.onrender.com/telegram/webhook"
    assert call["secret_token"] == WEBHOOK_SECRET
    assert call["allowed_updates"] == ["message"]


def test_webhook_accepts_valid_secret(stubs: Stubs) -> None:
    update: dict[str, Any] = {
        "update_id": 1,
        "message": {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 42, "type": "private"},
            "text": "hello",
        },
    }
    resp = stubs.client.post(
        "/telegram/webhook",
        json=update,
        headers={"X-Telegram-Bot-Api-Secret-Token": WEBHOOK_SECRET},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert len(stubs.ptb.processed) == 1
    assert stubs.ptb.processed[0].message.text == "hello"


def test_webhook_rejects_wrong_secret(stubs: Stubs) -> None:
    resp = stubs.client.post(
        "/telegram/webhook",
        json={"update_id": 2},
        headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
    )
    assert resp.status_code == 403
    assert stubs.ptb.processed == []
