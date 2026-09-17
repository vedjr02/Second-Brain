"""Tests for the POST /check-reminders cron endpoint (no network, no DB).

The lifespan is run with stubbed PTB construction (as in test_webhook), and
app.reminders.fire_due_reminders is replaced with a recorder.
"""

from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
import app.reminders as reminders_module
from app.main import app
from app.settings import Settings


class StubBot:
    async def set_webhook(self, **kwargs: Any) -> None:
        return None


class StubApplication:
    def __init__(self, bot: StubBot) -> None:
        self.bot = bot

    async def initialize(self) -> None:
        return None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None


def _settings(secret: str) -> Settings:
    return Settings(
        telegram_bot_token="stub-token",
        telegram_webhook_secret="hook-secret",
        webhook_base_url="",
        llm_api_key="stub-key",
        reminder_check_secret=secret,
    )


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, list[tuple[Settings, StubBot]]]]:
    stub_app = StubApplication(StubBot())
    monkeypatch.setattr(main_module, "load_settings", lambda: _settings("cron-secret"))
    monkeypatch.setattr(main_module, "setup_schema", lambda: None)
    monkeypatch.setattr(main_module, "build_application", lambda _s: stub_app)
    fired: list[tuple[Settings, StubBot]] = []

    async def fake_fire(settings: Settings, bot: StubBot, now: Any = None) -> int:
        fired.append((settings, bot))
        return 2

    monkeypatch.setattr(reminders_module, "fire_due_reminders", fake_fire)
    with TestClient(app) as test_client:
        yield test_client, fired


def test_requires_bearer_token(client: tuple[TestClient, list[tuple[Settings, StubBot]]]) -> None:
    test_client, fired = client
    resp = test_client.post("/check-reminders")
    assert resp.status_code == 403
    assert fired == []


def test_rejects_wrong_bearer_token(
    client: tuple[TestClient, list[tuple[Settings, StubBot]]],
) -> None:
    test_client, fired = client
    resp = test_client.post(
        "/check-reminders", headers={"Authorization": "Bearer nope"}
    )
    assert resp.status_code == 403
    assert fired == []


def test_fires_with_correct_bearer_token(
    client: tuple[TestClient, list[tuple[Settings, StubBot]]],
) -> None:
    test_client, fired = client
    resp = test_client.post(
        "/check-reminders", headers={"Authorization": "Bearer cron-secret"}
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "sent": 2}
    assert len(fired) == 1
    settings, bot = fired[0]
    assert settings.reminder_check_secret == "cron-secret"
    assert bot is main_module.app.state.ptb.bot


def test_endpoint_disabled_without_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "load_settings", lambda: _settings(""))
    monkeypatch.setattr(main_module, "setup_schema", lambda: None)
    monkeypatch.setattr(
        main_module, "build_application", lambda _s: StubApplication(StubBot())
    )
    with TestClient(app) as test_client:
        resp = test_client.post(
            "/check-reminders", headers={"Authorization": "Bearer anything"}
        )
    assert resp.status_code == 404
