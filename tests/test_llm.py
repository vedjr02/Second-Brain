"""Tests for app.llm: OpenAI-compatible client, parsing, retries, vision gate.

The HTTP layer is faked at _post_chat (and httpx.get for model listing), so
no network calls happen here.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

import app.llm as llm_module
from app.llm import (
    LLMClient,
    LLMVisionUnavailableError,
    ReminderParseError,
    _loads_json,
    _parse_classification,
    _parse_reminder,
)
from app.settings import Settings


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = dict(
        telegram_bot_token="stub-token",
        telegram_webhook_secret="s",
        llm_api_key="stub-key",
        llm_model="kimi-k3",  # default: skip auto model pick
        llm_vision_model="",
    )
    values.update(overrides)
    return Settings(**values)


def _chat_response(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


def _status_error(status: int) -> httpx.HTTPStatusError:
    response = MagicMock()
    response.status_code = status
    return httpx.HTTPStatusError(
        f"status {status}", request=MagicMock(), response=response
    )


class FakeChat:
    """Replaces LLMClient._post_chat with a scripted response queue."""

    def __init__(self, items: list[Any]) -> None:
        self.items = list(items)
        self.payloads: list[dict[str, Any]] = []

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _client_with(
    items: list[Any], settings: Settings | None = None
) -> tuple[LLMClient, FakeChat]:
    client = LLMClient(settings or _settings())
    fake = FakeChat(items)
    client._post_chat = fake  # type: ignore[method-assign]  # test seam
    return client, fake


# --- JSON parsing ------------------------------------------------------------


def test_parse_classification_full() -> None:
    raw = (
        '{"message_type": "reminder", "summary": "User wants a nudge",'
        ' "topics": ["errands"], "entities": ["dentist"],'
        ' "due_at": "2026-09-20T09:00:00+00:00"}'
    )
    c = _parse_classification(raw)
    assert c.message_type == "reminder"
    assert c.summary == "User wants a nudge"
    assert c.topics == ["errands"]
    assert c.entities == ["dentist"]
    assert c.due_at_iso == "2026-09-20T09:00:00+00:00"


def test_parse_classification_empty_due_at_is_none() -> None:
    raw = (
        '{"message_type": "note", "summary": "s", "topics": [],'
        ' "entities": [], "due_at": ""}'
    )
    assert _parse_classification(raw).due_at_iso is None


def test_parse_classification_rejects_unknown_type() -> None:
    raw = (
        '{"message_type": "junk", "summary": "s", "topics": [],'
        ' "entities": [], "due_at": ""}'
    )
    with pytest.raises(ValueError, match="invalid message_type"):
        _parse_classification(raw)


def test_loads_json_tolerates_code_fences() -> None:
    data = _loads_json('```json\n{"what": "x", "due_at": ""}\n```', "t")
    assert data == {"what": "x", "due_at": ""}


def test_loads_json_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="invalid JSON"):
        _loads_json("not json at all", "t")


def test_loads_json_rejects_non_object() -> None:
    with pytest.raises(ValueError, match="non-object"):
        _loads_json("[1, 2, 3]", "t")


# --- reminder parsing ---------------------------------------------------------


def test_parse_reminder_full() -> None:
    spec = _parse_reminder(
        '{"what": "Call the dentist to book a cleaning",'
        ' "due_at": "2026-09-20T09:00:00+02:00"}'
    )
    assert spec.what == "Call the dentist to book a cleaning"
    assert spec.due_at_iso == "2026-09-20T09:00:00+02:00"


def test_parse_reminder_empty_due_at_is_parse_error() -> None:
    with pytest.raises(ReminderParseError, match="no plausible future time"):
        _parse_reminder('{"what": "Call the dentist", "due_at": ""}')


def test_parse_reminder_invalid_due_at_is_parse_error() -> None:
    with pytest.raises(ReminderParseError, match="invalid due_at"):
        _parse_reminder('{"what": "Call the dentist", "due_at": "next sunday"}')


def test_parse_reminder_empty_what_is_parse_error() -> None:
    with pytest.raises(ReminderParseError, match="empty 'what'"):
        _parse_reminder('{"what": "  ", "due_at": "2026-09-20T09:00:00+00:00"}')


# --- client behavior ----------------------------------------------------------


def test_classify_sends_model_messages_and_json_mode() -> None:
    client, fake = _client_with(
        [_chat_response(
            '{"message_type": "note", "summary": "s", "topics": ["x"],'
            ' "entities": [], "due_at": ""}'
        )]
    )
    c = client.classify("buy milk", "2026-09-17T12:00:00+00:00")
    assert c.message_type == "note"
    payload = fake.payloads[0]
    assert payload["model"] == "kimi-k3"
    assert payload["response_format"] == {"type": "json_object"}
    assert "Current time: 2026-09-17T12:00:00+00:00" in payload["messages"][0]["content"]
    assert "Message:\nbuy milk" in payload["messages"][0]["content"]
    assert 'enum' in payload["messages"][0]["content"]  # schema embedded


def test_answer_with_no_chunks_never_calls_the_api() -> None:
    client, fake = _client_with([])
    assert client.answer("what did I save?", []) == (
        "I don't have anything saved about that yet."
    )
    assert fake.payloads == []


def test_answer_uses_chunk_bullets_and_question() -> None:
    client, fake = _client_with([_chat_response("You saved: dentist Friday.")])
    result = client.answer("when is the dentist?", ["dentist Friday", "booked 9am"])
    assert result == "You saved: dentist Friday."
    content = fake.payloads[0]["messages"][0]["content"]
    assert "- dentist Friday" in content
    assert "- booked 9am" in content
    assert "Question: when is the dentist?" in content


def test_answer_empty_content_falls_back() -> None:
    client, _fake = _client_with([_chat_response("  ")])
    assert client.answer("q", ["chunk"]) == (
        "Sorry, I had trouble generating an answer. Please try again."
    )


def test_transient_503_is_retried_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_module.time, "sleep", lambda _s: None)
    client, fake = _client_with(
        [_status_error(503), _status_error(503), _chat_response("pong reply")]
    )
    assert client.answer("q", ["chunk"]) == "pong reply"
    assert len(fake.payloads) == 3


def test_permanent_401_is_not_retried() -> None:
    client, fake = _client_with([_status_error(401)])
    with pytest.raises(httpx.HTTPStatusError, match="status 401"):
        client.answer("q", ["chunk"])
    assert len(fake.payloads) == 1


def test_retries_exhausted_raises_last_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_module.time, "sleep", lambda _s: None)
    client, _fake = _client_with([_status_error(503) for _ in range(4)])
    with pytest.raises(httpx.HTTPStatusError, match="status 503"):
        client.answer("q", ["chunk"])


def test_parse_reminder_call_injects_now_and_timezone() -> None:
    client, fake = _client_with(
        [_chat_response(
            '{"what": "Call the dentist", "due_at": "2026-09-20T09:00:00+02:00"}'
        )]
    )
    spec = client.parse_reminder(
        "remind me about the dentist sunday 9am",
        "2026-09-17T12:00:00+00:00",
        "Europe/Berlin",
    )
    assert spec.what == "Call the dentist"
    content = fake.payloads[0]["messages"][0]["content"]
    assert "Current time: 2026-09-17T12:00:00+00:00" in content
    assert "User's local timezone: Europe/Berlin" in content


def test_parse_reminder_requires_clock_and_timezone() -> None:
    client, _fake = _client_with([])
    with pytest.raises(ValueError, match="requires now_iso"):
        client.parse_reminder("text", "", "UTC")


# --- auto model pick -----------------------------------------------------------


def _fake_models_response(ids: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"data": [{"id": model_id} for model_id in ids]},
    )


def test_auto_pick_prefers_exact_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, str] = {}

    def fake_get(url: str, headers: Any = None, timeout: Any = None) -> Any:
        seen["url"] = url
        return _fake_models_response(
            ["moonshot-v1-8k", "kimi-k2", "kimi-k3", "kimi-latest"]
        )

    monkeypatch.setattr(llm_module.httpx, "get", fake_get)
    client = LLMClient(_settings(llm_model=""))
    assert client._model == "kimi-k3"
    assert seen["url"].endswith("/models")


def test_auto_pick_falls_back_to_substring_then_first_kimi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_module.httpx,
        "get",
        lambda url, headers=None, timeout=None: _fake_models_response(
            ["moonshot-v1-8k", "kimi-k2-turbo-preview-2026"]
        ),
    )
    client = LLMClient(_settings(llm_model=""))
    assert client._model == "kimi-k2-turbo-preview-2026"


def test_auto_pick_raises_with_helpful_message_when_no_kimi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_module.httpx,
        "get",
        lambda url, headers=None, timeout=None: _fake_models_response(
            ["some-other-model"]
        ),
    )
    with pytest.raises(RuntimeError, match="set LLM_MODEL"):
        LLMClient(_settings(llm_model=""))


# --- vision gate ---------------------------------------------------------------


def test_describe_image_without_vision_model_raises_unavailable() -> None:
    client, fake = _client_with([])
    with pytest.raises(LLMVisionUnavailableError, match="LLM_VISION_MODEL"):
        client.describe_image(b"img", "image/jpeg")
    assert fake.payloads == []


def test_describe_image_sends_base64_data_url() -> None:
    client, fake = _client_with(
        [_chat_response("A screenshot of a recipe video.")],
        _settings(llm_vision_model="kimi-vl"),
    )
    result = client.describe_image(b"img-bytes", "image/jpeg")
    assert result == "A screenshot of a recipe video."
    payload = fake.payloads[0]
    assert payload["model"] == "kimi-vl"  # the vision model, not the main one
    content = payload["messages"][0]["content"]
    assert len(content) == 2  # image part + text part
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_describe_image_requires_bytes() -> None:
    client, _fake = _client_with([], _settings(llm_vision_model="kimi-vl"))
    with pytest.raises(ValueError, match="requires image bytes"):
        client.describe_image(b"", "image/jpeg")
