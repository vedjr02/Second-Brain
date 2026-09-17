"""LLM client (OpenAI-compatible chat API — Kimi/Moonshot by default).

Replaces the original Gemini client. One module-level client; every call is
a plain POST to {base_url}/chat/completions. Classification and reminder
parsing run in JSON mode with the schema embedded in the prompt; answers are
grounded ONLY in retrieved memory chunks. Transient errors (429/5xx) are
retried with backoff so demand spikes never surface as failed messages.

Vision is optional: Kimi's text models cannot see images, so describe_image
only works when LLM_VISION_MODEL is configured — otherwise photos rely on
local OCR and the bot honestly says when it couldn't read anything.
"""

import base64
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from .settings import Settings

logger = logging.getLogger(__name__)

# Transient server-side failures worth retrying (demand spikes, rate limits).
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# Kept low on purpose: a person waiting in a chat would rather be told the
# call failed than watch nothing happen. Four 60s attempts with backoff meant
# ~4 minutes of silence for one flaky request.
_MAX_ATTEMPTS = 3
_REQUEST_TIMEOUT = 25.0

# Auto-pick preference (cheapest capable first) when LLM_MODEL is unset.
# Legacy moonshot-v1-* models are a last resort, never preferred over Kimi.
_PREFERRED_MODELS = (
    "kimi-k2.6",
    "kimi-k3",
    "kimi-k2-turbo-preview",
    "kimi-k2-0905-preview",
    "kimi-k2",
    "kimi-latest",
)
_LEGACY_FALLBACKS = ("moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k")


class LLMVisionUnavailableError(Exception):
    """No vision-capable model is configured; image description cannot run."""


@dataclass(frozen=True)
class Classification:
    message_type: str  # "note" | "question" | "reminder" | "bookmark" | "other"
    summary: str
    topics: list[str]
    entities: list[str]
    due_at_iso: str | None  # ISO 8601 with timezone, reminders only


class ReminderParseError(Exception):
    """The reminder request had no usable due date (or a malformed one)."""


@dataclass(frozen=True)
class ReminderSpec:
    """Result of the dedicated reminder-parsing call."""

    what: str  # short, self-contained text to show when the reminder fires
    due_at_iso: str  # exact instant, ISO 8601 with timezone


_CLASSIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "message_type": {
            "type": "string",
            "enum": ["note", "question", "reminder", "bookmark", "other"],
        },
        "summary": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "entities": {"type": "array", "items": {"type": "string"}},
        "due_at": {"type": "string"},
    },
    "required": [
        "message_type",
        "summary",
        "topics",
        "entities",
        "due_at",
    ],
}

_CLASSIFY_INSTRUCTIONS = (
    "Classify this Telegram message for a personal memory system. "
    "Rules: "
    "- message_type 'reminder' only when the user asks to be reminded or told "
    "something at a future time, and then set due_at to that time as ISO 8601 "
    "with timezone; otherwise due_at must be the empty string. "
    "- 'bookmark' when the user forwards a link to save. "
    "- 'note' for information the user wants kept. "
    "- 'question' when the user asks something. "
    "- 'other' otherwise. "
    "- summary: one short sentence. "
    "- topics and entities: short lowercase keywords, may be empty. "
    "Respond with ONLY a JSON object matching exactly this schema "
    "(no prose, no code fences):"
)


def _classify_message(text: str, now_iso: str) -> str:
    """The full classify prompt (schema inlined; no str.format on braces)."""
    return (
        f"{_CLASSIFY_INSTRUCTIONS}\n{json.dumps(_CLASSIFY_SCHEMA)}\n"
        f"Current time: {now_iso}.\n\nMessage:\n{text}"
    )

_REMINDER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "what": {"type": "string"},
        "due_at": {"type": "string"},
    },
    "required": ["what", "due_at"],
}

_REMINDER_INSTRUCTIONS = (
    "Parse this reminder request. "
    "- what: short phrase of what the user wants to be reminded about, "
    "rewritten so it is understandable on its own when shown later "
    "(e.g. 'Call the dentist to book a cleaning'). "
    "- due_at: the exact date and time to fire, as ISO 8601 WITH timezone "
    "offset (e.g. 2026-09-20T09:00:00+02:00). Resolve relative phrases like "
    "'tomorrow 8pm' against the provided current time, in the user's local "
    "timezone when the request does not state one. Include seconds as :00. "
    "If no plausible future time can be parsed, set due_at to the empty string. "
    "Respond with ONLY a JSON object matching exactly this schema "
    "(no prose, no code fences):"
)


def _reminder_message(text: str, now_iso: str, tz_name: str) -> str:
    """The full reminder prompt (schema inlined; no str.format on braces)."""
    return (
        f"{_REMINDER_INSTRUCTIONS}\n{json.dumps(_REMINDER_SCHEMA)}\n"
        f"Current time: {now_iso}. User's local timezone: {tz_name}.\n\n"
        f"Message:\n{text}"
    )

_ANSWER_PROMPT = (
    "You are the user's personal memory. Answer ONLY from the memory excerpts "
    "below. Never invent facts. If they do not contain the answer, reply with "
    "exactly: I don't have anything saved about that yet.\n"
    "When an excerpt has a save timestamp, mention roughly when it was saved "
    "(e.g. 'you saved this back in July') so the user can judge staleness.\n\n"
    "Memory excerpts:\n{context}\n\nQuestion: {question}"
)

_IMAGE_DESC_PROMPT = (
    "Describe this image in ONE short factual sentence for a personal memory "
    "system. Include the most useful searchable details (objects, people, "
    "places, visible text subject). Do not speculate beyond what is visible."
)

_MEDIA_SUMMARY_PROMPT = (
    "Combine the pieces below from one short video into a single consolidated "
    "summary (2-4 sentences) for a personal memory system. Use ONLY these "
    "pieces. If they do not add up to anything meaningful, reply with exactly: "
    "NOTHING UNDERSTOOD.\n\nTranscript:\n{transcript}\n\nOn-screen text:\n"
    "{ocr}\n\nUser's caption:\n{caption}"
)

_NO_MEMORY_REPLY = "I don't have anything saved about that yet."


class LLMClient:
    """Thin OpenAI-compatible chat client with retries and JSON mode."""

    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.llm_api_key
        self._base_url = settings.llm_base_url.rstrip("/")
        self._vision_model = settings.llm_vision_model
        self._model = settings.llm_model or self._pick_default_model()
        # Some models (e.g. kimi-k3) reject any temperature other than 1, so
        # we send none by default and let each model apply its own default.
        self._temperature: float | None = None
        self._http = httpx.Client(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=_REQUEST_TIMEOUT,
        )
        logger.info("LLM ready: base_url=%s model=%s", self._base_url, self._model)

    def _chat_payload(
        self, model: str, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """The base chat payload for a given model (no temperature by default)."""
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        return payload

    def _pick_default_model(self) -> str:
        """Choose the cheapest capable model from the account's model list."""
        try:
            response = httpx.get(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=30.0,
            )
            response.raise_for_status()
            ids = [str(item.get("id", "")) for item in response.json().get("data", [])]
        except Exception as exc:
            raise RuntimeError(
                f"could not list models at {self._base_url} ({exc}); "
                "set LLM_MODEL explicitly in .env to skip auto-detection"
            ) from exc
        # Match against the preference list in order: first any exact id,
        # then any id containing a preferred name — never alphabetical order,
        # which would silently prefer the oldest/cheapest-looking name.
        for preferred in _PREFERRED_MODELS:
            for model_id in ids:
                if model_id == preferred:
                    return model_id
        for preferred in _PREFERRED_MODELS:
            for model_id in ids:
                if preferred in model_id.lower():
                    return model_id
        kimi_ids = sorted(m for m in ids if "kimi" in m.lower())
        if kimi_ids:
            return kimi_ids[0]
        for legacy in _LEGACY_FALLBACKS:
            for model_id in ids:
                if model_id == legacy:
                    return model_id
        raise RuntimeError(
            f"no Kimi-like model found in the account's model list: {ids}; "
            "set LLM_MODEL explicitly in .env"
        )

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """One raw chat/completions POST; raises for HTTP and network errors."""
        response = self._http.post("/chat/completions", json=payload)
        if response.status_code >= 400:
            detail = response.text[:300]
            raise httpx.HTTPStatusError(
                f"chat/completions failed: {response.status_code} {detail}",
                request=response.request,
                response=response,
            )
        return response.json()

    def _chat(
        self, messages: list[dict[str, Any]], json_mode: bool = False
    ) -> str:
        """Send messages, retrying transient errors; returns the reply text."""
        payload = self._chat_payload(self._model, messages)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return self._post_with_retries(payload)

    def _post_with_retries(self, payload: dict[str, Any]) -> str:
        delay = 1.5
        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                data = self._post_chat(payload)
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("llm reply content is not a string")
                return content
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in _RETRYABLE_STATUS:
                    raise
                last_error = exc
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
            if attempt < _MAX_ATTEMPTS:
                logger.warning(
                    "llm call failed (attempt %d/%d) — retrying in %.0fs: %s",
                    attempt,
                    _MAX_ATTEMPTS,
                    delay,
                    str(last_error)[:120],
                )
                time.sleep(delay)
                delay *= 2.0
        assert last_error is not None
        raise last_error

    def classify(self, text: str, now_iso: str) -> Classification:
        """Classify a message under a strict JSON schema."""
        content = self._chat(
            [{"role": "user", "content": _classify_message(text, now_iso)}],
            json_mode=True,
        )
        parsed = _parse_classification(content)
        logger.info(
            "classify: type=%s summary=%r", parsed.message_type, parsed.summary[:60]
        )
        return parsed

    def parse_reminder(self, text: str, now_iso: str, tz_name: str) -> ReminderSpec:
        """Resolve a reminder request into exact what + due_at.

        Raises ReminderParseError when no plausible future time can be parsed
        (due_at empty) so the caller can treat it as a bad date-parse.
        """
        if not now_iso or not tz_name:
            raise ValueError("parse_reminder requires now_iso and tz_name")
        content = self._chat(
            [{"role": "user", "content": _reminder_message(text, now_iso, tz_name)}],
            json_mode=True,
        )
        parsed = _parse_reminder(content)
        logger.info(
            "reminder parse: what=%r due_at=%s", parsed.what[:60], parsed.due_at_iso
        )
        return parsed

    def describe_image(self, image_bytes: bytes, mime_type: str) -> str:
        """One-line factual description of an image (only with a vision model).

        Raises LLMVisionUnavailableError when no vision model is configured;
        the pipeline then relies on OCR alone and says so honestly.
        """
        if not image_bytes:
            raise ValueError("describe_image requires image bytes")
        if not self._vision_model:
            raise LLMVisionUnavailableError(
                "no vision model configured (set LLM_VISION_MODEL to enable)"
            )
        b64 = base64.b64encode(image_bytes).decode("ascii")
        payload = self._chat_payload(
            self._vision_model,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                        },
                        {"type": "text", "text": _IMAGE_DESC_PROMPT},
                    ],
                }
            ],
        )
        content = self._post_with_retries(payload)
        return content.strip()

    def summarize_media(
        self, transcript: str, ocr_text: str, caption: str
    ) -> str:
        """One consolidated video summary from transcript + OCR + caption.

        Returns the empty string when the pieces add up to nothing (the caller
        then treats the media as stored-but-not-understood — never faked).
        """
        if not any((transcript.strip(), ocr_text.strip(), caption.strip())):
            return ""
        content = self._chat(
            [{"role": "user", "content": _MEDIA_SUMMARY_PROMPT.format(
                transcript=transcript or "(none)",
                ocr=ocr_text or "(none)",
                caption=caption or "(none)",
            )}]
        ).strip()
        if not content or "NOTHING UNDERSTOOD" in content:
            return ""
        return content

    def answer(self, question: str, context_chunks: list[str]) -> str:
        """Answer a question strictly from the given memory chunks."""
        if not context_chunks:
            return _NO_MEMORY_REPLY
        context = "\n".join(f"- {chunk}" for chunk in context_chunks)
        stripped = self._chat(
            [{"role": "user", "content": _ANSWER_PROMPT.format(
                context=context, question=question)}]
        ).strip()
        if not stripped:
            logger.error("llm answer returned empty text")
            return "Sorry, I had trouble generating an answer. Please try again."
        return stripped


def _parse_reminder(raw: str) -> ReminderSpec:
    """Validate the reminder JSON, or raise ReminderParseError for bad dates."""
    data = _loads_json(raw, "reminder parse")
    what = str(data["what"]).strip()
    due_at = str(data["due_at"]).strip()
    if not what:
        raise ReminderParseError("llm returned an empty 'what' for a reminder")
    if not due_at:
        raise ReminderParseError(
            "no plausible future time could be parsed from the request"
        )
    try:
        datetime.fromisoformat(due_at)  # validity check only
    except ValueError as exc:
        raise ReminderParseError(f"llm returned an invalid due_at: {due_at!r}") from exc
    return ReminderSpec(what=what, due_at_iso=due_at)


def _parse_classification(raw: str) -> Classification:
    data = _loads_json(raw, "classification")
    message_type = str(data["message_type"])
    if message_type not in ("note", "question", "reminder", "bookmark", "other"):
        raise ValueError(f"invalid message_type from classifier: {message_type!r}")
    due_at_raw = str(data["due_at"])
    return Classification(
        message_type=message_type,
        summary=str(data["summary"]),
        topics=[str(t) for t in data["topics"]],
        entities=[str(e) for e in data["entities"]],
        due_at_iso=due_at_raw or None,
    )


def _loads_json(raw: str, what: str) -> dict[str, Any]:
    """Parse JSON from a model reply, tolerating ```json fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"llm {what} returned invalid JSON: {text[:120]!r}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"llm {what} returned non-object JSON: {text[:120]!r}")
    return data


_CLIENT: LLMClient | None = None


def get_llm(settings: Settings) -> LLMClient:
    """Return the process-wide LLM client, building it on first use."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient(settings)
    return _CLIENT
