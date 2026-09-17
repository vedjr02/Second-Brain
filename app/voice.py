"""Local speech-to-text via faster-whisper (free, runs on CPU).

Phase 4 per the build spec: voice notes, audio files, and Telegram circular
video notes are transcribed locally and their transcripts flow into the exact
same Phase 1 text pipeline (classify -> note/question/reminder/bookmark).

The model is loaded lazily on first use and kept for the process lifetime
(first transcription downloads ~75 MB of weights, then everything is cached).
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# tiny keeps CPU latency low for short personal clips; bump for accuracy.
_WHISPER_MODEL = "tiny"
_TRANSCRIBE_KWARGS: dict[str, Any] = {"beam_size": 1}

_MODEL: Any = None


def _get_model() -> Any:
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        logger.info("loading faster-whisper model %r (first run downloads it)", _WHISPER_MODEL)
        _MODEL = WhisperModel(_WHISPER_MODEL, device="cpu", compute_type="int8")
    return _MODEL


def transcribe(path: str) -> str:
    """Transcribe an audio/video file; returns whitespace-collapsed text.

    Raises on unreadable files — callers treat transcription failure as a
    routing error, never as empty content.
    """
    segments, _info = _get_model().transcribe(path, **_TRANSCRIBE_KWARGS)
    parts = [segment.text.strip() for segment in segments]
    return " ".join(part for part in parts if part)
