"""Reel/video processing (Phase 5): links via yt-dlp, files via Telegram.

Pipeline per the build spec: download from the source platform (bypasses
Telegram's file-size caps entirely), extract audio + a few keyframes with
ffmpeg, transcribe locally (faster-whisper), OCR the keyframes (pytesseract),
and combine transcript + OCR + caption into one summary via the LLM.

Rule 5: if the download fails for ANY reason (platform blocked the scraper,
private post, unsupported site), the caller stores the link + caption as a
lightweight bookmark instead — this phase is never a hard failure point, and
the bot says plainly when content only exists as a bookmark.
"""

import logging
import os
import subprocess
import tempfile

from PIL import Image

from . import voice
from .llm import get_llm
from .settings import Settings

logger = logging.getLogger(__name__)

# Keyframes sampled at these fractions of the video (deduped, 1..5 frames).
_KEYFRAME_POSITIONS = (0.1, 0.35, 0.6, 0.85)


def is_video_link(text: str) -> bool:
    """True when the message is a URL (yt-dlp handles which sites actually work)."""
    lowered = text.strip().lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def process_video_file(path: str, caption: str) -> str:
    """Full extraction for an already-downloaded video file (no re-download)."""
    return _process(path, caption)


def process_video_link(
    settings: Settings, url: str, caption: str, workdir: str
) -> str:
    """Download a video from a link with yt-dlp, then run the same extraction."""
    target = os.path.join(workdir, "source.m4a")  # audio-only keeps it small
    command = [
        "yt-dlp",
        "--no-warnings",
        "--quiet",
        "-f",
        "bestaudio[ext=m4a]/bestaudio/best",
        "-o",
        target,
        "--",
        url,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()[:200]}")
    if not os.path.exists(target):
        raise RuntimeError("yt-dlp reported success but produced no file")
    return _process(target, caption)


def _process(path: str, caption: str) -> str:
    """Transcribe audio + OCR keyframes + LLM summary for one video."""
    transcript = voice.transcribe(path)
    keyframe_texts = _ocr_keyframes(path)
    return build_media_summary(transcript, " ".join(keyframe_texts), caption)


def _ocr_keyframes(path: str) -> list[str]:
    """Extract a handful of keyframes with ffmpeg and OCR each one."""
    import pytesseract  # type: ignore[import-untyped]

    texts: list[str] = []
    try:
        duration = _probe_duration(path)
    except Exception:
        logger.warning("could not probe video duration; using default positions")
        duration = 0.0
    with tempfile.TemporaryDirectory(prefix="reel-") as tmp:
        for index, fraction in enumerate(_KEYFRAME_POSITIONS):
            at = duration * fraction if duration > 0 else fraction * 10
            frame = os.path.join(tmp, f"frame-{index}.jpg")
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{max(at, 0):.2f}",
                    "-i",
                    path,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "3",
                    frame,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0 or not os.path.exists(frame):
                continue
            try:
                with Image.open(frame) as image:
                    text = pytesseract.image_to_string(image)
                cleaned = " ".join(text.split())
                if cleaned:
                    texts.append(cleaned)
            except Exception:
                logger.warning("keyframe OCR failed for frame %d", index, exc_info=True)
    return texts


def _probe_duration(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return float(result.stdout.strip())


def build_media_summary(transcript: str, ocr_text: str, caption: str) -> str:
    """One consolidated summary via the LLM, honest about missing pieces.

    When there is no transcript and no OCR text (music-only reel, silent
    video), no LLM call is made and the caption alone is returned — the bot
    never pretends content was understood when it wasn't.
    """
    from .settings import load_settings

    settings = load_settings()
    llm = get_llm(settings)
    try:
        return llm.summarize_media(
            transcript=transcript, ocr_text=ocr_text, caption=caption
        )
    except Exception:
        logger.exception("media summary LLM call failed; using raw pieces")
        parts = [part for part in (caption, ocr_text, transcript) if part]
        return " ".join(parts)
