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
import re
import subprocess
import sys
import tempfile

from PIL import Image

from . import voice
from .llm import get_llm
from .settings import Settings

logger = logging.getLogger(__name__)

# Keyframes sampled at these fractions of the video (deduped, 1..5 frames).
_KEYFRAME_POSITIONS = (0.1, 0.35, 0.6, 0.85)

# Keeps a downloaded reel small enough to process on a free-tier box; the
# video stream is wanted (not just audio) because keyframe OCR is half of
# what makes a reel searchable.
_MAX_DOWNLOAD_MB = 80
_URL_PATTERN = re.compile(r"https?://\S+")


def extract_url(text: str) -> str:
    """First http(s) URL anywhere in the message ('' when there is none).

    Links rarely arrive alone — "save this pasta reel <url>" is the normal
    shape — so the URL is found wherever it sits, not only at the start.
    """
    match = _URL_PATTERN.search(text)
    if match is None:
        return ""
    return match.group(0).rstrip(').,!?"\'')


def strip_url(text: str) -> str:
    """The message with its URLs removed — i.e. the user's own caption."""
    return " ".join(_URL_PATTERN.sub(" ", text).split())


class DownloadError(RuntimeError):
    """yt-dlp could not fetch the video. `login_required` explains why."""

    def __init__(self, detail: str, login_required: bool = False) -> None:
        super().__init__(f"yt-dlp failed: {detail}")
        self.login_required = login_required


# What yt-dlp says when a platform will only serve a signed-in session.
_LOGIN_MARKERS = (
    "login required",
    "rate-limit reached",
    "requested content is not available",
    "sign in to confirm",
    "cookies",
    "private video",
)


def needs_login(stderr: str) -> bool:
    """Whether a download failed because the platform wanted a session."""
    lowered = stderr.lower()
    return any(marker in lowered for marker in _LOGIN_MARKERS)


def is_video_link(text: str) -> bool:
    """True when the message contains a URL (yt-dlp decides what actually works)."""
    return bool(extract_url(text))


def process_video_file(settings: Settings, path: str, caption: str) -> str:
    """Full extraction for an already-downloaded video file (no re-download)."""
    return _process(settings, path, caption)


def process_video_link(
    settings: Settings, url: str, caption: str, workdir: str
) -> str:
    """Download a video from a link with yt-dlp, then run the same extraction.

    Video+audio is preferred (keyframes carry on-screen text, which for many
    reels is the only content there is); if the platform only yields audio we
    still transcribe it rather than failing the whole message.
    """
    target = os.path.join(workdir, "source.%(ext)s")
    fmt = (
        f"best[filesize<{_MAX_DOWNLOAD_MB}M]/"
        f"bv*[filesize<{_MAX_DOWNLOAD_MB}M]+ba/b/bestaudio"
    )
    # Invoked through the running interpreter: the pip-installed yt-dlp
    # console script lives in the venv's bin/, which is NOT on PATH when the
    # app is started as `.venv/bin/python -m app`.
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-warnings",
        "--quiet",
        "--no-playlist",
        "-f",
        fmt,
        "-o",
        target,
        "--",
        url,
    ]
    if settings.ytdlp_cookies_from_browser:
        # Instagram and TikTok serve most posts only to a signed-in session.
        command += ["--cookies-from-browser", settings.ytdlp_cookies_from_browser]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    downloaded = _downloaded_file(workdir)
    if result.returncode != 0 or downloaded is None:
        stderr = (result.stderr or "").strip()
        raise DownloadError(stderr[:200] or "no file produced", needs_login(stderr))
    return _process(settings, downloaded, caption)


def _downloaded_file(workdir: str) -> str | None:
    """The file yt-dlp actually wrote (its extension depends on the format)."""
    candidates = [
        os.path.join(workdir, name)
        for name in os.listdir(workdir)
        if name.startswith("source.")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getsize)


def _process(settings: Settings, path: str, caption: str) -> str:
    """Transcribe audio + OCR keyframes + LLM summary for one video."""
    try:
        transcript = voice.transcribe(path)
    except Exception:
        logger.warning("transcription failed for %s; continuing on OCR", path, exc_info=True)
        transcript = ""
    keyframe_texts = _ocr_keyframes(path)
    return build_media_summary(
        settings, transcript, " ".join(keyframe_texts), caption
    )


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


def build_media_summary(
    settings: Settings, transcript: str, ocr_text: str, caption: str
) -> str:
    """One consolidated summary via the LLM, honest about missing pieces.

    When there is no transcript and no OCR text (music-only reel, silent
    video), no LLM call is made and the caption alone is returned — the bot
    never pretends content was understood when it wasn't.
    """
    llm = get_llm(settings)
    try:
        return llm.summarize_media(
            transcript=transcript, ocr_text=ocr_text, caption=caption
        )
    except Exception:
        logger.exception("media summary LLM call failed; using raw pieces")
        parts = [part for part in (caption, ocr_text, transcript) if part]
        return " ".join(parts)
