"""Tests for the Phase 5 reel helpers (URL handling, download, summary)."""

import os
import subprocess
from typing import Any

import pytest

import app.reels as reels
from app.settings import Settings


def _settings() -> Settings:
    return Settings(
        telegram_bot_token="t", telegram_webhook_secret="s", llm_api_key="k"
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("https://instagram.com/reel/abc", "https://instagram.com/reel/abc"),
        ("look at this https://youtu.be/xyz lol", "https://youtu.be/xyz"),
        ("(https://tiktok.com/@a/video/1)", "https://tiktok.com/@a/video/1"),
        ("no link here at all", ""),
        ("email me at a@b.com", ""),
    ],
)
def test_extract_url_finds_the_link_anywhere(text: str, expected: str) -> None:
    assert reels.extract_url(text) == expected
    assert reels.is_video_link(text) is bool(expected)


def test_strip_url_leaves_the_users_own_caption() -> None:
    assert (
        reels.strip_url("save this pasta reel https://insta.com/r/1 for dinner")
        == "save this pasta reel for dinner"
    )


def test_process_video_link_raises_when_yt_dlp_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A failed download must raise so the caller can bookmark instead (rule 5)."""

    def fake_run(*_args: Any, **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess([], 1, "", "ERROR: login required")

    monkeypatch.setattr(reels.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="login required"):
        reels.process_video_link(_settings(), "https://x/1", "", str(tmp_path))


def test_process_video_link_uses_whatever_extension_yt_dlp_wrote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    written = tmp_path / "source.mp4"

    def fake_run(*_args: Any, **_kwargs: Any) -> Any:
        written.write_bytes(b"video-bytes")
        return subprocess.CompletedProcess([], 0, "", "")

    processed: list[str] = []

    def fake_process(_settings: Any, path: str, _caption: str) -> str:
        processed.append(path)
        return "ok"

    monkeypatch.setattr(reels.subprocess, "run", fake_run)
    monkeypatch.setattr(reels, "_process", fake_process)

    result = reels.process_video_link(
        _settings(), "https://x/1", "caption", str(tmp_path)
    )

    assert result == "ok"
    assert processed == [str(written)]


def test_process_keeps_going_when_transcription_fails(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silent/undecodable audio track must not discard the keyframe OCR."""
    def boom(_path: str) -> str:
        raise RuntimeError("no audio stream")

    monkeypatch.setattr(reels.voice, "transcribe", boom)
    monkeypatch.setattr(reels, "_ocr_keyframes", lambda _p: ["50% OFF TODAY"])
    seen: dict[str, str] = {}
    monkeypatch.setattr(
        reels,
        "build_media_summary",
        lambda _s, transcript, ocr, caption: seen.update(
            transcript=transcript, ocr=ocr, caption=caption
        )
        or "summary",
    )

    assert reels._process(_settings(), "/tmp/x.mp4", "my caption") == "summary"
    assert seen == {"transcript": "", "ocr": "50% OFF TODAY", "caption": "my caption"}


def test_build_media_summary_falls_back_to_raw_pieces_when_the_llm_fails(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenLLM:
        def summarize_media(self, **_kwargs: Any) -> str:
            raise RuntimeError("api down")

    monkeypatch.setattr(reels, "get_llm", lambda _s: BrokenLLM())

    result = reels.build_media_summary(
        _settings(), "spoken words", "on screen", "my caption"
    )

    assert result == "my caption on screen spoken words"


@pytest.mark.parametrize(
    "stderr,expected",
    [
        ("ERROR: [Instagram] Requested content is not available, login required", True),
        ("ERROR: Sign in to confirm you're not a bot", True),
        ("ERROR: Unsupported URL", False),
        ("", False),
    ],
)
def test_needs_login_recognises_a_blocked_platform(stderr: str, expected: bool) -> None:
    assert reels.needs_login(stderr) is expected


def test_cookies_are_passed_to_yt_dlp_when_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    seen: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> Any:
        seen.append(command)
        (tmp_path / "source.mp4").write_bytes(b"v")
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(reels.subprocess, "run", fake_run)
    monkeypatch.setattr(reels, "_process", lambda _s, _p, _c: "ok")
    settings = Settings(
        telegram_bot_token="t", telegram_webhook_secret="s", llm_api_key="k",
        ytdlp_cookies_from_browser="chrome",
    )

    reels.process_video_link(settings, "https://x/1", "", str(tmp_path))

    assert "--cookies-from-browser" in seen[0]
    assert "chrome" in seen[0]
