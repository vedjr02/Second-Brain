# Progress log

## Where the project stands — 17 September 2026

**All six build-spec phases (0–5) are implemented, tested and type-clean.**
86 offline tests pass, `mypy` reports no issues, and the reel pipeline has been
run end to end against a live YouTube Shorts URL (download → merge → local
transcription → ffmpeg keyframes → OCR → LLM summary).

| Phase | What it covers | State |
|---|---|---|
| 0 | FastAPI scaffold, webhook endpoint + secret, DB schema & probe | done |
| 1 | Text classification (strict JSON), notes/bookmarks, grounded Q&A | done |
| 2 | Reminder parsing, `/check-reminders`, GitHub Actions cron | done |
| 3 | Photos: pointer-first, local OCR, optional vision one-liner | done |
| 4 | Voice notes / audio / video notes: local faster-whisper transcription | done |
| 5 | Reels & video links: yt-dlp + ffmpeg + OCR + consolidated summary | done |

Deviations from the original spec, both deliberate:
- **SQLite instead of Neon/Postgres+pgvector.** Embeddings are little-endian
  float32 BLOBs and cosine similarity is computed in Python — trivially fast at
  personal scale, and there is no external database to sign up for or keep warm.
- **Kimi (Moonshot) instead of Gemini** as the reasoning/classification model,
  via an OpenAI-compatible chat client. Vision is optional (`LLM_VISION_MODEL`);
  without it photos rely on local OCR alone and the bot says so honestly.

## This session (17 September 2026) — fixes and completion

Phases 4 and 5 existed in code but had defects that made the reel path
effectively dead on arrival, plus no test coverage. Fixed:

1. **yt-dlp was invoked as a bare `yt-dlp` command.** The pip-installed console
   script lives in `.venv/bin/`, which is not on `PATH` when the app runs as
   `.venv/bin/python -m app` — so *every* reel silently fell back to bookmark.
   Now invoked as `sys.executable -m yt_dlp`.
2. **Links were downloaded audio-only** (`bestaudio[ext=m4a]`), so keyframe
   extraction and OCR — half of what makes a reel searchable — could never
   produce anything. Now downloads video+audio under an 80 MB cap, with an
   audio-only last resort, and finds whatever extension yt-dlp actually wrote.
3. **Links were only detected at the very start of a message** and only when
   the classifier happened to say "bookmark". A realistic message ("save this
   pasta reel <url> for dinner") was stored as plain text. Now the URL is found
   anywhere in the message, for notes and bookmarks alike, and the surrounding
   text becomes the caption (the old caption extraction — deleting the first
   word from the message — was wrong whenever the link was not first).
4. **A reel that yielded nothing still replied "the video is summarized and
   searchable."** That breaks runtime rule 5. It now stores the link as a plain
   bookmark and says it never understood the contents.
5. **`build_media_summary` re-read the whole environment** (`load_settings()`)
   on every video instead of taking the settings it was called with. Settings
   are now threaded through `process_video_file` / `process_video_link`.
6. **An uploaded Telegram video whose download failed crashed the handler**
   with no reply to the user. Now handled like every other failure path.
7. **A failed transcription discarded working keyframe OCR.** The two
   extractors are now independent: either one alone still produces a memory.
8. **Blocking work ran on the event loop**: embedding (`embed_texts`) and note
   storage are CPU-bound and now run via `asyncio.to_thread`, as the rest of
   the pipeline already did. Also removed a stray `get_llm()` call in the photo
   handler that could hit the network for nothing.
9. **`httpx` was used directly by `app/llm.py` but unpinned** — it was only
   present as a transitive dependency of python-telegram-bot. Now in
   `requirements.txt`.
10. **Stale copy**: the bot told users voice and video were "coming in later
    phases", and docstrings still referenced Postgres/pgvector. Updated, along
    with the `/start` intro and the README.

Added `tests/test_reels.py` (Phase 5 had zero coverage) plus four pipeline
tests for video-link routing and both bookmark fallbacks: 72 → 86 tests.

## Verified this session

```
.venv/bin/python -m pytest -q      # 86 passed
.venv/bin/python -m mypy           # no issues in 24 source files
.venv/bin/python -m scripts.check_db   # probe insert/read/delete ok
```
Live: yt-dlp downloaded and merged a real YouTube Short, faster-whisper
transcribed it, ffmpeg + pytesseract ran on its keyframes, and the Kimi call
returned — with the clip having no speech and no on-screen text, the pipeline
correctly reported that it understood nothing rather than inventing a summary.

## Known gaps / next steps

- **Render free tier wipes the SQLite file on redeploy.** Attach a Render Disk
  and set `SQLITE_PATH=/var/data/second_brain.db` before relying on it.
- `LLM_VISION_MODEL` is unset, so photos without readable text are stored but
  not searchable. Set it to a Kimi VL model to enable one-line descriptions.
- Long notes are stored as a single chunk — fine for short personal notes,
  worth splitting if long articles get forwarded often.
- No way to list, edit, or delete saved memories from chat (e.g. `/recent`,
  "forget that"); everything is append-only today.
- yt-dlp needs periodic upgrades; platforms break scrapers regularly. The
  bookmark fallback means a stale yt-dlp degrades quietly rather than breaking.
