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

## Second session (17 September 2026) — durability, privacy, retrieval

The remaining blocker was never Neon vs SQLite; it was that a free host has no
durable disk. Solved without adding any service: **the database backs itself
up to Telegram**, which is already the blob store for media.

- `app/backup.py`: a consistent snapshot (SQLite's online backup API, not a
  file copy) is sent to the owner's chat every 6 hours and **pinned**. Pinning
  is what makes recovery possible — after a wipe nothing remains that knows
  where the backup went, but `getChat` hands back the pinned message and its
  `file_id`. On boot, an empty database restores itself from that file before
  the bot serves anything.
- Restore only runs against an empty database, so it can never clobber live
  memories. A failed backup is not recorded as successful, so it retries at
  the next tick instead of being skipped for hours.
- The cron tick that already fires reminders doubles as the backup scheduler —
  a webhook-mode service gets no other heartbeat.

**Privacy**: a bot username is public, so anyone who found yours could write
to your brain. Setting `OWNER_CHAT_ID` now locks the bot to your chat (and is
the same value the backup needs). Left open when unset so a fresh install is
usable before you know your chat id — `/chatid` tells you.

**Retrieval got substantially better.** Pure vector search is weak at exactly
what a second brain stores: phone numbers, names, passwords, product names —
rare tokens with no useful embedding that nonetheless match literally. Every
chunk is now scored twice (cosine similarity + the fraction of the question's
distinctive words it contains) and retrieved when either is convincing.
Verified live: "what was the electrician's number?" now finds "Dave the
electrician: 07700 900123", which pure cosine missed, and asking about "my
sister's flight" when only Mum's flight is saved still correctly answers
"I don't have anything saved about that yet."

**Commands added**: `/help`, `/recent`, `/forget <n>`, `/status`, `/backup`,
`/chatid` — the "no way to list or delete memories" gap from the last session.

Tests: 86 → 105 (new `tests/test_backup.py`, owner-gate and command tests,
hybrid-retrieval tests against a real temp database).

## Known gaps / next steps

- **Set `OWNER_CHAT_ID` before relying on this.** Until it is set the bot is
  open to anyone who finds it AND has no backups — the two things that make it
  actually yours. Send `/chatid`, put the number in the env, restart.
- `LLM_VISION_MODEL` is unset, so photos without readable text are stored but
  not searchable. Set it to a Kimi VL model to enable one-line descriptions.
- Long notes are stored as a single chunk — fine for short personal notes,
  worth splitting if long articles get forwarded often.
- Editing a memory in place is still not possible (delete and re-send).
- The backup has never been exercised against a live Telegram chat — the logic
  is covered by tests, but the first real `/backup` is worth watching.
- yt-dlp needs periodic upgrades; platforms break scrapers regularly. The
  bookmark fallback means a stale yt-dlp degrades quietly rather than breaking.
