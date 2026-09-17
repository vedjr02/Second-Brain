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

## Third session (17 September 2026) — why it went silent, and bursts

**"The bot ignored my messages" was not the bot ignoring anything.** Two
separate causes, one after the other:

1. Nothing was running. No process, no webhook (`WEBHOOK_BASE_URL` is empty —
   nothing is deployed yet), so the messages sat in Telegram's update queue.
   Started it with `python -m app.local` and set `OWNER_CHAT_ID=8787286992`,
   read out of the queued updates. The first real backup ran on boot and
   pinned itself — the untested path from last session, now proven live.
2. Then a Kimi call hit `The read operation timed out` and retried 4 times at
   a 60-second timeout with 2.5x backoff: roughly four minutes of silence
   before any reply. Measured `kimi-k2.6` directly at 2.9s, so this was a
   transient API blip, not a slow model. Timeout is now 25s, 3 attempts, 2x
   backoff, and the failure reply says what actually went wrong instead of a
   generic apology.

**Message grouping (`app/grouping.py`)** — the habit of sending one sentence
across several messages is now first-class. Text is debounced per chat for 8
seconds; each new message restarts the timer; the burst is joined into one
message, classified once, and answered under the last message of the burst.
Fragments get a full stop when joined so the classifier reads them as separate
clauses.

**Media context**: a text arriving within 5 minutes of a photo/voice/video is
filed against THAT message instead of its own. This is the exact case that
failed — an unreadable photo followed by "remember I want to post this
tomorrow" used to leave the photo unsearchable and the note pointing nowhere;
now both live in one memory that keeps the `file_id` pointer, and the bot says
"added to the photo you just sent". The link expires, so an old photo never
captures a later unrelated note.

Tests: 105 → 110.

## Fourth session (18 September 2026) — the rest of the backlog

Everything left on the list got built, in small commits pushed to
`github.com/vedjr02/Second-Brain` as they landed.

- **Buffered text survives a restart.** `grouping.flush_all()` is now wired
  into both shutdown paths; a Ctrl+C inside the 8-second window used to drop
  whatever had been typed.
- **Long text is chunked.** Sentence-aligned, overlapping windows, applied to
  notes and to reel/OCR transcripts. One MiniLM vector cannot represent a
  whole article, so the middle of a long note was previously unretrievable.
- **Media comes back.** When an answer comes from a photo, voice note or
  video, the original file is re-sent with the answer. The row only ever kept
  the `file_id`, so this costs nothing.
- **Duplicates are skipped.** Word-for-word repeats are not stored twice.
- **Commands**: `/find` (search with no model call), `/reminders`, `/cancel`,
  `/export` (plain text, so the notes are never trapped in the bot),
  `/timezone`. All registered with Telegram so they appear behind the "/"
  button, with a test pinning the menu against the registered handlers so a
  new command cannot be added and left hidden.
- **Timezone is settable from chat** and stored in the database, beating the
  env var. It was UTC, which would have fired every reminder at the wrong
  hour. Reminder parsing, the confirmation echo, the fired message,
  `/reminders` and `/status` all read the effective zone.
- **Reel downloads distinguish "blocked by login" from "broken".** Instagram
  and TikTok serve most posts only to a signed-in session;
  `YTDLP_COOKIES_FROM_BROWSER` lets yt-dlp reuse a browser session, and the
  failure reply now names that setting instead of vaguely blaming the platform.
- **Stale media links are swept**, so the "text after a photo" map cannot grow
  unbounded and a day-old photo cannot capture a new note.

Tests: 110 → 140.

## Known gaps / next steps

- **Not deployed.** It runs locally via `python -m app.local`, so it only
  works while the machine is awake. Render + webhook is the next real step.
- `LLM_VISION_MODEL` is unset, so photos without readable text are stored but
  not searchable. Set it to a Kimi VL model to enable one-line descriptions.
- Long notes are stored as a single chunk — fine for short personal notes,
  worth splitting if long articles get forwarded often.
- Editing a memory in place is still not possible (`/forget` then re-send).
- `/timezone` has not been set yet — it is still UTC until you set it.
- Asking to see a saved photo again does not re-fetch it from Telegram yet;
  the `file_id` is kept, so the plumbing for it exists.
- yt-dlp needs periodic upgrades; platforms break scrapers regularly. The
  bookmark fallback means a stale yt-dlp degrades quietly rather than breaking.
