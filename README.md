# Personal Second Brain over Telegram

Chat with your own memory system: forward it notes, photos, voice notes, and reels;
ask it questions later and get answers grounded in what you actually saved — never
invented. Full spec: `second-brain-telegram-buildspec.md`.

**Status: Phase 3 (photos) complete.** Phase 0 (webhook, schema, probe), Phase 1
(text + grounded Q&A) and Phase 2 (reminders) also done. Phases 4–5 (voice, reels)
not started.

## Project layout

```
app/
  main.py        FastAPI app: lifespan, /healthz, /telegram/webhook, /check-reminders
  telegram.py    python-telegram-bot wiring (webhook mode, no polling), handler registration
  pipeline.py    routing: classify -> note/question/reminder/bookmark/other (+ reminder confirm);
                 photo ingestion: pointer-first, fetch once, OCR, conditional vision
  llm.py         LLM client (Kimi/Moonshot): classifier, reminder date parser, grounded answers
  embeddings.py  fastembed (all-MiniLM-L6-v2, 384 dims, L2-normed) as a lazy singleton
  memory.py      Message/chunk/reminder persistence + local cosine similarity search
  reminders.py   Phase 2 worker: claims due reminders and pushes them to Telegram
  db.py          SQLite access (WAL), schema (messages/memory_chunks/reminders), probe
  settings.py    Env-based settings, fails loudly if anything required is missing
scripts/
  check_db.py    Standalone DB check: schema + insert/read/delete probe
tests/           65 offline tests (endpoints, secret auth, routing, reminders, photos, LLM parsing)
.github/workflows/check-reminders.yml  cron: POST /check-reminders every minute
```

## Setup

1. **Tesseract** (OCR binary, for photos): macOS `brew install tesseract`;
   Debian/Ubuntu `sudo apt install tesseract-ocr`. Without it, photos still
   save (pointer + optional vision description); with it, in-image text is OCR'd
   locally for free. On Render, add a build step: `apt-get update && apt-get
   install -y tesseract-ocr` (or use the `apt` build hook).
2. **BotFather**: create the bot with `/newbot`, copy the token.
3. **Database**: nothing to sign up for — the bot uses a local SQLite file
   (`second_brain.db`, created automatically on first boot). Optionally set
   `SQLITE_PATH` in `.env` to store it elsewhere.
4. **Kimi (Moonshot AI) API key**: create one at https://platform.moonshot.ai
   → API Keys. Classification, reminder parsing, and answers run on Kimi chat
   models (leave `LLM_MODEL` empty and the app auto-picks); embeddings and OCR
   run locally on CPU and are free. There is no free tier — you pay per token
   (fractions of a cent per message on flash-class models); keep an eye on your
   balance in the Moonshot console.
5. Configure and verify locally:
   ```bash
   cp .env.example .env          # fill in TELEGRAM_BOT_TOKEN, TELEGRAM_WEBHOOK_SECRET, LLM_API_KEY
   python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
   .venv/bin/python -m scripts.check_db        # must print probe JSON, no error
   .venv/bin/python -m pytest -q               # 56 passed, 1 skipped
   .venv/bin/python -m mypy                    # no issues
   ```
   `check_db` proves: the SQLite file is live, schema created, row insert → read → delete works.
5. **Local end-to-end** (optional, needs a tunnel):
   ```bash
   ngrok http 8000   # or cloudflared
   # add WEBHOOK_BASE_URL=https://<tunnel-host> to .env
   .venv/bin/python -m app
   ```
   Then in Telegram:
   - `/start` -> intro message
   - "Remember: my wifi password is hunter2" -> **Saved.**
   - "what is my wifi password?" -> grounded answer (with roughly when it was
     saved), or **I don't have anything saved about that yet.** when memory is empty
   - "remind me about the dentist Sunday 9am" ->
     **Got it — reminding you about Call the dentist … on Sunday 20 September at 09:00 UTC.**
     (set `USER_DISPLAY_TIMEZONE` to your IANA timezone so the echo is in local time)
   - send a photo of text (a business card, a screenshot) ->
     **Saved — text in the image is searchable.** then ask about it:
     "what was the electrician's number I sent?"

## Phase 2: reminders

- Reminder messages get a second LLM call that resolves the exact date/time
  (relative phrases resolved against the real current clock and your timezone)
  plus a clean "what" phrase. The bot echoes exactly what it understood
  ("Got it — reminding you about …") so a bad parse is caught immediately.
- If no plausible time can be parsed, the note is still saved but the bot says
  it couldn't work out the time — nothing silently misfires.
- A GitHub Actions cron (`.github/workflows/check-reminders.yml`) POSTs
  `/check-reminders` every minute with `Authorization: Bearer $REMINDER_CHECK_SECRET`.
  Setup: push the repo to GitHub, then in **Settings → Secrets and variables →
  Actions** add `REMINDER_BASE_URL` (your Render URL) and `REMINDER_CHECK_SECRET`
  (same value as the backend env var), and run the workflow once via
  **Run workflow** to enable the schedule.
- Firing is claim-before-send: an atomic `UPDATE … WHERE fired = 0` in SQLite
  prevents double-fires even if cron runs overlap; a failed Telegram
  send releases the claim so the next run retries.

## Deploy to Render (Docker)

The repo includes a `Dockerfile` bundling Python, tesseract, and ffmpeg, so
photo/voice/video pipelines behave identically to local.

1. Push this repo to GitHub.
2. Render dashboard → **New → Web Service** → connect the repo.
3. Runtime: **Docker** (auto-detected from the Dockerfile).
4. Env vars: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_WEBHOOK_SECRET`, `LLM_API_KEY`,
   `WEBHOOK_BASE_URL=https://<your-service>.onrender.com`,
   `USER_DISPLAY_TIMEZONE=<your IANA timezone>`,
   `REMINDER_CHECK_SECRET=<long random string>`, `LLM_MODEL=kimi-k2.6`
   (optional: `LLM_VISION_MODEL`, `SQLITE_PATH`, `SIMILARITY_THRESHOLD`,
   `SEARCH_TOP_K`)
5. Deploy → verify `https://<service>/healthz` → the app registers the
   Telegram webhook itself at boot (schema created automatically).
6. Reminder cron: repo → Settings → Secrets and variables → Actions → add
   `REMINDER_BASE_URL=https://<your-service>.onrender.com` and
   `REMINDER_CHECK_SECRET` (same value as the backend). The included workflow
   then fires due reminders every minute.

**Persistence caveat**: Render free-tier disks are ephemeral — a redeploy
wipes `second_brain.db`. Attach a Render Disk (paid) and set
`SQLITE_PATH=/var/data/second_brain.db`, or accept losing saved memories on
redeploys while testing.

## Local polling vs webhook

- `python -m app.local` — long polling for development, no tunnel needed;
  reminders fire from a built-in 30-second check loop.
- `python -m app` — webhook mode for production (needs `WEBHOOK_BASE_URL`);
  reminders fire from the GitHub Actions cron hitting `/check-reminders`.

## Phase 3: photos

- Photos are stored **pointer-first**: only the Telegram `file_id` (plus chat/message
  ids) lands in the database — the raw image never leaves Telegram's servers, per
  the storage strategy. The file is fetched exactly once, for extraction.
- The image is OCR'd locally with pytesseract (free). If OCR finds text, that
  text becomes the memory chunk — no vision call, no API spend.
- If OCR finds nothing usable and `LLM_VISION_MODEL` is set, one vision call
  adds a one-line factual
  description. If OCR fails because the binary is missing, vision takes over
  automatically; a vision failure never discards OCR text that already worked.
- If the pipeline can't extract anything, the bot says so plainly: the image is
  stored (pointer kept) but marked not searchable — never pretended-understood.
- Images over Telegram's 20 MB Bot API fetch cap are flagged, not silently dropped.

## Phase 1 semantics (per the build spec)

- Every incoming text is classified by the LLM under a strict JSON schema
  (note / question / reminder / bookmark / other + summary/topics/entities/due_at).
- Notes and bookmarks are embedded locally (MiniLM, no API cost) and stored in
  `memory_chunks` with their 384-dim vector and topic tags.
- Questions retrieve the top-k chunks (cosine similarity, per-chat, above the
  threshold) and the LLM answers ONLY from them — with an empty memory the bot
  says it has nothing saved, it never invents.
- Reminder requests also get a dedicated date-parsing call and are stored as
  `reminders` rows — see Phase 2 above.
- "Other" chatter gets a polite no-op and is not stored.
- Non-text messages (photos, voice, video) get a "coming soon" reply — Phase 3.

## What Phase 3 intentionally does NOT do

No voice notes or video ingestion (Phases 4–5), no chunking of long notes, no
per-photo classification beyond the vision one-liner, no auth beyond the
webhook + cron secrets.
