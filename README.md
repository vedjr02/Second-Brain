# Personal Second Brain over Telegram

Chat with your own memory system: forward it notes, photos, voice notes, and reels;
ask it questions later and get answers grounded in what you actually saved — never
invented. Full spec: `second-brain-telegram-buildspec.md`.

**Status: all phases (0–5) complete and verified.** Webhook + schema (0), text
ingestion and grounded Q&A (1), reminders (2), photos (3), voice notes (4), and
reels/video links (5). See `progress.md` for what was built when and what is
deliberately left out.

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
  voice.py       Phase 4: local faster-whisper transcription (lazy-loaded model)
  reels.py       Phase 5: yt-dlp download, ffmpeg keyframes, OCR + LLM summary
  reminders.py   Phase 2 worker: claims due reminders and pushes them to Telegram
  backup.py      Snapshots the database to Telegram and restores it after a wipe
  grouping.py    Debounces rapid-fire messages into one thought before routing
  db.py          SQLite access (WAL), schema (messages/memory_chunks/reminders), probe
  settings.py    Env-based settings, fails loudly if anything required is missing
scripts/
  check_db.py    Standalone DB check: schema + insert/read/delete probe
tests/           140 offline tests (endpoints, secret auth, routing, reminders, photos, LLM parsing)
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
   .venv/bin/python -m pytest -q               # 140 passed
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
   `OWNER_CHAT_ID=<your chat id>` (strongly recommended — it is what makes the
   bot private and its data survive redeploys),
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

**Persistence**: Render free-tier disks are ephemeral — a redeploy wipes
`second_brain.db`. Set `OWNER_CHAT_ID` (see below) and the bot backs itself up
to your own Telegram chat and restores from there automatically, so a wipe
costs you at most the last few hours. A Render Disk (paid) with
`SQLITE_PATH=/var/data/second_brain.db` is the belt-and-braces option.

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

## Durability: the database backs itself up to Telegram

Free hosting has no durable disk, and the whole point of a second brain is
that it does not forget. Telegram is already the blob store for media, so it
stores the database too:

- Every few hours (`BACKUP_INTERVAL_HOURS`, default 6) the bot takes a
  consistent snapshot with SQLite's online backup API, sends it to your own
  chat as a document, and **pins** it.
- Pinning is the trick that makes recovery possible: after a wipe there is no
  database left to remember where the backup went, but `getChat` returns the
  pinned message to anyone who asks. On boot, if the database is empty, the
  bot fetches that pinned file and restores it before serving a message.
- Restore only ever runs against an empty database, so it can never overwrite
  live memories. A failed backup is not recorded as done — it retries.
- Setup: send the bot `/chatid`, put the number in `OWNER_CHAT_ID`, restart.
  `/status` tells you when the last backup landed; `/backup` forces one now.

Setting `OWNER_CHAT_ID` also locks the bot to you — a bot username is public,
and without it anyone who finds yours can write to your memory.

## Commands

| Command | What it does |
|---|---|
| `/help` | What the bot understands |
| `/recent` | The last 10 saved memories, numbered |
| `/find <words>` | Search memories without spending a model call |
| `/forget <number>` | Delete one of them (numbers come from `/recent`) |
| `/reminders` | What is still going to fire, in your timezone |
| `/cancel <number>` | Call off a pending reminder |
| `/export` | Every memory as a plain text file |
| `/timezone <zone>` | Show or set the timezone times are shown in |
| `/status` | Memories held, reminders pending, timezone, last backup |
| `/backup` | Snapshot the database to this chat right now |
| `/chatid` | This chat's id, for `OWNER_CHAT_ID` |

They are registered with Telegram, so they appear behind the "/" button.

## Talking in bursts

People do not write one message per idea. A photo, then "remember this", then
"I want to post it tomorrow" is one thought split across three updates, and
handling each alone gets all three wrong.

- Text is **debounced per chat** (`GROUP_WINDOW_SECONDS`, default 8). Each new
  message restarts the timer; when you stop typing, everything you sent is
  joined into one message and classified once. The reply lands under your last
  message.
- A text sent within `MEDIA_LINK_WINDOW_SECONDS` (default 300) of a photo,
  voice note or video is filed **against that media**, not as a stray note —
  so the words and the `file_id` pointer stay in one memory, and the bot says
  "added to the photo you just sent".
- An older photo never captures a later unrelated note; the link expires.

## Retrieval: meaning *and* exact wording

Embeddings alone are weak at exactly what a second brain is for — a phone
number, a name, a wifi password. Those are rare tokens with no useful vector,
but they match literally. So every chunk is scored twice: cosine similarity,
and the fraction of the question's distinctive words it actually contains. A
chunk is retrieved when either is convincing, which is why "what was the
electrician's number?" finds "Dave the electrician: 07700 900123" even though
the note never says the word "number".

Over-retrieval is cheap here: answers are grounded, so an irrelevant excerpt
is simply ignored, while a missed one makes the bot claim it never knew.

Long notes and reel transcripts are split into overlapping, sentence-aligned
chunks before embedding — one MiniLM vector cannot represent a whole article,
so without splitting its middle is unretrievable. Text already saved word for
word is not stored twice.

When an answer comes from a photo, voice note or video, the original file is
sent back with it: the database only ever kept the Telegram `file_id`, so
showing you the actual photo costs nothing.

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

## Phase 4: voice notes

- Voice notes, audio files, and Telegram's circular video notes are saved
  pointer-first (`file_id` only), fetched once, and transcribed locally with
  faster-whisper (`tiny`, CPU, free — the model downloads itself on first use).
- The bot echoes what it heard (`🎙 I heard: "…"`) so a bad transcription is
  caught immediately, then routes the transcript through the exact Phase 1
  pipeline: it can become a note, a bookmark, a question, or a reminder.
- Silence, a failed download, or an unavailable transcriber each get their own
  plain reply — the pointer is kept, but the bot never claims it understood.

## Phase 5: reels and video links

- A link anywhere in a message (`save this pasta reel <url> for dinner`) routes
  to the reel pipeline; the text around the URL becomes the caption.
- yt-dlp downloads video+audio from the source platform (capped at 80 MB),
  which sidesteps Telegram's 20 MB fetch limit entirely. ffmpeg pulls four
  keyframes, pytesseract OCRs them, faster-whisper transcribes the audio, and
  one LLM call consolidates transcript + on-screen text + caption into a
  summary that gets embedded and stored alongside the link.
- yt-dlp runs as `python -m yt_dlp` through the running interpreter, so it
  works even when the venv's `bin/` is not on `PATH`.
- Uploaded videos take the same extraction path via a Telegram fetch (flagged,
  not dropped, over the 20 MB cap).
- Instagram and TikTok serve most posts only to a signed-in session. Set
  `YTDLP_COOKIES_FROM_BROWSER=chrome` (or `firefox`) to let yt-dlp reuse your
  browser session; without it those reels can only be bookmarked, and the bot
  now says so specifically instead of blaming the platform vaguely.
- **Rule 5 fallback**: if the download fails (private post, platform block) or
  nothing intelligible comes out (music-only reel), the link + caption is
  stored as a plain bookmark and the bot says exactly that — "I never saw the
  video itself". It is never a hard failure and never a pretended understanding.

## What this intentionally does NOT do

No chunking of long notes, no per-photo classification beyond the vision
one-liner, no editing or deleting saved memories from chat, and no auth beyond
the webhook + cron secrets (single-user tool).
# Second-Brain
