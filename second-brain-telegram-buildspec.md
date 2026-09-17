# Project Brief: Personal Second Brain over Telegram

## What this is
A personal memory system I chat with over Telegram. I forward it text notes, photos,
voice notes, and reels/video links throughout the day. I can later ask it natural-language
questions ("what was that electrician's number I saved?", "what was that pasta reel I sent
last month?") and it retrieves the actual thing I saved and answers from it — never from
general knowledge, never invented.

Solo, single-user, personal-use tool. Does not need to scale to other users. Needs to be
reliable, cheap to run for free indefinitely, and honest about what it does and doesn't know.

## Hard constraints — everything below must run on free tiers, no exceptions
- Language/stack: Python end to end (FastAPI backend). Every media-processing tool needed
  here (yt-dlp, faster-whisper, pytesseract) is native Python — don't introduce Node/Baileys,
  there's no reason to split languages.
- Interface: Telegram Bot API via `python-telegram-bot`, using **webhook mode** (Telegram
  pushes messages to our HTTPS endpoint). This lets the backend run as a normal stateless
  service — no persistent connection required.
- Hosting: Render (or equivalent) free web service is fine now, since webhook mode doesn't
  need an always-on process the way a websocket-based integration would.
- Reminder scheduling: webhooks are passive — nothing fires on a timer by itself. Use a
  free GitHub Actions scheduled workflow (cron) hitting a `/check-reminders` endpoint every
  minute to actually push due reminders back out.
- Database: Postgres with the pgvector extension, on Neon free tier. Stores structured
  metadata and embeddings ONLY — never raw media files (see storage strategy below).
- Media storage: none. Telegram is the blob store — see storage strategy below.
- Embeddings: a free local embedding model (e.g. all-MiniLM-L6-v2) — never spend the paid
  model's calls generating embeddings.
- OCR: pytesseract (free, local).
- Speech-to-text: faster-whisper (free, local, CPU is fine for short clips).
- Video download: yt-dlp (free, handles Instagram/TikTok/YouTube Shorts links).
- Reasoning/classification/QA/vision model: Google Gemini API, **paid tier only** (not the
  free tier — Gemini's free tier explicitly permits using inputs/files to improve Google's
  products; the paid tier explicitly does not). Use whichever current Flash-tier model is
  live in Google AI Studio (Gemini 2.5/3.x Flash — the mid-cost multimodal tier, not the
  most expensive Pro tier). Budget: a hard $10 spending cap set in Google Cloud console
  from day one, since pay-as-you-go APIs don't stop themselves if something loops. Never
  route any of my data to a model that trains on input data, and never substitute a
  different model for this role without telling me why.

## Storage strategy — read this before writing any media-handling code
Do not store raw photos, videos, or voice notes anywhere outside Telegram. When Telegram
sends us a message with media, it includes a `file_id` — this can be used to re-fetch that
exact file from Telegram's own servers at any time in the future. Store only the pointer
(`chat_id`, `telegram_message_id`, `file_id`) plus whatever text we extracted from the
media (transcript, OCR text, description). Only re-download the actual file on demand
(e.g. if I explicitly ask to see it again), never as a matter of course.

## Data model (Postgres — sketch, adjust as needed but keep the intent)
```sql
messages (
  id, chat_id, telegram_message_id, direction, raw_type,   -- text/photo/video/voice/link
  file_id,                        -- Telegram's own reference; null for plain text
  raw_content_text,               -- transcript/OCR/caption/original text, whatever applies
  received_at, classified_type, processed_at
)

memory_chunks (
  id, source_message_id references messages(id),
  chunk_text, embedding vector(384), tags text[], created_at
)

reminders (
  id, source_message_id references messages(id),
  reminder_text, due_at, fired boolean default false, created_at
)
```

## Critical runtime behavior rules (apply these in every prompt you write for the Gemini API)
1. Never answer a question with information that wasn't actually retrieved from
   `memory_chunks`. If retrieval returns nothing relevant above a similarity threshold,
   reply plainly: "I don't have anything saved about that."
2. When answering, mention roughly when the source memory was saved
   ("you saved this back in July") so I can judge if it's stale.
3. Keep replies Telegram-native: short, conversational, plain text is fine.
4. For reminders, always confirm back exactly what was understood
   ("Got it — reminding you about the electricity bill Friday at 6pm") so a bad
   date-parse gets caught immediately, not silently.
5. For reels, if the download/transcription pipeline fails for any reason, fall back to
   storing the link plus my caption as a lightweight bookmark. Never crash the pipeline
   and never pretend the content was understood when it wasn't — say so if I later ask
   about something that only exists as a bookmark.

## Build phases — build ONE phase at a time. Give me complete, runnable code for that
## phase only, tell me exactly how to test it, and then stop and wait for me to say
## "continue" before starting the next phase.

**Phase 0 — Connection test**
FastAPI project scaffold. Register the Telegram bot via BotFather, wire up the webhook
endpoint, confirm a message I send gets logged and the bot can reply. Postgres/Neon
connection with pgvector enabled, confirmed with a trivial test row.

**Phase 1 — Text ingestion + basic Q&A**
Every incoming text message gets classified by the Gemini API into one of: note, reminder,
question, bookmark (strict JSON-schema prompt, not free text). "Note" gets embedded and
stored in memory_chunks. "Question" triggers retrieval (cosine similarity, top-k) over
memory_chunks and a grounded answer following the runtime rules above.

**Phase 2 — Reminders**
Parse due-dates out of reminder messages (inject today's actual date/time into the prompt
so relative phrases resolve correctly). Store in `reminders`. Build the `/check-reminders`
endpoint the GitHub Actions cron will hit every minute, and push a Telegram message when a
reminder fires.

**Phase 3 — Photos**
Store the file_id (don't download unless needed). Fetch the file once to run OCR via
pytesseract. If the Gemini API's vision capability adds value beyond OCR (no on-screen
text, needs a visual description), get a one-line description from it. Combine and store
as a memory_chunk.

**Phase 4 — Voice notes**
Fetch the file, transcribe locally with faster-whisper, treat the transcript exactly like
an incoming text message from Phase 1 onward.

**Phase 5 — Reels/video links (build the real pipeline, not a stub)**
If the message is a link: download with yt-dlp directly from the source platform (bypasses
Telegram's file-size limits entirely). If it's an uploaded video file: fetch via Telegram
(note the 20MB standard Bot API cap — flag it to me if a file exceeds this rather than
failing silently). Then: extract audio + a handful of keyframes with ffmpeg, transcribe
the audio with faster-whisper, OCR the keyframes with pytesseract, get a one-line visual
description of a keyframe or two from the Gemini API, and combine transcript + OCR text +
visual description + original caption into one consolidated summary via the Gemini API.
Embed and store that summary as the memory_chunk. If the download step fails for any
reason (platform blocked the scraper, etc.), fall back to storing the link + my caption
as a bookmark, per runtime rule 5 — do not let this phase become a hard failure point.

## Your first task
Start with Phase 0 only. Ask me anything you need to know about my environment
(Python version, whether I already have a Neon project, whether I've registered a bot
with BotFather yet) before writing code.
