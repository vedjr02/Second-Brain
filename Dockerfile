FROM python:3.12-slim

# System deps: tesseract (OCR for photos/reels) + ffmpeg (audio/keyframe extraction)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY scripts/ scripts/

# WEBHOOK_BASE_URL must be set in the environment (Render service URL).
# Shell form so ${PORT} expands (Render injects it at runtime).
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
