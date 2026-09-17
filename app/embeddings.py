"""Local text embeddings via fastembed (all-MiniLM-L6-v2, 384 dims, L2-normed).

Runs fully offline on CPU (ONNX) — no per-call API cost, matching the Phase 1
decision in the build spec. The model is loaded lazily as a process-wide
singleton and shared across requests.
"""

import logging
import os
import threading

from fastembed import TextEmbedding

from .settings import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Must be set before tokenizers spawns threads; avoids fork warnings in servers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_LOCK = threading.Lock()
_MODEL: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                logger.info("loading embedding model %s", EMBEDDING_MODEL)
                _MODEL = TextEmbedding(EMBEDDING_MODEL)
    return _MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts into 384-dim L2-normalized vectors, preserving order."""
    if not texts:
        return []
    return [vec.tolist() for vec in _get_model().embed(texts)]
