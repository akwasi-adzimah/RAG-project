"""Central configuration. Every tunable lives here; override via rag_system/.env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

# Shell variables win over .env values.
load_dotenv(PACKAGE_DIR / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


@dataclass(frozen=True)
class Settings:
    # Where the source PDFs/DOCX live, and where derived data is cached.
    docs_dir: Path = field(default_factory=lambda: Path(_env("RAG_DOCS_DIR", str(PROJECT_ROOT))))
    cache_dir: Path = PACKAGE_DIR / ".cache"

    # Chunking: sentence-aware chunks of ~chunk_words, with ~overlap_words of
    # trailing sentences repeated at the start of the next chunk.
    chunk_words: int = 220
    overlap_words: int = 40

    # Retrieval.
    embed_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 6
    candidate_k: int = 30  # per-retriever candidates fed into rank fusion
    rrf_k: int = 60  # reciprocal-rank-fusion damping constant
    rerank: bool = True  # re-score fused candidates with a local cross-encoder
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_k: int = 20  # how many fused candidates the reranker re-scores

    # Generation.
    model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-opus-5"))
    effort: str = field(default_factory=lambda: _env("RAG_EFFORT", "medium"))
    max_tokens: int = 8000

    @property
    def api_key(self) -> str | None:
        return os.environ.get("ANTHROPIC_API_KEY") or None
