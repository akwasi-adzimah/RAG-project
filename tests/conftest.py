"""Shared fixtures: a tiny in-memory corpus, a deterministic fake embedder, and a
fake Anthropic client - so unit tests run offline, fast, and free."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from rag_system.config import Settings
from rag_system.documents import Chunk, Corpus, DocumentInfo
from rag_system.index import HybridIndex, index_text, tokenize
from rag_system.pipeline import RAGPipeline

# Words the fake embedder treats as synonyms, so "semantic" search can be tested.
_SYNONYMS = {"dev": "developer", "devs": "developer", "daily": "day", "everyday": "day"}


class FakeEmbedder:
    """Hashed bag-of-words vectors with a small synonym map."""

    name = "fake-embedder"
    dim = 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            tok = _SYNONYMS.get(tok, tok)
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text):
        return self._vec(text)


CHUNK_TEXTS = [
    ("survey.pdf", 5, "72% of developers who have tried AI use it every day."),
    ("survey.pdf", 8, "Only 43% of developers find AI effective for refactoring code."),
    ("metr.pdf", 1, "Allowing AI increased task completion time by 19% for experienced open-source developers."),
    ("enterprise.pdf", 5, "Weekly Enterprise messages have grown approximately 8x since November 2024."),
    ("teams.pdf", 7, "The Security Engineering team focuses on supply chain security."),
]


@pytest.fixture
def corpus() -> Corpus:
    chunks = [Chunk(id=i, source=s, page=p, text=t) for i, (s, p, t) in enumerate(CHUNK_TEXTS)]
    docs = [DocumentInfo(name=n, pages=10, words=100, chunks=1) for n in sorted({s for s, _, _ in CHUNK_TEXTS})]
    return Corpus(chunks=chunks, documents=docs, skipped=[])


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def index(corpus, embedder) -> HybridIndex:
    embeddings = embedder.embed_documents([index_text(c) for c in corpus.chunks])
    return HybridIndex(corpus, embeddings, embedder, candidate_k=10)


class FakeStream:
    def __init__(self, chunks: list[str], stop_reason: str):
        self.text_stream = iter(chunks)
        self._stop_reason = stop_reason

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return SimpleNamespace(stop_reason=self._stop_reason)


class FakeClient:
    """Records the last request and streams a canned reply."""

    def __init__(self, reply: str = "72% of developers use AI daily [1].", stop_reason: str = "end_turn"):
        self.reply, self.stop_reason, self.calls = reply, stop_reason, []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        words = self.reply.split(" ")
        return FakeStream([w + " " for w in words[:-1]] + words[-1:], self.stop_reason)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(docs_dir=tmp_path, cache_dir=tmp_path / ".cache", model="claude-opus-5",
                    effort="medium", top_k=3)


@pytest.fixture
def make_pipeline(index, settings):
    def _make(client=None) -> RAGPipeline:
        return RAGPipeline(index, settings, client)
    return _make
