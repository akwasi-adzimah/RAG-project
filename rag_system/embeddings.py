"""Local text embeddings (no API key, runs on CPU via ONNX)."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


def _load(cls, model_name: str, cache_dir: Path):
    try:  # skip the Hugging Face network check once the model is on disk
        return cls(model_name, cache_dir=str(cache_dir), local_files_only=True)
    except Exception:
        return cls(model_name, cache_dir=str(cache_dir))


class FastEmbedder:
    """Wraps fastembed. The model (~70 MB) downloads once into cache_dir."""

    def __init__(self, model_name: str, cache_dir: Path):
        from fastembed import TextEmbedding  # heavy import; defer until needed

        self.name = model_name
        self._model = _load(TextEmbedding, model_name, cache_dir)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.array(list(self._model.passage_embed(list(texts))), dtype=np.float32)
        return _normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.array(next(iter(self._model.query_embed(text))), dtype=np.float32)
        return _normalize(vector)


class FastReranker:
    """Cross-encoder that reads (query, passage) together - slower than
    embeddings, so it only re-scores the top candidates. ~80 MB, downloads once."""

    def __init__(self, model_name: str, cache_dir: Path):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model_name
        self._model = _load(TextCrossEncoder, model_name, cache_dir)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, list(texts))]
