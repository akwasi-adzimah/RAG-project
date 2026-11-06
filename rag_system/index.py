"""Hybrid retrieval: BM25 keyword search + dense semantic search, merged with
reciprocal rank fusion (RRF).

BM25 is precise for exact terms, names and numbers; embeddings catch
paraphrases ("how often do devs use AI" vs "use them every day"). RRF merges
the two rankings without needing their scores to be on the same scale.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from .documents import Chunk, Corpus, DocumentInfo, load_corpus, source_files
from .embeddings import Embedder, Reranker

log = logging.getLogger(__name__)

Mode = Literal["hybrid", "keyword", "semantic"]
MODES: tuple[Mode, ...] = ("hybrid", "keyword", "semantic")

INDEX_VERSION = 4  # bump when chunking/cleaning logic changes

_TOKEN = re.compile(r"[a-z0-9]+(?:[.%][0-9]+)?%?")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its of on or "  # noqa: SIM905
    "that the their them there these they this to was were what when where which who why "
    "will with you your".split()
)


def stem(token: str) -> str:
    """Tiny suffix stripper so 'focuses'/'focused'/'focusing' all match 'focus'."""
    for suffix in ("ing", "ies", "es", "ed", "ly", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            base = token[: -len(suffix)]
            return base + "y" if suffix == "ies" else base
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def index_text(chunk: Chunk) -> str:
    """What gets indexed: document name + page opening (heading) + chunk text.
    The extra context lets a mid-page chunk match queries about the page's topic."""
    context = "" if chunk.text.startswith(chunk.context) else f"{chunk.context} ... "
    return f"{Path(chunk.source).stem}: {context}{chunk.text}"


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    keyword_rank: int | None  # 1-based rank in BM25 results, None if not a candidate
    semantic_rank: int | None
    reranked: bool = False  # True when `score` is the cross-encoder relevance score

    def to_dict(self) -> dict:
        return {**self.chunk.to_dict(), "score": self.score, "keyword_rank": self.keyword_rank,
                "semantic_rank": self.semantic_rank, "reranked": self.reranked}


class HybridIndex:
    def __init__(self, corpus: Corpus, embeddings: np.ndarray, embedder: Embedder,
                 candidate_k: int = 30, rrf_k: int = 60, reranker: Reranker | None = None,
                 rerank_k: int = 20):
        if len(corpus.chunks) != len(embeddings):
            raise ValueError("chunks and embeddings are out of sync")
        self.corpus = corpus
        self.chunks = corpus.chunks
        self.embeddings = embeddings
        self.embedder = embedder
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.reranker = reranker
        self.rerank_k = rerank_k
        self._bm25 = BM25Okapi([tokenize(index_text(c)) for c in self.chunks]) if self.chunks else None

    # -- individual retrievers ------------------------------------------------

    def keyword_ranking(self, query: str, limit: int) -> list[tuple[int, float]]:
        terms = tokenize(query)
        if not terms or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(terms)
        order = np.argsort(-scores)[:limit]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def semantic_ranking(self, query: str, limit: int) -> list[tuple[int, float]]:
        if not len(self.embeddings):
            return []
        sims = self.embeddings @ self.embedder.embed_query(query)
        order = np.argsort(-sims)[:limit]
        return [(int(i), float(sims[i])) for i in order]

    # -- public API -----------------------------------------------------------

    def search(self, query: str, k: int = 6, mode: Mode = "hybrid",
               rerank: bool = True) -> list[SearchResult]:
        """Top-k chunks. With rerank (and a reranker configured), the top
        rerank_k candidates are re-scored by the cross-encoder before cutting to k."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        query = query.strip()
        if not query:
            return []

        depth = max(k, self.candidate_k)
        keyword = self.keyword_ranking(query, depth) if mode != "semantic" else []
        semantic = self.semantic_ranking(query, depth) if mode != "keyword" else []
        kw_rank = {idx: r for r, (idx, _) in enumerate(keyword, start=1)}
        sem_rank = {idx: r for r, (idx, _) in enumerate(semantic, start=1)}

        if mode == "keyword":
            scored = keyword
        elif mode == "semantic":
            scored = semantic
        else:
            fused: dict[int, float] = {}
            for ranks in (kw_rank, sem_rank):
                for idx, r in ranks.items():
                    fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + r)
            scored = sorted(fused.items(), key=lambda item: item[1], reverse=True)

        use_reranker = rerank and self.reranker is not None and len(scored) > 1
        if use_reranker:
            pool = scored[:max(k, self.rerank_k)]
            scores = self.reranker.score(query, [index_text(self.chunks[i]) for i, _ in pool])
            scored = sorted(zip((i for i, _ in pool), scores), key=lambda item: item[1], reverse=True)

        return [
            SearchResult(self.chunks[idx], score, kw_rank.get(idx), sem_rank.get(idx), use_reranker)
            for idx, score in scored[:k]
        ]


# --------------------------------------------------------------------------- #
# Building + on-disk cache
# --------------------------------------------------------------------------- #

def _fingerprint(docs_dir: Path, chunk_words: int, overlap_words: int, embed_model: str) -> str:
    files = [
        {"name": p.name, "size": p.stat().st_size, "mtime": p.stat().st_mtime}
        for p in source_files(docs_dir)
    ]
    payload = {"v": INDEX_VERSION, "files": files, "chunk_words": chunk_words,
               "overlap_words": overlap_words, "embed_model": embed_model}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def build_index(docs_dir: Path, cache_dir: Path, embedder: Embedder, *,
                chunk_words: int, overlap_words: int, candidate_k: int = 30,
                rrf_k: int = 60, reranker: Reranker | None = None, rerank_k: int = 20,
                rebuild: bool = False) -> HybridIndex:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "index.json"
    vec_path = cache_dir / "embeddings.npy"
    key = _fingerprint(docs_dir, chunk_words, overlap_words, embedder.name)

    if not rebuild and meta_path.exists() and vec_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("key") == key:
            corpus = Corpus(
                chunks=[Chunk(**c) for c in meta["chunks"]],
                documents=[DocumentInfo(**d) for d in meta["documents"]],
                skipped=meta["skipped"],
            )
            return HybridIndex(corpus, np.load(vec_path), embedder, candidate_k, rrf_k,
                               reranker, rerank_k)

    log.info("Building index from %s", docs_dir)
    corpus = load_corpus(docs_dir, chunk_words, overlap_words)
    texts = [index_text(c) for c in corpus.chunks]
    embeddings = embedder.embed_documents(texts) if texts else np.zeros((0, 1), np.float32)

    np.save(vec_path, embeddings)
    meta_path.write_text(json.dumps({
        "key": key,
        "chunks": [c.to_dict() for c in corpus.chunks],
        "documents": [asdict(d) for d in corpus.documents],
        "skipped": corpus.skipped,
    }), encoding="utf-8")
    return HybridIndex(corpus, embeddings, embedder, candidate_k, rrf_k, reranker, rerank_k)
