"""RAGPipeline: the single entry point used by the app, CLI, and tests."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import anthropic

from .config import Settings
from .embeddings import Embedder, FastEmbedder, FastReranker, Reranker
from .generator import NOT_FOUND, cited_numbers, stream_answer
from .index import HybridIndex, Mode, SearchResult, build_index

NO_RESULTS = f"{NOT_FOUND} No passage in the indexed documents matched this question."
NO_API_KEY = (
    "Retrieval-only mode: no ANTHROPIC_API_KEY is configured, so no answer was generated. "
    "The most relevant passages are listed under Sources."
)


@dataclass
class Answer:
    text: str
    sources: list[SearchResult]
    cited: list[int] = field(default_factory=list)  # 1-based indexes into sources

    @property
    def cited_sources(self) -> list[SearchResult]:
        return [self.sources[n - 1] for n in self.cited]


class RAGPipeline:
    def __init__(self, index: HybridIndex, settings: Settings,
                 client: anthropic.Anthropic | None = None):
        self.index = index
        self.settings = settings
        self.client = client

    @classmethod
    def load(cls, settings: Settings | None = None, *, rebuild: bool = False,
             embedder: Embedder | None = None, reranker: Reranker | None = None,
             client: anthropic.Anthropic | None | bool = True) -> RAGPipeline:
        """Build (or load cached) index. client=True means 'create one if a key is set'."""
        settings = settings or Settings()
        models_dir = settings.cache_dir / "models"
        embedder = embedder or FastEmbedder(settings.embed_model, models_dir)
        if reranker is None and settings.rerank:
            reranker = FastReranker(settings.rerank_model, models_dir)
        index = build_index(
            settings.docs_dir, settings.cache_dir, embedder,
            chunk_words=settings.chunk_words, overlap_words=settings.overlap_words,
            candidate_k=settings.candidate_k, rrf_k=settings.rrf_k,
            reranker=reranker, rerank_k=settings.rerank_k, rebuild=rebuild,
        )
        if client is True:
            client = anthropic.Anthropic(api_key=settings.api_key) if settings.api_key else None
        return cls(index, settings, client or None)

    @property
    def can_generate(self) -> bool:
        return self.client is not None

    # -- retrieval ------------------------------------------------------------

    @staticmethod
    def search_query(question: str, history: Sequence[dict] | None) -> str:
        """Short follow-ups ("what about 2024?") retrieve poorly on their own, so
        fold in the previous user question for retrieval."""
        prior = [m["content"] for m in (history or []) if m.get("role") == "user"]
        if prior and len(question.split()) <= 8:
            return f"{prior[-1]} {question}"
        return question

    def retrieve(self, question: str, *, k: int | None = None, mode: Mode = "hybrid",
                 rerank: bool = True, history: Sequence[dict] | None = None) -> list[SearchResult]:
        question = _validate(question)
        return self.index.search(self.search_query(question, history),
                                 k=k or self.settings.top_k, mode=mode, rerank=rerank)

    # -- generation -----------------------------------------------------------

    def stream(self, question: str, sources: Sequence[SearchResult],
               history: Sequence[dict] | None = None) -> Iterator[str]:
        """Stream the answer for already-retrieved sources."""
        question = _validate(question)
        if not sources:
            yield NO_RESULTS
            return
        if self.client is None:
            yield NO_API_KEY
            return
        yield from stream_answer(
            self.client, model=self.settings.model, effort=self.settings.effort,
            max_tokens=self.settings.max_tokens, question=question,
            results=sources, history=history,
        )

    def ask(self, question: str, *, k: int | None = None, mode: Mode = "hybrid",
            rerank: bool = True, history: Sequence[dict] | None = None) -> Answer:
        sources = self.retrieve(question, k=k, mode=mode, rerank=rerank, history=history)
        text = "".join(self.stream(question, sources, history))
        return Answer(text=text, sources=sources, cited=cited_numbers(text, len(sources)))


def _validate(question: str) -> str:
    if not isinstance(question, str):
        raise TypeError("Question must be a string.")
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    return question
