# test_adapter.py
"""Stable interface used by tests/, performance_test.py and stress_test.py."""

import sys
from functools import lru_cache
from pathlib import Path

if __package__ in (None, ""):  # allow `python rag_system/test_adapter.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_system.index import SearchResult
from rag_system.pipeline import RAGPipeline


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """
    Build (or load the cached) index once and reuse it for all tests.
    """
    return RAGPipeline.load()


def retrieve(question: str, top_k: int = 6) -> list[SearchResult]:
    """
    Test only the retrieval component of the RAG system.
    """
    return get_pipeline().retrieve(question, k=top_k)


def ask_chatbot(question: str, top_k: int = 6) -> str:
    """
    Run the complete RAG pipeline:
    retrieval + answer generation.
    """
    return get_pipeline().ask(question, k=top_k).text


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What percentage of developers use AI coding tools daily?"
    print(ask_chatbot(q))
