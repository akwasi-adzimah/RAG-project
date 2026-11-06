"""Hybrid retrieval behaviour and the on-disk index cache."""
import pytest
from docx import Document

from rag_system.index import build_index, stem, tokenize


def test_tokenize_drops_stopwords_and_keeps_percentages():
    assert tokenize("What is the 72% of developers?") == ["72%", "developer"]


@pytest.mark.parametrize("word, expected", [
    ("focuses", "focus"), ("focused", "focus"), ("focusing", "focus"),
    ("studies", "study"), ("is", "is"), ("gas", "gas"),
])
def test_stem(word, expected):
    assert stem(word) == expected


def test_keyword_search_finds_exact_figure(index):
    top = index.search("19% completion time", k=1, mode="keyword")[0]
    assert top.chunk.source == "metr.pdf"
    assert top.keyword_rank == 1 and top.semantic_rank is None


def test_semantic_search_matches_synonyms(index):
    # "devs"/"daily" never appear in the corpus; only the embedder knows the synonyms.
    assert index.search("devs daily", k=3, mode="keyword") == []
    top = index.search("devs daily", k=1, mode="semantic")[0]
    assert top.chunk.page == 5 and top.chunk.source == "survey.pdf"


def test_hybrid_fuses_both_rankings(index):
    results = index.search("security engineering supply chain", k=3, mode="hybrid")
    assert results[0].chunk.source == "teams.pdf"
    assert results[0].keyword_rank and results[0].semantic_rank
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_k_and_blank_queries(index):
    assert len(index.search("developers AI", k=2)) == 2
    assert index.search("   ") == []


def test_invalid_mode_raises(index):
    with pytest.raises(ValueError):
        index.search("anything", mode="fuzzy")


def test_build_index_uses_cache_until_files_change(tmp_path, embedder):
    docs, cache = tmp_path / "docs", tmp_path / "cache"
    docs.mkdir()
    doc = Document()
    doc.add_paragraph("Original content about quarterly revenue.")
    doc.save(docs / "a.docx")

    calls = []
    original = embedder.embed_documents
    embedder.embed_documents = lambda texts: calls.append(len(texts)) or original(texts)
    kwargs = {"chunk_words": 200, "overlap_words": 40}

    first = build_index(docs, cache, embedder, **kwargs)
    second = build_index(docs, cache, embedder, **kwargs)
    assert len(calls) == 1, "second build should come from cache"
    assert second.chunks == first.chunks

    doc.add_paragraph("New paragraph about headcount.")
    doc.save(docs / "a.docx")
    third = build_index(docs, cache, embedder, **kwargs)
    assert len(calls) == 2, "changed file should trigger a rebuild"
    assert "headcount" in third.chunks[0].text


class ReverseReranker:
    """Scores passages by how late they appear - deliberately reorders results."""
    name = "reverse"

    def __init__(self):
        self.calls = []

    def score(self, query, texts):
        self.calls.append(len(texts))
        return [float(i) for i in range(len(texts))]


def test_reranker_reorders_candidates(index):
    baseline = index.search("developers AI", k=5, rerank=False)
    index.reranker = ReverseReranker()
    reranked = index.search("developers AI", k=5)

    assert all(r.reranked for r in reranked)
    assert [r.chunk.id for r in reranked] == [r.chunk.id for r in reversed(baseline)]
    assert index.search("developers AI", k=5, rerank=False) == baseline  # opt-out works


def test_reranker_only_sees_top_candidates(index):
    index.reranker = ReverseReranker()
    index.rerank_k = 2
    results = index.search("developers AI", k=2)
    assert index.reranker.calls == [2] and len(results) == 2
