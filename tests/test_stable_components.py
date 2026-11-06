"""Stable components of the RAG pipeline, tested against the real documents.

Everything here is deterministic: PDF extraction, chunking, the pinned local
embedding and reranker models, retrieval and prompt construction. No Claude
call is made (the pipeline is loaded without a client), so these tests are
free, repeatable and run in CI. The LLM's wording is only checked in the e2e
suite (tests/test_rag_system.py).
"""
import json
from functools import cache
from pathlib import Path

import pytest

from rag_system.config import Settings
from rag_system.evaluate import _norm, is_relevant
from rag_system.generator import NOT_FOUND, SYSTEM_PROMPT, build_messages
from rag_system.pipeline import NO_API_KEY, RAGPipeline

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "test_cases.json").read_text(encoding="utf-8"))
TYPES = ("supported", "paraphrase", "ambiguous", "unsupported", "adversarial")
SCANNED_PDF = "AI Impact on Software Engineers 2026.pdf"


def cases_of(*types):
    return [c for c in CASES if c.get("type") in types]


GROUNDED = [c for c in CASES if c.get("evidence")]


@cache
def pipeline() -> RAGPipeline:
    return RAGPipeline.load(client=None)  # retrieval only: never calls the API


@pytest.fixture(scope="module")
def rag() -> RAGPipeline:
    return pipeline()


# --- the test case set itself -------------------------------------------------

def test_case_set_meets_coverage_requirements():
    """4-5 supported, 2 paraphrases, and at least 1 each of the risk categories."""
    counts = {t: len(cases_of(t)) for t in TYPES}
    assert len(CASES) >= 10, counts
    assert 4 <= counts["supported"] <= 5, counts
    assert counts["paraphrase"] == 2, counts
    assert all(counts[t] >= 1 for t in ("ambiguous", "unsupported", "adversarial")), counts
    ids = {c["id"] for c in CASES}
    assert all(c["paraphrase_of"] in ids for c in cases_of("paraphrase"))


def test_cases_are_well_formed():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    for c in CASES:
        assert c["type"] in TYPES, c["id"]
        assert c["question"].strip() and c["expected_behavior"].strip(), c["id"]
        if c["type"] in ("supported", "paraphrase"):
            assert c.get("expected_keywords") and c.get("evidence"), c["id"]


@pytest.mark.parametrize("case", GROUNDED, ids=lambda c: c["id"])
def test_evidence_exists_on_the_expected_page(rag, case):
    """The oracle is only as good as its expected answer: confirm each evidence
    sentence really is in the named document, on the named page."""
    text = " ".join(
        ch.text for ch in rag.index.chunks
        if ch.source == case["expected_source"] and ch.page in case["expected_pages"]
    )
    assert _norm(case["evidence"]) in _norm(text), (
        f"[{case['id']}] evidence not found in {case['expected_source']} "
        f"p.{case['expected_pages']}"
    )


# --- extraction and chunking ----------------------------------------------------

def test_corpus_loads_every_text_document_and_skips_the_scanned_pdf(rag):
    corpus = rag.index.corpus
    loaded = {d.name for d in corpus.documents}
    assert loaded == {
        "state-of-code-developer-survey-report.pdf",
        "MEASUR~1.PDF",
        "the-state-of-enterprise-ai_2025-report.pdf",
        "How-Anthropic-teams-use-Claude-Code_v2pdf.pdf",
    } | ({"Severe_Economic_Hardship_Statement.docx"} & loaded)
    assert corpus.skipped == [SCANNED_PDF]


def test_chunks_are_valid(rag):
    settings = Settings()
    pages = {d.name: d.pages for d in rag.index.corpus.documents}
    ids = [ch.id for ch in rag.index.chunks]
    assert ids == list(range(len(ids))), "chunk ids must be 0..n-1 (they index the embeddings)"
    for ch in rag.index.chunks:
        assert ch.text.strip(), ch.id
        assert len(ch.text.split()) <= settings.chunk_words, ch.id
        if ch.source.lower().endswith(".pdf"):
            assert 1 <= ch.page <= pages[ch.source], (ch.id, ch.page)


def test_embeddings_match_chunks(rag):
    embeddings = rag.index.embeddings
    assert embeddings.shape[0] == len(rag.index.chunks)
    norms = (embeddings ** 2).sum(axis=1) ** 0.5
    assert abs(norms.min() - 1) < 1e-3 and abs(norms.max() - 1) < 1e-3, "embeddings must be normalized"


# --- retrieval --------------------------------------------------------------------

@pytest.mark.parametrize("case", GROUNDED, ids=lambda c: c["id"])
def test_evidence_is_retrieved_in_top_k(rag, case):
    """Regression guard for retrieval quality (hybrid + rerank, the default)."""
    results = rag.retrieve(case["question"])
    ranks = [i for i, r in enumerate(results, 1) if is_relevant(r, case)]
    assert ranks, (
        f"[{case['id']}] evidence passage not in top {len(results)}. Retrieved: "
        + ", ".join(r.chunk.label for r in results)
    )


@pytest.mark.parametrize("mode", ["hybrid", "keyword", "semantic"])
def test_retrieval_is_deterministic(rag, mode):
    question = CASES[0]["question"]
    first = [(r.chunk.id, round(r.score, 6)) for r in rag.retrieve(question, mode=mode)]
    second = [(r.chunk.id, round(r.score, 6)) for r in rag.retrieve(question, mode=mode)]
    assert first == second


def test_results_are_ranked_and_reranked(rag):
    results = rag.retrieve(CASES[0]["question"])
    assert len(results) == Settings().top_k
    assert all(r.reranked for r in results)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert len({r.chunk.id for r in results}) == len(results), "no duplicate passages"


# --- prompt construction and the no-key path -------------------------------------

def test_prompt_carries_retrieved_evidence_with_its_citation(rag):
    case = cases_of("supported")[0]
    results = rag.retrieve(case["question"])
    prompt = build_messages(case["question"], results)[-1]["content"]

    assert prompt.count("<excerpt ") == len(results)
    assert prompt.rstrip().endswith(f"Question: {case['question']}")
    assert f'source="{case["expected_source"]}"' in prompt
    assert _norm(case["evidence"]) in _norm(prompt)
    assert NOT_FOUND in SYSTEM_PROMPT


def test_pipeline_without_key_returns_sources_but_no_generated_answer(rag):
    answer = rag.ask(CASES[0]["question"])
    assert answer.text == NO_API_KEY
    assert len(answer.sources) == Settings().top_k
    assert answer.cited == []
