"""Offline evaluation against test_cases.json.

Retrieval metrics (free, no API calls) for each retrieval configuration:
  hit@k  - share of supported questions with a correct chunk in the top k
  MRR    - mean reciprocal rank of the first correct chunk
A chunk is correct when it contains the case's `evidence` sentence (or, for
cases without one, comes from an expected page of the expected source).

With --answers, also runs full generation (costs API credits) and grades each
answer with `grade()` - the same oracle the e2e pytest suite uses.

    python -m rag_system.evaluate
    python -m rag_system.evaluate --answers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PROJECT_ROOT
from .generator import NOT_FOUND
from .pipeline import RAGPipeline

CASES_PATH = PROJECT_ROOT / "test_cases.json"


# (label, mode, rerank)
CONFIGS = [
    ("hybrid+rerank", "hybrid", True),
    ("hybrid", "hybrid", False),
    ("keyword", "keyword", False),
    ("semantic", "semantic", False),
]


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def is_relevant(result, case: dict) -> bool:
    if result.chunk.source != case.get("expected_source"):
        return False
    if case.get("evidence"):
        return _norm(case["evidence"]) in _norm(result.chunk.text)
    pages = case.get("expected_pages")
    return not pages or result.chunk.page in pages


# Phrases that signal the bot declined rather than answered.
CONSERVATIVE_MARKERS = [
    NOT_FOUND.lower(),
    "not in the provided",
    "not available",
    "cannot determine",
    "don't have enough information",
    "do not have enough information",
    "not found",
    "do not contain",
    "don't contain",
]


def grade(case: dict, answer: str) -> tuple[bool, str]:
    """Heuristic oracle for one answer. Returns (passed, reason).

    Keyword checks, not exact matching, because LLM wording varies. A failure
    means "inspect manually", not proof of hallucination.
    """
    text = _norm(answer)
    kind = case.get("type")
    keywords = [_norm(k) for k in case.get("expected_keywords", []) if k.strip()]
    hits = [k for k in keywords if k in text]
    forbidden = [f for f in (_norm(x) for x in case.get("forbidden_keywords", [])) if f in text]

    if kind in ("supported", "paraphrase"):
        if not keywords:
            return False, f"add verified expected_keywords for {case['id']}"
        return bool(hits), f"expected one of {keywords}"

    if kind == "ambiguous":
        clarified = [m for m in case.get("clarify_markers", []) if _norm(m) in text]
        needed = case.get("min_keyword_hits", 2)
        if clarified:
            return True, f"asked to clarify ({clarified[0]!r})"
        return len(hits) >= needed, f"needs a clarifying question or >= {needed} of {keywords}; found {hits}"

    if kind == "unsupported":
        declined = any(m in text for m in CONSERVATIVE_MARKERS)
        if forbidden:
            return False, f"contains invented-looking content {forbidden}"
        return declined, "expected a conservative / out-of-scope reply"

    if kind == "adversarial":
        if forbidden:
            return False, f"went along with the false premise {forbidden}"
        return bool(hits), f"expected a correction containing one of {keywords}"

    return False, f"unknown case type {kind!r}"


def evaluate_retrieval(pipeline: RAGPipeline, cases: list[dict], k: int) -> dict[str, dict]:
    supported = [c for c in cases if c.get("expected_source")]
    report: dict[str, dict] = {}
    for label, mode, rerank in CONFIGS:
        if rerank and pipeline.index.reranker is None:
            continue
        hits, rr, misses = 0, 0.0, []
        for case in supported:
            results = pipeline.retrieve(case["question"], k=k, mode=mode, rerank=rerank)
            rank = next((i for i, r in enumerate(results, start=1) if is_relevant(r, case)), None)
            if rank:
                hits += 1
                rr += 1 / rank
            else:
                misses.append(case["id"])
        n = len(supported) or 1
        report[label] = {"hit@k": hits / n, "mrr": rr / n, "misses": misses}
    return report


def evaluate_answers(pipeline: RAGPipeline, cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        answer = pipeline.ask(case["question"])
        passed, reason = grade(case, answer.text)
        rows.append({"id": case["id"], "type": case.get("type"), "passed": passed,
                     "reason": reason, "cited": answer.cited, "answer": answer.text})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("-k", type=int, default=6)
    parser.add_argument("--answers", action="store_true", help="also evaluate generated answers (uses the API)")
    args = parser.parse_args(argv)

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    pipeline = RAGPipeline.load()

    print(f"Retrieval ({sum(bool(c.get('expected_source')) for c in cases)} cases with a known source, k={args.k})")
    print(f"  {'config':<15}{'hit@k':>7}{'MRR':>7}  misses")
    for label, r in evaluate_retrieval(pipeline, cases, args.k).items():
        print(f"  {label:<15}{r['hit@k']:>7.0%}{r['mrr']:>7.2f}  {', '.join(r['misses']) or '-'}")

    if args.answers:
        if not pipeline.can_generate:
            print("\n--answers needs ANTHROPIC_API_KEY.", file=sys.stderr)
            return 1
        print(f"\nAnswers ({pipeline.settings.model})")
        rows = evaluate_answers(pipeline, cases)
        for row in rows:
            status = "PASS" if row["passed"] else "FAIL"
            snippet = " ".join(row["answer"].split())[:110]
            print(f"  {status}  {row['id']:<18} cites={row['cited']}  {snippet}")
            if not row["passed"]:
                print(f"        reason: {row['reason']}")
        passed = sum(r["passed"] for r in rows)
        print(f"\n  {passed}/{len(rows)} passed")
        return 0 if passed == len(rows) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
