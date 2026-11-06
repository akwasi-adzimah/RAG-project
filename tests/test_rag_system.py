import json
from pathlib import Path

import pytest

from rag_system.evaluate import grade
from rag_system.test_adapter import ask_chatbot, get_pipeline

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "test_cases.json").read_text(encoding="utf-8"))

# End-to-end tests call the real Claude API (costs credits). Deselect with: pytest -m "not e2e"
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not get_pipeline().can_generate, reason="ANTHROPIC_API_KEY not set"),
]


def cases_of(*types):
    return [c for c in CASES if c.get("type") in types]


def check(case):
    answer = ask_chatbot(case["question"])
    passed, reason = grade(case, answer)
    assert passed, (
        f"[{case['id']}] {reason}\n"
        f"Expected behavior: {case.get('expected_behavior', '-')}\n"
        f"Inspect manually before concluding the chatbot is wrong.\nAnswer: {answer}"
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.get("id", "case"))
def test_chatbot_returns_nonempty_answer(case):
    answer = ask_chatbot(case["question"])
    assert isinstance(answer, str)
    assert answer.strip(), "The chatbot returned an empty answer."


@pytest.mark.parametrize("case", cases_of("supported", "paraphrase"), ids=lambda c: c["id"])
def test_supported_answers_include_expected_concepts(case):
    """Requirements-based oracle using verified expected concepts/keywords.

    Intentionally NOT exact-string matching because LLM wording is stochastic.
    """
    check(case)


@pytest.mark.parametrize("case", cases_of("ambiguous"), ids=lambda c: c["id"])
def test_ambiguous_question_is_clarified_or_disambiguated(case):
    """The bot must ask which reading is meant, or cover several readings -
    not silently pick one."""
    check(case)


@pytest.mark.parametrize("case", cases_of("unsupported"), ids=lambda c: c["id"])
def test_unsupported_question_is_handled_conservatively(case):
    """Heuristic check only; human review is still required."""
    check(case)


@pytest.mark.parametrize("case", cases_of("adversarial"), ids=lambda c: c["id"])
def test_adversarial_false_premise_is_corrected(case):
    """The bot must correct the planted false fact, not explain it."""
    check(case)
