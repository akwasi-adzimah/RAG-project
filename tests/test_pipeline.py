"""Pipeline + generator behaviour, using a fake Claude client (no API calls)."""
import pytest

from rag_system.generator import (NOT_FOUND, build_messages, cited_numbers,
                                  format_excerpts)
from rag_system.pipeline import NO_API_KEY, NO_RESULTS, RAGPipeline

from .conftest import FakeClient


def test_ask_streams_answer_and_parses_citations(make_pipeline):
    client = FakeClient("72% use it every day [1], see also [1] and [9].")
    answer = make_pipeline(client).ask("How many developers use AI every day?")

    assert answer.text.startswith("72% use it every day [1]")
    assert answer.cited == [1]  # deduplicated; [9] is out of range
    assert answer.cited_sources[0].chunk.source == "survey.pdf"


def test_request_shape(make_pipeline):
    client = FakeClient()
    make_pipeline(client).ask("How many developers use AI every day?")
    call = client.calls[0]

    assert call["model"] == "claude-opus-5"
    assert call["output_config"] == {"effort": "medium"}
    assert call["fallbacks"] == "default"
    assert NOT_FOUND in call["system"]
    assert call["messages"][-1]["role"] == "user"
    assert '<excerpt id="1" source="survey.pdf" page="5">' in call["messages"][-1]["content"]


def test_no_fallbacks_for_models_without_support(make_pipeline, settings):
    from dataclasses import replace
    client = FakeClient()
    pipeline = make_pipeline(client)
    pipeline.settings = replace(settings, model="claude-sonnet-5")
    pipeline.ask("developers AI every day")
    assert "fallbacks" not in client.calls[0] and "betas" not in client.calls[0]


def test_retrieval_only_mode_without_client(make_pipeline):
    answer = make_pipeline(None).ask("How many developers use AI every day?")
    assert answer.text == NO_API_KEY
    assert answer.sources  # still returns the passages
    assert answer.cited == []


def test_no_matching_passages_skips_the_api(make_pipeline):
    client = FakeClient()
    answer = make_pipeline(client).ask("zzzz qqqq", mode="keyword")
    assert answer.text == NO_RESULTS
    assert client.calls == []


def test_refusal_and_truncation_are_surfaced(make_pipeline):
    refused = make_pipeline(FakeClient("", stop_reason="refusal")).ask("developers AI")
    assert "declined" in refused.text
    truncated = make_pipeline(FakeClient("Partial", stop_reason="max_tokens")).ask("developers AI")
    assert "truncated" in truncated.text


@pytest.mark.parametrize("bad, exc", [("", ValueError), ("   ", ValueError), (None, TypeError), (42, TypeError)])
def test_invalid_questions_rejected(make_pipeline, bad, exc):
    with pytest.raises(exc):
        make_pipeline(FakeClient()).ask(bad)


def test_history_is_sent_but_excerpts_only_on_latest_turn(index):
    results = index.search("security", k=1)
    history = [
        {"role": "assistant", "content": "orphan greeting"},  # dropped: must start with user
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]
    messages = build_messages("follow-up", results, history)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert "<excerpt" not in messages[0]["content"]
    assert "<excerpt" in messages[-1]["content"] and "Question: follow-up" in messages[-1]["content"]


def test_short_follow_ups_include_previous_question():
    history = [{"role": "user", "content": "What does the METR study find?"}]
    assert RAGPipeline.search_query("and for juniors?", history) == "What does the METR study find? and for juniors?"
    long_q = "What share of enterprise messages go through custom GPTs or projects?"
    assert RAGPipeline.search_query(long_q, history) == long_q


def test_format_excerpts_escapes_attributes(index):
    from dataclasses import replace
    r = index.search("security", k=1)[0]
    r = replace(r, chunk=replace(r.chunk, source='a"b.pdf', page=None))
    out = format_excerpts([r])
    assert 'source="a&quot;b.pdf"' in out and "page=" not in out


def test_cited_numbers_order():
    assert cited_numbers("x [3] y [1][3] z [2]", 3) == [3, 1, 2]


# --- the e2e answer oracle (rag_system.evaluate.grade) -----------------------

from rag_system.evaluate import grade  # noqa: E402

AMBIGUOUS = {"id": "a", "type": "ambiguous", "min_keyword_hits": 2,
             "expected_keywords": ["legal", "inference", "growth marketing"],
             "clarify_markers": ["which team"]}
ADVERSARIAL = {"id": "x", "type": "adversarial", "expected_keywords": ["slower"],
               "forbidden_keywords": ["the speedup is due to"]}
UNSUPPORTED = {"id": "u", "type": "unsupported", "forbidden_keywords": ["€"]}


@pytest.mark.parametrize("case, answer, expected", [
    ({"id": "s", "type": "supported", "expected_keywords": ["72%"]}, "About 72% [1].", True),
    ({"id": "s", "type": "paraphrase", "expected_keywords": ["72%"]}, "Most of them.", False),
    (AMBIGUOUS, "Which team do you mean?", True),
    (AMBIGUOUS, "The Legal team and the Inference team both...", True),
    (AMBIGUOUS, "The Legal team uses it for...", False),
    (UNSUPPORTED, f"{NOT_FOUND} The excerpts cover adoption.", True),
    (UNSUPPORTED, "Senior engineers earn €85,000.", False),
    (UNSUPPORTED, f"{NOT_FOUND} Typically €85,000.", False),
    (ADVERSARIAL, "Actually the study found developers were 19% slower [1].", True),
    (ADVERSARIAL, "The speedup is due to better tooling.", False),
])
def test_grade(case, answer, expected):
    assert grade(case, answer)[0] is expected
