"""Grounded answer generation with Claude."""
from __future__ import annotations

import re
from html import escape
from typing import Iterator, Sequence

import anthropic

from .index import SearchResult

NOT_FOUND = "The answer is not in the provided documents."

SYSTEM_PROMPT = f"""You answer questions about a private collection of reports, using only \
the numbered excerpts supplied with each question.

How to answer:
- Base every factual statement on the excerpts and cite it with the excerpt number in \
square brackets, e.g. [2] or [1][3]. Only cite numbers that exist.
- Lead with the direct answer, then add supporting detail. Keep it concise; use short \
bullet lists when comparing several figures or items.
- Quote figures exactly as written (percentages, counts, dates) and say which report or \
survey they come from when that matters.
- If the excerpts only partly answer the question, answer the part they support and say \
what is missing.
- If the excerpts do not answer the question at all, reply with exactly: "{NOT_FOUND}" \
Then, in one sentence, mention the closest related topic the excerpts do cover, if any.
- Never use outside knowledge or guess. Earlier turns are for context only; answer the \
latest question from its excerpts."""

# Models that accept server-side refusal fallbacks (`fallbacks: "default"`).
_FALLBACK_MODELS = {"claude-opus-5", "claude-opus-5-5", "claude-fable-5", "claude-fable-5-1"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_CITATION = re.compile(r"\[(\d+)\]")


def format_excerpts(results: Sequence[SearchResult]) -> str:
    blocks = []
    for n, r in enumerate(results, start=1):
        page = f' page="{r.chunk.page}"' if r.chunk.page else ""
        blocks.append(
            f'<excerpt id="{n}" source="{escape(r.chunk.source)}"{page}>\n'
            f"{r.chunk.text}\n</excerpt>"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, results: Sequence[SearchResult],
                   history: Sequence[dict] | None = None) -> list[dict]:
    """History is plain {role, content} turns; only the latest turn carries excerpts."""
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in (history or [])
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    # The API requires the conversation to start with a user turn.
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    messages.append({
        "role": "user",
        "content": f"<excerpts>\n{format_excerpts(results)}\n</excerpts>\n\nQuestion: {question}",
    })
    return messages


def cited_numbers(text: str, n_sources: int) -> list[int]:
    """1-based excerpt numbers cited in the answer, in first-cited order."""
    seen: list[int] = []
    for match in _CITATION.finditer(text):
        n = int(match.group(1))
        if 1 <= n <= n_sources and n not in seen:
            seen.append(n)
    return seen


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, anthropic.AuthenticationError):
        return "The Anthropic API key was rejected. Check ANTHROPIC_API_KEY in rag_system/.env."
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "This API key doesn't have access to the configured model."
    if isinstance(exc, anthropic.NotFoundError):
        return "The configured model was not found. Check ANTHROPIC_MODEL in rag_system/.env."
    if isinstance(exc, anthropic.RateLimitError):
        return "Rate limited by the Anthropic API. Wait a moment and try again."
    if isinstance(exc, anthropic.BadRequestError):
        return f"The API rejected the request: {exc.message}"
    if isinstance(exc, anthropic.APIStatusError):
        return f"Anthropic API error ({exc.status_code}). Try again shortly."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Couldn't reach the Anthropic API. Check your internet connection."
    return f"Unexpected error: {exc}"


def stream_answer(client: anthropic.Anthropic, *, model: str, effort: str, max_tokens: int,
                  question: str, results: Sequence[SearchResult],
                  history: Sequence[dict] | None = None) -> Iterator[str]:
    """Yield answer text as it streams. Raises anthropic errors to the caller."""
    kwargs: dict = {}
    if model in _FALLBACK_MODELS:
        kwargs = {"betas": [_FALLBACK_BETA], "fallbacks": "default"}

    with client.beta.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=build_messages(question, results, history),
        output_config={"effort": effort},
        **kwargs,
    ) as stream:
        yield from stream.text_stream
        final = stream.get_final_message()

    if final.stop_reason == "refusal":
        yield "\n\n_This request was declined by the model's safety system._"
    elif final.stop_reason == "max_tokens":
        yield "\n\n_(Answer truncated - it hit the length limit.)_"
