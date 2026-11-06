"""Streamlit chat UI.   Run:  streamlit run rag_system/app.py"""
from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic
import streamlit as st

from rag_system.config import PROJECT_ROOT
from rag_system.generator import cited_numbers, friendly_error
from rag_system.index import SearchResult
from rag_system.pipeline import RAGPipeline

st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="centered")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.5rem; max-width: 820px; }
      .src-head { font-size: 0.9rem; font-weight: 600; margin-bottom: 0.15rem; }
      .src-meta { font-size: 0.75rem; opacity: 0.65; }
      .pill { display: inline-block; font-size: 0.7rem; padding: 0.05rem 0.45rem;
              border-radius: 999px; background: rgba(46, 160, 67, 0.15);
              color: rgb(46, 160, 67); margin-left: 0.35rem; vertical-align: middle; }
      .doc-row { font-size: 0.85rem; line-height: 1.25; margin-bottom: 0.55rem; }
      .doc-row small { opacity: 0.65; }
    </style>
    """,
    unsafe_allow_html=True,
)

MODE_LABELS = {
    "hybrid": "Hybrid (recommended)",
    "keyword": "Keyword (BM25)",
    "semantic": "Semantic (embeddings)",
}


@st.cache_resource(show_spinner="Indexing documents - the first run downloads a ~70 MB embedding model…")
def load_pipeline() -> RAGPipeline:
    return RAGPipeline.load()


def example_questions() -> list[str]:
    try:
        cases = json.loads((PROJECT_ROOT / "test_cases.json").read_text(encoding="utf-8"))
        return [c["question"] for c in cases if c.get("type") == "supported"][:4]
    except (OSError, ValueError, KeyError):
        return []


def log_feedback(pipeline: RAGPipeline, msg: dict) -> None:
    rating = st.session_state.get(f"fb_{msg['id']}")
    if rating is None:
        return
    record = {
        "time": datetime.now(UTC).isoformat(),
        "rating": "up" if rating == 1 else "down",
        "question": msg["question"],
        "answer": msg["content"],
        "sources": [r.chunk.label for r in msg["sources"]],
        "model": pipeline.settings.model,
    }
    path = pipeline.settings.cache_dir / "feedback.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def render_sources(msg: dict, show_all: bool) -> None:
    sources: list[SearchResult] = msg.get("sources") or []
    if not sources:
        return
    cited = msg.get("cited") or []
    items = list(enumerate(sources, start=1))
    if cited and not show_all:
        items = [(n, r) for n, r in items if n in cited]
    title = f"Sources · {len(cited)} cited" if cited else f"Retrieved passages · {len(sources)}"

    with st.expander(title):
        for n, r in items:
            with st.container(border=True):
                pill = '<span class="pill">cited</span>' if n in cited else ""
                page = f" · page {r.chunk.page}" if r.chunk.page else ""
                ranks = []
                if r.reranked:
                    ranks.append(f"relevance {r.score:.1f}")
                if r.keyword_rank:
                    ranks.append(f"keyword #{r.keyword_rank}")
                if r.semantic_rank:
                    ranks.append(f"semantic #{r.semantic_rank}")
                st.markdown(
                    f'<div class="src-head">[{n}] {r.chunk.source}{page}{pill}</div>'
                    f'<div class="src-meta">{" · ".join(ranks)}</div>',
                    unsafe_allow_html=True,
                )
                text = r.chunk.text
                st.caption(text if len(text) <= 700 else text[:700] + "…")


def render_message(pipeline: RAGPipeline, msg: dict, show_all: bool) -> None:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📚"):
        st.markdown(msg["content"])
        if msg["role"] != "assistant":
            return
        if msg.get("meta"):
            st.caption(msg["meta"])
        render_sources(msg, show_all)
        if pipeline.can_generate and msg.get("generated"):
            st.feedback("thumbs", key=f"fb_{msg['id']}", on_change=log_feedback, args=(pipeline, msg))


# --------------------------------------------------------------------------- #

pipeline = load_pipeline()
corpus = pipeline.index.corpus
state = st.session_state
state.setdefault("messages", [])

with st.sidebar:
    st.header("📚 Library")
    if pipeline.can_generate:
        st.success(f"Claude connected · `{pipeline.settings.model}`", icon="✅")
    else:
        st.warning("Retrieval-only mode. Add `ANTHROPIC_API_KEY` to `rag_system/.env` "
                   "to get written answers.", icon="⚠️")

    for d in corpus.documents:
        st.markdown(
            f'<div class="doc-row">{d.name}<br><small>{d.pages} pages · {d.chunks} passages</small></div>',
            unsafe_allow_html=True,
        )
    for name in corpus.skipped:
        st.markdown(
            f'<div class="doc-row">⚠️ {name}<br><small>skipped - scanned PDF, needs OCR</small></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Settings")
    mode = st.radio("Retrieval", list(MODE_LABELS), format_func=MODE_LABELS.get,
                    help="Hybrid combines keyword matching with meaning-based search.")
    rerank = st.toggle("Rerank results", value=pipeline.index.reranker is not None,
                       disabled=pipeline.index.reranker is None,
                       help="A local cross-encoder re-reads the top 20 candidates to pick the best "
                            "passages. Slightly slower, noticeably more accurate.")
    top_k = st.slider("Passages per question", 3, 12, pipeline.settings.top_k)
    show_all = st.toggle("Show uncited passages", value=False)

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("New chat", icon="🗑️", use_container_width=True):
        state.messages = []
        st.rerun()
    if col2.button("Re-index", icon="🔄", use_container_width=True,
                   help="Re-read all documents (use after adding or editing files)."):
        with st.spinner("Re-indexing…"):
            RAGPipeline.load(rebuild=True, client=False)
            load_pipeline.clear()
        st.rerun()

st.title("Ask your documents")
st.caption(f"Answers come only from the {len(corpus.documents)} indexed documents, "
           "with numbered citations you can check.")

if not state.messages:
    examples = example_questions()
    if examples:
        st.markdown("##### Try asking")
        cols = st.columns(2)
        for i, q in enumerate(examples):
            if cols[i % 2].button(q, key=f"ex_{i}", use_container_width=True):
                state.pending = q
                st.rerun()

for msg in state.messages:
    render_message(pipeline, msg, show_all)

question = st.chat_input("Ask a question about the documents…") or state.pop("pending", None)

if question:
    history = [{"role": m["role"], "content": m["content"]}
               for m in state.messages if m["role"] == "user" or m.get("generated")]
    user_msg = {"id": uuid.uuid4().hex, "role": "user", "content": question}
    state.messages.append(user_msg)
    render_message(pipeline, user_msg, show_all)

    with st.chat_message("assistant", avatar="📚"):
        started = time.perf_counter()
        with st.spinner("Searching documents…"):
            sources = pipeline.retrieve(question, k=top_k, mode=mode, rerank=rerank, history=history)

        generated = pipeline.can_generate and bool(sources)
        try:
            text = st.write_stream(pipeline.stream(question, sources, history))
        except anthropic.APIError as exc:
            text, generated = friendly_error(exc), False
            st.error(text)

    elapsed = time.perf_counter() - started
    text = text if isinstance(text, str) else "".join(map(str, text))
    state.messages.append({
        "id": uuid.uuid4().hex,
        "role": "assistant",
        "content": text,
        "question": question,
        "sources": sources,
        "cited": cited_numbers(text, len(sources)) if generated else [],
        "generated": generated,
        "meta": f"⏱ {elapsed:.1f}s · {len(sources)} passages · "
                f"{MODE_LABELS[mode].split(' (')[0].lower()} search{' + rerank' if rerank else ''}",
    })
    st.rerun()  # re-render through render_message so sources + feedback appear
