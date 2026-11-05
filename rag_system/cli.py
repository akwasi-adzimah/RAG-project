"""Command-line interface.

    python -m rag_system.cli "What share of developers use AI daily?"
    python -m rag_system.cli                     # interactive chat
    python -m rag_system.cli --search-only "..."  # retrieval only, no API call
    python -m rag_system.cli --rebuild --stats   # re-index and show corpus stats
"""
from __future__ import annotations

import argparse

import anthropic

from .generator import cited_numbers, friendly_error
from .index import MODES
from .pipeline import RAGPipeline


def print_sources(results, cited: list[int] | None = None) -> None:
    print("\nSources:")
    for n, r in enumerate(results, start=1):
        mark = "*" if cited and n in cited else " "
        preview = " ".join(r.chunk.text.split())[:160]
        print(f" {mark}[{n}] {r.chunk.label}\n      {preview}...")


def ask(pipeline: RAGPipeline, question: str, args, history: list[dict]) -> str:
    sources = pipeline.retrieve(question, k=args.top_k, mode=args.mode, history=history)
    if args.search_only:
        print_sources(sources)
        return ""
    parts: list[str] = []
    try:
        for text in pipeline.stream(question, sources, history):
            print(text, end="", flush=True)
            parts.append(text)
    except anthropic.APIError as exc:
        print(f"\n[error] {friendly_error(exc)}")
        return ""
    print()
    if args.show_sources:
        print_sources(sources, cited_numbers("".join(parts), len(sources)))
    return "".join(parts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ask questions over the local documents.")
    parser.add_argument("question", nargs="?", help="omit for interactive mode")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--search-only", action="store_true", help="show retrieved passages only")
    parser.add_argument("--show-sources", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="force re-indexing")
    parser.add_argument("--stats", action="store_true", help="print corpus statistics")
    args = parser.parse_args(argv)

    pipeline = RAGPipeline.load(rebuild=args.rebuild)

    if args.stats:
        corpus = pipeline.index.corpus
        for d in corpus.documents:
            print(f"{d.name:<50} pages={d.pages:<4} words={d.words:<6} chunks={d.chunks}")
        for name in corpus.skipped:
            print(f"{name:<50} SKIPPED (no extractable text)")
        print(f"\nTotal chunks: {len(corpus.chunks)}")
        if not args.question:
            return

    if not pipeline.can_generate and not args.search_only:
        print("ANTHROPIC_API_KEY not set - running retrieval-only.\n")

    if args.question:
        ask(pipeline, args.question, args, [])
        return

    print("Ask about the documents. Ctrl-D to exit.")
    history: list[dict] = []
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question:
            reply = ask(pipeline, question, args, history)
            if reply:
                history += [{"role": "user", "content": question},
                            {"role": "assistant", "content": reply}]


if __name__ == "__main__":
    main()
