# Document Q&A (RAG)

Ask questions about the PDF/DOCX reports in the `QA/` folder. Answers are written
by Claude **only from retrieved passages**, with numbered citations back to the
exact file and page.

```
question ─► hybrid retrieval ─────────────► rerank ─► Claude ─► cited answer
            ├ BM25 keyword search ┐  RRF        cross-     (streamed)
            └ semantic embeddings ┘  fusion     encoder
```

| Stage | What it does | Why |
|---|---|---|
| Extraction (`documents.py`) | `pypdfium2` per page, drops duplicated text layers, rejoins hyphenated words | Cleaner text than `pypdf`; each chunk keeps its exact page number |
| Chunking | ~220-word sentence-aware chunks, ~40-word overlap, never crossing pages | Chunks read as whole sentences, and every citation points to one page |
| Keyword search | BM25 with stop-words and light stemming | Precise for names, figures, exact phrases |
| Semantic search | `BAAI/bge-small-en-v1.5` embeddings, run locally | Finds paraphrases ("devs" / "daily" → "developers" / "every day") |
| Fusion | Reciprocal rank fusion of both rankings | Combines them without calibrating scores |
| Rerank | `ms-marco-MiniLM-L-6-v2` cross-encoder re-scores the top 20 | Reads question and passage together, so the best passage ranks first |
| Generation (`generator.py`) | Claude (`claude-opus-5`, effort `medium`), streamed, refusal fallbacks on | Cites `[n]`; replies "The answer is not in the provided documents." when unsupported |

Both local models (~150 MB total) download once into `rag_system/.cache/models/`.
Nothing but the final prompt leaves your machine.

## Setup

```bash
cd /Users/radzimah/QA
source .venv/bin/activate
pip install -r rag_system/requirements.txt
cp rag_system/.env.example rag_system/.env   # then put your key in it
```

`.env` settings (all optional except the key):

```
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-opus-5   # or claude-sonnet-5 for lower cost
RAG_EFFORT=medium               # low | medium | high
RAG_DOCS_DIR=/path/to/docs      # defaults to the QA/ folder
```

Without an API key everything still runs in **retrieval-only mode**: you see
the matching passages but no written answer.

## Use it

```bash
streamlit run rag_system/app.py                           # chat app
python -m rag_system.cli "What share of devs use AI daily?" --show-sources
python -m rag_system.cli                                   # terminal chat
python -m rag_system.cli --search-only "METR slowdown"     # retrieval only, free
python -m rag_system.cli --rebuild --stats                 # re-index + corpus stats
```

The index is cached in `rag_system/.cache/` and rebuilds itself when a document
is added, removed or modified. The app's **Re-index** button does the same.

The app has:
- streamed answers
- source cards showing each passage's keyword, semantic and rerank scores
- a toggle between cited-only and all retrieved passages
- retrieval mode and rerank switches
- thumbs up/down feedback, logged to `rag_system/.cache/feedback.jsonl` for QA review

## Testing and evaluation

```bash
pytest -m "not e2e"                  # 76 offline tests, free (this is what CI runs)
pytest -m e2e                        # 20 end-to-end tests against Claude (costs credits)
python -m rag_system.evaluate        # retrieval quality, free
python -m rag_system.evaluate --answers   # + answer checks via Claude (costs credits)
```

| Test file | What it covers |
|---|---|
| `tests/test_documents.py` | Text cleaning, sentence chunking, overlap, DOCX, corrupt files |
| `tests/test_index.py` | Tokenizing, each retrieval mode, fusion, reranking, cache invalidation |
| `tests/test_pipeline.py` | Request shape, citations, retrieval-only mode, refusals, truncation, input validation, follow-up questions |
| `tests/test_app.py` | The Streamlit app renders against the real index |
| `tests/test_stable_components.py` | The deterministic parts on the **real** documents: evidence really is on the stated page, the scanned PDF is skipped, chunk limits, normalized embeddings, the evidence passage is retrieved in the top 6, retrieval is repeatable, the prompt carries the evidence, the no-key path; also checks the case set's category mix |
| `tests/test_rag_system.py` | End-to-end: one test per case category (supported, paraphrase, ambiguous, unsupported, adversarial) |

The unit tests use a fake embedder and a fake Claude client (`tests/conftest.py`),
so they never touch the network.

`test_cases.json` holds 10 cases chosen from a larger AI-generated candidate
list. The full list and the reasons for each pick are in
[`../test_case_candidates.md`](../test_case_candidates.md).

| Type | Count | Example | Pass condition (`evaluate.grade`) |
|---|---|---|---|
| supported | 5 | "What percentage of developers who have tried AI coding tools use them every day?" | Answer contains a verified keyword (`72%`) |
| paraphrase | 2 | "Among devs who've experimented with AI assistants, what share rely on them daily?" | Same keyword as the original question |
| ambiguous | 1 | "How does the team use Claude Code?" | Asks which team, or covers 2 or more teams |
| unsupported | 1 | "What salary does the survey report for senior engineers in Germany?" | Declines; no invented figure |
| adversarial | 1 | "Since METR found AI made developers 19% faster, what explains that speedup?" | Corrects the premise; doesn't explain a speedup |

Cases with a known source record the file, page and an exact `evidence`
sentence. Retrieval counts as a hit only when that sentence is in the top 6
passages:

| Config | hit@6 | MRR |
|---|---|---|
| **hybrid + rerank** (default) | **100%** | **0.84** |
| hybrid | 88% | 0.73 |
| keyword only (old system) | 88% | 0.57 |
| semantic only | 75% | 0.69 |

With only 8 cases that have a known source, a single miss moves hit@6 by 12
points, so add cases before trusting small differences.

## Known limitations

- `AI Impact on Software Engineers 2026.pdf` is a scanned image with no text
  layer, so it is skipped. It needs OCR to be searchable.
- Charts and figures are only indexed as the text they contain.
- The embedding library occasionally prints a harmless `libc++abi … mutex`
  message when a CLI run exits.

## Layout

```
rag_system/
  config.py      settings (+ .env loading)
  documents.py   extraction, cleaning, chunking
  embeddings.py  local embedder + reranker (fastembed / ONNX)
  index.py       BM25 + semantic + RRF + rerank, on-disk cache
  generator.py   prompt, streaming, citations, error messages
  pipeline.py    RAGPipeline - the single entry point
  app.py         Streamlit UI
  cli.py         terminal interface
  evaluate.py    retrieval / answer evaluation
  test_adapter.py  thin wrapper used by tests and load scripts
  _legacy/       previous BM25-only implementation (backup)
```
