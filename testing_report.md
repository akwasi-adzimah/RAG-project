# Testing report: RAG document Q&A chatbot

**System:** a chatbot that answers questions from five fixed PDF/DOCX reports.
Retrieval combines keyword (BM25) and semantic search, then a local
cross-encoder reranks the results. Claude (`claude-opus-5`) writes the answers,
with citations, and is told to answer only from the retrieved passages.

**Evidence:**

| Item | Location |
|---|---|
| Test cases | `test_cases.json` (10 reviewed cases) |
| Candidate cases and review | `test_case_candidates.md` |
| Failure analysis | `failure_analysis.md` |
| `pytest -v` output (69/69 passed) | `results/pytest_output.txt` |
| Performance and stress results | `results/performance_results.txt`, `results/stress_results.txt` |
| UAT ratings (AI-conducted) | `results/uat_ai_evaluation.md` |
| UAT form for a human tester | `uat_feedback.md` |

## 1. Which tests came from known requirements, and which were suggested by AI?

**From known requirements** (the assignment brief and the course test template):
- **System requirements:** answer only from the documents, give a non-empty answer, decline unsupported questions.
- **Template tests:** the three test shapes in `tests/test_rag_system.py` (non-empty answer, expected concepts present, unsupported handled conservatively).
- **Category mix:** 4–5 supported, 2 paraphrased, 1 ambiguous, 1 unsupported and 1 adversarial case, now enforced by `test_case_set_meets_coverage_requirements`.

**Suggested by AI (Claude):**
- **The 10 cases:** chosen from 26 AI-generated candidates, each checked against the extracted document text.
- **New checks:** the ambiguous-case and adversarial-case checks.
- **Retrieval evaluation:** the stricter version based on evidence sentences.
- **Unit tests:** all 48, covering chunking, retrieval, reranking, caching, request shape and the grading logic.

In practice almost every concrete test was written by AI. That matters for question 7.

## 2. Most useful and least useful AI-generated cases

**Most useful: `supported_04`** ("What areas does Anthropic's Security Engineering
team focus on?"). It was the only case that exposed a real defect. The chunk
containing the answer ranked 8th for keywords and 19th for meaning, so Claude
received a scrambled duplicate of the page and replied that the text was
truncated. Adding a cross-encoder reranker raised retrieval to 8 of 8 cases
(MRR 0.73 → 0.84). The adversarial false-premise case
(`adversarial_01`) came second: it checks a hallucination risk the template
didn't cover.

**Least useful, based on an invented assumption: `ambiguous_01`** ("How does
the team use Claude Code?"). The AI decided the bot "should ask which team",
but no requirement says that. The system prompt says nothing about asking for
clarification, and the AI then wrote a check loose enough to pass either way.
During review I also rejected candidate U3, which expected the bot to "note the
document couldn't be read" for the scanned PDF, another invented requirement.

## 3. A test passed although the answer wasn't good

Yes, in three ways (details in `failure_analysis.md`, outcomes 2–4):
- **No answer at all:** with no API key set, `test_chatbot_returns_nonempty_answer` passed for every case, but each "answer" was the fixed message "no Anthropic API key is configured".
- **The keyword checks accept wrong answers.** I gave the grading four deliberately wrong answers, and **all four passed**. For example, "AI made seasoned contributors 19% **faster**" passes `paraphrase_02` because it contains "19%", which is exactly the error that case is meant to catch.
- **The page-level retrieval metric** counted a hit while the end-to-end test failed, because a chunk from the right page didn't contain the answer.

So the final 21/21 end-to-end pass rate shows the answers *mention* the right
facts. The answers were confirmed correct only because I read them.

## 4. Most common failure modes

| Mode | Count | Example |
|---|---|---|
| Specification/oracle and weak assertion | **5** | placeholder template cases; keyword checks pass wrong answers; non-empty check passes fixed messages |
| Hallucinated test scenario | 1 | `ambiguous_01` assumed a clarification requirement |
| Retrieval | 1 | Security Engineering passage not retrieved (fixed) |
| Performance | 1 (mild) | +1–2 s latency at 5 concurrent requests |
| Generation | **0** | 8/8 spot-checked claims were found in the source |

The tests failed more often than the chatbot did. The one real system defect
was in retrieval, and Claude handled even that honestly: it said the context
was incomplete instead of making something up.

## 5. How latency changed as workload increased

Sequential (10 questions): min **2.08 s**, average **4.66 s**, median **3.28 s**,
max **9.41 s**, **0 failures**. Latency mostly follows answer length. The
single-sentence refusal was fastest, while the multi-team summary (9.41 s) and
the premise correction (8.07 s) were slowest.

| Concurrent requests | Avg latency, run 1 / run 2 | Same questions one at a time |
|---|---|---|
| 1 | 3.03 s / 3.61 s | 3.33 s |
| 3 | 3.78 s / 5.68 s | 4.23 s |
| 5 | 4.65 s / 5.37 s | 3.66 s |

Each request got about 1–2 s slower at 5 concurrent. Total throughput still
improved about 2.5×: 5 answers arrived in 6.5–7.6 s, against about 18 s one
after another. There were no failures and no rate-limit errors: the SDK log
showed 18/18 `200 OK` and zero retries. The load was light, so the point where
rate limits begin wasn't reached.

## 6. What did the UAT tester notice that the automated tests did not?

The UAT session used five realistic questions that aren't in the test set.
It was **conducted by Claude, not an independent human**; details are in
`results/uat_ai_evaluation.md`. Average ratings: relevance **4.8**, correctness
**4.6**, clarity **4.2**, trust **4.4**, usefulness **4.0** (overall 4.4/5). Every
figure in every answer was checked against its source. It noticed four things
the automated tests did not:

- **Multi-document questions fall back to one report.** "What are the benefits
  and risks of adopting AI coding tools, based on these reports?" retrieved
  6/6 passages from the Sonar survey. Several were vendor promotion pages. The
  answer missed the corpus's most relevant risk (METR's 19% slowdown) and the
  survey's own 35% productivity figure. It would pass any keyword test, because
  its weakness is what it left out.
- **Strict grounding can confuse users.** One answer opened with "the excerpts
  never name the study 'METR'" because the page with the name wasn't retrieved.
  That's accurate, but it reads as an error.
- **Answer length and wait time grow with question breadth.** Factual lookups
  took about 4–5 s; synthesis questions took 10–14 s and 400+ words.
- **Slight causal overreach.** One answer presented a list of *concerns* as the
  *reasons* for distrust.

A human UAT session (`uat_feedback.md`) is still recommended. An AI rating the
output of the system it helped build has the conflict described in question 7.

## 7. Would I trust the same model to be the sole generator and judge of its tests?

**No.** In this project the same model wrote the chatbot, the test cases and
the grading logic, and its blind spots carried across all three:
- **It invented a requirement** (ambiguous questions must be clarified) and then wrote a check that was sure to pass.
- **It wrote keyword checks that pass factually wrong answers.** That only came to light when wrong answers were fed in on purpose.
- **It tuned retrieval on the same 8 cases** it used to measure retrieval, so 100% hit@6 is partly overfitting.

AI was valuable for *breadth*: 26 candidates across categories I wouldn't all
have thought of, and fast unit-test coverage. But the checks that caught real
problems were independent ones: verifying expected answers against the source
text, feeding the grading deliberately wrong answers, and a person reading the
answers. AI should propose tests; humans and an independent oracle (people's
labels, or a different model as judge with its verdicts spot-checked) should
decide what passes.

## 8. Before calling it production-ready

1. **Stronger checks:** check the direction of a claim (`19%` **and** "slower"), plus forbidden words, or use an LLM judge scored against `expected_behavior` with people auditing a sample of its verdicts.
2. **A larger, independent test set:** 50–100 cases labeled by people, with a held-out portion not used for tuning. One miss currently moves hit@6 by 12 points.
3. **Citation verification:** check that each `[n]` citation supports its sentence, not just that the fact exists somewhere in the documents.
4. **A real ambiguity requirement:** specify the behavior, add it to the system prompt, then test it.
5. **Coverage gaps:** OCR for the scanned PDF, which is currently skipped. Tests for prompt injection and multi-document questions. Retrieval diversity (a limit on passages per document) so broad questions draw on every relevant report, which was the main UAT finding.
6. **Load and resilience:** load-test until rate limits appear; add backoff or a queue and a lock around the first pipeline load (a latent race); time each stage.
7. **Operations and security:**
   - **Rotate the API key**, which was exposed during development.
   - Add authentication to the app.
   - Set cost limits.
   - Monitor the 👍/👎 feedback log.
8. **Complete the UAT** and fix what users report.
