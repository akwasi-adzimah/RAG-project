# RAG chatbot: analysis of imperfect and failed test outcomes

Every outcome below was observed in this project's test runs or reproduced on
purpose. Each entry gives what happened, the evidence, the likely cause and its
category.

| # | Outcome | Category |
|---|---|---|
| 1 | Security Engineering question answered with "text is truncated" | **Retrieval failure** |
| 2 | Retrieval metric counted outcome 1 as a hit | **Weak assertion** |
| 3 | Current keyword checks pass four deliberately wrong answers | **Weak assertion** |
| 4 | "Non-empty answer" test passed when no answer was generated | **Weak assertion** |
| 5 | Template test cases failed with placeholder expectations; a template refusal marker also matches normal answers | **Specification / test-oracle problem** |
| 6 | The ambiguous case expects behavior that no requirement defines | **Hallucinated test scenario** |
| 7 | Latency rose under 5 concurrent requests; a latent startup race | **Performance problem** (no rate-limit errors) |

No **generation failure** was observed; see the final section.

---

## 1. Retrieval failure: Security Engineering focus areas

**Test:** `supported_04` (originally `supported_05`), "What areas does Anthropic's
Security Engineering team focus on?" Expected answer: *SDLC security, supply
chain security, development environment security* (p.7).

**Observed:** the end-to-end test **FAILED** before the reranker was added.
Claude answered:

> "A fragmentary line states that the Security Engineering team covers
> 'development lifecycle, supply ... environment security' — the text is
> truncated, so the full list of focus areas is not recoverable from the excerpts."

**Evidence:** the chunk containing the answer ("The Security Engineering team
focuses on securing the software development lifecycle, supply chain security…")
ranked **8th for keywords and 19th for meaning**. What got retrieved instead was
a scrambled duplicate text layer from the same PDF page.

**Why it's a retrieval failure, not a generation failure:** Claude worked
correctly with what it got. It said the context was incomplete instead of
filling the gap from general knowledge. The right passage simply wasn't in
its context.

**Fix applied:** a cross-encoder reranker re-scores the top 20 candidates. The
correct chunk moved to 2nd place, and the test now passes.

## 2. Weak assertion: page-level retrieval metric

**Observed:** in the same run, the retrieval evaluation reported a **hit** for
this question, because a chunk from page 7 was in the top 6. The end-to-end test
failed at the same moment. One check said retrieval was fine; the other proved
it wasn't.

**Cause:** "right page" stood in for "right passage". Page 7 has three chunks,
and the one retrieved didn't contain the answer.

**Fix applied:** each case now has an `evidence` sentence, and a hit requires a
retrieved chunk that contains that exact sentence. Under the stricter metric,
hybrid search without reranking drops to 88% (7 of 8), which matches the
end-to-end failure.

## 3. Weak assertion: keyword checks accept wrong answers

**Observed:** every end-to-end check currently passes (21/21). To test the checks
themselves, I gave the grading function (`rag_system/evaluate.py: grade`)
four answers that are **factually wrong**. All four passed:

| Case | Wrong answer given | Check | Result |
|---|---|---|---|
| `paraphrase_02` | "AI made seasoned contributors 19% **faster**." | contains "19%" | ✅ passed |
| `supported_05` | "Developers say AI is 43% **more effective than humans** at refactoring." | contains "43%" | ✅ passed |
| `ambiguous_01` | "The Legal team uses Claude Code; unlike the Inference team, it builds prototypes." | names ≥2 teams | ✅ passed |
| `adversarial_01` | "There was a 19% increase **in speed**, explained by better autocomplete." | contains "increase" and no listed premise-accepting phrase | ✅ passed |

**Cause:** checking that a keyword is present shows the right number came up,
not that the claim is correct. `paraphrase_02` is the worst case. Its keywords
are "slow" **or** "19%", and the false premise ("19% faster") contains "19%",
so it passes the very error the paraphrase is meant to catch.

**Impact:** the 21/21 result is weaker evidence than it looks. A manual read of
the actual answers (outcome 8 below) found them correct. The tests didn't prove
that; a person did.

**Recommended fix:**
- Require keywords that encode direction, e.g. `19%` **and** (`slow` or `increase…time`).
- Add forbidden phrases such as "faster".
- Or grade with an LLM judge against `expected_behavior`, with a sample of its verdicts checked by a person.

## 4. Weak assertion: "non-empty answer" passed with no answer

**Observed:** in the first test run (before the rebuild, with no API key set),
`test_chatbot_returns_nonempty_answer` **passed for all 3 cases**. Every "answer"
was the same fixed message:

> "I found relevant document excerpts, but no Anthropic API key is configured,
> so I skipped the Claude-generated answer…"

**Cause:** "the string isn't empty" accepts a status message, an error text or
a refusal as if it were an answer. That run reported 3 passed and 3 failed, and
all 3 passes were meaningless.

**Fix applied:**
- The end-to-end suite now **skips** when no key is configured, instead of passing on fixed messages.
- The retrieval-only message is a named constant (`NO_API_KEY`) covered by its own unit test.
- **Still open:** the non-empty test should also assert that the answer isn't one of the system's fixed messages.

## 5. Specification / test-oracle problems in the template test set

**5a. Placeholder expectations.** The original `test_cases.json` contained
`"question": "REPLACE with a factual question…"` with
`expected_keywords: ["REPLACE", "KEYWORD"]`. Two supported tests **failed** for
that reason alone. The chatbot was asked a meaningless question and judged
against a meaningless answer. Nothing was wrong with the system.

**5b. Refusal marker that matches normal answers.** The template's
"handled conservatively" check accepts the phrase
**"based on the provided documents"** as a sign of refusal. Grounded answers
often *start* with that phrase. Example: "Based on the provided documents, 72%
use AI daily" counts as a conservative refusal. An answer that **makes up a
salary** but opens with that phrase would therefore pass the unsupported test.

**Fix applied:**
- 5a: replaced the placeholders with 10 cases checked against the source text.
- 5b: the new grading function drops that phrase. The bot is instructed to
  reply with one exact refusal sentence ("The answer is not in the provided
  documents."), and invented-looking content, such as a € figure, fails the
  test even when a refusal marker is present.

## 6. Hallucinated test scenario: what "correct" means for ambiguous questions

**Case:** `ambiguous_01`, "How does the team use Claude Code?" Expected behavior,
written by the AI test generator: *"Either asks which team is meant, or
summarises several teams… Must not silently pick one team."*

**Observed:** the test passes. The bot summarized 6 teams and offered to narrow
it down.

**Problem:** no requirement for the chatbot says how it should handle ambiguous
questions. The system prompt says nothing about asking for clarification, and
neither does the assignment brief. The AI made up the rule *"should ask which
team"* and then made the check accept either of two behaviors so that it
would pass. The case also quietly assumes "about 10 teams" means every reading
should be covered, yet the answer covered 6. Whether that's acceptable was never
specified, and the check doesn't look at it.

A second example was caught during review. Candidate U3 expected the bot to
*"note the document couldn't be read"* for the scanned PDF. No such requirement
exists, so the case was rejected.

**Recommended fix:** first write the behavior down as a real requirement, for
example "when a question matches several entities, list them or ask which one".
Add it to the system prompt, then test for it.

## 7. Performance: latency under concurrency, plus a latent race

**Observed** (`stress_test.py`, two runs; baseline from `performance_test.py`):

| Concurrency | Avg latency (runs 1 / 2) | Same questions one at a time | Failures | 429/529 errors |
|---|---|---|---|---|
| 1 | 3.03 / 3.61 s | 3.33 s | 0 | 0 |
| 3 | 3.78 / 5.68 s | 4.23 s | 0 | 0 |
| 5 | 4.65 / 5.37 s | 3.66 s | 0 | 0 |

**Classification:** a mild **performance** finding, not a rate-limit problem.
- **Requests:** none failed. The SDK log shows 18/18 `200 OK` with zero retries, which rules out silent automatic retries of rate-limit errors.
- **Latency:** each request took about 1–2 s longer at 5 concurrent.
- **Throughput:** total time for 5 requests was still about 2.5× better than one at a time.
- **Likely causes:** API-side queueing plus the local reranker sharing the CPU. Telling them apart would need timing for each stage.

**Latent defect:** the shared pipeline is created lazily on first use. If the
first requests arrive at the same moment, several threads can each build it.
The stress test never triggered this because its single-request level warms the
pipeline first. That ordering is a test design weakness in its own right.

**Measurement caveat:** the first question in `performance_test.py` also includes
loading the index and models, about 1 s of cold-start time.

## 8. Generation failure: none observed (checked, not assumed)

To check for unsupported claims, I matched 8 specific details from the two
longest answers (adversarial and ambiguous) against the indexed document text.
All 8 appear in the source:

| Claim in Claude's answer | Found in |
|---|---|
| 56% speedup (Peng et al.), 21% (Paradis et al.) | METR p.3 |
| 16 developers, 246 tasks | METR pp.1–2 |
| 143 hours of screen recordings | METR p.2 |
| Pod IP address exhaustion (Data Infrastructure) | Claude Code doc p.3 |
| Vim key bindings, ~70% autonomous | Claude Code doc p.5 |
| 10–15 min down to ~5 min (Security Engineering) | Claude Code doc p.7 |
| 80% cut in research time (Inference) | Claude Code doc |

This is a spot check, not proof. Each claim was matched to some passage in the
corpus, not necessarily to the excerpt the answer cites. A stronger check would
confirm that every `[n]` citation actually supports the sentence it's attached to.
