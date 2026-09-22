# Query Router

Routes questions about JPMorgan Chase to Project 2's Text-to-SQL backend (XBRL facts,
`../Text-to-SQL`), Project 1's RAG backend (filing prose + earnings call, `../RAG/financial-rag`),
both, or flags them as ambiguous. Phase 1 only classifies; neither backend is called.

```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # OPENAI_API_KEY in .env
.venv/Scripts/python src/run_routing_eval.py --confirm   # -> baseline/v1/
.venv/Scripts/python -m pytest tests
```

`data/routing_eval.jsonl` has 18 questions, with the correct route as ground truth:
- 6 sql: from Text-to-SQL's verified eval set
- 4 rag: from financial-rag's narrative/transcript questions
- 4 both: written fresh, with premises checked against the XBRL facts
- 4 ambiguous: deliberately constructed, with the competing readings noted per question

## v1 baseline (gpt-4o-mini, $0.0019)

| route | correct |
|---|---|
| sql | 6/6 |
| rag | 2/4 |
| both | 4/4 |
| ambiguous | 0/4 |
| **overall** | **12/18** |

All four ambiguous questions were routed to a single route at confidence 0.9, which is the
failure that matters here. The model's stated confidence doesn't reflect genuine unclarity. r08 and r10
(rag) went to "both": each asks for an impact or a figure that only exists in prose/transcript.

## v2 ($0.0022): r15 relabeled sql; ambiguous = "both" + disclosure

| route | correct |
|---|---|
| sql | 7/7 |
| rag | 2/4 |
| both | 4/4 |
| ambiguous (graded as both + disclosure) | 1/3 |
| **overall** | **14/18** |

r16 now routes to both with a disclosure. r17 and r18 still go to rag at 0.9. Nothing previously
correct regressed. r08 and r10 (rag) still go to both, and r08 now carries a disclosure it shouldn't.

## Phase 2: execution

`src/router.py`: `answer(question)` classifies the question, then `src/execute.py` runs it.
- sql: Project 2's `answer_question` + `confidence.assess`
- rag: Project 1's search, generation, `verify_answer` and `compute_confidence`, called in the
  same order as its own CLI
- both: runs both, then merges them with a fixed template (not an LLM rewrite), tagging
  claims `[SQL]` / `[RAG-n]` with a source list

Neither sibling repo is modified. Tie-break: a RAG dollar figure within 10% of an SQL figure but
inconsistent at its own rounding counts as a conflict, and the SQL figure wins. RAG figures matching
no SQL figure are listed as unconfirmed. `src/costs.py` measures spend across all three
projects at the OpenAI SDK level.

Environment: the router venv carries both projects' runtime deps at their own pinned versions,
except openai (3.3.0, not financial-rag's 3.13.0): instructor 1.17 and openai 3.13 need
incompatible jiter versions.

Test run: 3/3 routing tests pass, $0.00297 measured. The real "both" example is in
`baseline/phase2_both_example.json`.

## Phase 3: end-to-end eval (e2e v1, frozen)

`python src/run_router_eval.py --confirm` sends all 18 questions through `router.answer()`
(classify, execute, merge) and grades two things separately:
- routing_correct: Phase 1's rule
- answer_correct: judged against `data/router_answer_truth.jsonl`. SQL values come from
  Text-to-SQL's verified set, RAG claims from financial-rag's set, and "both" drivers from the
  cited filing passages. Project 1's own `value_present` and `claims_score` do the checking.

| expected route | routing | answer |
|---|---|---|
| sql | 7/7 | 6/7 |
| rag | 2/4 | 1/4 |
| both | 4/4 | 1/4 |
| ambiguous | 0/3 | 0/3 |
| **overall** | **13/18** | **8/18** |

- Routing right, answer wrong: r07, r09, r12, r14 (RAG explanations miss the filing's drivers),
  r13 (gives the increase as "4%", not the FY2024 figure the truth requires), r15 (diluted EPS
  instead of net income, Project 2's e07 failure).
- Routing wrong, answer right: r08.
- r16 routed to "both" this run but without a disclosure (it had one in the routing-only v2
  run), so the classifier isn't stable run to run at temperature 0.

Cost $0.0213 (118 calls).
