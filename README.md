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
