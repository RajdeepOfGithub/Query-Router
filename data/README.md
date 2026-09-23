
# Data

No corpus to download here. This repo depends on two sibling repos being
present and already set up:

```
AI Projects/
├── RAG/financial-rag/       # answers via retrieval over JPMC filings
├── Text-to-SQL/             # answers via SQL over JPMC XBRL facts
└── Query-Router/            # this repo — imports functions from both
```

Query-Router imports the answer functions from `../RAG/financial-rag` and
`../Text-to-SQL` directly (no HTTP layer) to classify a question and route
it to the right backend, or run both and reconcile conflicts. See each
sibling's own `data/README.md` for how to fetch/regenerate its corpus:

- `../RAG/financial-rag/data/README.md`
- `../Text-to-SQL/data/README.md`

`routing_eval.jsonl` and `router_answer_truth.jsonl` in this directory are
this project's own eval sets and are committed, not regenerated from the
siblings' data.
