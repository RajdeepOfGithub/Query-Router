"""Single entry point: answer(question) -> classify, execute, return.

The user-facing answer is RouterResult.answer. sql_raw / rag_raw hold both source
systems' full outputs for evaluation and aren't meant to be shown to the user.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import costs  # noqa: E402
from classify import RouteDecision, classify  # noqa: E402
from execute import execute, jsonable  # noqa: E402

costs.install()


@dataclass
class RouterResult:
    question: str
    route: str
    decision: RouteDecision
    answer: str
    sql_raw: dict | None
    rag_raw: object | None
    conflicts: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return jsonable(self)


def answer(question: str) -> RouterResult:
    start = len(costs.CALLS)
    with costs.label("classify"):
        decision = classify(question)
    with costs.label(f"execute:{decision.route}"):
        ex = execute(question, decision.route, decision.ambiguity_disclosure)
    return RouterResult(question, ex.route, decision, ex.answer, ex.sql_raw, ex.rag_raw, ex.conflicts,
                        costs.summarize(costs.CALLS[start:]))


if __name__ == "__main__":
    r = answer(" ".join(sys.argv[1:]) or "What was total net revenue for the second quarter of 2026?")
    print(f"[route={r.route} conf={r.decision.confidence}]\n\n{r.answer}\n\nusage: {r.usage}")
