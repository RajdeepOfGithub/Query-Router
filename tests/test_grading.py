"""Routing eval grading rules (deterministic, no LLM calls)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from run_routing_eval import grade, read_questions  # noqa: E402


def _r(i, expected, got, conf=0.9):
    return {"id": i, "expected": expected, "route": got, "confidence": conf, "correct": expected == got}


def test_eval_set_shape():
    qs = read_questions()
    counts = {}
    for q in qs:
        counts[q["route"]] = counts.get(q["route"], 0) + 1
    assert counts == {"sql": 6, "rag": 4, "both": 4, "ambiguous": 4} and len({q["id"] for q in qs}) == 18


def test_both_gets_no_partial_credit():
    s = grade([_r("a", "both", "sql"), _r("b", "both", "rag"), _r("c", "both", "both")])
    assert s["by_route"]["both"] == "1/3"


def test_confident_routing_of_ambiguous_is_flagged():
    s = grade([_r("a", "ambiguous", "sql", 0.9), _r("b", "ambiguous", "rag", 0.5), _r("c", "ambiguous", "ambiguous")])
    assert [x["id"] for x in s["ambiguous_confidently_routed"]] == ["a"]
    assert s["by_route"]["ambiguous"] == "1/3"
