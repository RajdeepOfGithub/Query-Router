"""Routing eval grading rules (deterministic, no LLM calls)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from run_routing_eval import grade, is_correct, read_questions  # noqa: E402


def _r(i, expected, got, conf=0.9, disclosure=None):
    return {"id": i, "expected": expected, "route": got, "confidence": conf, "ambiguity_disclosure": disclosure,
            "correct": is_correct(expected, got, disclosure)}


def test_eval_set_shape():
    qs = read_questions()
    counts = {}
    for q in qs:
        counts[q["route"]] = counts.get(q["route"], 0) + 1
    assert counts == {"sql": 7, "rag": 4, "both": 4, "ambiguous": 3} and len({q["id"] for q in qs}) == 18
    assert "correction" in next(q for q in qs if q["id"] == "r15")  # relabel is documented, not silent


def test_both_gets_no_partial_credit():
    s = grade([_r("a", "both", "sql"), _r("b", "both", "rag"), _r("c", "both", "both")])
    assert s["by_route"]["both"] == "1/3"


def test_ambiguous_correct_only_as_both_with_disclosure():
    s = grade([_r("a", "ambiguous", "sql", 0.9), _r("b", "ambiguous", "rag", 0.5),
               _r("c", "ambiguous", "both"), _r("d", "ambiguous", "both", disclosure="figures vs commentary")])
    assert s["by_route"]["ambiguous"] == "1/4"  # 'both' without a disclosure is not enough
    assert [x["id"] for x in s["ambiguous_confidently_routed"]] == ["a"]
