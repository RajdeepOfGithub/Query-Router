"""End-to-end router eval: every routing_eval question through router.answer() (classify,
execute against the real systems, merge), graded on two separate axes:

  routing_correct  Phase 1's rule (run_routing_eval.is_correct): exact route for
                   sql/rag/both; for ambiguous ground truth, route 'both' + a disclosure.
  answer_correct   was the FINAL answer right, whatever route produced it
                   (data/router_answer_truth.jsonl):
    value            every expected figure present. For a SQL answer with a single headline
                     measure the headline itself must match, so a figure that appears only
                     in the alternatives list doesn't count. Otherwise Project 1's
                     grading.value_present is run on the final answer text (1% tolerance,
                     "$57.3 billion" == 57,347M).
    claims           Project 1's grading.claims_score (its own LLM judge) >= its
                     CLAIMS_PASS_MARK (0.75), run on the final answer text.
    ambiguous_rubric disclosure stated up front + a structured figure for the required
                     period + a non-declined explanation (no single correct answer exists).

Usage (from the repo root):
    python src/run_router_eval.py             # cost estimate only
    python src/run_router_eval.py --confirm   # -> baseline/e2e_v1/runs/{id}.json + summary.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import costs  # noqa: E402
import execute  # noqa: E402
from router import answer  # noqa: E402
from run_routing_eval import ROUTES, is_correct, read_questions  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRUTH = ROOT / "data" / "router_answer_truth.jsonl"
DEFAULT_OUT = ROOT / "baseline" / "e2e_v1"
EST_PER_QUESTION = 0.0016  # measured in Phase 2: classify ~0.0002, sql ~0.0013, rag ~0.0004, both ~0.0010, + judge


def _p1_grading():
    execute._add_path(execute.RAG_ROOT / "src")
    with execute._in_rag_root():
        import grading
    return grading


def load_truth() -> dict:
    truth = {t["id"]: t for t in (json.loads(x) for x in TRUTH.read_text(encoding="utf-8").splitlines() if x.strip())}
    p1 = {q["id"]: q for q in (json.loads(x) for x in
          (execute.RAG_ROOT / "data" / "baseline_questions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip())}
    for t in truth.values():
        if "claims_from" in t:
            t["claims"] = p1[t["claims_from"].split()[-1]]["expected"]["claims"]
    return truth


def _headline_values(r) -> list[float] | None:
    """Headline figure(s) of a SQL answer: the single measure, or every returned row."""
    if r.sql_raw is None or r.sql_raw["answer"].status != "answered":
        return None
    a = r.sql_raw["answer"]
    if a.measure is not None:
        return [a.measure.value_numeric]
    return [v["value"] for v in execute._sql_values(r.sql_raw)] or None


def check_values(r, values, grading) -> dict:
    out = {}
    headline = _headline_values(r) if r.route == "sql" else None
    for v in values:
        target = v["value"] * grading.UNIT_FACTORS.get(v["unit"], 1.0)
        if headline is not None:
            ok = any(abs(h - target) <= abs(target) * grading.VALUE_TOLERANCE for h in headline)
        else:
            ok = grading.value_present(r.answer, v["value"], v["unit"])
        out[v["label"]] = ok
    return out


def check_claims(r, claims, grading) -> float | None:
    with execute._in_rag_root(), costs.label("grade"):
        return grading.claims_score({"question": r.question, "answer_text": r.answer}, {"claims": claims}, "narrative")


def check_rubric(r, t) -> dict:
    sql_ok = False
    if r.sql_raw is not None and r.sql_raw["answer"].status == "answered":
        keys = [v["key"] for v in execute._sql_values(r.sql_raw)]
        sql_ok = bool(keys) and (t["required_sql_period"] is None or tuple(t["required_sql_period"]) in keys)
    return {"disclosure_up_front": r.answer.startswith("Note on the question:"),
            "structured_figure_for_period": sql_ok,
            "explanation_not_declined": r.rag_raw is not None and not getattr(r.rag_raw, "declined", True)}


def grade_answer(r, t, grading) -> dict:
    g = {"answer_type": t["answer_type"]}
    if t["answer_type"] in ("value", "value+claims"):
        g["values"] = check_values(r, t["values"], grading)
    if t["answer_type"] in ("claims", "value+claims"):
        g["claims_score"] = check_claims(r, t["claims"], grading)
    if t["answer_type"] == "ambiguous_rubric":
        g["rubric"] = check_rubric(r, t)
    parts = []
    if "values" in g:
        parts.append(all(g["values"].values()))
    if "claims_score" in g:
        parts.append(g["claims_score"] is not None and g["claims_score"] >= grading.CLAIMS_PASS_MARK)
    if "rubric" in g:
        parts.append(all(g["rubric"].values()))
    g["answer_correct"] = all(parts)
    return g


def summarize(results: list[dict], calls: list[dict]) -> dict:
    n = len(results)
    rc = sum(r["routing_correct"] for r in results)
    ac = sum(r["answer_correct"] for r in results)
    by = {k: {"n": 0, "routing": 0, "answer": 0} for k in ROUTES}
    for r in results:
        b = by[r["expected_route"]]
        b["n"] += 1
        b["routing"] += r["routing_correct"]
        b["answer"] += r["answer_correct"]
    cell = lambda rt, an: [r["id"] for r in results if r["routing_correct"] == rt and r["answer_correct"] == an]  # noqa: E731
    return {
        "routing_accuracy": f"{rc}/{n}", "answer_accuracy": f"{ac}/{n}",
        "by_expected_route": {k: f"routing {v['routing']}/{v['n']}, answer {v['answer']}/{v['n']}" for k, v in by.items()},
        "routing_right_answer_right": cell(True, True), "routing_right_answer_wrong": cell(True, False),
        "routing_wrong_answer_right": cell(False, True), "routing_wrong_answer_wrong": cell(False, False),
        "errors": [r["id"] for r in results if r.get("error")],
        "usage": costs.summarize(calls),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()
    questions, truth = read_questions(), load_truth()
    print(f"{len(questions)} questions, full pipeline + grading: estimated ${len(questions) * EST_PER_QUESTION:.3f}")
    if not args.confirm:
        print("No API calls made. Re-run with --confirm to execute.")
        raise SystemExit(1)

    grading = _p1_grading()
    runs = args.out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    start, results = len(costs.CALLS), []
    for q in questions:
        rec = {"id": q["id"], "question": q["question"], "expected_route": q["route"]}
        q_start = len(costs.CALLS)
        try:
            r = answer(q["question"])
            rec.update(route=r.route, decision=r.decision.model_dump(),
                       routing_correct=is_correct(q["route"], r.route, r.decision.ambiguity_disclosure),
                       **grade_answer(r, truth[q["id"]], grading), final_answer=r.answer, conflicts=r.conflicts,
                       sql_raw=execute.jsonable(r.sql_raw), rag_raw=execute.jsonable(r.rag_raw), error=None)
        except Exception as e:  # a crash is a result for this question, recorded, not raised
            rec.update(route=None, routing_correct=False, answer_correct=False,
                       error={"type": type(e).__name__, "message": str(e)[-1500:], "traceback": traceback.format_exc()[-3000:]})
        rec["usage"] = costs.summarize(costs.CALLS[q_start:])
        (runs / f"{q['id']}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        results.append(rec)
        detail = rec.get("values") or rec.get("rubric") or ""
        claims = f"claims={rec['claims_score']:.2f}" if rec.get("claims_score") is not None else ""
        err = f" ERROR {rec['error']['type']}" if rec["error"] else ""
        print(f"{q['id']} expected={q['route']:9s} got={str(rec['route']):5s} routing={'Y' if rec['routing_correct'] else 'N'} "
              f"answer={'Y' if rec['answer_correct'] else 'N'}  {detail} {claims}{err}")

    summary = summarize(results, costs.CALLS[start:])
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    summary.update(timestamp=datetime.now(timezone.utc).isoformat(),
                   git={"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("git", "timestamp")}, indent=2))


if __name__ == "__main__":
    main()
