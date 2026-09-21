"""Run the router over data/routing_eval.jsonl and grade against ground-truth routes.

Exact match only: a 'both' question classified as 'sql' or 'rag' is wrong, with no
partial credit for picking one half. The failure mode tracked separately:
ambiguous questions routed to a single route with confidence >= CONFIDENT
(false confidence on a genuinely unclear question).

Usage (from the repo root):
    python src/run_routing_eval.py              # cost estimate only
    python src/run_routing_eval.py --confirm    # -> baseline/v1/runs/{id}.json + summary.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import MODEL, ROOT, classify  # noqa: E402

QUESTIONS = ROOT / "data" / "routing_eval.jsonl"
DEFAULT_OUT = ROOT / "baseline" / "v1"
ROUTES = ["sql", "rag", "both", "ambiguous"]
CONFIDENT = 0.7
PRICE_IN, PRICE_CACHED, PRICE_OUT = 0.15, 0.075, 0.60  # gpt-4o-mini, USD per 1M tokens
EST_TOKENS = (750, 80)  # prompt, completion per question


def read_questions() -> list[dict]:
    return [json.loads(x) for x in QUESTIONS.read_text(encoding="utf-8").splitlines() if x.strip()]


def cost(calls: list[dict]) -> float:
    return sum(((c["prompt"] - c["cached"]) * PRICE_IN + c["cached"] * PRICE_CACHED + c["completion"] * PRICE_OUT) / 1e6
               for c in calls)


def grade(results: list[dict]) -> dict:
    by_route = defaultdict(lambda: [0, 0])
    for r in results:
        by_route[r["expected"]][0] += r["correct"]
        by_route[r["expected"]][1] += 1
    correct = sum(r["correct"] for r in results)
    return {
        "overall": f"{correct}/{len(results)}",
        "accuracy": round(correct / len(results), 3),
        "by_route": {k: f"{by_route[k][0]}/{by_route[k][1]}" for k in ROUTES},
        "misclassified": [{"id": r["id"], "expected": r["expected"], "got": r["route"], "confidence": r["confidence"]}
                          for r in results if not r["correct"]],
        "ambiguous_confidently_routed": [
            {"id": r["id"], "got": r["route"], "confidence": r["confidence"]} for r in results
            if r["expected"] == "ambiguous" and r["route"] != "ambiguous" and r["confidence"] >= CONFIDENT],
        "confusion": {e: {g: sum(1 for r in results if r["expected"] == e and r["route"] == g) for g in ROUTES}
                      for e in ROUTES},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()
    questions = read_questions()
    est = len(questions) * (EST_TOKENS[0] * PRICE_IN + EST_TOKENS[1] * PRICE_OUT) / 1e6
    print(f"{len(questions)} questions x 1 {MODEL} call: estimated ${est:.4f} (x3 if every call retries)")
    if not args.confirm:
        print("No API calls made. Re-run with --confirm to execute.")
        raise SystemExit(1)

    runs = args.out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    results, all_calls = [], []
    for q in questions:
        calls = []

        def record(resp, calls=calls):
            u = resp.usage
            calls.append({"prompt": u.prompt_tokens, "completion": u.completion_tokens,
                          "cached": getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0})

        d = classify(q["question"], on_response=record)
        r = {"id": q["id"], "question": q["question"], "expected": q["route"], "route": d.route,
             "confidence": d.confidence, "reasoning": d.reasoning, "correct": d.route == q["route"],
             "source": q["source"], "calls": calls, "cost_usd": round(cost(calls), 6)}
        (runs / f"{q['id']}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
        results.append(r)
        all_calls += calls
        print(f"{q['id']} expected={q['route']:9s} got={d.route:9s} conf={d.confidence:.2f} "
              f"{'OK ' if r['correct'] else 'MISS'} {d.reasoning[:110]}")

    summary = grade(results)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    summary.update(model=MODEL, timestamp=datetime.now(timezone.utc).isoformat(),
                   git={"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))},
                   calls=len(all_calls), cost_usd=round(cost(all_calls), 5))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\noverall {summary['overall']}  by route {summary['by_route']}")
    print(f"misclassified: {summary['misclassified']}")
    print(f"ambiguous routed with confidence >= {CONFIDENT}: {summary['ambiguous_confidently_routed']}")
    print(f"cost ${summary['cost_usd']:.5f} ({summary['calls']} calls)")


if __name__ == "__main__":
    main()
