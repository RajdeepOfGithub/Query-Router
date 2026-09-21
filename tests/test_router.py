"""Phase 2: routing wired to the real systems (calls gpt-4o-mini, Project 1's index, Project 2's DB).

Each test checks the backends that actually ran (execute.CALLED counters), not only the
classifier's label. The total measured cost of the run is written to baseline/phase2_test_cost.json.
"""
import json
import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import costs  # noqa: E402
import execute  # noqa: E402
from router import answer  # noqa: E402

load_dotenv(ROOT / ".env")
pytestmark = pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")


@pytest.fixture(scope="session", autouse=True)
def record_cost():
    start = len(costs.CALLS)
    yield
    summary = costs.summarize(costs.CALLS[start:])
    (ROOT / "baseline").mkdir(exist_ok=True)
    (ROOT / "baseline" / "phase2_test_cost.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


@pytest.fixture(autouse=True)
def reset_counters():
    execute.CALLED.update(sql=0, rag=0)


def test_sql_only_end_to_end():
    r = answer("What was total net revenue for the second quarter of 2026?")
    assert r.route == "sql"
    assert execute.CALLED == {"sql": 1, "rag": 0} and r.rag_raw is None  # Project 1 never called
    assert "$57,347M" in r.answer
    assert r.sql_raw["answer"].measure.concept == "us-gaap:RevenuesNetOfInterestExpense"
    assert r.sql_raw["confidence"] is not None  # Project 2's own hallucination check ran


def test_rag_only_end_to_end():
    r = answer("Why did investment banking fees increase in the second quarter of 2026?")
    assert r.route == "rag"
    assert execute.CALLED == {"sql": 0, "rag": 1} and r.sql_raw is None  # Project 2 never called
    text = r.answer.lower()
    # Project 1 ground truth (q09): underwriting (equity/debt) and advisory fee drivers
    assert "underwriting" in text and "advisory" in text, r.answer
    assert r.rag_raw.citations and r.rag_raw.citation_verification is not None  # Project 1's verification ran


def test_both_calls_both_and_attributes_sources():
    r = answer("By how much did investment banking fees grow in Q2 2026 versus Q2 2025, and what drove the growth?")
    assert r.route == "both"
    assert execute.CALLED == {"sql": 1, "rag": 1}  # both actually executed, not just classified
    assert r.sql_raw["answer"].status == "answered" and r.rag_raw.answer_text
    # content traceable to each system
    assert "[SQL]" in r.answer and "$3,208M" in r.answer  # SQL-sourced figure (Q2 2026 InvestmentBankingRevenue)
    assert "[RAG-" in r.answer
    for c in r.rag_raw.citations:
        assert f"[RAG-{c['number']}] financial-rag passage {c['chunk_id']}" in r.answer
