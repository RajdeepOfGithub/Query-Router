"""Period-first conflict matching between RAG-stated figures and SQL figures (deterministic)."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from execute import find_conflicts  # noqa: E402

COLS = ["concept", "dimensions_json", "period_start", "period_end", "value_numeric", "unit", "source_document"]
Q2_25 = ("us-gaap:InvestmentBankingRevenue", "{}", "2025-04-01", "2025-06-30", 2_499e6, "iso4217:USD", "10-Q")
Q2_26 = ("us-gaap:InvestmentBankingRevenue", "{}", "2026-04-01", "2026-06-30", 3_208e6, "iso4217:USD", "10-Q")


def sql(*rows):
    return {"answer": SimpleNamespace(status="answered", measure=None, alternatives=[], rows=list(rows), columns=COLS)}


def by_fig(conflicts):
    return {c["rag_figure"]: c for c in conflicts}


def test_fees_example_q2_2025_figure_not_paired_with_q2_2026():
    """The Phase 2 bug: $3,277M (stated for Q2 2025) was paired with SQL's Q2 2026 $3,208M by proximity."""
    text = "Investment banking fees grew by 30%, increasing from $3,277 million in Q2 2025 to $4,761 million in Q2 2026."
    c = by_fig(find_conflicts(sql(Q2_25, Q2_26), text))
    assert c["$3,277 million"]["rag_period"] == "2025-04-01 to 2025-06-30"
    assert c["$3,277 million"]["kind"] == "unmatched_same_period"   # compared with Q2 2025 ($2,499M), not Q2 2026
    assert c["$4,761 million"]["rag_period"] == "2026-04-01 to 2026-06-30"
    assert not any(x.get("sql_period") == "2026-04-01 to 2026-06-30" and x["rag_figure"] == "$3,277 million"
                   for x in c.values())


def test_rag_period_absent_from_sql_is_period_mismatch_not_conflict():
    c = by_fig(find_conflicts(sql(Q2_26), "fees were $3,277 million in Q2 2025"))
    assert c["$3,277 million"]["kind"] == "period_mismatch"  # old code: 'conflict' with Q2 2026


def test_same_period_disagreement_is_conflict_and_sql_wins():
    c = by_fig(find_conflicts(sql(Q2_26), "fees reached $3.3 billion in the second quarter of 2026"))
    assert c["$3.3 billion"]["kind"] == "conflict" and c["$3.3 billion"]["sql_value"] == 3_208e6


def test_same_period_consistent_at_stated_precision_is_not_flagged():
    assert find_conflicts(sql(Q2_26), "fees were $3.2 billion for the three months ended June 30, 2026") == []


def test_no_period_falls_back_but_is_labeled_unconfirmed():
    c = by_fig(find_conflicts(sql(Q2_26), "fees were $3,277 million"))
    assert c["$3,277 million"]["kind"] == "conflict_period_unconfirmed"
    assert c["$3,277 million"]["period_basis"] == "period unconfirmed"


def test_filing_metadata_period_is_context_not_match_basis():
    """A 10-Q chunk's metadata says Q2-2026 even when the figure is Q2 2025; it must not decide the match."""
    c = by_fig(find_conflicts(sql(Q2_26), "fees were $3,277 million in Q2 2025", filing_periods=["Q2-2026"]))
    assert c["$3,277 million"]["kind"] == "period_mismatch" and c["$3,277 million"]["rag_filing_periods"] == ["Q2-2026"]
