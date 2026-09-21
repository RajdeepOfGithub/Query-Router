"""Execute a routing decision against the real systems: routing and merging only.

  sql  -> Project 2 (../Text-to-SQL): sql_generate.answer_question, then its own
          confidence.assess (back-translation + sanity checks). Returned as-is.
  rag  -> Project 1 (../RAG/financial-rag): pipeline.search -> generation ->
          verify.verify_answer (citation verification) -> compute_confidence. Returned as-is.
  both -> call both, then merge the fact (SQL) and the reasoning (RAG) into one answer,
          every claim tagged [SQL] or [RAG-n] with a source list.
  ambiguous -> the classifier now emits "both" plus ambiguity_disclosure for these. Same as
          both, with the disclosure stated first. The branch is kept explicit.

Neither project's code is modified, and nothing here re-implements guardrails, citation
verification or hallucination detection: those run inside the called systems.

The merge is a deterministic template, not an LLM rewrite. A rewriting call would produce
new text that neither system has verified, which undoes Project 1's citation checks and
Project 2's back-translation.
"""
from __future__ import annotations

import contextlib
import dataclasses
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
RAG_ROOT = ROOT.parent / "RAG" / "financial-rag"
SQL_ROOT = ROOT.parent / "Text-to-SQL"

CALLED = {"sql": 0, "rag": 0}  # how many times each backend actually ran (tests assert on this)


# ------------------------------------------------------------ backend adapters

def _add_path(p: Path) -> None:
    if str(p) not in sys.path:
        sys.path.insert(1, str(p))  # after our own src/, so router module names win


@contextlib.contextmanager
def _in_rag_root():
    """Project 1 resolves chroma_db/, data/processed/bm25.pkl and .env relative to the CWD."""
    with contextlib.chdir(RAG_ROOT):
        yield


def run_sql(question: str) -> dict:
    _add_path(SQL_ROOT / "src")
    from confidence import assess
    from sql_generate import answer_question

    CALLED["sql"] += 1
    answer = answer_question(question)
    return {"answer": answer, "confidence": assess(answer)}


def run_rag(question: str):
    _add_path(RAG_ROOT / "src")
    with _in_rag_root():
        import generator
        from pipeline import search
        from verify import verify_answer

        CALLED["rag"] += 1
        # Same sequence as generator.py's __main__, Project 1's own CLI answer path,
        # with its defaults (n=20, top_k=5, default collection, verification on).
        results = search(question, n=20, top_k=5)
        context, citation_map = generator.build_context(results)
        answer_text = generator.call_llm(generator.SYSTEM_PROMPT, f"{context}\n\nQuestion: {question}")
        numbers = generator.parse_citations(answer_text)
        result = generator.AnswerResult(
            question=question, answer_text=answer_text,
            citations=generator.resolve_citations(numbers, citation_map, results), retrieved_chunks=results)
        verify_answer(result)
        result.confidence = generator.compute_confidence(result)
        result.declined = generator.is_declined(result)
        return result


# ------------------------------------------------------------ merge

_MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(billion|million|bn|B|M)\b", re.I)
_SCALE = {"billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "m": 1e6}
CONFLICT_BAND = 0.10


def _stated_figures(text: str) -> list[tuple[str, float, float]]:
    """(raw text, value in USD, half a unit of the stated precision) for each $ figure in text."""
    out = []
    for m in _MONEY.finditer(text):
        num, unit = m.group(1).replace(",", ""), _SCALE[m.group(2).lower()]
        decimals = len(num.split(".")[1]) if "." in num else 0
        out.append((m.group(0), float(num) * unit, 0.5 * unit * 10 ** -decimals))
    return out


def _period(start, end) -> str:
    return f"{start} to {end}" if start else f"as of {end}"


def _sql_values(sql: dict) -> list[dict]:
    """Every figure the SQL side returned: {value, concept, period}."""
    a = sql["answer"]
    vals = []
    if a.measure is not None:
        for m in [a.measure, *a.alternatives]:
            vals.append({"value": m.value_numeric, "concept": m.concept, "period": _period(m.period_start, m.period_end)})
    elif a.rows and "value_numeric" in a.columns:
        for r in (dict(zip(a.columns, row)) for row in a.rows):
            if r["value_numeric"] is not None:
                vals.append({"value": r["value_numeric"], "concept": r.get("concept", "?"),
                             "period": _period(r.get("period_start"), r.get("period_end", "?"))})
    return vals


def find_conflicts(sql: dict, rag_text: str) -> list[dict]:
    """TIE-BREAK RULE: when the RAG answer states a dollar figure that is close to an SQL
    value (within CONFLICT_BAND) but disagrees with it beyond the RAG figure's own rounding,
    both are describing the same number and the SQL figure wins. It comes from structured,
    guardrailed XBRL facts; the RAG figure is an LLM's restatement of a retrieved passage.
    RAG figures that match no SQL value at all are reported as "unmatched" rather than
    silently dropped: they may be a different line item, or the same item stated
    differently. The reader is told the structured figures take precedence.
    Heuristic limits: nearest-value pairing ignores which period the RAG text attaches to
    its figure, so a conflict note can pair figures from different periods (the SQL period
    is printed so a reader can see this); figures written in words aren't parsed."""
    conflicts = []
    sql_vals = [v for v in _sql_values(sql) if v["value"]]
    for raw, stated, half_step in _stated_figures(rag_text):
        if any(abs(stated - v["value"]) <= half_step for v in sql_vals):
            continue  # consistent with a structured figure at the RAG figure's own precision
        near = [v for v in sql_vals if abs(stated - v["value"]) / abs(v["value"]) <= CONFLICT_BAND]
        if near:
            v = min(near, key=lambda v: abs(stated - v["value"]))
            conflicts.append({"kind": "conflict", "rag_figure": raw, "rag_value": stated, "sql_value": v["value"],
                              "sql_concept": v["concept"], "sql_period": v["period"],
                              "resolution": "SQL figure used; RAG figure superseded"})
        else:
            conflicts.append({"kind": "unmatched", "rag_figure": raw, "rag_value": stated,
                              "resolution": "not confirmed by structured data; SQL figures take precedence "
                                            "where they describe the same item"})
    return conflicts


def _fmt_usd(v: float) -> str:
    return f"${v / 1e6:,.0f}M" if abs(v) >= 1e6 else f"${v:,.2f}"


def _sql_source_line(sql: dict) -> str:
    a, conf = sql["answer"], sql["confidence"]
    if a.measure is not None:
        m = a.measure
        dims = ", ".join(m.dimension_labels.values()) or "consolidated"
        period = f"{m.period_start} to {m.period_end}" if m.period_start else f"as of {m.period_end}"
        flag = "FLAGGED: " + "; ".join(conf.reasons) if conf.flagged else "not flagged"
        return (f"[SQL] Text-to-SQL over XBRL facts: {m.concept} ({dims}), {period}; "
                f"status={a.status}, confidence={conf.score} ({flag})")
    flag = ("FLAGGED: " + "; ".join(conf.reasons)) if conf.flagged else "not flagged"
    return (f"[SQL] Text-to-SQL over XBRL facts: status={a.status}, confidence={conf.score} ({flag})"
            + (f"; query: {a.sql}" if a.sql else ""))


def _rag_source_lines(rag) -> list[str]:
    supported = {}
    for v in rag.citation_verification or []:
        supported.setdefault(v["number"], []).append(v["supported"])
    lines = []
    for c in rag.citations:
        md = c["metadata"] or {}
        where = ", ".join(x for x in (md.get("form"), md.get("period"), md.get("section")) if x and x != "none")
        verdicts = supported.get(c["number"], [])
        check = ("verified" if verdicts and all(verdicts) else
                 "NOT supported by passage" if verdicts else "not checked")
        lines.append(f"[RAG-{c['number']}] financial-rag passage {c['chunk_id']} ({where}); citation {check}")
    return lines


def merge(question: str, sql: dict, rag, disclosure: str | None = None) -> tuple[str, list[dict]]:
    a = sql["answer"]
    rag_text = re.sub(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]",
                      lambda m: "[" + ", ".join(f"RAG-{n.strip()}" for n in m.group(1).split(",")) + "]",
                      rag.answer_text.strip())
    conflicts = find_conflicts(sql, rag.answer_text) if a.status == "answered" else []

    parts = []
    if disclosure:
        parts.append(f"Note on the question: {disclosure}")
    if a.status == "answered" and a.measure is None and a.rows:
        rows = "\n".join(f"- {v['concept']}, {v['period']}: {_fmt_usd(v['value'])}" for v in _sql_values(sql))
        parts.append(f"The figures, from structured XBRL data [SQL]:\n{rows}")
    elif a.status == "answered":
        parts.append(f"The figure, from structured XBRL data [SQL]:\n{a.text}")
    else:
        parts.append(f"The figure could not be retrieved from structured data [SQL]: {a.text}")
    if getattr(rag, "declined", False):
        parts.append(f"Explanation, from the filings and earnings call: the RAG system declined. {rag_text}")
    else:
        parts.append(f"Explanation, from the filings and earnings call:\n{rag_text}")
    verdicts = [v["supported"] for v in rag.citation_verification or []]
    if verdicts and not all(verdicts):
        parts.append(f"Caution: financial-rag's own citation check did not confirm {verdicts.count(False)} of "
                     f"{len(verdicts)} cited claims in the explanation.")
    for c in conflicts:
        if c["kind"] == "conflict":
            parts.append(f"Figure conflict: the explanation states {c['rag_figure']}; the structured data has "
                         f"{_fmt_usd(c['sql_value'])} for {c['sql_concept']} ({c['sql_period']}). The structured "
                         f"[SQL] figure is used.")
    unmatched = [c["rag_figure"] for c in conflicts if c["kind"] == "unmatched"]
    if unmatched:
        parts.append(f"The explanation also cites {', '.join(unmatched)}, which don't match any structured figure "
                     f"above. Where they describe the same item, the [SQL] figures take precedence.")
    parts.append("Sources:\n" + "\n".join([_sql_source_line(sql), *_rag_source_lines(rag)]))
    return "\n\n".join(parts), conflicts


# ------------------------------------------------------------ dispatch

@dataclass
class ExecutionResult:
    route: str
    answer: str
    sql_raw: dict | None = None
    rag_raw: object | None = None
    conflicts: list[dict] = field(default_factory=list)


def execute(question: str, route: str, disclosure: str | None = None) -> ExecutionResult:
    if route == "sql":
        sql = run_sql(question)
        return ExecutionResult("sql", sql["answer"].text, sql_raw=sql)
    if route == "rag":
        rag = run_rag(question)
        return ExecutionResult("rag", rag.answer_text, rag_raw=rag)
    if route in ("both", "ambiguous"):
        sql, rag = run_sql(question), run_rag(question)
        text, conflicts = merge(question, sql, rag, disclosure)
        return ExecutionResult(route, text, sql_raw=sql, rag_raw=rag, conflicts=conflicts)
    raise ValueError(f"unknown route {route!r}")


def jsonable(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        d = {f.name: jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        d.update({k: jsonable(v) for k, v in vars(obj).items() if k not in d})  # e.g. rag.declined
        return d
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    return obj
