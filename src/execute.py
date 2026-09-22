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


def _stated_figures(text: str) -> list[dict]:
    """Each $ figure in text: raw text, value in USD, half a unit of its stated precision, span."""
    out = []
    for m in _MONEY.finditer(text):
        num, unit = m.group(1).replace(",", ""), _SCALE[m.group(2).lower()]
        decimals = len(num.split(".")[1]) if "." in num else 0
        out.append({"raw": m.group(0), "value": float(num) * unit, "half_step": 0.5 * unit * 10 ** -decimals,
                    "span": m.span()})
    return out


# ---- period extraction for a RAG figure: from the answer text around the figure, never from chunk
# metadata. A Project 1 chunk's `period` field is the FILING's period (every 10-Q chunk says Q2-2026,
# including its Q2 2025 comparison columns), so matching on it would reproduce the period-blind
# pairing this is meant to prevent. The filing period is reported only as context.
_QEND = {1: ("01-01", "03-31"), 2: ("04-01", "06-30"), 3: ("07-01", "09-30"), 4: ("10-01", "12-31")}
_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4}
_MONTH_Q = {"march": 1, "june": 2, "september": 3, "december": 4}
_PERIOD_PATTERNS = [
    (re.compile(r"\bQ([1-4])[\s'-]*(20\d{2}|\d{2})\b", re.I), lambda m: ("q", int(m[1]), m[2])),
    (re.compile(r"\b([1-4])Q[\s'-]*(20\d{2}|\d{2})\b", re.I), lambda m: ("q", int(m[1]), m[2])),
    (re.compile(r"\b(first|second|third|fourth) quarter(?: of)?(?: fiscal)?(?: year)? (20\d{2})\b", re.I),
     lambda m: ("q", _ORD[m[1].lower()], m[2])),
    (re.compile(r"\bthree months ended (march|june|september|december) 3[01],? (20\d{2})\b", re.I),
     lambda m: ("q", _MONTH_Q[m[1].lower()], m[2])),
    (re.compile(r"\bsix months ended june 30,? (20\d{2})\b", re.I), lambda m: ("h1", 0, m[1])),
    (re.compile(r"\b(?:FY|fiscal year|full[- ]year|year ended december 31,?)\s*(20\d{2})\b", re.I),
     lambda m: ("fy", 0, m[1])),
    (re.compile(r"\b(?:in|for|during) (20\d{2})\b", re.I), lambda m: ("fy", 0, m[1])),
]


def _to_period(kind: str, q: int, year: str) -> tuple[str, str]:
    y = int(year) + (2000 if len(year) == 2 else 0)
    if kind == "q":
        s, e = _QEND[q]
        return f"{y}-{s}", f"{y}-{e}"
    if kind == "h1":
        return f"{y}-01-01", f"{y}-06-30"
    return f"{y}-01-01", f"{y}-12-31"


def _periods_in(text: str) -> list[tuple[int, tuple]]:
    found = {}
    for pat, parse in _PERIOD_PATTERNS:
        for m in pat.finditer(text):
            found.setdefault(m.start(), _to_period(*parse(m)))  # earlier (more specific) pattern wins
    return sorted(found.items())


def figure_period(text: str, fig: dict, figures: list[dict]) -> tuple | None:
    """Period the answer text attaches to a figure: the first period mentioned after it (up to the next
    figure or sentence end), else the last one mentioned before it (back to the previous figure or
    sentence start). None if the text states no period for this figure."""
    start, end = fig["span"]
    nxt = min([f["span"][0] for f in figures if f["span"][0] > start] + [len(text)])
    after = re.split(r"(?<=[.;])\s", text[end:nxt], maxsplit=1)[0][:80]
    hits = _periods_in(after)
    if hits:
        return hits[0][1]
    prev = max([f["span"][1] for f in figures if f["span"][1] <= start] + [0])
    before = re.split(r"[.;]\s", text[prev:start])[-1]
    hits = _periods_in(before)
    return hits[-1][1] if hits else None


def _period(start, end) -> str:
    return f"{start} to {end}" if start else f"as of {end}"


def _sql_values(sql: dict) -> list[dict]:
    """Every figure the SQL side returned: {value, concept, period, key=(start, end)}."""
    a = sql["answer"]
    vals = []
    if a.measure is not None:
        for m in [a.measure, *a.alternatives]:
            vals.append({"value": m.value_numeric, "concept": m.concept, "key": (m.period_start, m.period_end),
                         "period": _period(m.period_start, m.period_end)})
    elif a.rows and "value_numeric" in a.columns:
        for r in (dict(zip(a.columns, row)) for row in a.rows):
            if r["value_numeric"] is not None:
                key = (r.get("period_start"), r.get("period_end"))
                vals.append({"value": r["value_numeric"], "concept": r.get("concept", "?"), "key": key,
                             "period": _period(*key)})
    return vals


def find_conflicts(sql: dict, rag_text: str, filing_periods: list[str] | None = None) -> list[dict]:
    """TIE-BREAK RULE: when the RAG answer states a dollar figure that disagrees with an SQL figure
    for the same thing, the SQL figure wins. It comes from structured, guardrailed XBRL facts; the RAG
    figure is an LLM's restatement of a retrieved passage.

    Matching is by PERIOD FIRST, value proximity only within a period:
      1. The RAG figure's period is read from the answer text around it (figure_period).
      2. Period stated:
           - consistent with an SQL figure for that period (at the RAG figure's rounding): fine
           - an SQL figure for that period within CONFLICT_BAND: "conflict", SQL wins
           - SQL figures for that period, none close: "unmatched_same_period" (likely a different
             measure, e.g. a segment or gross figure; not presented as a conflict)
           - no SQL figure for that period: "period_mismatch", NOT compared with other periods
      3. No period stated: fall back to nearest-value matching, labeled "period unconfirmed", so
         it's never presented as equivalent to a period-matched comparison.
    Concept can't be extracted from RAG chunk metadata (chunks carry form/period/section, no
    concept), so within a period the concept match is still by value proximity.
    Limits: periods written unusually ("last quarter") aren't parsed and count as unconfirmed;
    figures written in words aren't parsed at all."""
    conflicts = []
    sql_vals = [v for v in _sql_values(sql) if v["value"]]
    sql_periods = sorted({v["period"] for v in sql_vals})
    figures = _stated_figures(rag_text)
    for fig in figures:
        stated, raw = fig["value"], fig["raw"]
        per = figure_period(rag_text, fig, figures)
        base = {"rag_figure": raw, "rag_value": stated, "rag_period": _period(*per) if per else None,
                "period_basis": "stated in answer text" if per else "period unconfirmed",
                "rag_filing_periods": filing_periods or []}
        pool = [v for v in sql_vals if per is None or v["key"] == per]
        if per is not None and not pool:
            conflicts.append({**base, "kind": "period_mismatch", "sql_periods": sql_periods,
                              "resolution": "not compared: no structured figure for this period"})
            continue
        if any(abs(stated - v["value"]) <= fig["half_step"] for v in pool):
            continue  # consistent with a structured figure (for the same period, if one was stated)
        near = [v for v in pool if abs(stated - v["value"]) / abs(v["value"]) <= CONFLICT_BAND]
        if near:
            v = min(near, key=lambda v: abs(stated - v["value"]))
            conflicts.append({**base, "kind": "conflict" if per else "conflict_period_unconfirmed",
                              "sql_value": v["value"], "sql_concept": v["concept"], "sql_period": v["period"],
                              "resolution": "SQL figure used; RAG figure superseded"})
        else:
            conflicts.append({**base, "kind": "unmatched_same_period" if per else "unmatched",
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
    filing_periods = sorted({(c["metadata"] or {}).get("period") for c in rag.citations} - {None})
    conflicts = find_conflicts(sql, rag.answer_text, filing_periods) if a.status == "answered" else []

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
    notes = []
    for c in conflicts:
        fig = f"{c['rag_figure']} ({c['rag_period']})" if c["rag_period"] else c["rag_figure"]
        if c["kind"] == "conflict":
            notes.append(f"Figure conflict, same period: the explanation states {fig}; the structured data has "
                         f"{_fmt_usd(c['sql_value'])} for {c['sql_concept']} ({c['sql_period']}). The [SQL] figure is used.")
        elif c["kind"] == "conflict_period_unconfirmed":
            notes.append(f"Possible figure conflict, period unconfirmed: the explanation states {c['rag_figure']} "
                         f"without a period; the closest structured figure is {_fmt_usd(c['sql_value'])} for "
                         f"{c['sql_concept']} ({c['sql_period']}). Not confirmed to be the same period. The [SQL] "
                         f"figure is used.")
        elif c["kind"] == "period_mismatch":
            notes.append(f"Not compared: the explanation states {fig}, a period the structured data here doesn't "
                         f"cover ({'; '.join(c['sql_periods'])}).")
        elif c["kind"] == "unmatched_same_period":
            notes.append(f"Unconfirmed: the explanation states {fig}, which matches no structured figure for that "
                         f"period (possibly a different measure). Where it describes the same item, the [SQL] "
                         f"figure takes precedence.")
        else:
            notes.append(f"Unconfirmed: the explanation states {c['rag_figure']} (no period stated), which matches "
                         f"no structured figure. Where it describes the same item, the [SQL] figures take precedence.")
    if notes:
        filing = f" (explanation drawn from {', '.join(filing_periods)} filings)" if filing_periods else ""
        parts.append(f"Figure check against structured data{filing}:\n" + "\n".join(f"- {n}" for n in notes))
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
