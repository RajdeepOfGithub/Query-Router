"""Route a question to the SQL (XBRL facts) backend, the RAG (filing prose +
earnings call) backend, both, or flag it as ambiguous.

Classification only: nothing here calls either backend.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You route questions about JPMorgan Chase to the backend that can answer them.

BACKENDS
- sql: a database of XBRL facts tagged in the 10-K FY2025 and 10-Q Q2 2026: reported financial
  statement line items and disclosures (revenue, expenses, net income, EPS, balances, segment
  figures), each with an exact period. It returns numbers. It cannot explain anything.
- rag: text retrieval over the same filings' prose (MD&A explanations, event descriptions) and the
  Q2 2026 earnings call transcript (management commentary, quotes, guidance, outlook). It explains
  causes and context, and holds numbers that are NOT tagged facts: guidance, targets, figures that
  appear only in commentary.

ROUTES
- sql: the question asks only for one or more reported figures.
- rag: the question asks for explanation, drivers, events, commentary, quotes or outlook. A number
  in the question or answer does not make it sql: guidance and figures stated only in commentary
  are rag.
- both: the question explicitly asks for a reported figure (or the size of a change in one) AND
  for the reasons or context behind it. Both parts must be requested. Topic overlap is not enough.
- ambiguous: the question can't be routed without guessing. Either reasonable readings need
  different routes, or it doesn't pin down what is being asked for (which measure, which sense of a
  vague word like "earn", "do", "big", "concern"), so any route would rest on an assumption the user
  didn't make. Use this instead of choosing a route when you would have to guess.

CONFIDENCE: your probability (0-1) that the chosen route is the one a careful analyst would pick.
If you can construct a reasonable reading that needs a different route, your confidence in a
single route should be low, and "ambiguous" is probably the right route."""

Route = Literal["sql", "rag", "both", "ambiguous"]


class RouteDecision(BaseModel):
    route: Route
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="one or two sentences: why this route")


@lru_cache(maxsize=1)
def _client():
    import instructor
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(ROOT / ".env")
    return instructor.from_openai(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))


def classify(question: str, on_response=None) -> RouteDecision:
    client = _client()
    if on_response is not None:
        client.on("completion:response", on_response)
    try:
        return client.chat.completions.create(
            model=MODEL, temperature=0, response_model=RouteDecision, max_retries=2,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
        )
    finally:
        if on_response is not None:
            client.off("completion:response", on_response)
