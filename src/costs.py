"""Measure OpenAI spend across all three projects in one process.

Project 1 calls the OpenAI SDK directly, Project 2 goes through instructor, and the
router uses instructor too. All of them end at the SDK's Completions.create /
Embeddings.create, so wrapping those two methods at runtime counts every call
without touching either project's code.
"""
from __future__ import annotations

from contextlib import contextmanager

from openai.resources.chat.completions import Completions
from openai.resources.embeddings import Embeddings

# USD per 1M tokens: (input, cached input, output)
PRICES = {"gpt-4o-mini": (0.15, 0.075, 0.60), "text-embedding-3-small": (0.02, 0.02, 0.0)}

CALLS: list[dict] = []
_label = {"current": None}
_installed = {"done": False}


@contextmanager
def label(name: str):
    prev, _label["current"] = _label["current"], name
    try:
        yield
    finally:
        _label["current"] = prev


def _record(model: str, usage) -> None:
    if usage is None:
        return
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    CALLS.append({"label": _label["current"], "model": model, "prompt_tokens": usage.prompt_tokens or 0,
                  "cached_tokens": cached, "completion_tokens": getattr(usage, "completion_tokens", 0) or 0})


def install() -> None:
    if _installed["done"]:
        return
    orig_chat, orig_embed = Completions.create, Embeddings.create

    def chat(self, *args, **kwargs):
        resp = orig_chat(self, *args, **kwargs)
        _record(getattr(resp, "model", kwargs.get("model", "")), getattr(resp, "usage", None))
        return resp

    def embed(self, *args, **kwargs):
        resp = orig_embed(self, *args, **kwargs)
        _record(kwargs.get("model", ""), getattr(resp, "usage", None))
        return resp

    Completions.create, Embeddings.create = chat, embed
    _installed["done"] = True


def cost(calls: list[dict]) -> float:
    total = 0.0
    for c in calls:
        key = next((k for k in PRICES if (c["model"] or "").startswith(k)), None)
        if key:
            p_in, p_cached, p_out = PRICES[key]
            total += ((c["prompt_tokens"] - c["cached_tokens"]) * p_in + c["cached_tokens"] * p_cached
                      + c["completion_tokens"] * p_out) / 1e6
    return total


def summarize(calls: list[dict]) -> dict:
    return {"calls": len(calls), "cost_usd": round(cost(calls), 5),
            "by_label": {lbl: round(cost([c for c in calls if c["label"] == lbl]), 5)
                         for lbl in dict.fromkeys(c["label"] for c in calls)}}
