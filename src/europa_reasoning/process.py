"""Reasoning-process metrics: how a model reasons, not only whether the answer is right.

All metrics are deterministic and computed from the generation alone. They are reported next to
accuracy so that a model which reaches answers by looping, running out of budget or switching
language is visible even when its accuracy looks acceptable.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .diagnostics import has_repeated_span
from .models import Generation, Item

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
_TAGS = ("<think>", "</think>")

# Unicode script of each language's letters (first declared script in the registry).
_SCRIPT_PREFIXES = {
    "Latn": ("LATIN",),
    "Cyrl": ("CYRILLIC",),
    "Grek": ("GREEK",),
    "Geor": ("GEORGIAN",),
}


def split_reasoning(text: str, *, prompt_opened: bool) -> tuple[str | None, str, bool]:
    """Split raw model text into (trace, final answer, trace_closed).

    ``prompt_opened`` is true when the chat template already ended the prompt with ``<think>``
    (the model then never emits the opener itself). A trace that never closes yields an empty
    answer: text produced inside an unfinished trace is not a final answer.
    """
    body = text
    opened = prompt_opened
    stripped = body.lstrip()
    if stripped.startswith(THINK_OPEN):
        opened = True
        body = stripped[len(THINK_OPEN):]
    if not opened:
        return None, text.strip(), False
    if THINK_CLOSE not in body:
        return body.strip(), "", False
    trace, answer = body.split(THINK_CLOSE, 1)
    return trace.strip(), answer.strip(), True


def script_share(text: str, script: str) -> float | None:
    """Share of letters in ``text`` that belong to ``script`` (None when there are no letters)."""
    prefixes = _SCRIPT_PREFIXES.get(script)
    if prefixes is None:
        return None
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return None
    hits = 0
    for ch in letters:
        name = unicodedata.name(ch, "")
        hits += name.startswith(prefixes)
    return hits / len(letters)


def detect_language(text: str) -> str | None:
    """Optional ISO 639-1 language guess for a trace (requires the ``langdetect`` extra)."""
    try:
        from langdetect import DetectorFactory, detect  # type: ignore[import-not-found]
    except ImportError:
        return None
    DetectorFactory.seed = 0
    sample = re.sub(r"[\d$\\{}^_=+\-*/()\[\]]+", " ", text)[:4000]
    if len(sample.split()) < 8:
        return None
    try:
        return str(detect(sample))
    except Exception:  # noqa: BLE001 - langdetect raises on undecidable input
        return None


def process_metrics(item: Item, generation: Generation, *, reasoning_enabled: bool) -> dict[str, Any]:
    """Return per-sample process evidence (never the trace text itself)."""
    trace = generation.reasoning
    answer = generation.text or ""
    metrics: dict[str, Any] = {
        "reasoning_enabled": reasoning_enabled,
        "trace_present": bool(trace),
        "trace_closed": None,
        "trace_loop": None,
        "trace_chars": len(trace) if trace else 0,
        "reasoning_tokens": generation.reasoning_tokens,
        "answer_present": bool(answer.strip()),
        "answer_chars": len(answer),
        "think_tag_in_answer": any(tag in answer for tag in _TAGS),
        "budget_exhausted": generation.finish_reason in {"length", "reasoning_unclosed"},
        "trace_script_share": None,
        "trace_language_detected": None,
    }
    if reasoning_enabled:
        metrics["trace_closed"] = bool(generation.raw and generation.raw.get("trace_closed"))
        if trace:
            metrics["trace_loop"] = has_repeated_span(trace, span_tokens=24)
            metrics["trace_script_share"] = script_share(trace, item.script)
            metrics["trace_language_detected"] = detect_language(trace)
    return metrics
