"""Aggregate run evidence without hiding coverage gaps."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from .diagnostics import has_prompt_echo, has_repeated_span
from .models import Item


def summarize(samples: list[dict[str, Any]], items: Iterable[Item]) -> dict[str, Any]:
    item_by_id = {item.id: item for item in items}
    scored = [sample for sample in samples if isinstance(sample.get("score"), (int, float))]
    failed_generations = [sample for sample in samples if sample.get("error")]
    unjudged = [sample for sample in samples if sample.get("score") is None and not sample.get("error")]
    malformed = [
        sample
        for sample in samples
        if sample.get("score_details", {}).get("malformed") is True
    ]

    summary: dict[str, Any] = {
        "counts": {
            "scheduled": len(samples),
            "scored": len(scored),
            "unjudged": len(unjudged),
            "generation_errors": len(failed_generations),
            "malformed": len(malformed),
        },
        "overall": _group_summary(scored),
        "domains": _summaries_by(samples=scored, item_by_id=item_by_id, attribute="domain"),
        "capabilities": _summaries_by(
            samples=scored, item_by_id=item_by_id, attribute="capability"
        ),
        "task_types": _summaries_by(
            samples=scored, item_by_id=item_by_id, attribute="task_type"
        ),
        "languages": _summaries_by(
            samples=scored, item_by_id=item_by_id, attribute="language"
        ),
        "contrast_sets": _contrast_summary(samples=samples, item_by_id=item_by_id),
    }
    domain_scores = [row["score"] for row in summary["domains"] if row["score"] is not None]
    summary["capability_profile"] = {
        "macro_domain_score": statistics.fmean(domain_scores) if domain_scores else None,
        "minimum_domain_score": min(domain_scores) if domain_scores else None,
        "covered_domains": len(domain_scores),
        "declared_domains": len({item.domain for item in item_by_id.values()}),
    }
    language_scores = [
        row["score"] for row in summary["languages"] if row["score"] is not None
    ]
    summary["language_profile"] = {
        "macro_language_score": statistics.fmean(language_scores) if language_scores else None,
        "minimum_language_score": min(language_scores) if language_scores else None,
        "covered_languages": len(language_scores),
        "declared_languages": len({item.language for item in item_by_id.values()}),
    }
    latencies = [float(sample["latency_ms"]) for sample in samples if sample.get("latency_ms") is not None]
    summary["latency"] = {
        "median_ms": statistics.median(latencies) if latencies else None,
        "p95_ms": _percentile(latencies, 0.95) if latencies else None,
    }
    response_entries = [
        (str(sample["item_id"]), str(sample["response"]))
        for sample in samples
        if isinstance(sample.get("response"), str) and str(sample["response"]).strip()
    ]
    prompt_echoes = [
        item_id
        for item_id, response in response_entries
        if has_prompt_echo(response, item=item_by_id.get(item_id))
    ]
    repetitions = [
        item_id for item_id, response in response_entries if has_repeated_span(response)
    ]
    output_tokens = [
        int(sample["output_tokens"])
        for sample in samples
        if isinstance(sample.get("output_tokens"), (int, float))
    ]
    summary["output_diagnostics"] = {
        "responses_checked": len(response_entries),
        "prompt_echo_count": len(prompt_echoes),
        "prompt_echo_rate": (
            len(prompt_echoes) / len(response_entries) if response_entries else None
        ),
        "prompt_echo_item_ids": list(dict.fromkeys(prompt_echoes)),
        "repeated_span_count": len(repetitions),
        "repeated_span_rate": (
            len(repetitions) / len(response_entries) if response_entries else None
        ),
        "repeated_span_item_ids": list(dict.fromkeys(repetitions)),
        "median_output_tokens": statistics.median(output_tokens) if output_tokens else None,
        "p95_output_tokens": _percentile(output_tokens, 0.95) if output_tokens else None,
        "total_output_tokens": sum(output_tokens) if output_tokens else None,
    }
    gated_samples = [
        sample
        for sample in scored
        if sample.get("score_details", {}).get("critical_gate_applied") is True
    ]
    dimension_failures: dict[str, int] = defaultdict(int)
    for sample in gated_samples:
        for dimension in sample.get("score_details", {}).get("critical_failures", {}):
            dimension_failures[str(dimension)] += 1
    summary["critical_gates"] = {
        "failure_count": len(gated_samples),
        "failure_rate": len(gated_samples) / len(scored) if scored else None,
        "item_ids": [str(sample["item_id"]) for sample in gated_samples],
        "dimension_counts": dict(sorted(dimension_failures.items())),
    }
    summary["process"] = process_summary(samples=samples, item_by_id=item_by_id)
    return summary


def process_summary(samples: list[dict[str, Any]], item_by_id: dict[str, Item]) -> dict[str, Any]:
    """Aggregate reasoning-process evidence overall and per language."""
    def block(group: list[dict[str, Any]]) -> dict[str, Any]:
        rows = [sample.get("process") or {} for sample in group if not sample.get("error")]
        reasoning = [row for row in rows if row.get("reasoning_enabled")]
        closed = [row for row in reasoning if row.get("trace_closed")]
        tokens = [int(row["reasoning_tokens"]) for row in closed if row.get("reasoning_tokens") is not None]
        script = [float(row["trace_script_share"]) for row in reasoning if row.get("trace_script_share") is not None]
        def rate(values: list[Any]) -> float | None:
            return sum(bool(v) for v in values) / len(values) if values else None
        scored = [s for s in group if isinstance(s.get("score"), (int, float))]
        closed_scored = [s for s in scored if (s.get("process") or {}).get("trace_closed")]
        return {
            "n": len(rows),
            "reasoning_n": len(reasoning),
            "trace_closure_rate": rate([row.get("trace_closed") for row in reasoning]),
            "trace_loop_rate": rate([row.get("trace_loop") for row in reasoning if row.get("trace_present")]),
            "budget_exhausted_rate": rate([row.get("budget_exhausted") for row in rows]),
            "answer_present_rate": rate([row.get("answer_present") for row in rows]),
            "think_tag_in_answer_rate": rate([row.get("think_tag_in_answer") for row in rows]),
            "median_reasoning_tokens_closed": statistics.median(tokens) if tokens else None,
            "p90_reasoning_tokens_closed": _percentile(tokens, 0.9) if tokens else None,
            "mean_trace_script_share": statistics.fmean(script) if script else None,
            "accuracy_when_closed": (
                statistics.fmean(float(s["score"]) for s in closed_scored) if closed_scored else None
            ),
        }
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        item = item_by_id.get(str(sample.get("item_id")))
        if item:
            by_language[item.language].append(sample)
    return {
        "overall": block(samples),
        "languages": [{"id": code, **block(group)} for code, group in sorted(by_language.items())],
    }


def render_text_report(run: dict[str, Any]) -> str:
    summary = run["summary"]
    profile = summary["capability_profile"]
    overall = summary["overall"]
    lines = [
        f"Run: {run['run_id']}",
        f"Model: {run['model']['id']} ({run['model']['backend']})",
        f"Status: {run['status']}",
        f"Scored: {summary['counts']['scored']}/{summary['counts']['scheduled']}",
        f"Micro score: {_percent(overall.get('score'))}",
        f"Macro domain score: {_percent(profile.get('macro_domain_score'))}",
        f"Minimum domain score: {_percent(profile.get('minimum_domain_score'))}",
        f"Macro language score: {_percent(summary['language_profile'].get('macro_language_score'))}",
        f"Languages covered: {summary['language_profile']['covered_languages']}/"
        f"{summary['language_profile']['declared_languages']}",
        "",
        "Domains",
    ]
    for row in summary["domains"]:
        lines.append(
            f"  {row['id']:<24} {_percent(row['score']):>8}  n={row['n']}  "
            f"95% CI {_interval(row['ci95'])}"
        )
    contrast = summary["contrast_sets"]
    lines.extend(
        [
            "",
            "Contrast sets",
            f"  Complete pairs: {contrast['complete_pairs']}",
            f"  Base score: {_percent(contrast.get('base_score'))}",
            f"  Challenge score: {_percent(contrast.get('challenge_score'))}",
            f"  Robustness gap: {_percentage_points(contrast.get('robustness_gap'))}",
        ]
    )
    if summary["counts"]["unjudged"]:
        lines.append(f"\nUnjudged rubric items: {summary['counts']['unjudged']}")
    diagnostics = summary.get("output_diagnostics", {})
    if diagnostics:
        lines.extend(
            [
                "",
                "Output diagnostics",
                f"  Prompt echoes: {diagnostics.get('prompt_echo_count', 0)}",
                f"  Repeated spans: {diagnostics.get('repeated_span_count', 0)}",
                f"  Median output tokens: {diagnostics.get('median_output_tokens', '—')}",
            ]
        )
    process = summary.get("process", {}).get("overall", {})
    if process.get("reasoning_n"):
        lines.extend(
            [
                "",
                "Reasoning process",
                f"  Trace closure: {_percent(process.get('trace_closure_rate'))}",
                f"  Loops (repeated 24-token span): {_percent(process.get('trace_loop_rate'))}",
                f"  Budget exhausted: {_percent(process.get('budget_exhausted_rate'))}",
                f"  Median reasoning tokens (closed): {process.get('median_reasoning_tokens_closed', '—')}",
                f"  Trace in prompt script: {_percent(process.get('mean_trace_script_share'))}",
                f"  Accuracy when closed: {_percent(process.get('accuracy_when_closed'))}",
            ]
        )
    gates = summary.get("critical_gates", {})
    if gates:
        lines.extend(
            [
                "",
                "Critical gates",
                f"  Failed items: {gates.get('failure_count', 0)}",
                "  Dimensions: "
                + (", ".join(
                    f"{name}={count}"
                    for name, count in gates.get("dimension_counts", {}).items()
                ) or "none"),
            ]
        )
    return "\n".join(lines)


def _summaries_by(
    samples: list[dict[str, Any]], item_by_id: dict[str, Item], attribute: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        item = item_by_id.get(str(sample.get("item_id")))
        if item:
            groups[str(getattr(item, attribute))].append(sample)
    return [
        {"id": group_id, **_group_summary(group_samples)}
        for group_id, group_samples in sorted(groups.items())
    ]


def _group_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(sample["score"]) for sample in samples if sample.get("score") is not None]
    if not values:
        return {"score": None, "n": 0, "ci95": [None, None], "pass_rate": None}
    passes = [sample.get("passed") for sample in samples if sample.get("passed") is not None]
    return {
        "score": statistics.fmean(values),
        "n": len(values),
        "ci95": list(_mean_interval(values)),
        "pass_rate": sum(bool(value) for value in passes) / len(passes) if passes else None,
    }


def _contrast_summary(samples: list[dict[str, Any]], item_by_id: dict[str, Item]) -> dict[str, Any]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for sample in samples:
        item = item_by_id.get(str(sample.get("item_id")))
        if not item or not item.pair_id or sample.get("score") is None:
            continue
        role = "base" if item.variant in {"base", "clean", "supported", "free"} else "challenge"
        grouped[item.pair_id][role] = float(sample["score"])
    complete = [values for values in grouped.values() if set(values) == {"base", "challenge"}]
    base = [values["base"] for values in complete]
    challenge = [values["challenge"] for values in complete]
    deltas = [before - after for before, after in zip(base, challenge)]
    return {
        "complete_pairs": len(complete),
        "base_score": statistics.fmean(base) if base else None,
        "challenge_score": statistics.fmean(challenge) if challenge else None,
        "robustness_gap": statistics.fmean(deltas) if deltas else None,
        "pair_retention_rate": (
            sum(after >= before for before, after in zip(base, challenge)) / len(complete)
            if complete
            else None
        ),
    }


def _mean_interval(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    radius = 1.96 * standard_error
    return max(0.0, mean - radius), min(1.0, mean + radius)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _interval(value: list[float | None]) -> str:
    if value[0] is None:
        return "—"
    return f"[{100 * float(value[0]):.1f}, {100 * float(value[1]):.1f}]"


def _percentage_points(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:+.1f} pp"
