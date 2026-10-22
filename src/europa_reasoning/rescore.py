"""Offline re-scoring of preserved model responses against a revised suite."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from .models import Item
from .reporting import summarize
from .scoring import score_item, score_judgment


def rescore_artifact(
    *,
    path: Path,
    suite_metadata: dict[str, Any],
    items: Iterable[Item],
    reason: str,
) -> dict[str, Any]:
    """Re-score deterministic samples in place without calling a model or judge."""
    run = json.loads(path.read_text(encoding="utf-8"))
    updated = rescore_run(
        run=run,
        suite_metadata=suite_metadata,
        items=items,
        reason=reason,
    )
    path.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return updated


def rescore_run(
    *,
    run: dict[str, Any],
    suite_metadata: dict[str, Any],
    items: Iterable[Item],
    reason: str,
) -> dict[str, Any]:
    """Return a revised run while preserving all generation and judge evidence."""
    if run.get("suite", {}).get("id") != suite_metadata["id"]:
        raise ValueError("run and suite IDs differ")
    item_by_id = {item.id: item for item in items}
    source_version = str(run.get("suite", {}).get("version", "unknown"))
    samples: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []

    for original in run.get("samples", []):
        sample = deepcopy(original)
        item_id = str(sample.get("item_id"))
        item = item_by_id.get(item_id)
        if item is None:
            raise ValueError(f"run contains item absent from target suite: {item_id}")
        response = sample.get("response")
        judgment_response = (sample.get("judgment") or {}).get("response")
        should_rescore = not sample.get("error") and isinstance(response, str)
        if should_rescore:
            if item.scoring["type"] == "rubric" and isinstance(judgment_response, str):
                score = score_judgment(
                    item=item,
                    judgment=judgment_response,
                    response=response,
                )
            else:
                score = score_item(item=item, response=response)
            before = {key: sample.get(key) for key in ("score", "passed", "score_details")}
            sample.update(
                {
                    "score": score.value,
                    "passed": score.passed,
                    "scorer": score.scorer,
                    "parsed": score.parsed,
                    "score_details": score.details,
                    "scoring_error": score.error,
                }
            )
            after = {key: sample.get(key) for key in ("score", "passed", "score_details")}
            if before != after:
                changes.append({"item_id": item_id, "before": before, "after": after})
        samples.append(sample)

    selected_items = [item_by_id[str(sample["item_id"])] for sample in samples]
    summary = summarize(samples=samples, items=selected_items)
    status = "completed"
    if summary["counts"]["generation_errors"]:
        status = "failed" if not summary["counts"]["scored"] else "partial"
    elif summary["counts"]["unjudged"]:
        status = "partial"

    updated = deepcopy(run)
    updated["suite"]["version"] = suite_metadata["version"]
    updated["samples"] = samples
    updated["summary"] = summary
    updated["status"] = status
    history = list(updated.get("rescoring_history", []))
    history.append(
        {
            "source_suite_version": source_version,
            "target_suite_version": suite_metadata["version"],
            "reason": reason,
            "model_responses_reused": True,
            "judge_outputs_reused": True,
            "changes": changes,
        }
    )
    updated["rescoring_history"] = history
    return updated
