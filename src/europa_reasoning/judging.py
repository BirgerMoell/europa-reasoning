"""Offline judging of preserved rubric responses.

This separates target generation from judge inference so large local models do
not need to reside in memory at the same time.
"""

from __future__ import annotations

import json
import platform
import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .backends import Backend, GenerationConfig
from .models import Item, Score
from .reporting import summarize
from .scoring import build_judge_prompt, score_judgment


_INCOMPLETE_LIMITATION = "The run is incomplete and must not be compared as full-suite evidence."
_NO_JUDGE_LIMITATION = "Rubric-scored items require a configured judge and remain unscored here."


def judge_artifact(
    *,
    path: Path,
    suite_metadata: dict[str, Any],
    items: Iterable[Item],
    judge_backend: Backend,
    judge_revision: str | None,
    config: GenerationConfig,
) -> dict[str, Any]:
    """Judge unscored rubric samples in place without regenerating target answers."""
    path = path.resolve()
    run = json.loads(path.read_text(encoding="utf-8"))
    updated = judge_run(
        run=run,
        suite_metadata=suite_metadata,
        items=items,
        judge_backend=judge_backend,
        judge_revision=judge_revision,
        config=config,
    )
    _write_json_atomic(path, updated)
    _append_updated_samples(path=path, before=run, after=updated)
    _update_manifest(
        path=path,
        judge_backend=judge_backend,
        judge_revision=judge_revision,
        config=config,
    )
    return updated


def judge_run(
    *,
    run: dict[str, Any],
    suite_metadata: dict[str, Any],
    items: Iterable[Item],
    judge_backend: Backend,
    judge_revision: str | None,
    config: GenerationConfig,
) -> dict[str, Any]:
    """Return a run with previously unjudged rubric samples scored."""
    if run.get("suite", {}).get("id") != suite_metadata["id"]:
        raise ValueError("run and suite IDs differ")
    if str(run.get("suite", {}).get("version")) != str(suite_metadata["version"]):
        raise ValueError("run and suite versions differ; rescore the run before judging")

    previous_judge = run.get("judge")
    judge_identity = {
        "id": judge_backend.model_id,
        "revision": judge_revision,
        "backend": judge_backend.name,
    }
    if previous_judge and previous_judge != judge_identity:
        raise ValueError("run already contains judgments from a different judge")

    item_by_id = {item.id: item for item in items}
    samples: list[dict[str, Any]] = []
    judged_ids: list[str] = []
    failed_ids: list[str] = []
    started_at = datetime.now(timezone.utc).isoformat()

    for original in run.get("samples", []):
        sample = deepcopy(original)
        item_id = str(sample.get("item_id"))
        item = item_by_id.get(item_id)
        if item is None:
            raise ValueError(f"run contains item absent from suite: {item_id}")
        should_judge = (
            item.scoring["type"] == "rubric"
            and sample.get("score") is None
            and not sample.get("error")
            and isinstance(sample.get("response"), str)
        )
        if should_judge:
            print(f"[{len(judged_ids) + len(failed_ids) + 1}] judge {item_id}", file=sys.stderr)
            try:
                generation = judge_backend.generate(
                    item=_judge_item(item=item, response=sample["response"]),
                    config=config,
                )
                judgment = {
                    "model": judge_backend.model_id,
                    "revision": judge_revision,
                    "backend": judge_backend.name,
                    "response": generation.text,
                    "latency_ms": generation.latency_ms,
                    "input_tokens": generation.input_tokens,
                    "output_tokens": generation.output_tokens,
                    "finish_reason": generation.finish_reason,
                    "generation_metadata": generation.raw or {},
                }
                sample["judgment"] = judgment
                try:
                    score = score_judgment(
                        item=item,
                        judgment=generation.text,
                        response=sample["response"],
                    )
                except ValueError as exc:
                    score = Score(
                        value=None,
                        passed=None,
                        scorer="rubric",
                        error=f"invalid judge response: {exc}",
                    )
                sample.update(
                    {
                        "score": score.value,
                        "passed": score.passed,
                        "parsed": score.parsed,
                        "score_details": score.details,
                        "scoring_error": score.error,
                    }
                )
                if score.value is None:
                    failed_ids.append(item_id)
                else:
                    judged_ids.append(item_id)
            except Exception as exc:  # noqa: BLE001 - preserve other judgments
                sample["scoring_error"] = f"judge error: {type(exc).__name__}: {exc}"
                failed_ids.append(item_id)
        samples.append(sample)

    selected_items = [item_by_id[str(sample["item_id"])] for sample in samples]
    summary = summarize(samples=samples, items=selected_items)
    status = _status_from_summary(summary)
    finished_at = datetime.now(timezone.utc).isoformat()

    updated = deepcopy(run)
    updated["samples"] = samples
    updated["summary"] = summary
    updated["status"] = status
    updated["judge"] = judge_identity
    updated.setdefault("protocol", {})["judge_backend_settings"] = (
        judge_backend.protocol_settings()
    )
    updated["protocol"]["judge_temperature"] = config.temperature
    updated["protocol"]["judge_seed"] = config.seed
    updated["protocol"]["judge_system_prompt"] = config.system_prompt
    updated["judging_environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    limitations = [
        value
        for value in updated.get("limitations", [])
        if value not in {_INCOMPLETE_LIMITATION, _NO_JUDGE_LIMITATION}
    ]
    if status != "completed":
        limitations.append(_INCOMPLETE_LIMITATION)
    updated["limitations"] = list(dict.fromkeys(limitations))
    history = list(updated.get("judging_history", []))
    history.append(
        {
            "started_at": started_at,
            "finished_at": finished_at,
            "judge": judge_identity,
            "target_responses_reused": True,
            "judged_item_ids": judged_ids,
            "failed_item_ids": failed_ids,
        }
    )
    updated["judging_history"] = history
    return updated


def _judge_item(*, item: Item, response: str) -> Item:
    payload = asdict(item)
    payload["id"] = f"judge::{item.id}"
    payload["prompt"] = build_judge_prompt(item=item, response=response)
    payload["context"] = ""
    payload["turns"] = ()
    payload["options"] = ()
    payload["max_tokens"] = 320
    return Item(**payload)


def _status_from_summary(summary: dict[str, Any]) -> str:
    if summary["counts"]["generation_errors"]:
        return "failed" if not summary["counts"]["scored"] else "partial"
    if summary["counts"]["unjudged"]:
        return "partial"
    return "completed"


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_updated_samples(
    *, path: Path, before: dict[str, Any], after: dict[str, Any]
) -> None:
    sidecar = path.with_suffix(".samples.jsonl")
    if not sidecar.exists():
        return
    before_by_id = {str(sample["item_id"]): sample for sample in before.get("samples", [])}
    with sidecar.open("a", encoding="utf-8") as handle:
        for sample in after.get("samples", []):
            if sample != before_by_id.get(str(sample["item_id"])):
                handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")


def _update_manifest(
    *,
    path: Path,
    judge_backend: Backend,
    judge_revision: str | None,
    config: GenerationConfig,
) -> None:
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "judge_id": judge_backend.model_id,
            "judge_revision": judge_revision,
            "judge_backend": judge_backend.name,
            "judge_backend_settings": judge_backend.protocol_settings(),
            "judge_temperature": config.temperature,
            "judge_seed": config.seed,
            "judge_system_prompt": config.system_prompt,
        }
    )
    _write_json_atomic(manifest_path, manifest)
