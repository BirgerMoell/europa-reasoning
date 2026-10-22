"""Merge a preserved earlier-suite run with a newly generated suite extension."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from .models import Item
from .reporting import summarize

_PUBLIC_LIMITATION = "The bundled suite is public and may be present in model training data."
_TARGET_PROTOCOL_KEYS = ("temperature", "seed", "system_prompt", "backend_settings")


def merge_extension_runs(
    *,
    base_run: dict[str, Any],
    extension_run: dict[str, Any],
    base_metadata: dict[str, Any],
    base_items: Iterable[Item],
    current_metadata: dict[str, Any],
    current_items: Iterable[Item],
) -> dict[str, Any]:
    """Return a full current-suite artifact without regenerating old responses.

    The merge is deliberately strict: identities, target-generation settings,
    expected item partitions, judgments and scores must all agree before the
    diagnostic extension may become publishable full-suite evidence.
    """
    base_items = list(base_items)
    current_items = list(current_items)
    base_ids = {item.id for item in base_items}
    current_ids = {item.id for item in current_items}
    extension_ids = current_ids - base_ids

    if base_metadata.get("id") != current_metadata.get("id"):
        raise ValueError("base and current suite IDs differ")
    _require_suite(run=base_run, metadata=base_metadata, label="base")
    _require_suite(run=extension_run, metadata=current_metadata, label="extension")
    if base_run.get("model") != extension_run.get("model"):
        raise ValueError("base and extension model identities differ")
    if base_run.get("judge") != extension_run.get("judge"):
        raise ValueError("base and extension judge identities differ")
    for key in _TARGET_PROTOCOL_KEYS:
        if base_run.get("protocol", {}).get(key) != extension_run.get("protocol", {}).get(key):
            raise ValueError(f"base and extension target protocol differ for {key}")

    base_samples = _sample_map(base_run, label="base")
    extension_samples = _sample_map(extension_run, label="extension")
    if set(base_samples) != base_ids:
        raise ValueError("base run does not contain exactly the archived base-suite items")
    if set(extension_samples) != extension_ids:
        raise ValueError("extension run does not contain exactly the newly added items")
    if base_ids & extension_ids or base_ids | extension_ids != current_ids:
        raise ValueError("base and extension item partitions do not form the current suite")

    samples = [
        deepcopy(base_samples.get(item.id, extension_samples.get(item.id)))
        for item in current_items
    ]
    for sample in samples:
        if sample is None:
            raise ValueError("merged sample unexpectedly missing")
        if sample.get("error"):
            raise ValueError(f"generation error remains for {sample['item_id']}")
        if not isinstance(sample.get("score"), (int, float)):
            raise TypeError(f"unscored item remains for {sample['item_id']}")

    summary = summarize(samples=samples, items=current_items)
    expected_count = len(current_items)
    if summary["counts"]["scored"] != expected_count:
        raise ValueError("merged run is not fully scored")

    protocol = deepcopy(base_run.get("protocol", {}))
    protocol.update(
        {
            key: deepcopy(value)
            for key, value in extension_run.get("protocol", {}).items()
            if key.startswith("judge_")
        }
    )
    protocol.update(
        {
            "limit": None,
            "domain_filter": [],
            "task_type_filter": [],
            "language_filter": [],
            "item_prefix_filter": [],
        }
    )

    old_generic = set(base_metadata.get("limitations", [])) | {_PUBLIC_LIMITATION}
    model_specific = [
        value for value in base_run.get("limitations", []) if value not in old_generic
    ]
    limitations = list(current_metadata.get("limitations", []))
    if current_metadata.get("public_development_set"):
        limitations.append(_PUBLIC_LIMITATION)
    limitations.extend(model_specific)
    limitations.append(
        f"Responses for the original {len(base_items)} items were preserved from "
        f"{base_metadata['id']} v{base_metadata['version']}; only the {len(extension_ids)} "
        "new items were generated under a protocol-equivalence check."
    )

    finished_at = extension_run.get("finished_at") or datetime.now(UTC).isoformat()
    merged = {
        "schema_version": max(base_run.get("schema_version", 1), extension_run.get("schema_version", 1)),
        "run_id": f"{base_run['run_id']}-v{str(current_metadata['version']).replace('.', '')}",
        "status": "completed",
        "diagnostic": False,
        "suite": {
            "id": current_metadata["id"],
            "version": current_metadata["version"],
            "split": current_metadata.get("split", "dev"),
            "public_development_set": bool(current_metadata.get("public_development_set", True)),
        },
        "model": deepcopy(base_run["model"]),
        "judge": deepcopy(base_run.get("judge")),
        "protocol": protocol,
        "environment": deepcopy(base_run.get("environment", {})),
        "started_at": base_run.get("started_at"),
        "finished_at": finished_at,
        "summary": summary,
        "samples": samples,
        "limitations": list(dict.fromkeys(limitations)),
        "extension_history": [
            {
                "merged_at": datetime.now(UTC).isoformat(),
                "base_run_id": base_run.get("run_id"),
                "base_suite_version": base_metadata["version"],
                "extension_run_id": extension_run.get("run_id"),
                "current_suite_version": current_metadata["version"],
                "preserved_item_ids": [item.id for item in base_items],
                "generated_item_ids": [item.id for item in current_items if item.id in extension_ids],
                "target_protocol_equivalent": True,
                "extension_environment": deepcopy(extension_run.get("environment", {})),
            }
        ],
    }
    for key in ("rescoring_history", "judging_history"):
        history = list(base_run.get(key, [])) + list(extension_run.get(key, []))
        if history:
            merged[key] = history
    if extension_run.get("judging_environment"):
        merged["extension_judging_environment"] = deepcopy(
            extension_run["judging_environment"]
        )
    return merged


def _require_suite(
    *, run: dict[str, Any], metadata: dict[str, Any], label: str
) -> None:
    suite = run.get("suite", {})
    if suite.get("id") != metadata.get("id") or str(suite.get("version")) != str(
        metadata.get("version")
    ):
        raise ValueError(f"{label} run does not match its declared suite")
    if run.get("status") != "completed":
        raise ValueError(f"{label} run is not complete")


def _sample_map(run: dict[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    samples = run.get("samples", [])
    mapped = {str(sample.get("item_id")): sample for sample in samples}
    if len(mapped) != len(samples):
        raise ValueError(f"{label} run contains duplicate item IDs")
    return mapped
