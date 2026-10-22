"""Build the static GitHub Pages data bundle from validated run artifacts."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from .data import load_suite, validate_suite


def build_site_data(
    *,
    docs_dir: Path,
    results_dir: Path | None = None,
    external_catalog: Path | None = None,
) -> dict[str, Any]:
    metadata, items = load_suite()
    errors = validate_suite(metadata=metadata, items=items)
    if errors:
        raise ValueError("cannot build site from invalid suite:\n" + "\n".join(errors))

    if external_catalog is None:
        external_catalog = Path(str(files("europa_reasoning").joinpath("resources/external.json")))
    integrations = json.loads(external_catalog.read_text(encoding="utf-8"))["integrations"]
    runs = _load_public_runs(
        results_dir,
        suite_id=metadata["id"],
        suite_version=metadata["version"],
        expected_items=len(items),
        item_by_id={item.id: item for item in items},
    )
    domains = []
    for domain in metadata["domains"]:
        domain_items = [item for item in items if item.domain == domain]
        domains.append(
            {
                "id": domain,
                "name": metadata["domain_labels"][domain],
                "description": metadata["domain_descriptions"][domain],
                "item_count": len(domain_items),
                "capabilities": sorted({item.capability for item in domain_items}),
                "task_types": sorted({item.task_type for item in domain_items}),
            }
        )
    catalog = {
        "project": {
            "name": "EUROPA Reasoning",
            "version": metadata["version"],
            "tagline": metadata["tagline"],
            "repository": "https://github.com/BirgerMoell/europa-reasoning",
        },
        "suite": {
            "id": metadata["id"],
            "name": metadata["name"],
            "status": metadata["status"],
            "items": len(items),
            "pairs": len({item.pair_id for item in items if item.pair_id}),
            "languages": [
                {
                    "code": code,
                    "bcp47": next(
                        item.metadata.get("bcp47", code)
                        for item in items
                        if item.language == code
                    ),
                    "name": metadata["language_labels"][code],
                    "autonym": metadata["language_autonyms"][code],
                    "group": metadata["language_groups"][code],
                    "scripts": metadata["language_scripts"][code],
                    "variants": metadata["language_variants"][code],
                    "item_count": len([item for item in items if item.language == code]),
                    "review_states": sorted(
                        {item.review_status for item in items if item.language == code}
                    ),
                }
                for code in metadata["languages"]
            ],
            "templates": metadata["template_ids"],
            "review_counts": metadata.get("review_counts", {}),
            "language_source": metadata.get("language_source"),
            "domains": domains,
            "task_types": [
                {"id": task, "name": metadata["task_type_labels"][task]}
                for task in metadata["task_types"]
            ],
            "methodology": metadata["methodology"],
            "limitations": metadata["limitations"],
        },
        "integrations": integrations,
    }
    data_dir = docs_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_payload = {"runs": runs}
    (data_dir / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (data_dir / "results.json").write_text(
        json.dumps(results_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (data_dir / "site-data.js").write_text(
        "window.EUROPA_REASONING_SITE_DATA = "
        + json.dumps(
            {"catalog": catalog, "results": results_payload},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + ";\n",
        encoding="utf-8",
    )
    return {"catalog": catalog, "runs": len(runs)}


def _load_public_runs(
    results_dir: Path | None,
    *,
    suite_id: str,
    suite_version: str,
    expected_items: int,
    item_by_id: dict[str, Any],
) -> list[dict[str, Any]]:
    if results_dir is None or not results_dir.exists():
        return []
    runs: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if not _is_publishable(
            run,
            suite_id=suite_id,
            suite_version=suite_version,
            expected_items=expected_items,
        ):
            continue
        runs.append(_public_run(run, item_by_id=item_by_id))
    return runs


def _is_publishable(
    run: dict[str, Any], *, suite_id: str, suite_version: str, expected_items: int
) -> bool:
    model = run.get("model", {})
    judge = run.get("judge", {}) or {}
    return bool(
        not run.get("diagnostic")
        and run.get("status") == "completed"
        and model.get("revision")
        and model.get("backend") != "oracle-diagnostic"
        and judge.get("id")
        and judge.get("revision")
        and judge.get("backend") != "oracle-diagnostic"
        and judge.get("id") != model.get("id")
        and run.get("suite", {}).get("id") == suite_id
        and run.get("suite", {}).get("version") == suite_version
        and run.get("summary", {}).get("counts", {}).get("scheduled") == expected_items
        and run.get("summary", {}).get("counts", {}).get("scored") == expected_items
    )


def _public_run(run: dict[str, Any], *, item_by_id: dict[str, Any]) -> dict[str, Any]:
    item_results = []
    for sample in run.get("samples", []):
        item = item_by_id.get(sample.get("item_id"))
        if item is None:
            continue
        item_results.append(
            {
                "item": {
                    "id": item.id,
                    "language": item.language,
                    "script": item.script,
                    "template_id": item.template_id,
                    "review_status": item.review_status,
                    "metadata": item.metadata,
                    "domain": item.domain,
                    "capability": item.capability,
                    "task_type": item.task_type,
                    "prompt": item.prompt,
                    "context": item.context,
                    "turns": list(item.turns),
                    "options": list(item.options),
                    "pair_id": item.pair_id,
                    "variant": item.variant,
                    "rubric": (
                        {
                            "dimensions": item.scoring["dimensions"],
                            "critical_dimensions": item.scoring.get("critical_dimensions", {}),
                            "pass_threshold": item.scoring.get("pass_threshold"),
                            "reference_answer": item.gold.get("reference_answer"),
                            "required_points": item.gold.get("required_points", []),
                        }
                        if item.scoring["type"] == "rubric"
                        else None
                    ),
                    "source": {
                        "title": item.source.title,
                        "url": item.source.url,
                        "license": item.source.license,
                    },
                },
                "sample": {
                    key: sample.get(key)
                    for key in (
                        "response",
                        "score",
                        "passed",
                        "scorer",
                        "parsed",
                        "score_details",
                        "scoring_error",
                        "latency_ms",
                        "input_tokens",
                        "output_tokens",
                        "finish_reason",
                        "generation_metadata",
                        "reasoning_tokens",
                        "process",
                        "judgment",
                        "error",
                    )
                    # "reasoning" (the trace) is deliberately never published.
                },
            }
        )
    return {
        "run_id": run["run_id"],
        "model": run["model"],
        "judge": run.get("judge"),
        "suite": run["suite"],
        "protocol": run.get("protocol", {}),
        "finished_at": run["finished_at"],
        "summary": run["summary"],
        "reasoning_mode": bool(run.get("protocol", {}).get("backend_settings", {}).get("think")),
        "limitations": run.get("limitations", []),
        "rescoring_history": run.get("rescoring_history", []),
        "judging_history": run.get("judging_history", []),
        "extension_history": run.get("extension_history", []),
        "items": item_results,
    }
