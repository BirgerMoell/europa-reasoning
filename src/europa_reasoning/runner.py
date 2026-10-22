"""Resumable evaluation runner with complete protocol provenance."""

from __future__ import annotations

import json
import platform
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .backends import Backend, GenerationConfig
from .models import Item, Score
from .process import process_metrics
from .reporting import summarize
from .scoring import build_judge_prompt, score_item, score_judgment


def run_evaluation(
    *,
    suite_metadata: dict[str, Any],
    items: Iterable[Item],
    backend: Backend,
    output: Path,
    config: GenerationConfig,
    model_revision: str | None = None,
    judge_backend: Backend | None = None,
    judge_revision: str | None = None,
    judge_config: GenerationConfig | None = None,
    limit: int | None = None,
    domains: set[str] | None = None,
    task_types: set[str] | None = None,
    languages: set[str] | None = None,
    item_prefixes: tuple[str, ...] = (),
    run_id: str | None = None,
    diagnostic: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    selected = [
        item
        for item in items
        if (not domains or item.domain in domains)
        and (not task_types or item.task_type in task_types)
        and (not languages or item.language in languages)
        and (not item_prefixes or item.id.startswith(item_prefixes))
    ]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("filters selected no evaluation items")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_suffix(".samples.jsonl")
    manifest_path = output.with_suffix(".manifest.json")
    effective_diagnostic = bool(
        diagnostic
        or limit is not None
        or domains
        or task_types
        or languages
        or item_prefixes
    )
    effective_judge_config = judge_config or config
    manifest = {
        "schema_version": 1,
        "suite_id": suite_metadata["id"],
        "suite_version": suite_metadata["version"],
        "item_ids": [item.id for item in selected],
        "model_id": backend.model_id,
        "model_revision": model_revision,
        "backend": backend.name,
        "backend_settings": backend.protocol_settings(),
        "judge_id": judge_backend.model_id if judge_backend else None,
        "judge_revision": judge_revision,
        "judge_backend": judge_backend.name if judge_backend else None,
        "judge_backend_settings": judge_backend.protocol_settings() if judge_backend else {},
        "temperature": config.temperature,
        "seed": config.seed,
        "system_prompt": config.system_prompt,
        "language_filter": sorted(languages) if languages else [],
    }
    if judge_backend:
        manifest.update(
            {
                "judge_temperature": effective_judge_config.temperature,
                "judge_seed": effective_judge_config.seed,
                "judge_system_prompt": effective_judge_config.system_prompt,
            }
        )
    _validate_or_write_manifest(path=manifest_path, sidecar=sidecar, manifest=manifest)
    completed = _load_sidecar(sidecar)
    started_at = datetime.now(timezone.utc).isoformat()

    pending: list[tuple[int, Item]] = []
    for position, item in enumerate(selected, start=1):
        if item.id in completed:
            rescored = _rescore_preserved_judgment(
                item=item, sample=completed[item.id]
            )
            if rescored is not None:
                completed[item.id] = rescored
                with sidecar.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(rescored, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                print(f"[{position}/{len(selected)}] rescore {item.id}", file=sys.stderr)
                continue
            print(f"[{position}/{len(selected)}] resume {item.id}", file=sys.stderr)
            continue
        pending.append((position, item))

    # Items are independent, so a batching server (vLLM, Ollama with parallel slots) can be kept
    # busy with concurrent requests. The sidecar stays append-only and resumable; only the write
    # order changes, and samples are re-ordered by suite order below.
    lock = threading.Lock()

    def _work(position: int, item: Item) -> None:
        print(f"[{position}/{len(selected)}] run    {item.id}", file=sys.stderr)
        sample = _run_item(
            item=item,
            backend=backend,
            config=config,
            judge_backend=judge_backend,
            judge_config=effective_judge_config,
        )
        with lock:
            completed[item.id] = sample
            with sidecar.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")

    if workers <= 1:
        for position, item in pending:
            _work(position, item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(_work, pos, it) for pos, it in pending]):
                future.result()

    samples = [completed[item.id] for item in selected]
    summary = summarize(samples=samples, items=selected)
    status = "completed"
    if summary["counts"]["generation_errors"]:
        status = "failed" if not summary["counts"]["scored"] else "partial"
    elif summary["counts"]["unjudged"]:
        status = "partial"
    if backend.name == "oracle-diagnostic" or (
        judge_backend is not None and judge_backend.name == "oracle-diagnostic"
    ):
        effective_diagnostic = True

    finished_at = datetime.now(timezone.utc).isoformat()
    run = {
        "schema_version": 1,
        "run_id": run_id or _default_run_id(backend.model_id),
        "status": status,
        "diagnostic": effective_diagnostic,
        "suite": {
            "id": suite_metadata["id"],
            "version": suite_metadata["version"],
            "split": suite_metadata.get("split", "dev"),
            "public_development_set": bool(suite_metadata.get("public_development_set", True)),
        },
        "model": {
            "id": backend.model_id,
            "revision": model_revision,
            "backend": backend.name,
        },
        "judge": (
            {
                "id": judge_backend.model_id,
                "revision": judge_revision,
                "backend": judge_backend.name,
            }
            if judge_backend
            else None
        ),
        "protocol": {
            "temperature": config.temperature,
            "seed": config.seed,
            "system_prompt": config.system_prompt,
            "limit": limit,
            "domain_filter": sorted(domains) if domains else [],
            "task_type_filter": sorted(task_types) if task_types else [],
            "language_filter": sorted(languages) if languages else [],
            "item_prefix_filter": list(item_prefixes),
            "backend_settings": backend.protocol_settings(),
            "judge_backend_settings": (
                judge_backend.protocol_settings() if judge_backend else {}
            ),
            "judge_temperature": (
                effective_judge_config.temperature if judge_backend else None
            ),
            "judge_seed": effective_judge_config.seed if judge_backend else None,
            "judge_system_prompt": (
                effective_judge_config.system_prompt if judge_backend else None
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "started_at": started_at,
        "finished_at": finished_at,
        "summary": summary,
        "samples": samples,
        "limitations": _limitations(
            status=status,
            diagnostic=effective_diagnostic,
            suite_metadata=suite_metadata,
            judge_backend=judge_backend,
        ),
    }
    output.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return run


def _run_item(
    *,
    item: Item,
    backend: Backend,
    config: GenerationConfig,
    judge_backend: Backend | None,
    judge_config: GenerationConfig,
) -> dict[str, Any]:
    try:
        generation = backend.generate(item=item, config=config)
        reasoning_enabled = bool(backend.protocol_settings().get("think"))
        process = process_metrics(item, generation, reasoning_enabled=reasoning_enabled)
        score = score_item(item=item, response=generation.text)
        judgment: dict[str, Any] | None = None
        if score.scorer == "rubric" and judge_backend is not None:
            judge_item = _judge_item(item=item, response=generation.text)
            judge_generation = judge_backend.generate(item=judge_item, config=judge_config)
            judgment = {
                "model": judge_backend.model_id,
                "response": judge_generation.text,
                "latency_ms": judge_generation.latency_ms,
                "generation_metadata": judge_generation.raw or {},
            }
            try:
                score = score_judgment(
                    item=item, judgment=judge_generation.text, response=generation.text
                )
            except ValueError as exc:
                score = Score(
                    value=None,
                    passed=None,
                    scorer="rubric",
                    error=f"invalid judge response: {exc}",
                )
        return {
            "item_id": item.id,
            "response": generation.text,
            "score": score.value,
            "passed": score.passed,
            "scorer": score.scorer,
            "parsed": score.parsed,
            "score_details": score.details,
            "scoring_error": score.error,
            "latency_ms": generation.latency_ms,
            "input_tokens": generation.input_tokens,
            "output_tokens": generation.output_tokens,
            "finish_reason": generation.finish_reason,
            "generation_metadata": generation.raw or {},
            "reasoning": generation.reasoning,
            "reasoning_tokens": generation.reasoning_tokens,
            "process": process,
            "judgment": judgment,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - one failed sample must not destroy a run
        return {
            "item_id": item.id,
            "response": None,
            "score": None,
            "passed": None,
            "scorer": item.scoring["type"],
            "parsed": None,
            "score_details": {},
            "scoring_error": None,
            "latency_ms": None,
            "input_tokens": None,
            "output_tokens": None,
            "finish_reason": None,
            "generation_metadata": {},
            "reasoning": None,
            "reasoning_tokens": None,
            "process": {},
            "judgment": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _judge_item(item: Item, response: str) -> Item:
    payload = asdict(item)
    payload["id"] = f"judge::{item.id}"
    payload["prompt"] = build_judge_prompt(item=item, response=response)
    payload["context"] = ""
    payload["turns"] = ()
    payload["options"] = ()
    payload["max_tokens"] = 320
    payload["source"] = item.source
    return Item(**payload)


def _rescore_preserved_judgment(
    *, item: Item, sample: dict[str, Any]
) -> dict[str, Any] | None:
    """Re-parse a preserved judge response after a scorer/parser improvement."""
    judgment = sample.get("judgment")
    if (
        sample.get("score") is not None
        or sample.get("error")
        or not isinstance(judgment, dict)
    ):
        return None
    response = judgment.get("response")
    if sample.get("scorer") != "rubric" or not isinstance(response, str):
        return None
    try:
        score = score_judgment(
            item=item, judgment=response, response=str(sample.get("response", ""))
        )
    except ValueError:
        return None
    return {
        **sample,
        "score": score.value,
        "passed": score.passed,
        "parsed": score.parsed,
        "score_details": score.details,
        "scoring_error": None,
    }


def _load_sidecar(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
            completed[str(sample["item_id"])] = sample
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"{path}:{line_number}: corrupt resume record: {exc}") from exc
    return completed


def _validate_or_write_manifest(
    *, path: Path, sidecar: Path, manifest: dict[str, Any]
) -> None:
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError(
                f"resume manifest differs for {path}; choose a new --output or restore the original protocol"
            )
        return
    if sidecar.exists():
        raise ValueError(f"resume sidecar exists without protocol manifest: {sidecar}")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _default_run_id(model_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = "".join(char if char.isalnum() else "-" for char in model_id).strip("-")
    return f"{stamp}-{safe_model[:48]}"


def _limitations(
    *,
    status: str,
    diagnostic: bool,
    suite_metadata: dict[str, Any],
    judge_backend: Backend | None,
) -> list[str]:
    values = list(suite_metadata.get("limitations", []))
    if suite_metadata.get("public_development_set"):
        values.append("The bundled suite is public and may be present in model training data.")
    if status != "completed":
        values.append("The run is incomplete and must not be compared as full-suite evidence.")
    if diagnostic:
        values.append("Diagnostic runs are software checks, not leaderboard evidence.")
    if judge_backend is None:
        values.append("Rubric-scored items require a configured judge and remain unscored here.")
    return list(dict.fromkeys(values))
