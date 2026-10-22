"""Suite loading and validation."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable

from .models import Item


SUPPORTED_SCORERS = {"choice", "exact", "contains_all", "numeric", "json_exact", "constraints", "rubric"}
SUPPORTED_SPLITS = {"dev", "validation", "test_public", "test_private"}
SUPPORTED_VARIANTS = {"base", "challenge", "clean", "distractor", "supported", "insufficient", "free", "strict"}
SUPPORTED_REVIEW_STATES = {"machine_translated", "native_reviewed", "source_native"}


def bundled_suite_dir() -> Path:
    return Path(
        str(files("europa_reasoning").joinpath("resources/suites/europa-reasoning-core-v0.1"))
    )


def resolve_suite(path: str | Path | None = None) -> Path:
    if path is None:
        return bundled_suite_dir()
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    return candidate


def load_suite(path: str | Path | None = None) -> tuple[dict[str, Any], list[Item]]:
    suite_dir = resolve_suite(path)
    metadata = json.loads((suite_dir / "suite.json").read_text(encoding="utf-8"))
    items: list[Item] = []
    data_path = suite_dir / metadata.get("data_file", "dev.jsonl")
    for line_number, line in enumerate(data_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(Item.from_dict(json.loads(line)))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{data_path}:{line_number}: invalid item: {exc}") from exc
    return metadata, items


def validate_suite(metadata: dict[str, Any], items: Iterable[Item]) -> list[str]:
    errors: list[str] = []
    required_meta = {
        "id",
        "version",
        "name",
        "status",
        "data_file",
        "domains",
        "task_types",
        "languages",
    }
    missing_meta = sorted(required_meta - metadata.keys())
    if missing_meta:
        errors.append(f"suite.json is missing: {', '.join(missing_meta)}")

    seen: set[str] = set()
    language_templates: dict[str, set[str]] = {}
    pair_members: dict[str, list[Item]] = {}
    materialized = list(items)
    if not materialized:
        errors.append("suite contains no items")
        return errors

    for item in materialized:
        prefix = item.id
        if item.id in seen:
            errors.append(f"{prefix}: duplicate id")
        seen.add(item.id)
        if item.suite_id != metadata.get("id"):
            errors.append(f"{prefix}: suite_id does not match suite.json")
        if item.suite_version != metadata.get("version"):
            errors.append(f"{prefix}: suite_version does not match suite.json")
        if item.split not in SUPPORTED_SPLITS:
            errors.append(f"{prefix}: unsupported split {item.split!r}")
        if item.scoring.get("type") not in SUPPORTED_SCORERS:
            errors.append(f"{prefix}: unsupported scorer {item.scoring.get('type')!r}")
        if not item.source.url or not item.source.license:
            errors.append(f"{prefix}: source URL and license are required")
        if not item.prompt.strip():
            errors.append(f"{prefix}: prompt is empty")
        for turn_index, turn in enumerate(item.turns):
            if turn.get("role") not in {"user", "assistant"}:
                errors.append(f"{prefix}: turn {turn_index} has unsupported role")
            if not turn.get("content", "").strip():
                errors.append(f"{prefix}: turn {turn_index} is empty")
        if item.turns and item.turns[-1].get("role") != "user":
            errors.append(f"{prefix}: prior conversation must end with a user turn")
        if item.max_tokens < 1:
            errors.append(f"{prefix}: max_tokens must be positive")
        if item.language not in metadata.get("languages", []):
            errors.append(f"{prefix}: undeclared language {item.language!r}")
        if not item.script:
            errors.append(f"{prefix}: script is required")
        if not item.template_id:
            errors.append(f"{prefix}: template_id is required")
        declared_scripts = metadata.get("language_scripts", {}).get(item.language, [])
        if declared_scripts and item.script not in declared_scripts:
            errors.append(f"{prefix}: script {item.script!r} is not declared for its language")
        if item.review_status not in SUPPORTED_REVIEW_STATES:
            errors.append(f"{prefix}: unsupported review status {item.review_status!r}")
        language_templates.setdefault(item.language, set()).add(item.template_id)
        if item.variant and item.variant not in SUPPORTED_VARIANTS:
            errors.append(f"{prefix}: unsupported variant {item.variant!r}")
        if item.pair_id:
            pair_members.setdefault(item.pair_id, []).append(item)
        elif item.variant:
            errors.append(f"{prefix}: variant requires pair_id")
        _validate_gold(item=item, errors=errors)

    for pair_id, members in pair_members.items():
        if len(members) != 2:
            errors.append(f"pair {pair_id!r}: expected exactly 2 members, found {len(members)}")
        variants = [member.variant for member in members]
        if len(set(variants)) != len(variants):
            errors.append(f"pair {pair_id!r}: variants must be unique")

    declared_domains = set(metadata.get("domains", []))
    actual_domains = {item.domain for item in materialized}
    if actual_domains != declared_domains:
        errors.append(
            "suite domains differ from data: "
            f"declared={sorted(declared_domains)}, actual={sorted(actual_domains)}"
        )
    declared_tasks = set(metadata.get("task_types", []))
    actual_tasks = {item.task_type for item in materialized}
    if actual_tasks != declared_tasks:
        errors.append(
            "suite task types differ from data: "
            f"declared={sorted(declared_tasks)}, actual={sorted(actual_tasks)}"
        )
    declared_languages = set(metadata.get("languages", []))
    if len(declared_languages) != len(metadata.get("languages", [])):
        errors.append("suite languages contain duplicates")
    if metadata.get("language_count") not in {None, len(declared_languages)}:
        errors.append("suite language_count does not match declared languages")
    actual_languages = {item.language for item in materialized}
    if actual_languages != declared_languages:
        errors.append(
            "suite languages differ from data: "
            f"declared={sorted(declared_languages)}, actual={sorted(actual_languages)}"
        )
    expected_templates = set(metadata.get("template_ids", []))
    if len(expected_templates) != len(metadata.get("template_ids", [])):
        errors.append("suite template_ids contain duplicates")
    if metadata.get("template_count") not in {None, len(expected_templates)}:
        errors.append("suite template_count does not match declared templates")
    for language in sorted(declared_languages):
        language_item_count = len([item for item in materialized if item.language == language])
        actual_templates = language_templates.get(language, set())
        if actual_templates != expected_templates:
            errors.append(
                f"language {language}: template coverage differs: "
                f"expected={sorted(expected_templates)}, actual={sorted(actual_templates)}"
            )
        if language_item_count != len(expected_templates):
            errors.append(
                f"language {language}: expected {len(expected_templates)} items, "
                f"found {language_item_count}"
            )
    actual_review_counts: dict[str, int] = {}
    for item in materialized:
        actual_review_counts[item.review_status] = (
            actual_review_counts.get(item.review_status, 0) + 1
        )
    declared_review_counts = metadata.get("review_counts")
    if declared_review_counts is not None and declared_review_counts != actual_review_counts:
        errors.append("suite review_counts do not match item review states")
    return errors


def _validate_gold(item: Item, errors: list[str]) -> None:
    scorer = item.scoring.get("type")
    if scorer in {"choice", "exact"} and "answer" not in item.gold:
        errors.append(f"{item.id}: {scorer} scorer requires gold.answer")
    elif scorer == "exact" and item.scoring.get("format") not in {None, "verbatim"}:
        errors.append(f"{item.id}: unsupported exact format {item.scoring.get('format')!r}")
    elif scorer == "contains_all" and not item.gold.get("required"):
        errors.append(f"{item.id}: contains_all scorer requires gold.required")
    elif scorer == "numeric" and "value" not in item.gold:
        errors.append(f"{item.id}: numeric scorer requires gold.value")
    elif scorer == "numeric":
        required_format = item.scoring.get("format")
        if required_format not in {None, "single_number"}:
            errors.append(f"{item.id}: unsupported numeric format {required_format!r}")
        partial_credit = item.scoring.get("partial_credit")
        if partial_credit not in {None, "relative_error"}:
            errors.append(
                f"{item.id}: unsupported numeric partial-credit method {partial_credit!r}"
            )
        if partial_credit == "relative_error" and float(item.gold["value"]) == 0:
            errors.append(f"{item.id}: relative-error partial credit requires non-zero gold")
    elif scorer == "json_exact" and "value" not in item.gold:
        errors.append(f"{item.id}: json_exact scorer requires gold.value")
    elif scorer == "json_exact" and "accepted_values" in item.gold:
        accepted = item.gold["accepted_values"]
        if not isinstance(accepted, list):
            errors.append(f"{item.id}: gold.accepted_values must be a list")
        elif any(type(value) is not type(item.gold["value"]) for value in accepted):
            errors.append(
                f"{item.id}: every accepted JSON value must have the primary value's type"
            )
    elif scorer == "constraints" and not item.scoring.get("rules"):
        errors.append(f"{item.id}: constraints scorer requires scoring.rules")
    elif scorer == "rubric" and not item.scoring.get("dimensions"):
        errors.append(f"{item.id}: rubric scorer requires scoring.dimensions")
    elif scorer == "rubric":
        dimensions = set(item.scoring["dimensions"])
        critical = item.scoring.get("critical_dimensions", {})
        if not isinstance(critical, dict):
            errors.append(f"{item.id}: critical_dimensions must be an object")
        else:
            unknown = set(critical) - dimensions
            if unknown:
                errors.append(f"{item.id}: unknown critical dimensions: {sorted(unknown)}")
            for name, minimum in critical.items():
                if isinstance(minimum, bool) or not isinstance(minimum, int) or not 0 <= minimum <= 4:
                    errors.append(
                        f"{item.id}: critical minimum for {name!r} must be an integer from 0 to 4"
                    )
