#!/usr/bin/env python3
"""Fail fast on incomplete, structurally unsafe or script-mismatched packs."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

from generate_language_packs import validate_translation


ROOT = Path(__file__).resolve().parents[1]
RESOURCE_DIR = ROOT / "src/europa_reasoning/resources"
EXPECTED_UNICODE_NAMES = {
    "Cyrl": "CYRILLIC",
    "Grek": "GREEK",
    "Geor": "GEORGIAN",
}
SCRIPT_SAMPLE_KEYS = (
    "racket_prompt",
    "bakery_prompt",
    "knights_prompt",
    "fermi_prompt",
)


def main() -> int:
    registry = read_json(RESOURCE_DIR / "languages.json")
    master = read_json(RESOURCE_DIR / "pack-source-en.json")["strings"]
    languages = registry["languages"]
    expected = {language["code"] for language in languages}
    pack_dir = RESOURCE_DIR / "language-packs"
    found = {path.stem for path in pack_dir.glob("*.json")}
    if found != expected:
        raise ValueError(
            f"language pack set differs: missing={sorted(expected - found)}, "
            f"extra={sorted(found - expected)}"
        )

    review_counts: dict[str, int] = {}
    errors: list[str] = []
    for language in languages:
        try:
            pack = read_json(pack_dir / f"{language['code']}.json")
            if pack["language"] != language:
                raise ValueError("registry metadata differs")
            state = str(pack["review_status"])
            review_counts[state] = review_counts.get(state, 0) + 1
            if language["code"] == "eng":
                if pack["strings"] != master or state != "source_native":
                    raise ValueError("authored source pack changed or has the wrong review state")
                continue
            if not pack.get("translation_model_revision"):
                raise ValueError("translation model revision is missing")
            validate_translation(master, pack["strings"], english=False, script=language["scripts"][0])
            if state not in {"machine_translated", "native_reviewed"}:
                raise ValueError(f"invalid generated-pack review state {state!r}")
            if corrections := pack.get("quality_corrections"):
                if corrections.get("method") != "manual_post_edit":
                    raise ValueError("quality correction method must be manual_post_edit")
                fields = corrections.get("fields")
                if not isinstance(fields, list) or not fields:
                    raise ValueError("quality correction fields must be a non-empty list")
                unknown = set(fields) - set(master)
                if unknown:
                    raise ValueError(f"quality correction names unknown fields: {sorted(unknown)}")
                if not corrections.get("date") or not corrections.get("reason"):
                    raise ValueError("quality correction date and reason are required")
            audit_script(language, pack["strings"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{language['code']}: {exc}")

    if errors:
        raise ValueError("language pack audit failed:\n" + "\n".join(errors))

    print(
        f"OK {len(languages)} language packs · "
        + " · ".join(f"{state}={count}" for state, count in sorted(review_counts.items()))
    )
    return 0


def audit_script(language: dict[str, Any], strings: dict[str, Any]) -> None:
    script = language["scripts"][0]
    if script == "Latn":
        unexpected = {
            character: unicodedata.name(character, "UNKNOWN")
            for character in flatten_text(strings)
            if character.isalpha() and "LATIN" not in unicodedata.name(character, "")
        }
        if unexpected:
            details = ", ".join(
                f"{character!r} ({name})" for character, name in sorted(unexpected.items())
            )
            raise ValueError(f"unexpected non-Latin letters: {details}")
        return
    expected_name = EXPECTED_UNICODE_NAMES.get(script)
    if not expected_name:
        return
    unexpected = {
        character: unicodedata.name(character, "UNKNOWN")
        for character in flatten_text(strings)
        if character.isalpha()
        and not any(
            allowed in unicodedata.name(character, "")
            for allowed in {"LATIN", expected_name}
        )
    }
    if unexpected:
        details = ", ".join(
            f"{character!r} ({name})" for character, name in sorted(unexpected.items())
        )
        raise ValueError(f"unexpected script letters: {details}")
    text = " ".join(str(strings[key]) for key in SCRIPT_SAMPLE_KEYS)
    letters = [character for character in text if character.isalpha()]
    expected_letters = [
        character
        for character in letters
        if expected_name in unicodedata.name(character, "")
    ]
    ratio = len(expected_letters) / len(letters) if letters else 0
    if ratio < 0.55:
        raise ValueError(
            f"expected {script} script ratio >= 0.55, found {ratio:.2f}"
        )


def flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(flatten_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    return str(value)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
