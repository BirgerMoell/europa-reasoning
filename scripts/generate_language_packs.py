#!/usr/bin/env python3
"""Translate the authored reasoning pack through a local Ollama model."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LANGUAGES_PATH = ROOT / "src/europa_reasoning/resources/languages.json"
MASTER_PATH = ROOT / "src/europa_reasoning/resources/pack-source-en.json"
PACK_DIR = ROOT / "src/europa_reasoning/resources/language-packs"
# Placeholders ({name1}, {a}, ...) are filled after translation; \\boxed examples define the
# answer format the scorers parse. Both must survive translation byte for byte.
PLACEHOLDER = re.compile(r"\{[a-z_0-9]+\}")
PROTECTED_LITERALS = ("\\boxed{}", "\\boxed{42}", "\\boxed{B}")
# LaTeX is masked as neutral tokens during translation and restored afterwards; models otherwise
# "translate" or drop the braces.
MASKS = {"\\boxed{42}": "[[BOX42]]", "\\boxed{B}": "[[BOXB]]", "\\boxed{}": "[[BOX]]"}


def mask(value: Any) -> Any:
    if isinstance(value, str):
        for literal, token in MASKS.items():
            value = value.replace(literal, token)
        return value
    if isinstance(value, list):
        return [mask(v) for v in value]
    return {k: mask(v) for k, v in value.items()}


def unmask(value: Any) -> Any:
    if isinstance(value, str):
        for literal, token in MASKS.items():
            value = value.replace(token, literal)
        return value
    if isinstance(value, list):
        return [unmask(v) for v in value]
    return {k: unmask(v) for k, v in value.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma4:31b")
    parser.add_argument("--model-revision", default="6316f0629137")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    registry = read_json(LANGUAGES_PATH)
    master = read_json(MASTER_PATH)
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    selected = set(args.only)
    failures: list[str] = []
    for language in registry["languages"]:
        code = language["code"]
        if selected and code not in selected:
            continue
        destination = PACK_DIR / f"{code}.json"
        if destination.exists() and not args.overwrite:
            try:
                existing = read_json(destination)
                validate_translation(master["strings"], existing["strings"], english=code == "eng",
                                     script=language["scripts"][0])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                print(f"regenerate {code}: {exc}")
            else:
                print(f"skip {code}: {destination.name} exists")
                continue

        if code == "eng":
            translated = master["strings"]
            generator = "authored-source"
            generator_revision = None
            review_status = "source_native"
            seed = None
            repaired_keys = []
        else:
            print(f"translate {code}: {language['name']}")
            last_error: Exception | None = None
            seed = 42
            translated = None
            repaired_keys: list[str] = []
            for attempt in range(1, 5):
                # Decoding is greedy, so an identical retry reproduces the same error: each retry
                # changes the seed and states the validation error back to the translator.
                seed = 42 + attempt - 1
                try:
                    translated = unmask(translate(
                        strings=mask(master["strings"]), language=language,
                        model=args.model, base_url=args.base_url, seed=seed,
                        feedback=str(last_error) if last_error else None,
                    ))
                    validate_translation(master["strings"], translated, english=False, script=language["scripts"][0])
                    break
                except Exception as exc:  # noqa: BLE001 - retry one isolated local generation
                    last_error = exc
                    print(f"retry {code} ({attempt}/4): {exc}", flush=True)
                    time.sleep(1)
            else:
                # Whole-pack retries keep failing on one string: re-translate only the failing
                # top-level keys, one at a time, with the validation error as feedback.
                repaired = _repair_keys(master["strings"], translated,
                                        language=language, model=args.model, base_url=args.base_url)
                if repaired is None:
                    print(f"FAILED {code}: {last_error}", flush=True)
                    failures.append(code)
                    continue
                repaired_keys = [k for k in repaired if translated is None or repaired[k] != translated.get(k)]
                translated = repaired
            generator = args.model
            generator_revision = args.model_revision
            review_status = "machine_translated"

        destination.write_text(
            json.dumps(
                {
                    "language": language,
                    "review_status": review_status,
                    "translation_generator": generator,
                    "translation_model_revision": generator_revision,
                    "translation_protocol": {"temperature": 0, "seed": seed if code != "eng" else None,
                                             "think": False, "latex_masked": True,
                                             **({"repaired_keys": repaired_keys, "repair_seeds": "101-105"}
                                                if repaired_keys else {})},
                    "strings": translated,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    if failures:
        print(f"failed languages: {' '.join(failures)}", flush=True)
        return 1
    return 0


def _repair_keys(source: dict[str, Any], translated: dict[str, Any] | None, *, language: dict[str, Any],
                 model: str, base_url: str) -> dict[str, Any] | None:
    if not isinstance(translated, dict) or translated.keys() != source.keys():
        return None
    repaired = dict(translated)
    for key in source:
        try:
            validate_translation({key: source[key]}, {key: repaired[key]}, english=False, path=f"strings.{key}",
                                 script=language["scripts"][0])
            continue
        except ValueError as exc:
            error: Exception | None = exc
        for attempt in range(1, 6):
            print(f"repair {language['code']}.{key} ({attempt}/5): {error}", flush=True)
            try:
                piece = unmask(translate(strings=mask({key: source[key]}), language=language, model=model,
                                         base_url=base_url, seed=100 + attempt,
                                         feedback=f"{error}. Every placeholder in the source must appear in the translation"))
                validate_translation({key: source[key]}, piece, english=False, path=f"strings.{key}",
                                     script=language["scripts"][0])
                repaired[key] = piece[key]
                error = None
                break
            except Exception as exc:  # noqa: BLE001
                error = exc
        if error is not None:
            return None
    try:
        validate_translation(source, repaired, english=False, script=language["scripts"][0])
    except ValueError:
        return None
    return repaired


def translate(
    *, strings: dict[str, Any], language: dict[str, Any], model: str, base_url: str,
    seed: int = 42, feedback: str | None = None,
) -> dict[str, Any]:
    system = (
        "You are a meticulous translator of reasoning-benchmark items. Return one valid JSON object "
        "with exactly the same keys, nesting and array lengths. Translate every natural-language "
        "string into the requested language. Preserve the exact mathematical and logical meaning: "
        "do not simplify, solve, hint at or change any problem, and keep every condition. Keep every "
        "placeholder in curly braces, such as {name1}, {a} or {w1}, exactly as written, even when it "
        "seems redundant, and keep the tokens [[BOX]], [[BOX42]] and [[BOXB]] unchanged. Placeholders stand for names, numbers or "
        "invented words and must stay grammatical in context. Do not add explanations."
    )
    user = (
        f"Target language: {language['name']} ({language['autonym']})\n"
        f"ISO 639-3: {language['code']}\n"
        f"Required script: {language['scripts'][0]}\n"
        "Use only the target language and its required script for natural-language text. "
        "Do not leave English fragments or introduce characters from another script, except "
        "for placeholders and [[BOX]] tokens. The array 'weekdays' lists the days of the "
        "week from Monday to Sunday in that order.\n\n"
        + (
            "Script requirement: write Serbian natural-language text in Cyrillic, not Latin, "
            "except for the protected identifiers.\n\n"
            if language["code"] == "srp"
            else ""
        )
        + (f"A previous attempt was rejected: {feedback}. Fix this.\n\n" if feedback else "")
        + json.dumps(strings, ensure_ascii=False, indent=2)
    )
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "think": False,
                "format": "json",
                "options": {"temperature": 0, "seed": seed, "num_predict": 12288},
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = str(payload.get("message", {}).get("content", ""))
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    decoded = json.loads(content)
    if not isinstance(decoded, dict):
        raise ValueError("translation is not a JSON object")
    return decoded


SCRIPT_UNICODE = {"Latn": "LATIN", "Cyrl": "CYRILLIC", "Grek": "GREEK", "Geor": "GEORGIAN"}


def foreign_letters(text: str, script: str | None) -> list[str]:
    """Letters outside the language's script; Latin is always allowed (names, placeholders)."""
    expected = SCRIPT_UNICODE.get(script or "")
    if expected is None:
        return []
    import unicodedata
    return sorted({ch for ch in text if ch.isalpha()
                   and not unicodedata.name(ch, "").startswith(("LATIN", expected))})


def validate_translation(source: Any, translated: Any, *, english: bool, path: str = "strings",
                         script: str | None = None) -> None:
    if isinstance(source, dict):
        if not isinstance(translated, dict) or source.keys() != translated.keys():
            raise ValueError(f"{path}: object keys differ")
        for key, value in source.items():
            validate_translation(value, translated[key], english=english, path=f"{path}.{key}", script=script)
        if path == "strings":
            serialized = json.dumps(translated, ensure_ascii=False)
            source_serialized = json.dumps(source, ensure_ascii=False)
            for literal in PROTECTED_LITERALS:
                if json.dumps(literal)[1:-1] in source_serialized and json.dumps(literal)[1:-1] not in serialized:
                    raise ValueError(f"protected literal disappeared: {literal}")
        return
    if isinstance(source, list):
        if not isinstance(translated, list) or len(source) != len(translated):
            raise ValueError(f"{path}: array shape changed")
        for index, value in enumerate(source):
            validate_translation(value, translated[index], english=english, path=f"{path}[{index}]", script=script)
        return
    if not isinstance(source, str) or not isinstance(translated, str) or not translated.strip():
        raise ValueError(f"{path}: expected non-empty text")
    if foreign := foreign_letters(translated, script):
        raise ValueError(f"{path}: letters from another script {foreign}; write only in the required script")
    if set(PLACEHOLDER.findall(source)) != set(PLACEHOLDER.findall(translated)):
        raise ValueError(f"{path}: placeholders changed: {sorted(set(PLACEHOLDER.findall(translated)))}")
    stripped = PLACEHOLDER.sub("", source).strip(" .,:;!?%…")
    if not english and stripped and translated.casefold().strip() == source.casefold().strip():
        raise ValueError(f"{path}: natural-language text was not translated")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
