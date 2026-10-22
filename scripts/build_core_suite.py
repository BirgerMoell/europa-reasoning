#!/usr/bin/env python3
"""Build the parallel 36-language EUROPA Reasoning suite from the translated packs.

Every template draws its numbers, names and option order from a generator seeded by
(parameter seed, template id), so all languages receive the same problem instance and the gold
answer is computed here, never translated. Placeholders are filled after translation, so one set of
translated packs serves any parameter seed: `--seed N` builds a fresh variant suite for checking
memorization of the published instance.
"""

from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
RESOURCE_DIR = ROOT / "src/europa_reasoning/resources"
LANGUAGES_PATH = RESOURCE_DIR / "languages.json"
PACK_DIR = RESOURCE_DIR / "language-packs"
REPOSITORY = "https://github.com/BirgerMoell/europa-reasoning"
VERSION = "0.1.0"
DEFAULT_SEED = 2026
NAMES = ["Ana", "Leo", "Mila", "Tom", "Nina", "Erik", "Sara", "Jan", "Eva", "Max", "Lena", "Ivo"]
NONSENSE = ["blickets", "daxes", "wugs", "toves", "gorps", "fendles"]
LETTERS = "ABCDEF"

DOMAINS = {
    "quantitative_reasoning": ("Quantitative reasoning", "Multi-step arithmetic, rates, units, time and counting with a single numeric answer."),
    "logical_reasoning": ("Logical reasoning", "Deduction from stated constraints: truth-tellers, orderings, calendars, conditionals and syllogisms."),
    "reasoning_robustness": ("Reasoning robustness", "Classic traps, missing information, correlation versus causation and base rates."),
    "explanation_quality": ("Explanation quality", "Open explanations judged for correctness, valid argument and clarity."),
}


class Template:
    def __init__(self, template_id: str, domain: str, capability: str, task_type: str,
                 build: Callable[[random.Random, dict[str, Any], dict[str, Any]], dict[str, Any]]):
        self.id, self.domain, self.capability, self.task_type, self.build = template_id, domain, capability, task_type, build


def fill(text: str, values: dict[str, Any]) -> str:
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    if "{" in text.replace("\\boxed{}", "").replace("\\boxed{42}", "").replace("\\boxed{B}", ""):
        raise ValueError(f"unfilled placeholder in: {text[:120]}")
    return text


def numeric(s: dict[str, Any], prompt_key: str, values: dict[str, Any], gold: int | float, calc: str) -> dict[str, Any]:
    return {
        "prompt": fill(s[prompt_key], values) + "\n\n" + s["numeric_instruction"],
        "gold": {"value": gold},
        "scoring": {"type": "numeric", "extract": "boxed", "tolerance": 1e-6},
        "metadata": {"gold_calculation": calc, "parameters": values},
        "max_tokens": 1024,
    }


def choice(s: dict[str, Any], rng: random.Random, prompt_key: str, values: dict[str, Any],
           options: list[tuple[str, str]], correct: str, rationale: str,
           record: dict[str, Any] | None = None) -> dict[str, Any]:
    """`options` are (key, text) pairs; the order is shuffled by the template's generator."""
    order = list(options)
    rng.shuffle(order)
    rendered = [f"{LETTERS[i]}. {fill(text, values)}" for i, (_, text) in enumerate(order)]
    answer = LETTERS[[key for key, _ in order].index(correct)]
    return {
        "prompt": fill(s[prompt_key], values) + "\n\n" + s["choice_instruction"],
        "options": rendered,
        "gold": {"answer": answer},
        "scoring": {"type": "choice", "extract": "boxed"},
        # `record` keeps the recorded parameters language-neutral when a value is a translated word.
        "metadata": {"gold_calculation": rationale, "parameters": record if record is not None else values},
        "max_tokens": 1024,
    }


def rubric(s: dict[str, Any], prompt_key: str, reference_key: str, values: dict[str, Any],
           dimensions: dict[str, str], critical: list[str], max_words: int, required: list[str]) -> dict[str, Any]:
    return {
        "prompt": fill(s[prompt_key], values),
        "gold": {"reference_answer": fill(s[reference_key], values), "required_points": required},
        "scoring": {
            "type": "rubric",
            "dimensions": dimensions,
            "critical_dimensions": {name: 3 for name in critical},
            "pass_threshold": 0.75,
            "response_constraints": {"forbid_prompt_echo": True, "max_words": max_words, "score_cap": 0.75},
        },
        "metadata": {"parameters": values},
        "max_tokens": 600,
    }


# ----------------------------------------------------------------------------- quantitative

def t_shopping(rng, s, lang):
    a, p, b, q = rng.randint(3, 6), rng.randint(2, 5), rng.randint(2, 5), rng.randint(1, 3)
    v = {"name1": rng.choice(NAMES), "a": a, "p": p, "b": b, "q": q, "note": 50}
    gold = 50 - (a * p + b * q)
    return numeric(s, "shopping_prompt", v, gold, f"50 - ({a}*{p} + {b}*{q}) = {gold}")


def t_pumps(rng, s, lang):
    a, b = rng.choice([(3, 6), (4, 12), (6, 12), (10, 15), (12, 24), (6, 30)])
    gold = Fraction(a * b, a + b)
    return numeric(s, "pumps_prompt", {"a": a, "b": b}, int(gold), f"1/(1/{a} + 1/{b}) = {a}*{b}/({a}+{b}) = {gold}")


def t_train(rng, s, lang):
    d, m = rng.choice([(45, 30), (70, 40), (36, 24), (150, 90), (55, 33), (84, 48)])
    gold = Fraction(d * 60, m)
    return numeric(s, "train_speed_prompt", {"d": d, "m": m}, int(gold), f"{d} km / ({m}/60 h) = {gold} km/h")


def t_night_shift(rng, s, lang):
    h1, m1 = rng.choice([21, 22, 23]), rng.choice([15, 30, 45])
    h2, m2 = rng.choice([5, 6, 7]), rng.choice([0, 10, 20, 40])
    gold = (24 * 60 - (h1 * 60 + m1)) + h2 * 60 + m2
    v = {"start": f"{h1:02d}:{m1:02d}", "end": f"{h2:02d}:{m2:02d}"}
    return numeric(s, "night_shift_prompt", v, gold, f"(24:00 - {v['start']}) + {v['end']} = {gold} min")


def t_grade(rng, s, lang):
    while True:
        w1 = rng.choice([20, 30, 40, 60])
        w2 = 100 - w1
        s1, t = rng.randint(50, 95), rng.randint(60, 85)
        need = Fraction(t * 100 - w1 * s1, w2)
        if need.denominator == 1 and 0 < need <= 100:
            break
    v = {"w1": w1, "w2": w2, "s1": s1, "t": t, "name1": rng.choice(NAMES)}
    return numeric(s, "grade_prompt", v, int(need), f"({t}*100 - {w1}*{s1}) / {w2} = {need}")


def t_handshakes(rng, s, lang):
    n = rng.randint(8, 16)
    return numeric(s, "handshakes_prompt", {"n": n}, n * (n - 1) // 2, f"{n}*({n}-1)/2 = {n * (n - 1) // 2}")


def t_sequence(rng, s, lang):
    c = rng.randint(1, 4)
    terms = [k * k + c * k for k in range(1, 7)]
    seq = ", ".join(str(x) for x in terms[:5])
    return numeric(s, "sequence_prompt", {"seq": seq}, terms[5], f"a(k) = k^2 + {c}k; a(6) = {terms[5]}")


def t_walk(rng, s, lang):
    dy, dx, h = rng.choice([(3, 4, 5), (6, 8, 10), (5, 12, 13), (8, 6, 10), (9, 12, 15)])
    c = rng.choice([20, 30, 50])
    v = {"name1": rng.choice(NAMES), "a": dy * 10 + c, "b": dx * 10, "c": c}
    return numeric(s, "walk_prompt", v, h * 10, f"north net {dy * 10} m, east {dx * 10} m; sqrt({dy * 10}^2+{dx * 10}^2) = {h * 10}")


# ----------------------------------------------------------------------------- robustness

def t_racket(rng, s, lang):
    total, diff = rng.choice([(110, 100), (220, 200), (330, 300), (120, 100), (240, 200)])
    gold = (total - diff) // 2
    return numeric(s, "racket_prompt", {"total": total, "diff": diff}, gold,
                   f"ball x, racket x+{diff}: 2x+{diff}={total} -> x={gold} (intuitive wrong answer {total - diff})")


def t_price_change(rng, s, lang):
    x, p = rng.choice([(20, 100), (20, 250), (10, 200), (50, 120), (25, 160)])
    gold = Fraction(p) * (100 + x) * (100 - x) / 10000
    return numeric(s, "price_change_prompt", {"p": p, "x": x}, int(gold),
                   f"{p}*(1+{x}/100)*(1-{x}/100) = {gold} (not {p})")


def t_bakery(rng, s, lang):
    a, b = rng.randint(30, 60), rng.randint(61, 90)
    v = {"a": a, "b": b, "sum": a + b, "next": a + 1, "diff": b - a}
    opts = [("sum", s["bakery_option_sum"]), ("next", s["bakery_option_next"]),
            ("diff", s["bakery_option_diff"]), ("unknown", s["bakery_option_unknown"])]
    return choice(s, rng, "bakery_prompt", v, opts, "unknown", "Tuesday is only bounded below; no value is determined")


def t_correlation(rng, s, lang):
    opts = [(k, s[f"correlation_option_{k}"]) for k in ("cause", "ban", "confounder", "reverse")]
    return choice(s, rng, "correlation_prompt", {}, opts, "confounder", "correlation with a plausible common cause (weather)")


def t_base_rate(rng, s, lang):
    p, f = rng.choice([(1, 5), (2, 5), (5, 5), (10, 5), (1, 2), (5, 2)])
    prob = p / (p + f / 100 * (1000 - p))
    correct = round(100 * prob)
    candidates = [correct, 100 - f, 50, f]
    while len(set(candidates)) < 4:
        candidates[-1] += 7
    keys = ["correct", "complement", "half", "false_positive"]
    v = {"p": p, "f": f}
    opts = [(k, s["base_rate_option"].replace("{value}", str(val))) for k, val in zip(keys, candidates)]
    return choice(s, rng, "base_rate_prompt", v, opts, "correct",
                  f"{p}/({p} + {f}% of {1000 - p}) = {p}/{p + f / 100 * (1000 - p):.1f} = {100 * prob:.1f}%")


# ----------------------------------------------------------------------------- logic

def t_knights(rng, s, lang):
    n1, n2 = rng.sample(NAMES, 2)
    v = {"name1": n1, "name2": n2}
    opts = [(k, s[f"knights_option_{k}"]) for k in ("both_knights", "both_knaves", "first_knight", "first_knave")]
    return choice(s, rng, "knights_prompt", v, opts, "first_knave",
                  "a knight cannot claim to be a knave, so name1 lies; 'both knaves' is false, so name2 is a knight")


def t_race(rng, s, lang):
    names = rng.sample(NAMES, 5)
    v = {f"n{i + 1}": name for i, name in enumerate(names)}
    opts = [(f"n{i + 1}", "{" + f"n{i + 1}" + "}") for i in range(5)]
    return choice(s, rng, "race_prompt", v, opts, "n4", "only order: n1, n2, n4, n3, n5 -> third is n4")


def t_weekday(rng, s, lang):
    days = s["weekdays"]
    today = rng.randrange(7)
    n = rng.choice([k for k in range(40, 200) if k % 7])
    target = (today + n) % 7
    distract = [(target + 1) % 7, (target - 1) % 7, (target + 3) % 7]
    v = {"today": days[today], "n": n}
    opts = [("correct", days[target])] + [(f"d{i}", days[d]) for i, d in enumerate(distract)]
    return choice(s, rng, "weekday_prompt", v, opts, "correct", f"({today} + {n}) mod 7 = {target} (0 = Monday)",
                  record={"today_index": today, "n": n})


def t_conditional(rng, s, lang):
    opts = [(k, s[f"conditional_option_{k}"]) for k in ("on", "off", "broken", "nothing")]
    return choice(s, rng, "conditional_prompt", {}, opts, "off", "modus tollens: not green -> alarm not on")


def t_seating(rng, s, lang):
    names = rng.sample(NAMES, 4)
    v = {f"n{i + 1}": name for i, name in enumerate(names)}
    opts = [(f"n{i + 1}", "{" + f"n{i + 1}" + "}") for i in range(4)]
    return choice(s, rng, "seating_prompt", v, opts, "n4", "n1's neighbours are n3 and n4; n3 is on the left, so n4 is on the right")


def t_syllogism(rng, s, lang):
    w1, w2, w3 = rng.sample(NONSENSE, 3)
    v = {"w1": w1, "w2": w2, "w3": w3}
    opts = [(k, s[f"syllogism_option_{k}"]) for k in ("yes", "no", "all")]
    return choice(s, rng, "syllogism_prompt", v, opts, "no", "the w2 that are w3 need not include any w1")


# ----------------------------------------------------------------------------- explanation

LANGQ = "Is clear, well organized and natural in {lang}."


def t_average_speed(rng, s, lang):
    v1, v2 = rng.choice([(60, 40), (90, 60), (30, 60), (120, 80), (60, 120)])
    d = 120
    t1, t2 = Fraction(d, v1), Fraction(d, v2)
    answer = Fraction(2 * d) / (t1 + t2)
    v = {"d": d, "v1": v1, "v2": v2, "wrong": (v1 + v2) // 2, "t1": _num(t1), "t2": _num(t2),
         "total_time": _num(t1 + t2), "total_distance": 2 * d, "answer": _num(answer)}
    return rubric(s, "average_speed_prompt", "average_speed_reference", v, {
        "correctness": "States that the student is wrong and why (total distance / total time; the slower leg takes longer).",
        "final_value": f"Gives the correct average speed of {_num(answer)} km/h.",
        "reasoning_quality": "The explanation is logically valid and the arithmetic shown is correct.",
        "language_quality": LANGQ.format(lang=lang),
    }, critical=["correctness"], max_words=150, required=["student is wrong", f"{_num(answer)} km/h"])


def t_fermi(rng, s, lang):
    k = rng.randint(2, 5)
    v = {"k": k, "low": k * 30 * 365, "high": k * 120 * 365}
    return rubric(s, "fermi_prompt", "fermi_reference", v, {
        "assumptions": "States explicit, plausible assumptions (showers per day, duration, flow rate).",
        "calculation": "The arithmetic from the assumptions to the result is correct and shown.",
        "plausibility": f"The final estimate is in a plausible range (roughly {v['low']} to {v['high']} litres per year).",
        "language_quality": LANGQ.format(lang=lang),
    }, critical=["calculation"], max_words=150, required=["assumptions", "yearly total"])


def t_parity(rng, s, lang):
    return rubric(s, "parity_prompt", "parity_reference", {}, {
        "validity": "Gives a correct general argument (for example 2a+1 + 2b+1 = 2(a+b+1)).",
        "generality": "Does not rely on examples alone; the argument covers all odd numbers.",
        "clarity": "The steps are easy to follow.",
        "language_quality": LANGQ.format(lang=lang),
    }, critical=["validity", "generality"], max_words=120, required=["general form of odd numbers", "sum is a multiple of 2"])


def t_risk(rng, s, lang):
    a, b = rng.choice([(2, 1), (4, 3), (5, 2), (8, 6), (10, 7)])
    rel = round(100 * (a - b) / a)
    v = {"a": a, "b": b, "rel": rel, "abs_diff": a - b, "abs_pct": _num(Fraction(a - b, 10)), "nnt": round(1000 / (a - b))}
    return rubric(s, "risk_prompt", "risk_reference", v, {
        "relative_risk": f"Correctly identifies the relative reduction ({rel}%).",
        "absolute_risk": f"Correctly computes the absolute reduction ({a - b} in 1000, i.e. {v['abs_pct']} percentage points).",
        "judgement": "Explains that reporting only the relative figure makes a small absolute benefit look large.",
        "language_quality": LANGQ.format(lang=lang),
    }, critical=["absolute_risk", "judgement"], max_words=150, required=["relative reduction", "absolute reduction", "misleading"])


def t_plans(rng, s, lang):
    while True:
        fa, ca, fb = rng.choice([5, 8, 10]), rng.choice([5, 10, 20]), rng.choice([20, 25, 30])
        breakeven = Fraction((fb - fa) * 100, ca)
        m = rng.choice([60, 90, 120, 150, 200, 250])
        cost_a = Fraction(fa) + Fraction(m * ca, 100)
        if breakeven.denominator == 1 and cost_a != fb and abs(cost_a - fb) >= 1:
            break
    cheaper = "A" if cost_a < fb else "B"
    v = {"fa": fa, "ca": ca, "fb": fb, "m": m, "name1": rng.choice(NAMES), "ca_euro": _num(Fraction(ca, 100)),
         "cost_a": _num(cost_a), "cheaper": cheaper, "saving": _num(abs(cost_a - fb)), "breakeven": int(breakeven)}
    return rubric(s, "plans_prompt", "plans_reference", v, {
        "correct_choice": f"Identifies plan {cheaper} as cheaper at {m} minutes.",
        "amounts": f"Gives the monthly costs ({v['cost_a']} and {fb} euros) and the difference ({v['saving']} euros).",
        "break_even": f"Gives the break-even point of {int(breakeven)} minutes per month, above which plan B is cheaper.",
        "language_quality": LANGQ.format(lang=lang),
    }, critical=["correct_choice", "amounts"], max_words=150, required=[f"plan {cheaper}", f"{int(breakeven)} minutes"])


def _num(value: Fraction | float | int) -> str:
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


TEMPLATES = [
    Template("shopping-change", "quantitative_reasoning", "multi_step_arithmetic", "numeric", t_shopping),
    Template("work-rate", "quantitative_reasoning", "rates", "numeric", t_pumps),
    Template("speed-units", "quantitative_reasoning", "unit_conversion", "numeric", t_train),
    Template("overnight-duration", "quantitative_reasoning", "time_arithmetic", "numeric", t_night_shift),
    Template("weighted-grade", "quantitative_reasoning", "equation_solving", "numeric", t_grade),
    Template("handshakes", "quantitative_reasoning", "counting", "numeric", t_handshakes),
    Template("sequence-rule", "quantitative_reasoning", "pattern_induction", "numeric", t_sequence),
    Template("walk-distance", "quantitative_reasoning", "spatial_geometry", "numeric", t_walk),
    Template("racket-ball", "reasoning_robustness", "intuitive_trap", "numeric", t_racket),
    Template("price-up-down", "reasoning_robustness", "percentage_trap", "numeric", t_price_change),
    Template("insufficient-information", "reasoning_robustness", "insufficient_information", "multiple_choice", t_bakery),
    Template("correlation-causation", "reasoning_robustness", "causal_reasoning", "multiple_choice", t_correlation),
    Template("base-rate", "reasoning_robustness", "probabilistic_reasoning", "multiple_choice", t_base_rate),
    Template("knights-knaves", "logical_reasoning", "truth_teller_logic", "multiple_choice", t_knights),
    Template("race-order", "logical_reasoning", "ordering_constraints", "multiple_choice", t_race),
    Template("weekday-offset", "logical_reasoning", "calendar_reasoning", "multiple_choice", t_weekday),
    Template("conditional-inference", "logical_reasoning", "propositional_logic", "multiple_choice", t_conditional),
    Template("table-seating", "logical_reasoning", "spatial_constraints", "multiple_choice", t_seating),
    Template("syllogism", "logical_reasoning", "quantifier_logic", "multiple_choice", t_syllogism),
    Template("average-speed-error", "explanation_quality", "error_diagnosis", "open_explanation", t_average_speed),
    Template("fermi-estimate", "explanation_quality", "estimation", "open_explanation", t_fermi),
    Template("parity-proof", "explanation_quality", "proof", "open_explanation", t_parity),
    Template("relative-absolute-risk", "explanation_quality", "statistical_literacy", "open_explanation", t_risk),
    Template("plan-comparison", "explanation_quality", "decision_analysis", "open_explanation", t_plans),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="parameter seed (default: published instance)")
    args = ap.parse_args()
    registry = read_json(LANGUAGES_PATH)
    languages = registry["languages"]
    if len(languages) != 36 or len({language["code"] for language in languages}) != 36:
        raise ValueError("the canonical registry must contain exactly 36 unique languages")
    suffix = "" if args.seed == DEFAULT_SEED else f"-seed{args.seed}"
    suite_dir = RESOURCE_DIR / f"suites/europa-reasoning-core-v0.1{suffix}"

    items: list[dict[str, Any]] = []
    review_counts: dict[str, int] = {}
    for language in languages:
        pack = read_json(PACK_DIR / f"{language['code']}.json")
        if pack["language"]["code"] != language["code"]:
            raise ValueError(f"pack identity mismatch for {language['code']}")
        review_counts[str(pack["review_status"])] = review_counts.get(str(pack["review_status"]), 0) + len(TEMPLATES)
        items.extend(build_language_items(language, pack, args.seed))

    metadata = build_metadata(registry, review_counts, args.seed)
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "suite.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (suite_dir / "dev.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in items), encoding="utf-8")
    print(f"built {len(items)} items across {len(languages)} languages (parameter seed {args.seed}) -> {suite_dir}")
    return 0


def build_language_items(language: dict[str, Any], pack: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    s = pack["strings"]
    code = language["code"]
    common_meta = {
        "language_name": language["name"],
        "language_autonym": language["autonym"],
        "bcp47": language["bcp47"],
        "catalogue_variants": language["variants"],
        "translation_generator": pack["translation_generator"],
        "translation_model_revision": pack.get("translation_model_revision"),
        "translation_protocol": pack["translation_protocol"],
        "parameter_seed": seed,
        **({"native_review": pack["native_review"]} if pack.get("native_review") else {}),
    }
    out = []
    for template in TEMPLATES:
        rng = random.Random(f"{seed}:{template.id}")
        body = template.build(rng, s, language["name"])
        metadata = {**common_meta, **body.pop("metadata", {})}
        out.append({
            "id": f"europa-reason-v01-{code}-{template.id}",
            "suite_id": "europa-reasoning-core",
            "suite_version": VERSION,
            "split": "dev",
            "language": code,
            "script": language["scripts"][0],
            "template_id": template.id,
            "domain": template.domain,
            "capability": template.capability,
            "task_type": template.task_type,
            "source": {"kind": "original", "title": "EUROPA Reasoning original parallel reasoning task",
                       "url": REPOSITORY, "license": "CC0-1.0", "accessed": "2026-10-22"},
            "review_status": pack["review_status"],
            "tags": ["parallel", "reasoning", code, language["group"]],
            **body,
            "metadata": metadata,
        })
    return out


def build_metadata(registry: dict[str, Any], review_counts: dict[str, int], seed: int) -> dict[str, Any]:
    languages = registry["languages"]
    codes = [language["code"] for language in languages]
    return {
        "id": "europa-reasoning-core",
        "version": VERSION,
        "name": "EUROPA Reasoning 36-language public pilot",
        "status": "translation-review-pilot",
        "tagline": "One transparent reasoning protocol across 36 European languages.",
        "split": "dev",
        "public_development_set": True,
        "parameter_seed": seed,
        "data_file": "dev.jsonl",
        "languages": codes,
        "language_count": len(codes),
        "language_source": registry["source"],
        "language_labels": {language["code"]: language["name"] for language in languages},
        "language_autonyms": {language["code"]: language["autonym"] for language in languages},
        "language_groups": {language["code"]: language["group"] for language in languages},
        "language_scripts": {language["code"]: language["scripts"] for language in languages},
        "language_variants": {language["code"]: language["variants"] for language in languages},
        "template_ids": [template.id for template in TEMPLATES],
        "template_count": len(TEMPLATES),
        "items_per_language": len(TEMPLATES),
        "review_counts": review_counts,
        "domains": list(DOMAINS),
        "domain_labels": {key: label for key, (label, _) in DOMAINS.items()},
        "domain_descriptions": {key: description for key, (_, description) in DOMAINS.items()},
        "task_types": ["numeric", "multiple_choice", "open_explanation"],
        "task_type_labels": {"numeric": "Numeric answer", "multiple_choice": "Multiple choice", "open_explanation": "Open explanation"},
        "methodology": [
            "Apply the same 24 problem templates with identical numbers in every language so cross-language reasoning gaps remain visible.",
            "Generate numbers, names and option order from a seeded generator and compute every gold answer in code; a new seed yields a fresh variant to test memorization.",
            "Score final answers deterministically from \\boxed{} where possible and use a pinned, independent LLM judge with item-specific rubrics for open explanations.",
            "Run each model in default and reasoning mode and report process metrics next to accuracy: trace closure, loops, budget exhaustion, reasoning length and trace script.",
            "Score only the visible final answer; an unfinished reasoning trace yields no answer.",
            "Publish final answers, scores and process metrics, never private reasoning traces.",
        ],
        "limitations": [
            "Version 0.1 is a public development and translation-review pilot, not a definitive reasoning leaderboard.",
            "Thirty-five packs are machine translated from authored English templates and require native-speaker review.",
            "Twenty-four items per language are intentionally lightweight; per-language scores have wide confidence intervals.",
            "The problems are short, school-level and text-only; they do not measure research-level mathematics or long-horizon agentic reasoning.",
            "LLM-judge scores for open explanations require multilingual human calibration.",
            "Public prompts may enter future training data; use a fresh parameter seed or private extensions for release decisions.",
        ],
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
