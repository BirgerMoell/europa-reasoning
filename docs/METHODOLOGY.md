# EUROPA Reasoning methodology

## Purpose

EUROPA Reasoning is a lightweight reasoning probe across the 36 macro-languages prioritized by OpenEuroLLM. It asks
two questions per language: does the model reach correct answers, and does its reasoning process behave well —
does it finish, avoid loops, stay in the prompt's language and spend a sensible number of tokens?

Version 0.1 is a public development and translation-review pilot, not a definitive leaderboard.

## Parallel, parametric design

Every language has exactly 24 items derived from the same authored English templates. Stable IDs use:

```text
europa-reason-v01-{iso639-3}-{template-id}
```

Numbers, names, invented words and option order are drawn from a generator seeded by `(parameter seed, template id)`,
so every language receives the same problem instance. Gold answers are computed in code and stored with their
derivation (`metadata.gold_calculation`). Because placeholders are filled after translation, a new seed rebuilds a
fresh variant from the same packs; comparing a model on the published seed and on a fresh seed separates reasoning
from memorization.

## Domains

1. **Quantitative reasoning** (8): change-making, work rates, speed and unit conversion, overnight durations, weighted
   grades, counting, sequence rules and straight-line distance.
2. **Logical reasoning** (6): knights and knaves, ordering constraints, weekday offsets, modus tollens, seating
   constraints and syllogisms with invented words.
3. **Reasoning robustness** (5): intuitive traps (bat-and-ball, equal percentage up and down), a question whose answer
   is not determined by the data, correlation versus causation and base-rate neglect.
4. **Explanation quality** (5): diagnosing a wrong average speed, a Fermi estimate, a parity proof, relative versus
   absolute risk and a cost decision with a break-even point.

## Modes

Each model is run twice with identical items:

- **default mode** — the chat template's normal mode (for hybrid models, thinking off);
- **reasoning mode** — thinking on (`--think`), with a separate reasoning-token budget (`--reasoning-tokens`, 8,192
  in published runs) added to each item's answer budget.

The reasoning trace is separated from the final answer (`<think> … </think>` in the template or a separate
`reasoning_content` / `thinking` field from the API). **Only the final answer is scored.** If the trace never closes,
the item has no answer and scores zero; text inside an unfinished trace is never graded.

## Scoring

### Deterministic items (19 per language)

Numeric and multiple-choice items require the final answer in `\boxed{}`. The scorer takes the last box in the final
answer, strips harmless LaTeX (`\text{}`, thin spaces, `%`, currency signs, `(C)` wrappers) and compares it with the
computed gold: numbers with an absolute tolerance of 1e-6 (decimal commas accepted), letters exactly. A missing box is
recorded as `boxed_found: false` and scores zero.

### Rubric items (5 per language)

Open explanations are scored by a named, pinned LLM judge that sees the task, the final answer, a reference answer and
item-specific dimensions, each scored 0–4. Correctness dimensions are critical: a score below 3 caps the item at that
dimension's level, so a fluent but wrong explanation cannot average its way to a pass. A word limit stated in the
prompt is enforced deterministically (exceeding it caps the item at 0.75). The judge never sees the reasoning trace.

## Process metrics

Computed deterministically from each generation and reported overall and per language:

| Metric | Definition |
|---|---|
| Trace closure rate | reasoning-mode generations whose trace closed before the budget |
| Loop rate | traces containing a repeated, non-overlapping 24-token span |
| Budget exhaustion | generations that stopped at the token limit |
| Median / p90 reasoning tokens | tokens before the closing tag, closed traces only |
| Trace script share | share of trace letters in the language's declared script (detects drift to Latin-script English in Greek, Cyrillic or Georgian packs) |
| Detected trace language | optional `langdetect` guess per trace (install the `process` extra) |
| Think tag in answer | think delimiters leaking into a default-mode answer |
| Accuracy when closed | accuracy restricted to closed traces |

## Limitations

- The problems are short, school-level and text-only.
- Thirty-five packs are machine translated and need native review; placeholders keep the arithmetic identical, but
  wording quality varies.
- 24 items per language give wide per-language intervals; compare languages by pattern, not by single points.
- LLM-judge scores for the five explanation items need multilingual human calibration.
- The trace-script metric cannot detect English drift inside Latin-script languages; the optional language detector
  covers only languages it supports.
- Public items may enter training data; use a fresh parameter seed for release decisions.
