# EUROPA Reasoning

**One transparent reasoning protocol across the 36 languages prioritized by OpenEuroLLM.**

EUROPA Reasoning measures whether a language model reasons correctly in every European language, and *how* it gets
there. It applies the same 24 problem templates, with the same numbers, in each language: 864 items. Each model runs
twice: once in its default (direct-answer) mode and once in reasoning mode. Accuracy is reported next to process
metrics that ordinary benchmarks hide:

| Process metric | What it catches |
|---|---|
| Trace closure | the model never finishes its reasoning (no answer) |
| Loops | verbatim repetition inside the reasoning trace |
| Budget exhaustion | reasoning that runs until the token limit |
| Reasoning length | median tokens to a closed trace: verbosity and overthinking |
| Trace script | reasoning that drifts out of the prompt language's script |
| Accuracy when closed | whether finished reasoning is also correct |

The project is a sibling of [EUROPA Eval](https://github.com/BirgerMoell/europa-eval) (capabilities) and
[EUROPA Alignment](https://github.com/BirgerMoell/europa-alignment) (safety). It keeps their design: deterministic
scoring where possible, resumable runs, pinned target and judge provenance, language parity, and answer-level evidence.

## What it measures

| Domain | Templates | Scoring |
|---|---|---|
| Quantitative reasoning | change-making, work rates, speed and units, overnight durations, weighted grades, counting, sequence rules, distances | numeric, deterministic (`\boxed{}`) |
| Logical reasoning | knights and knaves, race order, weekday offsets, conditional inference, table seating, syllogisms | multiple choice, deterministic (`\boxed{}`) |
| Reasoning robustness | intuitive traps (bat and ball, price up/down), missing information, correlation vs causation, base rates | numeric or multiple choice, deterministic |
| Explanation quality | diagnosing a wrong average speed, a Fermi estimate, a parity proof, relative vs absolute risk, a cost decision | named LLM judge, item rubric (0–4 per dimension), correctness as a critical gate |

Every final answer for a deterministic item must end in `\boxed{}`; the scorer reads only the last box. In reasoning
mode only the text after the closed trace is scored: **an unfinished trace yields no answer**.

### Parametric items

Numbers, names and option order come from a generator seeded per template; the gold answer is computed in code, never
translated. Placeholders are filled after translation, so the same packs build any variant:

```bash
python3 scripts/build_core_suite.py --seed 7   # fresh variant suite for a memorization check
```

The published instance uses seed 2026.

## Language scope and review status

The canonical scope follows the [OpenEuroLLM training-data catalogue](https://github.com/OpenEuroLLM/training-data-catalogue/tree/main/languages):
24 EU official languages plus Catalan, Basque, Galician, Bosnian, Georgian, Macedonian, Albanian, Serbian, Turkish,
Ukrainian, Icelandic and Norwegian.

Version 0.1 is a **translation-review pilot**:

- English is the authored source pack.
- The other 35 packs are machine translated with a pinned local model (`gemma4:31b`, temperature 0) and remain
  `machine_translated` until a native speaker reviews them. Placeholders and `\boxed{}` examples are checked after
  translation.
- 24 items per language are diagnostic, not comprehensive; per-language intervals are wide.
- Translation quality is uneven. Every pack passed automatic checks (placeholders, `\boxed{}` examples, and no letters
  from another script, which caught CJK, Hangul, Thai, Arabic and Cyrillic fragments the translator inserted). Three
  packs carry documented manual script fixes in `quality_corrections` (Slovene, Georgian, Maltese). **Maltese is the
  weakest pack** and needs native review before its scores are interpreted; meaning errors that keep the right script
  (for example a wrong adjective) are not caught automatically in any language.

## Quick start

The core has no runtime dependencies beyond Python 3.11+.

```bash
git clone https://github.com/BirgerMoell/europa-reasoning.git
cd europa-reasoning
python3 -m pip install -e .
europa-reasoning validate
europa-reasoning list
```

### Hugging Face Transformers, both modes

```bash
python3 -m pip install -e '.[local]'
# default mode
europa-reasoning run --backend huggingface --model ORG/MODEL --revision SHA --dtype bfloat16 \
  --judge-backend ollama --judge-model gemma4:31b --judge-revision JUDGE_DIGEST \
  --output results/runs/model-default.json
# reasoning mode: enable_thinking=True in the chat template, 8,192 extra tokens for the trace
europa-reasoning run --backend huggingface --model ORG/MODEL --revision SHA --dtype bfloat16 \
  --think --reasoning-tokens 8192 \
  --judge-backend ollama --judge-model gemma4:31b --judge-revision JUDGE_DIGEST \
  --output results/runs/model-think.json
```

`--think` works with `huggingface` (the template's `enable_thinking` switch), `openai-compatible` (sent as
`chat_template_kwargs`; a separate `reasoning_content` field is honoured) and `ollama` (native `think`).

### Judging later

Open explanations can be judged without regenerating answers:

```bash
europa-reasoning judge results/runs/model-think.json --backend ollama --model gemma4:31b --revision JUDGE_DIGEST
```

## Integrity rules

- A public run pins the target and judge revisions; target and judge must differ.
- Judges see only the final answer, never the reasoning trace.
- Run artifacts keep traces for process metrics; the public site publishes answers, scores and process metrics, never
  traces.
- The suite is public and may enter training data; use a fresh `--seed` variant for release decisions.

## Development

```bash
python3 -m pip install -e '.[dev]'
make check
```

Licensed under Apache-2.0 (code) and CC0-1.0 (items); see `LICENSE` and `DATA_LICENSE`.
