# Sources and provenance

## Canonical language scope

EUROPA Reasoning uses the 36 prioritized macro-languages listed by the [OpenEuroLLM training-data catalogue](https://github.com/OpenEuroLLM/training-data-catalogue/tree/main/languages) (Apache-2.0). The local registry preserves ISO 639-3 codes, BCP 47 tags, autonyms, primary scripts and catalogue variants.

## Original public-pilot items

All 24 English templates were authored for EUROPA Reasoning and released under CC0-1.0. No prompt is copied from the benchmarks below; classic puzzle *forms* (the bat-and-ball trap, knights and knaves, base-rate neglect, syllogisms) are rewritten with new wording and seeded parameters.

The source pack is `src/europa_reasoning/resources/pack-source-en.json`. Generated packs live under `src/europa_reasoning/resources/language-packs/` and retain translation model, immutable revision, decoding protocol, review state and any later corrections. Gold answers are computed by `scripts/build_core_suite.py` and recorded with their derivation in each item's `metadata.gold_calculation`.

## Design references

- [GSM8K](https://arxiv.org/abs/2110.14168) — grade-school multi-step arithmetic with a single numeric answer; the quantitative templates follow this answer format.
- [MGSM](https://arxiv.org/abs/2210.03057) — multilingual chain-of-thought on translated GSM8K problems; motivates parallel items across languages.
- [GSM-Symbolic](https://arxiv.org/abs/2410.05229) — symbolic templates with resampled numbers expose memorization and fragility; motivates seeded parametric items.
- [MATH](https://arxiv.org/abs/2103.03874) — the `\boxed{}` final-answer convention used by the scorers.
- [Cognitive Reflection Test](https://doi.org/10.1257/089533005775196732) — intuitive-but-wrong traps such as the bat-and-ball problem.
- [Do NOT Think That Much for 2+3=?](https://arxiv.org/abs/2412.21187) — overthinking in reasoning models; motivates reporting reasoning length next to accuracy.
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — reports language mixing in reasoning traces; motivates the trace-script metric.
- [EUROPA Eval](https://github.com/BirgerMoell/europa-eval) and [EUROPA Alignment](https://github.com/BirgerMoell/europa-alignment) — the runner, provenance, judge and reporting architecture from which this repository is derived.

These references have different scopes and licenses. EUROPA Reasoning links to them rather than repackaging their data.

## Translation generator

The v0.1 machine-translated packs are produced locally with the Ollama checkpoint `gemma4:31b`, revision `6316f0629137`, temperature 0, seed 42 and thinking disabled. This is generation provenance, not an endorsement or a native-quality claim.

## What a result must cite

A reproducible result identifies:

- EUROPA Reasoning suite ID, version and parameter seed;
- target model ID, immutable revision and backend;
- target decoding settings, reasoning mode and reasoning-token budget;
- independent judge model, immutable revision and protocol;
- exact item coverage and completion state; and
- model- or deployment-specific limitations.
