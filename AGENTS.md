# EUROPA Reasoning agent guide

EUROPA Reasoning is a 36-language reasoning benchmark that reports accuracy together with reasoning-process metrics.
Keep these contracts intact:

- Treat `src/europa_reasoning/resources/suites/` as canonical bundled suite data; rebuild it with
  `scripts/build_core_suite.py`, never edit it by hand.
- Keep exactly the declared template coverage for every declared language, with identical parameters across languages.
- Compute every gold answer in code from the template parameters; never translate or hand-enter a gold value.
- Translations must keep every `{placeholder}` and every `\boxed{}` example unchanged.
- Every item needs language, script, template, provenance, license, scoring, domain, capability, task type and review state.
- Deterministic items are scored from the last `\boxed{}` in the visible final answer only; an unclosed reasoning
  trace yields no answer and must not be scored from text inside the trace.
- Rubric correctness dimensions are gates; never average a critical failure away.
- Preserve raw model output, the reasoning trace (in the run artifact only), and the exact target and judge protocols.
- Never turn missing or unjudged evidence into zero.
- Do not publish a full run unless it uses a pinned independent judge and a pinned target revision.
- Never present diagnostic, oracle, partial, machine-translation-only or protocol-mismatched runs as a leaderboard.
- Do not publish reasoning traces. Publish final answers, deterministic evidence, process metrics and concise judge rationales.
- Keep the dependency-free core working on Python 3.11+ (`langdetect` is an optional extra).
- API keys are read only from named environment variables and are never saved.
- Use `python3 -m europa_reasoning validate`, `python3 -m unittest discover -s tests -v`, and
  `python3 -m europa_reasoning build-site` before publishing.

The public pilot is a development and translation-review instrument. Promote a language pack only after native review
is recorded in committed metadata.
