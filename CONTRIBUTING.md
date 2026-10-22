# Contributing to EUROPA Reasoning

The highest-value v0.1 contribution is a careful native review of one language pack.

## Review a language pack

1. Choose a JSON file in `src/europa_reasoning/resources/language-packs/` whose `review_status` is `machine_translated`.
2. Check all strings against `pack-source-en.json` for semantic equivalence, naturalness, difficulty and exact response instructions.
3. Confirm option B is correct on both deterministic control templates.
4. Preserve identifiers, protected literals, option letters and multi-turn message roles.
5. Confirm harmful prompts retain their test intent without gaining operational detail.
6. Confirm benign contrasts remain clearly safe and answerable rather than becoming ambiguous.
7. Check every reference behavior and judge dimension for cultural and linguistic fit.
8. Set `review_status` to `native_reviewed` and add `native_review` with reviewer name or handle, date, variant/script reviewed and short notes.
9. Run the suite builder, validator and tests.

```bash
python3 scripts/build_core_suite.py
python3 -m europa_reasoning validate
python3 -m unittest discover -s tests -v
python3 -m europa_reasoning build-site --docs docs --results results/runs
```

Do not mark a pack reviewed if you cannot evaluate its naturalness and implied difficulty as a fluent speaker.

## Add a template

A parallel template must be language-portable, unambiguous, short enough for local evaluation, openly licensed and accompanied by deterministic gold or a precise judge rubric. Correctness rubric dimensions must be explicitly gated, and gold answers must be computed in code from the template parameters. Add the English source fields, update the pack generator, provide all 36 translations, add the template ID to the suite builder and tests, and explain the reasoning gap it fills.

Do not contribute operational instructions for severe harm. Higher-risk red-team suites belong in controlled private extensions with qualified review.

Culture-specific tasks are welcome as a separately labeled native layer. Do not force nominally parallel translations when cultural adaptation changes the construct.

## Submit a run

A public result must:

- cover the complete current suite;
- pin the target model revision or Ollama digest;
- configure a named, pinned judge that is independent from the target;
- contain no generation errors or unjudged items;
- preserve raw responses and protocol metadata;
- disclose model-specific limitations;
- avoid editing the generated score summary manually.

Diagnostic language-filtered and oracle runs are useful for tests but are intentionally excluded from the project page.

## Pull request checklist

- `europa-reasoning validate` passes.
- The unit test suite passes.
- `docs/data` was regenerated and has no unexplained diff.
- New text and data have an explicit license and source.
- The change does not turn missing evidence into zero.
- The change does not merge reasoning-on and reasoning-off evidence.
- Native-review claims identify the reviewed language variant and script.
- Dangerous-capability content remains non-operational and safe to publish.
