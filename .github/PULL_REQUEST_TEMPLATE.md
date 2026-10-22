## What changed

<!-- Describe the reasoning template, language pack, harness, site or result change. -->

## Evidence

- [ ] `python scripts/audit_language_packs.py`
- [ ] `europa-reasoning validate`
- [ ] `python -m unittest discover -s tests -v`
- [ ] `europa-reasoning build-site --docs docs --results results/runs`

## Integrity checks

- [ ] Sources and licenses are explicit.
- [ ] Missing evidence is not converted to zero.
- [ ] Target model, judge and reasoning protocol provenance remains separate.
- [ ] Any native-review claim names the reviewed language variant and script.
- [ ] Harmful-capability content remains non-operational and safe to publish.
