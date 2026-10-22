# Result artifacts

Place completed run JSON files in `results/runs/`. Resume manifests and JSONL
sidecars are local execution state and are ignored by Git.

The site builder publishes only runs that:

- cover and score every item in the current suite;
- are not diagnostic or oracle runs;
- pin both target and judge revisions; and
- use different target and judge model IDs.

This publication gate prevents a partial, self-judged or unpinned experiment
from appearing as comparable reasoning evidence.
