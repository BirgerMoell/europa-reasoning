import tempfile
import unittest
from pathlib import Path

from europa_reasoning.backends import GenerationConfig, OracleBackend
from europa_reasoning.data import load_suite
from europa_reasoning.judging import judge_artifact, judge_run
from europa_reasoning.runner import run_evaluation


class OfflineJudgingTests(unittest.TestCase):
    def test_judges_preserved_response_without_target_generation(self):
        metadata, items = load_suite()
        rubric_item = next(item for item in items if item.scoring["type"] == "rubric")
        run = {
            "status": "partial",
            "suite": {"id": metadata["id"], "version": metadata["version"]},
            "model": {"id": "target", "revision": "target-rev", "backend": "test"},
            "judge": None,
            "protocol": {},
            "samples": [
                {
                    "item_id": rubric_item.id,
                    "response": rubric_item.gold["reference_answer"],
                    "score": None,
                    "passed": None,
                    "scorer": "rubric",
                    "parsed": None,
                    "score_details": {"status": "judge_required"},
                    "scoring_error": None,
                    "latency_ms": 1.0,
                    "error": None,
                }
            ],
            "limitations": [
                "The run is incomplete and must not be compared as full-suite evidence.",
                "Rubric-scored items require a configured judge and remain unscored here.",
            ],
        }

        updated = judge_run(
            run=run,
            suite_metadata=metadata,
            items=items,
            judge_backend=OracleBackend(),
            judge_revision="oracle-rev",
            config=GenerationConfig(),
        )

        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["samples"][0]["score"], 1.0)
        self.assertEqual(updated["samples"][0]["response"], run["samples"][0]["response"])
        self.assertEqual(updated["judge"]["revision"], "oracle-rev")
        self.assertEqual(updated["protocol"]["judge_temperature"], 0.0)
        self.assertEqual(
            updated["protocol"]["judge_system_prompt"],
            GenerationConfig().system_prompt,
        )
        self.assertTrue(updated["judging_history"][0]["target_responses_reused"])
        self.assertEqual(updated["limitations"], [])

    def test_artifact_updates_resume_files(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata, items = load_suite()
            rubric_item = next(item for item in items if item.scoring["type"] == "rubric")
            output = Path(directory) / "offline.json"
            run_evaluation(
                suite_metadata=metadata,
                items=[rubric_item],
                backend=OracleBackend(),
                output=output,
                config=GenerationConfig(),
            )

            updated = judge_artifact(
                path=output,
                suite_metadata=metadata,
                items=items,
                judge_backend=OracleBackend(),
                judge_revision="oracle-rev",
                config=GenerationConfig(),
            )

            self.assertEqual(updated["status"], "completed")
            self.assertEqual(len(output.with_suffix(".samples.jsonl").read_text().splitlines()), 2)
            self.assertIn(
                '"judge_revision": "oracle-rev"',
                output.with_suffix(".manifest.json").read_text(),
            )
