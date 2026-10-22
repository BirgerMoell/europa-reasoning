import tempfile
import unittest
from pathlib import Path

from europa_reasoning.backends import GenerationConfig, OracleBackend
from europa_reasoning.data import load_suite
from europa_reasoning.runner import run_evaluation


class RunnerTests(unittest.TestCase):
    def test_oracle_run_is_resumable_and_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata, items = load_suite()
            output = Path(directory) / "oracle.json"
            first = run_evaluation(
                suite_metadata=metadata,
                items=items,
                backend=OracleBackend(),
                judge_backend=OracleBackend(),
                judge_revision="oracle-rev",
                output=output,
                config=GenerationConfig(),
                limit=4,
            )
            second = run_evaluation(
                suite_metadata=metadata,
                items=items,
                backend=OracleBackend(),
                judge_backend=OracleBackend(),
                judge_revision="oracle-rev",
                output=output,
                config=GenerationConfig(),
                limit=4,
            )
            self.assertEqual(first["status"], "completed")
            self.assertTrue(first["diagnostic"])
            self.assertEqual(first["summary"]["overall"]["score"], 1.0)
            self.assertEqual(second["summary"]["counts"]["scheduled"], 4)
            self.assertEqual(len(output.with_suffix(".samples.jsonl").read_text().splitlines()), 4)

    def test_language_filter_is_recorded_and_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata, items = load_suite()
            run = run_evaluation(
                suite_metadata=metadata,
                items=items,
                backend=OracleBackend(),
                judge_backend=OracleBackend(),
                output=Path(directory) / "swedish.json",
                config=GenerationConfig(),
                languages={"swe"},
            )
            self.assertEqual(run["status"], "completed")
            self.assertTrue(run["diagnostic"])
            self.assertEqual(run["summary"]["counts"]["scheduled"], 24)
            # the oracle emits boxed gold answers: every deterministic item must score 1
            self.assertTrue(all(sample["score"] == 1.0 for sample in run["samples"]))
            self.assertEqual(run["protocol"]["language_filter"], ["swe"])
            self.assertEqual({sample["item_id"].split("-")[3] for sample in run["samples"]}, {"swe"})

    def test_oracle_judge_completes_rubric_item(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata, items = load_suite()
            rubric_item = next(item for item in items if item.scoring["type"] == "rubric")
            run = run_evaluation(
                suite_metadata=metadata,
                items=[rubric_item],
                backend=OracleBackend(),
                judge_backend=OracleBackend(),
                judge_revision="oracle-rev",
                output=Path(directory) / "judged.json",
                config=GenerationConfig(),
            )
            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["samples"][0]["score"], 1.0)
            self.assertEqual(run["judge"]["revision"], "oracle-rev")

    def test_resume_rejects_changed_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata, items = load_suite()
            output = Path(directory) / "changed.json"
            run_evaluation(
                suite_metadata=metadata,
                items=items,
                backend=OracleBackend(),
                output=output,
                config=GenerationConfig(system_prompt="First {language}"),
                limit=1,
            )
            with self.assertRaisesRegex(ValueError, "resume manifest differs"):
                run_evaluation(
                    suite_metadata=metadata,
                    items=items,
                    backend=OracleBackend(),
                    output=output,
                    config=GenerationConfig(system_prompt="Changed {language}"),
                    limit=1,
                )


if __name__ == "__main__":
    unittest.main()
