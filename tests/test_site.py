import json
import tempfile
import unittest
from pathlib import Path

from europa_reasoning.data import load_suite
from europa_reasoning.site import build_site_data


class SiteTests(unittest.TestCase):
    def test_site_builder_publishes_only_complete_full_suite_runs(self):
        metadata, items = load_suite()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            samples = [
                {
                    "item_id": item.id,
                    "response": "test",
                    "score": 1.0,
                    "passed": True,
                    "scorer": item.scoring["type"],
                    "parsed": None,
                    "score_details": {},
                    "scoring_error": None,
                    "latency_ms": 2.0,
                    "input_tokens": 10,
                    "output_tokens": 1,
                    "finish_reason": "stop",
                    "generation_metadata": {"reasoning_enabled": False},
                    "judgment": None,
                    "error": None,
                }
                for item in items
            ]
            base = {
                "run_id": "run-1",
                "status": "completed",
                "diagnostic": False,
                "model": {"id": "model", "revision": "abc", "backend": "ollama"},
                "judge": {"id": "judge", "revision": "judge-rev", "backend": "ollama"},
                "suite": {"id": metadata["id"], "version": metadata["version"]},
                "protocol": {"temperature": 0, "backend_settings": {"think": False}},
                "finished_at": "2026-09-03T00:00:00+00:00",
                "summary": {
                    "overall": {"score": 1.0},
                    "counts": {"scheduled": len(items), "scored": len(items)},
                },
                "samples": samples,
                "limitations": [],
            }
            (results / "publish.json").write_text(json.dumps(base))
            (results / "diagnostic.json").write_text(
                json.dumps({**base, "run_id": "run-2", "diagnostic": True})
            )
            (results / "partial.json").write_text(
                json.dumps({**base, "run_id": "run-3", "status": "partial"})
            )

            built = build_site_data(docs_dir=root / "docs", results_dir=results)
            catalog = json.loads((root / "docs/data/catalog.json").read_text())
            payload = json.loads((root / "docs/data/results.json").read_text())
            self.assertEqual(built["runs"], 1)
            self.assertEqual(catalog["suite"]["items"], 864)
            self.assertEqual(len(catalog["suite"]["languages"]), 36)
            self.assertEqual(payload["runs"][0]["items"][0]["item"]["language"], "bul")
            self.assertEqual(payload["runs"][0]["items"][0]["item"]["template_id"], "shopping-change")
            rubric_items = [entry for entry in payload["runs"][0]["items"] if entry["item"]["rubric"]]
            self.assertIn("critical_dimensions", rubric_items[0]["item"]["rubric"])
            self.assertTrue(all("reasoning" not in entry["sample"] for entry in payload["runs"][0]["items"]))
            self.assertTrue((root / "docs/data/site-data.js").read_text().startswith("window.EUROPA_REASONING_SITE_DATA = "))

    def test_site_rejects_self_judged_run(self):
        metadata, items = load_suite()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            run = {
                "run_id": "self-judge",
                "status": "completed",
                "diagnostic": False,
                "model": {"id": "same", "revision": "target-rev", "backend": "ollama"},
                "judge": {"id": "same", "revision": "judge-rev", "backend": "ollama"},
                "suite": {"id": metadata["id"], "version": metadata["version"]},
                "protocol": {},
                "finished_at": "2026-09-28T00:00:00+00:00",
                "summary": {"counts": {"scheduled": len(items), "scored": len(items)}},
                "samples": [],
            }
            (results / "self.json").write_text(json.dumps(run))

            built = build_site_data(docs_dir=root / "docs", results_dir=results)
            self.assertEqual(built["runs"], 0)


if __name__ == "__main__":
    unittest.main()
