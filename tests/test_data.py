import unittest

from europa_reasoning.data import load_suite, validate_suite


class DataTests(unittest.TestCase):
    def test_bundled_suite_is_parallel_and_valid(self):
        metadata, items = load_suite()

        self.assertEqual(validate_suite(metadata=metadata, items=items), [])
        self.assertEqual(metadata["version"], "0.1.0")
        self.assertEqual(len(items), 864)
        self.assertEqual(len(metadata["languages"]), 36)
        self.assertEqual(len(metadata["template_ids"]), 24)
        self.assertEqual(len({item.domain for item in items}), 4)
        self.assertEqual({item.task_type for item in items}, {"numeric", "multiple_choice", "open_explanation"})
        for language in metadata["languages"]:
            language_items = [item for item in items if item.language == language]
            self.assertEqual(len(language_items), 24)
            self.assertEqual({item.template_id for item in language_items}, set(metadata["template_ids"]))

    def test_every_item_has_multilingual_provenance(self):
        _, items = load_suite()

        self.assertTrue(all(item.source.url.startswith("https://") for item in items))
        self.assertTrue(all(item.source.license for item in items))
        self.assertTrue(all(item.language and item.script and item.template_id for item in items))
        counts = {
            state: len([item for item in items if item.review_status == state])
            for state in {item.review_status for item in items}
        }
        self.assertEqual(counts, {"source_native": 24, "machine_translated": 840})

    def test_gold_and_parameters_are_identical_across_languages(self):
        _, items = load_suite()
        by_template: dict[str, set] = {}
        for item in items:
            key = repr((sorted(item.gold.items()) if item.scoring["type"] != "rubric" else None,
                        sorted(item.metadata.get("parameters", {}).items())))
            by_template.setdefault(item.template_id, set()).add(key)
        self.assertTrue(all(len(keys) == 1 for keys in by_template.values()), by_template)

    def test_deterministic_items_require_a_boxed_answer(self):
        _, items = load_suite()
        for item in items:
            if item.scoring["type"] in {"numeric", "choice"}:
                self.assertEqual(item.scoring.get("extract"), "boxed", item.id)
                self.assertIn("\\boxed{", item.prompt, item.id)


if __name__ == "__main__":
    unittest.main()
