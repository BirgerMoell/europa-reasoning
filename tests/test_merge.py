import unittest

from europa_reasoning.data import load_suite
from europa_reasoning.merge import merge_extension_runs


class MergeExtensionTests(unittest.TestCase):
    def test_merge_rejects_target_protocol_change(self):
        metadata, items = load_suite()
        base_items = items[:6]
        current_items = items[:12]
        base = {
            "status": "completed",
            "suite": {"id": metadata["id"], "version": metadata["version"]},
            "model": {"id": "model", "revision": "rev", "backend": "ollama"},
            "judge": {"id": "judge", "revision": "rev", "backend": "ollama"},
            "protocol": {
                "temperature": 0.0,
                "seed": 42,
                "system_prompt": "same",
                "backend_settings": {},
            },
            "samples": [{"item_id": item.id} for item in base_items],
        }
        extension = {
            **base,
            "protocol": {**base["protocol"], "temperature": 0.2},
            "samples": [{"item_id": item.id} for item in current_items[6:]],
        }
        with self.assertRaisesRegex(ValueError, "temperature"):
            merge_extension_runs(
                base_run=base,
                extension_run=extension,
                base_metadata=metadata,
                base_items=base_items,
                current_metadata=metadata,
                current_items=current_items,
            )


if __name__ == "__main__":
    unittest.main()
