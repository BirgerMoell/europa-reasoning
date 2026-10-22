import json
import os
import unittest
from unittest.mock import patch

from europa_reasoning.backends import (
    GenerationConfig,
    HuggingFaceBackend,
    OllamaBackend,
    OpenAICompatibleBackend,
    _terminal_eos_id,
)
from europa_reasoning.data import load_suite


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(
            {
                "id": "chatcmpl-test",
                "created": 1,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "A"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 1},
            }
        ).encode()


class _FakeOllamaResponse(_FakeResponse):
    def read(self):
        return json.dumps(
            {
                "model": "gemma-test",
                "created_at": "2026-09-02T00:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": "A",
                    "thinking": "Private reasoning is intentionally not retained.",
                },
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 21,
                "eval_count": 2,
                "total_duration": 42,
                "load_duration": 3,
            }
        ).encode()


class BackendTests(unittest.TestCase):
    def _fake_hf_backend(self, think, captured):
        class FakeTokenizer:
            chat_template = "test-template"

            def apply_chat_template(self, messages, **kwargs):
                captured["messages"] = messages
                captured["kwargs"] = kwargs
                return "rendered"

        backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
        backend.tokenizer = FakeTokenizer()
        backend.device = "mps"
        backend.requested_dtype = "bfloat16"
        backend.think = think
        backend.reasoning_token_budget = 4096 if think else None
        backend.model = type(
            "FakeModel",
            (),
            {
                "parameters": lambda self: iter(
                    [type("FakeParameter", (), {"device": "mps:0", "dtype": "torch.bfloat16"})()]
                ),
                "config": type("FakeConfig", (), {"use_cache": True})(),
            },
        )()
        return backend

    def test_huggingface_backend_passes_the_thinking_switch_explicitly(self):
        messages = [{"role": "user", "content": "Hello"}]
        for think in (False, True):
            captured = {}
            backend = self._fake_hf_backend(think, captured)
            self.assertEqual(backend._render_prompt(messages, fallback="fallback"), "rendered")
            self.assertEqual(captured["messages"], messages)
            self.assertEqual(
                captured["kwargs"],
                {"tokenize": False, "add_generation_prompt": True, "enable_thinking": think},
            )
            self.assertEqual(
                backend.protocol_settings(),
                {
                    "think": think,
                    "reasoning_token_budget": 4096 if think else None,
                    "strip_terminal_eos": True,
                    "device": "mps",
                    "resolved_device": "mps:0",
                    "torch_dtype": "bfloat16",
                    "requested_dtype": "bfloat16",
                    "use_cache": True,
                },
            )

    def test_huggingface_decoding_excludes_non_special_terminal_eos(self):
        self.assertEqual(_terminal_eos_id([101, 102, 4], [262145, 4, 2]), 4)
        self.assertIsNone(_terminal_eos_id([101, 102], [262145, 4, 2]))
        self.assertEqual(_terminal_eos_id([101, 2], 2), 2)

    def test_openai_compatible_backend_uses_chat_contract(self):
        old_key = os.environ.get("EUROPA_TEST_KEY")
        os.environ["EUROPA_TEST_KEY"] = "secret-for-local-test"
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _FakeResponse()

        try:
            _, items = load_suite()
            backend = OpenAICompatibleBackend(
                model_id="test-model",
                base_url="http://127.0.0.1:8000/v1",
                api_key_env="EUROPA_TEST_KEY",
            )
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                generation = backend.generate(item=items[0], config=GenerationConfig())
        finally:
            if old_key is None:
                os.environ.pop("EUROPA_TEST_KEY", None)
            else:
                os.environ["EUROPA_TEST_KEY"] = old_key

        request = captured["request"]
        payload = json.loads(request.data)
        self.assertEqual(generation.text, "A")
        self.assertEqual(generation.input_tokens, 20)
        self.assertEqual(generation.output_tokens, 1)
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-for-local-test")
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["seed"], 17)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertIn("evaluation of reasoning in Bulgarian", payload["messages"][0]["content"])
        self.assertIn("<task>", payload["messages"][1]["content"])
        self.assertEqual(captured["timeout"], 120.0)

    def test_openai_backend_sends_thinking_switch_and_splits_inline_trace(self):
        captured = {}

        class _ThinkingResponse(_FakeResponse):
            def read(self):
                return json.dumps({"id": "x", "created": 1, "choices": [{
                    "message": {"role": "assistant", "content": "check 2+2\n</think>\n\\boxed{4}"},
                    "finish_reason": "stop"}], "usage": {}}).encode()

        def fake_urlopen(request, timeout):
            captured["request"] = request
            return _ThinkingResponse()

        _, items = load_suite()
        backend = OpenAICompatibleBackend(model_id="m", base_url="http://127.0.0.1:8000/v1",
                                          think=True, reasoning_token_budget=4096)
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            generation = backend.generate(item=items[0], config=GenerationConfig())
        payload = json.loads(captured["request"].data)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(payload["max_tokens"], items[0].max_tokens + 4096)
        self.assertEqual(generation.text, "\\boxed{4}")
        self.assertEqual(generation.reasoning, "check 2+2")
        self.assertTrue(generation.raw["trace_closed"])

    def test_ollama_backend_disables_thinking_and_uses_native_chat_contract(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _FakeOllamaResponse()

        _, items = load_suite()
        backend = OllamaBackend(model_id="gemma-test")
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            generation = backend.generate(item=items[0], config=GenerationConfig(seed=42))

        request = captured["request"]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertFalse(payload["think"])
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"]["seed"], 42)
        self.assertEqual(payload["options"]["num_predict"], items[0].max_tokens)
        self.assertEqual(generation.text, "A")
        self.assertEqual(generation.input_tokens, 21)
        self.assertEqual(generation.output_tokens, 2)
        self.assertEqual(backend.protocol_settings(), {"think": False})
        self.assertFalse(generation.raw["reasoning_enabled"])
        self.assertNotIn("thinking", generation.raw)

    def test_ollama_reasoning_is_a_separate_protocol_with_extra_budget(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            return _FakeOllamaResponse()

        _, items = load_suite()
        backend = OllamaBackend(
            model_id="reasoning-model",
            think=True,
            reasoning_token_budget=4096,
        )
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            generation = backend.generate(item=items[0], config=GenerationConfig(seed=42))

        payload = json.loads(captured["request"].data)
        self.assertTrue(payload["think"])
        self.assertEqual(payload["options"]["num_predict"], items[0].max_tokens + 4096)
        self.assertEqual(
            backend.protocol_settings(),
            {"think": True, "reasoning_token_budget": 4096},
        )
        self.assertGreater(generation.raw["reasoning_char_count"], 0)
        self.assertEqual(generation.raw["answer_token_budget"], items[0].max_tokens)
        self.assertEqual(
            generation.raw["total_generation_token_budget"],
            items[0].max_tokens + 4096,
        )
        self.assertNotIn("thinking", generation.raw)

    def test_ollama_reasoning_budget_requires_thinking(self):
        with self.assertRaisesRegex(ValueError, "requires thinking"):
            OllamaBackend(model_id="reasoning-model", reasoning_token_budget=128)
