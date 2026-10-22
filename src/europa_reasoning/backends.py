"""Generation backends for hosted APIs, Ollama and local Transformers models."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .models import Generation, Item
from .process import THINK_CLOSE, THINK_OPEN, split_reasoning


DEFAULT_SYSTEM_PROMPT = (
    "You are taking part in an evaluation of reasoning in {language}. "
    "Answer in {language} unless the task explicitly requests another language. "
    "Follow the response instructions exactly."
)


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.0
    seed: int | None = 17
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def system_prompt_for(self, item: Item) -> str:
        replacements = {
            "{language}": str(item.metadata.get("language_name", item.language)),
            "{language_code}": item.language,
            "{script}": item.script,
        }
        rendered = self.system_prompt
        for placeholder, value in replacements.items():
            rendered = rendered.replace(placeholder, value)
        return rendered


class Backend(ABC):
    name: str
    model_id: str

    @abstractmethod
    def generate(self, item: Item, config: GenerationConfig) -> Generation:
        """Generate one answer for an evaluation item."""

    def protocol_settings(self) -> dict[str, Any]:
        """Return non-secret backend settings that affect model output."""
        return {}


class OpenAICompatibleBackend(Backend):
    """Small dependency-free client for OpenAI-compatible chat endpoints."""

    name = "openai-compatible"

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_seconds: float = 120.0,
        think: bool | None = None,
        reasoning_token_budget: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.think = think
        self.reasoning_token_budget = reasoning_token_budget

    def protocol_settings(self) -> dict[str, Any]:
        settings: dict[str, Any] = {"think": self.think}
        if self.reasoning_token_budget is not None:
            settings["reasoning_token_budget"] = self.reasoning_token_budget
        return settings

    def generate(self, item: Item, config: GenerationConfig) -> Generation:
        url = self.base_url
        if not url.endswith("/chat/completions"):
            if not url.endswith("/v1"):
                url += "/v1"
            url += "/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": config.system_prompt_for(item)},
                *item.chat_messages(),
            ],
            "temperature": config.temperature,
            "max_tokens": item.max_tokens + (self.reasoning_token_budget or 0),
        }
        if self.think is not None:
            # vLLM / SGLang convention for Qwen3-style templates with a thinking switch.
            payload["chat_template_kwargs"] = {"enable_thinking": self.think}
        if config.seed is not None:
            payload["seed"] = config.seed
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env, "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"endpoint returned HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"could not reach endpoint: {exc.reason}") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        choice = raw["choices"][0]
        usage = raw.get("usage", {})
        message = choice["message"]
        content = str(message.get("content") or "")
        separate = message.get("reasoning_content") or message.get("reasoning")
        if separate:
            trace, answer, closed = str(separate).strip(), content.strip(), bool(content.strip())
        else:
            trace, answer, closed = split_reasoning(content, prompt_opened=bool(self.think))
        finish = choice.get("finish_reason")
        if self.think and not closed:
            finish = "reasoning_unclosed"
        return Generation(
            text=answer,
            latency_ms=latency_ms,
            input_tokens=_optional_int(usage.get("prompt_tokens")),
            output_tokens=_optional_int(usage.get("completion_tokens")),
            finish_reason=finish,
            raw={"id": raw.get("id"), "created": raw.get("created"), "trace_closed": closed},
            reasoning=trace,
        )


class OllamaBackend(Backend):
    """Dependency-free client for Ollama's native chat endpoint.

    The native endpoint exposes ``think`` explicitly. That matters for models
    whose hidden reasoning otherwise consumes a benchmark item's answer-token
    budget before a final answer is emitted.
    """

    name = "ollama"

    def __init__(
        self,
        model_id: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 120.0,
        think: bool = False,
        reasoning_token_budget: int | None = None,
    ) -> None:
        if reasoning_token_budget is not None and reasoning_token_budget < 0:
            raise ValueError("Ollama reasoning token budget must be non-negative")
        if reasoning_token_budget is not None and not think:
            raise ValueError("Ollama reasoning token budget requires thinking to be enabled")
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.think = think
        self.reasoning_token_budget = reasoning_token_budget

    def protocol_settings(self) -> dict[str, Any]:
        settings: dict[str, Any] = {"think": self.think}
        if self.reasoning_token_budget is not None:
            settings["reasoning_token_budget"] = self.reasoning_token_budget
        return settings

    def generate(self, item: Item, config: GenerationConfig) -> Generation:
        url = self.base_url
        if url.endswith("/v1"):
            url = url[:-3]
        if not url.endswith("/api/chat"):
            url += "/api/chat"
        total_generation_budget = item.max_tokens + (self.reasoning_token_budget or 0)
        options: dict[str, Any] = {
            "temperature": config.temperature,
            "num_predict": total_generation_budget,
        }
        if config.seed is not None:
            options["seed"] = config.seed
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": config.system_prompt_for(item)},
                *item.chat_messages(),
            ],
            "options": options,
            "stream": False,
            "think": self.think,
        }
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"could not reach Ollama: {exc.reason}") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        message = raw.get("message", {})
        thinking = message.get("thinking")
        answer = str(message.get("content", ""))
        closed = bool(self.think and answer.strip())
        finish = raw.get("done_reason")
        if self.think and not closed:
            finish = "reasoning_unclosed"
        return Generation(
            text=answer,
            reasoning=str(thinking).strip() if isinstance(thinking, str) and thinking.strip() else None,
            latency_ms=latency_ms,
            input_tokens=_optional_int(raw.get("prompt_eval_count")),
            output_tokens=_optional_int(raw.get("eval_count")),
            finish_reason=finish,
            raw={
                "trace_closed": closed,
                "created_at": raw.get("created_at"),
                "total_duration_ns": raw.get("total_duration"),
                "load_duration_ns": raw.get("load_duration"),
                "reasoning_enabled": self.think,
                "reasoning_char_count": len(thinking) if isinstance(thinking, str) else 0,
                "answer_char_count": len(str(message.get("content", ""))),
                "answer_token_budget": item.max_tokens,
                "total_generation_token_budget": total_generation_budget,
            },
        )


class HuggingFaceBackend(Backend):
    """Local text generation through Transformers, imported only when requested."""

    name = "huggingface"

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        use_cache: bool | None = None,
        think: bool = False,
        reasoning_token_budget: int | None = None,
    ) -> None:
        if reasoning_token_budget is not None and not think:
            raise ValueError("a reasoning token budget requires thinking to be enabled")
        self.think = think
        self.reasoning_token_budget = reasoning_token_budget
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers import __version__ as transformers_version
        except ImportError as exc:
            raise RuntimeError("install local dependencies with: pip install -e '.[local]'") from exc

        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.requested_dtype = dtype
        self.use_cache = use_cache
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
        dtype_key = "dtype" if int(transformers_version.split(".", 1)[0]) >= 5 else "torch_dtype"
        model_kwargs: dict[str, Any] = {"revision": revision, dtype_key: torch_dtype}
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if device != "auto":
            self.model.to(device)
        if use_cache is not None:
            self.model.config.use_cache = use_cache
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def protocol_settings(self) -> dict[str, Any]:
        parameter = next(self.model.parameters())
        return {
            "think": self.think,
            "reasoning_token_budget": self.reasoning_token_budget,
            "strip_terminal_eos": True,
            "device": self.device,
            "resolved_device": str(parameter.device),
            "torch_dtype": str(parameter.dtype).removeprefix("torch."),
            "requested_dtype": self.requested_dtype,
            "use_cache": bool(self.model.config.use_cache),
        }

    def _render_prompt(self, messages: list[dict[str, str]], fallback: str) -> str:
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.think,
            )
        return fallback

    def generate(self, item: Item, config: GenerationConfig) -> Generation:
        messages = [
            {"role": "system", "content": config.system_prompt_for(item)},
            *item.chat_messages(),
        ]
        rendered = self._render_prompt(
            messages,
            fallback=f"{config.system_prompt_for(item)}\n\n{item.rendered_conversation()}\n\nAnswer:",
        )
        inputs = self.tokenizer(rendered, return_tensors="pt")
        target_device = next(self.model.parameters()).device
        inputs = {key: value.to(target_device) for key, value in inputs.items()}
        budget = item.max_tokens + (self.reasoning_token_budget or 0)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": budget,
            "do_sample": config.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.use_cache is not None:
            generation_kwargs["use_cache"] = self.use_cache
        if config.temperature > 0:
            generation_kwargs["temperature"] = config.temperature
        if config.seed is not None:
            self._torch.manual_seed(config.seed)
        started = time.perf_counter()
        with self._torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)
        latency_ms = (time.perf_counter() - started) * 1000
        input_length = int(inputs["input_ids"].shape[-1])
        new_tokens = generated[0][input_length:]
        terminal_eos_id = _terminal_eos_id(
            new_tokens, self.model.generation_config.eos_token_id
        )
        answer_tokens = new_tokens[:-1] if terminal_eos_id is not None else new_tokens
        # Keep think delimiters (some tokenizers mark them special) and drop every other special token.
        text = self.tokenizer.decode(answer_tokens, skip_special_tokens=False)
        for special in self.tokenizer.all_special_tokens:
            if special not in (THINK_OPEN, THINK_CLOSE):
                text = text.replace(special, "")
        prompt_opened = rendered.rstrip().endswith(THINK_OPEN)
        trace, answer, closed = split_reasoning(text, prompt_opened=prompt_opened)
        reasoning_tokens = None
        close_id = self.tokenizer.convert_tokens_to_ids(THINK_CLOSE)
        token_list = [int(t) for t in answer_tokens]
        if trace is not None:
            reasoning_tokens = token_list.index(close_id) if close_id in token_list else len(token_list)
        finish = "length" if int(new_tokens.shape[-1]) >= budget else "stop"
        if self.think and not closed:
            finish = "reasoning_unclosed" if finish == "length" else "stop_without_close"
        raw: dict[str, Any] = {"trace_closed": closed, "prompt_opened_think": prompt_opened}
        if terminal_eos_id is not None:
            raw["terminal_eos_token_id"] = terminal_eos_id
        return Generation(
            text=answer if (self.think or trace is not None) else text.strip(),
            latency_ms=latency_ms,
            input_tokens=input_length,
            output_tokens=int(new_tokens.shape[-1]),
            finish_reason=finish,
            raw=raw,
            reasoning=trace,
            reasoning_tokens=reasoning_tokens,
        )


def _terminal_eos_id(tokens: Any, eos_token_id: int | list[int] | None) -> int | None:
    """Exclude a generation terminator even when the tokenizer calls it non-special."""
    if len(tokens) == 0 or eos_token_id is None:
        return None
    eos_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id)
    final_id = int(tokens[-1])
    return final_id if final_id in eos_ids else None


class OracleBackend(Backend):
    """Diagnostic backend that emits references. Never valid leaderboard evidence."""

    name = "oracle-diagnostic"
    model_id = "europa-reasoning/oracle-diagnostic"

    def generate(self, item: Item, config: GenerationConfig) -> Generation:
        del config
        if item.id.startswith("judge::"):
            scores = {dimension: 4 for dimension in item.scoring.get("dimensions", {})}
            return Generation(
                text=json.dumps(
                    {"scores": scores, "reason": "Diagnostic oracle response."},
                    ensure_ascii=False,
                ),
                latency_ms=0.0,
                finish_reason="oracle",
            )
        scorer = item.scoring["type"]
        boxed = item.scoring.get("extract") == "boxed"
        if scorer in {"choice", "exact"}:
            text = str(item.gold["answer"])
        elif scorer == "contains_all":
            text = str(item.gold.get("answer") or "; ".join(_first_aliases(item.gold["required"])))
        elif scorer == "numeric":
            text = str(item.gold["value"])
        elif scorer == "json_exact":
            text = json.dumps(item.gold["value"], ensure_ascii=False)
        else:
            text = str(item.gold.get("example") or item.gold.get("reference_answer") or "")
        if boxed:
            text = f"\\boxed{{{text}}}"
        return Generation(text=text, latency_ms=0.0, finish_reason="oracle")


def create_backend(
    kind: str,
    model_id: str,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    revision: str | None = None,
    device: str = "auto",
    timeout_seconds: float = 120.0,
    ollama_think: bool = False,
    ollama_reasoning_tokens: int | None = None,
    dtype: str = "auto",
    use_cache: bool | None = None,
    think: bool | None = None,
    reasoning_tokens: int | None = None,
) -> Backend:
    if kind == "openai-compatible":
        if not base_url:
            raise ValueError("--base-url is required for the openai-compatible backend")
        return OpenAICompatibleBackend(
            model_id=model_id,
            base_url=base_url,
            api_key_env=api_key_env,
            timeout_seconds=timeout_seconds,
            think=think,
            reasoning_token_budget=reasoning_tokens,
        )
    if kind == "huggingface":
        return HuggingFaceBackend(
            model_id=model_id, revision=revision, device=device,
            dtype=dtype, use_cache=use_cache,
            think=bool(think), reasoning_token_budget=reasoning_tokens,
        )
    if kind == "ollama":
        return OllamaBackend(
            model_id=model_id,
            base_url=base_url or "http://127.0.0.1:11434",
            timeout_seconds=timeout_seconds,
            think=ollama_think,
            reasoning_token_budget=ollama_reasoning_tokens,
        )
    if kind == "oracle":
        return OracleBackend()
    raise ValueError(f"unknown backend: {kind}")


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _first_aliases(groups: list[Any]) -> list[str]:
    return [str(group[0] if isinstance(group, list) else group) for group in groups]
