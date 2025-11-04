"""Lightweight LLM client wrappers used by QA evaluation."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from anthropic import Anthropic
from openai import OpenAI


@dataclass
class ModelSpec:
    """Configuration describing how to access an LLM endpoint."""

    api: str
    api_key: str
    model: str
    base_url: Optional[str] = None


class BaseClient:
    def __init__(self, spec: ModelSpec):
        if not spec.api_key:
            raise ValueError("An API key must be provided.")
        self.spec = spec
        if os.path.exists(spec.api_key):
            with open(spec.api_key, "r", encoding="utf-8") as handle:
                self.api_key = handle.read().strip()
        else:
            self.api_key = spec.api_key.strip()

    def complete(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        response = None
        attempts = 0
        while response is None:
            try:
                response = self._send(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
            except Exception as exc:  # pragma: no cover - network robustness
                attempts += 1
                print(exc)
                print(f"Attempt {attempts} failed, retrying in 5s...")
                time.sleep(5)
        return response

    def _send(self, prompt: str, max_tokens: int, temperature: float) -> str:
        raise NotImplementedError


class OpenAIClient(BaseClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        if spec.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=spec.base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)

    def _send(self, prompt: str, max_tokens: int, temperature: float) -> str:
        response = self.client.chat.completions.create(
            model=self.spec.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class AnthropicClient(BaseClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self.client = Anthropic(api_key=self.api_key)

    def _send(self, prompt: str, max_tokens: int, temperature: float) -> str:
        response = self.client.messages.create(
            model=self.spec.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class TogetherClient(OpenAIClient):
    def __init__(self, spec: ModelSpec):
        spec = ModelSpec(api="together", api_key=spec.api_key, model=spec.model, base_url="https://api.together.xyz/v1")
        super().__init__(spec)


def build_client(spec: ModelSpec) -> BaseClient:
    api = spec.api.lower()
    if api == "openai":
        return OpenAIClient(spec)
    if api == "anthropic":
        return AnthropicClient(spec)
    if api == "together":
        return TogetherClient(spec)
    raise ValueError(f"Unsupported API: {spec.api}")
