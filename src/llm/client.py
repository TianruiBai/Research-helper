"""llama.cpp server OpenAI-compatible HTTP wrapper — LLM client for local inference.

Works with any OpenAI-compatible backend:
  - llama-server (llama.cpp)  — default port 8080
  - Ollama /v1 endpoints      — port 11434
  - vLLM, LM Studio, etc.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

import httpx

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around the OpenAI-compatible chat completions API."""

    def __init__(
        self,
        model: str = "qwen35-reasoning",
        base_url: str = "http://localhost:8080",
        timeout: int = 120,
        web_search: bool = False,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.web_search = web_search
        # Set before LLM calls to stream tokens to a callback; clear after.
        self._stream_callback: Callable[[str], None] | None = None

    async def _complete_streaming(
        self,
        payload: dict[str, Any],
        token_callback: Callable[[str], None],
    ) -> str:
        """Stream SSE chunks from the server, call token_callback per delta, return full text."""
        full_text = ""
        stream_payload = {**payload, "stream": True}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=stream_payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content") or ""
                        if delta:
                            full_text += delta
                            token_callback(delta)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        return full_text

    async def health_check(self) -> bool:
        """Check if the LLM server is running and responsive."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                # llama-server exposes GET /health; Ollama /v1 returns 200 on GET /
                for path in ("/health", "/v1/models"):
                    try:
                        resp = await client.get(f"{self.base_url}{path}")
                        if resp.status_code == 200:
                            return True
                    except Exception:
                        continue
            return False
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """Return model IDs reported by the server."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self.base_url}/v1/models")
                resp.raise_for_status()
                data = resp.json()
                return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Send a prompt and return the text response."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a professional research analyst."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.web_search:
            payload["web_search_options"] = {"enable": True}
        if self._stream_callback is not None:
            return await self._complete_streaming(payload, self._stream_callback)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def complete_json(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict | list:
        """Send a prompt with JSON format enforcement and return parsed JSON."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Respond with valid JSON only. No markdown, no explanations, no thinking blocks."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        if self.web_search:
            payload["web_search_options"] = {"enable": True}
        if self._stream_callback is not None:
            raw_text = await self._complete_streaming(payload, self._stream_callback)
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                )
                resp.raise_for_status()
                raw_text = resp.json()["choices"][0]["message"]["content"]

        # Strip reasoning/thinking blocks (Qwen3.5, DeepSeek-R1, etc.)
        raw_text = re.sub(r"<think>[\s\S]*?</think>", "", raw_text).strip()

        # Parse JSON — handle common model quirks
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks

            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_text)
            if match:
                return json.loads(match.group(1).strip())
            # Try to find first { or [ and parse from there
            for start_char, end_char in [("{", "}"), ("[", "]")]:
                start_idx = raw_text.find(start_char)
                end_idx = raw_text.rfind(end_char)
                if start_idx != -1 and end_idx != -1:
                    return json.loads(raw_text[start_idx : end_idx + 1])
            logger.error("Failed to parse LLM JSON response: %s", raw_text[:200])
            return {}

    async def is_model_available(self) -> bool:
        """Check if the configured model is available on the server.

        llama-server loads a single model at startup and may not list it by
        alias in /v1/models, so we fall back to True when the server is healthy
        but the model list is empty.
        """
        if not await self.health_check():
            return False
        models = await self.list_models()
        if not models:
            # llama-server with no /v1/models listing: model is implicitly loaded
            return True
        return any(self.model in m for m in models)
