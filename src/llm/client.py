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

# Qwen3-series models emit <think> blocks by default.
_THINK_BLOCK_RE = re.compile(
    r"<think[^>]*>[\s\S]*?</think>",
    re.IGNORECASE,
)
_THINK_UNCLOSED_RE = re.compile(
    r"<think[^>]*>[\s\S]*$",
    re.IGNORECASE,
)
# Capture the *content* inside think blocks so we can search for JSON there
_THINK_CONTENT_RE = re.compile(
    r"<think[^>]*>([\s\S]*?)</think>",
    re.IGNORECASE,
)
_THINK_UNCLOSED_CONTENT_RE = re.compile(
    r"<think[^>]*>([\s\S]*)$",
    re.IGNORECASE,
)


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
    ) -> tuple[str, str]:
        """Stream SSE chunks from the server, call token_callback per delta.

        Returns (full_text, finish_reason).
        """
        full_text = ""
        finish_reason = ""
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
                        choice = chunk["choices"][0]
                        delta = choice["delta"].get("content") or ""
                        if delta:
                            full_text += delta
                            token_callback(delta)
                        # Capture the finish_reason from the final chunk
                        fr = choice.get("finish_reason")
                        if fr:
                            finish_reason = fr
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        return full_text, finish_reason

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
            text, _ = await self._complete_streaming(payload, self._stream_callback)
            return text
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
        # For reasoning models (Qwen3, DeepSeek-R1) add an explicit
        # /no_think instruction to suppress <think> blocks that corrupt JSON.
        user_content = "/no_think\n" + prompt

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Respond with valid JSON only. "
                        "Do not output any thinking, reasoning, explanation, "
                        "markdown formatting, or code fences. "
                        "Output must start with { or [ and end with } or ]."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        if self.web_search:
            payload["web_search_options"] = {"enable": True}

        finish_reason = ""
        if self._stream_callback is not None:
            raw_text, finish_reason = await self._complete_streaming(
                payload, self._stream_callback
            )
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                raw_text = choice["message"]["content"]
                finish_reason = choice.get("finish_reason", "")

        if finish_reason == "length":
            logger.warning(
                "LLM response truncated (finish_reason=length, max_tokens=%d). "
                "JSON is likely incomplete. Raw tail: ...%s",
                max_tokens,
                raw_text[-200:] if raw_text else "(empty)",
            )

        logger.debug("Raw LLM response (%d chars, finish=%s): %s",
                     len(raw_text), finish_reason, raw_text[:300])

        # ── Strip reasoning/thinking blocks (Qwen3, DeepSeek-R1) ──
        # Some models put valid JSON *inside* the think block; grab it first.
        cleaned = _THINK_BLOCK_RE.sub("", raw_text)
        cleaned = _THINK_UNCLOSED_RE.sub("", cleaned)
        cleaned = cleaned.strip()

        truncated = finish_reason == "length"

        if cleaned:
            return self._parse_json_safe(cleaned, truncated=truncated)

        # Nothing outside think blocks — search *inside* them for JSON
        logger.warning(
            "No content outside <think> blocks — searching inside think content "
            "(finish_reason=%s)",
            finish_reason,
        )
        think_json = self._extract_json_from_think(raw_text)
        if think_json is not None:
            return think_json

        logger.error(
            "LLM returned empty content after stripping think blocks "
            "(finish_reason=%s). The model may have spent all tokens reasoning.",
            finish_reason,
        )
        return {}

    # ------------------------------------------------------------------
    # JSON extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_from_think(raw_text: str) -> dict | list | None:
        """Search for parseable JSON inside <think> block content.

        Some models (especially when response_format is not respected)
        place their entire JSON output inside the reasoning block.
        """
        # Gather all think-block contents
        parts: list[str] = []
        for m in _THINK_CONTENT_RE.finditer(raw_text):
            parts.append(m.group(1))
        m = _THINK_UNCLOSED_CONTENT_RE.search(raw_text)
        if m:
            parts.append(m.group(1))

        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Try raw_decode on the last { or [ occurrence (JSON is usually at the end)
            decoder = json.JSONDecoder()
            for start_char in ("{", "["):
                idx = part.rfind(start_char)
                if idx == -1:
                    continue
                try:
                    obj, _ = decoder.raw_decode(part, idx)
                    logger.warning("Recovered JSON from inside <think> block")
                    return obj
                except json.JSONDecodeError:
                    # Also try the first occurrence
                    idx2 = part.find(start_char)
                    if idx2 != idx:
                        try:
                            obj, _ = decoder.raw_decode(part, idx2)
                            logger.warning("Recovered JSON from inside <think> block (first occurrence)")
                            return obj
                        except json.JSONDecodeError:
                            pass
        return None

    @staticmethod
    def _parse_json_safe(raw_text: str, truncated: bool = False) -> dict | list:
        """Try progressively more lenient strategies to extract JSON.

        Uses JSONDecoder.raw_decode() so trailing garbage (extra text after
        the closing brace) is silently ignored rather than causing a failure.

        When *truncated* is True (finish_reason=length), attempts to repair
        the JSON by closing open braces/brackets.
        """
        text = raw_text.strip()
        if not text:
            return {}

        # 1. Direct parse — happy path
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Markdown code block: ```json ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 3. Use raw_decode() starting from the first { or [ —
        #    correctly ignores trailing text / extra objects.
        decoder = json.JSONDecoder()
        for start_char in ("{" , "["):
            idx = text.find(start_char)
            if idx == -1:
                continue
            try:
                obj, _ = decoder.raw_decode(text, idx)
                return obj
            except json.JSONDecodeError:
                pass

        # 4. Truncated JSON repair — close open brackets/braces.
        #    This rescues partial results when finish_reason=length.
        if truncated:
            repaired = LLMClient._repair_truncated_json(text)
            if repaired is not None:
                logger.warning("Recovered truncated JSON via bracket repair")
                return repaired

        logger.error("Failed to parse LLM JSON response (%d chars): %.500s",
                     len(raw_text), raw_text)
        return {}

    @staticmethod
    def _repair_truncated_json(text: str) -> dict | list | None:
        """Attempt to repair JSON truncated by max_tokens.

        Strategy: find the first { or [, strip any trailing incomplete
        string/value, then close all open brackets and braces.
        """
        # Find where JSON starts
        for ch in ("{", "["):
            idx = text.find(ch)
            if idx != -1:
                break
        else:
            return None

        fragment = text[idx:]

        # Strip any trailing incomplete string (ends mid-quote)
        # Look for the last complete key-value or array element
        # by removing trailing partial tokens.
        fragment = re.sub(r',\s*"[^"]*$', '', fragment)       # trailing key without value
        fragment = re.sub(r',\s*$', '', fragment)               # trailing comma
        fragment = re.sub(r':\s*"[^"]*$', ': ""', fragment)  # incomplete string value

        # Count open vs close braces/brackets
        opens = []
        in_string = False
        escape = False
        for c in fragment:
            if escape:
                escape = False
                continue
            if c == '\\':
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c in ('{', '['):
                opens.append('}' if c == '{' else ']')
            elif c in ('}', ']'):
                if opens and opens[-1] == c:
                    opens.pop()

        # Close all remaining open brackets/braces
        fragment += ''.join(reversed(opens))

        try:
            return json.loads(fragment)
        except json.JSONDecodeError:
            return None

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
