"""Minimal OpenAI-compatible chat-completions client (stdlib only).

Works with any provider exposing ``POST {base_url}/chat/completions``
(OpenAI, DeepSeek, Moonshot, Qwen/DashScope compatible mode, vLLM, Ollama,
OpenRouter, ...). ``base_url`` may be given with or without a trailing ``/v1``.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

from deepseek_provider_reference import (
    DEEPSEEK_CAPABILITIES,
    GENERIC_CAPABILITIES,
    StreamAccumulator,
    build_chat_payload,
    parse_sse_data_lines,
)

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class EmptyContentError(LLMError):
    """The model produced reasoning/thinking but NO answer text.

    Observed with reasoning-style models (DeepSeek flash etc.): the whole
    ``max_tokens`` budget gets spent on ``reasoning_content``, the reply is
    truncated (``finish_reason == "length"``) and ``content`` comes back as
    an empty string.

    This is a MODEL OUTPUT problem, not a transport / API outage: the right
    reaction is to warn and ask the model again with a precise correction,
    NOT to count it as an API failure (which would eventually stop the run).
    """

    def __init__(
        self,
        message: str,
        finish_reason: str = "",
        reasoning: str = "",
    ):
        super().__init__(message)
        self.finish_reason = finish_reason
        self.reasoning = reasoning


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.4,
        max_tokens: int = 512,
        timeout: float = 90.0,
        max_retries: int = 2,
        raw_dump_path: str | None = None,
        thinking_enabled: bool | None = None,
        reasoning_effort: str | None = None,
        stream_mode: str = "off",
        provider_profile: str = "auto",
    ):
        self.base_url = (base_url or "").rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            # User provided the full endpoint -- use it verbatim.
            self.endpoint = self.base_url
        else:
            if self.base_url and not self.base_url.endswith("/v1"):
                # Common convention: providers serve under /v1.
                self.base_url += "/v1"
            self.endpoint = self.base_url + "/chat/completions"
        self.api_key = api_key or ""
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.last_reasoning: str = ""  # provider reasoning_content, if any
        self.last_finish_reason: str = ""  # provider finish_reason, if any
        # Token accounting for the UI (cumulative over this client's life).
        self.last_usage: dict[str, Any] = {}
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        # Optional forensic dump: the RAW response body as returned by the
        # relay/proxy, for pinpointing where content corruption happens
        # (relay bug vs. model output). One JSON line per call.
        self.raw_dump_path = raw_dump_path
        # ---- Provider capability support (beta) --------------------
        # provider_profile:
        #   "auto"    (default) native DeepSeek fields ONLY when the
        #             HOSTNAME is api.deepseek.com -- never inferred from
        #             the model name (a relay serving "deepseek-v4-pro"
        #             is still a generic OpenAI-compatible endpoint).
        #   "generic" always plain OpenAI-compatible payload.
        #   "deepseek" user explicitly forces the native DeepSeek profile.
        profile = (provider_profile or "auto").lower()
        if profile == "generic":
            self.caps = GENERIC_CAPABILITIES
        elif profile == "deepseek":
            self.caps = DEEPSEEK_CAPABILITIES
        else:  # auto: hostname check ONLY, never the model name.
            self.caps = (
                DEEPSEEK_CAPABILITIES
                if "api.deepseek.com" in (base_url or "").lower()
                else GENERIC_CAPABILITIES
            )
        self.thinking_enabled = thinking_enabled
        self.reasoning_effort = reasoning_effort
        self.stream_mode = (stream_mode or "off").lower()
        # Optional live-delta callbacks (UI streaming display).
        self.on_reasoning_delta = None  # callable(str)
        self.on_content_delta = None    # callable(str)
        # First-token latency (FIX: measured once per call, at the FIRST
        # callback delta, relative to the real call-start monotonic ts).
        self._call_start: float = 0.0
        self.first_reasoning_ms: int | None = None
        self.first_content_ms: int | None = None

    def _should_stream(self) -> bool:
        if self.stream_mode == "on":
            return True
        if self.stream_mode == "auto":
            # Only endpoints KNOWN to stream correctly (official DeepSeek).
            return (
                self.caps.provider == "deepseek"
                and "api.deepseek.com" in self.endpoint
            )
        return False

    def _dump_raw(self, body: bytes, content: str) -> None:
        if not self.raw_dump_path:
            return
        try:
            from pathlib import Path

            record = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "endpoint": self.endpoint,
                "body_utf8_lossy": body.decode("utf-8", errors="replace")[:20000],
                "extracted_content": content,
                "extracted_content_len": len(content),
            }
            with open(self.raw_dump_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _capture_meta(self, data: dict) -> None:
        """Record provider-side reasoning / finish_reason for a response."""
        try:
            rc = data["choices"][0].get("message", {}).get("reasoning_content")
            self.last_reasoning = rc if isinstance(rc, str) else ""
        except (KeyError, IndexError, AttributeError, TypeError):
            self.last_reasoning = ""
        try:
            self.last_finish_reason = str(
                data["choices"][0].get("finish_reason") or "")
        except (KeyError, IndexError, AttributeError, TypeError):
            self.last_finish_reason = ""

    def _record_usage(self, data: dict) -> None:
        """Accumulate token usage so the UI can show throughput / cost."""
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return
        self.last_usage = usage
        try:
            self.total_prompt_tokens += int(usage.get("prompt_tokens") or 0)
        except (TypeError, ValueError):
            pass
        try:
            self.total_completion_tokens += int(usage.get("completion_tokens") or 0)
        except (TypeError, ValueError):
            pass

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send a chat completion request; returns the assistant text."""
        # First-token latency baseline: one monotonic ts per call; the
        # first streaming delta writes it ONCE (never rewritten per token).
        self._call_start = time.monotonic()
        self.first_reasoning_ms = None
        self.first_content_ms = None
        use_stream = self._should_stream()
        payload = build_chat_payload(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=use_stream,
            capabilities=self.caps,
            thinking_enabled=self.thinking_enabled,
            reasoning_effort=self.reasoning_effort,
        )
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                req = urllib.request.Request(
                    self.endpoint, data=body, headers=headers, method="POST"
                )
                if use_stream:
                    text = self._chat_streaming(payload)
                    self._dump_raw(
                        json.dumps({
                            "streamed": True,
                            "finish_reason": self.last_finish_reason,
                        }).encode("utf-8"),
                        text,
                    )
                    return text
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw_body = resp.read()
                data = json.loads(raw_body.decode("utf-8"))
                # Record reasoning/finish_reason FIRST so that even a
                # recoverable "thinking only" reply is fully described.
                self._capture_meta(data)
                self._record_usage(data)
                # Raises EmptyContentError when the model returned thinking
                # but no answer text; the caller decides how to recover.
                text = self._extract_text(data)
                self._dump_raw(raw_body, text)
                return text
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                last_err = LLMError(f"HTTP {e.code} from LLM API: {detail}")
                if e.code not in (429, 500, 502, 503, 504):
                    raise last_err from e
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
                last_err = LLMError(f"LLM API request failed: {e}")
            if attempt <= self.max_retries:
                time.sleep(min(2 ** attempt, 8))
        raise last_err or LLMError("LLM API request failed")

    def _chat_streaming(self, payload: dict) -> str:
        """Streaming chat completion (SSE). Reasoning/content deltas are
        accumulated and only the COMPLETE content is returned -- partial
        JSON is never parsed or acted upon."""
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.endpoint, data=body, headers=headers, method="POST"
        )
        acc = StreamAccumulator()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for event in parse_sse_data_lines(resp):
                r_delta, c_delta = acc.feed_event(event)
                if r_delta:
                    # Write the first-token timestamp ONCE, based on the
                    # real call-start monotonic timestamp.
                    if self.first_reasoning_ms is None:
                        self.first_reasoning_ms = int(
                            (time.monotonic() - self._call_start) * 1000
                        )
                    if self.on_reasoning_delta is not None:
                        try:
                            self.on_reasoning_delta(r_delta)
                        except Exception:
                            pass
                if c_delta:
                    if self.first_content_ms is None:
                        self.first_content_ms = int(
                            (time.monotonic() - self._call_start) * 1000
                        )
                    if self.on_content_delta is not None:
                        try:
                            self.on_content_delta(c_delta)
                        except Exception:
                            pass
        self.last_reasoning = acc.reasoning
        self.last_finish_reason = acc.finish_reason
        if isinstance(acc.usage, dict):
            self.last_usage = acc.usage
            try:
                self.total_prompt_tokens += int(acc.usage.get("prompt_tokens") or 0)
            except (TypeError, ValueError):
                pass
            try:
                self.total_completion_tokens += int(
                    acc.usage.get("completion_tokens") or 0)
            except (TypeError, ValueError):
                pass
        if not acc.content.strip():
            if acc.reasoning.strip():
                raise EmptyContentError(
                    "流式响应只返回了思考过程(reasoning_content)，未输出回答文本"
                    f"（finish_reason={acc.finish_reason or '?'}，"
                    f"reasoning 共 {len(acc.reasoning)} 字符）",
                    finish_reason=acc.finish_reason,
                    reasoning=acc.reasoning,
                )
            raise LLMError("LLM API streaming response contained no text")
        return acc.content

    @staticmethod
    def _extract_text(data: dict) -> str:
        try:
            choices = data["choices"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected LLM API response shape: {data}") from e
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            text = "".join(parts)
            if text:
                return text
        # Fallback for non-chat-shaped responses
        text = choices[0].get("text", "")
        if text:
            return text
        # A reply carrying reasoning but no content is a distinct, recoverable
        # failure (thinking consumed the whole token budget): reclassify it so
        # the caller can warn + retry instead of treating it as an API outage.
        LLMClient._raise_if_reasoning_only(data)
        raise LLMError(f"LLM API response contained no text: {data}")

    @staticmethod
    def _raise_if_reasoning_only(data: dict) -> None:
        """Raise EmptyContentError when a reply has reasoning but no content.

        No-op for every other shape (i.e. when no reasoning is present), so it
        only reclassifies the specific "thinking only, no answer" case.
        """
        try:
            message = data["choices"][0].get("message", {}) or {}
            finish = str(data["choices"][0].get("finish_reason") or "")
        except (KeyError, IndexError, AttributeError, TypeError):
            return
        reasoning = message.get("reasoning_content")
        reasoning = reasoning if isinstance(reasoning, str) else ""
        if not reasoning.strip():
            return
        if finish == "length":
            hint = (
                "思考过程已耗尽 max_tokens 预算并被截断（finish_reason=length），"
                "请大幅精简思考后立即输出 JSON"
            )
        else:
            hint = "请跳过冗长推理，直接输出一个 JSON 对象"
        raise EmptyContentError(
            "模型只返回了思考过程(reasoning_content)，未输出任何回答文本"
            f"（content 为空，finish_reason={finish or '?'}，"
            f"reasoning 共 {len(reasoning)} 字符）——{hint}",
            finish_reason=finish,
            reasoning=reasoning,
        )
