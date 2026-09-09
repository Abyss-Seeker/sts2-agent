"""Reference helpers for DeepSeek-specific Chat Completions capabilities.

This is intentionally a helper, not a forced drop-in replacement for the
current llm_client.py.  Worker should merge these pieces behind capability
flags so generic OpenAI-compatible relays keep working.

DeepSeek official Chat Completions thinking controls (verified 2026-09):
- thinking: {"type": "enabled"|"disabled"}
- reasoning_effort: "low"|"high"|"max"
- reasoning_content is returned alongside content
- thinking mode ignores temperature/top_p/presence/frequency penalties
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable


@dataclass(frozen=True)
class ProviderCapabilities:
    provider: str = "generic"
    supports_reasoning_content: bool = False
    supports_reasoning_effort: bool = False
    supports_thinking_toggle: bool = False
    supports_streaming: bool = True
    supports_cache_metrics: bool = False
    supports_tool_calls: bool = True


DEEPSEEK_CAPABILITIES = ProviderCapabilities(
    provider="deepseek",
    supports_reasoning_content=True,
    supports_reasoning_effort=True,
    supports_thinking_toggle=True,
    supports_streaming=True,
    supports_cache_metrics=True,
    supports_tool_calls=True,
)

GENERIC_CAPABILITIES = ProviderCapabilities()


def detect_capabilities(base_url: str, model: str) -> ProviderCapabilities:
    text = f"{base_url} {model}".lower()
    if "api.deepseek.com" in text or model.lower().startswith("deepseek-"):
        return DEEPSEEK_CAPABILITIES
    return GENERIC_CAPABILITIES


def build_chat_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float | None,
    stream: bool,
    capabilities: ProviderCapabilities,
    thinking_enabled: bool | None = None,
    reasoning_effort: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "stream": bool(stream),
    }

    # In DeepSeek thinking mode these sampling params have no effect, so do
    # not pretend they are benchmark knobs. Generic providers keep the old
    # behavior.
    thinking_active = (
        capabilities.provider == "deepseek"
        and (thinking_enabled is None or thinking_enabled)
    )
    if temperature is not None and not thinking_active:
        payload["temperature"] = float(temperature)

    if capabilities.supports_thinking_toggle and thinking_enabled is not None:
        payload["thinking"] = {
            "type": "enabled" if thinking_enabled else "disabled"
        }

    if capabilities.supports_reasoning_effort and reasoning_effort:
        effort = str(reasoning_effort).lower()
        if effort not in {"low", "high", "max"}:
            raise ValueError(f"invalid reasoning_effort: {reasoning_effort!r}")
        payload["reasoning_effort"] = effort

    if tools:
        payload["tools"] = tools
    return payload


@dataclass
class StreamAccumulator:
    reasoning: str = ""
    content: str = ""
    finish_reason: str = ""
    usage: dict[str, Any] | None = None
    first_reasoning_seen: bool = False
    first_content_seen: bool = False

    def feed_event(self, event: dict[str, Any]) -> tuple[str, str]:
        """Consume one OpenAI-compatible stream chunk.

        Returns (new_reasoning_delta, new_content_delta).
        """
        if isinstance(event.get("usage"), dict):
            self.usage = event["usage"]

        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            return "", ""

        choice = choices[0] or {}
        if choice.get("finish_reason"):
            self.finish_reason = str(choice["finish_reason"])

        delta = choice.get("delta") or {}
        reasoning_delta = delta.get("reasoning_content")
        content_delta = delta.get("content")

        if not isinstance(reasoning_delta, str):
            reasoning_delta = ""
        if not isinstance(content_delta, str):
            content_delta = ""

        if reasoning_delta:
            self.reasoning += reasoning_delta
            self.first_reasoning_seen = True
        if content_delta:
            self.content += content_delta
            self.first_content_seen = True
        return reasoning_delta, content_delta


def parse_sse_data_lines(lines: Iterable[bytes | str]) -> list[dict[str, Any]]:
    """Parse `data: {...}` SSE frames from a fake/real HTTP byte iterator.

    Network ownership stays in llm_client.py; this helper is easy to unit-test.
    """
    events: list[dict[str, Any]] = []
    for raw in lines:
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events
