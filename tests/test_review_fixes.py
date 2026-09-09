"""Unit tests for the review-remediation fixes (FIX 6/7/8 + metrics).

  FIX 6  provider_profile: a GENERIC relay serving a model named
         "deepseek-v4-pro" must NOT receive native DeepSeek fields in
         auto mode; the official api.deepseek.com hostname must; explicit
         generic/deepseek profiles override.
  FIX 7  extract_json must repair ActionChunk JSON ("actions" top level).
  FIX 8  streaming must record first_reasoning_token_ms /
         first_content_token_ms exactly ONCE per call, and provider
         usage (tokens / cache / reasoning) must flow into metrics.
  FIX 5  BenchmarkMetrics request/success/failure + sent/confirmed/
         rejected accounting.

Run:  python .\\tests\\test_review_fixes.py
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import llm_client as llm_client_mod
from agent import _looks_like_decision_object, extract_json
from benchmark_metrics import BenchmarkMetrics
from deepseek_provider_reference import build_chat_payload
from llm_client import LLMClient


# ----------------------------------------------------------------
# FIX 6 — provider capability detection by profile, not model name
# ----------------------------------------------------------------

def _payload(llm: LLMClient, thinking=True, effort="high"):
    return build_chat_payload(
        model=llm.model,
        messages=[{"role": "user", "content": "x"}],
        max_tokens=100,
        temperature=0.4,
        stream=False,
        capabilities=llm.caps,
        thinking_enabled=thinking,
        reasoning_effort=effort,
    )


def test_generic_relay_with_deepseek_model_name_gets_no_native_fields():
    llm = LLMClient(base_url="http://localhost:3000", api_key="k",
                    model="deepseek-v4-pro")  # auto profile
    assert llm.caps.provider == "generic", llm.caps
    payload = _payload(llm)
    assert "thinking" not in payload, payload
    assert "reasoning_effort" not in payload, payload
    assert payload["temperature"] == 0.4  # generic keeps old behavior
    print("PASS FIX 6a generic_relay_no_native_fields")


def test_official_deepseek_hostname_gets_native_fields():
    llm = LLMClient(base_url="https://api.deepseek.com", api_key="k",
                    model="deepseek-chat")  # auto profile
    assert llm.caps.provider == "deepseek", llm.caps
    payload = _payload(llm)
    assert payload["thinking"] == {"type": "enabled"}, payload
    assert payload["reasoning_effort"] == "high", payload
    assert "temperature" not in payload  # no effect in thinking mode
    print("PASS FIX 6b official_deepseek_native_fields")


def test_explicit_profiles_override():
    forced_generic = LLMClient(base_url="https://api.deepseek.com",
                               api_key="k", model="deepseek-chat",
                               provider_profile="generic")
    assert forced_generic.caps.provider == "generic"
    assert "thinking" not in _payload(forced_generic)

    forced_deepseek = LLMClient(base_url="http://localhost:3000",
                                api_key="k", model="deepseek-chat",
                                provider_profile="deepseek")
    assert forced_deepseek.caps.provider == "deepseek"
    assert _payload(forced_deepseek)["thinking"] == {"type": "enabled"}
    print("PASS FIX 6c explicit_profiles_override")


# ----------------------------------------------------------------
# FIX 7 — extract_json repairs ActionChunk JSON
# ----------------------------------------------------------------

def test_extract_json_repairs_chunk() -> None:
    assert _looks_like_decision_object({"action": "end_turn"})
    assert _looks_like_decision_object({"actions": [{"kind": "end_turn"}]})
    assert not _looks_like_decision_object({"foo": 1})
    assert not _looks_like_decision_object({"actions": "nope"})

    # json-repair path: trailing comma + "actions" top level.
    obj = extract_json('{"thought":"x","actions":[{"kind":"end_turn"}],}')
    assert obj["actions"] == [{"kind": "end_turn"}], obj

    # Salvage/repair path: relay-corrupted TRUNCATED reply (no closing
    # braces, so the plain regex extractor cannot match) whose tail still
    # carries the "actions" key.
    obj = extract_json(
        'relay noise {"thought":"lost", "actions":[{"kind":"play",'
        '"card_ref":"h0"}'
    )
    assert isinstance(obj.get("actions"), list), obj
    assert obj["actions"][0]["kind"] == "play", obj
    print("PASS FIX 7 extract_json_repairs_chunk")


# ----------------------------------------------------------------
# FIX 8 — streaming first-token timing + usage into metrics
# ----------------------------------------------------------------

class _FakeSSEHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = (
            'data: {"choices":[{"delta":{"reasoning_content":"think "}}]}\n\n'
            'data: {"choices":[{"delta":{"reasoning_content":"more"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"{\\"kind\\":\\"end_turn\\"}"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":10,"completion_tokens":5,'
            '"prompt_cache_hit_tokens":4,"prompt_cache_miss_tokens":6,'
            '"completion_tokens_details":{"reasoning_tokens":3}}}\n\n'
            "data: [DONE]\n\n"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def test_streaming_first_token_and_usage_metrics() -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeSSEHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        llm = LLMClient(base_url=f"http://127.0.0.1:{port}", api_key="k",
                        model="deepseek-chat", provider_profile="deepseek",
                        stream_mode="on", max_tokens=100)
        reasoning_seen: list = []
        llm.on_reasoning_delta = lambda d: reasoning_seen.append(
            llm.first_reasoning_ms)

        text = llm.chat([{"role": "user", "content": "hi"}])
        assert text == '{"kind":"end_turn"}', text
        # First-token timestamps were written EXACTLY ONCE (the second
        # reasoning delta must not rewrite them).
        assert isinstance(llm.first_reasoning_ms, int), llm.first_reasoning_ms
        assert isinstance(llm.first_content_ms, int), llm.first_content_ms
        assert len(reasoning_seen) == 2, reasoning_seen
        assert reasoning_seen[0] is not None
        assert reasoning_seen[1] == reasoning_seen[0]

        # Provider usage (tokens / cache / reasoning) flows into metrics.
        m = BenchmarkMetrics()
        m.record_llm_request()
        m.record_llm_success(
            usage=llm.last_usage,
            first_reasoning_ms=llm.first_reasoning_ms,
            first_content_ms=llm.first_content_ms,
        )
        snap = m.snapshot()
        assert snap["llm_request_count"] == 1
        assert snap["llm_success_count"] == 1
        assert snap["llm_failed_request_count"] == 0
        assert snap["prompt_tokens"] == 10, snap
        assert snap["completion_tokens"] == 5, snap
        assert snap["reasoning_tokens"] == 3, snap
        assert snap["prompt_cache_hit_tokens"] == 4, snap
        assert snap["prompt_cache_miss_tokens"] == 6, snap
        assert abs(snap["cache_hit_ratio"] - 0.4) < 1e-9, snap
        assert snap["first_reasoning_token_ms_p50"] is not None
        assert snap["first_content_token_ms_p50"] is not None
    finally:
        srv.shutdown()
    print("PASS FIX 8 streaming_first_token_and_usage_metrics")


# ----------------------------------------------------------------
# FIX 5 — granular metrics accounting
# ----------------------------------------------------------------

def test_metrics_sent_confirmed_rejected_and_requests() -> None:
    m = BenchmarkMetrics()
    m.record_inspection()
    m.record_llm_request()
    m.record_llm_failure()          # attempt 1 failed (timeout)
    m.record_llm_request()
    m.record_llm_success(latency_ms=900)  # attempt 2 ok
    m.record_plan(3)
    m.record_action_sent()          # 3 planned actions sent...
    m.record_action_sent()
    m.record_action_sent()
    m.record_action_confirmed()     # ...all confirmed by the bridge
    m.record_action_confirmed()
    m.record_action_confirmed()
    m.record_action_sent()
    m.record_action_rejected()      # a 4th sent action was rejected

    snap = m.snapshot()
    assert snap["logical_inspection_count"] == 1, snap
    assert snap["llm_request_count"] == 2, snap
    assert snap["llm_success_count"] == 1, snap
    assert snap["llm_failed_request_count"] == 1, snap
    assert snap["game_action_sent_count"] == 4, snap
    assert snap["game_action_confirmed_count"] == 3, snap
    assert snap["game_action_rejected_count"] == 1, snap
    assert snap["game_action_count"] == 3  # compat alias == confirmed
    # actions_per_llm_call denominator = EVERY request (failed included).
    assert abs(snap["actions_per_llm_call"] - 1.5) < 1e-9, snap
    assert abs(snap["actions_per_logical_inspection"] - 3.0) < 1e-9, snap
    print("PASS FIX 5 metrics_sent_confirmed_rejected_and_requests")


# ----------------------------------------------------------------
# Item 3A — internal HTTP retries are each counted as a request
# ----------------------------------------------------------------

class _ScriptedHTTPHandler(BaseHTTPRequestHandler):
    """Returns scripted (status, body) pairs; 200 with usage by default."""
    responses: list[tuple[int, dict]] = []

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        status, body = type(self).responses.pop(0) if type(self).responses \
            else (200, {"choices": [{"message": {"content": "ok"},
                                     "finish_reason": "stop"}]})
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


def _start_scripted_server(responses):
    handler = type("_H", (_ScriptedHTTPHandler,), {"responses": responses})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_internal_http_retry_counted() -> None:
    # Fail twice (HTTP 500) then succeed: ONE llm.chat => THREE real HTTP
    # attempts, and llm_request_count must be 3 (sleep patched for speed).
    old_sleep = llm_client_mod.time.sleep
    llm_client_mod.time.sleep = lambda s: None
    srv = _start_scripted_server([
        (500, {"error": "boom"}),
        (500, {"error": "boom"}),
        (200, {"choices": [{"message": {"content": "ok"},
                            "finish_reason": "stop"}]}),
    ])
    try:
        m = BenchmarkMetrics()
        llm = LLMClient(base_url=f"http://127.0.0.1:{srv.server_address[1]}",
                        api_key="k", model="m", max_retries=2)
        attempts: list[int] = []
        llm.on_http_attempt = lambda a: (
            attempts.append(a), m.record_llm_request())
        text = llm.chat([{"role": "user", "content": "hi"}])
        assert text == "ok", text
        assert len(attempts) == 3, attempts
        assert m.llm_request_count == 3, m.snapshot()
        assert m.llm_failed_request_count == 0  # agent marks failures, not the client
        assert m.llm_success_count == 0
    finally:
        llm_client_mod.time.sleep = old_sleep
        srv.shutdown()
    print("PASS item3 internal_http_retry_counted")


def test_stale_last_usage_not_reused() -> None:
    srv = _start_scripted_server([
        (200, {"choices": [{"message": {"content": "one"},
                            "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 5}}),
        (200, {"choices": [{"message": {"content": "two"},
                            "finish_reason": "stop"}]}),  # NO usage
    ])
    try:
        m = BenchmarkMetrics()
        llm = LLMClient(base_url=f"http://127.0.0.1:{srv.server_address[1]}",
                        api_key="k", model="m")
        llm.on_http_attempt = lambda a: m.record_llm_request()
        llm.chat([{"role": "user", "content": "a"}])
        m.record_llm_success(usage=llm.last_usage)
        assert m.prompt_tokens == 10, m.snapshot()

        llm.chat([{"role": "user", "content": "b"}])
        # Per-call metadata reset: call 2 had NO usage -> last_usage is
        # empty and call 1's tokens are NOT counted a second time.
        assert llm.last_usage == {}, llm.last_usage
        m.record_llm_success(usage=llm.last_usage)
        snap = m.snapshot()
        assert snap["prompt_tokens"] == 10, snap
        assert snap["completion_tokens"] == 5, snap
    finally:
        srv.shutdown()
    print("PASS item5 stale_last_usage_not_reused")


def test_model_call_count_alias_consistent() -> None:
    m = BenchmarkMetrics()
    m.record_llm_request()
    m.record_llm_request()
    m.record_llm_failure()  # failures never change the alias
    snap = m.snapshot()
    assert snap["model_call_count"] == 2 == m.model_call_count, snap
    assert m.model_call_count == m.llm_request_count
    m.record_llm_request()
    assert m.model_call_count == m.llm_request_count == 3
    print("PASS item6 model_call_count_alias_consistent")


def test_formatter_error_fingerprint_is_none() -> None:
    """format_state catches formatter exceptions and returns a text that
    embeds the RAW payload (request_id / non-visible fields). Such a
    state must yield fingerprint None -> UNKNOWN_CONFIRMATION."""
    import agent as agent_mod

    original = agent_mod.format_state
    agent_mod.format_state = lambda *a, **k: (
        '== COMBAT (formatting error: boom) ==\nraw: {"request_id": "r1"}'
    )
    try:
        s = agent_mod.AgentSession()
        assert s._visible_state_fingerprint(
            {"type": "combat_action"}) is None
    finally:
        agent_mod.format_state = original
    print("PASS microfix formatter_error_fingerprint_is_none")


def run_all() -> None:
    tests = [
        test_generic_relay_with_deepseek_model_name_gets_no_native_fields,
        test_official_deepseek_hostname_gets_native_fields,
        test_explicit_profiles_override,
        test_extract_json_repairs_chunk,
        test_streaming_first_token_and_usage_metrics,
        test_metrics_sent_confirmed_rejected_and_requests,
        test_internal_http_retry_counted,
        test_stale_last_usage_not_reused,
        test_model_call_count_alias_consistent,
        test_formatter_error_fingerprint_is_none,
    ]
    for fn in tests:
        fn()
    print(f"\nALL {len(tests)} REVIEW-FIX TESTS PASSED")


if __name__ == "__main__":
    run_all()
