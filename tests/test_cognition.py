"""Cognition (adaptive reasoning) + COMBAT_RESOLVED tests.

Run:  python .\\tests\\test_cognition.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentSession
from benchmark_metrics import BenchmarkMetrics
from checkpoint import CheckpointReason
from context_manager import ContextConfig, ContextManager
from plan_executor import ActionChunkExecutor, ExecutorStatus


class _FakeCaps:
    def __init__(self, provider):
        self.provider = provider
        # Mirror the real capability objects: only DeepSeek exposes the
        # reasoning_effort parameter.
        self.supports_reasoning_effort = provider == "deepseek"


class _FakeLLM:
    def __init__(self, provider):
        self.caps = _FakeCaps(provider)
        self.reasoning_effort = "unset"


def make_session(provider="deepseek", policy="adaptive"):
    s = AgentSession()
    s._config["reasoning_policy"] = policy
    s._llm = _FakeLLM(provider)
    return s


# ----------------------------------------------------------------
# §3/§4/§48 fixed + adaptive effort resolution
# ----------------------------------------------------------------

def test_fixed_effort_used() -> None:
    s = make_session(provider="deepseek", policy="fixed")
    s._config["reasoning_effort_fixed"] = "low"
    s._apply_reasoning("noncombat")
    assert s._llm.reasoning_effort == "low"
    assert s._reasoning_context_class == "fixed"
    assert s._effective_reasoning_effort == "low"
    print("PASS fixed_effort_used")


def test_adaptive_combat_entry_high() -> None:
    s = make_session()
    s._apply_reasoning("combat_entry")
    assert s._llm.reasoning_effort == "high"  # default combat_entry = high
    assert s._reasoning_context_class == "combat_entry"
    print("PASS adaptive_combat_entry_high")


def test_adaptive_combat_followup_low() -> None:
    s = make_session()
    s._apply_reasoning("combat_followup")
    assert s._llm.reasoning_effort == "low"  # default followup = low
    print("PASS adaptive_combat_followup_low")


def test_adaptive_noncombat_high() -> None:
    s = make_session()
    s._apply_reasoning("noncombat")
    assert s._llm.reasoning_effort == "high"
    print("PASS adaptive_noncombat_high")


def test_adaptive_retry_high() -> None:
    s = make_session()
    s._apply_reasoning("combat_followup")  # would be low...
    s._apply_reasoning("combat_followup", retry=True)
    assert s._llm.reasoning_effort == "high"  # retry overrides to high
    assert s._reasoning_context_class == "retry"
    print("PASS adaptive_retry_high")


def test_provider_medium_mapping() -> None:
    s = make_session(provider="deepseek")
    s._config["reasoning_effort_combat_entry"] = "medium"
    s._apply_reasoning("combat_entry")
    # DeepSeek maps medium -> high (never pretends medium is a real tier).
    assert s._requested_reasoning_effort == "medium"
    assert s._effective_reasoning_effort == "high"
    assert s._llm.reasoning_effort == "high"
    print("PASS provider_medium_mapping")


def test_unsupported_effort_graceful() -> None:
    s = make_session(provider="generic")  # caps.supports_reasoning_effort False
    s._apply_reasoning("noncombat")
    assert s._llm.reasoning_effort is None  # parameter omitted, no failure
    assert s._effective_reasoning_effort == "provider_default"
    print("PASS unsupported_effort_graceful")


def test_legacy_reasoning_effort_migrates() -> None:
    from agent import migrate_reasoning_config

    # A legacy user config (only reasoning_effort) survives the merge.
    merged = {
        **{"reasoning_effort_fixed": "high"},
        **migrate_reasoning_config({"reasoning_effort": "low"}),
    }
    assert merged["reasoning_effort_fixed"] == "low", merged
    # An explicit new key is never overridden.
    kept = migrate_reasoning_config({"reasoning_effort": "low",
                                    "reasoning_effort_fixed": "max"})
    assert kept["reasoning_effort_fixed"] == "max"
    print("PASS legacy_reasoning_effort_migrates")


# ----------------------------------------------------------------
# §20 — combat resolved hard checkpoint
# ----------------------------------------------------------------

def combat_state(*, alive: bool, hand=None, potions=None):
    return {
        "type": "combat_action", "request_id": "r", "round": 1,
        "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 3,
                   "max_energy": 3},
        "hand": hand or [],
        "enemies": [{"id": "CULTIST", "hp": 48 if alive else 0,
                     "is_alive": alive, "intent": "ATTACK"}],
        "potions": potions or [],
    }


def test_last_enemy_dies_discards_remaining_chunk() -> None:
    state = combat_state(
        alive=True,
        hand=[{"id": "STRIKE", "target": "AnyEnemy", "playable": True},
              {"id": "STRIKE", "target": "AnyEnemy", "playable": True}],
    )
    chunk = agent_parse({
        "thought": "kill and keep swinging",
        "actions": [
            {"kind": "play", "card_ref": "h0", "target_ref": "e0"},
            {"kind": "play", "card_ref": "h1", "target_ref": "e0"},
        ],
    }, state)
    ex = ActionChunkExecutor()
    ex.submit(chunk)
    ev = ex.prepare_next(state, _ok_validate)
    ex.mark_sent(ev.prepared, state)

    dead = combat_state(alive=False, hand=[])
    ev = ex.accept_state(dead)
    assert ev.status.value == "NEED_MODEL", ev
    assert ev.checkpoint.reason == CheckpointReason.COMBAT_RESOLVED, ev
    assert not ex.has_pending_plan  # remainder discarded
    print("PASS last_enemy_dies_discards_remaining_chunk")


def test_last_enemy_dies_before_self_potion() -> None:
    state = combat_state(
        alive=True,
        hand=[{"id": "STRIKE", "target": "AnyEnemy", "playable": True}],
        potions=[{"slot": 0, "id": "BLOOD", "can_use": True}],
    )
    chunk = agent_parse({
        "thought": "kill then self potion",
        "actions": [
            {"kind": "play", "card_ref": "h0", "target_ref": "e0"},
            {"kind": "potion", "potion_slot": 0},
        ],
    }, state)
    ex = ActionChunkExecutor()
    ex.submit(chunk)
    ev = ex.prepare_next(state, _ok_validate)
    ex.mark_sent(ev.prepared, state)

    dead = combat_state(alive=False, hand=[],
                        potions=[{"slot": 0, "id": "BLOOD", "can_use": True}])
    ev = ex.accept_state(dead)
    assert ev.checkpoint.reason == CheckpointReason.COMBAT_RESOLVED, ev
    assert not ex.has_pending_plan
    print("PASS last_enemy_dies_before_self_potion")


def test_last_enemy_dies_before_end_turn() -> None:
    state = combat_state(
        alive=True,
        hand=[{"id": "STRIKE", "target": "AnyEnemy", "playable": True}],
    )
    chunk = agent_parse({
        "thought": "kill then end",
        "actions": [
            {"kind": "play", "card_ref": "h0", "target_ref": "e0"},
            {"kind": "end_turn"},
        ],
    }, state)
    ex = ActionChunkExecutor()
    ex.submit(chunk)
    ev = ex.prepare_next(state, _ok_validate)
    ex.mark_sent(ev.prepared, state)

    dead = combat_state(alive=False, hand=[])
    ev = ex.accept_state(dead)
    assert ev.checkpoint.reason == CheckpointReason.COMBAT_RESOLVED, ev
    assert not ex.has_pending_plan
    print("PASS last_enemy_dies_before_end_turn")


def agent_parse(obj, state):
    from action_plan import parse_action_chunk
    return parse_action_chunk(obj, state)


def _ok_validate(state, act):
    return act, ""


# ----------------------------------------------------------------
# §10 — context controls
# ----------------------------------------------------------------

def test_history_zero_and_three() -> None:
    ctx0 = ContextManager(config=ContextConfig(max_history_turns=0))
    for i in range(5):
        ctx0.add_decision(f"S{i}", "R", "")
    msgs = ctx0.build_messages("SYS", "MEM", "STATE")
    assert not any("S0" in m["content"] and "R" in m["content"]
                   for m in msgs), "history must be empty at 0 turns"

    ctx3 = ContextManager(config=ContextConfig(max_history_turns=3))
    for i in range(10):
        ctx3.add_decision("S" * 20 + str(i), "R" + str(i), "")
    msgs = ctx3.build_messages("SYS", "MEM", "STATE")
    joined = "".join(m["content"] for m in msgs)
    assert "R9" in joined and "R8" in joined and "R7" in joined
    assert "R0" not in joined  # only the most recent 3 kept
    print("PASS history_zero_and_three")


def test_current_state_never_soft_truncated() -> None:
    ctx = ContextManager(config=ContextConfig(
        max_context_chars=500))  # tiny budget
    big_state = "STATE-" + "x" * 5000
    ctx.add_decision("OLD" * 100, "R", "")
    msgs = ctx.build_messages("SYS", "MEM", big_state)
    user = msgs[-1]["content"]
    assert big_state in user  # current state preserved whole
    print("PASS current_state_never_soft_truncated")


def test_reasoning_metrics_by_effort() -> None:
    m = BenchmarkMetrics()
    m.record_llm_success(usage={"reasoning_tokens": 100},
                         latency_ms=1000, effort="low")
    m.record_llm_success(usage={"reasoning_tokens": 500},
                         latency_ms=3000, effort="high")
    snap = m.snapshot()
    by = snap["reasoning_by_effort"]
    assert by["low"]["calls"] == 1 and by["low"]["reasoning_tokens"] == 100
    assert by["high"]["calls"] == 1 and by["high"]["reasoning_tokens"] == 500
    assert by["low"]["latency_ms_p50"] == 1000.0
    print("PASS reasoning_metrics_by_effort")


def run_all() -> None:
    tests = [
        test_fixed_effort_used,
        test_adaptive_combat_entry_high,
        test_adaptive_combat_followup_low,
        test_adaptive_noncombat_high,
        test_adaptive_retry_high,
        test_provider_medium_mapping,
        test_unsupported_effort_graceful,
        test_legacy_reasoning_effort_migrates,
        test_last_enemy_dies_discards_remaining_chunk,
        test_last_enemy_dies_before_self_potion,
        test_last_enemy_dies_before_end_turn,
        test_history_zero_and_three,
        test_current_state_never_soft_truncated,
        test_reasoning_metrics_by_effort,
    ]
    for fn in tests:
        fn()
    print(f"\nALL {len(tests)} COGNITION TESTS PASSED")


if __name__ == "__main__":
    run_all()
