from __future__ import annotations

import sys
from pathlib import Path

# Core modules live at the repository root (repo-root import).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from action_plan import parse_action_chunk, PlanParseError
from plan_executor import ActionChunkExecutor, ExecutorStatus
from checkpoint import (
    ActionAcceptance,
    CheckpointReason,
    resolve_action_acceptance,
)
from state_diff import diff_states, render_delta
from benchmark_metrics import BenchmarkMetrics
from deepseek_provider_reference import (
    DEEPSEEK_CAPABILITIES,
    build_chat_payload,
    parse_sse_data_lines,
    StreamAccumulator,
)


def combat_state(
    *,
    request_id: str,
    energy: int,
    hand: list[dict],
    enemies: list[dict] | None = None,
    round_: int = 1,
    block: int = 0,
    draw_count: int = 5,
    discard_count: int = 0,
):
    return {
        "type": "combat_action",
        "request_id": request_id,
        "round": round_,
        "player": {
            "hp": 70,
            "max_hp": 80,
            "block": block,
            "energy": energy,
            "max_energy": 3,
            "gold": 99,
        },
        "hand": hand,
        "enemies": enemies or [
            {
                "id": "CULTIST",
                "hp": 40,
                "max_hp": 48,
                "block": 0,
                "is_alive": True,
                "intent": "ATTACK",
                "intent_damage": 6,
                "intent_hits": 1,
            }
        ],
        "potions": [],
        "draw_pile_count": draw_count,
        "discard_pile_count": discard_count,
        "exhaust_pile_count": 0,
    }


def card(cid: str, *, target="Self", playable=True, cost=1, upgraded=False):
    return {
        "id": cid,
        "display_name": cid.title(),
        "target": target,
        "playable": playable,
        "current_energy_cost": cost,
        "upgraded": upgraded,
    }


def validate_action(state, act):
    name = act.get("action")
    if name == "end_turn":
        return {"action": "end_turn"}, ""
    if name == "play":
        hand = state.get("hand") or []
        ci = act.get("card_index")
        if not isinstance(ci, int) or not (0 <= ci < len(hand)):
            return None, "bad card index"
        c = hand[ci]
        if not c.get("playable"):
            return None, "card not playable"
        ti = act.get("target_index", -1)
        if c.get("target") == "AnyEnemy":
            enemies = state.get("enemies") or []
            if not isinstance(ti, int) or not (0 <= ti < len(enemies)):
                return None, "bad target"
            if not enemies[ti].get("is_alive"):
                return None, "dead target"
        return {"action": "play", "card_index": ci, "target_index": ti}, ""
    if name == "potion":
        return act, ""
    return None, "unsupported"


def test_one_call_multiple_actions():
    s0 = combat_state(
        request_id="r0",
        energy=3,
        hand=[
            card("STRIKE", target="AnyEnemy"),
            card("DEFEND"),
        ],
    )
    chunk = parse_action_chunk(
        {
            "thought": "Attack, defend, then end the turn.",
            "actions": [
                {"kind": "play", "card_ref": "h0", "target_ref": "e0"},
                {"kind": "play", "card_ref": "h1"},
                {"kind": "end_turn"},
            ],
        },
        s0,
    )
    ex = ActionChunkExecutor()
    ex.submit(chunk)

    # Step 1
    ev = ex.prepare_next(s0, validate_action)
    assert ev.status == ExecutorStatus.READY_ACTION
    assert ev.prepared.bridge_action == {
        "action": "play",
        "card_index": 0,
        "target_index": 0,
    }
    ex.mark_sent(ev.prepared, s0)

    s1 = combat_state(
        request_id="r1",
        energy=2,
        hand=[card("DEFEND")],
        enemies=[{
            "id": "CULTIST",
            "hp": 34,
            "max_hp": 48,
            "block": 0,
            "is_alive": True,
            "intent": "ATTACK",
            "intent_damage": 6,
            "intent_hits": 1,
        }],
        discard_count=1,
    )
    ev = ex.accept_state(s1)
    assert ev.status == ExecutorStatus.READY_ACTION, ev
    ev = ex.prepare_next(s1, validate_action)
    assert ev.prepared.bridge_action == {
        "action": "play",
        "card_index": 0,  # original h1 shifted to current index 0
        "target_index": -1,
    }
    ex.mark_sent(ev.prepared, s1)

    s2 = combat_state(
        request_id="r2",
        energy=1,
        hand=[],
        enemies=s1["enemies"],
        block=5,
        discard_count=2,
    )
    ev = ex.accept_state(s2)
    assert ev.status == ExecutorStatus.READY_ACTION
    ev = ex.prepare_next(s2, validate_action)
    assert ev.prepared.bridge_action == {"action": "end_turn"}
    ex.mark_sent(ev.prepared, s2)

    s3 = combat_state(
        request_id="r3",
        energy=3,
        hand=[card("STRIKE", target="AnyEnemy"), card("DEFEND")],
        enemies=s1["enemies"],
        round_=2,
        block=0,
        draw_count=3,
        discard_count=2,
    )
    ev = ex.accept_state(s3)
    assert ev.status == ExecutorStatus.NEED_MODEL
    assert ev.checkpoint.reason == CheckpointReason.NEW_TURN


def test_draw_causes_checkpoint():
    s0 = combat_state(
        request_id="r0",
        energy=3,
        hand=[
            card("POMMEL_STRIKE", target="AnyEnemy"),
            card("STRIKE", target="AnyEnemy"),
        ],
    )
    chunk = parse_action_chunk(
        {
            "thought": "Pommel first; continue only if no new information appears.",
            "actions": [
                {"kind": "play", "card_ref": "h0", "target_ref": "e0"},
                {"kind": "play", "card_ref": "h1", "target_ref": "e0"},
            ],
        },
        s0,
    )
    ex = ActionChunkExecutor()
    ex.submit(chunk)
    ev = ex.prepare_next(s0, validate_action)
    ex.mark_sent(ev.prepared, s0)

    s1 = combat_state(
        request_id="r1",
        energy=2,
        hand=[
            card("STRIKE", target="AnyEnemy"),
            card("BASH", target="AnyEnemy", cost=2),
        ],
        enemies=[{
            "id": "CULTIST",
            "hp": 31,
            "max_hp": 48,
            "block": 0,
            "is_alive": True,
            "intent": "ATTACK",
            "intent_damage": 6,
            "intent_hits": 1,
        }],
        draw_count=4,
        discard_count=1,
    )
    ev = ex.accept_state(s1)
    assert ev.status == ExecutorStatus.NEED_MODEL
    assert ev.checkpoint.reason == CheckpointReason.HAND_CHANGED
    assert not ex.has_pending_plan


def test_target_gone_causes_checkpoint():
    s0 = combat_state(
        request_id="r0",
        energy=3,
        hand=[
            card("STRIKE", target="AnyEnemy"),
            card("STRIKE", target="AnyEnemy"),
        ],
        enemies=[{
            "id": "SLIME",
            "hp": 5,
            "max_hp": 20,
            "block": 0,
            "is_alive": True,
            "intent": "ATTACK",
        }],
    )
    chunk = parse_action_chunk(
        {
            "thought": "Attack twice if target survives.",
            "actions": [
                {"kind": "play", "card_ref": "h0", "target_ref": "e0"},
                {"kind": "play", "card_ref": "h1", "target_ref": "e0"},
            ],
        },
        s0,
    )
    ex = ActionChunkExecutor()
    ex.submit(chunk)
    ev = ex.prepare_next(s0, validate_action)
    ex.mark_sent(ev.prepared, s0)

    s1 = combat_state(
        request_id="r1",
        energy=2,
        hand=[card("STRIKE", target="AnyEnemy")],
        enemies=[{
            "id": "SLIME",
            "hp": 0,
            "max_hp": 20,
            "block": 0,
            "is_alive": False,
            "intent": "UNKNOWN",
        }],
        discard_count=1,
    )
    ev = ex.accept_state(s1)
    assert ev.status == ExecutorStatus.NEED_MODEL
    # SEMANTICS UPDATE (combat-resolved invariant): the single enemy is
    # dead AND the screen is still combat_action -> the combat is RESOLVED.
    # COMBAT_RESOLVED supersedes TARGET_GONE for a fully dead board; the
    # caller (agent) re-inspects at the next real screen either way.
    assert ev.checkpoint.reason in (
        CheckpointReason.TARGET_GONE,
        CheckpointReason.COMBAT_RESOLVED,
    ), ev


def test_duplicate_cards_resolve_after_shift():
    s0 = combat_state(
        request_id="r0",
        energy=3,
        hand=[
            card("STRIKE", target="AnyEnemy"),
            card("STRIKE", target="AnyEnemy"),
            card("DEFEND"),
        ],
    )
    chunk = parse_action_chunk(
        {
            "thought": "Use both identical Strikes.",
            "actions": [
                {"kind": "play", "card_ref": "h0", "target_ref": "e0"},
                {"kind": "play", "card_ref": "h1", "target_ref": "e0"},
            ],
        },
        s0,
    )
    ex = ActionChunkExecutor()
    ex.submit(chunk)
    ev = ex.prepare_next(s0, validate_action)
    ex.mark_sent(ev.prepared, s0)

    s1 = combat_state(
        request_id="r1",
        energy=2,
        hand=[
            card("STRIKE", target="AnyEnemy"),
            card("DEFEND"),
        ],
        discard_count=1,
    )
    ev = ex.accept_state(s1)
    assert ev.status == ExecutorStatus.READY_ACTION
    ev = ex.prepare_next(s1, validate_action)
    assert ev.prepared.bridge_action["card_index"] == 0


def test_unchanged_state_with_authoritative_rejection_is_rejected():
    """Game-side REJECTED + unchanged state => ACTION_REJECTED, plan reset."""
    s0 = combat_state(
        request_id="r0",
        energy=3,
        hand=[card("STRIKE", target="AnyEnemy")],
    )
    chunk = parse_action_chunk(
        {
            "thought": "Strike once.",
            "actions": [{"kind": "play", "card_ref": "h0", "target_ref": "e0"}],
        },
        s0,
    )
    ex = ActionChunkExecutor()
    ex.submit(chunk)
    ev = ex.prepare_next(s0, validate_action)
    ex.mark_sent(ev.prepared, s0)

    s1 = dict(s0)
    s1["request_id"] = "r1"
    ev = ex.accept_state(s1, acceptance=ActionAcceptance.REJECTED)
    assert ev.status == ExecutorStatus.NEED_MODEL
    assert ev.checkpoint.reason == CheckpointReason.ACTION_REJECTED
    assert not ex.has_pending_plan


def test_unchanged_state_without_authoritative_signal_is_conservative_wait():
    """UNKNOWN + unchanged state => ADVANCE_UNVERIFIED: conservative bounded
    WAITING. Never confirmed, never rejected, action retained."""
    s0 = combat_state(
        request_id="r0",
        energy=3,
        hand=[card("STRIKE", target="AnyEnemy")],
    )
    chunk = parse_action_chunk(
        {
            "thought": "Strike once.",
            "actions": [{"kind": "play", "card_ref": "h0", "target_ref": "e0"}],
        },
        s0,
    )
    ex = ActionChunkExecutor()
    ex.submit(chunk)
    ev = ex.prepare_next(s0, validate_action)
    ex.mark_sent(ev.prepared, s0)

    s1 = dict(s0)
    s1["request_id"] = "r1"
    ev = ex.accept_state(s1)  # acceptance defaults to UNKNOWN
    assert ev.status == ExecutorStatus.WAITING_ADVANCE, ev
    assert ev.checkpoint.reason == CheckpointReason.ADVANCE_UNVERIFIED
    assert ex.inflight is not None
    assert ex.has_pending_plan
    assert ex.index == 0


def test_unchanged_state_with_bridge_accept_is_awaiting_advance():
    """Bridge ACCEPTED the command + unchanged state => AWAITING_ADVANCE:
    the in-flight step/plan are RETAINED (no advance, no reset), and a later
    authoritative advance confirms the SAME command exactly once."""
    s0 = combat_state(
        request_id="r0",
        energy=3,
        hand=[card("STRIKE", target="AnyEnemy"), card("DEFEND")],
    )
    chunk = parse_action_chunk(
        {
            "thought": "Strike, then defend.",
            "actions": [
                {"kind": "play", "card_ref": "h0", "target_ref": "e0"},
                {"kind": "play", "card_ref": "h1"},
            ],
        },
        s0,
    )
    ex = ActionChunkExecutor()
    ex.submit(chunk)
    ev = ex.prepare_next(s0, validate_action)
    ex.mark_sent(ev.prepared, s0)

    # Accepted but the visible world has not moved: WAIT, do not reset.
    s1 = dict(s0)
    s1["request_id"] = "r1"
    ev = ex.accept_state(s1, acceptance=ActionAcceptance.ACCEPTED)
    assert ev.status == ExecutorStatus.WAITING_ADVANCE, ev
    assert ev.checkpoint.reason == CheckpointReason.AWAITING_ADVANCE
    assert ex.inflight is not None
    assert ex.has_pending_plan
    assert ex.index == 0

    # The authoritative world finally advances -> the ORIGINAL in-flight
    # action is confirmed and the chunk continues to its committed step 2.
    s2 = combat_state(
        request_id="r2",
        energy=2,
        hand=[card("DEFEND")],
        enemies=[{
            "id": "CULTIST",
            "hp": 34,
            "max_hp": 48,
            "block": 0,
            "is_alive": True,
            "intent": "ATTACK",
            "intent_damage": 6,
            "intent_hits": 1,
        }],
        discard_count=1,
    )
    ev = ex.accept_state(s2, acceptance=ActionAcceptance.ACCEPTED)
    assert ev.status == ExecutorStatus.READY_ACTION, ev
    assert ex.index == 1
    ev = ex.prepare_next(s2, validate_action)
    assert ev.prepared.bridge_action == {
        "action": "play",
        "card_index": 0,
        "target_index": -1,
    }


def test_action_acceptance_request_id_correlation():
    """The ACK must correspond to the EXACT handshake. A stale/mismatched
    request_id, a missing field, or a non-bool accepted never yields
    ACCEPTED or REJECTED -- it is UNKNOWN."""
    # Exact match.
    assert resolve_action_acceptance(
        {"request_id": "r7", "accepted": True}, "r7"
    ) is ActionAcceptance.ACCEPTED
    assert resolve_action_acceptance(
        {"request_id": "r7", "accepted": False}, "r7"
    ) is ActionAcceptance.REJECTED
    # Stale ACK from an earlier handshake (action N must not classify N+1).
    assert resolve_action_acceptance(
        {"request_id": "r6", "accepted": True}, "r7"
    ) is ActionAcceptance.UNKNOWN
    # Missing / malformed metadata.
    assert resolve_action_acceptance(None, "r7") is ActionAcceptance.UNKNOWN
    assert resolve_action_acceptance(
        {"accepted": True}, "r7"
    ) is ActionAcceptance.UNKNOWN
    assert resolve_action_acceptance(
        {"request_id": "r7", "accepted": "yes"}, "r7"
    ) is ActionAcceptance.UNKNOWN
    # No expected request id (we never sent a correlated command).
    assert resolve_action_acceptance(
        {"request_id": "r7", "accepted": True}, ""
    ) is ActionAcceptance.UNKNOWN


def test_model_requested_checkpoint():
    s0 = combat_state(
        request_id="r0",
        energy=3,
        hand=[card("STRIKE", target="AnyEnemy")],
    )
    chunk = parse_action_chunk(
        {
            "thought": "Play once, then inspect.",
            "actions": [{
                "kind": "play",
                "card_ref": "h0",
                "target_ref": "e0",
                "checkpoint_after": True,
            }],
        },
        s0,
    )
    ex = ActionChunkExecutor()
    ex.submit(chunk)
    ev = ex.prepare_next(s0, validate_action)
    ex.mark_sent(ev.prepared, s0)
    s1 = combat_state(
        request_id="r1",
        energy=2,
        hand=[],
        enemies=[{
            "id": "CULTIST",
            "hp": 34,
            "max_hp": 48,
            "block": 0,
            "is_alive": True,
            "intent": "ATTACK",
        }],
        discard_count=1,
    )
    ev = ex.accept_state(s1)
    assert ev.status == ExecutorStatus.NEED_MODEL
    assert ev.checkpoint.reason == CheckpointReason.MODEL_REQUESTED


def test_parser_rejects_actions_after_checkpoint():
    s0 = combat_state(
        request_id="r0",
        energy=3,
        hand=[card("STRIKE", target="AnyEnemy"), card("DEFEND")],
    )
    try:
        parse_action_chunk(
            {
                "thought": "bad plan",
                "actions": [
                    {
                        "kind": "play",
                        "card_ref": "h0",
                        "target_ref": "e0",
                        "checkpoint_after": True,
                    },
                    {"kind": "play", "card_ref": "h1"},
                ],
            },
            s0,
        )
    except PlanParseError:
        pass
    else:
        raise AssertionError("parser should reject actions after checkpoint")


def test_delta_render():
    s0 = combat_state(
        request_id="r0",
        energy=3,
        hand=[card("POMMEL", target="AnyEnemy")],
    )
    s1 = combat_state(
        request_id="r1",
        energy=2,
        hand=[card("BASH", target="AnyEnemy", cost=2)],
        discard_count=1,
    )
    d = diff_states(s0, s1)
    assert d.added_cards
    text = render_delta(s0, s1, executed="Pommel", checkpoint_reason="HAND_CHANGED")
    assert "Bash" in text and "HAND_CHANGED" in text


def test_deepseek_payload_and_stream_parser():
    payload = build_chat_payload(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "x"}],
        max_tokens=8192,
        temperature=0.4,
        stream=True,
        capabilities=DEEPSEEK_CAPABILITIES,
        thinking_enabled=True,
        reasoning_effort="high",
    )
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"
    assert "temperature" not in payload  # no effect in thinking mode

    lines = [
        'data: {"choices":[{"delta":{"reasoning_content":"think "}}]}',
        'data: {"choices":[{"delta":{"reasoning_content":"more"}}]}',
        'data: {"choices":[{"delta":{"content":"{\\"actions\\":[]}"}}]}',
        "data: [DONE]",
    ]
    events = parse_sse_data_lines(lines)
    acc = StreamAccumulator()
    for event in events:
        acc.feed_event(event)
    assert acc.reasoning == "think more"
    assert acc.content == '{"actions":[]}'


def test_metrics():
    m = BenchmarkMetrics()
    m.record_model_call(
        latency_ms=1000,
        first_reasoning_ms=200,
        first_content_ms=800,
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
        },
    )
    m.record_plan(3)
    for _ in range(3):
        m.record_game_action()
    s = m.snapshot()
    assert s["actions_per_llm_call"] == 3.0
    assert abs(s["cache_hit_ratio"] - 0.8) < 1e-9


def run_all():
    tests = [
        test_one_call_multiple_actions,
        test_draw_causes_checkpoint,
        test_target_gone_causes_checkpoint,
        test_duplicate_cards_resolve_after_shift,
        test_unchanged_state_with_authoritative_rejection_is_rejected,
        test_unchanged_state_without_authoritative_signal_is_conservative_wait,
        test_unchanged_state_with_bridge_accept_is_awaiting_advance,
        test_action_acceptance_request_id_correlation,
        test_model_requested_checkpoint,
        test_parser_rejects_actions_after_checkpoint,
        test_delta_render,
        test_deepseek_payload_and_stream_parser,
        test_metrics,
    ]
    for fn in tests:
        fn()
        print("PASS", fn.__name__)
    print(f"\nALL {len(tests)} BUNDLE CORE TESTS PASSED")


if __name__ == "__main__":
    run_all()
