"""Native selection lifecycle tests (offline, no game).

Covers the Headbutt-class P0: a gameplay command may synchronously await a
native modal selection. The selection must be serviced independently, must
preempt the underlying combat state, must never be classified as
ACTION_REJECTED, must discard the old ActionChunk remainder, and repeated
identical rejections must raise PROTOCOL_STALL instead of looping.

It also covers the accepted-but-not-yet-advanced (AWAITING_ADVANCE) waiting
state: an accepted command whose snapshot has not moved must RETAIN its
in-flight action/plan (no reset, no advance) and reconcile the SAME command
against a later authoritative state.

Run: python tests/test_selection_lifecycle.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import agent as agent_mod  # noqa: E402
from agent import (  # noqa: E402
    AgentSession, PROTOCOL_STALL_REPEATS, SELECTION_SCREEN_TYPES,
)
from benchmark_metrics import BenchmarkMetrics  # noqa: E402
from checkpoint import ActionAcceptance  # noqa: E402


def card(cid, *, target="Self", playable=True, cost=1):
    return {
        "id": cid, "display_name": cid.title(), "cost": cost,
        "current_energy_cost": cost,
        "type": "Attack" if target == "AnyEnemy" else "Skill",
        "target": target, "playable": playable,
    }


def cultist(hp=40):
    return {"id": "CULTIST", "hp": hp, "max_hp": 48, "block": 0,
            "is_alive": True, "intent": "ATTACK", "intent_damage": 6,
            "intent_hits": 1}


def combat_state(request_id, *, energy, hand, enemies=None, block=0,
                 round_=1):
    return {
        "type": "combat_action", "request_id": request_id,
        "floor": 1, "act": 1, "ascension": 0, "round": round_,
        "player": {"hp": 70, "max_hp": 80, "block": block,
                   "energy": energy, "max_energy": 3, "gold": 99},
        "hand": hand,
        "enemies": enemies if enemies is not None else [cultist()],
        "potions": [],
        "draw_pile_count": 5, "draw_pile": [],
        "discard_pile_count": 0, "discard_pile": [],
        "exhaust_pile_count": 0, "exhaust_pile": [],
        "relics": [],
    }


def selection_state(request_id, candidates=3):
    return {
        "type": "card_select", "request_id": request_id,
        "selection_kind": "hand_select",
        "min_select": 1, "max_select": 1,
        "options": [
            {"index": i, "id": f"CARD{i}", "display_name": f"Card{i}",
             "cost": 1, "current_energy_cost": 1, "type": "Attack",
             "target": "AnyEnemy", "playable": True}
            for i in range(candidates)
        ],
    }


class TestSelectionClassification(unittest.TestCase):
    """§9/§11/§27: selection is a real decision screen, not a rejection."""

    def test_card_select_is_selection_screen(self):
        self.assertIn("card_select", SELECTION_SCREEN_TYPES)

    def test_selection_is_not_action_rejected(self):
        """§11: the action INITIATED a modal; confirmed, never rejected."""
        s = AgentSession()
        s._metrics = BenchmarkMetrics()
        s._logs = []
        s._plan_executor.reset()
        s._pending_single_action = {"action": "play", "card_index": 0,
                                    "target_index": 0}
        s._pending_single_before_state = combat_state(
            "B", energy=3, hand=[card("HEADBUTT", target="AnyEnemy")])
        s._reconcile_pending_single_action(selection_state("S1"))
        self.assertEqual(s._metrics.game_action_confirmed_count, 1)
        self.assertEqual(s._metrics.game_action_rejected_count, 0)
        self.assertEqual(s._last_checkpoint_reason, "SELECTION_REQUIRED")

    def test_correlated_rejection_outranks_selection_shape(self):
        s = AgentSession()
        s._metrics = BenchmarkMetrics()
        s._logs = []
        s._pending_single_action = {"action": "play", "card_index": 0}
        s._pending_single_before_state = combat_state(
            "B", energy=3, hand=[card("HEADBUTT", target="AnyEnemy")])
        s._pending_action_request_id = "B"
        rejected = selection_state("S1")
        rejected["previous_action_result"] = {
            "request_id": "B", "accepted": False, "reason": "refused"}
        s._reconcile_pending_single_action(rejected)
        self.assertEqual(s._metrics.game_action_confirmed_count, 0)
        self.assertEqual(s._metrics.game_action_rejected_count, 1)

    def test_combat_selection_uses_followup_reasoning(self):
        class Caps:
            provider = "deepseek"
            supports_reasoning_effort = True

        class LLM:
            caps = Caps()
            reasoning_effort = None
            last_reasoning = ""
            last_usage = {}
            first_reasoning_ms = None
            first_content_ms = None

            def chat(self, messages):
                return '{"thought":"upgrade the attack","action":"choose","index":0}'

        s = AgentSession()
        s._metrics = BenchmarkMetrics()
        s._logs = []
        s._send_single_action = lambda act, state: "sent"
        state = selection_state("SEL")
        state["combat_context"] = {
            "in_combat": True,
            "player": {"hp": 70, "energy": 1},
            "hand": [],
            "enemies": [],
        }
        llm = LLM()
        s._llm = llm
        s._handle_state(state, llm, "SYS", 10.0, 5.0)
        self.assertEqual(s._reasoning_context_class, "combat_followup")
        self.assertEqual(llm.reasoning_effort, "low")
        self.assertEqual(s._metrics.combat_logical_inspection_count, 1)
        self.assertEqual(s._metrics.combat_llm_success_count, 1)


class TestAwaitingAdvanceLifecycle(unittest.TestCase):
    """P0: accepted-but-not-yet-observably-advanced is a WAITING state.

    The in-flight action/plan must be retained, and the no-advance stall guard
    must key on the ACTUAL unresolved action (not ``_pending_single_action``,
    which is None in ActionChunk mode).
    """

    def _executor_with_plan(self, s0):
        from action_plan import parse_action_chunk
        from plan_executor import ActionChunkExecutor

        chunk = parse_action_chunk({
            "thought": "headbutt then end",
            "actions": [
                {"kind": "play", "card_ref": "h0", "target_ref": "e0"},
                {"kind": "end_turn"},
            ],
        }, s0, max_actions=8)
        ex = ActionChunkExecutor()
        ex.submit(chunk)
        ev = ex.prepare_next(s0, agent_mod.validate_action)
        ex.mark_sent(ev.prepared, s0)
        return ex

    def test_awaiting_advance_retains_inflight_then_selection_preempts(self):
        from plan_executor import ExecutorStatus

        s0 = combat_state("S0", energy=3, hand=[
            card("HEADBUTT", target="AnyEnemy"),
            card("STRIKE", target="AnyEnemy")])
        ex = self._executor_with_plan(s0)
        # Accepted, unchanged combat snapshot -> WAITING_ADVANCE (retained).
        event = ex.accept_state(
            combat_state("S1", energy=3, hand=[
                card("HEADBUTT", target="AnyEnemy"),
                card("STRIKE", target="AnyEnemy")]),
            acceptance=ActionAcceptance.ACCEPTED,
        )
        self.assertEqual(event.status, ExecutorStatus.WAITING_ADVANCE)
        self.assertIsNotNone(ex.inflight)
        self.assertTrue(ex.has_pending_plan)
        self.assertEqual(ex.index, 0)

        # The native selection then arrives: it resolves the SAME in-flight
        # action as a screen change and discards the stale remainder.
        event = ex.accept_state(
            selection_state("SEL"), acceptance=ActionAcceptance.ACCEPTED)
        self.assertEqual(event.status, ExecutorStatus.NEED_MODEL)
        self.assertIn(event.checkpoint.reason.value,
                      ("SCREEN_CHANGED", "SELECTION_REQUIRED"))
        self.assertFalse(ex.has_pending_plan)

    def test_no_advance_key_uses_actual_inflight_action(self):
        import dataclasses

        s = AgentSession()
        s._metrics = BenchmarkMetrics()
        s._logs = []
        s._stop = agent_mod.threading.Event()
        s0 = combat_state("S0", energy=3, hand=[
            card("HEADBUTT", target="AnyEnemy"),
            card("STRIKE", target="AnyEnemy")])
        ex = self._executor_with_plan(s0)
        s._plan_executor = ex
        state = combat_state("S1", energy=3, hand=[
            card("HEADBUTT", target="AnyEnemy"),
            card("STRIKE", target="AnyEnemy")])

        s._on_no_advance(state)
        self.assertEqual(s._no_advance_streak, 1)
        first_key = s._no_advance_key
        self.assertNotIn("None", first_key)

        # SAME visible fingerprint, DIFFERENT unresolved action -> the streak
        # must restart (the key uses the real in-flight bridge action).
        ex._inflight = dataclasses.replace(
            ex.inflight, bridge_action={"action": "end_turn"})
        s._on_no_advance(state)
        self.assertNotEqual(s._no_advance_key, first_key)
        self.assertEqual(s._no_advance_streak, 1)


class TestProtocolStallGuard(unittest.TestCase):
    """§25/§26: repeated identical rejection -> controlled failure."""

    def test_same_action_same_state_rejection_guard(self):
        s = AgentSession()
        s._metrics = BenchmarkMetrics()
        s._logs = []
        s._stop = agent_mod.threading.Event()
        state = combat_state("X", energy=3, hand=[card("STRIKE",
                                                       target="AnyEnemy")])
        s._pending_single_action = {"action": "end_turn"}
        for i in range(PROTOCOL_STALL_REPEATS):
            s._on_action_rejected(state)
        self.assertFalse(s._metrics.benchmark_valid)
        self.assertTrue(s._stop.is_set())
        texts = [e["text"] for e in s._logs]
        self.assertTrue(any("PROTOCOL_STALL" in t for t in texts), texts)

    def test_different_state_resets_guard(self):
        s = AgentSession()
        s._metrics = BenchmarkMetrics()
        s._logs = []
        s._stop = agent_mod.threading.Event()
        s._on_action_rejected(combat_state("A", energy=3,
                                           hand=[card("STRIKE",
                                                      target="AnyEnemy")]))
        self.assertEqual(s._reject_streak, 1)
        # A different authoritative state starts a new streak.
        s._on_action_rejected(combat_state("B", energy=2,
                                           hand=[card("DEFEND")]))
        self.assertTrue(s._metrics.benchmark_valid)


class TestSelectionPreemptsCombat(unittest.TestCase):
    """§28: topmost player-actionable modal owns the decision. The
    coordinator's suspend/resume contract is enforced C#-side; here we
    assert the Python-side invariant: a card_select state must never be
    routed as a combat_action decision."""

    def test_selection_state_type_is_not_combat(self):
        st = selection_state("S1")
        self.assertNotEqual(st["type"], "combat_action")
        self.assertEqual(st["type"], "card_select")

    def test_selection_interrupts_chunk_and_discards_remainder(self):
        """§10/§15: a selection after step 3 hard-checkpoints the chunk and
        discards steps 4-5 (never resume stale D/E)."""
        from plan_executor import ActionChunkExecutor, ExecutorStatus

        s0 = combat_state("S0", energy=3, hand=[
            card("SETUP"), card("ATTACK", target="AnyEnemy"),
            card("HEADBUTT", target="AnyEnemy"),
            card("STRIKE", target="AnyEnemy")])
        chunk = json.loads(json.dumps({
            "thought": "big turn",
            "actions": [
                {"kind": "play", "card_ref": "h0"},
                {"kind": "play", "card_ref": "h1", "target_ref": "e0"},
                {"kind": "play", "card_ref": "h2", "target_ref": "e0"},
                {"kind": "play", "card_ref": "h3", "target_ref": "e0"},
                {"kind": "end_turn"},
            ],
        }))
        from action_plan import parse_action_chunk
        parsed_chunk = parse_action_chunk(chunk, s0, max_actions=8)
        ex = ActionChunkExecutor()
        ex.submit(parsed_chunk)
        # Step 1 (SETUP): prepare -> send -> confirmed by next state.
        ev = ex.prepare_next(s0, agent_mod.validate_action)
        self.assertEqual(ev.status, ExecutorStatus.READY_ACTION, ev)
        ex.mark_sent(ev.prepared, s0)
        s1 = combat_state("P1", energy=2, hand=[
            card("ATTACK", target="AnyEnemy"),
            card("HEADBUTT", target="AnyEnemy"),
            card("STRIKE", target="AnyEnemy")])
        ex.accept_state(s1)
        # Step 2 (ATTACK, ref h0 -> now index 0).
        ev = ex.prepare_next(s1, agent_mod.validate_action)
        self.assertEqual(ev.status, ExecutorStatus.READY_ACTION, ev)
        ex.mark_sent(ev.prepared, s1)
        s2 = combat_state("P2", energy=1, hand=[
            card("HEADBUTT", target="AnyEnemy"),
            card("STRIKE", target="AnyEnemy")])
        ex.accept_state(s2)
        # Step 3 (HEADBUTT, ref h0 -> now index 0).
        ev = ex.prepare_next(s2, agent_mod.validate_action)
        self.assertEqual(ev.status, ExecutorStatus.READY_ACTION, ev)
        ex.mark_sent(ev.prepared, s2)
        # Headbutt sent, then the NATIVE SELECTION appears.
        after_headbutt = selection_state("SEL1")
        event = ex.accept_state(after_headbutt)
        # Not a rejection: the plan is interrupted at a decision boundary
        # (screen changed / awaiting selection) and the remainder is gone.
        cp = event.checkpoint
        reason = cp.reason.value if cp is not None else ""
        self.assertNotEqual(reason, "ACTION_REJECTED", reason)
        self.assertIn(reason, ("SCREEN_CHANGED", "SELECTION_REQUIRED",
                               "MODEL_REQUESTED", "NEXT_ACTION_ILLEGAL"),
                      reason)
        self.assertIsNone(event.prepared,
                          "stale remainder (Strike/EndTurn) must not run")


if __name__ == "__main__":
    unittest.main(verbosity=2)
