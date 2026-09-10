"""ActionChunk cognition / prompt / config correctness tests (offline).

Run: python tests/test_chunk_cognition.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import agent as agent_mod  # noqa: E402
from agent import AgentSession, DEFAULT_CONFIG  # noqa: E402
from benchmark_metrics import BenchmarkMetrics  # noqa: E402
from plan_executor import ActionChunkExecutor, ExecutorStatus  # noqa: E402
from action_plan import parse_action_chunk  # noqa: E402
from prompts import DEFAULT_SYSTEM_TEMPLATE, RULEBOOK, RUN_OBJECTIVE, ACTION_CHUNK_CONTRACT  # noqa: E402


# ----------------------------------------------------------------
# P0/P2/P4: chunk ceiling normalization
# ----------------------------------------------------------------

class TestChunkLimit(unittest.TestCase):

    def _limit(self, value):
        s = AgentSession()
        s._config = {"action_chunk_max_actions": value}
        return s._action_chunk_limit()

    def test_chunk_limit_null_falls_back(self):
        self.assertEqual(self._limit(None), 8)

    def test_chunk_limit_invalid_string_falls_back(self):
        self.assertEqual(self._limit("abc"), 8)

    def test_chunk_limit_zero_clamped(self):
        self.assertEqual(self._limit(0), 1)

    def test_chunk_limit_valid_preserved(self):
        self.assertEqual(self._limit(5), 5)

    def test_default_config_is_eight(self):
        self.assertEqual(DEFAULT_CONFIG["action_chunk_max_actions"], 8)


# ----------------------------------------------------------------
# P2/P9: compact prompt policy
# ----------------------------------------------------------------

class TestCompactPrompt(unittest.TestCase):

    def test_compact_prompt_contains_no_runtime_web(self):
        self.assertIn("no runtime access to the web", DEFAULT_SYSTEM_TEMPLATE)
        for banned in ("use external public knowledge",
                       "consult current-build public reference"):
            self.assertNotIn(banned, DEFAULT_SYSTEM_TEMPLATE)

    def test_compact_prompt_allows_prior_model_knowledge(self):
        self.assertIn("already contained in your model",
                      DEFAULT_SYSTEM_TEMPLATE)

    def test_compact_prompt_contains_chunk_cognition_granularity_rule(self):
        self.assertIn(
            "do not stop after the first action merely because the game"
            " executes actions sequentially",
            DEFAULT_SYSTEM_TEMPLATE)
        # §13: no length reward -- one-action chunks stay legitimate.
        self.assertIn("A longer chunk is not inherently better",
                      DEFAULT_SYSTEM_TEMPLATE)

    def test_hidden_run_info_forbidden(self):
        self.assertIn("never use or infer the actual value of hidden"
                      " run-specific information",
                      DEFAULT_SYSTEM_TEMPLATE.lower()
                      if False else DEFAULT_SYSTEM_TEMPLATE)

    def test_prompt_composition_roles_are_separate(self):
        # RULEBOOK = mechanics only; OBJECTIVE = neutral target;
        # CONTRACT = response semantics. System template composes them.
        self.assertIn("{{RULEBOOK}}", DEFAULT_SYSTEM_TEMPLATE)
        self.assertIn("{{OBJECTIVE}}", DEFAULT_SYSTEM_TEMPLATE)
        self.assertIn("{{CONTRACT}}", DEFAULT_SYSTEM_TEMPLATE)
        self.assertIn("# OBJECTIVE", RUN_OBJECTIVE)
        self.assertNotIn("win the run", RULEBOOK)  # no strategy in rulebook


# ----------------------------------------------------------------
# §39 config defaults
# ----------------------------------------------------------------

class TestConfigDefaults(unittest.TestCase):

    def test_history_default_three(self):
        self.assertEqual(DEFAULT_CONFIG.get("max_history_turns"), 3)

    def test_delta_default_false(self):
        self.assertIs(DEFAULT_CONFIG.get("delta_observations"), False)

    def test_strict_failure_default(self):
        self.assertEqual(DEFAULT_CONFIG.get("failure_policy"),
                         "benchmark_strict")

    def test_thinking_default_true(self):
        self.assertIs(DEFAULT_CONFIG.get("thinking_enabled"), True)

    def test_adaptive_entry_high_followup_low(self):
        # Defaults encode the intended cognition split (§7).
        self.assertEqual(
            DEFAULT_CONFIG.get("reasoning_effort_combat_entry"), "high")
        self.assertEqual(
            DEFAULT_CONFIG.get("reasoning_effort_combat_followup"), "low")
        self.assertEqual(
            DEFAULT_CONFIG.get("reasoning_effort_noncombat"), "high")
        self.assertEqual(
            DEFAULT_CONFIG.get("reasoning_effort_retry"), "high")

    def test_config_audit_never_logs_api_key(self):
        s = AgentSession()
        s._config = dict(DEFAULT_CONFIG)
        s._config["api_key"] = "sk-SHOULD-NOT-APPEAR"
        s._logs = []
        s._log_config_audit()
        blob = json.dumps(s._logs, ensure_ascii=False)
        self.assertNotIn("sk-SHOULD-NOT-APPEAR", blob)


# ----------------------------------------------------------------
# P9/§31-33: chunk length distribution (planned, not executed)
# ----------------------------------------------------------------

class TestChunkLengthMetrics(unittest.TestCase):

    def test_chunk_length_metrics(self):
        m = BenchmarkMetrics()
        base = m.checkpoint()
        m.record_plan(1)               # len 1
        m.record_plan(2)               # len 2
        m.record_plan(3)               # len 3
        m.record_plan(5)               # len 4+
        m.record_plan(1, chunk=False)  # single-action mode: not a chunk
        snap = m.snapshot_since(base)
        self.assertEqual(snap["action_chunk_plan_count"], 4)
        self.assertEqual(snap["chunk_len_1"], 1)
        self.assertEqual(snap["chunk_len_2"], 1)
        self.assertEqual(snap["chunk_len_3"], 1)
        self.assertEqual(snap["chunk_len_4_plus"], 1)
        self.assertAlmostEqual(snap["mean_planned_actions_per_chunk"],
                               11 / 4)
        # Session snapshot includes the non-chunk plan in planned total.
        full = m.snapshot()
        self.assertEqual(full["planned_actions_total"], 12)

    def test_planned_vs_executed_metrics(self):
        """§32: plan 4, interrupted after 2 confirmed -> planned dist stays
        4, executed confirmed = 2."""
        m = BenchmarkMetrics()
        m.record_plan(4)
        m.record_action_sent(from_plan=True, combat=True)
        m.record_action_confirmed(from_plan=True, combat=True)
        m.record_action_sent(from_plan=True, combat=True)
        m.record_action_confirmed(from_plan=True, combat=True)
        snap = m.snapshot()
        self.assertEqual(snap["chunk_len_4_plus"], 1)
        self.assertAlmostEqual(snap["mean_planned_actions_per_chunk"], 4.0)
        self.assertEqual(snap["game_action_confirmed_count"], 2)
        self.assertAlmostEqual(
            snap["executed_vs_planned_ratio"], 2 / 4)


# ----------------------------------------------------------------
# §35-37: executor checkpoint discipline
# ----------------------------------------------------------------

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
                 round_=1, discard_count=0):
    return {
        "type": "combat_action", "request_id": request_id,
        "floor": 1, "act": 1, "ascension": 0, "round": round_,
        "player": {"hp": 70, "max_hp": 80, "block": block,
                   "energy": energy, "max_energy": 3, "gold": 99},
        "hand": hand,
        "enemies": enemies if enemies is not None else [cultist()],
        "potions": [],
        "draw_pile_count": 5, "draw_pile": [],
        "discard_pile_count": discard_count, "discard_pile": [],
        "exhaust_pile_count": 0, "exhaust_pile": [],
        "relics": [],
    }


class TestCheckpointDiscipline(unittest.TestCase):

    def _executor_with_plan(self):
        s0 = combat_state("S0", energy=3, hand=[
            card("STRIKE", target="AnyEnemy"),
            card("DEFEND"),
            card("BASH", target="AnyEnemy"),
        ])
        chunk = parse_action_chunk({
            "thought": "determined",
            "actions": [
                {"kind": "play", "card_ref": "h0", "target_ref": "e0"},
                {"kind": "play", "card_ref": "h1"},
                {"kind": "end_turn"},
            ],
        }, s0, max_actions=8)
        ex = ActionChunkExecutor()
        ex.submit(chunk)
        return ex, s0

    def _accept(self, ex, state):
        event = ex.accept_state(state)
        if event.status == ExecutorStatus.TERMINAL:
            return event
        next_event = ex.prepare_next(state, agent_mod.validate_action)
        return event, next_event

    def test_normal_card_removal_does_not_checkpoint(self):
        """§36: played Strike leaves the hand -> plan CONTINUES (h1 Defend
        resolves at its new index), no LLM recall."""
        ex, s0 = self._executor_with_plan()
        ex.mark_sent(ex.prepare_next(s0, agent_mod.validate_action).prepared,
                     s0)
        # After playing Strike: same hand minus index 0, energy 2, enemy
        # hp down, block up -- ALL normal, no new information.
        s1 = combat_state("S1", energy=2, hand=[
            card("DEFEND"), card("BASH", target="AnyEnemy")],
            enemies=[cultist(30)], block=5, discard_count=1)
        event, next_event = self._accept(ex, s1)
        self.assertIsNotNone(next_event)
        self.assertEqual(next_event.status, ExecutorStatus.READY_ACTION,
                         next_event.checkpoint)
        # The remaining ref resolved to the CURRENT index of Defend (0).
        self.assertEqual(next_event.prepared.bridge_action,
                         {"action": "play", "card_index": 0,
                          "target_index": -1})

    def test_enemy_hp_change_does_not_checkpoint(self):
        ex, s0 = self._executor_with_plan()
        ex.mark_sent(ex.prepare_next(s0, agent_mod.validate_action).prepared,
                     s0)
        s1 = combat_state("S1", energy=2, hand=[
            card("DEFEND"), card("BASH", target="AnyEnemy")],
            enemies=[cultist(25)], discard_count=1)
        event, next_event = self._accept(ex, s1)
        self.assertEqual(next_event.status, ExecutorStatus.READY_ACTION)

    def test_energy_change_does_not_checkpoint(self):
        ex, s0 = self._executor_with_plan()
        ex.mark_sent(ex.prepare_next(s0, agent_mod.validate_action).prepared,
                     s0)
        s1 = combat_state("S1", energy=1, hand=[
            card("DEFEND"), card("BASH", target="AnyEnemy")],
            discard_count=1)
        event, next_event = self._accept(ex, s1)
        self.assertEqual(next_event.status, ExecutorStatus.READY_ACTION)

    def test_new_hand_information_does_checkpoint(self):
        """§37: a DRAWN card is new decision-relevant information."""
        ex, s0 = self._executor_with_plan()
        ex.mark_sent(ex.prepare_next(s0, agent_mod.validate_action).prepared,
                     s0)
        s1 = combat_state("S1", energy=2, hand=[
            card("DEFEND"), card("BASH", target="AnyEnemy"),
            card("POMMEL_STRIKE", target="AnyEnemy")],
            discard_count=1)
        event = ex.accept_state(s1)
        reasons = [event.checkpoint]
        if event.checkpoint is None or event.checkpoint.reason is None:
            next_event = ex.prepare_next(s1, agent_mod.validate_action)
            reasons = [next_event.checkpoint]
        self.assertTrue(
            any(cp is not None and cp.reason is not None
                and "HAND_CHANGED" in str(cp.reason.value)
                for cp in reasons if cp is not None),
            f"expected HAND_CHANGED, got {reasons}")

    def test_block_change_does_not_checkpoint(self):
        """§36: gaining block (a normal combat-state advance) must not cause
        a checkpoint -- the plan continues exactly like enemy-hp/energy do."""
        ex, s0 = self._executor_with_plan()
        ex.mark_sent(ex.prepare_next(s0, agent_mod.validate_action).prepared,
                     s0)
        # Block went 0 -> 12 (e.g. a Defend resolved before this observe),
        # but it is a plain authoritative-state change, not new information.
        s1 = combat_state("S1", energy=2, hand=[
            card("DEFEND"), card("BASH", target="AnyEnemy")],
            enemies=[cultist(30)], block=12, discard_count=1)
        event, next_event = self._accept(ex, s1)
        self.assertIsNotNone(next_event)
        self.assertEqual(next_event.status, ExecutorStatus.READY_ACTION,
                         next_event.checkpoint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
