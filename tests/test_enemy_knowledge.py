"""Behavior reference coverage, information boundaries and request integration."""
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import AgentSession, DEFAULT_CONFIG
from enemy_knowledge import REFERENCE_PATH, _reference, enemy_behavior_context


def board(enemy_id="CEREMONIAL_BEAST"):
    return {
        "type": "combat_action", "request_id": "knowledge-test", "round": 1,
        "player": {"hp": 70, "max_hp": 80, "energy": 3, "max_energy": 3},
        "hand": [], "potions": [],
        "enemies": [{"id": enemy_id, "is_alive": True, "hp": 252,
                     "max_hp": 252, "intent": "BUFF"}],
    }


class EnemyKnowledgeTests(unittest.TestCase):
    def test_default_on_and_disabled_without_loading(self):
        self.assertIs(DEFAULT_CONFIG["enemy_behavior_knowledge"], True)
        with patch("enemy_knowledge._reference", side_effect=AssertionError):
            self.assertEqual(enemy_behavior_context(board(), False), "")

    def test_no_hidden_instance_fields_or_other_enemies(self):
        state = board()
        expected = enemy_behavior_context(state)
        state["enemies"][0].update(intent_move_id="SECRET_MOVE", ai_state="SECRET_AI")
        state["seed"] = "SECRET_SEED"
        self.assertEqual(expected, enemy_behavior_context(state))
        self.assertNotIn("ENEMY TYPE: FLYCONID", expected)
        self.assertIn("HP threshold, NOT Block", expected)
        self.assertIn("loses ALL Strength", expected)
        self.assertIn("already started any card play", expected)
        self.assertIn("target.CurrentHp <= base.Amount", expected)

    def test_cooldown_is_not_probability_weight(self):
        text = enemy_behavior_context(board("FLYCONID"))
        self.assertIn("Frail Spores 50%, Smash 50%", text)
        self.assertIn("3-move cooldown", text)
        self.assertIn("AddBranch(moveState, 3, MoveRepeatType.CannotRepeat)", text)
        self.assertNotIn("FrailSpores 2, Smash 1", text)

    def test_selection_dead_duplicates_and_unknown(self):
        state = board()
        state["enemies"].append(dict(state["enemies"][0]))
        text = enemy_behavior_context({"combat_context": state})
        self.assertEqual(text.count("ENEMY TYPE: CEREMONIAL_BEAST"), 1)
        for enemy in state["enemies"]:
            enemy["is_alive"] = False
        self.assertEqual(enemy_behavior_context(state), "")
        self.assertEqual(enemy_behavior_context({"type": "map"}), "")
        self.assertIn("unavailable", enemy_behavior_context(board("NEW_PATCH_MONSTER")))

    def test_artifact_has_source_provenance_and_all_monsters(self):
        data = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(data["monsters"]), 121)
        self.assertTrue(data["source_sha256"])
        self.assertIn("GetStateWeight", data["random_branch_semantics"])
        self.assertIn("GOOP_MOVE", data["monsters"]["LEAF_SLIME_S"]["mechanics"])
        self.assertIn("Slimed", data["monsters"]["LEAF_SLIME_S"]["mechanics"])

    def test_missing_reference_is_explicit(self):
        _reference.cache_clear()
        try:
            with patch("enemy_knowledge.REFERENCE_PATH", Path("missing-reference.json")):
                with self.assertLogs("enemy_knowledge", level="WARNING"):
                    self.assertIn("unavailable", enemy_behavior_context(board()))
        finally:
            _reference.cache_clear()

    def test_both_request_paths_honor_switch(self):
        class Captured(Exception):
            pass
        for enabled in (True, False):
            for chunk in (True, False):
                with self.subTest(enabled=enabled, chunk=chunk):
                    session = AgentSession()
                    session._config["enemy_behavior_knowledge"] = enabled
                    session._config["delta_observations"] = True
                    session._config["user_template"] = "CUSTOM {{STATE}}"
                    llm = Mock()
                    captured = []
                    def capture(*args, **kwargs):
                        self.assertEqual(kwargs["user_template"], "CUSTOM {{STATE}}")
                        captured.extend(args)
                        raise Captured()
                    session._ctx.build_messages = capture
                    with self.assertRaises(Captured):
                        if chunk:
                            session._request_combat_chunk(
                                board(), llm, "SYSTEM", time.monotonic() + 60, 30)
                        else:
                            session._handle_state(board(), llm, "SYSTEM", 60, 30)
                    self.assertEqual("ENEMY BEHAVIOR REFERENCE" in captured[0], enabled)
                    self.assertIn("ENEMY[0]", captured[2])


if __name__ == "__main__":
    unittest.main()
