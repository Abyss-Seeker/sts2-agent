"""Regression for card reward skip acknowledgement and unchanged UI."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import AgentSession


class CardRewardSkipTests(unittest.TestCase):
    def session(self):
        s = AgentSession()
        s._pending_single_action = {"action": "skip"}
        s._pending_action_request_id = "reward-1"
        s._pending_single_before_state = {
            "type": "card_reward", "request_id": "reward-1",
            "cards": [{"id": "STRIKE", "name": "Strike"}], "can_skip": True,
        }
        return s

    def test_failed_skip_releases_pending_action_on_identical_screen(self):
        s = self.session()
        state = dict(s._pending_single_before_state, request_id="reward-2")
        state["previous_action_result"] = {
            "request_id": "reward-1", "accepted": False,
            "reason": "card_reward_skip_unavailable",
        }
        self.assertFalse(s._reconcile_pending_single_action(state))
        self.assertIsNone(s._pending_single_action)
        self.assertEqual(s._metrics.game_action_rejected_count, 1)

    def test_successful_skip_confirms_return_to_rewards(self):
        s = self.session()
        state = {"type": "reward_screen", "request_id": "reward-2", "options": [],
                 "previous_action_result": {"request_id": "reward-1", "accepted": True}}
        self.assertFalse(s._reconcile_pending_single_action(state))
        self.assertIsNone(s._pending_single_action)
        self.assertEqual(s._metrics.game_action_rejected_count, 0)

    def test_stale_failure_does_not_authorize_another_action(self):
        s = self.session()
        state = dict(s._pending_single_before_state, request_id="reward-2")
        state["previous_action_result"] = {"request_id": "old", "accepted": False}
        self.assertTrue(s._reconcile_pending_single_action(state))
        self.assertIsNotNone(s._pending_single_action)


if __name__ == "__main__":
    unittest.main()
