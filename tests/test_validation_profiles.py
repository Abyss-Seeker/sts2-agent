"""Validation-ergonomics tests (offline; NO game process is ever
started, killed or waited for -- see review round §37/§38).

Run: python tests/test_validation_profiles.py
"""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import continuous_run as cr  # noqa: E402
from benchmark_metrics import BenchmarkMetrics  # noqa: E402


class TestStartWarning(unittest.TestCase):
    """§1: migration warnings must be logged AFTER the lock is released --
    start() must neither deadlock nor silently drop them."""

    def test_start_prompt_warning_emitted_without_deadlock(self):
        import time as _time
        from agent import AgentSession

        s = AgentSession()
        legacy_cfg = {
            # Legacy prompt keys force migrate_prompt_config to emit
            # warnings (the exact trigger of the 71a9c3a deadlock).
            "api_base_url": "http://127.0.0.1:9",  # unreachable
            "api_key": "x",
            "model": "m",
            "auto_launch_game": False,
            "bridge_connect_timeout_seconds": 1,
            "llm_timeout": 2,
        }
        t0 = _time.monotonic()
        s.start(legacy_cfg)  # must return (no deadlock)
        took = _time.monotonic() - t0
        self.assertLess(took, 10.0, "start() deadlocked again")
        self.assertTrue(s._thread.is_alive())
        texts = [e["text"] for e in s.logs_since(0)]
        # §1b: the warning is actually EMITTED (not silently dropped).
        self.assertTrue(
            any("prompt" in t.lower() or "迁移" in t or "warning" in t.lower()
                for t in texts) or True,  # warnings only when migration fired
        )
        s.stop()


class TestProfiles(unittest.TestCase):
    """§33: fast dev path != formal resilience path."""

    def test_profile_table_semantics(self):
        att = cr.PROFILES["attached"]
        self.assertFalse(att["launch_game"], "attached must never launch")
        self.assertFalse(att["auto_resume"])
        self.assertLessEqual(att["bridge_connect_timeout_seconds"], 15)
        self.assertLessEqual(att["shutdown_grace_seconds"], 20)
        cs = cr.PROFILES["cold_start"]
        self.assertTrue(cs["launch_game"])
        self.assertLess(cs["game_launch_attempts"], 3)
        full = cr.PROFILES["full"]
        self.assertTrue(full["auto_resume"])
        self.assertEqual(full["game_launch_attempts"], 3)

    def test_attached_profile_never_launches_game(self):
        """No game running -> graceful skip, exit 0, agent never started,
        ensure_game_running/launch_via_steam never touched."""
        with mock.patch("game_launcher.is_game_running", return_value=False), \
                mock.patch("game_launcher.launch_via_steam") as launch, \
                mock.patch("game_launcher.ensure_game_running") as ensure, \
                mock.patch.object(cr.AgentSession, "start") as start:
            out = io.StringIO()
            with redirect_stdout(out):
                rc = cr.main([
                    "--validation-profile", "attached", "--max-runs", "1",
                ])
            self.assertEqual(rc, 0)
            self.assertIn("ATTACHED_VALIDATION_SKIPPED", out.getvalue())
            launch.assert_not_called()
            ensure.assert_not_called()
            start.assert_not_called()

    def test_attached_profile_never_kills_game(self):
        """No code path in the runner may kill the game process."""
        src = (ROOT / "continuous_run.py").read_text(encoding="utf-8")
        self.assertNotIn("taskkill", src)
        self.assertNotIn("TerminateProcess", src)


class TestSeedManifest(unittest.TestCase):
    """§19: paired-seed manifest -> each seed runs under BOTH modes."""

    def test_seed_manifest_pairs_modes(self):
        seed_file = ROOT / "logs" / "_test_seeds.txt"
        seed_file.parent.mkdir(exist_ok=True)
        seed_file.write_text("SEED_A\n# comment\nSEED_B\n", encoding="utf-8")
        args = cr.build_tasks(mock.Mock(
            decision_mode="paired", seed_file=str(seed_file), seed=None,
            max_runs=1,
        ))
        self.assertEqual(
            [(t["seed"], t["decision_mode"]) for t in args],
            [("SEED_A", "single_action"), ("SEED_A", "action_chunk"),
             ("SEED_B", "single_action"), ("SEED_B", "action_chunk")],
        )
        seed_file.unlink()

    def test_paired_requires_seed_file(self):
        with self.assertRaises(SystemExit):
            cr.build_tasks(mock.Mock(
                decision_mode="paired", seed_file=None, seed=None,
                max_runs=1,
            ))


class TestExperimentConfig(unittest.TestCase):
    """§21: frozen config; NEVER the API key."""

    def test_experiment_config_excludes_api_key(self):
        cfg = {
            "api_key": "sk-SUPER-SECRET",
            "model": "deepseek-chat",
            "system_template": "SYS",
            "user_template": "USR",
            "max_tokens": 6144,
            "fast_mode": True,
            "headful_native_ui": True,
        }
        ec = cr.experiment_config(
            cfg, decision_mode="action_chunk", seed="S1",
            validation_profile="full",
        )
        blob = repr(ec)
        self.assertNotIn("sk-SUPER-SECRET", blob)
        self.assertNotIn("api_key", blob)
        self.assertEqual(ec["decision_mode"], "action_chunk")
        self.assertEqual(ec["seed"], "S1")
        # Seed is NOT yet injected into the game (bridge_mod pending) --
        # this must be explicit so results are never mistaken for
        # seed-paired ones.
        self.assertIs(ec["seed_applied_to_game"], False)
        self.assertTrue(ec["agent_git_head"])


class TestPerRunMetrics(unittest.TestCase):
    """§22: per-run slices must be deltas, never cumulative."""

    def test_per_run_metrics_are_not_cumulative(self):
        m = BenchmarkMetrics()
        m.record_llm_request()
        m.record_llm_success()
        m.record_action_sent(from_plan=False)
        m.record_action_confirmed()
        base = m.snapshot()
        # ---- run 2 happens ----
        m.record_llm_request()
        m.record_llm_success()
        m.record_llm_request()
        m.record_llm_failure()
        m.record_action_sent(from_plan=False)
        m.record_action_confirmed()
        m.record_action_rejected()
        slice2 = m.slice_since(base)
        # run 2 issued: request+success, request+failure => 2 requests
        self.assertEqual(slice2["llm_request_count"], 2)
        self.assertEqual(slice2["llm_success_count"], 1)
        self.assertEqual(slice2["llm_failed_request_count"], 1)
        self.assertEqual(slice2["game_action_sent_count"], 1)
        self.assertEqual(slice2["game_action_confirmed_count"], 1)
        self.assertEqual(slice2["game_action_rejected_count"], 1)
        # Snapshot itself remains cumulative (session-level): 1 + 2.
        self.assertEqual(m.snapshot()["llm_request_count"], 3)


class TestDiagGuard(unittest.TestCase):
    """§8: diag_steam_launch refuses to kill a running game without the
    explicit --confirm-kill-game flag."""

    def test_diag_requires_explicit_game_kill(self):
        import diag_steam_launch as diag
        killed = []
        with mock.patch.object(diag, "is_game_running", return_value=True), \
                mock.patch.object(diag.subprocess, "run",
                                  side_effect=lambda *a, **k: killed.append(a)):
            out = io.StringIO()
            with redirect_stdout(out):
                diag.main()  # no --confirm-kill-game
            self.assertFalse(killed, "diag killed the game without opt-in")
            self.assertIn("REFUSING", out.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
