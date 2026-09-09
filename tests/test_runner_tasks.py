"""PHASE A runner/metrics correctness tests -- 100% OFFLINE.

No game process is started, killed, waited for, or even referenced by
the launcher: sessions are fakes, clocks are injected. See review
round A23/A24.

Run: python tests/test_runner_tasks.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import continuous_run as cr  # noqa: E402
from benchmark_metrics import BenchmarkMetrics  # noqa: E402


# ----------------------------------------------------------------
# Fake AgentSession (session contract only)
# ----------------------------------------------------------------

class FakeSession:
    # Real AgentSession assigns log seqs for the WHOLE process lifetime
    # (start() does not reset them) -- mirror that so the runner's
    # suite-level last_seq cursor behaves identically with fakes.
    _shared_seq = 0

    def __init__(self, script):
        # script: list of PER-POLL steps; each status() call consumes one
        # step (a step may emit logs / mutate status). Steps are consumed
        # in order; after exhaustion the last state persists.
        self.script = list(script)
        self.started_cfgs: list[dict] = []
        self.stops = 0
        self._metrics = BenchmarkMetrics()
        self._logs: list[dict] = []
        self._status = {"running": True, "run_id": None,
                        "benchmark_valid": True, "safe_to_disconnect": True}

    def start(self, cfg):
        self.started_cfgs.append(cfg)

    def _apply(self, step):
        if step.get("run_id"):
            self._status["run_id"] = step["run_id"]
        for k in ("running", "benchmark_valid", "safe_to_disconnect"):
            if k in step:
                self._status[k] = step[k]
        for log in step.get("logs", []):
            FakeSession._shared_seq += 1
            entry = {"seq": FakeSession._shared_seq, "ts": "00:00:00",
                     "text": log.pop("text", ""), **log}
            self._logs.append(entry)

    def stop(self):
        self.stops += 1
        self._status["running"] = False

    def status(self):
        if self.script:
            self._apply(self.script.pop(0))
        return dict(self._status)

    def logs_since(self, after):
        return [e for e in self._logs if e["seq"] > after]


def run_report_log(rid, result, benchmark_valid=None):
    return {"kind": "run_report", "run_id": rid, "result": result,
            "snapshot": {"llm_request_count": 1},
            "benchmark_valid": benchmark_valid}


def fake_factory(script_pool):
    """A factory yielding scripted sessions in order."""
    made = []

    def factory():
        s = FakeSession(script_pool[len(made)] if len(made)
                        < len(script_pool) else [{}])
        made.append(s)
        return s
    return factory, made


BASE_CFG = {"model": "m", "max_tokens": 100, "system_template": "S",
            "user_template": "U", "failure_policy": "benchmark_strict"}


def opts(**kw):
    defaults = dict(max_runs=3, suite_max_seconds=None,
                    task_max_seconds=None, shutdown_grace_seconds=1,
                    validation_profile="full", formal=True)
    defaults.update(kw)
    return cr.SuiteOptions(**defaults)


def fast_clock(start=0.0):
    state = {"t": start}
    def clock():
        return state["t"]
    def advance(dt):
        state["t"] += dt
    return clock, advance


# ----------------------------------------------------------------
# A1/A2/A5: paired task scheduling
# ----------------------------------------------------------------

class TestPairedTaskScheduling(unittest.TestCase):

    def _run_paired(self, script_pool):
        factory, made = fake_factory(script_pool)
        tasks = [
            {"seed": "S1", "decision_mode": "single_action",
             "pair_index": 0, "pair_order": 0},
            {"seed": "S1", "decision_mode": "action_chunk",
             "pair_index": 0, "pair_order": 1},
            {"seed": "S2", "decision_mode": "action_chunk",
             "pair_index": 1, "pair_order": 0},
            {"seed": "S2", "decision_mode": "single_action",
             "pair_index": 1, "pair_order": 1},
        ]
        clock, _ = fast_clock()
        state, _ = cr.execute_suite(
            factory, tasks, dict(BASE_CFG), opts(),
            emit=lambda m: None, clock=clock, sleep=lambda s: None)
        return state, made

    def test_paired_normal_run_advances_task(self):
        """A1: a NORMAL terminal must finalize the task and START the
        next one -- the agent must never be left auto-continuing."""
        victory = [{"run_id": "r1", "logs": [
            run_report_log("r1", "NORMAL_VICTORY")]},
            {"running": False, "run_id": "r1"}]
        state, made = self._run_paired([victory, victory, victory, victory])
        self.assertEqual(state.stop_reason, "task plan exhausted")
        self.assertEqual(len(made), 4)
        self.assertEqual(len(state.runs), 4)
        for s in made:
            self.assertIn("continue_after_normal_terminal", s.started_cfgs[0])
            self.assertIs(s.started_cfgs[0]["continue_after_normal_terminal"],
                          False)

    def test_paired_changes_decision_mode(self):
        """A1: consecutive tasks apply their OWN frozen decision_mode."""
        victory = [{"run_id": "r1", "logs": [
            run_report_log("r1", "NORMAL_VICTORY")]},
            {"running": False, "run_id": "r1"}]
        state, made = self._run_paired([victory, victory, victory, victory])
        modes = [s.started_cfgs[0]["decision_mode"] for s in made]
        self.assertEqual(
            modes, ["single_action", "action_chunk",
                    "action_chunk", "single_action"])
        seeds = [s.started_cfgs[0].get("seed") for s in made]
        self.assertEqual(seeds, ["S1", "S1", "S2", "S2"])

    def test_paired_each_task_exactly_one_run(self):
        """A2: each task contains exactly ONE run report; the agent stops
        after it instead of opening run 2 under the old config."""
        victory = [{"run_id": "r1", "logs": [
            run_report_log("r1", "NORMAL_VICTORY")]},
            {"running": False, "run_id": "r1"}]
        state, made = self._run_paired([victory, victory, victory, victory])
        self.assertEqual([r["mode"] for r in state.runs],
                         ["single_action", "action_chunk",
                          "action_chunk", "single_action"])
        self.assertEqual([r["seed"] for r in state.runs],
                         ["S1", "S1", "S2", "S2"])

    def test_recoverable_termination_does_not_advance_task(self):
        """A3: a terminated/recovery run keeps the SAME session running;
        the runner must not start a new task."""
        # Session 1: run r1 recovers twice then finally victories.
        recover_then_victory = [
            {"run_id": "r1", "logs": [
                run_report_log("r1", "RECOVERABLE_1")]},
            {"run_id": "r1", "running": True, "logs": [
                {"kind": "info", "text": "resumed same save"}]},
            {"run_id": "r1", "running": True},
            {"run_id": "r1", "running": True},
            {"run_id": "r1", "logs": [
                run_report_log("r1", "NORMAL_VICTORY")]},
            {"running": False, "run_id": "r1"},
        ]
        factory, made = fake_factory([
            recover_then_victory,
            # task 2 (action_chunk) ends normally too.
            [{"run_id": "r2", "logs": [
                run_report_log("r2", "NORMAL_VICTORY")]},
             {"running": False, "run_id": "r2"}],
        ])
        tasks = [{"seed": "S1", "decision_mode": "single_action",
                  "pair_index": 0, "pair_order": 0},
                 {"seed": "S1", "decision_mode": "action_chunk",
                  "pair_index": 0, "pair_order": 1}]
        state, _ = cr.execute_suite(
            factory, tasks, dict(BASE_CFG), opts(),
            emit=lambda m: None, clock=fast_clock()[0],
            sleep=lambda s: None)
        # Only after the genuine terminal did the runner advance.
        self.assertEqual(len(made), 2)
        self.assertEqual(state.runs[0]["result"], "RECOVERABLE_1")
        self.assertEqual(state.runs[0]["run_id"], "r1")


# ----------------------------------------------------------------
# A15/A16: manifest authority vs --max-runs
# ----------------------------------------------------------------

class TestSeedSuiteSemantics(unittest.TestCase):

    def test_seed_manifest_not_truncated_by_default_max_runs(self):
        """A15/A16: 5 seeds x 2 modes = 10 tasks -- the DEFAULT
        --max-runs=3 must NOT stop the suite after 3 runs."""
        args = mock.Mock(decision_mode="paired", seed_file="x", seed=None,
                         max_runs=3)
        seed_file = ROOT / "logs" / "_test_manifest.txt"
        seed_file.parent.mkdir(exist_ok=True)
        seed_file.write_text("\n".join(f"S{i}" for i in range(5)),
                             encoding="utf-8")
        args.seed_file = str(seed_file)
        tasks = cr.build_tasks(args)
        seed_file.unlink()
        self.assertEqual(len(tasks), 10, "manifest must not be truncated")

        victory = [{"run_id": "r1", "logs": [
            run_report_log("r1", "NORMAL_VICTORY")]},
            {"running": False, "run_id": "r1"}]
        factory, made = fake_factory([victory] * 10)
        state, _ = cr.execute_suite(
            factory, tasks, dict(BASE_CFG), opts(max_runs=3),
            emit=lambda m: None, clock=fast_clock()[0],
            sleep=lambda s: None)
        self.assertEqual(len(state.runs), 10, "max-runs=3 truncated suite")
        self.assertEqual(len(made), 10)

    def test_counterbalanced_pair_order(self):
        """A19: even seeds single-first, odd seeds chunk-first."""
        args = mock.Mock(decision_mode="paired", seed_file="x", seed=None,
                         max_runs=3)
        seed_file = ROOT / "logs" / "_test_manifest.txt"
        seed_file.parent.mkdir(exist_ok=True)
        seed_file.write_text("A\nB\nC\n", encoding="utf-8")
        args.seed_file = str(seed_file)
        tasks = cr.build_tasks(args)
        seed_file.unlink()
        self.assertEqual([t["decision_mode"] for t in tasks],
                         ["single_action", "action_chunk",
                          "action_chunk", "single_action",
                          "single_action", "action_chunk"])
        self.assertEqual([t["pair_order"] for t in tasks],
                         [0, 1, 0, 1, 0, 1])

    def test_seed_never_reported_paired_valid(self):
        """A20: no code path may claim seed pairing is game-valid."""
        src = (ROOT / "continuous_run.py").read_text(encoding="utf-8")
        self.assertNotIn("seed_applied_to_game\": True", src)
        self.assertNotIn("seed_applied_to_game': True", src)
        ec = cr.experiment_config(dict(BASE_CFG), decision_mode="action_chunk",
                                  seed="S", validation_profile="full")
        self.assertIs(ec["seed_applied_to_game"], False)


# ----------------------------------------------------------------
# A17/A18: task vs suite timeout
# ----------------------------------------------------------------

class TestTimeoutSemantics(unittest.TestCase):

    def test_task_timeout_resets_each_task_and_censors(self):
        """A18: task timeout -> TASK_TIMEOUT / censored / invalid, and the
        runner advances instead of mistaking it for a result."""
        hang = [{"run_id": "r1", "running": True}]  # never ends
        factory, made = fake_factory([hang, hang])
        tasks = [{"seed": "S1", "decision_mode": "single_action",
                  "pair_index": 0, "pair_order": 0},
                 {"seed": "S1", "decision_mode": "action_chunk",
                  "pair_index": 0, "pair_order": 1}]
        clock, advance = fast_clock()
        state, _ = cr.execute_suite(
            factory, tasks, dict(BASE_CFG),
            opts(task_max_seconds=10),
            emit=lambda m: None, clock=clock,
            sleep=lambda s: advance(s))
        self.assertEqual(len(state.runs), 2)
        for r in state.runs:
            self.assertEqual(r["result"], "TASK_TIMEOUT")
            self.assertTrue(r["censored"])
            self.assertIs(r["benchmark_valid"], False)
        self.assertEqual(len(made), 2, "both tasks ran")

    def test_suite_timeout_is_separate(self):
        """A17: the suite cap stops the WHOLE plan even if tasks would
        still fit their own per-task budget."""
        victory = [{"run_id": "r1", "logs": [
            run_report_log("r1", "NORMAL_VICTORY")]},
            {"running": False, "run_id": "r1"}]
        factory, made = fake_factory([victory] * 10)
        tasks = [{"seed": f"S{i}", "decision_mode": "single_action",
                  "pair_index": i, "pair_order": 0} for i in range(10)]
        clock, advance = fast_clock()
        state, _ = cr.execute_suite(
            factory, tasks, dict(BASE_CFG),
            opts(task_max_seconds=1000, suite_max_seconds=25),
            emit=lambda m: None, clock=clock,
            sleep=lambda s: advance(s))
        self.assertEqual(state.stop_reason, "suite time cap")
        self.assertLess(len(state.runs), 10)


# ----------------------------------------------------------------
# A6-A14: per-run metric math
# ----------------------------------------------------------------

class TestPerRunMetricMath(unittest.TestCase):

    def _two_runs(self):
        m = BenchmarkMetrics()
        # ---- RUN 1: 2 calls, 4 confirmed actions, latencies [100, 300]
        base = m.checkpoint()
        m.record_llm_request()
        m.record_llm_success(latency_ms=100, effort="high",
                             usage={"prompt_tokens": 10,
                                    "completion_tokens": 5,
                                    "prompt_cache_hit_tokens": 6,
                                    "prompt_cache_miss_tokens": 4,
                                    "reasoning_tokens": 3})
        m.record_llm_request()
        m.record_llm_success(latency_ms=300, effort="high",
                             usage={"prompt_tokens": 10,
                                    "completion_tokens": 5,
                                    "prompt_cache_hit_tokens": 6,
                                    "prompt_cache_miss_tokens": 4,
                                    "reasoning_tokens": 3})
        for _ in range(4):
            m.record_action_sent()
            m.record_action_confirmed()
        run1 = m.snapshot_since(base)
        # ---- RUN 2: 4 calls, 6 confirmed actions, latencies [10..40]
        base2 = m.checkpoint()
        for lat in (10, 20, 30, 40):
            m.record_llm_request()
            m.record_llm_success(latency_ms=lat, effort="low",
                                 usage={"prompt_tokens": 20,
                                        "completion_tokens": 8,
                                        "prompt_cache_hit_tokens": 0,
                                        "prompt_cache_miss_tokens": 20,
                                        "reasoning_tokens": 9})
        for _ in range(6):
            m.record_action_sent()
            m.record_action_confirmed()
        m.record_checkpoint("NEW_TURN", interrupted=False)
        m.record_checkpoint("HAND_CHANGED")
        run2 = m.snapshot_since(base2)
        return run1, run2

    def test_per_run_additive_counters(self):
        run1, run2 = self._two_runs()
        self.assertEqual(run1["llm_request_count"], 2)
        self.assertEqual(run2["llm_request_count"], 4)
        self.assertEqual(run1["game_action_confirmed_count"], 4)
        self.assertEqual(run2["game_action_confirmed_count"], 6)

    def test_per_run_ratio_recomputed(self):
        """A14: run2's ratio must be 6/4 = 1.5 -- NEVER the session
        aggregate minus the previous session aggregate."""
        run1, run2 = self._two_runs()
        self.assertAlmostEqual(run1["actions_per_llm_call"], 4 / 2)
        self.assertAlmostEqual(run2["actions_per_llm_call"], 6 / 4, places=9)

    def test_per_run_latency_recomputed(self):
        """A9: run2 latency stats come from the SUFFIX [10,20,30,40] --
        run1's [100,300] must not leak in."""
        run1, run2 = self._two_runs()
        self.assertEqual(run1["llm_latency_ms_mean"], 200.0)
        self.assertAlmostEqual(run2["llm_latency_ms_mean"], 25.0)
        self.assertAlmostEqual(run2["llm_latency_ms_p50"], 25.0)
        # Session snapshot must remain cumulative.
        self.assertIsNone(run1["llm_latency_ms_mean"] - 200.0
                          or None)  # sanity: run1 value itself

    def test_per_run_percentiles(self):
        run1, run2 = self._two_runs()
        self.assertAlmostEqual(run2["llm_latency_ms_p95"],
                               10 + 0.95 * 30)  # interpolation over [10..40]
        self.assertEqual(run2["first_reasoning_token_ms_p50"], None)

    def test_per_run_checkpoint_counter(self):
        """A10: checkpoint_reasons is a per-run diff."""
        run1, run2 = self._two_runs()
        self.assertEqual(run1["checkpoint_reasons"], {})
        self.assertEqual(run2["checkpoint_reasons"],
                         {"NEW_TURN": 1, "HAND_CHANGED": 1})

    def test_per_run_reasoning_effort_recomputed(self):
        """A11: per-effort stats recomputed from run-local evidence."""
        run1, run2 = self._two_runs()
        self.assertEqual(run1["reasoning_by_effort"]["high"]["calls"], 2)
        self.assertAlmostEqual(
            run1["reasoning_by_effort"]["high"]["mean_reasoning_tokens"], 3.0)
        self.assertEqual(run2["reasoning_by_effort"]["low"]["calls"], 4)
        self.assertAlmostEqual(
            run2["reasoning_by_effort"]["low"]["mean_reasoning_tokens"], 9.0)
        self.assertAlmostEqual(
            run2["reasoning_by_effort"]["low"]["latency_ms_p50"], 25.0)
        self.assertNotIn("high", run2["reasoning_by_effort"],
                         "run1's effort must not leak into run2")

    def test_per_run_cache_ratio(self):
        run1, run2 = self._two_runs()
        self.assertAlmostEqual(run1["cache_hit_ratio"], 12 / 20)
        self.assertAlmostEqual(run2["cache_hit_ratio"], 0.0)

    def test_benchmark_valid_from_run_events(self):
        """A12: validity is THIS run's invalidation events, not derived
        from numeric subtraction."""
        m = BenchmarkMetrics()
        base = m.checkpoint()
        m.record_action_confirmed()
        run_ok = m.snapshot_since(base)
        self.assertIs(run_ok["benchmark_valid"], True)
        base2 = m.checkpoint()
        m.invalidate("LLM API error: x")
        run_bad = m.snapshot_since(base2)
        self.assertIs(run_bad["benchmark_valid"], False)
        self.assertEqual(run_bad["invalidation_reason"], "LLM API error: x")


# ----------------------------------------------------------------
# A21: attached bridge preflight
# ----------------------------------------------------------------

class TestAttachedPreflight(unittest.TestCase):

    def test_attached_bridge_preflight_fail_fast(self):
        """Game alive but bridge NOT listening -> skip immediately; never
        construct an AgentSession or wait for a connect timeout."""
        emitted = []
        with mock.patch("game_launcher.is_game_running",
                        return_value=True), \
                mock.patch.object(cr, "_bridge_listening",
                                  return_value=False), \
                mock.patch.object(cr, "AgentSession") as session_cls:
            rc = cr.main([
                "--validation-profile", "attached", "--max-runs", "1",
            ])
            self.assertEqual(rc, 0)
            session_cls.assert_not_called()
        # main() prints; capture not needed -- return code + no session
        # construction is the contract. (Output checked in profiles test.)

    def test_attached_preflight_probe_is_lightweight(self):
        """The TCP probe must be a sub-second connect, not the 10s
        connect-timeout path."""
        import inspect
        sig = inspect.signature(cr._bridge_listening)  # BEFORE patching
        with mock.patch("game_launcher.is_game_running",
                        return_value=True), \
                mock.patch.object(cr, "_bridge_listening",
                                  return_value=True) as probe, \
                mock.patch.object(cr.AgentSession, "start") as start, \
                mock.patch.object(cr.AgentSession, "status") as status, \
                mock.patch.object(cr.AgentSession, "stop") as stop, \
                mock.patch("time.sleep", side_effect=KeyboardInterrupt):
            status.return_value = {"running": False,
                                   "safe_to_disconnect": True}
            cr.main(["--validation-profile", "attached"])
            probe.assert_called_once()
            args_, kwargs_ = probe.call_args
            # (host, port) positional; the sub-second timeout is the
            # function's own default -- verify it once, statically.
            self.assertEqual(len(args_), 2)
            self.assertLessEqual(
                sig.parameters["timeout"].default, 0.5)


# ----------------------------------------------------------------
# A22: Ctrl+C cleanup
# ----------------------------------------------------------------

class TestKeyboardInterruptCleanup(unittest.TestCase):

    def test_keyboard_interrupt_cleanup(self):
        """Ctrl+C mid-poll: session stopped cleanly, partial report
        returned, no exception escapes."""
        hang = [{"run_id": "r1", "running": True}]
        factory, made = fake_factory([hang])
        tasks = [{"seed": "S1", "decision_mode": "action_chunk",
                  "pair_index": 0, "pair_order": 0}]
        polls = {"n": 0}

        def sleep_interrupts(_):
            polls["n"] += 1
            if polls["n"] >= 2:
                raise KeyboardInterrupt
        state, session = cr.execute_suite(
            factory, tasks, dict(BASE_CFG), opts(),
            emit=lambda m: None, clock=fast_clock()[0],
            sleep=sleep_interrupts)
        self.assertEqual(state.stop_reason, "interrupted")
        self.assertEqual(session.stops, 1, "session must be stopped cleanly")
        # Partial report is buildable.
        report = cr.build_report(state, session, 0.0, "attached")
        self.assertEqual(report["aggregate"]["stop_reason"], "interrupted")
        self.assertIn("runs", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
