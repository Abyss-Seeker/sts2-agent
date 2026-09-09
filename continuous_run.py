"""Continuous benchmark runner for the STS2 LLM agent.

Validation profiles (fast dev path != formal resilience path):

  attached    -- requires an ALREADY RUNNING game with a LISTENING bridge;
                 never launches, never kills, no auto-resume, short
                 connect/shutdown deadlines. Skips gracefully (exit 0)
                 when game/bridge is unavailable.
  cold_start  -- launches the game via Steam (short launch budget) for
                 targeted launcher/bridge verification. Never kills.
  full        -- formal behavior: auto-launch, auto-resume, long
                 deadlines. Use for gates / long smokes / final A/B.

NEVER kills the game in any profile (kill is only the explicit,
opt-in ``diag_steam_launch.py --confirm-kill-game``).

Formal paired tasks (``--seed-file``): ONE task = exactly ONE genuine
run. The agent runs with ``continue_after_normal_terminal=False``: after
victory/defeat it emits its run_report and STOPS at the safe terminal
boundary; the runner then applies the NEXT task's frozen config
(seed/decision_mode) in a fresh AgentSession. Recoverable terminations
(``result=terminated``) are SAME-RUN recovery inside the agent and never
advance the task (A3).

Seed pairing order is counterbalanced (A19): even-indexed seeds run
single_action first, odd-indexed seeds run action_chunk first; each task
records pair_index/pair_order. Seeds are metadata only until bridge_mod
supports explicit run-start-with-seed (``seed_applied_to_game`` stays
false -- never reported as paired-seed-valid, A20).

Timeout semantics (A17/A18): ``--task-max-seconds`` bounds ONE task and
marks it ``TASK_TIMEOUT`` / ``censored=true`` / ``benchmark_valid=false``
(never silently a victory/defeat); ``--suite-max-seconds`` is an optional
global emergency cap over the whole suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import AgentSession
from server import load_config

STATUS_KEYS = (
    "running", "bridge_connected", "current_state_type", "floor", "act",
    "hp", "agent_phase", "benchmark_valid", "invalidation_reason",
    "run_id", "safe_to_disconnect",
    "logical_inspection_count", "llm_request_count", "llm_success_count",
    "llm_failed_request_count", "game_action_sent_count",
    "game_action_confirmed_count", "game_action_rejected_count",
    "game_action_unconfirmable_count", "strategic_plan_count",
    "plan_completed_count", "plan_interrupted_count",
    "actions_per_llm_call", "combat_llm_request_count",
    "combat_game_action_confirmed_count", "combat_actions_per_llm_call",
    "combat_turn_count", "recoverable_termination_count",
    "bridge_reconnect_count", "game_relaunch_count",
    "safe_recovery_count", "transport_interrupted_action_count",
    "prompt_tokens", "completion_tokens", "reasoning_tokens",
    "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
)

LOG_EXTRA_KEYS = (
    "plan_id", "step_index", "action", "result", "reason",
    "executed_steps", "remaining_steps_discarded", "plan_completed",
    "llm_ms", "actions", "state_type",
)

NORMAL_RESULTS = ("NORMAL_VICTORY", "NORMAL_DEFEAT")
CONSECUTIVE_INVALID_TASK_LIMIT = 2

# §33: fast dev path != formal resilience path.
PROFILES: dict[str, dict] = {
    "attached": {
        "launch_game": False,
        "auto_resume": False,
        "bridge_connect_timeout_seconds": 10.0,
        "shutdown_grace_seconds": 15,
        "game_launch_attempts": 1,
        "game_process_timeout": 45.0,
        "suite_max_seconds": 300,
        "task_max_seconds": 300,
    },
    "cold_start": {
        "launch_game": True,
        "auto_resume": False,
        "bridge_connect_timeout_seconds": 120.0,
        "shutdown_grace_seconds": 30,
        "game_launch_attempts": 1,
        "game_process_timeout": 45.0,
        "suite_max_seconds": 900,
        "task_max_seconds": 900,
    },
    "full": {
        "launch_game": True,
        "auto_resume": True,
        "bridge_connect_timeout_seconds": 300.0,
        "shutdown_grace_seconds": 90,
        "game_launch_attempts": 3,
        "game_process_timeout": 180.0,
        "suite_max_seconds": 3600,
        "task_max_seconds": 1800,
    },
}


def snapshot_runs(session: AgentSession) -> dict[str, dict]:
    """Per-run metric slices keyed by run_id.

    The agent emits a ``run_report`` log at every terminal with a
    ``snapshot`` that is a PER-RUN slice (derived values recomputed from
    run-local evidence), never the cumulative session counter.
    """
    out: dict[str, dict] = {}
    for e in session.logs_since(0):
        if e.get("kind") != "run_report":
            continue
        rid = e.get("run_id") or "?"
        out[rid] = e.get("snapshot") or {}
    return out


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=5, cwd=str(Path(__file__).resolve().parent),
        ).stdout.strip()
    except Exception:
        return "unknown"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def experiment_config(cfg: dict, *, decision_mode: str, seed: str | None,
                      validation_profile: str) -> dict:
    """Frozen per-task experiment description (§21). NEVER contains the
    API key."""
    reasoning = {
        k: cfg.get(k)
        for k in (
            "reasoning_policy", "reasoning_effort_fixed",
            "reasoning_effort_combat_entry",
            "reasoning_effort_combat_followup",
            "reasoning_effort_noncombat", "reasoning_effort_retry",
        )
    }
    return {
        "agent_git_head": _git_head(),
        "decision_mode": decision_mode,
        "seed": seed,
        # The bridge protocol cannot inject a seed into the game yet;
        # a paired-seed A/B becomes valid only once bridge_mod supports
        # explicit run-start-with-seed. Recorded so results can never be
        # silently mistaken for seed-paired ones (A20).
        "seed_applied_to_game": False,
        "validation_profile": validation_profile,
        "model": cfg.get("model"),
        "provider_profile": cfg.get("provider_profile"),
        "system_prompt_sha256": _sha256(str(cfg.get("system_template", ""))),
        "user_prompt_sha256": _sha256(str(cfg.get("user_template", ""))),
        "reasoning": reasoning,
        "max_history_turns": cfg.get("max_history_turns"),
        "max_context_chars": cfg.get("max_context_chars"),
        "max_tokens": cfg.get("max_tokens"),
        "temperature": cfg.get("temperature"),
        "failure_policy": cfg.get("failure_policy"),
        "headful_native_ui": cfg.get("headful_native_ui"),
        "fast_mode": cfg.get("fast_mode"),
    }


def build_tasks(args) -> list[dict]:
    """Execution plan.

    - ``--seed-file``: formal paired mode; the manifest is the AUTHORITY
      (A15/A16) -- ``--max-runs`` never truncates it. Same seed runs
      under BOTH decision modes with a counterbalanced order (A19).
    - legacy (no seed file): a SINGLE task; the agent auto-continues
      runs and ``--max-runs`` bounds the collected run count.
    """
    if args.decision_mode == "paired" and not args.seed_file:
        raise SystemExit("--decision-mode paired requires --seed-file")

    if args.seed_file:
        seeds = [
            s.strip() for s in
            Path(args.seed_file).read_text(encoding="utf-8").splitlines()
            if s.strip() and not s.strip().startswith("#")
        ]
        if not seeds:
            raise SystemExit(f"seed file {args.seed_file} contains no seeds")
        tasks: list[dict] = []
        for i, seed in enumerate(seeds):
            # A19 counterbalance: alternating order per seed index.
            order = (["single_action", "action_chunk"] if i % 2 == 0
                     else ["action_chunk", "single_action"])
            for j, mode in enumerate(order):
                tasks.append({
                    "seed": seed,
                    "decision_mode": mode,
                    "pair_index": i,
                    "pair_order": j,
                })
        return tasks

    return [{"seed": getattr(args, "seed", None),
             "decision_mode": args.decision_mode,
             "pair_index": None, "pair_order": None}]


@dataclass
class SuiteOptions:
    max_runs: int = 3                 # legacy run-count cap only
    suite_max_seconds: float | None = None
    task_max_seconds: float | None = None
    shutdown_grace_seconds: int = 90
    validation_profile: str = "full"
    formal: bool = False              # seed-file mode: runner owns tasks


@dataclass
class SuiteState:
    runs: list[dict] = field(default_factory=list)
    experiment_configs: list[dict] = field(default_factory=list)
    invalid_run_ids: set = field(default_factory=set)
    stop_reason: str | None = None


def _bridge_listening(host: str, port: int, timeout: float = 0.4) -> bool:
    """Lightweight TCP preflight (A21): is bridge_mod listening?"""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def execute_suite(
    session_factory,
    tasks: list[dict],
    base_cfg: dict,
    opts: SuiteOptions,
    *,
    emit=print,
    clock=time.monotonic,
    sleep=time.sleep,
    poll_interval: float = 2.0,
) -> tuple[SuiteState, AgentSession | None]:
    """Run the task plan against injected sessions (testable offline).

    Session contract: ``start(cfg)``, ``stop()``, ``status()``,
    ``logs_since(seq)`` -- the real AgentSession or a fake in tests.
    Raises nothing for KeyboardInterrupt: it stops the current session
    cleanly, leaves any game process alone, and returns the partial
    report (A22).
    """
    state = SuiteState()
    session: AgentSession | None = None
    suite_t0 = clock()
    last_heartbeat = 0.0
    last_seq = 0
    consecutive_invalid_tasks = 0
    task_idx = 0
    suite_timed_out = False

    try:
        while state.stop_reason is None:
            # Suite-wide emergency cap (A17) applies in both modes.
            if (opts.suite_max_seconds is not None
                    and clock() - suite_t0 >= opts.suite_max_seconds):
                state.stop_reason = "suite time cap"
                break

            # ---- formal mode: run ONE task, then advance (A1/A2) ----
            if opts.formal:
                if task_idx >= len(tasks):
                    state.stop_reason = "task plan exhausted"
                    break
                task = tasks[task_idx]
                task_cfg = {
                    **base_cfg,
                    "decision_mode": task["decision_mode"],
                }
                if task.get("seed"):
                    task_cfg["seed"] = task["seed"]
                # A2/A4: formal tasks own their run boundary.
                task_cfg["continue_after_normal_terminal"] = False
                state.experiment_configs.append(experiment_config(
                    task_cfg, decision_mode=task["decision_mode"],
                    seed=task.get("seed"),
                    validation_profile=opts.validation_profile,
                ))
                emit(f"TASK {task_idx + 1}/{len(tasks)}:"
                     f" mode={task['decision_mode']}"
                     f" seed={task.get('seed') or '(none)'}"
                     f" pair_index={task.get('pair_index')}"
                     f" pair_order={task.get('pair_order')}")
                session = session_factory()
                session.start(task_cfg)
                task_t0 = clock()

            task = tasks[0] if not opts.formal else task
            if not opts.formal:
                # Legacy: ONE session that auto-continues runs itself;
                # --max-runs bounds the collected run count.
                session = session_factory()
                session.start({**base_cfg,
                               "decision_mode": task["decision_mode"]})
                task_t0 = clock()
            outcome: str | None = None  # formal task outcome
            while outcome is None and state.stop_reason is None:
                sleep(poll_interval)
                st = session.status()
                rid = st.get("run_id")

                # Drain logs: forward to stdout, collect run reports and
                # invalidation events (dedup by run id).
                for e in session.logs_since(last_seq):
                    last_seq = max(last_seq, e["seq"])
                    line = f"[{e['ts']}] {e['kind']}: {e['text']}"
                    extras = " | ".join(
                        f"{k}={e[k]}" for k in LOG_EXTRA_KEYS
                        if e.get(k) not in (None, "")
                    )
                    emit((line + (f" | {extras}" if extras else ""))[:600])
                    if e.get("kind") == "run_report":
                        state.runs.append({
                            "run_id": e.get("run_id") or "?",
                            "mode": task["decision_mode"],
                            "seed": task.get("seed"),
                            "pair_index": task.get("pair_index"),
                            "pair_order": task.get("pair_order"),
                            "result": e.get("result"),
                            "ended_at": e.get("ts"),
                            "benchmark_valid": None,  # stamped below
                            "censored": False,
                            "metrics": e.get("snapshot") or {},
                        })
                        emit(f"RUN REPORT: {e.get('run_id')}"
                             f" result={e.get('result')}")
                    elif e.get("kind") == "benchmark_invalidated":
                        state.invalid_run_ids.add(
                            e.get("run_id") or rid or "?")
                        emit(f"BENCHMARK INVALID"
                             f" (run {e.get('run_id') or '?'}): {e['text']}")

                # Stamp benchmark validity onto runs not yet stamped: the
                # agent invalidates the SESSION, so every run emitted
                # before the invalidation belongs to the invalid session.
                valid_now = st.get("benchmark_valid") is not False
                for r in state.runs:
                    if r["benchmark_valid"] is None:
                        r["benchmark_valid"] = valid_now
                        if not valid_now and r["result"] not in NORMAL_RESULTS:
                            r["censored"] = True

                elapsed = clock() - suite_t0
                if elapsed - last_heartbeat >= 30.0:
                    last_heartbeat = elapsed
                    emit("HEARTBEAT t=%.0fs " % elapsed
                         + " ".join(f"{k}={st.get(k)}"
                                    for k in STATUS_KEYS))

                if opts.formal:
                    # ---- formal task termination (A2/A3/A18) ----
                    if not st["running"]:
                        current = [r for r in state.runs
                                   if r["run_id"] == rid]
                        result = current[-1]["result"] if current else None
                        if result in NORMAL_RESULTS:
                            outcome = "completed"          # A1/A2
                        elif st.get("benchmark_valid") is False:
                            outcome = "benchmark_invalid"  # advance
                        else:
                            outcome = "fatal"              # stop suite
                    elif (opts.task_max_seconds is not None
                            and clock() - task_t0 >= opts.task_max_seconds):
                        # A18: a timeout is NOT a result -- censor it. The
                        # run never terminated, so synthesize its record.
                        emit(f"TASK TIMEOUT after {opts.task_max_seconds}s"
                             " -- marking censored.")
                        session.stop()
                        state.runs.append({
                            "run_id": rid or "?",
                            "mode": task["decision_mode"],
                            "seed": task.get("seed"),
                            "pair_index": task.get("pair_index"),
                            "pair_order": task.get("pair_order"),
                            "result": "TASK_TIMEOUT",
                            "ended_at": None,
                            "benchmark_valid": False,
                            "censored": True,
                            "metrics": {},
                        })
                        outcome = "timeout"
                else:
                    # ---- legacy: agent continues runs itself ----
                    if len(state.runs) >= opts.max_runs:
                        state.stop_reason = (
                            f"max runs reached ({opts.max_runs})")
                    elif (opts.task_max_seconds is not None
                            and clock() - task_t0
                            >= opts.task_max_seconds):
                        state.stop_reason = "task time cap"
                    elif not st["running"]:
                        state.stop_reason = (
                            "agent stopped with benchmark_valid=false"
                            if st.get("benchmark_valid") is False
                            else "agent stopped")

            if state.stop_reason is not None:
                break
            if outcome == "fatal":
                state.stop_reason = "agent stopped fatally"
                break
            if outcome == "benchmark_invalid":
                consecutive_invalid_tasks += 1
                if consecutive_invalid_tasks > CONSECUTIVE_INVALID_TASK_LIMIT:
                    state.stop_reason = (
                        "too many consecutive benchmark-invalid tasks")
                    break
            else:
                consecutive_invalid_tasks = 0
            if outcome == "timeout":
                rid = session.status().get("run_id")
                for r in state.runs:
                    if r["run_id"] == rid:
                        r["censored"] = True
                        r["benchmark_valid"] = False
                        r["result"] = "TASK_TIMEOUT"
            # Formal: advance to the NEXT task with its frozen config (the
            # agent deliberately did NOT auto-start another run). Legacy:
            # single task -- the agent continues runs itself.
            if opts.formal:
                task_idx += 1
    except KeyboardInterrupt:
        # A22: clean interrupt -- stop the session (short join), leave any
        # game process alive, and still produce the partial report.
        emit("KEYBOARD INTERRUPT: stopping the agent cleanly (the game"
             " process is left running).")
        state.stop_reason = "interrupted"
    finally:
        if session is not None:
            try:
                if session.status().get("running"):
                    deadline = clock() + max(0, opts.shutdown_grace_seconds)
                    while clock() < deadline:
                        if session.status().get("safe_to_disconnect"):
                            break
                        sleep(1.0)
                    if not session.status().get("safe_to_disconnect"):
                        emit("WARNING: agent never reported"
                             " safe_to_disconnect; forcing stop.")
                    session.stop()
            except KeyboardInterrupt:
                # Interrupt DURING the grace wait: stop immediately.
                try:
                    session.stop()
                except Exception:
                    pass
                emit("INTERRUPT: forced immediate stop during grace wait.")
    return state, session


def build_report(state: SuiteState, session, t0: float,
                 validation_profile: str) -> dict:
    metrics = (session._metrics.snapshot()
               if session is not None else {})
    return {
        "validation_profile": validation_profile,
        "final_session_metrics": metrics,
        "runs": state.runs,                    # §23: per-run slices
        "aggregate": {                         # §23: session-level totals
            "runs_finalized": len(state.runs),
            "benchmark_invalid_runs": len(state.invalid_run_ids),
            "stop_reason": state.stop_reason,
            "wall_clock_seconds": round(time.time() - t0, 1),
        },
        "experiment_configs": state.experiment_configs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="STS2 continuous runner.")
    parser.add_argument("--max-runs", type=int, default=3,
                        help="legacy run-count cap (no --seed-file only); "
                             "ignored for seed suites, whose task list is "
                             "the authority")
    parser.add_argument("--task-max-seconds", type=int, default=None,
                        help="per-task wall-clock cap; a timeout is "
                             "recorded as TASK_TIMEOUT/censored, never a "
                             "result (default: per profile)")
    parser.add_argument("--suite-max-seconds", type=int, default=None,
                        help="optional global emergency cap over the whole "
                             "suite (default: per profile)")
    parser.add_argument("--shutdown-grace-seconds", type=int, default=None,
                        help="stop reached -> wait up to N seconds for "
                             "safe_to_disconnect (default: per profile)")
    parser.add_argument("--bridge-connect-timeout", type=float, default=None,
                        help="override the profile's bridge connect deadline")
    parser.add_argument("--decision-mode", default="action_chunk",
                        choices=["single_action", "action_chunk", "paired"],
                        help="'paired' is only valid with --seed-file")
    parser.add_argument("--seed", default=None,
                        help="single seed (recorded in the experiment "
                             "config; game-side injection pending)")
    parser.add_argument("--seed-file", default=None,
                        help="paired-seed manifest: one seed per line; "
                             "each seed runs under BOTH decision modes")
    parser.add_argument("--validation-profile", default="full",
                        choices=sorted(PROFILES))
    args = parser.parse_args(argv)

    profile = dict(PROFILES[args.validation_profile])
    if args.decision_mode == "paired" and not args.seed_file:
        parser.error("--decision-mode paired requires --seed-file")

    connect_timeout = (
        args.bridge_connect_timeout
        if args.bridge_connect_timeout is not None
        else float(profile["bridge_connect_timeout_seconds"])
    )
    task_max = (
        args.task_max_seconds if args.task_max_seconds is not None
        else profile["task_max_seconds"]
    )
    suite_max = (
        args.suite_max_seconds if args.suite_max_seconds is not None
        else profile["suite_max_seconds"]
    )
    shutdown_grace = (
        args.shutdown_grace_seconds
        if args.shutdown_grace_seconds is not None
        else int(profile["shutdown_grace_seconds"])
    )

    tasks = build_tasks(args)
    formal = bool(args.seed_file)
    print(f"CONTINUOUS RUNNER: profile={args.validation_profile}"
          f" formal={formal} tasks={len(tasks) if formal else 'legacy'}"
          f" task_max={task_max}s suite_max={suite_max}s"
          f" shutdown_grace={shutdown_grace}s"
          f" connect_timeout={connect_timeout}s"
          f" mode={args.decision_mode}", flush=True)

    cfg = load_config()
    cfg.update({
        "failure_policy": "benchmark_strict",
        "delta_observations": False,
        "stream_mode": "off",
        "llm_retries": 0,
        "provider_profile": "auto",
        "thinking_enabled": True,
        "save_log": True,
        # NOTE (§26): reasoning policy/efforts are NOT overridden here --
        # the frozen cognition config comes from config.json as-is.
        "auto_launch_game": bool(profile["launch_game"]),
        "auto_resume": bool(profile["auto_resume"]),
        "bridge_connect_timeout_seconds": connect_timeout,
        "game_launch_attempts": int(profile["game_launch_attempts"]),
        "game_process_timeout": float(profile["game_process_timeout"]),
    })

    # §6/§21: attached NEVER launches the game and never waits for one.
    # Preflight = process AND a light TCP probe of the bridge port; a
    # running game with a dead bridge is a SKIP, not a connect timeout.
    if not profile["launch_game"]:
        from game_launcher import is_game_running
        game_alive = is_game_running()
        bridge_ok = _bridge_listening(
            cfg.get("bridge_host", "127.0.0.1"),
            int(cfg.get("bridge_port", 9002)),
        )
        if not game_alive:
            print("ATTACHED_VALIDATION_SKIPPED: game not running"
                  " (attached profile never launches the game).", flush=True)
            return 0
        if not bridge_ok:
            print("ATTACHED_VALIDATION_SKIPPED: game exists but bridge"
                  " unavailable (mod not listening).", flush=True)
            return 0

    t0 = time.time()
    opts = SuiteOptions(
        max_runs=args.max_runs,
        suite_max_seconds=suite_max,
        task_max_seconds=task_max,
        shutdown_grace_seconds=shutdown_grace,
        validation_profile=args.validation_profile,
        formal=formal,
    )
    try:
        state, session = execute_suite(
            AgentSession, tasks, cfg, opts,
            emit=lambda m: print(m, flush=True),
        )
    except KeyboardInterrupt:  # belt & braces: never orphan the session
        print("KEYBOARD INTERRUPT (outer): partial run, game left alive.",
              flush=True)
        return 130

    report = build_report(state, session, t0, args.validation_profile)
    out = Path(__file__).resolve().parent / "logs"
    out.mkdir(exist_ok=True)
    path = out / f"continuous_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print("\n===== FINAL SESSION METRICS =====", flush=True)
    print(json.dumps(report["final_session_metrics"], ensure_ascii=False,
                     indent=2), flush=True)
    print(f"REPORT WRITTEN: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
