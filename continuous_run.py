"""Continuous benchmark runner for the STS2 LLM agent.

Runs the agent continuously across runs:

  - recoverable termination (result=terminated): the agent resumes the
    SAME save with its run context preserved (built into AgentSession);
    the runner just keeps going.
  - victory / defeat: the runner finalizes the run (snapshots its
    PER-RUN metric slice under the run_id) and continues to the next run.
  - benchmark-invalid: the run is marked invalid in the report; in a
    formal benchmark the runner starts the next configured seed instead
    of stopping the whole session (bounded by CONSECUTIVE invalid runs,
    not a per-session transient counter).

Validation profiles (fast dev path != formal resilience path):

  attached    -- requires an ALREADY RUNNING game; never launches, never
                 kills, no auto-resume, short connect/shutdown deadlines.
                 Skips gracefully (exit 0) when no game is available.
  cold_start  -- launches the game via Steam (short launch budget) for
                 targeted launcher/bridge verification. Never kills.
  full        -- formal behavior: auto-launch, auto-resume, long
                 deadlines. Use for gates / long smokes / final A/B.

NEVER kills the game in any profile (kill is only the explicit,
opt-in ``diag_steam_launch.py --confirm-kill-game``).

Usage:
    python -u continuous_run.py --validation-profile attached
    python -u continuous_run.py --max-runs 3 --max-seconds 3600
    python -u continuous_run.py --seed-file seeds.txt   # paired A/B plan
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
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

# §33: fast dev path != formal resilience path.
PROFILES: dict[str, dict] = {
    "attached": {
        "launch_game": False,
        "auto_resume": False,
        "bridge_connect_timeout_seconds": 10.0,
        "shutdown_grace_seconds": 15,
        "game_launch_attempts": 1,
        "game_process_timeout": 45.0,
        "max_seconds": 90,
    },
    "cold_start": {
        "launch_game": True,
        "auto_resume": False,
        "bridge_connect_timeout_seconds": 120.0,
        "shutdown_grace_seconds": 30,
        "game_launch_attempts": 1,
        "game_process_timeout": 45.0,
        "max_seconds": 600,
    },
    "full": {
        "launch_game": True,
        "auto_resume": True,
        "bridge_connect_timeout_seconds": 300.0,
        "shutdown_grace_seconds": 90,
        "game_launch_attempts": 3,
        "game_process_timeout": 180.0,
        "max_seconds": 3600,
    },
}

CONSECUTIVE_INVALID_RESTART_LIMIT = 2


def snapshot_runs(session: AgentSession) -> dict[str, dict]:
    """Per-run metric slices keyed by run_id.

    The agent emits a ``run_report`` log at every terminal with a
    ``snapshot`` that is a PER-RUN DELTA (sliced at run start), never the
    cumulative session counter.
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
        # run-start-with-seed. Recorded so results can never be silently
        # mistaken for seed-paired ones.
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
    """Execution plan: paired (seed x mode) when a seed manifest is given,
    otherwise the legacy single-mode repetition."""
    modes = (
        ["single_action", "action_chunk"] if args.decision_mode == "paired"
        else [args.decision_mode]
    )
    tasks: list[dict] = []
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
        # PAIRED: the same seed runs under BOTH modes, consecutively.
        for seed in seeds:
            for mode in modes:
                tasks.append({"seed": seed, "decision_mode": mode})
        if args.decision_mode == "paired" and len(modes) != 2:
            raise SystemExit("paired mode requires both modes")
    else:
        seed = getattr(args, "seed", None)
        for _ in range(max(1, args.max_runs)):
            tasks.append({"seed": seed, "decision_mode": modes[0]})
    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="STS2 continuous runner.")
    parser.add_argument("--max-runs", type=int, default=3)
    parser.add_argument("--max-seconds", type=int, default=None,
                        help="wall-clock cap (default: per profile)")
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

    shutdown_grace = (
        args.shutdown_grace_seconds
        if args.shutdown_grace_seconds is not None
        else int(profile["shutdown_grace_seconds"])
    )
    connect_timeout = (
        args.bridge_connect_timeout
        if args.bridge_connect_timeout is not None
        else float(profile["bridge_connect_timeout_seconds"])
    )
    max_seconds = (
        args.max_seconds if args.max_seconds is not None
        else int(profile["max_seconds"])
    )

    tasks = build_tasks(args)
    print(f"CONTINUOUS RUNNER: profile={args.validation_profile}"
          f" tasks={len(tasks)} max_seconds={max_seconds}"
          f" shutdown_grace={shutdown_grace}s connect_timeout={connect_timeout}s"
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

    # §6: attached NEVER launches the game. No game -> skip gracefully.
    if not profile["launch_game"]:
        from game_launcher import is_game_running
        if not is_game_running():
            print("ATTACHED_VALIDATION_SKIPPED: game/bridge not available"
                  " (attached profile never launches the game).", flush=True)
            return 0

    session = AgentSession()
    t0 = time.time()
    last_seq = 0
    last_heartbeat = 0.0
    report_runs: list[dict] = []
    invalid_run_ids: set[str] = set()
    experiment_configs: list[dict] = []
    consecutive_invalid_restarts = 0
    task_index = 0
    previous_run_id = None

    def start_task(i: int) -> None:
        task = tasks[i]
        task_cfg = {**cfg, "decision_mode": task["decision_mode"]}
        if task.get("seed"):
            task_cfg["seed"] = task["seed"]
        ec = experiment_config(
            task_cfg, decision_mode=task["decision_mode"],
            seed=task.get("seed"),
            validation_profile=args.validation_profile,
        )
        experiment_configs.append(ec)
        print(f"TASK {i + 1}/{len(tasks)}: mode={task['decision_mode']}"
              f" seed={task.get('seed') or '(none)'}"
              f" agent_head={ec['agent_git_head'][:10]}", flush=True)
        session.start(task_cfg)

    start_task(task_index)

    while True:
        time.sleep(2.0)
        st = session.status()
        rid = st.get("run_id")

        # Run boundary: the agent assigns a NEW run_id whenever a run
        # genuinely ended (victory/defeat). The per-run metric slice was
        # already logged by the agent (run_report) -- pick it up here.
        if (
            previous_run_id is not None and rid and rid != previous_run_id
        ):
            runs = snapshot_runs(session)
            slice_metrics = runs.get(previous_run_id, {})
            was_invalid = previous_run_id in invalid_run_ids
            report_runs.append({
                "run_id": previous_run_id,
                "mode": experiment_configs[-1]["decision_mode"],
                "seed": experiment_configs[-1]["seed"],
                "ended_at": time.strftime("%H:%M:%S"),
                "benchmark_valid": not was_invalid,
                "metrics": slice_metrics,
            })
            if not was_invalid:
                # §25: the budget counts CONSECUTIVE invalid runs -- a
                # successful run boundary resets it.
                consecutive_invalid_restarts = 0
            print(f"RUN BOUNDARY: {previous_run_id} -> {rid}"
                  f" (finalized runs: {len(report_runs)})"
                  f" invalid={was_invalid}", flush=True)
        previous_run_id = rid

        for e in session.logs_since(last_seq):
            last_seq = max(last_seq, e["seq"])
            line = f"[{e['ts']}] {e['kind']}: {e['text']}"
            extras = " | ".join(
                f"{k}={e[k]}" for k in LOG_EXTRA_KEYS
                if e.get(k) not in (None, "")
            )
            print((line + (f" | {extras}" if extras else ""))[:600],
                  flush=True)
            if e.get("kind") == "benchmark_invalidated":
                # §24: dedup by RUN ID (an event counter would double-count
                # the invalidated log + the stopped-with-invalid state).
                invalid_run_ids.add(e.get("run_id") or st.get("run_id")
                                    or "?")
                print(
                    f"BENCHMARK INVALID (run {e.get('run_id') or '?'}):"
                    f" {e['text']} -- formal runner starts the next"
                    " repetition.", flush=True,
                )

        elapsed = time.time() - t0
        if elapsed - last_heartbeat >= 30.0:
            last_heartbeat = elapsed
            print(
                "HEARTBEAT t=%.0fs " % elapsed
                + " ".join(f"{k}={st.get(k)}" for k in STATUS_KEYS),
                flush=True,
            )

        runs_done = len(report_runs)
        stop_reason = None
        if not st["running"]:
            if st.get("benchmark_valid") is False:
                invalid_run_ids.add(str(st.get("run_id") or "?"))
                consecutive_invalid_restarts += 1
                if (
                    consecutive_invalid_restarts
                    <= CONSECUTIVE_INVALID_RESTART_LIMIT
                    and task_index + 1 < len(tasks)
                ):
                    print(
                        "CONTINUOUS: agent stopped with benchmark_valid="
                        "false; next task (consecutive-invalid restarts:"
                        f" {consecutive_invalid_restarts}/"
                        f"{CONSECUTIVE_INVALID_RESTART_LIMIT}).",
                        flush=True,
                    )
                    # Account the aborted task's run slice before moving on.
                    runs = snapshot_runs(session)
                    cur = str(st.get("run_id") or "?")
                    if cur != "?" and cur not in {
                        r["run_id"] for r in report_runs
                    }:
                        report_runs.append({
                            "run_id": cur,
                            "mode": experiment_configs[-1]["decision_mode"],
                            "seed": experiment_configs[-1]["seed"],
                            "ended_at": time.strftime("%H:%M:%S"),
                            "benchmark_valid": False,
                            "metrics": runs.get(cur, {}),
                        })
                    task_index += 1
                    start_task(task_index)
                    continue
            stop_reason = "agent stopped fatally"
        if stop_reason is None and runs_done >= args.max_runs:
            stop_reason = f"max runs reached ({args.max_runs})"
        if stop_reason is None and elapsed >= max_seconds:
            stop_reason = "time cap"

        if stop_reason:
            print(f"CONTINUOUS STOP: {stop_reason}", flush=True)
            if st["running"]:
                # §11/§34: short, profile-scaled grace (was a hardcoded
                # 300s). Ctrl+C during the wait interrupts immediately.
                deadline = time.time() + shutdown_grace
                while time.time() < deadline:
                    if session.status().get("safe_to_disconnect"):
                        break
                    time.sleep(1.0)
                if not session.status().get("safe_to_disconnect"):
                    print("WARNING: agent never reported"
                          " safe_to_disconnect; forcing stop.", flush=True)
                session.stop()
            break

    time.sleep(1.0)
    metrics = session._metrics.snapshot()
    runs = snapshot_runs(session)
    for run in report_runs:
        run.setdefault("metrics", runs.get(run["run_id"], {}))
    report = {
        "validation_profile": args.validation_profile,
        "final_session_metrics": metrics,
        "runs": report_runs,          # §23: per-run slices
        "aggregate": {                # §23: session totals, clearly separate
            "runs_finalized": len(report_runs),
            "benchmark_invalid_runs": len(invalid_run_ids),
            "wall_clock_seconds": round(time.time() - t0, 1),
        },
        "experiment_configs": experiment_configs,
    }
    out = Path(__file__).resolve().parent / "logs"
    out.mkdir(exist_ok=True)
    path = out / f"continuous_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print("\n===== FINAL SESSION METRICS =====", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"REPORT WRITTEN: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
