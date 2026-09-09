"""Continuous benchmark runner for the STS2 LLM agent.

Runs the agent continuously across runs:

  - recoverable termination (result=terminated): the agent resumes the
    SAME save with its run context preserved (built into AgentSession);
    the runner just keeps going.
  - victory / defeat: the runner finalizes the run (snapshots its
    metrics slice under the run_id) and continues to the next run.
  - benchmark-invalid: the run is marked invalid in the report; in a
    formal benchmark the runner starts the next configured seed instead
    of stopping the whole session.

Stop conditions: --max-runs reached, --max-seconds elapsed, or fatal
failure. Disconnecting is only allowed when the agent reports
``safe_to_disconnect`` (no in-flight HTTP inference, no chunk action
inflight, no pending single-action confirmation).

Usage:
    python continuous_run.py --max-runs 3 --max-seconds 3600
"""

from __future__ import annotations

import argparse
import json
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


def snapshot_runs(session: AgentSession) -> dict[str, dict]:
    """Per-run metric slices keyed by run_id (session-level aggregates
    live in the metrics object itself)."""
    out: dict[str, dict] = {}
    for e in session.logs_since(0):
        if e.get("kind") != "run_report":
            continue
        rid = e.get("run_id") or "?"
        out[rid] = e.get("snapshot") or {}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="STS2 continuous runner.")
    parser.add_argument("--max-runs", type=int, default=3)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--grace-seconds", type=int, default=300,
                        help="min runtime before an evidence-based stop")
    parser.add_argument("--decision-mode", default="action_chunk",
                        choices=["single_action", "action_chunk"])
    args = parser.parse_args(argv)

    cfg = load_config()
    cfg.update({
        "decision_mode": args.decision_mode,
        "failure_policy": "benchmark_strict",
        "delta_observations": False,
        "stream_mode": "off",
        "llm_retries": 0,
        "provider_profile": "auto",
        "thinking_enabled": True,
        "reasoning_effort": "high",
        "save_log": True,
        "auto_launch_game": True,
        "auto_resume": True,  # recovery is the whole point of this runner
    })
    print(f"CONTINUOUS RUNNER: max_runs={args.max_runs}"
          f" max_seconds={args.max_seconds} mode={args.decision_mode}",
          flush=True)

    session = AgentSession()
    session.start(cfg)
    t0 = time.time()
    last_seq = 0
    last_heartbeat = 0.0
    runs_finalized: dict[str, dict] = {}
    current_run_id = None
    benchmark_invalid_runs = 0
    restarts = 0

    while True:
        time.sleep(2.0)
        st = session.status()

        # Run boundary detection: the agent assigns a NEW run_id whenever
        # a run genuinely ended (victory/defeat -> new run context).
        rid = st.get("run_id")
        if current_run_id is None:
            current_run_id = rid
        elif rid and rid != current_run_id:
            runs_finalized[current_run_id] = {
                "ended_at": time.strftime("%H:%M:%S"),
                "metrics": session._metrics.snapshot(),
            }
            print(f"RUN BOUNDARY: {current_run_id} -> {rid}"
                  f" (finalized runs: {len(runs_finalized)})", flush=True)
            current_run_id = rid

        for e in session.logs_since(last_seq):
            last_seq = max(last_seq, e["seq"])
            line = f"[{e['ts']}] {e['kind']}: {e['text']}"
            extras = " | ".join(
                f"{k}={e[k]}" for k in LOG_EXTRA_KEYS
                if e.get(k) not in (None, "")
            )
            print((line + (f" | {extras}" if extras else ""))[:600],
                  flush=True)
            if (e.get("kind") == "benchmark_invalidated"):
                benchmark_invalid_runs += 1
                print(
                    f"BENCHMARK INVALID (run {current_run_id}): {e['text']}"
                    " -- formal runner would discard this run and start"
                    " the next configured repetition.",
                    flush=True,
                )

        elapsed = time.time() - t0
        if elapsed - last_heartbeat >= 30.0:
            last_heartbeat = elapsed
            print(
                "HEARTBEAT t=%.0fs " % elapsed
                + " ".join(f"{k}={st.get(k)}" for k in STATUS_KEYS),
                flush=True,
            )

        runs_done = len(runs_finalized)
        stop_reason = None
        if not st["running"]:
            if st.get("benchmark_valid") is False:
                # Benchmark-invalid run: mark it and start the next
                # configured repetition instead of killing the session
                # (bounded -- two consecutive invalid stops still stop).
                benchmark_invalid_runs += 1
                restarts += 1
                if restarts <= 2:
                    print(
                        "CONTINUOUS: agent stopped with benchmark_valid="
                        "false; starting the next run (bounded restart).",
                        flush=True,
                    )
                    session.start(cfg)
                    continue
            stop_reason = "agent stopped fatally"
        else:
            restarts = 0
        if stop_reason is None and runs_done >= args.max_runs:
            stop_reason = f"max runs reached ({args.max_runs})"
        if stop_reason is None and elapsed >= args.max_seconds:
            stop_reason = "time cap"

        if stop_reason:
            print(f"CONTINUOUS STOP: {stop_reason}", flush=True)
            if st["running"]:
                # V: only disconnect when the agent says it is safe.
                wait_until = time.time() + 300
                while time.time() < wait_until:
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
    report = {
        "final_metrics": metrics,
        "runs_finalized": runs_finalized,
        "benchmark_invalid_events": benchmark_invalid_runs,
        "wall_clock_seconds": round(time.time() - t0, 1),
    }
    out = Path(__file__).resolve().parent / "logs"
    out.mkdir(exist_ok=True)
    path = out / f"continuous_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print("\n===== FINAL METRICS =====", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"REPORT WRITTEN: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
