"""Real-game ActionChunk smoke runner.

Protocol/cognition validation only -- NOT an A/B benchmark, NOT a win-rate
attempt. Uses the current beta prompts untouched, strict benchmark policy,
no human correction of model output.

Config overrides (smoke contract):
    decision_mode=action_chunk, failure_policy=benchmark_strict,
    delta_observations=false, stream_mode=off, llm_retries=0,
    provider_profile=auto, thinking_enabled=true, reasoning_effort=high.

Stop conditions:
    - agent stopped (terminal / strict failure / bridge loss), OR
    - >=2 plans completed (evidence gathered) after a grace period, OR
    - 30-minute hard cap.

Output: streaming console log + logs/smoke_report_<ts>.md audit table +
final metrics snapshot JSON.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import AgentSession
from server import load_config

SMOKE_OVERRIDES = {
    "decision_mode": "action_chunk",
    "failure_policy": "benchmark_strict",
    "delta_observations": False,
    "stream_mode": "off",
    "llm_retries": 0,
    "provider_profile": "auto",
    "thinking_enabled": True,
    "reasoning_effort": "high",
    "save_log": True,
    "auto_launch_game": True,
}

MAX_SECONDS = 30 * 60
GRACE_SECONDS = 180  # minimum runtime once 2 plans completed

STATUS_KEYS = (
    "running", "bridge_connected", "current_state_type", "floor", "act",
    "hp", "agent_phase", "benchmark_valid", "invalidation_reason",
    "logical_inspection_count", "llm_request_count", "llm_success_count",
    "llm_failed_request_count", "game_action_sent_count",
    "game_action_confirmed_count", "game_action_rejected_count",
    "game_action_unconfirmable_count", "strategic_plan_count",
    "plan_completed_count", "plan_interrupted_count", "checkpoint_count",
    "current_plan_id", "current_plan_step", "current_plan_total",
    "last_checkpoint_reason", "prompt_tokens", "completion_tokens",
    "reasoning_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
)

LOG_EXTRA_KEYS = (
    "plan_id", "step_index", "action", "result", "reason",
    "executed_steps", "remaining_steps_discarded", "plan_completed",
    "llm_ms", "actions", "state_type", "source",
)


def audit_markdown(logs: list[dict], metrics: dict) -> str:
    """Per-plan audit table from the run log (real plan -> sent -> confirmed)."""
    plans: dict[str, dict] = {}
    order: list[str] = []
    for e in logs:
        kind = e.get("kind")
        if kind == "model_plan":
            pid = e.get("plan_id") or "?"
            if pid not in plans:
                plans[pid] = {"thought": "", "actions": "", "llm_ms": None,
                              "steps": [], "checkpoint": None}
                order.append(pid)
            plans[pid]["thought"] = e.get("text", "")
            plans[pid]["actions"] = e.get("actions", "")
            plans[pid]["llm_ms"] = e.get("llm_ms")
        elif kind == "game_action":
            pid = e.get("plan_id")
            if not pid:
                continue
            plans.setdefault(pid, {"thought": "", "actions": "",
                                   "llm_ms": None, "steps": [],
                                   "checkpoint": None})
            if pid not in order:
                order.append(pid)
            plans[pid]["steps"].append({
                "step_index": e.get("step_index"),
                "desc": e.get("text", ""),
                "action": e.get("action", ""),
                "result": e.get("result", ""),
            })
        elif kind == "plan_checkpoint":
            pid = e.get("plan_id")
            if not pid:
                continue
            plans.setdefault(pid, {"thought": "", "actions": "",
                                   "llm_ms": None, "steps": [],
                                   "checkpoint": None})
            if pid not in order:
                order.append(pid)
            plans[pid]["checkpoint"] = {
                "reason": e.get("reason"),
                "executed_steps": e.get("executed_steps"),
                "remaining_discarded": e.get("remaining_steps_discarded"),
                "plan_completed": e.get("plan_completed"),
                "detail": e.get("text", ""),
            }

    lines = ["# ActionChunk smoke -- real plan audit", ""]
    for i, pid in enumerate(order, 1):
        p = plans[pid]
        lines.append(f"## LLM CALL #{i} -- PLAN {pid} (llm_ms={p['llm_ms']})")
        lines.append(f"- thought: {p['thought']}")
        lines.append(f"- actions planned: {p['actions']}")
        lines.append("- execution:")
        if p["steps"]:
            for stp in p["steps"]:
                lines.append(
                    f"    - step {stp['step_index']}: {stp['desc']}"
                    f" | bridge_action={stp['action']}"
                    f" | SENT -> result: {stp['result']}"
                )
        else:
            lines.append("    - (no action was sent from this plan)")
        cp = p["checkpoint"]
        if cp:
            lines.append(
                f"- checkpoint: reason={cp['reason']}"
                f" executed_steps={cp['executed_steps']}"
                f" remaining_discarded={cp['remaining_discarded']}"
                f" plan_completed={cp['plan_completed']}"
                f" ({cp['detail']})"
            )
        lines.append("")

    lines.append("## Final metrics snapshot")
    lines.append("```json")
    lines.append(json.dumps(metrics, ensure_ascii=False, indent=2))
    lines.append("```")
    return "\n".join(lines)


def main() -> int:
    cfg = load_config()
    cfg.update(SMOKE_OVERRIDES)
    print(f"SMOKE CONFIG overrides: {json.dumps(SMOKE_OVERRIDES)}", flush=True)
    print(f"model={cfg.get('model')} base={cfg.get('api_base_url')}",
          flush=True)

    s = AgentSession()
    s.start(cfg)
    t0 = time.time()
    last_seq = 0
    last_heartbeat = 0.0

    while True:
        time.sleep(2.0)
        st = s.status()
        for e in s.logs_since(last_seq):
            last_seq = max(last_seq, e["seq"])
            line = f"[{e['ts']}] {e['kind']}: {e['text']}"
            extras = " | ".join(
                f"{k}={e[k]}" for k in LOG_EXTRA_KEYS if e.get(k) not in (None, "")
            )
            print((line + (f" | {extras}" if extras else ""))[:600], flush=True)

        elapsed = time.time() - t0
        if elapsed - last_heartbeat >= 30.0:
            last_heartbeat = elapsed
            hb = " ".join(f"{k}={st.get(k)}" for k in STATUS_KEYS)
            print(f"HEARTBEAT t={elapsed:.0f}s {hb}", flush=True)

        stop_reason = None
        if not st["running"]:
            stop_reason = "agent stopped"
        elif (st["plan_completed_count"] >= 2
              and elapsed >= GRACE_SECONDS):
            stop_reason = "evidence gathered (>=2 plans completed)"
        elif elapsed > MAX_SECONDS:
            stop_reason = "time cap"

        if stop_reason:
            # Never yank the connection while the model is mid-thought or a
            # plan action is in flight: the game would wait for a response
            # that never arrives and abort the episode. Wait for a safe gap
            # (bounded) before disconnecting.
            if st["running"]:
                print(
                    f"SMOKE STOP: {stop_reason}; waiting for a safe gap"
                    " (no in-flight decision/action) before disconnecting...",
                    flush=True,
                )
                safe_deadline = time.time() + 240
                while time.time() < safe_deadline:
                    st = s.status()
                    phase = st.get("agent_phase")
                    plan_busy = bool(st.get("current_plan_id"))
                    if (not st["running"]
                            or phase in ("idle", "checkpoint")
                            and not plan_busy):
                        break
                    time.sleep(1.0)
                print(
                    "SMOKE STOP: disconnecting. NOTE: the game will abort"
                    " this episode at its next decision point (expected).",
                    flush=True,
                )
                s.stop()
            break

    time.sleep(1.0)
    metrics = s._metrics.snapshot()
    print("\n===== FINAL METRICS =====", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)

    report = audit_markdown(s.logs_since(0), metrics)
    out = Path(__file__).resolve().parent / "logs"
    out.mkdir(exist_ok=True)
    path = out / f"smoke_report_{time.strftime('%Y%m%d_%H%M%S')}.md"
    path.write_text(report, encoding="utf-8")
    print(f"\nAUDIT REPORT WRITTEN: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
