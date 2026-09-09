"""Benchmark accounting for the LLM-only STS2 harness beta."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import statistics
from typing import Any


def extract_reasoning_tokens(usage: Any) -> int:
    """Unified reasoning-token extraction (§8): top-level preferred,
    nested fallback, NEVER summed across aliases."""
    if not isinstance(usage, dict):
        return 0
    top = usage.get("reasoning_tokens")
    if top is not None:
        try:
            return int(top or 0)
        except (TypeError, ValueError):
            return 0
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        nested = details.get("reasoning_tokens")
        if nested is not None:
            try:
                return int(nested or 0)
            except (TypeError, ValueError):
                return 0
    return 0


@dataclass
class BenchmarkMetrics:
    benchmark_valid: bool = True
    invalidation_reason: str = ""

    strategic_plan_count: int = 0
    game_action_count: int = 0
    model_inspection_count: int = 0
    checkpoint_count: int = 0
    plan_completed_count: int = 0
    plan_interrupted_count: int = 0
    invalid_plan_count: int = 0
    invalid_action_count: int = 0
    fallback_action_count: int = 0

    # ---- Granular accounting (review remediation) ------------------
    # A game action is SENT when handed to the bridge; it is CONFIRMED
    # only by the next authoritative state (the visible world moved
    # forward -- screen changes / new turns / terminal all count), or
    # REJECTED when the bridge re-emits an action-relevantly unchanged
    # state. game_action_count stays as the CONFIRMED alias.
    # UNCONFIRMABLE: the next state arrived but the confirmation could not
    # be judged reliably -- conservatively NOT counted as confirmed.
    game_action_sent_count: int = 0
    game_action_confirmed_count: int = 0
    game_action_rejected_count: int = 0
    game_action_unconfirmable_count: int = 0

    # logical_inspection_count: cognitive boundaries (model inspect
    # requests). llm_request_count: EVERY real llm.chat() / HTTP
    # inference attempt, including failures and retries.
    # One inspection may issue several requests (JSON/timeout retries):
    #   logical_inspection_count <= llm_request_count.
    logical_inspection_count: int = 0
    llm_request_count: int = 0
    llm_success_count: int = 0
    llm_failed_request_count: int = 0

    # ---- Combat-only layer (ActionChunk optimizes combat cognition) ----
    # Same semantics as the session-level counters, restricted to
    # combat_action decisions / chunk plan actions.
    combat_llm_request_count: int = 0
    combat_llm_success_count: int = 0
    combat_logical_inspection_count: int = 0
    combat_game_action_sent_count: int = 0
    combat_game_action_confirmed_count: int = 0
    combat_turn_count: int = 0
    combat_llm_latencies_ms: list[int] = field(default_factory=list)
    combat_prompt_tokens: int = 0
    combat_completion_tokens: int = 0
    combat_reasoning_tokens: int = 0

    # ---- Continuity / recovery accounting (session-level aggregates) ----
    recoverable_termination_count: int = 0
    bridge_reconnect_count: int = 0
    game_relaunch_count: int = 0
    safe_recovery_count: int = 0
    strategic_recovery_count: int = 0
    transport_interrupted_action_count: int = 0

    # ---- Reasoning-effort forensic statistics (per effective effort) ----
    llm_calls_by_reasoning_effort: Counter = field(default_factory=Counter)
    reasoning_tokens_by_effort: Counter = field(default_factory=Counter)
    latency_ms_by_effort: dict[str, list[int]] = field(default_factory=dict)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0

    llm_latencies_ms: list[int] = field(default_factory=list)
    first_reasoning_token_ms: list[int] = field(default_factory=list)
    first_content_token_ms: list[int] = field(default_factory=list)

    planned_actions_total: int = 0
    executed_planned_actions_total: int = 0
    checkpoint_reasons: Counter = field(default_factory=Counter)

    def invalidate(self, reason: str) -> None:
        if self.benchmark_valid:
            self.invalidation_reason = reason
        self.benchmark_valid = False

    def record_inspection(self, *, combat: bool = False) -> None:
        """One cognitive boundary / model inspect request."""
        self.logical_inspection_count += 1
        self.model_inspection_count += 1
        if combat:
            self.combat_logical_inspection_count += 1

    def record_llm_request(self, *, combat: bool = False) -> None:
        """EVERY real llm.chat() / HTTP inference attempt (call BEFORE
        the request so failures and retries are counted too)."""
        self.llm_request_count += 1
        if combat:
            self.combat_llm_request_count += 1

    def record_llm_success(
        self,
        *,
        latency_ms: int | None = None,
        first_reasoning_ms: int | None = None,
        first_content_ms: int | None = None,
        usage: dict[str, Any] | None = None,
        combat: bool = False,
        effort: str | None = None,
    ) -> None:
        self.llm_success_count += 1
        if latency_ms is not None:
            self.llm_latencies_ms.append(int(latency_ms))
        if first_reasoning_ms is not None:
            self.first_reasoning_token_ms.append(int(first_reasoning_ms))
        if first_content_ms is not None:
            self.first_content_token_ms.append(int(first_content_ms))
        if usage:
            self._add_usage(usage)
        if effort:
            # Per-effort forensic statistics (§12): is "high" actually
            # smarter, or just slower?
            self.llm_calls_by_reasoning_effort[str(effort)] += 1
            if usage:
                self.reasoning_tokens_by_effort[str(effort)] += (
                    extract_reasoning_tokens(usage))
            if latency_ms is not None:
                self.latency_ms_by_effort.setdefault(
                    str(effort), []).append(int(latency_ms))
        if combat:
            self.combat_llm_success_count += 1
            if latency_ms is not None:
                self.combat_llm_latencies_ms.append(int(latency_ms))
            if usage:
                try:
                    self.combat_prompt_tokens += int(usage.get("prompt_tokens") or 0)
                except (TypeError, ValueError):
                    pass
                try:
                    self.combat_completion_tokens += int(
                        usage.get("completion_tokens") or 0)
                except (TypeError, ValueError):
                    pass
                self.combat_reasoning_tokens += extract_reasoning_tokens(usage)

    def record_llm_failure(self) -> None:
        """One failed inference attempt (timeout / HTTP error / abandoned)."""
        self.llm_failed_request_count += 1

    @property
    def model_call_count(self) -> int:
        """Legacy alias -- NEVER a separately drifting counter: identical
        to llm_request_count by definition."""
        return self.llm_request_count

    def record_model_call(
        self,
        *,
        latency_ms: int | None = None,
        first_reasoning_ms: int | None = None,
        first_content_ms: int | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        """Legacy helper: one successful model call (request + success).
        Kept for the standalone core runtime tests; agent code uses the
        granular request/success/failure API."""
        self.record_llm_request()
        self.record_llm_success(
            latency_ms=latency_ms,
            first_reasoning_ms=first_reasoning_ms,
            first_content_ms=first_content_ms,
            usage=usage,
        )

    def record_plan(self, action_count: int) -> None:
        self.strategic_plan_count += 1
        self.planned_actions_total += max(0, int(action_count))

    def record_action_sent(self, *, from_plan: bool = True, combat: bool = False) -> None:
        """Handed to the bridge -- NOT yet confirmed. Does NOT touch
        executed_planned_actions_total: only CONFIRMED plan actions count
        as executed."""
        self.game_action_sent_count += 1
        if combat:
            self.combat_game_action_sent_count += 1

    def record_action_confirmed(
        self, *, from_plan: bool = False, combat: bool = False
    ) -> None:
        """Confirmed by the next authoritative bridge state."""
        self.game_action_confirmed_count += 1
        self.game_action_count += 1  # compat alias: game_action_count = CONFIRMED
        if from_plan:
            self.executed_planned_actions_total += 1
        if combat:
            self.combat_game_action_confirmed_count += 1

    def record_action_rejected(self) -> None:
        """Bridge re-emitted an action-relevantly unchanged state."""
        self.game_action_rejected_count += 1

    def record_action_unconfirmable(self) -> None:
        """Confirmation could not be judged reliably (formatter failure /
        unknown comparison) -- conservatively NOT counted as confirmed."""
        self.game_action_unconfirmable_count += 1

    def record_game_action(self, *, from_plan: bool = True) -> None:
        """Legacy alias: record a CONFIRMED game action."""
        self.record_action_confirmed(from_plan=from_plan)

    # ---- Continuity / recovery accounting ----

    def record_transport_interrupted(self) -> None:
        """A plan step could not even be handed to the bridge (transport
        failure): NOT sent, NOT confirmed, NOT executed."""
        self.transport_interrupted_action_count += 1

    def record_recoverable_termination(self) -> None:
        self.recoverable_termination_count += 1

    def record_bridge_reconnect(self) -> None:
        self.bridge_reconnect_count += 1

    def record_game_relaunch(self) -> None:
        self.game_relaunch_count += 1

    def record_safe_recovery(self) -> None:
        """A recovery that made NO strategic decision on the agent's
        behalf (reconnect / re-observe / re-enter bridge handler)."""
        self.safe_recovery_count += 1

    def record_strategic_recovery(self) -> None:
        """A non-LLM decision was made during recovery -- invalidates the
        benchmark (set by the caller via invalidate())."""
        self.strategic_recovery_count += 1

    def record_combat_turn(self) -> None:
        self.combat_turn_count += 1

    def record_checkpoint(self, reason: str, *, interrupted: bool = True) -> None:
        self.checkpoint_count += 1
        self.checkpoint_reasons[str(reason)] += 1
        if interrupted:
            self.plan_interrupted_count += 1

    def record_plan_complete(self) -> None:
        self.plan_completed_count += 1

    def record_invalid_plan(self) -> None:
        self.invalid_plan_count += 1

    def record_invalid_action(self) -> None:
        self.invalid_action_count += 1

    def record_fallback(self, reason: str) -> None:
        self.fallback_action_count += 1
        self.invalidate(f"non-LLM fallback used: {reason}")

    def _add_usage(self, usage: dict[str, Any]) -> None:
        def add(field: str, attr: str) -> None:
            try:
                setattr(self, attr, getattr(self, attr) + int(usage.get(field) or 0))
            except (TypeError, ValueError):
                pass

        add("prompt_tokens", "prompt_tokens")
        add("completion_tokens", "completion_tokens")
        add("prompt_cache_hit_tokens", "prompt_cache_hit_tokens")
        add("prompt_cache_miss_tokens", "prompt_cache_miss_tokens")
        # §8: top-level preferred, nested fallback, NEVER summed aliases.
        self.reasoning_tokens += extract_reasoning_tokens(usage)

    @staticmethod
    def _percentile(values: list[int], p: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])
        rank = (len(ordered) - 1) * p
        low = int(rank)
        high = min(low + 1, len(ordered) - 1)
        frac = rank - low
        return ordered[low] * (1 - frac) + ordered[high] * frac

    def slice_since(self, baseline: dict[str, Any]) -> dict[str, Any]:
        """Per-run metric slice: numeric fields diffed against a baseline
        snapshot, non-numeric state taken as-is. Latency/token LIST fields
        cannot be diffed and are omitted from the slice (session totals
        remain in snapshot()); the runner reports those at session level.
        """
        current = self.snapshot()
        out: dict[str, Any] = {}
        for key, val in current.items():
            base = baseline.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool) \
                    and isinstance(base, (int, float)) \
                    and not isinstance(base, bool):
                out[key] = max(0, val - base)
            elif isinstance(val, (list, tuple, dict)):
                continue  # non-diffable; session-level only
            else:
                out[key] = val
        return out

    def snapshot(self) -> dict[str, Any]:
        fresh = self.prompt_cache_miss_tokens
        hit = self.prompt_cache_hit_tokens
        cache_total = fresh + hit
        total_plans = self.plan_completed_count + self.plan_interrupted_count

        return {
            "benchmark_valid": self.benchmark_valid,
            "invalidation_reason": self.invalidation_reason,
            "model_call_count": self.model_call_count,  # alias of llm_request_count
            "strategic_plan_count": self.strategic_plan_count,
            "game_action_count": self.game_action_count,  # CONFIRMED alias
            "game_action_sent_count": self.game_action_sent_count,
            "game_action_confirmed_count": self.game_action_confirmed_count,
            "game_action_rejected_count": self.game_action_rejected_count,
            "game_action_unconfirmable_count": self.game_action_unconfirmable_count,
            "model_inspection_count": self.model_inspection_count,
            "logical_inspection_count": self.logical_inspection_count,
            "llm_request_count": self.llm_request_count,
            "llm_success_count": self.llm_success_count,
            "llm_failed_request_count": self.llm_failed_request_count,
            "checkpoint_count": self.checkpoint_count,
            "plan_completed_count": self.plan_completed_count,
            "plan_interrupted_count": self.plan_interrupted_count,
            "invalid_plan_count": self.invalid_plan_count,
            "invalid_action_count": self.invalid_action_count,
            "fallback_action_count": self.fallback_action_count,
            # HTTP-attempt layer alias (identical value; clearer name for
            # the "every real urlopen" accounting level).
            "http_attempt_count": self.llm_request_count,
            # Combat-only layer (X): ActionChunk targets combat cognition.
            "combat_llm_request_count": self.combat_llm_request_count,
            "combat_llm_success_count": self.combat_llm_success_count,
            "combat_logical_inspection_count": self.combat_logical_inspection_count,
            "combat_game_action_sent_count": self.combat_game_action_sent_count,
            "combat_game_action_confirmed_count": self.combat_game_action_confirmed_count,
            "combat_actions_per_llm_call": (
                self.combat_game_action_confirmed_count
                / self.combat_llm_request_count
                if self.combat_llm_request_count else 0.0
            ),
            "combat_actions_per_logical_inspection": (
                self.combat_game_action_confirmed_count
                / self.combat_logical_inspection_count
                if self.combat_logical_inspection_count else 0.0
            ),
            "combat_llm_latency_ms_p50": self._percentile(
                self.combat_llm_latencies_ms, 0.50),
            "combat_llm_latency_ms_p95": self._percentile(
                self.combat_llm_latencies_ms, 0.95),
            "combat_prompt_tokens": self.combat_prompt_tokens,
            "combat_completion_tokens": self.combat_completion_tokens,
            "combat_reasoning_tokens": self.combat_reasoning_tokens,
            "combat_turn_count": self.combat_turn_count,
            # Continuity / recovery (S)
            "recoverable_termination_count": self.recoverable_termination_count,
            "bridge_reconnect_count": self.bridge_reconnect_count,
            "game_relaunch_count": self.game_relaunch_count,
            "safe_recovery_count": self.safe_recovery_count,
            "strategic_recovery_count": self.strategic_recovery_count,
            "transport_interrupted_action_count": self.transport_interrupted_action_count,
            # Denominator is EVERY inference attempt (incl. failed/retried),
            # so the KPI cannot be flattered by silent failures.
            "actions_per_llm_call": (
                self.game_action_confirmed_count / self.llm_request_count
                if self.llm_request_count
                else 0.0
            ),
            "actions_per_logical_inspection": (
                self.game_action_confirmed_count / self.logical_inspection_count
                if self.logical_inspection_count
                else 0.0
            ),
            "plan_completion_ratio": (
                self.plan_completed_count / total_plans if total_plans else 0.0
            ),
            "executed_vs_planned_ratio": (
                self.executed_planned_actions_total / self.planned_actions_total
                if self.planned_actions_total
                else 0.0
            ),
            "llm_latency_ms_p50": self._percentile(self.llm_latencies_ms, 0.50),
            "llm_latency_ms_p95": self._percentile(self.llm_latencies_ms, 0.95),
            "llm_latency_ms_mean": (
                statistics.mean(self.llm_latencies_ms)
                if self.llm_latencies_ms
                else None
            ),
            "first_reasoning_token_ms_p50": self._percentile(
                self.first_reasoning_token_ms, 0.50
            ),
            "first_content_token_ms_p50": self._percentile(
                self.first_content_token_ms, 0.50
            ),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "prompt_cache_hit_tokens": hit,
            "prompt_cache_miss_tokens": fresh,
            "cache_hit_ratio": (hit / cache_total if cache_total else None),
            "checkpoint_reasons": dict(self.checkpoint_reasons),
            # Reasoning-effort forensics (§12)
            "reasoning_by_effort": {
                effort: {
                    "calls": calls,
                    "reasoning_tokens": self.reasoning_tokens_by_effort.get(
                        effort, 0),
                    "mean_reasoning_tokens": (
                        self.reasoning_tokens_by_effort.get(effort, 0) / calls
                        if calls else 0.0),
                    "latency_ms_p50": self._percentile(
                        self.latency_ms_by_effort.get(effort, []), 0.50),
                    "latency_ms_p95": self._percentile(
                        self.latency_ms_by_effort.get(effort, []), 0.95),
                }
                for effort, calls in sorted(
                    self.llm_calls_by_reasoning_effort.items())
            },
        }
