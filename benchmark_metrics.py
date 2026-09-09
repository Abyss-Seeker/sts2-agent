"""Benchmark accounting for the LLM-only STS2 harness beta."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import statistics
from typing import Any


@dataclass
class BenchmarkMetrics:
    benchmark_valid: bool = True
    invalidation_reason: str = ""

    model_call_count: int = 0
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
    game_action_sent_count: int = 0
    game_action_confirmed_count: int = 0
    game_action_rejected_count: int = 0

    # logical_inspection_count: cognitive boundaries (model inspect
    # requests). llm_request_count: EVERY real llm.chat() / HTTP
    # inference attempt, including failures and retries.
    # One inspection may issue several requests (JSON/timeout retries):
    #   logical_inspection_count <= llm_request_count.
    logical_inspection_count: int = 0
    llm_request_count: int = 0
    llm_success_count: int = 0
    llm_failed_request_count: int = 0

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

    def record_inspection(self) -> None:
        """One cognitive boundary / model inspect request."""
        self.logical_inspection_count += 1
        self.model_inspection_count += 1

    def record_llm_request(self) -> None:
        """EVERY real llm.chat() / HTTP inference attempt (call BEFORE
        the request so failures and retries are counted too)."""
        self.llm_request_count += 1

    def record_llm_success(
        self,
        *,
        latency_ms: int | None = None,
        first_reasoning_ms: int | None = None,
        first_content_ms: int | None = None,
        usage: dict[str, Any] | None = None,
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

    def record_llm_failure(self) -> None:
        """One failed inference attempt (timeout / HTTP error / abandoned)."""
        self.llm_failed_request_count += 1

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
        self.model_call_count += 1
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

    def record_action_sent(self, *, from_plan: bool = True) -> None:
        """Handed to the bridge -- NOT yet confirmed."""
        self.game_action_sent_count += 1
        if from_plan:
            self.executed_planned_actions_total += 1

    def record_action_confirmed(self) -> None:
        """Confirmed by the next authoritative bridge state."""
        self.game_action_confirmed_count += 1
        self.game_action_count += 1  # compat alias: game_action_count = CONFIRMED

    def record_action_rejected(self) -> None:
        """Bridge re-emitted an action-relevantly unchanged state."""
        self.game_action_rejected_count += 1

    def record_game_action(self, *, from_plan: bool = True) -> None:
        """Legacy alias: record a CONFIRMED game action."""
        self.record_action_confirmed()
        if from_plan:
            self.executed_planned_actions_total += 1

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
        add("reasoning_tokens", "reasoning_tokens")
        add("prompt_cache_hit_tokens", "prompt_cache_hit_tokens")
        add("prompt_cache_miss_tokens", "prompt_cache_miss_tokens")

        # Compatibility with providers that nest details.
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            try:
                self.reasoning_tokens += int(details.get("reasoning_tokens") or 0)
            except (TypeError, ValueError):
                pass

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

    def snapshot(self) -> dict[str, Any]:
        fresh = self.prompt_cache_miss_tokens
        hit = self.prompt_cache_hit_tokens
        cache_total = fresh + hit
        total_plans = self.plan_completed_count + self.plan_interrupted_count

        return {
            "benchmark_valid": self.benchmark_valid,
            "invalidation_reason": self.invalidation_reason,
            "model_call_count": self.model_call_count,
            "strategic_plan_count": self.strategic_plan_count,
            "game_action_count": self.game_action_count,  # CONFIRMED alias
            "game_action_sent_count": self.game_action_sent_count,
            "game_action_confirmed_count": self.game_action_confirmed_count,
            "game_action_rejected_count": self.game_action_rejected_count,
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
        }
