"""Safe sequential executor for model-produced ActionChunks.

The bridge remains one-state/one-action.  This executor merely reuses a
previous model commitment across multiple bridge handshakes.

It NEVER selects substitute actions.  When a step cannot be proven valid,
it returns NEED_MODEL/CHECKPOINT.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from action_plan import (
    ActionChunk,
    ActionKind,
    PlannedAction,
    ResolveError,
    resolve_card_ref,
    resolve_enemy_ref,
)
from checkpoint import (
    ActionAcceptance,
    CheckpointDecision,
    CheckpointReason,
    evaluate_after_action,
)


ValidateAction = Callable[
    [dict[str, Any], dict[str, Any]],
    tuple[dict[str, Any] | None, str],
]


class ExecutorStatus(str, Enum):
    NEED_MODEL = "NEED_MODEL"
    READY_ACTION = "READY_ACTION"
    WAITING_RESULT = "WAITING_RESULT"
    # Genuine in-flight/control-flow state: the bridge ACCEPTED the sent
    # command but the newest authoritative visible state has not advanced
    # yet. The in-flight step and the plan are RETAINED; the caller must
    # simply wait for another authoritative state -- never re-prompt the
    # model, never reset the chunk, never send another gameplay action.
    WAITING_ADVANCE = "WAITING_ADVANCE"
    TERMINAL = "TERMINAL"


@dataclass(frozen=True)
class PreparedStep:
    plan_id: str
    step_index: int
    planned: PlannedAction
    bridge_action: dict[str, Any]
    description: str


@dataclass(frozen=True)
class ExecutorEvent:
    status: ExecutorStatus
    checkpoint: CheckpointDecision | None = None
    prepared: PreparedStep | None = None
    completed_plan_id: str | None = None
    plan_completed: bool = False
    detail: str = ""


class ActionChunkExecutor:
    """Stateful pending-plan machine for one AgentSession."""

    def __init__(self) -> None:
        self.chunk: ActionChunk | None = None
        self.index: int = 0
        self._inflight: PreparedStep | None = None
        self._before_state: dict[str, Any] | None = None

    @property
    def has_pending_plan(self) -> bool:
        return self.chunk is not None

    @property
    def inflight(self) -> PreparedStep | None:
        return self._inflight

    def reset(self) -> None:
        self.chunk = None
        self.index = 0
        self._inflight = None
        self._before_state = None

    def submit(self, chunk: ActionChunk) -> None:
        if self._inflight is not None:
            raise RuntimeError("cannot replace plan while an action is in flight")
        self.chunk = chunk
        self.index = 0
        self._before_state = None

    def cancel(self) -> str | None:
        old = self.chunk.plan_id if self.chunk else None
        self.reset()
        return old

    def _planned_to_bridge(
        self,
        planned: PlannedAction,
        state: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        assert self.chunk is not None

        if planned.kind is ActionKind.END_TURN:
            return {"action": "end_turn"}, "end turn"

        if planned.kind is ActionKind.PLAY:
            if not planned.card_ref:
                raise ResolveError("planned play missing card_ref")
            card_ref = self.chunk.card_refs.get(planned.card_ref)
            if card_ref is None:
                raise ResolveError(f"unknown card_ref {planned.card_ref!r}")
            ci = resolve_card_ref(card_ref, state)

            ti = -1
            if planned.target_ref:
                enemy_ref = self.chunk.enemy_refs.get(planned.target_ref)
                if enemy_ref is None:
                    raise ResolveError(f"unknown target_ref {planned.target_ref!r}")
                ti = resolve_enemy_ref(enemy_ref, state)

            return (
                {"action": "play", "card_index": ci, "target_index": ti},
                f"play {planned.card_ref} at current HAND[{ci}]"
                + (f" -> {planned.target_ref}/ENEMY[{ti}]" if ti >= 0 else ""),
            )

        if planned.kind is ActionKind.POTION:
            if planned.potion_slot is None:
                raise ResolveError("planned potion missing potion_slot")
            ti = -1
            if planned.target_ref:
                enemy_ref = self.chunk.enemy_refs.get(planned.target_ref)
                if enemy_ref is None:
                    raise ResolveError(f"unknown target_ref {planned.target_ref!r}")
                ti = resolve_enemy_ref(enemy_ref, state)
            return (
                {
                    "action": "potion",
                    "slot": planned.potion_slot,
                    "target_index": ti,
                },
                f"use potion slot {planned.potion_slot}"
                + (f" -> {planned.target_ref}/ENEMY[{ti}]" if ti >= 0 else ""),
            )

        raise ResolveError(f"unsupported planned action kind {planned.kind!r}")

    def prepare_next(
        self,
        state: dict[str, Any],
        validate_action: ValidateAction,
    ) -> ExecutorEvent:
        """Prepare, resolve and validate ONE bridge action.

        Does not send anything. Caller must call mark_sent() only after it
        actually sends the returned action.
        """
        if self._inflight is not None:
            return ExecutorEvent(
                ExecutorStatus.WAITING_RESULT,
                detail="cannot prepare while previous action is in flight",
            )
        if self.chunk is None:
            return ExecutorEvent(ExecutorStatus.NEED_MODEL, detail="no pending plan")
        if self.index >= len(self.chunk.actions):
            plan_id = self.chunk.plan_id
            self.reset()
            return ExecutorEvent(
                ExecutorStatus.NEED_MODEL,
                checkpoint=CheckpointDecision(
                    True, CheckpointReason.PLAN_COMPLETE, "chunk exhausted"
                ),
                completed_plan_id=plan_id,
                plan_completed=True,
            )

        planned = self.chunk.actions[self.index]
        try:
            raw_action, desc = self._planned_to_bridge(planned, state)
        except ResolveError as exc:
            old = self.chunk.plan_id
            self.reset()
            reason = (
                CheckpointReason.TARGET_GONE
                if "target" in str(exc).lower()
                else CheckpointReason.CARD_GONE
            )
            return ExecutorEvent(
                ExecutorStatus.NEED_MODEL,
                checkpoint=CheckpointDecision(True, reason, str(exc)),
                completed_plan_id=old,
                detail=str(exc),
            )

        normalized, error = validate_action(state, raw_action)
        if normalized is None:
            old = self.chunk.plan_id
            self.reset()
            return ExecutorEvent(
                ExecutorStatus.NEED_MODEL,
                checkpoint=CheckpointDecision(
                    True,
                    CheckpointReason.NEXT_ACTION_ILLEGAL,
                    error or "validate_action rejected planned step",
                ),
                completed_plan_id=old,
                detail=error,
            )

        prepared = PreparedStep(
            plan_id=self.chunk.plan_id,
            step_index=self.index,
            planned=planned,
            bridge_action=normalized,
            description=desc,
        )
        return ExecutorEvent(ExecutorStatus.READY_ACTION, prepared=prepared)

    def mark_sent(
        self,
        prepared: PreparedStep,
        before_state: dict[str, Any],
    ) -> None:
        if self.chunk is None or prepared.plan_id != self.chunk.plan_id:
            raise RuntimeError("prepared step does not belong to current plan")
        if prepared.step_index != self.index:
            raise RuntimeError("prepared step index is stale")
        if self._inflight is not None:
            raise RuntimeError("another action is already in flight")
        self._inflight = prepared
        # The caller owns the original bridge dict.  We only read it.
        self._before_state = before_state

    def accept_state(
        self,
        after_state: dict[str, Any],
        *,
        acceptance: ActionAcceptance = ActionAcceptance.UNKNOWN,
    ) -> ExecutorEvent:
        """Reconcile the next authoritative bridge state after one send.

        If no checkpoint is needed, returns READY_ACTION only indirectly:
        caller should invoke prepare_next(after_state, validate_action).
        This separation keeps validation at the latest authoritative state.

        ``acceptance`` is the AUTHORITATIVE game-side outcome (see
        :class:`ActionAcceptance`) of the command that produced
        ``after_state``. It is REQUIRED to distinguish a real rejection from
        an accepted-but-not-yet-observable action, and from an unverifiable
        one. The executor owns that classification (and therefore the
        mutation decision) so it can NEVER irreversibly reset a plan that is
        merely waiting for the authoritative world to advance.
        """
        if self._inflight is None or self._before_state is None or self.chunk is None:
            return ExecutorEvent(
                ExecutorStatus.NEED_MODEL,
                checkpoint=CheckpointDecision(
                    True,
                    CheckpointReason.UNKNOWN_STATE_CHANGE,
                    "received state without a tracked in-flight plan action",
                ),
            )

        executed = self._inflight
        before = self._before_state
        next_index = self.index + 1
        next_action = (
            self.chunk.actions[next_index]
            if next_index < len(self.chunk.actions)
            else None
        )

        checkpoint, _delta = evaluate_after_action(
            before=before,
            after=after_state,
            chunk=self.chunk,
            executed_action=executed.planned,
            next_action=next_action,
            acceptance=acceptance,
        )
        chunk_exhausted = next_index >= len(self.chunk.actions)

        if checkpoint.reason in (
            CheckpointReason.AWAITING_ADVANCE,
            CheckpointReason.ADVANCE_UNVERIFIED,
        ):
            # ACCEPTED (or unverifiable) BUT NOT YET OBSERVABLY ADVANCED is a
            # WAITING state. Retain the in-flight step AND the original
            # before-state so the SAME command can be reconciled against a
            # later authoritative state. Do NOT advance the index, do NOT
            # reset the chunk.
            return ExecutorEvent(
                ExecutorStatus.WAITING_ADVANCE,
                checkpoint=checkpoint,
                prepared=executed,
                detail="awaiting authoritative advance",
            )

        # Clear in-flight tracking before any resolving return.
        self._inflight = None
        self._before_state = None

        if checkpoint.reason is CheckpointReason.ACTION_REJECTED:
            # Do NOT advance.  The model's intended action was not confirmed.
            old = self.chunk.plan_id
            self.reset()
            return ExecutorEvent(
                ExecutorStatus.NEED_MODEL,
                checkpoint=checkpoint,
                completed_plan_id=old,
                detail="planned step not confirmed",
            )

        # Every other post-action state change means the sent action was at
        # least accepted enough to move the visible state forward.
        self.index = next_index

        if checkpoint.reason is CheckpointReason.TERMINAL:
            old = self.chunk.plan_id
            self.reset()
            return ExecutorEvent(
                ExecutorStatus.TERMINAL,
                checkpoint=checkpoint,
                completed_plan_id=old,
                plan_completed=chunk_exhausted,
            )

        if checkpoint.needed:
            old = self.chunk.plan_id
            self.reset()
            return ExecutorEvent(
                ExecutorStatus.NEED_MODEL,
                checkpoint=checkpoint,
                completed_plan_id=old,
                plan_completed=chunk_exhausted,
            )

        # Plan continues. Caller now runs prepare_next() against after_state.
        return ExecutorEvent(
            ExecutorStatus.READY_ACTION,
            detail="plan remains epistemically valid; prepare next step",
        )
