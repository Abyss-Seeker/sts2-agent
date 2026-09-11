"""Conservative epistemic checkpoint policy for ActionChunk execution.

This module must NEVER decide strategy.  It only answers:
"Can the already-committed model plan still be executed against the new,
authoritative human-visible state without acquiring materially new information?"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from action_plan import ActionChunk, PlannedAction, ActionKind, ResolveError, resolve_card_ref, resolve_enemy_ref
from state_diff import StateDelta, diff_states


class ActionAcceptance(str, Enum):
    """AUTHORITATIVE game-side outcome of ONE sent gameplay command.

    SENT != ACCEPTED. A successful ``socket.sendall()`` only proves the
    bytes left Python; it says NOTHING about whether the game validated and
    started the command. The only authoritative source is the C# handler
    lifecycle, surfaced over the protocol as the next state's
    ``previous_action_result`` metadata:

        {"request_id": "<the state that was answered>",
         "accepted": true|false,
         "reason": "<diagnostic, optional>"}

    ACCEPTED -- the game-side handler validated and started the command
                (it may still be resolving: modal / animation / turn switch).
    REJECTED -- the game-side handler refused the command.
    UNKNOWN  -- no authoritative signal is available (legacy bridge, missing
                or stale/mismatched ACK, transport loss). It must NEVER be
                silently treated as ACCEPTED or as REJECTED.
    """

    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


def resolve_action_acceptance(
    previous_action_result: Any,
    expected_request_id: str | None,
) -> ActionAcceptance:
    """Map a state's ``previous_action_result`` metadata to an acceptance.

    The ACK must correspond to the EXACT state/action handshake: a stale ACK
    from action N must never classify action N+1. Any mismatch (missing
    metadata, missing request ids, different request id, non-bool
    ``accepted``) yields UNKNOWN -- never an invented acceptance.
    """
    if not isinstance(previous_action_result, dict):
        return ActionAcceptance.UNKNOWN
    expected = str(expected_request_id or "").strip()
    if not expected:
        return ActionAcceptance.UNKNOWN
    ack_rid = previous_action_result.get("request_id")
    if ack_rid is None or str(ack_rid).strip() != expected:
        # Stale / uncorrelated ACK: must not classify the current action.
        return ActionAcceptance.UNKNOWN
    accepted = previous_action_result.get("accepted")
    if accepted is True:
        return ActionAcceptance.ACCEPTED
    if accepted is False:
        return ActionAcceptance.REJECTED
    return ActionAcceptance.UNKNOWN


class CheckpointReason(str, Enum):
    NONE = "NONE"
    MODEL_REQUESTED = "MODEL_REQUESTED"
    SCREEN_CHANGED = "SCREEN_CHANGED"
    NEW_TURN = "NEW_TURN"
    HAND_CHANGED = "HAND_CHANGED"
    ACTION_REJECTED = "ACTION_REJECTED"
    # The game ACCEPTED the gameplay command, but the newest authoritative
    # human-visible state has not advanced yet (turn switch animation,
    # native modal about to appear, ...). This is a WAITING state, NOT a
    # rejection: the in-flight action must be retained and reconciled
    # against a LATER state.
    AWAITING_ADVANCE = "AWAITING_ADVANCE"
    # Action-relevantly unchanged state with NO authoritative acceptance
    # signal (UNKNOWN). Conservative protocol-uncertain WAITING: never
    # invented as accepted, never claimed as rejected.
    ADVANCE_UNVERIFIED = "ADVANCE_UNVERIFIED"
    PLAN_COMPLETE = "PLAN_COMPLETE"
    CARD_GONE = "CARD_GONE"
    TARGET_GONE = "TARGET_GONE"
    POTION_INVALID = "POTION_INVALID"
    NEXT_ACTION_ILLEGAL = "NEXT_ACTION_ILLEGAL"
    COMBAT_RESOLVED = "COMBAT_RESOLVED"
    TERMINAL = "TERMINAL"
    UNKNOWN_STATE_CHANGE = "UNKNOWN_STATE_CHANGE"


@dataclass(frozen=True)
class CheckpointDecision:
    needed: bool
    reason: CheckpointReason
    detail: str = ""


TERMINAL_TYPES = {"game_over", "run_complete"}


def _next_ref_resolvable(
    chunk: ActionChunk,
    next_action: PlannedAction | None,
    state: dict[str, Any],
) -> CheckpointDecision:
    if next_action is None:
        return CheckpointDecision(True, CheckpointReason.PLAN_COMPLETE, "chunk exhausted")

    if next_action.kind is ActionKind.PLAY:
        assert next_action.card_ref is not None
        try:
            resolve_card_ref(chunk.card_refs[next_action.card_ref], state)
        except (KeyError, ResolveError) as exc:
            return CheckpointDecision(True, CheckpointReason.CARD_GONE, str(exc))

        if next_action.target_ref:
            try:
                resolve_enemy_ref(chunk.enemy_refs[next_action.target_ref], state)
            except (KeyError, ResolveError) as exc:
                return CheckpointDecision(True, CheckpointReason.TARGET_GONE, str(exc))

    elif next_action.kind in (ActionKind.POTION, ActionKind.DISCARD_POTION):
        if next_action.potion_slot is None:
            return CheckpointDecision(
                True, CheckpointReason.POTION_INVALID, "planned potion has no slot"
            )
        potions = state.get("potions") or []
        found = None
        for i, p in enumerate(potions):
            if not isinstance(p, dict):
                continue
            slot = p.get("slot", i)
            if slot == next_action.potion_slot:
                found = p
                break
        if found is None or found.get("empty") or not found.get("can_discard" if next_action.kind is ActionKind.DISCARD_POTION else "can_use", False):
            return CheckpointDecision(
                True,
                CheckpointReason.POTION_INVALID,
                f"potion slot {next_action.potion_slot} no longer usable",
            )
        if next_action.target_ref:
            try:
                resolve_enemy_ref(chunk.enemy_refs[next_action.target_ref], state)
            except (KeyError, ResolveError) as exc:
                return CheckpointDecision(True, CheckpointReason.TARGET_GONE, str(exc))

    return CheckpointDecision(False, CheckpointReason.NONE)


def evaluate_after_action(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    chunk: ActionChunk,
    executed_action: PlannedAction,
    next_action: PlannedAction | None,
    acceptance: ActionAcceptance = ActionAcceptance.UNKNOWN,
) -> tuple[CheckpointDecision, StateDelta]:
    """Evaluate only epistemic/protocol continuity.

    Caller still performs authoritative validate_action() immediately before
    sending the next bridge action.

    ``acceptance`` is the AUTHORITATIVE game-side outcome of the command that
    produced ``after`` (see :class:`ActionAcceptance`). It is consumed HERE,
    before any executor mutation, so an action-relevantly unchanged snapshot
    is classified correctly and the executor never irreversibly resets a plan
    that is merely waiting for the visible world to advance:

        ACCEPTED + unchanged -> AWAITING_ADVANCE  (retain, wait)
        REJECTED + unchanged -> ACTION_REJECTED   (reset, re-decide)
        UNKNOWN  + unchanged -> ADVANCE_UNVERIFIED (conservative bounded wait)
    """
    delta = diff_states(before, after)
    stype = str(after.get("type", ""))

    if stype in TERMINAL_TYPES:
        return (
            CheckpointDecision(True, CheckpointReason.TERMINAL, stype),
            delta,
        )

    if delta.action_relevant_same:
        # A normal card play should remove/move the card or otherwise change
        # action-relevant visible state. Replaying blindly is more dangerous
        # than a conservative checkpoint -- but only for a command the game
        # did NOT accept. An accepted command whose visible effect has not
        # surfaced yet is a genuine WAITING state; an unverifiable one is a
        # conservative protocol-uncertain wait (never invented as either).
        if acceptance is ActionAcceptance.ACCEPTED:
            return (
                CheckpointDecision(
                    True,
                    CheckpointReason.AWAITING_ADVANCE,
                    "game accepted the command; authoritative state has"
                    " not advanced yet",
                ),
                delta,
            )
        if acceptance is ActionAcceptance.REJECTED:
            return (
                CheckpointDecision(
                    True,
                    CheckpointReason.ACTION_REJECTED,
                    "game rejected the command; state is unchanged",
                ),
                delta,
            )
        return (
            CheckpointDecision(
                True,
                CheckpointReason.ADVANCE_UNVERIFIED,
                "no authoritative acceptance signal for an unchanged state",
            ),
            delta,
        )

    # COMBAT RESOLVED lifecycle invariant: if the bridge still shows a
    # combat_action screen but NO enemy is alive, the combat is over. The
    # remaining chunk (potion, extra attacks, end_turn) must NEVER execute
    # against a dead board -- re-inspect at the next real screen instead.
    if stype == "combat_action":
        enemies = [
            e for e in (after.get("enemies") or [])
            if isinstance(e, dict)
        ]
        if enemies and not any(e.get("is_alive", False) for e in enemies):
            return (
                CheckpointDecision(
                    True,
                    CheckpointReason.COMBAT_RESOLVED,
                    "all enemies are dead; combat resolved",
                ),
                delta,
            )

    if executed_action.checkpoint_after:
        return (
            CheckpointDecision(
                True,
                CheckpointReason.MODEL_REQUESTED,
                "model explicitly requested inspect after this action",
            ),
            delta,
        )

    if delta.screen_changed:
        return (
            CheckpointDecision(
                True,
                CheckpointReason.SCREEN_CHANGED,
                f"{before.get('type')} -> {after.get('type')}",
            ),
            delta,
        )

    if delta.round_changed:
        return (
            CheckpointDecision(
                True,
                CheckpointReason.NEW_TURN,
                f"{before.get('round')} -> {after.get('round')}",
            ),
            delta,
        )

    # This is the key epistemic boundary for combat: genuinely new visible
    # hand information was introduced.  Removed cards alone are expected.
    if delta.hand_added_information:
        return (
            CheckpointDecision(
                True,
                CheckpointReason.HAND_CHANGED,
                "new card(s) became visible in hand: " + ", ".join(delta.added_cards),
            ),
            delta,
        )

    ref_check = _next_ref_resolvable(chunk, next_action, after)
    if ref_check.needed:
        return ref_check, delta

    return CheckpointDecision(False, CheckpointReason.NONE), delta
