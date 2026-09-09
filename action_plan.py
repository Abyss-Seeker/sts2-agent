"""Pure data model + parsing/resolution helpers for STS2 LLM ActionChunks.

This module is intentionally strategy-free.  It does not decide what to play.
It only:
- assigns plan-scoped human-readable refs to the CURRENT combat state;
- parses a model-produced action chunk;
- remembers what each ref meant when the plan was created;
- resolves those refs against later authoritative bridge states.

It has no dependency on agent.py so it can be unit-tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import uuid
from typing import Any


class PlanParseError(ValueError):
    """The model output cannot be represented as a safe ActionChunk."""


class ResolveError(RuntimeError):
    """A plan-scoped symbolic ref can no longer be resolved safely."""


class ActionKind(str, Enum):
    PLAY = "play"
    POTION = "potion"
    END_TURN = "end_turn"


@dataclass(frozen=True)
class CardIdentity:
    """Human-visible identity fields that survive hand index shifts.

    Deliberately excludes current cost/playable/index because those can change
    dynamically.  This is not a hidden engine-object id.
    """

    card_id: str
    display_name: str
    upgraded: bool
    upgrade_level: int | None
    enchantment: str
    affliction: str

    def key(self) -> tuple[Any, ...]:
        return (
            self.card_id,
            self.display_name,
            self.upgraded,
            self.upgrade_level,
            self.enchantment,
            self.affliction,
        )


@dataclass(frozen=True)
class EnemyIdentity:
    enemy_id: str
    display_name: str
    original_index: int

    def visible_key(self) -> tuple[str, str]:
        return (self.enemy_id, self.display_name)


@dataclass(frozen=True)
class CardRef:
    ref: str
    identity: CardIdentity
    original_index: int


@dataclass(frozen=True)
class EnemyRef:
    ref: str
    identity: EnemyIdentity


@dataclass(frozen=True)
class PlannedAction:
    kind: ActionKind
    card_ref: str | None = None
    target_ref: str | None = None
    potion_slot: int | None = None
    checkpoint_after: bool = False
    note: str = ""


@dataclass(frozen=True)
class ActionChunk:
    """One model commitment spanning zero or more bridge handshakes."""

    plan_id: str
    summary: str
    actions: tuple[PlannedAction, ...]
    source_request_id: str | None
    source_round: int | None
    card_refs: dict[str, CardRef] = field(default_factory=dict)
    enemy_refs: dict[str, EnemyRef] = field(default_factory=dict)
    memory_note: str = ""

    @property
    def ends_turn(self) -> bool:
        return bool(self.actions and self.actions[-1].kind is ActionKind.END_TURN)


def _modifier_label(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("id") or value.get("name") or value.get("display_name") or "")
    if value in (None, ""):
        return ""
    return str(value)


def card_identity(card: dict[str, Any]) -> CardIdentity:
    raw_level = card.get("upgrade_level")
    try:
        level = int(raw_level) if raw_level not in (None, "") else None
    except (TypeError, ValueError):
        level = None
    return CardIdentity(
        card_id=str(card.get("id") or card.get("card_id") or ""),
        display_name=str(card.get("display_name") or card.get("name") or ""),
        upgraded=bool(card.get("upgraded")),
        upgrade_level=level,
        enchantment=_modifier_label(card.get("enchantment")),
        affliction=_modifier_label(card.get("affliction")),
    )


def enemy_identity(enemy: dict[str, Any], index: int) -> EnemyIdentity:
    return EnemyIdentity(
        enemy_id=str(enemy.get("id") or enemy.get("enemy_id") or ""),
        display_name=str(enemy.get("display_name") or enemy.get("name") or ""),
        original_index=index,
    )


def build_card_refs(state: dict[str, Any]) -> dict[str, CardRef]:
    refs: dict[str, CardRef] = {}
    for index, raw in enumerate(state.get("hand") or []):
        if not isinstance(raw, dict):
            continue
        ref = f"h{index}"
        refs[ref] = CardRef(ref=ref, identity=card_identity(raw), original_index=index)
    return refs


def build_enemy_refs(state: dict[str, Any]) -> dict[str, EnemyRef]:
    refs: dict[str, EnemyRef] = {}
    for index, raw in enumerate(state.get("enemies") or []):
        if not isinstance(raw, dict):
            continue
        ref = f"e{index}"
        refs[ref] = EnemyRef(ref=ref, identity=enemy_identity(raw, index))
    return refs


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "yes", "1"}:
            return True
        if s in {"false", "no", "0"}:
            return False
    raise PlanParseError(f"expected boolean, got {value!r}")


def _as_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or value is None:
        raise PlanParseError(f"{field_name} must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise PlanParseError(f"{field_name} must be an integer, got {value!r}")


def _normalize_kind(raw: Any) -> ActionKind:
    text = str(raw or "").strip().lower()
    aliases = {
        "play_card": "play",
        "card": "play",
        "use_potion": "potion",
        "end": "end_turn",
        "endturn": "end_turn",
    }
    text = aliases.get(text, text)
    try:
        return ActionKind(text)
    except ValueError as exc:
        raise PlanParseError(f"unknown planned action kind: {raw!r}") from exc


def parse_action_chunk(
    obj: dict[str, Any],
    source_state: dict[str, Any],
    *,
    max_actions: int = 16,
) -> ActionChunk:
    """Parse a model JSON object into a conservative, plan-scoped chunk.

    Structural validation only.  Future energy/damage is not simulated;
    current legality is rechecked against the authoritative bridge state
    immediately before every action.
    """
    if not isinstance(obj, dict):
        raise PlanParseError("ActionChunk response must be a JSON object")

    raw_actions = obj.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise PlanParseError("'actions' must be a non-empty list")
    if len(raw_actions) > max_actions:
        raise PlanParseError(
            f"too many actions in one chunk ({len(raw_actions)} > {max_actions})"
        )

    card_refs = build_card_refs(source_state)
    enemy_refs = build_enemy_refs(source_state)
    parsed: list[PlannedAction] = []
    used_card_refs: set[str] = set()

    for index, raw in enumerate(raw_actions):
        if not isinstance(raw, dict):
            raise PlanParseError(f"actions[{index}] must be an object")
        kind = _normalize_kind(raw.get("kind", raw.get("action")))
        checkpoint_after = _as_bool(raw.get("checkpoint_after"), default=False)
        note = str(raw.get("note") or "")[:240]

        if kind is ActionKind.PLAY:
            ref = str(raw.get("card_ref") or "").strip()
            if not ref:
                raise PlanParseError(f"actions[{index}] play is missing card_ref")
            if ref not in card_refs:
                raise PlanParseError(
                    f"actions[{index}] card_ref {ref!r} does not exist in source hand"
                )
            if ref in used_card_refs:
                raise PlanParseError(
                    f"actions[{index}] reuses {ref!r}; repeated use of one "
                    "plan-scoped card ref is unsafe. Stop and inspect instead."
                )
            used_card_refs.add(ref)

            source_card = (source_state.get("hand") or [])[card_refs[ref].original_index]
            target_kind = str((source_card or {}).get("target", "None"))
            raw_target = raw.get("target_ref")
            target_ref = (
                str(raw_target).strip()
                if raw_target not in (None, "", -1, "-1")
                else None
            )
            if target_kind == "AnyEnemy":
                if not target_ref:
                    raise PlanParseError(
                        f"actions[{index}] card {ref} requires target_ref"
                    )
                if target_ref not in enemy_refs:
                    raise PlanParseError(
                        f"actions[{index}] target_ref {target_ref!r} is unknown"
                    )
            elif target_ref and target_ref not in enemy_refs:
                raise PlanParseError(
                    f"actions[{index}] target_ref {target_ref!r} is unknown"
                )

            parsed.append(
                PlannedAction(
                    kind=kind,
                    card_ref=ref,
                    target_ref=target_ref,
                    checkpoint_after=checkpoint_after,
                    note=note,
                )
            )

        elif kind is ActionKind.POTION:
            slot = _as_int(raw.get("potion_slot", raw.get("slot")), "potion_slot")
            raw_target = raw.get("target_ref")
            target_ref = (
                str(raw_target).strip()
                if raw_target not in (None, "", -1, "-1")
                else None
            )
            if target_ref and target_ref not in enemy_refs:
                raise PlanParseError(
                    f"actions[{index}] target_ref {target_ref!r} is unknown"
                )
            parsed.append(
                PlannedAction(
                    kind=kind,
                    potion_slot=slot,
                    target_ref=target_ref,
                    checkpoint_after=checkpoint_after,
                    note=note,
                )
            )

        else:  # END_TURN
            if index != len(raw_actions) - 1:
                raise PlanParseError("end_turn must be the final action in a chunk")
            if checkpoint_after:
                raise PlanParseError("end_turn cannot also request checkpoint_after")
            parsed.append(PlannedAction(kind=kind, note=note))

        if checkpoint_after and index != len(raw_actions) - 1:
            raise PlanParseError(
                "checkpoint_after=true means inspect after this action; "
                "there must be no later action in the same chunk"
            )

    thought = str(obj.get("thought") or obj.get("summary") or "").strip()
    if not thought:
        raise PlanParseError("missing non-empty 'thought' summary")

    round_value = source_state.get("round")
    try:
        source_round = int(round_value) if round_value is not None else None
    except (TypeError, ValueError):
        source_round = None

    return ActionChunk(
        plan_id=f"plan_{uuid.uuid4().hex[:10]}",
        summary=thought[:1000],
        actions=tuple(parsed),
        source_request_id=(
            str(source_state.get("request_id"))
            if source_state.get("request_id") is not None
            else None
        ),
        source_round=source_round,
        card_refs=card_refs,
        enemy_refs=enemy_refs,
        memory_note=str(obj.get("memory_note") or "")[:1000],
    )


def resolve_card_ref(ref: CardRef, state: dict[str, Any]) -> int:
    """Resolve a source-plan card ref to CURRENT hand index.

    Identical visible duplicates are equivalent at this abstraction layer.
    If no visible match remains, resolution fails; caller must checkpoint.
    """
    hand = state.get("hand") or []
    matches: list[int] = []
    for i, raw in enumerate(hand):
        if isinstance(raw, dict) and card_identity(raw).key() == ref.identity.key():
            matches.append(i)

    if not matches:
        raise ResolveError(f"card ref {ref.ref} no longer exists in current hand")
    if ref.original_index in matches:
        return ref.original_index
    return matches[0]


def resolve_enemy_ref(ref: EnemyRef, state: dict[str, Any]) -> int:
    """Resolve target conservatively; never silently retarget a dead enemy."""
    enemies = state.get("enemies") or []
    original = ref.identity.original_index

    if 0 <= original < len(enemies):
        raw = enemies[original]
        if isinstance(raw, dict):
            ident = enemy_identity(raw, original)
            if ident.visible_key() == ref.identity.visible_key():
                if raw.get("is_alive", False):
                    return original
                raise ResolveError(f"target {ref.ref} is dead")

    matches: list[int] = []
    for i, raw in enumerate(enemies):
        if not isinstance(raw, dict) or not raw.get("is_alive", False):
            continue
        ident = enemy_identity(raw, i)
        if ident.visible_key() == ref.identity.visible_key():
            matches.append(i)

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ResolveError(f"target {ref.ref} no longer exists/alive")
    raise ResolveError(
        f"target {ref.ref} became ambiguous: {len(matches)} alive visible matches"
    )


def chunk_to_jsonable(chunk: ActionChunk) -> dict[str, Any]:
    return {
        "plan_id": chunk.plan_id,
        "thought": chunk.summary,
        "actions": [
            {
                "kind": action.kind.value,
                **({"card_ref": action.card_ref} if action.card_ref else {}),
                **({"target_ref": action.target_ref} if action.target_ref else {}),
                **(
                    {"potion_slot": action.potion_slot}
                    if action.potion_slot is not None
                    else {}
                ),
                **(
                    {"checkpoint_after": True}
                    if action.checkpoint_after
                    else {}
                ),
                **({"note": action.note} if action.note else {}),
            }
            for action in chunk.actions
        ],
        **({"memory_note": chunk.memory_note} if chunk.memory_note else {}),
    }


def compact_ref_legend(state: dict[str, Any]) -> str:
    """Small prompt addendum; formatter can later inline refs directly."""
    lines = ["PLAN-SCOPED REFERENCES (use these in ActionChunk):"]
    hand = state.get("hand") or []
    for ref, item in build_card_refs(state).items():
        card = hand[item.original_index] if item.original_index < len(hand) else {}
        label = (
            str((card or {}).get("display_name") or "")
            or str((card or {}).get("id") or "?")
        )
        lines.append(f"  {ref} = HAND[{item.original_index}] {label}")

    enemies = state.get("enemies") or []
    for ref, item in build_enemy_refs(state).items():
        idx = item.identity.original_index
        enemy = enemies[idx] if idx < len(enemies) else {}
        label = (
            str((enemy or {}).get("display_name") or "")
            or str((enemy or {}).get("id") or "?")
        )
        alive = "alive" if (enemy or {}).get("is_alive", False) else "DEAD"
        lines.append(f"  {ref} = ENEMY[{idx}] {label} ({alive})")
    return "\n".join(lines)
