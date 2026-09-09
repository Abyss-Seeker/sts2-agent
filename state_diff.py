"""Objective state-diff helpers for the beta ActionChunk runtime.

No strategy is encoded here.  Diffs are derived from human-visible bridge
fields only and are used for:
- conservative checkpoint detection,
- compact model re-observation,
- audit logs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from typing import Any

from action_plan import card_identity


@dataclass(frozen=True)
class StateDelta:
    screen_changed: bool
    round_changed: bool
    added_cards: tuple[str, ...]
    removed_cards: tuple[str, ...]
    player_changes: tuple[str, ...]
    enemy_changes: tuple[str, ...]
    potion_changes: tuple[str, ...]
    action_relevant_same: bool

    @property
    def hand_added_information(self) -> bool:
        return bool(self.added_cards)


def _card_label(card: dict[str, Any]) -> str:
    name = str(card.get("display_name") or card.get("name") or "")
    cid = str(card.get("id") or card.get("card_id") or "?")
    up = "+" if card.get("upgraded") else ""
    return f"{name or cid}{up}"


def _hand_counter(state: dict[str, Any]) -> tuple[Counter, dict[tuple[Any, ...], str]]:
    counter: Counter = Counter()
    labels: dict[tuple[Any, ...], str] = {}
    for raw in state.get("hand") or []:
        if not isinstance(raw, dict):
            continue
        key = card_identity(raw).key()
        counter[key] += 1
        labels.setdefault(key, _card_label(raw))
    return counter, labels


def _player_subset(state: dict[str, Any]) -> dict[str, Any]:
    p = state.get("player") or {}
    return {
        "hp": p.get("hp"),
        "max_hp": p.get("max_hp"),
        "block": p.get("block"),
        "energy": p.get("energy"),
        "max_energy": p.get("max_energy"),
        "gold": p.get("gold"),
        "powers": p.get("powers"),
        "visible_character_state": p.get("visible_character_state"),
    }


def _enemy_subset(state: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in state.get("enemies") or []:
        if not isinstance(e, dict):
            continue
        out.append({
            "id": e.get("id"),
            "display_name": e.get("display_name") or e.get("name"),
            "hp": e.get("hp"),
            "max_hp": e.get("max_hp"),
            "block": e.get("block"),
            "is_alive": e.get("is_alive"),
            "intent": e.get("intent"),
            "intent_damage": e.get("intent_damage"),
            "intent_hits": e.get("intent_hits"),
            "powers": e.get("powers"),
        })
    return out


def _potion_subset(state: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in state.get("potions") or []:
        if not isinstance(p, dict):
            continue
        out.append({
            "slot": p.get("slot"),
            "id": p.get("id"),
            "empty": p.get("empty"),
            "can_use": p.get("can_use"),
            "requires_target": p.get("requires_target"),
        })
    return out


def action_relevant_fingerprint(state: dict[str, Any]) -> str:
    """Fingerprint fields whose equality strongly suggests no action occurred.

    request_id is deliberately excluded because the bridge can issue a new
    request for an unchanged state after rejecting an action.
    """
    payload = {
        "type": state.get("type"),
        "round": state.get("round"),
        "player": _player_subset(state),
        "hand": [
            {
                "identity": card_identity(c).key(),
                "playable": c.get("playable"),
                "cost": c.get("current_energy_cost", c.get("cost")),
            }
            for c in (state.get("hand") or [])
            if isinstance(c, dict)
        ],
        "enemies": _enemy_subset(state),
        "potions": _potion_subset(state),
        "draw_pile_count": state.get("draw_pile_count"),
        "discard_pile_count": state.get("discard_pile_count"),
        "exhaust_pile_count": state.get("exhaust_pile_count"),
    }
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def diff_states(before: dict[str, Any], after: dict[str, Any]) -> StateDelta:
    before_counter, before_labels = _hand_counter(before)
    after_counter, after_labels = _hand_counter(after)

    added: list[str] = []
    removed: list[str] = []
    for key, count in (after_counter - before_counter).items():
        added.extend([after_labels.get(key, str(key))] * count)
    for key, count in (before_counter - after_counter).items():
        removed.extend([before_labels.get(key, str(key))] * count)

    p0, p1 = _player_subset(before), _player_subset(after)
    player_changes: list[str] = []
    for key in ("hp", "max_hp", "block", "energy", "max_energy", "gold"):
        if p0.get(key) != p1.get(key):
            player_changes.append(f"{key}: {p0.get(key)} -> {p1.get(key)}")
    if p0.get("powers") != p1.get("powers"):
        player_changes.append("powers changed")
    if p0.get("visible_character_state") != p1.get("visible_character_state"):
        player_changes.append("visible_character_state changed")

    e0, e1 = _enemy_subset(before), _enemy_subset(after)
    enemy_changes: list[str] = []
    n = max(len(e0), len(e1))
    for i in range(n):
        old = e0[i] if i < len(e0) else None
        new = e1[i] if i < len(e1) else None
        if old == new:
            continue
        if old is None:
            enemy_changes.append(f"enemy[{i}] appeared: {new}")
            continue
        if new is None:
            enemy_changes.append(f"enemy[{i}] disappeared: {old}")
            continue
        oid = old.get("display_name") or old.get("id") or "?"
        for key in ("hp", "block", "is_alive", "intent", "intent_damage", "intent_hits"):
            if old.get(key) != new.get(key):
                enemy_changes.append(
                    f"enemy[{i}] {oid} {key}: {old.get(key)} -> {new.get(key)}"
                )
        if old.get("powers") != new.get("powers"):
            enemy_changes.append(f"enemy[{i}] {oid} powers changed")

    pot0, pot1 = _potion_subset(before), _potion_subset(after)
    potion_changes: list[str] = []
    if pot0 != pot1:
        potion_changes.append("potion belt changed")

    return StateDelta(
        screen_changed=str(before.get("type")) != str(after.get("type")),
        round_changed=before.get("round") != after.get("round"),
        added_cards=tuple(added),
        removed_cards=tuple(removed),
        player_changes=tuple(player_changes),
        enemy_changes=tuple(enemy_changes),
        potion_changes=tuple(potion_changes),
        action_relevant_same=(
            action_relevant_fingerprint(before) == action_relevant_fingerprint(after)
        ),
    )


def render_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    executed: str = "",
    checkpoint_reason: str = "",
) -> str:
    """Compact, objective model update.  Falls back to facts, never strategy."""
    delta = diff_states(before, after)
    lines = ["== CHECKPOINT UPDATE =="]
    if checkpoint_reason:
        lines.append(f"Reason: {checkpoint_reason}")
    if executed:
        lines.append(f"Just executed: {executed}")

    if delta.screen_changed:
        lines.append(
            f"Screen: {before.get('type', '?')} -> {after.get('type', '?')}"
        )
    if delta.round_changed:
        lines.append(f"Round: {before.get('round', '?')} -> {after.get('round', '?')}")

    if delta.added_cards:
        lines.append("New cards now visible in hand:")
        lines.extend(f"  + {x}" for x in delta.added_cards)
    if delta.removed_cards:
        lines.append("Cards no longer in hand:")
        lines.extend(f"  - {x}" for x in delta.removed_cards)

    if delta.player_changes:
        lines.append("Player changes:")
        lines.extend(f"  - {x}" for x in delta.player_changes)
    if delta.enemy_changes:
        lines.append("Enemy changes:")
        lines.extend(f"  - {x}" for x in delta.enemy_changes)
    if delta.potion_changes:
        lines.append("Potion changes:")
        lines.extend(f"  - {x}" for x in delta.potion_changes)

    if len(lines) <= 3:
        lines.append("No decision-relevant visible delta was detected.")
    lines.append(
        "Everything not listed above should be treated as unchanged from the "
        "previous model observation. If any doubt remains, request/use a FULL state."
    )
    return "\n".join(lines)
