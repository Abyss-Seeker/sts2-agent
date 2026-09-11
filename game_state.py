"""Game-state formatting: convert bridge JSON states into rich text that a
knowledge-free LLM can reason about, plus a persistent run-level memory.

CORE PRINCIPLES
---------------
1. HUMAN-VISIBLE INFORMATION PARITY: everything a human player can inspect
   in the current UI (without committing an action) must appear in the LLM
   text; nothing a human cannot know may appear (no draw-pile order, no RNG,
   no hidden room types behind '?', no future enemy moves, no hidden event
   outcomes).
2. SOURCE OF TRUTH: current runtime values outrank static reference text.
   Static KB text (from the current game build) is only a fallback for
   fields the bridge cannot serialize yet, and is labeled as static.
3. Every request is SELF-CONTAINED: card definitions are re-sent whenever
   they are relevant; the LLM never needs to remember earlier requests.
4. The CURRENT state is never character-truncated. Only history is trimmed.
"""

from __future__ import annotations

import logging

from typing import Any

from bridge_client import BridgeStateType
from knowledge import (
    INTENT_GLOSSARY,
    KEYWORD_GLOSSARY,
    card_full,
    card_line,
    power_text,
    relic_text,
)

CHOICE_TYPES = {
    BridgeStateType.CARD_SELECT,
    BridgeStateType.MAP_SELECT,
    BridgeStateType.REWARD_SCREEN,
    BridgeStateType.CARD_BUNDLE,
    BridgeStateType.CRYSTAL_SPHERE,
    BridgeStateType.CARD_REWARD,
    BridgeStateType.REST_SITE,
    BridgeStateType.SHOP,
    BridgeStateType.EVENT,
    BridgeStateType.TREASURE,
    BridgeStateType.BOSS_RELIC,
}

# States the agent is expected to DECIDE on. Anything decision-worthy that is
# not in COMBAT_ACTION / CHOICE_TYPES / TERMINAL goes through the generic
# unknown-state fallback formatter (never silently dropped, never guessed).
DECISION_TYPES = {BridgeStateType.COMBAT_ACTION} | CHOICE_TYPES

CHOICE_TYPE_TITLES = {
    BridgeStateType.MAP_SELECT: "MAP: choose the next room to visit",
    BridgeStateType.CARD_REWARD: "CARD REWARD: pick one card to ADD to your deck (or skip if allowed)",
    BridgeStateType.CARD_BUNDLE: "CARD BUNDLE REWARD: pick one bundle to ADD",
    BridgeStateType.REWARD_SCREEN: "REWARDS SCREEN: claim rewards, then pick the 'proceed' option",
    BridgeStateType.SHOP: "SHOP: spend gold on cards / relics / potions / services",
    BridgeStateType.REST_SITE: "REST SITE: choose one option (rest / smith / relic-granted options)",
    BridgeStateType.EVENT: "EVENT: a narrative event; pick one option",
    BridgeStateType.TREASURE: "TREASURE: take the relic from the opened chest",
    BridgeStateType.BOSS_RELIC: "BOSS RELIC: pick one relic",
    BridgeStateType.CRYSTAL_SPHERE: "CRYSTAL SPHERE: pick an option",
    BridgeStateType.CARD_SELECT: "CARD SELECT: select card(s) as instructed by the prompt",
}

TARGET_TYPE_HINTS = {
    "AnyEnemy": "requires choosing one living enemy as target (target_index)",
    "AllEnemies": "hits all enemies, no target needed (target_index=-1)",
    "Self": "targets yourself, no enemy target needed (target_index=-1)",
    "None": "no target needed (target_index=-1)",
    "RandomEnemy": "hits a random enemy, no target needed (target_index=-1)",
    "AnyAlly": "targets an ally (use target_index=-1 for yourself)",
    "AllAllies": "hits all allies, no target needed (target_index=-1)",
}


# ----------------------------------------------------------------
# Shared pieces
# ----------------------------------------------------------------

def _map_room_type(raw: Any) -> str:
    """Map a bridge room type to the human-visible label.

    'Unknown' nodes are shown as '?' in the game UI -- they MUST stay
    UNKNOWN here. Never substitute the backend's real room type (the
    bridge never sends it for these nodes anyway).
    """
    t = str(raw or "?")
    if t.lower() == "unknown":
        return "UNKNOWN (hidden room type, shown as '?' in the game)"
    return t.upper()


def _fmt_powers(powers: list[dict[str, Any]] | None, indent: str = "  ") -> list[str]:
    """Powers: runtime tooltip (the game's own HoverTips pipeline) first;
    the hardcoded glossary is only a fallback. Hidden powers were already
    filtered out by the bridge (IsVisible == false never serialized)."""
    if not powers:
        return []
    lines = []
    for i, p in enumerate(powers):
        pid = str(p.get("id", "?")).upper()
        name = str(p.get("name", "") or "")
        amount = p.get("amount", 0)
        display_amount = p.get("display_amount")
        amount_s = f"{amount}"
        if display_amount is not None and str(display_amount) != str(amount):
            amount_s = f"{amount} (displayed: {display_amount})"
        runtime_desc = str(p.get("description", "") or "")
        head = f"{name} ({pid})" if name else pid
        debuff = " [debuff]" if p.get("debuff") else ""
        lines.append(f"{indent}POWER[{i}] {head} {amount_s}{debuff}")
        if runtime_desc:
            lines.append(f"{indent}  Effect: {runtime_desc}")
        else:
            desc = power_text(pid)
            if desc:
                lines.append(f"{indent}  Effect (static reference): {desc}")
    return lines


def _fmt_relics(relics: list[Any] | None, indent: str = "  ") -> list[str]:
    """Full relic entries. Runtime tooltip (the game's own rendered relic
    description) always wins; the decompiled static reference is only a
    fallback for relics the bridge cannot describe yet. Only UI-visible
    counters are ever serialized."""
    if not relics:
        return [f"{indent}(none)"]
    lines = []
    for r in relics:
        if isinstance(r, dict):
            rid = str(r.get("id", "?"))
            extra = ""
            if r.get("counter") not in (None, 0):
                extra += f" | visible counter: {r['counter']}"
            if r.get("used_up"):
                extra += " | state: USED UP"
            name = str(r.get("name", "") or "")
            head = f"{name} ({rid})" if name else rid
            lines.append(f"{indent}{head}{extra}")
            runtime_desc = str(r.get("description", "") or "")
            if runtime_desc:
                lines.append(f"{indent}  Effect: {runtime_desc}")
            else:
                desc = relic_text(rid)
                if desc:
                    body = desc.splitlines()
                    lines.append(f"{indent}  Effect (static reference): {body[0]}")
                    for extra_line in body[1:]:
                        lines.append(f"{indent}    {extra_line}")
        else:
            lines.append(f"{indent}{r}")
    return lines


def _fmt_potion_slots(
    potions: list[Any] | None,
    capacity: int | None = None,
) -> list[str]:
    """Render every potion slot incl. EMPTY ones (humans see the whole belt).

    The bridge reports `potion_slot_capacity` and serializes empty slots
    explicitly. For backwards compatibility, empty slots are also derived
    from the reported capacity when slot entries are missing.
    """
    lines = ["POTION BELT (empty slots shown as EMPTY; potions do NOT cost energy):",
             'On ANY screen, when Usable now=yes: {"action":"potion","slot":N,"target_index":-1}.',
             'When Discardable now=yes: {"action":"discard_potion","slot":N}. Discard destroys it without its effect.',
             'These inventory actions preserve the current room choice; inspect the refreshed state afterward.']
    if not potions and not capacity:
        lines.append("  (belt empty or not reported)")
        return lines
    slots: dict[int, dict[str, Any]] = {}
    for p in potions or []:
        if isinstance(p, dict) and isinstance(p.get("slot"), int):
            slots[p["slot"]] = p
        elif isinstance(p, dict):
            slots[len(slots)] = p
    max_known = max(slots) + 1 if slots else 0
    total = capacity if capacity is not None else max_known
    for i in range(total):
        p = slots.get(i)
        if p is None or p.get("empty"):
            lines.append(f"  POTION[{i}]: EMPTY")
            continue
        pid = str(p.get("id", "?"))
        name = str(p.get("name", "") or "")
        head = f"{name} ({pid})" if name else pid
        usable = "yes" if p.get("can_use", False) else "NO"
        reason = ""
        if not p.get("can_use", False) and str(p.get("usage", "")).lower() == "automatic":
            reason = " (auto/trigger potion, not manually usable)"
        target = str(p.get("target", p.get("target_type", "Self")))
        lines.append(f"  POTION[{i}] {head}")
        lines.append(f"    Usable now: {usable}{reason} | Target: {target}")
        lines.append("    Discardable now: " + ("yes" if p.get("can_discard", False) else "NO"))
        effect = str(p.get("effect", "") or "")
        if effect:
            lines.append(f"    Effect: {effect}")
    if capacity is not None and max_known < capacity:
        pass  # empty slots already rendered from capacity
    return lines


def _card_group_label(entry: dict[str, Any]) -> str:
    """Group label preserving modifier identity: two cards the player could
    tell apart in the pile viewer (upgrade / enchantment / affliction) are
    never merged."""
    label = str(entry.get("id", "?"))
    if entry.get("upgraded"):
        label += "+"
    if entry.get("enchantment"):
        label += f" [Ench: {entry['enchantment']}]"
    if entry.get("affliction"):
        label += f" [Affliction: {entry['affliction']}]"
    return label


def _fmt_deck_groups(deck: list[Any]) -> tuple[list[str], list[str]]:
    """Render a deck composition.

    Returns (group_lines, definition_lines): the grouped 'Nx ID(+)' list and
    one full static definition per distinct group. Cards that differ in
    upgrade state, enchantment or affliction are NEVER merged.
    """
    groups: list[tuple[str, bool, int, dict[str, Any]]] = []
    for entry in deck:
        if isinstance(entry, dict):
            groups.append((
                str(entry.get("id", "?")),
                bool(entry.get("upgraded")),
                int(entry.get("count", 1)),
                entry,
            ))
        else:
            groups.append((str(entry), False, 1, {"id": str(entry)}))
    list_bits = [f"{n}x {_card_group_label(e)}" for _cid, _up, n, e in groups]
    lines = [", ".join(list_bits)]
    defs: list[str] = []
    seen: set[str] = set()
    for cid, up, _n, e in groups:
        label = _card_group_label(e)
        if label in seen:
            continue
        seen.add(label)
        full = card_full(cid, upgraded=up)
        if full:
            defs.append(f"  {label}: {full}")
    return lines, defs


def _visible_character_state_lines(data: dict[str, Any], indent: str = "  ") -> list[str]:
    """Render ONLY the explicit human-visible namespace
    (`visible_character_state`). The bridge decides what enters it; Python
    preserves semantics. Known entries get dedicated formatting (Stars,
    ordered ORB slots, Osty); everything else in the namespace is rendered
    as key=value so future character mechanics stay visible."""
    lines: list[str] = []

    if "stars" in data:
        lines.append(f"{indent}Stars: {data['stars']}")
    if "focus" in data:
        lines.append(f"{indent}Focus: {data['focus']}")
    if "orb_slots" in data:
        lines.append(f"{indent}ORB SLOTS (in UI order, left to right):")
        for orb in data["orb_slots"]:
            if not isinstance(orb, dict):
                continue
            slot = orb.get("slot", "?")
            if orb.get("empty"):
                lines.append(f"{indent}  ORB[{slot}]: EMPTY")
            else:
                lines.append(
                    f"{indent}  ORB[{slot}]: {orb.get('type', '?')}"
                    f" | Passive: {orb.get('passive', '?')} | Evoke: {orb.get('evoke', '?')}"
                )
    if "osty" in data and isinstance(data["osty"], dict):
        osty = data["osty"]
        present = bool(osty.get("present"))
        alive = bool(osty.get("alive"))
        lines.append(f"{indent}OSTY: {'present' if present else 'not summoned'}"
                     f" | Alive: {'yes' if alive else 'NO'}")
        if present:
            lines.append(
                f"{indent}  HP: {osty.get('hp', '?')}/{osty.get('max_hp', '?')}"
                f" | Block: {osty.get('block', 0)}"
            )
            for p in osty.get("powers", []) or []:
                if isinstance(p, dict):
                    desc = power_text(str(p.get("id", "")))
                    lines.append(
                        f"{indent}  Power: {p.get('id', '?')} {p.get('amount', 0)}"
                        + (f" -- {desc}" if desc else "")
                    )

    # Generic passthrough WITHIN the explicit visible namespace only.
    _known = {"stars", "focus", "orb_slots", "orb_slot_capacity", "osty"}
    for key, val in data.items():
        if key in _known or key.startswith("_"):
            continue
        lines.append(f"{indent}{key} = {val}")
    return lines


def _visible_namespace(state_or_player: dict[str, Any]) -> dict[str, Any]:
    """Return the explicit `visible_character_state` namespace, if present.
    NEVER reads arbitrary unknown fields from the state/player dict."""
    vcs = state_or_player.get("visible_character_state")
    return vcs if isinstance(vcs, dict) else {}


def _character_label(player: dict[str, Any]) -> str:
    """Character identity as reported by the bridge (runtime, visible on
    every UI screen). Falls back to '(not reported by bridge)' only while
    the running mod build predates the field."""
    cid = str(player.get("character_id", "") or "")
    cname = str(player.get("character_name", "") or "")
    if cname and cid:
        return f"{cname.upper()} ({cid})"
    if cid:
        return cid.upper()
    if cname:
        return cname
    return "(not reported by bridge)"


def _run_header(
    state: dict[str, Any],
    screen: str,
    player: dict[str, Any],
    extra: list[str] | None = None,
    potions: list[Any] | None = None,
) -> list[str]:
    """COMMON RUN HEADER: shown on every decision screen, exactly the
    top-bar / deck-viewer info a human always has access to."""
    hp = player.get("hp")
    max_hp = player.get("max_hp")
    lines = [
        "== RUN STATUS ==",
        f"Character: {_character_label(player)}",
        f"Ascension: {state.get('ascension', player.get('ascension', '?'))}",
        f"Act: {state.get('act', '?')}",
        f"Floor: {state.get('floor', '?')}",
        f"Current screen: {screen.upper()}",
    ]
    if hp is not None or max_hp is not None:
        lines.append(f"HP: {hp}/{max_hp}")
    if player.get("gold") is not None:
        lines.append(f"Gold: {player['gold']}")
    if player.get("block") is not None:
        lines.append(f"Block: {player['block']}")
    if player.get("energy") is not None:
        lines.append(
            f"Energy: {player['energy']}/{player.get('max_energy', '?')}"
        )
    if extra:
        lines.extend(extra)
    # Act identity + bosses (map-screen-visible information).
    act_info = state.get("act_info")
    if isinstance(act_info, dict):
        area = act_info.get("act_title", "")
        area_id = act_info.get("act_id", "")
        if area or area_id:
            label = area or area_id
            lines.append(f"Act area: {label}" + (f" ({area_id})" if area and area_id else ""))
        bosses = act_info.get("bosses")
        if isinstance(bosses, list) and bosses:
            lines.append("CURRENT ACT BOSS (shown on the map screen):")
            for i, b in enumerate(bosses):
                if isinstance(b, dict):
                    name = b.get("display_name") or b.get("id", "?")
                    lines.append(f"  BOSS[{i}] {name}")
    lines.append("")
    lines.append("RELICS:")
    lines.extend(_fmt_relics(player.get("relics")))
    lines.append("")
    lines.extend(_fmt_potion_slots(
        potions if potions is not None else player.get("potions"),
        capacity=player.get("potion_slot_capacity"),
    ))
    # Character-specific visible state (explicit namespace only).
    vcs_lines = _visible_character_state_lines(_visible_namespace(player))
    if vcs_lines:
        lines.append("")
        lines.append("CHARACTER-SPECIFIC RESOURCES (all currently shown by the game UI):")
        lines.extend(vcs_lines)
    return lines


def _run_memory_deck(run_memory: Any) -> tuple[list[Any], int]:
    """Cached deck snapshot (fallback ONLY). The bridge now serializes the
    current runtime master deck on every combat request, so this is used
    just for older mod builds."""
    if run_memory is None:
        return [], 0
    return run_memory.last_deck or [], run_memory.last_deck_count or 0


def _deck_block(deck: list[Any], deck_count: int, source_note: str) -> list[str]:
    if not deck:
        return []
    lines = [f"FULL DECK ({deck_count or 'unknown'} cards) [{source_note}]:"]
    groups, defs = _fmt_deck_groups(deck)
    lines.append(f"  Composition: {groups[0]}")
    if defs:
        lines.append("  Card definitions (static reference):")
        lines.extend(defs)
    return lines


# ----------------------------------------------------------------
# Combat formatter
# ----------------------------------------------------------------

def _playable_reason(card: dict[str, Any], energy: int, enemies: list[dict[str, Any]]) -> str:
    """Human-visible reason a card is not playable.

    Priority: the runtime `unplayable_reasons` flags reported by the bridge
    (the same engine reasons the UI uses to grey out the card), then
    UI-visible derivations (cost vs energy, target availability).
    """
    reasons = card.get("unplayable_reasons")
    if isinstance(reasons, list) and reasons:
        return "engine reason: " + ", ".join(str(r) for r in reasons)
    cost = card.get("current_energy_cost", card.get("cost", 0))
    if isinstance(cost, int) and 0 <= cost > energy:
        return f"not enough Energy (costs {cost}, you have {energy})"
    if isinstance(cost, str) and cost == "X" and energy <= 0:
        return "X-cost card but you have no Energy left"
    target = str(card.get("target", "None"))
    if target == "AnyEnemy" and not any(e.get("is_alive", False) for e in enemies):
        return "no living enemy to target"
    if card.get("playable") is False:
        return "not playable right now (exact engine reason not shown by the bridge)"
    return ""


def _format_enemy(enemy: dict[str, Any], index: int) -> list[str]:
    if not enemy.get("is_alive", False):
        return [f"  ENEMY[{index}] {enemy.get('id', '?')} (DEAD)"]
    lines = [
        f"  ENEMY[{index}] {enemy.get('id', '?')}  HP {enemy.get('hp', 0)}/{enemy.get('max_hp', 0)}"
        f", Block {enemy.get('block', 0)}"
    ]
    intent = str(enemy.get("intent", "UNKNOWN")).upper()
    desc = INTENT_GLOSSARY.get(intent, "intent unknown")
    intent_str = f"Intent: {intent} = {desc}"
    if intent in ("ATTACK", "MULTI_ATTACK") and "intent_damage" in enemy:
        # Exact value the game displays on the intent icon.
        hits = enemy.get("intent_hits", 1) or 1
        intent_str += f" -- displayed attack: {enemy['intent_damage']} damage x {hits} hit(s) on you"
    lines.append(f"      {intent_str}")
    ptext = _fmt_powers(enemy.get("powers"), indent="      ")
    if ptext:
        lines.append("      Powers:")
        lines.extend(ptext)
    return lines


def _cost_lines(card: dict[str, Any]) -> str:
    """Current vs base energy cost; 'X' for X-cost cards. Only mentions the
    base/permanent cost when it differs from the current one."""
    current = card.get("current_energy_cost", card.get("cost", 0))
    base = card.get("base_energy_cost")
    cur_s = str(current)
    parts = [f"Current cost: {cur_s} energy"]
    if base is not None and str(base) != cur_s:
        parts.append(f"Base/permanent cost: {base} energy")
    stars = card.get("current_star_cost")
    if stars is not None:
        parts.append(f"Star cost: {stars}")
    return " | ".join(parts)


def _runtime_card_lines(card: dict[str, Any], indent: str = "    ") -> list[str]:
    """Render the runtime card face (bridge-serialized rendered text),
    dynamic values, keywords and modifiers. Nothing arbitrary is read."""
    lines: list[str] = []
    display = str(card.get("current_display_text", card.get("display_text", "")) or "")
    if display:
        for ln in display.splitlines():
            if ln.strip():
                lines.append(f"{indent}Current displayed effect: {ln.strip()}")
    keywords = card.get("keywords")
    if isinstance(keywords, list) and keywords:
        lines.append(f"{indent}Keywords: {', '.join(str(k) for k in keywords)}")
    ench = card.get("enchantment")
    if isinstance(ench, dict) and ench:
        lines.append(
            f"{indent}Enchantment: {ench.get('name', ench.get('id', '?'))}"
            + (f" -- {ench['description']}" if ench.get("description") else "")
        )
    aff = card.get("affliction")
    if isinstance(aff, dict) and aff:
        lines.append(
            f"{indent}Affliction: {aff.get('name', aff.get('id', '?'))}"
            + (f" -- {aff['description']}" if aff.get("description") else "")
        )
    return lines


def _static_card_line(card: dict[str, Any]) -> str | None:
    """Static player-visible reference, used ONLY when the bridge could not
    provide the rendered card text. Never overrides runtime values."""
    cid = str(card.get("id", card.get("card_id", "")) or "")
    if not cid:
        return None
    up = bool(card.get("upgraded"))
    full = card_full(cid, upgraded=up)
    if full:
        return f"Card definition (static reference): {full}"
    if card_line(cid):
        return f"Card definition (static reference): {card_line(cid)}"
    return None


def _target_and_hover_lines(card: dict[str, Any], indent: str = "    ") -> list[str]:
    """Target-specific previews and keyword/modifier hover information
    (both serialized by the bridge from the game's own UI paths)."""
    lines: list[str] = []
    tps = card.get("target_previews")
    if isinstance(tps, list) and tps:
        lines.append(f"{indent}TARGET PREVIEWS (per living target, as the game"
                     " shows while hovering):")
        for j, tp in enumerate(tps):
            if isinstance(tp, dict):
                nm = tp.get("enemy_name") or tp.get("enemy_id", "?")
                for ln in str(tp.get("displayed_text", "")).splitlines():
                    if ln.strip():
                        lines.append(f"{indent}  vs ENEMY[{j}] {nm}: {ln.strip()}")
    elif card.get("target_preview_identical"):
        lines.append(f"{indent}Target-specific preview: identical for all legal targets.")
    hover = card.get("hover_info")
    if isinstance(hover, list) and hover:
        lines.append(f"{indent}Hover information:")
        for h in hover:
            if isinstance(h, dict):
                title = h.get("title", "")
                desc = h.get("description", "")
                lines.append(f"{indent}  {title}: {desc}" if title else f"{indent}  {desc}")
    return lines


def _format_hand(hand: list[dict[str, Any]], energy: int,
                 enemies: list[dict[str, Any]]) -> list[str]:
    lines = ["HAND (one entry per card, in game order):"]
    for i, card in enumerate(hand):
        cid = str(card.get("id", "UNKNOWN"))
        playable = card.get("playable")
        playable_s = "YES" if playable else "NO"
        target = str(card.get("target", "None"))
        t_hint = TARGET_TYPE_HINTS.get(target, target)
        name = str(card.get("display_name", "") or "")
        head = f"{name} ({cid})" if name and name != cid else cid
        up_level = card.get("upgrade_level")
        up_suffix = ""
        if card.get("upgraded"):
            up_suffix = f" (UPGRADED level {up_level})" if up_level else " (UPGRADED)"
        lines.append(f"  HAND[{i}] {head}{up_suffix}")
        lines.append(
            f"    {_cost_lines(card)} | Type: {card.get('type', '?')}"
            f" | Target: {target} ({t_hint})"
        )
        lines.append(f"    Playable now: {playable_s}")
        if playable is False:
            reason = _playable_reason(card, energy, enemies)
            if reason:
                lines.append(f"    Reason: {reason}")
        # Runtime rendered card face first (what the player actually sees).
        runtime_lines = _runtime_card_lines(card)
        lines.extend(runtime_lines)
        # Static reference only fills the gap when no rendered text exists.
        if not runtime_lines:
            static_line = _static_card_line(card)
            if static_line:
                lines.append(f"    {static_line}")
        preview = str(card.get("upgrade_preview_text", "") or "")
        if preview:
            lines.append(f"    Upgrade preview (as shown by the game): {preview}")
        lines.extend(_target_and_hover_lines(card))
    return lines


def _format_pile(name: str, count: int, composition: Any) -> list[str]:
    """Piles are serialized UNORDERED by the mod (same as the in-game pile
    viewer). Order is never available and never inferred. Modifier identity
    (upgrade/enchantment/affliction) is preserved in the grouping."""
    lines = [f"{name}: {count} cards (unordered, as shown by the pile viewer)"]
    if isinstance(composition, list) and composition:
        parts = []
        for entry in composition:
            if isinstance(entry, dict):
                parts.append(f"{entry.get('count', 1)}x {_card_group_label(entry)}")
            else:
                parts.append(str(entry))
        lines.append(f"  Composition: {', '.join(parts)}")
    return lines


def format_combat_state(
    state: dict[str, Any],
    run_memory: Any = None,
    response_mode: str = "single_action",
) -> str:
    player = state.get("player", {}) or {}
    enemies = state.get("enemies", []) or []
    energy = player.get("energy", 0)
    lines: list[str] = []

    # Combat payloads carry potion_slot_capacity at the top level; the run
    # header reads it from the player dict, so normalize first.
    if state.get("potion_slot_capacity") is not None:
        player.setdefault("potion_slot_capacity", state["potion_slot_capacity"])
    lines.extend(_run_header(
        state, "combat", player,
        potions=state.get("potions"),
    ))
    lines.append("== COMBAT: YOUR TURN ==")
    lines.append(f"Round: {state.get('round', '?')}")

    lines.append("")
    lines.append("PLAYER POWERS:")
    ptext = _fmt_powers(player.get("powers"))
    lines.extend(ptext if ptext else ["  (none)"])

    lines.append("")
    lines.append("ENEMIES (index = valid target_index; left-to-right as shown):")
    for i, enemy in enumerate(enemies):
        lines.extend(_format_enemy(enemy, i))

    lines.append("")
    lines.extend(_format_hand(state.get("hand", []) or [], energy, enemies))

    lines.append("")
    lines.extend(_format_pile("DRAW PILE", state.get("draw_pile_count", 0),
                              state.get("draw_pile")))
    lines.extend(_format_pile("DISCARD PILE", state.get("discard_pile_count", 0),
                              state.get("discard_pile")))
    lines.extend(_format_pile("EXHAUST PILE", state.get("exhaust_pile_count", 0),
                              state.get("exhaust_pile")))

    # Deck source priority: current runtime deck (bridge-serialized) >
    # cached RunMemory snapshot (older mod builds only).
    runtime_deck = state.get("deck")
    lines.append("")
    if isinstance(runtime_deck, list) and runtime_deck:
        lines.extend(_deck_block(
            runtime_deck, state.get("deck_count", len(runtime_deck)),
            "current runtime master deck, as shown by the in-game Deck viewer"))
    else:
        deck, deck_count = _run_memory_deck(run_memory)
        lines.extend(_deck_block(
            deck, deck_count,
            "cached from last confirmed runtime deck state (runtime deck not"
            " in this state; upgrade the bridge mod for live deck data)"))

    # ---- Legal actions (combat NEVER accepts "choose") ----
    playable_cards = [
        i for i, c in enumerate(state.get("hand", []) or []) if c.get("playable")
    ]
    usable_potions = [
        int(p.get("slot", i)) for i, p in enumerate(state.get("potions", []) or [])
        if isinstance(p, dict) and p.get("can_use", False)
    ]
    lines.append("")
    lines.append("LEGAL ACTIONS RIGHT NOW:")
    lines.append(f"  Playable cards (hand index): {playable_cards}")
    lines.append(f"  Usable potions (potion slot): {usable_potions}")
    lines.append(f"  Valid enemy targets: {[i for i, e in enumerate(enemies) if e.get('is_alive', False)]}")
    if response_mode == "single_action":
        lines.append("Allowed response shapes for THIS screen (choose EXACTLY ONE):")
        lines.append('  {"thought":"...","action":"play","card_index":N,"target_index":N}')
        lines.append('  {"thought":"...","action":"potion","slot":N,"target_index":N}')
        lines.append('  {"thought":"...","action":"discard_potion","slot":N} (only if discardable)')
        lines.append('  {"thought":"...","action":"end_turn"}')
        lines.append("Choose exactly one action. 'choose' is NOT a valid combat action.")
    else:
        # ActionChunk mode: plan-scoped refs + chunk schema. The ref
        # legend is objective, human-visible information only.
        # Local import keeps game_state importable even if the core
        # runtime module is absent (single-action mode still works).
        try:
            from action_plan import compact_ref_legend
            lines.append(compact_ref_legend(state))
        except ImportError:
            pass
        lines.append("ACTION-CHUNK RULE: You may commit several actions that are already")
        lines.append("determined by the information visible now. The harness validates every")
        lines.append("action against the updated game state. If a later action becomes")
        lines.append("invalid or new decision-relevant information appears, the harness")
        lines.append("stops the remaining chunk and asks you again.")
        lines.append('Respond with ONE JSON object: {"thought":"...","actions":[...]}')
        lines.append('  {"kind":"play","card_ref":"hN","target_ref":"eN"}  (omit target_ref when the card needs none)')
        lines.append('  {"kind":"potion","potion_slot":N,"target_ref":"eN"}')
        lines.append('  {"kind":"discard_potion","potion_slot":N} (only if discardable)')
        lines.append('  {"kind":"end_turn"}  (must be the final action)')
        lines.append('Optional "checkpoint_after":true on an action = inspect its result before'
                     ' deciding more (must then be the last action).')
        lines.append("Only reference cards/enemies listed above. 'choose' is NOT a valid combat action.")
    return "\n".join(lines)


# ----------------------------------------------------------------
# Choice screens
# ----------------------------------------------------------------

def _option_entries(state: dict[str, Any]) -> list[Any]:
    for key in ("options", "nodes", "choices", "items", "cards", "bundles", "relics"):
        val = state.get(key)
        if isinstance(val, list) and val:
            return val
    return []


def _format_option(i: int, opt: Any, stype: str = "") -> str:
    if not isinstance(opt, dict):
        return f"  OPTION[{i}] {opt}"
    bits: list[str] = []
    label = (
        opt.get("label")
        or opt.get("name")
        or opt.get("id")
        or opt.get("type")
        or opt.get("action")
        or ""
    )
    bits.append(str(label))
    for key in ("description", "price", "cost", "row", "col", "source_pile"):
        if opt.get(key) not in (None, ""):
            bits.append(f"{key}={opt[key]}")
    for tip in opt.get("hover_info") or []:
        if isinstance(tip, dict):
            bits.append(f"Hover [{tip.get('kind', 'tooltip')}] {tip.get('title', '')}: {tip.get('description', '')}")
    if opt.get("upgraded"):
        bits.append("upgraded")
    if "x" in opt and "y" in opt:
        bits.append(f"position=({opt['x']},{opt['y']})")
    text = f"  OPTION[{i}] " + " | ".join(bits)
    # Attach static full text for card/relic-style ids.
    cid = str(opt.get("id", opt.get("card_id", "")) or "")
    if cid:
        full = card_full(cid, upgraded=bool(opt.get("upgraded")))
        if full:
            text += f"\n      Full definition (static reference): {full}"
        elif card_line(cid):
            text += f"\n      Static reference: {card_line(cid)}"
    if opt.get("enabled", True) is False:
        text += " | (currently DISABLED - do not choose)"
    return text


def _format_player_summary(player: dict[str, Any]) -> list[str]:
    """Kept for states that carry only a partial snapshot; the common run
    header already renders HP/gold/relics/potions."""
    lines: list[str] = []
    deck = player.get("deck") or []
    if deck:
        lines.extend(_deck_block(deck, player.get("deck_count", len(deck)),
                                 "this screen shows the deck viewer"))
    return lines


def _format_full_map(full_map: list[dict[str, Any]], visited: list[dict[str, Any]]) -> list[str]:
    """Render the whole act map as rows (top row first), like the human map
    UI, with ALL visible edges. '?' nodes stay UNKNOWN."""
    visited_set = {(v.get("row"), v.get("col")) for v in visited if isinstance(v, dict)}
    by_row: dict[int, list[dict[str, Any]]] = {}
    for node in full_map:
        by_row.setdefault(int(node.get("row", 0)), []).append(node)
    lines = [
        "FULL ACT MAP (row: room type -> connected next nodes; '*' = on your visited path):"
    ]
    for row in sorted(by_row, reverse=True):
        cells = []
        for node in sorted(by_row[row], key=lambda n: int(n.get("col", 0))):
            col = int(node.get("col", 0))
            mark = "*" if (row, col) in visited_set else ""
            children = node.get("children") or []
            child_str = ""
            if children:
                child_str = " -> [" + ",".join(
                    f"({c.get('row')},{c.get('col')})"
                    for c in children if isinstance(c, dict)
                ) + "]"
            cells.append(
                f"({row},{col}) {mark}{_map_room_type(node.get('type', '?'))}{child_str}"
            )
        lines.append(f"  row {row}: " + " | ".join(cells))
    return lines


def _format_card_candidates(cards: list[Any]) -> list[str]:
    """Candidates on reward / bundle / select screens. Runtime rendered card
    face (incl. upgrade preview) first; static reference only fills gaps."""
    lines: list[str] = []
    for i, card in enumerate(cards):
        if not isinstance(card, dict):
            lines.append(f"  CANDIDATE[{i}] {card}")
            continue
        cid = str(card.get("id", card.get("card_id", "?")))
        up = bool(card.get("upgraded"))
        name = str(card.get("display_name", "") or "")
        head = f"{name} ({cid})" if name and name != cid else cid
        up_suffix = " (UPGRADED)" if up else ""
        lines.append(f"  CANDIDATE[{i}] {head}{up_suffix}")
        lines.append(f"    {_cost_lines(card)}")
        if card.get("type"):
            lines.append(f"    Type: {card['type']}")
        runtime_lines = _runtime_card_lines(card)
        lines.extend(runtime_lines)
        if not runtime_lines:
            static_line = _static_card_line(card)
            if static_line:
                lines.append(f"    {static_line}")
        preview = str(card.get("upgrade_preview_text", "") or "")
        if preview:
            lines.append(f"    Upgrade preview (as shown by the game): {preview}")
        lines.extend(_target_and_hover_lines(card))
    return lines


def _format_selection_combat_context(combat: dict[str, Any]) -> list[str]:
    """Board context for a card selection opened from INSIDE combat.

    Cards like "exhaust a card" / "upgrade a card in your hand" open a
    selection while the turn is running. A human deciding there still sees
    energy, the enemies and the rest of the hand; surface exactly that so the
    agent is not choosing blind.

    NOTE: the indices here are the HAND's own indices and are for
    identification only -- the answer must use the CANDIDATE index.
    """
    lines = [
        "COMBAT CONTEXT (this selection happens DURING your turn; the board"
        " below is still visible to you):",
        f"  Energy: {combat.get('energy', '?')}/{combat.get('max_energy', '?')}"
        f" | Block: {combat.get('block', '?')}"
        f" | Round: {combat.get('round', '?')}",
    ]
    trigger = str(combat.get("trigger", "") or "")
    if trigger:
        lines.append(f"  Triggered by your own action: {trigger}.")
    enemies = combat.get("enemies") or []
    lines.append("  ENEMIES:")
    if enemies:
        for i, enemy in enumerate(enemies):
            if isinstance(enemy, dict):
                lines.extend(_format_enemy(enemy, i))
    else:
        lines.append("    (none reported)")
    lines.append(
        "  YOUR HAND (identification only -- answer with the CANDIDATE index"
        " listed below, NOT with these hand indices):"
    )
    hand = combat.get("hand") or []
    if hand:
        for i, card in enumerate(hand):
            if not isinstance(card, dict):
                lines.append(f"    HAND[{i}] {card}")
                continue
            cid = str(card.get("id", "UNKNOWN"))
            name = str(card.get("display_name", "") or "")
            head = f"{name} ({cid})" if name and name != cid else cid
            playable = "YES" if card.get("playable") else "NO"
            lines.append(
                f"    HAND[{i}] {head} | {_cost_lines(card)} | Playable: {playable}"
            )
    else:
        lines.append("    (empty)")
    return lines


def format_choice_state(state: dict[str, Any], run_memory: Any = None) -> str:
    stype = str(state.get("type", ""))
    lines: list[str] = []
    title = CHOICE_TYPE_TITLES.get(stype, stype.upper() + ": pick an option by index")
    player = state.get("player") or {}
    lines.extend(_run_header(state, title.split(":")[0].strip(), player))
    lines.append(f"== {title} ==")

    # Screen-specific narrative / prompt context
    if stype == BridgeStateType.EVENT:
        # Event title + currently visible narrative body (the same text the
        # human reads on the event screen; flavor is kept, not trimmed).
        etitle = str(state.get("event_title", "") or "")
        ebody = str(state.get("event_description", "") or "")
        if etitle:
            lines.append(f"Event title: {etitle}")
        if ebody:
            lines.append(f"Event text: {ebody}")
    if stype == BridgeStateType.CARD_REWARD:
        lines.append(f"can_skip: {bool(state.get('can_skip', True))}")
    if stype == BridgeStateType.CARD_SELECT:
        lines.append(
            f"Selection prompt: select card(s) as required by the effect that"
            f" opened this screen (min_select={state.get('min_select', '?')},"
            f" max_select={state.get('max_select', '?')})."
        )
        # A card selection is frequently opened from INSIDE combat (e.g.
        # "exhaust a card", "upgrade a card in your hand"). A human choosing
        # there still sees the board, so the agent must see it too --
        # otherwise it picks which card to burn blind.
        combat = state.get("combat_context")
        if isinstance(combat, dict) and combat.get("in_combat"):
            lines.append("")
            lines.extend(_format_selection_combat_context(combat))
    if stype == BridgeStateType.SHOP:
        lines.append(
            "Prices are the current shop prices as shown; DISABLED = sold out or"
            " unaffordable. To leave, pick the 'Leave shop' option."
        )
    if stype == BridgeStateType.CRYSTAL_SPHERE:
        lines.append(
            "This is the run-start divination: every option is a HIDDEN cell"
            " whose outcome is unknown until revealed (bonus, combat, gold, or"
            " nothing). The hidden outcome is NOT included here -- a human"
            " cannot see it either. If a 'proceed' option is listed, choosing"
            " it finishes this screen."
        )
    lines.append("")

    # Player deck (human can always open the deck viewer)
    lines.extend(_format_player_summary(player))

    if stype == BridgeStateType.MAP_SELECT:
        full_map = state.get("full_map") or []
        if full_map:
            lines.extend(_format_full_map(full_map, state.get("visited") or []))
            lines.append("")
        lines.append("AVAILABLE NEXT ROOMS (these are the ONLY nodes you may pick):")
    else:
        lines.append("OPTIONS:")

    options = _option_entries(state)
    if options:
        if stype == BridgeStateType.CARD_BUNDLE:
            # Each option is a bundle of cards; render each bundle's cards.
            for i, opt in enumerate(options):
                if isinstance(opt, dict) and isinstance(opt.get("cards"), list):
                    lines.append(f"  BUNDLE[{i}] (choose this bundle to add all its cards):")
                    lines.extend(_format_card_candidates(opt["cards"]))
                else:
                    lines.append(_format_option(i, opt, stype))
        elif stype in (BridgeStateType.CARD_REWARD, BridgeStateType.CARD_SELECT):
            # Both screens list CARDS, so render the full runtime card face
            # (display name, current displayed effect, cost, type, target,
            # hover info) instead of a bare id. A combat selection such as
            # "exhaust a card" / "upgrade a card" is undecidable if the agent
            # only sees an id plus a cost.
            lines.extend(_format_card_candidates(options))
        else:
            gold = player.get("gold")
            for i, opt in enumerate(options):
                lines.append(_format_option(i, opt, stype))
                # Affordability note for priced options.
                if isinstance(opt, dict) and opt.get("price") is not None \
                        and gold is not None:
                    try:
                        if int(opt["price"]) > int(gold):
                            lines.append(
                                f"      Affordable now: NO (price {opt['price']} > gold {gold})"
                            )
                        else:
                            lines.append(f"      Affordable now: YES")
                    except (TypeError, ValueError):
                        pass
    else:
        lines.append("  (no option list was reported for this screen)")
    return "\n".join(lines)


def skip_allowed(state: dict[str, Any]) -> bool:
    """Mirror the mod's per-screen skip semantics EXACTLY.

    skip is honoured by the mod on: card_reward (can_skip) and card_select
    (min_select == 0). The shop maps skip to 'Leave shop', but the protocol
    deliberately does NOT advertise skip there -- leaving is 'choose 0'.
    Everywhere else skip falls back to a RANDOM option, so it is rejected
    by the validator before it can reach the game.
    """
    stype = str(state.get("type", ""))
    if stype == BridgeStateType.CARD_REWARD:
        return bool(state.get("can_skip", True))
    if stype == BridgeStateType.CARD_SELECT:
        try:
            return int(state.get("min_select", 0) or 0) <= 0
        except (TypeError, ValueError):
            return True
    return False


def _min_max(state: dict[str, Any]) -> tuple[int, int]:
    try:
        min_select = int(state.get("min_select", 0) or 0)
    except (TypeError, ValueError):
        min_select = 0
    try:
        max_select = int(state.get("max_select", 1) or 1)
    except (TypeError, ValueError):
        max_select = 1
    return min_select, max_select


def _legal_shapes_block(state: dict[str, Any]) -> str:
    """Per-screen legal response shapes. Only schemas that are actually
    legal on THIS screen are listed."""
    stype = str(state.get("type", ""))
    out = ["Allowed response shapes for THIS screen (choose EXACTLY ONE):"]
    if stype == BridgeStateType.CARD_SELECT:
        min_select, max_select = _min_max(state)
        if max_select > 1:
            out.append(
                '  {"thought":"...","action":"choose","indexes":[i1,i2,...]}'
                f" -- select between {min_select} and {max_select} candidates"
            )
        else:
            out.append(
                '  {"thought":"...","action":"choose","index":i} -- select exactly one candidate'
            )
        if skip_allowed(state):
            out.append('  {"thought":"...","action":"skip"} -- allowed (min_select is 0)')
        else:
            out.append("  skip is NOT allowed on this screen (min_select > 0).")
    elif stype == BridgeStateType.CARD_REWARD and skip_allowed(state):
        out.append('  {"thought":"...","action":"choose","index":i} to take the card')
        out.append('  {"thought":"...","action":"skip"} to take nothing')
    else:
        out.append('  {"thought":"...","action":"choose","index":i}')
        out.append("  skip is NOT allowed on this screen (it would trigger a random pick).")
    belt = state.get("potions", (state.get("player") or {}).get("potions")) or []
    for action, flag in (("potion", "can_use"), ("discard_potion", "can_discard")):
        slots = [p.get("slot", i) for i, p in enumerate(belt) if isinstance(p, dict) and p.get(flag) and not p.get("empty")]
        if slots:
            out.append(f'  {{"thought":"...","action":"{action}","slot":N}} -- legal slots: {slots}')
    return "\n".join(out)


def format_terminal_state(state: dict[str, Any]) -> str:
    """Result enum (unified everywhere): victory / defeat / terminated."""
    stype = str(state.get("type", ""))
    result = str(state.get("result", "")).lower()
    if stype == BridgeStateType.GAME_OVER:
        verdict = "Result: DEFEAT -- your HP reached 0. The run was lost."
    elif result == "terminated":
        verdict = (
            "Result: TERMINATED -- the run was aborted by the game side"
            " (error/timeout). This is NOT a real victory."
        )
    elif stype == BridgeStateType.RUN_COMPLETE:
        verdict = "Result: VICTORY -- you defeated the final boss and completed the run."
    else:
        verdict = f"Result: run ended ({result or 'unknown'})."
    lines = [
        "== RUN OVER ==",
        verdict,
        f"Floor {state.get('floor', '?')}, Act {state.get('act', '?')}",
    ]
    player = state.get("player") or {}
    if player:
        lines.append(
            f"Final: HP {player.get('hp', '?')}/{player.get('max_hp', '?')},"
            f" Gold {player.get('gold', '?')}"
        )
    lines.append("No action is required. The run has ended.")
    return "\n".join(lines)


# Whitelisted keys an unknown screen may expose to the LLM. Everything else
# in the raw payload is developer/debug visibility ONLY -- it is logged to
# the run log but NEVER enters the prompt (an unknown screen is exactly the
# case where we cannot judge which backend fields are hidden).
_UNKNOWN_SAFE_KEYS = {
    "visible_title",
    "visible_body",
    "visible_prompt",
    "visible_text",
    "min_select",
    "max_select",
    "can_skip",
    "floor",
    "act",
    "ascension",
}

_UNKNOWN_OPTION_SAFE_KEYS = {
    "label",
    "name",
    "description",
    "enabled",
    "price",
    "cost",
    "index",
}


def format_unknown_state(state: dict[str, Any]) -> str:
    """Generic fallback for unrecognized decision screens (new patches/mods
    add screens).

    Normal path: format the SAFE whitelist (explicit `visible_*` /
    `ui_visible` fields, canonical run status, visible option labels and
    prices) and let the LLM decide. The agent does NOT auto-select.

    Only in the failure path (all decision attempts failed / deadline hit)
    does the deterministic EMERGENCY fallback pick the first enabled option
    -- that is a last resort, not a 'safe' action, because its game effect
    is unknown.

    Raw payload fields beyond the whitelist are never shown to the LLM.
    """
    stype = str(state.get("type", "unknown"))
    lines = [
        f"== UNSUPPORTED SCREEN TYPE: {stype.upper()} ==",
        "WARNING: this screen has no dedicated formatter yet. Only the"
        " explicitly visible fields below are shown; base your decision"
        ' ONLY on them, use {"action":"choose","index":i} for an option you'
        " can identify from its visible label, and never guess a hidden"
        " outcome.",
        "",
    ]
    # Explicit ui_visible namespace (bridge-decided human-visible payload).
    ui_visible = state.get("ui_visible")
    if isinstance(ui_visible, dict):
        lines.append("VISIBLE SCREEN CONTENT (reported by the game):")
        for key, val in ui_visible.items():
            lines.append(f"  {key}: {val}")
        lines.append("")
    # Whitelisted top-level fields.
    for key in sorted(_UNKNOWN_SAFE_KEYS):
        if key in state and state[key] not in (None, ""):
            lines.append(f"{key}: {state[key]}")
    # Canonical run status (same as every other screen).
    player = state.get("player") or {}
    lines.append("")
    lines.extend(_run_header(state, stype, player))
    # Visible options: only the whitelisted option fields.
    options = _option_entries(state)
    if options:
        lines.append("")
        lines.append("OPTIONS:")
        for i, opt in enumerate(options):
            if isinstance(opt, dict):
                bits = [
                    f"{k}={opt[k]}"
                    for k in sorted(_UNKNOWN_OPTION_SAFE_KEYS)
                    if opt.get(k) not in (None, "")
                ]
                label = (
                    opt.get("label") or opt.get("name")
                    or opt.get("id") or "?"
                )
                text = f"  OPTION[{i}] {label}"
                if bits:
                    text += " | " + " | ".join(bits)
                if opt.get("enabled", True) is False:
                    text += " | (currently DISABLED)"
                lines.append(text)
                # Bundle-style card payloads are rendered card-by-card.
                if isinstance(opt.get("cards"), list):
                    lines.extend(_format_card_candidates(opt["cards"]))
            else:
                lines.append(f"  OPTION[{i}] {opt}")
    return "\n".join(lines)


def is_unsupported_state(state: dict[str, Any]) -> bool:
    stype = str(state.get("type", ""))
    return (
        stype not in DECISION_TYPES
        and stype not in (BridgeStateType.GAME_OVER, BridgeStateType.RUN_COMPLETE)
    )


def format_state(
    state: dict[str, Any],
    run_memory: Any = None,
    response_mode: str = "single_action",
) -> str:
    """Format any bridge state dict into LLM-readable text.

    ``response_mode`` only affects COMBAT states:
      - "single_action": the original one-JSON-action legal block (default,
        unchanged behavior);
      - "action_chunk": plan-scoped ref legend + ActionChunk schema.
    """
    stype = str(state.get("type", "unknown"))
    # Multiplayer boundary: this agent officially supports SINGLEPLAYER
    # only. Co-op states are flagged, never claimed as parity-complete.
    try:
        if int(state.get("player_count", 1) or 1) > 1:
            logger = logging.getLogger(__name__)
            logger.warning("Multiplayer state received (%s players); "
                           "singleplayer-only formatter in use.",
                           state.get("player_count"))
    except (TypeError, ValueError):
        pass
    if stype == BridgeStateType.COMBAT_ACTION:
        try:
            return format_combat_state(state, run_memory, response_mode=response_mode)
        except Exception as e:
            return f"== COMBAT (formatting error: {e}) ==\nraw: {state}"
    if stype in CHOICE_TYPES:
        try:
            text = format_choice_state(state, run_memory)
        except Exception as e:
            text = f"== CHOICE (formatting error: {e}) ==\nraw: {state}"
        return text + "\n\n" + _legal_shapes_block(state)
    if stype in (BridgeStateType.GAME_OVER, BridgeStateType.RUN_COMPLETE):
        return format_terminal_state(state)
    return format_unknown_state(state)


# ----------------------------------------------------------------
# Run-level memory
# ----------------------------------------------------------------

class RunMemory:
    """Durable context for the CURRENT run only (no cross-run learning).

    Deliberately NOT the source of truth: wherever the current runtime
    state provides a value (HP, gold, deck, relics, potions, floor), the
    current state wins. Memory only fills gaps (e.g. the deck on combat
    screens) and records history.
    """

    MAX_LOG_LINES = 40

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.floor = 0
        self.act = 0
        self.gold: int | None = None
        self.hp: int | None = None
        self.max_hp: int | None = None
        self.relics: list[str] = []
        self.combats_won = 0
        self.combats_lost = 0
        self.events: list[str] = []
        # Last deck snapshot observed on a screen that exposed it.
        self.last_deck: list[Any] = []
        self.last_deck_count = 0

    def observe(self, state: dict[str, Any]) -> None:
        """Update memory from a (possibly partial) state message."""
        stype = str(state.get("type", ""))
        try:
            self.floor = int(state.get("floor", self.floor) or self.floor)
            self.act = int(state.get("act", self.act) or self.act)
        except (TypeError, ValueError):
            pass
        player = state.get("player") or {}
        if player:
            try:
                self.gold = int(player.get("gold", self.gold))
                self.hp = int(player.get("hp", self.hp))
                self.max_hp = int(player.get("max_hp", self.max_hp))
            except (TypeError, ValueError):
                pass
            relics = player.get("relics")
            if isinstance(relics, list) and relics:
                self.relics = [
                    r.get("id", "?") if isinstance(r, dict) else str(r) for r in relics
                ]
            deck = player.get("deck")
            if isinstance(deck, list) and deck:
                # Runtime snapshot wins; keep exactly what was reported.
                self.last_deck = deck
                self.last_deck_count = int(player.get("deck_count", len(deck)))
        if stype == BridgeStateType.GAME_OVER:
            self.combats_lost += 1
            self._log("DEFEAT: HP reached 0.")
        elif stype == BridgeStateType.RUN_COMPLETE:
            result = str(state.get("result", "")).lower()
            if result == "terminated":
                self._log("Run terminated by the game (game-side error/timeout).")
            else:
                self.combats_won += 1
                self._log("VICTORY: run complete!")

    def note_combat_end(self, hp_before: int | None, hp_after: int | None) -> None:
        if hp_before is None or hp_after is None:
            return
        if hp_after < hp_before:
            self._log(f"Combat ended at floor {self.floor}: HP {hp_before} -> {hp_after}.")
        else:
            self._log(f"Combat ended at floor {self.floor}: no HP lost.")

    def note_event(self, text: str) -> None:
        self._log(text)

    def _log(self, text: str) -> None:
        if text:
            self.events.append(text)
            if len(self.events) > self.MAX_LOG_LINES:
                self.events = self.events[-self.MAX_LOG_LINES:]

    def to_text(self) -> str:
        hp_part = f"{self.hp}/{self.max_hp}" if self.max_hp else str(self.hp)
        lines = [
            "RUN MEMORY (context only -- if this conflicts with the CURRENT"
            " STATE, the CURRENT STATE is always correct):",
            f"- Position: Floor {self.floor}, Act {self.act}",
            f"- HP: {hp_part}, Gold: {self.gold}",
            f"- Relics: {', '.join(self.relics) if self.relics else '(none known)'}",
            f"- Combats won: {self.combats_won}, defeats: {self.combats_lost}",
        ]
        if self.events:
            lines.append("- Recent notable events (oldest first):")
            lines.extend(f"  * {e}" for e in self.events[-12:])
        return "\n".join(lines)
