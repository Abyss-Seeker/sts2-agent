"""Knowledge base for the LLM agent.

SOURCE-OF-TRUTH PRIORITY (highest first):
  1. Current runtime game state (serialized by bridge_mod from the live game)
  2. Current UI-visible text/labels (option labels/descriptions the mod
     screenshots from the actual UI nodes)
  3. Static knowledge base (docs/CARDS_REFERENCE.md, docs/RELICS_REFERENCE.md,
     docs/POWERS_REFERENCE.md -- all parsed from the CURRENT game build)
  4. Hardcoded glossaries below (core mechanics only)

The static KB is ONLY a fallback for fields the bridge cannot yet serialize
(e.g. card effect text, relic tooltips, potion effects). It must never
override a runtime value. Formatters label static text as such.

All static data is parsed from decompiled game source / the current build,
NOT from a community wiki.
"""

from __future__ import annotations

import logging
import os
import re
import threading

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# Hardcoded core glossaries (mechanics, not strategy)
# ----------------------------------------------------------------

POWER_GLOSSARY: dict[str, str] = {
    "STRENGTH": "Strength: attacker's physical damage is increased by this amount per hit.",
    "DEXTERITY": "Dexterity: Block gained from cards is increased by this amount.",
    "VULNERABLE": "Vulnerable: target takes 50% more attack damage. Decreases by 1 each turn.",
    "WEAK": "Weak: target deals 25% less attack damage. Decreases by 1 each turn.",
    "FRAIL": "Frail: target gains 25% less Block from cards. Decreases by 1 each turn.",
    "ARTIFACT": "Artifact: negates the next debuff applied to this creature.",
    "POISON": "Poison: at the start of its turn the creature loses HP equal to stacks, then stacks decrease by 1.",
    "REGEN": "Regeneration: at end of turn heal HP equal to stacks, then stacks decrease by 1.",
    "PLATED_ARMOR": "Plated Armor: gain Block equal to stacks at end of turn; losing HP from attacks reduces stacks by 1.",
    "RITUAL": "Ritual: gains Strength equal to stacks at end of its turn.",
    "CURL_UP": "Curl Up: when it first loses HP from an attack, gains Block equal to stacks.",
    "THORNS": "Thorns: when attacked, deals damage equal to stacks back to the attacker.",
    "METALLICIZE": "Metallicize: gains Block equal to stacks at the end of its turn.",
    "INTANGIBLE": "Intangible: reduces all damage taken to 1. Decreases by 1 each turn.",
    "ENTANGLED": "Entangled: cannot play Attack cards this turn.",
    "NO_DRAW": "No Draw: cannot draw cards this turn.",
    "BARRICADE": "Barricade: Block is no longer lost at the start of the turn.",
    "BURNING": "Burning: takes damage equal to stacks at the start of its turn, then stacks decrease by 1.",
    "MINION": "Minion: a summoned creature.",
}

INTENT_GLOSSARY: dict[str, str] = {
    "ATTACK": "will attack (exact damage shown)",
    "MULTI_ATTACK": "will attack multiple times (exact damage x hits shown)",
    "DEFEND": "will gain Block",
    "BUFF": "will strengthen itself",
    "DEBUFF": "will weaken you (e.g. Weak/Vulnerable/Frail)",
    "DEBUFF_STRONG": "will apply a strong debuff",
    "SLEEP": "is asleep (will not act until disturbed)",
    "SUMMON": "will summon minions",
    "ESCAPE": "will try to escape the fight",
    "STUN": "will be stunned (skips its turn)",
    "HEAL": "will heal itself or allies",
    "DEATH_BLOW": "will execute (likely lethal if it lands)",
    "CARD_DEBUFF": "will put status/curse cards into your deck or hand",
    "UNKNOWN": "intent unknown",
    "SINGLEATTACK": "will attack (exact damage shown)",
    "MULTIATTACK": "will attack multiple times (exact damage x hits shown)",
    "DEBUFFSTRONG": "will apply a strong debuff",
    "STATUS": "will add status cards",
    "STATUSCARD": "will add status cards",
    "CARDDEBUFF": "will debuff cards",
    "DEATHBLOW": "will attempt a lethal attack; inspect current powers and effects",
    "HIDDEN": "intent is not currently visible",
}

# Keyword tooltips (core set; extend as the game build grows).
KEYWORD_GLOSSARY: dict[str, str] = {
    "Exhaust": "Exhaust: when played, the card is removed for the rest of this combat.",
    "Ethereal": "Ethereal: when the turn ends, if still in hand the card is exhausted.",
    "Retain": "Retain: when the turn ends, the card stays in your hand instead of being discarded.",
    "Innate": "Innate: this card starts in your opening hand.",
    "Unplayable": "Unplayable: this card cannot be played.",
    "X-cost": "X-cost: playing it consumes ALL your remaining energy and its effect scales with X.",
    "Summon": "Summon: affects your summoned ally (e.g. Necrobinder's Osty).",
    "Stars": "Stars: Regent resource; some cards cost Stars in addition to (or instead of) energy.",
    "Forge": "Forge: increases a permanent counter on the card.",
    "Doom": "Doom: at the end of the afflicted creature's side's turn, it dies if its HP is at or below its Doom stacks (subject to death-prevention effects).",
    "Scry": "Scry: look at and optionally discard the top cards of your draw pile.",
    "Discard": "Discard: moves the card to your discard pile.",
    "Evoke": "Evoke: releases the top orb of your orb slots (Defect).",
    "Channel": "Channel: adds an orb to your orb slots (Defect).",
    "Focus": "Focus: increases Defect orb passive/evoke effect amounts.",
}


# ----------------------------------------------------------------
# Static reference parsing (docs parsed from the current game build)
# ----------------------------------------------------------------

_REPO_LOCK = threading.Lock()
_REPO_ROOT: str | None | None = None  # triple-state: None=unknown, None checked
_KB_LOCK = threading.Lock()
_CARD_DB: dict[str, dict[str, str]] | None = None
_RELIC_DB: dict[str, str] | None = None
_POWER_DB: dict[str, str] | None = None

_CARD_LINE_CACHE: dict[str, str] = {}
_CARD_FULL_CACHE: dict[tuple[str, bool], str] = {}
_RELIC_CACHE: dict[str, str] = {}
_POWER_CACHE: dict[str, str] = {}


def _ensure_sts2() -> bool:
    """Lazily make ``sts2_env`` importable (repo root on sys.path)."""
    try:
        import sys
        from pathlib import Path

        repo_root = str(Path(__file__).resolve().parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        import sts2_env.core.enums  # noqa: F401  (probe import)

        return True
    except Exception as e:
        logger.warning("sts2_env unavailable, simulator card registry disabled: %s", e)
        return False


def _repo_root() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parent.parent)


def _read(path: str) -> str:
    with _REPO_LOCK:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


def _parse_card_reference() -> dict[str, dict[str, str]]:
    """Parse docs/CARDS_REFERENCE.md into {ID: {field: value}}."""
    path = os.path.join(_repo_root(), "docs", "CARDS_REFERENCE.md")
    db: dict[str, dict[str, str]] = {}
    try:
        text = _read(path)
    except OSError:
        logger.warning("CARDS_REFERENCE.md not found; static card KB disabled")
        return db
    field_re = re.compile(r"^-\s+\*\*(\w+):\*\*\s*(.*)$")
    current: dict[str, str] | None = None
    for line in text.splitlines():
        if line.startswith("### "):
            current = {"name": line[4:].strip()}
            continue
        m = field_re.match(line)
        if m and current is not None:
            current[m.group(1).lower()] = m.group(2).strip()
            # entries are keyed after we see the ID field
            if m.group(1).lower() == "id":
                db[m.group(2).strip().upper()] = current
    return db


def _parse_generic_reference(filename: str) -> dict[str, str]:
    """Parse RELICS_REFERENCE.md / POWERS_REFERENCE.md style docs into
    {ID: description} using the '### Name' + '- ID: X' + '- Logic:' blocks."""
    path = os.path.join(_repo_root(), "docs", filename)
    db: dict[str, str] = {}
    try:
        text = _read(path)
    except OSError:
        logger.warning("%s not found; static KB for %s disabled", filename, filename)
        return db
    name = ""
    rid = ""
    lines: list[str] = []
    in_logic = False

    def flush() -> None:
        if rid and lines:
            db.setdefault(rid.upper(), f"{name}\n" + "\n".join(l for l in lines if l.strip()))

    for line in text.splitlines():
        if line.startswith("### "):
            flush()
            name, rid, lines, in_logic = line[4:].strip(), "", [], False
        elif line.startswith("- ID:"):
            flush()
            rid = line[len("- ID:"):].strip()
            lines = []
        elif line.startswith("- Logic:"):
            in_logic = True
        elif line.startswith("- ") or line.startswith("## "):
            in_logic = False
        elif in_logic and line.startswith("  - "):
            lines.append(line[4:].strip())
    flush()
    return db


def _card_db() -> dict[str, dict[str, str]]:
    global _CARD_DB
    with _KB_LOCK:
        if _CARD_DB is None:
            _CARD_DB = _parse_card_reference()
        return _CARD_DB


def _relic_db() -> dict[str, str]:
    global _RELIC_DB
    with _KB_LOCK:
        if _RELIC_DB is None:
            _RELIC_DB = _parse_generic_reference("RELICS_REFERENCE.md")
        return _RELIC_DB


def _power_db() -> dict[str, str]:
    global _POWER_DB
    with _KB_LOCK:
        if _POWER_DB is None:
            _POWER_DB = _parse_generic_reference("POWERS_REFERENCE.md")
        return _POWER_DB


# ----------------------------------------------------------------
# Public lookups (static fallbacks -- runtime values take priority)
# ----------------------------------------------------------------

def card_line(card_id: str) -> str:
    """One-line static summary for a card id, e.g.

    ``STRIKE_IRONCLAD`` -> ``1E, Deal Damage to target enemy, Attack``.
    Returns "" when unknown. STATIC fallback only.
    """
    if not card_id:
        return ""
    key = str(card_id).strip().upper()
    with _KB_LOCK:
        cached = _CARD_LINE_CACHE.get(key)
    if cached is not None:
        return cached
    info = _card_db().get(key)
    if not info:
        line = ""
    else:
        parts = []
        if info.get("cost"):
            parts.append(f"{info['cost']}E")
        if info.get("effect"):
            parts.append(info["effect"])
        if info.get("type"):
            parts.append(info["type"])
        line = ", ".join(parts)
    with _KB_LOCK:
        _CARD_LINE_CACHE[key] = line
    return line


def card_full(card_id: str, upgraded: bool = False) -> str:
    """Full static card definition text (multi-line). STATIC fallback for
    effect text the bridge cannot yet serialize."""
    if not card_id:
        return ""
    key = (str(card_id).strip().upper(), bool(upgraded))
    with _KB_LOCK:
        cached = _CARD_FULL_CACHE.get(key)
    if cached is not None:
        return cached
    info = _card_db().get(key[0])
    if not info:
        text = ""
    else:
        parts = [f"Name: {info.get('name', card_id)}"]
        if info.get("cost") is not None:
            parts.append(f"Base cost: {info['cost']} energy")
        if info.get("type"):
            parts.append(f"Type: {info['type']}")
        if info.get("rarity"):
            parts.append(f"Rarity: {info['rarity']}")
        if info.get("target"):
            parts.append(f"Target: {info['target']}")
        kw = info.get("keywords", "None")
        if kw and kw != "None":
            parts.append(f"Keywords: {kw}")
        if info.get("effect"):
            parts.append(f"Base effect: {info['effect']}")
        if info.get("vars") and info.get("vars") != "None":
            parts.append(f"Static values: {info['vars']}")
        if upgraded and info.get("upgrade") and info.get("upgrade") != "None":
            parts.append(f"Upgraded change: {info['upgrade']}")
        text = "; ".join(parts)
    with _KB_LOCK:
        _CARD_FULL_CACHE[key] = text
    return text


def relic_text(relic_id: str) -> str:
    """Static relic description (from the current build's decompiled logic)."""
    if not relic_id:
        return ""
    key = str(relic_id).strip().upper()
    with _KB_LOCK:
        cached = _RELIC_CACHE.get(key)
    if cached is not None:
        return cached
    text = _relic_db().get(key, "")
    with _KB_LOCK:
        _RELIC_CACHE[key] = text
    return text


def power_text(power_id: str) -> str:
    """Static power description: hardcoded glossary first, then the parsed
    build reference as fallback."""
    if not power_id:
        return ""
    key = str(power_id).strip().upper()
    with _KB_LOCK:
        cached = _POWER_CACHE.get(key)
    if cached is not None:
        return cached
    text = POWER_GLOSSARY.get(key, "")
    if not text:
        text = _power_db().get(key, "")
    with _KB_LOCK:
        _POWER_CACHE[key] = text
    return text


def warm_up() -> None:
    """Pre-parse the static KB in a background thread so the first
    in-combat lookup is fast."""

    def _worker() -> None:
        try:
            card_line("STRIKE_IRONCLAD")
            relic_text("BURNING_BLOOD")
            power_text("STRENGTH")
        except Exception:
            pass

    threading.Thread(target=_worker, name="kb-warmup", daemon=True).start()
