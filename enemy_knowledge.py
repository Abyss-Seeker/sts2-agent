"""Optional static enemy behavior knowledge, independent of live Monster AI state."""

from functools import lru_cache
import json
import logging
from pathlib import Path

REFERENCE_PATH = Path(__file__).parent / "docs" / "enemy_behaviors.json"

POLICY = """ENEMY BEHAVIOR REFERENCE (optional static encounter knowledge)
These are static build definitions, equivalent to learning encounter patterns.
Use them to plan across turns, including buffs, debuffs, status cards and phase changes.
Current visible intent, damage, powers and card text override these base values.
Definition variables are NOT current instance values. Never assume a hidden phase,
counter, random roll or future move has been observed. Infer timing only from
visible history; if ambiguous, retain all compatible possibilities.
The source snapshot may differ from the installed game; report conflicts as uncertainty.
C# AscensionHelper.GetValueIfAscension(level, higher, normal) chooses higher at
that ascension (ToughEnemies=8, DeadlyEnemies=9); current displayed numbers win.
Random branches: probability = eligible weight / sum of eligible weights.
Unspecified weight is 1. AddBranch(state, int, MoveRepeatType, [weight]) uses
the integer as COOLDOWN, not weight. AddBranch(state, int, [weight]) uses
the integer as maximum consecutive repeats. CannotRepeat excludes the last
state; UseOnlyOnce excludes previously used states; cooldown excludes moves
used in the last N moves. Re-normalize after exclusions and conditional weights.
Do not present unconditional percentages when eligibility/history is unknown.
MoveState FollowUpState defines deterministic transitions unless a condition,
stun, phase change or other gameplay hook overrides them. Cosmetic calls have
no tactical meaning. PowerCmd.Apply<T>(recipient, amount, ...) applies the
named effect to that recipient; base.Creature means the monster, targets means
its targets. Strength adds damage per attack hit; Weak reduces attack damage
25%, Vulnerable increases damage taken 50%, Frail reduces card Block 25%.
"""

NOTES = {
    "CEREMONIAL_BEAST": "Plow is an HP threshold, NOT Block: after unblocked damage, "
    "if its HP is <= Plow amount (150 normal / 160 DeadlyEnemies), it loses ALL "
    "Strength, is stunned, loses Plow, then enters Beast Cry -> Stomp -> Crush -> "
    "Beast Cry. Plow attacks gain 2 Strength each; Crush gains 3 / 4 Strength. "
    "Beast Cry applies Ringing until the end of the player's next turn: cards "
    "without an existing affliction gain Ringing, including newly entering cards. "
    "A Ringing-afflicted card cannot play if that player has already started any "
    "card play this turn. Cards with other afflictions are not converted. This "
    "restricts card plays; it does not skip the entire player turn.",
    "FLYCONID": "Opening: Frail Spores 50%, Smash 50%. Subsequently choose uniformly "
    "among eligible moves: Vulnerable Spores has a 3-move cooldown, Frail Spores "
    "a 2-move cooldown, and all moves CannotRepeat. Those 3 and 2 are NOT weights. "
    "Vulnerable Spores applies 2 Vulnerable; Frail Spores attacks and applies 2 Frail.",
}


@lru_cache(maxsize=1)
def _reference():
    try:
        return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))["monsters"]
    except (OSError, ValueError, KeyError):
        logging.getLogger(__name__).warning("Enemy behavior reference unavailable", exc_info=True)
        return {}


def enemy_behavior_context(state: dict, enabled: bool = True) -> str:
    if not enabled:
        return ""
    board = state.get("combat_context") or state
    enemies = board.get("enemies") or []
    if not enemies:
        return ""
    sections = []
    shared_powers = {}
    seen = set()
    for enemy in enemies:
        if not isinstance(enemy, dict) or not enemy.get("is_alive", False):
            continue
        # Only identity is read: no move IDs, AI counters, seeds or future rolls.
        enemy_id = str(enemy.get("id", "UNKNOWN")).upper()
        if enemy_id in seen:
            continue
        seen.add(enemy_id)
        entry = _reference().get(enemy_id)
        if entry is None:
            sections.append(f"{enemy_id}: behavior reference unavailable; do not invent probabilities.")
            continue
        sections.append(f"ENEMY TYPE: {enemy_id}\n" + NOTES.get(enemy_id, ""))
        sections.append(entry["mechanics"])
        shared_powers.update(entry["powers"])
    for name, definition in sorted(shared_powers.items()):
        sections.append(f"Effect definition: {name}\n{definition}")
    return POLICY + "\n\n" + "\n\n".join(sections) if sections else ""
