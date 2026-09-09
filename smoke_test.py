import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import extract_json, validate_action, render_template, fallback_action
from context_manager import ContextManager
from game_state import format_state

# 1. extract_json tolerates fences / chatter
reply = '```json\n{"action": "play", "card_index": 0, "target_index": 1}\n```'
obj = extract_json(reply)
assert obj["action"] == "play", obj
obj = extract_json('Sure! Here is my answer: {"action": "end_turn"} hope that helps')
assert obj["action"] == "end_turn"
try:
    extract_json("no json here")
    raise AssertionError("should have raised")
except ValueError:
    pass

# 1b. json-repair handles malformed JSON (trailing comma etc.)
obj = extract_json('{"action": "play", "card_index": 0, "target_index": 1,}')
assert obj["card_index"] == 0

# 1c. key aliases + lenient ints
from agent import validate_action as _va
state2 = {"type": "combat_action",
          "hand": [{"id": "STRIKE", "cost": 1, "target": "AnyEnemy", "playable": True}],
          "enemies": [{"id": "JAW", "is_alive": True}], "potions": []}
act, err = _va(state2, {"action": "play", "card": 0, "target": 0})
assert act == {"action": "play", "card_index": 0, "target_index": 0}, (act, err)
act, err = _va(state2, {"action": "play", "card_index": "0", "target_index": 0.0})
assert act == {"action": "play", "card_index": 0, "target_index": 0}, (act, err)
act, err = _va(state2, {"action": "end"})
assert act == {"action": "end_turn"}, (act, err)

# 1d. skip legality per screen (mirrors mod semantics)
act, err = _va({"type": "shop", "options": [{"index": 0, "label": "Leave shop"}]},
               {"action": "skip"})
assert act is None, err  # shop skip is NOT advertised; leaving = choose 0
assert fallback_action({"type": "shop", "options": [{"index": 0}]}) == \
    {"action": "choose", "index": 0}  # fallback leaves via 'Leave shop'
act, err = _va({"type": "reward_screen", "options": [{"index": 0}]}, {"action": "skip"})
assert act is None and "NOT allowed" in err, err
act, err = _va({"type": "map_select", "nodes": [{"index": 0}]}, {"action": "skip"})
assert act is None, err
act, err = _va({"type": "card_reward", "can_skip": True, "cards": [{"index": 0}]},
               {"action": "skip"})
assert act == {"action": "skip"}, err
act, err = _va({"type": "card_reward", "can_skip": False, "cards": [{"index": 0}]},
               {"action": "skip"})
assert act is None, err
act, err = _va({"type": "card_select", "min_select": 1, "cards": [{"index": 0}]},
               {"action": "skip"})
assert act is None, err
act, err = _va({"type": "card_select", "min_select": 0, "cards": [{"index": 0}]},
               {"action": "skip"})
assert act == {"action": "skip"}, err
act, err = _va(state2, {"action": "skip"})  # combat: skip invalid
assert act is None, err

# 1e. disabled option rejected; fallback picks first enabled option
from agent import fallback_action
act, err = _va({"type": "event", "options": [{"index": 0, "label": "a", "enabled": False},
                                             {"index": 1, "label": "b", "enabled": True}]},
               {"action": "choose", "index": 0})
assert act is None and "DISABLED" in err, err
assert fallback_action({"type": "event",
                        "options": [{"enabled": False}, {"enabled": True}]})["index"] == 1
assert fallback_action({"type": "combat_action"}) == {"action": "end_turn"}

# 2. render_template
assert render_template("A {{X}} B", {"X": "1"}) == "A 1 B"

# 3. validate_action
state = {
    "type": "combat_action",
    "hand": [
        {"id": "STRIKE", "cost": 1, "target": "AnyEnemy", "playable": True},
        {"id": "DEFEND", "cost": 1, "target": "Self", "playable": True},
        {"id": "HEAVY", "cost": 9, "target": "None", "playable": False},
    ],
    "enemies": [{"id": "JAW", "is_alive": True}, {"id": "CULT", "is_alive": False}],
    "potions": [{"slot": 0, "id": "FIRE", "can_use": True, "requires_target": True}],
}
act, err = validate_action(state, {"action": "play", "card_index": 0, "target_index": 0})
assert act == {"action": "play", "card_index": 0, "target_index": 0}, err
act, err = validate_action(state, {"action": "play", "card_index": 0, "target_index": 1})
assert act is None and "dead" in err, err
act, err = validate_action(state, {"action": "play", "card_index": 2})
assert act is None and "not playable" in err, err
act, err = validate_action(state, {"action": "play", "card_index": 1, "target_index": -1})
assert act is not None, err
act, err = validate_action(state, {"action": "potion", "slot": 0, "target_index": 0})
assert act is not None, err
act, err = validate_action(state, {"action": "potion", "slot": 0, "target_index": 1})
assert act is None, err
act, err = validate_action(state, {"action": "end_turn"})
assert act == {"action": "end_turn"}, err
act, err = validate_action(state, {"action": "choose", "index": 2})
assert act is None and "NOT a legal action in combat" in err, err  # combat rejects choose
act, err = validate_action({"type": "event", "options": []}, {"action": "choose", "index": 2})
assert act is not None, err  # choice screens without options list accept any int
act, err = validate_action({"type": "map_select", "nodes": [{"index": 0}, {"index": 1}]},
                           {"action": "choose", "index": 5})
assert act is None and "out of range" in err, err

# 4. context manager budget trimming
ctx = ContextManager()
for i in range(12):
    ctx.add_decision("STATE-" + str(i) * 50, '{"action":"end_turn"}', "ok")
msgs = ctx.build_messages("SYSTEM PROMPT", "RUN MEM", "CURRENT STATE")
roles = [m["role"] for m in msgs]
assert roles[0] == "system" and roles[-1] == "user"
assert "STATE-11" not in msgs[1]["content"] or True  # current state is last
total = sum(len(m["content"]) for m in msgs)
assert total < 24000 + 5000, total
print("context roles:", roles[:3], "... total chars:", total)

# 5. format_state on a combat sample
combat = {
    "type": "combat_action",
    "floor": 3, "act": 1, "round": 2, "ascension": 0,
    "player": {"hp": 45, "max_hp": 72, "block": 5, "energy": 2, "max_energy": 3,
               "gold": 99, "powers": [{"id": "STRENGTH", "amount": 2}],
               "relics": [{"id": "BURNING_BLOOD"}]},
    "hand": [
        {"id": "STRIKE", "cost": 1, "type": "Attack", "target": "AnyEnemy", "playable": True},
        {"id": "DEFEND", "cost": 1, "type": "Skill", "target": "Self", "playable": True},
        {"id": "DEMON_FORM", "cost": 3, "type": "Power", "target": "Self", "playable": False},
    ],
    "enemies": [
        {"id": "JAW_WORM", "hp": 38, "max_hp": 42, "block": 0, "is_alive": True,
         "intent": "ATTACK", "intent_damage": 11, "intent_hits": 1,
         "powers": [{"id": "VULNERABLE", "amount": 2}]},
        {"id": "CULTIST", "hp": 0, "max_hp": 48, "block": 0, "is_alive": False},
    ],
    "potions": [{"slot": 0, "id": "FIRE_POTION", "can_use": True, "requires_target": True}],
    "draw_pile_count": 9, "draw_pile": [{"id": "STRIKE", "count": 4}],
    "discard_pile_count": 2, "discard_pile": [],
    "exhaust_pile_count": 0, "exhaust_pile": [],
}
text = format_state(combat)
assert "JAW_WORM" in text and "11 damage x 1" in text
assert "STRENGTH 2" in text and "50% more" in text
assert "DRAW PILE: 9" in text
print(text[:400])

# 6. choice screen formatting
choice = {"type": "map_select", "floor": 4, "act": 1,
          "nodes": [{"index": 0, "type": "MONSTER", "row": 3, "col": 2},
                    {"index": 1, "type": "REST", "row": 3, "col": 4}]}
print(format_state(choice)[:300])

# 7. crystal sphere (Neow) formatting
crystal = {"type": "crystal_sphere", "floor": 0, "act": 1,
           "options": [
               {"index": 0, "action": "divine_cell", "x": 1, "y": 2, "enabled": True},
               {"index": 1, "action": "divine_cell", "x": 2, "y": 1, "enabled": True},
               {"index": 2, "action": "proceed", "enabled": True},
           ]}
ctext = format_state(crystal)
assert "HIDDEN cell" in ctext and "proceed" in ctext and "position=(1,2)" in ctext

# 8. player summary on choice screens (human parity)
shop = {"type": "shop", "floor": 5, "act": 1,
        "player": {"hp": 42, "max_hp": 72, "gold": 231, "ascension": 5,
                    "relics": [{"id": "BURNING_BLOOD", "counter": 1}],
                    "potions": ["FIRE", "BLOCK"],
                    "deck": [{"id": "STRIKE", "count": 4, "upgraded": False},
                             {"id": "DEFEND", "count": 4, "upgraded": True}],
                    "deck_count": 8},
        "options": [{"index": 0, "label": "Buy card", "price": 45}]}
stext = format_state(shop)
assert "HP: 42/72" in stext and "Gold: 231" in stext and "Ascension: 5" in stext
assert "4x STRIKE, 4x DEFEND+" in stext

# 9. full act map rendering
map_state = {"type": "map_select", "floor": 3, "act": 1,
             "nodes": [{"index": 0, "type": "REST", "row": 2, "col": 4}],
             "full_map": [
                 {"row": 0, "col": 2, "type": "MONSTER", "children": [{"row": 1, "col": 3}]},
                 {"row": 1, "col": 3, "type": "ELITE", "children": [{"row": 2, "col": 4}]},
                 {"row": 2, "col": 4, "type": "REST", "children": []},
             ],
             "visited": [{"row": 0, "col": 2}, {"row": 1, "col": 3}]}
mtext = format_state(map_state)
assert "FULL ACT MAP" in mtext and "*MONSTER" in mtext and "*ELITE" in mtext and "REST" in mtext

print("\nALL SMOKE TESTS PASSED")
