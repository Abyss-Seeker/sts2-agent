import pytest
from agent import validate_action
from game_state import format_state
from action_plan import parse_action_chunk, ActionKind
from plan_executor import ActionChunkExecutor
from presentation import overlay_snapshot

@pytest.mark.parametrize("screen", ["combat_action", "event", "shop", "map", "card_reward", "card_select", "rest_site"])
def test_inventory_actions_on_each_screen(screen):
    state = {"type": screen, "player": {"potions": [{"slot": 0, "id": "FRUIT_JUICE", "can_use": True, "can_discard": True}]}}
    assert validate_action(state, {"action": "potion", "slot": 0})[0]
    assert validate_action(state, {"action": "discard_potion", "slot": 0})[0]

@pytest.mark.parametrize("extra", [{"empty": True}, {"queued": True}, {"can_use": False, "can_discard": False}])
def test_illegal_inventory_actions_rejected(extra):
    state = {"type": "event", "potions": [{"can_use": True, "can_discard": True, **extra}]}
    for action in ("potion", "discard_potion"):
        assert validate_action(state, {"action": action, "slot": 0})[0] is None

def test_automatic_potion_can_be_discarded_without_using():
    state = {"type": "event", "potions": [{"can_use": False, "can_discard": True, "usage": "Automatic"}]}
    assert validate_action(state, {"action": "potion", "slot": 0})[0] is None
    assert validate_action(state, {"action": "discard_potion", "slot": 0})[0]

def test_event_hover_and_inventory_shapes_reach_model():
    text = format_state({"type": "event", "player": {"potions": [{"slot": 0, "id": "FOUL_POTION", "can_use": True, "can_discard": True}]},
        "options": [{"label": "Accept", "hover_info": [{"kind": "relic", "title": "Forgotten Soul", "description": "Visible relic effect"}]}]})
    assert "Hover [relic] Forgotten Soul: Visible relic effect" in text
    assert '"action":"discard_potion"' in text
    assert '"action":"potion"' in text

def test_discard_chunk_parses_and_resolves():
    state = {"type": "combat_action", "hand": [], "enemies": [], "potions": [{"slot": 0, "id": "TEST", "can_use": False, "can_discard": True}]}
    chunk = parse_action_chunk({"thought": "Make room.", "actions": [{"kind": "discard_potion", "potion_slot": 0}]}, state)
    assert chunk.actions[0].kind is ActionKind.DISCARD_POTION
    executor = ActionChunkExecutor()
    executor.chunk = chunk
    event = executor.prepare_next(state, validate_action)
    assert event.prepared.bridge_action["action"] == "discard_potion"
