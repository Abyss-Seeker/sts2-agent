import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import AgentSession, DEFAULT_CONFIG, migrate_prompt_config
from action_plan import parse_action_chunk
from context_manager import ContextConfig, ContextManager, _structured_compress
from game_state import format_state
from prompts import DEFAULT_SYSTEM_TEMPLATE, LEGACY_SCHEMA_3_SYSTEM_TEMPLATE, empty_answer_feedback


def combat():
    return {"type": "combat_action", "player": {"hp": 40, "energy": 3},
            "round": 1, "hand": [], "potions": [],
            "enemies": [{"id": "CEREMONIAL_BEAST", "is_alive": True,
                         "hp": 180, "intent": "SingleAttack", "intent_damage": 22}]}


@pytest.mark.parametrize("old", [LEGACY_SCHEMA_3_SYSTEM_TEMPLATE, "{{RULEBOOK}}\n\n{{CONTRACT}}"])
def test_old_defaults_migrate(old):
    result, _ = migrate_prompt_config({"system_template": old, "prompt_schema_version": 3})
    assert result["system_template"] == DEFAULT_SYSTEM_TEMPLATE
    assert result["prompt_schema_version"] == 4


def test_example_inherits_and_custom_template_survives():
    example = json.loads(Path("config.example.json").read_text())
    assert {**DEFAULT_CONFIG, **example}["system_template"] == DEFAULT_SYSTEM_TEMPLATE
    custom = "My own game prompt {{CONTRACT}}"
    assert migrate_prompt_config({"system_template": custom})[0]["system_template"] == custom


def test_user_template_is_rendered_without_losing_current_state():
    ctx = ContextManager(ContextConfig(max_history_turns=0, max_context_chars=1))
    for template in ("MY INSTRUCTION {{STATE}}", "MY INSTRUCTION"):
        messages = ctx.build_messages("SYS", "MEM", "CURRENT", user_template=template)
        assert "MY INSTRUCTION" in messages[-1]["content"]
        assert "CURRENT" in messages[-1]["content"]
    messages = ctx.build_messages("SYS", "MEM", "{{RUN_MEMORY}}", user_template="{{STATE}}")
    assert "{{RUN_MEMORY}}" in messages[-1]["content"]  # no recursive expansion of game text


def test_chunk_has_no_conflicting_potion_schema():
    state = combat()
    state["potions"] = [{"slot": 0, "id": "X", "can_use": True, "can_discard": True}]
    text = format_state(state, response_mode="action_chunk")
    assert '"action":"potion"' not in text
    assert '"kind":"potion"' in text
    assert "22 damage" in text


@pytest.mark.parametrize("screen", ["combat_action", "shop"])
def test_formatter_errors_never_dump_secrets(screen):
    state = {**combat(), "type": screen, "seed": "SECRET_SEED", "rng": "SECRET_RNG"}
    formatter = "format_combat_state" if screen == "combat_action" else "format_choice_state"
    with patch("game_state." + formatter, side_effect=ValueError("SECRET_ERROR")):
        text = format_state(state)
    assert "formatting error:" in text
    assert "SECRET" not in text
    assert "raw:" not in text


def test_delta_includes_full_board_without_history():
    session = AgentSession()
    session._config["delta_observations"] = True
    previous = combat()
    session._last_model_observation_state = previous
    session._last_checkpoint_reason = "MODEL_REQUESTED"
    current = combat()
    current["player"]["energy"] = 2
    text = session._model_observation_text(current)
    assert "CHECKPOINT UPDATE" in text
    assert "ENEMY[0]" in text and "HAND" in text
    assert '"kind":"end_turn"' in text


def test_missing_thought_does_not_reject_chunk():
    result = parse_action_chunk({"actions": [{"kind": "end_turn"}]}, combat())
    assert len(result.actions) == 1
    assert result.summary == ""


def test_retry_does_not_suggest_end_turn_or_limit_thinking_to_one_sentence():
    for chunk in (True, False):
        for truncated in (True, False):
            text = empty_answer_feedback(chunk=chunk, truncated=truncated)
            assert "end_turn" not in text
            assert "ONE short sentence" not in text
            assert "no action was executed" in text


def test_potion_hover_usage_and_merchant_context():
    state = {"type": "shop", "screen": "merchant_entrance", "potions": [
        {"slot": 0, "id": "FOUL_POTION", "usage": "AnyTime", "can_use": True,
         "hover_info": [{"title": "Special", "description": "visible detail"}]}]}
    text = format_state(state)
    assert "Usage class: AnyTime" in text and "visible detail" in text
    assert "Merchant visible; inventory closed" in text
    assert "EVERYONE includes you" in text


def test_full_belt_reward_is_visible_but_not_claimable():
    from agent import validate_action
    state = {"type": "reward_screen", "options": [
        {"label": "New Potion", "effect": "Gain 20 Block", "enabled": False,
         "requires_empty_potion_slot": True}], "potions": [
        {"slot": 0, "id": "OLD", "can_use": False, "can_discard": True}]}
    text = format_state(state)
    assert "Gain 20 Block" in text and "requires_empty_potion_slot=True" in text
    assert validate_action(state, {"action": "choose", "index": 0})[0] is None
    assert validate_action(state, {"action": "discard_potion", "slot": 0})[0]
    state["options"][0]["enabled"] = True
    state["options"][0]["requires_empty_potion_slot"] = False
    assert validate_action(state, {"action": "choose", "index": 0})[0]


def test_multiple_intents_reach_model_without_move_id():
    state = combat()
    state["enemies"][0].update(intents=["Attack", "Buff"], intent_move_id="HIDDEN_MOVE")
    text = format_state(state)
    assert "BUFF = will strengthen itself" in text
    assert "HIDDEN_MOVE" not in text


def test_keywords_do_not_suppress_missing_card_definition():
    state = combat()
    state["hand"] = [{"id": "TEST", "keywords": ["Exhaust"], "playable": True}]
    with patch("game_state.card_full", return_value="Deal 10 damage"):
        assert "Deal 10 damage" in format_state(state)


def test_target_preview_keeps_authoritative_enemy_index():
    state = combat()
    state["hand"] = [{"id": "TEST", "playable": True, "target_previews": [
        {"enemy_index": 2, "enemy_id": "TEST_ENEMY", "displayed_text": "Deal 15 damage"}]}]
    assert "vs ENEMY[2] TEST_ENEMY: Deal 15 damage" in format_state(state)


def test_essential_long_line_survives_compression():
    text = "  OPTION[0] " + "Important effect " * 1000
    assert _structured_compress(text, 100) == text
