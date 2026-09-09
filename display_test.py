import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import AgentSession

s = AgentSession()
llm = type("L", (), {"last_reasoning": ""})()

# Explanation line before JSON -> shown in brief mode
raw = 'Explanation: enemy intents deal 12, playing block is best.\n{"action": "play", "card_index": 1, "target_index": 0}'
out = s._decision_display(
    "brief", raw, {"action": "play", "card_index": 1, "target_index": 0}, llm)
assert out == "enemy intents deal 12, playing block is best.", out

# thought field still wins when present
out = s._decision_display(
    "brief",
    'Explanation: a\n{"thought":"b","action":"end_turn"}',
    {"thought": "b", "action": "end_turn"}, llm)
assert out == "b", out

# raw reply without explanation -> falls back to raw
out = s._decision_display(
    "brief", '{"action":"choose","index":0}', {"action": "choose", "index": 0}, llm)
assert out == '{"action":"choose","index":0}', out

# extract_json still parses replies with a leading explanation line
from agent import extract_json
obj = extract_json('Explanation: go left, safer.\n{"action": "choose", "index": 1}')
assert obj == {"action": "choose", "index": 1}, obj

# pre_json_text edge cases
assert s._pre_json_text('{"a":1}') == ""        # JSON starts immediately
assert s._pre_json_text("no json") == ""        # no JSON at all
assert s._pre_json_text("Explanation：先防御\n{...") == "先防御"  # full-width colon

print("EXPLANATION DISPLAY TESTS PASSED")
