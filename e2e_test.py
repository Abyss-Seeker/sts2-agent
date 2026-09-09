"""E2E: fake bridge TCP server + mocked LLM -> full agent loop test."""

import json
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent as agent_mod
from agent import AgentSession

FAKE_PORT = 9111
received_actions: list[dict] = []


def fake_bridge():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", FAKE_PORT))
    srv.listen(1)
    conn, _ = srv.accept()
    buf = b""

    def send(obj):
        conn.sendall(json.dumps(obj).encode() + b"\n")

    def recv_action():
        nonlocal buf
        while b"\n" not in buf:
            chunk = conn.recv(4096)
            if not chunk:
                return None
            buf += chunk
        line, buf = buf.split(b"\n", 1)
        obj = json.loads(line.decode())
        received_actions.append(obj)
        return obj

    # 1. crystal sphere (Neow)
    a = recv_action()  # set_fallback
    assert a and a.get("action") == "set_fallback", a
    a = recv_action()  # set_agent_timeout
    assert a and a.get("action") == "set_agent_timeout", a
    send({
        "type": "crystal_sphere", "floor": 0, "act": 1,
        "options": [
            {"index": 0, "action": "divine_cell", "x": 1, "y": 2, "enabled": True},
            {"index": 1, "action": "divine_cell", "x": 2, "y": 1, "enabled": True},
            {"index": 2, "action": "proceed", "enabled": True},
        ],
    })
    a = recv_action()
    assert a == {"action": "choose", "index": 0}, a

    # 2. combat: play card 0 then end turn
    for energy in (3, 2):
        send({
            "type": "combat_action", "floor": 1, "act": 1, "round": 1, "ascension": 0,
            "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": energy, "max_energy": 3, "gold": 99},
            "hand": [
                {"id": "STRIKE", "cost": 1, "type": "Attack", "target": "AnyEnemy", "playable": True},
                {"id": "DEFEND", "cost": 1, "type": "Skill", "target": "Self", "playable": True},
            ],
            "enemies": [{"id": "JAW_WORM", "hp": 40, "max_hp": 42, "block": 0, "is_alive": True,
                          "intent": "ATTACK", "intent_damage": 10, "intent_hits": 1}],
            "potions": [],
            "draw_pile_count": 8, "draw_pile": [], "discard_pile_count": 0, "discard_pile": [],
            "exhaust_pile_count": 0, "exhaust_pile": [],
        })
        a = recv_action()
        if energy == 3:
            assert a == {"action": "play", "card_index": 0, "target_index": 0}, a
        else:
            assert a == {"action": "end_turn"}, a

    # 3. event
    send({"type": "event", "floor": 2, "act": 1, "options": [
        {"index": 0, "action": "event_choice", "label": "Take damage", "enabled": True},
        {"index": 1, "action": "event_choice", "label": "Leave", "enabled": True},
    ]})
    a = recv_action()
    assert a == {"action": "choose", "index": 1}, a

    # 4. run over
    send({"type": "run_complete", "floor": 50, "act": 3})
    time.sleep(0.3)
    conn.close()
    srv.close()


# Mock the LLM
class MockLLM(agent_mod.LLMClient):
    def __init__(self, *a, **k):
        super().__init__(base_url="http://mock", api_key="", model="mock")
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        last = messages[-1]["content"]
        if "CRYSTAL SPHERE" in last:
            return '{"thought": "reveal first", "action": "choose", "index": 0}'
        if "COMBAT" in last:
            if '"*"' not in last and "STRIKE" in last:
                pass
            # first combat state -> play strike; second (energy=2) -> end turn
            if "Energy: 3/3" in last:
                return '{"thought": "hit it", "action": "play", "card_index": 0, "target_index": 0}'
            return '{"thought": "done", "action": "end_turn"}'
        if "EVENT" in last:
            return '{"thought": "safe", "action": "choose", "index": 1}'
        return '{"thought": "ok", "action": "skip"}'


agent_mod.LLMClient = MockLLM

t = threading.Thread(target=fake_bridge, daemon=True)
t.start()

s = AgentSession()
s.start({
    "bridge_host": "127.0.0.1", "bridge_port": FAKE_PORT,
    "save_log": False, "disable_fallback": True,
    "api_base_url": "http://mock", "api_key": "x", "model": "mock",
})
deadline = time.time() + 15
while time.time() < deadline and s.status()["running"]:
    time.sleep(0.2)
time.sleep(0.5)
st = s.status()
print("status:", st)
print("actions received by fake game:", received_actions)
logs = s.logs_since(0)
for e in logs:
    print(f"  [{e['kind']}] {e['text'][:80]}")

assert not st["running"], "agent should have stopped after run_complete"
assert st["decision_count"] == 4, st["decision_count"]
assert any(e["kind"] == "state" and "crystal_sphere" in e["text"] for e in logs)
assert received_actions[-1] == {"action": "choose", "index": 1}
print("\nE2E TEST PASSED")
