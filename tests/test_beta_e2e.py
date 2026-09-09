"""Beta E2E: action_chunk mode over a fake bridge + scripted MockLLM.

Implements the full-agent-loop acceptance tests from the worker prompt:

  TEST 1  one model call drives multiple bridge actions (the single most
          important acceptance test: MockLLM.calls == 1, actions == 3)
  TEST 2  draw creates a HAND_CHANGED checkpoint -> plan interrupted,
          remainder discarded, second model call
  TEST 7  action rejected (action-relevantly unchanged state) -> no
          automatic replay loop, the model regains control
  TEST 9  explicit "checkpoint_after": true stops the chunk after ONE
          action even without any other diff
  TEST 10 strict failure: LLM timeout -> no non-LLM fallback action,
          benchmark invalidated, agent stops

Run:  python .\\tests\\test_beta_e2e.py
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent as agent_mod
from agent import AgentSession
from llm_client import LLMError


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def card(cid: str, *, target: str = "Self", playable: bool = True, cost: int = 1):
    return {
        "id": cid,
        "display_name": cid.title(),
        "cost": cost,
        "current_energy_cost": cost,
        "type": "Attack" if target == "AnyEnemy" else "Skill",
        "target": target,
        "playable": playable,
    }


def cultist(hp: int = 40):
    return {
        "id": "CULTIST", "hp": hp, "max_hp": 48, "block": 0,
        "is_alive": True, "intent": "ATTACK", "intent_damage": 6,
        "intent_hits": 1,
    }


def combat_state(
    request_id: str,
    *,
    energy: int,
    hand: list[dict],
    enemies: list[dict] | None = None,
    round_: int = 1,
    block: int = 0,
    draw_count: int = 5,
    discard_count: int = 0,
) -> dict:
    return {
        "type": "combat_action",
        "request_id": request_id,
        "floor": 1, "act": 1, "ascension": 0,
        "round": round_,
        "player": {"hp": 70, "max_hp": 80, "block": block, "energy": energy,
                   "max_energy": 3, "gold": 99},
        "hand": hand,
        "enemies": enemies if enemies is not None else [cultist()],
        "potions": [],
        "draw_pile_count": draw_count, "draw_pile": [],
        "discard_pile_count": discard_count, "discard_pile": [],
        "exhaust_pile_count": 0, "exhaust_pile": [],
    }


class FakeBridge(threading.Thread):
    """Sends a scripted list of states, collecting the agent's actions."""

    def __init__(self, port: int, states: list[dict]):
        super().__init__(daemon=True)
        self.port = port
        self.states = states
        self.actions: list[dict] = []
        self.error: str | None = None

    def run(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", self.port))
        srv.listen(1)
        srv.settimeout(20)
        conn, _ = srv.accept()
        conn.settimeout(20)
        buf = b""

        def send(obj: dict) -> None:
            conn.sendall(json.dumps(obj).encode() + b"\n")

        def recv() -> dict | None:
            nonlocal buf
            while b"\n" not in buf:
                chunk = conn.recv(4096)
                if not chunk:
                    return None
                buf += chunk
            line, buf = buf.split(b"\n", 1)
            return json.loads(line.decode())

        try:
            a = recv()
            assert a and a.get("action") == "set_fallback", a
            a = recv()
            assert a and a.get("action") == "set_agent_timeout", a
            for st in self.states:
                send(st)
                a = recv()
                if a is None:
                    break
                # bridge_client auto-attaches the state's request_id;
                # strip it so assertions stay on the action payload.
                a.pop("request_id", None)
                self.actions.append(a)
            time.sleep(0.3)
        except Exception as e:  # keep the thread from dying loudly
            self.error = str(e)
        finally:
            try:
                conn.close()
            except OSError:
                pass
            srv.close()


class MockLLM(agent_mod.LLMClient):
    """Scripted responses; counts API calls."""

    def __init__(self, *a, **k):
        super().__init__(base_url="http://mock", api_key="", model="mock")
        self.calls = 0
        self.fail = False

    def pick(self, last: str) -> str:  # overridden per test
        raise NotImplementedError

    def chat(self, messages):
        self.calls += 1
        if self.fail:
            raise LLMError("simulated LLM timeout")
        return self.pick(messages[-1]["content"])


def run_agent(port: int, llm_cls, cfg: dict | None = None, states: list[dict] | None = None):
    bridge = FakeBridge(port, states or [])
    bridge.start()

    agent_mod.LLMClient = llm_cls
    s = AgentSession()
    s.start({
        "bridge_host": "127.0.0.1", "bridge_port": port,
        "save_log": False, "disable_fallback": True,
        "auto_launch_game": False, "dump_raw_responses": False,
        "api_base_url": "http://mock", "api_key": "x", "model": "mock",
        "decision_mode": "action_chunk",
        **(cfg or {}),
    })
    return s, bridge


def wait_until(predicate, timeout: float = 12.0, step: float = 0.1) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return False


CHUNK1 = json.dumps({
    "thought": "attack, defend, end",
    "actions": [
        {"kind": "play", "card_ref": "h0", "target_ref": "e0"},
        {"kind": "play", "card_ref": "h1"},
        {"kind": "end_turn"},
    ],
})


# ----------------------------------------------------------------
# TEST 1 — one model call, multiple bridge actions
# ----------------------------------------------------------------

def test_one_call_multiple_actions() -> None:
    class LLM(MockLLM):
        def pick(self, last: str) -> str:
            return CHUNK1

    s0 = combat_state("r0", energy=3,
                      hand=[card("STRIKE", target="AnyEnemy"), card("DEFEND")])
    s1 = combat_state("r1", energy=2, hand=[card("DEFEND")],
                      enemies=[cultist(34)], discard_count=1)
    s2 = combat_state("r2", energy=1, hand=[], enemies=[cultist(34)],
                      block=5, discard_count=2)

    s, bridge = run_agent(9121, LLM, states=[s0, s1, s2])
    ok = wait_until(lambda: not s.status()["running"])
    time.sleep(0.3)
    st = s.status()

    assert len(bridge.actions) == 3, bridge.actions
    assert bridge.actions[0] == {"action": "play", "card_index": 0, "target_index": 0}
    # h1 (DEFEND) shifted from index 1 to current index 0 and must resolve.
    assert bridge.actions[1] == {"action": "play", "card_index": 0, "target_index": -1}
    assert bridge.actions[2] == {"action": "end_turn"}
    assert st["model_call_count"] == 1, st
    assert st["game_action_count"] == 3, st
    assert st["actions_per_llm_call"] == 3.0, st
    assert st["benchmark_valid"], st
    assert ok or not st["running"]  # agent finished without hanging
    print("PASS TEST 1 one_call_multiple_actions")


# ----------------------------------------------------------------
# TEST 2 — draw creates a checkpoint (HAND_CHANGED)
# ----------------------------------------------------------------

def test_draw_causes_checkpoint() -> None:
    class LLM(MockLLM):
        def pick(self, last: str) -> str:
            if "CHECKPOINT UPDATE" in last:
                return json.dumps({
                    "thought": "replan after the draw",
                    "actions": [{"kind": "end_turn"}],
                })
            return json.dumps({
                "thought": "pommel first; drawn card may change the turn",
                "actions": [
                    {"kind": "play", "card_ref": "h0", "target_ref": "e0"},
                    {"kind": "play", "card_ref": "h1", "target_ref": "e0"},
                    {"kind": "end_turn"},
                ],
            })

    s0 = combat_state("r0", energy=3,
                      hand=[card("POMMEL_STRIKE", target="AnyEnemy"),
                            card("STRIKE", target="AnyEnemy")])
    # After Pommel Strike a NEW card (BASH) became visible -> checkpoint.
    s1 = combat_state("r1", energy=2,
                      hand=[card("STRIKE", target="AnyEnemy"),
                            card("BASH", target="AnyEnemy", cost=2)],
                      enemies=[cultist(31)], discard_count=1)

    s, bridge = run_agent(9122, LLM, states=[s0, s1])
    wait_until(lambda: len(bridge.actions) >= 2 or not s.status()["running"])
    time.sleep(0.3)
    st = s.status()

    # Only the Pommel Strike executed from the OLD plan; the rest was
    # discarded and the model was re-consulted (2 API calls total).
    assert bridge.actions[0] == {"action": "play", "card_index": 0, "target_index": 0}
    assert bridge.actions[-1] == {"action": "end_turn"}, bridge.actions
    assert len(bridge.actions) == 2, bridge.actions
    assert st["model_call_count"] == 2, st
    assert st["checkpoint_count"] >= 1, st
    assert st["last_checkpoint_reason"] == "HAND_CHANGED", st
    assert st["plan_interrupted_count"] >= 1, st
    print("PASS TEST 2 draw_causes_checkpoint")


# ----------------------------------------------------------------
# TEST 7 — action rejected / unchanged state: no replay loop
# ----------------------------------------------------------------

def test_rejected_action_no_replay() -> None:
    class LLM(MockLLM):
        def pick(self, last: str) -> str:
            if "CHECKPOINT UPDATE" in last:
                return json.dumps({
                    "thought": "the model regained control",
                    "actions": [{"kind": "end_turn"}],
                })
            return json.dumps({
                "thought": "strike once",
                "actions": [{"kind": "play", "card_ref": "h0", "target_ref": "e0"}],
            })

    s0 = combat_state("r0", energy=3, hand=[card("STRIKE", target="AnyEnemy")])
    # Same visible state, new request_id -> the mod rejected the action.
    s1 = combat_state("r1", energy=3, hand=[card("STRIKE", target="AnyEnemy")])

    s, bridge = run_agent(9123, LLM, states=[s0, s1])
    wait_until(lambda: len(bridge.actions) >= 2 or not s.status()["running"])
    time.sleep(0.3)
    st = s.status()

    plays = [a for a in bridge.actions if a.get("action") == "play"]
    assert len(plays) == 1, bridge.actions  # never replayed blindly
    assert bridge.actions[-1] == {"action": "end_turn"}, bridge.actions
    assert st["model_call_count"] == 2, st
    assert st["last_checkpoint_reason"] == "ACTION_REJECTED", st
    print("PASS TEST 7 rejected_action_no_replay")


# ----------------------------------------------------------------
# TEST 9 — explicit checkpoint_after=true
# ----------------------------------------------------------------

def test_explicit_checkpoint() -> None:
    class LLM(MockLLM):
        def pick(self, last: str) -> str:
            if "CHECKPOINT UPDATE" in last:
                return json.dumps({
                    "thought": "inspecting the result",
                    "actions": [{"kind": "end_turn"}],
                })
            return json.dumps({
                "thought": "play once, then inspect",
                "actions": [{
                    "kind": "play", "card_ref": "h0", "target_ref": "e0",
                    "checkpoint_after": True,
                }],
            })

    s0 = combat_state("r0", energy=3, hand=[card("STRIKE", target="AnyEnemy")])
    s1 = combat_state("r1", energy=2, hand=[], enemies=[cultist(34)],
                      discard_count=1)

    s, bridge = run_agent(9124, LLM, states=[s0, s1])
    wait_until(lambda: len(bridge.actions) >= 2 or not s.status()["running"])
    time.sleep(0.3)
    st = s.status()

    assert len(bridge.actions) == 2, bridge.actions
    assert st["model_call_count"] == 2, st  # stopped after ONE action
    assert st["last_checkpoint_reason"] == "MODEL_REQUESTED", st
    print("PASS TEST 9 explicit_checkpoint")


# ----------------------------------------------------------------
# TEST 10 — strict failure: no fallback, benchmark invalid
# ----------------------------------------------------------------

def test_strict_failure() -> None:
    class LLM(MockLLM):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.fail = True

    s0 = combat_state("r0", energy=3,
                      hand=[card("STRIKE", target="AnyEnemy"), card("DEFEND")])

    s, bridge = run_agent(9125, LLM, states=[s0])
    wait_until(lambda: not s.status()["running"], timeout=20)
    time.sleep(0.3)
    st = s.status()

    # NO strategic fallback action may be sent.
    assert bridge.actions == [], bridge.actions
    assert st["benchmark_valid"] is False, st
    assert "LLM API error" in st["invalidation_reason"], st
    assert st["fallback_action_count"] == 0, st
    print("PASS TEST 10 strict_failure")


def run_all() -> None:
    tests = [
        test_one_call_multiple_actions,
        test_draw_causes_checkpoint,
        test_rejected_action_no_replay,
        test_explicit_checkpoint,
        test_strict_failure,
    ]
    for fn in tests:
        fn()
    print(f"\nALL {len(tests)} BETA E2E TESTS PASSED")


if __name__ == "__main__":
    run_all()
