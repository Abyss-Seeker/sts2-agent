"""Beta E2E: action_chunk mode over a fake bridge + scripted MockLLM.

Acceptance tests (review-remediated semantics):

  TEST 1  one LLM plan -> Strike -> Defend -> EndTurn, followed by the
          post-EndTurn authoritative ROUND 2 state. The first plan is
          1 logical plan, 3 sent + 3 CONFIRMED actions, plan completed
          (not interrupted) with checkpoint reason NEW_TURN.
  TEST 1b draw checkpoint re-prompt must contain the FULL human-visible
          face of the NEW card (cost / playable / displayed text) --
          BLOCKER 1: forced-full observation.
  TEST 3  combat action followed by card_select: the in-flight action is
          reconciled (confirmed, checkpoint SCREEN_CHANGED, old plan
          discarded) BEFORE routing; card_select goes through the normal
          noncombat choice path; no stale combat action is executed.
  TEST 4  lethal action followed by reward_screen: in-flight action
          confirmed and the chunk exhausted -> plan COMPLETED.
  TEST 5 (P0) bridge ACCEPTED + action-relevantly UNCHANGED state:
          sent += 1, confirmed += 0, rejected += 0, unconfirmable += 1,
          the chunk is RETAINED, no re-prompt and no second gameplay
          action (AWAITING_ADVANCE is a waiting state, not a rejection).
  TEST 5b accepted + unchanged, THEN card_select: the selection preempts
          combat, the triggering action is not rejected, the stale chunk
          remainder is discarded, and no LLM call happened at the unchanged
          snapshot.
  TEST 5c accepted + unchanged, THEN normal advance: the SAME in-flight
          action is confirmed exactly once and the chunk continues.
  TEST 5d explicit bridge rejection (no acceptance marker) is still a real
          rejection.
  TEST 5e repeated accepted + unchanged -> bounded PROTOCOL_STALL.
  TEST 6  API timeout (strict): llm_request_count and
          llm_failed_request_count increase, benchmark invalid, ZERO
          fallback strategic actions.

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
from agent import AgentSession, PROTOCOL_STALL_REPEATS
from action_plan import parse_action_chunk
from benchmark_metrics import BenchmarkMetrics
from checkpoint import CheckpointReason
from llm_client import LLMError
from plan_executor import ExecutorStatus


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


def card_select_state(request_id: str) -> dict:
    return {
        "type": "card_select",
        "request_id": request_id,
        "floor": 1, "act": 1,
        "min_select": 1, "max_select": 1,
        "options": [{"index": 0, "label": "Exhaust target", "enabled": True}],
    }


def reward_screen_state(request_id: str) -> dict:
    return {
        "type": "reward_screen",
        "request_id": request_id,
        "floor": 1, "act": 1,
        "options": [
            {"index": 0, "label": "Gold", "enabled": True},
            {"index": 1, "label": "proceed", "enabled": True},
        ],
    }


_TIMEOUT = object()


class FakeBridge(threading.Thread):
    """Sends a scripted list of states, collecting the agent's actions.

    ``action_wait`` (seconds) models the asynchronous lifecycle:
      - None (default): after each state, block until the agent's gameplay
        action arrives (strict one-state/one-action handshake).
      - float: push the NEXT authoritative state after waiting AT MOST
        ``action_wait`` seconds for an action. This is REQUIRED for
        accepted-but-not-yet-advanced lifecycles, where one gameplay
        command legitimately produces several authoritative observations
        and the harness sends NO action for the unchanged ones.
    """

    def __init__(self, port: int, states: list[dict], *,
                 action_wait: float | None = None):
        super().__init__(daemon=True)
        self.port = port
        self.states = states
        self.action_wait = action_wait
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

        def recv(timeout: float | None = None):
            nonlocal buf
            conn.settimeout(20 if timeout is None else timeout)
            while b"\n" not in buf:
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    return _TIMEOUT
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
                deadline = (
                    None if self.action_wait is None
                    else time.time() + self.action_wait
                )
                # Drain protocol control commands (set_fallback/
                # set_agent_timeout/set_headful/set_fast_mode) until the
                # actual game action for this state arrives.
                while True:
                    if deadline is None:
                        a = recv()
                    else:
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            break
                        a = recv(timeout=remaining)
                    if a is None:
                        return
                    if a is _TIMEOUT:
                        # No action for this state (accepted-but-not-yet-
                        # advanced): move on to the NEXT authoritative
                        # state WITHOUT requiring a gameplay action.
                        break
                    # bridge_client auto-attaches the state's request_id;
                    # strip it so assertions stay on the action payload.
                    a.pop("request_id", None)
                    if str(a.get("action", "")).startswith("set_"):
                        continue
                    self.actions.append(a)
                    break
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
    """Scripted responses (class attr `script`); records every prompt."""

    script: list[str] = []
    fail = False

    def __init__(self, *a, **k):
        super().__init__(base_url="http://mock", api_key="", model="mock")
        self.calls = 0
        self.prompts: list[str] = []

    def chat(self, messages):
        self.calls += 1
        # Simulate the real client: one HTTP attempt per chat() call so
        # the agent's on_http_attempt accounting is exercised.
        self._notify_http_attempt(self.calls)
        self.prompts.append(messages[-1]["content"])
        if self.fail:
            raise LLMError("simulated LLM timeout")
        if self.script:
            return self.script.pop(0)
        raise AssertionError("unexpected extra LLM call")


def run_agent(port: int, llm_cls, cfg: dict | None = None,
              states: list[dict] | None = None, bridge_wait: float | None = None):
    bridge = FakeBridge(port, states or [], action_wait=bridge_wait)
    bridge.start()

    agent_mod.LLMClient = llm_cls
    s = AgentSession()
    s.start({
        "bridge_host": "127.0.0.1", "bridge_port": port,
        "save_log": False, "disable_fallback": True,
        "auto_launch_game": False, "dump_raw_responses": False,
        "auto_resume": False,  # tests assert termination, not recovery waits
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


CHUNK_STRIKE_DEFEND_END = json.dumps({
    "thought": "attack, defend, end",
    "actions": [
        {"kind": "play", "card_ref": "h0", "target_ref": "e0"},
        {"kind": "play", "card_ref": "h1"},
        {"kind": "end_turn"},
    ],
})
CHUNK_END_TURN = json.dumps({
    "thought": "nothing worth doing",
    "actions": [{"kind": "end_turn"}],
})
CHUNK_POMMEL = json.dumps({
    "thought": "pommel first; drawn card may change the turn",
    "actions": [
        {"kind": "play", "card_ref": "h0", "target_ref": "e0"},
        {"kind": "play", "card_ref": "h1", "target_ref": "e0"},
        {"kind": "end_turn"},
    ],
})
CHUNK_STRIKE_ONCE = json.dumps({
    "thought": "strike once",
    "actions": [{"kind": "play", "card_ref": "h0", "target_ref": "e0"}],
})
CHUNK_STRIKE_END = json.dumps({
    "thought": "strike then end the turn",
    "actions": [
        {"kind": "play", "card_ref": "h0", "target_ref": "e0"},
        {"kind": "end_turn"},
    ],
})
CHUNK_STRIKE_INSPECT = json.dumps({
    "thought": "play once, then inspect",
    "actions": [{
        "kind": "play", "card_ref": "h0", "target_ref": "e0",
        "checkpoint_after": True,
    }],
})
CHOOSE_0 = json.dumps({
    "thought": "pick the visible option",
    "action": "choose",
    "index": 0,
})


# ----------------------------------------------------------------
# TEST 1 — one plan -> 3 sent + 3 CONFIRMED actions, plan completed
# ----------------------------------------------------------------

def test_one_plan_three_confirmed_actions() -> None:
    class LLM(MockLLM):
        script = [CHUNK_STRIKE_DEFEND_END, CHUNK_END_TURN]

    s0 = combat_state("r0", energy=3,
                      hand=[card("STRIKE", target="AnyEnemy"), card("DEFEND")])
    s1 = combat_state("r1", energy=2, hand=[card("DEFEND")],
                      enemies=[cultist(34)], discard_count=1)
    s2 = combat_state("r2", energy=1, hand=[], enemies=[cultist(34)],
                      block=5, discard_count=2)
    # Post-EndTurn authoritative state: round 2, new hand / new energy.
    s3 = combat_state("r3", energy=3, round_=2,
                      hand=[card("STRIKE", target="AnyEnemy")],
                      enemies=[cultist(34)], draw_count=3, discard_count=2)

    s, bridge = run_agent(9121, LLM, states=[s0, s1, s2, s3])
    wait_until(lambda: not s.status()["running"])
    time.sleep(0.3)
    st = s.status()
    logs = s.logs_since(0)

    # First plan drove exactly these three actions, in order.
    assert bridge.actions[0] == {"action": "play", "card_index": 0, "target_index": 0}
    # h1 (DEFEND) shifted from index 1 to current index 0 and must resolve.
    assert bridge.actions[1] == {"action": "play", "card_index": 0, "target_index": -1}
    assert bridge.actions[2] == {"action": "end_turn"}
    # The harness legitimately started a second plan after round 2.
    assert bridge.actions[3] == {"action": "end_turn"}

    # Sent == confirmed accounting: the first plan's 3 actions were all
    # confirmed by S1/S2/S3 (the 4th action's confirming state never came
    # because the fake bridge closes -- it is sent but NOT confirmed).
    assert st["game_action_sent_count"] == 4, st
    assert st["game_action_confirmed_count"] == 3, st
    assert st["game_action_rejected_count"] == 0, st
    # 1 logical plan for the first chunk; 2 inference requests total.
    assert st["llm_request_count"] == 2, st
    assert st["llm_success_count"] == 2, st
    assert st["llm_failed_request_count"] == 0, st
    # The first plan COMPLETED (chunk exhausted) -- the NEW_TURN checkpoint
    # must NOT be counted as an interruption (review fix).
    assert st["plan_completed_count"] == 1, st
    assert st["plan_interrupted_count"] == 0, st
    assert st["last_checkpoint_reason"] == "NEW_TURN", st
    # executed_vs_planned counts only CONFIRMED plan actions: plan1's 3
    # confirmed; plan2's single action was sent but never confirmed.
    assert st["planned_actions_total"] == 4, st
    assert st["executed_planned_actions_total"] == 3, st
    assert abs(st["executed_vs_planned_ratio"] - 0.75) < 1e-9, st
    cps = [e for e in logs if e["kind"] == "plan_checkpoint"
           and e.get("reason") == "NEW_TURN"]
    assert len(cps) == 1 and cps[0]["executed_steps"] == 3, cps
    assert cps[0].get("plan_completed") is True, cps
    print("PASS TEST 1 one_plan_three_confirmed_actions")


# ----------------------------------------------------------------
# TEST 1b — draw checkpoint re-prompt carries FULL new card info
# ----------------------------------------------------------------

def test_draw_checkpoint_reprompt_contains_full_new_card_information() -> None:
    class LLM(MockLLM):
        script = [CHUNK_POMMEL, CHUNK_END_TURN]

    s0 = combat_state("r0", energy=3,
                      hand=[card("POMMEL_STRIKE", target="AnyEnemy"),
                            card("STRIKE", target="AnyEnemy")])
    bash = card("BASH", target="AnyEnemy", cost=2)
    bash["current_display_text"] = "Deal 10 damage. Apply 2 Vulnerable."
    s1 = combat_state("r1", energy=2,
                      hand=[card("STRIKE", target="AnyEnemy"), bash],
                      enemies=[cultist(31)], discard_count=1)

    s, bridge = run_agent(9122, LLM, states=[s0, s1])
    wait_until(lambda: not s.status()["running"])
    time.sleep(0.3)
    st = s.status()

    assert len(llm_prompts(s)) == 2
    second = llm_prompts(s)[1]
    # The model must be able to inspect the NEW card like a human would:
    # identity, current cost, playability and its current displayed text.
    assert "BASH" in second, second
    assert "Current cost: 2 energy" in second, second
    assert "Playable now: YES" in second, second
    assert "Deal 10 damage" in second, second
    # A FULL state was sent -- not a bare delta with only "+h Bash".
    assert "HAND (one entry per card" in second, second
    assert "CHECKPOINT UPDATE" not in second, second
    assert st["last_checkpoint_reason"] == "HAND_CHANGED", st
    print("PASS TEST 1b draw_checkpoint_full_card_information")


def llm_prompts(s: AgentSession) -> list[str]:
    llm = s._llm
    return list(getattr(llm, "prompts", []))


# ----------------------------------------------------------------
# TEST 3 — combat action followed by card_select: reconcile FIRST
# ----------------------------------------------------------------

def test_screen_change_reconciles_inflight_before_routing() -> None:
    class LLM(MockLLM):
        script = [
            json.dumps({
                "thought": "play, then end",
                "actions": [
                    {"kind": "play", "card_ref": "h0", "target_ref": "e0"},
                    {"kind": "end_turn"},
                ],
            }),
            CHOOSE_0,  # card_select handled by the normal choice path
        ]

    s0 = combat_state("r0", energy=3, hand=[card("STRIKE", target="AnyEnemy")])
    s1 = card_select_state("r1")

    s, bridge = run_agent(9126, LLM, states=[s0, s1])
    wait_until(lambda: not s.status()["running"])
    time.sleep(0.3)
    st = s.status()

    # The sent play was reconciled (confirmed + checkpoint-logged) even
    # though the NEXT screen is a card_select; the executor did NOT get a
    # silent reset before reconciliation.
    assert bridge.actions == [
        {"action": "play", "card_index": 0, "target_index": 0},
        {"action": "choose", "index": 0},
    ], bridge.actions
    assert st["game_action_confirmed_count"] >= 1, st
    assert st["game_action_rejected_count"] == 0, st
    assert st["last_checkpoint_reason"] == "SCREEN_CHANGED", st
    # Old plan discarded (interrupted), never executed further.
    assert st["plan_interrupted_count"] == 1, st
    assert st["plan_completed_count"] == 0, st
    # card_select went through the NORMAL noncombat choice path.
    logs = s.logs_since(0)
    assert any(e["kind"] == "decision" and e.get("state_type") == "card_select"
               for e in logs), logs
    # No second combat action was executed from the old chunk.
    assert not any(a.get("action") == "end_turn" for a in bridge.actions)
    print("PASS TEST 3 screen_change_reconciles_inflight")


# ----------------------------------------------------------------
# TEST 4 — lethal action followed by reward_screen: plan COMPLETED
# ----------------------------------------------------------------

def test_lethal_action_reward_screen_plan_completed() -> None:
    class LLM(MockLLM):
        script = [CHUNK_STRIKE_ONCE, CHOOSE_0]

    s0 = combat_state("r0", energy=3, hand=[card("STRIKE", target="AnyEnemy")])
    s1 = reward_screen_state("r1")

    s, bridge = run_agent(9127, LLM, states=[s0, s1])
    wait_until(lambda: not s.status()["running"])
    time.sleep(0.3)
    st = s.status()

    assert bridge.actions == [
        {"action": "play", "card_index": 0, "target_index": 0},
        {"action": "choose", "index": 0},
    ], bridge.actions
    # The single planned action was confirmed and the chunk was exhausted:
    # plan COMPLETED even though the checkpoint reason is SCREEN_CHANGED.
    assert st["game_action_confirmed_count"] >= 1, st
    assert st["game_action_rejected_count"] == 0, st
    assert st["plan_completed_count"] == 1, st
    assert st["plan_interrupted_count"] == 0, st
    assert st["last_checkpoint_reason"] == "SCREEN_CHANGED", st
    print("PASS TEST 4 lethal_action_reward_screen_plan_completed")


# ----------------------------------------------------------------
# TEST 5 (P0) — bridge accepted + unchanged: WAIT, never re-prompt
# ----------------------------------------------------------------

def test_accepted_no_advance_does_not_reprompt() -> None:
    """KEY REGRESSION: an accepted command whose authoritative snapshot has
    not advanced is a WAITING state. Exactly ONE cognition, ONE gameplay
    command, the chunk RETAINED -- never a rejection and never a re-prompt
    against the stale combat snapshot."""
    class LLM(MockLLM):
        script = [CHUNK_STRIKE_DEFEND_END]

    s0 = combat_state("r0", energy=3,
                      hand=[card("STRIKE", target="AnyEnemy"), card("DEFEND")])
    # Human-visibly IDENTICAL state, only request_id differs.
    s1 = combat_state("r1", energy=3,
                      hand=[card("STRIKE", target="AnyEnemy"), card("DEFEND")])

    s, bridge = run_agent(9123, LLM, states=[s0, s1], bridge_wait=0.6)
    wait_until(lambda: not s.status()["running"])
    time.sleep(0.3)
    st = s.status()

    # EXACTLY one gameplay command ever sent: no end_turn / potion poke.
    assert bridge.actions == [
        {"action": "play", "card_index": 0, "target_index": 0},
    ], bridge.actions
    assert st["game_action_sent_count"] == 1, st
    assert st["game_action_confirmed_count"] == 0, st
    assert st["game_action_rejected_count"] == 0, st
    assert st["game_action_unconfirmable_count"] >= 1, st
    # ONE cognition: the unchanged snapshot must NOT re-prompt the model.
    assert st["llm_request_count"] == 1, st
    assert st["last_checkpoint_reason"] == "AWAITING_ADVANCE", st
    # The executor RETAINS the unresolved original action and its plan.
    assert s._plan_executor.inflight is not None
    assert s._plan_executor.has_pending_plan
    assert s._plan_executor.index == 0
    print("PASS TEST 5 accepted_no_advance_does_not_reprompt")


# ----------------------------------------------------------------
# TEST 5b — accepted + unchanged, THEN native card_select
# ----------------------------------------------------------------

def test_accepted_no_advance_then_native_selection() -> None:
    """Headbutt-class lifecycle: an accepted command first yields an
    unchanged combat snapshot, then the native card_select. The selection
    must own the next cognition; the triggering action must not be
    rejected and the stale chunk remainder must be discarded."""
    class LLM(MockLLM):
        script = [
            json.dumps({
                "thought": "headbutt then end",
                "actions": [
                    {"kind": "play", "card_ref": "h0", "target_ref": "e0"},
                    {"kind": "end_turn"},
                ],
            }),
            CHOOSE_0,
        ]

    s0 = combat_state("r0", energy=3, hand=[
        card("HEADBUTT", target="AnyEnemy"), card("STRIKE", target="AnyEnemy")])
    # Accepted but not yet observably advanced.
    s1 = combat_state("r1", energy=3, hand=[
        card("HEADBUTT", target="AnyEnemy"), card("STRIKE", target="AnyEnemy")])
    s2 = card_select_state("r2")

    s, bridge = run_agent(9133, LLM, states=[s0, s1, s2], bridge_wait=0.6)
    wait_until(lambda: not s.status()["running"])
    time.sleep(0.3)
    st = s.status()

    assert bridge.actions == [
        {"action": "play", "card_index": 0, "target_index": 0},
        {"action": "choose", "index": 0},
    ], bridge.actions
    # Exactly TWO cognitions: the plan, then the selection decision. The
    # unchanged snapshot produced NO extra prompt.
    assert st["llm_request_count"] == 2, st
    # The triggering action was NOT rejected.
    assert st["game_action_rejected_count"] == 0, st
    assert st["game_action_confirmed_count"] >= 1, st
    # The stale chunk remainder (end_turn) is gone, never executed.
    assert not any(a.get("action") == "end_turn" for a in bridge.actions)
    assert st["plan_interrupted_count"] == 1, st
    logs = s.logs_since(0)
    cps = [e for e in logs if e["kind"] == "plan_checkpoint"
           and e.get("reason") == "SCREEN_CHANGED"]
    assert cps and cps[0].get("remaining_steps_discarded") == 1, cps
    # The selection screen got foreground ownership (normal choose path).
    assert any(e["kind"] == "decision" and e.get("state_type") == "card_select"
               for e in logs), logs
    print("PASS TEST 5b accepted_no_advance_then_native_selection")


# ----------------------------------------------------------------
# TEST 5c — accepted + unchanged, THEN normal advance: confirm once
# ----------------------------------------------------------------

def test_accepted_no_advance_then_normal_advance() -> None:
    class LLM(MockLLM):
        script = [CHUNK_STRIKE_END]

    s0 = combat_state("r0", energy=3, hand=[card("STRIKE", target="AnyEnemy")])
    # Accepted but unchanged.
    s1 = combat_state("r1", energy=3, hand=[card("STRIKE", target="AnyEnemy")])
    # The visible world finally advances: energy down, enemy hp down,
    # Strike left the hand.
    s2 = combat_state("r2", energy=2, hand=[], enemies=[cultist(34)],
                      discard_count=1)

    s, bridge = run_agent(9134, LLM, states=[s0, s1, s2], bridge_wait=0.6)
    wait_until(lambda: not s.status()["running"])
    time.sleep(0.3)
    st = s.status()

    # No extra action at S1; at S2 the SAME in-flight action was confirmed
    # and the chunk continued with its committed end_turn.
    assert bridge.actions == [
        {"action": "play", "card_index": 0, "target_index": 0},
        {"action": "end_turn"},
    ], bridge.actions
    assert st["llm_request_count"] == 1, st
    assert st["game_action_confirmed_count"] == 1, st
    assert st["game_action_rejected_count"] == 0, st
    assert st["game_action_sent_count"] == 2, st
    # The S1 observation is a transient awaiting observation (observation
    # counter), NOT a second final outcome.
    assert st["game_action_unconfirmable_count"] >= 1, st
    print("PASS TEST 5c accepted_no_advance_then_normal_advance")


# ----------------------------------------------------------------
# TEST 5d — a genuinely REJECTED command is still rejected
# ----------------------------------------------------------------

def test_explicit_bridge_rejection_is_rejected() -> None:
    """No bridge acceptance marker + action-relevantly unchanged state =>
    a REAL rejection: rejected += 1, unconfirmable unchanged, plan reset so
    fresh cognition is permitted."""
    s = AgentSession()
    s._metrics = BenchmarkMetrics()
    s._logs = []
    s._stop = threading.Event()
    s._pending_action_result = ""  # bridge reported NO acceptance

    s0 = combat_state("r0", energy=3, hand=[card("STRIKE", target="AnyEnemy")])
    s1 = combat_state("r1", energy=3, hand=[card("STRIKE", target="AnyEnemy")])
    chunk = parse_action_chunk({
        "thought": "strike once",
        "actions": [{"kind": "play", "card_ref": "h0", "target_ref": "e0"}],
    }, s0, max_actions=8)
    ex = s._plan_executor
    ex.submit(chunk)
    ev = ex.prepare_next(s0, agent_mod.validate_action)
    ex.mark_sent(ev.prepared, s0)

    event = s._reconcile_inflight_if_any(s1)
    assert event.status == ExecutorStatus.NEED_MODEL, event
    assert event.checkpoint.reason == CheckpointReason.ACTION_REJECTED, event
    assert s._metrics.game_action_rejected_count == 1
    assert s._metrics.game_action_unconfirmable_count == 0
    assert not ex.has_pending_plan
    print("PASS TEST 5d explicit_bridge_rejection_is_rejected")


# ----------------------------------------------------------------
# TEST 5e — repeated accepted + unchanged -> bounded PROTOCOL_STALL
# ----------------------------------------------------------------

def test_bounded_stall_on_repeated_accepted_no_advance() -> None:
    class LLM(MockLLM):
        script = [CHUNK_STRIKE_ONCE]

    s0 = combat_state("r0", energy=3, hand=[card("STRIKE", target="AnyEnemy")])
    repeats = [
        combat_state(f"r{i}", energy=3,
                     hand=[card("STRIKE", target="AnyEnemy")])
        for i in range(1, PROTOCOL_STALL_REPEATS + 1)
    ]

    s, bridge = run_agent(9132, LLM, states=[s0] + repeats, bridge_wait=0.4)
    wait_until(lambda: not s.status()["running"])
    time.sleep(0.3)
    st = s.status()

    # No strategic action was ever emitted to "poke" the state forward.
    assert bridge.actions == [
        {"action": "play", "card_index": 0, "target_index": 0},
    ], bridge.actions
    assert st["llm_request_count"] == 1, st
    assert st["game_action_sent_count"] == 1, st
    assert st["benchmark_valid"] is False, st
    assert "PROTOCOL_STALL" in st["invalidation_reason"], st
    assert s._stop.is_set()
    print("PASS TEST 5e bounded_stall_on_repeated_accepted_no_advance")


# ----------------------------------------------------------------
# TEST 9 — explicit checkpoint_after=true
# ----------------------------------------------------------------

def test_explicit_checkpoint() -> None:
    class LLM(MockLLM):
        script = [CHUNK_STRIKE_INSPECT, CHUNK_END_TURN]

    s0 = combat_state("r0", energy=3, hand=[card("STRIKE", target="AnyEnemy")])
    s1 = combat_state("r1", energy=2, hand=[], enemies=[cultist(34)],
                      discard_count=1)

    s, bridge = run_agent(9124, LLM, states=[s0, s1])
    wait_until(lambda: not s.status()["running"])
    time.sleep(0.3)
    st = s.status()

    assert len(bridge.actions) == 2, bridge.actions
    assert st["llm_request_count"] == 2, st  # stopped after ONE action
    assert st["last_checkpoint_reason"] == "MODEL_REQUESTED", st
    assert st["game_action_sent_count"] == 2, st
    assert st["game_action_confirmed_count"] == 1, st
    print("PASS TEST 9 explicit_checkpoint")


# ----------------------------------------------------------------
# Single-action confirmation semantics (identical to chunks):
#   SENT -> next authoritative state -> CONFIRMED / REJECTED
# ----------------------------------------------------------------

def test_single_action_confirmed_by_next_state() -> None:
    class LLM(MockLLM):
        script = [
            json.dumps({"thought": "hit it",
                        "action": "play", "card_index": 0, "target_index": 0}),
            json.dumps({"thought": "done", "action": "end_turn"}),
        ]

    s0 = combat_state("r0", energy=3, hand=[card("STRIKE", target="AnyEnemy")])
    # The strike LANDED: enemy HP down, energy down, Strike left the hand.
    s1 = combat_state("r1", energy=2, hand=[],
                      enemies=[cultist(34)], discard_count=1)

    s, bridge = run_agent(9128, LLM, cfg={"decision_mode": "single_action"},
                          states=[s0, s1])
    wait_until(lambda: not s.status()["running"])
    time.sleep(0.3)
    st = s.status()

    assert bridge.actions[0] == {"action": "play", "card_index": 0,
                                 "target_index": 0}
    # The visible world moved forward -> the sent strike is CONFIRMED by
    # the next authoritative state (same definition as chunks).
    assert st["game_action_sent_count"] == 2, st  # strike + re-prompted end_turn
    assert st["game_action_confirmed_count"] == 1, st
    assert st["game_action_rejected_count"] == 0, st
    assert st["llm_request_count"] == 2, st
    print("PASS single_action_confirmed_by_next_state")


def test_single_action_accepted_no_advance_waits() -> None:
    """Single-action mode: an ACCEPTED command whose snapshot is unchanged
    is a WAITING state -- no second cognition, no second gameplay action,
    and the pending action is retained for later reconciliation."""
    class LLM(MockLLM):
        script = [
            json.dumps({"thought": "hit it",
                        "action": "play", "card_index": 0, "target_index": 0}),
            json.dumps({"thought": "must never be asked", "action": "end_turn"}),
        ]

    s0 = combat_state("r0", energy=3, hand=[card("STRIKE", target="AnyEnemy")])
    # Human-visibly IDENTICAL state, only request_id differs; the bridge
    # ACCEPTED the play ("played card 0").
    s1 = combat_state("r1", energy=3, hand=[card("STRIKE", target="AnyEnemy")])

    s, bridge = run_agent(9129, LLM, cfg={"decision_mode": "single_action"},
                          states=[s0, s1], bridge_wait=0.6)
    wait_until(lambda: not s.status()["running"])
    time.sleep(0.3)
    st = s.status()

    # Exactly one gameplay command was ever sent (no poked end_turn).
    assert bridge.actions == [{"action": "play", "card_index": 0,
                               "target_index": 0}], bridge.actions
    assert st["llm_request_count"] == 1, st
    assert st["game_action_sent_count"] == 1, st
    assert st["game_action_confirmed_count"] == 0, st
    assert st["game_action_rejected_count"] == 0, st
    assert st["game_action_unconfirmable_count"] >= 1, st
    # The pending single action is RETAINED for later reconciliation.
    assert s._pending_single_action is not None
    print("PASS single_action_accepted_no_advance_waits")


# ----------------------------------------------------------------
# TEST 6 — strict failure: no fallback, request accounting
# ----------------------------------------------------------------

def test_strict_failure() -> None:
    class LLM(MockLLM):
        fail = True

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
    # The failed inference attempt IS counted (FIX: request accounting).
    assert st["llm_request_count"] == 1, st
    assert st["llm_failed_request_count"] == 1, st
    assert st["llm_success_count"] == 0, st
    assert st["game_action_sent_count"] == 0, st
    print("PASS TEST 6 strict_failure")


# ----------------------------------------------------------------
# Real-smoke regression: in action_chunk mode, NON-COMBAT screens use
# the single-action path -- their sends MUST still be reconciled against
# the next authoritative state (bug found in the first real-game smoke:
# sent=4 but confirmed=0, rejected choices looped until game timeout).
# ----------------------------------------------------------------

def event_state(request_id: str, label: str) -> dict:
    return {
        "type": "event",
        "request_id": request_id,
        "floor": 2, "act": 1,
        "options": [{"index": 0, "label": label, "enabled": True}],
    }


def test_chunk_mode_noncombat_action_is_confirmed() -> None:
    class LLM(MockLLM):
        script = [
            json.dumps({"thought": "leave", "action": "choose", "index": 0}),
            json.dumps({"thought": "pick", "action": "choose", "index": 0}),
        ]

    s0 = event_state("r0", "Take the gold")
    s1 = event_state("r1", "Different event body")  # visibly different

    s, bridge = run_agent(9130, LLM, states=[s0, s1])
    wait_until(lambda: not s.status()["running"])
    time.sleep(0.3)
    st = s.status()

    assert bridge.actions[0] == {"action": "choose", "index": 0}
    assert st["game_action_sent_count"] >= 1, st
    # The non-combat choose was CONFIRMED by the next authoritative state
    # even though decision_mode is action_chunk.
    assert st["game_action_confirmed_count"] >= 1, st
    assert st["game_action_rejected_count"] == 0, st
    print("PASS chunk_mode_noncombat_action_is_confirmed")


def test_chunk_mode_noncombat_accepted_no_advance_waits() -> None:
    """In action_chunk mode non-combat screens use the single-action path;
    an ACCEPTED choose whose event snapshot is unchanged must WAIT (no
    second cognition, no second choose), not be re-prompted."""
    class LLM(MockLLM):
        script = [
            json.dumps({"thought": "leave", "action": "choose", "index": 0}),
            json.dumps({"thought": "must never be asked",
                        "action": "choose", "index": 0}),
        ]

    s0 = event_state("r0", "Take the gold")
    s1 = event_state("r1", "Take the gold")  # IDENTICAL visible state

    s, bridge = run_agent(9131, LLM, states=[s0, s1], bridge_wait=0.6)
    wait_until(lambda: not s.status()["running"])
    time.sleep(0.3)
    st = s.status()

    assert bridge.actions == [{"action": "choose", "index": 0}], bridge.actions
    assert st["llm_request_count"] == 1, st
    assert st["game_action_rejected_count"] == 0, st
    assert st["game_action_unconfirmable_count"] >= 1, st
    assert st["game_action_confirmed_count"] == 0, st
    print("PASS chunk_mode_noncombat_accepted_no_advance_waits")


def run_all() -> None:
    tests = [
        test_one_plan_three_confirmed_actions,
        test_draw_checkpoint_reprompt_contains_full_new_card_information,
        test_screen_change_reconciles_inflight_before_routing,
        test_lethal_action_reward_screen_plan_completed,
        test_accepted_no_advance_does_not_reprompt,
        test_accepted_no_advance_then_native_selection,
        test_accepted_no_advance_then_normal_advance,
        test_explicit_bridge_rejection_is_rejected,
        test_bounded_stall_on_repeated_accepted_no_advance,
        test_explicit_checkpoint,
        test_strict_failure,
        test_single_action_confirmed_by_next_state,
        test_single_action_accepted_no_advance_waits,
        test_chunk_mode_noncombat_action_is_confirmed,
        test_chunk_mode_noncombat_accepted_no_advance_waits,
    ]
    for fn in tests:
        fn()
    print(f"\nALL {len(tests)} BETA E2E TESTS PASSED")


if __name__ == "__main__":
    run_all()
