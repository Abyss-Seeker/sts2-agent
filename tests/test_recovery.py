"""Recovery / continuity unit tests (A/B pre-flight hardening).

  chunk_send_error_not_counted_sent / leaves_no_inflight (D)
  recoverable_terminated_preserves_run_context (F/H)
  recoverable_terminated_reconnects_existing_game (G)
  recoverable_terminated_relaunches_missing_game (G)
  recovery_reapplies_bridge_settings
  true_victory_does_not_resume_same_run / defeat (E/T)
  runner_safe_to_disconnect (V)

Run:  python .\\tests\\test_recovery.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent as agent_mod
from agent import AgentSession
from plan_executor import ActionKind, PlannedAction, PreparedStep


def make_session() -> AgentSession:
    s = AgentSession()
    s._run_id = "run_old"
    return s


# ----------------------------------------------------------------
# D — ActionChunk transport send ERROR must NOT count as SENT
# ----------------------------------------------------------------

def test_chunk_send_error_not_counted_sent() -> None:
    s = make_session()
    m = s._metrics
    prepared = PreparedStep(
        plan_id="plan_x", step_index=0,
        planned=PlannedAction(kind=ActionKind.PLAY, card_ref="h0"),
        bridge_action={"action": "end_turn"}, description="play",
    )
    s._execute = lambda act: "ERROR: Not connected to STS2 bridge"
    s._send_prepared_plan_step(prepared, {"type": "combat_action"})

    assert m.game_action_sent_count == 0, m.snapshot()
    assert m.game_action_confirmed_count == 0
    assert m.executed_planned_actions_total == 0
    assert m.transport_interrupted_action_count == 1, m.snapshot()
    # no phantom inflight
    assert s._plan_executor.inflight is None
    print("PASS chunk_send_error_not_counted_sent")


def test_chunk_send_error_leaves_no_inflight_and_plan_retryable() -> None:
    s = make_session()
    # A real executor with a pending plan: the same step must stay
    # pending (retryable) after a transport failure.
    from action_plan import parse_action_chunk

    state = {
        "type": "combat_action", "round": 1,
        "hand": [{"id": "STRIKE", "target": "AnyEnemy", "playable": True}],
        "enemies": [{"id": "C", "is_alive": True}],
        "potions": [],
    }
    chunk = agent_mod.parse_action_chunk(
        {"thought": "t", "actions": [{"kind": "play", "card_ref": "h0",
                                      "target_ref": "e0"}]},
        state,
    )
    ex = s._plan_executor
    ex.submit(chunk)
    ev = ex.prepare_next(state, agent_mod.validate_action)
    assert ev.status.value == "READY_ACTION"
    prepared = ev.prepared

    s._execute = lambda act: "ERROR: Not connected to STS2 bridge"
    s._send_prepared_plan_step(prepared, state)

    assert ex.inflight is None
    assert ex.has_pending_plan  # plan still pending, same step index
    assert s._metrics.game_action_sent_count == 0
    # The step can be prepared again after a reconnect.
    ev2 = ex.prepare_next(state, agent_mod.validate_action)
    assert ev2.status.value == "READY_ACTION"
    print("PASS chunk_send_error_leaves_no_inflight")


# ----------------------------------------------------------------
# E/T — victory / defeat finalize the run context (new run)
# ----------------------------------------------------------------

def test_true_victory_does_not_resume_same_run() -> None:
    s = make_session()
    s._memory.observe({"type": "combat_action", "floor": 9,
                       "player": {"hp": 10, "gold": 5}})
    s._ctx.add_decision("state", "resp", "note")
    old_id = s._run_id
    s._finalize_run_context()
    assert s._run_id != old_id and s._run_id, "new run_id expected"
    assert s._memory.floor == 0 and s._memory.gold is None  # fresh memory
    assert s._ctx.history == []  # fresh LLM history
    print("PASS true_victory_does_not_resume_same_run")


def test_true_defeat_does_not_resume_same_run() -> None:
    s = make_session()
    s._memory.observe({"type": "combat_action", "floor": 3,
                       "player": {"hp": 1}})
    s._finalize_run_context()
    assert s._memory.floor == 0  # defeat = new run, fresh memory
    print("PASS true_defeat_does_not_resume_same_run")


# ----------------------------------------------------------------
# F/G/H — recoverable termination keeps the SAME run context
# ----------------------------------------------------------------

def test_recoverable_terminated_preserves_run_context() -> None:
    s = make_session()
    s._memory.observe({"type": "combat_action", "floor": 7,
                       "player": {"hp": 45, "gold": 99}})
    s._ctx.add_decision("state", "resp", "note")
    # An unconfirmed in-flight plan action must be dropped and counted.
    s._metrics.record_action_sent(from_plan=True, combat=True)
    s._plan_executor._inflight = object()  # simulate inflight marker

    calls: list[bool] = []
    s._maybe_resume = lambda *, same_run: calls.append(same_run) or True

    state = {"type": "game_over", "result": "terminated"}
    assert s._recover_interrupted_run(state) is True
    assert calls == [True], calls  # SAME run resume
    assert s._metrics.recoverable_termination_count == 1
    # run context preserved
    assert s._memory.floor == 7 and s._memory.hp == 45
    assert s._ctx.history, "LLM history must be preserved"
    assert s._run_id == "run_old"

    # Stale transport state: dropping it counts the unconfirmed action
    # (neither confirmed nor rejected) and clears the executor.
    s._clear_stale_transport_state()
    assert s._metrics.transport_interrupted_action_count == 1
    assert s._plan_executor.inflight is None
    assert not ex_has_plan(s)
    print("PASS recoverable_terminated_preserves_run_context")


def ex_has_plan(s: AgentSession) -> bool:
    return s._plan_executor.has_pending_plan


# ----------------------------------------------------------------
# G + reapply — reconnect / relaunch with settings re-applied
# ----------------------------------------------------------------

def _install_fake_game_launcher(missing_game: bool):
    state = {"running": not missing_game, "launches": 0}

    def is_game_running():
        return state["running"]

    def launch_via_steam(*a, **k):
        state["launches"] += 1
        state["running"] = True

    fake = types.SimpleNamespace(
        is_game_running=is_game_running,
        launch_via_steam=launch_via_steam,
    )
    import game_launcher as gl

    original = (gl.is_game_running, gl.launch_via_steam)
    gl.is_game_running = is_game_running
    gl.launch_via_steam = launch_via_steam
    return gl, state, original


class _FakeClient:
    def __init__(self, **k):
        self.fallback_calls: list[bool] = []
        self.timeout_calls: list[int] = []

    def connect(self, should_abort=None):
        return

    def disconnect(self):
        pass

    def set_fallback(self, enabled: bool):
        self.fallback_calls.append(enabled)

    def set_agent_timeout(self, seconds: int):
        self.timeout_calls.append(seconds)


def test_recoverable_terminated_reconnects_existing_game() -> None:
    gl, state, original = _install_fake_game_launcher(missing_game=False)
    agent_mod.STS2GameClient = _FakeClient
    try:
        s = make_session()
        s._config["auto_resume"] = True
        s._config["resume_wait_seconds"] = 20
        state_before = (s._memory.floor, s._run_id)
        assert s._recover_interrupted_run({"type": "game_over"}) is True
        assert s._metrics.bridge_reconnect_count == 1
        assert s._metrics.safe_recovery_count >= 1
        assert s._metrics.game_relaunch_count == 0  # game was still running
        assert s._memory.floor == state_before[0]  # run context preserved
        assert s._run_id == state_before[1]
    finally:
        gl.is_game_running, gl.launch_via_steam = original
    print("PASS recoverable_terminated_reconnects_existing_game")


def test_recoverable_terminated_relaunches_missing_game() -> None:
    gl, state, original = _install_fake_game_launcher(missing_game=True)
    agent_mod.STS2GameClient = _FakeClient
    try:
        s = make_session()
        s._config["auto_resume"] = True
        s._config["resume_wait_seconds"] = 20
        assert s._recover_interrupted_run({"type": "game_over"}) is True
        assert state["launches"] == 1
        assert s._metrics.game_relaunch_count == 1, s._metrics.snapshot()
        assert s._metrics.bridge_reconnect_count == 1
        assert s._run_id == "run_old"  # same save -> same run
    finally:
        gl.is_game_running, gl.launch_via_steam = original
    print("PASS recoverable_terminated_relaunches_missing_game")


def test_recovery_reapplies_bridge_settings() -> None:
    gl, state, original = _install_fake_game_launcher(missing_game=False)
    agent_mod.STS2GameClient = _FakeClient
    try:
        s = make_session()
        s._config["auto_resume"] = True
        s._config["resume_wait_seconds"] = 20
        s._config["disable_fallback"] = True
        s._config["agent_timeout"] = 120
        assert s._recover_interrupted_run({"type": "game_over"}) is True
        client = s._client
        assert client.fallback_calls == [False], client.fallback_calls
        assert client.timeout_calls == [120], client.timeout_calls
    finally:
        gl.is_game_running, gl.launch_via_steam = original
    print("PASS recovery_reapplies_bridge_settings")


# ----------------------------------------------------------------
# V — runner-facing safe_to_disconnect
# ----------------------------------------------------------------

def test_runner_safe_to_disconnect() -> None:
    s = make_session()
    s._running = True
    assert s.status()["safe_to_disconnect"] is True

    s._llm_inflight = 1
    assert s.status()["safe_to_disconnect"] is False
    s._llm_inflight = 0

    s._pending_single_action = {"action": "end_turn"}
    assert s.status()["safe_to_disconnect"] is False
    s._pending_single_action = None

    s._plan_executor._inflight = object()
    assert s.status()["safe_to_disconnect"] is False
    s._plan_executor.reset()

    s._agent_phase = "thinking"
    assert s.status()["safe_to_disconnect"] is False
    s._agent_phase = "idle"
    assert s.status()["safe_to_disconnect"] is True
    print("PASS runner_safe_to_disconnect")


def run_all() -> None:
    tests = [
        test_chunk_send_error_not_counted_sent,
        test_chunk_send_error_leaves_no_inflight_and_plan_retryable,
        test_true_victory_does_not_resume_same_run,
        test_true_defeat_does_not_resume_same_run,
        test_recoverable_terminated_preserves_run_context,
        test_recoverable_terminated_reconnects_existing_game,
        test_recoverable_terminated_relaunches_missing_game,
        test_recovery_reapplies_bridge_settings,
        test_runner_safe_to_disconnect,
    ]
    for fn in tests:
        fn()
    print(f"\nALL {len(tests)} RECOVERY TESTS PASSED")


if __name__ == "__main__":
    run_all()
