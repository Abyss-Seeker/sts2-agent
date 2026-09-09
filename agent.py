"""LLM agent loop: state -> prompt -> LLM -> validated action -> game.

The bridge protocol is synchronous: the game sends one state and waits for
one action. The agent therefore runs a simple blocking loop in a worker
thread, asking the LLM for a JSON decision for every state.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from collections import deque
from typing import Any

from bridge_client import (
    BridgeAction,
    BridgeStateType,
    CHOICE_SCREEN_TYPES,
    TERMINAL_SCREEN_TYPES,
    STS2GameClient,
)
from context_manager import ContextConfig, ContextManager
from game_state import (
    RunMemory,
    format_state,
    is_unsupported_state,
    _option_entries,
    skip_allowed,
)
from llm_client import EmptyContentError, LLMClient, LLMError
from prompts import DEFAULT_SYSTEM_TEMPLATE, DEFAULT_USER_TEMPLATE, RULEBOOK, CONTRACT

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    "bridge_host": "127.0.0.1",
    "bridge_port": 9002,
    "api_base_url": "https://api.openai.com",
    "api_key": "",
    "model": "gpt-4o-mini",
    "temperature": 0.4,
    "max_tokens": 512,
    "llm_timeout": 25,
    # transport_retries: HTTP-level retries for ONE LLM API call (0 keeps the
    # wall time inside the game's decision window). Distinct from
    # decision_validation_attempts below.
    "llm_retries": 0,
    # decision_validation_attempts: how many LLM decisions (parse/validate
    # attempts) may be made for ONE state. All attempts share one deadline.
    "decision_attempts": 3,
    "system_template": DEFAULT_SYSTEM_TEMPLATE,
    "user_template": DEFAULT_USER_TEMPLATE,
    "max_history_turns": 8,
    "max_state_chars": 4000,
    "max_context_chars": 24000,
    "disable_fallback": True,
    "save_log": True,
    "auto_launch_game": True,
    "steam_appid": "2868840",
    "action_delay": 0.0,
    "show_thinking": "brief",  # full | brief | hidden
    "agent_timeout": 90,  # game-side per-decision wait (10..300s)
    # Auto-resume: an aborted run KEEPS its save. Instead of ending the
    # session, wait for the game to come back -- the mod clicks "Continue"
    # and resumes the saved run by itself, so no progress is lost.
    "auto_resume": True,
    "resume_wait_seconds": 600,
    # Forensic: dump every raw LLM API response body to logs/llm_raw_*.jsonl
    # (used to pinpoint where a relay/proxy corrupts replies). Safe to turn
    # off once the relay is fixed.
    "dump_raw_responses": True,
}


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM reply (tolerates fences)."""
    if not text or not text.strip():
        raise ValueError("empty reply")
    cleaned = re.sub(r"```(?:json)?", "", text.strip()).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # Last resort: json-repair handles trailing commas, single quotes,
    # unquoted keys, python-style literals, etc. Its result is only
    # accepted when it actually contains an "action" field: for replies
    # corrupted by a streaming relay (bytes missing from the middle /
    # opening '{"thought":"' eaten), repair can produce a dict whose key
    # is a garbage text fragment with NO action -- such a result must not
    # be returned; the salvage path below handles those instead.
    try:
        from json_repair import repair_json

        obj = repair_json(cleaned, return_objects=True)
        if isinstance(obj, dict) and obj.get("action"):
            return obj
    except ImportError:
        pass
    except Exception:
        pass
    # Salvage attempt for replies corrupted by a streaming proxy: bytes go
    # missing from the middle of the content while reasoning fragments get
    # spliced in, e.g.
    #   'Explanation: FISTICUFFS is moreChoose the efficient 1-energy
    #    attack.","action":"choose","index":0}'
    # The opening '{"thought":"...' is gone but the tail starting at the
    # "action" (or "thought") key is intact. Rebuild a JSON object from the
    # last occurrence so a decision can still be parsed; "thought" may be
    # lost, which the reasoning check handles separately.
    for key in ('"action"', '"thought"'):
        pos = cleaned.rfind(key)
        if pos > 0:
            candidate = cleaned[pos - 1 if cleaned[pos - 1] == "{" else pos:]
            if not candidate.lstrip().startswith("{"):
                candidate = "{" + candidate
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict) and "action" in obj:
                    return obj
            except json.JSONDecodeError:
                pass
            try:
                from json_repair import repair_json

                obj = repair_json(candidate, return_objects=True)
                if isinstance(obj, dict) and obj.get("action"):
                    return obj
            except Exception:
                pass
    raise ValueError(
        "no valid JSON object found in reply (reply may be corrupted by the"
        " LLM relay/proxy -- check its streaming settings):"
        f" {text[:400]!r}"
    )


def render_template(template: str, values: dict[str, str]) -> str:
    out = template
    for key, val in values.items():
        out = out.replace("{{" + key + "}}", val)
    return out


# ----------------------------------------------------------------
# Action validation
# ----------------------------------------------------------------

# Common key aliases sloppy LLMs produce instead of the contract fields.
ACTION_KEY_ALIASES = {
    "card": "card_index",
    "card_idx": "card_index",
    "cardindex": "card_index",
    "target": "target_index",
    "target_idx": "target_index",
    "targetindex": "target_index",
    "enemy": "target_index",
    "enemy_index": "target_index",
    "choice": "index",
    "option": "index",
    "option_index": "index",
    "choice_index": "index",
    "slot_index": "slot",
    "potion_slot": "slot",
}


def _normalize_keys(act: dict[str, Any]) -> dict[str, Any]:
    out = dict(act)
    for alias, canonical in ACTION_KEY_ALIASES.items():
        if alias in out and canonical not in out:
            out[canonical] = out[alias]
    return out


def _as_int(v: Any) -> int | None:
    """Lenient integer coercion: 1, 1.0, "1" all work; bools/None/garbage fail."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if s.lstrip("-").isdigit():
            return int(s)
    return None


def _option_disabled(opt: Any) -> bool:
    return isinstance(opt, dict) and opt.get("enabled", True) is False


def validate_action(
    state: dict[str, Any], act: dict[str, Any]
) -> tuple[dict[str, Any] | None, str]:
    """Validate an LLM action dict against the current state.

    Returns (normalized_action, "") or (None, error_reason).
    """
    act = _normalize_keys(act)
    name = str(act.get("action", "")).lower().strip()
    if not name:
        return None, "missing 'action' field"

    if name in ("end_turn", "endturn", "end"):
        return {"action": BridgeAction.END_TURN}, ""

    if name == "play":
        hand = state.get("hand") or []
        ci = _as_int(act.get("card_index"))
        if ci is None:
            return None, f"'card_index' must be an integer, got: {act.get('card_index')!r}"
        if ci < 0 or ci >= len(hand):
            return None, f"card_index {ci} out of range (hand has {len(hand)} cards)"
        card = hand[ci] or {}
        if card.get("playable") is False:
            return None, f"card [{ci}] {card.get('id')} is not playable right now"
        enemies = state.get("enemies") or []
        ti = _as_int(act.get("target_index", -1))
        if ti is None:
            return None, f"'target_index' must be an integer, got: {act.get('target_index')!r}"
        target = str(card.get("target", "None"))
        if target == "AnyEnemy":
            if ti < 0 or ti >= len(enemies):
                return None, (
                    f"card [{ci}] {card.get('id')} targets AnyEnemy: target_index"
                    f" {ti} invalid (enemies: {len(enemies)})"
                )
            if not (enemies[ti] or {}).get("is_alive", False):
                return None, f"enemy [{ti}] is dead; pick a living enemy"
        return {"action": BridgeAction.PLAY, "card_index": ci, "target_index": ti}, ""

    if name == "potion":
        potions = state.get("potions") or []
        slot = _as_int(act.get("slot"))
        if slot is None:
            return None, f"'slot' must be an integer, got: {act.get('slot')!r}"
        if slot < 0 or slot >= len(potions):
            return None, f"potion slot {slot} out of range ({len(potions)} slots)"
        potion = potions[slot] or {}
        if potion.get("can_use") is False:
            return None, f"potion in slot {slot} cannot be used now"
        ti = _as_int(act.get("target_index", -1))
        if ti is None:
            return None, f"'target_index' must be an integer, got: {act.get('target_index')!r}"
        if potion.get("requires_target"):
            enemies = state.get("enemies") or []
            if ti < 0 or ti >= len(enemies) or not (enemies[ti] or {}).get("is_alive", False):
                return None, f"potion in slot {slot} needs a living enemy target"
        return {"action": BridgeAction.POTION, "slot": slot, "target_index": ti}, ""

    if name == "choose":
        if str(state.get("type")) == BridgeStateType.COMBAT_ACTION:
            return None, (
                '"choose" is NOT a legal action in combat. Legal actions are:'
                ' {"action":"play","card_index":N,"target_index":N},'
                ' {"action":"potion","slot":N,"target_index":N} or'
                ' {"action":"end_turn"}.'
            )
        options = _option_entries(state)

        def _check_selection_count(count: int) -> str:
            if str(state.get("type")) == BridgeStateType.CARD_SELECT:
                try:
                    min_select = int(state.get("min_select", 0) or 0)
                    max_select = int(state.get("max_select", 1) or 1)
                except (TypeError, ValueError):
                    return ""
                if count < min_select:
                    return (
                        f"selection too small: {count} selected, this screen"
                        f" requires at least {min_select}"
                    )
                if count > max_select:
                    return (
                        f"selection too large: {count} selected, this screen"
                        f" allows at most {max_select}"
                    )
            return ""

        indexes = act.get("indexes")
        if isinstance(indexes, list) and indexes:
            vals: list[int] = []
            for v in indexes:
                iv = _as_int(v)
                if iv is None:
                    return None, f"'indexes' must be integers, got: {v!r}"
                if options and (iv < 0 or iv >= len(options)):
                    return None, f"index {iv} out of range ({len(options)} options)"
                if options and _option_disabled(options[iv]):
                    return None, f"option {iv} is DISABLED; pick an enabled option"
                vals.append(iv)
            err = _check_selection_count(len(vals))
            if err:
                return None, err
            return {"action": BridgeAction.CHOOSE, "indexes": vals}, ""
        idx = _as_int(act.get("index"))
        if idx is None:
            return None, f"'index' must be an integer, got: {act.get('index')!r}"
        if options and (idx < 0 or idx >= len(options)):
            return None, f"index {idx} out of range ({len(options)} options)"
        if options and _option_disabled(options[idx]):
            return None, f"option {idx} is DISABLED; pick an enabled option"
        err = _check_selection_count(1)
        if err:
            return None, err
        return {"action": BridgeAction.CHOOSE, "index": idx}, ""

    if name == "skip":
        # The mod only honours skip on card_reward / card_select(min_select=0);
        # elsewhere it degenerates into a RANDOM pick (or buys shop item 0).
        if not skip_allowed(state):
            return None, (
                "skip is NOT allowed on this screen: the game would fall back to a"
                " RANDOM option (or buy the first shop item). Use 'choose' with a"
                " valid index instead."
            )
        return {"action": BridgeAction.SKIP}, ""

    return None, f"unknown action {name!r}"


def fallback_action(state: dict[str, Any], start_index: int = 0) -> dict[str, Any]:
    """Deterministic, protocol-safe fallback when no LLM decision is
    available. NEVER sends an action the validator would reject and NEVER
    triggers the game's random-choice path: skip is only used on screens
    where it is explicitly allowed (card reward with can_skip); the shop
    leaves through its explicit 'Leave shop' option (index 0)."""
    stype = str(state.get("type", ""))
    if stype == BridgeStateType.COMBAT_ACTION:
        # Play the LEFTMOST playable card rather than passing: in combat a
        # "pass" (end_turn) throws the whole turn away even when cards could
        # still have been played. Each fallback plays one card, so repeated
        # fallbacks naturally chew through the hand from left to right,
        # turning "3 failures -> lose the turn" into "3 failures -> still
        # spend the turn". Only when nothing is playable do we end the turn.
        hand = state.get("hand") or []
        enemies = state.get("enemies") or []
        for i, card in enumerate(hand):
            if i < start_index:
                continue   # already tried by an earlier fallback on this hand
            if not isinstance(card, dict) or not card.get("playable"):
                continue
            target_index = -1
            if str(card.get("target", "None")) == "AnyEnemy":
                target_index = next(
                    (j for j, e in enumerate(enemies)
                     if isinstance(e, dict) and e.get("is_alive", False)),
                    None,
                )
                if target_index is None:
                    continue  # no living enemy to target -- try the next card
            return {
                "action": BridgeAction.PLAY,
                "card_index": i,
                "target_index": target_index,
            }
        return {"action": BridgeAction.END_TURN}
    if stype == BridgeStateType.SHOP:
        # Option 0 is always 'Leave shop' (mod-side constant).
        return {"action": BridgeAction.CHOOSE, "index": 0}
    if stype == BridgeStateType.CARD_REWARD and skip_allowed(state):
        return {"action": BridgeAction.SKIP}
    if stype in CHOICE_SCREEN_TYPES:
        # First ENABLED option (index 0 may be disabled).
        for i, opt in enumerate(_option_entries(state)):
            if not _option_disabled(opt):
                return {"action": BridgeAction.CHOOSE, "index": i}
        return {"action": BridgeAction.CHOOSE, "index": 0}
    # Unknown screen: last resort is the first enabled option as well,
    # never a random pick.
    for i, opt in enumerate(_option_entries(state)):
        if not _option_disabled(opt):
            return {"action": BridgeAction.CHOOSE, "index": i}
    return {"action": BridgeAction.SKIP}


# ----------------------------------------------------------------
# Agent session
# ----------------------------------------------------------------

class AgentSession:
    """One agent run lifecycle: connect -> decide loop -> stop.

    Naming:
      - transport_retries (config ``llm_retries``): HTTP retries for a
        single LLM API call (LLMClient.max_retries).
      - decision_validation_attempts (config ``decision_attempts``): how
        many parse/validate attempts ONE state gets. All attempts share a
        single decision deadline so retries can never outlive the game's
        decision window.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._client: STS2GameClient | None = None
        self._memory = RunMemory()
        self._ctx = ContextManager()
        self._config: dict[str, Any] = dict(DEFAULT_CONFIG)
        self._logs: deque[dict[str, Any]] = deque(maxlen=1000)
        self._log_seq = 0
        self._decision_count = 0
        self._running = False
        self._bridge_connected = False
        self._current_state_type = ""
        self._last_error = ""
        self._log_file = None
        self._llm_fail_streak = 0
        self._llm: LLMClient | None = None
        # (hand signature, card_index) of the last combat fallback, used to
        # avoid retrying a card the game refused.
        self._fallback_attempt: tuple[tuple[tuple[str, bool], ...], int] | None = None

    # ---------------- logging ----------------

    def _log(self, kind: str, text: str, **extra: Any) -> None:
        with self._lock:
            self._log_seq += 1
            entry = {
                "seq": self._log_seq,
                "ts": time.strftime("%H:%M:%S"),
                "kind": kind,
                "text": text,
            }
            entry.update(extra)
            self._logs.append(entry)
            if self._log_file is not None:
                try:
                    self._log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    self._log_file.flush()
                except Exception:
                    pass

    def logs_since(self, after: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self._logs if e["seq"] > after]

    # ---------------- lifecycle ----------------

    def start(self, config: dict[str, Any]) -> None:
        with self._lock:
            if self._running:
                raise RuntimeError("Agent already running")
            self._config = {**DEFAULT_CONFIG, **(config or {})}
            self._stop.clear()
            self._running = True
            self._last_error = ""
            self._decision_count = 0
            self._llm_fail_streak = 0
            self._thread = threading.Thread(
                target=self._run, name="llm-agent", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._running = False
        self._bridge_connected = False
        self._log("info", "Agent stopped.")

    def status(self) -> dict[str, Any]:
        with self._lock:
            llm = self._llm
            prompt_tokens = int(getattr(llm, "total_prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(llm, "total_completion_tokens", 0) or 0)
            model = str(self._config.get("model", "") or "")
            return {
                "running": self._running,
                "bridge_connected": self._bridge_connected,
                "decision_count": self._decision_count,
                "current_state_type": self._current_state_type,
                "last_error": self._last_error,
                "floor": self._memory.floor,
                "act": self._memory.act,
                "hp": self._memory.hp,
                "max_hp": self._memory.max_hp,
                "gold": self._memory.gold,
                "log_seq": self._log_seq,
                # LLM usage / model info for the overlay display.
                "model": str(self._config.get("model", "") or ""),
                "llm_prompt_tokens": prompt_tokens,
                "llm_completion_tokens": completion_tokens,
                "llm_total_tokens": prompt_tokens + completion_tokens,
            }

    # ---------------- main loop ----------------

    def _open_log_file(self) -> None:
        if not self._config.get("save_log", True):
            return
        try:
            from pathlib import Path

            log_dir = Path(__file__).resolve().parent / "logs"
            log_dir.mkdir(exist_ok=True)
            name = time.strftime("run_%Y%m%d_%H%M%S.jsonl")
            self._log_file = open(log_dir / name, "w", encoding="utf-8")
        except Exception as e:
            logger.warning("Cannot open log file: %s", e)
            self._log_file = None

    def _run(self) -> None:
        cfg = self._config
        self._open_log_file()
        from knowledge import warm_up

        warm_up()
        self._log(
            "info",
            f"Connecting to bridge {cfg['bridge_host']}:{cfg['bridge_port']}...",
        )

        # ---- Game process / launch handling ---------------------------
        # The bridge only sends states while a run is in progress. AutoSlay
        # starts one run automatically at game launch; after a run ends the
        # game must be restarted (or the user starts a run manually).
        cold_start = False
        game_on = False
        if cfg.get("auto_launch_game", True):
            try:
                from game_launcher import is_game_running

                game_on = is_game_running()
            except Exception as e:
                self._log("info", f"游戏进程检测失败（按已运行处理）: {e}")
                game_on = True
            if not game_on:
                from game_launcher import launch_via_steam

                self._log(
                    "info",
                    "未检测到游戏进程（SlayTheSpire2.exe），正在通过 Steam 启动；"
                    "mod 会在主菜单自动开一局。",
                )
                launch_via_steam(
                    str(cfg.get("steam_appid", "2868840")),
                    log=lambda m: self._log("info", m),
                )
                cold_start = True
        if game_on or not cfg.get("auto_launch_game", True):
            if game_on:
                self._log(
                    "info",
                    "检测到游戏已在运行：若不在局中，请自己开一局，Agent 会自动接管。",
                )

        attempts = 150 if cold_start else 30  # cold start can take minutes
        self._client = STS2GameClient(
            host=cfg["bridge_host"],
            port=int(cfg["bridge_port"]),
            reconnect_attempts=attempts,
        )
        try:
            self._client.connect(should_abort=self._stop.is_set)
        except ConnectionError as e:
            if self._stop.is_set:
                self._log("info", "连接已取消。")
                self._running = False
                if self._log_file is not None:
                    self._log_file.close()
                    self._log_file = None
                return
            self._fail(f"Bridge connection failed: {e}")
            return
        self._bridge_connected = True
        self._log("info", "Connected to game bridge.")

        if cfg.get("disable_fallback", True):
            try:
                self._client.set_fallback(False)
                self._log("info", "Random fallback disabled (all decisions are the LLM's).")
            except ConnectionError as e:
                self._fail(f"Lost bridge while disabling fallback: {e}")
                return

        # Push the per-decision wait to the mod (requires the rebuilt mod).
        try:
            agent_timeout = max(10, min(300, int(cfg.get("agent_timeout") or 90)))
        except (TypeError, ValueError):
            agent_timeout = 90
        try:
            self._client.set_agent_timeout(agent_timeout)
            self._log("info", f"游戏端每决策等待窗口已设为 {agent_timeout}s。")
        except ConnectionError as e:
            self._fail(f"Lost bridge while setting agent timeout: {e}")
            return

        self._memory = RunMemory()
        self._ctx = ContextManager(
            config=ContextConfig(
                max_history_turns=int(cfg["max_history_turns"]),
                max_state_chars=int(cfg["max_state_chars"]),
                max_context_chars=int(cfg["max_context_chars"]),
            )
        )
        llm_timeout = float(cfg["llm_timeout"] or 25)
        # The game aborts the run if a decision exceeds its per-decision wait
        # (set above). Keep 5s headroom for game-side processing.
        llm_cap = max(10.0, agent_timeout - 5.0)
        if llm_timeout > llm_cap:
            self._log(
                "info",
                f"LLM 超时 {llm_timeout:.0f}s 超过游戏端 {agent_timeout}s 窗口的安全余量，"
                f"已钳制到 {llm_cap:.0f}s。",
            )
            llm_timeout = llm_cap
        # Retries multiply the wall time beyond the game window: default off.
        try:
            llm_retries = max(0, int(cfg.get("llm_retries", 0)))
        except (TypeError, ValueError):
            llm_retries = 0
        raw_dump = None
        if cfg.get("dump_raw_responses"):
            try:
                from pathlib import Path

                log_dir = Path(__file__).resolve().parent / "logs"
                log_dir.mkdir(exist_ok=True)
                raw_dump = str(
                    log_dir / f"llm_raw_{time.strftime('%Y%m%d')}.jsonl")
            except Exception:
                raw_dump = None
        llm = LLMClient(
            base_url=cfg["api_base_url"],
            api_key=cfg["api_key"],
            model=cfg["model"],
            temperature=float(cfg["temperature"]),
            max_tokens=int(cfg["max_tokens"]),
            timeout=llm_timeout,
            max_retries=llm_retries,
            raw_dump_path=raw_dump,
        )
        # Keep a handle so status() can report model / token usage.
        self._llm = llm
        system_prompt = render_template(
            cfg["system_template"],
            {"RULEBOOK": RULEBOOK, "CONTRACT": CONTRACT},
        )
        # Hard wall-clock budget for ONE DECISION, shared by all retry
        # attempts of the same state (see _handle_state).
        #
        #   decision_budget = agent_timeout - safety_margin
        #
        # agent_timeout is the game's whole per-decision window; llm_timeout
        # only caps a SINGLE API call. All decision_attempts must fit inside
        # the decision budget, and a send margin is reserved at the end so
        # the action still reaches the bridge in time.
        hard_deadline = max(10.0, agent_timeout - 4.0)

        try:
            while not self._stop.is_set():
                try:
                    state = self._client.receive_state()
                except (ConnectionError, TimeoutError, OSError) as e:
                    self._fail(f"Bridge connection lost: {e}")
                    return
                try:
                    self._handle_state(
                        state, llm, system_prompt, hard_deadline,
                        llm_call_cap=llm_timeout + 1.0,
                    )
                except Exception as e:
                    # Never let the worker thread die silently.
                    import traceback

                    logger.exception("Unhandled error in agent loop")
                    self._fail(
                        f"Agent loop error: {e}\n{traceback.format_exc(limit=3)}"
                    )
                    self._stop.set()
                    return
                if str(state.get("type")) in TERMINAL_SCREEN_TYPES:
                    self._log("info", "Run finished; agent stopping.")
                    # AUTO-RESUME: the run's save is kept, so wait for the
                    # game to come back instead of tearing the session down.
                    # The mod resumes the saved run on its own (it clicks
                    # "Continue" on the main menu).
                    if not self._maybe_resume():
                        break
                    continue
        finally:
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None
            self._running = False
            self._bridge_connected = False
            try:
                self._client.disconnect()
            except Exception:
                pass

    def _maybe_resume(self) -> bool:
        """After a terminated run, wait for the game and resume the save.

        An aborted run keeps its save: the next time the game reaches the main
        menu the mod clicks "Continue" and picks the run back up by itself. So
        instead of tearing the session down (and losing all context), wait for
        the bridge to come back and keep playing.

        Returns True when the bridge was re-established (caller should continue
        its loop), False to stop the agent.
        """
        if not self._config.get("auto_resume", True):
            return False

        self._log(
            "info",
            "存档已保留：下次进入主菜单时 mod 会自动点“继续”接着上次进度。"
            "正在等待游戏重连（可随时点“停止”中断）…",
        )
        try:
            wait_s = float(self._config.get("resume_wait_seconds", 600) or 600)
        except (TypeError, ValueError):
            wait_s = 600.0
        deadline = time.monotonic() + max(30.0, wait_s)

        self._bridge_connected = False
        while not self._stop.is_set() and time.monotonic() < deadline:
            try:
                if self._client is not None:
                    try:
                        self._client.disconnect()
                    except Exception:
                        pass
                self._client = STS2GameClient(
                    host=self._config["bridge_host"],
                    port=int(self._config["bridge_port"]),
                    reconnect_attempts=1,
                )
                self._client.connect(should_abort=self._stop.is_set)
            except Exception:
                # Game not back yet -- retry until the deadline or a stop.
                if self._stop.wait(3.0):
                    break
                continue

            self._bridge_connected = True
            self._log("info", "已重新连接游戏桥接，继续上次存档。")
            # Re-apply the settings the resumed run needs.
            try:
                if self._config.get("disable_fallback", True):
                    self._client.set_fallback(False)
            except Exception:
                pass
            try:
                agent_timeout = max(
                    10, min(300, int(self._config.get("agent_timeout") or 90)))
                self._client.set_agent_timeout(agent_timeout)
            except Exception:
                pass
            return True

        if self._stop.is_set():
            return False
        self._log(
            "error",
            f"等待游戏重连超时（{wait_s:.0f}s）。Agent 停止；存档仍在，"
            "重启游戏后会自动继续。",
        )
        return False

    def _fail(self, message: str) -> None:
        self._last_error = message
        self._log("error", message)
        self._running = False
        self._bridge_connected = False

    # ---------------- decision handling ----------------

    def _chat_with_deadline(
        self, llm: LLMClient, messages: list[dict[str, str]], deadline: float
    ) -> str:
        """Run llm.chat under a HARD wall-clock deadline.

        urlopen's timeout only bounds individual socket operations: a slowly
        streaming response can exceed it (observed 32.5s with timeout=25),
        which blows the game's 30s decision window and aborts the run. So run
        the call in a worker thread and abandon it at the deadline, falling
        back to the safe action instead.
        """
        out: queue.Queue = queue.Queue()

        def worker() -> None:
            try:
                out.put(llm.chat(messages))
            except BaseException as e:  # propagate to the main thread
                out.put(e)

        threading.Thread(target=worker, name="llm-call", daemon=True).start()
        try:
            result = out.get(timeout=deadline)
        except queue.Empty:
            raise LLMError(
                f"LLM 硬性时限 {deadline:.0f}s 超时（响应过慢/流式拖长）"
            )
        if isinstance(result, BaseException):
            raise result
        return result

    @staticmethod
    def _has_thought(
        raw_reply: str, parsed: dict[str, Any] | None, llm: LLMClient
    ) -> bool:
        """Whether the reply carries the contract's 'thought' field (or an
        equivalent reasoning carrier such as provider reasoning_content)."""
        if parsed and str(parsed.get("thought", "")).strip():
            return True
        if (getattr(llm, "last_reasoning", "") or "").strip():
            return True
        if AgentSession._pre_json_text(raw_reply):
            return True
        if parsed:
            for key in ("reasoning", "rationale", "explanation",
                        "reason", "message", "response", "content"):
                v = parsed.get(key)
                if isinstance(v, str) and v.strip():
                    return True
        return False

    def _fallback_action(self, state: dict[str, Any]) -> dict[str, Any]:
        """Fallback decision that remembers what it already tried.

        Two cases a naive "play the leftmost card" gets wrong:
        - a card reported playable but refused by the game: the next fallback
          sees the SAME hand and resumes AFTER that card instead of retrying
          it forever;
        - drawing new cards (cantrip, potion, draw effect): the hand
          signature changes, so the scan restarts at the leftmost card and
          the freshly drawn cards are taken into account.
        """
        if str(state.get("type", "")) != BridgeStateType.COMBAT_ACTION:
            self._fallback_attempt = None
            return fallback_action(state)

        hand = state.get("hand") or []
        sig = tuple(
            (str(c.get("id", "")), bool(c.get("playable")))
            for c in hand if isinstance(c, dict)
        )
        start = 0
        prev = self._fallback_attempt
        if prev is not None and prev[0] == sig:
            start = prev[1] + 1   # the previous pick did not resolve

        act = fallback_action(state, start_index=start)
        if act.get("action") == BridgeAction.PLAY:
            self._fallback_attempt = (sig, int(act["card_index"]))
        else:
            self._fallback_attempt = None
        return act

    def _handle_state(
        self,
        state: dict[str, Any],
        llm: LLMClient,
        system_prompt: str,
        hard_deadline: float,
        llm_call_cap: float,
    ) -> None:
        stype = str(state.get("type", "unknown"))
        self._current_state_type = stype
        self._memory.observe(state)
        state_text = format_state(state, self._memory)

        if stype in TERMINAL_SCREEN_TYPES:
            self._log("info", state_text)
            return

        if is_unsupported_state(state):
            logger.warning("Unsupported state type reached the LLM: %s", stype)
            self._log(
                "error",
                f"WARNING: no dedicated formatter for state type {stype!r}; "
                "generic fallback formatting was used. Add a proper handler.",
            )
            # Raw payload: developer/debug visibility ONLY. The LLM prompt
            # never contains it (format_unknown_state uses a whitelist).
            try:
                self._log(
                    "debug",
                    f"RAW UNKNOWN STATE ({stype}): "
                    + json.dumps(state, ensure_ascii=False, default=str)[:4000],
                )
            except Exception:
                pass

        self._log(
            "state",
            f"收到状态: {stype} (Floor {state.get('floor', '?')})，请求 LLM 决策中...",
        )

        user_template = self._config["user_template"]
        show_thinking = str(self._config.get("show_thinking", "brief"))
        try:
            attempts = max(1, int(self._config.get("decision_attempts", 3)))
        except (TypeError, ValueError):
            attempts = 3
        # One shared wall-clock deadline for ALL decision attempts of this
        # state. Retries use whatever time is left; they can never push the
        # total decision past the game's window.
        decision_deadline = time.monotonic() + hard_deadline
        feedback = ""
        prev_action_json = ""
        act: dict[str, Any] | None = None
        raw_reply = ""
        parsed: dict[str, Any] | None = None
        llm_ms: int | None = None

        for attempt in range(1, attempts + 1):
            remaining = decision_deadline - time.monotonic()
            # Keep a small margin for parsing/validating and sending the
            # action back to the bridge; never start an API call that would
            # eat the whole remaining window.
            send_margin = 1.5 if attempt < attempts else 0.5
            if remaining - send_margin < 3.0:
                self._log(
                    "error",
                    f"决策剩余时间不足（{remaining:.1f}s，需保留 {send_margin:.1f}s"
                    " 发送余量），停止重试以避免游戏端超时。",
                )
                break
            messages = self._ctx.build_messages(
                system_prompt, self._memory.to_text(), state_text
            )
            if feedback:
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your previous action was invalid.\n"
                        f"Previous action: {prev_action_json or raw_reply[:200]}\n"
                        f"Reason: {feedback}\n"
                        "The current state above has NOT changed and is still"
                        " accurate. The legal response shapes are listed at the"
                        " end of the state under 'Allowed response shapes for"
                        " THIS screen'. Respond again with EXACTLY ONE valid"
                        " JSON object and nothing else."
                    ),
                })
            # Single-call cap = configured llm_timeout (a per-call limit),
            # never more than the remaining decision time.
            call_budget = min(remaining - send_margin, llm_call_cap)
            try:
                t0 = time.monotonic()
                raw_reply = self._chat_with_deadline(llm, messages, call_budget)
                llm_ms = int((time.monotonic() - t0) * 1000)
                if llm_ms > 15000:
                    self._log("info", f"LLM 响应耗时 {llm_ms / 1000:.1f}s（较慢）")
                self._llm_fail_streak = 0
            except EmptyContentError as e:
                # The model emitted ONLY thinking and no answer text
                # (typically its reasoning consumed max_tokens and the reply
                # was truncated). This is a MODEL OUTPUT problem, not an API
                # outage: warn, tell the model exactly what was missing, and
                # retry inside the same decision window. It must NOT count
                # toward the consecutive-API-failure streak -- 3 of those
                # stop the whole agent, which is what used to happen here.
                self._log(
                    "warning",
                    f"LLM 只返回了思考过程、未给出决策文本"
                    f"（finish_reason={e.finish_reason or '?'}）: {e}",
                )
                feedback = (
                    "Your previous reply contained ONLY your reasoning /"
                    " thinking and NO answer text, so no decision was"
                    " received.\n"
                    "What was missing: the final JSON object with the"
                    ' "thought" and "action" fields.\n'
                    + (
                        "Cause: your thinking was cut off by the output length"
                        " limit (finish_reason=length) before you wrote any"
                        " JSON. Fix: keep your reasoning to ONE short sentence"
                        " and output the JSON immediately after it.\n"
                        if e.finish_reason == "length"
                        else "Fix: skip the long reasoning and output EXACTLY"
                        " ONE valid JSON object and nothing else.\n"
                    )
                    + 'Reply now with e.g. {"thought":"one short tactical'
                    ' sentence","action":"end_turn"}.\n'
                    "The current state above has NOT changed and is still"
                    " accurate. The legal response shapes are listed at the"
                    " end of the state under 'Allowed response shapes for"
                    " THIS screen'."
                )
                prev_action_json = ""
                raw_reply = ""
                continue
            except LLMError as e:
                # Don't kill the run for one API failure: play the safe
                # fallback so the game gets an answer inside its window.
                self._llm_fail_streak += 1
                self._log(
                    "error",
                    f"LLM API error (连续第 {self._llm_fail_streak} 次): {e}",
                )
                if self._llm_fail_streak >= 3:
                    self._fail("LLM API 连续 3 次失败，Agent 停止。请检查 API 配置/网络。")
                    self._stop.set()
                    return
                act = self._fallback_action(state)
                result_note = self._execute(act)
                self._decision_count += 1
                self._log(
                    "decision",
                    "(LLM 请求失败，已执行安全兜底动作)",
                    action=json.dumps(act),
                    result=result_note,
                    state_type=stype,
                )
                self._ctx.add_decision(state_text, json.dumps(act), result_note)
                return
            except Exception as e:
                self._fail(f"LLM 客户端意外错误: {e}")
                self._stop.set()
                return
            try:
                parsed = extract_json(raw_reply)
            except ValueError as e:
                feedback = str(e)
                self._log("error", f"Unparseable LLM reply (attempt {attempt}): {e}")
                continue
            act, err = validate_action(state, parsed)
            if act is None:
                feedback = err
                prev_action_json = json.dumps(parsed, ensure_ascii=False)
                self._log(
                    "error",
                    f"Invalid action (attempt {attempt}): {err} | reply: {raw_reply[:200]}",
                )
                continue
            # HARD CONTRACT: a bare decision without "thought" is rejected.
            if not self._has_thought(raw_reply, parsed, llm):
                feedback = (
                    "REJECTED: your reply contained ONLY the decision. Include a"
                    ' non-empty "thought" field with one concise sentence of'
                    " tactical reasoning INSIDE the same JSON object. Reply"
                    " again with exactly one JSON object and nothing else."
                )
                prev_action_json = json.dumps(parsed, ensure_ascii=False)
                self._log(
                    "error",
                    f"Missing thought (attempt {attempt}); 驳回并要求补充推理",
                )
                continue
            break

        if act is None:
            act = self._fallback_action(state)
            # Known screens: protocol-safe deterministic fallback
            # (leave shop / skip allowed card reward / end turn).
            # UNKNOWN screens: this is an EMERGENCY deterministic fallback
            # -- its game effect is unknown; it is a last resort, not safe.
            kind = (
                "emergency deterministic fallback (unknown screen)"
                if is_unsupported_state(state)
                else "protocol-safe fallback"
            )
            self._log(
                "error",
                f"{kind}: {json.dumps(act)} after {attempts} failed"
                f" decision attempts (shared decision deadline).",
            )

        result_note = self._execute(act)
        self._decision_count += 1
        self._log(
            "decision",
            self._decision_display(show_thinking, raw_reply, parsed, llm),
            action=json.dumps(act),
            result=result_note,
            state_type=stype,
            llm_ms=llm_ms,
        )
        self._ctx.add_decision(state_text, json.dumps(act), result_note)

    @staticmethod
    def _pre_json_text(raw_reply: str) -> str:
        """Return the natural-language text the model wrote before its JSON
        object, when it ignored the pure-JSON contract (kept for display
        and thought-detection fallback only)."""
        if not raw_reply:
            return ""
        idx = raw_reply.find("{")
        if idx <= 0:
            return ""
        head = raw_reply[:idx].strip()
        head = re.sub(r"^```(?:json)?\s*", "", head).strip()
        head = re.sub(r"^Explanation\s*[:：]\s*", "", head, flags=re.I).strip()
        return head[:300]

    def _decision_display(
        self,
        show_thinking: str,
        raw_reply: str,
        parsed: dict[str, Any] | None,
        llm: LLMClient,
    ) -> str:
        """Render the decision log entry according to the thinking mode."""
        reasoning = (getattr(llm, "last_reasoning", "") or "").strip()
        if show_thinking == "full":
            parts = []
            if reasoning:
                parts.append(f"[思考] {reasoning[:600]}")
            parts.append(raw_reply.strip()[:800] if raw_reply else "(fallback)")
            return "\n".join(parts)
        if show_thinking == "brief":
            if parsed:
                thought = str(parsed.get("thought", "")).strip()
                if thought:
                    return thought
                # Model omitted "thought": surface any other explanation field
                for key in ("reasoning", "rationale", "explanation",
                            "reason", "message", "response", "content"):
                    val = parsed.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()[:300]
            # The model ignored the contract's "thought" field and wrote
            # text before the JSON instead.
            pre = self._pre_json_text(raw_reply)
            if pre:
                return pre
            # Last resort: show the raw reply itself
            return raw_reply.strip()[:300] if raw_reply else "(fallback)"
        if show_thinking == "hidden":
            return "(已隐藏思考)"
        # Unknown mode fallback
        return raw_reply.strip()[:300] if raw_reply else "(fallback)"

    def _execute(self, act: dict[str, Any]) -> str:
        assert self._client is not None
        # Optional pacing: pause before sending each action (0 = unlimited).
        try:
            delay = float(self._config.get("action_delay") or 0)
        except (TypeError, ValueError):
            delay = 0.0
        if delay > 0:
            if self._stop.wait(delay):
                return "cancelled (stopped during delay)"
        name = act.get("action")
        try:
            if name == BridgeAction.PLAY:
                self._client.play_card(act["card_index"], act.get("target_index", -1))
                return f"played card {act['card_index']} (target {act.get('target_index', -1)})"
            if name == BridgeAction.END_TURN:
                self._client.end_turn()
                return "ended turn"
            if name == BridgeAction.POTION:
                self._client.use_potion(act["slot"], act.get("target_index", -1))
                return f"used potion slot {act['slot']}"
            if name == BridgeAction.CHOOSE:
                if "indexes" in act:
                    self._client.choose_many(act["indexes"])
                    return f"chose {act['indexes']}"
                self._client.choose(act["index"])
                return f"chose {act['index']}"
            if name == BridgeAction.SKIP:
                self._client.skip()
                return "skipped"
            return f"unknown action {name!r}"
        except ConnectionError as e:
            self._fail(f"Lost bridge while sending action: {e}")
            self._stop.set()
            return f"ERROR: {e}"
