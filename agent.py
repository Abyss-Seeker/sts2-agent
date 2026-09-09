"""LLM agent loop: state -> prompt -> LLM -> validated action -> game.

The bridge protocol is synchronous: the game sends one state and waits for
one action. The agent therefore runs a simple blocking loop in a worker
thread, asking the LLM for a JSON decision for every state.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import re
import threading
import time
from collections import deque
from typing import Any

from action_plan import ActionChunk, PlanParseError, parse_action_chunk
from benchmark_metrics import BenchmarkMetrics
from bridge_client import (
    BridgeAction,
    BridgeStateType,
    CHOICE_SCREEN_TYPES,
    TERMINAL_SCREEN_TYPES,
    STS2GameClient,
)
from checkpoint import CheckpointReason
from context_manager import ContextConfig, ContextManager
from plan_executor import ActionChunkExecutor, ExecutorStatus
from game_state import (
    RunMemory,
    format_state,
    is_unsupported_state,
    _option_entries,
    skip_allowed,
)
from llm_client import EmptyContentError, LLMClient, LLMError
from prompts import (
    DEFAULT_SYSTEM_TEMPLATE,
    DEFAULT_USER_TEMPLATE,
    RULEBOOK,
    ACTION_CHUNK_CONTRACT,
    SINGLE_ACTION_CONTRACT,
)
from state_diff import diff_states, render_delta

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    "bridge_host": "127.0.0.1",
    "bridge_port": 9002,
    "api_base_url": "https://api.openai.com",
    "api_key": "",
    "model": "gpt-4o-mini",
    "temperature": 0.4,
    # Reasoning models can burn the whole budget in reasoning_content; 512
    # truncates them before any JSON is emitted. 8192 is the beta default.
    "max_tokens": 8192,
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
    # ---- Beta: LLM-native ActionChunk mode -------------------------
    # decision_mode: "single_action" (baseline, one action per call) or
    # "action_chunk" (one model plan may drive several individually
    # confirmed bridge actions; combat only in this beta).
    "decision_mode": "single_action",
    # failure_policy: "benchmark_strict" (NO non-LLM fallback: on LLM
    # failure the benchmark is invalidated and the agent stops) or
    # "demo_resilient" (deterministic fallback may play; benchmark
    # invalidated and prominently logged).
    "failure_policy": "benchmark_strict",
    "action_chunk_max_actions": 16,
    # Delta observations: OFF by default for beta correctness -- a compact
    # diff cannot carry a newly drawn card's full human-visible face (cost,
    # playable, displayed text, modifiers, hover/preview). Even when
    # enabled, HAND_CHANGED / NEXT_ACTION_ILLEGAL / CARD_GONE / TARGET_GONE
    # / POTION_INVALID checkpoints and any new hand information FORCE a
    # full state (see _checkpoint_requires_full_state).
    "delta_observations": False,
    # DeepSeek thinking controls (generic endpoints ignore them).
    "thinking_enabled": True,
    "reasoning_effort": "high",  # low | high | max
    "stream_mode": "off",        # off | auto | on
    # Provider capability profile: "auto" enables native DeepSeek fields
    # ONLY on the official api.deepseek.com hostname (never by model
    # name); "generic" always plain; "deepseek" forces native fields.
    "provider_profile": "auto",
}


def _looks_like_decision_object(obj: Any) -> bool:
    """Whether a repaired/salvaged object plausibly carries a decision.

    BOTH response contracts must be accepted here: the single-action
    contract's top-level "action" AND the ActionChunk contract's
    top-level "actions" list.
    """
    return (
        isinstance(obj, dict)
        and (
            bool(obj.get("action"))
            or isinstance(obj.get("actions"), list)
        )
    )


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
        if _looks_like_decision_object(obj):
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
    for key in ('"action"', '"actions"', '"thought"'):
        pos = cleaned.rfind(key)
        if pos > 0:
            candidate = cleaned[pos - 1 if cleaned[pos - 1] == "{" else pos:]
            if not candidate.lstrip().startswith("{"):
                candidate = "{" + candidate
            try:
                obj = json.loads(candidate)
                if _looks_like_decision_object(obj):
                    return obj
            except json.JSONDecodeError:
                pass
            try:
                from json_repair import repair_json

                obj = repair_json(candidate, return_objects=True)
                if _looks_like_decision_object(obj):
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
        # ---- Beta: LLM-native ActionChunk state --------------------
        self._plan_executor = ActionChunkExecutor()
        self._metrics = BenchmarkMetrics()
        self._agent_phase = "idle"  # idle|waiting_game|thinking|plan_ready|executing_plan|checkpoint|failed|terminal
        self._live_reasoning = ""
        self._live_content = ""
        self._current_plan_id = ""
        self._current_plan_step = 0
        self._current_plan_total = 0
        self._current_plan_thought = ""
        self._plan_executed: list[str] = []
        self._plan_state_text = ""
        self._last_checkpoint_reason = ""
        self._last_model_observation_state: dict[str, Any] | None = None
        self._last_model_observation_text = ""
        self._last_plan_executed: list[str] = []
        self._last_model_failure_reason = ""
        # Single-action pending confirmation: the sent action is only
        # CONFIRMED/REJECTED by the NEXT authoritative state -- identical
        # semantics to ActionChunk reconciliation.
        self._pending_single_action: dict[str, Any] | None = None
        self._pending_single_before_state: dict[str, Any] | None = None

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
            # Normalize beta knobs to their allowed values.
            if self._config.get("decision_mode") not in ("single_action", "action_chunk"):
                self._config["decision_mode"] = "single_action"
            if self._config.get("failure_policy") not in ("benchmark_strict", "demo_resilient"):
                self._config["failure_policy"] = "benchmark_strict"
            if self._config.get("stream_mode") not in ("off", "auto", "on"):
                self._config["stream_mode"] = "off"
            if self._config.get("reasoning_effort") not in ("low", "high", "max"):
                self._config["reasoning_effort"] = "high"
            if self._config.get("provider_profile") not in ("auto", "generic", "deepseek"):
                self._config["provider_profile"] = "auto"
            self._stop.clear()
            self._running = True
            self._last_error = ""
            self._decision_count = 0
            self._llm_fail_streak = 0
            # Reset beta ActionChunk state for the new session.
            self._plan_executor.reset()
            self._metrics = BenchmarkMetrics()
            self._agent_phase = "idle"
            self._live_reasoning = ""
            self._live_content = ""
            self._current_plan_id = ""
            self._current_plan_step = 0
            self._current_plan_total = 0
            self._current_plan_thought = ""
            self._plan_executed = []
            self._plan_state_text = ""
            self._last_checkpoint_reason = ""
            self._last_model_observation_state = None
            self._last_model_observation_text = ""
            self._last_plan_executed = []
            self._last_model_failure_reason = ""
            self._pending_single_action = None
            self._pending_single_before_state = None
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
                # ---- Beta: ActionChunk runtime status/metrics ----
                "decision_mode": str(self._config.get("decision_mode", "single_action")),
                "agent_phase": self._agent_phase,
                "benchmark_valid": self._metrics.benchmark_valid,
                "invalidation_reason": self._metrics.invalidation_reason,
                # model_call_count is a compat alias: EVERY inference
                # request (successes AND failures/retries) counts.
                "model_call_count": self._metrics.llm_request_count,
                "llm_request_count": self._metrics.llm_request_count,
                "llm_success_count": self._metrics.llm_success_count,
                "llm_failed_request_count": self._metrics.llm_failed_request_count,
                "logical_inspection_count": self._metrics.logical_inspection_count,
                "model_inspection_count": self._metrics.model_inspection_count,
                "game_action_count": self._metrics.game_action_count,  # CONFIRMED
                "game_action_sent_count": self._metrics.game_action_sent_count,
                "game_action_confirmed_count": self._metrics.game_action_confirmed_count,
                "game_action_rejected_count": self._metrics.game_action_rejected_count,
                "game_action_unconfirmable_count": self._metrics.game_action_unconfirmable_count,
                "planned_actions_total": self._metrics.planned_actions_total,
                "executed_planned_actions_total": self._metrics.executed_planned_actions_total,
                "executed_vs_planned_ratio": (
                    self._metrics.executed_planned_actions_total
                    / self._metrics.planned_actions_total
                    if self._metrics.planned_actions_total else 0.0
                ),
                "checkpoint_count": self._metrics.checkpoint_count,
                "plan_completed_count": self._metrics.plan_completed_count,
                "plan_interrupted_count": self._metrics.plan_interrupted_count,
                "invalid_plan_count": self._metrics.invalid_plan_count,
                "fallback_action_count": self._metrics.fallback_action_count,
                "actions_per_llm_call": (
                    self._metrics.game_action_confirmed_count
                    / self._metrics.llm_request_count
                    if self._metrics.llm_request_count else 0.0
                ),
                "actions_per_logical_inspection": (
                    self._metrics.game_action_confirmed_count
                    / self._metrics.logical_inspection_count
                    if self._metrics.logical_inspection_count else 0.0
                ),
                # Real provider token/cache/reasoning usage (FIX 8).
                "prompt_tokens": self._metrics.prompt_tokens,
                "completion_tokens": self._metrics.completion_tokens,
                "reasoning_tokens": self._metrics.reasoning_tokens,
                "prompt_cache_hit_tokens": self._metrics.prompt_cache_hit_tokens,
                "prompt_cache_miss_tokens": self._metrics.prompt_cache_miss_tokens,
                "cache_hit_ratio": (
                    self._metrics.prompt_cache_hit_tokens
                    / (
                        self._metrics.prompt_cache_hit_tokens
                        + self._metrics.prompt_cache_miss_tokens
                    )
                    if (
                        self._metrics.prompt_cache_hit_tokens
                        + self._metrics.prompt_cache_miss_tokens
                    ) else None
                ),
                "llm_latency_ms_p50": self._metrics._percentile(
                    self._metrics.llm_latencies_ms, 0.50),
                "first_reasoning_token_ms_p50": self._metrics._percentile(
                    self._metrics.first_reasoning_token_ms, 0.50),
                "first_content_token_ms_p50": self._metrics._percentile(
                    self._metrics.first_content_token_ms, 0.50),
                "current_plan_id": self._current_plan_id,
                "current_plan_step": self._current_plan_step,
                "current_plan_total": self._current_plan_total,
                "last_checkpoint_reason": self._last_checkpoint_reason,
                "live_reasoning": self._live_reasoning[-400:],
                "live_content": self._live_content[-200:],
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
            thinking_enabled=bool(cfg.get("thinking_enabled", True)),
            reasoning_effort=str(cfg.get("reasoning_effort", "high")),
            stream_mode=str(cfg.get("stream_mode", "off")),
            provider_profile=str(cfg.get("provider_profile", "auto")),
        )
        # Live streaming display hooks (no-ops for nonstreaming calls).
        llm.on_reasoning_delta = lambda d: self._set_live("reasoning", d)
        llm.on_content_delta = lambda d: self._set_live("content", d)
        # llm_request_count counts EVERY real HTTP inference attempt
        # (including internal max_retries retries), fired by the client
        # right before each urlopen -- streaming and non-streaming alike.
        llm.on_http_attempt = lambda _attempt: self._metrics.record_llm_request()
        # Keep a handle so status() can report model / token usage.
        self._llm = llm
        # Two system prompts: never mix the single-action contract and the
        # ActionChunk contract in one model call.
        single_system_prompt = render_template(
            cfg["system_template"],
            {"RULEBOOK": RULEBOOK, "CONTRACT": SINGLE_ACTION_CONTRACT},
        )
        chunk_system_prompt = render_template(
            cfg["system_template"],
            {"RULEBOOK": RULEBOOK, "CONTRACT": ACTION_CHUNK_CONTRACT},
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

                # Step A: ALWAYS reconcile previously-sent actions FIRST,
                # whatever screen the new state shows -- confirmation
                # semantics are identical in both modes:
                #   SENT -> next authoritative state -> CONFIRMED / REJECTED.
                if cfg.get("decision_mode") == "action_chunk":
                    reconcile_event = None
                    if self._plan_executor.inflight is not None:
                        # The sent action must be confirmed/rejected,
                        # checkpoint-logged and the plan finalized BEFORE
                        # routing decides anything.
                        reconcile_event = self._reconcile_inflight_if_any(state)
                else:
                    # single_action mode: the sent action's confirmation
                    # also comes from the next authoritative state.
                    self._reconcile_pending_single_action(state)
                    reconcile_event = None

                # Step B: route the newly observed screen.
                stype = str(state.get("type", "unknown"))
                if stype in TERMINAL_SCREEN_TYPES:
                    self._plan_executor.reset()
                    self._agent_phase = "terminal"
                    self._log("info", "Run finished; agent stopping.")
                    # AUTO-RESUME: the run's save is kept, so wait for the
                    # game to come back instead of tearing the session down.
                    # The mod resumes the saved run on its own (it clicks
                    # "Continue" on the main menu).
                    if not self._maybe_resume():
                        break
                    continue
                try:
                    if (
                        cfg.get("decision_mode") == "action_chunk"
                        and stype == BridgeStateType.COMBAT_ACTION
                    ):
                        # Beta: combat turns may run a pending ActionChunk
                        # across multiple bridge handshakes. A possible
                        # in-flight action was ALREADY reconciled in Step A
                        # -- never accept_state() the same state twice.
                        self._handle_combat_chunk_state(
                            state, llm, chunk_system_prompt, hard_deadline,
                            llm_call_cap=llm_timeout + 1.0,
                            reconcile_event=reconcile_event,
                        )
                    else:
                        # Noncombat screens and single_action mode keep the
                        # original one-choice path. Any old combat plan was
                        # already reconciled above (checkpoint-logged, plan
                        # finalized); it must not survive a screen change.
                        self._plan_executor.reset()
                        self._handle_state(
                            state, llm, single_system_prompt, hard_deadline,
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

    # ---------------- beta: LLM-native ActionChunk ----------------
    #
    # Core invariant: the bridge still handshakes after EVERY action and
    # the game state is machine-observed after every action, but the LLM
    # is re-invoked only at cognitive boundaries. The harness decides WHEN
    # the model must look again -- never WHAT the model should do next.

    # Checkpoint reasons that stay inside the current combat frame; these
    # may use a compact objective delta instead of a full re-observation.
    _INTRA_ROUND_CHECKPOINTS = {
        CheckpointReason.HAND_CHANGED,
        CheckpointReason.TARGET_GONE,
        CheckpointReason.CARD_GONE,
        CheckpointReason.NEXT_ACTION_ILLEGAL,
        CheckpointReason.ACTION_REJECTED,
        CheckpointReason.MODEL_REQUESTED,
        CheckpointReason.POTION_INVALID,
    }

    def _set_live(self, which: str, delta: str) -> None:
        with self._lock:
            if which == "reasoning":
                self._live_reasoning += delta
                if len(self._live_reasoning) > 8000:
                    self._live_reasoning = self._live_reasoning[-4000:]
            else:
                self._live_content += delta
                if len(self._live_content) > 4000:
                    self._live_content = self._live_content[-2000:]

    def _checkpoint_requires_full_state(
        self,
        previous_model_state: dict[str, Any] | None,
        current_state: dict[str, Any],
        checkpoint_reason: str,
    ) -> bool:
        """Whether a checkpoint re-prompt must send the FULL state.

        Correctness first: any reason that involves unresolved/new card or
        target identity, or any newly visible hand information, must give
        the model the complete human-visible card faces -- a compact delta
        cannot express cost/playable/displayed text/modifiers of NEW cards.
        """
        if previous_model_state is None:
            return True
        if checkpoint_reason in {
            "HAND_CHANGED",
            "NEXT_ACTION_ILLEGAL",
            "CARD_GONE",
            "TARGET_GONE",
            "POTION_INVALID",
        }:
            return True
        delta = diff_states(previous_model_state, current_state)
        if delta.hand_added_information:
            return True
        return False

    def _model_observation_text(self, state: dict[str, Any]) -> str:
        """FULL or compact DELTA model observation (objective diffs only).

        Falls back to the full formatted state whenever anything is
        uncertain -- a full state is always safe.
        """
        prev = self._last_model_observation_state
        reason = self._last_checkpoint_reason
        if (
            self._config.get("delta_observations", False)
            and prev is not None
            and reason
            and str(prev.get("type")) == str(state.get("type"))
            and reason in {r.value for r in self._INTRA_ROUND_CHECKPOINTS}
            and not self._checkpoint_requires_full_state(prev, state, reason)
        ):
            executed = " -> ".join(self._last_plan_executed[-4:]) or "(none)"
            text = render_delta(
                prev, state, executed=executed, checkpoint_reason=reason
            )
            # The model answers with plan-scoped refs built from the
            # CURRENT hand; give it the objective ref mapping.
            try:
                from action_plan import compact_ref_legend

                text += "\n\n" + compact_ref_legend(state)
            except ImportError:
                pass
            return text
        return format_state(state, self._memory, response_mode="action_chunk")

    def _reconcile_inflight_if_any(self, state: dict[str, Any]) -> Any:
        """Step A of the main loop: reconcile the authoritative result of
        the previously sent plan action, whatever screen the new state
        shows. Returns the ExecutorEvent (never None when called).

        Action accounting: the sent action is either CONFIRMED by the
        authoritative next state (the visible world moved forward --
        screen changes / new turns / terminal all count) or REJECTED
        (action-relevantly unchanged state). A connection lost before the
        next state is never counted as confirmed.
        """
        executor = self._plan_executor
        event = executor.accept_state(state)
        if (
            event.checkpoint is not None
            and event.checkpoint.reason is CheckpointReason.ACTION_REJECTED
        ):
            self._metrics.record_action_rejected()
        else:
            self._metrics.record_action_confirmed(from_plan=True)
        self._consume_executor_event(event, state)
        return event

    def _handle_combat_chunk_state(
        self,
        state: dict[str, Any],
        llm: LLMClient,
        chunk_system_prompt: str,
        hard_deadline: float,
        llm_call_cap: float,
        reconcile_event: Any = None,
    ) -> None:
        stype = str(state.get("type", "unknown"))
        self._current_state_type = stype
        self._memory.observe(state)
        executor = self._plan_executor

        # The main loop already reconciled a possibly in-flight action in
        # Step A (reconcile_event). Never accept_state() the same state
        # twice -- only fall back to a local reconcile defensively.
        event = reconcile_event
        if event is None and executor.inflight is not None:
            event = executor.accept_state(state)
            if (
                event.checkpoint is not None
                and event.checkpoint.reason is CheckpointReason.ACTION_REJECTED
            ):
                self._metrics.record_action_rejected()
            else:
                self._metrics.record_action_confirmed()
            self._consume_executor_event(event, state)

        if event is not None:
            if event.status == ExecutorStatus.TERMINAL:
                return
            if event.status == ExecutorStatus.READY_ACTION:
                # Plan remains epistemically valid: prepare the next
                # committed action against the CURRENT state. No LLM call.
                next_event = executor.prepare_next(state, validate_action)
                if next_event.status == ExecutorStatus.READY_ACTION:
                    self._send_prepared_plan_step(next_event.prepared, state)
                    return
                self._consume_executor_event(next_event, state)
            # NEED_MODEL falls through to the model call below.
        elif executor.has_pending_plan:
            # Defensive: a plan exists but no action is in flight.
            next_event = executor.prepare_next(state, validate_action)
            if next_event.status == ExecutorStatus.READY_ACTION:
                self._send_prepared_plan_step(next_event.prepared, state)
                return
            self._consume_executor_event(next_event, state)

        # No valid pending plan action remains: ask the model for a chunk.
        # Never call the LLM before trying to continue a valid pending plan
        # -- that is the entire performance win.
        self._agent_phase = "thinking"
        # One cognitive boundary per state that needs the model.
        self._metrics.record_inspection()
        decision_deadline = time.monotonic() + hard_deadline
        try:
            attempts = max(1, int(self._config.get("decision_attempts", 3)))
        except (TypeError, ValueError):
            attempts = 3
        feedback = ""
        for _attempt in range(attempts):
            remaining = decision_deadline - time.monotonic()
            if remaining - 0.5 < 3.0:
                self._log(
                    "error",
                    f"决策剩余时间不足（{remaining:.1f}s），停止重试以避免游戏端超时。",
                )
                break
            chunk = self._request_combat_chunk(
                state, llm, chunk_system_prompt,
                decision_deadline=decision_deadline,
                llm_call_cap=llm_call_cap,
                feedback=feedback,
            )
            if chunk is None:
                self._handle_model_failure(
                    state,
                    self._last_model_failure_reason
                    or "combat ActionChunk request failed",
                )
                return
            executor.submit(chunk)
            self._current_plan_id = chunk.plan_id
            self._current_plan_total = len(chunk.actions)
            self._current_plan_step = 0
            self._current_plan_thought = chunk.summary
            self._plan_executed = []
            event = executor.prepare_next(state, validate_action)
            if event.status == ExecutorStatus.READY_ACTION:
                self._send_prepared_plan_step(event.prepared, state)
                return
            # The chunk could not yield even one executable action: retry
            # with feedback while decision time remains (never a strategic
            # fallback in strict mode).
            self._metrics.record_invalid_plan()
            self._consume_executor_event(event, state)
            feedback = (
                "Your chunk was rejected before its first action could be sent: "
                + (
                    event.checkpoint.detail
                    if event.checkpoint is not None and event.checkpoint.detail
                    else "the first action was invalid"
                )
                + "\nThe current state has NOT changed and the plan-scoped refs"
                " are unchanged. Respond again with EXACTLY ONE valid"
                " ActionChunk JSON object."
            )
        self._handle_model_failure(
            state, "no valid ActionChunk within the decision deadline"
        )

    def _request_combat_chunk(
        self,
        state: dict[str, Any],
        llm: LLMClient,
        chunk_system_prompt: str,
        decision_deadline: float,
        llm_call_cap: float,
        feedback: str = "",
    ) -> ActionChunk | None:
        """Request ONE ActionChunk from the model.

        Formats the observation, retries parse/first-step validation
        inside the shared decision deadline, and records metrics. It does
        NOT execute anything -- the executor does.
        """
        observation_text = self._model_observation_text(state)
        try:
            attempts = max(1, int(self._config.get("decision_attempts", 3)))
        except (TypeError, ValueError):
            attempts = 3
        attempt_feedback = feedback
        prev_chunk_json = ""
        llm_ms: int | None = None
        for attempt in range(1, attempts + 1):
            remaining = decision_deadline - time.monotonic()
            send_margin = 1.5 if attempt < attempts else 0.5
            if remaining - send_margin < 3.0:
                self._last_model_failure_reason = (
                    f"decision time exhausted ({remaining:.1f}s left)"
                )
                self._log(
                    "error",
                    f"决策剩余时间不足（{remaining:.1f}s，需保留 {send_margin:.1f}s"
                    " 发送余量），停止重试以避免游戏端超时。",
                )
                return None
            messages = self._ctx.build_messages(
                chunk_system_prompt, self._memory.to_text(), observation_text
            )
            if attempt_feedback:
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous ActionChunk was invalid.\n"
                        f"Previous response: {prev_chunk_json or attempt_feedback[:200]}\n"
                        f"Reason: {attempt_feedback}\n"
                        "The current state above has NOT changed and the"
                        " plan-scoped refs are unchanged. Respond again with"
                        " EXACTLY ONE valid ActionChunk JSON object and"
                        " nothing else."
                    ),
                })
            call_budget = min(remaining - send_margin, llm_call_cap)
            # (llm_request_count is recorded by llm.on_http_attempt --
            # one count per REAL HTTP attempt, retries included.)
            try:
                t0 = time.monotonic()
                raw_reply = self._chat_with_deadline(llm, messages, call_budget)
                llm_ms = int((time.monotonic() - t0) * 1000)
                if llm_ms > 15000:
                    self._log("info", f"LLM 响应耗时 {llm_ms / 1000:.1f}s（较慢）")
                self._llm_fail_streak = 0
            except EmptyContentError as e:
                self._metrics.record_llm_failure()
                # Thinking-only reply: warn + retry with a precise hint.
                self._log(
                    "warning",
                    f"LLM 只返回了思考过程、未给出 ActionChunk"
                    f"（finish_reason={e.finish_reason or '?'}）: {e}",
                )
                attempt_feedback = (
                    "Your previous reply contained ONLY your reasoning and NO"
                    " answer text, so no ActionChunk was received.\n"
                    "What was missing: the final JSON object with the"
                    ' "thought" and "actions" fields.\n'
                    + (
                        "Cause: your thinking was cut off by the output length"
                        " limit (finish_reason=length). Fix: keep reasoning to"
                        " ONE short sentence and output the JSON immediately.\n"
                        if e.finish_reason == "length"
                        else "Fix: skip the long reasoning and output EXACTLY"
                        " ONE valid JSON object and nothing else.\n"
                    )
                    + 'Reply now with e.g. {"thought":"one short tactical'
                    ' sentence","actions":[{"kind":"end_turn"}]}.\n'
                    "The current state has NOT changed and the plan-scoped"
                    " refs are unchanged."
                )
                prev_chunk_json = ""
                continue
            except LLMError as e:
                self._metrics.record_llm_failure()
                self._last_model_failure_reason = f"LLM API error: {e}"
                self._log("error", f"LLM API error: {e}")
                return None
            except Exception as e:
                self._metrics.record_llm_failure()
                self._last_model_failure_reason = f"unexpected LLM client error: {e}"
                self._log("error", f"LLM 客户端意外错误: {e}")
                return None
            self._metrics.record_llm_success(
                latency_ms=llm_ms,
                first_reasoning_ms=getattr(llm, "first_reasoning_ms", None),
                first_content_ms=getattr(llm, "first_content_ms", None),
                usage=getattr(llm, "last_usage", None) or None,
            )
            try:
                parsed = extract_json(raw_reply)
            except ValueError as e:
                attempt_feedback = str(e)
                self._log(
                    "error",
                    f"Unparseable ActionChunk reply (attempt {attempt}): {e}",
                )
                continue
            try:
                chunk = parse_action_chunk(
                    parsed, state,
                    max_actions=max(
                        1, int(self._config.get("action_chunk_max_actions", 16))
                    ),
                )
            except PlanParseError as e:
                attempt_feedback = str(e)
                prev_chunk_json = json.dumps(parsed, ensure_ascii=False)
                self._log(
                    "error",
                    f"Invalid ActionChunk (attempt {attempt}): {e}",
                )
                continue
            # Verify the FIRST action resolves + validates against the
            # CURRENT authoritative state (no send) using a throwaway
            # executor; the real executor re-checks before every send.
            probe = ActionChunkExecutor()
            probe.submit(chunk)
            probe_event = probe.prepare_next(state, validate_action)
            if probe_event.status != ExecutorStatus.READY_ACTION:
                detail = (
                    probe_event.checkpoint.detail
                    if probe_event.checkpoint is not None
                    else ""
                )
                attempt_feedback = (
                    "your chunk's FIRST action is not executable right now"
                    + (f": {detail}" if detail else "")
                    + ". Use only currently playable cards and living targets."
                )
                prev_chunk_json = json.dumps(parsed, ensure_ascii=False)
                self._metrics.record_invalid_plan()
                self._log(
                    "error",
                    f"ActionChunk first step invalid (attempt {attempt}):"
                    f" {detail}",
                )
                continue
            # Success bookkeeping.
            self._metrics.record_plan(len(chunk.actions))
            self._last_model_observation_state = state
            self._last_model_observation_text = observation_text
            self._plan_state_text = observation_text
            actions_repr = [
                {
                    "kind": a.kind.value,
                    **({"card_ref": a.card_ref} if a.card_ref else {}),
                    **({"target_ref": a.target_ref} if a.target_ref else {}),
                    **(
                        {"potion_slot": a.potion_slot}
                        if a.potion_slot is not None
                        else {}
                    ),
                    **({"checkpoint_after": True} if a.checkpoint_after else {}),
                }
                for a in chunk.actions
            ]
            self._log(
                "model_plan",
                chunk.summary,
                plan_id=chunk.plan_id,
                actions=json.dumps(actions_repr, ensure_ascii=False),
                llm_ms=llm_ms,
            )
            self._agent_phase = "plan_ready"
            return chunk
        self._last_model_failure_reason = (
            self._last_model_failure_reason or "all ActionChunk attempts failed"
        )
        return None

    def _send_prepared_plan_step(
        self, prepared: Any, state: dict[str, Any]
    ) -> None:
        """Mark + send ONE plan step. This only proves ACTION SENT; the
        CONFIRMED accounting happens in _reconcile_inflight_if_any() when
        the authoritative next state arrives."""
        self._plan_executor.mark_sent(prepared, state)
        result = self._execute(prepared.bridge_action)
        self._metrics.record_action_sent(from_plan=True)
        self._current_plan_step = prepared.step_index + 1
        self._plan_executed.append(prepared.description)
        self._agent_phase = "executing_plan"
        self._log(
            "game_action",
            prepared.description,
            plan_id=prepared.plan_id,
            step_index=prepared.step_index,
            action=json.dumps(prepared.bridge_action),
            result=result,
            source="llm_action_chunk",
        )

    def _consume_executor_event(
        self,
        event: Any,
        state: dict[str, Any],
    ) -> None:
        """Checkpoint logging/metrics/context for executor events."""
        if event.status in (
            ExecutorStatus.READY_ACTION,
            ExecutorStatus.WAITING_RESULT,
        ):
            return
        cp = event.checkpoint
        if cp is None or cp.reason is CheckpointReason.NONE:
            return
        reason = cp.reason
        # Plan completion and checkpoint reason are TWO DIFFERENT
        # dimensions (review fix): a plan whose last action moved the game
        # to a new turn / new screen / terminal is COMPLETED even though
        # the checkpoint reason is NEW_TURN / SCREEN_CHANGED / TERMINAL.
        # Only the ExecutorEvent's plan_completed flag knows whether the
        # chunk was exhausted when the checkpoint fired.
        completed = bool(getattr(event, "plan_completed", False))
        executed = list(self._plan_executed)
        self._last_plan_executed = executed
        self._last_checkpoint_reason = reason.value
        if completed:
            self._metrics.record_plan_complete()
        self._metrics.record_checkpoint(reason.value, interrupted=not completed)
        remaining = max(0, self._current_plan_total - len(executed))
        self._log(
            "plan_checkpoint",
            cp.detail or reason.value,
            reason=reason.value,
            plan_id=event.completed_plan_id or self._current_plan_id,
            executed_steps=len(executed),
            remaining_steps_discarded=0 if completed else remaining,
            plan_completed=completed,
        )
        self._add_plan_level_context(
            event.completed_plan_id or self._current_plan_id,
            executed,
            reason,
            cp.detail,
            completed,
        )
        # Reset per-plan bookkeeping (a fresh chunk sets it again).
        self._plan_executed = []
        self._current_plan_id = ""
        self._current_plan_step = 0
        self._current_plan_total = 0
        self._current_plan_thought = ""
        self._agent_phase = "checkpoint"

    def _add_plan_level_context(
        self,
        plan_id: str,
        executed: list[str],
        reason: CheckpointReason,
        detail: str,
        completed: bool,
    ) -> None:
        """Store ONE plan-level history turn per LLM plan -- never one turn
        per mechanically executed card."""
        if not self._plan_state_text:
            return
        lines = [f"PLAN {plan_id}"]
        if self._current_plan_thought:
            lines.append(f"MODEL PLAN: {self._current_plan_thought}")
        lines.append("EXECUTED: " + ("; ".join(executed) if executed else "(none)"))
        if completed:
            lines.append("RESULT: Plan completed.")
        else:
            lines.append(
                f"RESULT: Interrupted -- {reason.value}"
                + (f" ({detail})" if detail else "")
            )
        response = json.dumps({
            "plan_id": plan_id,
            "thought": self._current_plan_thought,
            "executed": executed,
            "result": (
                "completed" if completed else f"interrupted:{reason.value}"
            ),
        }, ensure_ascii=False)
        self._ctx.add_decision(
            self._plan_state_text, response, "\n".join(lines[1:])
        )
        self._plan_state_text = ""

    def _handle_model_failure(self, state: dict[str, Any], reason: str) -> None:
        """Dispatch an LLM failure according to ``failure_policy``.

        benchmark_strict: NO non-LLM strategic fallback -- invalidate the
        benchmark and stop. demo_resilient: the deterministic fallback may
        execute, but the benchmark is invalidated and prominently logged.
        """
        if self._config.get("failure_policy", "benchmark_strict") == "benchmark_strict":
            self._metrics.invalidate(reason)
            self._agent_phase = "failed"
            self._log("benchmark_invalidated", reason)
            self._log("error", "NON-LLM FALLBACK NOT USED — benchmark invalidated (strict mode)")
            self._fail(
                f"benchmark_strict：LLM 决策失败（{reason}）；"
                "不做任何非 LLM 策略兜底，Agent 停止，本局 benchmark 已失效。"
            )
            self._stop.set()
            return
        act = self._fallback_action(state)
        result = self._send_single_action(act, state)
        self._metrics.record_fallback(reason)
        self._decision_count += 1
        self._log("error", "NON-LLM FALLBACK USED — benchmark invalidated")
        self._log(
            "decision",
            "(LLM 请求失败，demo 兜底动作，benchmark 已失效)",
            action=json.dumps(act),
            result=result,
            state_type=str(state.get("type", "")),
        )
        self._ctx.add_decision(
            format_state(state, self._memory), json.dumps(act), result
        )

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
        # One cognitive boundary per state that needs the model.
        self._metrics.record_inspection()
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
            # (llm_request_count is recorded by llm.on_http_attempt --
            # one count per REAL HTTP attempt, retries included.)
            try:
                t0 = time.monotonic()
                raw_reply = self._chat_with_deadline(llm, messages, call_budget)
                llm_ms = int((time.monotonic() - t0) * 1000)
                if llm_ms > 15000:
                    self._log("info", f"LLM 响应耗时 {llm_ms / 1000:.1f}s（较慢）")
                self._llm_fail_streak = 0
            except EmptyContentError as e:
                self._metrics.record_llm_failure()
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
                self._metrics.record_llm_failure()
                # Strict benchmark mode: NO fallback, invalidate + stop.
                if self._config.get("failure_policy", "benchmark_strict") == "benchmark_strict":
                    self._handle_model_failure(state, f"LLM API error: {e}")
                    return
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
                result_note = self._send_single_action(act, state)
                self._metrics.record_fallback(f"LLM API error: {e}")
                self._log("error", "NON-LLM FALLBACK USED — benchmark invalidated")
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
                self._metrics.record_llm_failure()
                self._fail(f"LLM 客户端意外错误: {e}")
                self._stop.set()
                return
            self._metrics.record_llm_success(
                latency_ms=llm_ms if llm_ms is not None else None,
                first_reasoning_ms=getattr(llm, "first_reasoning_ms", None),
                first_content_ms=getattr(llm, "first_content_ms", None),
                usage=getattr(llm, "last_usage", None) or None,
            )
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
            if self._config.get("failure_policy", "benchmark_strict") == "benchmark_strict":
                self._handle_model_failure(state, "all decision attempts failed")
                return
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
            self._metrics.record_fallback("decision attempts exhausted")
            self._log(
                "error",
                f"{kind}: {json.dumps(act)} after {attempts} failed"
                f" decision attempts (shared decision deadline).",
            )

        if act is not None:
            self._metrics.record_plan(1)
        result_note = self._send_single_action(act, state)
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

    def _send_single_action(self, act: dict[str, Any], state: dict[str, Any]) -> str:
        """Single-action-mode send: this only proves ACTION SENT (never
        confirmed). The confirmation happens in
        _reconcile_pending_single_action() against the NEXT authoritative
        state -- exactly the same semantics as ActionChunk steps."""
        result = self._execute(act)
        if str(result).startswith("ERROR"):
            # The action never reached the bridge: not sent, never confirmed.
            return result
        self._metrics.record_action_sent(from_plan=False)
        # Snapshot the before-state (deep copy; the agent owns/reads it).
        try:
            snapshot = json.loads(json.dumps(state, ensure_ascii=False, default=str))
        except Exception:
            snapshot = dict(state)
        self._pending_single_action = act
        self._pending_single_before_state = snapshot
        return result

    def _visible_state_fingerprint(self, state: dict[str, Any]) -> str | None:
        """Generic human-visible state fingerprint (single-action
        confirmation).

        Uses the SAME formatter the LLM sees, so it works for every screen
        type (event->event, reward_screen->reward_screen, combat->combat):
        it contains only human-visible decision information, and
        request_id is never part of the formatted text. Returns None when
        formatting fails -- the caller must then NOT claim confirmation.
        """
        try:
            text = format_state(state, None)
        except Exception:
            return None
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _reconcile_pending_single_action(self, state: dict[str, Any]) -> None:
        """Confirm/reject the previously sent single action against the
        next authoritative bridge state (main-loop Step A for
        single_action mode)."""
        act = self._pending_single_action
        before = self._pending_single_before_state
        self._pending_single_action = None
        self._pending_single_before_state = None
        if act is None or before is None:
            return
        fp_before = self._visible_state_fingerprint(before)
        fp_after = self._visible_state_fingerprint(state)
        if fp_before is None or fp_after is None or not fp_before or not fp_after:
            # Conservative: never silently claim a confirmation.
            self._metrics.record_action_unconfirmable()
            self._log(
                "warning",
                "UNKNOWN_CONFIRMATION: 无法可靠比较动作前后的可见状态，"
                "该动作既不计为 confirmed 也不计为 rejected。",
            )
            return
        if fp_before == fp_after:
            # Action-relevantly unchanged state => the game did not accept
            # the action; identical to the chunk ACTION_REJECTED semantics.
            self._metrics.record_action_rejected()
        else:
            self._metrics.record_action_confirmed(from_plan=False)

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
