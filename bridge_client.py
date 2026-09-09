"""Standalone TCP client for the STS2 Bridge Mod.

Cloned from ``sts2_env/bridge/client.py`` + ``protocol.py`` so that the
sts2-agent package is fully self-contained (no sts2_env imports).

Connects to the game's TCP server (default: localhost:9002) and provides
methods for receiving game states and sending actions, using the
newline-delimited JSON protocol implemented by bridge_mod/BridgeServer.cs.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Protocol constants (mirrors BridgeServer.cs)
# ------------------------------------------------------------------

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9002

MSG_TYPE_PONG = "pong"
MSG_TYPE_OK = "ok"  # ack replies (e.g. set_fallback confirmation)
MSG_TYPE_ERROR = "error"


class BridgeStateType:
    """Values for the bridge state payload's top-level "type" field."""

    COMBAT_ACTION = "combat_action"
    CARD_SELECT = "card_select"
    MAP_SELECT = "map_select"
    REWARD_SCREEN = "reward_screen"
    CARD_BUNDLE = "card_bundle"
    CRYSTAL_SPHERE = "crystal_sphere"
    CARD_REWARD = "card_reward"
    REST_SITE = "rest_site"
    SHOP = "shop"
    EVENT = "event"
    TREASURE = "treasure"
    BOSS_RELIC = "boss_relic"
    GAME_OVER = "game_over"
    RUN_COMPLETE = "run_complete"
    PONG = "pong"
    ERROR = "error"


# Screen types where a single "choose index" style decision is expected.
CHOICE_SCREEN_TYPES = {
    BridgeStateType.CARD_SELECT,
    BridgeStateType.MAP_SELECT,
    BridgeStateType.REWARD_SCREEN,
    BridgeStateType.CARD_BUNDLE,
    BridgeStateType.CRYSTAL_SPHERE,
    BridgeStateType.CARD_REWARD,
    BridgeStateType.REST_SITE,
    BridgeStateType.SHOP,
    BridgeStateType.EVENT,
    BridgeStateType.TREASURE,
    BridgeStateType.BOSS_RELIC,
}

# Screen types that end a run.
TERMINAL_SCREEN_TYPES = {
    BridgeStateType.GAME_OVER,
    BridgeStateType.RUN_COMPLETE,
}


class BridgeAction:
    """Values for the action payload's "action" field."""

    PLAY = "play"
    END_TURN = "end_turn"
    CHOOSE = "choose"
    SKIP = "skip"
    POTION = "potion"
    PING = "ping"
    SET_FALLBACK = "set_fallback"


class STS2GameClient:
    """TCP client for the STS2 Bridge Mod.

    Usage::

        client = STS2GameClient()
        client.connect()
        while True:
            state = client.receive_state()
            if state["type"] == "combat_action":
                client.play_card(0, 0)
            else:
                client.choose(0)
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = 120.0,
        reconnect_attempts: int = 30,
        reconnect_delay: float = 2.0,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self._sock: socket.socket | None = None
        self._buffer: bytes = b""
        self._connected: bool = False
        self._last_request_id: str | None = None

    # ----------------------------------------------------------------
    # Connection management
    # ----------------------------------------------------------------

    def connect(self, should_abort=None) -> None:
        """Connect, retrying while the game server starts up.

        ``should_abort``: optional zero-arg callable; when it returns True the
        connect loop is abandoned with a ConnectionError.
        """
        for attempt in range(1, self.reconnect_attempts + 1):
            if should_abort is not None and should_abort():
                raise ConnectionError("Connection aborted by caller")
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.settimeout(self.timeout)
                self._sock.connect((self.host, self.port))
                self._buffer = b""
                self._connected = True
                logger.info(
                    "Connected to STS2 bridge at %s:%d (attempt %d)",
                    self.host, self.port, attempt,
                )
                return
            except (ConnectionRefusedError, OSError) as e:
                logger.warning(
                    "Connection attempt %d/%d failed: %s",
                    attempt, self.reconnect_attempts, e,
                )
                self._cleanup_socket()
                if attempt < self.reconnect_attempts:
                    time.sleep(self.reconnect_delay)

        raise ConnectionError(
            f"Could not connect to STS2 bridge at {self.host}:{self.port} "
            f"after {self.reconnect_attempts} attempts"
        )

    def disconnect(self) -> None:
        self._cleanup_socket()
        self._connected = False
        logger.info("Disconnected from STS2 bridge.")

    @property
    def connected(self) -> bool:
        return self._connected

    # ----------------------------------------------------------------
    # Message I/O
    # ----------------------------------------------------------------

    def receive_state(self) -> dict[str, Any]:
        """Block until a game message arrives; returns the parsed dict."""
        while True:
            msg = self._receive_line()
            if msg is None:
                raise ConnectionError("Connection lost while waiting for state")

            try:
                data = json.loads(msg)
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON from server: %s (raw: %s)", e, msg[:200])
                continue

            msg_type = data.get("type", "")
            if msg_type in (MSG_TYPE_PONG, MSG_TYPE_OK):
                # Ack messages are NOT game states -- treating them as such
                # makes the agent answer them and desync the game flow.
                logger.debug("Received ack: %s", msg_type)
                continue
            elif msg_type == MSG_TYPE_ERROR:
                # ERROR SEMANTICS (verified against BridgeServer.cs): the
                # mod never initiates {"type":"error"} messages; the filter
                # here is purely defensive for protocol acknowledgements.
                # When the game REJECTS an agent action it does NOT send an
                # error -- the relevant handler simply re-serializes the
                # unchanged state and asks again, i.e. the retry loop lives
                # on the mod side and surfaces as a new (identical) state.
                # There is therefore no "action rejected but game still
                # waiting" message type to forward to the agent.
                logger.warning("Server error: %s", data)
                continue
            else:
                self._last_request_id = data.get("request_id")
                return data

    def send_action(self, action: dict[str, Any]) -> None:
        """Send an action command to the game."""
        if not self._connected or self._sock is None:
            raise ConnectionError("Not connected to STS2 bridge")

        try:
            payload = dict(action)
            if "request_id" not in payload and self._last_request_id is not None:
                payload["request_id"] = self._last_request_id
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
            self._sock.sendall(data)
            logger.debug("Sent action: %s", payload)
            self._last_request_id = None
        except (BrokenPipeError, OSError) as e:
            self._connected = False
            raise ConnectionError(f"Lost connection while sending action: {e}") from e

    # ----------------------------------------------------------------
    # Convenience methods
    # ----------------------------------------------------------------

    def play_card(self, card_index: int, target_index: int = -1) -> None:
        self.send_action({
            "action": BridgeAction.PLAY,
            "card_index": card_index,
            "target_index": target_index,
        })

    def end_turn(self) -> None:
        self.send_action({"action": BridgeAction.END_TURN})

    def choose(self, choice_index: int) -> None:
        self.send_action({
            "action": BridgeAction.CHOOSE,
            "index": choice_index,
        })

    def choose_many(self, indexes: list[int]) -> None:
        self.send_action({
            "action": BridgeAction.CHOOSE,
            "indexes": indexes,
        })

    def skip(self) -> None:
        self.send_action({"action": BridgeAction.SKIP})

    def use_potion(self, slot: int, target_index: int = -1) -> None:
        self.send_action({
            "action": BridgeAction.POTION,
            "slot": slot,
            "target_index": target_index,
        })

    def set_fallback(self, enabled: bool) -> None:
        """Enable/disable the mod's random-decision fallback.

        The LLM agent should disable it so every outcome is its own decision.
        """
        self.send_action({
            "action": BridgeAction.SET_FALLBACK,
            "enabled": enabled,
        })

    def set_agent_timeout(self, seconds: int) -> None:
        """Retune the mod's per-decision wait (clamped to 10..300 mod-side)."""
        self.send_action({
            "action": "set_agent_timeout",
            "seconds": int(seconds),
        })

    def set_headful(self, enabled: bool) -> None:
        """Headful: game stays interactive (BGM/SFX/waits on)."""
        self.send_action({
            "action": "set_headful",
            "enabled": bool(enabled),
        })

    def set_fast_mode(self, enabled: bool) -> None:
        """FastMode accelerates native animations (independent of headful)."""
        self.send_action({
            "action": "set_fast_mode",
            "enabled": bool(enabled),
        })

    def ping(self) -> bool:
        try:
            self.send_action({"type": "PING"})
            return True
        except ConnectionError:
            return False

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _receive_line(self) -> str | None:
        if self._sock is None:
            return None

        while b"\n" not in self._buffer:
            try:
                chunk = self._sock.recv(8192)
            except socket.timeout:
                raise TimeoutError(
                    f"No data received within {self.timeout}s timeout"
                )
            except OSError as e:
                self._connected = False
                logger.error("Socket error during receive: %s", e)
                return None

            if not chunk:
                self._connected = False
                return None

            self._buffer += chunk

        line, self._buffer = self._buffer.split(b"\n", 1)
        return line.decode("utf-8", errors="replace").strip()

    def _cleanup_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        self._buffer = b""

    def __enter__(self) -> STS2GameClient:
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.disconnect()
