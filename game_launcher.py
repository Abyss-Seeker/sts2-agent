"""Game process detection + Steam launch (cloned from scripts/watch_phase1a.py).

- ``is_game_running()``: check whether SlayTheSpire2.exe is running.
- ``launch_via_steam()``: start the game through Steam; the bridge mod loads
  automatically and AutoSlay begins a run at the main menu, so the caller
  only needs to wait for the TCP port (connect() retries handle that).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

GAME_PROCESS_NAME = "SlayTheSpire2.exe"
STS2_APPID = "2868840"

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _registry_steam_path() -> str | None:
    if sys.platform != "win32":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
            value, _ = winreg.QueryValueEx(key, "SteamPath")
            return str(value)
    except OSError:
        return None


def find_steam_exe() -> str | None:
    """Locate steam.exe (registry first, then common install paths)."""
    from pathlib import Path

    candidates = []
    reg = _registry_steam_path()
    if reg:
        candidates.append(reg)
    candidates += [
        r"C:\Program Files (x86)\Steam",
        os.path.expanduser("~/.local/share/Steam"),
        os.path.expanduser("~/Library/Application Support/Steam"),
    ]
    for path in candidates:
        exe = Path(path) / "steam.exe"
        if exe.exists():
            return str(exe)
    return None


def is_game_running() -> bool:
    """Check whether the game process is running (Windows tasklist)."""
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {GAME_PROCESS_NAME}", "/NH"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=_CREATE_NO_WINDOW,
            )
            return GAME_PROCESS_NAME.lower() in (out.stdout or "").lower()
        except (OSError, subprocess.TimeoutExpired):
            return False
    try:
        out = subprocess.run(
            ["pgrep", "-f", GAME_PROCESS_NAME],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def launch_via_steam(app_id: str = STS2_APPID, log=print) -> bool:
    """Start StS2 through Steam. Returns True if the launch was requested."""
    steam = find_steam_exe()
    if steam is None:
        log("未找到 steam.exe（注册表与常见路径均未命中），请手动启动游戏。")
        return False
    try:
        subprocess.Popen(
            [steam, "-applaunch", str(app_id)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_CREATE_NO_WINDOW,
        )
        log(f"已请求 Steam 启动游戏（{steam} -applaunch {app_id}）。")
        return True
    except OSError as e:
        log(f"Steam 启动失败: {e}")
        return False


def wait_for_game_gone(timeout: float = 20.0, log=print) -> bool:
    """Wait until the game process is REALLY gone (§5 idempotency).

    Steam ignores `-applaunch` for an app that is still shutting down, so
    launching right after a kill silently fails. Returns True when the
    process has disappeared within the timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_game_running():
            return True
        time.sleep(1.0)
    log("等待游戏进程退出超时（Steam 可能忽略此次 -applaunch）。")
    return is_game_running() is False


def ensure_game_running(
    *,
    launch_attempts: int = 3,
    process_timeout: float = 180.0,
    log=print,
) -> bool:
    """Idempotent cold-start (§5/§6/§7): observe, launch at most
    ``launch_attempts`` times (re-observing before every attempt), and
    wait bounded for the game process to appear. Emits launch telemetry.

    Returns True once the game process is running. Bridge readiness is
    the caller's concern (connect retries).
    """
    if is_game_running():
        log("GAME_PROCESS_RUNNING（无需启动）")
        return True

    steam_alive = find_steam_exe() is not None
    log(f"GAME_PROCESS_NOT_FOUND; STEAM_PROCESS_FOUND={steam_alive}")

    for attempt in range(1, launch_attempts + 1):
        # Re-observe before every attempt; never spam the same command.
        if is_game_running():
            return True
        if not wait_for_game_gone(timeout=20.0, log=log):
            # Still shutting down -- wait and re-observe instead of launching.
            time.sleep(5.0)
            continue
        log(f"STEAM_LAUNCH_REQUESTED (attempt {attempt}/{launch_attempts})")
        if not launch_via_steam(log=log):
            time.sleep(5.0)
            continue
        deadline = time.monotonic() + process_timeout
        while time.monotonic() < deadline:
            if is_game_running():
                log(f"GAME_PROCESS_STARTED (attempt {attempt})")
                return True
            time.sleep(2.0)
        log(f"GAME_PROCESS_NOT_STARTED after attempt {attempt}（超时）。")
    return False
