"""P0 diagnostic: Steam cold-start, distinguishing failure classes A-E.

A. Steam command never executed
B. Steam received it, game never started
C. game started, mod did not load
D. mod loaded, bridge not listening
E. bridge latency beyond timeout

Run: python diag_steam_launch.py
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from game_launcher import find_steam_exe, is_game_running, launch_via_steam

GAME = "SlayTheSpire2.exe"


def steam_running() -> bool:
    out = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq steam.exe", "/NH"],
        capture_output=True, text=True, timeout=10,
        creationflags=0x08000000,
    )
    return "steam.exe" in (out.stdout or "").lower()


def bridge_open() -> bool:
    s = socket.socket()
    s.settimeout(0.5)
    try:
        return s.connect_ex(("127.0.0.1", 9002)) == 0
    finally:
        s.close()


def main() -> None:
    t0 = time.time()
    print(f"[{elapsed(t0)}] DIAG start", flush=True)

    if is_game_running():
        print(f"[{elapsed(t0)}] killing running game first (cold-start test)",
              flush=True)
        subprocess.run(["taskkill", "/F", "/IM", GAME],
                       capture_output=True, creationflags=0x08000000)
        time.sleep(6)

    print(f"[{elapsed(t0)}] GAME_PROCESS_NOT_FOUND (killed)", flush=True)
    print(f"[{elapsed(t0)}] steam.exe alive: {steam_running()}"
          "  <- False means class A (no Steam)", flush=True)

    steam = find_steam_exe()
    print(f"[{elapsed(t0)}] steam.exe path: {steam}", flush=True)
    ok = launch_via_steam(log=lambda m: print(f"[{elapsed(t0)}] {m}",
                                              flush=True))
    if not ok:
        print(f"[{elapsed(t0)}] RESULT: class A -- launch command not executed",
              flush=True)
        return
    print(f"[{elapsed(t0)}] STEAM_LAUNCH_REQUESTED", flush=True)

    game_t = None
    bridge_t = None
    while time.time() - t0 < 180:
        now = time.time()
        if game_t is None and is_game_running():
            game_t = now
            print(f"[{elapsed(now)}] GAME_PROCESS_STARTED (+{now-t0:.0f}s)"
                  "  <- never: class B", flush=True)
        if bridge_t is None and game_t is not None and bridge_open():
            bridge_t = now
            print(f"[{elapsed(now)}] BRIDGE_CONNECTED (+{now-t0:.0f}s,"
                  f" +{bridge_t-game_t:.0f}s after process)"
                  "  <- never: class C/D", flush=True)
            print("RESULT: PASS (class E only if this took > launch budget)",
                  flush=True)
            return
        if game_t is not None and now - game_t > 90 and bridge_t is None:
            print(f"[{elapsed(now)}] game up {now-game_t:.0f}s but no bridge"
                  " -- class C/D; check mod load in godot log", flush=True)
        time.sleep(2)

    print(f"[{elapsed(t0)}] RESULT: class B/C/D/E -- see markers above",
          flush=True)


def elapsed(t0: float) -> str:
    return f"+{time.time()-t0:.0f}s"


if __name__ == "__main__":
    main()
