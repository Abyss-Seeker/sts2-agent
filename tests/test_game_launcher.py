from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

import game_launcher


def test_machine_registry_fallback_after_missing_or_stale_user_path():
    winreg = pytest.importorskip("winreg")
    steam_dir = Path(r"E:\Program Files (x86)\Steam")

    def query(key, name):
        if name == "SteamExe":
            raise FileNotFoundError()
        if name == "SteamPath":
            return str(Path("old-install")), 1
        return str(steam_dir), 1

    with patch.object(game_launcher.sys, "platform", "win32"), \
         patch.object(Path, "is_file", autospec=True,
                      side_effect=lambda path: path == steam_dir / "steam.exe"), \
         patch.object(winreg, "OpenKey", return_value=MagicMock()) as opened, \
         patch.object(winreg, "QueryValueEx", side_effect=query):
        assert game_launcher.find_steam_exe() == str(steam_dir / "steam.exe")
        assert opened.call_args.args[0] == winreg.HKEY_LOCAL_MACHINE
        assert opened.call_args.args[3] & winreg.KEY_WOW64_32KEY


def test_launch_uses_discovered_executable_without_shell():
    executable = r"E:\Program Files (x86)\Steam\steam.exe"
    with patch.object(game_launcher, "find_steam_exe", return_value=executable), \
         patch.object(game_launcher.subprocess, "Popen") as launch:
        assert game_launcher.launch_via_steam(log=lambda _: None)
        assert launch.call_args.args[0] == [executable, "-applaunch", "2868840"]
        assert not launch.call_args.kwargs.get("shell")
