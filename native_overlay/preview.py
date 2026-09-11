"""Godot rendering/interaction smoke test with synthetic logs (no game or LLM).

Build Debug with -p:OverlayPreview=true first; pass the Godot console executable.
Artifacts and an isolated user settings directory are written under dist/.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from presentation import overlay_feed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("godot")
    parser.add_argument("--native-pck")
    args = parser.parse_args()
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    logs = [dict(seq=i, ts=f"12:00:{i:02}", kind="decision", text=f"第 {i} 次决策：先格挡，再利用剩余能量反击。",
                 action='{"type":"play","card_index":2,"target_index":0}', result="confirmed")
            for i in range(1, 9)]
    logs.append(dict(seq=9, ts="12:00:09", kind="warning", text="reasoning truncated — retrying"))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            after = int(parse_qs(urlparse(self.path).query).get("after", ["0"])[0])
            if after >= 9 and len(logs) == 9:
                logs.append(dict(seq=10, ts="12:00:10", kind="decision", text="New explanation must wait for the previous page."))
            body = json.dumps(overlay_feed(logs, {}, after, self.server.stream_id), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            logs[:] = [dict(seq=1, ts="13:00:00", kind="decision", text="NEW SESSION / 新会话")]
            self.server.stream_id = "preview-b"
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.stream_id = "preview-a"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    profile = dist / "preview_profile"
    user_dir = profile / "Godot/app_userdata/STS2AgentOverlay"
    user_dir.mkdir(parents=True, exist_ok=True)
    settings = {"Url": f"http://127.0.0.1:{server.server_port}/api/overlay"}
    # ConfigFile stores JSON as an escaped Variant string.
    (user_dir / "agent_overlay.cfg").write_text('[overlay]\njson=' + json.dumps(json.dumps(settings)) + '\n')
    (dist / "preview.tscn").write_text('''[gd_scene load_steps=4 format=3]
[ext_resource type="Script" path="res://Overlay.cs" id="1"]
[ext_resource type="Script" path="res://dist/capture.gd" id="2"]
[ext_resource type="Script" path="res://NativePreview.cs" id="3"]
[node name="Preview" type="Node"]
script = ExtResource("2")
[node name="HUD" type="CanvasLayer" parent="."]
script = ExtResource("1")
[node name="NativeFixture" type="Node" parent="."]
script = ExtResource("3")
''')
    (dist / "capture.gd").write_text('''extends Node
var failures = 0
func check(value, message):
    if not value:
        failures += 1
        push_error(message)
func toggle_menu():
    var key = InputEventKey.new()
    key.keycode = KEY_F8
    key.alt_pressed = true
    key.pressed = true
    Input.parse_input_event(key)
func _ready():
    await get_tree().create_timer(2.5).timeout
    var feed = $HUD.find_child("Feed", true, false)
    var menu = $HUD.get_node("SettingsMenu")
    check(feed.get_parsed_text().find("#9") < feed.get_parsed_text().find("#8"), "Newest entry must appear first")
    check(feed.get_parsed_text().contains("confirmed"), "Complete result field missing")
    check($HUD.get_node("Subtitle").visible, "Missing subtitle fallback outside combat")
    check($HUD.get_node("Subtitle").text.contains("我得再想想"), "New decision interrupted unfinished speech")
    check($HUD.get_node("Subtitle").get_visible_line_count() == $HUD.get_node("Subtitle").get_line_count(), "Subtitle clipped its last line")
    if not OS.get_environment("STS2_OVERLAY_PREVIEW_PCK").is_empty():
        check($NativeFixture.has_node("NativeExample"), "Native speech factory failed")
        var native_label = $NativeFixture/NativeExample.get_node("%Text")
        check(native_label.get_content_height() <= native_label.size.y, "Native dialogue text overflows its frame")
    await RenderingServer.frame_post_draw
    get_viewport().get_texture().get_image().save_png("res://dist/preview.png")
    toggle_menu()
    await get_tree().process_frame
    check(menu.visible, "Alt+F8 did not open settings")
    await RenderingServer.frame_post_draw
    get_viewport().get_texture().get_image().save_png("res://dist/settings.png")
    var playback = menu.find_child("SpeechPlayback", true, false)
    check(playback.selected == 1, "Readability default missing")
    playback.select(0)
    playback.item_selected.emit(0)
    toggle_menu()
    await get_tree().process_frame
    await get_tree().create_timer(0.1).timeout
    check($HUD.get_node("Subtitle").text.contains("New explanation"), "Immediacy did not clear backlog and show latest")
    toggle_menu()
    await get_tree().process_frame
    var toggles = menu.find_children("*", "CheckButton", true, false)
    check(toggles.size() >= 8, "Settings controls missing")
    toggles[0].button_pressed = false
    await get_tree().create_timer(0.1).timeout
    check(not $HUD.get_node("LogPanel").visible, "Disable overlay did not apply")
    var config = ConfigFile.new()
    check(config.load("user://agent_overlay.cfg") == OK, "Settings file missing")
    check(JSON.parse_string(config.get_value("overlay", "json")).Enabled == false, "Settings not saved")
    check(JSON.parse_string(config.get_value("overlay", "json")).QueueCommentary == false, "Playback preference not saved")
    toggle_menu()
    await get_tree().process_frame
    toggle_menu()
    await get_tree().process_frame
    check(menu.visible, "Cannot reopen settings with HUD disabled")
    toggles[0].button_pressed = true
    await get_tree().create_timer(1).timeout
    check(feed.get_parsed_text().count("#9") == 1, "Repeated poll duplicated records")
    for item in toggles:
        if item.text.contains("Follow newest"):
            item.button_pressed = false
    await get_tree().create_timer(0.1).timeout
    feed.get_v_scroll_bar().value = 100
    var reset = HTTPRequest.new()
    add_child(reset)
    reset.request(OS.get_environment("STS2_OVERLAY_PREVIEW_URL") + "/fixture/reset", [], HTTPClient.METHOD_POST)
    await reset.request_completed
    await get_tree().create_timer(2).timeout
    check(feed.get_parsed_text().contains("NEW SESSION"), "Server restart did not reset cursor")
    check(not feed.get_parsed_text().contains("#9"), "Old session history leaked after restart")
    print("OVERLAY_PREVIEW_OK: subtitle, history, Alt+F8, settings persistence, disabled HUD recovery")
    get_tree().quit(1 if failures else 0)
''', encoding="utf-8")
    startup = subprocess.STARTUPINFO()
    startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    try:
        with (dist / "preview_stdout.log").open("w", encoding="utf-8") as output:
            result = subprocess.run([args.godot, "--path", str(ROOT), "--rendering-method", "gl_compatibility",
                                 "--resolution", "1920x1080", "--log-file", str(dist / "preview.log"),
                                 "res://dist/preview.tscn"], env={**os.environ, "APPDATA": str(profile),
                                 "STS2_OVERLAY_PREVIEW_PCK": args.native_pck or "",
                                 "STS2_OVERLAY_PREVIEW_URL": f"http://127.0.0.1:{server.server_port}"},
                                startupinfo=startup, timeout=35, stdout=output, stderr=subprocess.STDOUT)
        output_text = (dist / "preview_stdout.log").read_text(encoding="utf-8", errors="replace")
        print(output_text[-6000:])
        if any(marker in output_text for marker in ("Exception:", "Cannot instantiate", "SCRIPT ERROR")):
            return 1
        return result.returncode
    finally:
        server.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
