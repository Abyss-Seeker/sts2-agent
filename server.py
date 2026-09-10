"""Local web UI + API server for the STS2 LLM agent.

Run from this directory (or anywhere, paths are file-relative):

    python server.py --port 8770

Then open http://127.0.0.1:8770/ in a browser, paste your API config and
prompt templates, start the game (with bridge_mod loaded) and press Start.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from agent import DEFAULT_CONFIG, AgentSession, migrate_prompt_config

PROMPT_MIGRATION_WARNINGS: list[str] = []

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
CONFIG_PATH = BASE_DIR / "config.json"
_config_lock = threading.Lock()


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                cfg.update(saved)
        except Exception as e:
            logger.warning("Could not read %s: %s", CONFIG_PATH, e)
    cfg, warnings = migrate_prompt_config(cfg)
    if warnings:
        PROMPT_MIGRATION_WARNINGS.extend(warnings)
        for w in warnings:
            print(f"[prompt-migration] {w}", flush=True)
        save_config(cfg)  # persist the migrated schema
    return cfg

session = AgentSession()


def save_config(cfg: dict) -> None:
    with _config_lock:
        try:
            CONFIG_PATH.write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.error("Could not save %s: %s", CONFIG_PATH, e)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        # 展示叠加层（录视频 / 直播用）：只有基本信息 + 最近决策，背景透明。
        if path in ("/overlay", "/overlay.html"):
            self._send_file(STATIC_DIR / "overlay.html", "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            name = path[len("/static/"):].lstrip("/")
            safe = STATIC_DIR / name
            if safe.resolve().parent == STATIC_DIR.resolve() and safe.exists():
                ctype = {
                    ".css": "text/css; charset=utf-8",
                    ".js": "application/javascript; charset=utf-8",
                    ".html": "text/html; charset=utf-8",
                }.get(safe.suffix, "application/octet-stream")
                self._send_file(safe, ctype)
                return
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if path == "/api/status":
            self._send_json(session.status())
            return
        if path == "/api/logs":
            qs = parse_qs(parsed.query)
            try:
                after = int(qs.get("after", ["0"])[0])
            except ValueError:
                after = 0
            self._send_json({"logs": session.logs_since(after)})
            return
        if path == "/api/config":
            self._send_json(load_config())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except json.JSONDecodeError:
            body = {}

        if parsed.path == "/api/config":
            cfg = load_config()
            cfg.update(body)
            save_config(cfg)
            self._send_json({"ok": True, "config": cfg})
            return
        if parsed.path == "/api/agent/start":
            cfg = load_config()
            cfg.update(body)
            save_config(cfg)
            try:
                session.start(cfg)
                self._send_json({"ok": True})
            except RuntimeError as e:
                self._send_json({"ok": False, "error": str(e)}, HTTPStatus.CONFLICT)
            return
        if parsed.path == "/api/agent/stop":
            session.stop()
            self._send_json({"ok": True})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the STS2 LLM-agent UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--open", action="store_true", help="Open the UI in a browser.")
    args = parser.parse_args(argv)
    server = make_server(args.host, args.port)
    url = f"http://{args.host}:{args.port}/"
    print(f"STS2 LLM-agent UI: {url}  (Ctrl+C to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
