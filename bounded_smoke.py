"""Real API smoke: at most five calls, no retries, four minute wall cap."""
import json
import time
from pathlib import Path

from agent import AgentSession
from llm_client import LLMError
from server import load_config


class BoundedSession(AgentSession):
    calls = 0

    def _chat_with_deadline(self, llm, messages, deadline):
        if self.calls >= 5:
            self._stop.set()
            raise LLMError("SMOKE_CALL_CAP: five API calls reached")
        self.calls += 1
        print(f"API_CALL {self.calls}/5", flush=True)
        return super()._chat_with_deadline(llm, messages, deadline)


if __name__ == "__main__":
    cfg = load_config()
    cfg.update(llm_retries=0, disable_fallback=True, auto_resume=False,
               auto_launch_game=True, save_log=True)
    session = BoundedSession()
    session.start(cfg)
    last = 0
    started = time.monotonic()
    try:
        while time.monotonic() - started < 240:
            time.sleep(0.5)
            for entry in session.logs_since(last):
                last = max(last, entry["seq"])
                print(json.dumps(entry, ensure_ascii=False), flush=True)
            if not session.status()["running"]:
                break
    finally:
        session.stop()
        result = {"calls": session.calls, "status": session.status(),
                  "logs": session.logs_since(0)}
        out = Path("logs") / "bounded_smoke_latest.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"SMOKE_FINISHED calls={session.calls} report={out}", flush=True)
