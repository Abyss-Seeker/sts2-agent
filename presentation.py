"""Read-only, bounded view model for the native recording HUD."""
from __future__ import annotations
import json


def language_instruction(language: str) -> str:
    name = "Simplified Chinese (简体中文)" if language != "en" else "English"
    return (f"\nPresentation language: {name}. Write the thought / summary field as a "
            "short first-person tactical explanation for the audience (at most two "
            "sentences). If the provider exposes reasoning text, prefer this language "
            "there too. Keep JSON keys, action types, IDs and references unchanged.\n")


def overlay_snapshot(logs: list[dict], config: dict) -> dict:
    chinese = config.get("presentation_language", "zh") != "en"
    thoughts = [e for e in logs if e.get("kind") in ("decision", "model_plan")]
    errors = [e for e in logs if e.get("kind") in ("error", "warning")]
    actions = [e for e in logs if e.get("action") or e.get("actions")]
    latest = thoughts[-1] if thoughts else {}
    error = errors[-1] if errors else {}
    text = str(latest.get("text", ""))
    seq = latest.get("seq", 0)
    if error.get("seq", 0) > seq:
        raw = str(error.get("text", "")).lower()
        truncated = any(s in raw for s in ("truncat", "thinking-only", "reasoning", "截断", "too long"))
        text = (("我得再想想……" if truncated else "稍等，我整理一下思路……") if chinese
                else ("Let me think a little more…" if truncated else "One moment, let me regroup…"))
        seq = error.get("seq", 0)
    action = actions[-1] if actions else {}
    return {
        "language": "zh" if chinese else "en",
        "bubble_seq": seq, "bubble": text,
        "action_seq": action.get("seq", 0),
        "action": str(action.get("action") or action.get("actions") or "")[:1800],
        "error_seq": error.get("seq", 0),
        "error": str(error.get("text", ""))[:700],
    }


def overlay_feed(logs: list[dict], config: dict, after: int, stream_id: str) -> dict:
    """Page the same complete records as /api/logs, without clipping their fields.

    Limit batches, not individual entries; a large diagnostic remains readable.
    The process token allows the HUD to discard stale history after server restart.
    """
    result = overlay_snapshot(logs, config)
    pending = [entry for entry in logs if entry["seq"] > after]
    batch: list[dict] = []
    size = 0
    for entry in pending:
        entry_size = len(json.dumps(entry, ensure_ascii=False).encode("utf-8"))
        if batch and (len(batch) >= 32 or size + entry_size > 256_000):
            break
        batch.append(entry)
        size += entry_size
    result.update(stream_id=stream_id, logs=batch,
                  next_seq=batch[-1]["seq"] if batch else after,
                  has_more=len(pending) > len(batch),
                  oldest_seq=logs[0]["seq"] if logs else 0)
    return result
