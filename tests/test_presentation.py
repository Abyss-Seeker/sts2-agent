import unittest
import json
import threading
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from unittest.mock import patch

from presentation import language_instruction, overlay_snapshot, overlay_feed


class PresentationTests(unittest.TestCase):
    def test_chunk_and_executed_action_are_independent(self):
        logs = [dict(seq=1, kind="model_plan", text="先格挡。", actions='[{"type":"play"}]'),
                dict(seq=2, kind="action", text="sent", action='{"type":"play","card_index":0}')]
        snap = overlay_snapshot(logs, {})
        self.assertEqual(snap["bubble"], "先格挡。")
        self.assertEqual(snap["action_seq"], 2)
        self.assertNotIn("logs", snap)

    def test_failure_is_friendly_but_original_is_preserved(self):
        logs = [dict(seq=3, kind="warning", text="reasoning truncated")]
        self.assertEqual(overlay_snapshot(logs, {})["bubble"], "我得再想想……")
        en = overlay_snapshot(logs, {"presentation_language": "en"})
        self.assertEqual(en["error"], "reasoning truncated")
        self.assertEqual(en["bubble"], "Let me think a little more…")
        logs.append(dict(seq=4, kind="decision", text="Recovered"))
        self.assertEqual(overlay_snapshot(logs, {})["bubble"], "Recovered")

    def test_empty_and_complete(self):
        self.assertEqual(overlay_snapshot([], {})["bubble"], "")
        snap = overlay_snapshot([dict(seq=1, kind="decision", text="x" * 9000)], {})
        self.assertEqual(snap["bubble"], "x" * 9000)

    def test_language_does_not_translate_protocol(self):
        self.assertIn("简体中文", language_instruction("zh"))
        self.assertIn("English", language_instruction("en"))
        self.assertIn("JSON keys, action types, IDs and references unchanged", language_instruction("zh"))

    def test_full_log_feed_pages_without_losing_or_clipping_fields(self):
        logs = [dict(seq=i, kind="state" if i % 2 else "decision", text="中文" * 2000,
                     action='{"type":"play"}', result="confirmed", extra={"n": i})
                for i in range(1, 76)]
        received, after = [], 0
        while True:
            page = overlay_feed(logs, {}, after, "run-a")
            received.extend(page["logs"])
            self.assertEqual(page["stream_id"], "run-a")
            after = page["next_seq"]
            if not page["has_more"]:
                break
        self.assertEqual(received, logs)
        self.assertEqual(overlay_feed(logs, {}, after, "run-a")["logs"], [])

    def test_large_record_is_not_silently_dropped(self):
        entry = dict(seq=1, kind="warning", text="x" * 300_000)
        page = overlay_feed([entry, dict(seq=2, kind="info", text="done")], {}, 0, "b")
        self.assertEqual(page["logs"], [entry])
        self.assertTrue(page["has_more"])


class OverlayApiTests(unittest.TestCase):
    def setUp(self):
        import server
        self.module = server
        class QuietHandler(server.Handler):
            def log_message(self, *_args):
                pass
        self.http = server.ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.http.server_port}"

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(timeout=2)

    def test_feed_cursor_and_full_record(self):
        records = [dict(seq=i, kind="decision", text="说明", actions='[{"type":"end_turn"}]')
                   for i in range(1, 4)]
        with patch.object(self.module, "load_config", return_value={}), \
                patch.object(self.module.session, "logs_since", return_value=records):
            with urlopen(self.base + "/api/overlay?after=1", timeout=3) as response:
                body = json.load(response)
            self.assertEqual(body["logs"], records[1:])
            self.assertEqual(body["next_seq"], 3)
            self.assertEqual(body["stream_id"], self.module.OVERLAY_STREAM_ID)

    def test_language_updates_only_language_without_returning_config(self):
        config = {"presentation_language": "zh", "api_key": "test-only-key", "model": "example"}
        with patch.object(self.module, "load_config", return_value=config), \
                patch.object(self.module, "save_config") as save:
            request = Request(self.base + "/api/overlay/language", data=b'{"language":"en"}',
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=3) as response:
                body = json.load(response)
            self.assertEqual(body, {"language": "en", "restart_agent_required": True})
            self.assertEqual(save.call_args.args[0], {**config, "presentation_language": "en"})
            self.assertNotIn("api_key", body)
            request = Request(self.base + "/api/overlay/language", data=b'{"language":"invalid"}')
            with self.assertRaises(HTTPError) as error:
                urlopen(request, timeout=3)
            self.assertEqual(error.exception.code, 400)
            self.assertEqual(save.call_count, 1)


if __name__ == "__main__":
    unittest.main()
