import io
import json
from unittest.mock import Mock

from agent import AgentSession
from presentation import overlay_feed


def test_full_messages_are_console_only_and_toggle_is_live():
    session = AgentSession()
    session._log_file = io.StringIO()
    client = Mock()
    client.chat.return_value = "reply"
    messages = [{"role": "system", "content": "context " * 20000},
                {"role": "user", "content": "<script>not markup</script>"}]
    session._chat_with_deadline(client, messages, 30)
    assert not session.logs_since()
    session.set_show_sent_messages(True)
    for _ in range(2):
        assert session._chat_with_deadline(client, messages, 30) == "reply"
    entries = session.logs_since()
    assert len(entries) == 2
    assert all(json.loads(e["text"]) == messages for e in entries)
    assert session._log_file.getvalue() == ""
    assert overlay_feed(entries, {}, 0, "test")["logs"] == []
    session.set_show_sent_messages(False)
    session._chat_with_deadline(client, messages, 30)
    assert len(session.logs_since()) == 2
    assert client.chat.call_args.args[0] is messages


def test_console_history_bounds_records_without_clipping_messages():
    session = AgentSession()
    session.set_show_sent_messages(True)
    for i in range(25):
        session._chat_with_deadline(Mock(), [{"role": "user", "content": str(i)}], 30)
    entries = session.logs_since()
    assert len(entries) == 20
    assert json.loads(entries[0]["text"])[0]["content"] == "5"
