import json
from unittest.mock import patch

from agent import AgentSession
from game_state import format_state
from llm_client import LLMClient


class Response:
    def __init__(self, lines=(), body=b''):
        self.lines = iter(lines)
        self.body = body
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def readline(self):
        return next(self.lines, b'')


def test_stream_guard_finalizes_draft_and_restores_config():
    llm = LLMClient('https://api.deepseek.com', '', 'deepseek-flash',
                    thinking_enabled=True, reasoning_effort='low',
                    reasoning_max_chars=20, max_retries=0)
    session = AgentSession()
    session._llm = llm
    stream = Response([b'data: ' + json.dumps({'choices': [{'delta': {
        'reasoning_content': 'Compare damage and block carefully.'}}]}).encode()])
    answer = Response()
    requests = []

    def open_request(req, **kwargs):
        requests.append(json.loads(req.data))
        return stream if len(requests) == 1 else answer

    final = b'{"choices":[{"message":{"content":"{\\"action\\":\\"end_turn\\"}"}}]}'
    with patch('urllib.request.urlopen', side_effect=open_request), \
            patch.object(llm, '_read_body', return_value=final):
        result = session._chat_with_deadline(llm, [{'role': 'user', 'content': 'STATE'}], 30)
    assert json.loads(result)['action'] == 'end_turn'
    assert stream.closed
    assert requests[0]['stream'] is True
    assert requests[1]['thinking'] == {'type': 'disabled'}
    assert 'Compare damage' in requests[1]['messages'][-1]['content']
    assert 'STATE' == requests[1]['messages'][0]['content']
    assert llm.thinking_enabled is True and llm.reasoning_effort == 'low'


def test_map_repeated_coordinates_are_one_room():
    node = {'type': 'Ancient', 'row': 0, 'col': 0}
    text = format_state({'type': 'map_select', 'nodes': [node],
                         'full_map': [node, dict(node)]})
    assert '1 distinct node(s)' in text
    assert text.count('(0,0) ANCIENT') <= 1
    assert 'SAME nodes' in text
