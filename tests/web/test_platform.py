import os
import threading
import time

import pytest
from dotenv import dotenv_values

pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient

from sr_harness.agents.sr_agent_interactive import SRAgentInteractive
from sr_harness.core import APICallResult, SearchRunState, ToolCall
from sr_harness.api import BaseAPI
from sr_harness.web.app import create_app
from sr_harness.runtime import InteractionController
from sr_harness.web.session import InteractiveSession


@pytest.fixture
def platform(tmp_path):
    session = InteractiveSession(tmp_path, agent_options={'tools': ['evaluate_formula', 'workspace_shell']})
    with TestClient(create_app(tmp_path, controller=session.controller, session=session)) as client:
        yield client, session


def test_workspace_roundtrip_and_boundaries(platform, tmp_path):
    client, session = platform
    assert client.get('/').status_code == 200
    assert client.get('/viewer').status_code == 200
    assert client.put('/api/workspace/upload?path=data/sample.csv', content=b'x,y\n1,2').status_code == 200
    assert client.get('/api/workspace/download?path=data/sample.csv').content == b'x,y\n1,2'
    assert client.get('/api/workspace?path=data').json()['entries'][0]['name'] == 'sample.csv'
    demo = client.post('/api/data/demo').json()
    assert demo['path'] == 'demo.csv'
    assert 'demo.csv' in client.get('/api/data/csv-files').json()['files']
    preview = client.get('/api/data/preview', params={'path': 'demo.csv', 'rows': 5}).json()
    assert preview['columns'] == ['x1', 'x2', 'x3', 'y']
    assert preview['numeric'] == {'x1': True, 'x2': True, 'x3': False, 'y': True}
    full_preview = client.get('/api/data/preview', params={'path': 'demo.csv', 'rows': 100}).json()
    assert {row['x3'] for row in full_preview['rows']} == {'alpha', 'beta', 'gamma', 'delta'}
    assert len(preview['rows']) == 5
    assert preview['truncated']
    prompts = client.post('/api/data/prompts', json={
        'dataset': 'demo.csv', 'target': 'y', 'features': ['x1', 'x2'],
        'problem_description': 'Explain the curve.',
        'variable_descriptions': {'x1': 'first input', 'y': 'measured response'},
    })
    assert prompts.status_code == 200
    assert 'Symbolic Regression Agent' in prompts.json()['system_prompt']
    assert 'Explain the curve.' in prompts.json()['user_prompt']
    assert "Feature names: ['x1', 'x2']" in prompts.json()['user_prompt']
    assert '- x1: first input' in prompts.json()['user_prompt']
    assert '- y: measured response' in prompts.json()['user_prompt']
    assert client.post('/api/data/prompts', json={
        'dataset': 'demo.csv', 'target': 'y', 'features': 'x',
    }).status_code == 400
    assert client.put('/api/workspace/upload?path=data/sample.csv', content=b'overwrite').status_code == 409
    for path in ['../outside', '/tmp/outside']:
        assert client.put('/api/workspace/upload', params={'path': path}, content=b'no').status_code == 400
    outside = tmp_path / 'outside'
    outside.write_text('private')
    (session.workspace / 'link').symlink_to(outside)
    assert client.get('/api/workspace/download?path=link').status_code == 400
    assert client.post('/api/session/start', json={'max_refinement_depth': 0}).status_code == 400
    assert session.state == 'idle'
    assert client.post('/api/session/start', json={'dataset': 'missing.csv'}).status_code == 400


def test_runtime_capabilities_can_be_configured(platform):
    client, session = platform
    capabilities = client.get('/api/session/capabilities')
    assert capabilities.status_code == 200
    payload = capabilities.json()
    tool_names = {tool['name'] for tool in payload['tools']}
    skill_names = {skill['name'] for skill in payload['skills']}
    assert {'evaluate_formula', 'workspace_shell'} <= tool_names
    assert 'code_executor' not in tool_names
    assert 'commit_data' not in tool_names
    assert 'discover-symbolic-laws' in skill_names

    response = client.post('/api/session/settings', json={
        'llm_provider': 'openrouter',
        'llm_model': 'test-model',
        'tools': ['evaluate_formula'],
        'skills': [],
        'max_refinement_depth': 12,
    })
    assert response.status_code == 200
    assert response.json()['settings']['tools'] == ['evaluate_formula']
    assert response.json()['settings']['skills'] == []
    assert response.json()['settings']['max_refinement_depth'] == 12
    assert client.post('/api/session/settings', json={
        'tools': ['not-a-tool'],
    }).status_code == 400


def test_provider_api_key_is_synced_to_dotenv_without_being_returned(
    tmp_path, monkeypatch,
):
    env_path = tmp_path / '.env'
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    session = InteractiveSession(tmp_path / 'logs', env_path=env_path)
    app = create_app(tmp_path, controller=session.controller, session=session)

    with TestClient(app) as client:
        initial = client.get(
            '/api/session/provider-credential', params={'provider': 'openrouter'},
        )
        assert initial.status_code == 200
        assert initial.json() == {
            'provider': 'openrouter',
            'env_var': 'OPENROUTER_API_KEY',
            'configured': False,
            'stored_in_env_file': False,
        }

        response = client.put('/api/session/provider-credential', json={
            'provider': 'openrouter', 'api_key': 'test-secret-key',
        })
        assert response.status_code == 200
        assert response.json()['configured'] is True
        assert response.json()['stored_in_env_file'] is True
        assert 'test-secret-key' not in response.text
        assert dotenv_values(env_path)['OPENROUTER_API_KEY'] == 'test-secret-key'
        assert os.environ['OPENROUTER_API_KEY'] == 'test-secret-key'

        assert client.get(
            '/api/session/provider-credential', params={'provider': 'unknown'},
        ).status_code == 400
        assert client.put('/api/session/provider-credential', json={
            'provider': 'openrouter', 'api_key': 'line-one\nline-two',
        }).status_code == 400


def test_data_agent_has_independent_runtime_settings(platform):
    client, session = platform
    capabilities = client.get(
        '/api/session/capabilities', params={'agent': 'data'},
    )
    assert capabilities.status_code == 200
    catalog = capabilities.json()
    assert {'commit_data', 'workspace_code_executor', 'read_skill'} <= {
        tool['name'] for tool in catalog['tools']
    }
    assert 'commit_data' in catalog['default_tools']

    response = client.put('/api/data/agent/settings', json={
        'llm_provider': 'openai',
        'llm_model': 'test-data-model',
        'tool_parser': 'json',
        'llm_max_tokens': 2048,
        'max_turns': 4,
        'tools': ['workspace_shell', 'commit_data'],
        'skills': [],
    })
    assert response.status_code == 200
    assert response.json()['data_agent_settings'] == {
        'llm_provider': 'openai',
        'llm_model': 'test-data-model',
        'tool_parser': 'json',
        'llm_max_tokens': 2048,
        'max_turns': 4,
        'tools': ['workspace_shell', 'commit_data'],
        'skills': [],
    }
    assert session.settings['llm_provider'] == 'openrouter'
    assert client.put('/api/data/agent/settings', json={
        'tools': ['workspace_shell'],
    }).status_code == 400


def test_data_agent_commits_excel_to_shared_context(platform, monkeypatch):
    client, session = platform
    import pandas as pd

    configured = client.put('/api/data/agent/settings', json={
        'llm_provider': 'deepseek',
        'llm_model': 'data-preparation-model',
        'tool_parser': 'openai',
        'llm_max_tokens': 1234,
        'max_turns': 3,
        'tools': ['commit_data'],
        'skills': [],
    })
    assert configured.status_code == 200

    source = session.workspace / '中国人口数量变化与GDP变化.xlsx'
    pd.DataFrame({
        '年份': [2020, 2021, 2022, 2023, 2024],
        'GDP': [101, 115, 121, 127, 134],
        '人口数量': [1412, 1413, 1412, 1410, 1408],
    }).to_excel(source, index=False)

    class FakeDataAPI:
        tool_description_json = []

        def __init__(self):
            self.turn = 0

        def __call__(self, prompt, **kwargs):
            self.turn += 1

            def generate():
                if self.turn == 1:
                    call = ToolCall('commit_data', {
                        'path': source.name,
                        'target': '人口数量',
                        'features': ['年份', 'GDP'],
                    }, id='commit')
                    message = {'role': 'assistant', 'content': '整理并提交数据。', 'tool_calls': [{
                        'id': 'commit', 'type': 'function', 'function': {
                            'name': call.name, 'arguments': '{}',
                        },
                    }]}
                    yield {'content': message['content'], 'tool_call': [call], 'message': message}
                else:
                    message = {'role': 'assistant', 'content': '数据已经准备好。'}
                    yield {'content': message['content'], 'tool_call': [], 'message': message}
                return {'usage': {'token': {}, 'price': {}}, 'responses': []}

            return APICallResult(generate())

    fake = FakeDataAPI()
    api_options = {}

    def create_api(provider, **kwargs):
        api_options.update(provider=provider, **kwargs)
        return fake

    monkeypatch.setattr(BaseAPI, 'create', create_api)
    response = client.post('/api/data/agent', json={
        'message': '研究人口数量与其他列的关系。',
    })
    assert response.status_code == 200, response.text
    session.data_thread.join(10)
    assert session.data_state == 'completed'
    assert api_options['provider'] == 'deepseek'
    assert api_options['model'] == 'data-preparation-model'
    assert session.data_agent.llm_max_tokens == 1234
    assert session.data_agent.max_turns == 3
    assert session.context.target == '人口数量'
    assert session.context.features == ['年份', 'GDP']
    data_events = session.controller.events()
    data_event_kinds = [event['kind'] for event in data_events]
    assert 'data_user' in data_event_kinds
    assert data_event_kinds.index('data_context') < data_event_kinds.index('data_assistant_start')
    assert data_event_kinds.index('data_assistant_start') < data_event_kinds.index('data_assistant')
    assistant_event = next(event for event in data_events if event['kind'] == 'data_assistant')
    assert assistant_event['payload']['provider'] == 'deepseek'
    assert assistant_event['payload']['model'] == 'data-preparation-model'
    preview = client.get('/api/data/context').json()
    assert preview['revision'] == 1
    assert preview['rows'] == 5
    assert preview['data'][0]['年份'] == 2020


def test_question_reconnect_and_stop():
    controller = InteractionController()
    results = []
    thread = threading.Thread(target=lambda: results.append(controller.ask('Continue?')))
    thread.start()
    for _ in range(100):
        if controller.status()['questions']:
            break
        time.sleep(.01)
    question = next(iter(controller.status()['questions']))
    # Pending questions must survive event buffer eviction/browser reconnect.
    for _ in range(1001):
        controller.publish('test', {})
    controller.reply(question, 'yes')
    thread.join(2)
    assert results == ['yes']
    with pytest.raises(ValueError):
        controller.reply(question, 'duplicate')
    controller.command('message', 'guidance')
    controller.wait_until_running()
    assert controller.checkpoint() == ['guidance']
    controller.command('stop')
    with pytest.raises(KeyboardInterrupt):
        controller.checkpoint()


def test_real_search_loop_with_fake_llm(platform, monkeypatch):
    client, session = platform
    prompts = []
    models = []

    class FakeAPI:
        tool_description_json = []
        def __call__(self, prompt, **kwargs):
            prompts.append(prompt.copy())
            if len(prompts) == 1:
                session.configure({
                    'llm_provider': 'openai',
                    'llm_model': 'test-next-model',
                    'tools': ['evaluate_formula'],
                    'skills': [],
                    'max_refinement_depth': 3,
                })
            def generate():
                call = ToolCall('evaluate_formula', {'f': 'x**2 + 2*x + 1'}, id='test')
                message = {'role': 'assistant', 'content': 'Evaluate a quadratic.', 'reasoning': 'Model reasoning',
                           'tool_calls': [{'id': 'test', 'type': 'function', 'function': {
                               'name': call.name, 'arguments': '{"f":"x**2 + 2*x + 1"}'}}]}
                yield {'content': message['content'], 'tool_call': [call], 'message': message}
                return {'usage': {'token': {}, 'price': {}}, 'responses': []}
            return APICallResult(generate())

    def create(provider, model, **kwargs):
        models.append((provider, model))
        return FakeAPI()

    monkeypatch.setattr(BaseAPI, 'create', create)
    client.put('/api/workspace/upload?path=notes.txt', content=b'research')
    client.post('/api/control/command', json={'action': 'message', 'message': 'Prefer simple formulas'})
    response = client.post('/api/session/start', json={
        'max_refinement_depth': 2, 'problem_description': 'Find the formula',
        'system_prompt': 'Custom system prompt', 'user_prompt': 'Custom user prompt',
    })
    assert response.status_code == 200, response.text
    session.thread.join(20)
    assert not session.thread.is_alive()
    assert session.state == 'completed', session.result
    topk = session.snapshot()['topk_records']
    assert topk
    assert topk[0]['mse'] < 1e-20
    assert client.get('/api/workspace/download?path=notes.txt').content == b'research'
    assert len(prompts) == 3
    assert models[-1] == ('openai', 'test-next-model')
    assert session.settings['llm_model'] == 'test-next-model'
    assert [tool.metadata.name for tool in session.sr_agent.tools] == ['evaluate_formula']
    assert prompts[0][:2] == [
        {'role': 'system', 'content': 'Custom system prompt'},
        {'role': 'user', 'content': 'Custom user prompt'},
    ]
    assert all(any(
        (m.get('content') or '').endswith('Prefer simple formulas') for m in p
    ) for p in prompts)
    events = session.controller.events()
    kinds = [e['kind'] for e in events]
    assert all(k in kinds for k in [
        'context', 'assistant_start', 'assistant', 'tool_start', 'tool_result', 'topk', 'lifecycle',
    ])
    assert kinds.index('context') < kinds.index('assistant_start') < kinds.index('assistant')
    assistant_events = [event for event in events if event['kind'] == 'assistant']
    assert all(event['payload']['provider'] for event in assistant_events)
    assert all(event['payload']['model'] for event in assistant_events)
    runs = client.get('/api/runs').json()['runs']
    assert runs[0]['record_count'] == 3
    assert client.post('/api/session/start', json={}).status_code == 400


def test_advance_to_next_branch_and_restart(platform, monkeypatch):
    client, session = platform
    calls = 0

    class AdvancingAPI:
        tool_description_json = []

        def __call__(self, prompt, **kwargs):
            nonlocal calls
            calls += 1
            session.controller.command('next_c' if calls == 1 else 'next_r')

            def generate():
                message = {'role': 'assistant', 'content': f'round {calls}'}
                yield {'content': message['content'], 'tool_call': [], 'message': message}
                return {'usage': {'token': {}, 'price': {}}, 'responses': []}

            return APICallResult(generate())

    monkeypatch.setattr(BaseAPI, 'create', lambda *args, **kwargs: AdvancingAPI())
    response = client.post('/api/session/start', json={
        'max_restart_loop': 2,
        'global_width': 2,
        'max_refinement_depth': 4,
    })
    assert response.status_code == 200, response.text
    session.thread.join(20)
    assert not session.thread.is_alive()
    assert session.state == 'completed'
    assert calls == 3
    coordinates = [
        event['payload']['coord']
        for event in session.controller.events()
        if event['kind'] == 'context'
    ]
    assert coordinates == [
        {'R': 1, 'C': 1, 'L': 1},
        {'R': 1, 'C': 2, 'L': 1},
        {'R': 2, 'C': 1, 'L': 1},
    ]


def test_csv_input_and_failure_state(platform, monkeypatch):
    client, session = platform
    client.put('/api/workspace/upload?path=measurements.csv',
               content=(b'temperature,humidity,station,pressure\n'
                        b'1,10,a,2\n2,20,b,4\n3,30,c,6\n4,40,d,8\n5,50,e,10\n'))
    seen = {}
    def fail_run(self, X, y, description):
        seen.update(X=X, y=y, description=description)
        raise RuntimeError('test provider unavailable')
    monkeypatch.setattr(SRAgentInteractive, 'run', fail_run)
    response = client.post('/api/session/start', json={
        'dataset': 'measurements.csv', 'target': 'pressure',
        'features': ['humidity'], 'prompt': 'Fit pressure'})
    assert response.status_code == 200
    session.thread.join(10)
    assert list(seen['X']) == ['humidity']
    assert list(seen['y']) == ['pressure']
    assert seen['description'] == 'Fit pressure'
    assert session.state == 'failed'
    assert client.get('/api/session').json()['result']['error'] == 'test provider unavailable'
    assert session.run_state is not None
    assert session.run_state.run_id == session.run_id
    assert session.run_state.node_count == 0


def test_current_tree_never_scans_history(platform):
    client, session = platform
    # Before the first complete iteration, respond promptly with 404.
    assert client.get(f'/api/runs/{session.run_id}/records').status_code == 404
    state = SearchRunState(
        save_path=None,
        ranking_metric='mse',
        larger_is_better=False,
        run_id=session.run_id,
    )
    assert state.latest_coordinate is None
    state.register_iteration(
        [('', [], {'role': 'assistant', 'content': ''})],
        [[]],
        (),
        [],
        {},
        R=1,
        L=1,
        C=1,
    )
    assert state.latest_coordinate == state.nodes[
        state.node_id(R=1, C=1, L=1, K=1)
    ].coordinate
    session.run_state = state
    response = client.get(f'/api/runs/{session.run_id}/records')
    assert response.status_code == 200
    assert response.json()['records'][0]['node_id'] == state.node_id(R=1, C=1, L=1, K=1)
    assert client.get(f'/api/runs/{session.run_id}/records?after_seq=1').json()['records'] == []
    assert client.get('/api/runs/unrelated/records').status_code == 404


def test_model_wait_pause_ack_and_stop(platform, monkeypatch):
    client, session = platform
    entered, release = threading.Event(), threading.Event()
    class WaitingAPI:
        tool_description_json = []
        def __call__(self, prompt, **kwargs):
            def generate():
                entered.set()
                assert release.wait(5)
                yield from []
                return {'usage': {'token': {}, 'price': {}}, 'responses': []}
            return APICallResult(generate())
    monkeypatch.setattr(BaseAPI, 'create', lambda *args, **kwargs: WaitingAPI())
    try:
        client.post('/api/session/start', json={'max_refinement_depth': 2})
        assert entered.wait(5)
        status = client.get('/api/session').json()
        assert status['activity']['phase'] == 'model'
        assert status['activity']['coord']['L'] == 1
        assert status['activity']['model'] == session.settings['llm_model']
        assert status['activity']['since'] <= status['server_time']
        client.post('/api/control/command', json={'action': 'pause'})
        assert not client.get('/api/session').json()['waiting_at_boundary']
        release.set()
        deadline = time.monotonic() + 5
        while not session.controller.status()['waiting_at_boundary'] and time.monotonic() < deadline:
            time.sleep(.01)
        assert client.get('/api/session').json()['waiting_at_boundary']
    finally:
        release.set()
        session.controller.command('stop')
        if session.thread:
            session.thread.join(5)
    assert session.state == 'interrupted'
