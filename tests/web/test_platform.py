import io
import json
import os
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
import sr_harness
from dotenv import dotenv_values

pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient

from sr_harness.agents.data_preparation_agent import DataPreparationAgent
from sr_harness.agents.evaluator_construction_agent import EvaluatorConstructionAgent
from sr_harness.agents.sr_agent_interactive import SRAgentInteractive
from sr_harness.core import APICallResult, SearchRunState, ToolCall, load_context_data
from sr_harness.api import BaseAPI
from sr_harness.web.app import create_app
from sr_harness.web.session import InteractiveSession


class CancellableFakeAPI:
    def cancel(self):
        return None


@pytest.fixture
def platform(tmp_path):
    session = InteractiveSession(tmp_path, agent_options={'tools': ['evaluate_formula', 'workspace_shell']})
    with TestClient(create_app(tmp_path, session=session)) as client:
        yield client, session


def test_workspace_roundtrip_and_boundaries(platform, tmp_path):
    client, session = platform
    page = client.get('/')
    assert page.status_code == 200
    assert 'class="badge"' not in page.text
    assert 'id="search-record-link"' not in page.text
    assert 'id="prepare-tab"' in page.text
    assert '<link rel="icon" href="/static/favicon.svg" type="image/svg+xml">' in page.text
    favicon = client.get('/static/favicon.svg')
    assert favicon.status_code == 200
    assert favicon.headers['content-type'].startswith('image/svg+xml')
    assert b'<svg' in favicon.content
    assert 'id="data-tab"' in page.text
    assert 'id="data-preparation"' in page.text
    assert 'id="data-setup"' in page.text
    assert 'id="run-setup"' in page.text
    assert "connected:'已连接到后端'" in page.text
    assert 'id="pause"' not in page.text
    assert 'id="stop"' not in page.text
    assert 'function renderComposerControl()' in page.text
    assert "button.classList.add('send-icon')" in page.text
    assert "if(state==='idle'){const label=_('startExploring');button.innerHTML='<svg" in page.text
    assert "action:'pause'" in page.text
    assert "action:'force_pause'" in page.text
    assert "const control=await api('/api/control/command',{action:'message',message:prompt})" in page.text
    assert '.composer #send.send-icon{display:grid;place-items:center;width:34px;height:34px' in page.text
    assert "for(const [inputId,buttonId] of [['R','next-c'],['C','next-r']]" in page.text
    assert "button.textContent='+1'" in page.text
    assert '#settings-pane-search .search-step-control{display:flex;align-items:center;gap:6px' in page.text
    assert '#settings-pane-search .search-step-control .advance-button{display:grid;place-items:center;flex:0 0 28px;width:28px;height:28px' in page.text
    assert "composerToolbar.prepend($('settings-toggle'),$('initial-prompt-toggle'))" in page.text
    assert "$('status').className='small muted composer-status-center';$('send').before($('status'),composerKeyboardHint('prompt-keyboard-hint'))" in page.text
    assert "composerEnterHint:'Shift/⌘ + Enter 换行；Enter 发送'" in page.text
    assert "function bindEnterToSend(input,button)" in page.text
    assert "$('data-insights-note')?.remove()" in page.text
    assert "dataInsightsRefresh.id='data-insights-refresh'" in page.text
    assert "dataInsightsRefresh.onclick=dataGuard(reloadCommittedContext)" in page.text
    assert "await api('/api/data/context/reload',{});await loadCommittedContext()" in page.text
    assert "selectCenterTab(document.querySelector('.tabs button.active')?.id.replace(/-tab$/,'')||'prepare')" in page.text
    assert "event.key!=='Enter'||event.shiftKey||event.metaKey||event.isComposing" in page.text
    assert "bindEnterToSend($('prompt'),$('send'))" in page.text
    assert "bindEnterToSend($('data-agent-input'),$('data-agent-send'))" in page.text
    assert "bindEnterToSend($('evaluator-agent-input'),$('evaluator-agent-send'))" in page.text
    assert '.composer-keyboard-hint{flex:0 0 auto;color:var(--muted);font-size:9px;white-space:nowrap}' in page.text
    assert '.composer-status-center{position:absolute!important;top:50%;left:50%' in page.text
    assert '.composer>.row #status{padding:0;border-radius:0;background:transparent;color:var(--muted);font-size:11px' in page.text
    assert 'id="plot-variable-palette"' in page.text
    assert 'id="relationship-preview-mode"' in page.text
    assert "halfAngle=(tip.role==='target'?4:12)*Math.PI/360" in page.text
    assert "function curveHasClearance(curveIndex,scale,scales)" in page.text
    assert "clearance=7,samples=28" in page.text
    assert "curveOrder=[sourceSource,...tips.map((_,index)=>index).filter(index=>index!==sourceSource)]" in page.text
    assert "for(const curveIndex of curveOrder)" in page.text
    assert "class:'relation-hyperedge'" in page.text
    assert "label:'T'" in page.text and "label:'S1'" in page.text and "label:'S2'" in page.text
    assert "viewport.classList.add('hyperedge-active');group.classList.add('active')" in page.text
    assert '.relation-viewport.hyperedge-active .relation-hyperedge:not(.active){opacity:.1}' in page.text
    assert '<option value="table">表格</option>' in page.text
    assert 'id="data-summary"' not in page.text
    assert "for(const column of dataPreview.columns)" in page.text
    assert "pill.ondragend=()=>{if(!plotDragAccepted" in page.text
    assert "function loadPlotPreview()" in page.text
    assert "id=\"plot-broadcast-info\"" in page.text
    assert 'id="start-prepared"' not in page.text
    data_view = page.text[page.text.index('id="data-setup"'):page.text.index('id="run-setup"')]
    run_setup = page.text[page.text.index('id="run-setup"'):page.text.index('id="feed"')]
    data_insights_start = page.text.index('id="data-insights"')
    data_insights = page.text[
        data_insights_start:page.text.index('</aside>', data_insights_start)
    ]
    assert '>符号回归</button>' in page.text
    assert 'id="data-preview-card"' not in data_view
    assert 'class="data-card wide relationship-card"' not in data_view
    assert 'id="variable-config-card"' in data_view
    assert 'id="task-problem-card"' in data_view
    assert "card.id='evaluator-config-card'" in page.text
    assert "content:'3'" in page.text
    assert 'class="evaluator-config-heading step-heading"' in page.text
    assert ".task-problem-heading,.evaluator-config-heading{margin-bottom:10px}" in page.text
    assert "evaluatorHelp:'选择内置评估器或自定义新的评估器，实现数据切分、模型拟合、模型评估'" in page.text
    assert 'class="evaluator-editor"' in page.text
    assert '.evaluator-editor{margin-top:10px;overflow:hidden;border:1px solid #d8dee4;border-radius:25px;background:#fff;box-shadow:var(--interactive-shadow)}' in page.text
    assert 'id="evaluator-code"' in page.text
    assert 'id="evaluator-agent-input"' in page.text
    assert "evaluatorToggle.id='evaluator-agent-toggle'" in page.text
    assert 'evaluatorAgent.hidden=true' in page.text
    assert "evaluatorAgentToggle:'也可以描述需求，让 Agent 创建 Evaluator'" in page.text
    assert 'id="evaluator-agent-feed" class="data-agent-feed evaluator-agent-feed" hidden' in page.text
    assert 'class="data-agent-compose evaluator-agent-compose"' in page.text
    assert 'class="data-agent-compose-toolbar evaluator-agent-compose-toolbar"' in page.text
    assert 'class="data-agent-send evaluator-agent-send"' in page.text
    assert 'id="evaluator-agent-settings-toggle" class="composer-settings-button"' in page.text
    assert 'id="evaluator-agent-settings" class="settings-panel data-agent-settings evaluator-agent-settings"' in page.text
    assert "guide.href='/evaluator-guide'" in page.text
    assert "data-evaluator-agent-settings-tab=\"capabilities\"" in page.text
    assert 'id="evaluator-agent-tool-options"' in page.text
    assert 'id="evaluator-agent-skill-options"' in page.text
    assert "button.innerHTML='<svg viewBox=\"0 0 20 20\"" in page.text
    assert "argsToggle.id='evaluator-args-toggle'" in page.text
    assert "'split_ood_variable','evaluatorSplitOodVariable'" in page.text
    assert "control.id='evaluator-arg-'+name.replaceAll('_','-')" in page.text
    assert "api('/api/session/settings',evaluatorArgsPayload())" in page.text
    assert '.evaluator-args-panel{container-type:inline-size' in page.text
    assert '.evaluator-args-grid{display:grid;grid-template-columns:1fr' in page.text
    assert '@container (min-width:590px){.evaluator-args-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}' in page.text
    assert '.evaluator-arg-copy b{display:block;color:var(--ink);font:600 10px/1.4' in page.text
    assert '.evaluator-arg-copy small{display:block;margin-top:2px;color:var(--muted)' in page.text
    assert '#evaluator-args-toggle{height:30px;padding:4px 8px;border:0;border-radius:7px;background:transparent' in page.text
    assert '#evaluator-args-toggle[aria-expanded="true"]{border:0;background:#eaf3f0;color:var(--accent)}' in page.text
    assert "api('/api/evaluator/agent/settings',evaluatorAgentSettingsPayload(),'PUT')" in page.text
    assert "api('/api/evaluator/agent/test',evaluatorAgentSettingsPayload())" in page.text
    assert "if(e.kind==='evaluator_user'){appendEvaluatorAgentEvent('user',p,e.timestamp);return}" in page.text
    assert 'function applyEvaluatorAgentResult()' in page.text
    assert "toolCallsBlock(card,payload.tool_calls)" in page.text
    assert 'evaluatorCustomSource' not in page.text
    assert "evaluatorConfiguration.evaluators.map(item=>{const option=el('option'" in page.text
    assert "Object.assign(el('option',_('evaluatorCustom'))" not in page.text
    assert "evaluatorArgsTitle:'context.args.*'" in page.text
    assert "copy.append(el('b',name),el('small',_(helpKey)))" in page.text
    assert 'class="evaluator-agent-heading step-heading"' in page.text
    assert ".evaluator-agent>.step-heading::before{content:'4'}" in page.text
    assert '.evaluator-editor-toolbar button{display:inline-flex;align-items:center;justify-content:center;height:34px;padding:0 14px' in page.text
    assert "api('/api/evaluator/test',evaluatorPayload())" in page.text
    assert "api('/api/evaluator/agent/start',{message,source:$('evaluator-code').value})" in page.text
    assert "api('/api/evaluator/agent/stop',{})" in page.text
    assert "['pausing','interrupting'].includes(agentState)" in page.text
    assert "button.classList.toggle('stop',active)" in page.text
    assert "if(evaluatorDirty)await saveEvaluatorConfiguration()" in page.text
    assert "const expandedDirectories=new Set(),shownSessionErrors=new Set()" in page.text
    assert "if(!shownSessionErrors.has(errorKey)){shownSessionErrors.add(errorKey);notice(sessionError)}" in page.text
    assert "shownSessionErrors.clear()" in page.text
    assert '<h3>配置变量描述</h3>' in data_view
    assert '点击颜色条切换变量角色，点击变量描述以编辑，也可拖动手柄调整变量顺序' in data_view
    assert '<h3 id="task-problem-title">配置问题描述</h3>' in data_view
    assert '说明研究目标作为初始提示词。变量描述中已有的内容无需重复。' in data_view
    assert '#data-setup>.data-layout{padding:0}' in page.text
    assert '#variable-config-card,#task-problem-card{padding:0;border:0;border-radius:0;background:transparent' in page.text
    assert '.data-agent-card{padding:0;border:0;background:transparent}' in page.text
    assert '.data-agent-compose{margin-bottom:20px}' in page.text
    assert '.data-agent-settings>.settings-heading{display:none}' in page.text
    assert '#data-agent-settings-title,#data-agent-settings-subtitle,#data-api-key-env{display:none}' in page.text
    assert '.data-agent-settings .settings-form label:has(#data-proxy),.data-agent-settings .settings-form .api-key-setting{grid-column:1;grid-template-columns:minmax(105px,.9fr) minmax(0,1.2fr)}' in page.text
    assert '.data-agent-settings .settings-scroll{container-type:inline-size}' in page.text
    assert '.data-agent-settings .settings-form{grid-template-columns:1fr}' in page.text
    assert '@container (min-width:590px){.data-agent-settings .settings-form{grid-template-columns:repeat(2,minmax(0,1fr))}}' in page.text
    assert '.data-agent-settings .settings-form label>span:first-child,.data-agent-settings .settings-form .api-key-setting>span:first-child b{white-space:nowrap}' in page.text
    assert '.data-agent-settings .capability-columns section:has(>#data-tool-options){container:data-tools/inline-size}' in page.text
    assert '#data-tool-options{grid-template-columns:1fr}' in page.text
    assert '@container data-tools (min-width:360px){#data-tool-options{grid-template-columns:repeat(2,minmax(0,1fr))}}' in page.text
    assert '.data-agent-settings .settings-form label:has(#data-proxy),.data-agent-settings .settings-form .api-key-setting{grid-column:auto}' in page.text
    assert '.data-agent-compose-toolbar{padding-right:7px;padding-bottom:7px}' in page.text
    assert 'button.composer-settings-button:is(#initial-prompt-toggle,#data-agent-settings-toggle,#settings-toggle){border:0;border-radius:7px;background:transparent;color:#647571;font-size:13px}' in page.text
    assert 'button.composer-settings-button:is(#initial-prompt-toggle,#data-agent-settings-toggle,#settings-toggle)[aria-expanded="true"]{border:0;border-radius:7px;background:#eaf3f0;color:var(--accent)}' in page.text
    assert ':root[data-theme="dark"] button.composer-settings-button:is(#initial-prompt-toggle,#data-agent-settings-toggle,#settings-toggle){border:0;border-radius:7px;background:transparent;color:#aebbb8}' in page.text
    assert ':root[data-theme="dark"] button.composer-settings-button:is(#initial-prompt-toggle,#data-agent-settings-toggle,#settings-toggle):hover{background:#22302d;color:#d2dfdc}' in page.text
    assert ':root[data-theme="dark"] button.composer-settings-button:is(#initial-prompt-toggle,#data-agent-settings-toggle,#settings-toggle)[aria-expanded="true"]{border:0;background:#29413c;color:#74c7bb}' in page.text
    assert ':root[data-theme="dark"] #initial-prompt-toggle[aria-expanded="true"]{background:#29413c!important;color:#74c7bb}' in page.text
    assert '.composer>#initial-prompt-panel{border-color:#9fc5be;border-radius:17px;background:#f3f9f7}' in page.text
    assert ':root[data-theme="dark"] .composer>#initial-prompt-panel{border-color:#4d8d83;background:#172522}' in page.text
    assert '.initial-prompt-body #system-prompt{border:1px solid #d8e5e2;border-radius:4px;background:#fff;box-shadow:0 1px 2px #263b3808;color:var(--ink)}' in page.text
    assert '.initial-prompt-body #system-prompt:focus{border-color:#a8c8c2;background:#fff;color:var(--ink)}' in page.text
    assert ':root[data-theme="dark"] .initial-prompt-body #system-prompt{border-color:#3d514d;background:#202b29;color:var(--ink)}' in page.text
    assert '.initial-prompt-panel>.settings-heading{display:none}' in page.text
    assert '.initial-prompt-body{padding:12px}' in page.text
    assert ':root[data-theme="dark"] .data-agent-send{background:var(--accent);color:#fff}' in page.text
    assert ':root[data-theme="dark"] .data-agent-send:hover{background:#14665f}' in page.text
    assert ':root[data-theme="dark"] .data-agent-send.stop{background:var(--danger)}' in page.text
    assert ':root[data-theme="dark"] .data-agent-send:disabled{background:#aebbb8}' in page.text
    assert '.data-agent-settings{margin-top:2px;border-color:#9fc5be;border-radius:17px;background:#f3f9f7;box-shadow:none}' in page.text
    assert '.data-agent-settings .settings-scroll{padding-bottom:5px}' in page.text
    assert '.data-agent-settings .settings-actions{padding-top:5px;border-top:0;border-radius:0 0 17px 17px}' in page.text
    assert '.data-agent-settings .settings-form label,.data-agent-settings .api-key-setting>span:first-child b,#data-settings-hint{color:#4f625e}' in page.text
    assert '.data-agent-settings .settings-form input,.data-agent-settings .settings-form select{border-color:#d8e5e2;background:#fff;box-shadow:0 1px 2px #263b3808}' in page.text
    assert '.data-agent-settings .api-key-control button{border-color:#d8e5e2;background:#fff}' in page.text
    assert '.data-agent-settings .api-key-control{position:relative;display:block}' in page.text
    assert '.data-agent-settings .settings-form .api-key-control input{width:100%;padding-right:36px;border-radius:4px}' in page.text
    assert '.data-agent-settings .api-key-control #data-api-key-visibility{position:absolute;top:50%;right:4px' in page.text
    assert "function renderDataApiKeyVisibility()" in page.text
    assert "button.setAttribute('aria-pressed',String(visible))" in page.text
    assert ':root[data-theme="dark"] .data-agent-settings .settings-form label,:root[data-theme="dark"] .data-agent-settings .api-key-setting>span:first-child b,:root[data-theme="dark"] #data-settings-hint{color:#b8c6c3}' in page.text
    assert '#data-model-test,#data-settings-cancel,#data-settings-apply{flex:0 0 auto;height:34px;border-radius:17px;white-space:nowrap}' in page.text
    assert "$('data-model-test').before($('data-settings-hint'))" in page.text
    assert "$('data-agent-state').classList.add('composer-status-center');$('data-agent-send').before($('data-agent-state'),composerKeyboardHint('data-agent-keyboard-hint'))" in page.text
    assert "$('data-api-key-status').remove()" in page.text
    assert "apiKeyConfigured:'已配置，输入新值以替换'" in page.text
    assert "input.placeholder=credential.configured?_('apiKeyConfigured')" in page.text
    assert "openrouter:'https://openrouter.ai/settings/keys'" in page.text
    assert "apiKeyGet:'获取 {{env}} ↗'" in page.text
    assert "link.textContent=_('apiKeyGet',{env:envVar})" in page.text
    assert "$('data-api-key-label').replaceChildren('API Key (',link,')')" in page.text
    assert '.composer>#settings-panel{margin-top:2px;border-color:#9fc5be;border-radius:17px;background:#f3f9f7;box-shadow:none}' in page.text
    assert '#settings-panel>.settings-heading,#api-key-env{display:none}' in page.text
    assert '#settings-panel .settings-scroll{container-type:inline-size;padding-bottom:5px}' in page.text
    assert '#settings-panel .settings-actions{padding-top:5px;border-top:0;border-radius:0 0 17px 17px}' in page.text
    assert '#settings-panel .settings-form,#settings-panel .settings-form.compact{grid-template-columns:1fr}' in page.text
    assert '@container (min-width:590px){#settings-panel .settings-form,#settings-panel .settings-form.compact{grid-template-columns:repeat(2,minmax(0,1fr))}}' in page.text
    assert '#settings-panel .settings-form label>span:first-child,#settings-panel .api-key-setting>span:first-child b{white-space:nowrap}' in page.text
    assert '#settings-panel .settings-form input,#settings-panel .settings-form select{border-color:#d8e5e2;background:#fff;box-shadow:0 1px 2px #263b3808}' in page.text
    assert '#settings-panel .api-key-control #api-key-visibility{position:absolute;top:50%;right:4px' in page.text
    assert '#settings-cancel,#apply{flex:0 0 auto;height:34px;border-radius:17px;white-space:nowrap}' in page.text
    assert '#model-test{flex:0 0 auto;height:34px;border-radius:17px;white-space:nowrap}' in page.text
    assert '@container run-tools (min-width:360px){#tool-options{grid-template-columns:repeat(2,minmax(0,1fr))}}' in page.text
    assert "function renderApiKeyVisibility()" in page.text
    assert "$('settings-cancel').before($('settings-hint'));$('api-key-status').remove()" in page.text
    assert "modelTestButton.id='model-test'" in page.text
    assert "$('settings-cancel').before(modelTestStatus,modelTestButton)" in page.text
    assert "api('/api/session/test',settingsPayload())" in page.text
    assert "$('api-key-label').replaceChildren('API Key (',link,')')" in page.text
    assert '#settings-panel .settings-form .api-key-setting{grid-column:auto;grid-template-columns:minmax(105px,.9fr) minmax(0,1.2fr)}' in page.text
    assert '#api-key-label a{color:var(--accent);text-decoration:underline;text-underline-offset:2px}' in page.text
    assert '#settings-pane-model,#settings-pane-search{display:grid;grid-template-columns:1fr;column-gap:18px;row-gap:5px}' in page.text
    assert '#settings-pane-model>.settings-form,#settings-pane-search>.settings-form,#settings-pane-search>.switch-list{display:contents}' in page.text
    assert '@container (min-width:590px){#settings-pane-model,#settings-pane-search{grid-template-columns:repeat(2,minmax(0,1fr))}}' in page.text
    assert "restartCount:'R · Restart Count'" in page.text
    assert "branchCount:'C · Independent Branches'" in page.text
    assert "conversationDepth:'L · Conversation Depth'" in page.text
    assert "samplesPerRound:'K · Samples per Round'" in page.text
    assert "function renderSearchSettingLabels()" in page.text
    assert "renderApiKeyVisibility();renderSearchSettingLabels()" in page.text
    assert 'id="documentation-link"' in page.text
    assert 'href="http://sim1.fiblab.tech:11005/"' in page.text
    assert "documentation:'Open documentation'" in page.text
    assert "function compactEventContent(e)" in page.text
    assert "eventContent(card,meta,metaTime,compactEventContent(e))" in page.text
    assert "s.interaction_state==='paused'" in page.text
    for removed_id in (
        'validation-fraction', 'split-by', 'split-seed', 'ranking-metric',
        'larger-better',
    ):
        assert f'id="{removed_id}"' not in page.text
    assert '#composer>#prompt{padding:14px 15px 7px}' in page.text
    assert '.composer #send.stop-pending{border-color:var(--danger);background:var(--danger);color:#fff}' in page.text
    assert '#composer>.row>#send.primary.send-icon.stop-pending{display:grid;visibility:visible;border-color:var(--danger);background:var(--danger);color:#fff;opacity:1}' in page.text
    assert "promptResizeHandle.className='prompt-resize-handle'" in page.text
    assert "promptResizeHandle.style.top=promptInput.offsetTop+7+'px'" in page.text
    assert 'resizePromptInput(startHeight-(moveEvent.clientY-startY))' in page.text
    assert ".data-ingest-intro::before{content:'1'}" in page.text
    assert '.data-ingest-intro{display:flex;align-items:flex-start;gap:10px}' in page.text
    assert '.data-ingest-intro::before{flex:0 0 24px;grid-row:auto}' in page.text
    assert '.data-ingest-intro-copy{display:flex;min-width:0;flex:1;flex-direction:column}' in page.text
    assert "dataIngestIntroCopy.className='data-ingest-intro-copy'" in page.text
    assert '.step-heading{display:flex!important;align-items:flex-start;gap:10px}' in page.text
    assert '.step-heading-copy{display:flex;min-width:0;flex:1;flex-direction:column}' in page.text
    assert ".data-agent-heading>div.step-heading::before{content:'2';grid-row:auto}" in page.text
    assert "#variable-config-card .variable-card-heading>div.step-heading::before{content:'1'}" in page.text
    assert "#task-problem-card .task-problem-heading::before{content:'2'}" in page.text
    assert '#problem-description,#variable-role-table,#variable-role-body{border-radius:25px}' in page.text
    assert '.task-problem-card textarea{min-height:50px}' in page.text
    assert '#variable-role-table{overflow:hidden}' in page.text
    assert '#variable-role-table>.variable-table-head{padding:8px 10px 5px;background:var(--interactive-surface)}' in page.text
    assert '#variable-role-table .variable-row:first-child{border-radius:0}' in page.text
    assert "$('variable-role-table').prepend(variableTableHead)" in page.text
    assert '#variable-role-body .variable-description{min-height:30px;max-height:none;resize:none;overflow:hidden;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere}' in page.text
    assert "const description=document.createElement('textarea')" in page.text
    assert "function resizeVariableDescription(description)" in page.text
    assert 'let variableTableWidth=-1;new ResizeObserver' in page.text
    assert '#variable-role-table{container-type:inline-size}' in page.text
    assert '#variable-role-body .variable-description{min-width:0;max-width:100%}' in page.text
    assert '#variable-role-table>.variable-table-head span{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}' in page.text
    assert '@container (max-width:400px){#variable-role-table>.variable-table-head,#variable-role-body>.variable-row{grid-template-columns:18px 28px minmax(48px,.4fr) minmax(0,1fr);gap:4px;padding-right:6px;padding-left:6px}' in page.text
    assert '#variable-role-table>.variable-table-head,#variable-role-body>.variable-row{padding-left:2px}' in page.text
    assert '#variable-role-table .variable-handle{margin-left:0;justify-self:start}' in page.text
    assert 'function structureStepHeading(host)' in page.text
    assert "taskProblemHeading.className='task-problem-heading step-heading'" in page.text
    assert '#variable-config-card .variable-card-heading .step-heading-copy{display:grid;grid-template-columns:minmax(0,1fr) auto;column-gap:8px}' in page.text
    assert '#variable-config-card .variable-card-heading .step-heading-copy>.variable-help{grid-column:1/-1}' in page.text
    assert '#variable-config-card .step-heading-copy>.data-refresh-button{align-self:start;margin:0;border:1px solid transparent;border-radius:999px}' in page.text
    assert "variableHeadingCopy.querySelector('h3').after($('data-refresh'))" in page.text
    assert 'variableHeadingActions.remove()' in page.text
    assert '#run-variable-preview .variable-role-table{overflow:hidden}' in page.text
    assert '#run-variable-preview .variable-table-head,#run-variable-preview .variable-preview-row{grid-template-columns:minmax(100px,.32fr) minmax(0,1fr)}' in page.text
    assert '#run-variable-preview .variable-preview-name{width:auto;padding:0;border-radius:0;background:transparent!important;color:var(--ink);font:700 10px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}' in page.text
    assert "if(role==='target')pill.append(el('span',roleLabels.target,'column-kind target-kind'))" in page.text
    assert 'runVariableTable.prepend(runVariableTableHead)' in page.text
    assert "head.closest('#run-variable-preview')" in page.text
    assert '#run-variable-preview .variable-description-preview{overflow:visible;text-overflow:clip;white-space:normal;overflow-wrap:anywhere;line-height:1.5}' in page.text
    assert '.center:has(>#composer:not(.data-mode)){background:#fbfcfd}' in page.text
    assert '.center:has(>#composer:not(.data-mode))>.bar{background:#fff}' in page.text
    assert ':root[data-theme="dark"] .center:has(>#composer:not(.data-mode)){background:#111817}' in page.text
    assert '#data-ingest,.data-agent-compose,#variable-role-table,#problem-description,#composer{box-shadow:var(--interactive-shadow)}' in page.text
    assert '.data-agent-compose:focus-within,#composer:focus-within{box-shadow:var(--interactive-shadow)}' in page.text
    assert ".data-agent-heading>div:first-child::before{content:'2'}" in page.text
    assert '#variable-config-card .variable-card-heading h3,#task-problem-title{display:flex;align-items:center;gap:10px;min-height:24px}' in page.text
    assert '#variable-config-card .variable-card-heading h3::before{content:\'1\'}' in page.text
    assert '#task-problem-title::before{content:\'2\'}' in page.text
    assert '.data-agent-feed{border:0;border-radius:0;background:#fff;box-shadow:0 -24px 32px -16px #fff,0 24px 32px -16px #fff}' in page.text
    assert '#data-ingest{height:50px;padding:7px 7px 7px 14px;border:1px solid #dce4e2;border-radius:25px;background:var(--interactive-surface)' in page.text
    assert '#data-ingest .data-ingest-prompt{min-width:0;flex:1;color:var(--muted);font-size:12px' in page.text
    assert '#data-ingest #data-upload{flex:0 0 auto;height:34px;margin-left:auto;padding:0 14px;border:0;border-radius:17px;background:var(--accent);color:#fff}' in page.text
    assert "dataDropCopy:'拖入文件以上传'" in page.text
    assert "MESSAGE_CATALOGS.zh.dataDropCopy" not in page.text
    assert "Object.assign(MESSAGE_CATALOGS['zh-CN']" in page.text
    assert "Object.assign(MESSAGE_CATALOGS.zh," not in page.text
    assert "uploadDataHelpPrefix:'上传数据文件，也可以从',uploadDataHelpSuffix:'快速开始'" in page.text
    assert "dataAgentHelp:'用自然语言指导 Agent 整理、清洗、补充或检查数据，结果将被保存在 context.data/ 目录以供使用'" in page.text
    assert "help.textContent=_('dataAgentHelp');link.textContent=_('dataAgentSafety')" in page.text
    assert "dataAgentSafety:'它是否会危害我的系统和数据？'" in page.text
    assert 'id="evaluator-agent-safety" class="data-agent-safety"' in page.text
    assert '.evaluator-editor-toolbar,.evaluator-line-numbers,.evaluator-editor-status{background:#fff}' in page.text
    assert '.evaluator-editor-toolbar #evaluator-select{min-width:220px;height:34px' in page.text
    assert '.evaluator-code{min-height:160px}' in page.text
    assert 'option.evaluator-option-invalid{color:var(--danger);font-weight:700}' in page.text
    assert "(item.invalid?'⚠ ':'')" in page.text
    assert "status.textContent=item?.invalid?'⚠ '+item.error" in page.text
    assert "$('evaluator-test').disabled=locked||abstract||invalid" in page.text
    assert "argsToggle.innerHTML='<svg" in page.text
    assert 'id="evaluator-args-toggle-label"' in page.text
    assert "evaluatorArgs:'参数配置'" in page.text
    assert "evaluatorTest:'测试'" in page.text
    assert "dataIngest.before(dataIngestIntro)" in page.text
    assert "dataIngest.append(dataDropCopy,$('data-upload'),$('data-file-input'))" in page.text
    assert "dataIngest.ondrop=dataGuard(" in page.text
    assert 'id="problem-description"' in data_view
    assert 'id="data-refresh"' in data_view
    assert 'id="save-data-selection"' not in data_view
    assert 'id="data-preview-card"' not in data_insights
    assert 'class="data-card wide relationship-card"' not in data_insights
    assert 'id="data-table-preview"' in data_insights
    assert 'data-preview-scroll' not in data_insights
    assert 'data-insights-resizer' not in data_insights
    assert "[['row','ROW'],['column','COL'],['color','MAP'],['z','Z']]" in page.text
    assert '#workspace{border-right:0}#insights-panel{border-left:0}' in page.text
    assert '#workspace>.bar{background:#fff}' in page.text
    assert '#workspace>.bar,.center>.bar{flex:0 0 57px;height:57px;min-height:57px}' in page.text
    assert '.center,#data-setup,#run-setup{background:#f6f7f9}#composer{background:#fff}' in page.text
    assert '#run-setup,#run-variable-preview .variable-role-table{background:transparent}' in page.text
    assert '.center,.center:has(>#composer:not(.data-mode)){background:#f6f7f9}' in page.text
    assert '.center>.bar{background:#fff}' in page.text
    assert '#data-insights{background:#fafbfc}#data-insights>.data-insights-scroll{background:transparent}#data-insights>.bar,#data-insights .relationship-preview-bar{background:#fff}' in page.text
    assert '#search-insights>.bar,#search-insights>.candidate-bar,#data-insights>.bar,#data-insights .relationship-preview-bar{height:57px;min-height:57px;flex:0 0 57px}' in page.text
    assert '.search-insights>.tree-wrap,.search-insights>#topk,.data-insights-scroll{background:#fafbfc}' in page.text
    assert 'id="search-insights"' in page.text
    assert 'id="run-variable-preview"' in run_setup
    assert 'class="problem-card"' not in run_setup
    assert '<input' not in run_setup
    assert '<button' not in run_setup
    assert 'id="initial-prompt-panel"' in page.text
    assert '<details id="initial-prompt-panel"' not in page.text
    assert 'id="initial-prompt-toggle"' in page.text
    assert 'data-initial-prompt-tab=' not in page.text
    assert 'id="user-prompt"' not in page.text
    assert 'id="system-prompt" class="prompt-editor" aria-label="System Prompt"' in page.text
    assert 'aria-label="System Prompt" readonly' not in page.text
    assert 'id="system-prompt-confirm-title">确认系统提示词</b>' in page.text
    assert "systemPromptConfirmHelp:'内置默认提示词，可在发送前编辑。'" in page.text
    assert "composerPurposeHelp:'根据任务配置自动生成，可在发送前直接编辑。'" in page.text
    assert "composerPurposeTitle:'确认用户提示词'" in page.text
    assert "promptRegenerateButton('composer-purpose-regenerate')" in page.text
    assert "promptRegenerateButton('system-prompt-regenerate')" in page.text
    assert "await refreshPromptPreview(kind)" in page.text
    assert '<span id="initial-prompt-toggle-label">系统提示词</span>' in page.text
    assert (
        "Find an interpretable formula explaining the selected target from the "
        "selected features."
    ) in data_view
    assert 'id="composer-purpose"' in page.text
    assert "promptEdited.user=true" in page.text
    assert "promptEdited.system=true" in page.text
    assert "if(!promptEdited.user)schedulePromptPreview()" not in page.text
    assert "if(!promptEdited.system)schedulePromptPreview()" not in page.text
    assert "system_prompt:$('system-prompt').value.trim(),user_prompt:userPrompt" in page.text
    assert "if(e.kind==='user')" in page.text
    assert "showTabHint(_('runStartedHint'))" not in page.text
    assert "syncResearchProblem($('problem-description').value)" in page.text
    assert 'function renderTimelineSurface()' in page.text
    assert "$('composer-purpose').hidden=active" in page.text
    assert "$('composer-purpose-title').textContent=_('composerPurposeTitle')" in page.text
    assert 'id="timeline-view-switch"' not in page.text
    assert 'function renderVariableRolePreview()' in page.text
    assert "variableOrder.filter(column=>variableRole(column)!=='unused')" in page.text
    assert "api('/api/data/selection'" in page.text
    assert 'function scheduleDataSelectionSave()' in page.text
    assert "api('/api/data/descriptions',{variable_descriptions:{...variableDescriptions}},'PUT')" in page.text
    assert 'setTimeout(flushDataSelectionSave,400)' in page.text
    assert 'id="data-refresh"' in page.text
    assert 'id="data-file"' not in page.text
    assert 'id="data-max-turns"' not in page.text
    assert "api('/api/data/context')" in page.text
    assert 'async function responseError(response' in page.text
    assert 'function compactStreamEventBatch(events)' in page.text
    assert "if((searchEvents.truncated&&seq)||(dataEvents.truncated&&dataSeq)||(evaluatorEvents.truncated&&evaluatorSeq))notice(_('eventBufferGap'))" in page.text
    assert "events.events[0].seq>seq+1" not in page.text
    assert "if(e.kind==='evaluator_context')" in page.text
    assert 'function renderDataContextEventCard(e,feed)' in page.text
    assert 'meta.append(dataContextLink(p.turn))' in page.text
    assert 'meta.append(el(\'span\',contextPayloadStats(p),\'context-size\'),time)' in page.text
    assert "contextScope:'evaluator'" in page.text
    assert 'function fitProblemDescription()' in page.text
    assert "requestAnimationFrame(fitProblemDescription)" in page.text
    assert "card.dataset.started=e.timestamp" in page.text
    assert "find(item=>Number(item.dataset.started)>Number(e.timestamp))" in page.text
    assert "statusStarted:performance.now()/1000" in page.text
    assert "state.statusStarted=performance.now()/1000" in page.text
    assert "streamProgress(state.statusStarted,state.status)" in page.text
    assert "const now=performance.now()/1000" in page.text
    assert "serverClockOffsetSeconds" not in page.text
    assert "if(e.kind==='prompt_added'){renderPromptAdded(e,$('feed'));return}" in page.text
    assert "if(e.kind==='prompt_added'){renderPromptAdded(e,$('data-agent-feed'),'data');return}" in page.text
    assert "if(e.kind==='prompt_added'){renderPromptAdded(e,$('evaluator-agent-feed'),'evaluator');return}" in page.text
    assert "if(e.kind==='evaluator_user')" in page.text
    assert '.data-agent-feed .event-kind:is(:hover,:focus)::after' in page.text
    assert "await responseError(response,_('uploadFailed'))" in page.text
    assert 'id="workspace-name-editor"' in page.text
    assert 'id="data-context-guide"' not in page.text
    assert (
        'manifest.json 记录变量与轴信息；网络或超图数据还会记录 num_nodes 和关系变量。'
        '&lt;name&gt;.npy 保存取值'
    ) in page.text
    assert 'id="data-agent-safety"' in page.text
    assert "api('/api/workspace/lock',{path,locked},'PUT')" in page.text
    safety = client.get('/data-agent-safety')
    assert safety.status_code == 200
    assert 'workspace_shell' in safety.text
    assert 'workspace_code_executor' in safety.text
    assert '操作系统级只读 bind mount' in safety.text
    assert 'id="context-data-guide-link"' in page.text
    assert 'href="/context-data-guide"' in page.text
    context_data_guide = client.get('/context-data-guide')
    assert context_data_guide.status_code == 200
    assert 'SRHarness · context.data 数据规范' in context_data_guide.text
    assert '硬约束' in context_data_guide.text
    assert '语义约定' in context_data_guide.text
    assert '不要泄露待发现的真实公式' in context_data_guide.text
    evaluator_guide = client.get('/evaluator-guide')
    assert evaluator_guide.status_code == 200
    assert 'Evaluator 的接口' in evaluator_guide.text
    assert 'evaluator.split(context)' in evaluator_guide.text
    assert 'evaluator.fit(f, y, train_context)' in evaluator_guide.text
    assert 'evaluator.evaluate' in evaluator_guide.text
    assert '<button id="context-tab" type="button" hidden>' in page.text
    assert "$('context-tab').hidden=tab!=='context'" in page.text
    assert '#context-tab{display:flex' in page.text
    assert 'function dataContextLink(turn)' in page.text
    assert "agent_scope:'data'" in page.text
    assert "/api/data/agent/stop" in page.text
    assert '.data-agent-send.stop' in page.text
    assert '.data-agent-card{display:grid;grid-template-columns:minmax(0,1fr)' in page.text
    assert '.reasoning-block.expanded .reasoning-content{display:block;width:100%' in page.text
    assert "previewBlock(card,p.content,'message-preview-block')" in page.text
    assert "if(dataMode)previewBlock(card,state.content,'message-preview-block stream-content')" in page.text
    assert "state.reasoningExpanded=reasoning.classList.contains('expanded')" in page.text
    assert "state.contentExpanded=content.classList.contains('expanded')" in page.text
    assert "const restoreExpansion=()=>" in page.text
    assert 'function unobservePreviewBlocks(parent)' in page.text
    assert 'id="up"' not in page.text
    assert 'id="path"' not in page.text
    assert 'function applyPromptPreview(preview,systemOnly=false)' in page.text
    assert 'const hasPreparedData=Boolean(session?.supplied_data||useCommittedContext)' in page.text
    assert 'applyPromptPreview(preview,!hasPreparedData)' in page.text
    assert "if(s.state==='idle')return refreshPromptPreview()" in page.text
    assert client.get('/viewer').status_code == 200
    initial_prompts = client.post('/api/data/prompts', json={})
    assert initial_prompts.status_code == 200
    assert 'Symbolic Regression Agent' in initial_prompts.json()['system_prompt']
    assert client.put('/api/workspace/upload?path=data/sample.csv', content=b'x,y\n1,2').status_code == 200
    assert client.get('/api/workspace/download?path=data/sample.csv').content == b'x,y\n1,2'
    assert client.get('/api/workspace?path=data').json()['entries'][0]['name'] == 'sample.csv'
    tree = client.get('/api/workspace', params={'recursive': True}).json()['entries']
    assert {entry['name'] for entry in tree} >= {'data'}
    assert 'context.evaluator' not in {entry['name'] for entry in tree}
    data_entry = next(entry for entry in tree if entry['name'] == 'data')
    assert data_entry['read_only'] is False
    assert data_entry['size'] is None
    assert data_entry['children'][0]['path'] == 'data/sample.csv'
    assert data_entry['children'][0]['read_only'] is False
    assert client.get(
        '/api/workspace/size', params={'path': 'data'},
    ).json()['size'] == len(b'x,y\n1,2')
    demo = client.post('/api/data/demo').json()
    assert demo['path'] == 'context.data'
    assert (session.workspace / 'context.data' / 'manifest.json').is_file()
    preview = client.get('/api/data/context', params={'rows': 5}).json()
    assert preview['columns'] == ['sample', 'x1', 'x2', 'y']
    assert preview['column_kinds'] == {
        'sample': 'axis', 'x1': 'variable', 'x2': 'variable',
        'y': 'variable',
    }
    assert len(preview['data']) == 5
    assert preview['truncated']
    assert client.post('/api/data/demo').status_code == 409
    prompts = client.post('/api/data/prompts', json={
        'target': 'y', 'features': ['x1', 'x2'],
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


@pytest.mark.parametrize(
    ('kind', 'variables', 'target'),
    [
        ('polynomial', {'x1', 'x2', 'y'}, 'y'),
        ('grouped_parameters', {'x1', 'x2', 'label', 'y'}, 'y'),
        ('driven_ode', {'x', 'dx_dt'}, 'dx_dt'),
        ('kuramoto_ba', {'omega', 'x', 'dx_dt', 'A'}, 'dx_dt'),
    ],
)
def test_demo_dataset_variants(tmp_path, kind, variables, target):
    session = InteractiveSession(tmp_path / kind)
    path = session.create_demo(kind)
    loaded = load_context_data(path)
    assert set(loaded['variable_axes']) == variables
    assert session.context.target == target
    if kind == 'polynomial':
        np.testing.assert_allclose(
            loaded['data']['y'],
            1 + loaded['data']['x1']**2 + 2*loaded['data']['x1']*loaded['data']['x2'],
        )
    elif kind == 'grouped_parameters':
        assert set(loaded['data']['label']) == {'alpha', 'beta', 'gamma'}
        assert loaded['data']['label'].dtype.kind == 'U'
    elif kind == 'driven_ode':
        assert loaded['data']['t'].shape == loaded['data']['x'].shape == loaded['data']['dx_dt'].shape
        expected = (
            1.15*np.sin(1.35*loaded['data']['t'])
            + 0.42*np.sin(2.7*loaded['data']['t'])
            - 0.24*loaded['data']['x']
            - 0.075*loaded['data']['x']**3
        )
        np.testing.assert_allclose(loaded['data']['dx_dt'], expected)
    else:
        assert loaded['data']['omega'].shape == loaded['data']['x'].shape == loaded['data']['dx_dt'].shape == (481, 10)
        assert loaded['data']['A'].shape == (34, 2)
        assert set(np.unique(loaded['data']['A'])) == set(range(10))
        assert loaded['num_nodes'] == 10
        assert loaded['relation_names'] == {'A'}
        np.testing.assert_array_equal(loaded['data']['endpoint'], ['target', 'source'])
        np.testing.assert_array_equal(loaded['data']['node'], [f'node{index}' for index in range(1, 11)])


def test_demo_dataset_rejects_unknown_kind(tmp_path):
    session = InteractiveSession(tmp_path)
    with pytest.raises(ValueError, match='Unknown sample dataset'):
        session.create_demo('unknown')


def test_workspace_npy_preview(platform):
    client, session = platform
    np.save(
        session.workspace / 'matrix.npy',
        np.array([[1.5, 2.5], [3.5, 4.5]]),
    )
    np.save(
        session.workspace / 'labels.npy',
        np.array(['alpha', 'beta', 'gamma']),
    )

    matrix = client.get(
        '/api/workspace/preview', params={'path': 'matrix.npy'},
    )
    assert matrix.status_code == 200
    assert matrix.json() == {
        'kind': 'npy',
        'shape': [2, 2],
        'dtype': 'float64',
        'size': 4,
        'text': '[[1.5 2.5]\n [3.5 4.5]]',
        'truncated': False,
    }

    labels = client.get(
        '/api/workspace/preview', params={'path': 'labels.npy'},
    ).json()
    assert labels['kind'] == 'npy'
    assert labels['shape'] == [3]
    assert labels['dtype'].startswith('<U')
    assert labels['text'] == "['alpha' 'beta' 'gamma']"


def test_workspace_items_can_be_locked_and_unlocked(platform):
    client, session = platform
    assert client.put(
        '/api/workspace/upload', params={'path': 'research/source.csv'}, content=b'x\n1\n',
    ).status_code == 200

    response = client.put('/api/workspace/lock', json={
        'path': 'research', 'locked': True,
    })
    assert response.status_code == 200, response.text
    tree = client.get('/api/workspace', params={'recursive': True}).json()['entries']
    directory = next(item for item in tree if item['name'] == 'research')
    source = directory['children'][0]
    assert directory['locked'] and directory['read_only'] and not directory['mounted']
    assert source['locked'] and source['read_only'] and not source['mounted']
    assert not (session.workspace / 'research').stat().st_mode & 0o222
    assert not (session.workspace / 'research' / 'source.csv').stat().st_mode & 0o222
    assert client.delete(
        '/api/workspace', params={'path': 'research/source.csv'},
    ).status_code == 400
    assert client.put(
        '/api/workspace/upload', params={'path': 'research/new.csv'}, content=b'x\n2\n',
    ).status_code == 400

    response = client.put('/api/workspace/lock', json={
        'path': 'research', 'locked': False,
    })
    assert response.status_code == 200, response.text
    assert (session.workspace / 'research').stat().st_mode & 0o200
    assert (session.workspace / 'research' / 'source.csv').stat().st_mode & 0o200
    assert client.delete(
        '/api/workspace', params={'path': 'research/source.csv'},
    ).status_code == 200


def test_workspace_upload_stages_on_destination_filesystem(platform, monkeypatch):
    client, session = platform
    replace = os.replace

    def same_directory_replace(source, destination):
        assert os.path.dirname(os.fspath(source)) == os.path.dirname(os.fspath(destination))
        return replace(source, destination)

    monkeypatch.setattr('sr_harness.web.platform.os.replace', same_directory_replace)
    response = client.put(
        '/api/workspace/upload', params={'path': 'context.data.zip'}, content=b'PK\x03\x04',
    )
    assert response.status_code == 200, response.text
    assert (session.workspace / 'context.data.zip').read_bytes() == b'PK\x03\x04'

    def failed_replace(source, destination):
        raise OSError('simulated storage failure')

    monkeypatch.setattr('sr_harness.web.platform.os.replace', failed_replace)
    response = client.put(
        '/api/workspace/upload', params={'path': 'failed.zip'}, content=b'PK\x03\x04',
    )
    assert response.status_code == 400
    assert response.json()['detail'] == 'Unable to upload file: simulated storage failure'


def test_context_data_preview_uses_aligned_one_dimensional_variables(platform):
    client, session = platform
    directory = session.workspace / 'context.data'
    directory.mkdir()
    (directory / 'manifest.json').write_text(json.dumps({
        'variables': {
            'x': {'file': 'x.npy', 'description': 'Input.', 'axes': ['sample']},
            'y': {'file': 'y.npy', 'description': 'Output.', 'axes': ['sample']},
            'A': {
                'file': 'A.npy', 'description': 'Directed edge list.',
                'axes': ['edge', 'endpoint'],
            },
        },
        'axes': {
            'sample': {'values': [0, 1, 2], 'description': 'Sample index.'},
            'edge': {'size': 2, 'description': 'Edge index.'},
            'endpoint': {
                'values': ['target', 'source'], 'description': 'Endpoint order.',
            },
        },
    }))
    np.save(directory / 'x.npy', np.array([1.0, 2.0, 3.0]))
    np.save(directory / 'y.npy', np.array([2.0, 4.0, 6.0]))
    np.save(directory / 'A.npy', np.array([[1, 0], [0, 1]]))
    session.context.commit_context_data(load_context_data(directory))

    preview = client.get('/api/data/context').json()
    assert preview['columns'] == ['sample', 'edge', 'endpoint', 'x', 'y', 'A']
    assert preview['preview_columns'] == ['sample', 'x', 'y']
    assert preview['column_kinds'] == {
        'sample': 'axis', 'edge': 'axis', 'endpoint': 'axis',
        'x': 'variable', 'y': 'variable', 'A': 'variable',
    }
    assert preview['rows'] == 3
    assert preview['variables']['A']['shape'] == [2, 2]
    assert preview['axes']['endpoint']['size'] == 2
    assert preview['data'] == [
        {'sample': 0, 'x': 1.0, 'y': 2.0},
        {'sample': 1, 'x': 2.0, 'y': 4.0},
        {'sample': 2, 'x': 3.0, 'y': 6.0},
    ]
    heatmap = client.post('/api/data/context/heatmap', json={
        'row': ['sample'], 'column': ['x', 'y'], 'color': '', 'z': '',
    })
    assert heatmap.status_code == 200, heatmap.text
    assert heatmap.json()['values'] == [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
    prompts = client.post('/api/data/prompts', json={
        'target': 'y', 'features': ['sample', 'x'],
        'variable_descriptions': {'sample': 'Sample index.'},
    })
    assert prompts.status_code == 200, prompts.text
    assert "Feature names: ['sample', 'x']" in prompts.json()['user_prompt']
    np.save(directory / 'x.npy', np.array([10.0, 20.0, 30.0]))
    reloaded = client.post('/api/data/context/reload')
    assert reloaded.status_code == 200, reloaded.text
    np.testing.assert_array_equal(session.context.data['x'], [10.0, 20.0, 30.0])
    assert client.get('/api/data/context').json()['data'][0]['x'] == 10.0


def test_context_data_roles_accept_multidimensional_network_variables(platform):
    client, session = platform
    directory = session.workspace / 'context.data'
    directory.mkdir()
    (directory / 'manifest.json').write_text(json.dumps({
        'num_nodes': 2,
        'variables': {
                'theta': {
                    'file': 'theta.npy', 'description': 'Node phases.',
                    'axes': ['time', 'node'],
            },
            'omega': {
                'file': 'omega.npy', 'description': 'Natural frequencies.',
                'axes': ['time', 'node'],
            },
            'A': {
                'file': 'A.npy', 'description': 'Directed edge list.',
                'axes': ['edge', 'endpoint'], 'kind': 'relation',
            },
                'dtheta_dt': {
                    'file': 'dtheta_dt.npy', 'description': 'Phase derivatives.',
                    'axes': ['time', 'node'],
            },
        },
        'axes': {
            'time': {'values': [0.0, 0.1, 0.2], 'description': 'Time.'},
            'node': {'size': 2, 'description': 'Node index.'},
            'edge': {'size': 2, 'description': 'Edge index.'},
            'endpoint': {
                'values': ['target', 'source'], 'description': 'Endpoint order.',
            },
        },
    }))
    np.save(directory / 'theta.npy', np.zeros((3, 2)))
    np.save(directory / 'omega.npy', np.ones((3, 2)))
    np.save(directory / 'A.npy', np.array([[0, 1], [1, 0]]))
    np.save(directory / 'dtheta_dt.npy', np.full((3, 2), 2.0))
    session.context.commit_context_data(load_context_data(directory))

    preview = client.get('/api/data/context').json()
    assert preview['columns'] == [
        'time', 'node', 'edge', 'endpoint',
        'theta', 'omega', 'A', 'dtheta_dt',
    ]
    assert preview['preview_columns'] == []
    assert preview['data'] == []
    relation_group = next(
        group for group in preview['preview_groups'] if group['variable'] == 'A'
    )
    assert relation_group['kind'] == 'relation'
    relation_preview = client.get(
        '/api/data/context/preview', params={'group': relation_group['id']},
    )
    assert relation_preview.status_code == 200, relation_preview.text
    assert relation_preview.json() == {
        **relation_group,
        'description': 'Directed edge list.',
        'values': [[0, 1], [1, 0]],
        'axis_values': {
            'edge': [0, 1], 'endpoint': ['target', 'source'],
        },
        'node_axis': 'node',
        'nodes': [{'id': 0, 'label': 0}, {'id': 1, 'label': 1}],
        'coordinates': [[0, 1], [1, 0]],
        'endpoint_count': 2,
        'relation_count': 2,
        'truncated': False,
    }
    plot_preview = client.get('/api/data/context/plot', params={
        'x': 'theta', 'y': 'omega', 'hue': 'node', 'size': 'time',
    })
    assert plot_preview.status_code == 200, plot_preview.text
    assert plot_preview.json()['axes'] == ['time', 'node']
    assert plot_preview.json()['shape'] == [3, 2]
    assert plot_preview.json()['values']['node'] == [0, 1, 0, 1, 0, 1]
    assert plot_preview.json()['values']['time'] == [0.0, 0.0, 0.1, 0.1, 0.2, 0.2]
    heatmap = client.post('/api/data/context/heatmap', json={
        'row': ['time'], 'column': ['node'], 'color': 'theta', 'z': '',
    })
    assert heatmap.status_code == 200, heatmap.text
    assert heatmap.json()['shape'] == [3, 2]
    assert heatmap.json()['row_labels'] == [0.0, 0.1, 0.2]
    assert heatmap.json()['column_labels'] == [0, 1]

    response = client.put('/api/data/selection', json={
        'target': 'dtheta_dt',
        'features': ['theta', 'omega', 'A'],
    })
    assert response.status_code == 200, response.text
    X, y = session._select_context_columns(
        session.context.target, list(session.context.feature_names()),
    )
    assert {name: value.shape for name, value in X.items()} == {
        'theta': (3, 2), 'omega': (3, 2), 'A': (2, 2),
    }
    assert y['dtheta_dt'].shape == (3, 2)
    assert X['A'].dtype == np.dtype('int64')


def test_context_data_preview_summarizes_higher_order_variables(platform):
    client, session = platform
    directory = session.workspace / 'context.data'
    directory.mkdir()
    (directory / 'manifest.json').write_text(json.dumps({
        'variables': {
            'tensor': {
                'file': 'tensor.npy', 'description': 'Four-dimensional data.',
                'axes': ['a', 'b', 'c', 'd'],
            },
        },
        'axes': {
            name: {'values': [0, 1], 'description': f'Axis {name}.'}
            for name in ['a', 'b', 'c', 'd']
        },
    }))
    np.save(directory / 'tensor.npy', np.arange(16).reshape(2, 2, 2, 2))
    session.context.commit_context_data(load_context_data(directory))

    overview = client.get('/api/data/context').json()
    group = next(
        item for item in overview['preview_groups']
        if item.get('variable') == 'tensor'
    )
    response = client.get('/api/data/context/preview', params={'group': group['id']})

    assert response.status_code == 200, response.text
    assert response.json()['sample_values'] == list(range(8))
    assert response.json()['value_range'] == [0.0, 15.0]


def test_variable_roles_can_be_updated_between_search_rounds(platform, monkeypatch):
    client, session = platform
    directory = session.workspace / 'context.data'
    directory.mkdir()
    (directory / 'manifest.json').write_text(json.dumps({
        'variables': {
            'x': {'file': 'x.npy', 'description': 'Original input.', 'axes': ['sample']},
            'z': {'file': 'z.npy', 'description': 'New input.', 'axes': ['sample']},
            'y': {'file': 'y.npy', 'description': 'Output.', 'axes': ['sample']},
        },
        'axes': {
            'sample': {'values': [2000, 2001, 2002], 'description': 'Year.'},
        },
    }))
    np.save(directory / 'x.npy', np.array([1.0, 2.0, 3.0]))
    np.save(directory / 'z.npy', np.array([2.0, 3.0, 5.0]))
    np.save(directory / 'y.npy', np.array([4.0, 6.0, 9.0]))
    loaded = load_context_data(directory)
    session.context.commit_context_data(loaded)

    response = client.put('/api/data/selection', json={
        'target': 'y',
        'features': ['sample', 'x'],
        'variable_descriptions': {'sample': 'Calendar year.', 'x': ''},
    })
    assert response.status_code == 200, response.text
    assert session.context.target == 'y'
    assert list(session.context.feature_names()) == ['x']
    assert session.context.variable_descriptions['sample'] == 'Calendar year.'
    assert session.context.variable_descriptions['x'] == ''
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['axes']['sample']['description'] == 'Calendar year.'
    assert manifest['variables']['x']['description'] == ''

    descriptions_only = client.put('/api/data/descriptions', json={
        'variable_descriptions': {'x': 'Saved without a valid role selection.'},
    })
    assert descriptions_only.status_code == 200, descriptions_only.text
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['variables']['x']['description'] == 'Saved without a valid role selection.'

    manifest['variables']['x']['description'] = 'Edited outside the browser.'
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    revision = session.context.args.data_revision
    snapshot = session.snapshot()
    assert session.context.variable_descriptions['x'] == 'Edited outside the browser.'
    assert snapshot['data_context']['variable_descriptions']['x'] == 'Edited outside the browser.'
    assert session.context.args.data_revision == revision + 1

    # Reloading the full manifest restores every non-axis variable as a feature.
    session.context.commit_context_data(loaded)
    assert list(session.context.feature_names()) == ['x', 'z']

    session.state = 'running'
    monkeypatch.setattr(session.sr_interaction_manager, 'status', lambda: {
        'interaction_state': 'running', 'paused': False, 'waiting_at_boundary': False,
    })
    blocked = client.put('/api/data/selection', json={
        'target': 'y', 'features': ['x', 'z'],
    })
    assert blocked.status_code == 409

    monkeypatch.setattr(session.sr_interaction_manager, 'status', lambda: {
        'interaction_state': 'paused', 'paused': True, 'waiting_at_boundary': True,
    })
    updated = client.put('/api/data/selection', json={
        'target': 'y', 'features': ['x', 'z'],
        'variable_descriptions': {'z': 'Feature added during the pause.'},
    })
    assert updated.status_code == 200, updated.text
    assert list(session.context.feature_names()) == ['x', 'z']


def test_read_only_startup_workspace_inputs_are_visible(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    table = source / 'observations.csv'
    table.write_text('x,y\n1,2\n')
    session = InteractiveSession(
        tmp_path / 'logs',
        workspace_files=[str(source)],
    )
    with TestClient(create_app(tmp_path, session=session)) as client:
        root = client.get('/api/workspace').json()['entries']
        assert {entry['name'] for entry in root} >= {'source'}
        assert 'context.evaluator' not in {entry['name'] for entry in root}
        assert root[0]['directory']
        recursive_root = client.get(
            '/api/workspace', params={'recursive': True},
        ).json()['entries']
        source_entry = next(entry for entry in recursive_root if entry['name'] == 'source')
        assert source_entry['read_only'] is True
        assert source_entry['mounted'] is True
        assert source_entry['locked'] is False
        assert source_entry['children'][0]['path'] == (
            'source/observations.csv'
        )
        assert source_entry['children'][0]['read_only'] is True
        assert client.put('/api/workspace/lock', json={
            'path': 'source', 'locked': False,
        }).status_code == 400
        assert client.get(
            '/api/workspace/size', params={'path': 'source'},
        ).json()['size'] == len(b'x,y\n1,2\n')
        mounted_archive = client.get(
            '/api/workspace/download', params={'path': 'source'},
        )
        assert mounted_archive.status_code == 200
        with zipfile.ZipFile(io.BytesIO(mounted_archive.content)) as archive:
            assert archive.read('source/observations.csv') == b'x,y\n1,2\n'
        assert client.get('/api/workspace/preview', params={
            'path': 'source/observations.csv',
        }).json()['text'] == 'x,y\n1,2\n'
        assert client.get('/api/data/csv-files').json()['files'] == [
            'source/observations.csv',
        ]
        assert client.put('/api/workspace/upload', params={
            'path': 'source/new.csv',
        }, content=b'x\n3\n').status_code == 400
        assert client.patch('/api/workspace', json={
            'source': 'source', 'destination': 'renamed-source',
        }).status_code == 400
        assert client.delete('/api/workspace', params={'path': 'source'}).status_code == 400


def test_workspace_directory_move_rename_and_delete(platform):
    client, _ = platform
    assert client.post('/api/workspace/directory', json={'path': 'research'}).json() == {
        'path': 'research',
    }
    assert client.put(
        '/api/workspace/upload', params={'path': 'notes.txt'}, content=b'notes',
    ).status_code == 200
    moved = client.patch('/api/workspace', json={
        'source': 'notes.txt', 'destination': 'research/notes.txt',
    })
    assert moved.status_code == 200
    directory_archive = client.get(
        '/api/workspace/download', params={'path': 'research'},
    )
    assert directory_archive.status_code == 200
    assert 'research.zip' in directory_archive.headers['content-disposition']
    with zipfile.ZipFile(io.BytesIO(directory_archive.content)) as archive:
        assert archive.read('research/notes.txt') == b'notes'
    renamed = client.patch('/api/workspace', json={
        'source': 'research/notes.txt', 'destination': 'research/summary.txt',
    })
    assert renamed.status_code == 200
    assert client.get(
        '/api/workspace/download', params={'path': 'research/summary.txt'},
    ).content == b'notes'
    assert client.patch('/api/workspace', json={
        'source': 'research', 'destination': 'renamed-research',
    }).status_code == 200
    assert client.patch('/api/workspace', json={
        'source': 'renamed-research',
        'destination': 'renamed-research/nested',
    }).status_code == 400
    assert client.delete(
        '/api/workspace', params={'path': 'renamed-research/summary.txt'},
    ).status_code == 200
    assert client.delete(
        '/api/workspace', params={'path': 'renamed-research'},
    ).status_code == 200


def test_session_temporary_and_explicit_workspace_lifetimes(tmp_path):
    temporary_session = InteractiveSession(tmp_path / 'logs')
    temporary_workspace = temporary_session.workspace
    assert temporary_workspace.exists()
    temporary_session.close()
    assert not temporary_workspace.exists()

    explicit_workspace = tmp_path / 'workspace'
    explicit_session = InteractiveSession(
        tmp_path / 'logs',
        workspace_path=explicit_workspace,
    )
    explicit_session.close()
    assert explicit_workspace.exists()


def test_runtime_capabilities_can_be_configured(platform):
    client, session = platform
    capabilities = client.get('/api/session/capabilities')
    assert capabilities.status_code == 200
    payload = capabilities.json()
    tool_names = {tool['name'] for tool in payload['tools']}
    skill_names = {skill['name'] for skill in payload['skills']}
    assert {'evaluate_formula', 'workspace_shell'} <= tool_names
    assert 'code_executor' not in tool_names
    assert {'validate_context_data', 'validate_evaluator'} <= tool_names
    assert {'commit_data', 'load_context_data'}.isdisjoint(tool_names)
    assert 'discover-symbolic-laws' in skill_names
    catalogs = {
        agent: client.get('/api/session/capabilities', params={'agent': agent}).json()
        for agent in ('search', 'data', 'evaluator')
    }
    assert catalogs['search']['tools'] == catalogs['data']['tools'] == catalogs['evaluator']['tools']
    assert catalogs['search']['skills'] == catalogs['data']['skills'] == catalogs['evaluator']['skills']
    assert catalogs['search']['default_tools'] != catalogs['data']['default_tools']
    assert catalogs['data']['default_tools'] != catalogs['evaluator']['default_tools']
    assert {
        'read_source', 'delegate_subagent', 'sr4mdl', 'nd2', 'evaluate_eic',
        'workspace_shell', 'workspace_code_executor',
    }.isdisjoint(catalogs['search']['default_tools'])
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


def test_search_agent_uses_restricted_default_tools(tmp_path):
    session = InteractiveSession(tmp_path / 'logs')
    try:
        assert session.settings['tools'] == session.capabilities('search')['default_tools']
        assert {
            'read_source', 'delegate_subagent', 'sr4mdl', 'nd2', 'evaluate_eic',
            'workspace_shell', 'workspace_code_executor',
        }.isdisjoint(session.settings['tools'])
    finally:
        session.close()


def test_evaluator_context_settings_update_context_args(platform):
    client, session = platform
    response = client.post('/api/session/settings', json={
        'validation_fraction': 0.3,
        'split_random_state': 17,
        'split_by': 'ood',
        'split_ood_variable': 'time',
        'ranking_metric': 'r2',
        'larger_is_better': True,
    })
    assert response.status_code == 200, response.text
    assert session.context.args.validation_fraction == 0.3
    assert session.context.args.split_random_state == 17
    assert session.context.args.split_by == 'ood'
    assert session.context.args.split_ood_variable == 'time'
    assert session.context.args.ranking_metric == 'r2'
    assert session.context.args.larger_is_better is True


def test_provider_api_key_is_synced_to_dotenv_without_being_returned(
    tmp_path, monkeypatch,
):
    env_path = tmp_path / '.env'
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    session = InteractiveSession(tmp_path / 'logs', env_path=env_path)
    app = create_app(tmp_path, session=session)

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
    assert {'validate_context_data', 'workspace_code_executor', 'read_skill'} <= {
        tool['name'] for tool in catalog['tools']
    }
    assert 'validate_context_data' in catalog['default_tools']
    assert 'commit_data' not in catalog['default_tools']
    assert 'load_context_data' not in catalog['default_tools']
    assert 'discover-symbolic-laws' in {
        skill['name'] for skill in catalog['skills']
    }
    assert 'discover-symbolic-laws' not in catalog['default_skills']
    assert 'discover-symbolic-laws' not in session.data_agent_settings['skills']
    assert 'max_turns' not in session.data_agent_settings

    response = client.put('/api/data/agent/settings', json={
        'llm_provider': 'openai',
        'llm_model': 'test-data-model',
        'tool_parser': 'json',
        'llm_max_tokens': 2048,
        'tools': ['workspace_shell', 'validate_context_data'],
        'skills': [],
        'proxy': '',
    })
    assert response.status_code == 200
    assert response.json()['data_agent_settings'] == {
        'llm_provider': 'openai',
        'llm_model': 'test-data-model',
        'tool_parser': 'json',
        'llm_max_tokens': 2048,
        'tools': ['workspace_shell', 'validate_context_data'],
        'skills': [],
        'proxy': '',
    }
    assert session.settings['llm_provider'] == 'openrouter'
    assert client.put('/api/data/agent/settings', json={
        'tools': ['workspace_shell'],
    }).status_code == 400


def test_data_agent_can_be_stopped_independently(platform, monkeypatch):
    client, session = platform
    started = threading.Event()

    def wait_for_stop(agent, instruction):
        started.set()
        while True:
            agent._check_stop()
            time.sleep(.005)

    monkeypatch.setattr(DataPreparationAgent, 'run', wait_for_stop)
    response = client.post('/api/data/agent', json={'message': 'Prepare data.'})
    assert response.status_code == 200, response.text
    assert started.wait(2)

    response = client.post('/api/data/agent/stop')
    assert response.status_code == 200, response.text
    assert response.json()['data_state'] == 'stopping'
    assert session.data_thread.is_alive()
    response = client.post('/api/data/agent/stop')
    assert response.status_code == 200, response.text
    assert response.json()['data_force_pause_requested'] is True
    session.data_thread.join(2)
    assert not session.data_thread.is_alive()
    assert session.data_state == 'stopped'
    assert session.data_result['status'] == 'stopped'


def test_evaluator_agent_can_be_stopped_independently(platform, monkeypatch):
    client, session = platform
    started = threading.Event()

    def wait_for_stop(agent, instruction):
        started.set()
        while True:
            agent._check_stop()
            time.sleep(.005)

    monkeypatch.setattr(EvaluatorConstructionAgent, 'run', wait_for_stop)
    response = client.post('/api/evaluator/agent/start', json={
        'message': 'Construct an evaluator.',
    })
    assert response.status_code == 200, response.text
    assert response.json()['evaluator_agent_state'] == 'running'
    assert started.wait(2)

    response = client.post('/api/evaluator/agent/stop')
    assert response.status_code == 200, response.text
    assert response.json()['evaluator_agent_state'] == 'stopping'
    assert session.evaluator_agent_thread.is_alive()
    response = client.post('/api/evaluator/agent/stop')
    assert response.status_code == 200, response.text
    assert response.json()['evaluator_force_pause_requested'] is True
    session.evaluator_agent_thread.join(2)
    assert not session.evaluator_agent_thread.is_alive()
    assert session.evaluator_agent_state == 'stopped'
    assert session.evaluator_agent_result['status'] == 'stopped'
    assert any(
        event['kind'] == 'execution_failed'
        and event['payload']['status'] == 'stopped'
        for event in session.evaluator_interaction_manager.get_recent_events()['events']
    )


def test_data_agent_model_test_checks_completion_and_tool_call(platform, monkeypatch):
    client, _ = platform

    class FakeAPI(CancellableFakeAPI):
        def __init__(self):
            self.requests = 0

        def __call__(self, prompt, **kwargs):
            self.requests += 1

            def generate():
                if self.requests == 1:
                    yield {
                        'content': 'SRHARNESS_OK',
                        'tool_call': [],
                        'message': {'role': 'assistant', 'content': 'SRHARNESS_OK'},
                    }
                else:
                    call = ToolCall(
                        'report_model_test',
                        {'answer': 'SRHARNESS_TOOL_OK'},
                        id='model-test',
                    )
                    yield {
                        'content': '',
                        'tool_call': [call],
                        'message': {'role': 'assistant', 'content': ''},
                    }
                return {'usage': {'token': {}, 'price': {}}, 'contents': [], 'tool_calls': []}

            return APICallResult(generate())

    fake = FakeAPI()
    created = {}

    def create_api(provider, **kwargs):
        created.update(provider=provider, **kwargs)
        return fake

    monkeypatch.setattr(BaseAPI, 'create', create_api)
    response = client.post('/api/data/agent/test', json={
        'llm_provider': 'deepseek',
        'llm_model': 'deepseek-v4-flash-0731',
        'tool_parser': 'openai',
        'llm_max_tokens': 512,
        'proxy': '',
    })
    assert response.status_code == 200, response.text
    assert response.json() == {
        'ok': True,
        'accessible': True,
        'plain_response': 'SRHARNESS_OK',
        'tool_call_supported': True,
        'called_tools': ['report_model_test'],
    }
    assert created['provider'] == 'deepseek'
    assert created['model'] == 'deepseek-v4-flash-0731'
    assert created['tool_parser_name'] == 'openai'
    assert created['tool_list'][0].metadata.name == 'report_model_test'

    fake.requests = 0
    response = client.post('/api/session/test', json={
        'llm_provider': 'openrouter',
        'llm_model': 'openai/gpt-4.1-mini',
        'tool_parser': 'openai',
        'llm_max_tokens': 256,
    })
    assert response.status_code == 200, response.text
    assert response.json() == {
        'ok': True,
        'accessible': True,
        'plain_response': 'SRHARNESS_OK',
        'tool_call_supported': True,
        'called_tools': ['report_model_test'],
    }
    assert created['provider'] == 'openrouter'
    assert created['model'] == 'openai/gpt-4.1-mini'
    assert created['tool_parser_name'] == 'openai'


def test_evaluator_configuration_test_and_restricted_agent(platform, monkeypatch):
    client, session = platform
    capabilities = client.get(
        '/api/session/capabilities', params={'agent': 'evaluator'},
    )
    assert capabilities.status_code == 200
    assert {'read_source', 'workspace_shell', 'workspace_code_executor', 'validate_evaluator', 'read_skill'} <= {
        tool['name'] for tool in capabilities.json()['tools']
    }
    assert capabilities.json()['default_tools'] == [
        'read_source', 'workspace_shell', 'workspace_code_executor', 'validate_evaluator', 'read_skill',
    ]
    assert 'discover-symbolic-laws' not in capabilities.json()['default_skills']
    assert 'discover-symbolic-laws' not in session.evaluator_agent_settings['skills']
    configuration = client.get('/api/evaluator')
    assert configuration.status_code == 200
    assert configuration.json()['selected'] == 'default'
    assert {item['id'] for item in configuration.json()['evaluators']} == {'default', 'graph'}
    assert not session.evaluator_workspace.exists()
    session.evaluator_workspace.mkdir()
    invalid_file = session.evaluator_workspace / 'broken_evaluator.py'
    invalid_file.write_text(
        'from .default_evaluator import DefaultEvaluator\n\n'
        'class BrokenEvaluator(DefaultEvaluator)\n    pass\n'
    )
    with_invalid = client.get('/api/evaluator')
    invalid = next(
        item for item in with_invalid.json()['evaluators']
        if item['id'] == 'custom:broken_evaluator.py'
    )
    assert invalid['invalid'] is True
    assert invalid['source'] == invalid_file.read_text()
    assert 'Invalid evaluator syntax' in invalid['error']
    invalid_file.unlink()
    session.evaluator_workspace.rmdir()
    evaluator_dir = Path(sr_harness.__file__).parent / 'evaluator'
    default = next(item for item in configuration.json()['evaluators'] if item['id'] == 'default')
    assert default['abstract'] is False
    assert 'class DefaultEvaluator:' in default['source']
    assert default['source'] == (evaluator_dir / 'default_evaluator.py').read_text()
    assert 'class CustomEvaluator(DefaultEvaluator):' in configuration.json()['custom_template']

    renamed_source = default['source'].replace(
        'class DefaultEvaluator:',
        'class ProjectEvaluator(DefaultEvaluator):',
    )
    renamed = client.put('/api/evaluator', json={
        'selected': 'default', 'source': renamed_source,
    })
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()['selected'] == 'custom:project_evaluator.py'
    assert renamed.json()['custom_name'] == 'ProjectEvaluator'
    saved_evaluator = session.workspace / 'context.evaluator' / 'project_evaluator.py'
    assert renamed.json()['custom_file'] == str(saved_evaluator)
    assert {
        item['id'] for item in renamed.json()['evaluators']
    } == {'default', 'graph', 'custom:project_evaluator.py'}
    assert saved_evaluator.read_text() == renamed_source
    assert type(session.context.evaluator).__name__ == 'ProjectEvaluator'
    assert type(session.context.evaluator).__module__ == 'sr_harness.evaluator.project_evaluator'

    conflicting = client.put('/api/evaluator', json={
        'selected': 'default', 'source': default['source'] + '\n# modified\n',
    })
    assert conflicting.status_code == 400
    assert 'conflicts with a built-in evaluator' in conflicting.text
    assert saved_evaluator.read_text() == renamed_source

    overwritten_source = renamed_source.replace(
        'class ProjectEvaluator(DefaultEvaluator):',
        'class ProjectEvaluator(DefaultEvaluator):\n    revision = 2',
    )
    overwritten = client.put('/api/evaluator', json={
        'selected': 'custom', 'source': overwritten_source,
    })
    assert overwritten.status_code == 200, overwritten.text
    assert saved_evaluator.read_text() == overwritten_source
    assert session.context.evaluator.revision == 2

    custom_source = '''from sr_harness import DefaultEvaluator

class CustomEvaluator(DefaultEvaluator):
    pass
'''
    configured = client.put('/api/evaluator', json={
        'selected': 'custom', 'source': custom_source,
    })
    assert configured.status_code == 200, configured.text
    assert configured.json()['selected'] == 'custom:custom_evaluator.py'
    assert type(session.context.evaluator).__name__ == 'CustomEvaluator'
    custom_ids = {item['id'] for item in configured.json()['evaluators']}
    assert custom_ids == {
        'default', 'graph', 'custom:custom_evaluator.py',
        'custom:project_evaluator.py',
    }
    project_item = next(
        item for item in configured.json()['evaluators']
        if item['id'] == 'custom:project_evaluator.py'
    )
    reselected = client.put('/api/evaluator', json={
        'selected': project_item['id'], 'source': project_item['source'],
    })
    assert reselected.status_code == 200, reselected.text
    assert reselected.json()['selected'] == 'custom:project_evaluator.py'
    assert type(session.context.evaluator).__name__ == 'ProjectEvaluator'
    forbidden = client.put('/api/evaluator', json={
        'selected': 'custom',
        'source': 'import os\nclass CustomEvaluator(DefaultEvaluator):\n    pass\n',
    })
    assert forbidden.status_code == 400
    assert 'scientific SRHarness modules' in forbidden.text

    invalid_annotation = client.put('/api/evaluator', json={
        'selected': 'custom',
        'source': (
            'import sr_harness_engine as engine\n'
            'from sr_harness import DefaultEvaluator\n'
            'class BadAnnotationEvaluator(DefaultEvaluator):\n'
            '    def helper(self, context: engine.AgentContext):\n'
            '        pass\n'
        ),
    })
    assert invalid_annotation.status_code == 400
    assert 'AgentContext' in invalid_annotation.text

    client.post('/api/data/demo')
    session.context.commit_context_data(
        load_context_data(session.workspace / 'context.data'),
    )
    session.context.update_selection(target='y', features=['x1', 'x2'])
    tested = client.post('/api/evaluator/test', json={
        'selected': 'custom',
        'source': custom_source,
        'formula': "param('scale') * x1",
    })
    assert tested.status_code == 200, tested.text
    assert 'complexity' in tested.json()['result']['data_split_results']['train']['metrics']
    assert tested.json()['result']['evaluator'] == 'CustomEvaluator'
    assert tested.json()['result']['evaluator_file'] is None

    class RankedCandidates:
        @staticmethod
        def ranked_candidates():
            return [type('Record', (), {'formula': 'x1 + x2'})()]

    session.run_state = RankedCandidates()
    tested_best = client.post('/api/evaluator/test', json={
        'selected': 'custom', 'source': custom_source,
    })
    session.run_state = None
    assert tested_best.status_code == 200, tested_best.text
    assert tested_best.json()['formula'] == 'x1 + x2'
    tested_baseline = client.post('/api/evaluator/test', json={
        'selected': 'custom', 'source': custom_source,
    })
    assert tested_baseline.status_code == 200, tested_baseline.text
    assert tested_baseline.json()['formula'] == (
        "param('intercept') + param('coefficient_1') * x1 + "
        "param('coefficient_2') * x2"
    )
def test_data_agent_proxy_setting_persists_to_env_file(platform, tmp_path, monkeypatch):
    client, session = platform
    session.env_path = tmp_path / '.env'
    for name in ('http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY'):
        monkeypatch.delenv(name, raising=False)

    response = client.put('/api/data/agent/settings', json={
        'proxy': 'http://127.0.0.1:7890',
    })
    assert response.status_code == 200, response.text
    assert dotenv_values(session.env_path)['HTTP_PROXY'] == 'http://127.0.0.1:7890'
    assert dotenv_values(session.env_path)['HTTPS_PROXY'] == 'http://127.0.0.1:7890'
    assert os.environ['HTTP_PROXY'] == 'http://127.0.0.1:7890'
    assert os.environ['HTTPS_PROXY'] == 'http://127.0.0.1:7890'

    response = client.put('/api/data/agent/settings', json={'proxy': ''})
    assert response.status_code == 200, response.text
    assert 'HTTP_PROXY' not in dotenv_values(session.env_path)
    assert 'HTTPS_PROXY' not in dotenv_values(session.env_path)
    assert 'HTTP_PROXY' not in os.environ
    assert 'HTTPS_PROXY' not in os.environ


def test_data_agent_validates_prepared_excel_data_for_shared_context(platform, monkeypatch):
    client, session = platform
    import pandas as pd

    configured = client.put('/api/data/agent/settings', json={
        'llm_provider': 'deepseek',
        'llm_model': 'data-preparation-model',
        'tool_parser': 'openai',
        'llm_max_tokens': 1234,
        'tools': ['validate_context_data'],
        'skills': [],
    })
    assert configured.status_code == 200

    source = session.workspace / '中国人口数量变化与GDP变化.xlsx'
    pd.DataFrame({
        '年份': [2020, 2021, 2022, 2023, 2024],
        'GDP': [101, 115, 121, 127, 134],
        '人口数量': [1412, 1413, 1412, 1410, 1408],
    }).to_excel(source, index=False)

    class FakeDataAPI(CancellableFakeAPI):
        tool_description_json = []

        def __init__(self):
            self.turn = 0

        def __call__(self, prompt, **kwargs):
            self.turn += 1

            def generate():
                if self.turn == 1:
                    frame = pd.read_excel(source)
                    directory = session.workspace / 'context.data'
                    directory.mkdir()
                    for name in frame.columns:
                        np.save(directory / f'{name}.npy', frame[name].to_numpy())
                    (directory / 'manifest.json').write_text(json.dumps({
                        'variables': {
                            name: {
                                'file': f'{name}.npy',
                                'description': f'{name} variable.',
                                'axes': ['sample'],
                            }
                            for name in frame.columns
                        },
                        'axes': {
                            'sample': {
                                'size': len(frame),
                                'description': 'Row index.',
                            },
                        },
                    }, ensure_ascii=False), encoding='utf-8')
                    call = ToolCall('validate_context_data', {}, id='validate')
                    message = {'role': 'assistant', 'content': '整理并验证数据。', 'tool_calls': [{
                        'id': 'validate', 'type': 'function', 'function': {
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
    assert session.context.target is None
    assert set(session.context.variable_names()) == {'年份', 'GDP', '人口数量'}
    data_events = session.data_interaction_manager.get_recent_events()['events']
    data_event_kinds = [event['kind'] for event in data_events]
    assert 'prompt_added' in data_event_kinds
    assert data_event_kinds.index('context') < data_event_kinds.index('assistant_started')
    assert data_event_kinds.index('assistant_started') < data_event_kinds.index('assistant_completed')
    context_event = next(event for event in data_events if event['kind'] == 'context')
    assert context_event['payload']['turn'] == 1
    assert context_event['payload']['messages'][0]['role'] == 'system'
    assert 'data-preparation agent' in context_event['payload']['messages'][0]['content']
    assistant_event = next(event for event in data_events if event['kind'] == 'assistant_completed')
    assert assistant_event['payload']['provider'] == 'deepseek'
    assert assistant_event['payload']['model'] == 'data-preparation-model'
    preview = client.get('/api/data/context').json()
    assert preview['revision'] == 1
    assert preview['rows'] == 5
    assert preview['data'][0]['年份'] == 2020


def test_data_agent_has_no_turn_limit_and_keeps_cumulative_turns(platform, monkeypatch):
    client, session = platform

    class FakeDataAPI(CancellableFakeAPI):
        tool_description_json = []

        def __init__(self):
            self.turn = 0

        def __call__(self, prompt, **kwargs):
            self.turn += 1
            current_turn = self.turn

            def generate():
                if current_turn <= 13:
                    call = ToolCall(
                        'workspace_shell', {'command': 'ls'}, id=f'call-{current_turn}',
                    )
                    message = {
                        'role': 'assistant',
                        'content': f'Working in turn {current_turn}.',
                        'tool_calls': [{
                            'id': call.id,
                            'type': 'function',
                            'function': {'name': call.name, 'arguments': '{"command":"ls"}'},
                        }],
                    }
                    yield {
                        'content': message['content'],
                        'tool_call': [call],
                        'message': message,
                    }
                else:
                    message = {'role': 'assistant', 'content': 'Finished.'}
                    yield {'content': 'Finished.', 'tool_call': [], 'message': message}
                return {'usage': {'token': {}, 'price': {}}, 'responses': []}

            return APICallResult(generate())

    fake = FakeDataAPI()
    monkeypatch.setattr(BaseAPI, 'create', lambda *args, **kwargs: fake)

    response = client.post('/api/data/agent', json={'message': 'Prepare the data.'})
    assert response.status_code == 200, response.text
    session.data_thread.join(10)
    assert session.data_state == 'completed'
    assert session.data_agent.turn_count == 14

    response = client.post('/api/data/agent', json={'message': 'Continue.'})
    assert response.status_code == 200, response.text
    session.data_thread.join(10)
    assert session.data_state == 'completed'
    assert session.data_agent.turn_count == 15

    context_events = [
        event for event in session.data_interaction_manager.get_recent_events()['events']
        if event['kind'] == 'context'
    ]
    assert [event['payload']['turn'] for event in context_events] == list(range(1, 16))
    assert context_events[-1]['payload']['messages'][-1] == {
        'role': 'user', 'content': 'Continue.',
    }


def test_real_search_loop_with_fake_llm(platform, monkeypatch):
    client, session = platform
    prompts = []
    models = []

    class FakeAPI(CancellableFakeAPI):
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
    events = session.sr_interaction_manager.get_recent_events()['events']
    kinds = [e['kind'] for e in events]
    assert all(k in kinds for k in [
        'prompt_added', 'context', 'assistant_started', 'assistant_completed',
        'tool_started', 'tool_completed', 'topk_updated', 'execution_completed',
    ])
    assert kinds.index('prompt_added') < kinds.index('context') < kinds.index('assistant_started') < kinds.index('assistant_completed')
    user_event = next(event for event in events if event['kind'] == 'prompt_added' and event['payload']['message']['role'] == 'user')
    assert user_event['payload']['message']['content'] == 'Custom user prompt'
    assistant_events = [event for event in events if event['kind'] == 'assistant_completed']
    assert all(event['payload']['provider'] for event in assistant_events)
    assert all(event['payload']['model'] for event in assistant_events)
    runs = client.get('/api/runs').json()['runs']
    assert runs[0]['record_count'] == 3
    assert client.post('/api/session/start', json={}).status_code == 400


def test_advance_to_next_branch_and_restart(platform, monkeypatch):
    client, session = platform
    calls = 0

    class AdvancingAPI(CancellableFakeAPI):
        tool_description_json = []

        def __call__(self, prompt, **kwargs):
            nonlocal calls
            calls += 1
            session.sr_interaction_manager.command('next_c' if calls == 1 else 'next_r')

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
        for event in session.sr_interaction_manager.get_recent_events()['events']
        if event['kind'] == 'context'
    ]
    assert coordinates == [
        {'R': 1, 'C': 1, 'L': 1},
        {'R': 1, 'C': 2, 'L': 1},
        {'R': 2, 'C': 1, 'L': 1},
    ]


def test_tool_free_search_response_pauses_without_question_card(platform, monkeypatch):
    client, session = platform
    prompts = []

    class YieldingAPI(CancellableFakeAPI):
        tool_description_json = []

        def __call__(self, prompt, **kwargs):
            prompts.append(prompt)
            turn = len(prompts)
            if turn == 2:
                session.sr_interaction_manager.command('next_r')

            def generate():
                message = {'role': 'assistant', 'content': f'round {turn}'}
                yield {'content': message['content'], 'tool_call': [], 'message': message}
                return {'usage': {'token': {}, 'price': {}}, 'responses': []}

            return APICallResult(generate())

    monkeypatch.setattr(BaseAPI, 'create', lambda *args, **kwargs: YieldingAPI())
    response = client.post('/api/session/start', json={'max_refinement_depth': 3})
    assert response.status_code == 200, response.text
    deadline = time.monotonic() + 5
    while not session.sr_interaction_manager.status()['waiting_at_boundary'] and time.monotonic() < deadline:
        time.sleep(.01)
    status = session.sr_interaction_manager.status()
    assert status['paused'] is True
    assert status['waiting_at_boundary'] is True

    reply = client.post(
        '/api/control/command', json={'action': 'message', 'message': 'Try a power law.'},
    )
    assert reply.status_code == 200, reply.text
    session.thread.join(10)
    assert not session.thread.is_alive()
    assert session.state == 'completed'
    assert any(
        message.get('content', '').endswith('Try a power law.')
        for message in prompts[1]
    )


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
