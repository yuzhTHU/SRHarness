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
from sr_harness.agents.sr_agent_interactive import SRAgentInteractive
from sr_harness.core import APICallResult, ContextDataLoader, SearchRunState, ToolCall
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
    page = client.get('/')
    assert page.status_code == 200
    assert 'class="badge"' not in page.text
    assert 'id="search-record-link"' not in page.text
    assert 'id="prepare-tab"' in page.text
    assert 'id="data-tab"' in page.text
    assert 'id="data-preparation"' in page.text
    assert 'id="data-setup"' in page.text
    assert 'id="run-setup"' in page.text
    assert "connected:'已连接到后端'" in page.text
    assert 'id="pause"' not in page.text
    assert 'id="stop"' not in page.text
    assert 'function renderComposerControl()' in page.text
    assert "if(!questionId&&!session.paused){const control=await api('/api/control/command',{action:'pause'})" in page.text
    assert "const control=await api('/api/control/command',{action:'message',message:prompt})" in page.text
    assert '.composer #send.send-icon{display:grid;place-items:center;width:34px;height:34px' in page.text
    assert "for(const [inputId,buttonId] of [['R','next-c'],['C','next-r']]" in page.text
    assert "button.textContent='+1'" in page.text
    assert '#settings-pane-search .search-step-control{display:flex;align-items:center;gap:6px' in page.text
    assert '#settings-pane-search .search-step-control .advance-button{display:grid;place-items:center;flex:0 0 28px;width:28px;height:28px' in page.text
    assert "composerToolbar.prepend($('settings-toggle'),$('initial-prompt-toggle'))" in page.text
    assert "$('status').className='small muted';$('send').before($('status'))" in page.text
    assert '.composer>.row #status{padding:0;border-radius:0;background:transparent;color:var(--muted);font-size:11px' in page.text
    assert 'id="plot-variable-palette"' in page.text
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
    assert 'id="evaluator-agent-feed" class="data-agent-feed evaluator-agent-feed" hidden' in page.text
    assert 'class="data-agent-compose evaluator-agent-compose"' in page.text
    assert 'class="data-agent-compose-toolbar evaluator-agent-compose-toolbar"' in page.text
    assert 'class="data-agent-send evaluator-agent-send"' in page.text
    assert "feed.hidden=false;feed.append" in page.text
    assert '.evaluator-editor-toolbar button{display:inline-flex;align-items:center;justify-content:center;height:34px;padding:0 14px' in page.text
    assert "api('/api/evaluator/test',evaluatorPayload())" in page.text
    assert "api('/api/evaluator/agent',{message,source:$('evaluator-code').value})" in page.text
    assert "if(evaluatorDirty)await saveEvaluatorConfiguration()" in page.text
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
    assert "$('data-agent-send').before($('data-agent-state'))" in page.text
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
    assert '#composer>#prompt{padding-right:34px;resize:none}' in page.text
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
    assert "help.replaceChildren(document.createTextNode(_('dataAgentHelp')+' ('),link,document.createTextNode(')'))" in page.text
    assert "dataIngest.before(dataIngestIntro)" in page.text
    assert "dataIngest.append(dataDropCopy,$('data-upload'),$('data-file-input'))" in page.text
    assert "dataIngest.ondrop=dataGuard(" in page.text
    assert 'id="problem-description"' in data_view
    assert 'id="data-refresh"' in data_view
    assert 'id="save-data-selection"' not in data_view
    assert 'id="data-preview-card"' in data_insights
    assert 'class="data-card wide relationship-card"' in data_insights
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
    assert 'id="system-prompt-confirm-help">可在发送前编辑。清空以重新生成。</span>' in page.text
    assert '<span id="initial-prompt-toggle-label">系统提示词</span>' in page.text
    assert (
        "Find an interpretable formula explaining the selected target from the "
        "selected features."
    ) in data_view
    assert 'id="composer-purpose"' in page.text
    assert "promptEdited.user=Boolean($('prompt').value.trim())" in page.text
    assert "promptEdited.system=Boolean($('system-prompt').value.trim())" in page.text
    assert "if(!promptEdited.user)schedulePromptPreview()" in page.text
    assert "if(!promptEdited.system)schedulePromptPreview()" in page.text
    assert "system_prompt:$('system-prompt').value.trim()||promptDefaults.system_prompt" in page.text
    assert "showTabHint(_('runStartedHint'))" not in page.text
    assert "syncResearchProblem($('problem-description').value)" in page.text
    assert 'function renderTimelineSurface()' in page.text
    assert "composerPurposeTitle:'检查用户提示词'" in page.text
    assert "$('composer-purpose').hidden=active" in page.text
    assert "$('composer-purpose-title').textContent=_('composerPurposeTitle')" in page.text
    assert 'id="timeline-view-switch"' not in page.text
    assert 'function renderVariableRolePreview()' in page.text
    assert "variableOrder.filter(column=>variableRole(column)!=='unused')" in page.text
    assert "api('/api/data/selection'" in page.text
    assert 'function scheduleDataSelectionSave()' in page.text
    assert 'setTimeout(flushDataSelectionSave,400)' in page.text
    assert 'id="data-refresh"' in page.text
    assert 'id="data-file"' not in page.text
    assert 'id="data-max-turns"' not in page.text
    assert "api('/api/data/context')" in page.text
    assert 'async function responseError(response' in page.text
    assert 'function compactStreamEventBatch(events)' in page.text
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
    assert 'function unobservePreviewBlocks(parent)' in page.text
    assert 'id="up"' not in page.text
    assert 'id="path"' not in page.text
    assert ':root[data-theme="dark"] #tree-status' in page.text
    assert 'function applyPromptPreview(preview,systemOnly=false)' in page.text
    assert 'const hasPreparedData=Boolean(session?.supplied_data||useCommittedContext)' in page.text
    assert 'applyPromptPreview(preview,!hasPreparedData)' in page.text
    assert "if(s.state==='idle')refreshPromptPreview()" in page.text
    assert client.get('/viewer').status_code == 200
    initial_prompts = client.post('/api/data/prompts', json={})
    assert initial_prompts.status_code == 200
    assert 'Symbolic Regression Agent' in initial_prompts.json()['system_prompt']
    assert client.put('/api/workspace/upload?path=data/sample.csv', content=b'x,y\n1,2').status_code == 200
    assert client.get('/api/workspace/download?path=data/sample.csv').content == b'x,y\n1,2'
    assert client.get('/api/workspace?path=data').json()['entries'][0]['name'] == 'sample.csv'
    tree = client.get('/api/workspace', params={'recursive': True}).json()['entries']
    assert tree[0]['name'] == 'data'
    assert tree[0]['read_only'] is False
    assert tree[0]['size'] is None
    assert tree[0]['children'][0]['path'] == 'data/sample.csv'
    assert tree[0]['children'][0]['read_only'] is False
    assert client.get(
        '/api/workspace/size', params={'path': 'data'},
    ).json()['size'] == len(b'x,y\n1,2')
    demo = client.post('/api/data/demo').json()
    assert demo['path'] == 'context.data'
    assert (session.workspace / 'context.data' / 'manifest.json').is_file()
    preview = client.get('/api/data/context', params={'rows': 5}).json()
    assert preview['columns'] == ['sample', 'x1', 'x2', 'x3', 'y']
    assert preview['column_kinds'] == {
        'sample': 'axis', 'x1': 'variable', 'x2': 'variable',
        'x3': 'variable', 'y': 'variable',
    }
    assert preview['variables']['x3']['dtype'].startswith('<U')
    assert {row['x3'] for row in preview['data']} <= {'alpha', 'beta', 'gamma', 'delta'}
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
    session.context.commit_context_data(ContextDataLoader(directory).load())

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
    prompts = client.post('/api/data/prompts', json={
        'target': 'y', 'features': ['sample', 'x'],
        'variable_descriptions': {'sample': 'Sample index.'},
    })
    assert prompts.status_code == 200, prompts.text
    assert "Feature names: ['sample', 'x']" in prompts.json()['user_prompt']


def test_context_data_roles_accept_multidimensional_network_variables(platform):
    client, session = platform
    directory = session.workspace / 'context.data'
    directory.mkdir()
    (directory / 'manifest.json').write_text(json.dumps({
        'num_nodes': 2,
        'variables': {
                'theta': {
                    'file': 'theta.npy', 'description': 'Node phases.',
                    'axes': ['time', 'node'], 'structure': 'A',
            },
            'omega': {
                'file': 'omega.npy', 'description': 'Natural frequencies.',
                'axes': ['time', 'node'],
            },
            'A': {
                'file': 'A.npy', 'description': 'Directed edge list.',
                'axes': ['edge', 'endpoint'],
            },
                'dtheta_dt': {
                    'file': 'dtheta_dt.npy', 'description': 'Phase derivatives.',
                    'axes': ['time', 'node'], 'structure': 'A',
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
    session.context.commit_context_data(ContextDataLoader(directory).load())

    preview = client.get('/api/data/context').json()
    assert preview['columns'] == [
        'time', 'node', 'edge', 'endpoint',
        'theta', 'omega', 'A', 'dtheta_dt',
    ]
    assert preview['preview_columns'] == []
    assert preview['data'] == []

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
    loaded = ContextDataLoader(directory).load()
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

    # Reloading the full manifest restores every non-axis variable as a feature.
    session.context.commit_context_data(loaded)
    assert list(session.context.feature_names()) == ['x', 'z']

    session.state = 'running'
    monkeypatch.setattr(session.controller, 'status', lambda: {
        'paused': False, 'waiting_at_boundary': False,
    })
    blocked = client.put('/api/data/selection', json={
        'target': 'y', 'features': ['x', 'z'],
    })
    assert blocked.status_code == 409

    monkeypatch.setattr(session.controller, 'status', lambda: {
        'paused': True, 'waiting_at_boundary': True,
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
    with TestClient(create_app(tmp_path, controller=session.controller, session=session)) as client:
        root = client.get('/api/workspace').json()['entries']
        assert root[0]['name'] == 'source'
        assert root[0]['directory']
        recursive_root = client.get(
            '/api/workspace', params={'recursive': True},
        ).json()['entries']
        assert recursive_root[0]['read_only'] is True
        assert recursive_root[0]['mounted'] is True
        assert recursive_root[0]['locked'] is False
        assert recursive_root[0]['children'][0]['path'] == (
            'source/observations.csv'
        )
        assert recursive_root[0]['children'][0]['read_only'] is True
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
    assert {'commit_data', 'load_context_data', 'workspace_code_executor', 'read_skill'} <= {
        tool['name'] for tool in catalog['tools']
    }
    assert 'commit_data' in catalog['default_tools']
    assert 'load_context_data' in catalog['default_tools']
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
        'tools': ['workspace_shell', 'commit_data'],
        'skills': [],
        'proxy': '',
    })
    assert response.status_code == 200
    assert response.json()['data_agent_settings'] == {
        'llm_provider': 'openai',
        'llm_model': 'test-data-model',
        'tool_parser': 'json',
        'llm_max_tokens': 2048,
        'tools': ['workspace_shell', 'commit_data'],
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
    session.data_thread.join(2)
    assert not session.data_thread.is_alive()
    assert session.data_state == 'stopped'
    assert session.data_result['status'] == 'stopped'


def test_data_agent_model_test_checks_completion_and_tool_call(platform, monkeypatch):
    client, _ = platform

    class FakeAPI:
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
    configuration = client.get('/api/evaluator')
    assert configuration.status_code == 200
    assert configuration.json()['selected'] == 'default'
    assert {item['id'] for item in configuration.json()['evaluators']} == {
        'base', 'default', 'graph',
    }
    base = next(item for item in configuration.json()['evaluators'] if item['id'] == 'base')
    assert base['abstract'] is True
    assert 'class BaseEvaluator(ABC)' in base['source']
    evaluator_dir = Path(sr_harness.__file__).parent / 'evaluator'
    assert base['source'] == (evaluator_dir / 'base_evaluator.py').read_text()
    default = next(item for item in configuration.json()['evaluators'] if item['id'] == 'default')
    assert default['abstract'] is False
    assert 'class DefaultEvaluator(BaseEvaluator)' in default['source']
    assert default['source'] == (evaluator_dir / 'default_evaluator.py').read_text()
    assert configuration.json()['custom_template'] == (evaluator_dir / 'template_custom_evaluator.py').read_text()

    custom_source = '''from sr_harness import DefaultEvaluator

class CustomEvaluator(DefaultEvaluator):
    pass
'''
    configured = client.put('/api/evaluator', json={
        'selected': 'custom', 'source': custom_source,
    })
    assert configured.status_code == 200, configured.text
    assert configured.json()['selected'] == 'custom'
    assert type(session.context.evaluator).__name__ == 'CustomEvaluator'
    forbidden = client.put('/api/evaluator', json={
        'selected': 'custom',
        'source': 'import os\nclass CustomEvaluator(DefaultEvaluator):\n    pass\n',
    })
    assert forbidden.status_code == 400
    assert 'scientific SRHarness modules' in forbidden.text

    client.post('/api/data/demo')
    session.context.commit_context_data(
        ContextDataLoader(session.workspace / 'context.data').load(),
    )
    session.context.update_selection(target='y', features=['x1', 'x2'])
    tested = client.post('/api/evaluator/test', json={
        'selected': 'custom',
        'source': custom_source,
        'formula': "param('scale') * x1",
    })
    assert tested.status_code == 200, tested.text
    assert 'complexity' in tested.json()['result']['data_split_results']['train']['metrics']

    class RestrictedAPI:
        def __init__(self):
            self.requests = 0

        def __call__(self, messages, **kwargs):
            self.requests += 1

            def generate():
                if self.requests == 1:
                    call = ToolCall(
                        'update_evaluator', {'source': custom_source}, id='update-evaluator',
                    )
                    yield {
                        'content': '', 'tool_call': [call],
                        'message': {'role': 'assistant', 'content': ''},
                    }
                elif self.requests == 2:
                    call = ToolCall(
                        'evaluate_formula',
                        {'f': "param('scale') * x1", 'fit': True, 'show_diagnostics': False},
                        id='test-evaluator',
                    )
                    yield {
                        'content': '', 'tool_call': [call],
                        'message': {'role': 'assistant', 'content': ''},
                    }
                else:
                    yield {
                        'content': 'The evaluator is ready.',
                        'tool_call': [],
                        'message': {'role': 'assistant', 'content': 'The evaluator is ready.'},
                    }
                return {'usage': {'token': {}, 'price': {}}, 'contents': [], 'tool_calls': []}

            return APICallResult(generate())

    created = {}

    def create_api(provider, **kwargs):
        created.update(provider=provider, **kwargs)
        return RestrictedAPI()

    monkeypatch.setattr(BaseAPI, 'create', create_api)
    assisted = client.post('/api/evaluator/agent', json={
        'message': 'Check this evaluator.',
        'source': custom_source,
    })
    assert assisted.status_code == 200, assisted.text
    assert assisted.json()['message'] == 'The evaluator is ready.'
    assert [event['tool'] for event in assisted.json()['tool_events']] == [
        'update_evaluator', 'evaluate_formula',
    ]
    assert all(event['ok'] for event in assisted.json()['tool_events'])
    assert {tool.metadata.name for tool in created['tool_list']} == {
        'update_evaluator', 'evaluate_formula',
    }


def test_data_agent_proxy_setting_persists_to_env_file(platform, tmp_path, monkeypatch):
    client, session = platform
    session.env_path = tmp_path / '.env'
    for name in ('MY_PROXY', 'my_proxy', 'http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY'):
        monkeypatch.delenv(name, raising=False)

    response = client.put('/api/data/agent/settings', json={
        'proxy': 'http://127.0.0.1:7890',
    })
    assert response.status_code == 200, response.text
    assert dotenv_values(session.env_path)['MY_PROXY'] == 'http://127.0.0.1:7890'
    assert os.environ['HTTPS_PROXY'] == 'http://127.0.0.1:7890'

    response = client.put('/api/data/agent/settings', json={'proxy': ''})
    assert response.status_code == 200, response.text
    assert 'MY_PROXY' not in dotenv_values(session.env_path)
    assert 'MY_PROXY' not in os.environ
    assert 'HTTPS_PROXY' not in os.environ


def test_data_agent_commits_excel_to_shared_context(platform, monkeypatch):
    client, session = platform
    import pandas as pd

    configured = client.put('/api/data/agent/settings', json={
        'llm_provider': 'deepseek',
        'llm_model': 'data-preparation-model',
        'tool_parser': 'openai',
        'llm_max_tokens': 1234,
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
    assert session.context.target == '人口数量'
    assert list(session.context.feature_names()) == ['年份', 'GDP']
    data_events = session.controller.events()
    data_event_kinds = [event['kind'] for event in data_events]
    assert 'data_user' in data_event_kinds
    assert data_event_kinds.index('data_context') < data_event_kinds.index('data_assistant_start')
    assert data_event_kinds.index('data_assistant_start') < data_event_kinds.index('data_assistant')
    context_event = next(event for event in data_events if event['kind'] == 'data_context')
    assert context_event['payload']['turn'] == 1
    assert context_event['payload']['messages'][0]['role'] == 'system'
    assert 'data-preparation agent' in context_event['payload']['messages'][0]['content']
    assistant_event = next(event for event in data_events if event['kind'] == 'data_assistant')
    assert assistant_event['payload']['provider'] == 'deepseek'
    assert assistant_event['payload']['model'] == 'data-preparation-model'
    preview = client.get('/api/data/context').json()
    assert preview['revision'] == 1
    assert preview['rows'] == 5
    assert preview['data'][0]['年份'] == 2020


def test_data_agent_has_no_turn_limit_and_keeps_cumulative_turns(platform, monkeypatch):
    client, session = platform

    class FakeDataAPI:
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
        event for event in session.controller.events()
        if event['kind'] == 'data_context'
    ]
    assert [event['payload']['turn'] for event in context_events] == list(range(1, 16))
    assert context_events[-1]['payload']['messages'][-1] == {
        'role': 'user', 'content': 'Continue.',
    }


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
    controller.command('pause')
    status = controller.command('message', 'guidance')
    assert status['paused'] is False
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


def test_tool_free_search_response_waits_for_web_guidance(platform, monkeypatch):
    client, session = platform
    prompts = []

    class YieldingAPI:
        tool_description_json = []

        def __call__(self, prompt, **kwargs):
            prompts.append(prompt)
            turn = len(prompts)
            if turn == 2:
                session.controller.command('next_r')

            def generate():
                message = {'role': 'assistant', 'content': f'round {turn}'}
                yield {'content': message['content'], 'tool_call': [], 'message': message}
                return {'usage': {'token': {}, 'price': {}}, 'responses': []}

            return APICallResult(generate())

    monkeypatch.setattr(BaseAPI, 'create', lambda *args, **kwargs: YieldingAPI())
    response = client.post('/api/session/start', json={'max_refinement_depth': 3})
    assert response.status_code == 200, response.text
    deadline = time.monotonic() + 5
    while not session.controller.status()['questions'] and time.monotonic() < deadline:
        time.sleep(.01)
    questions = session.controller.status()['questions']
    assert len(questions) == 1
    question_id, question = next(iter(questions.items()))
    assert 'replied without calling a tool' in question

    reply = client.post(
        f'/api/control/reply/{question_id}', json={'message': 'Try a power law.'},
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
