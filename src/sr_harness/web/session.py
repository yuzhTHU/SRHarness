"""Single interactive run and observable hooks, without changing the search loop."""
from __future__ import annotations

import json
import ast
import os
import shutil
import tempfile
import threading
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import numpy as np
from dotenv import dotenv_values, set_key, unset_key

from ..agents.data_preparation_agent import DataPreparationAgent
from ..agents.evaluator_construction_agent import EvaluatorConstructionAgent
from ..agents.sr_agent_interactive import SRAgentInteractive
from ..api import BaseAPI
from ..core import AgentContext, SearchRunState, json_value, load_context_data
from ..evaluator import DefaultEvaluator, GraphEvaluator, load_custom_evaluator
from ..evaluator.load_custom_evaluator import (
    BUILTIN_EVALUATOR_CLASS_NAMES,
    CUSTOM_TEMPLATE,
    create_builtin_evaluator,
    evaluator_catalog,
    evaluator_filename,
    evaluator_source,
)
from ..runtime import InteractionManager, ModelRouter, SRInteractionManager
from ..skills import SkillManager
from ..tools import BaseTool, ModelTestTool, ValidateEvaluatorTool
from ..tools.workspace_shell import Workspace
from .demo_data import build_demo


RUNTIME_SETTING_NAMES = (
    "llm_provider", "llm_model", "strong_llm_provider", "strong_llm_model",
    "auto_routing", "tool_parser", "llm_max_tokens", "tools", "skills",
    "local_sample_size", "max_refinement_depth", "global_width",
    "max_restart_loop", "restart_top_k", "max_workers", "validation_fraction",
    "split_by", "split_ood_variable", "split_random_state", "ranking_metric", "larger_is_better",
    "force_initial_diagnostics",
)

DATA_AGENT_SETTING_NAMES = (
    "llm_provider", "llm_model", "tool_parser", "llm_max_tokens",
    "tools", "skills", "proxy",
)

EVALUATOR_CONTEXT_SETTING_NAMES = (
    "validation_fraction", "split_random_state", "split_by",
    "split_ood_variable", "ranking_metric", "larger_is_better",
)

HIDDEN_CAPABILITY_TOOLS = frozenset({"code_executor"})
EVALUATOR_DEFAULT_TOOLS = (
    "read_source", "workspace_shell", "workspace_code_executor",
    "validate_evaluator", "read_skill",
)

PROVIDER_API_KEY_VARIABLES = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "siliconflow": "SILICONFLOW_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "lmstudio": "LMSTUDIO_API_KEY",
}


class InteractiveSession:
    """Own one run for the lifetime of the server; no account/session registry."""
    def __init__(
        self,
        log_dir,
        agent_options=None,
        data=None,
        initial_prompt="",
        env_path=None,
        workspace_files=None,
        run_dir=None,
        workspace_path=None,
    ):
        self.sr_interaction_manager = SRInteractionManager()
        self.data_interaction_manager = InteractionManager()
        self.evaluator_interaction_manager = InteractionManager()
        self.lock = threading.RLock()
        self.model_test_lock = threading.Lock()
        self.evaluator_agent_lock = threading.Lock()
        self.run_id = uuid.uuid4().hex
        self.run_dir = (
            Path(run_dir).expanduser().resolve()
            if run_dir is not None
            else Path(log_dir).expanduser().resolve() / self.run_id
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_manager = Workspace(
            workspace_files=workspace_files,
            path=workspace_path,
        )
        self.workspace = self.workspace_manager.path
        self.evaluator_workspace = self.workspace / "context.evaluator"
        self.context = AgentContext(workspace=self.workspace_manager)
        self.context.args.save_path = str(self.run_dir)
        self._capability_skill_manager = SkillManager()
        self.context.args.skill_manager = self._capability_skill_manager
        self.env_path = Path(env_path or Path.cwd() / ".env").resolve()
        env_values = dotenv_values(self.env_path) if self.env_path.exists() else {}
        file_http_proxy = env_values.get("HTTP_PROXY") or env_values.get("http_proxy")
        file_https_proxy = env_values.get("HTTPS_PROXY") or env_values.get("https_proxy")
        if file_http_proxy:
            os.environ.setdefault("HTTP_PROXY", file_http_proxy)
        if file_https_proxy:
            os.environ.setdefault("HTTPS_PROXY", file_https_proxy)
        configured_proxy = (
            os.environ.get("HTTPS_PROXY")
            or os.environ.get("https_proxy")
            or os.environ.get("HTTP_PROXY")
            or os.environ.get("http_proxy")
            or file_https_proxy
            or file_http_proxy
            or ""
        )
        self.settings = {
            "llm_provider": "openrouter",
            "llm_model": "deepseek/deepseek-v4-flash-0731",
            "strong_llm_provider": None,
            "strong_llm_model": None,
            "auto_routing": True,
            "tool_parser": "openai",
            "llm_max_tokens": 4096,
            "tools": None,
            "skills": None,
            "max_refinement_depth": 50,
            "local_sample_size": 1,
            "global_width": 1,
            "max_restart_loop": 1,
            "restart_top_k": 1,
            "max_workers": 0,
            "validation_fraction": 0.2,
            "split_by": "random",
            "split_ood_variable": None,
            "split_random_state": 42,
            "ranking_metric": "mse",
            "larger_is_better": False,
            "force_initial_diagnostics": False,
        }
        self.settings.update(agent_options or {})
        if self.settings["tools"] is None:
            self.settings["tools"] = self.capabilities()["default_tools"]
        for name in EVALUATOR_CONTEXT_SETTING_NAMES:
            setattr(self.context.args, name, self.settings[name])
        self.data_agent_settings = {
            "llm_provider": self.settings["llm_provider"],
            "llm_model": self.settings["llm_model"],
            "tool_parser": self.settings["tool_parser"],
            "llm_max_tokens": self.settings["llm_max_tokens"],
            "tools": list(DataPreparationAgent.DEFAULT_TOOLS),
            "skills": [
                name for name in self._capability_skill_manager.load_skills()
                if name not in DataPreparationAgent.DEFAULT_EXCLUDED_SKILLS
            ],
            "proxy": configured_proxy,
        }
        self.evaluator_agent_settings = {
            key: self.data_agent_settings[key]
            for key in ("llm_provider", "llm_model", "tool_parser", "llm_max_tokens", "proxy")
        }
        self.evaluator_agent_settings.update({
            # This model reliably reaches native tool calls when constructing an
            # evaluator; the general SR agent keeps the user's configured model.
            "llm_provider": "openrouter",
            "llm_model": "qwen/qwen3.5-flash-02-23",
            "tools": list(EVALUATOR_DEFAULT_TOOLS),
            "skills": [
                name for name in self._capability_skill_manager.load_skills()
                if name != "discover-symbolic-laws"
            ],
            "llm_max_tokens": max(8192, self.data_agent_settings["llm_max_tokens"]),
        })
        self.pending_settings = None
        self.data = data
        self.initial_prompt = initial_prompt
        self.prompt_overrides = {}
        self.variable_descriptions = {}
        self.state = "idle"
        self.result = None
        self.run_state = None
        self.thread = None
        self.data_thread = None
        self.data_state = "idle"
        self.data_force_pause_requested = False
        self.data_result = None
        self.data_agent = None
        self.evaluator_agent = None
        self.evaluator_agent_thread = None
        self.evaluator_agent_state = "idle"
        self.evaluator_force_pause_requested = False
        self.evaluator_agent_result = None
        self.sr_agent = None
        self._restored_data_agent_buffer = None
        self._manifest_description_revision = None

    def close(self) -> None:
        """Release temporary resources owned by the session."""
        if not self.workspace_manager.retain:
            self.workspace_manager.cleanup()

    def interrupt_active_work(self) -> None:
        """Force active model/tool operations toward a safe shutdown boundary."""
        for manager in (
            self.sr_interaction_manager,
            self.data_interaction_manager,
            self.evaluator_interaction_manager,
        ):
            if manager.state in {"running", "pausing"}:
                manager.command("force_pause")

    def snapshot(self):
        """Return a serializable snapshot of the current session."""
        with self.lock:
            self._sync_manifest_descriptions()
            topk = (
                []
                if self.run_state is None
                else json_value([
                    record.display_dict()
                    for record in self.run_state.ranked_candidates()
                ])
            )
            return {"state": self.state, "settings": self.settings.copy(),
                    "pending_settings": self.pending_settings, "topk_records": topk,
                    "result": self.result, "run_id": self.run_id,
                    "initial_prompt": self.initial_prompt,
                    "supplied_data": self.data is not None,
                    "context_ready": bool(self.context.data),
                    "workspace": str(self.workspace), "data_state": self.data_state,
                    "data_interaction_state": self.data_interaction_manager.state,
                    "data_force_pause_requested": self.data_force_pause_requested,
                    "data_agent_settings": self.data_agent_settings.copy(),
                    "evaluator_agent_settings": self.evaluator_agent_settings.copy(),
                    "evaluator_agent_state": self.evaluator_agent_state,
                    "evaluator_interaction_state": self.evaluator_interaction_manager.state,
                    "evaluator_force_pause_requested": self.evaluator_force_pause_requested,
                    "evaluator_agent_result": json_value(self.evaluator_agent_result),
                    "evaluator_id": self._evaluator_metadata()["selected"],
                    "data_result": json_value(self.data_result),
                    "data_context": json_value(self.context.schema()),
                    **self.sr_interaction_manager.status()}

    def export_persistent_state(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot that can be restored in a new process."""
        with self.lock:
            data_agent_buffer = (
                list(self.data_agent.buffer)
                if self.data_agent is not None
                else self._restored_data_agent_buffer
            )
            return json_value({
                "version": 1,
                "run_id": self.run_id,
                "state": self.state,
                "settings": self.settings,
                "pending_settings": self.pending_settings,
                "data_agent_settings": self.data_agent_settings,
                "evaluator_agent_settings": self.evaluator_agent_settings,
                "initial_prompt": self.initial_prompt,
                "prompt_overrides": self.prompt_overrides,
                "variable_descriptions": self.variable_descriptions,
                "result": self.result,
                "data_state": self.data_state,
                "data_result": self.data_result,
                "data_agent_buffer": data_agent_buffer,
                "evaluator_agent_state": self.evaluator_agent_state,
                "evaluator_agent_result": self.evaluator_agent_result,
                "context_target": self.context.target,
                "evaluator": self._evaluator_metadata(),
                "run_state": self.run_state.export_state() if self.run_state is not None else None,
                "interactions": {
                    "search": self.sr_interaction_manager.export_state(),
                    "data": self.data_interaction_manager.export_state(),
                    "evaluator": self.evaluator_interaction_manager.export_state(),
                },
            })

    def restore_persistent_state(self, snapshot: dict[str, Any]) -> None:
        """Restore a durable snapshot and terminate operations lost on restart."""
        expected_fields = {
            "version", "run_id", "state", "settings", "pending_settings",
            "data_agent_settings", "evaluator_agent_settings", "initial_prompt",
            "prompt_overrides", "variable_descriptions", "result", "data_state",
            "data_result", "data_agent_buffer", "evaluator_agent_state",
            "evaluator_agent_result", "context_target", "evaluator", "run_state",
            "interactions",
        }
        if set(snapshot) != expected_fields:
            raise ValueError("Persisted InteractiveSession does not match the current schema")
        if snapshot["version"] != 1:
            raise ValueError("Unsupported InteractiveSession snapshot version")
        with self.lock:
            self.run_id = str(snapshot["run_id"])
            restored_settings = dict(snapshot["settings"])
            if set(restored_settings) != set(self.settings):
                raise ValueError("Persisted settings do not match the current session schema")
            self.settings = restored_settings
            self.pending_settings = snapshot["pending_settings"]
            restored_data_settings = dict(snapshot["data_agent_settings"])
            if set(restored_data_settings) != set(self.data_agent_settings):
                raise ValueError("Persisted data-agent settings do not match the current schema")
            self.data_agent_settings = restored_data_settings
            restored_evaluator_settings = dict(snapshot["evaluator_agent_settings"])
            if set(restored_evaluator_settings) != set(self.evaluator_agent_settings):
                raise ValueError("Persisted evaluator-agent settings do not match the current schema")
            self.evaluator_agent_settings = restored_evaluator_settings
            self.initial_prompt = str(snapshot["initial_prompt"])
            self.prompt_overrides = dict(snapshot["prompt_overrides"])
            if "system" in self.prompt_overrides:
                self.prompt_overrides["system"] = SRAgentInteractive.normalize_system_prompt(
                    self.prompt_overrides["system"]
                )
            self.variable_descriptions = dict(snapshot["variable_descriptions"])
            self.result = snapshot["result"]
            self.data_result = snapshot["data_result"]
            self.evaluator_agent_result = snapshot["evaluator_agent_result"]
            buffer = snapshot["data_agent_buffer"]
            if buffer is not None and not isinstance(buffer, list):
                raise TypeError("data_agent_buffer must be a list or None")
            self._restored_data_agent_buffer = list(buffer) if buffer is not None else None
            for name in EVALUATOR_CONTEXT_SETTING_NAMES:
                setattr(self.context.args, name, self.settings[name])

            data_directory = self.workspace / "context.data"
            if (data_directory / "manifest.json").is_file():
                self.context.commit_context_data(load_context_data(data_directory))
                target = snapshot["context_target"]
                if target is not None and target not in self.context.data:
                    raise ValueError("Persisted context target is missing from context.data")
                self.context.target = target
                if self.variable_descriptions:
                    self.context.variable_descriptions.update({
                        name: description
                        for name, description in self.variable_descriptions.items()
                        if name in self.context.data
                    })

            evaluator = snapshot["evaluator"]
            if not isinstance(evaluator, dict):
                raise TypeError("evaluator snapshot must be a dictionary")
            selected = evaluator["selected"]
            if isinstance(selected, str) and selected.startswith("custom"):
                custom_file = evaluator["custom_file"]
                custom_source = evaluator["source"]
                self.context.evaluator = load_custom_evaluator(
                    file=Path(custom_file) if custom_file is not None else None,
                    source=custom_source,
                )
            elif selected in {"default", "graph"}:
                self.context.evaluator = create_builtin_evaluator(selected)
            else:
                raise ValueError(f"Unknown persisted evaluator: {selected!r}")
            self.context.invalidate_splits()

            run_state = snapshot["run_state"]
            if run_state is not None and not isinstance(run_state, dict):
                raise TypeError("run_state must be a dictionary or None")
            self.run_state = SearchRunState.from_state(run_state, save_path=self.run_dir) if run_state is not None else None
            interactions = snapshot["interactions"]
            search_interrupted = self.sr_interaction_manager.restore_state(
                interactions["search"]
            )
            data_interrupted = self.data_interaction_manager.restore_state(
                interactions["data"]
            )
            evaluator_interrupted = self.evaluator_interaction_manager.restore_state(
                interactions["evaluator"]
            )
            saved_state = str(snapshot["state"])
            if saved_state not in {"idle", "starting", "running", "completed", "early_stopped", "interrupted", "failed"}:
                raise ValueError(f"Unknown persisted session state: {saved_state!r}")
            if search_interrupted or saved_state in {"starting", "running"}:
                self.state = "idle"
                previous_result = self.result if isinstance(self.result, dict) else {}
                self.result = {**previous_result, "status": "interrupted", "restored_after_restart": True}
            else:
                self.state = saved_state
            saved_data_state = str(snapshot["data_state"])
            if saved_data_state not in {"idle", "running", "stopping", "stopped", "completed", "failed"}:
                raise ValueError(f"Unknown persisted data-agent state: {saved_data_state!r}")
            self.data_state = "stopped" if data_interrupted or saved_data_state in {"running", "stopping"} else saved_data_state
            saved_evaluator_state = str(snapshot["evaluator_agent_state"])
            if saved_evaluator_state not in {"idle", "running", "stopping", "stopped", "completed", "failed"}:
                raise ValueError(f"Unknown persisted evaluator-agent state: {saved_evaluator_state!r}")
            self.evaluator_agent_state = (
                "stopped"
                if evaluator_interrupted or saved_evaluator_state in {"running", "stopping"}
                else saved_evaluator_state
            )
            self.data_force_pause_requested = False
            self.evaluator_force_pause_requested = False

    def _sync_manifest_descriptions(self) -> None:
        """Reflect externally edited manifest descriptions in the live context."""
        manifest_path = self.workspace / "context.data" / "manifest.json"
        if not self.context.data or not manifest_path.is_file():
            return
        stat = manifest_path.stat()
        revision = (stat.st_mtime_ns, stat.st_size)
        if revision == self._manifest_description_revision:
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            descriptions = {}
            for section in ("variables", "axes"):
                for name, spec in manifest[section].items():
                    description = spec["description"]
                    if not isinstance(description, str):
                        return
                    descriptions[name] = description.strip()
        except (KeyError, TypeError, json.JSONDecodeError, OSError):
            return
        changed = False
        for name in self.context.data:
            if name in descriptions and self.context.variable_descriptions[name] != descriptions[name]:
                self.context.variable_descriptions[name] = descriptions[name]
                changed = True
        self._manifest_description_revision = revision
        if changed:
            self.variable_descriptions = dict(self.context.variable_descriptions)
            self.context.args.data_revision += 1

    def evaluator_configuration(self):
        """Return the selected evaluator and every available evaluator."""
        metadata = self._evaluator_metadata()
        return {
            "selected": metadata["selected"],
            "source": metadata["source"],
            "custom_template": CUSTOM_TEMPLATE,
            "evaluators": [*evaluator_catalog(), *self._custom_evaluator_catalog()],
            "custom_name": metadata["custom_name"],
            "custom_file": metadata["custom_file"],
        }

    def _custom_evaluator_catalog(self) -> list[dict[str, Any]]:
        """Return valid evaluator implementations saved in this workspace."""
        if not self.evaluator_workspace.is_dir():
            return []
        catalog = []
        for file in sorted(self.evaluator_workspace.glob("*.py"), key=lambda path: path.name):
            try:
                source = file.read_text(encoding="utf-8")
                evaluator = load_custom_evaluator(file=file)
            except (OSError, UnicodeError, ValueError) as exc:
                try:
                    source = file.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    source = ""
                catalog.append({
                    "id": f"custom:{file.name}",
                    "label": file.stem,
                    "label_en": file.stem,
                    "source": source,
                    "abstract": False,
                    "custom": True,
                    "file": str(file),
                    "invalid": True,
                    "error": str(exc),
                })
                continue
            class_name = type(evaluator).__name__
            catalog.append({
                "id": f"custom:{file.name}",
                "label": class_name,
                "label_en": class_name,
                "source": file.read_text(encoding="utf-8"),
                "abstract": False,
                "custom": True,
                "file": str(file),
                "invalid": False,
                "error": None,
            })
        return catalog

    def _evaluator_metadata(self) -> dict[str, Any]:
        evaluator = self.context.evaluator
        custom = getattr(type(evaluator), "CUSTOM_EVALUATOR", None)
        if isinstance(custom, dict) and isinstance(custom.get("source"), str):
            file = custom.get("file")
            return {
                "selected": f"custom:{Path(file).name}" if file is not None else "custom",
                "source": custom["source"],
                "custom_name": type(evaluator).__name__,
                "custom_file": str(file) if file is not None else None,
            }
        selected = "graph" if type(evaluator) is GraphEvaluator else "default"
        return {
            "selected": selected, "source": evaluator_source(selected),
            "custom_name": None, "custom_file": None,
        }

    def evaluator_editable(self) -> bool:
        """Return whether evaluator mutation is safe for the active search."""
        if self.state == "idle":
            return True
        control = self.sr_interaction_manager.status()
        return (
            self.state in {"starting", "running"}
            and control["interaction_state"] == "paused"
        )

    def configure_evaluator(self, payload):
        """Validate and select an evaluator before symbolic regression starts."""
        if not self.evaluator_editable():
            raise ValueError("Pause the search and wait for a safe boundary before changing the evaluator")
        evaluator_id = str(payload.get("selected", "")).strip()
        source = str(payload.get("source", ""))
        available = {
            item["id"]: item
            for item in [*evaluator_catalog(), *self._custom_evaluator_catalog()]
        }
        selected_item = available.get(evaluator_id)
        builtin_source = (
            evaluator_source(evaluator_id)
            if evaluator_id in {item["id"] for item in evaluator_catalog()}
            else None
        )
        if builtin_source is not None and source == builtin_source:
            evaluator = create_builtin_evaluator(evaluator_id)
            source = builtin_source
            custom_name = custom_file = None
        elif selected_item is not None and selected_item.get("custom") and source == selected_item["source"]:
            evaluator = load_custom_evaluator(file=Path(selected_item["file"]))
            custom_name = type(evaluator).__name__
            custom_file = selected_item["file"]
        else:
            try:
                declared_names = {
                    node.name for node in ast.parse(source, mode="exec").body
                    if isinstance(node, ast.ClassDef)
                }
            except SyntaxError:
                declared_names = set()
            if conflicts := declared_names & BUILTIN_EVALUATOR_CLASS_NAMES:
                custom_name = sorted(conflicts)[0]
                raise ValueError(
                    f"Evaluator class name {custom_name!r} conflicts with a built-in evaluator. "
                    "Rename the class before saving."
                )
            evaluator = load_custom_evaluator(source=source)
            custom_name = type(evaluator).__name__
            if custom_name in BUILTIN_EVALUATOR_CLASS_NAMES:
                raise ValueError(
                    f"Evaluator class name {custom_name!r} conflicts with a built-in evaluator. "
                    "Rename the class before saving."
                )
            filename = evaluator_filename(custom_name)
            save_dir = self.evaluator_workspace
            save_dir.mkdir(parents=True, exist_ok=True)
            destination = save_dir / filename
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.stem}-", suffix=".py", dir=save_dir)
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                temporary.write_text(source, encoding="utf-8")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            evaluator = load_custom_evaluator(file=destination)
            evaluator_id = f"custom:{destination.name}"
            custom_file = str(destination)
        with self.lock:
            self.context.evaluator = evaluator
            self.context.invalidate_splits()
        return self.evaluator_configuration()

    def test_evaluator(self, payload):
        """Validate the selected evaluator against the best available formula."""
        evaluator_id = str(payload.get("selected", self._evaluator_metadata()["selected"])).strip()
        source = str(payload.get("source", ""))
        available = {
            item["id"]: item
            for item in [*evaluator_catalog(), *self._custom_evaluator_catalog()]
        }
        selected_item = available.get(evaluator_id)
        if selected_item is not None and selected_item.get("custom") and source == selected_item["source"]:
            evaluator = load_custom_evaluator(file=Path(selected_item["file"]))
        elif evaluator_id in {item["id"] for item in evaluator_catalog()} and source == evaluator_source(evaluator_id):
            evaluator = create_builtin_evaluator(evaluator_id)
        else:
            evaluator = load_custom_evaluator(source=source)
        if not self.context.data or not self.context.target or not self.context.feature_names():
            raise ValueError("Prepare data and select a target plus at least one feature first")
        test_context = AgentContext(
            args=self.context.args,
            data=self.context.data,
            target=self.context.target,
            variable_descriptions=self.context.variable_descriptions,
            variable_axes=self.context.variable_axes,
            variable_structures=self.context.variable_structures,
            num_nodes=self.context.num_nodes,
            evaluator=evaluator,
            workspace=self.context.workspace,
        )
        ranked = self.run_state.ranked_candidates() if self.run_state is not None else []
        best_formula = next((record.formula for record in ranked if record.formula), None)
        features = self.context.feature_names()
        baseline = " + ".join(
            ["param('intercept')"]
            + [f"param('coefficient_{index}') * {name}" for index, name in enumerate(features, 1)]
        )
        formula = str(payload.get("formula") or best_formula or baseline).strip()
        result = ValidateEvaluatorTool(context=test_context)(
            evaluator_file=None,
            f=formula,
            fit=True,
            show_diagnostics=False,
        )
        if not result.ok:
            raise ValueError(result.result_str)
        return {"formula": formula, "result": json_value(result.result)}

    def assist_evaluator(self, payload):
        """Let a restricted agent construct, exercise, and repair an evaluator."""
        if not self.evaluator_editable():
            raise ValueError("Pause the search and wait for a safe boundary before using EvaluatorConstructionAgent")
        instruction = str(payload.get("message", "")).strip()
        if not instruction:
            raise ValueError("Evaluator construction instruction must not be empty")
        instruction = (
            instruction
            + "\n\nKeep evaluator scripts inside context.evaluator/ and validate the final file. "
            "The directory may not exist yet; create it with a workspace tool only when you are "
            "ready to write the first evaluator script."
        )

        settings = self.evaluator_agent_settings.copy()
        enabled_tools = settings.get("tools") or []
        classes = {tool_cls.metadata.name: tool_cls for tool_cls in BaseTool.load_tool_classes()}
        tools = [classes[name](context=self.context) for name in enabled_tools]
        previous_skills = getattr(self.context.args, "enabled_skills", None)
        self.context.args.enabled_skills = settings.get("skills") or []
        try:
            with self.evaluator_agent_lock, self.temporary_proxy(settings.get("proxy", "")):
                agent = EvaluatorConstructionAgent(
                    llm_provider=settings["llm_provider"],
                    llm_model=settings["llm_model"],
                    context=self.context,
                    tools=tools,
                    tool_parser=settings["tool_parser"],
                    llm_max_tokens=settings["llm_max_tokens"],
                    interaction_manager=self.evaluator_interaction_manager,
                )
                with self.lock:
                    self.evaluator_agent = agent
                try:
                    result = agent.run(instruction)
                finally:
                    with self.lock:
                        if self.evaluator_agent is agent:
                            self.evaluator_agent = None
        finally:
            if previous_skills is None:
                delattr(self.context.args, "enabled_skills")
            else:
                self.context.args.enabled_skills = previous_skills
        files = sorted(self.evaluator_workspace.glob("*.py"), key=lambda path: path.stat().st_mtime_ns)
        if not files:
            raise ValueError(
                "EvaluatorConstructionAgent did not create a Python file under context.evaluator/"
            )
        result_file = files[-1]
        return {
            **result,
            "source": result_file.read_text(encoding="utf-8"),
            "selected": f"custom:{result_file.name}",
            "custom_file": str(result_file),
            "provider": settings["llm_provider"],
            "model": settings["llm_model"],
        }

    def start_evaluator_assistance(self, payload):
        """Start evaluator construction in a cancellable background thread."""
        with self.lock:
            if self.evaluator_interaction_manager.state in {"running", "pausing", "interrupting"}:
                raise ValueError("EvaluatorConstructionAgent is already running")
            instruction = str(payload.get("message", "")).strip()
            if not instruction:
                raise ValueError("Evaluator construction instruction must not be empty")
            if self.evaluator_interaction_manager.state == "paused":
                self.evaluator_interaction_manager.command("message", instruction)
                self.evaluator_agent_state = "running"
                self.evaluator_force_pause_requested = False
                return self.snapshot()
            self.evaluator_interaction_manager.command("message", instruction)
            self.evaluator_agent_state = "running"
            self.evaluator_force_pause_requested = False
            self.evaluator_agent_result = None
            self.evaluator_agent_thread = threading.Thread(
                target=self._run_evaluator_assistance,
                args=(dict(payload),),
                daemon=True,
            )
            self.evaluator_agent_thread.start()
        return self.snapshot()

    def stop_evaluator_assistance(self):
        """Request cancellation of the active evaluator-construction turn."""
        with self.lock:
            if self.evaluator_interaction_manager.state not in {"running", "pausing", "interrupting"}:
                raise ValueError("EvaluatorConstructionAgent is not running")
            force = self.evaluator_interaction_manager.state == "pausing"
            self.evaluator_interaction_manager.command("force_pause" if force else "pause")
            self.evaluator_agent_state = "stopping"
            self.evaluator_force_pause_requested = force
        return self.snapshot()

    def _run_evaluator_assistance(self, payload) -> None:
        try:
            queued = self.evaluator_interaction_manager.start_agent_execution()
            if queued:
                payload["message"] = queued[-1].content
            result = self.assist_evaluator(payload)
            state = "completed"
        except InterruptedError as exc:
            result = {"status": "stopped", "message": str(exc)}
            state = "stopped"
        except Exception as exc:
            result = {"status": "failed", "error": str(exc)}
            state = "failed"
        with self.lock:
            self.evaluator_agent_result = result
            self.evaluator_agent_state = state
            self.evaluator_force_pause_requested = False
        self.evaluator_interaction_manager.publish_event(
            "execution_completed" if state == "completed" else "execution_failed",
            {**json_value(result), "status": state},
        )
        if self.evaluator_interaction_manager.state != "paused":
            self.evaluator_interaction_manager.finish_agent_execution()

    def configure_evaluator_agent(self, payload):
        """Validate and persist evaluator-construction agent model settings."""
        settings = self.validate_evaluator_agent_settings(payload)
        self.validate_capabilities(settings, agent="evaluator")
        with self.lock:
            if "proxy" in settings:
                self.set_proxy(settings["proxy"])
            self.evaluator_agent_settings = {**self.evaluator_agent_settings, **settings}
            if "proxy" in settings:
                self.data_agent_settings["proxy"] = settings["proxy"]
        return self.snapshot()

    def test_evaluator_agent_model(self, payload):
        """Test evaluator-construction agent model settings without applying them."""
        updates = self.validate_evaluator_agent_settings(payload)
        self.validate_capabilities(updates, agent="evaluator")
        settings = {**self.evaluator_agent_settings, **updates}
        return self._test_model(settings, proxy=settings.get("proxy", ""))

    def capabilities(self, agent: str = "search"):
        """Describe configurable tools and user-facing skills for the Web UI.

        Args:
            agent: The agent value.
        """
        if agent not in {"search", "data", "evaluator"}:
            raise ValueError(f"Unsupported agent capability scope: {agent}")
        tools = [
            {
                "name": tool_cls.metadata.name,
                "description": tool_cls.metadata.description,
            }
            for tool_cls in BaseTool.load_tool_classes()
            if tool_cls.metadata.name not in HIDDEN_CAPABILITY_TOOLS
        ]
        skills = [
            {"name": skill.name, "description": skill.description}
            for skill in self._capability_skill_manager.load_skills().values()
        ]
        tool_names = [tool["name"] for tool in tools]
        if agent == "data":
            default_tools = list(DataPreparationAgent.DEFAULT_TOOLS)
            default_skills = [
                skill["name"] for skill in skills
                if skill["name"] not in DataPreparationAgent.DEFAULT_EXCLUDED_SKILLS
            ]
        elif agent == "evaluator":
            default_tools = list(EVALUATOR_DEFAULT_TOOLS)
            default_skills = [
                skill["name"] for skill in skills
                if skill["name"] != "discover-symbolic-laws"
            ]
        else:
            default_tools = [
                name for name in SRAgentInteractive.DEFAULT_TOOLS
                if name in tool_names
            ]
            default_skills = [skill["name"] for skill in skills]
        return {
            "tools": tools,
            "skills": skills,
            "default_tools": default_tools,
            "default_skills": default_skills,
        }

    def validate_capabilities(self, settings, agent: str = "search"):
        """Validate capabilities.

        Args:
            settings: Runtime settings to validate or apply.
            agent: The agent value.
        """
        catalog = self.capabilities(agent)
        for key, catalog_key in (("tools", "tools"), ("skills", "skills")):
            if key not in settings:
                continue
            available = {item["name"] for item in catalog[catalog_key]}
            if unknown := set(settings[key]) - available:
                raise ValueError(f"Unknown {key}: {', '.join(sorted(unknown))}")
        if agent == "data" and "tools" in settings and "validate_context_data" not in settings["tools"]:
            raise ValueError("The data-preparation agent requires the validate_context_data tool")
        if (
            agent == "evaluator"
            and settings.get("skills")
            and "read_skill" not in settings.get("tools", self.evaluator_agent_settings["tools"])
        ):
            raise ValueError("Evaluator-agent skills require the read_skill tool")

    def provider_credential(self, provider: str):
        """Report credential availability without exposing the secret value.

        Args:
            provider: The provider value.
        """
        provider = str(provider).strip().lower()
        try:
            variable = PROVIDER_API_KEY_VARIABLES[provider]
        except KeyError as exc:
            raise ValueError(f"Unsupported web model provider: {provider}") from exc
        file_value = dotenv_values(self.env_path).get(variable) if self.env_path.exists() else None
        configured = bool(file_value or os.environ.get(variable))
        return {
            "provider": provider,
            "env_var": variable,
            "configured": configured,
            "stored_in_env_file": bool(file_value),
        }

    def set_provider_credential(self, provider: str, api_key: str):
        """Atomically update the project dotenv file and this server process.

        Args:
            provider: The provider value.
            api_key: The api key value.
        """
        status = self.provider_credential(provider)
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a non-empty string")
        api_key = api_key.strip()
        if len(api_key) > 16_384 or any(character in api_key for character in "\r\n\0"):
            raise ValueError("api_key contains unsupported characters")
        with self.lock:
            self.env_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.env_path.exists():
                self.env_path.touch(mode=0o600)
            set_key(
                self.env_path,
                status["env_var"],
                api_key,
                quote_mode="always",
            )
            os.environ[status["env_var"]] = api_key
        return self.provider_credential(provider)

    def set_proxy(self, proxy: str) -> None:
        """Persist the optional model proxy and update this server process.

        Args:
            proxy: Proxy URL, or an empty string to clear the configured proxy.
        """
        proxy = self.validate_proxy(proxy)
        previous = str(self.data_agent_settings.get("proxy", ""))
        self.env_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.env_path.exists():
            self.env_path.touch(mode=0o600)
        if proxy:
            set_key(self.env_path, "HTTP_PROXY", proxy, quote_mode="always")
            set_key(self.env_path, "HTTPS_PROXY", proxy, quote_mode="always")
            for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
                os.environ[name] = proxy
        else:
            unset_key(self.env_path, "HTTP_PROXY")
            unset_key(self.env_path, "HTTPS_PROXY")
            for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
                if previous and os.environ.get(name) == previous:
                    os.environ.pop(name, None)

    @staticmethod
    def validate_proxy(proxy: str) -> str:
        """Validate and normalize an optional HTTP or SOCKS proxy URL.

        Args:
            proxy: Proxy URL supplied by the Web UI.

        Returns:
            The stripped URL, or an empty string when proxying is disabled.
        """
        if not isinstance(proxy, str):
            raise ValueError("proxy must be a string")
        proxy = proxy.strip()
        if not proxy:
            return ""
        if len(proxy) > 4096 or any(character in proxy for character in "\r\n\0"):
            raise ValueError("proxy contains unsupported characters")
        parsed = urlparse(proxy)
        if parsed.scheme not in {"http", "https", "socks5", "socks5h"} or not parsed.netloc:
            raise ValueError("proxy must be an HTTP, HTTPS, SOCKS5, or SOCKS5H URL")
        return proxy

    @contextmanager
    def temporary_proxy(self, proxy: str):
        """Temporarily expose a proxy to provider clients during a model test.

        Args:
            proxy: Validated proxy URL, or an empty string to disable proxying.

        Yields:
            Control while the temporary environment is active.
        """
        names = ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY")
        previous = {name: os.environ.get(name) for name in names}
        try:
            if proxy:
                for name in names:
                    os.environ[name] = proxy
            else:
                configured = str(self.data_agent_settings.get("proxy", ""))
                for name in names:
                    if configured and os.environ.get(name) == configured:
                        os.environ.pop(name, None)
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    @staticmethod
    def validate_setting_dependencies(settings):
        """Validate setting dependencies.

        Args:
            settings: Runtime settings to validate or apply.
        """
        if not settings.get("force_initial_diagnostics"):
            return
        tools = settings.get("tools")
        required = {"statistics_analysis", "relationship_analysis", "read_skill"}
        if tools is not None and (missing := required - set(tools)):
            raise ValueError(
                "force_initial_diagnostics requires enabled tools: "
                + ", ".join(sorted(missing))
            )
        skills = settings.get("skills")
        if skills is not None and "discover-symbolic-laws" not in skills:
            raise ValueError(
                "force_initial_diagnostics requires the discover-symbolic-laws skill"
            )

    def start(self, payload):
        """Run the ``start`` operation.

        Args:
            payload: Serializable event payload.
        """
        with self.lock:
            if self.state != "idle":
                raise ValueError("This server already owns a run. Restart it to begin a new task.")
            if self.data_interaction_manager.state in {"running", "pausing", "interrupting"}:
                raise ValueError("Wait for the data-preparation agent to finish before starting")
            options = self.validate_settings(payload, initial=True)
            self.validate_capabilities(options)
            updated_settings = {**self.settings, **options}
            self.validate_setting_dependencies(updated_settings)
            self.settings = updated_settings
            description = str(payload.get(
                "problem_description",
                payload.get("prompt", "Find an interpretable formula explaining the data."),
            ))
            self.prompt_overrides = {
                role: (
                    SRAgentInteractive.normalize_system_prompt(str(payload[key]))
                    if role == "system"
                    else str(payload[key])
                )
                for role, key in (("system", "system_prompt"), ("user", "user_prompt"))
                if key in payload
            }
            self.variable_descriptions = self.validate_variable_descriptions(payload)
            if self.data is not None:
                X, y = self.data
                if not isinstance(y, dict):
                    y = {"target": y}
            elif payload.get("dataset"):
                import pandas as pd
                path = self.resolve(str(payload["dataset"]))
                frame = pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
                target = str(payload.get("target", "y"))
                if target not in frame or len(frame.columns) < 2 or len(frame) < 5:
                    raise ValueError("CSV needs a target column, at least one feature, and 5 rows.")
                requested_features = payload.get("features")
                if requested_features is None:
                    features = [str(column) for column in frame if column != target]
                elif (
                    not isinstance(requested_features, list)
                    or not requested_features
                    or any(not isinstance(column, str) for column in requested_features)
                ):
                    raise ValueError("features must be a non-empty list of column names")
                else:
                    features = list(dict.fromkeys(requested_features))
                missing = [column for column in features if column not in frame]
                if missing or target in features:
                    raise ValueError("Features must exist in the CSV and must not include the target")
                try:
                    values = frame[[*features, target]].to_numpy(dtype=float)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Selected target and feature columns must be numeric") from exc
                if not np.isfinite(values).all():
                    raise ValueError("Selected columns must contain finite values without missing data")
                X = {column: frame[column].to_numpy(dtype=float) for column in features}
                y = {target: frame[target].to_numpy(dtype=float)}
            elif self.context.data:
                target = str(payload.get("target") or self.context.target)
                features = payload.get("features") or list(self.context.feature_names())
                X, y = self._select_context_columns(target, features)
                self.context.variable_descriptions = self.variable_descriptions.copy()
            else:
                rng = np.random.default_rng(42)
                x = rng.uniform(-2, 2, 100)
                X, y = {"x": x}, {"y": x*x + 2*x + 1}
                self.create_demo()
            self.state = "starting"
            self.thread = threading.Thread(target=self._run, args=(X, y, description), daemon=True)
            self.thread.start()
        return self.snapshot()

    def prepare_data(self, instruction: str):
        """Continue the persistent data-agent conversation in the background.

        Args:
            instruction: Natural-language instruction for the agent.
        """
        with self.lock:
            if self.data_state in {"running", "stopping"}:
                raise ValueError("The data-preparation agent is already running")
            if self.state == "starting":
                raise ValueError("Wait for symbolic regression to reach a controllable boundary")
            control_status = self.sr_interaction_manager.status()
            if self.state == "running" and not (
                control_status["interaction_state"] == "paused"
            ):
                raise ValueError(
                    "Pause symbolic regression and wait for the safe-boundary acknowledgement "
                    "before changing its data"
                )
            settings = self.data_agent_settings.copy()
            if self.data_interaction_manager.state == "paused":
                self.data_interaction_manager.command("message", instruction)
                self.data_state = "running"
                self.data_force_pause_requested = False
                return self.snapshot()
            if self.data_agent is None:
                self.data_agent = DataPreparationAgent(
                    llm_provider=settings["llm_provider"],
                    llm_model=settings["llm_model"],
                    context=self.context,
                    tools=settings["tools"],
                    tool_parser=settings["tool_parser"],
                    llm_max_tokens=settings["llm_max_tokens"],
                    skills=settings["skills"],
                    interaction_manager=self.data_interaction_manager,
                )
                if self._restored_data_agent_buffer:
                    self.data_agent.buffer = list(self._restored_data_agent_buffer)
                    self._restored_data_agent_buffer = None
            else:
                self.data_agent.llm_provider = settings["llm_provider"]
                self.data_agent.llm_model = settings["llm_model"]
                self.data_agent.tool_parser = settings["tool_parser"]
                self.data_agent.llm_max_tokens = settings["llm_max_tokens"]
                self.data_agent.skills = settings["skills"]
                self.data_agent.tool_cls_list = BaseTool.load_tool_classes(settings["tools"])
                self.data_agent.initialize_tools(self.context)
            self.data_interaction_manager.command("message", instruction)
            self.data_state = "running"
            self.data_force_pause_requested = False
            self.data_result = None
            self.data_thread = threading.Thread(
                target=self._prepare_data,
                args=(instruction,),
                daemon=True,
            )
            self.data_thread.start()
        return self.snapshot()

    def stop_data_preparation(self):
        """Request cancellation of the active data-preparation turn.

        Returns:
            Updated session state showing that cancellation is pending.
        """
        with self.lock:
            if self.data_interaction_manager.state not in {"running", "pausing", "interrupting"}:
                raise ValueError("The data-preparation agent is not running")
            force = self.data_interaction_manager.state == "pausing"
            self.data_state = "stopping"
            self.data_force_pause_requested = force
            self.data_interaction_manager.command("force_pause" if force else "pause")
        return self.snapshot()

    def _prepare_data(self, instruction: str) -> None:
        data_directory = self.workspace / "context.data"

        def directory_revision():
            if not data_directory.is_dir():
                return None
            return tuple(
                (str(path.relative_to(data_directory)), path.stat().st_size, path.stat().st_mtime_ns)
                for path in sorted(data_directory.rglob("*")) if path.is_file()
            )

        previous_revision = directory_revision()
        try:
            queued = self.data_interaction_manager.start_agent_execution()
            result = self.data_agent.run(queued[-1].content if queued else instruction)
            if directory_revision() != previous_revision and (data_directory / "manifest.json").is_file():
                self.context.commit_context_data(load_context_data(data_directory))
                result = {**result, "context": self.context.schema()}
            state = "completed"
        except InterruptedError as exc:
            result = {
                "status": "stopped",
                "message": str(exc),
                "context": self.context.schema(),
            }
            state = "stopped"
            self.data_interaction_manager.publish_event("execution_failed", result)
        except Exception as exc:
            result = {"error": str(exc), "context": self.context.schema()}
            state = "failed"
            self.data_interaction_manager.publish_event("execution_failed", {"error": str(exc)})
        with self.lock:
            self.data_result = json_value(result)
            self.data_state = state
            self.data_force_pause_requested = False
        if self.data_interaction_manager.state != "paused":
            self.data_interaction_manager.finish_agent_execution()

    def preview_initial_prompts(self, payload):
        """Run the ``preview initial prompts`` operation.

        Args:
            payload: Serializable event payload.
        """
        description = str(payload.get(
            "problem_description",
            self.initial_prompt or "Find an interpretable formula explaining the selected target from the selected features.",
        ))
        if self.data is not None:
            X, y = self.data
            if not isinstance(y, dict):
                y = {"target": y}
        elif payload.get("dataset"):
            import pandas as pd
            source = self.resolve(str(payload["dataset"]))
            frame = (
                pd.read_excel(source, nrows=1)
                if source.suffix.lower() == ".xlsx"
                else pd.read_csv(source, nrows=1)
            )
            target = str(payload.get("target", ""))
            requested_features = payload.get("features")
            if requested_features is None:
                features = [str(column) for column in frame if str(column) != target]
            elif (
                not isinstance(requested_features, list)
                or not requested_features
                or any(not isinstance(column, str) for column in requested_features)
            ):
                raise ValueError("features must be a non-empty list of column names")
            else:
                features = list(dict.fromkeys(requested_features))
            if (
                target not in frame
                or target in features
                or any(column not in frame for column in features)
            ):
                raise ValueError("Select a target and at least one existing feature")
            X = {str(column): np.empty(1) for column in features}
            y = {target: np.empty(1)}
        elif self.context.data:
            target = str(payload.get("target") or self.context.target or "")
            features = payload.get("features") or list(self.context.feature_names())
            X, y = self._select_context_columns(target, features)
        else:
            X, y = {"x": np.empty(1)}, {"y": np.empty(1)}
        settings = self.settings
        preview_agent = object.__new__(SRAgentInteractive)
        preview_agent.model_router = ModelRouter(
            enabled=bool(settings.get("auto_routing", False)),
            base_provider=settings["llm_provider"],
            base_model=settings["llm_model"],
            strong_provider=settings.get("strong_llm_provider"),
            strong_model=settings.get("strong_llm_model"),
        )
        preview_agent.max_refinement_depth = settings["max_refinement_depth"]
        preview_agent.use_workspace = True
        preview_agent.interaction_manager = InteractionManager()
        preview_agent.variable_descriptions = self.validate_variable_descriptions(payload)
        preview_agent.prompt_overrides = {}
        preview_agent.ranking_metric = settings.get("ranking_metric", "mse")
        preview_agent.larger_is_better = bool(settings.get("larger_is_better", False))
        preview_agent.run_state = SimpleNamespace(ranked_candidates=lambda: [], pareto_indices=lambda records: [])
        messages = preview_agent.create_initial_prompt_messages(description, X, y, [])
        return {
            "problem_description": description,
            "system_prompt": next(message["content"] for message in messages if message["role"] == "system"),
            "user_prompt": next(message["content"] for message in messages if message["role"] == "user"),
        }

    def _select_context_columns(
        self,
        target: str,
        features: Any,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if (
            not target
            or not isinstance(features, list)
            or not features
            or target in features
        ):
            raise ValueError("Select a target and at least one distinct feature")

        def array(name: str) -> np.ndarray | None:
            return self.context.data.get(name)

        selected = {name: array(name) for name in [*features, target]}
        missing = [name for name, value in selected.items() if value is None]
        if missing:
            raise ValueError(f"Selected variables or axes do not exist: {missing}")
        arrays = {name: np.asarray(value) for name, value in selected.items()}
        return (
            {name: arrays[name] for name in features},
            {target: arrays[target]},
        )

    @staticmethod
    def validate_variable_descriptions(payload):
        """Validate variable descriptions.

        Args:
            payload: Serializable event payload.
        """
        descriptions = payload.get("variable_descriptions", {})
        if not isinstance(descriptions, dict) or any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in descriptions.items()
        ):
            raise ValueError("variable_descriptions must map column names to text")
        return {
            name: value.strip()
            for name, value in descriptions.items()
            if value.strip()
        }

    def create_demo(self, kind: str = "polynomial"):
        """Create and load a manifest-backed sample dataset.

        Returns:
            Path to the newly created ``context.data`` directory.

        Raises:
            FileExistsError: If ``context.data`` already exists and is not empty.
        """
        with self.lock:
            path = self.workspace / "context.data"
            if path.exists() or path.is_symlink():
                if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
                    raise FileExistsError("context.data already exists and is not empty")
                path.rmdir()
            staging = Path(tempfile.mkdtemp(prefix=".context-data-", dir=self.workspace))
            try:
                arrays, manifest, target = build_demo(kind)
                for name, value in arrays.items():
                    np.save(staging / f"{name}.npy", value)
                for name, spec in manifest["variables"].items():
                    spec["file"] = f"{name}.npy"
                (staging / "manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8",
                )
                load_context_data(staging)
                staging.replace(path)
                self.context.commit_context_data(load_context_data(path))
                self.context.target = target
            finally:
                shutil.rmtree(staging, ignore_errors=True)
            return path

    def reload_context_data(self):
        """Reload ``context.data`` from the workspace into the shared context."""
        with self.lock:
            if self.data_state in {"running", "stopping"}:
                raise ValueError("Wait for the data-preparation agent to finish before reloading context.data")
            if self.state == "starting":
                raise ValueError("Wait for symbolic regression to reach a controllable boundary")
            control = self.sr_interaction_manager.status()
            if self.state == "running" and not (
                control["interaction_state"] == "paused"
            ):
                raise ValueError(
                    "Pause symbolic regression and wait for the safe-boundary acknowledgement "
                    "before reloading context.data"
                )
            directory = self.workspace / "context.data"
            if not (directory / "manifest.json").is_file():
                raise ValueError("context.data/manifest.json does not exist")
            change = self.context.commit_context_data(load_context_data(directory))
            self.variable_descriptions = dict(self.context.variable_descriptions)
            return {**change, "context": self.context.schema()}

    def _run(self, X, y, description):
        self.context.args.sr_active = True
        try:
            options = dict(self.settings)
            save_path = options.pop("save_path", str(self.run_dir))
            options.update(
                save_path=save_path,
                run_id=self.run_id,
                use_workspace=True,
            )
            manager = self.sr_interaction_manager
            agent = SRAgentInteractive(
                interaction_manager=manager,
                context=self.context,
                **options,
            )
            agent.prompt_overrides = self.prompt_overrides.copy()
            agent.variable_descriptions = self.variable_descriptions.copy()
            agent.runtime_settings_supplier = self._take_pending_settings
            agent.runtime_settings_committer = self._commit_runtime_settings
            self.sr_agent = agent
            self.run_state = agent.run_state
            with self.lock:
                self.state = "running"
            agent.initial_messages = manager.start_agent_execution()
            result = agent.run(X, y, description)
        except KeyboardInterrupt as exc:
            result = getattr(exc, "partial_result", {}) | {"status": "interrupted"}
        except Exception as exc:
            result = getattr(exc, "partial_result", {}) | {"status": "failed", "error": str(exc)}
        with self.lock:
            self.result = json_value(result)
            self.state = result.get("status", "completed")
            self.context.args.sr_active = False
        manager.publish_event("execution_completed" if self.state != "failed" else "execution_failed", {
            "status": self.state,
            "result": self.result,
        })
        if manager.state != "paused":
            manager.finish_agent_execution()

    def _take_pending_settings(self) -> dict[str, Any] | None:
        with self.lock:
            settings = self.pending_settings
            self.pending_settings = None
            return settings

    def _commit_runtime_settings(self, settings: dict[str, Any]) -> None:
        with self.lock:
            self.settings.update(settings)

    @staticmethod
    def validate_settings(payload, initial=False):
        """Validate settings.

        Args:
            payload: Serializable event payload.
            initial: Optional initial parameter values.
        """
        allowed = set(RUNTIME_SETTING_NAMES)
        bounded_integers = {
            "max_refinement_depth", "local_sample_size", "global_width",
            "max_restart_loop", "restart_top_k",
        }
        nonnegative_integers = {"max_workers"}
        result = {k: v for k, v in payload.items() if k in allowed}
        for k, v in result.items():
            if k in bounded_integers:
                if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 1000:
                    raise ValueError(f"{k} must be an integer between 1 and 1000")
            elif k == "llm_max_tokens":
                if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 1_000_000:
                    raise ValueError("llm_max_tokens must be an integer between 1 and 1000000")
            elif k in nonnegative_integers:
                if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 1000:
                    raise ValueError(f"{k} must be an integer between 0 and 1000")
            elif k == "split_random_state":
                if isinstance(v, bool) or not isinstance(v, int):
                    raise ValueError("split_random_state must be an integer")
            elif k == "validation_fraction":
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v < 1:
                    raise ValueError("validation_fraction must be in [0, 1)")
            elif k in {"auto_routing", "larger_is_better", "force_initial_diagnostics"}:
                if not isinstance(v, bool):
                    raise ValueError(f"{k} must be a boolean")
            elif k in {"tools", "skills"}:
                if (
                    not isinstance(v, list)
                    or any(not isinstance(item, str) or not item.strip() for item in v)
                    or len(v) != len(set(v))
                ):
                    raise ValueError(f"{k} must be a list of unique non-empty names")
            elif k in {"strong_llm_provider", "strong_llm_model"}:
                if v is not None and (not isinstance(v, str) or not v.strip()):
                    raise ValueError(f"{k} must be null or a non-empty string")
            elif k == "split_ood_variable":
                if v is not None and (not isinstance(v, str) or not v.strip()):
                    raise ValueError(f"{k} must be null or a non-empty string")
            elif not isinstance(v, str) or not v.strip():
                raise ValueError(f"{k} must be non-empty")
        if "llm_provider" in result and result["llm_provider"] not in {
            "openrouter", "openai", "deepseek", "siliconflow", "gemini", "lmstudio"
        }:
            raise ValueError("Unsupported web model provider")
        if "tool_parser" in result and result["tool_parser"] not in {
            "openai", "text", "json", "xml"
        }:
            raise ValueError("Unsupported tool parser")
        if "split_by" in result and result["split_by"] not in {"random", "ood"}:
            raise ValueError("Unsupported validation split strategy")
        return result

    @staticmethod
    def validate_data_agent_settings(payload):
        """Validate data agent settings.

        Args:
            payload: Serializable event payload.
        """
        result = {key: value for key, value in payload.items() if key in DATA_AGENT_SETTING_NAMES}
        for key, value in result.items():
            if key == "llm_max_tokens":
                upper = 1_000_000
                if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
                    raise ValueError(f"{key} must be an integer between 1 and {upper}")
            elif key in {"tools", "skills"}:
                if (
                    not isinstance(value, list)
                    or any(not isinstance(item, str) or not item.strip() for item in value)
                    or len(value) != len(set(value))
                ):
                    raise ValueError(f"{key} must be a list of unique non-empty names")
            elif key == "proxy":
                result[key] = InteractiveSession.validate_proxy(value)
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be non-empty")
        if "llm_provider" in result and result["llm_provider"] not in PROVIDER_API_KEY_VARIABLES:
            raise ValueError("Unsupported web model provider")
        if "tool_parser" in result and result["tool_parser"] not in {
            "openai", "text", "json", "xml"
        }:
            raise ValueError("Unsupported tool parser")
        return result

    @staticmethod
    def validate_evaluator_agent_settings(payload):
        """Validate the restricted evaluator-agent model settings."""
        allowed = {
            "llm_provider", "llm_model", "tool_parser", "llm_max_tokens",
            "proxy", "tools", "skills",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unsupported evaluator-agent settings: {', '.join(sorted(unknown))}")
        return InteractiveSession.validate_data_agent_settings(payload)

    def configure_data_agent(self, payload):
        """Run the ``configure data agent`` operation.

        Args:
            payload: Serializable event payload.
        """
        with self.lock:
            settings = self.validate_data_agent_settings(payload)
            self.validate_capabilities(settings, agent="data")
            if "proxy" in settings:
                self.set_proxy(settings["proxy"])
            self.data_agent_settings = {**self.data_agent_settings, **settings}
            if "proxy" in settings:
                self.evaluator_agent_settings["proxy"] = settings["proxy"]
        return self.snapshot()

    def test_data_agent_model(self, payload):
        """Test plain completion and tool-call support without changing agent history.

        Args:
            payload: Data-agent settings currently entered in the Web UI.

        Returns:
            Connectivity and parsed tool-call diagnostics.
        """
        updates = self.validate_data_agent_settings(payload)
        self.validate_capabilities(updates, agent="data")
        settings = {**self.data_agent_settings, **updates}
        return self._test_model(settings, proxy=settings.get("proxy", ""))

    def test_model(self, payload):
        """Test the symbolic-regression model settings without applying them.

        Args:
            payload: Runtime settings currently entered in the Web UI.

        Returns:
            Connectivity and parsed tool-call diagnostics.
        """
        updates = self.validate_settings(payload, initial=self.state == "idle")
        self.validate_capabilities(updates)
        settings = {**self.settings, **updates}
        self.validate_setting_dependencies(settings)
        return self._test_model(settings)

    def _test_model(self, settings, *, proxy=None):
        """Run the shared plain-response and tool-call model probes."""
        probe_tool = ModelTestTool(context=self.context)
        proxy_context = self.temporary_proxy(proxy) if proxy is not None else nullcontext()
        with self.model_test_lock, proxy_context:
            api = BaseAPI.create(
                settings["llm_provider"],
                model=settings["llm_model"],
                tool_list=[probe_tool],
                tool_parser_name=settings["tool_parser"],
            )
            plain_rows = list(api(
                "Reply with exactly SRHARNESS_OK.",
                n=1,
                max_tokens=min(settings["llm_max_tokens"], 64),
            ))
            tool_rows = list(api(
                "Call report_model_test with answer set to SRHARNESS_TOOL_OK. "
                "Use the tool instead of answering in plain text.",
                n=1,
                max_tokens=min(settings["llm_max_tokens"], 128),
            ))
        plain_response = plain_rows[0][0] if plain_rows else ""
        calls = [call for _, row_calls, _ in tool_rows for call in (row_calls or [])]
        expected_call = next(
            (
                call for call in calls
                if call.name == "report_model_test"
                and call.params.get("answer") == "SRHARNESS_TOOL_OK"
            ),
            None,
        )
        accessible = bool(plain_rows)
        tool_call_supported = expected_call is not None
        return {
            "ok": accessible and tool_call_supported,
            "accessible": accessible,
            "plain_response": plain_response,
            "tool_call_supported": tool_call_supported,
            "called_tools": [call.name for call in calls],
        }

    def configure(self, payload):
        """Run the ``configure`` operation.

        Args:
            payload: Serializable event payload.
        """
        with self.lock:
            if self.state not in {"idle", "starting", "running"}:
                raise ValueError("Run has finished")
            settings = self.validate_settings(payload, initial=self.state == "idle")
            self.validate_capabilities(settings)
            if self.state == "idle":
                updated = {**self.settings, **settings}
                self.validate_setting_dependencies(updated)
                self.settings = updated
                for name in EVALUATOR_CONTEXT_SETTING_NAMES:
                    if name in settings:
                        setattr(self.context.args, name, settings[name])
                if set(settings) & {
                    "validation_fraction", "split_random_state", "split_by",
                    "split_ood_variable",
                }:
                    self.context.invalidate_splits()
            else:
                current = {**self.settings, **(self.pending_settings or {})}
                pending = {key: current.get(key) for key in RUNTIME_SETTING_NAMES}
                pending.update(settings)
                self.validate_setting_dependencies(pending)
                self.pending_settings = pending
        return self.snapshot()

    def resolve(self, path, *, write: bool = False):
        """Resolve .

        Args:
            path: Filesystem path.
            write: Whether the caller intends to modify the path.

        Returns:
            The resolved readable or writable path.
        """
        candidate = self.workspace_manager.resolve(str(path), write=write)
        if candidate is None:
            raise ValueError("Path must stay inside the workspace")
        return candidate
