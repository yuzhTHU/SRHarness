"""Single interactive run and observable hooks, without changing the search loop."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import numpy as np
from dotenv import dotenv_values, set_key, unset_key

from ..agents.data_preparation_agent import DataPreparationAgent
from ..agents.sr_agent_interactive import SRAgentInteractive
from ..api import BaseAPI
from ..core import AgentContext, ContextDataStore, ToolMetadata, json_value
from ..interaction import InteractionManager, WebInteractionManager
from ..runtime import InteractionController
from ..interaction.web import add_variable_descriptions
from ..runtime import ModelRouter
from ..skills import SkillManager
from ..tools import BaseTool
from ..tools.workspace_shell import Workspace


RUNTIME_SETTING_NAMES = (
    "llm_provider", "llm_model", "strong_llm_provider", "strong_llm_model",
    "auto_routing", "tool_parser", "llm_max_tokens", "tools", "skills",
    "local_sample_size", "max_refinement_depth", "global_width",
    "max_restart_loop", "restart_top_k", "max_workers", "validation_fraction",
    "split_by", "split_random_state", "ranking_metric", "larger_is_better",
    "force_initial_diagnostics",
)

DATA_AGENT_SETTING_NAMES = (
    "llm_provider", "llm_model", "tool_parser", "llm_max_tokens",
    "tools", "skills", "proxy",
)

PROVIDER_API_KEY_VARIABLES = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "siliconflow": "SILICONFLOW_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "lmstudio": "LMSTUDIO_API_KEY",
}


class _ModelTestTool(BaseTool):
    """Private tool used to verify that a model can emit a parsed tool call."""

    metadata = ToolMetadata(
        name="report_model_test",
        description="Report the requested value to complete an SRHarness model test.",
        parameters={
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The exact value requested by the model-test prompt.",
                },
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    )

    def execute(self, answer: str):
        """Return the value supplied by the model-test request.

        Args:
            answer: Exact value requested by the test prompt.

        Returns:
            The supplied test value.
        """
        return {"answer": answer}


class InteractiveSession:
    """Own one run for the lifetime of the server; no account/session registry."""
    def __init__(
        self,
        log_dir,
        controller=None,
        agent_options=None,
        data=None,
        initial_prompt="",
        env_path=None,
        workspace_files=None,
        run_dir=None,
        workspace_path=None,
    ):
        self.controller = controller or InteractionController()
        self.lock = threading.RLock()
        self.model_test_lock = threading.Lock()
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
        self.context = AgentContext(workspace=self.workspace_manager)
        self._capability_skill_manager = SkillManager()
        self.context["skill_manager"] = self._capability_skill_manager
        self.env_path = Path(env_path or Path.cwd() / ".env").resolve()
        env_values = dotenv_values(self.env_path) if self.env_path.exists() else {}
        configured_proxy = (
            env_values.get("MY_PROXY")
            or os.environ.get("MY_PROXY")
            or os.environ.get("my_proxy")
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
            "split_by": "ood",
            "split_random_state": 42,
            "ranking_metric": "mse",
            "larger_is_better": False,
            "force_initial_diagnostics": False,
        }
        self.settings.update(agent_options or {})
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
        self.data_result = None
        self.data_agent = None
        self.sr_agent = None

    def close(self) -> None:
        """Release temporary resources owned by the session."""
        if not self.workspace_manager.retain:
            self.workspace_manager.cleanup()

    def snapshot(self):
        """Return a serializable snapshot of the current session."""
        with self.lock:
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
                    "data_agent_settings": self.data_agent_settings.copy(),
                    "data_result": json_value(self.data_result),
                    "data_context": json_value(self.context.schema()),
                    **self.controller.status()}

    def capabilities(self, agent: str = "search"):
        """Describe configurable tools and user-facing skills for the Web UI.

        Args:
            agent: The agent value.
        """
        if agent not in {"search", "data"}:
            raise ValueError(f"Unsupported agent capability scope: {agent}")
        excluded_tools = {"code_executor"}
        if agent == "search":
            excluded_tools.add("commit_data")
        tools = [
            {
                "name": tool_cls.metadata.name,
                "description": tool_cls.metadata.description,
            }
            for tool_cls in BaseTool.load_tool_classes()
            if tool_cls.metadata.name not in excluded_tools
        ]
        skills = [
            {"name": skill.name, "description": skill.description}
            for skill in self._capability_skill_manager.load_skills().values()
        ]
        default_tools = (
            list(DataPreparationAgent.DEFAULT_TOOLS)
            if agent == "data"
            else None
        )
        default_skills = (
            [
                skill["name"] for skill in skills
                if skill["name"] not in DataPreparationAgent.DEFAULT_EXCLUDED_SKILLS
            ]
            if agent == "data"
            else None
        )
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
        if agent == "data" and "tools" in settings and "commit_data" not in settings["tools"]:
            raise ValueError("The data-preparation agent requires the commit_data tool")

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
            set_key(self.env_path, "MY_PROXY", proxy, quote_mode="always")
            os.environ["MY_PROXY"] = proxy
            for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
                os.environ[name] = proxy
        else:
            unset_key(self.env_path, "MY_PROXY")
            os.environ.pop("MY_PROXY", None)
            os.environ.pop("my_proxy", None)
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
        names = ("MY_PROXY", "my_proxy", "http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY")
        previous = {name: os.environ.get(name) for name in names}
        try:
            if proxy:
                os.environ["MY_PROXY"] = proxy
                for name in names[2:]:
                    os.environ[name] = proxy
            else:
                os.environ.pop("MY_PROXY", None)
                os.environ.pop("my_proxy", None)
                configured = str(self.data_agent_settings.get("proxy", ""))
                for name in names[2:]:
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
            if self.data_state in {"running", "stopping"}:
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
                role: str(payload[key])
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
                features = payload.get("features") or self.context.features
                X, y = self._select_context_columns(target, features)
                self.context.variable_descriptions = self.variable_descriptions.copy()
            else:
                rng = np.random.default_rng(42)
                x = rng.uniform(-2, 2, 100)
                X, y = {"x": x}, {"y": x*x + 2*x + 1}
                self.create_demo()
            self.state = "starting"
            self.controller.publish("activity", {"phase": "initializing"})
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
            control_status = self.controller.status()
            if self.state == "running" and not (
                control_status["paused"] and control_status["waiting_at_boundary"]
            ):
                raise ValueError(
                    "Pause symbolic regression and wait for the safe-boundary acknowledgement "
                    "before changing its data"
                )
            settings = self.data_agent_settings.copy()
            if self.data_agent is None:
                manager = WebInteractionManager(self)
                self.data_agent = DataPreparationAgent(
                    llm_provider=settings["llm_provider"],
                    llm_model=settings["llm_model"],
                    context=self.context,
                    tools=settings["tools"],
                    tool_parser=settings["tool_parser"],
                    llm_max_tokens=settings["llm_max_tokens"],
                    skills=settings["skills"],
                    interaction_manager=manager,
                )
            else:
                self.data_agent.llm_provider = settings["llm_provider"]
                self.data_agent.llm_model = settings["llm_model"]
                self.data_agent.tool_parser = settings["tool_parser"]
                self.data_agent.llm_max_tokens = settings["llm_max_tokens"]
                self.data_agent.skills = settings["skills"]
                self.data_agent.tool_cls_list = BaseTool.load_tool_classes(settings["tools"])
                self.data_agent.initialize_tools(self.context)
            self.data_agent.reset_stop()
            self.data_state = "running"
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
            if self.data_state != "running" or self.data_agent is None:
                raise ValueError("The data-preparation agent is not running")
            self.data_state = "stopping"
            self.data_agent.request_stop()
        return self.snapshot()

    def _prepare_data(self, instruction: str) -> None:
        try:
            result = self.data_agent.run(instruction)
            state = "completed"
        except InterruptedError as exc:
            result = {
                "status": "stopped",
                "message": str(exc),
                "context": self.context.schema(),
            }
            state = "stopped"
            self.controller.publish("data_complete", result)
        except Exception as exc:
            result = {"error": str(exc), "context": self.context.schema()}
            state = "failed"
            self.controller.publish("data_error", {"error": str(exc)})
        with self.lock:
            self.data_result = json_value(result)
            self.data_state = state

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
            features = payload.get("features") or self.context.features
            X, y = self._select_context_columns(target, features)
        else:
            X, y = {"x": np.empty(1)}, {"y": np.empty(1)}
        settings = self.settings
        preview_agent = SimpleNamespace(
            model_router=ModelRouter(
                enabled=bool(settings.get("auto_routing", False)),
                base_provider=settings["llm_provider"],
                base_model=settings["llm_model"],
                strong_provider=settings.get("strong_llm_provider"),
                strong_model=settings.get("strong_llm_model"),
            ),
            max_refinement_depth=settings["max_refinement_depth"],
            use_workspace=True,
            tools=[],
            interaction_manager=InteractionManager(),
        )
        messages = SRAgentInteractive.build_initial_prompt(
            preview_agent, description, X, y, [],
        )
        add_variable_descriptions(
            messages, self.validate_variable_descriptions(payload), [*X, *y],
        )
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
            if name in self.context.data:
                return self.context.data[name]
            axis = self.context.axes.get(name)
            return None if axis is None else axis.values

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

    def create_demo(self):
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
                rng = np.random.default_rng(42)
                x1 = rng.uniform(-2, 2, 100)
                x2 = rng.uniform(-1, 3, 100)
                x3 = np.tile(["alpha", "beta", "gamma", "delta"], 25)
                rng.shuffle(x3)
                y = x1*x1 + 2*x2 + 1
                for name, value in {"x1": x1, "x2": x2, "x3": x3, "y": y}.items():
                    np.save(staging / f"{name}.npy", value)
                manifest = {
                    "variables": {
                        "x1": {
                            "file": "x1.npy",
                            "description": "First numeric input sampled uniformly from -2 to 2.",
                            "axes": ["sample"],
                        },
                        "x2": {
                            "file": "x2.npy",
                            "description": "Second numeric input sampled uniformly from -1 to 3.",
                            "axes": ["sample"],
                        },
                        "x3": {
                            "file": "x3.npy",
                            "description": "Categorical label with four string values.",
                            "axes": ["sample"],
                        },
                        "y": {
                            "file": "y.npy",
                            "description": "Synthetic target defined as x1 squared plus 2 times x2 plus 1.",
                            "axes": ["sample"],
                        },
                    },
                    "axes": {
                        "sample": {
                            "size": 100,
                            "description": "Sample index.",
                        },
                    },
                }
                (staging / "manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8",
                )
                ContextDataStore(staging).load()
                staging.replace(path)
                self.context.commit_context_data(ContextDataStore(path).load())
            finally:
                shutil.rmtree(staging, ignore_errors=True)
            return path

    def _run(self, X, y, description):
        self.context["sr_active"] = True
        try:
            options = dict(self.settings)
            save_path = options.pop("save_path", str(self.run_dir))
            options.update(
                save_path=save_path,
                run_id=self.run_id,
                use_workspace=True,
            )
            options.setdefault("max_workers", 0)
            manager = WebInteractionManager(self)
            agent = SRAgentInteractive(
                interaction_manager=manager,
                context=self.context,
                **options,
            )
            self.sr_agent = agent
            with self.lock:
                self.state = "running"
            self.controller.publish("lifecycle", {"state": "running"})
            result = agent.run(X, y, description)
        except KeyboardInterrupt as exc:
            result = getattr(exc, "partial_result", {}) | {"status": "interrupted"}
        except Exception as exc:
            result = getattr(exc, "partial_result", {}) | {"status": "failed", "error": str(exc)}
        with self.lock:
            self.result = json_value(result)
            self.state = result.get("status", "completed")
            self.context["sr_active"] = False
        self.controller.publish("lifecycle", {"state": self.state, "result": self.result})
        self.controller.publish("activity", {"phase": self.state})

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
        probe_tool = _ModelTestTool(context=self.context)
        with self.model_test_lock, self.temporary_proxy(settings.get("proxy", "")):
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
