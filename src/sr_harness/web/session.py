"""Single interactive run and observable hooks, without changing the search loop."""
from __future__ import annotations

import csv
import os
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dotenv import dotenv_values, set_key

from ..agents.data_preparation_agent import DataPreparationAgent
from ..agents.sr_agent_interactive import SRAgentInteractive
from ..core import AgentContext, json_value
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
    "max_turns", "tools", "skills",
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
        controller=None,
        agent_options=None,
        data=None,
        initial_prompt="",
        env_path=None,
    ):
        self.controller = controller or InteractionController()
        self.lock = threading.RLock()
        self.run_id = uuid.uuid4().hex
        self.run_dir = Path(log_dir).resolve() / self.run_id
        self.workspace = self.run_dir / "workspace"
        self.workspace.mkdir(parents=True)
        self.context = AgentContext(workspace=Workspace(path=self.workspace))
        self._capability_skill_manager = SkillManager()
        self.env_path = Path(env_path or Path.cwd() / ".env").resolve()
        self.settings = {
            "llm_provider": "openrouter",
            "llm_model": "deepseek/deepseek-v4-flash",
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
            "max_turns": 12,
            "tools": list(DataPreparationAgent.DEFAULT_TOOLS),
            "skills": None,
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

    def snapshot(self):
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
        """Describe configurable tools and user-facing skills for the Web UI."""
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
        defaults = (
            list(DataPreparationAgent.DEFAULT_TOOLS)
            if agent == "data"
            else None
        )
        return {"tools": tools, "skills": skills, "default_tools": defaults}

    def validate_capabilities(self, settings, agent: str = "search"):
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
        """Report credential availability without exposing the secret value."""
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
        """Atomically update the project dotenv file and this server process."""
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

    @staticmethod
    def validate_setting_dependencies(settings):
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
        with self.lock:
            if self.state != "idle":
                raise ValueError("This server already owns a run. Restart it to begin a new task.")
            if self.data_state == "running":
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
            elif self.context.data and self.context.target and self.context.features:
                target = str(payload.get("target") or self.context.target)
                features = payload.get("features") or self.context.features
                if (
                    not isinstance(features, list)
                    or not features
                    or target in features
                    or any(name not in self.context.data for name in [target, *features])
                ):
                    raise ValueError("Select an existing target and at least one distinct feature")
                X = {name: self.context.data[name] for name in features}
                y = {target: self.context.data[target]}
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
        """Continue the persistent data-agent conversation in the background."""
        with self.lock:
            if self.data_state == "running":
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
                    max_turns=settings["max_turns"],
                    skills=settings["skills"],
                    interaction_manager=manager,
                )
            else:
                self.data_agent.llm_provider = settings["llm_provider"]
                self.data_agent.llm_model = settings["llm_model"]
                self.data_agent.tool_parser = settings["tool_parser"]
                self.data_agent.llm_max_tokens = settings["llm_max_tokens"]
                self.data_agent.max_turns = settings["max_turns"]
                self.data_agent.skills = settings["skills"]
                self.data_agent.tool_cls_list = BaseTool.load_tool_classes(settings["tools"])
                self.data_agent.initialize_tools(self.context)
            self.data_state = "running"
            self.data_result = None
            self.data_thread = threading.Thread(
                target=self._prepare_data,
                args=(instruction,),
                daemon=True,
            )
            self.data_thread.start()
        return self.snapshot()

    def _prepare_data(self, instruction: str) -> None:
        try:
            result = self.data_agent.run(instruction)
            state = "completed"
        except Exception as exc:
            result = {"error": str(exc), "context": self.context.schema()}
            state = "failed"
            self.controller.publish("data_error", {"error": str(exc)})
        with self.lock:
            self.data_result = json_value(result)
            self.data_state = state

    def preview_initial_prompts(self, payload):
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
        elif self.context.data and self.context.target and self.context.features:
            X = {name: self.context.data[name] for name in self.context.features}
            y = {self.context.target: self.context.data[self.context.target]}
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

    @staticmethod
    def validate_variable_descriptions(payload):
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
        with self.lock:
            path = self.workspace / "demo.csv"
            if not path.exists():
                rng = np.random.default_rng(42)
                x1 = rng.uniform(-2, 2, 100)
                x2 = rng.uniform(-1, 3, 100)
                x3 = np.tile(["alpha", "beta", "gamma", "delta"], 25)
                rng.shuffle(x3)
                y = x1*x1 + 2*x2 + 1
                with path.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["x1", "x2", "x3", "y"])
                    writer.writerows(zip(x1, x2, x3, y))
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
        result = {key: value for key, value in payload.items() if key in DATA_AGENT_SETTING_NAMES}
        for key, value in result.items():
            if key in {"llm_max_tokens", "max_turns"}:
                upper = 1_000_000 if key == "llm_max_tokens" else 1000
                if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
                    raise ValueError(f"{key} must be an integer between 1 and {upper}")
            elif key in {"tools", "skills"}:
                if (
                    not isinstance(value, list)
                    or any(not isinstance(item, str) or not item.strip() for item in value)
                    or len(value) != len(set(value))
                ):
                    raise ValueError(f"{key} must be a list of unique non-empty names")
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
        with self.lock:
            settings = self.validate_data_agent_settings(payload)
            self.validate_capabilities(settings, agent="data")
            self.data_agent_settings = {**self.data_agent_settings, **settings}
        return self.snapshot()

    def configure(self, payload):
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

    def resolve(self, path):
        root = self.workspace.resolve()
        candidate = (root / path).resolve()
        if Path(path).is_absolute() or not candidate.is_relative_to(root):
            raise ValueError("Path must stay inside the workspace")
        return candidate
