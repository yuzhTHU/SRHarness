# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""人机协同符号回归 Agent。

继承自 SRAgent，以 L（对话轮次）为搜索主体，支持人类实时干预和工作区文件操作。
"""
from __future__ import annotations
import logging
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional

from ..api import BaseAPI
from ..core import AgentContext, CandidateRecord, ToolCall, ToolCallResult
from ..runtime import PendingMessage, SRInteractionManager
from .sr_agent import Message, ModelResponse, SRAgent, Usage
from ..parser import BaseParser

_logger = logging.getLogger(f'sr_harness.{__name__}')


class SRAgentInteractive(SRAgent):
    """Interactive symbolic-regression agent controlled by an interaction manager."""

    _LEGACY_WORKSPACE_GUIDANCE = (
        "The structured arrays are already loaded into the scientific tools; analyze them "
        "there rather than reconstructing them from workspace files. The workspace contains "
        "supplemental files and reproducible artifacts."
    )
    _TOOL_USAGE_GUIDANCE = (
        "Use only the tools available in the current request. Do not announce a tool call "
        "without making it. If a needed capability is unavailable, explain the limitation "
        "and continue with the available scientific tools."
    )

    # Construction and frontend event bridge

    def __init__(
        self,
        llm_provider: str,
        llm_model: str,
        interaction_manager: SRInteractionManager,
        tools: list[str] | None = None,
        skills: List[str] | None = None,
        verbose: bool = False,
        tool_parser: str | BaseParser = 'openai',
        save_path: Optional[str] = None,
        run_id: str | None = None,
        local_sample_size: int = 1,
        max_refinement_depth: int = 50,
        global_width: int = 1,
        max_restart_loop: int = 1,
        restart_top_k: int = 1,
        llm_max_tokens: int = 4096,
        max_workers: int = 0,
        validation_fraction: float = 0.2,
        split_by: str = "random",
        split_ood_variable: str | None = None,
        split_random_state: int = 42,
        ranking_metric: str = "mse",
        larger_is_better: bool = False,
        use_workspace: bool = False,
        workspace_files: List[str | Path] | None = None,
        force_initial_diagnostics: bool = False,
        auto_routing: bool = True,
        strong_llm_provider: str | None = None,
        strong_llm_model: str | None = None,
        context: AgentContext | None = None,
    ) -> None:
        """初始化 SRAgentInteractive。

        Args:
            llm_provider: LLM 提供商名称。
            llm_model: 模型名称。
            tools: 可用工具名列表。None 表示使用 ``SRAgent.DEFAULT_TOOLS``。
            skills: 可供 Agent 读取的 skill 名称。None 表示使用全部 skill。
            verbose: 是否启用详细日志。
            tool_parser: 工具解析器类型。
            save_path: 日志保存路径。
            run_id: 本次运行的全局唯一标识。None 表示自动生成。
            local_sample_size: 每轮 LLM 采样数量（K）。
            max_refinement_depth: 最大对话轮次（L），也是搜索深度上限。
            global_width: 独立分支数量（C）。
            max_restart_loop: 重启次数（R）。
            restart_top_k: 重启时注入历史最佳结果数量。
            llm_max_tokens: 每次模型响应允许生成的最大 token 数。
            max_workers: 并行工作进程数（0 表示不并行）。
            validation_fraction: 验证集比例。
            split_by: 验证集划分方式，可选 "random" 或 "ood"。
            split_ood_variable: OOD 划分使用的变量名；随机划分时可留空。
            split_random_state: 数据划分随机种子。
            ranking_metric: 候选公式排序所用的指标键；默认使用 mse。
            larger_is_better: 排序指标是否越大越好；默认按越小越好排序。
            use_workspace: 是否使用工作区。
            workspace_files: 初始化到工作区的文件/目录路径列表。
            interaction_manager: 连接 Agent 与 Web、终端等交互界面的管理器。
            force_initial_diagnostics: 是否在每个分支开始时强制执行初始诊断。
            auto_routing: 是否根据任务复杂度在基础与强模型后端之间自动路由。
            strong_llm_provider: 复杂任务使用的后端；默认沿用 llm_provider。
            strong_llm_model: 复杂任务使用的模型。None 表示仅使用基础模型。
            context: 与数据准备 Agent 共享的数据和工作区上下文。
        """
        super().__init__(
            llm_provider=llm_provider,
            llm_model=llm_model,
            tools=tools,
            skills=skills,
            verbose=verbose,
            tool_parser=tool_parser,
            save_path=save_path,
            run_id=run_id,
            local_sample_size=local_sample_size,
            max_refinement_depth=max_refinement_depth,
            global_width=global_width,
            max_restart_loop=max_restart_loop,
            restart_top_k=restart_top_k,
            llm_max_tokens=llm_max_tokens,
            max_workers=max_workers,
            validation_fraction=validation_fraction,
            split_by=split_by,
            split_ood_variable=split_ood_variable,
            split_random_state=split_random_state,
            ranking_metric=ranking_metric,
            larger_is_better=larger_is_better,
            force_initial_diagnostics=force_initial_diagnostics,
            auto_routing=auto_routing,
            strong_llm_provider=strong_llm_provider,
            strong_llm_model=strong_llm_model,
            context=context,
        )

        # 工作区
        self.use_workspace = use_workspace
        self.workspace_files = workspace_files

        # 交互界面
        if not isinstance(interaction_manager, SRInteractionManager):
            raise TypeError("interaction_manager must be an SRInteractionManager")
        self.interaction_manager = interaction_manager
        self._last_iteration_had_tool_calls = True
        self._forced_interruption_pending = False
        self.initial_messages: list[PendingMessage] = []
        self.prompt_overrides: dict[str, str] = {}
        self.variable_descriptions: dict[str, str] = {}
        self.runtime_settings_supplier: Callable[[], dict[str, Any] | None] | None = None
        self.runtime_settings_committer: Callable[[dict[str, Any]], None] | None = None
        self._bind_tool_cancellation()

    def on_buffer_messages_added(self, messages: list[Message], **coordinate: int) -> None:
        """Publish each system or user message when it enters the buffer.

        Args:
            messages: Newly appended messages.
            **coordinate: Optional R/C/L search coordinate.
        """
        for message in messages:
            if message.get("role") not in {"system", "user"}:
                continue
            payload: dict[str, Any] = {"message": message}
            if coordinate:
                payload["coord"] = coordinate
            self.interaction_manager.publish_event("prompt_added", payload)

    # Runtime settings and tool context

    def _take_runtime_settings(self) -> dict[str, Any] | None:
        return self.runtime_settings_supplier() if self.runtime_settings_supplier else None

    def _commit_runtime_settings(self, settings: dict[str, Any]) -> None:
        if self.runtime_settings_committer is not None:
            self.runtime_settings_committer(settings)

    def _bind_tool_cancellation(self) -> None:
        """Share the hard-interrupt event with every active tool instance."""
        for tool in self.tools or []:
            tool.cancel_event = self.interaction_manager.cancellation_signal

    def initialize_tools(self, context: AgentContext) -> None:
        """Initialize tools and attach the active hard-interrupt event."""
        super().initialize_tools(context)
        if hasattr(self, "interaction_manager"):
            self._bind_tool_cancellation()

    @contextmanager
    def prepare_tool_context(self, tool_context: AgentContext) -> Iterator[AgentContext]:
        """Add interaction resources to the tool context for the duration of a run.

        Args:
            tool_context: Shared context used to initialize tools.

        Yields:
            Context enriched with a temporary workspace manager when enabled.
        """
        if not self.use_workspace:
            yield tool_context
            return

        workspace_manager = getattr(tool_context.args, "workspace_manager", None)
        if workspace_manager is not None:
            yield tool_context
            return

        from ..runtime.workspace import Workspace

        workspace = Workspace(tempfile.mkdtemp(prefix="sr_workspace_"))
        for source in self.workspace_files or ():
            workspace.mount(source)
        _logger.note(f"Workspace initialized at: {workspace.path}")
        tool_context.workspace = workspace.path
        tool_context.args.workspace_manager = workspace
        yield tool_context

    # Interactive search-boundary hooks

    def prepare_iteration(self, buffer: list[Message], R: int, L: int, C: int) -> str | None:
        """Apply queued human guidance before the prompt is constructed.

        Args:
            buffer: Conversation history buffer.
            R: One-based restart index.
            L: One-based refinement-step index.
            C: One-based conversation-branch index.

        Returns:
            str | None: The operation result.
        """
        self.interaction_manager.publish_event("search_position_changed", {"R": R, "C": C, "L": L})
        with self.interaction_manager.wait() as boundary_messages:
            messages = [*self.initial_messages, *boundary_messages]
            self.initial_messages.clear()
            if data_change := self.refresh_data():
                self._append_buffer_messages(
                    buffer,
                    [self.build_data_refresh_message(data_change)],
                    R=R,
                    C=C,
                    L=L,
                )
            for message in messages:
                prompt = {
                    "role": "user",
                    "content": f"[Human guidance injected during the run]\n{message.content}",
                }
                self._append_buffer_messages(buffer, [prompt], R=R, C=C, L=L)
        if settings := self._take_runtime_settings():
            self._apply_runtime_settings(settings)
        return self.interaction_manager.consume_search_transition()

    @staticmethod
    def build_data_refresh_message(change: dict[str, Any]) -> dict[str, str]:
        """Turn a structured data revision into guidance for the next model turn."""
        descriptions = change["variable_descriptions"]
        description_text = (
            "\nVariable descriptions:\n"
            + "\n".join(f"- {name}: {description}" for name, description in descriptions.items())
            if descriptions else ""
        )
        return {
            "role": "user",
            "content": (
                "[Structured data updated by the data-preparation agent]\n"
                f"Data revision changed from {change['previous_revision']} to {change['revision']}. "
                f"The target is {change['target']!r}; available features are {change['features']}. "
                "Reassess earlier evidence against the updated variables and continue the investigation."
                f"{description_text}"
            ),
        }

    def _apply_runtime_settings(self, settings: dict[str, Any]) -> None:
        """Apply frontend-requested model and capability changes at a safe boundary."""
        try:
            tool_names = settings.get("tools")
            requested_tools = (
                set(self.available_tool_classes) if tool_names is None else set(tool_names)
            )
            if unknown_tools := requested_tools - self.available_tool_classes.keys():
                raise ValueError(f"Unknown tools: {', '.join(sorted(unknown_tools))}")
            self.tool_cls_list = [
                tool_cls for name, tool_cls in self.available_tool_classes.items()
                if name in requested_tools
            ]
            registered_skills = self.skill_manager.load_skills()
            self.skill_manager.register_tool_docs([
                tool_cls for tool_cls in self.tool_cls_list
                if (doc := tool_cls.get_doc()) is not None
                and doc["name"] not in registered_skills
            ])
            available_skills = self.skill_manager.load_skills()
            skill_names = settings.get("skills")
            requested_skills = (
                set(available_skills) if skill_names is None else set(skill_names)
            )
            if unknown_skills := requested_skills - available_skills.keys():
                raise ValueError(f"Unknown skills: {', '.join(sorted(unknown_skills))}")
            requested_skills.update(
                doc["name"]
                for tool_cls in self.tool_cls_list
                if (doc := tool_cls.get_doc()) is not None
            )
            self.enabled_skills = requested_skills
            self.llm_provider = settings["llm_provider"]
            self.llm_model = settings["llm_model"]
            for name in (
                "local_sample_size",
                "max_refinement_depth",
                "global_width",
                "max_restart_loop",
                "restart_top_k",
                "llm_max_tokens",
                "max_workers",
                "validation_fraction",
                "split_by",
                "split_ood_variable",
                "split_random_state",
                "ranking_metric",
                "larger_is_better",
                "force_initial_diagnostics",
                "auto_routing",
                "tool_parser",
            ):
                if name in settings:
                    setattr(self, name, settings[name])
            self.strong_llm_provider = (
                self.llm_provider
                if settings.get("strong_llm_provider") is None
                else settings["strong_llm_provider"]
            )
            self.strong_llm_model = settings.get("strong_llm_model")
            if self.force_initial_diagnostics:
                required_tools = {"statistics_analysis", "relationship_analysis", "read_skill"}
                if missing := required_tools - requested_tools:
                    raise ValueError(
                        "force_initial_diagnostics requires enabled tools: "
                        + ", ".join(sorted(missing))
                    )
                if "discover-symbolic-laws" not in self.enabled_skills:
                    raise ValueError(
                        "force_initial_diagnostics requires the discover-symbolic-laws skill"
                    )
            self.context.args.validation_fraction = self.validation_fraction
            self.context.args.split_by = self.split_by
            self.context.args.split_ood_variable = self.split_ood_variable
            self.context.args.split_random_state = self.split_random_state
            self.context.invalidate_splits()
            self.context.args.llm_provider = self.llm_provider
            self.context.args.llm_model = self.llm_model
            self.context.args.llm_max_tokens = self.llm_max_tokens
            self.context.args.enabled_skills = sorted(self.enabled_skills)
            self.initialize_tools(self.context)
            self._bind_tool_cancellation()
            self.model_router.base_provider = self.llm_provider
            self.model_router.base_model = self.llm_model
            self.model_router.enabled = self.auto_routing
            self.model_router.strong_provider = self.strong_llm_provider
            self.model_router.strong_model = self.strong_llm_model
            self.run_state.ranking_metric = self.ranking_metric
            self.run_state.larger_is_better = self.larger_is_better
            self._strong_api = None
            self._commit_runtime_settings(settings)
            self.interaction_manager.publish_event("settings_applied", settings)
        except Exception as exc:
            self.interaction_manager.publish_event("settings_failed", {"error": str(exc)})

    def finish_iteration(self, buffer: list[Message], R: int, L: int, C: int) -> str | None:
        """Keep interactive runs open and yield tool-free responses to the human.

        Args:
            buffer: Conversation history buffer.
            R: One-based restart index.
            L: One-based refinement-step index.
            C: One-based conversation-branch index.

        Returns:
            str | None: The operation result.
        """
        if self._forced_interruption_pending or self.interaction_manager.is_interrupting:
            prompt = {"role": "user", "content": "用户强制中止"}
            self._append_buffer_messages(buffer, [prompt], R=R, C=C, L=L)
            self._forced_interruption_pending = False
            return None
        if not self._last_iteration_had_tool_calls:
            self.interaction_manager.request_pause()
        return None

    # Initial-prompt customization hooks

    def create_initial_system_prompt(self, restart_records: list[CandidateRecord]) -> str:
        """Create the interactive system prompt with capability-safe tool guidance.

        Args:
            restart_records: Ranked candidates used to set the next objective.

        Returns:
            Interactive system-prompt text.
        """
        mse_goal = self._build_mse_goal(restart_records)
        return (
            "You are a Symbolic Regression Agent working with a human researcher. "
            "Your goal is to discover simple, interpretable mathematical formulas that explain "
            "the relationship between feature variables and the target variable.\n\n"
            "Guidelines:\n"
            "- Explore data thoroughly before proposing formulas.\n"
            "- Prefer simple, interpretable expressions over complex ones.\n"
            f"- {mse_goal}\n"
            f"- {self._TOOL_USAGE_GUIDANCE}"
        )

    @classmethod
    def normalize_system_prompt(cls, prompt: str) -> str:
        """Remove obsolete generated guidance that advertises workspace-only tools.

        Args:
            prompt: Stored or newly submitted system prompt.

        Returns:
            System prompt without legacy workspace guidance. Capability-neutral guidance is
            appended when the removed paragraph came from an older generated prompt.
        """
        marker = cls._LEGACY_WORKSPACE_GUIDANCE
        start = prompt.find(marker)
        if start < 0:
            return prompt
        end = start + len(marker)
        for suffix in (
            " Use workspace_shell for bounded file operations and workspace_code_executor for Python analysis.",
            " Use workspace_shell for bounded file operations.",
            " Use workspace_code_executor for Python analysis.",
        ):
            if prompt.startswith(suffix, end):
                end += len(suffix)
                break
        before = prompt[:start].rstrip()
        after = prompt[end:].lstrip()
        replacement = f"- {cls._TOOL_USAGE_GUIDANCE}"
        return "\n".join(part for part in (before, replacement, after) if part)

    def customize_initial_prompts(self, messages: list[Message], *, X: dict[str, Any], y: dict[str, Any]) -> list[Message]:
        """Apply UI-provided descriptions and prompt overrides.

        Args:
            messages: Initial system and user messages.
            X: Feature arrays keyed by variable name.
            y: Target arrays keyed by variable name.

        Returns:
            Customized initial messages.
        """
        descriptions = getattr(self, "variable_descriptions", {})
        rows = [f"- {name}: {descriptions[name]}" for name in [*X, *y] if descriptions.get(name)]
        if rows:
            messages[1]["content"] += "\n\nVariable descriptions:\n" + "\n".join(rows)
        overrides = getattr(self, "prompt_overrides", {})
        for message in messages:
            if message.get("role") in overrides:
                content = overrides[message["role"]]
                if message["role"] == "system":
                    content = self.normalize_system_prompt(content)
                message["content"] = content
        return messages

    # Streaming model and tool events

    def request_llm(self, prompt: list[Message], R: int, L: int, C: int) -> tuple[list[ModelResponse], Usage]:
        """Request the model while publishing frontend-neutral progress events.

        Args:
            prompt: Prompt messages sent to the model.
            R: One-based restart index.
            L: One-based refinement-step index.
            C: One-based conversation-branch index.

        Returns:
            Parsed responses and usage, including partial responses after interruption.
        """
        coord = {"R": R, "C": C, "L": L}
        self.interaction_manager.publish_event("context", {"messages": prompt, "coord": coord})
        route = self.model_router.route(
            task_score=self._task_route_score,
            task_reasons=self._task_route_reasons,
            refinement_step=L,
        )
        tool_schemas = {
            tool.metadata.name: self.tool_schema(tool.metadata.name)
            for tool in self.tools
            if getattr(tool, "metadata", None) is not None
        }
        response_ids = {
            K: f"{self.run_state.run_id}:{R}:{C}:{L}:{K}"
            for K in range(1, self.local_sample_size + 1)
        }
        for K, response_id in response_ids.items():
            self.interaction_manager.publish_event("assistant_started", {
                "response_id": response_id,
                "coord": coord | {"K": K},
                "tool_schemas": tool_schemas,
                "provider": route.provider,
                "model": route.model,
            })
        last_stream_emit: dict[int, float] = {}
        latest_stream_updates: dict[int, dict[str, Any]] = {}

        def stream_callback(update: dict[str, Any]) -> None:
            K = max(1, min(int(update.get("sample", 1)), self.local_sample_size))
            latest_stream_updates[K] = update
            if self.interaction_manager.is_interrupting:
                raise InterruptedError("用户强制中止")
            now = time.monotonic()
            if (
                update.get("type") == "delta"
                and now - last_stream_emit.get(K, 0.0) < 0.08
            ):
                return
            last_stream_emit[K] = now
            self.interaction_manager.publish_event("assistant_delta", {
                **update,
                "response_id": response_ids[K],
                "coord": coord | {"K": K},
                "tool_schemas": tool_schemas,
                "provider": route.provider,
                "model": route.model,
            })

        interrupted = False
        try:
            with self.interaction_manager.cancellable(self.api.cancel):
                responses, usage = super().request_llm(
                    prompt,
                    R=R,
                    L=L,
                    C=C,
                    stream_callback=stream_callback,
                )
        except InterruptedError:
            interrupted = True
            self._forced_interruption_pending = True
            responses = []
            for K in range(1, self.local_sample_size + 1):
                update = latest_stream_updates.get(K, {})
                message = {
                    "role": "assistant",
                    "content": update.get("content", ""),
                }
                if update.get("reasoning"):
                    message["reasoning"] = update["reasoning"]
                responses.append((message["content"], [], message))
            usage = {"token": {}, "price": {}}
        except Exception as exc:
            for K, response_id in response_ids.items():
                self.interaction_manager.publish_event("assistant_failed", {
                    **latest_stream_updates.get(K, {}),
                    "response_id": response_id,
                    "coord": coord | {"K": K},
                    "provider": route.provider,
                    "model": route.model,
                    "error": str(exc),
                })
            raise
        self._last_iteration_had_tool_calls = any(
            calls for _, calls, _ in responses
        )
        cumulative_usage = {
            "token": self.token_counter.named_count,
            "price": self.money_counter.named_count,
        }
        for K, (content, calls, message) in enumerate(responses, 1):
            self.interaction_manager.publish_event(
                "assistant_failed" if interrupted else "assistant_completed",
                {
                "response_id": response_ids[K],
                "content": content,
                "message": message,
                "tool_calls": calls,
                "tool_schemas": tool_schemas,
                "coord": coord | {"K": K},
                "usage": usage,
                "cumulative_usage": cumulative_usage,
                "provider": route.provider,
                "model": route.model,
                **({"error": "用户强制中止", "interrupted": True} if interrupted else {}),
                },
            )
        return responses, usage

    def tool_schema(self, name: str) -> dict[str, Any]:
        """Return the schema exposed by one initialized tool.

        Args:
            name: Registered name.

        Returns:
            dict[str, Any]: The operation result.
        """
        return next(
            (
                {
                    "description": tool.metadata.description,
                    "parameters": tool.metadata.parameters or {},
                }
                for tool in self.tools
                if tool.metadata.name == name
            ),
            {},
        )

    def execute_action(self, actions: list[ToolCall]) -> list[ToolCallResult]:
        """Execute tools serially with safe control boundaries and UI events.

        Args:
            actions: Tool calls to execute.

        Returns:
            Tool results in call order.
        """
        results = []
        for action in actions:
            tool_schema = self.tool_schema(action.name)
            self.interaction_manager.publish_event("tool_started", {"call": action, "tool_schema": tool_schema})
            started_at = time.monotonic()
            tool = next((item for item in self.tools if item.metadata.name == action.name), None)
            if tool is None:
                result = super().execute_action([action])[0]
            else:
                with self.interaction_manager.cancellable(tool.cancel):
                    result = super().execute_action([action])[0]
            self.interaction_manager.publish_event("tool_completed", {
                "call": action,
                "tool_schema": tool_schema,
                "duration_seconds": time.monotonic() - started_at,
                "result": result,
            })
            results.append(result)
        return results

    def collect_candidates(self, *args: Any, **kwargs: Any) -> list[CandidateRecord]:
        """Update scientific state and publish its current ranked view.

        Args:
            *args: Arguments forwarded to the base candidate collector.
            **kwargs: Keyword arguments forwarded to the base candidate collector.

        Returns:
            Candidates ranked under the configured metric.
        """
        records = super().collect_candidates(*args, **kwargs)
        self.interaction_manager.publish_event("topk_updated", {"records": [record.display_dict() for record in records]})
        return records

    def record_tool_calls(self, tool_calls: list[ToolCall], results: list[ToolCallResult], R: int, L: int, C: int, forced: bool = False) -> None:
        """Persist tool calls and expose framework-enforced calls to the UI.

        Args:
            tool_calls: Tool calls returned by the model.
            results: Result records to process.
            R: One-based restart index.
            L: One-based refinement-step index.
            C: One-based conversation-branch index.
            forced: Whether the framework, rather than the model, initiated the call.
        """
        super().record_tool_calls(tool_calls, results, R=R, L=L, C=C, forced=forced)
        if forced:
            for call, result in zip(tool_calls, results):
                self.interaction_manager.publish_event("tool_completed", {
                    "call": call,
                    "tool_schema": self.tool_schema(call.name),
                    "result": result,
                    "forced": True,
                })

    def execute_action_parallel(self, actions: list[ToolCall], max_workers: int) -> list[ToolCallResult]:
        """Reject parallel execution in interactive mode.

        Args:
            actions: Tool calls to execute.
            max_workers: Maximum number of parallel workers.

        Raises:
            NotImplementedError: Always, because interactive workspace tools are not
                guaranteed to be read-only.
        """
        raise NotImplementedError(
            "Parallel execution is not supported in interactive mode, "
            "since workspace tools cannot guarantee read-only access. "
            "Please set max_workers=0 to disable parallel execution when using interactive tools."
        )
