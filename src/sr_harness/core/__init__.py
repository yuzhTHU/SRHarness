"""Dependency-light domain types and runtime state for SRHarness."""
from .api import APICallResult
from .context import AgentContext
from .context_data import (
    ContextManifestError,
    inspect_context_data,
    load_context_data,
    update_context_data_descriptions,
)
from .search import (
    CandidateRecord,
    ParentLink,
    ParentRelation,
    SearchCoordinate,
    SearchNode,
    SearchResult,
    SearchRunState,
    json_value,
)
from .tool import ToolCall, ToolCallResult, ToolMetadata
