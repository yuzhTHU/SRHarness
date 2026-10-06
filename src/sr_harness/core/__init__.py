"""Dependency-light domain types and runtime state for SRHarness."""
from .api import APICallResult
from .context import AgentContext
from .context_data import (
    ContextAxis,
    ContextData,
    ContextDataStore,
    ContextManifestError,
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
