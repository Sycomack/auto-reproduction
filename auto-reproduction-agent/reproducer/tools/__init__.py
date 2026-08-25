from .definitions import TOOL_DEFINITIONS
from .registry import ToolRegistry
from .workspace import WorkspaceTools, parse_tool_arguments, workspace_tool_handlers


def build_workspace_registry(tools: WorkspaceTools) -> ToolRegistry:
    return ToolRegistry.from_definitions(
        TOOL_DEFINITIONS, workspace_tool_handlers(tools)
    )


__all__ = [
    "TOOL_DEFINITIONS",
    "ToolRegistry",
    "WorkspaceTools",
    "build_workspace_registry",
    "parse_tool_arguments",
]
