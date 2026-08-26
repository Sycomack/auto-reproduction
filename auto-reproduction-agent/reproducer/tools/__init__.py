from .definitions import PAPER_VISUAL_TOOL_DEFINITION, TOOL_DEFINITIONS
from .registry import ToolRegistry
from .workspace import WorkspaceTools, parse_tool_arguments, workspace_tool_handlers


def build_workspace_registry(tools: WorkspaceTools) -> ToolRegistry:
    definitions = list(TOOL_DEFINITIONS)
    if tools.vision_client is not None:
        definitions.append(PAPER_VISUAL_TOOL_DEFINITION)
    return ToolRegistry.from_definitions(
        definitions, workspace_tool_handlers(tools)
    )


__all__ = [
    "TOOL_DEFINITIONS",
    "PAPER_VISUAL_TOOL_DEFINITION",
    "ToolRegistry",
    "WorkspaceTools",
    "build_workspace_registry",
    "parse_tool_arguments",
]
