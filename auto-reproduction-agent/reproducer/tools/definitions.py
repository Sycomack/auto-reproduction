from __future__ import annotations

from typing import Any


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files below a workspace-relative path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "max_depth": {"type": "integer", "minimum": 0, "maximum": 8, "default": 3},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1, "default": 1},
                    "max_lines": {"type": "integer", "minimum": 1, "maximum": 2000, "default": 400},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search text files using a regular expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "file_pattern": {"type": "string", "default": "*"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a UTF-8 file inside the run workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a command without a shell in a workspace-relative directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "cwd": {"type": "string", "default": "repository"},
                    "timeout_seconds": {"type": "integer", "minimum": 1},
                },
                "required": ["argv"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_numeric_points",
            "description": (
                "Compare expected and observed numeric points deterministically, "
                "calculate errors, and save a JSON comparison artifact."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "points": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "expected": {"type": "number"},
                                "observed": {"type": "number"},
                            },
                            "required": ["id", "expected", "observed"],
                        },
                    },
                    "absolute_tolerance": {"type": "number", "minimum": 0},
                    "relative_tolerance": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Optional fractional tolerance, such as 0.05.",
                    },
                    "output_path": {
                        "type": "string",
                        "default": "artifacts/numeric_comparison.json",
                    },
                },
                "required": ["points", "absolute_tolerance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish the run and save the final Markdown reproduction report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["completed", "inconclusive", "failed"]},
                    "report_markdown": {"type": "string"},
                },
                "required": ["status", "report_markdown"],
            },
        },
    },
]


PAPER_VISUAL_TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "inspect_paper_visual",
        "description": (
            "Render and visually inspect a paper figure or page when prepared "
            "paper_evidence.json is missing or insufficient."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Specific visual evidence to extract.",
                },
                "figure_label": {
                    "type": "string",
                    "description": "Caption label such as Figure 4 or Table 2.",
                    "default": "",
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional one-based PDF page number.",
                },
            },
            "required": ["prompt"],
        },
    },
}
