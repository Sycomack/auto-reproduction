import json
import sys
import tempfile
import unittest
from pathlib import Path

from reproducer.tools import ToolRegistry, WorkspaceTools, build_workspace_registry


class WorkspaceToolsTests(unittest.TestCase):
    def test_file_tools_and_path_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tools = WorkspaceTools(root, max_command_seconds=10)

            tools.write_file("notes/result.txt", "metric = 0.95\n")

            self.assertIn("metric = 0.95", tools.read_file("notes/result.txt"))
            self.assertIn("result.txt", tools.list_files("notes"))
            self.assertIn("result.txt:1", tools.search_files("metric", "notes"))
            with self.assertRaises(ValueError):
                tools.read_file("../outside.txt")

    def test_command_uses_argv_and_captures_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tools = WorkspaceTools(Path(temp_dir), max_command_seconds=10)
            output = tools.run_command(
                [sys.executable, "-c", "print('command-ok')"], cwd="."
            )
            self.assertIn("exit_code: 0", output)
            self.assertIn("command-ok", output)

    def test_numeric_comparison_writes_pointwise_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tools = WorkspaceTools(root, max_command_seconds=10)

            rendered = tools.compare_numeric_points(
                [
                    {"id": "h2o-20", "expected": 11.8, "observed": 11.5},
                    {"id": "local-20", "expected": 6.6, "observed": 8.0},
                ],
                absolute_tolerance=1.0,
            )

            result = json.loads(rendered)
            self.assertTrue(result["points"][0]["within_tolerance"])
            self.assertFalse(result["points"][1]["within_tolerance"])
            self.assertFalse(result["all_within_tolerance"])
            self.assertTrue((root / "artifacts" / "numeric_comparison.json").is_file())

    def test_registry_binds_definitions_and_handlers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = build_workspace_registry(
                WorkspaceTools(Path(temp_dir), max_command_seconds=10)
            )

            self.assertIn("read_file", registry.names)
            self.assertIn("finish", registry.names)
            self.assertEqual(len(registry.definitions), len(registry.names))
            unknown = registry.execute("missing", {})
            self.assertFalse(unknown["ok"])

    def test_visual_tool_is_registered_only_when_a_vision_client_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            without_vision = build_workspace_registry(
                WorkspaceTools(root, max_command_seconds=10)
            )
            with_vision = build_workspace_registry(
                WorkspaceTools(root, max_command_seconds=10, vision_client=object())
            )

            self.assertNotIn("inspect_paper_visual", without_vision.names)
            self.assertIn("inspect_paper_visual", with_vision.names)

    def test_registry_rejects_duplicate_names(self) -> None:
        registry = ToolRegistry()
        definition = {"type": "function", "function": {"name": "demo"}}
        registry.register(definition, lambda: "ok")
        with self.assertRaises(ValueError):
            registry.register(definition, lambda: "duplicate")


if __name__ == "__main__":
    unittest.main()
