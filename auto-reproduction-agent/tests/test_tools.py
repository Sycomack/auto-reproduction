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

    def test_registry_rejects_duplicate_names(self) -> None:
        registry = ToolRegistry()
        definition = {"type": "function", "function": {"name": "demo"}}
        registry.register(definition, lambda: "ok")
        with self.assertRaises(ValueError):
            registry.register(definition, lambda: "duplicate")


if __name__ == "__main__":
    unittest.main()
