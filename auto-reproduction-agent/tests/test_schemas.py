import json
import tempfile
import unittest
from pathlib import Path

from reproducer.task import TaskSpec, TaskValidationError


class TaskSpecTests(unittest.TestCase):
    def test_load_resolves_paths_relative_to_task_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "paper.pdf").write_bytes(b"%PDF-test")
            (root / "repository").mkdir()
            task_data = {
                "task_id": "demo",
                "title": "Demo",
                "paper": {"path": "paper.pdf"},
                "repository": {"path": "repository", "url": "https://example.test/repo"},
                "claims": [{"claim_id": "C1", "statement": "A test claim"}],
                "budget": {"max_agent_steps": 7, "max_command_seconds": 13},
            }
            task_file = root / "task.json"
            task_file.write_text(json.dumps(task_data), encoding="utf-8")

            task = TaskSpec.load(task_file)

            self.assertEqual(task.task_id, "demo")
            self.assertEqual(task.paper_path, (root / "paper.pdf").resolve())
            self.assertEqual(task.repository_path, (root / "repository").resolve())
            self.assertEqual(task.max_agent_steps, 7)

    def test_visual_input_requires_only_a_figure_label_not_crop_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "paper.pdf").write_bytes(b"%PDF-test")
            (root / "repository").mkdir()
            task_file = root / "task.json"
            task_file.write_text(
                json.dumps(
                    {
                        "task_id": "visual-demo",
                        "title": "Visual demo",
                        "paper": {"path": "paper.pdf"},
                        "repository": {"path": "repository"},
                        "claims": [{"claim_id": "C1", "statement": "Claim"}],
                        "visual_inputs": [
                            {
                                "id": "figure_4",
                                "figure_label": "Figure 4",
                                "purpose": "primary experiment",
                                "focus": "top-left panel",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            task = TaskSpec.load(task_file)

            self.assertEqual(len(task.visual_inputs), 1)
            self.assertEqual(task.visual_inputs[0].figure_label, "Figure 4")
            self.assertEqual(task.visual_inputs[0].focus, "top-left panel")
            self.assertIsNone(task.visual_inputs[0].page)

    def test_schema_v2_uses_project_resources_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_dir = root / "tasks" / "demo"
            task_dir.mkdir(parents=True)
            resource_dir = root / "resources" / "demo"
            resource_dir.mkdir(parents=True)
            (resource_dir / "paper.pdf").write_bytes(b"%PDF-test")
            (resource_dir / "repository").mkdir()
            task_file = task_dir / "task.json"
            task_file.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "task_id": "demo",
                        "title": "Demo",
                        "paper": {"path": "paper.pdf"},
                        "repository": {"path": "repository"},
                        "claims": [{"claim_id": "C1", "statement": "Claim"}],
                    }
                ),
                encoding="utf-8",
            )

            task = TaskSpec.load(task_file)

            self.assertEqual(task.resource_dir, resource_dir.resolve())
            self.assertEqual(task.paper_path, (resource_dir / "paper.pdf").resolve())
            self.assertEqual(
                task.repository_path, (resource_dir / "repository").resolve()
            )

    def test_missing_claims_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_file = root / "task.json"
            task_file.write_text(
                json.dumps(
                    {
                        "task_id": "demo",
                        "title": "Demo",
                        "paper": {},
                        "repository": {},
                        "claims": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(TaskValidationError):
                TaskSpec.load(task_file)


if __name__ == "__main__":
    unittest.main()
