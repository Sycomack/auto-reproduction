import json
import tempfile
import unittest
from pathlib import Path

from reproducer.agent import DirectReproductionStrategy
from reproducer.task import TaskSpec


class DirectReproductionStrategyTests(unittest.TestCase):
    def test_operational_task_metadata_is_in_initial_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "paper.pdf").write_bytes(b"%PDF-test")
            (root / "repository").mkdir()
            task_file = root / "task.json"
            task_file.write_text(
                json.dumps(
                    {
                        "task_id": "demo",
                        "title": "Demo",
                        "paper": {"path": "paper.pdf"},
                        "repository": {"path": "repository"},
                        "claims": [{"claim_id": "C1", "statement": "Claim"}],
                        "reproduction_protocol": {"rounds": 2},
                        "known_compatibility_issues": ["retired model"],
                    }
                ),
                encoding="utf-8",
            )
            task = TaskSpec.load(task_file)

            messages = DirectReproductionStrategy().initial_messages(task)
            user_prompt = messages[1]["content"]

            self.assertIn("reproduction_protocol", user_prompt)
            self.assertIn("known_compatibility_issues", user_prompt)
            self.assertIn("retired model", user_prompt)


if __name__ == "__main__":
    unittest.main()
