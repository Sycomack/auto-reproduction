import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reproducer.cli import main


class ReproducerCLITests(unittest.TestCase):
    def _run_visual_preparation(self, status: str) -> tuple[int, str, str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_path = root / "paper_evidence.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "visual_inputs": [
                            {"id": "figure_4_panel_1", "status": status}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            prepared = SimpleNamespace(
                run_dir=root / "run",
                workspace_dir=root / "run" / "workspace",
                paper_evidence_path=evidence_path,
            )
            stdout = StringIO()
            stderr = StringIO()
            with (
                patch("reproducer.cli.TaskSpec.load", return_value=object()),
                patch("reproducer.cli.OpenAICompatibleVisionClient"),
                patch("reproducer.cli.prepare_run", return_value=prepared),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                result = main(
                    [
                        "--task",
                        "task.json",
                        "--prepare-only",
                        "--prepare-visuals",
                    ]
                )
            return result, stdout.getvalue(), stderr.getvalue()

    def test_visual_preparation_succeeds_for_validated_evidence(self) -> None:
        result, stdout, stderr = self._run_visual_preparation("analyzed")

        self.assertEqual(result, 0)
        self.assertIn("Visual evidence [figure_4_panel_1]: analyzed", stdout)
        self.assertEqual(stderr, "")

    def test_visual_preparation_fails_for_invalid_evidence(self) -> None:
        result, stdout, stderr = self._run_visual_preparation("analysis_invalid")

        self.assertEqual(result, 1)
        self.assertIn("analysis_invalid", stdout)
        self.assertIn("validation did not pass", stderr)


if __name__ == "__main__":
    unittest.main()
