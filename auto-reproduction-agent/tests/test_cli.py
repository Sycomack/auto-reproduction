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
    def test_resume_reuses_prepared_workspace_and_global_step(self) -> None:
        prepared = object()
        resume_state = SimpleNamespace(
            task=SimpleNamespace(visual_inputs=()),
            prepared=prepared,
            next_step=56,
            additional_context="# Resume context\nContinue generation.",
        )
        run_result = SimpleNamespace(
            status="completed",
            report_path=Path("report.md"),
            trace_path=Path("trace.jsonl"),
            model_calls=1,
            token_usage=SimpleNamespace(as_dict=lambda: {}),
        )
        stdout = StringIO()
        with (
            patch("reproducer.cli.load_resume_state", return_value=resume_state),
            patch("reproducer.cli.OpenAICompatibleClient"),
            patch("reproducer.cli.ReproductionAgent") as agent_class,
            redirect_stdout(stdout),
        ):
            agent_class.return_value.run.return_value = run_result
            result = main(
                [
                    "--resume",
                    "runs/h2o-main",
                    "--additional-steps",
                    "80",
                ]
            )

        self.assertEqual(result, 0)
        kwargs = agent_class.call_args.kwargs
        self.assertIs(kwargs["prepared_workspace"], prepared)
        self.assertEqual(kwargs["start_step"], 56)
        self.assertEqual(kwargs["max_steps"], 80)
        self.assertIn("Continue generation", kwargs["resume_context"])

    def test_resume_requires_positive_additional_steps(self) -> None:
        stderr = StringIO()
        with redirect_stderr(stderr):
            result = main(["--resume", "runs/demo"])

        self.assertEqual(result, 2)
        self.assertIn("requires --additional-steps", stderr.getvalue())

    def test_additional_steps_requires_resume(self) -> None:
        stderr = StringIO()
        with redirect_stderr(stderr):
            result = main(
                ["--task", "task.json", "--additional-steps", "10"]
            )

        self.assertEqual(result, 2)
        self.assertIn("requires --resume", stderr.getvalue())

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
