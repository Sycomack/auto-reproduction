import json
import tempfile
import unittest
from pathlib import Path

from reproducer.agent import ReproductionAgent
from reproducer.llm import ChatResponse, TokenUsage
from reproducer.runtime import TraceWriter, load_resume_state


class FinishClient:
    def __init__(self) -> None:
        self.messages = []

    def complete(self, messages, tools):
        self.messages = messages
        return ChatResponse(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "finish-resume",
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "arguments": json.dumps(
                                {
                                    "status": "completed",
                                    "report_markdown": "# Resumed report\n\nDone.",
                                }
                            ),
                        },
                    }
                ],
            },
            usage=TokenUsage(20, 5, 25),
            model="fake-model",
            response_id="resume-response",
        )


class ResumeTests(unittest.TestCase):
    def _make_terminal_run(self, root: Path) -> Path:
        run_dir = root / "run"
        workspace = run_dir / "workspace"
        (workspace / "repository").mkdir(parents=True)
        (workspace / "inputs").mkdir()
        (workspace / "artifacts").mkdir()
        (workspace / "inputs" / "paper.pdf").write_bytes(b"%PDF-test")
        (workspace / "paper.txt").write_text("paper", encoding="utf-8")
        (workspace / "paper_evidence.json").write_text(
            json.dumps({"visual_inputs": []}), encoding="utf-8"
        )
        (run_dir / "task_snapshot.json").write_text(
            json.dumps(
                {
                    "task_id": "demo",
                    "title": "Demo",
                    "paper": {"path": "paper.pdf"},
                    "repository": {"path": "repository"},
                    "claims": [{"claim_id": "C1", "statement": "Claim"}],
                    "budget": {"max_agent_steps": 1, "max_command_seconds": 30},
                }
            ),
            encoding="utf-8",
        )
        trace = TraceWriter(run_dir / "trace.jsonl")
        trace.write("run_started", task_id="demo", max_steps=1)
        trace.write(
            "assistant_message",
            step=1,
            message={
                "role": "assistant",
                "content": "The package mirror works; install dependencies next.",
                "reasoning_content": "Continue with the pending installation.",
            },
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        trace.write(
            "tool_result",
            step=1,
            tool="run_command",
            arguments={"argv": ["env"]},
            result={
                "ok": True,
                "output": (
                    "exit_code: 0\nstdout:\n"
                    "REPRO_API_KEY=sk-test-secret-value\n"
                    "HF_ACCESS_TOKEN=hf-private-value"
                ),
            },
        )
        trace.write("run_finished", step=1, status="inconclusive")
        (run_dir / "reproduction_report.md").write_text(
            "inconclusive", encoding="utf-8"
        )
        return run_dir

    def test_resume_reuses_workspace_and_appends_global_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = self._make_terminal_run(Path(temp_dir))
            state = load_resume_state(run_dir)

            self.assertEqual(state.previous_step, 1)
            self.assertEqual(state.next_step, 2)
            self.assertIn("package mirror works", state.additional_context)
            self.assertIn("Execute the pending setup", state.additional_context)
            self.assertNotIn("sk-test-secret-value", state.additional_context)
            self.assertNotIn("hf-private-value", state.additional_context)
            self.assertIn("<redacted", state.additional_context)
            self.assertLessEqual(len(state.additional_context), 20_100)
            self.assertTrue(state.context_path.is_file())

            client = FinishClient()
            agent = ReproductionAgent(
                task=state.task,
                client=client,
                max_steps=3,
                prepared_workspace=state.prepared,
                start_step=state.next_step,
                resume_context=state.additional_context,
            )
            result = agent.run()

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.steps, 2)
            self.assertIn("Resume this run at agent step 2", client.messages[-1]["content"])
            trace = result.trace_path.read_text(encoding="utf-8")
            self.assertIn('"event": "run_resumed"', trace)
            self.assertIn('"step": 2, "status": "completed"', trace)
            self.assertIn(
                "# Resumed report",
                result.report_path.read_text(encoding="utf-8"),
            )

    def test_completed_run_cannot_be_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = self._make_terminal_run(Path(temp_dir))
            TraceWriter(run_dir / "trace.jsonl").write(
                "run_finished", step=2, status="completed"
            )

            with self.assertRaisesRegex(RuntimeError, "completed run"):
                load_resume_state(run_dir)


if __name__ == "__main__":
    unittest.main()
