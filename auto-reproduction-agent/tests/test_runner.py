import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproducer.agent import DirectReproductionStrategy, ReproductionAgent
from reproducer.llm import ChatResponse, TokenUsage
from reproducer.task import TaskSpec


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return ChatResponse(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "read-1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"path": "paper.txt"}),
                            },
                        },
                    ],
                },
                usage=TokenUsage(10, 4, 14),
                model="fake-model",
                response_id=f"response-{self.calls}",
            )
        return ChatResponse(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "finish-1",
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "arguments": json.dumps(
                                {
                                    "status": "completed",
                                    "report_markdown": "# Report\n\nVerdict: supported",
                                }
                            ),
                        },
                    }
                ],
            },
            usage=TokenUsage(12, 5, 17),
            model="fake-model",
            response_id=f"response-{self.calls}",
        )


class RecordingStrategy:
    def __init__(self) -> None:
        self.delegate = DirectReproductionStrategy()
        self.before_steps = []
        self.observations = []

    def initial_messages(self, task):
        return self.delegate.initial_messages(task)

    def before_step(self, task, step):
        self.before_steps.append(step)
        return []

    def after_tools(self, task, step, observations):
        self.observations.append((step, observations))
        return []

    def no_tool_message(self, step):
        return self.delegate.no_tool_message(step)


class RunnerTests(unittest.TestCase):
    def test_tool_loop_writes_trace_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            (source / "paper.pdf").write_bytes(b"%PDF-test")
            repository = source / "repository"
            repository.mkdir()
            (repository / "README.md").write_text("demo", encoding="utf-8")
            task_file = source / "task.json"
            task_file.write_text(
                json.dumps(
                    {
                        "task_id": "demo",
                        "title": "Demo task",
                        "paper": {"path": "paper.pdf"},
                        "repository": {"path": "repository"},
                        "claims": [{"claim_id": "C1", "statement": "Demo claim"}],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "run"
            strategy = RecordingStrategy()

            def fake_extract(_paper, text_path):
                text_path.write_text("paper text", encoding="utf-8")

            with patch("reproducer.runtime.workspace._extract_pdf", side_effect=fake_extract):
                agent = ReproductionAgent(
                    TaskSpec.load(task_file),
                    FakeClient(),
                    output_dir=output,
                    strategy=strategy,
                )
                result = agent.run()

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.steps, 2)
            self.assertEqual(result.model_calls, 2)
            self.assertEqual(result.token_usage, TokenUsage(22, 9, 31))
            self.assertEqual(strategy.before_steps, [1, 2])
            self.assertEqual(strategy.observations[0][0], 1)
            self.assertEqual(strategy.observations[0][1][0]["tool"], "read_file")
            self.assertIn("Verdict: supported", result.report_path.read_text(encoding="utf-8"))
            trace = result.trace_path.read_text(encoding="utf-8")
            self.assertIn('"event": "tool_result"', trace)
            self.assertIn('"model": "fake-model"', trace)
            self.assertTrue((output / "workspace" / "repository" / "README.md").is_file())


if __name__ == "__main__":
    unittest.main()
