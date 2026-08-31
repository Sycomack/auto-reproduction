import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproducer.agent import DirectReproductionStrategy, ReproductionAgent
from reproducer.llm import ChatResponse, TokenUsage
from reproducer.memory import ConversationMemory, MemoryConfig
from reproducer.runtime import prepare_run
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


class CompactingFakeClient:
    def __init__(self) -> None:
        self.main_calls = 0
        self.curator_calls = 0

    def complete(self, messages, tools):
        if not tools:
            self.curator_calls += 1
            return ChatResponse(
                message={
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "current_goal": "Complete the demo",
                            "next_action": "Submit the report",
                            "upsert": [
                                {
                                    "key": "paper-read",
                                    "category": "evidence",
                                    "content": "paper.txt was inspected",
                                }
                            ],
                            "resolve": [],
                            "supersede": [],
                        }
                    ),
                },
                usage=TokenUsage(3, 2, 5),
                model="fake-curator",
                response_id="curator-response",
            )

        self.main_calls += 1
        if self.main_calls <= 3:
            name = "read_file"
            arguments = {"path": "paper.txt"}
        else:
            name = "finish"
            arguments = {
                "status": "completed",
                "report_markdown": "# Report\n\nVerdict: supported",
            }
        return ChatResponse(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"{name}-{self.main_calls}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments),
                        },
                    }
                ],
            },
            usage=TokenUsage(10, 4, 14),
            model="fake-model",
            response_id=f"main-response-{self.main_calls}",
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
    def test_prepare_preserves_dangling_repository_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            (source / "paper.pdf").write_bytes(b"%PDF-test")
            repository = source / "repository"
            repository.mkdir()
            dangling_link = repository / "missing-test-data"
            try:
                dangling_link.symlink_to("not-present")
            except OSError as exc:
                self.skipTest(f"Symbolic links are unavailable: {exc}")
            task_file = source / "task.json"
            task_file.write_text(
                json.dumps(
                    {
                        "task_id": "symlink-demo",
                        "title": "Symlink demo",
                        "paper": {"path": "paper.pdf"},
                        "repository": {"path": "repository"},
                        "claims": [{"claim_id": "C1", "statement": "Demo claim"}],
                    }
                ),
                encoding="utf-8",
            )

            def fake_extract(_paper, text_path):
                text_path.write_text("paper text", encoding="utf-8")

            with patch("reproducer.runtime.workspace._extract_pdf", side_effect=fake_extract):
                prepared = prepare_run(TaskSpec.load(task_file), root / "run")

            copied_link = prepared.workspace_dir / "repository" / dangling_link.name
            self.assertTrue(copied_link.is_symlink())
            self.assertEqual(copied_link.readlink(), Path("not-present"))

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

    def test_tool_loop_traces_and_counts_memory_compaction(self) -> None:
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
                        "task_id": "memory-demo",
                        "title": "Memory demo task",
                        "paper": {"path": "paper.pdf"},
                        "repository": {"path": "repository"},
                        "claims": [{"claim_id": "C1", "statement": "Demo claim"}],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "run"
            memory = ConversationMemory(
                config=MemoryConfig(
                    max_context_tokens=1,
                    recent_steps=1,
                    min_compaction_steps=2,
                    max_state_items=10,
                ),
                state_path=output / "workspace" / "memory_state.json",
            )
            client = CompactingFakeClient()

            def fake_extract(_paper, text_path):
                text_path.write_text("paper text", encoding="utf-8")

            with patch(
                "reproducer.runtime.workspace._extract_pdf",
                side_effect=fake_extract,
            ):
                result = ReproductionAgent(
                    TaskSpec.load(task_file),
                    client,
                    output_dir=output,
                    max_steps=4,
                    memory=memory,
                ).run()

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.model_calls, 5)
            self.assertEqual(result.token_usage, TokenUsage(43, 18, 61))
            self.assertEqual(client.main_calls, 4)
            self.assertEqual(client.curator_calls, 1)
            self.assertTrue((output / "workspace" / "memory_state.json").is_file())
            trace = result.trace_path.read_text(encoding="utf-8")
            self.assertIn('"event": "memory_compacted"', trace)
            self.assertIn('"compacted_steps": [1, 2]', trace)
            self.assertIn('"model": "fake-curator"', trace)


if __name__ == "__main__":
    unittest.main()
