import json
import tempfile
import unittest
from pathlib import Path

from reproducer.llm import ChatResponse, TokenUsage
from reproducer.memory import (
    ConversationMemory,
    MemoryConfig,
    StructuredMemoryState,
)


class _CuratorClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append((messages, tools))
        return ChatResponse(
            message={"role": "assistant", "content": self.content},
            usage=TokenUsage(20, 5, 25),
            model="curator-model",
            response_id="curator-1",
        )


def _add_tool_step(memory: ConversationMemory, step: int) -> None:
    call_id = f"call-{step}"
    memory.begin_step(step)
    memory.add(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": f"artifact-{step}.txt"}),
                    },
                }
            ],
        }
    )
    memory.add(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps({"ok": True, "output": "x" * 200}),
        }
    )
    memory.finish_step(step)


class ConversationMemoryTests(unittest.TestCase):
    def test_memory_returns_a_copy_of_messages(self) -> None:
        memory = ConversationMemory()
        original = {"role": "user", "content": "hello"}
        memory.add(original)
        original["content"] = "changed"

        snapshot = memory.as_messages()
        snapshot[0]["content"] = "also changed"

        self.assertEqual(len(memory), 1)
        self.assertEqual(memory.as_messages()[0]["content"], "hello")

    def test_structured_state_applies_incremental_updates(self) -> None:
        state = StructuredMemoryState()
        state.apply_delta(
            {
                "current_goal": "Run baseline",
                "next_action": "Score output",
                "upsert": [
                    {
                        "key": "baseline-output",
                        "category": "artifact",
                        "content": "Generated 100 examples",
                        "evidence_paths": ["artifacts/baseline.jsonl"],
                    }
                ],
            },
            source_steps=[1],
            max_items=10,
        )
        state.apply_delta(
            {
                "upsert": [
                    {
                        "key": "baseline-output",
                        "category": "numeric_result",
                        "content": "Generated and scored 1000 examples: ROUGE-2=0.1254",
                        "source_steps": [2],
                        "evidence_paths": ["artifacts/baseline-score.json"],
                    }
                ],
                "resolve": ["baseline-output"],
            },
            source_steps=[2],
            max_items=10,
        )

        item = state.items["baseline-output"]
        self.assertEqual(item.status, "resolved")
        self.assertEqual(item.source_steps, [1, 2])
        self.assertEqual(len(item.evidence_paths), 2)
        self.assertIn("0.1254", item.content)

    def test_compaction_keeps_prefix_and_recent_complete_tool_step(self) -> None:
        delta = json.dumps(
            {
                "current_goal": "Inspect recent result",
                "next_action": "Continue from step 3",
                "upsert": [
                    {
                        "key": "older-artifacts",
                        "category": "artifact",
                        "content": "Steps 1 and 2 inspected two artifacts",
                        "source_steps": [1, 2],
                    }
                ],
                "resolve": [],
                "supersede": [],
            }
        )
        client = _CuratorClient(delta)
        config = MemoryConfig(
            max_context_tokens=1,
            recent_steps=1,
            min_compaction_steps=2,
            max_state_items=10,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "memory_state.json"
            memory = ConversationMemory(config=config, state_path=state_path)
            memory.extend(
                [
                    {"role": "system", "content": "system prompt"},
                    {"role": "user", "content": "task prompt"},
                ]
            )
            for step in range(1, 4):
                _add_tool_step(memory, step)

            result = memory.maybe_compact(client)
            messages = memory.as_messages()

            self.assertTrue(result.compacted)
            self.assertEqual(result.compacted_steps, (1, 2))
            self.assertEqual(result.usage, TokenUsage(20, 5, 25))
            self.assertTrue(state_path.is_file())
            self.assertEqual(messages[0]["content"], "system prompt")
            self.assertEqual(messages[1]["content"], "task prompt")
            self.assertIn("STRUCTURED WORKING MEMORY", messages[2]["content"])
            self.assertEqual(messages[3]["tool_calls"][0]["id"], "call-3")
            self.assertEqual(messages[4]["tool_call_id"], "call-3")
            self.assertEqual(client.calls[0][1], [])

    def test_invalid_curator_response_retains_raw_history(self) -> None:
        client = _CuratorClient("not-json")
        memory = ConversationMemory(
            config=MemoryConfig(
                max_context_tokens=1,
                recent_steps=1,
                min_compaction_steps=2,
                max_state_items=10,
            )
        )
        memory.add({"role": "system", "content": "prefix"})
        for step in range(1, 4):
            _add_tool_step(memory, step)
        before = memory.as_messages()

        result = memory.maybe_compact(client)

        self.assertTrue(result.attempted)
        self.assertFalse(result.compacted)
        self.assertIn("valid JSON", result.error)
        self.assertEqual(memory.as_messages(), before)
