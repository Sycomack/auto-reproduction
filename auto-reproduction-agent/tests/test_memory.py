import unittest

from reproducer.memory import ConversationMemory


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
