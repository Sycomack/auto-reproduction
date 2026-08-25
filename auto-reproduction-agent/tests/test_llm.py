import json
import unittest
from unittest.mock import patch

from reproducer.config import ModelConfig
from reproducer.llm import OpenAICompatibleClient, TokenUsage


class _FakeHTTPResponse:
    def __init__(self, body: dict) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.body).encode("utf-8")


class LLMContractTests(unittest.TestCase):
    def test_chat_client_returns_normalized_response_and_usage(self) -> None:
        body = {
            "id": "response-1",
            "model": "served-model",
            "choices": [{"message": {"role": "assistant", "content": "done"}}],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
        }
        config = ModelConfig("https://example.test/v1", "", "requested-model")
        client = OpenAICompatibleClient(config)

        with patch(
            "urllib.request.urlopen", return_value=_FakeHTTPResponse(body)
        ):
            response = client.complete([], [])

        self.assertEqual(response.message["content"], "done")
        self.assertEqual(response.model, "served-model")
        self.assertEqual(response.response_id, "response-1")
        self.assertEqual(response.usage, TokenUsage(11, 7, 18))

    def test_token_usage_accepts_alternative_names_and_bad_values(self) -> None:
        usage = TokenUsage.from_api(
            {"input_tokens": "5", "output_tokens": None, "total_tokens": "bad"}
        )
        self.assertEqual(usage, TokenUsage(5, 0, 5))


if __name__ == "__main__":
    unittest.main()
