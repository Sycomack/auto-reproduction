import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproducer.config import ModelConfig, VisionConfig
from reproducer.llm import (
    OpenAICompatibleClient,
    OpenAICompatibleVisionClient,
    TokenUsage,
)


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

    def test_vision_client_sends_image_content_block(self) -> None:
        body = {
            "model": "vision-model",
            "choices": [{"message": {"role": "assistant", "content": "{}"}}],
        }
        config = VisionConfig("https://example.test/v1", "key", "vision-model")
        client = OpenAICompatibleVisionClient(config)
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "figure.png"
            image_path.write_bytes(b"fake-png")
            with patch(
                "urllib.request.urlopen", return_value=_FakeHTTPResponse(body)
            ) as mocked_urlopen:
                response = client.analyze(image_path, "Inspect the figure")

        request = mocked_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        user_content = payload["messages"][1]["content"]
        self.assertEqual(payload["model"], "vision-model")
        self.assertEqual(user_content[0]["type"], "text")
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertTrue(user_content[1]["image_url"]["url"].startswith("data:image/png"))
        self.assertEqual(response["content"], "{}")


if __name__ == "__main__":
    unittest.main()
