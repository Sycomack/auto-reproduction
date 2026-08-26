import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from reproducer import check_api_cli
from reproducer.llm import ChatResponse, ModelClientError, TokenUsage


class _FakeClient:
    def __init__(self, response):
        self.response = response

    def complete(self, messages, tools):
        return self.response


class CheckAPICLITests(unittest.TestCase):
    def test_success_requires_a_tool_call(self) -> None:
        response = ChatResponse(
            message={
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "probe-1",
                        "type": "function",
                        "function": {
                            "name": "health_check",
                            "arguments": '{"value":"ok"}',
                        },
                    }
                ],
            },
            usage=TokenUsage(8, 3, 11),
            model="fake-model",
            response_id="fake-response",
        )
        stdout = io.StringIO()
        with patch.object(
            check_api_cli, "OpenAICompatibleClient", return_value=_FakeClient(response)
        ):
            with redirect_stdout(stdout):
                exit_code = check_api_cli.main()

        self.assertEqual(exit_code, 0)
        self.assertIn("Tool calling: supported", stdout.getvalue())

    def test_api_error_has_nonzero_exit(self) -> None:
        stderr = io.StringIO()
        with patch.object(
            check_api_cli,
            "OpenAICompatibleClient",
            side_effect=ModelClientError("denied"),
        ):
            with redirect_stderr(stderr):
                exit_code = check_api_cli.main()

        self.assertEqual(exit_code, 2)
        self.assertIn("denied", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
