from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ..llm import ChatClient, TokenUsage, VisionClient
from ..memory import ConversationMemory
from ..runtime import TraceWriter, prepare_run
from ..task import TaskSpec
from ..tools import (
    ToolRegistry,
    WorkspaceTools,
    build_workspace_registry,
    parse_tool_arguments,
)
from .state import RunResult
from .strategy import AgentStrategy, DirectReproductionStrategy


class ReproductionAgent:
    def __init__(
        self,
        task: TaskSpec,
        client: ChatClient,
        output_dir: str | Path | None = None,
        max_steps: int | None = None,
        vision_client: VisionClient | None = None,
        memory: ConversationMemory | None = None,
        strategy: AgentStrategy | None = None,
        tool_registry_factory: (
            Callable[[WorkspaceTools, TaskSpec], ToolRegistry] | None
        ) = None,
    ) -> None:
        self.prepared = prepare_run(task, output_dir, vision_client=vision_client)
        self.task = task
        self.client = client
        self.max_steps = task.max_agent_steps if max_steps is None else max_steps
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.memory = memory or ConversationMemory()
        self.strategy = strategy or DirectReproductionStrategy()
        self.trace = TraceWriter(self.prepared.trace_path)
        self.workspace_tools = WorkspaceTools(
            self.prepared.workspace_dir,
            task.max_command_seconds,
            vision_client=vision_client,
        )
        if tool_registry_factory is None:
            self.tools = build_workspace_registry(self.workspace_tools)
        else:
            self.tools = tool_registry_factory(self.workspace_tools, task)

    def run(self) -> RunResult:
        self.memory.extend(self.strategy.initial_messages(self.task))
        self.trace.write(
            "run_started",
            task_id=self.task.task_id,
            max_steps=self.max_steps,
            max_command_seconds=self.task.max_command_seconds,
        )

        model_calls = 0
        token_usage = TokenUsage()
        for step in range(1, self.max_steps + 1):
            self.memory.extend(self.strategy.before_step(self.task, step))
            response = self.client.complete(
                self.memory.as_messages(), self.tools.definitions
            )
            model_calls += 1
            token_usage = token_usage + response.usage
            message = response.message
            self.trace.write(
                "assistant_message",
                step=step,
                message=message,
                model=response.model,
                response_id=response.response_id,
                usage=response.usage.as_dict(),
            )
            self.memory.add(message)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                self.memory.add(self.strategy.no_tool_message(step))
                continue

            observations: list[dict[str, Any]] = []
            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                name = str(function.get("name", ""))
                arguments: dict[str, Any] = {}
                try:
                    arguments = parse_tool_arguments(function.get("arguments"))
                except (ValueError, json.JSONDecodeError) as exc:
                    result = {"ok": False, "error": f"Invalid tool arguments: {exc}"}
                else:
                    result = self.tools.execute(name, arguments)
                self.trace.write(
                    "tool_result", step=step, tool=name, arguments=arguments, result=result
                )
                observations.append(
                    {"tool": name, "arguments": arguments, "result": result}
                )
                self.memory.add(
                    {
                        "role": "tool",
                        "tool_call_id": str(tool_call.get("id", f"call-{step}")),
                        "content": json.dumps(result, ensure_ascii=True),
                    }
                )

                if name == "finish" and result.get("ok"):
                    report = str(arguments.get("report_markdown", "")).strip()
                    status = str(arguments.get("status", "completed"))
                    self.prepared.report_path.write_text(report + "\n", encoding="utf-8")
                    self.trace.write("run_finished", step=step, status=status)
                    return RunResult(
                        status=status,
                        steps=step,
                        run_dir=self.prepared.run_dir,
                        report_path=self.prepared.report_path,
                        trace_path=self.prepared.trace_path,
                        model_calls=model_calls,
                        token_usage=token_usage,
                    )

            self.memory.extend(
                self.strategy.after_tools(self.task, step, observations)
            )

        report = (
            f"# Reproduction report: {self.task.task_id}\n\n"
            "Status: inconclusive\n\n"
            f"The agent reached the configured limit of {self.max_steps} steps without "
            "submitting a final report. Inspect `trace.jsonl` and the workspace artifacts.\n"
        )
        self.prepared.report_path.write_text(report, encoding="utf-8")
        self.trace.write("run_finished", step=self.max_steps, status="inconclusive")
        return RunResult(
            status="inconclusive",
            steps=self.max_steps,
            run_dir=self.prepared.run_dir,
            report_path=self.prepared.report_path,
            trace_path=self.prepared.trace_path,
            model_calls=model_calls,
            token_usage=token_usage,
        )
