from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ..llm import ChatClient, TokenUsage, VisionClient
from ..memory import ConversationMemory
from ..runtime import PreparedWorkspace, TraceWriter, prepare_run
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
        prepared_workspace: PreparedWorkspace | None = None,
        start_step: int = 1,
        resume_context: str = "",
        tool_registry_factory: (
            Callable[[WorkspaceTools, TaskSpec], ToolRegistry] | None
        ) = None,
    ) -> None:
        if prepared_workspace is not None and output_dir is not None:
            raise ValueError("output_dir cannot be used with prepared_workspace")
        self.prepared = prepared_workspace or prepare_run(
            task, output_dir, vision_client=vision_client
        )
        self.task = task
        self.client = client
        self.max_steps = task.max_agent_steps if max_steps is None else max_steps
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if start_step < 1:
            raise ValueError("start_step must be at least 1")
        self.start_step = start_step
        self.resume_context = resume_context.strip()
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
        end_step = self.start_step + self.max_steps - 1
        if self.resume_context:
            self.memory.add(
                {
                    "role": "user",
                    "content": (
                        f"Resume this run at agent step {self.start_step}. "
                        f"You have {self.max_steps} additional steps, ending at step "
                        f"{end_step}. Use the compact handoff below and continue the "
                        "pending work without repeating prior investigation.\n\n"
                        + self.resume_context
                    ),
                }
            )
            self.trace.write(
                "run_resumed",
                task_id=self.task.task_id,
                previous_step=self.start_step - 1,
                additional_steps=self.max_steps,
                end_step=end_step,
                resume_context="workspace/resume_context.md",
            )
        else:
            self.trace.write(
                "run_started",
                task_id=self.task.task_id,
                max_steps=self.max_steps,
                max_command_seconds=self.task.max_command_seconds,
            )

        model_calls = 0
        token_usage = TokenUsage()
        for step in range(self.start_step, end_step + 1):
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
            f"The agent reached the configured limit at step {end_step} without "
            "submitting a final report. Inspect `trace.jsonl` and the workspace artifacts.\n"
        )
        self.prepared.report_path.write_text(report, encoding="utf-8")
        self.trace.write("run_finished", step=end_step, status="inconclusive")
        return RunResult(
            status="inconclusive",
            steps=end_step,
            run_dir=self.prepared.run_dir,
            report_path=self.prepared.report_path,
            trace_path=self.prepared.trace_path,
            model_calls=model_calls,
            token_usage=token_usage,
        )
