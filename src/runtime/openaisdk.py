from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from agents import Agent, ModelSettings, Runner
from agents.mcp.server import MCPServerStdio
from mcp import Tool as MCPTool
from mcp.types import CallToolResult

from src.runtime.common import (
    BatchRunSummary,
    RUNTIME_EXPORT_TOOL_NAME,
    build_runtime_server_args,
    build_task_run_result,
    coerce_final_output,
    normalize_runtime_export_payload,
    run_batch,
)
from src.types.BaseAgent import BaseAgent, TaskBundle, TaskRunResult


# OpenAI's tools API enforces tool names match `^[a-zA-Z0-9_-]+$`. The
# abstention MCP server namespaces tools with `.` (e.g.
# `email.read_email`), which Anthropic accepts but OpenAI rejects with
# HTTP 400 before any task runs. We round-trip names through `__` only
# at the OpenAI boundary: list_tools() exposes encoded names to the
# Runner (which forwards them to OpenAI), call_tool() decodes back to
# the dotted form expected by the MCP server. The MCP wire protocol,
# task artifacts, and execution log are untouched.
class _NameSafeMCPServer(MCPServerStdio):
    """MCPServerStdio that encodes `.` in tool names as `__` for OpenAI."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._encoded_to_original: dict[str, str] = {}

    @staticmethod
    def _encode(name: str) -> str:
        return name.replace(".", "__")

    def _decode(self, encoded: str) -> str:
        return self._encoded_to_original.get(encoded, encoded)

    async def list_tools(self, run_context=None, agent=None) -> list[MCPTool]:
        tools = await super().list_tools(run_context, agent)
        encoded_tools: list[MCPTool] = []
        new_mapping: dict[str, str] = {}
        for tool in tools:
            encoded = self._encode(tool.name)
            existing = new_mapping.get(encoded)
            if existing is not None and existing != tool.name:
                raise RuntimeError(
                    f"Tool name encoding collision on MCP server '{self.name}': "
                    f"both {existing!r} and {tool.name!r} encode to {encoded!r}. "
                    f"Pick a different separator in src/runtime/openaisdk.py."
                )
            new_mapping[encoded] = tool.name
            if encoded == tool.name:
                encoded_tools.append(tool)
            else:
                encoded_tools.append(tool.model_copy(update={"name": encoded}))
        self._encoded_to_original = new_mapping
        return encoded_tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        return await super().call_tool(self._decode(tool_name), arguments, meta=meta)


async def run_openai_task(agent: BaseAgent, bundle: TaskBundle, repo_root: Path) -> TaskRunResult:
    artifact_dir = agent.build_artifact_dir(bundle.category, bundle.task_id, bundle.task_type)
    final_output: str | None = None
    export_payload: dict[str, Any] | None = None
    run_error: str | None = None

    server = _NameSafeMCPServer(
        name="task_env",  # see claudesdk.py for rationale
        params={
            "command": sys.executable,
            "args": build_runtime_server_args(bundle),
            "cwd": str(repo_root),
        },
        tool_filter={"blocked_tool_names": [RUNTIME_EXPORT_TOOL_NAME]},
    )

    try:
        await server.connect()
        sdk_agent = Agent(
            name=f"{'_'.join(bundle.env_types)}_{bundle.task_type}_agent",
            instructions=bundle.task_yaml["system_prompt"],
            model=agent.model,
            model_settings=ModelSettings(temperature=agent.temperature),
            mcp_servers=[server],
        )
        try:
            run_result = await Runner.run(
                starting_agent=sdk_agent,
                input=bundle.task_yaml["instruction"],
                max_turns=agent.max_turns,
            )
            final_output = coerce_final_output(run_result.final_output)
        except Exception as exc:
            run_error = str(exc)

        try:
            export_payload = await _export_runtime_snapshot(server)
        except Exception as export_exc:
            if run_error is None:
                raise
            run_error = f"{run_error}; runtime export failed: {export_exc}"
    finally:
        try:
            await server.cleanup()
        except BaseException:
            pass

    return build_task_run_result(
        agent=agent,
        bundle=bundle,
        artifact_dir=artifact_dir,
        final_output=final_output,
        export_payload=export_payload,
        run_error=run_error,
    )


async def _export_runtime_snapshot(server: MCPServerStdio) -> dict[str, Any]:
    response = await server.call_tool(RUNTIME_EXPORT_TOOL_NAME, {})
    return normalize_runtime_export_payload(getattr(response, "structuredContent", None))
