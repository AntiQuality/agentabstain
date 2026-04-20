from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from claude_code_sdk import (
    ClaudeSDKClient,
    ClaudeCodeOptions,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
)
from mcp.client.stdio import StdioServerParameters

from agent.claudesdk.mcp_bridge import RuntimeMcpBridge
from src.runtime.common import (
    BatchRunSummary,
    RUNTIME_EXPORT_TOOL_NAME,
    build_runtime_server_args,
    build_task_run_result,
    normalize_runtime_export_payload,
    run_batch,
)
from src.types.BaseAgent import BaseAgent, TaskBundle, TaskRunResult


async def run_claudesdk_task(agent: BaseAgent, bundle: TaskBundle, repo_root: Path) -> TaskRunResult:
    artifact_dir = agent.build_artifact_dir(bundle.category, bundle.task_id, bundle.task_type)
    final_output: str | None = None
    export_payload: dict[str, Any] | None = None
    run_error: str | None = None
    # Keep app_name short — Claude Code SDK prefixes tool names as
    # `mcp__{app_name}__{mcp_tool_name}` before sending to Anthropic, whose
    # 128-char tool-name limit is easy to bust when multiple env names
    # concatenate into `app_name`. MCP tool names can already approach 100
    # chars post-`.`→`_` munging; keeping app_name short preserves
    # headroom.
    app_name = "task_env"

    bridge = RuntimeMcpBridge(
        connection_params=StdioServerParameters(
            command=sys.executable,
            args=build_runtime_server_args(bundle),
            cwd=str(repo_root),
        ),
        hidden_tool_names={RUNTIME_EXPORT_TOOL_NAME},
        server_name=app_name,
    )

    try:
        await bridge.connect()
        sdk_server_config = bridge.get_sdk_server_config()

        options = ClaudeCodeOptions(
            system_prompt=bundle.task_yaml["system_prompt"],
            mcp_servers={app_name: sdk_server_config},
            permission_mode="bypassPermissions",
            max_turns=agent.max_turns,
            model=agent.model,
        )

        client = ClaudeSDKClient(options=options)
        await client.connect()

        try:
            try:
                await client.query(bundle.task_yaml["instruction"])
                messages = []
                async for message in client.receive_response():
                    messages.append(message)
                final_output = _extract_final_output(messages)
            except Exception as exc:
                run_error = str(exc)
            finally:
                await client.disconnect()

            try:
                response = await bridge.call_hidden_tool(RUNTIME_EXPORT_TOOL_NAME, {})
                export_payload = normalize_runtime_export_payload(
                    getattr(response, "structuredContent", None)
                )
            except Exception as export_exc:
                if run_error is None:
                    raise
                run_error = f"{run_error}; runtime export failed: {export_exc}"
        except Exception:
            if run_error is None:
                raise
    finally:
        try:
            await bridge.close()
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


def _extract_final_output(messages: list[Any]) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, AssistantMessage):
            continue
        texts = [
            block.text
            for block in message.content
            if isinstance(block, TextBlock)
        ]
        if texts:
            return "\n".join(texts)
    return None
