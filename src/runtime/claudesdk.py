from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from claude_code_sdk import (
    ClaudeSDKClient,
    ClaudeCodeOptions,
    AssistantMessage,
    ResultMessage,
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
    # Sandbox cwd: an empty per-rollout directory under the artifact
    # dir. Claude Code auto-loads CLAUDE.md / .claude/settings.json /
    # AGENTS.md and walks up the cwd tree picking up parent CLAUDE.mds;
    # pointing it at an empty sibling directory cuts that channel
    # entirely while keeping the path stable so resume_session can
    # locate the session post-rollout. The MCP bridge subprocess keeps
    # its own cwd=repo_root for tool imports (independent process).
    sandbox_dir = artifact_dir / "claudesdk_sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    final_output: str | None = None
    export_payload: dict[str, Any] | None = None
    run_error: str | None = None
    session_id: str | None = None
    result_message_meta: dict[str, Any] | None = None
    # Keep app_name short — Claude Code SDK prefixes tool names as
    # `mcp__{app_name}__{mcp_tool_name}` before sending to Anthropic, whose
    # 128-char tool-name limit is easy to bust when multiple env names
    # concatenate into `app_name`. MCP tool names can already approach 100
    # chars post-`.`→`_` munging; keeping app_name short preserves
    # headroom.
    app_name = "task_env"

    # Deny every Claude Code built-in tool. The agent must operate
    # exclusively through env-specific MCP tools we bridge in via
    # `mcp_servers`. Without this denylist, `permission_mode=
    # "bypassPermissions"` lets the model silently call Read / Glob /
    # Bash / WebFetch and discover repo context (CLAUDE.md, .background/
    # taxonomy, task source code) that biases its abstention decision
    # — verified empirically: Sonnet's run-1 response named the
    # "AgentAbstain" benchmark, a string that only appears in CLAUDE.md.
    # Listed by canonical Claude Code tool name.
    builtin_tools_denied = [
        "Bash",
        "BashOutput",
        "KillBash",
        "Read",
        "Glob",
        "Grep",
        "Edit",
        "Write",
        "NotebookEdit",
        "WebFetch",
        "WebSearch",
        "TodoWrite",
        "Task",
        "ExitPlanMode",
    ]

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
            cwd=str(sandbox_dir),
            disallowed_tools=builtin_tools_denied,
        )

        client = ClaudeSDKClient(options=options)
        await client.connect()

        try:
            messages: list[Any] = []
            try:
                await client.query(bundle.task_yaml["instruction"])
                async for message in client.receive_response():
                    messages.append(message)
                final_output = _extract_final_output(messages)
            except Exception as exc:
                run_error = str(exc)
            finally:
                # Recover session_id even on partial/failed turns so the
                # operator can resume into the partial session for
                # post-mortem. Then disconnect — order matters: the SDK
                # may invalidate `messages` on disconnect in some
                # transports.
                try:
                    session_id, result_message_meta = _extract_session_metadata(messages)
                except Exception:
                    pass
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

    provider_metadata: dict[str, Any] = {
        "provider": "claudesdk",
        "session_id": session_id,
        # cwd Claude Code SDK actually ran under (the sandbox), captured
        # so resume can be done from the same workdir — Claude Code
        # session storage is keyed by cwd path. We persist the sandbox
        # path, not the repo_root, because that's the path the SDK saw.
        "cwd": str(sandbox_dir),
        "repo_root": str(repo_root),
        "model": agent.model,
        "disallowed_tools": list(builtin_tools_denied),
        "context_isolation": "sandboxed_cwd_no_builtin_tools",
    }
    if result_message_meta:
        provider_metadata["result_message"] = result_message_meta

    return build_task_run_result(
        agent=agent,
        bundle=bundle,
        artifact_dir=artifact_dir,
        final_output=final_output,
        export_payload=export_payload,
        run_error=run_error,
        provider_metadata=provider_metadata,
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


def _extract_session_metadata(messages: list[Any]) -> tuple[str | None, dict[str, Any] | None]:
    """Pull session_id (and a small slice of ResultMessage metadata) out of
    the message stream so the run can be resumed later via
    `ClaudeCodeOptions(resume=session_id)`. The SDK emits a ResultMessage as
    the terminal message of a turn; iterate from the end for an O(1) hit on
    the common case.
    """
    for message in reversed(messages):
        if isinstance(message, ResultMessage):
            meta = {
                "subtype": message.subtype,
                "duration_ms": message.duration_ms,
                "duration_api_ms": message.duration_api_ms,
                "is_error": message.is_error,
                "num_turns": message.num_turns,
                "total_cost_usd": message.total_cost_usd,
                "usage": message.usage,
            }
            return message.session_id, meta
    return None, None
