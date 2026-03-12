from __future__ import annotations

import sys
from pathlib import Path

from agents import Agent, ModelSettings, Runner
from agents.mcp.server import MCPServerStdio

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


async def run_openai_task(agent: BaseAgent, bundle: TaskBundle, repo_root: Path) -> TaskRunResult:
    artifact_dir = agent.build_artifact_dir(bundle.category, bundle.task_id, bundle.task_type)
    final_output: str | None = None
    export_payload: dict[str, Any] | None = None
    run_error: str | None = None

    server = MCPServerStdio(
        name=f"{bundle.env_type}_runtime",
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
            name=f"{bundle.env_type}_{bundle.task_type}_agent",
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
