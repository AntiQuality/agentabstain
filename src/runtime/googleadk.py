from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.runners import RunConfig, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_session_manager import StdioServerParameters
from google.genai import types

from agent.googleadk.toolset import RuntimeMcpToolset
from src.runtime.common import (
    BatchRunSummary,
    RUNTIME_EXPORT_TOOL_NAME,
    build_runtime_server_args,
    build_task_run_result,
    normalize_runtime_export_payload,
    run_batch,
)
from src.types.BaseAgent import BaseAgent, TaskBundle, TaskRunResult


async def run_googleadk_task(agent: BaseAgent, bundle: TaskBundle, repo_root: Path) -> TaskRunResult:
    artifact_dir = agent.build_artifact_dir(bundle.category, bundle.task_id, bundle.task_type)
    final_output: str | None = None
    export_payload: dict[str, Any] | None = None
    run_error: str | None = None
    # See claudesdk.py: keep app_name short to preserve tool-name headroom.
    app_name = "task_env"

    toolset = RuntimeMcpToolset(
        connection_params=StdioServerParameters(
            command=sys.executable,
            args=build_runtime_server_args(bundle),
            cwd=str(repo_root),
        ),
        hidden_tool_names={RUNTIME_EXPORT_TOOL_NAME},
    )
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    runtime_agent = LlmAgent(
        model=agent.model,
        name=f"{'_'.join(bundle.env_types)}_{bundle.task_type}_agent",
        instruction=bundle.task_yaml["system_prompt"],
        tools=[toolset],
        generate_content_config=types.GenerateContentConfig(temperature=agent.temperature),
    )
    runner = Runner(
        app_name=app_name,
        agent=runtime_agent,
        artifact_service=artifact_service,
        session_service=session_service,
    )
    session = await session_service.create_session(
        state={},
        app_name=app_name,
        user_id="user_default",
    )

    usage_totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "events_with_usage": 0,
    }
    try:
        try:
            events = runner.run_async(
                session_id=session.id,
                user_id=session.user_id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=bundle.task_yaml["instruction"])],
                ),
                run_config=RunConfig(max_llm_calls=agent.max_turns),
            )
            async for event in events:
                _accumulate_event_usage(event, usage_totals)
                event_output = _extract_final_output_from_event(event)
                if event_output is not None:
                    final_output = event_output
        except Exception as exc:
            run_error = str(exc)

        try:
            response = await toolset.call_hidden_tool(RUNTIME_EXPORT_TOOL_NAME, {})
            export_payload = normalize_runtime_export_payload(response.structuredContent)
        except Exception as export_exc:
            if run_error is None:
                raise
            run_error = f"{run_error}; runtime export failed: {export_exc}"
    finally:
        try:
            await toolset.close()
        except BaseException:
            pass

    provider_metadata: dict[str, Any] = {
        "provider": "googleadk",
        "model": agent.model,
    }
    if usage_totals["events_with_usage"] > 0:
        provider_metadata["usage"] = usage_totals

    return build_task_run_result(
        agent=agent,
        bundle=bundle,
        artifact_dir=artifact_dir,
        final_output=final_output,
        export_payload=export_payload,
        run_error=run_error,
        provider_metadata=provider_metadata,
    )


def _accumulate_event_usage(event: Any, totals: dict[str, int]) -> None:
    """Sum google-adk per-event `usage_metadata` (gemini token counts)
    into a running total. Schema field names follow Vertex's naming
    (prompt_token_count / candidates_token_count / total_token_count).
    Best-effort: any shape change just leaves totals at 0 and we keep going.
    """
    um = getattr(event, "usage_metadata", None)
    if um is None:
        return
    try:
        prompt = int(getattr(um, "prompt_token_count", 0) or 0)
        out = int(getattr(um, "candidates_token_count", 0) or 0)
        total = int(getattr(um, "total_token_count", 0) or 0)
    except Exception:
        return
    if prompt == 0 and out == 0 and total == 0:
        return
    totals["input_tokens"] += prompt
    totals["output_tokens"] += out
    totals["total_tokens"] += total or (prompt + out)
    totals["events_with_usage"] += 1


def _extract_final_output_from_event(event: Any) -> str | None:
    final_checker = getattr(event, "is_final_response", None)
    if not callable(final_checker):
        return None

    try:
        if not final_checker():
            return None
    except Exception:
        return None

    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    texts = [text for text in (getattr(part, "text", None) for part in parts) if text]
    if not texts:
        return None
    return "\n".join(texts)
