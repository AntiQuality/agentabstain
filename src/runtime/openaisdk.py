from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents import Agent, ModelSettings, Runner
from agents.mcp.server import MCPServerStdio

from abstention_factory.src.utils.file_io import ensure_dir, write_json
from src.runtime.config import InferenceConfig
from src.types.BaseAgent import BaseAgent, TaskBundle, TaskRunResult
from src.types.trajectory import Trajectory, TrajectoryStep

RUNTIME_EXPORT_TOOL_NAME = "__runtime_export_snapshot"


@dataclass
class BatchRunSummary:
    model: str
    results_root: str
    summary_path: str
    completed: list[dict[str, Any]]
    failed: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "results_root": self.results_root,
            "summary_path": self.summary_path,
            "completed": self.completed,
            "failed": self.failed,
        }


async def run_openai_task(agent: BaseAgent, bundle: TaskBundle, repo_root: Path) -> TaskRunResult:
    artifact_dir = agent.build_artifact_dir(bundle.category, bundle.task_id, bundle.task_type)
    final_output: str | None = None
    export_payload: dict[str, Any] | None = None
    run_error: str | None = None

    server = MCPServerStdio(
        name=f"{bundle.env_type}_runtime",
        params={
            "command": sys.executable,
            "args": [
                "-m",
                "agent.openaisdk.mcp_server",
                "--category",
                bundle.category,
                "--task-id",
                bundle.task_id,
                "--task-type",
                bundle.task_type,
            ],
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
            final_output = _coerce_final_output(run_result.final_output)
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

    if export_payload is None:
        export_payload = {"state": {}, "execution_log": []}

    trajectory = _build_trajectory(
        model=agent.model,
        category=bundle.category,
        task_id=bundle.task_id,
        task_type=bundle.task_type,
        instruction=bundle.task_yaml["instruction"],
        execution_log=export_payload["execution_log"],
        final_output=final_output,
    )
    result = TaskRunResult(
        model=agent.model,
        category=bundle.category,
        task_id=bundle.task_id,
        task_type=bundle.task_type,
        task_dir=str(bundle.task_dir),
        instruction=bundle.task_yaml["instruction"],
        system_prompt=bundle.task_yaml["system_prompt"],
        final_output=final_output,
        final_state=export_payload["state"],
        execution_log=export_payload["execution_log"],
        trajectory=trajectory,
        task_metadata=bundle.metadata,
        artifact_dir=str(artifact_dir),
        error=run_error,
    )
    agent.persist_run_artifacts(result)
    return result


def run_batch(agent: BaseAgent, config: InferenceConfig) -> BatchRunSummary:
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for task in config.tasks:
        try:
            result = agent.run(task.category, task.task_id, task.task_type)
            summary_item = {
                "category": task.category,
                "task_id": result.task_id,
                "task_type": task.task_type,
                "artifact_dir": result.artifact_dir,
                "final_output": result.final_output,
                "error": result.error,
            }
            if result.error is None:
                completed.append(summary_item)
            else:
                failed.append(summary_item)
        except Exception as exc:
            failed.append(
                {
                    "category": task.category,
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "error": str(exc),
                }
            )

    summary_path = _write_batch_summary(agent, config, completed, failed)
    return BatchRunSummary(
        model=agent.model,
        results_root=str(agent.results_root),
        summary_path=str(summary_path),
        completed=completed,
        failed=failed,
    )


async def _export_runtime_snapshot(server: MCPServerStdio) -> dict[str, Any]:
    response = await server.call_tool(RUNTIME_EXPORT_TOOL_NAME, {})
    structured = getattr(response, "structuredContent", None)
    if isinstance(structured, dict) and set(structured) == {"result"}:
        structured = structured["result"]
    if not isinstance(structured, dict):
        raise ValueError(f"Unexpected runtime export payload: {structured!r}")
    if "state" not in structured or "execution_log" not in structured:
        raise ValueError(f"Incomplete runtime export payload: {structured!r}")
    return structured


def _build_trajectory(
    model: str,
    category: str,
    task_id: str,
    task_type: str,
    instruction: str,
    execution_log: list[dict[str, Any]],
    final_output: str | None,
) -> Trajectory:
    trajectory = Trajectory(
        model=model,
        category=category,
        task_id=task_id,
        task_type=task_type,
    )
    trajectory.append(TrajectoryStep(type="user", content=instruction))

    for entry in execution_log:
        trajectory.append(
            TrajectoryStep(
                type="tool_call",
                tool=entry.get("tool"),
                params=entry.get("params"),
            )
        )
        trajectory.append(
            TrajectoryStep(
                type="tool_result",
                tool=entry.get("tool"),
                result=entry.get("result"),
                success=entry.get("success"),
                error=entry.get("error"),
            )
        )

    trajectory.append(TrajectoryStep(type="assistant", content=final_output or ""))
    return trajectory


def _coerce_final_output(final_output: Any) -> str | None:
    if final_output is None:
        return None
    if isinstance(final_output, str):
        return final_output
    return str(final_output)


def _write_batch_summary(
    agent: BaseAgent,
    config: InferenceConfig,
    completed: list[dict[str, Any]],
    failed: list[dict[str, Any]],
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_dir = agent.results_root / agent.model / "batch_runs" / timestamp
    ensure_dir(summary_dir)
    summary_path = summary_dir / "summary.json"
    write_json(
        summary_path,
        {
            "config": config.to_dict(),
            "completed": completed,
            "failed": failed,
        },
    )
    return summary_path
