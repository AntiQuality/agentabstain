from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

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
    skipped: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model": self.model,
            "results_root": self.results_root,
            "summary_path": self.summary_path,
            "completed": self.completed,
            "failed": self.failed,
        }
        if self.skipped:
            d["skipped"] = self.skipped
        return d


def build_runtime_server_args(bundle: TaskBundle) -> list[str]:
    return [
        "-m",
        "src.runtime.task_mcp_server",
        "--category",
        bundle.category,
        "--task-id",
        bundle.task_id,
        "--task-type",
        bundle.task_type,
    ]


def normalize_runtime_export_payload(payload: Any) -> dict[str, Any]:
    structured = payload
    if isinstance(structured, dict) and set(structured) == {"result"}:
        structured = structured["result"]
    if not isinstance(structured, dict):
        raise ValueError(f"Unexpected runtime export payload: {structured!r}")
    if "state" not in structured or "execution_log" not in structured:
        raise ValueError(f"Incomplete runtime export payload: {structured!r}")
    return {
        "state": structured["state"],
        "execution_log": structured["execution_log"],
    }


def default_runtime_export_payload() -> dict[str, Any]:
    return {
        "state": {},
        "execution_log": [],
    }


def coerce_final_output(final_output: Any) -> str | None:
    if final_output is None:
        return None
    if isinstance(final_output, str):
        return final_output
    return str(final_output)


def build_trajectory(
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


def build_task_run_result(
    agent: BaseAgent,
    bundle: TaskBundle,
    artifact_dir: Path,
    final_output: str | None,
    export_payload: dict[str, Any] | None,
    run_error: str | None,
) -> TaskRunResult:
    resolved_payload = export_payload or default_runtime_export_payload()
    trajectory = build_trajectory(
        model=agent.model,
        category=bundle.category,
        task_id=bundle.task_id,
        task_type=bundle.task_type,
        instruction=bundle.task_yaml["instruction"],
        execution_log=resolved_payload["execution_log"],
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
        final_state=resolved_payload["state"],
        execution_log=resolved_payload["execution_log"],
        trajectory=trajectory,
        task_metadata=bundle.metadata,
        artifact_dir=str(artifact_dir),
        error=run_error,
    )
    agent.persist_run_artifacts(result)
    return result


def _has_existing_run(agent: BaseAgent, task: "InferenceTaskConfig") -> bool:
    task_id = BaseAgent.normalize_task_id(task.task_id)
    task_dir = agent.results_root / agent.model / task.category / task_id / task.task_type
    if not task_dir.exists():
        return False
    return any(task_dir.iterdir())


def run_batch(agent: BaseAgent, config: InferenceConfig, *, multi_run: bool = False) -> BatchRunSummary:
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    pbar = tqdm(config.tasks, desc=f"{agent.model}", unit="task")
    for task in pbar:
        pbar.set_postfix_str(f"{task.category}/{task.task_id}/{task.task_type}")

        if not multi_run and _has_existing_run(agent, task):
            skipped.append({
                "category": task.category,
                "task_id": BaseAgent.normalize_task_id(task.task_id),
                "task_type": task.task_type,
                "skipped": True,
            })
            pbar.set_description(f"{agent.model} [✓{len(completed)} ✗{len(failed)} ⏭{len(skipped)}]")
            continue

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
        pbar.set_description(f"{agent.model} [✓{len(completed)} ✗{len(failed)}]")

    summary_path = _write_batch_summary(agent, config, completed, failed, skipped)
    return BatchRunSummary(
        model=agent.model,
        results_root=str(agent.results_root),
        summary_path=str(summary_path),
        completed=completed,
        failed=failed,
        skipped=skipped or None,
    )


def _write_batch_summary(
    agent: BaseAgent,
    config: InferenceConfig,
    completed: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    skipped: list[dict[str, Any]] | None = None,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_dir = agent.results_root / agent.model / "batch_runs" / timestamp
    ensure_dir(summary_dir)
    summary_path = summary_dir / "summary.json"
    payload: dict[str, Any] = {
        "config": config.to_dict(),
        "completed": completed,
        "failed": failed,
    }
    if skipped:
        payload["skipped"] = skipped
    write_json(summary_path, payload)
    return summary_path
