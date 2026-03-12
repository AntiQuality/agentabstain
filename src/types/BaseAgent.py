from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from abstention_factory.src.utils.file_io import ensure_dir, read_json, read_yaml, write_json
from src.types.trajectory import Trajectory


@dataclass
class TaskBundle:
    category: str
    task_id: str
    task_type: str
    task_dir: Path
    task_yaml: dict[str, Any]
    initial_state: dict[str, Any]
    metadata: dict[str, Any]
    env_type: str


@dataclass
class TaskRunResult:
    model: str
    category: str
    task_id: str
    task_type: str
    task_dir: str
    instruction: str
    system_prompt: str
    final_output: str | None
    final_state: dict[str, Any]
    execution_log: list[dict[str, Any]]
    trajectory: Trajectory
    task_metadata: dict[str, Any]
    artifact_dir: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "category": self.category,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "task_dir": self.task_dir,
            "instruction": self.instruction,
            "system_prompt": self.system_prompt,
            "final_output": self.final_output,
            "final_state": self.final_state,
            "execution_log": self.execution_log,
            "trajectory": self.trajectory.to_dict(),
            "task_metadata": self.task_metadata,
            "artifact_dir": self.artifact_dir,
            "error": self.error,
        }


class BaseAgent(ABC):
    TASKS_ROOT = Path("abstention_factory/tasks")

    def __init__(self, model: str, temperature: float, max_turns: int, results_root: str | Path):
        self.model = model
        self.temperature = temperature
        self.max_turns = max_turns
        self.results_root = Path(results_root)

    def run(self, category: str, task_id: str | int, task_type: str) -> TaskRunResult:
        return asyncio.run(self.arun(category, task_id, task_type))

    @abstractmethod
    async def arun(self, category: str, task_id: str | int, task_type: str) -> TaskRunResult:
        """Run one task and return runtime artifacts."""

    @classmethod
    def normalize_task_id(cls, task_id: str | int) -> str:
        raw = str(task_id).strip()
        if raw.startswith("task_"):
            suffix = raw[5:]
        else:
            suffix = raw

        if not suffix.isdigit():
            raise ValueError(f"task_id must be numeric or task_NNN, got: {task_id}")
        return f"task_{suffix.zfill(3)}"

    @classmethod
    def resolve_task_dir(cls, category: str, task_id: str | int, task_type: str) -> Path:
        if task_type not in {"act", "abstain"}:
            raise ValueError(f"task_type must be 'act' or 'abstain', got: {task_type}")
        normalized_task_id = cls.normalize_task_id(task_id)
        return cls.TASKS_ROOT / category / normalized_task_id / task_type

    @classmethod
    def load_task_bundle(cls, category: str, task_id: str | int, task_type: str) -> TaskBundle:
        task_dir = cls.resolve_task_dir(category, task_id, task_type)
        task_yaml_path = task_dir / "task.yaml"
        initial_state_path = task_dir / "initial_state.json"
        metadata_path = task_dir.parent / "metadata.yaml"

        missing = [str(path) for path in (task_yaml_path, initial_state_path, metadata_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing task artifacts: {missing}")

        task_yaml = read_yaml(task_yaml_path)
        initial_state = read_json(initial_state_path)
        metadata = read_yaml(metadata_path)
        env_type = metadata.get("environment")
        if not env_type:
            raise ValueError(f"Task metadata missing 'environment': {metadata_path}")

        if task_type == "act" and "expected_tool_sequence" not in task_yaml:
            raise ValueError(f"Act task is missing expected_tool_sequence: {task_yaml_path}")
        if task_type == "abstain" and "abstention_trigger" not in task_yaml:
            raise ValueError(f"Abstain task is missing abstention_trigger: {task_yaml_path}")

        return TaskBundle(
            category=category,
            task_id=cls.normalize_task_id(task_id),
            task_type=task_type,
            task_dir=task_dir,
            task_yaml=task_yaml,
            initial_state=initial_state,
            metadata=metadata,
            env_type=env_type,
        )

    def build_artifact_dir(self, category: str, task_id: str, task_type: str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        artifact_dir = self.results_root / self.model / category / task_id / task_type / timestamp
        ensure_dir(artifact_dir)
        return artifact_dir

    def persist_run_artifacts(self, result: TaskRunResult) -> None:
        artifact_dir = Path(result.artifact_dir)
        ensure_dir(artifact_dir)
        write_json(artifact_dir / "run_result.json", result.to_dict())
        write_json(artifact_dir / "final_state.json", result.final_state)
        write_json(artifact_dir / "trajectory.json", result.trajectory.to_dict())
