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
    # `initial_states` maps env_name → sub-env initial state. For the legacy
    # single-env path (no new artifacts yet), this dict has one key.
    initial_states: dict[str, dict[str, Any]]
    metadata: dict[str, Any]
    env_types: list[str]


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

        # Pure numeric: zero-pad and add default "task_" prefix
        if raw.isdigit():
            return f"task_{raw.zfill(3)}"

        # Already has a prefix (e.g. "dev_v1_001", "task_001"): use as-is
        # but zero-pad trailing numeric part if present
        if "_" in raw:
            parts = raw.rsplit("_", 1)
            if parts[-1].isdigit():
                return f"{parts[0]}_{parts[-1].zfill(3)}"
            return raw

        # Legacy: "task_NNN" pattern
        if raw.startswith("task_"):
            suffix = raw[5:]
            if suffix.isdigit():
                return f"task_{suffix.zfill(3)}"

        return raw

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
        initial_states_dir = task_dir / "initial_states"
        metadata_path = task_dir.parent / "metadata.yaml"

        missing = [
            str(path)
            for path in (task_yaml_path, initial_states_dir, metadata_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"Missing task artifacts: {missing}")

        task_yaml = read_yaml(task_yaml_path)
        metadata = read_yaml(metadata_path)
        env_types = metadata.get("environments")
        if not env_types or not isinstance(env_types, list):
            raise ValueError(
                f"Task metadata missing list 'environments': {metadata_path}"
            )

        initial_states: dict[str, Any] = {}
        for env_name in env_types:
            per_env = initial_states_dir / f"{env_name}.json"
            if not per_env.exists():
                raise FileNotFoundError(
                    f"Missing initial state for env {env_name!r}: {per_env}"
                )
            initial_states[env_name] = read_json(per_env)

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
            initial_states=initial_states,
            metadata=metadata,
            env_types=list(env_types),
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
