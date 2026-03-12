from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from abstention_factory.src.utils.file_io import read_yaml


@dataclass
class InferenceTaskConfig:
    category: str
    task_id: str
    task_type: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InferenceTaskConfig":
        missing = [key for key in ("category", "task_id", "task_type") if key not in payload]
        if missing:
            raise ValueError(f"Task config missing required keys: {missing}")
        return cls(
            category=str(payload["category"]),
            task_id=str(payload["task_id"]),
            task_type=str(payload["task_type"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "task_id": self.task_id,
            "task_type": self.task_type,
        }


@dataclass
class InferenceConfig:
    model: str
    temperature: float
    max_turns: int
    results_root: str
    tasks: list[InferenceTaskConfig]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InferenceConfig":
        payload = read_yaml(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Inference config must be a mapping: {path}")

        missing = [key for key in ("model", "temperature", "max_turns", "results_root", "tasks") if key not in payload]
        if missing:
            raise ValueError(f"Inference config missing required keys: {missing}")

        tasks_payload = payload["tasks"]
        if not isinstance(tasks_payload, list) or not tasks_payload:
            raise ValueError("Inference config 'tasks' must be a non-empty list")

        return cls(
            model=str(payload["model"]),
            temperature=float(payload["temperature"]),
            max_turns=int(payload["max_turns"]),
            results_root=str(payload["results_root"]),
            tasks=[InferenceTaskConfig.from_dict(item) for item in tasks_payload],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_turns": self.max_turns,
            "results_root": self.results_root,
            "tasks": [task.to_dict() for task in self.tasks],
        }
