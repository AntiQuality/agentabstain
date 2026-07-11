from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from abstention_factory.src.utils.file_io import read_yaml

DEFAULT_PROVIDER = "openaisdk"
SUPPORTED_PROVIDERS = frozenset({"openaisdk", "googleadk", "claudesdk", "openclaw"})


def _normalize_provider(provider: Any) -> str:
    value = DEFAULT_PROVIDER if provider is None else str(provider)
    if value not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {value}. Expected one of {sorted(SUPPORTED_PROVIDERS)}")
    return value


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
class RuntimeConfig:
    provider: str
    model: str
    temperature: float
    max_turns: int
    results_root: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RuntimeConfig":
        payload = read_yaml(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Runtime config must be a mapping: {path}")

        missing = [key for key in ("model", "temperature", "max_turns", "results_root") if key not in payload]
        if missing:
            raise ValueError(f"Runtime config missing required keys: {missing}")

        return cls(
            provider=_normalize_provider(payload.get("provider")),
            model=str(payload["model"]),
            temperature=float(payload["temperature"]) if payload["temperature"] is not None else None,
            max_turns=int(payload["max_turns"]),
            results_root=str(payload["results_root"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_turns": self.max_turns,
            "results_root": self.results_root,
        }


@dataclass
class TaskListConfig:
    tasks: list[InferenceTaskConfig]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaskListConfig":
        payload = read_yaml(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Task config must be a mapping: {path}")
        if "tasks" not in payload:
            raise ValueError(f"Task config missing required key: tasks")

        tasks_payload = payload["tasks"]
        if not isinstance(tasks_payload, list) or not tasks_payload:
            raise ValueError("Task config 'tasks' must be a non-empty list")

        return cls(tasks=[InferenceTaskConfig.from_dict(item) for item in tasks_payload])

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": [task.to_dict() for task in self.tasks],
        }


@dataclass
class InferenceConfig:
    runtime: RuntimeConfig
    task_list: TaskListConfig

    @property
    def model(self) -> str:
        return self.runtime.model

    @property
    def provider(self) -> str:
        return self.runtime.provider

    @property
    def temperature(self) -> float:
        return self.runtime.temperature

    @property
    def max_turns(self) -> int:
        return self.runtime.max_turns

    @property
    def results_root(self) -> str:
        return self.runtime.results_root

    @property
    def tasks(self) -> list[InferenceTaskConfig]:
        return self.task_list.tasks

    @classmethod
    def from_files(cls, runtime_path: str | Path, task_path: str | Path) -> "InferenceConfig":
        return cls(
            runtime=RuntimeConfig.from_yaml(runtime_path),
            task_list=TaskListConfig.from_yaml(task_path),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InferenceConfig":
        payload = read_yaml(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Inference config must be a mapping: {path}")
        if "tasks" not in payload:
            raise ValueError("Combined inference config must include 'tasks'")

        runtime = RuntimeConfig(
            provider=_normalize_provider(payload.get("provider")),
            model=str(payload["model"]),
            temperature=float(payload["temperature"]),
            max_turns=int(payload["max_turns"]),
            results_root=str(payload["results_root"]),
        )
        task_list = TaskListConfig(tasks=[InferenceTaskConfig.from_dict(item) for item in payload["tasks"]])
        return cls(runtime=runtime, task_list=task_list)

    def to_dict(self) -> dict[str, Any]:
        payload = self.runtime.to_dict()
        payload.update(self.task_list.to_dict())
        return payload
