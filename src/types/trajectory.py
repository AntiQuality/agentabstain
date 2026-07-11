from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


TrajectoryStepType = Literal["user", "assistant", "tool_call", "tool_result"]


@dataclass
class TrajectoryStep:
    type: TrajectoryStepType
    content: str | None = None
    tool: str | None = None
    params: dict[str, Any] | None = None
    result: Any = None
    success: bool | None = None
    error: str | None = None
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type}
        if self.content is not None:
            payload["content"] = self.content
        if self.tool is not None:
            payload["tool"] = self.tool
        if self.params is not None:
            payload["params"] = self.params
        if self.result is not None or self.type == "tool_result":
            payload["result"] = self.result
        if self.success is not None:
            payload["success"] = self.success
        if self.error is not None:
            payload["error"] = self.error
        if self.timestamp is not None:
            payload["timestamp"] = self.timestamp
        return payload


@dataclass
class Trajectory:
    model: str
    category: str
    task_id: str
    task_type: str
    steps: list[TrajectoryStep] = field(default_factory=list)

    def append(self, step: TrajectoryStep) -> None:
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "category": self.category,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "steps": [step.to_dict() for step in self.steps],
        }
