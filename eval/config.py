from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from abstention_factory.src.utils.file_io import read_yaml


@dataclass(frozen=True)
class JudgeModelConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JudgeModelConfig":
        missing = [key for key in ("provider", "model", "temperature", "max_tokens") if key not in payload]
        if missing:
            raise ValueError(f"Judge model config missing required keys: {missing}")
        return cls(
            provider=str(payload["provider"]),
            model=str(payload["model"]),
            temperature=float(payload["temperature"]),
            max_tokens=int(payload["max_tokens"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


@dataclass(frozen=True)
class EvaluationConfig:
    judge_models: list[JudgeModelConfig]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvaluationConfig":
        payload = read_yaml(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Evaluation config must be a mapping: {path}")
        judges_payload = payload.get("judge_models")
        if not isinstance(judges_payload, list) or not judges_payload:
            raise ValueError("Evaluation config must include a non-empty judge_models list")
        judge_models = [JudgeModelConfig.from_dict(item) for item in judges_payload]
        return cls(judge_models=judge_models)

    def to_dict(self) -> dict[str, Any]:
        return {"judge_models": [item.to_dict() for item in self.judge_models]}
