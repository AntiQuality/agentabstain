from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from abstention_factory.src.utils.file_io import read_json, read_yaml


@dataclass(frozen=True)
class EvaluationBundle:
    provider: str
    model: str
    category: str
    task_id: str
    task_type: str
    run_id: str
    artifact_dir: Path
    task_dir: Path
    run_result_path: Path
    run_result: dict[str, Any]
    task_yaml: dict[str, Any]
    metadata: dict[str, Any]


def load_evaluation_bundle(run_result_path: str | Path, provider: str, model: str) -> EvaluationBundle:
    result_path = Path(run_result_path)
    run_result = read_json(result_path)

    category = str(run_result["category"])
    task_id = str(run_result["task_id"])
    task_type = str(run_result["task_type"])
    artifact_dir = result_path.parent
    task_dir = Path(run_result["task_dir"])
    if not task_dir.is_absolute():
        task_dir = (Path.cwd() / task_dir).resolve()

    task_yaml_path = task_dir / "task.yaml"
    metadata_path = task_dir.parent / "metadata.yaml"

    missing = [path for path in (task_yaml_path, metadata_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing task artifacts for evaluation: {[str(path) for path in missing]}")

    return EvaluationBundle(
        provider=provider,
        model=model,
        category=category,
        task_id=task_id,
        task_type=task_type,
        run_id=artifact_dir.name,
        artifact_dir=artifact_dir,
        task_dir=task_dir,
        run_result_path=result_path,
        run_result=run_result,
        task_yaml=read_yaml(task_yaml_path),
        metadata=read_yaml(metadata_path),
    )
