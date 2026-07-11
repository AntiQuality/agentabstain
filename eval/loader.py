from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from abstention_factory.src.utils.file_io import read_json, read_yaml

# Current schema: `critical_actions` is a `list[str]` of namespaced tool
# names. Older corpora on disk use `[{node, must_yield}]`. The loader
# runs strict by default so regenerated tasks fail-loudly on accidental
# drift; setting `ALLOW_LEGACY_TASK_YAML=1` enables a transitional shim
# that accepts dict-shaped entries and resolves them to an empty
# critical_actions set — which makes commit_check on those tasks
# degenerate (always pass) but keeps the rest of the eval pipeline
# (response_llm_judge, identity drift handling) functional while a
# full re-generation of the task corpus is in progress.
_LEGACY_TASK_YAML_OK = os.environ.get("ALLOW_LEGACY_TASK_YAML") == "1"


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
    # Read directly from the on-disk persisted list. Empty set for
    # informational tasks (no commit in the act DAG).
    critical_actions: set[str] = field(default_factory=set)
    # action_type from metadata.yaml drives per-action_type analysis
    # in the statistics/reporting layer.
    action_type: str = ""


def load_evaluation_bundle(run_result_path: str | Path, provider: str, model: str) -> EvaluationBundle:
    result_path = Path(run_result_path)
    run_result = read_json(result_path)

    category = str(run_result["category"])
    task_id = str(run_result["task_id"])
    raw_task_type = str(run_result["task_type"])
    normalized = raw_task_type.strip().lower()
    task_type = normalized if normalized in {"act", "abstain"} else raw_task_type
    artifact_dir = result_path.parent
    task_dir = Path(run_result["task_dir"])
    if not task_dir.is_absolute():
        task_dir = (Path.cwd() / task_dir).resolve()

    task_yaml_path = task_dir / "task.yaml"
    metadata_path = task_dir.parent / "metadata.yaml"

    missing = [path for path in (task_yaml_path, metadata_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing task artifacts for evaluation: {[str(path) for path in missing]}")

    task_yaml = read_yaml(task_yaml_path)
    metadata = read_yaml(metadata_path)

    # Trust the persisted list. `WriteTaskPairNode` derives it from the
    # act DAG + composed env tool_kinds at write time, and the disk-vs-
    # derived CI test guards against drift.
    raw_critical = task_yaml.get("critical_actions") or []
    if not isinstance(raw_critical, list):
        raise ValueError(
            f"task.yaml at {task_yaml_path} has critical_actions of type "
            f"{type(raw_critical).__name__}; expected list[str]."
        )
    legacy_dict_entries = [e for e in raw_critical if isinstance(e, dict)]
    if legacy_dict_entries and _LEGACY_TASK_YAML_OK:
        # Legacy corpus fallback. We cannot faithfully convert old dict
        # entries (which key on DAG node ids, not tool kinds) into a
        # commit-tool set without the env's tool_kinds map — for tasks
        # whose env doesn't carry kinds, the map is empty. Degenerate
        # to an empty critical_actions set so commit_check trivially
        # passes; response_llm_judge still runs.
        critical_actions: set[str] = set()
    else:
        for entry in raw_critical:
            if not isinstance(entry, str):
                raise ValueError(
                    f"task.yaml at {task_yaml_path} has critical_actions "
                    f"entry {entry!r} of type {type(entry).__name__}; entries "
                    "must be namespaced tool name strings. Set "
                    "ALLOW_LEGACY_TASK_YAML=1 to evaluate older dict-shape "
                    "tasks with a degenerate commit_check."
                )
        critical_actions = set(raw_critical)

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
        task_yaml=task_yaml,
        metadata=metadata,
        critical_actions=critical_actions,
        action_type=str(metadata.get("action_type") or ""),
    )
