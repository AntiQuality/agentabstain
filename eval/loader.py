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
    # Canonicalize task_type at load time so downstream evaluators
    # (which key on `task_type == "act"` vs `"abstain"` exactly) don't
    # flip semantics under benign metadata drift like "Act" / "act ".
    # Evaluators read bundle.task_type; by normalizing here we keep
    # the fix in one place rather than scattering strip().lower()
    # calls across every evaluator. When the raw value cannot be
    # canonicalized, the runner's drift/diagnostic branch handles
    # emission before evaluators run.
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
    # On-disk critical_actions shape is [{node, must_yield}]. The
    # evaluators match on `tool` + `params`, which live in the
    # execution_dag node — not on the critical_actions entry itself.
    # Expand node refs in-memory so matchers see {tool, params,
    # must_yield}. For abstain tasks (which carry no execution_dag by
    # design) we look up the paired act task's DAG.
    task_yaml["critical_actions"] = _expand_critical_actions(
        task_yaml=task_yaml,
        task_dir=task_dir,
        task_type=task_type,
    )

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
        metadata=read_yaml(metadata_path),
    )


def _expand_critical_actions(
    task_yaml: dict[str, Any],
    task_dir: Path,
    task_type: str,
) -> list[dict[str, Any]]:
    """Resolve `critical_actions[i].node` against the execution_dag so
    matchers see `{tool, params, must_yield}` entries, then append any
    `additional_critical_actions` the abstain variant declared.

    Act tasks carry their own execution_dag. Abstain tasks do not --
    they inherit the paired act's DAG, loaded from the sibling
    ``<pair_id>/act/task.yaml`` written by WriteTaskPairNode.

    Missing DAG / missing node ids raise: silent-drop would recreate
    the silent matcher-never-matches pathology that prompted this fix.
    """
    critical_actions = list(task_yaml.get("critical_actions") or [])
    additional = list(task_yaml.get("additional_critical_actions") or [])

    # No node refs -> nothing to expand. Still honor additional.
    if not critical_actions and not additional:
        return []

    # Lazy DAG resolution: only reach for the paired-act DAG when at least
    # one critical_actions entry actually references a node. A future
    # abstain task whose proceed-path coverage lives entirely in
    # additional_critical_actions (e.g. empty critical_actions list) would
    # otherwise pay the cross-file lookup cost + its failure modes for
    # nothing.
    needs_dag = any(a.get("node") is not None for a in critical_actions)

    dag = task_yaml.get("execution_dag")
    if dag is None and needs_dag:
        # Canonicalize task_type before branching. Producer drift on
        # casing/whitespace (e.g. "Abstain", "abstain ") must not
        # break paired-act lookup when the bundle is otherwise
        # evaluable -- that would let metadata-only drift wipe out
        # abstain results.
        normalized_task_type = (task_type or "").strip().lower()
        if normalized_task_type == "abstain":
            # Abstain inherits act's DAG. Convention: sibling ../act/task.yaml.
            # Consistent with WriteTaskPairNode (abstention_factory/src/
            # nodes/write_task_pair.py:63-64, :86-87).
            act_task_yaml_path = task_dir.parent / "act" / "task.yaml"
            if not act_task_yaml_path.exists():
                raise FileNotFoundError(
                    f"Abstain task at {task_dir} references critical_actions by "
                    f"node but the paired act task.yaml was not found at "
                    f"{act_task_yaml_path}"
                )
            dag = read_yaml(act_task_yaml_path).get("execution_dag")
        if dag is None:
            raise ValueError(
                f"Cannot expand critical_actions for {task_dir}: no "
                f"execution_dag available (looked in task.yaml and, for "
                f"abstain, the paired act task.yaml)"
            )

    nodes_by_id = {n["id"]: n for n in ((dag or {}).get("nodes") or [])}
    expanded: list[dict[str, Any]] = []
    for action in critical_actions:
        node_id = action.get("node")
        if node_id is None:
            raise ValueError(
                f"critical_actions entry in {task_dir} is missing `node` "
                f"id: {action!r}"
            )
        node = nodes_by_id.get(node_id)
        if node is None:
            raise ValueError(
                f"critical_actions entry in {task_dir} references node "
                f"{node_id!r} which is not in execution_dag "
                f"(known: {sorted(nodes_by_id)})"
            )
        expanded.append(
            {
                "node": node_id,
                "tool": node.get("tool"),
                "params": node.get("params") or {},
                "must_yield": action.get("must_yield"),
            }
        )

    for entry in additional:
        if not isinstance(entry, dict):
            raise ValueError(
                f"additional_critical_actions entry in {task_dir} must be a "
                f"dict, got {type(entry).__name__}: {entry!r}"
            )
        tool = entry.get("tool")
        if not isinstance(tool, str) or not tool:
            raise ValueError(
                f"additional_critical_actions entry in {task_dir} must have "
                f"non-empty `tool` string: {entry!r}"
            )
        resolved: dict[str, Any] = {"tool": tool, "must_yield": None}
        # `params` deliberately preserved as missing-vs-present rather than
        # coerced to {}. Matchers interpret presence/absence:
        # absent -> tool-only match; {} -> require empty params.
        # See critical_action_strict.py for the matching rule.
        if "params" in entry:
            params = entry["params"]
            if not isinstance(params, dict):
                raise ValueError(
                    f"additional_critical_actions entry in {task_dir} has "
                    f"non-dict `params`: {entry!r}"
                )
            resolved["params"] = params
        expanded.append(resolved)

    return expanded
