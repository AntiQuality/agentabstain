from __future__ import annotations

import argparse
import json
from pathlib import Path

from abstention_factory.src.utils.file_io import read_yaml, utc_now_iso, write_json
from eval.config import EvaluationConfig
from eval.discovery import discover_run_result_paths
from eval.evaluators import (
    CommitCheckEvaluator,
    ResponseLLMJudgeEvaluator,
)
from eval.loader import load_evaluation_bundle
from tqdm import tqdm


DEFAULT_JUDGE_CONFIG_PATH = Path("eval/configs/default.yaml")


def _write_eval_if_changed(output_path: Path, new_payload: dict) -> bool:
    """Write eval.json only when the payload differs from the existing
    one (ignoring `generated_at`). Re-evaluation is idempotent for runs
    whose underlying state hasn't changed -- judge results are cached,
    commit_check is deterministic on a fixed bundle -- so rewriting the
    file just to bump generated_at churns mtimes and obscures which
    rows actually got recomputed. Returns True if a write occurred.
    """
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
        except Exception:
            existing = None
        if isinstance(existing, dict):
            existing_no_ts = {k: v for k, v in existing.items() if k != "generated_at"}
            new_no_ts = {k: v for k, v in new_payload.items() if k != "generated_at"}
            if existing_no_ts == new_no_ts:
                return False
    write_json(output_path, new_payload)
    return True

# Canonical vocabulary for the `task_type` field. Every task pair emits
# one act and one abstain task (see WriteTaskPairNode in
# abstention_factory/src/nodes/write_task_pair.py). Analysis code
# (eval/statistics/analysis.py) pivots specifically on these two values.
# Extending this set requires updating:
#   - this runner's load-error fallback
#   - eval/statistics/analysis.py pivot + summary code
#   - any downstream notebooks that compare act-vs-abstain behavior
# Unknown task_types surfaced at load time are flagged via a
# `task_type_unknown` metric rather than silently dropped, so schema
# drift is visible in the aggregate.
KNOWN_TASK_TYPES = frozenset({"act", "abstain"})


# Classification of provider-/runtime-level run errors. These are runs
# where the agent never produced a usable final answer (provider gate
# refused, transport failed, max turns hit, etc.). Such runs are NOT
# sent to the LLM judge and are NOT counted in the strict/loose/judge
# denominators -- they're surfaced as a separate `run_error_kind` so
# `eval/statistics/visualize_metrics.ipynb` can break them down by
# kind and the operator can decide which to re-run.
#
# Distinguishing kinds matters: `policy_refusal` is the Anthropic SDK
# usage-policy gate (Claude Code refused before the model saw the
# task), while `api_error` is transient transport that should re-run
# cleanly. `max_turns_exceeded` is the agent loop hitting its budget
# (model did try but couldn't finish in time) -- arguably a model
# failure but excluding it keeps the metrics about decisions the model
# actually made on its terms.
RUN_ERROR_KINDS = frozenset(
    {"policy_refusal", "api_error", "max_turns_exceeded", "runtime_error", "empty_output"}
)


def _classify_run_error(run_result):
    """Classify a run_result.json into a (kind, message) pair, or
    (None, None) if the run looks usable. See RUN_ERROR_KINDS for the
    vocabulary. Reads only top-level fields and provider_metadata so
    it works on both the full bundle and load-error partial dicts.
    """
    if not isinstance(run_result, dict):
        return (None, None)

    final_output = (run_result.get("final_output") or "").strip()
    pm = run_result.get("provider_metadata") or {}
    rm = pm.get("result_message") if isinstance(pm, dict) else None
    is_error_flag = rm.get("is_error") if isinstance(rm, dict) else None

    # claudesdk surfaces both gate refusals and transport failures via
    # is_error=True. Disambiguate by message: the gate response is a
    # known fixed string ("violate our Usage Policy"); everything else
    # under is_error is treated as transient API error.
    if is_error_flag is True:
        if "Usage Policy" in final_output or "violate our Usage Policy" in final_output:
            return ("policy_refusal", final_output[:200])
        return ("api_error", final_output[:200] or "is_error=True with empty message")

    top_error = run_result.get("error")
    if top_error:
        msg = str(top_error)
        low = msg.lower()
        if "max turn" in low or "max_turns" in low:
            return ("max_turns_exceeded", msg[:200])
        if (
            "rate limit" in low
            or "overloaded" in low
            or "timeout" in low
            or "timed out" in low
            or "connection" in low
            or "api error" in low
        ):
            return ("api_error", msg[:200])
        return ("runtime_error", msg[:200])

    # No flagged error but no final answer either -- the judge has
    # nothing to score, so treat as a run error rather than feeding
    # it through and getting a vacuous "missing final response" fail.
    if not final_output:
        return ("empty_output", "")

    return (None, None)


def _build_skipped_metrics(evaluators, kind: str) -> dict:
    """Build a metrics dict where every evaluator is marked as
    skipped due to a run-level error. `pass=None` so analysis layer
    drops it from denominators (NA path), and `skipped=<kind>` plus
    `run_error_kind=<kind>` are mirrored on each row so downstream
    code can attribute the skip without joining back to the top-level
    field."""
    return {
        evaluator.name: {
            "pass": None,
            "skipped": kind,
            "run_error_kind": kind,
        }
        for evaluator in evaluators
    }


def _peek_action_type(run_result_partial: dict) -> str:
    """Best-effort read of `action_type` from metadata.yaml when the
    bundle couldn't load or drifts. Preserves per-action-type denominator
    visibility on synthetic failure rows so the headline analysis splits
    don't silently drop the broken runs. Returns "" when unavailable.
    """
    task_dir = run_result_partial.get("task_dir")
    if not isinstance(task_dir, str) or not task_dir:
        return ""
    metadata_path = Path(task_dir).parent / "metadata.yaml"
    if not metadata_path.exists():
        return ""
    try:
        metadata = read_yaml(metadata_path)
    except Exception:
        return ""
    value = metadata.get("action_type") if isinstance(metadata, dict) else None
    return str(value) if isinstance(value, str) else ""


def _canonicalize_task_type(value):
    """Normalize common task_type drift (casing, whitespace) and match
    against KNOWN_TASK_TYPES. Returns the canonical form or None."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in KNOWN_TASK_TYPES:
        return normalized
    return None


def _resolve_eval_identity(run_result_path, run_result_partial):
    """Derive the (category, task_id, task_type) identity tuple that
    will key this eval row in `select_latest_runs()`.

    `is_canonical_slot = True` means: this artifact can be treated as
    part of a real task bucket and standard-metric failure rows are
    safe to emit under the resolved identity. It is only returned True
    when THREE sources agree that this is a real task slot:

      1. The PATH directory structure is canonical (task_type slot is
         "act"/"abstain" after strip+lower normalization).
      2. Either run_result.json has no `task_dir`, or its `task_dir`'s
         last three components (<category>/<task_id>/<task_type>) agree
         with the path's canonical identity. A misplaced
         run_result.json (copied from task A into task B's canonical
         results slot) would have task_dir pointing at task A while the
         path points at task B -- we refuse to emit standard failure
         rows in that case because they'd poison task B's bucket with
         metrics computed from task A's bundle.

    When either check fails, is_canonical_slot is False and the caller
    should emit only a diagnostic metric (no standard-metric failure
    rows) so the artifact stays visible without corrupting headline
    denominators.

    The canonical-path-only rule matches the write_task_pair convention
    (abstention_factory/src/nodes/write_task_pair.py): tasks live under
    tasks/<category>/<pair_id>/{act,abstain}/task.yaml so task_dir's
    parts[-3:] are (category, pair_id, task_type).

    Returns (category, task_id, task_type, is_canonical_slot).
    """
    path_parts = Path(run_result_path).parts
    path_category = path_parts[-5] if len(path_parts) >= 5 else None
    path_task_id = path_parts[-4] if len(path_parts) >= 4 else None
    path_task_type_raw = path_parts[-3] if len(path_parts) >= 3 else None
    path_task_type = _canonicalize_task_type(path_task_type_raw)

    if path_task_type is None:
        # Non-canonical path slot. Never promote to a canonical bucket
        # based on JSON alone -- a stray artifact that happens to have
        # task_type: "act" in its JSON must not manufacture failure
        # rows for a task that doesn't exist.
        json_category = run_result_partial.get("category") or path_category
        json_task_id = run_result_partial.get("task_id") or path_task_id
        raw_task_type = run_result_partial.get("task_type") or path_task_type_raw
        return (json_category, json_task_id, raw_task_type, False)

    # Path slot is canonical. Cross-check task_dir against path to
    # detect misplacement (run_result.json from task A copied into task
    # B's results slot).
    task_dir = run_result_partial.get("task_dir")
    if isinstance(task_dir, str) and task_dir:
        td_parts = Path(task_dir).parts
        td_category = td_parts[-3] if len(td_parts) >= 3 else None
        td_task_id = td_parts[-2] if len(td_parts) >= 2 else None
        td_task_type_raw = td_parts[-1] if len(td_parts) >= 1 else None
        td_task_type = _canonicalize_task_type(td_task_type_raw)
        path_identity = (path_category, path_task_id, path_task_type)
        td_identity = (td_category, td_task_id, td_task_type)
        if td_identity != path_identity:
            # task_dir says this is a different task than the path
            # slot claims. Cross-task contamination: emit diagnostic
            # only, don't write failure rows into the path bucket.
            return (path_category, path_task_id, path_task_type, False)

    # Path slot canonical AND task_dir consistent (or absent). Safe to
    # trust the path identity: any producer drift on JSON identity
    # fields is contained to the path-derived bucket.
    return (path_category, path_task_id, path_task_type, True)


def build_evaluators(config: EvaluationConfig):
    return [
        CommitCheckEvaluator(),
        ResponseLLMJudgeEvaluator(config.judge_models),
    ]


def evaluate_provider_model(
    *,
    provider: str,
    model: str,
    results_root: str | Path,
    judge_config_path: str | Path,
    override_judge: bool = False,
) -> list[Path]:
    config = EvaluationConfig.from_yaml(judge_config_path)
    evaluators = build_evaluators(config)
    written_paths: list[Path] = []
    run_result_paths = discover_run_result_paths(results_root, provider, model)
    progress = tqdm(
        run_result_paths,
        desc=f"evaluating {provider}/{model}",
        unit="run",
        leave=True,
    )

    for run_result_path in progress:
        try:
            bundle = load_evaluation_bundle(run_result_path, provider=provider, model=model)
        except Exception as exc:
            # A single malformed bundle (missing paired-act task.yaml,
            # broken node ref in critical_actions, corrupt YAML, etc.)
            # should not abort the entire provider/model sweep. Emit a
            # run-scoped error payload next to the run_result.json so
            # downstream analysis can see exactly which runs failed to
            # load, and keep going.
            #
            # Critical: we emit failing rows for EVERY standard evaluator
            # (not just a synthetic `bundle_load` metric). Without this,
            # select_latest_runs would replace an older successful eval
            # with this load-error row, dropping the task from the
            # strict/loose/response_llm_judge denominators entirely --
            # shrinking the reported sample and inflating pass rates.
            # With pass=False rows for each standard metric, the task
            # stays in every denominator as a failure.
            output_path = Path(run_result_path).parent / "eval.json"
            path = Path(run_result_path)
            run_result_partial: dict = {}
            try:
                run_result_partial = json.loads(path.read_text())
            except Exception:
                pass
            # Resolve the eval identity tuple. When the results-tree
            # path slot is canonical (after normalization), trust the
            # path for category/task_id/task_type so latest-run dedupe
            # supersedes any stale passing row in that slot even under
            # simultaneous producer drift on multiple identity fields.
            # See _resolve_eval_identity for the full rationale.
            category, task_id, task_type, is_canonical_slot = (
                _resolve_eval_identity(run_result_path, run_result_partial)
            )
            # Check path slot canonicity independently: controls
            # whether we can emit standard-metric failure rows safely.
            # task_dir disagreement (which flips is_canonical_slot=False)
            # shouldn't suppress standard rows -- the rows are
            # synthetic `pass=False` markers, not computed metrics, so
            # emitting them under path-derived identity correctly says
            # "this canonical slot has a broken artifact" and keeps
            # the task in the denominator.
            path_task_type_raw = (
                Path(run_result_path).parts[-3]
                if len(Path(run_result_path).parts) >= 3
                else None
            )
            path_slot_canonical = _canonicalize_task_type(path_task_type_raw) is not None
            error_str = f"{type(exc).__name__}: {exc}"
            failure_row = {"pass": False, "error": error_str}
            # Only suppress standard-metric rows when the PATH slot
            # itself is non-canonical (operator-dropped auxiliary
            # artifact, non-task directory). For canonical slots we
            # always emit failure rows to preserve denominators.
            if path_slot_canonical:
                load_error_metrics: dict[str, dict] = {
                    evaluator.name: dict(failure_row) for evaluator in evaluators
                }
            else:
                # Don't create act/abstain-shaped rows for unknown
                # task_types, but DO emit a dedicated `task_type_unknown`
                # metric so schema drift surfaces loudly in aggregate
                # reporting (analysis.py will show these as a distinct
                # metric_name with pass=False). Silent dropping would
                # look like a clean sweep.
                load_error_metrics = {
                    "task_type_unknown": {
                        "pass": False,
                        "error": (
                            f"task_type={task_type!r} is not in canonical "
                            f"vocabulary {sorted(KNOWN_TASK_TYPES)!r}. Either "
                            f"the results tree has a non-standard artifact "
                            f"at this path or the task_type vocabulary "
                            f"needs extending (see eval/runner.py:"
                            f"KNOWN_TASK_TYPES)."
                        ),
                    },
                }
            # Diagnostic metric always present so operators can spot-check
            # which runs failed to load without scanning load_error strings.
            load_error_metrics["bundle_load"] = dict(failure_row)
            _write_eval_if_changed(
                output_path,
                {
                    "provider": provider,
                    "model": model,
                    "category": category,
                    "task_id": task_id,
                    "task_type": task_type,
                    "expected_behavior": task_type,
                    "action_type": _peek_action_type(run_result_partial),
                    "run_id": path.parent.name,
                    "artifact_dir": str(path.parent),
                    "task_dir": run_result_partial.get("task_dir"),
                    "run_result_path": str(run_result_path),
                    "generated_at": utc_now_iso(),
                    "run_error": run_result_partial.get("error"),
                    "load_error": error_str,
                    "metrics": load_error_metrics,
                },
            )
            written_paths.append(output_path)
            continue
        progress.set_postfix_str(
            f"{bundle.category}/{bundle.task_id}/{bundle.task_type}/{bundle.run_id}",
            refresh=False,
        )
        # Successful-load identity handling:
        #   (1) Happy path -- path canonical AND task_dir agrees:
        #       bundle came from the task the slot claims. Run
        #       evaluators normally. If bundle.task_type/category/
        #       task_id have benign casing drift, the written eval.json
        #       still keys on the canonicalized (resolved) identity so
        #       dedupe is stable.
        #   (2) Path canonical BUT task_dir disagrees -- bundle came
        #       from a DIFFERENT task than the slot claims. Running
        #       evaluators here would compute wrong-task metrics under
        #       the slot's identity. Emit synthetic pass=False rows
        #       instead, so the slot stays in denominators as a
        #       failure without contaminating numerators.
        #   (3) Path non-canonical: no task slot to key rows under.
        #       Emit diagnostic only.
        canonical_bundle_task_type = _canonicalize_task_type(bundle.task_type)
        bundle_identity = _resolve_eval_identity(
            run_result_path,
            {
                "category": bundle.category,
                "task_id": bundle.task_id,
                "task_type": bundle.task_type,
                "task_dir": str(bundle.task_dir),
            },
        )
        bundle_category, bundle_task_id, resolved_task_type, is_canonical_slot = bundle_identity
        path_task_type_raw = (
            Path(run_result_path).parts[-3]
            if len(Path(run_result_path).parts) >= 3
            else None
        )
        path_slot_canonical = _canonicalize_task_type(path_task_type_raw) is not None
        # Drift branch fires on: (a) misplacement (path canonical +
        # task_dir disagrees), (b) non-canonical path slot, OR (c)
        # bundle.task_type that can't be canonicalized at all. Case (c)
        # matters because load_evaluation_bundle only normalizes
        # strip+lower hits against the canonical set; a wholly unknown
        # value like "ACTION" or "validation" slips through with
        # bundle.task_type preserved as the raw string. Evaluators key
        # on `task_type == "act"` exactly and would fall back to
        # abstain semantics, flipping pass/fail outcomes on an
        # otherwise-loadable bundle. Route these through drift so
        # synthetic failure rows preserve denominator visibility.
        misplaced_in_canonical = path_slot_canonical and not is_canonical_slot
        non_canonical_slot = not path_slot_canonical
        bundle_task_type_unknown = canonical_bundle_task_type is None
        if misplaced_in_canonical or non_canonical_slot or bundle_task_type_unknown:
            output_path = bundle.artifact_dir / "eval.json"
            # Read RAW task_type from run_result for diagnostic clarity.
            # bundle.task_type is canonicalized at load time so
            # evaluators see a known value; the raw form is only
            # needed here to help operators track down producer drift.
            raw_bundle_task_type = bundle.run_result.get("task_type")
            task_type_unknown_row = {
                "pass": False,
                "error": (
                    f"Identity drift detected: run_result reports "
                    f"category={bundle.category!r}, task_id={bundle.task_id!r}, "
                    f"task_type={raw_bundle_task_type!r} (canonicalized to "
                    f"{bundle.task_type!r}), but resolved identity is "
                    f"({bundle_category!r}, {bundle_task_id!r}, "
                    f"{resolved_task_type!r}). Producer drift (casing, "
                    f"whitespace, renamed fields) or schema extension. "
                    f"See eval/runner.py:KNOWN_TASK_TYPES."
                ),
            }
            # Gate standard-metric failure rows on PATH canonicity
            # only. task_dir misplacement (which flips
            # is_canonical_slot=False) should still emit pass=False
            # rows under the path-derived identity: the rows are
            # synthetic markers saying "this slot has a broken
            # artifact", not metrics computed from the (wrong) bundle.
            # Dropping them for misplacement would shrink the
            # strict/loose/judge denominators and inflate headline
            # pass rates, defeating the R8-F2 denominator-preservation
            # goal. Diagnostic-only is reserved for the case where
            # there is no canonical task slot to key rows under.
            if path_slot_canonical:
                failure_row = {
                    "pass": False,
                    "error": (
                        f"bundle identity drift ({bundle.category!r}, "
                        f"{bundle.task_id!r}, {bundle.task_type!r}) -> "
                        f"canonical ({bundle_category!r}, "
                        f"{bundle_task_id!r}, {resolved_task_type!r}); "
                        f"is_canonical_slot={is_canonical_slot} "
                        f"(task_dir={'agrees' if is_canonical_slot else 'disagrees'} with path)"
                    ),
                }
                metrics = {
                    evaluator.name: dict(failure_row) for evaluator in evaluators
                }
                metrics["task_type_unknown"] = task_type_unknown_row
            else:
                metrics = {"task_type_unknown": task_type_unknown_row}
            _write_eval_if_changed(
                output_path,
                {
                    "provider": provider,
                    "model": model,
                    "category": bundle_category,
                    "task_id": bundle_task_id,
                    "task_type": resolved_task_type,
                    "expected_behavior": resolved_task_type,
                    "action_type": bundle.action_type,
                    "run_id": bundle.run_id,
                    "artifact_dir": str(bundle.artifact_dir),
                    "task_dir": str(bundle.task_dir),
                    "run_result_path": str(run_result_path),
                    "generated_at": utc_now_iso(),
                    "run_error": bundle.run_result.get("error"),
                    "metrics": metrics,
                },
            )
            written_paths.append(output_path)
            continue
        output_path = bundle.artifact_dir / "eval.json"
        existing_payload = None
        if output_path.exists():
            existing_payload = json.loads(output_path.read_text())

        # Provider-/runtime-level run errors: the agent never produced
        # a scorable answer (gate refusal, transport, max-turns, etc.).
        # Skip the LLM judge entirely -- there's nothing to judge --
        # and emit pass=None metrics so the analysis layer drops them
        # from denominators rather than counting them as failures.
        # Top-level run_error_kind/run_error_message are surfaced for
        # the notebook's error-breakdown grid.
        run_error_kind, run_error_message = _classify_run_error(bundle.run_result)
        if run_error_kind is not None:
            eval_payload = {
                "provider": provider,
                "model": model,
                "category": bundle_category,
                "task_id": bundle_task_id,
                "task_type": resolved_task_type,
                "expected_behavior": resolved_task_type,
                "action_type": bundle.action_type,
                "run_id": bundle.run_id,
                "artifact_dir": str(bundle.artifact_dir),
                "task_dir": str(bundle.task_dir),
                "generated_at": utc_now_iso(),
                "run_error": bundle.run_result.get("error"),
                "run_error_kind": run_error_kind,
                "run_error_message": run_error_message,
                "metrics": _build_skipped_metrics(evaluators, run_error_kind),
            }
            _write_eval_if_changed(output_path, eval_payload)
            written_paths.append(output_path)
            continue

        metrics: dict[str, dict] = {}
        for evaluator in evaluators:
            if (
                evaluator.name == "response_llm_judge"
                and not override_judge
                and isinstance(existing_payload, dict)
                and isinstance(existing_payload.get("metrics"), dict)
                and evaluator.name in existing_payload["metrics"]
            ):
                metrics[evaluator.name] = existing_payload["metrics"][evaluator.name]
                continue
            try:
                metrics[evaluator.name] = evaluator.evaluate(bundle)
            except Exception as exc:
                metrics[evaluator.name] = {
                    "pass": None,
                    "error": str(exc),
                }

        # Write eval.json with canonicalized identity so benign
        # metadata drift (e.g. bundle.task_type="Act", path="/act/")
        # still dedupes into the canonical slot under select_latest_runs.
        # bundle_category/bundle_task_id/resolved_task_type come from
        # _resolve_eval_identity -- path-derived when the slot is
        # canonical (which is the case here: we only reach this
        # happy-path write when misplaced_in_canonical was False and
        # non_canonical_slot was False).
        eval_payload = {
            "provider": provider,
            "model": model,
            "category": bundle_category,
            "task_id": bundle_task_id,
            "task_type": resolved_task_type,
            "expected_behavior": resolved_task_type,
            "action_type": bundle.action_type,
            "run_id": bundle.run_id,
            "artifact_dir": str(bundle.artifact_dir),
            "task_dir": str(bundle.task_dir),
            "generated_at": utc_now_iso(),
            "run_error": bundle.run_result.get("error"),
            "run_error_kind": None,
            "run_error_message": None,
            "metrics": metrics,
        }
        _write_eval_if_changed(output_path, eval_payload)
        written_paths.append(output_path)

    return written_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all runs for one provider/model pair.")
    parser.add_argument("--provider", required=True, help="Runtime provider, e.g. openaisdk")
    parser.add_argument("--model", required=True, help="Model name under results/{provider}/{model}")
    parser.add_argument("--results-root", default="results", help="Root directory containing runtime results")
    parser.add_argument(
        "--judge-config",
        default=str(DEFAULT_JUDGE_CONFIG_PATH),
        help="YAML config defining judge_models",
    )
    parser.add_argument(
        "--override-judge",
        action="store_true",
        help="Re-run response_llm_judge even if eval.json already contains judge results",
    )
    args = parser.parse_args()

    written_paths = evaluate_provider_model(
        provider=args.provider,
        model=args.model,
        results_root=args.results_root,
        judge_config_path=args.judge_config,
        override_judge=args.override_judge,
    )
    print(
        json.dumps(
            {
                "provider": args.provider,
                "model": args.model,
                "evaluated_runs": len(written_paths),
                "eval_paths": [str(path) for path in written_paths],
            },
            indent=4,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
