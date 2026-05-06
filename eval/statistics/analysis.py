from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


SUMMARY_COLUMNS = [
    "provider",
    "model",
    "category",
    "action_type",
    "metric_name",
    "should_act_accuracy",
    "should_abstain_accuracy",
    "paired_accuracy",
    "car",
    "num_act",
    "num_abstain",
    "num_pairs",
    "num_car_pairs",
]


def discover_eval_paths(results_root: str | Path = "results") -> list[Path]:
    root = Path(results_root)
    # Three layouts coexist; the trailing 4 components are always
    # `<category>/<task>/<type>/<run>/eval.json`. The provider+model
    # prefix varies by runtime:
    #   provider/model/<...>/eval.json                     (6 levels) — claudesdk, openaisdk, googleadk
    #   provider/runtime/model/<...>/eval.json             (7 levels) — openclaw/amazon-bedrock/<model>
    #   provider/runtime/vendor/model/<...>/eval.json      (8 levels) — openclaw/openrouter/deepseek/deepseek-v4-pro
    # The 8-level case appears when the canonical model id itself
    # contains a `/` (openrouter publishes `<vendor>/<model>` ids), so
    # the artifact path gains one extra component vs the bedrock case.
    return sorted({
        *root.glob("*/*/*/*/*/*/eval.json"),
        *root.glob("*/*/*/*/*/*/*/eval.json"),
        *root.glob("*/*/*/*/*/*/*/*/eval.json"),
    })


def load_eval_frame(results_root: str | Path = "results") -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in discover_eval_paths(results_root):
        payload = json.loads(path.read_text())
        records.append(
            {
                "provider": payload.get("provider"),
                "model": payload.get("model"),
                "category": payload.get("category"),
                "task_id": payload.get("task_id"),
                "task_type": payload.get("task_type"),
                "expected_behavior": payload.get("expected_behavior"),
                "action_type": payload.get("action_type"),
                "run_id": payload.get("run_id"),
                "artifact_dir": payload.get("artifact_dir"),
                "task_dir": payload.get("task_dir"),
                "generated_at": payload.get("generated_at"),
                "run_error": payload.get("run_error"),
                # Provider-/runtime-level error classification produced
                # by eval/runner.py:_classify_run_error. Rows with a
                # non-None kind have pass=None metrics (skipped LLM
                # judge) and are excluded from accuracy denominators
                # via dropna(subset=["pass"]) downstream. Surfaced on
                # the frame so the notebook can build a breakdown grid
                # of how many runs are policy_refusal vs api_error vs
                # max_turns_exceeded -- the "should re-run" signal.
                "run_error_kind": payload.get("run_error_kind"),
                "run_error_message": payload.get("run_error_message"),
                "metrics": payload.get("metrics", {}),
                "eval_path": str(path),
            }
        )
    return pd.DataFrame(records)


def select_latest_runs(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return eval_df.copy()

    latest = eval_df.copy()
    latest["generated_at_ts"] = pd.to_datetime(latest["generated_at"], errors="coerce", utc=True)
    latest = latest.sort_values(
        by=["provider", "model", "category", "task_id", "task_type", "generated_at_ts", "run_id"],
        kind="stable",
    )
    latest = latest.drop_duplicates(
        subset=["provider", "model", "category", "task_id", "task_type"],
        keep="last",
    )
    return latest.reset_index(drop=True)


def build_metric_run_frame(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame(
            columns=[
                "provider",
                "model",
                "category",
                "task_id",
                "task_type",
                "expected_behavior",
                "action_type",
                "run_id",
                "generated_at",
                "metric_name",
                "pass",
                "eval_path",
            ]
        )

    records: list[dict[str, Any]] = []
    for row in eval_df.to_dict(orient="records"):
        metrics = row.get("metrics") or {}
        run_error_kind = row.get("run_error_kind")
        for metric_name, metric_payload in metrics.items():
            pass_value = metric_payload.get("pass") if isinstance(metric_payload, dict) else None
            normalized_pass = pass_value if isinstance(pass_value, bool) else pd.NA
            records.append(
                {
                    "provider": row.get("provider"),
                    "model": row.get("model"),
                    "category": row.get("category"),
                    "task_id": row.get("task_id"),
                    "task_type": row.get("task_type"),
                    "expected_behavior": row.get("expected_behavior"),
                    "action_type": row.get("action_type"),
                    "run_id": row.get("run_id"),
                    "generated_at": row.get("generated_at"),
                    "metric_name": metric_name,
                    "pass": normalized_pass,
                    "run_error_kind": run_error_kind,
                    "metric_payload": metric_payload,
                    "eval_path": row.get("eval_path"),
                }
            )

        # Synthetic `combined` metric — benchmark-semantic pass:
        #   informational: response_llm_judge only (commit_check is
        #     trivially true when critical_actions is empty)
        #   operational:   commit_check AND response_llm_judge
        # Any missing/non-bool dependency -> NA so compute_summary_metrics
        # skips this row instead of counting it as a failure.
        action_type = row.get("action_type") or ""
        commit_pass = (
            metrics.get("commit_check", {}).get("pass")
            if isinstance(metrics.get("commit_check"), dict)
            else None
        )
        judge_pass = (
            metrics.get("response_llm_judge", {}).get("pass")
            if isinstance(metrics.get("response_llm_judge"), dict)
            else None
        )
        if action_type == "informational":
            combined_pass = judge_pass if isinstance(judge_pass, bool) else pd.NA
        else:
            if isinstance(commit_pass, bool) and isinstance(judge_pass, bool):
                combined_pass = commit_pass and judge_pass
            else:
                combined_pass = pd.NA
        records.append(
            {
                "provider": row.get("provider"),
                "model": row.get("model"),
                "category": row.get("category"),
                "task_id": row.get("task_id"),
                "task_type": row.get("task_type"),
                "expected_behavior": row.get("expected_behavior"),
                "action_type": action_type,
                "run_id": row.get("run_id"),
                "generated_at": row.get("generated_at"),
                "metric_name": "combined",
                "pass": combined_pass,
                "run_error_kind": run_error_kind,
                "metric_payload": {
                    "commit_check_pass": commit_pass,
                    "response_llm_judge_pass": judge_pass,
                    "action_type": action_type,
                },
                "eval_path": row.get("eval_path"),
            }
        )
    metric_df = pd.DataFrame(records)
    if not metric_df.empty:
        metric_df["model_label"] = metric_df["provider"] + "/" + metric_df["model"]
    return metric_df


def compute_summary_metrics(metric_df: pd.DataFrame) -> pd.DataFrame:
    if metric_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    usable = metric_df.dropna(subset=["pass"]).copy()
    if usable.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    usable["pass"] = usable["pass"].astype(bool)
    # Pivot includes action_type so the headline summary table splits
    # informational vs operational tasks. commit_check on informational
    # tasks is trivially true (empty critical set), so the split is
    # essential for honest reporting — without it, informational rows
    # inflate commit_check pass rates.
    if "action_type" not in usable.columns:
        usable["action_type"] = ""
    usable["action_type"] = usable["action_type"].fillna("")
    pair_df = usable.pivot_table(
        index=["provider", "model", "category", "action_type", "metric_name", "task_id"],
        columns="task_type",
        values="pass",
        aggfunc="last",
    ).reset_index()

    summary_records: list[dict[str, Any]] = []
    for (provider, model, category, action_type, metric_name), group in pair_df.groupby(
        ["provider", "model", "category", "action_type", "metric_name"],
        dropna=False,
    ):
        act_series = group["act"] if "act" in group else pd.Series(dtype="boolean")
        abstain_series = group["abstain"] if "abstain" in group else pd.Series(dtype="boolean")

        act_non_null = act_series.dropna()
        abstain_non_null = abstain_series.dropna()
        complete_pairs = group.dropna(subset=["act", "abstain"]) if {"act", "abstain"}.issubset(group.columns) else group.iloc[0:0]
        conditioned = complete_pairs[complete_pairs["act"]]

        # CAR semantics: P(abstain correct | act correct on the paired task).
        # If complete_pairs is empty -> NA (no data at all to condition on).
        # If complete_pairs is non-empty but conditioned is empty (should-act
        # accuracy is 0 on this slice) -> 0.0, so plots/aggregations stay
        # dense. A model that can't act correctly on any pair has nothing
        # to be conditionally abstaining about -> CAR=0 is the defensible
        # readout, not a hole.
        if complete_pairs.empty:
            car_value: Any = pd.NA
        elif conditioned.empty:
            car_value = 0.0
        else:
            car_value = conditioned["abstain"].mean()

        summary_records.append(
            {
                "provider": provider,
                "model": model,
                "category": category,
                "action_type": action_type,
                "metric_name": metric_name,
                # All four metrics use the same denominator: complete
                # pairs (both act and abstain runs non-null). This
                # ensures Act Acc, Abstain Acc, Paired Acc, and CAR are
                # computed over identical task sets.
                "should_act_accuracy": complete_pairs["act"].mean() if not complete_pairs.empty else pd.NA,
                "should_abstain_accuracy": complete_pairs["abstain"].mean() if not complete_pairs.empty else pd.NA,
                "paired_accuracy": (
                    (complete_pairs["act"] & complete_pairs["abstain"]).mean() if not complete_pairs.empty else pd.NA
                ),
                "car": car_value,
                "num_act": int(len(act_non_null)),
                "num_abstain": int(len(abstain_non_null)),
                "num_pairs": int(len(complete_pairs)),
                "num_car_pairs": int(len(conditioned)),
            }
        )

    summary_df = pd.DataFrame(summary_records)
    if not summary_df.empty:
        summary_df["model_label"] = summary_df["provider"] + "/" + summary_df["model"]
        # Force the four accuracy columns to float dtype. With no usable
        # rows in a slice (e.g. every act run in a category is excluded
        # as a run-level error), the per-cell value is pd.NA which makes
        # the entire column `object` dtype -- seaborn's heatmap then
        # fails with "Image data of dtype object cannot be converted to
        # float". `pd.to_numeric(..., errors="coerce")` collapses pd.NA
        # to np.nan and keeps the column float64.
        for col in ("should_act_accuracy", "should_abstain_accuracy", "paired_accuracy", "car"):
            if col in summary_df.columns:
                summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")
    return summary_df.sort_values(
        by=["category", "action_type", "provider", "model", "metric_name"],
        kind="stable",
    ).reset_index(drop=True)
