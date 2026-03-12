from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


SUMMARY_COLUMNS = [
    "provider",
    "model",
    "category",
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
    return sorted(Path(results_root).glob("*/*/*/*/*/*/eval.json"))


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
                "run_id": payload.get("run_id"),
                "artifact_dir": payload.get("artifact_dir"),
                "task_dir": payload.get("task_dir"),
                "generated_at": payload.get("generated_at"),
                "run_error": payload.get("run_error"),
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
                    "run_id": row.get("run_id"),
                    "generated_at": row.get("generated_at"),
                    "metric_name": metric_name,
                    "pass": normalized_pass,
                    "metric_payload": metric_payload,
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
    pair_df = usable.pivot_table(
        index=["provider", "model", "category", "metric_name", "task_id"],
        columns="task_type",
        values="pass",
        aggfunc="last",
    ).reset_index()

    summary_records: list[dict[str, Any]] = []
    for (provider, model, category, metric_name), group in pair_df.groupby(
        ["provider", "model", "category", "metric_name"],
        dropna=False,
    ):
        act_series = group["act"] if "act" in group else pd.Series(dtype="boolean")
        abstain_series = group["abstain"] if "abstain" in group else pd.Series(dtype="boolean")

        act_non_null = act_series.dropna()
        abstain_non_null = abstain_series.dropna()
        complete_pairs = group.dropna(subset=["act", "abstain"]) if {"act", "abstain"}.issubset(group.columns) else group.iloc[0:0]
        conditioned = complete_pairs[complete_pairs["act"]]

        summary_records.append(
            {
                "provider": provider,
                "model": model,
                "category": category,
                "metric_name": metric_name,
                "should_act_accuracy": act_non_null.mean() if not act_non_null.empty else pd.NA,
                "should_abstain_accuracy": abstain_non_null.mean() if not abstain_non_null.empty else pd.NA,
                "paired_accuracy": (
                    (complete_pairs["act"] & complete_pairs["abstain"]).mean() if not complete_pairs.empty else pd.NA
                ),
                "car": conditioned["abstain"].mean() if not conditioned.empty else pd.NA,
                "num_act": int(len(act_non_null)),
                "num_abstain": int(len(abstain_non_null)),
                "num_pairs": int(len(complete_pairs)),
                "num_car_pairs": int(len(conditioned)),
            }
        )

    summary_df = pd.DataFrame(summary_records)
    if not summary_df.empty:
        summary_df["model_label"] = summary_df["provider"] + "/" + summary_df["model"]
    return summary_df.sort_values(
        by=["category", "provider", "model", "metric_name"],
        kind="stable",
    ).reset_index(drop=True)
