"""Generate the per-family scaling line plot.

Mirrors ``analysis/analyze.py::a7_scaling`` but restyled for the paper:
Lato font, the lifted ``environment`` 4-metric palette shared with
``figure_ranking_bar.py``, ``Average`` y-axis label, no chart title.
The four panels (Claude / GPT-5 / Gemini / Open-weight) are arranged
in one row at paper-friendly dimensions so font sizes don't shrink to
illegibility when included as ``\\includegraphics[width=\\linewidth]``.

Usage:
    python eval/statistics/figure_scaling_lineplot.py

Output:
    eval/statistics/figures/scaling_lineplot.pdf
    eval/statistics/figures/scaling_lineplot.png
"""
from __future__ import annotations

import math
import sys
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.statistics.analysis import (
    build_metric_run_frame,
    compute_summary_metrics,
    load_eval_frame,
    select_latest_runs,
)
from eval.statistics.figure_leaderboard_bar import MODEL_CONFIG, canon
from eval.statistics.figure_ranking_bar import METRIC_ORDER, PALETTES

plt.rcParams["font.family"] = "Lato"

OUTPUT_DIR = SCRIPT_DIR / "figures"

# Paper-friendly dimensions: 4 panels in one row, total width close to
# \linewidth so 11pt source text renders readably after scaling.
figure_width = 10.0
figure_height = 2.6

METRIC_COLORS = PALETTES["environment"]
LEGEND_LABELS = {
    "Act":     "Act Accuracy",
    "Abstain": "Abstain Accuracy",
    "Paired":  "Paired Accuracy",
    "CAR":     "CAR",
}

# Family groupings. Same model lists as analysis/analyze.py:MODEL_FAMILIES,
# with short panel-titles so the chart doesn't waste horizontal space on
# parenthetical scaling-axis labels.
MODEL_FAMILIES: dict[str, list[str]] = {
    "Claude": [
        "claudesdk/claude-haiku-4-5",
        "claudesdk/claude-sonnet-4-6",
        "claudesdk/claude-opus-4-7",
    ],
    "GPT-5": [
        "openaisdk/gpt-5-2025-08-07",
        "openaisdk/gpt-5.1-2025-11-13",
        "openaisdk/gpt-5.2-2025-12-11",
        "openaisdk/gpt-5.4-2026-03-05",
        "openaisdk/gpt-5.5-2026-04-23",
    ],
    "Gemini": [
        "googleadk/gemini-3-flash-preview",
        "googleadk/gemini-3.1-pro-preview",
    ],
    "Open-weight": [
        "amazon-bedrock/openai.gpt-oss-120b-1:0",
        "amazon-bedrock/minimax-m2.5",
        "amazon-bedrock/deepseek.v3.2",
        "amazon-bedrock/zai.glm-5",
        "amazon-bedrock/kimi-k2.5",
        "openrouter/deepseek-v4-pro",
    ],
}


def spearman_exact(x: list[float], y: list[float]) -> tuple[float, float]:
    """Spearman ρ + two-sided exact permutation p (matches analyze.py).

    Necessary because scipy's asymptotic p is wildly wrong at very small
    n. Falls back to asymptotic once n > 8 (8! = 40320 enumerations).
    """
    n = len(x)
    if n != len(y) or n < 2:
        return float("nan"), float("nan")
    rho_obs, _ = spearmanr(x, y)
    if n > 8:
        return float(rho_obs), float(spearmanr(x, y)[1])
    count = 0
    total = math.factorial(n)
    for perm in permutations(y):
        rho_perm, _ = spearmanr(x, perm)
        if abs(rho_perm) >= abs(rho_obs) - 1e-9:
            count += 1
    return float(rho_obs), count / total


def main() -> None:
    df = load_eval_frame(REPO_ROOT / "results")
    df = select_latest_runs(df)
    df["model"] = df["model"].map(canon)
    df = df[df["task_id"].fillna("").str.startswith("preview")]
    metric_df = build_metric_run_frame(df)
    summary = compute_summary_metrics(metric_df)

    combined = summary[summary["metric_name"] == "combined"].copy()
    combined["_model_label"] = combined["provider"] + "/" + combined["model"]

    # action_type-equal-weight rollup per (model, category), then macro
    # across the 8 categories per model — same as figure_ranking_bar.py.
    per_cat = (
        combined.groupby(["_model_label", "category"])
        .agg(
            Act=("should_act_accuracy", "mean"),
            Abstain=("should_abstain_accuracy", "mean"),
            Paired=("paired_accuracy", "mean"),
            CAR=("car", "mean"),
        )
        .reset_index()
    )
    per_model = per_cat.groupby("_model_label")[list(METRIC_ORDER)].mean()

    n_panels = len(MODEL_FAMILIES)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(figure_width, figure_height), dpi=300,
        sharey=True,
    )
    if n_panels == 1:
        axes = np.array([axes])

    for ax, (family, members) in zip(axes, MODEL_FAMILIES.items()):
        present = [m for m in members if m in per_model.index]
        if not present:
            ax.set_axis_off()
            ax.set_title(f"{family}\n(no data)", fontsize=10)
            continue

        x_pos = np.arange(1, len(present) + 1)
        display_names = [MODEL_CONFIG[(p.split("/", 1)[0], p.split("/", 1)[1])][0]
                         for p in present]

        for metric in METRIC_ORDER:
            y_vals = [float(per_model.loc[m, metric]) for m in present]
            ax.plot(
                x_pos, y_vals,
                marker="o", markersize=5, linewidth=1.8,
                color=METRIC_COLORS[metric],
            )

        # Spearman ρ on Paired across the scaling axis (matches paper text
        # which cites e.g. "ρ = -0.60" for the open-weight panel).
        paired_vals = [float(per_model.loc[m, "Paired"]) for m in present]
        if len(paired_vals) >= 2:
            rho, _p = spearman_exact(list(x_pos), paired_vals)
            title = f"{family}  ($\\rho_\\mathrm{{Paired}}$={rho:+.2f}, n={len(present)})"
        else:
            title = f"{family}  (n={len(present)})"
        ax.set_title(title, fontsize=10, color="#333333")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_names, fontsize=9, rotation=30,
                           ha="right", rotation_mode="anchor", color="#444444")
        ax.set_ylim(0.25, 1.0)
        ax.set_yticks(np.arange(0.3, 1.01, 0.2))
        ax.set_xlim(0.6, len(present) + 0.4)

        ax.yaxis.grid(True, alpha=0.25, zorder=0, color="#cccccc")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#999999")
        ax.spines["left"].set_color("#999999")
        ax.tick_params(axis="y", colors="#666666", labelsize=9)
        ax.tick_params(axis="x", length=0)

    axes[0].set_ylabel("Average", fontsize=11, color="#333333")

    # Bottom-pinned legend in figure coords, like figure_ranking_bar.py.
    plt.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.32)
    legend_handles = [
        plt.Line2D([0], [0], color=METRIC_COLORS[m], marker="o",
                   markersize=5, linewidth=1.8, label=LEGEND_LABELS[m])
        for m in METRIC_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        bbox_transform=fig.transFigure,
        fontsize=10, frameon=False,
        ncol=4, handlelength=1.8, handletextpad=0.5,
        columnspacing=1.6, borderpad=0.1,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "scaling_lineplot.pdf"
    png_path = OUTPUT_DIR / "scaling_lineplot.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    for family, members in MODEL_FAMILIES.items():
        present = [m for m in members if m in per_model.index]
        if not present:
            continue
        paired_vals = [float(per_model.loc[m, "Paired"]) for m in present]
        rho, p = spearman_exact(list(range(1, len(present) + 1)), paired_vals)
        print(f"  {family:<14s}  n={len(present)}  Paired ρ={rho:+.3f}  p={p:.3f}")


if __name__ == "__main__":
    main()
