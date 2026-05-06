"""Generate the act-vs-abstain dumbbell chart.

For each model, draws two dots (Act Accuracy + Abstain Accuracy) joined
by a light-gray connector, with the row label showing the model and its
Paired Accuracy. Models are sorted by Paired (desc).

Palette is the same lifted ``environment`` mapping as
``figure_ranking_bar.py``: blue = Act ("do"), coral = Abstain ("stop").
Connector is neutral gray so it doesn't compete with the metric dots —
it's pure structure (the visible gap between Act and Abstain).

Usage:
    python eval/statistics/figure_act_abstain_dumbbell.py

Output:
    eval/statistics/figures/act_abstain_dumbbell.pdf
    eval/statistics/figures/act_abstain_dumbbell.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
from eval.statistics.figure_ranking_bar import PALETTES

plt.rcParams["font.family"] = "Lato"

OUTPUT_DIR = SCRIPT_DIR / "figures"

# Half-column subfigure (sibling of centroid in `fig:both-panels`).
# Source dimensions chosen so the bumped 12pt model labels still scale
# to readable sizes after \includegraphics width=\linewidth in a
# 0.50\textwidth subfigure.
figure_width = 5.5
figure_height = 6.6

PALETTE = PALETTES["environment"]
COLOR_ACT      = PALETTE["Act"]      # denim blue
COLOR_ABSTAIN  = PALETTE["Abstain"]  # coral terracotta
COLOR_CONNECTOR = "#bbbbbb"          # neutral gray, structural only
COLOR_LABEL    = "#444444"


def main() -> None:
    df = load_eval_frame(REPO_ROOT / "results")
    df = select_latest_runs(df)
    df["model"] = df["model"].map(canon)
    df = df[df["task_id"].fillna("").str.startswith("preview")]
    metric_df = build_metric_run_frame(df)
    summary = compute_summary_metrics(metric_df)

    combined = summary[summary["metric_name"] == "combined"].copy()

    per_cat = (
        combined.groupby(["provider", "model", "category"])
        .agg(
            act=("should_act_accuracy", "mean"),
            abstain=("should_abstain_accuracy", "mean"),
            paired=("paired_accuracy", "mean"),
        )
        .reset_index()
    )
    macro = (
        per_cat.groupby(["provider", "model"])[["act", "abstain", "paired"]]
        .mean()
        .reset_index()
        .sort_values("paired", ascending=False)
        .reset_index(drop=True)
    )

    rows: list[tuple[str, float, float, float]] = []
    for _, r in macro.iterrows():
        key = (r["provider"], r["model"])
        if key not in MODEL_CONFIG:
            continue
        rows.append((MODEL_CONFIG[key][0], float(r["act"]), float(r["abstain"]),
                     float(r["paired"])))

    n = len(rows)
    # Top-ranked at the top of the chart, like the original dumbbell.
    y_pos = np.arange(n)[::-1]

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=300)

    for (name, act, abstain, paired), y in zip(rows, y_pos):
        # Connector first so dots draw on top.
        ax.plot([min(act, abstain), max(act, abstain)], [y, y],
                color=COLOR_CONNECTOR, linewidth=2.0, solid_capstyle="round",
                zorder=2)
        # Dots
        ax.scatter([act], [y], color=COLOR_ACT, s=70, zorder=3, edgecolor="white",
                   linewidth=0.8)
        ax.scatter([abstain], [y], color=COLOR_ABSTAIN, s=70, zorder=3,
                   edgecolor="white", linewidth=0.8)
        # Numeric labels — placed on the *outer* side of each dot so they
        # don't overlap each other or the connector.
        if act >= abstain:
            ax.text(act + 0.012, y, f"{act:.2f}", va="center", ha="left",
                    fontsize=10, color=COLOR_ACT, fontweight="bold")
            ax.text(abstain - 0.012, y, f"{abstain:.2f}", va="center", ha="right",
                    fontsize=10, color=COLOR_ABSTAIN, fontweight="bold")
        else:
            ax.text(act - 0.012, y, f"{act:.2f}", va="center", ha="right",
                    fontsize=10, color=COLOR_ACT, fontweight="bold")
            ax.text(abstain + 0.012, y, f"{abstain:.2f}", va="center", ha="left",
                    fontsize=10, color=COLOR_ABSTAIN, fontweight="bold")

    # Y-axis: model name + Paired in parens (bold) to anchor the row's
    # headline number without needing a separate column.
    y_labels = [f"{name}  ({paired:.2f})" for (name, _a, _b, paired) in rows]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=12, color=COLOR_LABEL)

    ax.set_xlim(0.30, 1.00)
    ax.set_xticks(np.arange(0.3, 1.01, 0.1))
    ax.set_xlabel("Accuracy", fontsize=13, color="#333333")

    ax.xaxis.grid(True, alpha=0.25, zorder=0, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", colors="#666666", labelsize=11)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, n - 0.4)

    # Bottom-pinned legend, matching figure_ranking_bar.py convention.
    plt.subplots_adjust(left=0.30, right=0.96, top=0.97, bottom=0.13)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", markersize=7,
                   markerfacecolor=COLOR_ACT, markeredgecolor="white",
                   markeredgewidth=0.8, label="Act Accuracy"),
        plt.Line2D([0], [0], marker="o", linestyle="None", markersize=7,
                   markerfacecolor=COLOR_ABSTAIN, markeredgecolor="white",
                   markeredgewidth=0.8, label="Abstain Accuracy"),
        plt.Line2D([0], [0], color=COLOR_CONNECTOR, linewidth=2.0,
                   label="Gap (row label = Paired Acc.)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        bbox_transform=fig.transFigure,
        fontsize=11, frameon=False,
        ncol=3, handlelength=1.6, handletextpad=0.5,
        columnspacing=1.4, borderpad=0.1,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "act_abstain_dumbbell.pdf"
    png_path = OUTPUT_DIR / "act_abstain_dumbbell.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    print(f"models plotted: {n}")
    for name, act, abstain, paired in rows:
        gap = act - abstain
        print(f"  {name:<22s}  Act={act:.3f}  Abstain={abstain:.3f}  "
              f"Paired={paired:.3f}  gap={gap:+.3f}")


if __name__ == "__main__":
    main()
