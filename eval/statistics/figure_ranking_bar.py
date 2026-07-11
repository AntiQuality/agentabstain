"""Generate the per-model 4-metric ranking bar chart.

Mirrors the computation in ``analysis/analyze.py::a5_ranking`` but
restyled for the paper: Lato font, no chart title, ``Average`` as the
only y-axis label.

Two palettes are supported:
- ``environment`` (default): the lifted editorial palette shared with
  ``visualize/visualize_environment.ipynb`` (medium denim blue / mint
  sage / coral terracotta / warm gold).
- ``leaderboard``: the four harness brand hues from
  ``figure_leaderboard_bar.py`` (kept as a comparison option).

Usage:
    python eval/statistics/figure_ranking_bar.py
    python eval/statistics/figure_ranking_bar.py --palette leaderboard

Output (canonical):
    eval/statistics/figures/ranking_bar.pdf
    eval/statistics/figures/ranking_bar.png
And per-palette comparison copies:
    eval/statistics/figures/ranking_bar_<palette>.png
"""
from __future__ import annotations

import argparse
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
from eval.statistics.figure_leaderboard_bar import (
    HARNESS_COLORS,
    MODEL_CONFIG,
    canon,
)

plt.rcParams["font.family"] = "Lato"

OUTPUT_DIR = SCRIPT_DIR / "figures"

# Source figure dimensions kept close to paper \linewidth so font
# sizes don't shrink to illegibility when included as
# \includegraphics[width=\linewidth]. NeurIPS \linewidth is ~5.5 in;
# at 8 in source width the rendered scale factor is ~0.69, so 11pt
# source text renders around 7.6pt in the paper.
figure_width = 10
figure_height = 2.8
group_width = 0.78
bar_edge = 0.4

# Direct-semantic mapping: blue = "do" (Act), red = "stop" (Abstain),
# green = "both passed" (Paired), gold = derived metric (CAR).
PALETTES = {
    # Lifted editorial palette — hue families from the original
    # visualize/visualize_environment.ipynb set, with balanced
    # lightness. Hex codes match the env notebook; only the
    # metric→color mapping is paper-specific.
    "environment": {
        "Act":     "#5b89b3",  # medium denim blue — "do"
        "Abstain": "#d46a52",  # coral terracotta — "stop" (red-as-stop)
        "Paired":  "#7fb084",  # mint sage — "both passed" (green-as-pass)
        "CAR":     "#e6b860",  # warm gold — derived metric
    },
    # Brand-saturated, matches figure_leaderboard_bar.py.
    "leaderboard": {
        "Act":     HARNESS_COLORS["Google ADK"],   # #4285F4 blue
        "Abstain": HARNESS_COLORS["OpenAI SDK"],   # #10A37F teal
        "Paired":  HARNESS_COLORS["OpenClaw"],     # #E15759 red
        "CAR":     HARNESS_COLORS["Claude SDK"],   # #FC8D59 orange
    },
}
METRIC_ORDER = ("Act", "Abstain", "Paired", "CAR")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--palette", choices=sorted(PALETTES), default="environment")
    args = parser.parse_args()
    metric_colors = PALETTES[args.palette]

    df = load_eval_frame(REPO_ROOT / "results")
    df = select_latest_runs(df)
    df["model"] = df["model"].map(canon)
    df = df[df["task_id"].fillna("").str.startswith("preview")]
    metric_df = build_metric_run_frame(df)
    summary = compute_summary_metrics(metric_df)

    combined = summary[summary["metric_name"] == "combined"].copy()

    # Average across action_types within each (provider, model, category)
    # — same equal-weight rollup as figure_leaderboard_bar.py — then
    # macro-average across the 8 categories per model.
    per_cat = (
        combined.groupby(["provider", "model", "category"])
        .agg(
            act=("should_act_accuracy", "mean"),
            abstain=("should_abstain_accuracy", "mean"),
            paired=("paired_accuracy", "mean"),
            car=("car", "mean"),
        )
        .reset_index()
    )
    macro = (
        per_cat.groupby(["provider", "model"])[["act", "abstain", "paired", "car"]]
        .mean()
        .reset_index()
        .sort_values("paired", ascending=False)
        .reset_index(drop=True)
    )

    display_names: list[str] = []
    rows: list[tuple[float, float, float, float]] = []
    for _, row in macro.iterrows():
        key = (row["provider"], row["model"])
        if key not in MODEL_CONFIG:
            continue
        display_names.append(MODEL_CONFIG[key][0])
        rows.append((row["act"], row["abstain"], row["paired"], row["car"]))

    n = len(display_names)
    values = np.array(rows, dtype=float)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=300)
    x_pos = np.arange(n)
    bar_w = group_width / len(METRIC_ORDER)
    for i, metric in enumerate(METRIC_ORDER):
        offsets = x_pos - group_width / 2 + (i + 0.5) * bar_w
        ax.bar(
            offsets,
            values[:, i],
            width=bar_w,
            color=metric_colors[metric],
            edgecolor="white",
            linewidth=bar_edge,
            label=metric,
            zorder=3,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, fontsize=11, rotation=30, ha="right",
                       rotation_mode="anchor", color="#444444")
    ax.set_xlim(-0.6, n - 0.4)

    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([f"{int(round(v * 100))}" for v in np.arange(0, 1.01, 0.2)])
    ax.set_ylabel("Average Accuracy (%)", fontsize=12, color="#333333")

    ax.yaxis.grid(True, alpha=0.25, zorder=0, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["left"].set_color("#999999")
    ax.tick_params(axis="y", colors="#666666", labelsize=10)
    ax.tick_params(axis="x", length=0)

    # Manual layout: tight_layout ignores bbox_to_anchor legends, so
    # reserve the bottom margin for both the rotated labels and the
    # legend ourselves. Labels live in the upper part of the bottom
    # band; the legend is pinned at the very bottom of the figure.
    plt.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.40)
    legend_labels = {
        "Act":     "Act Accuracy",
        "Abstain": "Abstain Accuracy",
        "Paired":  "Paired Accuracy",
        "CAR":     "CAR (Conditioned Abstention Rate)",
    }
    handles, _ = ax.get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=[legend_labels[m] for m in METRIC_ORDER],
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        bbox_transform=fig.transFigure,
        fontsize=11, frameon=False,
        ncol=4, handlelength=1.5, handleheight=0.6, handletextpad=0.5,
        columnspacing=1.4, borderpad=0.1,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "ranking_bar.pdf"
    png_path = OUTPUT_DIR / "ranking_bar.png"
    palette_png = OUTPUT_DIR / f"ranking_bar_{args.palette}.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(palette_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path}, {png_path}, {palette_png}  (palette={args.palette})")
    print(f"models plotted: {n}")
    for name, vals in zip(display_names, values):
        print(f"  {name:<22s}  Act={vals[0]:.4f}  Abstain={vals[1]:.4f}  "
              f"Paired={vals[2]:.4f}  CAR={vals[3]:.4f}")


if __name__ == "__main__":
    main()
