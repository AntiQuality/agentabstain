"""Restyle the appendix scaling_bar figure.

Companion to ``figure_scaling_lineplot.py`` in main: same per-family
data but rendered as 4-metric grouped bars per model. Reads
``analysis/output/tier_2/a7_scaling/scaling_long.csv``.

2x2 panel grid (Claude / GPT-5 / Gemini / Open-weight). Bar color =
metric (lifted env palette, matching every other 4-metric chart in
the paper).

Usage:
    python eval/statistics/appendix_analysis/figure_scaling_bar.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.statistics.figure_ranking_bar import METRIC_ORDER, PALETTES

plt.rcParams["font.family"] = "Lato"

OUTPUT_DIR = SCRIPT_DIR / "figures"
SOURCE_CSV = REPO_ROOT / "analysis/output/tier_2/a7_scaling/scaling_long.csv"

figure_width = 12.0
figure_height = 5.5

METRIC_COLORS = PALETTES["environment"]
LEGEND_LABELS = {
    "Act":     "Act Accuracy",
    "Abstain": "Abstain Accuracy",
    "Paired":  "Paired Accuracy",
    "CAR":     "CAR",
}

FAMILY_DISPLAY = {
    "Claude (size)":                    "Claude",
    "OpenAI GPT-5 (version)":           "GPT-5",
    "Gemini (size: Flash → Pro)":       "Gemini",
    "Open-weight (size: total params)": "Open-weight",
}


def main() -> None:
    df = pd.read_csv(SOURCE_CSV)
    families = list(FAMILY_DISPLAY.keys())

    fig, axes = plt.subplots(2, 2, figsize=(figure_width, figure_height),
                             sharey=True, dpi=300)
    axes_flat = axes.flatten()

    for ax, family in zip(axes_flat, families):
        sub = df[df["family"] == family]
        if sub.empty:
            ax.set_axis_off()
            continue
        models = (sub.drop_duplicates("scaling_position")
                  .sort_values("scaling_position")["Model"].tolist())
        n = len(models)
        x_pos = np.arange(n)
        bar_w = 0.78 / len(METRIC_ORDER)
        for i, metric in enumerate(METRIC_ORDER):
            metric_sub = sub[sub["metric"] == metric].set_index("Model")
            values = [metric_sub.loc[m, "value"] if m in metric_sub.index
                      else np.nan for m in models]
            offsets = x_pos - 0.78 / 2 + (i + 0.5) * bar_w
            ax.bar(offsets, values, width=bar_w,
                   color=METRIC_COLORS[metric], edgecolor="white",
                   linewidth=0.4, label=metric if ax is axes_flat[0] else None,
                   zorder=3)

        ax.set_title(FAMILY_DISPLAY[family], fontsize=11, color="#333333")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, fontsize=9, rotation=20,
                           ha="right", rotation_mode="anchor", color="#444444")
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xlim(-0.6, n - 0.4)

        ax.yaxis.grid(True, alpha=0.25, zorder=0, color="#cccccc")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#999999")
        ax.spines["left"].set_color("#999999")
        ax.tick_params(axis="y", colors="#666666", labelsize=9)
        ax.tick_params(axis="x", length=0)

    for ax in axes[:, 0]:
        ax.set_ylabel("Average", fontsize=11, color="#333333")

    plt.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.13,
                        wspace=0.07, hspace=0.45)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=METRIC_COLORS[m],
                      edgecolor="white", linewidth=0.4,
                      label=LEGEND_LABELS[m])
        for m in METRIC_ORDER
    ]
    fig.legend(handles=legend_handles,
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               bbox_transform=fig.transFigure,
               ncol=4, fontsize=10, frameon=False,
               handlelength=1.4, handletextpad=0.5,
               columnspacing=2.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "scaling_bar.pdf"
    png_path = OUTPUT_DIR / "scaling_bar.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    print(f"families plotted: {len(families)}")


if __name__ == "__main__":
    main()
