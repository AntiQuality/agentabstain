"""Restyle the appendix info_level_all_metrics_panel figure.

2x2 panel grid (Act / Abstain / Paired / CAR), each showing 17 models
x 3 information levels (grouped bars). Reads
``analysis/output/tier_2/a8_info_hierarchy/info_level_long.csv``.

Same denim/sage/coral progression for the 3 info levels as
``figure_abstain_by_info_level_bar.py``.

Usage:
    python eval/statistics/appendix_analysis/figure_info_level_all_metrics_panel.py
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

from eval.statistics.figure_ranking_bar import PALETTES

plt.rcParams["font.family"] = "Lato"

OUTPUT_DIR = SCRIPT_DIR / "figures"
SOURCE_CSV = REPO_ROOT / "analysis/output/tier_2/a8_info_hierarchy/info_level_long.csv"

figure_width = 14.0
figure_height = 7.5

INFO_PALETTE = {
    "Iq":      PALETTES["environment"]["Act"],
    "Iq+TL":   PALETTES["environment"]["Paired"],
    "Iq+TL+o": PALETTES["environment"]["Abstain"],
}
INFO_LABELS = {
    "Iq":      "Iq (query alone)",
    "Iq+TL":   "Iq+TL (+ tool list)",
    "Iq+TL+o": "Iq+TL+o (+ tool outputs)",
}
INFO_ORDER = ["Iq", "Iq+TL", "Iq+TL+o"]

METRIC_DISPLAY = {
    "act_accuracy":     "Act Accuracy",
    "abstain_accuracy": "Abstain Accuracy",
    "paired_accuracy":  "Paired Accuracy",
    "car":              "CAR",
}
METRIC_ORDER = ["act_accuracy", "abstain_accuracy", "paired_accuracy", "car"]

LEADERBOARD_ORDER = [
    "Gemini 3.1 Pro", "Claude Opus 4.7", "Claude Sonnet 4.6", "GPT-5.5",
    "Claude Haiku 4.5", "GPT-5", "GPT-5.4", "GLM-5", "GPT-OSS 120B",
    "GPT-5.2", "MiniMax M2.5", "DeepSeek V3.2", "GPT-5.1",
    "Gemini 3 Flash", "DeepSeek V4 Pro", "Kimi K2.5", "GPT-4o",
]


def main() -> None:
    df = pd.read_csv(SOURCE_CSV)

    models_in_data = set(df["model_display"].unique())
    model_order = [m for m in LEADERBOARD_ORDER if m in models_in_data]
    n = len(model_order)

    fig, axes = plt.subplots(2, 2, figsize=(figure_width, figure_height),
                             sharey=True, dpi=300)
    axes_flat = axes.flatten()

    bar_w = 0.78 / len(INFO_ORDER)
    x_pos = np.arange(n)

    for ax, metric in zip(axes_flat, METRIC_ORDER):
        sub = df[df["metric"] == metric]
        for i, info in enumerate(INFO_ORDER):
            info_sub = sub[sub["info_level"] == info].set_index("model_display")
            values = [info_sub.loc[m, "value"] if m in info_sub.index
                      else np.nan for m in model_order]
            offsets = x_pos - 0.78 / 2 + (i + 0.5) * bar_w
            ax.bar(offsets, values, width=bar_w,
                   color=INFO_PALETTE[info], edgecolor="white",
                   linewidth=0.4,
                   label=INFO_LABELS[info] if ax is axes_flat[0] else None,
                   zorder=3)
            info_mean = sub[sub["info_level"] == info]["value"].mean()
            ax.axhline(info_mean, color=INFO_PALETTE[info],
                       linestyle="--", linewidth=0.7, alpha=0.5, zorder=2)

        ax.set_title(METRIC_DISPLAY[metric], fontsize=11, color="#333333")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_order, fontsize=8, rotation=30,
                           ha="right", rotation_mode="anchor", color="#444444")
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.01, 0.2))

        ax.yaxis.grid(True, alpha=0.25, zorder=0, color="#cccccc")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#999999")
        ax.spines["left"].set_color("#999999")
        ax.tick_params(axis="y", colors="#666666", labelsize=8)
        ax.tick_params(axis="x", length=0)

    for ax in axes[:, 0]:
        ax.set_ylabel("Score", fontsize=11, color="#333333")

    plt.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.13,
                        wspace=0.05, hspace=0.55)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=INFO_PALETTE[info],
                      edgecolor="white", linewidth=0.4,
                      label=INFO_LABELS[info])
        for info in INFO_ORDER
    ]
    fig.legend(handles=legend_handles,
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               bbox_transform=fig.transFigure,
               ncol=3, fontsize=10, frameon=False,
               handlelength=1.4, handletextpad=0.5,
               columnspacing=2.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "info_level_all_metrics_panel.pdf"
    png_path = OUTPUT_DIR / "info_level_all_metrics_panel.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    print(f"models plotted: {n}")


if __name__ == "__main__":
    main()
