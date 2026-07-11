"""Restyle the appendix car_vs_act_bar figure.

Per-model grouped bar: Act Accuracy (denim) vs. CAR (gold). Reads
``analysis/output/tier_1/a4_car/car_summary.csv``. Models in
canonical leaderboard order (sorted by Paired Accuracy in main).

Usage:
    python eval/statistics/appendix_analysis/figure_car_vs_act_bar.py
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
SOURCE_CSV = REPO_ROOT / "analysis/output/tier_1/a4_car/car_summary.csv"

figure_width = 11.0
figure_height = 4.0

# Same metric->color mapping as the main-body ranking bar so a reader
# can carry the color key across figures.
COLOR_ACT = PALETTES["environment"]["Act"]   # denim blue
COLOR_CAR = PALETTES["environment"]["CAR"]   # gold

LEADERBOARD_ORDER = [
    "Gemini 3.1 Pro", "Claude Opus 4.7", "Claude Sonnet 4.6", "GPT-5.5",
    "Claude Haiku 4.5", "GPT-5", "GPT-5.4", "GLM-5", "GPT-OSS 120B",
    "GPT-5.2", "MiniMax M2.5", "DeepSeek V3.2", "GPT-5.1",
    "Gemini 3 Flash", "DeepSeek V4 Pro", "Kimi K2.5", "GPT-4o",
]


def main() -> None:
    df = pd.read_csv(SOURCE_CSV)
    df = df.set_index("Model")

    model_order = [m for m in LEADERBOARD_ORDER if m in df.index]
    n = len(model_order)
    df = df.reindex(model_order)

    # Use the macro values (mean across categories), matching A1
    # paired_accuracy_heatmap's Macro Avg semantics.
    act = df["Act Accuracy (macro)"].values
    car = df["CAR (macro)"].values

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=300)

    bar_w = 0.4
    x_pos = np.arange(n)

    ax.bar(x_pos - bar_w / 2, act, width=bar_w,
           color=COLOR_ACT, edgecolor="white", linewidth=0.4,
           label="Act Accuracy", zorder=3)
    ax.bar(x_pos + bar_w / 2, car, width=bar_w,
           color=COLOR_CAR, edgecolor="white", linewidth=0.4,
           label="CAR", zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_order, fontsize=9, rotation=30,
                       ha="right", rotation_mode="anchor", color="#444444")
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Score", fontsize=11, color="#333333")

    ax.yaxis.grid(True, alpha=0.25, zorder=0, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["left"].set_color("#999999")
    ax.tick_params(axis="y", colors="#666666", labelsize=9)
    ax.tick_params(axis="x", length=0)

    ax.legend(loc="upper right", fontsize=10, frameon=False,
              handlelength=1.4, handletextpad=0.5)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "car_vs_act_bar.pdf"
    png_path = OUTPUT_DIR / "car_vs_act_bar.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    print(f"models plotted: {n}")


if __name__ == "__main__":
    main()
