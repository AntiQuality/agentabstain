"""Restyle the appendix three_level_by_category_bar figure.

Per-category commit-check pass rate at three strictness levels (Loose,
Key-presence, Value-match), pooled across all 17 models on operational
ACT tasks. Reads
``analysis/output/tier_2/a10_three_level_critical/three_level_by_category.csv``.

Usage:
    python eval/statistics/appendix_analysis/figure_three_level_by_category_bar.py
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
SOURCE_CSV = REPO_ROOT / "analysis/output/tier_2/a10_three_level_critical/three_level_by_category.csv"

figure_width = 8.5
figure_height = 4.0

LEVEL_PALETTE = {
    "Loose":        PALETTES["environment"]["Act"],
    "Key-presence": PALETTES["environment"]["CAR"],
    "Value-match":  PALETTES["environment"]["Abstain"],
}
LEVEL_ORDER = ["Loose", "Key-presence", "Value-match"]


def main() -> None:
    df = pd.read_csv(SOURCE_CSV)
    df = df.sort_values("Loose", ascending=False).reset_index(drop=True)

    n = len(df)
    x_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=300)

    bar_w = 0.78 / len(LEVEL_ORDER)
    for i, level in enumerate(LEVEL_ORDER):
        offsets = x_pos - 0.78 / 2 + (i + 0.5) * bar_w
        ax.bar(offsets, df[level].values, width=bar_w,
               color=LEVEL_PALETTE[level], edgecolor="white",
               linewidth=0.4, label=level, zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(df["Category"], fontsize=10, rotation=20,
                       ha="right", rotation_mode="anchor", color="#444444")
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Commit-check pass rate", fontsize=11, color="#333333")

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
    pdf_path = OUTPUT_DIR / "three_level_by_category_bar.pdf"
    png_path = OUTPUT_DIR / "three_level_by_category_bar.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    print(f"categories plotted: {n}")


if __name__ == "__main__":
    main()
