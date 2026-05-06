"""Restyle the appendix paired-accuracy heatmap.

Reads the same per-model x per-category CSV that
``analysis/analyze.py:a1_main_results`` writes
(``analysis/output/tier_1/a1_main_results/paired_accuracy.csv``) and
renders it with the paper's Lato font and a denim sequential colormap
(lifted ``environment`` palette aesthetic). No internal chart title —
the LaTeX caption handles labeling.

Numbers are not recomputed; this is a pure restyle of the existing
analysis output.

Usage:
    python eval/statistics/appendix_analysis/figure_paired_accuracy_heatmap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.statistics.figure_ranking_bar import PALETTES

plt.rcParams["font.family"] = "Lato"

OUTPUT_DIR = SCRIPT_DIR / "figures"
SOURCE_CSV = REPO_ROOT / "analysis/output/tier_1/a1_main_results/paired_accuracy.csv"

# Sequential blue colormap (light cyan-blue to medium blue), tuned for
# heatmap legibility — lighter low end keeps small values readable on
# white background; darker high end is the colorbrewer "RdYlBu"
# upper-blue stop.
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "appendix_blue_seq",
    ["#E0F3F8", "#08306B"],
    N=256,
)

figure_width = 11.0
figure_height = 6.5


def main() -> None:
    df = pd.read_csv(SOURCE_CSV).set_index("Model")

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=300)

    values = df.values
    im = ax.imshow(values, cmap=HEATMAP_CMAP, vmin=0, vmax=1, aspect="auto")

    # Cell annotations. Text color flips to white where the cell is dark
    # enough that black text loses contrast.
    threshold = 0.65
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            color = "white" if v > threshold else "#222222"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color=color)

    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels(df.columns, fontsize=10, rotation=30,
                       ha="right", rotation_mode="anchor", color="#333333")
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels(df.index, fontsize=10, color="#333333")
    ax.set_xlabel("Category", fontsize=11, color="#333333")
    ax.set_ylabel("Model", fontsize=11, color="#333333")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["left"].set_color("#999999")
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Paired Accuracy", fontsize=10, color="#333333")
    cbar.ax.tick_params(labelsize=9, colors="#666666")
    cbar.outline.set_visible(False)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "paired_accuracy_heatmap.pdf"
    png_path = OUTPUT_DIR / "paired_accuracy_heatmap.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    print(f"shape: {values.shape[0]} models x {values.shape[1]} columns")


if __name__ == "__main__":
    main()
