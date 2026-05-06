"""Restyle the appendix provider_centroid_scatter figure.

Provider 'personality' on the Act-vs-Abstain plane: each individual
model is a faint dot, each provider centroid is a large X. Reads
``analysis/output/tier_2/a12_provider/per_model_long.csv`` (per-model
mean across the 8 categories per metric is implied by the source).

Provider colors reuse the harness palette used in the rest of the
appendix (Anthropic = orange, OpenAI = green, Google = blue,
Open-weight = red).

Usage:
    python eval/statistics/appendix_analysis/figure_provider_centroid_scatter.py
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

from eval.statistics.figure_leaderboard_bar import HARNESS_COLORS

plt.rcParams["font.family"] = "Lato"

OUTPUT_DIR = SCRIPT_DIR / "figures"
SOURCE_CSV = REPO_ROOT / "analysis/output/tier_2/a12_provider/per_model_long.csv"

figure_width = 6.5
figure_height = 6.5

PROVIDER_ORDER = ["Anthropic", "OpenAI", "Google", "Open-weight"]
PROVIDER_COLORS = {
    "Anthropic":   HARNESS_COLORS["Claude SDK"],
    "OpenAI":      HARNESS_COLORS["OpenAI SDK"],
    "Google":      HARNESS_COLORS["Google ADK"],
    "Open-weight": HARNESS_COLORS["OpenClaw"],
}

LABEL_OFFSETS = {
    "Anthropic":   (10, 4, "left", "bottom"),
    "OpenAI":      (10, 4, "left", "bottom"),
    "Google":      (10, -4, "left", "top"),
    "Open-weight": (-10, -4, "right", "top"),
}


def main() -> None:
    df = pd.read_csv(SOURCE_CSV)
    pivot = df.pivot_table(index=["Provider", "Model"],
                           columns="Metric", values="Value").reset_index()

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=300)

    # Diagonal y=x
    ax.plot([0, 1], [0, 1], color="#bbbbbb", linestyle="--",
            linewidth=1.0, alpha=0.7, zorder=1)

    # Faint per-model dots
    for prov in PROVIDER_ORDER:
        sub = pivot[pivot["Provider"] == prov]
        ax.scatter(sub["Act"], sub["Abstain"],
                   color=PROVIDER_COLORS[prov], s=40, alpha=0.35,
                   edgecolor="white", linewidth=0.4, zorder=2)

    # Centroids (mean of per-model values per provider)
    centroids = []
    for prov in PROVIDER_ORDER:
        sub = pivot[pivot["Provider"] == prov]
        cx = sub["Act"].mean()
        cy = sub["Abstain"].mean()
        n = len(sub)
        centroids.append((prov, cx, cy, n))
        ax.scatter(cx, cy, color=PROVIDER_COLORS[prov], s=320,
                   marker="X", edgecolor="white", linewidth=1.6,
                   zorder=4)

    for prov, cx, cy, n in centroids:
        dx, dy, ha, va = LABEL_OFFSETS[prov]
        ax.annotate(f"{prov} (n={n})", (cx, cy),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=10, color=PROVIDER_COLORS[prov],
                    ha=ha, va=va, fontweight="bold")

    ax.text(0.02, 1.02, "Above diagonal = better at abstaining",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color="#666666")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Mean Act Accuracy across 8 categories",
                  fontsize=11, color="#333333")
    ax.set_ylabel("Mean Abstain Accuracy across 8 categories",
                  fontsize=11, color="#333333")
    ax.tick_params(axis="x", colors="#666666", labelsize=9)
    ax.tick_params(axis="y", colors="#666666", labelsize=9)

    ax.grid(True, alpha=0.2, zorder=0, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["left"].set_color("#999999")

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "provider_centroid_scatter.pdf"
    png_path = OUTPUT_DIR / "provider_centroid_scatter.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    for prov, cx, cy, n in centroids:
        print(f"  {prov:12s} n={n}  Act={cx:.3f}  Abstain={cy:.3f}")


if __name__ == "__main__":
    main()
