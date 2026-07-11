"""Restyle the appendix provider_boxplot figure.

Per-provider distributions of the four headline metrics (Act, Abstain,
Paired, CAR), one panel per metric. Each box is a provider; individual
model dots overlaid. Reads
``analysis/output/tier_2/a12_provider/per_model_long.csv``.

Provider colors reuse the harness palette so a single visual identity
flows across the appendix (Anthropic = Claude SDK orange, OpenAI =
OpenAI SDK green, Google = Google ADK blue, Open-weight = OpenClaw red).

Usage:
    python eval/statistics/appendix_analysis/figure_provider_boxplot.py
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

figure_width = 13.5
figure_height = 3.8

PROVIDER_ORDER = ["Anthropic", "OpenAI", "Google", "Open-weight"]
PROVIDER_COLORS = {
    "Anthropic":   HARNESS_COLORS["Claude SDK"],
    "OpenAI":      HARNESS_COLORS["OpenAI SDK"],
    "Google":      HARNESS_COLORS["Google ADK"],
    "Open-weight": HARNESS_COLORS["OpenClaw"],
}
METRIC_ORDER = ["Act", "Abstain", "Paired", "CAR"]


def main() -> None:
    df = pd.read_csv(SOURCE_CSV)

    fig, axes = plt.subplots(1, 4, figsize=(figure_width, figure_height),
                             sharey=True, dpi=300)

    for ax, metric in zip(axes, METRIC_ORDER):
        sub = df[df["Metric"] == metric]
        data_per_provider = [
            sub[sub["Provider"] == p]["Value"].values for p in PROVIDER_ORDER
        ]

        positions = np.arange(len(PROVIDER_ORDER))
        bp = ax.boxplot(
            data_per_provider, positions=positions, widths=0.55,
            patch_artist=True, showmeans=True, zorder=3,
            medianprops=dict(color="white", linewidth=1.6),
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="#333333", markersize=6),
            whiskerprops=dict(color="#777777", linewidth=1.0),
            capprops=dict(color="#777777", linewidth=1.0),
            flierprops=dict(marker="o", markersize=4, markerfacecolor="white",
                            markeredgecolor="#888888", markeredgewidth=0.6,
                            alpha=0.85),
        )
        for patch, prov in zip(bp["boxes"], PROVIDER_ORDER):
            patch.set_facecolor(PROVIDER_COLORS[prov])
            patch.set_edgecolor("white")
            patch.set_linewidth(0.8)
            patch.set_alpha(0.85)

        rng = np.random.default_rng(42)
        for pos, prov, vals in zip(positions, PROVIDER_ORDER, data_per_provider):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals,
                       facecolor="white",
                       edgecolor=PROVIDER_COLORS[prov],
                       linewidth=0.9, s=22, zorder=4, alpha=0.9)

        ax.set_title(metric, fontsize=11, color="#333333", pad=6)
        ax.set_xticks(positions)
        ax.set_xticklabels(PROVIDER_ORDER, fontsize=9, rotation=20,
                           ha="right", rotation_mode="anchor",
                           color="#444444")
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.yaxis.grid(True, alpha=0.25, zorder=0, color="#cccccc")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#999999")
        ax.spines["left"].set_color("#999999")
        ax.tick_params(axis="y", colors="#666666", labelsize=9)
        ax.tick_params(axis="x", length=0)

    axes[0].set_ylabel("Score", fontsize=11, color="#333333")

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "provider_boxplot.pdf"
    png_path = OUTPUT_DIR / "provider_boxplot.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    print(f"providers: {PROVIDER_ORDER}    metrics: {METRIC_ORDER}")
    print(f"rows: {len(df)}")


if __name__ == "__main__":
    main()
