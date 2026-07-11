"""Restyle the appendix act_vs_abstain_by_category figure.

8-panel small-multiples scatter (one per category). Each panel plots
the 17 models at their (Act, Abstain) for that category, with Pearson
r in the panel title. Reads
``analysis/output/tier_1/a3_calibration_scatter/scatter_data.csv``.

Dot color = harness (matching the leaderboard / centroid in main
body), so a reader who has seen those figures already knows the color
coding. y=x diagonal as reference. No internal suptitle.

Usage:
    python eval/statistics/appendix_analysis/figure_act_vs_abstain_by_category.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.statistics.figure_leaderboard_bar import HARNESS_COLORS, MODEL_CONFIG, canon

plt.rcParams["font.family"] = "Lato"

OUTPUT_DIR = SCRIPT_DIR / "figures"
SOURCE_CSV = REPO_ROOT / "analysis/output/tier_1/a3_calibration_scatter/scatter_data.csv"

figure_width = 13.0
figure_height = 6.5

CATEGORY_ORDER = [
    "Missing Param", "Ambiguous Action", "Conflicting Constr.", "High Stakes",
    "Insuff. Tools", "Tool Failure", "Conflicting Evid.", "Emergent Risk",
]


def _sig_star(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _harness_for(model_display: str) -> str | None:
    """Map a scatter_data ``model_display`` to its harness.

    ``analyze.py:MODEL_DISPLAY`` prefixes Claude models with the family
    name (e.g. ``Claude Haiku 4.5``), while ``MODEL_CONFIG`` uses the
    short ``Haiku 4.5`` form. Try both variants so all 17 models are
    coloured rather than silently dropped.
    """
    for (_, _), (display, harness, _) in MODEL_CONFIG.items():
        if display == model_display or model_display.endswith(display):
            return harness
    return None


def main() -> None:
    df = pd.read_csv(SOURCE_CSV)
    df["harness"] = df["model_display"].map(_harness_for)
    df = df.dropna(subset=["harness"])

    cats_present = [c for c in CATEGORY_ORDER if c in df["category_display"].unique()]
    n_cats = len(cats_present)
    n_cols = 4
    n_rows = (n_cats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figure_width, figure_height),
                             sharex=True, sharey=True, dpi=300)
    axes_flat = axes.flatten()

    for ax, cat in zip(axes_flat, cats_present):
        sub = df[df["category_display"] == cat]
        for _, row in sub.iterrows():
            ax.scatter(row["act"], row["abstain"],
                       color=HARNESS_COLORS[row["harness"]],
                       s=40, alpha=0.9, edgecolor="white", linewidth=0.6,
                       zorder=3)
        # y=x diagonal + 0.5 quadrant guides
        ax.plot([0, 1], [0, 1], color="#bbbbbb", linestyle="--",
                linewidth=0.8, alpha=0.7, zorder=1)
        ax.axhline(0.5, color="#dddddd", linestyle=":",
                   linewidth=0.6, alpha=0.7, zorder=0)
        ax.axvline(0.5, color="#dddddd", linestyle=":",
                   linewidth=0.6, alpha=0.7, zorder=0)
        if len(sub) >= 3:
            r, p = pearsonr(sub["act"], sub["abstain"])
            title = f"{cat}    r = {r:+.2f} {_sig_star(p)}"
        else:
            title = cat
        ax.set_title(title, fontsize=10, color="#333333")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", colors="#666666", labelsize=8)
        ax.tick_params(axis="y", colors="#666666", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#bbbbbb")
        ax.spines["left"].set_color("#bbbbbb")

    # Empty axes (if any)
    for ax in axes_flat[n_cats:]:
        ax.set_axis_off()

    # Shared axis labels
    for ax in axes[-1]:
        ax.set_xlabel("Act Accuracy", fontsize=10, color="#333333")
    for ax in axes[:, 0]:
        ax.set_ylabel("Abstain Accuracy", fontsize=10, color="#333333")

    # Harness legend at the bottom of the figure.
    plt.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.10,
                        wspace=0.10, hspace=0.30)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", markersize=7,
                   markerfacecolor=HARNESS_COLORS[h], markeredgecolor="white",
                   markeredgewidth=0.6, label=h)
        for h in ("Claude SDK", "OpenAI SDK", "Google ADK", "OpenClaw")
    ] + [
        plt.Line2D([0], [0], color="#bbbbbb", linestyle="--",
                   linewidth=1.0, label="Act = Abstain"),
    ]
    fig.legend(handles=legend_handles,
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               bbox_transform=fig.transFigure,
               ncol=5, fontsize=10, frameon=False,
               handlelength=1.4, handletextpad=0.5,
               columnspacing=2.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "act_vs_abstain_by_category.pdf"
    png_path = OUTPUT_DIR / "act_vs_abstain_by_category.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    print(f"categories plotted: {n_cats}")


if __name__ == "__main__":
    main()
