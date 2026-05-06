"""Restyle the appendix phase_paired_bar figure.

Per-model Paired Accuracy grouped by execution phase (Pre-Execution vs.
Runtime). Reads the same long-format CSV that
``analysis/analyze.py:a2_phase_breakdown`` writes. Single-panel grouped
bar chart with dashed reference lines at each phase's cross-model mean.

Per-model significance stars (Bonferroni-corrected Fisher's exact)
preserved from the source ``phase_summary.csv`` ``sig`` column.

Usage:
    python eval/statistics/appendix_analysis/figure_phase_paired_bar.py
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
SOURCE_LONG = REPO_ROOT / "analysis/output/tier_1/a2_phase_breakdown/phase_breakdown_long.csv"
SOURCE_SUMMARY = REPO_ROOT / "analysis/output/tier_1/a2_phase_breakdown/phase_summary.csv"

figure_width = 11.0
figure_height = 4.0

PHASE_PALETTE = {
    "Pre-Execution (S1–S5)": PALETTES["environment"]["Act"],
    "Runtime (S6–S8)":       PALETTES["environment"]["Abstain"],
}
PHASE_ORDER = list(PHASE_PALETTE.keys())

LEADERBOARD_ORDER = [
    "Gemini 3.1 Pro", "Claude Opus 4.7", "Claude Sonnet 4.6", "GPT-5.5",
    "Claude Haiku 4.5", "GPT-5", "GPT-5.4", "GLM-5", "GPT-OSS 120B",
    "GPT-5.2", "MiniMax M2.5", "DeepSeek V3.2", "GPT-5.1",
    "Gemini 3 Flash", "DeepSeek V4 Pro", "Kimi K2.5", "GPT-4o",
]


def main() -> None:
    long = pd.read_csv(SOURCE_LONG)
    paired = long[long["metric"] == "paired_accuracy"].copy()
    summary = pd.read_csv(SOURCE_SUMMARY).set_index("Model")
    sig_lookup = summary["sig"].to_dict()

    models_in_data = set(paired["model_display"].unique())
    model_order = [m for m in LEADERBOARD_ORDER if m in models_in_data]
    n = len(model_order)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=300)

    bar_w = 0.4
    x_pos = np.arange(n)

    for i, phase in enumerate(PHASE_ORDER):
        phase_sub = paired[paired["phase_display"] == phase].set_index("model_display")
        values = [phase_sub.loc[m, "value"] if m in phase_sub.index else np.nan
                  for m in model_order]
        offset = (i - 0.5) * bar_w
        ax.bar(x_pos + offset, values, width=bar_w,
               color=PHASE_PALETTE[phase], edgecolor="white",
               linewidth=0.4, label=phase, zorder=3)
        # Cross-model mean for this phase.
        phase_mean = paired[paired["phase_display"] == phase]["value"].mean()
        ax.axhline(phase_mean, color=PHASE_PALETTE[phase],
                   linestyle="--", linewidth=0.9, alpha=0.6, zorder=2)

    # Per-model significance stars (Bonferroni Fisher's exact).
    for i, model in enumerate(model_order):
        marker = sig_lookup.get(model, "")
        if marker and marker != "ns":
            model_max = paired[paired["model_display"] == model]["value"].max()
            ax.text(i, min(model_max + 0.04, 0.97), marker,
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color="#333333")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_order, fontsize=9, rotation=30,
                       ha="right", rotation_mode="anchor", color="#444444")
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Paired Accuracy", fontsize=11, color="#333333")

    ax.yaxis.grid(True, alpha=0.25, zorder=0, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["left"].set_color("#999999")
    ax.tick_params(axis="y", colors="#666666", labelsize=9)
    ax.tick_params(axis="x", length=0)

    ax.legend(
        loc="upper right", fontsize=10, frameon=False,
        handlelength=1.4, handletextpad=0.5,
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "phase_paired_bar.pdf"
    png_path = OUTPUT_DIR / "phase_paired_bar.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    print(f"models plotted: {n}")


if __name__ == "__main__":
    main()
