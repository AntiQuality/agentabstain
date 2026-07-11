"""Restyle the appendix phase_act_abstain_bar figure.

Reads the same long-format CSV that
``analysis/analyze.py:a2_phase_breakdown`` writes
(``analysis/output/tier_1/a2_phase_breakdown/phase_breakdown_long.csv``)
and renders the two-panel grouped bar chart (Act Accuracy + Abstain
Accuracy, faceted; bars grouped by phase) with the paper's Lato font
and the lifted ``environment`` palette. Dashed horizontal reference
lines at each phase's cross-model mean are preserved.

No internal title — the LaTeX caption handles labeling.

Usage:
    python eval/statistics/appendix_analysis/figure_phase_act_abstain_bar.py
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

from eval.statistics.figure_leaderboard_bar import MODEL_CONFIG, canon
from eval.statistics.figure_ranking_bar import PALETTES

plt.rcParams["font.family"] = "Lato"

OUTPUT_DIR = SCRIPT_DIR / "figures"
SOURCE_CSV = REPO_ROOT / "analysis/output/tier_1/a2_phase_breakdown/phase_breakdown_long.csv"

figure_width = 13.0
figure_height = 4.0

# Phase encoded with denim (pre-execution) vs coral (runtime). Same
# semantic intent as Act/Abstain in the metric palette: cool = "before
# execution / planning" and warm = "during execution / response".
PHASE_PALETTE = {
    "Pre-Execution (S1–S5)": PALETTES["environment"]["Act"],
    "Runtime (S6–S8)":       PALETTES["environment"]["Abstain"],
}
PHASE_ORDER = list(PHASE_PALETTE.keys())

# Use the leaderboard's canonical order so x-axis order matches other
# main-body charts (sorted by Paired Accuracy ranking implicitly).
LEADERBOARD_ORDER = [
    "Gemini 3.1 Pro", "Claude Opus 4.7", "Claude Sonnet 4.6", "GPT-5.5",
    "Claude Haiku 4.5", "GPT-5", "GPT-5.4", "GLM-5", "GPT-OSS 120B",
    "GPT-5.2", "MiniMax M2.5", "DeepSeek V3.2", "GPT-5.1",
    "Gemini 3 Flash", "DeepSeek V4 Pro", "Kimi K2.5", "GPT-4o",
]


def main() -> None:
    df = pd.read_csv(SOURCE_CSV)
    df = df[df["metric"].isin(["act_accuracy", "abstain_accuracy"])].copy()
    df["metric_display"] = df["metric"].map({
        "act_accuracy": "Act Accuracy",
        "abstain_accuracy": "Abstain Accuracy",
    })

    models_in_data = set(df["model_display"].unique())
    model_order = [m for m in LEADERBOARD_ORDER if m in models_in_data]
    n = len(model_order)

    fig, axes = plt.subplots(1, 2, figsize=(figure_width, figure_height),
                             sharey=True, dpi=300)

    bar_w = 0.4
    x_pos = np.arange(n)

    for ax, metric_disp in zip(axes, ["Act Accuracy", "Abstain Accuracy"]):
        sub = df[df["metric_display"] == metric_disp]
        for i, phase in enumerate(PHASE_ORDER):
            phase_sub = sub[sub["phase_display"] == phase].set_index("model_display")
            values = [phase_sub.loc[m, "value"] if m in phase_sub.index else np.nan
                      for m in model_order]
            offset = (i - 0.5) * bar_w
            ax.bar(x_pos + offset, values, width=bar_w,
                   color=PHASE_PALETTE[phase], edgecolor="white",
                   linewidth=0.4, label=phase if ax is axes[0] else None,
                   zorder=3)
            # Dashed reference: cross-model mean for this phase.
            phase_mean = sub[sub["phase_display"] == phase]["value"].mean()
            ax.axhline(phase_mean, color=PHASE_PALETTE[phase],
                       linestyle="--", linewidth=0.8, alpha=0.6, zorder=2)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_order, fontsize=9, rotation=30, ha="right",
                           rotation_mode="anchor", color="#444444")
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        if ax is axes[0]:
            ax.set_ylabel("Accuracy", fontsize=11, color="#333333")
        ax.set_xlabel("")
        ax.text(0.02, 0.96, metric_disp, transform=ax.transAxes,
                ha="left", va="top", fontsize=11, color="#333333",
                fontweight="bold")

        ax.yaxis.grid(True, alpha=0.25, zorder=0, color="#cccccc")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#999999")
        ax.spines["left"].set_color("#999999")
        ax.tick_params(axis="y", colors="#666666", labelsize=9)
        ax.tick_params(axis="x", length=0)

    # Legend at the bottom of the figure (matches main-body convention).
    plt.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.27,
                        wspace=0.05)
    handles = [plt.Rectangle((0, 0), 1, 1,
                             facecolor=PHASE_PALETTE[p],
                             edgecolor="white", linewidth=0.5, label=p)
               for p in PHASE_ORDER]
    fig.legend(handles=handles,
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               bbox_transform=fig.transFigure,
               ncol=2, fontsize=10, frameon=False,
               handlelength=1.4, handletextpad=0.5,
               columnspacing=2.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "phase_act_abstain_bar.pdf"
    png_path = OUTPUT_DIR / "phase_act_abstain_bar.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    print(f"models plotted: {n}")


if __name__ == "__main__":
    main()
