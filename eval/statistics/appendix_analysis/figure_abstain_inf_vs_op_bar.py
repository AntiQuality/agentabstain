"""Restyle the appendix abstain_inf_vs_op_bar figure.

Per-model Abstain Accuracy split by action type (informational vs.
operational). Reads ``analysis/output/tier_2/a11_action_type/long.csv``.

Two-color encoding: denim (informational) vs. coral (operational).

Usage:
    python eval/statistics/appendix_analysis/figure_abstain_inf_vs_op_bar.py
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
SOURCE_CSV = REPO_ROOT / "analysis/output/tier_2/a11_action_type/long.csv"

figure_width = 12.0
figure_height = 4.2

ACTION_PALETTE = {
    "informational": PALETTES["environment"]["Act"],     # denim
    "operational":   PALETTES["environment"]["Abstain"], # coral
}
ACTION_LABELS = {
    "informational": "Informational",
    "operational":   "Operational",
}
ACTION_ORDER = ["informational", "operational"]

LEADERBOARD_ORDER = [
    "Gemini 3.1 Pro", "Claude Opus 4.7", "Claude Sonnet 4.6", "GPT-5.5",
    "Claude Haiku 4.5", "GPT-5", "GPT-5.4", "GLM-5", "GPT-OSS 120B",
    "GPT-5.2", "MiniMax M2.5", "DeepSeek V3.2", "GPT-5.1",
    "Gemini 3 Flash", "DeepSeek V4 Pro", "Kimi K2.5", "GPT-4o",
]


def main() -> None:
    df = pd.read_csv(SOURCE_CSV)
    df = df[(df["metric_variant"] == "combined") & (df["metric"] == "abstain_acc")]

    models_in_data = set(df["Model"].unique())
    model_order = [m for m in LEADERBOARD_ORDER if m in models_in_data]
    n = len(model_order)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=300)

    bar_w = 0.4
    x_pos = np.arange(n)

    for i, action in enumerate(ACTION_ORDER):
        action_sub = df[df["action_type"] == action].set_index("Model")
        values = [action_sub.loc[m, "value"] if m in action_sub.index
                  else np.nan for m in model_order]
        offset = (i - 0.5) * bar_w
        ax.bar(x_pos + offset, values, width=bar_w,
               color=ACTION_PALETTE[action], edgecolor="white",
               linewidth=0.4, label=ACTION_LABELS[action], zorder=3)
        action_mean = df[df["action_type"] == action]["value"].mean()
        ax.axhline(action_mean, color=ACTION_PALETTE[action],
                   linestyle="--", linewidth=0.8, alpha=0.6, zorder=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_order, fontsize=9, rotation=30,
                       ha="right", rotation_mode="anchor", color="#444444")
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Abstain Accuracy", fontsize=11, color="#333333")

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
    pdf_path = OUTPUT_DIR / "abstain_inf_vs_op_bar.pdf"
    png_path = OUTPUT_DIR / "abstain_inf_vs_op_bar.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    print(f"models plotted: {n}")


if __name__ == "__main__":
    main()
