"""Restyle the appendix abstain_by_info_level_bar figure.

Per-model Abstain Accuracy grouped by 3 information levels:
- Iq:        query alone (pre-execution categories with no tool list)
- Iq+TL:     query + tool list
- Iq+TL+o:   query + tool list + observed tool outputs (runtime)

Reads ``analysis/output/tier_2/a8_info_hierarchy/abstain_by_info_level.csv``.

Dashed horizontal reference lines for cross-model mean per info level.

Usage:
    python eval/statistics/appendix_analysis/figure_abstain_by_info_level_bar.py
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
SOURCE_CSV = REPO_ROOT / "analysis/output/tier_2/a8_info_hierarchy/abstain_by_info_level.csv"

figure_width = 12.0
figure_height = 4.2

# Three info levels — encode as a denim → sage → coral progression so
# the visual order tracks the information-richness axis.
INFO_PALETTE = {
    "Iq":       PALETTES["environment"]["Act"],     # denim (least info)
    "Iq+TL":    PALETTES["environment"]["Paired"],  # sage  (more info)
    "Iq+TL+o":  PALETTES["environment"]["Abstain"], # coral (full runtime info)
}
INFO_LABELS = {
    "Iq":       "Iq (query alone)",
    "Iq+TL":    "Iq+TL (+ tool list)",
    "Iq+TL+o":  "Iq+TL+o (+ tool outputs)",
}
INFO_ORDER = ["Iq", "Iq+TL", "Iq+TL+o"]

LEADERBOARD_ORDER = [
    "Gemini 3.1 Pro", "Claude Opus 4.7", "Claude Sonnet 4.6", "GPT-5.5",
    "Claude Haiku 4.5", "GPT-5", "GPT-5.4", "GLM-5", "GPT-OSS 120B",
    "GPT-5.2", "MiniMax M2.5", "DeepSeek V3.2", "GPT-5.1",
    "Gemini 3 Flash", "DeepSeek V4 Pro", "Kimi K2.5", "GPT-4o",
]


def main() -> None:
    df = pd.read_csv(SOURCE_CSV).set_index("Model")

    model_order = [m for m in LEADERBOARD_ORDER if m in df.index]
    n = len(model_order)
    df = df.reindex(model_order)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=300)

    bar_w = 0.78 / len(INFO_ORDER)
    x_pos = np.arange(n)

    for i, info in enumerate(INFO_ORDER):
        values = df[info].values
        offsets = x_pos - 0.78 / 2 + (i + 0.5) * bar_w
        ax.bar(offsets, values, width=bar_w,
               color=INFO_PALETTE[info], edgecolor="white",
               linewidth=0.4, label=INFO_LABELS[info], zorder=3)
        # Cross-model reference line.
        info_mean = df[info].mean()
        ax.axhline(info_mean, color=INFO_PALETTE[info],
                   linestyle="--", linewidth=0.8, alpha=0.6, zorder=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_order, fontsize=9, rotation=30,
                       ha="right", rotation_mode="anchor", color="#444444")
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 1.05)
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

    ax.legend(loc="upper right", fontsize=9, frameon=False,
              ncol=3, handlelength=1.4, handletextpad=0.5,
              columnspacing=1.4)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "abstain_by_info_level_bar.pdf"
    png_path = OUTPUT_DIR / "abstain_by_info_level_bar.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    print(f"models plotted: {n}")


if __name__ == "__main__":
    main()
