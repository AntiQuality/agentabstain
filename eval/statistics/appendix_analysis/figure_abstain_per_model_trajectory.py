"""Restyle the appendix abstain_per_model_trajectory figure.

Per-model line trajectory of Abstain Accuracy across 3 information
levels (Iq -> Iq+TL -> Iq+TL+o). Reads
``analysis/output/tier_2/a8_info_hierarchy/info_level_long.csv``.

Lines colored with the lifted per-model MODEL_PALETTE shades from
``figure_leaderboard_bar.py`` (within-family gradient).

Usage:
    python eval/statistics/appendix_analysis/figure_abstain_per_model_trajectory.py
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

from eval.statistics.figure_leaderboard_bar import MODEL_CONFIG, MODEL_PALETTE

plt.rcParams["font.family"] = "Lato"

OUTPUT_DIR = SCRIPT_DIR / "figures"
SOURCE_CSV = REPO_ROOT / "analysis/output/tier_2/a8_info_hierarchy/info_level_long.csv"

figure_width = 9.0
figure_height = 5.5

INFO_ORDER = ["Iq", "Iq+TL", "Iq+TL+o"]
INFO_DISPLAY = {
    "Iq": "Iq\n(query alone)",
    "Iq+TL": "Iq+TL\n(+ tool list)",
    "Iq+TL+o": "Iq+TL+o\n(+ tool outputs)",
}

MODEL_ORDER_DISPLAY = [
    "Claude Haiku 4.5", "Claude Sonnet 4.6", "Claude Opus 4.7",
    "GPT-4o", "GPT-5", "GPT-5.1", "GPT-5.2", "GPT-5.4", "GPT-5.5",
    "Gemini 3 Flash", "Gemini 3.1 Pro",
    "MiniMax M2.5", "Kimi K2.5", "GLM-5",
    "DeepSeek V3.2", "DeepSeek V4 Pro", "GPT-OSS 120B",
]


def _short_name_for(display: str) -> str:
    if display.startswith("Claude "):
        return display[len("Claude "):]
    return display


def _model_color(model_display: str) -> str:
    short = _short_name_for(model_display)
    for key, (cfg_display, _harness, _logo) in MODEL_CONFIG.items():
        if cfg_display == short:
            return MODEL_PALETTE.get(key, "#888888")
    return "#888888"


def main() -> None:
    df = pd.read_csv(SOURCE_CSV)
    abstain = df[df["metric"] == "abstain_accuracy"].copy()

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=300)

    x_pos = np.arange(len(INFO_ORDER))

    for model in MODEL_ORDER_DISPLAY:
        sub = abstain[abstain["model_display"] == model].set_index("info_level")
        if sub.empty:
            continue
        values = [sub.loc[lvl, "value"] if lvl in sub.index else np.nan
                  for lvl in INFO_ORDER]
        color = _model_color(model)
        ax.plot(x_pos, values, marker="o", markersize=5, linewidth=1.5,
                color=color, alpha=0.9, label=model, zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([INFO_DISPLAY[lvl] for lvl in INFO_ORDER],
                       fontsize=10, color="#444444")
    ax.set_xlim(-0.3, len(INFO_ORDER) - 0.7)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Abstain Accuracy", fontsize=11, color="#333333")
    ax.set_xlabel("Information level", fontsize=11, color="#333333")

    ax.yaxis.grid(True, alpha=0.25, zorder=0, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["left"].set_color("#999999")
    ax.tick_params(axis="y", colors="#666666", labelsize=9)
    ax.tick_params(axis="x", length=0)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=8, frameon=False, handlelength=1.4, handletextpad=0.4,
              labelspacing=0.4)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "abstain_per_model_trajectory.pdf"
    png_path = OUTPUT_DIR / "abstain_per_model_trajectory.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    print(f"models plotted: {len(MODEL_ORDER_DISPLAY)}")


if __name__ == "__main__":
    main()
