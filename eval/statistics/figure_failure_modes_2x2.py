"""Restyle Figure 7 (failure_modes_2x2) for the main body.

Per-model breakdown of abstain-task failure modes, ordered by Paired
Accuracy (best -> worst). Source:
``analysis/output/tier_3/a37_abstain_2x2/abstain_2x2_failure_modes_long.csv``

Three failure modes (mapped to the lifted editorial palette):
  - Silent failure              -> gold     (CAR slot — "neither did nor refused")
  - Committed before refusing   -> denim    (Act slot   — partial action)
  - Didn't abstain at all       -> coral    (Abstain slot — full action without refusal)

Usage:
    python eval/statistics/figure_failure_modes_2x2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.statistics.figure_ranking_bar import PALETTES

plt.rcParams["font.family"] = "Lato"

OUTPUT_DIR = SCRIPT_DIR / "figures"
SOURCE_CSV = REPO_ROOT / (
    "analysis/output/tier_3/a37_abstain_2x2/abstain_2x2_failure_modes_long.csv"
)

figure_width = 11.0
figure_height = 3.6

MODE_PALETTE = {
    "Silent failure":             PALETTES["environment"]["CAR"],     # gold
    "Committed before refusing":  PALETTES["environment"]["Act"],     # denim
    "Didn't abstain at all":      PALETTES["environment"]["Abstain"], # coral
}
MODE_ORDER = ["Silent failure", "Committed before refusing", "Didn't abstain at all"]

LEADERBOARD_ORDER = [
    "Gemini 3.1 Pro", "Claude Opus 4.7", "Claude Sonnet 4.6", "GPT-5.5",
    "Claude Haiku 4.5", "GPT-5", "GPT-5.4", "GLM-5", "GPT-OSS 120B",
    "GPT-5.2", "MiniMax M2.5", "DeepSeek V3.2", "GPT-5.1",
    "Gemini 3 Flash", "DeepSeek V4 Pro", "Kimi K2.5", "GPT-4o",
]


def main() -> None:
    df = pd.read_csv(SOURCE_CSV)

    models_in_data = set(df["Model"].unique())
    model_order = [m for m in LEADERBOARD_ORDER if m in models_in_data]
    n = len(model_order)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=300)

    bar_w = 0.78 / len(MODE_ORDER)
    x_pos = np.arange(n)

    for i, mode in enumerate(MODE_ORDER):
        mode_sub = df[df["Failure mode"] == mode].set_index("Model")
        values = [mode_sub.loc[m, "value"] if m in mode_sub.index
                  else np.nan for m in model_order]
        offsets = x_pos - 0.78 / 2 + (i + 0.5) * bar_w
        ax.bar(offsets, values, width=bar_w,
               color=MODE_PALETTE[mode], edgecolor="white",
               linewidth=0.4, label=mode, zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_order, fontsize=10, rotation=30,
                       ha="right", rotation_mode="anchor", color="#444444")
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 0.45)
    ax.set_yticks(np.arange(0, 0.46, 0.1))
    ax.set_yticklabels([f"{int(round(v * 100))}" for v in np.arange(0, 0.46, 0.1)])
    ax.set_ylabel("Share of abstain runs (%)", fontsize=11, color="#333333")

    ax.yaxis.grid(True, alpha=0.25, zorder=0, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["left"].set_color("#999999")
    ax.tick_params(axis="y", colors="#666666", labelsize=9)
    ax.tick_params(axis="x", length=0)

    ax.legend(loc="upper left", fontsize=10, frameon=False,
              handlelength=1.4, handletextpad=0.5,
              ncol=3, columnspacing=1.6)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "failure_modes_2x2.pdf"
    png_path = OUTPUT_DIR / "failure_modes_2x2.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    print(f"models plotted: {n}")
    for m in model_order:
        sub = df[df["Model"] == m].set_index("Failure mode")
        vals = "  ".join(
            f"{mode}={sub.loc[mode, 'value']:.3f}" if mode in sub.index
            else f"{mode}=NA" for mode in MODE_ORDER
        )
        print(f"  {m:<20s}  {vals}")


if __name__ == "__main__":
    main()
