"""Restyle the appendix provider_grouped_bar figure.

Cross-provider comparison across the four primary metrics. Bars =
mean across models within provider; error bars = std across models
within provider. Reads
``analysis/output/tier_2/a12_provider/provider_summary.csv``.

Usage:
    python eval/statistics/appendix_analysis/figure_provider_grouped_bar.py
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
SOURCE_CSV = REPO_ROOT / "analysis/output/tier_2/a12_provider/provider_summary.csv"

figure_width = 10.5
figure_height = 4.6

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
    df = df.set_index("Provider")

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=300)

    n_metrics = len(METRIC_ORDER)
    n_providers = len(PROVIDER_ORDER)
    x_pos = np.arange(n_metrics)
    bar_w = 0.84 / n_providers

    for i, prov in enumerate(PROVIDER_ORDER):
        means = [df.loc[prov, f"{m} mean"] for m in METRIC_ORDER]
        stds = [df.loc[prov, f"{m} std"] for m in METRIC_ORDER]
        n_models = int(df.loc[prov, "n_models"])
        offsets = x_pos - 0.84 / 2 + (i + 0.5) * bar_w
        ax.bar(offsets, means, width=bar_w, yerr=stds, capsize=3,
               color=PROVIDER_COLORS[prov], edgecolor="white",
               linewidth=0.6,
               label=f"{prov} (n={n_models})",
               error_kw=dict(ecolor="#555555", elinewidth=0.9, capthick=0.9),
               zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(METRIC_ORDER, fontsize=11, color="#333333")
    ax.set_xlim(-0.55, n_metrics - 0.45)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Mean across models in family ($\\pm$ std)",
                  fontsize=11, color="#333333")

    ax.yaxis.grid(True, alpha=0.25, zorder=0, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["left"].set_color("#999999")
    ax.tick_params(axis="y", colors="#666666", labelsize=9)
    ax.tick_params(axis="x", length=0)

    ax.legend(loc="upper right", fontsize=10, frameon=False, ncol=2,
              handlelength=1.4, handletextpad=0.5, columnspacing=1.4)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "provider_grouped_bar.pdf"
    png_path = OUTPUT_DIR / "provider_grouped_bar.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    for prov in PROVIDER_ORDER:
        print(f"  {prov:12s} n={int(df.loc[prov,'n_models'])}  "
              f"Act={df.loc[prov,'Act mean']:.3f}  "
              f"Abstain={df.loc[prov,'Abstain mean']:.3f}  "
              f"Paired={df.loc[prov,'Paired mean']:.3f}  "
              f"CAR={df.loc[prov,'CAR mean']:.3f}")


if __name__ == "__main__":
    main()
