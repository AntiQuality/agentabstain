"""Restyle the appendix provider_radar figure.

Provider 'personality' radar across the four primary metrics
(Act, Abstain, Paired, CAR). Each provider is a polygon whose vertices
are the mean across models in that family. Reads
``analysis/output/tier_2/a12_provider/provider_summary.csv``.

Usage:
    python eval/statistics/appendix_analysis/figure_provider_radar.py
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

figure_width = 6.5
figure_height = 6.0

PROVIDER_ORDER = ["Anthropic", "OpenAI", "Google", "Open-weight"]
PROVIDER_COLORS = {
    "Anthropic":   HARNESS_COLORS["Claude SDK"],
    "OpenAI":      HARNESS_COLORS["OpenAI SDK"],
    "Google":      HARNESS_COLORS["Google ADK"],
    "Open-weight": HARNESS_COLORS["OpenClaw"],
}
METRIC_ORDER = ["Act", "Abstain", "Paired", "CAR"]


def main() -> None:
    df = pd.read_csv(SOURCE_CSV).set_index("Provider")

    n = len(METRIC_ORDER)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(figure_width, figure_height),
                           subplot_kw=dict(polar=True), dpi=300)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles)
    ax.set_xticklabels(METRIC_ORDER, fontsize=11, color="#333333")
    ax.tick_params(axis="x", pad=10)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"],
                       fontsize=8, color="#888888")
    ax.set_rlabel_position(45)

    ax.grid(color="#cccccc", alpha=0.6, linewidth=0.7)
    ax.spines["polar"].set_color("#bbbbbb")
    ax.spines["polar"].set_linewidth(0.8)

    for prov in PROVIDER_ORDER:
        values = [df.loc[prov, f"{m} mean"] for m in METRIC_ORDER]
        values_closed = values + [values[0]]
        n_models = int(df.loc[prov, "n_models"])
        color = PROVIDER_COLORS[prov]
        ax.plot(angles_closed, values_closed, color=color,
                linewidth=2.0, marker="o", markersize=5,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.8,
                label=f"{prov} (n={n_models})", zorder=3)
        ax.fill(angles_closed, values_closed, color=color,
                alpha=0.10, zorder=2)

    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.10),
              fontsize=9, frameon=False,
              handlelength=1.4, handletextpad=0.5)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "provider_radar.pdf"
    png_path = OUTPUT_DIR / "provider_radar.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    for prov in PROVIDER_ORDER:
        vals = [df.loc[prov, f'{m} mean'] for m in METRIC_ORDER]
        print(f"  {prov:12s} " +
              "  ".join(f"{m}={v:.3f}" for m, v in zip(METRIC_ORDER, vals)))


if __name__ == "__main__":
    main()
