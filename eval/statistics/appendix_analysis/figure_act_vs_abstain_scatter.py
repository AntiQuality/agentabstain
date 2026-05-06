"""Restyle the appendix act_vs_abstain_scatter figure.

Single-panel calibration scatter: 17 models × 8 categories = 136
points, color = model (per-model lifted shades from
``figure_leaderboard_bar.MODEL_PALETTE``), marker = category. y=x
diagonal + 0.5 quadrant guides + corner labels for each quadrant.

Reads ``analysis/output/tier_1/a3_calibration_scatter/scatter_data.csv``.

Usage:
    python eval/statistics/appendix_analysis/figure_act_vs_abstain_scatter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.statistics.figure_leaderboard_bar import MODEL_CONFIG, MODEL_PALETTE

plt.rcParams["font.family"] = "Lato"

OUTPUT_DIR = SCRIPT_DIR / "figures"
SOURCE_CSV = REPO_ROOT / "analysis/output/tier_1/a3_calibration_scatter/scatter_data.csv"

figure_width = 10.0
figure_height = 7.5

# Marker shape per category. 8 visually distinct shapes.
CATEGORY_MARKERS = {
    "Missing Param":      "*",
    "Ambiguous Action":   "o",
    "Conflicting Constr.": "s",
    "Conflicting Evid.":   "^",
    "Tool Failure":        "D",
    "Emergent Risk":       "v",
    "High Stakes":         "P",
    "Insuff. Tools":       "X",
}

# Display order in legends.
MODEL_ORDER_DISPLAY = [
    "Claude Haiku 4.5", "Claude Sonnet 4.6", "Claude Opus 4.7",
    "GPT-4o", "GPT-5", "GPT-5.1", "GPT-5.2", "GPT-5.4", "GPT-5.5",
    "Gemini 3 Flash", "Gemini 3.1 Pro",
    "MiniMax M2.5", "Kimi K2.5", "GLM-5",
    "DeepSeek V3.2", "DeepSeek V4 Pro", "GPT-OSS 120B",
]


def _short_name_for(display: str) -> str:
    """Strip the leading family prefix Claude/GPT-X gives MODEL_CONFIG keys."""
    if display.startswith("Claude "):
        return display[len("Claude "):]
    return display


def _model_color(model_display: str) -> str:
    short = _short_name_for(model_display)
    for key, (cfg_display, _harness, _logo) in MODEL_CONFIG.items():
        if cfg_display == short:
            return MODEL_PALETTE.get(key, "#888888")
    return "#888888"


def _sig_star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def main() -> None:
    df = pd.read_csv(SOURCE_CSV)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=300)

    for _, row in df.iterrows():
        marker = CATEGORY_MARKERS.get(row["category_display"], "o")
        color = _model_color(row["model_display"])
        ax.scatter(row["act"], row["abstain"], color=color,
                   marker=marker, s=70, alpha=0.85,
                   edgecolor="white", linewidth=0.5, zorder=3)

    # Diagonal + 0.5 quadrant guides
    ax.plot([0, 1], [0, 1], color="#bbbbbb", linestyle="--",
            linewidth=1.0, alpha=0.7, zorder=1)
    ax.axhline(0.5, color="#dddddd", linestyle=":",
               linewidth=0.7, alpha=0.7, zorder=0)
    ax.axvline(0.5, color="#dddddd", linestyle=":",
               linewidth=0.7, alpha=0.7, zorder=0)

    # Corner annotations for the four quadrants.
    quad_text = {
        (0.97, 0.97, "right", "top"):    "Well-calibrated",
        (0.97, 0.03, "right", "bottom"): "Bias toward acting",
        (0.03, 0.97, "left",  "top"):    "Bias toward refusing",
        (0.03, 0.03, "left",  "bottom"): "Both fail",
    }
    for (x, y, ha, va), txt in quad_text.items():
        ax.text(x, y, txt, transform=ax.transAxes, ha=ha, va=va,
                fontsize=9, color="#666666", style="italic",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.85})

    overall_r, overall_p = pearsonr(df["act"], df["abstain"])
    ax.text(0.02, 1.04,
            f"{len(df)} (model × category) points    "
            f"Pearson r = {overall_r:+.3f} {_sig_star(overall_p)}    "
            f"(negative r = act/abstain trade-off)",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color="#666666")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Act Accuracy", fontsize=11, color="#333333")
    ax.set_ylabel("Abstain Accuracy", fontsize=11, color="#333333")
    ax.tick_params(axis="x", colors="#666666", labelsize=9)
    ax.tick_params(axis="y", colors="#666666", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["left"].set_color("#999999")

    # Two legends: color (model) and marker (category).
    color_handles = [
        plt.Line2D([], [], marker="o", color="w",
                   markerfacecolor=_model_color(m), markeredgecolor="white",
                   markeredgewidth=0.5, markersize=8, label=m)
        for m in MODEL_ORDER_DISPLAY
    ]
    marker_handles = [
        plt.Line2D([], [], marker=mk, color="#444444", linestyle="",
                   markersize=8, markeredgewidth=0.6, label=cat)
        for cat, mk in CATEGORY_MARKERS.items()
    ]
    leg1 = ax.legend(handles=color_handles, title="Model (color)",
                     loc="upper left", bbox_to_anchor=(1.02, 1.0),
                     fontsize=8, title_fontsize=9, frameon=False,
                     handletextpad=0.4)
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, title="Category (marker)",
              loc="upper left", bbox_to_anchor=(1.02, 0.45),
              fontsize=8, title_fontsize=9, frameon=False,
              handletextpad=0.4)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "act_vs_abstain_scatter.pdf"
    png_path = OUTPUT_DIR / "act_vs_abstain_scatter.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    print(f"points plotted: {len(df)}    overall Pearson r = {overall_r:+.4f} (p = {overall_p:.4f})")


if __name__ == "__main__":
    main()
