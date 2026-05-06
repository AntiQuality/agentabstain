"""Restyle the appendix inf_vs_op_paired_scatter figure.

Per-model scatter: x = Paired Accuracy on informational tasks, y =
Paired Accuracy on operational tasks. y=x diagonal as reference;
above-diagonal models do better on operational. Pearson r in title
area. Reads ``analysis/output/tier_2/a11_action_type/long.csv``.

Usage:
    python eval/statistics/appendix_analysis/figure_inf_vs_op_paired_scatter.py
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

from eval.statistics.figure_leaderboard_bar import HARNESS_COLORS, MODEL_CONFIG

plt.rcParams["font.family"] = "Lato"

OUTPUT_DIR = SCRIPT_DIR / "figures"
SOURCE_CSV = REPO_ROOT / "analysis/output/tier_2/a11_action_type/long.csv"

figure_width = 6.5
figure_height = 6.5


def _harness_for(model_display: str) -> str | None:
    for (_, _), (display, harness, _) in MODEL_CONFIG.items():
        if display == model_display or model_display.endswith(display):
            return harness
    return None


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
    df = df[(df["metric_variant"] == "combined") & (df["metric"] == "paired_acc")]
    pivot = df.pivot_table(index="Model", columns="action_type", values="value")

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=300)

    # Diagonal
    ax.plot([0, 1], [0, 1], color="#bbbbbb", linestyle="--",
            linewidth=1.0, alpha=0.7, zorder=1)

    LABEL_OFFSETS: dict[str, tuple[int, int, str, str]] = {
        # Right side defaults; place a few crowded points on the left.
        "Gemini 3.1 Pro":    (8, 4, "left", "bottom"),
        "Claude Opus 4.7":   (8, 4, "left", "bottom"),
        "Claude Sonnet 4.6": (8, -4, "left", "top"),
        "GPT-5.4":           (-8, 4, "right", "bottom"),
        "GPT-5.5":           (8, 4, "left", "bottom"),
        "GPT-5":             (-8, -4, "right", "top"),
        "Claude Haiku 4.5":  (-8, 4, "right", "bottom"),
        "GLM-5":             (8, 0, "left", "center"),
        "GPT-OSS 120B":      (-8, -4, "right", "top"),
        "GPT-5.2":           (-8, -4, "right", "top"),
        "MiniMax M2.5":      (8, -4, "left", "top"),
        "DeepSeek V3.2":     (-8, -4, "right", "top"),
        "GPT-5.1":           (-8, -4, "right", "top"),
        "Gemini 3 Flash":    (8, 4, "left", "bottom"),
        "DeepSeek V4 Pro":   (8, -4, "left", "top"),
        "Kimi K2.5":         (-8, 4, "right", "bottom"),
        "GPT-4o":            (8, 4, "left", "bottom"),
    }

    for model, row in pivot.iterrows():
        x_v, y_v = row["informational"], row["operational"]
        harness = _harness_for(model)
        color = HARNESS_COLORS.get(harness, "#888888") if harness else "#888888"
        ax.scatter(x_v, y_v, color=color, s=70, alpha=0.9,
                   edgecolor="white", linewidth=0.6, zorder=3)
        dx, dy, ha, va = LABEL_OFFSETS.get(model, (8, 4, "left", "bottom"))
        ax.annotate(model, (x_v, y_v), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8, color="#333333",
                    ha=ha, va=va)

    pr_r, pr_p = pearsonr(pivot["informational"], pivot["operational"])
    ax.text(0.02, 1.04,
            f"Above diagonal = better on operational    Pearson r = {pr_r:+.3f} {_sig_star(pr_p)}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color="#666666")

    ax.set_xlim(0.2, 0.7)
    ax.set_ylim(0.2, 0.7)
    ax.set_xlabel("Paired Accuracy on Informational tasks", fontsize=11, color="#333333")
    ax.set_ylabel("Paired Accuracy on Operational tasks", fontsize=11, color="#333333")
    ax.tick_params(axis="x", colors="#666666", labelsize=9)
    ax.tick_params(axis="y", colors="#666666", labelsize=9)

    ax.grid(True, alpha=0.2, zorder=0, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["left"].set_color("#999999")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", markersize=7,
                   markerfacecolor=HARNESS_COLORS[h], markeredgecolor="white",
                   markeredgewidth=0.6, label=h)
        for h in ("Claude SDK", "OpenAI SDK", "Google ADK", "OpenClaw")
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
              frameon=False, handlelength=1.4, handletextpad=0.5)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "inf_vs_op_paired_scatter.pdf"
    png_path = OUTPUT_DIR / "inf_vs_op_paired_scatter.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    print(f"models plotted: {len(pivot)}    Pearson r = {pr_r:+.4f} (p = {pr_p:.4f})")


if __name__ == "__main__":
    main()
