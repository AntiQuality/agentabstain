"""Generate the per-model Act vs Abstain calibration scatter.

Each dot is one model at its (mean Act Accuracy, mean Abstain Accuracy)
across the 8 categories. Dot color = harness (matching the leaderboard
bar chart in the abstract, so a reader who has seen that figure already
knows the color coding). The diagonal y=x reference shows the
competence-restraint balance — points above the diagonal are the (rare)
restraint-leaning models.

Usage:
    python eval/statistics/figure_act_vs_abstain_centroid.py

Output:
    eval/statistics/figures/act_vs_abstain_model_centroid.pdf
    eval/statistics/figures/act_vs_abstain_model_centroid.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.statistics.analysis import (
    build_metric_run_frame,
    compute_summary_metrics,
    load_eval_frame,
    select_latest_runs,
)
from eval.statistics.figure_leaderboard_bar import (
    HARNESS_COLORS,
    MODEL_CONFIG,
    canon,
)

plt.rcParams["font.family"] = "Lato"

OUTPUT_DIR = SCRIPT_DIR / "figures"

# Half-column subfigure (sibling of the dumbbell in `fig:both-panels`).
# Aspect kept ~square so the y=x diagonal reads at 45°. A bit larger
# than \linewidth/2 so 8pt labels don't shrink to illegibility.
figure_width = 5.0
figure_height = 4.5


def main() -> None:
    df = load_eval_frame(REPO_ROOT / "results")
    df = select_latest_runs(df)
    df["model"] = df["model"].map(canon)
    df = df[df["task_id"].fillna("").str.startswith("preview")]
    metric_df = build_metric_run_frame(df)
    summary = compute_summary_metrics(metric_df)

    combined = summary[summary["metric_name"] == "combined"].copy()
    per_cat = (
        combined.groupby(["provider", "model", "category"])
        .agg(act=("should_act_accuracy", "mean"),
             abstain=("should_abstain_accuracy", "mean"))
        .reset_index()
    )
    macro = (
        per_cat.groupby(["provider", "model"])[["act", "abstain"]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=300)

    # Diagonal + 0.5 quadrant guides
    ax.plot([0.3, 1.0], [0.3, 1.0], color="#bbbbbb", linestyle="--",
            linewidth=1.0, alpha=0.7, zorder=1, label="Act = Abstain")

    # One dot per model, colored by harness.
    plotted: list[tuple[str, str, float, float]] = []
    seen_harnesses: list[str] = []
    for _, row in macro.iterrows():
        key = (row["provider"], row["model"])
        if key not in MODEL_CONFIG:
            continue
        name, harness, _logo = MODEL_CONFIG[key]
        color = HARNESS_COLORS[harness]
        ax.scatter(row["act"], row["abstain"], color=color, s=70,
                   edgecolor="white", linewidth=0.8, zorder=3)
        plotted.append((name, harness, float(row["act"]), float(row["abstain"])))
        if harness not in seen_harnesses:
            seen_harnesses.append(harness)

    # Manual label placement — adjustText's auto-layout produced messy
    # connector lines through the central GPT cluster. Direction codes
    # below were chosen by neighbour analysis on actual dot coordinates
    # (act, abstain) so each label sits adjacent to its dot with no
    # connector line at all.
    LABEL_OFFSETS: dict[str, tuple[int, int, str, str]] = {
        # name -> (dx_pts, dy_pts, ha, va)
        # Top region (y > 0.65) — ample room
        "Opus 4.7":          ( -5,  0, "right",   "center"),
        # Sonnet RIGHT (was ABOVE) so it doesn't crowd GPT-5.4
        "Sonnet 4.6":        ( 5,  0, "left",   "bottom"),
        "Haiku 4.5":         ( 5,  0, "left",  "top"),
        "GPT-5":             ( 5,  0, "left",  "center"),
        # GPT-5.4 BELOW-LEFT to clear the row above it where Sonnet
        # now lives and the row below where Haiku sits to its right
        "GPT-5.4":           (-5, 0, "right",  "top"),
        "Gemini 3.1 Pro":    ( 5,  0, "left",   "center"),
        # Middle band (y 0.55-0.65) — densest cluster
        "GPT-5.2":           (-5,  0, "right",  "center"),
        "GPT-5.1":           (-5,  0, "right",  "center"),
        "GPT-OSS 120B":      (  5, 0, "left", "top"),
        # GLM-5 BELOW (was ABOVE) so it doesn't share a baseline with
        # Haiku 4.5's left label; lands clear of GPT-OSS below-left.
        "GLM-5":             (-5, 0, "right",   "center"),
        "GPT-5.5":           ( 5,  0, "left",   "center"),
        # Lower-middle (y 0.50-0.55)
        "Kimi K2.5":         ( 5,  0, "left",   "center"),
        "DeepSeek V3.2":     ( 5,  0, "left",   "center"),
        "MiniMax M2.5":      ( 5, 0, "left", "center"),
        # Bottom (y < 0.45) — three close points, fan them out
        "GPT-4o":            (-5,  0, "right",  "center"),
        "DeepSeek V4 Pro":   (-5, 0, "right", "top"),
        "Gemini 3 Flash":    (  5,  0, "left",   "bottom"),
    }
    for name, _harness, ax_x, ax_y in plotted:
        dx, dy, ha, va = LABEL_OFFSETS.get(name, (8, 4, "left", "bottom"))
        ax.annotate(name, (ax_x, ax_y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=9, color="#333333",
                    ha=ha, va=va)

    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.35, 0.85)
    ax.set_xlabel("Act Accuracy", fontsize=13, color="#333333")
    ax.set_ylabel("Abstain Accuracy", fontsize=13, color="#333333")
    ax.tick_params(axis="x", colors="#666666", labelsize=9)
    ax.tick_params(axis="y", colors="#666666", labelsize=9)

    ax.grid(True, alpha=0.2, zorder=0, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["left"].set_color("#999999")

    # Harness legend in the same order as the leaderboard bar (to keep
    # color → harness mapping consistent across the paper).
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", markersize=7,
                   markerfacecolor=HARNESS_COLORS[h], markeredgecolor="white",
                   markeredgewidth=0.8, label=h)
        for h in ("Claude SDK", "OpenAI SDK", "Google ADK", "OpenClaw")
    ]
    legend_handles.append(
        plt.Line2D([0], [0], color="#bbbbbb", linestyle="--", linewidth=1.0,
                   label="Act = Abstain")
    )
    ax.legend(
        handles=legend_handles, loc="upper left", fontsize=10, frameon=False,
        handlelength=1.4, handletextpad=0.5, borderpad=0.2,
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "act_vs_abstain_model_centroid.pdf"
    png_path = OUTPUT_DIR / "act_vs_abstain_model_centroid.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    print(f"models plotted: {len(plotted)}")


if __name__ == "__main__":
    main()
