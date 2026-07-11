"""Generate the per-category difficulty bar chart.

Tall, narrow horizontal-bar layout sized for a LaTeX ``wrapfigure``
(text in Finding 3 wraps around it, so it consumes essentially no
column-flow vertical space). Bars are ranked easiest (top) to hardest
(bottom) by mean Paired Accuracy across 17 models. Bar color encodes
deficiency type (Finding 3's four-bucket grouping), using the lifted
``environment`` palette. Diagonal hatching marks runtime categories;
solid fill marks pre-execution.

Usage:
    python eval/statistics/figure_category_difficulty.py

Output:
    eval/statistics/figures/category_difficulty_horizontal_companion.pdf
    eval/statistics/figures/category_difficulty_horizontal_companion.png
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
A30_SUMMARY = REPO_ROOT / "analysis/output/tier_3/a30_category_difficulty/category_difficulty_summary.csv"

figure_width = 5
figure_height = 3

# Map raw category id -> (deficiency type, scenario number for label).
# Deficiency types from Finding 3's taxonomy.
CATEGORY_META: dict[str, tuple[str, str]] = {
    "missing_critical_parameter":     ("Informational Gap",         "S1"),
    "ambiguous_action_specification": ("Informational Gap",         "S2"),
    "conflicting_constraints":        ("Logical Contradiction",     "S3"),
    "high_stakes_action":             ("Disproportionate Consequence", "S4"),
    "insufficient_tool_capability":   ("Capability Limitation",     "S5"),
    "critical_tool_failure":          ("Capability Limitation",     "S6"),
    "conflicting_evidence":           ("Informational Gap",         "S7"),
    "emergent_risk_discovery":        ("Disproportionate Consequence", "S8"),
}

# Lifted-env palette mapped to deficiency types by "alarm level":
# Informational Gap (about info)         -> denim   (cool, neutral)
# Capability Limitation (about tools)    -> sage    (cool, neutral)
# Logical Contradiction (about reasoning) -> gold   (warm, attention)
# Disproportionate Consequence (stakes)  -> coral   (warm, alert)
DEFICIENCY_COLORS: dict[str, str] = {
    "Informational Gap":            PALETTES["environment"]["Act"],      # denim
    "Capability Limitation":        PALETTES["environment"]["Paired"],   # sage
    "Logical Contradiction":        PALETTES["environment"]["CAR"],      # gold
    "Disproportionate Consequence": PALETTES["environment"]["Abstain"],  # coral
}
DEFICIENCY_ORDER = (
    "Informational Gap",
    "Capability Limitation",
    "Logical Contradiction",
    "Disproportionate Consequence",
)


def main() -> None:
    df = pd.read_csv(A30_SUMMARY)
    df["deficiency"] = df["category"].map(lambda c: CATEGORY_META[c][0])
    df["scenario_id"] = df["category"].map(lambda c: CATEGORY_META[c][1])
    df["display"] = df["scenario_id"] + "  " + df["Category"]

    # Sort hardest at bottom, easiest at top — matches conventional
    # "ranking lists" reading order top-down (easy to hard).
    df = df.sort_values("Mean", ascending=True).reset_index(drop=True)

    n = len(df)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=300)

    bar_colors = [DEFICIENCY_COLORS[d] for d in df["deficiency"]]
    err_low = df["Mean"] - df["Min"]
    err_high = df["Max"] - df["Mean"]

    # Phase encoded by hatch pattern on the bar fill (pre-execution =
    # solid, runtime = diagonal hatch). Color and hatch live on the bar
    # itself so readers don't need to scan a separate glyph column.
    hatches = ["" if p == "pre_execution" else "///" for p in df["phase"]]
    bars = ax.barh(
        y_pos, df["Mean"],
        height=0.5, color=bar_colors,
        edgecolor="#444444", linewidth=0.4,
        hatch=hatches, zorder=3,
    )
    ax.errorbar(
        df["Mean"], y_pos,
        xerr=[err_low, err_high],
        fmt="none", ecolor="#666666", elinewidth=0.8,
        capsize=3, capthick=0.8, zorder=4,
    )
    # Numeric mean to the right of each bar, just past the upper
    # error-bar cap.
    for y, mean, hi in zip(y_pos, df["Mean"], df["Max"]):
        ax.text(hi + 0.025, y, f"{mean:.2f}",
                ha="left", va="center", fontsize=8.5,
                color="#333333", fontweight="bold")

    # ascending sort + default yaxis: easiest mean at top, hardest at
    # bottom — matches the original layout's orientation.
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display"], fontsize=11, color="#444444")

    ax.set_xlim(0, 1.18)  # extra room for the numeric labels
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Mean Paired Accuracy", fontsize=13, color="#333333")
    ax.set_ylim(-0.6, n - 0.4)

    ax.xaxis.grid(True, alpha=0.25, zorder=0, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["left"].set_color("#999999")
    ax.tick_params(axis="x", colors="#666666", labelsize=8.5)
    ax.tick_params(axis="y", length=0)

    # Two-part legend: deficiency colors + phase encoded as solid vs.
    # hatched fill. Use a neutral light gray for the phase swatches so
    # they don't suggest a deficiency type.
    PHASE_SWATCH = "#cccccc"
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=DEFICIENCY_COLORS[d],
                      edgecolor="#444444", linewidth=0.4, label=d)
        for d in DEFICIENCY_ORDER
    ] + [
        plt.Rectangle((0, 0), 1, 1, facecolor=PHASE_SWATCH,
                      edgecolor="#444444", linewidth=0.4,
                      label="Pre-execution"),
        plt.Rectangle((0, 0), 1, 1, facecolor=PHASE_SWATCH,
                      edgecolor="#444444", linewidth=0.4,
                      hatch="///", label="Runtime"),
    ]
    # Legend above the plot (column-wide layout fits the narrow figure).
    ax.legend(
        handles=legend_handles,
        loc="lower center", bbox_to_anchor=(0.3, 1.02),
        fontsize=10, frameon=False,
        ncol=3,
        handlelength=1.0, handleheight=0.9, handletextpad=0.5,
        labelspacing=0.4, columnspacing=1.0, borderpad=0.1,
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "category_difficulty_horizontal_companion.pdf"
    png_path = OUTPUT_DIR / "category_difficulty_horizontal_companion.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")
    print(f"categories plotted: {n} (sorted easy→hard by Mean Paired Accuracy)")
    for _, r in df.iterrows():
        print(f"  {r['scenario_id']} {r['Category']:<22s}  "
              f"deficiency={r['deficiency']:<32s}  phase={r['phase']:<14s}  "
              f"mean={r['Mean']:.3f}  range=[{r['Min']:.3f}, {r['Max']:.3f}]")


if __name__ == "__main__":
    main()
