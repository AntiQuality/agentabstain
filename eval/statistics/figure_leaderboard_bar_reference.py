"""Generate the page-1 leaderboard bar chart (reference version).

This is the version that produced the latest leaderboard_bar.pdf.
To regenerate: python eval/statistics/figure_leaderboard_bar_reference.py

Output:
    eval/statistics/figures/leaderboard_bar.pdf
    eval/statistics/figures/leaderboard_bar.png
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch
from PIL import Image

plt.rcParams["font.family"] = "Lato"

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

_CLAUDE_RE = re.compile(r"claude-(?:opus|sonnet|haiku)-\d+-\d+")


def canon(name: str) -> str:
    if not isinstance(name, str):
        return name
    m = _CLAUDE_RE.search(name)
    if m:
        return m.group(0)
    for prefix in ("moonshotai.", "minimax.", "deepseek/"):
        if name.startswith(prefix):
            return name.split(prefix[-1], 1)[1]
    return name


LOGO_DIR = SCRIPT_DIR / "logos"
OUTPUT_DIR = SCRIPT_DIR / "figures"

# Dark, desaturated academic palette (per harness)
HARNESS_COLORS = {
    "Claude SDK": "#8B6B4A",
    "OpenAI SDK": "#4A6A8B",
    "Google ADK": "#8B7B3A",
    "OpenClaw": "#6B4A5A",
}

MODEL_CONFIG = {
    ("claudesdk", "claude-haiku-4-5"): ("Claude Haiku 4.5", "Claude SDK", "anthropic.png"),
    ("claudesdk", "claude-sonnet-4-6"): ("Claude Sonnet 4.6", "Claude SDK", "anthropic.png"),
    ("claudesdk", "claude-opus-4-7"): ("Claude Opus 4.7", "Claude SDK", "anthropic.png"),
    ("openaisdk", "gpt-4o-2024-08-06"): ("GPT-4o", "OpenAI SDK", "openai.png"),
    ("openaisdk", "gpt-5-2025-08-07"): ("GPT-5", "OpenAI SDK", "openai.png"),
    ("openaisdk", "gpt-5.1-2025-11-13"): ("GPT-5.1", "OpenAI SDK", "openai.png"),
    ("openaisdk", "gpt-5.2-2025-12-11"): ("GPT-5.2", "OpenAI SDK", "openai.png"),
    ("openaisdk", "gpt-5.4-2026-03-05"): ("GPT-5.4", "OpenAI SDK", "openai.png"),
    ("openaisdk", "gpt-5.5-2026-04-23"): ("GPT-5.5", "OpenAI SDK", "openai.png"),
    ("googleadk", "gemini-3-flash-preview"): ("Gemini 3 Flash", "Google ADK", "google.png"),
    ("googleadk", "gemini-3.1-pro-preview"): ("Gemini 3.1 Pro", "Google ADK", "google.png"),
    ("amazon-bedrock", "minimax-m2.5"): ("MiniMax M2.5", "OpenClaw", "minimax.png"),
    ("amazon-bedrock", "kimi-k2.5"): ("Kimi K2.5", "OpenClaw", "moonshot.png"),
    ("amazon-bedrock", "zai.glm-5"): ("GLM-5", "OpenClaw", "zhipu.png"),
    ("amazon-bedrock", "deepseek.v3.2"): ("DeepSeek V3.2", "OpenClaw", "deepseek.png"),
    ("openrouter", "deepseek-v4-pro"): ("DeepSeek V4 Pro", "OpenClaw", "deepseek.png"),
    ("amazon-bedrock", "openai.gpt-oss-120b-1:0"): ("GPT-OSS 120B", "OpenClaw", "gptoss.png"),
}


def load_logo(img_path: Path, target_size: int = 80) -> np.ndarray:
    """Load logo, remove white background, crop to content, resize to uniform square."""
    img = Image.open(img_path).convert("RGBA")
    data = np.array(img)
    r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    white_mask = (r > 235) & (g > 235) & (b > 235)
    data[white_mask, 3] = 0
    img = Image.fromarray(data)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    img = img.resize((target_size, target_size), Image.LANCZOS)
    return np.array(img)


def main() -> None:
    df = load_eval_frame(REPO_ROOT / "results")
    df = select_latest_runs(df)
    df["model"] = df["model"].map(canon)
    df = df[df["task_id"].fillna("").str.startswith("preview")]
    metric_df = build_metric_run_frame(df)
    summary = compute_summary_metrics(metric_df)

    combined = summary[summary["metric_name"] == "combined"].copy()
    per_cat = combined.groupby(["provider", "model", "category"])["paired_accuracy"].mean().reset_index()
    macro = per_cat.groupby(["provider", "model"])["paired_accuracy"].mean().reset_index()
    macro.columns = ["provider", "model", "paired_accuracy"]
    macro = macro.sort_values("paired_accuracy", ascending=False).reset_index(drop=True)

    display_names, values, bar_colors, logo_imgs = [], [], [], []

    for _, row in macro.iterrows():
        key = (row["provider"], row["model"])
        if key in MODEL_CONFIG:
            name, harness, logo_file = MODEL_CONFIG[key]
            display_names.append(name)
            values.append(row["paired_accuracy"] * 100)
            bar_colors.append(HARNESS_COLORS[harness])
            logo_path = LOGO_DIR / logo_file
            if logo_path.exists():
                logo_imgs.append(load_logo(logo_path, target_size=80))
            else:
                logo_imgs.append(None)

    n = len(display_names)
    fig, ax = plt.subplots(figsize=(8, 3.0))

    x_pos = np.arange(n)
    bars = ax.bar(x_pos, values, width=0.78, color=bar_colors, edgecolor="white", linewidth=0.4, zorder=3)

    for i, val in enumerate(values):
        ax.text(i, val + 0.8, f"{val:.1f}", ha="center", va="bottom",
                fontsize=5.5, fontweight="bold", color="#333333")

    for i, logo_arr in enumerate(logo_imgs):
        if logo_arr is not None:
            imagebox = OffsetImage(logo_arr, zoom=0.13)
            ab = AnnotationBbox(imagebox, (i, -7), frameon=False,
                                xycoords=("data", "data"), box_alignment=(0.5, 0.5),
                                clip_on=False)
            ax.add_artist(ab)

    for i, name in enumerate(display_names):
        ax.text(i, -14, name, ha="center", va="top", fontsize=4.5,
                rotation=0, color="#444444", linespacing=0.9)

    ax.set_xticks([])
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylabel("Paired Accuracy (%)", fontsize=7.5, color="#333333")

    ax.yaxis.grid(True, alpha=0.2, zorder=0, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color("#999999")
    ax.tick_params(axis="y", colors="#666666", labelsize=6.5)

    legend_elements = [Patch(facecolor=c, edgecolor="#cccccc", label=h) for h, c in HARNESS_COLORS.items()]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=5.5, frameon=False,
              ncol=4, handlelength=1.0, handletextpad=0.3, columnspacing=0.8)

    plt.subplots_adjust(bottom=0.28)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "leaderboard_bar.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUTPUT_DIR / "leaderboard_bar.png", bbox_inches="tight", dpi=300)
    print(f"Saved to {OUTPUT_DIR}/leaderboard_bar.pdf and .png")


if __name__ == "__main__":
    main()
