"""Generate the page-1 leaderboard bar chart.

Usage:
    python eval/statistics/figure_leaderboard_bar.py

Output:
    eval/statistics/figures/leaderboard_bar.pdf
    eval/statistics/figures/leaderboard_bar.png
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch
from PIL import Image
from scipy import ndimage

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

HARNESS_COLORS = {
    "Claude SDK": "#7A4F35",
    "OpenAI SDK": "#4E79A7",
    "Google ADK": "#D4A634",
    "OpenClaw": "#9476A8",
}

MODEL_PALETTE = {
    ("claudesdk", "claude-haiku-4-5"): "#EDD2AA",
    ("claudesdk", "claude-sonnet-4-6"): "#C9824A",
    ("claudesdk", "claude-opus-4-7"): "#6F3A1E",
    ("openaisdk", "gpt-4o-2024-08-06"): "#C6DBEF",
    ("openaisdk", "gpt-5-2025-08-07"): "#9ECAE1",
    ("openaisdk", "gpt-5.1-2025-11-13"): "#6BAED6",
    ("openaisdk", "gpt-5.2-2025-12-11"): "#4292C6",
    ("openaisdk", "gpt-5.4-2026-03-05"): "#2171B5",
    ("openaisdk", "gpt-5.5-2026-04-23"): "#08306B",
    ("googleadk", "gemini-3-flash-preview"): "#FFEAA7",
    ("googleadk", "gemini-3.1-pro-preview"): "#F1C40F",
    ("amazon-bedrock", "minimax-m2.5"): "#D62728",
    ("amazon-bedrock", "kimi-k2.5"): "#000000",
    ("amazon-bedrock", "zai.glm-5"): "#8E44AD",
    ("amazon-bedrock", "deepseek.v3.2"): "#22D3EE",
    ("openrouter", "deepseek-v4-pro"): "#0891B2",
    ("amazon-bedrock", "openai.gpt-oss-120b-1:0"): "#16A34A",
}

MODEL_CONFIG = {
    ("claudesdk", "claude-haiku-4-5"): ("Haiku 4.5", "Claude SDK", "anthropic.png"),
    ("claudesdk", "claude-sonnet-4-6"): ("Sonnet 4.6", "Claude SDK", "anthropic.png"),
    ("claudesdk", "claude-opus-4-7"): ("Opus 4.7", "Claude SDK", "anthropic.png"),
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


def remove_white_border(img_path: Path, threshold: int = 240, target_size: int = 128) -> np.ndarray:
    img = Image.open(img_path).convert("RGBA")
    data = np.array(img)
    r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    white_mask = (r > threshold) & (g > threshold) & (b > threshold)

    labeled, _ = ndimage.label(white_mask)
    border_labels = set()
    for edge in (labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]):
        border_labels.update(np.unique(edge).tolist())
    border_labels.discard(0)
    if border_labels:
        edge_white_mask = np.isin(labeled, list(border_labels))
        data[edge_white_mask, 3] = 0
    img = Image.fromarray(data)

    alpha = np.array(img)[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if rows.any() and cols.any():
        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]
        img = img.crop((x0, y0, x1 + 1, y1 + 1))

    w, h = img.size
    scale = target_size / max(w, h)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    img = img.resize(new_size, Image.LANCZOS)

    canvas = Image.new("RGBA", (target_size, target_size), (255, 255, 255, 0))
    px = (target_size - new_size[0]) // 2
    py = (target_size - new_size[1]) // 2
    canvas.paste(img, (px, py), img)
    return np.array(canvas)


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
                logo_imgs.append(remove_white_border(logo_path))
            else:
                logo_imgs.append(None)

    n = len(display_names)
    fig, ax = plt.subplots(figsize=(7, 3.0))

    x_pos = np.arange(n)
    bars = ax.bar(x_pos, values, width=0.6, color=bar_colors, edgecolor="white", linewidth=0.4, zorder=3)

    for i, val in enumerate(values):
        ax.text(i, val + 0.5, f"{val:.1f}", ha="center", va="bottom", fontsize=6, fontweight="bold", color="#333333")

    for i, logo_arr in enumerate(logo_imgs):
        if logo_arr is not None:
            imagebox = OffsetImage(logo_arr, zoom=0.085)
            ab = AnnotationBbox(imagebox, (i, 5.0), frameon=False, xycoords=("data", "data"), box_alignment=(0.5, 0.5))
            ax.add_artist(ab)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, fontsize=8, rotation=40, ha="right", rotation_mode="anchor", color="#444444")

    ax.set_ylim(0, 100)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylabel("Paired Accuracy (%)", fontsize=8.5, color="#333333")

    ax.yaxis.grid(True, alpha=0.2, zorder=0, color="#cccccc")
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["left"].set_color("#999999")
    ax.tick_params(axis="y", colors="#666666", labelsize=7)
    ax.tick_params(axis="x", length=0)

    legend_elements = [Patch(facecolor=c, edgecolor="none", label=h) for h, c in HARNESS_COLORS.items()]
    ax.legend(
        handles=legend_elements, loc="upper right", fontsize=7, frameon=False,
        ncol=4, handlelength=1.0, handletextpad=0.4,
        columnspacing=1.0, borderpad=0.3,
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "leaderboard_bar.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUTPUT_DIR / "leaderboard_bar.png", bbox_inches="tight", dpi=300)
    print(f"Saved to {OUTPUT_DIR}/leaderboard_bar.pdf and .png")


if __name__ == "__main__":
    main()
