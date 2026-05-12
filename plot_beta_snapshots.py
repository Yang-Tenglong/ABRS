from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


OUTPUT_PATH = Path(__file__).resolve().parent / "beta_change.pdf"

AXIS_LABEL_FONTSIZE = 21  # 坐标轴标签字号，例如 x 和 p(x)
TICK_LABEL_FONTSIZE = 13  # 横坐标刻度数字字号
TITLE_FONTSIZE = 20  # 子图标题字号，例如 alpha 和 beta 参数
ANNOTATION_FONTSIZE = 18  # 图中注释字号，例如 mu +/- sigma
MEAN_LABEL_FONTSIZE = 23  # 均值标记 mu 的字号

GOOD_STATE_BETAS = [
    (1.1, 8.0),
    (13.1, 7.6),
    (30.90, 15.60),
    (194.4, 50.10),
]

BAD_STATE_BETAS = [
    (1.6, 5.0),
    (3.3, 22.0),
    (7.98, 27.0),
    (8.0, 103.0),
]


def beta_mean_variance(alpha: float, beta: float) -> tuple[float, float]:
    denom = alpha + beta
    mean = alpha / denom
    variance = (alpha * beta) / (denom * denom * (denom + 1.0))
    return mean, variance


def beta_pdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    x = np.clip(x, 1e-12, 1.0 - 1e-12)
    log_norm = math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
    log_pdf = log_norm + (alpha - 1.0) * np.log(x) + (beta - 1.0) * np.log1p(-x)
    return np.exp(np.clip(log_pdf, -745.0, 700.0))


def beta_x_grid(mean: float, variance: float, points: int = 8000) -> np.ndarray:
    full_grid = np.linspace(0.0, 1.0, points)
    std = math.sqrt(max(variance, 0.0))
    local_width = max(8.0 * std, 1e-4)
    local_grid = np.linspace(max(0.0, mean - local_width), min(1.0, mean + local_width), points)
    return np.unique(np.concatenate([full_grid, local_grid]))


def draw_beta_panel(ax: plt.Axes, alpha: float, beta: float, color: str, show_ylabel: bool = False) -> None:
    mean, variance = beta_mean_variance(alpha, beta)
    std = math.sqrt(variance)
    left = max(0.0, mean - std)
    right = min(1.0, mean + std)

    x = beta_x_grid(mean, variance)
    y = beta_pdf(x, alpha, beta)
    y_max = float(np.max(y))
    arrow_y = y_max * 0.13
    arrow_left, arrow_right = left, right
    arrow_mid = (arrow_left + arrow_right) / 2.0
    arrow_width = arrow_right - arrow_left

    ax.plot(x, y, color=color, linewidth=2.6)
    ax.fill_between(x, 0.0, y, where=(x >= left) & (x <= right), color=color, alpha=0.18)
    ax.axvline(mean, color="#6b7280", linestyle="--", linewidth=1.8)
    ax.axvline(left, color=color, alpha=0.18, linewidth=1.2)
    ax.axvline(right, color=color, alpha=0.18, linewidth=1.2)
    if arrow_width >= 0.055:
        ax.annotate(
            "",
            xy=(arrow_right, arrow_y),
            xytext=(arrow_left, arrow_y),
            arrowprops={"arrowstyle": "<->", "color": "black", "linewidth": 1.6, "mutation_scale": 13},
        )
    else:
        cap_height = y_max * 0.035
        ax.hlines(arrow_y, arrow_left, arrow_right, color="black", linewidth=1.6)
        ax.vlines([arrow_left, arrow_right], arrow_y - cap_height, arrow_y + cap_height, color="black", linewidth=1.6)
    ax.text(arrow_mid, arrow_y + y_max * 0.045, r"$\mu \pm \sigma$", ha="center", va="bottom", fontsize=ANNOTATION_FONTSIZE)
    ax.text(min(mean + 0.015, 0.98), y_max * 0.86, r"$\mu$", ha="left", va="center", fontsize=MEAN_LABEL_FONTSIZE)
    ax.set_title(rf"$\alpha$={alpha:.2f}, $\beta$={beta:.2f}", fontsize=TITLE_FONTSIZE, pad=8)

    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(0.0, y_max * 1.08)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks([])
    ax.set_xlabel(r"$x$", fontsize=AXIS_LABEL_FONTSIZE)
    if show_ylabel:
        ax.set_ylabel(r"$p(x)$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE, width=1.1, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def plot_beta_change() -> None:
    beta_rows = [
        ("Good", GOOD_STATE_BETAS, "#2563eb"),
        ("Bad", BAD_STATE_BETAS, "#dc2626"),
    ]
    fig, axes = plt.subplots(
        len(beta_rows),
        len(GOOD_STATE_BETAS),
        figsize=(16, 8),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, (row_name, beta_params, color) in enumerate(beta_rows):
        for col_index, (alpha, beta) in enumerate(beta_params, start=1):
            draw_beta_panel(axes[row_index][col_index - 1], alpha, beta, color, show_ylabel=col_index == 1)

    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    print(f"Saved figure: {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_beta_change()
