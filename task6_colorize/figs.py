"""The one plotting style in this folder, so Task 6's figures match the audit's.

Same palette and rcParams as `../task6_audit_figs/` and as Tasks 1 and 3: colour carries
identity and never rank, every multi-series panel has a legend, nothing calls `plt.show()`
(so the scripts run headless and a figure is a file, not a window), and every number that
appears in a figure also appears in a `.json`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")            # never a window: a figure here is a file (house rule)
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is chosen)
import numpy as np

FIGURES = Path(__file__).resolve().parent / "figures"

INK, MUTED, GREY = "#0b0b0b", "#52514e", "#9b9a94"
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#1baf7a", "#c2453d"
GRID, SURFACE = "#e8e6e0", "#fcfcfb"
SERIES = (BLUE, ORANGE, GREEN, RED)
METHOD_COLORS = {"gray": GREY, "A": BLUE, "B": ORANGE, "B+": GREEN, "truth": INK,
                 "binned truth": RED}      # the fourth categorical hue, not a status colour


def style(size: int = 8) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": size,
        "axes.edgecolor": GREY, "axes.linewidth": 0.8,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.autolayout": False})


def save(fig, name: str) -> Path:
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(FIGURES.parent)}")
    return path


def photo_box(ax, rgb: np.ndarray, title: str = "") -> None:
    """One photo, no axes, no interpolation: these are 64x64 and must look like it."""
    ax.imshow(np.clip(rgb, 0, 255).astype(np.uint8))
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7.5, loc="left", color=INK)


def legend_on(ax, **kw) -> None:
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GREY, framealpha=0.92,
              fontsize=7, handlelength=1.4, **kw)


def annotate_peak(ax, x: float, y: float, label: str = "best epoch", color: str = ORANGE) -> None:
    ax.axvline(x, color=color, linestyle=":", linewidth=1.0)
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, -2),
                fontsize=7, color=color, va="top")
