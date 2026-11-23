"""Figures for results/: convergence curves and the best tours."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
COLOURS = {"GA": "#2a78d6", "SA": "#eb6834", "NN": MUTED, "optimum": INK}  # slots 1-2 + neutrals

plt.rcParams.update(
    {
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def step_values(history: list[list[int]], grid: np.ndarray) -> np.ndarray:
    """Best length known at each grid point (the last record at or before it)."""
    evals = np.array([e for e, _ in history])
    best = np.array([b for _, b in history], dtype=float)
    idx = np.searchsorted(evals, grid, side="right") - 1
    return np.where(idx >= 0, best[np.clip(idx, 0, None)], np.nan)


def convergence(runs: dict, path: Path) -> None:
    """Gap to the optimum against evaluations: mean over seeds with the min-max range shaded."""
    names = list(runs["instances"])
    cols = 2
    rows = (len(names) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.4 * rows), squeeze=False)
    grid = np.unique(np.geomspace(100, runs["evaluations"], 300).astype(int))
    for ax, name in zip(axes.flat, names, strict=False):
        entry = runs["instances"][name]
        optimum = entry["optimum"]
        for algorithm in ("GA", "SA"):
            curves = np.array(
                [step_values(h, grid) for h in entry["algorithms"][algorithm]["histories"]]
            )
            gaps = 100 * (curves - optimum) / optimum
            colour = COLOURS[algorithm]
            ax.fill_between(grid, np.nanmin(gaps, 0), np.nanmax(gaps, 0), color=colour,
                            alpha=0.18, linewidth=0)  # fmt: skip
            ax.plot(grid, np.nanmean(gaps, 0), color=colour, linewidth=2, label=algorithm)
        for algorithm, style in (("NN", "--"), ("NN+2opt", ":")):
            level = entry["algorithms"][algorithm]["mean_gap"]
            ax.axhline(level, color=MUTED, linestyle=style, linewidth=1.2)
            ax.text(grid[0] * 1.15, level, f" {algorithm} {level:.1f} %", color=INK_2,
                    va="bottom", fontsize=8)  # fmt: skip
        ax.set_xscale("log")
        ax.set_ylim(0, max(40, entry["algorithms"]["NN"]["mean_gap"] * 1.25))
        ax.set_xlim(grid[0], grid[-1])
        ax.set_title(f"{name} ({entry['n']} cities)", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("tour evaluations")
        ax.set_ylabel("gap to optimum (%)")
        ax.grid(color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, loc="upper right")
    for ax in axes.flat[len(names) :]:
        ax.set_visible(False)
    fig.suptitle(
        f"Best tour so far, mean of {len(runs['seeds'])} seeds (band: best to worst seed)",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def tours(runs: dict, path: Path) -> None:
    """Best tour of NN, GA and SA next to the optimal tour, one row per instance."""
    names = list(runs["instances"])
    panels = (("NN", "NN"), ("GA", "GA"), ("SA", "SA"), ("optimum", "optimal tour"))
    fig, axes = plt.subplots(len(names), 4, figsize=(11, 2.8 * len(names)), squeeze=False)
    for row, name in zip(axes, names, strict=True):
        entry = runs["instances"][name]
        coords = np.array(entry["coords"])
        for ax, (key, title) in zip(row, panels, strict=True):
            if key == "optimum":
                tour, length = entry["optimal_tour"], entry["optimum"]
            else:
                tour, length = (
                    entry["algorithms"][key]["best_tour"],
                    entry["algorithms"][key]["best"],
                )
            closed = coords[[*tour, tour[0]]]
            ax.plot(closed[:, 0], closed[:, 1], color=COLOURS[key], linewidth=1.4)
            ax.plot(coords[:, 0], coords[:, 1], "o", color=INK, markersize=2.5)
            gap = 100 * (length - entry["optimum"]) / entry["optimum"]
            label = f"{title}: {length:,}" + ("" if key == "optimum" else f" (+{gap:.1f} %)")
            ax.set_title(label, fontsize=9, color=INK)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        row[0].set_ylabel(name, fontsize=10, color=INK)
    fig.suptitle(f"Best tour found in {len(runs['seeds'])} seeds", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
