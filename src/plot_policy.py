"""Visualize a learned Taxi policy as arrow grids + value heatmaps.

The state space is 500 = 25 taxi-positions x 5 passenger-locations x 4 destinations,
so we can't show everything in one picture. Instead we render a 2x4 panel:
two passenger scenarios x four destination corners.

Top row    : passenger waiting at R, taxi must pick them up.
Bottom row : passenger already in the taxi, taxi must drop them off.
Each cell  : 5x5 grid showing V(s) heatmap and the greedy action arrow.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from .env import GRID_SHAPE, LOC_NAMES, LOCATIONS

# (drow, dcol) for each movement action; pickup/dropoff drawn as a marker.
_ARROW = {
    0: (1, 0),    # south
    1: (-1, 0),   # north
    2: (0, 1),    # east
    3: (0, -1),   # west
}
_PICKUP, _DROPOFF = 4, 5


def _encode(row: int, col: int, pass_loc: int, dest: int) -> int:
    return ((row * 5 + col) * 5 + pass_loc) * 4 + dest


def _draw_cell(ax, Q: np.ndarray, pass_loc: int, dest: int, title: str) -> None:
    n_rows, n_cols = GRID_SHAPE
    V = np.zeros((n_rows, n_cols))
    greedy = np.zeros((n_rows, n_cols), dtype=int)
    for r in range(n_rows):
        for c in range(n_cols):
            s = _encode(r, c, pass_loc, dest)
            V[r, c] = Q[s].max()
            greedy[r, c] = int(Q[s].argmax())

    im = ax.imshow(V, cmap="viridis", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)

    for r in range(n_rows):
        for c in range(n_cols):
            a = greedy[r, c]
            if a in _ARROW:
                dy, dx = _ARROW[a]
                ax.arrow(c, r, 0.3 * dx, 0.3 * dy,
                         head_width=0.15, head_length=0.15,
                         fc="white", ec="black", length_includes_head=True)
            elif a == _PICKUP:
                ax.text(c, r, "P", color="white", ha="center", va="center",
                        fontsize=10, fontweight="bold")
            elif a == _DROPOFF:
                ax.text(c, r, "D", color="white", ha="center", va="center",
                        fontsize=10, fontweight="bold")

    if pass_loc < 4:
        pr, pc = LOCATIONS[pass_loc]
        ax.add_patch(plt.Circle((pc, pr), 0.35, fill=False,
                                edgecolor="orange", linewidth=2))
    dr, dc = LOCATIONS[dest]
    ax.add_patch(plt.Rectangle((dc - 0.4, dr - 0.4), 0.8, 0.8, fill=False,
                               edgecolor="red", linewidth=2))

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_title(title, fontsize=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Learned Taxi policy")
    args = parser.parse_args()

    Q = np.load(args.weights)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for dest in range(4):
        # Row 0: passenger waiting at R (pass_loc=0). Row 1: passenger in taxi (pass_loc=4).
        _draw_cell(axes[0, dest], Q, pass_loc=0, dest=dest,
                   title=f"Pickup at R, drop at {LOC_NAMES[dest]}")
        _draw_cell(axes[1, dest], Q, pass_loc=4, dest=dest,
                   title=f"In taxi, drop at {LOC_NAMES[dest]}")

    fig.suptitle(args.title + "  (orange=passenger, red=destination)", fontsize=12)
    plt.tight_layout()
    plt.savefig(args.output, dpi=120)
    print(f"Saved policy plot to {args.output}")


if __name__ == "__main__":
    main()
