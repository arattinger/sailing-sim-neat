"""Visualize a polar diagram as a polar plot."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt

from sim.polar import PolarDiagram


def _full_circle_angles(angles_deg: List[float]) -> List[float]:
    if len(angles_deg) < 2:
        return angles_deg
    mirror = [360.0 - angle for angle in reversed(angles_deg[1:-1])]
    return angles_deg + mirror + [360.0]


def _full_circle_speeds(speeds: List[float]) -> List[float]:
    if len(speeds) < 2:
        return speeds
    return speeds + list(reversed(speeds[1:-1])) + [speeds[0]]


def _plot_polar(diagram: PolarDiagram) -> None:
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)

    angles = _full_circle_angles(diagram.true_wind_angles_deg)
    theta = [math.radians(angle) for angle in angles]
    cmap = plt.get_cmap("viridis")

    for idx, tws in enumerate(diagram.true_wind_speeds):
        speeds = _full_circle_speeds(diagram.boat_speeds[idx])
        color = cmap(idx / max(1, len(diagram.true_wind_speeds) - 1))
        ax.plot(theta, speeds, label=f"{tws:.1f} m/s", color=color, linewidth=2)

    title = diagram.name or "Polar Diagram"
    ax.set_title(title, fontsize=12, pad=20)
    ax.set_rlabel_position(225)
    ax.grid(True, alpha=0.4)
    ax.legend(title="True wind speed", bbox_to_anchor=(1.2, 1.05))

    fig.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("polar", type=Path, help="Path to a JSON polar file")
    parser.add_argument("--save", type=Path, default=None, help="Optional output image path")
    args = parser.parse_args()

    diagram = PolarDiagram.load(args.polar)
    _plot_polar(diagram)

    if args.save:
        plt.savefig(args.save, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()
