"""Baseline hand-coded controller to validate environment wiring.

This script spins up :class:`sim.env.SailingEnv`, rolls out a single episode
with a simple heading controller, and saves a JSONL trace for inspection.
Use ``python -m sim.run_episode --help`` for options.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Tuple

from .actions import TurnAction
from .env import SailingEnv
from .map_io import infer_bounds, load_map
from .polar import PolarDiagram
from .mapgen import GoalMapSpec


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def run_episode(
    steps: int,
    start: Tuple[float, float],
    heading_deg: float,
    trace_path: Path,
    seed: int | None,
    map_file: Path | None,
    polar_file: Path | None,
    mark_count: int | None,
    mark_min_distance: float,
    mark_max_distance: float,
    mark_radius: float,
    mark_angle_jitter: float,
    bounds_padding: float,
) -> None:
    bounds = None
    marks = None
    wind = None
    wind_schedule = None
    polar: PolarDiagram | None = None
    if map_file:
        race_config = load_map(map_file, default_radius=mark_radius)
        marks = race_config.marks
        bounds = race_config.bounds
        wind = race_config.wind
        wind_schedule = race_config.wind_schedule
        polar = race_config.polar
        if bounds is None:
            bounds = infer_bounds(marks, padding=bounds_padding)
    if polar_file:
        polar = PolarDiagram.load(polar_file)
    elif polar is None:
        default_polar = Path("data/polars/sample.json")
        if default_polar.exists():
            polar = PolarDiagram.load(default_polar)
    elif mark_count:
        spec = GoalMapSpec(
            count=mark_count,
            min_distance=mark_min_distance,
            max_distance=mark_max_distance,
            radius=mark_radius,
            angle_jitter=mark_angle_jitter,
            start=start,
        )
        marks = spec.generate(random.Random(seed) if seed is not None else random.Random())

    env = SailingEnv(
        seed=seed,
        trace_path=str(trace_path),
        marks=marks,
        bounds=bounds if bounds is not None else (-100.0, 100.0, -100.0, 100.0),
        wind=wind,
        wind_schedule=wind_schedule,
        polar=polar,
    )
    env.reset(start=start, heading=math.radians(heading_deg))

    for _ in range(steps):
        observation = env.observation()
        if observation is None:
            break

        # Proportional heading hold toward the next mark in boat frame.
        heading_error = math.atan2(float(observation["target_dy"]), float(observation["target_dx"]))
        turn_rate = _clamp(heading_error, env.model.max_turn_rate)
        action = TurnAction(turn_rate=turn_rate)
        result = env.step(action)

        if result.done:
            break

    print(f"Episode finished: done={env.done}, reason={env.reason}, marks={env.mark_index}")
    print(f"Trace saved to: {trace_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400, help="maximum steps to run")
    parser.add_argument("--start-x", type=float, default=0.0, help="starting x coordinate")
    parser.add_argument("--start-y", type=float, default=0.0, help="starting y coordinate")
    parser.add_argument(
        "--heading-deg",
        type=float,
        default=0.0,
        help="initial heading in degrees (0 = +x axis)",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("data/traces/episode.jsonl"),
        help="output JSONL trace path",
    )
    parser.add_argument("--map-file", type=Path, default=None, help="Path to a JSON map file with mark coordinates")
    parser.add_argument("--polar-file", type=Path, default=None, help="Path to a JSON polar file")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--mark-count", type=int, default=None, help="number of marks/goals to generate")
    parser.add_argument("--mark-min-distance", type=float, default=40.0, help="minimum radial distance when generating marks")
    parser.add_argument("--mark-max-distance", type=float, default=90.0, help="maximum radial distance when generating marks")
    parser.add_argument("--mark-radius", type=float, default=8.0, help="radius to use for generated marks")
    parser.add_argument(
        "--mark-angle-jitter",
        type=float,
        default=math.pi / 6,
        help="angular jitter applied to evenly spaced marks",
    )
    parser.add_argument(
        "--map-bounds-padding",
        type=float,
        default=25.0,
        help="Padding added around map coordinates when inferring bounds",
    )
    args = parser.parse_args()

    if args.map_file and args.mark_count is not None:
        raise ValueError("Use either --map-file or --mark-count, not both")

    run_episode(
        steps=args.steps,
        start=(args.start_x, args.start_y),
        heading_deg=args.heading_deg,
        trace_path=args.trace,
        seed=args.seed,
        map_file=args.map_file,
        polar_file=args.polar_file,
        mark_count=args.mark_count,
        mark_min_distance=args.mark_min_distance,
        mark_max_distance=args.mark_max_distance,
        mark_radius=args.mark_radius,
        mark_angle_jitter=args.mark_angle_jitter,
        bounds_padding=args.map_bounds_padding,
    )


if __name__ == "__main__":
    main()
