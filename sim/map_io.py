"""Utilities for loading user-defined race definitions.

Race files are JSON and extend the existing mark map format with optional wind
metadata. Two formats are accepted:

```
# Shorthand: a list of coordinate pairs uses generated names and default radius
[[50, 0], [75, 25]]

# Explicit: customise names/radii, optional bounds, and optional wind config
{
  "marks": [
    {"name": "m1", "position": [50, 0], "radius": 6},
    {"name": "finish", "position": [75, 25], "radius": 8}
  ],
  "bounds": [-120, 120, -120, 120],
  "wind": {"speed": 6.0, "direction_deg": 45.0},
  "wind_schedule": [
    {"step": 100, "direction_deg": 135.0},
    {"step": 200, "speed": 8.5, "direction_deg": 180.0}
  ],
  "polar": "data/polars/sample.json"
}
```

When names are omitted they are generated (``m1``...``finish``) and missing
radii fall back to the provided ``default_radius``. Wind fields are optional:
``wind`` sets the initial static true wind and ``wind_schedule`` supplies
step-indexed updates. ``polar`` can be a file path (resolved relative to the
map file when possible) or inline JSON polar. Wind directions are expressed in
**degrees** with 0 along the positive x-axis and increase counter-clockwise.
Directions indicate where the wind is coming from (``direction`` in radians is
also accepted when ``direction_deg`` is omitted).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from .env import Mark, WindUpdate, WindState
from .polar import PolarDiagram

Bounds = Tuple[float, float, float, float]


@dataclass
class RaceConfig:
    marks: List[Mark]
    bounds: Bounds | None
    wind: WindState | None
    wind_schedule: List[WindUpdate]
    polar: PolarDiagram | None


def _normalize_position(raw_position: Sequence[float]) -> Tuple[float, float]:
    if len(raw_position) != 2:
        raise ValueError(f"Map positions must have two elements, got {raw_position}")
    return float(raw_position[0]), float(raw_position[1])


def _parse_direction(raw_direction: object, context: str) -> float:
    if raw_direction is None:
        raise ValueError(f"Wind direction missing for {context}")

    if isinstance(raw_direction, (int, float)):
        return float(raw_direction)

    raise ValueError(f"Wind direction for {context} must be a number, got {raw_direction}")


def _parse_wind(payload: object) -> WindState:
    if not isinstance(payload, dict):
        raise ValueError(f"wind must be an object, got {payload}")

    if "direction_deg" in payload:
        direction = math.radians(_parse_direction(payload["direction_deg"], "wind"))
    else:
        direction = _parse_direction(payload.get("direction"), "wind")

    if "speed" not in payload:
        raise ValueError("wind must include a 'speed' field")

    return WindState(speed=float(payload["speed"]), direction=direction % (2 * math.pi))


def _parse_wind_schedule(payload: object) -> List[WindUpdate]:
    if not isinstance(payload, list):
        raise ValueError("wind_schedule must be a list of objects")

    schedule: List[WindUpdate] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError(f"wind_schedule entries must be objects, got {entry}")

        if "step" not in entry:
            raise ValueError(f"wind_schedule entries must include 'step': {entry}")

        step = int(entry["step"])
        if step < 0:
            raise ValueError(f"wind_schedule step must be non-negative, got {step}")

        direction: float | None
        if "direction_deg" in entry:
            direction = math.radians(_parse_direction(entry["direction_deg"], f"wind_schedule step {step}"))
        elif "direction" in entry:
            direction = _parse_direction(entry["direction"], f"wind_schedule step {step}") % (2 * math.pi)
        else:
            direction = None

        speed = entry.get("speed")
        if speed is not None:
            speed = float(speed)

        if direction is None and speed is None:
            raise ValueError(f"wind_schedule entry must specify speed or direction: {entry}")

        schedule.append(WindUpdate(step=step, direction=direction, speed=speed))

    return sorted(schedule, key=lambda update: update.step)


def _parse_mark(raw_mark: object, index: int, total: int, default_radius: float) -> Mark:
    if isinstance(raw_mark, (list, tuple)):
        position = _normalize_position(raw_mark)
        radius = default_radius
        name = "finish" if index == total - 1 else f"m{index + 1}"
        return Mark(name=name, position=position, radius=radius)

    if isinstance(raw_mark, dict):
        if "position" not in raw_mark:
            raise ValueError(f"Mark entry missing 'position': {raw_mark}")
        position = _normalize_position(raw_mark["position"])
        name = raw_mark.get("name") or ("finish" if index == total - 1 else f"m{index + 1}")
        radius = float(raw_mark.get("radius", default_radius))
        return Mark(name=str(name), position=position, radius=radius)

    raise ValueError(f"Unsupported mark entry: {raw_mark}")


def _parse_polar(payload: object, map_path: Path) -> PolarDiagram:
    if isinstance(payload, str):
        polar_path = Path(payload).expanduser()
        if not polar_path.is_absolute():
            candidate = map_path.parent / polar_path
            if candidate.exists():
                polar_path = candidate
        return PolarDiagram.load(polar_path)
    if isinstance(payload, dict):
        return PolarDiagram.from_payload(payload)
    raise ValueError(f"polar must be a file path or object, got {payload}")


def load_map(path: Path, default_radius: float = 8.0) -> RaceConfig:
    """Load marks, bounds, and wind metadata from a JSON race file."""

    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    bounds: Bounds | None = None
    wind: WindState | None = None
    wind_schedule: List[WindUpdate] = []
    polar: PolarDiagram | None = None
    raw_marks: Iterable[object]

    if isinstance(payload, dict):
        raw_marks = payload.get("marks") or []
        if "bounds" in payload:
            raw_bounds = payload["bounds"]
            if not isinstance(raw_bounds, (list, tuple)) or len(raw_bounds) != 4:
                raise ValueError("bounds must be a sequence of four numbers [xmin, xmax, ymin, ymax]")
            bounds = tuple(float(x) for x in raw_bounds)  # type: ignore[assignment]
        if "wind" in payload:
            wind = _parse_wind(payload["wind"])
        if "wind_schedule" in payload:
            wind_schedule = _parse_wind_schedule(payload["wind_schedule"])
        if "polar" in payload:
            polar = _parse_polar(payload["polar"], path)
    elif isinstance(payload, list):
        raw_marks = payload
    else:
        raise ValueError("Map file must be a list of coordinates or an object with a 'marks' key")

    marks = [
        _parse_mark(raw_mark, idx, len(raw_marks), default_radius)
        for idx, raw_mark in enumerate(raw_marks)
    ]

    if not marks:
        raise ValueError(f"No marks found in map file: {path}")

    return RaceConfig(marks=marks, bounds=bounds, wind=wind, wind_schedule=wind_schedule, polar=polar)


def infer_bounds(marks: Sequence[Mark], padding: float = 20.0) -> Bounds:
    """Infer a bounding box from the provided marks with extra padding."""

    xs = [mark.position[0] for mark in marks]
    ys = [mark.position[1] for mark in marks]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return (
        xmin - padding,
        xmax + padding,
        ymin - padding,
        ymax + padding,
    )
