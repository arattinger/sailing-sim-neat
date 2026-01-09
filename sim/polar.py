"""Polar diagram utilities for boat speed lookup."""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


def _to_float_list(values: Sequence[object], label: str) -> List[float]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{label} must be a list, got {values}")
    return [float(v) for v in values]


def _is_strictly_increasing(values: Sequence[float]) -> bool:
    return all(values[i] < values[i + 1] for i in range(len(values) - 1))


@dataclass
class PolarDiagram:
    name: str | None
    true_wind_speeds: List[float]
    true_wind_angles_deg: List[float]
    boat_speeds: List[List[float]]

    @classmethod
    def load(cls, path: Path) -> "PolarDiagram":
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return cls.from_payload(payload)

    @classmethod
    def from_payload(cls, payload: object) -> "PolarDiagram":
        if not isinstance(payload, dict):
            raise ValueError(f"Polar must be an object, got {payload}")

        raw_angles = payload.get("true_wind_angles_deg") or payload.get("angles_deg")
        raw_speeds = payload.get("true_wind_speeds")
        raw_boat = payload.get("boat_speeds") or payload.get("speeds")

        if raw_angles is None or raw_speeds is None or raw_boat is None:
            raise ValueError("Polar requires true_wind_angles_deg, true_wind_speeds, and boat_speeds")

        angles = _to_float_list(raw_angles, "true_wind_angles_deg")
        tws = _to_float_list(raw_speeds, "true_wind_speeds")

        if not angles or not tws:
            raise ValueError("Polar angles and wind speeds cannot be empty")
        if not _is_strictly_increasing(angles):
            raise ValueError("Polar true_wind_angles_deg must be strictly increasing")
        if not _is_strictly_increasing(tws):
            raise ValueError("Polar true_wind_speeds must be strictly increasing")
        if min(angles) < 0.0 or max(angles) > 180.0:
            raise ValueError("Polar angles must be within [0, 180] degrees")

        if not isinstance(raw_boat, (list, tuple)):
            raise ValueError("boat_speeds must be a list of rows")
        boat_speeds: List[List[float]] = []
        for row in raw_boat:
            speeds_row = _to_float_list(row, "boat_speeds row")
            if len(speeds_row) != len(angles):
                raise ValueError("Each boat_speeds row must match the angle count")
            boat_speeds.append(speeds_row)
        if len(boat_speeds) != len(tws):
            raise ValueError("boat_speeds must have one row per true wind speed")

        name = payload.get("name")
        return cls(
            name=str(name) if name is not None else None,
            true_wind_speeds=tws,
            true_wind_angles_deg=angles,
            boat_speeds=boat_speeds,
        )

    def speed(self, true_wind_speed: float, relative_wind_angle: float) -> float:
        if true_wind_speed <= 0.0:
            return 0.0

        twa = abs(math.atan2(math.sin(relative_wind_angle), math.cos(relative_wind_angle)))
        twa_deg = math.degrees(twa)
        twa_deg = min(max(twa_deg, self.true_wind_angles_deg[0]), self.true_wind_angles_deg[-1])
        tws = min(max(true_wind_speed, self.true_wind_speeds[0]), self.true_wind_speeds[-1])

        ti0, ti1, t = _interp_indices(self.true_wind_speeds, tws)
        ai0, ai1, a = _interp_indices(self.true_wind_angles_deg, twa_deg)

        s00 = self.boat_speeds[ti0][ai0]
        s01 = self.boat_speeds[ti0][ai1]
        s10 = self.boat_speeds[ti1][ai0]
        s11 = self.boat_speeds[ti1][ai1]

        s0 = s00 + (s01 - s00) * a
        s1 = s10 + (s11 - s10) * a
        return max(0.0, s0 + (s1 - s0) * t)


def _interp_indices(values: Sequence[float], target: float) -> tuple[int, int, float]:
    if target <= values[0]:
        return 0, 0, 0.0
    if target >= values[-1]:
        last = len(values) - 1
        return last, last, 0.0

    idx = bisect_right(values, target)
    low = idx - 1
    high = idx
    span = values[high] - values[low]
    t = 0.0 if span <= 0.0 else (target - values[low]) / span
    return low, high, t
