"""Map generation utilities for configurable mark layouts.

The :class:`GoalMapSpec` helper builds a sequence of marks positioned around
an origin. It is intentionally simple—marks are distributed around the boat
with optional bounds clamping—so tests, training, and simulations can easily
request new goal layouts without hand-coding mark coordinates.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

from .env import Mark


@dataclass
class GoalMapSpec:
    """Specification for procedurally generating race marks.

    Attributes:
        count: Number of marks/goals to generate.
        min_distance: Minimum radial distance from ``start`` to each mark.
        max_distance: Maximum radial distance from ``start`` to each mark.
        radius: Radius applied to all generated marks.
        angle_jitter: Random offset applied to each evenly spaced bearing.
        start: Origin used when placing marks.
        bounds: Optional bounding box ``(xmin, xmax, ymin, ymax)`` used to
            clamp generated coordinates.
    """

    count: int
    min_distance: float = 40.0
    max_distance: float = 90.0
    radius: float = 8.0
    angle_jitter: float = math.pi / 6
    start: Tuple[float, float] = (0.0, 0.0)
    bounds: Optional[Tuple[float, float, float, float]] = None

    def for_stage(self, start: Tuple[float, float], bounds: Tuple[float, float, float, float]) -> "GoalMapSpec":
        """Return a copy configured for a specific stage context."""

        return replace(self, start=start, bounds=bounds)

    def _clamp_to_bounds(self, x: float, y: float) -> Tuple[float, float]:
        if not self.bounds:
            return x, y
        xmin, xmax, ymin, ymax = self.bounds
        return max(xmin, min(x, xmax)), max(ymin, min(y, ymax))

    def generate(self, rng: random.Random) -> Sequence[Mark]:
        """Generate a deterministic sequence of marks using ``rng``.

        Marks are distributed evenly around the boat's starting position with
        angular jitter and randomized distances. The final mark is labeled
        ``"finish"`` to remain compatible with existing visualizations.
        """

        if self.count < 1:
            raise ValueError("GoalMapSpec.count must be at least 1")

        angle_step = 2 * math.pi / self.count
        marks: List[Mark] = []
        for i in range(self.count):
            base_angle = angle_step * i
            angle = base_angle + rng.uniform(-self.angle_jitter, self.angle_jitter)
            distance = rng.uniform(self.min_distance, self.max_distance)
            x = self.start[0] + distance * math.cos(angle)
            y = self.start[1] + distance * math.sin(angle)
            x, y = self._clamp_to_bounds(x, y)
            name = "finish" if i == self.count - 1 else f"m{i + 1}"
            marks.append(Mark(name=name, position=(x, y), radius=self.radius))

        return marks
