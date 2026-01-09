"""Observation utilities for sailing-sim.

This module provides a compact observation vector for controllers. The
:class:`ObservationBuilder` converts the environment state into a dict with the
following keys:

``target_dx``
    Target displacement in meters along the boat's forward axis (boat frame).
``target_dy``
    Target displacement in meters to port (boat frame).
``apparent_wind_dx``
    Apparent wind vector x-component in meters per second (boat frame).
``apparent_wind_dy``
    Apparent wind vector y-component in meters per second (boat frame).
``boat_speed``
    Predicted through-water boat speed from the current heading and true wind.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

from .boat import BoatModelV1, BoatState, WindState

if TYPE_CHECKING:  # pragma: no cover
    from .env import Mark


@dataclass
class Observation:
    target_dx: float
    target_dy: float
    apparent_wind_dx: float
    apparent_wind_dy: float
    boat_speed: float

    def to_dict(self) -> Dict[str, float]:
        data: Dict[str, float] = {
            "target_dx": self.target_dx,
            "target_dy": self.target_dy,
            "apparent_wind_dx": self.apparent_wind_dx,
            "apparent_wind_dy": self.apparent_wind_dy,
            "boat_speed": self.boat_speed,
        }
        return data


class ObservationBuilder:
    """Build observation dictionaries for controllers.

    Args:
        model: Boat model used to estimate speed from heading and wind.
    """

    def __init__(self, model: BoatModelV1) -> None:
        self.model = model

    def build(self, state: BoatState, wind: WindState, mark: "Mark") -> Observation:
        dx = mark.position[0] - state.position[0]
        dy = mark.position[1] - state.position[1]
        cos_h = math.cos(state.heading)
        sin_h = math.sin(state.heading)
        target_dx = dx * cos_h + dy * sin_h
        target_dy = -dx * sin_h + dy * cos_h

        boat_speed = self.model.boat_speed(state, wind)
        boat_dx_world = math.cos(state.heading) * boat_speed
        boat_dy_world = math.sin(state.heading) * boat_speed
        wind_heading = wind.direction + math.pi
        wind_dx_world = math.cos(wind_heading) * wind.speed
        wind_dy_world = math.sin(wind_heading) * wind.speed
        apparent_dx_world = wind_dx_world - boat_dx_world
        apparent_dy_world = wind_dy_world - boat_dy_world
        apparent_wind_dx = apparent_dx_world * cos_h + apparent_dy_world * sin_h
        apparent_wind_dy = -apparent_dx_world * sin_h + apparent_dy_world * cos_h

        return Observation(
            target_dx=target_dx,
            target_dy=target_dy,
            apparent_wind_dx=apparent_wind_dx,
            apparent_wind_dy=apparent_wind_dy,
            boat_speed=boat_speed,
        )
