import math
from dataclasses import dataclass
from typing import Dict, Tuple

from .polar import PolarDiagram


@dataclass
class BoatState:
    position: Tuple[float, float]
    heading: float
    speed: float = 0.0


@dataclass
class WindState:
    speed: float
    direction: float


class BoatModelV1:
    """Simple point-mass boat model with a basic polar diagram.

    The model assumes a constant true wind vector during a step and integrates
    heading based on the commanded turn rate. Boat speed is derived from a
    configurable polar diagram; if none is provided a coarse sine-based polar
    is used.
    """

    def __init__(
        self,
        no_go_degrees: float = 40.0,
        base_speed: float = 4.0,
        max_turn_rate: float = math.radians(45.0),
        speed_time_constant: float = 0.0,
        apply_turn_penalty: bool = True,
        turn_penalty_scale: float = 0.5,
        turn_change_penalty_scale: float = 0.05,
        polar: PolarDiagram | None = None,
        wind_push_scale: float = 0.05,
    ) -> None:
        self.no_go_radians = math.radians(no_go_degrees)
        self.base_speed = base_speed
        self.max_turn_rate = max_turn_rate
        self.speed_time_constant = max(0.0, speed_time_constant)
        self.apply_turn_penalty = apply_turn_penalty
        self.turn_penalty_scale = max(0.0, min(1.0, turn_penalty_scale))
        self.turn_change_penalty_scale = max(0.0, min(1.0, turn_change_penalty_scale))
        self.polar = polar
        self.wind_push_scale = max(0.0, wind_push_scale)
        self.last_turn_rate = 0.0

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + 2 * math.pi) % (2 * math.pi)

    def boat_speed(self, state: BoatState, wind: WindState) -> float:
        """Compute the forward boat speed for the current heading and wind.

        Args:
            state: Current boat state.
            wind: True wind state.

        Returns:
            The scalar boat speed in meters per second according to the polar
            diagram.
        """

        return float(state.speed)

    def _polar_speed(self, relative_wind_angle: float, true_wind_speed: float) -> float:
        """Return forward boat speed given the relative wind angle.

        The speed rises with the magnitude of the sine of the relative wind
        angle and drops to zero inside the no-go zone.
        """

        if self.polar:
            return self.polar.speed(true_wind_speed, relative_wind_angle)

        off_wind = abs(math.atan2(math.sin(relative_wind_angle), math.cos(relative_wind_angle)))
        if off_wind < self.no_go_radians:
            return 0.0
        return max(0.0, math.sin(off_wind)) * true_wind_speed / 2 + self.base_speed * 0.1

    def _speed_target(self, heading: float, wind: WindState, turn_rate: float) -> float:
        relative_angle = self._wrap_angle(wind.direction - heading)
        speed = self._polar_speed(relative_angle, wind.speed)
        if not self.apply_turn_penalty:
            return speed

        normalized_turn = min(1.0, abs(turn_rate) / self.max_turn_rate)
        penalty = 1.0 - self.turn_penalty_scale * (normalized_turn**2)
        if self.turn_penalty_scale > 0.0:
            extra_penalty = 1.0 - (self.turn_penalty_scale * 0.5) * (normalized_turn**4)
            penalty *= extra_penalty
        return max(0.0, speed * penalty)

    def step(
        self,
        state: BoatState,
        action: Dict[str, float],
        wind: WindState,
        dt: float,
    ) -> BoatState:
        """Advance the state by ``dt`` seconds.

        Args:
            state: Current boat state.
            action: Contains ``turn_rate`` in radians per second.
            wind: True wind state.
            dt: Time step in seconds.

        Returns:
            The updated boat state.
        """

        turn_rate = float(action.get("turn_rate", 0.0))
        turn_rate = max(-self.max_turn_rate, min(self.max_turn_rate, turn_rate))
        new_heading = self._wrap_angle(state.heading + turn_rate * dt)

        target_speed = self._speed_target(new_heading, wind, turn_rate)
        if self.turn_change_penalty_scale > 0.0:
            turn_delta = abs(turn_rate - self.last_turn_rate)
            normalized_delta = min(1.0, turn_delta / self.max_turn_rate)
            maneuver_penalty = 1.0 - self.turn_change_penalty_scale * (normalized_delta**2)
            target_speed = max(0.0, target_speed * maneuver_penalty)
        if self.speed_time_constant > 0.0:
            alpha = 1.0 - math.exp(-dt / self.speed_time_constant)
            speed = state.speed + (target_speed - state.speed) * alpha
        else:
            speed = target_speed

        # Wind direction is "from", so drift pushes downwind.
        wind_heading = wind.direction + math.pi
        wind_dx = math.cos(wind_heading) * wind.speed * self.wind_push_scale
        wind_dy = math.sin(wind_heading) * wind.speed * self.wind_push_scale
        dx = (math.cos(new_heading) * speed + wind_dx) * dt
        dy = (math.sin(new_heading) * speed + wind_dy) * dt
        new_position = (state.position[0] + dx, state.position[1] + dy)

        self.last_turn_rate = turn_rate
        return BoatState(position=new_position, heading=new_heading, speed=speed)

    def initial_speed(self, heading: float, wind: WindState) -> float:
        return self._speed_target(heading, wind, 0.0)

    def simulate(
        self,
        initial_state: BoatState,
        actions: Tuple[Dict[str, float], ...],
        wind: WindState,
        dt: float,
    ) -> Tuple[BoatState, ...]:
        state = initial_state
        history = []
        for act in actions:
            state = self.step(state, act, wind, dt)
            history.append(state)
        return tuple(history)
