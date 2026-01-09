import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .actions import TurnAction
from .boat import BoatModelV1, BoatState, WindState
from .polar import PolarDiagram
from .observation import ObservationBuilder


@dataclass
class Mark:
    name: str
    position: Tuple[float, float]
    radius: float = 5.0


@dataclass
class WindShiftModel:
    every_steps: int
    direction_std: float
    speed_std: float = 0.0

    def apply(self, wind: WindState, rng: random.Random) -> WindState:
        return WindState(
            speed=float(wind.speed + rng.gauss(0.0, self.speed_std)
                        ) if self.speed_std > 0 else float(wind.speed),
            direction=float(
                (wind.direction + rng.gauss(0.0, self.direction_std)) % (2 * math.pi)),
        )


@dataclass
class WindUpdate:
    step: int
    direction: float | None = None
    speed: float | None = None

    def apply(self, wind: WindState) -> WindState:
        direction = wind.direction if self.direction is None else float(
            self.direction)
        speed = wind.speed if self.speed is None else float(self.speed)
        return WindState(speed=speed, direction=direction % (2 * math.pi))


@dataclass
class StepResult:
    state: BoatState
    wind: WindState
    done: bool
    reason: Optional[str]
    mark_index: int


class SailingEnv:
    def __init__(
        self,
        seed: Optional[int] = None,
        dt: float = 1.0,
        bounds: Tuple[float, float, float,
                      float] = (-100.0, 100.0, -100.0, 100.0),
        marks: Optional[Sequence[Mark]] = None,
        start: Tuple[float, float] = (0.0, 0.0),
        start_heading: float = 0.0,
        trace_path: Optional[str] = None,
        wind: Optional[WindState] = None,
        wind_noise: float = 0.0,
        wind_speed_range: Tuple[float, float] = (3.0, 8.0),
        wind_direction_range: Tuple[float, float] = (0.0, 2 * math.pi),
        wind_shift_model: Optional[WindShiftModel] = None,
        wind_schedule: Sequence[WindUpdate] | None = None,
        polar: PolarDiagram | None = None,
        wind_push_scale: float = 0.05,
        speed_time_constant: float = 5.0,
        turn_change_penalty_scale: float = 0.1,
    ) -> None:
        self.dt = dt
        self.bounds = bounds
        self.rng = random.Random(seed)
        self.wind_noise = wind_noise
        self.wind_speed_range = wind_speed_range
        self.wind_direction_range = wind_direction_range
        self.wind_shift_model = wind_shift_model
        self.wind_schedule = {
            update.step: update for update in (wind_schedule or [])}

        start_heading = float(start_heading)
        start_position = (float(start[0]), float(start[1]))

        self.wind = wind or WindState(
            speed=float(self.rng.uniform(*self.wind_speed_range)),
            direction=float(self.rng.uniform(
                *self.wind_direction_range) % (2 * math.pi)),
        )

        self.marks: List[Mark] = list(marks) if marks else [
            Mark("m1", (50.0, 0.0), radius=8.0),
            Mark("finish", (75.0, 25.0), radius=10.0),
        ]
        self.mark_index = 0

        self.model = BoatModelV1(
            polar=polar,
            wind_push_scale=wind_push_scale,
            speed_time_constant=speed_time_constant,
            turn_change_penalty_scale=turn_change_penalty_scale,
        )
        initial_speed = self.model.initial_speed(start_heading, self.wind)
        self.boat_state = BoatState(
            start_position, start_heading, initial_speed)
        self.observer = ObservationBuilder(self.model)
        self.time = 0.0
        self.step_count = 0
        self.done = False
        self.reason: Optional[str] = None
        self.trace_path = trace_path
        self._metadata_logged = False
        if trace_path:
            os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)

    def reset(self, start: Tuple[float, float], heading: float) -> None:
        initial_speed = self.model.initial_speed(heading, self.wind)
        self.boat_state = BoatState(
            (float(start[0]), float(start[1])), heading, initial_speed)
        self.mark_index = 0
        self.time = 0.0
        self.step_count = 0
        self.done = False
        self.reason = None
        self._metadata_logged = False
        if hasattr(self.model, "last_turn_rate"):
            self.model.last_turn_rate = 0.0

    def _log_metadata(self) -> None:
        if not self.trace_path or self._metadata_logged:
            return

        record = {
            "meta": "sailing-sim.v1",
            "marks": [
                {"name": mark.name, "position": list(
                    mark.position), "radius": mark.radius}
                for mark in self.marks
            ],
        }
        if self.wind:
            record["wind"] = {"speed": float(
                self.wind.speed), "direction": float(self.wind.direction)}
        if self.wind_schedule:
            record["wind_schedule"] = [
                {
                    "step": step,
                    "speed": update.speed,
                    "direction": update.direction,
                }
                for step, update in sorted(self.wind_schedule.items())
            ]
        with open(self.trace_path, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(record))
            fp.write("\n")

        self._metadata_logged = True

    def _log_trace(self, action: Dict[str, float]) -> None:
        if not self.trace_path:
            return

        if not self._metadata_logged:
            self._log_metadata()

        record = {
            "time": self.time,
            "position": list(self.boat_state.position),
            "heading": self.boat_state.heading,
            "wind": {"speed": self.wind.speed, "direction": self.wind.direction},
            "mark_index": self.mark_index,
            "done": self.done,
            "reason": self.reason,
            "action": action,
        }
        with open(self.trace_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record))
            fp.write("\n")

    def _update_wind(self) -> None:
        updated = self.wind
        if self.wind_schedule and self.step_count in self.wind_schedule:
            updated = self.wind_schedule[self.step_count].apply(updated)
        if self.wind_shift_model and self.step_count > 0 and self.step_count % self.wind_shift_model.every_steps == 0:
            updated = self.wind_shift_model.apply(updated, self.rng)

        if self.wind_noise > 0:
            updated = WindState(
                speed=float(updated.speed),
                direction=float(
                    (updated.direction + self.rng.gauss(0.0, self.wind_noise)) % (2 * math.pi)),
            )

        self.wind = updated

    def _normalize_action(self, action: Union[Dict[str, float], TurnAction]) -> Dict[str, float]:
        if isinstance(action, TurnAction):
            return action.to_dict()
        return action

    def _check_bounds(self) -> bool:
        x, y = self.boat_state.position
        xmin, xmax, ymin, ymax = self.bounds
        return x < xmin or x > xmax or y < ymin or y > ymax

    def _check_marks(self) -> None:
        if self.mark_index >= len(self.marks):
            return
        target = self.marks[self.mark_index]
        dx = self.boat_state.position[0] - target.position[0]
        dy = self.boat_state.position[1] - target.position[1]
        distance = math.hypot(dx, dy)
        if distance <= target.radius:
            self.mark_index += 1
            if self.mark_index >= len(self.marks):
                self.done = True
                self.reason = "finished"

    def step(self, action: Union[Dict[str, float], TurnAction]) -> StepResult:
        if self.done:
            return StepResult(self.boat_state, self.wind, self.done, self.reason, self.mark_index)

        action_dict = self._normalize_action(action)
        self.step_count += 1
        self._update_wind()
        self.boat_state = self.model.step(
            self.boat_state, action_dict, self.wind, self.dt)
        self.time += self.dt

        if self._check_bounds():
            self.done = True
            self.reason = "out_of_bounds"
        else:
            self._check_marks()

        self._log_trace(action_dict)

        return StepResult(self.boat_state, self.wind, self.done, self.reason, self.mark_index)

    def observation(self) -> Optional[Dict[str, float]]:
        if self.mark_index >= len(self.marks):
            return None
        mark = self.marks[self.mark_index]
        return self.observer.build(self.boat_state, self.wind, mark).to_dict()
