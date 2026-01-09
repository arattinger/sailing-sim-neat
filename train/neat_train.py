"""NEAT training harness for the sailing simulator.

The trainer uses a config-driven ``neat-python`` setup, evaluates genomes over
multiple random seeds to smooth variance, shapes fitness based on mark progress
and distance, and logs per-generation metrics to CSV and JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import shutil
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import neat

from sim.env import Mark, SailingEnv, WindShiftModel, WindState, WindUpdate
from sim.map_io import infer_bounds, load_map
from sim.polar import PolarDiagram
from sim.mapgen import GoalMapSpec


@dataclass
class FitnessParameters:
    max_steps: int
    seeds: Sequence[int]
    progress_scale: float
    goal_progress_scale: float
    vmg_scale: float
    stall_speed_threshold: float
    stall_penalty: float
    finish_time_scale: float
    mark_time_scale: float
    step_penalty: float
    turn_rate_penalty: float
    finish_bonus: float
    out_of_bounds_penalty: float
    tack_penalty: float
    oscillation_penalty: float

    def apply_overrides(self, overrides: "StagePenalties") -> "FitnessParameters":
        return FitnessParameters(
            max_steps=self.max_steps,
            seeds=self.seeds,
            progress_scale=overrides.progress_scale if overrides.progress_scale is not None else self.progress_scale,
            goal_progress_scale=(
                overrides.goal_progress_scale
                if overrides.goal_progress_scale is not None
                else self.goal_progress_scale
            ),
            vmg_scale=overrides.vmg_scale if overrides.vmg_scale is not None else self.vmg_scale,
            stall_speed_threshold=(
                overrides.stall_speed_threshold
                if overrides.stall_speed_threshold is not None
                else self.stall_speed_threshold
            ),
            stall_penalty=overrides.stall_penalty if overrides.stall_penalty is not None else self.stall_penalty,
            finish_time_scale=(
                overrides.finish_time_scale
                if overrides.finish_time_scale is not None
                else self.finish_time_scale
            ),
            mark_time_scale=overrides.mark_time_scale if overrides.mark_time_scale is not None else self.mark_time_scale,
            step_penalty=overrides.step_penalty if overrides.step_penalty is not None else self.step_penalty,
            turn_rate_penalty=(
                overrides.turn_rate_penalty
                if overrides.turn_rate_penalty is not None
                else self.turn_rate_penalty
            ),
            finish_bonus=overrides.finish_bonus if overrides.finish_bonus is not None else self.finish_bonus,
            out_of_bounds_penalty=
            overrides.out_of_bounds_penalty if overrides.out_of_bounds_penalty is not None else self.out_of_bounds_penalty,
            tack_penalty=overrides.tack_penalty if overrides.tack_penalty is not None else self.tack_penalty,
            oscillation_penalty=overrides.oscillation_penalty if overrides.oscillation_penalty is not None else self.oscillation_penalty,
        )


@dataclass
class StagePenalties:
    progress_scale: Optional[float] = None
    goal_progress_scale: Optional[float] = None
    vmg_scale: Optional[float] = None
    stall_speed_threshold: Optional[float] = None
    stall_penalty: Optional[float] = None
    finish_time_scale: Optional[float] = None
    mark_time_scale: Optional[float] = None
    step_penalty: Optional[float] = None
    turn_rate_penalty: Optional[float] = None
    finish_bonus: Optional[float] = None
    out_of_bounds_penalty: Optional[float] = None
    tack_penalty: Optional[float] = None
    oscillation_penalty: Optional[float] = None


@dataclass
class WindShiftSpec:
    every_steps: int
    direction_std: float
    speed_std: float = 0.0

    def to_model(self) -> WindShiftModel:
        return WindShiftModel(self.every_steps, self.direction_std, self.speed_std)


@dataclass
class StageDefinition:
    name: str
    bounds: Tuple[float, float, float, float]
    marks: Optional[Sequence[Mark]] = None
    mark_generator: Optional[GoalMapSpec] = None
    start: Tuple[float, float] = (0.0, 0.0)
    heading: float = 0.0
    wind: WindState | None = None
    wind_schedule: Sequence[WindUpdate] | None = None
    wind_speed_range: Tuple[float, float] = (3.0, 8.0)
    wind_direction_range: Tuple[float, float] = (0.0, 2 * math.pi)
    wind_noise: float = 0.05
    wind_shift: Optional[WindShiftSpec] = None
    seeds: Optional[Sequence[int]] = None
    penalties: StagePenalties = field(default_factory=StagePenalties)
    randomize_heading: bool = False
    heading_jitter: float = math.pi / 2
    randomize_wind: bool = False
    wind_direction_jitter: float = math.pi / 3
    wind_speed_jitter: float = 0.0

    def wind_shift_model(self) -> Optional[WindShiftModel]:
        return self.wind_shift.to_model() if self.wind_shift else None

    def heading_for_seed(self, seed: int) -> float:
        if not self.randomize_heading:
            return self.heading
        rng = random.Random(seed + 101)
        return float(self.heading + rng.uniform(-self.heading_jitter, self.heading_jitter))

    def wind_for_seed(self, seed: int) -> WindState | None:
        if self.wind is None:
            return None
        if not self.randomize_wind:
            return self.wind
        rng = random.Random(seed + 303)
        speed_jitter = rng.uniform(-self.wind_speed_jitter, self.wind_speed_jitter)
        direction_jitter = rng.uniform(-self.wind_direction_jitter, self.wind_direction_jitter)
        return WindState(
            speed=max(0.0, float(self.wind.speed + speed_jitter)),
            direction=float((self.wind.direction + direction_jitter) % (2 * math.pi)),
        )

    def marks_for_seed(self, seed: int) -> Sequence[Mark]:
        if self.mark_generator:
            return self.mark_generator.for_stage(self.start, self.bounds).generate(random.Random(seed))
        if self.marks is None:
            raise ValueError("StageDefinition requires either marks or a mark_generator")
        return self.marks


@dataclass
class StageStats:
    generation: int
    stage_name: str
    stage_index: int
    finishes: int
    episodes: int

    @property
    def success_rate(self) -> float:
        return 0.0 if self.episodes == 0 else self.finishes / self.episodes


class CurriculumManager:
    def __init__(self, stages: Sequence[StageDefinition], mode: str, success_threshold: float) -> None:
        if not stages:
            raise ValueError("At least one stage is required for the curriculum")
        self.stages = list(stages)
        self.mode = mode
        self.success_threshold = success_threshold
        self.stage_index = 0
        self.last_stage_stats: StageStats | None = None

    @property
    def current_stage(self) -> StageDefinition:
        return self.stages[self.stage_index]

    def record_generation(self, stats: StageStats) -> None:
        self.last_stage_stats = stats
        if self.mode == "cycle":
            self.stage_index = (self.stage_index + 1) % len(self.stages)
        elif self.mode == "progress" and stats.success_rate >= self.success_threshold and self.stage_index < len(self.stages) - 1:
            self.stage_index += 1


def _network_inputs(observation: Dict[str, float], scales: Tuple[float, float, float]) -> List[float]:
    target_scale, wind_scale, speed_scale = scales
    return [
        float(observation["target_dx"]) / target_scale,
        float(observation["target_dy"]) / target_scale,
        float(observation["apparent_wind_dx"]) / wind_scale,
        float(observation["apparent_wind_dy"]) / wind_scale,
        float(observation["boat_speed"]) / speed_scale,
    ]


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


@dataclass
class EpisodeResult:
    fitness: float
    finished: bool
    steps: int


def _evaluate_episode(
    genome: neat.DefaultGenome,
    config: neat.Config,
    seed: int,
    fitness_params: FitnessParameters,
    stage: StageDefinition,
    polar: PolarDiagram | None = None,
    trace_path: Path | None = None,
) -> EpisodeResult:
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    stage_heading = stage.heading_for_seed(seed)
    stage_wind = stage.wind_for_seed(seed) or stage.wind
    env = SailingEnv(
        seed=seed,
        trace_path=str(trace_path) if trace_path else None,
        bounds=stage.bounds,
        marks=stage.marks_for_seed(seed),
        start=stage.start,
        start_heading=stage_heading,
        wind=stage_wind,
        wind_schedule=stage.wind_schedule,
        wind_noise=stage.wind_noise,
        wind_speed_range=stage.wind_speed_range,
        wind_direction_range=stage.wind_direction_range,
        wind_shift_model=stage.wind_shift_model(),
        polar=polar,
    )
    env.reset(start=stage.start, heading=stage_heading)

    observation = env.observation()
    if observation is None:
        return EpisodeResult(0.0, False, 0)

    def _distance_to_mark() -> float:
        if env.mark_index >= len(env.marks):
            return 0.0
        mark = env.marks[env.mark_index]
        dx = mark.position[0] - env.boat_state.position[0]
        dy = mark.position[1] - env.boat_state.position[1]
        return math.hypot(dx, dy)

    initial_distance = _distance_to_mark()
    mark_initial_distance = max(1.0, initial_distance)
    progress_reward = 0.0
    finish_position = env.marks[-1].position
    finish_initial_distance = math.hypot(
        finish_position[0] - env.boat_state.position[0],
        finish_position[1] - env.boat_state.position[1],
    )
    goal_progress_reward = 0.0
    mark_time_bonus = 0.0
    vmg_progress = 0.0
    stall_steps = 0
    tack_count = 0
    oscillations = 0
    turn_magnitude = 0.0
    previous_action: float | None = None
    prev_position = env.boat_state.position
    prev_mark_index = env.mark_index
    prev_mark_distance = initial_distance
    prev_finish_distance = finish_initial_distance
    last_mark_step = 0
    steps_taken = 0
    bounds = stage.bounds
    span_x = bounds[1] - bounds[0]
    span_y = bounds[3] - bounds[2]
    target_scale = max(1.0, math.hypot(span_x, span_y))
    wind_speed_cap = stage.wind.speed if stage.wind else 0.0
    wind_scale = max(1.0, stage.wind_speed_range[1], wind_speed_cap)
    speed_scale = wind_scale
    input_scales = (target_scale, wind_scale, speed_scale)

    for step_idx in range(fitness_params.max_steps):
        steps_taken = step_idx + 1
        inputs = _network_inputs(observation, input_scales)
        turn_rate_norm = float(network.activate(inputs)[0])
        turn_rate = _clamp(turn_rate_norm, 1.0) * env.model.max_turn_rate
        turn_magnitude += abs(turn_rate) / max(1e-6, env.model.max_turn_rate)

        if previous_action is not None:
            sign_change = math.copysign(1.0, turn_rate) != math.copysign(1.0, previous_action)
            if sign_change and abs(turn_rate) > 1e-3 and abs(previous_action) > 1e-3:
                oscillations += 1
                if abs(turn_rate) > 0.5 * env.model.max_turn_rate and abs(previous_action) > 0.5 * env.model.max_turn_rate:
                    tack_count += 1
        previous_action = turn_rate

        result = env.step({"turn_rate": turn_rate})

        observation = env.observation()
        step_dx = env.boat_state.position[0] - prev_position[0]
        step_dy = env.boat_state.position[1] - prev_position[1]
        target_dx = finish_position[0] - prev_position[0]
        target_dy = finish_position[1] - prev_position[1]
        target_distance = math.hypot(target_dx, target_dy)
        if env.boat_state.speed < fitness_params.stall_speed_threshold:
            stall_steps += 1
        if target_distance > 1e-6:
            target_dir_x = target_dx / target_distance
            target_dir_y = target_dy / target_distance
            step_vmg = step_dx * target_dir_x + step_dy * target_dir_y
            vmg_progress += step_vmg / max(1.0, finish_initial_distance)
        prev_position = env.boat_state.position
        current_mark_distance = _distance_to_mark()
        current_mark_index = result.mark_index
        if current_mark_index > prev_mark_index:
            marks_advanced = current_mark_index - prev_mark_index
            mark_steps = steps_taken - last_mark_step
            mark_fraction = mark_steps / max(1, fitness_params.max_steps)
            mark_time_bonus += (
                fitness_params.mark_time_scale * max(0.0, 1.0 - mark_fraction) * marks_advanced
            )
            last_mark_step = steps_taken
            progress_reward += marks_advanced
            prev_mark_index = current_mark_index
            prev_mark_distance = current_mark_distance
            mark_initial_distance = max(1.0, prev_mark_distance)
        else:
            distance_delta = prev_mark_distance - current_mark_distance
            progress_reward += distance_delta / mark_initial_distance
            prev_mark_distance = current_mark_distance

        finish_distance = math.hypot(
            finish_position[0] - env.boat_state.position[0],
            finish_position[1] - env.boat_state.position[1],
        )
        goal_delta = prev_finish_distance - finish_distance
        goal_progress_reward += goal_delta / max(1.0, finish_initial_distance)
        prev_finish_distance = finish_distance

        if result.done:
            break

    finished = env.done and env.reason == "finished"
    fitness = (
        progress_reward * fitness_params.progress_scale
        + goal_progress_reward * fitness_params.goal_progress_scale
        + vmg_progress * fitness_params.vmg_scale
        + mark_time_bonus
    )
    fitness -= stall_steps * fitness_params.stall_penalty
    if finished:
        fitness += fitness_params.finish_bonus
        finish_fraction = steps_taken / max(1, fitness_params.max_steps)
        fitness += fitness_params.finish_time_scale * max(0.0, 1.0 - finish_fraction)
    if env.reason == "out_of_bounds":
        fitness -= fitness_params.out_of_bounds_penalty

    fitness -= fitness_params.tack_penalty * tack_count
    fitness -= fitness_params.oscillation_penalty * oscillations
    fitness -= steps_taken * fitness_params.step_penalty
    fitness -= turn_magnitude * fitness_params.turn_rate_penalty

    return EpisodeResult(fitness, finished, steps_taken)


class StageEvaluator:
    def __init__(
        self,
        base_fitness_params: FitnessParameters,
        curriculum: "CurriculumManager",
        polar: PolarDiagram | None = None,
    ) -> None:
        self.base_fitness_params = base_fitness_params
        self.curriculum = curriculum
        self.polar = polar

    def evaluate_genomes(
        self,
        genomes: Iterable[tuple[int, neat.DefaultGenome]],
        config: neat.Config,
        generation: int,
    ) -> None:
        stage = self.curriculum.current_stage
        stage_fitness = self.base_fitness_params.apply_overrides(stage.penalties)
        seeds = stage.seeds or stage_fitness.seeds

        finishes = 0
        episodes = 0

        for _, genome in genomes:
            results = [
                _evaluate_episode(genome, config, seed, stage_fitness, stage, polar=self.polar)
                for seed in seeds
            ]
            genome.fitness = float(sum(result.fitness for result in results) / len(results))
            finishes += sum(1 for result in results if result.finished)
            episodes += len(results)

        self.curriculum.record_generation(StageStats(generation, stage.name, self.curriculum.stage_index, finishes, episodes))


def _build_default_curriculum(
    mark_count: int | None,
    map_spec: GoalMapSpec | None,
    map_marks: Sequence[Mark] | None,
    map_bounds: Tuple[float, float, float, float] | None,
    map_wind: WindState | None,
    map_wind_schedule: Sequence[WindUpdate] | None,
    bounds_padding: float,
) -> List[StageDefinition]:
    def _generator(bounds: Tuple[float, float, float, float], start: Tuple[float, float]) -> GoalMapSpec | None:
        if map_marks or map_spec is None:
            return None
        return map_spec.for_stage(start=start, bounds=bounds)

    def _marks(default: Sequence[Mark]) -> Sequence[Mark] | None:
        if map_marks:
            return map_marks
        if mark_count:
            return None
        return default

    def _bounds(default: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        if map_marks:
            return map_bounds or infer_bounds(map_marks, padding=bounds_padding)
        return default

    short_bounds = _bounds((-80.0, 80.0, -80.0, 80.0))
    first_bounds = _bounds((-120.0, 120.0, -120.0, 120.0))
    second_bounds = _bounds((-100.0, 100.0, -100.0, 100.0))
    third_bounds = _bounds((-80.0, 80.0, -80.0, 80.0))

    return [
        StageDefinition(
            name="short_tack_intro",
            bounds=short_bounds,
            marks=_marks([Mark("finish", (30.0, 0.0), radius=12.0)]),
            mark_generator=_generator(short_bounds, (0.0, 0.0)),
            start=(0.0, 0.0),
            heading=0.0,
            wind=map_wind,
            wind_schedule=map_wind_schedule,
            wind_speed_range=(3.0, 5.0),
            wind_direction_range=(-math.pi / 6, math.pi / 6),
            wind_noise=0.0,
            penalties=StagePenalties(out_of_bounds_penalty=0.5, tack_penalty=0.0),
            randomize_heading=False,
            heading_jitter=math.pi / 2,
            randomize_wind=False,
            wind_direction_jitter=math.pi / 6,
            wind_speed_jitter=0.5,
        ),
        StageDefinition(
            name="single_mark_stable",
            bounds=first_bounds,
            marks=_marks([Mark("finish", (60.0, 0.0), radius=10.0)]),
            mark_generator=_generator(first_bounds, (0.0, 0.0)),
            start=(0.0, 0.0),
            heading=0.0,
            wind=map_wind,
            wind_schedule=map_wind_schedule,
            wind_speed_range=(3.0, 6.0),
            wind_direction_range=(-math.pi / 4, math.pi / 4),
            wind_noise=0.0,
            penalties=StagePenalties(out_of_bounds_penalty=1.0, tack_penalty=0.0),
            randomize_heading=False,
            heading_jitter=math.pi / 2,
            randomize_wind=False,
            wind_direction_jitter=math.pi / 4,
            wind_speed_jitter=0.8,
        ),
        StageDefinition(
            name="multi_mark_mid_wind",
            bounds=second_bounds,
            marks=_marks(
                [
                    Mark("m1", (50.0, 0.0), radius=8.0),
                    Mark("finish", (75.0, 25.0), radius=10.0),
                ]
            ),
            mark_generator=_generator(second_bounds, (0.0, 0.0)),
            wind=map_wind,
            wind_schedule=map_wind_schedule,
            wind_speed_range=(4.0, 8.0),
            wind_direction_range=(0.0, math.pi / 2),
            wind_noise=0.02,
            penalties=StagePenalties(progress_scale=2.5),
            randomize_heading=False,
            heading_jitter=math.pi / 2,
            randomize_wind=False,
            wind_direction_jitter=math.pi / 6,
            wind_speed_jitter=0.8,
        ),
        StageDefinition(
            name="shifting_tight_bounds",
            bounds=third_bounds,
            marks=_marks(
                [
                    Mark("m1", (40.0, 0.0), radius=7.0),
                    Mark("m2", (65.0, -10.0), radius=7.0),
                    Mark("finish", (80.0, 20.0), radius=8.0),
                ]
            ),
            mark_generator=_generator(third_bounds, (0.0, 0.0)),
            wind=map_wind,
            wind_schedule=map_wind_schedule,
            wind_speed_range=(3.0, 9.0),
            wind_direction_range=(-math.pi / 2, math.pi / 2),
            wind_noise=0.05,
            wind_shift=WindShiftSpec(every_steps=20, direction_std=0.35, speed_std=0.1),
            penalties=StagePenalties(out_of_bounds_penalty=3.0, tack_penalty=0.0, oscillation_penalty=0.0),
            randomize_heading=False,
            heading_jitter=math.pi / 2,
            randomize_wind=False,
            wind_direction_jitter=math.pi / 4,
            wind_speed_jitter=1.0,
        ),
    ]


class MetricsLogger(neat.reporting.BaseReporter):
    def __init__(
        self,
        csv_path: Path,
        json_path: Path,
        stats: neat.StatisticsReporter,
        curriculum: CurriculumManager,
        stage_csv_path: Path,
        stage_json_path: Path,
    ) -> None:
        self.csv_path = csv_path
        self.json_path = json_path
        self.stats = stats
        self.curriculum = curriculum
        self.stage_csv_path = stage_csv_path
        self.stage_json_path = stage_json_path
        self.current_generation = -1

        os.makedirs(self.csv_path.parent, exist_ok=True)
        with open(self.csv_path, "w", encoding="utf-8") as csv_file:
            csv_file.write("generation,best,mean,stdev,min,max\n")

        os.makedirs(self.json_path.parent, exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as json_file:
            json.dump([], json_file)

        with open(self.stage_csv_path, "w", encoding="utf-8") as csv_file:
            csv_file.write("generation,stage,stage_index,success_rate,finishes,episodes\n")

        with open(self.stage_json_path, "w", encoding="utf-8") as json_file:
            json.dump([], json_file)

    def start_generation(self, generation: int) -> None:
        self.current_generation = generation

    def post_evaluate(self, config: neat.Config, population: Dict[int, neat.DefaultGenome], species, best_genome: neat.DefaultGenome) -> None:
        fitness_values = [g.fitness for g in population.values() if g.fitness is not None]
        if not fitness_values:
            return

        mean = statistics.fmean(fitness_values)
        stdev = statistics.pstdev(fitness_values)
        entry = {
            "generation": self.current_generation,
            "best": max(fitness_values),
            "mean": mean,
            "stdev": stdev,
            "min": min(fitness_values),
            "max": max(fitness_values),
        }

        with open(self.csv_path, "a", encoding="utf-8") as csv_file:
            csv_file.write(
                f"{entry['generation']},{entry['best']},{entry['mean']},{entry['stdev']},{entry['min']},{entry['max']}\n"
            )

        with open(self.json_path, "r+", encoding="utf-8") as json_file:
            data = json.load(json_file)
            data.append(entry)
            json_file.seek(0)
            json.dump(data, json_file, indent=2)
            json_file.truncate()

        if self.curriculum.last_stage_stats:
            stage_entry = self.curriculum.last_stage_stats
            with open(self.stage_csv_path, "a", encoding="utf-8") as csv_file:
                csv_file.write(
                    f"{stage_entry.generation},{stage_entry.stage_name},{stage_entry.stage_index},{stage_entry.success_rate},{stage_entry.finishes},{stage_entry.episodes}\n"
                )

            with open(self.stage_json_path, "r+", encoding="utf-8") as json_file:
                stage_data = json.load(json_file)
                stage_data.append(
                    {
                        "generation": stage_entry.generation,
                        "stage": stage_entry.stage_name,
                        "stage_index": stage_entry.stage_index,
                        "success_rate": stage_entry.success_rate,
                        "finishes": stage_entry.finishes,
                        "episodes": stage_entry.episodes,
                    }
                )
                json_file.seek(0)
                json.dump(stage_data, json_file, indent=2)
                json_file.truncate()


class SnapshotReporter(neat.reporting.BaseReporter):
    def __init__(
        self,
        output_dir: Path,
        interval: int,
        fitness_params: FitnessParameters,
        curriculum: CurriculumManager,
        polar: PolarDiagram | None = None,
        stage_mode: str = "current",
        min_finish_improvement: int = 10,
        min_fitness_improvement: float = 0.5,
        max_interval: int = 0,
    ) -> None:
        if interval < 1:
            raise ValueError("snapshot interval must be at least 1")
        self.output_dir = output_dir
        self.interval = interval
        self.fitness_params = fitness_params
        self.curriculum = curriculum
        self.polar = polar
        self.stage_mode = stage_mode
        self.min_finish_improvement = max(1, min_finish_improvement)
        self.min_fitness_improvement = max(0.0, min_fitness_improvement)
        self.max_interval = max(0, max_interval)
        self.current_generation = -1
        self.best_finish_steps: int | None = None
        self.best_fitness: float | None = None
        self.last_saved_generation: int | None = None

    def start_generation(self, generation: int) -> None:
        self.current_generation = generation

    def post_evaluate(
        self,
        config: neat.Config,
        population: Dict[int, neat.DefaultGenome],
        species,
        best_genome: neat.DefaultGenome,
    ) -> None:
        if self.current_generation < 0 or self.current_generation % self.interval != 0:
            return

        stage, stage_index = _select_stage(self.curriculum, self.stage_mode)

        stage_fitness = self.fitness_params.apply_overrides(stage.penalties)
        seeds = stage.seeds or stage_fitness.seeds

        results = [
            _evaluate_episode(best_genome, config, seed, stage_fitness, stage, polar=self.polar)
            for seed in seeds
        ]
        finish_steps = [result.steps for result in results if result.finished]
        should_save = False
        forced_save = False
        best_fitness = best_genome.fitness if best_genome.fitness is not None else None

        if finish_steps:
            finish_step = min(finish_steps)
            if self.best_finish_steps is None:
                should_save = True
            elif self.best_finish_steps - finish_step >= self.min_finish_improvement:
                should_save = True
            if should_save:
                self.best_finish_steps = finish_step
                if best_fitness is not None and (self.best_fitness is None or best_fitness > self.best_fitness):
                    self.best_fitness = best_fitness
        else:
            if best_fitness is not None:
                if self.best_fitness is None or best_fitness >= self.best_fitness + self.min_fitness_improvement:
                    should_save = True
                    self.best_fitness = best_fitness
        if not should_save and self.max_interval > 0:
            if self.last_saved_generation is None:
                should_save = True
                forced_save = True
            elif self.current_generation - self.last_saved_generation >= self.max_interval:
                should_save = True
                forced_save = True
        if not should_save:
            return

        snapshot_dir = self.output_dir / f"gen_{self.current_generation:04d}"
        traces_dir = snapshot_dir / "traces"
        _save_genome(best_genome, snapshot_dir / "champion.pkl")
        _record_traces(best_genome, config, stage_fitness, traces_dir, stage, seeds, polar=self.polar)

        meta = {
            "generation": self.current_generation,
            "stage": stage.name,
            "stage_index": stage_index,
            "seeds": list(seeds),
            "best_fitness": best_genome.fitness,
            "finished": bool(finish_steps),
            "finish_steps": min(finish_steps) if finish_steps else None,
            "forced": forced_save,
        }
        os.makedirs(snapshot_dir, exist_ok=True)
        with open(snapshot_dir / "meta.json", "w", encoding="utf-8") as meta_file:
            json.dump(meta, meta_file, indent=2)
        self.last_saved_generation = self.current_generation


class GenerationTraceReporter(neat.reporting.BaseReporter):
    def __init__(
        self,
        output_dir: Path,
        fitness_params: FitnessParameters,
        curriculum: CurriculumManager,
        polar: PolarDiagram | None = None,
        stage_mode: str = "current",
    ) -> None:
        self.output_dir = output_dir
        self.fitness_params = fitness_params
        self.curriculum = curriculum
        self.polar = polar
        self.stage_mode = stage_mode
        self.current_generation = -1

    def start_generation(self, generation: int) -> None:
        self.current_generation = generation

    def post_evaluate(
        self,
        config: neat.Config,
        population: Dict[int, neat.DefaultGenome],
        species,
        best_genome: neat.DefaultGenome,
    ) -> None:
        if self.current_generation < 0:
            return

        stage, stage_index = _select_stage(self.curriculum, self.stage_mode)
        stage_fitness = self.fitness_params.apply_overrides(stage.penalties)
        seeds = stage.seeds or stage_fitness.seeds

        snapshot_dir = self.output_dir / f"gen_{self.current_generation:04d}"
        traces_dir = snapshot_dir / "traces"
        _save_genome(best_genome, snapshot_dir / "champion.pkl")
        results = _record_traces(best_genome, config, stage_fitness, traces_dir, stage, seeds, polar=self.polar)
        finish_steps = [result.steps for result in results if result.finished]

        meta = {
            "generation": self.current_generation,
            "stage": stage.name,
            "stage_index": stage_index,
            "seeds": list(seeds),
            "best_fitness": best_genome.fitness,
            "finished": bool(finish_steps),
            "finish_steps": min(finish_steps) if finish_steps else None,
        }
        os.makedirs(snapshot_dir, exist_ok=True)
        with open(snapshot_dir / "meta.json", "w", encoding="utf-8") as meta_file:
            json.dump(meta, meta_file, indent=2)


def _select_stage(curriculum: CurriculumManager, stage_mode: str) -> tuple[StageDefinition, int]:
    if stage_mode == "final":
        stage = curriculum.stages[-1]
        stage_index = len(curriculum.stages) - 1
    elif curriculum.last_stage_stats:
        stage_index = curriculum.last_stage_stats.stage_index
        stage = curriculum.stages[stage_index]
    else:
        stage_index = curriculum.stage_index
        stage = curriculum.current_stage
    return stage, stage_index


def _save_genome(genome: neat.DefaultGenome, path: Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "wb") as fp:
        pickle.dump(genome, fp)


def _record_traces(
    genome: neat.DefaultGenome,
    config: neat.Config,
    fitness_params: FitnessParameters,
    trace_dir: Path,
    stage: StageDefinition,
    seeds: Sequence[int],
    polar: PolarDiagram | None = None,
) -> List[EpisodeResult]:
    trace_dir.mkdir(parents=True, exist_ok=True)
    results: List[EpisodeResult] = []
    for seed in seeds:
        trace_path = trace_dir / f"champion_seed_{seed}.jsonl"
        results.append(
            _evaluate_episode(genome, config, seed, fitness_params, stage, polar=polar, trace_path=trace_path)
        )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("train/configs/neat_config.txt"), help="path to NEAT config file")
    parser.add_argument("--output", type=Path, default=Path("data/neat_runs/latest"), help="output directory for logs and artifacts")
    parser.add_argument("--generations", type=int, default=30, help="number of generations to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="seeds used for multi-episode evaluation")
    parser.add_argument("--max-steps", type=int, default=400, help="maximum steps per episode")
    parser.add_argument("--finish-bonus", type=float, default=5.0, help="fitness bonus for completing the course")
    parser.add_argument("--progress-scale", type=float, default=2.0, help="scale applied to progress-based fitness")
    parser.add_argument(
        "--goal-progress-scale",
        type=float,
        default=0.2,
        help="scale applied to progress toward the final mark",
    )
    parser.add_argument(
        "--vmg-scale",
        type=float,
        default=0.3,
        help="scale applied to positive velocity made good toward the final mark",
    )
    parser.add_argument(
        "--stall-speed-threshold",
        type=float,
        default=0.5,
        help="speed below which a stall penalty applies (m/s)",
    )
    parser.add_argument(
        "--stall-penalty",
        type=float,
        default=0.01,
        help="penalty applied per step when below the stall speed threshold",
    )
    parser.add_argument(
        "--finish-time-scale",
        type=float,
        default=1.0,
        help="bonus scale for finishing earlier within the step budget",
    )
    parser.add_argument(
        "--mark-time-scale",
        type=float,
        default=0.5,
        help="bonus scale for reaching intermediate marks quickly",
    )
    parser.add_argument(
        "--step-penalty",
        type=float,
        default=0.002,
        help="penalty applied per step regardless of outcome",
    )
    parser.add_argument(
        "--turn-rate-penalty",
        type=float,
        default=0.02,
        help="penalty applied per step based on turn-rate magnitude",
    )
    parser.add_argument("--out-of-bounds-penalty", type=float, default=2.0, help="penalty when a rollout exits bounds")
    parser.add_argument("--tack-penalty", type=float, default=0.0, help="penalty per tack-like oscillation")
    parser.add_argument("--oscillation-penalty", type=float, default=0.0, help="penalty per action sign change")
    parser.add_argument(
        "--curriculum-mode",
        choices=["cycle", "progress"],
        default="progress",
        help="cycle through stages each generation or progress once a threshold is met",
    )
    parser.add_argument(
        "--stage-success-threshold",
        type=float,
        default=0.6,
        help="success rate required before advancing stages in progress mode",
    )
    parser.add_argument(
        "--map-file",
        type=Path,
        default=Path("data/maps/bigmap.json"),
        help="Path to a JSON map file with mark coordinates",
    )
    parser.add_argument(
        "--mark-count",
        type=int,
        default=None,
        help="number of generated marks; when omitted default fixed courses are used",
    )
    parser.add_argument(
        "--mark-min-distance",
        type=float,
        default=40.0,
        help="minimum radial distance used when generating marks",
    )
    parser.add_argument(
        "--mark-max-distance",
        type=float,
        default=90.0,
        help="maximum radial distance used when generating marks",
    )
    parser.add_argument(
        "--mark-radius",
        type=float,
        default=8.0,
        help="radius applied to generated marks",
    )
    parser.add_argument(
        "--mark-angle-jitter",
        type=float,
        default=math.pi / 6,
        help="angular jitter in radians around evenly spaced bearings",
    )
    parser.add_argument(
        "--map-bounds-padding",
        type=float,
        default=25.0,
        help="Padding applied around map coordinates when inferring bounds",
    )
    parser.add_argument("--polar-file", type=Path, default=None, help="path to a JSON polar file")
    parser.add_argument("--snapshot-interval", type=int, default=0, help="save a champion snapshot every N generations")
    parser.add_argument(
        "--snapshot-stage",
        choices=["current", "final"],
        default="current",
        help="stage used when recording snapshot traces",
    )
    parser.add_argument(
        "--snapshot-min-finish-improvement",
        type=int,
        default=20,
        help="minimum step improvement required to keep a finished snapshot",
    )
    parser.add_argument(
        "--snapshot-min-fitness-improvement",
        type=float,
        default=0.5,
        help="minimum fitness improvement required to keep an unfinished snapshot",
    )
    parser.add_argument(
        "--snapshot-max-interval",
        type=int,
        default=0,
        help="force saving a snapshot if none were kept for N generations",
    )
    parser.add_argument(
        "--generation-traces",
        action="store_true",
        help="record champion traces every generation for smoother visualization",
    )
    parser.add_argument(
        "--generation-traces-dir",
        type=Path,
        default=None,
        help="output directory for per-generation traces (defaults to <output>/generation_traces)",
    )
    parser.add_argument(
        "--generation-traces-stage",
        choices=["current", "final"],
        default="current",
        help="stage used when recording per-generation traces",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mark_count is not None and args.mark_count < 1:
        raise ValueError("--mark-count must be at least 1 when provided")
    if args.map_file and args.mark_count is not None:
        raise ValueError("Use either --map-file or --mark-count, not both")

    map_marks: Sequence[Mark] | None = None
    map_bounds: Tuple[float, float, float, float] | None = None
    map_wind: WindState | None = None
    map_wind_schedule: Sequence[WindUpdate] | None = None
    polar: PolarDiagram | None = None
    if args.map_file:
        map_config = load_map(args.map_file, default_radius=args.mark_radius)
        map_marks = map_config.marks
        map_bounds = map_config.bounds
        map_wind = map_config.wind
        map_wind_schedule = map_config.wind_schedule
        polar = map_config.polar
        if map_bounds is None:
            map_bounds = infer_bounds(map_marks, padding=args.map_bounds_padding)
    if args.polar_file:
        polar = PolarDiagram.load(args.polar_file)
    elif polar is None:
        default_polar = Path("data/polars/sample.json")
        if default_polar.exists():
            polar = PolarDiagram.load(default_polar)
    fitness_params = FitnessParameters(
        max_steps=args.max_steps,
        seeds=args.seeds,
        progress_scale=args.progress_scale,
        goal_progress_scale=args.goal_progress_scale,
        vmg_scale=args.vmg_scale,
        stall_speed_threshold=args.stall_speed_threshold,
        stall_penalty=args.stall_penalty,
        finish_time_scale=args.finish_time_scale,
        mark_time_scale=args.mark_time_scale,
        step_penalty=args.step_penalty,
        turn_rate_penalty=args.turn_rate_penalty,
        finish_bonus=args.finish_bonus,
        out_of_bounds_penalty=args.out_of_bounds_penalty,
        tack_penalty=args.tack_penalty,
        oscillation_penalty=args.oscillation_penalty,
    )

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(args.config),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    csv_log = args.output / "fitness_log.csv"
    json_log = args.output / "fitness_log.json"

    stage_log = args.output / "stage_log.csv"
    stage_json_log = args.output / "stage_log.json"

    map_spec = (
        GoalMapSpec(
            count=args.mark_count,
            min_distance=args.mark_min_distance,
            max_distance=args.mark_max_distance,
            radius=args.mark_radius,
            angle_jitter=args.mark_angle_jitter,
        )
        if args.mark_count
        else None
    )

    curriculum = CurriculumManager(
        _build_default_curriculum(
            args.mark_count,
            map_spec,
            map_marks,
            map_bounds,
            map_wind,
            map_wind_schedule,
            args.map_bounds_padding,
        ),
        args.curriculum_mode,
        args.stage_success_threshold,
    )
    stage_evaluator = StageEvaluator(fitness_params, curriculum, polar=polar)

    stats = neat.StatisticsReporter()
    population = neat.Population(config)
    population.add_reporter(stats)
    population.add_reporter(neat.StdOutReporter(True))
    metrics_logger = MetricsLogger(csv_log, json_log, stats, curriculum, stage_log, stage_json_log)
    population.add_reporter(metrics_logger)
    if args.snapshot_interval > 0:
        snapshot_dir = args.output / "snapshots"
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        snapshot_reporter = SnapshotReporter(
            snapshot_dir,
            args.snapshot_interval,
            fitness_params,
            curriculum,
            polar=polar,
            stage_mode=args.snapshot_stage,
            min_finish_improvement=args.snapshot_min_finish_improvement,
            min_fitness_improvement=args.snapshot_min_fitness_improvement,
            max_interval=args.snapshot_max_interval,
        )
        population.add_reporter(snapshot_reporter)
    if args.generation_traces:
        generation_traces_dir = args.generation_traces_dir or (args.output / "generation_traces")
        if generation_traces_dir.exists():
            shutil.rmtree(generation_traces_dir)
        generation_reporter = GenerationTraceReporter(
            generation_traces_dir,
            fitness_params,
            curriculum,
            polar=polar,
            stage_mode=args.generation_traces_stage,
        )
        population.add_reporter(generation_reporter)

    def evaluation_callback(genomes: Iterable[tuple[int, neat.DefaultGenome]], neat_config: neat.Config) -> None:
        generation = metrics_logger.current_generation if metrics_logger.current_generation >= 0 else 0
        stage_evaluator.evaluate_genomes(genomes, neat_config, generation)

    winner = population.run(evaluation_callback, n=args.generations)

    genome_path = args.output / "champion.pkl"
    _save_genome(winner, genome_path)

    trace_dir = args.output / "traces"
    trace_stage = curriculum.stages[curriculum.last_stage_stats.stage_index] if curriculum.last_stage_stats else curriculum.current_stage
    trace_fitness = fitness_params.apply_overrides(trace_stage.penalties)
    trace_seeds = trace_stage.seeds or trace_fitness.seeds
    _record_traces(winner, config, trace_fitness, trace_dir, trace_stage, trace_seeds, polar=polar)

    print(f"Saved champion genome to {genome_path}")
    print(f"Saved traces to {trace_dir}")


if __name__ == "__main__":
    main()
