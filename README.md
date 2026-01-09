# sailing-sim

Lightweight sailing environment with a simple boat model, observation builder,
and baseline controller. It is a work in progress. 

![Training evolution](assets/training_evolution.gif)

## Setup

- **Python**: 3.10+ .
- **Dependencies** (pinned in `requirements.txt`):
  - `pytest==8.2.0`
  - `neat-python==0.92`
  - `matplotlib>3.8.4`

Create and activate a virtual environment, then install the pinned packages:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The simulator writes traces and training outputs under `data/`; ensure that
directory is writable in your environment.

## Running a demo episode

Roll out the deterministic heading-hold controller and emit a JSONL trace for
validation:

```bash
python -m sim.run_episode --steps 400 --trace data/traces/example.jsonl
```

Use ``--mark-count`` plus the ``--mark-*-distance`` options to procedurally
generate a new course layout for the simulation run (e.g.,
``--mark-count 3 --mark-min-distance 30 --mark-max-distance 80``).

### Custom maps

You can also drive the simulator with an explicit race definition file instead
of procedural generation. A race JSON file describes the mark coordinates,
optional bounds, and optional wind configuration. This repository ships with a
ready-to-use six-mark example at ``data/maps/default.json``:

```json
{
  "marks": [
    {"name": "m1", "position": [0.0, 0.0], "radius": 8},
    {"name": "m2", "position": [60.0, -20.0], "radius": 8},
    {"name": "m3", "position": [120.0, -10.0], "radius": 8},
    {"name": "m4", "position": [180.0, 15.0], "radius": 8},
    {"name": "m5", "position": [240.0, -5.0], "radius": 8},
    {"name": "finish", "position": [40.0, 20.0], "radius": 10}
  ],
  "bounds": [-40.0, 340.0, -80.0, 80.0],
  "wind": {"speed": 6.0, "direction_deg": 45.0},
  "wind_schedule": [
    {"step": 150, "direction_deg": 90.0},
    {"step": 300, "direction_deg": 135.0, "speed": 7.5}
  ]
}
```

If ``bounds`` are omitted they are inferred from the mark coordinates with a
padding margin. Names default to ``m1`` ... ``finish`` and missing radii fall
back to ``--mark-radius`` / ``--map-default-radius``. Wind is optional: ``wind``
sets the initial true-wind vector (using degrees for readability) and
``wind_schedule`` applies step-indexed updates during the episode. When a
schedule is provided only the specified fields (speed, direction) are changed
at each step. Step indices are 1-based, matching the environment step counter.

- **Simulate/test**: ``python -m sim.run_episode --map-file data/maps/default.json --map-bounds-padding 30``
- **Train**: ``python -m train.neat_train --map-file data/maps/default.json --map-bounds-padding 30``
- **Visualize**: ``python -m viz.replay data/traces/example.jsonl --map-file data/maps/default.json``

Expected artifacts and outputs:

- Console summary with the terminal status (e.g., `done=True` reason and mark
  index completed).
- JSONL trace at `data/traces/example.jsonl` with per-step state, wind,
  mark index, and applied actions.

Use `--help` for customization (start position, heading, random seed).

## Training (NEAT curriculum)

Train a NEAT controller using the bundled curriculum and fitness shaping:

```bash
python -m train.neat_train --generations 30 --output data/neat_runs/latest
```

Key artifacts written to the output directory:

- `fitness_log.csv` / `fitness_log.json`: per-generation population metrics.
- `stage_log.csv` / `stage_log.json`: curriculum stage transitions and success
  rates.
- `champion.pkl`: pickled winning genome after training.
- `traces/`: JSONL rollouts of the champion on the final stage.
- `snapshots/` (optional): periodic generation checkpoints with traces.

Adjust hyperparameters with flags such as `--max-steps`, `--seeds`,
`--finish-bonus`, `--goal-progress-scale`, `--vmg-scale`, `--stall-penalty`,
`--finish-time-scale`, `--step-penalty`, `--turn-rate-penalty`, and `--curriculum-mode`. To evaluate
across a configurable number of marks, add `--mark-count` along with the mark
distance and radius generation parameters. The default NEAT configuration lives in
`train/configs/neat_config.txt`.

Training stages randomize the start heading and wind slightly by default to
encourage tacking and reduce one-sided steering biases.

To record snapshots every N generations, add `--snapshot-interval`. For example:

```bash
python -m train.neat_train --generations 200 --snapshot-interval 50 --output data/neat_runs/latest
```

When snapshots are enabled, the existing `snapshots/` directory is cleared at
the start of the run. You can also set thresholds to keep only significant
updates (finish-time improvements are prioritized), and force a save if no
snapshots were kept for a while:

```bash
python -m train.neat_train --snapshot-interval 50 --snapshot-min-finish-improvement 20 --snapshot-min-fitness-improvement 0.5 --snapshot-max-interval 200
```

For smooth per-generation playback, record every generation's champion traces:

```bash
python -m train.neat_train --generation-traces --output data/neat_runs/latest
```

The simulator uses a polar diagram to convert true wind angle/speed into boat
speed. A sample polar lives at `data/polars/sample.json` and is loaded by
default; override it with `--polar-file` or by adding `"polar"` to a map file.
The boat model also includes a small wind push so stalled boats drift with the
wind (see `sim/boat.py`).

## Visualization and export

Replay a saved trace with a simple matplotlib animation:

```bash
python -m viz.replay data/traces/example.jsonl --interval-ms 150
```

The viewer highlights the active mark, boat heading, path, and wind arrow over
time. For non-interactive environments, set `MPLBACKEND=Agg` and add a
`plt.savefig(...)` call inside `viz/replay.py` to export a static frame.

Visualize a polar diagram:

```bash
python -m viz.polar_viz data/polars/sample.json
```

Compare snapshot traces side by side:

```bash
python -m viz.compare_replays data/neat_runs/latest/snapshots --seed 0 --cols 3
```

Animate champion courses across generations (one full path per frame):

```bash
python -m viz.generation_replay data/neat_runs/latest/snapshots --seed 0
```

Use per-generation traces for a fluid animation:

```bash
python -m viz.generation_replay data/neat_runs/latest/generation_traces --seed 0
```

Override marks with a map file:

```bash
python -m viz.compare_replays data/neat_runs/latest/snapshots --seed 0 --cols 3 --map-file data/maps/default.json
```

Limit to specific generations:

```bash
python -m viz.compare_replays data/neat_runs/latest/snapshots --seed 0 --runs 100,200,300
```

## Interface contract

### Actions
- Controllers should emit a ``turn_rate`` command in **radians/second**.
- Use :class:`sim.actions.TurnAction` or a raw dict with ``{"turn_rate": value}``.
- The environment clamps the command to the boat model's ``max_turn_rate`` before
  integration.

### Observations
Built via :class:`sim.observation.ObservationBuilder` (accessible from
``SailingEnv.observation()``). Fields:

- ``target_dx``: target displacement along the boat's forward axis (meters).
- ``target_dy``: target displacement to port (meters).
- ``apparent_wind_dx``: apparent wind x-component in the boat frame (m/s).
- ``apparent_wind_dy``: apparent wind y-component in the boat frame (m/s).
- ``boat_speed``: predicted forward speed from the current heading and wind.

### Boat model and wind behavior

- :class:`sim.boat.BoatModelV1` uses a simple polar: speed scales with the sine
  of the true-wind angle off the bow and drops to **zero inside a configurable
  no-go zone** (40° by default). The model also applies a quadratic turn-rate
  penalty with an extra high-order term so aggressive maneuvers reduce forward
  speed more sharply. It also applies an extra penalty for sharp course changes,
  and includes momentum via a speed time constant (3s default) so speed decays
  smoothly toward the polar target instead of snapping to it each step.
- Deterministic winds from a map file are respected by default. The environment
  no longer adds per-step wind-direction noise unless you explicitly provide a
  non-zero ``wind_noise`` value when constructing :class:`sim.env.SailingEnv`.

## Troubleshooting

- **Import errors for matplotlib**: ensure the pinned dependencies are
  installed (`pip install -r requirements.txt`).
- **Permission issues writing traces**: verify that the `data/` directory
  exists and is writable (`mkdir -p data/traces`).
- **Python version mismatches**: typing features require Python 3.10+; create a
  fresh virtual environment with a compatible interpreter.

## Example training command with snapshots
```bash
python -m train.neat_train \
  --generations 1000 \
  --map-file data/maps/bigmap.json \
  --max-steps 4000 \
  --snapshot-interval 50 \
  --snapshot-min-finish-improvement 20 \
  --snapshot-min-fitness-improvement 0.5 \
  --snapshot-max-interval 500 \
  --output data/neat_runs/latest

python -m viz.compare_replays data/neat_runs/latest/snapshots --seed 0 --cols 3 --map-file data/maps/bigmap.json --interval-ms 50
```
