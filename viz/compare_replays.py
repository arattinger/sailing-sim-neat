"""Compare multiple trace files side by side."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
from matplotlib import animation, patches

from sim.map_io import load_map
from viz.replay import TraceData, TraceRecord, _boat_triangle, _calc_bounds, load_trace


@dataclass
class TraceView:
    records: List[TraceRecord]
    marks: Sequence
    title: str
    ax: plt.Axes
    path_line: plt.Line2D
    boat_patch: patches.Polygon
    mark_patches: List[patches.Circle]
    title_text: plt.Text
    wind_quiver: plt.Quiver
    wind_label: plt.Text
    wind_anchor: tuple[float, float]


def _prepare_view(
    ax: plt.Axes,
    trace: TraceData,
    title: str,
    marks_override: Sequence | None = None,
) -> TraceView:
    records = trace.records
    marks = marks_override if marks_override is not None else (trace.marks or [])
    bounds = _calc_bounds(records, marks) if marks else _calc_bounds(records, [])

    ax.set_facecolor("#8ecae6")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])

    mark_patches: List[patches.Circle] = []
    for mark in marks:
        patch = patches.Circle(
            mark.position, mark.radius, facecolor="#ffd166", edgecolor="#f3722c", alpha=0.7
        )
        ax.add_patch(patch)
        mark_patches.append(patch)

    path_line, = ax.plot([], [], color="#1d3557", linewidth=2)
    initial_vertices = _boat_triangle(records[0].position, records[0].heading)
    boat_patch = patches.Polygon(
        initial_vertices, closed=True, facecolor="#ff7f51", edgecolor="#d1495b"
    )
    ax.add_patch(boat_patch)
    title_text = ax.text(0.02, 0.95, title, transform=ax.transAxes, ha="left", va="top")
    wind_anchor = (bounds[0] + 0.35 * (bounds[1] - bounds[0]), bounds[3] - 0.15 * (bounds[3] - bounds[2]))
    wind_quiver = ax.quiver([], [], [], [], color="#0d3b66", width=0.005, scale=1)
    wind_label = ax.text(
        wind_anchor[0],
        wind_anchor[1],
        "",
        ha="left",
        va="top",
        color="#0d3b66",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )
    return TraceView(
        records=records,
        marks=marks,
        title=title,
        ax=ax,
        path_line=path_line,
        boat_patch=boat_patch,
        mark_patches=mark_patches,
        title_text=title_text,
        wind_quiver=wind_quiver,
        wind_label=wind_label,
        wind_anchor=wind_anchor,
    )


def _update_mark_highlight(view: TraceView, mark_index: int) -> None:
    for idx, patch in enumerate(view.mark_patches):
        if idx == mark_index:
            patch.set_edgecolor("red")
            patch.set_linewidth(2.5)
            patch.set_alpha(0.9)
        else:
            patch.set_edgecolor("#f3722c")
            patch.set_linewidth(1.0)
            patch.set_alpha(0.7)


def _update_frame(idx: int, views: List[TraceView]):
    artists = []
    for view in views:
        record_idx = min(idx, len(view.records) - 1)
        record = view.records[record_idx]

        path_x = [r.position[0] for r in view.records[: record_idx + 1]]
        path_y = [r.position[1] for r in view.records[: record_idx + 1]]
        view.path_line.set_data(path_x, path_y)

        vertices = _boat_triangle(record.position, record.heading)
        view.boat_patch.set_xy(vertices)

        if view.mark_patches:
            if record.mark_index < len(view.mark_patches):
                _update_mark_highlight(view, record.mark_index)
            else:
                _update_mark_highlight(view, -1)

        wind_scale = 0.01
        wind_heading = record.wind_direction + math.pi
        u = math.cos(wind_heading) * record.wind_speed * wind_scale
        v = math.sin(wind_heading) * record.wind_speed * wind_scale
        view.wind_quiver.set_offsets([view.wind_anchor])
        view.wind_quiver.set_UVC([u], [v])
        view.wind_label.set_text(f"Wind (to): {record.wind_speed:.1f} m/s")

        artists.extend(
            [
                view.path_line,
                view.boat_patch,
                view.wind_quiver,
                view.wind_label,
                *view.mark_patches,
                view.title_text,
            ]
        )
    return artists


def _discover_traces(snapshots_dir: Path, seed: int, max_items: int | None) -> List[Path]:
    trace_paths = []
    for snapshot_dir in sorted(snapshots_dir.glob("gen_*")):
        trace_path = snapshot_dir / "traces" / f"champion_seed_{seed}.jsonl"
        if trace_path.exists():
            trace_paths.append(trace_path)
    if max_items is not None:
        trace_paths = trace_paths[:max_items]
    return trace_paths


def _parse_runs(value: str) -> List[int]:
    runs: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            runs.append(int(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid run '{part}', expected integers") from exc
    return runs


def _load_meta(trace_path: Path) -> dict:
    meta_path = trace_path.parent.parent / "meta.json"
    if meta_path.exists():
        try:
            import json

            with open(meta_path, "r", encoding="utf-8") as meta_file:
                return json.load(meta_file)
        except Exception:
            return {}
    return {}


def _title_for_snapshot(trace_path: Path, meta: dict) -> str:
    if meta:
        return f"Gen {meta.get('generation', '')} ({meta.get('stage', '')})"
    return trace_path.parent.parent.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots_dir", type=Path, help="Path to snapshot directory with gen_* folders")
    parser.add_argument("--seed", type=int, default=0, help="seed index for trace files")
    parser.add_argument("--cols", type=int, default=3, help="number of columns in the grid")
    parser.add_argument("--interval-ms", type=int, default=200, help="delay between frames in milliseconds")
    parser.add_argument("--max-items", type=int, default=None, help="limit number of snapshots shown")
    parser.add_argument("--runs", type=_parse_runs, default=None, help="comma-separated generation list (e.g. 100,200,300)")
    parser.add_argument("--map-file", type=Path, default=None, help="optional map file to override marks")
    parser.add_argument(
        "--map-default-radius",
        type=float,
        default=8.0,
        help="radius used when map entries omit one",
    )
    args = parser.parse_args()

    trace_paths = _discover_traces(args.snapshots_dir, args.seed, args.max_items)
    if not trace_paths:
        raise SystemExit(f"No traces found under {args.snapshots_dir}")

    entries = []
    for path in trace_paths:
        meta = _load_meta(path)
        entries.append({"path": path, "meta": meta})

    if args.runs:
        run_set = set(args.runs)
        filtered = []
        for entry in entries:
            generation = entry["meta"].get("generation")
            if isinstance(generation, int) and generation in run_set:
                filtered.append(entry)
        if not filtered:
            raise SystemExit(f"No snapshots match runs: {sorted(run_set)}")
        entries = filtered

    def _best_key(entry: dict) -> tuple[int, float]:
        finish_steps = entry["meta"].get("finish_steps")
        if isinstance(finish_steps, (int, float)):
            return 0, float(finish_steps)
        fitness = entry["meta"].get("best_fitness")
        if isinstance(fitness, (int, float)):
            return 1, -float(fitness)
        generation = entry["meta"].get("generation")
        if isinstance(generation, int):
            return 2, -float(generation)
        return 3, 0.0

    if len(entries) > 1:
        best_index = min(range(len(entries)), key=lambda idx: _best_key(entries[idx]))
        if best_index != len(entries) - 1:
            entries.append(entries.pop(best_index))

    traces = [load_trace(entry["path"]) for entry in entries]
    titles = [_title_for_snapshot(entry["path"], entry["meta"]) for entry in entries]
    marks_override = None
    if args.map_file:
        race_config = load_map(args.map_file, default_radius=args.map_default_radius)
        marks_override = race_config.marks

    cols = max(1, args.cols)
    rows = math.ceil(len(traces) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if isinstance(axes, plt.Axes):
        axes_list = [axes]
    else:
        axes_list = list(axes.ravel())

    views: List[TraceView] = []
    for idx, trace in enumerate(traces):
        view = _prepare_view(axes_list[idx], trace, titles[idx], marks_override=marks_override)
        views.append(view)

    for ax in axes_list[len(views):]:
        ax.axis("off")

    max_frames = max(len(view.records) for view in views)
    anim = animation.FuncAnimation(
        fig,
        _update_frame,
        frames=max_frames,
        interval=args.interval_ms,
        fargs=(views,),
        blit=False,
        repeat=False,
    )
    _ = anim
    plt.show()


if __name__ == "__main__":
    main()
