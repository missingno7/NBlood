#!/usr/bin/env python3
"""Run and summarize the LLMapper bot against a retail Blood map corpus.

The AGTST suite is the hard correctness gate.  This utility deliberately
treats campaign maps as a soft benchmark: every run is retained, unfinished
maps receive evidence-based stop categories, and reports preserve raw counts
instead of interpreting entered/total sectors as literal completion percent.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


RETAIL_MAP = re.compile(r"^E(?:1|2|3|4|6)M\d+\.MAP$", re.IGNORECASE)
FIELD = re.compile(r"(\w+)=([^\s]+)")


def read_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except (ValueError, TypeError):
            pass
    return rows


def fields(row: dict) -> dict[str, str]:
    return dict(FIELD.findall(str(row.get("detail", ""))))


def integer(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def event_counts(rows: list[dict]) -> collections.Counter:
    return collections.Counter(
        row.get("event") for row in rows if row.get("type") == "event"
    )


def classify_stop(result: str, mode: str, recent: collections.Counter,
                  damage_from_dudes: bool, unresolved_prerequisites: int,
                  seconds_since_progress: int) -> str:
    if result == "COMPLETED":
        return "completed"
    if result == "DIED":
        if damage_from_dudes:
            return "combat_blocked"
        if mode == "nodudes":
            return "environmental_or_scripted_damage"
        return "combat_or_environmental_death"
    if result in ("RUNTIME_ERROR", "CRASHED"):
        return "crashed_or_runtime_error"
    if result == "WALL_TIMEOUT":
        return "wall_clock_performance_limit"
    if result == "LOOP_DETECTED":
        return "navigation_or_goal_loop"
    if result == "TIMEOUT" and seconds_since_progress <= 30:
        return "timeout_while_progressing"
    if unresolved_prerequisites > 0:
        return "prerequisite_unresolved"
    if (recent["interaction_failed"] + recent["interaction_unavailable"]
            + recent["interaction_refused"]):
        return "interaction_unresolved"
    if (recent["nav_route_unavailable"] + recent["route_step_failed"]
            + recent["nav_edge_rejected"] + recent["stationary_deadlock"]):
        return "navigation_dead_end"
    if recent["jump_traversal"] >= 5:
        return "traversal_execution_stall"
    if recent["loop_break"] or recent["objective_budget_exhausted"] >= 3:
        return "exploration_oscillation"
    if result == "TIMEOUT":
        return "timeout_or_efficiency"
    if mode == "nodudes":
        return "unknown_no_progress_nodudes"
    return "unknown_no_progress"


def summarize_run(map_name: str, mode: str, telemetry: Path,
                  trajectory: Path) -> dict:
    rows = read_rows(telemetry)
    path = read_rows(trajectory)
    summary = next((row for row in reversed(rows)
                    if row.get("type") == "summary"), {})
    counts = event_counts(rows)
    end_time = integer(summary.get("game_time"),
                       integer(path[-1].get("game_time")) if path else 0)

    entered = []
    first_entry = {}
    branches = set()
    depths = []
    last_new_sector_time = 0
    for row in rows:
        if row.get("event") != "sector_entered":
            continue
        data = fields(row)
        sector = integer(data.get("sector"), -1)
        if sector < 0:
            continue
        if sector not in first_entry:
            first_entry[sector] = integer(row.get("game_time"))
            last_new_sector_time = integer(row.get("game_time"))
        entered.append(sector)
        depths.append(integer(data.get("depth")))
        if "branch" in data:
            branches.add(data["branch"])

    observed_objects = {}
    active_objective = None
    pickups = collections.Counter()
    pickup_ids = set()
    keys = set()
    interactions = set()
    activated = set()
    objectives = collections.Counter()
    for row in rows:
        event = row.get("event")
        data = fields(row)
        if event == "observed_object":
            sprite = integer(data.get("sprite"), -1)
            observed_objects[sprite] = data
        elif event == "objective_selected":
            active_objective = (integer(data.get("type"), -1),
                                integer(data.get("id"), -1))
            objectives[active_objective] += 1
        elif event in ("objective_completed", "objective_invalidated"):
            if event == "objective_completed" and active_objective \
                    and active_objective[0] in (1, 2):
                pickup_ids.add(active_objective[1])
            active_objective = None
        elif event == "pickup_confirmed_collected" and active_objective:
            pickup_ids.add(active_objective[1])
        elif event in ("acquired_key", "key_acquired"):
            key = integer(data.get("key"), -1)
            if key >= 0:
                keys.add(key)
        elif event == "discovered_interaction":
            interactions.add((data.get("kind"), data.get("id")))
        elif event == "interaction_world_delta":
            activated.add((data.get("kind"), data.get("id")))

    for sprite in pickup_ids:
        data = observed_objects.get(sprite, {})
        category = data.get("category") or data.get("kind") or "unknown"
        pickups[category] += 1

    transitions = collections.Counter()
    last_sector = None
    stationary = 0
    longest_stationary = 0
    for before, after in zip(path, path[1:]):
        sector = after.get("sector")
        if sector is not None and sector != last_sector:
            if last_sector is not None:
                transitions[(last_sector, sector)] += 1
            last_sector = sector
        moved = (before.get("x"), before.get("y"), before.get("z")) != (
            after.get("x"), after.get("y"), after.get("z"))
        stationary = 0 if moved else stationary + max(
            0, integer(after.get("game_time")) - integer(before.get("game_time")))
        longest_stationary = max(longest_stationary, stationary)

    recent = collections.Counter()
    for row in rows:
        if row.get("type") == "event" \
                and integer(row.get("game_time")) >= last_new_sector_time:
            recent[row.get("event")] += 1

    progress_events = {
        "sector_entered", "interaction_world_delta", "pickup_confirmed_collected",
        "key_acquired", "acquired_key", "level_completed",
    }
    last_progress_time = max(
        (integer(row.get("game_time")) for row in rows
         if row.get("event") in progress_events), default=0)
    seconds_since_progress = max(0, end_time - last_progress_time)
    unavailable = counts["action_prerequisite_unavailable"]
    rearmed = sum(integer(fields(row).get("rearmed")) for row in rows
                  if row.get("event") == "deferred_opportunities_rearmed")
    unresolved_prerequisites = max(0, unavailable - rearmed)

    damage_from_dudes = any(
        row.get("event") == "bot_damaged"
        and integer(fields(row).get("source_stat"), -1) == 6
        for row in rows)

    started = next((row for row in rows if row.get("event") == "world_loaded"), {})
    total = integer(summary.get("total_sectors"),
                    integer(fields(started).get("total_sectors")))
    visited = integer(summary.get("visited_sectors"), len(set(entered)))
    observed = integer(summary.get("observed_sectors"), visited)
    result = summary.get("result", "RUNTIME_ERROR")
    repeat_selections = sum(max(0, count - 1) for count in objectives.values())
    hottest = max(transitions.values()) if transitions else 0

    return {
        "map": map_name,
        "mode": mode,
        "result": result,
        "stop_category": classify_stop(
            result, mode, recent, damage_from_dudes,
            unresolved_prerequisites, seconds_since_progress),
        "failure_reason": summary.get("failure_reason", "missing summary"),
        "game_time": end_time,
        "world": {
            "total_sectors": total,
            "sectors_entered": visited,
            "sector_entered_percent": round(100.0 * visited / total, 2) if total else None,
            "sectors_observed": observed,
            "sector_observed_percent": round(100.0 * observed / total, 2) if total else None,
            "regions_reached": len(branches),
            "max_exploration_depth": max(depths) if depths else 0,
            "longest_no_new_sector_seconds": max(0, end_time - last_new_sector_time),
            "seconds_since_meaningful_progress": seconds_since_progress,
        },
        "objects": {
            "items_discovered": sum(1 for data in observed_objects.values()
                                    if data.get("category") not in (None, "none")),
            "items_picked_up": len(pickup_ids),
            "pickups_by_category": dict(sorted(pickups.items())),
            "keys_collected": len(keys),
            "interactions_discovered": len(interactions),
            "interactions_activated": len(activated),
            "enemies_seen": counts["enemy_observed"],
            "enemies_killed": counts["enemy_killed"],
        },
        "navigation": {
            "unique_destinations_attempted": len(objectives),
            "repeated_target_selections": repeat_selections,
            "route_plans": counts["nav_route_selected"],
            "route_plan_failures": counts["nav_route_unavailable"],
            "edge_rejections": counts["nav_edge_rejected"] + counts["route_step_failed"],
            "objective_timeouts": counts["objective_budget_exhausted"],
            "opportunity_suppressions": counts["opportunity_dormant"],
            "loop_breaks": counts["loop_break"],
            "portals_traversed": counts["portal_traversed"],
            "distinct_transitions": len(transitions),
            "hottest_transition_count": hottest,
            "jumps_attempted": counts["jump_input_emitted"],
            "jumps_succeeded": counts["jump_succeeded"],
            "crouches_started": counts["crouch_started"],
            "sprite_support_poses": sum(
                1 for row in rows if row.get("event") == "nav_surface_entered"
                and fields(row).get("support_kind") == "1"),
            "longest_stationary_seconds": longest_stationary,
            "topology_rebuilds": counts["nav_topology_rebuilt"],
            "dynamic_link_refreshes": counts["nav_links_refreshed"],
            "moving_mesh_periods": counts["nav_mesh_in_motion"],
        },
        "stop_signals": dict(sorted(recent.items())),
    }


def flatten(row: dict) -> dict:
    world, objects, nav = row["world"], row["objects"], row["navigation"]
    return {
        "map": row["map"], "mode": row["mode"], "result": row["result"],
        "stop_category": row["stop_category"], "game_time": row["game_time"],
        "total_sectors": world["total_sectors"],
        "sectors_entered": world["sectors_entered"],
        "sector_entered_percent": world["sector_entered_percent"],
        "sectors_observed": world["sectors_observed"],
        "max_depth": world["max_exploration_depth"],
        "items_discovered": objects["items_discovered"],
        "items_picked_up": objects["items_picked_up"],
        "keys": objects["keys_collected"],
        "interactions_discovered": objects["interactions_discovered"],
        "interactions_activated": objects["interactions_activated"],
        "route_plans": nav["route_plans"],
        "route_failures": nav["route_plan_failures"],
        "repeat_selections": nav["repeated_target_selections"],
        "loop_breaks": nav["loop_breaks"],
        "topology_rebuilds": nav["topology_rebuilds"],
        "no_new_sector_seconds": world["longest_no_new_sector_seconds"],
        "wall_seconds": row.get("wall_seconds"),
    }


def aggregate_runs(runs: list[dict]) -> dict:
    aggregate = {}
    for mode in sorted({row["mode"] for row in runs}):
        selected = [row for row in runs if row["mode"] == mode]
        known = [row for row in selected if row["world"]["total_sectors"]]
        entered = sum(row["world"]["sectors_entered"] for row in selected)
        entered_known = sum(row["world"]["sectors_entered"] for row in known)
        total = sum(row["world"]["total_sectors"] for row in known)
        percentages = [row["world"]["sector_entered_percent"] for row in selected
                       if row["world"]["sector_entered_percent"] is not None]
        aggregate[mode] = {
            "runs": len(selected),
            "results": dict(sorted(collections.Counter(
                row["result"] for row in selected).items())),
            "stop_categories": dict(sorted(collections.Counter(
                row["stop_category"] for row in selected).items())),
            "sectors_entered": entered,
            "sectors_entered_with_known_denominators": entered_known,
            "sectors_in_known_denominators": total,
            "weighted_raw_sector_percent": (
                round(100.0 * entered_known / total, 2) if total else None),
            "mean_raw_sector_percent": (
                round(statistics.mean(percentages), 2) if percentages else None),
            "median_raw_sector_percent": (
                round(statistics.median(percentages), 2) if percentages else None),
            "items_picked_up": sum(
                row["objects"]["items_picked_up"] for row in selected),
            "interactions_activated": sum(
                row["objects"]["interactions_activated"] for row in selected),
            "topology_rebuilds": sum(
                row["navigation"]["topology_rebuilds"] for row in selected),
        }
    aggregate["all"] = {
        "runs": len(runs),
        "maps": len({row["map"] for row in runs}),
        "completed": sum(row["result"] == "COMPLETED" for row in runs),
        "wall_clock_limited": sum(row["result"] == "WALL_TIMEOUT" for row in runs),
    }
    return aggregate


def write_report(root: Path, runs: list[dict], config: dict) -> dict:
    report = {
        "schema": 1,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "policy": "retail maps are a soft benchmark; AGTST is the hard gate",
        "config": config,
        "aggregate": aggregate_runs(runs),
        "runs": sorted(runs, key=lambda row: (row["map"], row["mode"])),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    flat = [flatten(row) for row in report["runs"]]
    if flat:
        with (root / "summary.csv").open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    lines = [
        "# LLMapper Blood campaign corpus", "",
        "Retail maps are a soft behavioral benchmark; sector coverage is a raw",
        "entered/total geometry ratio, not literal level-completion percentage.", "",
        "## Aggregate", "",
        "| Mode | Runs | Entered / known sectors | Weighted % | Mean % | Median % | Pickups | Interactions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in sorted(key for key in report["aggregate"] if key != "all"):
        item = report["aggregate"][mode]
        lines.append(
            "| {mode} | {runs} | {entered}/{total} | {weighted} | {mean} | "
            "{median} | {pickups} | {interactions} |".format(
                mode=mode, runs=item["runs"],
                entered=item["sectors_entered_with_known_denominators"],
                total=item["sectors_in_known_denominators"],
                weighted=item["weighted_raw_sector_percent"],
                mean=item["mean_raw_sector_percent"],
                median=item["median_raw_sector_percent"],
                pickups=item["items_picked_up"],
                interactions=item["interactions_activated"]))
    lines.extend([
        "", "## Per-map results", "",
        "| Map | Mode | Result | Stop category | Time | Sectors | % | Items | Keys | Interactions |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["runs"]:
        world, objects = row["world"], row["objects"]
        lines.append(
            "| {map} | {mode} | {result} | {stop_category} | {game_time} | "
            "{entered}/{total} | {percent} | {items} | {keys} | {activated}/{known} |".format(
                map=row["map"], mode=row["mode"], result=row["result"],
                stop_category=row["stop_category"], game_time=row["game_time"],
                entered=world["sectors_entered"], total=world["total_sectors"],
                percent=world["sector_entered_percent"],
                items=objects["items_picked_up"], keys=objects["keys_collected"],
                activated=objects["interactions_activated"],
                known=objects["interactions_discovered"]))
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def corpus_maps(directory: Path, pattern: str) -> list[Path]:
    matcher = re.compile(pattern, re.IGNORECASE) if pattern else RETAIL_MAP
    return sorted((path for path in directory.iterdir()
                   if path.is_file() and path.suffix.lower() == ".map"
                   and matcher.fullmatch(path.name)), key=lambda path: path.name)


def run_command(args) -> int:
    exe, maps_dir, game_dir, output = map(
        Path, (args.exe, args.maps_dir, args.game_dir, args.output))
    exe, maps_dir, game_dir, output = (
        path.resolve() for path in (exe, maps_dir, game_dir, output))
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if any(mode not in ("normal", "nodudes") for mode in modes):
        print("unsupported mode list: %s" % ",".join(modes), file=sys.stderr)
        return 2
    if not exe.exists() or not maps_dir.is_dir() or not game_dir.is_dir():
        print("missing executable, game directory, or matching maps", file=sys.stderr)
        return 2
    maps = corpus_maps(maps_dir, args.pattern)
    if not maps:
        print("no matching maps", file=sys.stderr)
        return 2
    config = {
        "exe": str(exe), "maps_dir": str(maps_dir), "game_dir": str(game_dir),
        "difficulty": args.difficulty, "timeout": args.timeout,
        "stall": args.stall, "wall_timeout": args.wall_timeout, "modes": modes,
    }
    runs = []
    for level in maps:
        for mode in modes:
            run_dir = output / "runs" / mode / level.stem.upper()
            telemetry = run_dir / "telemetry.ndjson"
            trajectory = run_dir / "trajectory.ndjson"
            metrics = run_dir / "metrics.json"
            if args.resume and metrics.exists():
                runs.append(json.loads(metrics.read_text(encoding="utf-8")))
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [str(exe), "-usecwd", "-nosetup", "-noautoload",
                       "-s", str(args.difficulty), "-map", str(level), "-bot",
                       "-bot_timeout", str(args.timeout), "-bot_stall", str(args.stall),
                       "-bot_telemetry", str(telemetry),
                       "-bot_trajectory", str(trajectory)]
            if mode == "nodudes":
                command.extend(["-nodudes", "1"])
            print("%-7s %-8s" % (level.stem.upper(), mode), flush=True)
            wall_limited = False
            wall_started = time.monotonic()
            with (run_dir / "process.log").open("w", encoding="utf-8") as log:
                try:
                    completed = subprocess.run(
                        command, cwd=game_dir, stdout=log,
                        stderr=subprocess.STDOUT, check=False,
                        timeout=args.wall_timeout if args.wall_timeout > 0 else None)
                    exit_code = completed.returncode
                except subprocess.TimeoutExpired:
                    wall_limited = True
                    exit_code = -1
            row = summarize_run(level.stem.upper(), mode, telemetry, trajectory)
            row["wall_seconds"] = round(time.monotonic() - wall_started, 3)
            if wall_limited:
                row["result"] = "WALL_TIMEOUT"
                row["stop_category"] = "wall_clock_performance_limit"
                row["failure_reason"] = (
                    "corpus wall-clock safety limit reached; bot simulated-time "
                    "limit was not reached")
            row["process_exit_code"] = exit_code
            metrics.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            runs.append(row)
            write_report(output, runs, config)
    write_report(output, runs, config)
    return 0


def summarize_command(args) -> int:
    root = Path(args.output).resolve()
    runs = []
    for telemetry in root.glob("runs/*/*/telemetry.ndjson"):
        mode, map_name = telemetry.parent.parent.name, telemetry.parent.name
        metrics = telemetry.parent / "metrics.json"
        previous = (json.loads(metrics.read_text(encoding="utf-8"))
                    if metrics.exists() else {})
        row = summarize_run(map_name, mode, telemetry,
                            telemetry.parent / "trajectory.ndjson")
        for key in ("wall_seconds", "process_exit_code"):
            if key in previous:
                row[key] = previous[key]
        if previous.get("result") == "WALL_TIMEOUT":
            row["result"] = "WALL_TIMEOUT"
            row["stop_category"] = "wall_clock_performance_limit"
            row["failure_reason"] = previous.get("failure_reason", "wall-clock limit")
        metrics.write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8")
        runs.append(row)
    previous_report = root / "summary.json"
    config = {"source": "existing run directory"}
    if previous_report.exists():
        config = json.loads(previous_report.read_text(encoding="utf-8")).get(
            "config", config)
    write_report(root, runs, config)
    return 0


def compare_command(args) -> int:
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    old = {(row["map"], row["mode"]): row for row in before["runs"]}
    deltas = []
    for row in after["runs"]:
        key = (row["map"], row["mode"])
        previous = old.get(key)
        if not previous:
            continue
        deltas.append({
            "map": key[0], "mode": key[1],
            "result_before": previous["result"], "result_after": row["result"],
            "sector_delta": row["world"]["sectors_entered"]
                            - previous["world"]["sectors_entered"],
            "observed_delta": row["world"]["sectors_observed"]
                              - previous["world"]["sectors_observed"],
            "time_delta": row["game_time"] - previous["game_time"],
            "interaction_delta": row["objects"]["interactions_activated"]
                                 - previous["objects"]["interactions_activated"],
            "repeat_selection_delta": row["navigation"]["repeated_target_selections"]
                                      - previous["navigation"]["repeated_target_selections"],
            "topology_rebuild_delta": row["navigation"]["topology_rebuilds"]
                                      - previous["navigation"]["topology_rebuilds"],
            "wall_seconds_delta": (
                round(row["wall_seconds"] - previous["wall_seconds"], 3)
                if row.get("wall_seconds") is not None
                and previous.get("wall_seconds") is not None else None),
        })
    payload = {"schema": 1, "before": str(args.before), "after": str(args.after),
               "aggregate_before": before.get("aggregate"),
               "aggregate_after": after.get("aggregate"), "deltas": deltas}
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--exe", required=True)
    run.add_argument("--maps-dir", required=True)
    run.add_argument("--game-dir", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--modes", default="normal,nodudes")
    run.add_argument("--pattern", default="")
    run.add_argument("--difficulty", type=int, default=0)
    run.add_argument("--timeout", type=int, default=600)
    run.add_argument("--stall", type=int, default=300)
    run.add_argument("--wall-timeout", type=int, default=30,
                     help="per-run wall-clock safety cap; 0 disables it")
    run.add_argument("--resume", action="store_true")
    run.set_defaults(function=run_command)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--output", required=True)
    summarize.set_defaults(function=summarize_command)
    compare = commands.add_parser("compare")
    compare.add_argument("before")
    compare.add_argument("after")
    compare.add_argument("--output")
    compare.set_defaults(function=compare_command)
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
