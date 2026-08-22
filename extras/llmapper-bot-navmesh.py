#!/usr/bin/env python3
"""Render one LLMapper physical-nav snapshot as an orbitable 3D HTML view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODE_NAMES = {
    0: "walk",
    1: "step",
    2: "jump",
    3: "crouch",
    4: "drop",
    5: "ride",
}


def read_json_lines(path: Path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_snapshot(path: Path, requested: int | None):
    revisions = []
    records = {}
    times = {}
    for row in read_json_lines(path):
        revision = row.get("revision")
        if row.get("type") == "snapshot":
            revisions.append(revision)
            times[revision] = row.get("game_time", 0)
            records.setdefault(revision, {"cells": [], "edges": []})
        elif row.get("type") in ("cell", "edge"):
            records.setdefault(revision, {"cells": [], "edges": []})[
                row["type"] + "s"
            ].append(row)
    if not revisions:
        raise ValueError("no complete navigation snapshots found")
    revision = requested if requested is not None else revisions[-1]
    if revision not in records:
        raise ValueError(f"revision {revision} is not present")
    later_times = [times[item] for item in revisions
                   if item > revision and times[item] >= times[revision]]
    end_time = min(later_times) if later_times else None
    return revision, times[revision], end_time, records[revision]


def load_trajectory(path: Path | None, start: int, end: int | None):
    if not path or not path.exists():
        return []
    return [row for row in read_json_lines(path)
            if row.get("game_time", -1) >= start
            and (end is None or row.get("game_time", -1) <= end)]


def load_targets(path: Path | None, start: int, end: int | None):
    if not path or not path.exists():
        return []
    targets = {}
    for row in read_json_lines(path):
        when = row.get("game_time", -1)
        if when < start or (end is not None and when > end):
            continue
        if row.get("event") != "interaction_pose_search":
            continue
        fields = {}
        for token in row.get("detail", "").replace("target=(", "target=").split():
            if "=" in token:
                key, value = token.rstrip(",)").split("=", 1)
                fields[key] = value
        try:
            xyz = [int(value) for value in fields["target"].split(",")]
            key = (int(fields["kind"]), int(fields["id"]))
            targets[key] = {"kind": key[0], "id": key[1],
                            "x": xyz[0], "y": xyz[1], "z": xyz[2]}
        except (KeyError, ValueError, IndexError):
            continue
    return list(targets.values())


def scaled(row):
    return [round(row["x"] / 1024, 3), round(row["y"] / 1024, 3),
            round(-row["z"] / 4096, 3)]


def make_payload(snapshot, trajectory, targets, revision, start, end):
    cells = snapshot["cells"]
    by_id = {cell["id"]: cell for cell in cells}
    node_groups = []
    surface_groups = []
    for support_kind, name in ((0, "sector-floor poses"),
                               (1, "sprite/voxel poses")):
        members = [cell for cell in cells
                   if cell["support_kind"] == support_kind]
        vertices = []
        triangles = []
        hover = []
        half_cell = 116 / 1024
        for cell in members:
            x, y, z = scaled(cell)
            base = len(vertices)
            vertices.extend([
                [x - half_cell, y - half_cell, z],
                [x + half_cell, y - half_cell, z],
                [x + half_cell, y + half_cell, z],
                [x - half_cell, y + half_cell, z],
            ])
            triangles.extend([[base, base + 1, base + 2],
                              [base, base + 2, base + 3]])
            label = (f"cell {cell['id']} · sector {cell['sector']} · "
                     f"support {cell['support']} · area {cell['area']} · "
                     f"z {cell['z']} · clearance {cell['clearance']}")
            hover.extend([label] * 4)
        surface_groups.append({
            "name": name.replace("poses", "standable surfaces"),
            "xyz": vertices,
            "triangles": triangles,
            "hover": hover,
        })
        node_groups.append({
            "name": name,
            "xyz": [scaled(cell) for cell in members],
            "hover": [
                f"cell {cell['id']} · sector {cell['sector']} · "
                f"support {cell['support']} · area {cell['area']} · "
                f"z {cell['z']} · clearance {cell['clearance']}"
                for cell in members
            ],
        })
    edge_groups = []
    for mode, name in MODE_NAMES.items():
        coords = []
        count = 0
        for edge in snapshot["edges"]:
            if edge["mode"] != mode or edge["from"] not in by_id \
                    or edge["to"] not in by_id:
                continue
            coords.extend([scaled(by_id[edge["from"]]),
                           scaled(by_id[edge["to"]]), None])
            count += 1
        if coords:
            edge_groups.append({"mode": mode, "name": name,
                                "xyz": coords, "count": count})
    return {
        "revision": revision,
        "start": start,
        "end": end,
        "surfaces": surface_groups,
        "nodes": node_groups,
        "edges": edge_groups,
        "trajectory": [scaled(row) for row in trajectory],
        "targets": [{**target, "xyz": scaled(target)} for target in targets],
    }


def fragment(payload):
    data = json.dumps(payload, separators=(",", ":"))
    return f'''<div id="llmapper-navmesh-3d">
  <h2>Physical navigation graph · revision {payload["revision"]}</h2>
  <div class="viz-row text-small" aria-label="Navigation graph legend">
    <span>Orbit: drag</span><span>Pan: shift-drag</span><span>Zoom: wheel</span>
    <span>Click legend entries to isolate layers</span>
  </div>
  <div id="llmapper-navmesh-plot" role="img" aria-label="Orbitable 3D physical navigation graph with standable poses, directed traversal links, interaction targets, and bot trajectory"></div>
  <div id="llmapper-navmesh-detail" class="text-small text-muted"></div>
</div>
<style>
#llmapper-navmesh-3d {{ width:100%; color:var(--foreground); }}
#llmapper-navmesh-3d h2 {{ margin-bottom:8px; }}
#llmapper-navmesh-3d .viz-row {{ gap:14px; margin-bottom:6px; }}
#llmapper-navmesh-plot {{ width:100%; height:620px; min-height:420px; }}
#llmapper-navmesh-detail {{ margin-top:6px; }}
@media (max-width:520px) {{ #llmapper-navmesh-plot {{ height:480px; }} }}
</style>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<script>
(() => {{
  const root = document.getElementById('llmapper-navmesh-3d');
  const plot = document.getElementById('llmapper-navmesh-plot');
  const detail = document.getElementById('llmapper-navmesh-detail');
  const payload = {data};
  const css = getComputedStyle(root);
  const token = name => css.getPropertyValue(name).trim();
  const colors = [token('--viz-series-1'), token('--viz-series-2'),
    token('--viz-series-3'), token('--viz-series-4'), token('--viz-series-5'),
    token('--viz-series-6')];
  const traces = [];
  payload.surfaces.forEach((group,index) => {{
    traces.push({{type:'mesh3d', name:group.name,
      x:group.xyz.map(p=>p[0]), y:group.xyz.map(p=>p[1]), z:group.xyz.map(p=>p[2]),
      i:group.triangles.map(t=>t[0]), j:group.triangles.map(t=>t[1]),
      k:group.triangles.map(t=>t[2]), text:group.hover,
      hovertemplate:'%{{text}}<extra></extra>', flatshading:true,
      color:index ? colors[4] : colors[0], opacity:index ? .62 : .34,
      lighting:{{ambient:.9,diffuse:.25,specular:0,roughness:1}},
      lightposition:{{x:0,y:0,z:1000}}}});
  }});
  payload.edges.forEach(group => {{
    const x=[], y=[], z=[];
    group.xyz.forEach(point => {{
      if (point === null) {{ x.push(null); y.push(null); z.push(null); }}
      else {{ x.push(point[0]); y.push(point[1]); z.push(point[2]); }}
    }});
    traces.push({{type:'scatter3d', mode:'lines', name:`${{group.name}} (${{group.count}})`,
      x,y,z, hoverinfo:'skip', visible: group.mode === 2 || group.mode === 5 ? true : 'legendonly',
      line:{{color:colors[group.mode % colors.length], width:group.mode === 2 ? 5 : 2}}}});
  }});
  payload.nodes.forEach((group,index) => {{
    traces.push({{type:'scatter3d', mode:'markers', name:group.name,
      x:group.xyz.map(p=>p[0]), y:group.xyz.map(p=>p[1]), z:group.xyz.map(p=>p[2]),
      text:group.hover, hovertemplate:'%{{text}}<extra></extra>',
      visible:'legendonly', marker:{{size:index ? 5 : 3,
      color:index ? colors[4] : colors[0], opacity:index ? 1 : .72}}}});
  }});
  if (payload.trajectory.length) traces.push({{type:'scatter3d', mode:'lines+markers',
    name:'bot trajectory', x:payload.trajectory.map(p=>p[0]),
    y:payload.trajectory.map(p=>p[1]), z:payload.trajectory.map(p=>p[2]),
    hovertemplate:'trajectory (%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}})<extra></extra>',
    line:{{color:colors[1],width:7}}, marker:{{color:colors[1],size:2}}}});
  if (payload.targets.length) traces.push({{type:'scatter3d', mode:'markers+text',
    name:'interaction targets', x:payload.targets.map(t=>t.xyz[0]),
    y:payload.targets.map(t=>t.xyz[1]), z:payload.targets.map(t=>t.xyz[2]),
    text:payload.targets.map(t=>`target ${{t.kind}}:${{t.id}}`), textposition:'top center',
    hovertemplate:'%{{text}}<extra></extra>', marker:{{color:colors[3],size:8,symbol:'diamond'}}}});
  const foreground=token('--foreground'), grid=token('--border');
  Plotly.newPlot(plot,traces,{{margin:{{l:0,r:0,t:8,b:0}},paper_bgcolor:'rgba(0,0,0,0)',
    plot_bgcolor:'rgba(0,0,0,0)',font:{{color:foreground}},showlegend:true,
    legend:{{orientation:'h',x:0,y:1}},scene:{{aspectmode:'data',
      xaxis:{{title:'X (1024 map units)',gridcolor:grid,zerolinecolor:grid}},
      yaxis:{{title:'Y (1024 map units)',gridcolor:grid,zerolinecolor:grid}},
      zaxis:{{title:'Elevation (-Z / 4096)',gridcolor:grid,zerolinecolor:grid}},
      camera:{{eye:{{x:1.45,y:1.45,z:1.15}}}}}}}},{{responsive:true,displaylogo:false}});
  const interval = payload.end === null ? `from ${{payload.start}}s` : `${{payload.start}}–${{payload.end}}s`;
  detail.textContent = `Snapshot ${{payload.revision}} · ${{interval}} · ${{payload.nodes.reduce((n,g)=>n+g.xyz.length,0)}} poses · hover a pose for physical support identity`;
}})();
</script>
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--navmesh", required=True, type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--revision", type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fragment", action="store_true")
    args = parser.parse_args()
    revision, start, end, snapshot = load_snapshot(args.navmesh, args.revision)
    payload = make_payload(snapshot,
                           load_trajectory(args.trajectory, start, end),
                           load_targets(args.telemetry, start, end),
                           revision, start, end)
    content = fragment(payload)
    if not args.fragment:
        content = "<!doctype html><meta charset=\"utf-8\"><title>LLMapper navmesh</title>" + content
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
