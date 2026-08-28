# Original Blood level corpus navigation audit

Date: 2026-08-20

Retail maps are a soft behavioral benchmark. The AGTST suite remains the hard
functional contract. Raw sector coverage below is not level-completion
percentage: map totals include technical, decorative, secret, and potentially
unreachable sectors.

## Scope and method

The available loose-map corpus contains 43 single-player maps:

- Episode 1: E1M1-E1M8 (8)
- Episode 2: E2M1-E2M9 (9)
- Episode 3: E3M1-E3M8 (8)
- Episode 4: E4M1-E4M9 (9)
- Episode 6: E6M1-E6M9 (9)

Each map was run at difficulty 0 in two modes, for 86 runs per revision:

- `normal`
- `nodudes`, using Blood's `-nodudes 1` startup option

Runs use a 600-second simulated-time limit, a 300-second no-progress limit,
and a 30-second wall-clock safety cap. The wall-clock cap is reported as a
performance limit, never as a navigation verdict. It was reached only by both
E2M1 runs. No retail map completed, crashed, or asserted in either corpus pass.

The new runner writes telemetry and trajectory files for every run plus
per-run JSON, aggregate JSON, CSV, Markdown, and paired deltas. It is resumable
and rewrites the aggregate after every case.

`nodudes` is not a reliable enemy-free mode in this build. New damage-source
telemetry identified a `kStatDude` source in 9 of the 10 deaths in the second
`nodudes` corpus. Results therefore retain the literal mode name and do not
claim semantic equivalence to a neutralized-enemy level.

## Baseline corpus

| Mode | Outcomes | Entered / known sectors | Weighted raw % | Mean % | Median % | Pickups | Interactions | Topology rebuilds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| normal | 29 died, 6 loops, 6 stalled, 1 timeout, 1 wall limit | 449/13,830 | 3.25 | 6.07 | 2.52 | 85 | 18 | 3,358 |
| nodudes | 11 died, 10 loops, 14 stalled, 7 timeouts, 1 wall limit | 714/13,830 | 5.16 | 8.49 | 3.61 | 165 | 33 | 6,419 |

The strongest raw-coverage sensitivity cases were:

| Map | Mode | Entered / total | Raw % | Outcome |
|---|---|---:|---:|---|
| E1M7 | nodudes | 31/53 | 58.49 | timeout |
| E6M8 | nodudes | 46/85 | 54.12 | timeout while progressing |
| E4M8 | both | 26/80 | 32.50 | timeout/navigation dead end |
| E6M1 | nodudes | 28/123 | 22.76 | stalled/navigation dead end |
| E1M1 | both | 26/155 | 16.77 | death |

E4M6 destroyed a blocker at 598 seconds, and E6M8 entered a new sector at 576
seconds. Both reached 600 seconds while still making meaningful progress and
must not be classified as dead ends.

## Failure taxonomy

The categories are telemetry-based first-blocker hypotheses, not assertions
that a retail puzzle has been fully understood.

- Damage/combat: normal mode ended in death on 28 of 43 maps after the change.
  The purported `nodudes` mode still recorded nine dude-sourced deaths and one
  environmental/scripted death.
- Navigation or goal loops: explicit repeated position/goal/knowledge state
  ended 6 normal and 10 `nodudes` runs. Several spent roughly 278-299 seconds
  stationary before termination.
- Navigation dead ends: 5 normal and 8 `nodudes` runs ended with route
  unavailability, failed route steps, or stationary deadlock after prolonged
  lack of world progress.
- Traversal execution stalls: 2 normal and 4 `nodudes` runs repeatedly entered
  jump/traversal phases without reaching new geometry. E1M4, for example,
  repeated the same bounded jump takeoff until loop detection.
- Exploration oscillation: 3 `nodudes` runs repeatedly exhausted or changed
  objectives without obtaining a new world-state result.
- Prerequisite unresolved: 1 normal and 2 `nodudes` runs retained unmatched
  prerequisite-unavailable evidence. Rearmed opportunities are subtracted, so
  a prerequisite that later became satisfiable is not counted as unresolved.
- Interaction unresolved: one `nodudes` run ended around repeated interaction
  failures. This is a hypothesis for inspection, not a map-specific diagnosis.
- Efficiency: E4M6 and E6M8 timed out while progressing; E4M9 timed out after a
  long period without meaningful progress.
- Wall-clock performance: E2M1 builds an observed mesh of about 20,000 cells
  and was the only map to hit the external safety limit in both modes.

## Architectural findings

### Actual bug fixed

`navPoseStableTicks` counted calls to `ensureNavTopology()`, not simulation
ticks. Several planning layers call that accessor in one rendered frame, so
continuously moving geometry could be declared settled, invalidate the mesh,
and rebuild it repeatedly at the same simulated second.

Stability now uses the simulation clock. A continuously changing remote pose
may request one interim rebuild per simulated second, while a genuinely stable
pose receives its final rebuild after eight rendered frames. Dynamic links are
still refreshed while the mesh is live, and all AGTST moving-support behavior
remains intact.

### Architectural weaknesses

- The fixed 256-unit grid can produce very large graphs. E2M1 reached about
  20,000 cells after observing only 14 sectors. Full-graph reconstruction and
  repeated routing over that graph remain expensive even after rebuild
  throttling.
- Route-unavailable telemetry is extremely high on some runs: E6M3 `nodudes`
  recorded 2,267 failures, E6M1 recorded 1,823, and E2M3 recorded 1,274. This
  suggests repeated requests against a known-unavailable graph state need
  stronger generation-based suppression or cheaper cached negative answers.
- Execution failures and goal failures are not cleanly separated in the final
  result. Explicit phase-cycle signatures would distinguish “the selected
  route is impossible to execute” from “the exploration policy keeps selecting
  equivalent goals.”
- Combat, hazards, and navigation materially affect one another. The current
  startup option does not provide a trustworthy neutral-enemy control corpus.
- `regions_reached` in the machine report is an exploration-branch proxy, not
  a mathematically exact connected-component count. Pickups are also inferred
  from objective completion/confirmation events. Both should remain labelled
  as proxies until engine callbacks expose authoritative values.

### Cleanup and performance opportunities

- A compiler warning in combat-outcome telemetry used a tracked field directly
  in `snprintf`; explicit integer conversion now makes the logging contract
  clear.
- Adaptive grid resolution, hierarchical routing, or lazy per-sector graph
  materialization should be evaluated for very large open sectors. This was
  not attempted here because it changes traversal fidelity and deserves an
  isolated synthetic contract before use.
- Stable topology identity, live pose, and dynamic interaction state are
  conceptually distinct but still trigger broad work in neighboring code.
  Narrower dirty-region ownership could avoid reconstructing unaffected cells.

## Before/after evidence

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| normal entered / known sectors | 449/13,830 (3.25%) | 452/13,830 (3.27%) | +3 sectors |
| nodudes entered / known sectors | 714/13,830 (5.16%) | 719/13,830 (5.20%) | +5 sectors |
| normal pickups | 85 | 95 | +10 |
| nodudes pickups | 165 | 168 | +3 |
| interactions activated | 51 | 51 | unchanged |
| topology rebuilds | 9,777 | 2,702 | -7,075 (-72.4%) |

Of 86 paired runs, 69 had identical entered-sector counts, 11 increased, and
6 decreased. Nine terminal outcomes changed, which is expected for nonlinear
combat and time-limited trajectories. The largest decline was E3M4 normal
(19 to 7 sectors); its `nodudes` pair remained exactly 7 sectors before and
after, while the old normal trajectory had nine damage events. That makes it
an enemy-influenced trajectory difference rather than evidence that supported
navigation regressed.

In the pathological E2M1 case, the same 30-second wall budget advanced from 16
simulated seconds before to 65 seconds normal / 71 seconds `nodudes` after.
It still entered four sectors, so this is specifically a throughput improvement
and not a claim of better exploration.

## Hard regression result

All AGTST maps completed after the change:

| Map | Time |
|---|---:|
| AGTST1 | 29s |
| AGTST2 | 77s |
| AGTST3 | 61s |
| AGTST4 | 283s |
| AGTST5 | 49s |
| AGTST6 | 81s |
| AGTST7 | 20s |
| AGTST8 | 86s |
| AGTST9 | 66s |
| AGTST10 | 61s |

All 51 navigation architecture tests pass, the final build succeeds, and
`git diff --check` is clean.

## Reproduction

From the NBlood repository root:

```powershell
python extras\llmapper-bot-corpus.py run `
  --exe .\nblood.exe `
  --maps-dir ..\maps\blood `
  --game-dir ..\reference\blood `
  --output obj\llmapper-corpus `
  --modes normal,nodudes `
  --difficulty 0 --timeout 600 --stall 300 --wall-timeout 30
```

Use `--resume` to continue an interrupted corpus. Rebuild reports from saved
traces with `summarize`, or compare two `summary.json` files with `compare`.
The checked-in machine-readable snapshots and paired deltas are in
`extras/llmapper-corpus-results/`.

## Remaining high-value work

1. Add an authoritative neutral-enemy test mode that keeps level scripting and
   enemy-owned progression objects intact while suppressing hostile damage/AI.
2. Prototype hierarchical or adaptive navigation grids against E2M1, guarded
   by a small synthetic large-open-sector fidelity test and the full AGTST suite.
3. Add execution-phase cycle detection and generation-keyed negative route
   caching; evaluate E1M4, E1M5, E6M1, E6M3, and E2M3 as soft sensitivity cases.
4. Add authoritative item/interaction callbacks and a true topology-region
   metric so proxy fields can be retired.
5. Investigate representative prerequisite and interaction traces, then create
   a small AGTST map only if they reveal a missing general mechanism.
