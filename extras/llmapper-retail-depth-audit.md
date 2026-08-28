# Retail exploration depth audit

Date: 2026-08-21

This pass used all 43 accessible original single-player maps in `-nodudes`
mode as a soft behavioral benchmark. Each run had a 600-second simulated-time
limit, a 300-second no-progress stall limit, and a 30-second wall-clock safety
limit. AGTST1--13 remained the hard correctness contract.

## Outcome

The main defect was not lack of candidate goals. Several stalled maps kept an
unreachable interaction or portal active and recomputed the same impossible
path every decision frame. Moving actuator coordinates and interpolation
`busy` counters also made physically unchanged questions look new, defeating
bounded-failure memory.

The fix gives negative route answers a navigation-state lifetime, makes
failed interaction approaches dormant without forgetting the interaction,
owns repeated failure by stable work/pose/topology identity, and lets moving
sector escape reject one failed exit and try another. Dynamic navigation
invalidation now treats `busy` as moving/not-moving while still hashing the
authoritative sector and wall coordinates, state, and collision flags.

No map name, sector, wall, sprite, tile, or coordinate is special-cased.

## Corpus comparison

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| Maps | 43 | 43 | 0 |
| Sectors entered | 912/14,079 | 895/14,079 | -17 |
| Weighted raw sector ratio | 6.48% | 6.36% | -0.12 pp |
| Mean raw sector ratio | 9.90% | 9.62% | -0.28 pp |
| Median raw sector ratio | 4.70% | 4.70% | 0 |
| Route-plan failures | 10,012 | 3,948 | -6,064 (-60.6%) |
| Dynamic-link refreshes | 3,082 | 205 | -2,877 (-93.3%) |
| Loop breaks | 30 | 25 | -5 |
| Corpus wall time | 319.00 s | 131.06 s | -187.94 s (-58.9%) |
| Items picked up | 178 | 178 | 0 |
| Interactions activated | 40 | 39 | -1 |

Entered/total sectors remains a raw coverage measure, not literal completion:
retail maps contain technical, decorative, secret, state-dependent, and
currently unsupported areas.

Coverage changed on four maps:

| Map | Before | After | Interpretation |
|---|---:|---:|---|
| E1M1 | 22/155 | 25/155 | Three additional sectors; repeated disconnected-route work fell from 116 to 8 failures |
| E6M3 | 10/240 | 11/240 | A failed moving-sector exit is retired and another exit is tried; 4,468 route failures fell to 1,024 |
| E1M8 | 56/433 | 55/433 | One-sector path-order variation; both runs time out |
| E3M7 | 39/142 | 19/142 | The new route enters a damaging/scripted-dude region and death terminates the run at 426 s |

E3M7 is the reason aggregate sector coverage is lower despite the two useful
navigation gains. The repeat run reproduced the terminal death. It is retained
as honest soft-benchmark evidence: avoiding or surviving that route requires
hazard/combat policy outside this navigation pass, so no brittle coverage
heuristic was added.

## Representative stall evidence

- E6M9 previously emitted the identical failed local-portal hypothesis 1,631
  times. It now makes four bounded, backoff-spaced attempts and remains at
  2/157 sectors. This is a large rationality/CPU improvement without claiming
  false exploration progress.
- E3M1 remains at 21/382 sectors, but route failures fall from 264 to 7. An
  unreachable interaction approach is released so other ledger work can run.
- E1M2 remains prerequisite-blocked at 7/313 sectors. Route failures fall from
  1,587 to 1,369; no unsupported prerequisite behavior is invented.
- E6M3 advances through one additional sector and changes its earliest blocker
  from a navigation dead end to a later unresolved interaction.

## Deferred mechanisms

E2M1 remains a wall-clock performance limit at 7/492 sectors. Its large open
area and water surface expose swimming, diving, drowning, and water-cost
modeling that the bot does not yet represent. Water mechanics are explicitly
deferred; E2M1 is recorded as evidence, not used to justify an incomplete
water workaround.

Other unfinished retail maps continue to expose prerequisite, combat,
interaction, thin-transit, hazard, and exploration-policy gaps. Retail map
completion is not a regression requirement.

## Hard regression contract

| Map | Result | Simulated time |
|---|---|---:|
| AGTST1 | Completed | 31 s |
| AGTST2 | Completed | 76 s |
| AGTST3 | Completed | 25 s |
| AGTST4 | Completed | 491 s |
| AGTST5 | Completed | 52 s |
| AGTST6 | Completed | 74 s |
| AGTST7 | Completed | 89 s |
| AGTST8 | Completed | 81 s |
| AGTST9 | Completed | 76 s |
| AGTST10 | Completed | 36 s |
| AGTST11 | Completed | 17 s |
| AGTST12 | Completed | 15 s |
| AGTST13 | Completed | 102 s |

All 54 navigation architecture tests pass. The full `nblood`, `rednukem`, and
`pcexhumed` build passes.

## Artifacts

- `llmapper-corpus-results/2026-08-21-depth-before.csv`
- `llmapper-corpus-results/2026-08-21-depth-after.csv`
- `llmapper-corpus-results/2026-08-21-depth-comparison.json`

The CSV files retain per-map outcomes and the new portal, moving-sector escape,
and interaction-approach route-failure counters. The JSON file retains the
machine-readable before/after aggregate and per-map deltas.

## Next high-value work

1. Add an explicit hazard-cost/survivability model before trying to improve
   E3M7; keep it separate from future swimming/diving work.
2. Represent narrow technical sectors as transit geometry without requiring a
   standable destination cell.
3. Improve prerequisite-source discovery on maps such as E1M2 and E3M1 rather
   than repeatedly selecting their blocked goal.
4. Return to E2M1 only when swimming, diving, breath/drowning, and water route
   costs can be modeled together.
