# LLMapper bot architecture sanitisation report

Date: 2026-08-22

This pass deliberately removes behaviour that invents evidence. It does not
claim that every newly honest failure has already been repaired. The hard
question for each retained mechanism is whether it models Blood or masks a
disagreement between observation, physical navigation, task state, and
execution.

## Decision/recovery inventory

| Mechanism | Class | Activation / termination evidence | Result of this pass |
|---|---|---|---|
| Simulated run timeout and no-semantic-progress timeout | B watchdog | Elapsed simulated time; terminates run and reports failure | Retained. They create no world knowledge. |
| Explicit jump phases, grounded landing/support confirmation | A physical protocol | Concrete `NavRouteStep`, engine jump state, Z velocity, player support | Retained. Jump input now requires the owning physical edge. |
| Jump action timeout and bounded identical-edge attempts | B/C boundary | Settled player after a failed explicit arc | Retained for now. The timeout is a watchdog; repeated identical attempts remain debt to replace with a typed failure result. |
| Speculative `SEARCH_CURRENT_AREA` jump probes | C compensation | No selected work; ended by time/arrival/count | Removed. `NO_APPLICABLE_ACTION` is now reported instead. |
| `jumpFallbackEdges` / WALK failure to JUMP | C compensation | Collision-safe WALK approaches exhausted | Removed. A traversal mode cannot change without new physical evidence. |
| Physical edge expiry (25/50/100/200 seconds) | C compensation | Time alone erased `failedEdges` / `navEdgeFailures` | Removed. Failures remain until the relevant portal/geometry signature changes. |
| Objective suppression backoff and `wakeSoonestDormant` | C compensation | Time or exhaustion of alternatives re-armed work | Replaced. Suppression is tied to topology/capability evidence; no-options does not wake it. |
| Portal `inside(receivingSector)` fallback | C compensation | Collision-safe local planner failed but point was topologically inside | Removed. `inside()` is not proof that the player cylinder can cross. |
| Cross-sector nearest-nav-cell fallback | D workaround | Strict target resolution failed | Removed. A sector-qualified pose query cannot silently change physical owner. |
| `max input - 1` throttle | D workaround | AGTST6 happened to retain a route under one microtrajectory | Removed. Running now uses the engine's natural maximum input. |
| Interaction input/animation/outcome windows | A/B | ActionScan/weapon transaction, engine busy state, observed consequence | Retained. Fixed durations should still be replaced by engine state wherever possible. |
| Explosive prime, retreat and outcome window | A/B | Weapon phase, safe distance and observed destruction/world delta | Retained as a bounded physical transaction. |
| Movement, waypoint, stationary and objective stall detectors | B/C overlap | Lack of pose/route/objective progress | Retained but still overlapping. They now suppress under stable evidence rather than manufacturing retry evidence. A single typed transition result should replace them. |
| Objective hard lifetime and door/mechanism wait limit | B watchdog | Transaction exceeds bound | Retained; expiry reports failure. |
| Interaction reactivation cooldown, max activation attempts and repeated surface probes | C debt | Fixed time/count under otherwise deterministic state | Retained for a later isolated pass. Desired replacement is explicit issued/accepted/changing/settled/consequence state. |
| Loop breaker | C debt | Repeated quantized pose/goal/knowledge | Partially simplified: duplicate suppression was removed. The remaining loop-level punishment should become diagnosis of the concrete transition cycle. |
| Opened-route, reopened-return and recent-causal grace windows | C debt | Fixed 2/12/15-second associations | Retained for a later transaction-identity pass. Stable action-to-consequence ownership should replace them. |
| Support/moving-geometry settle checks | A physical protocol | Engine busy/state and stable support pose | Retained. |
| Debug/telemetry throttles | Non-behavioural | Output rate only | Retained; they do not affect decisions. |

## Representation inventory

The same progression intention is currently projected through these layers:

1. observed `Portal` / visible object / interaction surface;
2. cached `knownGraph` facts;
3. authoritative physical `NavCell` / `NavLink` pose graph;
4. persistent interaction, object and failure memory;
5. opportunity ledger and selected `Mission`;
6. executable `Objective`;
7. installed `navRoute` and local portal plan;
8. movement target and jump/gap executor state;
9. opened-route, follow-through, dynamic-prerequisite and causal-continuation state.

Only layers 1, 3, 4 and 5 add distinct information. `Mission` and `Objective`
are useful short-lived projections, while movement targets, portal plans,
jump state and opened-route/follow-through state should be transaction-local
execution state. The remaining danger is that these projections still copy
identity, reachability and commitment and can disagree. This pass removed
route planning's ability to rewrite a physical route endpoint with task or
gateway coordinates; route installation now accepts a concrete pose cell.

## Complexity delta

Counts below cover the audited behavioural-debt families, not every field in
the bot.

| Measure | Before | After | Delta |
|---|---:|---:|---:|
| Mutable recovery/failure fields | 14 | 0 | -14 |
| Invented/resurrection fallback paths | 6 | 0 | -6 |
| Fixed behavioural backoff/cooldown constants in those paths | 2 | 0 | -2 |
| Test-trajectory throttle workarounds | 1 | 0 | -1 |
| Duplicate loop-break suppressions per event | 2 | 1 | -1 |

Removed mutable fields include speculative-search pose/count/timing, jump
fallback identity, jump cooldown, both physical-failure expiry fields, and
the last-failed-opportunity time gate.

## Focused failure pipeline

| Map | Intended physical action | Earliest confirmed divergence | Current evidence/result |
|---|---|---|---|
| AGTST4 | Reach and operate the crypt continuation/door | Physical pose connectivity: the observed work is conserved but its approach/continuation becomes unreachable after thin/transit geometry is projected into cells | Final ledger contains one unreachable frontier; `STALLED` at 376s. No door-specific patch was made. |
| AGTST7 | Open the door, continue, use exit | No current divergence | Door is observed, selected and crossed; `COMPLETED` at 44s. The previously reported failure is not reproducible on this revision. |
| AGTST8 | Treat the solid sprite as the owner of blockage/support and continue through the lift/bridges | First confirmed divergence is execution/failure attribution: 23 reachable tasks are eventually suppressed under unchanged evidence; current telemetry does not yet prove the blocking sprite became the failed transition owner | `STALLED` at 333s with 23 dormant tasks. This is now an honest diagnostic instead of periodic retry activity. |
| AGTST9 | Apply a valid damage effect to the blocking gib, then continue | No current divergence | Sprite 5 is observed as a damageable body, an effect task is created, removal is observed at 1s, and the map `COMPLETED` at 58s. |
| AGTST13 | After acquiring the key, return across the physical support chain to the exit | Physical pose graph connectivity. In the key-acquired trace the exit task changes from missing-key to satisfiable, but remains at `hops=-1`. The graph splits the inbound and outbound portions of sector 5 instead of representing their real connecting pose/transition | Current deterministic run stalls at 332s; a captured key-acquired run proves task memory/prerequisite reasoning is not the first failure. |

The failures therefore do not collapse into one sprite-specific rule. AGTST4
and AGTST13 share a lower physical-pose connectivity defect. AGTST8 exposes
failure ownership plus excessive independent objective suppression. AGTST7
and AGTST9 currently demonstrate that door and damage-affordance pipelines can
work without map-specific code.

## Newly exposed honest failures

Removing unowned jumping exposed two important dependencies:

* AGTST12 previously completed only after `executeJumpTraversal()` was called
  repeatedly without an owning `NavRouteStep`. It now rejects those calls with
  `jump_transition_rejected reason=no_owning_physical_edge` and stalls. The
  missing fix belongs in support-transition construction, not recovery.
* AGTST6 no longer depends on the max-minus-one throttle trajectory or a local
  speculative jump. It ends with four reachable but evidence-dormant tasks.

These are regressions in completion count, but improvements in falsifiability:
the former green results depended on behaviour that violated the requested
physical-evidence invariants.

## Final AGTST regression

Difficulty 0, normal mode, simulated timeout 600 seconds, stall window 300
seconds. Machine-readable data is in `_tmp_sanitise_final` and
`_tmp_sanitise_final_agtst15`.

| Map | Result | Simulated time |
|---|---|---:|
| AGTST1 | COMPLETED | 30s |
| AGTST2 | COMPLETED | 73s |
| AGTST3 | COMPLETED | 32s |
| AGTST4 | STALLED | 376s |
| AGTST5 | COMPLETED | 57s |
| AGTST6 | STALLED | 342s |
| AGTST7 | COMPLETED | 44s |
| AGTST8 | STALLED | 333s |
| AGTST9 | COMPLETED | 58s |
| AGTST10 | COMPLETED | 36s |
| AGTST11 | COMPLETED | 61s |
| AGTST12 | STALLED | 338s |
| AGTST13 | STALLED | 332s |
| AGTST14 | STALLED | 333s |
| AGTST15 | STALLED | 301s |
| AGTST16 | COMPLETED | 40s |

All 54 navigation architecture tests pass and the final executable links.

## Next evidence-driven work

1. Diagnose why AGTST12's real support-to-support jump is not emitted as a
   `NavLink`; this is the smallest proof case for the no-unowned-jump invariant.
2. Compare AGTST4 and AGTST13 graph components at the first split and repair
   thin/transit/support sampling without restoring cross-sector target lookup.
3. Propagate the exact engine collision owner into the failed physical
   transition and affordance ledger, then re-check AGTST8.
4. Replace interaction retry counts/cooldowns with a stable action transaction.
5. Replace opened-route and recent-causal time windows with transaction-owned
   consequences and committed downstream work.
