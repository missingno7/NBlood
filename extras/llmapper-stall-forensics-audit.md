# Retail stall forensics: portal-route ownership

Date: 2026-08-21

This focused pass investigated the E1M1 `-nodudes` back-and-forth stall near
2:20. E1M1 and E2M4 were run at difficulty 0 with a 600-second simulated-time
limit. Retail results remain soft diagnostics; AGTST1--13 remain the hard
contract.

## Stall ledger

| Map | Last meaningful progress | Remaining knowledge | Active objective | Observed symptom | Root cause | Before | After | New blocker |
|---|---|---|---|---|---|---:|---:|---|
| E1M1 | Weapon pickup in sector 65 at 126 s | 3 local frontiers, 17 remote frontiers, 25 interactions, 23 objects | Cross wall 538 from sector 65 to 145 | Support route and local portal plan repeatedly pull in opposite directions | Per-frame direct-probe results could steal ownership from an active support route; a portal plan created for an earlier moving-sector escape was also reused by a later frontier objective | 25 sectors; wall 538 crossed at 305 s; 217 plans; 122 ping-pongs | 25 sectors; wall 538 crossed at 137 s; 69 plans; 0 ping-pongs | Later attempts around wall 563 and an unresolved prerequisite; the reported open-area stall is gone |
| E2M4 | Frontier chain reached sector 333 | A frontier through wall 2456 remained actionable | Cross wall 2456 toward sector 174 | Two approach targets alternate almost every decision for roughly five minutes | The same support-route/local-plan ownership conflict, with a fresh local plan, proving stale age was not the only cause | 47 sectors; stalled at 494 s; 419 plans; 378 ping-pongs | 61 sectors; 69 plans; 0 ping-pongs | A later scripted/combat death at 384 s; combat remains out of scope |

## Causal reconstruction

E1M1 did not have an empty ledger and did not lack a destination. It had
already discovered wall 538 and correctly selected it as a traversable
frontier. The support-aware navigator produced a route toward
`(14336,48128)`. A local corner plan for the same wall targeted
`(9216,47392)`.

The local plan had originally been created while wall 538 was being used as an
emergency exit from a moving sector. Its cache identity contained the wall and
geometry, but not the high-level work which owned the plan. Later, a one-frame
collision probe near the boundary could report the direct approach reachable
and bypass the still-active support route. The cached corner plan then replaced
the global route. On the next frame the probe changed its answer, so the
support route replaced the corner plan again.

Between 120 and 240 seconds, the baseline constructed 128 routes and changed
route target 126 times. The player was not reconsidering meaningful
alternatives; two existing planners were continuously invalidating one another.

## Generic changes

1. A support-aware portal approach now owns locomotion until its route
   completes or explicitly invalidates itself. A transient direct collision
   answer cannot silently replace that route family.
2. A local portal plan is now owned by the objective key, objective attempt,
   and execution goal which created it, in addition to wall geometry. Reusing
   the same wall for different work causes a bounded replan from the current
   pose instead of replaying old waypoints.
3. Telemetry now records retained portal-route ownership and portal-plan owner
   changes.
4. The corpus summarizer records short-interval ABA route-target ping-pongs.
   This distinguishes the two-planner overwrite signature from ordinary
   progress through a sequence of different route legs.

No map, wall, sector, sprite, tile, or coordinate is special-cased. No new
navigation capability or subsystem was added; the fix makes the lifetime and
ownership of existing plans coherent.

## Before/after evidence

| Map | Sectors | Route plans | Route failures | ABA ping-pongs | Outcome |
|---|---:|---:|---:|---:|---|
| E1M1 before | 25/155 | 217 | 8 | 122 | Wall 538 crossed at 305 s |
| E1M1 after | 25/155 | 69 | 7 | 0 | Wall 538 crossed at 137 s; later unrelated blocker |
| E2M4 before | 47/632 | 419 | 25 | 378 | Navigation stall at 494 s |
| E2M4 after | 61/632 | 69 | 6 | 0 | Progressed 14 more sectors, then later scripted/combat death |

E1M1's total sector count is unchanged because both 600-second runs eventually
reach the same later unresolved region. The improvement is the disappearance
of the reported 168-second planning oscillation and the resulting 68% reduction
in route construction. E2M4 supplies independent coverage evidence and reaches
a genuinely different blocker.

## Unsupported/deferred behavior

The E2M4 after-run terminates on damage associated with a later scripted/combat
dependency even in `-nodudes` mode. Combat behavior was not added. Water,
swimming, diving, drowning, and elaborate hazard policy also remain deferred.

## Hard contract

| Map | Result | Simulated time |
|---|---|---:|
| AGTST1 | Completed | 32 s |
| AGTST2 | Completed | 76 s |
| AGTST3 | Completed | 28 s |
| AGTST4 | Completed | 498 s |
| AGTST5 | Completed | 52 s |
| AGTST6 | Completed | 75 s |
| AGTST7 | Completed | 89 s |
| AGTST8 | Completed | 81 s |
| AGTST9 | Completed | 76 s |
| AGTST10 | Completed | 36 s |
| AGTST11 | Completed | 17 s |
| AGTST12 | Completed | 15 s |
| AGTST13 | Completed | 106 s |

All 54 navigation architecture tests pass. The full `nblood`, `rednukem`, and
`pcexhumed` build passes. `git diff --check` is clean.
