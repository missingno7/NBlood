# Focused E1M1 / AGTST10--12 navigation audit

This report records the focused `E1M1` `nodudes` investigation and the
corresponding AGTST regression pass. Retail-map completion is diagnostic;
AGTST1--AGTST12 remains the hard contract.

## Findings

### E1M1 pushable fence

The fence was visible as solid sprite geometry, but the portal collision path
did not preserve the blocking sprite's Push affordance as an executable world
interaction. The generic interaction scan now carries a pushable blocker into
the portal opportunity and approaches it from the reachable side.

Final-run evidence:

- sprite 37 is observed as `interactive`, `block=1`, `push=1` at 61 seconds;
- wall 36 (sector 5 to sector 65) gets an interaction with
  `blocker_sprite=37` at 67 seconds;
- the boundary changes from blocked to traversable at 76 seconds;
- the bot crosses wall 36 into sector 65 at 77 seconds.

There are no map-name, sprite-ID, wall-ID, sector-ID, or coordinate checks in
the implementation.

### Sector 17 and sprite 33

Sector 17 is narrow and produces only six useful navigation cells, but the bot
is not physically trapped there and does leave it. The earlier apparent trap
was a downstream goal/state failure, not evidence that sprite 33 corrupts
support detection or removes every exit. No sprite-33 special case was added.

### Remaining work and termination

The final E1M1 run does not declare the map exhausted: it reaches the simulated
600-second cap with unresolved prerequisites and frontiers. It enters sector
65, later reaches upper sector 90, and selects the skull-key-side frontier
(wall 538 toward sector 145) at 444 seconds. Execution of that route eventually
loops and is temporarily suppressed; the bot then returns to other unresolved
interactions. The final run therefore does **not** collect the skull key. This
is now an observable path-execution/priority issue, not a false `NO_IDEAS`
termination, and remains high-value follow-up work.

### Ordinary solid sprites as traversal support

The old mesh treated upright blocking sprites primarily as obstacle volumes.
It could represent floor-aligned sprite bridges, but did not seed conservative
top support or connect it with physically simulated jumps.

The navigation graph now:

- obtains upright-solid top and bottom extents from Build's collision helper;
- seeds a top surface only when `GetZRangeAtXYZ` confirms that the player's
  real clip circle can stand there with sufficient headroom;
- retains sector, support object, and support Z in every cell identity;
- simulates standing-jump reach to create support-to-support and
  floor-to-support links;
- rejects jumps that never reach the destination height, cross a sector wall,
  or intersect an unrelated blocking sprite;
- requires an actual grounded landing on the intended sprite before completing
  a jump step;
- can use an established sprite top as the takeoff support for a higher target.

The standing-jump simulator also had a concrete takeoff bug: it used an
artificially remote floor on the first frame, so the player was considered
airborne before horizontal acceleration was applied. The first frame now uses
the real takeoff floor and later frames use the destination floor.

Final E1M1 evidence includes a grounded landing on sprite 453 at 99 seconds, a
grounded landing on sprite 96 at 423 seconds, entry into sector 90 at about 425
seconds, and a high-target takeoff from sprite 1 at 442 seconds. This directly
addresses the previous behavior of avoiding a blocking sprite or running into
its side without understanding its top.

No tile, voxel, gameplay type, or map identity participates in this decision.
An upright solid with a finite collision volume can contribute a top support.
A wall-aligned sprite is a two-dimensional collision sheet without a finite
top footprint, while floor/slope-aligned support continues through the same
engine-confirmed surface enumeration used by sprite bridges.

### AGTST12 collision-support chain

AGTST12 originally stalled in the starting sector while running at walls. The
engine visual extent and collision extent of one upright solid differ by 128 Z
units; probing from the visual top began inside the collider and falsely found
the sector floor. Navigation now uses `spriteheightofs(..., 1)`, the same
collision extent convention consumed by Build, then asks `GetZRangeAtXYZ` to
confirm each candidate footprint.

The final run lands in order on two upright collision volumes and two sprite
bridge surfaces, enters the raised destination sector, and completes in 15
simulated seconds. Five planned jumps were attempted and all five succeeded.
The graph gives high-rise jumps a nonlinear cost, so intermediate physical
supports are preferred without making a direct jump impossible.

### Exact vector targets and opportunity identity

AGTST11 completes in 17 seconds with each remote damage target confirmed from
the authoritative post-shot callback. Masked vector-triggered walls exposed a
related physical-aiming case in AGTST5: their geometric midpoint may be a
transparent pixel. The bot now samples a bounded set of points on the authored
wall surface and accepts only a point whose canonical engine vector resolves
to that exact wall. Masked-wall hit code 4 is treated as a wall hit both before
and after firing.

AGTST5 also exposed a systematic ledger identity collision. A structured
interaction key could already be in the 4,000,000 range; adding the former
1,000,000 prefix aliased pickup sprite IDs such as 5,000,002. Interaction
opportunities now occupy a disjoint 10,000,000-based namespace, so resolving a
pickup cannot accidentally return a locked door. AGTST5 completes in 58
seconds in the final full-suite run.

### AGTST3 steering ownership

The rapid left/right motion beginning near 23 seconds was not an identity
collision. A valid nav route intentionally emitted a neutral frame while its
jump state changed from ALIGN to TAKEOFF. `steerPortal` interpreted zero input
as route rejection and selected a different fallback target in the same tick;
the route selected its original target again on the next decision.

A route now retains steering ownership until it explicitly invalidates itself,
including deliberate neutral transition frames. AGTST3 improves from 78 to 39
seconds, route plans fall from 195 to 4, and its loop-break count falls from 1
to 0.

### AGTST3 repeated waypoint jumps

The later repeated jumping between roughly 20 and 37 seconds had a separate
cause. `navigateTo` initialized every step's traversal capability from the
final destination. When the last portal required a jump, preceding ordinary
`WALK` cells therefore inherited `kTraversalJumpable` and emitted unnecessary
jumps in place.

Traversal capability is now derived from the active route edge. Walking cells
remain walking cells; only an explicit jump edge receives jump capability.
AGTST3 completes in 25 simulated seconds, with jump input confined to the
intended gap/final portal rather than intermediate waypoints.

### Maskwall continuation

The new support chain composes far enough to discover wall 792 from upper
sector 90. It is classified as a blocked, actionable boundary toward sector 23.
The run does not destroy/cross it: sector 23 is only 1024 units wide and is
currently rejected as lacking player-sized standable space. Repeated selection
of the high vector target also competes with the skull-key route. The sprite
support capability is therefore present, while thin-sector transit and final
destructible-boundary execution remain unresolved.

### AGTST10 jumping and explosive safety

The repeated wall jump was not a legitimate planned shortcut. Same-sector
progress reset the local jump-attempt state, allowing the same recovery jump
to be emitted repeatedly. Same-sector motion now preserves the bounded attempt
state so the failed hypothesis is retired and normal walking can resume.

Explosion planning now obtains TNT/environmental blast radii from the engine's
explosion data, derives a player-radius safety envelope, tracks the delivered
projectile/detonation phase, and retreats before intentional detonation when
inside that envelope. AGTST10 completes in 34 simulated seconds without
intentional self-damage.

### Moving geometry and deferred prerequisites

AGTST9 exposed a stale deferred frontier: wall 24 was suppressed at 17 seconds,
then live geometry changed and the navigation topology rebuilt at 18 seconds,
but the old cooldown remained. Frontier suppressions recorded against the old
physical pose are now re-armed after a genuine live-geometry settle, not after
ordinary discovery of another static sector.

Ranged actuators that require standing on a carrier are now ranked at that
support location. Until the support has been reached, its ordinary exploration
frontier is selected first. This is intentionally limited to vector/ranged
carrier actions: ordinary Use controls may call a reusable elevator back after
the bot steps or falls off it. AGTST8 and AGTST9 exercise both sides of this
distinction.

### ROR / overlapping-sector hypothesis

The observed E1M1 oscillation was not caused by sector 65 and sector 90 being
collapsed in XY. `NavCell` identity already includes owning sector, Z, and a
`SupportRef`; grid lookup and nearest-cell projection are sector-scoped; and
cross-sector graph links are created only by explicit portal/traversal edges.
Telemetry during the late oscillation alternates real sector transitions and
goals rather than snapping a single point between overlapping mesh layers.

The actual late behavior is repeated competition between the upper actuator,
the key-side frontier, and currently unexecutable paths around thin/destructible
geometry. No speculative ROR special case or duplicate architecture test was
added; the existing architecture suite already contains same-XY/different-Z
and stacked-support isolation tests.

### AGTST13 deferred key and return-route reconsideration

An authoritative rejected Use action previously made a sprite interaction
terminal even when the engine reported a missing key. The interaction now
records the discovered key prerequisite, remains deferred, and is re-armed by
the generic inventory-capability change path when that key is acquired.

Return planning also exposed two route-lifetime problems. The coarse known
graph admitted directed boundaries that were neither walkable nor jumpable,
and an interaction could oscillate between exact support navigation and the
broader sector transport graph as it crossed layered sectors. Impossible
directed boundaries are now excluded. Interaction objectives choose a route
family when acquired and retain it: a local physical approach keeps the exact
support mesh, while a remembered remote opportunity recomputes through known
directed transport. A committed gap transaction also owns execution through
landing instead of being cancelled by its parent's time budget on the launch
frame.

AGTST13 now learns key 1 from the exit button, continues exploring, acquires
the key, re-arms the deferred interaction, and returns over the valid alternate
route rather than trying to reverse the earlier one-way drop. It completes in
100 simulated seconds. The same route-family rule preserves AGTST8's exact
moving-support/sprite-bridge approach.

The descent into sector 4 is still permitted. This run did not establish that
an already-known reversible route was being ignored at the descent decision,
and the descent is not a permanent softlock: after pickup, the directed model
discovers the 4 -> 0 -> 3 return opening and then recomputes the remaining route
to sector 9. Penalizing that descent without stronger evidence would make a
valid exploration move look impossible.

## Regression results

| Map | Result | Simulated time |
|---|---|---:|
| AGTST1 | Completed | 29 s |
| AGTST2 | Completed | 79 s |
| AGTST3 | Completed | 25 s |
| AGTST4 | Completed | 370 s |
| AGTST5 | Completed | 50 s |
| AGTST6 | Completed | 73 s |
| AGTST7 | Completed | 27 s |
| AGTST8 | Completed | 80 s |
| AGTST9 | Completed | 76 s |
| AGTST10 | Completed | 36 s |
| AGTST11 | Completed | 17 s |
| AGTST12 | Completed | 15 s |
| AGTST13 | Completed | 100 s |

Additional checks:

- navigation architecture tests: 54/54 passed;
- full `nblood`, `rednukem`, and `pcexhumed` build: passed;
- `git diff --check`: clean;
- final E1M1 `nodudes`: timeout at 600 simulated seconds, 23/155 sectors
  entered and 27 observed; sector 65 and upper sector 90 reached, wall 792
  discovered, skull key not collected in the final run.

## Remaining high-value work

1. Represent narrow technical sectors as transit without requiring them to be
   standable destinations, then retry the wall-792 continuation.
2. Prevent a repeatedly unexecutable high interaction from reclaiming priority
   over a reachable key/frontier after its bounded budget expires.
3. Reduce repeated route reconstruction when start and goal cells are in known
   disconnected support components; this is the dominant late E1M1 telemetry
   noise and a likely source of avoidable planner cost.

## Exploration and recovery audit (AGTST4, E1M1, E1M7, E1M8, E3M1)

This follow-up used the reported timestamps as diagnostic examples. None of
the changes below checks a map name, object number, wall number, sector number,
tile, or coordinate.

### Findings and generic changes

- **AGTST4 sprite 74:** telemetry identifies it as a zero-clip-distance gib.
  It was incidental collision geometry, not the selected target and not a
  destruction opportunity. The active failure was an interaction approach
  repeatedly rebuilding/losing its route. No gib-specific rule and no policy
  to destroy clutter was added.
- **Stable support routes:** a valid route on the same topology/support is now
  retained instead of being rebuilt every rendered frame. This removes the
  rapid direction changes caused by repeatedly changing first waypoints.
- **Pickup physical layers:** optional pickups are local only when their
  navigation cell is on the player's current support layer (or within an
  ordinary walkable step). A pickup visible through overlapping geometry no
  longer captures navigation before its physical layer is reachable. Keys and
  supplies satisfying a known capability prerequisite remain promotable.
- **Exact pickup support:** pickup execution carries the destination's actual
  support identity through planning. Reaching the same XY/sector on the wrong
  layer is not success; a bounded failed support hypothesis is rejected rather
  than reconstructed forever.
- **CPU cost:** standing-jump reach is cached by height delta during each
  topology build. Equal-height support pairs no longer repeat the same
  240-frame physical simulation. AGTST8 also benefits from retained routes.
- **Interaction outcomes:** accepted use is not automatically useful. Repeated
  accepted/no-world-delta attempts consume a bounded budget, while a real
  geometry/state change resets the budget. Conversely, an accepted target
  disappearing from a stable aligned pose is treated as evidence that a
  moving mechanism changed the world.
- **Physical-affordance identity:** collision discovery reuses an already known
  interaction for the same wall. It cannot create a second opportunity that
  operates a reversible door again and undoes the first action.
- **Thin transit geometry:** a portal into a sector with no standable cell is
  retained as a low-priority transit hypothesis if its opening physically fits
  the player. It is investigated only after concrete frontiers, retires when a
  mechanism is found, and is reconsidered only if its geometry changes.
- **Short causal memory:** a mechanism discovered by collision immediately
  after a deliberate boundary probe inherits that route for two seconds. If
  operating it changes the world, the bot immediately follows through the
  probed boundary instead of forgetting why it used the mechanism.

### Retail diagnostic evidence

| Map | Before | After | Interpretation |
|---|---:|---:|---|
| E1M1 `nodudes` | 22/155 sectors, 315 plans | 22/155 sectors, 215 plans | wall 1389 is no longer duplicated/re-operated; the bot abandons the bounded hypothesis and continues, although the later map remains unresolved |
| E1M7 `nodudes` | 22/53 sectors, 2,226 plans | 29/53 sectors, 576 plans | the different-layer weapon no longer causes the reported 1:29 bridge rotation; it is collected later when physically local |
| E1M8 `nodudes` | 5/433 sectors, 2,287 plans | 56/433 sectors, 539 plans | retaining a valid same-support route removes the starting-area planner churn |
| E3M1 `nodudes` | 13/382 sectors, door not found | 21/382 sectors, 22 interactions known | collision discovers the sector-207 Wallpush mechanism, Use changes it, and the bot immediately crosses wall 13 through sectors 206/208 |

Retail runs remain soft diagnostics and stop at 600 simulated seconds. The
after results above do not claim level completion.

### Final hard regression contract

| Map | Result | Simulated time |
|---|---|---:|
| AGTST1 | Completed | 31 s |
| AGTST2 | Completed | 81 s |
| AGTST3 | Completed | 25 s |
| AGTST4 | Completed | 406 s |
| AGTST5 | Completed | 52 s |
| AGTST6 | Completed | 73 s |
| AGTST7 | Completed | 27 s |
| AGTST8 | Completed | 81 s |
| AGTST9 | Completed | 76 s |
| AGTST10 | Completed | 36 s |
| AGTST11 | Completed | 17 s |
| AGTST12 | Completed | 15 s |
| AGTST13 | Completed | 102 s |

Post-change verification: 54/54 navigation architecture tests passed; the
full `nblood`, `rednukem`, and `pcexhumed` build passed; `git diff --check` was
clean.
