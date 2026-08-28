# LLMapper bot architectural cleanup

Date: 2026-08-22

This document is the audit log for the reduction pass.  The intended data
flow is:

    engine truth -> world model -> physical transitions -> work
        -> plan -> one executing transition -> typed result -> knowledge

The first rule of this pass is replacement rather than layering.  Code is
not moved into modules until its concepts have survived this inventory.

## Baseline complexity

| Measure | Baseline |
|---|---:|
| `bot.cpp` | 17,365 lines |
| `nav_kernel.cpp` | 1,170 lines |
| `nav_kernel.h` | 831 lines |
| `player_capability.h` | 448 lines |
| Mutable fields directly owned by `LLMapperBot::Impl` | over 200 |
| Persistent/temporary failure stores | 5 |
| Intent/task projections | 5 |
| Executable route stores | 2 |
| Jump executors | 2 |
| Opened-route/causal-continuation representations | 5 |

The line baseline includes the live debug renderer because it directly reads
the bot's internal representations.  Generated corpus output is excluded.

## Structural inventory

| Responsibility | Current owners | Duplicated or overloaded concepts | Cleanup decision |
|---|---|---|---|
| Observation | `observeWorld`, `Observation`, `VisibleObject`, `Portal` derivation | `Portal` combines boundary fact, mechanism state, traversal classification and planner hints | Split conceptually into observed boundary facts and derived transitions; do not let `Portal` remain executable truth. |
| World knowledge | `knownGraph`, doors, interactions, object memory, observed/visited sets | sector graph and physical graph both claim connectivity; transient action state lives in persistent interaction memory | Retain observed facts; remove executable authority from `knownGraph`; split action transaction state from mechanism knowledge. |
| Support detection | `standableSurfacesAt`, nav-cell construction, live `GetZRange` helpers | containing sector, sector floor and concrete collision support are still occasionally interchangeable | Concrete engine floor-hit identity is authoritative. |
| Hazard classification | sector damage queries, learned damage, `hazardSupports` | historical sector-level inheritance conflicts with sprite support | Hazard belongs to the collision support that causes `actTouchFloor`. |
| Traversal classification | `Portal` flags, `classifyTraversal`, nav-link construction, local probes | several definitions of walk/step/jump/drop | One transition builder using engine-derived capability queries. |
| Navigation graph | `knownGraph`, `NavCell`/`NavLink`, dynamic portal links | sector adjacency and pose graph disagree | Pose graph is executable truth; sector graph becomes an observation index only. |
| Route planning | `navRoute`, `portalPlan`, direct crossing, navigation detours | two route buffers plus direct and fallback steering | Delete `portalPlan` and direct gap executor; one plan contains concrete transition IDs/steps. |
| Jump execution | `JumpExecutionState`, `steerGapJump`/`continueGapJump` | two solvers and two lifecycle machines for the same physical action | One jump executor owned by a validated jump transition. |
| Interaction discovery | `InteractionCandidate`, surfaces in `InteractionMemory`, collision discovery | logical receiver and executable physical surface can overwrite each other | Preserve all physical actuator surfaces under one mechanism identity. |
| Mechanism knowledge | doors, interactions, causal graph | `DoorMemory` overlaps generic interaction state | Fold door-specific knowledge into mechanism/boundary facts where possible. |
| Damage interaction | fields inside `InteractionMemory`, damage executor | persistent knowledge contains projectile/weapon transaction state | Extract one short-lived action transaction/executor. |
| Work | `Opportunity`, mission-named IDs, `Objective`, movement goal | the same work is copied and translated repeatedly | Ledger work item is persistent; active plan references its stable ID. Delete Mission/Objective projections rather than wrapping them. |
| Failure memory | `failedEdges`, `navEdgeFailures`, local signatures, unreachable sets, attempt counters | failures key different representations and some are time/count driven | One attempt ledger keyed by physical transition plus relevant world revision. |
| Causal continuation | follow-through, causal continuation cells, opened routes, pending opened routes, reopened-return window | one action consequence has five identities and several timers | One action transaction owns observed world deltas and enabled transitions/work. |
| Moving supports | stable poses, support ride settle, dynamic prerequisites | support occupancy is copied into special mechanism modes | Movement changes pose connectivity; current support plus action consequence is sufficient. |
| Stall/recovery | movement, waypoint, objective, stationary and loop watchdogs | watchdogs can compete and mutate planning | Executor watchdog produces a typed failure only; it cannot invent another action. |
| Telemetry/debug | event strings, renderer, corpus | renderer traverses raw duplicate graph state and can expose quadratic clutter | Render only authoritative poses/transitions/work/active plan. |

## First deletion order

1. Delete the local `portalPlan` and gap-jump controller.  Both duplicate
   `navRoute` plus the jump transition executor.
2. Replace failure stores with one transition/revision attempt ledger.
3. Replace opened-route/follow-through timing families with an action
   transaction carrying observed deltas and enabled physical transitions.
4. Make the persistent work ledger the only task identity and replace
   `Objective` with a small active plan reference.
5. After those deletions, extract engine physics, world facts, transition
   graph, work policy, execution and diagnostics into ownership modules.

## Physics defects already exposed by the inventory

- The former walk-step threshold was `4096`; Build's player clip test uses
  `CLIPCURBHEIGHT` (`256`).  Navigation now reads that engine constant.
- Sprite landing reach added the player foot offset to a support-plane delta
  even though the motion model already adds the foot offset.  That mixed an
  interaction/body coordinate with support Z and created false long jumps.
- A jump solver returned the least unstable candidate even when the player
  could not remain on the destination support.  The replacement validator
  must prove the engine continues to name the target collision support until
  ground drag settles the player.
- Damaging-sector state was inherited by every support inside the sector.
  Damage is now keyed by the concrete floor-hit support, so sprite platforms
  above a damaging floor remain safe.

This inventory is deliberately written before modularization.  Later sections
will record the exact concepts removed, regression points, and final ownership
boundaries.
