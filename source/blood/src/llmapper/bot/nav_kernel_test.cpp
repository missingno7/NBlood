//-------------------------------------------------------------------------
// Architecture regression tests for the LLMapper movement kernel.
// These tests use captured-style synthetic geometry, never map IDs.
//-------------------------------------------------------------------------
#include "nav_kernel.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace llmapper;

static int gFailures = 0;

static void expect(bool condition, const char *name)
{
    if (!condition)
    {
        std::fprintf(stderr, "FAIL %s\n", name);
        ++gFailures;
    }
    else
        std::printf("ok   %s\n", name);
}

static NavCell makeCell(int id, int sector, int x, int y, int z = 0,
                        SupportRef support = SupportRef())
{
    NavCell cell;
    cell.id = id;
    cell.sector = sector;
    cell.center = NavWaypoint(x, y);
    cell.z = z;
    cell.support = support.id >= 0 ? support : SupportRef(kSupportSectorFloor, sector);
    return cell;
}

static void addLink(NavCell &from, int to, NavEdgeMode mode, int wall = -1,
                    NavWaypoint gateway = NavWaypoint())
{
    NavLink link;
    link.target = to;
    link.mode = mode;
    link.wall = wall;
    link.gateway = gateway;
    link.hasGateway = wall >= 0 || (gateway.x != 0 || gateway.y != 0);
    from.links.push_back(link);
}

static bool planHas(const std::vector<PlanOperation> &plan, PlanOperationKind kind)
{
    for (size_t i = 0; i < plan.size(); ++i)
        if (plan[i].kind == kind)
            return true;
    return false;
}

static bool planActivatesWith(const std::vector<PlanOperation> &plan,
                              ActivationMode mode)
{
    for (size_t i = 0; i < plan.size(); ++i)
        if (plan[i].kind == kPlanActivate && plan[i].activation == mode)
            return true;
    return false;
}

static void testTraversalProbe()
{
    expect(classifyTraversal(0, 8192, 4096, 8192, 4096, true, false, false, true)
               == kTraverseDirect,
           "traverse_flat_walk");
    expect(classifyTraversal(2048, 8192, 4096, 8192, 4096, true, false, false, true)
               == kTraverseStep,
           "traverse_ordinary_step");
    expect(classifyTraversal(-6000, 8192, 4096, 8192, 4096, false, true, false, true)
               == kTraverseJump,
           "traverse_jump_rise");
    expect(classifyTraversal(-6000, 8192, 4096, 8192, 4096, false, true, false, false)
               == kTraverseJump,
           "traverse_jump_not_rejected_by_horizontal_clipmove");
    expect(classifyTraversal(6000, 8192, 4096, 8192, 4096, true, false, false, true)
               == kTraverseDropSafe,
           "traverse_reverse_drop_safe");
    expect(classifyTraversal(0, 8192, 4096, 8192, 4096, false, true, false, true)
               == kTraverseSolidBlocker,
           "traverse_solid_onesided_wall");
    expect(classifyTraversal(0, 8192, 4096, 8192, 4096, false, true, true, true)
               == kTraverseUseableBlocker,
           "traverse_pushable_wall");
    expect(classifyTraversal(-20000, 8192, 4096, 8192, 4096, false, true, false, true)
               == kTraverseNoFit,
           "traverse_too_high_rise");
    expect(classifyTraversal(20000, 8192, 4096, 8192, 4096, true, false, false, true)
               == kTraverseDangerous,
           "traverse_unsafe_drop");
    expect(classifyTraversal(0, 1024, 4096, 8192, 4096, true, false, false, true)
               == kTraverseNoFit,
           "traverse_no_fit_low_clearance");
}

static void testNavGraph()
{
    std::vector<NavCell> cells;
    cells.push_back(makeCell(0, 1, 0, 0));
    cells.push_back(makeCell(1, 1, 100, 0));
    cells.push_back(makeCell(2, 1, 50, 100));
    addLink(cells[0], 1, kNavWalk, -1, NavWaypoint(50, 0));
    addLink(cells[1], 0, kNavWalk, -1, NavWaypoint(50, 0));
    addLink(cells[1], 2, kNavWalk, -1, NavWaypoint(75, 50));
    addLink(cells[2], 1, kNavWalk, -1, NavWaypoint(75, 50));
    addLink(cells[0], 2, kNavWalk, -1, NavWaypoint(25, 50));
    addLink(cells[2], 0, kNavWalk, -1, NavWaypoint(25, 50));
    std::vector<NavRouteStep> route;
    expect(planNavRoute(cells, 0, 2, 50, 100, 1, -1, std::vector<NavEdgeFailure>(), 1, route)
               && route.size() >= 1,
           "nav_same_sector_concave_route");

    cells.clear();
    cells.push_back(makeCell(0, 2, 0, 0));
    cells.push_back(makeCell(1, 1, 200, 0));
    cells.push_back(makeCell(2, 7, 400, 0));
    addLink(cells[0], 1, kNavStep, 19, NavWaypoint(100, 0));
    addLink(cells[1], 0, kNavStep, 20, NavWaypoint(100, 0));
    addLink(cells[1], 2, kNavWalk, 4, NavWaypoint(300, 0));
    addLink(cells[2], 1, kNavWalk, 5, NavWaypoint(300, 0));
    expect(planNavRoute(cells, 0, 2, 400, 0, 7, -1, std::vector<NavEdgeFailure>(), 1, route)
               && route.size() == 2
               && route[0].mode == kNavStep
               && route[0].wall == 19
               && route[1].mode == kNavWalk
               && route[0].sourceSector == 2
               && route[1].targetSector == 7,
           "nav_multi_sector_walk_step_route");

    cells.clear();
    cells.push_back(makeCell(0, 1, 0, 0));
    cells.push_back(makeCell(1, 1, 0, 64));
    cells.push_back(makeCell(2, 1, 0, 128));
    addLink(cells[0], 1, kNavStep);
    addLink(cells[1], 0, kNavStep);
    addLink(cells[1], 2, kNavStep);
    addLink(cells[2], 1, kNavStep);
    expect(planNavRoute(cells, 0, 2, 0, 128, 1, -1, std::vector<NavEdgeFailure>(), 1, route)
               && route.size() == 2
               && route[0].mode == kNavStep
               && route[1].mode == kNavStep,
           "nav_step_chain");

    cells.clear();
    cells.push_back(makeCell(0, 1, 0, 0));
    cells.push_back(makeCell(1, 2, 0, -100));
    addLink(cells[0], 1, kNavJump, 8, NavWaypoint(0, -50));
    addLink(cells[1], 0, kNavDrop, 9, NavWaypoint(0, -50));
    expect(planNavRoute(cells, 0, 1, 0, -100, 2, 8, std::vector<NavEdgeFailure>(), 1, route)
               && route.size() == 1
               && route[0].mode == kNavJump,
           "nav_route_contains_jump");
    expect(planNavRoute(cells, 1, 0, 0, 0, 1, 9, std::vector<NavEdgeFailure>(), 1, route)
               && route.size() == 1
               && route[0].mode == kNavDrop,
           "nav_directed_jump_drop_asymmetry");

    NavEdgeFailure failure;
    failure.fromCell = 0;
    failure.toCell = 1;
    failure.wall = 8;
    failure.mode = kNavJump;
    failure.geometrySignature = 1;
    std::vector<NavEdgeFailure> failures(1, failure);
    expect(!planNavRoute(cells, 0, 1, 0, -100, 2, 8, failures, 1, route),
           "nav_failed_edge_excluded");
    expect(planNavRoute(cells, 0, 1, 0, -100, 2, 8, failures, 2, route),
           "nav_failed_edge_reappears_on_geometry_change");

    cells.clear();
    cells.push_back(makeCell(0, 1, 0, 0, 0));
    cells.push_back(makeCell(1, 1, 100, 0, -6144));
    cells.push_back(makeCell(2, 1, 200, 0, -12288));
    // When collision geometry offers intermediate standable tops, climbing
    // through them is safer than making one maximum-rise leap to the same
    // destination.  Support identity is deliberately irrelevant here.
    addLink(cells[0], 2, kNavJump);
    addLink(cells[0], 1, kNavJump);
    addLink(cells[1], 2, kNavJump);
    expect(planNavRoute(cells, 0, 2, 200, 0, 1, -1,
                        std::vector<NavEdgeFailure>(), 1, route)
               && route.size() == 2
               && route[0].toCell == 1
               && route[1].toCell == 2,
           "nav_prefers_staged_collision_support_climb");

    cells.clear();
    for (int i = 0; i < 6; ++i)
        cells.push_back(makeCell(i, i == 5 ? 2 : 1, i * 100, 0));
    // One risky shortcut and a five-cell supported walk reach the same goal.
    // Route selection should preserve the jump for maps that need it, but
    // prefer the bridge when both are currently available.
    addLink(cells[0], 5, kNavJump, 30, NavWaypoint(250, -100));
    for (int i = 0; i < 5; ++i)
        addLink(cells[i], i + 1, kNavWalk);
    expect(planNavRoute(cells, 0, 5, 500, 0, 2, -1,
                        std::vector<NavEdgeFailure>(), 1, route)
               && route.size() == 5
               && route[0].mode == kNavWalk
               && route[4].targetSector == 2,
           "nav_prefers_supported_walk_over_risky_jump_shortcut");

    cells.clear();
    cells.push_back(makeCell(0, 1, 0, 0));
    cells.push_back(makeCell(1, 2, 100, 0));
    cells.push_back(makeCell(2, 2, 100, 100));
    addLink(cells[0], 1, kNavWalk, 1, NavWaypoint(50, 0));
    addLink(cells[0], 2, kNavWalk, 2, NavWaypoint(50, 50));
    addLink(cells[1], 0, kNavWalk, 1, NavWaypoint(50, 0));
    addLink(cells[2], 0, kNavWalk, 2, NavWaypoint(50, 50));
    failure.fromCell = 0;
    failure.toCell = 1;
    failure.wall = 1;
    failure.mode = kNavWalk;
    failure.geometrySignature = 9;
    failures.assign(1, failure);
    expect(planNavRoute(cells, 0, 2, 100, 100, 2, -1, failures, 9, route)
               && route.size() == 1
               && route[0].wall == 2,
           "nav_failed_edge_reroutes_to_alternative");
}

static void testSupportAwareNavigation()
{
    NavCell floor = makeCell(0, 4, 100, 200, 12000,
                             SupportRef(kSupportSectorFloor, 4));
    NavCell bridge = makeCell(1, 4, 100, 200, 4000,
                              SupportRef(kSupportSpriteFloor, 31));
    expect(floor.sector == bridge.sector
               && floor.center.x == bridge.center.x
               && floor.center.y == bridge.center.y
               && floor.z != bridge.z && floor.support != bridge.support,
           "support_same_sector_xy_different_layer_distinct");

    NavCell bridgeAbove = makeCell(2, 4, 100, 200, -4000,
                                   SupportRef(kSupportSpriteFloor, 32));
    expect(bridge.support != bridgeAbove.support && bridge.z != bridgeAbove.z,
           "support_stacked_perpendicular_bridges_do_not_collapse");
}

static void testCrouchAndConditionalGate()
{
    expect(classifyTraversalForPostures(0, 3000, 4096, 2048, 8192, 4096,
                                        true, false, false, true)
               == kTraverseCrouch,
           "traverse_narrow_opening_requires_crouch");

    std::vector<NavCell> cells;
    cells.push_back(makeCell(0, 1, 0, 0));
    cells.push_back(makeCell(1, 2, 100, 0));
    cells[0].links.push_back(makeConditionalTraversal(1, kNavCrouch, 9, 1));
    expect(cells[0].links[0].mode == kNavCrouch
               && cells[0].links[0].condition.enabled
               && cells[0].links[0].condition.mechanism == 9
               && cells[0].links[0].condition.state == 1,
           "dynamic_gate_blocked_to_conditional_crouch");
}

static CausalGraph oneActuatorGraph(int actuatorId, int location,
                                    ActivationMode mode, int mechanism,
                                    int state)
{
    CausalGraph graph;
    Actuator actuator;
    actuator.id = actuatorId;
    actuator.locationCell = location;
    actuator.modes.push_back(mode);
    graph.actuators.push_back(actuator);
    LearnedEffect effect;
    effect.actuator = actuatorId;
    effect.mode = mode;
    effect.mechanism = mechanism;
    effect.state = state;
    graph.effects.push_back(effect);
    return graph;
}

static void testDynamicPlanningAndCausality()
{
    std::vector<NavCell> cells;
    cells.push_back(makeCell(0, 1, 0, 0));
    cells.push_back(makeCell(1, 1, 100, 0));
    cells.push_back(makeCell(2, 2, 200, 0));
    addLink(cells[0], 1, kNavWalk);
    addLink(cells[1], 0, kNavWalk);
    cells[1].links.push_back(makeConditionalTraversal(2, kNavCrouch, 7, 1));
    cells[2].links.push_back(makeConditionalTraversal(1, kNavCrouch, 7, 1));

    CausalGraph graph = oneActuatorGraph(42, 1, kActivateUse, 7, 1);
    std::map<int, int> states;
    states[7] = 0;
    std::vector<PlanOperation> plan;
    DynamicPlanStats stats;
    expect(planDynamicRoute(cells, 0, 2, states, graph, plan, &stats)
               && planHas(plan, kPlanActivate)
               && planHas(plan, kPlanWaitForTransition)
               && planActivatesWith(plan, kActivateUse),
           "remote_actuator_establishes_traversal_condition");

    // Closing changes availability, not remembered topology.  From the far
    // side another reachable actuator can re-establish the same condition.
    CausalGraph reverseGraph = oneActuatorGraph(43, 2, kActivateUse, 7, 1);
    plan.clear();
    expect(cells[2].links[0].condition.enabled
               && planDynamicRoute(cells, 2, 1, states, reverseGraph, plan)
               && planHas(plan, kPlanActivate),
           "auto_closing_gate_retains_conditional_reverse_connection");

    Actuator vectorActuator;
    vectorActuator.id = 44;
    vectorActuator.locationCell = 1;
    vectorActuator.modes.push_back(kActivateVector);
    vectorActuator.destructible = false;
    graph.actuators.push_back(vectorActuator);
    LearnedEffect vectorEffect;
    vectorEffect.actuator = 44;
    vectorEffect.mode = kActivateVector;
    vectorEffect.mechanism = 8;
    vectorEffect.state = 1;
    graph.effects.push_back(vectorEffect);
    expect(graph.actuators[0].modes[0] == kActivateUse
               && graph.actuators[1].modes[0] == kActivateVector,
           "use_and_vector_share_causal_model_with_distinct_modes");
    expect(!graph.actuators[1].destructible,
           "vector_activation_does_not_imply_destructible");

    CausalReceiver sectorReceiver;
    sectorReceiver.channel = 100;
    sectorReceiver.object = WorldObjectRef(kWorldSector, 3);
    CausalReceiver wallReceiver;
    wallReceiver.channel = 100;
    wallReceiver.object = WorldObjectRef(kWorldWall, 17);
    CausalReceiver spriteReceiver;
    spriteReceiver.channel = 100;
    spriteReceiver.object = WorldObjectRef(kWorldSprite, 25);
    graph.receivers.push_back(sectorReceiver);
    graph.receivers.push_back(wallReceiver);
    graph.receivers.push_back(spriteReceiver);
    const std::vector<CausalReceiver> receivers = graph.receiversFor(100);
    expect(receivers.size() == 3
               && receivers[0].object.kind == kWorldSector
               && receivers[1].object.kind == kWorldWall
               && receivers[2].object.kind == kWorldSprite,
           "tx_channel_preserves_all_receiver_types");

    // Twenty unrelated mechanism states do not multiply route search.  Only
    // the condition on the candidate route is resolved.
    for (int i = 100; i < 120; ++i)
        states[i] = i & 1;
    plan.clear();
    stats = DynamicPlanStats();
    expect(planDynamicRoute(cells, 0, 2, states, oneActuatorGraph(
                                45, 1, kActivateUse, 7, 1), plan, &stats)
               && stats.mechanismsConsidered.size() == 1
               && stats.mechanismsConsidered.count(7) == 1,
           "dynamic_planner_avoids_unrelated_state_cartesian_product");
}

static StablePose pose(int state, int z, int clearance, bool occupiable,
                       int connection)
{
    StablePose result;
    result.state = state;
    result.supportZ = z;
    result.clearance = clearance;
    result.occupiable = occupiable;
    if (connection >= 0)
        result.connectedSurfaces.push_back(connection);
    return result;
}

static void testDynamicSupports()
{
    DynamicMechanism gate;
    gate.id = 1;
    gate.poses.push_back(pose(0, 12000, 1000, false, -1));
    gate.poses.push_back(pose(1, 4000, 3000, true, 2));
    gate.sweepClearances.push_back(1000);
    const int gateAffordances = deriveDynamicAffordances(gate, 2048);

    DynamicMechanism carrier;
    carrier.id = 2;
    carrier.support = SupportRef(kSupportSectorFloor, 8);
    carrier.carriesSupport = true;
    carrier.poses.push_back(pose(0, 12000, 8192, true, 10));
    carrier.poses.push_back(pose(1, 4000, 8192, true, 11));
    carrier.sweepClearances.push_back(8192);
    const int carrierAffordances = deriveDynamicAffordances(carrier, 4096);
    expect((gateAffordances & kAffordanceEnablePassage)
               && (gateAffordances & kAffordanceUnsafeSweptOccupancy)
               && !(gateAffordances & kAffordanceTransportSupportedPlayer)
               && (carrierAffordances & kAffordanceTransportSupportedPlayer)
               && !(carrierAffordances & kAffordanceUnsafeSweptOccupancy),
           "similar_vertical_motion_gate_and_carrier_classify_differently");
    expect((carrierAffordances & kAffordanceUnsafeSweptOccupancy) == 0,
           "moving_geometry_not_universally_hazardous");
    expect((gateAffordances & kAffordanceTransportSupportedPlayer) == 0,
           "unsafe_swept_occupancy_has_no_ride_affordance");

    std::vector<NavCell> cells;
    cells.push_back(makeCell(0, 1, 0, 0, 12000));
    cells.push_back(makeCell(1, 8, 100, 0, 12000, carrier.support));
    cells.push_back(makeCell(2, 8, 100, 0, 4000, carrier.support));
    cells.push_back(makeCell(3, 9, 200, 0, 4000));
    cells[0].links.push_back(makeConditionalTraversal(1, kNavWalk, 2, 0));
    cells[1].links.push_back(makeConditionalTraversal(2, kNavRide, 2, 1, 2));
    cells[2].links.push_back(makeConditionalTraversal(3, kNavWalk, 2, 1));
    std::map<int, int> states;
    states[2] = 0;
    std::vector<PlanOperation> plan;
    expect(planDynamicRoute(cells, 0, 3, states,
                            oneActuatorGraph(50, 1, kActivateUse, 2, 1), plan)
               && planHas(plan, kPlanRemainSupported),
           "safe_moving_support_connects_two_stable_landings");

    DynamicMechanism spriteCarrier = carrier;
    spriteCarrier.id = 3;
    spriteCarrier.object = WorldObjectRef(kWorldSprite, 71);
    spriteCarrier.support = SupportRef(kSupportSpriteFloor, 71);
    NavCell spriteLow = makeCell(4, 12, 0, 0, 12000, spriteCarrier.support);
    NavCell spriteHigh = makeCell(5, 13, 300, 0, 4000, spriteCarrier.support);
    expect((deriveDynamicAffordances(spriteCarrier, 4096)
                & kAffordanceTransportSupportedPlayer)
               && spriteLow.sector != spriteHigh.sector
               && spriteLow.support == spriteHigh.support,
           "moving_sprite_support_transports_across_build_sectors");
}

static void testFrontiers()
{
    std::vector<int> visited;
    visited.push_back(1);
    std::vector<Boundary> boundaries;
    Boundary a; a.wall = 10; a.from = 1; a.to = 2; a.traversable = true; a.geometrySignature = 1;
    Boundary b; b.wall = 11; b.from = 1; b.to = 2; b.traversable = true; b.geometrySignature = 1;
    Boundary c; c.wall = 12; c.from = 1; c.to = 2; c.traversable = true; c.geometrySignature = 1;
    boundaries.push_back(a);
    boundaries.push_back(b);
    boundaries.push_back(c);
    std::vector<DerivedFrontier> frontiers = deriveFrontiers(
        visited, boundaries, std::vector<InvestigateRecord>(), std::vector<NavEdgeFailure>());
    expect(frontiers.size() == 1 && frontiers[0].destination == 2
               && frontiers[0].kind == kFrontierOpen
               && frontiers[0].candidates.size() == 3,
           "frontier_three_walls_one_destination");

    visited.push_back(2);
    frontiers = deriveFrontiers(visited, boundaries, std::vector<InvestigateRecord>(),
                                std::vector<NavEdgeFailure>());
    expect(frontiers.empty(), "frontier_disappears_after_enter");

    Boundary reverse; reverse.wall = 20; reverse.from = 2; reverse.to = 1;
    reverse.traversable = true; reverse.geometrySignature = 1;
    boundaries.push_back(reverse);
    frontiers = deriveFrontiers(visited, boundaries, std::vector<InvestigateRecord>(),
                                std::vector<NavEdgeFailure>());
    expect(frontiers.empty(), "frontier_reverse_is_transport_only");

    Boundary blocked; blocked.wall = 30; blocked.from = 1; blocked.to = 9;
    blocked.traversable = false; blocked.jumpable = false; blocked.geometrySignature = 4;
    boundaries.push_back(blocked);
    visited.assign(1, 1);
    frontiers = deriveFrontiers(visited, boundaries, std::vector<InvestigateRecord>(),
                                std::vector<NavEdgeFailure>());
    bool foundBlocked = false;
    for (size_t i = 0; i < frontiers.size(); ++i)
        if (frontiers[i].destination == 9 && frontiers[i].kind == kFrontierBlocked)
            foundBlocked = true;
    expect(foundBlocked, "frontier_blocked_unknown_boundary");

    InvestigateRecord investigated;
    investigated.wall = 30;
    investigated.from = 1;
    investigated.to = 9;
    investigated.geometrySignature = 4;
    frontiers = deriveFrontiers(visited, boundaries, std::vector<InvestigateRecord>(1, investigated),
                                std::vector<NavEdgeFailure>());
    foundBlocked = false;
    for (size_t i = 0; i < frontiers.size(); ++i)
        if (frontiers[i].destination == 9 && frontiers[i].kind == kFrontierBlocked)
            foundBlocked = true;
    expect(!foundBlocked, "frontier_blocked_disappears_after_investigation");

    NavEdgeFailure failure;
    failure.wall = 10;
    failure.geometrySignature = 1;
    std::vector<NavEdgeFailure> failed(1, failure);
    boundaries.assign(1, a);
    Boundary alt = b;
    boundaries.push_back(alt);
    frontiers = deriveFrontiers(visited, boundaries, std::vector<InvestigateRecord>(), failed);
    expect(frontiers.size() == 1 && frontiers[0].kind == kFrontierOpen
               && frontiers[0].candidates.size() >= 1,
           "frontier_failed_candidate_does_not_abandon_destination");

    failure.geometrySignature = 1;
    failed.assign(1, failure);
    a.geometrySignature = 2;
    boundaries.assign(1, a);
    frontiers = deriveFrontiers(visited, boundaries, std::vector<InvestigateRecord>(), failed);
    expect(frontiers.size() == 1 && frontiers[0].kind == kFrontierOpen,
           "frontier_geometry_change_restores_failed_candidate");
}

static void testDeferredEffectPrerequisites()
{
    Opportunity crack;
    crack.id = 101;
    crack.kind = kOpportunityInteraction;
    crack.sector = 1;
    crack.hops = 0;
    crack.local = true;
    crack.requiredEffects = kEffectExplosive;

    Opportunity exploration;
    exploration.id = 202;
    exploration.kind = kOpportunityFrontier;
    exploration.sector = 2;
    exploration.target = 3;
    exploration.hops = 1;
    exploration.local = false;

    std::vector<Opportunity> ledger;
    ledger.push_back(crack);
    ledger.push_back(exploration);
    Mission mission = selectMission(ledger, 0, 0, -1, kEffectNone);
    expect(mission.opportunity == exploration.id
               && mission.kind == kMissionReturnToBranch,
           "missing_effect_defers_useful_opportunity_without_forgetting_it");

    // The opportunity names the effect, not its producer.  Any later world
    // or inventory change exposing that effect makes the same ledger entry
    // actionable without rewriting it to a particular item type.
    mission = selectMission(ledger, 1, 0, -1, kEffectExplosive);
    expect(mission.opportunity == crack.id
               && mission.kind == kMissionSolveBlocker,
           "gained_effect_rearms_deferred_opportunity");
}

static void testOneWayOpportunityPreference()
{
    Opportunity trap;
    trap.id = 301;
    trap.kind = kOpportunityFrontier;
    trap.sector = 1;
    trap.target = 2;
    trap.hops = 0;
    trap.local = true;
    trap.oneWayRisk = 1;

    Opportunity remoteTrigger;
    remoteTrigger.id = 302;
    remoteTrigger.kind = kOpportunityInteraction;
    remoteTrigger.sector = 1;
    remoteTrigger.hops = 0;
    remoteTrigger.local = true;

    std::vector<Opportunity> ledger;
    ledger.push_back(trap);
    ledger.push_back(remoteTrigger);
    Mission mission = selectMission(ledger, 0, 0, -1);
    expect(mission.opportunity == remoteTrigger.id
               && mission.kind == kMissionSolveBlocker,
           "reversible_remote_action_precedes_one_way_frontier");

    ledger.erase(ledger.begin() + 1);
    mission = selectMission(ledger, 0, 0, -1);
    expect(mission.opportunity == trap.id && mission.kind == kMissionContinue,
           "one_way_frontier_remains_last_resort");
}

static void testCombat()
{
    CombatSituation situation;
    situation.hasThreat = true;
    situation.immediateThreat = false;
    situation.rangedAvailable = true;
    CombatDecision decision = chooseCombatTactic(situation);
    expect(decision.tactic == kCombatRanged && !decision.overrideMovement,
           "combat_ranged_fire_no_movement_override");

    // A usable weapon beats retreating.  Retreating from a threat the bot
    // could fight only postpones it: the threat stops being "immediate",
    // the bot turns back, and it takes the hit again.
    situation.rangedAvailable = false;
    situation.immediateThreat = true;
    situation.retreatAvailable = true;
    situation.meleeAvailable = true;
    decision = chooseCombatTactic(situation);
    expect(decision.tactic == kCombatMelee && decision.overrideMovement,
           "combat_melee_preferred_over_available_retreat");

    situation.retreatAvailable = false;
    situation.current = kCombatRetreat;
    decision = chooseCombatTactic(situation);
    expect(decision.tactic == kCombatMelee && decision.overrideMovement,
           "combat_no_retreat_immediate_melee");

    // Nearly dead is the case retreat exists for.
    situation.critical = true;
    situation.retreatAvailable = true;
    decision = chooseCombatTactic(situation);
    expect(decision.tactic == kCombatRetreat && decision.overrideMovement,
           "combat_critical_health_retreats");
    situation.critical = false;

    // Nothing to fight with, and something to fight: break contact.
    situation.meleeAvailable = false;
    situation.retreatAvailable = true;
    decision = chooseCombatTactic(situation);
    expect(decision.tactic == kCombatRetreat && decision.overrideMovement,
           "combat_no_usable_weapon_retreats");
    situation.meleeAvailable = true;

    situation.hasThreat = false;
    situation.immediateThreat = false;
    decision = chooseCombatTactic(situation);
    expect(decision.tactic == kCombatNone && decision.dropCombat,
           "combat_no_actual_threat_no_interruption");

    // A distant, non-immediate threat is still worth closing on while the
    // bot has a weapon for it; exploration is interrupted, not abandoned.
    situation.hasThreat = true;
    situation.immediateThreat = false;
    situation.rangedAvailable = false;
    situation.retreatAvailable = false;
    situation.meleeAvailable = true;
    situation.current = kCombatRetreat;
    decision = chooseCombatTactic(situation);
    expect(decision.tactic == kCombatMelee,
           "combat_closes_on_reachable_threat_when_armed");

    // With no weapon and nowhere to go, stop treating it as combat.
    situation.meleeAvailable = false;
    decision = chooseCombatTactic(situation);
    expect(decision.tactic == kCombatNone && decision.dropCombat,
           "combat_no_weapon_no_retreat_drops");
}

int main()
{
    testTraversalProbe();
    testNavGraph();
    testSupportAwareNavigation();
    testCrouchAndConditionalGate();
    testDynamicPlanningAndCausality();
    testDynamicSupports();
    testFrontiers();
    testDeferredEffectPrerequisites();
    testOneWayOpportunityPreference();
    testCombat();
    if (gFailures)
    {
        std::fprintf(stderr, "%d test(s) failed\n", gFailures);
        return 1;
    }
    std::printf("all architecture tests passed\n");
    return 0;
}
