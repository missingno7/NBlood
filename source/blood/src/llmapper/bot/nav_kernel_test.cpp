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
    expect(planNavRoute(cells, 0, 2, std::vector<NavEdgeFailure>(), 1, route)
               && route.size() >= 1,
           "nav_same_sector_concave_route");

    route.clear();
    expect(planNavRoute(cells, 0, 0, std::vector<NavEdgeFailure>(), 1, route)
               && route.empty(),
           "nav_same_pose_route_is_empty");

    // Two routes with the same number and kind of links are not physically
    // equivalent. A sector graph (and the old unit-cost cell search) chose
    // the lower-id detour; concrete pose distance must choose the straight
    // crossing independently of map/cell insertion order.
    cells.clear();
    cells.push_back(makeCell(0, 1, 0, 0));
    cells.push_back(makeCell(1, 1, -1000, 100));
    cells.push_back(makeCell(2, 1, 100, 0));
    cells.push_back(makeCell(3, 2, 200, 0));
    addLink(cells[0], 1, kNavWalk, 4);
    addLink(cells[1], 3, kNavWalk, 5);
    addLink(cells[0], 2, kNavWalk, 6);
    addLink(cells[2], 3, kNavWalk, 7);
    expect(planNavRoute(cells, 0, 3, std::vector<NavEdgeFailure>(), 1, route)
               && route.size() == 2
               && route[0].toCell == 2
               && route[1].toCell == 3,
           "nav_equal_hops_choose_shorter_physical_crossing");

    // Clearance is execution quality, not reachability. A slightly longer
    // supported route should beat a corner-hugging route when both exist;
    // the narrow route remains usable if it is the only connection.
    cells.clear();
    cells.push_back(makeCell(0, 1, 0, 0));
    cells.push_back(makeCell(1, 1, 100, 0));
    cells.push_back(makeCell(2, 1, 0, 400));
    cells.push_back(makeCell(3, 1, 200, 0));
    cells[1].clearance = 100;
    cells[2].clearance = 512;
    addLink(cells[0], 1, kNavWalk, 8);
    addLink(cells[1], 3, kNavWalk, 9);
    addLink(cells[0], 2, kNavWalk, 10);
    addLink(cells[2], 3, kNavWalk, 11);
    expect(planNavRoute(cells, 0, 3, std::vector<NavEdgeFailure>(), 1, route)
               && route.size() == 2 && route[0].toCell == 2,
           "nav_prefers_clear_supported_route_over_tight_corner");
    cells[0].links.resize(1);
    expect(planNavRoute(cells, 0, 3, std::vector<NavEdgeFailure>(), 1, route)
               && route.size() == 2 && route[0].toCell == 1,
           "nav_narrow_route_remains_reachable_when_only_option");

    cells.clear();
    cells.push_back(makeCell(0, 2, 0, 0));
    cells.push_back(makeCell(1, 1, 200, 0));
    cells.push_back(makeCell(2, 7, 400, 0));
    addLink(cells[0], 1, kNavStep, 19, NavWaypoint(100, 0));
    addLink(cells[1], 0, kNavStep, 20, NavWaypoint(100, 0));
    addLink(cells[1], 2, kNavWalk, 4, NavWaypoint(300, 0));
    addLink(cells[2], 1, kNavWalk, 5, NavWaypoint(300, 0));
    expect(planNavRoute(cells, 0, 2, std::vector<NavEdgeFailure>(), 1, route)
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
    expect(planNavRoute(cells, 0, 2, std::vector<NavEdgeFailure>(), 1, route)
               && route.size() == 2
               && route[0].mode == kNavStep
               && route[1].mode == kNavStep,
           "nav_step_chain");

    cells.clear();
    cells.push_back(makeCell(0, 1, 0, 0));
    cells.push_back(makeCell(1, 2, 0, -100));
    addLink(cells[0], 1, kNavJump, 8, NavWaypoint(0, -50));
    addLink(cells[1], 0, kNavDrop, 9, NavWaypoint(0, -50));
    expect(planNavRoute(cells, 0, 1, std::vector<NavEdgeFailure>(), 1, route)
               && route.size() == 1
               && route[0].mode == kNavJump,
           "nav_route_contains_jump");
    expect(planNavRoute(cells, 1, 0, std::vector<NavEdgeFailure>(), 1, route)
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
    expect(!planNavRoute(cells, 0, 1, failures, 1, route),
           "nav_failed_edge_excluded");
    expect(planNavRoute(cells, 0, 1, failures, 2, route),
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
    expect(planNavRoute(cells, 0, 2, std::vector<NavEdgeFailure>(), 1, route)
               && route.size() == 2
               && route[0].toCell == 1
               && route[1].toCell == 2,
           "nav_prefers_staged_collision_support_climb");

    cells.clear();
    cells.push_back(makeCell(0, 1, 0, 0, 0));
    cells.push_back(makeCell(1, 1, 900, 0, 0));
    cells.push_back(makeCell(2, 1, 1000, 0, -4096));
    // Walking to a nearby takeoff and then jumping covers the same total XY
    // distance and uses the same capability as a direct leap. The shorter
    // flight is physically easier to land and brake, especially when the
    // destination is a narrow sprite collision top.
    addLink(cells[0], 2, kNavJump);
    addLink(cells[0], 1, kNavWalk);
    addLink(cells[1], 2, kNavJump);
    expect(planNavRoute(cells, 0, 2, std::vector<NavEdgeFailure>(), 1, route)
               && route.size() == 2
               && route[0].toCell == 1
               && route[1].toCell == 2,
           "nav_prefers_near_takeoff_over_equal_distance_long_jump");

    cells.clear();
    for (int i = 0; i < 6; ++i)
        cells.push_back(makeCell(i, i == 5 ? 2 : 1, i * 100, 0));
    // One risky shortcut and a five-cell supported walk reach the same goal.
    // Route selection should preserve the jump for maps that need it, but
    // prefer the bridge when both are currently available.
    addLink(cells[0], 5, kNavJump, 30, NavWaypoint(250, -100));
    for (int i = 0; i < 5; ++i)
        addLink(cells[i], i + 1, kNavWalk);
    expect(planNavRoute(cells, 0, 5, std::vector<NavEdgeFailure>(), 1, route)
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
    expect(planNavRoute(cells, 0, 2, failures, 9, route)
               && route.size() == 1
               && route[0].wall == 2,
           "nav_failed_edge_reroutes_to_alternative");

    // A sector id is not a support component.  Stacked/ROR geometry can
    // report two cells in the same Build sector while no physical route
    // connects them.
    cells.clear();
    cells.push_back(makeCell(0, 9, 0, 0, 0));
    cells.push_back(makeCell(1, 9, 100, 0, 0));
    cells.push_back(makeCell(2, 9, 100, 100, -4096));
    addLink(cells[0], 1, kNavWalk, 40);
    addLink(cells[1], 0, kNavWalk, 41);
    std::vector<char> reachable;
    markReachableNavCells(cells, 0, std::vector<NavEdgeFailure>(), 1,
                          reachable);
    expect(reachable[0] && reachable[1] && !reachable[2],
           "same_sector_disconnected_support_not_actionable");

    addLink(cells[1], 2, kNavJump, 42);
    markReachableNavCells(cells, 0, std::vector<NavEdgeFailure>(), 1,
                          reachable);
    expect(reachable[2],
           "support_transition_rearms_disconnected_approach");

    failure = NavEdgeFailure();
    failure.fromCell = 1;
    failure.toCell = 2;
    failure.wall = 42;
    failure.mode = kNavJump;
    failure.geometrySignature = 1;
    failures.assign(1, failure);
    markReachableNavCells(cells, 0, failures, 1, reachable);
    expect(!reachable[2],
           "failed_support_transition_does_not_mark_far_pose_reachable");

    // Overlapping sector-local surfaces remain distinct until the engine's
    // stacked-room markers provide an explicit translated transition.
    cells.clear();
    cells.push_back(makeCell(0, 90, 0, 0, -12288));
    cells.push_back(makeCell(1, 65, 1024, 0, 28672));
    markReachableNavCells(cells, 1, std::vector<NavEdgeFailure>(), 1,
                          reachable);
    expect(reachable[1] && !reachable[0],
           "overlapping_layers_do_not_connect_by_xy_alone");
    expect(linkTranslatedNavLayers(cells, 90, 65, 1024, 0, 512, -2) == 2
               && cells[1].links.size() == 1
               && cells[1].links[0].target == 0
               && cells[1].links[0].mode == kNavWalk
               && cells[1].links[0].transition == 90,
           "explicit_ror_transition_connects_sector_layers");
    markReachableNavCells(cells, 1, std::vector<NavEdgeFailure>(), 1,
                          reachable);
    expect(reachable[0],
           "ror_transition_is_reachable_like_open_space");

    expect(selectTargetNavCell(17, 3400LL * 3400, 29,
                               12000LL * 12000, 2048LL * 2048) == 17,
           "reachable_xy_pose_beats_remote_overlapping_z_projection");
    expect(selectTargetNavCell(17, 6000LL * 6000, 29,
                               256LL * 256, 2048LL * 2048) == 29,
           "near_exact_support_endpoint_beats_remote_same_area_pose");
    expect(selectTargetNavCell(17, 512LL * 512, 29,
                               64LL * 64, 2048LL * 2048) == 17,
           "local_stance_tolerance_keeps_reachable_component");
}

static void testContinuousWalkWaypointProgress()
{
    const NavWaypoint source(0, 0);
    const NavWaypoint waypoint(256, 0);
    expect(crossedWaypointCorridor(source, waypoint,
                                   NavWaypoint(300, 100), 192),
           "walk_waypoint_crossed_inside_corridor");
    expect(!crossedWaypointCorridor(source, waypoint,
                                    NavWaypoint(200, 0), 192),
           "walk_waypoint_not_crossed_before_progress_plane");
    expect(!crossedWaypointCorridor(source, waypoint,
                                    NavWaypoint(300, 300), 192),
           "walk_waypoint_not_crossed_outside_corridor");
    expect(crossedWaypointCorridor(NavWaypoint(0, 0),
                                   NavWaypoint(512, -384),
                                   NavWaypoint(738, -336), 192),
           "diagonal_walk_waypoint_accepts_safe_running_pass");
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

    Actuator siblingA;
    siblingA.id = 61;
    siblingA.tx = 100;
    siblingA.locationCell = 0;
    Actuator siblingB;
    siblingB.id = 62;
    siblingB.tx = 100;
    siblingB.locationCell = 1;
    graph.actuators.push_back(siblingA);
    graph.actuators.push_back(siblingB);
    expect(graph.actuatorById(61) != nullptr
               && graph.actuatorById(62) != nullptr
               && graph.actuatorById(61)->locationCell == 0
               && graph.actuatorById(62)->locationCell == 1,
           "shared_receiver_actuators_keep_distinct_attempt_identity");

    const int receiver = 50;
    const int firstDoor = wallInteractionAttemptKey(
        418, 55, 50, true, true, false, receiver);
    const int siblingFace = wallInteractionAttemptKey(
        419, 55, 50, false, true, false, receiver);
    const int secondDoor = wallInteractionAttemptKey(
        412, 55, 51, true, true, false, receiver);
    expect(firstDoor == siblingFace && firstDoor != secondDoor,
           "physical_wall_actuators_do_not_alias_shared_causal_receiver");
    const int previewSector = wallInteractionActuatorSector(-1, receiver);
    const int storedSector = wallInteractionActuatorSector(receiver, receiver);
    expect(wallInteractionAttemptKey(419, 65, previewSector,
                                     true, false, false, receiver)
               == wallInteractionAttemptKey(419, 65, storedSector,
                                            true, false, false, receiver),
           "one_sided_wall_identity_stable_after_receiver_resolution");
    expect(!shouldReselectInteractionSurface(true, true, true)
               && shouldReselectInteractionSurface(false, true, true)
               && preserveActiveInteractionSurface(true, true)
               && !preserveActiveInteractionSurface(true, false),
           "active_interaction_surface_retry_is_not_overwritten_by_observation");

    CausalGraph chain;
    CausalReceiver intermediate;
    intermediate.channel = 105;
    intermediate.object = WorldObjectRef(kWorldSprite, 542);
    intermediate.outgoingChannel = 106;
    intermediate.command = 5;
    chain.receivers.push_back(intermediate);
    CausalReceiver finalDoor;
    finalDoor.channel = 106;
    finalDoor.object = WorldObjectRef(kWorldSector, 51);
    chain.receivers.push_back(finalDoor);
    const std::vector<CausalReceiver> allChain =
        chain.receiversReachableFrom(105, 8, false);
    const std::vector<CausalReceiver> finalChain =
        chain.receiversReachableFrom(105, 8, true);
    expect(allChain.size() == 2 && finalChain.size() == 1
               && finalChain[0].object == WorldObjectRef(kWorldSector, 51),
           "causal_chain_follows_intermediate_rx_to_final_world_effect");

    CausalReceiver cycleBack;
    cycleBack.channel = 106;
    cycleBack.object = WorldObjectRef(kWorldSprite, 543);
    cycleBack.outgoingChannel = 105;
    chain.receivers.clear();
    chain.receivers.push_back(intermediate);
    chain.receivers.push_back(cycleBack);
    expect(chain.receiversReachableFrom(105, 8, true).empty(),
           "causal_chain_cycle_is_bounded_without_fake_terminal_effect");

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

    Boundary probeA;
    probeA.wall = 50;
    probeA.from = 1;
    probeA.to = 10;
    probeA.geometrySignature = 5;
    Boundary probeB = probeA;
    probeB.wall = 51;
    std::vector<Boundary> parallel;
    parallel.push_back(probeA);
    parallel.push_back(probeB);
    InvestigateRecord inspectedA;
    inspectedA.wall = 50;
    inspectedA.from = 1;
    inspectedA.to = 10;
    inspectedA.geometrySignature = 5;
    visited.assign(1, 1);
    frontiers = deriveFrontiers(visited, parallel,
                                std::vector<InvestigateRecord>(1, inspectedA),
                                std::vector<NavEdgeFailure>());
    expect(frontiers.size() == 1 && frontiers[0].destination == 10
               && frontiers[0].candidates.size() == 1
               && frontiers[0].candidates[0].wall == 51,
           "one_failed_boundary_inspection_does_not_exhaust_destination");
}

static VisibilityCell visibilityCell(int id, int x, int y, bool observed,
                                     int partition, int area = 0)
{
    VisibilityCell cell;
    cell.id = id;
    cell.x = x;
    cell.y = y;
    cell.area = area;
    cell.partition = partition;
    cell.observed = observed;
    cell.reachable = true;
    return cell;
}

static void connectVisibility(std::vector<VisibilityCell> &cells, int a, int b)
{
    cells[size_t(a)].neighbors.push_back(cells[size_t(b)].id);
    cells[size_t(b)].neighbors.push_back(cells[size_t(a)].id);
}

static void testSemanticVisibilityExploration()
{
    std::vector<VisibilityCell> oneSector;
    std::vector<VisibilityCell> microsectors;
    for (int i = 0; i < 8; ++i)
    {
        oneSector.push_back(visibilityCell(i, i * 256, 0, true, 0));
        microsectors.push_back(visibilityCell(i, i * 256, 0, true, i));
        if (i > 0)
        {
            connectVisibility(oneSector, i - 1, i);
            connectVisibility(microsectors, i - 1, i);
        }
    }
    const std::vector<VisibilityFrontier> simple =
        deriveVisibilityFrontiers(oneSector, 512, 2048);
    const std::vector<VisibilityFrontier> fragmented =
        deriveVisibilityFrontiers(microsectors, 512, 2048);
    expect(simple.empty() && fragmented.empty(),
           "microsector_invariance_visible_empty_corridor_has_no_work");

    // The first two samples are visible, while the corridor continues around
    // an occluding bend.  Only the visibility boundary is semantic work.
    std::vector<VisibilityCell> bend;
    bend.push_back(visibilityCell(0, 0, 0, true, 0));
    bend.push_back(visibilityCell(1, 256, 0, true, 1));
    bend.push_back(visibilityCell(2, 256, 256, false, 2));
    bend.push_back(visibilityCell(3, 256, 512, false, 3));
    connectVisibility(bend, 0, 1);
    connectVisibility(bend, 1, 2);
    connectVisibility(bend, 2, 3);
    std::vector<VisibilityFrontier> frontier =
        deriveVisibilityFrontiers(bend, 512, 2048);
    expect(frontier.size() == 1 && frontier[0].cell == 2
               && frontier[0].approachCell == 1
               && frontier[0].informationGain == 2,
           "occluded_bend_generates_positive_information_gain_viewpoint");
    bend[2].observed = true;
    bend[3].observed = true;
    expect(deriveVisibilityFrontiers(bend, 512, 2048).empty(),
           "empty_hidden_space_disappears_when_visible_without_sector_entry");

    // An overlapping support layer remains distinct because physical reach
    // and graph adjacency, not 2D sector ownership, define the frontier.
    std::vector<VisibilityCell> layers;
    layers.push_back(visibilityCell(10, 0, 0, true, 7, 0));
    layers.push_back(visibilityCell(11, 0, 0, false, 7, 1));
    layers[0].neighbors.push_back(11);
    layers[1].neighbors.push_back(10);
    frontier = deriveVisibilityFrontiers(layers, 512, 2048);
    expect(frontier.size() == 1 && frontier[0].cell == 11
               && frontier[0].approachCell == 10,
           "occluded_reachable_support_layer_remains_semantic_unknown");

    layers[1].reachable = false;
    frontier = deriveVisibilityFrontiers(layers, 512, 2048);
    expect(frontier.size() == 1 && !frontier[0].reachable
               && frontier[0].approachCell == 10,
           "temporarily_unreachable_space_retains_reachable_boundary_pose");

    Opportunity distantPickup;
    distantPickup.kind = kOpportunityPickup;
    distantPickup.id = WorkId(kWorkObject, 42);
    distantPickup.sector = 8;
    distantPickup.hops = 3;
    distantPickup.local = false;
    distantPickup.ready = true;
    Opportunity unknownView;
    unknownView.kind = kOpportunityCoverage;
    unknownView.id = WorkId(kWorkPose, 11);
    unknownView.sector = 1;
    unknownView.hops = 0;
    unknownView.local = true;
    std::vector<Opportunity> semanticWork;
    semanticWork.push_back(unknownView);
    semanticWork.push_back(distantPickup);
    WorkSelection selection = selectWork(semanticWork, 0);
    expect(selection.work == distantPickup.id,
           "ready_task_precedes_exploration_for_more_work");

    semanticWork[1].ready = false;
    selection = selectWork(semanticWork, 0);
    expect(selection.work == unknownView.id,
           "exploration_precedes_task_blocked_on_approach_pose");

    semanticWork.erase(semanticWork.begin());
    selection = selectWork(semanticWork, 0);
    expect(selection.work == distantPickup.id,
           "deferred_task_revisited_after_discovery_exhausted");

    Opportunity visibleExit;
    visibleExit.kind = kOpportunityExit;
    visibleExit.id = WorkId(kWorkExit, 8);
    visibleExit.sector = 8;
    visibleExit.hops = 4;
    visibleExit.ready = false;
    semanticWork.push_back(visibleExit);
    selection = selectWork(semanticWork, 0);
    expect(selection.work == distantPickup.id,
           "nearest_discovered_task_selected_before_farther_task");

    semanticWork.clear();
    semanticWork.push_back(visibleExit);
    selection = selectWork(semanticWork, 0);
    expect(selection.work == visibleExit.id,
           "visible_exit_remains_ordinary_actionable_work");

    Opportunity enabledView = unknownView;
    enabledView.id = WorkId(kWorkPose, 12);
    enabledView.hops = 9;
    enabledView.local = false;
    enabledView.continuation = true;
    semanticWork.push_back(enabledView);
    selection = selectWork(semanticWork, 0);
    expect(selection.work == enabledView.id,
           "world_change_successor_is_consumed_before_unrelated_work");

    semanticWork.push_back(distantPickup);
    selection = selectWork(semanticWork, 0);
    expect(selection.work == distantPickup.id,
           "ready_task_precedes_causal_coverage_continuation");
}

static void testDeferredEffectPrerequisites()
{
    Opportunity crack;
    crack.kind = kOpportunityInteraction;
    crack.id = WorkId(kWorkMechanism, 101);
    crack.sector = 1;
    crack.hops = 0;
    crack.local = true;
    crack.ready = true;
    crack.requiredEffects = kEffectExplosive;

    Opportunity exploration;
    exploration.kind = kOpportunityFrontier;
    exploration.id = WorkId(kWorkBoundary, 202, 2, 3);
    exploration.sector = 2;
    exploration.target = 3;
    exploration.hops = 1;
    exploration.local = false;

    std::vector<Opportunity> ledger;
    ledger.push_back(crack);
    ledger.push_back(exploration);
    WorkSelection selection = selectWork(ledger, 0, kEffectNone);
    expect(selection.work == exploration.id,
           "missing_effect_defers_useful_opportunity_without_forgetting_it");

    // The opportunity names the effect, not its producer.  Any later world
    // or inventory change exposing that effect makes the same ledger entry
    // actionable without rewriting it to a particular item type.
    selection = selectWork(ledger, 0, kEffectExplosive);
    expect(selection.work == crack.id,
           "gained_effect_rearms_deferred_opportunity");

    // Accepted effects are alternatives, not a checklist. A collision body
    // vulnerable to either bullets or explosions is actionable when the
    // player can produce either one.
    crack.requiredEffects = kEffectExplosive | kEffectBulletDamage;
    ledger[0] = crack;
    selection = selectWork(ledger, 0, kEffectBulletDamage);
    expect(selection.work == crack.id,
           "one_of_multiple_accepted_damage_effects_is_sufficient");
}

static void testConservedExplorationLedger()
{
    Opportunity deferred;
    deferred.kind = kOpportunityFrontier;
    deferred.id = WorkId(kWorkBoundary, 3001, 2, 3);
    deferred.sector = 2;
    deferred.target = 3;
    deferred.depth = 4;
    deferred.hops = -1;

    std::vector<Opportunity> ledger(1, deferred);
    WorkSelection selection = selectWork(ledger, 0);
    expect(!selection,
           "temporarily_unreachable_opportunity_is_deferred_not_actionable");
    ledger[0].hops = 3;
    selection = selectWork(ledger, 0);
    expect(selection.work == deferred.id,
           "unresolved_opportunity_survives_temporary_unreachability");

    Opportunity deadEnd;
    deadEnd.kind = kOpportunityFrontier;
    deadEnd.id = WorkId(kWorkBoundary, 3002, 8, 9);
    deadEnd.sector = 8;
    deadEnd.target = 9;
    deadEnd.depth = 9;
    deadEnd.hops = -1;
    Opportunity older = deferred;
    older.id = WorkId(kWorkBoundary, 3003, 2, 3);
    older.depth = 2;
    older.hops = 7;
    ledger.clear();
    ledger.push_back(deadEnd);
    ledger.push_back(older);
    selection = selectWork(ledger, 0);
    expect(selection.work == older.id,
           "dead_end_backtracks_to_older_unresolved_branch");

    // Elapsed time by itself is not failure.  A long valid route through
    // visited space remains selected while its opportunity is actionable.
    selection = selectWork(ledger, 0);
    expect(selection.work == older.id,
           "long_successful_backtracking_is_not_a_stall");

    // The caller rejects one stale/uncommittable candidate by annotating only
    // that ledger entry unreachable for the current selection pass.
    ledger[0].hops = 1;
    ledger[0].depth = 10;
    selection = selectWork(ledger, 0);
    expect(selection.work == deadEnd.id,
           "highest_ranked_candidate_selected_before_commit_validation");
    ledger[0].hops = -1;
    selection = selectWork(ledger, 0);
    expect(selection.work == older.id,
           "next_mission_selected_after_uncommittable_candidate");

    // Progress epochs are independent: a later loop cannot erase conserved
    // work merely because an earlier loop happened before real progress.
    older.hops = 2;
    ledger.assign(1, older);
    selection = selectWork(ledger, 0);
    expect(selection.work == older.id,
           "independent_loops_separated_by_progress_do_not_terminate_work");

    Opportunity coverage;
    coverage.kind = kOpportunityCoverage;
    coverage.id = WorkId(kWorkPose, 7001);
    coverage.sector = 4;
    coverage.target = 4;
    coverage.hops = -1;
    ledger.assign(1, coverage);
    selection = selectWork(ledger, 0);
    expect(!selection,
           "failed_coverage_navigation_does_not_claim_observation");
    ledger[0].hops = 0;
    selection = selectWork(ledger, 0);
    expect(selection.work == coverage.id,
           "failed_coverage_viewpoint_remains_reconsiderable");

    // Once the key prerequisite is satisfied, the same physical interaction
    // can be represented as ordinary interaction work.  It must retain the
    // stronger causal priority over an unrelated local frontier.
    Opportunity keyedInteraction;
    keyedInteraction.kind = kOpportunityInteraction;
    keyedInteraction.id = WorkId(kWorkMechanism, 42);
    keyedInteraction.sector = 4;
    keyedInteraction.target = 5;
    keyedInteraction.local = true;
    keyedInteraction.hops = 0;
    keyedInteraction.ready = true;
    keyedInteraction.requiredKey = 1;
    Opportunity unrelatedFrontier;
    unrelatedFrontier.kind = kOpportunityFrontier;
    unrelatedFrontier.id = WorkId(kWorkBoundary, 43, 4, 6);
    unrelatedFrontier.sector = 4;
    unrelatedFrontier.target = 6;
    unrelatedFrontier.local = true;
    unrelatedFrontier.depth = 20;
    unrelatedFrontier.hops = 0;
    ledger.clear();
    ledger.push_back(unrelatedFrontier);
    ledger.push_back(keyedInteraction);
    selection = selectWork(ledger, 1u << 1);
    expect(selection.work == keyedInteraction.id,
           "satisfied_key_makes_same_deferred_interaction_actionable");
}

static void testOneWayOpportunityPreference()
{
    Opportunity trap;
    trap.kind = kOpportunityFrontier;
    trap.id = WorkId(kWorkBoundary, 301, 1, 2);
    trap.sector = 1;
    trap.target = 2;
    trap.hops = 0;
    trap.local = true;
    trap.oneWayRisk = 1;

    Opportunity remoteTrigger;
    remoteTrigger.kind = kOpportunityInteraction;
    remoteTrigger.id = WorkId(kWorkMechanism, 302);
    remoteTrigger.sector = 1;
    remoteTrigger.hops = 0;
    remoteTrigger.local = true;
    remoteTrigger.ready = true;

    std::vector<Opportunity> ledger;
    ledger.push_back(trap);
    ledger.push_back(remoteTrigger);
    WorkSelection selection = selectWork(ledger, 0);
    expect(selection.work == remoteTrigger.id,
           "reversible_remote_action_precedes_one_way_frontier");

    ledger.erase(ledger.begin() + 1);
    selection = selectWork(ledger, 0);
    expect(selection.work == trap.id,
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
    testContinuousWalkWaypointProgress();
    testNavGraph();
    testSupportAwareNavigation();
    testCrouchAndConditionalGate();
    testDynamicPlanningAndCausality();
    testDynamicSupports();
    testFrontiers();
    testSemanticVisibilityExploration();
    testDeferredEffectPrerequisites();
    testConservedExplorationLedger();
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
