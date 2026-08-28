//-------------------------------------------------------------------------
// Architecture regression tests for the LLMapper movement kernel.
// These tests use captured-style synthetic geometry, never map IDs.
//-------------------------------------------------------------------------
#include "nav_kernel.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

using namespace llmapper;

static int gFailures = 0;

static_assert(!std::is_same<SupportId, ObjectId>::value,
              "support and object identity must not be interchangeable");
static_assert(!std::is_same<PoseId, RegionId>::value,
              "pose and region identity must not be interchangeable");
static_assert(!std::is_convertible<ObjectId, SupportId>::value,
              "engine objects must not leak into semantic support identity");

static void expect(bool condition, const char *name);

static void testSparseConvexSkeleton()
{
    const std::vector<NavWaypoint> rectangle = {
        { 0, 0 }, { 1000, 0 }, { 1000, 400 }, { 0, 400 },
    };
    ConvexSkeleton skeleton = buildConvexSkeleton(rectangle);
    expect(skeleton.cells.size() == 1 && skeleton.gateways.empty()
               && skeleton.cells[0].center.x == 500
               && skeleton.cells[0].center.y == 200,
           "empty_rectangle_has_one_center_pose");

    const std::vector<NavWaypoint> lShape = {
        { 0, 0 }, { 1200, 0 }, { 1200, 400 },
        { 400, 400 }, { 400, 1200 }, { 0, 1200 },
    };
    skeleton = buildConvexSkeleton(lShape);
    expect(skeleton.cells.size() == 2 && skeleton.gateways.size() == 1,
           "empty_l_shape_has_two_centers_and_one_bend_gateway");

    std::vector<NavWaypoint> reversed = lShape;
    std::reverse(reversed.begin(), reversed.end());
    skeleton = buildConvexSkeleton(reversed);
    expect(skeleton.cells.size() == 2 && skeleton.gateways.size() == 1,
           "convex_skeleton_is_winding_independent");

    // Arbitrary source tessellation must not change a homogeneous space.
    // These extra collinear vertices model several engine partitions along
    // the walls of one empty room.
    const std::vector<NavWaypoint> partitionedRectangle = {
        { 0, 0 }, { 250, 0 }, { 700, 0 }, { 1000, 0 },
        { 1000, 200 }, { 1000, 400 }, { 400, 400 }, { 0, 400 },
        { 0, 150 },
    };
    skeleton = buildConvexSkeleton(partitionedRectangle);
    expect(skeleton.cells.size() == 1 && skeleton.gateways.empty(),
           "empty_space_is_invariant_to_boundary_partitioning");

    const std::vector<NavWaypoint> uShape = {
        { 0, 0 }, { 1200, 0 }, { 1200, 1200 }, { 800, 1200 },
        { 800, 400 }, { 400, 400 }, { 400, 1200 }, { 0, 1200 },
    };
    skeleton = buildConvexSkeleton(uShape);
    expect(skeleton.cells.size() == 3 && skeleton.gateways.size() == 2,
           "arbitrary_concave_space_uses_only_centers_and_gateways");

    const std::vector<std::vector<NavWaypoint> > roomWithObstacle = {
        { { -4000, -4000 }, { 4000, -4000 },
          { 4000, 4000 }, { -4000, 4000 } },
        { { -3000, -1000 }, { -3000, 1000 },
          { 2000, 1000 }, { 2000, -1000 } },
    };
    skeleton = buildConvexSkeleton(roomWithObstacle);
    bool centerInsideHole = false;
    for (const SkeletonCell &cell : skeleton.cells)
        centerInsideHole = centerInsideHole
            || (cell.center.x > -3000 && cell.center.x < 2000
                && cell.center.y > -1000 && cell.center.y < 1000);
    expect(skeleton.cells.size() == 4 && skeleton.gateways.size() == 4
               && !centerInsideHole,
           "multiple_contours_form_sparse_connected_free_space");
}

static void testSemanticBoundary()
{
    NavCell pose;
    pose.id = PoseId(4);
    pose.region = RegionId(7);
    pose.support = SupportId(2);

    Affordance affordance;
    affordance.id = AffordanceId(5);
    affordance.target = ObjectId(11);
    affordance.executionDomain.selected = PoseId(4);
    affordance.action = kActionUse;

    expect(affordance.executionDomain.selected == pose.id,
           "semantic_boundary_affordance_pose_identity");
}

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

static NavCell makeCell(int id, int region, int x, int y, int z = 0,
                        SupportId support = SupportId())
{
    NavCell cell;
    cell.id = id;
    cell.region = region;
    cell.center = NavWaypoint(x, y);
    cell.z = z;
    cell.support = support ? support : SupportId(region);
    // A pose a test constructs is a place that exists.  Occupancy is the
    // adapter's business; the graph algorithms only ask whether the pose is
    // still part of the world.
    cell.exists = true;
    return cell;
}

static void addLink(NavCell &from, int to, NavEdgeMode mode, int boundary = -1,
                    NavWaypoint gateway = NavWaypoint())
{
    NavLink link;
    link.target = to;
    link.mode = mode;
    link.boundary = boundary;
    link.gateway = gateway;
    link.hasGateway = boundary >= 0 || (gateway.x != 0 || gateway.y != 0);
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
                              ActionKind action)
{
    for (size_t i = 0; i < plan.size(); ++i)
        if (plan[i].kind == kPlanActivate && plan[i].action == action)
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
    expect(planNavRoute(cells, 0, 2, route)
               && route.size() >= 1,
           "nav_same_sector_concave_route");

    route.clear();
    expect(planNavRoute(cells, 0, 0, route)
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
    expect(planNavRoute(cells, 0, 3, route)
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
    expect(planNavRoute(cells, 0, 3, route)
               && route.size() == 2 && route[0].toCell == 2,
           "nav_prefers_clear_supported_route_over_tight_corner");
    cells[0].links.resize(1);
    expect(planNavRoute(cells, 0, 3, route)
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
    expect(planNavRoute(cells, 0, 2, route)
               && route.size() == 2
               && route[0].mode == kNavStep
               && route[0].boundary == 19
               && route[1].mode == kNavWalk
               && route[0].sourceRegion == 2
               && route[1].targetRegion == 7,
           "nav_multi_sector_walk_step_route");

    cells.clear();
    cells.push_back(makeCell(0, 1, 0, 0));
    cells.push_back(makeCell(1, 1, 0, 64));
    cells.push_back(makeCell(2, 1, 0, 128));
    addLink(cells[0], 1, kNavStep);
    addLink(cells[1], 0, kNavStep);
    addLink(cells[1], 2, kNavStep);
    addLink(cells[2], 1, kNavStep);
    expect(planNavRoute(cells, 0, 2, route)
               && route.size() == 2
               && route[0].mode == kNavStep
               && route[1].mode == kNavStep,
           "nav_step_chain");

    cells.clear();
    cells.push_back(makeCell(0, 1, 0, 0));
    cells.push_back(makeCell(1, 2, 0, -100));
    addLink(cells[0], 1, kNavJump, 8, NavWaypoint(0, -50));
    addLink(cells[1], 0, kNavDrop, 9, NavWaypoint(0, -50));
    expect(planNavRoute(cells, 0, 1, route)
               && route.size() == 1
               && route[0].mode == kNavJump,
           "nav_route_contains_jump");
    expect(planNavRoute(cells, 1, 0, route)
               && route.size() == 1
               && route[0].mode == kNavDrop,
           "nav_directed_jump_drop_asymmetry");

    AttemptLedger blocked;
    AttemptSubject jumped(kAttemptTraverse, 8);
    jumped.fromPose = 0;
    jumped.toPose = 1;
    jumped.mode = kNavJump;
    blocked.record(jumped, 1, 0);
    expect(blocked.tried(jumped, 1)
               && planNavRoute(cells, 0, 1, route)
               && route.size() == 1,
           "nav_execution_failure_does_not_delete_certified_edge");

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
    expect(planNavRoute(cells, 0, 2, route)
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
    expect(planNavRoute(cells, 0, 2, route)
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
    expect(planNavRoute(cells, 0, 5, route)
               && route.size() == 5
               && route[0].mode == kNavWalk
               && route[4].targetRegion == 2,
           "nav_prefers_supported_walk_over_risky_jump_shortcut");

    cells.clear();
    cells.push_back(makeCell(0, 1, 0, 0));
    cells.push_back(makeCell(1, 2, 100, 0));
    cells.push_back(makeCell(2, 2, 100, 100));
    addLink(cells[0], 1, kNavWalk, 1, NavWaypoint(50, 0));
    addLink(cells[0], 2, kNavWalk, 2, NavWaypoint(50, 50));
    addLink(cells[1], 0, kNavWalk, 1, NavWaypoint(50, 0));
    addLink(cells[2], 0, kNavWalk, 2, NavWaypoint(50, 50));
    expect(planNavRoute(cells, 0, 2, route)
               && route.size() == 1
               && route[0].boundary == 2,
           "nav_route_selects_shorter_certified_alternative");

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
    markReachableNavCells(cells, 0, reachable);
    expect(reachable[0] && reachable[1] && !reachable[2],
           "same_sector_disconnected_support_not_actionable");

    addLink(cells[1], 2, kNavJump, 42);
    markReachableNavCells(cells, 0, reachable);
    expect(reachable[2],
           "support_transition_rearms_disconnected_approach");

    AttemptLedger missed;
    AttemptSubject leapt(kAttemptTraverse, 42);
    leapt.fromPose = 1;
    leapt.toPose = 2;
    leapt.mode = kNavJump;
    missed.record(leapt, 1, 0);
    markReachableNavCells(cells, 0, reachable);
    expect(missed.tried(leapt, 1) && reachable[2],
           "execution_failure_does_not_change_physical_reachability");

    // Overlapping layers remain distinct until an engine-certified physical
    // transition is published by the adapter.
    cells.clear();
    cells.push_back(makeCell(0, 90, 0, 0, -12288));
    cells.push_back(makeCell(1, 65, 1024, 0, 28672));
    markReachableNavCells(cells, 1, reachable);
    expect(reachable[1] && !reachable[0],
           "overlapping_layers_do_not_connect_by_xy_alone");
    NavLink certified;
    certified.target = 0;
    certified.mode = kNavJump;
    certified.gateway = cells[1].center;
    certified.hasGateway = true;
    certified.transition = 17;
    certified.airAngle = 777;
    certified.hasAirAngle = true;
    cells[1].links.push_back(certified);
    std::vector<NavRouteStep> translatedRoute;
    expect(planNavRoute(cells, 1, 0, translatedRoute)
               && translatedRoute.size() == 1
               && translatedRoute[0].boundary == BoundaryId()
               && translatedRoute[0].transition == 17
               && translatedRoute[0].hasGateway
               && translatedRoute[0].gateway.x == cells[1].center.x
               && translatedRoute[0].gateway.y == cells[1].center.y
               && translatedRoute[0].hasAirAngle
               && translatedRoute[0].airAngle == 777,
           "engine_certified_route_preserves_execution_evidence");
    markReachableNavCells(cells, 1, reachable);
    expect(reachable[0],
           "engine_certified_transition_is_reachable");

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
    NavCell floor = makeCell(0, 4, 100, 200, 12000, SupportId(40));
    NavCell bridge = makeCell(1, 4, 100, 200, 4000,
                              SupportId(41));
    expect(floor.region == bridge.region
               && floor.center.x == bridge.center.x
               && floor.center.y == bridge.center.y
               && floor.z != bridge.z && floor.support != bridge.support,
           "support_same_sector_xy_different_layer_distinct");

    NavCell bridgeAbove = makeCell(2, 4, 100, 200, -4000,
                                   SupportId(42));
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
               && cells[0].links[0].condition.variable == 9
               && cells[0].links[0].condition.state == 1,
           "dynamic_gate_blocked_to_conditional_crouch");
}

static CausalGraph oneAffordanceGraph(int affordanceId, int location,
                                    ActionKind action, int variable,
                                    int state)
{
    CausalGraph graph;
    Affordance affordance;
    affordance.id = affordanceId;
    affordance.executionDomain.selected = location;
    affordance.action = action;
    graph.affordances.push_back(affordance);
    LearnedEffect effect;
    effect.affordance = affordanceId;
    effect.action = action;
    effect.variable = variable;
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

    CausalGraph graph = oneAffordanceGraph(42, 1, kActionUse, 7, 1);
    std::map<StateVariableId, int> states;
    states[7] = 0;
    std::vector<PlanOperation> plan;
    DynamicPlanStats stats;
    expect(planDynamicRoute(cells, 0, 2, states, graph, plan, &stats)
               && planHas(plan, kPlanActivate)
               && planHas(plan, kPlanWaitForTransition)
               && planActivatesWith(plan, kActionUse),
           "remote_affordance_establishes_traversal_condition");

    // Closing changes availability, not remembered topology.  From the far
    // side another reachable affordance can re-establish the same condition.
    CausalGraph reverseGraph = oneAffordanceGraph(43, 2, kActionUse, 7, 1);
    plan.clear();
    expect(cells[2].links[0].condition.enabled
               && planDynamicRoute(cells, 2, 1, states, reverseGraph, plan)
               && planHas(plan, kPlanActivate),
           "auto_closing_gate_retains_conditional_reverse_connection");

    Affordance remoteEffect;
    remoteEffect.id = 44;
    remoteEffect.executionDomain.selected = 1;
    remoteEffect.executionDomain.kind = kExecutionLineOfEffect;
    remoteEffect.action = kActionDeliverRemoteEffect;
    remoteEffect.destructible = false;
    graph.affordances.push_back(remoteEffect);
    LearnedEffect remoteResult;
    remoteResult.affordance = 44;
    remoteResult.action = kActionDeliverRemoteEffect;
    remoteResult.variable = 8;
    remoteResult.state = 1;
    graph.effects.push_back(remoteResult);
    expect(graph.affordances[0].action == kActionUse
               && graph.affordances[1].action
                    == kActionDeliverRemoteEffect,
           "use_and_effect_delivery_share_causal_model");
    expect(!graph.affordances[1].destructible,
           "effect_delivery_does_not_imply_destructible");

    // Authored routing is translated by the adapter. The semantic graph
    // contains only an affordance-to-state relationship and no channel.
    const std::vector<LearnedEffect> effects = graph.effectsEstablishing(8, 1);
    expect(effects.size() == 1 && effects[0].affordance == 44
               && effects[0].variable == 8,
           "causal_routing_becomes_semantic_state_relationship");

    Affordance siblingA;
    siblingA.id = 61;
    siblingA.executionDomain.selected = 0;
    Affordance siblingB;
    siblingB.id = 62;
    siblingB.executionDomain.selected = 1;
    graph.affordances.push_back(siblingA);
    graph.affordances.push_back(siblingB);
    expect(graph.affordanceById(61) != nullptr
               && graph.affordanceById(62) != nullptr
               && graph.affordanceById(61)->executionDomain.selected == 0
               && graph.affordanceById(62)->executionDomain.selected == 1,
           "shared_effect_affordances_keep_typed_identity");

    expect(!shouldReselectInteractionSurface(true, true, true)
               && shouldReselectInteractionSurface(false, true, true)
               && preserveActiveInteractionSurface(true, true)
               && !preserveActiveInteractionSurface(true, false),
           "active_interaction_surface_retry_is_not_overwritten_by_observation");

    // Twenty unrelated state variables do not multiply route search. Only
    // the condition on the candidate route is resolved.
    for (int i = 100; i < 120; ++i)
        states[i] = i & 1;
    plan.clear();
    stats = DynamicPlanStats();
    expect(planDynamicRoute(cells, 0, 2, states, oneAffordanceGraph(
                                45, 1, kActionUse, 7, 1), plan, &stats)
               && stats.variablesConsidered.size() == 1
               && stats.variablesConsidered.count(7) == 1,
           "dynamic_planner_avoids_unrelated_state_cartesian_product");
}







static void testDeferredEffectPrerequisites()
{
    Opportunity crack;
    crack.kind = kOpportunityInteraction;
    crack.id = WorkId(kWorkStateVariable, 101);
    crack.pose = 1;
    crack.hops = 0;
    crack.local = true;
    crack.ready = true;
    crack.spatial = Opportunity::kSpatialProvenPassable;
    crack.requiredEffects = kEffectExplosive;

    Opportunity exploration;
    exploration.kind = kOpportunityFrontier;
    exploration.id = WorkId(kWorkBoundary, 202, 2, 3);
    exploration.pose = 2;
    exploration.destination = 3;
    exploration.hops = 1;
    exploration.local = false;
    exploration.spatial = Opportunity::kSpatialProvenPassable;

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
    deferred.pose = 2;
    deferred.destination = 3;
    deferred.depth = 4;
    deferred.hops = -1;
    deferred.approach = deferred.pose;
    deferred.spatial = Opportunity::kSpatialUnknown;

    std::vector<Opportunity> ledger(1, deferred);
    WorkSelection selection = selectWork(ledger, 0);
    expect(!selection,
           "temporarily_unreachable_opportunity_is_deferred_not_actionable");
    ledger[0].hops = 3;
    selection = selectWork(ledger, 0);
    expect(!selection,
           "unresolved_topology_is_knowledge_not_an_operation");

    Opportunity deadEnd;
    deadEnd.kind = kOpportunityFrontier;
    deadEnd.id = WorkId(kWorkBoundary, 3002, 8, 9);
    deadEnd.pose = 8;
    deadEnd.destination = 9;
    deadEnd.depth = 9;
    deadEnd.hops = -1;
    deadEnd.approach = deadEnd.pose;
    deadEnd.spatial = Opportunity::kSpatialUnknown;
    Opportunity older = deferred;
    older.id = WorkId(kWorkBoundary, 3003, 2, 3);
    older.depth = 2;
    older.hops = 7;
    older.spatial = Opportunity::kSpatialProvenPassable;
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

    // A reachable source does not manufacture an executable crossing.  The
    // topology fact remains UNKNOWN until the physical layer publishes an
    // engine-backed execution domain.
    ledger[0].hops = 1;
    ledger[0].depth = 10;
    selection = selectWork(ledger, 0);
    expect(selection.work == older.id,
           "unknown_topology_does_not_become_a_fictitious_operation");
    ledger[0].spatial = Opportunity::kSpatialProvenPassable;
    selection = selectWork(ledger, 0);
    expect(selection.work == deadEnd.id,
           "engine_backed_domain_makes_topology_executable");
    ledger[0].spatial = Opportunity::kSpatialCurrentlyBlocked;
    selection = selectWork(ledger, 0);
    expect(selection.work == older.id,
           "concrete_blocker_evidence_defers_only_named_operation");

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
    coverage.pose = 4;
    coverage.destination = 4;
    coverage.hops = -1;
    coverage.spatial = Opportunity::kSpatialUnknown;
    ledger.assign(1, coverage);
    selection = selectWork(ledger, 0);
    expect(!selection,
           "failed_coverage_navigation_does_not_claim_observation");
    ledger[0].hops = 0;
    ledger[0].spatial = Opportunity::kSpatialProvenPassable;
    selection = selectWork(ledger, 0);
    expect(selection.work == coverage.id,
           "failed_coverage_viewpoint_remains_reconsiderable");

    // Once the key prerequisite is satisfied, the same physical interaction
    // can be represented as ordinary interaction work.  It must retain the
    // stronger causal priority over an unrelated local frontier.
    Opportunity keyedInteraction;
    keyedInteraction.kind = kOpportunityInteraction;
    keyedInteraction.id = WorkId(kWorkStateVariable, 42);
    keyedInteraction.pose = 4;
    keyedInteraction.destination = 5;
    keyedInteraction.local = true;
    keyedInteraction.hops = 0;
    keyedInteraction.ready = true;
    keyedInteraction.spatial = Opportunity::kSpatialProvenPassable;
    keyedInteraction.requiredKey = 1;
    Opportunity unrelatedFrontier;
    unrelatedFrontier.kind = kOpportunityFrontier;
    unrelatedFrontier.id = WorkId(kWorkBoundary, 43, 4, 6);
    unrelatedFrontier.pose = 4;
    unrelatedFrontier.destination = 6;
    unrelatedFrontier.local = true;
    unrelatedFrontier.depth = 20;
    unrelatedFrontier.hops = 0;
    unrelatedFrontier.spatial = Opportunity::kSpatialProvenPassable;
    ledger.clear();
    ledger.push_back(unrelatedFrontier);
    ledger.push_back(keyedInteraction);
    selection = selectWork(ledger, 1u << 1);
    expect(selection.work == keyedInteraction.id,
           "satisfied_key_makes_same_deferred_interaction_actionable");

    // A causal continuation is the physical successor operation of the
    // action which just changed the world.  It retains plan ownership until
    // that successor is occupied; an unrelated ready interaction must not
    // discard the sequence at the newly opened crossing.
    Opportunity enabledCoverage = coverage;
    enabledCoverage.id = WorkId(kWorkPose, 7002);
    enabledCoverage.hops = 1;
    enabledCoverage.local = true;
    enabledCoverage.continuation = true;
    enabledCoverage.spatial = Opportunity::kSpatialProvenPassable;
    keyedInteraction.requiredKey = 0;
    ledger.clear();
    ledger.push_back(enabledCoverage);
    ledger.push_back(keyedInteraction);
    selection = selectWork(ledger, 0);
    expect(selection.work == enabledCoverage.id,
           "enabled_space_continuation_retains_action_plan_ownership");

    Opportunity incidentalPickup;
    incidentalPickup.kind = kOpportunityPickup;
    incidentalPickup.id = WorkId(kWorkObject, 7003);
    incidentalPickup.hops = 0;
    incidentalPickup.local = true;
    incidentalPickup.ready = true;
    incidentalPickup.spatial = Opportunity::kSpatialProvenPassable;
    ledger.clear();
    ledger.push_back(incidentalPickup);
    ledger.push_back(enabledCoverage);
    selection = selectWork(ledger, 0);
    expect(selection.work == enabledCoverage.id,
           "enabled_space_continuation_precedes_incidental_pickup");

    Opportunity blockedBoundary;
    blockedBoundary.kind = kOpportunityBlocked;
    blockedBoundary.id = WorkId(kWorkBoundary, 7004, 4, 5);
    blockedBoundary.hops = 0;
    blockedBoundary.local = true;
    blockedBoundary.ready = true;
    blockedBoundary.spatial = Opportunity::kSpatialProvenPassable;
    ledger[0] = blockedBoundary;
    selection = selectWork(ledger, 0);
    expect(selection.work == enabledCoverage.id,
           "enabled_space_continuation_precedes_boundary_investigation");

    enabledCoverage.continuation = false;
    ledger[0] = incidentalPickup;
    ledger[1] = enabledCoverage;
    selection = selectWork(ledger, 0);
    expect(selection.work == incidentalPickup.id,
           "useful_pickup_precedes_unrelated_coverage");
}

static void testOneWayOpportunityPreference()
{
    Opportunity trap;
    trap.kind = kOpportunityFrontier;
    trap.id = WorkId(kWorkBoundary, 301, 1, 2);
    trap.pose = 1;
    trap.destination = 2;
    trap.hops = 0;
    trap.local = true;
    trap.oneWayRisk = 1;
    trap.spatial = Opportunity::kSpatialProvenPassable;

    Opportunity remoteTrigger;
    remoteTrigger.kind = kOpportunityInteraction;
    remoteTrigger.id = WorkId(kWorkStateVariable, 302);
    remoteTrigger.pose = 1;
    remoteTrigger.hops = 0;
    remoteTrigger.local = true;
    remoteTrigger.ready = true;
    remoteTrigger.spatial = Opportunity::kSpatialProvenPassable;

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
    testSparseConvexSkeleton();
    testSemanticBoundary();
    testTraversalProbe();
    testContinuousWalkWaypointProgress();
    testNavGraph();
    testSupportAwareNavigation();
    testCrouchAndConditionalGate();
    testDynamicPlanningAndCausality();
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
