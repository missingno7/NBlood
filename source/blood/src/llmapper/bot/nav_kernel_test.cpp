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

static NavCell makeCell(int id, int sector, int x, int y)
{
    NavCell cell;
    cell.id = id;
    cell.sector = sector;
    cell.center = NavWaypoint(x, y);
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
    testFrontiers();
    testCombat();
    if (gFailures)
    {
        std::fprintf(stderr, "%d test(s) failed\n", gFailures);
        return 1;
    }
    std::printf("all architecture tests passed\n");
    return 0;
}
