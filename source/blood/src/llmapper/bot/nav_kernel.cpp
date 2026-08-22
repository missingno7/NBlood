//-------------------------------------------------------------------------
// Engine-free movement / exploration kernel for the LLMapper bot.
//-------------------------------------------------------------------------
#include "nav_kernel.h"

#include <cmath>
#include <deque>
#include <map>
#include <set>

namespace llmapper
{

const char *navEdgeModeName(NavEdgeMode mode)
{
    switch (mode)
    {
    case kNavStep: return "STEP";
    case kNavJump: return "JUMP";
    case kNavCrouch: return "CROUCH";
    case kNavDrop: return "DROP_SAFE";
    case kNavRide: return "REMAIN_SUPPORTED";
    case kNavInteraction: return "INTERACTION";
    case kNavBlocked: return "BLOCKED";
    default: return "WALK";
    }
}

const char *traversalResultName(TraversalResult result)
{
    switch (result)
    {
    case kTraverseStep: return "STEP";
    case kTraverseJump: return "JUMP";
    case kTraverseCrouch: return "CROUCH";
    case kTraverseDropSafe: return "DROP_SAFE";
    case kTraverseUseableBlocker: return "USEABLE_BLOCKER";
    case kTraverseSolidBlocker: return "SOLID_BLOCKER";
    case kTraverseDangerous: return "DANGEROUS";
    case kTraverseNoFit: return "NO_FIT";
    default: return "DIRECT";
    }
}

const char *combatTacticName(CombatTactic tactic)
{
    switch (tactic)
    {
    case kCombatRanged: return "RANGED_ATTACK";
    case kCombatRetreat: return "RETREAT";
    case kCombatMelee: return "MELEE_ENGAGE";
    default: return "NONE";
    }
}

void assignWalkAreas(std::vector<NavCell> &cells)
{
    for (size_t i = 0; i < cells.size(); ++i)
        cells[i].walkArea = -1;
    int nextArea = 0;
    for (size_t i = 0; i < cells.size(); ++i)
    {
        if (cells[i].walkArea >= 0)
            continue;
        std::deque<int> queue;
        queue.push_back(int(i));
        cells[i].walkArea = nextArea;
        while (!queue.empty())
        {
            const int current = queue.front();
            queue.pop_front();
            const std::vector<NavLink> &links = cells[size_t(current)].links;
            for (size_t l = 0; l < links.size(); ++l)
            {
                if (!walkMode(links[l].mode) || links[l].condition.enabled)
                    continue;
                if (links[l].target < 0 || links[l].target >= int(cells.size()))
                    continue;
                if (cells[size_t(links[l].target)].walkArea < 0)
                {
                    cells[size_t(links[l].target)].walkArea = nextArea;
                    queue.push_back(links[l].target);
                }
            }
        }
        ++nextArea;
    }
}

void markReachableNavCells(const std::vector<NavCell> &cells, int startCell,
                           const std::vector<NavEdgeFailure> &failures,
                           int geometrySignature,
                           std::vector<char> &reachable)
{
    reachable.assign(cells.size(), 0);
    if (startCell < 0 || startCell >= int(cells.size()))
        return;
    std::deque<int> queue;
    reachable[size_t(startCell)] = 1;
    queue.push_back(startCell);
    while (!queue.empty())
    {
        const int current = queue.front();
        queue.pop_front();
        const NavCell &cell = cells[size_t(current)];
        for (const NavLink &link : cell.links)
        {
            // Conditional links describe a route after some prerequisite has
            // changed.  They are conserved by the causal planner, but are not
            // physically reachable in the world state being ranked now.
            if (!traversableMode(link.mode) || link.condition.enabled
                || link.target < 0 || link.target >= int(cells.size())
                || reachable[size_t(link.target)]
                || edgeFailedAny(failures, current, link.target, link.wall,
                                 link.mode, geometrySignature))
                continue;
            reachable[size_t(link.target)] = 1;
            queue.push_back(link.target);
        }
    }
}

int linkTranslatedNavLayers(std::vector<NavCell> &cells, int upperSector,
                            int lowerSector, int deltaX, int deltaY,
                            int maximumError, int transitionEdge)
{
    const int64_t maximumError2 = int64_t(maximumError) * maximumError;
    int linked = 0;
    auto addLink = [&](int from, int to, NavEdgeMode mode,
                       int transitionSector, const NavWaypoint &gateway) {
        if (from < 0 || to < 0 || from >= int(cells.size())
            || to >= int(cells.size()) || from == to)
            return false;
        for (const NavLink &link : cells[size_t(from)].links)
            if (link.target == to && link.wall == transitionEdge)
                return false;
        NavLink link;
        link.target = to;
        link.mode = mode;
        link.wall = transitionEdge;
        link.gateway = gateway;
        link.hasGateway = true;
        link.transition = transitionSector;
        cells[size_t(from)].links.push_back(link);
        return true;
    };

    for (const NavCell &upper : cells)
    {
        if (upper.sector != upperSector
            || upper.support != SupportRef(kSupportSectorFloor, upperSector))
            continue;
        const int wantedX = upper.center.x + deltaX;
        const int wantedY = upper.center.y + deltaY;
        int lowerId = -1;
        int64_t bestDistance = maximumError2 + 1;
        for (const NavCell &lower : cells)
        {
            if (lower.sector != lowerSector
                || lower.support != SupportRef(kSupportSectorFloor, lowerSector))
                continue;
            const int64_t candidate = int64_t(lower.center.x - wantedX)
                    * (lower.center.x - wantedX)
                + int64_t(lower.center.y - wantedY)
                    * (lower.center.y - wantedY);
            if (candidate < bestDistance)
            {
                bestDistance = candidate;
                lowerId = lower.id;
            }
        }
        if (lowerId < 0 || lowerId >= int(cells.size()))
            continue;
        const NavCell &lower = cells[size_t(lowerId)];
        // A room-over-room link is an engine coordinate-space portal.  The
        // player walks through its source pose and the engine translates the
        // body to the receiving layer; no ballistic capability is involved.
        // Labelling this DROP/JUMP handed a remote translated coordinate to
        // the jump executor and invented a flight across ordinary geometry.
        if (addLink(upper.id, lower.id, kNavWalk, lowerSector,
                    upper.center))
            ++linked;
        if (addLink(lower.id, upper.id, kNavWalk, upperSector,
                    lower.center))
            ++linked;
    }
    return linked;
}

static const NavLink *findLink(const NavCell &cell, int target, int wall)
{
    for (size_t i = 0; i < cell.links.size(); ++i)
    {
        const NavLink &link = cell.links[i];
        if (link.target != target)
            continue;
        if (wall >= 0 && link.wall >= 0 && link.wall != wall)
            continue;
        return &link;
    }
    return nullptr;
}

static bool conditionSatisfied(const NavCondition &condition,
                               const std::map<int, int> &states)
{
    if (!condition.enabled)
        return true;
    std::map<int, int>::const_iterator found = states.find(condition.mechanism);
    return found != states.end() && found->second == condition.state;
}

bool planNavRoute(const std::vector<NavCell> &cells, int startCell,
                  int targetCell,
                  const std::vector<NavEdgeFailure> &failures,
                  int geometrySignature,
                  std::vector<NavRouteStep> &outRoute)
{
    outRoute.clear();
    if (startCell < 0 || startCell >= int(cells.size()))
        return false;

    // Routing begins and ends at concrete physical poses.  A Build-sector
    // label is useful metadata for collision calls and telemetry, but it is
    // too lossy to manufacture a goal: several disconnected floors/supports
    // can share it, and several sector labels can describe one continuous
    // floor.  The caller must resolve its task to a NavCell first.
    const int goal = targetCell;
    if (goal < 0 || goal >= int(cells.size()))
        return false;
    if (goal == startCell)
        return true;

    // Prefer physically conservative routes.  A jump or irreversible drop
    // is not equivalent to one ordinary grid step merely because both are
    // represented by one graph link.  The old breadth-first search chose a
    // long leap across a pit over the adjacent sprite bridge because it had
    // fewer links.  Dijkstra costs keep those capabilities available while
    // preferring a modest walk around whenever one is known.
    auto traversalPenalty = [](NavEdgeMode mode) {
        switch (mode)
        {
        case kNavStep: return 1;
        case kNavCrouch: return 2;
        case kNavRide: return 3;
        case kNavDrop: return 15;
        case kNavJump: return 31;
        default: return 0;
        }
    };
    std::set<std::pair<int, int> > queue;
    std::vector<int> parent(cells.size(), -1);
    std::vector<int> viaWall(cells.size(), -1);
    std::vector<NavEdgeMode> viaMode(cells.size(), kNavWalk);
    std::vector<int> bestCost(cells.size(), 0x3fffffff);
    bestCost[size_t(startCell)] = 0;
    queue.insert(std::make_pair(0, startCell));
    while (!queue.empty())
    {
        const std::pair<int, int> next = *queue.begin();
        queue.erase(queue.begin());
        const int current = next.second;
        if (next.first != bestCost[size_t(current)])
            continue;
        if (current == goal)
            break;
        const NavCell &cell = cells[size_t(current)];
        for (size_t i = 0; i < cell.links.size(); ++i)
        {
            const NavLink &link = cell.links[i];
            if (!traversableMode(link.mode))
                continue;
            // This legacy entry point plans current geometry only.  A
            // conditional connection stays in the graph, but requires the
            // interaction-aware planner below to establish its condition.
            if (link.condition.enabled)
                continue;
            if (link.target < 0 || link.target >= int(cells.size()))
                continue;
            if (edgeFailedAny(failures, current, link.target, link.wall, link.mode,
                              geometrySignature))
                continue;
            // A graph link is a concrete movement between two poses.  Count
            // its physical span, not merely one abstract hop: otherwise two
            // equal-hop routes through different doorways are tied and cell
            // insertion order chooses the crossing.  The 256-unit penalty
            // scale preserves the established preference for supported walk
            // over jump/drop shortcuts while making geometry authoritative
            // within each traversal class.
            const int64_t dx = int64_t(cells[size_t(link.target)].center.x)
                - cell.center.x;
            const int64_t dy = int64_t(cells[size_t(link.target)].center.y)
                - cell.center.y;
            int edgeCost = std::max(1, int(std::sqrt(double(dx * dx + dy * dy))))
                + traversalPenalty(link.mode) * 256;
            // A running actor has a finite turn/coast envelope.  Among
            // otherwise equivalent supported routes, prefer cells with room
            // to execute the turn instead of shaving a corner beside a pit.
            // This is deliberately a penalty rather than a rejection so a
            // genuinely narrow corridor never becomes falsely unreachable.
            const int desiredClearance = 512;
            const int edgeClearance = std::min(
                cell.clearance, cells[size_t(link.target)].clearance);
            if (edgeClearance < desiredClearance)
                edgeCost += (desiredClearance - edgeClearance) * 4;
            if (link.mode == kNavJump)
            {
                // Near-apex jumps are disproportionately fragile: a small
                // steering or collision error loses the landing entirely.
                // Prefer a staircase of known supports when it exists while
                // retaining the direct jump as a valid fallback.
                const int rise = std::max(0, cells[size_t(current)].z
                                             - cells[size_t(link.target)].z);
                const int riseUnits = (rise + 1023) / 1024;
                edgeCost += riseUnits * riseUnits * 256;
                // A long jump is also harder to execute and stop than the
                // same physical distance walked to a nearby takeoff first.
                // Linear distance alone makes those routes exactly tied, so
                // insertion order can select a full-speed leap onto a narrow
                // collision support even when connected ground reaches its
                // edge. Penalize flight span quadratically; indispensable
                // long jumps remain reachable, while a stable short takeoff
                // is preferred whenever the authoritative graph provides it.
                const int spanUnits = (int(std::sqrt(double(dx * dx + dy * dy)))
                                       + 255) / 256;
                edgeCost += spanUnits * spanUnits * 64;
            }
            const int candidateCost = bestCost[size_t(current)] + edgeCost;
            if (candidateCost >= bestCost[size_t(link.target)])
                continue;
            if (bestCost[size_t(link.target)] < 0x3fffffff)
                queue.erase(std::make_pair(bestCost[size_t(link.target)], link.target));
            bestCost[size_t(link.target)] = candidateCost;
            parent[size_t(link.target)] = current;
            viaWall[size_t(link.target)] = link.wall;
            viaMode[size_t(link.target)] = link.mode;
            queue.insert(std::make_pair(candidateCost, link.target));
        }
    }
    if (bestCost[size_t(goal)] == 0x3fffffff)
        return false;

    std::vector<int> cellsPath;
    for (int cursor = goal; cursor >= 0; cursor = parent[size_t(cursor)])
    {
        cellsPath.push_back(cursor);
        if (cursor == startCell)
            break;
    }
    if (cellsPath.empty() || cellsPath.back() != startCell)
        return false;
    std::reverse(cellsPath.begin(), cellsPath.end());
    for (size_t i = 1; i < cellsPath.size(); ++i)
    {
        const int from = cellsPath[i - 1];
        const int to = cellsPath[i];
        NavRouteStep step;
        step.fromCell = from;
        step.toCell = to;
        step.mode = viaMode[size_t(to)];
        step.wall = viaWall[size_t(to)];
        step.sourceSector = cells[size_t(from)].sector;
        step.targetSector = cells[size_t(to)].sector;
        step.sourceZ = cells[size_t(from)].z;
        step.targetZ = cells[size_t(to)].z;
        step.sourceSupport = cells[size_t(from)].support;
        step.targetSupport = cells[size_t(to)].support;
        const NavLink *link = findLink(cells[size_t(from)], to, step.wall);
        if (link && link->hasGateway)
        {
            step.gateway = link->gateway;
            step.hasGateway = true;
        }
        if (link && link->hasTakeoff)
        {
            step.takeoff = link->takeoff;
            step.hasTakeoff = true;
        }
        // Gateway and destination are different physical facts.  The former
        // is a doorway/takeoff pose on the source side; the latter is the
        // target support pose.  Collapsing both into the gateway made jump
        // execution aim at its own takeoff point and then wait for a landing
        // it could never reach.
        step.destination = cells[size_t(to)].center;
        if (link)
        {
            step.condition = link->condition;
            step.transition = link->transition;
            step.airControl = link->airControl;
            step.airFrames = link->airFrames;
            step.hasAirControl = link->hasAirControl;
        }
        outRoute.push_back(step);
    }
    return !outRoute.empty();
}

std::vector<CausalReceiver> CausalGraph::receiversFor(int channel) const
{
    std::vector<CausalReceiver> result;
    for (size_t i = 0; i < receivers.size(); ++i)
        if (receivers[i].channel == channel)
            result.push_back(receivers[i]);
    return result;
}

std::vector<CausalReceiver> CausalGraph::receiversReachableFrom(
    int channel, int maxDepth, bool terminalOnly) const
{
    std::vector<CausalReceiver> result;
    if (channel <= 0 || maxDepth < 0)
        return result;

    struct PendingChannel
    {
        int channel;
        int depth;
        PendingChannel(int aChannel, int aDepth)
            : channel(aChannel), depth(aDepth) {}
    };
    std::deque<PendingChannel> pending;
    std::set<int> visitedChannels;
    std::set<int64_t> emittedObjects;
    pending.push_back(PendingChannel(channel, 0));
    visitedChannels.insert(channel);
    while (!pending.empty())
    {
        const PendingChannel current = pending.front();
        pending.pop_front();
        const std::vector<CausalReceiver> direct = receiversFor(current.channel);
        for (size_t i = 0; i < direct.size(); ++i)
        {
            const CausalReceiver &receiver = direct[i];
            const bool hasOutgoing = receiver.outgoingChannel > 0;
            const bool canFollow = hasOutgoing && current.depth < maxDepth
                && !visitedChannels.count(receiver.outgoingChannel);
            if (canFollow)
            {
                visitedChannels.insert(receiver.outgoingChannel);
                pending.push_back(PendingChannel(receiver.outgoingChannel,
                                                 current.depth + 1));
            }

            // A receiver that forwards is an intermediate causal node, not
            // the final effect. A cycle or depth cap terminates exploration
            // safely but does not promote that relay into a fake leaf.
            if (terminalOnly && (hasOutgoing || canFollow))
                continue;
            const int64_t identity = (int64_t(receiver.object.kind) << 32)
                | uint32_t(receiver.object.id);
            if (emittedObjects.insert(identity).second)
                result.push_back(receiver);
        }
    }
    return result;
}

const Actuator *CausalGraph::actuatorById(int id) const
{
    for (size_t i = 0; i < actuators.size(); ++i)
        if (actuators[i].id == id)
            return &actuators[i];
    return nullptr;
}

std::vector<LearnedEffect> CausalGraph::effectsEstablishing(int mechanism,
                                                            int state) const
{
    std::vector<LearnedEffect> result;
    for (size_t i = 0; i < effects.size(); ++i)
        if (effects[i].mechanism == mechanism && effects[i].state == state)
            result.push_back(effects[i]);
    return result;
}

int deriveDynamicAffordances(const DynamicMechanism &mechanism,
                             int requiredClearance)
{
    int result = kAffordanceNone;
    bool anyPassable = false;
    bool anyBlocked = false;
    bool endpointsSafe = mechanism.poses.size() >= 2;
    std::set<int> firstConnections;
    bool differentConnections = false;
    for (size_t i = 0; i < mechanism.poses.size(); ++i)
    {
        const StablePose &pose = mechanism.poses[i];
        const bool passable = pose.occupiable && pose.clearance >= requiredClearance;
        anyPassable = anyPassable || passable;
        anyBlocked = anyBlocked || !passable;
        endpointsSafe = endpointsSafe && passable;
        const std::set<int> connections(pose.connectedSurfaces.begin(),
                                        pose.connectedSurfaces.end());
        if (i == 0)
            firstConnections = connections;
        else if (connections != firstConnections)
            differentConnections = true;
    }
    if (anyPassable && anyBlocked)
        result |= kAffordanceEnablePassage;

    bool sweepSafe = endpointsSafe;
    for (size_t i = 0; i < mechanism.sweepClearances.size(); ++i)
        if (mechanism.sweepClearances[i] < requiredClearance)
            sweepSafe = false;
    if (mechanism.crush && mechanism.sweepClearances.empty())
        sweepSafe = false;

    if (mechanism.carriesSupport && sweepSafe && differentConnections)
        result |= kAffordanceTransportSupportedPlayer;
    if (!sweepSafe)
        result |= kAffordanceUnsafeSweptOccupancy;
    return result;
}

NavLink makeConditionalTraversal(int target, NavEdgeMode mode, int mechanism,
                                 int state, int transition)
{
    NavLink link;
    link.target = target;
    link.mode = mode;
    link.condition = NavCondition(mechanism, state);
    link.transition = transition;
    return link;
}

struct AvailableRoute
{
    std::vector<int> cells;
    std::vector<int> links;
};

static bool findAvailableRoute(const std::vector<NavCell> &cells, int start,
                               int goal, const std::map<int, int> &states,
                               AvailableRoute &route,
                               std::vector<char> *reachable = nullptr)
{
    route.cells.clear();
    route.links.clear();
    if (start < 0 || goal < 0 || start >= int(cells.size())
        || goal >= int(cells.size()))
        return false;
    std::deque<int> queue;
    std::vector<int> parent(cells.size(), -1);
    std::vector<int> via(cells.size(), -1);
    std::vector<char> seen(cells.size(), 0);
    queue.push_back(start);
    seen[size_t(start)] = 1;
    while (!queue.empty())
    {
        const int current = queue.front();
        queue.pop_front();
        const NavCell &cell = cells[size_t(current)];
        for (size_t i = 0; i < cell.links.size(); ++i)
        {
            const NavLink &link = cell.links[i];
            if (!traversableMode(link.mode)
                || !conditionSatisfied(link.condition, states)
                || link.target < 0 || link.target >= int(cells.size())
                || seen[size_t(link.target)])
                continue;
            seen[size_t(link.target)] = 1;
            parent[size_t(link.target)] = current;
            via[size_t(link.target)] = int(i);
            queue.push_back(link.target);
        }
    }
    if (reachable)
        *reachable = seen;
    if (!seen[size_t(goal)])
        return false;
    for (int cursor = goal; cursor >= 0; cursor = parent[size_t(cursor)])
    {
        route.cells.push_back(cursor);
        if (cursor == start)
            break;
        route.links.push_back(via[size_t(cursor)]);
    }
    if (route.cells.empty() || route.cells.back() != start)
        return false;
    std::reverse(route.cells.begin(), route.cells.end());
    std::reverse(route.links.begin(), route.links.end());
    return true;
}

static void appendAvailableRoute(const std::vector<NavCell> &cells,
                                 const AvailableRoute &route,
                                 std::vector<PlanOperation> &plan)
{
    for (size_t i = 1; i < route.cells.size(); ++i)
    {
        const int from = route.cells[i - 1];
        const int to = route.cells[i];
        const NavLink &link = cells[size_t(from)].links[size_t(route.links[i - 1])];
        if (link.mode == kNavRide)
        {
            PlanOperation remain;
            remain.kind = kPlanRemainSupported;
            remain.fromCell = from;
            remain.toCell = to;
            remain.mechanism = link.condition.mechanism;
            remain.state = link.condition.state;
            plan.push_back(remain);
        }
        PlanOperation operation;
        operation.kind = kPlanTraverse;
        operation.fromCell = from;
        operation.toCell = to;
        operation.traversal = link.mode;
        operation.mechanism = link.condition.mechanism;
        operation.state = link.condition.state;
        plan.push_back(operation);
    }
}

static bool planDynamicRouteRecursive(
    const std::vector<NavCell> &cells, int start, int goal,
    std::map<int, int> &states, const CausalGraph &causality,
    std::set<int64_t> &resolving, std::vector<PlanOperation> &plan,
    DynamicPlanStats &stats, int depth)
{
    ++stats.routeSearches;
    if (depth > int(cells.size()) + int(causality.effects.size()) + 4)
        return false;

    AvailableRoute direct;
    std::vector<char> reachable;
    if (findAvailableRoute(cells, start, goal, states, direct, &reachable))
    {
        appendAvailableRoute(cells, direct, plan);
        return true;
    }

    // Only conditions on the boundary of space reachable right now matter.
    // Unrelated mechanisms are never assigned or enumerated.
    for (size_t c = 0; c < cells.size(); ++c)
    {
        if (!reachable[c])
            continue;
        const NavCell &cell = cells[c];
        for (size_t l = 0; l < cell.links.size(); ++l)
        {
            const NavLink &link = cell.links[l];
            if (!traversableMode(link.mode) || !link.condition.enabled
                || conditionSatisfied(link.condition, states))
                continue;
            const int64_t key = (int64_t(link.condition.mechanism) << 32)
                ^ uint32_t(link.condition.state);
            if (resolving.count(key))
                continue;
            const std::vector<LearnedEffect> effects = causality.effectsEstablishing(
                link.condition.mechanism, link.condition.state);
            for (size_t e = 0; e < effects.size(); ++e)
            {
                const LearnedEffect &effect = effects[e];
                const Actuator *actuator = causality.actuatorById(effect.actuator);
                if (!actuator || actuator->locationCell < 0
                    || actuator->locationCell >= int(cells.size()))
                    continue;
                resolving.insert(key);
                std::map<int, int> candidateStates = states;
                std::vector<PlanOperation> candidatePlan = plan;
                if (!planDynamicRouteRecursive(cells, start, actuator->locationCell,
                                               candidateStates, causality, resolving,
                                               candidatePlan, stats, depth + 1))
                {
                    resolving.erase(key);
                    continue;
                }

                PlanOperation activate;
                activate.kind = kPlanActivate;
                activate.fromCell = actuator->locationCell;
                activate.toCell = actuator->locationCell;
                activate.actuator = actuator->id;
                activate.activation = effect.mode;
                activate.mechanism = effect.mechanism;
                activate.state = effect.state;
                candidatePlan.push_back(activate);

                PlanOperation wait;
                wait.kind = kPlanWaitForTransition;
                wait.fromCell = actuator->locationCell;
                wait.toCell = actuator->locationCell;
                wait.mechanism = effect.mechanism;
                wait.state = effect.state;
                candidatePlan.push_back(wait);

                candidateStates[effect.mechanism] = effect.state;
                ++stats.prerequisiteExpansions;
                stats.mechanismsConsidered.insert(effect.mechanism);
                if (planDynamicRouteRecursive(cells, actuator->locationCell, goal,
                                              candidateStates, causality, resolving,
                                              candidatePlan, stats, depth + 1))
                {
                    states.swap(candidateStates);
                    plan.swap(candidatePlan);
                    resolving.erase(key);
                    return true;
                }
                resolving.erase(key);
            }
        }
    }
    return false;
}

bool planDynamicRoute(const std::vector<NavCell> &cells, int startCell,
                      int targetCell, const std::map<int, int> &mechanismStates,
                      const CausalGraph &causality,
                      std::vector<PlanOperation> &outPlan,
                      DynamicPlanStats *stats)
{
    outPlan.clear();
    DynamicPlanStats localStats;
    std::map<int, int> states = mechanismStates;
    std::set<int64_t> resolving;
    const bool result = planDynamicRouteRecursive(cells, startCell, targetCell,
                                                   states, causality, resolving,
                                                   outPlan, localStats, 0);
    if (!result)
        outPlan.clear();
    if (stats)
        *stats = localStats;
    return result;
}

static bool crossingFailed(const std::vector<NavEdgeFailure> &failures, int wall,
                           int from, int to, int geometrySignature)
{
    for (size_t i = 0; i < failures.size(); ++i)
    {
        const NavEdgeFailure &failure = failures[i];
        if (failure.wall < 0 && failure.fromCell < 0 && failure.toCell < 0)
            continue;
        if (failure.geometrySignature != 0 && geometrySignature != 0
            && failure.geometrySignature != geometrySignature)
            continue;
        if (failure.wall == wall)
            return true;
        if (failure.wall < 0 && failure.fromCell == from && failure.toCell == to)
            return true;
    }
    return false;
}

std::vector<VisibilityFrontier> deriveVisibilityFrontiers(
    const std::vector<VisibilityCell> &cells, int mergeRadius,
    int gainRadius, int approachRadius, int maximumRise)
{
    std::map<int, size_t> byId;
    for (size_t i = 0; i < cells.size(); ++i)
        byId[cells[i].id] = i;

    std::vector<VisibilityFrontier> candidates;
    const int64_t gainRadius2 = int64_t(gainRadius) * gainRadius;
    for (size_t i = 0; i < cells.size(); ++i)
    {
        const VisibilityCell &cell = cells[i];
        if (cell.observed)
            continue;
        bool bordersKnownSpace = false;
        int approachCell = -1;
        for (size_t n = 0; n < cell.neighbors.size(); ++n)
        {
            std::map<int, size_t>::const_iterator neighbor =
                byId.find(cell.neighbors[n]);
            if (neighbor != byId.end() && cells[neighbor->second].observed)
            {
                bordersKnownSpace = true;
                if (approachCell < 0 || cells[neighbor->second].reachable)
                    approachCell = cells[neighbor->second].id;
                if (cells[neighbor->second].reachable)
                    break;
            }
        }
        if (!bordersKnownSpace)
            continue;

        // A raised support or sprite top may be visible without yet having a
        // traversal edge to the floor below it. The missing prerequisite is
        // a valid takeoff/inspection pose, not proof that the surface is
        // impossible. Use the nearest observed reachable physical pose as
        // the boundary approach; execution can then discover the jump/link.
        if (!cell.reachable)
        {
            if (approachCell >= 0)
            {
                const VisibilityCell &neighbor = cells[byId.find(approachCell)->second];
                if (neighbor.reachable && neighbor.z - cell.z > maximumRise)
                    approachCell = -1;
            }
            const int64_t approachRadius2 = int64_t(approachRadius)
                * approachRadius;
            int64_t bestApproachDistance2 = approachRadius2 + 1;
            for (size_t j = 0; j < cells.size(); ++j)
            {
                const VisibilityCell &known = cells[j];
                if (!known.observed || !known.reachable)
                    continue;
                const int rise = known.z - cell.z;
                if (rise > maximumRise)
                    continue;
                const int64_t dx = int64_t(known.x) - cell.x;
                const int64_t dy = int64_t(known.y) - cell.y;
                const int64_t distance2 = dx * dx + dy * dy;
                if (distance2 < bestApproachDistance2)
                {
                    bestApproachDistance2 = distance2;
                    approachCell = known.id;
                }
            }
        }

        VisibilityFrontier frontier;
        frontier.cell = cell.id;
        frontier.approachCell = approachCell;
        frontier.reachable = cell.reachable;
        for (size_t j = 0; j < cells.size(); ++j)
        {
            const VisibilityCell &unknown = cells[j];
            if (unknown.observed || unknown.area != cell.area)
                continue;
            const int64_t dx = int64_t(unknown.x) - cell.x;
            const int64_t dy = int64_t(unknown.y) - cell.y;
            if (dx * dx + dy * dy <= gainRadius2)
                ++frontier.informationGain;
        }
        candidates.push_back(frontier);
    }

    std::sort(candidates.begin(), candidates.end(),
              [&cells, &byId](const VisibilityFrontier &a,
                              const VisibilityFrontier &b)
              {
                  const VisibilityCell &aCell = cells[byId.find(a.cell)->second];
                  const VisibilityCell &bCell = cells[byId.find(b.cell)->second];
                  const std::map<int, size_t>::const_iterator aApproachIndex =
                      byId.find(a.approachCell);
                  const std::map<int, size_t>::const_iterator bApproachIndex =
                      byId.find(b.approachCell);
                  const int64_t aDistance2 = aApproachIndex == byId.end()
                      ? INT64_MAX
                      : (int64_t(aCell.x) - cells[aApproachIndex->second].x)
                            * (int64_t(aCell.x) - cells[aApproachIndex->second].x)
                        + (int64_t(aCell.y) - cells[aApproachIndex->second].y)
                            * (int64_t(aCell.y) - cells[aApproachIndex->second].y);
                  const int64_t bDistance2 = bApproachIndex == byId.end()
                      ? INT64_MAX
                      : (int64_t(bCell.x) - cells[bApproachIndex->second].x)
                            * (int64_t(bCell.x) - cells[bApproachIndex->second].x)
                        + (int64_t(bCell.y) - cells[bApproachIndex->second].y)
                            * (int64_t(bCell.y) - cells[bApproachIndex->second].y);
                  if (aDistance2 != bDistance2)
                      return aDistance2 < bDistance2;
                  if (a.informationGain != b.informationGain)
                      return a.informationGain > b.informationGain;
                  return a.cell < b.cell;
              });

    std::vector<VisibilityFrontier> result;
    const int64_t mergeRadius2 = int64_t(mergeRadius) * mergeRadius;
    for (size_t i = 0; i < candidates.size(); ++i)
    {
        std::map<int, size_t>::const_iterator candidateIndex =
            byId.find(candidates[i].cell);
        if (candidateIndex == byId.end())
            continue;
        const VisibilityCell &candidate = cells[candidateIndex->second];
        bool merged = false;
        for (size_t j = 0; j < result.size(); ++j)
        {
            std::map<int, size_t>::const_iterator selectedIndex =
                byId.find(result[j].cell);
            if (selectedIndex == byId.end())
                continue;
            const VisibilityCell &selected = cells[selectedIndex->second];
            if (selected.area != candidate.area
                || result[j].reachable != candidates[i].reachable)
                continue;
            const int64_t dx = int64_t(selected.x) - candidate.x;
            const int64_t dy = int64_t(selected.y) - candidate.y;
            if (dx * dx + dy * dy <= mergeRadius2)
            {
                merged = true;
                break;
            }
        }
        if (!merged)
            result.push_back(candidates[i]);
    }
    return result;
}

std::vector<DerivedFrontier> deriveFrontiers(
    const std::vector<int> &visitedSectors, const std::vector<Boundary> &boundaries,
    const std::vector<InvestigateRecord> &investigated,
    const std::vector<NavEdgeFailure> &failedCrossings)
{
    std::set<int> visited(visitedSectors.begin(), visitedSectors.end());
    std::map<int, DerivedFrontier> openByDest;
    std::map<int, DerivedFrontier> blockedByDest;
    for (size_t i = 0; i < boundaries.size(); ++i)
    {
        const Boundary &boundary = boundaries[i];
        if (visited.find(boundary.from) == visited.end())
            continue;
        if (visited.find(boundary.to) != visited.end())
            continue;
        const bool open = (boundary.traversable || boundary.jumpable)
            && !crossingFailed(failedCrossings, boundary.wall, boundary.from,
                               boundary.to, boundary.geometrySignature);
        if (open)
        {
            DerivedFrontier &frontier = openByDest[boundary.to];
            frontier.destination = boundary.to;
            frontier.kind = kFrontierOpen;
            frontier.candidates.push_back(boundary);
            continue;
        }
        if (investigatedNow(investigated, boundary.wall, boundary.from, boundary.to,
                            boundary.geometrySignature))
            continue;
        DerivedFrontier &frontier = blockedByDest[boundary.to];
        frontier.destination = boundary.to;
        frontier.kind = kFrontierBlocked;
        frontier.candidates.push_back(boundary);
    }
    std::vector<DerivedFrontier> result;
    for (std::map<int, DerivedFrontier>::iterator it = openByDest.begin();
         it != openByDest.end(); ++it)
        result.push_back(it->second);
    for (std::map<int, DerivedFrontier>::iterator it = blockedByDest.begin();
         it != blockedByDest.end(); ++it)
    {
        if (openByDest.find(it->first) != openByDest.end())
            continue;
        result.push_back(it->second);
    }
    return result;
}

int selectFrontierIndex(const std::vector<DerivedFrontier> &frontiers,
                        int currentSector, const int *hops, int hopCount)
{
    int bestOpenLocal = -1;
    int bestOpenRemote = -1;
    int bestOpenHops = 0x7fffffff;
    int bestBlockedLocal = -1;
    int bestBlockedRemote = -1;
    int bestBlockedHops = 0x7fffffff;
    for (size_t i = 0; i < frontiers.size(); ++i)
    {
        const DerivedFrontier &frontier = frontiers[i];
        bool local = false;
        for (size_t c = 0; c < frontier.candidates.size(); ++c)
        {
            if (frontier.candidates[c].from == currentSector)
            {
                local = true;
                break;
            }
        }
        const int hop = (frontier.destination >= 0 && frontier.destination < hopCount)
            ? hops[frontier.destination] : -1;
        if (frontier.kind == kFrontierOpen)
        {
            if (local && bestOpenLocal < 0)
                bestOpenLocal = int(i);
            else if (!local && hop >= 0 && hop < bestOpenHops)
            {
                bestOpenHops = hop;
                bestOpenRemote = int(i);
            }
        }
        else if (frontier.kind == kFrontierBlocked)
        {
            if (local && bestBlockedLocal < 0)
                bestBlockedLocal = int(i);
            else if (!local && hop >= 0 && hop < bestBlockedHops)
            {
                bestBlockedHops = hop;
                bestBlockedRemote = int(i);
            }
        }
    }
    if (bestOpenLocal >= 0)
        return bestOpenLocal;
    if (bestOpenRemote >= 0)
        return bestOpenRemote;
    if (bestBlockedLocal >= 0)
        return bestBlockedLocal;
    return bestBlockedRemote;
}

static const char *workReason(const Opportunity &work)
{
    switch (work.kind)
    {
    case kOpportunityFrontier:
        return work.local ? "CONTINUE_FORWARD" : "RETURN_TO_UNEXPLORED_BRANCH";
    case kOpportunityPickup:
        return "COLLECT_ON_THE_WAY";
    case kOpportunityCoverage:
        return "EXPOSE_UNSEEN_LOCAL_SPACE";
    case kOpportunityExit:
        return "CONTINUE_FORWARD";
    default:
        return "SOLVE_BLOCKING_OBSTACLE";
    }
}

static bool availableNow(const Opportunity &opportunity, unsigned heldKeys,
                          unsigned availableEffects)
{
    if (opportunity.hops < 0)
        return false;
    if (opportunity.requiredKey > 0
        && !(heldKeys & (1u << unsigned(opportunity.requiredKey & 31))))
        return false;
    if (!effectRequirementSatisfied(opportunity.requiredEffects,
                                    availableEffects))
        return false;
    return true;
}

static bool isDiscoveredTask(const Opportunity &opportunity)
{
    return opportunity.kind != kOpportunityFrontier
        && opportunity.kind != kOpportunityCoverage;
}

static int workClass(const Opportunity &opportunity)
{
    if (isDiscoveredTask(opportunity) && opportunity.ready)
        return 0; // executable at the actor's present pose/component
    if (!isDiscoveredTask(opportunity))
        return 1; // explore to discover work or a missing route/prerequisite
    return 2;     // remembered task, deferred on reaching a valid pose
}

static bool betterWork(const Opportunity &candidate, const Opportunity &best)
{
    // First decide what kind of work can actually be executed.  Route risk
    // is an ordering fact between equivalent tasks, not permission for
    // generic exploration to suppress a known actionable mechanism.
    const int candidateClass = workClass(candidate);
    const int bestClass = workClass(best);
    if (candidateClass != bestClass)
        return candidateClass < bestClass;

    // Preserve future options. A known irreversible drop is considered only
    // after reversible work of the same execution class, but is never erased
    // from the ledger.
    const bool candidateSafe = candidate.oneWayRisk <= 0;
    const bool bestSafe = best.oneWayRisk <= 0;
    if (candidateSafe != bestSafe)
        return candidateSafe;

    // Exploration discovers work and the physical poses that make remembered
    // work executable. A task already at a valid action pose goes first. If
    // every known task is still pose-blocked, inspect unknown physical space;
    // once discovery is exhausted, return to the nearest deferred task. The
    // task itself never disappears from the ledger during that process.
    // A causal successor breaks ties between work of the same execution
    // class. It must not make generic coverage outrank a known task whose
    // action pose is now reachable: the world change exists to enable useful
    // work, not to impose a separate exploration mission.
    if (candidate.continuation != best.continuation)
        return candidate.continuation;

    // Exhaust the current physical component before backtracking through
    // remembered connectivity to another one, within the same class of work.
    if (candidate.local != best.local)
        return candidate.local;
    if (candidate.hops != best.hops)
        return candidate.hops < best.hops;
    if (candidate.descent != best.descent)
        return candidate.descent < best.descent;
    return candidate.id < best.id;
}

WorkSelection selectWork(const std::vector<Opportunity> &ledger,
                         unsigned heldKeys,
                         unsigned availableEffects)
{
    WorkSelection selection;
    const Opportunity *best = 0;

    for (size_t i = 0; i < ledger.size(); ++i)
    {
        const Opportunity &candidate = ledger[i];
        if (!availableNow(candidate, heldKeys, availableEffects))
            continue;
        if (!best || betterWork(candidate, *best))
            best = &candidate;
    }

    if (!best)
        return selection;
    selection.work = best->id;
    if (best->continuation)
        selection.reason = "CONSUME_ENABLED_SPACE";
    else
        selection.reason = workReason(*best);
    return selection;
}

CombatDecision chooseCombatTactic(const CombatSituation &situation)
{
    CombatDecision decision;
    if (!situation.hasThreat)
    {
        decision.tactic = kCombatNone;
        decision.overrideMovement = false;
        decision.dropCombat = true;
        decision.reason = "no_threat";
        return decision;
    }
    // Nearly dead and able to break contact: living to explore beats trading.
    if (situation.critical && situation.retreatAvailable)
    {
        decision.tactic = kCombatRetreat;
        decision.overrideMovement = true;
        decision.dropCombat = false;
        decision.reason = "critical_health_retreat";
        return decision;
    }
    if (situation.rangedAvailable)
    {
        decision.tactic = kCombatRanged;
        decision.overrideMovement = false;
        decision.dropCombat = false;
        decision.reason = "ranged_attack_line";
        return decision;
    }
    // Melee outranks retreat.  An enemy in front of the bot must never be
    // able to kill it while the bot does nothing.
    if (situation.meleeAvailable)
    {
        decision.tactic = kCombatMelee;
        decision.overrideMovement = true;
        decision.dropCombat = false;
        decision.reason = situation.immediateThreat
            ? "immediate_threat_melee" : "closing_to_melee";
        return decision;
    }
    if (situation.immediateThreat && situation.retreatAvailable)
    {
        decision.tactic = kCombatRetreat;
        decision.overrideMovement = true;
        decision.dropCombat = false;
        decision.reason = "no_usable_weapon_retreat";
        return decision;
    }
    if (situation.current == kCombatRetreat && !situation.retreatAvailable)
    {
        if (situation.immediateThreat && situation.meleeAvailable)
        {
            decision.tactic = kCombatMelee;
            decision.overrideMovement = true;
            decision.dropCombat = false;
            decision.reason = "retreat_unavailable_melee";
            return decision;
        }
        decision.tactic = kCombatNone;
        decision.overrideMovement = false;
        decision.dropCombat = true;
        decision.reason = "retreat_unavailable_no_immediate_threat";
        return decision;
    }
    decision.tactic = kCombatNone;
    decision.overrideMovement = false;
    decision.dropCombat = true;
    decision.reason = situation.immediateThreat ? "no_viable_tactic"
                                                : "threat_not_immediate";
    return decision;
}

} // namespace llmapper
