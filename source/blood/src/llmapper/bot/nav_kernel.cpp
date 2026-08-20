//-------------------------------------------------------------------------
// Engine-free movement / exploration kernel for the LLMapper bot.
//-------------------------------------------------------------------------
#include "nav_kernel.h"

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

bool planNavRoute(const std::vector<NavCell> &cells, int startCell, int targetCell,
                  int targetX, int targetY, int targetSector, int crossingWall,
                  const std::vector<NavEdgeFailure> &failures, int geometrySignature,
                  std::vector<NavRouteStep> &outRoute)
{
    outRoute.clear();
    if (startCell < 0 || startCell >= int(cells.size()))
        return false;

    int goal = targetCell;
    if (goal < 0 || goal >= int(cells.size()))
    {
        int best = -1;
        int bestDistance = 0x7fffffff;
        for (size_t i = 0; i < cells.size(); ++i)
        {
            if (targetSector >= 0 && cells[i].sector != targetSector)
                continue;
            const int dx = cells[i].center.x - targetX;
            const int dy = cells[i].center.y - targetY;
            const int distance = dx * dx + dy * dy;
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = int(i);
            }
        }
        goal = best;
    }
    if (goal < 0)
        return false;
    if (goal == startCell)
    {
        NavRouteStep step;
        step.fromCell = startCell;
        step.toCell = startCell;
        step.destination = NavWaypoint(targetX, targetY);
        step.mode = kNavWalk;
        step.sourceSector = cells[size_t(startCell)].sector;
        step.targetSector = targetSector >= 0 ? targetSector : step.sourceSector;
        step.wall = crossingWall;
        outRoute.push_back(step);
        return true;
    }

    // Prefer physically conservative routes.  A jump or irreversible drop
    // is not equivalent to one ordinary grid step merely because both are
    // represented by one graph link.  The old breadth-first search chose a
    // long leap across a pit over the adjacent sprite bridge because it had
    // fewer links.  Dijkstra costs keep those capabilities available while
    // preferring a modest walk around whenever one is known.
    auto traversalCost = [](NavEdgeMode mode) {
        switch (mode)
        {
        case kNavStep: return 2;
        case kNavCrouch: return 3;
        case kNavRide: return 4;
        case kNavDrop: return 16;
        case kNavJump: return 32;
        default: return 1;
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
            const int candidateCost = bestCost[size_t(current)]
                + traversalCost(link.mode);
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
            step.destination = link->gateway;
        }
        else
            step.destination = cells[size_t(to)].center;
        if (link)
        {
            step.condition = link->condition;
            step.transition = link->transition;
        }
        outRoute.push_back(step);
    }
    if (!outRoute.empty())
    {
        NavRouteStep &last = outRoute.back();
        last.destination = NavWaypoint(targetX, targetY);
        if (crossingWall >= 0)
            last.wall = crossingWall;
        if (targetSector >= 0)
            last.targetSector = targetSector;
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

const char *missionReason(MissionKind kind)
{
    switch (kind)
    {
    case kMissionContinue: return "CONTINUE_FORWARD";
    case kMissionReturnForKey: return "RETURN_FOR_KEY_DOOR";
    case kMissionReturnToBranch: return "RETURN_TO_UNEXPLORED_BRANCH";
    case kMissionSolveBlocker: return "SOLVE_BLOCKING_OBSTACLE";
    case kMissionCollect: return "COLLECT_ON_THE_WAY";
    case kMissionExpose: return "EXPOSE_UNSEEN_LOCAL_SPACE";
    default: return "NO_KNOWN_PROGRESS";
    }
}

static bool availableNow(const Opportunity &opportunity, int tick, unsigned heldKeys,
                         unsigned availableEffects)
{
    if (opportunity.hops < 0)
        return false;
    if (opportunity.dormantUntil > tick)
        return false;
    if (opportunity.requiredKey > 0
        && !(heldKeys & (1u << unsigned(opportunity.requiredKey & 31))))
        return false;
    if ((opportunity.requiredEffects & availableEffects)
        != opportunity.requiredEffects)
        return false;
    return true;
}

// Deeper first, then closer.  Depth is the branch the bot is already on.
static bool deeperThan(const Opportunity &candidate, const Opportunity &best)
{
    if (candidate.depth != best.depth)
        return candidate.depth > best.depth;
    if (candidate.hops != best.hops)
        return candidate.hops < best.hops;
    // Equally deep and equally close, so prefer the one that does not cost
    // height.  A drop is cheap to take and expensive to undo -- a chain of
    // individually survivable ones walks the bot down into somewhere it
    // cannot climb out of -- while a ledge across the way leaves the rest of
    // the level exactly as reachable as it was.  Without this the tie falls
    // to whichever was discovered first, which is always the plain doorway.
    return candidate.descent < best.descent;
}

Mission selectMission(const std::vector<Opportunity> &ledger, int tick,
                      unsigned heldKeys, int committedOpportunity,
                      unsigned availableEffects)
{
    Mission mission;
    const Opportunity *keyDoor = 0;
    const Opportunity *localFrontier = 0;
    const Opportunity *remoteFrontier = 0;
    const Opportunity *localPickup = 0;
    const Opportunity *blocker = 0;
    const Opportunity *localBlocker = 0;
    const Opportunity *coverage = 0;
    const Opportunity *committed = 0;

    for (size_t i = 0; i < ledger.size(); ++i)
    {
        const Opportunity &candidate = ledger[i];
        if (candidate.id == committedOpportunity && candidate.hops >= 0)
            committed = &candidate;
        if (!availableNow(candidate, tick, heldKeys, availableEffects))
            continue;
        switch (candidate.kind)
        {
        case kOpportunityLocked:
            // Reaching this arm means the key is now held: a remembered
            // locked door has just become the strongest progression clue
            // in the level.  Nearest one wins; depth is irrelevant here.
            if (!keyDoor || candidate.hops < keyDoor->hops)
                keyDoor = &candidate;
            break;
        case kOpportunityFrontier:
            if (candidate.local)
            {
                if (!localFrontier || deeperThan(candidate, *localFrontier))
                    localFrontier = &candidate;
            }
            else if (!remoteFrontier || deeperThan(candidate, *remoteFrontier))
                remoteFrontier = &candidate;
            break;
        case kOpportunityPickup:
            if (candidate.local && (!localPickup || candidate.hops < localPickup->hops))
                localPickup = &candidate;
            break;
        case kOpportunityBlocked:
        case kOpportunityInteraction:
            if (candidate.local)
            {
                if (!localBlocker || deeperThan(candidate, *localBlocker))
                    localBlocker = &candidate;
            }
            else if (!blocker || deeperThan(candidate, *blocker))
                blocker = &candidate;
            break;
        case kOpportunityCoverage:
            // Nearest unseen space first: a Build sector is not an
            // observation unit, and the interesting thing is usually just
            // around the corner of wherever the bot already is.
            if (!coverage || candidate.hops < coverage->hops
                || (candidate.hops == coverage->hops && candidate.local && !coverage->local))
                coverage = &candidate;
            break;
        }
    }

    // A held key changes the progression model immediately, and an obvious
    // threat-free pickup underfoot is free.  Everything else defers to the
    // standing commitment so the bot keeps its momentum.
    if (keyDoor)
    {
        mission.kind = kMissionReturnForKey;
        mission.opportunity = keyDoor->id;
    }
    else if (localPickup)
    {
        mission.kind = kMissionCollect;
        mission.opportunity = localPickup->id;
    }
    else if (committed && committed->dormantUntil <= tick)
    {
        mission.kind = committed->kind == kOpportunityFrontier
            ? (committed->local ? kMissionContinue : kMissionReturnToBranch)
            : kMissionSolveBlocker;
        mission.opportunity = committed->id;
    }
    else if (localFrontier)
    {
        mission.kind = kMissionContinue;
        mission.opportunity = localFrontier->id;
    }
    // An obstacle right here that the bot knows how to solve beats walking
    // to the far side of the level for a branch it could take afterwards.
    // Ranking a remote frontier first is what made the bot arrive at a shut
    // door, give up within seconds, and trek away without ever trying the
    // mechanism standing in front of it.
    else if (localBlocker)
    {
        mission.kind = kMissionSolveBlocker;
        mission.opportunity = localBlocker->id;
    }
    else if (remoteFrontier)
    {
        mission.kind = kMissionReturnToBranch;
        mission.opportunity = remoteFrontier->id;
    }
    else if (blocker)
    {
        mission.kind = kMissionSolveBlocker;
        mission.opportunity = blocker->id;
    }
    // Nothing known is actionable.  Before concluding the level is finished,
    // go and look at the reachable space that has never been observed --
    // entering a sector is not the same as having seen what is in it.
    else if (coverage)
    {
        mission.kind = kMissionExpose;
        mission.opportunity = coverage->id;
    }
    mission.reason = missionReason(mission.kind);
    return mission;
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
