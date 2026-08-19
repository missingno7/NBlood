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
                if (!walkMode(links[l].mode))
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

    std::deque<int> queue;
    std::vector<int> parent(cells.size(), -1);
    std::vector<int> viaWall(cells.size(), -1);
    std::vector<NavEdgeMode> viaMode(cells.size(), kNavWalk);
    std::vector<char> reached(cells.size(), 0);
    queue.push_back(startCell);
    reached[size_t(startCell)] = 1;
    while (!queue.empty() && !reached[size_t(goal)])
    {
        const int current = queue.front();
        queue.pop_front();
        const NavCell &cell = cells[size_t(current)];
        for (size_t i = 0; i < cell.links.size(); ++i)
        {
            const NavLink &link = cell.links[i];
            if (!traversableMode(link.mode))
                continue;
            if (link.target < 0 || link.target >= int(cells.size()))
                continue;
            if (edgeFailedAny(failures, current, link.target, link.wall, link.mode,
                              geometrySignature))
                continue;
            if (reached[size_t(link.target)])
                continue;
            reached[size_t(link.target)] = 1;
            parent[size_t(link.target)] = current;
            viaWall[size_t(link.target)] = link.wall;
            viaMode[size_t(link.target)] = link.mode;
            queue.push_back(link.target);
        }
    }
    if (!reached[size_t(goal)])
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
        const NavLink *link = findLink(cells[size_t(from)], to, step.wall);
        if (link && link->hasGateway)
        {
            step.gateway = link->gateway;
            step.hasGateway = true;
            step.destination = link->gateway;
        }
        else
            step.destination = cells[size_t(to)].center;
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

static bool availableNow(const Opportunity &opportunity, int tick, unsigned heldKeys)
{
    if (opportunity.hops < 0)
        return false;
    if (opportunity.dormantUntil > tick)
        return false;
    if (opportunity.requiredKey > 0
        && !(heldKeys & (1u << unsigned(opportunity.requiredKey & 31))))
        return false;
    return true;
}

// Deeper first, then closer.  Depth is the branch the bot is already on.
static bool deeperThan(const Opportunity &candidate, const Opportunity &best)
{
    if (candidate.depth != best.depth)
        return candidate.depth > best.depth;
    return candidate.hops < best.hops;
}

Mission selectMission(const std::vector<Opportunity> &ledger, int tick,
                      unsigned heldKeys, int committedOpportunity)
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
        if (!availableNow(candidate, tick, heldKeys))
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
