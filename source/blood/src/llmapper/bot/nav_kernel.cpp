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

static int64_t skeletonCross(const NavWaypoint &a, const NavWaypoint &b,
                             const NavWaypoint &c)
{
    return int64_t(b.x - a.x) * (c.y - a.y)
        - int64_t(b.y - a.y) * (c.x - a.x);
}

static int64_t skeletonArea2(const std::vector<NavWaypoint> &polygon)
{
    int64_t area = 0;
    for (size_t i = 0; i < polygon.size(); ++i)
    {
        const NavWaypoint &a = polygon[i];
        const NavWaypoint &b = polygon[(i + 1) % polygon.size()];
        area += int64_t(a.x) * b.y - int64_t(a.y) * b.x;
    }
    return area;
}

static void removeSkeletonCollinear(std::vector<NavWaypoint> &polygon)
{
    for (size_t i = 0; polygon.size() >= 3 && i < polygon.size(); )
    {
        const size_t previous = (i + polygon.size() - 1) % polygon.size();
        const size_t next = (i + 1) % polygon.size();
        if (skeletonCross(polygon[previous], polygon[i], polygon[next]) == 0)
            polygon.erase(polygon.begin() + i);
        else
            ++i;
    }
}

static bool skeletonPointInTriangle(const NavWaypoint &point,
                                    const NavWaypoint &a,
                                    const NavWaypoint &b,
                                    const NavWaypoint &c)
{
    const int64_t ab = skeletonCross(a, b, point);
    const int64_t bc = skeletonCross(b, c, point);
    const int64_t ca = skeletonCross(c, a, point);
    return ab >= 0 && bc >= 0 && ca >= 0;
}

static bool skeletonConvex(const std::vector<NavWaypoint> &polygon)
{
    if (polygon.size() < 3)
        return false;
    for (size_t i = 0; i < polygon.size(); ++i)
        if (skeletonCross(
                polygon[i], polygon[(i + 1) % polygon.size()],
                polygon[(i + 2) % polygon.size()]) < 0)
            return false;
    return true;
}

static bool mergeSkeletonPolygons(const std::vector<NavWaypoint> &first,
                                  const std::vector<NavWaypoint> &second,
                                  std::vector<NavWaypoint> &merged)
{
    for (size_t i = 0; i < first.size(); ++i)
    {
        const NavWaypoint &a = first[i];
        const NavWaypoint &b = first[(i + 1) % first.size()];
        for (size_t j = 0; j < second.size(); ++j)
        {
            const NavWaypoint &reverseA = second[j];
            const NavWaypoint &reverseB = second[(j + 1) % second.size()];
            if (a.x != reverseB.x || a.y != reverseB.y
                || b.x != reverseA.x || b.y != reverseA.y)
                continue;
            merged.clear();
            for (size_t k = 0; k < first.size(); ++k)
                merged.push_back(first[(i + 1 + k) % first.size()]);
            for (size_t k = 1; k + 1 < second.size(); ++k)
                merged.push_back(second[(j + 1 + k) % second.size()]);
            removeSkeletonCollinear(merged);
            if (skeletonConvex(merged))
                return true;
        }
    }
    return false;
}

static NavWaypoint skeletonCentroid(const std::vector<NavWaypoint> &polygon)
{
    int64_t area = 0;
    int64_t weightedX = 0;
    int64_t weightedY = 0;
    for (size_t i = 0; i < polygon.size(); ++i)
    {
        const NavWaypoint &a = polygon[i];
        const NavWaypoint &b = polygon[(i + 1) % polygon.size()];
        const int64_t cross = int64_t(a.x) * b.y
            - int64_t(a.y) * b.x;
        area += cross;
        weightedX += int64_t(a.x + b.x) * cross;
        weightedY += int64_t(a.y + b.y) * cross;
    }
    if (area == 0)
    {
        int64_t x = 0, y = 0;
        for (const NavWaypoint &point : polygon)
        {
            x += point.x;
            y += point.y;
        }
        return NavWaypoint(int(x / int64_t(polygon.size())),
                           int(y / int64_t(polygon.size())));
    }
    return NavWaypoint(int(weightedX / (3 * area)),
                       int(weightedY / (3 * area)));
}

static bool skeletonPointOnSegment(const NavWaypoint &point,
                                   const NavWaypoint &a,
                                   const NavWaypoint &b)
{
    if (skeletonCross(a, b, point) != 0)
        return false;
    return point.x >= std::min(a.x, b.x)
        && point.x <= std::max(a.x, b.x)
        && point.y >= std::min(a.y, b.y)
        && point.y <= std::max(a.y, b.y);
}

// Split long shared edges at every incident polygon vertex.  A sweep cell
// can meet two cells across different portions of one edge; representing
// those portions explicitly gives the sparse graph one gateway per real
// adjacency without inserting any navigation sample interval.
static void alignSkeletonPieceEdges(
    std::vector<std::vector<NavWaypoint> > &pieces)
{
    std::vector<NavWaypoint> vertices;
    for (const std::vector<NavWaypoint> &piece : pieces)
        vertices.insert(vertices.end(), piece.begin(), piece.end());
    for (std::vector<NavWaypoint> &piece : pieces)
    {
        std::vector<NavWaypoint> aligned;
        for (size_t edge = 0; edge < piece.size(); ++edge)
        {
            const NavWaypoint &a = piece[edge];
            const NavWaypoint &b = piece[(edge + 1) % piece.size()];
            aligned.push_back(a);
            std::vector<NavWaypoint> interior;
            for (const NavWaypoint &candidate : vertices)
            {
                if ((candidate.x == a.x && candidate.y == a.y)
                    || (candidate.x == b.x && candidate.y == b.y)
                    || !skeletonPointOnSegment(candidate, a, b))
                    continue;
                if (std::find_if(interior.begin(), interior.end(),
                    [&](const NavWaypoint &known) {
                        return known.x == candidate.x
                            && known.y == candidate.y;
                    }) == interior.end())
                    interior.push_back(candidate);
            }
            std::sort(interior.begin(), interior.end(),
                [&](const NavWaypoint &first, const NavWaypoint &second) {
                    const int64_t firstDistance =
                        int64_t(first.x - a.x) * (first.x - a.x)
                        + int64_t(first.y - a.y) * (first.y - a.y);
                    const int64_t secondDistance =
                        int64_t(second.x - a.x) * (second.x - a.x)
                        + int64_t(second.y - a.y) * (second.y - a.y);
                    return firstDistance < secondDistance;
                });
            aligned.insert(aligned.end(), interior.begin(), interior.end());
        }
        piece.swap(aligned);
    }
}

static ConvexSkeleton finishConvexSkeleton(
    std::vector<std::vector<NavWaypoint> > pieces)
{
    ConvexSkeleton result;
    if (pieces.empty())
        return result;

    alignSkeletonPieceEdges(pieces);
    // Maximal convex merging makes the result independent of arbitrary
    // triangle or sweep density.
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (size_t i = 0; i < pieces.size() && !changed; ++i)
            for (size_t j = i + 1; j < pieces.size(); ++j)
            {
                std::vector<NavWaypoint> merged;
                if (!mergeSkeletonPolygons(pieces[i], pieces[j], merged))
                    continue;
                pieces[i] = merged;
                pieces.erase(pieces.begin() + j);
                changed = true;
                break;
            }
    }
    alignSkeletonPieceEdges(pieces);

    for (const std::vector<NavWaypoint> &piece : pieces)
    {
        SkeletonCell cell;
        cell.polygon = piece;
        cell.center = skeletonCentroid(piece);
        result.cells.push_back(cell);
    }
    for (size_t i = 0; i < pieces.size(); ++i)
        for (size_t j = i + 1; j < pieces.size(); ++j)
            for (size_t a = 0; a < pieces[i].size(); ++a)
            {
                const NavWaypoint &start = pieces[i][a];
                const NavWaypoint &end = pieces[i][(a + 1) % pieces[i].size()];
                for (size_t b = 0; b < pieces[j].size(); ++b)
                {
                    const NavWaypoint &reverseStart = pieces[j][b];
                    const NavWaypoint &reverseEnd = pieces[j][
                        (b + 1) % pieces[j].size()];
                    if (start.x != reverseEnd.x || start.y != reverseEnd.y
                        || end.x != reverseStart.x || end.y != reverseStart.y)
                        continue;
                    SkeletonGateway gateway;
                    gateway.first = int(i);
                    gateway.second = int(j);
                    gateway.center = NavWaypoint(
                        (start.x + end.x) / 2,
                        (start.y + end.y) / 2);
                    result.gateways.push_back(gateway);
                }
            }
    return result;
}

ConvexSkeleton buildConvexSkeleton(
    const std::vector<NavWaypoint> &footprint)
{
    std::vector<NavWaypoint> polygon = footprint;
    if (polygon.size() > 1
        && polygon.front().x == polygon.back().x
        && polygon.front().y == polygon.back().y)
        polygon.pop_back();
    removeSkeletonCollinear(polygon);
    if (polygon.size() < 3)
        return ConvexSkeleton();
    if (skeletonArea2(polygon) < 0)
        std::reverse(polygon.begin(), polygon.end());

    std::vector<int> remaining;
    for (size_t i = 0; i < polygon.size(); ++i)
        remaining.push_back(int(i));
    std::vector<std::vector<NavWaypoint> > pieces;
    while (remaining.size() > 3)
    {
        bool clipped = false;
        for (size_t cursor = 0; cursor < remaining.size(); ++cursor)
        {
            const int previous = remaining[
                (cursor + remaining.size() - 1) % remaining.size()];
            const int current = remaining[cursor];
            const int next = remaining[(cursor + 1) % remaining.size()];
            if (skeletonCross(polygon[size_t(previous)],
                              polygon[size_t(current)],
                              polygon[size_t(next)]) <= 0)
                continue;
            bool contains = false;
            for (int candidate : remaining)
            {
                if (candidate == previous || candidate == current
                    || candidate == next)
                    continue;
                if (skeletonPointInTriangle(
                        polygon[size_t(candidate)],
                        polygon[size_t(previous)],
                        polygon[size_t(current)],
                        polygon[size_t(next)]))
                {
                    contains = true;
                    break;
                }
            }
            if (contains)
                continue;
            pieces.push_back({ polygon[size_t(previous)],
                               polygon[size_t(current)],
                               polygon[size_t(next)] });
            remaining.erase(remaining.begin() + cursor);
            clipped = true;
            break;
        }
        if (!clipped)
            return ConvexSkeleton();
    }
    pieces.push_back({ polygon[size_t(remaining[0])],
                       polygon[size_t(remaining[1])],
                       polygon[size_t(remaining[2])] });

    return finishConvexSkeleton(pieces);
}

ConvexSkeleton buildConvexSkeleton(
    const std::vector<std::vector<NavWaypoint> > &inputContours)
{
    std::vector<std::vector<NavWaypoint> > contours;
    std::vector<int> splitX;
    for (std::vector<NavWaypoint> contour : inputContours)
    {
        if (contour.size() > 1
            && contour.front().x == contour.back().x
            && contour.front().y == contour.back().y)
            contour.pop_back();
        removeSkeletonCollinear(contour);
        if (contour.size() < 3)
            continue;
        for (const NavWaypoint &point : contour)
            splitX.push_back(point.x);
        contours.push_back(contour);
    }
    if (contours.empty())
        return ConvexSkeleton();
    if (contours.size() == 1)
        return buildConvexSkeleton(contours.front());
    std::sort(splitX.begin(), splitX.end());
    splitX.erase(std::unique(splitX.begin(), splitX.end()), splitX.end());

    struct Crossing
    {
        const NavWaypoint *a;
        const NavWaypoint *b;
        long double y;
    };
    auto edgeY = [](const NavWaypoint &a, const NavWaypoint &b,
                    long double x) {
        return a.y + (x - a.x) * (b.y - a.y)
            / static_cast<long double>(b.x - a.x);
    };
    std::vector<std::vector<NavWaypoint> > pieces;
    for (size_t slab = 0; slab + 1 < splitX.size(); ++slab)
    {
        const int left = splitX[slab];
        const int right = splitX[slab + 1];
        if (left == right)
            continue;
        const long double middle =
            (static_cast<long double>(left)
             + static_cast<long double>(right)) / 2.0L;
        std::vector<Crossing> crossings;
        for (const std::vector<NavWaypoint> &contour : contours)
            for (size_t edge = 0; edge < contour.size(); ++edge)
            {
                const NavWaypoint &a = contour[edge];
                const NavWaypoint &b = contour[(edge + 1) % contour.size()];
                if (a.x == b.x
                    || middle <= std::min(a.x, b.x)
                    || middle >= std::max(a.x, b.x))
                    continue;
                crossings.push_back(Crossing{ &a, &b, edgeY(a, b, middle) });
            }
        std::sort(crossings.begin(), crossings.end(),
                  [](const Crossing &a, const Crossing &b) {
                      return a.y < b.y;
                  });
        // Even/odd fill handles outer contours and arbitrarily nested holes
        // without depending on source winding or engine ownership.
        for (size_t crossing = 0; crossing + 1 < crossings.size();
             crossing += 2)
        {
            const Crossing &lower = crossings[crossing];
            const Crossing &upper = crossings[crossing + 1];
            const int lowerLeft = int(std::llround(
                edgeY(*lower.a, *lower.b, left)));
            const int lowerRight = int(std::llround(
                edgeY(*lower.a, *lower.b, right)));
            const int upperLeft = int(std::llround(
                edgeY(*upper.a, *upper.b, left)));
            const int upperRight = int(std::llround(
                edgeY(*upper.a, *upper.b, right)));
            std::vector<NavWaypoint> piece = {
                { left, lowerLeft }, { right, lowerRight },
                { right, upperRight }, { left, upperLeft },
            };
            for (size_t i = 0; piece.size() > 1 && i < piece.size(); )
            {
                const size_t next = (i + 1) % piece.size();
                if (piece[i].x == piece[next].x
                    && piece[i].y == piece[next].y)
                    piece.erase(piece.begin() + next);
                else
                    ++i;
            }
            removeSkeletonCollinear(piece);
            if (piece.size() < 3 || skeletonArea2(piece) == 0)
                continue;
            if (skeletonArea2(piece) < 0)
                std::reverse(piece.begin(), piece.end());
            pieces.push_back(piece);
        }
    }
    return finishConvexSkeleton(pieces);
}

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

void markReachableNavCells(const std::vector<NavCell> &cells, PoseId startCell,
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
                || edgeFailedAny(failures, current, link.target, link.boundary,
                                 link.mode, geometrySignature))
                continue;
            reachable[size_t(link.target)] = 1;
            queue.push_back(link.target);
        }
    }
}

int linkTranslatedNavLayers(std::vector<NavCell> &cells, RegionId upperRegion,
                            SupportId upperSupport, RegionId lowerRegion,
                            SupportId lowerSupport, int deltaX, int deltaY,
                            int maximumError, TransitionId transition)
{
    const int64_t maximumError2 = int64_t(maximumError) * maximumError;
    int linked = 0;
    auto addLink = [&](PoseId from, PoseId to, NavEdgeMode mode,
                       const NavWaypoint &gateway) {
        if (from < 0 || to < 0 || from >= int(cells.size())
            || to >= int(cells.size()) || from == to)
            return false;
        for (const NavLink &link : cells[size_t(from)].links)
            if (link.target == to && link.transition == transition)
                return false;
        NavLink link;
        link.target = to;
        link.mode = mode;
        link.gateway = gateway;
        link.hasGateway = true;
        link.transition = transition;
        cells[size_t(from)].links.push_back(link);
        return true;
    };

    for (const NavCell &upper : cells)
    {
        if (upper.region != upperRegion || upper.support != upperSupport)
            continue;
        const int wantedX = upper.center.x + deltaX;
        const int wantedY = upper.center.y + deltaY;
        int lowerId = -1;
        int64_t bestDistance = maximumError2 + 1;
        for (const NavCell &lower : cells)
        {
            if (lower.region != lowerRegion || lower.support != lowerSupport)
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
        // A translated-layer link connects two coordinate-space projections. The
        // player walks through its source pose and the engine translates the
        // body to the receiving layer; no ballistic capability is involved.
        // Labelling this DROP/JUMP handed a remote translated coordinate to
        // the jump executor and invented a flight across ordinary geometry.
        if (addLink(upper.id, lower.id, kNavWalk, upper.center))
            ++linked;
        if (addLink(lower.id, upper.id, kNavWalk, lower.center))
            ++linked;
    }
    return linked;
}

static const NavLink *findLink(const NavCell &cell, PoseId target,
                               BoundaryId boundary)
{
    for (size_t i = 0; i < cell.links.size(); ++i)
    {
        const NavLink &link = cell.links[i];
        if (link.target != target)
            continue;
        if (boundary && link.boundary && link.boundary != boundary)
            continue;
        return &link;
    }
    return nullptr;
}

static bool conditionSatisfied(const NavCondition &condition,
                               const std::map<StateVariableId, int> &states)
{
    if (!condition.enabled)
        return true;
    std::map<StateVariableId, int>::const_iterator found =
        states.find(condition.variable);
    return found != states.end() && found->second == condition.state;
}

bool planNavRoute(const std::vector<NavCell> &cells, PoseId startCell,
                  PoseId targetCell,
                  const std::vector<NavEdgeFailure> &failures,
                  int geometrySignature,
                  std::vector<NavRouteStep> &outRoute)
{
    outRoute.clear();
    if (startCell < 0 || startCell >= int(cells.size()))
        return false;

    // Routing begins and ends at concrete physical poses. A region label is
    // too lossy to manufacture a goal: several disconnected supports can
    // share it. The caller must resolve its task to a pose first.
    const int goal = targetCell;
    if (goal < 0 || goal >= int(cells.size()))
        return false;
    if (goal == startCell)
        return true;

    // Prefer physically conservative routes.  A jump or irreversible drop
    // is not equivalent to one ordinary grid step merely because both are
    // represented by one graph link.  The old breadth-first search chose a
    // long leap across a pit over an adjacent support because it had
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
    std::vector<BoundaryId> viaBoundary(cells.size());
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
            if (edgeFailedAny(failures, current, link.target, link.boundary, link.mode,
                              geometrySignature))
                continue;
            // A graph link is a concrete movement between two poses.  Count
            // its physical span, not merely one abstract hop: otherwise two
            // equal-hop routes through different boundaries are tied and pose
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
            viaBoundary[size_t(link.target)] = link.boundary;
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
        step.boundary = viaBoundary[size_t(to)];
        step.sourceRegion = cells[size_t(from)].region;
        step.targetRegion = cells[size_t(to)].region;
        step.sourceZ = cells[size_t(from)].z;
        step.targetZ = cells[size_t(to)].z;
        step.sourceSupport = cells[size_t(from)].support;
        step.targetSupport = cells[size_t(to)].support;
        const NavLink *link = findLink(cells[size_t(from)], to, step.boundary);
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
        // is a gateway/takeoff pose on the source side; the latter is the
        // target support pose.  Collapsing both into the gateway made jump
        // execution aim at its own takeoff point and then wait for a landing
        // it could never reach.
        step.destination = cells[size_t(to)].center;
        if (link)
        {
            step.condition = link->condition;
            step.transition = link->transition;
            step.airControl = link->airControl;
            step.airControlAfter = link->airControlAfter;
            step.airControlSwitchFrame = link->airControlSwitchFrame;
            step.airFrames = link->airFrames;
            step.launchVelocity = link->launchVelocity;
            step.hasAirControl = link->hasAirControl;
        }
        outRoute.push_back(step);
    }
    return !outRoute.empty();
}

const Affordance *CausalGraph::affordanceById(AffordanceId id) const
{
    for (size_t i = 0; i < affordances.size(); ++i)
        if (affordances[i].id == id)
            return &affordances[i];
    return nullptr;
}

std::vector<LearnedEffect> CausalGraph::effectsEstablishing(
    StateVariableId variable, int state) const
{
    std::vector<LearnedEffect> result;
    for (size_t i = 0; i < effects.size(); ++i)
        if (effects[i].variable == variable && effects[i].state == state)
            result.push_back(effects[i]);
    return result;
}

int deriveDynamicAffordances(const StatefulGeometry &geometry,
                             int requiredClearance)
{
    int result = kAffordanceNone;
    bool anyPassable = false;
    bool anyBlocked = false;
    bool endpointsSafe = geometry.poses.size() >= 2;
    std::set<int> firstConnections;
    bool differentConnections = false;
    for (size_t i = 0; i < geometry.poses.size(); ++i)
    {
        const StablePose &pose = geometry.poses[i];
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
    for (size_t i = 0; i < geometry.sweepClearances.size(); ++i)
        if (geometry.sweepClearances[i] < requiredClearance)
            sweepSafe = false;
    if (geometry.crush && geometry.sweepClearances.empty())
        sweepSafe = false;

    if (geometry.carriesSupport && sweepSafe && differentConnections)
        result |= kAffordanceTransportSupportedPlayer;
    if (!sweepSafe)
        result |= kAffordanceUnsafeSweptOccupancy;
    return result;
}

NavLink makeConditionalTraversal(PoseId target, NavEdgeMode mode,
                                 StateVariableId variable, int state,
                                 TransitionId transition)
{
    NavLink link;
    link.target = target;
    link.mode = mode;
    link.condition = NavCondition(variable, state);
    link.transition = transition;
    return link;
}

struct AvailableRoute
{
    std::vector<int> cells;
    std::vector<int> links;
};

static bool findAvailableRoute(const std::vector<NavCell> &cells, int start,
                               int goal,
                               const std::map<StateVariableId, int> &states,
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
            remain.variable = link.condition.variable;
            remain.state = link.condition.state;
            plan.push_back(remain);
        }
        PlanOperation operation;
        operation.kind = kPlanTraverse;
        operation.fromCell = from;
        operation.toCell = to;
        operation.traversal = link.mode;
        operation.variable = link.condition.variable;
        operation.state = link.condition.state;
        plan.push_back(operation);
    }
}

static bool planDynamicRouteRecursive(
    const std::vector<NavCell> &cells, int start, int goal,
    std::map<StateVariableId, int> &states, const CausalGraph &causality,
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
    // Unrelated state variables are never assigned or enumerated.
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
            const int64_t key = (int64_t(link.condition.variable.value) << 32)
                ^ uint32_t(link.condition.state);
            if (resolving.count(key))
                continue;
            const std::vector<LearnedEffect> effects = causality.effectsEstablishing(
                link.condition.variable, link.condition.state);
            for (size_t e = 0; e < effects.size(); ++e)
            {
                const LearnedEffect &effect = effects[e];
                const Affordance *affordance =
                    causality.affordanceById(effect.affordance);
                if (!affordance || !affordance->actionPose
                    || affordance->actionPose >= int(cells.size()))
                    continue;
                resolving.insert(key);
                std::map<StateVariableId, int> candidateStates = states;
                std::vector<PlanOperation> candidatePlan = plan;
                if (!planDynamicRouteRecursive(cells, start, affordance->actionPose,
                                               candidateStates, causality, resolving,
                                               candidatePlan, stats, depth + 1))
                {
                    resolving.erase(key);
                    continue;
                }

                PlanOperation activate;
                activate.kind = kPlanActivate;
                activate.fromCell = affordance->actionPose;
                activate.toCell = affordance->actionPose;
                activate.affordance = affordance->id;
                activate.action = effect.action;
                activate.variable = effect.variable;
                activate.state = effect.state;
                candidatePlan.push_back(activate);

                PlanOperation wait;
                wait.kind = kPlanWaitForTransition;
                wait.fromCell = affordance->actionPose;
                wait.toCell = affordance->actionPose;
                wait.variable = effect.variable;
                wait.state = effect.state;
                candidatePlan.push_back(wait);

                candidateStates[effect.variable] = effect.state;
                ++stats.prerequisiteExpansions;
                stats.variablesConsidered.insert(effect.variable);
                if (planDynamicRouteRecursive(cells, affordance->actionPose, goal,
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

bool planDynamicRoute(const std::vector<NavCell> &cells, PoseId startCell,
                      PoseId targetCell,
                      const std::map<StateVariableId, int> &worldState,
                      const CausalGraph &causality,
                      std::vector<PlanOperation> &outPlan,
                      DynamicPlanStats *stats)
{
    outPlan.clear();
    DynamicPlanStats localStats;
    std::map<StateVariableId, int> states = worldState;
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

static bool crossingFailed(const std::vector<NavEdgeFailure> &failures,
                           BoundaryId boundary, int geometrySignature)
{
    for (size_t i = 0; i < failures.size(); ++i)
    {
        const NavEdgeFailure &failure = failures[i];
        if (!failure.boundary)
            continue;
        if (failure.geometrySignature != 0 && geometrySignature != 0
            && failure.geometrySignature != geometrySignature)
            continue;
        if (failure.boundary == boundary)
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

        // A raised support may be visible without yet having a
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
    const std::vector<RegionId> &visitedRegions,
    const std::vector<Boundary> &boundaries,
    const std::vector<InvestigateRecord> &investigated,
    const std::vector<NavEdgeFailure> &failedCrossings)
{
    std::set<RegionId> visited(visitedRegions.begin(), visitedRegions.end());
    std::map<RegionId, DerivedFrontier> openByDest;
    std::map<RegionId, DerivedFrontier> blockedByDest;
    for (size_t i = 0; i < boundaries.size(); ++i)
    {
        const Boundary &boundary = boundaries[i];
        if (visited.find(boundary.source) == visited.end())
            continue;
        if (visited.find(boundary.destination) != visited.end())
            continue;
        const bool open = (boundary.traversable || boundary.jumpable)
            && !crossingFailed(failedCrossings, boundary.id,
                               boundary.geometrySignature);
        if (open)
        {
            DerivedFrontier &frontier = openByDest[boundary.destination];
            frontier.destination = boundary.destination;
            frontier.kind = kFrontierOpen;
            frontier.candidates.push_back(boundary);
            continue;
        }
        if (investigatedNow(investigated, boundary.id, boundary.source,
                            boundary.destination,
                            boundary.geometrySignature))
            continue;
        DerivedFrontier &frontier = blockedByDest[boundary.destination];
        frontier.destination = boundary.destination;
        frontier.kind = kFrontierBlocked;
        frontier.candidates.push_back(boundary);
    }
    std::vector<DerivedFrontier> result;
    for (std::map<RegionId, DerivedFrontier>::iterator it = openByDest.begin();
         it != openByDest.end(); ++it)
        result.push_back(it->second);
    for (std::map<RegionId, DerivedFrontier>::iterator it = blockedByDest.begin();
         it != blockedByDest.end(); ++it)
    {
        if (openByDest.find(it->first) != openByDest.end())
            continue;
        result.push_back(it->second);
    }
    return result;
}

int selectFrontierIndex(const std::vector<DerivedFrontier> &frontiers,
                        RegionId currentRegion, const int *hops, int hopCount)
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
            if (frontier.candidates[c].source == currentRegion)
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
        return "EXECUTE_AFFORDANCE";
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
    // A world-changing action and the newly enabled physical continuation
    // form one causal plan. Once that successor is executable, retain plan
    // ownership until it is consumed; an unrelated pickup discovered before
    // the action must not make the actor turn away from the opening it just
    // created.
    if (opportunity.continuation)
        return -1;
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
    // A causal successor also breaks ties within its execution class.
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
