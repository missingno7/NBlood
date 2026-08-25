#include "terrain_model.h"

#include <algorithm>
#include <map>
#include <set>

namespace terrain {

using semantic::ClearanceClass;
using semantic::Gateway;
using semantic::Loop;
using semantic::Plane;
using semantic::Polygon;
using semantic::Region;
using semantic::Segment;
using semantic::SpatialRelation;
using semantic::Vec2;

namespace {

uint64_t mixHash(uint64_t value, uint64_t item)
{
    value ^= item + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
    return value;
}

struct Bounds
{
    int minX = 0, minY = 0, maxX = 0, maxY = 0;

    bool overlaps(const Bounds &other, int margin) const
    {
        return minX - margin <= other.maxX && other.minX - margin <= maxX
            && minY - margin <= other.maxY && other.minY - margin <= maxY;
    }
};

Bounds boundsOf(const Loop &loop)
{
    Bounds bounds;
    if (loop.empty())
        return bounds;
    bounds.minX = bounds.maxX = loop[0].x;
    bounds.minY = bounds.maxY = loop[0].y;
    for (const Vec2 &point : loop)
    {
        bounds.minX = std::min(bounds.minX, point.x);
        bounds.maxX = std::max(bounds.maxX, point.x);
        bounds.minY = std::min(bounds.minY, point.y);
        bounds.maxY = std::max(bounds.maxY, point.y);
    }
    return bounds;
}

struct UnionFind
{
    std::vector<size_t> parent;

    explicit UnionFind(size_t count) : parent(count)
    {
        for (size_t i = 0; i < count; ++i)
            parent[i] = i;
    }
    size_t find(size_t node)
    {
        while (parent[node] != node)
        {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        return node;
    }
    void join(size_t left, size_t right)
    {
        const size_t a = find(left);
        const size_t b = find(right);
        if (a != b)
            parent[std::max(a, b)] = std::min(a, b);
    }
};

// A seam carries physical meaning when crossing it changes something about
// the place: the surface underfoot, how much room there is over it, what the
// ground does to you, or whether the ground can move. Otherwise it is a line
// an author drew and nothing else.
//
// The ceiling is compared only through its clearance class. A ceiling two
// units lower is not a different place; a ceiling low enough to force a
// crouch is.
bool seamIsMeaningless(const SupportFace &left, const SupportFace &right,
                       const Seam &seam)
{
    if (seam.solid)
        return false;
    if (left.clearance == ClearanceClass::None
        || right.clearance == ClearanceClass::None)
        return false;
    return left.support == right.support && left.clearance == right.clearance
        && left.hazard == right.hazard && left.stateTag == right.stateTag;
}

using DirectedEdge = std::pair<Vec2, Vec2>;

struct EdgeLess
{
    bool operator()(const DirectedEdge &left, const DirectedEdge &right) const
    {
        if (left.first.x != right.first.x)
            return left.first.x < right.first.x;
        if (left.first.y != right.first.y)
            return left.first.y < right.first.y;
        if (left.second.x != right.second.x)
            return left.second.x < right.second.x;
        return left.second.y < right.second.y;
    }
};

// The union of a group of faces, obtained by cancelling every directed edge
// that has an opposite twin. Whatever survives is the outside of the group,
// so an internal seam disappears no matter how the group was subdivided.
void unionBoundary(const std::vector<const SupportFace *> &group,
                   std::vector<Loop> &loops)
{
    // Two faces meeting along one stretch of boundary may have described it
    // with different numbers of vertices -- one long wall against two short
    // ones. Cancelling only matches identical edges, so every edge is first
    // cut at any vertex lying on it. Otherwise a shared boundary survives
    // and the union comes out with a slot through it.
    std::vector<Loop> outlines;
    std::vector<Vec2> vertices;
    for (const SupportFace *face : group)
    {
        Loop outline = face->outline;
        semantic::makeCounterClockwise(outline);
        for (const Vec2 &point : outline)
            vertices.push_back(point);
        outlines.push_back(outline);
    }
    std::sort(vertices.begin(), vertices.end(),
        [](const Vec2 &left, const Vec2 &right) {
            return left.x != right.x ? left.x < right.x : left.y < right.y;
        });
    vertices.erase(std::unique(vertices.begin(), vertices.end()),
                   vertices.end());

    std::map<DirectedEdge, int, EdgeLess> edges;
    std::vector<std::pair<int64_t, Vec2>> cuts;
    for (const Loop &outline : outlines)
        for (size_t i = 0; i < outline.size(); ++i)
        {
            const Vec2 &a = outline[i];
            const Vec2 &b = outline[(i + 1) % outline.size()];
            const int64_t dx = int64_t(b.x) - a.x;
            const int64_t dy = int64_t(b.y) - a.y;
            const int64_t span = dx * dx + dy * dy;
            if (span == 0)
                continue;
            cuts.clear();
            for (const Vec2 &point : vertices)
            {
                if (point == a || point == b)
                    continue;
                if (semantic::crossOf(a, b, point) != 0)
                    continue;
                const int64_t along = (int64_t(point.x) - a.x) * dx
                    + (int64_t(point.y) - a.y) * dy;
                if (along <= 0 || along >= span)
                    continue;
                cuts.push_back({ along, point });
            }
            std::sort(cuts.begin(), cuts.end(),
                [](const std::pair<int64_t, Vec2> &left,
                   const std::pair<int64_t, Vec2> &right) {
                    return left.first < right.first;
                });
            Vec2 previous = a;
            for (const auto &cut : cuts)
            {
                if (!(cut.second == previous))
                {
                    const DirectedEdge edge = { previous, cut.second };
                    const DirectedEdge twin = { edge.second, edge.first };
                    auto found = edges.find(twin);
                    if (found != edges.end())
                    {
                        if (--found->second == 0)
                            edges.erase(found);
                    }
                    else
                        ++edges[edge];
                }
                previous = cut.second;
            }
            if (!(previous == b))
            {
                const DirectedEdge edge = { previous, b };
                const DirectedEdge twin = { edge.second, edge.first };
                auto found = edges.find(twin);
                if (found != edges.end())
                {
                    if (--found->second == 0)
                        edges.erase(found);
                }
                else
                    ++edges[edge];
            }
        }

    std::multimap<std::pair<int, int>, Vec2> outgoing;
    for (const auto &entry : edges)
        for (int count = 0; count < entry.second; ++count)
            outgoing.insert({ { entry.first.first.x, entry.first.first.y },
                              entry.first.second });

    while (!outgoing.empty())
    {
        Loop loop;
        auto start = outgoing.begin();
        Vec2 first = { start->first.first, start->first.second };
        Vec2 current = first;
        size_t guard = outgoing.size() + 4;
        while (guard-- > 0)
        {
            auto step = outgoing.find({ current.x, current.y });
            if (step == outgoing.end())
                break;
            const Vec2 next = step->second;
            outgoing.erase(step);
            loop.push_back(current);
            current = next;
            if (current == first)
                break;
        }
        // A vertex left over from an author's internal wall meeting the
        // outer one lies on the straight line between its neighbours and
        // describes nothing. Dropping it here is what stops the subdivision
        // reaching anything downstream.
        semantic::removeCollinear(loop);
        if (loop.size() >= 3)
            loops.push_back(loop);
    }
}

} // namespace

void build(const std::vector<SupportFace> &faces,
           const std::vector<Seam> &seams, BuildResult &out)
{
    out.regions.clear();
    out.relations.clear();
    out.obstructions.clear();
    out.clusters = 0;
    out.barriers = 0;
    if (faces.empty())
        return;

    // 1. Dissolve the seams that carry no physical meaning. Nothing is held
    //    back: two pieces of floor that differ in no way a body could feel
    //    are one piece of floor, and a wall that ends up inside the result
    //    is kept as a wall inside it rather than as a reason to keep the
    //    space split.
    UnionFind clusters(faces.size());
    std::vector<const Seam *> meaningful;
    for (const Seam &seam : seams)
    {
        if (seam.left >= faces.size() || seam.right >= faces.size()
            || seam.left == seam.right)
            continue;
        if (seamIsMeaningless(faces[seam.left], faces[seam.right], seam))
            clusters.join(seam.left, seam.right);
        else
            meaningful.push_back(&seam);
    }

    std::map<size_t, std::vector<size_t>> members;
    for (size_t i = 0; i < faces.size(); ++i)
        if (faces[i].clearance != ClearanceClass::None)
            members[clusters.find(i)].push_back(i);
    out.clusters = int(members.size());

    // 2. One continuous piece of space, one Region.
    std::vector<size_t> regionCluster;
    std::vector<Bounds> regionBounds;
    std::map<size_t, size_t> regionOfCluster;
    for (const auto &entry : members)
    {
        std::vector<const SupportFace *> group;
        uint64_t supportTag = 0;
        bool firstMember = true;
        std::vector<uint64_t> provenance;
        std::vector<uint64_t> supports;
        for (size_t index : entry.second)
        {
            group.push_back(&faces[index]);
            provenance.push_back(faces[index].provenance);
            supports.push_back(faces[index].supportTag);
            if (firstMember || faces[index].supportTag < supportTag)
                supportTag = faces[index].supportTag;
            firstMember = false;
        }
        std::sort(provenance.begin(), provenance.end());
        std::sort(supports.begin(), supports.end());
        supports.erase(std::unique(supports.begin(), supports.end()),
                       supports.end());

        std::vector<Loop> loops;
        unionBoundary(group, loops);
        if (loops.empty())
            continue;

        size_t outerIndex = 0;
        int64_t bestArea = 0;
        for (size_t i = 0; i < loops.size(); ++i)
        {
            const int64_t area = std::abs(semantic::signedDoubleArea(loops[i]));
            if (i == 0 || area > bestArea)
            {
                bestArea = area;
                outerIndex = i;
            }
        }

        BuiltRegion built;
        Region &region = built.region;
        region.footprint = loops[outerIndex];
        semantic::makeCounterClockwise(region.footprint);
        for (size_t i = 0; i < loops.size(); ++i)
            if (i != outerIndex)
                region.holes.push_back(loops[i]);
        // Solid things standing in the space are holes in it. They are not
        // reasons to have two spaces.
        for (const SupportFace *face : group)
            for (const Loop &obstacle : face->obstacles)
                region.holes.push_back(obstacle);

        const SupportFace &sample = *group.front();
        region.support = sample.support;
        region.ceiling = sample.ceiling;
        region.hazard = sample.hazard;
        region.clearance = sample.clearance;
        region.supports = supports;
        region.interior = semantic::representativePoint(region.shape());
        built.key = mixHash(supportTag, semantic::shapeKey(region.footprint));
        built.provenance = provenance;

        regionOfCluster[entry.first] = out.regions.size();
        out.regions.push_back(built);
        regionCluster.push_back(entry.first);
        regionBounds.push_back(boundsOf(region.footprint));
    }

    // 3. A wall that divides part of one space without cutting it in two is
    //    an obstacle in that space. The union boundary has already cancelled
    //    it out of the outline, so it is put back explicitly -- as something
    //    to walk around, not as a second Region.
    std::map<size_t, std::vector<Segment>> internalWalls;
    std::map<std::pair<size_t, size_t>, std::vector<Segment>> openings;
    std::map<std::pair<size_t, size_t>, std::vector<Segment>> closures;
    // An opening with something standing in it. Kept apart from both: it is
    // a real opening between two spaces, and nothing goes through it now.
    std::map<std::pair<size_t, size_t>,
             std::vector<std::pair<Segment, uint64_t>>> obstructed;
    for (const Seam *seam : meaningful)
    {
        const size_t left = clusters.find(seam->left);
        const size_t right = clusters.find(seam->right);
        const Segment segment = { seam->from, seam->to };
        if (left == right)
        {
            if (seam->solid && regionOfCluster.count(left))
            {
                internalWalls[left].push_back(segment);
                ++out.barriers;
            }
            continue;
        }
        if (!regionOfCluster.count(left) || !regionOfCluster.count(right))
            continue;
        const std::pair<size_t, size_t> pair = { std::min(left, right),
                                                 std::max(left, right) };
        if (seam->solid)
            closures[pair].push_back(segment);
        else if (seam->obstruction != 0)
            obstructed[pair].push_back({ segment, seam->obstruction });
        else
            openings[pair].push_back(segment);
    }
    for (auto &entry : internalWalls)
    {
        Region &region = out.regions[regionOfCluster[entry.first]].region;
        semantic::mergeCollinearSegments(entry.second, region.barriers);
    }

    // 4. Relations. A gateway is a real seam between two distinct spaces,
    //    with an author's subdivision of it merged away first. Two separate
    //    openings between the same pair stay two, because that is two ways
    //    to go.
    auto addRelation = [&](size_t left, size_t right, const Gateway &gateway,
                           bool solid, uint64_t obstruction = 0) {
        const Region &a = out.regions[left].region;
        const Region &b = out.regions[right].region;
        const Vec2 at = gateway.midpoint();
        SpatialRelation relation;
        relation.from = semantic::RegionId(left);
        relation.to = semantic::RegionId(right);
        relation.gateway = gateway;
        relation.verticalStep = b.support.zAt(at.x, at.y)
            - a.support.zAt(at.x, at.y);
        relation.gap = 0;
        relation.clearance = std::min(a.clearanceAt(at), b.clearanceAt(at));
        // Shut is shut, whether by the world's own shape or by something
        // standing in the way. Which of the two it is decides whether the
        // spaces either side are one space, and that was settled above; here
        // both simply mean that nothing goes through as things are.
        relation.blocked = solid || obstruction != 0;
        out.relations.push_back(relation);
        out.obstructions.push_back(obstruction);
    };
    auto addPair = [&](size_t left, size_t right,
                       const std::vector<Segment> &intervals, bool solid,
                       uint64_t obstruction = 0) {
        for (const Segment &interval : intervals)
        {
            Gateway gateway;
            gateway.from = interval.from;
            gateway.to = interval.to;
            addRelation(left, right, gateway, solid, obstruction);
            Gateway reverse;
            reverse.from = interval.to;
            reverse.to = interval.from;
            addRelation(right, left, reverse, solid, obstruction);
        }
    };

    std::set<std::pair<size_t, size_t>> joined;
    for (auto &entry : openings)
    {
        std::vector<Segment> merged;
        semantic::mergeCollinearSegments(entry.second, merged);
        addPair(regionOfCluster[entry.first.first],
                regionOfCluster[entry.first.second], merged, false);
        joined.insert(entry.first);
    }
    for (auto &entry : obstructed)
    {
        for (const auto &blockedInterval : entry.second)
            addPair(regionOfCluster[entry.first.first],
                    regionOfCluster[entry.first.second],
                    { blockedInterval.first }, false, blockedInterval.second);
        joined.insert(entry.first);
    }
    for (auto &entry : closures)
    {
        if (joined.count(entry.first))
            continue;
        std::vector<Segment> merged;
        semantic::mergeCollinearSegments(entry.second, merged);
        addPair(regionOfCluster[entry.first.first],
                regionOfCluster[entry.first.second], merged, true);
        joined.insert(entry.first);
    }

    // 4b. A space a body cannot be in is not a place -- but the places on
    //     either side of it are still next to each other, with the way
    //     between them shut. A closed door is the case that matters: drop it
    //     entirely and the rooms it joins stop being neighbours, and nothing
    //     is left to go and look at.
    for (size_t face = 0; face < faces.size(); ++face)
    {
        if (faces[face].clearance != ClearanceClass::None)
            continue;
        std::map<size_t, std::vector<Segment>> sides;
        for (const Seam *seam : meaningful)
        {
            size_t other = faces.size();
            if (seam->left == face)
                other = seam->right;
            else if (seam->right == face)
                other = seam->left;
            if (other >= faces.size())
                continue;
            const size_t cluster = clusters.find(other);
            if (!regionOfCluster.count(cluster))
                continue;
            sides[regionOfCluster[cluster]].push_back({ seam->from,
                                                        seam->to });
        }
        for (auto left = sides.begin(); left != sides.end(); ++left)
        {
            auto right = left;
            for (++right; right != sides.end(); ++right)
            {
                std::vector<Segment> here;
                std::vector<Segment> there;
                semantic::mergeCollinearSegments(left->second, here);
                semantic::mergeCollinearSegments(right->second, there);
                if (here.empty() || there.empty())
                    continue;
                // Each side leaves through its own mouth of the closed
                // space, so the two directions do not share one gateway.
                Gateway out;
                out.from = here.front().from;
                out.to = here.front().to;
                Gateway back;
                back.from = there.front().from;
                back.to = there.front().to;
                addRelation(left->first, right->first, out, true);
                addRelation(right->first, left->first, back, true);
                joined.insert({ std::min(regionCluster[left->first],
                                         regionCluster[right->first]),
                                std::max(regionCluster[left->first],
                                         regionCluster[right->first]) });
            }
        }
    }

    // 5. Spaces that meet without a seam between them: surfaces standing on
    //    one another, and support that is not made of walls at all.
    for (size_t left = 0; left < out.regions.size(); ++left)
        for (size_t right = left + 1; right < out.regions.size(); ++right)
        {
            const std::pair<size_t, size_t> pair = {
                std::min(regionCluster[left], regionCluster[right]),
                std::max(regionCluster[left], regionCluster[right]) };
            if (joined.count(pair))
                continue;
            if (!regionBounds[left].overlaps(regionBounds[right], 1))
                continue;
            const Region &a = out.regions[left].region;
            const Region &b = out.regions[right].region;
            std::vector<Segment> shared;
            semantic::sharedBoundaries(a.footprint, b.footprint, shared);
            if (!shared.empty())
            {
                addPair(left, right, shared, false);
                joined.insert(pair);
                continue;
            }
            if (!semantic::loopsOverlap(a.footprint, b.footprint))
                continue;
            // Surfaces stacked over one another share ground rather than an
            // edge. Where to cross is chosen at execution time; what matters
            // here is that the two are spatially related.
            const Vec2 middle = a.footprint.size() <= b.footprint.size()
                ? a.interior : b.interior;
            Gateway gateway;
            gateway.from = middle;
            gateway.to = middle;
            addRelation(left, right, gateway, false);
            Gateway reverse;
            reverse.from = middle;
            reverse.to = middle;
            addRelation(right, left, reverse, false);
            joined.insert(pair);
        }
}

} // namespace terrain
