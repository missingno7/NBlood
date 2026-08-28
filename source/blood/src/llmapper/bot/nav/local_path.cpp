#include "local_path.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>

namespace nav {

using semantic::Loop;
using semantic::Polygon;
using semantic::Segment;
using semantic::Vec2;

namespace {

uint64_t mixHash(uint64_t value, uint64_t item)
{
    value ^= item + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
    return value;
}

struct Direction
{
    double x = 0;
    double y = 0;
};

Direction unitFrom(const Vec2 &from, const Vec2 &to)
{
    const double dx = double(to.x) - from.x;
    const double dy = double(to.y) - from.y;
    const double length = std::sqrt(dx * dx + dy * dy);
    if (length < 1e-9)
        return { 0.0, 0.0 };
    return { dx / length, dy / length };
}

bool lyingOn(const Vec2 &point, const Segment &segment)
{
    return semantic::pointSegmentDistanceSquared(point, segment.from,
                                                 segment.to) <= 1;
}

void boundaryOf(const Polygon &shape, std::vector<Segment> &out)
{
    auto addLoop = [&](const Loop &loop) {
        for (size_t i = 0; i < loop.size(); ++i)
            out.push_back({ loop[i], loop[(i + 1) % loop.size()] });
    };
    addLoop(shape.outer);
    for (const Loop &hole : shape.holes)
        addLoop(hole);
}

} // namespace

uint64_t localSignature(const semantic::Region &region,
                        const std::vector<Segment> &openings, int radius)
{
    uint64_t hash = 1469598103934665603ULL;
    hash = mixHash(hash, semantic::shapeKey(region.footprint));
    for (const Loop &hole : region.holes)
        hash = mixHash(hash, semantic::shapeKey(hole));
    for (const Segment &barrier : region.barriers)
    {
        hash = mixHash(hash, uint64_t(uint32_t(barrier.from.x)));
        hash = mixHash(hash, uint64_t(uint32_t(barrier.from.y)));
        hash = mixHash(hash, uint64_t(uint32_t(barrier.to.x)));
        hash = mixHash(hash, uint64_t(uint32_t(barrier.to.y)));
    }
    for (const Segment &opening : openings)
    {
        hash = mixHash(hash, uint64_t(uint32_t(opening.from.x)));
        hash = mixHash(hash, uint64_t(uint32_t(opening.from.y)));
        hash = mixHash(hash, uint64_t(uint32_t(opening.to.x)));
        hash = mixHash(hash, uint64_t(uint32_t(opening.to.y)));
    }
    return mixHash(hash, uint64_t(uint32_t(radius)));
}

void openingsOf(const semantic::SemanticWorld &world,
                semantic::RegionId region, const std::vector<char> &usable,
                std::vector<Segment> &out)
{
    out.clear();
    for (const semantic::SpatialRelation &relation : world.relations())
    {
        if (!relation.exists || relation.blocked || relation.id
                == semantic::kNoId)
            continue;
        // A way out of this region, and only that. Ways between two spaces
        // are not the same in both directions: a ledge a body steps down
        // off is a way out of the high side and a wall to the low one, and
        // reading it as an opening from below erases the wall a body
        // walking there would actually run into. Every shared boundary is
        // described from both sides, so nothing is lost by asking about
        // this side.
        if (relation.from != region)
            continue;
        if (relation.gateway.width() <= 0)
            continue;
        if (!usable.empty()
            && (size_t(relation.id) >= usable.size()
                || !usable[size_t(relation.id)]))
            continue;
        out.push_back({ relation.gateway.from, relation.gateway.to });
    }
}

void LocalMap::build(const Polygon &shape,
                     const std::vector<Segment> &barriers,
                     const std::vector<Segment> &openings, int radius,
                     const Vec2 *interior)
{
    m_shape = shape;
    m_openings = openings;
    m_radius = std::max(0, radius);
    m_nodes.clear();
    m_edges.clear();
    m_solid.clear();
    m_studs.clear();

    // The solid part of the boundary is the boundary minus the ways out,
    // plus any wall the region wraps around.
    std::vector<Segment> boundary;
    boundaryOf(shape, boundary);
    semantic::subtractCollinear(boundary, openings, m_solid);
    for (const Segment &barrier : barriers)
        m_solid.push_back(barrier);

    // The engine does not clip a wall as a line alone: it puts an
    // axis-aligned square of the body's own width at each end of it. So the
    // end of a wall is a square obstacle, and a route that only keeps its
    // distance from the line will still be stopped by the corner.
    for (const Segment &segment : m_solid)
    {
        for (const Vec2 &end : { segment.from, segment.to })
        {
            bool known = false;
            for (const Vec2 &stud : m_studs)
                if (stud == end)
                {
                    known = true;
                    break;
                }
            if (!known)
                m_studs.push_back(end);
        }
    }

    // Corners the body has to round. Two things have to be true of a node:
    // it clears every wall by the body's own width, and the straight legs
    // between neighbouring nodes clear them too. Around the end of a wall
    // that means standing off on a ring rather than at a point, and the ring
    // is sampled by a polygon drawn outside it -- which is why the radius is
    // divided by the cosine of half a step, and not by a chosen margin.
    std::map<std::pair<int, int>, std::vector<Direction>> incident;
    for (const Segment &segment : m_solid)
    {
        const Direction forward = unitFrom(segment.from, segment.to);
        const Direction backward = unitFrom(segment.to, segment.from);
        if (forward.x == 0.0 && forward.y == 0.0)
            continue;
        incident[{ segment.from.x, segment.from.y }].push_back(forward);
        incident[{ segment.to.x, segment.to.y }].push_back(backward);
    }
    // Where the space turns is where a route across it may have to turn, and
    // space turns at its own corners -- whether what lies past one is a wall
    // or another region this route is not going through. A region whose
    // boundary is mostly ways out has almost no wall ends, and with only
    // those to work from there is nothing to get around the pieces of the
    // world that stick into it.
    //
    // Only the corners the space wraps around, though. A route bends to get
    // past something; at a corner the space turns away from, there is
    // nothing to get past, and a node there is one more point to test every
    // other point against for the rest of the region's life. The test is the
    // shape's own: step a little way out of the corner's wedge, and ask
    // whether that is still inside.
    auto wrapsAround = [&shape](const Vec2 &before, const Vec2 &at,
                                const Vec2 &after)
    {
        const double ax = double(before.x) - at.x;
        const double ay = double(before.y) - at.y;
        const double bx = double(after.x) - at.x;
        const double by = double(after.y) - at.y;
        const double alen = std::sqrt(ax * ax + ay * ay);
        const double blen = std::sqrt(bx * bx + by * by);
        if (alen < 1.0 || blen < 1.0)
            return true;
        const double wx = ax / alen + bx / blen;
        const double wy = ay / alen + by / blen;
        const double wlen = std::sqrt(wx * wx + wy * wy);
        if (wlen < 1e-6)
            return true;   // a spur tip: the space goes all the way round it
        const Vec2 outside = { at.x - int(std::lround(wx / wlen * 4.0)),
                               at.y - int(std::lround(wy / wlen * 4.0)) };
        return semantic::pointInPolygon(shape, outside);
    };
    auto offerLoop = [&](const semantic::Loop &loop)
    {
        for (size_t i = 0; i < loop.size(); ++i)
        {
            const Vec2 &before = loop[(i + loop.size() - 1) % loop.size()];
            const Vec2 &after = loop[(i + 1) % loop.size()];
            if (wrapsAround(before, loop[i], after))
                incident[{ loop[i].x, loop[i].y }];
        }
    };
    offerLoop(shape.outer);
    for (const semantic::Loop &hole : shape.holes)
        offerLoop(hole);

    // One unit past the exact clearance: integer coordinates round, and a
    // node that lands one unit inside the wall is no node at all.
    const double clearance = double(m_radius) + 1.0;
    constexpr int kRingSamples = 8;
    const double ringStep = 2.0 * 3.14159265358979323846 / kRingSamples;
    // Far enough out to be clear of the corner's square from any direction --
    // its circumradius -- and far enough that the chords between samples are
    // clear too.
    const double ringStandoff = clearance * 1.4142135623730951
        / std::cos(ringStep * 0.5);

    std::vector<Vec2> candidates;
    auto offer = [&](const Vec2 &at, double dirX, double dirY,
                     double distance) {
        if (distance <= 0.0 || distance > 1e7)
            return;
        candidates.push_back({ int(std::lround(at.x + dirX * distance)),
                               int(std::lround(at.y + dirY * distance)) });
    };
    for (const auto &entry : incident)
    {
        const Vec2 corner = { entry.first.first, entry.first.second };
        const std::vector<Direction> &ways = entry.second;
        for (int sample = 0; sample < kRingSamples; ++sample)
        {
            const double angle = ringStep * sample;
            offer(corner, std::cos(angle), std::sin(angle), ringStandoff);
        }
        // A corner between two walls has an exact place to stand: where the
        // two walls' offset lines meet. For a sharp corner that is much
        // further out than the ring, and it is the only node out there.
        if (ways.size() == 2)
        {
            const double sx = ways[0].x + ways[1].x;
            const double sy = ways[0].y + ways[1].y;
            const double length = std::sqrt(sx * sx + sy * sy);
            const double half = std::sqrt(std::max(0.0,
                1.0 - (length * 0.5) * (length * 0.5)));
            if (length > 1e-6 && half > 1e-3)
            {
                const double distance = clearance / half;
                const double dirX = -sx / length;
                const double dirY = -sy / length;
                offer(corner, dirX, dirY, distance);
                // And either side of it, so the corner can be approached
                // from either wall without the leg clipping the other.
                const double turn = std::cos(ringStep * 0.5);
                const double lift = std::sin(ringStep * 0.5);
                offer(corner, dirX * turn - dirY * lift,
                      dirX * lift + dirY * turn, distance);
                offer(corner, dirX * turn + dirY * lift,
                      -dirX * lift + dirY * turn, distance);
            }
        }
    }
    candidates.push_back(semantic::representativePoint(shape));

    // And a ladder of witnesses stepping in from every corner the space
    // wraps around.
    //
    // The ring above stands off a corner by the body's own width, which is
    // where a body has to be to round a wall. It is not where a body can be:
    // a rim of floor around a pit is bounded by drops rather than walls, so
    // nothing erodes it and every bit of it holds a body -- but it is sixty
    // four units across, the ring lands outside it in every direction, and
    // the rim ends up witnessed by nothing. A region no node stands in has
    // no piece of free space this layer can name, so every way on and off it
    // is a crossing the planner is never offered. Region 80 on AGTST18 is
    // exactly this, and it is why the two rooms below it were unreachable.
    //
    // So step in along the corner's own bisector, and keep halving. Whatever
    // the space is wide, one of these lands in it; where the corner is made
    // of walls rather than drops, free() throws all of them away, which is
    // the same answer as before. Nothing here decides how wide is wide
    // enough -- the erosion does.
    auto ladder = [&](const semantic::Loop &loop)
    {
        for (size_t i = 0; i < loop.size(); ++i)
        {
            const Vec2 &at = loop[i];
            const Vec2 &before = loop[(i + loop.size() - 1) % loop.size()];
            const Vec2 &after = loop[(i + 1) % loop.size()];
            const double ax = double(before.x) - at.x;
            const double ay = double(before.y) - at.y;
            const double bx = double(after.x) - at.x;
            const double by = double(after.y) - at.y;
            const double alen = std::sqrt(ax * ax + ay * ay);
            const double blen = std::sqrt(bx * bx + by * by);
            if (alen < 1.0 || blen < 1.0)
                continue;
            double wx = ax / alen + bx / blen;
            double wy = ay / alen + by / blen;
            double wlen = std::sqrt(wx * wx + wy * wy);
            if (wlen < 1e-6)
            {
                // A spur tip: the two edges double back, so the way in is
                // square to them.
                wx = -(ay / alen);
                wy = ax / alen;
                wlen = 1.0;
            }
            wx /= wlen;
            wy /= wlen;
            for (double step = clearance; step >= 2.0; step *= 0.5)
                offer(at, wx, wy, step);
        }
    };
    ladder(shape.outer);
    for (const semantic::Loop &hole : shape.holes)
        ladder(hole);

    // A point on a way out is in two spaces at once, and a body driven at it
    // comes to rest on whichever side it likes. That is the whole point at
    // the end of a leg, where crossing is what is wanted. It is useless on
    // the way across this region: making for it leaves, and then the route
    // across is worked out again from the other side, which makes for a
    // point back in here. Somewhere to make for on the way has to be here.
    auto onAWayOut = [&](const Vec2 &at)
    {
        for (const Segment &opening : openings)
            if (semantic::pointSegmentDistanceSquared(at, opening.from,
                                                      opening.to) == 0)
                return true;
        return false;
    };

    for (const Vec2 &candidate : candidates)
    {
        if (!free(candidate) || onAWayOut(candidate))
            continue;
        bool duplicate = false;
        for (const Vec2 &known : m_nodes)
            if (known == candidate)
            {
                duplicate = true;
                break;
            }
        if (!duplicate)
            m_nodes.push_back(candidate);
    }

    // And one at the point the region calls its middle.
    //
    // Every node above is a corner the body has to go round, which is the
    // right set for finding a way past things and the wrong set for saying
    // where the body can be. A region with nothing to go round has none of
    // them -- a plain room, a strip of floor the width of a wall, a rim
    // whose whole boundary is a drop -- and then the free space it plainly
    // has is witnessed by nothing, so it holds no Place, so every way in or
    // out of it is a crossing the planner never hears about. Thirteen of
    // them on one map, into rooms the engine was standing the body in at the
    // time.
    //
    // The middle is the region's own interior point, so it is inside the
    // outline and outside every hole by construction. It joins nothing that
    // was separate: it is only ever added where the body fits.
    if (interior && free(*interior))
    {
        bool duplicate = false;
        for (const Vec2 &known : m_nodes)
            if (known == *interior)
            {
                duplicate = true;
                break;
            }
        if (!duplicate)
            m_nodes.push_back(*interior);
    }

    m_edges.assign(m_nodes.size(), {});
    for (size_t i = 0; i < m_nodes.size(); ++i)
        for (size_t j = i + 1; j < m_nodes.size(); ++j)
            if (visible(m_nodes[i], m_nodes[j]))
            {
                m_edges[i].push_back(uint32_t(j));
                m_edges[j].push_back(uint32_t(i));
            }

    // Label the pieces of free space. Two nodes are in the same piece when a
    // body of this width can get from one to the other without leaving the
    // region.
    m_label.assign(m_nodes.size(), -1);
    m_components = 0;
    std::vector<uint32_t> pending;
    for (size_t seed = 0; seed < m_nodes.size(); ++seed)
    {
        if (m_label[seed] >= 0)
            continue;
        const int mark = m_components++;
        pending.clear();
        pending.push_back(uint32_t(seed));
        m_label[seed] = mark;
        while (!pending.empty())
        {
            const uint32_t node = pending.back();
            pending.pop_back();
            for (uint32_t next : m_edges[node])
                if (m_label[next] < 0)
                {
                    m_label[next] = mark;
                    pending.push_back(next);
                }
        }
    }
    m_signature = 0;
}

bool LocalMap::clearOfStuds(const Vec2 &at) const
{
    for (const Vec2 &stud : m_studs)
        if (semantic::pointInBox(at, stud, m_radius))
            return false;
    return true;
}

int64_t LocalMap::clearanceSquared(const Vec2 &at) const
{
    int64_t best = std::numeric_limits<int64_t>::max();
    for (const Segment &segment : m_solid)
        best = std::min(best, semantic::pointSegmentDistanceSquared(at,
            segment.from, segment.to));
    return best;
}

bool LocalMap::free(const Vec2 &at) const
{
    if (!semantic::pointInPolygon(m_shape, at))
        return false;
    if (!clearOfStuds(at))
        return false;
    return clearanceSquared(at) >= int64_t(m_radius) * m_radius;
}

bool LocalMap::nearlyInside(const Vec2 &at) const
{
    if (semantic::pointInPolygon(m_shape, at))
        return true;
    const int64_t reach = int64_t(m_radius) * m_radius;
    auto near = [&](const semantic::Loop &loop)
    {
        for (size_t i = 0; i < loop.size(); ++i)
            if (semantic::pointSegmentDistanceSquared(at, loop[i],
                    loop[(i + 1) % loop.size()]) <= reach)
                return true;
        return false;
    };
    if (near(m_shape.outer))
        return true;
    for (const semantic::Loop &hole : m_shape.holes)
        if (near(hole))
            return true;
    return false;
}

bool LocalMap::visible(const Vec2 &a, const Vec2 &b) const
{
    const int64_t needed = int64_t(m_radius) * m_radius;
    for (const Segment &segment : m_solid)
    {
        const int64_t distance = semantic::segmentDistanceSquared(a, b,
            segment.from, segment.to);
        if (distance >= needed)
            continue;
        // The body is closer to this wall than its own width, which is only
        // acceptable where it already is: the closest approach happens at an
        // end of the leg, and that end is a place the body is standing.
        const int64_t atA = semantic::pointSegmentDistanceSquared(a,
            segment.from, segment.to);
        const int64_t atB = semantic::pointSegmentDistanceSquared(b,
            segment.from, segment.to);
        if (atA <= distance + 1 || atB <= distance + 1)
            continue;
        return false;
    }
    // And the squares the engine puts at the ends of every wall.
    for (const Vec2 &stud : m_studs)
    {
        if (!semantic::segmentEntersBox(a, b, stud, m_radius))
            continue;
        if (semantic::pointInBox(a, stud, m_radius)
            || semantic::pointInBox(b, stud, m_radius))
            continue; // already there; going is the only way out
        return false;
    }
    // A leg across this region has to be across this region. An end of it
    // somewhere else is not a leg at all, and the answer is no rather than
    // yes: saying yes hands back a straight line through whatever lies
    // between, and the body drives into another room, arrives somewhere the
    // goal is not, and is sent back -- which is a body taking turns at a
    // doorway until the level ends.
    //
    // A hair outside still counts, because the engine leaves bodies standing
    // on boundaries, and a hair is the body's own width.
    if (!nearlyInside(a) || !nearlyInside(b))
        return false;
    // Meeting a way out at all is leaving through it, not just crossing it
    // at an angle.
    //
    // A doorway is a line, and a leg can run straight down that line without
    // ever "properly intersecting" it -- overlapping is not crossing. So a
    // route was allowed to travel the length of a doorway, in and out along
    // the boundary between two rooms. Worse, only a point exactly on that
    // line could make the leg: from thirty units off, the same leg does cross
    // the doorway and is refused. So the route insisted on a waypoint the
    // body had to stand exactly on, the body could not stand exactly
    // anywhere, and it rocked back and forth beside it until the goal was
    // given up on. Narrow places are where this bites, because that is where
    // the only route runs along a boundary.
    for (const Segment &opening : m_openings)
    {
        if (semantic::segmentDistanceSquared(a, b, opening.from,
                                             opening.to) != 0)
            continue;
        if (lyingOn(a, opening) || lyingOn(b, opening))
            continue;
        return false; // it leaves through this one
    }
    return true;
}

void LocalMap::componentsAt(const Vec2 &at, std::vector<int> &out) const
{
    out.clear();
    for (size_t node = 0; node < m_nodes.size(); ++node)
    {
        // Which pieces this point is in, not which nodes it can see. A node
        // in a piece already accounted for cannot add one, so it is not
        // worth asking about -- and asking costs a clearance test against
        // every wall and every wall end in the region.
        //
        // This is the difference between one such test and two hundred and
        // fifty of them per call, on a call made for every crossing, every
        // stance of every action, and once a tick for the actor. Almost
        // every region is one piece of space, and the first node settles it.
        const int mark = m_label[node];
        bool known = false;
        for (int seen : out)
            if (seen == mark)
            {
                known = true;
                break;
            }
        if (known)
            continue;
        if (!visible(at, m_nodes[node]))
            continue;
        out.push_back(mark);
        if (int(out.size()) >= m_components)
            return;   // there are no more pieces to be in
    }
}

int LocalMap::componentNear(const Vec2 &at) const
{
    int answer = -1;
    int64_t best = 0;
    bool bestVisible = false;
    for (size_t node = 0; node < m_nodes.size(); ++node)
    {
        const int64_t dx = int64_t(m_nodes[node].x) - at.x;
        const int64_t dy = int64_t(m_nodes[node].y) - at.y;
        const int64_t span = dx * dx + dy * dy;
        // One this point can actually walk to beats a nearer one it cannot,
        // because the piece it belongs to is the piece the body is in.
        const bool seen = visible(at, m_nodes[node]);
        if (answer >= 0 && bestVisible && !seen)
            continue;
        if (answer >= 0 && (seen == bestVisible) && span >= best)
            continue;
        answer = m_label[node];
        best = span;
        bestVisible = seen;
    }
    return answer;
}

void LocalMap::visibleFrom(const Vec2 &at, std::vector<Vec2> &out) const
{
    out.clear();
    std::vector<std::pair<int64_t, size_t>> ranked;
    for (size_t node = 0; node < m_nodes.size(); ++node)
    {
        if (!visible(at, m_nodes[node]))
            continue;
        const int64_t dx = int64_t(m_nodes[node].x) - at.x;
        const int64_t dy = int64_t(m_nodes[node].y) - at.y;
        ranked.push_back({ dx * dx + dy * dy, node });
    }
    std::sort(ranked.begin(), ranked.end());
    for (const auto &entry : ranked)
        out.push_back(m_nodes[entry.second]);
}

bool LocalMap::path(const Vec2 &from, const Vec2 &to,
                    std::vector<Vec2> &out) const
{
    out.clear();
    if (visible(from, to))
    {
        out.push_back(to);
        return true;
    }
    const size_t count = m_nodes.size();
    const size_t start = count;
    const size_t goal = count + 1;
    std::vector<Vec2> points = m_nodes;
    points.push_back(from);
    points.push_back(to);

    std::vector<std::vector<uint32_t>> edges(count + 2);
    for (size_t i = 0; i < count; ++i)
    {
        edges[i] = m_edges[i];
        if (visible(m_nodes[i], from))
        {
            edges[i].push_back(uint32_t(start));
            edges[start].push_back(uint32_t(i));
        }
        if (visible(m_nodes[i], to))
        {
            edges[i].push_back(uint32_t(goal));
            edges[goal].push_back(uint32_t(i));
        }
    }

    const double infinity = std::numeric_limits<double>::max() / 4.0;
    std::vector<double> distance(count + 2, infinity);
    std::vector<size_t> parent(count + 2, count + 2);
    using Step = std::pair<double, size_t>;
    std::priority_queue<Step, std::vector<Step>, std::greater<Step>> queue;
    distance[start] = 0.0;
    queue.push({ 0.0, start });
    while (!queue.empty())
    {
        const Step top = queue.top();
        queue.pop();
        if (top.first > distance[top.second])
            continue;
        if (top.second == goal)
            break;
        for (uint32_t next : edges[top.second])
        {
            const double step = double(semantic::planarDistance(
                points[top.second], points[next]));
            if (top.first + step >= distance[next])
                continue;
            distance[next] = top.first + step;
            parent[next] = top.second;
            queue.push({ distance[next], next });
        }
    }
    if (distance[goal] >= infinity)
        return false;
    std::vector<Vec2> reversed;
    for (size_t node = goal; node != start; node = parent[node])
    {
        reversed.push_back(points[node]);
        if (parent[node] == count + 2)
            return false;
    }
    out.assign(reversed.rbegin(), reversed.rend());
    return true;
}

const LocalMap &Navigator::mapFor(const semantic::Region &region,
                                  const std::vector<Segment> &openings,
                                  int radius)
{
    const uint64_t signature = localSignature(region, openings, radius);
    for (Entry &entry : m_entries)
        if (entry.id == region.id)
        {
            if (entry.signature != signature)
            {
                entry.map.build(region.shape(), region.barriers, openings,
                                radius, &region.interior);
                entry.signature = signature;
                ++entry.generation;
                ++entry.rebuilds;
                ++m_builds;
            }
            return entry.map;
        }
    m_entries.push_back(Entry());
    Entry &entry = m_entries.back();
    entry.id = region.id;
    entry.signature = signature;
    entry.generation = 1;
    entry.map.build(region.shape(), region.barriers, openings, radius,
                    &region.interior);
    ++m_builds;
    return entry.map;
}

int Navigator::generation(semantic::RegionId region) const
{
    for (const Entry &entry : m_entries)
        if (entry.id == region)
            return entry.generation;
    return -1;
}

void Navigator::clear()
{
    m_entries.clear();
}

} // namespace nav
