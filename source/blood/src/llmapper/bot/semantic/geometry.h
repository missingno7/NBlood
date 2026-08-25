//-------------------------------------------------------------------------
// Plane geometry for the semantic world.
//
// Engine-independent by construction: integer map coordinates, polygons with
// holes, and the predicates needed to reason about them. Nothing here knows
// what a sector is.
//
// There is deliberately no convex decomposition in this file. A piece of
// physical space is described by its own outline and the things standing in
// it; cutting it into convex pieces is a way of walking across it, and that
// belongs to whoever is doing the walking.
//-------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <vector>

namespace semantic {

struct Vec2
{
    int x = 0;
    int y = 0;

    bool operator==(const Vec2 &other) const
    {
        return x == other.x && y == other.y;
    }
    bool operator!=(const Vec2 &other) const { return !(*this == other); }
};

struct Vec3
{
    int x = 0;
    int y = 0;
    int z = 0;
};

using Loop = std::vector<Vec2>;

// A piece of boundary. Openings, internal walls and shared edges are all
// this shape, so they can be merged, subtracted and measured by one set of
// operations.
struct Segment
{
    Vec2 from;
    Vec2 to;
};

// A height field over the xy plane, written so a flat surface and a sloped
// one are the same kind of thing. Coefficients are fixed point.
constexpr int64_t kPlaneScale = 65536;

struct Plane
{
    int64_t z0 = 0;
    int64_t dzdx = 0;
    int64_t dzdy = 0;

    int zAt(int x, int y) const
    {
        return int((z0 + dzdx * int64_t(x) + dzdy * int64_t(y)) / kPlaneScale);
    }
    bool operator==(const Plane &other) const
    {
        return z0 == other.z0 && dzdx == other.dzdx && dzdy == other.dzdy;
    }
};

Plane flatPlane(int z);

// One piece of floor: its outline, and the outlines of whatever solid
// things stand inside it.
struct Polygon
{
    Loop outer;
    std::vector<Loop> holes;
};

int64_t crossOf(const Vec2 &origin, const Vec2 &a, const Vec2 &b);
int64_t signedDoubleArea(const Loop &loop);
void makeCounterClockwise(Loop &loop);
// Points that lie on the straight line between their neighbours describe
// nothing. They are where an author's internal wall met an outer one, and
// leaving them in lets that authoring decision reach everything downstream.
void removeCollinear(Loop &loop);
bool pointInLoop(const Loop &loop, const Vec2 &point);
bool pointInPolygon(const Polygon &polygon, const Vec2 &point);
Vec2 centroidOf(const Loop &loop);
int planarDistance(const Vec2 &a, const Vec2 &b);
int planarDistance(const Vec3 &a, const Vec3 &b);
Vec2 closestPointOnSegment(const Vec2 &point, const Vec2 &a, const Vec2 &b);
int64_t pointSegmentDistanceSquared(const Vec2 &point, const Vec2 &a,
                                    const Vec2 &b);
int64_t segmentDistanceSquared(const Vec2 &a1, const Vec2 &a2,
                               const Vec2 &b1, const Vec2 &b2);
bool segmentsProperlyIntersect(const Vec2 &a1, const Vec2 &a2,
                               const Vec2 &b1, const Vec2 &b2);

// Whether a segment gets inside an axis-aligned square. Corners of solid
// geometry are square, not round, so this is the shape a route has to keep
// out of near one.
bool segmentEntersBox(const Vec2 &a, const Vec2 &b, const Vec2 &centre,
                      int half);
bool pointInBox(const Vec2 &point, const Vec2 &centre, int half);

// How far a straight push from `from` toward `to` stays inside a shape.
//
// The answer is the distance to the first boundary it crosses, or the whole
// distance if it crosses none. `from` is expected to be on the boundary --
// that is what this is for -- so leaving at the very start is not leaving.
int depthInto(const Polygon &polygon, const Vec2 &from, const Vec2 &to);

// A point a little way inside a shape from a point on its boundary, along
// the inward normal of the boundary segment given, and never deeper than
// the shape actually is there.
//
// A boundary point belongs to both sides of the boundary, so it is not
// somewhere to go: a body driven at one stops on the line and ends up on
// whichever side it likes. Somewhere to go is just inside.
Vec2 justInside(const Polygon &shape, const Vec2 &edgeFrom, const Vec2 &edgeTo,
                const Vec2 &at, int inset);

// A point guaranteed to be inside the free space -- not the centroid, which
// for a concave or holed shape is regularly in a wall. Found by taking the
// widest interior span of the widest horizontal slab, so it is also a point
// with room around it.
Vec2 representativePoint(const Polygon &polygon);

// Every maximal stretch of boundary two outlines share, not just the
// longest. Two rooms joined by two separate doorways share two stretches,
// and that is two ways between them.
void sharedBoundaries(const Loop &left, const Loop &right,
                      std::vector<Segment> &out);

// Segments lying on one line and touching or overlapping are one segment.
// This is what makes a wall's authored subdivision invisible.
void mergeCollinearSegments(const std::vector<Segment> &in,
                            std::vector<Segment> &out);

// `in` minus the parts covered by `cuts`, keeping only collinear overlap.
// Used to take a region's openings out of its solid boundary.
void subtractCollinear(const std::vector<Segment> &in,
                       const std::vector<Segment> &cuts,
                       std::vector<Segment> &out);

// Whether two footprints cover any common ground.
bool loopsOverlap(const Loop &left, const Loop &right);

// Whether a segment runs into a footprint at all.
bool segmentCrossesLoop(const Vec2 &from, const Vec2 &to, const Loop &loop);

// A key that depends on the shape and nothing else: the same physical
// outline gives the same value however it was authored, drawn or reached.
uint64_t shapeKey(const Loop &loop);

} // namespace semantic
