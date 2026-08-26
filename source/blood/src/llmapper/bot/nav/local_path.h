//-------------------------------------------------------------------------
// Getting across one Region.
//
// A Region is one continuous piece of space; it is not convex, it has holes,
// and it may have walls inside it that it wraps around. Crossing it is
// therefore a real navigation problem, and this is where it is solved -- in
// the actor's own configuration space, which is the free space shrunk by the
// body's radius, so a route that touches a wall is not a route.
//
// Nothing above execution ever sees any of this. There are no cells here
// that could be mistaken for places, and nothing computed here is stored in
// the world.
//-------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <vector>

#include "../semantic/semantic_world.h"

namespace nav {

// The free space of one Region for one body width, as a visibility graph
// over the corners the body has to round.
class LocalMap
{
public:
    // `interior` is a point the region guarantees is inside itself. It is a
    // witness, not a sample: free space with no corner to go round would
    // otherwise be witnessed by nothing at all, and a piece of space no node
    // stands in is a piece of space this layer cannot name.
    void build(const semantic::Polygon &shape,
               const std::vector<semantic::Segment> &barriers,
               const std::vector<semantic::Segment> &openings, int radius,
               const semantic::Vec2 *interior = nullptr);

    // Steering points from `from` to `to`, excluding `from`. False when the
    // body cannot get there through this region at all -- which is a fact
    // about the body and the shape, not a failure to try hard enough.
    bool path(const semantic::Vec2 &from, const semantic::Vec2 &to,
              std::vector<semantic::Vec2> &out) const;

    // Whether a body of this width could stand here.
    bool free(const semantic::Vec2 &at) const;
    // Is this somewhere this region can be said to contain? Inside it, or
    // on its edge -- the engine regularly leaves a body standing on a
    // boundary, and a body's own width is how far off one it can be. A
    // point further out than that is somewhere else, and a leg to it is not
    // a leg across this region.
    bool nearlyInside(const semantic::Vec2 &at) const;

    // Which pieces of this region's free space a body standing here is in.
    //
    // A Region is one continuous piece of space, but a body has width, and a
    // wall that leaves a gap narrower than the body cuts the space in two for
    // that body while leaving it one space. Which pieces there are is a fact
    // about the world and the body together, so it is derived here and never
    // stored in the world.
    //
    // More than one answer means this point joins them: it can be stood in,
    // and from it both are reachable.
    void componentsAt(const semantic::Vec2 &at,
                      std::vector<int> &out) const;
    // Which piece a body standing here belongs to when the free space does
    // not describe where it is standing -- on a boundary, or hard enough
    // against a wall that nothing is a clear stride away. The nearest place
    // it could stand answers for it: a body is never further than its own
    // width from one. Saying "all of them" instead is how two pieces of a
    // region that a body cannot walk between become one, and a route then
    // crosses a wall.
    int componentNear(const semantic::Vec2 &at) const;
    int components() const { return m_components; }

    // Every place a body of this width can stand from which this point is a
    // straight walk away, nearest first. Visibility here means the leg
    // between them clears every wall by the body's own width, so whoever
    // drives it does not have to discover a wall in the middle of it.
    //
    // These are candidates, not answers. Whether the world actually holds a
    // body up at one of them is a question for the physics layer: this map
    // is flat, and a neighbour's floor can reach into the hull from a height
    // this map knows nothing about.
    void visibleFrom(const semantic::Vec2 &at,
                     std::vector<semantic::Vec2> &out) const;
    // Every place the shape offers, whether or not anything can be reached
    // from any other.
    const std::vector<semantic::Vec2> &places() const { return m_nodes; }
    // Whether the straight leg between two points clears every wall by the
    // body's own width.
    bool clearBetween(const semantic::Vec2 &a, const semantic::Vec2 &b) const
    {
        return visible(a, b);
    }

    int radius() const { return m_radius; }
    size_t corners() const { return m_nodes.size(); }
    size_t solidEdges() const { return m_solid.size(); }
    uint64_t signature() const { return m_signature; }

private:
    bool visible(const semantic::Vec2 &a, const semantic::Vec2 &b) const;
    bool clearOfStuds(const semantic::Vec2 &at) const;
    int64_t clearanceSquared(const semantic::Vec2 &at) const;

    semantic::Polygon m_shape;
    std::vector<semantic::Segment> m_solid;
    std::vector<semantic::Vec2> m_studs;
    std::vector<semantic::Segment> m_openings;
    std::vector<semantic::Vec2> m_nodes;
    std::vector<std::vector<uint32_t>> m_edges;
    std::vector<int> m_label;   // free-space component per node
    int m_components = 0;
    int m_radius = 0;
    uint64_t m_signature = 0;
};

// One local map per region and body width, kept until the shape of the
// region or the width of the body changes.
class Navigator
{
public:
    const LocalMap &mapFor(const semantic::Region &region,
                           const std::vector<semantic::Segment> &openings,
                           int radius);
    void clear();

    int builds() const { return m_builds; }
    // How many times this region's map has been made. Whoever cached an
    // answer that came out of one can tell whether it still stands.
    int generation(semantic::RegionId region) const;
    int queries() const { return m_queries; }
    void countQuery() { ++m_queries; }

    struct Entry
    {
        semantic::RegionId id = semantic::kNoId;
        uint64_t signature = 0;
        int generation = 0;
        int rebuilds = 0;
        LocalMap map;
    };

    // How many times each region's map has been thrown away and made again,
    // for finding out which region is churning and why.
    const std::vector<Entry> &entries() const { return m_entries; }

private:

    std::vector<Entry> m_entries;
    int m_builds = 0;
    int m_queries = 0;
};

// The signature of everything a local map depends on.
uint64_t localSignature(const semantic::Region &region,
                        const std::vector<semantic::Segment> &openings,
                        int radius);

// The openings out of one region, as the local map needs them: the parts of
// its boundary that are not wall *for the body being asked about*.
//
// `usable` says, per relation id, whether this actor can actually go that
// way. A ledge it cannot climb and a drop it cannot drive are walls: leaving
// them out of the solid boundary would let a route pass through them, and
// then the executor would drive the body at a cliff. An empty `usable`
// means every unblocked relation counts, which is the answer when nothing
// has been derived yet.
void openingsOf(const semantic::SemanticWorld &world,
                semantic::RegionId region, const std::vector<char> &usable,
                std::vector<semantic::Segment> &out);

} // namespace nav
