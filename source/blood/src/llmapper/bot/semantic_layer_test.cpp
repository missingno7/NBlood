//-------------------------------------------------------------------------
// Spatial architecture and layering tests.
//
// This translation unit includes the semantic world, the terrain builder,
// the local-path layer, the traversal model, the planner and the executor --
// and nothing else. It is built with no Blood or Build include path at all,
// so if any of those ever reaches for the engine, the build of this file
// fails.
//
// The spatial cases below are the ones that decide whether the abstraction
// leaks. They check what the model says, not only how many of them there
// are: the same physical space authored two different ways has to come out
// with the same footprints and the same gateways, and a thing standing in a
// room has to be a thing standing in a room.
//-------------------------------------------------------------------------
#include "exec/bot_executor.h"
#include "nav/local_path.h"
#include "planner/bot_planner.h"
#include "semantic/semantic_world.h"
#include "terrain/terrain_model.h"
#include "traversal/traversal_model.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <set>
#include <string>

using semantic::Loop;
using semantic::Region;
using semantic::Segment;
using semantic::SemanticWorld;
using semantic::SpatialRelation;
using semantic::Vec2;
using semantic::WorldDelta;
using semantic::kNoId;

namespace {

constexpr int kFloor = 0;
constexpr int kCeiling = -4096; // Blood z grows downward
constexpr int kBodyRadius = 64;

terrain::SupportFace rectangle(int x1, int y1, int x2, int y2, uint64_t tag,
                               int floorZ = kFloor)
{
    terrain::SupportFace face;
    face.outline = { { x1, y1 }, { x2, y1 }, { x2, y2 }, { x1, y2 } };
    face.support = semantic::flatPlane(floorZ);
    face.ceiling = semantic::flatPlane(kCeiling);
    face.supportTag = tag;
    face.provenance = tag;
    return face;
}

Loop box(int x1, int y1, int x2, int y2)
{
    return { { x1, y1 }, { x2, y1 }, { x2, y2 }, { x1, y2 } };
}

terrain::Seam seam(size_t left, size_t right, int x1, int y1, int x2, int y2)
{
    terrain::Seam value;
    value.left = left;
    value.right = right;
    value.from = { x1, y1 };
    value.to = { x2, y2 };
    return value;
}

// A physics authority that says yes to walking anywhere flat and continuous.
// The point of the spatial tests is the shape of the world, so the actor is
// held constant and trivial.
class FlatOracle : public traversal::PhysicsOracle
{
public:
    FlatOracle()
    {
        m_profile.radius = kBodyRadius;
        m_profile.revision = 1;
    }

    void setRevision(uint64_t revision) { m_profile.revision = revision; }
    void setStepLimit(int limit) { m_step = limit; }
    int calls() const { return m_calls; }

    const traversal::ActorProfile &profile() const override
    {
        return m_profile;
    }
    bool canStand(const Region &region, const semantic::Vec2 &at) const
                  override
    {
        return region.exists && semantic::pointInPolygon(region.shape(), at);
    }

    bool canTraverse(traversal::Mode mode, const SpatialRelation &relation,
                     const Region &from, const Region &to,
                     const semantic::Vec2 *, semantic::Vec2 &crossing,
                     semantic::Vec2 &arrival,
                     semantic::Vec2 &departure) const override
    {
        ++m_calls;
        crossing = relation.gateway.midpoint();
        arrival = to.interior;
        departure = from.interior;
        if (!from.exists || !to.exists || relation.blocked)
            return false;
        if (mode == traversal::Mode::Walk)
            return relation.gateway.width() > 0
                && relation.verticalStep <= m_step
                && relation.verticalStep >= -m_step;
        if (mode == traversal::Mode::Drop)
            return relation.verticalStep > m_step;
        return false;
    }

private:
    traversal::ActorProfile m_profile;
    int m_step = 0;
    mutable int m_calls = 0;
};

// Everything a storage boundary could leak into: how many places there are,
// what shape each of them is, where the ways between them are, and what can
// be reached. Counts alone would hide a partition that moved.
struct Shape
{
    size_t regions = 0;
    size_t relations = 0;
    size_t reachable = 0;
    std::set<uint64_t> footprints;
    std::set<std::string> gateways;

    bool operator==(const Shape &other) const
    {
        return regions == other.regions && relations == other.relations
            && reachable == other.reachable
            && footprints == other.footprints && gateways == other.gateways;
    }
};

std::string gatewayText(const semantic::Gateway &gateway)
{
    // Direction-independent: the same opening described from either side is
    // the same opening.
    char text[96];
    const bool forward = gateway.from.x != gateway.to.x
        ? gateway.from.x < gateway.to.x : gateway.from.y <= gateway.to.y;
    const Vec2 low = forward ? gateway.from : gateway.to;
    const Vec2 high = forward ? gateway.to : gateway.from;
    std::snprintf(text, sizeof(text), "%d,%d-%d,%d", low.x, low.y, high.x,
                  high.y);
    return text;
}

void publish(const terrain::BuildResult &built, WorldDelta &delta)
{
    for (size_t i = 0; i < built.regions.size(); ++i)
    {
        Region region = built.regions[i].region;
        region.id = semantic::RegionId(i);
        delta.regions.push_back(region);
    }
    for (size_t i = 0; i < built.relations.size(); ++i)
    {
        SpatialRelation relation = built.relations[i];
        relation.id = semantic::RelationId(i);
        delta.relations.push_back(relation);
    }
    delta.actor.region = 0;
    delta.actor.alive = true;
}

Shape measure(const std::vector<terrain::SupportFace> &faces,
              const std::vector<terrain::Seam> &seams)
{
    terrain::BuildResult built;
    terrain::build(faces, seams, built);

    WorldDelta delta;
    publish(built, delta);

    SemanticWorld world;
    world.apply(delta);
    FlatOracle oracle;
    traversal::TraversalModel traversal;
    traversal.setExecutableModes(traversal::modeBit(traversal::Mode::Walk));
    traversal.update(world, oracle);

    std::vector<semantic::RegionId> reachable;
    traversal.reachableFrom(0, reachable);

    Shape shape;
    shape.regions = world.liveRegionCount();
    shape.relations = world.liveRelationCount();
    shape.reachable = reachable.size();
    for (const Region &region : world.regions())
        if (region.exists)
            shape.footprints.insert(semantic::shapeKey(region.footprint));
    for (const SpatialRelation &relation : world.relations())
        if (relation.exists)
            shape.gateways.insert(gatewayText(relation.gateway));
    return shape;
}

// ---------------------------------------------------------------- shapes

void oneSectorCorridor(std::vector<terrain::SupportFace> &faces,
                       std::vector<terrain::Seam> &seams)
{
    faces.clear();
    seams.clear();
    faces.push_back(rectangle(0, 0, 8192, 1024, 100));
}

void fragmentedCorridor(std::vector<terrain::SupportFace> &faces,
                        std::vector<terrain::Seam> &seams, int pieces)
{
    faces.clear();
    seams.clear();
    const int span = 8192 / pieces;
    for (int i = 0; i < pieces; ++i)
        faces.push_back(rectangle(i * span, 0, (i + 1) * span, 1024,
                                  100 + uint64_t(i)));
    for (int i = 0; i + 1 < pieces; ++i)
        seams.push_back(seam(size_t(i), size_t(i + 1), (i + 1) * span, 0,
                             (i + 1) * span, 1024));
}

// ------------------------------------------------------------- the tests

void testTessellationIsInvariant()
{
    // The same corridor authored one, four and eight ways. Not just the same
    // number of regions: the same footprints, and the same gateways.
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    oneSectorCorridor(faces, seams);
    const Shape single = measure(faces, seams);
    fragmentedCorridor(faces, seams, 4);
    const Shape four = measure(faces, seams);
    fragmentedCorridor(faces, seams, 8);
    const Shape eight = measure(faces, seams);

    std::printf("  tessellation: 1 sector -> %u regions, 4 -> %u, 8 -> %u;"
                " footprints identical = %d\n",
                unsigned(single.regions), unsigned(four.regions),
                unsigned(eight.regions),
                single.footprints == eight.footprints ? 1 : 0);
    assert(single.regions == 1);
    assert(single == four);
    assert(single == eight);
}

void testAuthoringDirectionIsInvariant()
{
    // The same corridor with extra vertices left on its walls where an
    // author's internal walls met them, and listed the other way round. A
    // vertex that lies between its neighbours describes nothing.
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    oneSectorCorridor(faces, seams);
    const Shape plain = measure(faces, seams);

    faces.clear();
    seams.clear();
    terrain::SupportFace face = rectangle(0, 0, 8192, 1024, 100);
    face.outline = { { 0, 1024 },  { 4096, 1024 }, { 8192, 1024 },
                     { 8192, 0 },  { 5000, 0 },    { 2048, 0 },
                     { 0, 0 } };
    faces.push_back(face);
    const Shape tessellated = measure(faces, seams);

    std::printf("  authoring: 4 corners -> %u regions, 7 corners wound the"
                " other way -> %u\n", unsigned(plain.regions),
                unsigned(tessellated.regions));
    assert(plain == tessellated);
}

void testOneConcaveSpaceIsOneRegion()
{
    // One U-shaped piece of floor. Getting from one arm to the other is a
    // real navigation problem and no part of it is a different kind of
    // place, so it is one Region and the problem belongs to the local path.
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    terrain::SupportFace face = rectangle(0, 0, 1, 1, 100);
    face.outline = { { 0, 0 },       { 4096, 0 },    { 4096, 1024 },
                     { 1024, 1024 }, { 1024, 3072 }, { 4096, 3072 },
                     { 4096, 4096 }, { 0, 4096 } };
    faces.push_back(face);
    const Shape shape = measure(faces, seams);

    std::printf("  concave: one U-shaped space -> %u region(s), %u"
                " relation(s)\n", unsigned(shape.regions),
                unsigned(shape.relations));
    assert(shape.regions == 1);
    assert(shape.relations == 0);

    // And the body can still get from one arm to the other, around the
    // middle, without ever touching it.
    terrain::BuildResult built;
    terrain::build(faces, seams, built);
    nav::LocalMap map;
    map.build(built.regions[0].region.shape(), {}, {}, kBodyRadius);
    std::vector<Vec2> path;
    const Vec2 start = { 3072, 512 };
    const Vec2 finish = { 3072, 3584 };
    const bool found = map.path(start, finish, path);
    double length = 0.0;
    Vec2 previous = start;
    for (const Vec2 &point : path)
    {
        length += semantic::planarDistance(previous, point);
        previous = point;
    }
    std::printf("  concave: local path around the middle = %u leg(s),"
                " %.0f units (straight line is %d)\n",
                unsigned(path.size()), length,
                semantic::planarDistance(start, finish));
    assert(found);
    assert(path.size() >= 2);
    assert(length > double(semantic::planarDistance(start, finish)) * 1.4);
}

void testRoomWithObstaclesIsOneRegion()
{
    // A room with three crates in it is a room with three crates in it.
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    terrain::SupportFace room = rectangle(0, 0, 4096, 4096, 100);
    room.obstacles.push_back(box(800, 800, 1300, 1300));
    room.obstacles.push_back(box(1800, 1800, 2300, 2300));
    room.obstacles.push_back(box(2800, 2800, 3300, 3300));
    faces.push_back(room);

    terrain::BuildResult built;
    terrain::build(faces, seams, built);
    std::printf("  obstacles: one room + 3 crates -> %u region(s), %u"
                " hole(s), %u relation(s)\n",
                unsigned(built.regions.size()),
                unsigned(built.regions[0].region.holes.size()),
                unsigned(built.relations.size()));
    assert(built.regions.size() == 1);
    assert(built.regions[0].region.holes.size() == 3);
    assert(built.relations.empty());

    // The crates are things to walk past. The route to the far corner has to
    // exist, and a body cannot stand inside one.
    nav::LocalMap map;
    map.build(built.regions[0].region.shape(), {}, {}, kBodyRadius);
    std::vector<Vec2> path;
    assert(map.path({ 200, 200 }, { 3900, 3900 }, path));
    assert(!path.empty());
    assert(!map.free({ 1000, 1000 }));
    assert(map.free({ 200, 200 }));
    std::printf("  obstacles: route to the far corner = %u leg(s), and the"
                " inside of a crate is not standable\n",
                unsigned(path.size()));
}

void testWallSpurStaysInsideItsRegion()
{
    // The AGTST6 case in miniature: a sharp wall spur poking into one
    // continuous space. It is an obstacle, not a boundary -- so there is one
    // Region, no interior gateway, and no way for a route to be planned
    // through the tip.
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    terrain::SupportFace field = rectangle(0, 0, 1, 1, 100);
    field.outline = { { 0, 0 },    { 8192, 0 },    { 8192, 8192 },
                      { 0, 8192 }, { 0, 5096 },    { 6000, 4096 },
                      { 0, 3096 } };
    faces.push_back(field);

    terrain::BuildResult built;
    terrain::build(faces, seams, built);
    std::printf("  spur: one space with a sharp spur in it -> %u region(s),"
                " %u relation(s)\n", unsigned(built.regions.size()),
                unsigned(built.relations.size()));
    assert(built.regions.size() == 1);
    assert(built.relations.empty());

    nav::LocalMap map;
    map.build(built.regions[0].region.shape(), {}, {}, kBodyRadius);
    std::vector<Vec2> path;
    const bool found = map.path({ 1000, 1500 }, { 1000, 6500 }, path);
    assert(found);
    // Round the tip, not through it. Every leg is checked against the spur
    // by the map itself; what is asserted here is that the route actually
    // goes past the end of it.
    int furthest = 0;
    for (const Vec2 &point : path)
        furthest = std::max(furthest, point.x);
    std::printf("  spur: route from below to above reaches x=%d (the tip is"
                " at x=6000) in %u leg(s)\n", furthest,
                unsigned(path.size()));
    assert(path.size() >= 2);
    assert(furthest > 6000);
    // The tip itself is not a place a body of this width can be.
    assert(!map.free({ 5990, 4096 }));
}

void testTwoDoorwaysAreTwoWays()
{
    // Two rooms at different heights, divided by a wall with two separate
    // openings in it. Two ways is two relations; the wall between them is
    // not a third.
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    faces.push_back(rectangle(0, 0, 4096, 2048, 100, 0));
    faces.push_back(rectangle(0, 2048, 4096, 4096, 200, -256));
    terrain::Seam leftWall = seam(0, 1, 0, 2048, 1024, 2048);
    leftWall.solid = true;
    terrain::Seam middleWall = seam(0, 1, 2048, 2048, 3072, 2048);
    middleWall.solid = true;
    seams.push_back(leftWall);
    seams.push_back(seam(0, 1, 1024, 2048, 2048, 2048));
    seams.push_back(middleWall);
    seams.push_back(seam(0, 1, 3072, 2048, 4096, 2048));

    terrain::BuildResult built;
    terrain::build(faces, seams, built);
    std::set<std::string> gateways;
    for (const SpatialRelation &relation : built.relations)
        gateways.insert(gatewayText(relation.gateway));
    std::printf("  doorways: two openings in one wall -> %u region(s), %u"
                " relation(s), %u distinct gateway(s)\n",
                unsigned(built.regions.size()),
                unsigned(built.relations.size()), unsigned(gateways.size()));
    assert(built.regions.size() == 2);
    assert(gateways.size() == 2);
    assert(built.relations.size() == 4); // two ways, both directions
}

void testAnInternalWallStaysAWall()
{
    // Three pieces of floor. A and C are divided by a wall; both open onto
    // B. The space is continuous, so it is one Region -- and the wall is
    // still there, inside it, as something to walk around.
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    faces.push_back(rectangle(0, 0, 1024, 1024, 100));       // A
    faces.push_back(rectangle(1024, 0, 2048, 1024, 200));    // C
    faces.push_back(rectangle(0, 1024, 2048, 2048, 300));    // B
    terrain::Seam divider = seam(0, 1, 1024, 0, 1024, 1024);
    divider.solid = true;
    seams.push_back(divider);
    seams.push_back(seam(0, 2, 0, 1024, 1024, 1024));
    seams.push_back(seam(1, 2, 1024, 1024, 2048, 1024));

    terrain::BuildResult built;
    terrain::build(faces, seams, built);
    const Region &region = built.regions[0].region;
    const long long area = semantic::signedDoubleArea(region.footprint) / 2;
    std::printf("  internal wall: %d space(s), %u region(s), %u wall(s)"
                " inside it, floor area %lld\n", built.clusters,
                unsigned(built.regions.size()),
                unsigned(region.barriers.size()), area);
    assert(built.clusters == 1);
    assert(built.regions.size() == 1);
    assert(region.barriers.size() == 1);
    assert(built.relations.empty());
    assert(area == 1024 * 1024 * 4);

    // Getting from A to C means going round through B, and the wall is what
    // makes that true.
    nav::LocalMap map;
    map.build(region.shape(), region.barriers, {}, kBodyRadius);
    std::vector<Vec2> path;
    assert(map.path({ 512, 300 }, { 1536, 300 }, path));
    int highest = 0;
    for (const Vec2 &point : path)
        highest = std::max(highest, point.y);
    std::printf("  internal wall: A to C goes up to y=%d (the wall ends at"
                " y=1024) in %u leg(s)\n", highest, unsigned(path.size()));
    assert(path.size() >= 2);
    assert(highest > 1024);
}

void testMeaningfulSeamsSurvive()
{
    // Same two rectangles, same seam, but the floors are at different
    // heights: this seam is a step, and dissolving it would be wrong.
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    faces.push_back(rectangle(0, 0, 1024, 1024, 100, 0));
    faces.push_back(rectangle(1024, 0, 2048, 1024, 200, -512));
    seams.push_back(seam(0, 1, 1024, 0, 1024, 1024));
    const Shape shape = measure(faces, seams);

    std::printf("  step: two heights across one seam -> %u regions\n",
                unsigned(shape.regions));
    assert(shape.regions == 2);
}

void testClearanceClassSplitsAndSmallChangesDoNot()
{
    // A ceiling a little lower is the same place. A ceiling low enough to
    // change what can happen there is not.
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    faces.push_back(rectangle(0, 0, 1024, 1024, 100));
    terrain::SupportFace lower = rectangle(1024, 0, 2048, 1024, 200);
    lower.ceiling = semantic::flatPlane(kCeiling + 64);
    faces.push_back(lower);
    seams.push_back(seam(0, 1, 1024, 0, 1024, 1024));
    const Shape same = measure(faces, seams);

    faces[1].clearance = semantic::ClearanceClass::Crouching;
    const Shape crouch = measure(faces, seams);

    std::printf("  clearance: a ceiling 64 units lower -> %u region(s); a"
                " crouch class boundary -> %u\n", unsigned(same.regions),
                unsigned(crouch.regions));
    assert(same.regions == 1);
    assert(crouch.regions == 2);
}

void testStackedBridgesOverOneRoom()
{
    // Two plank bridges at different heights over the same room floor, each
    // made of three separate objects. Nine surfaces over one footprint: the
    // question is whether the model keeps them apart by what holds a body up
    // rather than by where things are in plan.
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    faces.push_back(rectangle(0, 0, 6144, 4096, 900, 0));          // the room
    for (int i = 0; i < 3; ++i)                                     // low bridge
        faces.push_back(rectangle(i * 2048, 1024, (i + 1) * 2048,
                                  2048, 100 + uint64_t(i), -8192));
    for (int i = 0; i < 3; ++i)                                     // high one
        faces.push_back(rectangle(i * 2048, 1536, (i + 1) * 2048,
                                  2560, 200 + uint64_t(i), -20480));

    terrain::BuildResult built;
    terrain::build(faces, seams, built);
    WorldDelta delta;
    publish(built, delta);
    SemanticWorld world;
    world.apply(delta);
    FlatOracle oracle;
    traversal::TraversalModel traversal;
    traversal.setExecutableModes(traversal::modeBit(traversal::Mode::Walk));
    traversal.update(world, oracle);

    // Every plank is its own place, because every plank is its own thing to
    // stand on. Nothing here was merged for being at the same height or
    // separated for overlapping in plan.
    assert(world.liveRegionCount() == 7);
    std::set<int> heights;
    for (const Region &region : world.regions())
        if (region.exists)
            heights.insert(region.anchor().z);
    std::printf("  stacked bridges: 7 surfaces over one room -> %u region(s)"
                " at %u distinct heights\n",
                unsigned(world.liveRegionCount()), unsigned(heights.size()));
    assert(heights.size() == 3);

    // Along a bridge you walk; between bridges, and down to the floor, you
    // do not -- the flat oracle refuses every step, and these are all steps.
    std::vector<semantic::RegionId> reachable;
    traversal.reachableFrom(1, reachable);
    std::printf("  stacked bridges: from one plank of the low bridge,"
                " %u place(s) reachable by walking\n",
                unsigned(reachable.size()));
    assert(reachable.size() == 3);   // its own bridge, and only its own

    // And the model still knows the ones it cannot walk to are there.
    int stacked = 0;
    for (const SpatialRelation &relation : world.relations())
        if (relation.exists && relation.gateway.width() == 0)
            ++stacked;
    std::printf("  stacked bridges: %d relation(s) between surfaces that"
                " share ground rather than an edge\n", stacked);
    assert(stacked > 0);
}

void testStackedSupportsStaySeparate()
{
    // A surface directly above another one over the same ground. Two
    // different places, and walking is not how you get between them.
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    faces.push_back(rectangle(0, 0, 2048, 2048, 100, 0));
    faces.push_back(rectangle(512, 512, 1536, 1536, 200, -2048));
    faces.push_back(rectangle(768, 768, 1280, 1280, 300, -3072));
    const Shape shape = measure(faces, seams);

    std::printf("  stacked: 3 surfaces over one footprint -> %u regions,"
                " reachable from the bottom = %u\n",
                unsigned(shape.regions), unsigned(shape.reachable));
    assert(shape.regions == 3);
    assert(shape.reachable == 1); // the flat oracle refuses every step
}

void testCrossedSupportsDoNotConnect()
{
    // Two paths that cross in plan but at different heights. They are
    // spatially related, and they are not walk-connected.
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    faces.push_back(rectangle(0, 1024, 4096, 2048, 100, 0));
    faces.push_back(rectangle(1024, 0, 2048, 4096, 200, -2048));
    const Shape shape = measure(faces, seams);

    std::printf("  crossed: %u regions, %u relations, reachable = %u\n",
                unsigned(shape.regions), unsigned(shape.relations),
                unsigned(shape.reachable));
    assert(shape.relations >= 2); // the crossing is known about
    assert(shape.reachable == 1); // and it is not a connection
}

void testAffordancesDoNotPartitionTerrain()
{
    // A pickup in the middle of a room, and a switch on its wall. Both are
    // things that can be done in that space. Neither is a place.
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    faces.push_back(rectangle(0, 0, 4096, 4096, 100));
    const Shape bare = measure(faces, seams);

    terrain::BuildResult built;
    terrain::build(faces, seams, built);
    WorldDelta delta;
    publish(built, delta);
    semantic::Affordance pickup;
    pickup.id = 0;
    pickup.action = semantic::ActionKind::Collect;
    pickup.target = { 2048, 2048, 0 };
    pickup.domain.push_back({ 0, { 2048, 2048, 0 } });
    pickup.observed = true;
    pickup.executable = true;
    delta.affordances.push_back(pickup);
    semantic::Affordance button;
    button.id = 1;
    button.action = semantic::ActionKind::Use;
    button.target = { 4096, 1024, -1024 };
    button.domain.push_back({ 0, { 3500, 1024, 0 } });
    button.observed = true;
    button.executable = true;
    delta.affordances.push_back(button);

    SemanticWorld world;
    world.apply(delta);
    std::set<uint64_t> footprints;
    for (const Region &region : world.regions())
        if (region.exists)
            footprints.insert(semantic::shapeKey(region.footprint));

    std::printf("  affordances: room -> %u region(s) bare, %u with a pickup"
                " and a button; footprints unchanged = %d\n",
                unsigned(bare.regions), unsigned(world.liveRegionCount()),
                footprints == bare.footprints ? 1 : 0);
    assert(world.liveRegionCount() == bare.regions);
    assert(world.liveRelationCount() == bare.relations);
    assert(footprints == bare.footprints);
    assert(world.affordances().size() == 2);
}

void testWorldSurvivesActorChange()
{
    // The Jumping Boots invariant. Changing the actor re-derives what is
    // traversable and touches nothing about the world.
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    faces.push_back(rectangle(0, 0, 1024, 1024, 100, 0));
    faces.push_back(rectangle(1024, 0, 2048, 1024, 200, -512));
    seams.push_back(seam(0, 1, 1024, 0, 1024, 1024));

    terrain::BuildResult built;
    terrain::build(faces, seams, built);
    WorldDelta delta;
    publish(built, delta);
    SemanticWorld world;
    world.apply(delta);

    FlatOracle oracle;
    traversal::TraversalModel traversal;
    traversal.setExecutableModes(traversal::modeBit(traversal::Mode::Walk));
    traversal.update(world, oracle);
    std::vector<semantic::RegionId> reachable;
    traversal.reachableFrom(0, reachable);
    assert(reachable.size() == 1); // a step this actor cannot take

    const size_t regionsBefore = world.liveRegionCount();
    const size_t relationsBefore = world.liveRelationCount();
    const uint64_t keyBefore = built.regions[1].key;

    // New boots: the actor changes, the world does not.
    oracle.setStepLimit(1024);
    oracle.setRevision(2);
    traversal.update(world, oracle);
    traversal.reachableFrom(0, reachable);

    std::printf("  boots: reachable %u -> %u with regions %u -> %u\n",
                1u, unsigned(reachable.size()), unsigned(regionsBefore),
                unsigned(world.liveRegionCount()));
    assert(reachable.size() == 2);
    assert(world.liveRegionCount() == regionsBefore);
    assert(world.liveRelationCount() == relationsBefore);

    terrain::BuildResult again;
    terrain::build(faces, seams, again);
    assert(again.regions[1].key == keyBefore); // identity is shape-derived

    // And an unchanged actor over an unchanged world costs nothing.
    const int before = oracle.calls();
    traversal.update(world, oracle);
    std::printf("  incremental: re-deriving an unchanged world made %d"
                " physics queries\n", oracle.calls() - before);
    assert(oracle.calls() == before);
}

void testPossibleIsNotExecutable()
{
    // A drop this actor can physically make, that this bot has no executor
    // for. It stays known, it is reported, and it is never selected.
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    faces.push_back(rectangle(0, 0, 1024, 1024, 100, 0));
    faces.push_back(rectangle(1024, 0, 2048, 1024, 200, 2048));
    seams.push_back(seam(0, 1, 1024, 0, 1024, 1024));

    terrain::BuildResult built;
    terrain::build(faces, seams, built);
    WorldDelta delta;
    publish(built, delta);
    SemanticWorld world;
    world.apply(delta);

    FlatOracle oracle;
    traversal::TraversalModel traversal;
    traversal.setExecutableModes(traversal::modeBit(traversal::Mode::Walk));
    traversal.update(world, oracle);

    std::printf("  capability: %d physically real transition(s) this bot"
                " cannot drive\n", traversal.knownButUnexecutable());
    assert(traversal.knownButUnexecutable() > 0);

    // A way it cannot drive is still a way it can go and look at, once.
    planner::Decision decision = planner::choose(world, traversal);
    assert(decision.intent == planner::Intent::Approach);
    assert(decision.diagnosis.possibleButUnexecutable > 0);
    for (const SpatialRelation &relation : world.relations())
        world.noteInspected(relation.id);
    decision = planner::choose(world, traversal);
    std::printf("  capability: after looking at it, intent is %s\n",
                planner::intentName(decision.intent));
    assert(decision.intent == planner::Intent::None);
}

void testFrontierIsGatewaysAndExecutionIsLocal()
{
    // Two rooms joined by a doorway, with a crate in the first one sitting
    // between the actor and the door. The planner picks the way it has not
    // been through; the executor works out how to get past the crate.
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    terrain::SupportFace first = rectangle(0, 0, 4096, 4096, 100);
    first.obstacles.push_back(box(1800, 1500, 2400, 2600));
    faces.push_back(first);
    terrain::SupportFace second = rectangle(4096, 0, 8192, 4096, 200);
    second.hazard.harmful = true;
    faces.push_back(second);
    seams.push_back(seam(0, 1, 4096, 1792, 4096, 2304));

    terrain::BuildResult built;
    terrain::build(faces, seams, built);
    WorldDelta delta;
    publish(built, delta);
    delta.actor.position = { 400, 2048, 0 };
    delta.regions[0].occupied = true; // the actor is standing in it
    SemanticWorld world;
    world.apply(delta);

    FlatOracle oracle;
    traversal::TraversalModel traversal;
    traversal.setExecutableModes(traversal::modeBit(traversal::Mode::Walk));
    traversal.update(world, oracle);

    planner::Decision decision = planner::choose(world, traversal);
    assert(decision.intent == planner::Intent::GoTo);
    assert(decision.destination == 1);
    assert(decision.diagnosis.uncrossedGateways > 0);

    exec::Executor executor;
    executor.begin(decision, world);
    semantic::MotorCommand command;
    const exec::Outcome outcome = executor.tick(world, traversal,
                                                oracle.profile(), command);
    std::printf("  execution: aiming at (%d,%d) with a crate at"
                " x=1800-2400, y=1500-2600 in the way\n",
                command.moveToward.x, command.moveToward.y);
    assert(outcome == exec::Outcome::Running);
    assert(command.move);
    assert(!command.act);
    // The straight line to the doorway runs through the crate, so the first
    // steering point cannot be the doorway itself.
    assert(!(command.moveToward.x == 4096 && command.moveToward.y == 2048));
    // And it has to be a place a body can stand.
    nav::LocalMap map;
    std::vector<Segment> openings;
    nav::openingsOf(world, 0, {}, openings);
    map.build(world.region(0)->shape(), world.region(0)->barriers, openings,
              oracle.profile().radius);
    assert(map.free({ command.moveToward.x, command.moveToward.y }));

    // Having gone through, that gateway stops being a frontier.
    WorldDelta moved = delta;
    moved.actor.region = 1;
    moved.actor.position = { 6000, 2048, 0 };
    for (Region &region : moved.regions)
        region.occupied = true;
    world.apply(moved);
    traversal.update(world, oracle);
    decision = planner::choose(world, traversal);
    std::printf("  frontier: after crossing, %u gateway(s) uncrossed, intent"
                " is %s\n", unsigned(world.uncrossedRelationCount()),
                planner::intentName(decision.intent));
    assert(world.uncrossedRelationCount() == 0);
    assert(decision.intent == planner::Intent::None);
}

void testAffordancePolicy()
{
    SemanticWorld world;
    WorldDelta delta;
    Region only;
    only.id = 0;
    only.footprint = { { 0, 0 }, { 1024, 0 }, { 1024, 1024 }, { 0, 1024 } };
    only.interior = { 512, 512 };
    only.observed = true;
    only.occupied = true;
    delta.regions.push_back(only);
    semantic::Affordance affordance;
    affordance.id = 0;
    affordance.target = { 512, 1100, 0 };
    affordance.domain.push_back({ 0, { 512, 900, 0 } });
    affordance.observed = true;
    affordance.executable = true;
    delta.affordances.push_back(affordance);
    delta.actor.region = 0;
    delta.actor.alive = true;
    world.apply(delta);

    FlatOracle oracle;
    traversal::TraversalModel traversal;
    traversal.setExecutableModes(traversal::modeBit(traversal::Mode::Walk));
    traversal.update(world, oracle);

    planner::Decision decision = planner::choose(world, traversal);
    assert(decision.intent == planner::Intent::ExecuteAffordance);

    // Once tried, it is not offered again for having worked.
    //
    // "It moved the world last time" is a reason to have done it, not a
    // reason to do it now, and a planner that accepts it walks back across
    // the level to press a switch whose effect it no longer needs. Doing
    // something again has to be answered by what is shut now.
    world.beginAttempt(0, 4);
    world.closeAttempt(4);
    decision = planner::choose(world, traversal);
    assert(decision.intent == planner::Intent::None);
    assert(decision.diagnosis.affordancesAttemptedInertly == 1);

    world.beginAttempt(0, 4);
    world.closeAttempt(7);   // it widened the world this time
    assert(world.affordance(0)->lastAttemptOpenedWay);
    decision = planner::choose(world, traversal);
    assert(decision.intent == planner::Intent::None);   // still not a reason

    // What is a reason: a way that is shut, and this thing known to move it.
    // The knowing is not assumed -- it is written down by watching the way
    // change while the thing was being acted on.
    WorldDelta shut;
    shut.regions.push_back(only);
    Region beyond;
    beyond.id = 1;
    beyond.footprint = { { 1024, 0 }, { 2048, 0 }, { 2048, 1024 },
                         { 1024, 1024 } };
    beyond.interior = { 1536, 512 };
    beyond.observed = true;
    shut.regions.push_back(beyond);
    SpatialRelation way;
    way.id = 0;
    way.from = 0;
    way.to = 1;
    way.gateway = { { 1024, 256 }, { 1024, 768 } };
    way.blocked = true;
    shut.relations.push_back(way);
    shut.affordances.push_back(affordance);
    shut.actor = world.actor();
    world.apply(shut);

    // Acting on it was seen to change whether this way can be gone through.
    // Whether it can is a question about a body, so the layer that has one
    // says so; nothing is inferred here from the way merely having moved.
    world.beginAttempt(0, 4);
    WorldDelta opened = shut;
    opened.relations[0].blocked = false;
    world.apply(opened);
    world.noteAffects(0, 0);
    world.closeAttempt(4);
    assert(world.affects(0, 0));

    world.apply(shut);   // and now it is shut again
    traversal.update(world, oracle);
    decision = planner::choose(world, traversal);
    std::printf("  affordance: tried -> not offered for having worked ->"
                " offered again for a way it is known to move\n");
    assert(decision.intent == planner::Intent::ExecuteAffordance);
}

// An actor that can step down as far as it likes and up only so far. Which
// is what every actor is: no engine has ever had a symmetric step.
class SteppingOracle : public traversal::PhysicsOracle
{
public:
    explicit SteppingOracle(int rise) : m_rise(rise)
    {
        m_profile.radius = 128;
        m_profile.standHeight = 1024;
        m_profile.stepUp = rise;
        m_profile.revision = 7;
    }
    const traversal::ActorProfile &profile() const override
    {
        return m_profile;
    }
    bool canStand(const Region &region, const semantic::Vec2 &at) const
                  override
    {
        return region.exists && semantic::pointInPolygon(region.shape(), at);
    }
    bool canTraverse(traversal::Mode mode, const SpatialRelation &relation,
                     const Region &from, const Region &to,
                     const semantic::Vec2 *, semantic::Vec2 &crossing,
                     semantic::Vec2 &arrival,
                     semantic::Vec2 &departure) const override
    {
        crossing = relation.gateway.midpoint();
        arrival = to.interior;
        departure = from.interior;
        if (!from.exists || !to.exists || relation.blocked)
            return false;
        if (mode != traversal::Mode::Walk || relation.gateway.width() <= 0)
            return false;
        // A rise the actor cannot make. Descending the same seam is fine,
        // which is the whole point: the two directions are different facts.
        return -relation.verticalStep <= m_rise;
    }

private:
    traversal::ActorProfile m_profile;
    int m_rise = 0;
};

// A step down that is not a step up.
void testOpeningsAreDirected()
{
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    faces.push_back(rectangle(0, 0, 2048, 2048, 100, 0));
    terrain::SupportFace step = rectangle(2048, 0, 4096, 2048, 200, -4096);
    // A raised floor with the ceiling raised over it: a step, not a space
    // with no headroom.
    step.ceiling = semantic::flatPlane(kCeiling - 4096);
    faces.push_back(step);
    seams.push_back(seam(0, 1, 2048, 0, 2048, 2048));

    terrain::BuildResult built;
    terrain::build(faces, seams, built);
    WorldDelta delta;
    publish(built, delta);
    SemanticWorld world;
    world.apply(delta);

    SteppingOracle oracle(1024);
    traversal::TraversalModel traversal;
    traversal.setExecutableModes(traversal::modeBit(traversal::Mode::Walk));
    traversal.update(world, oracle);

    int downhill = 0;
    int uphill = 0;
    for (const SpatialRelation &relation : world.relations())
    {
        if (!relation.exists)
            continue;
        if (!traversal.derived(relation.id, traversal::Mode::Walk))
            continue;
        if (relation.verticalStep > 0)
            ++downhill;
        else if (relation.verticalStep < 0)
            ++uphill;
    }
    std::printf("  directed: %d way(s) down the step, %d back up\n",
                downhill, uphill);
    assert(downhill > 0);
    assert(uphill == 0);

    // The walking is directed, and so is what counts as a wall.
    //
    // An opening is somewhere the engine does not clip *this body*, which is
    // not the same for both sides of a step: a rise past what the body steps
    // over is a ledge from above and a wall from below. Erode the low room's
    // floor against it and a pose can never land inside the clip line the
    // engine keeps there; call it open and poses do land there, routes lead
    // to them, and the body pushes into an invisible wall for as long as the
    // level lasts.
    //
    // This does not erase the room above -- see the ledge test. A room's
    // free space comes from its own ways out, and stepping down is not
    // something the engine stops.
    std::vector<semantic::Segment> high;
    std::vector<semantic::Segment> low;
    // Region 1 is the higher floor: z counts downwards.
    traversal.openingsFor(world, 1, high);
    traversal.openingsFor(world, 0, low);
    std::printf("  directed: ways out of the high side %u, out of the low"
                " side %u -- the same step, a ledge and a wall\n",
                unsigned(high.size()), unsigned(low.size()));
    assert(!high.empty());   // stepping down is not stopped
    assert(low.empty());     // stepping up past the allowance is a wall
}

// A crossing belongs to the boundary it crossed.
void testCrossingsBelongToTheirOwnGateway()
{
    // One room, and beside it a step up with two ways onto it a short walk
    // apart. A stance beside one of them is a stance beside the other, as
    // far as distance can tell.
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    faces.push_back(rectangle(0, 0, 4096, 2048, 100, 0));
    faces.push_back(rectangle(4096, 0, 6144, 2048, 200, -256));
    terrain::Seam middle = seam(0, 1, 4096, 512, 4096, 1536);
    middle.solid = true;
    seams.push_back(seam(0, 1, 4096, 0, 4096, 512));
    seams.push_back(middle);
    seams.push_back(seam(0, 1, 4096, 1536, 4096, 2048));

    terrain::BuildResult built;
    terrain::build(faces, seams, built);
    WorldDelta delta;
    publish(built, delta);
    SemanticWorld world;
    world.apply(delta);

    SteppingOracle oracle(1024);
    traversal::TraversalModel traversal;
    traversal.setExecutableModes(traversal::modeBit(traversal::Mode::Walk));
    traversal.update(world, oracle);

    int checked = 0;
    for (const SpatialRelation &relation : world.relations())
    {
        if (!relation.exists)
            continue;
        semantic::Vec2 crossing;
        semantic::Vec2 arrival;
        if (!traversal.crossingFor(relation.id, crossing, arrival))
            continue;
        // On this gateway, not on the one next to it. Two ways out of one
        // room a short walk apart are exactly how a crossing gets checked
        // from a stance beside the wrong one.
        const int64_t off = semantic::pointSegmentDistanceSquared(crossing,
            relation.gateway.from, relation.gateway.to);
        assert(off == 0);
        ++checked;
    }
    std::printf("  provenance: %d crossing(s), every one on its own"
                " gateway\n", checked);
    assert(checked > 0);
}

// One answer to "can that be done from here".
void testPlannerAndExecutorAgreeOnReach()
{
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    faces.push_back(rectangle(0, 0, 4096, 4096, 100, 0));
    faces.push_back(rectangle(4096, 0, 8192, 4096, 200, -256));
    seams.push_back(seam(0, 1, 4096, 0, 4096, 4096));

    terrain::BuildResult built;
    terrain::build(faces, seams, built);
    WorldDelta delta;
    semantic::Affordance button;
    button.id = 0;
    button.exists = true;
    button.observed = true;
    button.executable = true;
    button.action = semantic::ActionKind::Use;
    button.domain.push_back({ 1, { 6144, 2048, 200 } });
    delta.affordances.push_back(button);
    publish(built, delta);
    SemanticWorld world;
    world.apply(delta);

    SteppingOracle oracle(1024);
    traversal::TraversalModel traversal;
    traversal.setExecutableModes(traversal::modeBit(traversal::Mode::Walk));
    traversal.update(world, oracle);

    // Exploring comes first, so give it nothing left to explore: walk the
    // actor through and back, which is how the world learns a way has been
    // used.
    WorldDelta there;
    there.actor.region = 1;
    there.actor.alive = true;
    world.apply(there);
    WorldDelta back;
    back.actor.region = 0;
    back.actor.alive = true;
    world.apply(back);
    traversal.update(world, oracle);
    const planner::Decision decision = planner::choose(world, traversal);
    assert(decision.intent == planner::Intent::ExecuteAffordance);
    // Whatever the planner priced, the executor must be able to walk. One
    // search answers both, so this cannot come apart without someone adding
    // a second one back.
    std::vector<semantic::RegionId> path;
    std::vector<semantic::RelationId> via;
    const bool reachable = traversal.optionRoute(world.actor().region,
        decision.affordance, decision.option, path, via);
    std::printf("  authority: planner chose action %u option %u; the way"
                " there is %u region(s)\n", unsigned(decision.affordance),
                unsigned(decision.option), unsigned(path.size()));
    assert(reachable);
    assert(!path.empty());

    // And when the executor finds a leg it cannot drive, the planner stops
    // offering it -- rather than the two of them disagreeing for ever.
    for (const SpatialRelation &relation : world.relations())
        if (relation.exists && relation.from == world.actor().region)
            traversal.refuseCrossing(relation.id);
    int cost = 0;
    const bool priced = traversal.optionCost(world.actor().region,
        decision.affordance, decision.option, cost);
    std::printf("  authority: after the crossing was refused, the planner"
                " prices it as %s\n", priced ? "reachable" : "out of reach");
    assert(!priced);
}

// A room above a ledge too high to climb is still a room.
//
// What a body can walk across and where a body can be are different
// questions. If the second is answered with the first, a step nobody can
// climb deletes the space above it, and the model reports a room that plainly
// exists as nowhere at all.
void testAHighLedgeDoesNotEraseTheRoomAboveIt()
{
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    faces.push_back(rectangle(0, 0, 4096, 4096, 100, 0));
    terrain::SupportFace above = rectangle(4096, 0, 8192, 4096, 200, -10000);
    above.ceiling = semantic::flatPlane(kCeiling - 10000);
    faces.push_back(above);
    seams.push_back(seam(0, 1, 4096, 0, 4096, 4096));

    terrain::BuildResult built;
    terrain::build(faces, seams, built);
    WorldDelta delta;
    publish(built, delta);
    SemanticWorld world;
    world.apply(delta);

    SteppingOracle oracle(6659);
    traversal::TraversalModel traversal;
    traversal.setExecutableModes(traversal::modeBit(traversal::Mode::Walk));
    traversal.update(world, oracle);

    const int below = traversal.piecesOf(0);
    const int high = traversal.piecesOf(1);
    std::printf("  ledge: room below has %d place(s), room above has %d\n",
                below, high);
    assert(below > 0);
    assert(high > 0);   // it is still a room

    int up = 0;
    int down = 0;
    for (const SpatialRelation &relation : world.relations())
    {
        if (!relation.exists
            || !traversal.derived(relation.id, traversal::Mode::Walk))
            continue;
        if (relation.from == 0)
            ++up;
        else
            ++down;
    }
    std::printf("  ledge: walk up %d, walk down %d -- each derived on its"
                " own\n", up, down);
    assert(up == 0);    // 10000 is past what this body steps
    assert(down > 0);   // and walking off it is another question entirely
}

// A gap too low to stand in does not empty the room behind it.
void testALowDoorwayDoesNotEraseTheRoomBehindIt()
{
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    faces.push_back(rectangle(0, 0, 4096, 4096, 100, 0));
    // A short stretch between them: the same floor, a ceiling a body cannot
    // stand under. It is its own space because its headroom differs.
    terrain::SupportFace gap = rectangle(4096, 0, 5120, 4096, 200, 0);
    gap.ceiling = semantic::flatPlane(-512);
    gap.clearance = semantic::ClearanceClass::Crouching;
    faces.push_back(gap);
    faces.push_back(rectangle(5120, 0, 9216, 4096, 300, 0));
    seams.push_back(seam(0, 1, 4096, 0, 4096, 4096));
    seams.push_back(seam(1, 2, 5120, 0, 5120, 4096));

    terrain::BuildResult built;
    terrain::build(faces, seams, built);
    WorldDelta delta;
    publish(built, delta);
    SemanticWorld world;
    world.apply(delta);

    SteppingOracle oracle(6659);
    traversal::TraversalModel traversal;
    traversal.setExecutableModes(traversal::modeBit(traversal::Mode::Walk));
    traversal.update(world, oracle);
    std::printf("  headroom: %u space(s); places %d / %d / %d\n",
                unsigned(world.liveRegionCount()), traversal.piecesOf(0),
                traversal.piecesOf(1), traversal.piecesOf(2));
    // Both rooms are rooms. The crawlspace between them is not somewhere
    // this body can be, and that is a fact about the crawlspace alone.
    assert(traversal.piecesOf(0) > 0);
    assert(traversal.piecesOf(2) > 0);
}

// Every stance is a witness of somewhere the body can be.
//
// A stance that belongs to no Place is a body standing in a room the model
// says has no room in it, and every way in or out of that room becomes a
// crossing the planner is never offered. The two have to agree, and the way
// they agree is that the free space says where the stances may be, never the
// other way round.
void testEveryStanceBelongsToExactlyOnePlace()
{
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    // A room, a rim of floor round a pit, and a strip as wide as a wall --
    // the three shapes whose free space a corner graph does not witness.
    faces.push_back(rectangle(0, 0, 4096, 4096, 100));
    faces.push_back(rectangle(4096, 0, 4160, 4096, 200));
    faces.push_back(rectangle(4160, 0, 8192, 4096, 300));
    seams.push_back(seam(0, 1, 4096, 0, 4096, 4096));
    seams.push_back(seam(1, 2, 4160, 0, 4160, 4096));

    terrain::BuildResult built;
    terrain::build(faces, seams, built);
    WorldDelta delta;
    publish(built, delta);
    SemanticWorld world;
    world.apply(delta);

    FlatOracle oracle;
    traversal::TraversalModel traversal;
    traversal.setExecutableModes(traversal::modeBit(traversal::Mode::Walk));
    traversal.update(world, oracle);

    int stances = 0;
    int homeless = 0;
    for (const Region &region : world.regions())
    {
        if (!region.exists || region.id == kNoId)
            continue;
        const std::vector<semantic::Vec2> *standing =
            traversal.stancesOf(region.id);
        if (!standing)
            continue;
        stances += int(standing->size());
        if (!standing->empty() && traversal.piecesOf(region.id) == 0)
            homeless += int(standing->size());
    }
    std::printf("  stances: %d witness(es), %d with no place to belong to\n",
                stances, homeless);
    assert(stances > 0);
    assert(homeless == 0);
}

// A thing standing in a doorway changes what can be walked and nothing else.
//
// Regions are the world's own spaces and gateways are the world's own
// boundaries; neither is a fact about what happens to be standing in them at
// the moment. If a curtain closing renumbered the rooms, then every crossing,
// every route and every Place would be about a world that stops existing
// whenever anything moves.
void testAMovingBlockerDoesNotChangeIdentity()
{
    std::vector<terrain::SupportFace> faces;
    // Two rooms a kerb apart, so they stay two rooms.
    faces.push_back(rectangle(0, 0, 4096, 4096, 100, kFloor));
    faces.push_back(rectangle(4096, 0, 8192, 4096, 200, kFloor - 256));

    auto worldWith = [&](uint64_t obstruction, SemanticWorld &into)
    {
        std::vector<terrain::Seam> seams;
        terrain::Seam between = seam(0, 1, 4096, 1024, 4096, 3072);
        between.obstruction = obstruction;
        seams.push_back(between);
        terrain::BuildResult built;
        terrain::build(faces, seams, built);
        WorldDelta delta;
        publish(built, delta);
        into.apply(delta);
    };

    SemanticWorld world;
    worldWith(0, world);
    std::vector<semantic::RegionId> regionsOpen;
    std::vector<semantic::RelationId> waysOpen;
    for (const Region &region : world.regions())
        if (region.exists)
            regionsOpen.push_back(region.id);
    int crossableOpen = 0;
    for (const SpatialRelation &relation : world.relations())
        if (relation.exists)
        {
            waysOpen.push_back(relation.id);
            if (!relation.blocked)
                ++crossableOpen;
        }

    // Now something stands in it. Same world, same rooms, same doorway.
    worldWith(4242, world);
    std::vector<semantic::RegionId> regionsShut;
    std::vector<semantic::RelationId> waysShut;
    for (const Region &region : world.regions())
        if (region.exists)
            regionsShut.push_back(region.id);
    int crossableShut = 0;
    for (const SpatialRelation &relation : world.relations())
        if (relation.exists)
        {
            waysShut.push_back(relation.id);
            if (!relation.blocked)
                ++crossableShut;
        }

    std::printf("  blocker: regions %u -> %u, ways %u -> %u,"
                " crossable %d -> %d\n",
                unsigned(regionsOpen.size()), unsigned(regionsShut.size()),
                unsigned(waysOpen.size()), unsigned(waysShut.size()),
                crossableOpen, crossableShut);
    assert(regionsOpen == regionsShut);   // the rooms did not move
    assert(waysOpen == waysShut);         // nor did the doorway
    assert(crossableOpen > 0);
    assert(crossableShut < crossableOpen); // only the walking changed
}

// The same moving geometry, authored two ways, is the same moving geometry.
//
// This is the test that says why a Region's identity cannot simply become
// the set of engine pieces it is made of. How a level's author chose to cut
// one physical thing into sectors is not a fact about the world, so it must
// not reach anything above the mapper -- not the number of Regions, not the
// ways between them, and not the identity of the thing that moves.
void testHowMovingGeometryIsCutUpDoesNotReachTheModel()
{
    const uint64_t mechanism = 77;

    auto worldFrom = [&](bool subdivided, int floorZ)
    {
        std::vector<terrain::SupportFace> faces;
        std::vector<terrain::Seam> seams;
        faces.push_back(rectangle(0, 0, 3072, 3072, 1, kFloor));
        if (!subdivided)
        {
            terrain::SupportFace slab = rectangle(3072, 0, 6144, 3072, 2,
                                                  floorZ);
            slab.stateTag = mechanism;
            faces.push_back(slab);
            seams.push_back(seam(0, 1, 3072, 0, 3072, 3072));
        }
        else
        {
            for (int part = 0; part < 3; ++part)
            {
                terrain::SupportFace slab = rectangle(3072 + part * 1024, 0,
                    4096 + part * 1024, 3072, uint64_t(2 + part), floorZ);
                slab.stateTag = mechanism;   // one mechanism, three pieces
                faces.push_back(slab);
            }
            seams.push_back(seam(0, 1, 3072, 0, 3072, 3072));
            seams.push_back(seam(1, 2, 4096, 0, 4096, 3072));
            seams.push_back(seam(2, 3, 5120, 0, 5120, 3072));
        }
        terrain::BuildResult built;
        terrain::build(faces, seams, built);
        return built;
    };

    const terrain::BuildResult whole = worldFrom(false, kFloor);
    const terrain::BuildResult cut = worldFrom(true, kFloor);
    std::printf("  moving tessellation: one piece -> %u region(s), three"
                " pieces -> %u region(s)\n",
                unsigned(whole.regions.size()),
                unsigned(cut.regions.size()));
    assert(whole.regions.size() == cut.regions.size());
    assert(whole.relations.size() == cut.relations.size());

    size_t movingWhole = 0;
    size_t movingCut = 0;
    for (const terrain::BuiltRegion &region : whole.regions)
        if (region.stateTag == mechanism)
            ++movingWhole;
    for (const terrain::BuiltRegion &region : cut.regions)
        if (region.stateTag == mechanism)
            ++movingCut;
    std::printf("  moving tessellation: the moving space is %u region(s)"
                " either way\n", unsigned(movingWhole));
    assert(movingWhole == 1);
    assert(movingCut == 1);

    // Now move it. The space it makes may be decomposed differently and get
    // a different RegionId; the mechanism is the same mechanism.
    const terrain::BuildResult lifted = worldFrom(false, kFloor - 8192);
    uint64_t before = 0;
    uint64_t after = 0;
    for (const terrain::BuiltRegion &region : whole.regions)
        if (region.stateTag == mechanism)
            before = region.stateTag;
    for (const terrain::BuiltRegion &region : lifted.regions)
        if (region.stateTag == mechanism)
            after = region.stateTag;
    std::printf("  moving tessellation: after moving, it is the same"
                " mechanism\n");
    assert(before != 0 && before == after);
}

} // namespace

int main()
{
    std::printf("spatial architecture\n");
    testTessellationIsInvariant();
    testAuthoringDirectionIsInvariant();
    testOneConcaveSpaceIsOneRegion();
    testRoomWithObstaclesIsOneRegion();
    testWallSpurStaysInsideItsRegion();
    testTwoDoorwaysAreTwoWays();
    testAnInternalWallStaysAWall();
    testMeaningfulSeamsSurvive();
    testClearanceClassSplitsAndSmallChangesDoNot();
    testStackedSupportsStaySeparate();
    testStackedBridgesOverOneRoom();
    testCrossedSupportsDoNotConnect();
    testAffordancesDoNotPartitionTerrain();
    std::printf("layering\n");
    testWorldSurvivesActorChange();
    testPossibleIsNotExecutable();
    testFrontierIsGatewaysAndExecutionIsLocal();
    testAffordancePolicy();
    testOpeningsAreDirected();
    testCrossingsBelongToTheirOwnGateway();
    testPlannerAndExecutorAgreeOnReach();
    testAHighLedgeDoesNotEraseTheRoomAboveIt();
    testALowDoorwayDoesNotEraseTheRoomBehindIt();
    testHowMovingGeometryIsCutUpDoesNotReachTheModel();
    testEveryStanceBelongsToExactlyOnePlace();
    testAMovingBlockerDoesNotChangeIdentity();
    return 0;
}
