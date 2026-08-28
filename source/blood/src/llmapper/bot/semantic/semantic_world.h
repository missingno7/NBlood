//-------------------------------------------------------------------------
// The semantic 3D world: what physically exists.
//
// This layer says what is there and how it is arranged. It does not say what
// the actor can do with it -- that needs the actor, and the actor lives in
// the physics layer. Nothing here names a sector, a wall or a sprite: the
// mapper mints these identifiers and keeps the mapping back to itself.
//-------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <vector>

#include "geometry.h"

namespace semantic {

using RegionId = uint32_t;
using RelationId = uint32_t;
using AffordanceId = uint32_t;
// A piece of world geometry that can be in more than one configuration, kept
// apart from the space that geometry currently makes.
//
// A Region is space in one configuration; when geometry moves, the space it
// makes may legitimately be decomposed differently and get a different
// RegionId. What must survive the motion is the *thing that moved*, because
// that is what a body stands on while it carries it, and what a route means
// when it says "when this is in that configuration". So this identity is
// separate from RegionId on purpose, and neither is derived from the other.
//
// It is opaque above this line. It is interned from whatever the engine uses
// to keep track of the mechanism, and nothing outside the adapter that
// interns it may assume what that was. Nothing here means door, or lift, or
// bridge: those are words for what a configuration happens to connect, and
// they are read off the connectivity, never written into it.
using GeometryId = uint32_t;

constexpr uint32_t kNoId = 0xFFFFFFFFu;

// What a piece of ground does to whoever stands on it, as a property of the
// world rather than of the actor.
struct Hazard
{
    bool harmful = false;
    bool submerged = false;
    // The geometry here is moving at this moment: walls sweeping through the
    // space, a floor on its way somewhere. Deliberately not part of what
    // makes two pieces of ground the same ground -- a room is the same room
    // while its door is swinging, and letting this split a Region would mean
    // Region identity changing every time anything moved.
    bool shifting = false;
    // Standing here moves the body whether it asks to or not: the floor
    // itself travels. Also not part of what makes two pieces of ground the
    // same ground -- a conveyor is a floor -- and also a fact about now,
    // since it depends on whether the thing is running.
    bool carrying = false;

    bool operator==(const Hazard &other) const
    {
        return harmful == other.harmful && submerged == other.submerged;
    }
};

// How much room there is over a surface, in the only terms that change what
// can happen there. A ceiling two units lower is the same place; a ceiling
// low enough to force a crouch is not.
enum class ClearanceClass
{
    Standing,
    Crouching,
    None,
};

const char *clearanceName(ClearanceClass clearance);

// One physically continuous piece of standable space.
//
// Concave, holed and large are all normal. The outline is where the space
// ends; the holes are solid things standing inside it; the barriers are
// walls that divide part of it without cutting it in two. None of those
// makes a second Region, because none of them changes what kind of place
// this is -- they change how you walk across it, which is a different
// question asked by a different layer.
struct Region
{
    RegionId id = kNoId;
    Loop footprint;
    std::vector<Loop> holes;      // solid objects standing in the space
    std::vector<Segment> barriers; // walls inside it that it wraps around
    Vec2 interior;                // a point known to be in the free space
    // The things that hold a body up over this space, as opaque identifiers
    // minted by the mapper. Compared, never interpreted.
    //
    // This is what tells one place from another where they are stacked, and
    // it is the only honest way to ask "is the body standing in this Region":
    // the plane below says what the floor *is*, and a body standing next to
    // a step is held up by the step, not by the plane it is standing beside.
    std::vector<uint64_t> supports;
    // The stateful geometry this space is made of, when it is made of any.
    // Static ground has none. Two Regions with the same one are two
    // configurations of the same physical thing -- possibly at once, if a
    // configuration decomposes into more spaces than another did.
    GeometryId mover = kNoId;
    Plane support;
    Plane ceiling;
    Hazard hazard;
    ClearanceClass clearance = ClearanceClass::Standing;

    bool exists = true;    // the world offers this piece of space
    bool observed = false; // it has been seen, from anywhere, at any time
    bool occupied = false; // the actor has stood in it

    Polygon shape() const { return Polygon{ footprint, holes }; }

    bool holds(uint64_t support) const
    {
        for (uint64_t owner : supports)
            if (owner == support)
                return true;
        return false;
    }

    int supportZAt(const Vec2 &point) const
    {
        return support.zAt(point.x, point.y);
    }
    int clearanceAt(const Vec2 &point) const
    {
        return support.zAt(point.x, point.y) - ceiling.zAt(point.x, point.y);
    }
    Vec3 anchor() const
    {
        return { interior.x, interior.y,
                 support.zAt(interior.x, interior.y) };
    }
};

// The stretch of boundary two regions share. It is an interval, not a point:
// a concrete crossing point is chosen by the executor when it gets there,
// and how much of the interval a body can use is the physics layer's answer,
// not a property stored here.
struct Gateway
{
    Vec2 from;
    Vec2 to;

    Vec2 midpoint() const
    {
        return { (from.x + to.x) / 2, (from.y + to.y) / 2 };
    }
    int width() const { return planarDistance(from, to); }
};

// Two distinct pieces of space are next to each other in a way worth
// evaluating. This is not yet a traversal: whether a body can get across one
// depends on the body.
struct SpatialRelation
{
    RelationId id = kNoId;
    RegionId from = kNoId;
    RegionId to = kNoId;
    Gateway gateway;

    int verticalStep = 0;   // support of `to` minus support of `from`
    int gap = 0;            // horizontal separation, zero when they touch
    int clearance = 0;      // smallest free height across the crossing
    bool blocked = false;   // the world itself closes this off right now
    bool exists = true;
    // Owned by SemanticWorld; never overwritten by an incoming delta. The
    // actor has actually gone this way, which is what makes a gateway stop
    // being a frontier. `inspected` is the weaker fact for a way that
    // cannot be gone through: the actor has been to it and seen what it
    // offers, so there is nothing further to learn by standing there again.
    bool crossed = false;
    bool inspected = false;
    // The thing standing in this way, when there is one and it is a thing
    // that can be acted on. A way shut by a shape is shut; a way shut by an
    // object has something to be done about it, and this says what.
    //
    // World state, not memory: the mapper works it out afresh from what is
    // actually standing where, so when the thing moves this stops saying so
    // without anyone having to remember to forget.
    AffordanceId obstruction = kNoId;
};

enum class ActionKind
{
    Use,
    Collect,
};

const char *actionName(ActionKind action);

// One place the action can be carried out from: a pose the world accepts,
// and the piece of space that pose is in. A doorway switch can be pressed
// from either side of the door, which is two options in two regions and one
// affordance.
struct ExecutionOption
{
    RegionId region = kNoId;
    Vec3 at;
};

// One thing that can be done, wherever the possibility came from. There is
// no wall, sector or sprite variant, and no pickup variant: an affordance is
// an affordance. It lives at a place in space and is carried out from
// somewhere in its execution domain -- it never divides the terrain.
struct Affordance
{
    AffordanceId id = kNoId;
    ActionKind action = ActionKind::Use;
    Vec3 target;                  // where the thing itself is
    // Everywhere the world accepts it from. Which of them to use is a
    // question about what the actor can reach, and that is the planner's.
    std::vector<ExecutionOption> domain;
    bool exists = true;
    bool observed = false;
    bool executable = false;      // the world currently accepts the action
    // Owned by SemanticWorld; never overwritten by an incoming delta. What
    // is recorded is not that the world moved -- a switch that toggles moves
    // it every time -- but that trying this opened something up.
    int attempts = 0;
    bool lastAttemptOpenedWay = false;
    // This thing has been found standing in a way. A barrier, then, rather
    // than a switch that acts at a distance -- and a barrier that is not in
    // anything's way any more is a barrier that is already open. Pressing it
    // again is how a bot spends its afternoon closing and reopening the same
    // curtain.
    bool barrier = false;
    // The thing itself is in the middle of changing: the door is swinging,
    // the sector is on its way. Not a note that an attempt was made and not
    // a clock -- it is read off the object every tick and stops being true
    // the moment the object stops moving. Acting on something mid-change is
    // how a switch gets pressed eight times in fifty ticks, each press
    // undoing the last.
    bool settling = false;
    // Ways this thing has been seen to change. Owned by SemanticWorld and
    // built by watching: while an attempt on this thing is open, every way
    // whose shape or state moves is written down here. It says nothing about
    // how -- a switch that opens a door and a door that is the door land the
    // same entry -- only that acting on this changed that. It is what makes
    // "the way is shut and something can be done about it" answerable for a
    // switch across the room as well as for a curtain in the doorway.
    // Stateful geometry this act was seen to move. Observed, never
    // declared: the act is done, and whatever changed configuration while it
    // settled is what it moves. Nothing here knows what the geometry is for.
    std::vector<GeometryId> moves;
    // Stateful geometry this act addresses, according to the world's own
    // wiring rather than according to having done it.
    //
    // This is the same kind of fact as "this wall borders that space": the
    // world holds it, the mapper reads it, and nothing above knows what it
    // was read from. It is what lets an act be understood before it has ever
    // been done -- which is the difference between finding out what a switch
    // does by pressing it and knowing beforehand that two switches work the
    // same thing.
    std::vector<GeometryId> commands;
    // Ways this act was tried against and left shut, with what was standing
    // in each at the time. An experiment already run: doing it again with
    // the same thing in the way, in the same state, asks the same question
    // and gets the same answer.
    struct Fruitless
    {
        RelationId way = kNoId;
        AffordanceId blocking = kNoId;
        uint64_t blockingState = 0;
    };
    std::vector<Fruitless> triedInVain;
    std::vector<RelationId> affects;
    // How each of those ways stood when this thing was last acted on.
    // Parallel to `affects`.
    std::vector<uint64_t> affectedAt;
    // How the ways this thing moves stood the last time it was acted on.
    //
    // Doing it again in exactly the situation it was last done in will do
    // exactly what it did then, and what it did then is already the world we
    // are looking at. That is not a timer and nothing is being suppressed
    // for a while: the moment anything about those ways is different, the
    // situation is different and the act is on offer again.
    uint64_t triedAt = 0;
    bool triedAtKnown = false;
};

// One configuration a piece of stateful geometry can rest in.
//
// Deliberately not named. It is a support height and a ceiling height and
// nothing else: which of them is "open", "shut", "up" or "down" is not a
// property of the configuration, it is a property of what the connectivity
// derived in that configuration turns out to be, and it is read off there.
struct GeometryConfiguration
{
    int supportZ = 0;
    int ceilingZ = 0;
    // How far this configuration displaces the thing from the first one.
    //
    // A discriminator and a thing to read in a trace, not a description of
    // the shape: something that turns is not described by an offset, and
    // nothing may reconstruct an outline from this. What the space actually
    // is in a configuration is a question for whoever can put the world into
    // it and look.
    Vec2 shift;
};

// A piece of geometry that rests in more than one configuration.
struct StatefulGeometry
{
    GeometryId id = kNoId;
    // Which configuration it is resting in, or is heading for. An index into
    // `configurations`, never a flag: nothing here assumes there are two.
    uint32_t state = 0;
    bool moving = false;   // between configurations right now
    std::vector<GeometryConfiguration> configurations;
    // Whether the configurations above say anything different from each
    // other.
    //
    // A configuration is a support height and a ceiling height, so something
    // that moves in the plane -- sliding, turning, running along a path --
    // comes back with the same numbers at both ends. That is not a thing
    // with one configuration; it is a thing whose configurations this model
    // does not describe, and the two must not be confused. Nothing may read
    // "I cannot tell these apart" as "moving it changes nothing".
    bool configurationsDiffer = false;
};

struct ActorState
{
    Vec3 position;
    RegionId region = kNoId;
    // Which stateful geometry is holding the body up, if any. This is the
    // fact that survives the geometry moving: the Region underfoot may be
    // re-derived into something with a different id, and the body is still
    // standing on the same thing, and is carried by it.
    GeometryId supportedBy = kNoId;
    bool alive = false;
};

struct WorldDelta
{
    std::vector<Region> regions;
    std::vector<SpatialRelation> relations;
    std::vector<Affordance> affordances;
    std::vector<StatefulGeometry> geometry;
    ActorState actor;
    bool changed = false;
    bool settling = false;
    // What version of the world this is. Anything that could change an
    // answer about it moves this number; nothing reads it for meaning.
    uint64_t revision = 0;
    // The mapper still has questions outstanding about what the world
    // offers. Nothing is missing from what is here; there is simply more to
    // come, so "nothing to do" is not yet a conclusion anyone can draw.
    bool establishing = false;
    AffordanceId resolvedAffordance = kNoId;
};

class SemanticWorld
{
public:
    void apply(const WorldDelta &delta);

    const ActorState &actor() const { return m_actor; }
    // Is the world still answering?
    //
    // Its own motion, plus the tick on which an act was made. Blood resolves
    // a use within the tick it happens, so what an act set in motion is not
    // visible until the next one -- and an act judged on the tick it was
    // made is judged before the world has said anything. That is how an act
    // gets called finished, the planner moves on, and whatever the act
    // actually moved is then credited to whatever was being done next.
    bool settling() const { return m_settling || m_attemptFresh; }
    bool establishing() const { return m_establishing; }
    uint64_t revision() const { return m_revision; }

    const std::vector<Region> &regions() const { return m_regions; }
    const std::vector<SpatialRelation> &relations() const
    {
        return m_relations;
    }
    // Every piece of geometry that rests in more than one configuration.
    const std::vector<StatefulGeometry> &geometry() const
    {
        return m_geometry;
    }
    const StatefulGeometry *geometryOf(GeometryId id) const
    {
        for (const StatefulGeometry &piece : m_geometry)
            if (piece.id == id)
                return &piece;
        return nullptr;
    }
    const std::vector<Affordance> &affordances() const
    {
        return m_affordances;
    }
    const Region *region(RegionId id) const;
    const SpatialRelation *relation(RelationId id) const;
    const Affordance *affordance(AffordanceId id) const;

    size_t liveRegionCount() const;
    size_t liveRelationCount() const;
    size_t uncrossedRelationCount() const;

    // Attempt bookkeeping. An accepted action is never treated as a
    // permanently finished one; all that is recorded is that it was tried
    // and whether the world moved afterwards.
    // `possibilities` is how much the world currently offers, counted by
    // whoever knows: this layer does not derive reachability.
    // The actor has been to this way and looked at it. Recorded by whoever
    // drove it there; this layer does not measure distances.
    void noteInspected(RelationId id);
    // What stands in a way. Recorded by whoever asked the world; this layer
    // keeps it because a delta describes the world's shape, not what has
    // been found out about it.
    void noteObstruction(RelationId id, AffordanceId thing);

    void beginAttempt(AffordanceId id, int possibilities);
    void closeAttempt(int possibilities);
    // Acting on this thing was seen to change whether that way can be gone
    // through. Told from outside, because whether a way can be gone through
    // is a question about a body and this layer does not have one.
    void noteAffects(AffordanceId thing, RelationId way);
    // Acting on this thing was seen to change that geometry's configuration.
    void noteMoves(AffordanceId thing, GeometryId geometry);
    // This act was done to open that way, and the way is still shut.
    void noteFruitless(AffordanceId thing, RelationId way,
                       uint64_t blockingState);
    // Is doing it again a different experiment from the last one?
    bool worthTryingFor(AffordanceId thing, RelationId way,
                        uint64_t blockingState) const;
    // How the thing standing in this way stands right now, as one number.
    // Told from outside, because what a blocker's state is belongs to
    // whichever layer can see the engine.
    uint64_t blockerState(RelationId way) const
    {
        return way != kNoId && size_t(way) < m_blockerState.size()
            ? m_blockerState[size_t(way)] : 0;
    }
    void noteBlockerState(RelationId way, uint64_t state)
    {
        if (way == kNoId)
            return;
        if (m_blockerState.size() <= size_t(way))
            m_blockerState.resize(size_t(way) + 1, 0);
        m_blockerState[size_t(way)] = state;
    }
    bool moves(AffordanceId thing, GeometryId geometry) const;
    // Everything that is known to move that geometry: what has been seen to,
    // and what the world says addresses it.
    bool works(AffordanceId thing, GeometryId geometry) const;
    // Has anything that works this geometry already been done? An act whose
    // effect is already known is not an experiment, whatever its own count
    // of attempts says.
    bool effectKnown(AffordanceId thing) const;
    // Whether acting on this thing has ever been seen to change this way.
    bool affects(AffordanceId thing, RelationId way) const;
    // Is acting on this thing worth doing for this way?
    //
    // Only if it is known to move it, and only if the way is not standing
    // exactly as it stood the last time this was done. Pressing a switch
    // that has already been pressed with this way in this state does what
    // it did then, and what it did then is the world being looked at.
    //
    // Per way, deliberately. A door that moves changes the shape of every
    // room it borders, so it ends up known to move dozens of ways; asking
    // whether *anything* it moves has changed is always yes while the door
    // itself is swinging, and then it is pressed again, and again.
    bool worthTrying(AffordanceId thing, RelationId way) const;
    bool attemptOpen() const { return m_awaiting != kNoId; }
    AffordanceId openAttempt() const { return m_awaiting; }

private:
    void noteMovement(RegionId previous, RegionId now);

    std::vector<Region> m_regions;
    std::vector<SpatialRelation> m_relations;
    std::vector<Affordance> m_affordances;
    std::vector<StatefulGeometry> m_geometry;
    std::vector<uint64_t> m_blockerState;
    ActorState m_actor;
    bool m_settling = false;
    bool m_attemptFresh = false;
    // Everything about a way that acting on something could change.
    uint64_t stateOf(const SpatialRelation &relation) const;
    // What every way looked like when the open attempt began.
    std::vector<uint64_t> m_waysAtAttempt;
    bool m_establishing = false;
    uint64_t m_revision = 0;
    AffordanceId m_awaiting = kNoId;
    int m_possibilitiesAtAttempt = 0;
};

} // namespace semantic
