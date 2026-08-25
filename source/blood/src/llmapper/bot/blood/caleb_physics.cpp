#include "caleb_physics.h"

#include <algorithm>
#include <cmath>

#include "fix16.h"
#include "build.h"
#include "../../../actor.h"
#include "../../../blood.h"
#include "../../../common_game.h"
#include "../../../globals.h"
#include "../../../player.h"

namespace bloodmap {

namespace {

using semantic::Gateway;
using semantic::Region;
using semantic::SpatialRelation;
using semantic::Vec2;

uint64_t mixHash(uint64_t value, uint64_t item)
{
    value ^= item + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
    return value;
}

// Standing in a region means being inside its outline, out of whatever is
// standing in it, and held up by something the region is made of.
//
// The last test is ownership, not height. The engine's own answer for what
// is holding a body up is `florhit`, and the mapper mints a region's support
// identifiers in the same terms, so the two compare directly. Comparing the
// engine's answer against the region's plane instead would call a body
// standing one width from a step "not in this region", because within a hull
// width of a step the engine really is holding the body up with the step.
// Is the body, as the engine has placed it, in this region?
//
// Two answers count. The engine names the thing holding the body up, and a
// region made of that thing is where the body is -- that is what tells a
// plank from the pit under it. But when the hull straddles a join the engine
// may name either side of it, so a support at the height this region's floor
// has here is also this region: the two tests fail in different places and
// neither is a tolerance on the other.
PhysicalPose stanceIn(const Region &region, const Vec2 &at,
                      const BodyShape &body, int *wanted = nullptr,
                      int *found = nullptr, int *sector = nullptr)
{
    PhysicalPose pose;
    if (!semantic::pointInPolygon(region.shape(), at))
        return pose;
    const int surface = region.support.zAt(at.x, at.y);
    if (wanted)
        *wanted = surface;
    pose = supportAt(at.x, at.y, surface - body.footOffset, -1);
    if (found)
        *found = pose.valid ? pose.supportZ : 0;
    if (sector)
        *sector = pose.valid ? pose.sector : -1;
    if (pose.valid && !bodyRestsOn(region, pose, body))
        pose.valid = false;
    return pose;
}

// How far past a boundary a body has to be before the engine stops holding it
// up on what is behind it.
//
// GetZRange does not sweep a disc. Build walks a square box of walldist
// either side of the position, and any sector whose wall falls inside that
// box -- touching it counts -- gets a say in what the floor is. So a body
// exactly one hull width past a ledge is still standing on the ledge, and
// the engine is right about that: half its square is over the ledge.
//
// It is also why walking off a ledge has to carry the body a square clear
// rather than a radius clear. Stopping a radius past the edge is stopping at
// the one distance where the answer is still the old floor, so the body
// finishes in the room below with the floor above still under it -- which
// reads as having ended up somewhere it does not belong, and the way down
// gets thrown away. The shaft with AGTST18's exit switch at the bottom has
// three ways into it and every one of them was refused for this.
//
// For a boundary at an angle it is the square's reach in that boundary's own
// direction, which is what a square's support function comes to.
int hullClearance(const Gateway &gateway, int radius)
{
    const double dx = double(gateway.to.x) - gateway.from.x;
    const double dy = double(gateway.to.y) - gateway.from.y;
    const double width = std::sqrt(dx * dx + dy * dy);
    if (width < 1.0)
        return radius + 1;
    const double alongX = std::abs(dy / width);
    const double alongY = std::abs(dx / width);
    return int(std::lround(double(radius) * (alongX + alongY))) + 1;
}

// A point past an opening, on the side the given region is, and as far into
// the space there as that space actually goes.
//
// Nothing here asks whether a whole body fits at that point, and asking
// would be the same mistake as asking whether a body fits between two walls.
// Blood adds no clip line to a wall with passable space behind it, so a body
// walks down a corridor narrower than itself and stands on a ledge thinner
// than itself all day. The point is a direction to push in and a distance
// worth pushing; where the body then is, is the engine's answer.
//
// Which side is which comes from the region's own interior point, so no
// guess is involved and no fallback toward a centroid is needed.
Vec2 insideFrom(const Gateway &gateway, const Region &into, const Vec2 &at,
                int inset)
{
    const double dx = double(gateway.to.x) - gateway.from.x;
    const double dy = double(gateway.to.y) - gateway.from.y;
    const double width = std::sqrt(dx * dx + dy * dy);
    if (width < 1.0)
        return at;
    // Which way is in is decided beside the opening, not from the far end
    // of the region. A region that wraps around the space on the other side
    // of this opening keeps its own interior point on the wrong side of this
    // opening's line, and the line is all a cross product can see.
    const semantic::Polygon shape = into.shape();
    const double ux = -dy / width;
    const double uy = dx / width;
    const Vec2 plus = { at.x + int(std::lround(ux * 4)),
                        at.y + int(std::lround(uy * 4)) };
    const Vec2 minus = { at.x - int(std::lround(ux * 4)),
                         at.y - int(std::lround(uy * 4)) };
    const bool towardPlus = semantic::pointInPolygon(shape, plus);
    const bool towardMinus = semantic::pointInPolygon(shape, minus);
    double sign;
    if (towardPlus != towardMinus)
        sign = towardPlus ? 1.0 : -1.0;
    else
        sign = semantic::crossOf(gateway.from, gateway.to,
                                 into.interior) >= 0 ? 1.0 : -1.0;
    const double nx = ux * sign;
    const double ny = uy * sign;
    // How much room there is, asked for twice the distance wanted, so that
    // "there is at least this much" and "there is only this much" are
    // different answers rather than the same one.
    const Vec2 away = { at.x + int(std::lround(nx * inset * 2)),
                        at.y + int(std::lround(ny * inset * 2)) };
    const int room = semantic::depthInto(shape, at, away);
    // Room enough: stand where we meant to. Not enough: the middle of what
    // there is, so a thin space is entered near its centre rather than
    // against the wall on its other side.
    const int depth = room >= inset ? inset : std::max(1, room / 2);
    return { at.x + int(std::lround(nx * depth)),
             at.y + int(std::lround(ny * depth)) };
}

// Walk the body from a pose to a point and say whether it got there. The
// answer is the engine's: repeated ClipMove with the live hull, settling on
// whatever the engine reports underneath.
bool reaches(PhysicalPose &pose, const Vec2 &target, const BodyShape &body)
{
    const int distance = planarDistance(pose.x, pose.y, target.x, target.y);
    if (distance <= body.radius)
        return true;
    const WalkProbe probe = walkTowards(pose, target.x - pose.x,
                                        target.y - pose.y, distance, body);
    if (!probe.end.valid)
        return false;
    pose = probe.end;
    return planarDistance(pose.x, pose.y, target.x, target.y) <= body.radius;
}

// Where along an opening to try. How many places there are to try is the
// opening's question and the body's -- how many body widths fit in it -- and
// never a chosen number. The middle is tried first, so an opening that is
// plainly clear costs one query.
Vec2 alongOpening(const Gateway &gateway, int index, int slots)
{
    const double t = (double(index) + 0.5) / double(slots);
    return { int(std::lround(gateway.from.x
                             + t * (gateway.to.x - gateway.from.x))),
             int(std::lround(gateway.from.y
                             + t * (gateway.to.y - gateway.from.y))) };
}

int slotsAcross(const Gateway &gateway, int radius)
{
    return std::max(1, gateway.width() / std::max(1, radius * 2));
}

int slotOrder(int step, int slots)
{
    return (step % 2 == 0) ? slots / 2 + step / 2
                           : slots / 2 - 1 - step / 2;
}

} // namespace

void CalebPhysics::refresh()
{
    traversal::ActorProfile next;
    if (!gMe || !gMe->pSprite)
    {
        next.revision = 0;
        m_profile = next;
        return;
    }
    const BodyShape body = liveBody();
    const POSTURE &standing = gMe->pPosture[gMe->lifeMode][kPostureStand];
    const POSTURE &crouching = gMe->pPosture[gMe->lifeMode][kPostureCrouch];
    next.radius = body.radius;
    next.standHeight = body.height;
    next.crouchHeight = body.height - (standing.eyeAboveZ
                                       - crouching.eyeAboveZ);
    // How far up this body gets in one step, by cliptestsector's own
    // arithmetic. Reported so a run can be checked against what the body
    // actually managed; it decides nothing here, because the crossing
    // queries below ask the engine itself.
    next.stepUp = stepAllowance(body);
    next.walkSpeed = standing.frontAccel;
    next.jumpImpulse = packItemActive(gMe, kPackJumpBoots)
        ? standing.pwupJumpZ : standing.normalJumpZ;
    next.gravity = kDudeGravity;

    uint64_t revision = 1469598103934665603ULL;
    revision = mixHash(revision, uint64_t(uint32_t(next.radius)));
    revision = mixHash(revision, uint64_t(uint32_t(next.standHeight)));
    revision = mixHash(revision, uint64_t(uint32_t(next.crouchHeight)));
    revision = mixHash(revision, uint64_t(uint32_t(next.stepUp)));
    revision = mixHash(revision, uint64_t(uint32_t(next.jumpImpulse)));
    revision = mixHash(revision, uint64_t(uint32_t(next.walkSpeed)));
    revision = mixHash(revision, uint64_t(gMe->posture));
    revision = mixHash(revision, uint64_t(gMe->lifeMode));
    next.revision = revision;
    m_profile = next;
}

// Crossing an opening is a local act: stand a body width inside one region
// beside the opening, go through it, and end up standing a body width inside
// the other. Getting to that first stance from wherever the body happens to
// be is a different question, asked of the region's own free space by
// whoever is doing the walking.
bool CalebPhysics::canStand(const Region &region, const Vec2 &at) const
{
    if (!region.exists)
        return false;
    ++m_queries;
    return stanceIn(region, at, liveBody()).valid;
}

void CalebPhysics::note(semantic::RelationId relation, Refusal refusal) const
{
    if (relation == semantic::kNoId)
        return;
    if (m_refusal.size() <= size_t(relation))
        m_refusal.resize(size_t(relation) + 1, Refusal::None);
    m_refusal[size_t(relation)] = refusal;
}

CalebPhysics::Refusal CalebPhysics::refusalFor(
    semantic::RelationId relation) const
{
    if (relation == semantic::kNoId || size_t(relation) >= m_refusal.size())
        return Refusal::None;
    return m_refusal[size_t(relation)];
}

int CalebPhysics::obstacleFor(semantic::RelationId relation) const
{
    if (relation == semantic::kNoId || size_t(relation) >= m_obstacle.size())
        return -1;
    return m_obstacle[size_t(relation)];
}

const CalebPhysics::Evidence &CalebPhysics::evidenceFor(
    semantic::RelationId relation) const
{
    static const Evidence nothing;
    if (relation == semantic::kNoId || size_t(relation) >= m_evidence.size())
        return nothing;
    return m_evidence[size_t(relation)];
}

const char *CalebPhysics::refusalName(Refusal refusal)
{
    switch (refusal)
    {
    case Refusal::None: return "none";
    case Refusal::NotOffered: return "not_offered";
    case Refusal::NoWidth: return "no_width";
    case Refusal::NoRoom: return "no_room_near_side";
    case Refusal::NoLanding: return "no_room_far_side";
    case Refusal::NoStance: return "no_stance";
    case Refusal::OpeningUnreached: return "opening_unreached";
    case Refusal::LandingUnreached: return "landing_unreached";
    case Refusal::OutsideTarget: return "outside_target";
    case Refusal::WrongSupport: return "wrong_support";
    }
    return "unknown";
}

bool CalebPhysics::walkAcross(const SpatialRelation &relation,
                              const Region &from, const Region &to,
                              const semantic::Vec2 *startFrom,
                              semantic::Vec2 &crossing,
                              semantic::Vec2 &arrival,
                              semantic::Vec2 &departure) const
{
    const BodyShape body = liveBody();
    const Gateway &gateway = relation.gateway;
    if (gateway.width() <= 0)
    {
        note(relation.id, Refusal::NoWidth);
        return false; // stacked surfaces are not walked between
    }
    // Not into ground that is on the move.
    //
    // A sector Blood is animating has its walls travelling across the space
    // it covers, and there is no standing still in the way of that: the body
    // is pushed into whatever is behind it and squashed. This is a fact
    // about right now rather than about the place, so it stops being true
    // when the sector does, and the crossing is derived again then.
    if (to.hazard.shifting)
    {
        note(relation.id, Refusal::NoStance);
        return false;
    }
    Refusal furthest = Refusal::NoRoom;
    if (relation.id != semantic::kNoId)
    {
        if (m_obstacle.size() <= size_t(relation.id))
            m_obstacle.resize(size_t(relation.id) + 1, -1);
        m_obstacle[size_t(relation.id)] = -1;
    }

    // A rise past what the engine steps over cannot be walked, and there was
    // once a dispute about whether that was really true, so every number
    // crossing into the engine was written down whenever one came up.
    //
    // That is not a rare event. A map has plenty of steps taller than a body
    // climbs, every one of them is looked at again whenever anything near it
    // moves, and each look writes a line per stance and per step of the walk.
    // On AGTST18 it came to four hundred and twenty thousand trace lines in
    // five and a half minutes of play -- seventy-seven megabytes, each line
    // built as a string inside the probe and then flushed to disk on its own
    // -- for a question that was settled a long time ago. Diagnostics that
    // fire on the ordinary case are not diagnostics.
    //
    // The machinery stays: traceMotion(true) still turns it all on, and it
    // is worth turning on by hand the next time the model and the engine
    // disagree about a step. Nothing turns it on by itself.
    const bool suspect = false;
    if (suspect)
    {
        char text[224];
        std::snprintf(text, sizeof(text),
            "crossing rel=%u %u->%u step=%d gate=(%d,%d)-(%d,%d) "
            "from_floor=%d to_floor=%d radius=%d foot=%d fd=%d cd=%d "
            "height=%d allow=%d",
            unsigned(relation.id), unsigned(from.id), unsigned(to.id),
            relation.verticalStep, gateway.from.x, gateway.from.y,
            gateway.to.x, gateway.to.y,
            from.support.zAt(gateway.midpoint().x, gateway.midpoint().y),
            to.support.zAt(gateway.midpoint().x, gateway.midpoint().y),
            body.radius, body.footOffset, body.floorDistance,
            body.ceilingDistance, body.height, stepAllowance(body));
        traceMotion(true);
        traceLine(text);
    }
    struct TraceOff
    {
        ~TraceOff() { traceMotion(false); }
    } traceOff;

    const int slots = slotsAcross(gateway, body.radius);
    for (int step = 0; step < slots; ++step)
    {
        const int index = slotOrder(step, slots);
        if (index < 0 || index >= slots)
            continue;
        const Vec2 at = alongOpening(gateway, index, slots);
        Vec2 start;
        Vec2 landing;
        if (tracingMotion())
        {
            char text[128];
            std::snprintf(text, sizeof(text), "slot %d of %d at=(%d,%d)",
                          index, slots, at.x, at.y);
            traceLine(text);
        }
        // Where the body starts is not this layer's guess. The caller
        // worked out where a body of this width fits in this region and
        // handed one such place over; only when it has none does the stance
        // get made up from the opening's own geometry.
        // Beside this opening, on this side of it. Crossing is a local act,
        // and a stance somewhere else in the region is not a stance at this
        // way out: from beside the next opening along, the body reaches this
        // one by going through whatever lies between, and then a crossing of
        // some other boundary gets written down as a crossing of this one.
        // Where the caller says a body of this width fits is the fallback,
        // for an opening whose own geometry offers nowhere to stand.
        // A whole hull width in, not half of one. The engine holds a body up
        // on whatever its hull touches, so a body standing exactly a radius
        // from a step is standing on the step: its hull reaches it. Setting
        // off from there is setting off from the top of a stair, and the
        // stair after it then looks like one stride.
        start = insideFrom(gateway, from, at, body.radius);
        if (!semantic::pointInPolygon(from.shape(), start))
            start = insideFrom(gateway, from, at, body.radius / 2);
        if (!semantic::pointInPolygon(from.shape(), start))
        {
            if (!startFrom)
                continue;
            start = *startFrom;
        }
        furthest = std::max(furthest, Refusal::NoLanding);
        // And a whole hull width out on the far side, for the same reason
        // read the other way: within one width of the opening the hull is
        // still over the near side, and the engine is still holding the body
        // up on it, so nothing has been crossed yet.
        landing = insideFrom(gateway, to, at,
                             hullClearance(gateway, body.radius));
        furthest = std::max(furthest, Refusal::NoStance);
        int wanted = 0;
        int found = 0;
        int sector = -1;
        PhysicalPose pose = stanceIn(from, start, body, &wanted, &found,
                                     &sector);
        if (!pose.valid)
        {
            if (relation.id != semantic::kNoId)
            {
                if (m_evidence.size() <= size_t(relation.id))
                    m_evidence.resize(size_t(relation.id) + 1);
                Evidence &record = m_evidence[size_t(relation.id)];
                record.at = start;
                record.wanted = wanted;
                record.found = found;
                record.container = sector;
                record.located = sector >= 0;
            }
            continue;
        }
        if (tracingMotion())
        {
            char text[160];
            std::snprintf(text, sizeof(text),
                "stance start=(%d,%d) landing=(%d,%d) pose=(%d,%d,%d) "
                "sec=%d sup=%d hit=%d",
                start.x, start.y, landing.x, landing.y, pose.x, pose.y,
                pose.z, pose.sector, pose.supportZ, pose.supportHit);
            traceLine(text);
        }
        furthest = std::max(furthest, Refusal::OpeningUnreached);
        // One walk, from a stance on this side to a point on the other,
        // straight through the opening -- the two are collinear with it by
        // construction. The engine moves a body once; splitting a crossing
        // into "get to the doorway" and "now go through it" invents a stop
        // in the middle, and then something has to decide how near the
        // doorway counts as being at it. A clip line stands off by exactly
        // one hull width, so that decision lands on a knife edge and the
        // answer turns on a single unit.
        const int reach = std::max(1, planarDistance(pose.x, pose.y,
                                                     landing.x, landing.y));
        const WalkProbe through = walkTowards(pose, landing.x - pose.x,
                                              landing.y - pose.y, reach, body);
        if (!through.end.valid)
            continue;
        // Whatever the engine ran the body into on the way through. Kept
        // even when a later slot succeeds is wrong, so it is only kept while
        // the crossing is still being refused.
        if (relation.id != semantic::kNoId && through.stoppedBy >= 0
            && m_obstacle[size_t(relation.id)] < 0)
            m_obstacle[size_t(relation.id)] = through.stoppedBy;
        // Walking on until the ground stops being there is not walking. The
        // engine says when that happened, and where the body comes down is
        // the descent modes' question, not this one.
        if (through.leftGround)
            continue;
        pose = through.end;
        // Which side of the opening the body finished on is the opening's
        // own question, and the landing is on the far side by construction.
        const Vec2 finished = { pose.x, pose.y };
        if ((semantic::crossOf(gateway.from, gateway.to, finished) >= 0)
            != (semantic::crossOf(gateway.from, gateway.to, landing) >= 0))
            continue;
        furthest = std::max(furthest, Refusal::LandingUnreached);
        furthest = std::max(furthest, Refusal::OutsideTarget);
        if (relation.id != semantic::kNoId)
        {
            if (m_evidence.size() <= size_t(relation.id))
                m_evidence.resize(size_t(relation.id) + 1);
            Evidence &record = m_evidence[size_t(relation.id)];
            record.aimed = landing;
            record.ended = { pose.x, pose.y };
            record.found = pose.supportZ;
            record.wanted = to.support.zAt(pose.x, pose.y);
            record.container = pose.sector;
            record.located = true;
        }
        if (!semantic::pointInPolygon(to.shape(), Vec2{ pose.x, pose.y }))
            continue;
        furthest = std::max(furthest, Refusal::WrongSupport);
        if (!bodyRestsOn(to, pose, body))
            continue;
        if (tracingMotion())
        {
            char text[160];
            std::snprintf(text, sizeof(text),
                "accepted pose=(%d,%d,%d) sec=%d sup=%d hit=%d",
                pose.x, pose.y, pose.z, pose.sector, pose.supportZ,
                pose.supportHit);
            traceLine(text);
        }
        note(relation.id, Refusal::None);
        if (relation.id != semantic::kNoId)
            m_obstacle[size_t(relation.id)] = -1;
        crossing = at;
        arrival = { pose.x, pose.y };
        departure = start;
        return true;
    }
    note(relation.id, furthest);
    return false;
}

bool CalebPhysics::dropAcross(const SpatialRelation &relation,
                              const Region &from, const Region &to,
                              const semantic::Vec2 *startFrom,
                              semantic::Vec2 &crossing,
                              semantic::Vec2 &arrival,
                              semantic::Vec2 &departure) const
{
    if (relation.verticalStep <= curbHeight())
        return false; // not a descent; z grows downward in Blood
    const BodyShape body = liveBody();
    const Gateway &gateway = relation.gateway;
    const Vec2 opening = gateway.width() > 0 ? gateway.midpoint()
                                             : gateway.from;
    Vec2 start;
    Vec2 landing;
    if (startFrom)
        start = *startFrom;
    else
        start = insideFrom(gateway, from, opening, body.radius);
    landing = insideFrom(gateway, to, opening, body.radius);
    PhysicalPose pose = stanceIn(from, start, body);
    if (!pose.valid)
        return false;
    if (gateway.width() > 0 && !reaches(pose, opening, body))
        return false;

    // Walking on runs out of ground: that is what makes it a drop rather
    // than a step. Where the body comes down is then the engine's answer at
    // the place it left from.
    const int distance = std::max(1, planarDistance(pose.x, pose.y,
                                                    landing.x, landing.y));
    const WalkProbe probe = walkTowards(pose, landing.x - pose.x,
                                        landing.y - pose.y, distance, body);
    if (!probe.leftGround)
        return false;
    const PhysicalPose below = supportAt(landing.x, landing.y,
        to.support.zAt(landing.x, landing.y) - body.footOffset, pose.sector);
    if (!below.valid || !bodyRestsOn(to, below, body))
        return false;
    // Ground that hurts is not somewhere to step off into, and a fall the
    // body does not walk away from is not a way to get anywhere. Both are
    // the world's own answers: the sector says it damages, and Blood's own
    // gravity and landing arithmetic say what the drop costs.
    if (to.hazard.harmful || to.hazard.shifting)
        return false;
    if (fallDamage(below.supportZ - pose.supportZ) > 0)
        return false;
    crossing = opening;
    arrival = landing;
    departure = start;
    return true;
}

bool CalebPhysics::canTraverse(traversal::Mode mode,
                               const SpatialRelation &relation,
                               const Region &from, const Region &to,
                               const semantic::Vec2 *startFrom,
                               semantic::Vec2 &crossing,
                               semantic::Vec2 &arrival,
                               semantic::Vec2 &departure) const
{
    if (!from.exists || !to.exists || !relation.exists || relation.blocked)
    {
        if (mode == traversal::Mode::Walk)
            note(relation.id, Refusal::NotOffered);
        return false;
    }
    ++m_queries;
    switch (mode)
    {
    case traversal::Mode::Walk:
        return walkAcross(relation, from, to, startFrom, crossing, arrival,
                          departure);
    case traversal::Mode::Drop:
        return dropAcross(relation, from, to, startFrom, crossing, arrival,
                          departure);
    case traversal::Mode::Crouch:
    case traversal::Mode::Jump:
    case traversal::Mode::Ride:
        // No engine-backed query is written for these yet. Reporting false
        // says "not derived", and the planner treats it the same as any
        // other mode it has no transition for. Nothing else in the model
        // needs to change when one of them is added here.
        return false;
    }
    return false;
}

} // namespace bloodmap
