#include "caleb_physics.h"
#include "blood_terrain.h"

#include <algorithm>
#include <deque>
#include <map>
#include <cmath>

#include "fix16.h"
#include "build.h"
#include "../../../actor.h"
#include "../../../blood.h"
#include "../../../common_game.h"
#include "../../../globals.h"
#include "../../../player.h"

// Blood's own sector translation. It has external linkage and no header, so
// it is declared here rather than reached for through one -- the alternative
// is a second implementation of the same arithmetic, and a second opinion
// about where a wall is.
void TranslateSector(int nSector, int a2, int a3, int a4, int a5, int a6,
                     int a7, int a8, int a9, int a10, int a11,
                     char bAllWalls);

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

// The body measured once per posture, rather than once per animation frame.
BodyShape CalebPhysics::stableBody() const
{
    const int key = gMe ? (gMe->lifeMode * 8 + gMe->posture) : -1;
    if (key < 0)
        return liveBody();
    for (const Measured &known : m_measured)
        if (known.key == key)
            return known.shape;
    Measured fresh;
    fresh.key = key;
    fresh.shape = liveBody();
    m_measured.push_back(fresh);
    return fresh.shape;
}

void CalebPhysics::refresh()
{
    traversal::ActorProfile next;
    if (!gMe || !gMe->pSprite)
    {
        next.revision = 0;
        m_profile = next;
        return;
    }
    // The body as it is when it is not mid-stride.
    //
    // GetSpriteExtents measures the tile the player sprite is showing, and
    // the frames of the walk cycle are not all the same height. So the
    // measured body breathes as Caleb walks -- five different heights on one
    // level, and a step allowance moving with them -- and since this
    // profile's revision is what says whether a derived crossing still
    // stands, every frame of the walk animation threw away every crossing in
    // the map and worked it out again with engine probes. On E2M3 that was
    // two hundred and thirty-nine thousand re-derivations against fifteen
    // hundred real changes, fifty-eight milliseconds a decision, and
    // second-long hangs.
    //
    // What the body really is changes with posture and with life mode --
    // crouching, swimming, shrunk, beast -- so that is what it is keyed on.
    // Which frame of the walk it happens to be showing is not a fact about
    // the body. Probes still ask the engine with the live extents wherever
    // they need them; this is only what the model is told the body is.
    const BodyShape body = stableBody();
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
    next.posture = gMe->posture;
    next.lifeMode = gMe->lifeMode;

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
    // Not into ground that is on the move. This reads the flag as the
    // terrain snapshot left it, which is stale far more often than not --
    // see the note in blood_world_adapter about why making it live is not
    // as simple as reading gBusy here.
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

namespace {

// Put a piece of stateful geometry into one of its configurations, ask, and
// put it back exactly as it was.
//
// Blood keeps a moving sector's two ends in its XSECTOR and interpolates
// between them; setting floorz and ceilingz to one of those ends is the
// world as it rests in that configuration, and every z query the engine
// answers -- GetZRange, getflorzofslope, the clip tests -- then answers for
// that configuration. Which is the point: the engine stays the only thing
// that says what is walkable, whichever configuration is being asked about.
//
// baseFloor/baseCeil go with them: they are what the engine measures its own
// motion against, and leaving them behind would make the next real movement
// of this geometry start from the wrong place.
// Where a mover's walls are in one configuration.
//
// Blood keeps the two ends of a sliding or turning sector as a pair of
// marker sprites and a position along the way between them; posing it is
// putting it at one end. Restoring is putting it back where it was, by the
// same route -- not by saving wall coordinates, because moving a wall moves
// every wall that shares its corners, including in sectors this one has
// nothing to do with.
//
// Only a mover at rest can be posed at all. One in transit is not in a
// configuration, and where it is cannot be recovered by asking for a number
// again, so the answer to any question about it is that there is no answer.
void translateTo(uint64_t stateTag, int busy)
{
    XSECTOR &extra = xsector[stateTag];
    const int owner = extra.reference;
    if (!validSector(owner) || !validSprite(extra.marker0))
        return;
    const spritetype &first = sprite[extra.marker0];
    const int type = sector[owner].type;
    if (type == kSectorSlide || type == kSectorSlideMarked)
    {
        if (!validSprite(extra.marker1))
            return;
        const spritetype &second = sprite[extra.marker1];
        TranslateSector(owner, 0, busy, first.x, first.y, first.x, first.y,
                        first.ang, second.x, second.y, second.ang,
                        type == kSectorSlide);
        return;
    }
    TranslateSector(owner, 0, busy, first.x, first.y, first.x, first.y, 0,
                    first.x, first.y, first.ang, type == kSectorRotate);
}

// How far a mover's walls actually went, read back rather than worked out.
//
// Posing puts the engine into another configuration. The Region descriptions
// handed around above still describe the old one, and anything that consults
// them -- which canTraverse does -- is then reasoning about half a world:
// engine geometry in the new pose, outlines in the old. That is how a doorway
// a sliding wall has just closed still answers "you can walk through this".
//
// So the motion is measured off the walls after posing and applied to the
// description too. A motion that is not the same for every wall is not a
// translation, and this says so rather than inventing one.
struct WallShift
{
    bool known = false;
    int dx = 0;
    int dy = 0;
};

std::vector<std::pair<int, int>> wallsOf(int owner)
{
    std::vector<std::pair<int, int>> out;
    if (!validSector(owner))
        return out;
    const int first = sector[owner].wallptr;
    for (int i = 0; i < sector[owner].wallnum; ++i)
        out.push_back({ wall[first + i].x, wall[first + i].y });
    return out;
}

WallShift shiftOf(int owner, const std::vector<std::pair<int, int>> &before)
{
    WallShift said;
    if (!validSector(owner) || before.empty()
        || int(before.size()) != sector[owner].wallnum)
        return said;
    const int first = sector[owner].wallptr;
    const int dx = wall[first].x - before[0].first;
    const int dy = wall[first].y - before[0].second;
    for (int i = 1; i < sector[owner].wallnum; ++i)
        if (wall[first + i].x - before[size_t(i)].first != dx
            || wall[first + i].y - before[size_t(i)].second != dy)
            return said;   // a turn, or a shape change: not one offset
    said.known = true;
    said.dx = dx;
    said.dy = dy;
    return said;
}

void shiftRegion(Region &region, const WallShift &by)
{
    for (semantic::Vec2 &corner : region.footprint)
    {
        corner.x += by.dx;
        corner.y += by.dy;
    }
    for (semantic::Loop &hole : region.holes)
        for (semantic::Vec2 &corner : hole)
        {
            corner.x += by.dx;
            corner.y += by.dy;
        }
    region.interior.x += by.dx;
    region.interior.y += by.dy;
}

class PosedGeometry
{
public:
    // `part` is how far along the travel to put it, as a fraction of 0x10000.
    PosedGeometry(uint64_t stateTag, int part)
    {
        if (stateTag == 0 || stateTag >= kMaxXSectors)
            return;
        XSECTOR &extra = xsector[stateTag];
        const int owner = extra.reference;
        if (!validSector(owner))
            return;
        if (movesInThePlane(sector[owner].type)
            && (extra.busy == 0 || extra.busy == 0x10000))
        {
            // Only something at rest can be posed and put back. One in
            // transit is not at any configuration, and where it was cannot be
            // recovered by asking for a number again.
            m_outline = stateTag;
            m_busy = extra.busy;
            translateTo(stateTag, part);
        }
        m_sector = owner;
        m_floor = sector[owner].floorz;
        m_ceiling = sector[owner].ceilingz;
        m_baseFloor = baseFloor[owner];
        m_baseCeil = baseCeil[owner];
        const int floorZ = extra.offFloorZ
            + int((int64_t(extra.onFloorZ - extra.offFloorZ) * part) / 0x10000);
        const int ceilZ = extra.offCeilZ
            + int((int64_t(extra.onCeilZ - extra.offCeilZ) * part) / 0x10000);
        sector[owner].floorz = floorZ;
        sector[owner].ceilingz = ceilZ;
        baseFloor[owner] = floorZ;
        baseCeil[owner] = ceilZ;
    }

    ~PosedGeometry()
    {
        if (m_outline != 0)
            translateTo(m_outline, m_busy);
        if (m_sector < 0)
            return;
        sector[m_sector].floorz = m_floor;
        sector[m_sector].ceilingz = m_ceiling;
        baseFloor[m_sector] = m_baseFloor;
        baseCeil[m_sector] = m_baseCeil;
    }

    PosedGeometry(const PosedGeometry &) = delete;
    PosedGeometry &operator=(const PosedGeometry &) = delete;

private:
    uint64_t m_outline = 0;
    int m_busy = 0;
    int m_sector = -1;
    int m_floor = 0;
    int m_ceiling = 0;
    int m_baseFloor = 0;
    int m_baseCeil = 0;
};

} // namespace

bool CalebPhysics::anyWayOut(
    const std::vector<std::pair<semantic::GeometryId, uint32_t>> &posed,
    const Region &from, const semantic::Vec2 &at,
    const std::vector<traversal::PhysicsOracle::WayOut> &ways) const
{
    // Everything held at once, and put back in the reverse order it was
    // taken, which is what a stack of these does on the way out of scope.
    std::deque<PosedGeometry> holding;
    std::map<semantic::GeometryId, WallShift> moved;
    for (const auto &one : posed)
    {
        const uint64_t tag = m_geometryTag ? m_geometryTag(one.first) : 0;
        if (tag == 0 || tag >= kMaxXSectors)
            continue;
        const int owner = xsector[tag].reference;
        const std::vector<std::pair<int, int>> before = wallsOf(owner);
        holding.emplace_back(tag, one.second == 0 ? 0 : 0x10000);
        moved[one.first] = shiftOf(owner, before);
    }
    // The description of a space that has just moved, moved with it.
    auto posedAs = [&](const Region &space, Region &out) {
        out = space;
        if (space.mover == semantic::kNoId)
            return true;
        auto found = moved.find(space.mover);
        if (found == moved.end())
            return true;   // not one of the things being held
        if (!found->second.known)
            return false;  // it turned, and this cannot say where it went
        shiftRegion(out, found->second);
        return true;
    };
    for (const traversal::PhysicsOracle::WayOut &way : ways)
    {
        if (!way.relation || !way.to || !way.to->exists)
            continue;
        Region posedFrom;
        Region posedTo;
        // Somewhere this cannot describe is not somewhere to call a way out.
        // For the question this answers -- is the body about to shut itself
        // in -- an unanswerable way is not one to count on.
        if (!posedAs(from, posedFrom) || !posedAs(*way.to, posedTo))
            continue;
        semantic::Vec2 crossing = way.relation->gateway.midpoint();
        semantic::Vec2 arrival = posedTo.interior;
        semantic::Vec2 departure = at;
        ++m_queries;
        if (canTraverse(traversal::Mode::Walk, *way.relation, posedFrom,
                        posedTo, &at, crossing, arrival, departure))
            return true;
    }
    return false;
}

void CalebPhysics::sweptThrough(
    semantic::GeometryId geometry, uint32_t step, uint32_t steps,
    const std::vector<traversal::PhysicsOracle::Stance> &places,
    std::vector<char> &out) const
{
    out.assign(places.size(), 0);
    const uint64_t tag = m_geometryTag ? m_geometryTag(geometry) : 0;
    if (tag == 0 || tag >= kMaxXSectors)
        return;
    const int owner = xsector[tag].reference;
    if (!validSector(owner))
        return;
    const int part = steps < 2 ? 0
        : int((int64_t(0x10000) * step) / (steps - 1));
    const int reach = m_profile.radius;
    PosedGeometry held(tag, part);
    for (size_t index = 0; index < places.size(); ++index)
    {
        // The body is a hull, not a point, so the corners of it are asked
        // about too: a wall arriving at a shoulder arrives at the body.
        const semantic::Vec2 &at = places[index].at;
        const int probe[5][2] = {
            { at.x, at.y },
            { at.x + reach, at.y }, { at.x - reach, at.y },
            { at.x, at.y + reach }, { at.x, at.y - reach },
        };
        for (const auto &spot : probe)
            if (inside(spot[0], spot[1], int16_t(owner)) == 1)
            {
                out[index] = 1;
                break;
            }
    }
}

void CalebPhysics::canStandThrough(
    semantic::GeometryId geometry, uint32_t step, uint32_t steps,
    const std::vector<traversal::PhysicsOracle::Stance> &places,
    std::vector<char> &out) const
{
    out.assign(places.size(), 0);
    const uint64_t tag = m_geometryTag ? m_geometryTag(geometry) : 0;
    const XSECTOR *extra = tag == 0 ? nullptr : &xsector[tag];
    const int part = steps < 2 ? 0
        : int((int64_t(0x10000) * step) / (steps - 1));
    // The floor plane is part of the world as the region describes it, so a
    // region made of this geometry has to be told where its floor is at the
    // pose asked about -- but only where the geometry moves in z. Something
    // that slides has no such numbers, and writing the zeros it reports into
    // the plane puts its floor at the origin.
    const bool inZ = extra
        && (extra->offFloorZ != extra->onFloorZ
            || extra->offCeilZ != extra->onCeilZ);
    const int floorZ = !extra ? 0
        : extra->offFloorZ
            + int((int64_t(extra->onFloorZ - extra->offFloorZ) * part)
                  / 0x10000);
    PosedGeometry held(tag, part);
    for (size_t index = 0; index < places.size(); ++index)
    {
        if (!places[index].region)
            continue;
        Region posed = *places[index].region;
        if (inZ && posed.mover == geometry)
            posed.support = semantic::flatPlane(floorZ);
        ++m_queries;
        out[index] = stanceIn(posed, places[index].at, liveBody()).valid
            ? 1 : 0;
    }
}

bool CalebPhysics::canTraverseWith(semantic::GeometryId geometry,
                                   uint32_t configuration,
                                   traversal::Mode mode,
                                   const SpatialRelation &relation,
                                   const Region &from, const Region &to,
                                   const semantic::Vec2 *startFrom,
                                   semantic::Vec2 &crossing,
                                   semantic::Vec2 &arrival,
                                   semantic::Vec2 &departure) const
{
    const uint64_t tag = m_geometryTag ? m_geometryTag(geometry) : 0;
    if (tag == 0)
        return canTraverse(mode, relation, from, to, startFrom, crossing,
                           arrival, departure);
    // The regions either side describe the world as it stands, so the one
    // this geometry makes has to be told where its floor is in the
    // configuration being asked about. Nothing else about it changes: the
    // outline it presents is the outline it has.
    const XSECTOR &extra = xsector[tag];
    const int floorZ = configuration == 0 ? extra.offFloorZ : extra.onFloorZ;
    // Only where the geometry moves in z. A slide reports zero at both ends
    // because those fields are for something else, and writing that into the
    // support plane derives every crossing against a floor at the origin.
    const bool inZ = extra.offFloorZ != extra.onFloorZ
        || extra.offCeilZ != extra.onCeilZ;
    Region posedFrom = from;
    Region posedTo = to;
    if (inZ && from.mover == geometry)
        posedFrom.support = semantic::flatPlane(floorZ);
    if (inZ && to.mover == geometry)
        posedTo.support = semantic::flatPlane(floorZ);

    PosedGeometry posed(tag, configuration == 0 ? 0 : 0x10000);
    return canTraverse(mode, relation, posedFrom, posedTo, startFrom,
                       crossing, arrival, departure);
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
