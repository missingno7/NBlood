#include "blood_physics.h"

#include <algorithm>
#include <cmath>

#include "fix16.h"
#include "build.h"
#include "clip.h"
#include "../../../actor.h"
#include "../../../blood.h"
#include "../../../common_game.h"
#include "../../../db.h"
#include "../../../gameutil.h"
#include "../../../globals.h"
#include "../../../player.h"
#include "../../../trig.h"

namespace bloodmap {

namespace {

int gProbes = 0;

struct ClipGuard
{
    spritetype *actor = nullptr;
    int saved = 0;

    ClipGuard()
    {
        // MoveDude takes the live player out of its own clip set before
        // asking the engine anything; a probe has to do the same or the body
        // collides with itself.
        if (gMe && gMe->pSprite)
        {
            actor = gMe->pSprite;
            saved = actor->cstat;
            actor->cstat &= ~257;
        }
    }

    ~ClipGuard()
    {
        if (actor)
            actor->cstat = int16_t(saved);
    }
};

// The floor MoveDude puts the player on at this position.
//
// One query, with the hull's own clip distance, which is the only width
// Blood ever asks about: `GetZRange(pSprite, ..., wd, CLIPMASK0, ...)` with
// `wd = clipdist<<2`, every time, in every dude path. Asking about a wider
// hull than the engine does reports the floor of whatever is standing near
// the body rather than the floor under it, and a body next to a step is then
// judged to be standing on the step.
void settleSupport(int x, int y, int z, int sectorId, const BodyShape &body,
                   int &ceilingZ, int &floorZ, int &floorHit)
{
    int ceilingHit = 0;
    GetZRangeAtXYZ(x, y, z, sectorId, &ceilingZ, &ceilingHit, &floorZ,
        &floorHit, body.radius, CLIPMASK0,
        PARALLAXCLIP_CEILING | PARALLAXCLIP_FLOOR);
}

} // namespace

int probeCount() { return gProbes; }

void splitObstacle(int hit, int &wallId, int &spriteId)
{
    wallId = -1;
    spriteId = -1;
    if (hit >= 49152)
        spriteId = hit - 49152;
    else if (hit >= 32768)
        wallId = hit - 32768;
}

namespace {
bool gTracing = false;
std::vector<std::string> gTrace;
}

void traceMotion(bool on) { gTracing = on; }
bool tracingMotion() { return gTracing; }
void traceLine(const std::string &line)
{
    if (gTrace.size() < 512)
        gTrace.push_back(line);
}
const std::vector<std::string> &motionTrace() { return gTrace; }
void clearMotionTrace() { gTrace.clear(); }

namespace {
std::string describeStep(const char *what, int a, int b, int c, int d, int e,
                         int f, int g, int h)
{
    char text[256];
    std::snprintf(text, sizeof(text), "%s %d %d %d %d %d %d %d %d", what,
                  a, b, c, d, e, f, g, h);
    return text;
}
}

bool bodyStandsOn(const semantic::Region &region, const PhysicalPose &pose)
{
    if (!pose.valid)
        return false;
    const int surface = region.support.zAt(pose.x, pose.y);
    if (std::abs(pose.supportZ - surface) <= curbHeight())
        return true;
    return region.holds(uint64_t(uint32_t(pose.supportHit)))
        && pose.supportZ <= surface + curbHeight();
}

int stepAllowance(const BodyShape &body)
{
    return std::max(curbHeight(),
                    body.footOffset - body.floorDistance - 1);
}

// How far walking may take a body downwards.
//
// Not the same number as stepAllowance, and it was a mistake to use it as
// though it were. Going up, a wall stops the body: cliptestsector refuses to
// step it over a rise past what its hull clears, and that is the end of the
// matter. Going down, nothing stops it at all. The floor simply is not there
// any more, MoveDude carries on with the same forward velocity, gravity
// brings the body down, and the player -- who has done nothing but hold
// forward -- keeps walking when it lands. There is no second control, no
// mode and no decision; walking off a ledge is walking.
//
// The engine does draw a line, and it draws it at the landing rather than at
// the ledge. MoveDude works out the impact from the speed the body came down
// at, forgives kFallDamageFloor of it, and hurts the body for the rest. Below
// that the descent costs nothing whatsoever. So that is where walking ends
// and falling begins, and it is Blood's own number rather than one chosen
// here.
//
// Turning it into a height means running Blood's own integrator backwards.
// Per frame MoveDude does:
//
//     z    += zvel >> 8
//     z    += (kDudeGravity * 2) >> 8
//     zvel += kDudeGravity
//
// so after n frames from rest the speed is n*kDudeGravity and the distance
// fallen is kDudeGravity * n * (n + 3) / 512. The impact the engine forgives
// is mulscale30(v, v) <= kFallDamageFloor, which is v <= sqrt(kFallDamageFloor
// << 30) -- and for a dude on a flat floor actFloorBounceVector hands back
// the landing speed unaltered, so no elasticity enters into it.
int descentAllowance()
{
    static const int allowance = []()
    {
        // The fastest landing the engine charges nothing for.
        int64_t speed = 0;
        const int64_t forgiven = int64_t(kFallDamageFloor) << 30;
        while ((speed + 1) * (speed + 1) <= forgiven)
            ++speed;
        const int64_t frames = speed / kDudeGravity;
        return int(int64_t(kDudeGravity) * frames * (frames + 3) / 512);
    }();
    return allowance;
}

bool bodyRestsOn(const semantic::Region &region, const PhysicalPose &pose,
                 const BodyShape &body)
{
    if (!pose.valid)
        return false;
    if (region.holds(uint64_t(uint32_t(pose.supportHit))))
        return true;
    const int surface = region.support.zAt(pose.x, pose.y);
    if (pose.supportZ > surface + curbHeight())
        return false;   // fell through to a layer under this one
    // Held up by something above this region's floor. That is ordinary --
    // a body in a corridor narrower than itself is held up the whole way by
    // the room beside it -- but only as far up as the engine could have put
    // it from here. Further up is a storey, not a step: a body hovering over
    // a pit is standing on the ledge, and is not in the pit.
    return surface - pose.supportZ <= stepAllowance(body);
}

int fallDamage(int drop)
{
    if (drop <= 0)
        return 0;
    // MoveDude, per tick: the body moves by its velocity, gravity adds its
    // half step, and the velocity grows. Integrated here exactly as there
    // rather than by a formula that happens to be close.
    int fallen = 0;
    int velocity = 0;
    int guard = 4096;
    while (fallen < drop && guard-- > 0)
    {
        fallen += velocity >> 8;
        fallen += ((kDudeGravity * 4) / 2) >> 8;
        velocity += kDudeGravity;
    }
    // Landing on level ground, actFloorBounceVector with no elasticity
    // returns the velocity itself, and the damage is its square.
    const int64_t impact = int64_t(velocity) * velocity;
    return int(impact >> 30) - kFallDamageFloor;
}

int curbHeight() { return CLIPCURBHEIGHT; }

bool validSector(int sectorId)
{
    return sectorId >= 0 && sectorId < numsectors;
}

bool validWall(int wallId)
{
    return wallId >= 0 && wallId < numwalls;
}

bool validSprite(int spriteId)
{
    return spriteId >= 0 && spriteId < kMaxSprites
        && sprite[spriteId].statnum != kStatFree;
}

int64_t planarDistanceSquared(int x1, int y1, int x2, int y2)
{
    const int64_t dx = int64_t(x2) - x1;
    const int64_t dy = int64_t(y2) - y1;
    return dx * dx + dy * dy;
}

int planarDistance(int x1, int y1, int x2, int y2)
{
    return int(std::lround(
        std::sqrt(double(planarDistanceSquared(x1, y1, x2, y2)))));
}

BodyShape liveBody()
{
    BodyShape shape;
    if (!gMe || !gMe->pSprite)
        return shape;
    int top = 0;
    int bottom = 0;
    GetSpriteExtents(gMe->pSprite, &top, &bottom);
    shape.radius = gMe->pSprite->clipdist << 2;
    shape.ceilingDistance = std::max(0, (gMe->pSprite->z - top) / 4);
    shape.floorDistance = std::max(0, (bottom - gMe->pSprite->z) / 4);
    shape.footOffset = bottom - gMe->pSprite->z;
    shape.height = bottom - top;
    shape.eyeAbove = gMe->pPosture[gMe->lifeMode][gMe->posture].eyeAboveZ;
    return shape;
}

PhysicalPose supportAt(int x, int y, int referenceZ, int referenceSector)
{
    PhysicalPose pose;
    if (!gMe || !gMe->pSprite)
        return pose;
    int16_t sectorId = int16_t(validSector(referenceSector)
        ? referenceSector : gMe->pSprite->sectnum);
    updatesectorz(x, y, referenceZ, &sectorId);
    if (!validSector(sectorId) || inside(x, y, sectorId) != 1)
        return pose;

    const BodyShape body = liveBody();
    ClipGuard guard;
    ++gProbes;
    int ceilingZ = 0;
    int floorZ = 0;
    int floorHit = -1;
    settleSupport(x, y, referenceZ, sectorId, body, ceilingZ, floorZ,
                  floorHit);
    pose.x = x;
    pose.y = y;
    pose.z = floorZ - body.footOffset;
    pose.sector = sectorId;
    pose.supportZ = floorZ;
    pose.supportHit = floorHit;
    pose.valid = floorZ - ceilingZ >= body.height;
    return pose;
}

PhysicalPose livePose()
{
    PhysicalPose pose;
    if (!gMe || !gMe->pSprite || !gMe->pXSprite)
        return pose;
    const BodyShape body = liveBody();
    const spritetype *actor = gMe->pSprite;
    int ceilingZ = 0;
    int floorZ = 0;
    int floorHit = -1;
    {
        ClipGuard guard;
        ++gProbes;
        settleSupport(actor->x, actor->y, actor->z, actor->sectnum, body,
                      ceilingZ, floorZ, floorHit);
    }
    pose.x = actor->x;
    pose.y = actor->y;
    pose.z = actor->z;
    pose.sector = actor->sectnum;
    pose.supportZ = floorZ;
    pose.supportHit = floorHit;
    pose.valid = validSector(actor->sectnum);
    return pose;
}

WalkProbe walkTowards(const PhysicalPose &from, int dirX, int dirY,
                      int maxDistance, const BodyShape &body)
{
    WalkProbe probe;
    probe.end = from;
    if (!from.valid || !validSector(from.sector) || !gMe || !gMe->pSprite)
        return probe;
    const double length = std::sqrt(double(dirX) * dirX + double(dirY) * dirY);
    if (length < 1.0)
        return probe;

    const int step = std::max(32, body.radius);
    ClipGuard guard;

    PhysicalPose current = from;
    int travelled = 0;
    while (travelled < maxDistance)
    {
        const int span = std::min(step, maxDistance - travelled);
        const int deltaX = int(std::lround(double(dirX) / length * span));
        const int deltaY = int(std::lround(double(dirY) / length * span));
        if (!deltaX && !deltaY)
            break;

        int x = current.x;
        int y = current.y;
        int z = current.z;
        int sectorId = current.sector;
        const int wasSector = sectorId;
        ++gProbes;
        const int hit = int(ClipMove(&x, &y, &z, &sectorId, deltaX, deltaY,
            body.radius, body.ceilingDistance, body.floorDistance,
            CLIPMASK0));
        if (gTracing)
        {
            char text[256];
            std::snprintf(text, sizeof(text),
                "clipmove in=(%d,%d,%d) sec=%d d=(%d,%d) wd=%d cd=%d fd=%d "
                "-> ret=%d out=(%d,%d,%d) sec=%d floor_in=%d floor_out=%d",
                current.x, current.y, current.z, wasSector, deltaX, deltaY,
                body.radius, body.ceilingDistance, body.floorDistance,
                hit, x, y, z, sectorId,
                validSector(wasSector) ? getflorzofslope(int16_t(wasSector),
                    current.x, current.y) : 0,
                validSector(sectorId) ? getflorzofslope(int16_t(sectorId),
                    x, y) : 0);
            traceLine(text);
        }
        if (!validSector(sectorId))
            break;

        int ceilingZ = 0;
        int floorZ = 0;
        int floorHit = -1;
        settleSupport(x, y, z, sectorId, body, ceilingZ, floorZ, floorHit);
        if (gTracing)
        {
            char text[192];
            std::snprintf(text, sizeof(text),
                "zrange at=(%d,%d,%d) sec=%d wd=%d -> ceil=%d floor=%d "
                "florhit=%d was_sup=%d rise=%d allow=%d",
                x, y, z, sectorId, body.radius, ceilingZ, floorZ, floorHit,
                current.supportZ, current.supportZ - floorZ,
                stepAllowance(body));
            traceLine(text);
        }
        if (floorZ - ceilingZ < body.height)
            break; // the body does not fit here
        const int bottom = z + body.footOffset;
        // Ground running out is a fall; ground sloping away is not, and
        // neither is a step down. The difference is whether the thing
        // holding the body up is still the same thing -- a ramp descends
        // under the feet as fast as it likes and stays one surface -- and
        // then how far the new thing is below the old one.
        //
        // How far is the same number as how far up: a rise the engine takes
        // in one step is a step, and so is the identical drop. A body
        // Walking off a kerb is walking. It is still walking when the kerb
        // is taller than the body can climb back up, because coming down is
        // not the same act as going up and Blood does not treat it as one:
        // the floor stops being there, the body keeps the speed it had, and
        // it lands still walking. Measuring the way down with the way up is
        // what shut the bottom of the shaft on AGTST18 -- and the exit
        // switch with it -- over a ledge a player steps off without noticing.
        if (floorHit != current.supportHit
            && floorZ - bottom > descentAllowance())
        {
            probe.leftGround = true;
            break; // far enough to be hurt by: that is falling, not walking
        }
        // How far up a step may be is the engine's own number, and this is
        // where it has to be applied.
        //
        // GetZRange lifts a body onto whatever its hull touches, and a hull
        // touches the top of a step a body width before the clip line stops
        // the body reaching it. Taking that lift at face value stands the
        // body on top of something it has not climbed -- and then the step
        // after it is measured from up there, so a staircase reads as one
        // stride. A rise past what cliptestsector steps over is a wall,
        // whichever surface the range query happened to name.
        if (current.supportZ - floorZ > stepAllowance(body))
            break;

        const int advanced = planarDistance(current.x, current.y, x, y);
        const bool changedSupport = floorHit != current.supportHit;
        current.x = x;
        current.y = y;
        current.z = floorZ - body.footOffset;
        current.sector = sectorId;
        current.supportZ = floorZ;
        current.supportHit = floorHit;
        current.valid = true;
        travelled += advanced;
        if (changedSupport)
            probe.supportChanged = true;

        // Brushing a wall is not the end of a walk: Blood slides the body
        // along it and the player keeps going. What ends a walk is the body
        // stopping, which the progress test below is what notices.
        if (hit != 0)
        {
            probe.collided = true;
            probe.stoppedBy = hit;
        }
        if (advanced * 4 < span)
            break; // the engine stopped making progress along this line
    }
    probe.end = current;
    probe.travelled = travelled;
    return probe;
}

bool eyeCanSee(int x, int y, int z, int sectorId)
{
    if (!gMe || !gMe->pSprite || !validSector(sectorId))
        return false;
    return cansee(gMe->pSprite->x, gMe->pSprite->y, gMe->zView,
                  gMe->pSprite->sectnum, x, y, z, sectorId) != 0;
}

} // namespace bloodmap
