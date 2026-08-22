//-------------------------------------------------------------------------
// What Caleb can physically do, asked of the engine.
//
// The bot plans over the player's real capabilities: how wide he is, how
// high he jumps, how far he falls without being hurt, how far a swing
// reaches. Every one of those has an exact answer inside Blood, and this
// header is the only place the bot is allowed to work one out. Nothing here
// is fitted to a recording or rounded to a convenient number: each function
// either reads an engine value or replays the engine's own arithmetic.
//
// Anything that is a bot *policy* -- how much margin to leave, how much
// damage is acceptable, how close to stand before pressing something --
// belongs in bot.cpp, stated as a margin on top of one of these facts.
//-------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <cstdlib>

#include "build.h"
#include "../../actor.h"
#include "../../controls.h"
#include "../../db.h"
#include "../../gameutil.h"
#include "../../globals.h"
#include "../../player.h"
#include "../../trig.h"

namespace llmapper {
namespace capability {

// The body Blood clips with. MoveDude and playerProcess both build this from
// the live sprite before every ClipMove, so an animation that changes the
// player's extents changes this too.
struct Envelope
{
    bool known = false;
    int radius = 0;          // ClipMove `wd`
    int ceilingDistance = 0; // ClipMove `cd`
    int floorDistance = 0;   // ClipMove `fd`
    int top = 0;
    int bottom = 0;

    // The physical height of the body, not a reconstruction from the two
    // quarter-scaled collision distances.
    int height() const { return bottom - top; }
};

inline Envelope playerEnvelope()
{
    Envelope body;
    if (!gMe || !gMe->pSprite)
        return body;
    spritetype *pSprite = gMe->pSprite;
    GetSpriteExtents(pSprite, &body.top, &body.bottom);
    body.known = true;
    body.radius = pSprite->clipdist << 2;
    body.ceilingDistance = (pSprite->z - body.top) / 4;
    body.floorDistance = (body.bottom - pSprite->z) / 4;
    return body;
}

// How far the sprite's bottom sits below its origin. MoveDude measures
// airborne height from the bottom, so any prediction of the player's motion
// has to carry the same offset.
inline int playerFootOffset()
{
    const Envelope body = playerEnvelope();
    return body.known ? body.bottom - gMe->pSprite->z : 0;
}

// The floor MoveDude will actually settle the player onto this frame.
// GetZRange sweeps the whole footprint, and MoveDude widens it by 16 for a
// player, so a single getflorzofslope at the sprite's centre is not the same
// question and disagrees at every ledge.
inline int playerFloorZ()
{
    if (!gMe || !gMe->pSprite)
        return 0;
    int ceilZ = 0, ceilHit = 0, floorZ = 0, floorHit = 0;
    GetZRange(gMe->pSprite, &ceilZ, &ceilHit, &floorZ, &floorHit,
              (gMe->pSprite->clipdist << 2) + 16, CLIPMASK0,
              PARALLAXCLIP_CEILING | PARALLAXCLIP_FLOOR);
    return floorZ;
}

// The air drag actProcessSprites applies to every dude, every frame, before
// MoveDude runs: velocity is pulled a fixed fraction of the way towards the
// sector's wind, in all three axes.  Small -- a five-hundredth per frame --
// but it is on every frame of a flight, and leaving it out makes a predicted
// arc land long.
struct AirDrag
{
    int factor = 128;  // actAirDrag's a2; 5376 underwater
    int windX = 0;
    int windY = 0;
};

inline AirDrag playerAirDrag()
{
    AirDrag drag;
    if (!gMe || !gMe->pSprite)
        return drag;
    const int nSector = gMe->pSprite->sectnum;
    if (nSector < 0 || nSector >= numsectors)
        return drag;
    const int extra = sector[nSector].extra;
    if (extra <= 0)
        return drag;
    const XSECTOR &xsect = xsector[extra];
    if (xsect.Underwater)
        drag.factor = 5376;
    if (xsect.windVel && (xsect.windAlways || xsect.busy))
    {
        int wind = xsect.windVel << 12;
        if (!xsect.windAlways && xsect.busy)
            wind = mulscale16(wind, xsect.busy);
        drag.windX = mulscale30(wind, Cos(xsect.windAng));
        drag.windY = mulscale30(wind, Sin(xsect.windAng));
    }
    return drag;
}

// Is the player in water, goo, or a sector with Depth?  MoveDude changes the
// vertical acceleration in all three, and the model does not follow it there.
inline bool playerInDepth()
{
    if (!gMe || !gMe->pSprite)
        return false;
    const int nSector = gMe->pSprite->sectnum;
    if (nSector < 0 || nSector >= numsectors)
        return false;
    const int extra = sector[nSector].extra;
    if (extra > 0 && (xsector[extra].Underwater || xsector[extra].Depth))
        return true;
    const int upper = gUpperLink[nSector];
    const int lower = gLowerLink[nSector];
    if (upper >= 0 && (sprite[upper].type == kMarkerUpWater
                       || sprite[upper].type == kMarkerUpGoo))
        return true;
    if (lower >= 0 && (sprite[lower].type == kMarkerLowWater
                       || sprite[lower].type == kMarkerLowGoo))
        return true;
    return false;
}

inline const POSTURE *playerPosture(int posture)
{
    return gMe ? &gMe->pPosture[gMe->lifeMode][posture] : nullptr;
}

// Vertical velocity a jump starts with. Jump Boots change it, so this is a
// question about the current state, not a constant.
//
// Always the standing posture. ProcessInput only jumps out of `default:`,
// which is standing, and crouch is released to standing before a jump can
// happen at all; the other postures reuse normalJumpZ for something else
// entirely -- swimming adds it as a paddle -- and crouch's is positive, so
// reading the live posture makes the bot believe mid-crouch that it cannot
// jump, and every ledge in the level appears to close and reopen around the
// calibration.
inline int playerJumpImpulse()
{
    const POSTURE *pPosture = playerPosture(kPostureStand);
    if (!pPosture)
        return 0;
    if (packItemActive(gMe, kPackJumpBoots) && pPosture->pwupJumpZ != 0)
        return pPosture->pwupJumpZ;
    return pPosture->normalJumpZ;
}

// The engine's own vertical integration, one game frame at a time. MoveDude
// advances z by zvel>>8, then adds gravity to both.
struct VerticalStep
{
    int z = 0;
    int zvel = 0;
    int airFactor = 128;

    void advance()
    {
        zvel -= mulscale16(zvel, airFactor);
        z += zvel >> 8;
        z += ((kDudeGravity * 4) / 2) >> 8;
        zvel += kDudeGravity;
    }
};

// Highest point a jump taken now reaches, as a positive rise. Replayed
// rather than approximated: a scale-shift of the impulse is out by a quarter.
inline int playerJumpApex()
{
    VerticalStep step;
    step.zvel = playerJumpImpulse();
    if (step.zvel >= 0)
        return 0;
    int highest = 0;
    for (int frame = 0; frame < 240 && step.zvel < 0; ++frame)
    {
        step.advance();
        highest = std::min(highest, step.z);
    }
    return -highest;
}

// How many frames a jump spends off the ground before it comes back down to
// `rise` above (negative) or below (positive) the take-off floor.
inline int playerJumpAirFrames(int rise)
{
    VerticalStep step;
    step.zvel = playerJumpImpulse();
    for (int frame = 1; frame <= 240; ++frame)
    {
        step.advance();
        if (frame > 2 && step.z >= rise && step.zvel > 0)
            return frame;
    }
    return 0;
}

// Impact velocity at which MoveDude's landing damage becomes non-zero.
// nDamage = mulscale30(v, v) - kFallDamageFloor, so anything at or under this
// hurts nobody at all.
inline int playerHarmlessImpactVelocity()
{
    int velocity = 0;
    for (int bit = 30; bit >= 0; --bit)
    {
        const int candidate = velocity | (1 << bit);
        if (mulscale30(candidate, candidate) <= kFallDamageFloor)
            velocity = candidate;
    }
    return velocity;
}

// Damage a fall from `drop` units above the landing surface would do, on
// Blood's 16x health scale. Zero means the landing is free.
inline int playerFallDamage(int drop)
{
    if (drop <= 0)
        return 0;
    VerticalStep step;
    for (int frame = 0; frame < 240; ++frame)
    {
        step.advance();
        if (step.z >= drop)
            break;
    }
    return std::max(0, mulscale30(step.zvel, step.zvel) - kFallDamageFloor);
}

// Tallest drop that costs no health.
inline int playerHarmlessDropHeight()
{
    const int limit = playerHarmlessImpactVelocity();
    VerticalStep step;
    int safe = 0;
    for (int frame = 0; frame < 240; ++frame)
    {
        VerticalStep next = step;
        next.advance();
        if (next.zvel > limit)
            break;
        step = next;
        safe = step.z;
    }
    return safe;
}

// Largest forward/strafe the control layer can produce. ProcessInput scales
// the posture's acceleration by it, so it is the full-throttle input.
inline int playerMoveInputMax() { return kMaxMoveInput; }

// Largest upward floor change which MoveDude's grounded player can cross.
// ClipMove does not compare the floor delta with CLIPCURBHEIGHT alone.  Its
// portal predicate compares the sprite origin with the destination floor
// after applying MoveDude's `fd = (bottom-z)/4`; CLIPCURBHEIGHT only exempts
// still smaller curbs from that test.  For a grounded player the exact last
// accepted rise is therefore (bottom-origin)-fd, or CLIPCURBHEIGHT when the
// body is unusually short.  Reading the live envelope preserves animation
// and player-scale changes and reproduces the predicate used by ClipMove.
inline int playerStepHeight()
{
    const Envelope body = playerEnvelope();
    if (!body.known || !gMe || !gMe->pSprite)
        return CLIPCURBHEIGHT;
    const int footOffset = body.bottom - gMe->pSprite->z;
    return std::max(CLIPCURBHEIGHT, footOffset - body.floorDistance);
}

// Pitch limits ProcessInput clamps the view to.
inline int playerLookUpLimit() { return kLookUpLimit; }
inline int playerLookDownLimit() { return kLookDownLimit; }

// How far a pitchfork swing actually reaches: FirePitchfork fires
// kVectorTine, and actFireVector stops the trace at the vector's maxDist.
inline int playerPitchforkReach() { return gVectorData[kVectorTine].maxDist; }

// Health a fresh Caleb starts with, on Blood's 16x scale.
inline int playerStartHealth()
{
    return gMe && gMe->pDudeInfo ? (gMe->pDudeInfo->startHealth << 4) : 0;
}

//-------------------------------------------------------------------------
// One frame of the player's horizontal and vertical motion.
//
// This is ProcessInput's acceleration followed by MoveDude's move, gravity
// and drag, in that order and with the engine's constants. It leaves out
// only clipping, so it predicts flight through open air exactly and is not a
// substitute for a collision probe anywhere else.
//-------------------------------------------------------------------------
struct MotionState
{
    int x = 0;
    int y = 0;
    int z = 0; // sprite origin, as the engine stores it
    int xvel = 0;
    int yvel = 0;
    int zvel = 0;
};

// XSPRITE::height as MoveDude recomputes it: measured from the sprite's
// bottom, not its origin.
inline int airborneHeight(const MotionState &state, int floorZ, int footOffset)
{
    return ClipLow(floorZ - (state.z + footOffset), 0) >> 8;
}

inline void stepPlayerMotion(MotionState &state, int forwardInput, int angle,
                             int floorZ, int frontAccel, int footOffset,
                             const AirDrag &air)
{
    // ProcessInput: acceleration, scaled down the further off the floor the
    // player is, and gone entirely once fully airborne.
    const int height = airborneHeight(state, floorZ, footOffset);
    if (forwardInput && height < kDudeAirborneHeight)
    {
        int speed = 0x10000;
        if (height > 0)
            speed -= divscale16(height, kDudeAirborneHeight);
        int forward = mulscale8(frontAccel, forwardInput);
        if (height)
            forward = mulscale16(forward, speed);
        state.xvel += mulscale30(forward, Cos(angle));
        state.yvel += mulscale30(forward, Sin(angle));
    }

    // actProcessSprites: air drag on all three axes, before MoveDude.
    state.xvel += mulscale16(air.windX - state.xvel, air.factor);
    state.yvel += mulscale16(air.windY - state.yvel, air.factor);
    state.zvel -= mulscale16(state.zvel, air.factor);

    // MoveDude: the horizontal move, then the vertical one.  The three
    // vertical parts are separately gated in the engine and have to stay
    // that way: the velocity always moves the sprite, gravity is skipped
    // only while the body is already resting on the floor, and the landing
    // is a third test made after both.
    state.x += state.xvel >> 12;
    state.y += state.yvel >> 12;
    if (state.zvel)
        state.z += state.zvel >> 8;
    if (state.z + footOffset < floorZ)
    {
        state.z += ((kDudeGravity * 4) / 2) >> 8;
        state.zvel += kDudeGravity;
    }
    if (state.z + footOffset >= floorZ)
    {
        // MoveDude sets the sprite down on the floor, and over level ground
        // with no floor velocity actFloorBounceVector takes the whole
        // downward velocity out.  A body moving up is left alone.
        state.z = floorZ - footOffset;
        if (state.zvel > 0)
            state.zvel = 0;
    }

    // MoveDude's drag, applied only while actually moving and with the height
    // recomputed after the vertical step.
    if (state.xvel || state.yvel)
    {
        const int settled = airborneHeight(state, floorZ, footOffset);
        if (settled < kDudeAirborneHeight)
        {
            int drag = gDudeDrag;
            if (settled > 0)
                drag -= scale(gDudeDrag, settled, kDudeAirborneHeight);
            state.xvel -= mulscale16r(state.xvel, drag);
            state.yvel -= mulscale16r(state.yvel, drag);
            if (approxDist(state.xvel, state.yvel) < 0x1000)
                state.xvel = state.yvel = 0;
        }
    }
}

// Top running speed in world units per frame, where the posture's
// acceleration and the dude drag balance.
inline int playerRunSpeed()
{
    const POSTURE *stand = playerPosture(kPostureStand);
    if (!stand)
        return 0;
    const int accel = mulscale8(stand->frontAccel, playerMoveInputMax());
    const AirDrag air = playerAirDrag();
    int velocity = 0;
    for (int frame = 0; frame < 128; ++frame)
    {
        velocity += accel;
        velocity -= mulscale16(velocity, air.factor);
        velocity -= mulscale16r(velocity, gDudeDrag);
    }
    return velocity >> 12;
}

// Ground needed to get there from a standing start.
inline int playerRunUpDistance()
{
    const POSTURE *stand = playerPosture(kPostureStand);
    if (!stand)
        return 0;
    const int accel = mulscale8(stand->frontAccel, playerMoveInputMax());
    const AirDrag air = playerAirDrag();
    const int top = playerRunSpeed();
    const int target = top - top / 16;
    int velocity = 0;
    int travelled = 0;
    for (int frame = 0; frame < 128; ++frame)
    {
        velocity += accel;
        velocity -= mulscale16(velocity, air.factor);
        // MoveDude advances XY before applying grounded dude drag.  Keep the
        // distance clock in that exact order: jump planning uses the first
        // frame whose movement crosses a concrete takeoff pose.
        travelled += velocity >> 12;
        velocity -= mulscale16r(velocity, gDudeDrag);
        if ((velocity >> 12) >= target)
            break;
    }
    return travelled;
}

// Velocity produced by a real ground run of the requested length.  The
// return value uses Blood's internal 20.12 horizontal-velocity units so it
// can seed the same airborne motion replay used by navigation and execution.
inline int playerRunVelocityForDistance(int distance)
{
    const POSTURE *stand = playerPosture(kPostureStand);
    if (!stand || distance <= 0)
        return 0;
    const int accel = mulscale8(stand->frontAccel, playerMoveInputMax());
    const AirDrag air = playerAirDrag();
    int velocity = 0;
    int travelled = 0;
    for (int frame = 0; frame < 128 && travelled < distance; ++frame)
    {
        velocity += accel;
        velocity -= mulscale16(velocity, air.factor);
        travelled += velocity >> 12;
        velocity -= mulscale16r(velocity, gDudeDrag);
    }
    return velocity;
}

// How much further the player slides after letting go, under drag alone.
inline int playerCoastDistance(int speed)
{
    const AirDrag air = playerAirDrag();
    int velocity = speed << 12;
    int travelled = 0;
    for (int frame = 0; frame < 128; ++frame)
    {
        velocity -= mulscale16(velocity, air.factor);
        velocity -= mulscale16r(velocity, gDudeDrag);
        if (velocity < 0x1000)
            break;
        travelled += velocity >> 12;
    }
    return travelled;
}

} // namespace capability
} // namespace llmapper
