//-------------------------------------------------------------------------
// Low-level Blood physical queries.
//
// Every predicate here is either an engine call or a direct transcription of
// what MoveDude does to the player each frame. Nothing invents a second set
// of movement rules on top of the engine's.
//-------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../semantic/semantic_world.h"

namespace bloodmap {

// The live player hull, in exactly the terms ClipMove and GetZRange take.
struct BodyShape
{
    int radius = 128;        // ClipMove walldist: clipdist<<2
    int ceilingDistance = 0; // ClipMove ceildist: (z - top)/4
    int floorDistance = 0;   // ClipMove flordist: (bottom - z)/4
    int footOffset = 0;      // bottom - z
    int height = 0;          // bottom - top
    int eyeAbove = 0;        // posture eye height above the sprite origin
};

BodyShape liveBody();

// A written record of what the engine was actually asked and what it said.
//
// Diagnostic only -- nothing reads it to decide anything. It exists because
// a disagreement between this model and Blood cannot be settled by reading
// either of them; the numbers that crossed the boundary have to be seen.
void traceMotion(bool on);
bool tracingMotion();
void traceLine(const std::string &line);
const std::vector<std::string> &motionTrace();
void clearMotionTrace();

// A physical standing pose. supportHit is the engine's own answer to "what
// is holding this body up" -- GetZRange's florhit, which names a sector
// floor or a specific solid sprite.
struct PhysicalPose
{
    int x = 0;
    int y = 0;
    int z = 0;          // sprite origin, as Blood stores it
    int sector = -1;
    int supportZ = 0;
    int supportHit = -1;
    bool valid = false;
};

bool validSector(int sectorId);
bool validWall(int wallId);
bool validSprite(int spriteId);
int planarDistance(int x1, int y1, int x2, int y2);
int64_t planarDistanceSquared(int x1, int y1, int x2, int y2);

// Build's own idea of a floor difference that is still continuous ground.
int curbHeight();

// Where the body would come to rest at xy, entering from referenceZ. The
// entry height selects which support layer is found, so a caller standing on
// a raised surface keeps standing on it.
PhysicalPose supportAt(int x, int y, int referenceZ, int referenceSector);

PhysicalPose livePose();

struct WalkProbe
{
    PhysicalPose end;
    int travelled = 0;
    bool collided = false;    // the engine reported the body touching
    bool supportChanged = false;
    bool leftGround = false;  // continuing would have been a fall
    // What the engine said the body ran into, in its own encoding, or -1.
    // ClipMove names the wall or the sprite that stopped the move, and a way
    // that is shut by a thing is shut by a thing that can be looked up.
    int stoppedBy = -1;
};

// Split the engine's answer for what stopped a move. Exactly one of the two
// comes back set; both are -1 when nothing stopped it.
void splitObstacle(int hit, int &wallId, int &spriteId);

// Walk from a pose toward a point, the way the player walks: repeated
// ClipMove with the live hull, settling on whatever surface the engine
// reports underneath afterwards. Stops at the first thing the engine says
// the body cannot walk through, off, or under.
WalkProbe walkTowards(const PhysicalPose &from, int dirX, int dirY,
                      int maxDistance, const BodyShape &body);

bool eyeCanSee(int x, int y, int z, int sectorId);

// How far up the engine will take this body in one step.
//
// This is cliptestsector's own arithmetic and not a number chosen here. A
// neighbouring floor stops the body when it stands more than a curb above
// this one and the body is not already clear of it:
//
//     daz2 < daz - CLIPCURBHEIGHT  &&  posz >= daz2 - (flordist - 1)
//
// posz being the body's own z, which sits footOffset above whatever holds it
// up. Rearranged, the rise a body walking on this floor gets over is its
// footOffset less its flordist, less one.
int stepAllowance(const BodyShape &body);
// How far down one stride may take a body and still be a stride: the tallest
// ledge Blood lets a body walk off and land from for nothing. Derived from
// kDudeGravity and kFallDamageFloor, which is where the engine itself puts
// the line between walking down and falling.
int descentAllowance();

// Is the body, as the engine has placed it, in this region?
//
// Which space a body is in is where it is, not what is under it. Blood says
// so twice over: `updatesector` answers by plan containment alone, and
// `getzrange` then stands the body on the highest floor its hull reaches --
// which is regularly the floor next door. A body walking a corridor
// narrower than itself is held up the whole way by the room beside it, and
// is nonetheless in the corridor. Measured on AGTST18: the player crossing
// sector 3 stands at sector 27's floor height for every tick of it.
//
// So the only thing a support can say about being here is negative: a
// support *below* this region's floor means the pose fell through to a
// layer underneath. There is deliberately no allowance in the other
// direction, and in particular no body-height rule -- `getzrange` already
// declines to name a floor the body could not legally be on, because it
// consults `cliptestsector` before it descends into a neighbour.
bool bodyRestsOn(const semantic::Region &region, const PhysicalPose &pose,
                 const BodyShape &body);

// Is the body standing on this region's own floor?
//
// A stricter question than the one above, and asked for a different reason.
// Where a body *is* has to admit the hull straddling a join, because that is
// where bodies stand. Where a crossing *sets off from* must not: a pose the
// engine has lifted onto the step beside it is a pose halfway up something,
// and a crossing checked from there is a crossing that begins with a climb
// nobody checked. Chain two of those and the model walks up a staircase one
// storey at a time while the body, on the floor, walks into the bottom step.
bool bodyStandsOn(const semantic::Region &region, const PhysicalPose &pose);

// What a fall of this height costs, by Blood's own arithmetic: its gravity
// integrated the way MoveDude integrates it, then the landing damage
// actFloorBounceVector and kFallDamageFloor work out for the velocity that
// reaches. Zero means the body walks away from it.
//
// No height limit is chosen here. Blood lets a body fall a long way without
// being hurt, and how far is the engine's answer, not a number of ours.
int fallDamage(int drop);

int probeCount();

} // namespace bloodmap
