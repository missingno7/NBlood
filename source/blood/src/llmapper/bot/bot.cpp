//-------------------------------------------------------------------------
// LLMapper autonomous Blood playtest bot.
//
// This is deliberately a small in-process vertical slice. It keeps a
// discovered graph, plans only over observations, and emits NDJSON telemetry
// plus a trajectory while driving the ordinary GINPUT path.
//-------------------------------------------------------------------------
#include "bot.h"
#include "nav_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "build.h"
#include "../../common_game.h"
#include "../../config.h"
#include "../../db.h"
#include "../../demo.h"
#include "../../gameutil.h"
#include "../../globals.h"
#include "../../levels.h"
#include "../../view.h"
#include "../../network.h"
#include "../../player.h"
#include "../../trig.h"
#include "../../triggers.h"

enum TraversalCapability
{
    kTraversalUnknown,
    kTraversalWalkable,
    kTraversalJumpable,
    kTraversalCrouchable,
    kTraversalDropSafe,
    kTraversalInteractionBlocked,
    kTraversalCurrentlyUnavailable,
};

// A planner label is not an input.  These states are deliberately small and
// exist only for locomotion that Blood requires us to perform explicitly.
enum JumpExecutionState
{
    kJumpInactive,
    kJumpApproachTakeoff,
    kJumpAlign,
    kJumpTakeoff,
    kJumpAirborne,
};

using llmapper::NavEdgeMode;
using llmapper::kNavWalk;
using llmapper::kNavStep;
using llmapper::kNavJump;
using llmapper::kNavCrouch;
using llmapper::kNavDrop;
using llmapper::kNavInteraction;
using llmapper::kNavBlocked;
using llmapper::NavWaypoint;
using llmapper::NavLink;
using llmapper::NavCell;
using llmapper::NavRouteStep;
using llmapper::NavEdgeFailure;
using llmapper::DerivedFrontier;
using llmapper::Boundary;
using llmapper::InvestigateRecord;
using llmapper::CombatSituation;
using llmapper::CombatDecision;
using llmapper::CombatTactic;
using llmapper::kCombatNone;
using llmapper::kCombatRanged;
using llmapper::kCombatRetreat;
using llmapper::kCombatMelee;
using llmapper::kFrontierOpen;
using llmapper::kFrontierBlocked;
using llmapper::TraversalResult;
using llmapper::kTraverseDirect;
using llmapper::kTraverseStep;
using llmapper::kTraverseJump;
using llmapper::kTraverseDropSafe;
using llmapper::kTraverseUseableBlocker;
using llmapper::kTraverseSolidBlocker;
using llmapper::kTraverseDangerous;
using llmapper::kTraverseNoFit;
using llmapper::navEdgeModeName;
typedef llmapper::NavWaypoint LocalWaypoint;

namespace
{
constexpr int kDefaultTimeoutSeconds = 30 * 60;
constexpr int kDefaultStallSeconds = 45;
constexpr int kObservationPeriod = 4;
constexpr int kTrajectoryPeriod = 8;
constexpr int kMovementStuckTicks = 2 * kTicsPerSec;
constexpr int kJumpCooldownTicks = 2 * kTicsPerSec;
constexpr int kMaxJumpAttemptsPerTarget = 3;
constexpr int kJumpTakeoffRange = 1536;
constexpr int kJumpActionTimeoutTicks = 2 * kTicsPerSec;
// Widest the player is.  An opening narrower than this cannot be walked
// through however inviting the geometry looks, so it must not be offered as
// a route -- the bot was planning between columns it could never fit past.
static int playerPassageWidth()
{
    const int radius = gMe && gMe->pSprite ? (gMe->pSprite->clipdist << 2) : 128;
    return radius * 2;
}
constexpr int kMaxWalkableStep = 4096;
constexpr int kActionScanRange = 1024;
constexpr int kActionApproachRange = 2048;
constexpr int kUseStopRange = kActionApproachRange;
constexpr int kInteractionTimeoutTicks = 6 * kTicsPerSec;
// Every committed objective is bounded.  An objective that stops making
// progress, or simply runs too long, releases the bot back to exploration
// and is suppressed for a growing cooldown instead of owning the run.
constexpr int kObjectiveStallTicks = 5 * kTicsPerSec;
constexpr int kObjectiveHardTicks = 60 * kTicsPerSec;
constexpr int kObjectiveProgressEpsilon2 = 192 * 192;
constexpr int kSuppressionBaseTicks = 25 * kTicsPerSec;
// Navigation grid resolution in Build units (1 << 8 = 256), a little wider
// than the player's clip radius so an ordinary corridor keeps several cells.
constexpr int kNavGridShift = 8;
// How long a route the bot just opened stays the most interesting thing in
// the level.  Long enough to walk to it, short enough that a stale opening
// does not outrank live exploration.
constexpr int kOpenedRouteWindowTicks = 12 * kTicsPerSec;
// How long the player may fail to close on one route waypoint before that
// step is treated as genuinely unusable rather than merely awkward.
constexpr int kWaypointStallTicks = 2 * kTicsPerSec;
// Blood health is scaled by 16; this is roughly a quarter of a fresh start.
constexpr int kCriticalHealth = 25 * 16;
// If Caleb is alive, the level is unfinished, and nothing is deliberately
// waiting on moving geometry, he should be moving.  Longer than this and
// something in the execution layer is deadlocked.
constexpr int kStationaryLimitTicks = 2 * kTicsPerSec;
// How close a hostile must be before closing to melee is worth interrupting
// exploration for.  Chasing a dude across the level is not combat, it is a
// distraction that costs the whole run.
constexpr int kMeleeEngageRange = 2048;
// How far the bot credits itself with having seen local space, and how far
// above the floor the sight test is taken (roughly knee height, so a low
// switch on a far wall still counts as observable).
constexpr int kCoverageSightRange = 6144;
constexpr int kCoverageEyeOffset = 4096;
// Consecutive decisions in which navigation produced no usable movement
// before the destination is released.  The objective budget and the
// stationary watchdog are the primary bounds; this is a backstop.
constexpr int kNavigationFailureLimit = 40;
// Long enough for Blood to switch the player to the crouch sequence and for
// one observation to sample the resulting body extents.
constexpr int kCrouchCalibrationTicks = kTicsPerSec;
// Extra route cost charged for a connection that is currently shut but has
// a known mechanism, so an open way round is preferred when one exists.
constexpr int kGatedHopCost = 4;
// Minimum gap between deliberate activations of the same mechanism, so a
// reusable door is not restarted before it has finished moving.
constexpr int kReactivationCooldownTicks = 5 * kTicsPerSec;
// How many times the bot may break out of a detected loop before the run is
// declared genuinely stuck.
constexpr int kMaxLoopBreaks = 6;
// Deliberate activations of one mechanism before the bot accepts that
// pressing it again is not going to help.
constexpr int kMaxActivationAttempts = 3;
// Longest straight shortcut route smoothing may create.  Long enough to
// remove grid staircases, short enough that it cannot skip a corner.
constexpr int kRouteSmoothingSpan = 2048;
// Upper bound on waiting for one mechanism to finish travelling.  Blood
// states the real figure per door; this only guards against a mechanism
// that never settles.
constexpr int kMaxDoorWaitTicks = 20 * kTicsPerSec;
// Grace period after an accepted Use during which the bot keeps watching
// for the world to change, before deciding nothing came of it.
constexpr int kMechanismSettleTicks = 3 * kTicsPerSec;
// How fast the view returns to level while simply travelling.
constexpr int kLookLevelRate = 24;
// Damage arriving this close together counts as one continuing source, and
// this much health lost to it means the ground itself is the problem.
constexpr int kDamageBurstWindow = kTicsPerSec;
constexpr int kDamageBurstSevere = 200;
constexpr int kHazardAvoidTicks = 30 * kTicsPerSec;
// Probes at a surface the engine will not name as a target before the bot
// goes and tries a different surface of the same mechanism.
constexpr int kSurfaceRetryProbes = 8;
// Ground that has to be covered before the bot counts as having closed on a
// target.  Linear, in world units: a quarter of a navigation grid square.
constexpr int kProgressStep = 24;
// How far off the bot's heading a target may be and still be worth a shot
// taken in passing.  256 of 2048 is 45 degrees.
constexpr int kOpportunisticAimCone = 256;
// Pitchfork reach, used when a breakable obstacle must be hit by hand.
constexpr int kMeleeReach = 1024;
constexpr int kLookUpLimit = 289;
constexpr int kLookDownLimit = -347;

enum ObjectKind
{
    kObjectEnemy,
    kObjectKey,
    kObjectInteractive,
    kObjectPickup,
};

struct VisibleObject
{
    int sprite = -1;
    int sector = -1;
    int type = 0;
    int x = 0;
    int y = 0;
    int z = 0;
    ObjectKind kind = kObjectPickup;
};

struct Portal
{
    int wall = -1;
    int from = -1;
    int to = -1;
    int x = 0;
    int y = 0;
    int z = 0;
    int floorZ = 0;
    int ceilingZ = 0;
    int fromFloorZ = 0;
    int fromCeilingZ = 0;
    int toFloorZ = 0;
    int toCeilingZ = 0;
    int crouchClearance = 0;
    int jumpRiseLimit = 0;
    int dropLimit = 0;
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
    int key = 0;
    int mechanismTx = 0;
    int openingWidth = 0;
    int floorDelta = 0;
    int clearance = 0;
    bool wallPush = false;
    bool shootable = false;
    bool sectorPush = false;
    bool sectorPushCurrent = false;
    bool directUse = false;
    bool visible = false;
    bool localGeometry = false;
    bool walkable = false;
    bool jumpable = false;
    bool crouchable = false;
    bool dropSafe = false;
    bool traversable = false;
    bool locked = false;
    bool interactionAffordance = false;
    bool blockedBySprite = false;
    bool currentlyAvailable = false;
    TraversalCapability capability = kTraversalUnknown;
    int unavailableReason = 0;
    int wallState = -1;
    int wallBusy = 0;
    int sectorState = -1;
    int sectorBusy = 0;
    int interactionTopZ = 0;
    int interactionBottomZ = 0;
    bool interactionCrouch = false;
};

enum InteractionKind
{
    kInteractionWall,
    kInteractionSector,
    kInteractionSprite,
};

struct InteractionCandidate
{
    InteractionKind kind = kInteractionWall;
    int id = -1;
    int fromSector = -1;
    int targetSector = -1;
    int x = 0;
    int y = 0;
    int z = 0;
    int key = 0;
    bool locked = false;
    bool reversible = false;
    Portal target;
};

struct Observation
{
    int tick = 0;
    int sector = -1;
    int x = 0;
    int y = 0;
    int z = 0;
    int angle = 0;
    int health = 0;
    bool exitHere = false;
    int localPortalCount = 0;
    int localJumpableCount = 0;
    int localMaxRise = 0;
    int localMaxDrop = 0;
    int localClearance = 0;
    int playerCeilingDistance = 0;
    int playerFloorDistance = 0;
    int localSectorExtra = 0;
    int localSectorState = 0;
    int localSectorBusy = 0;
    int playerZVelocity = 0;
    std::vector<int> visibleSectors;
    std::vector<Portal> portals;
    std::vector<InteractionCandidate> interactions;
    std::vector<VisibleObject> objects;
};

struct MovementProbe
{
    bool reachable = false;
    int wall = -1;
    int hit = 0;
    int x = 0;
    int y = 0;
    int sector = -1;
};

static int wrapAngle(int angle)
{
    angle &= 2047;
    return angle;
}

static int angleDelta(int target, int current)
{
    int delta = wrapAngle(target) - wrapAngle(current);
    if (delta > 1024)
        delta -= 2048;
    if (delta < -1024)
        delta += 2048;
    return delta;
}

static int distance2(int x1, int y1, int x2, int y2)
{
    const int64_t dx = int64_t(x2) - x1;
    const int64_t dy = int64_t(y2) - y1;
    return int(std::min<int64_t>(INT32_MAX, dx * dx + dy * dy));
}

static bool inRange(int value, int low, int high)
{
    return value >= low && value < high;
}

static bool isKeyType(int type)
{
    return type >= kItemKeyBase && type < kItemKeyMax;
}

static bool isEnemyType(int type)
{
    return type >= kDudeBase && type < kDudeMax && type != kDudeHand;
}

static bool validXSprite(int extra)
{
    return extra > 0 && extra < kMaxXSprites;
}

static int playerJumpRiseLimit()
{
    if (!gMe)
        return 0;
    const POSTURE &stand = gMe->pPosture[gMe->lifeMode][kPostureStand];
    // Blood applies normalJumpZ as the initial vertical velocity. This
    // scale-derived bound is deliberately conservative and replaces a
    // generic "stuck means jump" fallback.
    return std::max(kMaxWalkableStep, std::abs(stand.normalJumpZ) >> 6);
}

static void playerCollisionDistances(int &ceilingDistance, int &floorDistance);

static int playerBodyClearance()
{
    int ceilingDistance = 0;
    int floorDistance = 0;
    playerCollisionDistances(ceilingDistance, floorDistance);
    return std::max(1, (ceilingDistance + floorDistance) * 4);
}

// Smallest body envelope actually observed while crouched.  Blood swaps the
// player to the crouch animation sequence, and GetSpriteExtents follows that
// sequence, so the crouched collision body really is shorter than the
// standing one -- but only measurable once the bot has crouched at least
// once.  Until then, fall back to the engine's own posture table.
static int gObservedCrouchClearance = 0;
static int gStandingClearance = 0;

static int playerStandingClearance()
{
    return gStandingClearance > 0 ? gStandingClearance : playerBodyClearance();
}

static int playerCrouchClearance()
{
    const int standing = playerStandingClearance();
    if (gObservedCrouchClearance > 0)
        return std::min(standing, gObservedCrouchClearance);
    if (!gMe)
        return standing;
    const POSTURE &stand = gMe->pPosture[gMe->lifeMode][kPostureStand];
    const POSTURE &crouch = gMe->pPosture[gMe->lifeMode][kPostureCrouch];
    const int drop = std::max(0, stand.eyeAboveZ - crouch.eyeAboveZ);
    return std::max(standing / 4, standing - drop);
}

static void playerCollisionDistances(int &ceilingDistance, int &floorDistance)
{
    ceilingDistance = 64 << 4;
    floorDistance = 64 << 4;
    if (!gMe || !gMe->pSprite)
        return;

    // Keep copied-state probes identical to playerProcess().  The engine
    // derives these distances from the player's actual sprite extents; the
    // posture eye height is not the collision height.
    int top = gMe->pSprite->z;
    int bottom = gMe->pSprite->z;
    GetSpriteExtents(gMe->pSprite, &top, &bottom);
    ceilingDistance = std::max(0, (gMe->pSprite->z - top) / 4);
    floorDistance = std::max(0, (bottom - gMe->pSprite->z) / 4);
}

static int lookAngleForTarget(int eyeZ, int targetZ, int horizontal)
{
    const double angle = std::atan2(double(eyeZ - targetZ), double(std::max(1, horizontal)))
        * 1024.0 / 3.14159265358979323846;
    return std::max(kLookDownLimit, std::min(kLookUpLimit, int(std::lround(angle))));
}

// Does this sector contain a floor-aligned sprite the player can stand on?
// Asked first because the answer is almost always no, and the surface query
// below is expensive: everywhere else the sector floor is the whole story
// and nothing changes.
static bool sectorHasFloorSprite(int sectorId)
{
    if (!inRange(sectorId, 0, numsectors))
        return false;
    for (int nSprite = headspritesect[sectorId]; nSprite >= 0;
         nSprite = nextspritesect[nSprite])
    {
        const spritetype &record = sprite[nSprite];
        if (!(record.cstat & CSTAT_SPRITE_BLOCK))
            continue;
        const int alignment = record.cstat & CSTAT_SPRITE_ALIGNMENT_MASK;
        if (alignment == CSTAT_SPRITE_ALIGNMENT_FLOOR
            || alignment == CSTAT_SPRITE_ALIGNMENT_SLOPE)
            return true;
    }
    return false;
}

// What the player would be standing on here, as the engine resolves it.
//
// A bridge made of floor sprites is a floor: reading the sector below it
// instead sees the pit it spans, so the crossing looks like a fatal drop and
// the route over it does not exist.  The probe is deliberately narrow -- the
// surface under this point, not any surface within arm's reach -- so the
// mesh does not place a cell in mid-air off the end of a plank.
static int standingFloorZ(int sectorId, int x, int y)
{
    const int sectorFloor = inRange(sectorId, 0, numsectors)
        ? getflorzofslope(sectorId, x, y) : 0;
    if (!sectorHasFloorSprite(sectorId))
        return sectorFloor;
    int ceilZ = 0;
    int ceilHit = 0;
    int floorZ = 0;
    int floorHit = 0;
    GetZRangeAtXYZ(x, y, getceilzofslope(sectorId, x, y) + 1, sectorId,
                   &ceilZ, &ceilHit, &floorZ, &floorHit, 4, CLIPMASK0);
    if ((floorHit & 0xc000) == 0xc000 && floorZ < sectorFloor)
        return floorZ;
    return sectorFloor;
}

static int64_t segmentDistance2(int x, int y, int x1, int y1, int x2, int y2)
{
    const int64_t dx = x2 - x1;
    const int64_t dy = y2 - y1;
    const int64_t length2 = dx * dx + dy * dy;
    if (length2 <= 0)
        return int64_t(x - x1) * (x - x1) + int64_t(y - y1) * (y - y1);
    int64_t t = ((int64_t(x - x1) * dx) + (int64_t(y - y1) * dy)) * 1024 / length2;
    if (t < 0)
        t = 0;
    if (t > 1024)
        t = 1024;
    const int px = int(x1 + dx * t / 1024);
    const int py = int(y1 + dy * t / 1024);
    return int64_t(x - px) * (x - px) + int64_t(y - py) * (y - py);
}

// The two ends of a wall-aligned sprite, as the engine computes them when it
// clips against one.  A wall sprite is a line, not a post: a panel across a
// doorway is several hundred units wide, and treating it as a clipdist-sized
// circle at its centre misses everything but the middle.
static void wallSpriteSpan(const spritetype &record, int &x1, int &y1,
                           int &x2, int &y2)
{
    const int span = tilesiz[record.picnum].x;
    const int offset = (record.cstat & CSTAT_SPRITE_XFLIP)
        ? -(picanm[record.picnum].xofs + record.xoffset)
        : (picanm[record.picnum].xofs + record.xoffset);
    const int dax = sintable[record.ang & 2047] * record.xrepeat;
    const int day = sintable[(record.ang + 1536) & 2047] * record.xrepeat;
    const int anchor = (span >> 1) + offset;
    x1 = record.x - mulscale16(dax, anchor);
    y1 = record.y - mulscale16(day, anchor);
    x2 = x1 + mulscale16(dax, span);
    y2 = y1 + mulscale16(day, span);
}

// A sprite standing at this point that the player's body cannot pass.
// Returns its index, or -1.
//
// Only horizontal obstruction is asked about.  A floor-aligned sprite is a
// surface, not a barrier -- it is what a sprite bridge is made of -- and
// clipping against one walls off the very route it provides.
static int solidSpriteAt(int sectorId, int x, int y, int radius)
{
    if (!inRange(sectorId, 0, numsectors))
        return -1;
    const int floorZ = getflorzofslope(sectorId, x, y);
    const int headroom = playerCrouchClearance();
    for (int nSprite = headspritesect[sectorId]; nSprite >= 0;
         nSprite = nextspritesect[nSprite])
    {
        const spritetype &record = sprite[nSprite];
        if (!(record.cstat & CSTAT_SPRITE_BLOCK))
            continue;
        if (gMe && gMe->pSprite && nSprite == gMe->pSprite->index)
            continue;
        const int alignment = record.cstat & CSTAT_SPRITE_ALIGNMENT_MASK;
        if (alignment == CSTAT_SPRITE_ALIGNMENT_FLOOR
            || alignment == CSTAT_SPRITE_ALIGNMENT_SLOPE)
            continue;
        // Something hanging clear above head height is not in the way.
        int top = 0;
        int bottom = 0;
        GetSpriteExtents(&record, &top, &bottom);
        if (bottom < floorZ - headroom || top > floorZ)
            continue;
        if (alignment == CSTAT_SPRITE_ALIGNMENT_WALL)
        {
            int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
            wallSpriteSpan(record, x1, y1, x2, y2);
            if (segmentDistance2(x, y, x1, y1, x2, y2) < int64_t(radius) * radius)
                return nSprite;
            continue;
        }
        const int64_t margin = int64_t(radius) + (record.clipdist << 2);
        if (distance2(x, y, record.x, record.y) < margin * margin)
            return nSprite;
    }
    return -1;
}

static void setInteractionGeometry(Portal &portal)
{
    if (!inRange(portal.from, 0, numsectors))
        return;
    const int fromFloor = getflorzofslope(portal.from, portal.x, portal.y);
    const int fromCeiling = getceilzofslope(portal.from, portal.x, portal.y);
    int bottom = fromFloor;
    int top = fromCeiling;
    if (inRange(portal.to, 0, numsectors))
    {
        bottom = std::min(bottom, getflorzofslope(portal.to, portal.x, portal.y));
        top = std::max(top, getceilzofslope(portal.to, portal.x, portal.y));
    }
    if (bottom <= top)
    {
        bottom = fromFloor;
        top = fromCeiling;
    }
    portal.interactionBottomZ = bottom;
    portal.interactionTopZ = top;
    portal.z = (top + bottom) / 2;
}

// A usable sprite is aimed at by its own extents, not by the room it is in.
// Describing a low switch as if it filled the sector makes the bot stare
// straight ahead at mid-height and ActionScan never resolves it.
static void setSpriteInteractionGeometry(Portal &portal, int spriteIndex)
{
    if (!inRange(spriteIndex, 0, kMaxSprites))
        return;
    spritetype &record = sprite[spriteIndex];
    int top = record.z;
    int bottom = record.z;
    GetSpriteExtents(&record, &top, &bottom);
    if (bottom < top)
        std::swap(top, bottom);
    portal.interactionTopZ = top;
    portal.interactionBottomZ = bottom;
    portal.z = record.z;
}

static const char *itemCategory(int type)
{
    if (type >= kItemWeaponBase && type < kItemWeaponMax)
        return "weapon";
    if (type >= kItemAmmoBase && type < kItemAmmoMax)
        return "ammo";
    if (type >= kItemKeyBase && type < kItemKeyMax)
        return "key";
    if (type >= kItemHealthDoctorBag && type <= kItemHealthRedPotion)
        return "health";
    if (type >= kItemBase && type < kItemMax)
        return "pickup";
    return nullptr;
}

static void appendUnique(std::vector<int> &values, int value)
{
    if (std::find(values.begin(), values.end(), value) == values.end())
        values.push_back(value);
}

static Observation observeWorld()
{
    Observation result;
    if (!gMe || !gMe->pSprite || !gMe->pXSprite)
        return result;

    spritetype *player = gMe->pSprite;
    result.tick = gFrame * kTicsPerFrame;
    result.sector = player->sectnum;
    result.x = player->x;
    result.y = player->y;
    result.z = player->z;
    result.angle = player->ang;
    result.health = gMe->pXSprite->health;
    appendUnique(result.visibleSectors, result.sector);

    if (inRange(result.sector, 0, numsectors))
    {
        const sectortype &current = sector[result.sector];
        result.localSectorExtra = current.extra;
        if (current.extra > 0 && current.extra < kMaxXSectors)
        {
            result.localSectorState = xsector[current.extra].state;
            result.localSectorBusy = xsector[current.extra].busy;
        }
        result.playerZVelocity = zvel[player->index];
        result.localClearance = getflorzofslope(result.sector, result.x, result.y)
            - getceilzofslope(result.sector, result.x, result.y);
        playerCollisionDistances(result.playerCeilingDistance, result.playerFloorDistance);
        if (current.extra > 0 && current.extra < kMaxXSectors && xsector[current.extra].Exit)
            result.exitHere = true;

        // A current-sector Push is an ActionScan target in its own right.
        // It is not represented by a portal and must be remembered even when
        // every surrounding wall is one-sided.
        if (current.extra > 0 && current.extra < kMaxXSectors && xsector[current.extra].Push)
        {
            InteractionCandidate candidate;
            candidate.kind = kInteractionSector;
            candidate.id = result.sector;
            candidate.fromSector = result.sector;
            candidate.targetSector = result.sector;
            candidate.x = result.x;
            candidate.y = result.y;
            candidate.z = result.z;
            candidate.reversible = true;
            candidate.target.from = result.sector;
            candidate.target.to = result.sector;
            candidate.target.x = result.x;
            candidate.target.y = result.y;
            candidate.target.sectorPushCurrent = true;
            setInteractionGeometry(candidate.target);
            candidate.z = candidate.target.z;
            result.interactions.push_back(candidate);
        }

        for (int i = 0; i < current.wallnum; ++i)
        {
            const int wallIndex = current.wallptr + i;
            const walltype &wallRecord = wall[wallIndex];
            const walltype &nextWall = wall[wallRecord.point2];
            if (!inRange(wallRecord.nextsector, 0, numsectors))
            {
                // XWALL triggerPush is usable without a nextsector.  It is
                // discovered from the reachable current sector, then the
                // real ActionScan preview remains the activation authority.
                if (wallRecord.extra > 0 && wallRecord.extra < kMaxXWalls
                    && xwall[wallRecord.extra].triggerPush)
                {
                    InteractionCandidate candidate;
                    candidate.kind = kInteractionWall;
                    candidate.id = wallIndex;
                    candidate.fromSector = result.sector;
                    candidate.x = (wallRecord.x + nextWall.x) / 2;
                    candidate.y = (wallRecord.y + nextWall.y) / 2;
                    candidate.z = result.z;
                    candidate.key = xwall[wallRecord.extra].key;
                    candidate.locked = xwall[wallRecord.extra].locked != 0;
                    candidate.reversible = true;
                    candidate.target.wall = wallIndex;
                    candidate.target.from = result.sector;
                    candidate.target.to = -1;
            candidate.target.x = candidate.x;
            candidate.target.y = candidate.y;
            candidate.target.wallPush = true;
            candidate.target.mechanismTx = xwall[wallRecord.extra].txID;
            candidate.target.x1 = wallRecord.x;
            candidate.target.y1 = wallRecord.y;
            candidate.target.x2 = nextWall.x;
            candidate.target.y2 = nextWall.y;
            setInteractionGeometry(candidate.target);
            candidate.z = candidate.target.z;
            result.interactions.push_back(candidate);
                }
                continue;
            }
            ++result.localPortalCount;

            const int midX = (wallRecord.x + nextWall.x) / 2;
            const int midY = (wallRecord.y + nextWall.y) / 2;
            // The surface the player would stand on at the threshold, which
            // is a bridge plank when one lies across the doorway.
            const int fromFloor = standingFloorZ(result.sector, midX, midY);
            const int toFloor = standingFloorZ(wallRecord.nextsector, midX, midY);
            const int fromCeiling = getceilzofslope(result.sector, midX, midY);
            const int toCeiling = getceilzofslope(wallRecord.nextsector, midX, midY);
            const int openingWidth = int(std::sqrt(double(distance2(wallRecord.x, wallRecord.y, nextWall.x, nextWall.y))));
            const int clearance = std::min(fromFloor - fromCeiling, toFloor - toCeiling);
            const int floorDelta = toFloor - fromFloor;
            const int bodyClearance = playerBodyClearance();
            const int crouchClearance = playerCrouchClearance();
            const int jumpRiseLimit = playerJumpRiseLimit();
            const int dropLimit = jumpRiseLimit;
            const bool enoughWidth = openingWidth >= playerPassageWidth();
            const bool standingClearance = clearance >= bodyClearance;
            const bool crouchingClearance = clearance >= crouchClearance;
            const bool walkable = enoughWidth && standingClearance && std::abs(floorDelta) <= kMaxWalkableStep;
            const bool crouchable = enoughWidth && !standingClearance && crouchingClearance
                && std::abs(floorDelta) <= kMaxWalkableStep;
            const bool upward = floorDelta < 0;
            const bool downward = floorDelta > 0;
            const bool jumpable = enoughWidth && standingClearance && upward
                && -floorDelta <= jumpRiseLimit;
            const bool dropSafe = enoughWidth && standingClearance && downward
                && floorDelta <= dropLimit;
            if (jumpable && !(wallRecord.cstat & 1))
            {
                ++result.localJumpableCount;
                result.localMaxRise = std::max(result.localMaxRise, -floorDelta);
            }
            if (downward)
                result.localMaxDrop = std::max(result.localMaxDrop, floorDelta);
            const int midZ = fromFloor;
            const bool visible = cansee(result.x, result.y, result.z, result.sector,
                                         midX, midY, midZ, wallRecord.nextsector);

            Portal portal;
            portal.wall = wallIndex;
            portal.from = result.sector;
            portal.to = wallRecord.nextsector;
            portal.x = midX;
            portal.y = midY;
            portal.z = midZ;
            portal.floorZ = toFloor;
            portal.ceilingZ = toCeiling;
            portal.fromFloorZ = fromFloor;
            portal.fromCeilingZ = fromCeiling;
            portal.toFloorZ = toFloor;
            portal.toCeilingZ = toCeiling;
            portal.crouchClearance = crouchClearance;
            portal.jumpRiseLimit = jumpRiseLimit;
            portal.dropLimit = dropLimit;
            portal.x1 = wallRecord.x;
            portal.y1 = wallRecord.y;
            portal.x2 = nextWall.x;
            portal.y2 = nextWall.y;
            portal.visible = visible;
            portal.openingWidth = openingWidth;
            portal.floorDelta = floorDelta;
            portal.clearance = clearance;
            // The player is physically inside result.sector, so the complete
            // static wall loop of this sector is local knowledge even when a
            // corner hides the portal from the current view.  Visibility is
            // still retained separately for objects and remote geometry.
            portal.localGeometry = true;
            portal.walkable = !(wallRecord.cstat & 1) && walkable;
            portal.jumpable = !(wallRecord.cstat & 1) && !walkable && jumpable;
            portal.crouchable = !(wallRecord.cstat & 1) && crouchable;
            portal.dropSafe = !(wallRecord.cstat & 1) && dropSafe;
            portal.traversable = portal.walkable || portal.crouchable || portal.jumpable || portal.dropSafe;
            portal.capability = portal.walkable ? kTraversalWalkable
                : portal.crouchable ? kTraversalCrouchable
                : portal.jumpable ? kTraversalJumpable
                : portal.dropSafe ? kTraversalDropSafe
                : kTraversalCurrentlyUnavailable;
            portal.locked = false;
            if (wallRecord.extra > 0 && wallRecord.extra < kMaxXWalls)
            {
                const XWALL &extra = xwall[wallRecord.extra];
                portal.wallPush = extra.triggerPush != 0;
                portal.shootable = extra.triggerVector != 0 && !extra.isTriggered;
                portal.mechanismTx = extra.txID;
                portal.key = extra.key;
                portal.locked = extra.locked != 0;
                portal.wallState = extra.state;
                portal.wallBusy = extra.busy;
            }
            if (sector[portal.to].extra > 0 && sector[portal.to].extra < kMaxXSectors)
            {
                const XSECTOR &extra = xsector[sector[portal.to].extra];
                if (!portal.key)
                    portal.key = extra.Key;
                portal.locked = extra.locked != 0;
                portal.sectorPush = extra.Wallpush != 0;
                portal.sectorState = extra.state;
                portal.sectorBusy = extra.busy;
                // A damaging floor only hurts the player who touches it.
                // Blood calls actTouchFloor for a sector hit and not for a
                // sprite hit, so a bridge laid across the pit is a way over
                // it -- refusing the whole sector makes such a crossing
                // invisible.
                const bool touchesDamagingFloor =
                    standingFloorZ(portal.to, portal.x, portal.y)
                        >= getflorzofslope(portal.to, portal.x, portal.y);
                if ((extra.damageType != 0 || sector[portal.to].type == kSectorDamage)
                    && touchesDamagingFloor)
                {
                    portal.walkable = false;
                    portal.jumpable = false;
                    portal.crouchable = false;
                    portal.dropSafe = false;
                    portal.traversable = false;
                    portal.capability = kTraversalCurrentlyUnavailable;
                }
            }
            // A solid sprite standing in a doorway shuts it as surely as a
            // closed door does, and the wall geometry says nothing about it.
            // Without this a barricaded opening read as walkable, and the bot
            // shouldered it until the objective budget ran out instead of
            // treating it as an obstacle with something to be done about it.
            if (portal.traversable)
            {
                const int radius = gMe && gMe->pSprite
                    ? (gMe->pSprite->clipdist << 2) : 128;
                // Sample across the opening rather than only its midpoint.
                // A barricade often covers part of a doorway and leaves a
                // gap at one end; judging the whole boundary by its centre
                // either shuts a passable door or waves the bot at a solid
                // one.  Where a clear stretch exists, aim at the middle of
                // that stretch instead of at the middle of the wall.
                const int samples = 9;
                int clearFirst = -1;
                int clearLast = -1;
                int obstruction = -1;
                for (int step = 0; step < samples; ++step)
                {
                    const int px = portal.x1
                        + int(int64_t(portal.x2 - portal.x1) * step / (samples - 1));
                    const int py = portal.y1
                        + int(int64_t(portal.y2 - portal.y1) * step / (samples - 1));
                    int hit = solidSpriteAt(portal.from, px, py, radius);
                    if (hit < 0)
                        hit = solidSpriteAt(portal.to, px, py, radius);
                    if (hit >= 0)
                    {
                        obstruction = hit;
                        continue;
                    }
                    if (clearFirst < 0)
                        clearFirst = step;
                    clearLast = step;
                }
                if (clearFirst >= 0 && obstruction >= 0)
                {
                    const int middle = (clearFirst + clearLast) / 2;
                    portal.x = portal.x1
                        + int(int64_t(portal.x2 - portal.x1) * middle / (samples - 1));
                    portal.y = portal.y1
                        + int(int64_t(portal.y2 - portal.y1) * middle / (samples - 1));
                    obstruction = -1;
                }
                if (obstruction >= 0)
                {
                    portal.walkable = false;
                    portal.jumpable = false;
                    portal.crouchable = false;
                    portal.dropSafe = false;
                    portal.traversable = false;
                    portal.capability = kTraversalCurrentlyUnavailable;
                    portal.blockedBySprite = true;
                    // Blocked, but not necessarily hopeless: a barricade the
                    // player can operate is progression work, not a wall.
                    const spritetype &blocker = sprite[obstruction];
                    if (validXSprite(blocker.extra)
                        && (xsprite[blocker.extra].Push || xsprite[blocker.extra].Vector))
                        portal.interactionAffordance = true;
                }
            }
            if (sector[portal.from].extra > 0 && sector[portal.from].extra < kMaxXSectors)
                portal.sectorPushCurrent = xsector[sector[portal.from].extra].Push != 0;
            portal.interactionAffordance = portal.wallPush || portal.sectorPush
                || portal.sectorPushCurrent || portal.shootable;
            portal.currentlyAvailable = portal.traversable || portal.interactionAffordance;
            if (!portal.currentlyAvailable)
            {
                portal.capability = kTraversalCurrentlyUnavailable;
                portal.unavailableReason = (wallRecord.cstat & 1) ? 1 : 2;
            }
            portal.directUse = (portal.wallPush || portal.sectorPush || portal.sectorPushCurrent)
                && (!portal.traversable || std::abs(portal.floorDelta) > kMaxWalkableStep);

            if (portal.shootable)
            {
                InteractionCandidate candidate;
                candidate.kind = kInteractionWall;
                candidate.id = wallIndex;
                candidate.fromSector = result.sector;
                candidate.targetSector = portal.to;
                candidate.x = midX;
                candidate.y = midY;
                candidate.z = midZ;
                candidate.reversible = false;
                candidate.target = portal;
                setInteractionGeometry(candidate.target);
                candidate.x = candidate.target.x;
                candidate.y = candidate.target.y;
                candidate.z = candidate.target.z;
                result.interactions.push_back(candidate);
            }
            else if (portal.interactionAffordance)
            {
                InteractionCandidate candidate;
                candidate.kind = kInteractionWall;
                candidate.id = wallIndex;
                candidate.fromSector = result.sector;
                candidate.targetSector = portal.to;
                candidate.x = midX;
                candidate.y = midY;
                candidate.z = midZ;
                candidate.key = portal.key;
                candidate.locked = portal.locked;
                candidate.reversible = true;
                candidate.target = portal;
                setInteractionGeometry(candidate.target);
                candidate.x = candidate.target.x;
                candidate.y = candidate.target.y;
                candidate.z = candidate.target.z;
                result.interactions.push_back(candidate);
            }

            // A compact mechanism room can contain one-sided XWALL push
            // surfaces immediately beyond an otherwise ordinary portal.  A
            // player can see and use those surfaces before the portal is
            // physically consumed, while they are not present in the
            // current sector's wall list.  Discover only the adjacent
            // sector's currently visible trigger walls; ActionScanPreview
            // still decides whether the current pose can actually use one.
            if (inRange(portal.to, 0, numsectors))
            {
                const sectortype &adjacent = sector[portal.to];
                for (int j = 0; j < adjacent.wallnum; ++j)
                {
                    const int adjacentWallIndex = adjacent.wallptr + j;
                    const walltype &adjacentWall = wall[adjacentWallIndex];
                    const walltype &adjacentNext = wall[adjacentWall.point2];
                    if (inRange(adjacentWall.nextsector, 0, numsectors)
                        || !inRange(adjacentWall.extra, 1, kMaxXWalls)
                        || !xwall[adjacentWall.extra].triggerPush)
                        continue;
                    const int adjacentX = (adjacentWall.x + adjacentNext.x) / 2;
                    const int adjacentY = (adjacentWall.y + adjacentNext.y) / 2;
                    const bool visibleFromPlayer = cansee(
                        result.x, result.y, result.z, result.sector,
                        adjacentX, adjacentY, result.z, portal.to);
                    if (!visibleFromPlayer)
                        continue;
                    InteractionCandidate candidate;
                    candidate.kind = kInteractionWall;
                    candidate.id = adjacentWallIndex;
                    candidate.fromSector = result.sector;
                    candidate.targetSector = portal.to;
                    candidate.x = adjacentX;
                    candidate.y = adjacentY;
                    candidate.z = result.z;
                    candidate.key = xwall[adjacentWall.extra].key;
                    candidate.locked = xwall[adjacentWall.extra].locked != 0;
                    candidate.reversible = true;
                    candidate.target.wall = adjacentWallIndex;
                    candidate.target.from = result.sector;
                    candidate.target.to = -1;
                    candidate.target.x = adjacentX;
                    candidate.target.y = adjacentY;
                    candidate.target.wallPush = true;
                    candidate.target.mechanismTx = xwall[adjacentWall.extra].txID;
                    candidate.target.x1 = adjacentWall.x;
                    candidate.target.y1 = adjacentWall.y;
                    candidate.target.x2 = adjacentNext.x;
                    candidate.target.y2 = adjacentNext.y;
                    setInteractionGeometry(candidate.target);
                    candidate.z = candidate.target.z;
                    result.interactions.push_back(candidate);
                }
            }

            if (visible)
                appendUnique(result.visibleSectors, portal.to);
            result.portals.push_back(portal);
        }
    }

    // The scan is over the engine's sprite list, but only objects passing the
    // authoritative cansee() test enter the bot's observation.
    for (int i = 0; i < kMaxSprites; ++i)
    {
        spritetype &candidate = sprite[i];
        const bool isEnemy = candidate.statnum == kStatDude;
        const bool isItem = candidate.statnum == kStatItem;
        const bool isThing = candidate.statnum == kStatThing;
        const bool isSwitch = candidate.type >= kSwitchBase && candidate.type < kSwitchMax;
        // Whether the player can operate a sprite is a fact Blood records on
        // the sprite, not something to be inferred from its status list or
        // its type number.  A mapper is free to wire a plain decoration to a
        // channel, and one that answers to Use is a switch whatever tile it
        // wears; keying off the type whitelist alone left the bot walking
        // past hand-built mechanisms as though they were scenery.
        const bool operable = !isEnemy && validXSprite(candidate.extra)
            && (xsprite[candidate.extra].Push || xsprite[candidate.extra].Vector);
        if (!isEnemy && !isItem && !isThing && !isSwitch && !operable)
            continue;
        if (candidate.index == player->index || candidate.sectnum < 0)
            continue;
        if (isEnemy && (!isEnemyType(candidate.type) || !validXSprite(candidate.extra) || xsprite[candidate.extra].health == 0))
            continue;
        if (isItem && !operable && itemCategory(candidate.type) == nullptr)
            continue;
        if ((isThing || isSwitch) && !operable)
            continue;
        if (!cansee(result.x, result.y, result.z, result.sector,
                    candidate.x, candidate.y, candidate.z, candidate.sectnum))
            continue;

        VisibleObject object;
        object.sprite = i;
        object.sector = candidate.sectnum;
        object.type = candidate.type;
        object.x = candidate.x;
        object.y = candidate.y;
        object.z = candidate.z;
        object.kind = isEnemy ? kObjectEnemy
                         : (isItem && isKeyType(candidate.type)) ? kObjectKey
                         : (isThing || isSwitch || operable) ? kObjectInteractive
                         : kObjectPickup;
        result.objects.push_back(object);
        if (object.kind == kObjectInteractive && validXSprite(candidate.extra)
            && xsprite[candidate.extra].Push)
        {
            InteractionCandidate interaction;
            interaction.kind = kInteractionSprite;
            interaction.id = i;
            // Where the bot can operate it from, which is where it is
            // standing when it sees it -- the same rule wall mechanisms use.
            // Filing it under the sprite's own sector meant a mechanism
            // standing in the doorway of a room the bot had not entered was
            // unreachable by construction, so the bot could see the thing
            // blocking its way and never form the intention to push it.
            interaction.fromSector = result.sector;
            interaction.targetSector = candidate.sectnum;
            interaction.x = candidate.x;
            interaction.y = candidate.y;
            interaction.z = candidate.z;
            interaction.reversible = true;
            interaction.target.wall = -1;
            interaction.target.from = candidate.sectnum;
            interaction.target.to = candidate.sectnum;
            interaction.target.x = candidate.x;
            interaction.target.y = candidate.y;
            setInteractionGeometry(interaction.target);
            setSpriteInteractionGeometry(interaction.target, i);
            interaction.z = interaction.target.z;
            result.interactions.push_back(interaction);
        }
    }
    return result;
}

static const char *objectName(ObjectKind kind)
{
    switch (kind)
    {
    case kObjectEnemy: return "enemy";
    case kObjectKey: return "key";
    case kObjectInteractive: return "interactive";
    default: return "pickup";
    }
}

}

struct LLMapperBot::Impl
{
    enum ObjectiveType
    {
        kObjectiveNone,
        kObjectivePickup,
        kObjectiveKey,
        kObjectiveInteraction,
        kObjectiveFrontier,
        kObjectiveInvestigate,
        kObjectiveExpose,
    };

    struct Objective
    {
        ObjectiveType type = kObjectiveNone;
        bool active = false;
        int id = -1;
        int sector = -1;
        int targetSector = -1;
        int x = 0;
        int y = 0;
        int z = 0;
        int interactionKey = -1;
        int wall = -1;
        int routeStepWall = -1;
        int routeStepFrom = -1;
        int routeStepTo = -1;
        int startedTick = -1;
        int inspectX = 0;
        int inspectY = 0;
        int inspectZ = 0;
        bool inspectPoseValid = false;
    };

    struct ObjectMemory
    {
        VisibleObject object;
        bool observed = false;
        bool collected = false;
        int lastSeenTick = -1;
    };

    struct EdgeFailure
    {
        int geometrySignature = 0;
        int attempts = 0;
        int expiresTick = 0;
    };

    struct DoorMemory
    {
        enum Availability
        {
            kActionable,
            kMissingKey,
            kStructurallyBlocked,
            kCurrentlyUnavailable,
        };

        enum InteractionState
        {
            kIdle,
            kWaiting,
            kOpening,
            kOpen,
            kFailed,
        };

        int id = -1;
        Portal portal;
        int attempts = 0;
        Availability availability = kActionable;
        bool opened = false;
        int unavailableReason = 0;
        InteractionState interaction = kIdle;
        int interactionStartedTick = -1;
        int interactionDeadlineTick = -1;
        int unavailableFromSector = -1;
    };

    struct InteractionMemory
    {
        InteractionKind kind = kInteractionWall;
        int id = -1;
        int fromSector = -1;
        int targetSector = -1;
        int x = 0;
        int y = 0;
        int z = 0;
        int key = 0;
        bool locked = false;
        bool reversible = false;
        bool observed = false;
        bool attempted = false;
        bool activated = false;
        int activationCount = 0;
        int lastActivationTick = -1;
        int beforeState = 0;
        int afterState = 0;
        int unavailableFromSector = -1;
        int unavailableState = 0;
        int unavailableAttempts = 0;
        int unavailablePose = 0;
        int state = 0;
        bool engineAccepted = false;
        bool observedLocalEffect = false;
        bool observedKnownWorldDelta = false;
        bool recoveryAttempted = false;
        bool traversed = false;
        std::vector<Portal> beforePortals;
        Portal target;
    };

    FILE *telemetry = nullptr;
    FILE *trajectory = nullptr;
    std::string telemetryPath = "llmapper-bot.ndjson";
    std::string trajectoryPath = "llmapper-bot.trajectory.ndjson";
    std::string demoPath = "llmapper-bot.dem";
    int timeoutSeconds = kDefaultTimeoutSeconds;
    int stallSeconds = kDefaultStallSeconds;
    int lastObservationTick = -1;
    int lastTrajectoryTick = -1;
    int lastSemanticProgressTick = 0;
    bool explorationSnapshotEmitted = false;
    int lastStateLocation = -1;
    std::string lastStateGoal;
    int lastStateTarget = -1;
    int lastStateKnowledge = -1;
    int lastStateInventory = -1;
    int lastEngineMoveHit = -1;
    int mechanismWaitTick = -1;
    int repeatedStateCount = 0;
    int loopBreakCount = 0;
    int pendingUse = -1;
    int pendingUseTick = -1;
    int pendingInteractionKey = -1;
    std::string result;
    std::string failureReason;
    std::string currentGoal;
    Objective currentObjective;
    Objective suspendedObjective;
    bool suspendedObjectiveActive = false;
    Observation observation;
    std::set<int> observedSectors;
    std::set<int> visitedSectors;
    std::map<int, DoorMemory> doors;
    std::map<int, InteractionMemory> interactions;
    std::map<int, ObjectMemory> objectMemory;
    VisibleObject selectedObject;
    bool recoveryMode = false;
    std::set<int> knownObjects;
    std::set<int> knownKeys;   // key types the bot has ever seen in the level
    std::set<int> heldKeys;    // key ids actually in Caleb's inventory
    std::set<int> unreachableObjects;
    std::set<int> unreachableEnemies;
    std::set<int> seenEdges;
    std::set<int> visitedEdges;
    std::map<int, EdgeFailure> failedEdges;
    std::map<int, int> localFailureSignatures;
    std::vector<NavEdgeFailure> navEdgeFailures;
    std::vector<InvestigateRecord> investigatedBoundaries;
    std::vector<DerivedFrontier> derivedFrontiers;
    int exploreDestination = -1;
    int exploreCrossingWall = -1;
    int exploreCrossingFrom = -1;
    bool investigateCrouch = false;
    int investigateLookTick = 0;
    CombatTactic combatTactic = kCombatNone;
    int combatTacticStartTick = -1;
    int lastAttacker = -1;
    int lastAttackerDamageType = -1;
    bool combatRetreatUnavailableEmitted = false;
    int lastCombatTacticEvent = -1;
    // One place converts a wanted change in the player's pitch into the
    // q16mlook the engine expects.  Outside a vanilla demo, ProcessInput
    // applies `q16mlook << 3`, so a raw value swings the view eight times as
    // far as asked -- which is what made the bot flick up and down every
    // other tick once anything asked it to level off.  Scale in fixed point
    // rather than dividing the integer first, so a one-unit correction still
    // moves the view instead of truncating to nothing and never settling.
    static fix16_t encodeLook(int wantedDelta)
    {
        const fix16_t requested = fix16_from_int(wantedDelta);
        return gDemo.VanillaDemo() ? requested : requested / 8;
    }

    struct MovementIntent
    {
        int forward = 0;
        int strafe = 0;
        int turn = 0;
        fix16_t look = 0;
        bool jump = false;
        bool crouch = false;
        bool run = true;
        bool valid = false;
    };
    struct CombatIntent
    {
        int turn = 0;
        fix16_t look = 0;
        bool shoot = false;
        int weapon = 0;
        bool aim = false;
        // Whether this is a threat worth taking the controls away from
        // whatever the bot was doing.  Anything else is opportunistic.
        bool urgent = false;
        bool valid = false;
    };
    struct UseIntent
    {
        bool action = false;
        bool shoot = false;
        int weapon = 0;
    };
    std::set<int> jumpFallbackEdges;
    std::set<int> openedRoutes;
    std::map<int, std::vector<Portal>> knownGraph;
    Portal routePortal;
    Portal localJumpPortal;
    int knowledgeRevision = 0;
    int inventoryRevision = 0;
    int lastObservedSector = -1;
    int lastGeometryTelemetrySector = -1;
    bool localDynamicJumpAttempted = false;
    int localDynamicJumpUntilTick = -1;
    int localJumpProbeCount = 0;
    int lastTransitionFrom = -1;
    int lastTransitionTo = -1;
    int lastTransitionTick = -1;
    int lastTransitionKnowledge = -1;
    int lastTransitionInventory = -1;
    int repeatedBacktrackCount = 0;
    int currentGoalTarget = -1;
    bool movementTargetActive = false;
    int movementTargetX = 0;
    int movementTargetY = 0;
    int movementTargetSector = -1;
    int movementTargetId = -1;
    int movementTargetFrom = -1;
    TraversalCapability movementTargetCapability = kTraversalUnknown;
    std::string movementTargetGoal;
    int targetLastX = 0;
    int targetLastY = 0;
    int targetLastZ = 0;
    int targetLastDistance2 = 0;
    int targetBestDistance2 = 0;
    int targetLastProgressTick = 0;
    int jumpCooldownTick = 0;
    int jumpAttempts = 0;
    JumpExecutionState jumpState = kJumpInactive;
    int jumpStateTick = -1;
    int jumpTargetX = 0;
    int jumpTargetY = 0;
    int jumpTargetSector = -1;
    int jumpSourceSector = -1;
    int searchAngle = -1;
    bool searchProbeActive = false;
    int searchProbeX = 0;
    int searchProbeY = 0;
    int searchProbeStartTick = 0;
    int lastDoorActionEventTick = -1;
    int lastUseProbeDoor = -1;
    int lastUseProbeTick = -1;
    std::string lastUseProbeStatus;
    int lastUsedDoor = -1;
    int lastUsedDoorFrom = -1;
    int lastUsedDoorTo = -1;
    int lastUsedDoorTick = -1;
    int portalWaypointWall = -1;
    int portalWaypointX = 0;
    int portalWaypointY = 0;
    bool portalWaypointActive = false;
    std::vector<LocalWaypoint> portalPlan;
    size_t portalPlanIndex = 0;
    int portalPlanWall = -1;
    int portalPlanSignature = 0;
    int portalPlanAttempts = 0;
    bool portalCrossingActive = false;
    int portalCrossingX = 0;
    int portalCrossingY = 0;
    int portalCrossingDestinationX = 0;
    int portalCrossingDestinationY = 0;
    bool portalCrossingDestinationValid = false;
    int portalCrossingStartTick = -1;
    bool navigationActive = false;
    int navigationOriginalX = 0;
    int navigationOriginalY = 0;
    int navigationOriginalZ = 0;
    int navigationOriginalSector = -1;
    int navigationOriginalId = -1;
    int navigationOriginalSignature = 0;
    int navigationDetourWall = -1;
    int navigationDetourDepth = 0;
    int navigationWaypointX = 0;
    int navigationWaypointY = 0;
    bool navigationWaypointActive = false;
    bool navigationUsingDetour = false;
    int navigationActionableBlocker = -1;
    std::set<int> navigationRecentWalls;
    std::vector<NavRouteStep> navRoute;
    size_t navRouteIndex = 0;
    int navRouteSignature = 0;
    int navRouteTopologyRevision = 0;
    int navRouteRejectedSignature = 0;
    int lastNavWaypointX = INT32_MIN;
    int lastNavWaypointY = INT32_MIN;
    int navWaypointBestDistance2 = INT32_MAX;
    int navWaypointProgressTick = 0;
    struct GridCell
    {
        int gx = 0;
        int gy = 0;
        int x = 0;
        int y = 0;
    };
    struct SectorGrid
    {
        int signature = 0;
        // False when no point in the sector clears the player's own radius
        // from its walls -- a slot narrower than Caleb.  Such a sector is
        // not a destination and not a route, only scenery.
        bool admitsPlayer = true;
        std::vector<GridCell> cells;
    };
    std::map<int, SectorGrid> sectorGrids;
    std::map<int64_t, int> navCellIndex;
    std::vector<NavCell> navCells;
    int navTopologySignature = 0;
    int navPoseSignatureValue = 0;
    int navDynamicSignatureValue = 0;
    int navTopologyRevision = 0;
    int navPoseStableTicks = 0;
    bool navMeshInMotion = false;
    int lastNavWaypointEvent = -1;
    int navigationFailureSignature = 0;
    int navigationFailureCount = 0;
    int lastNavigationFailureTick = -1;
    int lastFrontierWall = -1;
    int lastFrontierSource = -1;
    int lastFrontierTarget = -1;
    int frontierSelectionCount = 0;
    int lastTransportRouteWall = -1;
    int lastTransportRouteFrom = -1;
    int lastTransportRouteTo = -1;
    int lastCombatRetreatCell = -1;
    bool crouchTargetActive = false;
    int followThroughKey = -1;
    int followThroughStartTick = -1;
    int followThroughFrom = -1;
    Portal followThroughPortal;
    bool cryptExitProven = false;
    int lastObservedHealth = -1;
    int lastDamageTick = -1;
    int lastInteractionWaitKey = -1;
    std::string cameraOwner;
    int cameraTarget = -1;
    std::set<int> engagedEnemies;
    std::map<int, int> enemyHealth;
    std::set<int> countedKills;
    int lastLedgerDumpTick = -1;
    int lastFailedOpportunity = -1;
    std::map<int, int> suppressionEvidence;
    int lastFailedOpportunityTick = -1;
    int stationaryX = INT32_MIN;
    int stationaryY = INT32_MIN;
    int stationaryZ = INT32_MIN;
    int stationarySinceTick = 0;
    int lastCommandedMoveTick = 0;
    std::set<int64_t> observedCells;
    int coverageTargetX = 0;
    int coverageTargetY = 0;
    int coverageTargetSector = -1;
    // Bounded-objective bookkeeping.  suppressedUntil keeps a failed
    // objective dormant instead of deleting the knowledge behind it.
    int objectiveProgressTick = -1;
    int objectiveBestDistance2 = INT32_MAX;
    int waitingSinceTick = -1;
    int objectiveBestRemainingSteps = INT32_MAX;
    int objectiveProgressSector = -1;
    std::map<int, int> suppressedUntil;
    std::map<int, int> suppressionCount;
    // Exploration tree: the branch structure of the level as the bot has
    // actually walked it.  parent/depth make "keep going forward" and
    // "back out to the nearest unresolved branch" the same ordering rule.
    struct TreeNode
    {
        int parent = -1;
        int depth = 0;
        int enteredTick = 0;
    };
    std::map<int, TreeNode> explorationTree;
    std::vector<llmapper::Opportunity> ledger;
    llmapper::Mission mission;
    int missionOpportunity = -1;
    int missionStartedTick = -1;
    unsigned heldKeyMask = 0;
    unsigned lastHeldKeyMask = 0;
    std::map<int, int> lockedDoorKey;
    mutable std::map<int, int> mechanismReceivers;
    mutable std::vector<int> routeGateWall;
    mutable std::vector<int> routeGateSector;
    std::map<int, int> inertBoundaries;
    std::set<int> narrowSectors;
    std::map<int, bool> boundaryOpen;
    std::map<int, int> boundaryClearance;
    int lastCrossingWall = -1;
    int lastFrontierChoice = 0;
    int directCrossingWall = -1;
    int directCrossingBlockedTicks = 0;
    int directCrossingBlockedUntil = -1;
    int lastSuppressedJumpTarget = -1;
    int lastAcceptedUseTick = -1;
    int clearingSector = -1;
    int clearingWall = -1;
    int healthLostSinceObservation = 0;
    int damageBurstStartTick = -1;
    int damageBurstCount = 0;
    int damageBurstHealth = 0;
    std::map<int, int> hazardSectors;   // sector -> tick it stops being avoided
    std::set<int> triedSurfaces;        // mechanism/wall pairs already attempted
    std::set<int> reopenableConnections;
    int openedRouteWall = -1;
    int openedRouteFrom = -1;
    int openedRouteTo = -1;
    int openedRouteTick = -1;

    void openFiles()
    {
        telemetry = fopen(telemetryPath.c_str(), "wb");
        trajectory = fopen(trajectoryPath.c_str(), "wb");
    }

    void event(const char *name, const char *detail = nullptr)
    {
        if (!telemetry)
            return;
        const int gameSeconds = (gFrame * kTicsPerFrame) / kTicsPerSec;
        fprintf(telemetry, "{\"type\":\"event\",\"game_time\":%d,\"event\":\"%s\"",
                gameSeconds, name);
        if (detail)
            fprintf(telemetry, ",\"detail\":\"%s\"", detail);
        fprintf(telemetry, "}\n");
        fflush(telemetry);
    }

    void retirePendingInteraction(int key)
    {
        (void)key;
    }

    void retirePendingFrontier(int wall, int from, int to)
    {
        (void)wall;
        (void)from;
        (void)to;
        exploreDestination = -1;
        exploreCrossingWall = -1;
        exploreCrossingFrom = -1;
    }

    int interactionSourceSignature(const InteractionMemory &memory) const
    {
        return interactionStateSignature(memory);
    }

    void emitInteractionPortalState(const InteractionMemory &memory, const char *phase)
    {
        if (memory.target.wall < 0 || !inRange(memory.target.from, 0, numsectors)
            || !inRange(memory.target.wall, 0, numwalls))
            return;
        const walltype &wallRecord = wall[memory.target.wall];
        if (!inRange(wallRecord.nextsector, 0, numsectors))
            return;
        const walltype &nextWall = wall[wallRecord.point2];
        const int midX = (wallRecord.x + nextWall.x) / 2;
        const int midY = (wallRecord.y + nextWall.y) / 2;
        const int fromFloor = getflorzofslope(memory.target.from, midX, midY);
        const int fromCeiling = getceilzofslope(memory.target.from, midX, midY);
        const int toFloor = getflorzofslope(wallRecord.nextsector, midX, midY);
        const int toCeiling = getceilzofslope(wallRecord.nextsector, midX, midY);
        const int toSector = wallRecord.nextsector;
        const int cstat = wallRecord.cstat;
        const int width = int(std::sqrt(double(distance2(wallRecord.x, wallRecord.y,
                                                      nextWall.x, nextWall.y))));
        const int clearance = std::min(fromFloor - fromCeiling, toFloor - toCeiling);
        char detail[256];
        snprintf(detail, sizeof(detail),
                 "phase=%s wall=%d from=%d to=%d cstat=%d floor_delta=%d clearance=%d width=%d observed_portal=%d",
                 phase, memory.target.wall, memory.target.from, toSector,
                 cstat, toFloor - fromFloor, clearance, width,
                 std::any_of(observation.portals.begin(), observation.portals.end(),
                             [&wallRecord](const Portal &portal)
                             {
                                 return portal.wall == (&wallRecord - wall)
                                     && portal.to == wallRecord.nextsector;
                             }) ? 1 : 0);
        event("interaction_portal_state", detail);
    }

    void emitInteractionGeometryState(const InteractionMemory &memory, const char *phase)
    {
        if (memory.target.wall < 0 || !inRange(memory.target.wall, 0, numwalls))
            return;
        const walltype &wallRecord = wall[memory.target.wall];
        if (!inRange(wallRecord.point2, 0, numwalls))
            return;
        const walltype &nextWall = wall[wallRecord.point2];
        const int width = int(std::sqrt(double(distance2(wallRecord.x, wallRecord.y,
                                                      nextWall.x, nextWall.y))));
        char detail[320];
        snprintf(detail, sizeof(detail),
                 "phase=%s kind=%d id=%d wall=%d from=%d to=%d x1=%d y1=%d x2=%d y2=%d width=%d target_top_z=%d target_bottom_z=%d crouch=%d",
                 phase, int(memory.kind), memory.id, memory.target.wall,
                 memory.target.from, memory.target.to, int(wallRecord.x), int(wallRecord.y),
                 int(nextWall.x), int(nextWall.y), width, memory.target.interactionTopZ,
                 memory.target.interactionBottomZ, memory.target.interactionCrouch ? 1 : 0);
        event("interaction_geometry_state", detail);
    }

    void updateFollowThrough(InteractionMemory &memory)
    {
        if (!memory.engineAccepted || memory.beforePortals.empty())
            return;
        for (const Portal &portal : observation.portals)
        {
            if (portal.from != memory.fromSector || portal.to == portal.from)
                continue;
            if (memory.target.wall >= 0 && portal.wall != memory.target.wall
                && memory.target.to != portal.to)
                continue;
            if (memory.target.to >= 0 && memory.target.to != portal.to)
                continue;
            if (!portal.traversable && !portal.jumpable)
                continue;
            bool wasTraversable = false;
            for (const Portal &before : memory.beforePortals)
            {
                if (before.wall == portal.wall && before.to == portal.to)
                {
                    wasTraversable = before.traversable || before.jumpable;
                    break;
                }
            }
            if (wasTraversable)
                continue;

            followThroughKey = interactionMemoryKey(memory);
            followThroughStartTick = observation.tick;
            followThroughFrom = portal.from;
            followThroughPortal = portal;
            char detail[192];
            snprintf(detail, sizeof(detail),
                     "kind=%d id=%d wall=%d from=%d to=%d before_traversable=0 after_traversable=%d floor_delta=%d clearance=%d",
                     int(memory.kind), memory.id, portal.wall, portal.from, portal.to,
                     portal.traversable || portal.jumpable ? 1 : 0,
                     portal.floorDelta, portal.clearance);
            event("interaction_follow_through_ready", detail);
            return;
        }
    }

    void actionResolved(int hit, int target, int extra, bool accepted, int key)
    {
        if (accepted)
            lastAcceptedUseTick = observation.tick;
        const char *kind = hit == 0 ? "wall" : hit == 3 ? "sprite" : hit == 6 ? "sector" : "none";
        char detail[256];
        snprintf(detail, sizeof(detail),
                 "source=gameplay_tick hit=%d kind=%s target=%d extra=%d trigger_dispatched=%d key=%d goal=%s",
                 hit, kind, target, extra, accepted ? 1 : 0, key,
                 currentGoal.c_str());
        event("engine_use_resolved", detail);
        for (auto &entry : interactions)
        {
            InteractionMemory &memory = entry.second;
            if (entry.first == pendingInteractionKey && memory.state == 1
                && interactionTargetMatches(memory, hit, target))
            {
                int sourceExtra = -1;
                int sourceState = -1;
                int sourceBusy = 0;
                if (hit == 6 && inRange(target, 0, numsectors))
                {
                    sourceExtra = sector[target].extra;
                    if (sourceExtra > 0 && sourceExtra < kMaxXSectors)
                    {
                        sourceState = xsector[sourceExtra].state;
                        sourceBusy = xsector[sourceExtra].busy;
                    }
                }
                else if (hit == 0 && inRange(target, 0, numwalls))
                {
                    sourceExtra = wall[target].extra;
                    if (sourceExtra > 0 && sourceExtra < kMaxXWalls)
                    {
                        sourceState = xwall[sourceExtra].state;
                        sourceBusy = xwall[sourceExtra].busy;
                    }
                }
                else if (hit == 3 && inRange(target, 0, kMaxSprites))
                {
                    sourceExtra = sprite[target].extra;
                    if (sourceExtra > 0 && sourceExtra < kMaxXSprites)
                    {
                        sourceState = xsprite[sourceExtra].state;
                        sourceBusy = xsprite[sourceExtra].busy;
                    }
                }
                char interactionDetail[320];
                memory.engineAccepted = accepted;
                if (accepted)
                {
                    memory.activated = true;
                    memory.observedKnownWorldDelta = false;
                    retirePendingInteraction(entry.first);
                }
                else
                    memory.state = 3;
                snprintf(interactionDetail, sizeof(interactionDetail),
                         "kind=%d id=%d from_sector=%d player_sector=%d hit=%d target=%d extra=%d accepted=%d source_extra=%d source_state=%d source_busy=%d source_signature_before=%d source_signature_after=%d",
                         int(memory.kind), memory.id, memory.fromSector, observation.sector,
                         hit, target, extra, accepted ? 1 : 0, sourceExtra, sourceState,
                         sourceBusy, memory.beforeState, interactionSourceSignature(memory));
                event("interaction_engine_resolved", interactionDetail);
                emitInteractionPortalState(memory, "after_process_input");
                emitInteractionGeometryState(memory, "after_process_input");

                if (memory.target.wall >= 0)
                {
                    auto door = doors.find(memory.target.wall);
                    if (door != doors.end())
                    {
                        door->second.unavailableFromSector = -1;
                        door->second.interaction = accepted
                            ? DoorMemory::kWaiting : DoorMemory::kFailed;
                        door->second.interactionDeadlineTick = accepted
                            ? observation.tick + kInteractionTimeoutTicks : -1;
                    }
                }
            }
        }
        if (lastUsedDoor >= 0)
        {
            auto door = doors.find(lastUsedDoor);
            if (door != doors.end() && accepted)
                door->second.interaction = DoorMemory::kWaiting;
        }
        pendingInteractionKey = -1;
    }

    // `look` is sampled because a view that will not settle is invisible in
    // position alone: the bot stands still and flicks up and down.
    void trajectorySample()
    {
        if (!trajectory || !gMe || !gMe->pSprite)
            return;
        fprintf(trajectory, "{\"game_time\":%d,\"x\":%d,\"y\":%d,\"z\":%d,\"sector\":%d,\"angle\":%d,\"look\":%d,\"health\":%d}\n",
                (gFrame * kTicsPerFrame) / kTicsPerSec,
                int(gMe->pSprite->x), int(gMe->pSprite->y), int(gMe->pSprite->z), int(gMe->pSprite->sectnum),
                int(gMe->pSprite->ang), fix16_to_int(gMe->q16look),
                int(gMe->pXSprite ? gMe->pXSprite->health : 0));
        fflush(trajectory);
    }

    // The extended sector a push surface operates, or -1 when the surface is
    // a mechanism in its own right.  Several walls around one door sector
    // all drive that sector, and to the bot they are one thing to do.
    // The sector listening on this channel.  Cached: the scan is over every
    // sector, and the wiring does not change during a level.
    int mechanismReceiver(int tx) const
    {
        if (tx <= 0)
            return -1;
        auto known = mechanismReceivers.find(tx);
        if (known != mechanismReceivers.end())
            return known->second;
        int found = -1;
        for (int i = 0; i < numsectors; ++i)
        {
            const int extra = sector[i].extra;
            if (extra > 0 && extra < kMaxXSectors && xsector[extra].rxID == tx)
            {
                found = i;
                break;
            }
        }
        mechanismReceivers[tx] = found;
        return found;
    }

    int mechanismSector(const Portal &target, int targetSector) const
    {
        // A channel names the mechanism outright, whichever surface is used.
        if (target.wallPush && target.mechanismTx > 0)
        {
            const int receiver = mechanismReceiver(target.mechanismTx);
            if (receiver >= 0)
                return receiver;
        }
        if (target.sectorPush && inRange(target.to, 0, numsectors))
            return target.to;
        if (target.sectorPushCurrent && inRange(target.from, 0, numsectors))
            return target.from;
        if (target.wallPush && inRange(targetSector, 0, numsectors)
            && sector[targetSector].extra > 0
            && sector[targetSector].extra < kMaxXSectors)
            return targetSector;
        return -1;
    }

    int interactionKey(const InteractionCandidate &candidate) const
    {
        // Several linedefs can expose the same XSECTOR target.  Their engine
        // identity is the sector mechanism, not each duplicate wall record,
        // so one activation must be remembered once.
        if (candidate.kind == kInteractionWall)
        {
            const int mechanism = mechanismSector(candidate.target, candidate.targetSector);
            if (mechanism >= 0)
                return int(kInteractionSector) * 1000000 + mechanism + 1;
        }
        return int(candidate.kind) * 1000000 + candidate.id + 1;
    }

    int interactionMemoryKey(const InteractionMemory &memory) const
    {
        if (memory.kind == kInteractionWall)
        {
            const int mechanism = mechanismSector(memory.target, memory.targetSector);
            if (mechanism >= 0)
                return int(kInteractionSector) * 1000000 + mechanism + 1;
        }
        return int(memory.kind) * 1000000 + memory.id + 1;
    }

    int interactionStateSignature(const InteractionMemory &memory) const
    {
        int signature = int(memory.kind) * 131 + memory.id;
        if (memory.kind == kInteractionWall && memory.target.sectorPush
            && inRange(memory.target.to, 0, numsectors))
        {
            const sectortype &sectorRecord = sector[memory.target.to];
            signature = signature * 31 + sectorRecord.floorz;
            signature = signature * 31 + sectorRecord.ceilingz;
            if (sectorRecord.extra > 0 && sectorRecord.extra < kMaxXSectors)
            {
                signature = signature * 31 + xsector[sectorRecord.extra].state;
                signature = signature * 31 + xsector[sectorRecord.extra].busy;
            }
        }
        else if (memory.kind == kInteractionWall && memory.target.sectorPushCurrent
                 && inRange(memory.target.from, 0, numsectors))
        {
            const sectortype &sectorRecord = sector[memory.target.from];
            signature = signature * 31 + sectorRecord.floorz;
            signature = signature * 31 + sectorRecord.ceilingz;
            if (sectorRecord.extra > 0 && sectorRecord.extra < kMaxXSectors)
            {
                signature = signature * 31 + xsector[sectorRecord.extra].state;
                signature = signature * 31 + xsector[sectorRecord.extra].busy;
            }
        }
        else if (memory.kind == kInteractionWall && inRange(memory.target.wall, 0, numwalls))
        {
            const walltype &wallRecord = wall[memory.target.wall];
            signature = signature * 31 + wallRecord.cstat;
            signature = signature * 31 + wallRecord.nextsector;
            signature = signature * 31 + wallRecord.x;
            signature = signature * 31 + wallRecord.y;
            if (inRange(wallRecord.point2, 0, numwalls))
            {
                signature = signature * 31 + wall[wallRecord.point2].x;
                signature = signature * 31 + wall[wallRecord.point2].y;
            }
            if (wallRecord.extra > 0 && wallRecord.extra < kMaxXWalls)
            {
                signature = signature * 31 + xwall[wallRecord.extra].state;
                signature = signature * 31 + xwall[wallRecord.extra].busy;
            }
        }

        else if (memory.kind == kInteractionSector && inRange(memory.fromSector, 0, numsectors))
        {
            const sectortype &sectorRecord = sector[memory.fromSector];
            signature = signature * 31 + sectorRecord.floorz;
            signature = signature * 31 + sectorRecord.ceilingz;
            if (sectorRecord.extra > 0 && sectorRecord.extra < kMaxXSectors)
            {
                signature = signature * 31 + xsector[sectorRecord.extra].state;
                signature = signature * 31 + xsector[sectorRecord.extra].busy;
            }
        }
        else if (memory.kind == kInteractionSprite && inRange(memory.id, 0, kMaxSprites))
        {
            const spritetype &spriteRecord = sprite[memory.id];
            signature = signature * 31 + spriteRecord.x;
            signature = signature * 31 + spriteRecord.y;
            signature = signature * 31 + spriteRecord.sectnum;
            if (validXSprite(spriteRecord.extra))
            {
                signature = signature * 31 + xsprite[spriteRecord.extra].state;
                signature = signature * 31 + xsprite[spriteRecord.extra].busy;
            }
        }
        return signature;
    }

    bool interactionTargetMatches(const InteractionMemory &memory, int hit, int target) const
    {
        if (memory.kind == kInteractionWall)
        {
            if (memory.target.wallPush && hit == 0 && target == memory.target.wall)
                return true;
            // A door ringed by several push walls answers on whichever of
            // them the player is facing; they are all the same mechanism.
            // The push surfaces usually belong to the moving sector itself
            // and face outward, so accept a wall on either side of it.
            if (memory.target.wallPush && hit == 0 && inRange(target, 0, numwalls))
            {
                const int mechanism = mechanismSector(memory.target, memory.targetSector);
                if (!inRange(wall[target].extra, 1, kMaxXWalls)
                    || !xwall[wall[target].extra].triggerPush)
                    return false;
                if (memory.target.mechanismTx > 0
                    && xwall[wall[target].extra].txID == memory.target.mechanismTx)
                    return true;
                if (mechanism >= 0
                    && (wall[target].nextsector == mechanism
                        || wallOwnerSector(target) == mechanism))
                    return true;
            }
            if (memory.target.sectorPush && hit == 6 && target == memory.target.to)
                return true;
            if (memory.target.sectorPushCurrent && hit == 6 && target == memory.target.from)
                return true;
        }
        if (memory.kind == kInteractionSector)
            return hit == 6 && target == memory.fromSector;
        if (memory.kind == kInteractionSprite)
            return hit == 3 && target == memory.id;
        return false;
    }

    int interactionLookTarget(const InteractionMemory &memory, bool crouched) const
    {
        if (!gMe || !gMe->pSprite)
            return 0;
        const POSTURE &posture = gMe->pPosture[gMe->lifeMode]
            [crouched ? kPostureCrouch : kPostureStand];
        const int eyeZ = gMe->pSprite->z - posture.eyeAboveZ;
        const int margin = 128;
        const int lower = memory.target.interactionTopZ + margin;
        const int upper = memory.target.interactionBottomZ - margin;
        const int targetZ = lower <= upper ? std::max(lower, std::min(upper, eyeZ))
                                           : memory.target.z;
        const int horizontal = int(std::sqrt(double(distance2(
            observation.x, observation.y, memory.target.x, memory.target.y))));
        return lookAngleForTarget(eyeZ, targetZ, horizontal);
    }

    int interactionFacingAngle(const InteractionMemory &memory) const
    {
        if (memory.kind != kInteractionWall || memory.target.wall < 0)
            return memory.kind == kInteractionSector
                ? observation.angle
                : getangle(memory.x - observation.x, memory.y - observation.y);

        // Sector-push ActionScan targets are resolved against the sector's
        // usable opening and have historically worked from the approach
        // point.  Only a genuine wall-push target needs the wall-face normal;
        // treating every sector-push marker as a wall face makes the bot
        // strafe sideways and can lose the engine's sector ActionScan target.
        if (!memory.target.wallPush)
            return getangle(memory.x - observation.x, memory.y - observation.y);

        // ActionScan is directional.  For a wall-backed sector push, the
        // sector centroid is not a useful facing target: the player must face
        // the movable wall surface itself.  Choose the wall normal pointing
        // toward the observed approach side.  This also covers one-sided
        // push walls and the canonical sector-push records that represent the
        // same mechanism through several adjacent wall markers.
        const int tangentX = memory.target.x2 - memory.target.x1;
        const int tangentY = memory.target.y2 - memory.target.y1;
        const int normal1X = -tangentY;
        const int normal1Y = tangentX;
        const int normal2X = -normal1X;
        const int normal2Y = -normal1Y;
        const int toX = memory.target.x - observation.x;
        const int toY = memory.target.y - observation.y;
        const int64_t dot1 = int64_t(normal1X) * toX + int64_t(normal1Y) * toY;
        const int64_t dot2 = int64_t(normal2X) * toX + int64_t(normal2Y) * toY;
        return getangle(dot2 > dot1 ? normal2X : normal1X,
                        dot2 > dot1 ? normal2Y : normal1Y);
    }

    bool interactionNeedsCrouch(const InteractionMemory &memory) const
    {
        if (!gMe || !gMe->pSprite)
            return false;
        if (memory.target.interactionBottomZ <= memory.target.interactionTopZ)
            return false;
        // A large upward sector mechanism is commonly operated from a low
        // approach pose even when the adjacent portal is technically
        // traversable.  This is a capability/geometry trial, not a map ID
        // special case; ActionScanPreview remains the final authority.
        if (memory.target.sectorPush && memory.target.floorDelta < -kMaxWalkableStep)
            return true;
        const POSTURE &stand = gMe->pPosture[gMe->lifeMode][kPostureStand];
        const POSTURE &crouch = gMe->pPosture[gMe->lifeMode][kPostureCrouch];
        const int horizontal = int(std::sqrt(double(distance2(
            observation.x, observation.y, memory.target.x, memory.target.y))));
        const int margin = 128;
        const int lower = memory.target.interactionTopZ + margin;
        const int upper = memory.target.interactionBottomZ - margin;
        const int standingEye = gMe->pSprite->z - stand.eyeAboveZ;
        const int crouchedEye = gMe->pSprite->z - crouch.eyeAboveZ;
        const int standingTarget = lower <= upper ? std::max(lower, std::min(upper, standingEye))
                                                  : memory.target.z;
        const int crouchedTarget = lower <= upper ? std::max(lower, std::min(upper, crouchedEye))
                                                  : memory.target.z;
        const int standingRaw = int(std::lround(std::atan2(
            double(standingEye - standingTarget),
            double(std::max(1, horizontal))) * 1024.0 / 3.14159265358979323846));
        const int crouchedRaw = int(std::lround(std::atan2(
            double(crouchedEye - crouchedTarget),
            double(std::max(1, horizontal))) * 1024.0 / 3.14159265358979323846));
        return standingRaw < kLookDownLimit && crouchedRaw >= kLookDownLimit;
    }

    void rememberInteraction(const InteractionCandidate &candidate)
    {
        const int key = interactionKey(candidate);
        InteractionMemory &memory = interactions[key];
        const bool first = !memory.observed;
        const int oldState = memory.observed ? interactionStateSignature(memory) : 0;
        const bool canonicalSectorPush = candidate.kind == kInteractionWall
            && mechanismSector(candidate.target, candidate.targetSector) >= 0;
        memory.kind = candidate.kind;
        const int candidateMechanism = candidate.kind == kInteractionWall
            ? mechanismSector(candidate.target, candidate.targetSector) : -1;
        memory.id = candidateMechanism >= 0 ? candidateMechanism : candidate.id;
        if (candidateMechanism >= 0 && candidate.targetSector < 0)
            memory.targetSector = candidateMechanism;
        const bool sideChanged = memory.observed && memory.fromSector != candidate.fromSector;
        // Among several surfaces driving one mechanism, the useful one is the
        // one the player can reach and face right now.
        const bool closerSurface = memory.observed && candidate.kind == kInteractionWall
            && candidate.target.wallPush && memory.target.wallPush
            && candidate.target.wall != memory.target.wall
            && distance2(observation.x, observation.y, candidate.x, candidate.y)
                < distance2(observation.x, observation.y, memory.x, memory.y);
        memory.fromSector = candidate.fromSector;
        memory.targetSector = candidate.targetSector;
        // A reversible mechanism is usable from more than one side, and the
        // side that matters is the one the bot is standing on now.  Keeping
        // the first-seen approach point sent the bot at a pose it could no
        // longer reach once the door it came through closed behind it.
        if (first || !canonicalSectorPush || sideChanged || closerSurface)
        {
            memory.x = candidate.x;
            memory.y = candidate.y;
            memory.z = candidate.z;
            if (closerSurface && !sideChanged)
            {
                char detail[176];
                snprintf(detail, sizeof(detail),
                         "mechanism=%d from_wall=%d to_wall=%d at=(%d,%d)",
                         memory.id, memory.target.wall, candidate.target.wall,
                         candidate.x, candidate.y);
                event("interaction_surface_reselected", detail);
            }
            if (sideChanged)
            {
                char detail[160];
                snprintf(detail, sizeof(detail),
                         "kind=%d id=%d approach_sector=%d at=(%d,%d)",
                         int(candidate.kind), memory.id, candidate.fromSector,
                         candidate.x, candidate.y);
                event("interaction_approach_side_changed", detail);
            }
        }
        // Several wall markers can dispatch the same sector ActionScan
        // target.  Keep the first observed approach point: unlike a genuine
        // wall-push, the sector target is resolved through the usable portal
        // and the nearest marker is not necessarily the pose from which
        // ActionScan can see that sector.
        memory.key = candidate.key;
        memory.locked = candidate.locked;
        memory.reversible = candidate.reversible;
        memory.observed = true;
        memory.target = candidate.target;
        const int newState = interactionStateSignature(memory);
        if (memory.unavailableFromSector >= 0 && memory.unavailableState != newState)
        {
            memory.unavailableFromSector = -1;
            memory.unavailableAttempts = 0;
            memory.unavailablePose = 0;
        }
        // One delta per activation, not one per observation.  An activation
        // transaction begins when the bot acts and ends when the world
        // answers; without that bound a burst of shots at one wall
        // rediscovered the same effect every tick, and each rediscovery
        // refreshed the semantic-progress clock and hid a real stall.
        if (memory.state == 1 && !memory.observedKnownWorldDelta
            && newState != memory.beforeState)
        {
            memory.state = 2;
            memory.activated = true;
            memory.observedLocalEffect = true;
            memory.observedKnownWorldDelta = true;
            recoveryMode = false;
            memory.afterState = newState;
            char detail[128];
            snprintf(detail, sizeof(detail), "kind=%d id=%d before=%d after=%d", int(memory.kind), memory.id,
                     memory.beforeState, newState);
            event("interaction_world_delta", detail);
            // The bot just changed the world on purpose.  Anything it had
            // given up on around this mechanism deserves another look: on
            // AGTST4 the frontier into the door's own sector went dormant
            // seconds before the door was opened, so when it did open there
            // was no live opportunity left to carry the bot through it.
            rearmAroundMechanism(memory);
            lastSemanticProgressTick = observation.tick;
        }
        updateFollowThrough(memory);
        if (first)
        {
            recoveryMode = false;
            ++knowledgeRevision;
            char detail[128];
            snprintf(detail, sizeof(detail), "kind=%d id=%d sector=%d key=%d", int(memory.kind), memory.id,
                     memory.fromSector, memory.key);
            event("discovered_interaction", detail);
            lastSemanticProgressTick = observation.tick;
        }
        else if (oldState != newState && memory.state != 1)
        {
            // State refresh is local knowledge maintenance.  It does not
            // expose objects in sectors the bot has never observed.
            event("interaction_state_refreshed", "source=known_world_revisit");
        }
    }

    // Re-arm work that a mechanism's effect could plausibly have unblocked.
    // Keyed on the sector the mechanism drives, so this stays a statement
    // about the machine rather than about any particular map.
    void rearmAroundMechanism(const InteractionMemory &memory)
    {
        const int driven = memory.targetSector >= 0
            ? memory.targetSector
            : mechanismSector(memory.target, memory.targetSector);
        if (!inRange(driven, 0, numsectors))
            return;
        int rearmed = 0;
        const sectortype &record = sector[driven];
        for (int i = 0; i < record.wallnum; ++i)
        {
            const int wallId = record.wallptr + i;
            if (!inRange(wallId, 0, numwalls))
                continue;
            if (suppressedUntil.erase(3000000 + wallId * 8 + int(kObjectiveFrontier)))
                ++rearmed;
            suppressionCount.erase(3000000 + wallId * 8 + int(kObjectiveFrontier));
            const int neighbour = wall[wallId].nextsector;
            if (inRange(neighbour, 0, numsectors))
            {
                failedEdges.erase(wallId * 65536 + neighbour);
                localFailureSignatures.erase(wallId * 65536 + neighbour);
                if (inRange(wall[wallId].nextwall, 0, numwalls))
                {
                    failedEdges.erase(wall[wallId].nextwall * 65536 + driven);
                    if (suppressedUntil.erase(3000000 + wall[wallId].nextwall * 8
                                              + int(kObjectiveFrontier)))
                        ++rearmed;
                }
            }
        }
        if (suppressedUntil.erase(4000000 + driven))
            ++rearmed;
        navEdgeFailures.erase(
            std::remove_if(navEdgeFailures.begin(), navEdgeFailures.end(),
                           [&](const NavEdgeFailure &failure)
                           {
                               return failure.wall >= 0 && inRange(failure.wall, 0, numwalls)
                                   && (wallOwnerSector(failure.wall) == driven
                                       || wall[failure.wall].nextsector == driven);
                           }),
            navEdgeFailures.end());
        if (rearmed > 0)
        {
            char detail[160];
            snprintf(detail, sizeof(detail), "mechanism_sector=%d rearmed=%d", driven, rearmed);
            event("opportunities_rearmed_by_world_change", detail);
        }
    }

    // Whether the engine accepted an activation aimed at this boundary.
    // Asked rather than mirrored: the door record used to carry its own copy
    // of this fact, and nothing wrote that copy on the path that actually
    // resolves a Use, so an activation the engine had accepted was reported
    // seconds later as having produced no response.  Execution state has one
    // owner, and it is the interaction ledger.
    bool activationAccepted(int wallId) const
    {
        if (wallId < 0)
            return false;
        for (const auto &entry : interactions)
            if (entry.second.engineAccepted && entry.second.target.wall == wallId)
                return true;
        return false;
    }

    void updateKnowledge()
    {
        observation = observeWorld();
        if (observation.sector < 0)
            return;
        if (gMe->posture == kPostureCrouch)
        {
            const int crouched = playerBodyClearance();
            if (crouched > 0
                && (gObservedCrouchClearance == 0 || crouched < gObservedCrouchClearance))
            {
                const int standing = gStandingClearance;
                gObservedCrouchClearance = crouched;
                char detail[128];
                snprintf(detail, sizeof(detail), "crouch_clearance=%d standing_clearance=%d",
                         crouched, standing);
                event("crouch_envelope_measured", detail);
            }
        }
        else if (gMe->posture == kPostureStand)
            gStandingClearance = playerBodyClearance();
        healthLostSinceObservation = lastObservedHealth >= 0
            ? std::max(0, lastObservedHealth - observation.health) : 0;
        if (healthLostSinceObservation > 0)
            lastDamageTick = observation.tick;
        lastObservedHealth = observation.health;
        if (observation.sector != lastGeometryTelemetrySector)
        {
            char detail[160];
            snprintf(detail, sizeof(detail),
                     "sector=%d clearance=%d player_ceiling=%d player_floor=%d portals=%d jumpable=%d max_rise=%d max_drop=%d xsector=%d state=%d busy=%d zvel=%d",
                     observation.sector, observation.localClearance,
                     observation.playerCeilingDistance, observation.playerFloorDistance,
                     observation.localPortalCount,
                     observation.localJumpableCount, observation.localMaxRise, observation.localMaxDrop,
                     observation.localSectorExtra, observation.localSectorState,
                     observation.localSectorBusy, observation.playerZVelocity);
            event("local_geometry", detail);
            lastGeometryTelemetrySector = observation.sector;
        }
        const size_t oldSectors = observedSectors.size();
        observedSectors.insert(observation.visibleSectors.begin(), observation.visibleSectors.end());
        const int previousSector = lastObservedSector;
        const bool sectorChanged = observation.sector != lastObservedSector;
        if (sectorChanged)
        {
            localDynamicJumpAttempted = false;
            localDynamicJumpUntilTick = -1;
            localJumpProbeCount = 0;
            // Ground the bot has not stood on before is a genuinely new
            // situation and cancels the probe.  Being shoved back and forth
            // between two rooms it already knows is not: restarting on every
            // such bounce recomputed the endpoint from the player's position
            // and so let it track him after all, which is the thing fixing
            // the endpoint was meant to stop.
            if (!visitedSectors.count(observation.sector))
                searchProbeActive = false;
        }
        if (observation.localSectorBusy != 0 && observation.playerZVelocity != 0)
            localDynamicJumpUntilTick = observation.tick + 4 * kTicsPerSec;
        // Which boundary the player physically crossed is a fact about the
        // two sectors, not about what the route was aiming at.  A route's
        // final destination wall is frequently several rooms further on, and
        // recording it here wrote a connection into the world graph that
        // does not exist -- which then excluded the real frontier from
        // future exploration.
        const int crossedWall = wallJoinsSectors(movementTargetId, previousSector,
                                                 observation.sector)
            ? movementTargetId
            : crossingWallBetween(previousSector, observation.sector);
        if (sectorChanged && previousSector >= 0 && movementTargetActive
            && movementTargetFrom == previousSector
            && movementTargetSector == observation.sector && crossedWall >= 0)
        {
            const int edgeId = crossedWall * 65536 + observation.sector;
            visitedEdges.insert(edgeId);
            failedEdges.erase(edgeId);
            char detail[160];
            snprintf(detail, sizeof(detail), "from=%d to=%d wall=%d aimed_at=%d",
                     previousSector, observation.sector, crossedWall, movementTargetId);
            event("portal_traversed", detail);

            const bool isActiveFrontier = currentObjective.active
                && currentObjective.type == kObjectiveFrontier
                && currentObjective.wall == movementTargetId
                && currentObjective.sector == previousSector
                && currentObjective.targetSector == observation.sector;
            if (isActiveFrontier)
            {
                retirePendingFrontier(movementTargetId, previousSector, observation.sector);
                char frontierDetail[192];
                snprintf(frontierDetail, sizeof(frontierDetail),
                         "wall=%d source=%d target=%d current=%d state=CONSUMED",
                         movementTargetId, previousSector, observation.sector,
                         observation.sector);
                event("frontier_crossed", frontierDetail);
            }
            else if (currentObjective.active && currentObjective.type == kObjectiveFrontier)
            {
                // This was transportation toward the committed frontier, not
                // a replacement exploration objective.  Keep the original
                // work identity intact and make the route step explicit.
                char routeDetail[224];
                snprintf(routeDetail, sizeof(routeDetail),
                         "frontier_wall=%d source=%d target=%d route_wall=%d from=%d to=%d",
                         currentObjective.wall, currentObjective.sector,
                         currentObjective.targetSector, movementTargetId,
                         previousSector, observation.sector);
                event("frontier_route_step", routeDetail);
            }
            for (auto &entry : interactions)
            {
                InteractionMemory &memory = entry.second;
                if (memory.kind == kInteractionWall && memory.target.wall == movementTargetId
                    && memory.target.from == previousSector
                    && memory.target.to == observation.sector)
                {
                    memory.traversed = true;
                    memory.afterState = interactionStateSignature(memory);
                    event("interaction_route_traversed", "source=portal_transition");
                }
            }
            lastSemanticProgressTick = observation.tick;
        }
        if (sectorChanged && previousSector >= 0 && followThroughKey >= 0
            && previousSector == followThroughFrom
            && observation.sector == followThroughPortal.to)
        {
            char detail[256];
            snprintf(detail, sizeof(detail),
                     "before_sector=%d after_sector=%d interaction_key=%d portal_wall=%d portal_from=%d portal_to=%d engine_accepted=1 game_time=%d",
                     previousSector, observation.sector, followThroughKey,
                     followThroughPortal.wall, followThroughPortal.from,
                     followThroughPortal.to, (gFrame * kTicsPerFrame) / kTicsPerSec);
            event("interaction_follow_through_crossed", detail);
            if (!cryptExitProven)
            {
                event("CRYPT_EXIT_CROSSED", detail);
                cryptExitProven = true;
            }
            followThroughKey = -1;
            followThroughStartTick = -1;
            followThroughFrom = -1;
            followThroughPortal = Portal{};
            lastSemanticProgressTick = observation.tick;
        }
        if (sectorChanged && previousSector >= 0 && movementTargetId >= 0)
        {
            auto door = doors.find(movementTargetId);
            if (door != doors.end() && door->second.portal.interactionAffordance
                && door->second.portal.from == previousSector
                && door->second.portal.to == observation.sector)
            {
                const bool usedRecently = lastUsedDoor == movementTargetId
                    && lastUsedDoorFrom == previousSector
                    && lastUsedDoorTo == observation.sector
                    && observation.tick - lastUsedDoorTick <= 4 * kTicsPerSec;
                if (!usedRecently)
                {
                    door->second.opened = true;
                    door->second.interaction = DoorMemory::kOpen;
                    door->second.unavailableFromSector = -1;
                    event("door_open", "response=authoritative_traversable_without_use");
                }
            }
        }
        if (sectorChanged && movementTargetSector == observation.sector)
        {
            movementTargetActive = false;
            currentGoal.clear();
            currentGoalTarget = -1;
        }
        if (sectorChanged && lastUsedDoor >= 0
            && previousSector == lastUsedDoorFrom && observation.sector == lastUsedDoorTo
            && observation.tick - lastUsedDoorTick <= 4 * kTicsPerSec)
        {
            auto door = doors.find(lastUsedDoor);
            if (door != doors.end())
            {
                const bool alreadyOpen = door->second.interaction == DoorMemory::kOpen;
                door->second.interaction = DoorMemory::kOpen;
                door->second.interactionDeadlineTick = -1;
                if (!alreadyOpen)
                    event("door_open", "response=sector_crossing");
                visitedEdges.insert(lastUsedDoor * 65536 + lastUsedDoorTo);
                openedRoutes.insert(lastUsedDoorFrom * 65536 + lastUsedDoorTo);
                openedRoutes.insert(lastUsedDoorTo * 65536 + lastUsedDoorFrom);
                for (auto &entry : doors)
                {
                    DoorMemory &candidate = entry.second;
                    if (!candidate.portal.key && !candidate.portal.locked
                        && candidate.portal.from == lastUsedDoorFrom
                        && candidate.portal.to == lastUsedDoorTo)
                    {
                        candidate.opened = true;
                        visitedEdges.insert(candidate.portal.wall * 65536 + candidate.portal.to);
                    }
                    if (!candidate.portal.key && !candidate.portal.locked
                        && candidate.portal.from == lastUsedDoorTo
                        && candidate.portal.to == lastUsedDoorFrom)
                        visitedEdges.insert(candidate.portal.wall * 65536 + candidate.portal.to);
                }
                char detail[96];
                snprintf(detail, sizeof(detail), "door=%d from=%d to=%d", lastUsedDoor,
                         lastUsedDoorFrom, lastUsedDoorTo);
                event("door_traversed", detail);
                lastSemanticProgressTick = observation.tick;
            }
            lastUsedDoor = -1;
        }
        else if (lastUsedDoor >= 0 && observation.tick - lastUsedDoorTick > 4 * kTicsPerSec)
        {
            lastUsedDoor = -1;
        }
        if (sectorChanged && previousSector >= 0)
        {
            const bool reverse = lastTransitionFrom == observation.sector
                && lastTransitionTo == previousSector
                && lastTransitionKnowledge == knowledgeRevision
                && lastTransitionInventory == inventoryRevision;
            if (reverse)
                ++repeatedBacktrackCount;
            else
                repeatedBacktrackCount = 0;
            if (repeatedBacktrackCount >= 2)
            {
                char detail[128];
                snprintf(detail, sizeof(detail), "from=%d to=%d count=%d", previousSector,
                         observation.sector, repeatedBacktrackCount);
                event("repeated_backtrack", detail);
            }
        }
        if (clearingSector >= 0 && clearingSector != observation.sector)
        {
            clearingSector = -1;
            clearingWall = -1;
        }
        if (visitedSectors.insert(observation.sector).second)
        {
            ++knowledgeRevision;
            TreeNode node;
            node.parent = previousSector;
            node.enteredTick = observation.tick;
            auto parent = explorationTree.find(previousSector);
            node.depth = parent == explorationTree.end() ? 0 : parent->second.depth + 1;
            explorationTree[observation.sector] = node;
            char detail[128];
            snprintf(detail, sizeof(detail), "sector=%d parent=%d depth=%d branch=%d",
                     observation.sector, node.parent, node.depth, branchRoot(observation.sector));
            event("sector_entered", detail);
            rememberEnteredSectorBoundaries(observation.sector);
        }
        if (sectorChanged)
        {
            lastTransitionFrom = previousSector;
            lastTransitionTo = observation.sector;
            lastTransitionTick = observation.tick;
            lastTransitionKnowledge = knowledgeRevision;
            lastTransitionInventory = inventoryRevision;
            lastObservedSector = observation.sector;
        }
        if (observedSectors.size() != oldSectors)
        {
            lastSemanticProgressTick = observation.tick;
            event("discovered_sector");
        }

        trackBoundaryStates();
        trackCombatOutcomes();
        trackEnvironmentalDamage();
        updateLocalCoverage();

        for (const Portal &portal : observation.portals)
        {
            const int edgeId = portal.wall * 65536 + portal.to;
            auto localFailure = localFailureSignatures.find(edgeId);
            if (localFailure != localFailureSignatures.end()
                && localFailure->second != portalGeometrySignature(portal))
                localFailureSignatures.erase(localFailure);
            const bool newEdge = seenEdges.insert(edgeId).second;
            if (newEdge && !visitedSectors.count(portal.to))
            {
                recoveryMode = false;
                ++knowledgeRevision;
                lastSemanticProgressTick = observation.tick;
                char detail[96];
                snprintf(detail, sizeof(detail), "wall=%d from=%d to=%d", portal.wall,
                         portal.from, portal.to);
                event("discovered_frontier", detail);
            }
            auto &edges = knownGraph[portal.from];
            auto edge = std::find_if(edges.begin(), edges.end(), [&portal](const Portal &known)
            {
                return known.wall == portal.wall && known.to == portal.to;
            });
            // Keep blocked boundaries as knowledge.  They are investigation
            // frontiers, not deleted topology.
            if (edge == edges.end())
                edges.push_back(portal);
            else
                *edge = portal;
            if (portal.traversable || portal.jumpable)
            {
                // A visible two-sided Build opening is a known route in both
                // directions.  Keeping only the observed wall direction made
                // DFS backtracking impossible after a one-way sector entry.
                Portal reverse = portal;
                reverse.from = portal.to;
                reverse.to = portal.from;
                reverse.floorZ = portal.fromFloorZ;
                reverse.ceilingZ = portal.fromCeilingZ;
                reverse.fromFloorZ = portal.toFloorZ;
                reverse.fromCeilingZ = portal.toCeilingZ;
                reverse.toFloorZ = portal.fromFloorZ;
                reverse.toCeilingZ = portal.fromCeilingZ;
                reverse.floorDelta = -portal.floorDelta;
                int reverseWallId = -1;
                if (inRange(reverse.from, 0, numsectors))
                {
                    const sectortype &reverseSector = sector[reverse.from];
                    for (int i = 0; i < reverseSector.wallnum; ++i)
                    {
                        const int candidateWallId = reverseSector.wallptr + i;
                        if (!inRange(candidateWallId, 0, numwalls))
                            continue;
                        const walltype &candidateWall = wall[candidateWallId];
                        const walltype &candidateEnd = wall[candidateWall.point2];
                        const bool sameSegment =
                            (candidateWall.x == portal.x1 && candidateWall.y == portal.y1
                             && candidateEnd.x == portal.x2 && candidateEnd.y == portal.y2)
                            || (candidateWall.x == portal.x2 && candidateWall.y == portal.y2
                                && candidateEnd.x == portal.x1 && candidateEnd.y == portal.y1);
                        if (candidateWall.nextsector == reverse.to && sameSegment)
                        {
                            reverseWallId = candidateWallId;
                            break;
                        }
                    }
                }
                if (reverseWallId < 0)
                    reverseWallId = -1;
                if (reverseWallId >= 0)
                {
                reverse.wall = reverseWallId;
                // Reclassify the opposite direction from the real opposite
                // wall/sector records.  A drop in one direction can be a
                // jump or a blocked rise in the other; copying the original
                // mode would corrupt NavTopology and walkArea.
                const walltype &reverseWall = wall[reverseWallId];
                const bool reverseEnoughWidth = reverse.openingWidth >= playerPassageWidth();
                const bool reverseStanding = reverse.clearance >= playerBodyClearance();
                const bool reverseCrouching = reverse.clearance >= playerCrouchClearance();
                const int reverseRise = playerJumpRiseLimit();
                const bool reverseWalk = reverseEnoughWidth && reverseStanding
                    && std::abs(reverse.floorDelta) <= kMaxWalkableStep;
                const bool reverseJump = reverseEnoughWidth && reverseStanding
                    && reverse.floorDelta < 0 && -reverse.floorDelta <= reverseRise;
                const bool reverseDrop = reverseEnoughWidth && reverseStanding
                    && reverse.floorDelta > 0 && reverse.floorDelta <= reverseRise;
                reverse.walkable = !(reverseWall.cstat & 1) && reverseWalk;
                reverse.crouchable = !(reverseWall.cstat & 1) && !reverseStanding
                    && reverseCrouching && std::abs(reverse.floorDelta) <= kMaxWalkableStep;
                reverse.jumpable = !(reverseWall.cstat & 1) && !reverseWalk && reverseJump;
                reverse.dropSafe = !(reverseWall.cstat & 1) && reverseDrop;
                reverse.traversable = reverse.walkable || reverse.crouchable
                    || reverse.jumpable || reverse.dropSafe;
                reverse.capability = reverse.walkable ? kTraversalWalkable
                    : reverse.crouchable ? kTraversalCrouchable
                    : reverse.jumpable ? kTraversalJumpable
                    : reverse.dropSafe ? kTraversalDropSafe
                    : kTraversalCurrentlyUnavailable;
                reverse.wallPush = false;
                reverse.key = 0;
                reverse.locked = false;
                reverse.wallState = 0;
                reverse.wallBusy = 0;
                if (reverseWall.extra > 0 && reverseWall.extra < kMaxXWalls)
                {
                    const XWALL &extra = xwall[reverseWall.extra];
                    reverse.wallPush = extra.triggerPush != 0;
                    reverse.key = extra.key;
                    reverse.locked = extra.locked != 0;
                    reverse.wallState = extra.state;
                    reverse.wallBusy = extra.busy;
                }
                reverse.sectorPush = false;
                reverse.sectorPushCurrent = false;
                reverse.sectorState = 0;
                reverse.sectorBusy = 0;
                if (inRange(reverse.to, 0, numsectors)
                    && sector[reverse.to].extra > 0 && sector[reverse.to].extra < kMaxXSectors)
                {
                    const XSECTOR &extra = xsector[sector[reverse.to].extra];
                    reverse.sectorPush = extra.Wallpush != 0;
                    reverse.sectorState = extra.state;
                    reverse.sectorBusy = extra.busy;
                }
                if (inRange(reverse.from, 0, numsectors)
                    && sector[reverse.from].extra > 0 && sector[reverse.from].extra < kMaxXSectors)
                    reverse.sectorPushCurrent = xsector[sector[reverse.from].extra].Push != 0;
                reverse.interactionAffordance = reverse.wallPush || reverse.sectorPush
                    || reverse.sectorPushCurrent;
                reverse.currentlyAvailable = reverse.traversable || reverse.interactionAffordance;
                auto &reverseEdges = knownGraph[reverse.from];
                auto reverseEdge = std::find_if(reverseEdges.begin(), reverseEdges.end(),
                                                [&reverse](const Portal &known)
                {
                    return known.wall == reverse.wall && known.to == reverse.to;
                });
                if (reverseEdge == reverseEdges.end())
                    reverseEdges.push_back(reverse);
                else
                    *reverseEdge = reverse;
                }
            }
            else if (inRange(portal.to, 0, numsectors))
            {
                // The cached reverse route describes current topology, not
                // historical reachability.  Remove it when this re-observed
                // opening is no longer usable.
                auto &reverseEdges = knownGraph[portal.to];
                for (auto reverseEdge = reverseEdges.begin(); reverseEdge != reverseEdges.end(); )
                {
                    if (reverseEdge->to == portal.from
                        && reverseEdge->x1 == portal.x1 && reverseEdge->y1 == portal.y1
                        && reverseEdge->x2 == portal.x2 && reverseEdge->y2 == portal.y2)
                        reverseEdge = reverseEdges.erase(reverseEdge);
                    else
                        ++reverseEdge;
                }
            }
            const bool blockedInteractivePortal = portal.interactionAffordance;
            if (portal.key || portal.locked || blockedInteractivePortal
                || (!portal.traversable && !portal.jumpable))
            {
                DoorMemory &door = doors[portal.wall];
                if (door.id < 0)
                {
                    door.id = portal.wall;
                    ++knowledgeRevision;
                    char detail[128];
                    snprintf(detail, sizeof(detail), "door=%d key=%d locked=%d wall_push=%d sector_push=%d direct_use=%d",
                             portal.wall, portal.key, portal.locked ? 1 : 0,
                             portal.wallPush ? 1 : 0, portal.sectorPush ? 1 : 0,
                             portal.directUse ? 1 : 0);
                    event("discovered_door", detail);
                }
                const Portal previousPortal = door.portal;
                const bool hadPortal = door.id >= 0;
                const bool response = hadPortal
                    && (previousPortal.wallState != portal.wallState
                        || previousPortal.wallBusy != portal.wallBusy
                        || previousPortal.sectorState != portal.sectorState
                        || previousPortal.sectorBusy != portal.sectorBusy
                        || previousPortal.floorZ != portal.floorZ
                        || previousPortal.ceilingZ != portal.ceilingZ
                        || previousPortal.floorDelta != portal.floorDelta
                        || previousPortal.x1 != portal.x1 || previousPortal.y1 != portal.y1
                        || previousPortal.x2 != portal.x2 || previousPortal.y2 != portal.y2
                        || previousPortal.traversable != portal.traversable);
                door.portal = portal;
                if (response && door.opened && !portal.traversable
                    && portal.interactionAffordance)
                {
                    door.opened = false;
                    door.interaction = DoorMemory::kIdle;
                    door.interactionDeadlineTick = -1;
                    event("door_closed", "response=authoritative_geometry_changed");
                }
                if (response && (door.interaction == DoorMemory::kWaiting
                                 || door.interaction == DoorMemory::kOpening))
                {
                    if (door.interaction == DoorMemory::kWaiting)
                    {
                        door.interaction = DoorMemory::kOpening;
                        char detail[96];
                        snprintf(detail, sizeof(detail), "door=%d wall_busy=%d sector_busy=%d",
                                 portal.wall, portal.wallBusy, portal.sectorBusy);
                        event("door_opening", detail);
                    }
                    lastSemanticProgressTick = observation.tick;
                    if (!portal.directUse && portal.traversable && !portal.wallBusy && !portal.sectorBusy)
                    {
                        door.interaction = DoorMemory::kOpen;
                        event("door_open", "response=authoritative_traversable");
                    }
                }
                const bool unavailableFromThisSide = door.unavailableFromSector == observation.sector
                    && portal.from == observation.sector;
                const DoorMemory::Availability oldAvailability = door.availability;
                if (portal.key && !hasKey(portal.key))
                    door.availability = DoorMemory::kMissingKey;
                else if (unavailableFromThisSide)
                    door.availability = DoorMemory::kCurrentlyUnavailable;
                else if (portal.key || portal.locked || blockedInteractivePortal || portal.traversable)
                    door.availability = DoorMemory::kActionable;
                else
                {
                    door.availability = DoorMemory::kCurrentlyUnavailable;
                    door.unavailableReason = portal.unavailableReason;
                }
                if (door.availability == DoorMemory::kCurrentlyUnavailable
                    && oldAvailability != DoorMemory::kCurrentlyUnavailable)
                {
                    char detail[96];
                    snprintf(detail, sizeof(detail), "wall=%d reason=%d affordance=%d",
                             portal.wall, portal.unavailableReason,
                             portal.interactionAffordance ? 1 : 0);
                    event("portal_unavailable", detail);
                }
            }
        }

        // Refresh interaction memory from current engine state whenever the
        // area is observed again.  This includes one-sided walls and current
        // sector pushes, which never enter the portal graph.
        for (const InteractionCandidate &candidate : observation.interactions)
            rememberInteraction(candidate);

        for (const VisibleObject &object : observation.objects)
        {
            ObjectMemory &remembered = objectMemory[object.sprite];
            const bool firstObjectObservation = !remembered.observed;
            remembered.object = object;
            remembered.observed = true;
            remembered.collected = false;
            remembered.lastSeenTick = observation.tick;
            if (object.kind == kObjectKey && knownKeys.insert(object.type - kItemKeyBase + 1).second)
                ++knowledgeRevision;
            if (knownObjects.insert(object.sprite).second)
            {
                char detail[160];
                const char *category = itemCategory(object.type);
                snprintf(detail, sizeof(detail), "sprite=%d kind=%s category=%s sector=%d type=%d",
                         object.sprite, objectName(object.kind), category ? category : "none", object.sector, object.type);
                event("observed_object", detail);
                lastSemanticProgressTick = observation.tick;
                ++knowledgeRevision;
            }
            if (firstObjectObservation)
                event("object_remembered", object.kind == kObjectKey ? "kind=key" : "kind=pickup");
        }

        heldKeyMask = 0;
        for (int key = 1; key < 8; ++key)
        {
            if (!gMe->hasKey[key])
                continue;
            heldKeyMask |= 1u << unsigned(key);
            if (!heldKeys.insert(key).second)
                continue;
            // A new key re-arms every remembered door that wanted it, and
            // clears their dormancy so the progression model changes at
            // once rather than after a cooldown.
            int rearmed = 0;
            for (const auto &door : lockedDoorKey)
            {
                if (door.second != key)
                    continue;
                suppressedUntil.erase(3000000 + door.first * 8 + int(kObjectiveFrontier));
                suppressionCount.erase(3000000 + door.first * 8 + int(kObjectiveFrontier));
                ++rearmed;
            }
            char detail[96];
            snprintf(detail, sizeof(detail), "key=%d rearmed_doors=%d", key, rearmed);
            event("acquired_key", detail);
            event("key_acquired", detail);
            lastSemanticProgressTick = observation.tick;
            ++inventoryRevision;
        }

        for (auto &entry : doors)
        {
            DoorMemory &door = entry.second;
            if ((door.interaction == DoorMemory::kWaiting || door.interaction == DoorMemory::kOpening)
                && door.interactionDeadlineTick >= 0 && observation.tick > door.interactionDeadlineTick)
            {
                if (activationAccepted(door.id))
                {
                    door.interaction = DoorMemory::kOpening;
                    event("interaction_settled", "door_engine_accepted=1 observed_local_effect=0");
                }
                else
                {
                    door.interaction = DoorMemory::kFailed;
                    char detail[144];
                    snprintf(detail, sizeof(detail), "door=%d waited_ticks=%d reason=INTERACTION_VALID_BUT_NO_RESPONSE", door.id,
                             observation.tick - door.interactionStartedTick);
                    event("interaction_failed", detail);
                }
                door.interactionDeadlineTick = -1;
            }
            if (door.portal.key && hasKey(door.portal.key) && door.availability == DoorMemory::kMissingKey)
            {
                door.availability = DoorMemory::kActionable;
                ++knowledgeRevision;
                char detail[64];
                snprintf(detail, sizeof(detail), "door=%d key=%d", door.id, door.portal.key);
                event("door_ready_after_key", detail);
            }
        }

        if (pendingUse >= 0 && pendingUseTick < observation.tick)
        {
            auto door = doors.find(pendingUse);
            if (door != doors.end() && door->second.portal.key && !hasKey(door->second.portal.key))
            {
                door->second.availability = DoorMemory::kMissingKey;
                door->second.interaction = DoorMemory::kFailed;
                event("interaction_failed", "reason=missing_key");
                char detail[96];
                snprintf(detail, sizeof(detail), "door=%d key=%d", pendingUse, door->second.portal.key);
                event("blocked_key_required", detail);
            }
            pendingUse = -1;
        }
    }

    bool hasKey(int key) const
    {
        return key <= 0 || (key < 8 && gMe && gMe->hasKey[key]);
    }

    int portalGeometrySignature(const Portal &portal) const
    {
        // A failed approach is valid only for the geometry that was actually
        // tested.  Discovery of an unrelated pickup or sector must not make
        // every old failure disappear, while a changed wall/sector must.
        int signature = portal.wall * 31 + portal.from * 131 + portal.to * 977;
        signature = signature * 31 + portal.x1;
        signature = signature * 31 + portal.y1;
        signature = signature * 31 + portal.x2;
        signature = signature * 31 + portal.y2;
        signature = signature * 31 + portal.wallState;
        signature = signature * 31 + portal.wallBusy;
        signature = signature * 31 + portal.sectorState;
        signature = signature * 31 + portal.sectorBusy;
        signature = signature * 31 + portal.floorZ;
        signature = signature * 31 + portal.ceilingZ;
        signature = signature * 31 + portal.floorDelta;
        signature = signature * 31 + portal.clearance;
        signature = signature * 31 + (portal.traversable ? 1 : 0);
        signature = signature * 31 + (portal.crouchable ? 1 : 0);
        return signature;
    }

    int portalIdentitySignature(const Portal &portal) const
    {
        int signature = llmapper::mixHash(portal.wall, portal.from);
        signature = llmapper::mixHash(signature, portal.to);
        signature = llmapper::mixHash(signature, portal.wallState);
        signature = llmapper::mixHash(signature, portal.wallBusy != 0);
        signature = llmapper::mixHash(signature, portal.sectorState);
        signature = llmapper::mixHash(signature, portal.sectorBusy != 0);
        signature = llmapper::mixHash(signature, portal.traversable ? 1 : 0);
        signature = llmapper::mixHash(signature, portal.jumpable ? 1 : 0);
        signature = llmapper::mixHash(signature, portal.locked ? 1 : 0);
        signature = llmapper::mixHash(signature, portal.key);
        return signature;
    }

    // Cooldown for a route that one approach failed to execute.  Growing
    // backoff keeps a genuinely impossible crossing from being retried in a
    // tight loop, while a merely awkward one comes back into play.
    int failureCooldownTicks(int attempts) const
    {
        const int shift = attempts > 3 ? 3 : (attempts > 0 ? attempts - 1 : 0);
        return kSuppressionBaseTicks << shift;
    }

    void pruneNavEdgeFailures()
    {
        for (auto it = navEdgeFailures.begin(); it != navEdgeFailures.end(); )
        {
            if (it->expiresTick > 0 && observation.tick >= it->expiresTick)
            {
                char detail[128];
                snprintf(detail, sizeof(detail), "wall=%d from_cell=%d to_cell=%d attempts=%d",
                         it->wall, it->fromCell, it->toCell, it->attempts);
                event("nav_edge_failure_rearmed", detail);
                it = navEdgeFailures.erase(it);
                continue;
            }
            if (it->wall < 0)
            {
                ++it;
                continue;
            }
            const Portal *portal = portalByWallId(it->wall);
            if (portal && portalIdentitySignature(*portal) != it->geometrySignature)
                it = navEdgeFailures.erase(it);
            else
                ++it;
        }
        for (auto it = failedEdges.begin(); it != failedEdges.end(); )
        {
            if (it->second.expiresTick > 0 && observation.tick >= it->second.expiresTick)
            {
                localFailureSignatures.erase(it->first);
                it = failedEdges.erase(it);
            }
            else
                ++it;
        }
    }

    const Portal *findPortalForEdge(int edgeId) const
    {
        for (const Portal &portal : observation.portals)
            if (portal.wall * 65536 + portal.to == edgeId)
                return &portal;
        for (const auto &entry : knownGraph)
            for (const Portal &portal : entry.second)
                if (portal.wall * 65536 + portal.to == edgeId)
                    return &portal;
        return nullptr;
    }

    bool edgeFailed(int edgeId) const
    {
        auto failed = failedEdges.find(edgeId);
        if (failed == failedEdges.end())
            return false;
        if (failed->second.expiresTick > 0 && observation.tick >= failed->second.expiresTick)
            return false;
        const Portal *current = findPortalForEdge(edgeId);
        // If the edge is currently not observable, do not turn an old local
        // trajectory failure into a permanent topological fact.
        return current && failed->second.geometrySignature == portalGeometrySignature(*current);
    }

    bool localEdgeFailed(int edgeId) const
    {
        auto failure = failedEdges.find(edgeId);
        if (failure == failedEdges.end())
            return false;
        if (failure->second.expiresTick > 0 && observation.tick >= failure->second.expiresTick)
            return false;
        return localFailureSignatures.count(edgeId) != 0 || failure->second.attempts > 0;
    }

    void recordEdgeFailure(const Portal &portal, const char *eventName, const char *reason)
    {
        // Geometry in motion is not evidence about a boundary.  Failing to
        // get through a door mid-travel says only that the bot arrived at
        // the wrong moment, and writing that down as a property of the edge
        // is how a route the bot later walks came to be remembered as
        // impossible.
        if (portal.wallBusy || portal.sectorBusy || doorTiming(portal.to).moving)
        {
            char pending[192];
            snprintf(pending, sizeof(pending),
                     "wall=%d from=%d to=%d reason=%s evidence=geometry_in_motion",
                     portal.wall, portal.from, portal.to, reason);
            event("edge_failure_withheld", pending);
            return;
        }
        const int edgeId = portal.wall * 65536 + portal.to;
        EdgeFailure &failure = failedEdges[edgeId];
        const int signature = portalGeometrySignature(portal);
        if (failure.geometrySignature != signature)
        {
            failure.geometrySignature = signature;
            failure.attempts = 0;
        }
        ++failure.attempts;
        failure.expiresTick = observation.tick + failureCooldownTicks(failure.attempts);
        if (strcmp(eventName, "local_portal_failed") == 0)
            localFailureSignatures[edgeId] = signature;
        char detail[160];
        snprintf(detail, sizeof(detail), "wall=%d from=%d to=%d reason=%s attempts=%d signature=%d",
                 portal.wall, portal.from, portal.to, reason, failure.attempts, signature);
        event(eventName, detail);
    }

    const Portal *selectPortal()
    {
        const bool avoidBacktrack = repeatedBacktrackCount >= 2
            && lastTransitionTo == observation.sector;
        if (currentGoal == "EXPLORE_FRONTIER" && currentGoalTarget >= 0)
        {
            for (const Portal &portal : observation.portals)
            {
                const int edgeId = portal.wall * 65536 + portal.to;
                if (portal.wall == currentGoalTarget && !edgeFailed(edgeId)
                    && (portal.traversable || portal.jumpable)
                    && (!portal.key || hasKey(portal.key))
                    && (!avoidBacktrack || portal.to != lastTransitionFrom))
                    return &portal;
            }

            auto known = knownGraph.find(observation.sector);
            if (known != knownGraph.end())
            {
                for (const Portal &portal : known->second)
                {
                    const int edgeId = portal.wall * 65536 + portal.to;
                    if (portal.wall == currentGoalTarget && !edgeFailed(edgeId)
                        && (!portal.key || hasKey(portal.key))
                        && (!avoidBacktrack || portal.to != lastTransitionFrom))
                        return &portal;
                }
            }
        }

        // Pending work is gone.  Local open frontiers are derived from
        // current entered-sector knowledge whenever a new objective is needed.

        bool unvisitedAvailable = false;
        for (const Portal &portal : observation.portals)
        {
            if ((portal.traversable || portal.jumpable)
                && (!portal.key || hasKey(portal.key))
                && !visitedSectors.count(portal.to)
                && !edgeFailed(portal.wall * 65536 + portal.to)
                && !openedRoutes.count(portal.from * 65536 + portal.to)
                && (!avoidBacktrack || portal.to != lastTransitionFrom))
            {
                unvisitedAvailable = true;
                break;
            }
        }

        const Portal *best = nullptr;
        int bestDistance = INT32_MAX;
        for (const Portal &portal : observation.portals)
        {
            if (!(portal.traversable || portal.jumpable) || (portal.key && !hasKey(portal.key)))
                continue;
            if (visitedSectors.count(portal.to))
                continue;
            if (avoidBacktrack && portal.to == lastTransitionFrom)
                continue;
            const int edgeId = portal.wall * 65536 + portal.to;
            if (edgeFailed(edgeId))
                continue;
            const bool alreadyExplored = visitedSectors.count(portal.to)
                || openedRoutes.count(portal.from * 65536 + portal.to);
            if (!unvisitedAvailable || alreadyExplored)
                continue;
            const int score = distance2(observation.x, observation.y, portal.x, portal.y);
            if (score < bestDistance)
            {
                best = &portal;
                bestDistance = score;
            }
        }
        if (best)
            return best;

        return nullptr;
    }

    void rememberEnteredSectorBoundaries(int sectorId)
    {
        if (!inRange(sectorId, 0, numsectors))
            return;
        const sectortype &record = sector[sectorId];
        for (int i = 0; i < record.wallnum; ++i)
        {
            const int wallId = record.wallptr + i;
            if (!inRange(wallId, 0, numwalls))
                continue;
            const walltype &wallRecord = wall[wallId];
            if (!inRange(wallRecord.nextsector, 0, numsectors)
                || !inRange(wallRecord.point2, 0, numwalls))
                continue;
            Portal portal;
            portal.wall = wallId;
            portal.from = sectorId;
            portal.to = wallRecord.nextsector;
            portal.x1 = wallRecord.x;
            portal.y1 = wallRecord.y;
            portal.x2 = wall[wallRecord.point2].x;
            portal.y2 = wall[wallRecord.point2].y;
            portal.x = (portal.x1 + portal.x2) / 2;
            portal.y = (portal.y1 + portal.y2) / 2;
            portal.localGeometry = true;
            portal.visible = false;
            setInteractionGeometry(portal);
            portal.floorZ = getflorzofslope(sectorId, portal.x, portal.y);
            portal.ceilingZ = getceilzofslope(sectorId, portal.x, portal.y);
            portal.fromFloorZ = portal.floorZ;
            portal.fromCeilingZ = portal.ceilingZ;
            portal.toFloorZ = getflorzofslope(portal.to, portal.x, portal.y);
            portal.toCeilingZ = getceilzofslope(portal.to, portal.x, portal.y);
            portal.floorDelta = portal.toFloorZ - portal.fromFloorZ;
            portal.clearance = std::min(portal.fromFloorZ - portal.fromCeilingZ,
                                        portal.toFloorZ - portal.toCeilingZ);
            portal.openingWidth = int(std::sqrt(double(distance2(portal.x1, portal.y1,
                                                                 portal.x2, portal.y2))));
            portal.walkable = !(wallRecord.cstat & 1)
                && portal.openingWidth >= playerPassageWidth()
                && portal.clearance >= playerBodyClearance()
                && std::abs(portal.floorDelta) <= kMaxWalkableStep;
            portal.jumpable = !(wallRecord.cstat & 1)
                && portal.openingWidth >= playerPassageWidth()
                && portal.clearance >= playerBodyClearance()
                && portal.floorDelta < 0 && -portal.floorDelta <= playerJumpRiseLimit();
            portal.traversable = portal.walkable;
            auto &edges = knownGraph[sectorId];
            auto existing = std::find_if(edges.begin(), edges.end(), [&portal](const Portal &known)
            {
                return known.wall == portal.wall && known.to == portal.to;
            });
            if (existing == edges.end())
            {
                edges.push_back(portal);
                if (!visitedSectors.count(portal.to))
                {
                    char detail[96];
                    snprintf(detail, sizeof(detail), "wall=%d from=%d to=%d source=entered_sector_geometry",
                             portal.wall, portal.from, portal.to);
                    event("discovered_frontier", detail);
                }
            }
        }
    }

    void refreshDerivedFrontiers()
    {
        pruneNavEdgeFailures();
        std::vector<int> visited(visitedSectors.begin(), visitedSectors.end());
        std::vector<Boundary> boundaries;
        for (const auto &entry : knownGraph)
        {
            for (const Portal &portal : entry.second)
            {
                Boundary boundary;
                boundary.wall = portal.wall;
                boundary.from = portal.from;
                boundary.to = portal.to;
                boundary.traversable = portal.traversable && (!portal.key || hasKey(portal.key))
                    && !edgeFailed(portal.wall * 65536 + portal.to)
                    && !localEdgeFailed(portal.wall * 65536 + portal.to);
                boundary.jumpable = portal.jumpable && !edgeFailed(portal.wall * 65536 + portal.to);
                boundary.geometrySignature = portalIdentitySignature(portal);
                boundaries.push_back(boundary);
            }
        }
        derivedFrontiers = llmapper::deriveFrontiers(visited, boundaries, investigatedBoundaries,
                                                     navEdgeFailures);
    }

    const Portal *portalByWallId(int wallId) const
    {
        if (wallId < 0)
            return nullptr;
        for (const auto &entry : knownGraph)
        {
            for (const Portal &portal : entry.second)
                if (portal.wall == wallId)
                    return &portal;
        }
        for (const Portal &portal : observation.portals)
            if (portal.wall == wallId)
                return &portal;
        return nullptr;
    }

    const Portal *portalByWall(int wall, int from, int to) const
    {
        // Live observation first.  The cached graph is remembered structure;
        // its traversability is only as fresh as the last time that boundary
        // was looked at.  Consulting it first meant a door that had just
        // opened still read as shut, so the crossing fell through to ordinary
        // navigation and the bot spent two seconds not going through a door
        // it was standing in front of -- long enough to be caught by one
        // that closes.
        for (const Portal &portal : observation.portals)
            if (portal.wall == wall && portal.to == to)
                return &portal;
        auto known = knownGraph.find(from);
        if (known != knownGraph.end())
        {
            for (const Portal &portal : known->second)
                if (portal.wall == wall && portal.to == to)
                    return &portal;
        }
        return portalByWallId(wall);
    }

    // Mark every standable cell the player can currently see.  Sight, not
    // presence, is what reveals an affordance, so cansee() is the test.
    void updateLocalCoverage()
    {
        ensureNavTopology();
        const int reach = kCoverageSightRange;
        for (const NavCell &cell : navCells)
        {
            const int64_t handle = navCellHandle(cell.sector, cell.gx, cell.gy);
            if (observedCells.count(handle))
                continue;
            if (distance2(observation.x, observation.y, cell.center.x, cell.center.y)
                > reach * reach)
                continue;
            const int floorZ = inRange(cell.sector, 0, numsectors)
                ? getflorzofslope(cell.sector, cell.center.x, cell.center.y) : observation.z;
            if (!cansee(observation.x, observation.y, observation.z, observation.sector,
                        cell.center.x, cell.center.y, floorZ - kCoverageEyeOffset, cell.sector))
                continue;
            observedCells.insert(handle);
        }
    }

    // Nearest reachable cell of this sector that has never been observed.
    bool findUnobservedCell(int sectorId, int &outX, int &outY, int &outCount) const
    {
        int best = -1;
        int bestDistance = INT32_MAX;
        outCount = 0;
        const int playerCell = nearestNavCell(observation.sector, observation.x, observation.y);
        const int playerArea = playerCell >= 0 ? navCells[size_t(playerCell)].walkArea : -1;
        for (const NavCell &cell : navCells)
        {
            if (cell.sector != sectorId)
                continue;
            if (observedCells.count(navCellHandle(cell.sector, cell.gx, cell.gy)))
                continue;
            // Unreachable pockets are not unexposed space; they are scenery.
            if (playerArea >= 0 && cell.walkArea != playerArea)
                continue;
            ++outCount;
            const int distance = distance2(observation.x, observation.y,
                                           cell.center.x, cell.center.y);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = cell.id;
            }
        }
        if (best < 0)
            return false;
        outX = navCells[size_t(best)].center.x;
        outY = navCells[size_t(best)].center.y;
        return true;
    }

    // The first sector below the spawn node on this branch.  Purely for
    // telemetry: it makes "which branch is the bot on" legible in a run log.
    int branchRoot(int sectorId) const
    {
        int cursor = sectorId;
        for (int guard = 0; guard < 256; ++guard)
        {
            auto node = explorationTree.find(cursor);
            if (node == explorationTree.end() || node->second.depth <= 1)
                return cursor;
            cursor = node->second.parent;
        }
        return cursor;
    }

    int sectorDepth(int sectorId) const
    {
        auto node = explorationTree.find(sectorId);
        return node == explorationTree.end() ? 0 : node->second.depth;
    }

    // Route distance in sectors over the graph the bot has actually walked.
    // Visited space is transportation, so nothing here penalises revisiting.
    void computeHops(std::vector<int> &hops) const
    {
        hops.assign(size_t(std::max(int(numsectors), 1)), -1);
        routeGateWall.assign(hops.size(), -1);
        routeGateSector.assign(hops.size(), -1);
        if (!inRange(observation.sector, 0, numsectors))
            return;
        std::deque<int> queue;
        hops[size_t(observation.sector)] = 0;
        queue.push_back(observation.sector);
        while (!queue.empty())
        {
            const int current = queue.front();
            queue.pop_front();
            auto graph = knownGraph.find(current);
            if (graph == knownGraph.end())
                continue;
            for (const Portal &edge : graph->second)
            {
                if (!inRange(edge.to, 0, int(hops.size())) || hops[size_t(edge.to)] >= 0)
                    continue;
                if (!visitedSectors.count(edge.to))
                    continue;
                if (edge.key && !hasKey(edge.key))
                    continue;
                if (hazardousSector(edge.to))
                    continue;
                int cost = 1;
                if (!(edge.traversable || edge.jumpable))
                {
                    // Blocked right now.  If the bot knows a mechanism that
                    // reopens it, this is still a way through -- it just
                    // costs an interaction.
                    if (!edge.interactionAffordance && !reopenableConnections.count(edge.wall)
                        && !knownMechanismFor(edge))
                        continue;
                    cost = kGatedHopCost;
                }
                hops[size_t(edge.to)] = hops[size_t(current)] + cost;
                // Inherit the first shut connection on this route, or become
                // it.  This is what turns "unreachable" into "reopen that".
                if (cost == 1)
                {
                    routeGateWall[size_t(edge.to)] = routeGateWall[size_t(current)];
                    routeGateSector[size_t(edge.to)] = routeGateSector[size_t(current)];
                }
                else if (routeGateWall[size_t(current)] >= 0)
                {
                    routeGateWall[size_t(edge.to)] = routeGateWall[size_t(current)];
                    routeGateSector[size_t(edge.to)] = routeGateSector[size_t(current)];
                }
                else
                {
                    routeGateWall[size_t(edge.to)] = edge.wall;
                    routeGateSector[size_t(edge.to)] = edge.from;
                }
                queue.push_back(edge.to);
            }
        }
    }

    // Everything the bot has learned that could change a past verdict.  Any
    // movement in this number is grounds for reconsidering suppressed work.
    int worldEvidenceRevision() const
    {
        return knowledgeRevision * 8191 + inventoryRevision * 131
            + navTopologyRevision;
    }

    int opportunityDormantUntil(int key) const
    {
        auto entry = suppressedUntil.find(key);
        return entry == suppressedUntil.end() ? 0 : entry->second;
    }

    // Rebuild the ledger of unresolved work from current knowledge.  This is
    // a projection, not a second source of truth: dormancy and the tree are
    // the only persistent state, everything else is re-derived.
    void rebuildLedger()
    {
        ledger.clear();
        std::vector<int> hops;
        computeHops(hops);
        pruneNavEdgeFailures();

        for (const auto &entry : knownGraph)
        {
            const int from = entry.first;
            if (!visitedSectors.count(from) || !inRange(from, 0, int(hops.size())))
                continue;
            const int reach = hops[size_t(from)];
            if (reach < 0)
                continue;
            for (const Portal &portal : entry.second)
            {
                if (portal.to < 0 || visitedSectors.count(portal.to))
                    continue;
                // A sector with no standable square is thin -- a step, a
                // ledge, a door track.  The player passes through such a
                // place rather than standing in it, so it is poor as a
                // destination but perfectly good as transit, and refusing it
                // outright severed real routes.
                if (!sectorHasStandableSpace(portal.to))
                {
                    noteTooNarrow(portal);
                    continue;
                }
                const int edgeId = portal.wall * 65536 + portal.to;
                llmapper::Opportunity opportunity;
                opportunity.sector = from;
                opportunity.target = portal.to;
                opportunity.wall = portal.wall;
                opportunity.requiredKey = portal.key;
                opportunity.depth = sectorDepth(from);
                opportunity.hops = reach;
                opportunity.local = from == observation.sector;
                const bool crossable = (portal.traversable || portal.jumpable)
                    && !edgeFailed(edgeId) && !localEdgeFailed(edgeId);
                if (portal.key && !hasKey(portal.key))
                {
                    opportunity.kind = llmapper::kOpportunityLocked;
                    opportunity.id = 3000000 + portal.wall * 8 + int(kObjectiveFrontier);
                    lockedDoorKey[portal.wall] = portal.key;
                }
                else if (crossable)
                {
                    opportunity.kind = llmapper::kOpportunityFrontier;
                    opportunity.id = 3000000 + portal.wall * 8 + int(kObjectiveFrontier);
                }
                else if (knownMechanismFor(portal))
                {
                    // The mechanism that could open this is already in the
                    // ledger as its own interaction opportunity.  Adding the
                    // boundary too would make the bot alternate between
                    // walking up to look at the door and actually using it.
                    continue;
                }
                else if (portal.interactionAffordance)
                {
                    if (llmapper::investigatedNow(investigatedBoundaries, portal.wall,
                                                  portal.from, portal.to,
                                                  portalIdentitySignature(portal)))
                        continue;
                    // Keyed by where it leads, not by which of its walls the
                    // bot happens to be looking at.
                    opportunity.kind = llmapper::kOpportunityBlocked;
                    opportunity.id = 4000000 + portal.to;
                    bool alreadyOffered = false;
                    for (const llmapper::Opportunity &known : ledger)
                        if (known.id == opportunity.id)
                            alreadyOffered = true;
                    if (alreadyOffered)
                        continue;
                }
                else
                {
                    // Structurally known, currently solid, and nothing the
                    // bot knows how to act on from this side.  Remember the
                    // boundary and leave it alone; evidence reaching it from
                    // the other side can make it actionable later.
                    noteInertBoundary(portal);
                    continue;
                }
                opportunity.dormantUntil = opportunityDormantUntil(opportunity.id);
                ledger.push_back(opportunity);
            }
        }

        for (const auto &entry : interactions)
        {
            const InteractionMemory &memory = entry.second;
            if (!memory.observed)
                continue;
            if (memory.attempted && !interactionNeedsReactivation(memory))
                continue;
            if (!inRange(memory.fromSector, 0, int(hops.size())))
                continue;
            const int reach = hops[size_t(memory.fromSector)];
            if (reach < 0)
                continue;
            llmapper::Opportunity opportunity;
            opportunity.kind = memory.key && !hasKey(memory.key)
                ? llmapper::kOpportunityLocked : llmapper::kOpportunityInteraction;
            opportunity.id = 1000000 + interactionMemoryKey(memory);
            opportunity.sector = memory.fromSector;
            opportunity.target = memory.targetSector;
            opportunity.wall = memory.target.wall;
            opportunity.requiredKey = memory.key;
            opportunity.depth = sectorDepth(memory.fromSector);
            opportunity.hops = reach;
            opportunity.local = memory.fromSector == observation.sector;
            opportunity.dormantUntil = opportunityDormantUntil(opportunity.id);
            ledger.push_back(opportunity);
        }

        for (int sectorId : visitedSectors)
        {
            if (!inRange(sectorId, 0, int(hops.size())))
                continue;
            const int reach = hops[size_t(sectorId)];
            if (reach < 0)
                continue;
            // A doorway is a threshold, not a room worth surveying.
            if (movingSectorHazard(sectorId))
                continue;
            int cellX = 0;
            int cellY = 0;
            int unobserved = 0;
            if (!findUnobservedCell(sectorId, cellX, cellY, unobserved))
                continue;
            llmapper::Opportunity opportunity;
            opportunity.kind = llmapper::kOpportunityCoverage;
            opportunity.id = 7000000 + sectorId;
            opportunity.sector = sectorId;
            opportunity.target = sectorId;
            opportunity.depth = sectorDepth(sectorId);
            opportunity.hops = reach;
            opportunity.local = sectorId == observation.sector;
            opportunity.dormantUntil = opportunityDormantUntil(opportunity.id);
            ledger.push_back(opportunity);
        }

        for (const auto &entry : objectMemory)
        {
            const ObjectMemory &memory = entry.second;
            if (!memory.observed || memory.collected || !objectStillPresent(memory))
                continue;
            if (memory.object.kind != kObjectPickup && memory.object.kind != kObjectKey)
                continue;
            if (!inRange(memory.object.sector, 0, int(hops.size())))
                continue;
            const int reach = hops[size_t(memory.object.sector)];
            if (reach < 0)
                continue;
            llmapper::Opportunity opportunity;
            opportunity.kind = llmapper::kOpportunityPickup;
            opportunity.id = 5000000 + memory.object.sprite;
            opportunity.sector = memory.object.sector;
            opportunity.depth = sectorDepth(memory.object.sector);
            opportunity.hops = reach;
            // A pickup counts as on-the-way only from the room the bot is
            // standing in.  Anything further is not worth abandoning a
            // branch for, and stays in the ledger for a later pass.
            opportunity.local = reach == 0;
            opportunity.dormantUntil = opportunityDormantUntil(opportunity.id);
            ledger.push_back(opportunity);
        }
    }

    // Is there a concrete, still-usable mechanism the bot has registered
    // that could change this boundary?  Affordance comes from Blood's own
    // interaction records; it says "a player can push this", never "this
    // opens the level".
    const InteractionMemory *knownMechanismFor(const Portal &portal) const
    {
        for (const auto &entry : interactions)
        {
            const InteractionMemory &memory = entry.second;
            if (!memory.observed)
                continue;
            if (memory.attempted && !interactionNeedsReactivation(memory))
                continue;
            if (memory.target.wall >= 0 && memory.target.wall == portal.wall)
                return &memory;
            // A mechanism whose sector is the space beyond this boundary is
            // the thing that opens it, even when the wall itself is inert.
            if (memory.targetSector >= 0 && memory.targetSector == portal.to)
                return &memory;
        }
        return nullptr;
    }

    void noteTooNarrow(const Portal &portal)
    {
        if (!narrowSectors.insert(portal.to).second)
            return;
        char detail[176];
        snprintf(detail, sizeof(detail),
                 "sector=%d via_wall=%d width=%d reason=no_standable_space_for_player",
                 portal.to, portal.wall, portal.openingWidth);
        event("boundary_too_narrow", detail);
    }

    void noteInertBoundary(const Portal &portal)
    {
        const int signature = portalIdentitySignature(portal);
        auto known = inertBoundaries.find(portal.wall);
        if (known != inertBoundaries.end() && known->second == signature)
            return;
        inertBoundaries[portal.wall] = signature;
        char detail[176];
        snprintf(detail, sizeof(detail),
                 "wall=%d from=%d to=%d clearance=%d floor_delta=%d reason=no_known_affordance_from_this_side",
                 portal.wall, portal.from, portal.to, portal.clearance, portal.floorDelta);
        event("boundary_inert", detail);
    }

    bool hazardousSector(int sectorId) const
    {
        auto known = hazardSectors.find(sectorId);
        return known != hazardSectors.end() && observation.tick < known->second;
    }

    // Somewhere adjacent that is not currently hurting the bot.
    int escapeHazard() const
    {
        int fallback = -1;
        for (const Portal &portal : observation.portals)
        {
            if (portal.from != observation.sector || portal.to == observation.sector)
                continue;
            if (!(portal.traversable || portal.jumpable))
                continue;
            if (hazardousSector(portal.to))
                continue;
            if (doorTiming(portal.to).closing)
                continue;
            if (portal.to != lastTransitionFrom)
                return portal.wall;
            fallback = portal.wall;
        }
        return fallback;
    }

    // Watch for damage that is not coming from anything the bot could fight.
    void trackEnvironmentalDamage()
    {
        const int lost = healthLostSinceObservation;
        if (lost <= 0)
        {
            if (damageBurstStartTick >= 0
                && observation.tick - damageBurstStartTick > kDamageBurstWindow)
            {
                damageBurstStartTick = -1;
                damageBurstCount = 0;
            }
            return;
        }
        if (damageBurstStartTick < 0
            || observation.tick - damageBurstStartTick > kDamageBurstWindow)
        {
            damageBurstStartTick = observation.tick;
            damageBurstCount = 0;
            // Health before this observation's loss, so a single heavy hit
            // registers at once instead of reading as zero damage so far.
            damageBurstHealth = observation.health + lost;
        }
        ++damageBurstCount;
        const int bled = damageBurstHealth - observation.health;
        if (damageBurstCount < 3 && bled < kDamageBurstSevere)
            return;
        if (selectObject(kObjectEnemy))
            return;   // something is shooting: that is combat, not terrain
        if (!inRange(observation.sector, 0, numsectors))
            return;
        if (!hazardousSector(observation.sector))
        {
            char detail[192];
            snprintf(detail, sizeof(detail),
                     "sector=%d hits=%d lost=%d health=%d",
                     observation.sector, damageBurstCount, bled, observation.health);
            event("hazard_sector_detected", detail);
        }
        hazardSectors[observation.sector] = observation.tick + kHazardAvoidTicks;
    }

    // Does this wall actually separate these two sectors?  A route's final
    // destination wall is not proof of which boundary the player physically
    // crossed on the way there, and recording the wrong one writes a
    // connection into the world graph that does not exist -- which then
    // excludes the real frontier from future exploration.
    static bool wallJoinsSectors(int wallId, int a, int b)
    {
        if (!inRange(wallId, 0, numwalls) || a < 0 || b < 0)
            return false;
        const int owner = sectorofwall(int16_t(wallId));
        const int other = wall[wallId].nextsector;
        return (owner == a && other == b) || (owner == b && other == a);
    }

    // The boundary between two sectors nearest the player -- the one he most
    // plausibly just stepped over.
    int crossingWallBetween(int from, int to) const
    {
        if (!inRange(from, 0, numsectors) || !inRange(to, 0, numsectors))
            return -1;
        const sectortype &record = sector[from];
        int best = -1;
        int64_t bestDistance = INT64_MAX;
        for (int i = 0; i < record.wallnum; ++i)
        {
            const int wallId = record.wallptr + i;
            if (!inRange(wallId, 0, numwalls) || wall[wallId].nextsector != to)
                continue;
            if (!inRange(wall[wallId].point2, 0, numwalls))
                continue;
            const int64_t distance = dist2ToSegment(
                observation.x, observation.y, wall[wallId].x, wall[wallId].y,
                wall[wall[wallId].point2].x, wall[wall[wallId].point2].y);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = wallId;
            }
        }
        return best;
    }

    // Which boundary of the sector the bot is standing in leads out of it.
    // Prefer continuing away from where the bot came from, so leaving a
    // doorway still counts as making progress rather than retreating.
    int escapeMovingSector() const
    {
        int fallback = -1;
        for (const Portal &portal : observation.portals)
        {
            if (portal.from != observation.sector || portal.to == observation.sector)
                continue;
            if (!(portal.traversable || portal.jumpable))
                continue;
            if (movingSectorHazard(portal.to))
                continue;
            if (portal.to != lastTransitionFrom)
                return portal.wall;
            fallback = portal.wall;
        }
        return fallback;
    }

    // The far side of a door sector, entered from `from`.
    int sectorBeyondDoor(int doorSector, int from) const
    {
        if (!inRange(doorSector, 0, numsectors))
            return -1;
        const sectortype &record = sector[doorSector];
        for (int i = 0; i < record.wallnum; ++i)
        {
            const int wallId = record.wallptr + i;
            if (!inRange(wallId, 0, numwalls))
                continue;
            const int next = wall[wallId].nextsector;
            if (inRange(next, 0, numsectors) && next != from && next != doorSector)
                return next;
        }
        return -1;
    }

    struct DoorTiming
    {
        bool moving = false;      // geometry is travelling right now
        bool closing = false;     // and travelling toward the shut state
        bool autoCloses = false;  // reverts on its own once it settles
        int remainingTicks = 0;   // until the current travel finishes
        int holdTicks = 0;        // how long it stays open once open
        int openClearance = 0;    // floor-to-ceiling gap when fully open
    };

    // Sign of the sector's live busy delta: positive travels toward the ON
    // state, negative toward OFF.  Falls back to the caller's guess when the
    // engine has no entry for this sector.
    static bool busyDeltaHeadsOn(int sectorId, bool fallback)
    {
        for (int i = 0; i < gBusyCount; ++i)
            if (gBusy[i].at0 == sectorId)
                return gBusy[i].at4 > 0;
        return fallback;
    }

    // Blood drives a sector mechanism by stepping `busy` between 0 and
    // 65536 over 12*busyTime ticks, then posts the reverse command after
    // 12*waitTime ticks.  Read that rather than inferring it.
    DoorTiming doorTiming(int sectorId) const
    {
        DoorTiming timing;
        if (!inRange(sectorId, 0, numsectors))
            return timing;
        const int extra = sector[sectorId].extra;
        if (extra <= 0 || extra >= kMaxXSectors)
            return timing;
        const XSECTOR &record = xsector[extra];
        const int busy = int(record.busy) & 0xffff;
        // Opening is the OFF->ON direction; which one is "open" is decided
        // by the geometry the two states describe, not by the state number.
        const int gapOn = record.onFloorZ - record.onCeilZ;
        const int gapOff = record.offFloorZ - record.offCeilZ;
        timing.openClearance = std::max(gapOn, gapOff);
        const bool openingIsOn = gapOn >= gapOff;
        const int travelToOpen = kTicsPerSec * int(openingIsOn ? record.busyTimeA
                                                               : record.busyTimeB) / 10;
        const int travelToShut = kTicsPerSec * int(openingIsOn ? record.busyTimeB
                                                               : record.busyTimeA) / 10;
        timing.holdTicks = kTicsPerSec * int(openingIsOn ? record.waitTimeA
                                                         : record.waitTimeB) / 10;
        timing.autoCloses = timing.holdTicks > 0;
        timing.moving = busy != 0;
        if (!timing.moving)
            return timing;
        // Which way it is travelling comes from the engine's own busy entry.
        // `state` is NOT that answer: SetSectorState only runs when the
        // travel finishes, so throughout the motion `state` still describes
        // where the sector came from.  Reading it as the heading inverted
        // every verdict -- an opening door looked like a closing one, so the
        // bot stood and waited out doors it could already have walked
        // through, and a genuinely closing door looked safe to enter.
        const bool headingOn = busyDeltaHeadsOn(sectorId, record.state == 0);
        // Walking into a gap that is still widening is fine; walking into
        // one that is narrowing is how the bot gets crushed.
        timing.closing = headingOn != openingIsOn;
        const int travel = (headingOn == openingIsOn) ? travelToOpen : travelToShut;
        const int fraction = headingOn ? (65536 - busy) : busy;
        timing.remainingTicks = travel > 0 ? int((int64_t(travel) * fraction) / 65536) : 0;
        return timing;
    }

    // Is this sector a piece of machinery the player should not loiter in?
    bool movingSectorHazard(int sectorId) const
    {
        const DoorTiming timing = doorTiming(sectorId);
        return timing.moving || timing.autoCloses;
    }

    // Watch known boundaries flip between blocked and traversable.  This is
    // the observable signature of Blood's moving geometry and the evidence
    // that an interaction actually did something.
    void trackBoundaryStates()
    {
        for (const Portal &portal : observation.portals)
        {
            const bool open = portal.traversable || portal.jumpable;
            auto known = boundaryOpen.find(portal.wall);
            if (known == boundaryOpen.end())
            {
                boundaryOpen[portal.wall] = open;
                continue;
            }
            // Watching the envelope move is how the bot learns what an
            // interaction actually did.  Report meaningful changes even when
            // they do not yet cross the traversable threshold.
            auto lastClearance = boundaryClearance.find(portal.wall);
            if (lastClearance == boundaryClearance.end()
                || std::abs(lastClearance->second - portal.clearance) >= 256)
            {
                boundaryClearance[portal.wall] = portal.clearance;
                char progress[224];
                snprintf(progress, sizeof(progress),
                         "wall=%d from=%d to=%d clearance=%d body=%d crouch=%d floor_delta=%d busy=%d",
                         portal.wall, portal.from, portal.to, portal.clearance,
                         playerBodyClearance(), playerCrouchClearance(), portal.floorDelta,
                         portal.wallBusy || portal.sectorBusy ? 1 : 0);
                event("boundary_clearance_changed", progress);
            }
            if (known->second == open)
                continue;
            known->second = open;
            char detail[224];
            snprintf(detail, sizeof(detail),
                     "wall=%d from=%d to=%d transition=%s clearance=%d floor_delta=%d busy=%d",
                     portal.wall, portal.from, portal.to,
                     open ? "blocked_to_traversable" : "traversable_to_blocked",
                     portal.clearance, portal.floorDelta,
                     portal.wallBusy || portal.sectorBusy ? 1 : 0);
            event(open ? "boundary_opened" : "boundary_closed", detail);
            if (open)
            {
                // Newly opened space is the point of having pressed the
                // button.  Offer it as the next mission before anything else
                // reclaims the bot's attention, because Blood doors close.
                openedRouteWall = portal.wall;
                openedRouteFrom = portal.from;
                openedRouteTo = portal.to;
                openedRouteTick = observation.tick;
                // Reaching it may need a route the old failure memory blocks.
                failedEdges.erase(portal.wall * 65536 + portal.to);
                suppressedUntil.erase(3000000 + portal.wall * 8 + int(kObjectiveFrontier));
            }
            else
            {
                // A connection that closed is still a connection.  Remember
                // that it is reopenable so a later route can plan through it
                // instead of concluding the way home vanished.
                if (reopenableConnections.insert(portal.wall).second)
                    event("connection_reopenable", detail);
            }
        }
    }

    // Full, auditable statement of what unresolved work the bot believes
    // exists.  Emitted at most once per few seconds when the bot thinks it
    // is out of options, so a stall can always be explained.
    void dumpLedger(const char *reason)
    {
        if (lastLedgerDumpTick >= 0
            && observation.tick - lastLedgerDumpTick < 5 * kTicsPerSec)
            return;
        lastLedgerDumpTick = observation.tick;
        rebuildLedger();
        int pending = 0;
        int dormant = 0;
        int unreachable = 0;
        int lockedNoKey = 0;
        for (const llmapper::Opportunity &opportunity : ledger)
        {
            if (opportunity.hops < 0)
                ++unreachable;
            else if (opportunity.requiredKey > 0
                     && !(heldKeyMask & (1u << unsigned(opportunity.requiredKey & 31))))
                ++lockedNoKey;
            else if (opportunity.dormantUntil > observation.tick)
                ++dormant;
            else
                ++pending;
        }
        char detail[256];
        snprintf(detail, sizeof(detail),
                 "reason=%s sector=%d ledger=%u pending=%d dormant=%d unreachable=%d locked_no_key=%d visited=%u inert=%u nav_failures=%u",
                 reason, observation.sector, unsigned(ledger.size()), pending, dormant,
                 unreachable, lockedNoKey, unsigned(visitedSectors.size()),
                 unsigned(inertBoundaries.size()), unsigned(navEdgeFailures.size()));
        event("ledger_snapshot", detail);
        int reported = 0;
        for (const llmapper::Opportunity &opportunity : ledger)
        {
            if (reported++ >= 12)
                break;
            char entry[224];
            snprintf(entry, sizeof(entry),
                     "id=%d kind=%d sector=%d target=%d wall=%d key=%d depth=%d hops=%d dormant_s=%d",
                     opportunity.id, int(opportunity.kind), opportunity.sector,
                     opportunity.target, opportunity.wall, opportunity.requiredKey,
                     opportunity.depth, opportunity.hops,
                     opportunity.dormantUntil > observation.tick
                         ? (opportunity.dormantUntil - observation.tick) / kTicsPerSec : 0);
            event("ledger_entry", entry);
        }
    }

    const llmapper::Opportunity *ledgerEntry(int id) const
    {
        for (const llmapper::Opportunity &opportunity : ledger)
            if (opportunity.id == id)
                return &opportunity;
        return nullptr;
    }

    // Nothing is actionable right now, but the bot still knows unresolved
    // work.  Re-arm the entry that has been dormant longest and go at it
    // again rather than declaring the level finished.
    llmapper::Mission wakeSoonestDormant()
    {
        llmapper::Mission woken;
        const llmapper::Opportunity *best = nullptr;
        for (const llmapper::Opportunity &opportunity : ledger)
        {
            if (opportunity.hops < 0)
                continue;
            if (opportunity.requiredKey > 0
                && !(heldKeyMask & (1u << unsigned(opportunity.requiredKey & 31))))
                continue;
            if (opportunity.id == lastFailedOpportunity
                && observation.tick - lastFailedOpportunityTick < kTicsPerSec * 8)
                continue;
            // Dormant means dormant.  It ends when its cooldown ends, or when
            // something the bot has since learned invalidates the reason it
            // was suppressed -- a key collected, a mechanism moved, geometry
            // changed.  Running out of other ideas is not new evidence.
            if (opportunity.dormantUntil > observation.tick)
            {
                auto known = suppressionEvidence.find(opportunity.id);
                if (known != suppressionEvidence.end()
                    && known->second == worldEvidenceRevision())
                    continue;
            }
            if (!best || opportunity.dormantUntil < best->dormantUntil)
                best = &opportunity;
        }
        if (!best)
            return woken;
        suppressedUntil.erase(best->id);
        woken.opportunity = best->id;
        woken.kind = best->kind == llmapper::kOpportunityPickup
            ? llmapper::kMissionCollect
            : best->kind == llmapper::kOpportunityFrontier
            ? (best->local ? llmapper::kMissionContinue : llmapper::kMissionReturnToBranch)
            : llmapper::kMissionSolveBlocker;
        woken.reason = "RETRY_DORMANT_OPPORTUNITY";
        char detail[192];
        snprintf(detail, sizeof(detail),
                 "id=%d kind=%d sector=%d target=%d wall=%d hops=%d slept_s=%d",
                 best->id, int(best->kind), best->sector, best->target, best->wall,
                 best->hops,
                 best->dormantUntil > observation.tick
                     ? (best->dormantUntil - observation.tick) / kTicsPerSec : 0);
        event("dormant_rearmed_early", detail);
        return woken;
    }

    // Turn the chosen mission into the concrete objective the execution
    // layer already knows how to run.
    // If the route to this work first passes through a connection that is
    // currently shut, the real next step is opening that connection.
    const llmapper::Opportunity *gateBefore(const llmapper::Opportunity &target) const
    {
        if (!inRange(target.sector, 0, int(routeGateWall.size())))
            return nullptr;
        const int gate = routeGateWall[size_t(target.sector)];
        if (gate < 0 || gate == target.wall)
            return nullptr;
        for (const llmapper::Opportunity &candidate : ledger)
        {
            if (candidate.wall != gate)
                continue;
            if (candidate.kind != llmapper::kOpportunityInteraction
                && candidate.kind != llmapper::kOpportunityBlocked)
                continue;
            return &candidate;
        }
        // The gate is known but not currently offered as work: fall back to
        // the mechanism memory that covers it.
        return nullptr;
    }

    bool commitMission(const llmapper::Mission &chosen)
    {
        const llmapper::Opportunity *opportunity = ledgerEntry(chosen.opportunity);
        if (!opportunity)
            return false;
        if (const llmapper::Opportunity *gate = gateBefore(*opportunity))
        {
            char detail[224];
            snprintf(detail, sizeof(detail),
                     "for_opportunity=%d for_sector=%d gate_wall=%d gate_sector=%d hops=%d",
                     opportunity->id, opportunity->sector, gate->wall, gate->sector,
                     opportunity->hops);
            event("reopen_route_to_objective", detail);
            opportunity = gate;
        }
        Objective objective;
        switch (opportunity->kind)
        {
        case llmapper::kOpportunityFrontier:
        case llmapper::kOpportunityLocked:
        case llmapper::kOpportunityBlocked:
            if (opportunity->wall < 0)
                return false;
            objective.type = opportunity->kind == llmapper::kOpportunityBlocked
                ? kObjectiveInvestigate : kObjectiveFrontier;
            objective.id = opportunity->wall;
            objective.wall = opportunity->wall;
            objective.sector = opportunity->sector;
            objective.targetSector = opportunity->target;
            exploreDestination = opportunity->target;
            exploreCrossingWall = opportunity->wall;
            exploreCrossingFrom = opportunity->sector;
            break;
        case llmapper::kOpportunityInteraction:
        {
            auto memory = interactions.find(opportunity->id - 1000000);
            if (memory == interactions.end())
                return false;
            objective.type = kObjectiveInteraction;
            objective.id = memory->second.id;
            objective.sector = memory->second.fromSector;
            objective.targetSector = memory->second.targetSector;
            objective.x = memory->second.x;
            objective.y = memory->second.y;
            objective.z = memory->second.z;
            objective.interactionKey = interactionMemoryKey(memory->second);
            break;
        }
        case llmapper::kOpportunityCoverage:
        {
            int cellX = 0;
            int cellY = 0;
            int unobserved = 0;
            if (!findUnobservedCell(opportunity->sector, cellX, cellY, unobserved))
                return false;
            objective.type = kObjectiveExpose;
            objective.id = opportunity->sector;
            objective.sector = opportunity->sector;
            objective.targetSector = opportunity->sector;
            objective.x = cellX;
            objective.y = cellY;
            objective.z = observation.z;
            coverageTargetX = cellX;
            coverageTargetY = cellY;
            coverageTargetSector = opportunity->sector;
            char expose[192];
            snprintf(expose, sizeof(expose),
                     "sector=%d unobserved_cells=%d target=(%d,%d) hops=%d",
                     opportunity->sector, unobserved, cellX, cellY, opportunity->hops);
            event("exploration_viewpoint_selected", expose);
            break;
        }
        case llmapper::kOpportunityPickup:
        {
            auto memory = objectMemory.find(opportunity->id - 5000000);
            if (memory == objectMemory.end())
                return false;
            objective.type = memory->second.object.kind == kObjectKey
                ? kObjectiveKey : kObjectivePickup;
            objective.id = memory->second.object.sprite;
            objective.sector = memory->second.object.sector;
            objective.targetSector = memory->second.object.sector;
            objective.x = memory->second.object.x;
            objective.y = memory->second.object.y;
            objective.z = memory->second.object.z;
            break;
        }
        default:
            return false;
        }

        const bool backtrack = chosen.kind == llmapper::kMissionReturnToBranch
            || chosen.kind == llmapper::kMissionReturnForKey;
        selectObjective(objective, opportunity->id == chosen.opportunity
                                       ? chosen.reason : "REOPEN_ROUTE_TO_OBJECTIVE");
        missionOpportunity = opportunity->id;
        missionStartedTick = observation.tick;
        mission = chosen;
        char detail[256];
        snprintf(detail, sizeof(detail),
                 "reason=%s kind=%d opportunity=%d sector=%d target=%d wall=%d depth=%d hops=%d key=%d from_sector=%d",
                 chosen.reason, int(opportunity->kind), opportunity->id,
                 opportunity->sector, opportunity->target, opportunity->wall,
                 opportunity->depth, opportunity->hops, opportunity->requiredKey,
                 observation.sector);
        event("mission_selected", detail);
        if (backtrack)
            event("backtrack_started", detail);
        return true;
    }

    bool selectExploreObjective(bool blocked)
    {
        refreshDerivedFrontiers();
        std::vector<int> hopStorage(std::max(int(numsectors), 1) + 1, -1);
        const int hopCount = int(hopStorage.size());
        std::deque<int> queue;
        if (observation.sector >= 0 && observation.sector < hopCount)
        {
            hopStorage[size_t(observation.sector)] = 0;
            queue.push_back(observation.sector);
        }
        while (!queue.empty())
        {
            const int current = queue.front();
            queue.pop_front();
            auto graph = knownGraph.find(current);
            if (graph == knownGraph.end())
                continue;
            for (const Portal &edge : graph->second)
            {
                if (!(edge.traversable || edge.jumpable) || edge.to < 0 || edge.to >= hopCount)
                    continue;
                if (hopStorage[size_t(edge.to)] >= 0)
                    continue;
                hopStorage[size_t(edge.to)] = hopStorage[size_t(current)] + 1;
                queue.push_back(edge.to);
            }
        }
        // A dormant opportunity must not hide the frontiers behind it.
        // Filter suppressed work out of the candidate set, then let the
        // kernel rank what is genuinely still actionable.
        std::vector<DerivedFrontier> live;
        live.reserve(derivedFrontiers.size());
        for (const DerivedFrontier &candidateFrontier : derivedFrontiers)
        {
            if (blocked != (candidateFrontier.kind == kFrontierBlocked))
                continue;
            DerivedFrontier filtered;
            filtered.destination = candidateFrontier.destination;
            filtered.kind = candidateFrontier.kind;
            for (size_t i = 0; i < candidateFrontier.candidates.size(); ++i)
            {
                const Boundary &boundary = candidateFrontier.candidates[i];
                if (objectiveSuppressed(blocked
                        ? 4000000 + boundary.to
                        : 3000000 + boundary.wall * 8 + int(kObjectiveFrontier)))
                    continue;
                filtered.candidates.push_back(boundary);
            }
            if (!filtered.candidates.empty())
                live.push_back(filtered);
        }
        const int index = llmapper::selectFrontierIndex(live, observation.sector,
                                                        hopStorage.data(), hopCount);
        if (index < 0 || index >= int(live.size()))
            return false;
        const DerivedFrontier &frontier = live[size_t(index)];
        const Boundary *chosen = nullptr;
        for (size_t i = 0; i < frontier.candidates.size(); ++i)
        {
            const Boundary &candidate = frontier.candidates[i];
            if (blocked || candidate.traversable || candidate.jumpable)
            {
                chosen = &candidate;
                if (candidate.from == observation.sector)
                    break;
            }
        }
        if (!chosen)
            return false;
        Objective objective;
        objective.type = blocked ? kObjectiveInvestigate : kObjectiveFrontier;
        objective.id = chosen->wall;
        objective.sector = chosen->from;
        objective.targetSector = frontier.destination;
        objective.wall = chosen->wall;
        exploreDestination = frontier.destination;
        exploreCrossingWall = chosen->wall;
        exploreCrossingFrom = chosen->from;
        selectObjective(objective, blocked ? "INVESTIGATE_BLOCKED_FRONTIER"
                                           : (chosen->from == observation.sector
                                              ? "EXPLORE_LOCAL_PORTAL" : "EXPLORE_FRONTIER"));
        char detail[192];
        snprintf(detail, sizeof(detail),
                 "kind=%s dest=%d wall=%d from=%d candidates=%u",
                 blocked ? "blocked" : "open", frontier.destination, chosen->wall,
                 chosen->from, unsigned(frontier.candidates.size()));
        event(blocked ? "blocked_frontier_selected" : "frontier_selected", detail);
        return true;
    }

    const Portal *selectLocalPortal()
    {
        if (currentGoal == "EXPLORE_LOCAL_PORTAL" && currentGoalTarget >= 0)
        {
            for (const Portal &portal : observation.portals)
            {
                const int edgeId = portal.wall * 65536 + portal.to;
                if (portal.wall == currentGoalTarget && portal.traversable
                    && !visitedEdges.count(edgeId) && !edgeFailed(edgeId)
                    && !localEdgeFailed(edgeId))
                {
                    localJumpPortal = portal;
                    if (jumpFallbackEdges.count(edgeId))
                        localJumpPortal.capability = kTraversalJumpable;
                    return &localJumpPortal;
                }
            }
        }
        bool walkableAvailable = false;
        for (const Portal &portal : observation.portals)
        {
            if (!portal.localGeometry || portal.capability != kTraversalWalkable)
                continue;
            if (visitedSectors.count(portal.to))
                continue;
            const int edgeId = portal.wall * 65536 + portal.to;
            if (edgeFailed(edgeId) || localEdgeFailed(edgeId) || visitedEdges.count(edgeId)
                || openedRoutes.count(portal.from * 65536 + portal.to)
                || portal.to == lastTransitionFrom)
                continue;
            walkableAvailable = true;
            break;
        }

        const Portal *best = nullptr;
        int bestDistance = INT32_MAX;
        for (const Portal &portal : observation.portals)
        {
            if (!portal.localGeometry || !portal.traversable)
                continue;
            if (visitedSectors.count(portal.to))
                continue;
            if (walkableAvailable && portal.capability != kTraversalWalkable)
                continue;
            const int edgeId = portal.wall * 65536 + portal.to;
            if (edgeFailed(edgeId) || localEdgeFailed(edgeId) || visitedEdges.count(edgeId)
                || openedRoutes.count(portal.from * 65536 + portal.to)
                || portal.to == lastTransitionFrom)
                continue;
            const int score = distance2(observation.x, observation.y, portal.x, portal.y);
            if (score < bestDistance)
            {
                bestDistance = score;
                localJumpPortal = portal;
                if (jumpFallbackEdges.count(edgeId))
                    localJumpPortal.capability = kTraversalJumpable;
                best = &localJumpPortal;
            }
        }
        return best;
    }

    // Damage dealt and kills are read back out of the engine, so the
    // scorecard reports what happened rather than what the bot attempted.
    void trackCombatOutcomes()
    {
        for (int sprite_id : engagedEnemies)
        {
            if (!inRange(sprite_id, 0, kMaxSprites))
                continue;
            const spritetype &dude = sprite[sprite_id];
            const int health = validXSprite(dude.extra) ? xsprite[dude.extra].health : 0;
            auto known = enemyHealth.find(sprite_id);
            if (known == enemyHealth.end())
            {
                enemyHealth[sprite_id] = health;
                continue;
            }
            if (health < known->second)
            {
                char detail[96];
                snprintf(detail, sizeof(detail), "sprite=%d amount=%d",
                         sprite_id, known->second - health);
                event("damage_dealt", detail);
            }
            known->second = health;
            const bool dead = health <= 0 || dude.sectnum < 0 || dude.statnum != kStatDude;
            if (dead && countedKills.insert(sprite_id).second)
            {
                char detail[96];
                snprintf(detail, sizeof(detail), "sprite=%d type=%d", sprite_id, dude.type);
                event("enemy_killed", detail);
                lastSemanticProgressTick = observation.tick;
            }
        }
    }

    // Live, currently visible hostile.  Enemies are never taken from memory:
    // a remembered enemy informs awareness, it is not a valid target.
    const VisibleObject *selectVisibleEnemy()
    {
        const VisibleObject *best = nullptr;
        int bestDistance = INT32_MAX;
        for (const VisibleObject &object : observation.objects)
        {
            if (object.kind != kObjectEnemy)
                continue;
            if (unreachableEnemies.count(object.sprite))
                continue;
            if (!inRange(object.sprite, 0, kMaxSprites))
                continue;
            const spritetype &live = sprite[object.sprite];
            if (live.sectnum < 0 || live.statnum != kStatDude
                || !validXSprite(live.extra) || xsprite[live.extra].health <= 0)
                continue;
            const int distance = distance2(observation.x, observation.y, object.x, object.y);
            if (distance < bestDistance)
            {
                best = &object;
                bestDistance = distance;
            }
        }
        if (!best)
            return nullptr;
        selectedObject = *best;
        // Refresh to the engine's current pose: even one observation period
        // of lag is enough to make the shot miss a moving dude.
        selectedObject.x = sprite[best->sprite].x;
        selectedObject.y = sprite[best->sprite].y;
        selectedObject.z = sprite[best->sprite].z;
        selectedObject.sector = sprite[best->sprite].sectnum;
        if (engagedEnemies.insert(best->sprite).second)
        {
            char detail[128];
            snprintf(detail, sizeof(detail), "sprite=%d type=%d sector=%d distance=%d",
                     best->sprite, best->type, selectedObject.sector,
                     int(std::sqrt(double(bestDistance))));
            event("enemy_observed", detail);
        }
        return &selectedObject;
    }

    const VisibleObject *selectObject(ObjectKind kind)
    {
        if (kind == kObjectEnemy)
            return selectVisibleEnemy();
        const VisibleObject *best = nullptr;
        int bestDistance = INT32_MAX;
        for (const VisibleObject &object : observation.objects)
        {
            if (object.kind != kind)
                continue;
            const int distance = distance2(observation.x, observation.y, object.x, object.y);
            if (distance < bestDistance)
            {
                best = &object;
                bestDistance = distance;
            }
        }
        if (best)
        {
            selectedObject = *best;
            return &selectedObject;
        }
        for (const auto &entry : objectMemory)
        {
            const ObjectMemory &memory = entry.second;
            if (!memory.observed || memory.collected || memory.object.kind != kind)
                continue;
            if (memory.object.sprite < 0 || memory.object.sprite >= kMaxSprites
                || sprite[memory.object.sprite].sectnum < 0)
                continue;
            const int distance = distance2(observation.x, observation.y,
                                           memory.object.x, memory.object.y);
            if (distance < bestDistance)
            {
                selectedObject = memory.object;
                bestDistance = distance;
                best = &selectedObject;
            }
        }
        return best;
    }

    const VisibleObject *selectLocalAreaObject(ObjectKind kind)
    {
        ensureNavTopology();
        const int playerCell = nearestNavCell(observation.sector, observation.x, observation.y);
        if (playerCell < 0 || navCells[playerCell].walkArea < 0)
            return nullptr;
        const int playerArea = navCells[playerCell].walkArea;
        const VisibleObject *best = nullptr;
        int bestDistance = INT32_MAX;
        auto consider = [&](const VisibleObject &object)
        {
            if (object.kind != kind || objectiveSuppressed(5000000 + object.sprite))
                return;
            const int objectCell = nearestNavCell(object.sector, object.x, object.y);
            if (objectCell < 0 || navCells[objectCell].walkArea != playerArea)
                return;
            const int distance = distance2(observation.x, observation.y, object.x, object.y);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                selectedObject = object;
                best = &selectedObject;
            }
        };
        for (const VisibleObject &object : observation.objects)
            consider(object);
        for (const auto &entry : objectMemory)
        {
            const ObjectMemory &memory = entry.second;
            if (!memory.observed || memory.collected || !objectStillPresent(memory))
                continue;
            consider(memory.object);
        }
        return best;
    }

    bool objectStillPresent(const ObjectMemory &memory) const
    {
        const int id = memory.object.sprite;
        if (id < 0 || id >= kMaxSprites)
            return false;
        const spritetype &candidate = sprite[id];
        if (candidate.sectnum < 0)
            return false;
        if (memory.object.kind == kObjectEnemy)
            return candidate.statnum == kStatDude && validXSprite(candidate.extra)
                && xsprite[candidate.extra].health > 0;
        if (memory.object.kind == kObjectKey || memory.object.kind == kObjectPickup)
            return candidate.statnum == kStatItem && itemCategory(candidate.type) != nullptr;
        return true;
    }

    InteractionMemory *selectNewInteraction(bool includeRecovery = false)
    {
        InteractionMemory *best = nullptr;
        int bestDistance = INT32_MAX;
        for (auto &entry : interactions)
        {
            InteractionMemory &memory = entry.second;
            if (!memory.observed)
                continue;
            if (objectiveSuppressed(1000000 + interactionMemoryKey(memory)))
                continue;
            const bool reactivation = interactionNeedsReactivation(memory);
            if (memory.attempted && !reactivation
                && !(includeRecovery && recoveryMode && memory.activated
                     && memory.reversible && !memory.recoveryAttempted))
                continue;
            if (memory.unavailableFromSector == observation.sector)
                continue;
            if (memory.fromSector != observation.sector && !includeRecovery)
                continue;
            // A currently traversable portal is already a valid exploration
            // frontier.  Its optional Wallpush metadata must not turn the
            // route into a separate interaction goal that pulls the bot back
            // to the source side after it has crossed (notably for safe
            // drop/elevator portals).  Closed portals and one-sided push
            // walls remain first-class interaction work.
            if (!reactivation && memory.target.wall >= 0 && memory.target.traversable
                && !memory.target.sectorPush)
                continue;
            int distance = INT32_MAX;
            if (memory.fromSector == observation.sector)
                distance = distance2(observation.x, observation.y, memory.x, memory.y);
            else
            {
                Portal route;
                if (!findKnownRoute(memory.fromSector, route))
                    continue;
                distance = distance2(observation.x, observation.y, route.x, route.y) + 1000000;
            }
            // When several visible mechanisms share a compact room, prefer
            // the one the real ActionScan already resolves from this pose.
            // This keeps a moving slide-door sequence ordered by the engine's
            // current target instead of by map-record iteration order.
            int hit = -1;
            int target = -1;
            int extra = -1;
            hit = memory.fromSector == observation.sector
                ? ActionScanPreview(gMe, &target, &extra) : -1;
            const bool actionValid = memory.fromSector == observation.sector
                && interactionTargetMatches(memory, hit, target);
            if (!actionValid)
                distance += 100000000;
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = &memory;
            }
        }
        return best;
    }

    bool interactionNeedsReactivation(const InteractionMemory &memory) const
    {
        if (!memory.attempted || !memory.activated || !memory.reversible
            || memory.target.wall < 0
            || memory.target.to < 0 || memory.target.from != observation.sector)
            return false;
        // Crossing it once is not a precondition for wanting it open again;
        // a door the bot opened but never reached is exactly the case that
        // most needs reopening.
        if (!memory.traversed && !visitedSectors.count(memory.targetSector)
            && memory.activationCount >= kMaxActivationAttempts)
            return false;
        // Leave alone while it is finishing motion the bot asked for.
        if (memory.activationCount > 0 && interactionStillBusy(memory))
            return false;
        if (memory.lastActivationTick >= 0
            && observation.tick - memory.lastActivationTick < kReactivationCooldownTicks)
            return false;
        if (interactionStateSignature(memory) == memory.afterState)
            return false;
        // An activation the world visibly answered has already done its job.
        // Asking for it again needs evidence that what it produced is gone --
        // the mechanism back at the state it started from -- not merely that
        // this particular boundary is still shut.  A mechanism whose effect
        // is elsewhere (a rotating sector clearing floor space, a door on a
        // remote TX channel) never makes its own wall passable, so that test
        // alone had the bot pressing a switch a second time and undoing the
        // opening it had just made for itself.
        if (memory.observedKnownWorldDelta
            && interactionStateSignature(memory) != memory.beforeState)
            return false;
        return memory.target.interactionAffordance && !memory.target.traversable;
    }

    // At any moment it must be possible to say who is aiming the camera and
    // at what.  Emitted only on change, so a run log stays readable.
    void noteCameraOwner(const char *owner, int target, int angle, int look)
    {
        if (cameraOwner == owner && cameraTarget == target)
            return;
        cameraOwner = owner;
        cameraTarget = target;
        char detail[160];
        snprintf(detail, sizeof(detail), "owner=%s target=%d want_angle=%d want_look=%d goal=%s",
                 owner, target, angle, look, currentGoal.c_str());
        event("camera_owner", detail);
    }

    // Move this mechanism's approach point to a surface that has not been
    // tried from the bot's current side.  Returns false when they have all
    // been tried, which is the honest answer that it cannot be used here.
    bool advanceInteractionSurface(InteractionMemory &memory)
    {
        const int key = interactionMemoryKey(memory);
        const InteractionCandidate *next = nullptr;
        int bestDistance = INT32_MAX;
        for (const InteractionCandidate &candidate : observation.interactions)
        {
            if (interactionKey(candidate) != key)
                continue;
            if (candidate.target.wall >= 0
                && triedSurfaces.count(key * 8192 + candidate.target.wall))
                continue;
            const int distance = distance2(observation.x, observation.y,
                                           candidate.x, candidate.y);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                next = &candidate;
            }
        }
        if (!next)
            return false;
        if (memory.target.wall >= 0)
            triedSurfaces.insert(key * 8192 + memory.target.wall);
        char detail[208];
        snprintf(detail, sizeof(detail),
                 "mechanism=%d from_wall=%d to_wall=%d at=(%d,%d) distance=%d",
                 memory.id, memory.target.wall, next->target.wall, next->x, next->y,
                 int(std::sqrt(double(bestDistance))));
        event("interaction_surface_retried", detail);
        memory.target = next->target;
        memory.x = next->x;
        memory.y = next->y;
        memory.z = next->z;
        memory.unavailablePose = 0;
        resetNavigation();
        movementTargetActive = false;
        return true;
    }

    // Break a wall Blood marks as answering to weapon impact.  Any weapon
    // will do the job; the pitchfork is the fallback when nothing else is
    // to hand, since refusing to try it would leave the route shut.
    GINPUT shootObstacle(InteractionMemory &memory)
    {
        setGoal("BREAK_OBSTACLE", memory.id);
        GINPUT input = {};
        // Aim at the wall, not at an ActionScan pose.  A push mechanism is
        // approached along the surface normal and aimed at the middle of the
        // sector's height; for a shot that means firing at the ceiling, and
        // from off to one side, at nothing at all.  A wall spans the full
        // height, so the shot is simply level, straight at its midpoint.
        const int aimAngle = getangle(memory.x - observation.x, memory.y - observation.y);
        const int turn = angleDelta(aimAngle, observation.angle);
        const int horizontal = std::max(1, int(std::sqrt(double(distance2(
            observation.x, observation.y, memory.x, memory.y)))));
        const POSTURE &stand = gMe->pPosture[gMe->lifeMode][kPostureStand];
        const int eyeZ = gMe->pSprite->z - stand.eyeAboveZ;
        const int aimZ = std::max(memory.target.interactionTopZ + 256,
                                  std::min(memory.target.interactionBottomZ - 256, eyeZ));
        const int desiredLook = lookAngleForTarget(eyeZ, aimZ, horizontal);
        const int lookDelta = desiredLook - fix16_to_int(gMe->q16look);

        int weapon = 0;
        const bool ranged = rangedWeaponAvailable(weapon);
        if (!ranged && meleeWeaponAvailable())
            weapon = kWeaponPitchfork;
        if (!weapon)
        {
            memory.unavailableFromSector = observation.sector;
            event("break_obstacle_unarmed", "reason=no_weapon_available");
            return input;
        }
        // Melee has to be within reach; a firearm needs a clear shot.  Firing
        // from wherever the approach happened to stop just buries the burst
        // in whatever stands between the bot and the target: the ammunition
        // drains, the aim reads as perfect, and the wall never registers.
        const bool clearShot = cansee(observation.x, observation.y, eyeZ,
                                      observation.sector, memory.x, memory.y, aimZ,
                                      memory.fromSector >= 0 ? memory.fromSector
                                                             : observation.sector);
        const int reach = weapon == kWeaponPitchfork ? kMeleeReach : kActionApproachRange * 2;
        const int distance2ToWall = distance2(observation.x, observation.y,
                                              memory.x, memory.y);
        if (distance2ToWall > reach * reach || !clearShot)
        {
            noteCameraOwner("NAVIGATION", memory.id, aimAngle, 0);
            return navigateTo(memory.x, memory.y, memory.z, memory.fromSector,
                              memory.id, kTraversalUnknown);
        }

        input.q16turn = fix16_from_int(turn);
        input.q16mlook = encodeLook(lookDelta);
        noteCameraOwner("BREAK_OBSTACLE", memory.id, aimAngle, desiredLook);
        if (gMe->curWeapon != weapon)
        {
            input.syncFlags.weaponChange = 1;
            input.newWeapon = uint8_t(weapon);
            if (memory.activationCount == 0)
            {
                char detail[176];
                snprintf(detail, sizeof(detail), "wall=%d weapon=%d ranged=%d distance=%d",
                         memory.target.wall, weapon, ranged ? 1 : 0, horizontal);
                event("break_obstacle_started", detail);
            }
            return input;
        }
        if (std::abs(turn) < 96 && std::abs(lookDelta) < 96 && clearShot)
        {
            input.buttonFlags.shoot = 1;
            memory.attempted = true;
            // One transaction covers the whole burst.  Re-snapshotting the
            // pre-state for every projectile meant the bot never had a fixed
            // "before" to compare against, so the same effect was discovered
            // over and over and the shooting never concluded.
            if (memory.activationCount == 0)
            {
                memory.beforeState = interactionStateSignature(memory);
                memory.observedKnownWorldDelta = false;
            }
            if (memory.state != 2)
                memory.state = 1;
            memory.lastActivationTick = observation.tick;
            ++memory.activationCount;
            if ((memory.activationCount % 8) == 1)
            {
                int wallState = -1;
                int triggered = -1;
                if (inRange(memory.target.wall, 0, numwalls)
                    && inRange(wall[memory.target.wall].extra, 1, kMaxXWalls))
                {
                    const XWALL &record = xwall[wall[memory.target.wall].extra];
                    wallState = record.state;
                    triggered = record.isTriggered;
                }
                const int ammo = weapon - 1 >= 0
                    && weapon - 1 < int(sizeof(gMe->ammoCount) / sizeof(gMe->ammoCount[0]))
                    ? gMe->ammoCount[weapon - 1] : -1;
                char detail[256];
                snprintf(detail, sizeof(detail),
                         "wall=%d weapon=%d shots=%d state=%d triggered=%d ammo=%d turn=%d look=%d distance=%d",
                         memory.target.wall, weapon, memory.activationCount,
                         wallState, triggered, ammo, turn, lookDelta, horizontal);
                event("break_obstacle_shot", detail);
            }
        }
        return input;
    }

    GINPUT steerInteraction(InteractionMemory &memory)
    {
        if (memory.fromSector != observation.sector)
        {
            // Route through visited space with the ordinary navigator: the
            // mechanism's own sector is the destination, and crossing any
            // number of already-walked boundaries to get there is normal
            // transportation, not a new exploration decision.
            setGoal("NAVIGATE_TO_NEW_INTERACTION", memory.id);
            const GINPUT approach = navigateTo(memory.x, memory.y, memory.z,
                                               memory.fromSector, memory.id,
                                               kTraversalUnknown);
            if (approach.forward || approach.strafe || approach.q16turn
                || approach.q16mlook || approach.buttonFlags.jump
                || approach.buttonFlags.crouch)
                return approach;
            // The navigator produced nothing at all: fall back to the single
            // known crossing toward that sector before giving up on it.
            Portal route;
            if (findKnownRoute(memory.fromSector, route))
                return steerPortal(route);
            memory.unavailableFromSector = observation.sector;
            return GINPUT{};
        }

        if (memory.target.shootable)
        {
            // Once the wall has answered, stop.  A vector-activated wall is
            // not necessarily destroyed by the first hit, but it has done
            // something, and continuing to fire is ammunition spent on a
            // solved problem.
            if (memory.state == 2 || memory.observedKnownWorldDelta)
            {
                event("break_obstacle_answered", "reason=world_delta_observed");
                return GINPUT{};
            }
            return shootObstacle(memory);
        }

        int hit = -1;
        int target = -1;
        int extra = -1;
        const bool interactionCrouch = interactionNeedsCrouch(memory);
        memory.target.interactionCrouch = interactionCrouch;
        const int expectedDistance = memory.kind == kInteractionSector
            ? 0 : distance2(observation.x, observation.y, memory.x, memory.y);
        const int expectedAngle = interactionFacingAngle(memory);
        const int desiredLook = interactionLookTarget(memory, interactionCrouch);
        const int currentLook = fix16_to_int(gMe->q16look);
        const int lookDelta = desiredLook - currentLook;
        hit = ActionScanPreview(gMe, &target, &extra);
        const bool valid = interactionTargetMatches(memory, hit, target);
        const bool missingKey = memory.key && !hasKey(memory.key);
        const char *status = missingKey ? "LOCKED_KEY_REQUIRED"
            : valid ? "VALID_ACTION_TARGET"
            : expectedDistance > kActionScanRange * kActionScanRange ? "NOT_IN_USE_RANGE"
            : std::abs(angleDelta(expectedAngle, observation.angle)) >= 96 ? "NOT_FACING_ACTION_TARGET"
            : std::abs(lookDelta) >= 64 ? "NOT_AIMED_AT_ACTION_TARGET"
            : "NO_ACTION_TARGET";
        char detail[360];
        snprintf(detail, sizeof(detail),
                 "kind=%d id=%d status=%s hit=%d target=%d extra=%d distance=%d angle_delta=%d look=%d desired_look=%d look_delta=%d target_z=%d target_top_z=%d target_bottom_z=%d crouch=%d attempted=%d activations=%d",
                 int(memory.kind), memory.id, status, hit, target, extra,
                 int(std::sqrt(double(expectedDistance))), angleDelta(expectedAngle, observation.angle),
                 currentLook, desiredLook, lookDelta, memory.target.z,
                 memory.target.interactionTopZ, memory.target.interactionBottomZ,
                 interactionCrouch ? 1 : 0, memory.attempted ? 1 : 0,
                 memory.activationCount);
        event("interaction_probe", detail);

        // Do not interrupt a mechanism the bot itself just set in motion.
        // Some Blood mechanisms animate continuously, though, and treating
        // "busy" as untouchable made those permanently unusable: the bot
        // stood in front of a rotating door reporting a valid action target
        // for a minute without ever pressing Use.
        const bool selfCausedMotion = memory.lastActivationTick >= 0
            && observation.tick - memory.lastActivationTick < kReactivationCooldownTicks;
        const bool mechanismMoving = interactionStillBusy(memory)
            && (selfCausedMotion || memory.activationCount > 0);
        bool issueUse = false;
        const bool recoveryActivation = recoveryMode && memory.attempted
            && memory.activated && memory.reversible && !memory.recoveryAttempted;
        if (recoveryActivation)
            memory.recoveryAttempted = true;
        const bool reactivation = interactionNeedsReactivation(memory);
        // Having pressed Use once, long ago, is not a reason never to press
        // it again.  The old condition meant the bot could stand in front of
        // a mechanism the engine was actively reporting as a valid action
        // target and simply refuse to touch it -- for a minute at a time.
        // Press when the engine says the target is good and the mechanism is
        // not already moving, bounded by a cooldown and an attempt budget;
        // the objective budget and dormancy handle the rest.
        const int movedSector = mechanismSector(memory.target, memory.targetSector);
        const bool wouldMoveOwnSector = movedSector >= 0
            && movedSector == observation.sector && !memory.target.sectorPushCurrent;
        if (wouldMoveOwnSector && !memory.attempted)
        {
            char detail[176];
            snprintf(detail, sizeof(detail),
                     "kind=%d id=%d mechanism_sector=%d player_sector=%d reason=would_move_occupied_sector",
                     int(memory.kind), memory.id, movedSector, observation.sector);
            event("interaction_refused", detail);
        }
        const bool cooldownElapsed = memory.lastActivationTick < 0
            || observation.tick - memory.lastActivationTick >= kReactivationCooldownTicks;
        const bool budgetLeft = memory.activationCount < kMaxActivationAttempts;
        if (valid && !missingKey && !mechanismMoving && cooldownElapsed
            && !wouldMoveOwnSector
            && (!memory.attempted || recoveryActivation || reactivation || budgetLeft))
        {
            if (reactivation)
                event("interaction_reactivation", "reason=previously_traversed_route_closed");
            memory.attempted = true;
            memory.state = 1;
            memory.activated = false;
            triedSurfaces.erase(interactionMemoryKey(memory) * 8192 + memory.target.wall);
            memory.lastActivationTick = observation.tick;
            memory.beforeState = interactionStateSignature(memory);
            memory.beforePortals = observation.portals;
            memory.engineAccepted = false;
            memory.observedLocalEffect = false;
            memory.observedKnownWorldDelta = false;
            emitInteractionPortalState(memory, "before_use_pulse");
            emitInteractionGeometryState(memory, "before_use_pulse");
            ++memory.activationCount;
            pendingInteractionKey = interactionMemoryKey(memory);
            issueUse = true;
            snprintf(detail, sizeof(detail), "kind=%d id=%d hit=%d target=%d attempt=%d",
                     int(memory.kind), memory.id, hit, target, memory.activationCount);
            event("interaction_started", detail);
            event("use_pulse", detail);
        }
        if (!issueUse && memory.attempted && memory.state == 1
            && observation.tick - memory.lastActivationTick > kInteractionTimeoutTicks)
        {
            if (memory.engineAccepted)
            {
                memory.state = 2;
                memory.activated = true;
                memory.afterState = interactionStateSignature(memory);
                event("interaction_settled", "engine_accepted=1 observed_local_effect=0 observed_known_world_delta=0");
            }
            else
            {
                memory.state = 3;
                event("interaction_failed", "reason=INTERACTION_VALID_BUT_NO_RESPONSE");
            }
        }
        if (!valid && !missingKey && memory.state != 1
            && expectedDistance <= kActionScanRange * kActionScanRange
            && std::abs(angleDelta(expectedAngle, observation.angle)) < 96
            && std::abs(lookDelta) < 64)
        {
            ++memory.unavailableAttempts;
            memory.unavailableState = interactionStateSignature(memory);
            if (memory.unavailableAttempts >= kSurfaceRetryProbes)
            {
                // Squared up, in range, and the engine still will not name
                // this as a target.  Try a different surface of the same
                // mechanism before concluding it cannot be used from here.
                if (advanceInteractionSurface(memory))
                    memory.unavailableAttempts = 0;
                else
                {
                    memory.unavailableFromSector = observation.sector;
                    event("interaction_unavailable",
                          "reason=CURRENTLY_UNAVAILABLE_FROM_THIS_SIDE attempts=3");
                }
            }
            else
                event("interaction_alternative_pose", "reason=NO_ACTION_TARGET");
        }
        else if (!valid && !missingKey && expectedDistance <= kActionScanRange * kActionScanRange
                 && std::abs(angleDelta(expectedAngle, observation.angle)) < 96
                 && std::abs(lookDelta) >= 64)
        {
            event(interactionCrouch ? "interaction_crouch_pose" : "interaction_vertical_pose",
                  interactionCrouch ? "reason=target_below_standing_look_range"
                                     : "reason=target_height_requires_vertical_aim");
        }
        setGoal(memory.kind == kInteractionSprite ? "USE_NEW_SPRITE_INTERACTION"
                : memory.kind == kInteractionSector ? "USE_NEW_SECTOR_INTERACTION"
                : "USE_NEW_WALL_INTERACTION", memory.id);
        // Crouching is an interaction *pose*, not a way to travel.  Walking
        // the whole approach crouched halves the bot's speed and leaves it
        // grinding along walls; the posture is applied at the point of use.
        const TraversalCapability interactionCapability = kTraversalUnknown;
        // Preview was taken from this exact player pose.  A USE pulse must
        // not also turn or move the player before ProcessInput performs its
        // authoritative ActionScan, otherwise the preview and real scan can
        // resolve different walls/objects.
        GINPUT input = {};
        if (issueUse)
            input.keyFlags.action = 1;
        else
        {
            // Stand on the side the mechanism is being operated from.  Aiming
            // navigation at the mechanism's target sector asks the bot to walk
            // through the very door it is trying to open, which is exactly the
            // situation after a door closes behind it.
            const int navigationSector = memory.fromSector >= 0
                ? memory.fromSector : observation.sector;
            const MovementProbe interactionPath = probeMovement(
                observation.x, observation.y, observation.z, observation.sector,
                memory.x, memory.y, navigationSector, std::max(256, kActionApproachRange / 2));
            // A body probe can quite correctly report the intended push wall
            // as the blocker.  That is the interaction pose, not a detour
            // opportunity: let the real player collision settle against the
            // surface while ActionScan remains authoritative.
            if (memory.target.wall >= 0 && interactionPath.wall == memory.target.wall)
            {
                resetNavigation();
                input = steerTo(memory.x, memory.y, false, false,
                                memory.id, navigationSector, interactionCapability);
            }
            else
                input = navigateTo(memory.x, memory.y, memory.z, navigationSector,
                                   memory.id, interactionCapability);
            if (navigationActionableBlocker >= 0
                && navigationActionableBlocker != memory.target.wall)
            {
                const int blockerKey = int(kInteractionWall) * 1000000
                    + navigationActionableBlocker + 1;
                auto blocker = interactions.find(blockerKey);
                if (blocker != interactions.end() && blocker->second.observed)
                {
                    char blockerDetail[96];
                    snprintf(blockerDetail, sizeof(blockerDetail),
                             "wall=%d original=%d", navigationActionableBlocker,
                             memory.target.wall);
                    event("navigation_interaction_opportunity", blockerDetail);
                }
                navigationActionableBlocker = -1;
                return GINPUT{};
            }
            if (navigationUsingDetour)
                return input;
            if (expectedDistance > kActionScanRange * kActionScanRange)
            {
                // Still travelling.  Navigation owns movement and the camera,
                // so the bot looks along the route it is actually walking.
                noteCameraOwner("NAVIGATION", memory.id,
                                observation.angle + fix16_to_int(input.q16turn), 0);
                return input;
            }
            // Inside ActionScan range: square up to the target and aim.  The
            // engine's preview, not this code, decides whether the pose works.
            const int turnToTarget = angleDelta(expectedAngle, observation.angle);
            input = GINPUT{};
            input.syncFlags.run = 0;
            input.q16turn = fix16_from_int(turnToTarget);
            input.q16mlook = encodeLook(lookDelta);
            if (interactionCrouch)
                input.buttonFlags.crouch = 1;
            // Close the last stretch only once roughly squared up, so the
            // bot walks in facing the surface rather than sliding past it.
            const int snug = std::max(playerClipRadius() + 128, kActionScanRange / 3);
            if (std::abs(turnToTarget) < 96 && expectedDistance > snug * snug)
                input.forward = 1024;
            noteCameraOwner("INTERACTION", memory.id, expectedAngle, desiredLook);
        }
        return input;
    }

    bool findKnownRoute(int targetSector, Portal &route) const
    {
        if (targetSector < 0 || targetSector == observation.sector)
            return false;
        std::vector<int> frontier(1, observation.sector);
        std::set<int> reached;
        std::map<int, Portal> parent;
        reached.insert(observation.sector);
        for (size_t index = 0; index < frontier.size(); ++index)
        {
            const int current = frontier[index];
            auto graph = knownGraph.find(current);
            if (graph == knownGraph.end())
                continue;
            for (const Portal &edge : graph->second)
            {
                if (edgeFailed(edge.wall * 65536 + edge.to))
                    continue;
                if (reached.insert(edge.to).second)
                {
                    parent[edge.to] = edge;
                    frontier.push_back(edge.to);
                }
            }
        }
        if (!reached.count(targetSector))
            return false;
        int cursor = targetSector;
        route = parent[cursor];
        while (route.from != observation.sector)
        {
            cursor = route.from;
            auto edge = parent.find(cursor);
            if (edge == parent.end())
                return false;
            route = edge->second;
        }
        return true;
    }

    bool directObjectReachable(const VisibleObject &object) const
    {
        if (object.sector != observation.sector
            || std::abs(object.z - observation.z) > kMaxWalkableStep)
            return false;
        const int radius = gMe && gMe->pSprite ? (gMe->pSprite->clipdist << 2) : 128;
        return probeMovement(observation.x, observation.y, observation.z, observation.sector,
                             object.x, object.y, object.sector, std::max(512, radius * 2)).reachable;
    }

    TraversalCapability currentClearanceCapability() const
    {
        if (!inRange(observation.sector, 0, numsectors))
            return kTraversalUnknown;
        const int floorZ = getflorzofslope(observation.sector, observation.x, observation.y);
        const int ceilingZ = getceilzofslope(observation.sector, observation.x, observation.y);
        const int clearance = floorZ - ceilingZ;
        if (clearance < playerBodyClearance() && clearance >= playerCrouchClearance())
            return kTraversalCrouchable;
        return kTraversalUnknown;
    }

    TraversalCapability currentRecoveryCapability() const
    {
        const TraversalCapability clearance = currentClearanceCapability();
        if (clearance == kTraversalCrouchable)
            return clearance;
        if (!localDynamicJumpAttempted && observation.tick <= localDynamicJumpUntilTick)
            return kTraversalJumpable;
        return kTraversalUnknown;
    }

    bool rangedWeaponAvailable(int &weapon) const
    {
        static const int candidates[] = {
            kWeaponTommy, kWeaponShotgun, kWeaponFlare, kWeaponTesla, kWeaponNapalm,
        };
        for (int candidate : candidates)
        {
            if (!gMe->hasWeapon[candidate])
                continue;
            const int ammo = candidate - 1;
            if (gInfiniteAmmo || (ammo >= 0 && ammo < int(sizeof(gMe->ammoCount) / sizeof(gMe->ammoCount[0]))
                                  && gMe->ammoCount[ammo] > 0))
            {
                weapon = candidate;
                return true;
            }
        }
        return false;
    }

    GINPUT aimAndShoot(const VisibleObject &enemy, int weapon)
    {
        GINPUT input = {};
        input.syncFlags.run = 1;
        const int targetAngle = getangle(enemy.x - observation.x, enemy.y - observation.y);
        input.q16turn = fix16_from_int(angleDelta(targetAngle, observation.angle));
        const int horizontal = std::max(1, int(std::sqrt(double(distance2(observation.x, observation.y,
                                                                         enemy.x, enemy.y)))));
        const int targetZ = enemy.z;
        const double pitch = std::atan2(double(observation.z - targetZ), double(horizontal))
            * 1024.0 / 3.14159265358979323846;
        const int desiredLook = std::max(-347, std::min(289, int(std::lround(pitch))));
        const int currentLook = fix16_to_int(gMe->q16look);
        input.q16mlook = encodeLook(desiredLook - currentLook);
        if (gMe->curWeapon == weapon)
            input.buttonFlags.shoot = 1;
        else
        {
            input.newWeapon = uint8_t(weapon);
            char detail[96];
            snprintf(detail, sizeof(detail), "enemy=%d weapon=%d", enemy.sprite, weapon);
            event("ranged_weapon_selected", detail);
        }
        return input;
    }

    bool actionTargetMatches(const Portal &portal, int hit, int target) const
    {
        if (portal.wallPush && hit == 0 && target == portal.wall)
            return true;
        if (portal.sectorPush && hit == 6 && target == portal.to)
            return true;
        if (portal.sectorPushCurrent && hit == 6 && target == portal.from)
            return true;
        return false;
    }

    void emitUseProbe(const Portal &portal, const DoorMemory &memory, bool atDoor,
                      int hit, int target, int extra, const char *status)
    {
        if (lastUseProbeDoor == portal.wall && lastUseProbeStatus == status
            && observation.tick - lastUseProbeTick < kTicsPerSec)
            return;
        lastUseProbeDoor = portal.wall;
        lastUseProbeTick = observation.tick;
        lastUseProbeStatus = status;
        const int targetAngle = getangle(portal.x - observation.x, portal.y - observation.y);
        char detail[320];
        snprintf(detail, sizeof(detail),
                 "status=%s at_door=%d goal=%s expected_wall=%d expected_from=%d expected_sector=%d expected_wall_push=%d expected_sector_push=%d expected_current_sector_push=%d hit=%d target=%d extra=%d distance=%d angle_delta=%d key=%d locked=%d interaction=%d attempts=%d",
                 status, atDoor ? 1 : 0, currentGoal.c_str(), portal.wall, portal.from, portal.to, portal.wallPush ? 1 : 0,
                 portal.sectorPush ? 1 : 0, portal.sectorPushCurrent ? 1 : 0, hit, target, extra,
                 int(std::sqrt(double(distance2(observation.x, observation.y, portal.x, portal.y)))),
                 angleDelta(targetAngle, observation.angle), portal.key, portal.locked ? 1 : 0,
                 int(memory.interaction), memory.attempts);
        event("use_probe", detail);
    }

    void setGoal(const char *goal, int target = -1)
    {
        if (currentGoal == goal && currentGoalTarget == target)
            return;
        currentGoal = goal;
        currentGoalTarget = target;
        char detail[160];
        snprintf(detail, sizeof(detail), "goal=%s target=%d", goal, target);
        event("goal_changed", detail);
    }

    // A stable identity for one unit of committed work.  Suppression and
    // progress accounting are keyed on this, never on the tick-local object.
    int objectiveKey(const Objective &objective) const
    {
        switch (objective.type)
        {
        case kObjectiveInteraction:
            return 1000000 + objective.interactionKey;
        case kObjectiveFrontier:
            return 3000000 + objective.wall * 8 + int(objective.type);
        case kObjectiveInvestigate:
            return 4000000 + objective.targetSector;
        case kObjectivePickup:
        case kObjectiveKey:
            return 5000000 + objective.id;
        case kObjectiveExpose:
            return 7000000 + objective.sector;
        default:
            return -1;
        }
    }

    bool objectiveSuppressed(int key) const
    {
        if (key < 0)
            return false;
        auto entry = suppressedUntil.find(key);
        return entry != suppressedUntil.end() && observation.tick < entry->second;
    }

    void suppressObjective(int key, const char *reason)
    {
        if (key < 0)
            return;
        lastFailedOpportunity = key;
        lastFailedOpportunityTick = observation.tick;
        const int count = ++suppressionCount[key];
        const int shift = count > 3 ? 3 : count - 1;
        const int cooldown = kSuppressionBaseTicks << shift;
        suppressedUntil[key] = observation.tick + cooldown;
        // What the bot knew when it gave up.  Waking this work again before
        // the cooldown needs a reason, and "nothing else to do" is not one:
        // retrying the same failure on the same knowledge is the thrash the
        // backoff exists to prevent.
        suppressionEvidence[key] = worldEvidenceRevision();
        char detail[160];
        snprintf(detail, sizeof(detail), "key=%d attempts=%d cooldown_s=%d reason=%s",
                 key, count, cooldown / kTicsPerSec, reason);
        event("opportunity_dormant", detail);
    }

    // Where the current objective is trying to get to, in world space.
    bool objectiveAnchor(int &outX, int &outY) const
    {
        switch (currentObjective.type)
        {
        case kObjectiveInteraction:
        case kObjectivePickup:
        case kObjectiveKey:
            outX = currentObjective.x;
            outY = currentObjective.y;
            return currentObjective.x != 0 || currentObjective.y != 0;
        case kObjectiveFrontier:
        case kObjectiveInvestigate:
        {
            const Portal *crossing = portalByWallId(currentObjective.wall);
            if (!crossing)
                return false;
            outX = crossing->x;
            outY = crossing->y;
            return true;
        }
        default:
            return false;
        }
    }

    // Is the crossing this objective depends on currently in motion?
    bool committedMechanismBusy() const
    {
        if (!currentObjective.active)
            return false;
        if (observation.localSectorBusy != 0)
            return true;
        for (const Portal &portal : observation.portals)
        {
            if (currentObjective.wall >= 0 && portal.wall != currentObjective.wall)
                continue;
            if (currentObjective.wall < 0
                && portal.to != currentObjective.targetSector)
                continue;
            if (portal.wallBusy || portal.sectorBusy)
                return true;
        }
        if (inRange(currentObjective.targetSector, 0, numsectors))
        {
            const int extra = sector[currentObjective.targetSector].extra;
            if (extra > 0 && extra < kMaxXSectors && xsector[extra].busy != 0)
                return true;
        }
        return false;
    }

    // Bound every committed objective.  Returns true when the objective was
    // released, so the caller must reselect rather than keep executing it.
    bool enforceObjectiveBudget()
    {
        if (!currentObjective.active)
            return false;
        int anchorX = 0;
        int anchorY = 0;
        const bool hasAnchor = objectiveAnchor(anchorX, anchorY);
        if (objectiveProgressTick < 0)
        {
            objectiveProgressTick = observation.tick;
            objectiveBestDistance2 = hasAnchor
                ? distance2(observation.x, observation.y, anchorX, anchorY) : INT32_MAX;
            objectiveProgressSector = observation.sector;
            objectiveBestRemainingSteps = INT32_MAX;
        }
        bool progressed = observation.sector != objectiveProgressSector;
        if (hasAnchor)
        {
            const int current = distance2(observation.x, observation.y, anchorX, anchorY);
            if (current + kObjectiveProgressEpsilon2 < objectiveBestDistance2)
            {
                objectiveBestDistance2 = current;
                progressed = true;
            }
        }
        // Following the plan is progress even while the plan leads away.
        if (!navRoute.empty())
        {
            const int remaining = int(navRoute.size()) - int(navRouteIndex);
            if (remaining < objectiveBestRemainingSteps)
            {
                objectiveBestRemainingSteps = remaining;
                progressed = true;
            }
            if (observation.tick - navWaypointProgressTick <= kTicsPerSec)
                progressed = true;
        }
        else
            objectiveBestRemainingSteps = INT32_MAX;
        // Waiting on moving geometry counts as progress, but not forever.
        // Some Blood mechanisms never stop moving, and an unbounded grace
        // meant the objective could never time out: the bot stood in front
        // of one for the rest of the run, reporting that it was waiting.
        if (!progressed && committedMechanismBusy()
            && (waitingSinceTick < 0
                || observation.tick - waitingSinceTick <= kMaxDoorWaitTicks))
        {
            if (waitingSinceTick < 0)
                waitingSinceTick = observation.tick;
            // The world is still changing in the bot's favour.  Waiting for
            // it is the plan, not a failure of the plan.
            if (mechanismWaitTick < 0
                || observation.tick - mechanismWaitTick > 4 * kTicsPerSec)
            {
                mechanismWaitTick = observation.tick;
                char detail[160];
                snprintf(detail, sizeof(detail), "wall=%d target_sector=%d goal=%s",
                         currentObjective.wall, currentObjective.targetSector,
                         currentGoal.c_str());
                event("waiting_for_mechanism", detail);
            }
            progressed = true;
        }
        if (progressed)
        {
            objectiveProgressSector = observation.sector;
            objectiveProgressTick = observation.tick;
            waitingSinceTick = -1;
            return false;
        }
        const int stalled = observation.tick - objectiveProgressTick;
        const int lifetime = currentObjective.startedTick >= 0
            ? observation.tick - currentObjective.startedTick : 0;
        if (stalled < kObjectiveStallTicks && lifetime < kObjectiveHardTicks)
            return false;
        const int key = objectiveKey(currentObjective);
        const char *reason = stalled >= kObjectiveStallTicks
            ? "no_progress_toward_objective" : "objective_time_budget";
        char detail[224];
        snprintf(detail, sizeof(detail),
                 "type=%d id=%d key=%d stalled_s=%d lifetime_s=%d sector=%d reason=%s",
                 int(currentObjective.type), currentObjective.id, key,
                 stalled / kTicsPerSec, lifetime / kTicsPerSec, observation.sector, reason);
        event("objective_budget_exhausted", detail);
        suppressObjective(key, reason);
        invalidateObjective(reason, false);
        return true;
    }

    void selectObjective(const Objective &objective, const char *goal)
    {
        currentObjective = objective;
        currentObjective.active = true;
        currentObjective.startedTick = observation.tick;
        objectiveProgressTick = -1;
        objectiveBestDistance2 = INT32_MAX;
        objectiveProgressSector = observation.sector;
        waitingSinceTick = -1;
        setGoal(goal, objective.id);
        if (objective.type == kObjectiveFrontier || objective.type == kObjectiveInvestigate)
        {
            if (lastFrontierTarget == objective.targetSector)
            {
                ++frontierSelectionCount;
                char detail[192];
                snprintf(detail, sizeof(detail),
                         "wall=%d source=%d target=%d current=%d count=%d",
                         objective.wall, objective.sector, objective.targetSector,
                         observation.sector, frontierSelectionCount);
                event("frontier_reselected", detail);
            }
            else
            {
                lastFrontierWall = objective.wall;
                lastFrontierSource = objective.sector;
                lastFrontierTarget = objective.targetSector;
                frontierSelectionCount = 1;
            }
        }
        char detail[192];
        snprintf(detail, sizeof(detail), "type=%d id=%d sector=%d target_sector=%d wall=%d current=%d",
                 int(objective.type), objective.id, objective.sector,
                 objective.targetSector, objective.wall, observation.sector);
        event("objective_selected", detail);
    }

    void completeObjective(const char *reason)
    {
        if (!currentObjective.active)
            return;
        if (currentObjective.type == kObjectiveFrontier)
            retirePendingFrontier(currentObjective.wall, currentObjective.sector,
                                  currentObjective.targetSector);
        char detail[128];
        snprintf(detail, sizeof(detail), "type=%d id=%d wall=%d source=%d target=%d reason=%s",
                 int(currentObjective.type), currentObjective.id, currentObjective.wall,
                 currentObjective.sector, currentObjective.targetSector, reason);
        event("objective_completed", detail);
        objectiveProgressTick = -1;
        lastSemanticProgressTick = observation.tick;
        currentObjective = Objective{};
        movementTargetActive = false;
        resetNavigation();
    }

    // Every abandonment makes the opportunity dormant.  Without this an
    // objective that fails is simply reselected on the next tick, which is
    // the mechanism behind most of the bot's tight local loops.  Knowledge
    // is kept; only the bot's attention moves on.
    void invalidateObjective(const char *reason, bool dormant = true)
    {
        if (!currentObjective.active)
            return;
        if (currentObjective.type == kObjectiveExpose)
        {
            // Give up on this particular viewpoint, not on looking around.
            // Retiring the unreachable square lets the next attempt pick a
            // different one instead of circling the same pocket.
            const int64_t handle = navCellHandle(currentObjective.sector,
                                                 currentObjective.x >> kNavGridShift,
                                                 currentObjective.y >> kNavGridShift);
            observedCells.insert(handle);
            char detail[160];
            snprintf(detail, sizeof(detail), "sector=%d at=(%d,%d) reason=%s",
                     currentObjective.sector, currentObjective.x, currentObjective.y, reason);
            event("coverage_target_unreachable", detail);
            suppressObjective(7000000 + currentObjective.sector, "coverage_target_unreachable");
            currentObjective = Objective{};
            objectiveProgressTick = -1;
            movementTargetActive = false;
            resetNavigation();
            return;
        }
        if (dormant)
            suppressObjective(objectiveKey(currentObjective), reason);
        if (currentObjective.type == kObjectiveFrontier)
        {
            char stepDetail[192];
            snprintf(stepDetail, sizeof(stepDetail),
                     "step_wall=%d step_from=%d step_to=%d dest=%d remains_active=0 reason=%s",
                     currentObjective.routeStepWall, currentObjective.routeStepFrom,
                     currentObjective.routeStepTo, currentObjective.targetSector, reason);
            event("route_step_failed", stepDetail);
        }
        char detail[128];
        snprintf(detail, sizeof(detail), "type=%d id=%d wall=%d source=%d target=%d reason=%s",
                 int(currentObjective.type), currentObjective.id, currentObjective.wall,
                 currentObjective.sector, currentObjective.targetSector, reason);
        event("objective_invalidated", detail);
        objectiveProgressTick = -1;
        currentObjective = Objective{};
        movementTargetActive = false;
        resetNavigation();
        currentGoal.clear();
        currentGoalTarget = -1;
    }

    bool urgentThreat()
    {
        if (lastDamageTick >= 0 && observation.tick - lastDamageTick <= 2 * kTicsPerSec)
            return true;
        const VisibleObject *enemy = selectObject(kObjectEnemy);
        return enemy && !unreachableEnemies.count(enemy->sprite)
            && enemy->sector == observation.sector
            && distance2(enemy->x, enemy->y, observation.x, observation.y) <= 1024 * 1024
            && directObjectReachable(*enemy);
    }

    GINPUT retreatFromEnemy(const VisibleObject &enemy)
    {
        ensureNavTopology();
        const int playerCell = nearestNavCell(observation.sector, observation.x, observation.y);
        if (playerCell < 0 || navCells[playerCell].walkArea < 0)
            return GINPUT{};
        const int area = navCells[playerCell].walkArea;
        const int currentEnemyDistance = distance2(observation.x, observation.y,
                                                   enemy.x, enemy.y);
        int bestCell = -1;
        int bestEnemyDistance = currentEnemyDistance;
        for (const NavCell &cell : navCells)
        {
            if (cell.walkArea != area)
                continue;
            const MovementProbe probe = probeMovement(
                observation.x, observation.y, observation.z, observation.sector,
                cell.center.x, cell.center.y, cell.sector, 512);
            if (!probe.reachable)
                continue;
            const int enemyDistance = distance2(cell.center.x, cell.center.y,
                                                enemy.x, enemy.y);
            if (enemyDistance > bestEnemyDistance)
            {
                bestEnemyDistance = enemyDistance;
                bestCell = cell.id;
            }
        }
        if (bestCell < 0)
        {
            // Raw opposite-point movement can send the player through a wall.
            // A retreat is valid only when the same validated local topology
            // proves the destination reachable.
            if (!combatRetreatUnavailableEmitted)
            {
                combatRetreatUnavailableEmitted = true;
                event("combat_retreat_unavailable", "reason=no_validated_local_cell");
            }
            return GINPUT{};
        }
        combatRetreatUnavailableEmitted = false;
        if (bestCell != lastCombatRetreatCell)
        {
            lastCombatRetreatCell = bestCell;
            char detail[160];
            snprintf(detail, sizeof(detail), "enemy=%d cell=%d distance=%d->%d",
                     enemy.sprite, bestCell,
                     int(std::sqrt(double(currentEnemyDistance))),
                     int(std::sqrt(double(bestEnemyDistance))));
            event("combat_retreat", detail);
        }
        setGoal("COMBAT_RETREAT", enemy.sprite);
        const NavCell &cell = navCells[bestCell];
        return navigateTo(cell.center.x, cell.center.y, observation.z,
                          cell.sector, -1, kTraversalUnknown);
    }

    bool meleeWeaponAvailable() const
    {
        return gMe && gMe->hasWeapon[kWeaponPitchfork];
    }

    GINPUT meleeAttack(const VisibleObject &enemy)
    {
        GINPUT input = {};
        input.syncFlags.run = 1;
        input.q16turn = fix16_from_int(angleDelta(
            getangle(enemy.x - observation.x, enemy.y - observation.y), observation.angle));
        if (!meleeWeaponAvailable())
            return input;
        if (gMe->curWeapon != kWeaponPitchfork)
        {
            input.syncFlags.weaponChange = 1;
            input.newWeapon = uint8_t(kWeaponPitchfork);
            event("melee_weapon_selected", "weapon=pitchfork");
            return input;
        }
        input.buttonFlags.shoot = 1;
        return input;
    }

    bool interactionStillBusy(const InteractionMemory &memory) const
    {
        if (memory.target.wall >= 0 && inRange(memory.target.wall, 0, numwalls))
        {
            const int extra = wall[memory.target.wall].extra;
            if (extra > 0 && extra < kMaxXWalls && xwall[extra].busy != 0)
                return true;
        }
        const int sectorId = memory.target.sectorPush ? memory.target.to
            : memory.target.sectorPushCurrent ? memory.target.from : -1;
        if (inRange(sectorId, 0, numsectors))
        {
            const int extra = sector[sectorId].extra;
            if (extra > 0 && extra < kMaxXSectors && xsector[extra].busy != 0)
                return true;
        }
        return false;
    }

    int playerClipRadius() const
    {
        return gMe && gMe->pSprite ? (gMe->pSprite->clipdist << 2) : 128;
    }

    int currentMoveWallHit() const
    {
        if (!gMe || !gMe->pSprite || gMe->pSprite->extra <= 0 || gMe->pSprite->extra >= kMaxXSprites)
            return -1;
        const int moveHit = gSpriteHit[gMe->pSprite->extra].hit;
        if ((moveHit & 0xc000) == 0x8000)
            return moveHit & 0x3fff;
        return -1;
    }

    bool wallsShareVertex(int a, int b) const
    {
        if (!inRange(a, 0, numwalls) || !inRange(b, 0, numwalls))
            return false;
        const int ax1 = wall[a].x;
        const int ay1 = wall[a].y;
        const int ax2 = wall[wall[a].point2].x;
        const int ay2 = wall[wall[a].point2].y;
        const int bx1 = wall[b].x;
        const int by1 = wall[b].y;
        const int bx2 = wall[wall[b].point2].x;
        const int by2 = wall[wall[b].point2].y;
        return (ax1 == bx1 && ay1 == by1) || (ax1 == bx2 && ay1 == by2)
            || (ax2 == bx1 && ay2 == by1) || (ax2 == bx2 && ay2 == by2);
    }

    int64_t dist2ToSegment(int x, int y, int x1, int y1, int x2, int y2) const
    {
        return segmentDistance2(x, y, x1, y1, x2, y2);
    }

    bool sourceInspectionPose(const Portal &portal, int &outX, int &outY, int &outZ)
    {
        const int mx = (portal.x1 + portal.x2) / 2;
        const int my = (portal.y1 + portal.y2) / 2;
        const int dx = portal.x2 - portal.x1;
        const int dy = portal.y2 - portal.y1;
        const double length = std::sqrt(double(int64_t(dx) * dx + int64_t(dy) * dy));
        const int inset = playerClipRadius() + 96;
        if (length < 1.0)
        {
            outX = mx;
            outY = my;
            outZ = inRange(portal.from, 0, numsectors) ? getflorzofslope(portal.from, mx, my) : observation.z;
            return false;
        }
        const int nx = int(std::lround(-double(dy) * inset / length));
        const int ny = int(std::lround(double(dx) * inset / length));
        const int candidates[2][2] = {{mx + nx, my + ny}, {mx - nx, my - ny}};
        int bestX = candidates[0][0];
        int bestY = candidates[0][1];
        bool found = false;
        int bestPlayerDistance = INT32_MAX;
        for (int i = 0; i < 2; ++i)
        {
            int x = candidates[i][0];
            int y = candidates[i][1];
            int16_t sect = int16_t(portal.from);
            updatesector(x, y, &sect);
            if (int(sect) != portal.from)
                continue;
            found = true;
            const int distance = distance2(observation.x, observation.y, x, y);
            if (distance < bestPlayerDistance)
            {
                bestPlayerDistance = distance;
                bestX = x;
                bestY = y;
            }
        }
        outX = bestX;
        outY = bestY;
        outZ = inRange(portal.from, 0, numsectors) ? getflorzofslope(portal.from, bestX, bestY) : observation.z;
        return found;
    }

    int wallFacingAngle(const Portal &portal) const
    {
        const int tangentX = portal.x2 - portal.x1;
        const int tangentY = portal.y2 - portal.y1;
        const int normal1X = -tangentY;
        const int normal1Y = tangentX;
        const int toX = portal.x - observation.x;
        const int toY = portal.y - observation.y;
        const int64_t dot = int64_t(normal1X) * toX + int64_t(normal1Y) * toY;
        if (dot >= 0)
            return getangle(normal1X, normal1Y);
        return getangle(-normal1X, -normal1Y);
    }

    bool discoverNearbyTriggerPush(const Portal &crossing)
    {
        if (!inRange(observation.sector, 0, numsectors))
            return false;
        bool found = false;
        const sectortype &record = sector[observation.sector];
        const int reach = std::max(768, playerClipRadius() * 4);
        for (int i = 0; i < record.wallnum; ++i)
        {
            const int wallIndex = record.wallptr + i;
            if (wallIndex != crossing.wall && !wallsShareVertex(wallIndex, crossing.wall)
                && dist2ToSegment(observation.x, observation.y, wall[wallIndex].x, wall[wallIndex].y,
                                  wall[wall[wallIndex].point2].x, wall[wall[wallIndex].point2].y)
                    > int64_t(reach) * reach)
                continue;
            if (rememberCollisionInteraction(wallIndex))
                found = true;
        }
        if (inRange(crossing.to, 0, numsectors))
        {
            const sectortype &adjacent = sector[crossing.to];
            for (int i = 0; i < adjacent.wallnum; ++i)
            {
                const int wallIndex = adjacent.wallptr + i;
                const walltype &adjacentWall = wall[wallIndex];
                if (inRange(adjacentWall.nextsector, 0, numsectors))
                    continue;
                if (!wallsShareVertex(wallIndex, crossing.wall)
                    && dist2ToSegment(observation.x, observation.y, adjacentWall.x, adjacentWall.y,
                                      wall[adjacentWall.point2].x, wall[adjacentWall.point2].y)
                        > int64_t(reach) * reach)
                    continue;
                if (rememberCollisionInteraction(wallIndex))
                    found = true;
            }
        }
        return found;
    }

    GINPUT finishBlockedInvestigation(const Portal &crossing, const char *reason)
    {
        const int targetAngle = wallFacingAngle(crossing);
        GINPUT input = {};
        input.q16turn = fix16_from_int(angleDelta(targetAngle, observation.angle));
        const int look = lookAngleForTarget(observation.z, crossing.z,
            std::max(1, int(std::sqrt(double(distance2(observation.x, observation.y,
                                                      crossing.x, crossing.y))))));
        input.q16mlook = encodeLook(look - fix16_to_int(gMe->q16look));
        if (crossing.crouchable || crossing.interactionCrouch)
            input.buttonFlags.crouch = 1;
        if (std::abs(angleDelta(targetAngle, observation.angle)) > 96
            && observation.tick - currentObjective.startedTick < 2 * kTicsPerSec
            && strcmp(reason, "investigation_timeout") != 0)
            return input;

        int hit = -1, target = -1, extra = -1;
        hit = ActionScanPreview(gMe, &target, &extra);
        const bool nearby = discoverNearbyTriggerPush(crossing);
        if (hit >= 0)
        {
            char detail[160];
            snprintf(detail, sizeof(detail), "hit=%d target=%d wall=%d dest=%d reason=%s",
                     hit, target, crossing.wall, currentObjective.targetSector, reason);
            event("blocked_frontier_action_found", detail);
            const bool discovered = rememberCollisionInteraction(
                target >= 0 ? target : crossing.wall);
            if (!discovered)
            {
                // Already known and already tried.  Re-investigating it is
                // not new knowledge, so let the boundary rest rather than
                // rediscovering the same mechanism every tick.
                suppressObjective(4000000 + currentObjective.targetSector,
                                  "mechanism_already_known");
            }
            completeObjective("investigation_found_interaction");
            return input;
        }
        if (nearby || selectNewInteraction(false))
        {
            event("blocked_frontier_action_found", "source=nearby_trigger_push");
            completeObjective("investigation_found_interaction");
            return input;
        }
        InvestigateRecord record;
        record.wall = crossing.wall;
        record.from = crossing.from;
        record.to = crossing.to;
        record.geometrySignature = portalIdentitySignature(crossing);
        investigatedBoundaries.push_back(record);
        char investigated[160];
        snprintf(investigated, sizeof(investigated), "wall=%d dest=%d hit=%d reason=%s",
                 crossing.wall, currentObjective.targetSector, hit, reason);
        event("blocked_frontier_investigated", investigated);
        suppressObjective(4000000 + currentObjective.targetSector, "investigated_no_action");
        completeObjective("investigation_no_action");
        return input;
    }

    GINPUT executeObjective()
    {
        if (!currentObjective.active)
            return GINPUT{};
        if (currentObjective.type == kObjectiveInteraction)
        {
            auto memory = interactions.find(currentObjective.interactionKey);
            if (memory == interactions.end() || !memory->second.observed)
            {
                invalidateObjective("interaction_not_known", false);
                return GINPUT{};
            }
            InteractionMemory &interaction = memory->second;
            if (interaction.state != 1 && interactionNeedsReactivation(interaction))
            {
                interaction.state = 0;
                interaction.attempted = false;
                interaction.activated = false;
                interaction.engineAccepted = false;
                interaction.unavailableAttempts = 0;
                interaction.unavailableFromSector = -1;
                char detail[160];
                snprintf(detail, sizeof(detail),
                         "kind=%d id=%d from=%d to=%d reason=known_route_closed_again",
                         int(interaction.kind), interaction.id, interaction.fromSector,
                         interaction.targetSector);
                event("interaction_rearmed", detail);
            }
            if (interaction.engineAccepted && interaction.state == 2
                && interactionStillBusy(interaction)
                && observation.tick - interaction.lastActivationTick < 2 * kTicsPerSec)
            {
                const int key = interactionMemoryKey(interaction);
                if (lastInteractionWaitKey != key)
                {
                    lastInteractionWaitKey = key;
                    event("interaction_waiting_for_settle", "engine_accepted=1 source_busy=1");
                }
                return GINPUT{};
            }
            if (interaction.engineAccepted && interaction.state == 2)
            {
                // Stay for the result.  Walking off the moment the engine
                // accepted the Use meant the bot was elsewhere by the time
                // the geometry finished moving, so it never observed the
                // route it had just created and never used it.  Re-pressing
                // is already blocked while the mechanism moves, so holding
                // here is safe; the engine's own remaining travel time
                // bounds the wait.
                const DoorTiming opening = doorTiming(interaction.targetSector);
                const bool watchWorthwhile = opening.moving
                    || (interaction.targetSector >= 0
                        && !visitedSectors.count(interaction.targetSector)
                        && observation.tick - interaction.lastActivationTick
                            < kMechanismSettleTicks);
                if (watchWorthwhile && opening.remainingTicks <= kMaxDoorWaitTicks)
                {
                    const int key = interactionMemoryKey(interaction);
                    if (lastInteractionWaitKey != key)
                    {
                        lastInteractionWaitKey = key;
                        char detail[192];
                        snprintf(detail, sizeof(detail),
                                 "kind=%d id=%d target_sector=%d remaining_s=%d hold_s=%d",
                                 int(interaction.kind), interaction.id,
                                 interaction.targetSector,
                                 opening.remainingTicks / kTicsPerSec,
                                 opening.holdTicks / kTicsPerSec);
                        event("interaction_waiting_for_settle", detail);
                    }
                    // Wait *at* the opening, not wherever the bot happens to
                    // be, so it can step through the moment there is room.
                    const Portal *gap = portalByWall(interaction.target.wall,
                                                     interaction.fromSector,
                                                     interaction.targetSector);
                    if (gap)
                        return steerPortal(*gap);
                    return GINPUT{};
                }
                lastInteractionWaitKey = -1;
                completeObjective("engine_accepted_and_settled");
                return GINPUT{};
            }
            if (interaction.state == 3)
            {
                invalidateObjective("interaction_rejected");
                return GINPUT{};
            }
            // A vector activation has no Use for the engine to accept, so it
            // reports its own success: the world answering is the whole of
            // the evidence there is.
            if (interaction.target.shootable
                && (interaction.state == 2 || interaction.observedKnownWorldDelta))
            {
                completeObjective("vector_activation_answered");
                return GINPUT{};
            }
            return steerInteraction(interaction);
        }
        if (currentObjective.type == kObjectiveFrontier
            || currentObjective.type == kObjectiveInvestigate)
        {
            // Arriving means standing there, not having stood there once.
            // Treating any previously visited destination as already reached
            // made every deliberate re-crossing -- backtracking, or walking
            // back through a door the bot had just reopened -- complete
            // instantly without the bot moving a step.
            if (observation.sector == currentObjective.targetSector
                && currentObjective.type == kObjectiveFrontier)
            {
                completeObjective("destination_entered");
                return GINPUT{};
            }
            const bool exactCrossing = observation.sector == currentObjective.targetSector
                && lastTransitionTick == observation.tick;
            if (exactCrossing && currentObjective.type == kObjectiveFrontier)
            {
                completeObjective("sector_transition");
                return GINPUT{};
            }
            const Portal *crossing = portalByWall(currentObjective.wall, currentObjective.sector,
                                                  currentObjective.targetSector);
            if (!crossing)
            {
                auto known = knownGraph.find(currentObjective.sector);
                if (known != knownGraph.end())
                    for (const Portal &portal : known->second)
                        if (portal.to == currentObjective.targetSector)
                        {
                            crossing = &portal;
                            break;
                        }
            }
            if (!crossing)
            {
                invalidateObjective("frontier_crossing_unknown");
                return GINPUT{};
            }
            if (currentObjective.type == kObjectiveInvestigate)
            {
                if (!currentObjective.inspectPoseValid)
                {
                    currentObjective.inspectPoseValid = sourceInspectionPose(
                        *crossing, currentObjective.inspectX, currentObjective.inspectY,
                        currentObjective.inspectZ);
                    char pose[160];
                    snprintf(pose, sizeof(pose), "wall=%d pose=(%d,%d) valid=%d",
                             crossing->wall, currentObjective.inspectX, currentObjective.inspectY,
                             currentObjective.inspectPoseValid ? 1 : 0);
                    event("blocked_frontier_inspect_pose", pose);
                }
                const int hitWall = currentMoveWallHit();
                const int arrive = std::max(160, playerClipRadius() + 32);
                const bool atPose = distance2(observation.x, observation.y,
                                              currentObjective.inspectX, currentObjective.inspectY)
                    <= arrive * arrive;
                const bool nearWall = dist2ToSegment(observation.x, observation.y,
                                                     crossing->x1, crossing->y1,
                                                     crossing->x2, crossing->y2)
                    <= int64_t(playerClipRadius() + 48) * (playerClipRadius() + 48);
                const bool hitting = hitWall == crossing->wall
                    || wallsShareVertex(hitWall, crossing->wall);
                const bool timedOut = currentObjective.startedTick >= 0
                    && observation.tick - currentObjective.startedTick >= 2 * kTicsPerSec;
                if (observation.sector == crossing->from
                    && (atPose || nearWall || hitting || timedOut))
                    return finishBlockedInvestigation(*crossing,
                        timedOut && !atPose && !nearWall && !hitting
                            ? "investigation_timeout" : "investigation_pose");
                return navigateTo(currentObjective.inspectX, currentObjective.inspectY,
                                  currentObjective.inspectZ, crossing->from,
                                  crossing->wall, kTraversalUnknown);
            }
            currentObjective.routeStepWall = crossing->wall;
            currentObjective.routeStepFrom = observation.sector;
            currentObjective.routeStepTo = crossing->from == observation.sector
                ? crossing->to : crossing->from;
            // Which of the two ways of reaching a frontier is in use, and
            // why.  A crossing that silently falls back to the planner is
            // how the bot came to stand still in front of an open door.
            const bool onNearSide = observation.sector == crossing->from;
            const bool passable = crossing->traversable || crossing->jumpable;
            const int choice = crossing->wall * 8 + (onNearSide ? 1 : 0)
                + (passable ? 2 : 0) + 4;
            if (choice != lastFrontierChoice)
            {
                lastFrontierChoice = choice;
                char detail[224];
                snprintf(detail, sizeof(detail),
                         "wall=%d want_from=%d portal_from=%d here=%d to=%d "
                         "traversable=%d jumpable=%d clearance=%d cap=%d path=%s",
                         crossing->wall, currentObjective.sector, crossing->from,
                         observation.sector, crossing->to,
                         crossing->traversable ? 1 : 0, crossing->jumpable ? 1 : 0,
                         crossing->clearance, int(crossing->capability),
                         (onNearSide && passable) ? "portal" : "planner");
                event("frontier_route_choice", detail);
            }
            if (onNearSide && passable
                && currentObjective.type == kObjectiveFrontier)
                return steerPortal(*crossing);
            return navigateTo(crossing->x, crossing->y, crossing->z, crossing->from,
                              crossing->wall, crossing->capability);
        }
        if (currentObjective.type == kObjectiveExpose)
        {
            // Done as soon as the target square has actually been seen, or
            // the sector has nothing left unobserved -- standing on it is
            // not required, looking at it is.
            int cellX = 0;
            int cellY = 0;
            int unobserved = 0;
            if (!findUnobservedCell(currentObjective.sector, cellX, cellY, unobserved))
            {
                char detail[96];
                snprintf(detail, sizeof(detail), "sector=%d", currentObjective.sector);
                event("local_coverage_complete", detail);
                completeObjective("local_space_observed");
                return GINPUT{};
            }
            if (cellX != currentObjective.x || cellY != currentObjective.y)
            {
                currentObjective.x = cellX;
                currentObjective.y = cellY;
            }
            return navigateTo(currentObjective.x, currentObjective.y, observation.z,
                              currentObjective.sector, -3, kTraversalUnknown);
        }
        if (currentObjective.type == kObjectivePickup
            || currentObjective.type == kObjectiveKey)
        {
            const ObjectKind kind = currentObjective.type == kObjectiveKey
                ? kObjectKey : kObjectPickup;
            const VisibleObject *object = nullptr;
            for (const VisibleObject &candidate : observation.objects)
            {
                if (candidate.sprite == currentObjective.id && candidate.kind == kind)
                {
                    object = &candidate;
                    break;
                }
            }
            if (!object)
            {
                auto remembered = objectMemory.find(currentObjective.id);
                if (remembered == objectMemory.end()
                    || !remembered->second.observed
                    || remembered->second.collected)
                {
                    completeObjective("pickup_confirmed_collected");
                    return GINPUT{};
                }
                if (!objectStillPresent(remembered->second))
                {
                    remembered->second.collected = true;
                    event("pickup_confirmed_collected", "source=sprite_removed_or_inventory");
                    completeObjective("pickup_confirmed_collected");
                    return GINPUT{};
                }
                // The observation is deliberately non-omniscient.  Keep
                // pursuing the last known pose while a corner or moving
                // wall temporarily occludes the pickup.
                selectedObject = remembered->second.object;
                object = &selectedObject;
            }
            if (object->sector != observation.sector)
            {
                return navigateTo(object->x, object->y, object->z, object->sector,
                                  object->sprite, kTraversalUnknown);
            }
            return navigateTo(object->x, object->y, object->z, object->sector,
                              object->sprite, kTraversalUnknown);
        }
        return GINPUT{};
    }

    void setMovementTarget(int x, int y, int targetSector, int targetId,
                           TraversalCapability capability = kTraversalUnknown)
    {
        if (movementTargetActive && movementTargetGoal == currentGoal
            && movementTargetX == x && movementTargetY == y
            && movementTargetSector == targetSector && movementTargetId == targetId
            && movementTargetCapability == capability)
            return;

        movementTargetActive = true;
        movementTargetX = x;
        movementTargetY = y;
        movementTargetSector = targetSector;
        movementTargetId = targetId;
        movementTargetFrom = observation.sector;
        movementTargetCapability = capability;
        movementTargetGoal = currentGoal;
        targetLastX = observation.x;
        targetLastY = observation.y;
        targetLastZ = observation.z;
        targetLastDistance2 = distance2(observation.x, observation.y, x, y);
        targetBestDistance2 = targetLastDistance2;
        targetLastProgressTick = observation.tick;
        jumpCooldownTick = observation.tick;
        jumpAttempts = 0;
        jumpState = kJumpInactive;
        jumpStateTick = -1;
        jumpTargetX = x;
        jumpTargetY = y;
        jumpTargetSector = targetSector;
        jumpSourceSector = observation.sector;
    }

    void updateMovementProgress()
    {
        if (!movementTargetActive)
            return;

        const int currentDistance2 = distance2(observation.x, observation.y,
                                               movementTargetX, movementTargetY);
        // Compare distances, not squares of distances.  A fixed slack on d^2
        // is a different amount of ground at every range: eighty units out
        // it means real movement, eight thousand units out a single step
        // clears it, so almost anything counted as closing on the target and
        // the stuck detector never fired.
        const int currentDistance = int(std::sqrt(double(currentDistance2)));
        const int bestDistance = int(std::sqrt(double(targetBestDistance2)));
        // Closing on the target: strictly nearer than the bot has yet been.
        const bool horizontalProgress = bestDistance - currentDistance >= kProgressStep;
        const bool verticalProgress = std::abs(observation.z - targetLastZ) >= 256;
        // Arriving in the target sector is progress.  This read the other way
        // round, so being anywhere else counted as having got somewhere --
        // which reset the jump budget after every failed hop.
        const bool sectorProgress = movementTargetSector >= 0
            && observation.sector == movementTargetSector;
        // Walking is not the same as closing, and the two answer different
        // questions.  The stuck detector asks whether the bot is going
        // anywhere at all -- a detour round a corner runs away from the
        // target for a while and is still perfectly good movement -- while
        // the jump budget and the record distance ask whether it is getting
        // nearer.  Conflating them either strands the bot on every concave
        // room or lets it bounce in place for ever.
        const int stepped = int(std::sqrt(double(distance2(
            observation.x, observation.y, targetLastX, targetLastY))));
        const bool moving = stepped >= kProgressStep;
        if (horizontalProgress || sectorProgress)
        {
            if (jumpAttempts > 0)
            {
                char detail[128];
                snprintf(detail, sizeof(detail),
                         "target=%d dx=%d dy=%d dz=%d attempts=%d reason=%s",
                         movementTargetId, observation.x - targetLastX, observation.y - targetLastY,
                         observation.z - targetLastZ, jumpAttempts,
                         horizontalProgress ? "closed_on_target" : "entered_target_sector");
                event("jump_succeeded", detail);
                jumpAttempts = 0;
            }
            targetBestDistance2 = std::min(targetBestDistance2, currentDistance2);
        }
        else if (jumpAttempts > 0 && verticalProgress)
        {
            // Height alone is not traversal.  Report it, but do not let it
            // buy another jump.
            char detail[128];
            snprintf(detail, sizeof(detail), "target=%d dx=%d dy=%d dz=%d attempts=%d",
                     movementTargetId, observation.x - targetLastX, observation.y - targetLastY,
                     observation.z - targetLastZ, jumpAttempts);
            event("jump_progress", detail);
        }
        if (horizontalProgress || verticalProgress || moving)
        {
            targetLastX = observation.x;
            targetLastY = observation.y;
            targetLastZ = observation.z;
            targetLastDistance2 = currentDistance2;
            targetLastProgressTick = observation.tick;
        }
    }

    bool movementNeedsJump() const
    {
        return movementTargetActive && jumpAttempts < kMaxJumpAttemptsPerTarget
            && movementTargetCapability == kTraversalJumpable
            && observation.tick >= jumpCooldownTick
            && observation.tick - targetLastProgressTick >= kMovementStuckTicks
            && distance2(observation.x, observation.y, movementTargetX, movementTargetY) > 4096;
    }

    MovementProbe probeMovement(int startX, int startY, int startZ, int startSector,
                                int targetX, int targetY, int targetSector, int tolerance) const
    {
        MovementProbe result;
        result.x = startX;
        result.y = startY;
        result.sector = startSector;
        if (!inRange(startSector, 0, numsectors) || !inRange(targetSector, 0, numsectors))
            return result;
        vec3_t position = { startX, startY, startZ };
        int16_t sectorNumber = int16_t(startSector);
        const int64_t dx = int64_t(targetX) - startX;
        const int64_t dy = int64_t(targetY) - startY;
        const int32_t xvect = int32_t(std::max<int64_t>(INT32_MIN + 1, std::min<int64_t>(INT32_MAX, dx << 14)));
        const int32_t yvect = int32_t(std::max<int64_t>(INT32_MIN + 1, std::min<int64_t>(INT32_MAX, dy << 14)));
        const int radius = gMe && gMe->pSprite ? (gMe->pSprite->clipdist << 2) : 128;
        int ceilingDistance = 0;
        int floorDistance = 0;
        playerCollisionDistances(ceilingDistance, floorDistance);
        result.hit = clipmove(&position, &sectorNumber, xvect, yvect, radius,
                              ceilingDistance, floorDistance, CLIPMASK0);
        result.x = position.x;
        result.y = position.y;
        result.sector = sectorNumber;
        if ((result.hit & 0xc000) == 0x8000)
            result.wall = result.hit & 0x3fff;
        const int remaining = distance2(position.x, position.y, targetX, targetY);
        result.reachable = sectorNumber == targetSector && remaining <= tolerance * tolerance;
        return result;
    }

    static int64_t navCross(const LocalWaypoint &a, const LocalWaypoint &b,
                            const LocalWaypoint &c)
    {
        return int64_t(b.x - a.x) * (c.y - a.y)
            - int64_t(b.y - a.y) * (c.x - a.x);
    }

    static bool navPointInTriangle(const LocalWaypoint &p, const LocalWaypoint *v)
    {
        const int64_t a = navCross(v[0], v[1], p);
        const int64_t b = navCross(v[1], v[2], p);
        const int64_t c = navCross(v[2], v[0], p);
        return (a >= 0 && b >= 0 && c >= 0)
            || (a <= 0 && b <= 0 && c <= 0);
    }

    int navTopologyIdentity() const
    {
        int signature = 17;
        for (int sectorId : visitedSectors)
        {
            if (!inRange(sectorId, 0, numsectors))
                continue;
            const sectortype &sectorRecord = sector[sectorId];
            signature = llmapper::mixHash(signature, sectorId);
            signature = llmapper::mixHash(signature, sectorRecord.wallptr);
            signature = llmapper::mixHash(signature, sectorRecord.wallnum);
            for (int i = 0; i < sectorRecord.wallnum; ++i)
            {
                const int wallId = sectorRecord.wallptr + i;
                if (!inRange(wallId, 0, numwalls))
                    continue;
                signature = llmapper::mixHash(signature, wallId);
                signature = llmapper::mixHash(signature, wall[wallId].point2);
                signature = llmapper::mixHash(signature, wall[wallId].nextsector);
            }
        }
        return signature;
    }

    int navPoseSignature() const
    {
        int signature = 19;
        for (int sectorId : visitedSectors)
        {
            if (!inRange(sectorId, 0, numsectors))
                continue;
            const sectortype &sectorRecord = sector[sectorId];
            signature = llmapper::mixHash(signature, sectorId);
            signature = llmapper::mixHash(signature, sectorRecord.floorz);
            signature = llmapper::mixHash(signature, sectorRecord.ceilingz);
            for (int i = 0; i < sectorRecord.wallnum; ++i)
            {
                const int wallId = sectorRecord.wallptr + i;
                if (!inRange(wallId, 0, numwalls))
                    continue;
                signature = llmapper::mixHash(signature, wall[wallId].x);
                signature = llmapper::mixHash(signature, wall[wallId].y);
            }
        }
        return signature;
    }

    int navGeometrySignature() const
    {
        return navTopologyIdentity();
    }

    int navDynamicSignature() const
    {
        int signature = 23;
        std::set<int> sectors = observedSectors;
        if (inRange(observation.sector, 0, numsectors))
            sectors.insert(observation.sector);
        for (int sectorId : sectors)
        {
            if (!inRange(sectorId, 0, numsectors))
                continue;
            const sectortype &record = sector[sectorId];
            signature = signature * 31 + sectorId;
            signature = signature * 31 + record.floorz;
            signature = signature * 31 + record.ceilingz;
            if (record.extra > 0 && record.extra < kMaxXSectors)
            {
                signature = signature * 31 + xsector[record.extra].state;
                signature = signature * 31 + xsector[record.extra].busy;
            }
            for (int i = 0; i < record.wallnum; ++i)
            {
                const int wallId = record.wallptr + i;
                if (!inRange(wallId, 0, numwalls))
                    continue;
                const walltype &wallRecord = wall[wallId];
                // Wall positions are part of the topology.  Blood builds
                // doors out of walls that physically slide apart, so a mesh
                // that ignores where the walls are cannot notice that a
                // passage just appeared -- the bot activates the mechanism,
                // watches the world change, and still believes it is walled in.
                signature = signature * 31 + wallRecord.x;
                signature = signature * 31 + wallRecord.y;
                signature = signature * 31 + wallRecord.nextsector;
                signature = signature * 31 + wallRecord.cstat;
                if (wallRecord.extra > 0 && wallRecord.extra < kMaxXWalls)
                {
                    signature = signature * 31 + xwall[wallRecord.extra].state;
                    signature = signature * 31 + xwall[wallRecord.extra].busy;
                }
            }
        }
        return signature;
    }

    void addNavLink(int from, int to, NavEdgeMode mode, int wallId,
                    LocalWaypoint gateway = {}, bool hasGateway = false)
    {
        if (from < 0 || to < 0 || from == to)
            return;
        NavCell &cell = navCells[from];
        for (const NavLink &link : cell.links)
            if (link.target == to && link.wall == wallId)
                return;
        NavLink link;
        link.target = to;
        link.mode = mode;
        link.wall = wallId;
        link.gateway = gateway;
        link.hasGateway = hasGateway;
        cell.links.push_back(link);
    }

    void refreshDynamicNavLinks()
    {
        int linked = 0;
        int rejected = 0;
        // Triangle connectivity is planar and remains cached.  Only links
        // whose capability depends on current sector/wall state are rebuilt.
        for (NavCell &cell : navCells)
            cell.links.erase(std::remove_if(cell.links.begin(), cell.links.end(),
                                            [](const NavLink &link)
                                            { return link.wall >= 0; }), cell.links.end());
        for (const auto &entry : knownGraph)
            for (const Portal &portal : entry.second)
            {
                if (!(portal.traversable || portal.jumpable)
                    || !visitedSectors.count(portal.from)
                    || !visitedSectors.count(portal.to))
                {
                    ++rejected;
                    continue;
                }
                const NavEdgeMode mode = navModeForPortal(portal);
                if (!llmapper::traversableMode(mode))
                {
                    ++rejected;
                    continue;
                }
                const int from = nearestNavCell(portal.from, portal.x, portal.y);
                const int to = nearestNavCell(portal.to, portal.x, portal.y);
                if (from < 0 || to < 0)
                    continue;
                addNavLink(from, to, mode, portal.wall,
                           LocalWaypoint(portal.x, portal.y), true);
                // Visited space is the bot's transport network.  A crossing
                // it already walked must be usable in both directions, or
                // deliberate backtracking is impossible.  A one-way drop
                // stays one-way.
                if (mode != kNavDrop)
                    addNavLink(to, from, mode == kNavJump ? kNavWalk : mode, portal.wall,
                               LocalWaypoint(portal.x, portal.y), true);
                ++linked;
            }
        llmapper::assignWalkAreas(navCells);
        char detail[96];
        snprintf(detail, sizeof(detail), "linked=%d rejected=%d", linked, rejected);
        event("nav_portal_links", detail);
    }

    static int64_t navCellHandle(int sectorId, int gx, int gy)
    {
        return (int64_t(sectorId) << 42) ^ (int64_t(gx & 0xfffff) << 21)
            ^ int64_t(gy & 0xfffff);
    }

    int navCellAt(int sectorId, int gx, int gy) const
    {
        auto entry = navCellIndex.find(navCellHandle(sectorId, gx, gy));
        return entry == navCellIndex.end() ? -1 : entry->second;
    }

    // Strict: only ever returns a cell of the sector asked for.
    int navCellInSector(int sectorId, int x, int y) const
    {
        const int exact = navCellAt(sectorId, x >> kNavGridShift, y >> kNavGridShift);
        if (exact >= 0)
            return exact;
        int best = -1;
        int bestDistance = INT32_MAX;
        for (const NavCell &cell : navCells)
        {
            if (cell.sector != sectorId)
                continue;
            const int currentDistance = distance2(x, y, cell.center.x, cell.center.y);
            if (currentDistance < bestDistance)
            {
                bestDistance = currentDistance;
                best = cell.id;
            }
        }
        return best;
    }

    // Can the player physically occupy this sector at all?  Answered from
    // the same body envelope the grid is built with, so a slot narrower than
    // Caleb is simply not somewhere he can go.
    bool sectorHasStandableSpace(int sectorId)
    {
        if (!inRange(sectorId, 0, numsectors))
            return false;
        auto grid = sectorGrids.find(sectorId);
        if (grid == sectorGrids.end()
            || grid->second.signature != sectorGeometrySignature(sectorId))
        {
            buildNavSector(sectorId);
            grid = sectorGrids.find(sectorId);
        }
        return grid != sectorGrids.end() && grid->second.admitsPlayer
            && !grid->second.cells.empty();
    }

    // The closest cell of one walk area -- somewhere the bot can get to on
    // foot from where it is standing.
    int nearestNavCellInArea(int x, int y, int area) const
    {
        if (area < 0)
            return -1;
        int best = -1;
        int bestDistance = INT32_MAX;
        for (const NavCell &cell : navCells)
        {
            if (cell.walkArea != area)
                continue;
            const int currentDistance = distance2(x, y, cell.center.x, cell.center.y);
            if (currentDistance < bestDistance)
            {
                bestDistance = currentDistance;
                best = cell.id;
            }
        }
        return best;
    }

    int nearestNavCell(int sectorId, int x, int y) const
    {
        const int exact = navCellAt(sectorId, x >> kNavGridShift, y >> kNavGridShift);
        if (exact >= 0)
            return exact;
        // Off-grid poses (a doorway tighter than the body margin, a moving
        // floor mid-travel) still need a handle on the mesh: fall back to
        // the closest cell of the same sector, then to the closest anywhere.
        int best = -1;
        int bestDistance = INT32_MAX;
        int fallback = -1;
        int fallbackDistance = INT32_MAX;
        for (const NavCell &cell : navCells)
        {
            const int currentDistance = distance2(x, y, cell.center.x, cell.center.y);
            if (cell.sector == sectorId)
            {
                if (currentDistance < bestDistance)
                {
                    bestDistance = currentDistance;
                    best = cell.id;
                }
            }
            else if (currentDistance < fallbackDistance)
            {
                fallbackDistance = currentDistance;
                fallback = cell.id;
            }
        }
        return best >= 0 ? best : fallback;
    }

    // Does any point in this sector leave the player's own radius clear of
    // its walls?  Only asked when the 256-unit grid produced nothing, so the
    // area being swept is small; a sector big enough to make the sweep
    // expensive is by definition not the narrow case.
    //
    // Width only.  How much headroom a sector has right now is world state
    // -- a shut door has none and opens later -- and folding it in here
    // would turn "closed" into "does not exist".
    bool sectorAdmitsPlayer(int sectorId, int minX, int maxX, int minY, int maxY) const
    {
        const int radius = playerClipRadius();
        const int step = std::max(32, radius / 2);
        const int64_t required = int64_t(radius) * radius;
        int samples = 0;
        for (int y = minY; y <= maxY; y += step)
        {
            for (int x = minX; x <= maxX; x += step)
            {
                if (++samples > 8192)
                    return true;
                if (inside(x, y, sectorId) != 1)
                    continue;
                if (blockedBySolidSprite(sectorId, x, y))
                    continue;
                if (nearestWallDistance2(sectorId, x, y) >= required)
                    return true;
            }
        }
        return false;
    }

    // Regenerate one sector's standable grid.  Cached per sector and only
    // recomputed when that sector's own geometry signature changes, so a
    // moving door does not cost a whole-map rebuild.
    void buildNavSector(int sectorId)
    {
        if (!inRange(sectorId, 0, numsectors))
            return;
        SectorGrid &grid = sectorGrids[sectorId];
        grid.cells.clear();
        const sectortype &sectorRecord = sector[sectorId];
        if (sectorRecord.wallnum <= 0)
            return;

        int minX = INT32_MAX, maxX = INT32_MIN, minY = INT32_MAX, maxY = INT32_MIN;
        for (int i = 0; i < sectorRecord.wallnum; ++i)
        {
            const int wallId = sectorRecord.wallptr + i;
            if (!inRange(wallId, 0, numwalls))
                continue;
            minX = std::min(minX, int(wall[wallId].x));
            maxX = std::max(maxX, int(wall[wallId].x));
            minY = std::min(minY, int(wall[wallId].y));
            maxY = std::max(maxY, int(wall[wallId].y));
        }
        if (minX > maxX || minY > maxY)
            return;

        const int clearance = playerBodyClearance();
        const int radius = playerClipRadius();
        // A visited sector that produces no cells disconnects the whole
        // graph, so the filters relax in order rather than failing closed:
        // body margin first, then standing clearance.  The bot has stood in
        // this sector, so some representation of it is always correct.
        const int step = 1 << (kNavGridShift - 2);
        // Whether the cells that survived were placed by a pass that still
        // demanded the player's own half-width.  The later, relaxed passes
        // exist to keep a sector represented at all, and their cells say
        // nothing about whether a body fits there.
        bool marginRespected = false;
        for (int pass = 0; pass < 5 && grid.cells.empty(); ++pass)
        {
            // pass 0: plain grid centres with a body margin.
            // pass 1: the same, but sampling inside each square for the most
            //         open point -- rescues corridors narrower than the grid.
            // pass 2: drop the margin.  pass 3: drop the clearance too.
            const bool sample = pass >= 1;
            // Demand the player's real half-width first: allowing less let
            // the bot plan between columns it can never fit past.  Relax it
            // only where insisting would leave the sector unrepresented,
            // which is the case for small chambers like a doorway.
            const int64_t margin = pass <= 2 ? int64_t(radius) * radius : 0;
            const int required = pass < 4 ? clearance : 0;
            int insideSquares = 0;
            for (int gy = minY >> kNavGridShift; gy <= (maxY >> kNavGridShift); ++gy)
            {
                for (int gx = minX >> kNavGridShift; gx <= (maxX >> kNavGridShift); ++gx)
                {
                    const int baseX = (gx << kNavGridShift) + (1 << (kNavGridShift - 1));
                    const int baseY = (gy << kNavGridShift) + (1 << (kNavGridShift - 1));
                    if (inside(baseX, baseY, sectorId) == 1)
                        ++insideSquares;
                    int bestX = baseX;
                    int bestY = baseY;
                    int64_t bestClearance = -1;
                    for (int ox = sample ? -1 : 0; ox <= (sample ? 1 : 0); ++ox)
                    {
                        for (int oy = sample ? -1 : 0; oy <= (sample ? 1 : 0); ++oy)
                        {
                            const int cx = baseX + ox * step;
                            const int cy = baseY + oy * step;
                            if (inside(cx, cy, sectorId) != 1)
                                continue;
                            if (standingFloorZ(sectorId, cx, cy)
                                - getceilzofslope(sectorId, cx, cy) < required)
                                continue;
                            if (blockedBySolidSprite(sectorId, cx, cy))
                                continue;
                            const int64_t clear = nearestWallDistance2(sectorId, cx, cy);
                            if (clear > bestClearance)
                            {
                                bestClearance = clear;
                                bestX = cx;
                                bestY = cy;
                            }
                        }
                    }
                    if (bestClearance < 0 || bestClearance < margin)
                        continue;
                    GridCell cell;
                    cell.gx = gx;
                    cell.gy = gy;
                    cell.x = bestX;
                    cell.y = bestY;
                    grid.cells.push_back(cell);
                }
            }
            // A sector whose walkable band is narrower than the grid keeps
            // most of its squares and loses their centres.  That is the case
            // sampling exists for; anywhere else, leave the plain grid alone.
            if (pass == 0 && int(grid.cells.size()) * 2 < insideSquares)
            {
                char detail[160];
                snprintf(detail, sizeof(detail), "sector=%d centred_cells=%u inside_squares=%d",
                         sectorId, unsigned(grid.cells.size()), insideSquares);
                event("nav_sector_too_tight_for_grid", detail);
                grid.cells.clear();
            }
            if (!grid.cells.empty() && margin > 0)
                marginRespected = true;
        }
        // A sector too thin to hold a standable square is still crossed --
        // a step, a ledge, a door track.  Give it a transit cell at each of
        // its doorways so routes can pass through, even though nothing will
        // choose to stand there.  Without this the strict destination lookup
        // severs every route that runs through such a place.
        //
        // But only where the player fits at all.  A 128-unit slot between
        // two rooms is not a tight corridor, it is a wall with a seam in it,
        // and handing it transit cells is what had the bot shouldering the
        // masonry beside it for the rest of the run.
        grid.admitsPlayer = marginRespected
            || sectorAdmitsPlayer(sectorId, minX, maxX, minY, maxY);
        if (grid.cells.empty() && grid.admitsPlayer)
        {
            for (int i = 0; i < sectorRecord.wallnum; ++i)
            {
                const int wallId = sectorRecord.wallptr + i;
                if (!inRange(wallId, 0, numwalls) || !inRange(wall[wallId].point2, 0, numwalls))
                    continue;
                if (!inRange(wall[wallId].nextsector, 0, numsectors))
                    continue;
                const walltype &start = wall[wallId];
                const walltype &end = wall[start.point2];
                const int midX = (start.x + end.x) / 2;
                const int midY = (start.y + end.y) / 2;
                const int gx = midX >> kNavGridShift;
                const int gy = midY >> kNavGridShift;
                bool present = false;
                for (const GridCell &cell : grid.cells)
                    if (cell.gx == gx && cell.gy == gy)
                        present = true;
                if (present)
                    continue;
                GridCell cell;
                cell.gx = gx;
                cell.gy = gy;
                cell.x = midX;
                cell.y = midY;
                grid.cells.push_back(cell);
            }
            if (!grid.cells.empty())
            {
                char detail[128];
                snprintf(detail, sizeof(detail), "sector=%d transit_cells=%u",
                         sectorId, unsigned(grid.cells.size()));
                event("nav_sector_transit_only", detail);
            }
        }

        // A sector the player is standing in always contains at least the
        // player's own square, even if it is off-grid or too tight above.
        if (observation.sector == sectorId)
        {
            const int gx = observation.x >> kNavGridShift;
            const int gy = observation.y >> kNavGridShift;
            bool present = false;
            for (const GridCell &cell : grid.cells)
                if (cell.gx == gx && cell.gy == gy)
                    present = true;
            if (!present)
            {
                GridCell cell;
                cell.gx = gx;
                cell.gy = gy;
                cell.x = (gx << kNavGridShift) + (1 << (kNavGridShift - 1));
                cell.y = (gy << kNavGridShift) + (1 << (kNavGridShift - 1));
                grid.cells.push_back(cell);
            }
        }
        grid.signature = sectorGeometrySignature(sectorId);
    }

    // Does the straight line between two points cross a wall of this sector?
    bool segmentCrossesSectorWall(int sectorId, int ax, int ay, int bx, int by) const
    {
        if (!inRange(sectorId, 0, numsectors))
            return true;
        const sectortype &record = sector[sectorId];
        for (int i = 0; i < record.wallnum; ++i)
        {
            const int wallId = record.wallptr + i;
            if (!inRange(wallId, 0, numwalls) || !inRange(wall[wallId].point2, 0, numwalls))
                continue;
            const walltype &start = wall[wallId];
            const walltype &end = wall[start.point2];
            const int64_t d1 = int64_t(bx - ax) * (start.y - ay) - int64_t(by - ay) * (start.x - ax);
            const int64_t d2 = int64_t(bx - ax) * (end.y - ay) - int64_t(by - ay) * (end.x - ax);
            if ((d1 > 0 && d2 > 0) || (d1 < 0 && d2 < 0))
                continue;
            const int64_t d3 = int64_t(end.x - start.x) * (ay - start.y)
                - int64_t(end.y - start.y) * (ax - start.x);
            const int64_t d4 = int64_t(end.x - start.x) * (by - start.y)
                - int64_t(end.y - start.y) * (bx - start.x);
            if ((d3 > 0 && d4 > 0) || (d3 < 0 && d4 < 0))
                continue;
            return true;
        }
        return false;
    }

    // Does a solid sprite stand where the player wants to be?  Build's
    // blocking bit (cstat 1) is the same thing clipmove honours.
    bool blockedBySolidSprite(int sectorId, int x, int y) const
    {
        return solidSpriteAt(sectorId, x, y, playerClipRadius()) >= 0;
    }

    int64_t nearestWallDistance2(int sectorId, int x, int y) const
    {
        const sectortype &sectorRecord = sector[sectorId];
        int64_t best = INT64_MAX;
        for (int i = 0; i < sectorRecord.wallnum; ++i)
        {
            const int wallId = sectorRecord.wallptr + i;
            if (!inRange(wallId, 0, numwalls) || !inRange(wall[wallId].point2, 0, numwalls))
                continue;
            const walltype &wallRecord = wall[wallId];
            const walltype &nextWall = wall[wallRecord.point2];
            best = std::min(best, dist2ToSegment(x, y, wallRecord.x, wallRecord.y,
                                                 nextWall.x, nextWall.y));
        }
        return best;
    }

    int sectorGeometrySignature(int sectorId) const
    {
        if (!inRange(sectorId, 0, numsectors))
            return 0;
        const sectortype &record = sector[sectorId];
        int signature = llmapper::mixHash(17, record.floorz);
        signature = llmapper::mixHash(signature, record.ceilingz);
        signature = llmapper::mixHash(signature, record.wallnum);
        for (int nSprite = headspritesect[sectorId]; nSprite >= 0;
             nSprite = nextspritesect[nSprite])
        {
            if (!(sprite[nSprite].cstat & CSTAT_SPRITE_BLOCK))
                continue;
            signature = llmapper::mixHash(signature, sprite[nSprite].x);
            signature = llmapper::mixHash(signature, sprite[nSprite].y);
        }
        for (int i = 0; i < record.wallnum; ++i)
        {
            const int wallId = record.wallptr + i;
            if (!inRange(wallId, 0, numwalls))
                continue;
            signature = llmapper::mixHash(signature, wall[wallId].x);
            signature = llmapper::mixHash(signature, wall[wallId].y);
            signature = llmapper::mixHash(signature, wall[wallId].nextsector);
            signature = llmapper::mixHash(signature, wall[wallId].cstat);
        }
        return signature;
    }

    NavEdgeMode navModeForPortal(const Portal &portal) const
    {
        if (portal.interactionAffordance && !portal.traversable)
            return kNavInteraction;
        if (portal.jumpable)
            return kNavJump;
        if (portal.crouchable)
            return kNavCrouch;
        if (portal.dropSafe)
            return kNavDrop;
        if (portal.floorDelta != 0)
            return kNavStep;
        if (portal.walkable)
            return kNavWalk;
        return kNavBlocked;
    }

    void updateLiveNavGeometry()
    {
        // Grid centers are absolute world positions, so they do not drift
        // with a translating sector.  Only the in-motion flag is live; the
        // affected sector's grid is regenerated once its motion settles.
        for (NavCell &cell : navCells)
            cell.inMotion = navMeshInMotion;
    }

    void ensureNavTopology()
    {
        const int topology = navTopologyIdentity();
        const int pose = navPoseSignature();
        const int dynamicSignature = navDynamicSignature();
        if (topology == navTopologySignature && !navCells.empty())
        {
            const bool poseChanged = pose != navPoseSignatureValue;
            updateLiveNavGeometry();
            if (poseChanged)
            {
                if (!navMeshInMotion)
                {
                    navMeshInMotion = true;
                    navRoute.clear();
                    event("nav_mesh_in_motion", "reason=live_geometry_translating");
                }
                navPoseStableTicks = 0;
                navPoseSignatureValue = pose;
            }
            else if (navMeshInMotion)
            {
                ++navPoseStableTicks;
                // The second clause is deliberate and load-bearing.  While
                // any watched sector is travelling the pose signature changes
                // every tick, so the stable-tick count never accumulates and
                // the mesh would stay frozen -- and unroutable -- for the
                // whole door cycle.  The bot's own sector being idle is the
                // statement that its immediate surroundings are usable now,
                // which is what planning the next few steps needs; route
                // validity against remote motion is handled by the dynamic
                // link refresh below.  Dropping it strands the bot beside
                // every moving door until the door closes on it.
                if (navPoseStableTicks >= 8 || observation.localSectorBusy == 0)
                {
                    navMeshInMotion = false;
                    refreshDynamicNavLinks();
                    navDynamicSignatureValue = dynamicSignature;
                    event("nav_mesh_settled", "reason=pose_stable");
                }
            }
            else if (dynamicSignature != navDynamicSignatureValue)
            {
                refreshDynamicNavLinks();
                navDynamicSignatureValue = dynamicSignature;
                event("nav_links_refreshed", "source=dynamic_world_state");
            }
            return;
        }
        navTopologySignature = topology;
        navPoseSignatureValue = pose;
        navDynamicSignatureValue = dynamicSignature;
        navMeshInMotion = false;
        navPoseStableTicks = 0;
        navCells.clear();
        navCellIndex.clear();
        for (int sectorId : visitedSectors)
        {
            if (!inRange(sectorId, 0, numsectors))
                continue;
            auto grid = sectorGrids.find(sectorId);
            if (grid == sectorGrids.end()
                || grid->second.signature != sectorGeometrySignature(sectorId))
            {
                buildNavSector(sectorId);
                grid = sectorGrids.find(sectorId);
            }
            if (grid == sectorGrids.end())
                continue;
            for (const GridCell &source : grid->second.cells)
            {
                NavCell cell;
                cell.id = int(navCells.size());
                cell.sector = sectorId;
                cell.gx = source.gx;
                cell.gy = source.gy;
                cell.center = LocalWaypoint(source.x, source.y);
                navCellIndex[navCellHandle(sectorId, source.gx, source.gy)] = cell.id;
                navCells.push_back(cell);
            }
        }

        // Four-neighbour adjacency inside each sector.  Both centers are
        // already standable; the midpoint test rejects the rare case where
        // a pinch between two inner loops separates them.
        // Eight-connected: a diagonal link heals the thin necks that a
        // four-connected grid turns into a false wall between two halves of
        // one physical room.
        static const int kStepX[8] = { 1, -1, 0, 0, 1, 1, -1, -1 };
        static const int kStepY[8] = { 0, 0, 1, -1, 1, -1, 1, -1 };
        for (size_t i = 0; i < navCells.size(); ++i)
        {
            const NavCell &cell = navCells[i];
            for (int d = 0; d < 8; ++d)
            {
                const int neighbour = navCellAt(cell.sector, cell.gx + kStepX[d],
                                                cell.gy + kStepY[d]);
                if (neighbour < 0 || neighbour <= int(i))
                    continue;
                const NavCell &other = navCells[size_t(neighbour)];
                const int midX = (cell.center.x + other.center.x) / 2;
                const int midY = (cell.center.y + other.center.y) / 2;
                if (inside(midX, midY, cell.sector) != 1)
                    continue;
                // The midpoint test alone is not enough: a wall lying exactly
                // between two cell centres puts the midpoint on the wall,
                // where inside() is ambiguous, and a diagonal can slip past
                // the corner of a thin obstruction entirely.  Ask the engine
                // whether the two squares can actually see each other; a
                // one-sided wall between them blocks the ray.
                if (blockedBySolidSprite(cell.sector, midX, midY))
                    continue;
                // Deliberately geometry, not cansee(): that is a sight test
                // and is blocked by sprites, so a courtyard with scenery in
                // it was carved into separate walk areas by its own torches.
                if (segmentCrossesSectorWall(cell.sector, cell.center.x, cell.center.y,
                                             other.center.x, other.center.y))
                    continue;
                const int floorDelta = standingFloorZ(cell.sector, other.center.x, other.center.y)
                    - standingFloorZ(cell.sector, cell.center.x, cell.center.y);
                if (std::abs(floorDelta) > kMaxWalkableStep)
                    continue;
                const NavEdgeMode mode = floorDelta == 0 ? kNavWalk : kNavStep;
                const LocalWaypoint gateway(midX, midY);
                addNavLink(int(i), neighbour, mode, -1, gateway, true);
                addNavLink(neighbour, int(i), mode, -1, gateway, true);
            }
        }

        refreshDynamicNavLinks();
        ++navTopologyRevision;
        int areas = 0;
        for (const NavCell &cell : navCells)
            if (cell.walkArea + 1 > areas)
                areas = cell.walkArea + 1;
        char detail[128];
        snprintf(detail, sizeof(detail), "revision=%d cells=%u areas=%d sectors=%u",
                 navTopologyRevision, unsigned(navCells.size()), areas,
                 unsigned(visitedSectors.size()));
        event("nav_topology_rebuilt", detail);
        if (areas > 1)
        {
            // A fragmented mesh silently reports real routes as unreachable.
            // Name the split so it is diagnosable from telemetry alone.
            for (int sectorId : visitedSectors)
            {
                std::set<int> sectorAreas;
                int cells = 0;
                for (const NavCell &cell : navCells)
                    if (cell.sector == sectorId)
                    {
                        ++cells;
                        sectorAreas.insert(cell.walkArea);
                    }
                char split[192];
                int written = snprintf(split, sizeof(split), "sector=%d cells=%d areas=",
                                       sectorId, cells);
                for (int area : sectorAreas)
                    written += snprintf(split + written, sizeof(split) - size_t(written),
                                        "%d,", area);
                event("nav_area_split", split);
            }
        }
    }

    TraversalResult probeTraversal(int startX, int startY, int startZ, int startSector,
                                   int targetX, int targetY, int targetSector, int wallId)
    {
        const int radius = gMe && gMe->pSprite ? (gMe->pSprite->clipdist << 2) : 128;
        const MovementProbe probe = probeMovement(startX, startY, startZ, startSector,
                                                  targetX, targetY, targetSector,
                                                  std::max(256, radius));
        const int fromFloor = inRange(startSector, 0, numsectors)
            ? getflorzofslope(startSector, startX, startY) : startZ;
        const int toFloor = inRange(targetSector, 0, numsectors)
            ? getflorzofslope(targetSector, targetX, targetY) : startZ;
        const int fromCeil = inRange(startSector, 0, numsectors)
            ? getceilzofslope(startSector, startX, startY) : startZ;
        const int toCeil = inRange(targetSector, 0, numsectors)
            ? getceilzofslope(targetSector, targetX, targetY) : startZ;
        const int clearance = std::min(fromFloor - fromCeil, toFloor - toCeil);
        bool useable = false;
        if (probe.wall >= 0 && inRange(probe.wall, 0, numwalls)
            && wall[probe.wall].extra > 0 && wall[probe.wall].extra < kMaxXWalls
            && xwall[wall[probe.wall].extra].triggerPush)
            useable = true;
        if (wallId >= 0 && inRange(wallId, 0, numwalls)
            && wall[wallId].extra > 0 && wall[wallId].extra < kMaxXWalls
            && xwall[wall[wallId].extra].triggerPush)
            useable = true;
        return llmapper::classifyTraversal(toFloor - fromFloor, clearance, playerBodyClearance(),
                                           playerJumpRiseLimit(), kMaxWalkableStep, probe.reachable,
                                           probe.wall >= 0, useable, probe.sector >= 0);
    }

    // String pulling.  A grid route is a staircase of 256-unit hops; walking
    // it literally looks like a machine feeling its way along.  Collapse any
    // run of ordinary walk/step waypoints the player can cross in one clean
    // clipmove, and keep every special-capability step (jump, crouch, drop,
    // portal crossing) as its own waypoint.
    void smoothNavRoute(std::vector<NavRouteStep> &route) const
    {
        if (route.size() < 3)
            return;
        std::vector<NavRouteStep> pulled;
        size_t anchor = 0;
        int fromX = observation.x;
        int fromY = observation.y;
        int fromSector = observation.sector;
        while (anchor < route.size())
        {
            size_t furthest = anchor;
            for (size_t probe = anchor; probe < route.size(); ++probe)
            {
                const NavRouteStep &step = route[probe];
                if (step.wall >= 0 || !llmapper::walkMode(step.mode))
                    break;
                const int toX = step.hasGateway ? step.gateway.x : step.destination.x;
                const int toY = step.hasGateway ? step.gateway.y : step.destination.y;
                // Bound the span.  An unbounded shortcut turns a route that
                // correctly rounds a concave corner into a straight line
                // through the wall, and the bot then paces back and forth in
                // front of it.
                if (distance2(fromX, fromY, toX, toY)
                    > kRouteSmoothingSpan * kRouteSmoothingSpan)
                    break;
                // Two independent checks: the body must fit along the way,
                // and there must be nothing solid in between.
                if (!collisionProbe(fromX, fromY, observation.z, fromSector, toX, toY,
                                    step.targetSector, playerClipRadius()))
                    break;
                if (!cansee(fromX, fromY, observation.z, fromSector,
                            toX, toY, observation.z, step.targetSector >= 0
                                ? step.targetSector : fromSector))
                    break;
                furthest = probe;
            }
            pulled.push_back(route[furthest]);
            const NavRouteStep &taken = route[furthest];
            fromX = taken.hasGateway ? taken.gateway.x : taken.destination.x;
            fromY = taken.hasGateway ? taken.gateway.y : taken.destination.y;
            if (taken.targetSector >= 0)
                fromSector = taken.targetSector;
            anchor = furthest + 1;
        }
        if (pulled.size() < route.size())
            route.swap(pulled);
    }

    bool buildNavRoute(int targetX, int targetY, int targetSector, int signature)
    {
        ensureNavTopology();
        const int start = nearestNavCell(observation.sector, observation.x, observation.y);
        if (start < 0)
            return false;
        int target = navCellInSector(targetSector, targetX, targetY);
        if (target < 0)
        {
            // The mesh holds only sectors the bot has stood in, so a
            // frontier's far side has no cells of its own.  Aim at the
            // nearest cell the bot can actually walk to -- one in its own
            // walk area -- rather than the nearest cell anywhere, which is
            // routinely across a wall and turns a crossable boundary into an
            // unreachable one.
            target = nearestNavCellInArea(targetX, targetY,
                                          navCells[size_t(start)].walkArea);
        }
        // NOTE: the last resort is still the loose lookup, which can land in
        // another sector entirely.  navCellInSector() above is the strict
        // answer and is correct in principle, but making it the only answer
        // costs AGTST4 two thirds of its exploration -- routes that
        // legitimately pass through thin geometry stop resolving.  Fixing
        // that properly means representing thin sectors as real transit
        // rather than as cells, which is not done yet.
        if (target < 0)
            target = nearestNavCell(targetSector, targetX, targetY);
        std::vector<NavRouteStep> route;
        if (!llmapper::planNavRoute(navCells, start, target, targetX, targetY, targetSector, -1,
                                    navEdgeFailures, 0, route))
        {
            char failure[224];
            snprintf(failure, sizeof(failure),
                     "start_cell=%d goal_cell=%d dest_sector=%d want=(%d,%d) cells=%u failures=%u start_area=%d goal_area=%d",
                     start, target, targetSector, targetX, targetY,
                     unsigned(navCells.size()), unsigned(navEdgeFailures.size()),
                     start >= 0 && start < int(navCells.size()) ? navCells[size_t(start)].walkArea : -1,
                     target >= 0 && target < int(navCells.size()) ? navCells[size_t(target)].walkArea : -1);
            event("nav_route_unavailable", failure);
            return false;
        }
        smoothNavRoute(route);
        navRoute = route;
        navRouteIndex = 0;
        navRouteSignature = signature;
        navRouteTopologyRevision = navTopologyRevision;
        char routeDetail[256];
        snprintf(routeDetail, sizeof(routeDetail),
                 "steps=%u start_cell=%d goal_cell=%d dest_sector=%d want=(%d,%d) goal_cell_at=(%d,%d) player=(%d,%d,s%d)",
                 unsigned(navRoute.size()), start, target, targetSector, targetX, targetY,
                 target >= 0 && target < int(navCells.size()) ? navCells[size_t(target)].center.x : 0,
                 target >= 0 && target < int(navCells.size()) ? navCells[size_t(target)].center.y : 0,
                 observation.x, observation.y, observation.sector);
        event("nav_route_selected", routeDetail);
        return !navRoute.empty();
    }

    bool collisionProbe(int startX, int startY, int startZ, int startSector,
                        int targetX, int targetY, int targetSector, int tolerance) const
    {
        return probeMovement(startX, startY, startZ, startSector,
                             targetX, targetY, targetSector, tolerance).reachable;
    }

    int navigationSignature(int x, int y, int z, int sector, int id) const
    {
        int signature = x * 31 + y;
        signature = signature * 31 + z;
        signature = signature * 31 + sector;
        signature = signature * 31 + id;
        return signature;
    }

    void resetNavigation()
    {
        navigationActive = false;
        navigationWaypointActive = false;
        navigationUsingDetour = false;
        navigationDetourWall = -1;
        navigationDetourDepth = 0;
        navigationRecentWalls.clear();
        navRoute.clear();
        navRouteIndex = 0;
        navRouteSignature = 0;
        navRouteRejectedSignature = 0;
        navigationFailureSignature = 0;
        navigationFailureCount = 0;
        lastNavigationFailureTick = -1;
        lastNavWaypointX = INT32_MIN;
        lastNavWaypointY = INT32_MIN;
    }

    int wallOwnerSector(int wallId) const
    {
        if (!inRange(wallId, 0, numwalls))
            return -1;
        for (int sectorId = 0; sectorId < numsectors; ++sectorId)
        {
            const sectortype &record = sector[sectorId];
            if (wallId >= record.wallptr && wallId < record.wallptr + record.wallnum)
                return sectorId;
        }
        return -1;
    }

    // clipmove reports the physical wall it touched, not necessarily the
    // directional linedef owned by the player's current sector.  Resolve the
    // actual side before classifying vertical traversal.
    int directionalNavigationWall(int wallId) const
    {
        if (!inRange(wallId, 0, numwalls))
            return -1;
        const int owner = wallOwnerSector(wallId);
        if (owner == observation.sector)
            return wallId;
        const walltype &record = wall[wallId];
        if (record.nextsector == observation.sector
            && inRange(record.nextwall, 0, numwalls))
            return record.nextwall;
        return -1;
    }

    NavEdgeMode classifyNavigationBlock(int wallId, int x, int y) const
    {
        (void)x;
        (void)y;
        const int directionalWall = directionalNavigationWall(wallId);
        if (!inRange(directionalWall, 0, numwalls))
            return kNavBlocked;
        const walltype &wallRecord = wall[directionalWall];
        if (!inRange(wallRecord.point2, 0, numwalls))
            return kNavBlocked;
        const int sampleX = (wallRecord.x + wall[wallRecord.point2].x) / 2;
        const int sampleY = (wallRecord.y + wall[wallRecord.point2].y) / 2;
        if (!inRange(wallRecord.nextsector, 0, numsectors))
        {
            if (inRange(wallRecord.extra, 1, kMaxXWalls)
                && xwall[wallRecord.extra].triggerPush)
                return kNavInteraction;
            return kNavBlocked;
        }
        if (wallRecord.cstat & 1)
        {
            if (inRange(wallRecord.extra, 1, kMaxXWalls)
                && xwall[wallRecord.extra].triggerPush)
                return kNavInteraction;
            return kNavBlocked;
        }
        const int fromSector = observation.sector;
        // A collision can stop at different points along a sloped wall.  Use
        // the portal midpoint for capability classification so the same
        // directional opening does not oscillate between BLOCKED and JUMP
        // merely because clipmove stopped at a different x/y.
        const int fromFloor = getflorzofslope(fromSector, sampleX, sampleY);
        const int toFloor = getflorzofslope(wallRecord.nextsector, sampleX, sampleY);
        const int fromCeiling = getceilzofslope(fromSector, sampleX, sampleY);
        const int toCeiling = getceilzofslope(wallRecord.nextsector, sampleX, sampleY);
        const int clearance = std::min(fromFloor - fromCeiling, toFloor - toCeiling);
        const int width = int(std::sqrt(double(distance2(
            wallRecord.x, wallRecord.y, wall[wallRecord.point2].x,
            wall[wallRecord.point2].y))));
        const int floorDelta = toFloor - fromFloor;
        const int body = playerBodyClearance();
        const int rise = playerJumpRiseLimit();
        bool mechanism = false;
        if (inRange(wallRecord.extra, 1, kMaxXWalls)
            && xwall[wallRecord.extra].triggerPush)
            mechanism = true;
        if (inRange(wallRecord.nextsector, 0, numsectors)
            && sector[wallRecord.nextsector].extra > 0
            && sector[wallRecord.nextsector].extra < kMaxXSectors
            && xsector[sector[wallRecord.nextsector].extra].Wallpush)
            mechanism = true;
        if (width >= playerPassageWidth() && clearance >= body
            && std::abs(floorDelta) <= kMaxWalkableStep)
            return floorDelta == 0 ? kNavWalk : kNavStep;
        if (width >= playerPassageWidth() && clearance >= body
            && !mechanism && floorDelta < 0 && -floorDelta <= rise)
            return kNavJump;
        if (width >= playerPassageWidth() && clearance >= body
            && floorDelta > 0 && floorDelta <= rise)
            return kNavDrop;
        if (inRange(wallRecord.extra, 1, kMaxXWalls)
            && xwall[wallRecord.extra].triggerPush)
            return kNavInteraction;
        return kNavBlocked;
    }

    bool chooseNavigationDetour(const MovementProbe &blocked)
    {
        if (blocked.wall < 0 || !inRange(blocked.wall, 0, numwalls)
            || navigationDetourDepth >= 4
            || navigationRecentWalls.count(blocked.wall))
            return false;

        const walltype &obstacle = wall[blocked.wall];
        if (!inRange(obstacle.point2, 0, numwalls))
            return false;
        const walltype &obstacleEnd = wall[obstacle.point2];
        const int radius = gMe && gMe->pSprite ? (gMe->pSprite->clipdist << 2) : 128;
        const int margin = std::max(512, radius + 256);
        const int dirs[][2] = {{ 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 },
                               { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 }};
        const int corners[][2] = {{ obstacle.x, obstacle.y },
                                  { obstacleEnd.x, obstacleEnd.y }};
        struct Candidate { int x; int y; bool finalReachable; int distance; };
        std::vector<Candidate> candidates;
        for (const auto &corner : corners)
        {
            for (const auto &dir : dirs)
            {
                const int norm = (dir[0] && dir[1]) ? 181 : 256;
                Candidate candidate;
                candidate.x = corner[0] + dir[0] * margin * 256 / norm;
                candidate.y = corner[1] + dir[1] * margin * 256 / norm;
                const MovementProbe fromPlayer = probeMovement(
                    observation.x, observation.y, observation.z, observation.sector,
                    candidate.x, candidate.y, observation.sector, std::max(256, radius));
                if (!fromPlayer.reachable)
                    continue;
                const MovementProbe toOriginal = probeMovement(
                    candidate.x, candidate.y, observation.z, observation.sector,
                    navigationOriginalX, navigationOriginalY, navigationOriginalSector,
                    std::max(256, radius));
                candidate.finalReachable = toOriginal.reachable;
                candidate.distance = distance2(candidate.x, candidate.y,
                                               navigationOriginalX, navigationOriginalY);
                candidates.push_back(candidate);
            }
        }
        if (candidates.empty())
            return false;
        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const Candidate &left, const Candidate &right)
                         {
                             if (left.finalReachable != right.finalReachable)
                                 return left.finalReachable > right.finalReachable;
                             return left.distance < right.distance;
                         });
        const Candidate &selected = candidates.front();
        navigationDetourWall = blocked.wall;
        navigationRecentWalls.insert(blocked.wall);
        navigationWaypointX = selected.x;
        navigationWaypointY = selected.y;
        navigationWaypointActive = true;
        navigationUsingDetour = true;
        ++navigationDetourDepth;
        char detail[192];
        snprintf(detail, sizeof(detail), "wall=%d waypoint=(%d,%d) depth=%d final_direct=%d",
                 blocked.wall, selected.x, selected.y, navigationDetourDepth,
                 selected.finalReachable ? 1 : 0);
        event("navigation_detour", detail);
        return true;
    }

    bool rememberCollisionInteraction(int wallIndex)
    {
        if (!inRange(wallIndex, 0, numwalls)
            || !inRange(observation.sector, 0, numsectors))
            return false;

        const walltype &wallRecord = wall[wallIndex];
        if (!inRange(wallRecord.point2, 0, numwalls))
            return false;
        if (!inRange(wallRecord.extra, 1, kMaxXWalls)
            || !xwall[wallRecord.extra].triggerPush)
            return false;

        const walltype &nextWall = wall[wallRecord.point2];
        InteractionCandidate candidate;
        candidate.kind = kInteractionWall;
        candidate.id = wallIndex;
        candidate.fromSector = observation.sector;
        candidate.targetSector = -1;
        candidate.x = (wallRecord.x + nextWall.x) / 2;
        candidate.y = (wallRecord.y + nextWall.y) / 2;
        candidate.z = observation.z;
        candidate.key = xwall[wallRecord.extra].key;
        candidate.locked = xwall[wallRecord.extra].locked != 0;
        candidate.reversible = true;
        candidate.target.wall = wallIndex;
        candidate.target.from = observation.sector;
        candidate.target.to = -1;
        candidate.target.x = candidate.x;
        candidate.target.y = candidate.y;
        candidate.target.wallPush = true;
        candidate.target.x1 = wallRecord.x;
        candidate.target.y1 = wallRecord.y;
        candidate.target.x2 = nextWall.x;
        candidate.target.y2 = nextWall.y;
        setInteractionGeometry(candidate.target);
        candidate.z = candidate.target.z;
        rememberInteraction(candidate);
        return true;
    }

    GINPUT navigateTo(int x, int y, int z, int targetSector, int targetId = -1,
                      TraversalCapability capability = kTraversalUnknown, bool shoot = false)
    {
        navigationActionableBlocker = -1;
        const int signature = navigationSignature(x, y, z, targetSector, targetId);
        if (!navigationActive || navigationOriginalSignature != signature)
        {
            // steerPortal may have prepared a route before entering this
            // shared primitive.  Preserve it when it belongs to exactly this
            // target; all other navigation state is reset normally.
            const bool preserveRoute = navRouteSignature == signature && !navRoute.empty();
            if (!preserveRoute)
                resetNavigation();
            navigationActive = true;
            navigationOriginalX = x;
            navigationOriginalY = y;
            navigationOriginalZ = z;
            navigationOriginalSector = targetSector;
            navigationOriginalId = targetId;
            navigationOriginalSignature = signature;
        }

        const int radius = gMe && gMe->pSprite ? (gMe->pSprite->clipdist << 2) : 128;
        const int tolerance = std::max(256, radius);
        MovementProbe direct = probeMovement(observation.x, observation.y, observation.z,
                                             observation.sector, x, y, targetSector, tolerance);
        if (direct.reachable)
        {
            if (navigationUsingDetour)
                event("navigation_retarget_original");
            navigationWaypointActive = false;
            navigationUsingDetour = false;
            navigationDetourDepth = 0;
            navigationRecentWalls.clear();
            return steerTo(x, y, false, shoot, targetId, targetSector, capability);
        }

        if (!navRoute.empty() && navRouteSignature == signature
            && navRouteTopologyRevision == navTopologyRevision)
        {
            const NavRouteStep &step = navRoute[navRouteIndex];
            const int waypointX = step.hasGateway ? step.gateway.x : step.destination.x;
            const int waypointY = step.hasGateway ? step.gateway.y : step.destination.y;
            if (distance2(observation.x, observation.y, waypointX, waypointY)
                <= tolerance * tolerance)
            {
                ++navRouteIndex;
                if (navRouteIndex >= navRoute.size())
                    navRoute.clear();
                else
                    event("nav_retarget_original");
            }
            if (!navRoute.empty())
            {
                const NavRouteStep &next = navRoute[navRouteIndex];
                currentObjective.routeStepWall = next.wall;
                currentObjective.routeStepFrom = next.sourceSector;
                currentObjective.routeStepTo = next.targetSector;
                const int nextX = next.hasGateway ? next.gateway.x : next.destination.x;
                const int nextY = next.hasGateway ? next.gateway.y : next.destination.y;
                const TraversalResult traversal = probeTraversal(
                    observation.x, observation.y, observation.z, observation.sector,
                    nextX, nextY, next.targetSector >= 0 ? next.targetSector : observation.sector,
                    next.wall);
                if (lastNavWaypointX != nextX || lastNavWaypointY != nextY)
                {
                    lastNavWaypointX = nextX;
                    lastNavWaypointY = nextY;
                    navWaypointBestDistance2 = distance2(observation.x, observation.y,
                                                         nextX, nextY);
                    navWaypointProgressTick = observation.tick;
                    char waypointDetail[192];
                    snprintf(waypointDetail, sizeof(waypointDetail),
                             "mode=%s at=(%d,%d) sector=%d remaining=%u player=(%d,%d)",
                             llmapper::navEdgeModeName(next.mode), nextX, nextY,
                             next.targetSector,
                             unsigned(navRoute.size() - navRouteIndex),
                             observation.x, observation.y);
                    event("nav_cell_waypoint", waypointDetail);
                }
                const int toWaypoint = distance2(observation.x, observation.y, nextX, nextY);
                if (toWaypoint + kObjectiveProgressEpsilon2 < navWaypointBestDistance2)
                {
                    navWaypointBestDistance2 = toWaypoint;
                    navWaypointProgressTick = observation.tick;
                }
                if (traversal == kTraverseUseableBlocker)
                {
                    rememberCollisionInteraction(next.wall >= 0 ? next.wall : direct.wall);
                    event("navigation_interaction_opportunity", "source=traversal_probe");
                }
                const bool waypointStalled =
                    observation.tick - navWaypointProgressTick >= kWaypointStallTicks;
                if (!waypointStalled)
                {
                    TraversalCapability cap = capability;
                    if (traversal == kTraverseJump || next.mode == kNavJump)
                        cap = kTraversalJumpable;
                    else if (next.mode == kNavCrouch)
                        cap = kTraversalCrouchable;
                    return steerTo(nextX, nextY, false, shoot, targetId,
                                   next.targetSector >= 0 ? next.targetSector : observation.sector,
                                   cap);
                }
                // Genuinely stuck against this step after really trying it.
                //
                // Only boundaries are worth condemning.  Cell indices are
                // reassigned on every mesh rebuild, so a failure recorded
                // against a plain intra-sector grid link points at a
                // different pair of cells moments later and silently cuts
                // the graph somewhere unrelated -- which is how a two-sector
                // run ended up with no route between two cells in the same
                // walk area.  A wall is a stable identity; a cell id is not.
                if (next.wall >= 0)
                {
                    NavEdgeFailure failure;
                    failure.fromCell = -1;
                    failure.toCell = -1;
                    failure.wall = next.wall;
                    failure.mode = next.mode;
                    const Portal *stepPortal = portalByWallId(next.wall);
                    failure.geometrySignature = stepPortal ? portalIdentitySignature(*stepPortal)
                        : llmapper::mixHash(next.wall, next.sourceSector);
                    failure.attempts = 1;
                    failure.expiresTick = observation.tick + failureCooldownTicks(1);
                    navEdgeFailures.push_back(failure);
                    char failDetail[176];
                    snprintf(failDetail, sizeof(failDetail),
                             "wall=%d from=%d to=%d result=%s stalled_s=%d",
                             next.wall, next.sourceSector, next.targetSector,
                             llmapper::traversalResultName(traversal),
                             (observation.tick - navWaypointProgressTick) / kTicsPerSec);
                    event("nav_edge_rejected", failDetail);
                }
                navRoute.clear();
                lastNavWaypointX = INT32_MIN;
                if (buildNavRoute(x, y, targetSector, signature))
                {
                    const NavRouteStep &retry = navRoute.front();
                    const int retryX = retry.hasGateway ? retry.gateway.x : retry.destination.x;
                    const int retryY = retry.hasGateway ? retry.gateway.y : retry.destination.y;
                    navWaypointBestDistance2 = distance2(observation.x, observation.y,
                                                         retryX, retryY);
                    navWaypointProgressTick = observation.tick;
                    return steerTo(retryX, retryY, false, false, targetId,
                                   retry.targetSector >= 0 ? retry.targetSector : observation.sector,
                                   capability);
                }
            }
        }

        if (navigationWaypointActive)
        {
            if (distance2(observation.x, observation.y, navigationWaypointX,
                          navigationWaypointY) <= tolerance * tolerance)
            {
                navigationWaypointActive = false;
                event("navigation_retarget_original");
                return steerTo(x, y, false, shoot, targetId, targetSector, capability);
            }
            const MovementProbe waypoint = probeMovement(
                observation.x, observation.y, observation.z, observation.sector,
                navigationWaypointX, navigationWaypointY, observation.sector, tolerance);
            if (!waypoint.reachable && waypoint.wall >= 0
                && waypoint.wall != navigationDetourWall)
            {
                navigationWaypointActive = false;
                navigationUsingDetour = false;
            }
            else
            {
                return steerTo(navigationWaypointX, navigationWaypointY, false, false,
                               targetId, observation.sector, capability);
            }
        }

        if (direct.wall >= 0)
        {
            const NavEdgeMode blockMode = classifyNavigationBlock(direct.wall, direct.x, direct.y);
            const int ownerSector = wallOwnerSector(direct.wall);
            const int directionalWall = directionalNavigationWall(direct.wall);
            const int nextSector = inRange(directionalWall, 0, numwalls)
                ? wall[directionalWall].nextsector : -1;
            const int floorDelta = inRange(directionalWall, 0, numwalls)
                && inRange(nextSector, 0, numsectors)
                ? getflorzofslope(nextSector,
                                  (wall[directionalWall].x
                                   + wall[wall[directionalWall].point2].x) / 2,
                                  (wall[directionalWall].y
                                   + wall[wall[directionalWall].point2].y) / 2)
                    - getflorzofslope(observation.sector,
                                      (wall[directionalWall].x
                                       + wall[wall[directionalWall].point2].x) / 2,
                                      (wall[directionalWall].y
                                       + wall[wall[directionalWall].point2].y) / 2)
                : 0;
            const int nextWall = inRange(directionalWall, 0, numwalls)
                ? wall[directionalWall].nextwall : -1;
            const int openingWidth = inRange(directionalWall, 0, numwalls)
                && inRange(wall[directionalWall].point2, 0, numwalls)
                ? int(std::sqrt(double(distance2(
                    wall[directionalWall].x, wall[directionalWall].y,
                    wall[wall[directionalWall].point2].x,
                    wall[wall[directionalWall].point2].y)))) : 0;
            char classification[192];
            snprintf(classification, sizeof(classification),
                     "wall=%d directional_wall=%d owner=%d player_side=%d nextwall=%d mode=%s nextsector=%d floor_delta=%d width=%d stop=(%d,%d)",
                     direct.wall, directionalWall, ownerSector, observation.sector,
                     nextWall, navEdgeModeName(blockMode), nextSector, floorDelta, openingWidth,
                     direct.x, direct.y);
            event("navigation_block_classified", classification);
            if (blockMode == kNavWalk || blockMode == kNavStep || blockMode == kNavDrop)
            {
                char detail[96];
                snprintf(detail, sizeof(detail), "wall=%d mode=%s", direct.wall,
                         navEdgeModeName(blockMode));
                event("nav_edge", detail);
                return steerTo(x, y, false, shoot, targetId, targetSector, capability);
            }
            if (blockMode == kNavJump)
            {
                event("nav_edge", "mode=JUMP");
                return steerTo(x, y, false, shoot, targetId, targetSector,
                               kTraversalJumpable);
            }
            bool actionable = false;
            for (const InteractionCandidate &candidate : observation.interactions)
            {
                if (candidate.kind == kInteractionWall && candidate.id == direct.wall)
                {
                    actionable = true;
                    break;
                }
            }
            char detail[160];
            snprintf(detail, sizeof(detail), "wall=%d target=%d actionable=%d stop=(%d,%d)",
                     direct.wall, targetId, actionable ? 1 : 0, direct.x, direct.y);
            event("navigation_blocked", detail);
            if (rememberCollisionInteraction(direct.wall))
                event("navigation_interaction_opportunity", "source=clipmove_collision_hit");
            ensureNavTopology();
            if (navRouteRejectedSignature != signature
                && buildNavRoute(x, y, targetSector, signature))
            {
                const NavRouteStep &waypoint = navRoute.front();
                const int waypointX = waypoint.hasGateway ? waypoint.gateway.x : waypoint.destination.x;
                const int waypointY = waypoint.hasGateway ? waypoint.gateway.y : waypoint.destination.y;
                return steerTo(waypointX, waypointY, false, false, targetId,
                               waypoint.targetSector >= 0 ? waypoint.targetSector : observation.sector,
                               capability);
            }
            if (navMeshInMotion && chooseNavigationDetour(direct))
                return steerTo(navigationWaypointX, navigationWaypointY, false, false,
                               targetId, observation.sector, capability);
        }
        else
        {
            // The straight probe failed without hitting anything nameable --
            // typically because the destination lies past a boundary the
            // copied clipmove will not cross.  That is precisely what the
            // navigation mesh is for, so it must still be consulted.  Skipping
            // it here left whole maps unexplored: the bot declared its very
            // first frontier unreachable without ever planning a route.
            ensureNavTopology();
            if (navRouteRejectedSignature != signature
                && buildNavRoute(x, y, targetSector, signature))
            {
                const NavRouteStep &waypoint = navRoute.front();
                const int waypointX = waypoint.hasGateway ? waypoint.gateway.x
                                                          : waypoint.destination.x;
                const int waypointY = waypoint.hasGateway ? waypoint.gateway.y
                                                          : waypoint.destination.y;
                navWaypointBestDistance2 = distance2(observation.x, observation.y,
                                                     waypointX, waypointY);
                navWaypointProgressTick = observation.tick;
                return steerTo(waypointX, waypointY, false, false, targetId,
                               waypoint.targetSector >= 0 ? waypoint.targetSector
                                                          : observation.sector,
                               capability);
            }
        }
        if (navigationFailureSignature != signature)
        {
            navigationFailureSignature = signature;
            navigationFailureCount = 0;
        }
        ++navigationFailureCount;
        if (navigationFailureCount == 1)
            event("navigation_failed", "reason=no_bounded_collision_safe_detour");
        if (navigationFailureCount >= kNavigationFailureLimit && currentObjective.active
            && currentObjective.type != kObjectiveInteraction
            && currentObjective.type != kObjectiveInvestigate)
        {
            event("navigation_failed_bounded", "reason=objective_temporarily_unreachable");
            if (currentObjective.active && currentObjective.type == kObjectiveFrontier)
            {
                currentObjective.routeStepWall = movementTargetActive ? movementTargetId : currentObjective.wall;
                currentObjective.routeStepFrom = observation.sector;
                currentObjective.routeStepTo = movementTargetActive ? movementTargetSector
                                                                    : currentObjective.targetSector;
            }
            invalidateObjective("navigation_failed_bounded");
        }
        return GINPUT{};
    }

    void preparePortalWaypoint(const Portal &portal)
    {
        portalWaypointWall = portal.wall;
        portalPlanWall = portal.wall;
        portalPlanSignature = portalGeometrySignature(portal);
        portalPlan.clear();
        portalPlanIndex = 0;
        portalWaypointActive = false;
        portalCrossingActive = false;
        portalCrossingDestinationValid = false;
        portalCrossingStartTick = -1;
        portalWaypointX = portal.x;
        portalWaypointY = portal.y;

        const int dx = portal.x2 - portal.x1;
        const int dy = portal.y2 - portal.y1;
        const int length = int(std::sqrt(double(int64_t(dx) * dx + int64_t(dy) * dy)));
        const int radius = gMe && gMe->pSprite ? (gMe->pSprite->clipdist << 2) : 128;
        const int margin = std::max(192, radius + 96);
        if (length <= margin * 2)
            return;

        // The crossing interval is the wall segment after removing one
        // player radius at either end.  Test several points, not just the
        // midpoint, because Build corners commonly block that line.
        const double tangentX = double(dx) / length;
        const double tangentY = double(dy) / length;
        const int safeHalf = length / 2 - margin;
        std::vector<LocalWaypoint> crossings;
        // Prefer the center of the receiving opening.  End samples are still
        // useful around ordinary corners, but slide-door mechanisms often
        // place a second blocking wall immediately beyond one endpoint.
        const int offsets[] = { 0, -safeHalf / 2, safeHalf / 2, -safeHalf, safeHalf };
        for (int offset : offsets)
        {
            LocalWaypoint candidate;
            candidate.x = portal.x + int(std::lround(tangentX * offset));
            candidate.y = portal.y + int(std::lround(tangentY * offset));
            crossings.push_back(candidate);
        }

        const int startTolerance = std::max(192, radius + 128);
        // Enter only a small body-clearance beyond the portal.  A deeper
        // target can land on the first internal sliding-wall segment of a
        // mechanism room even when the portal itself is clear.  The next
        // observation can then discover and use that room's mechanism.
        const int crossingDepth = std::max(radius + 64, 256);
        auto crossingDestinations = [&](const LocalWaypoint &crossing) {
            const int normalX = -(portal.y2 - portal.y1);
            const int normalY = portal.x2 - portal.x1;
            const int normalLength = std::max(1, int(std::sqrt(double(
                int64_t(normalX) * normalX + int64_t(normalY) * normalY))));
            std::vector<LocalWaypoint> destinations;
            destinations.push_back({
                crossing.x + normalX * crossingDepth / normalLength,
                crossing.y + normalY * crossingDepth / normalLength
            });
            destinations.push_back({
                crossing.x - normalX * crossingDepth / normalLength,
                crossing.y - normalY * crossingDepth / normalLength
            });
            return destinations;
        };
        for (const LocalWaypoint &crossing : crossings)
        {
            for (const LocalWaypoint &destination : crossingDestinations(crossing))
            {
                if (collisionProbe(observation.x, observation.y, observation.z, observation.sector,
                                   destination.x, destination.y, portal.to, startTolerance))
                {
                    portalPlan.push_back(crossing);
                    portalWaypointX = crossing.x;
                    portalWaypointY = crossing.y;
                    portalCrossingDestinationX = destination.x;
                    portalCrossingDestinationY = destination.y;
                    portalCrossingDestinationValid = true;
                    portalWaypointActive = true;
                    event("local_plan_created", "mode=direct collision_probe=clipmove nodes=1");
                    return;
                }
            }
        }

        // Construct a deliberately small local visibility graph.  Candidate
        // nodes are player-sized offsets around nearby wall corners.  Every
        // edge is validated by copied-state clipmove; cansee() is never used
        // as a body-clearance proof.
        std::vector<LocalWaypoint> nodes;
        nodes.push_back({ observation.x, observation.y });
        for (int i = 0; i < sector[observation.sector].wallnum; ++i)
        {
            const walltype &wallRecord = wall[sector[observation.sector].wallptr + i];
            const walltype &nextWall = wall[wallRecord.point2];
            const int corners[2][2] = {{ wallRecord.x, wallRecord.y }, { nextWall.x, nextWall.y }};
            for (const auto &corner : corners)
            {
                const int dirs[][2] = {{ 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 },
                                        { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 }};
                for (const auto &dir : dirs)
                {
                    const int norm = (dir[0] && dir[1]) ? 181 : 256;
                    LocalWaypoint node;
                    node.x = corner[0] + dir[0] * margin * 256 / norm;
                    node.y = corner[1] + dir[1] * margin * 256 / norm;
                    if (collisionProbe(observation.x, observation.y, observation.z, observation.sector,
                                       node.x, node.y, observation.sector, startTolerance))
                        nodes.push_back(node);
                    if (nodes.size() >= 25)
                        break;
                }
                if (nodes.size() >= 25)
                    break;
            }
            if (nodes.size() >= 25)
                break;
        }

        struct SearchNode { LocalWaypoint point; int parent = -1; };
        std::vector<SearchNode> search;
        search.push_back({ nodes.front(), -1 });
        std::deque<int> queue;
        queue.push_back(0);
        std::vector<bool> used(nodes.size(), false);
        used[0] = true;
        int goal = -1;
        while (!queue.empty() && goal < 0)
        {
            const int current = queue.front();
            queue.pop_front();
            const LocalWaypoint &from = search[current].point;
            for (size_t i = 1; i < nodes.size(); ++i)
            {
                if (used[i])
                    continue;
                if (!collisionProbe(from.x, from.y, observation.z, observation.sector,
                                     nodes[i].x, nodes[i].y, observation.sector, startTolerance))
                    continue;
                used[i] = true;
                search.push_back({ nodes[i], current });
                queue.push_back(int(search.size() - 1));
                for (const LocalWaypoint &crossing : crossings)
                {
                    for (const LocalWaypoint &destination : crossingDestinations(crossing))
                    {
                        if (collisionProbe(nodes[i].x, nodes[i].y, observation.z, observation.sector,
                                           destination.x, destination.y, portal.to, startTolerance))
                        {
                            search.push_back({ crossing, int(search.size() - 1) });
                            portalCrossingDestinationX = destination.x;
                            portalCrossingDestinationY = destination.y;
                            portalCrossingDestinationValid = true;
                            goal = int(search.size() - 1);
                            break;
                        }
                    }
                    if (goal >= 0)
                        break;
                }
                if (goal >= 0)
                    break;
            }
        }
        if (goal < 0)
        {
            // Some Build door rooms deliberately overlap their entrance
            // geometry, so copied clipmove can reject every straight probe
            // even though the receiving sector contains a valid central
            // point.  Keep this as a bounded last resort: topology still
            // comes from inside(), and the real player tick remains the
            // authority on whether the crossing succeeds.
            const int normalX = -(portal.y2 - portal.y1);
            const int normalY = portal.x2 - portal.x1;
            const int normalLength = std::max(1, int(std::sqrt(double(
                int64_t(normalX) * normalX + int64_t(normalY) * normalY))));
            for (int direction : { 1, -1 })
            {
                const int destinationX = portal.x + direction * normalX * crossingDepth / normalLength;
                const int destinationY = portal.y + direction * normalY * crossingDepth / normalLength;
                if (inRange(portal.to, 0, numsectors)
                    && inside(destinationX, destinationY, portal.to) == 1)
                {
                    portalWaypointX = portal.x;
                    portalWaypointY = portal.y;
                    portalCrossingDestinationX = destinationX;
                    portalCrossingDestinationY = destinationY;
                    portalCrossingDestinationValid = true;
                    event("local_plan_fallback", "reason=receiving_sector_inside central_crossing=1");
                    break;
                }
            }
            event("local_plan_failed", "reason=no_collision_safe_path alternatives=5 corner_nodes=24");
            return;
        }
        std::vector<LocalWaypoint> reverse;
        for (int cursor = goal; cursor >= 0; cursor = search[cursor].parent)
            reverse.push_back(search[cursor].point);
        for (auto it = reverse.rbegin(); it != reverse.rend(); ++it)
            if (portalPlan.empty() || it->x != portalPlan.back().x || it->y != portalPlan.back().y)
                portalPlan.push_back(*it);
        if (!portalPlan.empty())
        {
            // The first node is the copied current pose, not a movement goal.
            portalPlanIndex = portalPlan.size() > 1 ? 1 : 0;
            portalWaypointX = portalPlan.back().x;
            portalWaypointY = portalPlan.back().y;
            portalWaypointActive = true;
        }
        char detail[128];
        snprintf(detail, sizeof(detail), "mode=corner_graph collision_probe=clipmove nodes=%u waypoints=%u",
                 unsigned(nodes.size()), unsigned(portalPlan.size()));
        event("local_plan_created", detail);
    }

    // A point just beyond the threshold, on the far side of the boundary.
    bool portalCrossingPoint(const Portal &portal, int &outX, int &outY) const
    {
        if (!inRange(portal.to, 0, numsectors))
            return false;
        const int normalX = -(portal.y2 - portal.y1);
        const int normalY = portal.x2 - portal.x1;
        const int length = std::max(1, int(std::sqrt(double(
            int64_t(normalX) * normalX + int64_t(normalY) * normalY))));
        const int depth = std::max(256, playerClipRadius() + 192);
        for (int direction = -1; direction <= 1; direction += 2)
        {
            const int candidateX = portal.x + direction * normalX * depth / length;
            const int candidateY = portal.y + direction * normalY * depth / length;
            if (inside(candidateX, candidateY, portal.to) == 1)
            {
                outX = candidateX;
                outY = candidateY;
                return true;
            }
        }
        return false;
    }

    // What posture does this opening currently require?  Re-derived from the
    // live clearance so a door that is still rising is handled correctly.
    TraversalCapability portalPosture(const Portal &portal) const
    {
        if (portal.clearance >= playerStandingClearance())
            return portal.floorDelta < -kMaxWalkableStep ? kTraversalJumpable
                                                         : kTraversalWalkable;
        if (portal.clearance >= playerCrouchClearance())
            return kTraversalCrouchable;
        return portal.capability;
    }

    GINPUT steerPortal(const Portal &portal)
    {
        const int bodyRadius = gMe && gMe->pSprite ? (gMe->pSprite->clipdist << 2) : 128;
        const int portalApproachTolerance = std::max(1024, bodyRadius + 1024);
        // Portal approach is now a specialization of the same local
        // topology used by arbitrary targets.  The old corner planner remains
        // below as a dynamic-geometry fallback, but do not ask it to steer a
        // straight line through a known same-sector NavCell route.
        if (!portalCrossingActive && portal.from == observation.sector
            && distance2(observation.x, observation.y, portal.x, portal.y)
                > portalApproachTolerance * portalApproachTolerance)
        {
            const MovementProbe sideProbe = probeMovement(
                observation.x, observation.y, observation.z, observation.sector,
                portal.x, portal.y, observation.sector, std::max(256, bodyRadius));
            if (!sideProbe.reachable)
            {
                const int routeSignature = navigationSignature(
                    portal.x, portal.y, portal.z, observation.sector, portal.wall);
                if (navRouteRejectedSignature != routeSignature
                    && (navRouteSignature != routeSignature
                    || navRouteTopologyRevision != navTopologyRevision
                    || navRoute.empty()))
                    buildNavRoute(portal.x, portal.y, observation.sector, routeSignature);
                if (!navRoute.empty() && navRouteSignature == routeSignature)
                {
                    GINPUT routeInput = navigateTo(portal.x, portal.y, portal.z,
                                                   observation.sector, portal.wall,
                                                   portal.capability);
                    if (routeInput.forward || routeInput.strafe || routeInput.q16turn
                        || routeInput.q16mlook || routeInput.buttonFlags.jump
                        || routeInput.buttonFlags.crouch)
                        return routeInput;
                    // A validated gateway route can be rejected by the real
                    // copied-state probe after geometry changes.  Continue
                    // into the bounded portal corner planner rather than
                    // returning an idle input and abandoning the crossing.
                }
            }
        }
        // Already at the threshold: walk through it.
        //
        // The window on an auto-closing door is short, so a crossing the bot
        // has committed to is driven straight from wherever it stands in the
        // near sector rather than being handed back to the corner planner.
        // Two seconds spent re-deciding after the door opened was the
        // difference between getting through and being caught in it.
        //
        // Driving straight only makes sense while the line is actually
        // clear, though.  With a pillar or a rotating door between the bot
        // and the gap it just walks into the obstacle and grinds along it
        // until the objective times out, so the long reach is granted only
        // when the body-sized probe says the run is unobstructed.
        const int nearby = std::max(1024, bodyRadius + 512);
        int adjacent = nearby;
        if (currentObjective.active && currentObjective.wall == portal.wall)
        {
            const int committed = std::max(4096, bodyRadius + 512);
            const int64_t range = distance2(observation.x, observation.y,
                                            portal.x, portal.y);
            if (range <= int64_t(nearby) * nearby)
                adjacent = committed;
            else if (range <= int64_t(committed) * committed
                     && probeMovement(observation.x, observation.y, observation.z,
                                      observation.sector, portal.x, portal.y,
                                      observation.sector,
                                      std::max(256, bodyRadius)).reachable)
                adjacent = committed;
        }
        // Driving straight is only worth anything while the bot is actually
        // getting somewhere.  Scraping along some other surface -- a pillar,
        // a rotating door sweeping the floor between here and the gap --
        // means the line is a fiction, and repeating it until the objective
        // budget runs out wastes the whole opening.  Hand those to the
        // corner planner below, which is the dynamic-geometry fallback.
        if (directCrossingWall != portal.wall)
        {
            directCrossingWall = portal.wall;
            directCrossingBlockedTicks = 0;
            directCrossingBlockedUntil = -1;
        }
        const int grindingOn = currentMoveWallHit();
        if (grindingOn >= 0 && grindingOn != portal.wall
            && !wallsShareVertex(grindingOn, portal.wall))
            ++directCrossingBlockedTicks;
        else if (directCrossingBlockedTicks > 0)
            --directCrossingBlockedTicks;
        if (directCrossingBlockedTicks >= kTicsPerSec)
        {
            // Hand the crossing to the planner for a few seconds rather than
            // one tick: alternating between the two every frame is its own
            // way of standing still.
            directCrossingBlockedTicks = 0;
            directCrossingBlockedUntil = observation.tick + 3 * kTicsPerSec;
            char detail[160];
            snprintf(detail, sizeof(detail), "wall=%d grinding_on=%d reason=straight_line_obstructed",
                     portal.wall, grindingOn);
            event("portal_crossing_deferred", detail);
        }
        const bool lineObstructed = observation.tick < directCrossingBlockedUntil;

        int throughX = 0;
        int throughY = 0;
        if (portal.from == observation.sector
            && !lineObstructed
            && distance2(observation.x, observation.y, portal.x, portal.y)
                <= adjacent * adjacent
            && portalCrossingPoint(portal, throughX, throughY))
        {
            const TraversalCapability posture = portalPosture(portal);
            const DoorTiming beyond = doorTiming(portal.to);
            if (posture == kTraversalCurrentlyUnavailable || beyond.closing)
            {
                // The opening is not passable yet.  Face it and wait rather
                // than walking into it: pushing at a door that is still
                // moving is how the bot ended up shouldering a closing door
                // on the way back instead of ducking under an open one.
                // Wait at the threshold, not back in the room.  Waiting
                // where it happened to be standing meant the bot only began
                // walking once the door was open, and arrived as it shut.
                GINPUT hold = {};
                const int toGap = angleDelta(
                    getangle(portal.x - observation.x, portal.y - observation.y),
                    observation.angle);
                hold.q16turn = fix16_from_int(toGap);
                const int stand = playerClipRadius() + 192;
                if (distance2(observation.x, observation.y, portal.x, portal.y)
                        > stand * stand
                    && std::abs(toGap) < 96)
                    hold.forward = 2047;
                // The engine already says how much room this will leave.
                const DoorTiming ahead = doorTiming(portal.to);
                const bool willNeedCrouch = ahead.openClearance > 0
                    && ahead.openClearance < playerStandingClearance();
                if (willNeedCrouch)
                {
                    hold.buttonFlags.crouch = 1;
                    if (!crouchTargetActive)
                    {
                        crouchTargetActive = true;
                        event("crouch_started", "reason=door_will_open_low");
                    }
                }
                if (lastCrossingWall != -portal.wall - 1)
                {
                    lastCrossingWall = -portal.wall - 1;
                    char detail[192];
                    snprintf(detail, sizeof(detail),
                             "wall=%d from=%d to=%d clearance=%d crouch=%d busy=%d",
                             portal.wall, portal.from, portal.to, portal.clearance,
                             playerCrouchClearance(),
                             portal.wallBusy || portal.sectorBusy ? 1 : 0);
                    event("portal_waiting_to_open", detail);
                }
                noteCameraOwner("PORTAL_WAIT", portal.wall, observation.angle, 0);
                return hold;
            }
            if (lastCrossingWall != portal.wall)
            {
                lastCrossingWall = portal.wall;
                char detail[192];
                snprintf(detail, sizeof(detail),
                         "wall=%d from=%d to=%d through=(%d,%d) clearance=%d posture=%d",
                         portal.wall, portal.from, portal.to, throughX, throughY,
                         portal.clearance, int(posture));
                event("portal_crossing_direct", detail);
            }
            return steerTo(throughX, throughY, false, false, portal.wall, portal.to,
                           posture);
        }
        lastCrossingWall = -1;

        const int signature = portalGeometrySignature(portal);
        if (portalPlanWall != portal.wall || portalPlanSignature != signature)
        {
            portalPlanAttempts = 0;
            preparePortalWaypoint(portal);
        }

        const int radius = gMe && gMe->pSprite ? (gMe->pSprite->clipdist << 2) : 128;
        const int waypointTolerance = std::max(512, radius * 2);
        if (portalWaypointActive && portalPlanIndex < portalPlan.size()
            && distance2(observation.x, observation.y, portalPlan[portalPlanIndex].x,
                         portalPlan[portalPlanIndex].y) <= waypointTolerance * waypointTolerance)
        {
            ++portalPlanIndex;
            if (portalPlanIndex >= portalPlan.size())
                portalWaypointActive = false;
        }
        int targetX = portalWaypointActive ? portalPlan[portalPlanIndex].x : portal.x;
        int targetY = portalWaypointActive ? portalPlan[portalPlanIndex].y : portal.y;
        const bool intermediate = portalWaypointActive && portalPlanIndex + 1 < portalPlan.size();
        const bool crouch = portal.capability == kTraversalCrouchable;
        if (crouch && !crouchTargetActive)
        {
            event("crouch_started", "reason=low_portal");
            crouchTargetActive = true;
        }

        // A bad local plan gets bounded alternatives before the edge is
        // recorded as failed.  The edge itself remains eligible if its
        // geometry signature changes later.
        const bool crossingStuck = portalCrossingActive && portalCrossingStartTick >= 0
            && observation.tick - portalCrossingStartTick >= kMovementStuckTicks;
        if ((movementTargetActive && movementTargetGoal == currentGoal
             && movementTargetId == portal.wall
             && observation.tick - targetLastProgressTick >= kMovementStuckTicks)
            || crossingStuck)
        {
            if (portalPlanAttempts < 3)
            {
                ++portalPlanAttempts;
                char detail[96];
                snprintf(detail, sizeof(detail), "wall=%d attempt=%d reason=waypoint_stuck", portal.wall, portalPlanAttempts);
                event("local_plan_retry", detail);
                portalPlanWall = -1;
                movementTargetActive = false;
                preparePortalWaypoint(portal);
            }
            else
            {
                const int edgeId = portal.wall * 65536 + portal.to;
                if (portal.capability == kTraversalWalkable && portal.floorDelta < 0
                    && -portal.floorDelta <= portal.jumpRiseLimit
                    && jumpFallbackEdges.insert(edgeId).second)
                {
                    event("local_jump_fallback", "reason=walkable_collision_approaches_exhausted");
                    portalPlanWall = -1;
                    movementTargetActive = false;
                    return steerPortal(portal);
                }
                recordEdgeFailure(portal, "local_portal_failed", "bounded_collision_safe_approaches_exhausted");
                movementTargetActive = false;
            }
        }
        const int targetSector = intermediate ? observation.sector : portal.to;
        if (!intermediate && !portalWaypointActive
            && distance2(observation.x, observation.y, portal.x, portal.y)
                <= (radius + 1024) * (radius + 1024))
        {
            // Once the player-sized probe has reached the safe crossing
            // interval, aim through the portal rather than at the wall line.
            // This keeps forward motion normal to the actual crossing and
            // avoids oscillating on a corner when the midpoint is exactly on
            // the collision plane.
            if (!portalCrossingActive)
            {
                if (portalCrossingDestinationValid)
                {
                    portalCrossingX = portalCrossingDestinationX;
                    portalCrossingY = portalCrossingDestinationY;
                }
                else
                {
                    const int dxToPortal = portalWaypointX - observation.x;
                    const int dyToPortal = portalWaypointY - observation.y;
                    const int distance = std::max(1, int(std::sqrt(double(distance2(observation.x, observation.y,
                                                                                    portalWaypointX, portalWaypointY)))));
                    // The crossing target must be just inside the receiving
                    // sector.  Sending the player deep into a compact
                    // mechanism room can hit its first internal wall before
                    // the bot has had a chance to operate it.
                    const int crossingDepth = std::max(radius + 64, 256);
                    portalCrossingX = portalWaypointX + dxToPortal * crossingDepth / distance;
                    portalCrossingY = portalWaypointY + dyToPortal * crossingDepth / distance;
                }
                portalCrossingActive = true;
                portalCrossingStartTick = observation.tick;
                char detail[96];
                snprintf(detail, sizeof(detail), "wall=%d x=%d y=%d", portal.wall, portalCrossingX, portalCrossingY);
                event("portal_crossing_push", detail);
            }
            targetX = portalCrossingX;
            targetY = portalCrossingY;
        }
        // The final push through a portal is a specialization of the generic
        // navigator.  Once the safe crossing point has been selected, retain
        // the proven shallow Build movement semantics: a moving-door portal
        // can still report its own slide-wall as a clipmove obstacle while
        // the real player tick is already able to consume the opening.
        if (portalCrossingActive)
            return steerTo(targetX, targetY, false, false, portal.wall, targetSector,
                           portal.capability);
        return navigateTo(targetX, targetY, portal.z, targetSector, portal.wall, portal.capability);
    }

    void executeJumpTraversal(GINPUT &input, int x, int y, int targetSector, int targetId,
                              int targetAngle, int angleError)
    {
        if (!gMe || !gMe->pSprite)
            return;
        if (jumpState == kJumpInactive || jumpTargetX != x || jumpTargetY != y
            || jumpTargetSector != targetSector)
        {
            // A bounded local recovery probe already stands at its takeoff
            // point.  It is not a route edge with a distant gateway, so do
            // not walk into the obstacle before issuing the real jump input.
            jumpState = targetId == -2 ? kJumpTakeoff : kJumpApproachTakeoff;
            jumpStateTick = observation.tick;
            jumpTargetX = x;
            jumpTargetY = y;
            jumpTargetSector = targetSector;
            jumpSourceSector = observation.sector;
            char detail[320];
            snprintf(detail, sizeof(detail),
                     "phase=%s route_mode=JUMP target=%d source_sector=%d target_sector=%d target=(%d,%d) player=(%d,%d,%d) posture=%d zvel=%d cant_jump=%d",
                     targetId == -2 ? "TAKEOFF" : "APPROACH_TAKEOFF",
                     targetId, observation.sector, targetSector, x, y, observation.x, observation.y,
                     observation.z, gMe->posture, observation.playerZVelocity, gMe->cantJump);
            event("jump_traversal", detail);
        }

        const int distance = int(std::sqrt(double(distance2(observation.x, observation.y, x, y))));
        if (jumpState != kJumpApproachTakeoff
            && ((targetSector >= 0 && targetSector != jumpSourceSector
                 && observation.sector == targetSector)
                || (targetSector == jumpSourceSector
                    && observation.sector != jumpSourceSector)))
        {
            event("jump_traversal", "phase=SUCCESS reason=target_sector_reached");
            jumpState = kJumpInactive;
            return;
        }

        switch (jumpState)
        {
        case kJumpApproachTakeoff:
            if (std::abs(angleError) >= 96)
            {
                input.forward = 0;
                jumpState = kJumpAlign;
                jumpStateTick = observation.tick;
                event("jump_traversal", "phase=ALIGN reason=takeoff_heading");
            }
            else if (distance <= kJumpTakeoffRange)
            {
                input.forward = 0;
                jumpState = kJumpTakeoff;
                jumpStateTick = observation.tick;
                event("jump_traversal", "phase=TAKEOFF reason=at_source_side_takeoff_point");
            }
            break;
        case kJumpAlign:
            input.forward = 0;
            if (std::abs(angleError) < 64)
            {
                jumpState = kJumpTakeoff;
                jumpStateTick = observation.tick;
                event("jump_traversal", "phase=TAKEOFF reason=heading_aligned");
            }
            break;
        case kJumpTakeoff:
            input.forward = 2047;
            if (std::abs(angleError) < 96 && !gMe->cantJump
                && (!gMe->pXSprite || gMe->pXSprite->height == 0))
            {
                input.buttonFlags.jump = 1;
                ++jumpAttempts;
                if (targetId == -2)
                    localDynamicJumpAttempted = true;
                jumpState = kJumpAirborne;
                jumpStateTick = observation.tick;
                char detail[320];
                snprintf(detail, sizeof(detail),
                         "phase=TAKEOFF jump_requested=1 jump_input_emitted=1 target=%d target_sector=%d player_z=%d zvel_before=%d posture=%d cant_jump=%d attempts=%d yaw=%d",
                         targetId, targetSector, observation.z, observation.playerZVelocity,
                         gMe->posture, gMe->cantJump, jumpAttempts, targetAngle);
                event("jump_input_emitted", detail);
            }
            break;
        case kJumpAirborne:
            input.forward = 2047;
            if (std::abs(observation.playerZVelocity) >= 64 || gMe->cantJump)
            {
                char detail[192];
                snprintf(detail, sizeof(detail),
                         "phase=AIRBORNE zvel=%d posture=%d cant_jump=%d sector=%d",
                         observation.playerZVelocity, gMe->posture, gMe->cantJump, observation.sector);
                event("jump_airborne_observed", detail);
            }
            if (observation.tick - jumpStateTick > kJumpActionTimeoutTicks)
            {
                if (jumpAttempts < kMaxJumpAttemptsPerTarget && !gMe->cantJump)
                {
                    jumpState = kJumpTakeoff;
                    jumpStateTick = observation.tick;
                    event("jump_traversal", "phase=TAKEOFF reason=bounded_retry");
                }
                else if (jumpAttempts >= kMaxJumpAttemptsPerTarget)
                {
                    event("jump_traversal", "phase=FAIL reason=bounded_attempts_exhausted");
                    jumpState = kJumpInactive;
                    if (targetId == -2 && localJumpProbeCount < 3)
                    {
                        ++localJumpProbeCount;
                        localDynamicJumpAttempted = false;
                        searchProbeActive = false;
                        searchAngle = wrapAngle(searchAngle + 512);
                        char detail[96];
                        snprintf(detail, sizeof(detail),
                                 "next_heading=%d probe=%d reason=jump_direction_failed",
                                 searchAngle, localJumpProbeCount + 1);
                        event("local_jump_probe_reoriented", detail);
                    }
                }
            }
            break;
        default:
            break;
        }
    }

    GINPUT steerTo(int x, int y, bool use, bool shoot, int targetId = -1, int targetSector = -1,
                   TraversalCapability capability = kTraversalUnknown)
    {
        if (capability != kTraversalCrouchable && crouchTargetActive)
        {
            event("crouch_released", "reason=normal_clearance");
            crouchTargetActive = false;
        }
        setMovementTarget(x, y, targetSector, targetId, capability);
        updateMovementProgress();
        GINPUT input = {};
        const int targetAngle = getangle(x - observation.x, y - observation.y);
        const int delta = angleDelta(targetAngle, observation.angle);
        input.syncFlags.run = 1;
        if (capability == kTraversalCrouchable)
        {
            input.buttonFlags.crouch = 1;
            if (!crouchTargetActive)
            {
                crouchTargetActive = true;
                event("crouch_started", "reason=explicit_crouch_traversal");
            }
            if (gMe && gMe->posture == kPostureCrouch)
                event("crouch_posture_observed", "phase=MOVE_CROUCHED");
        }
        // Aim the camera at the target, but do not stop to do it.
        //
        // Requiring the heading to be within ~17 degrees before applying any
        // forward input made the bot halt and pivot on the spot every time a
        // new objective pointed somewhere other than straight ahead -- the
        // "look back, look front, then continue" that shows up at every
        // doorway, and which is fatal in one that closes.  A player walks and
        // turns at once, so decompose the direction of travel into the
        // engine's current forward/strafe basis and keep moving while the
        // turn resolves.
        input.q16turn = fix16_from_int(delta);
        const int targetDistance2 = distance2(observation.x, observation.y, x, y);
        if (!use || targetDistance2 > kUseStopRange * kUseStopRange)
        {
            // Turn toward the target and walk once roughly facing it.
            if (std::abs(delta) < 96)
                input.forward = 2047;
        }
        // Travelling: return the view to level.  Nothing ever undid the
        // downward aim an interaction had asked for, so once the bot had
        // stooped to look at one low switch it walked the rest of the level
        // staring at the floor.
        {
            const int currentLook = fix16_to_int(gMe->q16look);
            if (currentLook != 0)
            {
                const int correction = std::max(-kLookLevelRate,
                                                std::min(kLookLevelRate, -currentLook));
                input.q16mlook = encodeLook(correction);
                noteCameraOwner("NAVIGATION", targetId, targetAngle, 0);
            }
        }
        if (use && targetDistance2 < kActionApproachRange * kActionApproachRange
            && std::abs(delta) < 96)
            input.keyFlags.action = 1;
        if (shoot)
            input.buttonFlags.shoot = 1;

        // Only jump when the ground actually demands it.  A stale jumpable
        // capability on an ordinary crossing had the bot hopping up every
        // step of a staircase, which is both slow and conspicuous.
        bool needsLift = capability == kTraversalJumpable;
        // Only second-guess a crossing into another sector.  A deliberate
        // same-sector jump is a probe or an escape and must be left alone.
        if (needsLift && targetSector != observation.sector
            && inRange(observation.sector, 0, numsectors)
            && inRange(targetSector, 0, numsectors))
        {
            const int here = getflorzofslope(observation.sector, observation.x, observation.y);
            const int there = getflorzofslope(targetSector, x, y);
            if (here - there <= kMaxWalkableStep)
            {
                needsLift = false;
                if (lastSuppressedJumpTarget != targetId)
                {
                    lastSuppressedJumpTarget = targetId;
                    char detail[160];
                    snprintf(detail, sizeof(detail), "target=%d rise=%d walk_step=%d",
                             targetId, here - there, kMaxWalkableStep);
                    event("jump_not_required", detail);
                }
            }
        }
        if (needsLift)
        {
            // A kNavJump route is an explicit Blood input sequence, never a
            // generic movement failure recovery.  playerProcess consumes the
            // normal GINPUT jump flag on the following game tick.
            executeJumpTraversal(input, x, y, targetSector, targetId, targetAngle, delta);
        }
        else if (movementNeedsJump())
        {
            char detail[256];
            snprintf(detail, sizeof(detail),
                     "goal=%s target=%d sector=%d target_sector=%d capability=%d distance=%d dx=%d dy=%d dz=%d forward=%d turn=%d jump_attempts=%d height=%d cant_jump=%d posture=%d",
                     currentGoal.c_str(), movementTargetId, observation.sector, movementTargetSector,
                     int(movementTargetCapability),
                     int(std::sqrt(double(distance2(observation.x, observation.y, movementTargetX, movementTargetY)))),
                     observation.x - targetLastX, observation.y - targetLastY, observation.z - targetLastZ,
                     int(input.forward), int(input.q16turn), jumpAttempts,
                     gMe->pXSprite ? gMe->pXSprite->height : -1, gMe->cantJump, gMe->posture);
            event("movement_stuck", detail);
            input.buttonFlags.jump = 1;
            input.forward = 2047;
            ++jumpAttempts;
            if (currentGoal == "SEARCH_CURRENT_AREA"
                && movementTargetCapability == kTraversalJumpable)
                localDynamicJumpAttempted = true;
            jumpCooldownTick = observation.tick + kJumpCooldownTicks;
            event("jump_attempt", detail);
        }
        else if ((currentGoal == "EXPLORE_FRONTIER" || currentGoal == "EXPLORE_LOCAL_PORTAL")
                 && movementTargetActive
                 && jumpAttempts >= kMaxJumpAttemptsPerTarget && movementTargetId >= 0
                 && movementTargetSector >= 0)
        {
            const Portal *portal = findPortalForEdge(movementTargetId * 65536 + movementTargetSector);
            if (portal)
                recordEdgeFailure(*portal, "frontier_failed", "jump_alternatives_exhausted");
            movementTargetActive = false;
        }
        return input;
    }

    GINPUT composeInput(const MovementIntent &move, const CombatIntent &combat,
                        const UseIntent &use) const
    {
        GINPUT input = {};
        input.syncFlags.run = move.run ? 1 : 0;
        input.buttonFlags.jump = move.jump ? 1 : 0;
        input.buttonFlags.crouch = move.crouch ? 1 : 0;
        input.keyFlags.action = use.action ? 1 : 0;
        // Who owns the camera.  Navigation keeps it while it is actually
        // going somewhere, unless the threat is pressing enough to be worth
        // stopping for: movement only walks once it is facing its waypoint,
        // so a distant enemy that took the camera left the bot standing
        // still -- and standing still was then read as the route being
        // impossible.  Fire from the route when the route points that way.
        const bool opportunistic = combat.aim && !combat.urgent && move.valid
            && std::abs(combat.turn) > kOpportunisticAimCone;
        if (combat.aim && !opportunistic)
        {
            input.q16turn = fix16_from_int(combat.turn);
            input.q16mlook = combat.look;
            if (combat.weapon && gMe->curWeapon != combat.weapon)
            {
                input.syncFlags.weaponChange = 1;
                input.newWeapon = uint8_t(combat.weapon);
            }
            else
                input.buttonFlags.shoot = combat.shoot ? 1 : 0;
            if (move.valid)
            {
                const int moveAngle = wrapAngle(observation.angle + combat.turn);
                const int worldAngle = wrapAngle(observation.angle + move.turn);
                const int forwardWorld = move.forward;
                const int strafeWorld = move.strafe;
                const int worldX = mulscale16(Cos(worldAngle), forwardWorld)
                    + mulscale16(-Sin(worldAngle), strafeWorld);
                const int worldY = mulscale16(Sin(worldAngle), forwardWorld)
                    + mulscale16(Cos(worldAngle), strafeWorld);
                input.forward = int16_t(mulscale16(Cos(moveAngle), worldX)
                    + mulscale16(Sin(moveAngle), worldY));
                input.strafe = int16_t(mulscale16(-Sin(moveAngle), worldX)
                    + mulscale16(Cos(moveAngle), worldY));
            }
            return input;
        }
        if (use.weapon && gMe->curWeapon != use.weapon)
        {
            input.syncFlags.weaponChange = 1;
            input.newWeapon = uint8_t(use.weapon);
        }
        else if (use.shoot)
            input.buttonFlags.shoot = 1;
        if (move.valid)
        {
            input.forward = int16_t(move.forward);
            input.strafe = int16_t(move.strafe);
            input.q16turn = fix16_from_int(move.turn);
            input.q16mlook = move.look;
        }
        return input;
    }

    MovementIntent intentFromInput(const GINPUT &input) const
    {
        MovementIntent move;
        move.forward = input.forward;
        move.strafe = input.strafe;
        move.turn = fix16_to_int(input.q16turn);
        move.look = input.q16mlook;
        move.jump = input.buttonFlags.jump;
        move.crouch = input.buttonFlags.crouch;
        move.run = input.syncFlags.run;
        move.valid = input.forward || input.strafe || input.q16turn || input.q16mlook
            || input.buttonFlags.jump || input.buttonFlags.crouch;
        return move;
    }

    int findRetreatCell(const VisibleObject &enemy)
    {
        ensureNavTopology();
        const int playerCell = nearestNavCell(observation.sector, observation.x, observation.y);
        if (playerCell < 0 || navCells[playerCell].walkArea < 0)
            return -1;
        const int area = navCells[playerCell].walkArea;
        const int currentEnemyDistance = distance2(observation.x, observation.y,
                                                   enemy.x, enemy.y);
        int bestCell = -1;
        int bestEnemyDistance = currentEnemyDistance;
        for (const NavCell &cell : navCells)
        {
            if (cell.walkArea != area)
                continue;
            const int enemyDistance = distance2(cell.center.x, cell.center.y, enemy.x, enemy.y);
            if (enemyDistance > bestEnemyDistance)
            {
                bestEnemyDistance = enemyDistance;
                bestCell = cell.id;
            }
        }
        return bestCell;
    }

    CombatIntent combatIntentFor(const VisibleObject &enemy, CombatTactic tactic,
                                 bool urgent = true)
    {
        CombatIntent combat;
        combat.aim = true;
        combat.urgent = urgent;
        combat.turn = angleDelta(getangle(enemy.x - observation.x, enemy.y - observation.y),
                                 observation.angle);
        const int horizontal = std::max(1, int(std::sqrt(double(distance2(
            observation.x, observation.y, enemy.x, enemy.y)))));
        const int currentLook = fix16_to_int(gMe->q16look);
        const int desiredLook = lookAngleForTarget(observation.z, enemy.z, horizontal);
        combat.look = encodeLook(desiredLook - currentLook);
        int weapon = 0;
        if (tactic == kCombatRanged && rangedWeaponAvailable(weapon))
        {
            combat.weapon = weapon;
            combat.shoot = gMe->curWeapon == weapon;
        }
        else if (tactic == kCombatMelee && meleeWeaponAvailable())
        {
            combat.weapon = kWeaponPitchfork;
            const int meleeRange = 1024;
            // Swing when in reach and roughly on target.  Flailing at the
            // air while still turning wastes the swing animation.
            combat.shoot = gMe->curWeapon == kWeaponPitchfork
                && distance2(enemy.x, enemy.y, observation.x, observation.y)
                    <= meleeRange * meleeRange
                && std::abs(combat.turn) < 200;
        }
        if (combat.shoot)
        {
            char detail[128];
            snprintf(detail, sizeof(detail), "sprite=%d weapon=%d distance=%d",
                     enemy.sprite, combat.weapon,
                     int(std::sqrt(double(distance2(observation.x, observation.y,
                                                    enemy.x, enemy.y)))));
            event(tactic == kCombatMelee ? "melee_swing" : "attack_fired", detail);
            noteCameraOwner("COMBAT", enemy.sprite, observation.angle + combat.turn,
                            desiredLook);
        }
        combat.valid = true;
        return combat;
    }

    // Every decision leaves a record of whether the bot actually asked to
    // move.  The stationary watchdog needs it: standing still because some
    // controller is deliberately holding position -- aiming at a switch,
    // waiting out a door, lining up a shot -- is not a movement failure, and
    // treating it as one suppressed perfectly good routes.
    GINPUT decide()
    {
        const GINPUT input = decideInput();
        if (input.forward || input.strafe)
            lastCommandedMoveTick = observation.tick;
        return input;
    }

    GINPUT decideInput()
    {
        GINPUT idle = {};
        if (!gMe || !gMe->pSprite || !gMe->pXSprite)
            return idle;
        if (result.size())
            return idle;

        // Calibration crouch: measure the crouched collision envelope once,
        // before any capability judgement depends on it.
        if (gObservedCrouchClearance == 0 && observation.tick < kCrouchCalibrationTicks)
        {
            if (currentGoal != "MEASURE_CROUCH_ENVELOPE")
                setGoal("MEASURE_CROUCH_ENVELOPE", -1);
            GINPUT calibrate = {};
            calibrate.buttonFlags.crouch = 1;
            return calibrate;
        }

        MovementIntent move;
        CombatIntent combat;
        UseIntent use;
        const VisibleObject *enemy = selectObject(kObjectEnemy);
        const bool attackerAlive = lastAttacker >= 0 && lastAttacker < kMaxSprites
            && sprite[lastAttacker].statnum < kMaxStatus;
        CombatSituation situation;
        situation.hasThreat = enemy && !unreachableEnemies.count(enemy->sprite);
        situation.immediateThreat = urgentThreat()
            || (lastDamageTick >= 0 && observation.tick - lastDamageTick <= 2 * kTicsPerSec
                && (attackerAlive || situation.hasThreat));
        int rangedWeapon = 0;
        situation.rangedAvailable = situation.hasThreat && rangedWeaponAvailable(rangedWeapon);
        const int enemyDistance2 = enemy
            ? distance2(observation.x, observation.y, enemy->x, enemy->y) : INT32_MAX;
        situation.meleeAvailable = meleeWeaponAvailable()
            && enemyDistance2 <= kMeleeEngageRange * kMeleeEngageRange;
        situation.retreatAvailable = situation.hasThreat && findRetreatCell(*enemy) >= 0;
        situation.critical = observation.health > 0 && observation.health < kCriticalHealth;
        situation.current = combatTactic;
        const CombatDecision decision = llmapper::chooseCombatTactic(situation);
        if (decision.tactic != combatTactic)
        {
            combatTactic = decision.tactic;
            combatTacticStartTick = observation.tick;
            char detail[128];
            snprintf(detail, sizeof(detail), "tactic=%s reason=%s override=%d",
                     llmapper::combatTacticName(decision.tactic), decision.reason,
                     decision.overrideMovement ? 1 : 0);
            event("combat_tactic_changed", detail);
            if (decision.tactic != kCombatNone && enemy)
            {
                char engaged[160];
                snprintf(engaged, sizeof(engaged), "sprite=%d tactic=%s distance=%d health=%d",
                         enemy->sprite, llmapper::combatTacticName(decision.tactic),
                         int(std::sqrt(double(distance2(observation.x, observation.y,
                                                        enemy->x, enemy->y)))),
                         observation.health);
                event("combat_engaged", engaged);
            }
            if (decision.tactic != kCombatRetreat)
                combatRetreatUnavailableEmitted = false;
        }
        if (situation.hasThreat && !situation.retreatAvailable && situation.immediateThreat
            && !combatRetreatUnavailableEmitted && !situation.rangedAvailable)
        {
            combatRetreatUnavailableEmitted = true;
            event("combat_retreat_unavailable", "reason=no_validated_reachable_cell");
        }
        if (decision.dropCombat)
            combatTactic = kCombatNone;
        if ((combatTactic == kCombatRetreat || combatTactic == kCombatMelee)
            && combatTacticStartTick >= 0
            && observation.tick - combatTacticStartTick > 6 * kTicsPerSec)
        {
            event("combat_tactic_timeout", llmapper::combatTacticName(combatTactic));
            combatTactic = kCombatNone;
        }
        if (combatTactic == kCombatRanged && enemy)
            combat = combatIntentFor(*enemy, kCombatRanged,
                                     situation.immediateThreat || situation.critical);
        else if (combatTactic == kCombatMelee && enemy)
        {
            combat = combatIntentFor(*enemy, kCombatMelee);
            GINPUT approach = navigateTo(enemy->x, enemy->y, enemy->z, enemy->sector,
                                         enemy->sprite, kTraversalUnknown);
            move = intentFromInput(approach);
            return composeInput(move, combat, use);
        }
        else if (combatTactic == kCombatRetreat && enemy)
        {
            const int cellId = findRetreatCell(*enemy);
            if (cellId >= 0)
            {
                const NavCell &cell = navCells[size_t(cellId)];
                GINPUT retreat = navigateTo(cell.center.x, cell.center.y, observation.z,
                                            cell.sector, -1, kTraversalUnknown);
                move = intentFromInput(retreat);
                combat = combatIntentFor(*enemy, kCombatRetreat);
                return composeInput(move, combat, use);
            }
        }

        if (observation.exitHere)
        {
            setGoal("USE_OBSERVED_EXIT", observation.sector);
            use.action = true;
            return composeInput(move, combat, use);
        }

        // Standing in a doorway while it is actually travelling is how the
        // bot gets crushed.  Getting out then is more urgent than any
        // exploration choice -- but merely passing through an idle door that
        // happens to close on a timer is normal, and treating that as an
        // emergency made the bot bounce in and out of every doorway.
        if (hazardousSector(observation.sector))
        {
            const int out = escapeHazard();
            const Portal *away = out >= 0 ? portalByWall(out, observation.sector, -1) : nullptr;
            if (away)
            {
                if (currentGoal != "LEAVE_HAZARD")
                {
                    char detail[160];
                    snprintf(detail, sizeof(detail), "sector=%d via_wall=%d to=%d health=%d",
                             observation.sector, away->wall, away->to, observation.health);
                    event("leave_hazard", detail);
                }
                setGoal("LEAVE_HAZARD", away->wall);
                return composeInput(intentFromInput(steerPortal(*away)), combat, use);
            }
        }

        const DoorTiming standingIn = doorTiming(observation.sector);
        // Leave a door that is shutting, and leave one that will shut on a
        // timer if there is no longer any reason to be standing in it.  With
        // nothing to execute the bot otherwise waits out the mechanism, or
        // starts a search probe, in the one place on the level where staying
        // put is fatal.
        const bool loiteringInMechanism = standingIn.autoCloses
            && !currentObjective.active;
        if (standingIn.closing || loiteringInMechanism)
        {
            if (clearingSector != observation.sector)
            {
                clearingSector = observation.sector;
                clearingWall = escapeMovingSector();
                if (clearingWall >= 0)
                {
                    char detail[208];
                    snprintf(detail, sizeof(detail),
                             "sector=%d via_wall=%d moving=%d auto_closes=%d remaining_s=%d hold_s=%d reason=%s",
                             observation.sector, clearingWall, standingIn.moving ? 1 : 0,
                             standingIn.autoCloses ? 1 : 0,
                             standingIn.remainingTicks / kTicsPerSec,
                             standingIn.holdTicks / kTicsPerSec,
                             standingIn.closing ? "closing_on_the_player" : "no_reason_to_stay");
                    event("clear_moving_sector", detail);
                }
            }
            const Portal *out = clearingWall >= 0
                ? portalByWall(clearingWall, observation.sector, -1) : nullptr;
            if (out)
            {
                setGoal("CLEAR_MOVING_SECTOR", out->wall);
                return composeInput(intentFromInput(steerPortal(*out)), combat, use);
            }
        }
        else if (clearingSector == observation.sector)
        {
            clearingSector = -1;
            clearingWall = -1;
        }

        // One selection point, one commitment, one reason.
        //
        // The bot keeps executing the mission it already chose.  It only
        // reconsiders when that mission finished, its budget ran out, or a
        // newly acquired key changed what the level offers.  Re-ranking
        // every affordance on every tick is what made the old behaviour
        // look like a committee rather than a player.
        const bool keysChanged = heldKeyMask != lastHeldKeyMask;
        if (keysChanged)
        {
            lastHeldKeyMask = heldKeyMask;
            if (currentObjective.active && mission.kind != llmapper::kMissionReturnForKey)
            {
                event("mission_interrupted", "reason=key_acquired");
                invalidateObjective("superseded_by_key_acquisition", false);
            }
        }
        if (observation.x != stationaryX || observation.y != stationaryY
            || observation.z != stationaryZ)
        {
            stationaryX = observation.x;
            stationaryY = observation.y;
            stationaryZ = observation.z;
            stationarySinceTick = observation.tick;
        }
        else if (observation.tick - stationarySinceTick > kStationaryLimitTicks
                 && currentGoal != "WAIT_MOVING_MECHANISM"
                 && !committedMechanismBusy())
        {
            // Not moving is only a failure if the bot was asking to move --
            // and only the route's failure if the route was in charge.  A
            // melee engagement commands translation and then stands still
            // because an enemy body is in the way; blaming the frontier for
            // that is how a perfectly good route came to be suppressed.
            const bool asked = observation.tick - lastCommandedMoveTick
                    <= kStationaryLimitTicks
                && (cameraOwner.empty() || cameraOwner == "NAVIGATION");
            char detail[192];
            snprintf(detail, sizeof(detail),
                     "seconds=%d sector=%d goal=%s objective_active=%d camera_owner=%s",
                     (observation.tick - stationarySinceTick) / kTicsPerSec,
                     observation.sector, currentGoal.c_str(),
                     currentObjective.active ? 1 : 0, cameraOwner.c_str());
            stationarySinceTick = observation.tick;
            if (!asked)
            {
                // Someone is holding position on purpose.  Say so and leave
                // the objective alone; the objective budget still bounds it.
                event("stationary_hold", detail);
            }
            else
            {
                event("stationary_deadlock", detail);
                if (currentObjective.active)
                    invalidateObjective("stationary_deadlock");
                resetNavigation();
            }
        }
        if (currentObjective.active && enforceObjectiveBudget())
        {
            // released; fall through and choose again this tick
        }
        // Access the bot just created is perishable.  Interrupt whatever it
        // was doing and go through, before the mechanism closes again.
        if (openedRouteWall >= 0)
        {
            const bool fresh = observation.tick - openedRouteTick <= kOpenedRouteWindowTicks
                || committedMechanismBusy();
            const bool deliberate = lastAcceptedUseTick >= 0
                && openedRouteTick - lastAcceptedUseTick <= kOpenedRouteWindowTicks;
            const bool worthwhile = fresh
                && (!visitedSectors.count(openedRouteTo) || deliberate);
            if (!worthwhile)
            {
                openedRouteWall = -1;
            }
            else if (!currentObjective.active
                     || currentObjective.wall != openedRouteWall)
            {
                Objective objective;
                objective.type = kObjectiveFrontier;
                objective.id = openedRouteWall;
                objective.wall = openedRouteWall;
                objective.sector = openedRouteFrom;
                objective.targetSector = openedRouteTo;
                exploreDestination = openedRouteTo;
                exploreCrossingWall = openedRouteWall;
                exploreCrossingFrom = openedRouteFrom;

                if (currentObjective.active)
                    invalidateObjective("superseded_by_opened_route", false);
                selectObjective(objective, "CONSUME_OPENED_ROUTE");
                missionOpportunity = 3000000 + openedRouteWall * 8 + int(kObjectiveFrontier);
                mission.kind = llmapper::kMissionContinue;
                mission.reason = "CONSUME_OPENED_ROUTE";
                char detail[192];
                snprintf(detail, sizeof(detail),
                         "wall=%d from=%d to=%d age_s=%d",
                         openedRouteWall, openedRouteFrom, openedRouteTo,
                         (observation.tick - openedRouteTick) / kTicsPerSec);
                event("consume_opened_route", detail);
                openedRouteWall = -1;
            }
        }
        if (!currentObjective.active)
        {
            rebuildLedger();
            llmapper::Mission chosen = llmapper::selectMission(
                ledger, observation.tick, heldKeyMask, missionOpportunity);
            if (chosen.kind == llmapper::kMissionNone)
                chosen = wakeSoonestDormant();
            if (chosen.kind == llmapper::kMissionNone || !commitMission(chosen))
            {
                missionOpportunity = -1;
                mission = llmapper::Mission();
            }
        }
        if (currentObjective.active)
        {
            GINPUT objectiveInput = executeObjective();
            move = intentFromInput(objectiveInput);
            use.action = objectiveInput.keyFlags.action != 0;
            use.shoot = objectiveInput.buttonFlags.shoot != 0;
            use.weapon = objectiveInput.syncFlags.weaponChange
                ? objectiveInput.newWeapon : 0;
            return composeInput(move, combat, use);
        }

        // Once a jump was deliberately launched, keep feeding forward motion
        // through the real airborne phase.  A transient moving-sector wait
        // must not erase the input sequence on the very next bot decision.
        if (jumpState != kJumpInactive && movementTargetActive
            && movementTargetCapability == kTraversalJumpable)
            return composeInput(intentFromInput(steerTo(movementTargetX, movementTargetY,
                                                        false, false, movementTargetId,
                                                        movementTargetSector,
                                                        movementTargetCapability)), combat, use);

        // Only geometry the bot is standing on, or the crossing it is
        // committed to, is worth waiting for.
        bool mechanismBusy = observation.localSectorBusy != 0;
        if (!mechanismBusy && currentObjective.active && currentObjective.wall >= 0)
        {
            for (const Portal &portal : observation.portals)
                if (portal.wall == currentObjective.wall
                    && (portal.wallBusy || portal.sectorBusy))
                    mechanismBusy = true;
        }
        if (mechanismBusy && currentGoal != "NO_KNOWN_PROGRESS")
        {
            if (currentGoal != "WAIT_MOVING_MECHANISM")
            {
                setGoal("WAIT_MOVING_MECHANISM", observation.sector);
                event("waiting_for_mechanism", "reason=known_moving_geometry");
                mechanismWaitTick = observation.tick;
            }
            else if (mechanismWaitTick >= 0
                && observation.tick - mechanismWaitTick > 3 * kTicsPerSec)
                mechanismBusy = false;
            if (mechanismBusy)
                return composeInput(move, combat, use);
        }

        // NO_KNOWN_PROGRESS is terminal only after bounded, executable
        // affordances have been tried.  This is a deterministic local probe,
        // not random roaming: it preserves the previously demonstrated
        // start-area jump without encoding any sector or wall identity.
        if (!localDynamicJumpAttempted && localJumpProbeCount < 4)
        {
            setGoal("SEARCH_CURRENT_AREA", -2);
            if (searchAngle < 0)
                searchAngle = observation.angle;
            // Fix the endpoint when the probe starts.  Recomputing it from
            // where the player is standing turns a bounded probe into an
            // infinite ray that retreats exactly as fast as he walks, so it
            // never ends and never rotates to the next heading; the bot just
            // shuttles between rooms it has already seen.
            if (!searchProbeActive)
            {
                searchProbeActive = true;
                searchProbeX = observation.x + mulscale30(Cos(searchAngle), 8192);
                searchProbeY = observation.y + mulscale30(Sin(searchAngle), 8192);
                searchProbeStartTick = observation.tick;
                char detail[160];
                snprintf(detail, sizeof(detail),
                         "heading=%d to_x=%d to_y=%d probe=%d reason=bounded_unexplained_blocked_frontier",
                         searchAngle, searchProbeX, searchProbeY, localJumpProbeCount + 1);
                event("local_jump_probe", detail);
            }
            // The probe survives being interrupted -- combat, a door, a
            // moment of the goal being something else.  Re-issuing the
            // movement target is fine; recomputing where the probe was going
            // is not, or an interruption every second or two silently
            // restarts the same heading for ever.
            if (!movementTargetActive || movementTargetGoal != currentGoal
                || movementTargetId != -2)
                setMovementTarget(searchProbeX, searchProbeY, observation.sector, -2,
                                  kTraversalJumpable);
            // A probe that has run its time, or arrived, is finished: turn to
            // the next heading rather than pushing at the same one.
            const int arrive = std::max(512, playerClipRadius() * 4);
            if (observation.tick - searchProbeStartTick > 6 * kTicsPerSec
                || distance2(observation.x, observation.y, searchProbeX, searchProbeY)
                    <= arrive * arrive)
            {
                searchProbeActive = false;
                movementTargetActive = false;
                ++localJumpProbeCount;
                searchAngle = wrapAngle(searchAngle + 512);
                char detail[128];
                snprintf(detail, sizeof(detail), "next_heading=%d probe=%d reason=probe_finished",
                         searchAngle, localJumpProbeCount + 1);
                event("local_jump_probe_reoriented", detail);
                return composeInput(move, combat, use);
            }
            return composeInput(intentFromInput(steerTo(searchProbeX, searchProbeY,
                                                        false, false, -2,
                                                        observation.sector,
                                                        kTraversalJumpable)), combat, use);
        }

        dumpLedger("no_known_progress");
        refreshDerivedFrontiers();
        setGoal("NO_KNOWN_PROGRESS", observation.sector);
        if (!explorationSnapshotEmitted)
        {
            char detail[224];
            snprintf(detail, sizeof(detail),
                     "reason=NO_KNOWN_PROGRESS sector=%d open=%u blocked=%u interactions=%u objects=%u",
                     observation.sector, unsigned(derivedFrontiers.size()),
                     unsigned(investigatedBoundaries.size()),
                     unsigned(interactions.size()), unsigned(objectMemory.size()));
            event("exploration_stuck_snapshot", detail);
            explorationSnapshotEmitted = true;
        }
        return composeInput(move, combat, use);
    }

    void detectStall()
    {
        if (!explorationSnapshotEmitted
            && observation.tick - lastSemanticProgressTick > 10 * kTicsPerSec)
        {
            ensureNavTopology();
            refreshDerivedFrontiers();
            int localFrontiers = 0;
            int remoteFrontiers = 0;
            for (const DerivedFrontier &frontier : derivedFrontiers)
            {
                bool local = false;
                for (size_t i = 0; i < frontier.candidates.size(); ++i)
                    if (frontier.candidates[i].from == observation.sector)
                        local = true;
                if (local)
                    ++localFrontiers;
                else
                    ++remoteFrontiers;
            }
            char detail[256];
            snprintf(detail, sizeof(detail),
                     "reason=NO_KNOWN_PROGRESS sector=%d cells=%u objective=%d local_frontiers=%d remote_frontiers=%d interactions=%u objects=%u",
                     observation.sector, unsigned(navCells.size()),
                     int(currentObjective.type), localFrontiers, remoteFrontiers,
                     unsigned(interactions.size()), unsigned(objectMemory.size()));
            event("exploration_stuck_snapshot", detail);
            explorationSnapshotEmitted = true;
        }
        if (observation.tick - lastSemanticProgressTick > stallSeconds * kTicsPerSec)
        {
            result = "STALLED";
            failureReason = "no meaningful world or knowledge progress";
            event("failure", failureReason.c_str());
            gQuitGame = true;
            return;
        }

        if (currentGoal != "WAIT_MOVING_MECHANISM" && currentGoal != "NO_KNOWN_PROGRESS")
        {
        const int quantized = (observation.x >> 10) ^ ((observation.y >> 10) << 10) ^ (observation.sector << 20);
        if (quantized == lastStateLocation && currentGoal == lastStateGoal
            && currentGoalTarget == lastStateTarget && knowledgeRevision == lastStateKnowledge
            && inventoryRevision == lastStateInventory)
            ++repeatedStateCount;
        else
            repeatedStateCount = 0;
        lastStateLocation = quantized;
        lastStateGoal = currentGoal;
        lastStateTarget = currentGoalTarget;
        lastStateKnowledge = knowledgeRevision;
        lastStateInventory = inventoryRevision;
        if (repeatedStateCount > 3 * kTicsPerSec)
        {
            repeatedStateCount = 0;
            // A detected loop is a reason to change what the bot is doing,
            // not a reason to end the run.  Retire the work it is circling
            // on, hard, and let selection pick something else.  Only give up
            // after several escalations have failed to change anything.
            ++loopBreakCount;
            char detail[224];
            snprintf(detail, sizeof(detail),
                     "attempt=%d sector=%d goal=%s target=%d objective=%d",
                     loopBreakCount, observation.sector, currentGoal.c_str(),
                     currentGoalTarget, int(currentObjective.type));
            event("loop_break", detail);
            if (currentObjective.active)
            {
                const int key = objectiveKey(currentObjective);
                suppressObjective(key, "loop_break");
                suppressObjective(key, "loop_break");
                invalidateObjective("loop_break", false);
            }
            resetNavigation();
            lastStateGoal.clear();
            lastStateLocation = -1;
            if (loopBreakCount >= kMaxLoopBreaks)
            {
                result = "LOOP_DETECTED";
                failureReason = "repeated position/goal/knowledge state";
                event("failure", failureReason.c_str());
                gQuitGame = true;
            }
        }
        }
    }

    void close(const char *reason)
    {
        if (!result.size())
            result = reason ? reason : "RUNTIME_ERROR";
        fprintf(stderr,
                "LLMAPPER BOT TERMINATION result=%s reason=%s game_time=%d goal=%s target=%d sector=%d\n",
                result.c_str(), failureReason.c_str(), (gFrame * kTicsPerFrame) / kTicsPerSec,
                currentGoal.c_str(), currentGoalTarget, observation.sector);
        fflush(stderr);
        if (telemetry)
        {
            fprintf(telemetry, "{\"type\":\"summary\",\"result\":\"%s\",\"failure_reason\":\"%s\",\"game_time\":%d,\"visited_sectors\":%u,\"observed_sectors\":%u}\n",
                    result.c_str(), failureReason.c_str(), (gFrame * kTicsPerFrame) / kTicsPerSec,
                    unsigned(visitedSectors.size()), unsigned(observedSectors.size()));
            fclose(telemetry);
            telemetry = nullptr;
        }
        if (trajectory)
        {
            fclose(trajectory);
            trajectory = nullptr;
        }
        if (gDemo.at0)
            gDemo.Close();
    }
};

LLMapperBot::LLMapperBot()
    : m_impl(new Impl), m_enabled(false), m_fast(true), m_visible(false)
{
}

LLMapperBot::~LLMapperBot()
{
    Finish("RUNTIME_ERROR");
    delete m_impl;
}

void LLMapperBot::Enable(const char *telemetry, const char *trajectory, const char *demo)
{
    m_enabled = true;
    if (telemetry && *telemetry)
        m_impl->telemetryPath = telemetry;
    if (trajectory && *trajectory)
        m_impl->trajectoryPath = trajectory;
    if (demo && *demo)
        m_impl->demoPath = demo;
}

void LLMapperBot::ConfigureTimeout(int seconds)
{
    if (seconds > 0)
        m_impl->timeoutSeconds = seconds;
}

void LLMapperBot::ConfigureStallTimeout(int seconds)
{
    if (seconds > 0)
        m_impl->stallSeconds = seconds;
}

void LLMapperBot::SetFast(bool fast)
{
    m_fast = fast;
}

void LLMapperBot::SetVisible(bool visible)
{
    m_visible = visible;
    if (visible)
        m_fast = false;
}

void LLMapperBot::PrepareLaunch()
{
    if (!m_enabled)
        return;
    m_impl->openFiles();
    m_impl->event("run_started");
    if (!gDemo.at0 && !gDemo.at1 && !gDemo.Create(m_impl->demoPath.c_str()))
    {
        m_impl->result = "RUNTIME_ERROR";
        m_impl->failureReason = "could not create demo file";
        m_impl->event("runtime_error", m_impl->failureReason.c_str());
        gQuitGame = true;
    }
}

GINPUT LLMapperBot::GetInput()
{
    if (!m_enabled)
        return {};
    if (m_impl->lastObservationTick != gFrame * kTicsPerFrame)
    {
        m_impl->updateKnowledge();
        m_impl->lastObservationTick = gFrame * kTicsPerFrame;
    }
    return m_impl->decide();
}

void LLMapperBot::OnFrame()
{
    if (!m_enabled || !gGameStarted || !gMe || !gMe->pXSprite)
        return;
    m_impl->updateKnowledge();
    m_impl->lastObservationTick = gFrame * kTicsPerFrame;
    if (gMe->pSprite->extra > 0 && gMe->pSprite->extra < kMaxXSprites)
    {
        const int moveHit = gSpriteHit[gMe->pSprite->extra].hit;
        const int wallHit = (moveHit & 0xc000) == 0x8000 ? moveHit & 0x3fff : -1;
        if (wallHit != m_impl->lastEngineMoveHit)
        {
            m_impl->lastEngineMoveHit = wallHit;
            if (wallHit >= 0)
            {
                char detail[160];
                snprintf(detail, sizeof(detail), "wall=%d sector=%d x=%d y=%d hit=0x%x goal=%s target=%d",
                         wallHit, m_impl->observation.sector, int(gMe->pSprite->x), int(gMe->pSprite->y),
                         moveHit, m_impl->currentGoal.c_str(), m_impl->lastStateTarget);
                m_impl->event("engine_move_wall_hit", detail);
            }
        }
    }
    if ((gGameOptions.uGameFlags & kGameFlagContinuing) && m_impl->result.empty())
        OnLevelExit(kLevelExitNormal);
    if (m_impl->lastTrajectoryTick < gFrame * kTicsPerFrame - kTrajectoryPeriod)
    {
        m_impl->trajectorySample();
        m_impl->lastTrajectoryTick = gFrame * kTicsPerFrame;
    }
    if (m_impl->observation.tick % (kObservationPeriod * kTicsPerFrame) == 0)
        m_impl->detectStall();
    if (m_impl->observation.tick >= m_impl->timeoutSeconds * kTicsPerSec)
    {
        m_impl->result = "TIMEOUT";
        m_impl->failureReason = "simulated time limit reached";
        m_impl->event("failure", m_impl->failureReason.c_str());
        gQuitGame = true;
    }
    if (m_impl->observation.health == 0 && m_impl->result.empty())
    {
        m_impl->result = "DIED";
        m_impl->failureReason = "player health reached zero";
        m_impl->event("failure", m_impl->failureReason.c_str());
        gQuitGame = true;
    }
}

void LLMapperBot::OnActionResolved(int hit, int target, int extra, bool accepted, int key)
{
    if (m_enabled)
        m_impl->actionResolved(hit, target, extra, accepted, key);
}

void LLMapperBot::OnBotDamaged(int source, int damageType, int amount)
{
    if (!m_enabled || !m_impl)
        return;
    m_impl->lastAttacker = source;
    m_impl->lastAttackerDamageType = damageType;
    m_impl->lastDamageTick = m_impl->observation.tick;
    char detail[128];
    snprintf(detail, sizeof(detail), "source=%d type=%d amount=%d", source, damageType, amount);
    m_impl->event("bot_damaged", detail);
}

void LLMapperBot::OnLevelExit(int exitType)
{
    if (!m_enabled || !m_impl->result.empty())
        return;
    m_impl->result = "COMPLETED";
    char detail[48];
    snprintf(detail, sizeof(detail), "exit_type=%d", exitType);
    m_impl->event("level_completed", detail);
    gQuitGame = true;
}

void LLMapperBot::DrawStatus()
{
    if (!m_enabled || !m_visible || !m_impl || !gGameStarted)
        return;

    const int seconds = (gFrame * kTicsPerFrame) / kTicsPerSec;
    const LLMapperBot::Impl &bot = *m_impl;
    char line[128];
    int y = 4;
    const int x = 2;

    // Line 1: the clock, so anything seen on screen can be named by time.
    snprintf(line, sizeof(line), "T %d:%02d  sect %d  depth %d  seen %d",
             seconds / 60, seconds % 60, bot.observation.sector,
             bot.sectorDepth(bot.observation.sector),
             int(bot.visitedSectors.size()));
    viewDrawText(3, line, x, y, -128, 0, 0, true, 256);
    y += 8;

    // Line 2: what it is trying to do and why.
    const char *reason = bot.mission.reason ? bot.mission.reason : "-";
    snprintf(line, sizeof(line), "%s", bot.currentGoal.empty()
             ? reason : bot.currentGoal.c_str());
    viewDrawText(3, line, x, y, -128, 0, 0, true, 256);
    y += 8;

    // Line 3: the object of that intent, and how far off it is.
    if (bot.currentObjective.active)
    {
        int anchorX = 0;
        int anchorY = 0;
        const int distance = bot.objectiveAnchor(anchorX, anchorY)
            ? int(std::sqrt(double(distance2(bot.observation.x, bot.observation.y,
                                             anchorX, anchorY))))
            : -1;
        snprintf(line, sizeof(line), "obj t%d id%d ->s%d d%d %ds",
                 int(bot.currentObjective.type), bot.currentObjective.id,
                 bot.currentObjective.targetSector, distance,
                 bot.currentObjective.startedTick >= 0
                     ? (bot.observation.tick - bot.currentObjective.startedTick) / kTicsPerSec
                     : 0);
    }
    else
        snprintf(line, sizeof(line), "obj none  reason %s", reason);
    viewDrawText(3, line, x, y, -128, 0, 0, true, 256);
    y += 8;

    // Line 4: the shape of what it still has to do.
    int pending = 0;
    int dormant = 0;
    for (size_t i = 0; i < bot.ledger.size(); ++i)
    {
        if (bot.ledger[i].hops < 0)
            continue;
        if (bot.ledger[i].dormantUntil > bot.observation.tick)
            ++dormant;
        else
            ++pending;
    }
    snprintf(line, sizeof(line), "todo %d  asleep %d  keys %u  hp %d",
             pending, dormant, unsigned(bot.heldKeys.size()), bot.observation.health / 16);
    viewDrawText(3, line, x, y, -128, 0, 0, true, 256);
    y += 8;

    // Line 5: only when something is actively in the bot's way.
    const int quiet = bot.observation.tick - bot.lastSemanticProgressTick;
    if (quiet > 3 * kTicsPerSec)
    {
        snprintf(line, sizeof(line), "no progress %ds", quiet / kTicsPerSec);
        viewDrawText(3, line, x, y, -128, 0, 0, true, 256);
    }
}

void LLMapperBot::Finish(const char *reason)
{
    if (m_enabled && m_impl->telemetry)
        m_impl->close(reason);
}

LLMapperBot gLLMapperBot;
