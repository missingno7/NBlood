//-------------------------------------------------------------------------
// LLMapper autonomous Blood playtest bot.
//
// This is deliberately a small in-process vertical slice. It keeps a
// discovered graph, plans only over observations, and emits NDJSON telemetry
// plus a trajectory while driving the ordinary GINPUT path.
//-------------------------------------------------------------------------
#include "bot.h"
#include "nav_kernel.h"
#include "player_capability.h"

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
#include "../../actor.h"
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
#include "../../controls.h"
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
using llmapper::SupportRef;
using llmapper::kSupportSectorFloor;
using llmapper::kSupportSpriteFloor;
using llmapper::WorldObjectRef;
using llmapper::kWorldSector;
using llmapper::kWorldWall;
using llmapper::kWorldSprite;
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
//
// A cheap geometric rejection only.  ClipMove stays the authority on whether
// the body actually gets through.
static int playerPassageWidth()
{
    return llmapper::capability::playerEnvelope().radius * 2;
}
constexpr int kMaxWalkableStep = 4096;
constexpr int kActionScanRange = 1024;
constexpr int kActionApproachRange = 2048;
constexpr int kUseStopRange = kActionApproachRange;
constexpr int kInteractionTimeoutTicks = 6 * kTicsPerSec;
// Added to the engine's real damaging radius.  This is clearance for Caleb's
// body plus a small uncertainty allowance, not a guessed explosion size.
constexpr int kExplosiveSafetyMargin = 256;
constexpr int kExplosiveThrowAllowance = 1536;
constexpr int kExplosivePrimeTicks = kTicsPerSec / 3;
constexpr int kExplosiveOutcomeTicks = 6 * kTicsPerSec;
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
constexpr int kSupportPoseSettleTicks = kTicsPerSec / 2;
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
// Decisions a landing is given to slide clear of the lip before the bot
// judges whether the crossing was made.
constexpr int kGapLandingSettleDecisions = 4;
// Divergences between the motion model and the engine worth reporting.  If
// the model is wrong the first few say why; after that it is noise.
constexpr int kMotionAuditReports = 12;
// How far off the bot's heading a target may be and still be worth a shot
// taken in passing.  256 of 2048 is 45 degrees.
constexpr int kOpportunisticAimCone = 256;
// Clearance the bot leaves inside the pitchfork's actual reach, so a swing
// taken from the edge of it still connects after a frame of drift.
constexpr int kMeleeReachMargin = 128;

// Full forward input.
//
// The control layer clamps forward and strafe to playerMoveInputMax(), which
// is 2048 and is exactly what a running keyboard press produces, so that is
// the engine's answer.  The bot holds one unit under it, and not for any
// reason in Blood: raising it by that single unit moves the bot along a
// fractionally different line, and over forty seconds that was enough for
// AGTST6 to stop completing.  Its route to the exit turns out to depend on
// incidentally clipping the corner of one sector on the way past, and once
// the bot no longer clips it the frontiers on the far side of the sprite
// bridge become unroutable and are dropped from the ledger without a word.
// The constant is not the defect.  That dependence is, and it is not fixed.
static const int kFullThrottle = llmapper::capability::playerMoveInputMax() - 1;

// How far the pitchfork actually reaches, less the margin above.
static int meleeReach()
{
    return llmapper::capability::playerPitchforkReach() - kMeleeReachMargin;
}

// Health below which the run is in trouble: a quarter of what Caleb starts
// with, on Blood's own 16x scale rather than a restatement of it.
static int criticalHealth()
{
    return llmapper::capability::playerStartHealth() / 4;
}

enum ObjectKind
{
    kObjectEnemy,
    kObjectKey,
    kObjectInteractive,
    kObjectDamageable,
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
    unsigned acceptedEffects = llmapper::kEffectNone;
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
    int blockerSprite = -1;
    // A crossing made through the air over a gap rather than through a
    // doorway.  The two sectors do not touch; the takeoff is on this side of
    // the drop and the landing is the far lip.
    bool gapJump = false;
    int takeoffX = 0;
    int takeoffY = 0;
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
    llmapper::ActivationMode activationMode = llmapper::kActivateUse;
    unsigned requiredEffects = llmapper::kEffectNone;
    int causalWall = -1;
    int causalFrom = -1;
    int causalTo = -1;
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
    return angle & kAngMask;
}

static int angleDelta(int target, int current)
{
    return DANGLE(target, current);
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

// How much height the player can gain by jumping.  Replayed from the
// engine's own impulse and gravity, and it follows Jump Boots, so there is
// exactly one account of Caleb's jump in the bot.
static int playerJumpRiseLimit()
{
    return llmapper::capability::playerJumpApex();
}

// How far the player may step down and still be able to get back up.  This
// is a reversibility question, not a survival one: a drop the bot cannot
// climb out of turns a wrong turn into the end of the run.
static int playerReversibleDrop()
{
    return playerJumpRiseLimit();
}

static int playerStandingClearance();

// The player's body as it is right this frame, mid-animation included.  Only
// the calibration asks this: everything the bot records about the world has
// to be stated against a body that does not change under it, or crouching
// once makes every doorway in the level appear to open and shut again.
static int playerBodyClearance()
{
    const llmapper::capability::Envelope body = llmapper::capability::playerEnvelope();
    return std::max(1, body.height());
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

// Until the bot has actually crouched, it does not know how short crouching
// makes it.  The posture table's eyeAboveZ is a camera height, not a
// collision envelope, and guessing from it invents clearance the body may
// not have.  Unknown is reported as the standing envelope, which only ever
// makes the bot refuse a gap it might have fitted through.
static int playerCrouchClearance()
{
    const int standing = playerStandingClearance();
    if (gObservedCrouchClearance > 0)
        return std::min(standing, gObservedCrouchClearance);
    return standing;
}

static void playerCollisionDistances(int &ceilingDistance, int &floorDistance)
{
    const llmapper::capability::Envelope body = llmapper::capability::playerEnvelope();
    ceilingDistance = std::max(0, body.ceilingDistance);
    floorDistance = std::max(0, body.floorDistance);
}

static int lookAngleForTarget(int eyeZ, int targetZ, int horizontal)
{
    const double angle = std::atan2(double(eyeZ - targetZ),
                                    double(std::max(1, horizontal)))
        * 1024.0 / 3.14159265358979323846;
    return std::max(llmapper::capability::playerLookDownLimit(),
                    std::min(llmapper::capability::playerLookUpLimit(),
                             int(std::lround(angle))));
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

// Blood does not use q16look as a geometric weapon pitch. ProcessInput
// derives horiz=100*tan(look), then slope=-horiz*128, and VectorScan applies
// slope at 1/1024 per horizontal map unit. Account for that 12.5x scale for
// ranged activation; otherwise rounds hit the ceiling/floor before reaching
// a vertically displaced target. Keep interaction posing separate because
// ActionScan has its own short-range acquisition behavior.
static int vectorLookAngleForTarget(int eyeZ, int targetZ, int horizontal)
{
    const double angle = std::atan2(double(eyeZ - targetZ) * 8.0,
                                    double(std::max(1, horizontal)) * 100.0)
        * 1024.0 / 3.14159265358979323846;
    return std::max(llmapper::capability::playerLookDownLimit(),
                    std::min(llmapper::capability::playerLookUpLimit(),
                             int(std::lround(angle))));
}

struct StandableSurface
{
    SupportRef support;
    int z = 0;
    int ceilingZ = 0;
};

// GetSpriteExtents is Blood's visual/gameplay extent helper.  Build collision
// applies the tile y-offset through spriteheightofs(..., 1), which can differ
// by a small but decisive amount: probing the visual top may start just below
// the collision top and make getzrange report the sector floor.  Navigation
// needs the plane Caleb's body collides with.
static void spriteCollisionExtents(int spriteId, int &top, int &bottom)
{
    if (!inRange(spriteId, 0, kMaxSprites))
    {
        top = bottom = 0;
        return;
    }
    int height = 0;
    bottom = sprite[spriteId].z + spriteheightofs(spriteId, &height, 1);
    top = bottom - height;
}

// Enumerate every floor the engine can resolve at one XY, probing from just
// above each candidate plane.  One query from the sector ceiling only finds
// the topmost sprite and collapses stacked bridges; probing each known plane
// preserves every physically distinct support while still leaving GetZRange
// authoritative on footprint, one-sidedness, slope and clipping.  Build's
// getzrange treats the top extent of blocking face/wall sprites as a floor as
// well as floor-aligned sprites.  Use that same physical answer here instead
// of maintaining a narrower orientation-specific idea of support.
static std::vector<StandableSurface> standableSurfacesAt(
    int sectorId, int x, int y, int radius, int requiredClearance)
{
    std::vector<StandableSurface> result;
    if (!inRange(sectorId, 0, numsectors))
        return result;

    auto probe = [&](const SupportRef &support, int surfaceZ, int expectedHit) {
        int ceilingZ = 0, ceilingHit = 0, floorZ = 0, floorHit = 0;
        GetZRangeAtXYZ(x, y, surfaceZ - 1, sectorId,
                       &ceilingZ, &ceilingHit, &floorZ, &floorHit,
                       radius, CLIPMASK0,
                       PARALLAXCLIP_CEILING | PARALLAXCLIP_FLOOR);
        if (floorZ != surfaceZ)
            return;
        if (expectedHit >= 0 && floorHit != expectedHit)
            return;
        if (surfaceZ - ceilingZ < requiredClearance)
            return;
        for (const StandableSurface &known : result)
            if (known.support == support && known.z == surfaceZ)
                return;
        StandableSurface surface;
        surface.support = support;
        surface.z = surfaceZ;
        surface.ceilingZ = ceilingZ;
        result.push_back(surface);
    };

    const int sectorFloor = getflorzofslope(sectorId, x, y);
    probe(SupportRef(kSupportSectorFloor, sectorId), sectorFloor, -1);
    for (int nSprite = headspritesect[sectorId]; nSprite >= 0;
         nSprite = nextspritesect[nSprite])
    {
        const spritetype &record = sprite[nSprite];
        if (gMe && gMe->pSprite && nSprite == gMe->pSprite->index)
            continue;
        if (!(record.cstat & CSTAT_SPRITE_BLOCK))
            continue;
        const int alignment = record.cstat & CSTAT_SPRITE_ALIGNMENT_MASK;
        int surfaceZ = 0;
        if (alignment == CSTAT_SPRITE_ALIGNMENT_FLOOR
            || alignment == CSTAT_SPRITE_ALIGNMENT_SLOPE)
            surfaceZ = spriteGetZOfSlope(uint16_t(nSprite), { x, y });
        else
        {
            int bottom = 0;
            spriteCollisionExtents(nSprite, surfaceZ, bottom);
        }
        probe(SupportRef(kSupportSpriteFloor, nSprite), surfaceZ, 0xc000 | nSprite);
    }
    return result;
}

static SupportRef currentPlayerSupport()
{
    if (!gMe || !gMe->pSprite)
        return SupportRef();
    int ceilingZ = 0, ceilingHit = 0, floorZ = 0, floorHit = 0;
    GetZRange(gMe->pSprite, &ceilingZ, &ceilingHit, &floorZ, &floorHit,
              (gMe->pSprite->clipdist << 2) + 16, CLIPMASK0,
              PARALLAXCLIP_CEILING | PARALLAXCLIP_FLOOR);
    if ((floorHit & 0xc000) == 0xc000)
        return SupportRef(kSupportSpriteFloor, floorHit & 0x3fff);
    return SupportRef(kSupportSectorFloor, gMe->pSprite->sectnum);
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

// The two ends of a wall-aligned sprite, as clipmove computes them.  A wall
// sprite is a line, not a post: a panel across a doorway is several hundred
// units wide, and treating it as a clipdist-sized circle at its centre
// misses everything but the middle.
//
// This deliberately does not call Blood's own GetSpriteExtents, which
// answers a slightly different question: it mirrors the x offset on
// CSTAT_SPRITE_YFLIP, while the engine's get_wallspr_dims -- the one
// clipmove actually clips with -- mirrors on CSTAT_SPRITE_XFLIP.  The bot is
// asking what the player's body will hit, so clipmove is the authority.
static void wallSpriteSpan(const spritetype &record, int &x1, int &y1,
                           int &x2, int &y2)
{
    const int span = tilesiz[record.picnum].x;
    const int offset = (record.cstat & CSTAT_SPRITE_XFLIP)
        ? -(picanm[record.picnum].xofs + record.xoffset)
        : (picanm[record.picnum].xofs + record.xoffset);
    const int dax = sintable[record.ang & kAngMask] * record.xrepeat;
    const int day = sintable[(record.ang + kAng90 + kAng180) & kAngMask] * record.xrepeat;
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
        spriteCollisionExtents(nSprite, top, bottom);
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

// How far the player travels through the air, given how much height the
// landing gains or loses.  Blood adds `normalJumpZ` as the initial vertical
// velocity and 58254 per tick of gravity, so the flight time is exact; the
// horizontal reach is that time at a deliberately modest running speed, well
// under what the bot actually manages, because a jump planned too long ends
// in the pit.
// Which way a wall faces, away from the sector that owns it.  Derived by
// asking the engine which side of the wall is inside, so no assumption about
// Build's winding order is baked in.
static bool wallOutwardNormal(int x1, int y1, int x2, int y2, int ownerSector,
                              double &nx, double &ny)
{
    const double dx = double(x2 - x1);
    const double dy = double(y2 - y1);
    const double length = std::sqrt(dx * dx + dy * dy);
    if (length < 1.0)
        return false;
    nx = dy / length;
    ny = -dx / length;
    const int midX = (x1 + x2) / 2;
    const int midY = (y1 + y2) / 2;
    if (inside(midX + int(nx * 64), midY + int(ny * 64), ownerSector) == 1)
    {
        nx = -nx;
        ny = -ny;
    }
    return true;
}

static int bodyRadiusOf()
{
    return llmapper::capability::playerEnvelope().radius;
}

// Blood stores explosion radii in the EXPLOSION table in 1/16 map units;
// CheckProximity shifts XY deltas by four before comparing them.  Keep that
// conversion next to the engine table so navigation never grows a second,
// guessed set of TNT/barrel distances.
static int engineExplosionRadius(int explosionType)
{
    if (!inRange(explosionType, 0, kExplosionMax))
        return 0;
    return int(explodeInfo[explosionType].radius) << 4;
}

static int explosiveWeaponRadius(int weapon)
{
    switch (weapon)
    {
    case kWeaponTNT:
        return engineExplosionRadius(kExplosionStandard);
    case kWeaponNapalm:
        // The projectile and its secondary radius-damage path are both
        // engine-defined; the larger is the conservative deliberate-action
        // envelope.
        return std::max(engineExplosionRadius(kExplosionNapalm), 128 << 4);
    default:
        return 0;
    }
}

static int explosiveSpriteRadius(const spritetype &record)
{
    switch (record.type)
    {
    case kThingArmedTNTStick:
        return engineExplosionRadius(kExplosionSmall);
    case kThingArmedProxBomb:
    case kThingArmedRemoteBomb:
    case kThingArmedTNTBundle:
        return engineExplosionRadius(kExplosionStandard);
    case kThingTNTBarrel:
        return engineExplosionRadius(kExplosionLarge);
    default:
        return 0;
    }
}

static int safeExplosionSeparation(int damagingRadius)
{
    return damagingRadius > 0
        ? damagingRadius + bodyRadiusOf() + kExplosiveSafetyMargin : 0;
}

static int jumpAirTicks(int rise)
{
    return llmapper::capability::playerJumpAirFrames(rise);
}

static int jumpReach(int rise)
{
    return jumpAirTicks(rise) * llmapper::capability::playerRunSpeed();
}

// How far apart two segments come, and where in the middle of that closest
// stretch to cross.
//
// The nearest single pair of points sits on a corner whenever the two edges
// run parallel, which is the usual case for a gap: aiming at a corner leaves
// no ground behind the lip to run up on and no ground in front to land on.
// Average the whole set of near-minimum pairs instead, which puts both ends
// of the jump in the middle of the ledge.
static int64_t closestApproach(int ax1, int ay1, int ax2, int ay2,
                               int bx1, int by1, int bx2, int by2,
                               int &outAx, int &outAy, int &outBx, int &outBy)
{
    const int steps = 16;
    int64_t best = INT64_MAX;
    for (int pass = 0; pass < 2; ++pass)
    {
        int64_t sumAx = 0, sumAy = 0, sumBx = 0, sumBy = 0;
        int found = 0;
        for (int i = 0; i <= steps; ++i)
        {
            const int px = ax1 + int(int64_t(ax2 - ax1) * i / steps);
            const int py = ay1 + int(int64_t(ay2 - ay1) * i / steps);
            for (int j = 0; j <= steps; ++j)
            {
                const int qx = bx1 + int(int64_t(bx2 - bx1) * j / steps);
                const int qy = by1 + int(int64_t(by2 - by1) * j / steps);
                const int64_t d = int64_t(px - qx) * (px - qx)
                    + int64_t(py - qy) * (py - qy);
                if (pass == 0)
                {
                    best = std::min(best, d);
                    continue;
                }
                // A tenth over the minimum still counts as the closest
                // stretch; that is wide enough to cover a parallel pair and
                // narrow enough to exclude the rest of the wall.
                if (d <= best + best / 10 + 1024)
                {
                    sumAx += px; sumAy += py;
                    sumBx += qx; sumBy += qy;
                    ++found;
                }
            }
        }
        if (pass == 1 && found > 0)
        {
            outAx = int(sumAx / found);
            outAy = int(sumAy / found);
            outBx = int(sumBx / found);
            outBy = int(sumBy / found);
        }
    }
    return best;
}

// A hole in the floor with something on the other side of it is a crossing,
// even though the two sides do not touch.  Blood levels are full of them --
// a chasm with pillars standing in it, a broken walkway -- and a bot that
// only knows how to walk through doorways reads the far lip as unreachable
// and gives up in front of it.
//
// The chasm is a sector the bot refuses to step into because the drop is
// unsurvivable.  The landing is any sector on the far side of that chasm
// whose floor stands clear of it, close enough to reach through the air.
static void appendGapCrossings(Observation &result)
{
    const size_t doorways = result.portals.size();
    const int bodyClearance = playerStandingClearance();
    const int riseLimit = playerJumpRiseLimit();
    for (size_t index = 0; index < doorways; ++index)
    {
        const Portal ledge = result.portals[index];
        // Only a drop the bot will not simply take is a gap worth jumping.
        if (ledge.to < 0 || ledge.walkable || ledge.crouchable || ledge.dropSafe)
            continue;
        if (ledge.floorDelta <= kMaxWalkableStep)
            continue;
        const int chasm = ledge.to;
        if (!inRange(chasm, 0, numsectors))
            continue;
        const int chasmFloor = getflorzofslope(chasm, ledge.x, ledge.y);
        const sectortype &pit = sector[chasm];
        for (int i = 0; i < pit.wallnum; ++i)
        {
            const int wallId = pit.wallptr + i;
            if (!inRange(wallId, 0, numwalls) || !inRange(wall[wallId].point2, 0, numwalls))
                continue;
            const int landingSector = wall[wallId].nextsector;
            if (!inRange(landingSector, 0, numsectors) || landingSector == chasm
                || landingSector == result.sector)
                continue;
            if (wall[wallId].cstat & 1)
                continue;
            const walltype &edge = wall[wallId];
            const walltype &edgeEnd = wall[edge.point2];
            int takeoffX = 0, takeoffY = 0, landingX = 0, landingY = 0;
            const int64_t span2 = closestApproach(
                ledge.x1, ledge.y1, ledge.x2, ledge.y2,
                edge.x, edge.y, edgeEnd.x, edgeEnd.y,
                takeoffX, takeoffY, landingX, landingY);
            // Aim past the lip, not at it: landing on the very edge is how
            // a jump that cleared the gap slides back off it.
            {
                const int64_t intoX = int64_t(landingX) - takeoffX;
                const int64_t intoY = int64_t(landingY) - takeoffY;
                const double run = std::sqrt(double(intoX * intoX + intoY * intoY));
                const int step = bodyRadiusOf() + 128;
                if (run > 1.0)
                {
                    const int px = landingX + int(intoX * step / run);
                    const int py = landingY + int(intoY * step / run);
                    if (inside(px, py, landingSector) == 1)
                    {
                        landingX = px;
                        landingY = py;
                    }
                }
            }
            const int landingFloor = getflorzofslope(landingSector, landingX, landingY);
            // The far side has to stand clear of the chasm, or this is the
            // same low ground reached the long way round rather than a gap.
            if (chasmFloor - landingFloor <= kMaxWalkableStep)
                continue;
            // A crossing may go up, stay level, or come down a little: one
            // pillar in a row is rarely the same height as the next.  Both
            // directions are bounded by the same jump, though.  Up, by how
            // high it reaches; down, by how far the bot could climb back --
            // land further below than that and this was never a crossing,
            // it was a fall into the hole that happened to have a floor.
            const int rise = landingFloor - ledge.fromFloorZ;
            if (-rise > riseLimit || rise > playerReversibleDrop())
                continue;
            const int reach = jumpReach(rise) - bodyRadiusOf();
            if (reach <= 0 || span2 > int64_t(reach) * reach)
                continue;
            // A sliver of a ledge is not a landing.
            const int landingWidth = int(std::sqrt(double(distance2(
                edge.x, edge.y, edgeEnd.x, edgeEnd.y))));
            if (landingWidth < playerPassageWidth())
                continue;
            // A jump leaves a ledge across it and arrives at the far one
            // head on.  Without that, any two edges of the same pit look
            // like a crossing -- including a leap off the side of a pillar
            // aimed lengthways down the chasm, which flies over the very
            // ledge it was supposed to land on.
            const double flightX = double(landingX - takeoffX);
            const double flightY = double(landingY - takeoffY);
            const double flight = std::sqrt(flightX * flightX + flightY * flightY);
            if (flight < 1.0)
                continue;
            double outX = 0.0, outY = 0.0;
            if (!wallOutwardNormal(ledge.x1, ledge.y1, ledge.x2, ledge.y2,
                                   result.sector, outX, outY))
                continue;
            if ((flightX * outX + flightY * outY) / flight < 0.5)
                continue;
            double faceX = 0.0, faceY = 0.0;
            if (!wallOutwardNormal(edge.x, edge.y, edgeEnd.x, edgeEnd.y,
                                   landingSector, faceX, faceY))
                continue;
            if ((flightX * faceX + flightY * faceY) / flight > -0.5)
                continue;
            // The line has to run over the hole rather than through the
            // masonry beside it, with room overhead to travel it.
            const int midX = (takeoffX + landingX) / 2;
            const int midY = (takeoffY + landingY) / 2;
            if (inside(midX, midY, chasm) != 1)
                continue;
            if (landingFloor - getceilzofslope(chasm, midX, midY) < bodyClearance)
                continue;
            if (landingFloor - getceilzofslope(landingSector, landingX, landingY) < bodyClearance)
                continue;

            Portal crossing;
            crossing.wall = wallId;
            crossing.from = result.sector;
            crossing.to = landingSector;
            crossing.x = landingX;
            crossing.y = landingY;
            crossing.z = landingFloor;
            crossing.floorZ = landingFloor;
            crossing.ceilingZ = getceilzofslope(landingSector, landingX, landingY);
            crossing.fromFloorZ = ledge.fromFloorZ;
            crossing.fromCeilingZ = ledge.fromCeilingZ;
            crossing.toFloorZ = landingFloor;
            crossing.toCeilingZ = crossing.ceilingZ;
            crossing.x1 = edge.x;
            crossing.y1 = edge.y;
            crossing.x2 = edgeEnd.x;
            crossing.y2 = edgeEnd.y;
            crossing.openingWidth = landingWidth;
            crossing.floorDelta = rise;
            crossing.clearance = std::min(ledge.fromFloorZ - ledge.fromCeilingZ,
                                          landingFloor - crossing.ceilingZ);
            crossing.jumpRiseLimit = riseLimit;
            crossing.visible = ledge.visible;
            crossing.localGeometry = true;
            crossing.jumpable = true;
            crossing.traversable = true;
            crossing.capability = kTraversalJumpable;
            crossing.gapJump = true;
            crossing.takeoffX = takeoffX;
            crossing.takeoffY = takeoffY;
            crossing.currentlyAvailable = true;
            result.portals.push_back(crossing);
        }
    }
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

// Translate the engine's accepted damage classes into effects the planner
// can reason about.  This deliberately asks THINGINFO rather than naming
// wall-crack sprites: any live thing whose damage table accepts explosion
// damage exposes the same abstract opportunity.
static unsigned acceptedDamageEffects(const spritetype &record)
{
    if (record.statnum != kStatThing
        || record.type < kThingBase || record.type >= kThingMax
        || !validXSprite(record.extra) || xsprite[record.extra].health <= 0)
        return llmapper::kEffectNone;
    // Damageability alone is not a progression affordance (a thrown charge
    // is damageable too).  Destruction must either remove solid collision or
    // dispatch a mapper-authored world change.  This keeps volatile effect
    // producers available to a future satisfier search without mistaking
    // them for the blocked traversal itself.
    if (!(record.cstat & 1) && xsprite[record.extra].txID == 0)
        return llmapper::kEffectNone;
    const THINGINFO &info = thingInfo[record.type - kThingBase];
    unsigned effects = llmapper::kEffectNone;
    if (info.dmgControl[kDamageExplode] > 0)
        effects |= llmapper::kEffectExplosive;
    return effects;
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
            // is a bridge plank when one lies across the doorway.  Do not
            // query exactly on the shared wall: a floor sprite ending at the
            // boundary is deliberately ambiguous there and GetZRange can
            // return the pit below it.  Sample a short distance into each
            // sector, where the player's centre actually stands while
            // crossing the threshold.
            auto interiorSample = [&](int sectorId, int &sampleX, int &sampleY) {
                sampleX = midX;
                sampleY = midY;
                const int normalX = -(nextWall.y - wallRecord.y);
                const int normalY = nextWall.x - wallRecord.x;
                const int length = std::max(1, int(std::sqrt(double(
                    int64_t(normalX) * normalX + int64_t(normalY) * normalY))));
                const int radius = gMe && gMe->pSprite
                    ? (gMe->pSprite->clipdist << 2) : 128;
                // The player's centre cannot occupy the first clip-radius
                // band beside a wall.  Sampling shallower than that still
                // lands on the ambiguous bevel at the end of a bridge
                // sprite, especially on its one-sided edge.
                const int depth = std::max(384, std::min(512, radius * 3));
                for (int direction = -1; direction <= 1; direction += 2)
                {
                    const int x = midX + direction * normalX * depth / length;
                    const int y = midY + direction * normalY * depth / length;
                    if (inside(x, y, sectorId) == 1)
                    {
                        sampleX = x;
                        sampleY = y;
                        return;
                    }
                }
            };
            int fromSampleX = midX, fromSampleY = midY;
            int toSampleX = midX, toSampleY = midY;
            interiorSample(result.sector, fromSampleX, fromSampleY);
            interiorSample(wallRecord.nextsector, toSampleX, toSampleY);
            const int fromFloor = standingFloorZ(result.sector,
                                                  fromSampleX, fromSampleY);
            const int toFloor = standingFloorZ(wallRecord.nextsector,
                                                toSampleX, toSampleY);
            const int fromCeiling = getceilzofslope(result.sector,
                                                     fromSampleX, fromSampleY);
            const int toCeiling = getceilzofslope(wallRecord.nextsector,
                                                   toSampleX, toSampleY);
            const int openingWidth = int(std::sqrt(double(distance2(wallRecord.x, wallRecord.y, nextWall.x, nextWall.y))));
            const int clearance = std::min(fromFloor - fromCeiling, toFloor - toCeiling);
            const int floorDelta = toFloor - fromFloor;
            const int bodyClearance = playerStandingClearance();
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
                std::vector<int> operableObstructions;
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
                        const spritetype &blocker = sprite[hit];
                        if (validXSprite(blocker.extra)
                            && (xsprite[blocker.extra].Push
                                || xsprite[blocker.extra].Vector))
                            appendUnique(operableObstructions, hit);
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
                    portal.blockerSprite = obstruction;
                    // Blocked, but not necessarily hopeless: a barricade the
                    // player can operate is progression work, not a wall.
                    if (!operableObstructions.empty())
                        portal.interactionAffordance = true;
                }

                // A sprite barricade is operated as a sprite, not as the
                // otherwise ordinary wall portal it happens to cover.  Keep
                // the approach on the reachable side of the obstruction;
                // a long-range visibility observation of the same sprite in
                // the unopened destination sector must not make the useful
                // action look unreachable.
                if (portal.blockedBySprite)
                {
                    for (int blockerId : operableObstructions)
                    {
                        const spritetype &blocker = sprite[blockerId];
                        const XSPRITE &extra = xsprite[blocker.extra];
                        InteractionCandidate candidate;
                        candidate.kind = kInteractionSprite;
                        candidate.id = blockerId;
                        candidate.fromSector = result.sector;
                        candidate.targetSector = portal.to;
                        candidate.x = blocker.x;
                        candidate.y = blocker.y;
                        candidate.z = blocker.z;
                        candidate.reversible = true;
                        candidate.activationMode = extra.Vector
                            ? llmapper::kActivateVector : llmapper::kActivateUse;
                        candidate.target.wall = -1;
                        candidate.target.from = result.sector;
                        candidate.target.to = portal.to;
                        candidate.target.x = blocker.x;
                        candidate.target.y = blocker.y;
                        candidate.target.shootable = extra.Vector != 0;
                        candidate.target.mechanismTx = extra.txID;
                        candidate.target.blockedBySprite = true;
                        candidate.target.blockerSprite = blockerId;
                        setSpriteInteractionGeometry(candidate.target, blockerId);
                        candidate.z = candidate.target.z;
                        result.interactions.push_back(candidate);
                    }
                }
            }
            if (sector[portal.from].extra > 0 && sector[portal.from].extra < kMaxXSectors)
                portal.sectorPushCurrent = xsector[sector[portal.from].extra].Push != 0;
            // Preserve a sprite affordance discovered by the collision scan.
            // Assignment here used to erase it, leaving a portal correctly
            // classified as sprite-blocked but with no actionable blocker.
            portal.interactionAffordance = portal.interactionAffordance
                || portal.wallPush || portal.sectorPush
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
                candidate.activationMode = llmapper::kActivateVector;
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

    appendGapCrossings(result);

    // The scan is over the engine's sprite list, but only objects passing the
    // authoritative cansee() test enter the bot's observation.
    for (int i = 0; i < kMaxSprites; ++i)
    {
        spritetype &candidate = sprite[i];
        const bool isEnemy = candidate.statnum == kStatDude;
        const bool isItem = candidate.statnum == kStatItem;
        const bool isThing = candidate.statnum == kStatThing;
        const bool isSwitch = candidate.type >= kSwitchBase && candidate.type < kSwitchMax;
        const unsigned damageEffects = acceptedDamageEffects(candidate);
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
        if ((isThing || isSwitch) && !operable && damageEffects == llmapper::kEffectNone)
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
                         // Explicit mapper-authored Use/Vector semantics are
                         // more specific than inferred damageability.  A
                         // solid actuator may also have health; replacing
                         // its Vector contract with ExplosiveEffect would
                         // discard the actual satisfier the map supplied.
                         : operable ? kObjectInteractive
                         : damageEffects != llmapper::kEffectNone ? kObjectDamageable
                         : (isThing || isSwitch) ? kObjectInteractive
                         : kObjectPickup;
        object.acceptedEffects = damageEffects;
        result.objects.push_back(object);
        if ((object.kind == kObjectInteractive && validXSprite(candidate.extra)
             && (xsprite[candidate.extra].Push || xsprite[candidate.extra].Vector))
            || object.kind == kObjectDamageable)
        {
            InteractionCandidate interaction;
            interaction.kind = kInteractionSprite;
            interaction.id = i;
            // The sprite's containing sector is the stable approach region.
            // A sprite close enough to touch from across a doorway may use
            // the observer's side, but mere long-range visibility must not
            // continually rewrite the remembered approach to wherever the
            // player happens to be standing.
            // Vector activation is explicitly a remote interaction.  The
            // cansee() result above proves that the current physical layer
            // is a valid firing region even when the actuator belongs to a
            // different sector.  Requiring its containing sector first can
            // strand a shootable trigger behind the very transition it
            // controls (AGTST11).  Use remains local and keeps the stable
            // containing-sector approach unless it is actually in reach.
            const bool remoteVector = validXSprite(candidate.extra)
                && xsprite[candidate.extra].Vector;
            interaction.fromSector = remoteVector
                ? result.sector
                : distance2(result.x, result.y, candidate.x, candidate.y)
                    <= kActionScanRange * kActionScanRange
                ? result.sector : candidate.sectnum;
            interaction.targetSector = candidate.sectnum;
            interaction.x = candidate.x;
            interaction.y = candidate.y;
            interaction.z = candidate.z;
            interaction.reversible = object.kind != kObjectDamageable;
            interaction.activationMode = object.kind == kObjectDamageable
                ? llmapper::kActivateDamage
                : xsprite[candidate.extra].Vector
                    ? llmapper::kActivateVector : llmapper::kActivateUse;
            interaction.requiredEffects = object.kind == kObjectDamageable
                ? object.acceptedEffects : unsigned(llmapper::kEffectNone);
            interaction.target.wall = -1;
            interaction.target.from = candidate.sectnum;
            interaction.target.to = candidate.sectnum;
            interaction.target.x = candidate.x;
            interaction.target.y = candidate.y;
            interaction.target.shootable = xsprite[candidate.extra].Vector != 0;
            interaction.target.mechanismTx = xsprite[candidate.extra].txID;
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
    case kObjectDamageable: return "damageable";
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
        // Keep one route family for the lifetime of an interaction objective.
        // 1 uses the exact support mesh; 2 uses the broader directed sector
        // transport graph.  This prevents layered geometry from changing the
        // route policy merely because the actor crossed a sector boundary.
        int interactionRouteMode = 0;
        // Some world actions are only useful if Caleb is carried by the
        // mechanism they start.  This is a physical precondition of the
        // action, independent of which wall/sprite/sector acts as its
        // actuator.  -1 means the action may be performed from its ordinary
        // approach side (for example, calling a lift to the landing).
        int requiredSupportSector = -1;
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
        llmapper::ActivationMode activationMode = llmapper::kActivateUse;
        unsigned requiredEffects = llmapper::kEffectNone;
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
        int consecutiveNoEffectAttempts = 0;
        int causalWall = -1;
        int causalFrom = -1;
        int causalTo = -1;
        bool recoveryAttempted = false;
        bool traversed = false;
        bool activationPoseOnReceiver = false;
        int effectPhase = 0;
        int effectStartedTick = -1;
        int effectWeapon = 0;
        int effectAmmoBefore = -1;
        int effectProjectile = -1;
        int effectBlastRadius = 0;
        int vectorShotsResolved = 0;
        int confirmedVectorHits = 0;
        int firstVectorHitTick = -1;
        bool firingSolutionReported = false;
        bool blockedFiringSolutionReported = false;
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
    int dynamicPrerequisiteMechanism = -1;
    int dynamicPrerequisiteState = -1;
    Observation observation;
    std::set<int> observedSectors;
    std::set<int> visitedSectors;
    std::map<int, DoorMemory> doors;
    std::map<int, InteractionMemory> interactions;
    std::set<int> learnedWallMechanisms;
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
    bool hazardEscapeUnavailableEmitted = false;
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
    int lastRangedCapability = -1;
    bool rangedPrerequisitePending = false;
    unsigned lastEffectCapabilities = ~0u;
    unsigned pendingEffectPrerequisites = llmapper::kEffectNone;
    int lastObservedSector = -1;
    int lastGeometryTelemetrySector = -1;
    bool worldLoadedTelemetryEmitted = false;
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
        int z = 0;
        SupportRef support;
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
    std::map<int64_t, std::vector<int>> navCellIndex;
    std::vector<NavCell> navCells;
    int navTopologySignature = 0;
    int navPoseSignatureValue = 0;
    int navDynamicSignatureValue = 0;
    int navTopologyRevision = 0;
    bool rearmFrontiersOnNextTopologyRebuild = false;
    int navPoseChangedTick = 0;
    int navPoseRebuildTick = 0;
    bool navMeshInMotion = false;
    int lastDynamicPrerequisiteProbeSignature = 0;
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
    std::set<int> reportedSpriteSupports;
    std::set<int> reportedRejectedSpriteSupports;
    std::set<int> reportedSpriteSupportLinks;
    std::set<int64_t> reportedSpriteSupportTakeoffs;
    std::set<int> reportedRaisedPortalAudits;
    std::set<int64_t> reportedRaisedPortalRoutes;
    bool reportedNavStartReconnect = false;
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
    std::set<int64_t> visitedSupportPoses;
    int64_t pendingSupportPose = INT64_MIN;
    int pendingSupportPoseTick = -1;
    int supportRideSettleUntil = -1;
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
    mutable std::map<int, std::vector<WorldObjectRef>> mechanismReceivers;
    mutable std::vector<int> routeGateWall;
    mutable std::vector<int> routeGateSector;
    std::map<int, int> inertBoundaries;
    std::set<int> narrowSectors;
    std::set<int> transitOnlySectors;
    // A wall can expose different connections at different support poses.
    // Collapsing those alternatives to one bit made a stacked bridge's open
    // layer and closed layer overwrite one another every observation tick.
    std::map<int64_t, bool> boundaryOpen;
    std::map<int64_t, int> boundaryClearance;
    int lastCrossingWall = -1;
    int lastFrontierChoice = 0;
    int directCrossingWall = -1;
    int directCrossingBlockedTicks = 0;
    // Motion-model audit state (see recordMotionPrediction).
    llmapper::capability::MotionState motionAuditPredicted;
    llmapper::capability::MotionState motionAuditBefore;
    int motionAuditAngle = 0;
    int motionAuditFoot = 0;
    bool motionAuditValid = false;
    bool motionAuditJumped = false;
    int motionAuditFrame = -1;
    int motionAuditForward = 0;
    int motionAuditFloorZ = 0;
    int motionAuditAgreed = 0;
    int motionAuditDiverged = 0;
    int motionAuditClipped = 0;
    int motionAuditReshaped = 0;
    int motionAuditBuoyant = 0;
    int motionAuditPushed = 0;
    bool motionAuditDepth = false;

    bool gapJumpActive = false;
    bool gapRunning = false;
    int gapSettleUntilTick = -1;
    int gapRunWall = -1;
    int gapJumpWall = -1;
    int gapJumpFrom = -1;
    int gapJumpTo = -1;
    int gapJumpX = 0;
    int gapJumpY = 0;
    int gapJumpFloorZ = 0;
    int gapJumpSettling = 0;
    int gapJumpTick = -1;
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
    int recentInvestigatedWall = -1;
    int recentInvestigatedFrom = -1;
    int recentInvestigatedTo = -1;
    int recentInvestigatedTick = -1;

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
                else if (key > 0 && !hasKey(key))
                {
                    // ActionScan is authoritative about why the activation
                    // was refused.  The map record visible to navigation may
                    // not advertise its key beforehand (sprite exit switches
                    // are one example), so learn the abstract prerequisite
                    // from the real Use result and keep the action pending.
                    // This is not a failed interaction: it is useful work that
                    // cannot be performed with the current inventory.
                    memory.key = key;
                    memory.locked = true;
                    memory.state = 0;
                    memory.attempted = false;
                    memory.activated = false;
                    const int opportunity = interactionOpportunityId(entry.first);
                    suppressedUntil.erase(opportunity);
                    suppressionEvidence.erase(opportunity);
                    suppressionCount.erase(opportunity);
                    char prerequisite[160];
                    snprintf(prerequisite, sizeof(prerequisite),
                             "kind=%d id=%d required_key=%d reason=engine_rejected_use",
                             int(memory.kind), memory.id, key);
                    event("interaction_prerequisite_discovered", prerequisite);
                    event("opportunity_deferred", prerequisite);
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
    // Every object listening on this channel.  Blood's event bucket contains
    // XSECTOR, XWALL and XSPRITE receivers and dispatches to all of them; the
    // bot mirrors that causal fan-out instead of silently choosing the first
    // receiving sector.
    const std::vector<WorldObjectRef> &mechanismReceiversFor(int tx) const
    {
        static const std::vector<WorldObjectRef> empty;
        if (tx <= 0)
            return empty;
        auto known = mechanismReceivers.find(tx);
        if (known != mechanismReceivers.end())
            return known->second;
        std::vector<WorldObjectRef> found;
        for (int i = 0; i < numsectors; ++i)
        {
            const int extra = sector[i].extra;
            if (extra > 0 && extra < kMaxXSectors && xsector[extra].rxID == tx)
                found.push_back(WorldObjectRef(kWorldSector, i));
        }
        for (int i = 0; i < numwalls; ++i)
        {
            const int extra = wall[i].extra;
            if (extra > 0 && extra < kMaxXWalls && xwall[extra].rxID == tx)
                found.push_back(WorldObjectRef(kWorldWall, i));
        }
        for (int i = 0; i < kMaxSprites; ++i)
        {
            if (sprite[i].statnum >= kMaxStatus || !validXSprite(sprite[i].extra))
                continue;
            if (xsprite[sprite[i].extra].rxID == tx)
                found.push_back(WorldObjectRef(kWorldSprite, i));
        }
        mechanismReceivers[tx] = found;
        return mechanismReceivers.find(tx)->second;
    }

    void vectorResolved(int hit, int sectorId, int wallId, int spriteId,
                        int x, int y, int z)
    {
        if (currentGoal != "VECTOR_ACTIVATE" || !currentObjective.active
            || currentObjective.type != kObjectiveInteraction)
            return;
        auto found = interactions.find(currentObjective.interactionKey);
        if (found == interactions.end())
            return;
        InteractionMemory &memory = found->second;
        if (memory.activationMode != llmapper::kActivateVector)
            return;
        ++memory.vectorShotsResolved;
        const bool intended = (memory.kind == kInteractionSprite
                               && hit == 3 && spriteId == memory.id)
            || (memory.kind == kInteractionWall && (hit == 0 || hit == 4)
                && (wallId == memory.id || wallId == memory.target.wall));
        if (!intended)
            return;

        ++memory.confirmedVectorHits;
        if (memory.firstVectorHitTick < 0)
            memory.firstVectorHitTick = observation.tick;
        memory.attempted = true;
        memory.activated = true;
        memory.state = 2;
        memory.observedLocalEffect = true;
        memory.afterState = interactionStateSignature(memory);
        lastSemanticProgressTick = observation.tick;
        recoveryMode = false;

        int tx = memory.target.mechanismTx;
        int health = -1;
        int vectorFlag = -1;
        if (memory.kind == kInteractionSprite && inRange(memory.id, 0, kMaxSprites)
            && validXSprite(sprite[memory.id].extra))
        {
            const XSPRITE &record = xsprite[sprite[memory.id].extra];
            if (tx <= 0)
                tx = record.txID;
            health = record.health;
            vectorFlag = record.Vector;
        }
        const int ammoSlot = gMe ? gMe->curWeapon - 1 : -1;
        const int ammo = gMe && ammoSlot >= 0
            && ammoSlot < int(sizeof(gMe->ammoCount) / sizeof(gMe->ammoCount[0]))
            ? gMe->ammoCount[ammoSlot] : -1;
        char detail[320];
        snprintf(detail, sizeof(detail),
                 "kind=%d instance=%d tx=%d hit=%d sector=%d wall=%d sprite=%d at=(%d,%d,%d) weapon=%d ammo=%d resolved=%d confirmed=%d health=%d vector=%d",
                 int(memory.kind), memory.id, tx, hit, sectorId, wallId, spriteId,
                 x, y, z, gMe ? gMe->curWeapon : 0, ammo,
                 memory.vectorShotsResolved, memory.confirmedVectorHits,
                 health, vectorFlag);
        event("ranged_interaction_confirmed", detail);
    }

    int mechanismReceiver(int tx) const
    {
        const std::vector<WorldObjectRef> &receivers = mechanismReceiversFor(tx);
        for (const WorldObjectRef &receiver : receivers)
            if (receiver.kind == kWorldSector)
                return receiver.id;
        return -1;
    }

    int mechanismSector(const Portal &target, int targetSector) const
    {
        // A channel names the mechanism outright, whichever surface is used.
        if (target.mechanismTx > 0)
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
        // Several wall records can be faces of one physical actuator.  Keep
        // that actuator canonical while reserving a distinct namespace from
        // current-sector Push; the old shared key let the landing control
        // overwrite the actuator reachable on the carrier itself.
        if (candidate.kind == kInteractionWall)
        {
            const int mechanism = mechanismSector(candidate.target,
                                                  candidate.targetSector);
            if (mechanism >= 0)
                return 4000000 + mechanism + 1;
        }
        return int(candidate.kind) * 1000000 + candidate.id + 1;
    }

    int interactionMemoryKey(const InteractionMemory &memory) const
    {
        if (memory.kind == kInteractionWall)
        {
            const int mechanism = mechanismSector(memory.target,
                                                  memory.targetSector);
            if (mechanism >= 0)
                return 4000000 + mechanism + 1;
        }
        return int(memory.kind) * 1000000 + memory.id + 1;
    }

    // Ledger identities must not overlap the pickup/frontier namespaces.
    // interactionMemoryKey is already a structured key and may itself be in
    // the 4,000,000 range, so adding only 1,000,000 aliases real sprite IDs.
    static int interactionOpportunityId(int interactionKey)
    {
        return 10000000 + interactionKey;
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
            signature = signature * 31 + spriteRecord.statnum;
            signature = signature * 31 + spriteRecord.type;
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
        const int horizontal = int(std::sqrt(double(distance2(
            observation.x, observation.y, memory.target.x, memory.target.y))));
        // Wall-sprite switches are hitscan-selected by ActionScan. Their
        // transparent border is not an actionable surface, so aiming at the
        // closest vertical edge can remain perfectly aligned yet never name
        // the sprite. Aim at the visible centre and use Blood's real slope
        // conversion, just as for a Vector hit. Other interaction kinds keep
        // their established short-range approach behavior.
        if (memory.kind == kInteractionSprite && inRange(memory.id, 0, kMaxSprites)
            && (sprite[memory.id].cstat & CSTAT_SPRITE_ALIGNMENT_MASK)
                == CSTAT_SPRITE_ALIGNMENT_WALL)
        {
            const int targetZ = int((int64_t(memory.target.interactionTopZ)
                                    + int64_t(memory.target.interactionBottomZ)) / 2);
            return vectorLookAngleForTarget(eyeZ, targetZ, horizontal);
        }
        const int targetZ = lower <= upper ? std::max(lower, std::min(upper, eyeZ))
                                           : memory.target.z;
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

    void shareLearnedWallEffect(const InteractionMemory &source)
    {
        if (source.kind != kInteractionWall)
            return;
        const int driven = mechanismSector(source.target, source.targetSector);
        if (driven < 0)
            return;
        learnedWallMechanisms.insert(driven);
        for (auto &entry : interactions)
        {
            InteractionMemory &sibling = entry.second;
            if (!sibling.observed || sibling.kind != kInteractionWall
                || sibling.id == source.id
                || mechanismSector(sibling.target, sibling.targetSector) != driven)
                continue;
            // These remain separate actuators/approach poses.  What is shared
            // is only the learned causal fact that their common receiver has
            // answered, preventing the planner from toggling one mechanism
            // once per surrounding wall.  Sector Push is intentionally not a
            // sibling: it is the actuator reachable after the ride.
            sibling.attempted = true;
            sibling.activated = true;
            sibling.state = 2;
            sibling.observedLocalEffect = true;
            sibling.observedKnownWorldDelta = true;
            sibling.consecutiveNoEffectAttempts = 0;
            sibling.afterState = interactionStateSignature(sibling);
        }
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
        const bool storedApproachReachable = memory.observed
            && (memory.fromSector == observation.sector
                || visitedSectors.count(memory.fromSector));
        const bool candidateApproachReachable = candidate.fromSector == observation.sector
            || visitedSectors.count(candidate.fromSector);
        // A pushable sprite seen through its closed portal is reported by
        // two layers: the portal collision scan supplies the reachable side,
        // while the general visibility scan supplies the sprite's containing
        // sector.  Preserve the executable pose until that containing sector
        // really becomes reachable.
        const bool preserveReachableSpriteApproach = memory.observed
            && candidate.kind == kInteractionSprite
            && storedApproachReachable && !candidateApproachReachable;
        // Vector visibility proves that the current sector is *a* firing
        // region; it does not make every sector seen while walking past the
        // new canonical one.  Keep the first reachable firing layer stable.
        // This matters for mechanisms observed from a moving support: a later
        // glimpse from below must not erase the remembered aboard pose.  A
        // trigger first discovered remotely (AGTST11) still records that
        // remote sector on the first observation.
        const bool preserveRemoteVectorApproach = memory.observed
            && candidate.kind == kInteractionSprite
            && candidate.activationMode == llmapper::kActivateVector
            && storedApproachReachable
            // The containing sector is only the discovery default.  Replace
            // it once with the first proven remote firing layer (for example
            // while aboard a platform), then keep that remote layer stable.
            && memory.fromSector != memory.targetSector
            && memory.fromSector != candidate.fromSector;
        const bool sideChanged = memory.observed
            && memory.fromSector != candidate.fromSector
            && !preserveReachableSpriteApproach
            && !preserveRemoteVectorApproach;
        // Among several surfaces driving one mechanism, the useful one is the
        // one the player can reach and face right now.
        const bool closerSurface = memory.observed && candidate.kind == kInteractionWall
            && candidate.target.wallPush && memory.target.wallPush
            && candidate.target.wall != memory.target.wall
            && distance2(observation.x, observation.y, candidate.x, candidate.y)
                < distance2(observation.x, observation.y, memory.x, memory.y);
        memory.targetSector = candidate.targetSector >= 0
            ? candidate.targetSector : candidateMechanism;
        // A reversible mechanism is usable from more than one side, and the
        // side that matters is the one the bot is standing on now.  Keeping
        // the first-seen approach point sent the bot at a pose it could no
        // longer reach once the door it came through closed behind it.
        if (!preserveReachableSpriteApproach && !preserveRemoteVectorApproach
            && (first || !canonicalSectorPush || sideChanged || closerSurface))
        {
            memory.fromSector = candidate.fromSector;
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
        // Observation can omit a prerequisite that the authoritative Use
        // path subsequently revealed.  Do not erase learned causal knowledge
        // merely because the same visible sprite/wall is sampled again.
        if (candidate.key > 0 || memory.key == 0)
            memory.key = candidate.key;
        memory.locked = candidate.locked || memory.locked;
        memory.reversible = candidate.reversible;
        memory.activationMode = candidate.activationMode;
        memory.requiredEffects = candidate.requiredEffects;
        if (candidate.causalWall >= 0)
        {
            memory.causalWall = candidate.causalWall;
            memory.causalFrom = candidate.causalFrom;
            memory.causalTo = candidate.causalTo;
        }
        memory.observed = true;
        memory.target = candidate.target;
        const int receiver = mechanismSector(candidate.target,
                                             candidate.targetSector);
        if (receiver >= 0 && candidate.fromSector == receiver)
            memory.activationPoseOnReceiver = true;
        if (first && candidate.kind == kInteractionWall
            && candidateMechanism >= 0
            && learnedWallMechanisms.count(candidateMechanism))
        {
            memory.attempted = true;
            memory.activated = true;
            memory.state = 2;
            memory.observedLocalEffect = true;
            memory.observedKnownWorldDelta = true;
            memory.consecutiveNoEffectAttempts = 0;
            memory.afterState = interactionStateSignature(memory);
        }
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
            memory.consecutiveNoEffectAttempts = 0;
            recoveryMode = false;
            memory.afterState = newState;
            char detail[128];
            snprintf(detail, sizeof(detail), "kind=%d id=%d before=%d after=%d", int(memory.kind), memory.id,
                     memory.beforeState, newState);
            event("interaction_world_delta", detail);
            shareLearnedWallEffect(memory);
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
            char detail[224];
            snprintf(detail, sizeof(detail),
                     "kind=%d id=%d approach_sector=%d target_sector=%d key=%d mode=%d at=(%d,%d,%d) blocker_sprite=%d",
                     int(memory.kind), memory.id, memory.fromSector,
                     memory.targetSector, memory.key, int(memory.activationMode),
                     memory.x, memory.y, memory.z, memory.target.blockerSprite);
            event("discovered_interaction", detail);
            const unsigned missing = memory.requiredEffects
                & ~availableEffectCapabilities();
            if (missing)
            {
                char deferred[176];
                snprintf(deferred, sizeof(deferred),
                         "kind=%d id=%d useful_action=open_traversal missing_effects=%u",
                         int(memory.kind), memory.id, missing);
                event("opportunity_deferred", deferred);
                event("action_prerequisite_unavailable", deferred);
            }
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
        if (!worldLoadedTelemetryEmitted)
        {
            char detail[64];
            snprintf(detail, sizeof(detail), "total_sectors=%d", numsectors);
            event("world_loaded", detail);
            worldLoadedTelemetryEmitted = true;
        }
        const int collisionWall = currentMoveWallHit();
        if (collisionWall >= 0)
            rememberCollisionInteraction(collisionWall);
        if (gMe->posture == kPostureCrouch)
        {
            // Keep the shortest crouching frame seen.  Blood's crouch is a
            // hold, not a cycle: the sequence plays down to a settled
            // envelope and stays there, so the smallest sample is the body
            // the bot will actually have while it holds the crouch and every
            // taller one is the animation on the way into it.  Keeping the
            // tallest instead measures the transition and costs the bot
            // slots it fits through perfectly well.
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
        const int liveFloor = llmapper::capability::playerFloorZ();
        const SupportRef liveSupport = currentPlayerSupport();
        // MoveDude's height is the engine's grounded authority.  Sprite art
        // extents are not the collision feet (the standing body bottom is
        // thousands of Z units above the floor), so comparing those values
        // silently discarded valid support landings.
        if (gMe && gMe->pXSprite && gMe->pXSprite->height == 0
            && stableSupportPose(liveSupport, liveFloor))
        {
            const SupportRef support = liveSupport;
            const int supportZ = liveFloor;
            const int64_t pose = supportPoseKey(support, supportZ);
            if (pendingSupportPose != pose)
            {
                pendingSupportPose = pose;
                pendingSupportPoseTick = observation.tick;
            }
            if (observation.tick - pendingSupportPoseTick >= kSupportPoseSettleTicks
                && visitedSupportPoses.insert(pose).second)
            {
                ++knowledgeRevision;
                lastSemanticProgressTick = observation.tick;
                char detail[160];
                snprintf(detail, sizeof(detail),
                         "sector=%d support_kind=%d support=%d z=%d",
                         observation.sector, int(support.kind), support.id, supportZ);
                event("nav_surface_entered", detail);
            }
        }
        else
        {
            pendingSupportPose = INT64_MIN;
            pendingSupportPoseTick = -1;
        }
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
            for (int spriteId = headspritesect[observation.sector]; spriteId >= 0;
                 spriteId = nextspritesect[spriteId])
            {
                const spritetype &record = sprite[spriteId];
                if (!(record.cstat & CSTAT_SPRITE_BLOCK))
                    continue;
                int top = record.z, bottom = record.z;
                GetSpriteExtents(&record, &top, &bottom);
                const bool extended = validXSprite(record.extra);
                char spriteDetail[320];
                snprintf(spriteDetail, sizeof(spriteDetail),
                         "sprite=%d type=%d stat=%d picnum=%d voxel=%d repeat=(%d,%d) pos=(%d,%d,%d) cstat=%d alignment=%d clipdist=%d top=%d bottom=%d push=%d vector=%d",
                         spriteId, int(record.type), int(record.statnum),
                         int(record.picnum), inRange(record.picnum, 0, MAXTILES)
                             ? int(tiletovox[record.picnum]) : -1,
                         int(record.xrepeat), int(record.yrepeat),
                         int(record.x), int(record.y), int(record.z), int(record.cstat),
                         int(record.cstat) & CSTAT_SPRITE_ALIGNMENT_MASK,
                         record.clipdist << 2, top, bottom,
                         extended && xsprite[record.extra].Push ? 1 : 0,
                         extended && xsprite[record.extra].Vector ? 1 : 0);
                event("local_solid_sprite", spriteDetail);
            }
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
        if (busyValueInMotion(observation.localSectorBusy)
            && observation.playerZVelocity != 0)
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
            {
                edges.push_back(portal);
                // The observed far-side landing is now eligible for the
                // one-hop support mesh.  Rebuild even if the enclosing
                // sector/wall identity itself did not change.
                navTopologySignature = 0;
            }
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
                const bool reverseStanding = reverse.clearance >= playerStandingClearance();
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

        // A destroyed sprite leaves the observation list immediately, so it
        // cannot report its own final state through rememberInteraction().
        // Close the transaction from authoritative sprite/health state.
        for (auto &entry : interactions)
        {
            InteractionMemory &memory = entry.second;
            if (memory.activationMode != llmapper::kActivateDamage
                || memory.state != 1 || memory.kind != kInteractionSprite)
                continue;
            const bool stillDamageable = inRange(memory.id, 0, kMaxSprites)
                && (acceptedDamageEffects(sprite[memory.id])
                    & memory.requiredEffects) != 0;
            if (stillDamageable)
                continue;
            memory.state = 2;
            memory.activated = true;
            memory.observedLocalEffect = true;
            memory.observedKnownWorldDelta = true;
            memory.consecutiveNoEffectAttempts = 0;
            memory.afterState = interactionStateSignature(memory);
            ++knowledgeRevision;
            lastSemanticProgressTick = observation.tick;
            char detail[144];
            snprintf(detail, sizeof(detail),
                     "kind=%d id=%d effect=%u reason=damageable_removed",
                     int(memory.kind), memory.id, memory.requiredEffects);
            event("interaction_world_delta", detail);
            event("deferred_opportunity_completed", detail);
        }

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
                char detail[320];
                const char *category = itemCategory(object.type);
                const spritetype &record = sprite[object.sprite];
                int top = record.z, bottom = record.z;
                GetSpriteExtents(&record, &top, &bottom);
                const bool operable = validXSprite(record.extra)
                    && (xsprite[record.extra].Push || xsprite[record.extra].Vector);
                snprintf(detail, sizeof(detail),
                         "sprite=%d kind=%s category=%s sector=%d type=%d pos=(%d,%d,%d) cstat=%d alignment=%d block=%d clipdist=%d top=%d bottom=%d push=%d vector=%d",
                         object.sprite, objectName(object.kind), category ? category : "none",
                         object.sector, object.type, int(record.x), int(record.y),
                         int(record.z), int(record.cstat),
                         int(record.cstat) & CSTAT_SPRITE_ALIGNMENT_MASK,
                         (record.cstat & CSTAT_SPRITE_BLOCK) ? 1 : 0,
                         record.clipdist << 2, top, bottom,
                         operable && xsprite[record.extra].Push ? 1 : 0,
                         operable && xsprite[record.extra].Vector ? 1 : 0);
                event("observed_object", detail);
                lastSemanticProgressTick = observation.tick;
                ++knowledgeRevision;
            }
            if (firstObjectObservation)
                event("object_remembered",
                      object.kind == kObjectKey ? "kind=key"
                      : object.kind == kObjectDamageable ? "kind=damageable"
                      : "kind=pickup");
        }

        heldKeyMask = 0;
        for (int key = 1; key < 8; ++key)
        {
            if (!gMe->hasKey[key])
                continue;
            heldKeyMask |= 1u << unsigned(key);
            if (!heldKeys.insert(key).second)
                continue;
            // A new key re-arms every remembered opportunity that wanted it,
            // whether the gated object is a door wall, a sprite switch, or a
            // sector interaction.  The engine's rejected Use may be the first
            // place the prerequisite became observable.
            int rearmedDoors = 0;
            for (const auto &door : lockedDoorKey)
            {
                if (door.second != key)
                    continue;
                suppressedUntil.erase(3000000 + door.first * 8 + int(kObjectiveFrontier));
                suppressionCount.erase(3000000 + door.first * 8 + int(kObjectiveFrontier));
                ++rearmedDoors;
            }
            int rearmedInteractions = 0;
            for (auto &entry : interactions)
            {
                InteractionMemory &memory = entry.second;
                if (memory.key != key || memory.engineAccepted)
                    continue;
                memory.state = 0;
                memory.attempted = false;
                memory.activated = false;
                memory.unavailableAttempts = 0;
                memory.unavailableFromSector = -1;
                const int opportunity = interactionOpportunityId(entry.first);
                suppressedUntil.erase(opportunity);
                suppressionEvidence.erase(opportunity);
                suppressionCount.erase(opportunity);
                ++rearmedInteractions;
            }
            char detail[128];
            snprintf(detail, sizeof(detail),
                     "key=%d rearmed_doors=%d rearmed_interactions=%d",
                     key, rearmedDoors, rearmedInteractions);
            event("acquired_key", detail);
            event("key_acquired", detail);
            lastSemanticProgressTick = observation.tick;
            ++inventoryRevision;
        }

        // Capability changes are evidence just like acquiring a key.  A
        // previously remembered action whose required support had no melee
        // pose becomes actionable as soon as a ranged weapon is collected.
        int rangedWeapon = 0;
        const int rangedCapability = rangedWeaponAvailable(rangedWeapon) ? 1 : 0;
        if (lastRangedCapability < 0)
            lastRangedCapability = rangedCapability;
        else if (lastRangedCapability != rangedCapability)
        {
            lastRangedCapability = rangedCapability;
            if (rangedCapability)
                rangedPrerequisitePending = false;
            ++inventoryRevision;
            char detail[96];
            snprintf(detail, sizeof(detail), "available=%d weapon=%d",
                     rangedCapability, rangedWeapon);
            event("ranged_capability_changed", detail);
        }

        // Re-derive missing prerequisites from remembered useful actions.
        // This is the deferred-opportunity link: it persists independently
        // of whichever pickup or world mechanism may eventually satisfy it.
        const unsigned effectCapabilities = availableEffectCapabilities();
        unsigned missingEffects = llmapper::kEffectNone;
        for (const auto &entry : interactions)
        {
            const InteractionMemory &memory = entry.second;
            if (!memory.observed || memory.state == 2)
                continue;
            missingEffects |= memory.requiredEffects & ~effectCapabilities;
        }
        pendingEffectPrerequisites = missingEffects;
        if (lastEffectCapabilities == ~0u)
            lastEffectCapabilities = effectCapabilities;
        else if (lastEffectCapabilities != effectCapabilities)
        {
            const unsigned gained = effectCapabilities & ~lastEffectCapabilities;
            lastEffectCapabilities = effectCapabilities;
            ++inventoryRevision;
            int rearmed = 0;
            for (const auto &entry : interactions)
            {
                const InteractionMemory &memory = entry.second;
                if (!memory.observed || !(memory.requiredEffects & gained))
                    continue;
                const int opportunity = interactionOpportunityId(
                    interactionMemoryKey(memory));
                suppressedUntil.erase(opportunity);
                suppressionEvidence.erase(opportunity);
                suppressionCount.erase(opportunity);
                ++rearmed;
            }
            int producer = 0;
            explosiveEffectProducer(producer);
            char detail[160];
            snprintf(detail, sizeof(detail),
                     "available=%u gained=%u producer_weapon=%d rearmed=%d",
                     effectCapabilities, gained, producer, rearmed);
            event("effect_capability_changed", detail);
            if (gained && rearmed)
                event("deferred_opportunities_rearmed", detail);
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

    bool recordEdgeFailure(const Portal &portal, const char *eventName, const char *reason)
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
            return false;
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
        return true;
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
                && portal.clearance >= playerStandingClearance()
                && std::abs(portal.floorDelta) <= kMaxWalkableStep;
            portal.jumpable = !(wallRecord.cstat & 1)
                && portal.openingWidth >= playerPassageWidth()
                && portal.clearance >= playerStandingClearance()
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
            if (portal.wall == wall && portal.from == from && portal.to == to)
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
        const int playerCell = nearestNavCell(observation.sector, observation.x,
                                              observation.y,
                                              llmapper::capability::playerFloorZ());
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

    // A frontier failure describes one particular geometry pose.  Once the
    // support/portal topology has been rebuilt, retaining that verdict until
    // its wall-clock cooldown expires confuses "failed before the mechanism
    // moved" with "still impossible now".  Re-arm only frontier work whose
    // suppression was recorded against older evidence; interaction and item
    // prerequisites retain their own, capability-specific wake-up rules.
    void rearmFrontiersForNewTopology()
    {
        int rearmed = 0;
        std::vector<int> stale;
        const int currentEvidence = worldEvidenceRevision();
        for (const auto &entry : suppressedUntil)
        {
            const int key = entry.first;
            if (key < 3000000 || key >= 4000000)
                continue;
            auto evidence = suppressionEvidence.find(key);
            if (evidence != suppressionEvidence.end()
                && evidence->second == currentEvidence)
                continue;
            stale.push_back(key);
        }
        for (int key : stale)
        {
            const int wallId = (key - 3000000 - int(kObjectiveFrontier)) / 8;
            suppressedUntil.erase(key);
            suppressionEvidence.erase(key);
            suppressionCount.erase(key);
            if (inRange(wallId, 0, numwalls))
            {
                failedEdges.erase(wallId * 65536 + wall[wallId].nextsector);
                localFailureSignatures.erase(wallId * 65536 + wall[wallId].nextsector);
            }
            ++rearmed;
        }
        if (rearmed > 0)
        {
            char detail[96];
            snprintf(detail, sizeof(detail), "revision=%d count=%d",
                     navTopologyRevision, rearmed);
            event("deferred_frontiers_rearmed", detail);
        }
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
                bool transitHypothesis = false;
                // A sector with no standable square is thin -- a step, a
                // ledge, a door track.  The player passes through such a
                // place rather than standing in it, so it is poor as a
                // destination but perfectly good as transit, and refusing it
                // outright severed real routes.
                if (!sectorHasStandableSpace(portal.to))
                {
                    const int transitWidth = playerClipRadius() * 2 + 64;
                    if (!portal.traversable || portal.openingWidth < transitWidth)
                    {
                        noteTooNarrow(portal);
                        continue;
                    }
                    transitHypothesis = true;
                    if (transitOnlySectors.insert(portal.to).second)
                    {
                        char detail[160];
                        snprintf(detail, sizeof(detail),
                                 "sector=%d via_wall=%d width=%d required=%d reason=cross_through_only",
                                 portal.to, portal.wall, portal.openingWidth,
                                 transitWidth);
                        event("boundary_transit_only", detail);
                    }
                }
                const int edgeId = portal.wall * 65536 + portal.to;
                llmapper::Opportunity opportunity;
                opportunity.sector = from;
                opportunity.target = portal.to;
                opportunity.wall = portal.wall;
                opportunity.requiredKey = portal.key;
                opportunity.depth = sectorDepth(from);
                opportunity.hops = reach;
                opportunity.descent = std::max(0, portal.floorDelta);
                opportunity.local = from == observation.sector;
                const bool crossable = (portal.traversable || portal.jumpable)
                    && !edgeFailed(edgeId) && !localEdgeFailed(edgeId);
                // A downward transition taller than Caleb can jump back up
                // destroys the current component's optionality.  Keep it as
                // a valid hypothesis, but let every known reversible action
                // run first.  Geometry supplies this confidence directly;
                // an unseen destination by itself is not considered risky.
                if (crossable && portal.floorDelta > playerReversibleDrop())
                    opportunity.oneWayRisk = portal.floorDelta;
                if (transitHypothesis)
                {
                    // This is evidence that a route may exist, not yet an
                    // observed actionable obstacle.  Let concrete unexplored
                    // frontiers run first even when the sliver is beside the
                    // player; the probe remains available when ordinary
                    // exploration is exhausted.
                    opportunity.local = false;
                    // Once observation or a collision has supplied a concrete
                    // mechanism for this boundary, the mechanism is the useful
                    // opportunity.  Keeping the weaker geometry hypothesis as
                    // well makes selection alternate between inspecting the
                    // door and actually operating it.
                    if (knownMechanismFor(portal))
                        continue;
                    if (llmapper::investigatedNow(investigatedBoundaries,
                                                  portal.wall, portal.from,
                                                  portal.to,
                                                  portalIdentitySignature(portal)))
                        continue;
                    opportunity.kind = llmapper::kOpportunityBlocked;
                    opportunity.id = 4000000 + portal.to;
                }
                else if (portal.key && !hasKey(portal.key))
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
            // Rank an action from the place where its physical prerequisite
            // can actually be satisfied.  A remotely visible actuator can be
            // in a visited room while requiring Caleb to stand on an unseen
            // carrier.  Treating the visible room as the opportunity location
            // commits the action too early and invents the actuator's XY on
            // that carrier; leaving it unreachable lets the carrier's normal
            // exploration frontier be chosen first.
            // The early-ordering constraint is specific to a ranged actuator
            // that would send its carrier away if fired from elsewhere.
            // Ordinary Use mechanisms can also be lift-call controls; making
            // their last occupied carrier the ledger location prevents a bot
            // that fell or stepped off from calling the lift back.
            const int requiredSupport = memory.activationMode == llmapper::kActivateVector
                ? actionSupportPrerequisite(memory) : -1;
            const int opportunitySector = requiredSupport >= 0
                ? requiredSupport : memory.fromSector;
            if (!inRange(opportunitySector, 0, int(hops.size())))
                continue;
            const int reach = hops[size_t(opportunitySector)];
            if (reach < 0)
                continue;
            llmapper::Opportunity opportunity;
            opportunity.kind = memory.key && !hasKey(memory.key)
                ? llmapper::kOpportunityLocked : llmapper::kOpportunityInteraction;
            opportunity.id = interactionOpportunityId(
                interactionMemoryKey(memory));
            opportunity.sector = opportunitySector;
            opportunity.target = memory.targetSector;
            opportunity.wall = memory.target.wall;
            opportunity.requiredKey = memory.key;
            opportunity.requiredEffects = memory.requiredEffects;
            opportunity.depth = sectorDepth(opportunitySector);
            opportunity.hops = reach;
            opportunity.local = opportunitySector == observation.sector;
            opportunity.dormantUntil = opportunityDormantUntil(opportunity.id);
            ledger.push_back(opportunity);
        }

        std::set<int> navSectors = knownNavSectors();
        // A locally observed portal exposes the geometry immediately on its
        // far side even before Caleb has entered that sector.  Include that
        // one-hop landing layer so a support touching the boundary can be a
        // prerequisite pose (voxel -> raised sector, bridge -> doorway).
        // This is bounded to edges observed from known space; it does not
        // flood through undiscovered map topology.
        for (int sectorId : navSectors)
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
            // A pickup normally counts as on-the-way only from the room the
            // bot is standing in.  Once a remembered action has proved that
            // ranged activation is a prerequisite, however, a reachable
            // weapon is no longer optional loot: promote it until that
            // capability is acquired.
            const bool suppliesRangedPrerequisite = rangedPrerequisitePending
                && memory.object.type >= kItemWeaponBase
                && memory.object.type < kItemWeaponMax;
            const bool suppliesEffectPrerequisite =
                (pickupEffectCapabilities(memory.object.type)
                    & pendingEffectPrerequisites) != 0;
            bool physicallyOnTheWay = reach == 0;
            if (physicallyOnTheWay && memory.object.kind == kObjectPickup)
            {
                ensureNavTopology();
                const int objectCell = navCellInSector(
                    memory.object.sector, memory.object.x, memory.object.y,
                    memory.object.z);
                // Sharing a sector number is not sharing a physical layer.
                // Retail ROR rooms and bridges can put a visible pickup tens
                // of thousands of Z units below the player in the same
                // sector.  Keep optional loot deferred until its collision-
                // derived support is genuinely local; prerequisites may still
                // promote it deliberately.
                physicallyOnTheWay = objectCell >= 0
                    && (navCells[size_t(objectCell)].support == currentPlayerSupport()
                        || std::abs(navCells[size_t(objectCell)].z
                                    - llmapper::capability::playerFloorZ())
                            <= kMaxWalkableStep);
            }
            opportunity.local = physicallyOnTheWay || suppliesRangedPrerequisite
                || suppliesEffectPrerequisite;
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
            const int edgeId = portal.wall * 65536 + portal.to;
            if (edgeFailed(edgeId) || localEdgeFailed(edgeId))
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

    // Blood's busy value is 16.16 fixed point.  Both 0 (OFF) and 65536
    // (ON) are settled endpoints; only a non-zero fractional part means the
    // mechanism is actually travelling.
    static bool busyValueInMotion(int busy)
    {
        return (uint32_t(busy) & 0xffffu) != 0;
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

    bool sectorSweptOccupancySafe(int sectorId) const
    {
        if (!inRange(sectorId, 0, numsectors))
            return false;
        const int extra = sector[sectorId].extra;
        if (extra <= 0 || extra >= kMaxXSectors)
            return false;
        const XSECTOR &dynamic = xsector[extra];
        // A support must actually move.  Sliding walls and ceiling-only
        // gates may expose a passage, but they do not transport their
        // occupant merely because their XSECTOR is busy.
        if (dynamic.offFloorZ == dynamic.onFloorZ)
            return false;
        const int endpointClearance = std::min(
            dynamic.offFloorZ - dynamic.offCeilZ,
            dynamic.onFloorZ - dynamic.onCeilZ);
        // ZTranslateSector interpolates both planes monotonically with the
        // same wave fraction, so endpoint clearance proves the entire sweep.
        return endpointClearance >= playerStandingClearance();
    }

    int actionSupportPrerequisite(const InteractionMemory &memory) const
    {
        const int receiver = mechanismSector(memory.target, memory.targetSector);
        // Calling an unseen carrier from a landing is valid and is how the
        // bot first makes many lifts boardable.  Once it has occupied a safe
        // carrier, however, an activation intended to explore the carrier's
        // other pose is only useful while the player remains aboard.
        if (!sectorSweptOccupancySafe(receiver))
            return -1;
        // A ranged/vector actuator on a carrier is not a lift-call button:
        // using it from the landing can send the carrier away without the
        // player.  Its support precondition exists even before the carrier
        // has been visited.  Ordinary Use actuators may still call an unseen
        // carrier; after boarding, subsequent transitions must stay aboard.
        if (memory.activationMode != llmapper::kActivateVector)
        {
            if (!visitedSectors.count(receiver)
                || !memory.activationPoseOnReceiver)
                return -1;
        }
        return receiver;
    }

    // Is this sector a piece of machinery the player should not loiter in?
    bool movingSectorHazard(int sectorId) const
    {
        const DoorTiming timing = doorTiming(sectorId);
        return (timing.moving || timing.autoCloses)
            && !sectorSweptOccupancySafe(sectorId);
    }

    static int64_t boundaryPoseKey(const Portal &portal)
    {
        // Wall ids fit in 16 bits in Blood maps.  Quantize Z only for the
        // associative key; traversal still uses the full values.  Engine Z
        // coordinates are 16x map units, so this preserves sub-step poses.
        const uint64_t wallPart = uint64_t(uint32_t(portal.wall) & 0xffffu) << 48;
        const uint64_t fromPart = uint64_t(uint32_t(portal.fromFloorZ >> 4)
                                           & 0xffffffu) << 24;
        const uint64_t toPart = uint64_t(uint32_t(portal.toFloorZ >> 4)
                                         & 0xffffffu);
        return int64_t(wallPart ^ fromPart ^ toPart);
    }

    bool boundaryUsesMovingSupport(const Portal &portal) const
    {
        const int sectors[2] = { portal.from, portal.to };
        for (int i = 0; i < 2; ++i)
        {
            if (!inRange(sectors[i], 0, numsectors))
                continue;
            const int extra = sector[sectors[i]].extra;
            if (extra <= 0 || extra >= kMaxXSectors)
                continue;
            const XSECTOR &dynamic = xsector[extra];
            if (dynamic.offFloorZ != dynamic.onFloorZ)
                return true;
        }
        return false;
    }

    // Watch known boundaries flip between blocked and traversable.  This is
    // the observable signature of Blood's moving geometry and the evidence
    // that an interaction actually did something.
    void trackBoundaryStates()
    {
        for (const Portal &portal : observation.portals)
        {
            const bool open = portal.traversable || portal.jumpable;
            const int64_t poseKey = boundaryPoseKey(portal);
            auto known = boundaryOpen.find(poseKey);
            if (known == boundaryOpen.end())
            {
                boundaryOpen[poseKey] = open;
                continue;
            }
            // Watching the envelope move is how the bot learns what an
            // interaction actually did.  Report meaningful changes even when
            // they do not yet cross the traversable threshold.
            auto lastClearance = boundaryClearance.find(poseKey);
            if (lastClearance == boundaryClearance.end()
                || std::abs(lastClearance->second - portal.clearance) >= 256)
            {
                boundaryClearance[poseKey] = portal.clearance;
                char progress[224];
                snprintf(progress, sizeof(progress),
                         "wall=%d from=%d to=%d clearance=%d body=%d crouch=%d floor_delta=%d busy=%d",
                         portal.wall, portal.from, portal.to, portal.clearance,
                         playerStandingClearance(), playerCrouchClearance(), portal.floorDelta,
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
                // A moving support creates a new pose rather than opening a
                // door at the old pose.  Its destination is selected from the
                // support-aware ledger after the ride settles; treating every
                // intermediate floor height as perishable access caused an
                // endless chain of unrelated objective preemptions.
                if (!boundaryUsesMovingSupport(portal))
                {
                    openedRouteWall = portal.wall;
                    openedRouteFrom = portal.from;
                    openedRouteTo = portal.to;
                    openedRouteTick = observation.tick;
                }
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
        int missingEffect = 0;
        const unsigned effects = availableEffectCapabilities();
        for (const llmapper::Opportunity &opportunity : ledger)
        {
            if (opportunity.hops < 0)
                ++unreachable;
            else if (opportunity.requiredKey > 0
                     && !(heldKeyMask & (1u << unsigned(opportunity.requiredKey & 31))))
                ++lockedNoKey;
            else if ((opportunity.requiredEffects & effects)
                     != opportunity.requiredEffects)
                ++missingEffect;
            else if (opportunity.dormantUntil > observation.tick)
                ++dormant;
            else
                ++pending;
        }
        char detail[256];
        snprintf(detail, sizeof(detail),
                 "reason=%s sector=%d ledger=%u pending=%d dormant=%d unreachable=%d locked_no_key=%d missing_effect=%d visited=%u inert=%u nav_failures=%u",
                 reason, observation.sector, unsigned(ledger.size()), pending, dormant,
                 unreachable, lockedNoKey, missingEffect, unsigned(visitedSectors.size()),
                 unsigned(inertBoundaries.size()), unsigned(navEdgeFailures.size()));
        event("ledger_snapshot", detail);
        int reported = 0;
        for (const llmapper::Opportunity &opportunity : ledger)
        {
            if (reported++ >= 12)
                break;
            char entry[224];
            snprintf(entry, sizeof(entry),
                     "id=%d kind=%d sector=%d target=%d wall=%d key=%d effects=%u depth=%d hops=%d dormant_s=%d",
                     opportunity.id, int(opportunity.kind), opportunity.sector,
                     opportunity.target, opportunity.wall, opportunity.requiredKey,
                     opportunity.requiredEffects, opportunity.depth, opportunity.hops,
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
            if ((opportunity.requiredEffects & availableEffectCapabilities())
                != opportunity.requiredEffects)
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
        // Work already in Caleb's current sector has no route gate.  A stale
        // first-gate annotation for that sector must not replace a reachable
        // pickup or viewpoint with an unrelated boundary interaction.
        if (target.sector == observation.sector)
            return nullptr;
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
            auto memory = interactions.find(opportunity->id - 10000000);
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
            objective.requiredSupportSector = actionSupportPrerequisite(memory->second);
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
                 "reason=%s kind=%d opportunity=%d sector=%d target=%d wall=%d depth=%d hops=%d key=%d one_way_risk=%d from_sector=%d",
                 chosen.reason, int(opportunity->kind), opportunity->id,
                 opportunity->sector, opportunity->target, opportunity->wall,
                 opportunity->depth, opportunity->hops, opportunity->requiredKey,
                 opportunity->oneWayRisk,
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
                snprintf(detail, sizeof(detail), "sprite=%d type=%d",
                         sprite_id, int(dude.type));
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
        const int playerCell = nearestNavCell(observation.sector, observation.x,
                                              observation.y,
                                              llmapper::capability::playerFloorZ());
        if (playerCell < 0 || navCells[playerCell].walkArea < 0)
            return nullptr;
        const int playerArea = navCells[playerCell].walkArea;
        const VisibleObject *best = nullptr;
        int bestDistance = INT32_MAX;
        auto consider = [&](const VisibleObject &object)
        {
            if (object.kind != kind || objectiveSuppressed(5000000 + object.sprite))
                return;
            const int objectCell = nearestNavCell(object.sector, object.x, object.y,
                                                  object.z);
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
            if (objectiveSuppressed(interactionOpportunityId(
                    interactionMemoryKey(memory))))
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
        if (memory.consecutiveNoEffectAttempts >= kMaxActivationAttempts)
            return false;
        if (supportTransitionUseful(memory))
            return true;
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

    static int64_t supportPoseKey(const SupportRef &support, int z)
    {
        return (int64_t(int(support.kind) & 0xff) << 56)
            ^ (int64_t(support.id & 0xffffff) << 32) ^ uint32_t(z);
    }

    bool stableSupportPose(const SupportRef &support, int z) const
    {
        // Intermediate animation frames are not reusable world states.  They
        // used to enter visitedSupportPoses every tick, so a single ride
        // looked like endless semantic exploration and defeated stall
        // detection.  Only mechanism endpoints (or genuinely stationary
        // sprite supports) are stable navigation poses.
        if (busyValueInMotion(observation.localSectorBusy))
            return false;
        if (support.kind == kSupportSpriteFloor)
        {
            if (!inRange(support.id, 0, kMaxSprites))
                return false;
            if (xvel[support.id] != 0 || yvel[support.id] != 0
                || zvel[support.id] != 0)
                return false;
        }
        if (support.kind != kSupportSectorFloor
            || !inRange(support.id, 0, numsectors))
            return true;
        const sectortype &record = sector[support.id];
        if (record.extra <= 0 || record.extra >= kMaxXSectors)
            return true;
        const XSECTOR &dynamic = xsector[record.extra];
        if (dynamic.offFloorZ == dynamic.onFloorZ)
            return true;
        return z == dynamic.offFloorZ || z == dynamic.onFloorZ;
    }

    // Is this interaction a safe transition of the support under the player
    // to a stable pose the bot has not occupied yet?  This is deliberately a
    // geometry/collision consequence, not an elevator label: a floor motion
    // that loses body clearance is rejected, while any monotonic support
    // motion whose whole clearance sweep fits may carry the player.
    bool supportTransitionUseful(const InteractionMemory &memory) const
    {
        if (!memory.attempted || !memory.activated || !memory.reversible
            || busyValueInMotion(observation.localSectorBusy)
            || !inRange(observation.sector, 0, numsectors))
            return false;
        const int receiver = mechanismSector(memory.target, memory.targetSector);
        if (receiver != observation.sector
            || !sectorSweptOccupancySafe(receiver))
            return false;
        const sectortype &record = sector[observation.sector];
        if (record.extra <= 0 || record.extra >= kMaxXSectors)
            return false;
        const XSECTOR &dynamic = xsector[record.extra];
        if (dynamic.offFloorZ == dynamic.onFloorZ)
            return false;
        const SupportRef support = currentPlayerSupport();
        if (support != SupportRef(kSupportSectorFloor, observation.sector))
            return false;
        const int currentFloor = llmapper::capability::playerFloorZ();
        if (currentFloor != record.floorz)
            return false;
        const int targetState = dynamic.state ? 0 : 1;
        const int targetFloor = targetState ? dynamic.onFloorZ : dynamic.offFloorZ;
        const int targetCeiling = targetState ? dynamic.onCeilZ : dynamic.offCeilZ;
        const int currentEndpointFloor = dynamic.state
            ? dynamic.onFloorZ : dynamic.offFloorZ;
        const int currentEndpointCeiling = dynamic.state
            ? dynamic.onCeilZ : dynamic.offCeilZ;
        // Floor/ceiling interpolation is monotonic in VDoorBusy, so the
        // minimum endpoint clearance proves every intermediate clearance.
        const int sweptClearance = std::min(targetFloor - targetCeiling,
                                            currentEndpointFloor
                                                - currentEndpointCeiling);
        if (sweptClearance < playerStandingClearance())
            return false;
        // Do not immediately undo a pose that has just exposed unexplored
        // space.  World actions are prerequisites for traversal, not states
        // to enumerate for their own sake; consume the new connection before
        // considering another support transition.
        auto localGraph = knownGraph.find(observation.sector);
        if (localGraph != knownGraph.end())
            for (const Portal &portal : localGraph->second)
                if ((portal.traversable || portal.jumpable)
                    && portal.to >= 0 && !visitedSectors.count(portal.to))
                    return false;
        return !visitedSupportPoses.count(supportPoseKey(support, targetFloor));
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

    // Pick a stance on a required support that belongs to the player's
    // currently reachable component.  The actuator can be thousands of
    // units away from the carrier it controls, so its XY is an aiming point,
    // never an invented location inside the support sector.
    int reachableSupportPose(int supportSector, int aimX, int aimY)
    {
        ensureNavTopology();
        const int start = currentNavStartCell();
        if (start < 0)
            return -1;
        std::vector<char> reachable(navCells.size(), 0);
        std::deque<int> reachQueue;
        reachable[size_t(start)] = 1;
        reachQueue.push_back(start);
        while (!reachQueue.empty())
        {
            const int current = reachQueue.front();
            reachQueue.pop_front();
            for (const NavLink &link : navCells[size_t(current)].links)
            {
                if (!llmapper::traversableMode(link.mode)
                    || !inRange(link.target, 0, int(navCells.size()))
                    || reachable[size_t(link.target)])
                    continue;
                reachable[size_t(link.target)] = 1;
                reachQueue.push_back(link.target);
            }
        }
        const SupportRef required(kSupportSectorFloor, supportSector);
        int best = -1;
        int bestAimDistance = INT32_MAX;
        int bestPlayerDistance = INT32_MAX;
        for (const NavCell &cell : navCells)
        {
            if (cell.sector != supportSector || cell.support != required
                || !reachable[size_t(cell.id)])
                continue;
            const int aimDistance = distance2(cell.center.x, cell.center.y, aimX, aimY);
            const int playerDistance = distance2(cell.center.x, cell.center.y,
                                                 observation.x, observation.y);
            if (aimDistance < bestAimDistance
                || (aimDistance == bestAimDistance
                    && playerDistance < bestPlayerDistance))
            {
                best = cell.id;
                bestAimDistance = aimDistance;
                bestPlayerDistance = playerDistance;
            }
        }
        return best;
    }

    int interactionExplosionRadius(const InteractionMemory &memory,
                                   int producerWeapon = 0) const
    {
        int radius = explosiveWeaponRadius(producerWeapon);
        if (memory.kind == kInteractionSprite
            && inRange(memory.id, 0, kMaxSprites)
            && sprite[memory.id].statnum < kMaxStatus)
            radius = std::max(radius, explosiveSpriteRadius(sprite[memory.id]));
        return radius;
    }

    int findDeliveredExplosive(const InteractionMemory &memory) const
    {
        int best = -1;
        int bestDistance = INT32_MAX;
        for (int spriteId = headspritestat[kStatThing]; spriteId >= 0;
             spriteId = nextspritestat[spriteId])
        {
            const spritetype &record = sprite[spriteId];
            if (record.type != kThingArmedTNTBundle
                && record.type != kThingArmedTNTStick
                && record.type != kThingArmedRemoteBomb
                && record.type != kThingArmedProxBomb)
                continue;
            const int distance = distance2(record.x, record.y,
                                           memory.x, memory.y);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = spriteId;
            }
        }
        return best;
    }

    GINPUT retreatFromExplosion(int hazardX, int hazardY, int targetId)
    {
        GINPUT input = {};
        input.syncFlags.run = 1;
        const int face = getangle(hazardX - observation.x,
                                  hazardY - observation.y);
        const int turn = angleDelta(face, observation.angle);
        input.q16turn = fix16_from_int(turn);
        if (std::abs(turn) < 128)
            input.forward = -kFullThrottle;
        noteCameraOwner("EXPLOSIVE_RETREAT", targetId, face, 0);
        return input;
    }

    // Deliver a Vector activation with the weapon's real vector reach.  The
    // target may move itself, operate something remote, or merely change
    // state; Vector is an activation mode and says nothing about destruction.
    GINPUT shootObstacle(InteractionMemory &memory)
    {
        setGoal("VECTOR_ACTIVATE", memory.id);
        GINPUT input = {};
        // Aim at the wall, not at an ActionScan pose.  A push mechanism is
        // approached along the surface normal and aimed at the middle of the
        // sector's height; for a shot that means firing at the ceiling, and
        // from off to one side, at nothing at all.  A wall spans the full
        // height, so the shot is simply level, straight at its midpoint.
        int aimX = memory.x;
        int aimY = memory.y;
        // Match FireTommy/FireShotgun's actual vector origin.  Eye height is
        // close enough for visibility, but not for a thin distant actuator.
        const int shotZ = gMe->zWeapon;
        const int targetSpan = std::max(1, memory.target.interactionBottomZ
                                           - memory.target.interactionTopZ);
        const int edgeInset = std::min(1024, std::max(64, targetSpan / 4));
        // A sprite actuator may occupy a small opening far above or below
        // eye level.  Its recorded centre is the engine-authored vector-hit
        // target; clamping eye height into its bounding box biases toward an
        // edge and can put the ray into the wall surrounding that opening.
        // Walls themselves still use the level-shot behaviour described
        // above because they span the opening.
        const int preferredAimZ = memory.kind == kInteractionSprite
            ? memory.z : shotZ;
        int aimZ = std::max(memory.target.interactionTopZ + edgeInset,
                            std::min(memory.target.interactionBottomZ - edgeInset,
                                     preferredAimZ));

        int weapon = 0;
        VECTOR_TYPE vectorType = kVectorTine;
        const bool ranged = vectorWeaponAvailable(weapon, vectorType);
        if (!ranged && meleeWeaponAvailable())
            weapon = kWeaponPitchfork;
        if (!weapon)
        {
            memory.unavailableFromSector = observation.sector;
            event("break_obstacle_unarmed", "reason=no_weapon_available");
            return input;
        }

        // A vector-triggered masked wall can be mostly transparent. Its
        // geometric midpoint is then a valid line of sight through the
        // opening, but not a hit on the actuator itself. Search a small,
        // deterministic set of points on the authored wall surface and keep
        // only one whose canonical engine vector actually resolves to that
        // wall. This reasons from collision/hitscan geometry, not texture or
        // map identity, and avoids relying on weapon spread for success.
        if (ranged && memory.kind == kInteractionWall
            && inRange(memory.target.wall, 0, numwalls)
            && inRange(wall[memory.target.wall].point2, 0, numwalls))
        {
            const walltype &start = wall[memory.target.wall];
            const walltype &end = wall[start.point2];
            static const int fractions[] = { 1, 3, 5, 7, 8, 9, 11, 13, 15 };
            bool foundSurface = false;
            const HITINFO savedHit = gHitInfo;
            for (int zFraction = 1; zFraction <= 3 && !foundSurface; ++zFraction)
            {
                const int candidateZ = memory.target.interactionTopZ
                    + int(int64_t(memory.target.interactionBottomZ
                                  - memory.target.interactionTopZ)
                          * zFraction / 4);
                for (int fraction : fractions)
                {
                    const int candidateX = start.x
                        + int(int64_t(end.x - start.x) * fraction / 16);
                    const int candidateY = start.y
                        + int(int64_t(end.y - start.y) * fraction / 16);
                    const int candidateAngle = getangle(
                        candidateX - observation.x, candidateY - observation.y);
                    const int candidateDistance = std::max(1, int(std::sqrt(
                        double(distance2(observation.x, observation.y,
                                         candidateX, candidateY)))));
                    const int candidateDz = divscale10(candidateZ - shotZ,
                                                       candidateDistance);
                    vec3_t adjustedOrigin = {};
                    const int hit = !VanillaMode()
                        ? VectorScanROR(gMe->pSprite, 0, shotZ - gMe->pSprite->z,
                                        Cos(candidateAngle) >> 16,
                                        Sin(candidateAngle) >> 16, candidateDz,
                                        gVectorData[vectorType].maxDist, 1,
                                        &adjustedOrigin)
                        : VectorScan(gMe->pSprite, 0,
                                     shotZ - gMe->pSprite->z,
                                     Cos(candidateAngle) >> 16,
                                     Sin(candidateAngle) >> 16, candidateDz,
                                     gVectorData[vectorType].maxDist, 1);
                    if ((hit == 0 || hit == 4)
                        && gHitInfo.hitwall == memory.target.wall)
                    {
                        aimX = candidateX;
                        aimY = candidateY;
                        aimZ = candidateZ;
                        foundSurface = true;
                        break;
                    }
                }
            }
            gHitInfo = savedHit;
        }

        const int aimAngle = getangle(aimX - observation.x,
                                      aimY - observation.y);
        const int turn = angleDelta(aimAngle, observation.angle);
        const int horizontal = std::max(1, int(std::sqrt(double(distance2(
            observation.x, observation.y, aimX, aimY)))));
        const int desiredLook = vectorLookAngleForTarget(shotZ, aimZ, horizontal);
        const int lookDelta = desiredLook - fix16_to_int(gMe->q16look);
        const int hazardRadius = interactionExplosionRadius(memory);
        const int safeDistance = safeExplosionSeparation(hazardRadius);
        if (safeDistance > 0)
        {
            if (!ranged)
            {
                memory.unavailableFromSector = observation.sector;
                event("explosive_interaction_deferred",
                      "reason=no_safe_ranged_satisfier");
                return input;
            }
            if (memory.effectBlastRadius != hazardRadius)
            {
                memory.effectBlastRadius = hazardRadius;
                char detail[176];
                snprintf(detail, sizeof(detail),
                         "kind=%d id=%d radius=%d safe_distance=%d source=engine_explosion_table",
                         int(memory.kind), memory.id, hazardRadius, safeDistance);
                event("explosive_safety_envelope", detail);
            }
            if (horizontal < safeDistance)
                return retreatFromExplosion(memory.x, memory.y, memory.id);
        }
        // Melee has to be within reach; a firearm needs a clear shot.  Firing
        // from wherever the approach happened to stop just buries the burst
        // in whatever stands between the bot and the target: the ammunition
        // drains, the aim reads as perfect, and the wall never registers.
        const int activationSector = currentObjective.requiredSupportSector >= 0
            ? currentObjective.requiredSupportSector
            : memory.fromSector >= 0 ? memory.fromSector : observation.sector;
        // A sprite endpoint belongs to its containing sector. A wall
        // endpoint belongs to the shooter's side of the boundary: asking
        // cansee() to resolve the same XY in the receiving sector places the
        // endpoint behind the closed wall and rejects a physically valid
        // vector hit. These are engine geometry semantics, independent of
        // what remote receiver the activation may drive.
        const int shotTargetSector = memory.kind == kInteractionSprite
            && memory.targetSector >= 0 ? memory.targetSector : activationSector;
        const bool clearShot = cansee(observation.x, observation.y, shotZ,
                                      observation.sector, aimX, aimY,
                                      aimZ, shotTargetSector);
        const int reach = meleeReach();
        const int distance2ToWall = distance2(observation.x, observation.y,
                                              memory.x, memory.y);
        if ((!ranged && distance2ToWall > reach * reach) || !clearShot)
        {
            noteCameraOwner("NAVIGATION", memory.id, aimAngle, 0);
            // If target projection deliberately brought us onto an upright
            // sprite top, that support is a take-off point, not the final
            // answer.  A high actuator or opening may only become visible
            // during the next jump (stacked/ROR geometry is a common Build
            // example).  Continue the physically implied climb while we are
            // grounded on the exact support selected for this target; without
            // this, ordinary steering immediately walks off the narrow top
            // and reconstructs the same climb forever.
            const SupportRef standingSupport = currentPlayerSupport();
            if (standingSupport.kind == kSupportSpriteFloor
                && gMe && gMe->pXSprite && gMe->pXSprite->height == 0)
            {
                ensureNavTopology();
                const int stance = navCellInSector(activationSector,
                                                   memory.x, memory.y,
                                                   memory.z);
                if (stance >= 0
                    && navCells[size_t(stance)].support == standingSupport
                    && memory.z < llmapper::capability::playerFloorZ()
                    && llmapper::capability::playerJumpImpulse() < 0)
                {
                    const int64_t takeoffKey = (int64_t(standingSupport.id) << 32)
                        ^ uint32_t(memory.id);
                    if (reportedSpriteSupportTakeoffs.insert(takeoffKey).second)
                    {
                        char detail[160];
                        snprintf(detail, sizeof(detail),
                                 "sprite=%d target=%d support_z=%d target_z=%d",
                                 standingSupport.id, memory.id,
                                 llmapper::capability::playerFloorZ(), memory.z);
                        event("nav_sprite_support_takeoff", detail);
                    }
                    return steerTo(memory.x, memory.y, false, false, memory.id,
                                   activationSector, kTraversalJumpable);
                }
            }
            if (currentObjective.requiredSupportSector >= 0)
            {
                const int pose = reachableSupportPose(
                    currentObjective.requiredSupportSector, memory.x, memory.y);
                if (pose >= 0)
                {
                    const NavCell &cell = navCells[size_t(pose)];
                    return navigateTo(cell.center.x, cell.center.y, cell.z,
                                      currentObjective.requiredSupportSector,
                                      memory.id, kTraversalUnknown);
                }
                return GINPUT{};
            }
            return navigateTo(memory.x, memory.y, memory.z, activationSector,
                              memory.id, kTraversalUnknown);
        }

        input.q16turn = fix16_from_int(turn);
        input.q16mlook = encodeLook(lookDelta);
        noteCameraOwner("VECTOR_ACTIVATE", memory.id, aimAngle, desiredLook);
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
        // Verify the canonical, unspread ray with the engine before spending
        // ammunition.  Yaw/look are input requests; gMe->aim is the
        // interpolated/auto-aimed vector the weapon will actually consume.
        // The authoritative post-shot callback remains the success test
        // because the weapon adds its own dispersion after this point.
        const HITINFO savedHit = gHitInfo;
        vec3_t adjustedOrigin = {};
        const int predictedHit = !VanillaMode()
            ? VectorScanROR(gMe->pSprite, 0, shotZ - gMe->pSprite->z,
                            gMe->aim.dx, gMe->aim.dy, gMe->aim.dz,
                            gVectorData[vectorType].maxDist, 1, &adjustedOrigin)
            : VectorScan(gMe->pSprite, 0, shotZ - gMe->pSprite->z,
                         gMe->aim.dx, gMe->aim.dy, gMe->aim.dz,
                         gVectorData[vectorType].maxDist, 1);
        const int predictedWall = gHitInfo.hitwall;
        const int predictedSprite = gHitInfo.hitsprite;
        const int predictedSector = gHitInfo.hitsect;
        gHitInfo = savedHit;
        const bool predictedTarget = (memory.kind == kInteractionSprite
                                      && predictedHit == 3
                                      && predictedSprite == memory.id)
            || (memory.kind == kInteractionWall
                && (predictedHit == 0 || predictedHit == 4)
                && (predictedWall == memory.id
                    || predictedWall == memory.target.wall));
        const int actualAimAngle = getangle(gMe->aim.dx, gMe->aim.dy);
        const int aimYawError = angleDelta(aimAngle, actualAimAngle);
        const int desiredDz = divscale10(aimZ - shotZ, horizontal);
        const int aimSlopeError = desiredDz - gMe->aim.dz;
        if (!memory.firingSolutionReported && predictedTarget)
        {
            memory.firingSolutionReported = true;
            char solution[384];
            snprintf(solution, sizeof(solution),
                     "kind=%d instance=%d tx=%d origin=(%d,%d,%d,s%d) target=(%d,%d,%d,s%d) desired_yaw=%d desired_dz=%d actual_yaw=%d actual_dz=%d yaw_error=%d slope_error=%d predicted_hit=%d predicted_sector=%d predicted_wall=%d predicted_sprite=%d weapon=%d vector=%d",
                     int(memory.kind), memory.id, memory.target.mechanismTx,
                     observation.x, observation.y, shotZ, observation.sector,
                     memory.x, memory.y, aimZ, shotTargetSector,
                     aimAngle, desiredDz, actualAimAngle, gMe->aim.dz,
                     aimYawError, aimSlopeError, predictedHit, predictedSector,
                     predictedWall, predictedSprite, weapon, int(vectorType));
            event("ranged_firing_solution", solution);
        }
        if (!memory.blockedFiringSolutionReported
            && std::abs(turn) < 96 && std::abs(lookDelta) < 96
            && clearShot && !predictedTarget)
        {
            memory.blockedFiringSolutionReported = true;
            int predictedExtra = -1;
            int predictedVector = 0;
            int predictedTx = 0;
            int predictedOwner = -1;
            int predictedNext = -1;
            if (inRange(predictedWall, 0, numwalls))
            {
                predictedExtra = wall[predictedWall].extra;
                predictedOwner = wallOwnerSector(predictedWall);
                predictedNext = wall[predictedWall].nextsector;
                if (inRange(predictedExtra, 1, kMaxXWalls))
                {
                    predictedVector = xwall[predictedExtra].triggerVector;
                    predictedTx = xwall[predictedExtra].txID;
                }
            }
            char blocked[448];
            snprintf(blocked, sizeof(blocked),
                     "kind=%d instance=%d desired_wall=%d predicted_hit=%d predicted_sector=%d predicted_wall=%d predicted_sprite=%d predicted_extra=%d predicted_vector=%d predicted_tx=%d predicted_owner=%d predicted_next=%d desired_yaw=%d actual_yaw=%d yaw_error=%d desired_dz=%d actual_dz=%d slope_error=%d",
                     int(memory.kind), memory.id, memory.target.wall,
                     predictedHit, predictedSector, predictedWall,
                     predictedSprite, predictedExtra, predictedVector,
                     predictedTx, predictedOwner, predictedNext,
                     aimAngle, actualAimAngle, aimYawError,
                     desiredDz, gMe->aim.dz, aimSlopeError);
            event("ranged_firing_solution_blocked", blocked);
        }
        if (std::abs(turn) < 96 && std::abs(lookDelta) < 96 && clearShot
            && predictedTarget)
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

    GINPUT deliverDamageEffect(InteractionMemory &memory)
    {
        setGoal("DELIVER_REQUIRED_EFFECT", memory.id);
        GINPUT input = {};
        input.syncFlags.run = 1;

        int weapon = 0;
        if (!(memory.requiredEffects & llmapper::kEffectExplosive)
            || !explosiveEffectProducer(weapon))
            return input;

        const int dx = memory.x - observation.x;
        const int dy = memory.y - observation.y;
        const int horizontal = std::max(1, int(std::sqrt(double(distance2(
            observation.x, observation.y, memory.x, memory.y)))));
        const int aimAngle = getangle(dx, dy);
        const int turn = angleDelta(aimAngle, observation.angle);
        const POSTURE &stand = gMe->pPosture[gMe->lifeMode][kPostureStand];
        const int eyeZ = gMe->pSprite->z - stand.eyeAboveZ;
        const int span = std::max(1, memory.target.interactionBottomZ
                                     - memory.target.interactionTopZ);
        const int inset = std::min(1024, std::max(64, span / 4));
        const int aimZ = std::max(memory.target.interactionTopZ + inset,
                                  std::min(memory.target.interactionBottomZ - inset,
                                           eyeZ));
        const int desiredLook = vectorLookAngleForTarget(eyeZ, aimZ, horizontal);
        const int lookDelta = desiredLook - fix16_to_int(gMe->q16look);
        const bool clearDelivery = cansee(observation.x, observation.y, eyeZ,
                                          observation.sector, memory.x, memory.y,
                                          aimZ, memory.targetSector);
        const int blastRadius = interactionExplosionRadius(memory, weapon);
        const int safeDistance = safeExplosionSeparation(blastRadius);
        const int maxStandOff = safeDistance + kExplosiveThrowAllowance;

        // Establish a real delivery pose before consuming explosives.  Too
        // far makes a throw unreliable; too close turns a valid solution
        // into self-damage.  Facing the target while backpedalling keeps the
        // camera and retreat direction stable.
        if (horizontal > maxStandOff || !clearDelivery)
            return navigateTo(memory.x, memory.y, memory.z, memory.fromSector,
                              memory.id, kTraversalUnknown);
        input.q16turn = fix16_from_int(turn);
        input.q16mlook = encodeLook(lookDelta);
        noteCameraOwner("EXPLOSIVE_EFFECT", memory.id, aimAngle, desiredLook);
        if (horizontal < safeDistance)
        {
            input.forward = -kFullThrottle;
            return input;
        }

        if (gMe->curWeapon != weapon)
        {
            input.syncFlags.weaponChange = 1;
            input.newWeapon = uint8_t(weapon);
            if (memory.effectWeapon != weapon)
            {
                memory.effectWeapon = weapon;
                char detail[240];
                snprintf(detail, sizeof(detail),
                         "kind=%d id=%d required_effects=%u producer_weapon=%d distance=%d blast_radius=%d safe_distance=%d source=engine_explosion_table",
                         int(memory.kind), memory.id, memory.requiredEffects, weapon,
                         horizontal, blastRadius, safeDistance);
                event("effect_satisfier_selected", detail);
            }
            return input;
        }

        const bool aligned = std::abs(turn) < 80 && std::abs(lookDelta) < 96;
        if (weapon == kWeaponNapalm)
        {
            if (!aligned)
                return input;
            if (!memory.attempted)
            {
                memory.attempted = true;
                memory.state = 1;
                memory.beforeState = interactionStateSignature(memory);
                memory.beforePortals = observation.portals;
                memory.effectStartedTick = observation.tick;
                memory.effectAmmoBefore = gMe->ammoCount[weapon - 1];
                memory.effectBlastRadius = blastRadius;
                ++memory.activationCount;
                event("effect_delivery_started", "delivery=explosive_projectile");
                input.buttonFlags.shoot = 1;
                memory.lastActivationTick = observation.tick;
                return input;
            }
            if (horizontal < safeDistance + kExplosiveSafetyMargin
                && observation.tick - memory.effectStartedTick < 3 * kTicsPerSec)
                return retreatFromExplosion(memory.x, memory.y, memory.id);
            return input;
        }

        // TNT's ordinary weapon state is a transaction: press to prime and
        // build throw power, release to run the ThrowBundle QAV, then retreat
        // while the impact/fuse resolves.  The bot never spawns or damages
        // anything directly.
        if (memory.effectPhase == 0)
        {
            if (!aligned)
                return input;
            if (!memory.attempted)
            {
                memory.attempted = true;
                memory.state = 1;
                memory.beforeState = interactionStateSignature(memory);
                memory.beforePortals = observation.portals;
            }
            memory.effectStartedTick = observation.tick;
            memory.effectAmmoBefore = gMe->ammoCount[weapon - 1];
            memory.effectProjectile = -1;
            memory.effectBlastRadius = blastRadius;
            memory.effectPhase = 1;
            ++memory.activationCount;
            input.buttonFlags.shoot = 1;
            event("effect_delivery_started", "delivery=primed_throw effect=explosive");
            return input;
        }
        if (memory.effectPhase == 1)
        {
            if (observation.tick - memory.effectStartedTick < kExplosivePrimeTicks)
            {
                input.buttonFlags.shoot = 1;
                return input;
            }
            memory.effectPhase = 2;
            memory.effectStartedTick = observation.tick;
            memory.lastActivationTick = observation.tick;
            event("effect_delivery_released", "delivery=throw effect=explosive");
            return input;
        }
        if (memory.effectPhase == 2 && gMe->weaponState != 6)
        {
            memory.effectPhase = 3;
            memory.effectStartedTick = observation.tick;
        }
        if (memory.effectPhase >= 2)
        {
            if (memory.effectProjectile < 0)
            {
                memory.effectProjectile = findDeliveredExplosive(memory);
                if (memory.effectProjectile >= 0)
                {
                    char detail[176];
                    snprintf(detail, sizeof(detail),
                             "sprite=%d type=%d radius=%d safe_distance=%d",
                             memory.effectProjectile,
                             int(sprite[memory.effectProjectile].type),
                             memory.effectBlastRadius,
                             safeExplosionSeparation(memory.effectBlastRadius));
                    event("explosive_projectile_tracked", detail);
                }
            }
            int hazardX = memory.x;
            int hazardY = memory.y;
            if (inRange(memory.effectProjectile, 0, kMaxSprites)
                && sprite[memory.effectProjectile].statnum < kMaxStatus)
            {
                hazardX = sprite[memory.effectProjectile].x;
                hazardY = sprite[memory.effectProjectile].y;
            }
            const int actualDistance = int(std::sqrt(double(distance2(
                observation.x, observation.y, hazardX, hazardY))));
            const int required = safeExplosionSeparation(memory.effectBlastRadius);
            if (actualDistance < required
                && observation.tick - memory.effectStartedTick < 4 * kTicsPerSec)
                return retreatFromExplosion(hazardX, hazardY, memory.id);
            if (observation.tick - memory.effectStartedTick > kExplosiveOutcomeTicks)
            {
                memory.effectPhase = 0;
                memory.effectStartedTick = -1;
                memory.effectProjectile = -1;
                event("effect_delivery_retry", "reason=no_world_delta_yet");
            }
        }
        return input;
    }

    GINPUT steerInteraction(InteractionMemory &memory)
    {
        const int requiredApproachSector = currentObjective.requiredSupportSector >= 0
            ? currentObjective.requiredSupportSector : memory.fromSector;
        if (requiredApproachSector != observation.sector)
        {
            // Route through visited space with the ordinary navigator: the
            // mechanism's own sector is the destination, and crossing any
            // number of already-walked boundaries to get there is normal
            // transportation, not a new exploration decision.
            setGoal("NAVIGATE_TO_NEW_INTERACTION", memory.id);
            int approachX = memory.x;
            int approachY = memory.y;
            int approachZ = memory.z;
            if (currentObjective.requiredSupportSector >= 0)
            {
                const int pose = reachableSupportPose(
                    currentObjective.requiredSupportSector, memory.x, memory.y);
                if (pose >= 0)
                {
                    const NavCell &cell = navCells[size_t(pose)];
                    approachX = cell.center.x;
                    approachY = cell.center.y;
                    approachZ = cell.z;
                }
            }
            // A support-constrained activation or an objective that began as a
            // local physical approach needs the detailed graph first: it
            // carries exact moving-support and sprite-layer state that a sector
            // route deliberately omits.  A remembered remote interaction return
            // instead prefers the broader directed sector graph, which retains
            // valid gap/elevator transport even when the detailed mesh is split
            // into disconnected vertical components.
            const bool needsExactSupport =
                currentObjective.requiredSupportSector >= 0
                || currentObjective.interactionRouteMode == 1;
            Portal route;
            if (!needsExactSupport
                && findKnownRoute(requiredApproachSector, route))
                return steerPortal(route);

            const int approachSignature = navigationSignature(
                approachX, approachY, approachZ, requiredApproachSector,
                memory.id);
            ensureNavTopology();
            const bool detailedRoute = (!navRoute.empty()
                    && navRouteSignature == approachSignature
                    && navRouteTopologyRevision == navTopologyRevision)
                || (navRouteRejectedSignature != approachSignature
                    && buildNavRoute(approachX, approachY, approachZ,
                                     requiredApproachSector,
                                     approachSignature));
            if (detailedRoute)
            {
                const GINPUT approach = navigateTo(
                    approachX, approachY, approachZ, requiredApproachSector,
                    memory.id, kTraversalUnknown);
                // A deliberate route state can emit a neutral transition
                // frame.  It still owns steering until it clears itself.
                if ((!navRoute.empty()
                     && navRouteSignature == approachSignature)
                    || approach.forward || approach.strafe || approach.q16turn
                    || approach.q16mlook || approach.buttonFlags.jump
                    || approach.buttonFlags.crouch)
                    return approach;
            }

            // When the detailed mesh cannot connect vertical/gap components,
            // execute the first currently valid directed sector transition.
            // Point navigation toward a remote switch can see a straight XY
            // line through a pit, while steerPortal retains the required
            // run-up/jump, raised-support, elevator, or walk transaction.
            if (findKnownRoute(requiredApproachSector, route))
                return steerPortal(route);
            const GINPUT approach = navigateTo(approachX, approachY, approachZ,
                                               requiredApproachSector, memory.id,
                                               kTraversalUnknown);
            if (approach.forward || approach.strafe || approach.q16turn
                || approach.q16mlook || approach.buttonFlags.jump
                || approach.buttonFlags.crouch)
                return approach;
            // Neither the known directed transport graph nor the local mesh
            // currently reaches the required approach layer.
            memory.unavailableFromSector = observation.sector;
            memory.unavailableState = interactionStateSignature(memory);
            memory.unavailablePose = navTopologyRevision;
            char unavailable[192];
            snprintf(unavailable, sizeof(unavailable),
                     "kind=%d id=%d from=%d required=%d topology=%d reason=NO_ROUTE_TO_APPROACH",
                     int(memory.kind), memory.id, observation.sector,
                     requiredApproachSector, navTopologyRevision);
            event("interaction_unavailable", unavailable);
            // Route lifetime and objective lifetime are distinct.  Retain the
            // useful interaction in memory, but release this execution
            // attempt so other ledger work can run.  Suppression evidence
            // will re-arm it after topology/knowledge changes or cooldown.
            invalidateObjective("interaction_approach_unreachable");
            return GINPUT{};
        }

        if (memory.activationMode == llmapper::kActivateVector)
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
        if (memory.activationMode == llmapper::kActivateDamage)
        {
            if (memory.state == 2 || memory.observedKnownWorldDelta)
                return GINPUT{};
            return deliverDamageEffect(memory);
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

        const bool stillAtActionPose = expectedDistance
                <= kActionScanRange * kActionScanRange
            && std::abs(angleDelta(expectedAngle, observation.angle)) < 96
            && std::abs(lookDelta) < 64;
        if (memory.state == 1 && memory.engineAccepted && !valid
            && stillAtActionPose && memory.lastActivationTick >= 0
            && observation.tick - memory.lastActivationTick >= kTicsPerSec / 4)
        {
            // ActionScan itself is an observable part of world state.  When a
            // target accepted Use and then disappears while the bot has held
            // the same valid pose, a moving/rotating mechanism answered even
            // if its mapper state bits do not encode the wall motion.  A
            // sound-only surface remains targetable and therefore does not
            // satisfy this evidence rule.
            memory.state = 2;
            memory.activated = true;
            memory.observedLocalEffect = true;
            memory.observedKnownWorldDelta = true;
            memory.consecutiveNoEffectAttempts = 0;
            memory.afterState = interactionStateSignature(memory);
            shareLearnedWallEffect(memory);
            rearmAroundMechanism(memory);
            lastSemanticProgressTick = observation.tick;
            event("interaction_world_delta",
                  "reason=accepted_action_target_disappeared_at_stable_pose");
            if (memory.causalWall >= 0 && memory.causalFrom >= 0
                && memory.causalTo >= 0)
            {
                openedRouteWall = memory.causalWall;
                openedRouteFrom = memory.causalFrom;
                openedRouteTo = memory.causalTo;
                openedRouteTick = observation.tick;
            }
        }

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
        // Incidental feedback (for example a voice line) is not a useful
        // world-state transition.  Bound consecutive no-effect attempts even
        // when geometry still labels the affordance reversible.  A real
        // world delta resets this counter, preserving reusable elevators and
        // doors after the bot returns to them.
        const bool budgetLeft = memory.consecutiveNoEffectAttempts
            < kMaxActivationAttempts;
        if (valid && !missingKey && !mechanismMoving && cooldownElapsed
            && !wouldMoveOwnSector
            && budgetLeft
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
                ++memory.consecutiveNoEffectAttempts;
                event("interaction_settled", "engine_accepted=1 observed_local_effect=0 observed_known_world_delta=0");
            }
            else
            {
                memory.state = 3;
                ++memory.consecutiveNoEffectAttempts;
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
            const int navigationSector = currentObjective.requiredSupportSector >= 0
                ? currentObjective.requiredSupportSector
                : memory.fromSector >= 0 ? memory.fromSector : observation.sector;
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
                // This fallback is a transport route through known space, not
                // raw map adjacency.  A solid or impossibly high boundary may
                // connect two Build sectors topologically while being useless
                // in this direction.  Including it made breadth-first search
                // prefer short impossible climbs over longer stair/jump routes.
                if (!edge.traversable && !edge.jumpable)
                    continue;
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
        if (clearance < playerStandingClearance() && clearance >= playerCrouchClearance())
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

    unsigned effectCapabilitiesForWeapon(int weapon) const
    {
        switch (weapon)
        {
        case kWeaponTNT:
        case kWeaponNapalm:
            return llmapper::kEffectExplosive;
        default:
            return llmapper::kEffectNone;
        }
    }

    bool explosiveEffectProducer(int &weapon) const
    {
        // These are different physical satisfiers for the same world effect:
        // an impact/fused thrown charge and an explosive projectile.  The
        // opportunity never depends on either inventory item by name.
        static const int candidates[] = { kWeaponTNT, kWeaponNapalm };
        for (int candidate : candidates)
        {
            if (!gMe->hasWeapon[candidate])
                continue;
            const int ammo = candidate - 1;
            if (candidate == kWeaponTNT && gMe->isUnderwater)
                continue;
            if (gInfiniteAmmo || (ammo >= 0
                && ammo < int(sizeof(gMe->ammoCount) / sizeof(gMe->ammoCount[0]))
                && gMe->ammoCount[ammo] > 0))
            {
                weapon = candidate;
                return true;
            }
        }
        return false;
    }

    unsigned availableEffectCapabilities() const
    {
        unsigned effects = llmapper::kEffectNone;
        int producer = 0;
        if (explosiveEffectProducer(producer))
            effects |= llmapper::kEffectExplosive;
        return effects;
    }

    unsigned pickupEffectCapabilities(int type) const
    {
        if (type >= kItemWeaponBase && type < kItemWeaponMax)
            return effectCapabilitiesForWeapon(
                gWeaponItemData[type - kItemWeaponBase].type);
        if (type >= kItemAmmoBase && type < kItemAmmoMax)
            return effectCapabilitiesForWeapon(
                gAmmoItemData[type - kItemAmmoBase].weaponType);
        return llmapper::kEffectNone;
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
            return interactionOpportunityId(objective.interactionKey);
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
            const Portal *crossing = portalByWall(currentObjective.wall,
                                                  currentObjective.sector,
                                                  currentObjective.targetSector);
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

    bool supportLayerTouchesArea(int sectorId, int z, int area) const
    {
        const SupportRef floorSupport(kSupportSectorFloor, sectorId);
        for (const NavCell &cell : navCells)
            if (cell.sector == sectorId && cell.z == z
                && cell.support == floorSupport && cell.walkArea == area)
                return true;
        return false;
    }

    // Vector-triggered objects require a hitscan producer.  A flare or
    // explosive projectile is ranged combat capability, but it cannot
    // satisfy XSPRITE::Vector and must not be selected for this action.
    bool vectorWeaponAvailable(int &weapon, VECTOR_TYPE &vectorType) const
    {
        static const struct Candidate
        {
            int weapon;
            VECTOR_TYPE vector;
        } candidates[] = {
            { kWeaponTommy, kVectorTommyregular },
            { kWeaponShotgun, kVectorShell },
        };
        for (const Candidate &candidate : candidates)
        {
            if (!gMe->hasWeapon[candidate.weapon])
                continue;
            const int ammo = candidate.weapon - 1;
            if (gInfiniteAmmo || (ammo >= 0
                && ammo < int(sizeof(gMe->ammoCount) / sizeof(gMe->ammoCount[0]))
                && gMe->ammoCount[ammo] > 0))
            {
                weapon = candidate.weapon;
                vectorType = candidate.vector;
                return true;
            }
        }
        return false;
    }

    // Find a known way to change one mechanism that is reachable on the
    // player's present support component.  Receiver identity is deliberately
    // independent of actuator identity: a wall, sprite, or the carrier
    // itself can all establish the same world state.
    InteractionMemory *reachableActuatorForMechanism(int mechanism, int area)
    {
        InteractionMemory *best = nullptr;
        int bestDistance = INT32_MAX;
        for (auto &entry : interactions)
        {
            InteractionMemory &memory = entry.second;
            if (!memory.observed || !memory.reversible || memory.locked
                || (memory.key && !hasKey(memory.key)) || memory.state == 1
                || memory.state == 3
                || memory.activationCount >= kMaxActivationAttempts)
                continue;
            const int receiver = mechanismSector(memory.target, memory.targetSector);
            if (receiver != mechanism && memory.targetSector != mechanism
                && memory.id != mechanism)
                continue;
            const int cell = navCellInSectorArea(memory.fromSector,
                                                 memory.x, memory.y, area);
            if (cell < 0)
                continue;
            const int distance = distance2(observation.x, observation.y,
                                           navCells[size_t(cell)].center.x,
                                           navCells[size_t(cell)].center.y);
            if (!best || distance < bestDistance)
            {
                best = &memory;
                bestDistance = distance;
            }
        }
        return best;
    }

    bool beginDynamicPrerequisite(InteractionMemory &memory, int mechanism,
                                  int state, const char *reason,
                                  bool remainOnSupport = false)
    {
        if (!currentObjective.active || suspendedObjectiveActive)
            return false;
        suspendedObjective = currentObjective;
        suspendedObjectiveActive = true;
        dynamicPrerequisiteMechanism = mechanism;
        dynamicPrerequisiteState = state;
        currentObjective = Objective{};
        movementTargetActive = false;
        resetNavigation();

        // This is a new causal transaction for a previously learned,
        // reversible actuator.  Keep its lifetime attempt count, but discard
        // the completed transaction flags so the ordinary interaction
        // executor can issue exactly one fresh activation.
        memory.state = 0;
        memory.attempted = false;
        memory.activated = false;
        memory.engineAccepted = false;
        memory.observedLocalEffect = false;
        memory.observedKnownWorldDelta = false;
        memory.unavailableAttempts = 0;
        memory.unavailableFromSector = -1;

        Objective prerequisite;
        prerequisite.type = kObjectiveInteraction;
        prerequisite.id = memory.id;
        prerequisite.sector = memory.fromSector;
        prerequisite.targetSector = memory.targetSector;
        prerequisite.x = memory.x;
        prerequisite.y = memory.y;
        prerequisite.z = memory.z;
        prerequisite.interactionKey = interactionMemoryKey(memory);
        prerequisite.requiredSupportSector = remainOnSupport ? mechanism : -1;
        selectObjective(prerequisite, "ESTABLISH_ROUTE_CONDITION");
        char detail[256];
        snprintf(detail, sizeof(detail),
                 "mechanism=%d state=%d actuator_kind=%d actuator=%d from_sector=%d required_support=%d suspended_type=%d suspended_id=%d reason=%s",
                 mechanism, state, int(memory.kind), memory.id, memory.fromSector,
                 prerequisite.requiredSupportSector,
                 int(suspendedObjective.type), suspendedObjective.id, reason);
        event("dynamic_prerequisite_selected", detail);
        return true;
    }

    // Resolve only a condition that blocks the currently committed route.
    // This is the production counterpart of planDynamicRoute(): no global
    // product of mechanism states is enumerated.  At most one receiver on a
    // reachable-space boundary is established, the objective is resumed,
    // and the next boundary is considered from the new world state.
    bool selectDynamicPrerequisite()
    {
        if (!currentObjective.active || currentObjective.type == kObjectiveInteraction
            || suspendedObjectiveActive)
            return false;
        ensureNavTopology();
        const int start = currentNavStartCell();
        if (start < 0)
            return false;
        const int startArea = navCells[size_t(start)].walkArea;

        int anchorX = 0;
        int anchorY = 0;
        if (!objectiveAnchor(anchorX, anchorY))
            return false;
        int goalSector = currentObjective.sector;
        int goalZ = currentObjective.z;
        const Portal *crossing = nullptr;
        if (currentObjective.type == kObjectiveFrontier
            || currentObjective.type == kObjectiveInvestigate)
        {
            crossing = portalByWall(currentObjective.wall, currentObjective.sector,
                                    currentObjective.targetSector);
            if (crossing)
            {
                goalSector = crossing->from;
                goalZ = crossing->fromFloorZ;
            }
        }
        const int goal = navCellInSector(goalSector, anchorX, anchorY, goalZ);
        if (goal < 0)
            return false;
        const int goalArea = navCells[size_t(goal)].walkArea;
        const int probeSignature = llmapper::mixHash(
            llmapper::mixHash(objectiveKey(currentObjective), navTopologyRevision),
            navDynamicSignatureValue);
        const bool emitProbe = probeSignature != lastDynamicPrerequisiteProbeSignature;
        if (emitProbe)
        {
            lastDynamicPrerequisiteProbeSignature = probeSignature;
            char detail[224];
            snprintf(detail, sizeof(detail),
                     "objective_type=%d id=%d start_cell=%d start_area=%d goal_cell=%d goal_area=%d goal_sector=%d wall=%d",
                     int(currentObjective.type), currentObjective.id, start, startArea,
                     goal, goalArea, goalSector,
                     crossing ? crossing->wall : -1);
            event("dynamic_prerequisite_probe", detail);
        }

        // A moving source sector is itself a conditional boundary.  At its
        // wrong endpoint a perfectly ordinary exit looks like a wall or an
        // impossible vertical step; restore the endpoint whose floor meets
        // the fixed landing before investigating that wall.
        if (crossing && inRange(crossing->from, 0, numsectors))
        {
            const int mechanism = crossing->from;
            const sectortype &record = sector[mechanism];
            if (record.extra > 0 && record.extra < kMaxXSectors)
            {
                const XSECTOR &dynamic = xsector[record.extra];
                if (!busyValueInMotion(dynamic.busy)
                    && dynamic.offFloorZ != dynamic.onFloorZ)
                {
                    const int landingZ = crossing->toFloorZ;
                    const int offDistance = std::abs(dynamic.offFloorZ - landingZ);
                    const int onDistance = std::abs(dynamic.onFloorZ - landingZ);
                    const int desiredState = onDistance < offDistance ? 1 : 0;
                    const int desiredDistance = desiredState ? onDistance : offDistance;
                    const int currentDistance = std::abs(record.floorz - landingZ);
                    const int usefulReach = std::max(kMaxWalkableStep,
                                                     playerJumpRiseLimit());
                    const int desiredClearance = desiredState
                        ? dynamic.onFloorZ - dynamic.onCeilZ
                        : dynamic.offFloorZ - dynamic.offCeilZ;
                    if (desiredState != dynamic.state
                        && desiredDistance <= usefulReach
                        && desiredDistance < currentDistance
                        && desiredClearance >= playerStandingClearance())
                    {
                        InteractionMemory *actuator = reachableActuatorForMechanism(
                            mechanism, startArea);
                        if (actuator)
                            return beginDynamicPrerequisite(*actuator, mechanism,
                                                            desiredState,
                                                            "align_source_support_pose");
                    }
                }
            }
        }

        // First establish the pose required on the far side of the committed
        // boundary.  Doing this before arranging transport to the boundary
        // matters when the actuator is on the base component and the
        // boundary is on a raised component.
        if (crossing && inRange(crossing->to, 0, numsectors))
        {
            const int mechanism = crossing->to;
            const sectortype &record = sector[mechanism];
            if (record.extra > 0 && record.extra < kMaxXSectors)
            {
                const XSECTOR &dynamic = xsector[record.extra];
                if (!busyValueInMotion(dynamic.busy)
                    && dynamic.offFloorZ != dynamic.onFloorZ)
                {
                    const int sourceZ = crossing->fromFloorZ;
                    const int offDistance = std::abs(dynamic.offFloorZ - sourceZ);
                    const int onDistance = std::abs(dynamic.onFloorZ - sourceZ);
                    const int desiredState = onDistance < offDistance ? 1 : 0;
                    const int desiredDistance = desiredState ? onDistance : offDistance;
                    const int currentDistance = std::abs(record.floorz - sourceZ);
                    const int usefulReach = std::max(kMaxWalkableStep,
                                                     playerJumpRiseLimit());
                    const int desiredClearance = desiredState
                        ? dynamic.onFloorZ - dynamic.onCeilZ
                        : dynamic.offFloorZ - dynamic.offCeilZ;
                    if (desiredState != dynamic.state
                        && desiredDistance <= usefulReach
                        && desiredDistance < currentDistance
                        && desiredClearance >= playerStandingClearance())
                    {
                        InteractionMemory *actuator = reachableActuatorForMechanism(
                            mechanism, startArea);
                        if (actuator)
                            return beginDynamicPrerequisite(*actuator, mechanism,
                                                            desiredState,
                                                            "align_boundary_support_pose");
                    }
                }
            }
        }

        if (startArea < 0 || goalArea < 0 || startArea == goalArea)
            return false;

        // A safe moving floor can be the missing edge between two otherwise
        // disconnected support components.  Select it only when its live
        // endpoint touches the player's component and its opposite endpoint
        // touches the committed goal component.
        for (auto &entry : interactions)
        {
            InteractionMemory &memory = entry.second;
            if (!memory.observed || !memory.reversible
                || memory.kind != kInteractionSector
                || !memory.target.sectorPushCurrent)
                continue;
            const int mechanism = memory.fromSector;
            if (!inRange(mechanism, 0, numsectors)
                || !sectorSweptOccupancySafe(mechanism))
                continue;
            const sectortype &record = sector[mechanism];
            if (record.extra <= 0 || record.extra >= kMaxXSectors)
                continue;
            const XSECTOR &dynamic = xsector[record.extra];
            if (busyValueInMotion(dynamic.busy)
                || dynamic.offFloorZ == dynamic.onFloorZ)
                continue;
            const int currentZ = record.floorz;
            const int otherState = dynamic.state ? 0 : 1;
            const int otherZ = otherState ? dynamic.onFloorZ : dynamic.offFloorZ;
            const bool currentTouchesStart = supportLayerTouchesArea(
                mechanism, currentZ, startArea);
            const bool otherTouchesGoal = supportLayerTouchesArea(
                mechanism, otherZ, goalArea);
            if (emitProbe)
            {
                char detail[256];
                snprintf(detail, sizeof(detail),
                         "mechanism=%d state=%d current_z=%d other_state=%d other_z=%d current_touches_start=%d other_touches_goal=%d safe=%d activation_count=%d",
                         mechanism, dynamic.state, currentZ, otherState, otherZ,
                         currentTouchesStart ? 1 : 0, otherTouchesGoal ? 1 : 0,
                         sectorSweptOccupancySafe(mechanism) ? 1 : 0,
                         memory.activationCount);
                event("dynamic_support_candidate", detail);
            }
            if (!currentTouchesStart || !otherTouchesGoal)
                continue;
            InteractionMemory *actuator = reachableActuatorForMechanism(
                mechanism, startArea);
            if (actuator)
                return beginDynamicPrerequisite(*actuator, mechanism, otherState,
                                                "transfer_between_support_components",
                                                true);
        }
        return false;
    }

    bool dynamicPrerequisiteEstablished() const
    {
        if (!suspendedObjectiveActive
            || dynamicPrerequisiteMechanism < 0
            || !inRange(dynamicPrerequisiteMechanism, 0, numsectors))
            return false;
        const int extra = sector[dynamicPrerequisiteMechanism].extra;
        if (extra <= 0 || extra >= kMaxXSectors)
            return false;
        const XSECTOR &dynamic = xsector[extra];
        return !busyValueInMotion(dynamic.busy)
            && dynamic.state == dynamicPrerequisiteState;
    }

    // Is the crossing this objective depends on currently in motion?
    bool committedMechanismBusy() const
    {
        if (!currentObjective.active)
            return false;
        if (busyValueInMotion(observation.localSectorBusy))
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
            if (extra > 0 && extra < kMaxXSectors
                && busyValueInMotion(xsector[extra].busy))
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
        // A validated gap traversal is its own bounded physical transaction.
        // The high-level objective may be moving away from its remote XY
        // anchor while establishing the run-up, and expiring it on the launch
        // frame hands steering to a different mission in mid-air.  Resume the
        // parent budget after the crossing reports landed/short/timed_out.
        if (gapRunning || gapJumpActive)
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
        // A released explosive is still executing the chosen world action
        // while its ordinary weapon animation, flight and fuse resolve.
        // Give that transaction its bounded outcome window instead of
        // classifying the physically necessary retreat/wait as a stall.
        if (!progressed && currentObjective.type == kObjectiveInteraction)
        {
            auto interaction = interactions.find(currentObjective.interactionKey);
            if (interaction != interactions.end()
                && interaction->second.activationMode == llmapper::kActivateDamage
                && interaction->second.effectPhase > 0
                && interaction->second.effectStartedTick >= 0
                && observation.tick - interaction->second.effectStartedTick
                    <= kExplosiveOutcomeTicks)
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
        if (currentObjective.type == kObjectiveInteraction
            && currentObjective.interactionRouteMode == 0)
        {
            // A locally acquired physical target starts with exact pose/support
            // navigation.  A remembered remote target starts with the directed
            // transport graph.  Preserve that decision while travelling so
            // overlapping/layered sectors cannot make routing oscillate.
            currentObjective.interactionRouteMode =
                observation.sector == currentObjective.sector ? 1 : 2;
        }
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
        snprintf(detail, sizeof(detail), "type=%d id=%d sector=%d target_sector=%d wall=%d current=%d route_mode=%d",
                 int(objective.type), objective.id, objective.sector,
                 objective.targetSector, objective.wall, observation.sector,
                 currentObjective.interactionRouteMode);
        event("objective_selected", detail);
    }

    void completeObjective(const char *reason)
    {
        if (!currentObjective.active)
            return;
        const bool completedPrerequisite = suspendedObjectiveActive
            && currentObjective.type == kObjectiveInteraction;
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
        if (completedPrerequisite)
        {
            const Objective resumed = suspendedObjective;
            suspendedObjective = Objective{};
            suspendedObjectiveActive = false;
            dynamicPrerequisiteMechanism = -1;
            dynamicPrerequisiteState = -1;
            selectObjective(resumed, "RESUME_AFTER_ROUTE_CONDITION");
            char resumedDetail[160];
            snprintf(resumedDetail, sizeof(resumedDetail),
                     "type=%d id=%d sector=%d target=%d",
                     int(resumed.type), resumed.id, resumed.sector,
                     resumed.targetSector);
            event("dynamic_prerequisite_satisfied", resumedDetail);
        }
    }

    // Every abandonment makes the opportunity dormant.  Without this an
    // objective that fails is simply reselected on the next tick, which is
    // the mechanism behind most of the bot's tight local loops.  Knowledge
    // is kept; only the bot's attention moves on.
    void invalidateObjective(const char *reason, bool dormant = true)
    {
        if (!currentObjective.active)
            return;
        const bool failedPrerequisite = suspendedObjectiveActive
            && currentObjective.type == kObjectiveInteraction;
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
        if (failedPrerequisite)
        {
            const Objective resumed = suspendedObjective;
            suspendedObjective = Objective{};
            suspendedObjectiveActive = false;
            dynamicPrerequisiteMechanism = -1;
            dynamicPrerequisiteState = -1;
            selectObjective(resumed, "RESUME_AFTER_FAILED_ROUTE_CONDITION");
            event("dynamic_prerequisite_failed", reason);
        }
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
        const int playerCell = nearestNavCell(observation.sector, observation.x,
                                              observation.y,
                                              llmapper::capability::playerFloorZ());
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
            if (extra > 0 && extra < kMaxXWalls
                && busyValueInMotion(xwall[extra].busy))
                return true;
        }
        const int sectorId = memory.target.sectorPush ? memory.target.to
            : memory.target.sectorPushCurrent ? memory.target.from : -1;
        if (inRange(sectorId, 0, numsectors))
        {
            const int extra = sector[sectorId].extra;
            if (extra > 0 && extra < kMaxXSectors
                && busyValueInMotion(xsector[extra].busy))
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

    void rememberInvestigatedBoundary(const Portal &crossing, int hit,
                                      const char *reason)
    {
        // Collision resolution can arrive on the frame after the deliberate
        // inspection has completed.  Retain only a tiny causal window: enough
        // to associate that contact with the route being probed, but not long
        // enough for an unrelated later interaction to inherit stale intent.
        recentInvestigatedWall = crossing.wall;
        recentInvestigatedFrom = crossing.from;
        recentInvestigatedTo = crossing.to;
        recentInvestigatedTick = observation.tick;
        const int signature = portalIdentitySignature(crossing);
        if (llmapper::investigatedNow(investigatedBoundaries, crossing.wall,
                                      crossing.from, crossing.to, signature))
            return;
        InvestigateRecord record;
        record.wall = crossing.wall;
        record.from = crossing.from;
        record.to = crossing.to;
        record.geometrySignature = signature;
        investigatedBoundaries.push_back(record);
        char detail[160];
        snprintf(detail, sizeof(detail), "wall=%d dest=%d hit=%d reason=%s",
                 crossing.wall, currentObjective.targetSector, hit, reason);
        event("blocked_frontier_investigated", detail);
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
            (void)discovered;
            // The concrete interaction now represents this possible route.
            // Retire the weaker geometry probe whether the interaction was
            // learned just now or was already in memory.
            rememberInvestigatedBoundary(crossing, hit,
                                          "actionable_mechanism_found");
            suppressObjective(4000000 + currentObjective.targetSector,
                              "actionable_mechanism_found");
            completeObjective("investigation_found_interaction");
            return input;
        }
        if (nearby || selectNewInteraction(false))
        {
            event("blocked_frontier_action_found", "source=nearby_trigger_push");
            rememberInvestigatedBoundary(crossing, hit,
                                          "nearby_action_opportunity_found");
            suppressObjective(4000000 + currentObjective.targetSector,
                              "nearby_action_opportunity_found");
            completeObjective("investigation_found_interaction");
            return input;
        }
        rememberInvestigatedBoundary(crossing, hit, reason);
        suppressObjective(4000000 + currentObjective.targetSector, "investigated_no_action");
        completeObjective("investigation_no_action");
        return input;
    }

    GINPUT executeObjective()
    {
        if (gapJumpActive)
            return continueGapJump();
        if (observation.tick < gapSettleUntilTick)
        {
            // Momentum off a landing is an asset when the next jump goes the
            // same way.  A run of pillars is meant to be taken at a run --
            // stopping on each one wastes the speed and then has to find
            // room to rebuild it on a block a thousand units across.
            if (!momentumServesTheNextJump())
                return brakeAfterLanding();
            gapSettleUntilTick = -1;
        }
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
            if (interaction.key > 0 && !hasKey(interaction.key))
            {
                char detail[160];
                snprintf(detail, sizeof(detail),
                         "kind=%d id=%d required_key=%d reason=key_not_held",
                         int(interaction.kind), interaction.id, interaction.key);
                event("action_prerequisite_unavailable", detail);
                event("opportunity_deferred", detail);
                // The opportunity itself remains live in memory.  It is not
                // dormant: acquiring the key should reconsider it immediately.
                invalidateObjective("missing_key_prerequisite", false);
                return GINPUT{};
            }
            const unsigned missingEffects = interaction.requiredEffects
                & ~availableEffectCapabilities();
            if (missingEffects)
            {
                pendingEffectPrerequisites |= missingEffects;
                char detail[192];
                snprintf(detail, sizeof(detail),
                         "kind=%d id=%d required_effects=%u missing_effects=%u reason=no_current_satisfier",
                         int(interaction.kind), interaction.id,
                         interaction.requiredEffects, missingEffects);
                event("action_prerequisite_unavailable", detail);
                event("opportunity_deferred", detail);
                invalidateObjective("missing_effect_capability");
                return GINPUT{};
            }
            if (currentObjective.requiredSupportSector >= 0
                && interaction.activationMode == llmapper::kActivateVector)
            {
                int rangedWeapon = 0;
                if (!rangedWeaponAvailable(rangedWeapon))
                {
                    ensureNavTopology();
                    bool meleePoseOnSupport = false;
                    const int reach = meleeReach();
                    const SupportRef requiredFloor(kSupportSectorFloor,
                                                   currentObjective.requiredSupportSector);
                    for (const NavCell &cell : navCells)
                    {
                        if (cell.support != requiredFloor
                            || cell.sector != currentObjective.requiredSupportSector)
                            continue;
                        if (distance2(cell.center.x, cell.center.y,
                                      interaction.x, interaction.y) <= reach * reach)
                        {
                            meleePoseOnSupport = true;
                            break;
                        }
                    }
                    if (!meleePoseOnSupport)
                    {
                        rangedPrerequisitePending = true;
                        char detail[192];
                        snprintf(detail, sizeof(detail),
                                 "kind=%d id=%d required_support=%d reason=no_melee_pose_on_required_support",
                                 int(interaction.kind), interaction.id,
                                 currentObjective.requiredSupportSector);
                        event("action_prerequisite_unavailable", detail);
                        invalidateObjective("missing_ranged_capability");
                        return GINPUT{};
                    }
                }
            }
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
                    if (gap && currentObjective.requiredSupportSector < 0)
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
            if (interaction.activationMode == llmapper::kActivateVector
                && (interaction.state == 2 || interaction.observedKnownWorldDelta))
            {
                completeObjective("vector_activation_answered");
                return GINPUT{};
            }
            if (interaction.activationMode == llmapper::kActivateDamage
                && (interaction.state == 2 || interaction.observedKnownWorldDelta))
            {
                completeObjective("required_effect_changed_world");
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
            if (!onNearSide && currentObjective.type == kObjectiveFrontier)
            {
                // A frontier several rooms away is a high-level sector goal,
                // not a straight-line point in the remote room.  Follow the
                // first known portal toward its source and let the local mesh
                // solve only that room-sized step.  This is also the robust
                // fallback when decorative collision fragments the detailed
                // mesh: already traversed doors remain reusable transport
                // instead of every remaining TODO failing at the same remote
                // coordinate.
                // Prefer the detailed, support-aware route whenever it can
                // represent the whole trip.  The coarse graph is only a
                // fallback after that exact destination has failed once;
                // taking it eagerly loses valid vertical/jump paths whose
                // intermediate sector transitions are not ordinary portal
                // walks (AGTST3/8 exercise both cases).
                const int detailedSignature = navigationSignature(
                    crossing->x, crossing->y, crossing->z,
                    crossing->from, crossing->wall);
                Portal route;
                if (navigationFailureSignature == detailedSignature
                    && navigationFailureCount > 0
                    && findKnownRoute(crossing->from, route))
                {
                    char detail[192];
                    snprintf(detail, sizeof(detail),
                             "frontier_wall=%d destination_source=%d route_wall=%d from=%d to=%d",
                             crossing->wall, crossing->from, route.wall,
                             route.from, route.to);
                    event("frontier_coarse_route_step", detail);
                    currentObjective.routeStepWall = route.wall;
                    currentObjective.routeStepFrom = route.from;
                    currentObjective.routeStepTo = route.to;
                    return steerPortal(route);
                }
            }
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
            // A pickup is collected by occupying its physical support, not by
            // aiming at its sprite origin.  Preserve the exact collision-
            // derived support under it; the more permissive same-area stance
            // used for switches can otherwise substitute the pit floor under
            // a bridge and make the bot orbit beside the item forever.
            ensureNavTopology();
            const int stance = navCellInSector(object->sector, object->x, object->y,
                                                object->z);
            if (stance >= 0)
            {
                const SupportRef requiredSupport = navCells[size_t(stance)].support;
                return navigateTo(object->x, object->y, object->z, object->sector,
                                  object->sprite, kTraversalUnknown, false,
                                  &requiredSupport);
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
            && movementTargetSector != movementTargetFrom
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

    std::set<int> knownNavSectors() const
    {
        std::set<int> sectors = observedSectors;
        sectors.insert(visitedSectors.begin(), visitedSectors.end());
        if (inRange(observation.sector, 0, numsectors))
            sectors.insert(observation.sector);
        for (const Portal &portal : observation.portals)
            if (inRange(portal.to, 0, numsectors))
                sectors.insert(portal.to);
        for (const auto &entry : knownGraph)
        {
            if (!visitedSectors.count(entry.first)
                && !observedSectors.count(entry.first))
                continue;
            for (const Portal &portal : entry.second)
                if (inRange(portal.to, 0, numsectors))
                    sectors.insert(portal.to);
        }
        return sectors;
    }

    int navTopologyIdentity() const
    {
        int signature = 17;
        const std::set<int> sectors = knownNavSectors();
        for (int sectorId : sectors)
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
        const std::set<int> sectors = knownNavSectors();
        for (int sectorId : sectors)
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
            for (int nSprite = headspritesect[sectorId]; nSprite >= 0;
                 nSprite = nextspritesect[nSprite])
            {
                if (gMe && gMe->pSprite && nSprite == gMe->pSprite->index)
                    continue;
                if (!(sprite[nSprite].cstat & CSTAT_SPRITE_BLOCK)
                    || sprite[nSprite].statnum == kStatDude)
                    continue;
                signature = llmapper::mixHash(signature, nSprite);
                signature = llmapper::mixHash(signature, sprite[nSprite].x);
                signature = llmapper::mixHash(signature, sprite[nSprite].y);
                signature = llmapper::mixHash(signature, sprite[nSprite].z);
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
        const std::set<int> sectors = knownNavSectors();
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
                signature = signature * 31 + (xsector[record.extra].busy != 0);
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
                    signature = signature * 31 + (xwall[wallRecord.extra].busy != 0);
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

    int64_t supportCenterDistance2(const NavCell &cell) const
    {
        if (cell.support.kind != kSupportSpriteFloor
            || !inRange(cell.support.id, 0, kMaxSprites))
            return 0;
        const spritetype &support = sprite[cell.support.id];
        return distance2(cell.center.x, cell.center.y, support.x, support.y);
    }

    // Floor sprites commonly end exactly on a two-sided sector wall.  At
    // that mathematical edge GetZRange may report the pit below on one side
    // even though cells a short distance into both sectors are confirmed
    // standable at the same height.  Join those two engine-confirmed cells;
    // this is a support transition across a real portal, not permission to
    // pass through arbitrary blocking sprites.
    bool addSpriteSupportBoundaryLink(const Portal &portal)
    {
        if (!inRange(portal.wall, 0, numwalls)
            || (wall[portal.wall].cstat & 1)
            || portal.openingWidth < playerPassageWidth())
            return false;
        const int boundaryReach = 3 << kNavGridShift;
        const int64_t near2 = int64_t(boundaryReach) * boundaryReach;
        const int span = 4 << kNavGridShift;
        const int64_t span2 = int64_t(span) * span;
        int bestFrom = -1;
        int bestTo = -1;
        int64_t bestCost = INT64_MAX;
        for (const NavCell &from : navCells)
        {
            if (from.sector != portal.from
                || dist2ToSegment(from.center.x, from.center.y,
                                  portal.x1, portal.y1,
                                  portal.x2, portal.y2) > near2)
                continue;
            for (const NavCell &to : navCells)
            {
                if (to.sector != portal.to
                    || (from.support.kind != kSupportSpriteFloor
                        && to.support.kind != kSupportSpriteFloor)
                    || std::abs(to.z - from.z) > kMaxWalkableStep
                    || dist2ToSegment(to.center.x, to.center.y,
                                      portal.x1, portal.y1,
                                      portal.x2, portal.y2) > near2)
                    continue;
                const int64_t distance = distance2(from.center.x, from.center.y,
                                                   to.center.x, to.center.y);
                const int64_t cost = distance + supportCenterDistance2(from)
                    + supportCenterDistance2(to);
                if (distance > span2 || cost >= bestCost)
                    continue;
                bestCost = cost;
                bestFrom = from.id;
                bestTo = to.id;
            }
        }
        if (bestFrom < 0 || bestTo < 0)
            return false;
        const int delta = navCells[size_t(bestTo)].z - navCells[size_t(bestFrom)].z;
        NavEdgeMode forwardMode = delta == 0 ? kNavWalk : kNavStep;
        NavEdgeMode reverseMode = forwardMode;
        // clipmove evaluates the sector planes while the player's centre is
        // still on the sprite side of the wall.  Leaving a bridge for a
        // higher sector can therefore require a jump over that portal lip,
        // even though the sprite surface and landing cells have equal Z.
        if (portal.floorDelta < -kMaxWalkableStep)
            forwardMode = kNavJump;
        if (portal.floorDelta > kMaxWalkableStep)
            reverseMode = kNavJump;
        // The generic portal point may have been shifted along the wall to
        // avoid a blocking sprite.  That is useful for an open-floor
        // doorway, but lethal for a narrow bridge: it turns the final hop
        // into a diagonal leap off the plank.  This link already has an
        // engine-confirmed support cell on each side, so make the cell being
        // entered the directed crossing waypoint.  Steering therefore
        // continues across the boundary and onto known floor rather than
        // declaring success at an arbitrary point on the wall plane.
        const LocalWaypoint forwardGateway(navCells[size_t(bestTo)].center.x,
                                            navCells[size_t(bestTo)].center.y);
        const LocalWaypoint reverseGateway(navCells[size_t(bestFrom)].center.x,
                                            navCells[size_t(bestFrom)].center.y);
        addNavLink(bestFrom, bestTo, forwardMode, portal.wall, forwardGateway, true);
        addNavLink(bestTo, bestFrom, reverseMode, portal.wall, reverseGateway, true);
        return true;
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
                    || navCellInSector(portal.from, portal.x, portal.y,
                                       portal.fromFloorZ) < 0
                    || navCellInSector(portal.to, portal.x, portal.y,
                                       portal.toFloorZ) < 0)
                {
                    if (addSpriteSupportBoundaryLink(portal))
                    {
                        ++linked;
                        continue;
                    }
                    ++rejected;
                    continue;
                }
                const NavEdgeMode mode = navModeForPortal(portal);
                if (!llmapper::traversableMode(mode))
                {
                    ++rejected;
                    continue;
                }
                // A gap crossing has no doorway: its two ends are a lip and
                // a landing several thousand units apart, and feeding those
                // to the doorway linker made a nav edge between whatever
                // cells happened to lie near each -- a route straight over
                // the hole that the bot then walked into.  Link the lip to
                // the landing explicitly, one way, as a jump.
                if (portal.gapJump)
                {
                    const int lip = navCellInSector(portal.from, portal.takeoffX,
                                                    portal.takeoffY,
                                                    portal.fromFloorZ);
                    const int landing = navCellInSector(portal.to, portal.x, portal.y,
                                                        portal.toFloorZ);
                    if (lip < 0 || landing < 0)
                    {
                        ++rejected;
                        continue;
                    }
                    addNavLink(lip, landing, kNavJump, portal.wall,
                               LocalWaypoint(portal.takeoffX, portal.takeoffY), true);
                    ++linked;
                    continue;
                }
                const int from = nearestNavCell(portal.from, portal.x, portal.y,
                                                portal.fromFloorZ);
                const int to = nearestNavCell(portal.to, portal.x, portal.y,
                                              portal.toFloorZ);
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

    std::vector<int> navCellsAt(int sectorId, int gx, int gy) const
    {
        auto entry = navCellIndex.find(navCellHandle(sectorId, gx, gy));
        return entry == navCellIndex.end() ? std::vector<int>() : entry->second;
    }

    int navCellAt(int sectorId, int gx, int gy, int z = INT32_MIN,
                  const SupportRef *support = nullptr) const
    {
        const std::vector<int> candidates = navCellsAt(sectorId, gx, gy);
        int best = -1;
        int bestZ = INT32_MAX;
        for (int id : candidates)
        {
            const NavCell &cell = navCells[size_t(id)];
            if (support && cell.support != *support)
                continue;
            const int dz = z == INT32_MIN ? cell.z : std::abs(cell.z - z);
            if (best < 0 || dz < bestZ)
            {
                best = id;
                bestZ = dz;
            }
        }
        return best;
    }

    // Strict: only ever returns a cell of the sector asked for.
    int navCellInSector(int sectorId, int x, int y, int z = INT32_MIN,
                        const SupportRef *support = nullptr) const
    {
        const int exact = navCellAt(sectorId, x >> kNavGridShift,
                                    y >> kNavGridShift, z, support);
        if (exact >= 0)
            return exact;
        int best = -1;
        int bestDistance = INT32_MAX;
        int bestZ = INT32_MAX;
        for (const NavCell &cell : navCells)
        {
            if (cell.sector != sectorId)
                continue;
            if (support && cell.support != *support)
                continue;
            const int currentDistance = distance2(x, y, cell.center.x, cell.center.y);
            const int currentZ = z == INT32_MIN ? cell.z : std::abs(cell.z - z);
            if (currentZ < bestZ || (currentZ == bestZ && currentDistance < bestDistance))
            {
                bestZ = currentZ;
                bestDistance = currentDistance;
                best = cell.id;
            }
        }
        return best;
    }

    int navCellInSectorArea(int sectorId, int x, int y, int area) const
    {
        int best = -1;
        int bestDistance = INT32_MAX;
        for (const NavCell &cell : navCells)
        {
            if (cell.sector != sectorId || cell.walkArea != area)
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

    // Horizontal distance a standing jump can cover by the time the feet
    // descend through a destination support height.  This uses the same
    // per-frame player-motion routine as the live jump predictor, starting
    // from rest, so isolated sprite tops are linked by actual Caleb physics
    // rather than a guessed platform distance.  Collision is checked
    // separately when the candidate graph edge is built.
    int standingJumpReachForDelta(int destinationFloorDelta) const
    {
        if (!gMe)
            return 0;
        const POSTURE &stand = gMe->pPosture[gMe->lifeMode][kPostureStand];
        const int foot = llmapper::capability::playerFootOffset();
        llmapper::capability::MotionState state;
        state.z = -foot;
        state.zvel = llmapper::capability::playerJumpImpulse();
        // The first movement frame still starts on the take-off floor.  Using
        // the eventual landing plane for that frame makes a lower landing
        // look as though Caleb begins high in the air; using an artificial
        // floor far below does the same for every jump.  ProcessInput then
        // suppresses all horizontal acceleration (height >=
        // kDudeAirborneHeight), so a perfectly valid standing jump is
        // measured as having zero range.  Advance once against the real
        // take-off floor, then expose a lower destination so the arc is not
        // snapped back to the ledge on descent.
        bool reachedDestinationHeight = destinationFloorDelta >= 0;
        for (int frame = 1; frame <= 240; ++frame)
        {
            const bool falling = state.zvel > 0;
            const int simulationFloor = frame == 1
                ? 0 : std::max(0, destinationFloorDelta);
            llmapper::capability::stepPlayerMotion(
                state, kFullThrottle, 0, simulationFloor,
                stand.frontAccel, foot,
                llmapper::capability::playerAirDrag());
            if (state.z + foot <= destinationFloorDelta)
                reachedDestinationHeight = true;
            if (falling && reachedDestinationHeight
                && state.z + foot >= destinationFloorDelta)
                return std::abs(state.x);
        }
        return 0;
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
    int nearestNavCellInArea(int x, int y, int area, int z = INT32_MIN) const
    {
        if (area < 0)
            return -1;
        int best = -1;
        int bestDistance = INT32_MAX;
        int bestZ = INT32_MAX;
        for (const NavCell &cell : navCells)
        {
            if (cell.walkArea != area)
                continue;
            const int currentDistance = distance2(x, y, cell.center.x, cell.center.y);
            const int currentZ = z == INT32_MIN ? cell.z : std::abs(cell.z - z);
            if (currentZ < bestZ || (currentZ == bestZ && currentDistance < bestDistance))
            {
                bestZ = currentZ;
                bestDistance = currentDistance;
                best = cell.id;
            }
        }
        return best;
    }

    int nearestNavCell(int sectorId, int x, int y, int z = INT32_MIN,
                       const SupportRef *support = nullptr) const
    {
        const int exact = navCellAt(sectorId, x >> kNavGridShift,
                                    y >> kNavGridShift, z, support);
        if (exact >= 0)
            return exact;
        // Off-grid poses (a doorway tighter than the body margin, a moving
        // floor mid-travel) still need a handle on the mesh: fall back to
        // the closest cell of the same sector, then to the closest anywhere.
        int best = -1;
        int bestDistance = INT32_MAX;
        int bestZ = INT32_MAX;
        int fallback = -1;
        int fallbackDistance = INT32_MAX;
        int fallbackZ = INT32_MAX;
        for (const NavCell &cell : navCells)
        {
            if (support && cell.support != *support)
                continue;
            const int currentDistance = distance2(x, y, cell.center.x, cell.center.y);
            const int currentZ = z == INT32_MIN ? cell.z : std::abs(cell.z - z);
            if (cell.sector == sectorId)
            {
                if (currentZ < bestZ
                    || (currentZ == bestZ && currentDistance < bestDistance))
                {
                    bestZ = currentZ;
                    bestDistance = currentDistance;
                    best = cell.id;
                }
            }
            else if (currentZ < fallbackZ
                     || (currentZ == fallbackZ && currentDistance < fallbackDistance))
            {
                fallbackZ = currentZ;
                fallbackDistance = currentDistance;
                fallback = cell.id;
            }
        }
        return best >= 0 ? best : fallback;
    }

    int currentNavStartCell()
    {
        const int exact = nearestNavCell(observation.sector, observation.x,
                                         observation.y,
                                         llmapper::capability::playerFloorZ());
        if (exact < 0 || !navCells[size_t(exact)].links.empty())
            return exact;

        // A live off-grid anchor can be isolated while the ordinary floor
        // mesh is one step away (notably while the initial body envelope is
        // settling). Enter through the nearest floor cell the copied player
        // body can really reach instead of making all supports another area.
        int best = -1;
        int64_t bestDistance = INT64_MAX;
        for (const NavCell &cell : navCells)
        {
            if (cell.sector != observation.sector || cell.links.empty()
                || std::abs(cell.z - llmapper::capability::playerFloorZ())
                    > kMaxWalkableStep)
                continue;
            const int64_t distance = distance2(observation.x, observation.y,
                                               cell.center.x, cell.center.y);
            if (distance >= bestDistance)
                continue;
            const MovementProbe entry = probeMovement(
                observation.x, observation.y, observation.z,
                observation.sector, cell.center.x, cell.center.y,
                cell.sector, std::max(256, playerClipRadius()));
            if (!entry.reachable)
                continue;
            bestDistance = distance;
            best = cell.id;
        }
        if (best >= 0 && !reportedNavStartReconnect)
        {
            event("nav_start_anchor_reconnected",
                  "reason=isolated_live_pose reachable_floor_cell=1");
            reportedNavStartReconnect = true;
        }
        return best >= 0 ? best : exact;
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

        auto appendCell = [&](int gx, int gy, int x, int y, int z,
                              const SupportRef &support) {
            for (const GridCell &known : grid.cells)
                if (known.gx == gx && known.gy == gy && known.z == z
                    && known.support == support)
                    return;
            GridCell cell;
            cell.gx = gx;
            cell.gy = gy;
            cell.x = x;
            cell.y = y;
            cell.z = z;
            cell.support = support;
            grid.cells.push_back(cell);
        };

        auto appendSurfaces = [&](int gx, int gy, int x, int y, int required) {
            const std::vector<StandableSurface> surfaces = standableSurfacesAt(
                sectorId, x, y, playerClipRadius(), required);
            for (const StandableSurface &surface : surfaces)
                appendCell(gx, gy, x, y, surface.z, surface.support);

            // A moving sector floor is one physical support with multiple
            // stable poses.  Keep both endpoint layers in the graph even
            // while only one is present in live collision geometry.
            if (sectorRecord.extra > 0 && sectorRecord.extra < kMaxXSectors)
            {
                const XSECTOR &dynamic = xsector[sectorRecord.extra];
                if (dynamic.offFloorZ != dynamic.onFloorZ)
                {
                    if (dynamic.offFloorZ - dynamic.offCeilZ >= required)
                        appendCell(gx, gy, x, y, dynamic.offFloorZ,
                                   SupportRef(kSupportSectorFloor, sectorId));
                    if (dynamic.onFloorZ - dynamic.onCeilZ >= required)
                        appendCell(gx, gy, x, y, dynamic.onFloorZ,
                                   SupportRef(kSupportSectorFloor, sectorId));
                }
            }
        };

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

        const int clearance = playerStandingClearance();
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
                    appendSurfaces(gx, gy, bestX, bestY, required);
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

        // Upright solid sprites are obstacle volumes at the sector floor,
        // but Build exposes their top extents as real getzrange floors.  The
        // ordinary floor pass intentionally rejects points occupied by those
        // volumes, so seed their independently valid upper support layers in
        // a bounded box around each sprite.  Every emitted cell is confirmed
        // by the engine for the player's actual clip radius and headroom.
        for (int spriteId = headspritesect[sectorId]; spriteId >= 0;
             spriteId = nextspritesect[spriteId])
        {
            const spritetype &supportSprite = sprite[spriteId];
            if (gMe && gMe->pSprite && spriteId == gMe->pSprite->index)
                continue;
            if (!(supportSprite.cstat & CSTAT_SPRITE_BLOCK))
                continue;
            const int alignment = supportSprite.cstat & CSTAT_SPRITE_ALIGNMENT_MASK;
            // A wall-aligned sprite is a two-dimensional collision sheet.
            // GetZRange can report its upper extent at the exact line, but it
            // has no finite top footprint on which the player's clip circle
            // can settle. Other upright solids have a collision volume, so
            // their engine-confirmed top can be a support irrespective of
            // tile, voxel, visual shape, or gameplay type.
            if (alignment == CSTAT_SPRITE_ALIGNMENT_FLOOR
                || alignment == CSTAT_SPRITE_ALIGNMENT_SLOPE
                || alignment == CSTAT_SPRITE_ALIGNMENT_WALL)
                continue;
            int supportTop = supportSprite.z, supportBottom = supportSprite.z;
            spriteCollisionExtents(spriteId, supportTop, supportBottom);
            if (supportTop >= sectorRecord.floorz
                || supportTop - getceilzofslope(sectorId,
                                                supportSprite.x,
                                                supportSprite.y) < clearance)
                continue;

            int supportMinX = supportSprite.x;
            int supportMaxX = supportSprite.x;
            int supportMinY = supportSprite.y;
            int supportMaxY = supportSprite.y;
            const int footprintMargin = radius + (supportSprite.clipdist << 2) + 64;
            supportMinX -= footprintMargin;
            supportMaxX += footprintMargin;
            supportMinY -= footprintMargin;
            supportMaxY += footprintMargin;
            const SupportRef wanted(kSupportSpriteFloor, spriteId);
            const size_t cellsBeforeSupport = grid.cells.size();
            for (int gy = supportMinY >> kNavGridShift;
                 gy <= (supportMaxY >> kNavGridShift); ++gy)
            {
                for (int gx = supportMinX >> kNavGridShift;
                     gx <= (supportMaxX >> kNavGridShift); ++gx)
                {
                    const int baseX = (gx << kNavGridShift)
                        + (1 << (kNavGridShift - 1));
                    const int baseY = (gy << kNavGridShift)
                        + (1 << (kNavGridShift - 1));
                    bool placed = false;
                    for (int ox = -1; ox <= 1 && !placed; ++ox)
                    {
                        for (int oy = -1; oy <= 1 && !placed; ++oy)
                        {
                            const int x = baseX + ox * step;
                            const int y = baseY + oy * step;
                            if (inside(x, y, sectorId) != 1)
                                continue;
                            const std::vector<StandableSurface> surfaces =
                                standableSurfacesAt(sectorId, x, y, radius,
                                                    clearance);
                            for (const StandableSurface &surface : surfaces)
                            {
                                if (surface.support != wanted)
                                    continue;
                                appendCell(gx, gy, x, y, surface.z,
                                           surface.support);
                                placed = true;
                                break;
                            }
                        }
                    }
                }
            }
            const size_t supportCellCount = grid.cells.size() - cellsBeforeSupport;
            if (supportCellCount > 0
                && reportedSpriteSupports.insert(spriteId).second)
            {
                char detail[192];
                snprintf(detail, sizeof(detail),
                         "sprite=%d sector=%d alignment=%d top=%d cells=%u",
                         spriteId, sectorId, alignment, supportTop,
                         unsigned(supportCellCount));
                event("nav_sprite_support_seeded", detail);
            }
            else if (supportCellCount == 0
                      && reportedRejectedSpriteSupports.insert(spriteId).second)
            {
                int probeCeiling = 0, probeCeilingHit = 0;
                int probeFloor = 0, probeFloorHit = 0;
                GetZRangeAtXYZ(supportSprite.x, supportSprite.y, supportTop - 1,
                               sectorId, &probeCeiling, &probeCeilingHit,
                               &probeFloor, &probeFloorHit, radius, CLIPMASK0,
                               PARALLAXCLIP_CEILING | PARALLAXCLIP_FLOOR);
                char detail[256];
                snprintf(detail, sizeof(detail),
                         "sprite=%d sector=%d alignment=%d collision_top=%d collision_bottom=%d floor=%d floor_hit=0x%x ceiling=%d clearance=%d radius=%d",
                         spriteId, sectorId, alignment,
                         supportTop, supportBottom, probeFloor, probeFloorHit,
                         probeCeiling, supportTop - probeCeiling, radius);
                event("nav_sprite_support_rejected", detail);
            }
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
                appendSurfaces(gx, gy, midX, midY, 0);
                if (grid.cells.empty())
                    appendCell(gx, gy, midX, midY,
                               getflorzofslope(sectorId, midX, midY),
                               SupportRef(kSupportSectorFloor, sectorId));
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
            const SupportRef support = currentPlayerSupport();
            const int playerFloor = llmapper::capability::playerFloorZ();
            bool present = false;
            for (const GridCell &cell : grid.cells)
                if (cell.gx == gx && cell.gy == gy && cell.z == playerFloor
                    && cell.support == support)
                    present = true;
            if (!present)
            {
                appendCell(gx, gy,
                           (gx << kNavGridShift) + (1 << (kNavGridShift - 1)),
                           (gy << kNavGridShift) + (1 << (kNavGridShift - 1)),
                           playerFloor, support);
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

    // A jump may legitimately pass through an open two-sided boundary while
    // climbing onto a support beside/above a raised sector.  Such a boundary
    // changes sector ownership during the arc, but is not a collision wall.
    // Keep one-sided and explicitly blocking walls authoritative.
    bool segmentCrossesBlockingWall(int sectorId, int ax, int ay,
                                    int bx, int by) const
    {
        if (!inRange(sectorId, 0, numsectors))
            return true;
        const sectortype &record = sector[sectorId];
        for (int i = 0; i < record.wallnum; ++i)
        {
            const int wallId = record.wallptr + i;
            if (!inRange(wallId, 0, numwalls)
                || !inRange(wall[wallId].point2, 0, numwalls))
                continue;
            const walltype &start = wall[wallId];
            if (inRange(start.nextsector, 0, numsectors)
                && !(start.cstat & CSTAT_WALL_BLOCK)
                && (!inRange(start.nextwall, 0, numwalls)
                    || !(wall[start.nextwall].cstat & CSTAT_WALL_BLOCK)))
                continue;
            const walltype &end = wall[start.point2];
            const int64_t d1 = int64_t(bx - ax) * (start.y - ay)
                - int64_t(by - ay) * (start.x - ax);
            const int64_t d2 = int64_t(bx - ax) * (end.y - ay)
                - int64_t(by - ay) * (end.x - ax);
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
            if (!(sprite[nSprite].cstat & CSTAT_SPRITE_BLOCK)
                || sprite[nSprite].statnum == kStatDude)
                continue;
            signature = llmapper::mixHash(signature, sprite[nSprite].x);
            signature = llmapper::mixHash(signature, sprite[nSprite].y);
            signature = llmapper::mixHash(signature, sprite[nSprite].z);
            signature = llmapper::mixHash(signature, sprite[nSprite].cstat);
            signature = llmapper::mixHash(signature, sprite[nSprite].ang);
            signature = llmapper::mixHash(signature, sprite[nSprite].xrepeat);
            signature = llmapper::mixHash(signature, sprite[nSprite].yrepeat);
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
                navPoseChangedTick = observation.tick;
                navPoseSignatureValue = pose;
            }
            else if (navMeshInMotion)
            {
                // Stability is elapsed simulation time, not a count of calls
                // to ensureNavTopology().  Several planning layers call this
                // accessor in one frame; counting those calls made continuously
                // moving retail geometry rebuild a 20k-cell mesh many times in
                // the same simulated second.  An idle local sector still permits
                // an interim rebuild while a remote door/support travels, but
                // no more than once per second.  A genuinely settled pose gets
                // its final graph after eight rendered frames.
                const bool poseStable = observation.tick - navPoseChangedTick
                    >= 8 * kTicsPerFrame;
                const bool periodicLivePose =
                    !busyValueInMotion(observation.localSectorBusy)
                    && observation.tick - navPoseRebuildTick >= kTicsPerSec;
                if (poseStable || periodicLivePose)
                {
                    navMeshInMotion = false;
                    refreshDynamicNavLinks();
                    navDynamicSignatureValue = dynamicSignature;
                    // Moving sprites can cross sector boundaries and change
                    // which support layer occupies a grid square.  Rebuild
                    // the affected surface graph once the pose is stable.
                    navTopologySignature = 0;
                    rearmFrontiersOnNextTopologyRebuild = true;
                    event("nav_mesh_settled", poseStable
                          ? "reason=pose_stable"
                          : "reason=periodic_live_pose");
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
        navPoseChangedTick = observation.tick;
        navPoseRebuildTick = observation.tick;
        navCells.clear();
        navCellIndex.clear();
        // The identity above and the cells below must describe the same
        // topology.  In particular, a visible/remembered portal makes its
        // far-side landing eligible before Caleb enters that sector; omitting
        // it here left the cached signature up to date while rebuilding only
        // the already-entered side.
        const std::set<int> navSectors = knownNavSectors();
        for (int sectorId : navSectors)
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
                cell.z = source.z;
                cell.support = source.support;
                cell.center = LocalWaypoint(source.x, source.y);
                navCellIndex[navCellHandle(sectorId, source.gx, source.gy)].push_back(cell.id);
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
                const std::vector<int> neighbours = navCellsAt(
                    cell.sector, cell.gx + kStepX[d], cell.gy + kStepY[d]);
                for (int neighbour : neighbours)
                {
                    if (neighbour <= int(i))
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
                    const int midpointBlocker = solidSpriteAt(
                        cell.sector, midX, midY, playerClipRadius());
                    const bool blockerIsSupport = midpointBlocker >= 0
                        && ((cell.support.kind == kSupportSpriteFloor
                             && cell.support.id == midpointBlocker)
                            || (other.support.kind == kSupportSpriteFloor
                                && other.support.id == midpointBlocker));
                    if (midpointBlocker >= 0 && !blockerIsSupport)
                        continue;
                // Deliberately geometry, not cansee(): that is a sight test
                // and is blocked by sprites, so a courtyard with scenery in
                // it was carved into separate walk areas by its own torches.
                    if (segmentCrossesSectorWall(cell.sector, cell.center.x, cell.center.y,
                                                 other.center.x, other.center.y))
                        continue;
                    const int floorDelta = other.z - cell.z;
                    const LocalWaypoint gateway(midX, midY);
                    auto directedMode = [&](int delta) {
                        if (std::abs(delta) <= kMaxWalkableStep)
                            return delta == 0 ? kNavWalk : kNavStep;
                        if (delta < 0 && -delta <= playerJumpRiseLimit())
                            return kNavJump;
                        if (delta > 0 && delta <= playerJumpRiseLimit())
                            return kNavDrop;
                        return kNavBlocked;
                    };
                    const NavEdgeMode forward = directedMode(floorDelta);
                    const NavEdgeMode reverse = directedMode(-floorDelta);
                    if (llmapper::traversableMode(forward))
                        addNavLink(int(i), neighbour, forward, -1, gateway, true);
                    if (llmapper::traversableMode(reverse))
                        addNavLink(neighbour, int(i), reverse, -1, gateway, true);
                }
            }
        }

        // An isolated upright sprite (grave, post, fence) can be a real
        // stepping stone even though its top does not touch the eight
        // neighbouring floor cells.  The local grid above represents the
        // surfaces correctly but cannot express a jump across the XY gap.
        // Add only edges involving an engine-confirmed sprite support, bound
        // their range with the player's simulated standing jump, and reject
        // any segment crossing a sector wall or an unrelated solid sprite.
        // Ordinary floor-to-floor gaps remain the explicit gap-crossing
        // system's responsibility.
        const int landingTolerance = playerClipRadius() + (1 << (kNavGridShift - 2));
        // Every sprite-support candidate used to replay the same 240-frame
        // standing-jump simulation.  A topology contains many cells but only
        // a handful of distinct support-height deltas, so cache the engine-
        // derived answer for this rebuild.  The cache is deliberately local:
        // inventory/posture capability changes naturally receive fresh
        // physics on the next topology construction.
        std::map<int, int> standingReachByDelta;
        auto cachedStandingReach = [&](int delta) {
            auto found = standingReachByDelta.find(delta);
            if (found != standingReachByDelta.end())
                return found->second;
            const int reach = standingJumpReachForDelta(delta);
            standingReachByDelta.emplace(delta, reach);
            return reach;
        };
        auto addSpriteJump = [&](size_t fromId, size_t toId) {
            if (fromId == toId)
                return;
            const NavCell &from = navCells[fromId];
            const NavCell &to = navCells[toId];
            if (from.sector != to.sector || from.support == to.support)
                return;
            const int delta = to.z - from.z;
            // MoveDude settles a player on a blocking sprite once the
            // player's centre is above the sprite plane and the body bottom
            // overlaps it.  Requiring the feet arc itself to clear the plane
            // underestimates climbable solid objects by the measured foot
            // offset.  Sector-floor transitions keep their stricter portal
            // model; this adjustment is only for an engine-resolved sprite
            // destination.
            const int landingDelta = to.support.kind == kSupportSpriteFloor
                ? delta + llmapper::capability::playerFootOffset() : delta;
            const int reach = cachedStandingReach(landingDelta);
            if (reach <= 0)
                return;
            const int64_t span = distance2(from.center.x, from.center.y,
                                           to.center.x, to.center.y);
            const int maxSpan = reach + landingTolerance;
            if (span > int64_t(maxSpan) * maxSpan)
                return;
            if (segmentCrossesBlockingWall(from.sector,
                                           from.center.x, from.center.y,
                                           to.center.x, to.center.y))
                return;
            for (int sample = 1; sample < 4; ++sample)
            {
                const int x = from.center.x
                    + int(int64_t(to.center.x - from.center.x) * sample / 4);
                const int y = from.center.y
                    + int(int64_t(to.center.y - from.center.y) * sample / 4);
                const int blocker = solidSpriteAt(from.sector, x, y,
                                                  playerClipRadius());
                const bool isEndpointSupport = blocker >= 0
                    && ((from.support.kind == kSupportSpriteFloor
                         && from.support.id == blocker)
                        || (to.support.kind == kSupportSpriteFloor
                            && to.support.id == blocker));
                if (blocker >= 0 && !isEndpointSupport)
                    return;
            }
            addNavLink(int(fromId), int(toId), kNavJump, -1,
                       LocalWaypoint(from.center.x, from.center.y), true);
        };
        std::vector<size_t> spriteSupportCells;
        for (size_t i = 0; i < navCells.size(); ++i)
            if (navCells[i].support.kind == kSupportSpriteFloor)
                spriteSupportCells.push_back(i);
        for (size_t supportCell : spriteSupportCells)
        {
            for (size_t j = 0; j < navCells.size(); ++j)
            {
                addSpriteJump(supportCell, j);
                if (navCells[j].support.kind != kSupportSpriteFloor)
                    addSpriteJump(j, supportCell);
            }
        }
        // Keep the capability diagnosable as a graph fact, not merely as
        // surface discovery.  A seeded top with no incoming jump edge is
        // still an unreachable island and explains the visible behaviour of
        // running into (or avoiding) the obstacle instead of climbing it.
        std::set<int> linkedSpriteSupports;
        for (size_t supportCell : spriteSupportCells)
            linkedSpriteSupports.insert(navCells[supportCell].support.id);
        for (int spriteId : linkedSpriteSupports)
        {
            // This is explanatory telemetry, not graph construction.  Once a
            // support has been reported, avoid repeating the quadratic scan
            // on every moving-geometry rebuild.
            if (reportedSpriteSupportLinks.count(spriteId))
                continue;
            int cells = 0;
            int incoming = 0;
            int outgoing = 0;
            int nearestSpan = INT32_MAX;
            int nearestDelta = 0;
            for (size_t i = 0; i < navCells.size(); ++i)
            {
                const NavCell &cell = navCells[i];
                if (cell.support == SupportRef(kSupportSpriteFloor, spriteId))
                {
                    ++cells;
                    for (const NavLink &link : cell.links)
                        if (link.mode == kNavJump && link.target >= 0
                            && link.target < int(navCells.size())
                            && navCells[size_t(link.target)].support != cell.support)
                            ++outgoing;
                    for (const NavCell &other : navCells)
                    {
                        if (other.sector != cell.sector
                            || other.support == cell.support)
                            continue;
                        const int span = int(std::sqrt(double(distance2(
                            cell.center.x, cell.center.y,
                            other.center.x, other.center.y))));
                        if (span < nearestSpan)
                        {
                            nearestSpan = span;
                            nearestDelta = cell.z - other.z;
                        }
                    }
                }
                for (const NavLink &link : cell.links)
                    if (link.mode == kNavJump && link.target >= 0
                        && link.target < int(navCells.size())
                        && navCells[size_t(link.target)].support
                            == SupportRef(kSupportSpriteFloor, spriteId)
                        && cell.support != navCells[size_t(link.target)].support)
                        ++incoming;
            }
            reportedSpriteSupportLinks.insert(spriteId);
            char supportDetail[192];
            snprintf(supportDetail, sizeof(supportDetail),
                     "sprite=%d cells=%d incoming=%d outgoing=%d nearest_span=%d nearest_delta=%d jump_reach=%d",
                     spriteId, cells, incoming, outgoing,
                     nearestSpan == INT32_MAX ? -1 : nearestSpan, nearestDelta,
                     standingJumpReachForDelta(nearestDelta));
            event("nav_sprite_support_links", supportDetail);
        }

        refreshDynamicNavLinks();
        ++navTopologyRevision;
        if (rearmFrontiersOnNextTopologyRebuild)
        {
            rearmFrontiersOnNextTopologyRebuild = false;
            rearmFrontiersForNewTopology();
        }
        navPoseRebuildTick = observation.tick;
        int areas = 0;
        for (const NavCell &cell : navCells)
            if (cell.walkArea + 1 > areas)
                areas = cell.walkArea + 1;
        char detail[128];
        snprintf(detail, sizeof(detail), "revision=%d cells=%u areas=%d sectors=%u",
                 navTopologyRevision, unsigned(navCells.size()), areas,
                 unsigned(navSectors.size()));
        event("nav_topology_rebuilt", detail);
        if (areas > 1)
        {
            // A fragmented mesh silently reports real routes as unreachable.
            // Name the split so it is diagnosable from telemetry alone.
            for (int sectorId : navSectors)
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
        return llmapper::classifyTraversal(toFloor - fromFloor, clearance, playerStandingClearance(),
                                           playerJumpRiseLimit(), kMaxWalkableStep, probe.reachable,
                                           probe.wall >= 0, useable, probe.sector >= 0);
    }

    LocalWaypoint navStepWaypoint(const NavRouteStep &step) const
    {
        // A cell along the edge of a floor sprite is engine-standable, but it
        // is not a robust locomotion target.  Execute a sprite-support step
        // through the live object centre so a bridge chain becomes the same
        // narrow centerline a player follows.  Reading the live coordinates
        // also preserves this rule for moving sprite supports.
        if (step.targetSupport.kind == kSupportSpriteFloor
            && inRange(step.targetSupport.id, 0, kMaxSprites))
        {
            const spritetype &support = sprite[step.targetSupport.id];
            return LocalWaypoint(support.x, support.y);
        }
        return step.hasGateway ? step.gateway : step.destination;
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
                // A floor-sprite route is a centerline, not open floor.
                // Horizontal clipmove can validate a diagonal shortcut while
                // saying nothing about the pit beneath it.  Keep every
                // confirmed support cell so steering cannot cut off the side
                // of a narrow bridge between two plank centres.
                if (step.sourceSupport.kind == kSupportSpriteFloor
                    || step.targetSupport.kind == kSupportSpriteFloor)
                    break;
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

    bool buildNavRoute(int targetX, int targetY, int targetZ, int targetSector,
                       int signature, const SupportRef *requiredSupport = nullptr)
    {
        ensureNavTopology();
        const int start = currentNavStartCell();
        if (start < 0)
            return false;
        const int startArea = navCells[size_t(start)].walkArea;
        // Object/switch Z is aiming geometry, not necessarily the floor on
        // which the player must stand.  Prefer the target XY on the support
        // component reachable now; only use targetZ to disambiguate when the
        // destination is genuinely on another (conditional) component.
        const int areaTarget = requiredSupport ? -1
            : navCellInSectorArea(targetSector, targetX, targetY, startArea);
        const int strictTarget = navCellInSector(targetSector, targetX, targetY,
                                                 targetZ, requiredSupport);
        int target = areaTarget;
        if (areaTarget >= 0 && strictTarget >= 0)
        {
            // A reachable support at the requested XY is the right stance
            // for a switch whose sprite Z is merely its aim point.  It is not
            // a valid substitute when it is on the far side of the sector:
            // AGTST8's first sprite bridge deliberately puts the doorway on a
            // different support component, and the old lookup silently chose
            // a same-area cell more than 6000 units away.  Retain the stance
            // tolerance, but preserve the exact support endpoint outside it
            // so planNavRoute can use the bridge's conditional jump edge.
            const int kTargetStanceSpan = 4 << kNavGridShift;
            const NavCell &areaCell = navCells[size_t(areaTarget)];
            if (distance2(areaCell.center.x, areaCell.center.y, targetX, targetY)
                > kTargetStanceSpan * kTargetStanceSpan)
                target = strictTarget;
        }
        else if (target < 0)
            target = strictTarget;
        if (areaTarget >= 0 && strictTarget >= 0 && areaTarget != strictTarget
            && target != areaTarget)
        {
            char resolution[256];
            const NavCell &areaCell = navCells[size_t(areaTarget)];
            const NavCell &strictCell = navCells[size_t(strictTarget)];
            snprintf(resolution, sizeof(resolution),
                     "area_cell=%d at=(%d,%d,z%d,a%d) strict_cell=%d at=(%d,%d,z%d,a%d) chosen=%d want=(%d,%d,z%d)",
                     areaTarget, areaCell.center.x, areaCell.center.y,
                     areaCell.z, areaCell.walkArea,
                     strictTarget, strictCell.center.x, strictCell.center.y,
                     strictCell.z, strictCell.walkArea, target,
                     targetX, targetY, targetZ);
            event("nav_route_target_support_override", resolution);
        }
        if (target < 0)
        {
            // The mesh holds only sectors the bot has stood in, so a
            // frontier's far side has no cells of its own.  Aim at the
            // nearest cell the bot can actually walk to -- one in its own
            // walk area -- rather than the nearest cell anywhere, which is
            // routinely across a wall and turns a crossable boundary into an
            // unreachable one.
            target = nearestNavCellInArea(targetX, targetY,
                                          navCells[size_t(start)].walkArea,
                                          targetZ);
        }
        // NOTE: the last resort is still the loose lookup, which can land in
        // another sector entirely.  navCellInSector() above is the strict
        // answer and is correct in principle, but making it the only answer
        // costs AGTST4 two thirds of its exploration -- routes that
        // legitimately pass through thin geometry stop resolving.  Fixing
        // that properly means representing thin sectors as real transit
        // rather than as cells, which is not done yet.
        if (target < 0)
            target = nearestNavCell(targetSector, targetX, targetY, targetZ);
        std::vector<NavRouteStep> route;
        if (!llmapper::planNavRoute(navCells, start, target, targetX, targetY, targetSector, -1,
                                    navEdgeFailures, 0, route))
        {
            // This graph has already answered this exact question.  Keep the
            // negative result for the lifetime of the current navigation
            // state instead of running the same full search again from every
            // caller and every decision frame.  resetNavigation() clears the
            // verdict when the objective or its world evidence changes.
            navRouteRejectedSignature = signature;
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
        navRouteRejectedSignature = 0;
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

    // clipmove answers horizontal body clearance, not whether the floor
    // continues under the swept segment.  Sample the engine's standable
    // surfaces along a proposed direct line and carry forward only support
    // heights reachable by an ordinary step.  This accepts a long clear
    // room even if the coarse mesh happened to split it, while rejecting a
    // horizontally unobstructed line over a pit.  Sprite bridges naturally
    // work because their live floor surfaces participate in the same sweep.
    bool directLineKeepsSupport(int targetX, int targetY) const
    {
        if (!gMe || !gMe->pSprite || !inRange(observation.sector, 0, numsectors))
            return false;
        const int dx = targetX - observation.x;
        const int dy = targetY - observation.y;
        const int length = int(std::sqrt(double(int64_t(dx) * dx + int64_t(dy) * dy)));
        const int sampleSpan = std::max(64, playerClipRadius());
        const int samples = std::max(1, (length + sampleSpan - 1) / sampleSpan);
        int16_t sampleSector = int16_t(observation.sector);
        int supportZ = llmapper::capability::playerFloorZ();
        const int bodyAboveSupport = observation.z - supportZ;
        for (int step = 1; step <= samples; ++step)
        {
            const int x = observation.x + int(int64_t(dx) * step / samples);
            const int y = observation.y + int(int64_t(dy) * step / samples);
            updatesectorz(x, y, supportZ + bodyAboveSupport, &sampleSector);
            if (!inRange(int(sampleSector), 0, numsectors))
                return false;
            int ceilingZ = 0, ceilingHit = 0, floorZ = 0, floorHit = 0;
            GetZRangeAtXYZ(x, y, supportZ - 1, sampleSector,
                           &ceilingZ, &ceilingHit, &floorZ, &floorHit,
                           4, CLIPMASK0,
                           PARALLAXCLIP_CEILING | PARALLAXCLIP_FLOOR);
            if (std::abs(floorZ - supportZ) > kMaxWalkableStep)
                return false;
            supportZ = floorZ;
        }
        return true;
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
        const int body = playerStandingClearance();
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
        const bool wallPush = inRange(wallRecord.extra, 1, kMaxXWalls)
            && xwall[wallRecord.extra].triggerPush;
        int mechanismSector = -1;
        const int ownerSector = wallOwnerSector(wallIndex);
        if (inRange(ownerSector, 0, numsectors)
            && inRange(sector[ownerSector].extra, 1, kMaxXSectors)
            && xsector[sector[ownerSector].extra].Wallpush)
            mechanismSector = ownerSector;
        else if (inRange(wallRecord.nextsector, 0, numsectors)
                 && inRange(sector[wallRecord.nextsector].extra, 1, kMaxXSectors)
                 && xsector[sector[wallRecord.nextsector].extra].Wallpush)
            mechanismSector = wallRecord.nextsector;
        if (!wallPush && mechanismSector < 0)
            return false;

        // Observation may already have represented this exact physical wall
        // through its destination-sector mechanism.  A later collision is
        // stronger evidence that the affordance is reachable, but it is not a
        // second affordance.  Duplicating it lets the bot operate the same
        // reversible door twice under two IDs and undo its own progress.
        for (const auto &entry : interactions)
        {
            const InteractionMemory &memory = entry.second;
            if (memory.observed && memory.kind == kInteractionWall
                && memory.target.wall == wallIndex)
                return true;
        }

        const walltype &nextWall = wall[wallRecord.point2];
        InteractionCandidate candidate;
        candidate.kind = kInteractionWall;
        candidate.id = wallIndex;
        candidate.fromSector = observation.sector;
        candidate.targetSector = mechanismSector;
        candidate.x = (wallRecord.x + nextWall.x) / 2;
        candidate.y = (wallRecord.y + nextWall.y) / 2;
        candidate.z = observation.z;
        candidate.key = wallPush ? xwall[wallRecord.extra].key
            : xsector[sector[mechanismSector].extra].Key;
        candidate.locked = wallPush ? xwall[wallRecord.extra].locked != 0
            : xsector[sector[mechanismSector].extra].locked != 0;
        candidate.reversible = true;
        candidate.target.wall = wallIndex;
        candidate.target.from = observation.sector;
        candidate.target.to = mechanismSector;
        candidate.target.x = candidate.x;
        candidate.target.y = candidate.y;
        candidate.target.wallPush = wallPush;
        candidate.target.sectorPush = mechanismSector >= 0;
        candidate.target.interactionAffordance = true;
        candidate.target.mechanismTx = wallPush ? xwall[wallRecord.extra].txID
            : xsector[sector[mechanismSector].extra].txID;
        candidate.target.x1 = wallRecord.x;
        candidate.target.y1 = wallRecord.y;
        candidate.target.x2 = nextWall.x;
        candidate.target.y2 = nextWall.y;
        setInteractionGeometry(candidate.target);
        candidate.z = candidate.target.z;
        if (currentObjective.active
            && (currentObjective.type == kObjectiveFrontier
                || currentObjective.type == kObjectiveInvestigate)
            && currentObjective.wall >= 0)
        {
            candidate.causalWall = currentObjective.wall;
            candidate.causalFrom = currentObjective.sector;
            candidate.causalTo = currentObjective.targetSector;
        }
        else if (recentInvestigatedWall >= 0
                 && recentInvestigatedTick >= 0
                 && observation.tick - recentInvestigatedTick <= 2 * kTicsPerSec
                 && observation.sector == recentInvestigatedFrom)
        {
            candidate.causalWall = recentInvestigatedWall;
            candidate.causalFrom = recentInvestigatedFrom;
            candidate.causalTo = recentInvestigatedTo;
        }
        const int memoryKey = interactionKey(candidate);
        const auto known = interactions.find(memoryKey);
        const bool newlyDiscovered = known == interactions.end()
            || !known->second.observed;
        rememberInteraction(candidate);
        if (newlyDiscovered && candidate.causalWall >= 0
            && currentObjective.active
            && currentObjective.type == kObjectiveFrontier)
        {
            event("frontier_actionable_blocker_discovered",
                  "reason=engine_collision_identified_world_affordance");
            suppressObjective(objectiveKey(currentObjective),
                              "actionable_blocker_discovered");
            invalidateObjective("actionable_blocker_discovered", false);
        }
        return true;
    }

    GINPUT navigateTo(int x, int y, int z, int targetSector, int targetId = -1,
                      TraversalCapability capability = kTraversalUnknown, bool shoot = false,
                      const SupportRef *requiredSupport = nullptr)
    {
        navigationActionableBlocker = -1;
        int signature = navigationSignature(x, y, z, targetSector, targetId);
        if (requiredSupport)
            signature = llmapper::mixHash(signature,
                int(requiredSupport->kind) * 65536 + requiredSupport->id);
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
        if (requiredSupport && currentPlayerSupport() != *requiredSupport)
        {
            ensureNavTopology();
            const bool ownsRequiredRoute = !navRoute.empty()
                && navRouteSignature == signature
                && navRouteTopologyRevision == navTopologyRevision;
            if (!ownsRequiredRoute && navRouteRejectedSignature != signature)
                buildNavRoute(x, y, z, targetSector, signature, requiredSupport);
            direct.reachable = false;
        }
        bool supportCenterlineActive = currentPlayerSupport().kind == kSupportSpriteFloor;
        if (!supportCenterlineActive && !navRoute.empty()
            && navRouteSignature == signature
            && navRouteTopologyRevision == navTopologyRevision
            && navRouteIndex < navRoute.size())
        {
            const NavRouteStep &next = navRoute[navRouteIndex];
            supportCenterlineActive = next.sourceSupport.kind == kSupportSpriteFloor
                || next.targetSupport.kind == kSupportSpriteFloor;
        }

        // clipmove proves that the player's cylinder does not hit a wall; it
        // does not prove that a floor remains underneath the whole segment.
        // That distinction matters for pits, disconnected ledges, and sprite
        // bridges: a distant switch can be horizontally visible while the
        // straight line to it walks off an edge.  Keep direct steering when
        // a sampled support sweep proves the floor continuous; otherwise
        // make longer movement use the support-aware navigation graph even
        // when the collision-only probe succeeds.
        const int kLocalDirectSpan = 2 << kNavGridShift;
        if (direct.reachable && !supportCenterlineActive
            && distance2(observation.x, observation.y, x, y)
                > kLocalDirectSpan * kLocalDirectSpan
            && !directLineKeepsSupport(direct.x, direct.y))
        {
            ensureNavTopology();
            const bool ownsSupportRoute = !navRoute.empty()
                && navRouteSignature == signature
                && navRouteTopologyRevision == navTopologyRevision;
            if (ownsSupportRoute
                || (navRouteRejectedSignature != signature
                    && buildNavRoute(x, y, z, targetSector, signature,
                                     requiredSupport)))
            {
                direct.reachable = false;
                if (!ownsSupportRoute)
                    event("navigation_support_route_required",
                          "reason=distant_collision_probe_has_no_floor_guarantee");
            }
            else
            {
                // Do not fall back to the unsafe straight line merely because
                // no route is known yet.  The opportunity remains available
                // and will be reconsidered when topology changes.
                direct.reachable = false;
            }
        }
        if (direct.reachable && !supportCenterlineActive)
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
            const LocalWaypoint waypoint = navStepWaypoint(step);
            const int waypointX = waypoint.x;
            const int waypointY = waypoint.y;
            const bool withinWaypoint =
                distance2(observation.x, observation.y, waypointX, waypointY)
                    <= tolerance * tolerance;
            // Reaching the XY centre while sailing over a narrow obstacle is
            // not the same as landing on it.  If the jump step is meant to
            // establish sprite support, keep steering to the centre until the
            // engine reports that exact support under a grounded player.
            // Otherwise the route advances in mid-air and the next objective
            // immediately pulls Caleb off the far edge.
            const bool requiresSpriteLanding = step.mode == kNavJump
                && step.targetSupport.kind == kSupportSpriteFloor;
            const bool touchingTargetSupport = requiresSpriteLanding
                && currentPlayerSupport() == step.targetSupport;
            const bool landedOnTargetSupport = !requiresSpriteLanding
                || (touchingTargetSupport && observation.playerZVelocity == 0
                    && gMe && gMe->pXSprite && gMe->pXSprite->height == 0
                    && coastDistance() <= playerClipRadius());
            const bool finalSupportPending = requiredSupport
                && navRouteIndex + 1 == navRoute.size()
                && currentPlayerSupport() != *requiredSupport;
            // A rising player can be snapped to a solid top for one frame
            // while retaining upward velocity.  Do not turn that transient
            // contact into the next route leg; neutral input lets the same
            // named support catch the descending body.
            if (requiresSpriteLanding && touchingTargetSupport
                && !landedOnTargetSupport)
                return brakeAfterLanding();
            const bool waypointReached = (requiresSpriteLanding
                ? landedOnTargetSupport : withinWaypoint) && !finalSupportPending;
            if (waypointReached)
            {
                if (requiresSpriteLanding)
                {
                    char landed[96];
                    snprintf(landed, sizeof(landed), "sprite=%d sector=%d",
                             step.targetSupport.id, observation.sector);
                    event("nav_sprite_support_landed", landed);
                }
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
                const LocalWaypoint nextWaypoint = navStepWaypoint(next);
                const int nextX = nextWaypoint.x;
                const int nextY = nextWaypoint.y;
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
                    // The route's current physical edge owns locomotion.  A
                    // jump required at the final portal must not leak backward
                    // into ordinary WALK waypoints leading to its takeoff.
                    TraversalCapability cap = kTraversalUnknown;
                    if (traversal == kTraverseJump || next.mode == kNavJump)
                        cap = kTraversalJumpable;
                    else if (next.mode == kNavCrouch)
                        cap = kTraversalCrouchable;
                    return steerTo(nextX, nextY, false, shoot, targetId,
                                   next.targetSector >= 0 ? next.targetSector : observation.sector,
                                   cap);
                }
                if (finalSupportPending)
                {
                    // The graph endpoint claimed a support which the engine
                    // never established at that pose.  Do not rebuild the
                    // same nominal one-step route forever: reject this exact
                    // target/support hypothesis and let the opportunity
                    // budget preserve other exploration choices.
                    navRouteRejectedSignature = signature;
                    navRoute.clear();
                    event("nav_target_support_unreached",
                          "reason=engine_support_disagrees_with_route_endpoint");
                    return GINPUT{};
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
                if (buildNavRoute(x, y, z, targetSector, signature, requiredSupport))
                {
                    const NavRouteStep &retry = navRoute.front();
                    const LocalWaypoint retryWaypoint = navStepWaypoint(retry);
                    const int retryX = retryWaypoint.x;
                    const int retryY = retryWaypoint.y;
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
                && buildNavRoute(x, y, z, targetSector, signature, requiredSupport))
            {
                const NavRouteStep &waypoint = navRoute.front();
                const LocalWaypoint selectedWaypoint = navStepWaypoint(waypoint);
                const int waypointX = selectedWaypoint.x;
                const int waypointY = selectedWaypoint.y;
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
                && buildNavRoute(x, y, z, targetSector, signature, requiredSupport))
            {
                const NavRouteStep &waypoint = navRoute.front();
                const LocalWaypoint selectedWaypoint = navStepWaypoint(waypoint);
                const int waypointX = selectedWaypoint.x;
                const int waypointY = selectedWaypoint.y;
                navWaypointBestDistance2 = distance2(observation.x, observation.y,
                                                     waypointX, waypointY);
                navWaypointProgressTick = observation.tick;
                return steerTo(waypointX, waypointY, false, false, targetId,
                               waypoint.targetSector >= 0 ? waypoint.targetSector
                                                          : observation.sector,
                               capability);
            }
        }
        int failureSignature = signature;
        if (currentObjective.active
            && currentObjective.type == kObjectiveInteraction)
        {
            // A moving actuator changes its world-space approach point every
            // frame, but that is not evidence that a stationary player has
            // made progress toward it.  Own repeated failure by the stable
            // work identity, the player's coarse physical pose, and the
            // topology that answered the route query.  Actual movement or a
            // rebuilt topology still reopens the question.
            failureSignature = llmapper::mixHash(
                objectiveKey(currentObjective), observation.sector);
            failureSignature = llmapper::mixHash(
                failureSignature, observation.x >> kNavGridShift);
            failureSignature = llmapper::mixHash(
                failureSignature, observation.y >> kNavGridShift);
            failureSignature = llmapper::mixHash(
                failureSignature, navTopologyRevision);
        }
        else if (!currentObjective.active
                 && currentGoal == "CLEAR_MOVING_SECTOR")
        {
            // Emergency egress is also stable work even when the portal
            // itself is moving.  Its changing midpoint must not erase the
            // fact that the player is stationary and this escape hypothesis
            // has no route from the current topology.
            failureSignature = llmapper::mixHash(
                currentGoalTarget, observation.sector);
            failureSignature = llmapper::mixHash(
                failureSignature, observation.x >> kNavGridShift);
            failureSignature = llmapper::mixHash(
                failureSignature, observation.y >> kNavGridShift);
            failureSignature = llmapper::mixHash(
                failureSignature, navTopologyRevision);
        }
        if (navigationFailureSignature != failureSignature)
        {
            navigationFailureSignature = failureSignature;
            navigationFailureCount = 0;
        }
        ++navigationFailureCount;
        if (navigationFailureCount == 1)
            event("navigation_failed", "reason=no_bounded_collision_safe_detour");
        if (navigationFailureCount >= kNavigationFailureLimit
            && !currentObjective.active
            && currentGoal == "CLEAR_MOVING_SECTOR"
            && clearingWall >= 0)
        {
            const Portal *failedEscape = portalByWall(
                clearingWall, observation.sector, -1);
            if (failedEscape
                && recordEdgeFailure(*failedEscape, "navigation_escape_failed",
                                     "no_bounded_route_to_exit"))
            {
                event("moving_sector_escape_reconsidered",
                      "reason=selected_exit_route_unavailable");
                clearingWall = -1;
                resetNavigation();
                currentGoal.clear();
                currentGoalTarget = -1;
            }
        }
        else if (navigationFailureCount >= kNavigationFailureLimit
                 && currentObjective.active
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

    // Cross a gap: stand at the lip, face the far side, run and jump.
    //
    // This cannot go through the ordinary jump traversal, which walks
    // forward until it is within a fixed distance of its target before
    // launching.  Across a hole that distance is over open air, so the bot
    // walks off the edge and falls in.  The takeoff point is the near lip,
    // wherever the landing happens to be.
    // Horizontal speed the player is carrying right now.
    int groundSpeed() const
    {
        if (!gMe || !gMe->pSprite)
            return 0;
        const int index = gMe->pSprite->index;
        const int vx = xvel[index] >> 12;
        const int vy = yvel[index] >> 12;
        return int(std::sqrt(double(int64_t(vx) * vx + int64_t(vy) * vy)));
    }

    // A landing keeps the speed the jump was made at.  On a pillar a
    // thousand units across that carries the bot over the far edge in three
    // decisions -- it lands, coasts, and falls off the other side while
    // lining up the next jump.  Coasting is not enough: put the ground to
    // work by pushing back against the drift.
    // Which way the player is actually travelling, and how fast.
    bool driftHeading(int &outAngle, int &outSpeed) const
    {
        if (!gMe || !gMe->pSprite)
            return false;
        const int index = gMe->pSprite->index;
        const int vx = xvel[index] >> 12;
        const int vy = yvel[index] >> 12;
        outSpeed = int(std::sqrt(double(int64_t(vx) * vx + int64_t(vy) * vy)));
        if (outSpeed <= 0)
            return false;
        outAngle = getangle(vx, vy);
        return true;
    }

    // The player's live motion state, ready to be stepped forward.
    llmapper::capability::MotionState currentMotion() const
    {
        llmapper::capability::MotionState state;
        if (!gMe || !gMe->pSprite)
            return state;
        const int index = gMe->pSprite->index;
        state.x = gMe->pSprite->x;
        state.y = gMe->pSprite->y;
        state.z = gMe->pSprite->z;
        state.xvel = xvel[index];
        state.yvel = yvel[index];
        state.zvel = zvel[index];
        return state;
    }

    // Where an arc ends, given the vertical velocity it starts with and the
    // input held through it.  Run with the engine's own arithmetic, so this
    // is a prediction rather than an estimate: the only thing it leaves out
    // is clipping, and a jump over a hole has nothing to clip against.
    bool predictArc(int startZVelocity, int landingFloorZ, int forwardInput,
                    int angle, int settleFrames, int &outX, int &outY,
                    int &outFrames) const
    {
        if (!gMe || !gMe->pSprite)
            return false;
        const POSTURE &stand = gMe->pPosture[gMe->lifeMode][kPostureStand];
        const int foot = llmapper::capability::playerFootOffset();
        const llmapper::capability::AirDrag air = llmapper::capability::playerAirDrag();
        llmapper::capability::MotionState state = currentMotion();
        state.zvel = startZVelocity;
        for (int frame = 1; frame <= 240; ++frame)
        {
            // Whether the body was on its way down going into this frame.
            // The step itself sets the floor down and takes the velocity out,
            // exactly as MoveDude does, so asking afterwards always says no.
            const bool falling = state.zvel > 0;
            llmapper::capability::stepPlayerMotion(state, forwardInput, angle,
                                                   landingFloorZ, stand.frontAccel,
                                                   foot, air);
            if (frame > settleFrames && falling && state.z + foot >= landingFloorZ)
            {
                outX = state.x;
                outY = state.y;
                outFrames = frame;
                return true;
            }
        }
        return false;
    }

    // Where a jump taken now would put the bot.  The first frames are
    // excused because the launch happens from the floor.
    bool predictLanding(int landingFloorZ, int forwardInput, int angle,
                        int &outX, int &outY, int &outFrames) const
    {
        return predictArc(llmapper::capability::playerJumpImpulse(), landingFloorZ,
                          forwardInput, angle, 2, outX, outY, outFrames);
    }

    // Same question for a jump already under way: keep the current vertical
    // velocity rather than starting a fresh one.
    bool predictRemainingFlight(int landingFloorZ, int forwardInput, int angle,
                                int &outX, int &outY) const
    {
        int frames = 0;
        return predictArc(gMe && gMe->pSprite ? zvel[gMe->pSprite->index] : 0,
                          landingFloorZ, forwardInput, angle, 0, outX, outY, frames);
    }

    // Predict a support-to-support jump without pretending the destination
    // plane covers the entire map.  A sprite top only becomes the player's
    // floor once the horizontal footprint reaches it; applying that floor at
    // takeoff snaps the simulated body upward immediately.  Keep the source
    // floor for the first movement frame (and for a higher destination), then
    // report the descending point at which the body can settle on the target.
    bool predictSupportArc(int startZVelocity, int sourceZ, int targetZ,
                           int forwardInput, int angle, int &outX, int &outY,
                           int &outFrames) const
    {
        if (!gMe || !gMe->pSprite)
            return false;
        const POSTURE &stand = gMe->pPosture[gMe->lifeMode][kPostureStand];
        const int foot = llmapper::capability::playerFootOffset();
        const llmapper::capability::AirDrag air = llmapper::capability::playerAirDrag();
        llmapper::capability::MotionState state = currentMotion();
        state.zvel = startZVelocity;
        bool centreClearedTarget = state.z < targetZ;
        for (int frame = 1; frame <= 240; ++frame)
        {
            const bool falling = state.zvel > 0;
            const int simulationFloor = frame == 1 ? sourceZ
                                                    : std::max(sourceZ, targetZ);
            llmapper::capability::stepPlayerMotion(
                state, forwardInput, angle, simulationFloor,
                stand.frontAccel, foot, air);
            if (state.z < targetZ)
                centreClearedTarget = true;
            if (falling && centreClearedTarget && state.z + foot >= targetZ)
            {
                outX = state.x;
                outY = state.y;
                outFrames = frame;
                return true;
            }
        }
        return false;
    }

    bool solveSupportArc(int startZVelocity, const NavRouteStep &step,
                         int targetX, int targetY, int angle,
                         int &outInput) const
    {
        const int samples = 16;
        bool found = false;
        int64_t bestMiss = INT64_MAX;
        for (int sample = 0; sample <= samples; ++sample)
        {
            // Reverse input is useful while trimming an already fast arc and
            // harmless to consider at launch; the solver will choose it only
            // when the predicted landing is closer.
            const int forward = -kFullThrottle
                + (2 * kFullThrottle * sample) / samples;
            int landX = 0, landY = 0, frames = 0;
            if (!predictSupportArc(startZVelocity, step.sourceZ, step.targetZ,
                                   forward, angle, landX, landY, frames))
                continue;
            const int64_t miss = distance2(landX, landY, targetX, targetY);
            if (miss >= bestMiss)
                continue;
            found = true;
            bestMiss = miss;
            outInput = forward;
        }
        return found;
    }

    // Which forward input, held for the rest of the flight, puts the bot on
    // the far ledge?
    //
    // Blood scales a dude's input authority and its drag by how far it is
    // off the floor, and both only cut out above a height a jump never
    // reaches, so about three quarters of the ground's control is still
    // there at the apex.  A jump is therefore steerable after it has been
    // taken, and the question at the lip is not whether full throttle
    // happens to land right -- over a pillar a thousand units deep it sails
    // clean over -- but whether any input lands right.  Answered by running
    // the engine's own arithmetic across the range of inputs and keeping the
    // one that comes down nearest the middle of the ledge.  The launch, the
    // mid-air trim and the decision to carry momentum all ask this same
    // question, so there is one answer to it.
    bool solveGapArc(int startZVelocity, const Portal &crossing, int angle,
                     int settleFrames, int &outInput, int &outX, int &outY,
                     int &outFrames) const
    {
        const int steps = 8;
        bool found = false;
        int64_t bestMiss = 0;
        for (int step = 0; step <= steps; ++step)
        {
            const int forward = kFullThrottle * step / steps;
            int x = 0;
            int y = 0;
            int frames = 0;
            if (!predictArc(startZVelocity, crossing.floorZ, forward, angle,
                            settleFrames, x, y, frames))
                continue;
            if (inside(x, y, crossing.to) != 1)
                continue;
            const int64_t miss = distance2(x, y, crossing.x, crossing.y);
            if (found && miss >= bestMiss)
                continue;
            found = true;
            bestMiss = miss;
            outInput = forward;
            outX = x;
            outY = y;
            outFrames = frames;
        }
        return found;
    }

    // Is the bot already travelling at the thing it is about to jump to?
    bool momentumServesTheNextJump()
    {
        if (!currentObjective.active || currentObjective.type != kObjectiveFrontier)
            return false;
        const Portal *crossing = portalByWall(currentObjective.wall,
                                              currentObjective.sector,
                                              currentObjective.targetSector);
        if (!crossing || !crossing->gapJump || crossing->from != observation.sector)
            return false;
        // Momentum off a landing is an asset exactly when it would carry
        // the bot to the next ledge, and a liability otherwise.  That is the
        // same question the launch asks, so there is no separate rule for
        // chaining: if the arc reaches from here, take it.
        const int wanted = getangle(crossing->x - observation.x,
                                    crossing->y - observation.y);
        int landX = 0;
        int landY = 0;
        int ticks = 0;
        int hold = 0;
        if (!solveGapArc(llmapper::capability::playerJumpImpulse(), *crossing, wanted,
                         2, hold, landX, landY, ticks))
            return false;
        gapRunning = true;
        gapRunWall = crossing->wall;
        char detail[208];
        snprintf(detail, sizeof(detail),
                 "wall=%d to=%d speed=%d predicted=(%d,%d) reason=arc_already_reaches",
                 crossing->wall, crossing->to, groundSpeed(), landX, landY);
        event("gap_jump_chained", detail);
        return true;
    }

    // How much further the bot slides if it stops pushing now, under the
    // engine's own drag.
    int coastDistance() const
    {
        return llmapper::capability::playerCoastDistance(groundSpeed());
    }

    GINPUT brakeAfterLanding()
    {
        GINPUT input = {};
        if (coastDistance() <= playerClipRadius())
        {
            gapSettleUntilTick = -1;
            return input;
        }
        setGoal("SHED_LANDING_SPEED", -1);
        // Facing is still roughly along the flight, so reverse thrust on the
        // forward axis takes the drift out.  Do not turn: a turn mid-slide
        // just converts the drift into a different direction.
        input.forward = -kFullThrottle;
        return input;
    }

    GINPUT steerGapJump(const Portal &portal)
    {
        setGoal("JUMP_THE_GAP", portal.wall);
        const int radius = playerClipRadius();

        // Back away from the lip along the line of the jump, far enough to
        // be at running speed by the time the ground runs out.  A jump from
        // a standing start carries almost no horizontal velocity, which is
        // how the first attempts at this landed in the hole.
        const int64_t backX = int64_t(portal.takeoffX) - portal.x;
        const int64_t backY = int64_t(portal.takeoffY) - portal.y;
        const double length = std::sqrt(double(backX * backX + backY * backY));
        int markX = portal.takeoffX;
        int markY = portal.takeoffY;
        if (length > 1.0)
        {
            const int runUp = radius + llmapper::capability::playerRunUpDistance();
            for (int back = runUp; back >= radius + 96; back -= 256)
            {
                const int px = portal.takeoffX + int(backX * back / length);
                const int py = portal.takeoffY + int(backY * back / length);
                if (inside(px, py, portal.from) == 1)
                {
                    markX = px;
                    markY = py;
                    break;
                }
            }
        }

        if (gapRunWall != portal.wall)
        {
            gapRunWall = portal.wall;
            gapRunning = false;
        }
        const int toLanding = getangle(portal.x - observation.x, portal.y - observation.y);
        const int heading = angleDelta(toLanding, observation.angle);
        const int arrive = radius + 192;
        if (!gapRunning
            && distance2(observation.x, observation.y, markX, markY) > arrive * arrive)
        {
            noteCameraOwner("NAVIGATION", portal.wall, toLanding, 0);
            return navigateTo(markX, markY, observation.z, portal.from,
                              portal.wall, kTraversalWalkable);
        }

        GINPUT input = {};
        input.syncFlags.run = 1;
        input.q16turn = fix16_from_int(heading);
        noteCameraOwner("JUMP_THE_GAP", portal.wall, toLanding, 0);
        if (!gapRunning)
        {
            // Arriving at the run-up mark from an arbitrary previous route can
            // leave substantial lateral velocity.  Facing the gap does not
            // remove that drift, and a launch predictor/executor pair cannot
            // make a repeatable crossing while the approach starts sideways.
            // Let engine drag settle first, then build a fresh straight run.
            if (coastDistance() > radius)
                return input;
            // Line up first: a jump taken off-heading is wasted, and there
            // is no correcting it once the ground has gone.
            if (std::abs(heading) >= 48)
                return input;
            gapRunning = true;
            char detail[192];
            snprintf(detail, sizeof(detail),
                     "wall=%d from=%d to=%d run_up_from=(%d,%d) lip=(%d,%d)",
                     portal.wall, portal.from, portal.to, markX, markY,
                     portal.takeoffX, portal.takeoffY);
            event("gap_jump_run_up", detail);
        }
        input.forward = kFullThrottle;
        if (gMe->cantJump || (gMe->pXSprite && gMe->pXSprite->height != 0))
            return input;
        // Leave the ground when the simulated arc actually ends on the far
        // ledge.  A decision covers four ticks and the bot crosses several
        // hundred units in that time, so a launch line drawn at the lip is
        // missed as often as it is hit -- and missing it means the ground
        // has already run out.  Asking where the jump would land removes the
        // guesswork from both the distance and the speed.
        int landX = 0;
        int landY = 0;
        int ticks = 0;
        int hold = kFullThrottle;
        const bool reaches = solveGapArc(llmapper::capability::playerJumpImpulse(),
                                         portal, toLanding, 2, hold, landX, landY, ticks);
        // Out of ground once one more frame of travel would carry the body
        // over the lip.  A fixed line at the lip is sampled at running speed
        // and stepped over as often as it is landed on, and stepping over it
        // is the fall.
        const int lip = radius + 96 + coastDistance();
        const bool outOfGround =
            distance2(observation.x, observation.y, portal.takeoffX, portal.takeoffY)
                <= lip * lip;
        if (!reaches)
        {
            if (!outOfGround)
                return input;   // still ground to build speed over
            // Out of run-up and the arc still misses.  Shed the speed and
            // set the approach up again rather than stepping into the hole.
            gapRunning = false;
            gapSettleUntilTick = observation.tick + 2 * kTicsPerSec;
            int coastX = 0, coastY = 0, coastFrames = 0;
            int pushX = 0, pushY = 0, pushFrames = 0;
            const int impulse = llmapper::capability::playerJumpImpulse();
            predictArc(impulse, portal.floorZ, 0, toLanding, 2, coastX, coastY, coastFrames);
            predictArc(impulse, portal.floorZ, kFullThrottle, toLanding, 2,
                       pushX, pushY, pushFrames);
            char detail[320];
            snprintf(detail, sizeof(detail),
                     "wall=%d to=%d from=(%d,%d) aim=(%d,%d) coast=(%d,%d) push=(%d,%d)"
                     " coast_in=%d push_in=%d speed=%d reason=arc_misses_the_landing",
                     portal.wall, portal.to, observation.x, observation.y,
                     portal.x, portal.y, coastX, coastY, pushX, pushY,
                     inside(coastX, coastY, portal.to), inside(pushX, pushY, portal.to),
                     groundSpeed());
            event("gap_jump_aborted", detail);
            return GINPUT{};
        }
        input.forward = hold;
        input.buttonFlags.jump = 1;
        gapRunning = false;
        gapJumpActive = true;
        gapJumpWall = portal.wall;
        gapJumpFrom = portal.from;
        gapJumpTo = portal.to;
        gapJumpX = portal.x;
        gapJumpY = portal.y;
        gapJumpFloorZ = portal.floorZ;
        gapJumpTick = observation.tick;
        char detail[224];
        snprintf(detail, sizeof(detail),
                 "wall=%d from=%d to=%d takeoff=(%d,%d) aim=(%d,%d) predicted=(%d,%d) ticks=%d speed=%d",
                 portal.wall, portal.from, portal.to, observation.x, observation.y,
                 portal.x, portal.y, landX, landY, ticks, groundSpeed());
        event("gap_jump_launched", detail);
        return input;
    }

    // A launched jump is committed.  The bot is in the air over a sector it
    // must not stop in, and re-deciding mid-flight is exactly how it ends up
    // in the hole.
    GINPUT continueGapJump()
    {
        GINPUT input = {};
        input.syncFlags.run = 1;
        const bool airborne = observation.playerZVelocity != 0;
        // Where the body actually is, not only which sector the engine has
        // it filed under.  Coming down on the lip of a ledge, the sector the
        // player is registered in lags the position by a frame, and reading
        // only that made the bot report a crossing it had just made as a
        // fall into the hole it had cleared.
        const bool arrived = observation.sector == gapJumpTo
            || inside(observation.x, observation.y, gapJumpTo) == 1;
        const bool spent = gapJumpTick >= 0
            && observation.tick - gapJumpTick > 4 * kTicsPerSec;
        // A jump that has touched down is not necessarily over.  The frame
        // the body meets the floor it can still be on the lip, moving, and
        // filed under the sector it flew across; calling the outcome there
        // reported a crossing the bot had just made as a fall.  Give a
        // landing that has not arrived a moment to slide onto the ledge
        // before deciding it fell short.
        const bool down = !airborne && observation.tick - gapJumpTick > kTicsPerSec / 2;
        if (down && !arrived && ++gapJumpSettling <= kGapLandingSettleDecisions)
            return GINPUT{};
        if (arrived || spent || down)
        {
            char detail[192];
            snprintf(detail, sizeof(detail), "wall=%d to=%d landed_in=%d result=%s",
                     gapJumpWall, gapJumpTo, observation.sector,
                     arrived ? "landed" : spent ? "timed_out" : "short");
            event("gap_jump_finished", detail);
            gapJumpSettling = 0;
            gapJumpActive = false;
            gapRunning = false;
            gapRunWall = -1;
            gapSettleUntilTick = observation.tick + 2 * kTicsPerSec;
            return GINPUT{};
        }
        const int toLanding = getangle(gapJumpX - observation.x, gapJumpY - observation.y);
        input.q16turn = fix16_from_int(angleDelta(toLanding, observation.angle));
        noteCameraOwner("JUMP_THE_GAP", gapJumpWall, toLanding, 0);
        // Trim the arc while it is still in the air, asking the same question
        // the launch asked but from where the bot now is.  If some input
        // still lands it on the ledge, hold that one; if none does the arc is
        // already lost and full throttle is the best remaining chance.
        Portal landing;
        landing.to = gapJumpTo;
        landing.x = gapJumpX;
        landing.y = gapJumpY;
        landing.floorZ = gapJumpFloorZ;
        int hold = kFullThrottle;
        int landX = 0;
        int landY = 0;
        int frames = 0;
        const int zv = gMe && gMe->pSprite ? zvel[gMe->pSprite->index] : 0;
        if (!solveGapArc(zv, landing, toLanding, 0, hold, landX, landY, frames))
            hold = kFullThrottle;
        input.forward = hold;
        return input;
    }

    // A portal's floor delta is measured from the sector floor.  That is not
    // necessarily Caleb's take-off support: a chain of solid objects and
    // sprite bridges can establish a much higher, reversible approach.  Find
    // a reachable support near the boundary whose final transition is within
    // the ordinary jump envelope.  The support graph supplies the route; no
    // sprite type or map identity participates in the decision.
    int raisedPortalSupportPose(const Portal &portal)
    {
        if (portal.floorDelta >= -kMaxWalkableStep
            || !inRange(portal.from, 0, numsectors)
            || !inRange(portal.to, 0, numsectors))
            return -1;
        ensureNavTopology();
        const int start = currentNavStartCell();
        if (start < 0)
            return -1;
        std::vector<char> reachable(navCells.size(), 0);
        std::deque<int> reachQueue;
        reachable[size_t(start)] = 1;
        reachQueue.push_back(start);
        while (!reachQueue.empty())
        {
            const int current = reachQueue.front();
            reachQueue.pop_front();
            for (const NavLink &link : navCells[size_t(current)].links)
            {
                if (!llmapper::traversableMode(link.mode)
                    || !inRange(link.target, 0, int(navCells.size()))
                    || reachable[size_t(link.target)])
                    continue;
                reachable[size_t(link.target)] = 1;
                reachQueue.push_back(link.target);
            }
        }
        const int tolerance = playerClipRadius() + (1 << (kNavGridShift - 1));
        std::map<int, int> nearestBySupport;
        for (const NavCell &cell : navCells)
        {
            if (cell.sector != portal.from
                || cell.support.kind != kSupportSpriteFloor
                // standingFloorZ() has already resolved the portal's actual
                // take-off layer.  Match that layer instead of comparing it
                // with the enclosing sector floor; on a bridge doorway the
                // former is the sprite top by definition.
                || std::abs(cell.z - portal.fromFloorZ) > kMaxWalkableStep)
                continue;
            const int64_t boundaryDistance = dist2ToSegment(
                cell.center.x, cell.center.y, portal.x1, portal.y1,
                portal.x2, portal.y2);
            auto known = nearestBySupport.find(cell.support.id);
            if (known == nearestBySupport.end()
                || boundaryDistance < dist2ToSegment(
                    navCells[size_t(known->second)].center.x,
                    navCells[size_t(known->second)].center.y,
                    portal.x1, portal.y1, portal.x2, portal.y2))
                nearestBySupport[cell.support.id] = cell.id;
        }

        const bool report = reportedRaisedPortalAudits.insert(portal.wall).second;
        if (report)
        {
            char summary[192];
            snprintf(summary, sizeof(summary),
                     "wall=%d from=%d to=%d floor_delta=%d from_floor=%d to_floor=%d start=%d sprite_supports=%u",
                     portal.wall, portal.from, portal.to, portal.floorDelta,
                     portal.fromFloorZ, portal.toFloorZ, start,
                     unsigned(nearestBySupport.size()));
            event("raised_portal_support_audit", summary);
        }
        int best = -1;
        int64_t bestScore = INT64_MAX;
        for (const auto &entry : nearestBySupport)
        {
            const NavCell &cell = navCells[size_t(entry.second)];
            const int landingDelta = portal.toFloorZ - cell.z;
            const bool reversible = std::abs(landingDelta) <= playerReversibleDrop();
            const int reach = standingJumpReachForDelta(landingDelta);
            const int64_t boundaryDistance = dist2ToSegment(
                cell.center.x, cell.center.y, portal.x1, portal.y1,
                portal.x2, portal.y2);
            std::vector<NavRouteStep> route;
            const bool routeFound = llmapper::planNavRoute(navCells, start, cell.id,
                                        cell.center.x, cell.center.y,
                                        cell.sector, -1, navEdgeFailures, 0,
                                        route);
            if (report)
            {
                int incoming = 0;
                int reachableIncoming = 0;
                for (const NavCell &source : navCells)
                    for (const NavLink &link : source.links)
                        if (link.target == cell.id
                            && llmapper::traversableMode(link.mode))
                        {
                            ++incoming;
                            if (reachable[size_t(source.id)])
                                ++reachableIncoming;
                        }
                char detail[256];
                snprintf(detail, sizeof(detail),
                         "wall=%d support=%d cell=%d area=%d z=%d landing_delta=%d reversible=%d boundary_distance=%d reach=%d route=%d steps=%u incoming=%d reachable_incoming=%d start_area=%d",
                         portal.wall, cell.support.id, cell.id, cell.walkArea,
                         cell.z,
                         landingDelta, reversible ? 1 : 0,
                         int(std::sqrt(double(boundaryDistance))), reach,
                         routeFound ? 1 : 0, unsigned(route.size()), incoming,
                         reachableIncoming, navCells[size_t(start)].walkArea);
                event("raised_portal_support_candidate", detail);
            }
            if (!reversible || reach <= 0
                || boundaryDistance > int64_t(reach + tolerance)
                                       * (reach + tolerance)
                || !routeFound)
                continue;
            const int64_t score = int64_t(std::abs(landingDelta)) * 4096
                + boundaryDistance + int64_t(route.size()) * 1024;
            if (score < bestScore)
            {
                bestScore = score;
                best = cell.id;
            }
        }
        return best;
    }

    GINPUT steerPortal(const Portal &knownPortal)
    {
        Portal portal = knownPortal;
        const int supportPose = raisedPortalSupportPose(portal);
        if (supportPose >= 0)
        {
            const NavCell &pose = navCells[size_t(supportPose)];
            const SupportRef standing = currentPlayerSupport();
            if (standing != pose.support
                || std::abs(llmapper::capability::playerFloorZ() - pose.z)
                    > kMaxWalkableStep)
            {
                const int signature = navigationSignature(
                    pose.center.x, pose.center.y, pose.z, pose.sector,
                    6000000 + portal.wall);
                if (navRouteSignature != signature || navRoute.empty()
                    || navRouteTopologyRevision != navTopologyRevision)
                    buildNavRoute(pose.center.x, pose.center.y, pose.z,
                                  pose.sector, signature, &pose.support);
                if (!navRoute.empty())
                {
                    const int64_t routeKey = (int64_t(portal.wall) << 32)
                        ^ uint32_t(pose.support.id);
                    if (reportedRaisedPortalRoutes.insert(routeKey).second)
                    {
                        char detail[224];
                        snprintf(detail, sizeof(detail),
                                 "wall=%d from=%d to=%d support_kind=%d support=%d support_z=%d sector_floor_z=%d landing_z=%d",
                                 portal.wall, portal.from, portal.to,
                                 int(pose.support.kind), pose.support.id,
                                 pose.z, portal.fromFloorZ, portal.toFloorZ);
                        event("raised_portal_support_route", detail);
                    }
                    return navigateTo(pose.center.x, pose.center.y, pose.z,
                                      pose.sector, 6000000 + portal.wall,
                                      kTraversalUnknown);
                }
            }
            else
            {
                // Reclassify only from the support actually under Caleb.
                // The original sector-floor geometry remains unchanged and
                // will be reconsidered if he falls off.
                portal.fromFloorZ = pose.z;
                portal.floorDelta = portal.toFloorZ - pose.z;
                portal.jumpable = portal.floorDelta < -kMaxWalkableStep;
                portal.dropSafe = portal.floorDelta > kMaxWalkableStep;
                portal.walkable = std::abs(portal.floorDelta) <= kMaxWalkableStep;
                portal.traversable = true;
                portal.capability = portal.jumpable ? kTraversalJumpable
                    : portal.dropSafe ? kTraversalDropSafe
                                      : kTraversalWalkable;
            }
        }
        if (portal.gapJump)
            return steerGapJump(portal);
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
                    buildNavRoute(portal.x, portal.y, portal.z,
                                  observation.sector, routeSignature);
                if (!navRoute.empty() && navRouteSignature == routeSignature)
                {
                    GINPUT routeInput = navigateTo(portal.x, portal.y, portal.z,
                                                   observation.sector, portal.wall,
                                                   portal.capability);
                    // A route owns steering until it explicitly invalidates
                    // itself. Its jump state machine deliberately emits a
                    // neutral frame when ALIGN becomes TAKEOFF; interpreting
                    // that as rejection lets the portal fallback choose a
                    // different target in the same tick, producing a stable
                    // left/right camera oscillation with no movement.
                    if ((!navRoute.empty()
                         && navRouteSignature == routeSignature)
                        || routeInput.forward || routeInput.strafe
                        || routeInput.q16turn
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
                    hold.forward = kFullThrottle;
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
                if (!localEdgeFailed(edgeId)
                    && recordEdgeFailure(portal, "local_portal_failed",
                                         "bounded_collision_safe_approaches_exhausted")
                    && currentObjective.active)
                {
                    // The physical route has exhausted its bounded poses.
                    // Keeping the objective active only calls this same
                    // settled failure again next frame; make the work dormant
                    // while preserving it for changed geometry or backoff.
                    invalidateObjective("local_portal_failed");
                    return GINPUT{};
                }
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

        const NavRouteStep *supportStep = nullptr;
        if (!navRoute.empty() && navRouteIndex < navRoute.size())
        {
            const NavRouteStep &candidate = navRoute[navRouteIndex];
            const LocalWaypoint waypoint = navStepWaypoint(candidate);
            if (candidate.mode == kNavJump
                && candidate.targetSupport.kind == kSupportSpriteFloor
                && waypoint.x == x && waypoint.y == y)
                supportStep = &candidate;
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
            input.forward = 0;
            if (std::abs(angleError) < 96 && !gMe->cantJump
                && observation.playerZVelocity == 0
                && (!gMe->pXSprite || gMe->pXSprite->height == 0))
            {
                input.forward = kFullThrottle;
                if (supportStep)
                {
                    int hold = input.forward;
                    if (solveSupportArc(llmapper::capability::playerJumpImpulse(),
                                        *supportStep, x, y, targetAngle, hold))
                        input.forward = int16_t(hold);
                }
                input.buttonFlags.jump = 1;
                ++jumpAttempts;
                if (targetId == -2)
                    localDynamicJumpAttempted = true;
                jumpState = kJumpAirborne;
                jumpStateTick = observation.tick;
                char detail[320];
                snprintf(detail, sizeof(detail),
                         "phase=TAKEOFF jump_requested=1 jump_input_emitted=1 target=%d target_sector=%d player_z=%d zvel_before=%d posture=%d cant_jump=%d attempts=%d yaw=%d forward=%d",
                         targetId, targetSector, observation.z, observation.playerZVelocity,
                         gMe->posture, gMe->cantJump, jumpAttempts, targetAngle,
                         int(input.forward));
                event("jump_input_emitted", detail);
            }
            break;
        case kJumpAirborne:
            input.forward = kFullThrottle;
            if (supportStep)
            {
                int hold = input.forward;
                if (solveSupportArc(observation.playerZVelocity, *supportStep,
                                    x, y, targetAngle, hold))
                    input.forward = int16_t(hold);
            }
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
                const bool settled = observation.playerZVelocity == 0
                    && (!gMe->pXSprite || gMe->pXSprite->height == 0);
                if (!settled)
                    break;
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
                input.forward = kFullThrottle;
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
        // A local recovery probe first tests ordinary supported walking.  It
        // may add one bounded jump only after the movement-progress detector
        // proves the walk is blocked; treating its label as an immediate jump
        // edge caused the AGTST10 wall-hopping loop.
        bool needsLift = capability == kTraversalJumpable && targetId != -2;
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
            input.forward = kFullThrottle;
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
        const int playerCell = nearestNavCell(observation.sector, observation.x,
                                              observation.y,
                                              llmapper::capability::playerFloorZ());
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
        auditPredictedMotion();
        const GINPUT input = decideInput();
        if (input.forward || input.strafe)
            lastCommandedMoveTick = observation.tick;
        recordMotionPrediction(input);
        return input;
    }

    //---------------------------------------------------------------------
    // Does the motion model actually agree with the engine?
    //
    // Everything the bot decides about a jump rests on stepPlayerMotion
    // reproducing one frame of Blood exactly.  Rather than trust that, step
    // the model forward from the state the bot saw last frame with the input
    // it actually sent, and compare against what the engine did.  A model
    // that agrees says nothing; a model that is wrong names the term.
    //
    // This is a check on the model, not a source of behaviour: nothing reads
    // the result, and no constant is fitted to it.
    //---------------------------------------------------------------------
    void recordMotionPrediction(const GINPUT &input)
    {
        motionAuditValid = false;
        if (!gMe || !gMe->pSprite || input.strafe)
            return;   // strafing is not in the model, so do not claim it is
        const POSTURE &posture = gMe->pPosture[gMe->lifeMode][gMe->posture];
        const int foot = llmapper::capability::playerFootOffset();
        motionAuditFloorZ = llmapper::capability::playerFloorZ();
        motionAuditDepth = llmapper::capability::playerInDepth();
        motionAuditForward = input.forward;
        motionAuditJumped = input.buttonFlags.jump && !gMe->cantJump
            && gMe->pXSprite && gMe->pXSprite->height == 0;
        motionAuditBefore = currentMotion();
        motionAuditAngle = gMe->pSprite->ang;
        motionAuditFoot = foot;
        motionAuditPredicted = motionAuditBefore;
        if (motionAuditJumped)
            motionAuditPredicted.zvel = llmapper::capability::playerJumpImpulse();
        llmapper::capability::stepPlayerMotion(motionAuditPredicted, input.forward,
                                               motionAuditAngle, motionAuditFloorZ,
                                               posture.frontAccel, foot,
                                               llmapper::capability::playerAirDrag());
        motionAuditFrame = gFrame;
        motionAuditValid = true;
    }

    void auditPredictedMotion()
    {
        if (!motionAuditValid || !gMe || !gMe->pSprite)
            return;
        motionAuditValid = false;
        if (gFrame != motionAuditFrame + 1)
            return;   // frames were skipped; the comparison would be meaningless
        // Things the model does not claim to reproduce, counted rather than
        // reported: clipping, which it has no geometry for, and the body's
        // extents changing with the animation inside the frame the model has
        // already predicted.
        if (gMe->pSprite->extra > 0 && gMe->pSprite->extra < kMaxXSprites
            && (gSpriteHit[gMe->pSprite->extra].hit != 0
                || gSpriteHit[gMe->pSprite->extra].ceilhit != 0))
        {
            // A wall takes the velocity into it away through
            // actWallBounceVector, and a ceiling reverses an eighth of the
            // rise; both are clipping, and the model has no geometry.
            ++motionAuditClipped;
            return;
        }
        if (llmapper::capability::playerFootOffset() != motionAuditFoot)
        {
            ++motionAuditReshaped;
            return;
        }
        if (motionAuditDepth || llmapper::capability::playerInDepth())
        {
            ++motionAuditBuoyant;   // water and goo change gravity; not modelled
            return;
        }
        if (llmapper::capability::playerFloorZ() != motionAuditFloorZ)
        {
            ++motionAuditReshaped;  // the ground moved or the footprint left it
            return;
        }
        const llmapper::capability::MotionState actual = currentMotion();
        const int dx = actual.x - motionAuditPredicted.x;
        const int dy = actual.y - motionAuditPredicted.y;
        const int dz = actual.z - motionAuditPredicted.z;
        const int dxv = actual.xvel - motionAuditPredicted.xvel;
        const int dyv = actual.yvel - motionAuditPredicted.yvel;
        const int dzv = actual.zvel - motionAuditPredicted.zvel;
        // A one-unit rounding difference is not a disagreement.  Velocities
        // are 1/4096 of a world unit, so tolerate the same magnitude there.
        if (std::abs(dx) <= 1 && std::abs(dy) <= 1 && std::abs(dz) <= 1
            && std::abs(dxv) <= 0x1000 && std::abs(dyv) <= 0x1000
            && std::abs(dzv) <= 0x1000)
        {
            ++motionAuditAgreed;
            return;
        }
        if (dxv == 0 && dyv == 0 && dzv == 0)
        {
            // Same velocity, different place: playerProcess pushes the body
            // out of geometry with pushmove_old before ProcessInput runs, and
            // that moves the player without touching the velocity.  Another
            // thing the model has no geometry for.
            ++motionAuditPushed;
            return;
        }
        ++motionAuditDiverged;
        if (motionAuditDiverged > kMotionAuditReports)
            return;
        char detail[448];
        snprintf(detail, sizeof(detail),
                 "forward=%d jumped=%d ang=%d foot=%d floor=%d"
                 " from=(%d,%d,%d) vel0=(%d,%d,%d)"
                 " predicted=(%d,%d,%d) predvel=(%d,%d,%d)"
                 " actual=(%d,%d,%d) actvel=(%d,%d,%d)"
                 " d=(%d,%d,%d) dvel=(%d,%d,%d) agreed=%d clipped=%d reshaped=%d",
                 motionAuditForward, motionAuditJumped ? 1 : 0, motionAuditAngle,
                 motionAuditFoot, motionAuditFloorZ,
                 motionAuditBefore.x, motionAuditBefore.y, motionAuditBefore.z,
                 motionAuditBefore.xvel, motionAuditBefore.yvel, motionAuditBefore.zvel,
                 motionAuditPredicted.x, motionAuditPredicted.y, motionAuditPredicted.z,
                 motionAuditPredicted.xvel, motionAuditPredicted.yvel, motionAuditPredicted.zvel,
                 actual.x, actual.y, actual.z, actual.xvel, actual.yvel, actual.zvel,
                 dx, dy, dz, dxv, dyv, dzv, motionAuditAgreed,
                 motionAuditClipped, motionAuditReshaped);
        event("motion_model_diverged", detail);
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
        situation.critical = observation.health > 0 && observation.health < criticalHealth();
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
                hazardEscapeUnavailableEmitted = false;
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
            // Hazard avoidance is an exclusive state.  If the safe exit is
            // temporarily unavailable, executing the suspended exploration
            // objective in alternating frames sends the bot directly back
            // toward the hazard and continually resets both routes.  Hold
            // the current layer until an exit reopens or the evidence ages
            // out; the objective itself remains intact.
            if (!hazardEscapeUnavailableEmitted)
            {
                hazardEscapeUnavailableEmitted = true;
                event("hazard_escape_waiting",
                      "reason=no_currently_traversable_safe_exit");
            }
            setGoal("WAIT_FOR_SAFE_HAZARD_EXIT", observation.sector);
            return composeInput(move, combat, use);
        }
        hazardEscapeUnavailableEmitted = false;

        const DoorTiming standingIn = doorTiming(observation.sector);
        const SupportRef standingSupport = currentPlayerSupport();
        const bool supportBelongsToCurrentSector =
            (standingSupport.kind == kSupportSectorFloor
             && standingSupport.id == observation.sector)
            || (standingSupport.kind == kSupportSpriteFloor
                && inRange(standingSupport.id, 0, kMaxSprites)
                && sprite[standingSupport.id].sectnum == observation.sector);
        const bool ridingSafeSupport = standingIn.moving
            && supportBelongsToCurrentSector
            && sectorSweptOccupancySafe(observation.sector);
        if (ridingSafeSupport)
            supportRideSettleUntil = observation.tick + 2 * kSupportPoseSettleTicks;
        if (ridingSafeSupport || observation.tick < supportRideSettleUntil)
        {
            // This is a world-action step, not idleness.  Retain the active
            // mission but deliberately issue no translation until the
            // support reaches a reusable stable pose.  Walking off during
            // the sweep made carriers indistinguishable from transient
            // doors and discarded the very state the actuator established.
            setGoal(ridingSafeSupport ? "REMAIN_SUPPORTED" : "SETTLE_SUPPORT_POSE",
                    observation.sector);
            noteCameraOwner("WORLD_ACTION_WAIT", observation.sector,
                            observation.angle, fix16_to_int(gMe->q16look));
            return composeInput(move, combat, use);
        }
        // Leave a door that is shutting, and leave one that will shut on a
        // timer if there is no longer any reason to be standing in it.  With
        // nothing to execute the bot otherwise waits out the mechanism, or
        // starts a search probe, in the one place on the level where staying
        // put is fatal.
        const bool loiteringInMechanism = standingIn.autoCloses
            && !currentObjective.active;
        if ((standingIn.closing && !sectorSweptOccupancySafe(observation.sector))
            || loiteringInMechanism)
        {
            if (clearingSector != observation.sector || clearingWall < 0)
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
        if (currentObjective.active && dynamicPrerequisiteEstablished())
            completeObjective("route_condition_established");
        if (currentObjective.active && !suspendedObjectiveActive)
            selectDynamicPrerequisite();
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
                     || ((currentObjective.type == kObjectiveInteraction
                          || currentObjective.type == kObjectiveInvestigate)
                         && currentObjective.wall != openedRouteWall))
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
            else if (currentObjective.type == kObjectiveFrontier)
            {
                // The interaction has already handed control to a concrete
                // connection.  Other receivers on the same TX channel may
                // open too, but they remain ordinary conditional frontiers;
                // they must not steal the active route mid-crossing.
                openedRouteWall = -1;
            }
        }
        if (!currentObjective.active)
        {
            rebuildLedger();
            llmapper::Mission chosen = llmapper::selectMission(
                ledger, observation.tick, heldKeyMask, missionOpportunity,
                availableEffectCapabilities());
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
        bool mechanismBusy = busyValueInMotion(observation.localSectorBusy);
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
        if ((searchProbeActive || !localDynamicJumpAttempted)
            && localJumpProbeCount < 4)
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
                if (localJumpProbeCount == 0)
                    dumpLedger("before_local_recovery_probe");
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
                localDynamicJumpAttempted = false;
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
            // Whether the bot's account of Blood's physics held up for the
            // whole run.  Anything other than zero divergences means a jump
            // was planned against arithmetic the engine does not run.
            char audit[192];
            snprintf(audit, sizeof(audit),
                     "agreed=%d diverged=%d clipped=%d pushed=%d reshaped=%d buoyant=%d",
                     motionAuditAgreed, motionAuditDiverged, motionAuditClipped,
                     motionAuditPushed, motionAuditReshaped, motionAuditBuoyant);
            event("motion_model_audit", audit);
            fprintf(telemetry, "{\"type\":\"summary\",\"result\":\"%s\",\"failure_reason\":\"%s\",\"game_time\":%d,\"total_sectors\":%d,\"visited_sectors\":%u,\"observed_sectors\":%u}\n",
                    result.c_str(), failureReason.c_str(), (gFrame * kTicsPerFrame) / kTicsPerSec,
                    numsectors, unsigned(visitedSectors.size()), unsigned(observedSectors.size()));
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
    char detail[192];
    if (inRange(source, 0, MAXSPRITES) && sprite[source].statnum != MAXSTATUS)
        snprintf(detail, sizeof(detail),
                 "source=%d source_type=%d source_stat=%d type=%d amount=%d",
                 source, int(sprite[source].type), int(sprite[source].statnum),
                 damageType, amount);
    else
        snprintf(detail, sizeof(detail),
                 "source=%d source_type=-1 source_stat=-1 type=%d amount=%d",
                 source, damageType, amount);
    m_impl->event("bot_damaged", detail);
}

void LLMapperBot::OnVectorResolved(int hit, int sector, int wall, int spriteId,
                                   int x, int y, int z)
{
    if (!m_enabled || !m_impl || m_impl->currentGoal != "VECTOR_ACTIVATE")
        return;
    m_impl->vectorResolved(hit, sector, wall, spriteId, x, y, z);
    char detail[192];
    snprintf(detail, sizeof(detail),
             "hit=%d sector=%d wall=%d sprite=%d at=(%d,%d,%d) objective=%d",
             hit, sector, wall, spriteId, x, y, z, m_impl->lastStateTarget);
    m_impl->event("vector_scan_resolved", detail);
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
