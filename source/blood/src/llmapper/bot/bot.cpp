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
#include <tuple>
#include <vector>

#include "build.h"
#include "colmatch.h"
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
#include "../../tile.h"
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

// Persistent observation/task identity for one concrete standable pose.
// The sector-local grid bucket below deliberately omits Z because it indexes
// every collision layer at one XY.  Persistent world knowledge must not:
// sprite tops, ROR layers and moving-support endpoints can all occupy the
// same bucket while remaining different physical places.
using PhysicalPoseKey = std::tuple<int, int, int, int, int, int>;

namespace
{
constexpr int kDefaultTimeoutSeconds = 30 * 60;
constexpr int kDefaultStallSeconds = 45;
constexpr int kObservationPeriod = 4;
constexpr int kTrajectoryPeriod = 8;
enum DebugOverlayLayer : uint32_t
{
    kDebugMesh = 1u << 0,
    kDebugPoses = 1u << 1,
    kDebugWalk = 1u << 2,
    kDebugJump = 1u << 3,
    kDebugCrouch = 1u << 4,
    kDebugRide = 1u << 5,
    kDebugObservation = 1u << 6,
    kDebugInteractions = 1u << 7,
    kDebugTasks = 1u << 8,
    kDebugRoute = 1u << 9,
    kDebugHypothetical = 1u << 10,
    kDebugAll = (1u << 11) - 1,
};
constexpr uint32_t kDefaultDebugLayers = kDebugMesh | kDebugJump
    | kDebugCrouch | kDebugRide | kDebugObservation
    | kDebugInteractions | kDebugTasks | kDebugRoute;
constexpr int kDebugOverlayRadius = 12288;
constexpr int kMovementStuckTicks = 2 * kTicsPerSec;
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
// progress, or simply runs too long, releases the bot back to selection and
// records a deterministic failure under the current relevant evidence.
constexpr int kObjectiveStallTicks = 5 * kTicsPerSec;
constexpr int kObjectiveHardTicks = 60 * kTicsPerSec;
constexpr int kObjectiveProgressEpsilon2 = 192 * 192;
// Navigation grid resolution in Build units (1 << 8 = 256), a little wider
// than the player's clip radius so an ordinary corridor keeps several cells.
constexpr int kNavGridShift = 8;
// How long a route the bot just opened stays the most interesting thing in
// the level.  Long enough to walk to it, short enough that a stale opening
// does not outrank live exploration.
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
// Consecutive decisions in which navigation produced no usable movement
// before the destination is released.  The objective budget and the
// stationary watchdog are the primary bounds; this is a backstop.
constexpr int kNavigationFailureLimit = 40;
// Long enough for Blood to switch the player to the crouch sequence and for
// one observation to sample the resulting body extents.
constexpr int kCrouchCalibrationTicks = kTicsPerSec;
// Minimum gap between deliberate activations of the same mechanism, so a
// reusable door is not restarted before it has finished moving.
constexpr int kReactivationCooldownTicks = 5 * kTicsPerSec;
// How many times the bot may break out of a detected loop before the run is
// declared genuinely stuck.
constexpr int kMaxLoopBreaks = 6;
// Deliberate activations of one mechanism before the bot accepts that
// pressing it again is not going to help.
constexpr int kMaxActivationAttempts = 3;
// Longest straight shortcut route smoothing may create. Keep this local to
// three physical mesh samples: smoothing is presentation/execution polish,
// never permission to skip a concrete crossing or support state.
constexpr int kRouteSmoothingSpan = 3 << kNavGridShift;
// Negative synthetic edge identity reserved for Blood's explicit stacked
// room/link transition.  Ordinary intra-sector links use -1 and wall portal
// links use their non-negative engine wall id.
constexpr int kNavRorTransitionEdge = -2;
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
// Divergences between the motion model and the engine worth reporting.  If
// the model is wrong the first few say why; after that it is noise.
constexpr int kMotionAuditReports = 12;
// How far off the bot's heading a target may be and still be worth a shot
// taken in passing.  256 of 2048 is 45 degrees.
constexpr int kOpportunisticAimCone = 256;
// Clearance the bot leaves inside the pitchfork's actual reach, so a swing
// taken from the edge of it still connects after a frame of drift.
constexpr int kMeleeReachMargin = 128;

// Match the engine's ordinary running input exactly. Navigation correctness
// must not depend on a one-unit trajectory perturbation.
static const int kFullThrottle = llmapper::capability::playerMoveInputMax();

// How far the pitchfork actually reaches, less the margin above.
static int meleeReach()
{
    return llmapper::capability::playerPitchforkReach() - kMeleeReachMargin;
}

// VECTORDATA::maxDist uses zero as the engine sentinel for an unbounded
// vector scan (actFireVector/VectorScan preserve that meaning).  Planner
// range comparisons need an ordinary upper bound, otherwise a Tommy gun's
// native zero rejects every non-zero firing pose.
static int plannerVectorReach(VECTOR_TYPE vectorType)
{
    const int engineReach = gVectorData[vectorType].maxDist;
    return engineReach > 0 ? engineReach : INT32_MAX;
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
    int sprite = -1;
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
static int playerCollisionClearance()
{
    const llmapper::capability::Envelope body = llmapper::capability::playerEnvelope();
    // playerProcess and MoveDude pass these quarter-scaled distances to
    // pushmove_old.  Sector Z values are compared against that collision
    // envelope; the sprite's full visual top-to-bottom height is not the
    // hull Blood uses for passage clearance.
    return std::max(1, body.ceilingDistance + body.floorDistance);
}

// Smallest body envelope actually observed while crouched.  Blood swaps the
// player to the crouch animation sequence, and GetSpriteExtents follows that
// sequence, so the crouched collision body really is shorter than the
// standing one -- but only measurable once the bot has crouched at least
// once.  Until then, fall back to the engine's own posture table.
static int gObservedCrouchClearance = 0;
static int gStandingClearance = 0;

struct PlayerCollisionShape
{
    bool known = false;
    int radius = 0;
    int ceilingDistance = 0;
    int floorDistance = 0;
    int footOffset = 0;
};

static PlayerCollisionShape gObservedCrouchShape;

static int playerStandingClearance()
{
    return gStandingClearance > 0 ? gStandingClearance
                                  : playerCollisionClearance();
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

static PlayerCollisionShape livePlayerCollisionShape()
{
    PlayerCollisionShape shape;
    const llmapper::capability::Envelope body =
        llmapper::capability::playerEnvelope();
    if (!body.known || !gMe || !gMe->pSprite)
        return shape;
    shape.known = true;
    shape.radius = body.radius;
    shape.ceilingDistance = std::max(0, body.ceilingDistance);
    shape.floorDistance = std::max(0, body.floorDistance);
    shape.footOffset = body.bottom - gMe->pSprite->z;
    return shape;
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

// Ask NBlood which concrete collision surface owns the floor at a proposed
// player pose.  Navigation support identity is deliberately nothing more
// than this engine answer: artwork alignment and Build sector ownership do
// not grant or deny accessibility on their own.
static bool engineHasSupportAt(const SupportRef &support, int sectorId,
                               int x, int y, int surfaceZ, int radius)
{
    if (!inRange(sectorId, 0, numsectors) || support.id < 0)
        return false;
    int ceilingZ = 0, ceilingHit = 0, floorZ = 0, floorHit = 0;
    GetZRangeAtXYZ(x, y, surfaceZ - 1, sectorId,
                   &ceilingZ, &ceilingHit, &floorZ, &floorHit,
                   radius, CLIPMASK0,
                   PARALLAXCLIP_CEILING | PARALLAXCLIP_FLOOR);
    const int expected = support.kind == kSupportSpriteFloor
        ? (0xc000 | support.id) : (0x4000 | support.id);
    return floorHit == expected && floorZ == surfaceZ;
}

// Resolve the physical plane owned by a support at one XY pose.  A support
// is not necessarily flat: sector floors and sloped floor sprites can both
// change Z while retaining the same collision identity.
static bool engineSupportZAt(const SupportRef &support, int x, int y,
                             int &surfaceZ)
{
    if (support.kind == kSupportSectorFloor)
    {
        if (!inRange(support.id, 0, numsectors)
            || inside(x, y, support.id) != 1)
            return false;
        surfaceZ = getflorzofslope(support.id, x, y);
        return true;
    }
    if (support.kind != kSupportSpriteFloor
        || !inRange(support.id, 0, kMaxSprites)
        || sprite[support.id].statnum >= kMaxStatus)
        return false;
    const spritetype &record = sprite[support.id];
    const int alignment = record.cstat & CSTAT_SPRITE_ALIGNMENT_MASK;
    if (alignment == CSTAT_SPRITE_ALIGNMENT_FLOOR
        || alignment == CSTAT_SPRITE_ALIGNMENT_SLOPE)
        surfaceZ = spriteGetZOfSlope(uint16_t(support.id), { x, y });
    else
    {
        int bottom = 0;
        spriteCollisionExtents(support.id, surfaceZ, bottom);
    }
    return true;
}

// How much horizontal error a concrete sprite-support pose tolerates before
// NBlood stops naming that same collision surface as the player's floor.
// This is intentionally sampled through GetZRange rather than reconstructed
// from artwork dimensions: floor sprites, slopes, wall sprites, voxels and
// ordinary blocking sprites all keep the exact collision semantics used by
// live movement. The result is route quality, not reachability; an edge pose
// remains in the graph with zero margin when it is the only physical option.
static int spriteSupportClearance(int sectorId, int spriteId, int x, int y,
                                  int radius)
{
    if (!inRange(sectorId, 0, numsectors)
        || !inRange(spriteId, 0, kMaxSprites))
        return 0;
    auto retained = [&](int px, int py) {
        int surfaceZ = 0;
        const spritetype &record = sprite[spriteId];
        const int alignment = record.cstat & CSTAT_SPRITE_ALIGNMENT_MASK;
        if (alignment == CSTAT_SPRITE_ALIGNMENT_FLOOR
            || alignment == CSTAT_SPRITE_ALIGNMENT_SLOPE)
            surfaceZ = spriteGetZOfSlope(uint16_t(spriteId), { px, py });
        else
        {
            int bottom = 0;
            spriteCollisionExtents(spriteId, surfaceZ, bottom);
        }
        return engineHasSupportAt(SupportRef(kSupportSpriteFloor, spriteId),
                                  sectorId, px, py, surfaceZ, radius);
    };

    static const int directions[8][2] = {
        { 256, 0 }, { -256, 0 }, { 0, 256 }, { 0, -256 },
        { 181, 181 }, { 181, -181 }, { -181, 181 }, { -181, -181 },
    };
    constexpr int maximum = 512;
    constexpr int stride = 64;
    int clearance = maximum;
    for (const auto &direction : directions)
    {
        int retainedDistance = 0;
        for (int distance = stride; distance <= maximum; distance += stride)
        {
            const int px = x + direction[0] * distance / 256;
            const int py = y + direction[1] * distance / 256;
            if (!retained(px, py))
                break;
            retainedDistance = distance;
        }
        clearance = std::min(clearance, retainedDistance);
    }
    return clearance;
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
    // getzrange sweeps the whole player cylinder.  At a raised boundary the
    // floor which actually catches Caleb can therefore belong to an adjacent
    // sector while his origin (and pSprite->sectnum) is still on the lower
    // side.  floorHit is the collision authority in exactly that case.  Using
    // sectnum invented impossible states such as "sector-5 support at the
    // sector-2 floor height" and stranded routes on a pose already passed.
    if ((floorHit & 0xc000) == 0x4000)
        return SupportRef(kSupportSectorFloor, floorHit & 0x3fff);
    return SupportRef(kSupportSectorFloor, gMe->pSprite->sectnum);
}

// Sector-floor ids are collision metadata, not distinct navigation places.
// Thin and overlapping Build partitions can describe the same occupiable
// support plane while the engine legitimately keeps either sector number in
// the player sprite.  A sprite/voxel support retains exact collision identity;
// a sector floor pose is established when the player is physically inside
// that polygon on the expected plane.
static bool playerOccupiesSupportPose(const SupportRef &expected,
                                      int expectedZ)
{
    if (!gMe || !gMe->pSprite)
        return false;
    const int liveFloor = llmapper::capability::playerFloorZ();
    if (std::abs(liveFloor - expectedZ)
        > llmapper::capability::playerStepHeight() / 2)
        return false;
    // The floor hit returned for the player's whole collision cylinder owns
    // support even when the origin has already crossed an adjacent sector
    // boundary. Never weaken that direct engine answer with a point-in-polygon
    // test intended only for equivalent overlapping floor metadata.
    if (currentPlayerSupport() == expected)
        return true;
    if (expected.kind == kSupportSpriteFloor)
        return false;
    if (expected.kind != kSupportSectorFloor
        || !inRange(expected.id, 0, numsectors)
        || inside(gMe->pSprite->x, gMe->pSprite->y, expected.id) != 1)
        return false;
    return std::abs(getflorzofslope(expected.id, gMe->pSprite->x,
                                    gMe->pSprite->y) - liveFloor)
        <= llmapper::capability::playerStepHeight() / 2;
}

// Exact environmental-contact rule used by actTouchFloor.  It is only ever
// true for the sector floor collision surface; a sprite floor hit does not
// call actTouchFloor and therefore must not inherit the sector's damage.
static bool engineSupportDamagesPlayer(const SupportRef &support)
{
    if (support.kind != kSupportSectorFloor
        || !inRange(support.id, 0, numsectors))
        return false;
    const sectortype &record = sector[support.id];
    const XSECTOR *extra = record.extra > 0 && record.extra < kMaxXSectors
        ? &xsector[record.extra] : nullptr;
    bool damages = extra
        && (record.type == kSectorDamage || extra->damageType > 0);
#ifdef NOONE_EXTENSIONS
    if (gModernMap && damages && record.type == kSectorDamage && !extra->state)
        damages = false;
#endif
    return damages || tileGetSurfType(support.id, 0x4000) == kSurfLava;
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

static int bodyRadiusOf()
{
    return llmapper::capability::playerEnvelope().radius;
}

// A NavCell stores the physical support plane under the player's feet.
// Convert that plane to the same sprite-origin Z used by live cansee() calls
// with the engine-derived sprite extents, never a guessed "eye offset".
static int playerOriginAtSupport(int supportZ)
{
    return supportZ - llmapper::capability::playerFootOffset();
}

// Ask the same collision resolver Blood uses after moving actors whether the
// player's complete hull can remain at this support pose.  A geometric
// distance-to-wall estimate is not equivalent around corners, narrow stairs,
// sprites, or overlapping sectors: pushmove is the authoritative predicate.
static bool enginePlayerPoseFits(int sectorId, int x, int y, int supportZ,
                                 bool crouched = false)
{
    if (!inRange(sectorId, 0, numsectors) || !gMe || !gMe->pSprite)
        return false;
    int32_t resolvedX = x;
    int32_t resolvedY = y;
    const PlayerCollisionShape liveShape = livePlayerCollisionShape();
    const PlayerCollisionShape &shape = crouched && gObservedCrouchShape.known
        ? gObservedCrouchShape : liveShape;
    if (!shape.known)
        return false;
    const int originZ = supportZ - shape.footOffset;
    int32_t resolvedZ = originZ;
    int16_t resolvedSector = int16_t(sectorId);
    int ceilingDistance = 0;
    int floorDistance = 0;
    ceilingDistance = shape.ceilingDistance;
    floorDistance = shape.floorDistance;
    const int result = pushmove_old(
        &resolvedX, &resolvedY, &resolvedZ, &resolvedSector,
        shape.radius,
        ceilingDistance, floorDistance, CLIPMASK0);
    return result >= 0 && resolvedX == x && resolvedY == y
        && resolvedZ == originZ
        && inRange(int(resolvedSector), 0, numsectors);
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
        const int toFloor = getflorzofslope(portal.to, portal.x, portal.y);
        const int toCeiling = getceilzofslope(portal.to, portal.x, portal.y);
        // ActionScan resolves a two-sided wall only where hitscan actually
        // strikes its solid upper/lower band. The shared vertical overlap is
        // the opening and aiming there passes straight through it.
        if (toFloor != fromFloor)
        {
            top = std::min(fromFloor, toFloor);
            bottom = std::max(fromFloor, toFloor);
        }
        else if (toCeiling != fromCeiling)
        {
            top = std::min(fromCeiling, toCeiling);
            bottom = std::max(fromCeiling, toCeiling);
        }
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
    if (info.dmgControl[kDamageBullet] > 0)
        effects |= llmapper::kEffectBulletDamage;
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
            const bool walkable = enoughWidth && standingClearance && std::abs(floorDelta) <= llmapper::capability::playerStepHeight();
            const bool crouchable = enoughWidth && !standingClearance && crouchingClearance
                && std::abs(floorDelta) <= llmapper::capability::playerStepHeight();
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
                && (!portal.traversable || std::abs(portal.floorDelta) > llmapper::capability::playerStepHeight());

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
            else if (portal.interactionAffordance && !portal.blockedBySprite)
            {
                // A sprite obstruction makes the *route* actionable, not the
                // wall behind it.  The collision scan above has already
                // emitted the operable sprite as the concrete affordance, and
                // ActionScan will hit that sprite before it can ever name this
                // wall.  Recording both creates a phantom task which survives
                // after the sprite opens the route and later drags exploration
                // back to a surface the engine never considered usable.  If
                // the wall has an independent push affordance it will be
                // observed normally once the obstruction has moved away.
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
        const unsigned damageEffects = acceptedDamageEffects(candidate);
        // Whether the player can operate a sprite is a fact Blood records on
        // the sprite, not something to be inferred from its status list or
        // its type number.  A mapper is free to wire a plain decoration to a
        // channel, and one that answers to Use is a switch whatever tile it
        // wears; keying off the type whitelist alone left the bot walking
        // past hand-built mechanisms as though they were scenery.
        const bool operable = !isEnemy && validXSprite(candidate.extra)
            && (xsprite[candidate.extra].Push || xsprite[candidate.extra].Vector);
        const bool authoredDamageEffect = damageEffects != llmapper::kEffectNone
            && validXSprite(candidate.extra) && xsprite[candidate.extra].txID > 0;
        const bool vectorDamageReceiver = damageEffects != llmapper::kEffectNone
            && validXSprite(candidate.extra) && xsprite[candidate.extra].Vector
            && xsprite[candidate.extra].health > 0;
        const bool knownTraversalBlocker = damageEffects != llmapper::kEffectNone
            && std::any_of(result.interactions.begin(), result.interactions.end(),
                [&](const InteractionCandidate &interaction) {
                    return interaction.kind == kInteractionSprite
                        && interaction.id == i
                        && interaction.activationMode == llmapper::kActivateDamage;
                });
        const bool usefulDamage = authoredDamageEffect || knownTraversalBlocker;
        if (!isEnemy && !isItem && !isThing && !isSwitch && !operable)
            continue;
        if (candidate.index == player->index || candidate.sectnum < 0)
            continue;
        if (isEnemy && (!isEnemyType(candidate.type) || !validXSprite(candidate.extra) || xsprite[candidate.extra].health == 0))
            continue;
        if (isItem && !operable && itemCategory(candidate.type) == nullptr)
            continue;
        if ((isThing || isSwitch) && !operable && !usefulDamage)
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
                         // Vector is the delivery mechanism for a
                         // health-bearing receiver, not proof that one hit
                         // completed it. Keep shooting until the receiver's
                         // authored effect changes the world or its collision
                         // body is actually removed.
                         : vectorDamageReceiver ? kObjectDamageable
                         : operable ? kObjectInteractive
                         : usefulDamage ? kObjectDamageable
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
        kObjectiveExit,
    };

    struct Objective
    {
        ObjectiveType type = kObjectiveNone;
        bool active = false;
        llmapper::WorkId work;
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
        int routeStepX = INT32_MIN;
        int routeStepY = INT32_MIN;
        int approachCell = -1;
        int startedTick = -1;
        int inspectX = 0;
        int inspectY = 0;
        int inspectZ = 0;
        bool inspectPoseValid = false;
        bool requiresOccupancy = false;
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
        // Stable key assigned when the persistent record is created.  The
        // target surface/approach sector may legitimately change as more is
        // observed; those mutable execution facts must never re-key the
        // remembered work item underneath an active objective.
        int identity = -1;
        InteractionKind kind = kInteractionWall;
        int id = -1;
        int fromSector = -1;
        int targetSector = -1;
        int x = 0;
        int y = 0;
        int z = 0;
        int approachFloorZ = INT32_MAX;
        int key = 0;
        bool locked = false;
        bool reversible = false;
        llmapper::ActivationMode activationMode = llmapper::kActivateUse;
        unsigned requiredEffects = llmapper::kEffectNone;
        // One logical receiver may expose several distinct physical faces.
        // Keep those faces as execution alternatives instead of letting the
        // last observation overwrite the only ActionScan target we retain.
        std::vector<InteractionCandidate> surfaces;
        bool observed = false;
        bool attempted = false;
        bool activated = false;
        int activationCount = 0;
        int lastActivationTick = -1;
        int activationWorldRevision = -1;
        int beforeState = 0;
        int afterState = 0;
        int unavailableFromSector = -1;
        int unavailableState = 0;
        int unavailableAttempts = 0;
        int unavailablePose = 0;
        // An explicit surface retry owns its selected wall for this objective.
        // Passive observations may still choose a nearer equivalent surface
        // while approaching the initial candidate.
        int selectedSurfaceObjectiveTick = -1;
        int state = 0;
        bool engineAccepted = false;
        bool observedLocalEffect = false;
        bool observedKnownWorldDelta = false;
        int consecutiveNoEffectAttempts = 0;
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
        int firingPoseCell = -1;
        int firingPoseTopologyRevision = -1;
        int firingPoseSignature = 0;
        // A Use target is reachable only when a connected standable pose
        // exists from which Blood's real ActionScan resolves that target.
        // The target's containing sector and XY are observation metadata,
        // never substitutes for this physical pose.
        int actionPoseCell = -1;
        int actionPoseTopologyRevision = -1;
        int actionPoseSignature = 0;
        int actionPoseStartArea = -1;
        SupportRef actionPoseStartSupport;
        int actionPoseHops = -1;
        int actionPoseAngle = 0;
        int actionPoseLook = 0;
        bool actionPoseCrouch = false;
        std::vector<WorldObjectRef> causalEffects;
        // Physical component before this activation. Comparing it with the
        // component after an observed world delta yields the action's causal
        // successor: newly reachable space that must be consumed before
        // unrelated backtracking. Handles survive nav-mesh rebuilds.
        std::set<PhysicalPoseKey> reachableBeforeActivation;
        int continuationTopologyRevision = -1;
        Portal target;
    };

    // A capability name is not yet a usable action.  This record binds an
    // accepted effect to one producer and to the physical distance envelope
    // in which that producer can deliver it.  Ledger projection and
    // execution consume the same answer, so "bullet damage available" can
    // no longer mean infinite range to one layer and pitchfork range to the
    // next.
    struct EffectDelivery
    {
        bool available = false;
        bool ranged = false;
        bool explosive = false;
        int weapon = 0;
        VECTOR_TYPE vectorType = kVectorTine;
        int minimumDistance = 0;
        int maximumDistance = 0;
    };

    FILE *telemetry = nullptr;
    FILE *trajectory = nullptr;
    FILE *navMeshDump = nullptr;
    std::string telemetryPath = "llmapper-bot.ndjson";
    std::string trajectoryPath = "llmapper-bot.trajectory.ndjson";
    std::string navMeshDumpPath;
    std::string demoPath = "llmapper-bot.dem";
    int timeoutSeconds = kDefaultTimeoutSeconds;
    int stallSeconds = kDefaultStallSeconds;
    uint32_t debugLayers = kDefaultDebugLayers;
    int debugLedgerTick = -1;
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
    std::vector<NavEdgeFailure> navEdgeFailures;
    std::vector<InvestigateRecord> investigatedBoundaries;
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
    std::map<int, std::vector<Portal>> knownGraph;
    Portal routePortal;
    int knowledgeRevision = 0;
    int inventoryRevision = 0;
    int lastRangedCapability = -1;
    bool rangedPrerequisitePending = false;
    unsigned lastEffectCapabilities = ~0u;
    unsigned pendingEffectPrerequisites = llmapper::kEffectNone;
    int lastObservedSector = -1;
    int lastGeometryTelemetrySector = -1;
    bool worldLoadedTelemetryEmitted = false;
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
    int jumpAttempts = 0;
    JumpExecutionState jumpState = kJumpInactive;
    int jumpStateTick = -1;
    int jumpTargetX = 0;
    int jumpTargetY = 0;
    int jumpTargetSector = -1;
    int jumpSourceSector = -1;
    int jumpHeading = 0;
    int jumpForwardInput = 0;
    int jumpRouteFromCell = -1;
    int jumpRouteToCell = -1;
    NavRouteStep jumpRouteStep;
    bool jumpRouteStepValid = false;
    int lastDoorActionEventTick = -1;
    int lastUseProbeDoor = -1;
    int lastUseProbeTick = -1;
    std::string lastUseProbeStatus;
    int lastUsedDoor = -1;
    int lastUsedDoorFrom = -1;
    int lastUsedDoorTo = -1;
    int lastUsedDoorTick = -1;
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
        int clearance = 0;
        bool live = true;
        bool crouchOnly = false;
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
    mutable std::map<std::pair<int, int>, int> jumpReachCache;
    std::map<int64_t, std::vector<int>> navCellIndex;
    std::vector<NavCell> navCells;
    int navTopologySignature = 0;
    int navPoseSignatureValue = 0;
    int navDynamicSignatureValue = 0;
    int navTopologyRevision = 0;
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
    std::set<int> reportedRetainedPortalApproachRoutes;
    std::set<int> reportedFiringPoseSearches;
    bool reportedNavStartReconnect = false;
    int lastLedgerDumpTick = -1;
    int stationaryX = INT32_MIN;
    int stationaryY = INT32_MIN;
    int stationaryZ = INT32_MIN;
    int stationarySinceTick = 0;
    int lastCommandedMoveTick = 0;
    std::set<PhysicalPoseKey> observedCells;
    int visibilityFrontierTopologyRevision = -1;
    size_t visibilityFrontierObservedCount = size_t(-1);
    size_t visibilityFrontierFailureCount = size_t(-1);
    size_t visibilityFrontierVisitedSupportCount = size_t(-1);
    int visibilityFrontierReachableSignature = 0;
    std::vector<llmapper::VisibilityFrontier> visibilityFrontierCache;
    // Unobserved cells made reachable by deliberate world-changing actions.
    // This is conserved unresolved work, not a short timer or sector hint.
    // Physical observation work newly enabled by an action, owned by that
    // action transaction so it can be consumed when its proven boundary is
    // crossed. A global unowned set let an old lift activation dominate later
    // regions indefinitely.
    std::map<PhysicalPoseKey, int> causalContinuationCells;
    std::set<int64_t> visitedSupportPoses;
    int64_t pendingSupportPose = INT64_MIN;
    int pendingSupportPoseTick = -1;
    int supportRideSettleUntil = -1;
    int coverageTargetX = 0;
    int coverageTargetY = 0;
    int coverageTargetSector = -1;
    // Bounded-objective progress observation. A failed objective records one
    // typed attempt against relevant evidence; it does not put work to sleep.
    int objectiveProgressTick = -1;
    int objectiveBestDistance2 = INT32_MAX;
    int waitingSinceTick = -1;
    int objectiveBestRemainingSteps = INT32_MAX;
    int objectiveProgressSector = -1;
    std::set<int> objectiveProgressSectors;
    int objectiveProgressStepWall = -1;
    int objectiveProgressStepFrom = -1;
    int objectiveProgressStepTo = -1;
    int objectiveProgressStepX = INT32_MIN;
    int objectiveProgressStepY = INT32_MIN;
    int objectiveBestStepDistance2 = INT32_MAX;
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
    std::string selectedWorkReason = "NO_APPLICABLE_ACTION";
    unsigned heldKeyMask = 0;
    unsigned lastHeldKeyMask = 0;
    std::map<int, int> lockedDoorKey;
    mutable llmapper::CausalGraph mechanismCausality;
    mutable bool mechanismCausalityBuilt = false;
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

    int clearingSector = -1;
    int clearingWall = -1;
    int healthLostSinceObservation = 0;
    int damageBurstStartTick = -1;
    int damageBurstCount = 0;
    int damageBurstHealth = 0;
    // Damage belongs to the collision surface actTouchFloor actually names,
    // not to every pose sharing its Build sector.  A sprite bridge above a
    // damaging floor therefore remains a distinct safe support.
    std::map<SupportRef, int> hazardSupports;
    std::set<int> triedSurfaces;        // mechanism/wall pairs already attempted
    std::set<int> reopenableConnections;

    void openFiles()
    {
        telemetry = fopen(telemetryPath.c_str(), "wb");
        trajectory = fopen(trajectoryPath.c_str(), "wb");
        if (!navMeshDumpPath.empty())
            navMeshDump = fopen(navMeshDumpPath.c_str(), "wb");
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

    void dumpPhysicalNavMesh()
    {
        if (!navMeshDump)
            return;
        const int gameSeconds = (gFrame * kTicsPerFrame) / kTicsPerSec;
        fprintf(navMeshDump,
                "{\"type\":\"snapshot\",\"revision\":%d,\"game_time\":%d,\"cells\":%u}\n",
                navTopologyRevision, gameSeconds, unsigned(navCells.size()));
        for (const NavCell &cell : navCells)
        {
            fprintf(navMeshDump,
                    "{\"type\":\"cell\",\"revision\":%d,\"id\":%d,\"x\":%d,\"y\":%d,\"z\":%d,\"sector\":%d,\"support_kind\":%d,\"support\":%d,\"area\":%d,\"clearance\":%d,\"live\":%d}\n",
                    navTopologyRevision, cell.id, cell.center.x, cell.center.y,
                    cell.z, cell.sector, int(cell.support.kind), cell.support.id,
                    cell.walkArea, cell.clearance, cell.live ? 1 : 0);
        }
        for (const NavCell &cell : navCells)
        {
            for (const NavLink &link : cell.links)
            {
                if (!inRange(link.target, 0, int(navCells.size())))
                    continue;
                fprintf(navMeshDump,
                        "{\"type\":\"edge\",\"revision\":%d,\"from\":%d,\"to\":%d,\"mode\":%d,\"wall\":%d,\"gateway_x\":%d,\"gateway_y\":%d,\"condition_enabled\":%d,\"condition_mechanism\":%d,\"condition_state\":%d,\"transition\":%d,\"dynamic\":%d}\n",
                        navTopologyRevision, cell.id, link.target, int(link.mode),
                        link.wall, link.gateway.x, link.gateway.y,
                        link.condition.enabled ? 1 : 0, link.condition.mechanism,
                        link.condition.state, link.transition,
                        link.dynamic ? 1 : 0);
            }
        }
        fprintf(navMeshDump,
                "{\"type\":\"snapshot_end\",\"revision\":%d}\n",
                navTopologyRevision);
        fflush(navMeshDump);
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

    void actionResolved(int hit, int target, int extra, bool accepted, int key)
    {
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
    // Blood mechanisms form a small directed channel graph. Build it once
    // from the authoritative extended records: an RX object can itself emit
    // another TX/command, so the first receiver is not necessarily the world
    // effect the planner should associate with the action.
    const llmapper::CausalGraph &mechanismCausalGraph() const
    {
        if (mechanismCausalityBuilt)
            return mechanismCausality;
        mechanismCausalityBuilt = true;
        for (int i = 0; i < numsectors; ++i)
        {
            const int extra = sector[i].extra;
            if (extra <= 0 || extra >= kMaxXSectors || xsector[extra].rxID <= 0)
                continue;
            llmapper::CausalReceiver receiver;
            receiver.channel = xsector[extra].rxID;
            receiver.object = WorldObjectRef(kWorldSector, i);
            receiver.mechanism = i;
            receiver.outgoingChannel = xsector[extra].txID;
            receiver.command = xsector[extra].command;
            mechanismCausality.receivers.push_back(receiver);
        }
        for (int i = 0; i < numwalls; ++i)
        {
            const int extra = wall[i].extra;
            if (extra <= 0 || extra >= kMaxXWalls || xwall[extra].rxID <= 0)
                continue;
            llmapper::CausalReceiver receiver;
            receiver.channel = xwall[extra].rxID;
            receiver.object = WorldObjectRef(kWorldWall, i);
            receiver.outgoingChannel = xwall[extra].txID;
            receiver.command = xwall[extra].command;
            mechanismCausality.receivers.push_back(receiver);
        }
        for (int i = 0; i < kMaxSprites; ++i)
        {
            if (sprite[i].statnum >= kMaxStatus || !validXSprite(sprite[i].extra))
                continue;
            const XSPRITE &record = xsprite[sprite[i].extra];
            if (record.rxID <= 0)
                continue;
            llmapper::CausalReceiver receiver;
            receiver.channel = record.rxID;
            receiver.object = WorldObjectRef(kWorldSprite, i);
            receiver.outgoingChannel = record.txID;
            receiver.command = record.command;
            mechanismCausality.receivers.push_back(receiver);
        }
        return mechanismCausality;
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
        const std::vector<llmapper::CausalReceiver> receivers =
            mechanismCausalGraph().receiversFor(tx);
        for (const llmapper::CausalReceiver &receiver : receivers)
            if (receiver.object.kind == kWorldSector)
                return receiver.object.id;
        return -1;
    }

    std::vector<WorldObjectRef> finalMechanismEffects(int tx) const
    {
        std::vector<WorldObjectRef> result;
        if (tx <= 0)
            return result;
        const std::vector<llmapper::CausalReceiver> terminal =
            mechanismCausalGraph().receiversReachableFrom(tx, 8, true);
        for (const llmapper::CausalReceiver &receiver : terminal)
            if (std::find(result.begin(), result.end(), receiver.object) == result.end())
                result.push_back(receiver.object);
        return result;
    }

    void appendEffectSectors(const WorldObjectRef &effect,
                             std::set<int> &sectors) const
    {
        if (effect.kind == kWorldSector && inRange(effect.id, 0, numsectors))
            sectors.insert(effect.id);
        else if (effect.kind == kWorldWall && inRange(effect.id, 0, numwalls))
        {
            const int owner = wallOwnerSector(effect.id);
            if (inRange(owner, 0, numsectors))
                sectors.insert(owner);
            if (inRange(wall[effect.id].nextsector, 0, numsectors))
                sectors.insert(wall[effect.id].nextsector);
        }
        else if (effect.kind == kWorldSprite && inRange(effect.id, 0, kMaxSprites)
                 && inRange(sprite[effect.id].sectnum, 0, numsectors))
            sectors.insert(sprite[effect.id].sectnum);
    }

    bool interactionAffectsSector(const InteractionMemory &memory,
                                  int sectorId) const
    {
        if (memory.targetSector == sectorId
            || mechanismSector(memory.target, memory.targetSector) == sectorId)
            return true;
        std::set<int> effects;
        for (const WorldObjectRef &effect : memory.causalEffects)
            appendEffectSectors(effect, effects);
        return effects.count(sectorId) != 0;
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
        // Several wall records can be faces of one physical actuator sector.
        // Canonicalize those faces without merging a different actuator that
        // happens to share its causal receiver or TX channel.
        if (candidate.kind == kInteractionWall)
        {
            const int mechanism = mechanismSector(candidate.target,
                                                  candidate.targetSector);
            // One-sided XWALLs discovered across a narrow neighboring gap
            // have no geometric targetSector yet. Their immediate RX sector
            // is still the physical actuator sector and must be used before
            // inserting the memory; otherwise the map key changes after the
            // first observation and commitMission cannot find it.
            const int actuatorSector = llmapper::wallInteractionActuatorSector(
                candidate.targetSector, mechanism);
            const int physical = llmapper::wallInteractionAttemptKey(
                candidate.target.wall, candidate.fromSector,
                actuatorSector, candidate.target.wallPush,
                candidate.target.sectorPush,
                candidate.target.sectorPushCurrent, mechanism);
            if (physical >= 0)
                return physical;
        }
        int key = int(candidate.kind) * 1000000 + candidate.id + 1;
        // One sprite can expose both a mapper-authored Vector actuator and a
        // collision-removal affordance. Their completion evidence differs,
        // so they must not overwrite one another in persistent memory.
        if (candidate.kind == kInteractionSprite
            && candidate.activationMode == llmapper::kActivateDamage)
            key += 500000;
        return key;
    }

    int interactionMemoryKey(const InteractionMemory &memory) const
    {
        if (memory.identity >= 0)
            return memory.identity;
        if (memory.kind == kInteractionWall)
        {
            const int mechanism = mechanismSector(memory.target,
                                                  memory.targetSector);
            const int physical = llmapper::wallInteractionAttemptKey(
                memory.target.wall, memory.fromSector, memory.targetSector,
                memory.target.wallPush, memory.target.sectorPush,
                memory.target.sectorPushCurrent, mechanism);
            if (physical >= 0)
                return physical;
        }
        int key = int(memory.kind) * 1000000 + memory.id + 1;
        if (memory.kind == kInteractionSprite
            && memory.activationMode == llmapper::kActivateDamage)
            key += 500000;
        return key;
    }

    static llmapper::WorkId interactionWorkId(int interactionKey)
    {
        return llmapper::WorkId(llmapper::kWorkMechanism, interactionKey);
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

    int interactionLookTargetFromPose(const InteractionMemory &memory,
                                      bool crouched, int poseX, int poseY,
                                      int poseSpriteZ) const
    {
        if (!gMe || !gMe->pSprite)
            return 0;
        const POSTURE &posture = gMe->pPosture[gMe->lifeMode]
            [crouched ? kPostureCrouch : kPostureStand];
        const int eyeZ = poseSpriteZ - posture.eyeAboveZ;
        const int margin = 128;
        const int lower = memory.target.interactionTopZ + margin;
        const int upper = memory.target.interactionBottomZ - margin;
        const int horizontal = int(std::sqrt(double(distance2(
            poseX, poseY, memory.target.x, memory.target.y))));
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

    int interactionLookTarget(const InteractionMemory &memory, bool crouched) const
    {
        return interactionLookTargetFromPose(memory, crouched,
                                             observation.x, observation.y,
                                             gMe && gMe->pSprite
                                                 ? gMe->pSprite->z : observation.z);
    }

    int interactionFacingAngleFromPose(const InteractionMemory &memory,
                                       int poseX, int poseY, int /*poseAngle*/) const
    {
        if (memory.kind != kInteractionWall || memory.target.wall < 0)
            return getangle(memory.x - poseX, memory.y - poseY);

        // Sector-push ActionScan targets are resolved against the sector's
        // usable opening and have historically worked from the approach
        // point.  Only a genuine wall-push target needs the wall-face normal;
        // treating every sector-push marker as a wall face makes the bot
        // strafe sideways and can lose the engine's sector ActionScan target.
        if (!memory.target.wallPush)
            return getangle(memory.x - poseX, memory.y - poseY);

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
        const int toX = memory.target.x - poseX;
        const int toY = memory.target.y - poseY;
        const int64_t dot1 = int64_t(normal1X) * toX + int64_t(normal1Y) * toY;
        const int64_t dot2 = int64_t(normal2X) * toX + int64_t(normal2Y) * toY;
        return getangle(dot2 > dot1 ? normal2X : normal1X,
                        dot2 > dot1 ? normal2Y : normal1Y);
    }

    int interactionFacingAngle(const InteractionMemory &memory) const
    {
        return interactionFacingAngleFromPose(memory, observation.x,
                                              observation.y,
                                              observation.angle);
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
        if (memory.target.sectorPush && memory.target.floorDelta < -llmapper::capability::playerStepHeight())
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
        if (memory.identity < 0)
            memory.identity = key;
        const int oldState = memory.observed ? interactionStateSignature(memory) : 0;
        const bool canonicalSectorPush = candidate.kind == kInteractionWall
            && mechanismSector(candidate.target, candidate.targetSector) >= 0;
        if (candidate.kind == kInteractionWall && candidate.target.wall >= 0)
        {
            auto surface = std::find_if(
                memory.surfaces.begin(), memory.surfaces.end(),
                [&](const InteractionCandidate &known)
                {
                    return known.target.wall == candidate.target.wall
                        && known.fromSector == candidate.fromSector;
                });
            if (surface == memory.surfaces.end())
                memory.surfaces.push_back(candidate);
            else
                *surface = candidate;
        }
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
        const bool preserveBoardableTransportApproach = memory.observed
            && supportTransportBoardableAtActionPose(memory)
            && !supportTransportBoardableAtActionPose(
                candidate.target, candidate.targetSector,
                candidate.fromSector == observation.sector
                    ? llmapper::capability::playerFloorZ()
                    : candidate.target.fromFloorZ);
        // Once execution has selected an action pose, incidental visibility
        // from another side must not rewrite that pose while the route is in
        // flight.  Otherwise crossing an intermediate sector can splice the
        // old target payload to a new approach point and make a reachable
        // remembered task route away from itself.  Explicit surface recovery
        // updates the memory directly and remains available after a failed
        // probe.
        const bool activeSurfaceSearch = currentObjective.active
            && currentObjective.type == kObjectiveInteraction
            && currentObjective.interactionKey == key
            && memory.selectedSurfaceObjectiveTick == currentObjective.startedTick;
        const bool sideChanged = memory.observed
            && memory.fromSector != candidate.fromSector
            && !preserveReachableSpriteApproach
            && !preserveRemoteVectorApproach
            && !preserveBoardableTransportApproach
            && !activeSurfaceSearch;
        // Among several surfaces driving one mechanism, the useful one is the
        // one the player can reach and face right now.
        const bool differentWallSurface = memory.observed
            && candidate.kind == kInteractionWall
            && candidate.target.wallPush && memory.target.wallPush
            && candidate.target.wall != memory.target.wall;
        const bool preserveActiveApproach = activeSurfaceSearch
            && memory.observed
            && (memory.fromSector != candidate.fromSector
                || memory.x != candidate.x || memory.y != candidate.y
                || memory.target.wall != candidate.target.wall);
        const bool closerSurface = llmapper::shouldReselectInteractionSurface(
            activeSurfaceSearch, differentWallSurface,
            differentWallSurface
                && distance2(observation.x, observation.y,
                             candidate.x, candidate.y)
                    < distance2(observation.x, observation.y,
                                memory.x, memory.y));
        memory.targetSector = candidate.targetSector >= 0
            ? candidate.targetSector : candidateMechanism;
        // A reversible mechanism is usable from more than one side, and the
        // side that matters is the one the bot is standing on now.  Keeping
        // the first-seen approach point sent the bot at a pose it could no
        // longer reach once the door it came through closed behind it.
        if (!preserveReachableSpriteApproach && !preserveRemoteVectorApproach
            && !preserveBoardableTransportApproach
            && !preserveActiveApproach
            && (first || !canonicalSectorPush || sideChanged || closerSurface))
        {
            memory.fromSector = candidate.fromSector;
            memory.x = candidate.x;
            memory.y = candidate.y;
            memory.z = candidate.z;
            memory.approachFloorZ = candidate.fromSector == observation.sector
                ? llmapper::capability::playerFloorZ()
                : candidate.target.fromFloorZ;
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
        memory.observed = true;
        // Keep both the approach coordinates and the target payload owned by
        // the active executor. Updating only the former still reset wall 412
        // to whichever sibling face happened to be observed last each tick.
        if (!llmapper::preserveActiveInteractionSurface(
                activeSurfaceSearch, differentWallSurface)
            && !preserveBoardableTransportApproach
            && !preserveActiveApproach)
            memory.target = candidate.target;
        if (candidate.target.mechanismTx > 0)
        {
            memory.causalEffects = finalMechanismEffects(
                candidate.target.mechanismTx);
            if (first)
            {
                const std::vector<llmapper::CausalReceiver> chain =
                    mechanismCausalGraph().receiversReachableFrom(
                        candidate.target.mechanismTx, 8, false);
                std::string effects;
                for (const WorldObjectRef &effect : memory.causalEffects)
                {
                    char item[40];
                    snprintf(item, sizeof(item), "%s%d:%d",
                             effects.empty() ? "" : ",",
                             int(effect.kind), effect.id);
                    effects += item;
                }
                char detail[384];
                snprintf(detail, sizeof(detail),
                         "tx=%d actuator_kind=%d actuator=%d chain_nodes=%d final_effects=%d effects=%s",
                         candidate.target.mechanismTx, int(candidate.kind),
                         candidate.id, int(chain.size()),
                         int(memory.causalEffects.size()), effects.c_str());
                event("causal_chain_resolved", detail);
            }
        }
        const int receiver = mechanismSector(candidate.target,
                                             candidate.targetSector);
        if (receiver >= 0 && candidate.fromSector == receiver)
            memory.activationPoseOnReceiver = true;
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
            // The bot just changed the world on purpose.  Anything it had
            // given up on around this mechanism deserves another look: on
            // AGTST4 the frontier into the door's own sector went dormant
            // seconds before the door was opened, so when it did open there
            // was no live opportunity left to carry the bot through it.
            rearmAroundMechanism(memory);
        }
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
        std::set<int> drivenSectors;
        if (inRange(memory.targetSector, 0, numsectors))
            drivenSectors.insert(memory.targetSector);
        const int immediate = mechanismSector(memory.target, memory.targetSector);
        if (inRange(immediate, 0, numsectors))
            drivenSectors.insert(immediate);
        for (const WorldObjectRef &effect : memory.causalEffects)
            appendEffectSectors(effect, drivenSectors);
        for (int driven : drivenSectors)
        {
            const sectortype &record = sector[driven];
            for (int i = 0; i < record.wallnum; ++i)
            {
                const int wallId = record.wallptr + i;
                if (!inRange(wallId, 0, numwalls))
                    continue;
                const int neighbour = wall[wallId].nextsector;
                if (inRange(neighbour, 0, numsectors))
                {
                    clearTransitionAttempts(wallId);
                    if (inRange(wall[wallId].nextwall, 0, numwalls))
                        clearTransitionAttempts(wall[wallId].nextwall);
                }
            }
            navEdgeFailures.erase(
                std::remove_if(navEdgeFailures.begin(), navEdgeFailures.end(),
                               [&](const NavEdgeFailure &failure)
                               {
                                   return failure.wall >= 0
                                       && inRange(failure.wall, 0, numwalls)
                                       && (wallOwnerSector(failure.wall) == driven
                                           || wall[failure.wall].nextsector == driven);
                               }),
                navEdgeFailures.end());
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
            const int crouched = playerCollisionClearance();
            if (crouched > 0
                && (gObservedCrouchClearance == 0 || crouched < gObservedCrouchClearance))
            {
                const int standing = gStandingClearance;
                gObservedCrouchClearance = crouched;
                gObservedCrouchShape = livePlayerCollisionShape();
                char detail[128];
                snprintf(detail, sizeof(detail), "crouch_clearance=%d standing_clearance=%d",
                         crouched, standing);
                event("crouch_envelope_measured", detail);
            }
        }
        else if (gMe->posture == kPostureStand)
            gStandingClearance = playerCollisionClearance();
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
        // Crossing an arbitrary Build partition does not change the semantic
        // search state. Visibility gain below creates or consumes work when
        // the move actually reveals something.
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
            clearTransitionAttempts(crossedWall);
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
            event("discovered_sector");

        trackBoundaryStates();
        trackCombatOutcomes();
        trackEnvironmentalDamage();
        const int newlyVisibleCells = updateLocalCoverage();
        if (newlyVisibleCells > 0)
        {
            ++knowledgeRevision;
            lastSemanticProgressTick = observation.tick;
            explorationSnapshotEmitted = false;
            char detail[128];
            snprintf(detail, sizeof(detail),
                     "new_cells=%d known_cells=%u source=physical_visibility",
                     newlyVisibleCells, unsigned(observedCells.size()));
            event("unknown_space_observed", detail);
        }
        refreshCausalContinuations();

        for (const Portal &portal : observation.portals)
        {
            const int edgeId = portal.wall * 65536 + portal.to;
            const bool newEdge = seenEdges.insert(edgeId).second;
            if (newEdge && !visitedSectors.count(portal.to))
            {
                recoveryMode = false;
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
            // A visible two-sided doorway has the same physical opening in
            // both directions. Ballistic support transitions are represented
            // separately by concrete directed pose edges.
            const bool supportsObservedReverse =
                portal.traversable || portal.jumpable;
            if (supportsObservedReverse)
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
                if (inRange(portal.wall, 0, numwalls)
                    && inRange(wall[portal.wall].nextwall, 0, numwalls))
                {
                    const int paired = wall[portal.wall].nextwall;
                    if (wall[paired].nextsector == reverse.to)
                        reverseWallId = paired;
                }
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
                // Overlapping/ROR-style physical connections need not have
                // an opposite Build wall record.  That does not make a
                // physically reversible drop disappear: retain the observed
                // boundary identity and classify the reverse from its real
                // takeoff/landing heights.  The source wall remains engine
                // metadata for collision; no reverse interaction affordance
                // is inferred without a real reverse XWALL.
                reverse.wall = reverseWallId >= 0
                    ? reverseWallId : portal.wall;
                // Reclassify the opposite direction from the real opposite
                // wall/sector records.  A drop in one direction can be a
                // jump or a blocked rise in the other; copying the original
                // mode would corrupt NavTopology and walkArea.
                const walltype &reverseWall = wall[reverse.wall];
                const bool reverseEnoughWidth = reverse.openingWidth >= playerPassageWidth();
                const bool reverseStanding = reverse.clearance >= playerStandingClearance();
                const bool reverseCrouching = reverse.clearance >= playerCrouchClearance();
                const int reverseRise = playerJumpRiseLimit();
                const bool reverseWalk = reverseEnoughWidth && reverseStanding
                    && std::abs(reverse.floorDelta) <= llmapper::capability::playerStepHeight();
                const bool reverseJump = reverseEnoughWidth && reverseStanding
                    && reverse.floorDelta < 0 && -reverse.floorDelta <= reverseRise;
                const bool reverseDrop = reverseEnoughWidth && reverseStanding
                    && reverse.floorDelta > 0 && reverse.floorDelta <= reverseRise;
                reverse.walkable = !(reverseWall.cstat & 1) && reverseWalk;
                reverse.crouchable = !(reverseWall.cstat & 1) && !reverseStanding
                    && reverseCrouching && std::abs(reverse.floorDelta) <= llmapper::capability::playerStepHeight();
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
                if (reverseWallId >= 0
                    && reverseWall.extra > 0 && reverseWall.extra < kMaxXWalls)
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
                char detail[384];
                const char *category = itemCategory(object.type);
                const spritetype &record = sprite[object.sprite];
                int top = record.z, bottom = record.z;
                GetSpriteExtents(&record, &top, &bottom);
                const bool operable = validXSprite(record.extra)
                    && (xsprite[record.extra].Push || xsprite[record.extra].Vector);
                const int pickupUseful = object.kind == kObjectPickup
                    || object.kind == kObjectKey
                    ? playerCanBenefitFromPickup(gMe, &sprite[object.sprite]) : -1;
                snprintf(detail, sizeof(detail),
                         "sprite=%d kind=%s category=%s sector=%d type=%d pos=(%d,%d,%d) cstat=%d alignment=%d block=%d clipdist=%d top=%d bottom=%d push=%d vector=%d pickup_useful=%d",
                         object.sprite, objectName(object.kind), category ? category : "none",
                         object.sector, object.type, int(record.x), int(record.y),
                         int(record.z), int(record.cstat),
                         int(record.cstat) & CSTAT_SPRITE_ALIGNMENT_MASK,
                         (record.cstat & CSTAT_SPRITE_BLOCK) ? 1 : 0,
                         record.clipdist << 2, top, bottom,
                         operable && xsprite[record.extra].Push ? 1 : 0,
                         operable && xsprite[record.extra].Vector ? 1 : 0,
                         pickupUseful);
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
            int rearmed = 0;
            if (rangedCapability)
            {
                for (const auto &entry : interactions)
                {
                    const InteractionMemory &memory = entry.second;
                    if (!memory.observed
                        || memory.activationMode != llmapper::kActivateVector)
                        continue;
                    ++rearmed;
                }
            }
            char detail[96];
            snprintf(detail, sizeof(detail),
                     "available=%d weapon=%d rearmed=%d",
                     rangedCapability, rangedWeapon, rearmed);
            event("ranged_capability_changed", detail);
        }

        // Re-derive missing prerequisites from remembered useful actions.
        // This is the deferred-opportunity link: it persists independently
        // of whichever pickup or world mechanism may eventually satisfy it.
        const unsigned effectCapabilities = availableEffectCapabilities();
        unsigned missingEffects = llmapper::kEffectNone;
        for (auto &entry : interactions)
        {
            InteractionMemory &memory = entry.second;
            if (!memory.observed || memory.state == 2)
                continue;
            if (!llmapper::effectRequirementSatisfied(
                    memory.requiredEffects, effectCapabilities))
                missingEffects |= memory.requiredEffects;
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

        // Interaction memory is refreshed near the end of observation. Run
        // the causal reachability delta here as well so newly enabled space
        // exists in the ledger before the next objective can be selected.
        refreshCausalContinuations();
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

    void clearTransitionAttempts(int wallId)
    {
        navEdgeFailures.erase(
            std::remove_if(navEdgeFailures.begin(), navEdgeFailures.end(),
                           [wallId](const NavEdgeFailure &failure)
                           {
                               return failure.wall == wallId;
                           }),
            navEdgeFailures.end());
    }

    void pruneNavEdgeFailures()
    {
        for (auto it = navEdgeFailures.begin(); it != navEdgeFailures.end(); )
        {
            if (it->wall < 0)
            {
                ++it;
                continue;
            }
            const Portal *portal = portalByWallId(it->wall);
            if (portal
                && portalGeometrySignature(*portal) != it->geometrySignature)
                it = navEdgeFailures.erase(it);
            else
                ++it;
        }
    }

    bool edgeFailed(int edgeId) const
    {
        const Portal *current = findPortalForEdge(edgeId);
        if (!current)
            return false;
        const int revision = portalGeometrySignature(*current);
        for (const NavEdgeFailure &attempt : navEdgeFailures)
            if (attempt.wall == current->wall
                && attempt.geometrySignature == revision)
                return true;
        return false;
    }

    bool recordEdgeFailure(const Portal &portal, const char *eventName,
                           const char *reason)
    {
        // A moving boundary is different relevant evidence, not a
        // deterministic failure of the settled transition.
        if (portal.wallBusy || portal.sectorBusy || doorTiming(portal.to).moving)
        {
            char pending[192];
            snprintf(pending, sizeof(pending),
                     "wall=%d from=%d to=%d reason=%s evidence=geometry_in_motion",
                     portal.wall, portal.from, portal.to, reason);
            event("transition_attempt_withheld", pending);
            return false;
        }

        const int revision = portalGeometrySignature(portal);
        NavEdgeFailure *record = nullptr;
        for (NavEdgeFailure &attempt : navEdgeFailures)
            if (attempt.wall == portal.wall
                && attempt.geometrySignature == revision)
            {
                record = &attempt;
                break;
            }
        if (!record)
        {
            NavEdgeFailure attempt;
            attempt.wall = portal.wall;
            attempt.mode = kNavBlocked; // observed-boundary mode wildcard
            attempt.geometrySignature = revision;
            navEdgeFailures.push_back(attempt);
            record = &navEdgeFailures.back();
        }
        ++record->attempts;
        char detail[192];
        snprintf(detail, sizeof(detail),
                 "wall=%d from=%d to=%d reason=%s attempts=%d revision=%d",
                 portal.wall, portal.from, portal.to, reason,
                 record->attempts, revision);
        event(eventName, detail);
        return true;
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
                && std::abs(portal.floorDelta) <= llmapper::capability::playerStepHeight();
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
    int updateLocalCoverage()
    {
        ensureNavTopology();
        int newlyObserved = 0;
        for (const NavCell &cell : navCells)
        {
            const PhysicalPoseKey handle = physicalPoseKey(cell);
            if (observedCells.count(handle))
                continue;
            // The NavCell is a concrete physical support pose.  Its Z may be
            // a sprite bridge, voxel top, ROR layer, or moving support and is
            // deliberately independent of the containing sector floor.  The
            // observation query must test the same pose the planner proved;
            // substituting the sector floor made one physical fact change
            // meaning between planning and perception.
            const int supportZ = cell.z;
            if (!cansee(observation.x, observation.y, observation.z, observation.sector,
                        cell.center.x, cell.center.y,
                        playerOriginAtSupport(supportZ), cell.sector))
                continue;
            observedCells.insert(handle);
            ++newlyObserved;
        }
        return newlyObserved;
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

    // Coarse per-sector distance cache derived from the authoritative
    // physical pose graph.  This is only a ranking/telemetry hint: callers
    // must still prove that their concrete approach cell is reachable.
    // Taking the minimum over cells deliberately does not make every other
    // support layer in that Build sector reachable.
    void computeHops(std::vector<int> &hops) const
    {
        hops.assign(size_t(std::max(int(numsectors), 1)), -1);
        if (!inRange(observation.sector, 0, numsectors) || navCells.empty())
            return;

        const int start = nearestNavCell(
            observation.sector, observation.x, observation.y,
            llmapper::capability::playerFloorZ());
        if (!inRange(start, 0, int(navCells.size())))
            return;

        std::set<std::pair<int, int> > queue;
        std::vector<int> distance(navCells.size(), INT32_MAX);
        distance[size_t(start)] = 0;
        queue.insert(std::make_pair(0, start));
        while (!queue.empty())
        {
            const std::pair<int, int> next = *queue.begin();
            queue.erase(queue.begin());
            const int current = next.second;
            if (next.first != distance[size_t(current)])
                continue;
            const NavCell &from = navCells[size_t(current)];
            for (const NavLink &link : from.links)
            {
                if (!llmapper::traversableMode(link.mode)
                    || link.condition.enabled
                    || !inRange(link.target, 0, int(navCells.size()))
                    || llmapper::edgeFailedAny(
                        navEdgeFailures, current, link.target, link.wall,
                        link.mode, 0))
                    continue;
                const NavCell &to = navCells[size_t(link.target)];
                const int64_t dx = int64_t(to.center.x) - from.center.x;
                const int64_t dy = int64_t(to.center.y) - from.center.y;
                const int span = std::max(
                    1, int(std::sqrt(double(dx * dx + dy * dy))));
                const int candidate = distance[size_t(current)] + span;
                if (candidate >= distance[size_t(link.target)])
                    continue;
                if (distance[size_t(link.target)] != INT32_MAX)
                    queue.erase(std::make_pair(
                        distance[size_t(link.target)], link.target));
                distance[size_t(link.target)] = candidate;
                queue.insert(std::make_pair(candidate, link.target));
            }
        }

        for (const NavCell &cell : navCells)
        {
            if (!inRange(cell.id, 0, int(distance.size()))
                || distance[size_t(cell.id)] == INT32_MAX
                || !inRange(cell.sector, 0, int(hops.size())))
                continue;
            const int hint = distance[size_t(cell.id)] / 1024;
            int &known = hops[size_t(cell.sector)];
            if (known < 0 || hint < known)
                known = hint;
        }
    }

    llmapper::WorkId boundaryWorkId(int wallId, int from, int to) const
    {
        return llmapper::WorkId(llmapper::kWorkBoundary, wallId, from, to);
    }

    // Rebuild the ledger of unresolved work from persistent world knowledge.
    // Reachability is only a current-state annotation.  A closed door, moved
    // lift, or temporary component split must never erase remembered work;
    // entries with hops == -1 remain conserved and are reconsidered after a
    // later topology/world-state change reconnects them.
    // Count directed route steps for which the physical graph has no
    // separately validated reverse edge.  Reversibility is not inferred
    // from sector adjacency or from the forward transition: if the graph did
    // not prove B->A, consuming A->B loses an option.
    int routeOneWayRisk(int startCell, int targetCell) const
    {
        if (!inRange(startCell, 0, int(navCells.size()))
            || !inRange(targetCell, 0, int(navCells.size()))
            || startCell == targetCell)
            return 0;
        std::vector<NavRouteStep> route;
        if (!llmapper::planNavRoute(navCells, startCell, targetCell,
                                    navEdgeFailures, 0, route))
            return 0;
        int risk = 0;
        for (const NavRouteStep &step : route)
        {
            if (!inRange(step.toCell, 0, int(navCells.size())))
                continue;
            const NavCell &destination = navCells[size_t(step.toCell)];
            const bool reverseProven = std::any_of(
                destination.links.begin(), destination.links.end(),
                [&](const NavLink &link)
                {
                    return link.target == step.fromCell
                        && llmapper::traversableMode(link.mode)
                        && !llmapper::edgeFailedAny(
                            navEdgeFailures, step.toCell, step.fromCell,
                            link.wall, link.mode, 0);
                });
            if (!reverseProven)
                ++risk;
        }
        return risk;
    }

    void rebuildLedger()
    {
        ledger.clear();
        ensureNavTopology();
        pruneNavEdgeFailures();
        const int ledgerStartCell = currentNavStartCell();
        std::vector<char> reachableNavCells;
        llmapper::markReachableNavCells(navCells, ledgerStartCell,
                                        navEdgeFailures, 0,
                                        reachableNavCells);
        std::vector<int> hops;
        computeHops(hops);
        std::set<int> sectorsWithUnobservedCells;
        for (const NavCell &cell : navCells)
        {
            if (!observedCells.count(physicalPoseKey(cell)))
                sectorsWithUnobservedCells.insert(cell.sector);
        }

        for (const auto &entry : knownGraph)
        {
            const int from = entry.first;
            if ((!visitedSectors.count(from) && !observedSectors.count(from))
                || !inRange(from, 0, int(hops.size())))
                continue;
            const int reach = hops[size_t(from)];
            for (const Portal &portal : entry.second)
            {
                if (portal.to < 0)
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
                opportunity.depth = 0;
                opportunity.hops = reach;
                opportunity.descent = std::max(0, portal.floorDelta);
                opportunity.local = false;
                // A Build sector is a container, not a connected navigation
                // state.  ROR geometry, slopes and stacked supports can put
                // the player in one physical component while a boundary of
                // the same sector belongs to another.  Do not manufacture a
                // local frontier from transient sector membership; conserve
                // it as temporarily unreachable until topology reconnects
                // the actual approach support.
                int approachCell = navCellInSector(
                    portal.from, portal.x, portal.y, portal.fromFloorZ);
                // A boundary is approached from whatever collision support
                // actually reaches its near side, not necessarily from the
                // enclosing sector floor sampled at its midpoint. Sprite
                // bridges and solid objects commonly provide that support.
                if (!(approachCell >= 0
                      && approachCell < int(reachableNavCells.size())
                      && reachableNavCells[size_t(approachCell)]))
                {
                    int64_t bestBoundaryDistance = INT64_MAX;
                    for (const NavCell &cell : navCells)
                    {
                        if (cell.sector != portal.from
                            || !inRange(cell.id, 0, int(reachableNavCells.size()))
                            || !reachableNavCells[size_t(cell.id)])
                            continue;
                        const int64_t boundaryDistance = dist2ToSegment(
                            cell.center.x, cell.center.y,
                            portal.x1, portal.y1, portal.x2, portal.y2);
                        if (boundaryDistance
                                > int64_t(kActionApproachRange)
                                    * kActionApproachRange
                            || boundaryDistance >= bestBoundaryDistance)
                            continue;
                        bestBoundaryDistance = boundaryDistance;
                        approachCell = cell.id;
                    }
                }
                const int64_t approachDistance = ledgerStartCell >= 0
                    && approachCell >= 0
                    ? distance2(navCells[size_t(ledgerStartCell)].center.x,
                                navCells[size_t(ledgerStartCell)].center.y,
                                navCells[size_t(approachCell)].center.x,
                                navCells[size_t(approachCell)].center.y)
                    : 0;
                opportunity.local = approachCell >= 0
                    && approachCell < int(reachableNavCells.size())
                    && reachableNavCells[size_t(approachCell)];
                if (opportunity.local)
                    opportunity.hops = std::max(0,
                        int(std::sqrt(double(approachDistance))) / 1024);
                if (ledgerStartCell >= 0 && approachCell >= 0
                    && approachCell < int(reachableNavCells.size())
                    && !reachableNavCells[size_t(approachCell)]
                    // Very small/transit sectors often have no centred grid
                    // square connecting their two portal anchors.  A nearby
                    // boundary is still a bounded direct collision probe;
                    // the false-local failure is a remote boundary elsewhere
                    // in the same stacked/ROR container.
                    && approachDistance
                        > int64_t(kRouteSmoothingSpan) * kRouteSmoothingSpan)
                {
                    opportunity.hops = -1;
                    if (opportunity.local)
                    {
                        char detail[192];
                        snprintf(detail, sizeof(detail),
                                 "wall=%d sector=%d start_cell=%d approach_cell=%d start_area=%d approach_area=%d distance=%d reason=disconnected_support_component",
                                 portal.wall, portal.from, ledgerStartCell,
                                 approachCell,
                                 navCells[size_t(ledgerStartCell)].walkArea,
                                 navCells[size_t(approachCell)].walkArea,
                                 int(std::sqrt(double(approachDistance))));
                        event("frontier_temporarily_unreachable", detail);
                    }
                    opportunity.local = false;
                }
                // Portal geometry is observation evidence, not a physical
                // route.  A floor delta can look jumpable while the player's
                // cylinder has no concrete takeoff/landing sequence there
                // (or while the real route first climbs sprite supports).
                // Only the authoritative pose graph may promote the
                // remembered boundary to immediately executable work.
                bool physicalFarSideReachable = false;
                for (const NavCell &cell : navCells)
                {
                    if (cell.sector != portal.to
                        || !inRange(cell.id, 0,
                                   int(reachableNavCells.size()))
                        || !reachableNavCells[size_t(cell.id)])
                        continue;
                    const int64_t boundaryDistance = dist2ToSegment(
                        cell.center.x, cell.center.y,
                        portal.x1, portal.y1, portal.x2, portal.y2);
                    if (boundaryDistance
                        <= int64_t(kActionApproachRange)
                            * kActionApproachRange)
                    {
                        physicalFarSideReachable = true;
                        break;
                    }
                }
                const bool crossable = (portal.traversable || portal.jumpable)
                    && physicalFarSideReachable && !edgeFailed(edgeId);
                // A downward transition taller than Caleb can jump back up
                // destroys the current component's optionality.  Keep it as
                // a valid hypothesis, but let every known reversible action
                // run first.  Geometry supplies this confidence directly;
                // an unseen destination by itself is not considered risky.
                // Preserve optionality even while the exact traversal is
                // still only a hypothesis.  A large downward boundary is a
                // one-way commitment whether collision already calls it a
                // drop or the executor would have to discover intermediate
                // support.  Restricting this annotation to `crossable`
                // preferred speculative irreversible probes over nearby
                // reversible frontiers and could strand the exploration in
                // the first dead end it inspected.
                if (portal.floorDelta > playerReversibleDrop())
                    opportunity.oneWayRisk = portal.floorDelta;
                if (transitHypothesis)
                {
                    // A thin Build partition is transport geometry, not an
                    // exploration hypothesis. Any occluded space reached
                    // through it is represented by visibility gain.
                    continue;
                }
                else if (portal.key && !hasKey(portal.key))
                {
                    opportunity.kind = llmapper::kOpportunityLocked;
                    opportunity.id = boundaryWorkId(
                        portal.wall, portal.from, portal.to);
                    lockedDoorKey[portal.wall] = portal.key;
                }
                else if (crossable)
                {
                    if (visitedSectors.count(portal.to)
                        || !sectorsWithUnobservedCells.count(portal.to))
                    {
                        // Traversal is a way to reveal physical work, not a
                        // sector-visit objective. If visibility has already
                        // exhausted the far side, its pickups, interactions,
                        // exits and causal continuations have their own ledger
                        // entries and this edge is only transportation.
                        continue;
                    }
                    // Seeing all sampled floor cells on the far side is not
                    // the same as inspecting its outgoing connectivity and
                    // affordances.  Until the endpoint has been traversed,
                    // conserve the boundary as unresolved work. Thin Build
                    // partitions were filtered above as transit geometry.
                    opportunity.kind = llmapper::kOpportunityFrontier;
                    opportunity.id = boundaryWorkId(
                        portal.wall, portal.from, portal.to);
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
                    if (visitedSectors.count(portal.to))
                    {
                        // A closed edge into space already entered is
                        // remembered connectivity, not unfinished
                        // exploration.  ReopenRouteGateFor() may activate it
                        // later when some real unresolved work depends on
                        // that transport edge.  Offering it on its own makes
                        // the bot enumerate door/elevator states and wander
                        // back into exhausted regions.
                        continue;
                    }
                    if (llmapper::investigatedNow(investigatedBoundaries, portal.wall,
                                                  portal.from, portal.to,
                                                  portalIdentitySignature(portal)))
                        continue;
                    // Keyed by where it leads, not by which of its walls the
                    // bot happens to be looking at.
                    opportunity.kind = llmapper::kOpportunityBlocked;
                    opportunity.id = llmapper::WorkId(
                        llmapper::kWorkBoundary, portal.wall,
                        portal.from, portal.to);
                    bool alreadyOffered = false;
                    for (const llmapper::Opportunity &known : ledger)
                        if (known.id == opportunity.id)
                            alreadyOffered = true;
                    if (alreadyOffered)
                        continue;
                }
                else
                {
                    if (!sectorsWithUnobservedCells.count(portal.to))
                    {
                        // Solid geometry with no unknown physical space
                        // behind it is exhausted scenery.
                        noteInertBoundary(portal);
                        continue;
                    }
                    // A high ledge or support chain can make an otherwise
                    // open two-sided Build boundary look non-traversable from
                    // the floor.  It remains unresolved work, but the
                    // boundary itself is not an executable route until the
                    // support graph connects its far side.  Previously this
                    // branch labelled every such wall immediately local and
                    // sent the executor into collision; an engine-confirmed
                    // failed edge was therefore retried as a fresh frontier.
                    opportunity.kind = llmapper::kOpportunityFrontier;
                    opportunity.id = boundaryWorkId(
                        portal.wall, portal.from, portal.to);
                    opportunity.hops = -1;
                    opportunity.local = false;
                }
                opportunity.continuation = std::any_of(
                    causalContinuationCells.begin(),
                    causalContinuationCells.end(),
                    [&portal](const std::pair<const PhysicalPoseKey, int> &entry)
                    {
                        return std::get<0>(entry.first) == portal.to;
                    });
                if (opportunity.kind == llmapper::kOpportunityBlocked
                    || opportunity.kind == llmapper::kOpportunityLocked)
                {
                    opportunity.ready = opportunity.local
                        && approachDistance
                            <= int64_t(kActionApproachRange * 2)
                                * (kActionApproachRange * 2);
                }
                ledger.push_back(opportunity);
            }
        }

        for (auto &entry : interactions)
        {
            InteractionMemory &memory = entry.second;
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
            int opportunitySector = -1;
            int approachCell = -1;
            if (requiredSupport >= 0)
            {
                // A sector number is not proof of standing on its moving
                // floor: overlapping layers and sprite tops can share both
                // sector and XY. The prerequisite is the exact collision
                // support, so rank only a currently reachable cell owned by
                // that support.
                const SupportRef support(kSupportSectorFloor, requiredSupport);
                int bestDistance = INT32_MAX;
                for (const NavCell &cell : navCells)
                {
                    if (cell.sector != requiredSupport
                        || cell.support != support
                        || !inRange(cell.id, 0, int(reachableNavCells.size()))
                        || !reachableNavCells[size_t(cell.id)])
                        continue;
                    const int distance = distance2(
                        cell.center.x, cell.center.y, memory.x, memory.y);
                    if (distance >= bestDistance)
                        continue;
                    bestDistance = distance;
                    approachCell = cell.id;
                }
                if (approachCell >= 0)
                    opportunitySector = navCells[size_t(approachCell)].sector;
            }
            else if (memory.activationMode == llmapper::kActivateUse)
            {
                const int poseSignature = interactionStateSignature(memory);
                const int poseStartCell = currentNavStartCell();
                const int poseStartArea = inRange(
                    poseStartCell, 0, int(navCells.size()))
                    ? navCells[size_t(poseStartCell)].walkArea : -1;
                const SupportRef poseStartSupport = inRange(
                    poseStartCell, 0, int(navCells.size()))
                    ? navCells[size_t(poseStartCell)].support : SupportRef();
                const bool cachedPoseReachable = inRange(
                        memory.actionPoseCell, 0, int(reachableNavCells.size()))
                    && reachableNavCells[size_t(memory.actionPoseCell)];
                if (memory.actionPoseTopologyRevision != navTopologyRevision
                    || memory.actionPoseSignature != poseSignature
                    || memory.actionPoseStartArea != poseStartArea
                    || memory.actionPoseStartSupport != poseStartSupport
                    || (memory.actionPoseCell >= 0 && !cachedPoseReachable))
                {
                    reachableInteractionPose(memory, &reachableNavCells);
                }
                approachCell = memory.actionPoseCell;
                if (approachCell >= 0)
                    opportunitySector = navCells[size_t(approachCell)].sector;
            }
            else
            {
                // Aim Z belongs to the target ray; it is never a player-floor
                // coordinate. Project ranged/damage work through a real,
                // reachable standable pose whose weapon origin can see the
                // target. The old target-Z lookup made a floor-level blocker
                // permanently unreachable whenever its sprite centre was
                // above Caleb's feet.
                const EffectDelivery delivery = effectDeliveryFor(memory);
                const int targetSector = memory.kind == kInteractionSprite
                    && memory.targetSector >= 0
                    ? memory.targetSector
                    : memory.fromSector;
                const int shotZ = gMe ? gMe->zWeapon : observation.z;
                if (delivery.available)
                    approachCell = reachableFiringPose(
                        memory.x, memory.y, memory.z, targetSector, shotZ,
                        delivery.minimumDistance,
                        delivery.maximumDistance);
                if (approachCell >= 0)
                    opportunitySector = navCells[size_t(approachCell)].sector;
            }
            bool physicallyReachable = approachCell >= 0
                && approachCell < int(reachableNavCells.size())
                && reachableNavCells[size_t(approachCell)];
            const bool occupiesRequiredSupport = requiredSupport < 0
                || currentPlayerSupport()
                    == SupportRef(kSupportSectorFloor, requiredSupport);
            const bool encounteredHere = physicallyReachable
                && approachCell == ledgerStartCell;
            const bool currentRegion = physicallyReachable || encounteredHere;
            llmapper::Opportunity opportunity;
            opportunity.kind = memory.key && !hasKey(memory.key)
                ? llmapper::kOpportunityLocked : llmapper::kOpportunityInteraction;
            opportunity.id = interactionWorkId(
                interactionMemoryKey(memory));
            opportunity.sector = opportunitySector;
            opportunity.target = memory.targetSector;
            opportunity.wall = memory.target.wall;
            opportunity.requiredKey = memory.key;
            opportunity.requiredEffects = memory.requiredEffects;
            opportunity.depth = 0;
            opportunity.approach = approachCell;
            opportunity.hops = encounteredHere ? 0
                : currentRegion && memory.activationMode == llmapper::kActivateUse
                    ? memory.actionPoseHops
                : currentRegion ? 1 : -1;
            const bool usefulSupportTransition = supportTransitionUseful(memory);
            const bool maintenanceReactivation = memory.attempted
                && interactionNeedsReactivation(memory)
                && !usefulSupportTransition;
            const bool carrierStateTransition = memory.attempted
                && interactionNeedsReactivation(memory)
                && usefulSupportTransition;
            const bool transportBoardable =
                supportTransportBoardableAtActionPose(memory);
            opportunity.local = (currentRegion || opportunity.hops == 0)
                && !maintenanceReactivation
                && transportBoardable;
            // "Ready" is a task-state fact, not an ActionScan proximity
            // test.  Once all semantic prerequisites are satisfied and the
            // remembered action pose is in the currently reachable physical
            // component, approaching it is execution of known work.  Treating
            // every distant switch as deferred made exploration outrank an
            // exit the bot had already seen and could simply walk toward.
            opportunity.ready = currentRegion && transportBoardable;
            // Moving the support under Caleb to another stable endpoint is
            // reversible in the mechanism, but it temporarily gives up all
            // connectivity available at the current endpoint. Preserve that
            // option until ordinary safe work has been consumed. The action
            // remains selectable when it is the only remaining work, so a
            // fall or backtrack can still reuse the carrier.
            if (carrierStateTransition)
                opportunity.oneWayRisk = 1;
            if (currentRegion && approachCell != ledgerStartCell)
                opportunity.oneWayRisk = std::max(
                    opportunity.oneWayRisk,
                    routeOneWayRisk(ledgerStartCell, approachCell));
            // Occupying a carrier is an established physical prerequisite,
            // not merely proximity to another affordance. Consume the
            // aboard-only action before a nearby reusable call control can
            // move the carrier away again. This is the same causal
            // continuation rule used for newly opened space: once a
            // prerequisite has been established, finish the work it enabled.
            opportunity.continuation = requiredSupport >= 0
                && occupiesRequiredSupport
                && !memory.attempted;
            if (requiredSupport >= 0 && !physicallyReachable
                && !encounteredHere)
            {
                // Known useful work, blocked on reaching its exact support.
                // Keep it deferred until topology/support evidence changes;
                // do not let a sector-level route estimate masquerade as an
                // immediately executable task and starve discovery.
                opportunity.hops = -1;
                opportunity.local = false;
                opportunity.continuation = false;
            }
            if (!transportBoardable)
            {
                // The affordance is real and remains in the ledger.  Its
                // current pose simply cannot consume the resulting carrier
                // state, so a later support/topology observation must make
                // it actionable instead of this remote Use counting as work.
                opportunity.hops = -1;
                opportunity.local = false;
            }
            if (rangedPrerequisitePending
                && memory.activationMode == llmapper::kActivateVector
                && requiredSupport >= 0)
            {
                // The remembered action remains in the ledger, but its
                // learned capability prerequisite is currently unsatisfied.
                // Do not retry the actuator while the prerequisite source is
                // the active causal subgoal.
                opportunity.hops = -1;
                opportunity.local = false;
            }
            ledger.push_back(opportunity);
        }

        std::map<int, int> visibleExitCells;
        for (const NavCell &cell : navCells)
        {
            if (!inRange(cell.sector, 0, numsectors)
                || !observedCells.count(physicalPoseKey(cell)))
                continue;
            const int extra = sector[cell.sector].extra;
            if (extra <= 0 || extra >= kMaxXSectors || !xsector[extra].Exit)
                continue;
            auto existing = visibleExitCells.find(cell.sector);
            if (existing == visibleExitCells.end()
                || distance2(observation.x, observation.y,
                             cell.center.x, cell.center.y)
                    < distance2(observation.x, observation.y,
                                navCells[size_t(existing->second)].center.x,
                                navCells[size_t(existing->second)].center.y))
                visibleExitCells[cell.sector] = cell.id;
        }
        for (const auto &entry : visibleExitCells)
        {
            const NavCell &cell = navCells[size_t(entry.second)];
            llmapper::Opportunity opportunity;
            opportunity.kind = llmapper::kOpportunityExit;
            opportunity.id = llmapper::WorkId(llmapper::kWorkExit,
                                               cell.sector);
            opportunity.sector = cell.sector;
            opportunity.target = cell.id;
            opportunity.depth = 0;
            const bool currentRegion = inRange(cell.id, 0,
                int(reachableNavCells.size()))
                && reachableNavCells[size_t(cell.id)];
            const int distance = int(std::sqrt(double(distance2(
                observation.x, observation.y, cell.center.x, cell.center.y))));
            opportunity.hops = currentRegion ? std::max(0, distance / 1024)
                : -1;
            opportunity.local = currentRegion;
            opportunity.ready = currentRegion;
            ledger.push_back(opportunity);
        }

        const std::vector<llmapper::VisibilityFrontier> visibilityFrontiers =
            semanticVisibilityFrontiers(reachableNavCells);
        for (const llmapper::VisibilityFrontier &frontier : visibilityFrontiers)
        {
            if (!inRange(frontier.cell, 0, int(navCells.size())))
                continue;
            const NavCell &cell = navCells[size_t(frontier.cell)];
            if (movingSectorHazard(cell.sector))
                continue;
            llmapper::Opportunity opportunity;
            opportunity.kind = llmapper::kOpportunityCoverage;
            opportunity.id = llmapper::WorkId(llmapper::kWorkPose,
                                               frontier.cell);
            opportunity.sector = cell.sector;
            opportunity.target = frontier.cell;
            opportunity.depth = 0;
            const bool targetReachable = inRange(frontier.cell, 0,
                int(reachableNavCells.size()))
                && reachableNavCells[size_t(frontier.cell)];
            int approachCell = targetReachable ? frontier.cell
                : frontier.approachCell;
            int64_t bestApproachDistance = INT64_MAX;
            if (!targetReachable)
            {
                // An approach is an observation pose, not merely a nearby
                // reachable square.  Prove with the real visibility query
                // that standing there can expose the unknown physical pose.
                // This prevents a disconnected pit cell from borrowing a
                // platform-side XY and turning that boundary point into an
                // invented traversal goal.
                const bool retainedApproachValid = inRange(
                        approachCell, 0, int(navCells.size()))
                    && inRange(approachCell, 0,
                               int(reachableNavCells.size()))
                    && reachableNavCells[size_t(approachCell)]
                    && cansee(
                        navCells[size_t(approachCell)].center.x,
                        navCells[size_t(approachCell)].center.y,
                        playerOriginAtSupport(
                            navCells[size_t(approachCell)].z),
                        navCells[size_t(approachCell)].sector,
                        cell.center.x, cell.center.y,
                        playerOriginAtSupport(cell.z), cell.sector);
                // semanticVisibilityFrontiers already performed this exact
                // physical proof and selected the observation pose.  Keep
                // that concrete result instead of running a second search
                // with different (nearest-only) ranking and silently
                // replacing it. Re-search only if topology changed between
                // the two queries.
                if (!retainedApproachValid)
                    approachCell = -1;
                for (const NavCell &candidate : navCells)
                {
                    if (retainedApproachValid)
                        break;
                    if (!inRange(candidate.id, 0,
                                 int(reachableNavCells.size()))
                        || !reachableNavCells[size_t(candidate.id)]
                        || !cansee(candidate.center.x, candidate.center.y,
                                   playerOriginAtSupport(candidate.z),
                                   candidate.sector,
                                   cell.center.x, cell.center.y,
                                   playerOriginAtSupport(cell.z), cell.sector))
                        continue;
                    const int64_t distance = distance2(
                        observation.x, observation.y,
                        candidate.center.x, candidate.center.y);
                    if (distance < bestApproachDistance)
                    {
                        bestApproachDistance = distance;
                        approachCell = candidate.id;
                    }
                }
            }
            opportunity.approach = approachCell;
            const bool approachReachable = inRange(approachCell, 0,
                int(reachableNavCells.size()))
                && reachableNavCells[size_t(approachCell)];
            const NavCell &workCell = !frontier.reachable && approachReachable
                ? navCells[size_t(approachCell)] : cell;
            const int distance = int(std::sqrt(double(distance2(
                observation.x, observation.y,
                workCell.center.x, workCell.center.y))));
            opportunity.hops = frontier.reachable || approachReachable
                ? std::max(0, distance / 1024) : -1;
            opportunity.local = frontier.reachable || approachReachable;
            opportunity.continuation = causalContinuationCells.count(
                physicalPoseKey(cell)) != 0;
            opportunity.requiresOccupancy = frontier.requiresOccupancy;
            ledger.push_back(opportunity);
        }

        for (const auto &entry : objectMemory)
        {
            const ObjectMemory &memory = entry.second;
            if (!memory.observed || memory.collected || !objectStillPresent(memory))
                continue;
            if (memory.object.kind != kObjectPickup && memory.object.kind != kObjectKey)
                continue;
            // Presence is not availability.  Full health/ammo/armour, a full
            // pack, an already-held key, or an unusable duplicate weapon
            // remains remembered but is not actionable until player state
            // makes the engine pickup beneficial again.
            if (!pickupUsefulNow(memory.object))
                continue;
            if (!inRange(memory.object.sector, 0, int(hops.size())))
                continue;
            const int objectCell = navCellInSector(
                memory.object.sector, memory.object.x, memory.object.y,
                memory.object.z);
            const bool currentRegion = objectCell >= 0
                && objectCell < int(reachableNavCells.size())
                && reachableNavCells[size_t(objectCell)];
            llmapper::Opportunity opportunity;
            opportunity.kind = llmapper::kOpportunityPickup;
            opportunity.id = llmapper::WorkId(llmapper::kWorkObject,
                                               memory.object.sprite);
            opportunity.sector = memory.object.sector;
            opportunity.depth = 0;
            opportunity.hops = currentRegion
                ? std::max(0, int(std::sqrt(double(distance2(
                    observation.x, observation.y,
                    memory.object.x, memory.object.y)))) / 1024)
                : -1;
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
            bool physicallyOnTheWay = currentRegion;
            if (physicallyOnTheWay && memory.object.kind == kObjectPickup)
            {
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
                            <= llmapper::capability::playerStepHeight());
            }
            opportunity.local = physicallyOnTheWay;
            opportunity.ready = physicallyOnTheWay;
            opportunity.continuation = suppliesRangedPrerequisite
                || suppliesEffectPrerequisite;
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
            // Static causal reachability says an actuator may affect some
            // state in the destination sector; it does not prove that it
            // opens this exact boundary. Keep blocker selection tied to the
            // physical target until an observed transition establishes a
            // narrower relationship.
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

    bool hazardousSupport(const SupportRef &support) const
    {
        auto known = hazardSupports.find(support);
        return known != hazardSupports.end() && observation.tick < known->second;
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
            // A remembered damaging sector floor says nothing about a
            // sprite support above it.  Ordinary portal travel lands on the
            // destination sector floor; sprite-support transitions are
            // represented by their own physical graph edges.
            if (hazardousSupport(SupportRef(kSupportSectorFloor, portal.to)))
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
        const SupportRef support = currentPlayerSupport();
        if (!hazardousSupport(support))
        {
            char detail[192];
            snprintf(detail, sizeof(detail),
                     "sector=%d support_kind=%d support=%d hits=%d lost=%d health=%d",
                     observation.sector, int(support.kind), support.id,
                     damageBurstCount, bled, observation.health);
            event("hazard_support_detected", detail);
        }
        hazardSupports[support] = observation.tick + kHazardAvoidTicks;
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
            if (edgeFailed(edgeId))
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

    // Reaching an actuator is not the same thing as being able to use the
    // transport it controls.  A lift button may be reachable through a thin
    // floor from a lower layer even though neither stable carrier endpoint
    // meets that layer.  Keep that action in memory, but defer it until the
    // current physical floor matches a boardable endpoint.  Non-transport
    // receivers are unaffected.
    bool supportTransportBoardableAtActionPose(
        const Portal &target, int targetSector, int actionFloorZ) const
    {
        const int receiver = mechanismSector(target, targetSector);
        if (!sectorSweptOccupancySafe(receiver))
            return true;
        const int extra = sector[receiver].extra;
        if (extra <= 0 || extra >= kMaxXSectors)
            return true;
        const XSECTOR &dynamic = xsector[extra];
        if (dynamic.offFloorZ == dynamic.onFloorZ)
            return true;
        // The ledger may select a remembered remote action and navigate back
        // to it, so judge the recorded action support rather than Caleb's
        // unrelated present floor.  This is what distinguishes the same S3
        // actuator observed from the 392p and 292p layers.
        return std::abs(actionFloorZ - dynamic.offFloorZ) <= llmapper::capability::playerStepHeight()
            || std::abs(actionFloorZ - dynamic.onFloorZ) <= llmapper::capability::playerStepHeight();
    }

    bool supportTransportBoardableAtActionPose(
        const InteractionMemory &memory) const
    {
        const int actionFloorZ = memory.approachFloorZ != INT32_MAX
            ? memory.approachFloorZ : memory.target.fromFloorZ;
        return supportTransportBoardableAtActionPose(
            memory.target, memory.targetSector, actionFloorZ);
    }

    int interactionApproachZ(const InteractionMemory &memory) const
    {
        // For a moving-sector receiver, the remembered standing floor is the
        // transport prerequisite. Feeding its wall/sprite aim Z to the route
        // planner can select an overlapping layer that cannot board it.
        // Ordinary interactions are different: their target Z is valuable
        // projection evidence for selecting the far support component (for
        // example an exit visible beyond a chain of jumpable pillars).
        const int receiver = mechanismSector(memory.target,
                                             memory.targetSector);
        if (inRange(receiver, 0, numsectors)
            && sector[receiver].extra > 0
            && sector[receiver].extra < kMaxXSectors)
        {
            const XSECTOR &dynamic = xsector[sector[receiver].extra];
            if (dynamic.offFloorZ != dynamic.onFloorZ
                && memory.approachFloorZ != INT32_MAX)
                return memory.approachFloorZ;
        }
        return memory.z;
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
        // firing it from the landing can send the carrier away without the
        // player, so its useful action pose must remain aboard.  Ordinary
        // Use actuators are also valid call controls; requiring the receiver
        // after the first ride made a reusable lift impossible to call back
        // from its lower landing.  Endpoint boardability is modelled
        // separately by supportTransportBoardableAtActionPose().
        return memory.activationMode == llmapper::kActivateVector
            ? receiver : -1;
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

    bool boundarySupportAtStableEndpoint(const Portal &portal) const
    {
        const int sectors[2] = { portal.from, portal.to };
        for (int i = 0; i < 2; ++i)
        {
            if (!inRange(sectors[i], 0, numsectors))
                continue;
            const sectortype &record = sector[sectors[i]];
            const int extra = record.extra;
            if (extra <= 0 || extra >= kMaxXSectors)
                continue;
            const XSECTOR &dynamic = xsector[extra];
            if (dynamic.offFloorZ == dynamic.onFloorZ)
                continue;
            if (busyValueInMotion(dynamic.busy)
                || (record.floorz != dynamic.offFloorZ
                    && record.floorz != dynamic.onFloorZ))
                return false;
        }
        return true;
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
            const int previousClearance = lastClearance == boundaryClearance.end()
                ? portal.clearance : lastClearance->second;
            const bool standingCapabilityChanged =
                (previousClearance < playerStandingClearance())
                != (portal.clearance < playerStandingClearance());
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
            if (standingCapabilityChanged)
                navTopologySignature = 0;
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
            // A capability boundary changed, so the currently published
            // pose graph is no longer true even if the geometry has not yet
            // reached a stable endpoint.  Rebuild at this semantic change;
            // ordinary intermediate motion remains coalesced by
            // ensureNavTopology().
            navTopologySignature = 0;
            if (open)
            {
                // The world delta makes the graph edge and any newly
                // reachable observation work authoritative.  No separate
                // opened-route controller or freshness window is needed.
                clearTransitionAttempts(portal.wall);
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
    void dumpLedger(const char *reason, bool refresh = true)
    {
        if (lastLedgerDumpTick >= 0
            && observation.tick - lastLedgerDumpTick < 5 * kTicsPerSec)
            return;
        lastLedgerDumpTick = observation.tick;
        if (refresh)
            rebuildLedger();
        int pending = 0;
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
            else if (!llmapper::effectRequirementSatisfied(
                         opportunity.requiredEffects, effects))
                ++missingEffect;
            else
                ++pending;
        }
        char detail[256];
        snprintf(detail, sizeof(detail),
                 "reason=%s sector=%d ledger=%u pending=%d unreachable=%d locked_no_key=%d missing_effect=%d visited=%u inert=%u nav_failures=%u",
                 reason, observation.sector, unsigned(ledger.size()), pending,
                 unreachable, lockedNoKey, missingEffect, unsigned(visitedSectors.size()),
                 unsigned(inertBoundaries.size()), unsigned(navEdgeFailures.size()));
        event("ledger_snapshot", detail);
        int reported = 0;
        for (const llmapper::Opportunity &opportunity : ledger)
        {
            if (reported++ >= 12)
                break;
            const char *availability = opportunity.hops < 0
                ? "temporarily_unreachable"
                : opportunity.requiredKey > 0
                    && !(heldKeyMask & (1u << unsigned(opportunity.requiredKey & 31)))
                ? "missing_key"
                : !llmapper::effectRequirementSatisfied(
                      opportunity.requiredEffects, effects)
                ? "missing_effect"
                : "actionable";
            char entry[256];
            snprintf(entry, sizeof(entry),
                     "work_kind=%d subject=%d from=%d to=%d kind=%d availability=%s sector=%d target=%d wall=%d key=%d effects=%u depth=%d hops=%d local=%d ready=%d continuation=%d risk=%d",
                     int(opportunity.id.kind), opportunity.id.subject,
                     opportunity.id.from, opportunity.id.to,
                     int(opportunity.kind), availability,
                     opportunity.sector, opportunity.target, opportunity.wall, opportunity.requiredKey,
                     opportunity.requiredEffects, opportunity.depth, opportunity.hops,
                     opportunity.local ? 1 : 0, opportunity.ready ? 1 : 0,
                     opportunity.continuation ? 1 : 0,
                     opportunity.oneWayRisk);
            event("ledger_entry", entry);
        }
    }

    const llmapper::Opportunity *ledgerEntry(const llmapper::WorkId &id) const
    {
        for (const llmapper::Opportunity &opportunity : ledger)
            if (opportunity.id == id)
                return &opportunity;
        return nullptr;
    }

    bool commitWork(const llmapper::WorkSelection &chosen)
    {
        const llmapper::Opportunity *opportunity = ledgerEntry(chosen.work);
        if (!opportunity)
            return false;
        Objective objective;
        objective.work = chosen.work;
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
            auto memory = interactions.find(opportunity->id.subject);
            if (memory == interactions.end())
                return false;
            objective.type = kObjectiveInteraction;
            objective.id = memory->second.id;
            objective.approachCell = opportunity->approach;
            objective.sector = inRange(objective.approachCell, 0,
                                        int(navCells.size()))
                ? navCells[size_t(objective.approachCell)].sector : -1;
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
            if (!inRange(opportunity->target, 0, int(navCells.size())))
                return false;
            const NavCell &cell = navCells[size_t(opportunity->target)];
            const PhysicalPoseKey handle = physicalPoseKey(cell);
            if (observedCells.count(handle)
                && !opportunity->requiresOccupancy)
                return false;
            objective.type = kObjectiveExpose;
            objective.id = opportunity->target;
            objective.targetSector = cell.sector;
            const int approach = inRange(opportunity->approach, 0,
                int(navCells.size())) ? opportunity->approach : opportunity->target;
            objective.approachCell = approach;
            objective.requiresOccupancy = opportunity->requiresOccupancy;
            objective.sector = navCells[size_t(approach)].sector;
            objective.x = navCells[size_t(approach)].center.x;
            objective.y = navCells[size_t(approach)].center.y;
            objective.z = navCells[size_t(approach)].z;
            coverageTargetX = objective.x;
            coverageTargetY = objective.y;
            coverageTargetSector = navCells[size_t(approach)].sector;
            char expose[256];
            snprintf(expose, sizeof(expose),
                     "cell=%d sector=%d target=(%d,%d,%d) approach_cell=%d approach=(%d,%d,%d,s%d) information_boundary=1 hops=%d",
                     cell.id, cell.sector, cell.center.x, cell.center.y, cell.z,
                     approach, objective.x, objective.y,
                     navCells[size_t(approach)].z,
                     navCells[size_t(approach)].sector, opportunity->hops);
            event("exploration_viewpoint_selected", expose);
            break;
        }
        case llmapper::kOpportunityExit:
        {
            if (!inRange(opportunity->target, 0, int(navCells.size())))
                return false;
            const NavCell &cell = navCells[size_t(opportunity->target)];
            objective.type = kObjectiveExit;
            objective.id = cell.sector;
            objective.sector = cell.sector;
            objective.targetSector = cell.sector;
            objective.x = cell.center.x;
            objective.y = cell.center.y;
            objective.z = cell.z;
            break;
        }
        case llmapper::kOpportunityPickup:
        {
            auto memory = objectMemory.find(opportunity->id.subject);
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

        const bool backtrack = opportunity->kind == llmapper::kOpportunityFrontier
            && !opportunity->local;
        selectObjective(objective, opportunity->id == chosen.work
                                       ? chosen.reason : "REOPEN_ROUTE_TO_OBJECTIVE");
        selectedWorkReason = chosen.reason;
        char detail[256];
        snprintf(detail, sizeof(detail),
                 "reason=%s kind=%d work_kind=%d subject=%d work_from=%d work_to=%d sector=%d target=%d wall=%d depth=%d hops=%d key=%d one_way_risk=%d from_sector=%d",
                 chosen.reason, int(opportunity->kind), int(opportunity->id.kind),
                 opportunity->id.subject, opportunity->id.from,
                 opportunity->id.to,
                 opportunity->sector, opportunity->target, opportunity->wall,
                 opportunity->depth, opportunity->hops, opportunity->requiredKey,
                 opportunity->oneWayRisk,
                 observation.sector);
        event("work_selected", detail);
        if (backtrack)
            event("backtrack_started", detail);
        return true;
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
            if (object.kind != kind)
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
            return candidate.statnum == kStatItem
                && candidate.type == memory.object.type
                && itemCategory(candidate.type) != nullptr;
        return true;
    }

    bool pickupUsefulNow(const VisibleObject &object) const
    {
        return gMe && inRange(object.sprite, 0, kMaxSprites)
            && playerCanBenefitFromPickup(gMe, &sprite[object.sprite]);
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
                continue;
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
        // Engine acceptance is not proof that useful work was done. Keep the
        // affordance failed in the same world state, then reconsider it when
        // another action has changed that state. This is the generic
        // prerequisite case: an actuator may be real but ineffective until a
        // different mechanism establishes the pose/geometry it consumes.
        if (memory.attempted && memory.activated
            && !memory.observedLocalEffect && !memory.observedKnownWorldDelta)
        {
            return memory.activationWorldRevision >= 0
                && knowledgeRevision != memory.activationWorldRevision;
        }
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
        memory.selectedSurfaceObjectiveTick = currentObjective.startedTick;
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
        std::vector<char> reachable;
        llmapper::markReachableNavCells(navCells, start, navEdgeFailures, 0,
                                        reachable);
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

    // Evaluate a hypothetical standable pose through the same engine query
    // that an actual Use pulse will invoke. HitScan must receive the real
    // global sprite because it temporarily clears that sprite's collision;
    // use a synchronous save/relocate/scan/restore transaction for the
    // player origin/view. ActionScanPreview has no trigger side effects, and
    // gHitInfo is restored so the query cannot leak into unrelated gameplay.
    bool previewInteractionFromPose(const InteractionMemory &memory,
                                    const NavCell &cell, bool crouched,
                                    int &angle, int &look) const
    {
        if (!gMe || !gMe->pSprite || !inRange(cell.sector, 0, numsectors))
            return false;

        spritetype &origin = *gMe->pSprite;
        const spritetype savedOrigin = origin;
        const int savedPosture = gMe->posture;
        const int savedZView = gMe->zView;
        const fix16_t savedAngle = gMe->q16ang;
        const fix16_t savedLook = gMe->q16look;
        const fix16_t savedHoriz = gMe->q16horiz;
        const int savedSlope = gMe->slope;
        const int16_t savedSpriteAngle = gMe->pSprite->ang;
        origin.x = cell.center.x;
        origin.y = cell.center.y;
        origin.z = playerOriginAtSupport(cell.z);
        origin.sectnum = int16_t(cell.sector);
        angle = interactionFacingAngleFromPose(
            memory, origin.x, origin.y, origin.ang);
        origin.ang = int16_t(angle);
        gMe->q16ang = fix16_from_int(angle);
        gMe->pSprite->ang = int16_t(angle);
        gMe->posture = crouched ? kPostureCrouch : kPostureStand;
        const POSTURE &posture = gMe->pPosture[gMe->lifeMode][gMe->posture];
        gMe->zView = origin.z - posture.eyeAboveZ;
        look = interactionLookTargetFromPose(memory, crouched,
                                             origin.x, origin.y, origin.z);
        gMe->q16look = fix16_from_int(look);
        if (VanillaMode())
        {
            if (look > 0)
                gMe->q16horiz = fix16_from_int(
                    mulscale30(120, Sin(look << 3)));
            else if (look < 0)
                gMe->q16horiz = fix16_from_int(
                    mulscale30(180, Sin(look << 3)));
            else
                gMe->q16horiz = 0;
        }
        else
        {
            gMe->q16horiz = fix16_from_float(
                100.f * tanf(float(look) * 3.14159265358979323846f / 1024.f));
        }
        gMe->slope = (-fix16_to_int(gMe->q16horiz)) << 7;

        const HITINFO savedHit = gHitInfo;
        int target = -1;
        int extra = -1;
        const int hit = ActionScanPreview(gMe, &target, &extra);
        origin = savedOrigin;
        gMe->posture = savedPosture;
        gMe->zView = savedZView;
        gMe->q16ang = savedAngle;
        gMe->q16look = savedLook;
        gMe->q16horiz = savedHoriz;
        gMe->slope = savedSlope;
        gMe->pSprite->ang = savedSpriteAngle;
        gHitInfo = savedHit;
        return interactionTargetMatches(memory, hit, target);
    }

    // Can this exact physical origin operate the affordance if the player
    // merely turns/aims (and adopts its required posture)?  The live overlay
    // uses this same side-effect-free ActionScan authority as planning.  It
    // does not substitute distance, sector ownership, or line of sight for
    // the engine's real interaction acquisition.
    bool previewInteractionFromCurrentOrigin(const InteractionMemory &memory) const
    {
        if (!gMe || !gMe->pSprite)
            return false;
        const int savedPosture = gMe->posture;
        const int savedZView = gMe->zView;
        const fix16_t savedAngle = gMe->q16ang;
        const fix16_t savedLook = gMe->q16look;
        const fix16_t savedHoriz = gMe->q16horiz;
        const int savedSlope = gMe->slope;
        const int16_t savedSpriteAngle = gMe->pSprite->ang;
        const int angle = interactionFacingAngleFromPose(
            memory, gMe->pSprite->x, gMe->pSprite->y, gMe->pSprite->ang);
        const bool crouched = interactionNeedsCrouch(memory);
        gMe->q16ang = fix16_from_int(angle);
        gMe->pSprite->ang = int16_t(angle);
        gMe->posture = crouched ? kPostureCrouch : kPostureStand;
        const POSTURE &posture = gMe->pPosture[gMe->lifeMode][gMe->posture];
        gMe->zView = gMe->pSprite->z - posture.eyeAboveZ;
        const int look = interactionLookTargetFromPose(
            memory, crouched, gMe->pSprite->x, gMe->pSprite->y,
            gMe->pSprite->z);
        gMe->q16look = fix16_from_int(look);
        if (VanillaMode())
        {
            if (look > 0)
                gMe->q16horiz = fix16_from_int(mulscale30(120, Sin(look << 3)));
            else if (look < 0)
                gMe->q16horiz = fix16_from_int(mulscale30(180, Sin(look << 3)));
            else
                gMe->q16horiz = 0;
        }
        else
            gMe->q16horiz = fix16_from_float(
                100.f * tanf(float(look) * 3.14159265358979323846f / 1024.f));
        gMe->slope = (-fix16_to_int(gMe->q16horiz)) << 7;
        const HITINFO savedHit = gHitInfo;
        int target = -1;
        int extra = -1;
        const int hit = ActionScanPreview(gMe, &target, &extra);
        gMe->posture = savedPosture;
        gMe->zView = savedZView;
        gMe->q16ang = savedAngle;
        gMe->q16look = savedLook;
        gMe->q16horiz = savedHoriz;
        gMe->slope = savedSlope;
        gMe->pSprite->ang = savedSpriteAngle;
        gHitInfo = savedHit;
        return interactionTargetMatches(memory, hit, target);
    }

    // Find work in pose space, not sector space: traverse the currently
    // connected physical support graph and retain the cheapest stance whose
    // real engine ActionScan names the remembered affordance.  Sector ids on
    // cells are passed to the engine only so hitscan/collision can locate the
    // pose in Build's representation.
    int reachableInteractionPose(InteractionMemory &memory,
                                 const std::vector<char> *knownReachable = nullptr)
    {
        ensureNavTopology();
        const int start = currentNavStartCell();
        if (start < 0)
            return -1;

        std::vector<char> ownedReachable;
        if (!knownReachable)
        {
            llmapper::markReachableNavCells(navCells, start, navEdgeFailures,
                                            0, ownedReachable);
            knownReachable = &ownedReachable;
        }
        std::vector<int> hops(navCells.size(), -1);
        std::deque<int> queue;
        hops[size_t(start)] = 0;
        queue.push_back(start);
        while (!queue.empty())
        {
            const int current = queue.front();
            queue.pop_front();
            for (const NavLink &link : navCells[size_t(current)].links)
            {
                if (!llmapper::traversableMode(link.mode)
                    || link.condition.enabled
                    || !inRange(link.target, 0, int(navCells.size()))
                    || hops[size_t(link.target)] >= 0
                    || !(*knownReachable)[size_t(link.target)]
                    || !navCells[size_t(link.target)].live)
                    continue;
                hops[size_t(link.target)] = hops[size_t(current)] + 1;
                queue.push_back(link.target);
            }
        }

        int best = -1;
        int bestHops = INT32_MAX;
        int bestAimDistance = INT32_MAX;
        int bestAngle = 0;
        int bestLook = 0;
        bool bestCrouch = false;
        const InteractionCandidate *bestSurface = nullptr;
        int reachableCount = 0;
        int nearbyCount = 0;
        int highestNearbyZ = INT32_MAX;
        int lowestNearbyZ = INT32_MIN;
        int closestDistance = INT32_MAX;
        int closestZ = 0;
        const int maximumCandidateDistance = kActionScanRange + (1 << kNavGridShift);
        for (const NavCell &cell : navCells)
        {
            if (!inRange(cell.id, 0, int(knownReachable->size()))
                || !(*knownReachable)[size_t(cell.id)]
                || hops[size_t(cell.id)] < 0
                || !cell.live)
                continue;
            ++reachableCount;
            auto inspectSurface = [&](const InteractionCandidate *surface)
            {
                InteractionMemory physical = memory;
                if (surface)
                {
                    physical.fromSector = surface->fromSector;
                    physical.targetSector = surface->targetSector;
                    physical.x = surface->x;
                    physical.y = surface->y;
                    physical.z = surface->z;
                    physical.target = surface->target;
                }
                const int aimDistance = distance2(
                    cell.center.x, cell.center.y, physical.x, physical.y);
                if (physical.kind != kInteractionSector
                    && aimDistance
                        > maximumCandidateDistance * maximumCandidateDistance)
                    return;
                ++nearbyCount;
                highestNearbyZ = std::min(highestNearbyZ, cell.z);
                lowestNearbyZ = std::max(lowestNearbyZ, cell.z);
                if (aimDistance < closestDistance)
                {
                    closestDistance = aimDistance;
                    closestZ = cell.z;
                }
                for (int posture = 0; posture < 2; ++posture)
                {
                    int angle = 0;
                    int look = 0;
                    const bool crouched = posture != 0;
                    if (!previewInteractionFromPose(physical, cell, crouched,
                                                    angle, look))
                        continue;
                    // Pose discovery and execution must agree on what
                    // "actionable" means.  A lift control may ActionScan
                    // through thin geometry from a layer which cannot board
                    // either stable carrier endpoint.  Keep the affordance
                    // remembered, but do not publish that unusable stance as
                    // its executable action pose.
                    if (!supportTransportBoardableAtActionPose(
                            physical.target, physical.targetSector, cell.z))
                        continue;
                    if (hops[size_t(cell.id)] < bestHops
                        || (hops[size_t(cell.id)] == bestHops
                            && aimDistance < bestAimDistance))
                    {
                        best = cell.id;
                        bestHops = hops[size_t(cell.id)];
                        bestAimDistance = aimDistance;
                        bestAngle = angle;
                        bestLook = look;
                        bestCrouch = crouched;
                        bestSurface = surface;
                    }
                    break;
                }
            };
            if (memory.kind == kInteractionWall && !memory.surfaces.empty())
            {
                for (const InteractionCandidate &surface : memory.surfaces)
                    inspectSurface(&surface);
            }
            else
                inspectSurface(nullptr);
        }

        if (bestSurface)
        {
            memory.fromSector = bestSurface->fromSector;
            memory.targetSector = bestSurface->targetSector;
            memory.x = bestSurface->x;
            memory.y = bestSurface->y;
            memory.z = bestSurface->z;
            memory.target = bestSurface->target;
        }
        memory.actionPoseCell = best;
        memory.actionPoseTopologyRevision = navTopologyRevision;
        memory.actionPoseSignature = interactionStateSignature(memory);
        memory.actionPoseStartArea = navCells[size_t(start)].walkArea;
        memory.actionPoseStartSupport = navCells[size_t(start)].support;
        memory.actionPoseHops = best >= 0 ? bestHops : -1;
        memory.actionPoseAngle = bestAngle;
        memory.actionPoseLook = bestLook;
        memory.actionPoseCrouch = bestCrouch;
        char detail[320];
        snprintf(detail, sizeof(detail),
                 "kind=%d id=%d target=(%d,%d,%d) start_cell=%d reachable_cells=%d nearby_cells=%d nearby_high_z=%d nearby_low_z=%d closest_distance=%d closest_z=%d result_cell=%d result_hops=%d result_sector=%d result_z=%d topology=%d",
                 int(memory.kind), memory.id, memory.x, memory.y, memory.z,
                 start, reachableCount, nearbyCount,
                 nearbyCount ? highestNearbyZ : 0,
                 nearbyCount ? lowestNearbyZ : 0,
                 closestDistance < INT32_MAX
                     ? int(std::sqrt(double(closestDistance))) : -1,
                 closestDistance < INT32_MAX ? closestZ : 0,
                 best, memory.actionPoseHops,
                 best >= 0 ? navCells[size_t(best)].sector : -1,
                 best >= 0 ? navCells[size_t(best)].z : 0,
                 navTopologyRevision);
        event("interaction_pose_search", detail);
        return best;
    }

    // Find a currently reachable stance whose real player-height weapon
    // origin can see the actuator.  The actuator's Z is aiming geometry, not
    // a floor height: projecting it into the support graph can select an
    // unrelated high crate/voxel and manufacture a climb that has nothing to
    // do with obtaining a shot.  Search collision-derived standable cells and
    // use engine visibility to expose an actual firing pose instead.
    int reachableFiringPose(int aimX, int aimY, int aimZ, int targetSector,
                            int shotZ, int minimumDistance,
                            int maximumDistance = INT32_MAX)
    {
        ensureNavTopology();
        const int start = currentNavStartCell();
        if (start < 0)
            return -1;
        std::vector<int> distance(navCells.size(), -1);
        std::deque<int> queue;
        distance[size_t(start)] = 0;
        queue.push_back(start);
        while (!queue.empty())
        {
            const int current = queue.front();
            queue.pop_front();
            for (const NavLink &link : navCells[size_t(current)].links)
            {
                if (!llmapper::traversableMode(link.mode)
                    || !inRange(link.target, 0, int(navCells.size()))
                    || distance[size_t(link.target)] >= 0)
                    continue;
                distance[size_t(link.target)] = distance[size_t(current)] + 1;
                queue.push_back(link.target);
            }
        }

        const int weaponAboveFloor = shotZ
            - llmapper::capability::playerFloorZ();
        int best = -1;
        int bestRouteDistance = INT32_MAX;
        int bestAimDistance = INT32_MAX;
        int connectedCount = 0;
        int separatedCount = 0;
        int rangedCount = 0;
        int visibleCount = 0;
        for (const NavCell &cell : navCells)
        {
            if (distance[size_t(cell.id)] < 0)
                continue;
            ++connectedCount;
            const int horizontal = distance2(cell.center.x, cell.center.y,
                                             aimX, aimY);
            // Do not ask cansee() to resolve a zero-length endpoint.  Vector
            // interactions also need some separation to form a useful ray.
            if (horizontal < 256 * 256)
                continue;
            ++separatedCount;
            if (minimumDistance > 0
                && horizontal < minimumDistance * minimumDistance)
                continue;
            if (maximumDistance < INT32_MAX
                && horizontal > int64_t(maximumDistance) * maximumDistance)
                continue;
            ++rangedCount;
            const int candidateShotZ = cell.z + weaponAboveFloor;
            if (!cansee(cell.center.x, cell.center.y, candidateShotZ,
                        cell.sector, aimX, aimY, aimZ, targetSector))
                continue;
            ++visibleCount;
            const int routeDistance = distance[size_t(cell.id)];
            if (routeDistance < bestRouteDistance
                || (routeDistance == bestRouteDistance
                    && horizontal < bestAimDistance))
            {
                best = cell.id;
                bestRouteDistance = routeDistance;
                bestAimDistance = horizontal;
            }
        }
        int audit = llmapper::mixHash(navTopologyRevision, aimX);
        audit = llmapper::mixHash(audit, aimY);
        audit = llmapper::mixHash(audit, aimZ);
        audit = llmapper::mixHash(audit, targetSector);
        audit = llmapper::mixHash(audit, minimumDistance);
        audit = llmapper::mixHash(audit, maximumDistance);
        if (reportedFiringPoseSearches.insert(audit).second)
        {
            char detail[288];
            snprintf(detail, sizeof(detail),
                     "target=(%d,%d,%d,s%d) start=%d connected=%d separated=%d in_range=%d visible=%d result=%d result_sector=%d result_z=%d min=%d max=%d topology=%d",
                     aimX, aimY, aimZ, targetSector, start, connectedCount,
                     separatedCount, rangedCount, visibleCount, best,
                     best >= 0 ? navCells[size_t(best)].sector : -1,
                     best >= 0 ? navCells[size_t(best)].z : 0,
                     minimumDistance, maximumDistance, navTopologyRevision);
            event("firing_pose_search", detail);
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
        setGoal(memory.activationMode == llmapper::kActivateDamage
                    ? "DAMAGE_BLOCKER" : "VECTOR_ACTIVATE",
                memory.id);
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
        if (!ranged && interactionNeedsCrouch(memory))
            input.buttonFlags.crouch = 1;

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
            if (currentObjective.requiredSupportSector < 0)
            {
                int signature = llmapper::mixHash(17, aimX);
                signature = llmapper::mixHash(signature, aimY);
                signature = llmapper::mixHash(signature, aimZ);
                signature = llmapper::mixHash(signature, shotTargetSector);
                signature = llmapper::mixHash(signature, ranged ? 1 : 0);
                if (memory.firingPoseTopologyRevision != navTopologyRevision
                    || memory.firingPoseSignature != signature)
                {
                    memory.firingPoseCell = reachableFiringPose(
                        aimX, aimY, aimZ, shotTargetSector, shotZ,
                        ranged ? safeDistance : 0,
                        ranged ? plannerVectorReach(vectorType) : reach);
                    memory.firingPoseTopologyRevision = navTopologyRevision;
                    memory.firingPoseSignature = signature;
                    if (memory.firingPoseCell >= 0)
                    {
                        const NavCell &pose = navCells[size_t(memory.firingPoseCell)];
                        char detail[224];
                        snprintf(detail, sizeof(detail),
                                 "target=%d cell=%d sector=%d support_kind=%d support_id=%d at=(%d,%d,%d) reason=reachable_engine_visible_stance",
                                 memory.id, pose.id, pose.sector,
                                 int(pose.support.kind), pose.support.id,
                                 pose.center.x, pose.center.y, pose.z);
                        event("ranged_firing_pose_selected", detail);
                    }
                }
                if (inRange(memory.firingPoseCell, 0, int(navCells.size())))
                {
                    const NavCell &pose = navCells[size_t(memory.firingPoseCell)];
                    return navigateTo(pose.center.x, pose.center.y, pose.z,
                                      pose.sector, memory.id,
                                      kTraversalUnknown);
                }
            }
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
            return navigateTo(memory.x, memory.y, interactionApproachZ(memory), activationSector,
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
                snapshotReachableBeforeAction(memory);
            }
            if (memory.state != 2)
                memory.state = 1;
            memory.lastActivationTick = observation.tick;
            memory.activationWorldRevision = knowledgeRevision;
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

        const unsigned available = availableEffectCapabilities();
        if ((memory.requiredEffects & llmapper::kEffectBulletDamage)
            && (available & llmapper::kEffectBulletDamage))
            return shootObstacle(memory);

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
        {
            int signature = llmapper::mixHash(17, memory.x);
            signature = llmapper::mixHash(signature, memory.y);
            signature = llmapper::mixHash(signature, aimZ);
            signature = llmapper::mixHash(signature, memory.targetSector);
            signature = llmapper::mixHash(signature, safeDistance);
            if (memory.firingPoseTopologyRevision != navTopologyRevision
                || memory.firingPoseSignature != signature)
            {
                memory.firingPoseCell = reachableFiringPose(
                    memory.x, memory.y, aimZ, memory.targetSector, eyeZ,
                    safeDistance, maxStandOff);
                memory.firingPoseTopologyRevision = navTopologyRevision;
                memory.firingPoseSignature = signature;
            }
            if (inRange(memory.firingPoseCell, 0, int(navCells.size())))
            {
                const NavCell &pose = navCells[size_t(memory.firingPoseCell)];
                return navigateTo(pose.center.x, pose.center.y, pose.z,
                                  pose.sector, memory.id,
                                  kTraversalUnknown);
            }
            return input;
        }
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
                snapshotReachableBeforeAction(memory);
                memory.effectStartedTick = observation.tick;
                memory.effectAmmoBefore = gMe->ammoCount[weapon - 1];
                memory.effectBlastRadius = blastRadius;
                ++memory.activationCount;
                event("effect_delivery_started", "delivery=explosive_projectile");
                input.buttonFlags.shoot = 1;
                memory.lastActivationTick = observation.tick;
                memory.activationWorldRevision = knowledgeRevision;
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
                snapshotReachableBeforeAction(memory);
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
            memory.activationWorldRevision = knowledgeRevision;
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
        if (memory.activationMode == llmapper::kActivateUse)
        {
            ensureNavTopology();
            std::vector<char> reachable;
            llmapper::markReachableNavCells(navCells, currentNavStartCell(),
                                            navEdgeFailures, 0, reachable);
            const int poseSignature = interactionStateSignature(memory);
            const int poseStartCell = currentNavStartCell();
            const int poseStartArea = inRange(
                poseStartCell, 0, int(navCells.size()))
                ? navCells[size_t(poseStartCell)].walkArea : -1;
            const SupportRef poseStartSupport = inRange(
                poseStartCell, 0, int(navCells.size()))
                ? navCells[size_t(poseStartCell)].support : SupportRef();
            const bool cachedPoseReachable = inRange(
                    memory.actionPoseCell, 0, int(reachable.size()))
                && reachable[size_t(memory.actionPoseCell)];
            if (memory.actionPoseTopologyRevision != navTopologyRevision
                || memory.actionPoseSignature != poseSignature
                || memory.actionPoseStartArea != poseStartArea
                || memory.actionPoseStartSupport != poseStartSupport
                || !cachedPoseReachable)
            {
                reachableInteractionPose(memory, &reachable);
            }
            currentObjective.approachCell = memory.actionPoseCell;
            if (!inRange(currentObjective.approachCell, 0, int(navCells.size())))
            {
                memory.unavailableFromSector = observation.sector;
                memory.unavailableState = poseSignature;
                memory.unavailablePose = navTopologyRevision;
                char unavailable[192];
                snprintf(unavailable, sizeof(unavailable),
                         "kind=%d id=%d topology=%d reason=NO_REACHABLE_ACTIONSCAN_POSE",
                         int(memory.kind), memory.id, navTopologyRevision);
                event("interaction_unavailable", unavailable);
                invalidateObjective("interaction_pose_unreachable");
                return GINPUT{};
            }

            const NavCell &pose = navCells[size_t(currentObjective.approachCell)];
            currentObjective.sector = pose.sector;
            // The sampled cell proves that a reachable solution exists; it
            // is not an exact coordinate contract. Execution asks the same
            // authoritative engine question again at the live player origin.
            // As soon as ActionScan can resolve the affordance after merely
            // turning/aiming here, movement is finished here. A transport
            // action additionally requires a boardable endpoint: being able
            // to call a carrier through thin geometry is not the same as
            // being able to consume its motion. Chasing the sample centre
            // conflated pose-space evidence with a waypoint and made the bot
            // orbit an already usable interaction stance.
            const bool engineActionableHere =
                previewInteractionFromCurrentOrigin(memory);
            const bool usefulTransportPose =
                supportTransportBoardableAtActionPose(
                    memory.target, memory.targetSector,
                    llmapper::capability::playerFloorZ());
            if (!engineActionableHere || !usefulTransportPose)
            {
                setGoal("NAVIGATE_TO_INTERACTION_POSE", memory.id);
                const int approachSignature = navigationSignature(
                    pose.center.x, pose.center.y, pose.z, pose.sector,
                    memory.id);
                const bool detailedRoute = (!navRoute.empty()
                        && navRouteSignature == approachSignature
                        && navRouteTopologyRevision == navTopologyRevision)
                    || (navRouteRejectedSignature != approachSignature
                        && installNavRoute(pose.id, approachSignature, true));
                if (detailedRoute)
                {
                    const GINPUT approach = navigateTo(
                        pose.center.x, pose.center.y, pose.z, pose.sector,
                        memory.id, kTraversalUnknown);
                    if ((!navRoute.empty()
                         && navRouteSignature == approachSignature)
                        || approach.forward || approach.strafe || approach.q16turn
                        || approach.q16mlook || approach.buttonFlags.jump
                        || approach.buttonFlags.crouch)
                        return approach;
                }

                memory.unavailableFromSector = observation.sector;
                memory.unavailableState = poseSignature;
                memory.unavailablePose = navTopologyRevision;
                char unavailable[224];
                snprintf(unavailable, sizeof(unavailable),
                         "kind=%d id=%d pose_cell=%d pose=(%d,%d,%d) topology=%d reason=NO_PHYSICAL_ROUTE_TO_ACTIONSCAN_POSE",
                         int(memory.kind), memory.id, pose.id,
                         pose.center.x, pose.center.y, pose.z,
                         navTopologyRevision);
                event("interaction_unavailable", unavailable);
                invalidateObjective("interaction_pose_route_unreachable");
                return GINPUT{};
            }
        }

        const int requiredApproachSector = currentObjective.requiredSupportSector >= 0
            ? currentObjective.requiredSupportSector : memory.fromSector;
        if (memory.activationMode != llmapper::kActivateUse
            && requiredApproachSector != observation.sector)
        {
            // Route through visited space with the ordinary navigator: the
            // mechanism's own sector is the destination, and crossing any
            // number of already-walked boundaries to get there is normal
            // transportation, not a new exploration decision.
            setGoal("NAVIGATE_TO_NEW_INTERACTION", memory.id);
            int approachX = memory.x;
            int approachY = memory.y;
            int approachZ = interactionApproachZ(memory);
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
            // Every return route is resolved on concrete support poses and
            // crossings. A Build-sector route cannot preserve which doorway,
            // support layer, traversal mode, or gate state is being used.
            const int approachSignature = navigationSignature(
                approachX, approachY, approachZ, requiredApproachSector,
                memory.id);
            ensureNavTopology();
            const int approachPose = resolveNavPose(
                approachX, approachY, approachZ, requiredApproachSector);
            const bool detailedRoute = (!navRoute.empty()
                    && navRouteSignature == approachSignature
                    && navRouteTopologyRevision == navTopologyRevision)
                || (navRouteRejectedSignature != approachSignature
                    && approachPose >= 0
                    && installNavRoute(approachPose, approachSignature));
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

            // No concrete pose/crossing route currently reaches the required
            // approach layer. Keep the work deferred; do not replace that
            // physical answer with sector adjacency or a straight-line guess.
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
        const bool interactionCrouch = memory.actionPoseCell >= 0
            ? memory.actionPoseCrouch : interactionNeedsCrouch(memory);
        memory.target.interactionCrouch = interactionCrouch;
        const int expectedDistance = memory.kind == kInteractionSector
            ? 0 : distance2(observation.x, observation.y, memory.x, memory.y);
        const int expectedAngle = interactionFacingAngle(memory);
        const int desiredLook = interactionLookTarget(memory, interactionCrouch);
        const int currentLook = fix16_to_int(gMe->q16look);
        const int lookDelta = desiredLook - currentLook;
        // ActionScan is a ray, not a contact sensor.  Standing flush against
        // one face of a compound/rotating door can put the intended push
        // wall behind the near face; a valid pose may be near the outer edge
        // of ActionScan range.  Treat distance as an inspection annulus and
        // back away before declaring the surface unavailable.
        const int inspectionStandoff = std::max(playerClipRadius() + 128,
                                                kActionScanRange - 128);
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
            rearmAroundMechanism(memory);
            lastSemanticProgressTick = observation.tick;
            event("interaction_world_delta",
                  "reason=accepted_action_target_disappeared_at_stable_pose");
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
        // One activation is one transaction. Do not send another Use pulse
        // while its result is still pending merely because the cooldown
        // elapsed. A settled no-effect transaction is deferred until new
        // world evidence makes reactivation meaningful.
        const int movedSector = mechanismSector(memory.target, memory.targetSector);
        const SupportRef standingSupport = currentPlayerSupport();
        bool reversibleWallMotion = false;
        if (movedSector >= 0 && movedSector == observation.sector
            && memory.kind == kInteractionSprite && memory.reversible)
        {
            const int extra = sector[movedSector].extra;
            if (extra > 0 && extra < kMaxXSectors)
            {
                const XSECTOR &dynamic = xsector[extra];
                reversibleWallMotion = dynamic.offFloorZ == dynamic.onFloorZ
                    && dynamic.offCeilZ == dynamic.onCeilZ
                    && dynamic.offFloorZ - dynamic.offCeilZ
                        >= playerStandingClearance();
            }
        }
        const bool safeOccupiedSupportTransition = movedSector >= 0
            && movedSector == observation.sector
            && ((sectorSweptOccupancySafe(movedSector)
                 && standingSupport == SupportRef(kSupportSectorFloor, movedSector)
                 && llmapper::capability::playerFloorZ()
                    == sector[movedSector].floorz)
                // A physical reversible actuator placed inside a sector
                // whose floor and ceiling do not move is not an elevator or
                // crusher.  Its receiver is wall motion (rotation/sliding),
                // and refusing it merely because Build reports the player
                // in the same sector blocks legitimate causal exploration.
                || reversibleWallMotion);
        const bool wouldMoveOwnSector = movedSector >= 0
            && movedSector == observation.sector
            && !memory.target.sectorPushCurrent
            && !safeOccupiedSupportTransition;
        if (safeOccupiedSupportTransition && !memory.attempted)
        {
            char detail[192];
            snprintf(detail, sizeof(detail),
                     "kind=%d id=%d mechanism_sector=%d support_z=%d wall_motion=%d reason=occupied_transition_proven_reversible",
                     int(memory.kind), memory.id, movedSector,
                     llmapper::capability::playerFloorZ(),
                     reversibleWallMotion ? 1 : 0);
            event("occupied_support_transition_accepted", detail);
        }
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
            && (!memory.attempted || recoveryActivation || reactivation))
        {
            if (reactivation)
            {
                event("interaction_reactivation",
                      !memory.observedLocalEffect && !memory.observedKnownWorldDelta
                          ? "reason=world_changed_after_no_effect"
                          : "reason=previously_traversed_route_closed");
            }
            memory.attempted = true;
            memory.state = 1;
            memory.activated = false;
            triedSurfaces.erase(interactionMemoryKey(memory) * 8192 + memory.target.wall);
            memory.lastActivationTick = observation.tick;
            memory.activationWorldRevision = knowledgeRevision;
            memory.beforeState = interactionStateSignature(memory);
            snapshotReachableBeforeAction(memory);
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
            && expectedDistance >= inspectionStandoff * inspectionStandoff
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
            const bool hasPhysicalUsePose = memory.activationMode
                    == llmapper::kActivateUse
                && inRange(currentObjective.approachCell, 0,
                           int(navCells.size()));
            if (!hasPhysicalUsePose)
            {
                // Non-Use effect delivery retains its established approach;
                // those paths replace this with a firing/delivery pose before
                // issuing their engine action.
                const int navigationSector = currentObjective.requiredSupportSector >= 0
                    ? currentObjective.requiredSupportSector
                    : memory.fromSector >= 0 ? memory.fromSector : observation.sector;
                const MovementProbe interactionPath = probeMovement(
                    observation.x, observation.y, observation.z, observation.sector,
                    memory.x, memory.y, navigationSector,
                    std::max(256, kActionApproachRange / 2));
                if (memory.target.wall >= 0
                    && interactionPath.wall == memory.target.wall
                    && directLineKeepsSupport(memory.x, memory.y))
                {
                    resetNavigation();
                    input = steerTo(memory.x, memory.y, false, false,
                                    memory.id, navigationSector,
                                    interactionCapability);
                }
                else
                    input = navigateTo(memory.x, memory.y,
                                       interactionApproachZ(memory),
                                       navigationSector, memory.id,
                                       interactionCapability);
            }
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
            if (!hasPhysicalUsePose
                && expectedDistance > kActionScanRange * kActionScanRange)
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
            if (!hasPhysicalUsePose && std::abs(turnToTarget) < 96)
            {
                if (!valid && expectedDistance
                        < inspectionStandoff * inspectionStandoff)
                    input.forward = -512;
                else if (expectedDistance
                         > inspectionStandoff * inspectionStandoff)
                    input.forward = 1024;
            }
            noteCameraOwner("INTERACTION", memory.id, expectedAngle, desiredLook);
        }
        return input;
    }

    bool directObjectReachable(const VisibleObject &object) const
    {
        if (object.sector != observation.sector
            || std::abs(object.z - observation.z) > llmapper::capability::playerStepHeight())
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

    unsigned effectCapabilitiesForWeapon(int weapon) const
    {
        switch (weapon)
        {
        case kWeaponPitchfork:
        case kWeaponShotgun:
        case kWeaponTommy:
            return llmapper::kEffectBulletDamage;
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
        if (gMe && (gMe->hasWeapon[kWeaponPitchfork]
                    || gMe->hasWeapon[kWeaponShotgun]
                    || gMe->hasWeapon[kWeaponTommy]))
            effects |= llmapper::kEffectBulletDamage;
        return effects;
    }

    EffectDelivery effectDeliveryFor(const InteractionMemory &memory) const
    {
        EffectDelivery delivery;
        int weapon = 0;
        VECTOR_TYPE vectorType = kVectorTine;

        // Vector activation is an engine hitscan affordance.  Prefer an
        // actual ranged Vector producer, falling back to the pitchfork only
        // when a connected pose exists inside its real reach.
        if (memory.activationMode == llmapper::kActivateVector)
        {
            if (vectorWeaponAvailable(weapon, vectorType))
            {
                delivery.available = true;
                delivery.ranged = true;
                delivery.weapon = weapon;
                delivery.vectorType = vectorType;
                delivery.minimumDistance = safeExplosionSeparation(
                    interactionExplosionRadius(memory));
                delivery.maximumDistance = plannerVectorReach(vectorType);
            }
            else if (meleeWeaponAvailable())
            {
                delivery.available = true;
                delivery.weapon = kWeaponPitchfork;
                delivery.maximumDistance = meleeReach();
            }
            return delivery;
        }

        if (memory.activationMode != llmapper::kActivateDamage)
            return delivery;

        // Accepted damage bits are alternatives.  Prefer a ranged bullet
        // producer when present, then an explosive producer, then melee.
        // This ordering preserves consumable explosives and never promotes
        // melee capability into an invented long-range capability.
        if ((memory.requiredEffects & llmapper::kEffectBulletDamage)
            && vectorWeaponAvailable(weapon, vectorType))
        {
            delivery.available = true;
            delivery.ranged = true;
            delivery.weapon = weapon;
            delivery.vectorType = vectorType;
            delivery.maximumDistance = plannerVectorReach(vectorType);
            return delivery;
        }
        if ((memory.requiredEffects & llmapper::kEffectExplosive)
            && explosiveEffectProducer(weapon))
        {
            delivery.available = true;
            delivery.ranged = true;
            delivery.explosive = true;
            delivery.weapon = weapon;
            delivery.minimumDistance = safeExplosionSeparation(
                interactionExplosionRadius(memory, weapon));
            delivery.maximumDistance = delivery.minimumDistance
                + kExplosiveThrowAllowance;
            return delivery;
        }
        if ((memory.requiredEffects & llmapper::kEffectBulletDamage)
            && meleeWeaponAvailable())
        {
            delivery.available = true;
            delivery.weapon = kWeaponPitchfork;
            delivery.maximumDistance = meleeReach();
        }
        return delivery;
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

    int objectiveWorkHash(const Objective &objective) const
    {
        return llmapper::workIdHash(objective.work);
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

        // An inactive moving-support endpoint is a possible mechanism state,
        // not a currently standable pose, so it must not be inserted into the
        // live navigation graph.  Dynamic prerequisite planning still needs
        // to ask whether that hypothetical endpoint would meet a real landing.
        // Answer that locally at the engine boundary: does the endpoint height
        // fit a concrete cell in the requested live component?  This preserves
        // elevator reasoning without making future geometry reachable now.
        if (!inRange(sectorId, 0, numsectors))
            return false;
        const sectortype &moving = sector[sectorId];
        const int boundaryReach = 2 << kNavGridShift;
        const int64_t boundaryReach2 = int64_t(boundaryReach) * boundaryReach;
        const int usefulReach = std::max(llmapper::capability::playerStepHeight(),
                                         playerJumpRiseLimit());
        for (int i = 0; i < moving.wallnum; ++i)
        {
            const int wallId = moving.wallptr + i;
            if (!inRange(wallId, 0, numwalls)
                || !inRange(wall[wallId].point2, 0, numwalls)
                || !inRange(wall[wallId].nextsector, 0, numsectors)
                || (wall[wallId].cstat & CSTAT_WALL_BLOCK))
                continue;
            if (inRange(wall[wallId].nextwall, 0, numwalls)
                && (wall[wall[wallId].nextwall].cstat & CSTAT_WALL_BLOCK))
                continue;
            const walltype &start = wall[wallId];
            const walltype &end = wall[start.point2];
            for (const NavCell &cell : navCells)
            {
                if (cell.sector != start.nextsector || cell.walkArea != area
                    || std::abs(cell.z - z) > usefulReach)
                    continue;
                if (dist2ToSegment(cell.center.x, cell.center.y,
                                   start.x, start.y, end.x, end.y)
                    <= boundaryReach2)
                    return true;
            }
        }
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
            const int receiver = mechanismSector(memory.target,
                                                 memory.targetSector);
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

    void snapshotReachableBeforeAction(InteractionMemory &memory)
    {
        ensureNavTopology();
        std::vector<char> reachable;
        llmapper::markReachableNavCells(navCells, currentNavStartCell(),
                                        navEdgeFailures, 0, reachable);
        memory.reachableBeforeActivation.clear();
        for (const NavCell &cell : navCells)
            if (inRange(cell.id, 0, int(reachable.size()))
                && reachable[size_t(cell.id)])
                memory.reachableBeforeActivation.insert(physicalPoseKey(cell));
        memory.continuationTopologyRevision = -1;
    }

    void refreshCausalContinuations()
    {
        ensureNavTopology();
        for (auto cursor = causalContinuationCells.begin();
             cursor != causalContinuationCells.end();)
        {
            if (observedCells.count(cursor->first))
                cursor = causalContinuationCells.erase(cursor);
            else
                ++cursor;
        }

        std::vector<char> reachable;
        llmapper::markReachableNavCells(navCells, currentNavStartCell(),
                                        navEdgeFailures, 0, reachable);
        for (auto &entry : interactions)
        {
            InteractionMemory &memory = entry.second;
            if (!memory.observedKnownWorldDelta
                || memory.reachableBeforeActivation.empty()
                || memory.continuationTopologyRevision == navTopologyRevision)
                continue;
            int added = 0;
            for (const NavCell &cell : navCells)
            {
                if (!inRange(cell.id, 0, int(reachable.size()))
                    || !reachable[size_t(cell.id)])
                    continue;
                const PhysicalPoseKey handle = physicalPoseKey(cell);
                if (memory.reachableBeforeActivation.count(handle)
                    || observedCells.count(handle))
                    continue;
                if (causalContinuationCells.emplace(handle, entry.first).second)
                    ++added;
            }
            memory.continuationTopologyRevision = navTopologyRevision;
            if (added > 0)
            {
                char detail[160];
                snprintf(detail, sizeof(detail),
                         "interaction=%d topology_revision=%d new_unobserved_cells=%d pending=%u",
                         entry.first, navTopologyRevision, added,
                         unsigned(causalContinuationCells.size()));
                event("causal_continuation_discovered", detail);

                // There can be one observation frame between crossing the
                // opened boundary and rebuilding the support graph on its
                // far side. Do not let arbitrary work selected in that gap
                // become a stronger "commitment" than the action's actual
                // successor. The route transaction itself and a suspended
                // parent objective retain ownership.
                bool consumesContinuation = false;
                if (currentObjective.active
                    && currentObjective.type == kObjectiveExpose
                    && inRange(currentObjective.id, 0, int(navCells.size())))
                {
                    const NavCell &cell = navCells[size_t(currentObjective.id)];
                    const PhysicalPoseKey target = physicalPoseKey(cell);
                    consumesContinuation = causalContinuationCells.count(target) != 0;
                }
                if (currentObjective.active && !suspendedObjectiveActive
                    && currentObjective.type != kObjectiveInteraction
                    && currentObjective.type != kObjectiveFrontier
                    && !consumesContinuation)
                    invalidateObjective("superseded_by_causal_continuation");
            }
        }
    }

    std::vector<llmapper::VisibilityFrontier> semanticVisibilityFrontiers(
        const std::vector<char> &reachable)
    {
        // This query is used by work construction and stall diagnostics in
        // the same observation. Its bounded line-of-sight search is the
        // expensive part and its inputs change only with physical topology,
        // accumulated observation, or deterministic edge evidence—not with
        // the number of callers or elapsed time.
        int reachableSignature = 17;
        for (size_t i = 0; i < reachable.size(); ++i)
            if (reachable[i])
                reachableSignature = llmapper::mixHash(
                    reachableSignature, int(i));
        if (visibilityFrontierTopologyRevision == navTopologyRevision
            && visibilityFrontierObservedCount == observedCells.size()
            && visibilityFrontierFailureCount == navEdgeFailures.size()
            && visibilityFrontierVisitedSupportCount
                == visitedSupportPoses.size()
            && visibilityFrontierReachableSignature == reachableSignature)
            return visibilityFrontierCache;
        std::vector<llmapper::VisibilityCell> samples;
        samples.reserve(navCells.size());
        for (const NavCell &cell : navCells)
        {
            llmapper::VisibilityCell sample;
            sample.id = cell.id;
            sample.x = cell.center.x;
            sample.y = cell.center.y;
            sample.z = cell.z;
            sample.area = cell.walkArea;
            sample.partition = cell.sector;
            sample.observed = observedCells.count(physicalPoseKey(cell)) != 0;
            sample.reachable = inRange(cell.id, 0, int(reachable.size()))
                && reachable[size_t(cell.id)];
            for (const NavLink &link : cell.links)
                if (llmapper::traversableMode(link.mode))
                    sample.neighbors.push_back(link.target);
            samples.push_back(sample);
        }
        std::vector<llmapper::VisibilityFrontier> result =
            llmapper::deriveVisibilityFrontiers(
                samples, 768, 2048, kJumpTakeoffRange,
                playerJumpRiseLimit());

        std::set<int> represented;
        for (const llmapper::VisibilityFrontier &frontier : result)
            represented.insert(frontier.cell);
        // Surface visibility and player-pose observation are different
        // facts. A bridge or stair top can be plainly visible from below
        // while occupying it exposes a new camera volume (and objects hidden
        // behind the lip). Keep one representative per stable, reachable,
        // unoccupied support pose as legitimate observation work. Collision
        // support identity is authoritative; geometry type is irrelevant.
        std::map<int64_t, int> novelSupportViewpoints;
        for (const NavCell &candidate : navCells)
        {
            if (!candidate.live
                || !inRange(candidate.id, 0, int(reachable.size()))
                || !reachable[size_t(candidate.id)]
                || !stableSupportPose(candidate.support, candidate.z))
                continue;
            const int64_t pose = supportPoseKey(candidate.support,
                                                candidate.z);
            if (visitedSupportPoses.count(pose))
                continue;
            auto previous = novelSupportViewpoints.find(pose);
            if (previous == novelSupportViewpoints.end()
                || distance2(observation.x, observation.y,
                             candidate.center.x, candidate.center.y)
                    < distance2(observation.x, observation.y,
                                navCells[size_t(previous->second)].center.x,
                                navCells[size_t(previous->second)].center.y))
                novelSupportViewpoints[pose] = candidate.id;
        }
        for (const auto &entry : novelSupportViewpoints)
        {
            const int cellId = entry.second;
            bool alreadyRepresented = false;
            for (llmapper::VisibilityFrontier &frontier : result)
                if (frontier.cell == cellId)
                {
                    frontier.approachCell = cellId;
                    frontier.reachable = true;
                    frontier.requiresOccupancy = true;
                    alreadyRepresented = true;
                    break;
                }
            if (alreadyRepresented)
                continue;
            llmapper::VisibilityFrontier frontier;
            frontier.cell = cellId;
            frontier.approachCell = cellId;
            frontier.reachable = true;
            frontier.informationGain = 1;
            frontier.requiresOccupancy = true;
            result.push_back(frontier);
        }
        visibilityFrontierTopologyRevision = navTopologyRevision;
        visibilityFrontierObservedCount = observedCells.size();
        visibilityFrontierFailureCount = navEdgeFailures.size();
        visibilityFrontierVisitedSupportCount = visitedSupportPoses.size();
        visibilityFrontierReachableSignature = reachableSignature;
        visibilityFrontierCache = result;
        return visibilityFrontierCache;
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
                // Build's sector label may change while Caleb remains on
                // the same physical floor (overlapping/ROR layers and
                // support transitions do this routinely).  The route
                // prerequisite is the physical near-side pose, not equality
                // with the wall owner's sector number.
                if (physicallyOnPortalNearSide(*crossing))
                {
                    goalSector = observation.sector;
                    goalZ = llmapper::capability::playerFloorZ();
                }
                else
                {
                    goalSector = crossing->from;
                    goalZ = crossing->fromFloorZ;
                }
            }
        }
        const int goal = navCellInSector(goalSector, anchorX, anchorY, goalZ);
        if (goal < 0)
            return false;
        const int goalArea = navCells[size_t(goal)].walkArea;
        const int probeSignature = llmapper::mixHash(
            llmapper::mixHash(objectiveWorkHash(currentObjective), navTopologyRevision),
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
                    const int usefulReach = std::max(llmapper::capability::playerStepHeight(),
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
                    const int usefulReach = std::max(llmapper::capability::playerStepHeight(),
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
        const int lifetime = currentObjective.startedTick >= 0
            ? observation.tick - currentObjective.startedTick : 0;
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
            objectiveProgressSectors.clear();
            objectiveProgressSectors.insert(observation.sector);
        }
        // Crossing into a sector not yet reached by this objective is real
        // return-route progress. Merely alternating between two already
        // visited sectors is a loop and must not keep the objective alive.
        bool progressed = objectiveProgressSectors.insert(observation.sector).second;
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
        }
        // A coarse transport route is executed one physical transition at a
        // time and may deliberately lead away from the final task.  Measure
        // progress toward that current portal/run-up pose as well as toward
        // the remote objective.  Without this, a five-second approach to a
        // valid gap could expire the parent task at the lip and release the
        // actor into fallback exploration over open air.
        if (currentObjective.routeStepWall >= 0
            && currentObjective.routeStepX != INT32_MIN
            && currentObjective.routeStepY != INT32_MIN
            && currentObjective.routeStepFrom == observation.sector)
        {
            const bool changed = objectiveProgressStepWall
                    != currentObjective.routeStepWall
                || objectiveProgressStepFrom != currentObjective.routeStepFrom
                || objectiveProgressStepTo != currentObjective.routeStepTo
                || objectiveProgressStepX != currentObjective.routeStepX
                || objectiveProgressStepY != currentObjective.routeStepY;
            const int stepDistance = distance2(
                observation.x, observation.y,
                currentObjective.routeStepX, currentObjective.routeStepY);
            if (changed)
            {
                objectiveProgressStepWall = currentObjective.routeStepWall;
                objectiveProgressStepFrom = currentObjective.routeStepFrom;
                objectiveProgressStepTo = currentObjective.routeStepTo;
                objectiveProgressStepX = currentObjective.routeStepX;
                objectiveProgressStepY = currentObjective.routeStepY;
                objectiveBestStepDistance2 = stepDistance;
                progressed = true;
            }
            else if (stepDistance + kObjectiveProgressEpsilon2
                     < objectiveBestStepDistance2)
            {
                objectiveBestStepDistance2 = stepDistance;
                progressed = true;
            }
        }
        // Keep the best remaining-step evidence across route reconstruction.
        // Resetting it whenever a route briefly cleared made every replay of
        // the same four-step cycle look like fresh progress.
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
        if (progressed && lifetime < kObjectiveHardTicks)
        {
            objectiveProgressSector = observation.sector;
            objectiveProgressTick = observation.tick;
            waitingSinceTick = -1;
            return false;
        }
        // The hard lifetime is global to the objective.  Genuine local
        // motion cannot renew it after the deadline: replaying a valid jump,
        // route prefix, or mechanism wait is still failure to accomplish the
        // parent world action.
        const int stalled = observation.tick - objectiveProgressTick;
        if (stalled < kObjectiveStallTicks && lifetime < kObjectiveHardTicks)
            return false;
        const int key = objectiveWorkHash(currentObjective);
        const char *reason = stalled >= kObjectiveStallTicks
            ? "no_progress_toward_objective" : "objective_time_budget";
        char detail[224];
        snprintf(detail, sizeof(detail),
                 "type=%d id=%d key=%d stalled_s=%d lifetime_s=%d sector=%d reason=%s",
                 int(currentObjective.type), currentObjective.id, key,
                 stalled / kTicsPerSec, lifetime / kTicsPerSec, observation.sector, reason);
        event("objective_budget_exhausted", detail);
        invalidateObjective(reason);
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
        objectiveProgressSectors.clear();
        objectiveProgressSectors.insert(observation.sector);
        objectiveProgressStepWall = -1;
        objectiveProgressStepFrom = -1;
        objectiveProgressStepTo = -1;
        objectiveProgressStepX = INT32_MIN;
        objectiveProgressStepY = INT32_MIN;
        objectiveBestStepDistance2 = INT32_MAX;
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

    // Abandon only the current physical plan.  Persistent work remains in
    // the ledger and may acquire a different route immediately; an executor
    // failure is never evidence that the world affordance ceased to exist.
    void invalidateObjective(const char *reason)
    {
        if (!currentObjective.active)
            return;
        const bool failedPrerequisite = suspendedObjectiveActive
            && currentObjective.type == kObjectiveInteraction;
        if (currentObjective.type == kObjectiveExpose)
        {
            // Give up on this particular viewpoint, not on looking around.
            // A route failure is not proof that the square was observed.
            // The attempt record annotates the same conserved ledger entry; it does
            // not manufacture observed space or delete the frontier.
            char detail[160];
            snprintf(detail, sizeof(detail), "sector=%d at=(%d,%d) reason=%s",
                     currentObjective.sector, currentObjective.x, currentObjective.y, reason);
            event("coverage_target_unreachable", detail);
            currentObjective = Objective{};
            objectiveProgressTick = -1;
            movementTargetActive = false;
            resetNavigation();
            return;
        }
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

    // MoveDude deliberately expands the player's GetZRange footprint by 16
    // after using the ordinary clipdist radius for horizontal ClipMove.
    // Support ownership must use that exact, distinct engine radius.  Using
    // playerClipRadius() here admitted edge poses which the live player was
    // still grounded on the neighbouring support at.
    int playerSupportRadius() const
    {
        return playerClipRadius() + 16;
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

    LocalWaypoint projectToSegment(int x, int y,
                                   int x1, int y1, int x2, int y2) const
    {
        const int64_t dx = int64_t(x2) - x1;
        const int64_t dy = int64_t(y2) - y1;
        const int64_t length2 = dx * dx + dy * dy;
        if (length2 <= 0)
            return LocalWaypoint(x1, y1);
        int64_t numerator = (int64_t(x) - x1) * dx
            + (int64_t(y) - y1) * dy;
        numerator = std::max<int64_t>(0, std::min(numerator, length2));
        return LocalWaypoint(
            x1 + int((dx * numerator + length2 / 2) / length2),
            y1 + int((dy * numerator + length2 / 2) / length2));
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
            completeObjective("investigation_found_interaction");
            return input;
        }
        if (nearby || selectNewInteraction(false))
        {
            event("blocked_frontier_action_found", "source=nearby_trigger_push");
            rememberInvestigatedBoundary(crossing, hit,
                                          "nearby_action_opportunity_found");
            completeObjective("investigation_found_interaction");
            return input;
        }
        rememberInvestigatedBoundary(crossing, hit, reason);
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
                invalidateObjective("interaction_not_known");
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
                // failed: acquiring the key should reconsider it immediately.
                invalidateObjective("missing_key_prerequisite");
                return GINPUT{};
            }
            const unsigned missingEffects =
                llmapper::effectRequirementSatisfied(
                    interaction.requiredEffects,
                    availableEffectCapabilities())
                ? unsigned(llmapper::kEffectNone)
                : interaction.requiredEffects;
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
            const bool routeContinuesBeyondTarget = !navRoute.empty()
                && navRouteIndex < navRoute.size()
                && navRoute.back().targetSector
                    != currentObjective.targetSector;
            if (observation.sector == currentObjective.targetSector
                && currentObjective.type == kObjectiveFrontier
                && !routeContinuesBeyondTarget)
            {
                completeObjective("destination_entered");
                return GINPUT{};
            }
            const bool exactCrossing = observation.sector == currentObjective.targetSector
                && lastTransitionTick == observation.tick;
            if (exactCrossing && currentObjective.type == kObjectiveFrontier
                && !routeContinuesBeyondTarget)
            {
                completeObjective("sector_transition");
                return GINPUT{};
            }
            const Portal *crossing = portalByWall(currentObjective.wall, currentObjective.sector,
                                                  currentObjective.targetSector);
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
            Portal steeringCrossing = *crossing;
            const bool onNearSide = physicallyOnPortalNearSide(*crossing);
            if (onNearSide)
                steeringCrossing.from = observation.sector;
            currentObjective.routeStepWall = crossing->wall;
            currentObjective.routeStepFrom = observation.sector;
            currentObjective.routeStepTo = onNearSide
                ? crossing->to : crossing->from;
            // Which of the two ways of reaching a frontier is in use, and
            // why.  A crossing that silently falls back to the planner is
            // how the bot came to stand still in front of an open door.
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
                return steerPortal(steeringCrossing);
            return navigateTo(crossing->x, crossing->y, crossing->z, crossing->from,
                              crossing->wall, crossing->capability);
        }
        if (currentObjective.type == kObjectiveExit)
        {
            setGoal("REACH_OBSERVED_EXIT", currentObjective.sector);
            return navigateTo(currentObjective.x, currentObjective.y,
                              currentObjective.z, currentObjective.sector,
                              -4, kTraversalUnknown);
        }
        if (currentObjective.type == kObjectiveExpose)
        {
            // The objective is one spatial visibility boundary, not all cells
            // owned by a Build sector. Standing on it is unnecessary once a
            // viewpoint has exposed it.
            if (!inRange(currentObjective.id, 0, int(navCells.size())))
            {
                invalidateObjective("visibility_cell_no_longer_exists");
                return GINPUT{};
            }
            const NavCell &unknown = navCells[size_t(currentObjective.id)];
            const PhysicalPoseKey handle = physicalPoseKey(unknown);
            if (currentObjective.requiresOccupancy)
            {
                const int64_t pose = supportPoseKey(unknown.support,
                                                    unknown.z);
                if (visitedSupportPoses.count(pose))
                {
                    char detail[128];
                    snprintf(detail, sizeof(detail), "cell=%d sector=%d",
                             currentObjective.id, currentObjective.sector);
                    event("visibility_viewpoint_occupied", detail);
                    completeObjective("observation_pose_occupied");
                    return GINPUT{};
                }
                return navigateTo(unknown.center.x, unknown.center.y,
                                  unknown.z, unknown.sector, -3,
                                  kTraversalUnknown, false,
                                  &unknown.support);
            }
            if (observedCells.count(handle))
            {
                char detail[128];
                snprintf(detail, sizeof(detail), "cell=%d sector=%d",
                         currentObjective.id, currentObjective.sector);
                event("visibility_frontier_resolved", detail);
                completeObjective("unknown_space_observed");
                return GINPUT{};
            }
            ensureNavTopology();
            std::vector<char> reachable;
            llmapper::markReachableNavCells(navCells, currentNavStartCell(),
                                            navEdgeFailures, 0, reachable);
            if (inRange(currentObjective.id, 0, int(reachable.size()))
                && reachable[size_t(currentObjective.id)])
                return navigateTo(unknown.center.x, unknown.center.y, unknown.z,
                                  unknown.sector, -3, kTraversalUnknown,
                                  false, &unknown.support);
            if (!inRange(currentObjective.approachCell, 0, int(navCells.size())))
            {
                invalidateObjective("visibility_approach_no_longer_exists");
                return GINPUT{};
            }
            const NavCell &approach = navCells[size_t(currentObjective.approachCell)];
            // The visibility proof belongs to this concrete sampled pose.
            // A normal navigation arrival radius is much too broad at an
            // occlusion boundary: stopping 150 units to one side can put the
            // same target behind the wall again.  Consume the pose closely
            // before judging the proof invalid.
            constexpr int kObservationPoseTolerance = 64;
            if (distance2(observation.x, observation.y,
                          approach.center.x, approach.center.y)
                    <= kObservationPoseTolerance * kObservationPoseTolerance)
            {
                // Reaching an observation pose does not authorize traversal
                // to the thing being observed.  If the engine visibility
                // proof no longer holds, discard this derived plan; the
                // unknown physical work remains and may acquire a different
                // pose after the world model changes.
                invalidateObjective("observation_pose_did_not_expose_target");
                return GINPUT{};
            }
            return navigateTo(approach.center.x, approach.center.y, approach.z,
                              approach.sector, -3, kTraversalUnknown, false,
                              &approach.support);
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
            if (!pickupUsefulNow(*object))
            {
                completeObjective("pickup_not_currently_useful");
                return GINPUT{};
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
        // A launched jump is an indivisible physical transaction.  Seeing a
        // new task or rebuilding the graph near the apex must not replace its
        // target and convert the remaining flight into ordinary steering.
        if ((jumpState == kJumpTakeoff || jumpState == kJumpAirborne)
            && movementTargetActive)
            return;
        if (movementTargetActive && movementTargetGoal == currentGoal
            && movementTargetX == x && movementTargetY == y
            && movementTargetSector == targetSector && movementTargetId == targetId
            && movementTargetCapability == capability)
            return;

        const bool samePhysicalJumpEdge = jumpState != kJumpInactive
            && capability == kTraversalJumpable
            && !navRoute.empty() && navRouteIndex < navRoute.size()
            && (navRoute[navRouteIndex].mode == kNavJump
                || navRoute[navRouteIndex].mode == kNavDrop)
            && navRoute[navRouteIndex].fromCell == jumpRouteFromCell
            && navRoute[navRouteIndex].toCell == jumpRouteToCell;
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
        if (!samePhysicalJumpEdge)
        {
            if (jumpState != kJumpInactive)
            {
                char detail[240];
                snprintf(detail, sizeof(detail),
                         "state=%d old_edge=%d->%d new_target=(%d,%d,s%d,id%d) capability=%d route_signature=%d navigation_signature=%d",
                         int(jumpState), jumpRouteFromCell, jumpRouteToCell,
                         x, y, targetSector, targetId, int(capability),
                         navRouteSignature, navigationOriginalSignature);
                event("jump_commitment_cancelled", detail);
            }
            jumpAttempts = 0;
            jumpState = kJumpInactive;
            jumpStateTick = -1;
            jumpRouteFromCell = -1;
            jumpRouteToCell = -1;
        }
        jumpTargetX = x;
        jumpTargetY = y;
        jumpTargetSector = targetSector;
        if (!samePhysicalJumpEdge)
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
        const NavRouteStep *pendingJumpStep = !navRoute.empty()
            && navRouteIndex < navRoute.size()
            && (navRoute[navRouteIndex].mode == kNavJump
                || navRoute[navRouteIndex].mode == kNavDrop)
            ? &navRoute[navRouteIndex] : nullptr;
        const bool pendingJumpLanded = pendingJumpStep
            && playerOccupiesSupportPose(pendingJumpStep->targetSupport,
                                         pendingJumpStep->targetZ)
            && observation.playerZVelocity == 0
            && gMe && gMe->pXSprite && gMe->pXSprite->height == 0
            && std::abs(llmapper::capability::playerFloorZ()
                        - pendingJumpStep->targetZ) <= llmapper::capability::playerStepHeight() / 2;
        if (horizontalProgress || sectorProgress)
        {
            // Horizontal closing and Build-sector membership are useful
            // motion signals, but neither proves a graph jump succeeded.  A
            // flying body can satisfy both on its way to the wrong floor.
            // Retain the bounded attempt count until the concrete physical
            // destination pose is grounded.
            if (jumpAttempts > 0 && (!pendingJumpStep || pendingJumpLanded))
            {
                char detail[128];
                snprintf(detail, sizeof(detail),
                         "target=%d dx=%d dy=%d dz=%d attempts=%d reason=%s",
                         movementTargetId, observation.x - targetLastX, observation.y - targetLastY,
                         observation.z - targetLastZ, jumpAttempts,
                         pendingJumpLanded ? "physical_pose_established"
                                           : (horizontalProgress ? "closed_on_target"
                                                                 : "entered_target_sector"));
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

    MovementProbe probeMovement(int startX, int startY, int startZ, int startSector,
                                int targetX, int targetY, int targetSector,
                                int tolerance, bool crouched = false) const
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
        const int32_t xvect = int32_t(std::max<int64_t>(INT32_MIN + 1,
            std::min<int64_t>(INT32_MAX, dx * (int64_t(1) << 14))));
        const int32_t yvect = int32_t(std::max<int64_t>(INT32_MIN + 1,
            std::min<int64_t>(INT32_MAX, dy * (int64_t(1) << 14))));
        const PlayerCollisionShape liveShape = livePlayerCollisionShape();
        const PlayerCollisionShape &shape = crouched
                && gObservedCrouchShape.known
            ? gObservedCrouchShape : liveShape;
        const int radius = shape.known ? shape.radius : 128;
        int ceilingDistance = 0;
        int floorDistance = 0;
        if (shape.known)
        {
            ceilingDistance = shape.ceilingDistance;
            floorDistance = shape.floorDistance;
        }
        else
            playerCollisionDistances(ceilingDistance, floorDistance);
        result.hit = clipmove(&position, &sectorNumber, xvect, yvect, radius,
                              ceilingDistance, floorDistance, CLIPMASK0);
        result.x = position.x;
        result.y = position.y;
        result.sector = sectorNumber;
        if ((result.hit & 0xc000) == 0x8000)
            result.wall = result.hit & 0x3fff;
        else if ((result.hit & 0xc000) == 0xc000)
            result.sprite = result.hit & 0x3fff;
        const int remaining = distance2(position.x, position.y, targetX, targetY);
        // Reaching a physical XY pose does not require the engine to choose
        // one particular Build owner when polygons overlap or a partition is
        // thinner than the player cylinder.  Accept equivalent target-sector
        // floor geometry at the reached point; distinct height layers remain
        // separate because their physical support planes differ.
        bool equivalentTargetPose = sectorNumber == targetSector;
        if (!equivalentTargetPose && remaining <= tolerance * tolerance
            && inside(position.x, position.y, targetSector) == 1)
        {
            int liveCeiling = 0, liveCeilingHit = 0;
            int liveFloor = 0, liveFloorHit = 0;
            GetZRangeAtXYZ(position.x, position.y, position.z, sectorNumber,
                           &liveCeiling, &liveCeilingHit,
                           &liveFloor, &liveFloorHit,
                           radius, CLIPMASK0,
                           PARALLAXCLIP_CEILING | PARALLAXCLIP_FLOOR);
            const int targetFloor = getflorzofslope(
                targetSector, position.x, position.y);
            equivalentTargetPose = std::abs(liveFloor - targetFloor)
                <= llmapper::capability::playerStepHeight() / 2;
        }
        result.reachable = equivalentTargetPose
            && remaining <= tolerance * tolerance;
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

        // Passive two-sided Build partitions are transparent to the physical
        // model. Expand the nav mesh through their connected component even
        // before Caleb crosses each partition; live NavLink construction
        // still decides whether any particular floor/clearance pose is
        // traversable. This makes one corridor and the same corridor split
        // into thirty sectors produce the same semantic visibility boundary.
        bool passiveExpanded = true;
        while (passiveExpanded)
        {
            passiveExpanded = false;
            const std::vector<int> known(sectors.begin(), sectors.end());
            for (int sectorId : known)
            {
                if (!inRange(sectorId, 0, numsectors))
                    continue;
                const sectortype &record = sector[sectorId];
                for (int i = 0; i < record.wallnum; ++i)
                {
                    const int wallId = record.wallptr + i;
                    if (!inRange(wallId, 0, numwalls)
                        || (wall[wallId].cstat & 1)
                        || !inRange(wall[wallId].nextsector, 0, numsectors))
                        continue;
                    if (sectors.insert(wall[wallId].nextsector).second)
                        passiveExpanded = true;
                }
            }
        }

        // A Blood stacked-room marker is authoritative topology in exactly
        // the same sense as a two-sided wall.  Pull its paired layer into the
        // mesh as soon as either side is known; waiting until the player has
        // accidentally crossed it makes deliberate ROR travel impossible.
        bool expanded = true;
        while (expanded)
        {
            expanded = false;
            const std::vector<int> known(sectors.begin(), sectors.end());
            for (int sectorId : known)
            {
                if (!inRange(sectorId, 0, numsectors))
                    continue;
                const int links[2] = { gUpperLink[sectorId],
                                       gLowerLink[sectorId] };
                for (int linkSprite : links)
                {
                    if (!inRange(linkSprite, 0, kMaxSprites))
                        continue;
                    const int owner = sprite[linkSprite].owner;
                    if (!inRange(owner, 0, kMaxSprites)
                        || !inRange(sprite[owner].sectnum, 0, numsectors))
                        continue;
                    if (sectors.insert(sprite[owner].sectnum).second)
                        expanded = true;
                }
            }
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
            const int links[2] = { gUpperLink[sectorId],
                                   gLowerLink[sectorId] };
            for (int linkSprite : links)
            {
                signature = llmapper::mixHash(signature, linkSprite);
                if (!inRange(linkSprite, 0, kMaxSprites))
                    continue;
                signature = llmapper::mixHash(signature,
                                               sprite[linkSprite].owner);
                signature = llmapper::mixHash(signature,
                                               sprite[linkSprite].x);
                signature = llmapper::mixHash(signature,
                                               sprite[linkSprite].y);
            }
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
                    LocalWaypoint gateway = {}, bool hasGateway = false,
                    int transition = -1,
                    LocalWaypoint takeoff = {}, bool hasTakeoff = false,
                    bool dynamic = false,
                    int airControl = 0, int airFrames = 0,
                    bool hasAirControl = false)
    {
        if (from < 0 || to < 0 || from == to)
            return;
        const SupportRef &sourceSupport = navCells[size_t(from)].support;
        const SupportRef &targetSupport = navCells[size_t(to)].support;
        // A damaging contact may still be the player's starting support, so
        // retain travel within it and every edge leaving it.  What navigation
        // must not do is choose to enter a distinct damaging support.  This
        // is directed for the same reason a drop is directed.
        if (targetSupport != sourceSupport
            && engineSupportDamagesPlayer(targetSupport))
            return;
        NavCell &cell = navCells[from];
        for (NavLink &link : cell.links)
            if (link.target == to && link.wall == wallId)
            {
                // Dynamic links are added from broad physical inference
                // first and measured portal evidence second.  The later,
                // concrete observation owns mode and gateway when both name
                // the same crossing; returning here used to leave the lossy
                // inferred edge authoritative.
                link.mode = mode;
                link.gateway = gateway;
                link.hasGateway = hasGateway;
                link.takeoff = takeoff;
                link.hasTakeoff = hasTakeoff;
                link.transition = transition;
                link.airControl = airControl;
                link.airFrames = airFrames;
                link.hasAirControl = hasAirControl;
                link.dynamic = dynamic;
                return;
            }
        NavLink link;
        link.target = to;
        link.mode = mode;
        link.wall = wallId;
        link.gateway = gateway;
        link.hasGateway = hasGateway;
        link.takeoff = takeoff;
        link.hasTakeoff = hasTakeoff;
        link.transition = transition;
        link.airControl = airControl;
        link.airFrames = airFrames;
        link.hasAirControl = hasAirControl;
        link.dynamic = dynamic;
        cell.links.push_back(link);
    }

    int addRorNavLinks()
    {
        int linked = 0;
        for (int upperSector = 0; upperSector < numsectors; ++upperSector)
        {
            const int upperSprite = gUpperLink[upperSector];
            if (!inRange(upperSprite, 0, kMaxSprites))
                continue;
            const int upperType = sprite[upperSprite].type;
            // Water/goo links require swimming semantics and remain deferred.
            // Plain links and stacked-room links are freely traversable
            // vertical engine portals.
            if (upperType != kMarkerUpLink && upperType != kMarkerUpStack)
                continue;
            const int lowerSprite = sprite[upperSprite].owner;
            if (!inRange(lowerSprite, 0, kMaxSprites))
                continue;
            const int lowerType = sprite[lowerSprite].type;
            if (lowerType != kMarkerLowLink && lowerType != kMarkerLowStack)
                continue;
            const int lowerSector = sprite[lowerSprite].sectnum;
            if (!inRange(lowerSector, 0, numsectors))
                continue;
            linked += llmapper::linkTranslatedNavLayers(
                navCells, upperSector, lowerSector,
                sprite[lowerSprite].x - sprite[upperSprite].x,
                sprite[lowerSprite].y - sprite[upperSprite].y,
                2 << kNavGridShift, kNavRorTransitionEdge);
        }
        if (linked > 0)
        {
            char detail[96];
            snprintf(detail, sizeof(detail),
                     "links=%d source=engine_link_markers", linked);
            event("nav_ror_links", detail);
        }
        return linked;
    }

    // Link concrete standable poses across every currently open two-sided
    // engine boundary in the known physical mesh.  Requiring a Portal record
    // first makes connectivity depend on having occupied the wall's owner
    // sector: a fully visible staircase then consists of disconnected sector
    // islands until Caleb gratuitously visits every step.  Build ownership is
    // irrelevant here; the live wall, clearance and support heights are the
    // physical crossing.
    int addPhysicalBoundaryLinks()
    {
        const int boundaryReach = 2 << kNavGridShift;
        const int64_t near2 = int64_t(boundaryReach) * boundaryReach;
        // `inside` accepts points on a polygon boundary, but such a sample is
        // not an occupiable pose for the player's collision cylinder.  A
        // jump aimed at it is clipped back onto the source side even though
        // the graph calls the endpoint part of the destination sector.  Keep
        // concrete takeoff and landing cells at least one body radius inside
        // their respective physical regions.
        // One real body radius is the engine's occupiable inset.  The former
        // extra quarter-cell margin disconnected corridors and stair treads
        // which fit Caleb exactly even though every endpoint had already been
        // accepted by the engine support/collision predicates.
        const int minimumInset = playerClipRadius();
        const int64_t minimumInset2 = int64_t(minimumInset) * minimumInset;
        const int maximumSpan = 3 << kNavGridShift;
        const int64_t maximumSpan2 = int64_t(maximumSpan) * maximumSpan;
        const int standing = playerStandingClearance();
        const int crouched = playerCrouchClearance();
        int linked = 0;
        auto directedMode = [&](const NavCell &from, const NavCell &to) {
            const int delta = to.z - from.z;
            const int ceiling = std::max(
                getceilzofslope(from.sector, from.center.x, from.center.y),
                getceilzofslope(to.sector, to.center.x, to.center.y));
            const int clearance = std::min(from.z, to.z) - ceiling;
            if (clearance < crouched)
                return kNavBlocked;
            if (std::abs(delta) <= llmapper::capability::playerStepHeight())
                return clearance < standing ? kNavCrouch
                    : delta == 0 ? kNavWalk : kNavStep;
            if (clearance < standing)
                return kNavBlocked;
            // Discontinuous vertical transitions are derived by the
            // support-to-support validator.  A wall boundary alone cannot
            // prove a takeoff, landing, retained support or safe drop.
            return kNavBlocked;
        };

        for (int wallId = 0; wallId < numwalls; ++wallId)
        {
            const walltype &boundary = wall[wallId];
            if (!inRange(boundary.nextsector, 0, numsectors)
                || !inRange(boundary.nextwall, 0, numwalls)
                || wallId > boundary.nextwall
                || !inRange(boundary.point2, 0, numwalls)
                || (boundary.cstat & CSTAT_WALL_BLOCK)
                || (wall[boundary.nextwall].cstat & CSTAT_WALL_BLOCK))
                continue;
            const int owner = wallOwnerSector(wallId);
            if (!inRange(owner, 0, numsectors))
                continue;
            const walltype &end = wall[boundary.point2];
            if (distance2(boundary.x, boundary.y, end.x, end.y)
                < playerPassageWidth() * playerPassageWidth())
                continue;

            std::vector<int> fromCells;
            std::vector<int> toCells;
            for (const NavCell &cell : navCells)
            {
                if (cell.sector != owner
                    && cell.sector != boundary.nextsector)
                    continue;
                const int64_t boundaryDistance = dist2ToSegment(
                    cell.center.x, cell.center.y,
                    boundary.x, boundary.y, end.x, end.y);
                if (boundaryDistance > near2
                    || boundaryDistance < minimumInset2)
                    continue;
                (cell.sector == owner ? fromCells : toCells).push_back(cell.id);
            }

            const int linksBeforeBoundary = linked;
            int spanRejected = 0;
            int modeRejected = 0;
            int collisionRejected = 0;

            auto connect = [&](const std::vector<int> &sources,
                               const std::vector<int> &destinations,
                               int crossingWall) {
                for (int source : sources)
                {
                    const NavCell &from = navCells[size_t(source)];
                    int best = -1;
                    int64_t bestCost = INT64_MAX;
                    NavEdgeMode bestMode = kNavBlocked;
                    for (int destination : destinations)
                    {
                        const NavCell &to = navCells[size_t(destination)];
                        const int64_t span = distance2(
                            from.center.x, from.center.y,
                            to.center.x, to.center.y);
                        if (span > maximumSpan2)
                        {
                            ++spanRejected;
                            continue;
                        }
                        const NavEdgeMode mode = directedMode(from, to);
                        if (!llmapper::traversableMode(mode))
                        {
                            ++modeRejected;
                            continue;
                        }
                        const int64_t cost = span
                            + int64_t(std::abs(to.z - from.z)) * 16;
                        if (cost < bestCost)
                        {
                            best = destination;
                            bestCost = cost;
                            bestMode = mode;
                        }
                    }
                    if (best < 0)
                        continue;
                    const NavCell &to = navCells[size_t(best)];
                    // Polygon adjacency is only a candidate. In overlapping
                    // Build geometry, require the real horizontal collision
                    // transition for ordinary walking/stepping links. A
                    // ballistic edge is verified by its landing support.
                    if (bestMode == kNavWalk || bestMode == kNavStep
                        || bestMode == kNavCrouch)
                    {
                        const bool crouched = bestMode == kNavCrouch
                            || from.crouchOnly || to.crouchOnly;
                        const int originZ = from.z
                            - (crouched && gObservedCrouchShape.known
                               ? gObservedCrouchShape.footOffset
                               : llmapper::capability::playerFootOffset());
                        const MovementProbe physical = probeMovement(
                            from.center.x, from.center.y,
                            originZ, from.sector,
                            to.center.x, to.center.y, to.sector,
                            std::max(256, playerClipRadius()), crouched);
                        if (!physical.reachable)
                        {
                            ++collisionRejected;
                            continue;
                        }
                    }
                    // A boundary is not a player pose.  Preserve its real
                    // physical location as the crossing constraint instead
                    // of relabelling the midpoint of two standable cells as
                    // a gateway.  The latter can be a full grid cell beside
                    // a narrow doorway and makes an otherwise valid route
                    // execute parallel to the opening.
                    const walltype &crossing = wall[crossingWall];
                    const walltype &crossingEnd = wall[crossing.point2];
                    const LocalWaypoint between(
                        (from.center.x + to.center.x) / 2,
                        (from.center.y + to.center.y) / 2);
                    const LocalWaypoint gateway = projectToSegment(
                        between.x, between.y,
                        crossing.x, crossing.y,
                        crossingEnd.x, crossingEnd.y);
                    addNavLink(source, best, bestMode, crossingWall,
                               gateway, true, -1, {}, false, true);
                    ++linked;
                }
            };
            connect(fromCells, toCells, wallId);
            connect(toCells, fromCells, wallId);
            if (linked == linksBeforeBoundary)
            {
                char detail[224];
                snprintf(detail, sizeof(detail),
                         "wall=%d from_sector=%d to_sector=%d from_poses=%u to_poses=%u span_rejected=%d mode_rejected=%d collision_rejected=%d",
                         wallId, owner, int(boundary.nextsector),
                         unsigned(fromCells.size()), unsigned(toCells.size()),
                         spanRejected, modeRejected, collisionRejected);
                event("nav_boundary_unlinked", detail);
            }
        }
        return linked;
    }

    // A Build partition may be thinner than the player's collision radius,
    // so no valid player-centre pose can belong to it.  Requiring one NavCell
    // per sector then disconnects two ordinary pieces of the same physical
    // corridor.  Join nearby concrete poses directly when NBlood's live
    // ClipMove accepts the complete transition.  Bucketing keeps this local;
    // it is not a second coarse sector graph and it grants no capability that
    // the engine did not just validate.
    int addPhysicalLocalLinks()
    {
        const int maximumSpan = 3 << kNavGridShift;
        const int64_t maximumSpan2 = int64_t(maximumSpan) * maximumSpan;
        const int bucketRadius = maximumSpan >> kNavGridShift;
        std::map<std::pair<int, int>, std::vector<int> > buckets;
        for (const NavCell &cell : navCells)
            if (cell.live)
                buckets[std::make_pair(cell.gx, cell.gy)].push_back(cell.id);

        int linked = 0;
        for (const NavCell &from : navCells)
        {
            if (!from.live)
                continue;
            for (int gx = from.gx - bucketRadius;
                 gx <= from.gx + bucketRadius; ++gx)
                for (int gy = from.gy - bucketRadius;
                     gy <= from.gy + bucketRadius; ++gy)
                {
                    auto bucket = buckets.find(std::make_pair(gx, gy));
                    if (bucket == buckets.end())
                        continue;
                    for (int target : bucket->second)
                    {
                        if (target == from.id)
                            continue;
                        const NavCell &to = navCells[size_t(target)];
                        // Ordinary same-sector mesh adjacency is already
                        // represented once.  This link exists solely to make
                        // engine partitions non-authoritative.
                        if (to.sector == from.sector
                            || distance2(from.center.x, from.center.y,
                                         to.center.x, to.center.y)
                                > maximumSpan2)
                            continue;
                        const int delta = to.z - from.z;
                        // This index repairs a missing pose inside a thin
                        // Build partition; it is not a second step planner.
                        // A different support height is a real physical
                        // transition and must be represented by an observed
                        // boundary or a separately validated support edge.
                        // Letting this shortcut span the full player step
                        // envelope connected unrelated stable elevator
                        // layers through corners which ClipMove happened to
                        // approach within the endpoint tolerance.
                        if (delta != 0)
                            continue;
                        MovementProbe physical = probeMovement(
                            from.center.x, from.center.y,
                            playerOriginAtSupport(from.z), from.sector,
                            to.center.x, to.center.y, to.sector, 64);
                        NavEdgeMode mode = kNavWalk;
                        if (!physical.reachable && gObservedCrouchShape.known)
                        {
                            physical = probeMovement(
                                from.center.x, from.center.y,
                                from.z - gObservedCrouchShape.footOffset,
                                from.sector, to.center.x, to.center.y,
                                to.sector, 64, true);
                            if (physical.reachable)
                                mode = kNavCrouch;
                        }
                        if (!physical.reachable)
                            continue;
                        const size_t before = navCells[size_t(from.id)].links.size();
                        addNavLink(from.id, target, mode, -1,
                                   {}, false, -1, {}, false, true);
                        if (navCells[size_t(from.id)].links.size() > before)
                            ++linked;
                    }
                }
        }
        return linked;
    }

    void refreshDynamicNavLinks()
    {
        // Intra-support mesh connectivity is static.  Every edge accepted by
        // live collision is rebuilt, regardless of whether one Build wall
        // happens to name the transition.
        for (NavCell &cell : navCells)
            cell.links.erase(std::remove_if(cell.links.begin(), cell.links.end(),
                                            [](const NavLink &link)
                                            { return link.dynamic; }), cell.links.end());
        const int physicalLinks = addPhysicalBoundaryLinks();
        const int localLinks = addPhysicalLocalLinks();
        for (auto &entry : knownGraph)
            for (Portal &portal : entry.second)
            {
                // Portal identity and affordance are persistent knowledge;
                // its landing pose is live world state.  A remote mechanism
                // can move the far support while this boundary is outside the
                // current observation.  Refresh the physical measurements so
                // a remembered crossing reconnects to the current floor, not
                // an obsolete stable endpoint.
                if (inRange(portal.from, 0, numsectors)
                    && inRange(portal.to, 0, numsectors))
                {
                    auto movingFloor = [](int sectorId) {
                        if (!inRange(sectorId, 0, numsectors))
                            return false;
                        const int extra = sector[sectorId].extra;
                        return inRange(extra, 1, kMaxXSectors)
                            && xsector[extra].offFloorZ
                                != xsector[extra].onFloorZ;
                    };
                    // A portal may have been measured while standing on a
                    // sprite/voxel support whose height intentionally differs
                    // from its containing sector floor. Preserve that physical
                    // support measurement. Only a sector with alternate floor
                    // states owns a floor endpoint that can become stale.
                    if (movingFloor(portal.from))
                    {
                        portal.fromFloorZ = getflorzofslope(
                            portal.from, portal.x, portal.y);
                        portal.fromCeilingZ = getceilzofslope(
                            portal.from, portal.x, portal.y);
                    }
                    if (movingFloor(portal.to))
                    {
                        portal.toFloorZ = getflorzofslope(
                            portal.to, portal.x, portal.y);
                        portal.toCeilingZ = getceilzofslope(
                            portal.to, portal.x, portal.y);
                    }
                    portal.floorZ = portal.fromFloorZ;
                    portal.ceilingZ = portal.fromCeilingZ;
                    portal.floorDelta = portal.toFloorZ - portal.fromFloorZ;
                    portal.clearance = std::min(
                        portal.fromFloorZ - portal.fromCeilingZ,
                        portal.toFloorZ - portal.toCeilingZ);
                    const bool blocked = inRange(portal.wall, 0, numwalls)
                        && (wall[portal.wall].cstat & CSTAT_WALL_BLOCK);
                    const bool width = portal.openingWidth
                        >= playerPassageWidth();
                    const bool standing = portal.clearance
                        >= playerStandingClearance();
                    const bool crouching = portal.clearance
                        >= playerCrouchClearance();
                    portal.walkable = !blocked && width && standing
                        && std::abs(portal.floorDelta) <= llmapper::capability::playerStepHeight();
                    portal.crouchable = !blocked && width && !standing
                        && crouching
                        && std::abs(portal.floorDelta) <= llmapper::capability::playerStepHeight();
                    portal.jumpable = !blocked && width && standing
                        && portal.floorDelta < -llmapper::capability::playerStepHeight()
                        && -portal.floorDelta <= playerJumpRiseLimit();
                    portal.dropSafe = !blocked && width && standing
                        && portal.floorDelta > llmapper::capability::playerStepHeight()
                        && portal.floorDelta <= playerReversibleDrop();
                    portal.traversable = portal.walkable
                        || portal.crouchable || portal.jumpable
                        || portal.dropSafe;
                }
                // Portal records are persistent observation and affordance
                // knowledge. They are not a second executable graph.  Live
                // collision-derived boundaries above are the sole owner of
                // WALK/STEP/CROUCH links; ballistic support links are proved
                // separately by the same player-physics transition model.
            }
        llmapper::assignWalkAreas(navCells);
        char detail[96];
        snprintf(detail, sizeof(detail),
                 "boundary_links=%d local_links=%d source=live_collision",
                 physicalLinks, localLinks);
        event("nav_portal_links", detail);
    }

    static int64_t navCellHandle(int sectorId, int gx, int gy)
    {
        return (int64_t(sectorId) << 42) ^ (int64_t(gx & 0xfffff) << 21)
            ^ int64_t(gy & 0xfffff);
    }

    static PhysicalPoseKey physicalPoseKey(const NavCell &cell)
    {
        return std::make_tuple(cell.sector, cell.gx, cell.gy, cell.z,
                               int(cell.support.kind), cell.support.id);
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

    // Horizontal distance a jump with one concrete launch velocity can cover
    // by the time the feet descend through a destination support height. The
    // broad phase and the retained-landing validator therefore use the same
    // Blood motion model. A zero launch velocity describes a standing jump;
    // a run-up uses the velocity produced by Blood's own acceleration/drag.
    // Collision and availability of the required run-up are proved
    // separately before an edge is published.
    int jumpReachForDelta(int destinationFloorDelta, int launchVelocity) const
    {
        if (!gMe)
            return 0;
        const auto key = std::make_pair(destinationFloorDelta, launchVelocity);
        const auto cached = jumpReachCache.find(key);
        if (cached != jumpReachCache.end())
            return cached->second;
        const POSTURE &stand = gMe->pPosture[gMe->lifeMode][kPostureStand];
        const int foot = llmapper::capability::playerFootOffset();
        llmapper::capability::MotionState state;
        state.z = -foot;
        state.xvel = launchVelocity;
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
        // Once the footprint overlaps a higher receiver, ClipMove may hand
        // the body to it within the real curb envelope. Clamp that handoff to
        // the takeoff plane: adding the full allowance below the source made
        // modest rises mathematically unreachable because the source floor
        // correctly prevents the body from ever descending that far.
        const int contactDelta = destinationFloorDelta < 0
            ? std::min(0, destinationFloorDelta
                           + llmapper::capability::playerStepHeight())
            : destinationFloorDelta;
        bool reachedDestinationHeight = contactDelta >= 0;
        for (int frame = 1; frame <= 240; ++frame)
        {
            const bool falling = state.zvel > 0;
            const int simulationFloor = frame == 1
                ? 0 : std::max(0, destinationFloorDelta);
            llmapper::capability::stepPlayerMotion(
                state, kFullThrottle, 0, simulationFloor,
                stand.frontAccel, foot,
                llmapper::capability::playerAirDrag());
            if (state.z + foot <= contactDelta)
                reachedDestinationHeight = true;
            if (falling && reachedDestinationHeight
                && state.z + foot >= contactDelta)
            {
                const int reach = std::abs(state.x);
                jumpReachCache[key] = reach;
                return reach;
            }
        }
        jumpReachCache[key] = 0;
        return 0;
    }

    int standingJumpReachForDelta(int destinationFloorDelta) const
    {
        return jumpReachForDelta(destinationFloorDelta, 0);
    }

    int runningJumpReachForDelta(int destinationFloorDelta) const
    {
        const int runUp = llmapper::capability::playerRunUpDistance();
        const int velocity =
            llmapper::capability::playerRunVelocityForDistance(runUp);
        return jumpReachForDelta(destinationFloorDelta, velocity);
    }

    // Prove a concrete standing jump between two graph poses with the same
    // player-motion arithmetic used by Blood, then ask GetZRange whether the
    // descending body is actually owned by the requested landing support.
    // A scalar reach envelope is useful for spatial pruning, but it cannot
    // authorize an edge: it says nothing about where the target support
    // begins or whether Caleb can remain on it after touchdown.
    struct AirTransitionAudit
    {
        int samples = 0;
        int noContact = 0;
        int supportMiss = 0;
        int poseMiss = 0;
        int retentionMiss = 0;
        int blockedFrames = 0;
        int bestRetention = INT32_MIN;
        int bestFromX = 0;
        int bestFromY = 0;
        int bestToX = 0;
        int bestToY = 0;
        int bestLandX = 0;
        int bestLandY = 0;
        int bestSpeed = 0;
        int bestInput = 0;
        int bestLaunchVelocity = 0;
    };

    bool retainedAirTransition(const NavCell &from, const NavCell &to,
                               int verticalVelocity,
                               int launchVelocity = 0,
                               AirTransitionAudit *audit = nullptr,
                               int *validatedControl = nullptr,
                               int *validatedFrames = nullptr) const
    {
        if (!gMe || !gMe->pSprite || from.support == to.support)
            return false;
        const POSTURE &stand = gMe->pPosture[gMe->lifeMode][kPostureStand];
        const int foot = llmapper::capability::playerFootOffset();
        const int angle = getangle(to.center.x - from.center.x,
                                   to.center.y - from.center.y);
        const int poseTolerance = (1 << kNavGridShift) + playerClipRadius();
        int ceilingDistance = 0;
        int floorDistance = 0;
        playerCollisionDistances(ceilingDistance, floorDistance);
        // Sample the control lattice once while creating the executable
        // physical edge. Blood still gives the player horizontal input
        // authority through this jump envelope. In particular, a running jump onto a
        // narrow support may require counter-thrust during flight; validating
        // only positive throttle made the graph reject transitions which the
        // executor and the real player can retain.  A standing launch cannot
        // benefit from accelerating away from its target, so keep its half
        // lattice while including neutral input.
        const int controlSamples = launchVelocity > 0 ? 32 : 16;
        for (int sample = 0; sample <= controlSamples; ++sample)
        {
            if (audit)
                ++audit->samples;
            const int forward = launchVelocity > 0
                ? -kFullThrottle
                    + (2 * kFullThrottle * sample) / controlSamples
                : kFullThrottle * sample / controlSamples;
            llmapper::capability::MotionState state;
            state.x = from.center.x;
            state.y = from.center.y;
            state.z = from.z - foot;
            state.xvel = mulscale30(launchVelocity, Cos(angle));
            state.yvel = mulscale30(launchVelocity, Sin(angle));
            state.zvel = verticalVelocity;
            int16_t stateSector = int16_t(from.sector);
            // Once the airborne footprint overlaps a higher receiver,
            // ClipMove/GetZRange can establish that floor within the same
            // curb envelope used by grounded movement. The scalar broad
            // phase still measures rise against the real support plane; this
            // retained replay models the engine's collision handoff.
            const int contactZ = to.z < from.z
                ? std::min(from.z,
                           to.z + llmapper::capability::playerStepHeight())
                : to.z;
            bool clearedTarget = state.z + foot < contactZ;
            for (int frame = 1; frame <= 240; ++frame)
            {
                const llmapper::capability::MotionState previous = state;
                const bool falling = state.zvel > 0;
                // The engine does not stop supporting the player merely
                // because a planner has named a lower destination.  A DROP
                // begins only after horizontal motion carries the real hull
                // off its source support (a jump impulse can of course lift
                // it first).  Switching to the deeper floor after one frame
                // manufactured zero-motion drops straight through solid
                // sprites: planning accepted them, while live Blood kept the
                // player standing on the sprite forever.
                int liveSourceZ = 0;
                const bool sourceStillSupports = engineSupportZAt(
                        from.support, state.x, state.y, liveSourceZ)
                    && liveSourceZ == from.z
                    && engineHasSupportAt(
                        from.support, from.sector, state.x, state.y, from.z,
                        playerSupportRadius());
                const int simulationFloor = sourceStillSupports
                    ? from.z : std::max(from.z, to.z);
                llmapper::capability::stepPlayerMotion(
                    state, forward, angle, simulationFloor,
                    stand.frontAccel, foot,
                    llmapper::capability::playerAirDrag());
                // MoveDude resolves the horizontal part of every airborne
                // frame through clipmove at the body's current Z.  Replaying
                // only the ballistic arithmetic treats walls as absent;
                // rejecting the whole arc with one ground-level clipmove does
                // the opposite and mistakes a jumpable floor discontinuity
                // for an infinitely tall wall.  Apply the engine predicate at
                // the same phase and height as the real player instead.
                vec3_t clipped = { previous.x, previous.y, previous.z };
                const int64_t dx = int64_t(state.x) - previous.x;
                const int64_t dy = int64_t(state.y) - previous.y;
                const int32_t xvect = int32_t(std::max<int64_t>(
                    INT32_MIN + 1, std::min<int64_t>(
                        INT32_MAX, dx * (int64_t(1) << 14))));
                const int32_t yvect = int32_t(std::max<int64_t>(
                    INT32_MIN + 1, std::min<int64_t>(
                        INT32_MAX, dy * (int64_t(1) << 14))));
                const int moveHit = clipmove(
                    &clipped, &stateSector, xvect, yvect,
                    playerClipRadius(), ceilingDistance, floorDistance,
                    CLIPMASK0);
                if (audit && (std::abs(clipped.x - previous.x)
                              + std::abs(clipped.y - previous.y)) * 4
                                 < std::abs(state.x - previous.x)
                                   + std::abs(state.y - previous.y))
                    ++audit->blockedFrames;
                // The transition executor holds one validated control input;
                // it has no separate obstacle-avoidance path while airborne.
                // Therefore a wall or unrelated sprite returned by the same
                // ClipMove predicate is a failed transition, not a trajectory
                // that may continue through the collision.  Contact with an
                // endpoint support is allowed because it can be the physical
                // landing edge itself.
                const bool endpointSpriteHit = (moveHit & 0xc000) == 0xc000
                    && (((from.support.kind == kSupportSpriteFloor)
                         && (moveHit & 0x3fff) == from.support.id)
                        || ((to.support.kind == kSupportSpriteFloor)
                            && (moveHit & 0x3fff) == to.support.id));
                if (moveHit && !endpointSpriteHit)
                    break;
                state.x = clipped.x;
                state.y = clipped.y;
                if (state.z + foot < contactZ)
                    clearedTarget = true;
                if (!falling || !clearedTarget
                    || state.z + foot < contactZ)
                    continue;

                // MoveDude resolves vertical support after horizontal
                // ClipMove.  Reproduce that ordering and accept only the
                // first surface the player's real GetZRange footprint would
                // land on.  Merely proving that a deeper requested floor
                // exists at the same XY allowed arcs to pass through a solid
                // floor sprite and publish a transition the live player
                // could never execute.
                int ceilingZ = 0, ceilingHit = 0;
                int floorZ = 0, floorHit = 0;
                GetZRangeAtXYZ(state.x, state.y, state.z, stateSector,
                               &ceilingZ, &ceilingHit, &floorZ, &floorHit,
                               playerSupportRadius(), CLIPMASK0,
                               PARALLAXCLIP_CEILING | PARALLAXCLIP_FLOOR);
                if (state.z + foot < floorZ)
                    continue;
                const bool requestedSupport = to.support.kind
                        == kSupportSpriteFloor
                    ? floorHit == (0xc000 | to.support.id)
                    : (floorHit & 0xc000) == 0x4000;
                if (!requestedSupport || floorZ != to.z)
                {
                    if (audit)
                        ++audit->supportMiss;
                    break;
                }
                if (distance2(state.x, state.y, to.center.x, to.center.y)
                    > poseTolerance * poseTolerance)
                {
                    if (audit)
                        ++audit->poseMiss;
                    break;
                }
                const int speed = approxDist(state.xvel, state.yvel) >> 12;
                int margin = to.support.kind == kSupportSpriteFloor
                    ? spriteSupportClearance(
                        to.sector, to.support.id, state.x, state.y,
                        playerSupportRadius())
                    : to.clearance;
                const int retention = margin
                    - llmapper::capability::playerCoastDistance(speed);
                if (audit && retention > audit->bestRetention)
                {
                    audit->bestRetention = retention;
                    audit->bestFromX = from.center.x;
                    audit->bestFromY = from.center.y;
                    audit->bestToX = to.center.x;
                    audit->bestToY = to.center.y;
                    audit->bestLandX = state.x;
                    audit->bestLandY = state.y;
                    audit->bestSpeed = speed;
                    audit->bestInput = forward;
                    audit->bestLaunchVelocity = launchVelocity >> 12;
                }
                // The transition endpoint is the concrete supported player
                // pose proved above.  Requiring passive drag to remove all
                // horizontal velocity before the far edge adds a planner-only
                // capability restriction: Blood still accepts input in air
                // and after touchdown, and velocity is execution state rather
                // than part of support reachability.  The sole executor owns
                // that live state after contact.
                // The lattice is ordered from the least aggressive accepted
                // control.  Preserve that stable choice, but carry the exact
                // validated duration with it so execution does not reject a
                // slow valid transition using an unrelated fixed timeout.
                if (validatedControl)
                    *validatedControl = forward;
                if (validatedFrames)
                    *validatedFrames = frame;
                return true;
            }
            if (audit && !clearedTarget)
                ++audit->noContact;
        }
        return false;
    }

    bool retainedJump(const NavCell &from, const NavCell &to,
                      int launchVelocity = 0,
                      AirTransitionAudit *audit = nullptr,
                      int *validatedControl = nullptr,
                      int *validatedFrames = nullptr) const
    {
        return retainedAirTransition(
            from, to, llmapper::capability::playerJumpImpulse(),
            launchVelocity, audit, validatedControl, validatedFrames);
    }

    bool retainedDrop(const NavCell &from, const NavCell &to,
                      AirTransitionAudit *audit = nullptr,
                      int *validatedControl = nullptr,
                      int *validatedFrames = nullptr) const
    {
        return retainedAirTransition(from, to, 0, 0, audit,
                                     validatedControl, validatedFrames);
    }

    // Find a concrete pose behind a jump lip from which the same physical
    // support provides a straight engine-valid run-up.  This is not a larger
    // guessed jump envelope: the resulting ground velocity is replayed from
    // Blood's acceleration/drag, and the airborne transition is accepted
    // only if that exact velocity lands and remains on the requested support.
    int jumpRunUpCell(const NavCell &takeoff, const NavCell &landing) const
    {
        const int dx = landing.center.x - takeoff.center.x;
        const int dy = landing.center.y - takeoff.center.y;
        const double length = std::sqrt(double(int64_t(dx) * dx
                                                + int64_t(dy) * dy));
        if (length < 1.0)
            return -1;
        const int maximum = llmapper::capability::playerRunUpDistance();
        int best = -1;
        int bestAlong = 0;
        for (const NavCell &candidate : navCells)
        {
            if (candidate.support != takeoff.support
                || candidate.z != takeoff.z)
                continue;
            const int backX = takeoff.center.x - candidate.center.x;
            const int backY = takeoff.center.y - candidate.center.y;
            const int along = int(std::lround(
                (double(backX) * dx + double(backY) * dy) / length));
            const int lateral = int(std::lround(std::abs(
                double(backX) * dy - double(backY) * dx) / length));
            if (along <= bestAlong || along > maximum
                || lateral > playerClipRadius())
                continue;
            if (!lineRetainsSupport(
                    candidate.center.x, candidate.center.y,
                    candidate.sector, candidate.z, candidate.support,
                    takeoff.center.x, takeoff.center.y, takeoff.z))
                continue;
            const int samples = std::max(1, along / 64);
            bool retained = true;
            for (int sample = 0; sample <= samples && retained; ++sample)
            {
                const int x = candidate.center.x
                    + int(int64_t(takeoff.center.x - candidate.center.x)
                          * sample / samples);
                const int y = candidate.center.y
                    + int(int64_t(takeoff.center.y - candidate.center.y)
                          * sample / samples);
                int supportZ = 0;
                retained = engineSupportZAt(candidate.support, x, y, supportZ)
                    && supportZ == candidate.z
                    && engineHasSupportAt(candidate.support, candidate.sector,
                                          x, y, supportZ,
                                          playerSupportRadius());
            }
            if (!retained)
                continue;
            best = candidate.id;
            bestAlong = along;
        }
        return best;
    }

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
        // floor mid-travel) may use the closest cell of the same sector. A
        // sector-qualified query must never silently change physical owner.
        int best = -1;
        int bestDistance = INT32_MAX;
        int bestZ = INT32_MAX;
        for (const NavCell &cell : navCells)
        {
            if (support && cell.support != *support)
                continue;
            if (cell.sector != sectorId)
                continue;
            const int currentDistance = distance2(x, y, cell.center.x, cell.center.y);
            const int currentZ = z == INT32_MIN ? cell.z : std::abs(cell.z - z);
            if (currentZ < bestZ
                || (currentZ == bestZ && currentDistance < bestDistance))
            {
                bestZ = currentZ;
                bestDistance = currentDistance;
                best = cell.id;
            }
        }
        return best;
    }

    int currentNavStartCell()
    {
        const SupportRef liveSupport = currentPlayerSupport();
        const int exact = nearestNavCell(observation.sector, observation.x,
                                         observation.y,
                                         llmapper::capability::playerFloorZ(),
                                         &liveSupport);
        auto entryReachable = [&](const NavCell &cell)
        {
            if (!cell.live || cell.support != liveSupport
                || std::abs(cell.z - llmapper::capability::playerFloorZ())
                    > llmapper::capability::playerStepHeight())
                return false;
            const MovementProbe entry = probeMovement(
                observation.x, observation.y, observation.z,
                observation.sector, cell.center.x, cell.center.y,
                cell.sector, std::max(128, playerClipRadius()));
            return entry.reachable;
        };
        if (inRange(exact, 0, int(navCells.size()))
            && !navCells[size_t(exact)].links.empty()
            && entryReachable(navCells[size_t(exact)]))
            return exact;

        // A nearest cell is not necessarily on the player's side of a thin
        // wall sprite or other collision body. Anchor the live pose only to
        // a cell the copied engine movement can actually enter; otherwise a
        // route begins beyond the blocker and can never be executed.
        int best = -1;
        int64_t bestDistance = INT64_MAX;
        for (const NavCell &cell : navCells)
        {
            if (cell.links.empty() || !entryReachable(cell))
                continue;
            const int64_t distance = distance2(observation.x, observation.y,
                                               cell.center.x, cell.center.y);
            if (distance >= bestDistance)
                continue;
            bestDistance = distance;
            best = cell.id;
        }
        if (best >= 0 && best != exact && !reportedNavStartReconnect)
        {
            char detail[176];
            snprintf(detail, sizeof(detail),
                     "nearest=%d selected=%d support_kind=%d support=%d reason=engine_collision_separates_nearest_cell",
                     exact, best, int(liveSupport.kind), liveSupport.id);
            event("nav_start_anchor_reconnected", detail);
            reportedNavStartReconnect = true;
        }
        return best >= 0 ? best : exact;
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
                              const SupportRef &support, bool live = true,
                              bool crouchOnly = false) {
            const bool currentSectorFloor = support.kind == kSupportSectorFloor
                && support.id == sectorId && z == sectorRecord.floorz;
            live = live || currentSectorFloor;
            if (live && !enginePlayerPoseFits(
                    sectorId, x, y, z, crouchOnly))
                return;
            for (GridCell &known : grid.cells)
                if (known.gx == gx && known.gy == gy && known.z == z
                    && known.support == support)
                {
                    known.live = known.live || live;
                    known.crouchOnly = known.crouchOnly && crouchOnly;
                    return;
                }
            GridCell cell;
            cell.gx = gx;
            cell.gy = gy;
            cell.x = x;
            cell.y = y;
            cell.z = z;
            const int64_t clearance2 = nearestWallDistance2(sectorId, x, y);
            cell.clearance = clearance2 == INT64_MAX
                ? INT32_MAX
                : int(std::sqrt(double(clearance2)));
            if (support.kind == kSupportSpriteFloor)
                cell.clearance = std::min(
                    cell.clearance,
                    spriteSupportClearance(sectorId, support.id, x, y,
                                           playerSupportRadius()));
            cell.live = live;
            cell.crouchOnly = crouchOnly;
            cell.support = support;
            grid.cells.push_back(cell);
        };

        auto appendSurfaces = [&](int gx, int gy, int x, int y, int required,
                                  bool crouchOnly = false) {
            const std::vector<StandableSurface> surfaces = standableSurfacesAt(
                sectorId, x, y, playerSupportRadius(), required);
            for (const StandableSurface &surface : surfaces)
                appendCell(gx, gy, x, y, surface.z, surface.support, true,
                           crouchOnly);

            // Preserve possible stable mechanism endpoints for conditional
            // transport planning, but label them as hypothetical.  They may
            // describe how an elevator could connect components; they are not
            // present collision poses and cannot satisfy ActionScan until the
            // engine actually moves the support there.
            if (sectorRecord.extra > 0 && sectorRecord.extra < kMaxXSectors)
            {
                const XSECTOR &dynamic = xsector[sectorRecord.extra];
                if (dynamic.offFloorZ != dynamic.onFloorZ)
                {
                    if (dynamic.offFloorZ - dynamic.offCeilZ >= required)
                        appendCell(gx, gy, x, y, dynamic.offFloorZ,
                                   SupportRef(kSupportSectorFloor, sectorId),
                                   dynamic.offFloorZ == sectorRecord.floorz,
                                   crouchOnly);
                    if (dynamic.onFloorZ - dynamic.onCeilZ >= required)
                        appendCell(gx, gy, x, y, dynamic.onFloorZ,
                                   SupportRef(kSupportSectorFloor, sectorId),
                                   dynamic.onFloorZ == sectorRecord.floorz,
                                   crouchOnly);
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
        const int radius = playerSupportRadius();
        const int step = 1 << (kNavGridShift - 2);
        for (int pass = 0; pass < 2 && grid.cells.empty(); ++pass)
        {
            // pass 0: plain grid centres with a body margin.
            // pass 1: the same, but sampling inside each square for the most
            //         open point -- rescues corridors narrower than the grid.
            const bool sample = pass >= 1;
            const int required = clearance;
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
                    if (bestClearance < 0)
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
        }

        // A physical pose includes the posture needed to occupy it.  Sample
        // low space with the measured crouch collision hull even when the
        // same Build sector also contains standing poses; sector ownership
        // does not imply one vertical clearance throughout the region.
        if (gObservedCrouchShape.known
            && playerCrouchClearance() < playerStandingClearance())
        {
            const int required = playerCrouchClearance();
            for (int gy = minY >> kNavGridShift;
                 gy <= (maxY >> kNavGridShift); ++gy)
                for (int gx = minX >> kNavGridShift;
                     gx <= (maxX >> kNavGridShift); ++gx)
                {
                    const int baseX = (gx << kNavGridShift)
                        + (1 << (kNavGridShift - 1));
                    const int baseY = (gy << kNavGridShift)
                        + (1 << (kNavGridShift - 1));
                    int bestX = baseX;
                    int bestY = baseY;
                    int64_t bestClearance = -1;
                    for (int ox = -1; ox <= 1; ++ox)
                        for (int oy = -1; oy <= 1; ++oy)
                        {
                            const int cx = baseX + ox * step;
                            const int cy = baseY + oy * step;
                            if (inside(cx, cy, sectorId) != 1)
                                continue;
                            const int vertical = standingFloorZ(
                                sectorId, cx, cy)
                                - getceilzofslope(sectorId, cx, cy);
                            if (vertical < required
                                || vertical >= playerStandingClearance()
                                || blockedBySolidSprite(sectorId, cx, cy))
                                continue;
                            const int64_t clear = nearestWallDistance2(
                                sectorId, cx, cy);
                            if (clear > bestClearance)
                            {
                                bestClearance = clear;
                                bestX = cx;
                                bestY = cy;
                            }
                        }
                    if (bestClearance >= 0)
                        appendSurfaces(gx, gy, bestX, bestY, required, true);
                }
        }

        // Non-floor-aligned solid sprites are obstacles at the sector floor,
        // but Build exposes their top extents as real getzrange floors.  That
        // includes wall sprites: although their artwork is a zero-thickness
        // sheet, getzrange expands the collision line by the querying body's
        // radius and Caleb can settle on the resulting narrow top.  The
        // ordinary floor pass intentionally rejects points occupied by these
        // obstacles, so seed their independently valid upper support layers
        // in a bounded box around the engine collision footprint.  Every
        // emitted cell is confirmed by the engine for the player's actual
        // clip radius and headroom.
        for (int spriteId = headspritesect[sectorId]; spriteId >= 0;
             spriteId = nextspritesect[spriteId])
        {
            const spritetype &supportSprite = sprite[spriteId];
            if (gMe && gMe->pSprite && spriteId == gMe->pSprite->index)
                continue;
            if (!(supportSprite.cstat & CSTAT_SPRITE_BLOCK))
                continue;
            const int alignment = supportSprite.cstat & CSTAT_SPRITE_ALIGNMENT_MASK;
            // Floor/slope sprites already participate in the ordinary grid
            // pass. All other blocking alignments are eligible here solely
            // when GetZRange confirms their collision top; tile, voxel,
            // visual shape and gameplay type are irrelevant.
            if (alignment == CSTAT_SPRITE_ALIGNMENT_FLOOR
                || alignment == CSTAT_SPRITE_ALIGNMENT_SLOPE)
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
            int footprintMargin = radius + (supportSprite.clipdist << 2) + 64;
            int wallSpanX1 = 0, wallSpanY1 = 0, wallSpanX2 = 0, wallSpanY2 = 0;
            if (alignment == CSTAT_SPRITE_ALIGNMENT_WALL)
            {
                wallSpriteSpan(supportSprite, wallSpanX1, wallSpanY1,
                               wallSpanX2, wallSpanY2);
                supportMinX = std::min(wallSpanX1, wallSpanX2);
                supportMaxX = std::max(wallSpanX1, wallSpanX2);
                supportMinY = std::min(wallSpanY1, wallSpanY2);
                supportMaxY = std::max(wallSpanY1, wallSpanY2);
                // getzrange uses precisely the query radius around a wall
                // sprite's collision segment.  The small sampling allowance
                // lets an off-grid segment still contribute a pose; the
                // subsequent engine query remains authoritative.
                footprintMargin = radius + 64;
            }
            supportMinX -= footprintMargin;
            supportMaxX += footprintMargin;
            supportMinY -= footprintMargin;
            supportMaxY += footprintMargin;
            const SupportRef wanted(kSupportSpriteFloor, spriteId);
            const size_t cellsBeforeSupport = grid.cells.size();
            if (alignment == CSTAT_SPRITE_ALIGNMENT_WALL)
            {
                // The collision line is the medial axis of every pose on a
                // wall sprite's narrow top. Sampling that axis gives route
                // execution the maximum real NBlood support margin. A broad
                // grid query can also confirm poses near the edge of the
                // radius-expanded band, but selecting those as landing
                // centres makes a mathematically valid pose needlessly
                // unstable.
                const int spanX = wallSpanX2 - wallSpanX1;
                const int spanY = wallSpanY2 - wallSpanY1;
                const int spanLength = int(std::sqrt(double(
                    int64_t(spanX) * spanX + int64_t(spanY) * spanY)));
                const int samples = std::max(1, (spanLength + 127) / 128);
                for (int sampleId = 0; sampleId <= samples; ++sampleId)
                {
                    const int x = wallSpanX1
                        + int(int64_t(spanX) * sampleId / samples);
                    const int y = wallSpanY1
                        + int(int64_t(spanY) * sampleId / samples);
                    if (inside(x, y, sectorId) != 1)
                        continue;
                    const std::vector<StandableSurface> surfaces =
                        standableSurfacesAt(sectorId, x, y, radius, clearance);
                    for (const StandableSurface &surface : surfaces)
                    {
                        if (surface.support != wanted)
                            continue;
                        appendCell(x >> kNavGridShift, y >> kNavGridShift,
                                   x, y, surface.z, surface.support);
                        break;
                    }
                }
            }
            else
            {
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
        // The live player's origin is direct engine proof of one occupiable
        // pose.  Preserve that exact position; snapping it to the grid centre
        // can move the supposedly proven pose through a nearby wall.
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
                appendCell(gx, gy, observation.x, observation.y,
                           playerFloor, support, true,
                           gMe->posture == kPostureCrouch);
            }
        }
        grid.admitsPlayer = !grid.cells.empty();
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
        if (!navCells.empty())
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
                // Never publish intermediate moving geometry, even when
                // opening it also reveals another Build sector and changes
                // the set used by the topology identity.  That identity is
                // metadata; the moving physical pose is the stronger fact.
                return;
            }
            if (navMeshInMotion)
            {
                // Stability is elapsed simulation time, not a count of calls
                // to ensureNavTopology(). Several planning layers call this
                // accessor in one frame. Intermediate positions are neither
                // stable player poses nor useful route endpoints, so rebuilding
                // the full physical graph while geometry is still translating
                // is both semantically false and extremely expensive. Publish
                // exactly one new graph after the engine pose settles.
                const bool poseStable = observation.tick - navPoseChangedTick
                    >= 8 * kTicsPerFrame;
                if (poseStable)
                {
                    navMeshInMotion = false;
                    refreshDynamicNavLinks();
                    navDynamicSignatureValue = dynamicSignature;
                    // Moving sprites can cross sector boundaries and change
                    // which support layer occupies a grid square.  Rebuild
                    // the affected surface graph once the pose is stable.
                    navTopologySignature = 0;
                    event("nav_mesh_settled", "reason=pose_stable");
                }
                return;
            }
        }
        if (topology == navTopologySignature && !navCells.empty())
        {
            if (dynamicSignature != navDynamicSignatureValue)
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
        // Cell ids are stable only inside one published physical graph.
        // Preserve wall/mechanism attempts, but retire failures whose
        // identity is a concrete cell-to-cell transition before assigning
        // ids in the replacement graph.
        navEdgeFailures.erase(
            std::remove_if(navEdgeFailures.begin(), navEdgeFailures.end(),
                           [](const NavEdgeFailure &failure)
                           { return failure.wall < 0; }),
            navEdgeFailures.end());
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
                || grid->second.signature != sectorGeometrySignature(sectorId)
                || (sectorId == observation.sector
                    && grid->second.cells.empty()))
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
                cell.clearance = source.clearance;
                cell.live = source.live;
                cell.crouchOnly = source.crouchOnly;
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
        int internalCandidates = 0;
        int internalMidpointRejected = 0;
        int internalSpriteRejected = 0;
        int internalCollisionRejected = 0;
        int internalWallRejected = 0;
        int internalSupportRejected = 0;
        int internalLinked = 0;
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
                    ++internalCandidates;
                    const NavCell &other = navCells[size_t(neighbour)];
                    const int midX = (cell.center.x + other.center.x) / 2;
                    const int midY = (cell.center.y + other.center.y) / 2;
                    const bool samePhysicalSupport =
                        cell.support == other.support;
                    // Sector polygons are backend ownership metadata.  Two
                    // occupiable poses on one physical sprite/voxel support
                    // remain adjacent even when that support lies on or
                    // crosses a Build-sector boundary, where inside() is
                    // deliberately ambiguous.
                    if (!samePhysicalSupport
                        && inside(midX, midY, cell.sector) != 1)
                    {
                        ++internalMidpointRejected;
                        continue;
                    }
                // The midpoint test alone is not enough: a wall lying exactly
                // between two cell centres puts the midpoint on the wall,
                // where inside() is ambiguous, and a diagonal can slip past
                // the corner of a thin obstruction entirely.  Ask the engine
                // whether the two squares can actually see each other; a
                // one-sided wall between them blocks the ray.
                    bool spriteBlocked = false;
                    for (int sample = 1; sample < 4 && !spriteBlocked; ++sample)
                    {
                        const int sampleX = cell.center.x
                            + int(int64_t(other.center.x - cell.center.x)
                                  * sample / 4);
                        const int sampleY = cell.center.y
                            + int(int64_t(other.center.y - cell.center.y)
                                  * sample / 4);
                        const int blocker = solidSpriteAt(
                            cell.sector, sampleX, sampleY,
                            playerClipRadius());
                        const bool blockerIsSupport = blocker >= 0
                            && ((cell.support.kind == kSupportSpriteFloor
                                 && cell.support.id == blocker)
                                || (other.support.kind == kSupportSpriteFloor
                                    && other.support.id == blocker));
                        spriteBlocked = blocker >= 0 && !blockerIsSupport;
                    }
                    if (spriteBlocked)
                    {
                        ++internalSpriteRejected;
                        continue;
                    }
                    // Sampling is an acceleration only. A thin wall sprite
                    // can lie between all three quarter samples while
                    // clipmove still stops the player's full cylinder on it.
                    // The engine collision result owns this edge.
                    {
                        const bool crouched = cell.crouchOnly
                            || other.crouchOnly;
                        const int originZ = cell.z
                            - (crouched && gObservedCrouchShape.known
                               ? gObservedCrouchShape.footOffset
                               : llmapper::capability::playerFootOffset());
                        const MovementProbe physical = probeMovement(
                            cell.center.x, cell.center.y,
                            originZ,
                            cell.sector, other.center.x, other.center.y,
                            other.sector,
                            std::max(64, playerClipRadius()), crouched);
                        const bool endpointSprite = physical.sprite >= 0
                            && ((cell.support.kind == kSupportSpriteFloor
                                 && cell.support.id == physical.sprite)
                                || (other.support.kind == kSupportSpriteFloor
                                    && other.support.id == physical.sprite));
                        // ClipMove reports the first surface touched even
                        // when it slides the cylinder along that surface and
                        // reaches the requested pose.  The resolved endpoint
                        // is the traversal predicate; incidental side-wall
                        // feedback is blocker evidence only when the endpoint
                        // was not reached.
                        if (!physical.reachable
                            && ((physical.sprite >= 0 && !endpointSprite)
                                || physical.wall >= 0))
                        {
                            ++internalCollisionRejected;
                            continue;
                        }
                    }
                // Deliberately geometry, not cansee(): that is a sight test
                // and is blocked by sprites, so a courtyard with scenery in
                // it was carved into separate walk areas by its own torches.
                    if (cell.support.kind == kSupportSectorFloor
                        && segmentCrossesSectorWall(
                            cell.sector, cell.center.x, cell.center.y,
                            other.center.x, other.center.y))
                    {
                        ++internalWallRejected;
                        continue;
                    }
                    if (samePhysicalSupport
                        && !lineRetainsSupport(
                            cell.center.x, cell.center.y, cell.sector,
                            cell.z, cell.support,
                            other.center.x, other.center.y, other.z))
                    {
                        ++internalSupportRejected;
                        continue;
                    }
                    const int floorDelta = other.z - cell.z;
                    const LocalWaypoint gateway(midX, midY);
                    auto directedMode = [&](int delta) {
                        if (std::abs(delta) <= llmapper::capability::playerStepHeight())
                            return cell.crouchOnly || other.crouchOnly
                                ? kNavCrouch
                                : delta == 0 ? kNavWalk : kNavStep;
                        // Grid proximity and a height delta do not prove an
                        // air transition.  Only retainedAirTransition() may
                        // publish JUMP/DROP because it records the concrete
                        // source, engine collision replay, control and named
                        // landing support consumed by the sole executor.
                        return kNavBlocked;
                    };
                    const NavEdgeMode forward = directedMode(floorDelta);
                    const NavEdgeMode reverse = directedMode(-floorDelta);
                    if (llmapper::traversableMode(forward))
                    {
                        addNavLink(int(i), neighbour, forward, -1, gateway, true);
                        ++internalLinked;
                    }
                    if (llmapper::traversableMode(reverse))
                        addNavLink(neighbour, int(i), reverse, -1, gateway, true);
                }
            }
        }
        {
            char audit[224];
            snprintf(audit, sizeof(audit),
                     "candidates=%d linked=%d midpoint=%d sprite=%d collision=%d wall=%d support=%d",
                     internalCandidates, internalLinked,
                     internalMidpointRejected, internalSpriteRejected,
                     internalCollisionRejected, internalWallRejected,
                     internalSupportRejected);
            event("nav_internal_link_audit", audit);
        }

        // Concrete support poses may be separated by free space regardless
        // of whether either support is a sector floor, sprite, voxel or
        // moving object.  Build one bounded candidate set for all support
        // pairs.  A scalar jump envelope prunes the spatial query; only a
        // retainedStandingJump() proof may authorize a JUMP edge.
        const int landingTolerance = playerClipRadius() + (1 << (kNavGridShift - 2));
        // Every sprite-support candidate used to replay the same 240-frame
        // standing-jump simulation.  A topology contains many cells but only
        // a handful of distinct support-height deltas, so cache the engine-
        // derived answer for this rebuild.  The cache is deliberately local:
        // inventory/posture capability changes naturally receive fresh
        // physics on the next topology construction.
        std::map<int, int> standingReachByDelta;
        std::map<int, bool> harmlessDropByDelta;
        auto cachedStandingReach = [&](int delta) {
            auto found = standingReachByDelta.find(delta);
            if (found != standingReachByDelta.end())
                return found->second;
            const int reach = standingJumpReachForDelta(delta);
            standingReachByDelta.emplace(delta, reach);
            return reach;
        };
        auto harmlessDrop = [&](int delta) {
            if (delta <= 0)
                return true;
            auto found = harmlessDropByDelta.find(delta);
            if (found != harmlessDropByDelta.end())
                return found->second;
            const bool harmless = llmapper::capability::playerFallDamage(delta) == 0;
            harmlessDropByDelta.emplace(delta, harmless);
            return harmless;
        };
        std::map<std::pair<int, int>, std::vector<size_t> > physicalBuckets;
        std::set<int> supportHeights;
        for (const NavCell &cell : navCells)
        {
            physicalBuckets[std::make_pair(cell.gx, cell.gy)].push_back(
                size_t(cell.id));
            supportHeights.insert(cell.z);
        }
        int maximumCandidateSpan = 0;
        std::set<int> reportedJumpDeltas;
        for (int fromZ : supportHeights)
            for (int toZ : supportHeights)
            {
                const int delta = toZ - fromZ;
                if (!harmlessDrop(delta))
                    continue;
                if (reportedJumpDeltas.insert(delta).second)
                {
                    char detail[192];
                    snprintf(detail, sizeof(detail),
                             "delta=%d step=%d apex=%d standing_reach=%d running_reach=%d run_up=%d",
                             delta, llmapper::capability::playerStepHeight(),
                             llmapper::capability::playerJumpApex(),
                             cachedStandingReach(delta),
                             runningJumpReachForDelta(delta),
                             llmapper::capability::playerRunUpDistance());
                    event("nav_jump_envelope", detail);
                }
                maximumCandidateSpan = std::max(
                    maximumCandidateSpan,
                    std::max(cachedStandingReach(delta),
                             runningJumpReachForDelta(delta)));
            }
        maximumCandidateSpan += landingTolerance;
        struct SupportTransitionCandidate
        {
            size_t from = 0;
            size_t to = 0;
            NavEdgeMode mode = kNavBlocked;
            int64_t score = INT64_MAX;
            int clearance = 0;
            int airControl = 0;
            int airFrames = 0;
            bool hasAirControl = false;
            LocalWaypoint takeoff;
            bool hasTakeoff = false;
        };
        struct SupportPairAudit
        {
            int considered = 0;
            int heightOrHazard = 0;
            int noJumpEnvelope = 0;
            int span = 0;
            int wall = 0;
            int sprite = 0;
            int walkCollision = 0;
            int arc = 0;
            int accepted = 0;
            AirTransitionAudit air;
        };
        std::map<std::pair<size_t, SupportRef>,
                  SupportTransitionCandidate> supportTransitions;
        // Expensive engine arc simulation is the final predicate, not the
        // spatial search algorithm. Retain a few best geometric landings per
        // concrete takeoff pose/support pair, then validate those in order.
        // This preserves alternate physical crossings while avoiding one
        // 240-frame simulation for every interior grid cell on a support.
        std::map<std::pair<size_t, SupportRef>,
                 std::vector<SupportTransitionCandidate> > jumpOptions;
        std::map<std::pair<SupportRef, SupportRef>, SupportPairAudit>
            supportPairAudit;
        std::map<std::pair<int, int>, int> supportEndDistanceCache;
        auto supportEndsToward = [&](const NavCell &from,
                                     const NavCell &to) {
            const int dx = to.center.x - from.center.x;
            const int dy = to.center.y - from.center.y;
            const int length = int(std::sqrt(double(
                int64_t(dx) * dx + int64_t(dy) * dy)));
            if (length <= 0)
                return false;
            // A ballistic transition launches at a support boundary.  Cells
            // farther inside the same support are dominated: moving the
            // launch toward the lip preserves more flight time, while a
            // running transition already stores its separate preparation
            // pose.  Use GetZRange ownership one grid step toward the target
            // to identify that boundary instead of comparing Build sectors.
            // Quantize only the query direction, not the graph pose.  Each
            // cell then asks the engine at most sixteen support-boundary
            // questions per rebuild instead of repeating GetZRange for every
            // destination cell in that direction.
            const int direction = ((getangle(dx, dy) + 64) & 2047) >> 7;
            const auto cacheKey = std::make_pair(from.id, direction);
            auto cached = supportEndDistanceCache.find(cacheKey);
            if (cached == supportEndDistanceCache.end())
            {
                // A transition source may include a running approach, so its
                // graph pose need not be the last cell at the lip.  Search
                // only through the maximum engine-derived air envelope and
                // remember the first distance where the named support no
                // longer carries the player hull.
                int endDistance = INT_MAX;
                const int stride = 1 << kNavGridShift;
                const int supportSector =
                    from.support.kind == kSupportSectorFloor
                    ? from.support.id : from.sector;
                const int probeAngle = direction << 7;
                for (int probeDistance = stride;
                     probeDistance <= maximumCandidateSpan;
                     probeDistance += stride)
                {
                    const int probeX = from.center.x
                        + mulscale30(probeDistance, Cos(probeAngle));
                    const int probeY = from.center.y
                        + mulscale30(probeDistance, Sin(probeAngle));
                    int supportZ = 0;
                    if (!engineSupportZAt(
                            from.support, probeX, probeY, supportZ)
                        || supportZ != from.z
                        || !engineHasSupportAt(
                            from.support, supportSector, probeX, probeY,
                            supportZ, playerSupportRadius()))
                    {
                        endDistance = probeDistance;
                        break;
                    }
                }
                cached = supportEndDistanceCache.emplace(
                    cacheKey, endDistance).first;
            }
            return cached->second <= length + landingTolerance;
        };
        auto addSupportTransition = [&](size_t fromId, size_t toId) {
            if (fromId == toId)
                return;
            const NavCell &from = navCells[fromId];
            const NavCell &to = navCells[toId];
            // Build sector ownership is not physical connectivity.  A jump
            // may start on a sector floor and land on a sprite whose owning
            // sector is across an open boundary (or vice versa).  The copied
            // ClipMove probe below is the authority for crossing that
            // boundary; rejecting the pair here made AGTST15's real sprite
            // supports unreachable solely because of map partitioning.
            if (from.support == to.support)
                return;
            SupportPairAudit &audit = supportPairAudit[
                std::make_pair(from.support, to.support)];
            ++audit.considered;
            const int delta = to.z - from.z;
            if (!harmlessDrop(delta)
                || engineSupportDamagesPlayer(to.support))
            {
                ++audit.heightOrHazard;
                return;
            }
            // NavCell::z is always the physical support plane under the
            // player's feet.  standingJumpReachForDelta() advances a centre
            // position but compares state.z + foot against that same support
            // plane.  Adding the foot offset again for sprite supports mixed
            // target geometry with player-pose geometry and granted a much
            // longer, physically false jump only when the destination was a
            // sprite.
            // Broad-phase pruning must include the longest transition the
            // engine can produce. The old standing-only envelope discarded
            // running jumps before jumpRunUpCell() could prove their concrete
            // takeoff, even though the retained engine replay accepted them.
            const int reach = std::max(cachedStandingReach(delta),
                                       runningJumpReachForDelta(delta));
            if (reach <= 0)
            {
                ++audit.noJumpEnvelope;
                return;
            }
            const int64_t span = distance2(from.center.x, from.center.y,
                                           to.center.x, to.center.y);
            const int maxSpan = reach + landingTolerance;
            if (span > int64_t(maxSpan) * maxSpan)
            {
                ++audit.span;
                return;
            }
            // A higher support can itself be the collision boundary: the
            // source floor may continue underneath it.  Preserve that case
            // only when the engine attributes the obstructed ground path to
            // the destination support.  Merely being higher and nearby is
            // not evidence of a transition; treating it as such makes every
            // unrelated floor layer a jump candidate.
            const bool climbsOntoHigherSupport = to.z < from.z;
            const bool sourceSupportEnds = supportEndsToward(from, to);
            bool destinationOwnsClimbBoundary = false;
            if (!sourceSupportEnds && climbsOntoHigherSupport)
            {
                if (to.support.kind == kSupportSpriteFloor)
                    destinationOwnsClimbBoundary = true;
                else
                {
                    const int originZ = from.z
                        - llmapper::capability::playerFootOffset();
                    const MovementProbe barrier = probeMovement(
                        from.center.x, from.center.y, originZ,
                        from.sector, to.center.x, to.center.y,
                        to.sector, 64, false);
                    destinationOwnsClimbBoundary = barrier.wall >= 0
                        && inRange(barrier.wall, 0, numwalls)
                        && wall[barrier.wall].nextsector == to.sector;
                }
            }
            if (!sourceSupportEnds && !destinationOwnsClimbBoundary)
            {
                ++audit.span;
                return;
            }
            bool walks = false;
            if (std::abs(delta)
                <= llmapper::capability::playerStepHeight())
            {
                const bool crouched = from.crouchOnly || to.crouchOnly;
                const int originZ = from.z
                    - (crouched && gObservedCrouchShape.known
                       ? gObservedCrouchShape.footOffset
                       : llmapper::capability::playerFootOffset());
                const MovementProbe physical = probeMovement(
                    from.center.x, from.center.y, originZ,
                    from.sector, to.center.x, to.center.y,
                    to.sector, 64, crouched);
                // ClipMove owns horizontal hull clearance but deliberately
                // does not prove that a floor carries the player along the
                // segment.  Require both engine facts.  Without the support
                // sweep a clear XY line across AGTST3's gap was mislabeled
                // WALK and then discarded as a duplicate before the air
                // validator could ever see it.
                walks = physical.reachable
                    && lineKeepsSupport(
                        from.center.x, from.center.y, from.sector,
                        from.z, from.support,
                        to.center.x, to.center.y);
            }
            // Two sector floors meet through a concrete Build boundary whose
            // wall/door identity and live state must own the WALK edge.
            // Publishing an additional anonymous support edge while a door
            // happened to be open made that connection permanent after the
            // door closed. addPhysicalBoundaryLinks() supplies the one live,
            // invalidatable representation of this crossing.
            if (walks
                && from.support.kind == kSupportSectorFloor
                && to.support.kind == kSupportSectorFloor)
                return;
            // Ordinary motion between different support owners is already
            // published once by addPhysicalLocalLinks() (or by the concrete
            // wall boundary).  This builder owns only ballistic transitions;
            // retaining another anonymous WALK/STEP edge here gives one
            // physical crossing two identities and two invalidation rules.
            if (walks)
                return;
            const NavEdgeMode mode = walks
                ? (from.crouchOnly || to.crouchOnly
                   ? kNavCrouch
                   : delta == 0 ? kNavWalk : kNavStep)
                : (delta > 0 ? kNavDrop : kNavJump);
            // Cheap geometry ranks ballistic candidates; the retained engine
            // simulation below owns all jump collision. Ordinary walking has
            // no later simulation, so validate its swept cylinder here.
            if (walks)
            {
                if (segmentCrossesBlockingWall(from.sector,
                                               from.center.x, from.center.y,
                                               to.center.x, to.center.y))
                {
                    ++audit.wall;
                    return;
                }
                const bool crouched = mode == kNavCrouch;
                const int originZ = from.z
                    - (crouched && gObservedCrouchShape.known
                       ? gObservedCrouchShape.footOffset
                       : llmapper::capability::playerFootOffset());
                const MovementProbe physical = probeMovement(
                    from.center.x, from.center.y,
                    originZ, from.sector, to.center.x, to.center.y,
                    to.sector, 64, crouched);
                const bool endpointSprite = physical.sprite >= 0
                    && ((from.support.kind == kSupportSpriteFloor
                         && from.support.id == physical.sprite)
                        || (to.support.kind == kSupportSpriteFloor
                            && to.support.id == physical.sprite));
                if ((physical.sprite >= 0 && !endpointSprite)
                    || physical.wall >= 0)
                {
                    ++audit.walkCollision;
                    return;
                }
            }
            const int64_t score = span
                + int64_t(std::abs(delta)) * 16
                + (mode == kNavJump ? 1 << 20 : 0);
            // Preserve distinct physical crossings.  Collapsing the whole
            // pair of supports to one edge recreates the old sector-graph
            // "first wall wins" defect.  One best target per concrete source
            // pose and destination support stays bounded while retaining
            // separate takeoff locations.
            if (mode == kNavJump || mode == kNavDrop)
            {
                SupportTransitionCandidate option;
                option.from = fromId;
                option.to = toId;
                option.mode = mode;
                option.score = score;
                option.clearance = to.clearance;
                std::vector<SupportTransitionCandidate> &options =
                    jumpOptions[std::make_pair(fromId, to.support)];
                // Keep the non-dominated physical landing poses.  Nearest is
                // not synonymous with executable: on a narrow platform all
                // nearest grid cells can lie along the same unsafe lip while
                // a slightly farther interior cell is the only retainable
                // landing.  The former fixed "four nearest" truncation lost
                // that pose before the engine replay ever saw it.  Cost and
                // actual support clearance are the two facts the retained
                // transition consumes, so their Pareto frontier is both
                // complete for this validator and naturally bounded.
                const bool dominated = std::any_of(
                    options.begin(), options.end(),
                    [&](const SupportTransitionCandidate &existing)
                    {
                        return existing.score <= option.score
                            && existing.clearance >= option.clearance;
                    });
                if (dominated)
                    return;
                options.erase(std::remove_if(
                    options.begin(), options.end(),
                    [&](const SupportTransitionCandidate &existing)
                    {
                        return option.score <= existing.score
                            && option.clearance >= existing.clearance;
                    }), options.end());
                options.push_back(option);
                std::sort(options.begin(), options.end(),
                          [](const SupportTransitionCandidate &a,
                             const SupportTransitionCandidate &b)
                          {
                              if (a.score != b.score)
                                  return a.score < b.score;
                              return a.to < b.to;
                          });
                return;
            }
            const auto key = std::make_pair(fromId, to.support);
            const auto previous = supportTransitions.find(key);
            if (previous != supportTransitions.end()
                && score >= previous->second.score)
                return;
            ++audit.accepted;
            SupportTransitionCandidate &candidate = supportTransitions[key];
            if (score < candidate.score)
            {
                candidate.from = fromId;
                candidate.to = toId;
                candidate.mode = mode;
                candidate.score = score;
            }
        };
        // Bound the spatial candidate query from the same simulated jump
        // reach used to validate an edge.  The old nested scan compared every
        // sprite-support cell with every cell in the known world, producing
        // a real O(N^2) graph-build cost and thousands of meaningless long
        // candidates in AGTST15.  Height layers are few; derive the largest
        // physically relevant reach across those layers, then only inspect
        // nearby XY buckets.
        const int bucketRadius = std::max(
            1, (maximumCandidateSpan + (1 << kNavGridShift) - 1)
                >> kNavGridShift);
        for (size_t supportCell = 0; supportCell < navCells.size(); ++supportCell)
        {
            const NavCell &source = navCells[supportCell];
            for (int gx = source.gx - bucketRadius;
                 gx <= source.gx + bucketRadius; ++gx)
                for (int gy = source.gy - bucketRadius;
                     gy <= source.gy + bucketRadius; ++gy)
                {
                    auto bucket = physicalBuckets.find(std::make_pair(gx, gy));
                    if (bucket == physicalBuckets.end())
                        continue;
                    for (size_t j : bucket->second)
                    {
                        addSupportTransition(supportCell, j);
                    }
                }
        }
        for (const auto &entry : jumpOptions)
        {
            for (const SupportTransitionCandidate &option : entry.second)
            {
                const NavCell &from = navCells[option.from];
                const NavCell &to = navCells[option.to];
                size_t routeFrom = option.from;
                LocalWaypoint takeoff;
                bool hasTakeoff = false;
                // A lower landing does not by itself make the transition a
                // DROP.  Across a horizontal gap the engine may require a
                // jump impulse even though the destination support is below
                // the takeoff plane.  Classify from the replayed motion: a
                // zero-impulse arc owns DROP; otherwise a jump-impulse arc
                // owns JUMP.  Height sign is only a spatial pruning fact.
                NavEdgeMode validatedMode = option.mode;
                bool accepted = false;
                int validatedControl = 0;
                int validatedFrames = 0;
                SupportPairAudit &audit = supportPairAudit[
                    std::make_pair(from.support, to.support)];
                if (option.mode == kNavDrop)
                {
                    accepted = retainedDrop(from, to, &audit.air,
                                            &validatedControl,
                                            &validatedFrames);
                    if (!accepted)
                    {
                        validatedMode = kNavJump;
                        accepted = retainedJump(from, to, 0, &audit.air,
                                                &validatedControl,
                                                &validatedFrames);
                    }
                }
                else
                    accepted = retainedJump(from, to, 0, &audit.air,
                                            &validatedControl,
                                            &validatedFrames);
                if (!accepted && validatedMode == kNavJump)
                {
                    const int preparation = jumpRunUpCell(from, to);
                    if (preparation >= 0)
                    {
                        const NavCell &runUp = navCells[size_t(preparation)];
                        const int runUpDistance = int(std::sqrt(double(
                            distance2(runUp.center.x, runUp.center.y,
                                      from.center.x, from.center.y))));
                        const int velocity =
                            llmapper::capability::playerRunVelocityForDistance(
                                runUpDistance);
                        if (velocity > 0
                            && retainedJump(from, to, velocity, &audit.air,
                                            &validatedControl,
                                            &validatedFrames))
                        {
                            routeFrom = size_t(preparation);
                            takeoff = from.center;
                            hasTakeoff = true;
                            accepted = true;
                        }
                    }
                }
                if (!accepted)
                {
                    ++audit.arc;
                    continue;
                }
                const auto key = std::make_pair(routeFrom, to.support);
                const int64_t validatedScore = option.score
                    + (option.mode != kNavJump
                       && validatedMode == kNavJump ? 1 << 20 : 0);
                auto previous = supportTransitions.find(key);
                if (previous != supportTransitions.end()
                    && validatedScore >= previous->second.score)
                    break;
                SupportTransitionCandidate candidate = option;
                candidate.from = routeFrom;
                candidate.mode = validatedMode;
                candidate.score = validatedScore;
                candidate.airControl = validatedControl;
                candidate.airFrames = validatedFrames;
                candidate.hasAirControl = true;
                candidate.takeoff = takeoff;
                candidate.hasTakeoff = hasTakeoff;
                supportTransitions[key] = candidate;
                ++audit.accepted;
                break;
            }
        }
        for (const auto &entry : supportTransitions)
        {
            const SupportTransitionCandidate &candidate = entry.second;
            const NavCell &from = navCells[candidate.from];
            addNavLink(int(candidate.from), int(candidate.to), candidate.mode,
                       -1, LocalWaypoint(from.center.x, from.center.y), true,
                       -1, candidate.takeoff, candidate.hasTakeoff, false,
                       candidate.airControl, candidate.airFrames,
                       candidate.hasAirControl);
        }
        for (const auto &entry : supportPairAudit)
        {
            const SupportPairAudit &audit = entry.second;
            if (audit.accepted > 0 || audit.considered == 0)
                continue;
            char detail[640];
            snprintf(detail, sizeof(detail),
                     "from=(%d,%d) to=(%d,%d) considered=%d height_or_hazard=%d no_jump_envelope=%d span=%d wall=%d sprite=%d walk_collision=%d arc=%d air_samples=%d no_contact=%d support_miss=%d pose_miss=%d retention_miss=%d blocked_frames=%d best_retention=%d best_from=(%d,%d) best_to=(%d,%d) best_land=(%d,%d) best_speed=%d best_input=%d best_launch=%d",
                     int(entry.first.first.kind), entry.first.first.id,
                     int(entry.first.second.kind), entry.first.second.id,
                     audit.considered, audit.heightOrHazard,
                     audit.noJumpEnvelope, audit.span,
                     audit.wall, audit.sprite, audit.walkCollision, audit.arc,
                     audit.air.samples, audit.air.noContact,
                     audit.air.supportMiss, audit.air.poseMiss,
                     audit.air.retentionMiss, audit.air.blockedFrames,
                     audit.air.bestRetention,
                     audit.air.bestFromX, audit.air.bestFromY,
                     audit.air.bestToX, audit.air.bestToY,
                     audit.air.bestLandX, audit.air.bestLandY,
                     audit.air.bestSpeed, audit.air.bestInput,
                     audit.air.bestLaunchVelocity);
            event("nav_support_pair_unlinked", detail);
        }
        // Keep the capability diagnosable as a graph fact, not merely as
        // surface discovery.  A seeded top with no incoming jump edge is
        // still an unreachable island and explains the visible behaviour of
        // running into (or avoiding) the obstacle instead of climbing it.
        std::set<int> linkedSpriteSupports;
        for (const NavCell &cell : navCells)
            if (cell.support.kind == kSupportSpriteFloor)
                linkedSpriteSupports.insert(cell.support.id);
        std::map<int, int> supportCellCounts;
        std::map<int, int> supportIncoming;
        std::map<int, int> supportOutgoing;
        for (const NavCell &cell : navCells)
        {
            if (cell.support.kind == kSupportSpriteFloor)
                ++supportCellCounts[cell.support.id];
            for (const NavLink &link : cell.links)
            {
                if (link.mode != kNavJump
                    || !inRange(link.target, 0, int(navCells.size())))
                    continue;
                const NavCell &target = navCells[size_t(link.target)];
                if (cell.support == target.support)
                    continue;
                if (cell.support.kind == kSupportSpriteFloor)
                    ++supportOutgoing[cell.support.id];
                if (target.support.kind == kSupportSpriteFloor)
                    ++supportIncoming[target.support.id];
            }
        }
        for (int spriteId : linkedSpriteSupports)
        {
            if (reportedSpriteSupportLinks.count(spriteId))
                continue;
            reportedSpriteSupportLinks.insert(spriteId);
            char supportDetail[192];
            snprintf(supportDetail, sizeof(supportDetail),
                     "sprite=%d cells=%d incoming=%d outgoing=%d candidate_radius=%d",
                     spriteId, supportCellCounts[spriteId],
                     supportIncoming[spriteId], supportOutgoing[spriteId],
                     maximumCandidateSpan);
            event("nav_sprite_support_links", supportDetail);
        }

        addRorNavLinks();
        refreshDynamicNavLinks();
        ++navTopologyRevision;
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
        dumpPhysicalNavMesh();
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
        if (probe.sprite >= 0 && inRange(probe.sprite, 0, kMaxSprites))
        {
            const spritetype &blocker = sprite[probe.sprite];
            useable = (validXSprite(blocker.extra)
                       && (xsprite[blocker.extra].Push
                           || xsprite[blocker.extra].Vector))
                || acceptedDamageEffects(blocker) != llmapper::kEffectNone;
        }
        return llmapper::classifyTraversal(toFloor - fromFloor, clearance, playerStandingClearance(),
                                           playerJumpRiseLimit(), llmapper::capability::playerStepHeight(), probe.reachable,
                                           probe.wall >= 0 || probe.sprite >= 0,
                                           useable, probe.sector >= 0);
    }

    LocalWaypoint navJumpTakeoff(const NavRouteStep &step) const
    {
        if (step.hasTakeoff)
            return step.takeoff;
        // A gateway is the measured boundary crossing. The player's origin
        // generally cannot occupy that wall plane because clipmove stops the
        // body one collision radius short. The source nav cell is the actual
        // engine-confirmed standable pose from which this edge was derived.
        if (inRange(step.fromCell, 0, int(navCells.size())))
            return LocalWaypoint(navCells[size_t(step.fromCell)].center.x,
                                 navCells[size_t(step.fromCell)].center.y);
        return step.hasGateway ? step.gateway : step.destination;
    }

    bool navAirSourceEstablished(const NavRouteStep &step) const
    {
        if (step.mode != kNavJump && step.mode != kNavDrop)
            return false;
        if (!gMe || !gMe->pXSprite
            || !playerOccupiesSupportPose(step.sourceSupport, step.sourceZ)
            || observation.playerZVelocity != 0 || gMe->pXSprite->height != 0)
            return false;
        // Every air edge was validated from this concrete source pose.  For
        // a running jump it is the beginning of the proved run-up, not the
        // later takeoff boundary.
        if (!inRange(step.fromCell, 0, int(navCells.size())))
            return false;
        const NavWaypoint source = navCells[size_t(step.fromCell)].center;
        // A standing edge was validated with zero launch velocity.
        // with zero launch velocity. Collision-radius proximity is not the
        // same state: on a small support it can be a different lip, and any
        // retained velocity changes the arc.
        // A graph pose is a player-hull-sized region, not a mathematical
        // point.  The retained landing proof already allows the same hull
        // displacement; requiring sub-radius equality here created an
        // unreachable 64-unit point inside a controller whose real collision
        // body is wider than that.  Support identity and rest are the exact
        // physical facts; one clip radius is the engine-derived pose extent.
        const int poseTolerance = playerClipRadius();
        if (coastDistance() > poseTolerance)
            return false;
        return distance2(observation.x, observation.y,
                         source.x, source.y)
            <= int64_t(poseTolerance) * poseTolerance;
    }

    bool navAirTakeoffCrossed(const NavRouteStep &step) const
    {
        if (!step.hasTakeoff)
            return navAirSourceEstablished(step);
        if (!inRange(step.fromCell, 0, int(navCells.size())))
            return false;
        const NavWaypoint source = navCells[size_t(step.fromCell)].center;
        const LocalWaypoint takeoff = navJumpTakeoff(step);
        return llmapper::crossedWaypointCorridor(
            source, takeoff, NavWaypoint(observation.x, observation.y),
            playerClipRadius());
    }

    LocalWaypoint navStepWaypoint(const NavRouteStep &step) const
    {
        // ROR stores the translated receiving pose as its destination, but
        // execution occurs at the local source portal.  Crossing that pose
        // makes the engine establish the destination; steering directly at
        // the translated coordinates is physically meaningless.
        if (step.wall == kNavRorTransitionEdge && step.hasGateway)
            return step.gateway;
        const bool airTransition = step.mode == kNavJump
            || step.mode == kNavDrop;
        if (airTransition)
        {
            const bool ownsEdge = step.fromCell == jumpRouteFromCell
                && step.toCell == jumpRouteToCell;
            if (ownsEdge && jumpState == kJumpAirborne)
                return step.destination;
            if (ownsEdge && jumpState == kJumpTakeoff && step.hasTakeoff)
                return navJumpTakeoff(step);
            if (inRange(step.fromCell, 0, int(navCells.size())))
            {
                const NavWaypoint &source =
                    navCells[size_t(step.fromCell)].center;
                return LocalWaypoint(source.x, source.y);
            }
            return step.destination;
        }
        // A validated ordinary support transition executes toward its
        // concrete destination pose.  The gateway is a crossing constraint
        // used when building the edge, not another destination or execution
        // phase.  Switching between gateway and landing based on distance
        // creates a two-point oscillator at moving doors: one tick approaches
        // the wall plane, the next heads through it, then growing distance
        // from the plane sends the player back again.
        const bool supportTransition = step.targetZ != step.sourceZ
            || step.targetSupport != step.sourceSupport;
        if (step.hasGateway && !supportTransition)
            return step.gateway;
        return step.destination;
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
        int fromSupportZ = llmapper::capability::playerFloorZ();
        SupportRef fromSupport = currentPlayerSupport();
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
                // A shortcut whose endpoints belong to one Build container
                // may still leave it through an inner loop (a pit/shaft) and
                // re-enter on the far side. Preserve the concrete cell path
                // around that boundary; collapsing it loses the very crossing
                // identity the physical graph was built to retain.
                if (step.targetSector == fromSector
                    && segmentCrossesSectorWall(fromSector, fromX, fromY,
                                                toX, toY))
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
                // clipmove and cansee are both horizontal tests. They accept
                // a chord across an open pit, which turns a valid route around
                // the rim into a straight run through unsupported air. Keep
                // the shortcut only when live engine collision supplies a
                // continuous standable surface along the whole segment.
                if (!lineKeepsSupport(fromX, fromY, fromSector, fromSupportZ,
                                      fromSupport, toX, toY))
                    break;
                furthest = probe;
            }
            pulled.push_back(route[furthest]);
            const NavRouteStep &taken = route[furthest];
            fromX = taken.hasGateway ? taken.gateway.x : taken.destination.x;
            fromY = taken.hasGateway ? taken.gateway.y : taken.destination.y;
            if (taken.targetSector >= 0)
                fromSector = taken.targetSector;
            fromSupportZ = taken.targetZ;
            fromSupport = taken.targetSupport;
            anchor = furthest + 1;
        }
        if (pulled.size() < route.size())
            route.swap(pulled);
    }

    // Resolve task geometry to one concrete player pose.  This is the only
    // boundary where target coordinates/Build ownership are projected onto
    // navigation state; route planning below accepts pose identities only.
    int resolveNavPose(int targetX, int targetY, int targetZ, int targetSector,
                       const SupportRef *requiredSupport = nullptr,
                       int exactTargetCell = -1)
    {
        ensureNavTopology();
        const int start = currentNavStartCell();
        if (start < 0)
            return -1;
        const int startArea = navCells[size_t(start)].walkArea;
        // Object/switch Z is aiming geometry, not necessarily the floor on
        // which the player must stand.  Prefer the target XY on the support
        // component reachable now; only use targetZ to disambiguate when the
        // destination is genuinely on another (conditional) component.
        int areaTarget = -1;
        int strictTarget = -1;
        int target = -1;
        if (inRange(exactTargetCell, 0, int(navCells.size())))
        {
            // A caller that already proved a concrete physical pose must not
            // be projected back through sector/XY. Overlapping support layers
            // can share both and differ only in the support state that makes
            // the action possible.
            target = strictTarget = exactTargetCell;
        }
        else
        {
            areaTarget = requiredSupport ? -1
                : navCellInSectorArea(targetSector, targetX, targetY, startArea);
            strictTarget = navCellInSector(targetSector, targetX, targetY,
                                           targetZ, requiredSupport);
            target = areaTarget;
        }
        if (exactTargetCell < 0 && areaTarget >= 0 && strictTarget >= 0)
        {
            // A reachable support at the requested XY is the right stance
            // for a switch whose sprite Z is merely its aim point.  It is not
            // a valid substitute when it is on the far side of the sector:
            // AGTST8's first sprite bridge deliberately puts the doorway on a
            // different support component, and the old lookup silently chose
            // a same-area cell more than 6000 units away.  Retain the stance
            // tolerance, but preserve the exact support endpoint outside it
            // so planNavRoute can use the bridge's conditional jump edge.
            // Conversely, an overlapping/ROR layer can have the closest Z
            // while being thousands of units farther from the actuator in
            // XY.  Vertical similarity is not proof of causal support; only
            // prefer the strict endpoint when it is also the closer physical
            // projection.
            const int kTargetStanceSpan = 4 << kNavGridShift;
            const NavCell &areaCell = navCells[size_t(areaTarget)];
            const NavCell &strictCell = navCells[size_t(strictTarget)];
            target = llmapper::selectTargetNavCell(
                areaTarget,
                distance2(areaCell.center.x, areaCell.center.y,
                          targetX, targetY),
                strictTarget,
                distance2(strictCell.center.x, strictCell.center.y,
                          targetX, targetY),
                int64_t(kTargetStanceSpan) * kTargetStanceSpan);
        }
        else if (exactTargetCell < 0 && target < 0)
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
        return target;
    }

    // Install a route between two concrete pose cells.  In particular this
    // function cannot reinterpret the destination as an aim point, gateway,
    // wall, sector, or final task coordinate.
    bool installNavRoute(int targetCell, int signature,
                         bool currentGeometryOnly = false)
    {
        ensureNavTopology();
        const int start = currentNavStartCell();
        if (!inRange(start, 0, int(navCells.size()))
            || !inRange(targetCell, 0, int(navCells.size())))
            return false;
        std::vector<NavCell> liveRouteCells;
        const std::vector<NavCell> *routeCells = &navCells;
        if (currentGeometryOnly)
        {
            // Exact interaction stances are executable present-world plans.
            // Hypothetical stable mechanism endpoints remain in the broader
            // graph for prerequisite reasoning, but an inactive endpoint may
            // not act as a bridge between two live poses.
            liveRouteCells = navCells;
            for (NavCell &cell : liveRouteCells)
            {
                if (!cell.live)
                {
                    cell.links.clear();
                    continue;
                }
                cell.links.erase(std::remove_if(
                    cell.links.begin(), cell.links.end(),
                    [&](const NavLink &link) {
                        return !inRange(link.target, 0,
                                        int(liveRouteCells.size()))
                            || !liveRouteCells[size_t(link.target)].live;
                    }), cell.links.end());
            }
            routeCells = &liveRouteCells;
        }
        std::vector<NavRouteStep> route;
        if (!llmapper::planNavRoute(*routeCells, start, targetCell,
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
                     "start_cell=%d goal_cell=%d cells=%u failures=%u start_area=%d goal_area=%d",
                     start, targetCell,
                     unsigned(navCells.size()), unsigned(navEdgeFailures.size()),
                     start >= 0 && start < int(navCells.size()) ? navCells[size_t(start)].walkArea : -1,
                     navCells[size_t(targetCell)].walkArea);
            event("nav_route_unavailable", failure);
            return false;
        }
        navRouteRejectedSignature = 0;
        smoothNavRoute(route);
        // The player is already in the requested pose cell.  Any remaining
        // local approach or action belongs to the caller; do not manufacture
        // or install a transition that does not exist in the physical graph.
        if (route.empty())
        {
            navRoute.clear();
            navRouteIndex = 0;
            navRouteSignature = 0;
            return false;
        }
        navRoute = route;
        navRouteIndex = 0;
        navRouteSignature = signature;
        navRouteTopologyRevision = navTopologyRevision;
        char routeDetail[256];
        snprintf(routeDetail, sizeof(routeDetail),
                 "steps=%u start_cell=%d goal_cell=%d goal_pose=(%d,%d,z%d,s%d) player=(%d,%d,s%d)",
                 unsigned(navRoute.size()), start, targetCell,
                 navCells[size_t(targetCell)].center.x,
                 navCells[size_t(targetCell)].center.y,
                 navCells[size_t(targetCell)].z,
                 navCells[size_t(targetCell)].sector,
                 observation.x, observation.y, observation.sector);
        event("nav_route_selected", routeDetail);
        for (size_t i = 0; i < navRoute.size() && i < 16; ++i)
        {
            const NavRouteStep &step = navRoute[i];
            const LocalWaypoint waypoint = navStepWaypoint(step);
            char stepDetail[256];
            snprintf(stepDetail, sizeof(stepDetail),
                     "index=%u from_cell=%d to_cell=%d mode=%s wall=%d from_sector=%d to_sector=%d waypoint=(%d,%d) source_z=%d target_z=%d source_support=(%d,%d) target_support=(%d,%d)",
                     unsigned(i), step.fromCell, step.toCell,
                     llmapper::navEdgeModeName(step.mode), step.wall,
                     step.sourceSector, step.targetSector,
                     waypoint.x, waypoint.y, step.sourceZ, step.targetZ,
                     int(step.sourceSupport.kind), step.sourceSupport.id,
                     int(step.targetSupport.kind), step.targetSupport.id);
            event("nav_route_step", stepDetail);
        }
        return !navRoute.empty();
    }

    bool collisionProbe(int startX, int startY, int startZ, int startSector,
                        int targetX, int targetY, int targetSector, int tolerance) const
    {
        return probeMovement(startX, startY, startZ, startSector,
                             targetX, targetY, targetSector, tolerance).reachable;
    }

    // Verify continuity of one already named support without asking
    // updatesectorz() to reinterpret its Build ownership.  This is the
    // correct predicate for adjacent poses on the same sprite or overlapping
    // floor layer: both endpoints already name the physical surface, and a
    // different containing sector selected at the midpoint must not split it.
    bool lineRetainsSupport(int startX, int startY, int startSector,
                            int startSupportZ, const SupportRef &support,
                            int targetX, int targetY,
                            int targetSupportZ) const
    {
        const int dx = targetX - startX;
        const int dy = targetY - startY;
        const int length = int(std::sqrt(double(int64_t(dx) * dx
                                                + int64_t(dy) * dy)));
        const int samples = std::max(
            1, (length + playerClipRadius() - 1) / playerClipRadius());
        int previousZ = startSupportZ;
        const int querySector = support.kind == kSupportSectorFloor
            ? support.id : startSector;
        for (int sample = 0; sample <= samples; ++sample)
        {
            const int x = startX + int(int64_t(dx) * sample / samples);
            const int y = startY + int(int64_t(dy) * sample / samples);
            int supportZ = 0;
            if (!engineSupportZAt(support, x, y, supportZ)
                || std::abs(supportZ - previousZ)
                    > llmapper::capability::playerStepHeight())
                return false;
            if (support.kind == kSupportSectorFloor)
            {
                int ceilingZ = 0, ceilingHit = 0, floorZ = 0, floorHit = 0;
                GetZRangeAtXYZ(x, y, supportZ - 1, querySector,
                               &ceilingZ, &ceilingHit, &floorZ, &floorHit,
                               playerSupportRadius(), CLIPMASK0,
                               PARALLAXCLIP_CEILING | PARALLAXCLIP_FLOOR);
                // The player footprint may overlap an adjacent Build
                // polygon and make GetZRange name that coplanar sector as
                // floor owner.  The physical support plane is unchanged;
                // sector ownership is not a discontinuity.
                if ((floorHit & 0xc000) != 0x4000 || floorZ != supportZ)
                    return false;
            }
            else if (!engineHasSupportAt(
                         support, querySector, x, y, supportZ,
                         playerSupportRadius()))
                return false;
            previousZ = supportZ;
        }
        return previousZ == targetSupportZ;
    }

    // clipmove answers horizontal body clearance, not whether the floor
    // continues under the swept segment.  Sample the engine's standable
    // surfaces along a proposed direct line and carry forward only support
    // heights reachable by an ordinary step.  This accepts a long clear
    // room even if the coarse mesh happened to split it, while rejecting a
    // horizontally unobstructed line over a pit.  Sprite bridges naturally
    // work because their live floor surfaces participate in the same sweep.
    bool lineKeepsSupport(int startX, int startY, int startSector,
                          int startSupportZ, const SupportRef &startSupport,
                          int targetX, int targetY) const
    {
        if (!gMe || !gMe->pSprite || !inRange(startSector, 0, numsectors))
            return false;
        const int dx = targetX - startX;
        const int dy = targetY - startY;
        const int length = int(std::sqrt(double(int64_t(dx) * dx + int64_t(dy) * dy)));
        const int sampleSpan = std::max(1, playerClipRadius());
        const int samples = std::max(1, (length + sampleSpan - 1) / sampleSpan);
        int16_t sampleSector = int16_t(startSector);
        int supportZ = startSupportZ;
        if (startSupport.kind == kSupportSpriteFloor
            && inRange(startSupport.id, 0, kMaxSprites)
            && (xvel[startSupport.id] != 0 || yvel[startSupport.id] != 0
                || zvel[startSupport.id] != 0))
            return false;
        int previousX = startX;
        int previousY = startY;
        for (int step = 1; step <= samples; ++step)
        {
            const int x = startX + int(int64_t(dx) * step / samples);
            const int y = startY + int(int64_t(dy) * step / samples);
            int16_t targetSector = sampleSector;
            updatesectorz(x, y, playerOriginAtSupport(supportZ), &targetSector);
            if (!inRange(int(targetSector), 0, numsectors))
                return false;
            const MovementProbe movement = probeMovement(
                previousX, previousY, playerOriginAtSupport(supportZ),
                sampleSector, x, y, targetSector,
                std::max(1, playerClipRadius() / 4));
            if (!movement.reachable)
                return false;
            sampleSector = int16_t(movement.sector);
            int ceilingZ = 0, ceilingHit = 0, floorZ = 0, floorHit = 0;
            GetZRangeAtXYZ(x, y, supportZ - 1, sampleSector,
                           &ceilingZ, &ceilingHit, &floorZ, &floorHit,
                           playerClipRadius() + 16, CLIPMASK0,
                           PARALLAXCLIP_CEILING | PARALLAXCLIP_FLOOR);
            if (floorZ - ceilingZ < playerStandingClearance())
                return false;
            if (std::abs(floorZ - supportZ) > llmapper::capability::playerStepHeight())
                return false;
            if ((floorHit & 0xc000) == 0xc000)
            {
                const int nextSupportId = floorHit & 0x3fff;
                // A moving solid sprite is real collision geometry, but it
                // is not a stable ordinary-walk shortcut. Treat boarding it
                // as its own transport/support transition so route execution
                // can reason about where it goes; otherwise a passing gib can
                // momentarily validate a line across a pit and move away
                // under the player.
                if (inRange(nextSupportId, 0, kMaxSprites)
                    && (xvel[nextSupportId] != 0
                        || yvel[nextSupportId] != 0
                        || zvel[nextSupportId] != 0))
                    return false;
            }
            supportZ = floorZ;
            previousX = x;
            previousY = y;
        }
        return true;
    }

    bool directLineKeepsSupport(int targetX, int targetY) const
    {
        return lineKeepsSupport(observation.x, observation.y,
                                observation.sector,
                                llmapper::capability::playerFloorZ(),
                                currentPlayerSupport(),
                                targetX, targetY);
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
        if ((jumpState == kJumpTakeoff || jumpState == kJumpAirborne)
            && jumpRouteStepValid)
        {
            event("navigation_reset_deferred",
                  "reason=airborne_physical_transaction");
            return;
        }
        if (jumpState != kJumpInactive)
        {
            char detail[192];
            snprintf(detail, sizeof(detail),
                     "state=%d edge=%d->%d navigation_signature=%d route_signature=%d",
                     int(jumpState), jumpRouteFromCell, jumpRouteToCell,
                     navigationOriginalSignature, navRouteSignature);
            event("jump_commitment_reset", detail);
        }
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
        jumpState = kJumpInactive;
        jumpStateTick = -1;
        jumpRouteFromCell = -1;
        jumpRouteToCell = -1;
        jumpRouteStepValid = false;
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
            && std::abs(floorDelta) <= llmapper::capability::playerStepHeight())
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
        const int memoryKey = interactionKey(candidate);
        const auto known = interactions.find(memoryKey);
        const bool newlyDiscovered = known == interactions.end()
            || !known->second.observed;
        rememberInteraction(candidate);
        if (newlyDiscovered && currentObjective.active
            && (currentObjective.type == kObjectiveFrontier
                || currentObjective.type == kObjectiveInvestigate))
        {
            event("frontier_actionable_blocker_discovered",
                  "reason=engine_collision_identified_world_affordance");
            invalidateObjective("actionable_blocker_discovered");
        }
        return true;
    }

    bool rememberSpriteCollisionInteraction(int spriteIndex)
    {
        if (!inRange(spriteIndex, 0, kMaxSprites)
            || !inRange(observation.sector, 0, numsectors))
            return false;
        const spritetype &blocker = sprite[spriteIndex];
        const bool operable = validXSprite(blocker.extra)
            && (xsprite[blocker.extra].Push || xsprite[blocker.extra].Vector);
        const unsigned effects = acceptedDamageEffects(blocker);
        if (!operable && effects == llmapper::kEffectNone)
            return false;

        InteractionCandidate candidate;
        candidate.kind = kInteractionSprite;
        candidate.id = spriteIndex;
        candidate.fromSector = observation.sector;
        candidate.targetSector = blocker.sectnum;
        candidate.x = blocker.x;
        candidate.y = blocker.y;
        candidate.z = blocker.z;
        // Collision is evidence about a blocked body, not merely about any
        // Vector trigger the same XSPRITE may also expose. If accepted damage
        // can remove that body, retain a distinct removal affordance.
        const bool removable = effects != llmapper::kEffectNone;
        candidate.reversible = operable && !removable;
        candidate.activationMode = removable ? llmapper::kActivateDamage
            : xsprite[blocker.extra].Vector
                ? llmapper::kActivateVector : llmapper::kActivateUse;
        candidate.requiredEffects = removable ? effects
            : unsigned(llmapper::kEffectNone);
        candidate.target.wall = -1;
        candidate.target.from = observation.sector;
        candidate.target.to = blocker.sectnum;
        candidate.target.x = blocker.x;
        candidate.target.y = blocker.y;
        candidate.target.blockedBySprite = true;
        candidate.target.blockerSprite = spriteIndex;
        candidate.target.shootable = operable && xsprite[blocker.extra].Vector;
        candidate.target.mechanismTx = validXSprite(blocker.extra)
            ? xsprite[blocker.extra].txID : 0;
        setSpriteInteractionGeometry(candidate.target, spriteIndex);
        candidate.z = candidate.target.z;
        rememberInteraction(candidate);
        return true;
    }

    GINPUT navigateTo(int x, int y, int z, int targetSector, int targetId = -1,
                      TraversalCapability capability = kTraversalUnknown, bool shoot = false,
                      const SupportRef *requiredSupport = nullptr,
                      int exactTargetCell = -1)
    {
        navigationActionableBlocker = -1;
        int signature = navigationSignature(x, y, z, targetSector, targetId);
        if (requiredSupport)
            signature = llmapper::mixHash(signature,
                int(requiredSupport->kind) * 65536 + requiredSupport->id);
        if (exactTargetCell >= 0)
            signature = llmapper::mixHash(signature, exactTargetCell);
        auto installResolvedRoute = [&]() {
            const int pose = resolveNavPose(x, y, z, targetSector,
                                            requiredSupport,
                                            exactTargetCell);
            return pose >= 0 && installNavRoute(pose, signature);
        };
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
                installResolvedRoute();
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
                    && installResolvedRoute()))
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
        const bool physicalRouteActive = !navRoute.empty()
            && navRouteSignature == signature
            && navRouteTopologyRevision == navTopologyRevision
            && navRouteIndex < navRoute.size();
        // A collision-only direct probe is an optimization for unplanned
        // local walking, never an authority over an existing support route.
        // Once the physical graph owns a journey, let it consume each named
        // support pose before considering a shortcut to the final XY. This is
        // essential when several floor layers are horizontally unobstructed
        // but only one is actually connected to the target affordance.
        if (direct.reachable && !supportCenterlineActive && !physicalRouteActive)
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
            // One engine jump may legitimately catch a later raised support
            // in the same planned chain.  This is common at short stairs:
            // GetZRange resolves the player's full cylinder against the next
            // riser before the player origin leaves the intervening Build
            // sector.  Do not wait forever for an earlier support which is
            // now physically below the grounded player.  Rejoin only this
            // already-validated route, at the first later target matching the
            // collision-owned support and height; this cannot invent a new
            // shortcut or skip an unplanned prerequisite.
            const bool groundedForRouteRejoin = observation.playerZVelocity == 0
                && gMe && gMe->pXSprite && gMe->pXSprite->height == 0;
            if (groundedForRouteRejoin && navRouteIndex < navRoute.size())
            {
                const SupportRef liveSupport = currentPlayerSupport();
                const int liveFloor = llmapper::capability::playerFloorZ();
                const NavRouteStep &pending = navRoute[navRouteIndex];
                // Conditional/mechanism route steps may deliberately carry
                // anonymous supports.  They are not physical claims about
                // where the engine must have grounded the player, so they
                // cannot participate in support-based route invalidation.
                // Treating (-1 -> -1) as a missed named edge caused a stable
                // route to be destroyed and rebuilt every decision.
                const bool pendingSupportsNamed = pending.sourceSupport.id >= 0
                    && pending.targetSupport.id >= 0;
                const bool pendingSourceMatches = playerOccupiesSupportPose(
                    pending.sourceSupport, pending.sourceZ);
                const bool pendingPoseMatches = playerOccupiesSupportPose(
                    pending.targetSupport, pending.targetZ);
                bool rejoinedLater = false;
                if (pendingSupportsNamed
                    && !pendingSourceMatches && !pendingPoseMatches)
                {
                    for (size_t rejoin = navRouteIndex + 1;
                         rejoin < navRoute.size(); ++rejoin)
                    {
                        const NavRouteStep &candidate = navRoute[rejoin];
                        if (!playerOccupiesSupportPose(candidate.targetSupport,
                                                       candidate.targetZ))
                            continue;
                        char detail[192];
                        snprintf(detail, sizeof(detail),
                                 "from_index=%u through_index=%u support_kind=%d support=%d floor_z=%d player_sector=%d",
                                 unsigned(navRouteIndex), unsigned(rejoin),
                                 int(liveSupport.kind), liveSupport.id, liveFloor,
                                 observation.sector);
                        event("nav_route_rejoined_at_later_pose", detail);
                        navRouteIndex = rejoin + 1;
                        jumpState = kJumpInactive;
                        jumpStateTick = -1;
                        jumpRouteFromCell = -1;
                        jumpRouteToCell = -1;
                        jumpRouteStepValid = false;
                        lastNavWaypointX = INT32_MIN;
                        lastNavWaypointY = INT32_MIN;
                        if (navRouteIndex >= navRoute.size())
                            navRoute.clear();
                        rejoinedLater = true;
                        break;
                    }

                    // A route is a sequence of named physical supports. If
                    // the engine settles the player on neither side of the
                    // pending edge nor on a later named pose, the route no
                    // longer describes reality (for example, a missed jump
                    // landed on the floor below). Re-anchor from collision-
                    // owned state instead of retrying the old edge from an
                    // invented takeoff support.
                    if (!rejoinedLater)
                    {
                        char detail[208];
                        snprintf(detail, sizeof(detail),
                                 "index=%u source_kind=%d source=%d target_kind=%d target=%d live_kind=%d live=%d live_z=%d reason=engine_grounded_on_unplanned_support",
                                 unsigned(navRouteIndex),
                                 int(pending.sourceSupport.kind), pending.sourceSupport.id,
                                 int(pending.targetSupport.kind), pending.targetSupport.id,
                                 int(liveSupport.kind), liveSupport.id, liveFloor);
                        event("nav_route_invalidated", detail);
                        navRoute.clear();
                        jumpState = kJumpInactive;
                        jumpStateTick = -1;
                        jumpRouteFromCell = -1;
                        jumpRouteToCell = -1;
                        jumpRouteStepValid = false;
                        lastNavWaypointX = INT32_MIN;
                        lastNavWaypointY = INT32_MIN;
                        installResolvedRoute();
                    }
                }
            }
            if (navRoute.empty())
                return steerTo(x, y, false, shoot, targetId, targetSector, capability);
            const NavRouteStep &step = navRoute[navRouteIndex];
            const LocalWaypoint waypoint = navStepWaypoint(step);
            const int waypointX = waypoint.x;
            const int waypointY = waypoint.y;
            // A route waypoint is a concrete physical pose/crossing, not the
            // coarse final-object target.  Treating it as reached from one
            // whole grid cell away discards corners before Caleb gets to
            // them; his retained velocity then cuts across pits and other
            // inner-loop boundaries even though every graph edge is valid.
            // Consume physical poses at body scale.  The broader tolerance
            // above remains appropriate for the final interaction/object.
            const int routeTolerance = std::max(64, radius);
            const bool withinWaypoint =
                distance2(observation.x, observation.y, waypointX, waypointY)
                    <= routeTolerance * routeTolerance;
            // A route edge represents a concrete physical pose transition.
            // Passing through its XY while airborne is not completion, and
            // neither is landing on a lower floor which happens to belong to
            // the same Build sector.  Require the engine to establish the
            // named support at the destination height before consuming every
            // jump/height/support transition.  This is deliberately agnostic
            // about whether that support is a sector floor, sprite, voxel, or
            // moving surface.
            const bool requiresPhysicalPose = step.mode == kNavJump
                || step.targetZ != step.sourceZ
                || step.targetSupport != step.sourceSupport;
            const bool touchingTargetSupport = requiresPhysicalPose
                && playerOccupiesSupportPose(step.targetSupport, step.targetZ);
            const bool groundedOnTargetPose = !requiresPhysicalPose
                || (touchingTargetSupport && observation.playerZVelocity == 0
                    && gMe && gMe->pXSprite && gMe->pXSprite->height == 0
                    && std::abs(llmapper::capability::playerFloorZ() - step.targetZ)
                        <= llmapper::capability::playerStepHeight() / 2);
            const bool rorTransition = step.wall == kNavRorTransitionEdge;
            const bool rorTransitionReached = rorTransition
                && observation.sector == step.targetSector;
            const bool rorTransitionPending = rorTransition
                && !rorTransitionReached;
            const bool finalSupportPending = requiredSupport
                && navRouteIndex + 1 == navRoute.size()
                && !playerOccupiesSupportPose(*requiredSupport, step.targetZ);
            const bool actorGrounded = observation.playerZVelocity == 0
                && gMe && gMe->pXSprite && gMe->pXSprite->height == 0;
            // An ordinary grid sample is a progress cross-section through a
            // continuous walk corridor, not a point Caleb must touch.  At
            // running speed the body can cross that section slightly beside
            // its sampled centre.  Turning back to hit the centre loses the
            // route's direction and creates a loop at every bend.  Consume
            // the sample when the body has crossed its forward plane while
            // remaining inside the same sampled corridor.  Real transitions
            // (support/height changes, portals, jumps) retain the exact
            // engine-state requirements above.
            bool crossedOrdinaryWaypoint = false;
            if (!requiresPhysicalPose && actorGrounded
                && step.wall < 0 && step.mode == kNavWalk
                && step.sourceSupport == step.targetSupport
                && inRange(step.fromCell, 0, int(navCells.size())))
            {
                const NavWaypoint &source =
                    navCells[size_t(step.fromCell)].center;
                const int corridorHalfWidth = 3 * (1 << kNavGridShift) / 4;
                crossedOrdinaryWaypoint = llmapper::crossedWaypointCorridor(
                    source, waypoint, NavWaypoint(observation.x, observation.y),
                    corridorHalfWidth);
            }
            // A rising player can be snapped to a solid top for one frame
            // while retaining upward velocity.  Do not turn that transient
            // contact into the next route leg; neutral input lets the same
            // named support catch the descending body.
            if (requiresPhysicalPose && touchingTargetSupport
                && !groundedOnTargetPose && observation.playerZVelocity == 0)
                return brakeAtSupport(step.targetSupport, step.targetSector,
                                      step.targetZ, playerClipRadius());
            // A landing transition still owns input while the retained arc
            // sheds its horizontal velocity.  Advancing the route at first
            // floor contact handed that velocity to an unrelated next plan
            // and let Caleb coast off narrow supports he had reached
            // correctly.
            if (requiresPhysicalPose && groundedOnTargetPose
                && coastDistance() > playerClipRadius())
                return brakeAtSupport(step.targetSupport, step.targetSector,
                                      step.targetZ, playerClipRadius());
            const bool waypointReached = rorTransitionReached
                || ((requiresPhysicalPose ? groundedOnTargetPose
                                          : actorGrounded
                                              && (withinWaypoint
                                                  || crossedOrdinaryWaypoint))
                    && !finalSupportPending && !rorTransitionPending);
            if (waypointReached)
            {
                if (requiresPhysicalPose)
                {
                    char landed[160];
                    snprintf(landed, sizeof(landed),
                             "support_kind=%d support=%d sector=%d floor_z=%d target_z=%d",
                             int(step.targetSupport.kind), step.targetSupport.id,
                             observation.sector,
                             llmapper::capability::playerFloorZ(), step.targetZ);
                    event("nav_physical_pose_established", landed);
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
                // A jump can spend several simulated seconds above the
                // destination before descending onto it.  The XY progress
                // clock must not reject that physical edge while the engine
                // still reports an airborne body; failure is meaningful only
                // after it has settled somewhere and the requested pose was
                // not established.
                const bool waypointStalled = observation.playerZVelocity == 0
                    && !(next.mode == kNavJump
                         && jumpState == kJumpAirborne)
                    && observation.tick - navWaypointProgressTick
                        >= kWaypointStallTicks;
                if (!waypointStalled)
                {
                    // The route's current physical edge owns locomotion.  A
                    // jump required at the final portal must not leak backward
                    // into ordinary WALK waypoints leading to its takeoff.
                    TraversalCapability cap = kTraversalUnknown;
                    // The physical graph owns the traversal capability.  A
                    // local probe taken while Caleb is still airborne can
                    // classify the next perfectly flat WALK waypoint as a
                    // jump.  Promoting that transient observation restarted
                    // the jump state machine in mid-flight and made the bot
                    // bounce around the AGTST3 takeoff instead of settling on
                    // the named support.  Probes may reject an edge, but may
                    // not change its traversal mode during execution.
                    if (next.mode == kNavJump)
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
                if (installResolvedRoute())
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

        if (direct.sprite >= 0)
        {
            ensureNavTopology();
            if (navRouteRejectedSignature != signature
                && installResolvedRoute())
            {
                const NavRouteStep &waypoint = navRoute.front();
                const LocalWaypoint selectedWaypoint = navStepWaypoint(waypoint);
                return steerTo(selectedWaypoint.x, selectedWaypoint.y,
                               false, false, targetId,
                               waypoint.targetSector >= 0
                                   ? waypoint.targetSector
                                   : observation.sector,
                               capability);
            }
            // Collision creates blocker work only when it obstructs an
            // already-discovered persistent objective and the physical graph
            // found no alternative. A disposable coverage viewpoint is not
            // itself useful work, so bumping into scenery while looking
            // around must not turn every gib into a destruction mission.
            const bool blocksPersistentWork = currentObjective.active;
            const bool actionable = blocksPersistentWork
                && rememberSpriteCollisionInteraction(direct.sprite);
            char detail[208];
            snprintf(detail, sizeof(detail),
                     "sprite=%d target=%d persistent_work=%d alternative_route=0 actionable=%d stop=(%d,%d) hit=0x%x",
                     direct.sprite, targetId, blocksPersistentWork ? 1 : 0,
                     actionable ? 1 : 0, direct.x, direct.y, direct.hit);
            event("navigation_sprite_blocked", detail);
        }
        else if (direct.wall >= 0)
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
                // The collision owns only the immediate raised crossing.  It
                // says nothing about the traversal mode of the distant final
                // target.  Passing that target to the jump executor made one
                // local ledge launch toward an arbitrary cell several rooms
                // away.  Materialize the concrete support/crossing sequence
                // and let its first physical edge own the jump instead.
                ensureNavTopology();
                if (navRouteRejectedSignature != signature
                    && installResolvedRoute())
                {
                    const NavRouteStep &next = navRoute.front();
                    const LocalWaypoint waypoint = navStepWaypoint(next);
                    const TraversalCapability nextCapability =
                        (next.mode == kNavJump || next.mode == kNavDrop)
                            ? kTraversalJumpable
                            : next.mode == kNavCrouch
                                ? kTraversalCrouchable : kTraversalUnknown;
                    char detail[176];
                    snprintf(detail, sizeof(detail),
                             "collision_wall=%d route_wall=%d mode=%s from_cell=%d to_cell=%d",
                             direct.wall, next.wall,
                             llmapper::navEdgeModeName(next.mode),
                             next.fromCell, next.toCell);
                    event("navigation_collision_routed", detail);
                    return steerTo(waypoint.x, waypoint.y, false, shoot,
                                   targetId,
                                   next.targetSector >= 0
                                       ? next.targetSector : observation.sector,
                                   nextCapability);
                }
                event("nav_edge", "mode=JUMP reason=no_physical_route_to_goal");
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
                && installResolvedRoute())
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
                && installResolvedRoute())
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
                objectiveWorkHash(currentObjective), observation.sector);
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

    // A point just beyond the threshold, on the far side of the boundary.
    bool physicallyOnPortalNearSide(const Portal &portal)
    {
        if (observation.sector == portal.from)
            return true;
        const int nearDistance = std::max(4096, playerClipRadius() * 4);
        if (distance2(observation.x, observation.y, portal.x, portal.y)
                > int64_t(nearDistance) * nearDistance)
            return false;
        // Build can relabel the actor to the enclosing floor for a moment
        // immediately after a ledge landing.  Preserve the source-side fact
        // only inside that small temporal and spatial envelope.  A remote
        // route which left the source several seconds ago (such as a lower
        // sprite bridge under an upper doorway) gets no such alias.
        if (lastTransitionFrom == portal.from
            && observation.tick - lastTransitionTick <= kTicsPerSec)
            return true;
        const int startCell = currentNavStartCell();
        const int approachCell = navCellInSector(
            portal.from, portal.x, portal.y, portal.fromFloorZ);
        if (inRange(startCell, 0, int(navCells.size()))
            && inRange(approachCell, 0, int(navCells.size())))
        {
            // Equal XY/Z in overlapping Build containers is not equal
            // physical space.  When collision-derived support cells exist
            // for both poses, their connected component is authoritative.
            // This keeps a route on a lower sprite bridge from being
            // truncated merely because an upper doorway overlaps it.
            return navCells[size_t(startCell)].walkArea
                == navCells[size_t(approachCell)].walkArea;
        }
        // Sector identity is only a container label.  When two layers or a
        // solid support hand the player between Build sectors at the same
        // pose, floor height and proximity provide the stable physical
        // identity needed to finish the committed crossing.
        const int floorZ = llmapper::capability::playerFloorZ();
        if (std::abs(floorZ - portal.fromFloorZ) <= llmapper::capability::playerStepHeight())
            return true;
        // GetZRange can switch to the lower enclosing sector floor for a
        // frame while the player is airborne over a solid support.  Direct
        // line of sight to the remembered near-side floor is the equivalent
        // physical evidence in that transient state, and naturally also
        // covers an open ROR boundary.
        return cansee(observation.x, observation.y, observation.z,
                      observation.sector, portal.x, portal.y,
                       playerOriginAtSupport(portal.fromFloorZ), portal.from);
    }

    // What posture does this opening currently require?  Re-derived from the
    // live clearance so a door that is still rising is handled correctly.
    TraversalCapability portalPosture(const Portal &portal) const
    {
        if (portal.clearance >= playerStandingClearance())
            return portal.floorDelta < -llmapper::capability::playerStepHeight() ? kTraversalJumpable
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

    // The physical edge owns the airborne control selected by
    // retainedAirTransition().  Execution replays it; there is no second,
    // collision-free jump solver here.
    int coastDistance() const
    {
        return llmapper::capability::playerCoastDistance(groundSpeed());
    }

    GINPUT brakeAtSupport(const SupportRef &support, int supportSector,
                          int supportZ, int stopTolerance)
    {
        GINPUT input = {};
        if (coastDistance() <= stopTolerance)
            return input;
        setGoal("SHED_LANDING_SPEED", -1);
        int driftAngle = 0;
        int driftSpeed = 0;
        if (!driftHeading(driftAngle, driftSpeed) || !gMe || !gMe->pSprite)
            return input;

        // Establishing a pose includes its usable velocity state.  Select
        // one frame of opposing input by replaying Blood's canonical motion
        // equation, and only accept candidates whose predicted body remains
        // on the named landing support.  This is active braking derived from
        // engine acceleration/drag, not a second movement heuristic.
        const int controlAngle = (driftAngle + 1024) & 2047;
        const POSTURE &posture =
            gMe->pPosture[gMe->lifeMode][gMe->posture];
        const int floorZ = llmapper::capability::playerFloorZ();
        const int foot = llmapper::capability::playerFootOffset();
        const llmapper::capability::MotionState before = currentMotion();
        int bestInput = 0;
        int64_t bestSpeed2 = int64_t(before.xvel) * before.xvel
            + int64_t(before.yvel) * before.yvel;
        constexpr int kBrakeSamples = 32;
        for (int sample = 1; sample <= kBrakeSamples; ++sample)
        {
            const int candidateInput = kFullThrottle * sample / kBrakeSamples;
            llmapper::capability::MotionState predicted = before;
            llmapper::capability::stepPlayerMotion(
                predicted, candidateInput, controlAngle, floorZ,
                posture.frontAccel, foot,
                llmapper::capability::playerAirDrag());
            if (!engineHasSupportAt(support,
                                       supportSector,
                                       predicted.x, predicted.y,
                                       supportZ,
                                       playerSupportRadius()))
                continue;
            const int64_t speed2 = int64_t(predicted.xvel) * predicted.xvel
                + int64_t(predicted.yvel) * predicted.yvel;
            if (speed2 < bestSpeed2)
            {
                bestSpeed2 = speed2;
                bestInput = candidateInput;
            }
        }
        const int controlDelta = angleDelta(controlAngle, observation.angle);
        input.forward = int16_t(mulscale30(Cos(controlDelta), bestInput));
        input.strafe = int16_t(-mulscale30(Sin(controlDelta), bestInput));
        return input;
    }

    // A route through a crouch-only or otherwise intermediate pose is not
    // consumed merely because the engine changed Build-sector ownership.
    // Continue to the nearest live pose on the far side where standing is
    // physically possible. This is a graph property, not corridor/door
    // special handling, and prevents task state from stopping the player
    // inside a transient collision envelope.
    int routePosePastBoundary(int startCell, int sourceSector,
                              bool leaveReceivingSector) const
    {
        if (!inRange(startCell, 0, int(navCells.size())))
            return startCell;
        std::vector<char> seen(navCells.size(), 0);
        std::deque<int> queue;
        seen[size_t(startCell)] = 1;
        queue.push_back(startCell);
        const int receivingSector = navCells[size_t(startCell)].sector;
        while (!queue.empty())
        {
            const int current = queue.front();
            queue.pop_front();
            const NavCell &cell = navCells[size_t(current)];
            if (current != startCell && cell.live && !cell.crouchOnly
                && cell.sector != sourceSector
                && (!leaveReceivingSector || cell.sector != receivingSector))
                return current;
            for (const NavLink &link : cell.links)
            {
                if (!llmapper::traversableMode(link.mode)
                    || link.condition.enabled
                    || !inRange(link.target, 0, int(navCells.size()))
                    || seen[size_t(link.target)]
                    || !navCells[size_t(link.target)].live
                    || llmapper::edgeFailedAny(
                        navEdgeFailures, current, link.target, link.wall,
                        link.mode, 0))
                    continue;
                seen[size_t(link.target)] = 1;
                queue.push_back(link.target);
            }
        }
        return startCell;
    }

    GINPUT steerPortal(const Portal &knownPortal)
    {
        // A boundary is observation evidence, not a second executable route.
        // Resolve its receiving-side physical floor pose and let the one
        // transition graph and route executor perform WALK/STEP/CROUCH/JUMP/
        // DROP as derived for that concrete edge.
        Portal portal = knownPortal;
        const Portal *live = portalByWall(portal.wall, portal.from, portal.to);
        if (live)
            portal = *live;
        if (!inRange(portal.to, 0, numsectors))
            return GINPUT{};

        ensureNavTopology();
        const SupportRef targetSupport(kSupportSectorFloor, portal.to);
        int targetCell = navCellInSector(
            portal.to, portal.x, portal.y, portal.toFloorZ, &targetSupport);
        if (!inRange(targetCell, 0, int(navCells.size())))
        {
            event("transition_plan_failed",
                  "type=boundary reason=no_receiving_physical_pose");
            return GINPUT{};
        }
        const DoorTiming receivingMotion = doorTiming(portal.to);
        const bool transientReceivingSpace =
            (receivingMotion.moving || receivingMotion.autoCloses)
            && !sectorSweptOccupancySafe(portal.to);
        if (navCells[size_t(targetCell)].crouchOnly || transientReceivingSpace)
            targetCell = routePosePastBoundary(
                targetCell, portal.from, transientReceivingSpace);
        const NavCell &pose = navCells[size_t(targetCell)];
        return navigateTo(pose.center.x, pose.center.y, pose.z, pose.sector,
                          portal.wall, portalPosture(portal), false,
                          &pose.support, targetCell);
    }
    void executeJumpTraversal(GINPUT &input, int x, int y, int targetSector,
                              int targetId)
    {
        if (!gMe || !gMe->pSprite)
            return;
        const NavRouteStep *supportStep = nullptr;
        if (!navRoute.empty() && navRouteIndex < navRoute.size())
        {
            const NavRouteStep &candidate = navRoute[navRouteIndex];
            const LocalWaypoint waypoint = navStepWaypoint(candidate);
            if ((candidate.mode == kNavJump || candidate.mode == kNavDrop)
                && waypoint.x == x && waypoint.y == y)
                supportStep = &candidate;
        }
        if (!supportStep && jumpState != kJumpInactive && jumpRouteStepValid
            && jumpRouteStep.fromCell == jumpRouteFromCell
            && jumpRouteStep.toCell == jumpRouteToCell)
            supportStep = &jumpRouteStep;
        const bool samePhysicalJumpEdge = supportStep
            && supportStep->fromCell == jumpRouteFromCell
            && supportStep->toCell == jumpRouteToCell;
        const int landingX = supportStep ? supportStep->destination.x : x;
        const int landingY = supportStep ? supportStep->destination.y : y;
        const LocalWaypoint physicalTakeoff = supportStep
            ? navJumpTakeoff(*supportStep) : LocalWaypoint(x, y);
        const int landingAngle = supportStep
            ? getangle(landingX - physicalTakeoff.x,
                       landingY - physicalTakeoff.y)
            : getangle(landingX - observation.x, landingY - observation.y);
        const int landingAngleError = angleDelta(landingAngle,
                                                 observation.angle);
        if (jumpState == kJumpInactive
            || (!samePhysicalJumpEdge
                && (jumpTargetX != x || jumpTargetY != y
                    || jumpTargetSector != targetSector)))
        {
            // A jump is executable only as a concrete physical route edge.
            // The caller may retain the underlying task when no such edge is
            // available, but it must not manufacture jump input from a
            // generic movement target.
            if (!supportStep)
            {
                event("jump_transition_rejected", "reason=no_owning_physical_edge");
                return;
            }
            jumpState = kJumpApproachTakeoff;
            jumpStateTick = observation.tick;
            jumpTargetX = x;
            jumpTargetY = y;
            jumpTargetSector = targetSector;
            jumpSourceSector = observation.sector;
            jumpRouteFromCell = supportStep ? supportStep->fromCell : -1;
            jumpRouteToCell = supportStep ? supportStep->toCell : -1;
            jumpRouteStepValid = supportStep != nullptr;
            if (supportStep)
                jumpRouteStep = *supportStep;
            char detail[320];
            snprintf(detail, sizeof(detail),
                     "phase=%s route_mode=JUMP target=%d source_sector=%d target_sector=%d target=(%d,%d) player=(%d,%d,%d) posture=%d zvel=%d cant_jump=%d",
                     "APPROACH_TAKEOFF",
                     targetId, observation.sector, targetSector, x, y, observation.x, observation.y,
                     observation.z, gMe->posture, observation.playerZVelocity, gMe->cantJump);
            event("jump_traversal", detail);
        }

        const int distance = int(std::sqrt(double(distance2(observation.x, observation.y, x, y))));
        const bool groundedAtRoutePose = supportStep
            && playerOccupiesSupportPose(supportStep->targetSupport,
                                         supportStep->targetZ)
            && observation.playerZVelocity == 0
            && gMe->pXSprite && gMe->pXSprite->height == 0
            && std::abs(llmapper::capability::playerFloorZ() - supportStep->targetZ)
                <= llmapper::capability::playerStepHeight() / 2;
        // Sector membership is only a side effect of a physical jump. Success
        // means landing on the concrete target support/height.
        if (jumpState != kJumpApproachTakeoff && groundedAtRoutePose)
        {
            // Landing contact is not yet an executable destination pose when
            // retained velocity would carry the body off its support.  Keep
            // this edge alive and let the canonical motion controller
            // establish the pose before committing it to the route.
            if (coastDistance() > playerClipRadius())
            {
                input = brakeAtSupport(supportStep->targetSupport,
                                       supportStep->targetSector,
                                       supportStep->targetZ,
                                       playerClipRadius());
                return;
            }
            // The jump edge still owns this landing frame.  steerTo() has
            // already composed movement toward the route waypoint; allowing
            // that inherited input through after contact adds acceleration
            // which was not part of the retained-landing proof and can carry
            // the player straight off a narrow support before the route loop
            // gets its next observation and brakes.
            input.forward = 0;
            input.strafe = 0;
            input.q16turn = 0;
            event("jump_traversal", "phase=SUCCESS reason=physical_pose_established");
            // The executor owns this concrete transition and therefore owns
            // its commit.  Leaving navRouteIndex on the completed jump until
            // another planning pass made the next frame restart the old edge
            // from its landing support, corrupting an otherwise correct
            // multi-step route.
            if (supportStep && navRouteIndex < navRoute.size()
                && navRoute[navRouteIndex].fromCell == supportStep->fromCell
                && navRoute[navRouteIndex].toCell == supportStep->toCell)
            {
                ++navRouteIndex;
                if (navRouteIndex >= navRoute.size())
                    navRoute.clear();
                else
                    event("nav_retarget_original");
            }
            jumpState = kJumpInactive;
            jumpRouteFromCell = -1;
            jumpRouteToCell = -1;
            jumpRouteStepValid = false;
            return;
        }
        auto rejectJumpEdge = [&](const char *reason) {
            char detail[160];
            snprintf(detail, sizeof(detail), "reason=%s edge=%d->%d",
                     reason, jumpRouteFromCell, jumpRouteToCell);
            event("jump_transition_rejected", detail);
            if (supportStep)
            {
                bool recorded = false;
                for (NavEdgeFailure &failure : navEdgeFailures)
                    if (failure.wall < 0
                        && failure.fromCell == supportStep->fromCell
                        && failure.toCell == supportStep->toCell
                        && failure.mode == supportStep->mode)
                    {
                        ++failure.attempts;
                        recorded = true;
                        break;
                    }
                if (!recorded)
                {
                    NavEdgeFailure failure;
                    failure.fromCell = supportStep->fromCell;
                    failure.toCell = supportStep->toCell;
                    failure.wall = -1;
                    failure.mode = supportStep->mode;
                    failure.geometrySignature = navTopologyRevision;
                    failure.attempts = 1;
                    navEdgeFailures.push_back(failure);
                }
            }
            jumpState = kJumpInactive;
            jumpRouteFromCell = -1;
            jumpRouteToCell = -1;
            jumpRouteStepValid = false;
            navRoute.clear();
            movementTargetActive = false;
            input.forward = 0;
            input.strafe = 0;
            input.q16turn = 0;
        };
        // One execution transaction gets one watchdog. Internal phase
        // changes are not new attempts and must not refresh the clock.
        const int validatedExecutionTicks = supportStep
            && supportStep->airFrames > 0
                ? (supportStep->airFrames + 8) * kTicsPerFrame
                : kJumpActionTimeoutTicks;
        if (jumpState == kJumpAirborne
            && observation.tick - jumpStateTick > validatedExecutionTicks)
        {
            const bool airborne = observation.playerZVelocity != 0
                || (gMe->pXSprite && gMe->pXSprite->height != 0);
            if (!airborne)
                rejectJumpEdge("execution_watchdog_without_result");
            return;
        }
        // Bot decisions can be many simulated frames apart in accelerated
        // play.  Therefore reaching a takeoff pose and emitting the jump
        // input must be one transaction.  The former two-decision sequence
        // left full-throttle run-up input latched while waiting for the next
        // decision and carried Caleb far beyond the lip before JUMP was ever
        // pressed.
        auto emitJumpFromCurrentPose = [&]() -> bool {
            input.forward = 0;
            input.strafe = 0;
            if (supportStep
                && !playerOccupiesSupportPose(supportStep->sourceSupport,
                                              supportStep->sourceZ))
                return false;
            if (supportStep && !supportStep->hasTakeoff
                && coastDistance() > playerClipRadius())
                return false;
            const bool jumpImpulse = supportStep->mode == kNavJump;
            if (std::abs(landingAngleError) >= 96
                || (jumpImpulse && gMe->cantJump)
                || observation.playerZVelocity != 0
                || (gMe->pXSprite && gMe->pXSprite->height != 0))
                return false;

            jumpHeading = landingAngle;
            if (!supportStep || !supportStep->hasAirControl)
            {
                rejectJumpEdge("physical_edge_missing_validated_control");
                return true;
            }
            input.forward = int16_t(supportStep->airControl);
            jumpForwardInput = supportStep->airControl;
            input.buttonFlags.jump = jumpImpulse ? 1 : 0;
            ++jumpAttempts;
            jumpState = kJumpAirborne;
            jumpStateTick = observation.tick;
            char detail[320];
            snprintf(detail, sizeof(detail),
                     "phase=TAKEOFF jump_requested=%d jump_input_emitted=%d target=%d target_sector=%d player_z=%d zvel_before=%d posture=%d cant_jump=%d attempts=%d yaw=%d forward=%d source=validated_physical_edge",
                     jumpImpulse ? 1 : 0, jumpImpulse ? 1 : 0,
                     targetId, targetSector, observation.z,
                     observation.playerZVelocity, gMe->posture,
                     gMe->cantJump, jumpAttempts, landingAngle,
                     int(input.forward));
            event("jump_input_emitted", detail);
            return true;
        };
        switch (jumpState)
        {
        case kJumpApproachTakeoff:
            if (supportStep && !navAirSourceEstablished(*supportStep))
            {
                // The validated edge begins at its from-cell, at rest.  That
                // is a physical pose, distinct from its later crossing/lip.
                // Take ownership early enough to brake onto that pose rather
                // than allowing the route walker to consume it as an
                // ordinary waypoint and starting the arc from arbitrary
                // retained velocity.
                const NavWaypoint &source =
                    navCells[size_t(supportStep->fromCell)].center;
                const int sourceDistance = int(std::sqrt(double(distance2(
                    observation.x, observation.y, source.x, source.y))));
                const int sourceCoast = coastDistance();
                const int poseTolerance = playerClipRadius();
                if (sourceCoast > poseTolerance
                    && sourceDistance <= sourceCoast + poseTolerance)
                {
                    input = brakeAtSupport(supportStep->sourceSupport,
                                           supportStep->sourceSector,
                                           supportStep->sourceZ,
                                           poseTolerance);
                }
            }
            else if (std::abs(landingAngleError) >= 96)
            {
                input.forward = 0;
                input.strafe = 0;
                input.q16turn = fix16_from_int(landingAngleError);
                jumpState = kJumpAlign;
                event("jump_traversal", "phase=ALIGN reason=takeoff_heading");
            }
            else if (supportStep
                     || (!supportStep && distance <= kJumpTakeoffRange))
            {
                input.forward = 0;
                input.strafe = 0;
                jumpState = kJumpTakeoff;
                event("jump_traversal", "phase=TAKEOFF reason=source_pose_established");
                if (!supportStep || !supportStep->hasTakeoff)
                    emitJumpFromCurrentPose();
            }
            break;
        case kJumpAlign:
            input.forward = 0;
            input.strafe = 0;
            input.q16turn = fix16_from_int(landingAngleError);
            if (supportStep && !navAirSourceEstablished(*supportStep))
            {
                jumpState = kJumpApproachTakeoff;
                event("jump_traversal", "phase=APPROACH_TAKEOFF reason=left_source_pose");
            }
            else if (std::abs(landingAngleError) < 64)
            {
                jumpState = kJumpTakeoff;
                event("jump_traversal", "phase=TAKEOFF reason=heading_aligned");
                if (!supportStep || !supportStep->hasTakeoff)
                    emitJumpFromCurrentPose();
            }
            break;
        case kJumpTakeoff:
            input.q16turn = fix16_from_int(landingAngleError);
            input.strafe = 0;
            if (supportStep && supportStep->hasTakeoff
                && !navAirTakeoffCrossed(*supportStep))
            {
                // Replay the proved run-up from the exact source pose to its
                // concrete crossing.  The transition executor, rather than
                // the surrounding route walker, owns this movement.
                input.forward = kFullThrottle;
            }
            else
                emitJumpFromCurrentPose();
            break;
        case kJumpAirborne:
            // A committed arc has one flight axis. Turning back toward a
            // waypoint after passing it converts forward/reverse trim into a
            // spiral and invalidates the motion prediction used to choose
            // the input. Hold the launch heading; signed forward input can
            // still accelerate or brake along that same physical axis.
            input.q16turn = fix16_from_int(
                angleDelta(jumpHeading, observation.angle));
            input.forward = int16_t(jumpForwardInput);
            if (std::abs(observation.playerZVelocity) >= 64 || gMe->cantJump)
            {
                char detail[192];
                snprintf(detail, sizeof(detail),
                         "phase=AIRBORNE zvel=%d posture=%d cant_jump=%d sector=%d",
                         observation.playerZVelocity, gMe->posture, gMe->cantJump, observation.sector);
                event("jump_airborne_observed", detail);
            }
            break;
        default:
            break;
        }
    }

    GINPUT steerTo(int x, int y, bool use, bool shoot, int targetId = -1, int targetSector = -1,
                   TraversalCapability capability = kTraversalUnknown)
    {
        // A physical edge owns locomotion only while it is the current step
        // of the installed route.  Planning queries are pure, so there is no
        // second cached executor target to restore or arbitrate here.
        const bool physicalRouteOwnsMovement = !navRoute.empty()
            && navRouteIndex < navRoute.size()
            && navRouteSignature == navigationOriginalSignature
            && navRouteTopologyRevision == navTopologyRevision
            && navStepWaypoint(navRoute[navRouteIndex]).x == x
            && navStepWaypoint(navRoute[navRouteIndex]).y == y;
        // Posture belongs to the pose the body currently occupies, not only
        // to the edge that entered it.  A CROUCH edge followed by an ordinary
        // WALK edge through the same low volume must keep the hull crouched
        // until a standing pose is physically established.
        const bool occupiedPoseRequiresCrouch =
            currentClearanceCapability() == kTraversalCrouchable
            // A route that entered a crouch envelope owns that posture until
            // it establishes the route endpoint.  Rebuilding the remaining
            // route after a sector/support update must not silently stand the
            // body up in the middle of the same physical traversal.
            || (crouchTargetActive && physicalRouteOwnsMovement);
        const TraversalCapability movementCapability =
            (capability == kTraversalCrouchable || occupiedPoseRequiresCrouch)
                ? kTraversalCrouchable : capability;
        if (movementCapability != kTraversalCrouchable && crouchTargetActive)
        {
            event("crouch_released", "reason=normal_clearance");
            crouchTargetActive = false;
        }
        setMovementTarget(x, y, targetSector, targetId, movementCapability);
        updateMovementProgress();
        GINPUT input = {};
        const int targetAngle = getangle(x - observation.x, y - observation.y);
        const int movementDelta = angleDelta(targetAngle, observation.angle);
        int cameraAngle = targetAngle;
        // A physical WALK route is a polyline.  The current sample constrains
        // where the body moves, but looking back at that sample until the
        // instant it is consumed makes the camera describe every small
        // position correction.  Face along the next segment while movement
        // continues to satisfy the current corridor cross-section.  Do not
        // look through a real crossing or support transition.
        if (physicalRouteOwnsMovement
            && navRouteIndex + 1 < navRoute.size())
        {
            const NavRouteStep &current = navRoute[navRouteIndex];
            const NavRouteStep &lookahead = navRoute[navRouteIndex + 1];
            if (current.mode == kNavWalk && current.wall < 0
                && current.sourceSupport == current.targetSupport
                && lookahead.mode == kNavWalk && lookahead.wall < 0
                && lookahead.sourceSupport == current.targetSupport
                && lookahead.targetSupport == current.targetSupport)
            {
                const LocalWaypoint ahead = navStepWaypoint(lookahead);
                cameraAngle = getangle(ahead.x - observation.x,
                                       ahead.y - observation.y);
            }
        }
        const int cameraDelta = angleDelta(cameraAngle, observation.angle);
        const int targetDistance2 = distance2(observation.x, observation.y, x, y);
        input.syncFlags.run = 1;
        if (movementCapability == kTraversalCrouchable)
        {
            input.buttonFlags.crouch = 1;
            if (!crouchTargetActive)
            {
                crouchTargetActive = true;
                char detail[128];
                snprintf(detail, sizeof(detail),
                         "reason=physical_pose_requires_crouch speed=%d coast=%d target_distance=%d",
                         groundSpeed(), coastDistance(),
                         int(std::sqrt(double(targetDistance2))));
                event("crouch_started", detail);
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
        input.q16turn = fix16_from_int(cameraDelta);
        if (!use || targetDistance2 > kUseStopRange * kUseStopRange)
        {
            // Locomotion is a world-space intent.  Preserve that intent while
            // the camera turns by expressing it in the player's current
            // forward/strafe basis.  The old implementation stopped issuing
            // movement outside a narrow facing cone; Blood retained the
            // previous velocity, carried Caleb past the waypoint, and then
            // made the camera turn back toward the missed point.  Repeating
            // that feedback loop at every grid waypoint produced the apparent
            // search circles even though the objective and route were stable.
            int controlAngle = targetAngle;
            int controlMagnitude = kFullThrottle;
            const NavRouteStep *physicalStep = physicalRouteOwnsMovement
                ? &navRoute[navRouteIndex] : nullptr;
            const bool constrainedPose = physicalStep
                && physicalStep->mode != kNavJump
                && (physicalStep->mode != kNavWalk
                    || physicalStep->wall >= 0
                    || physicalStep->sourceZ != physicalStep->targetZ);
            if (constrainedPose && gMe && gMe->pSprite)
            {
                // A constrained transition ends at an occupiable pose, not
                // merely at an XY plane crossed at any velocity.  Select the
                // fastest target velocity whose passive stopping distance,
                // replayed from Blood's drag, still fits before the pose.
                // Then accelerate toward the velocity error.  This cancels
                // lateral momentum from the preceding polyline instead of
                // driving the player diagonally into a narrow opening.
                const int distance = int(std::sqrt(double(targetDistance2)));
                // A support-changing endpoint is not reached merely because
                // its XY is within the player's collision radius.  The body
                // must continue far enough for GetZRange to transfer support
                // ownership.  Using clipdist as the stopping tolerance made
                // short DROP/STEP edges command zero velocity forever while
                // still standing on their source support.
                constexpr int kPhysicalPoseTolerance = 64;
                const int stoppingRoom = std::max(
                    0, distance - kPhysicalPoseTolerance);
                int low = 0;
                int high = llmapper::capability::playerRunSpeed();
                while (low < high)
                {
                    const int candidate = low + (high - low + 1) / 2;
                    if (llmapper::capability::playerCoastDistance(candidate)
                        <= stoppingRoom)
                        low = candidate;
                    else
                        high = candidate - 1;
                }
                const int desiredSpeed = low;
                const int desiredX = distance > 0
                    ? int(int64_t(x - observation.x) * desiredSpeed / distance) : 0;
                const int desiredY = distance > 0
                    ? int(int64_t(y - observation.y) * desiredSpeed / distance) : 0;
                const int spriteIndex = gMe->pSprite->index;
                const int errorX = desiredX - (xvel[spriteIndex] >> 12);
                const int errorY = desiredY - (yvel[spriteIndex] >> 12);
                const int errorSpeed = int(std::sqrt(
                    double(int64_t(errorX) * errorX + int64_t(errorY) * errorY)));
                if (errorSpeed == 0)
                    controlMagnitude = 0;
                else
                    controlAngle = getangle(errorX, errorY);
            }
            const int controlDelta = angleDelta(controlAngle, observation.angle);
            input.forward = int16_t(
                mulscale30(Cos(controlDelta), controlMagnitude));
            // Blood's positive strafe axis is camera-left:
            //   velocity += (sin(yaw), -cos(yaw)) * strafe
            // so a positive relative target angle requires negative strafe.
            input.strafe = int16_t(
                -mulscale30(Sin(controlDelta), controlMagnitude));
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
            && std::abs(movementDelta) < 96)
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
        // A concrete physical edge already encodes its source pose, target
        // support, body clearance and traversal mode.  Do not reinterpret it
        // from the two enclosing sector floors: those may describe a lower
        // pit, another ROR layer, or simply the wrong point on a slope.  That
        // sector-floor override suppressed a real AGTST3 support jump and
        // walked Caleb off the ledge into the damaging space below.
        const bool ownsAirTransition = physicalRouteOwnsMovement
            && (navRoute[navRouteIndex].mode == kNavJump
                || navRoute[navRouteIndex].mode == kNavDrop);
        if (movementCapability == kTraversalJumpable || ownsAirTransition)
        {
            // A kNavJump route is an explicit Blood input sequence, never a
            // generic movement failure recovery.  playerProcess consumes the
            // normal GINPUT jump flag on the following game tick.
            executeJumpTraversal(input, x, y, targetSector, targetId);
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

        // A launched physical transition is an action transaction, not a
        // suggestion to the task selector.  Build-sector changes and newly
        // observed work may update knowledge during the flight, but cannot
        // replace its landing support or input sequence.  Resume ordinary
        // task selection only after the engine reports landing or failure.
        if (jumpState == kJumpAirborne && jumpRouteStepValid)
        {
            const NavRouteStep &step = jumpRouteStep;
            return steerTo(step.destination.x, step.destination.y, false, false,
                           movementTargetId, step.targetSector,
                           kTraversalJumpable);
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
        // Finish the concrete physical transition already in progress before
        // high-level task selection can ask another reachability question.
        // Ownership starts at the takeoff approach, not only after the body
        // becomes airborne: replanning during ALIGN used the pit floor below
        // a ledge as a new start support, overwrote the route, and walked the
        // player off AGTST15's valid sprite platform before jump input.
        if (jumpState != kJumpInactive && movementTargetActive
            && movementTargetCapability == kTraversalJumpable)
        {
            int executionX = movementTargetX;
            int executionY = movementTargetY;
            int executionSector = movementTargetSector;
            if (jumpRouteStepValid)
            {
                const LocalWaypoint pose = navStepWaypoint(jumpRouteStep);
                executionX = pose.x;
                executionY = pose.y;
                executionSector = jumpRouteStep.targetSector;
            }
            return composeInput(intentFromInput(steerTo(
                                    executionX, executionY,
                                    false, false, movementTargetId,
                                    executionSector,
                                    movementTargetCapability)), combat, use);
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
        if (hazardousSupport(currentPlayerSupport()))
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
            if (currentObjective.active)
            {
                event("mission_interrupted", "reason=key_acquired");
                invalidateObjective("superseded_by_key_acquisition");
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
        if (!currentObjective.active)
        {
            rebuildLedger();
            // Selection is the point at which the work model actually drives
            // behaviour.  A bounded snapshot here makes contradictions such
            // as "known but never selectable" diagnosable without a second
            // telemetry-only model of the world.
            dumpLedger("mission_selection", false);
            // A ledger entry can become physically stale between projection
            // and objective construction (for example, an expose cell is
            // consumed by the same observation).  Reject only that candidate
            // for this selection pass and try the next conserved piece of
            // work.  Repeatedly selecting the same uncommittable head entry
            // otherwise turns one stale hypothesis into NO_APPLICABLE_ACTION.
            bool committed = false;
            for (size_t attempt = 0; attempt <= ledger.size() && !committed; ++attempt)
            {
                llmapper::WorkSelection chosen = llmapper::selectWork(
                    ledger, heldKeyMask, availableEffectCapabilities());
                if (!chosen)
                    break;
                if (commitWork(chosen))
                {
                    committed = true;
                    break;
                }
                char detail[128];
                snprintf(detail, sizeof(detail),
                         "work_kind=%d subject=%d from=%d to=%d reason=objective_construction_rejected",
                         int(chosen.work.kind), chosen.work.subject,
                         chosen.work.from, chosen.work.to);
                event("work_candidate_rejected", detail);
                for (llmapper::Opportunity &candidate : ledger)
                    if (candidate.id == chosen.work)
                        candidate.hops = -1;
            }
            if (!committed)
            {
                selectedWorkReason = "NO_APPLICABLE_ACTION";
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
        if (mechanismBusy && currentGoal != "NO_APPLICABLE_ACTION")
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

        dumpLedger("no_known_progress");
        ensureNavTopology();
        std::vector<char> reachable;
        llmapper::markReachableNavCells(navCells, currentNavStartCell(),
                                        navEdgeFailures, 0, reachable);
        const std::vector<llmapper::VisibilityFrontier> visibility =
            semanticVisibilityFrontiers(reachable);
        setGoal("NO_APPLICABLE_ACTION", observation.sector);
        if (!explorationSnapshotEmitted)
        {
            char detail[224];
            snprintf(detail, sizeof(detail),
                     "reason=NO_APPLICABLE_ACTION sector=%d visibility_frontiers=%u blocked=%u interactions=%u objects=%u",
                     observation.sector, unsigned(visibility.size()),
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
            std::vector<char> reachable;
            llmapper::markReachableNavCells(navCells, currentNavStartCell(),
                                            navEdgeFailures, 0, reachable);
            const std::vector<llmapper::VisibilityFrontier> visibility =
                semanticVisibilityFrontiers(reachable);
            int informationGain = 0;
            for (const llmapper::VisibilityFrontier &frontier : visibility)
                informationGain += frontier.informationGain;
            char detail[256];
            snprintf(detail, sizeof(detail),
                     "reason=NO_APPLICABLE_ACTION sector=%d cells=%u objective=%d visibility_frontiers=%u expected_information_gain=%d interactions=%u objects=%u",
                     observation.sector, unsigned(navCells.size()),
                     int(currentObjective.type), unsigned(visibility.size()),
                     informationGain,
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

        if (currentGoal != "WAIT_MOVING_MECHANISM" && currentGoal != "NO_APPLICABLE_ACTION")
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
                invalidateObjective("loop_break");
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
        if (navMeshDump)
        {
            fclose(navMeshDump);
            navMeshDump = nullptr;
        }
        if (gDemo.at0)
            gDemo.Close();
    }
};

LLMapperBot::LLMapperBot()
    : m_impl(new Impl), m_enabled(false), m_fast(true), m_visible(false),
      m_debugOverlay(false)
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
    if ((!m_enabled && !m_debugOverlay) || !gGameStarted || !gMe || !gMe->pXSprite)
        return;
    m_impl->updateKnowledge();
    m_impl->lastObservationTick = gFrame * kTicsPerFrame;
    if (m_debugOverlay)
    {
        // Manual observer mode owns no controls and no mission lifecycle, but
        // it continuously maintains the exact same physical topology and
        // opportunity ledger the autonomous policy would consume.
        m_impl->ensureNavTopology();
        if (m_impl->debugLedgerTick < 0
            || m_impl->observation.tick - m_impl->debugLedgerTick
                >= kTicsPerSec / 4)
        {
            m_impl->rebuildLedger();
            m_impl->debugLedgerTick = m_impl->observation.tick;
        }
    }
    if (gMe->pSprite->extra > 0 && gMe->pSprite->extra < kMaxXSprites)
    {
        const int moveHit = gSpriteHit[gMe->pSprite->extra].hit;
        const int wallHit = (moveHit & 0xc000) == 0x8000 ? moveHit & 0x3fff : -1;
        const int spriteHit = (moveHit & 0xc000) == 0xc000 ? moveHit & 0x3fff : -1;
        const int ownedHit = wallHit >= 0 || spriteHit >= 0 ? moveHit : -1;
        if (ownedHit != m_impl->lastEngineMoveHit)
        {
            m_impl->lastEngineMoveHit = ownedHit;
            if (wallHit >= 0)
            {
                char detail[160];
                snprintf(detail, sizeof(detail), "wall=%d sector=%d x=%d y=%d hit=0x%x goal=%s target=%d",
                         wallHit, m_impl->observation.sector, int(gMe->pSprite->x), int(gMe->pSprite->y),
                         moveHit, m_impl->currentGoal.c_str(), m_impl->lastStateTarget);
                m_impl->event("engine_move_wall_hit", detail);
            }
            else if (spriteHit >= 0)
            {
                char detail[192];
                snprintf(detail, sizeof(detail),
                         "sprite=%d sector=%d x=%d y=%d hit=0x%x goal=%s target=%d",
                         spriteHit, m_impl->observation.sector,
                         int(gMe->pSprite->x), int(gMe->pSprite->y), moveHit,
                         m_impl->currentGoal.c_str(), m_impl->lastStateTarget);
                m_impl->event("engine_move_sprite_hit", detail);
            }
        }
    }
    if (m_enabled && (gGameOptions.uGameFlags & kGameFlagContinuing)
        && m_impl->result.empty())
        OnLevelExit(kLevelExitNormal);
    if (m_impl->lastTrajectoryTick < gFrame * kTicsPerFrame - kTrajectoryPeriod)
    {
        m_impl->trajectorySample();
        m_impl->lastTrajectoryTick = gFrame * kTicsPerFrame;
    }
    if (m_enabled
        && m_impl->observation.tick % (kObservationPeriod * kTicsPerFrame) == 0)
        m_impl->detectStall();
    if (m_enabled
        && m_impl->observation.tick >= m_impl->timeoutSeconds * kTicsPerSec)
    {
        m_impl->result = "TIMEOUT";
        m_impl->failureReason = "simulated time limit reached";
        m_impl->event("failure", m_impl->failureReason.c_str());
        gQuitGame = true;
    }
    if (m_enabled && m_impl->observation.health == 0 && m_impl->result.empty())
    {
        m_impl->result = "DIED";
        m_impl->failureReason = "player health reached zero";
        m_impl->event("failure", m_impl->failureReason.c_str());
        gQuitGame = true;
    }
}

void LLMapperBot::OnActionResolved(int hit, int target, int extra, bool accepted, int key)
{
    if (m_enabled || m_debugOverlay)
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

void LLMapperBot::ConfigureNavMeshDump(const char *path)
{
    if (path && *path)
        m_impl->navMeshDumpPath = path;
}

bool LLMapperBot::ConfigureDebugOverlay(const char *setting)
{
    if (!setting || !*setting || !strcmp(setting, "toggle"))
    {
        m_debugOverlay = !m_debugOverlay;
        return true;
    }

    if (!strcmp(setting, "on"))
    {
        m_debugOverlay = true;
        return true;
    }
    if (!strcmp(setting, "off"))
    {
        m_debugOverlay = false;
        return true;
    }
    if (!strcmp(setting, "all"))
    {
        m_impl->debugLayers = kDebugAll;
        m_debugOverlay = true;
        return true;
    }
    if (!strcmp(setting, "default"))
    {
        m_impl->debugLayers = kDefaultDebugLayers;
        m_debugOverlay = true;
        return true;
    }
    struct NamedLayer { const char *name; uint32_t bit; };
    static const NamedLayer layers[] = {
        { "mesh", kDebugMesh }, { "poses", kDebugPoses },
        { "walk", kDebugWalk },
        { "jump", kDebugJump }, { "crouch", kDebugCrouch },
        { "ride", kDebugRide }, { "observation", kDebugObservation },
        { "interactions", kDebugInteractions }, { "tasks", kDebugTasks },
        { "route", kDebugRoute }, { "hypothetical", kDebugHypothetical },
    };
    for (const NamedLayer &layer : layers)
    {
        if (strcmp(setting, layer.name))
            continue;
        m_impl->debugLayers ^= layer.bit;
        m_debugOverlay = true;
        return true;
    }
    return false;
}

void LLMapperBot::DescribeDebugOverlay(char *buffer, int size) const
{
    if (!buffer || size <= 0)
        return;
    if (!m_debugOverlay)
    {
        snprintf(buffer, size, "off");
        return;
    }
    int used = snprintf(buffer, size, "on:");
    struct NamedLayer { const char *name; uint32_t bit; };
    static const NamedLayer layers[] = {
        { "mesh", kDebugMesh }, { "poses", kDebugPoses },
        { "walk", kDebugWalk },
        { "jump", kDebugJump }, { "crouch", kDebugCrouch },
        { "ride", kDebugRide }, { "observation", kDebugObservation },
        { "interactions", kDebugInteractions }, { "tasks", kDebugTasks },
        { "route", kDebugRoute }, { "hypothetical", kDebugHypothetical },
    };
    for (const NamedLayer &layer : layers)
        if ((m_impl->debugLayers & layer.bit) && used < size)
            used += snprintf(buffer + used, size_t(size - used), " %s", layer.name);
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
    const char *reason = bot.selectedWorkReason.empty()
        ? "-" : bot.selectedWorkReason.c_str();
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
    for (size_t i = 0; i < bot.ledger.size(); ++i)
    {
        if (bot.ledger[i].hops < 0)
            continue;
        ++pending;
    }
    snprintf(line, sizeof(line), "todo %d  keys %u  hp %d",
             pending, unsigned(bot.heldKeys.size()),
             bot.observation.health / 16);
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

void LLMapperBot::DrawDebugOverlay(int cameraX, int cameraY, int cameraZ,
                                   fix16_t cameraAngle, fix16_t cameraHoriz)
{
    if (!m_debugOverlay || !m_impl || !gGameStarted || !gMe || !gMe->pSprite
        || gViewMode != 3)
        return;

    Impl &bot = *m_impl;
    if (bot.navCells.empty())
        return;
    const uint32_t layers = bot.debugLayers;
    const int viewWidth = std::max(1, gViewX1 - gViewX0 + 1);
    const int viewHeight = std::max(1, gViewY1 - gViewY0 + 1);
    const double centerX = gViewX0 + viewWidth * 0.5;
    const double centerY = gViewY0
        + fix16_to_float(cameraHoriz) * viewHeight / 200.0;
    const int angle = fix16_to_int(cameraAngle) & kAngMask;
    const double cosine = double(Cos(angle)) / double(1 << 30);
    const double sine = double(Sin(angle)) / double(1 << 30);
    const double horizontalScale = viewWidth * 0.5 * 65536.0
        / std::max(1, viewingrange);
    const int colorWhite = paletteGetClosestColor(255, 255, 255);
    const int colorGreen = paletteGetClosestColor(32, 255, 96);
    const int colorCyan = paletteGetClosestColor(32, 224, 255);
    const int colorBlue = paletteGetClosestColor(64, 128, 255);
    const int colorYellow = paletteGetClosestColor(255, 224, 32);
    const int colorOrange = paletteGetClosestColor(255, 128, 24);
    const int colorMagenta = paletteGetClosestColor(255, 48, 224);
    const int colorRed = paletteGetClosestColor(255, 48, 48);
    const int colorGray = paletteGetClosestColor(160, 160, 160);

    struct ScreenPoint
    {
        int x = 0;
        int y = 0;
        double depth = 0;
        bool visible = false;
    };
    auto project = [&](int x, int y, int z) {
        ScreenPoint point;
        const double dx = double(x - cameraX);
        const double dy = double(y - cameraY);
        const double forward = dx * cosine + dy * sine;
        if (forward <= 32.0)
            return point;
        const double side = dy * cosine - dx * sine;
        point.x = int(std::lround(centerX + side * horizontalScale / forward));
        point.y = int(std::lround(centerY
            + double(z - cameraZ) * horizontalScale / (forward * 16.0)));
        point.depth = forward;
        point.visible = point.x >= gViewX0 - 32 && point.x <= gViewX1 + 32
            && point.y >= gViewY0 - 32 && point.y <= gViewY1 + 32;
        return point;
    };
    auto local = [&](int x, int y) {
        return distance2(cameraX, cameraY, x, y)
            <= kDebugOverlayRadius * kDebugOverlayRadius;
    };
    auto line = [&](int x1, int y1, int z1, int x2, int y2, int z2, int color) {
        const ScreenPoint a = project(x1, y1, z1);
        const ScreenPoint b = project(x2, y2, z2);
        if ((!a.visible && !b.visible) || a.depth <= 0 || b.depth <= 0)
            return;
        renderDrawLine(a.x << 12, a.y << 12, b.x << 12, b.y << 12,
                       char(color));
    };
    auto marker = [&](int x, int y, int z, int radius, int color) {
        line(x - radius, y, z, x + radius, y, z, color);
        line(x, y - radius, z, x, y + radius, z, color);
    };
    // viewDrawText() uses RS_AUTO and therefore consumes Blood's historical
    // aspect-corrected 320x200 coordinate system. World projection above is
    // deliberately in framebuffer pixels so polygons line up with Polymost.
    // Convert only the position; retaining the Blood font is intentional.
    const int aspectWidth = std::min(scale(ydim, 4, 3), xdim);
    auto drawWorldText = [&](const char *text, const ScreenPoint &point) {
        const int textX = scale(point.x, 320, std::max(1, aspectWidth));
        const int textY = scale(point.y, 200, std::max(1, ydim));
        viewDrawText(3, text, textX + 3, textY - 3,
                     -128, 0, 0, true, 256);
    };

    struct DebugPolygon
    {
        int x[4];
        int y[4];
        double depth;
        uint8_t red;
        uint8_t green;
        uint8_t blue;
        uint8_t alpha;
        int outline;
    };
    auto projectQuad = [&](const int worldX[4], const int worldY[4],
                           const int worldZ[4], uint8_t red, uint8_t green,
                           uint8_t blue, uint8_t alpha, int outline,
                           DebugPolygon &polygon) {
        bool anyVisible = false;
        polygon.depth = 0.0;
        for (int i = 0; i < 4; ++i)
        {
            const ScreenPoint point = project(worldX[i], worldY[i], worldZ[i]);
            // Tiles intersecting the camera plane turn into enormous screen
            // quadrilaterals. The player cannot usefully inspect a support
            // tile underneath their own collision cylinder, so cull it until
            // every corner is a stable distance in front of the camera.
            if (point.depth <= 512.0)
                return false;
            polygon.x[i] = point.x;
            polygon.y[i] = point.y;
            polygon.depth += point.depth;
            anyVisible = anyVisible || point.visible;
        }
        polygon.depth *= 0.25;
        polygon.red = red;
        polygon.green = green;
        polygon.blue = blue;
        polygon.alpha = alpha;
        polygon.outline = outline;
        return anyVisible;
    };
    auto drawPolygon = [&](const DebugPolygon &polygon) {
        renderDrawPolygonRGBA(polygon.x, polygon.y, 4, polygon.red,
                              polygon.green, polygon.blue, polygon.alpha);
        if (polygon.outline < 0)
            return;
        for (int i = 0; i < 4; ++i)
        {
            const int next = (i + 1) & 3;
            renderDrawLine(polygon.x[i] << 12, polygon.y[i] << 12,
                           polygon.x[next] << 12, polygon.y[next] << 12,
                           char(polygon.outline));
        }
    };

    // A nav cell is an authoritative collision-derived standable sample, so
    // draw it as a real height-correct tile rather than a point cloud. Small
    // gaps make separate samples/layers readable. The polygons are sorted
    // back-to-front because this is intentionally an x-ray overlay.
    if (layers & (kDebugMesh | kDebugHypothetical))
    {
        std::vector<DebugPolygon> surfaces;
        surfaces.reserve(bot.navCells.size());
        constexpr int halfCell = (1 << (kNavGridShift - 1)) - 12;
        for (const NavCell &cell : bot.navCells)
        {
            if (!local(cell.center.x, cell.center.y))
                continue;
            if ((cell.live && !(layers & kDebugMesh))
                || (!cell.live && !(layers & kDebugHypothetical)))
                continue;
            const int x[4] = {
                cell.center.x - halfCell, cell.center.x + halfCell,
                cell.center.x + halfCell, cell.center.x - halfCell,
            };
            const int y[4] = {
                cell.center.y - halfCell, cell.center.y - halfCell,
                cell.center.y + halfCell, cell.center.y + halfCell,
            };
            const int surfaceZ = cell.z - 48;
            const int z[4] = { surfaceZ, surfaceZ, surfaceZ, surfaceZ };
            const bool observed = bot.observedCells.count(
                bot.physicalPoseKey(cell)) != 0;
            uint8_t red = observed ? 32 : 255;
            uint8_t green = observed ? 224 : 176;
            uint8_t blue = observed ? 96 : 24;
            int outline = observed ? colorGreen : colorYellow;
            if (cell.support.kind == kSupportSpriteFloor)
            {
                red = observed ? 224 : 255;
                green = observed ? 48 : 112;
                blue = observed ? 255 : 24;
                outline = observed ? colorMagenta : colorOrange;
            }
            if (!cell.live)
            {
                red = 96;
                green = 128;
                blue = 192;
                outline = colorBlue;
            }
            DebugPolygon polygon;
            if (projectQuad(x, y, z, red, green, blue,
                            cell.live ? 54 : 30, outline, polygon))
                surfaces.push_back(polygon);
        }
        std::sort(surfaces.begin(), surfaces.end(),
                  [](const DebugPolygon &a, const DebugPolygon &b) {
                      return a.depth > b.depth;
                  });
        for (const DebugPolygon &surface : surfaces)
            drawPolygon(surface);
    }

    // Poses are the collision-derived surface samples the planner actually
    // searches.  Observation already changes the mesh fill/outline above;
    // do not implicitly add two more line segments for every cell merely
    // because that layer is enabled.  Raw pose crosses are deliberately an
    // opt-in diagnostic for small local geometry.
    if (layers & kDebugPoses)
    {
        for (const NavCell &cell : bot.navCells)
        {
            if (!local(cell.center.x, cell.center.y))
                continue;
            const bool observed = bot.observedCells.count(bot.physicalPoseKey(cell)) != 0;
            int color = cell.support.kind == kSupportSpriteFloor
                ? colorMagenta : colorCyan;
            if (layers & kDebugObservation)
                color = observed ? colorGreen : colorYellow;
            marker(cell.center.x, cell.center.y, cell.z - 160,
                   cell.support.kind == kSupportSpriteFloor ? 72 : 38, color);
        }
    }

    auto drawNavEdge = [&](const NavCell &cell, const NavCell &target,
                           NavEdgeMode mode, int color) {
        const int sourceZ = cell.z - 256;
        const int targetZ = target.z - 256;
        if (mode == kNavJump)
        {
            const int midX = (cell.center.x + target.center.x) / 2;
            const int midY = (cell.center.y + target.center.y) / 2;
            const int midZ = std::min(sourceZ, targetZ) - 1024;
            line(cell.center.x, cell.center.y, sourceZ,
                 midX, midY, midZ, color);
            line(midX, midY, midZ,
                 target.center.x, target.center.y, targetZ, color);
        }
        else
            line(cell.center.x, cell.center.y, sourceZ,
                 target.center.x, target.center.y, targetZ, color);
    };

    // The planner owns concrete cell-to-cell links, but drawing every member
    // of a many-to-many support transition makes the graph unreadable.  Keep
    // raw walk adjacency available as an explicit layer.  For jump, crouch
    // and ride, display the shortest representative for each physical
    // support/connected-area pair in each coarse crossing neighbourhood.
    // This changes presentation only; route search still sees every edge.
    struct DebugTransition
    {
        int from = -1;
        int to = -1;
        NavEdgeMode mode = kNavBlocked;
        int color = 0;
        int64_t span = INT64_MAX;
    };
    using DebugTransitionKey = std::tuple<int, int, int, int, int, int, int,
                                          int, int, int, int>;
    std::map<DebugTransitionKey, DebugTransition> transitions;
    size_t enabledEdgeCount = 0;
    size_t displayedEdgeCount = 0;
    auto crossingBucket = [](int coordinate) {
        constexpr int bucketSize = 1024;
        return coordinate >= 0 ? coordinate / bucketSize
            : -int((int64_t(-coordinate) + bucketSize - 1) / bucketSize);
    };
    for (const NavCell &cell : bot.navCells)
    {
        if (!local(cell.center.x, cell.center.y))
            continue;
        for (const NavLink &edge : cell.links)
        {
            if (!inRange(edge.target, 0, int(bot.navCells.size())))
                continue;
            const NavCell &target = bot.navCells[size_t(edge.target)];
            uint32_t layer = 0;
            int color = colorWhite;
            switch (edge.mode)
            {
            case kNavWalk:
            case kNavStep:
            case kNavDrop:
                layer = kDebugWalk;
                color = edge.mode == kNavDrop ? colorOrange : colorGreen;
                break;
            case kNavJump: layer = kDebugJump; color = colorMagenta; break;
            case kNavCrouch: layer = kDebugCrouch; color = colorYellow; break;
            case llmapper::kNavRide: layer = kDebugRide; color = colorBlue; break;
            default: continue;
            }
            if (!(layers & layer))
                continue;
            ++enabledEdgeCount;
            if ((edge.mode == kNavWalk || edge.mode == kNavStep)
                && edge.target < cell.id)
                continue;

            if (layer == kDebugWalk)
            {
                drawNavEdge(cell, target, edge.mode, color);
                ++displayedEdgeCount;
                continue;
            }

            const int midX = (cell.center.x + target.center.x) / 2;
            const int midY = (cell.center.y + target.center.y) / 2;
            const DebugTransitionKey key(
                int(edge.mode), cell.walkArea, int(cell.support.kind),
                cell.support.id, target.walkArea, int(target.support.kind),
                target.support.id, crossingBucket(midX), crossingBucket(midY),
                edge.wall, edge.transition);
            const int64_t span = distance2(cell.center.x, cell.center.y,
                                           target.center.x, target.center.y)
                + int64_t(std::abs(target.z - cell.z))
                    * std::abs(target.z - cell.z) / 256;
            DebugTransition &representative = transitions[key];
            if (representative.from < 0 || span < representative.span)
            {
                representative.from = cell.id;
                representative.to = target.id;
                representative.mode = edge.mode;
                representative.color = color;
                representative.span = span;
            }
        }
    }
    for (const auto &entry : transitions)
    {
        const DebugTransition &edge = entry.second;
        if (!inRange(edge.from, 0, int(bot.navCells.size()))
            || !inRange(edge.to, 0, int(bot.navCells.size())))
            continue;
        drawNavEdge(bot.navCells[size_t(edge.from)],
                    bot.navCells[size_t(edge.to)], edge.mode, edge.color);
        ++displayedEdgeCount;
    }

    if ((layers & kDebugRoute) && !bot.navRoute.empty())
    {
        for (size_t i = bot.navRouteIndex; i < bot.navRoute.size(); ++i)
        {
            const NavRouteStep &step = bot.navRoute[i];
            if (!inRange(step.fromCell, 0, int(bot.navCells.size()))
                || !inRange(step.toCell, 0, int(bot.navCells.size())))
                continue;
            const NavCell &from = bot.navCells[size_t(step.fromCell)];
            const NavCell &to = bot.navCells[size_t(step.toCell)];
            const double dx = double(to.center.x - from.center.x);
            const double dy = double(to.center.y - from.center.y);
            const double length = std::sqrt(dx * dx + dy * dy);
            if (length > 1.0)
            {
                constexpr double halfWidth = 44.0;
                const int nx = int(std::lround(-dy * halfWidth / length));
                const int ny = int(std::lround(dx * halfWidth / length));
                const int x[4] = {
                    from.center.x + nx, to.center.x + nx,
                    to.center.x - nx, from.center.x - nx,
                };
                const int y[4] = {
                    from.center.y + ny, to.center.y + ny,
                    to.center.y - ny, from.center.y - ny,
                };
                const int z[4] = {
                    from.z - 448, to.z - 448,
                    to.z - 448, from.z - 448,
                };
                DebugPolygon ribbon;
                if (projectQuad(x, y, z, 255, 255, 255, 184,
                                colorWhite, ribbon))
                    drawPolygon(ribbon);
            }
            line(from.center.x, from.center.y, from.z - 384,
                 to.center.x, to.center.y, to.z - 384, colorWhite);
        }
    }

    int labels = 0;
    if (layers & (kDebugInteractions | kDebugTasks))
    {
        const unsigned effects = bot.availableEffectCapabilities();
        for (auto &entry : bot.interactions)
        {
            Impl::InteractionMemory &memory = entry.second;
            if (!memory.observed || !local(memory.x, memory.y))
                continue;
            bool currentlyVisible = false;
            for (const InteractionCandidate &seen : bot.observation.interactions)
                if (seen.kind == memory.kind && seen.id == memory.id)
                    currentlyVisible = true;
            if (!(layers & kDebugTasks) && !currentlyVisible)
                continue;
            const llmapper::WorkId opportunityId = bot.interactionWorkId(
                bot.interactionMemoryKey(memory));
            const llmapper::Opportunity *work = bot.ledgerEntry(opportunityId);
            const bool readyHere = currentlyVisible
                && distance2(bot.observation.x, bot.observation.y,
                             memory.x, memory.y)
                    <= (kActionScanRange + 256) * (kActionScanRange + 256)
                && bot.previewInteractionFromCurrentOrigin(memory);
            const char *state = "APPROACH";
            int color = currentlyVisible ? colorCyan : colorGray;
            if (readyHere)
            {
                state = "READY";
                color = colorGreen;
            }
            else if (memory.key && !bot.hasKey(memory.key))
            {
                state = "NEEDS KEY";
                color = colorYellow;
            }
            else if (!llmapper::effectRequirementSatisfied(
                         memory.requiredEffects, effects))
            {
                state = "NEEDS EFFECT";
                color = colorYellow;
            }
            else if (work && work->hops < 0)
            {
                state = "NO ROUTE";
                color = colorRed;
            }
            else if (memory.activated && !bot.interactionNeedsReactivation(memory))
            {
                state = "DONE";
                color = colorGreen;
            }
            marker(memory.x, memory.y, memory.z, 144, color);
            line(memory.x, memory.y, memory.z - 512,
                 memory.x, memory.y, memory.z + 512, color);
            const ScreenPoint point = project(memory.x, memory.y, memory.z - 640);
            if (point.visible && labels < 18)
            {
                char text[96];
                snprintf(text, sizeof(text), "I %d:%d %s%s",
                         int(memory.kind), memory.id, state,
                         currentlyVisible ? " SEEN" : "");
                drawWorldText(text, point);
                ++labels;
            }
        }
    }

    if (layers & kDebugTasks)
    {
        for (const llmapper::Opportunity &work : bot.ledger)
        {
            if (work.kind == llmapper::kOpportunityInteraction
                || work.kind == llmapper::kOpportunityExit
                || work.kind == llmapper::kOpportunityPickup
                || !inRange(work.approach, 0, int(bot.navCells.size())))
                continue;
            const NavCell &pose = bot.navCells[size_t(work.approach)];
            if (!local(pose.center.x, pose.center.y))
                continue;
            const bool ready = work.hops >= 0
                && (!work.requiredKey || bot.hasKey(work.requiredKey))
                && llmapper::effectRequirementSatisfied(
                    work.requiredEffects, bot.availableEffectCapabilities());
            marker(pose.center.x, pose.center.y, pose.z - 640,
                   112, ready ? colorGreen : colorOrange);
            const ScreenPoint point = project(pose.center.x, pose.center.y,
                                              pose.z - 768);
            if (point.visible && labels < 18)
            {
                char text[96];
                snprintf(text, sizeof(text), "T %d:%d %s",
                         int(work.kind), work.id.subject,
                         ready ? "READY" : "DEFERRED");
                drawWorldText(text, point);
                ++labels;
            }
        }
    }

    if (layers & kDebugInteractions)
    {
        for (const VisibleObject &object : bot.observation.objects)
        {
            if (!local(object.x, object.y))
                continue;
            const bool pickup = object.kind == kObjectPickup
                || object.kind == kObjectKey;
            const bool useful = !pickup || bot.pickupUsefulNow(object);
            marker(object.x, object.y, object.z, 112,
                   useful ? colorCyan : colorGray);
            const ScreenPoint point = project(object.x, object.y, object.z - 512);
            if (point.visible && labels < 18)
            {
                char text[64];
                snprintf(text, sizeof(text), "O %d:%d %s",
                         int(object.kind), object.sprite,
                         useful ? "SEEN" : "FULL");
                drawWorldText(text, point);
                ++labels;
            }
        }
    }

    char status[192];
    const SupportRef standing = currentPlayerSupport();
    snprintf(status, sizeof(status),
             "BOTDBG topo %d%s cells %u edges %u/%u support %d:%d z%d todo %u",
             bot.navTopologyRevision, bot.navMeshInMotion ? " MOVING" : "",
             unsigned(bot.navCells.size()), unsigned(displayedEdgeCount),
             unsigned(enabledEdgeCount), int(standing.kind), standing.id,
             llmapper::capability::playerFloorZ(), unsigned(bot.ledger.size()));
    viewDrawText(3, status, 2, m_enabled && m_visible ? 44 : 4,
                 -128, 0, 0, true, 256);
}

void LLMapperBot::Finish(const char *reason)
{
    if (m_enabled && m_impl->telemetry)
        m_impl->close(reason);
}

LLMapperBot gLLMapperBot;
