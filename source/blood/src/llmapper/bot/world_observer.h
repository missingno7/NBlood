//-------------------------------------------------------------------------
// Blood observation and physics queries for the LLMapper bot.
//
// This layer is the only place that reads Blood's world arrays and asks the
// engine what the player can occupy, reach, stand on and see.  Everything
// above it consumes those answers; nothing above it reaches back down.  The
// dependency is one-way by construction -- this header knows nothing about
// the bot, its goals, or its plans.
//-------------------------------------------------------------------------
#pragma once

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
#include "../../warp.h"

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
using SupportRef = llmapper::SupportId;
using WorldObjectRef = llmapper::ObjectId;
using llmapper::NavRouteStep;
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

// Blood identity ends here. Semantic IDs are allocated from adapter-owned
// registries, never by integer offsets or sentinel namespaces. The kernel can
// compare these tokens but cannot recover or interpret their engine origin.
enum EngineHandleKind
{
    kEngineSectorFloor,
    kEngineSpriteFloor,
    kEngineSector,
    kEngineWall,
    kEngineSprite,
};

static constexpr EngineHandleKind kSupportSectorFloor = kEngineSectorFloor;
static constexpr EngineHandleKind kSupportSpriteFloor = kEngineSpriteFloor;
static constexpr EngineHandleKind kWorldSector = kEngineSector;
static constexpr EngineHandleKind kWorldWall = kEngineWall;
static constexpr EngineHandleKind kWorldSprite = kEngineSprite;

struct EngineHandle
{
    EngineHandleKind kind;
    int index;
    bool operator<(const EngineHandle &other) const
    {
        return kind < other.kind || (kind == other.kind && index < other.index);
    }
};

static std::map<EngineHandle, SupportRef> gSupportIds;
static std::map<SupportRef, EngineHandle> gSupportHandles;
static std::map<EngineHandle, WorldObjectRef> gObjectIds;
static std::map<WorldObjectRef, EngineHandle> gObjectHandles;
struct EngineAffordanceHandle
{
    EngineHandle target;
    llmapper::ActionKind action;
    bool operator<(const EngineAffordanceHandle &other) const
    {
        return target < other.target
            || (!(other.target < target) && action < other.action);
    }
};
static std::map<EngineAffordanceHandle, llmapper::AffordanceId> gAffordanceIds;
extern int gNextSupportId;
extern int gNextObjectId;
extern int gNextAffordanceId;
namespace llmapper
{
using ActivationMode = ActionKind;
static constexpr ActionKind kActivateUse = kActionUse;
static constexpr ActionKind kActivateVector = kActionDeliverRemoteEffect;
static constexpr ActionKind kActivateImpact = kActionCollide;
static constexpr ActionKind kActivateTouch = kActionTouch;
static constexpr ActionKind kActivateEnter = kActionEnter;
static constexpr ActionKind kActivateExit = kActionExit;
static constexpr ActionKind kActivatePickup = kActionPickup;
static constexpr ActionKind kActivateSight = kActionObserve;
static constexpr ActionKind kActivateProximity = kActionApproach;
static constexpr ActionKind kActivateDamage = kActionDeliverDamage;
}

// Persistent observation/task identity for one concrete standable pose.
// Position and support layer are both part of the identity: sprite tops, ROR
// layers and moving-support endpoints can overlap in XY while remaining
// different physical places.
using PhysicalPoseKey = std::tuple<int, int, int, int, int, int>;

// These types and constants are the vocabulary of the engine-facing
// layer.  They are published, not file-local: two translation units
// that disagree about what an EngineBoundaryObservation is cannot
// pass one to each other.
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
constexpr int kJumpTakeoffRange = 1536;
constexpr int kJumpActionTimeoutTicks = 2 * kTicsPerSec;
constexpr int kActionScanRange = 1024;
constexpr int kActionApproachRange = 2048;
constexpr int kUseStopRange = kActionApproachRange;
constexpr int kInteractionTimeoutTicks = 6 * kTicsPerSec;
// Added to the engine's real damaging radius.  This is clearance for Caleb's
// body plus a small uncertainty allowance, not a guessed explosion size.
constexpr int kExplosiveSafetyMargin = 256;
constexpr int kExplosiveThrowAllowance = 1536;
constexpr int kExplosiveOutcomeTicks = 6 * kTicsPerSec;
// Every committed objective is bounded.  An objective that stops making
// progress, or simply runs too long, releases the bot back to selection and
// records a deterministic failure under the current relevant evidence.
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
// How close a hostile must be before closing to melee is worth interrupting
// exploration for.  Chasing a dude across the level is not combat, it is a
// distraction that costs the whole run.
constexpr int kMeleeEngageRange = 2048;
// How far the bot credits itself with having seen local space, and how far
// above the floor the sight test is taken (roughly knee height, so a low
// switch on a far wall still counts as observable).
constexpr int kCoverageSightRange = 6144;
// Long enough for Blood to switch the player to the crouch sequence and for
// one observation to sample the resulting body extents.
constexpr int kCrouchCalibrationTicks = kTicsPerSec;
// Negative synthetic edge identity reserved for Blood's explicit stacked
// room/link transition.  Ordinary intra-sector links use -1 and wall portal
// links use their non-negative engine wall id.
constexpr int kNavRorTransitionEdge = -2;
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

struct EngineBoundaryObservation
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
    int openingWidth = 0;
    int floorDelta = 0;
    int clearance = 0;
    bool wallPush = false;
    bool shootable = false;
    bool sectorPush = false;
    bool sectorPushCurrent = false;
    bool visible = false;
    bool localGeometry = false;
    bool walkable = false;
    bool jumpable = false;
    bool crouchable = false;
    bool dropSafe = false;
    bool traversable = false;
    bool blockedBySprite = false;
    int blockerSprite = -1;
    TraversalCapability capability = kTraversalUnknown;
    int wallState = -1;
    int wallBusy = 0;
    int sectorState = -1;
    int sectorBusy = 0;
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
    // Blood RX/TX provenance used only as a dependency oracle while the
    // semantic affordance is constructed. It is not boundary geometry and
    // does not by itself assert a learned effect.
    int dependencyChannel = 0;
    bool reversible = false;
    llmapper::ActivationMode activationMode = llmapper::kActivateUse;
    unsigned requiredEffects = llmapper::kEffectNone;
    int targetTopZ = 0;
    int targetBottomZ = 0;
    EngineBoundaryObservation target;
};

// Exact Blood signal routing is adapter data. It is resolved here into
// opaque objects before any causal fact is exposed to semantic planning.
struct BloodSignalReceiver
{
    int channel = 0;
    WorldObjectRef object;
    int stateVariable = -1;
    int outgoingChannel = 0;
    int command = 0;
};

struct BloodSignalGraph
{
    std::vector<BloodSignalReceiver> receivers;

    std::vector<BloodSignalReceiver> receiversFor(int channel) const
    {
        std::vector<BloodSignalReceiver> result;
        for (const BloodSignalReceiver &receiver : receivers)
            if (receiver.channel == channel)
                result.push_back(receiver);
        return result;
    }

    std::vector<BloodSignalReceiver> receiversReachableFrom(
        int channel, int maxDepth, bool terminalOnly) const
    {
        std::vector<BloodSignalReceiver> result;
        if (channel <= 0 || maxDepth < 0)
            return result;
        std::deque<std::pair<int, int>> pending;
        std::set<int> visitedChannels;
        std::set<WorldObjectRef> emittedObjects;
        pending.push_back(std::make_pair(channel, 0));
        visitedChannels.insert(channel);
        while (!pending.empty())
        {
            const std::pair<int, int> current = pending.front();
            pending.pop_front();
            for (const BloodSignalReceiver &receiver : receiversFor(current.first))
            {
                const bool hasOutgoing = receiver.outgoingChannel > 0;
                const bool canFollow = hasOutgoing && current.second < maxDepth
                    && !visitedChannels.count(receiver.outgoingChannel);
                if (canFollow)
                {
                    visitedChannels.insert(receiver.outgoingChannel);
                    pending.push_back(std::make_pair(receiver.outgoingChannel,
                                                     current.second + 1));
                }
                if (terminalOnly && hasOutgoing)
                    continue;
                if (emittedObjects.insert(receiver.object).second)
                    result.push_back(receiver);
            }
        }
        return result;
    }
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
    std::vector<EngineBoundaryObservation> portals;
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












// Smallest body envelope actually observed while crouched.  Blood swaps the
// player to the crouch animation sequence, and GetSpriteExtents follows that
// sequence, so the crouched collision body really is shorter than the
// standing one -- but only measurable once the bot has crouched at least
// once.  Until then, fall back to the engine's own posture table.
extern int gObservedCrouchClearance;
extern int gStandingClearance;
struct PlayerCollisionShape
{
    bool known = false;
    int radius = 0;
    int ceilingDistance = 0;
    int floorDistance = 0;
    int footOffset = 0;
};

// Collision envelopes produced by Blood's moving-player SEQ. playerProcess
// selects this sequence from physical input, seqProcess advances it before
// MoveDude, and MoveDude then derives both ClipMove distances and floor
// contact from the resulting sprite extents. Air-trajectory replay consumes
// this engine-owned cycle instead of freezing the planning-frame sprite.
struct PlayerCollisionCycle
{
    std::vector<PlayerCollisionShape> frames;
    int ticksPerFrame = 0;
    bool looping = false;
};

struct PlayerGroundObservation
{
    int floorZ = 0;
    SupportRef support;
    llmapper::capability::GroundContactMotion motion;
};

// One translation between two coordinate frames which Blood treats as a
// continuous physical space.  The regions in each frame are connected by
// ordinary Build boundaries; crossing between the frames remains subject to
// CheckLink and the player's real motion.  Consumers never inspect link
// marker sprites or reconstruct their ownership.
struct EngineSpaceTranslation
{
    int upperRegion = -1;
    int lowerRegion = -1;
    int upperToLowerX = 0;
    int upperToLowerY = 0;
    int upperToLowerZ = 0;
    int upperAnchorX = 0;
    int upperAnchorY = 0;
    int lowerAnchorX = 0;
    int lowerAnchorY = 0;
    std::vector<int> upperFrameRegions;
    std::vector<int> lowerFrameRegions;
};

// A concrete solution for Blood's held-and-released TNT throw. Power is the
// exact 16.16 value consumed by ThrowBundle; holdTicks is the corresponding
// gFrameClock duration used by processTNT. This is physical execution data,
// not persistent affordance knowledge.
struct TntThrowSolution
{
    bool valid = false;
    int angle = 0;
    int look = 0;
    int power = 0;
    int holdTicks = 0;
    int closestDistance = INT32_MAX;
    int closestFrame = -1;
    int restX = 0;
    int restY = 0;
    int restZ = 0;
    int restSector = -1;
};

extern PlayerCollisionShape gObservedCrouchShape;
struct StandableSurface
{
    SupportRef support;
    int z = 0;
    int ceilingZ = 0;
};




























//-------------------------------------------------------------------------
// Questions answered by the engine.  Each asks about the physical world,
// never about what the actor wants.
//-------------------------------------------------------------------------

SupportRef engineSupport(EngineHandleKind kind, int index);
WorldObjectRef engineObject(EngineHandleKind kind, int index);
int interactionOwnerSector(int observedTargetSector,
                                  int immediateReceiverSector);
llmapper::AffordanceId physicalAffordanceId(
    int wallId, int fromSector, int targetSector, bool wallPush,
    bool sectorPush, bool sectorPushCurrent, int causalReceiver,
    llmapper::ActionKind action);
llmapper::AffordanceId semanticAffordanceId(
    EngineHandleKind kind, int index, llmapper::ActionKind action);
EngineHandleKind supportKind(SupportRef id);
int supportIndex(SupportRef id);
EngineHandle engineObjectHandle(WorldObjectRef id);
int playerPassageWidth();
int meleeReach();
int plannerVectorReach(VECTOR_TYPE vectorType);
int criticalHealth();
int wrapAngle(int angle);
int angleDelta(int target, int current);
int distance2(int x1, int y1, int x2, int y2);
bool inRange(int value, int low, int high);
bool validXSprite(int extra);
int playerJumpRiseLimit();
int playerReversibleDrop();
int playerCollisionClearance();
int playerStandingClearance();
int playerCrouchClearance();
void playerCollisionDistances(int &ceilingDistance, int &floorDistance);
int lookAngleForTarget(int eyeZ, int targetZ, int horizontal);
PlayerCollisionShape livePlayerCollisionShape();
PlayerCollisionCycle playerMovingCollisionCycle();
PlayerGroundObservation enginePlayerGroundObservationAt(
    int x, int y, int z, int sectorNumber, int footOffset = -1);
std::vector<EngineSpaceTranslation> engineSpaceTranslations();
int engineClipMoveIgnoringActor(
    vec3_t &position, int16_t &sectorNumber,
    int32_t xvect, int32_t yvect, int radius,
    int ceilingDistance, int floorDistance, unsigned clipMask);
void engineGetZRangeIgnoringActor(
    int x, int y, int z, int sectorNumber,
    int *ceilingZ, int *ceilingHit, int *floorZ, int *floorHit,
    int radius, unsigned clipMask, int flags);
MovementProbe probeMovement(
    int startX, int startY, int startZ, int startSector,
    int targetX, int targetY, int targetSector,
    int tolerance, bool crouched = false);
int vectorLookAngleForTarget(int eyeZ, int targetZ, int horizontal);
// Ask Blood whether a player standing on this exact support pose can deliver
// the selected Vector to the semantic target.  Unlike cansee(), this applies
// the player's finite look range and the weapon's real VectorScan geometry.
bool engineVectorHitsTargetFromPose(
    int poseX, int poseY, int supportZ, int poseSector,
    int weaponAboveFloor, int aimX, int aimY, int aimZ,
    VECTOR_TYPE vectorType, int targetSprite, int targetWall,
    int alternateTargetWall = -1);
TntThrowSolution engineSolveTntThrow(
    int targetSprite,
    int targetX, int targetY, int targetTopZ, int targetBottomZ,
    int targetSector, int damagingRadius);
std::vector<StandableSurface> standableSurfacesAt(
    int sectorId, int x, int y, int radius, int requiredClearance);
bool engineHasSupportAt(const SupportRef &support, int sectorId,
                               int x, int y, int surfaceZ, int radius);
bool engineSupportZAt(const SupportRef &support, int x, int y,
                             int &surfaceZ);
int spriteSupportClearance(int sectorId, int spriteId, int x, int y,
                                  int radius);
SupportRef currentPlayerSupport();
bool playerOccupiesSupportPose(const SupportRef &expected,
                                      int expectedZ);
bool engineSupportDamagesPlayer(const SupportRef &support);
bool engineRegionGeometryMoving(int sectorId);
int engineObjectStateSignature(WorldObjectRef object);
int64_t segmentDistance2(int x, int y, int x1, int y1, int x2, int y2);
int solidSpriteAt(int sectorId, int x, int y, int radius);
int playerOriginAtSupport(int supportZ);
bool enginePlayerPoseFits(int sectorId, int x, int y, int supportZ,
                                 bool crouched = false);
int explosiveWeaponRadius(int weapon);
int explosiveSpriteRadius(const spritetype &record);
int safeExplosionSeparation(int damagingRadius);
void setInteractionGeometry(InteractionCandidate &interaction);
void setSpriteInteractionGeometry(InteractionCandidate &interaction, int spriteIndex);
const char *itemCategory(int type);
unsigned acceptedDamageEffects(const spritetype &record);
Observation observeWorld();
// Ask Blood's real Use acquisition which affordances are executable from the
// current physical pose while the actor merely turns in place. This is a lazy
// execution-domain query, not a second interaction store.
std::vector<InteractionCandidate> engineUseAffordancesAtCurrentPose();
const char *objectName(ObjectKind kind);
