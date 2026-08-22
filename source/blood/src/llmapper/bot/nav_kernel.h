//-------------------------------------------------------------------------
// Engine-free movement / exploration kernel for the LLMapper bot.
// Geometry proposes traversability; the NBlood player model validates it.
//-------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <set>
#include <vector>

namespace llmapper
{

enum NavEdgeMode
{
    kNavWalk,
    kNavStep,
    kNavJump,
    kNavCrouch,
    kNavDrop,
    kNavRide,
    kNavInteraction, // prerequisite/affordance, never a route edge
    kNavBlocked,
};

// A Build sector is a topological container, not a height layer.  SupportRef
// names the physical thing under the player so two floors at the same XY do
// not become the same navigation state.
enum SupportKind
{
    kSupportSectorFloor,
    kSupportSpriteFloor,
};

struct SupportRef
{
    SupportKind kind;
    int id;
    SupportRef() : kind(kSupportSectorFloor), id(-1) {}
    SupportRef(SupportKind aKind, int anId) : kind(aKind), id(anId) {}
    bool operator==(const SupportRef &other) const
    {
        return kind == other.kind && id == other.id;
    }
    bool operator!=(const SupportRef &other) const { return !(*this == other); }
    bool operator<(const SupportRef &other) const
    {
        return kind < other.kind || (kind == other.kind && id < other.id);
    }
};

struct NavCondition
{
    int mechanism;
    int state;
    bool enabled;
    NavCondition() : mechanism(-1), state(-1), enabled(false) {}
    NavCondition(int aMechanism, int aState)
        : mechanism(aMechanism), state(aState), enabled(true) {}
};

enum ActivationMode
{
    kActivateUse,
    kActivateVector,
    kActivateImpact,
    kActivateTouch,
    kActivateEnter,
    kActivateExit,
    kActivatePickup,
    kActivateSight,
    kActivateProximity,
    // Damage-driven interactions are not Use/Vector triggers.  The planner
    // records the effect the world object accepts and lets inventory or
    // nearby-world satisfiers provide that effect later.
    kActivateDamage,
};

enum EffectCapability
{
    kEffectNone = 0,
    kEffectExplosive = 1u << 0,
    kEffectBulletDamage = 1u << 1,
};

// Damage opportunities contain alternative accepted effect classes. One
// available producer is sufficient; an empty requirement remains satisfied.
inline bool effectRequirementSatisfied(unsigned accepted, unsigned available)
{
    return accepted == kEffectNone || (accepted & available) != 0;
}

enum WorldObjectKind
{
    kWorldSector,
    kWorldWall,
    kWorldSprite,
};

struct WorldObjectRef
{
    WorldObjectKind kind;
    int id;
    WorldObjectRef() : kind(kWorldSector), id(-1) {}
    WorldObjectRef(WorldObjectKind aKind, int anId) : kind(aKind), id(anId) {}
    bool operator==(const WorldObjectRef &other) const
    {
        return kind == other.kind && id == other.id;
    }
};

enum TraversalResult
{
    kTraverseDirect,
    kTraverseStep,
    kTraverseJump,
    kTraverseCrouch,
    kTraverseDropSafe,
    kTraverseUseableBlocker,
    kTraverseSolidBlocker,
    kTraverseDangerous,
    kTraverseNoFit,
};

enum FrontierKind
{
    kFrontierOpen,
    kFrontierBlocked,
    kFrontierTransport,
};

enum CombatTactic
{
    kCombatNone,
    kCombatRanged,
    kCombatRetreat,
    kCombatMelee,
};

struct NavWaypoint
{
    int x;
    int y;
    NavWaypoint() : x(0), y(0) {}
    NavWaypoint(int ax, int ay) : x(ax), y(ay) {}
};

// A sample along an ordinary walk corridor represents a cross-section, not
// an exact pose.  Report progress only after the player has crossed the
// sample's forward plane and remains within the corridor around that edge.
// Special physical transitions deliberately do not use this helper.
inline bool crossedWaypointCorridor(const NavWaypoint &source,
                                    const NavWaypoint &waypoint,
                                    const NavWaypoint &player,
                                    int corridorHalfWidth)
{
    const int64_t axisX = int64_t(waypoint.x) - source.x;
    const int64_t axisY = int64_t(waypoint.y) - source.y;
    const int64_t pastX = int64_t(player.x) - waypoint.x;
    const int64_t pastY = int64_t(player.y) - waypoint.y;
    const int64_t axisLength2 = axisX * axisX + axisY * axisY;
    if (axisLength2 <= 0 || pastX * axisX + pastY * axisY < 0)
        return false;
    const int64_t cross = pastX * axisY - pastY * axisX;
    const long double cross2 = static_cast<long double>(cross) * cross;
    const long double corridor2 =
        static_cast<long double>(corridorHalfWidth) * corridorHalfWidth;
    return cross2 <= corridor2 * axisLength2;
}

struct NavLink
{
    int target;
    NavEdgeMode mode;
    int wall;
    NavWaypoint gateway;
    bool hasGateway;
    NavWaypoint takeoff;
    bool hasTakeoff;
    NavCondition condition;
    int transition;
    int airControl;
    int airFrames;
    bool hasAirControl;
    // Revalidate this edge whenever live collision/topology changes.  This
    // is deliberately independent of wall identity: a physically valid
    // local transition may cross several tiny Build partitions and therefore
    // have no single wall which owns it.
    bool dynamic;
    NavLink()
        : target(-1), mode(kNavWalk), wall(-1), gateway(), hasGateway(false),
          takeoff(), hasTakeoff(false), condition(), transition(-1),
          airControl(0), airFrames(0), hasAirControl(false), dynamic(false)
    {
    }
};

// One standable square of a sector's interior.  A uniform grid is used
// instead of a triangulation because Build sectors routinely contain inner
// wall loops (pillars, pits, alcoves); per-loop ear clipping silently
// fragments exactly those rooms, and a fragmented mesh reports real routes
// as unreachable.  Grid adjacency is also an O(1) lookup rather than an
// O(cells^2) shared-edge search.
struct NavCell
{
    int id;
    int sector;
    int gx;
    int gy;
    int z;
    SupportRef support;
    NavWaypoint center;
    // Distance to the nearest sector boundary at this concrete pose.  It is
    // a soft execution-safety cost, never a reachability test: narrow routes
    // remain valid when they are the only physical route.
    int clearance;
    int walkArea;
    bool inMotion;
    // True when this pose exists in current engine collision.  A mechanism
    // may also contribute hypothetical stable endpoint cells for conditional
    // planning; those must never be mistaken for a presently occupiable
    // interaction stance.
    bool live;
    // The support pose exists only with the crouched player collision hull.
    // This is occupancy state, not a property of its Build sector.
    bool crouchOnly;
    std::vector<NavLink> links;
    NavCell()
        : id(-1), sector(-1), gx(0), gy(0), z(0), support(), center(),
          clearance(INT32_MAX), walkArea(-1), inMotion(false), live(true),
          crouchOnly(false)
    {
    }
};

struct NavRouteStep
{
    int fromCell;
    int toCell;
    NavWaypoint gateway;
    bool hasGateway;
    NavWaypoint takeoff;
    bool hasTakeoff;
    NavWaypoint destination;
    NavEdgeMode mode;
    int wall;
    int sourceSector;
    int targetSector;
    int sourceZ;
    int targetZ;
    SupportRef sourceSupport;
    SupportRef targetSupport;
    NavCondition condition;
    int transition;
    int airControl;
    int airFrames;
    bool hasAirControl;
    NavRouteStep()
        : fromCell(-1), toCell(-1), gateway(), hasGateway(false), takeoff(),
          hasTakeoff(false), destination(), mode(kNavWalk), wall(-1),
          sourceSector(-1), targetSector(-1),
          sourceZ(0), targetZ(0), sourceSupport(), targetSupport(), condition(),
          transition(-1), airControl(0), airFrames(0), hasAirControl(false)
    {
    }
};

// Stable poses and their physical consequences.  The flags are derived from
// occupancy/clearance/connectivity; they are not mapper object classes.
struct StablePose
{
    int state;
    int supportZ;
    int clearance;
    bool occupiable;
    std::vector<int> connectedSurfaces;
    StablePose() : state(0), supportZ(0), clearance(0), occupiable(false) {}
};

enum DynamicAffordance
{
    kAffordanceNone = 0,
    kAffordanceEnablePassage = 1 << 0,
    kAffordanceTransportSupportedPlayer = 1 << 1,
    kAffordanceUnsafeSweptOccupancy = 1 << 2,
};

struct DynamicMechanism
{
    int id;
    WorldObjectRef object;
    SupportRef support;
    std::vector<StablePose> poses;
    std::vector<int> sweepClearances;
    bool crush;
    bool carriesSupport;
    DynamicMechanism()
        : id(-1), object(), support(), poses(), sweepClearances(), crush(false),
          carriesSupport(false)
    {
    }
};

struct Actuator
{
    int id;
    WorldObjectRef object;
    int locationCell;
    int tx;
    int command;
    bool destructible;
    std::vector<ActivationMode> modes;
    Actuator()
        : id(-1), object(), locationCell(-1), tx(0), command(0),
          destructible(false), modes()
    {
    }
};

struct CausalReceiver
{
    int channel;
    WorldObjectRef object;
    int mechanism;
    int outgoingChannel;
    int command;
    CausalReceiver()
        : channel(0), object(), mechanism(-1), outgoingChannel(0), command(0)
    {
    }
};

struct LearnedEffect
{
    int actuator;
    ActivationMode mode;
    int mechanism;
    int state;
    LearnedEffect()
        : actuator(-1), mode(kActivateUse), mechanism(-1), state(-1) {}
};

struct CausalGraph
{
    std::vector<Actuator> actuators;
    std::vector<CausalReceiver> receivers;
    std::vector<LearnedEffect> effects;

    std::vector<CausalReceiver> receiversFor(int channel) const;
    std::vector<CausalReceiver> receiversReachableFrom(
        int channel, int maxDepth, bool terminalOnly) const;
    const Actuator *actuatorById(int id) const;
    std::vector<LearnedEffect> effectsEstablishing(int mechanism, int state) const;
};

enum PlanOperationKind
{
    kPlanNavigate,
    kPlanTraverse,
    kPlanActivate,
    kPlanWaitForTransition,
    kPlanRemainSupported,
};

struct PlanOperation
{
    PlanOperationKind kind;
    int fromCell;
    int toCell;
    NavEdgeMode traversal;
    int actuator;
    ActivationMode activation;
    int mechanism;
    int state;
    PlanOperation()
        : kind(kPlanNavigate), fromCell(-1), toCell(-1), traversal(kNavWalk),
          actuator(-1), activation(kActivateUse), mechanism(-1), state(-1)
    {
    }
};

struct DynamicPlanStats
{
    int routeSearches;
    int prerequisiteExpansions;
    std::set<int> mechanismsConsidered;
    DynamicPlanStats() : routeSearches(0), prerequisiteExpansions(0) {}
};

struct NavEdgeFailure
{
    int fromCell;
    int toCell;
    int wall;
    // kNavBlocked is the wildcard used when the attempted transition was an
    // observed boundary and its concrete graph mode may be re-derived.
    NavEdgeMode mode;
    int geometrySignature;
    int attempts;
    NavEdgeFailure()
        : fromCell(-1), toCell(-1), wall(-1), mode(kNavWalk), geometrySignature(0),
          attempts(0)
    {
    }
};

struct Boundary
{
    int wall;
    int from;
    int to;
    bool traversable;
    bool jumpable;
    int geometrySignature;
    Boundary()
        : wall(-1), from(-1), to(-1), traversable(false), jumpable(false),
          geometrySignature(0)
    {
    }
};

struct DerivedFrontier
{
    int destination;
    FrontierKind kind;
    std::vector<Boundary> candidates;
    DerivedFrontier() : destination(-1), kind(kFrontierOpen) {}
};

// Semantic exploration samples deliberately contain no Build-sector
// identity.  `partition` exists only so representation-invariance tests can
// prove that changing mapper partitions cannot change the derived work.
// Physical routing remains in NavCell/NavLink; this projection answers only
// whether a reachable visibility boundary is worth investigating.
struct VisibilityCell
{
    int id;
    int x;
    int y;
    int z;
    int area;
    int partition;
    bool observed;
    bool reachable;
    std::vector<int> neighbors;
    VisibilityCell()
        : id(-1), x(0), y(0), z(0), area(-1), partition(-1), observed(false),
          reachable(false)
    {
    }
};

struct VisibilityFrontier
{
    int cell;
    int approachCell;
    int informationGain;
    bool reachable;
    bool requiresOccupancy;
    VisibilityFrontier()
        : cell(-1), approachCell(-1), informationGain(0), reachable(false),
          requiresOccupancy(false) {}
};

struct InvestigateRecord
{
    int wall;
    int from;
    int to;
    int geometrySignature;
    InvestigateRecord() : wall(-1), from(-1), to(-1), geometrySignature(0) {}
};

struct CombatSituation
{
    bool hasThreat;
    bool immediateThreat;
    bool rangedAvailable;
    bool retreatAvailable;
    bool meleeAvailable;
    bool critical;
    CombatTactic current;
    CombatSituation()
        : hasThreat(false), immediateThreat(false), rangedAvailable(false),
          retreatAvailable(false), meleeAvailable(false), critical(false),
          current(kCombatNone)
    {
    }
};

struct CombatDecision
{
    CombatTactic tactic;
    bool overrideMovement;
    bool dropCombat;
    const char *reason;
    CombatDecision()
        : tactic(kCombatNone), overrideMovement(false), dropCombat(true),
          reason("no_threat")
    {
    }
};

// ---------------------------------------------------------------------
// Exploration model.
//
// Everything unresolved lives in one ledger. Reachability, prerequisites
// and retry backoff only annotate that work; they never delete it. Selection
// preserves an actionable commitment, consumes physical space enabled by a
// world-changing action, finishes the current reachable region, then returns
// to the nearest remembered work. Build sectors are not exploration state.
// ---------------------------------------------------------------------

enum OpportunityKind
{
    kOpportunityFrontier,   // a boundary into space never entered
    kOpportunityLocked,     // a continuation gated on a key the bot lacks
    kOpportunityBlocked,    // a continuation with an unresolved obstacle
    kOpportunityInteraction,// an affordance worth using when relevant
    kOpportunityPickup,     // something useful to collect
    kOpportunityExit,       // a visible reachable level-exit affordance
    kOpportunityCoverage,   // reachable space here that has not been looked at
};

inline int mixHash(int hash, int value);

enum WorkIdentityKind
{
    kWorkBoundary,
    kWorkMechanism,
    kWorkObject,
    kWorkPose,
    kWorkExit,
};

// Stable semantic identity for one unit of work.  The old implementation
// encoded the work kind in decimal integer ranges (3,000,000 for boundaries,
// 5,000,000 for pickups, and so on).  Besides being collision-prone, that
// made every consumer reverse-engineer the payload from the number.  Keep the
// identity typed and retain the physical context needed to distinguish two
// directed uses of the same boundary.
struct WorkId
{
    WorkIdentityKind kind;
    int subject;
    int from;
    int to;
    bool valid;
    WorkId()
        : kind(kWorkBoundary), subject(-1), from(-1), to(-1),
          valid(false)
    {
    }
    WorkId(WorkIdentityKind aKind, int aSubject, int aFrom = -1, int aTo = -1)
        : kind(aKind), subject(aSubject), from(aFrom), to(aTo), valid(true)
    {
    }
    bool operator==(const WorkId &other) const
    {
        return valid == other.valid
            && (!valid || (kind == other.kind && subject == other.subject
                           && from == other.from && to == other.to));
    }
    bool operator!=(const WorkId &other) const { return !(*this == other); }
    bool operator<(const WorkId &other) const
    {
        if (valid != other.valid)
            return valid < other.valid;
        if (!valid)
            return false;
        if (kind != other.kind)
            return kind < other.kind;
        if (subject != other.subject)
            return subject < other.subject;
        if (from != other.from)
            return from < other.from;
        return to < other.to;
    }
    explicit operator bool() const { return valid; }
};

inline int workIdHash(const WorkId &work)
{
    if (!work)
        return -1;
    int hash = mixHash(int(work.kind), work.subject);
    hash = mixHash(hash, work.from);
    return mixHash(hash, work.to);
}

struct Opportunity
{
    WorkId id;
    OpportunityKind kind;
    int sector;        // where the bot must stand to act
    int target;        // sector it leads to, -1 when not a crossing
    int approach;      // reachable observation/takeoff pose, or -1
    int wall;
    int requiredKey;   // 0 when no key is involved
    unsigned requiredEffects; // abstract effects, independent of their satisfier
    int depth;         // retained for telemetry; never used for selection
    int hops;          // route distance from the bot right now, -1 unreachable
    int descent;       // height given up by taking it, 0 when level or upward
    int oneWayRisk;    // 0 reversible/unknown-safe, >0 known loss of optionality
    bool local;        // physically in the player's current reachable region
    bool continuation;// observes space newly enabled by a causal world change
    bool ready;        // actor currently has a valid pose for the task
    bool requiresOccupancy; // player must occupy the pose; seeing its surface is insufficient
    Opportunity()
        : id(), kind(kOpportunityFrontier), sector(-1), target(-1), approach(-1), wall(-1),
          requiredKey(0), requiredEffects(kEffectNone), depth(0), hops(-1),
          descent(0), oneWayRisk(0),
          local(false), continuation(false), ready(false),
          requiresOccupancy(false)
    {
    }
};

// The policy returns only stable work identity and an explanatory reason.
// It does not create a second intent hierarchy between WorkItem and Plan.
struct WorkSelection
{
    WorkId work;
    const char *reason;
    WorkSelection() : work(), reason("NO_APPLICABLE_ACTION") {}
    explicit operator bool() const { return bool(work); }
};

// Rank the ledger and return the single next thing to do.
//
// `heldKeys` is a bitmask of key ids 1..15 the bot currently carries.
// The selector is intentionally kind-agnostic. It does not have separate
// priority ladders for keys, exits, pickups, doors, and coverage. Those are
// all unresolved work with availability and distance annotations.
WorkSelection selectWork(const std::vector<Opportunity> &ledger,
                         unsigned heldKeys,
                         unsigned availableEffects = ~0u);

// Resolve an interaction/object XY onto navigation support without letting
// an unrelated overlapping layer win merely because its floor Z resembles
// the target's aim Z.  The exact-support projection is useful when it is
// genuinely the closer physical endpoint (for example the far end of a
// bridge); otherwise the currently reachable stance is the causal approach.
inline int selectTargetNavCell(int areaCell, int64_t areaDistance2,
                               int strictCell, int64_t strictDistance2,
                               int64_t stanceDistance2)
{
    if (areaCell < 0)
        return strictCell;
    if (strictCell < 0)
        return areaCell;
    if (areaDistance2 <= stanceDistance2
        || areaDistance2 <= strictDistance2)
        return areaCell;
    return strictCell;
}

// Physical attempt identity is deliberately separate from causal receiver
// identity. Several faces around one pushable sector are one actuator, but
// two actuator sectors remain two things to try even when their TX channels
// ultimately affect the same receiver. A bare XWALL has no actuator sector,
// so its own wall record is the stable physical identity.
inline int wallInteractionAttemptKey(int wallId, int fromSector,
                                     int targetSector, bool wallPush,
                                     bool sectorPush, bool sectorPushCurrent,
                                     int causalReceiver)
{
    if (sectorPush && targetSector >= 0)
        return 4000000 + targetSector + 1;
    if (sectorPushCurrent && fromSector >= 0)
        return 4000000 + fromSector + 1;
    if (wallPush && targetSector >= 0)
        return 4000000 + targetSector + 1;
    if (wallPush && wallId >= 0)
        return 5000000 + wallId + 1;
    if (causalReceiver >= 0)
        return 4000000 + causalReceiver + 1;
    return -1;
}

inline int wallInteractionActuatorSector(int observedTargetSector,
                                         int immediateReceiverSector)
{
    return observedTargetSector >= 0
        ? observedTargetSector : immediateReceiverSector;
}

inline bool shouldReselectInteractionSurface(bool activeObjectiveOwnsMemory,
                                             bool differentSurface,
                                             bool candidateIsCloser)
{
    return !activeObjectiveOwnsMemory && differentSurface && candidateIsCloser;
}

inline bool preserveActiveInteractionSurface(bool activeObjectiveOwnsMemory,
                                             bool differentSurface)
{
    return activeObjectiveOwnsMemory && differentSurface;
}

inline int mixHash(int hash, int value)
{
    return hash * 31 + value;
}

inline bool traversableMode(NavEdgeMode mode)
{
    return mode == kNavWalk || mode == kNavStep || mode == kNavJump
        || mode == kNavCrouch || mode == kNavDrop || mode == kNavRide;
}

inline bool walkMode(NavEdgeMode mode)
{
    return mode == kNavWalk || mode == kNavStep;
}

inline bool edgeFailed(const NavEdgeFailure &failure, int fromCell, int toCell,
                       int wall, NavEdgeMode mode, int geometrySignature)
{
    // Unset fields act as wildcards, so a record that identifies no wall and
    // no cells would match every link and erase the whole mesh.  Such a
    // record describes nothing and must never block anything.
    if (failure.wall < 0 && failure.fromCell < 0 && failure.toCell < 0)
        return false;
    if (geometrySignature != 0 && failure.geometrySignature != 0
        && failure.geometrySignature != geometrySignature)
        return false;
    // Only unset fields in the RECORD are wildcards.  Treating an unset
    // field in the QUERY as a wildcard too meant a record naming one wall
    // matched every wall-less intra-sector link -- that is, the whole
    // navigation mesh -- and the bot lost the ability to cross its own room.
    if (failure.wall >= 0 && failure.wall != wall)
        return false;
    if (failure.fromCell >= 0 && failure.fromCell != fromCell)
        return false;
    if (failure.toCell >= 0 && failure.toCell != toCell)
        return false;
    if (failure.mode != kNavBlocked && failure.mode != mode)
        return false;
    return true;
}

inline bool edgeFailedAny(const std::vector<NavEdgeFailure> &failures, int fromCell,
                          int toCell, int wall, NavEdgeMode mode,
                          int geometrySignature)
{
    for (size_t i = 0; i < failures.size(); ++i)
    {
        if (edgeFailed(failures[i], fromCell, toCell, wall, mode, geometrySignature))
            return true;
    }
    return false;
}

inline TraversalResult classifyTraversal(int floorDelta, int clearance,
                                         int bodyClearance, int jumpRise,
                                         int maxWalkStep, bool clipReachable,
                                         bool hitSolid, bool useableBlocker,
                                         bool clipFits)
{
    // A horizontal clipmove probe is the authority for ordinary movement,
    // not for a future jump arc.  A raised, clear portal may deliberately
    // fail the horizontal probe until the player presses Blood's jump input.
    // Keep body clearance as a hard constraint, but preserve that special
    // capability edge even when the probe ended at the source-side wall.
    if (clearance < bodyClearance)
        return kTraverseNoFit;
    const int rise = floorDelta < 0 ? -floorDelta : 0;
    const int drop = floorDelta > 0 ? floorDelta : 0;
    if (drop > jumpRise)
        return kTraverseDangerous;
    if (rise > jumpRise)
        return kTraverseNoFit;
    if (rise > maxWalkStep && rise <= jumpRise)
        return kTraverseJump;
    if (!clipFits)
        return kTraverseNoFit;
    if (rise <= maxWalkStep && drop <= maxWalkStep)
    {
        if (clipReachable)
            return floorDelta == 0 ? kTraverseDirect : kTraverseStep;
        if (useableBlocker)
            return kTraverseUseableBlocker;
        if (hitSolid)
            return kTraverseSolidBlocker;
        return kTraverseNoFit;
    }
    if (drop > maxWalkStep && drop <= jumpRise)
        return kTraverseDropSafe;
    if (useableBlocker)
        return kTraverseUseableBlocker;
    if (hitSolid)
        return kTraverseSolidBlocker;
    if (!clipReachable)
        return kTraverseNoFit;
    return kTraverseDirect;
}

inline TraversalResult classifyTraversalForPostures(
    int floorDelta, int clearance, int standingClearance, int crouchClearance,
    int jumpRise, int maxWalkStep, bool clipReachable, bool hitSolid,
    bool useableBlocker, bool clipFits)
{
    if (clearance < standingClearance)
    {
        if (clearance < crouchClearance || std::abs(floorDelta) > maxWalkStep
            || !clipFits)
            return kTraverseNoFit;
        if (clipReachable)
            return kTraverseCrouch;
        if (useableBlocker)
            return kTraverseUseableBlocker;
        if (hitSolid)
            return kTraverseSolidBlocker;
        return kTraverseNoFit;
    }
    return classifyTraversal(floorDelta, clearance, standingClearance, jumpRise,
                             maxWalkStep, clipReachable, hitSolid, useableBlocker,
                             clipFits);
}

inline NavEdgeMode modeFromTraversal(TraversalResult result)
{
    switch (result)
    {
    case kTraverseStep: return kNavStep;
    case kTraverseJump: return kNavJump;
    case kTraverseCrouch: return kNavCrouch;
    case kTraverseDropSafe: return kNavDrop;
    case kTraverseUseableBlocker: return kNavInteraction;
    case kTraverseSolidBlocker:
    case kTraverseDangerous:
    case kTraverseNoFit: return kNavBlocked;
    default: return kNavWalk;
    }
}

int deriveDynamicAffordances(const DynamicMechanism &mechanism,
                             int requiredClearance);

NavLink makeConditionalTraversal(int target, NavEdgeMode mode, int mechanism,
                                 int state, int transition = -1);

bool planDynamicRoute(const std::vector<NavCell> &cells, int startCell,
                      int targetCell, const std::map<int, int> &mechanismStates,
                      const CausalGraph &causality,
                      std::vector<PlanOperation> &outPlan,
                      DynamicPlanStats *stats = nullptr);

void assignWalkAreas(std::vector<NavCell> &cells);

// Mark the cells physically reachable in the current geometry.  Sector ids
// are only Build containers: one sector can contain several disconnected
// support layers, so sector membership alone is not proof that an approach
// pose can be reached.
void markReachableNavCells(const std::vector<NavCell> &cells, int startCell,
                           const std::vector<NavEdgeFailure> &failures,
                           int geometrySignature,
                           std::vector<char> &reachable);

// Blood's stacked-room links are explicit engine transitions.  Connect two
// otherwise independent sector-local layers through the marker-authored XY
// translation; mere XY overlap never creates an edge.  The upper-to-lower
// direction is a fall and the reverse direction requires a jump through the
// lower ceiling.
int linkTranslatedNavLayers(std::vector<NavCell> &cells, int upperSector,
                            int lowerSector, int deltaX, int deltaY,
                            int maximumError, int transitionEdge);

bool planNavRoute(const std::vector<NavCell> &cells, int startCell,
                  int targetCell,
                  const std::vector<NavEdgeFailure> &failures,
                  int geometrySignature,
                  std::vector<NavRouteStep> &outRoute);

std::vector<DerivedFrontier> deriveFrontiers(
    const std::vector<int> &visitedSectors, const std::vector<Boundary> &boundaries,
    const std::vector<InvestigateRecord> &investigated,
    const std::vector<NavEdgeFailure> &failedCrossings);

int selectFrontierIndex(const std::vector<DerivedFrontier> &frontiers,
                        int currentSector, const int *hops, int hopCount);

std::vector<VisibilityFrontier> deriveVisibilityFrontiers(
    const std::vector<VisibilityCell> &cells, int mergeRadius,
    int gainRadius, int approachRadius = 0x7fffffff,
    int maximumRise = 0x7fffffff);

const char *navEdgeModeName(NavEdgeMode mode);
const char *traversalResultName(TraversalResult result);
const char *combatTacticName(CombatTactic tactic);

CombatDecision chooseCombatTactic(const CombatSituation &situation);

inline bool investigatedNow(const std::vector<InvestigateRecord> &records,
                            int wall, int from, int to, int geometrySignature)
{
    for (size_t i = 0; i < records.size(); ++i)
    {
        if (records[i].wall == wall && records[i].from == from && records[i].to == to
            && records[i].geometrySignature == geometrySignature)
            return true;
    }
    return false;
}

} // namespace llmapper
