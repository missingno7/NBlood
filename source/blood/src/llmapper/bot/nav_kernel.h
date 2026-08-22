//-------------------------------------------------------------------------
// Engine-free movement / exploration kernel for the LLMapper bot.
// Adapter observations propose traversability; engine physics validates it.
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

template <typename Tag>
struct SemanticId
{
    int value;
    SemanticId() : value(-1) {}
    SemanticId(int aValue) : value(aValue) {}
    explicit operator bool() const { return value >= 0; }
    operator int() const { return value; }
    bool operator==(const SemanticId &other) const { return value == other.value; }
    bool operator!=(const SemanticId &other) const { return value != other.value; }
    bool operator<(const SemanticId &other) const { return value < other.value; }
    bool operator==(int other) const { return value == other; }
    bool operator!=(int other) const { return value != other; }
    bool operator<(int other) const { return value < other; }
    bool operator<=(int other) const { return value <= other; }
    bool operator>(int other) const { return value > other; }
    bool operator>=(int other) const { return value >= other; }
};

struct SupportTag;
struct ObjectTag;
struct RegionTag;
struct PoseTag;
struct BoundaryTag;
struct TransitionTag;
struct AffordanceTag;
struct StateVariableTag;

using SupportId = SemanticId<SupportTag>;
using ObjectId = SemanticId<ObjectTag>;
using RegionId = SemanticId<RegionTag>;
using PoseId = SemanticId<PoseTag>;
using BoundaryId = SemanticId<BoundaryTag>;
using TransitionId = SemanticId<TransitionTag>;
using AffordanceId = SemanticId<AffordanceTag>;
using StateVariableId = SemanticId<StateVariableTag>;

struct NavCondition
{
    StateVariableId variable;
    int state;
    bool enabled;
    NavCondition() : variable(), state(-1), enabled(false) {}
    NavCondition(StateVariableId aVariable, int aState)
        : variable(aVariable), state(aState), enabled(true) {}
};

// Actions describe what the actor does. The adapter decides which native
// delivery modality realizes an effect.
enum ActionKind
{
    kActionUse,
    kActionDeliverRemoteEffect,
    kActionCollide,
    kActionTouch,
    kActionEnter,
    kActionExit,
    kActionPickup,
    kActionObserve,
    kActionApproach,
    kActionDeliverDamage,
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

// Adapter-facing convex decomposition of one 2D support footprint. The
// planner still receives only arbitrary XYZ poses; these polygons are a
// bounded construction aid for placing those poses on the free-space
// centerline instead of beside source-geometry walls.
struct SkeletonCell
{
    std::vector<NavWaypoint> polygon;
    NavWaypoint center;
};

struct SkeletonGateway
{
    int first;
    int second;
    NavWaypoint center;
    SkeletonGateway() : first(-1), second(-1), center() {}
};

struct ConvexSkeleton
{
    std::vector<SkeletonCell> cells;
    std::vector<SkeletonGateway> gateways;
};

ConvexSkeleton buildConvexSkeleton(
    const std::vector<NavWaypoint> &footprint);

// Geometry-driven decomposition for a support footprint with any number of
// boundary loops (outer contours and holes).  Split coordinates come only
// from actual vertices; this is not spatial rasterization.
ConvexSkeleton buildConvexSkeleton(
    const std::vector<std::vector<NavWaypoint> > &contours);

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
    PoseId target;
    NavEdgeMode mode;
    BoundaryId boundary;
    NavWaypoint gateway;
    bool hasGateway;
    NavWaypoint takeoff;
    bool hasTakeoff;
    NavCondition condition;
    TransitionId transition;
    int airControl;
    int airControlAfter;
    int airControlSwitchFrame;
    int airFrames;
    int launchVelocity;
    bool hasAirControl;
    // Revalidate this edge whenever live collision/topology changes. This is
    // independent of adapter provenance: a physical transition may cross
    // several engine partitions and have no single boundary owner.
    bool dynamic;
    NavLink()
        : target(), mode(kNavWalk), boundary(), gateway(), hasGateway(false),
          takeoff(), hasTakeoff(false), condition(), transition(),
          airControl(0), airControlAfter(0), airControlSwitchFrame(0),
          airFrames(0), launchVelocity(0),
          hasAirControl(false), dynamic(false)
    {
    }
};

// One standable pose. Spatial partitioning and tessellation are adapter
// concerns; the planner sees only a region, a support and physical links.
struct NavCell
{
    PoseId id;
    RegionId region;
    int z;
    SupportId support;
    NavWaypoint center;
    // Distance to the nearest boundary at this concrete pose. It is
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
    // This is occupancy state, not a property of an engine partition.
    bool crouchOnly;
    // This concrete pose lies on the visible side of an occlusion with
    // engine-valid but not-yet-visible occupiable space beyond it. It is a
    // spatial exploration fact, not an adapter-region boundary.
    bool informationFrontier;
    std::vector<NavLink> links;
    NavCell()
        : id(), region(), z(0), support(), center(),
          clearance(INT32_MAX), walkArea(-1), inMotion(false), live(true),
          crouchOnly(false), informationFrontier(false)
    {
    }
};

struct NavRouteStep
{
    PoseId fromCell;
    PoseId toCell;
    NavWaypoint gateway;
    bool hasGateway;
    NavWaypoint takeoff;
    bool hasTakeoff;
    NavWaypoint destination;
    NavEdgeMode mode;
    BoundaryId boundary;
    RegionId sourceRegion;
    RegionId targetRegion;
    int sourceZ;
    int targetZ;
    SupportId sourceSupport;
    SupportId targetSupport;
    NavCondition condition;
    TransitionId transition;
    int airControl;
    int airControlAfter;
    int airControlSwitchFrame;
    int airFrames;
    int launchVelocity;
    bool hasAirControl;
    NavRouteStep()
        : fromCell(), toCell(), gateway(), hasGateway(false), takeoff(),
          hasTakeoff(false), destination(), mode(kNavWalk), boundary(),
          sourceRegion(), targetRegion(),
          sourceZ(0), targetZ(0), sourceSupport(), targetSupport(), condition(),
          transition(), airControl(0), airControlAfter(0),
          airControlSwitchFrame(0), airFrames(0), launchVelocity(0),
          hasAirControl(false)
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

struct StatefulGeometry
{
    StateVariableId stateVariable;
    ObjectId object;
    SupportId support;
    std::vector<StablePose> poses;
    std::vector<int> sweepClearances;
    bool crush;
    bool carriesSupport;
    StatefulGeometry()
        : stateVariable(), object(), support(), poses(), sweepClearances(), crush(false),
          carriesSupport(false)
    {
    }
};

struct Affordance
{
    AffordanceId id;
    ObjectId target;
    PoseId actionPose;
    int command;
    bool destructible;
    std::vector<ActionKind> actions;
    Affordance()
        : id(), target(), actionPose(), command(0),
          destructible(false), actions()
    {
    }
};

struct LearnedEffect
{
    AffordanceId affordance;
    ActionKind action;
    StateVariableId variable;
    int state;
    LearnedEffect()
        : affordance(), action(kActionUse), variable(), state(-1) {}
};

struct CausalGraph
{
    std::vector<Affordance> affordances;
    std::vector<LearnedEffect> effects;

    const Affordance *affordanceById(AffordanceId id) const;
    std::vector<LearnedEffect> effectsEstablishing(StateVariableId variable,
                                                    int state) const;
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
    PoseId fromCell;
    PoseId toCell;
    NavEdgeMode traversal;
    AffordanceId affordance;
    ActionKind action;
    StateVariableId variable;
    int state;
    PlanOperation()
        : kind(kPlanNavigate), fromCell(), toCell(), traversal(kNavWalk),
          affordance(), action(kActionUse), variable(), state(-1)
    {
    }
};

struct DynamicPlanStats
{
    int routeSearches;
    int prerequisiteExpansions;
    std::set<StateVariableId> variablesConsidered;
    DynamicPlanStats() : routeSearches(0), prerequisiteExpansions(0) {}
};

struct NavEdgeFailure
{
    PoseId fromCell;
    PoseId toCell;
    BoundaryId boundary;
    // kNavBlocked is the wildcard used when the attempted transition was an
    // observed boundary and its concrete graph mode may be re-derived.
    NavEdgeMode mode;
    int geometrySignature;
    int attempts;
    NavEdgeFailure()
        : fromCell(), toCell(), boundary(), mode(kNavWalk), geometrySignature(0),
          attempts(0)
    {
    }
};

struct Boundary
{
    BoundaryId id;
    RegionId source;
    RegionId destination;
    bool traversable;
    bool jumpable;
    int geometrySignature;
    Boundary()
        : id(), source(), destination(), traversable(false), jumpable(false),
          geometrySignature(0)
    {
    }
};

struct DerivedFrontier
{
    RegionId destination;
    FrontierKind kind;
    std::vector<Boundary> candidates;
    DerivedFrontier() : destination(), kind(kFrontierOpen) {}
};

// Semantic exploration samples contain no engine-container identity.
// `partition` exists only so representation-invariance tests can prove that
// changing adapter tessellation cannot change the derived work.
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
    BoundaryId boundary;
    RegionId source;
    RegionId destination;
    int geometrySignature;
    InvestigateRecord() : boundary(), source(), destination(), geometrySignature(0) {}
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
// to the nearest remembered work. Engine partitions are not exploration state.
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
    kWorkStateVariable,
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
    PoseId pose;                 // where the actor must stand to act
    RegionId destination;       // region it exposes, invalid when not spatial
    PoseId approach;             // reachable observation/takeoff pose
    BoundaryId transition;
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
        : id(), kind(kOpportunityFrontier), pose(), destination(), approach(), transition(),
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
// priority ladders for keys, exits, pickups, mechanisms, and coverage. Those are
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

inline bool edgeFailed(const NavEdgeFailure &failure, PoseId fromCell, PoseId toCell,
                       BoundaryId boundary, NavEdgeMode mode, int geometrySignature)
{
    // Unset fields act as wildcards, so a record that identifies no boundary and
    // no cells would match every link and erase the whole mesh.  Such a
    // record describes nothing and must never block anything.
    if (!failure.boundary && !failure.fromCell && !failure.toCell)
        return false;
    if (geometrySignature != 0 && failure.geometrySignature != 0
        && failure.geometrySignature != geometrySignature)
        return false;
    // Only unset fields in the RECORD are wildcards.  Treating an unset
    // field in the QUERY as a wildcard too meant a record naming one boundary
    // matched every boundary-less local link -- that is, the whole
    // navigation mesh -- and the bot lost the ability to cross its own room.
    if (failure.boundary && failure.boundary != boundary)
        return false;
    if (failure.fromCell >= 0 && failure.fromCell != fromCell)
        return false;
    if (failure.toCell >= 0 && failure.toCell != toCell)
        return false;
    if (failure.mode != kNavBlocked && failure.mode != mode)
        return false;
    return true;
}

inline bool edgeFailedAny(const std::vector<NavEdgeFailure> &failures, PoseId fromCell,
                          PoseId toCell, BoundaryId boundary, NavEdgeMode mode,
                          int geometrySignature)
{
    for (size_t i = 0; i < failures.size(); ++i)
    {
        if (edgeFailed(failures[i], fromCell, toCell, boundary, mode, geometrySignature))
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
    // not for a future jump arc. A raised, clear opening may deliberately
    // fail the horizontal probe until the actor applies jump input.
    // Keep body clearance as a hard constraint, but preserve that special
    // capability edge even when the probe ended at the source boundary.
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

int deriveDynamicAffordances(const StatefulGeometry &geometry,
                             int requiredClearance);

NavLink makeConditionalTraversal(PoseId target, NavEdgeMode mode,
                                 StateVariableId variable, int state,
                                 TransitionId transition = TransitionId());

bool planDynamicRoute(const std::vector<NavCell> &cells, PoseId startCell,
                      PoseId targetCell,
                      const std::map<StateVariableId, int> &worldState,
                      const CausalGraph &causality,
                      std::vector<PlanOperation> &outPlan,
                      DynamicPlanStats *stats = nullptr);

void assignWalkAreas(std::vector<NavCell> &cells);

// Mark the poses physically reachable in the current geometry. Region
// membership alone is not proof that an approach pose can be reached.
void markReachableNavCells(const std::vector<NavCell> &cells, PoseId startCell,
                           const std::vector<NavEdgeFailure> &failures,
                           int geometrySignature,
                           std::vector<char> &reachable);

// Connect two otherwise independent region-local layers through an
// adapter-observed translation; mere XY overlap never creates an edge.
int linkTranslatedNavLayers(std::vector<NavCell> &cells, RegionId upperRegion,
                            SupportId upperSupport, RegionId lowerRegion,
                            SupportId lowerSupport, int deltaX, int deltaY,
                            int maximumError, TransitionId transition);

bool planNavRoute(const std::vector<NavCell> &cells, PoseId startCell,
                  PoseId targetCell,
                  const std::vector<NavEdgeFailure> &failures,
                  int geometrySignature,
                  std::vector<NavRouteStep> &outRoute);

std::vector<DerivedFrontier> deriveFrontiers(
    const std::vector<RegionId> &visitedRegions,
    const std::vector<Boundary> &boundaries,
    const std::vector<InvestigateRecord> &investigated,
    const std::vector<NavEdgeFailure> &failedCrossings);

int selectFrontierIndex(const std::vector<DerivedFrontier> &frontiers,
                        RegionId currentRegion, const int *hops, int hopCount);

std::vector<VisibilityFrontier> deriveVisibilityFrontiers(
    const std::vector<VisibilityCell> &cells, int mergeRadius,
    int gainRadius, int approachRadius = 0x7fffffff,
    int maximumRise = 0x7fffffff);

const char *navEdgeModeName(NavEdgeMode mode);
const char *traversalResultName(TraversalResult result);
const char *combatTacticName(CombatTactic tactic);

CombatDecision chooseCombatTactic(const CombatSituation &situation);

inline bool investigatedNow(const std::vector<InvestigateRecord> &records,
                            BoundaryId boundary, RegionId source,
                            RegionId destination, int geometrySignature)
{
    for (size_t i = 0; i < records.size(); ++i)
    {
        if (records[i].boundary == boundary && records[i].source == source
            && records[i].destination == destination
            && records[i].geometrySignature == geometrySignature)
            return true;
    }
    return false;
}

} // namespace llmapper
