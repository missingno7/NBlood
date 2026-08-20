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
};

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

struct NavLink
{
    int target;
    NavEdgeMode mode;
    int wall;
    NavWaypoint gateway;
    bool hasGateway;
    NavCondition condition;
    int transition;
    NavLink()
        : target(-1), mode(kNavWalk), wall(-1), gateway(), hasGateway(false),
          condition(), transition(-1)
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
    int walkArea;
    bool inMotion;
    std::vector<NavLink> links;
    NavCell()
        : id(-1), sector(-1), gx(0), gy(0), z(0), support(), center(),
          walkArea(-1), inMotion(false)
    {
    }
};

struct NavRouteStep
{
    int fromCell;
    int toCell;
    NavWaypoint gateway;
    bool hasGateway;
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
    NavRouteStep()
        : fromCell(-1), toCell(-1), gateway(), hasGateway(false), destination(),
          mode(kNavWalk), wall(-1), sourceSector(-1), targetSector(-1),
          sourceZ(0), targetZ(0), sourceSupport(), targetSupport(), condition(),
          transition(-1)
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
    CausalReceiver() : channel(0), object(), mechanism(-1) {}
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
    NavEdgeMode mode;
    int geometrySignature;
    // A local trajectory failure is temporary knowledge, not topology.
    // The owner expires the record so a real route is never deleted for
    // good by one bad approach.  Zero means "no expiry recorded".
    int expiresTick;
    int attempts;
    NavEdgeFailure()
        : fromCell(-1), toCell(-1), wall(-1), mode(kNavWalk), geometrySignature(0),
          expiresTick(0), attempts(0)
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
// The bot follows one branch into new territory and remembers what it
// passes.  Everything unresolved lives in one ledger; most of it stays
// dormant until circumstances make it useful.  Selection is a single
// ranked choice with an explicit, reportable reason, so any run can answer
// "what am I doing and why" from telemetry alone.
// ---------------------------------------------------------------------

enum OpportunityKind
{
    kOpportunityFrontier,   // a boundary into space never entered
    kOpportunityLocked,     // a continuation gated on a key the bot lacks
    kOpportunityBlocked,    // a continuation with an unresolved obstacle
    kOpportunityInteraction,// an affordance worth using when relevant
    kOpportunityPickup,     // something useful to collect
    kOpportunityCoverage,   // reachable space here that has not been looked at
};

enum MissionKind
{
    kMissionNone,
    kMissionContinue,        // keep pushing the branch the bot is on
    kMissionReturnForKey,    // a held key just made a known door actionable
    kMissionReturnToBranch,  // this branch ended; go back to unresolved work
    kMissionSolveBlocker,    // the way forward is blocked; work the obstacle
    kMissionCollect,         // pick something up that is on the way
    kMissionExpose,          // go look at reachable space not yet observed
};

struct Opportunity
{
    int id;
    OpportunityKind kind;
    int sector;        // where the bot must stand to act
    int target;        // sector it leads to, -1 when not a crossing
    int wall;
    int requiredKey;   // 0 when no key is involved
    unsigned requiredEffects; // abstract effects, independent of their satisfier
    int depth;         // exploration-tree depth of the discovering node
    int hops;          // route distance from the bot right now, -1 unreachable
    int dormantUntil;  // tick before which this stays out of the way
    int descent;       // height given up by taking it, 0 when level or upward
    bool local;        // discovered from, and actionable in, the current sector
    Opportunity()
        : id(-1), kind(kOpportunityFrontier), sector(-1), target(-1), wall(-1),
          requiredKey(0), requiredEffects(kEffectNone), depth(0), hops(-1), dormantUntil(0), descent(0),
          local(false)
    {
    }
};

struct Mission
{
    MissionKind kind;
    int opportunity;
    const char *reason;
    Mission() : kind(kMissionNone), opportunity(-1), reason("NO_KNOWN_PROGRESS") {}
};

// Rank the ledger and return the single next thing to do.
//
// `heldKeys` is a bitmask of key ids 1..15 the bot currently carries.
// Ordering is deliberately depth-first: a deeper pending frontier is the
// continuation of the branch already being followed, so preferring depth
// gives forward momentum, and popping to the next-deepest gives a natural
// backtrack instead of a random hop across the map.
Mission selectMission(const std::vector<Opportunity> &ledger, int tick,
                      unsigned heldKeys, int committedOpportunity,
                      unsigned availableEffects = ~0u);

const char *missionReason(MissionKind kind);

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
    if (failure.mode != mode)
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

bool planNavRoute(const std::vector<NavCell> &cells, int startCell, int targetCell,
                  int targetX, int targetY, int targetSector, int crossingWall,
                  const std::vector<NavEdgeFailure> &failures, int geometrySignature,
                  std::vector<NavRouteStep> &outRoute);

std::vector<DerivedFrontier> deriveFrontiers(
    const std::vector<int> &visitedSectors, const std::vector<Boundary> &boundaries,
    const std::vector<InvestigateRecord> &investigated,
    const std::vector<NavEdgeFailure> &failedCrossings);

int selectFrontierIndex(const std::vector<DerivedFrontier> &frontiers,
                        int currentSector, const int *hops, int hopCount);

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
