//-------------------------------------------------------------------------
// Traversal derivation: world geometry plus the actor's current physics.
//
// A SpatialRelation says two pieces of space are next to each other. Whether
// a body can get across one is a different question, and it is answered here
// by asking the physics authority -- never by comparing a height difference
// to a number. All modes come out of the same relations into the same
// topology; there is no walk graph and no jump graph.
//
// The graph is not over Regions. A Region is one continuous piece of space,
// but a body has width, and a wall leaving a gap narrower than the body cuts
// that space in two for that body while leaving it one space. So the nodes
// here are Places -- a Region and one piece of its free space -- which is a
// fact about the world and the actor together, exactly what this layer is
// for. Change the body and the Places are re-derived; the world does not
// move.
//
// This file has no engine dependency. The oracle it consults does.
//-------------------------------------------------------------------------
#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <vector>

#include "../nav/local_path.h"
#include "../semantic/semantic_world.h"

namespace traversal {

enum class Mode
{
    Walk,
    Crouch,
    Jump,
    Drop,
    Ride,
};

constexpr int kModeCount = 5;

inline uint32_t modeBit(Mode mode) { return 1u << int(mode); }

const char *modeName(Mode mode);

// What the actor physically is, right now. Changing this changes what is
// traversable and nothing about the world.
struct ActorProfile
{
    int radius = 0;
    int standHeight = 0;
    int crouchHeight = 0;
    int stepUp = 0;
    int walkSpeed = 0;
    int jumpImpulse = 0;
    int gravity = 0;
    uint64_t revision = 0; // bumped by the physics layer when anything moves

    bool operator==(const ActorProfile &other) const
    {
        return revision == other.revision;
    }
};

class PhysicsOracle
{
public:
    virtual ~PhysicsOracle() = default;
    virtual const ActorProfile &profile() const = 0;
    // Can a body of this actor's size be at this point in this region?
    //
    // The free-space map proposes places from the shape of the floor alone.
    // Whether the world actually holds the body up at one of them is this
    // layer's answer, and it is not the same question: a neighbour's floor
    // can reach into the hull from a height a flat map knows nothing about.
    virtual bool canStand(const semantic::Region &region,
                          const semantic::Vec2 &at) const = 0;

    // The authority. Given the situation the world describes, can the actor
    // as it currently is carry out this mode across it?
    //
    // `startFrom` is a place in `from` this same layer has already agreed
    // the body can be, supplied by the caller because the caller is the one
    // that worked out where this body fits. Null means there is no such
    // place. Asking the physics layer to invent a stance instead is how a
    // model ends up disagreeing with itself about where a body can be.
    //
    // `crossing` comes back as the point on the gateway the answer was
    // established at, and `arrival` as the point on the far side the body
    // was actually left standing at, so the executor can drive the line that
    // was verified rather than deriving its own and disagreeing.
    //
    // Both are needed, and for different reasons. An opening is a line
    // between two spaces, so a body driven at a point on it has no reason to
    // go through: it arrives and stops, and which side it is then on is a
    // coin toss. What makes a crossing happen is aiming past it.
    virtual bool canTraverse(Mode mode, const semantic::SpatialRelation &relation,
                             const semantic::Region &from,
                             const semantic::Region &to,
                             const semantic::Vec2 *startFrom,
                             semantic::Vec2 &crossing,
                             semantic::Vec2 &arrival,
                             semantic::Vec2 &departure) const = 0;
};

struct Transition
{
    semantic::RelationId relation = semantic::kNoId;
    semantic::RegionId from = semantic::kNoId;
    semantic::RegionId to = semantic::kNoId;
    Mode mode = Mode::Walk;
    semantic::Vec2 crossing;
    semantic::Vec2 arrival;
    int cost = 0;
    // The two halves of that cost: getting to the opening from where this
    // region is reckoned to be, and getting away from it on the other side.
    // Kept apart so the ends of a journey can be measured from where the
    // body actually is and where it is actually going, rather than from the
    // middles of the rooms at each end.
    int head = 0;
    int tail = 0;
    bool possible = false;   // actual Caleb can perform it
    bool executable = false; // this bot has an executor for it
};

// A crossing the physics layer said the body can make, that this layer then
// did not offer the planner. Diagnostic only, and the answer to "the way out
// is right there and it will not take it".
struct DroppedCrossing
{
    semantic::RelationId relation = semantic::kNoId;
    int mode = 0;
    int leaves = 0;      // pieces of the near side the crossing was placed in
    int arrives = 0;     // and of the far side
    int fromStances = 0; // places the physics layer agreed a body can be
    int toStances = 0;
    // The two poses whose piece of free space could not be named.
    semantic::Vec2 departure;
    semantic::Vec2 arrival;
};

class TraversalModel
{
public:
    // Which modes this bot can actually drive. Changing it changes what the
    // planner may select and nothing else.
    void setExecutableModes(uint32_t mask);
    uint32_t executableModes() const { return m_executable; }

    void update(const semantic::SemanticWorld &world,
                const PhysicsOracle &oracle);
    void clear();

    // Shared with the executor so a region's free space is worked out once,
    // and bounded by the same walls in both.
    nav::Navigator &navigator() { return m_navigator; }
    const nav::Navigator &navigator() const { return m_navigator; }
    void openingsFor(const semantic::SemanticWorld &world,
                     semantic::RegionId region,
                     std::vector<semantic::Segment> &out) const;
    // Every region's ways out, gathered in one pass over the relations.
    // Asking region by region costs a pass each, and the passes are over
    // every relation in the map.
    void openingsInto(const semantic::SemanticWorld &world,
                      semantic::RegionId region,
                      std::vector<semantic::Segment> &out) const;
    void gatherOpenings(const semantic::SemanticWorld &world,
                        std::map<semantic::RegionId,
                                 std::vector<semantic::Segment>> &out) const;
    // How many pieces of free space the actor's own body makes of the world.
    size_t places() const { return m_places.size(); }
    size_t splitRegions() const;
    // How many pieces of free space this region has for the current body.
    int piecesOf(semantic::RegionId region) const;
    // How many places in this region the physics layer agreed the body can be.
    int stancesIn(semantic::RegionId region) const;
    // The witness poses themselves. Diagnostic: when the stance list and the
    // free-space components disagree about a region, the poses are the thing
    // to put to the engine.
    const std::vector<semantic::Vec2> *stancesOf(
        semantic::RegionId region) const;

    const std::vector<Transition> &transitions() const
    {
        return m_transitions;
    }
    // Crossings the physics layer verified and this layer then dropped,
    // because it could not say which piece of free space either end of them
    // is in. Every one of these is a way the body can go that the planner
    // will never be offered.
    int orphanedCrossings() const { return m_orphaned; }
    // Why an act's option was or was not given a Place. See m_optionWhy.
    int optionVerdict(semantic::AffordanceId action, uint32_t option) const;
    const std::vector<DroppedCrossing> &droppedCrossings() const
    {
        return m_dropped;
    }
    // The worst single pass through each phase of a derivation, in
    // milliseconds. This runs on the game's own thread, so a phase that
    // takes longer than a tick is a frame the game did not draw.
    double worstStandingMs() const { return m_worstStanding; }
    double worstCrossingsMs() const { return m_worstCrossings; }
    double worstPiecesMs() const { return m_worstPieces; }
    double totalStandingMs() const { return m_totalStanding; }
    double totalCrossingsMs() const { return m_totalCrossings; }
    double totalPiecesMs() const { return m_totalPieces; }
    int derivations() const { return m_derivations; }
    int standingBuilds() const { return m_navigator.builds(); }
    int evaluationsLastUpdate() const { return m_lastEvaluations; }
    int totalEvaluations() const { return m_evaluations; }

    // Navigation over what the bot can actually execute.
    void reachableFrom(semantic::RegionId origin,
                       std::vector<semantic::RegionId> &out) const;
    bool route(semantic::RegionId from, semantic::RegionId to,
               std::vector<semantic::RegionId> &path,
               std::vector<semantic::RelationId> &via) const;
    bool routeCost(semantic::RegionId from, semantic::RegionId to,
                   int &cost) const;
    // What it costs to get to one particular place an action can be taken
    // from. Not the same question as reaching its Region: a Region is one
    // piece of space, and a body with width may not be able to get from one
    // part of it to another.
    bool optionCost(semantic::RegionId from, semantic::AffordanceId action,
                    uint32_t option, int &cost) const;
    // The way there, priced by exactly the search that priced it. The
    // planner asks what an action costs and the executor asks how to get to
    // it; if those are two searches they will disagree, and then the planner
    // chooses something the executor turns straight back down.
    bool optionRoute(semantic::RegionId from, semantic::AffordanceId action,
                     uint32_t option, std::vector<semantic::RegionId> &path,
                     std::vector<semantic::RelationId> &via) const;

    // The executor could not drive something the derivation offered. That
    // is a fact about this actor in this world, so it is recorded here and
    // forgotten when the world changes -- or when it turns out to have been
    // the only way left, which cannot be true of somewhere the body walked
    // into. See forgetRefusals.
    void refuseOption(semantic::AffordanceId action, uint32_t option);
    void refuseRegion(semantic::RegionId region);
    // The body was driven at this crossing and did not get across. Which
    // crossing failed is what the executor actually learned; the place it
    // was ultimately headed for had nothing to do with it, and refusing that
    // instead leaves every other route through the same stuck leg on offer.
    void refuseCrossing(semantic::RelationId relation);
    int refusals() const;
    // Give up on having given up. A refusal is what the executor managed,
    // not what the world allows, and a set of them that says the body is
    // sealed into a place it walked into says something that cannot be
    // true. Whoever finds it has nothing left to do says so here.
    void forgetRefusals();

    // Diagnostics: physically real, but this bot cannot drive it yet.
    int knownButUnexecutable() const;
    bool derived(semantic::RelationId relation, Mode mode) const;
    // Where on a relation's gateway the executable crossing was verified,
    // and where on the far side it left the body.
    bool crossingFor(semantic::RelationId relation, semantic::Vec2 &out,
                     semantic::Vec2 &arrival) const;

private:
    struct Entry
    {
        uint64_t signature = 0;
        uint64_t profileRevision = 0;
        bool valid = false;
        // This crossing was worked out again, so which pieces of free space
        // its ends are in has to be worked out again too. Only this one:
        // that pass costs a visibility test against every node of both
        // regions, and running it over the whole map because one door moved
        // is most of what the bot costs the game.
        bool piecesStale = true;
        // The versions of the two free-space maps the last answer came out
        // of. A map that has not been made again cannot have changed which
        // piece of it a point is in.
        int leftGeneration = -1;
        int enteredGeneration = -1;
        bool possible[kModeCount] = {};
        semantic::Vec2 crossing[kModeCount];
        semantic::Vec2 arrival[kModeCount];
        semantic::Vec2 departure[kModeCount];
        // Which pieces of each side's free space the crossing point touches.
        std::vector<int> leaves[kModeCount];
        std::vector<int> arrives[kModeCount];
        int cost[kModeCount] = {};
        int head[kModeCount] = {};
        int tail[kModeCount] = {};
        semantic::RegionId from = semantic::kNoId;
        semantic::RegionId to = semantic::kNoId;
        bool exists = false;
    };

    // One Region and one piece of its free space.
    struct Place
    {
        semantic::RegionId region = semantic::kNoId;
        int component = 0;
    };

    struct Crossing
    {
        size_t leaves = 0;
        size_t arrives = 0;
    };

    void rebuildIndex() const;
    void refreshPlaces(const semantic::SemanticWorld &world, int radius);
    // Which piece of free space the body is standing in. The places
    // themselves do not move when the body does, so on a tick where nothing
    // else has changed this is the whole of the work.
    void locateActor(const semantic::SemanticWorld &world, int radius);
    // Everything a derivation depends on, in one number: the shape of the
    // world as this model sees it, the size of the body, and what the
    // executor has since refused.
    //
    // Deliberately not the engine's own revision. That moves whenever
    // anything anywhere moves -- a crate settling, a platform rising in a
    // room never visited -- and none of that is a reason to work out again
    // what this model already says.
    uint64_t derivationSignature(const semantic::SemanticWorld &world,
                                 const PhysicsOracle &oracle) const;
    void refreshStanding(const semantic::SemanticWorld &world,
                         const PhysicsOracle &oracle, uint64_t revision);
    const semantic::Vec2 *stanceFor(const semantic::SemanticWorld &world,
                                    semantic::RegionId region,
                                    const semantic::Vec2 &toward,
                                    int radius) const;
    void refreshComponents(const semantic::SemanticWorld &world, int radius);
    void refreshOptions(const semantic::SemanticWorld &world, int radius);
    bool refusedOption(semantic::AffordanceId action, uint32_t option) const;
    bool refusedRegion(semantic::RegionId region) const;
    bool refusedCrossing(semantic::RelationId relation) const;
    size_t placeFor(semantic::RegionId region, int component);
    size_t rootOf(size_t place) const;
    void joinPlaces(size_t left, size_t right);
    // Where the actor is, and any piece of a region, as graph nodes.
    size_t actorPlace() const;
    size_t placeIn(semantic::RegionId region) const;
    size_t startPlace(semantic::RegionId from) const;
    // What one crossing costs to walk, from wherever the body is coming from.
    int legCost(size_t transition, size_t place,
                const semantic::Vec2 &at) const;
    // The one search. Everything that asks whether somewhere can be got to,
    // what it costs, or how to get there asks this, so there is a single
    // answer to be had. `goalPlace` names one piece of free space;
    // `goalRegion` names any piece of a region. Exactly one is given.
    bool findRoute(size_t start, size_t goalPlace,
                   semantic::RegionId goalRegion,
                   std::vector<semantic::RegionId> *path,
                   std::vector<semantic::RelationId> *via, int *cost) const;

    std::vector<Entry> m_entries;
    std::vector<Transition> m_transitions;
    std::vector<Crossing> m_transitionPlaces; // parallel to m_transitions
    uint32_t m_executable = 0;
    int m_evaluations = 0;
    int m_orphaned = 0;
    std::vector<DroppedCrossing> m_dropped;
    double m_worstStanding = 0.0;
    double m_worstCrossings = 0.0;
    double m_worstPieces = 0.0;
    double m_totalStanding = 0.0;
    double m_totalCrossings = 0.0;
    double m_totalPieces = 0.0;
    int m_derivations = 0;
    int m_lastEvaluations = 0;

    // The one reading of where this body can be.
    //
    // There used to be two: one that counted every opening the world offers
    // and one that counted only the crossings this bot had verified. They
    // disagreed -- a room whose only ways out are steps too tall to climb
    // came out of the second one with no free space at all, while the engine
    // was perfectly happy to stand a body in it. Whether a body fits
    // somewhere and whether it can leave by a particular way are two
    // questions, and only the first one is about free space.
    //
    // So the boundary here is what the world actually closes: solid walls,
    // things standing in openings, and the region's own barriers. Crossing
    // to another Place happens through a transition, never by a path finding
    // its own way out through a doorway -- which is what the leaving rule in
    // LocalMap::visible is for.
    // What the body is, kept so free space can be worked out from it.
    ActorProfile m_profile;
    nav::Navigator m_navigator;
    // Per region, the places the physics layer agreed the body can be, in the
    // order the free-space map offered them.
    std::map<semantic::RegionId, std::vector<semantic::Vec2>> m_stances;
    std::vector<Place> m_places;
    std::vector<size_t> m_placeParent;   // pieces joined by a shared crossing
    std::vector<semantic::Segment> m_openings;
    std::vector<int> m_scratch;
    // Per affordance, per option: which piece of free space it is in.
    std::vector<std::vector<size_t>> m_optionPlace;
    // Where each option actually is, so the last leg of a journey to it can
    // be measured to the thing rather than to the middle of its room.
    std::vector<std::vector<semantic::Vec2>> m_optionAt;
    // Where each region is reckoned to be.
    std::map<semantic::RegionId, semantic::Vec2> m_regionAnchor;
    int journeyCost(const std::vector<semantic::RelationId> &via,
                    const semantic::Vec2 &target) const;
    // Why each option got no Place, so a withheld act can be read rather
    // than guessed at. 0 kept, 1 nowhere, 2 no room for the body there,
    // 3 room but no piece of free space this layer can name.
    std::vector<std::vector<uint8_t>> m_optionWhy;
    uint64_t m_domainSignature = 0;
    std::vector<std::pair<semantic::AffordanceId, uint32_t>> m_refusedOptions;
    std::vector<semantic::RegionId> m_refusedRegions;
    // A crossing the executor could not drive, and what the world was
    // saying about it at the time. Re-deriving it is not news; the entry
    // gets worked out again whenever anything near it moves. What lifts the
    // refusal is the world's own account of this crossing changing.
    struct Refused
    {
        semantic::RelationId relation = semantic::kNoId;
        uint64_t signature = 0;
    };
    std::vector<Refused> m_refusedCrossings;
    uint64_t m_standingSignature = 0;
    // Per region, the version of its free-space map the stances were found
    // in. A region whose map has not been made again has not changed shape
    // or changed its ways out, so where a body can stand in it is the same
    // answer as last time -- and finding it again costs an engine query at
    // every candidate.
    std::map<semantic::RegionId, int> m_stanceGeneration;
    uint64_t m_stanceProfile = 0;
    uint64_t m_derived = 0;
    bool m_derivedKnown = false;
    bool m_standingKnown = false;
    int m_mapsAtComponents = -1;
    semantic::RegionId m_actorRegion = semantic::kNoId;
    semantic::Vec2 m_actorAt;
    size_t m_actorPlace = 0;
    bool m_actorPlaceKnown = false;

    mutable std::vector<std::vector<size_t>> m_outgoing;
    mutable bool m_indexStale = true;
};

} // namespace traversal
