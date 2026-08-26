#include "traversal_model.h"
#include <tuple>

#include <algorithm>
#include <limits>
#include <queue>
#include <string>

namespace traversal {

using semantic::Region;
using semantic::RegionId;
using semantic::RelationId;
using semantic::SemanticWorld;
using semantic::SpatialRelation;
using semantic::kNoId;

namespace {

uint64_t mixHash(uint64_t value, uint64_t item)
{
    value ^= item + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
    return value;
}

// Everything about the situation that could change the answer. When this is
// unchanged and the actor is unchanged, the verdict is still good.
uint64_t signatureOf(const SpatialRelation &relation, const Region &from,
                     const Region &to)
{
    uint64_t hash = 1469598103934665603ULL;
    hash = mixHash(hash, uint64_t(uint32_t(relation.verticalStep)));
    hash = mixHash(hash, uint64_t(uint32_t(relation.gap)));
    hash = mixHash(hash, uint64_t(uint32_t(relation.clearance)));
    hash = mixHash(hash, relation.blocked ? 1 : 0);
    hash = mixHash(hash, relation.exists ? 1 : 0);
    hash = mixHash(hash, uint64_t(uint32_t(relation.gateway.from.x)));
    hash = mixHash(hash, uint64_t(uint32_t(relation.gateway.from.y)));
    hash = mixHash(hash, uint64_t(uint32_t(relation.gateway.to.x)));
    hash = mixHash(hash, uint64_t(uint32_t(relation.gateway.to.y)));
    hash = mixHash(hash, from.exists ? 1 : 0);
    hash = mixHash(hash, to.exists ? 1 : 0);
    hash = mixHash(hash, uint64_t(uint32_t(from.support.zAt(0, 0))));
    hash = mixHash(hash, uint64_t(uint32_t(to.support.zAt(0, 0))));
    // Regions are whole spaces now, so their shape is part of the question:
    // the same opening between differently shaped neighbours is a different
    // crossing.
    hash = mixHash(hash, uint64_t(uint32_t(from.interior.x)));
    hash = mixHash(hash, uint64_t(uint32_t(from.interior.y)));
    hash = mixHash(hash, uint64_t(uint32_t(to.interior.x)));
    hash = mixHash(hash, uint64_t(uint32_t(to.interior.y)));
    // Shapes, not just how many corners they have. What stands in a region
    // can move without the count of things changing, and this signature is
    // the whole of what says an answer is still good.
    auto shape = [&hash](const Region &region)
    {
        hash = mixHash(hash, uint64_t(region.footprint.size()));
        if (!region.footprint.empty())
        {
            hash = mixHash(hash,
                uint64_t(uint32_t(region.footprint.front().x)));
            hash = mixHash(hash,
                uint64_t(uint32_t(region.footprint.front().y)));
        }
        hash = mixHash(hash, uint64_t(region.holes.size()));
        for (const semantic::Loop &hole : region.holes)
        {
            hash = mixHash(hash, uint64_t(hole.size()));
            if (!hole.empty())
            {
                hash = mixHash(hash, uint64_t(uint32_t(hole.front().x)));
                hash = mixHash(hash, uint64_t(uint32_t(hole.front().y)));
            }
        }
        hash = mixHash(hash, uint64_t(region.supports.size()));
        hash = mixHash(hash, uint64_t(uint32_t(region.ceiling.zAt(0, 0))));
        hash = mixHash(hash, uint64_t(int(region.clearance)));
        hash = mixHash(hash, region.hazard.harmful ? 1u : 0u);
        // Whether the ground is moving at this instant is deliberately not
        // in here. Nothing this signature guards depends on it any more --
        // the executor declines to step into a moving room, the derivation
        // does not -- and putting a per-tick fact into a test for whether a
        // verdict is stale means every verdict is stale every tick.
    };
    shape(from);
    shape(to);
    return hash;
}

} // namespace

const char *modeName(Mode mode)
{
    switch (mode)
    {
    case Mode::Walk: return "walk";
    case Mode::Crouch: return "crouch";
    case Mode::Jump: return "jump";
    case Mode::Drop: return "drop";
    case Mode::Ride: return "ride";
    }
    return "unknown";
}

void TraversalModel::setExecutableModes(uint32_t mask)
{
    if (mask == m_executable)
        return;
    m_executable = mask;
    // Which modes this bot can drive changes what is selectable. It does not
    // change the world and it does not change what Caleb can physically do,
    // so nothing is re-evaluated -- only relabelled.
    for (Transition &transition : m_transitions)
        transition.executable = transition.possible
            && (m_executable & modeBit(transition.mode)) != 0;
    m_indexStale = true;
}

void TraversalModel::clear()
{
    m_entries.clear();
    m_transitions.clear();
    m_standingKnown = false;
    m_derivedKnown = false;
    m_indexStale = true;
}

void TraversalModel::update(const SemanticWorld &world,
                            const PhysicsOracle &oracle)
{
    // Two things can make a stored verdict wrong: the actor changing, and
    // this crossing changing. The first is versioned; the second is what the
    // signature below is for.
    //
    // The world's own revision is deliberately not in here. It moves when
    // anything anywhere moves -- a door swinging on the far side of the
    // level, a platform rising in a room never visited -- and putting it in
    // the test for one crossing means every crossing in the map is worked
    // out again, with engine queries, on any of those ticks. That is most of
    // what this bot costs the game.
    const uint64_t revision = oracle.profile().revision;
    const std::vector<SpatialRelation> &relations = world.relations();
    if (m_entries.size() < relations.size())
        m_entries.resize(relations.size());
    m_lastEvaluations = 0;
    m_profile = oracle.profile();
    // Nothing this derivation reads has changed, so nothing it says would
    // change either. All that moves every tick is the body, and where the
    // body is standing is a question about the places, not a reason to work
    // them out again. Deriving regardless is most of what the bot costs the
    // game, on every tick of every run.
    const uint64_t derivation = derivationSignature(world, oracle);
    if (m_derivedKnown && derivation == m_derived)
    {
        // Nothing about the world has changed. What an action offers still
        // might have -- its stances are worked out a few at a time -- and
        // that has its own test, so ask it. It is the cheap half.
        refreshOptions(world, oracle.profile().radius);
    refreshSurvival(world, oracle);
    refreshSealing(world, oracle);
        locateActor(world, oracle.profile().radius);
        return;
    }
    m_derived = derivation;
    m_derivedKnown = true;
    ++m_derivations;
    auto began = std::chrono::steady_clock::now();
    auto lap = [&began]()
    {
        const auto now = std::chrono::steady_clock::now();
        const double spent = std::chrono::duration<double, std::milli>(
            now - began).count();
        began = now;
        return spent;
    };
    refreshStanding(world, oracle, derivation);
    { const double spent = lap();
      m_worstStanding = std::max(m_worstStanding, spent);
      m_totalStanding += spent; }

    for (size_t index = 0; index < relations.size(); ++index)
    {
        const SpatialRelation &relation = relations[index];
        Entry &entry = m_entries[index];
        if (relation.id == kNoId)
        {
            entry.valid = false;
            continue;
        }
        const Region *from = world.region(relation.from);
        const Region *to = world.region(relation.to);
        if (!from || !to)
        {
            entry.valid = false;
            entry.exists = false;
            continue;
        }
        const uint64_t signature = signatureOf(relation, *from, *to);
        if (entry.valid && entry.signature == signature
            && entry.profileRevision == revision)
            continue;
        if (entry.valid && entry.signature == signature)
            ++m_staleByBody;
        else
            ++m_staleByCrossing;

        // What the world says about this crossing has changed, so what the
        // executor found out about the old answer is about an answer that no
        // longer stands. Only this one, and only on a real change: entries
        // are worked out again whenever anything anywhere moves, and taking
        // that for news would forgive every stuck leg every few ticks.
        for (size_t seen = 0; seen < m_refusedCrossings.size(); ++seen)
            if (m_refusedCrossings[seen].relation == relation.id
                && m_refusedCrossings[seen].signature != signature)
            {
                m_refusedCrossings.erase(m_refusedCrossings.begin()
                                         + ptrdiff_t(seen));
                break;
            }
        entry.signature = signature;
        entry.profileRevision = revision;
        entry.piecesStale = true;
        entry.valid = true;
        entry.from = relation.from;
        entry.to = relation.to;
        entry.exists = relation.exists && from->exists && to->exists;

        const semantic::Vec2 *startFrom = entry.exists
            ? stanceFor(world, relation.from, relation.gateway.midpoint(),
                        oracle.profile().radius)
            : nullptr;
        for (int mode = 0; mode < kModeCount; ++mode)
        {
            semantic::Vec2 crossing = relation.gateway.midpoint();
            semantic::Vec2 arrival = to->interior;
            semantic::Vec2 departure = from->interior;
            entry.possible[mode] = entry.exists
                && oracle.canTraverse(Mode(mode), relation, *from, *to,
                                      startFrom, crossing, arrival, departure);
            entry.crossing[mode] = crossing;
            entry.arrival[mode] = arrival;
            entry.departure[mode] = departure;
            // What it costs to walk this crossing: in to the opening and out
            // the other side, not the straight line between two regions'
            // middles. A region can be enormous, and its middle is not on
            // the way to anything.
            const semantic::Vec3 gate = { crossing.x, crossing.y, 0 };
            entry.head[mode] = semantic::planarDistance(from->anchor(), gate);
            entry.tail[mode] = semantic::planarDistance(gate, to->anchor());
            entry.cost[mode] = std::max(1,
                entry.head[mode] + entry.tail[mode]);
            ++m_evaluations;
            ++m_lastEvaluations;
        }

        // And the same question again for each configuration of whatever
        // stateful geometry this crossing is made of, if it is made of any.
        //
        // Only the crossing's own geometry: a way is decided by the ground
        // at its two ends, so asking about anything else would be asking the
        // whole world at once, which is the combinatorial explosion this is
        // written to avoid. Everything else stays unconditional.
        entry.conditionedOn = kNoId;
        entry.modesInConfiguration.clear();
        const semantic::GeometryId mover = from->mover != kNoId ? from->mover
                                                                : to->mover;
        const semantic::StatefulGeometry *piece = mover == kNoId
            ? nullptr : world.geometryOf(mover);
        if (entry.exists && piece && piece->configurations.size() > 1)
        {
            entry.conditionedOn = mover;
            entry.modesInConfiguration.assign(
                piece->configurations.size(), 0u);
            for (uint32_t which = 0;
                 which < uint32_t(piece->configurations.size()); ++which)
                for (int mode = 0; mode < kModeCount; ++mode)
                {
                    semantic::Vec2 crossing = relation.gateway.midpoint();
                    semantic::Vec2 arrival = to->interior;
                    semantic::Vec2 departure = from->interior;
                    if (!oracle.canTraverseWith(mover, which, Mode(mode),
                            relation, *from, *to, startFrom, crossing,
                            arrival, departure))
                        continue;
                    entry.modesInConfiguration[which] |=
                        modeBit(Mode(mode));
                }
        }
    }

    { const double spent = lap();
      m_worstCrossings = std::max(m_worstCrossings, spent);
      m_totalCrossings += spent; }
    // What the body can actually go through decides what counts as a wall,
    // and what counts as a wall decides what free space there is. So the
    // verdicts come first, and only then the shape of the space they make.
    refreshPlaces(world, oracle.profile().radius);
    refreshComponents(world, oracle.profile().radius);
    refreshOptions(world, oracle.profile().radius);
    { const double spent = lap();
      m_worstPieces = std::max(m_worstPieces, spent);
      m_totalPieces += spent; }
    if (m_lastEvaluations > 0)
    {
        // Something was re-derived, so anything the executor could not do
        // before deserves another answer.
        m_refusedOptions.clear();
        m_refusedRegions.clear();
    }

    // A crossing point that touches several pieces of one side's free space
    // joins them: it can be stood in, and from it both are reachable.
    for (const Entry &entry : m_entries)
    {
        if (!entry.valid || !entry.exists)
            continue;
        for (int mode = 0; mode < kModeCount; ++mode)
        {
            if (!entry.possible[mode])
                continue;
            for (size_t i = 1; i < entry.leaves[mode].size(); ++i)
                joinPlaces(placeFor(entry.from, entry.leaves[mode][0]),
                           placeFor(entry.from, entry.leaves[mode][i]));
            for (size_t i = 1; i < entry.arrives[mode].size(); ++i)
                joinPlaces(placeFor(entry.to, entry.arrives[mode][0]),
                           placeFor(entry.to, entry.arrives[mode][i]));
        }
    }

    // Which piece of each side's free space a conditional crossing touches.
    //
    // The gateway is where the crossing happens, and it stays where it is
    // when the geometry moves -- only the heights change -- so which piece
    // of the free space as it stands it belongs to is a question that can be
    // asked now, in the configuration the world is actually in.
    {
        std::map<RegionId, std::vector<semantic::Segment>> ways;
        gatherOpenings(world, ways);
        static const std::vector<semantic::Segment> none;
        const int radius = oracle.profile().radius;
        auto pieceAt = [&](RegionId which, const semantic::Vec2 &at) {
            const Region *space = world.region(which);
            if (!space || !space->exists)
                return m_places.size();
            auto found = ways.find(which);
            const nav::LocalMap &map = m_navigator.mapFor(*space,
                found == ways.end() ? none : found->second, radius);
            const int piece = map.componentNear(at);
            if (piece < 0)
                return m_places.size();
            return rootOf(placeFor(which, piece));
        };
        for (size_t index = 0; index < m_entries.size(); ++index)
        {
            Entry &entry = m_entries[index];
            entry.leavesPlace = m_places.size();
            entry.arrivesPlace = m_places.size();
            if (!entry.valid || !entry.exists
                || entry.conditionedOn == semantic::kNoId)
                continue;
            const semantic::SpatialRelation *way =
                world.relation(RelationId(index));
            if (!way || !way->exists)
                continue;
            const semantic::Vec2 mid = way->gateway.midpoint();
            entry.leavesPlace = pieceAt(entry.from, mid);
            entry.arrivesPlace = pieceAt(entry.to, mid);
        }
    }

    m_transitions.clear();
    m_transitionPlaces.clear();
    m_dropped.clear();
    m_orphaned = 0;
    for (size_t index = 0; index < m_entries.size(); ++index)
    {
        const Entry &entry = m_entries[index];
        if (!entry.valid || !entry.exists)
            continue;
        for (int mode = 0; mode < kModeCount; ++mode)
        {
            if (!entry.possible[mode])
                continue;
            // Nowhere the body can be. A gap narrower than the body is not
            // somewhere to go, however plainly the floor carries on through
            // it: the engine will hold the body up with whatever is next to
            // the gap, not with the gap. Walking into one is how a bot ends
            // up somewhere it cannot leave.
            if (stancesIn(entry.from) == 0 || stancesIn(entry.to) == 0)
            {
                m_dropped.push_back({ RelationId(index), mode,
                    int(entry.leaves[mode].size()),
                    int(entry.arrives[mode].size()),
                    stancesIn(entry.from), stancesIn(entry.to),
                    entry.departure[mode], entry.arrival[mode] });
                continue;
            }
            if (entry.leaves[mode].empty() || entry.arrives[mode].empty())
            {
                m_dropped.push_back({ RelationId(index), mode,
                    int(entry.leaves[mode].size()),
                    int(entry.arrives[mode].size()),
                    stancesIn(entry.from), stancesIn(entry.to),
                    entry.departure[mode], entry.arrival[mode] });
                // Both ends are real space this body fits in, and this layer
                // still cannot say which piece of it either end is. That is
                // a way the body can go which the planner will never see.
                ++m_orphaned;
                continue;
            }
            Transition transition;
            transition.relation = RelationId(index);
            transition.from = entry.from;
            transition.to = entry.to;
            transition.mode = Mode(mode);
            transition.crossing = entry.crossing[mode];
            transition.arrival = entry.arrival[mode];
            transition.cost = entry.cost[mode];
            transition.head = entry.head[mode];
            transition.tail = entry.tail[mode];
            transition.possible = true;
            transition.executable = (m_executable & modeBit(Mode(mode))) != 0;
            m_transitions.push_back(transition);
            m_transitionPlaces.push_back(
                { rootOf(placeFor(entry.from, entry.leaves[mode][0])),
                  rootOf(placeFor(entry.to, entry.arrives[mode][0])) });
        }
    }
    m_indexStale = true;
}

void TraversalModel::openingsFor(const SemanticWorld &world, RegionId region,
                                 std::vector<semantic::Segment> &out) const
{
    // Where this region stops being closed in.
    //
    // This is the whole of the answer, and there is deliberately only one of
    // it. An opening is somewhere the world is not wall; whether this body
    // can usefully go through it is a different question, asked of the
    // transitions, and answering it here would mean a step nobody can climb
    // walls off the ground on this side of it too -- ground the engine is
    // perfectly happy to stand a body on, right up to the lip.
    //
    // Asking it here as well is what put two different readings of the same
    // region through one cache: free space was worked out from every opening
    // and routes from a smaller set, so the map was torn down and rebuilt
    // every time the caller changed, and the pieces derived from one reading
    // were used against the other. Twenty-two thousand rebuilds for a hundred
    // and nineteen regions, and a decision that took a sixth of a second.
    // One rule, one map.
    //
    // Nothing escapes through an opening by accident: a route may not cross
    // one at all (see LocalMap::visible), so leaving a Place is something
    // only a transition does.
    openingsInto(world, region, out);
}

// The one rule, for one region: where the engine does not clip this body.
//
// An opening is somewhere the world is not wall *to this body*, and Build
// decides that with cliptestsector: a two-sided wall stops a body when the
// floor behind it is further up than the body steps over, or when there is
// not the height to stand in. Where it stops the body, it is wall, and free
// space has to be eroded against it like any other.
//
// Leaving that out -- counting every adjacency as open, whoever is asking --
// puts free space where the engine will not let a body be. Poses land within
// a hull of a clip line that nothing in the model knows about, routes lead to
// them, and the body drives into the invisible wall and pushes. On AGTST14 it
// pushed for twenty-six thousand ticks, eighty-three units from a switch it
// could see, wedged on a boundary the model called a doorway.
//
// This is directional and it has to be. A step too tall is a wall from below
// and a ledge from above, so it erodes the low room's floor and not the high
// room's -- which is also why the room above a ledge nobody can climb keeps
// its free space: that room's own ways out are its own question.
void TraversalModel::openingsInto(const SemanticWorld &world, RegionId region,
                                  std::vector<semantic::Segment> &out) const
{
    out.clear();
    for (const SpatialRelation &relation : world.relations())
    {
        if (!relation.exists || relation.blocked || relation.id == kNoId)
            continue;
        if (relation.from != region || relation.gateway.width() <= 0)
            continue;
        // A rise this body does not step over is not an opening, and
        // neither is one with nowhere to be on the other side.
        if (-relation.verticalStep > m_profile.stepUp)
            continue;
        if (relation.clearance < m_profile.standHeight)
            continue;
        out.push_back({ relation.gateway.from, relation.gateway.to });
    }
}

// The one rule again, for every region at once: one pass over the relations
// rather than one pass per region. It has to agree with openingsInto exactly
// -- they feed the same cache, and two readings of one region is the bug this
// pair replaced.
void TraversalModel::gatherOpenings(const SemanticWorld &world,
    std::map<RegionId, std::vector<semantic::Segment>> &out) const
{
    out.clear();
    for (const SpatialRelation &relation : world.relations())
    {
        if (!relation.exists || relation.blocked || relation.id == kNoId)
            continue;
        if (relation.gateway.width() <= 0)
            continue;
        if (-relation.verticalStep > m_profile.stepUp)
            continue;
        if (relation.clearance < m_profile.standHeight)
            continue;
        out[relation.from].push_back({ relation.gateway.from,
                                       relation.gateway.to });
    }
}

// Where this body fits, reading every opening the world offers as one. This
// is the only question the physics layer is asked about stances, and it is
// deliberately the permissive reading: a body may stand with its hull over a
// ledge it cannot climb, and refusing to admit that leaves the model unable
// to describe a body that is plainly standing there.
void TraversalModel::refreshStanding(const SemanticWorld &world,
                                     const PhysicsOracle &oracle,
                                     uint64_t revision)
{
    // Nothing it reads has moved, so nothing it says has changed. The world
    // and the body are both versioned, and this depends on exactly those
    // two. Doing the work again costs a walk over every region crossed with
    // every relation plus an engine query at every candidate place, every
    // tick -- which is most of what the bot costs the game.
    if (m_standingKnown && revision == m_standingSignature)
        return;
    m_standingSignature = revision;
    m_standingKnown = true;

    const int radius = oracle.profile().radius;
    // A different body stands in different places; the same body in an
    // unchanged region stands where it stood.
    const bool sameBody = m_stanceProfile == oracle.profile().revision;
    m_stanceProfile = oracle.profile().revision;
    std::map<semantic::RegionId, std::vector<semantic::Vec2>> kept;
    std::map<semantic::RegionId, int> keptAt;
    std::vector<semantic::Vec2> candidates;
    // Every region's ways out. Asked of the one function that answers it.
    //
    // This used to be a third copy of the rule, written out inline, and it
    // was the copy that got missed when the rule changed. Everything else
    // asked for the filtered set and this asked for the unfiltered one, so
    // every region's free space was built one way here and the other way
    // everywhere else -- and since they share a cache keyed on the openings,
    // each of the hundred and nineteen regions was thrown away and rebuilt
    // twice on every derivation. Twenty thousand rebuilds, and a decision
    // that took a fifth of a second.
    std::map<RegionId, std::vector<semantic::Segment>> ways;
    gatherOpenings(world, ways);
    static const std::vector<semantic::Segment> none;
    for (const Region &region : world.regions())
    {
        if (!region.exists || region.id == kNoId)
            continue;
        auto found = ways.find(region.id);
        const nav::LocalMap &map = m_navigator.mapFor(region,
            found == ways.end() ? none : found->second, radius);
        const int generation = m_navigator.generation(region.id);
        auto before = m_stanceGeneration.find(region.id);
        auto standing = m_stances.find(region.id);
        if (sameBody && before != m_stanceGeneration.end()
            && before->second == generation && standing != m_stances.end())
        {
            kept[region.id] = standing->second;
            keptAt[region.id] = generation;
            continue;
        }
        // Every place the shape offers, plus the point it calls its middle.
        // Which of them the world actually holds a body up at is the next
        // question, and it is not one a flat map can answer.
        candidates = map.places();
        candidates.push_back(region.interior);
        std::vector<semantic::Vec2> &accepted = kept[region.id];
        keptAt[region.id] = generation;
        for (const semantic::Vec2 &candidate : candidates)
            if (oracle.canStand(region, candidate))
                accepted.push_back(candidate);
    }
    m_stances.swap(kept);
    m_stanceGeneration.swap(keptAt);
}

// The place in this region the body can be that is nearest what it is trying
// to get to, and from which getting there is a straight walk. Both halves
// matter: a stance the world holds the body up at is no use if reaching the
// opening from it means going through a wall.
const semantic::Vec2 *TraversalModel::stanceFor(const SemanticWorld &world,
                                                RegionId region,
                                                const semantic::Vec2 &toward,
                                                int radius) const
{
    auto found = m_stances.find(region);
    if (found == m_stances.end() || found->second.empty())
        return nullptr;
    const Region *shape = world.region(region);
    if (!shape || !shape->exists)
        return nullptr;
    std::vector<semantic::Segment> openings;
    openingsInto(world, region, openings);
    const nav::LocalMap &map =
        const_cast<nav::Navigator &>(m_navigator).mapFor(*shape, openings,
                                                        radius);
    const semantic::Vec2 *best = nullptr;
    const semantic::Vec2 *closest = nullptr;
    int64_t nearest = 0;
    int64_t any = 0;
    for (const semantic::Vec2 &stance : found->second)
    {
        const int64_t dx = int64_t(stance.x) - toward.x;
        const int64_t dy = int64_t(stance.y) - toward.y;
        const int64_t distance = dx * dx + dy * dy;
        if (!closest || distance < any)
        {
            closest = &stance;
            any = distance;
        }
        if (best && distance >= nearest)
            continue;
        if (!map.clearBetween(toward, stance))
            continue;
        best = &stance;
        nearest = distance;
    }
    // Where nothing has a clear line to the opening, the nearest place the
    // body can actually be is still a better starting point than one made up
    // from the opening's geometry: whether the walk from it gets there is
    // then the engine's answer rather than this layer's assumption.
    return best ? best : closest;
}

// Which relations this bot can actually go through. Everything else is a
// wall, whatever the world says is next to what.
// Which piece of each side's free space every crossing can be reached from.
// A gateway the body cannot get to from where it is standing is not a way
// out for that body, however plainly the two spaces touch.
void TraversalModel::refreshComponents(const SemanticWorld &world, int radius)
{
    // Every free-space map was thrown away and made again, so every answer
    // that came from one is stale. Otherwise only the crossings that were
    // themselves worked out again are.
    const bool anyMaps = m_mapsAtComponents != m_navigator.builds();
    if (m_lastEvaluations == 0 && !anyMaps)
        return;
    m_mapsAtComponents = m_navigator.builds();
    std::map<RegionId, std::vector<semantic::Segment>> ways;
    gatherOpenings(world, ways);
    static const std::vector<semantic::Segment> none;
    for (size_t index = 0; index < m_entries.size(); ++index)
    {
        Entry &entry = m_entries[index];
        if (!entry.valid || !entry.exists)
            continue;
        const Region *from = world.region(entry.from);
        const Region *to = world.region(entry.to);
        if (!from || !to)
            continue;
        // Either this crossing was worked out again, or one of the two maps
        // its ends were placed in was made again. Anything else and the last
        // answer still stands -- and testing it again costs a visibility
        // test against every node of both regions.
        const int left = m_navigator.generation(entry.from);
        const int entered = m_navigator.generation(entry.to);
        if (!entry.piecesStale && left == entry.leftGeneration
            && entered == entry.enteredGeneration)
            continue;
        entry.piecesStale = false;
        entry.leftGeneration = left;
        entry.enteredGeneration = entered;
        auto leaving = ways.find(entry.from);
        auto arriving = ways.find(entry.to);
        const std::vector<semantic::Segment> &out =
            leaving == ways.end() ? none : leaving->second;
        const std::vector<semantic::Segment> &in =
            arriving == ways.end() ? none : arriving->second;
        for (int mode = 0; mode < kModeCount; ++mode)
        {
            entry.leaves[mode].clear();
            entry.arrives[mode].clear();
            if (!entry.possible[mode])
                continue;
            // Which piece of free space each end of this crossing is in,
            // asked at the two places a body actually was: the stance it set
            // off from and the pose the engine left it in. Not at the point
            // on the opening itself, which lies on a boundary where the
            // eroded free space may have nothing at all.
            m_navigator.mapFor(*from, out, radius)
                .componentsAt(entry.departure[mode], entry.leaves[mode]);
            m_navigator.mapFor(*to, in, radius)
                .componentsAt(entry.arrival[mode], entry.arrives[mode]);
            // A crossing the physics agreed to, whose end the free space
            // cannot label, is a way out the planner will never be offered.
            //
            // It happens where the body ends up hard against something --
            // which at a doorway is most of the time, and after a wall has
            // slid past it is nearly always. The nearest place the body
            // could stand answers for it, the same way it answers for the
            // actor's own position: "I cannot see a node from here" is not
            // "there is no space here".
            if (entry.leaves[mode].empty())
            {
                const int piece = m_navigator.mapFor(*from, out, radius)
                    .componentNear(entry.departure[mode]);
                if (piece >= 0)
                    entry.leaves[mode].push_back(piece);
            }
            if (entry.arrives[mode].empty())
            {
                const int piece = m_navigator.mapFor(*to, in, radius)
                    .componentNear(entry.arrival[mode]);
                if (piece >= 0)
                    entry.arrives[mode].push_back(piece);
            }
        }
    }
}

// Where each place an action can be taken from actually is, in free space.
void TraversalModel::refreshOptions(const SemanticWorld &world, int radius)
{
    uint64_t signature = 1469598103934665603ULL;
    for (const semantic::Affordance &affordance : world.affordances())
        for (const semantic::ExecutionOption &option : affordance.domain)
        {
            signature = mixHash(signature, uint64_t(uint32_t(option.at.x)));
            signature = mixHash(signature, uint64_t(uint32_t(option.at.y)));
            signature = mixHash(signature, uint64_t(option.region));
        }
    if (signature == m_domainSignature
        && m_mapsAtComponents == m_navigator.builds()
        && m_optionPlace.size() == world.affordances().size())
        return;
    m_domainSignature = signature;
    // Kept, not cleared. One action's stances are worked out per tick, so
    // this runs most ticks -- and rebuilding the placements of all of them
    // every time is most of what the bot costs on a large level, where there
    // are hundreds of stances and each one asks a region's free space where
    // it falls. Only the action whose stances actually moved is placed again.
    const size_t count = world.affordances().size();
    m_optionPlace.resize(count);
    m_optionWhy.resize(count);
    m_optionAt.resize(count);
    m_optionSignature.resize(count, 0);
    const bool mapsMoved = m_mapsAtComponents != m_navigator.builds();
    // Every region's ways out, once. Working them out inside the loop below
    // walks all five hundred and sixty-eight relations for every stance of
    // every action -- seven hundred of them once each action kept all the
    // stances the engine accepts rather than one -- which is four hundred
    // thousand relation visits to answer a question the world only has one
    // answer to.
    std::map<RegionId, std::vector<semantic::Segment>> ways;
    gatherOpenings(world, ways);
    static const std::vector<semantic::Segment> none;
    for (const semantic::Affordance &affordance : world.affordances())
    {
        if (affordance.id == kNoId
            || size_t(affordance.id) >= m_optionPlace.size())
            continue;
        // What this one's stances are, on their own.
        uint64_t mine = 1469598103934665603ULL;
        for (const semantic::ExecutionOption &option : affordance.domain)
        {
            mine = mixHash(mine, uint64_t(uint32_t(option.at.x)));
            mine = mixHash(mine, uint64_t(uint32_t(option.at.y)));
            mine = mixHash(mine, uint64_t(option.region));
        }
        if (!mapsMoved && mine == m_optionSignature[size_t(affordance.id)]
            && m_optionPlace[size_t(affordance.id)].size()
                   == affordance.domain.size())
            continue;
        m_optionSignature[size_t(affordance.id)] = mine;
        std::vector<size_t> &places = m_optionPlace[size_t(affordance.id)];
        std::vector<uint8_t> &why = m_optionWhy[size_t(affordance.id)];
        std::vector<semantic::Vec2> &at = m_optionAt[size_t(affordance.id)];
        places.assign(affordance.domain.size(), m_places.size());
        why.assign(affordance.domain.size(), 1);
        at.assign(affordance.domain.size(), semantic::Vec2{ 0, 0 });
        for (size_t option = 0; option < affordance.domain.size(); ++option)
            at[option] = { affordance.domain[option].at.x,
                           affordance.domain[option].at.y };
        for (size_t option = 0; option < affordance.domain.size(); ++option)
        {
            const semantic::ExecutionOption &where = affordance.domain[option];
            const Region *region = world.region(where.region);
            if (!region || !region->exists)
                continue;
            auto found = ways.find(where.region);
            const nav::LocalMap &map = m_navigator.mapFor(*region,
                found == ways.end() ? none : found->second, radius);
            // A pose an act works from is only an option if the body can be
            // there. The engine will tell you a switch is in reach of a spot
            // and say nothing about whether it would ever leave a body
            // standing on it -- reach is a ray, and standing is a hull. A
            // pose closer to a wall than the body is wide is not somewhere
            // to go; driving at one is walking into the wall.
            if (!map.free({ where.at.x, where.at.y }))
            {
                why[option] = 2;
                continue;
            }
            map.componentsAt({ where.at.x, where.at.y }, m_scratch);
            if (m_scratch.empty())
            {
                // Somewhere the body fits but nothing is a clear stride
                // away: hard against the wall the switch is on, which is
                // where a switch is pressed from. Throwing the stance away
                // for that is throwing away the act.
                //
                // On AGTST8 it threw away every stance for calling the
                // second lift from the ledge it serves, so the only way left
                // to call it was from the floor below -- and the plan came
                // out as "walk back off the ledge, drop into the pit, press
                // it there, and ride all the way round again".
                const int piece =
                    map.componentNear({ where.at.x, where.at.y });
                if (piece < 0)
                {
                    why[option] = 3;
                    continue;
                }
                why[option] = 0;
                places[option] = rootOf(placeFor(where.region, piece));
                continue;
            }
            why[option] = 0;
            places[option] = rootOf(placeFor(where.region, m_scratch[0]));
        }
        for (size_t option = 0; option < affordance.domain.size(); ++option)
            m_sealedRegionOf[std::make_pair(size_t(affordance.id), option)] =
                affordance.domain[option].region;
    }
}

namespace {
constexpr size_t kAnyPlace = size_t(-1);
}

// Whether an act done from a stance leaves the body anywhere to be.
//
// The act commands geometry; the geometry goes somewhere; the stance is a
// point in the world the geometry is going through. So the question is asked
// the only way it can be answered -- by putting the geometry in each
// configuration it could arrive in and asking the engine whether the body
// still fits where it is standing. That is the same machinery that answers
// "is this crossing walkable in that configuration", used for the actor
// rather than for a doorway.
//
// Where the configurations say nothing -- which today is everything that
// moves in the plane -- there is one thing still known: where the geometry
// is now. A stance whose hull is inside something about to move, with no
// account of where it moves to, is not a stance to wait on. That is not a
// rule about doors; it is a refusal to stand inside an unanswered question.
void TraversalModel::refreshSurvival(const SemanticWorld &world,
                                     const PhysicsOracle &oracle)
{
    const size_t count = world.affordances().size();
    m_optionSurvival.resize(count);
    m_optionEscape.resize(count);
    m_survivalSignature.resize(count, 0);
    const int radius = oracle.profile().radius;
    // The spaces each piece of stateful geometry currently occupies.
    std::map<semantic::GeometryId, std::vector<const Region *>> occupies;
    for (const Region &region : world.regions())
        if (region.exists && region.mover != semantic::kNoId)
            occupies[region.mover].push_back(&region);

    struct Ask
    {
        size_t thing = 0;
        size_t option = 0;
        size_t candidate = 0;
    };
    std::map<semantic::GeometryId, std::vector<Ask>> asking;
    std::map<semantic::GeometryId,
             std::vector<PhysicsOracle::Stance>> places;
    std::map<size_t, std::vector<semantic::GeometryId>> commanded;

    // Which acts need answering again.
    //
    // Nothing about this changes while an act's stances stay where they are
    // and it works the same things: the travel it would set off is fixed by
    // the world, and the answer is about those two together. So it is worked
    // out for one act when that act moves and not otherwise -- it is by a
    // long way the most expensive question this layer asks, and asking it
    // every tick was eighty milliseconds a tick on a level full of movers,
    // against a budget of thirty-three.
    for (const semantic::Affordance &thing : world.affordances())
    {
        if (thing.id == kNoId || size_t(thing.id) >= count)
            continue;
        const size_t slot = size_t(thing.id);
        std::vector<semantic::GeometryId> worked = thing.commands;
        for (semantic::GeometryId also : thing.moves)
        {
            bool known = false;
            for (semantic::GeometryId already : worked)
                known = known || already == also;
            if (!known)
                worked.push_back(also);
        }
        uint64_t mine = 1469598103934665603ULL;
        for (const semantic::ExecutionOption &option : thing.domain)
        {
            mine = mixHash(mine, uint64_t(uint32_t(option.at.x)));
            mine = mixHash(mine, uint64_t(uint32_t(option.at.y)));
            mine = mixHash(mine, uint64_t(option.region));
        }
        for (semantic::GeometryId which : worked)
            mine = mixHash(mine, uint64_t(which) + 1);
        if (mine == m_survivalSignature[slot]
            && m_optionSurvival[slot].size() == thing.domain.size())
            continue;
        m_survivalSignature[slot] = mine;
        m_optionSurvival[slot].assign(thing.domain.size(),
                                      uint8_t(Survival::Remain));
        m_optionEscape[slot].assign(thing.domain.size(),
                                    semantic::Vec2{ 0, 0 });
        if (worked.empty())
            continue;   // it moves nothing: there is nothing to survive
        commanded[slot] = worked;

        for (size_t option = 0; option < thing.domain.size(); ++option)
        {
            const semantic::ExecutionOption &where = thing.domain[option];
            const Region *standing = world.region(where.region);
            if (!standing || !standing->exists)
                continue;
            const semantic::Vec2 at = { where.at.x, where.at.y };
            Survival worst = Survival::Remain;
            for (semantic::GeometryId which : worked)
            {
                const semantic::StatefulGeometry *piece =
                    world.geometryOf(which);
                if (!piece)
                    continue;
                if (!piece->configurationsDiffer)
                {
                    // Nothing describes where this goes. Standing clear of it
                    // is all that can be said for a stance; standing in it
                    // cannot be vouched for at all.
                    bool inside = false;
                    auto found = occupies.find(which);
                    if (found != occupies.end())
                        for (const Region *space : found->second)
                            inside = inside
                                || semantic::touches(space->shape(), at,
                                                     radius);
                    const Survival said = inside ? Survival::Unsafe
                                                 : Survival::Unknown;
                    if (int(said) > int(worst))
                        worst = said;
                    continue;
                }
                Ask ask;
                ask.thing = slot;
                ask.option = option;
                asking[which].push_back(ask);
                places[which].push_back({ standing, at });
            }
            m_optionSurvival[slot][option] = uint8_t(worst);
        }
    }
    if (asking.empty())
        return;

    // Whether the body can be where it is standing, all the way through what
    // the act sets off. Each configuration is posed once and every place
    // wanted from it is asked while it is there.
    const uint32_t kPoses = 5;
    auto sweep = [&](const std::map<semantic::GeometryId,
                                    std::vector<Ask>> &questions,
                     std::map<semantic::GeometryId,
                              std::vector<PhysicsOracle::Stance>> &where,
                     std::map<std::tuple<size_t, size_t, size_t>, bool> &into) {
        for (const auto &entry : questions)
        {
            std::vector<char> answer;
            std::vector<char> swept;
            for (uint32_t c = 0; c < kPoses; ++c)
            {
                oracle.canStandThrough(entry.first, c, kPoses,
                                       where[entry.first], answer);
                // And whether the thing itself arrives at this place on the
                // way. Somewhere the body can stand and the geometry comes
                // to is not somewhere to be: it is carried off, which is
                // riding when the place is the thing's own and being run
                // over when it is not.
                oracle.sweptThrough(entry.first, c, kPoses,
                                    where[entry.first], swept);
                for (size_t i = 0; i < entry.second.size()
                         && i < answer.size(); ++i)
                {
                    bool safe = answer[i] != 0;
                    if (i < swept.size() && swept[i]
                        && !(where[entry.first][i].region
                             && where[entry.first][i].region->mover
                                    == entry.first))
                        safe = false;
                    const Ask &ask = entry.second[i];
                    const auto key = std::make_tuple(ask.thing, ask.option,
                                                     ask.candidate);
                    auto found = into.find(key);
                    if (found == into.end())
                        into[key] = safe;
                    else
                        found->second = found->second && safe;
                }
            }
        }
    };

    std::map<std::tuple<size_t, size_t, size_t>, bool> stays;
    sweep(asking, places, stays);


    // Only where staying is not an answer is there any point asking where
    // else the body could be. Working that out means a visibility sweep of
    // the region for every candidate, so it is not done for stances that
    // never needed it -- which is nearly all of them.
    std::map<RegionId, std::vector<semantic::Segment>> ways;
    std::map<semantic::GeometryId, std::vector<Ask>> running;
    std::map<semantic::GeometryId,
             std::vector<PhysicsOracle::Stance>> runningTo;
    std::map<std::pair<size_t, size_t>, std::vector<semantic::Vec2>> options;
    bool gathered = false;
    static const std::vector<semantic::Segment> none;
    for (const auto &verdict : stays)
    {
        if (verdict.second)
            continue;
        const size_t slot = std::get<0>(verdict.first);
        const size_t option = std::get<1>(verdict.first);
        const semantic::Affordance *thing =
            world.affordance(semantic::AffordanceId(slot));
        if (!thing || option >= thing->domain.size())
            continue;
        const semantic::ExecutionOption &where = thing->domain[option];
        const Region *standing = world.region(where.region);
        if (!standing || !standing->exists)
            continue;
        if (!gathered)
        {
            gatherOpenings(world, ways);
            gathered = true;
        }
        const semantic::Vec2 at = { where.at.x, where.at.y };
        auto opening = ways.find(where.region);
        const nav::LocalMap &map = m_navigator.mapFor(*standing,
            opening == ways.end() ? none : opening->second, radius);
        std::vector<semantic::Vec2> reachable;
        map.visibleFrom(at, reachable);
        std::vector<semantic::Vec2> &candidates =
            options[std::make_pair(slot, option)];
        const size_t kMostConsidered = 6;
        for (const semantic::Vec2 &spot : reachable)
        {
            candidates.push_back(spot);
            if (candidates.size() >= kMostConsidered)
                break;
        }
        auto worked = commanded.find(slot);
        if (worked == commanded.end())
            continue;
        for (semantic::GeometryId which : worked->second)
        {
            const semantic::StatefulGeometry *piece =
                world.geometryOf(which);
            if (!piece || !piece->configurationsDiffer)
                continue;
            for (size_t c = 0; c < candidates.size(); ++c)
            {
                Ask run;
                run.thing = slot;
                run.option = option;
                run.candidate = c + 1;
                running[which].push_back(run);
                runningTo[which].push_back({ standing, candidates[c] });
            }
        }
    }
    std::map<std::tuple<size_t, size_t, size_t>, bool> reaches;
    if (!running.empty())
        sweep(running, runningTo, reaches);

    // Somewhere the body cannot be throughout is somewhere it has to leave,
    // and leaving is only an answer if there is somewhere to go.
    for (const auto &verdict : stays)
    {
        if (verdict.second)
            continue;
        const size_t slot = std::get<0>(verdict.first);
        const size_t option = std::get<1>(verdict.first);
        if (slot >= m_optionSurvival.size()
            || option >= m_optionSurvival[slot].size())
            continue;
        Survival said = Survival::Unsafe;
        auto candidates = options.find(std::make_pair(slot, option));
        if (candidates != options.end())
            for (size_t c = 0; c < candidates->second.size(); ++c)
            {
                auto found = reaches.find(
                    std::make_tuple(slot, option, c + 1));
                if (found == reaches.end() || !found->second)
                    continue;
                m_optionEscape[slot][option] = candidates->second[c];
                said = Survival::Escape;
                break;
            }
        if (int(said) > int(m_optionSurvival[slot][option]))
            m_optionSurvival[slot][option] = uint8_t(said);
    }
}

// Would doing this shut the body into the room it is done from?
//
// Separate from whether the body survives, because it does survive: it is
// alive and it is sealed in. On AGTST18 that is a wall sliding across the
// only doorway of the room the switch is in, worked by a switch that only
// works once.
//
// The derived table cannot see it. A crossing is conditioned on the ground
// at its two ends, and the wall that shuts this one belongs to neither room
// -- so the engine is asked directly, with everything the act works held
// where the act would leave it.
//
// On its own cache, and not the one the stance verdicts use. Where a stance
// is depends on the stance; whether it seals depends on where everything is
// resting right now, so this is worked out again when a resting position
// changes and not otherwise. Never while anything is still travelling: the
// question has no answer half way through.
void TraversalModel::refreshSealing(const SemanticWorld &world,
                                    const PhysicsOracle &oracle)
{
    uint64_t resting = 1469598103934665603ULL;
    for (const semantic::StatefulGeometry &piece : world.geometry())
    {
        if (piece.moving)
            return;   // nothing to be said until it stops
        resting = mixHash(resting, uint64_t(piece.id) + 1);
        resting = mixHash(resting, uint64_t(piece.state) + 1);
    }
    if (resting == m_sealingSignature && !m_sealed.empty())
        return;
    m_sealingSignature = resting;
    m_sealed.clear();
    m_sealingLooked = 0;
    m_sealingSkipped = 0;
    m_sealingAudit.clear();
    for (const semantic::Affordance &thing : world.affordances())
    {
        if (thing.id == kNoId || !thing.exists || thing.commands.empty())
            continue;
        std::vector<std::pair<semantic::GeometryId, uint32_t>> after;
        for (semantic::GeometryId which : thing.commands)
        {
            const semantic::StatefulGeometry *piece = world.geometryOf(which);
            if (!piece || piece->configurations.size() < 2
                || !piece->configurationsDiffer)
                continue;
            after.push_back({ which, piece->state == 0 ? 1u : 0u });
        }
        if (after.empty())
            continue;
        std::set<RegionId> asked;
        for (const semantic::ExecutionOption &where : thing.domain)
        {
            if (!asked.insert(where.region).second)
                continue;
            const Region *standing = world.region(where.region);
            if (!standing || !standing->exists)
                continue;
            std::vector<PhysicsOracle::WayOut> ways;
            for (const semantic::SpatialRelation &way : world.relations())
            {
                if (!way.exists || way.id == kNoId || way.from != where.region
                    || way.blocked || way.gateway.width() <= 0)
                    continue;
                ways.push_back({ &way, world.region(way.to) });
            }
            if (ways.empty())
            {
                ++m_sealingSkipped;
                continue;
            }
            const semantic::Vec2 at = { where.at.x, where.at.y };
            // Only where there is a way out now. An act is not to blame for
            // a room that was already shut.
            ++m_sealingLooked;
            Sealing said;
            said.thing = size_t(thing.id);
            said.region = where.region;
            said.ways = int(ways.size());
            said.openNow = oracle.anyWayOut({}, *standing, at, ways);
            said.openAfter = said.openNow
                && oracle.anyWayOut(after, *standing, at, ways);
            m_sealingAudit.push_back(said);
            if (!said.openNow)
            {
                ++m_sealingSkipped;
                continue;
            }
            if (!said.openAfter)
                m_sealed.insert({ size_t(thing.id), where.region });
        }
    }
}

bool TraversalModel::wouldSealIn(semantic::AffordanceId action,
                                 uint32_t option) const
{
    if (size_t(action) >= m_optionAt.size())
        return false;
    const semantic::Region *nothing = nullptr;
    (void)nothing;
    auto found = m_sealedRegionOf.find(
        std::make_pair(size_t(action), size_t(option)));
    if (found == m_sealedRegionOf.end())
        return false;
    return m_sealed.count({ size_t(action), found->second }) != 0;
}

TraversalModel::Survival TraversalModel::survivalOf(
    semantic::AffordanceId action, uint32_t option) const
{
    if (size_t(action) >= m_optionSurvival.size()
        || size_t(option) >= m_optionSurvival[size_t(action)].size())
        return Survival::Remain;
    return Survival(m_optionSurvival[size_t(action)][size_t(option)]);
}

bool TraversalModel::escapeFrom(semantic::AffordanceId action,
                                uint32_t option, semantic::Vec2 &out) const
{
    if (survivalOf(action, option) != Survival::Escape)
        return false;
    out = m_optionEscape[size_t(action)][size_t(option)];
    return true;
}

// Where the piece of free space an action is taken from is, if the model has
// one for it.
int TraversalModel::optionVerdict(semantic::AffordanceId action,
                                  uint32_t option) const
{
    if (size_t(action) >= m_optionWhy.size()
        || size_t(option) >= m_optionWhy[size_t(action)].size())
        return -1;
    return m_optionWhy[size_t(action)][size_t(option)];
}

bool TraversalModel::optionCost(RegionId from, semantic::AffordanceId action,
                                uint32_t option, int &cost) const
{
    // An option the executor has been unable to drive is not offered again
    // until something changes. This is planner policy and not topology --
    // reachableFrom and findRoute know nothing about it -- and everything
    // that asks "can this act be reached" has to ask the same way, or the
    // planner proposes what the router has quietly banned and the two
    // disagree for ever.
    if (refusedOption(action, option))
        return false;
    // Nor one with no survivable trajectory through what the act sets off.
    // This is not a cost and it is not a preference: there is no journey to
    // price to a place the body cannot come back from.
    if (survivalOf(action, option) == Survival::Unsafe)
        return false;
    // Nor one that would shut the body into the room it is done from.
    if (wouldSealIn(action, option))
        return false;
    std::vector<RegionId> path;
    std::vector<RelationId> via;
    if (size_t(action) >= m_optionPlace.size()
        || size_t(option) >= m_optionPlace[size_t(action)].size())
        return false;
    if (m_indexStale)
        rebuildIndex();
    if (!findRoute(startPlace(from), m_optionPlace[size_t(action)][option],
                   kNoId, &path, &via, nullptr))
        return false;
    cost = journeyCost(via, m_optionAt[size_t(action)][size_t(option)]);
    return true;
}

bool TraversalModel::optionRoute(RegionId from, semantic::AffordanceId action,
                                 uint32_t option,
                                 std::vector<RegionId> &path,
                                 std::vector<RelationId> &via,
                                 std::vector<size_t> *through) const
{
    path.clear();
    via.clear();
    if (through)
        through->clear();
    if (size_t(action) >= m_optionPlace.size()
        || size_t(option) >= m_optionPlace[size_t(action)].size())
        return false;
    if (m_indexStale)
        rebuildIndex();
    return findRoute(startPlace(from), m_optionPlace[size_t(action)][option],
                     kNoId, &path, &via, nullptr, through);
}

void TraversalModel::refuseOption(semantic::AffordanceId action,
                                  uint32_t option)
{
    if (refusedOption(action, option))
        return;
    m_refusedOptions.push_back({ action, option });
}

void TraversalModel::refuseRegion(RegionId region)
{
    if (!refusedRegion(region))
        m_refusedRegions.push_back(region);
}

bool TraversalModel::refusedOption(semantic::AffordanceId action,
                                   uint32_t option) const
{
    for (const auto &refused : m_refusedOptions)
        if (refused.first == action && refused.second == option)
            return true;
    return false;
}

void TraversalModel::refuseCrossing(semantic::RelationId relation)
{
    if (relation == kNoId || refusedCrossing(relation))
        return;
    Refused refused;
    refused.relation = relation;
    if (size_t(relation) < m_entries.size())
        refused.signature = m_entries[size_t(relation)].signature;
    m_refusedCrossings.push_back(refused);
}

bool TraversalModel::refusedCrossing(semantic::RelationId relation) const
{
    for (const Refused &refused : m_refusedCrossings)
        if (refused.relation == relation)
            return true;
    return false;
}

bool TraversalModel::refusedRegion(RegionId region) const
{
    for (RegionId refused : m_refusedRegions)
        if (refused == region)
            return true;
    return false;
}

void TraversalModel::forgetRefusals()
{
    m_refusedOptions.clear();
    m_refusedRegions.clear();
    m_refusedCrossings.clear();
}

int TraversalModel::refusals() const
{
    return int(m_refusedOptions.size() + m_refusedRegions.size()
               + m_refusedCrossings.size());
}

// Every live region's free space, worked out once for the current body.
void TraversalModel::refreshPlaces(const SemanticWorld &world, int radius)
{
    m_places.clear();
    m_placeParent.clear();
    m_actorPlaceKnown = false;
    m_actorRegion = world.actor().region;
    m_actorAt = { world.actor().position.x, world.actor().position.y };
    // Where each region is reckoned to be, kept so a journey's cost can be
    // asked for later without the world in hand.
    m_regionAnchor.clear();
    for (const Region &region : world.regions())
        if (region.exists && region.id != kNoId)
        {
            const semantic::Vec3 anchor = region.anchor();
            m_regionAnchor[region.id] = { anchor.x, anchor.y };
        }
    std::map<RegionId, std::vector<semantic::Segment>> ways;
    gatherOpenings(world, ways);
    static const std::vector<semantic::Segment> none;
    for (const Region &region : world.regions())
    {
        if (!region.exists || region.id == kNoId)
            continue;
        auto standing = m_stances.find(region.id);
        if (standing == m_stances.end() || standing->second.empty())
            continue;  // nowhere in it the world holds this body up
        auto found = ways.find(region.id);
        const nav::LocalMap &map = m_navigator.mapFor(region,
            found == ways.end() ? none : found->second, radius);
        for (int component = 0; component < map.components(); ++component)
            placeFor(region.id, component);
    }
    locateActor(world, radius);
}

void TraversalModel::locateActor(const SemanticWorld &world, int radius)
{
    m_actorPlaceKnown = false;
    m_actorRegion = world.actor().region;
    m_actorAt = { world.actor().position.x, world.actor().position.y };
    if (m_actorRegion == kNoId)
        return;
    const Region *here = world.region(m_actorRegion);
    if (!here || !here->exists)
        return;
    openingsFor(world, m_actorRegion, m_openings);
    const nav::LocalMap &map = m_navigator.mapFor(*here, m_openings, radius);
    map.componentsAt(m_actorAt, m_scratch);
    if (m_scratch.empty())
    {
        // Standing somewhere the free space does not describe -- on a
        // boundary, or hard against a wall. The nearest place the body could
        // stand says which piece it is in.
        //
        // This used to join every piece of the region into one instead, on
        // the grounds that saying nothing was safer than saying something
        // wrong. It is not: joining them says the body can walk between
        // them, which on AGTST8 is the difference between the half of the
        // mid ledge that reaches the second lift and the half that does not.
        // The route came out as "step off the first lift and board the
        // second", the executor found no way across the region it had just
        // been routed through, and the run went round again.
        const int piece = map.componentNear(m_actorAt);
        if (piece >= 0)
        {
            m_actorPlace = rootOf(placeFor(m_actorRegion, piece));
            m_actorPlaceKnown = true;
        }
        return;
    }
    for (size_t i = 1; i < m_scratch.size(); ++i)
        joinPlaces(placeFor(m_actorRegion, m_scratch[0]),
                   placeFor(m_actorRegion, m_scratch[i]));
    m_actorPlace = rootOf(placeFor(m_actorRegion, m_scratch[0]));
    m_actorPlaceKnown = true;
}

uint64_t TraversalModel::derivationSignature(const SemanticWorld &world,
                                             const PhysicsOracle &oracle) const
{
    uint64_t hash = mixHash(oracle.profile().revision,
                            uint64_t(m_executable));
    for (const Region &region : world.regions())
    {
        if (!region.exists || region.id == kNoId)
            continue;
        hash = mixHash(hash, uint64_t(region.id));
        hash = mixHash(hash, uint64_t(region.footprint.size()));
        hash = mixHash(hash, uint64_t(region.holes.size()));
        hash = mixHash(hash, uint64_t(uint32_t(region.support.zAt(0, 0))));
        hash = mixHash(hash, uint64_t(uint32_t(region.interior.x)));
        hash = mixHash(hash, uint64_t(uint32_t(region.interior.y)));
        hash = mixHash(hash, uint64_t(int(region.clearance)));
    }
    for (const SpatialRelation &relation : world.relations())
    {
        if (!relation.exists || relation.id == kNoId)
            continue;
        hash = mixHash(hash, uint64_t(relation.id));
        hash = mixHash(hash, relation.blocked ? 1u : 0u);
        hash = mixHash(hash, uint64_t(uint32_t(relation.verticalStep)));
        hash = mixHash(hash, uint64_t(uint32_t(relation.gateway.from.x)));
        hash = mixHash(hash, uint64_t(uint32_t(relation.gateway.to.y)));
    }
    // What actions offer is deliberately not in here.
    //
    // It has its own test (see refreshOptions), and putting it here as well
    // meant that working out one action's stances -- which happens for a
    // different action every tick, because the geometry moving marks them
    // all unknown and only one is redone per tick -- re-derived the whole
    // world: every stance of every region, every crossing, every piece of
    // free space. Two thousand seven hundred full derivations in a run of
    // five thousand six hundred ticks, and a decision that cost more than
    // the frame it had to fit in.
    return hash;
}

size_t TraversalModel::placeFor(RegionId region, int component)
{
    for (size_t index = 0; index < m_places.size(); ++index)
        if (m_places[index].region == region
            && m_places[index].component == component)
            return index;
    // A piece of space nothing had named before. Whatever index was built
    // over the old set does not know about it, and routing with a stale one
    // hands back a route through an opening that has no crossing -- which
    // the executor can only report as the model contradicting itself.
    m_places.push_back({ region, component });
    m_placeParent.push_back(m_places.size() - 1);
    m_indexStale = true;
    return m_places.size() - 1;
}

size_t TraversalModel::rootOf(size_t place) const
{
    while (m_placeParent[place] != place)
        place = m_placeParent[place];
    return place;
}

void TraversalModel::joinPlaces(size_t left, size_t right)
{
    const size_t a = rootOf(left);
    const size_t b = rootOf(right);
    if (a != b)
        m_placeParent[std::max(a, b)] = std::min(a, b);
}

size_t TraversalModel::actorPlace() const
{
    return m_actorPlaceKnown ? m_actorPlace : m_places.size();
}

size_t TraversalModel::placeIn(RegionId region) const
{
    for (size_t index = 0; index < m_places.size(); ++index)
        if (m_places[index].region == region)
            return rootOf(index);
    return m_places.size();
}

const std::vector<semantic::Vec2> *TraversalModel::stancesOf(
    RegionId region) const
{
    auto found = m_stances.find(region);
    return found == m_stances.end() ? nullptr : &found->second;
}

int TraversalModel::stancesIn(RegionId region) const
{
    auto found = m_stances.find(region);
    return found == m_stances.end() ? 0 : int(found->second.size());
}

int TraversalModel::piecesOf(RegionId region) const
{
    int pieces = 0;
    for (const Place &place : m_places)
        if (place.region == region)
            ++pieces;
    return pieces;
}

size_t TraversalModel::splitRegions() const
{
    std::vector<RegionId> seen;
    for (const Place &place : m_places)
        if (place.component > 0)
        {
            bool known = false;
            for (RegionId id : seen)
                if (id == place.region)
                    known = true;
            if (!known)
                seen.push_back(place.region);
        }
    return seen.size();
}

void TraversalModel::rebuildIndex() const
{
    m_outgoing.assign(m_places.size(), {});
    for (size_t index = 0; index < m_transitions.size(); ++index)
    {
        if (!m_transitions[index].executable)
            continue;
        const size_t from = m_transitionPlaces[index].leaves;
        if (from < m_outgoing.size())
            m_outgoing[from].push_back(index);
    }
    m_indexStale = false;
}

// What the first leg of a route costs is measured from where the body
// actually is, not from the middle of the region it happens to be standing
// in. Without that, something on the far side of a large room looks as near
// as something underfoot, and the order things get done in stops having
// anything to do with where they are.
int TraversalModel::legCost(size_t transition, size_t place,
                            const semantic::Vec2 &at) const
{
    if (place != m_actorPlace || !m_actorPlaceKnown)
        return m_transitions[transition].cost;
    const semantic::Vec2 &crossing = m_transitions[transition].crossing;
    // Out to the opening from where the body is, then the far half of the
    // stored cost, which is the opening to the next region's middle.
    return std::max(1, semantic::planarDistance(at, crossing)
        + m_transitions[transition].cost / 2);
}

// Where a route starts. For the region the actor is standing in that is the
// piece of free space it is actually in; for anywhere else it is any piece,
// because the question being asked is about that region as a whole.
void TraversalModel::placesOf(RegionId region,
                              std::vector<size_t> &out) const
{
    out.clear();
    if (m_indexStale)
        rebuildIndex();
    for (size_t index = 0; index < m_places.size(); ++index)
    {
        if (m_places[index].region != region)
            continue;
        const size_t root = rootOf(index);
        bool known = false;
        for (size_t already : out)
            known = known || already == root;
        if (!known)
            out.push_back(root);
    }
}

bool TraversalModel::placesAcross(semantic::RelationId way, size_t &leaves,
                                  size_t &arrives) const
{
    if (m_indexStale)
        rebuildIndex();
    for (size_t index = 0; index < m_transitions.size(); ++index)
        if (m_transitions[index].relation == way)
        {
            leaves = m_transitionPlaces[index].leaves;
            arrives = m_transitionPlaces[index].arrives;
            return true;
        }
    return false;
}

size_t TraversalModel::startPlace(RegionId from) const
{
    if (from == kNoId)
        return m_places.size();
    if (from == m_actorRegion && m_actorPlaceKnown)
        return m_actorPlace;
    return placeIn(from);
}

void TraversalModel::reachableFrom(RegionId origin,
                                   std::vector<RegionId> &out) const
{
    out.clear();
    if (m_indexStale)
        rebuildIndex();
    if (origin == kNoId)
        return;
    // Standing somewhere is always being somewhere, whether or not anything
    // leads out of it.
    out.push_back(origin);
    const size_t start = startPlace(origin);
    if (start >= m_places.size())
        return;
    std::vector<char> seen(m_places.size(), 0);
    std::vector<size_t> pending;
    pending.push_back(start);
    seen[start] = 1;
    while (!pending.empty())
    {
        const size_t node = pending.back();
        pending.pop_back();
        const RegionId region = m_places[node].region;
        bool known = false;
        for (RegionId already : out)
            if (already == region)
            {
                known = true;
                break;
            }
        if (!known)
            out.push_back(region);
        for (size_t index : m_outgoing[node])
        {
            const size_t next = m_transitionPlaces[index].arrives;
            if (next >= seen.size() || seen[next])
                continue;
            // What can be reached is what the world and this body allow, not
            // that minus what has been tried and gone wrong. A failure under
            // an unchanged world is a contradiction to be explained, and
            // subtracting it from the map hides the thing worth finding.
            seen[next] = 1;
            pending.push_back(next);
        }
    }
}

bool TraversalModel::route(RegionId from, RegionId to,
                           std::vector<RegionId> &path,
                           std::vector<RelationId> &via,
                           std::vector<size_t> *through) const
{
    path.clear();
    via.clear();
    if (through)
        through->clear();
    if (m_indexStale)
        rebuildIndex();
    if (from == kNoId || to == kNoId)
        return false;
    if (from == to)
    {
        path.push_back(from);
        if (through)
            through->push_back(startPlace(from));
        return true;
    }
    return findRoute(startPlace(from), kAnyPlace, to, &path, &via, nullptr,
                     through);
}

// A plan that may have to arrange the world before it can be walked.
//
// Dijkstra over (Place, what the plan has committed the geometry to). The
// commitments are lazy: a piece of geometry only enters the search state
// when an edge is taken that depends on it, so a level full of movers costs
// nothing until one of them is actually on the way.
//
// Three kinds of edge, and not one of them is a new thing the body can do:
//   walking a crossing that works as the world stands
//   walking a crossing that works with some geometry resting elsewhere
//   acting on something that moves that geometry
//
// The goal is anywhere the body has not stood. Multi-goal, so one search
// answers both "is there anything left worth doing" and "what is the first
// step of getting there".
bool TraversalModel::planSomewhereNew(const semantic::SemanticWorld &world,
                                      RegionId from, PlanStep &first,
                                      std::vector<PlanStep> *whole) const
{
    if (m_indexStale)
        rebuildIndex();
    const size_t places = m_places.size();
    const size_t start = startPlace(from);
    if (start >= places)
        return false;

    using Commitment = std::vector<std::pair<semantic::GeometryId, uint32_t>>;
    struct Node { size_t place; Commitment held; };
    auto keyOf = [](size_t place, const Commitment &held) {
        std::string key = std::to_string(place);
        for (const auto &one : held)
            key += ":" + std::to_string(one.first) + "="
                + std::to_string(one.second);
        return key;
    };
    auto restingAt = [&](const Commitment &held, semantic::GeometryId which) {
        for (const auto &one : held)
            if (one.first == which)
                return one.second;
        const semantic::StatefulGeometry *piece = world.geometryOf(which);
        return piece ? piece->state : 0u;
    };
    // Is everything the plan has committed to already true? If so this node
    // is somewhere the body can get to right now, and an act done here is an
    // act the executor can actually be sent to do. If not, the node only
    // exists inside the plan, and an act here is not yet a first step.
    auto standingNow = [&](const Commitment &held) {
        for (const auto &one : held)
        {
            const semantic::StatefulGeometry *piece =
                world.geometryOf(one.first);
            if (!piece || piece->state != one.second)
                return false;
        }
        return true;
    };
    // Holding more than a handful of things in one configuration at once is
    // not a plan this is going to find, and the bound is what keeps a level
    // full of movers from becoming a search over every combination of them.
    const size_t kMostHeld = 4;

    // What the body can already walk to. A plan is for somewhere it cannot.
    //
    // Stated any other way, this rule goes wrong. "A plan containing an act"
    // is what it used to ask for, and where the walking it wanted was
    // already possible, a search told to find an act had to invent one: the
    // cheapest invention is a round trip that puts a lift back where it
    // started, and doing the first half of that moves the lift out from
    // under the very walk the plan was for. Asking instead for somewhere
    // unreachable says what is actually wanted, and leaves a plan free to
    // move the same lift twice -- which is exactly what calling a lift and
    // then riding it is.
    std::vector<char> walkable(places, 0);
    {
        std::vector<size_t> front{ start };
        walkable[start] = 1;
        while (!front.empty())
        {
            const size_t at = front.back();
            front.pop_back();
            for (size_t index : m_outgoing[at])
            {
                const size_t next = m_transitionPlaces[index].arrives;
                if (next < places && !walkable[next])
                {
                    walkable[next] = 1;
                    front.push_back(next);
                }
            }
        }
    }

    std::map<std::string, int64_t> seen;
    std::map<std::string, PlanStep> opening;
    std::map<std::string, Node> nodes;
    std::map<std::string, std::string> cameFrom;
    std::map<std::string, PlanStep> arrivedBy;
    using Queued = std::pair<int64_t, std::string>;
    std::priority_queue<Queued, std::vector<Queued>, std::greater<Queued>>
        queue;
    const std::string origin = keyOf(start, Commitment());
    seen[origin] = 0;
    nodes[origin] = Node{ start, Commitment() };
    opening[origin] = PlanStep();
    queue.push({ 0, origin });

    while (!queue.empty())
    {
        const Queued top = queue.top();
        queue.pop();
        auto known = seen.find(top.second);
        if (known == seen.end() || top.first != known->second)
            continue;
        const Node here = nodes[top.second];
        const PlanStep arrived = opening[top.second];

        // Somewhere the body has not stood and cannot walk to, reached by
        // a plan that arranges the world on the way.
        //
        // Somewhere it can already walk to is not this rule's business:
        // ordinary routing already offers it, and offering the first stride
        // of it here just means arriving, re-planning, and being handed the
        // same stride again. What is wanted is the first *act* -- the
        // walking before it is reachable as things stand, by construction,
        // so the executor gets there on its own.
        if (arrived.act && !walkable[here.place])
        {
            const Region *space = world.region(m_places[here.place].region);
            if (space && space->exists && !space->occupied)
            {
                first = arrived;
                first.cost = int(top.first);
                if (whole)
                {
                    whole->clear();
                    std::string walk = top.second;
                    while (arrivedBy.count(walk))
                    {
                        whole->push_back(arrivedBy[walk]);
                        walk = cameFrom[walk];
                    }
                    std::reverse(whole->begin(), whole->end());
                }
                return true;
            }
        }

        auto relax = [&](size_t place, const Commitment &held, int64_t cost,
                         const PlanStep &step) {
            if (place >= places || held.size() > kMostHeld)
                return;
            const std::string key = keyOf(place, held);
            auto found = seen.find(key);
            if (found != seen.end() && found->second <= cost)
                return;
            seen[key] = cost;
            nodes[key] = Node{ place, held };
            cameFrom[key] = top.second;
            arrivedBy[key] = step;
            // Only an act reachable in the world as it stands can be the
            // first step. An act the plan can only get to after moving
            // something else is a later step; the executor would be sent to
            // a place it cannot route to and would come straight back with
            // no route, over and over.
            opening[key] = arrived.act
                ? arrived
                : ((step.act && standingNow(here.held)) ? step : PlanStep());
            queue.push({ cost, key });
        };

        // Walking, as the world stands.
        for (size_t index : m_outgoing[here.place])
        {
            const size_t next = m_transitionPlaces[index].arrives;
            if (next >= places)
                continue;
            // A crossing that works as the world stands only works while the
            // world still stands that way. If the plan has already committed
            // the geometry this crossing is made of to some other
            // configuration, this is not a crossing the plan may use --
            // however plainly it can be walked at this moment.
            //
            // Leaving this out is how the plan came out as "send the lift
            // down, get on it, and walk off at the top": the last step was
            // derived while the lift was at the top, and the plan had
            // already sent it to the bottom to get on it.
            const RelationId about = m_transitions[index].relation;
            const semantic::GeometryId shaped = conditionOf(about);
            if (shaped != semantic::kNoId)
            {
                bool held = false;
                uint32_t wanted = 0;
                for (const auto &one : here.held)
                    if (one.first == shaped)
                    {
                        held = true;
                        wanted = one.second;
                    }
                if (held && !possibleInConfiguration(about, wanted,
                                                     Mode::Walk))
                    continue;
            }
            PlanStep step;
            step.crossing = m_transitions[index].relation;
            step.region = m_places[here.place].region;
            step.steps = arrived.steps + 1;
            relax(next, here.held,
                  top.first + legCost(index, here.place, m_actorAt), step);
        }

        // Walking, with some geometry resting where the plan has put it.
        for (size_t index = 0; index < m_entries.size(); ++index)
        {
            const Entry &entry = m_entries[index];
            if (!entry.valid || !entry.exists
                || entry.conditionedOn == semantic::kNoId)
                continue;
            if ((m_executable & modeBit(Mode::Walk)) == 0)
                continue;
            if (entry.leavesPlace != here.place
                || entry.arrivesPlace >= places)
                continue;
            // Only for geometry the plan has actually committed to moving.
            //
            // Where it is resting now, the authoritative answer is the
            // derived transition above -- that came from asking the engine
            // about the world as it is, while these came from asking it
            // about a world posed by hand, and the two can disagree at the
            // margin. Trusting the posed answer for the present produces
            // plans whose walking prefix the executor cannot actually route,
            // and it comes straight back with no route, for ever.
            bool committed = false;
            for (const auto &one : here.held)
                committed = committed || one.first == entry.conditionedOn;
            if (!committed)
                continue;
            const uint32_t resting = restingAt(here.held,
                                               entry.conditionedOn);
            if (resting >= entry.modesInConfiguration.size())
                continue;
            if ((entry.modesInConfiguration[resting]
                    & modeBit(Mode::Walk)) == 0)
                continue;
            const Commitment held = here.held;
            PlanStep step;
            step.crossing = RelationId(index);
            step.region = entry.from;
            step.steps = arrived.steps + 1;
            relax(entry.arrivesPlace, held,
                  top.first + entry.cost[int(Mode::Walk)], step);
        }

        // Acting on something that moves geometry, from where we are.
        for (const semantic::Affordance &thing : world.affordances())
        {
            if (thing.id == semantic::kNoId || !thing.exists
                || !thing.executable || thing.moves.empty())
                continue;
            for (size_t option = 0; option < thing.domain.size(); ++option)
            {
                if (size_t(thing.id) >= m_optionPlace.size()
                    || option >= m_optionPlace[size_t(thing.id)].size())
                    continue;
                if (m_optionPlace[size_t(thing.id)][option] != here.place)
                    continue;
                // The same policy the router uses. A plan whose first act the
                // executor has already been unable to drive is a plan that
                // will be proposed, refused, and proposed again.
                if (refusedOption(thing.id, uint32_t(option)))
                    continue;
                if (survivalOf(thing.id, uint32_t(option))
                        == Survival::Unsafe)
                    continue;
                for (semantic::GeometryId which : thing.moves)
                {
                    const semantic::StatefulGeometry *piece =
                        world.geometryOf(which);
                    if (!piece || piece->configurations.size() < 2)
                        continue;
                    const uint32_t resting = restingAt(here.held, which);
                    for (uint32_t to = 0;
                         to < uint32_t(piece->configurations.size()); ++to)
                    {
                        if (to == resting)
                            continue;
                        Commitment held;
                        for (const auto &one : here.held)
                            if (one.first != which)
                                held.push_back(one);
                        held.push_back({ which, to });
                        std::sort(held.begin(), held.end());
                        PlanStep step;
                        step.act = true;
                        step.affordance = thing.id;
                        step.option = uint32_t(option);
                        step.region = m_places[here.place].region;
                        step.steps = arrived.steps + 1;
                        relax(here.place, held, top.first + 1, step);
                    }
                }
            }
        }
    }
    return false;
}

bool TraversalModel::findRoute(size_t start, size_t goalPlace,
                               RegionId goalRegion,
                               std::vector<RegionId> *path,
                               std::vector<RelationId> *via, int *cost,
                               std::vector<size_t> *through) const
{
    const size_t count = m_places.size();
    if (start >= count)
        return false;
    if (goalPlace != kAnyPlace && goalPlace >= count)
        return false;
    if (start == goalPlace)
    {
        if (path)
            path->push_back(m_places[start].region);
        if (through)
            through->push_back(start);
        if (cost)
            *cost = 0;
        return true;
    }
    const int64_t infinity = std::numeric_limits<int64_t>::max() / 4;
    std::vector<int64_t> distance(count, infinity);
    std::vector<size_t> parent(count, count);
    std::vector<size_t> parentEdge(count, m_transitions.size());
    using Step = std::pair<int64_t, size_t>;
    std::priority_queue<Step, std::vector<Step>, std::greater<Step>> queue;
    distance[start] = 0;
    queue.push({ 0, start });
    size_t goal = count;
    while (!queue.empty())
    {
        const Step top = queue.top();
        queue.pop();
        const size_t node = top.second;
        if (top.first != distance[node])
            continue;
        if (goalPlace != kAnyPlace ? node == goalPlace
                                   : m_places[node].region == goalRegion)
        {
            goal = node;
            break;
        }
        for (size_t index : m_outgoing[node])
        {
            const size_t next = m_transitionPlaces[index].arrives;
            if (next >= count)
                continue;
            if (refusedCrossing(m_transitions[index].relation))
                continue;
            const int64_t candidate = distance[node]
                + legCost(index, node, m_actorAt);
            if (candidate >= distance[next])
                continue;
            distance[next] = candidate;
            parent[next] = node;
            parentEdge[next] = index;
            queue.push({ candidate, next });
        }
    }
    if (goal == count)
        return false;
    if (cost)
        *cost = int(distance[goal]);
    if (!path && !via && !through)
        return true;
    std::vector<RegionId> reverseNodes;
    std::vector<RelationId> reverseEdges;
    std::vector<size_t> reversePlaces;
    for (size_t node = goal; node != count; node = parent[node])
    {
        reverseNodes.push_back(m_places[node].region);
        reversePlaces.push_back(node);
        if (node == start)
            break;
        reverseEdges.push_back(m_transitions[parentEdge[node]].relation);
    }
    if (path)
        path->assign(reverseNodes.rbegin(), reverseNodes.rend());
    if (via)
        via->assign(reverseEdges.rbegin(), reverseEdges.rend());
    if (through)
        through->assign(reversePlaces.rbegin(), reversePlaces.rend());
    return true;
}

// What a journey actually costs, from where the body is.
//
// Summing the crossings alone says a journey with no crossings costs
// nothing, so everything in the room the body is standing in ties at zero
// and the winner is whichever the loop happened to reach first. That is the
// whole of why the order looks arbitrary: within a room there is no order,
// and a pickup at the far end of the hall and one at the bot's feet are the
// same price. Across rooms it is nearly as bad, because each crossing is
// priced from the middle of the room it leaves to the middle of the room it
// enters, so which end of a room something is at makes no difference either.
//
// So: the crossings in the middle are priced as before, and the two ends are
// measured properly -- from the body to the first opening, and from the last
// opening to the thing itself.
int TraversalModel::journeyCost(const std::vector<RelationId> &via,
                                const semantic::Vec2 &target) const
{
    if (via.empty())
        return semantic::planarDistance(m_actorAt, target);
    int cost = 0;
    const Transition *first = nullptr;
    const Transition *last = nullptr;
    for (RelationId relation : via)
        for (const Transition &transition : m_transitions)
            if (transition.relation == relation)
            {
                cost += transition.cost;
                if (!first)
                    first = &transition;
                last = &transition;
                break;
            }
    if (first)
        cost += semantic::planarDistance(m_actorAt, first->crossing)
            - first->head;
    if (last)
        cost += semantic::planarDistance(last->crossing, target)
            - last->tail;
    return std::max(1, cost);
}

semantic::GeometryId TraversalModel::conditionOf(RelationId relation) const
{
    if (size_t(relation) >= m_entries.size())
        return kNoId;
    return m_entries[size_t(relation)].conditionedOn;
}

bool TraversalModel::possibleInConfiguration(RelationId relation,
                                             uint32_t configuration,
                                             Mode mode) const
{
    if (size_t(relation) >= m_entries.size())
        return false;
    const Entry &entry = m_entries[size_t(relation)];
    if (configuration >= entry.modesInConfiguration.size())
        return false;
    return (entry.modesInConfiguration[configuration] & modeBit(mode)) != 0;
}

bool TraversalModel::routeCost(RegionId from, RegionId to, int &cost) const
{
    if (refusedRegion(to))
        return false;
    std::vector<RegionId> path;
    std::vector<RelationId> via;
    if (!route(from, to, path, via))
        return false;
    auto anchor = m_regionAnchor.find(to);
    cost = journeyCost(via, anchor == m_regionAnchor.end() ? m_actorAt
                                                          : anchor->second);
    return true;
}

TraversalModel::Verdict TraversalModel::verdictFor(
    semantic::RelationId relation, Mode mode) const
{
    Verdict said;
    if (relation == kNoId || size_t(relation) >= m_entries.size())
        return said;
    const Entry &entry = m_entries[size_t(relation)];
    said.valid = entry.valid;
    said.exists = entry.exists;
    said.possible = entry.possible[int(mode)];
    said.executable = (m_executable & modeBit(mode)) != 0;
    said.refused = refusedCrossing(relation);
    said.leaves = int(entry.leaves[int(mode)].size());
    said.arrives = int(entry.arrives[int(mode)].size());
    said.from = stancesIn(entry.from) > 0 ? 1 : 0;
    said.to = stancesIn(entry.to) > 0 ? 1 : 0;
    size_t leaves = 0;
    size_t arrives = 0;
    said.routed = placesAcross(relation, leaves, arrives);
    return said;
}

bool TraversalModel::derived(RelationId relation, Mode mode) const
{
    for (const Transition &transition : m_transitions)
        if (transition.relation == relation && transition.mode == mode)
            return true;
    return false;
}

bool TraversalModel::crossingFor(semantic::RelationId relation,
                                 semantic::Vec2 &out,
                                 semantic::Vec2 &arrival) const
{
    for (const Transition &transition : m_transitions)
    {
        if (transition.relation != relation || !transition.executable)
            continue;
        out = transition.crossing;
        arrival = transition.arrival;
        return true;
    }
    return false;
}

int TraversalModel::knownButUnexecutable() const
{
    int count = 0;
    for (const Transition &transition : m_transitions)
        if (transition.possible && !transition.executable)
            ++count;
    return count;
}

} // namespace traversal
