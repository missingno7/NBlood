#include "traversal_model.h"

#include <algorithm>
#include <limits>
#include <queue>

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
        // Whether the ground here is moving at this moment. Not part of what
        // makes a Region, and every part of what makes a crossing into one
        // available, so it belongs here and not in the clustering.
        hash = mixHash(hash, region.hazard.shifting ? 1u : 0u);
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

// The one rule, for one region.
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
    // Every region's ways out, gathered in one pass over the relations
    // rather than one pass per region.
    std::map<RegionId, std::vector<semantic::Segment>> ways;
    for (const SpatialRelation &relation : world.relations())
    {
        if (!relation.exists || relation.blocked || relation.id == kNoId)
            continue;
        if (relation.gateway.width() <= 0)
            continue;
        ways[relation.from].push_back({ relation.gateway.from,
                                        relation.gateway.to });
    }
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
    m_optionPlace.assign(world.affordances().size(), {});
    m_optionWhy.assign(world.affordances().size(), {});
    m_optionAt.assign(world.affordances().size(), {});
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
                why[option] = 3;
                continue;
            }
            why[option] = 0;
            places[option] = rootOf(placeFor(where.region, m_scratch[0]));
        }
    }
}

namespace {
constexpr size_t kAnyPlace = size_t(-1);
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
    std::vector<RegionId> path;
    std::vector<RelationId> via;
    if (refusedOption(action, option))
        return false;
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
                                 std::vector<RelationId> &via) const
{
    path.clear();
    via.clear();
    if (refusedOption(action, option))
        return false;
    if (size_t(action) >= m_optionPlace.size()
        || size_t(option) >= m_optionPlace[size_t(action)].size())
        return false;
    if (m_indexStale)
        rebuildIndex();
    return findRoute(startPlace(from), m_optionPlace[size_t(action)][option],
                     kNoId, &path, &via, nullptr);
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
        // Standing somewhere the free space does not describe. Say nothing
        // rather than something wrong: every piece of this region counts as
        // where the actor is, which is what the model assumed before free
        // space was split at all.
        for (int component = 0; component + 1 < map.components(); ++component)
            joinPlaces(placeFor(m_actorRegion, 0),
                       placeFor(m_actorRegion, component + 1));
        if (map.components() > 0)
        {
            m_actorPlace = rootOf(placeFor(m_actorRegion, 0));
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
                           std::vector<RelationId> &via) const
{
    path.clear();
    via.clear();
    if (m_indexStale)
        rebuildIndex();
    if (from == kNoId || to == kNoId)
        return false;
    if (from == to)
    {
        path.push_back(from);
        return true;
    }
    return findRoute(startPlace(from), kAnyPlace, to, &path, &via, nullptr);
}

bool TraversalModel::findRoute(size_t start, size_t goalPlace,
                               RegionId goalRegion,
                               std::vector<RegionId> *path,
                               std::vector<RelationId> *via, int *cost) const
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
    if (!path && !via)
        return true;
    std::vector<RegionId> reverseNodes;
    std::vector<RelationId> reverseEdges;
    for (size_t node = goal; node != count; node = parent[node])
    {
        reverseNodes.push_back(m_places[node].region);
        if (node == start)
            break;
        reverseEdges.push_back(m_transitions[parentEdge[node]].relation);
    }
    if (path)
        path->assign(reverseNodes.rbegin(), reverseNodes.rend());
    if (via)
        via->assign(reverseEdges.rbegin(), reverseEdges.rend());
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
