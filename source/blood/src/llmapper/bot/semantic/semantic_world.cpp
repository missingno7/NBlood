#include "semantic_world.h"

#include <algorithm>

namespace semantic {

namespace {

template <typename T>
T &slotFor(std::vector<T> &store, uint32_t id)
{
    if (store.size() <= size_t(id))
        store.resize(size_t(id) + 1);
    return store[size_t(id)];
}

} // namespace

const char *clearanceName(ClearanceClass clearance)
{
    switch (clearance)
    {
    case ClearanceClass::Standing: return "standing";
    case ClearanceClass::Crouching: return "crouching";
    case ClearanceClass::None: return "none";
    }
    return "unknown";
}

const char *actionName(ActionKind action)
{
    switch (action)
    {
    case ActionKind::Use: return "use";
    case ActionKind::Collect: return "collect";
    }
    return "unknown";
}

void SemanticWorld::apply(const WorldDelta &delta)
{    const bool wasFresh = m_attemptFresh;
    m_attemptFresh = false;
    (void)wasFresh;
    if (!delta.geometry.empty())
        m_geometry = delta.geometry;

    for (const Region &incoming : delta.regions)
    {
        if (incoming.id == kNoId)
            continue;
        slotFor(m_regions, incoming.id) = incoming;
    }
    for (const SpatialRelation &incoming : delta.relations)
    {
        if (incoming.id == kNoId)
            continue;
        SpatialRelation &stored = slotFor(m_relations, incoming.id);
        // The mapper owns what the world offers; this layer owns the history
        // of where the actor has been.
        const bool crossed = stored.crossed;
        const bool inspected = stored.inspected;
        stored = incoming;
        stored.crossed = crossed;
        stored.inspected = inspected;
    }
    for (const Affordance &incoming : delta.affordances)
    {
        if (incoming.id == kNoId)
            continue;
        Affordance &stored = slotFor(m_affordances, incoming.id);
        const int attempts = stored.attempts;
        const bool openedWay = stored.lastAttemptOpenedWay;
        std::vector<RelationId> affects;
        std::vector<uint64_t> affectedAt;
        std::vector<GeometryId> moves;
        std::vector<Affordance::Fruitless> vain;
        affects.swap(stored.affects);
        vain.swap(stored.triedInVain);
        // `commands` is not carried across: the mapper reads it from the
        // world every build, like everything else the world holds.
        affectedAt.swap(stored.affectedAt);
        moves.swap(stored.moves);
        const uint64_t triedAt = stored.triedAt;
        const bool triedAtKnown = stored.triedAtKnown;
        stored = incoming;
        stored.attempts = attempts;
        stored.lastAttemptOpenedWay = openedWay;
        stored.affects.swap(affects);
        stored.affectedAt.swap(affectedAt);
        stored.moves.swap(moves);
        stored.triedInVain.swap(vain);
        stored.triedAt = triedAt;
        stored.triedAtKnown = triedAtKnown;
    }
    const RegionId previous = m_actor.region;
    m_actor = delta.actor;
    noteMovement(previous, m_actor.region);
    m_settling = delta.settling;
    m_establishing = delta.establishing;
    m_revision = delta.revision;
}

// Going from one piece of space to the next means a way between them has
// been used. Both directions are recorded: having walked through a doorway
// is knowing what is on the other side of it, whichever way round it is
// asked about.
void SemanticWorld::noteMovement(RegionId previous, RegionId now)
{
    if (previous == kNoId || now == kNoId || previous == now)
        return;
    for (SpatialRelation &relation : m_relations)
    {
        if (relation.id == kNoId)
            continue;
        if ((relation.from == previous && relation.to == now)
            || (relation.from == now && relation.to == previous))
            relation.crossed = true;
    }
}

const Region *SemanticWorld::region(RegionId id) const
{
    if (id == kNoId || size_t(id) >= m_regions.size())
        return nullptr;
    return &m_regions[size_t(id)];
}

const SpatialRelation *SemanticWorld::relation(RelationId id) const
{
    if (id == kNoId || size_t(id) >= m_relations.size())
        return nullptr;
    return &m_relations[size_t(id)];
}

const Affordance *SemanticWorld::affordance(AffordanceId id) const
{
    if (id == kNoId || size_t(id) >= m_affordances.size())
        return nullptr;
    return &m_affordances[size_t(id)];
}

size_t SemanticWorld::liveRegionCount() const
{
    size_t live = 0;
    for (const Region &region : m_regions)
        if (region.exists && region.id != kNoId)
            ++live;
    return live;
}

size_t SemanticWorld::liveRelationCount() const
{
    size_t live = 0;
    for (const SpatialRelation &relation : m_relations)
        if (relation.exists && relation.id != kNoId)
            ++live;
    return live;
}

size_t SemanticWorld::uncrossedRelationCount() const
{
    size_t open = 0;
    for (const SpatialRelation &relation : m_relations)
        if (relation.exists && relation.id != kNoId && !relation.crossed)
            ++open;
    return open;
}

void SemanticWorld::noteObstruction(RelationId id, AffordanceId thing)
{
    if (id == kNoId || size_t(id) >= m_relations.size())
        return;
    m_relations[size_t(id)].obstruction = thing;
    if (thing != kNoId && size_t(thing) < m_affordances.size())
        m_affordances[size_t(thing)].barrier = true;
}

void SemanticWorld::noteInspected(RelationId id)
{
    if (id == kNoId || size_t(id) >= m_relations.size())
        return;
    m_relations[size_t(id)].inspected = true;
    // Looking at a way is looking at both ends of it: the same opening
    // described from the other side has been seen too.
    const RegionId near = m_relations[size_t(id)].from;
    const RegionId far = m_relations[size_t(id)].to;
    for (SpatialRelation &relation : m_relations)
        if (relation.from == far && relation.to == near)
            relation.inspected = true;
}

// Everything about a way that acting on something could change. Not what the
// way means -- whether it is shut, how tall the step is, where its edges are.
// A wall travelling down a corridor moves the edges; a door opening changes
// what stands in it; a lift changes the step. All three land here.
uint64_t SemanticWorld::stateOf(const SpatialRelation &relation) const
{
    uint64_t hash = 1469598103934665603ULL;
    auto mix = [&hash](uint64_t item)
    {
        hash ^= item + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    };
    mix(relation.exists ? 1 : 0);
    mix(relation.blocked ? 1 : 0);
    mix(uint64_t(uint32_t(relation.verticalStep)));
    mix(uint64_t(uint32_t(relation.clearance)));
    mix(uint64_t(uint32_t(relation.gateway.from.x)));
    mix(uint64_t(uint32_t(relation.gateway.from.y)));
    mix(uint64_t(uint32_t(relation.gateway.to.x)));
    mix(uint64_t(uint32_t(relation.gateway.to.y)));
    // And the shape of the two spaces it joins.
    //
    // What shuts a way is not always on the way. A wall travelling down the
    // corridor beyond a doorway leaves the doorway exactly as it was and
    // makes it useless all the same, and a record that watches only the
    // doorway learns nothing from having watched. The rooms either side are
    // part of what the way is for.
    auto shape = [&](RegionId id)
    {
        const Region *space = region(id);
        if (!space)
        {
            mix(0);
            return;
        }
        mix(space->exists ? 1 : 0);
        mix(uint64_t(space->footprint.size()));
        for (const Vec2 &corner : space->footprint)
        {
            mix(uint64_t(uint32_t(corner.x)));
            mix(uint64_t(uint32_t(corner.y)));
        }
        // Holes and all. What a room wraps around is as much its shape as
        // its outline is, and a wall travelling through the middle of one
        // moves nothing else about it.
        mix(uint64_t(space->holes.size()));
        for (const Loop &hole : space->holes)
        {
            mix(uint64_t(hole.size()));
            for (const Vec2 &corner : hole)
            {
                mix(uint64_t(uint32_t(corner.x)));
                mix(uint64_t(uint32_t(corner.y)));
            }
        }
        mix(uint64_t(uint32_t(space->interior.x)));
        mix(uint64_t(uint32_t(space->interior.y)));
    };
    shape(relation.from);
    shape(relation.to);
    return hash;
}

void SemanticWorld::beginAttempt(AffordanceId id, int possibilities)
{
    m_attemptFresh = true;
    if (size_t(id) >= m_affordances.size())
        return;
    // Whatever was being watched, stop watching it now and keep what was
    // learned. Goals no longer stand still waiting for the world to come to
    // rest, so a second act can begin while the first is still playing out,
    // and silently dropping the first one's record loses the only evidence
    // there is about what it does.
    if (m_awaiting != kNoId && m_awaiting != id)
        closeAttempt(possibilities);
    ++m_affordances[size_t(id)].attempts;
    m_awaiting = id;
    m_possibilitiesAtAttempt = possibilities;
    // What every way looks like now, so that what acting on this thing
    // changes can be read off afterwards rather than assumed.
    m_waysAtAttempt.assign(m_relations.size(), 0);
    for (size_t index = 0; index < m_relations.size(); ++index)
        m_waysAtAttempt[index] = stateOf(m_relations[index]);
}

void SemanticWorld::noteMoves(AffordanceId thing, GeometryId geometry)
{
    if (thing == kNoId || geometry == kNoId
        || size_t(thing) >= m_affordances.size())
        return;
    Affordance &acting = m_affordances[size_t(thing)];
    for (GeometryId known : acting.moves)
        if (known == geometry)
            return;
    acting.moves.push_back(geometry);
}

bool SemanticWorld::moves(AffordanceId thing, GeometryId geometry) const
{
    if (thing == kNoId || geometry == kNoId
        || size_t(thing) >= m_affordances.size())
        return false;
    for (GeometryId known : m_affordances[size_t(thing)].moves)
        if (known == geometry)
            return true;
    return false;
}

bool SemanticWorld::works(AffordanceId thing, GeometryId geometry) const
{
    if (thing == kNoId || geometry == kNoId
        || size_t(thing) >= m_affordances.size())
        return false;
    const Affordance &acting = m_affordances[size_t(thing)];
    for (GeometryId known : acting.moves)
        if (known == geometry)
            return true;
    for (GeometryId known : acting.commands)
        if (known == geometry)
            return true;
    return false;
}

bool SemanticWorld::effectKnown(AffordanceId thing) const
{
    if (thing == kNoId || size_t(thing) >= m_affordances.size())
        return false;
    const Affordance &acting = m_affordances[size_t(thing)];
    if (acting.attempts > 0)
        return true;
    if (acting.commands.empty())
        return false;   // nothing says what it works: only doing it will
    for (const Affordance &other : m_affordances)
    {
        if (other.id == thing || other.attempts == 0)
            continue;
        for (GeometryId mine : acting.commands)
            if (works(other.id, mine))
                return true;
    }
    return false;
}

void SemanticWorld::noteFruitless(AffordanceId thing, RelationId way,
                                  uint64_t blockingState)
{
    if (thing == kNoId || way == kNoId
        || size_t(thing) >= m_affordances.size()
        || size_t(way) >= m_relations.size())
        return;
    Affordance &acting = m_affordances[size_t(thing)];
    const AffordanceId blocking = m_relations[size_t(way)].obstruction;
    for (Affordance::Fruitless &known : acting.triedInVain)
        if (known.way == way)
        {
            known.blocking = blocking;
            known.blockingState = blockingState;
            return;
        }
    acting.triedInVain.push_back({ way, blocking, blockingState });
}

bool SemanticWorld::worthTryingFor(AffordanceId thing, RelationId way,
                                   uint64_t blockingState) const
{
    if (thing == kNoId || way == kNoId
        || size_t(thing) >= m_affordances.size()
        || size_t(way) >= m_relations.size())
        return true;
    const AffordanceId blocking = m_relations[size_t(way)].obstruction;
    for (const Affordance::Fruitless &known
             : m_affordances[size_t(thing)].triedInVain)
        if (known.way == way)
            return known.blocking != blocking
                || known.blockingState != blockingState;
    return true;
}

void SemanticWorld::noteAffects(AffordanceId thing, RelationId way)
{
    if (thing == kNoId || way == kNoId
        || size_t(thing) >= m_affordances.size()
        || size_t(way) >= m_relations.size())
        return;
    Affordance &acting = m_affordances[size_t(thing)];
    for (size_t index = 0; index < acting.affects.size(); ++index)
        if (acting.affects[index] == way)
        {
            if (index < acting.affectedAt.size())
                acting.affectedAt[index] = stateOf(m_relations[size_t(way)]);
            return;
        }
    acting.affects.push_back(way);
    acting.affectedAt.push_back(stateOf(m_relations[size_t(way)]));
}

bool SemanticWorld::worthTrying(AffordanceId thing, RelationId way) const
{
    if (thing == kNoId || way == kNoId
        || size_t(thing) >= m_affordances.size()
        || size_t(way) >= m_relations.size())
        return false;
    const Affordance &acting = m_affordances[size_t(thing)];
    for (size_t index = 0; index < acting.affects.size(); ++index)
    {
        if (acting.affects[index] != way)
            continue;
        if (index >= acting.affectedAt.size())
            return true;   // known to move it, never yet tried against it
        return stateOf(m_relations[size_t(way)]) != acting.affectedAt[index];
    }
    return false;
}

bool SemanticWorld::affects(AffordanceId thing, RelationId way) const
{
    if (thing == kNoId || way == kNoId
        || size_t(thing) >= m_affordances.size())
        return false;
    for (RelationId known : m_affordances[size_t(thing)].affects)
        if (known == way)
            return true;
    return false;
}

void SemanticWorld::closeAttempt(int possibilities)
{
    if (m_awaiting == kNoId)
        return;
    if (size_t(m_awaiting) < m_affordances.size())
    {
        Affordance &thing = m_affordances[size_t(m_awaiting)];
        thing.lastAttemptOpenedWay = possibilities > m_possibilitiesAtAttempt;
        // What it changed is not worked out here any more.
        //
        // Watching for "this way's state moved" credits an act with every
        // way whose rooms it reshaped, which for a door is every way those
        // rooms touch -- including the ledges nobody can climb, which never
        // become passable however often the door is pushed. The bot then
        // pushes the door for ever trying to open a step. What matters is
        // not that a way moved but that whether it can be gone through
        // moved, and that is a question about a body, which this layer does
        // not have. Whoever does calls noteAffects.
    }
    m_waysAtAttempt.clear();
    m_awaiting = kNoId;
}

} // namespace semantic
