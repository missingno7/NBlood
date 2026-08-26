#include "blood_world_adapter.h"

#include "../../../triggers.h"

#include <algorithm>
#include <cmath>

#include "build.h"
#include "../../../actor.h"
#include "../../../blood.h"
#include "../../../common_game.h"
#include "../../../db.h"
#include "../../../gameutil.h"
#include "../../../globals.h"
#include "../../../player.h"
#include "../../../trig.h"

namespace bloodmap {

namespace {

using semantic::Region;
using semantic::RegionId;
using semantic::RelationId;
using semantic::Vec2;
using semantic::kNoId;

uint64_t mixHash(uint64_t value, uint64_t item)
{
    value ^= item + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
    return value;
}

// The two control inputs this phase needs, in the engine's own terms. Blood
// owns the posture-dependent speed and acceleration; the bot asks for the
// same run control a player holds, and the same action key.
GINPUT moveInput(int forward, int strafe, fix16_t turn, fix16_t look)
{
    GINPUT input = {};
    input.syncFlags.run = 1;
    input.forward = int16_t(std::max(-kMaxMoveInput,
        std::min(kMaxMoveInput, forward)));
    input.strafe = int16_t(std::max(-kMaxMoveInput,
        std::min(kMaxMoveInput, strafe)));
    input.q16turn = turn;
    input.q16mlook = look;
    return input;
}

GINPUT actionInput()
{
    GINPUT input = {};
    input.keyFlags.action = 1;
    return input;
}

uint64_t relationKey(RegionId from, RegionId to,
                     const semantic::Gateway &gateway)
{
    uint64_t hash = 1469598103934665603ULL;
    hash = mixHash(hash, from);
    hash = mixHash(hash, to);
    hash = mixHash(hash, uint64_t(uint32_t(gateway.from.x)));
    hash = mixHash(hash, uint64_t(uint32_t(gateway.from.y)));
    hash = mixHash(hash, uint64_t(uint32_t(gateway.to.x)));
    hash = mixHash(hash, uint64_t(uint32_t(gateway.to.y)));
    return hash;
}

} // namespace

void WorldAdapter::reset()
{
    m_regions.clear();
    m_regionIds.clear();
    m_relations.clear();
    m_relationObstruction.clear();
    m_relationIds.clear();
    m_actions.clear();
    m_actionIndex.clear();
    m_pendingSectors.clear();
    m_rejected.clear();
    m_actorRegion = kNoId;
    m_signature = 0;
    m_started = false;
    m_terrainDirty = true;
    m_resolved = kNoId;
    m_counters = AdapterCounters();
}

// Regions are rebuilt from geometry, never from where the observer happens
// to be standing. Identity comes from the shape of the space, so a region
// whose outline is unchanged keeps its id across a rebuild.
void WorldAdapter::rebuildTerrain()
{
    TerrainSnapshot snapshot;
    extractTerrain(snapshot);
    terrain::BuildResult built;
    terrain::build(snapshot.faces, snapshot.seams, built);

    m_rejected.clear();
    for (size_t index = 0; index < snapshot.faces.size(); ++index)
    {
        const terrain::SupportFace &face = snapshot.faces[index];
        if (face.clearance != semantic::ClearanceClass::None)
            continue;
        const Vec2 middle = semantic::centroidOf(face.outline);
        RejectedSpace rejected;
        rejected.sector = index < snapshot.faceSector.size()
            ? snapshot.faceSector[index] : -1;
        rejected.supportZ = face.support.zAt(middle.x, middle.y);
        rejected.ceilingZ = face.ceiling.zAt(middle.x, middle.y);
        rejected.freeHeight = rejected.supportZ - rejected.ceilingZ;
        if (validSector(rejected.sector) && sector[rejected.sector].extra > 0
            && sector[rejected.sector].extra < kMaxXSectors)
        {
            const XSECTOR &extra = xsector[sector[rejected.sector].extra];
            rejected.push = extra.Push != 0;
            rejected.wallPush = extra.Wallpush != 0;
            rejected.onEnter = extra.Enter != 0;
            rejected.remote = extra.rxID != 0;
            rejected.locked = extra.locked != 0;
            rejected.state = extra.state;
        }
        m_rejected.push_back(rejected);
    }

    ++m_counters.terrainBuilds;
    m_counters.facesIn = int(snapshot.faces.size());
    m_counters.clusters = built.clusters;
    m_counters.regionsBuilt = int(built.regions.size());

    for (RegionRecord &record : m_regions)
        record.region.exists = false;
    std::vector<RegionId> assigned(built.regions.size(), kNoId);
    for (size_t index = 0; index < built.regions.size(); ++index)
    {
        const terrain::BuiltRegion &fresh = built.regions[index];
        RegionId id;
        auto found = m_regionIds.find(fresh.key);
        if (found != m_regionIds.end())
            id = found->second;
        else
        {
            id = RegionId(m_regions.size());
            m_regions.push_back(RegionRecord());
            m_regionIds[fresh.key] = id;
        }
        RegionRecord &record = m_regions[size_t(id)];
        const bool observed = record.observed;
        const bool occupied = record.occupied;
        record.region = fresh.region;
        record.region.id = id;
        record.region.exists = true;
        record.region.observed = observed;
        record.region.occupied = occupied;
        record.key = fresh.key;
        record.region.mover = internGeometry(fresh.stateTag);
        record.provenance = fresh.provenance;
        assigned[index] = id;
    }

    for (semantic::SpatialRelation &relation : m_relations)
        relation.exists = false;
    for (size_t index = 0; index < built.relations.size(); ++index)
    {
        const semantic::SpatialRelation &fresh = built.relations[index];
        if (size_t(fresh.from) >= assigned.size()
            || size_t(fresh.to) >= assigned.size())
            continue;
        semantic::SpatialRelation relation = fresh;

        relation.from = assigned[size_t(fresh.from)];
        relation.to = assigned[size_t(fresh.to)];
        const uint64_t key = relationKey(relation.from, relation.to,
                                         relation.gateway);
        auto found = m_relationIds.find(key);
        if (found != m_relationIds.end())
            relation.id = found->second;
        else
        {
            relation.id = RelationId(m_relations.size());
            m_relations.push_back(relation);
            m_relationIds[key] = relation.id;
        }
        relation.exists = true;
        m_relations[size_t(relation.id)] = relation;
        if (m_relationObstruction.size() <= size_t(relation.id))
            m_relationObstruction.resize(size_t(relation.id) + 1, 0);
        m_relationObstruction[size_t(relation.id)] =
            index < built.obstructions.size() ? built.obstructions[index] : 0;
    }

    // Geometry moved, so every stance an action was going to be taken from
    // has to be established again.
    for (Action &action : m_actions)
        action.domainKnown = false;
}

// Which region the body is actually in: the one whose footprint covers it
// and whose surface is the one holding it up. Two regions over the same
// ground are told apart by that second test and nothing else.
RegionId WorldAdapter::locate(const PhysicalPose &pose) const
{
    if (!pose.valid)
        return kNoId;
    const Vec2 at = { pose.x, pose.y };
    const uint64_t holding = uint64_t(uint32_t(pose.supportHit));

    // Where no region's outline contains the body's centre at all, what the
    // engine says is holding it up settles it.
    //
    // A body standing across the join between two platforms has its centre
    // in neither of them, so nothing below is a candidate and whatever else
    // happens to cover that point wins by default -- on AGTST8 the two
    // sprite bridges cross, so the one overhead at the top of the level
    // claimed a body standing on the one below it. Only a fallback, though:
    // one support can belong to several regions, so where the outlines do
    // answer, they answer.
    auto standingOn = [&]() {
        RegionId onIt = kNoId;
        if (holding == 0)
            return onIt;
        for (const RegionRecord &record : m_regions)
            if (record.region.exists && record.region.holds(holding)
                && onIt == kNoId)
                onIt = record.region.id;
        return onIt;
    };

    RegionId best = kNoId;
    int bestGap = 0;
    int bestDepth = -1;
    for (const RegionRecord &record : m_regions)
    {
        if (!record.region.exists)
            continue;
        // The whole shape, holes and all.
        //
        // Asking only the outer loop makes a region that wraps around others
        // contain every point inside it -- and one of them here wraps around
        // the entire level. So every point in the map was a candidate for
        // it, and when the body stood on a bridge whose own region did not
        // claim the exact centre, the tie below picked whichever floor
        // height was numerically nearest: the top, which the body had never
        // been to, over the pit it was actually above. Everything after that
        // is routing from somewhere the body is not.
        const semantic::Polygon shape = record.region.shape();
        if (!semantic::pointInPolygon(shape, at))
            continue;
        const semantic::Loop &outline = record.region.footprint;
        // The engine says what is holding the body up. A region made of that
        // thing is where the body is, whatever the heights work out to.
        const int gap = record.region.holds(holding)
            ? 0 : std::abs(record.region.support.zAt(at.x, at.y)
                           - pose.supportZ) + 1;
        // A body on the line between two regions is in both. Prefer the one
        // it is furthest inside, so standing on a boundary does not make the
        // answer flap between them.
        // How far inside it is -- measured against everything that bounds
        // this space, the holes included. A region measured only against its
        // outer loop looks enormously deep at a point that is in fact right
        // beside one of the rooms it wraps around.
        int depth = INT32_MAX;
        auto against = [&](const semantic::Loop &loop) {
            for (size_t i = 0; i < loop.size(); ++i)
                depth = std::min(depth, semantic::planarDistance(at,
                    semantic::closestPointOnSegment(at, loop[i],
                        loop[(i + 1) % loop.size()])));
        };
        against(outline);
        for (const semantic::Loop &hole : shape.holes)
            against(hole);
        if (best == kNoId || gap < bestGap
            || (gap == bestGap && depth > bestDepth))
        {
            best = record.region.id;
            bestGap = gap;
            bestDepth = depth;
        }
    }
    // The body is somewhere, so the surface test only chooses between
    // candidates; it never rules the last one out. Reporting no region for a
    // body that is plainly standing in one would be the model contradicting
    // the world.
    if (best == kNoId)
        return standingOn();
    return best;
}

// Being somewhere means staying there until you leave it. The engine works
// this way too -- a sprite keeps its sector until it crosses out of it, and
// never flickers between two. Recomputing the answer from geometry every
// tick does flicker, and then the two sides of one opening take turns
// sending the body at each other.
RegionId WorldAdapter::locateActor(const PhysicalPose &pose)
{
    const RegionId fresh = locate(pose);
    if (m_actorRegion != kNoId && m_actorRegion != fresh
        && size_t(m_actorRegion) < m_regions.size())
    {
        const Region &was = m_regions[size_t(m_actorRegion)].region;
        // Its whole shape, holes and all. A region that wraps around other
        // rooms -- a hall with doors off it is one, and one of them here has
        // eleven -- contains every one of them inside its outer loop, so
        // asking only that loop says the body never left the hall no matter
        // which room it walked into. It then routes through a hall it is not
        // in, from a point that is not in it, and every leg comes back with
        // nowhere to go.
        if (was.exists
            && semantic::pointInPolygon(was.shape(), Vec2{ pose.x, pose.y })
            && bodyRestsOn(was, pose, liveBody()))
            return m_actorRegion;   // it has not left
    }
    m_actorRegion = fresh;
    return fresh;
}

void WorldAdapter::domainAudit(semantic::AffordanceId id, int &candidates,
                               int &inReach, int &accepted) const
{
    candidates = inReach = accepted = -1;
    if (size_t(id) >= m_actions.size())
        return;
    candidates = m_actions[size_t(id)].candidates;
    inReach = m_actions[size_t(id)].inReach;
    accepted = m_actions[size_t(id)].accepted;
}

void WorldAdapter::refreshExecutionDomain(semantic::AffordanceId id)
{
    if (size_t(id) >= m_actions.size())
        return;
    Action &action = m_actions[size_t(id)];
    action.domainKnown = true;
    action.domain.clear();
    action.candidates = 0;
    action.inReach = 0;
    action.accepted = 0;
    ++m_counters.domainQueries;

    std::vector<ExecutionPose> domain;
    for (size_t face = 0; face < action.faces.size(); ++face)
    {
    // Only from a surface that is actually there and actually openable. A
    // door's walls share a channel but not always a lock or a lifetime.
    if (!interactionUnlocked(action.faces[face].key))
        continue;
    bloodmap::DomainAudit audit;
    executionDomain(action.faces[face], domain, &audit);
    action.candidates += audit.candidates;
    action.inReach += audit.inReach;
    action.accepted += audit.accepted;
    for (const ExecutionPose &candidate : domain)
    {
        // Every stance the world accepts. Which of them to use is a
        // question about what the actor can reach and where it fits, and it
        // is not answered here: a switch beside a shut door can be pressed
        // from either side, and only one of those sides may be reachable.
        //
        // Keeping one per region was answering it here, badly. A region is
        // not one piece of free space, and the one kept was whichever came
        // nearest the actor when the domain was first worked out -- so the
        // exit switch on AGTST18 was offered from a spot a hundred units off
        // a wall, inside the body's own width of it, and the bot spent the
        // rest of the level walking into that wall. The engine accepted
        // seven other poses for the same switch; nothing here has any
        // business throwing them away.
        const RegionId region = locate(candidate.pose);
        if (region == kNoId)
            continue;
        Stance stance;
        stance.region = region;
        stance.face = face;
        stance.pose = candidate;
        action.domain.push_back(stance);
    }
    }
}

bool WorldAdapter::objectState(semantic::AffordanceId id,
                               ObjectState &out) const
{
    if (size_t(id) >= m_actions.size())
        return false;
    const InteractionRecord &record = m_actions[size_t(id)].record();
    out = ObjectState();
    out.tag = record.key.tag;
    out.id = record.key.id;
    out.channel = interactionChannel(record.key);
    if (record.key.tag == 0 && validWall(record.key.id))
    {
        const walltype &face = wall[record.key.id];
        out.x = face.x;
        out.y = face.y;
        out.cstat = face.cstat;
        out.solid = (face.cstat & 1) != 0 || face.nextsector < 0;
        if (face.extra > 0 && face.extra < kMaxXWalls)
        {
            out.state = xwall[face.extra].state;
            out.busy = xwall[face.extra].busy;
        }
        return true;
    }
    if (validSprite(record.key.id))
    {
        const spritetype &thing = sprite[record.key.id];
        out.x = thing.x;
        out.y = thing.y;
        out.z = thing.z;
        out.cstat = thing.cstat;
        out.statnum = thing.statnum;
        out.solid = (thing.cstat & 1) != 0;
        if (thing.extra > 0 && thing.extra < kMaxXSprites)
        {
            out.state = xsprite[thing.extra].state;
            out.busy = xsprite[thing.extra].busy;
        }
        return true;
    }
    return false;
}

// Every configuration the engine will rest this geometry in.
//
// Blood stores the two ends of a motion and interpolates between them, so
// the ends are the rest configurations and everything between is transit.
// They are read out as heights and nothing more: which of them is "open" or
// "up" is not written down here, because it is not a fact about the
// geometry -- it is a fact about what walking turns out to be possible in
// each, and that is derived, not declared.
void WorldAdapter::readConfigurations(uint64_t stateTag,
                                      semantic::StatefulGeometry &out) const
{
    out.configurations.clear();
    if (stateTag == 0 || stateTag >= kMaxXSectors)
        return;
    const XSECTOR &extra = xsector[stateTag];
    // Two ends. Stored as an ordered list so that nothing above depends on
    // there being exactly two of them.
    semantic::GeometryConfiguration first;
    first.supportZ = extra.offFloorZ;
    first.ceilingZ = extra.offCeilZ;
    semantic::GeometryConfiguration second;
    second.supportZ = extra.onFloorZ;
    second.ceilingZ = extra.onCeilZ;
    out.configurations.push_back(first);
    out.configurations.push_back(second);
    // Whether the two ends are anything different.
    //
    // Heights say so for something that moves up and down. For something
    // that moves in the plane the heights are identical at both ends and the
    // difference is where its walls are -- which Blood keeps as a pair of
    // marker sprites, so two markers that are not the same place, or not the
    // same facing, are two configurations.
    //
    // Saying so is what makes the difference between a thing whose motion is
    // described and one whose motion is not, and those lead to opposite
    // decisions everywhere: "the same" and "not described" must never be
    // confused.
    out.configurationsDiffer = first.supportZ != second.supportZ
        || first.ceilingZ != second.ceilingZ;
    const int owner = extra.reference;
    if (validSector(owner) && movesInThePlane(sector[owner].type)
        && validSprite(extra.marker0))
    {
        const spritetype &from = sprite[extra.marker0];
        const spritetype *to = validSprite(extra.marker1)
            ? &sprite[extra.marker1] : nullptr;
        const bool turns = sector[owner].type == kSectorRotate
            || sector[owner].type == kSectorRotateMarked;
        if (turns)
            out.configurationsDiffer = out.configurationsDiffer
                || from.ang != 0;
        if (to)
        {
            out.configurationsDiffer = out.configurationsDiffer
                || to->x != from.x || to->y != from.y
                || to->ang != from.ang;
            second.shift = { to->x - from.x, to->y - from.y };
            out.configurations[1] = second;
        }
    }
    out.state = extra.state ? 1u : 0u;
    // In transit, not "not at zero".
    //
    // busy is where the motion has got to, not whether it is happening: it
    // runs from 0 at one end to 0x10000 at the other and stays there. So
    // "busy != 0" reads as permanently moving for anything resting in its
    // second configuration, and every plan that waits for it to stop waits
    // for ever. Between the two ends is the only thing that means moving.
    out.moving = extra.busy != 0 && extra.busy != 0x10000;
}

// Which channel each piece of stateful geometry listens on.
//
// Read from the world, once per build, exactly like the shape of a room. It
// is the other half of the wiring an act's own channel is one end of.
void WorldAdapter::readWiring()
{
    m_listening.clear();
    for (int sectorId = 0; sectorId < numsectors; ++sectorId)
    {
        const uint64_t tag = moverTagOf(sectorId);
        if (tag == 0 || tag >= kMaxXSectors)
            continue;
        m_listening[tag] = xsector[tag].rxID;
    }
}

semantic::GeometryId WorldAdapter::internGeometry(uint64_t stateTag)
{
    if (stateTag == 0)
        return semantic::kNoId;   // ground that does not move
    for (size_t index = 0; index < m_geometry.size(); ++index)
        if (m_geometry[index] == stateTag)
            return semantic::GeometryId(index);
    m_geometry.push_back(stateTag);
    return semantic::GeometryId(m_geometry.size() - 1);
}

// What is holding this pose up, if it is something that can move.
//
// Asked of the surface the engine actually reported, not of the region the
// model thinks the body is in: when geometry moves, which Region the space
// is called can change under a body that has not moved at all, and the point
// of this identity is that it does not.
semantic::GeometryId WorldAdapter::geometryUnder(
    const PhysicalPose &pose) const
{
    if (!pose.valid)
        return semantic::kNoId;
    int wallId = -1;
    int spriteId = -1;
    splitObstacle(pose.supportHit, wallId, spriteId);
    uint64_t tag = 0;
    if (spriteId >= 0 && validSprite(spriteId)
        && sprite[spriteId].extra > 0)
        tag = uint64_t(sprite[spriteId].extra);
    else if (validSector(pose.sector))
        tag = moverTagOf(pose.sector);
    if (tag == 0)
        return semantic::kNoId;
    for (size_t index = 0; index < m_geometry.size(); ++index)
        if (m_geometry[index] == tag)
            return semantic::GeometryId(index);
    return semantic::kNoId;
}

semantic::AffordanceId WorldAdapter::actionAt(int obstacle) const
{
    int wallId = -1;
    int spriteId = -1;
    splitObstacle(obstacle, wallId, spriteId);
    // The tags are the mapper's own: a wall that can be pushed, a sprite
    // that can be pushed, a sprite that can be taken. Whichever of them this
    // thing was made an action under is the action standing in the way.
    static const int kWallTags[] = { 0 };
    static const int kSpriteTags[] = { 3, kPickupTag };
    if (wallId >= 0)
        for (int tag : kWallTags)
        {
            auto found = m_actionIndex.find(InteractionKey{ tag, wallId });
            if (found != m_actionIndex.end())
                return found->second;
        }
    if (spriteId >= 0)
        for (int tag : kSpriteTags)
        {
            auto found = m_actionIndex.find(InteractionKey{ tag, spriteId });
            if (found != m_actionIndex.end())
                return found->second;
        }
    return semantic::kNoId;
}

// One thing to do, however many surfaces it has.
//
// A Blood door is a sector's worth of walls and every one of them can be
// pushed. They are not eight doors. They all send the same command on the
// same channel, so pushing any one of them is the same act with the same
// consequence, and the only difference between them is where you stand.
// Treating them as eight was why one door on AGTST18 was pushed twelve
// times: each wall was untried in its own right, so each one was worth a
// trip, and the door swung back and forth while the bot worked through them.
//
// The channel is the identity. Where there is no channel -- most pickups,
// and anything wired to nothing -- the object is its own identity, as
// before. Tag 0 stays out of it: in Blood that is "sends nothing".
InteractionKey WorldAdapter::identityOf(const InteractionRecord &record) const
{
    if (record.kind != semantic::ActionKind::Use)
        return record.key;
    const int channel = interactionChannel(record.key);
    if (channel <= 0)
        return record.key;
    return InteractionKey{ kChannelTag, channel };
}

semantic::AffordanceId WorldAdapter::internAction(
    const InteractionRecord &record)
{
    const InteractionKey identity = identityOf(record);
    auto found = m_actionIndex.find(identity);
    if (found != m_actionIndex.end())
    {
        // Remember this surface too, so that a thing found standing in a
        // way can still be traced back to the act that moves it.
        m_actionIndex[record.key] = found->second;
        Action &action = m_actions[size_t(found->second)];
        for (InteractionRecord &known : action.faces)
            if (known.facingWall == record.facingWall)
            {
                known = record;
                return found->second;
            }
        // Another surface of the same thing. A door pushed from the far
        // side is the same door, and the stances on that side are part of
        // the same execution domain.
        action.faces.push_back(record);
        action.domainKnown = false;
        return found->second;
    }
    const semantic::AffordanceId id =
        semantic::AffordanceId(m_actions.size());
    Action action;
    action.faces.push_back(record);
    m_actions.push_back(action);
    m_actionIndex[identity] = id;
    m_actionIndex[record.key] = id;
    return id;
}

semantic::WorldDelta WorldAdapter::observe()
{
    semantic::WorldDelta delta;
    if (!gGameStarted || !gMe || !gMe->pSprite || !gMe->pXSprite)
        return delta;
    if (!m_started)
    {
        reset();
        m_started = true;
        m_terrainDirty = true;
    }

    const BodyShape body = liveBody();
    const PhysicalPose actorPose = livePose();
    delta.actor.position = { actorPose.x, actorPose.y, actorPose.z };
    delta.actor.alive = gMe->pXSprite->health > 0;

    const uint64_t signature = worldSignature();
    delta.changed = m_signature != 0 && signature != m_signature;
    delta.revision = signature;
    m_signature = signature;
    delta.settling = worldSettling();

    // Geometry in motion reports a change every tick it moves. Rebuilding
    // against a transient would be measuring nothing, so the affected
    // regions are noted and the world is re-derived once it comes to rest.
    std::vector<int> changed;
    collectChangedSectors(changed);
    m_pendingSectors.insert(changed.begin(), changed.end());
    // Which sectors changed is the whole of the question. A single number
    // for the whole map answers "did anything anywhere move", and anything
    // anywhere is always moving, so the terrain was rebuilt from nothing
    // every few seconds for the whole run.
    if (!delta.settling && (!m_pendingSectors.empty() || m_terrainDirty))
    {
        rebuildTerrain();
        m_pendingSectors.clear();
        m_terrainDirty = false;
    }

    const RegionId here = locateActor(actorPose);
    delta.actor.region = here;
    // What is under the body, named so that it stays the same thing while it
    // moves. The Region underfoot can be re-derived into a differently
    // decomposed space by the very motion the body is riding; this does not.
    delta.actor.supportedBy = geometryUnder(actorPose);
    // What every piece of stateful geometry can be, and what it is now.
    readWiring();
    delta.geometry.clear();
    std::map<semantic::GeometryId, bool> stillGoing;
    for (size_t index = 0; index < m_geometry.size(); ++index)
    {
        semantic::StatefulGeometry piece;
        piece.id = semantic::GeometryId(index);
        readConfigurations(m_geometry[index], piece);
        if (piece.configurations.empty())
            continue;
        // Between the ends of its travel is one way to be still going. The
        // other is to have arrived this very tick: Blood resolves a use
        // inside the tick it happens, and one of these does not accept a use
        // until it has finished arriving, so a press on the tick it lands is
        // a press the world drops. Reading "it is not where it was a tick
        // ago" covers both, and needs no clock.
        const uint64_t tag = m_geometry[index];
        const int along = tag < kMaxXSectors ? int(xsector[tag].busy) : 0;
        auto before = m_geometryWas.find(tag);
        if (before != m_geometryWas.end() && before->second != along)
            piece.moving = true;
        m_geometryWas[tag] = along;
        stillGoing[piece.id] = piece.moving;
        delta.geometry.push_back(piece);
    }
    if (here != kNoId)
    {
        RegionRecord &record = m_regions[size_t(here)];
        record.occupied = true;
        record.observed = true;
        record.region.occupied = true;
        record.region.observed = true;
    }

    // Seeing a region is separate from it existing: this only ever turns
    // observation on, so looking away never takes knowledge back.
    //
    // Looking costs a ray for every region not yet seen, so it is worth
    // doing when there is something new to see: the body has moved, or the
    // world has. Standing still and looking again at the same place from the
    // same place cannot answer differently.
    const bool worthLooking = m_terrainDirty
        || semantic::planarDistance({ actorPose.x, actorPose.y },
                                    m_lookedFrom) > body.radius;
    if (worthLooking)
        m_lookedFrom = { actorPose.x, actorPose.y };
    for (RegionRecord &record : m_regions)
    {
        if (!worthLooking)
            break;
        if (record.observed || !record.region.exists)
            continue;
        const semantic::Vec3 anchor = record.region.anchor();
        const int eyeZ = anchor.z - body.footOffset - body.eyeAbove;
        int16_t sectorId = int16_t(actorPose.sector);
        updatesectorz(anchor.x, anchor.y, eyeZ, &sectorId);
        if (!validSector(sectorId))
            continue;
        if (!eyeCanSee(anchor.x, anchor.y, eyeZ, sectorId))
            continue;
        record.observed = true;
        record.region.observed = true;
    }

    std::vector<InteractionRecord> seen;
    discoverInteractions(seen);
    for (const InteractionRecord &record : seen)
    {
        const semantic::AffordanceId id = internAction(record);
        // Seeing it is not what makes it exist, and looking away does not
        // unsee it.
        if (record.visible)
            m_actions[size_t(id)].observed = true;
    }

    // One action's stances worked out per tick, and the next one next tick.
    //
    // Going round rather than starting from the beginning each time. Every
    // rebuild of the geometry marks all of them unknown, and a door moving
    // rebuilds the geometry, so starting from the beginning means the ones
    // at the front are done over and over and the ones behind them are never
    // reached at all. AGTST18's exit switch is action sixty-five of
    // eighty-eight: its stances were worked out once, early, and it was
    // still holding that answer twenty minutes later -- one stance, against
    // a wall, from a moment when the engine happened to accept only that
    // one. The bot stood next to the switch that ends the level and had
    // nowhere it could press it from.
    if (!m_actions.empty())
    {
        for (size_t step = 0; step < m_actions.size(); ++step)
        {
            const size_t index = (m_domainCursor + step) % m_actions.size();
            Action &action = m_actions[index];
            if (action.domainKnown || !action.anyUnlocked())
                continue;
            refreshExecutionDomain(semantic::AffordanceId(index));
            m_domainCursor = (index + 1) % m_actions.size();
            break;
        }
    }

    // What the mapper found standing in each way, said as one of this
    // world's own actions. Resolved here rather than when the geometry was
    // built, because a thing only becomes an action once it has been seen
    // for what it is, and that happens above.
    for (semantic::SpatialRelation &relation : m_relations)
    {
        if (size_t(relation.id) >= m_relationObstruction.size())
            continue;
        const uint64_t tag = m_relationObstruction[size_t(relation.id)];
        relation.obstruction = tag == 0 ? semantic::kNoId
                                        : actionAt(int(tag));
    }

    for (const RegionRecord &record : m_regions)
        delta.regions.push_back(record.region);
    delta.relations = m_relations;
    for (size_t index = 0; index < m_actions.size(); ++index)
    {
        Action &action = m_actions[index];
        semantic::Affordance affordance;
        affordance.id = semantic::AffordanceId(index);
        affordance.action = action.record().kind;
        affordance.target = { action.record().x, action.record().y,
                              action.record().targetZ };
        affordance.exists = action.anyExists();
        affordance.observed = action.observed;
        affordance.executable = affordance.exists && action.anyUnlocked();
        // Is this thing moving? Asked by looking at it twice. The engine
        // publishes no "still going" flag that means the same for a door, a
        // sector and a sprite, but all three are somewhere and in some state,
        // and a thing whose somewhere or state is not what it was a tick ago
        // is on its way to being something else.
        {
            ObjectState now;
            if (objectState(affordance.id, now))
            {
                action.moving = action.wasKnown
                    && (now.x != action.was.x || now.y != action.was.y
                        || now.z != action.was.z
                        || now.state != action.was.state
                        || now.busy != action.was.busy);
                action.was = now;
                action.wasKnown = true;
            }
            else
            {
                action.moving = false;
                action.wasKnown = false;
            }
        }
        // What this act addresses, from the world's own wiring: it speaks on
        // one channel, and whatever stateful geometry listens on that
        // channel is what it works. Read here and published as GeometryIds,
        // so nothing above this line knows there is a channel at all.
        //
        // Without it an act can only be understood by doing it, and two
        // controls for one door are two mysteries rather than one fact.
        {
            const int channel = bloodmap::interactionChannel(
                action.record().key);
            if (channel > 0)
                for (size_t index = 0; index < m_geometry.size(); ++index)
                {
                    auto heard = m_listening.find(m_geometry[index]);
                    if (heard != m_listening.end() && heard->second == channel)
                        affordance.commands.push_back(
                            semantic::GeometryId(index));
                }
        }
        // Still going means what it works is still going, not that the thing
        // it is drawn on moved. A switch on a wall does not stir while the
        // door it opens travels, so watching the switch says a door that is
        // halfway shut is at rest -- and these doors do not accept a use
        // until they have finished, so the press is dropped and the world
        // looks as though it refused.
        affordance.settling = action.moving;
        for (semantic::GeometryId which : affordance.commands)
        {
            auto going = stillGoing.find(which);
            affordance.settling = affordance.settling
                || (going != stillGoing.end() && going->second);
        }
        if (affordance.executable)
            for (const Stance &stance : action.domain)
            {
                semantic::ExecutionOption option;
                option.region = stance.region;
                option.at = { stance.pose.pose.x, stance.pose.pose.y,
                              stance.pose.pose.supportZ };
                affordance.domain.push_back(option);
            }
        // Something taken is resolved by having gone; there is no engine
        // callback for it, so the mapper notices it stopped being offered.
        if (action.observed && action.wasPresent && !affordance.exists
            && m_resolved == kNoId)
            m_resolved = affordance.id;
        action.wasPresent = affordance.exists;
        delta.affordances.push_back(affordance);
    }

    // Anything still to be established? Until the mapper has answered where
    // each action can be taken from, "nothing to do" is not a conclusion.
    delta.establishing = m_terrainDirty || !m_pendingSectors.empty();
    for (const Action &action : m_actions)
        if (!action.domainKnown && action.anyUnlocked())
            delta.establishing = true;

    delta.resolvedAffordance = m_resolved;
    m_resolved = kNoId;
    return delta;
}

void WorldAdapter::noteEngineAction(int hit, int target, bool)
{
    const InteractionKey key = { hit, target };
    auto found = m_actionIndex.find(key);
    if (found != m_actionIndex.end())
        m_resolved = found->second;
}

GINPUT WorldAdapter::steer(int targetX, int targetY) const
{
    if (!gMe || !gMe->pSprite)
        return GINPUT();
    const spritetype *actor = gMe->pSprite;
    const int targetAngle = getangle(targetX - actor->x, targetY - actor->y);
    const fix16_t turn = fix16_from_int(
        std::max(-48, std::min(48, DANGLE(targetAngle, actor->ang))));
    const int remaining = planarDistance(actor->x, actor->y, targetX, targetY);
    // Full control, always. Easing off near the target was a way of not
    // overshooting it, but Blood's input is acceleration against drag, so a
    // fraction of it is a speed the body settles at -- and a small enough
    // fraction is a speed drag cancels, which is a body that creeps to a
    // halt short of where it was going and stays there. What stops the
    // overshoot is the braking term below, which is the momentum itself.
    const int magnitude = kMaxMoveInput;

    int movementAngle = targetAngle;
    const int spriteId = actor->index;
    const double velocityX = double(xvel[spriteId]);
    const double velocityY = double(yvel[spriteId]);
    const double velocityLength = std::sqrt(
        velocityX * velocityX + velocityY * velocityY);
    if (remaining > 0 && velocityLength > 1.0)
    {
        // Blood input is acceleration, not velocity, so what is asked for is
        // a push against the momentum there already is. Two different jobs,
        // and they were one.
        //
        // Travelling: only the sideways momentum is wrong. Cancelling the
        // forward part too swings the push either side of the line, and the
        // body wanders -- two or three turn reversals a second, all the way
        // down a corridor.
        //
        // Arriving: all of it is wrong. What is wanted is to stop on a spot
        // the width of the body, and momentum carried into that spot goes
        // straight through it; the body then drives at a stance it can never
        // settle on and the leg is reported as going nowhere.
        //
        // Which job this is, is how far there is left to go, measured in
        // bodies. The width is the engine's own for this body, not a number
        // chosen here.
        const double towardX = double(targetX - actor->x) / remaining;
        const double towardY = double(targetY - actor->y) / remaining;
        const int width = actor->clipdist << 2;
        const bool arriving = remaining < width * 4;
        double controlX = 0.0;
        double controlY = 0.0;
        if (arriving)
        {
            controlX = 2.0 * towardX - velocityX / velocityLength;
            controlY = 2.0 * towardY - velocityY / velocityLength;
        }
        else
        {
            const double along = velocityX * towardX + velocityY * towardY;
            const double push = std::max(1.0, velocityLength);
            controlX = towardX * push - (velocityX - along * towardX);
            controlY = towardY * push - (velocityY - along * towardY);
        }
        movementAngle = getangle(int(std::lround(controlX)),
                                 int(std::lround(controlY)));
    }
    // Blood resolves forward and strafe in the live facing basis, so the
    // desired world direction is expressed in that basis here.
    const int delta = DANGLE(movementAngle, actor->ang);
    return moveInput(mulscale30(magnitude, Cos(delta)),
                     -mulscale30(magnitude, Sin(delta)), turn, 0);
}

GINPUT WorldAdapter::address(const Action &action, uint32_t option) const
{
    if (!gMe || !gMe->pSprite)
        return GINPUT();
    const Stance &stance = action.domain[size_t(option)];
    const InteractionRecord &face = action.faces[stance.face];
    int target = -1;
    int extra = -1;
    const int hit = ActionScanPreview(gMe, &target, &extra);
    if (interactionMatches(face.key, hit, target))
        return actionInput();

    const PhysicalPose pose = livePose();
    // The domain was established at one exact stance, and a body never comes
    // to rest on exactly one point. So the question is put to the engine
    // again from where the body actually is: is there a way to look from
    // here that resolves? If there is, look that way. If there is not, go to
    // the stance the engine did agree to.
    int desiredAngle = 0;
    int desiredLook = 0;
    if (!aimFrom(face, pose, desiredAngle, desiredLook))
        return steer(stance.pose.pose.x, stance.pose.pose.y);
    const fix16_t turn = fix16_from_int(std::max(-48,
        std::min(48, DANGLE(desiredAngle, gMe->pSprite->ang))));
    const int lookDelta = desiredLook - fix16_to_int(gMe->q16look);
    const fix16_t look = fix16_from_int(
        std::max(-32, std::min(32, lookDelta / 8)));
    return moveInput(0, 0, turn, look);
}

GINPUT WorldAdapter::toInput(const semantic::MotorCommand &command)
{
    if (command.move)
        return steer(command.moveToward.x, command.moveToward.y);
    if (command.act && command.address != kNoId
        && size_t(command.address) < m_actions.size()
        && size_t(command.option)
               < m_actions[size_t(command.address)].domain.size())
        return address(m_actions[size_t(command.address)], command.option);
    return GINPUT();
}

bool WorldAdapter::regionProvenance(RegionId id, RegionProvenance &out) const
{
    if (size_t(id) >= m_regions.size())
        return false;
    const RegionRecord &record = m_regions[size_t(id)];
    out.sectors.clear();
    out.sprites.clear();
    for (uint64_t entry : record.provenance)
    {
        if (entry & 0x100000000ull)
            out.sprites.push_back(int(entry & 0xffffffffull));
        else
            out.sectors.push_back(int(entry));
    }
    out.mover = record.region.mover == semantic::kNoId
        ? -1 : int(record.region.mover);
    out.vertices = int(record.region.footprint.size());
    out.holes = int(record.region.holes.size());
    out.barriers = int(record.region.barriers.size());
    return true;
}

bool WorldAdapter::affordanceProvenance(semantic::AffordanceId id,
                                        AffordanceProvenance &out) const
{
    if (size_t(id) >= m_actions.size())
        return false;
    const InteractionRecord &record = m_actions[size_t(id)].record();
    out.tag = record.key.tag;
    out.id = record.key.id;
    out.x = record.x;
    out.y = record.y;
    out.z = record.targetZ;
    out.container = record.referenceSector;
    out.kind = interactionKind(record.key);
    out.requiredKey = record.requiredKey;
    out.channel = interactionChannel(record.key);
    return true;
}

} // namespace bloodmap
