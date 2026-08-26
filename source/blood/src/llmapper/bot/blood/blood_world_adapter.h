//-------------------------------------------------------------------------
// The Blood boundary.
//
// Everything Blood-shaped ends here. This class runs the terrain extraction,
// mints the semantic identifiers, keeps the mapping back to the engine
// entirely to itself, and answers in the vocabulary of semantic/. Nothing
// above it is given a way to ask which container a place or an action came
// from.
//-------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <vector>

#include "fix16.h"
#include "../../../controls.h"
#include "../semantic/actor_control.h"
#include "../semantic/semantic_world.h"
#include "../terrain/terrain_model.h"
#include "blood_interactions.h"
#include "blood_physics.h"
#include "blood_terrain.h"

namespace bloodmap {

// What the debug module is allowed to read back. Nothing in the planning
// path can reach these.
struct RegionProvenance
{
    int mover = -1;   // the stateful geometry this space is made of, or -1
    std::vector<int> sectors;
    std::vector<int> sprites;
    int vertices = 0;
    int holes = 0;
    int barriers = 0;
};

struct AffordanceProvenance
{
    int tag = -1;
    int id = -1;
    // Where the thing itself is, which is not where it is done from. A
    // pickup with no execution domain says nothing about which pickup it
    // was; this says.
    int x = 0;
    int y = 0;
    int z = 0;
    int container = -1;
    int kind = -1;   // the engine's own type for the thing
    int requiredKey = 0;  // the key it wants, by Blood's numbering
    int channel = -1;     // what it transmits on, which is what it does
};

// A piece of the world the mapper found and then did not make a place of.
// Read only by the debug layer; the answer to "why is there floor here with
// no Region on it" has to be visible, or it is unanswerable.
struct RejectedSpace
{
    int sector = -1;
    int freeHeight = 0;
    int supportZ = 0;
    int ceilingZ = 0;
    // What, if anything, would open it again.
    bool push = false;
    bool wallPush = false;
    bool onEnter = false;
    bool remote = false;
    bool locked = false;
    int state = 0;
};

struct AdapterCounters
{
    int terrainBuilds = 0;
    int regionsBuilt = 0;
    int clusters = 0;
    int facesIn = 0;
    int domainQueries = 0;
};

class WorldAdapter
{
public:
    void reset();

    // One observation of the world: geometry, what has been seen of it, and
    // what it currently offers.
    semantic::WorldDelta observe();

    GINPUT toInput(const semantic::MotorCommand &command);
    void noteEngineAction(int hit, int target, bool accepted);

    bool regionProvenance(semantic::RegionId id, RegionProvenance &out) const;
    bool affordanceProvenance(semantic::AffordanceId id,
                              AffordanceProvenance &out) const;
    const std::vector<RejectedSpace> &rejectedSpaces() const
    {
        return m_rejected;
    }
    const AdapterCounters &counters() const { return m_counters; }
    // The action that stands in a way, if the thing the engine named is one
    // this mapper has already made an action of. Answers in the model's own
    // terms; the caller never learns what kind of object it was.
    semantic::AffordanceId actionAt(int obstacle) const;
    // How the last working-out of an act's stances came out.
    void domainAudit(semantic::AffordanceId id, int &candidates,
                     int &inReach, int &accepted) const;
    int obstructionTag(semantic::RelationId id) const
    {
        return size_t(id) < m_relationObstruction.size()
            ? int(m_relationObstruction[size_t(id)]) : 0;
    }
    // The engine's own account of one action's object, as it stands now.
    // Debug only: what a Use actually changed cannot be worked out from the
    // model, because the model is the thing being checked.
    struct ObjectState
    {
        int tag = -1;
        int id = -1;
        int x = 0;
        int y = 0;
        int z = 0;
        int cstat = -1;
        int statnum = -1;
        int state = -1;
        int busy = -1;
        int channel = -1;
        bool solid = false;
    };
    bool objectState(semantic::AffordanceId id, ObjectState &out) const;

private:
    struct RegionRecord
    {
        semantic::Region region;
        uint64_t key = 0;
        std::vector<uint64_t> provenance;
        bool observed = false;
        bool occupied = false;
    };

    // One accepted place to act from, the exact pose behind it, and which
    // face of the thing it was established against.
    struct Stance
    {
        semantic::RegionId region = semantic::kNoId;
        size_t face = 0;
        ExecutionPose pose;
    };

    // One thing that can be done. The same door can be pushed on from
    // either side, and those are two faces of one action, not two actions.
    struct Action
    {
        std::vector<InteractionRecord> faces;
        bool observed = false;
        std::vector<Stance> domain;
        bool domainKnown = false;
        int candidates = 0;   // how the last domain came out; see DomainAudit
        int inReach = 0;
        int accepted = 0;
        // What the thing looked like last tick, to tell a thing that is
        // moving from one that has stopped.
        ObjectState was;
        bool wasKnown = false;
        bool moving = false;
        bool wasPresent = false;

        const InteractionRecord &record() const { return faces.front(); }
        // A door with eight walls is there while any of its walls is, and
        // can be opened while any of them can be. Asking the first face
        // alone makes the whole door vanish because one wall of it was
        // triggered.
        bool anyExists() const
        {
            for (const InteractionRecord &face : faces)
                if (interactionExists(face.key))
                    return true;
            return false;
        }
        bool anyUnlocked() const
        {
            for (const InteractionRecord &face : faces)
                if (interactionUnlocked(face.key))
                    return true;
            return false;
        }
    };

    void rebuildTerrain();
    semantic::RegionId locate(const PhysicalPose &pose) const;
    // Where the actor is. Not the same call as the one above: a body resting
    // on a boundary is in two places as far as geometry can tell, and the
    // answer has to stop flapping between them.
    semantic::RegionId locateActor(const PhysicalPose &pose);
    void refreshExecutionDomain(semantic::AffordanceId id);
    // Which act a surface belongs to: its channel where it has one, so all
    // of a door's walls are one door.
    InteractionKey identityOf(const InteractionRecord &record) const;
    semantic::AffordanceId internAction(const InteractionRecord &record);

    GINPUT steer(int targetX, int targetY) const;
    GINPUT address(const Action &action, uint32_t option) const;

    std::vector<RegionRecord> m_regions;
    std::map<uint64_t, semantic::RegionId> m_regionIds;
    std::vector<semantic::SpatialRelation> m_relations;
    // Per relation, what the mapper found standing in it, in the engine's
    // own naming. Kept because the actions are interned after the geometry
    // is built, so what a thing *is* is not known when the way it shuts is.
    std::vector<uint64_t> m_relationObstruction;
    std::map<uint64_t, semantic::RelationId> m_relationIds;
    std::vector<Action> m_actions;
    std::map<InteractionKey, semantic::AffordanceId> m_actionIndex;
    std::set<int> m_pendingSectors;
    std::vector<RejectedSpace> m_rejected;
    // Engine handles for stateful geometry, interned so that nothing above
    // this layer ever sees an engine index. Position in the vector is the
    // GeometryId; the value is whatever the mapper used to name the thing
    // that moves (for Blood, the xsector/xsprite handle, which the engine
    // keeps pointing at the same mechanism however far its walls travel).
    std::vector<uint64_t> m_geometry;
    // Which channel each piece of stateful geometry answers on, so an act
    // can be matched to what it works without anyone pressing it.
    std::map<uint64_t, int> m_listening;
    // Where each piece of stateful geometry was along its travel a tick ago,
    // so that "still going" can be read the same way it is read for anything
    // else: it is not where it was.
    std::map<uint64_t, int> m_geometryWas;
    void readWiring();
public:
    // Which channel a piece of stateful geometry answers on, or -1.
    // Diagnostic: it is how two controls for one thing can be told from two
    // controls for two things.
    int listensOn(semantic::GeometryId id) const
    {
        if (size_t(id) >= m_geometry.size())
            return -1;
        auto found = m_listening.find(m_geometry[size_t(id)]);
        return found == m_listening.end() ? -1 : found->second;
    }
private:
    semantic::GeometryId internGeometry(uint64_t stateTag);
public:
    // The engine handle a GeometryId was interned from. For the physics
    // layer, which has to put the engine into a configuration to ask about
    // it. Nothing above these two layers may call it.
    uint64_t geometryTag(semantic::GeometryId id) const
    {
        return size_t(id) < m_geometry.size() ? m_geometry[size_t(id)] : 0;
    }
private:
    void readConfigurations(uint64_t stateTag,
                            semantic::StatefulGeometry &out) const;
    // Which stateful geometry, if any, is holding a body up at this pose.
    semantic::GeometryId geometryUnder(const PhysicalPose &pose) const;

    // Where the round of working out execution stances got to.
    size_t m_domainCursor = 0;
    semantic::RegionId m_actorRegion = semantic::kNoId;
    semantic::Vec2 m_lookedFrom;

    uint64_t m_signature = 0;
    bool m_started = false;
    bool m_terrainDirty = true;
    semantic::AffordanceId m_resolved = semantic::kNoId;
    AdapterCounters m_counters;
};

} // namespace bloodmap
