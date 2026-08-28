//-------------------------------------------------------------------------
// Blood interaction discovery and execution domains.
//
// Blood offers the same player action through three unrelated containers.
// This file is where that stops being three things: everything below turns
// into one record with one question attached to it -- from which poses does
// the engine's own action scan resolve to it.
//-------------------------------------------------------------------------
#pragma once

#include <vector>

#include "../semantic/semantic_world.h"
#include "blood_physics.h"

namespace bloodmap {

// The engine object an interaction belongs to. The tag is Blood's own
// ActionScan result code, so a discovered interaction and a resolved one
// compare directly. Nothing above the adapter ever sees this type.
struct InteractionKey
{
    int tag = -1;
    int id = -1;

    bool operator<(const InteractionKey &other) const
    {
        return tag != other.tag ? tag < other.tag : id < other.id;
    }
    bool operator==(const InteractionKey &other) const
    {
        return tag == other.tag && id == other.id;
    }
};

// The engine's own ActionScan codes are 0, 3 and 6. A pickup is not
// something ActionScan can ever resolve to, so it is tagged apart -- below
// the boundary, where the difference is still about engine mechanics and not
// yet about what the bot can do.
constexpr int kPickupTag = 100;

// The tag an act carries when its identity is the channel it sends on
// rather than the surface it was found on. Chosen clear of the engine's own
// tags (0 wall, 3 sprite, 6 sector, and the pickup tag).
constexpr int kChannelTag = 9;

struct InteractionRecord
{
    InteractionKey key;
    semantic::ActionKind kind = semantic::ActionKind::Use;
    bool visible = false;   // seen from where the actor is standing now
    int x = 0;
    int y = 0;
    int targetZ = 0;
    int referenceSector = -1;
    int facingWall = -1;   // the surface it was seen on, when it had one
    // Which way the surface it is on faces, as a Build angle, or -1. A
    // switch mounted flat on a wall is pressed from in front of it, and
    // that is the one direction worth trying first.
    int facingAngle = -1;
    int requiredKey = 0;
};

struct ExecutionPose
{
    PhysicalPose pose;
    int angle = 0;
    int look = 0;
};

// Everything in the loaded world that offers an action, with a note of
// whether it can be seen from where the actor is standing.
//
// The world's geometry is read whole, not discovered by looking at it, and
// what can be done in it is read the same way: a model that knew about the
// floor of a room but not the switch on its wall would be inconsistent with
// itself. Whether a thing has actually been seen is recorded separately and
// is what `observed` means.
void discoverInteractions(std::vector<InteractionRecord> &out);

bool interactionExists(const InteractionKey &key);
// The engine's own type number for whatever this is. Debug only.
int interactionKind(const InteractionKey &key);
// What pressing this transmits on. A level ends because something sends on
// the channel the level's end is listening to, so this is what says which
// button finishes the map. Debug only.
int interactionChannel(const InteractionKey &key);
bool interactionUnlocked(const InteractionKey &key);
bool interactionMatches(const InteractionKey &key, int hit, int target);

// Poses from which the engine's real action scan resolves to this
// interaction, best first. This is the execution domain: it is the engine's
// answer, not a distance test.
// How a domain came out, for reading a withheld act rather than guessing.
struct DomainAudit
{
    int candidates = 0;   // stances the geometry offered
    int inReach = 0;      // ... near enough for the engine to reach from
    int accepted = 0;     // ... that the engine's own scan resolved from
};

void executionDomain(const InteractionRecord &record,
                     std::vector<ExecutionPose> &out,
                     DomainAudit *audit = nullptr);

// Whether the engine would resolve an action from exactly this pose.
bool resolvesFrom(const InteractionKey &key, const PhysicalPose &pose,
                  int angle, int look);

// Where to look from this pose for the engine to resolve this action, if
// anywhere. Asked of the engine, from where the body actually is, so a body
// that came to rest a little off the stance the domain was established at
// still gets a true answer rather than a stale one.
bool aimFrom(const InteractionRecord &record, const PhysicalPose &pose,
             int &angle, int &look);

// The aim that points the live eye at a target from a pose.
int aimAngle(const PhysicalPose &pose, int targetX, int targetY);
int aimLook(const PhysicalPose &pose, int targetX, int targetY, int targetZ);

} // namespace bloodmap
