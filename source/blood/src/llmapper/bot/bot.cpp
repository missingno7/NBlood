//-------------------------------------------------------------------------
// LLMapper autonomous Blood playtest bot.
//
// This is deliberately a small in-process vertical slice. It keeps a
// discovered graph, plans only over observations, and emits NDJSON telemetry
// plus a trajectory while driving the ordinary GINPUT path.
//-------------------------------------------------------------------------
#include "bot.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "build.h"
#include "common_game.h"
#include "config.h"
#include "db.h"
#include "demo.h"
#include "gameutil.h"
#include "globals.h"
#include "levels.h"
#include "network.h"
#include "player.h"
#include "trig.h"
#include "triggers.h"

enum TraversalCapability
{
    kTraversalUnknown,
    kTraversalWalkable,
    kTraversalJumpable,
    kTraversalCrouchable,
    kTraversalDropSafe,
    kTraversalInteractionBlocked,
    kTraversalCurrentlyUnavailable,
};

namespace
{
constexpr int kDefaultTimeoutSeconds = 30 * 60;
constexpr int kDefaultStallSeconds = 45;
constexpr int kObservationPeriod = 4;
constexpr int kTrajectoryPeriod = 8;
constexpr int kMovementStuckTicks = 2 * kTicsPerSec;
constexpr int kJumpCooldownTicks = 2 * kTicsPerSec;
constexpr int kMaxJumpAttemptsPerTarget = 3;
constexpr int kPlayerPassageWidth = 384;
constexpr int kPlayerStandingClearance = 5632;
constexpr int kMaxWalkableStep = 4096;
constexpr int kActionScanRange = 1024;
constexpr int kActionApproachRange = 2048;
constexpr int kUseStopRange = kActionApproachRange;
constexpr int kInteractionTimeoutTicks = 6 * kTicsPerSec;

enum ObjectKind
{
    kObjectEnemy,
    kObjectKey,
    kObjectInteractive,
    kObjectPickup,
};

struct VisibleObject
{
    int sprite = -1;
    int sector = -1;
    int type = 0;
    int x = 0;
    int y = 0;
    int z = 0;
    ObjectKind kind = kObjectPickup;
};

struct Portal
{
    int wall = -1;
    int from = -1;
    int to = -1;
    int x = 0;
    int y = 0;
    int z = 0;
    int floorZ = 0;
    int ceilingZ = 0;
    int fromFloorZ = 0;
    int fromCeilingZ = 0;
    int toFloorZ = 0;
    int toCeilingZ = 0;
    int crouchClearance = 0;
    int jumpRiseLimit = 0;
    int dropLimit = 0;
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
    int key = 0;
    int openingWidth = 0;
    int floorDelta = 0;
    int clearance = 0;
    bool wallPush = false;
    bool sectorPush = false;
    bool sectorPushCurrent = false;
    bool directUse = false;
    bool visible = false;
    bool localGeometry = false;
    bool walkable = false;
    bool jumpable = false;
    bool crouchable = false;
    bool dropSafe = false;
    bool traversable = false;
    bool locked = false;
    bool interactionAffordance = false;
    bool currentlyAvailable = false;
    TraversalCapability capability = kTraversalUnknown;
    int unavailableReason = 0;
    int wallState = -1;
    int wallBusy = 0;
    int sectorState = -1;
    int sectorBusy = 0;
};

struct Observation
{
    int tick = 0;
    int sector = -1;
    int x = 0;
    int y = 0;
    int z = 0;
    int angle = 0;
    int health = 0;
    bool exitHere = false;
    int localPortalCount = 0;
    int localJumpableCount = 0;
    int localMaxRise = 0;
    int localMaxDrop = 0;
    int localClearance = 0;
    int localSectorExtra = 0;
    int localSectorState = 0;
    int localSectorBusy = 0;
    int playerZVelocity = 0;
    std::vector<int> visibleSectors;
    std::vector<Portal> portals;
    std::vector<VisibleObject> objects;
};

static int wrapAngle(int angle)
{
    angle &= 2047;
    return angle;
}

static int angleDelta(int target, int current)
{
    int delta = wrapAngle(target) - wrapAngle(current);
    if (delta > 1024)
        delta -= 2048;
    if (delta < -1024)
        delta += 2048;
    return delta;
}

static int distance2(int x1, int y1, int x2, int y2)
{
    const int64_t dx = int64_t(x2) - x1;
    const int64_t dy = int64_t(y2) - y1;
    return int(std::min<int64_t>(INT32_MAX, dx * dx + dy * dy));
}

static bool inRange(int value, int low, int high)
{
    return value >= low && value < high;
}

static bool isKeyType(int type)
{
    return type >= kItemKeyBase && type < kItemKeyMax;
}

static bool isEnemyType(int type)
{
    return type >= kDudeBase && type < kDudeMax && type != kDudeHand;
}

static bool validXSprite(int extra)
{
    return extra > 0 && extra < kMaxXSprites;
}

static int playerJumpRiseLimit()
{
    if (!gMe)
        return 0;
    const POSTURE &stand = gMe->pPosture[gMe->lifeMode][kPostureStand];
    // Blood applies normalJumpZ as the initial vertical velocity. This
    // scale-derived bound is deliberately conservative and replaces a
    // generic "stuck means jump" fallback.
    return std::max(kMaxWalkableStep, std::abs(stand.normalJumpZ) >> 6);
}

static int playerCrouchClearance()
{
    if (!gMe)
        return kPlayerStandingClearance * 2 / 3;
    const POSTURE &stand = gMe->pPosture[gMe->lifeMode][kPostureStand];
    const POSTURE &crouch = gMe->pPosture[gMe->lifeMode][kPostureCrouch];
    if (stand.eyeAboveZ <= 0)
        return kPlayerStandingClearance * 2 / 3;
    return std::max(1, kPlayerStandingClearance * crouch.eyeAboveZ / stand.eyeAboveZ);
}

static const char *itemCategory(int type)
{
    if (type >= kItemWeaponBase && type < kItemWeaponMax)
        return "weapon";
    if (type >= kItemAmmoBase && type < kItemAmmoMax)
        return "ammo";
    if (type >= kItemKeyBase && type < kItemKeyMax)
        return "key";
    if (type >= kItemHealthDoctorBag && type <= kItemHealthRedPotion)
        return "health";
    if (type >= kItemBase && type < kItemMax)
        return "pickup";
    return nullptr;
}

static void appendUnique(std::vector<int> &values, int value)
{
    if (std::find(values.begin(), values.end(), value) == values.end())
        values.push_back(value);
}

static Observation observeWorld()
{
    Observation result;
    if (!gMe || !gMe->pSprite || !gMe->pXSprite)
        return result;

    spritetype *player = gMe->pSprite;
    result.tick = gFrame * kTicsPerFrame;
    result.sector = player->sectnum;
    result.x = player->x;
    result.y = player->y;
    result.z = player->z;
    result.angle = player->ang;
    result.health = gMe->pXSprite->health;
    appendUnique(result.visibleSectors, result.sector);

    if (inRange(result.sector, 0, numsectors))
    {
        const sectortype &current = sector[result.sector];
        result.localSectorExtra = current.extra;
        if (current.extra > 0 && current.extra < kMaxXSectors)
        {
            result.localSectorState = xsector[current.extra].state;
            result.localSectorBusy = xsector[current.extra].busy;
        }
        result.playerZVelocity = zvel[player->index];
        result.localClearance = getflorzofslope(result.sector, result.x, result.y)
            - getceilzofslope(result.sector, result.x, result.y);
        if (current.extra > 0 && current.extra < kMaxXSectors && xsector[current.extra].Exit)
            result.exitHere = true;

        for (int i = 0; i < current.wallnum; ++i)
        {
            const int wallIndex = current.wallptr + i;
            const walltype &wallRecord = wall[wallIndex];
            if (!inRange(wallRecord.nextsector, 0, numsectors))
                continue;
            ++result.localPortalCount;

            const walltype &nextWall = wall[wallRecord.point2];
            const int midX = (wallRecord.x + nextWall.x) / 2;
            const int midY = (wallRecord.y + nextWall.y) / 2;
            const int fromFloor = getflorzofslope(result.sector, midX, midY);
            const int toFloor = getflorzofslope(wallRecord.nextsector, midX, midY);
            const int fromCeiling = getceilzofslope(result.sector, midX, midY);
            const int toCeiling = getceilzofslope(wallRecord.nextsector, midX, midY);
            const int openingWidth = int(std::sqrt(double(distance2(wallRecord.x, wallRecord.y, nextWall.x, nextWall.y))));
            const int clearance = std::min(fromFloor - fromCeiling, toFloor - toCeiling);
            const int floorDelta = toFloor - fromFloor;
            const int crouchClearance = playerCrouchClearance();
            const int jumpRiseLimit = playerJumpRiseLimit();
            const int dropLimit = jumpRiseLimit;
            const bool enoughWidth = openingWidth >= kPlayerPassageWidth;
            const bool standingClearance = clearance >= kPlayerStandingClearance;
            const bool crouchingClearance = clearance >= crouchClearance;
            const bool walkable = enoughWidth && standingClearance && std::abs(floorDelta) <= kMaxWalkableStep;
            const bool crouchable = enoughWidth && !standingClearance && crouchingClearance
                && std::abs(floorDelta) <= kMaxWalkableStep;
            const bool upward = floorDelta < 0;
            const bool downward = floorDelta > 0;
            const bool jumpable = enoughWidth && standingClearance && upward
                && -floorDelta <= jumpRiseLimit;
            const bool dropSafe = enoughWidth && standingClearance && downward
                && floorDelta <= dropLimit;
            if (jumpable && !(wallRecord.cstat & 1))
            {
                ++result.localJumpableCount;
                result.localMaxRise = std::max(result.localMaxRise, -floorDelta);
            }
            if (downward)
                result.localMaxDrop = std::max(result.localMaxDrop, floorDelta);
            const int midZ = fromFloor;
            const bool visible = cansee(result.x, result.y, result.z, result.sector,
                                         midX, midY, midZ, wallRecord.nextsector);

            Portal portal;
            portal.wall = wallIndex;
            portal.from = result.sector;
            portal.to = wallRecord.nextsector;
            portal.x = midX;
            portal.y = midY;
            portal.z = midZ;
            portal.floorZ = toFloor;
            portal.ceilingZ = toCeiling;
            portal.fromFloorZ = fromFloor;
            portal.fromCeilingZ = fromCeiling;
            portal.toFloorZ = toFloor;
            portal.toCeilingZ = toCeiling;
            portal.crouchClearance = crouchClearance;
            portal.jumpRiseLimit = jumpRiseLimit;
            portal.dropLimit = dropLimit;
            portal.x1 = wallRecord.x;
            portal.y1 = wallRecord.y;
            portal.x2 = nextWall.x;
            portal.y2 = nextWall.y;
            portal.visible = visible;
            portal.openingWidth = openingWidth;
            portal.floorDelta = floorDelta;
            portal.clearance = clearance;
            portal.localGeometry = !visible;
            portal.walkable = !(wallRecord.cstat & 1) && walkable;
            portal.jumpable = !(wallRecord.cstat & 1) && !walkable && jumpable;
            portal.crouchable = !(wallRecord.cstat & 1) && crouchable;
            portal.dropSafe = !(wallRecord.cstat & 1) && dropSafe;
            portal.traversable = portal.walkable || portal.crouchable || portal.jumpable || portal.dropSafe;
            portal.capability = portal.walkable ? kTraversalWalkable
                : portal.crouchable ? kTraversalCrouchable
                : portal.jumpable ? kTraversalJumpable
                : portal.dropSafe ? kTraversalDropSafe
                : kTraversalCurrentlyUnavailable;
            portal.locked = false;
            if (wallRecord.extra > 0 && wallRecord.extra < kMaxXWalls)
            {
                const XWALL &extra = xwall[wallRecord.extra];
                portal.wallPush = extra.triggerPush != 0;
                portal.key = extra.key;
                portal.locked = extra.locked != 0;
                portal.wallState = extra.state;
                portal.wallBusy = extra.busy;
            }
            if (sector[portal.to].extra > 0 && sector[portal.to].extra < kMaxXSectors)
            {
                const XSECTOR &extra = xsector[sector[portal.to].extra];
                if (!portal.key)
                    portal.key = extra.Key;
                portal.locked = extra.locked != 0;
                portal.sectorPush = extra.Wallpush != 0;
                portal.sectorState = extra.state;
                portal.sectorBusy = extra.busy;
                if (extra.damageType != 0 || sector[portal.to].type == kSectorDamage)
                {
                    portal.walkable = false;
                    portal.jumpable = false;
                    portal.crouchable = false;
                    portal.dropSafe = false;
                    portal.traversable = false;
                    portal.capability = kTraversalCurrentlyUnavailable;
                }
            }
            if (sector[portal.from].extra > 0 && sector[portal.from].extra < kMaxXSectors)
                portal.sectorPushCurrent = xsector[sector[portal.from].extra].Push != 0;
            portal.interactionAffordance = portal.wallPush || portal.sectorPush || portal.sectorPushCurrent;
            portal.currentlyAvailable = portal.traversable || portal.interactionAffordance;
            if (!portal.currentlyAvailable)
            {
                portal.capability = kTraversalCurrentlyUnavailable;
                portal.unavailableReason = (wallRecord.cstat & 1) ? 1 : 2;
            }
            portal.directUse = (portal.wallPush || portal.sectorPush || portal.sectorPushCurrent)
                && (!portal.traversable || std::abs(portal.floorDelta) > kMaxWalkableStep);

            if (!visible && !portal.traversable)
                continue;
            if (visible)
                appendUnique(result.visibleSectors, portal.to);
            result.portals.push_back(portal);
        }
    }

    // The scan is over the engine's sprite list, but only objects passing the
    // authoritative cansee() test enter the bot's observation.
    for (int i = 0; i < kMaxSprites; ++i)
    {
        spritetype &candidate = sprite[i];
        const bool isEnemy = candidate.statnum == kStatDude;
        const bool isItem = candidate.statnum == kStatItem;
        const bool isThing = candidate.statnum == kStatThing;
        const bool isSwitch = candidate.type >= kSwitchBase && candidate.type < kSwitchMax;
        if (!isEnemy && !isItem && !isThing && !isSwitch)
            continue;
        if (candidate.index == player->index || candidate.sectnum < 0)
            continue;
        if (isEnemy && (!isEnemyType(candidate.type) || !validXSprite(candidate.extra) || xsprite[candidate.extra].health == 0))
            continue;
        if (isItem && itemCategory(candidate.type) == nullptr)
            continue;
        if ((isThing || isSwitch) && (!validXSprite(candidate.extra)
                        || (!xsprite[candidate.extra].Push && !xsprite[candidate.extra].Vector)))
            continue;
        if (!cansee(result.x, result.y, result.z, result.sector,
                    candidate.x, candidate.y, candidate.z, candidate.sectnum))
            continue;

        VisibleObject object;
        object.sprite = i;
        object.sector = candidate.sectnum;
        object.type = candidate.type;
        object.x = candidate.x;
        object.y = candidate.y;
        object.z = candidate.z;
        object.kind = isEnemy ? kObjectEnemy
                         : isKeyType(candidate.type) ? kObjectKey
                         : (isThing || isSwitch) ? kObjectInteractive : kObjectPickup;
        result.objects.push_back(object);
    }
    return result;
}

static const char *objectName(ObjectKind kind)
{
    switch (kind)
    {
    case kObjectEnemy: return "enemy";
    case kObjectKey: return "key";
    case kObjectInteractive: return "interactive";
    default: return "pickup";
    }
}

}

struct LLMapperBot::Impl
{
    struct DoorMemory
    {
        enum Availability
        {
            kActionable,
            kMissingKey,
            kStructurallyBlocked,
            kCurrentlyUnavailable,
        };

        enum InteractionState
        {
            kIdle,
            kWaiting,
            kOpening,
            kOpen,
            kFailed,
        };

        int id = -1;
        Portal portal;
        int attempts = 0;
        Availability availability = kActionable;
        bool opened = false;
        int unavailableReason = 0;
        InteractionState interaction = kIdle;
        int interactionStartedTick = -1;
        int interactionDeadlineTick = -1;
        int unavailableFromSector = -1;
    };

    FILE *telemetry = nullptr;
    FILE *trajectory = nullptr;
    std::string telemetryPath = "llmapper-bot.ndjson";
    std::string trajectoryPath = "llmapper-bot.trajectory.ndjson";
    std::string demoPath = "llmapper-bot.dem";
    int timeoutSeconds = kDefaultTimeoutSeconds;
    int stallSeconds = kDefaultStallSeconds;
    int lastObservationTick = -1;
    int lastTrajectoryTick = -1;
    int lastSemanticProgressTick = 0;
    int lastStateLocation = -1;
    std::string lastStateGoal;
    int lastStateTarget = -1;
    int lastStateKnowledge = -1;
    int lastStateInventory = -1;
    int repeatedStateCount = 0;
    int pendingUse = -1;
    int pendingUseTick = -1;
    std::string result;
    std::string failureReason;
    std::string currentGoal;
    Observation observation;
    std::set<int> observedSectors;
    std::set<int> visitedSectors;
    std::map<int, DoorMemory> doors;
    std::set<int> knownObjects;
    std::set<int> knownKeys;
    std::set<int> unreachableObjects;
    std::set<int> unreachableEnemies;
    std::set<int> seenEdges;
    std::set<int> visitedEdges;
    std::map<int, int> failedEdges;
    std::set<int> openedRoutes;
    std::map<int, std::vector<Portal>> knownGraph;
    Portal routePortal;
    Portal localJumpPortal;
    int selectedDoorId = -1;
    int knowledgeRevision = 0;
    int inventoryRevision = 0;
    int lastObservedSector = -1;
    int lastGeometryTelemetrySector = -1;
    bool localDynamicJumpAttempted = false;
    int localDynamicJumpUntilTick = -1;
    int lastTransitionFrom = -1;
    int lastTransitionTo = -1;
    int lastTransitionKnowledge = -1;
    int lastTransitionInventory = -1;
    int repeatedBacktrackCount = 0;
    int currentGoalTarget = -1;
    bool movementTargetActive = false;
    int movementTargetX = 0;
    int movementTargetY = 0;
    int movementTargetSector = -1;
    int movementTargetId = -1;
    int movementTargetFrom = -1;
    TraversalCapability movementTargetCapability = kTraversalUnknown;
    std::string movementTargetGoal;
    int targetLastX = 0;
    int targetLastY = 0;
    int targetLastZ = 0;
    int targetLastDistance2 = 0;
    int targetLastProgressTick = 0;
    int jumpCooldownTick = 0;
    int jumpAttempts = 0;
    int searchAngle = -1;
    int lastDoorActionEventTick = -1;
    int lastUseProbeDoor = -1;
    int lastUseProbeTick = -1;
    std::string lastUseProbeStatus;
    int lastUsedDoor = -1;
    int lastUsedDoorFrom = -1;
    int lastUsedDoorTo = -1;
    int lastUsedDoorTick = -1;
    int portalWaypointWall = -1;
    int portalWaypointX = 0;
    int portalWaypointY = 0;
    bool portalWaypointActive = false;
    bool crouchTargetActive = false;

    void openFiles()
    {
        telemetry = fopen(telemetryPath.c_str(), "wb");
        trajectory = fopen(trajectoryPath.c_str(), "wb");
    }

    void event(const char *name, const char *detail = nullptr)
    {
        if (!telemetry)
            return;
        const int gameSeconds = (gFrame * kTicsPerFrame) / kTicsPerSec;
        fprintf(telemetry, "{\"type\":\"event\",\"game_time\":%d,\"event\":\"%s\"",
                gameSeconds, name);
        if (detail)
            fprintf(telemetry, ",\"detail\":\"%s\"", detail);
        fprintf(telemetry, "}\n");
        fflush(telemetry);
    }

    void actionResolved(int hit, int target, int extra, bool accepted, int key)
    {
        const char *kind = hit == 0 ? "wall" : hit == 3 ? "sprite" : hit == 6 ? "sector" : "none";
        char detail[256];
        snprintf(detail, sizeof(detail),
                 "source=gameplay_tick hit=%d kind=%s target=%d extra=%d trigger_dispatched=%d key=%d goal=%s selected_door=%d",
                 hit, kind, target, extra, accepted ? 1 : 0, key,
                 currentGoal.c_str(), selectedDoorId);
        event("engine_use_resolved", detail);
    }

    void trajectorySample()
    {
        if (!trajectory || !gMe || !gMe->pSprite)
            return;
        fprintf(trajectory, "{\"game_time\":%d,\"x\":%d,\"y\":%d,\"z\":%d,\"sector\":%d,\"angle\":%d,\"health\":%d}\n",
                (gFrame * kTicsPerFrame) / kTicsPerSec,
                int(gMe->pSprite->x), int(gMe->pSprite->y), int(gMe->pSprite->z), int(gMe->pSprite->sectnum),
                int(gMe->pSprite->ang), int(gMe->pXSprite ? gMe->pXSprite->health : 0));
        fflush(trajectory);
    }

    void updateKnowledge()
    {
        observation = observeWorld();
        if (observation.sector < 0)
            return;
        if (observation.sector != lastGeometryTelemetrySector)
        {
            char detail[160];
            snprintf(detail, sizeof(detail),
                     "sector=%d clearance=%d portals=%d jumpable=%d max_rise=%d max_drop=%d xsector=%d state=%d busy=%d zvel=%d",
                     observation.sector, observation.localClearance, observation.localPortalCount,
                     observation.localJumpableCount, observation.localMaxRise, observation.localMaxDrop,
                     observation.localSectorExtra, observation.localSectorState,
                     observation.localSectorBusy, observation.playerZVelocity);
            event("local_geometry", detail);
            lastGeometryTelemetrySector = observation.sector;
        }
        const size_t oldSectors = observedSectors.size();
        observedSectors.insert(observation.visibleSectors.begin(), observation.visibleSectors.end());
        const int previousSector = lastObservedSector;
        const bool sectorChanged = observation.sector != lastObservedSector;
        if (sectorChanged)
        {
            localDynamicJumpAttempted = false;
            localDynamicJumpUntilTick = -1;
        }
        if (observation.localSectorBusy != 0 && observation.playerZVelocity != 0)
            localDynamicJumpUntilTick = observation.tick + 4 * kTicsPerSec;
        if (sectorChanged && previousSector >= 0 && movementTargetFrom == previousSector
            && movementTargetSector == observation.sector && movementTargetId >= 0)
        {
            const int edgeId = movementTargetId * 65536 + observation.sector;
            visitedEdges.insert(edgeId);
            failedEdges.erase(edgeId);
            char detail[128];
            snprintf(detail, sizeof(detail), "from=%d to=%d wall=%d", previousSector,
                     observation.sector, movementTargetId);
            event("portal_traversed", detail);
            lastSemanticProgressTick = observation.tick;
        }
        if (sectorChanged && previousSector >= 0 && movementTargetId >= 0)
        {
            auto door = doors.find(movementTargetId);
            if (door != doors.end() && door->second.portal.interactionAffordance
                && door->second.portal.from == previousSector
                && door->second.portal.to == observation.sector)
            {
                const bool usedRecently = lastUsedDoor == movementTargetId
                    && lastUsedDoorFrom == previousSector
                    && lastUsedDoorTo == observation.sector
                    && observation.tick - lastUsedDoorTick <= 4 * kTicsPerSec;
                if (!usedRecently)
                {
                    door->second.opened = true;
                    door->second.interaction = DoorMemory::kOpen;
                    door->second.unavailableFromSector = -1;
                    event("door_open", "response=authoritative_traversable_without_use");
                }
            }
        }
        if (sectorChanged && movementTargetSector == observation.sector)
        {
            movementTargetActive = false;
            currentGoal.clear();
            currentGoalTarget = -1;
        }
        if (sectorChanged && lastUsedDoor >= 0
            && previousSector == lastUsedDoorFrom && observation.sector == lastUsedDoorTo
            && observation.tick - lastUsedDoorTick <= 4 * kTicsPerSec)
        {
            auto door = doors.find(lastUsedDoor);
            if (door != doors.end())
            {
                const bool alreadyOpen = door->second.interaction == DoorMemory::kOpen;
                door->second.interaction = DoorMemory::kOpen;
                door->second.interactionDeadlineTick = -1;
                if (!alreadyOpen)
                    event("door_open", "response=sector_crossing");
                visitedEdges.insert(lastUsedDoor * 65536 + lastUsedDoorTo);
                openedRoutes.insert(lastUsedDoorFrom * 65536 + lastUsedDoorTo);
                openedRoutes.insert(lastUsedDoorTo * 65536 + lastUsedDoorFrom);
                for (auto &entry : doors)
                {
                    DoorMemory &candidate = entry.second;
                    if (!candidate.portal.key && !candidate.portal.locked
                        && candidate.portal.from == lastUsedDoorFrom
                        && candidate.portal.to == lastUsedDoorTo)
                    {
                        candidate.opened = true;
                        visitedEdges.insert(candidate.portal.wall * 65536 + candidate.portal.to);
                    }
                    if (!candidate.portal.key && !candidate.portal.locked
                        && candidate.portal.from == lastUsedDoorTo
                        && candidate.portal.to == lastUsedDoorFrom)
                        visitedEdges.insert(candidate.portal.wall * 65536 + candidate.portal.to);
                }
                char detail[96];
                snprintf(detail, sizeof(detail), "door=%d from=%d to=%d", lastUsedDoor,
                         lastUsedDoorFrom, lastUsedDoorTo);
                event("door_traversed", detail);
                lastSemanticProgressTick = observation.tick;
            }
            lastUsedDoor = -1;
        }
        else if (lastUsedDoor >= 0 && observation.tick - lastUsedDoorTick > 4 * kTicsPerSec)
        {
            lastUsedDoor = -1;
        }
        if (sectorChanged && previousSector >= 0)
        {
            const bool reverse = lastTransitionFrom == observation.sector
                && lastTransitionTo == previousSector
                && lastTransitionKnowledge == knowledgeRevision
                && lastTransitionInventory == inventoryRevision;
            if (reverse)
                ++repeatedBacktrackCount;
            else
                repeatedBacktrackCount = 0;
            if (repeatedBacktrackCount >= 2)
            {
                char detail[128];
                snprintf(detail, sizeof(detail), "from=%d to=%d count=%d", previousSector,
                         observation.sector, repeatedBacktrackCount);
                event("repeated_backtrack", detail);
            }
        }
        if (visitedSectors.insert(observation.sector).second)
        {
            ++knowledgeRevision;
            char detail[64];
            snprintf(detail, sizeof(detail), "sector=%d", observation.sector);
            event("sector_entered", detail);
        }
        if (sectorChanged)
        {
            lastTransitionFrom = previousSector;
            lastTransitionTo = observation.sector;
            lastTransitionKnowledge = knowledgeRevision;
            lastTransitionInventory = inventoryRevision;
            lastObservedSector = observation.sector;
        }
        if (observedSectors.size() != oldSectors)
        {
            lastSemanticProgressTick = observation.tick;
            event("discovered_sector");
        }

        for (const Portal &portal : observation.portals)
        {
            if (portal.localGeometry)
                continue;
            const int edgeId = portal.wall * 65536 + portal.to;
            const bool newEdge = seenEdges.insert(edgeId).second;
            if (newEdge)
            {
                ++knowledgeRevision;
                lastSemanticProgressTick = observation.tick;
                char detail[96];
                snprintf(detail, sizeof(detail), "wall=%d from=%d to=%d", portal.wall,
                         portal.from, portal.to);
                event("discovered_frontier", detail);
            }
            if (portal.traversable || portal.jumpable)
            {
                auto &edges = knownGraph[portal.from];
                auto edge = std::find_if(edges.begin(), edges.end(), [&portal](const Portal &known)
                {
                    return known.wall == portal.wall && known.to == portal.to;
                });
                if (edge == edges.end())
                    edges.push_back(portal);
                else
                    *edge = portal;
            }
            const bool blockedInteractivePortal = portal.interactionAffordance;
            if (portal.key || portal.locked || blockedInteractivePortal
                || (!portal.traversable && !portal.jumpable))
            {
                DoorMemory &door = doors[portal.wall];
                if (door.id < 0)
                {
                    door.id = portal.wall;
                    ++knowledgeRevision;
                    char detail[128];
                    snprintf(detail, sizeof(detail), "door=%d key=%d locked=%d wall_push=%d sector_push=%d direct_use=%d",
                             portal.wall, portal.key, portal.locked ? 1 : 0,
                             portal.wallPush ? 1 : 0, portal.sectorPush ? 1 : 0,
                             portal.directUse ? 1 : 0);
                    event("discovered_door", detail);
                }
                const Portal previousPortal = door.portal;
                const bool hadPortal = door.id >= 0;
                const bool response = hadPortal
                    && (previousPortal.wallState != portal.wallState
                        || previousPortal.wallBusy != portal.wallBusy
                        || previousPortal.sectorState != portal.sectorState
                        || previousPortal.sectorBusy != portal.sectorBusy
                        || previousPortal.floorZ != portal.floorZ
                        || previousPortal.ceilingZ != portal.ceilingZ
                        || previousPortal.floorDelta != portal.floorDelta
                        || previousPortal.traversable != portal.traversable);
                door.portal = portal;
                if (response && (door.interaction == DoorMemory::kWaiting
                                 || door.interaction == DoorMemory::kOpening))
                {
                    if (door.interaction == DoorMemory::kWaiting)
                    {
                        door.interaction = DoorMemory::kOpening;
                        char detail[96];
                        snprintf(detail, sizeof(detail), "door=%d wall_busy=%d sector_busy=%d",
                                 portal.wall, portal.wallBusy, portal.sectorBusy);
                        event("door_opening", detail);
                    }
                    lastSemanticProgressTick = observation.tick;
                    if (!portal.directUse && portal.traversable && !portal.wallBusy && !portal.sectorBusy)
                    {
                        door.interaction = DoorMemory::kOpen;
                        event("door_open", "response=authoritative_traversable");
                    }
                }
                const bool unavailableFromThisSide = door.unavailableFromSector == observation.sector
                    && portal.from == observation.sector;
                const DoorMemory::Availability oldAvailability = door.availability;
                if (portal.key && !hasKey(portal.key))
                    door.availability = DoorMemory::kMissingKey;
                else if (unavailableFromThisSide)
                    door.availability = DoorMemory::kCurrentlyUnavailable;
                else if (portal.key || portal.locked || blockedInteractivePortal || portal.traversable)
                    door.availability = DoorMemory::kActionable;
                else
                {
                    door.availability = DoorMemory::kCurrentlyUnavailable;
                    door.unavailableReason = portal.unavailableReason;
                }
                if (door.availability == DoorMemory::kCurrentlyUnavailable
                    && oldAvailability != DoorMemory::kCurrentlyUnavailable)
                {
                    char detail[96];
                    snprintf(detail, sizeof(detail), "wall=%d reason=%d affordance=%d",
                             portal.wall, portal.unavailableReason,
                             portal.interactionAffordance ? 1 : 0);
                    event("portal_unavailable", detail);
                }
            }
        }

        for (const VisibleObject &object : observation.objects)
        {
            if (object.kind == kObjectKey && knownKeys.insert(object.type - kItemKeyBase + 1).second)
                ++knowledgeRevision;
            if (knownObjects.insert(object.sprite).second)
            {
                char detail[160];
                const char *category = itemCategory(object.type);
                snprintf(detail, sizeof(detail), "sprite=%d kind=%s category=%s sector=%d type=%d",
                         object.sprite, objectName(object.kind), category ? category : "none", object.sector, object.type);
                event("observed_object", detail);
                lastSemanticProgressTick = observation.tick;
                ++knowledgeRevision;
            }
        }

        for (int key = 1; key < 8; ++key)
        {
            if (gMe->hasKey[key] && knownKeys.insert(key).second)
            {
                char detail[64];
                snprintf(detail, sizeof(detail), "key=%d", key);
                event("acquired_key", detail);
                lastSemanticProgressTick = observation.tick;
                ++inventoryRevision;
            }
        }

        for (auto &entry : doors)
        {
            DoorMemory &door = entry.second;
            if ((door.interaction == DoorMemory::kWaiting || door.interaction == DoorMemory::kOpening)
                && door.interactionDeadlineTick >= 0 && observation.tick > door.interactionDeadlineTick)
            {
                door.interaction = DoorMemory::kFailed;
                char detail[144];
                snprintf(detail, sizeof(detail), "door=%d waited_ticks=%d reason=INTERACTION_VALID_BUT_NO_RESPONSE", door.id,
                         observation.tick - door.interactionStartedTick);
                event("interaction_failed", detail);
            }
            if (door.portal.key && hasKey(door.portal.key) && door.availability == DoorMemory::kMissingKey)
            {
                door.availability = DoorMemory::kActionable;
                ++knowledgeRevision;
                char detail[64];
                snprintf(detail, sizeof(detail), "door=%d key=%d", door.id, door.portal.key);
                event("door_ready_after_key", detail);
            }
        }

        if (pendingUse >= 0 && pendingUseTick < observation.tick)
        {
            auto door = doors.find(pendingUse);
            if (door != doors.end() && door->second.portal.key && !hasKey(door->second.portal.key))
            {
                door->second.availability = DoorMemory::kMissingKey;
                door->second.interaction = DoorMemory::kFailed;
                event("interaction_failed", "reason=missing_key");
                char detail[96];
                snprintf(detail, sizeof(detail), "door=%d key=%d", pendingUse, door->second.portal.key);
                event("blocked_key_required", detail);
            }
            pendingUse = -1;
        }
    }

    bool hasKey(int key) const
    {
        return key <= 0 || (key < 8 && gMe && gMe->hasKey[key]);
    }

    bool edgeFailed(int edgeId) const
    {
        auto failed = failedEdges.find(edgeId);
        return failed != failedEdges.end() && failed->second == knowledgeRevision;
    }

    const Portal *selectPortal()
    {
        const bool avoidBacktrack = repeatedBacktrackCount >= 2
            && lastTransitionTo == observation.sector;
        if (currentGoal == "EXPLORE_FRONTIER" && currentGoalTarget >= 0)
        {
            for (const Portal &portal : observation.portals)
            {
                if (!portal.visible)
                    continue;
                const int edgeId = portal.wall * 65536 + portal.to;
                if (portal.wall == currentGoalTarget && !edgeFailed(edgeId)
                    && (portal.traversable || portal.jumpable)
                    && (!portal.key || hasKey(portal.key))
                    && (!avoidBacktrack || portal.to != lastTransitionFrom))
                    return &portal;
            }

            auto known = knownGraph.find(observation.sector);
            if (known != knownGraph.end())
            {
                for (const Portal &portal : known->second)
                {
                    const int edgeId = portal.wall * 65536 + portal.to;
                    if (portal.wall == currentGoalTarget && !edgeFailed(edgeId)
                        && (!portal.key || hasKey(portal.key))
                        && (!avoidBacktrack || portal.to != lastTransitionFrom))
                        return &portal;
                }
            }
        }

        bool unvisitedAvailable = false;
        for (const Portal &portal : observation.portals)
        {
            if (!portal.visible)
                continue;
            if ((portal.traversable || portal.jumpable)
                && (!portal.key || hasKey(portal.key))
                && !visitedEdges.count(portal.wall * 65536 + portal.to)
                && !edgeFailed(portal.wall * 65536 + portal.to)
                && !openedRoutes.count(portal.from * 65536 + portal.to)
                && (!avoidBacktrack || portal.to != lastTransitionFrom))
            {
                unvisitedAvailable = true;
                break;
            }
        }

        const Portal *best = nullptr;
        int bestDistance = INT32_MAX;
        for (const Portal &portal : observation.portals)
        {
            if (!portal.visible)
                continue;
            if (!(portal.traversable || portal.jumpable) || (portal.key && !hasKey(portal.key)))
                continue;
            if (avoidBacktrack && portal.to == lastTransitionFrom)
                continue;
            const int edgeId = portal.wall * 65536 + portal.to;
            if (edgeFailed(edgeId))
                continue;
            const bool alreadyExplored = visitedEdges.count(edgeId)
                || openedRoutes.count(portal.from * 65536 + portal.to);
            if (unvisitedAvailable && alreadyExplored)
                continue;
            const int score = alreadyExplored ? 100000000 : distance2(observation.x, observation.y, portal.x, portal.y);
            if (score < bestDistance)
            {
                best = &portal;
                bestDistance = score;
            }
        }
        return best;
    }

    const Portal *selectLocalPortal()
    {
        const Portal *best = nullptr;
        int bestDistance = INT32_MAX;
        for (const Portal &portal : observation.portals)
        {
            if (!portal.localGeometry || !portal.traversable)
                continue;
            const int edgeId = portal.wall * 65536 + portal.to;
            if (edgeFailed(edgeId) || visitedEdges.count(edgeId)
                || openedRoutes.count(portal.from * 65536 + portal.to)
                || portal.to == lastTransitionFrom)
                continue;
            const int score = distance2(observation.x, observation.y, portal.x, portal.y);
            if (score < bestDistance)
            {
                bestDistance = score;
                localJumpPortal = portal;
                best = &localJumpPortal;
            }
        }
        return best;
    }

    const VisibleObject *selectObject(ObjectKind kind)
    {
        const VisibleObject *best = nullptr;
        int bestDistance = INT32_MAX;
        for (const VisibleObject &object : observation.objects)
        {
            if (object.kind != kind)
                continue;
            const int distance = distance2(observation.x, observation.y, object.x, object.y);
            if (distance < bestDistance)
            {
                best = &object;
                bestDistance = distance;
            }
        }
        return best;
    }

    bool findKnownRoute(int targetSector, Portal &route) const
    {
        if (targetSector < 0 || targetSector == observation.sector)
            return false;
        std::vector<int> frontier(1, observation.sector);
        std::set<int> reached;
        std::map<int, Portal> parent;
        reached.insert(observation.sector);
        for (size_t index = 0; index < frontier.size(); ++index)
        {
            const int current = frontier[index];
            auto graph = knownGraph.find(current);
            if (graph == knownGraph.end())
                continue;
            for (const Portal &edge : graph->second)
            {
                if (edgeFailed(edge.wall * 65536 + edge.to))
                    continue;
                if (reached.insert(edge.to).second)
                {
                    parent[edge.to] = edge;
                    frontier.push_back(edge.to);
                }
            }
        }
        if (!reached.count(targetSector))
            return false;
        int cursor = targetSector;
        route = parent[cursor];
        while (route.from != observation.sector)
        {
            cursor = route.from;
            auto edge = parent.find(cursor);
            if (edge == parent.end())
                return false;
            route = edge->second;
        }
        return true;
    }

    bool directObjectReachable(const VisibleObject &object) const
    {
        return object.sector == observation.sector
            && std::abs(object.z - observation.z) <= kMaxWalkableStep;
    }

    TraversalCapability currentClearanceCapability() const
    {
        if (!inRange(observation.sector, 0, numsectors))
            return kTraversalUnknown;
        const int floorZ = getflorzofslope(observation.sector, observation.x, observation.y);
        const int ceilingZ = getceilzofslope(observation.sector, observation.x, observation.y);
        const int clearance = floorZ - ceilingZ;
        if (clearance < kPlayerStandingClearance && clearance >= playerCrouchClearance())
            return kTraversalCrouchable;
        return kTraversalUnknown;
    }

    TraversalCapability currentRecoveryCapability() const
    {
        const TraversalCapability clearance = currentClearanceCapability();
        if (clearance == kTraversalCrouchable)
            return clearance;
        if (!localDynamicJumpAttempted && observation.tick <= localDynamicJumpUntilTick)
            return kTraversalJumpable;
        return kTraversalUnknown;
    }

    bool rangedWeaponAvailable(int &weapon) const
    {
        static const int candidates[] = {
            kWeaponTommy, kWeaponShotgun, kWeaponFlare, kWeaponTesla, kWeaponNapalm,
        };
        for (int candidate : candidates)
        {
            if (!gMe->hasWeapon[candidate])
                continue;
            const int ammo = candidate - 1;
            if (gInfiniteAmmo || (ammo >= 0 && ammo < int(sizeof(gMe->ammoCount) / sizeof(gMe->ammoCount[0]))
                                  && gMe->ammoCount[ammo] > 0))
            {
                weapon = candidate;
                return true;
            }
        }
        return false;
    }

    GINPUT aimAndShoot(const VisibleObject &enemy, int weapon)
    {
        GINPUT input = {};
        input.syncFlags.run = 1;
        const int targetAngle = getangle(enemy.x - observation.x, enemy.y - observation.y);
        input.q16turn = fix16_from_int(angleDelta(targetAngle, observation.angle));
        const int horizontal = std::max(1, int(std::sqrt(double(distance2(observation.x, observation.y,
                                                                         enemy.x, enemy.y)))));
        const int targetZ = enemy.z;
        const double pitch = std::atan2(double(observation.z - targetZ), double(horizontal))
            * 1024.0 / 3.14159265358979323846;
        const int desiredLook = std::max(-347, std::min(289, int(std::lround(pitch))));
        const int currentLook = fix16_to_int(gMe->q16look);
        input.q16mlook = fix16_from_int((desiredLook - currentLook) / 8);
        if (gMe->curWeapon == weapon)
            input.buttonFlags.shoot = 1;
        else
        {
            input.newWeapon = uint8_t(weapon);
            char detail[96];
            snprintf(detail, sizeof(detail), "enemy=%d weapon=%d", enemy.sprite, weapon);
            event("ranged_weapon_selected", detail);
        }
        return input;
    }

    bool actionTargetMatches(const Portal &portal, int hit, int target) const
    {
        if (portal.wallPush && hit == 0 && target == portal.wall)
            return true;
        if (portal.sectorPush && hit == 6 && target == portal.to)
            return true;
        if (portal.sectorPushCurrent && hit == 6 && target == portal.from)
            return true;
        return false;
    }

    void emitUseProbe(const Portal &portal, const DoorMemory &memory, bool atDoor,
                      int hit, int target, int extra, const char *status)
    {
        if (lastUseProbeDoor == portal.wall && lastUseProbeStatus == status
            && observation.tick - lastUseProbeTick < kTicsPerSec)
            return;
        lastUseProbeDoor = portal.wall;
        lastUseProbeTick = observation.tick;
        lastUseProbeStatus = status;
        const int targetAngle = getangle(portal.x - observation.x, portal.y - observation.y);
        char detail[320];
        snprintf(detail, sizeof(detail),
                 "status=%s at_door=%d goal=%s selected_door=%d expected_wall=%d expected_from=%d expected_sector=%d expected_wall_push=%d expected_sector_push=%d expected_current_sector_push=%d hit=%d target=%d extra=%d distance=%d angle_delta=%d key=%d locked=%d interaction=%d attempts=%d",
                 status, atDoor ? 1 : 0, currentGoal.c_str(), selectedDoorId, portal.wall, portal.from, portal.to, portal.wallPush ? 1 : 0,
                 portal.sectorPush ? 1 : 0, portal.sectorPushCurrent ? 1 : 0, hit, target, extra,
                 int(std::sqrt(double(distance2(observation.x, observation.y, portal.x, portal.y)))),
                 angleDelta(targetAngle, observation.angle), portal.key, portal.locked ? 1 : 0,
                 int(memory.interaction), memory.attempts);
        event("use_probe", detail);
    }

    const Portal *selectKnownDoor()
    {
        const Portal *best = nullptr;
        selectedDoorId = -1;
        int bestDistance = INT32_MAX;
        const bool avoidBacktrack = repeatedBacktrackCount >= 2
            && lastTransitionTo == observation.sector;
        int waitingDoor = -1;
        for (const auto &entry : doors)
        {
            if (entry.second.interaction == DoorMemory::kWaiting
                || entry.second.interaction == DoorMemory::kOpening)
            {
                waitingDoor = entry.first;
                break;
            }
        }
        for (auto &entry : doors)
        {
            DoorMemory &door = entry.second;
            if (waitingDoor >= 0 && door.id != waitingDoor)
                continue;
            if (!door.portal.interactionAffordance)
                continue;
            if (door.availability == DoorMemory::kStructurallyBlocked
                || door.availability == DoorMemory::kCurrentlyUnavailable
                || (door.availability == DoorMemory::kMissingKey && door.attempts > 0 && !hasKey(door.portal.key)))
                continue;
            if (door.opened && !door.portal.key && !door.portal.locked)
                continue;
            if (avoidBacktrack && door.portal.from == observation.sector
                && door.portal.to == lastTransitionFrom && !door.portal.key && !door.portal.locked)
                continue;

            // Search only the discovered, traversable graph. This keeps a
            // remembered locked door useful without turning the loaded MAP
            // into an omniscient navigation graph.
            std::vector<int> frontier(1, observation.sector);
            std::set<int> reached;
            std::map<int, Portal> parent;
            reached.insert(observation.sector);
            for (size_t index = 0; index < frontier.size(); ++index)
            {
                const int current = frontier[index];
                if (current == door.portal.from)
                    break;
                auto graph = knownGraph.find(current);
                if (graph == knownGraph.end())
                    continue;
                for (const Portal &edge : graph->second)
                {
                    if (reached.insert(edge.to).second)
                    {
                        parent[edge.to] = edge;
                        frontier.push_back(edge.to);
                    }
                }
            }
            if (!reached.count(door.portal.from))
                continue;

            Portal candidateRoute = door.portal;
            int cursor = door.portal.from;
            while (cursor != observation.sector)
            {
                const Portal edge = parent[cursor];
                candidateRoute = edge;
                cursor = edge.from;
            }
            if (avoidBacktrack && candidateRoute.from == observation.sector
                && candidateRoute.to == lastTransitionFrom && !candidateRoute.key && !candidateRoute.locked)
                continue;
            const int distance = distance2(observation.x, observation.y, candidateRoute.x, candidateRoute.y);
            if (distance < bestDistance)
            {
                routePortal = candidateRoute;
                best = &routePortal;
                bestDistance = distance;
                selectedDoorId = door.id;
            }
        }
        return best;
    }

    void setGoal(const char *goal, int target = -1)
    {
        if (currentGoal == goal && currentGoalTarget == target)
            return;
        currentGoal = goal;
        currentGoalTarget = target;
        char detail[160];
        snprintf(detail, sizeof(detail), "goal=%s target=%d", goal, target);
        event("goal_changed", detail);
    }

    void setMovementTarget(int x, int y, int targetSector, int targetId,
                           TraversalCapability capability = kTraversalUnknown)
    {
        if (movementTargetActive && movementTargetGoal == currentGoal
            && movementTargetX == x && movementTargetY == y
            && movementTargetSector == targetSector && movementTargetId == targetId
            && movementTargetCapability == capability)
            return;

        movementTargetActive = true;
        movementTargetX = x;
        movementTargetY = y;
        movementTargetSector = targetSector;
        movementTargetId = targetId;
        movementTargetFrom = observation.sector;
        movementTargetCapability = capability;
        movementTargetGoal = currentGoal;
        targetLastX = observation.x;
        targetLastY = observation.y;
        targetLastZ = observation.z;
        targetLastDistance2 = distance2(observation.x, observation.y, x, y);
        targetLastProgressTick = observation.tick;
        jumpCooldownTick = observation.tick;
        jumpAttempts = 0;
    }

    void updateMovementProgress()
    {
        if (!movementTargetActive)
            return;

        const int currentDistance2 = distance2(observation.x, observation.y,
                                               movementTargetX, movementTargetY);
        const bool horizontalProgress = targetLastDistance2 - currentDistance2 >= 4096;
        const bool verticalProgress = std::abs(observation.z - targetLastZ) >= 256;
        const bool sectorProgress = movementTargetSector >= 0 && observation.sector != movementTargetSector;
        if (horizontalProgress || verticalProgress)
        {
            if (jumpAttempts > 0)
            {
                char detail[128];
                snprintf(detail, sizeof(detail), "target=%d dx=%d dy=%d dz=%d attempts=%d",
                         movementTargetId, observation.x - targetLastX, observation.y - targetLastY,
                         observation.z - targetLastZ, jumpAttempts);
                event(horizontalProgress || sectorProgress ? "jump_succeeded" : "jump_progress", detail);
                if (horizontalProgress || sectorProgress)
                    jumpAttempts = 0;
            }
            targetLastX = observation.x;
            targetLastY = observation.y;
            targetLastZ = observation.z;
            targetLastDistance2 = currentDistance2;
            targetLastProgressTick = observation.tick;
        }
    }

    bool movementNeedsJump() const
    {
        return movementTargetActive && jumpAttempts < kMaxJumpAttemptsPerTarget
            && movementTargetCapability == kTraversalJumpable
            && observation.tick >= jumpCooldownTick
            && observation.tick - targetLastProgressTick >= kMovementStuckTicks
            && distance2(observation.x, observation.y, movementTargetX, movementTargetY) > 4096;
    }

    void preparePortalWaypoint(const Portal &portal)
    {
        portalWaypointWall = portal.wall;
        portalWaypointActive = false;
        portalWaypointX = portal.x;
        portalWaypointY = portal.y;

        const int dx = portal.x2 - portal.x1;
        const int dy = portal.y2 - portal.y1;
        const int length = int(std::sqrt(double(int64_t(dx) * dx + int64_t(dy) * dy)));
        const int radius = gMe && gMe->pSprite ? (gMe->pSprite->clipdist << 2) : 128;
        const int safeHalf = length / 2 - std::max(128, radius + 64);
        if (safeHalf <= 128 || length <= 0)
            return;

        const double tangentX = double(dx) / length;
        const double tangentY = double(dy) / length;
        const int offset = safeHalf / 2;
        const int candidates[2] = { offset, -offset };
        for (int side : candidates)
        {
            const int candidateX = portal.x + int(std::lround(tangentX * side));
            const int candidateY = portal.y + int(std::lround(tangentY * side));
            if (!cansee(observation.x, observation.y, observation.z, observation.sector,
                        candidateX, candidateY, observation.z, observation.sector))
                continue;
            portalWaypointX = candidateX;
            portalWaypointY = candidateY;
            portalWaypointActive = true;
            return;
        }
    }

    GINPUT steerPortal(const Portal &portal)
    {
        if (portalWaypointWall != portal.wall)
            preparePortalWaypoint(portal);
        const int radius = gMe && gMe->pSprite ? (gMe->pSprite->clipdist << 2) : 128;
        if (portalWaypointActive
            && distance2(observation.x, observation.y, portalWaypointX, portalWaypointY)
                <= std::max(512, radius * 2) * std::max(512, radius * 2))
        {
            portalWaypointActive = false;
        }
        const int targetX = portalWaypointActive ? portalWaypointX : portal.x;
        const int targetY = portalWaypointActive ? portalWaypointY : portal.y;
        const bool crouch = portal.capability == kTraversalCrouchable;
        if (crouch && !crouchTargetActive)
        {
            event("crouch_started", "reason=low_portal");
            crouchTargetActive = true;
        }
        return steerTo(targetX, targetY, false, false, portal.wall, portal.to, portal.capability);
    }

    GINPUT steerTo(int x, int y, bool use, bool shoot, int targetId = -1, int targetSector = -1,
                   TraversalCapability capability = kTraversalUnknown)
    {
        if (capability != kTraversalCrouchable && crouchTargetActive)
        {
            event("crouch_released", "reason=normal_clearance");
            crouchTargetActive = false;
        }
        setMovementTarget(x, y, targetSector, targetId, capability);
        updateMovementProgress();
        GINPUT input = {};
        const int targetAngle = getangle(x - observation.x, y - observation.y);
        const int delta = angleDelta(targetAngle, observation.angle);
        input.syncFlags.run = 1;
        if (capability == kTraversalCrouchable)
            input.buttonFlags.crouch = 1;
        // The bot has no mouse inertia to model. Aim the player/camera at the
        // target in one correction, then let the next frame issue Use once the
        // observed heading confirms alignment.
        input.q16turn = fix16_from_int(delta);
        const int targetDistance2 = distance2(observation.x, observation.y, x, y);
        if (std::abs(delta) < 96 && (!use || targetDistance2 > kUseStopRange * kUseStopRange))
            input.forward = 2047;
        if (use && targetDistance2 < kActionApproachRange * kActionApproachRange
            && std::abs(delta) < 96)
            input.keyFlags.action = 1;
        if (shoot)
            input.buttonFlags.shoot = 1;

        if (movementNeedsJump())
        {
            char detail[256];
            snprintf(detail, sizeof(detail),
                     "goal=%s target=%d sector=%d target_sector=%d capability=%d distance=%d dx=%d dy=%d dz=%d forward=%d turn=%d jump_attempts=%d height=%d cant_jump=%d posture=%d",
                     currentGoal.c_str(), movementTargetId, observation.sector, movementTargetSector,
                     int(movementTargetCapability),
                     int(std::sqrt(double(distance2(observation.x, observation.y, movementTargetX, movementTargetY)))),
                     observation.x - targetLastX, observation.y - targetLastY, observation.z - targetLastZ,
                     int(input.forward), int(input.q16turn), jumpAttempts,
                     gMe->pXSprite ? gMe->pXSprite->height : -1, gMe->cantJump, gMe->posture);
            event("movement_stuck", detail);
            input.buttonFlags.jump = 1;
            input.forward = 2047;
            ++jumpAttempts;
            if (currentGoal == "SEARCH_CURRENT_AREA"
                && movementTargetCapability == kTraversalJumpable)
                localDynamicJumpAttempted = true;
            jumpCooldownTick = observation.tick + kJumpCooldownTicks;
            event("jump_attempt", detail);
        }
        else if ((currentGoal == "EXPLORE_FRONTIER" || currentGoal == "EXPLORE_LOCAL_PORTAL")
                 && movementTargetActive
                 && jumpAttempts >= kMaxJumpAttemptsPerTarget && movementTargetId >= 0
                 && movementTargetSector >= 0)
        {
            const int edgeId = movementTargetId * 65536 + movementTargetSector;
            if (!edgeFailed(edgeId))
            {
                failedEdges[edgeId] = knowledgeRevision;
                char detail[96];
                snprintf(detail, sizeof(detail), "wall=%d from=%d to=%d attempts=%d",
                         movementTargetId, movementTargetFrom, movementTargetSector, jumpAttempts);
                event("frontier_failed", detail);
            }
            movementTargetActive = false;
        }
        else if (currentGoal == "EXPLORE_LOCAL_PORTAL" && movementTargetActive
                 && movementTargetCapability != kTraversalJumpable
                 && observation.tick - targetLastProgressTick >= kMovementStuckTicks)
        {
            const int edgeId = movementTargetId * 65536 + movementTargetSector;
            failedEdges[edgeId] = knowledgeRevision;
            char detail[96];
            snprintf(detail, sizeof(detail), "wall=%d from=%d to=%d reason=local_approach_blocked",
                     movementTargetId, movementTargetFrom, movementTargetSector);
            event("local_portal_failed", detail);
            movementTargetActive = false;
        }
        return input;
    }

    GINPUT decide()
    {
        GINPUT idle = {};
        if (!gMe || !gMe->pSprite || !gMe->pXSprite)
            return idle;
        if (result.size())
            return idle;

        if (const VisibleObject *enemy = selectObject(kObjectEnemy))
        {
            Portal route;
            int rangedWeapon = kWeaponNone;
            if (enemy->sector != observation.sector && findKnownRoute(enemy->sector, route))
            {
                setGoal("NAVIGATE_TO_VISIBLE_ENEMY", enemy->sprite);
                return steerPortal(route);
            }
            if (directObjectReachable(*enemy))
            {
                setGoal("COMBAT_MELEE_REACHABLE_ENEMY", enemy->sprite);
                return steerTo(enemy->x, enemy->y, false, true, enemy->sprite, enemy->sector);
            }
            if (rangedWeaponAvailable(rangedWeapon))
            {
                setGoal("COMBAT_RANGED_VISIBLE_ENEMY", enemy->sprite);
                return aimAndShoot(*enemy, rangedWeapon);
            }
            if (unreachableEnemies.insert(enemy->sprite).second)
            {
                char detail[96];
                snprintf(detail, sizeof(detail), "sprite=%d sector=%d dz=%d reason=no_route_or_ranged_weapon",
                         enemy->sprite, enemy->sector, enemy->z - observation.z);
                event("enemy_unreachable", detail);
            }
        }
        if (const VisibleObject *key = selectObject(kObjectKey))
        {
            Portal route;
            if (directObjectReachable(*key))
            {
                setGoal("COLLECT_VISIBLE_KEY", key->sprite);
                return steerTo(key->x, key->y, false, false, key->sprite, key->sector);
            }
            if (findKnownRoute(key->sector, route))
            {
                setGoal("NAVIGATE_TO_VISIBLE_KEY", key->sprite);
                return steerPortal(route);
            }
            unreachableObjects.insert(key->sprite);
        }
        if (const VisibleObject *interactive = selectObject(kObjectInteractive))
        {
            Portal route;
            if (directObjectReachable(*interactive))
            {
                setGoal("USE_VISIBLE_INTERACTIVE", interactive->sprite);
                return steerTo(interactive->x, interactive->y, true, false, interactive->sprite, interactive->sector);
            }
            if (findKnownRoute(interactive->sector, route))
            {
                setGoal("NAVIGATE_TO_VISIBLE_INTERACTIVE", interactive->sprite);
                return steerPortal(route);
            }
            unreachableObjects.insert(interactive->sprite);
        }
        if (const VisibleObject *pickup = selectObject(kObjectPickup))
        {
            Portal route;
            if (directObjectReachable(*pickup))
            {
                setGoal("COLLECT_VISIBLE_PICKUP", pickup->sprite);
                return steerTo(pickup->x, pickup->y, false, false, pickup->sprite, pickup->sector);
            }
            if (findKnownRoute(pickup->sector, route))
            {
                setGoal("NAVIGATE_TO_VISIBLE_PICKUP", pickup->sprite);
                return steerPortal(route);
            }
            unreachableObjects.insert(pickup->sprite);
        }
        if (observation.exitHere)
        {
            setGoal("USE_OBSERVED_EXIT", observation.sector);
            GINPUT input = {};
            input.keyFlags.action = 1;
            return input;
        }
        if (const Portal *door = selectKnownDoor())
        {
            const int actualDoorId = selectedDoorId;
            auto doorMemory = doors.find(actualDoorId);
            if (doorMemory == doors.end())
            {
                setGoal("NAVIGATE_TO_DOOR", actualDoorId);
                return steerPortal(*door);
            }

            DoorMemory &memory = doorMemory->second;
            // selectKnownDoor() may return an intermediate route portal. Do
            // not probe or mutate the route as if it were the remembered
            // interaction target; only the actual selected door may receive
            // a USE pulse.
            if (door->wall != actualDoorId)
            {
                setGoal("NAVIGATE_TO_DOOR", actualDoorId);
                return steerPortal(*door);
            }

            const Portal &expected = memory.portal;
            int hit = -1;
            int target = -1;
            int extra = -1;
            hit = ActionScanPreview(gMe, &target, &extra);
            const bool atDoor = actionTargetMatches(expected, hit, target);
            const int expectedAngle = getangle(expected.x - observation.x, expected.y - observation.y);
            const int expectedDistance = distance2(observation.x, observation.y, expected.x, expected.y);
            const bool missingKey = atDoor && expected.key && !hasKey(expected.key);
            const char *probeStatus = nullptr;
            if (memory.interaction == DoorMemory::kWaiting
                || memory.interaction == DoorMemory::kOpening)
                probeStatus = "OPENING";
            else if (memory.interaction == DoorMemory::kOpen)
                probeStatus = "OPEN";
            else if (missingKey)
                probeStatus = "LOCKED_KEY_REQUIRED";
            else if (atDoor)
                probeStatus = "VALID_ACTION_TARGET";
            else if (expectedDistance > kActionScanRange * kActionScanRange)
                probeStatus = "NOT_IN_USE_RANGE";
            else if (std::abs(angleDelta(expectedAngle, observation.angle)) >= 96)
                probeStatus = "NOT_FACING_ACTION_TARGET";
            else
                probeStatus = "NO_ACTION_TARGET";

            if (probeStatus == std::string("NO_ACTION_TARGET")
                && expectedDistance <= kActionScanRange * kActionScanRange
                && std::abs(angleDelta(expectedAngle, observation.angle)) < 96)
            {
                memory.availability = DoorMemory::kCurrentlyUnavailable;
                memory.unavailableReason = 3;
                memory.unavailableFromSector = observation.sector;
                probeStatus = "CURRENTLY_UNAVAILABLE_FROM_THIS_SIDE";
                char detail[128];
                snprintf(detail, sizeof(detail), "wall=%d reason=CURRENTLY_UNAVAILABLE_FROM_THIS_SIDE affordance=1",
                         expected.wall);
                event("portal_unavailable", detail);
            }
            emitUseProbe(expected, memory, atDoor, hit, target, extra, probeStatus);

            setGoal(missingKey ? "TRY_LOCKED_DOOR" : "REVISIT_LOCKED_DOOR", expected.wall);
            bool issueUse = false;
            if (atDoor)
            {
                const bool waiting = memory.interaction == DoorMemory::kWaiting
                    || memory.interaction == DoorMemory::kOpening;
                const bool retryAllowed = memory.interaction == DoorMemory::kIdle
                    || memory.interaction == DoorMemory::kFailed;
                if (!waiting && retryAllowed && !(missingKey && memory.attempts > 0))
                {
                    memory.interaction = DoorMemory::kWaiting;
                    memory.interactionStartedTick = observation.tick;
                    memory.interactionDeadlineTick = observation.tick + kInteractionTimeoutTicks;
                    ++memory.attempts;
                    issueUse = true;
                    char detail[160];
                    snprintf(detail, sizeof(detail),
                             "door=%d from=%d sector=%d target_sector=%d key=%d locked=%d wall_push=%d sector_push=%d attempt=%d",
                             expected.wall, expected.from, observation.sector, expected.to, expected.key,
                             expected.locked ? 1 : 0, expected.wallPush ? 1 : 0,
                             expected.sectorPush ? 1 : 0, memory.attempts);
                    event("interaction_started", detail);
                    event("use_pulse", detail);
                    event("interaction_waiting", detail);
                    event("door_use_attempt", detail);
                    lastDoorActionEventTick = observation.tick;
                    pendingUse = expected.wall;
                    pendingUseTick = observation.tick;
                    lastUsedDoor = expected.wall;
                    lastUsedDoorFrom = expected.from;
                    lastUsedDoorTo = expected.to;
                    lastUsedDoorTick = observation.tick;
                }
            }
            return steerTo(expected.x, expected.y, issueUse, false, expected.wall, expected.to);
        }
        if (const Portal *localPortal = selectLocalPortal())
        {
            const bool newLocalPortal = currentGoal != "EXPLORE_LOCAL_PORTAL"
                || currentGoalTarget != localPortal->wall;
            setGoal("EXPLORE_LOCAL_PORTAL", localPortal->wall);
            if (newLocalPortal)
            {
                char detail[160];
                snprintf(detail, sizeof(detail), "wall=%d from=%d to=%d x=%d y=%d capability=%d floor_delta=%d width=%d clearance=%d",
                         localPortal->wall, localPortal->from, localPortal->to,
                         localPortal->x, localPortal->y,
                         int(localPortal->capability), localPortal->floorDelta,
                         localPortal->openingWidth, localPortal->clearance);
                event("local_portal_target", detail);
            }
            return steerPortal(*localPortal);
        }
        if (const Portal *portal = selectPortal())
        {
            const bool newFrontier = currentGoal != "EXPLORE_FRONTIER" || currentGoalTarget != portal->wall;
            setGoal("EXPLORE_FRONTIER", portal->wall);
            if (newFrontier)
            {
                char detail[192];
                snprintf(detail, sizeof(detail), "wall=%d from=%d to=%d walkable=%d jumpable=%d width=%d floor_delta=%d clearance=%d",
                         portal->wall, portal->from, portal->to, portal->walkable ? 1 : 0,
                         portal->jumpable ? 1 : 0, portal->openingWidth, portal->floorDelta, portal->clearance);
                event("frontier_target", detail);
            }
            return steerPortal(*portal);
        }

        // A short deterministic turn-and-walk probe resolves frontiers that
        // were not visible from the previous sample without consulting the
        // loaded map's complete topology.
        setGoal("SEARCH_CURRENT_AREA", -2);
        idle.syncFlags.run = 1;
        if (searchAngle < 0)
            searchAngle = observation.angle;
        const TraversalCapability searchCapability = currentRecoveryCapability();
        const bool searchTargetStalled = movementTargetActive
            && movementTargetGoal == currentGoal
            && observation.tick - targetLastProgressTick >= kMovementStuckTicks;
        const bool reorientSearch = !movementTargetActive
            || movementTargetGoal != currentGoal
            || jumpAttempts >= kMaxJumpAttemptsPerTarget
            || (searchTargetStalled && searchCapability != kTraversalJumpable);
        if (reorientSearch)
        {
            if (movementTargetActive)
            {
                searchAngle = wrapAngle(searchAngle + 512);
                char detail[64];
                snprintf(detail, sizeof(detail), "angle=%d", searchAngle);
                event("search_reoriented", detail);
            }
            jumpAttempts = 0;
            const int probeX = observation.x + mulscale30(Cos(searchAngle), 8192);
            const int probeY = observation.y + mulscale30(Sin(searchAngle), 8192);
            if (searchCapability == kTraversalCrouchable)
                event("local_crouch_recovery", "reason=standing_clearance_unavailable");
            else if (searchCapability == kTraversalJumpable)
                event("local_dynamic_jump", "reason=moving_sector_with_vertical_motion");
            setMovementTarget(probeX, probeY, observation.sector, -2, searchCapability);
        }
        return steerTo(movementTargetX, movementTargetY, false, false, -2, observation.sector,
                       searchCapability);
    }

    void detectStall()
    {
        if (observation.tick - lastSemanticProgressTick > stallSeconds * kTicsPerSec)
        {
            result = "STALLED";
            failureReason = "no meaningful world or knowledge progress";
            event("failure", failureReason.c_str());
            gQuitGame = true;
            return;
        }

        const int quantized = (observation.x >> 10) ^ ((observation.y >> 10) << 10) ^ (observation.sector << 20);
        if (quantized == lastStateLocation && currentGoal == lastStateGoal
            && currentGoalTarget == lastStateTarget && knowledgeRevision == lastStateKnowledge
            && inventoryRevision == lastStateInventory)
            ++repeatedStateCount;
        else
            repeatedStateCount = 0;
        lastStateLocation = quantized;
        lastStateGoal = currentGoal;
        lastStateTarget = currentGoalTarget;
        lastStateKnowledge = knowledgeRevision;
        lastStateInventory = inventoryRevision;
        if (repeatedStateCount > 3 * kTicsPerSec)
        {
            result = "LOOP_DETECTED";
            failureReason = "repeated position/goal/knowledge state";
            event("failure", failureReason.c_str());
            gQuitGame = true;
        }
    }

    void close(const char *reason)
    {
        if (!result.size())
            result = reason ? reason : "RUNTIME_ERROR";
        fprintf(stderr,
                "LLMAPPER BOT TERMINATION result=%s reason=%s game_time=%d goal=%s target=%d sector=%d\n",
                result.c_str(), failureReason.c_str(), (gFrame * kTicsPerFrame) / kTicsPerSec,
                currentGoal.c_str(), currentGoalTarget, observation.sector);
        fflush(stderr);
        if (telemetry)
        {
            fprintf(telemetry, "{\"type\":\"summary\",\"result\":\"%s\",\"failure_reason\":\"%s\",\"game_time\":%d,\"visited_sectors\":%u,\"observed_sectors\":%u}\n",
                    result.c_str(), failureReason.c_str(), (gFrame * kTicsPerFrame) / kTicsPerSec,
                    unsigned(visitedSectors.size()), unsigned(observedSectors.size()));
            fclose(telemetry);
            telemetry = nullptr;
        }
        if (trajectory)
        {
            fclose(trajectory);
            trajectory = nullptr;
        }
        if (gDemo.at0)
            gDemo.Close();
    }
};

LLMapperBot::LLMapperBot()
    : m_impl(new Impl), m_enabled(false), m_fast(true), m_visible(false)
{
}

LLMapperBot::~LLMapperBot()
{
    Finish("RUNTIME_ERROR");
    delete m_impl;
}

void LLMapperBot::Enable(const char *telemetry, const char *trajectory, const char *demo)
{
    m_enabled = true;
    if (telemetry && *telemetry)
        m_impl->telemetryPath = telemetry;
    if (trajectory && *trajectory)
        m_impl->trajectoryPath = trajectory;
    if (demo && *demo)
        m_impl->demoPath = demo;
}

void LLMapperBot::ConfigureTimeout(int seconds)
{
    if (seconds > 0)
        m_impl->timeoutSeconds = seconds;
}

void LLMapperBot::ConfigureStallTimeout(int seconds)
{
    if (seconds > 0)
        m_impl->stallSeconds = seconds;
}

void LLMapperBot::SetFast(bool fast)
{
    m_fast = fast;
}

void LLMapperBot::SetVisible(bool visible)
{
    m_visible = visible;
    if (visible)
        m_fast = false;
}

void LLMapperBot::PrepareLaunch()
{
    if (!m_enabled)
        return;
    m_impl->openFiles();
    m_impl->event("run_started");
    if (!gDemo.at0 && !gDemo.at1 && !gDemo.Create(m_impl->demoPath.c_str()))
    {
        m_impl->result = "RUNTIME_ERROR";
        m_impl->failureReason = "could not create demo file";
        m_impl->event("runtime_error", m_impl->failureReason.c_str());
        gQuitGame = true;
    }
}

GINPUT LLMapperBot::GetInput()
{
    if (!m_enabled)
        return {};
    if (m_impl->lastObservationTick != gFrame * kTicsPerFrame)
    {
        m_impl->updateKnowledge();
        m_impl->lastObservationTick = gFrame * kTicsPerFrame;
    }
    return m_impl->decide();
}

void LLMapperBot::OnFrame()
{
    if (!m_enabled || !gGameStarted || !gMe || !gMe->pXSprite)
        return;
    m_impl->updateKnowledge();
    m_impl->lastObservationTick = gFrame * kTicsPerFrame;
    if ((gGameOptions.uGameFlags & kGameFlagContinuing) && m_impl->result.empty())
        OnLevelExit(kLevelExitNormal);
    if (m_impl->lastTrajectoryTick < gFrame * kTicsPerFrame - kTrajectoryPeriod)
    {
        m_impl->trajectorySample();
        m_impl->lastTrajectoryTick = gFrame * kTicsPerFrame;
    }
    if (m_impl->observation.tick % (kObservationPeriod * kTicsPerFrame) == 0)
        m_impl->detectStall();
    if (m_impl->observation.tick >= m_impl->timeoutSeconds * kTicsPerSec)
    {
        m_impl->result = "TIMEOUT";
        m_impl->failureReason = "simulated time limit reached";
        m_impl->event("failure", m_impl->failureReason.c_str());
        gQuitGame = true;
    }
    if (m_impl->observation.health == 0 && m_impl->result.empty())
    {
        m_impl->result = "DIED";
        m_impl->failureReason = "player health reached zero";
        m_impl->event("failure", m_impl->failureReason.c_str());
        gQuitGame = true;
    }
}

void LLMapperBot::OnActionResolved(int hit, int target, int extra, bool accepted, int key)
{
    if (m_enabled)
        m_impl->actionResolved(hit, target, extra, accepted, key);
}

void LLMapperBot::OnLevelExit(int exitType)
{
    if (!m_enabled || !m_impl->result.empty())
        return;
    m_impl->result = "COMPLETED";
    char detail[48];
    snprintf(detail, sizeof(detail), "exit_type=%d", exitType);
    m_impl->event("level_completed", detail);
    gQuitGame = true;
}

void LLMapperBot::Finish(const char *reason)
{
    if (m_enabled && m_impl->telemetry)
        m_impl->close(reason);
}

LLMapperBot gLLMapperBot;
