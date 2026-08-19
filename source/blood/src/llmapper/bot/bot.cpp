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
#include <deque>
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

enum NavEdgeMode
{
    kNavWalk,
    kNavStep,
    kNavJump,
    kNavCrouch,
    kNavDrop,
    kNavInteraction,
    kNavBlocked,
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
constexpr int kMaxWalkableStep = 4096;
constexpr int kActionScanRange = 1024;
constexpr int kActionApproachRange = 2048;
constexpr int kUseStopRange = kActionApproachRange;
constexpr int kInteractionTimeoutTicks = 6 * kTicsPerSec;
constexpr int kLookUpLimit = 289;
constexpr int kLookDownLimit = -347;

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
    int interactionTopZ = 0;
    int interactionBottomZ = 0;
    bool interactionCrouch = false;
};

enum InteractionKind
{
    kInteractionWall,
    kInteractionSector,
    kInteractionSprite,
};

struct InteractionCandidate
{
    InteractionKind kind = kInteractionWall;
    int id = -1;
    int fromSector = -1;
    int targetSector = -1;
    int x = 0;
    int y = 0;
    int z = 0;
    int key = 0;
    bool locked = false;
    bool reversible = false;
    Portal target;
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
    int playerCeilingDistance = 0;
    int playerFloorDistance = 0;
    int localSectorExtra = 0;
    int localSectorState = 0;
    int localSectorBusy = 0;
    int playerZVelocity = 0;
    std::vector<int> visibleSectors;
    std::vector<Portal> portals;
    std::vector<InteractionCandidate> interactions;
    std::vector<VisibleObject> objects;
};

struct LocalWaypoint
{
    int x = 0;
    int y = 0;
};

struct MovementProbe
{
    bool reachable = false;
    int wall = -1;
    int hit = 0;
    int x = 0;
    int y = 0;
    int sector = -1;
};

static const char *navEdgeModeName(NavEdgeMode mode)
{
    switch (mode)
    {
    case kNavStep: return "STEP";
    case kNavJump: return "JUMP";
    case kNavCrouch: return "CROUCH";
    case kNavDrop: return "DROP_SAFE";
    case kNavInteraction: return "INTERACTION";
    case kNavBlocked: return "BLOCKED";
    default: return "WALK";
    }
}

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

static void playerCollisionDistances(int &ceilingDistance, int &floorDistance);

static int playerBodyClearance()
{
    int ceilingDistance = 0;
    int floorDistance = 0;
    playerCollisionDistances(ceilingDistance, floorDistance);
    return std::max(1, (ceilingDistance + floorDistance) * 4);
}

static int playerCrouchClearance()
{
    // Blood changes the view posture when crouching, but playerProcess()
    // still uses the same sprite extents for collision.  Keep both portal
    // capability checks grounded in that authoritative body envelope.
    return playerBodyClearance();
}

static void playerCollisionDistances(int &ceilingDistance, int &floorDistance)
{
    ceilingDistance = 64 << 4;
    floorDistance = 64 << 4;
    if (!gMe || !gMe->pSprite)
        return;

    // Keep copied-state probes identical to playerProcess().  The engine
    // derives these distances from the player's actual sprite extents; the
    // posture eye height is not the collision height.
    int top = gMe->pSprite->z;
    int bottom = gMe->pSprite->z;
    GetSpriteExtents(gMe->pSprite, &top, &bottom);
    ceilingDistance = std::max(0, (gMe->pSprite->z - top) / 4);
    floorDistance = std::max(0, (bottom - gMe->pSprite->z) / 4);
}

static int lookAngleForTarget(int eyeZ, int targetZ, int horizontal)
{
    const double angle = std::atan2(double(eyeZ - targetZ), double(std::max(1, horizontal)))
        * 1024.0 / 3.14159265358979323846;
    return std::max(kLookDownLimit, std::min(kLookUpLimit, int(std::lround(angle))));
}

static void setInteractionGeometry(Portal &portal)
{
    if (!inRange(portal.from, 0, numsectors))
        return;
    const int fromFloor = getflorzofslope(portal.from, portal.x, portal.y);
    const int fromCeiling = getceilzofslope(portal.from, portal.x, portal.y);
    int bottom = fromFloor;
    int top = fromCeiling;
    if (inRange(portal.to, 0, numsectors))
    {
        bottom = std::min(bottom, getflorzofslope(portal.to, portal.x, portal.y));
        top = std::max(top, getceilzofslope(portal.to, portal.x, portal.y));
    }
    if (bottom <= top)
    {
        bottom = fromFloor;
        top = fromCeiling;
    }
    portal.interactionBottomZ = bottom;
    portal.interactionTopZ = top;
    portal.z = (top + bottom) / 2;
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
        playerCollisionDistances(result.playerCeilingDistance, result.playerFloorDistance);
        if (current.extra > 0 && current.extra < kMaxXSectors && xsector[current.extra].Exit)
            result.exitHere = true;

        // A current-sector Push is an ActionScan target in its own right.
        // It is not represented by a portal and must be remembered even when
        // every surrounding wall is one-sided.
        if (current.extra > 0 && current.extra < kMaxXSectors && xsector[current.extra].Push)
        {
            InteractionCandidate candidate;
            candidate.kind = kInteractionSector;
            candidate.id = result.sector;
            candidate.fromSector = result.sector;
            candidate.targetSector = result.sector;
            candidate.x = result.x;
            candidate.y = result.y;
            candidate.z = result.z;
            candidate.reversible = true;
            candidate.target.from = result.sector;
            candidate.target.to = result.sector;
            candidate.target.x = result.x;
            candidate.target.y = result.y;
            candidate.target.sectorPushCurrent = true;
            setInteractionGeometry(candidate.target);
            candidate.z = candidate.target.z;
            result.interactions.push_back(candidate);
        }

        for (int i = 0; i < current.wallnum; ++i)
        {
            const int wallIndex = current.wallptr + i;
            const walltype &wallRecord = wall[wallIndex];
            const walltype &nextWall = wall[wallRecord.point2];
            if (!inRange(wallRecord.nextsector, 0, numsectors))
            {
                // XWALL triggerPush is usable without a nextsector.  It is
                // discovered from the reachable current sector, then the
                // real ActionScan preview remains the activation authority.
                if (wallRecord.extra > 0 && wallRecord.extra < kMaxXWalls
                    && xwall[wallRecord.extra].triggerPush
                    && cansee(result.x, result.y, result.z, result.sector,
                              (wallRecord.x + nextWall.x) / 2,
                              (wallRecord.y + nextWall.y) / 2,
                              result.z, result.sector))
                {
                    InteractionCandidate candidate;
                    candidate.kind = kInteractionWall;
                    candidate.id = wallIndex;
                    candidate.fromSector = result.sector;
                    candidate.x = (wallRecord.x + nextWall.x) / 2;
                    candidate.y = (wallRecord.y + nextWall.y) / 2;
                    candidate.z = result.z;
                    candidate.key = xwall[wallRecord.extra].key;
                    candidate.locked = xwall[wallRecord.extra].locked != 0;
                    candidate.reversible = true;
                    candidate.target.wall = wallIndex;
                    candidate.target.from = result.sector;
                    candidate.target.to = -1;
            candidate.target.x = candidate.x;
            candidate.target.y = candidate.y;
            candidate.target.wallPush = true;
            candidate.target.x1 = wallRecord.x;
            candidate.target.y1 = wallRecord.y;
            candidate.target.x2 = nextWall.x;
            candidate.target.y2 = nextWall.y;
            setInteractionGeometry(candidate.target);
            candidate.z = candidate.target.z;
            result.interactions.push_back(candidate);
                }
                continue;
            }
            ++result.localPortalCount;

            const int midX = (wallRecord.x + nextWall.x) / 2;
            const int midY = (wallRecord.y + nextWall.y) / 2;
            const int fromFloor = getflorzofslope(result.sector, midX, midY);
            const int toFloor = getflorzofslope(wallRecord.nextsector, midX, midY);
            const int fromCeiling = getceilzofslope(result.sector, midX, midY);
            const int toCeiling = getceilzofslope(wallRecord.nextsector, midX, midY);
            const int openingWidth = int(std::sqrt(double(distance2(wallRecord.x, wallRecord.y, nextWall.x, nextWall.y))));
            const int clearance = std::min(fromFloor - fromCeiling, toFloor - toCeiling);
            const int floorDelta = toFloor - fromFloor;
            const int bodyClearance = playerBodyClearance();
            const int crouchClearance = playerCrouchClearance();
            const int jumpRiseLimit = playerJumpRiseLimit();
            const int dropLimit = jumpRiseLimit;
            const bool enoughWidth = openingWidth >= kPlayerPassageWidth;
            const bool standingClearance = clearance >= bodyClearance;
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

            if (portal.interactionAffordance)
            {
                InteractionCandidate candidate;
                candidate.kind = kInteractionWall;
                candidate.id = wallIndex;
                candidate.fromSector = result.sector;
                candidate.targetSector = portal.to;
                candidate.x = midX;
                candidate.y = midY;
                candidate.z = midZ;
                candidate.key = portal.key;
                candidate.locked = portal.locked;
                candidate.reversible = true;
                candidate.target = portal;
                setInteractionGeometry(candidate.target);
                candidate.x = candidate.target.x;
                candidate.y = candidate.target.y;
                candidate.z = candidate.target.z;
                result.interactions.push_back(candidate);
            }

            // A compact mechanism room can contain one-sided XWALL push
            // surfaces immediately beyond an otherwise ordinary portal.  A
            // player can see and use those surfaces before the portal is
            // physically consumed, while they are not present in the
            // current sector's wall list.  Discover only the adjacent
            // sector's currently visible trigger walls; ActionScanPreview
            // still decides whether the current pose can actually use one.
            if (inRange(portal.to, 0, numsectors))
            {
                const sectortype &adjacent = sector[portal.to];
                for (int j = 0; j < adjacent.wallnum; ++j)
                {
                    const int adjacentWallIndex = adjacent.wallptr + j;
                    const walltype &adjacentWall = wall[adjacentWallIndex];
                    const walltype &adjacentNext = wall[adjacentWall.point2];
                    if (inRange(adjacentWall.nextsector, 0, numsectors)
                        || !inRange(adjacentWall.extra, 1, kMaxXWalls)
                        || !xwall[adjacentWall.extra].triggerPush)
                        continue;
                    const int adjacentX = (adjacentWall.x + adjacentNext.x) / 2;
                    const int adjacentY = (adjacentWall.y + adjacentNext.y) / 2;
                    const bool visibleFromPlayer = cansee(
                        result.x, result.y, result.z, result.sector,
                        adjacentX, adjacentY, result.z, portal.to);
                    if (!visibleFromPlayer)
                        continue;
                    InteractionCandidate candidate;
                    candidate.kind = kInteractionWall;
                    candidate.id = adjacentWallIndex;
                    candidate.fromSector = result.sector;
                    candidate.targetSector = portal.to;
                    candidate.x = adjacentX;
                    candidate.y = adjacentY;
                    candidate.z = result.z;
                    candidate.key = xwall[adjacentWall.extra].key;
                    candidate.locked = xwall[adjacentWall.extra].locked != 0;
                    candidate.reversible = true;
                    candidate.target.wall = adjacentWallIndex;
                    candidate.target.from = result.sector;
                    candidate.target.to = -1;
                    candidate.target.x = adjacentX;
                    candidate.target.y = adjacentY;
                    candidate.target.wallPush = true;
                    candidate.target.x1 = adjacentWall.x;
                    candidate.target.y1 = adjacentWall.y;
                    candidate.target.x2 = adjacentNext.x;
                    candidate.target.y2 = adjacentNext.y;
                    setInteractionGeometry(candidate.target);
                    candidate.z = candidate.target.z;
                    result.interactions.push_back(candidate);
                }
            }

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
        if (object.kind == kObjectInteractive && validXSprite(candidate.extra)
            && xsprite[candidate.extra].Push)
        {
            InteractionCandidate interaction;
            interaction.kind = kInteractionSprite;
            interaction.id = i;
            interaction.fromSector = candidate.sectnum;
            interaction.targetSector = candidate.sectnum;
            interaction.x = candidate.x;
            interaction.y = candidate.y;
            interaction.z = candidate.z;
            interaction.reversible = true;
            interaction.target.wall = -1;
            interaction.target.from = candidate.sectnum;
            interaction.target.to = candidate.sectnum;
            interaction.target.x = candidate.x;
            interaction.target.y = candidate.y;
            setInteractionGeometry(interaction.target);
            interaction.z = interaction.target.z;
            result.interactions.push_back(interaction);
        }
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
    enum ObjectiveType
    {
        kObjectiveNone,
        kObjectivePickup,
        kObjectiveKey,
        kObjectiveInteraction,
        kObjectiveFrontier,
        kObjectiveCombat,
    };

    struct Objective
    {
        ObjectiveType type = kObjectiveNone;
        bool active = false;
        int id = -1;
        int sector = -1;
        int targetSector = -1;
        int x = 0;
        int y = 0;
        int z = 0;
        int interactionKey = -1;
        int wall = -1;
    };

    struct ObjectMemory
    {
        VisibleObject object;
        bool observed = false;
        bool collected = false;
        int lastSeenTick = -1;
    };

    struct NavLink
    {
        int target = -1;
        NavEdgeMode mode = kNavWalk;
        int wall = -1;
        LocalWaypoint gateway;
        bool hasGateway = false;
    };

    struct NavCell
    {
        int id = -1;
        int sector = -1;
        LocalWaypoint vertex[3];
        LocalWaypoint center;
        int walkArea = -1;
        std::vector<NavLink> links;
    };

    struct EdgeFailure
    {
        int geometrySignature = 0;
        int attempts = 0;
    };

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
        bool engineAccepted = false;
        int unavailableReason = 0;
        InteractionState interaction = kIdle;
        int interactionStartedTick = -1;
        int interactionDeadlineTick = -1;
        int unavailableFromSector = -1;
    };

    struct InteractionMemory
    {
        InteractionKind kind = kInteractionWall;
        int id = -1;
        int fromSector = -1;
        int targetSector = -1;
        int x = 0;
        int y = 0;
        int z = 0;
        int key = 0;
        bool locked = false;
        bool reversible = false;
        bool observed = false;
        bool attempted = false;
        bool activated = false;
        int activationCount = 0;
        int lastActivationTick = -1;
        int beforeState = 0;
        int afterState = 0;
        int unavailableFromSector = -1;
        int unavailableState = 0;
        int unavailableAttempts = 0;
        int unavailablePose = 0;
        int state = 0;
        bool engineAccepted = false;
        bool observedLocalEffect = false;
        bool observedKnownWorldDelta = false;
        bool recoveryAttempted = false;
        bool traversed = false;
        std::vector<Portal> beforePortals;
        Portal target;
    };

    struct PendingWork
    {
        enum FrontierState
        {
            kDiscovered,
            kActive,
            kConsumed,
            kTemporarilyUnavailable,
            kStale,
        };

        int kind = 0; // 1 interaction, 2 frontier
        int id = -1;
        int sector = -1;
        int edgeTarget = -1;
        FrontierState state = kDiscovered;
        int stateSignature = 0;
        bool resolved = false; // retained for old telemetry/state compatibility
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
    int lastEngineMoveHit = -1;
    int repeatedStateCount = 0;
    int pendingUse = -1;
    int pendingUseTick = -1;
    int pendingInteractionKey = -1;
    std::string result;
    std::string failureReason;
    std::string currentGoal;
    Objective currentObjective;
    Objective suspendedObjective;
    bool suspendedObjectiveActive = false;
    Observation observation;
    std::set<int> observedSectors;
    std::set<int> visitedSectors;
    std::map<int, DoorMemory> doors;
    std::map<int, InteractionMemory> interactions;
    std::vector<PendingWork> pendingWork;
    std::map<int, ObjectMemory> objectMemory;
    VisibleObject selectedObject;
    bool recoveryMode = false;
    std::set<int> knownObjects;
    std::set<int> knownKeys;
    std::set<int> unreachableObjects;
    std::set<int> unreachableEnemies;
    std::set<int> seenEdges;
    std::set<int> visitedEdges;
    std::map<int, EdgeFailure> failedEdges;
    std::map<int, int> localFailureSignatures;
    std::set<int> jumpFallbackEdges;
    std::set<int> openedRoutes;
    std::map<int, std::vector<Portal>> knownGraph;
    Portal routePortal;
    Portal localJumpPortal;
    int knowledgeRevision = 0;
    int inventoryRevision = 0;
    int lastObservedSector = -1;
    int lastGeometryTelemetrySector = -1;
    bool localDynamicJumpAttempted = false;
    int localDynamicJumpUntilTick = -1;
    int lastTransitionFrom = -1;
    int lastTransitionTo = -1;
    int lastTransitionTick = -1;
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
    int targetBestDistance2 = 0;
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
    std::vector<LocalWaypoint> portalPlan;
    size_t portalPlanIndex = 0;
    int portalPlanWall = -1;
    int portalPlanSignature = 0;
    int portalPlanAttempts = 0;
    bool portalCrossingActive = false;
    int portalCrossingX = 0;
    int portalCrossingY = 0;
    int portalCrossingDestinationX = 0;
    int portalCrossingDestinationY = 0;
    bool portalCrossingDestinationValid = false;
    int portalCrossingStartTick = -1;
    bool navigationActive = false;
    int navigationOriginalX = 0;
    int navigationOriginalY = 0;
    int navigationOriginalZ = 0;
    int navigationOriginalSector = -1;
    int navigationOriginalId = -1;
    int navigationOriginalSignature = 0;
    int navigationDetourWall = -1;
    int navigationDetourDepth = 0;
    int navigationWaypointX = 0;
    int navigationWaypointY = 0;
    bool navigationWaypointActive = false;
    bool navigationUsingDetour = false;
    int navigationActionableBlocker = -1;
    std::set<int> navigationRecentWalls;
    std::vector<LocalWaypoint> navRoute;
    size_t navRouteIndex = 0;
    int navRouteSignature = 0;
    int navRouteTopologyRevision = 0;
    std::vector<NavCell> navCells;
    int navTopologySignature = 0;
    int navTopologyRevision = 0;
    int navigationFailureSignature = 0;
    int navigationFailureCount = 0;
    int lastNavigationFailureTick = -1;
    int lastFrontierWall = -1;
    int lastFrontierSource = -1;
    int lastFrontierTarget = -1;
    int frontierSelectionCount = 0;
    bool crouchTargetActive = false;
    int followThroughKey = -1;
    int followThroughStartTick = -1;
    int followThroughFrom = -1;
    Portal followThroughPortal;
    bool cryptExitProven = false;
    int lastObservedHealth = -1;
    int lastDamageTick = -1;
    int lastInteractionWaitKey = -1;

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

    void retirePendingInteraction(int key)
    {
        for (PendingWork &pending : pendingWork)
        {
            if (pending.kind == 1 && pending.id == key)
                pending.resolved = true;
        }
    }

    void retirePendingFrontier(int wall, int from, int to)
    {
        for (PendingWork &pending : pendingWork)
        {
            if (pending.kind == 2 && pending.id == wall
                && pending.sector == from && pending.edgeTarget == to)
            {
                pending.state = PendingWork::kConsumed;
                pending.resolved = true;
            }
        }
    }

    PendingWork *findPendingFrontier(int wall, int from, int to)
    {
        for (PendingWork &pending : pendingWork)
            if (pending.kind == 2 && pending.id == wall
                && pending.sector == from && pending.edgeTarget == to)
                return &pending;
        return nullptr;
    }

    bool pendingFrontierLive(const PendingWork &pending) const
    {
        if (pending.resolved || pending.kind != 2
            || pending.state == PendingWork::kConsumed
            || pending.state == PendingWork::kStale)
            return false;
        if (pending.sector == observation.sector)
        {
            for (const Portal &portal : observation.portals)
            {
                if (portal.wall == pending.id && portal.to == pending.edgeTarget)
                    return portal.traversable || portal.jumpable;
            }
            return false;
        }
        auto graph = knownGraph.find(pending.sector);
        if (graph == knownGraph.end())
            return false;
        return std::any_of(graph->second.begin(), graph->second.end(), [&pending](const Portal &portal)
        {
            return portal.wall == pending.id && portal.to == pending.edgeTarget
                && (portal.traversable || portal.jumpable);
        });
    }

    bool pendingWorkLive(const PendingWork &pending) const
    {
        if (pending.resolved)
            return false;
        if (pending.kind == 2)
            return pendingFrontierLive(pending);
        if (pending.kind != 1)
            return false;
        auto memory = interactions.find(pending.id);
        return memory != interactions.end() && memory->second.observed
            && !memory->second.attempted;
    }

    int interactionSourceSignature(const InteractionMemory &memory) const
    {
        return interactionStateSignature(memory);
    }

    void emitInteractionPortalState(const InteractionMemory &memory, const char *phase)
    {
        if (memory.target.wall < 0 || !inRange(memory.target.from, 0, numsectors)
            || !inRange(memory.target.wall, 0, numwalls))
            return;
        const walltype &wallRecord = wall[memory.target.wall];
        if (!inRange(wallRecord.nextsector, 0, numsectors))
            return;
        const walltype &nextWall = wall[wallRecord.point2];
        const int midX = (wallRecord.x + nextWall.x) / 2;
        const int midY = (wallRecord.y + nextWall.y) / 2;
        const int fromFloor = getflorzofslope(memory.target.from, midX, midY);
        const int fromCeiling = getceilzofslope(memory.target.from, midX, midY);
        const int toFloor = getflorzofslope(wallRecord.nextsector, midX, midY);
        const int toCeiling = getceilzofslope(wallRecord.nextsector, midX, midY);
        const int toSector = wallRecord.nextsector;
        const int cstat = wallRecord.cstat;
        const int width = int(std::sqrt(double(distance2(wallRecord.x, wallRecord.y,
                                                      nextWall.x, nextWall.y))));
        const int clearance = std::min(fromFloor - fromCeiling, toFloor - toCeiling);
        char detail[256];
        snprintf(detail, sizeof(detail),
                 "phase=%s wall=%d from=%d to=%d cstat=%d floor_delta=%d clearance=%d width=%d observed_portal=%d",
                 phase, memory.target.wall, memory.target.from, toSector,
                 cstat, toFloor - fromFloor, clearance, width,
                 std::any_of(observation.portals.begin(), observation.portals.end(),
                             [&wallRecord](const Portal &portal)
                             {
                                 return portal.wall == (&wallRecord - wall)
                                     && portal.to == wallRecord.nextsector;
                             }) ? 1 : 0);
        event("interaction_portal_state", detail);
    }

    void emitInteractionGeometryState(const InteractionMemory &memory, const char *phase)
    {
        if (memory.target.wall < 0 || !inRange(memory.target.wall, 0, numwalls))
            return;
        const walltype &wallRecord = wall[memory.target.wall];
        if (!inRange(wallRecord.point2, 0, numwalls))
            return;
        const walltype &nextWall = wall[wallRecord.point2];
        const int width = int(std::sqrt(double(distance2(wallRecord.x, wallRecord.y,
                                                      nextWall.x, nextWall.y))));
        char detail[320];
        snprintf(detail, sizeof(detail),
                 "phase=%s kind=%d id=%d wall=%d from=%d to=%d x1=%d y1=%d x2=%d y2=%d width=%d target_top_z=%d target_bottom_z=%d crouch=%d",
                 phase, int(memory.kind), memory.id, memory.target.wall,
                 memory.target.from, memory.target.to, int(wallRecord.x), int(wallRecord.y),
                 int(nextWall.x), int(nextWall.y), width, memory.target.interactionTopZ,
                 memory.target.interactionBottomZ, memory.target.interactionCrouch ? 1 : 0);
        event("interaction_geometry_state", detail);
    }

    void updateFollowThrough(InteractionMemory &memory)
    {
        if (!memory.engineAccepted || memory.beforePortals.empty())
            return;
        for (const Portal &portal : observation.portals)
        {
            if (portal.from != memory.fromSector || portal.to == portal.from)
                continue;
            if (memory.target.wall >= 0 && portal.wall != memory.target.wall
                && memory.target.to != portal.to)
                continue;
            if (memory.target.to >= 0 && memory.target.to != portal.to)
                continue;
            if (!portal.traversable && !portal.jumpable)
                continue;
            bool wasTraversable = false;
            for (const Portal &before : memory.beforePortals)
            {
                if (before.wall == portal.wall && before.to == portal.to)
                {
                    wasTraversable = before.traversable || before.jumpable;
                    break;
                }
            }
            if (wasTraversable)
                continue;

            followThroughKey = interactionMemoryKey(memory);
            followThroughStartTick = observation.tick;
            followThroughFrom = portal.from;
            followThroughPortal = portal;
            char detail[192];
            snprintf(detail, sizeof(detail),
                     "kind=%d id=%d wall=%d from=%d to=%d before_traversable=0 after_traversable=%d floor_delta=%d clearance=%d",
                     int(memory.kind), memory.id, portal.wall, portal.from, portal.to,
                     portal.traversable || portal.jumpable ? 1 : 0,
                     portal.floorDelta, portal.clearance);
            event("interaction_follow_through_ready", detail);
            return;
        }
    }

    void actionResolved(int hit, int target, int extra, bool accepted, int key)
    {
        const char *kind = hit == 0 ? "wall" : hit == 3 ? "sprite" : hit == 6 ? "sector" : "none";
        char detail[256];
        snprintf(detail, sizeof(detail),
                 "source=gameplay_tick hit=%d kind=%s target=%d extra=%d trigger_dispatched=%d key=%d goal=%s",
                 hit, kind, target, extra, accepted ? 1 : 0, key,
                 currentGoal.c_str());
        event("engine_use_resolved", detail);
        for (auto &entry : interactions)
        {
            InteractionMemory &memory = entry.second;
            if (entry.first == pendingInteractionKey && memory.state == 1
                && interactionTargetMatches(memory, hit, target))
            {
                int sourceExtra = -1;
                int sourceState = -1;
                int sourceBusy = 0;
                if (hit == 6 && inRange(target, 0, numsectors))
                {
                    sourceExtra = sector[target].extra;
                    if (sourceExtra > 0 && sourceExtra < kMaxXSectors)
                    {
                        sourceState = xsector[sourceExtra].state;
                        sourceBusy = xsector[sourceExtra].busy;
                    }
                }
                else if (hit == 0 && inRange(target, 0, numwalls))
                {
                    sourceExtra = wall[target].extra;
                    if (sourceExtra > 0 && sourceExtra < kMaxXWalls)
                    {
                        sourceState = xwall[sourceExtra].state;
                        sourceBusy = xwall[sourceExtra].busy;
                    }
                }
                else if (hit == 3 && inRange(target, 0, kMaxSprites))
                {
                    sourceExtra = sprite[target].extra;
                    if (sourceExtra > 0 && sourceExtra < kMaxXSprites)
                    {
                        sourceState = xsprite[sourceExtra].state;
                        sourceBusy = xsprite[sourceExtra].busy;
                    }
                }
                char interactionDetail[320];
                memory.engineAccepted = accepted;
                if (accepted)
                {
                    memory.activated = true;
                    memory.observedKnownWorldDelta = false;
                    retirePendingInteraction(entry.first);
                }
                else
                    memory.state = 3;
                snprintf(interactionDetail, sizeof(interactionDetail),
                         "kind=%d id=%d from_sector=%d player_sector=%d hit=%d target=%d extra=%d accepted=%d source_extra=%d source_state=%d source_busy=%d source_signature_before=%d source_signature_after=%d",
                         int(memory.kind), memory.id, memory.fromSector, observation.sector,
                         hit, target, extra, accepted ? 1 : 0, sourceExtra, sourceState,
                         sourceBusy, memory.beforeState, interactionSourceSignature(memory));
                event("interaction_engine_resolved", interactionDetail);
                emitInteractionPortalState(memory, "after_process_input");
                emitInteractionGeometryState(memory, "after_process_input");

                if (memory.target.wall >= 0)
                {
                    auto door = doors.find(memory.target.wall);
                    if (door != doors.end())
                    {
                        door->second.unavailableFromSector = -1;
                        door->second.interaction = accepted
                            ? DoorMemory::kWaiting : DoorMemory::kFailed;
                        door->second.interactionDeadlineTick = accepted
                            ? observation.tick + kInteractionTimeoutTicks : -1;
                    }
                }
            }
        }
        if (lastUsedDoor >= 0)
        {
            auto door = doors.find(lastUsedDoor);
            if (door != doors.end())
            {
                door->second.engineAccepted = accepted;
                if (accepted)
                    door->second.interaction = DoorMemory::kWaiting;
            }
        }
        pendingInteractionKey = -1;
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

    int interactionKey(const InteractionCandidate &candidate) const
    {
        // Several linedefs can expose the same XSECTOR Wallpush target.
        // Their engine identity is the sector mechanism, not each duplicate
        // wall record, so one activation must be remembered once.
        if (candidate.kind == kInteractionWall && candidate.target.sectorPush)
            return int(kInteractionSector) * 1000000 + candidate.target.to + 1;
        if (candidate.kind == kInteractionWall && candidate.target.sectorPushCurrent)
            return int(kInteractionSector) * 1000000 + candidate.target.from + 1;
        return int(candidate.kind) * 1000000 + candidate.id + 1;
    }

    int interactionMemoryKey(const InteractionMemory &memory) const
    {
        if (memory.kind == kInteractionWall && memory.target.sectorPush)
            return int(kInteractionSector) * 1000000 + memory.target.to + 1;
        if (memory.kind == kInteractionWall && memory.target.sectorPushCurrent)
            return int(kInteractionSector) * 1000000 + memory.target.from + 1;
        return int(memory.kind) * 1000000 + memory.id + 1;
    }

    int interactionStateSignature(const InteractionMemory &memory) const
    {
        int signature = int(memory.kind) * 131 + memory.id;
        if (memory.kind == kInteractionWall && memory.target.sectorPush
            && inRange(memory.target.to, 0, numsectors))
        {
            const sectortype &sectorRecord = sector[memory.target.to];
            signature = signature * 31 + sectorRecord.floorz;
            signature = signature * 31 + sectorRecord.ceilingz;
            if (sectorRecord.extra > 0 && sectorRecord.extra < kMaxXSectors)
            {
                signature = signature * 31 + xsector[sectorRecord.extra].state;
                signature = signature * 31 + xsector[sectorRecord.extra].busy;
            }
        }
        else if (memory.kind == kInteractionWall && memory.target.sectorPushCurrent
                 && inRange(memory.target.from, 0, numsectors))
        {
            const sectortype &sectorRecord = sector[memory.target.from];
            signature = signature * 31 + sectorRecord.floorz;
            signature = signature * 31 + sectorRecord.ceilingz;
            if (sectorRecord.extra > 0 && sectorRecord.extra < kMaxXSectors)
            {
                signature = signature * 31 + xsector[sectorRecord.extra].state;
                signature = signature * 31 + xsector[sectorRecord.extra].busy;
            }
        }
        else if (memory.kind == kInteractionWall && inRange(memory.target.wall, 0, numwalls))
        {
            const walltype &wallRecord = wall[memory.target.wall];
            signature = signature * 31 + wallRecord.cstat;
            signature = signature * 31 + wallRecord.nextsector;
            signature = signature * 31 + wallRecord.x;
            signature = signature * 31 + wallRecord.y;
            if (inRange(wallRecord.point2, 0, numwalls))
            {
                signature = signature * 31 + wall[wallRecord.point2].x;
                signature = signature * 31 + wall[wallRecord.point2].y;
            }
            if (wallRecord.extra > 0 && wallRecord.extra < kMaxXWalls)
            {
                signature = signature * 31 + xwall[wallRecord.extra].state;
                signature = signature * 31 + xwall[wallRecord.extra].busy;
            }
        }

        else if (memory.kind == kInteractionSector && inRange(memory.fromSector, 0, numsectors))
        {
            const sectortype &sectorRecord = sector[memory.fromSector];
            signature = signature * 31 + sectorRecord.floorz;
            signature = signature * 31 + sectorRecord.ceilingz;
            if (sectorRecord.extra > 0 && sectorRecord.extra < kMaxXSectors)
            {
                signature = signature * 31 + xsector[sectorRecord.extra].state;
                signature = signature * 31 + xsector[sectorRecord.extra].busy;
            }
        }
        else if (memory.kind == kInteractionSprite && inRange(memory.id, 0, kMaxSprites))
        {
            const spritetype &spriteRecord = sprite[memory.id];
            signature = signature * 31 + spriteRecord.x;
            signature = signature * 31 + spriteRecord.y;
            signature = signature * 31 + spriteRecord.sectnum;
            if (validXSprite(spriteRecord.extra))
            {
                signature = signature * 31 + xsprite[spriteRecord.extra].state;
                signature = signature * 31 + xsprite[spriteRecord.extra].busy;
            }
        }
        return signature;
    }

    bool interactionTargetMatches(const InteractionMemory &memory, int hit, int target) const
    {
        if (memory.kind == kInteractionWall)
        {
            if (memory.target.wallPush && hit == 0 && target == memory.target.wall)
                return true;
            if (memory.target.sectorPush && hit == 6 && target == memory.target.to)
                return true;
            if (memory.target.sectorPushCurrent && hit == 6 && target == memory.target.from)
                return true;
        }
        if (memory.kind == kInteractionSector)
            return hit == 6 && target == memory.fromSector;
        if (memory.kind == kInteractionSprite)
            return hit == 3 && target == memory.id;
        return false;
    }

    int interactionLookTarget(const InteractionMemory &memory, bool crouched) const
    {
        if (!gMe || !gMe->pSprite)
            return 0;
        if (memory.target.wall < 0)
            return fix16_to_int(gMe->q16look);
        const POSTURE &posture = gMe->pPosture[gMe->lifeMode]
            [crouched ? kPostureCrouch : kPostureStand];
        const int eyeZ = gMe->pSprite->z - posture.eyeAboveZ;
        const int margin = 128;
        const int lower = memory.target.interactionTopZ + margin;
        const int upper = memory.target.interactionBottomZ - margin;
        const int targetZ = lower <= upper ? std::max(lower, std::min(upper, eyeZ))
                                           : memory.target.z;
        const int horizontal = int(std::sqrt(double(distance2(
            observation.x, observation.y, memory.target.x, memory.target.y))));
        return lookAngleForTarget(eyeZ, targetZ, horizontal);
    }

    int interactionFacingAngle(const InteractionMemory &memory) const
    {
        if (memory.kind != kInteractionWall || memory.target.wall < 0)
            return memory.kind == kInteractionSector
                ? observation.angle
                : getangle(memory.x - observation.x, memory.y - observation.y);

        // Sector-push ActionScan targets are resolved against the sector's
        // usable opening and have historically worked from the approach
        // point.  Only a genuine wall-push target needs the wall-face normal;
        // treating every sector-push marker as a wall face makes the bot
        // strafe sideways and can lose the engine's sector ActionScan target.
        if (!memory.target.wallPush)
            return getangle(memory.x - observation.x, memory.y - observation.y);

        // ActionScan is directional.  For a wall-backed sector push, the
        // sector centroid is not a useful facing target: the player must face
        // the movable wall surface itself.  Choose the wall normal pointing
        // toward the observed approach side.  This also covers one-sided
        // push walls and the canonical sector-push records that represent the
        // same mechanism through several adjacent wall markers.
        const int tangentX = memory.target.x2 - memory.target.x1;
        const int tangentY = memory.target.y2 - memory.target.y1;
        const int normal1X = -tangentY;
        const int normal1Y = tangentX;
        const int normal2X = -normal1X;
        const int normal2Y = -normal1Y;
        const int toX = memory.target.x - observation.x;
        const int toY = memory.target.y - observation.y;
        const int64_t dot1 = int64_t(normal1X) * toX + int64_t(normal1Y) * toY;
        const int64_t dot2 = int64_t(normal2X) * toX + int64_t(normal2Y) * toY;
        return getangle(dot2 > dot1 ? normal2X : normal1X,
                        dot2 > dot1 ? normal2Y : normal1Y);
    }

    bool interactionNeedsCrouch(const InteractionMemory &memory) const
    {
        if (!gMe || !gMe->pSprite || memory.target.wall < 0)
            return false;
        // A large upward sector mechanism is commonly operated from a low
        // approach pose even when the adjacent portal is technically
        // traversable.  This is a capability/geometry trial, not a map ID
        // special case; ActionScanPreview remains the final authority.
        if (memory.target.sectorPush && memory.target.floorDelta < -kMaxWalkableStep)
            return true;
        const POSTURE &stand = gMe->pPosture[gMe->lifeMode][kPostureStand];
        const POSTURE &crouch = gMe->pPosture[gMe->lifeMode][kPostureCrouch];
        const int horizontal = int(std::sqrt(double(distance2(
            observation.x, observation.y, memory.target.x, memory.target.y))));
        const int margin = 128;
        const int lower = memory.target.interactionTopZ + margin;
        const int upper = memory.target.interactionBottomZ - margin;
        const int standingEye = gMe->pSprite->z - stand.eyeAboveZ;
        const int crouchedEye = gMe->pSprite->z - crouch.eyeAboveZ;
        const int standingTarget = lower <= upper ? std::max(lower, std::min(upper, standingEye))
                                                  : memory.target.z;
        const int crouchedTarget = lower <= upper ? std::max(lower, std::min(upper, crouchedEye))
                                                  : memory.target.z;
        const int standingRaw = int(std::lround(std::atan2(
            double(standingEye - standingTarget),
            double(std::max(1, horizontal))) * 1024.0 / 3.14159265358979323846));
        const int crouchedRaw = int(std::lround(std::atan2(
            double(crouchedEye - crouchedTarget),
            double(std::max(1, horizontal))) * 1024.0 / 3.14159265358979323846));
        return standingRaw < kLookDownLimit && crouchedRaw >= kLookDownLimit;
    }

    void rememberInteraction(const InteractionCandidate &candidate)
    {
        const int key = interactionKey(candidate);
        InteractionMemory &memory = interactions[key];
        const bool first = !memory.observed;
        const int oldState = memory.observed ? interactionStateSignature(memory) : 0;
        const bool canonicalSectorPush = candidate.kind == kInteractionWall
            && (candidate.target.sectorPush || candidate.target.sectorPushCurrent);
        memory.kind = candidate.kind;
        memory.id = (candidate.kind == kInteractionWall && candidate.target.sectorPush)
            ? candidate.target.to
            : (candidate.kind == kInteractionWall && candidate.target.sectorPushCurrent)
            ? candidate.target.from : candidate.id;
        memory.fromSector = candidate.fromSector;
        memory.targetSector = candidate.targetSector;
        if (first || !canonicalSectorPush)
        {
            memory.x = candidate.x;
            memory.y = candidate.y;
            memory.z = candidate.z;
        }
        // Several wall markers can dispatch the same sector ActionScan
        // target.  Keep the first observed approach point: unlike a genuine
        // wall-push, the sector target is resolved through the usable portal
        // and the nearest marker is not necessarily the pose from which
        // ActionScan can see that sector.
        memory.key = candidate.key;
        memory.locked = candidate.locked;
        memory.reversible = candidate.reversible;
        memory.observed = true;
        memory.target = candidate.target;
        const int newState = interactionStateSignature(memory);
        if (memory.unavailableFromSector >= 0 && memory.unavailableState != newState)
        {
            memory.unavailableFromSector = -1;
            memory.unavailableAttempts = 0;
            memory.unavailablePose = 0;
        }
        if (memory.state == 1 && newState != memory.beforeState)
        {
            memory.state = 2;
            memory.activated = true;
            memory.observedLocalEffect = true;
            memory.observedKnownWorldDelta = true;
            recoveryMode = false;
            memory.afterState = newState;
            char detail[128];
            snprintf(detail, sizeof(detail), "kind=%d id=%d before=%d after=%d", int(memory.kind), memory.id,
                     memory.beforeState, newState);
            event("interaction_world_delta", detail);
            lastSemanticProgressTick = observation.tick;
        }
        updateFollowThrough(memory);
        if (first)
        {
            recoveryMode = false;
            pendingWork.push_back({ 1, key, candidate.fromSector, candidate.targetSector });
            ++knowledgeRevision;
            char detail[128];
            snprintf(detail, sizeof(detail), "kind=%d id=%d sector=%d key=%d", int(memory.kind), memory.id,
                     memory.fromSector, memory.key);
            event("discovered_interaction", detail);
            lastSemanticProgressTick = observation.tick;
        }
        else if (oldState != newState && memory.state != 1)
        {
            // State refresh is local knowledge maintenance.  It does not
            // expose objects in sectors the bot has never observed.
            event("interaction_state_refreshed", "source=known_world_revisit");
        }
    }

    void updateKnowledge()
    {
        observation = observeWorld();
        if (observation.sector < 0)
            return;
        if (lastObservedHealth >= 0 && observation.health < lastObservedHealth)
            lastDamageTick = observation.tick;
        lastObservedHealth = observation.health;
        if (observation.sector != lastGeometryTelemetrySector)
        {
            char detail[160];
            snprintf(detail, sizeof(detail),
                     "sector=%d clearance=%d player_ceiling=%d player_floor=%d portals=%d jumpable=%d max_rise=%d max_drop=%d xsector=%d state=%d busy=%d zvel=%d",
                     observation.sector, observation.localClearance,
                     observation.playerCeilingDistance, observation.playerFloorDistance,
                     observation.localPortalCount,
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
        if (sectorChanged && previousSector >= 0 && movementTargetActive
            && movementTargetFrom == previousSector
            && movementTargetSector == observation.sector && movementTargetId >= 0)
        {
            const int edgeId = movementTargetId * 65536 + observation.sector;
            visitedEdges.insert(edgeId);
            failedEdges.erase(edgeId);
            char detail[128];
            snprintf(detail, sizeof(detail), "from=%d to=%d wall=%d", previousSector,
                     observation.sector, movementTargetId);
            event("portal_traversed", detail);

            const bool isActiveFrontier = currentObjective.active
                && currentObjective.type == kObjectiveFrontier
                && currentObjective.wall == movementTargetId
                && currentObjective.sector == previousSector
                && currentObjective.targetSector == observation.sector;
            if (isActiveFrontier)
            {
                retirePendingFrontier(movementTargetId, previousSector, observation.sector);
                char frontierDetail[192];
                snprintf(frontierDetail, sizeof(frontierDetail),
                         "wall=%d source=%d target=%d current=%d state=CONSUMED",
                         movementTargetId, previousSector, observation.sector,
                         observation.sector);
                event("frontier_crossed", frontierDetail);
            }
            else if (currentObjective.active && currentObjective.type == kObjectiveFrontier)
            {
                // This was transportation toward the committed frontier, not
                // a replacement exploration objective.  Keep the original
                // work identity intact and make the route step explicit.
                char routeDetail[224];
                snprintf(routeDetail, sizeof(routeDetail),
                         "frontier_wall=%d source=%d target=%d route_wall=%d from=%d to=%d",
                         currentObjective.wall, currentObjective.sector,
                         currentObjective.targetSector, movementTargetId,
                         previousSector, observation.sector);
                event("frontier_route_step", routeDetail);
            }
            for (auto &entry : interactions)
            {
                InteractionMemory &memory = entry.second;
                if (memory.kind == kInteractionWall && memory.target.wall == movementTargetId
                    && memory.target.from == previousSector
                    && memory.target.to == observation.sector)
                {
                    memory.traversed = true;
                    memory.afterState = interactionStateSignature(memory);
                    event("interaction_route_traversed", "source=portal_transition");
                }
            }
            lastSemanticProgressTick = observation.tick;
        }
        if (sectorChanged && previousSector >= 0 && followThroughKey >= 0
            && previousSector == followThroughFrom
            && observation.sector == followThroughPortal.to)
        {
            char detail[256];
            snprintf(detail, sizeof(detail),
                     "before_sector=%d after_sector=%d interaction_key=%d portal_wall=%d portal_from=%d portal_to=%d engine_accepted=1 game_time=%d",
                     previousSector, observation.sector, followThroughKey,
                     followThroughPortal.wall, followThroughPortal.from,
                     followThroughPortal.to, (gFrame * kTicsPerFrame) / kTicsPerSec);
            event("interaction_follow_through_crossed", detail);
            if (!cryptExitProven)
            {
                event("CRYPT_EXIT_CROSSED", detail);
                cryptExitProven = true;
            }
            followThroughKey = -1;
            followThroughStartTick = -1;
            followThroughFrom = -1;
            followThroughPortal = Portal{};
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
            lastTransitionTick = observation.tick;
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
            const int edgeId = portal.wall * 65536 + portal.to;
            auto localFailure = localFailureSignatures.find(edgeId);
            if (localFailure != localFailureSignatures.end()
                && localFailure->second != portalGeometrySignature(portal))
                localFailureSignatures.erase(localFailure);
            const bool knownEdge = std::any_of(knownGraph[portal.from].begin(),
                                               knownGraph[portal.from].end(),
                                               [&portal](const Portal &known)
            {
                return known.wall == portal.wall && known.to == portal.to;
            });
            const bool newEdge = !portal.localGeometry && seenEdges.insert(edgeId).second;
            if (newEdge)
            {
                recoveryMode = false;
                PendingWork pending;
                pending.kind = 2;
                pending.id = portal.wall;
                pending.sector = portal.from;
                pending.edgeTarget = portal.to;
                pending.state = PendingWork::kDiscovered;
                pending.stateSignature = portalGeometrySignature(portal);
                pendingWork.push_back(pending);
                ++knowledgeRevision;
                lastSemanticProgressTick = observation.tick;
                char detail[96];
                snprintf(detail, sizeof(detail), "wall=%d from=%d to=%d", portal.wall,
                         portal.from, portal.to);
                event("discovered_frontier", detail);
            }
            else if (PendingWork *pending = findPendingFrontier(portal.wall, portal.from, portal.to))
            {
                const int currentSignature = portalGeometrySignature(portal);
                if (pending->stateSignature != 0 && pending->stateSignature != currentSignature
                    && (pending->state == PendingWork::kTemporarilyUnavailable
                        || pending->state == PendingWork::kStale))
                {
                    pending->state = PendingWork::kDiscovered;
                    pending->resolved = false;
                    event("frontier_revalidated", "reason=geometry_signature_changed");
                }
                pending->stateSignature = currentSignature;
            }
            auto &edges = knownGraph[portal.from];
            auto edge = std::find_if(edges.begin(), edges.end(), [&portal](const Portal &known)
            {
                return known.wall == portal.wall && known.to == portal.to;
            });
            if (portal.traversable || portal.jumpable)
            {
                // A local-geometry observation may refresh an edge already
                // known, but must not create a route from an unseen corner.
                if (edge == edges.end() && !portal.localGeometry)
                    edges.push_back(portal);
            else if (edge != edges.end())
                *edge = portal;
            }
            else if (edge != edges.end())
                edges.erase(edge);
            if (!portal.localGeometry && (portal.traversable || portal.jumpable))
            {
                // A visible two-sided Build opening is a known route in both
                // directions.  Keeping only the observed wall direction made
                // DFS backtracking impossible after a one-way sector entry.
                Portal reverse = portal;
                reverse.from = portal.to;
                reverse.to = portal.from;
                reverse.floorZ = portal.fromFloorZ;
                reverse.ceilingZ = portal.fromCeilingZ;
                reverse.fromFloorZ = portal.toFloorZ;
                reverse.fromCeilingZ = portal.toCeilingZ;
                reverse.toFloorZ = portal.fromFloorZ;
                reverse.toCeilingZ = portal.fromCeilingZ;
                reverse.floorDelta = -portal.floorDelta;
                int reverseWallId = -1;
                if (inRange(reverse.from, 0, numsectors))
                {
                    const sectortype &reverseSector = sector[reverse.from];
                    for (int i = 0; i < reverseSector.wallnum; ++i)
                    {
                        const int candidateWallId = reverseSector.wallptr + i;
                        if (!inRange(candidateWallId, 0, numwalls))
                            continue;
                        const walltype &candidateWall = wall[candidateWallId];
                        const walltype &candidateEnd = wall[candidateWall.point2];
                        const bool sameSegment =
                            (candidateWall.x == portal.x1 && candidateWall.y == portal.y1
                             && candidateEnd.x == portal.x2 && candidateEnd.y == portal.y2)
                            || (candidateWall.x == portal.x2 && candidateWall.y == portal.y2
                                && candidateEnd.x == portal.x1 && candidateEnd.y == portal.y1);
                        if (candidateWall.nextsector == reverse.to && sameSegment)
                        {
                            reverseWallId = candidateWallId;
                            break;
                        }
                    }
                }
                if (reverseWallId < 0)
                    reverseWallId = -1;
                if (reverseWallId >= 0)
                {
                reverse.wall = reverseWallId;
                // Reclassify the opposite direction from the real opposite
                // wall/sector records.  A drop in one direction can be a
                // jump or a blocked rise in the other; copying the original
                // mode would corrupt NavTopology and walkArea.
                const walltype &reverseWall = wall[reverseWallId];
                const bool reverseEnoughWidth = reverse.openingWidth >= kPlayerPassageWidth;
                const bool reverseStanding = reverse.clearance >= playerBodyClearance();
                const bool reverseCrouching = reverse.clearance >= playerCrouchClearance();
                const int reverseRise = playerJumpRiseLimit();
                const bool reverseWalk = reverseEnoughWidth && reverseStanding
                    && std::abs(reverse.floorDelta) <= kMaxWalkableStep;
                const bool reverseJump = reverseEnoughWidth && reverseStanding
                    && reverse.floorDelta < 0 && -reverse.floorDelta <= reverseRise;
                const bool reverseDrop = reverseEnoughWidth && reverseStanding
                    && reverse.floorDelta > 0 && reverse.floorDelta <= reverseRise;
                reverse.walkable = !(reverseWall.cstat & 1) && reverseWalk;
                reverse.crouchable = !(reverseWall.cstat & 1) && !reverseStanding
                    && reverseCrouching && std::abs(reverse.floorDelta) <= kMaxWalkableStep;
                reverse.jumpable = !(reverseWall.cstat & 1) && !reverseWalk && reverseJump;
                reverse.dropSafe = !(reverseWall.cstat & 1) && reverseDrop;
                reverse.traversable = reverse.walkable || reverse.crouchable
                    || reverse.jumpable || reverse.dropSafe;
                reverse.capability = reverse.walkable ? kTraversalWalkable
                    : reverse.crouchable ? kTraversalCrouchable
                    : reverse.jumpable ? kTraversalJumpable
                    : reverse.dropSafe ? kTraversalDropSafe
                    : kTraversalCurrentlyUnavailable;
                reverse.wallPush = false;
                reverse.key = 0;
                reverse.locked = false;
                reverse.wallState = 0;
                reverse.wallBusy = 0;
                if (reverseWall.extra > 0 && reverseWall.extra < kMaxXWalls)
                {
                    const XWALL &extra = xwall[reverseWall.extra];
                    reverse.wallPush = extra.triggerPush != 0;
                    reverse.key = extra.key;
                    reverse.locked = extra.locked != 0;
                    reverse.wallState = extra.state;
                    reverse.wallBusy = extra.busy;
                }
                reverse.sectorPush = false;
                reverse.sectorPushCurrent = false;
                reverse.sectorState = 0;
                reverse.sectorBusy = 0;
                if (inRange(reverse.to, 0, numsectors)
                    && sector[reverse.to].extra > 0 && sector[reverse.to].extra < kMaxXSectors)
                {
                    const XSECTOR &extra = xsector[sector[reverse.to].extra];
                    reverse.sectorPush = extra.Wallpush != 0;
                    reverse.sectorState = extra.state;
                    reverse.sectorBusy = extra.busy;
                }
                if (inRange(reverse.from, 0, numsectors)
                    && sector[reverse.from].extra > 0 && sector[reverse.from].extra < kMaxXSectors)
                    reverse.sectorPushCurrent = xsector[sector[reverse.from].extra].Push != 0;
                reverse.interactionAffordance = reverse.wallPush || reverse.sectorPush
                    || reverse.sectorPushCurrent;
                reverse.currentlyAvailable = reverse.traversable || reverse.interactionAffordance;
                auto &reverseEdges = knownGraph[reverse.from];
                auto reverseEdge = std::find_if(reverseEdges.begin(), reverseEdges.end(),
                                                [&reverse](const Portal &known)
                {
                    return known.wall == reverse.wall && known.to == reverse.to;
                });
                if (reverseEdge == reverseEdges.end())
                    reverseEdges.push_back(reverse);
                else
                    *reverseEdge = reverse;
                }
            }
            else if (!portal.localGeometry && inRange(portal.to, 0, numsectors))
            {
                // The cached reverse route describes current topology, not
                // historical reachability.  Remove it when this re-observed
                // opening is no longer usable.
                auto &reverseEdges = knownGraph[portal.to];
                for (auto reverseEdge = reverseEdges.begin(); reverseEdge != reverseEdges.end(); )
                {
                    if (reverseEdge->to == portal.from
                        && reverseEdge->x1 == portal.x1 && reverseEdge->y1 == portal.y1
                        && reverseEdge->x2 == portal.x2 && reverseEdge->y2 == portal.y2)
                        reverseEdge = reverseEdges.erase(reverseEdge);
                    else
                        ++reverseEdge;
                }
            }
            if (portal.localGeometry && !knownEdge)
                continue;
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
                        || previousPortal.x1 != portal.x1 || previousPortal.y1 != portal.y1
                        || previousPortal.x2 != portal.x2 || previousPortal.y2 != portal.y2
                        || previousPortal.traversable != portal.traversable);
                door.portal = portal;
                if (response && door.opened && !portal.traversable
                    && portal.interactionAffordance)
                {
                    door.opened = false;
                    door.interaction = DoorMemory::kIdle;
                    door.interactionDeadlineTick = -1;
                    event("door_closed", "response=authoritative_geometry_changed");
                }
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

        // Refresh interaction memory from current engine state whenever the
        // area is observed again.  This includes one-sided walls and current
        // sector pushes, which never enter the portal graph.
        for (const InteractionCandidate &candidate : observation.interactions)
            rememberInteraction(candidate);

        for (const VisibleObject &object : observation.objects)
        {
            ObjectMemory &remembered = objectMemory[object.sprite];
            const bool firstObjectObservation = !remembered.observed;
            remembered.object = object;
            remembered.observed = true;
            remembered.collected = false;
            remembered.lastSeenTick = observation.tick;
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
            if (firstObjectObservation)
                event("object_remembered", object.kind == kObjectKey ? "kind=key" : "kind=pickup");
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
                if (door.engineAccepted)
                {
                    door.interaction = DoorMemory::kOpening;
                    event("interaction_settled", "door_engine_accepted=1 observed_local_effect=0");
                }
                else
                {
                    door.interaction = DoorMemory::kFailed;
                    char detail[144];
                    snprintf(detail, sizeof(detail), "door=%d waited_ticks=%d reason=INTERACTION_VALID_BUT_NO_RESPONSE", door.id,
                             observation.tick - door.interactionStartedTick);
                    event("interaction_failed", detail);
                }
                door.interactionDeadlineTick = -1;
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

    int portalGeometrySignature(const Portal &portal) const
    {
        // A failed approach is valid only for the geometry that was actually
        // tested.  Discovery of an unrelated pickup or sector must not make
        // every old failure disappear, while a changed wall/sector must.
        int signature = portal.wall * 31 + portal.from * 131 + portal.to * 977;
        signature = signature * 31 + portal.x1;
        signature = signature * 31 + portal.y1;
        signature = signature * 31 + portal.x2;
        signature = signature * 31 + portal.y2;
        signature = signature * 31 + portal.wallState;
        signature = signature * 31 + portal.wallBusy;
        signature = signature * 31 + portal.sectorState;
        signature = signature * 31 + portal.sectorBusy;
        signature = signature * 31 + portal.floorZ;
        signature = signature * 31 + portal.ceilingZ;
        signature = signature * 31 + portal.floorDelta;
        signature = signature * 31 + portal.clearance;
        signature = signature * 31 + (portal.traversable ? 1 : 0);
        signature = signature * 31 + (portal.crouchable ? 1 : 0);
        return signature;
    }

    const Portal *findPortalForEdge(int edgeId) const
    {
        for (const Portal &portal : observation.portals)
            if (portal.wall * 65536 + portal.to == edgeId)
                return &portal;
        for (const auto &entry : knownGraph)
            for (const Portal &portal : entry.second)
                if (portal.wall * 65536 + portal.to == edgeId)
                    return &portal;
        return nullptr;
    }

    bool edgeFailed(int edgeId) const
    {
        auto failed = failedEdges.find(edgeId);
        if (failed == failedEdges.end())
            return false;
        const Portal *current = findPortalForEdge(edgeId);
        // If the edge is currently not observable, do not turn an old local
        // trajectory failure into a permanent topological fact.
        return current && failed->second.geometrySignature == portalGeometrySignature(*current);
    }

    bool localEdgeFailed(int edgeId) const
    {
        if (localFailureSignatures.count(edgeId) != 0)
            return true;
        auto failure = failedEdges.find(edgeId);
        return failure != failedEdges.end() && failure->second.attempts > 0;
    }

    void recordEdgeFailure(const Portal &portal, const char *eventName, const char *reason)
    {
        const int edgeId = portal.wall * 65536 + portal.to;
        EdgeFailure &failure = failedEdges[edgeId];
        const int signature = portalGeometrySignature(portal);
        if (failure.geometrySignature != signature)
        {
            failure.geometrySignature = signature;
            failure.attempts = 0;
        }
        ++failure.attempts;
        if (strcmp(eventName, "local_portal_failed") == 0)
            localFailureSignatures[edgeId] = signature;
        char detail[160];
        snprintf(detail, sizeof(detail), "wall=%d from=%d to=%d reason=%s attempts=%d signature=%d",
                 portal.wall, portal.from, portal.to, reason, failure.attempts, signature);
        event(eventName, detail);
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

        // Pending work is a persistent DFS-like stack.  Prefer the most
        // recently discovered unresolved edge in this sector before falling
        // back to geometric nearest-neighbour scoring.
        for (auto pending = pendingWork.rbegin(); pending != pendingWork.rend(); ++pending)
        {
            if (!pendingWorkLive(*pending) || pending->kind != 2
                || pending->sector != observation.sector)
                continue;
            for (const Portal &portal : observation.portals)
            {
                const int edgeId = portal.wall * 65536 + portal.to;
                if (portal.wall == pending->id && portal.to == pending->edgeTarget
                    && (portal.traversable || portal.jumpable)
                    && !visitedEdges.count(edgeId) && !edgeFailed(edgeId)
                    && (!portal.key || hasKey(portal.key)))
                    return &portal;
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
            if (!unvisitedAvailable || alreadyExplored)
                continue;
            const int score = distance2(observation.x, observation.y, portal.x, portal.y);
            if (score < bestDistance)
            {
                best = &portal;
                bestDistance = score;
            }
        }
        if (best)
            return best;

        return nullptr;
    }

    PendingWork *selectRemoteFrontier()
    {
        // This returns the work item, never the first transport edge on the
        // route to it.  The committed Objective retains the frontier's
        // source/target/wall identity until that exact directional crossing
        // is observed.
        for (auto pending = pendingWork.rbegin(); pending != pendingWork.rend(); ++pending)
        {
            if (!pendingWorkLive(*pending) || pending->kind != 2
                || pending->sector == observation.sector)
                continue;
            Portal route;
            if (!findKnownRoute(pending->sector, route))
                continue;
            const int edgeId = pending->id * 65536 + pending->edgeTarget;
            if (edgeFailed(edgeId))
                continue;
            return &*pending;
        }
        return nullptr;
    }

    const Portal *selectLocalPortal()
    {
        if (currentGoal == "EXPLORE_LOCAL_PORTAL" && currentGoalTarget >= 0)
        {
            for (const Portal &portal : observation.portals)
            {
                const int edgeId = portal.wall * 65536 + portal.to;
                if (portal.wall == currentGoalTarget && portal.traversable
                    && !visitedEdges.count(edgeId) && !edgeFailed(edgeId)
                    && !localEdgeFailed(edgeId))
                {
                    localJumpPortal = portal;
                    if (jumpFallbackEdges.count(edgeId))
                        localJumpPortal.capability = kTraversalJumpable;
                    return &localJumpPortal;
                }
            }
        }
        for (auto pending = pendingWork.rbegin(); pending != pendingWork.rend(); ++pending)
        {
            if (!pendingWorkLive(*pending) || pending->kind != 2
                || pending->sector != observation.sector)
                continue;
            for (const Portal &portal : observation.portals)
            {
                const int edgeId = portal.wall * 65536 + portal.to;
                if (portal.wall == pending->id && portal.to == pending->edgeTarget
                    && portal.localGeometry && portal.traversable
                    && !visitedEdges.count(edgeId) && !edgeFailed(edgeId)
                    && !localEdgeFailed(edgeId))
                {
                    localJumpPortal = portal;
                    if (jumpFallbackEdges.count(edgeId))
                        localJumpPortal.capability = kTraversalJumpable;
                    return &localJumpPortal;
                }
            }
        }
        bool walkableAvailable = false;
        for (const Portal &portal : observation.portals)
        {
            if (!portal.localGeometry || portal.capability != kTraversalWalkable)
                continue;
            const int edgeId = portal.wall * 65536 + portal.to;
            if (edgeFailed(edgeId) || localEdgeFailed(edgeId) || visitedEdges.count(edgeId)
                || openedRoutes.count(portal.from * 65536 + portal.to)
                || portal.to == lastTransitionFrom)
                continue;
            walkableAvailable = true;
            break;
        }

        const Portal *best = nullptr;
        int bestDistance = INT32_MAX;
        for (const Portal &portal : observation.portals)
        {
            if (!portal.localGeometry || !portal.traversable)
                continue;
            if (walkableAvailable && portal.capability != kTraversalWalkable)
                continue;
            const int edgeId = portal.wall * 65536 + portal.to;
            if (edgeFailed(edgeId) || localEdgeFailed(edgeId) || visitedEdges.count(edgeId)
                || openedRoutes.count(portal.from * 65536 + portal.to)
                || portal.to == lastTransitionFrom)
                continue;
            const int score = distance2(observation.x, observation.y, portal.x, portal.y);
            if (score < bestDistance)
            {
                bestDistance = score;
                localJumpPortal = portal;
                if (jumpFallbackEdges.count(edgeId))
                    localJumpPortal.capability = kTraversalJumpable;
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
        if (best)
        {
            selectedObject = *best;
            return &selectedObject;
        }
        for (const auto &entry : objectMemory)
        {
            const ObjectMemory &memory = entry.second;
            if (!memory.observed || memory.collected || memory.object.kind != kind)
                continue;
            if (memory.object.sprite < 0 || memory.object.sprite >= kMaxSprites
                || sprite[memory.object.sprite].sectnum < 0)
                continue;
            const int distance = distance2(observation.x, observation.y,
                                           memory.object.x, memory.object.y);
            if (distance < bestDistance)
            {
                selectedObject = memory.object;
                bestDistance = distance;
                best = &selectedObject;
            }
        }
        return best;
    }

    const VisibleObject *selectLocalAreaObject(ObjectKind kind)
    {
        ensureNavTopology();
        const int playerCell = nearestNavCell(observation.sector, observation.x, observation.y);
        if (playerCell < 0 || navCells[playerCell].walkArea < 0)
            return nullptr;
        const int playerArea = navCells[playerCell].walkArea;
        const VisibleObject *best = nullptr;
        int bestDistance = INT32_MAX;
        auto consider = [&](const VisibleObject &object)
        {
            if (object.kind != kind || object.sector != observation.sector)
                return;
            const int objectCell = nearestNavCell(object.sector, object.x, object.y);
            if (objectCell < 0 || navCells[objectCell].walkArea != playerArea)
                return;
            const int distance = distance2(observation.x, observation.y, object.x, object.y);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                selectedObject = object;
                best = &selectedObject;
            }
        };
        for (const VisibleObject &object : observation.objects)
            consider(object);
        for (const auto &entry : objectMemory)
        {
            const ObjectMemory &memory = entry.second;
            if (!memory.observed || memory.collected || !objectStillPresent(memory))
                continue;
            consider(memory.object);
        }
        return best;
    }

    bool objectStillPresent(const ObjectMemory &memory) const
    {
        const int id = memory.object.sprite;
        if (id < 0 || id >= kMaxSprites)
            return false;
        const spritetype &candidate = sprite[id];
        if (candidate.sectnum < 0)
            return false;
        if (memory.object.kind == kObjectEnemy)
            return candidate.statnum == kStatDude && validXSprite(candidate.extra)
                && xsprite[candidate.extra].health > 0;
        if (memory.object.kind == kObjectKey || memory.object.kind == kObjectPickup)
            return candidate.statnum == kStatItem && itemCategory(candidate.type) != nullptr;
        return true;
    }

    InteractionMemory *selectNewInteraction(bool includeRecovery = false)
    {
        InteractionMemory *best = nullptr;
        int bestDistance = INT32_MAX;
        for (auto &entry : interactions)
        {
            InteractionMemory &memory = entry.second;
            if (!memory.observed)
                continue;
            const bool reactivation = interactionNeedsReactivation(memory);
            if (memory.attempted && !reactivation
                && !(includeRecovery && recoveryMode && memory.activated
                     && memory.reversible && !memory.recoveryAttempted))
                continue;
            if (memory.unavailableFromSector == observation.sector)
                continue;
            if (memory.fromSector != observation.sector && !includeRecovery)
                continue;
            // A currently traversable portal is already a valid exploration
            // frontier.  Its optional Wallpush metadata must not turn the
            // route into a separate interaction goal that pulls the bot back
            // to the source side after it has crossed (notably for safe
            // drop/elevator portals).  Closed portals and one-sided push
            // walls remain first-class interaction work.
            if (!reactivation && memory.target.wall >= 0 && memory.target.traversable
                && !memory.target.sectorPush)
                continue;
            int distance = INT32_MAX;
            if (memory.fromSector == observation.sector)
                distance = distance2(observation.x, observation.y, memory.x, memory.y);
            else
            {
                Portal route;
                if (!findKnownRoute(memory.fromSector, route))
                    continue;
                distance = distance2(observation.x, observation.y, route.x, route.y) + 1000000;
            }
            // When several visible mechanisms share a compact room, prefer
            // the one the real ActionScan already resolves from this pose.
            // This keeps a moving slide-door sequence ordered by the engine's
            // current target instead of by map-record iteration order.
            int hit = -1;
            int target = -1;
            int extra = -1;
            hit = memory.fromSector == observation.sector
                ? ActionScanPreview(gMe, &target, &extra) : -1;
            const bool actionValid = memory.fromSector == observation.sector
                && interactionTargetMatches(memory, hit, target);
            if (!actionValid)
                distance += 100000000;
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = &memory;
            }
        }
        return best;
    }

    bool interactionNeedsReactivation(const InteractionMemory &memory) const
    {
        if (!memory.attempted || !memory.activated || !memory.reversible
            || !memory.traversed || memory.target.wall < 0
            || memory.target.to < 0 || memory.target.from != observation.sector)
            return false;
        if (interactionStateSignature(memory) == memory.afterState)
            return false;
        return memory.target.interactionAffordance && !memory.target.traversable;
    }

    GINPUT steerInteraction(InteractionMemory &memory)
    {
        if (memory.fromSector != observation.sector)
        {
            Portal route;
            if (findKnownRoute(memory.fromSector, route))
            {
                setGoal("NAVIGATE_TO_NEW_INTERACTION", memory.id);
                return steerPortal(route);
            }
            memory.unavailableFromSector = observation.sector;
            return GINPUT{};
        }

        int hit = -1;
        int target = -1;
        int extra = -1;
        const bool interactionCrouch = interactionNeedsCrouch(memory);
        memory.target.interactionCrouch = interactionCrouch;
        const int expectedDistance = memory.kind == kInteractionSector
            ? 0 : distance2(observation.x, observation.y, memory.x, memory.y);
        const int expectedAngle = interactionFacingAngle(memory);
        const int desiredLook = interactionLookTarget(memory, interactionCrouch);
        const int currentLook = fix16_to_int(gMe->q16look);
        const int lookDelta = desiredLook - currentLook;
        hit = ActionScanPreview(gMe, &target, &extra);
        const bool valid = interactionTargetMatches(memory, hit, target);
        const bool missingKey = memory.key && !hasKey(memory.key);
        const char *status = missingKey ? "LOCKED_KEY_REQUIRED"
            : valid ? "VALID_ACTION_TARGET"
            : expectedDistance > kActionScanRange * kActionScanRange ? "NOT_IN_USE_RANGE"
            : std::abs(angleDelta(expectedAngle, observation.angle)) >= 96 ? "NOT_FACING_ACTION_TARGET"
            : std::abs(lookDelta) >= 64 ? "NOT_AIMED_AT_ACTION_TARGET"
            : "NO_ACTION_TARGET";
        char detail[360];
        snprintf(detail, sizeof(detail),
                 "kind=%d id=%d status=%s hit=%d target=%d extra=%d distance=%d angle_delta=%d look=%d desired_look=%d look_delta=%d target_z=%d target_top_z=%d target_bottom_z=%d crouch=%d attempted=%d activations=%d",
                 int(memory.kind), memory.id, status, hit, target, extra,
                 int(std::sqrt(double(expectedDistance))), angleDelta(expectedAngle, observation.angle),
                 currentLook, desiredLook, lookDelta, memory.target.z,
                 memory.target.interactionTopZ, memory.target.interactionBottomZ,
                 interactionCrouch ? 1 : 0, memory.attempted ? 1 : 0,
                 memory.activationCount);
        event("interaction_probe", detail);

        bool issueUse = false;
        const bool recoveryActivation = recoveryMode && memory.attempted
            && memory.activated && memory.reversible && !memory.recoveryAttempted;
        if (recoveryActivation)
            memory.recoveryAttempted = true;
        const bool reactivation = interactionNeedsReactivation(memory);
        if (valid && !missingKey && (!memory.attempted || recoveryActivation || reactivation))
        {
            if (reactivation)
                event("interaction_reactivation", "reason=previously_traversed_route_closed");
            memory.attempted = true;
            memory.state = 1;
            memory.activated = false;
            memory.lastActivationTick = observation.tick;
            memory.beforeState = interactionStateSignature(memory);
            memory.beforePortals = observation.portals;
            memory.engineAccepted = false;
            memory.observedLocalEffect = false;
            memory.observedKnownWorldDelta = false;
            emitInteractionPortalState(memory, "before_use_pulse");
            emitInteractionGeometryState(memory, "before_use_pulse");
            ++memory.activationCount;
            pendingInteractionKey = interactionMemoryKey(memory);
            issueUse = true;
            snprintf(detail, sizeof(detail), "kind=%d id=%d hit=%d target=%d attempt=%d",
                     int(memory.kind), memory.id, hit, target, memory.activationCount);
            event("interaction_started", detail);
            event("use_pulse", detail);
        }
        if (!issueUse && memory.attempted && memory.state == 1
            && observation.tick - memory.lastActivationTick > kInteractionTimeoutTicks)
        {
            if (memory.engineAccepted)
            {
                memory.state = 2;
                memory.activated = true;
                memory.afterState = interactionStateSignature(memory);
                event("interaction_settled", "engine_accepted=1 observed_local_effect=0 observed_known_world_delta=0");
            }
            else
            {
                memory.state = 3;
                event("interaction_failed", "reason=INTERACTION_VALID_BUT_NO_RESPONSE");
            }
        }
        if (!valid && !missingKey && memory.state != 1 && memory.state != 2
            && expectedDistance <= kActionScanRange * kActionScanRange
            && std::abs(angleDelta(expectedAngle, observation.angle)) < 96
            && std::abs(lookDelta) < 64)
        {
            const int pose = (observation.x >> 9) ^ ((observation.y >> 9) << 11)
                ^ (wrapAngle(observation.angle) >> 7);
            if (pose != memory.unavailablePose)
            {
                memory.unavailablePose = pose;
                ++memory.unavailableAttempts;
            }
            memory.unavailableState = interactionStateSignature(memory);
            if (memory.unavailableAttempts >= 3)
            {
                memory.unavailableFromSector = observation.sector;
                event("interaction_unavailable", "reason=CURRENTLY_UNAVAILABLE_FROM_THIS_SIDE attempts=3");
            }
            else
                event("interaction_alternative_pose", "reason=NO_ACTION_TARGET");
        }
        else if (!valid && !missingKey && expectedDistance <= kActionScanRange * kActionScanRange
                 && std::abs(angleDelta(expectedAngle, observation.angle)) < 96
                 && std::abs(lookDelta) >= 64)
        {
            event(interactionCrouch ? "interaction_crouch_pose" : "interaction_vertical_pose",
                  interactionCrouch ? "reason=target_below_standing_look_range"
                                     : "reason=target_height_requires_vertical_aim");
        }
        setGoal(memory.kind == kInteractionSprite ? "USE_NEW_SPRITE_INTERACTION"
                : memory.kind == kInteractionSector ? "USE_NEW_SECTOR_INTERACTION"
                : "USE_NEW_WALL_INTERACTION", memory.id);
        const TraversalCapability interactionCapability = interactionCrouch
            ? kTraversalCrouchable : kTraversalUnknown;
        // Preview was taken from this exact player pose.  A USE pulse must
        // not also turn or move the player before ProcessInput performs its
        // authoritative ActionScan, otherwise the preview and real scan can
        // resolve different walls/objects.
        GINPUT input = {};
        if (issueUse)
            input.keyFlags.action = 1;
        else
        {
            const int navigationSector = memory.targetSector >= 0
                ? memory.targetSector : observation.sector;
            const MovementProbe interactionPath = probeMovement(
                observation.x, observation.y, observation.z, observation.sector,
                memory.x, memory.y, navigationSector, std::max(256, kActionApproachRange / 2));
            // A body probe can quite correctly report the intended push wall
            // as the blocker.  That is the interaction pose, not a detour
            // opportunity: let the real player collision settle against the
            // surface while ActionScan remains authoritative.
            if (memory.target.wall >= 0 && interactionPath.wall == memory.target.wall)
            {
                resetNavigation();
                input = steerTo(memory.x, memory.y, false, false,
                                memory.id, navigationSector, interactionCapability);
            }
            else
                input = navigateTo(memory.x, memory.y, memory.z, navigationSector,
                                   memory.id, interactionCapability);
            if (navigationActionableBlocker >= 0
                && navigationActionableBlocker != memory.target.wall)
            {
                const int blockerKey = int(kInteractionWall) * 1000000
                    + navigationActionableBlocker + 1;
                auto blocker = interactions.find(blockerKey);
                if (blocker != interactions.end() && blocker->second.observed)
                {
                    char blockerDetail[96];
                    snprintf(blockerDetail, sizeof(blockerDetail),
                             "wall=%d original=%d", navigationActionableBlocker,
                             memory.target.wall);
                    event("navigation_interaction_opportunity", blockerDetail);
                }
                navigationActionableBlocker = -1;
                return GINPUT{};
            }
            if (navigationUsingDetour)
                return input;
            // steerTo aims movement at the approach point.  Keep the camera
            // aimed at the real wall normal so the authoritative ActionScan
            // sees the same target on the next tick.
            if (memory.target.wallPush)
            {
                input.q16turn = fix16_from_int(angleDelta(expectedAngle, observation.angle));
            }
            if (memory.target.wallPush
                && expectedDistance > kActionApproachRange * kActionApproachRange)
            {
                // Forward-only steering cannot approach an off-axis wall
                // point while the player is required to face its normal.
                // Decompose the approach vector into the engine's current
                // forward/strafe basis instead of turning away from the
                // ActionScan target.
                const int moveAngle = getangle(memory.x - observation.x,
                                               memory.y - observation.y);
                const int moveDelta = angleDelta(moveAngle, expectedAngle);
                input.forward = int16_t(mulscale16(Cos(moveDelta), 2047));
                input.strafe = int16_t(mulscale16(-Sin(moveDelta), 2047));
            }
            input.q16mlook = fix16_from_int(lookDelta / 8);
        }
        return input;
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
        if (object.sector != observation.sector
            || std::abs(object.z - observation.z) > kMaxWalkableStep)
            return false;
        const int radius = gMe && gMe->pSprite ? (gMe->pSprite->clipdist << 2) : 128;
        return probeMovement(observation.x, observation.y, observation.z, observation.sector,
                             object.x, object.y, object.sector, std::max(512, radius * 2)).reachable;
    }

    TraversalCapability currentClearanceCapability() const
    {
        if (!inRange(observation.sector, 0, numsectors))
            return kTraversalUnknown;
        const int floorZ = getflorzofslope(observation.sector, observation.x, observation.y);
        const int ceilingZ = getceilzofslope(observation.sector, observation.x, observation.y);
        const int clearance = floorZ - ceilingZ;
        if (clearance < playerBodyClearance() && clearance >= playerCrouchClearance())
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
                 "status=%s at_door=%d goal=%s expected_wall=%d expected_from=%d expected_sector=%d expected_wall_push=%d expected_sector_push=%d expected_current_sector_push=%d hit=%d target=%d extra=%d distance=%d angle_delta=%d key=%d locked=%d interaction=%d attempts=%d",
                 status, atDoor ? 1 : 0, currentGoal.c_str(), portal.wall, portal.from, portal.to, portal.wallPush ? 1 : 0,
                 portal.sectorPush ? 1 : 0, portal.sectorPushCurrent ? 1 : 0, hit, target, extra,
                 int(std::sqrt(double(distance2(observation.x, observation.y, portal.x, portal.y)))),
                 angleDelta(targetAngle, observation.angle), portal.key, portal.locked ? 1 : 0,
                 int(memory.interaction), memory.attempts);
        event("use_probe", detail);
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

    void selectObjective(const Objective &objective, const char *goal)
    {
        currentObjective = objective;
        currentObjective.active = true;
        setGoal(goal, objective.id);
        if (objective.type == kObjectiveFrontier)
        {
            PendingWork *pending = findPendingFrontier(objective.wall,
                                                       objective.sector,
                                                       objective.targetSector);
            if (pending)
                pending->state = PendingWork::kActive;
            if (lastFrontierWall == objective.wall
                && lastFrontierSource == objective.sector
                && lastFrontierTarget == objective.targetSector)
            {
                ++frontierSelectionCount;
                char detail[192];
                snprintf(detail, sizeof(detail),
                         "wall=%d source=%d target=%d current=%d count=%d",
                         objective.wall, objective.sector, objective.targetSector,
                         observation.sector, frontierSelectionCount);
                event("frontier_reselected", detail);
            }
            else
            {
                lastFrontierWall = objective.wall;
                lastFrontierSource = objective.sector;
                lastFrontierTarget = objective.targetSector;
                frontierSelectionCount = 1;
            }
        }
        char detail[192];
        snprintf(detail, sizeof(detail), "type=%d id=%d sector=%d target_sector=%d wall=%d current=%d",
                 int(objective.type), objective.id, objective.sector,
                 objective.targetSector, objective.wall, observation.sector);
        event("objective_selected", detail);
    }

    void completeObjective(const char *reason)
    {
        if (!currentObjective.active)
            return;
        if (currentObjective.type == kObjectiveFrontier)
            retirePendingFrontier(currentObjective.wall, currentObjective.sector,
                                  currentObjective.targetSector);
        char detail[128];
        snprintf(detail, sizeof(detail), "type=%d id=%d wall=%d source=%d target=%d reason=%s",
                 int(currentObjective.type), currentObjective.id, currentObjective.wall,
                 currentObjective.sector, currentObjective.targetSector, reason);
        event("objective_completed", detail);
        currentObjective = Objective{};
        movementTargetActive = false;
        resetNavigation();
    }

    void invalidateObjective(const char *reason)
    {
        if (!currentObjective.active)
            return;
        if (currentObjective.type == kObjectiveFrontier)
        {
            PendingWork *pending = findPendingFrontier(currentObjective.wall,
                                                       currentObjective.sector,
                                                       currentObjective.targetSector);
            if (pending)
            {
                const bool sourceWasObserved = observation.sector == currentObjective.sector;
                pending->state = sourceWasObserved
                    ? PendingWork::kTemporarilyUnavailable
                    : PendingWork::kDiscovered;
                pending->resolved = false;
            }
        }
        char detail[128];
        snprintf(detail, sizeof(detail), "type=%d id=%d wall=%d source=%d target=%d reason=%s",
                 int(currentObjective.type), currentObjective.id, currentObjective.wall,
                 currentObjective.sector, currentObjective.targetSector, reason);
        event("objective_invalidated", detail);
        currentObjective = Objective{};
        movementTargetActive = false;
        resetNavigation();
        currentGoal.clear();
        currentGoalTarget = -1;
    }

    bool urgentThreat()
    {
        if (lastDamageTick >= 0 && observation.tick - lastDamageTick <= 2 * kTicsPerSec)
            return true;
        const VisibleObject *enemy = selectObject(kObjectEnemy);
        return enemy && !unreachableEnemies.count(enemy->sprite)
            && enemy->sector == observation.sector
            && distance2(enemy->x, enemy->y, observation.x, observation.y) <= 4096 * 4096;
    }

    GINPUT retreatFromEnemy(const VisibleObject &enemy)
    {
        ensureNavTopology();
        const int playerCell = nearestNavCell(observation.sector, observation.x, observation.y);
        if (playerCell < 0 || navCells[playerCell].walkArea < 0)
            return GINPUT{};
        const int area = navCells[playerCell].walkArea;
        const int currentEnemyDistance = distance2(observation.x, observation.y,
                                                   enemy.x, enemy.y);
        int bestCell = -1;
        int bestEnemyDistance = currentEnemyDistance;
        for (const NavCell &cell : navCells)
        {
            if (cell.sector != observation.sector || cell.walkArea != area)
                continue;
            const MovementProbe probe = probeMovement(
                observation.x, observation.y, observation.z, observation.sector,
                cell.center.x, cell.center.y, observation.sector, 512);
            if (!probe.reachable)
                continue;
            const int enemyDistance = distance2(cell.center.x, cell.center.y,
                                                enemy.x, enemy.y);
            if (enemyDistance > bestEnemyDistance)
            {
                bestEnemyDistance = enemyDistance;
                bestCell = cell.id;
            }
        }
        if (bestCell < 0)
        {
            // A partially observed area can have no currently usable cell
            // center.  Keep the no-ranged policy safe anyway: move away from
            // the attacker, never toward it, and let the next observation
            // discover a better retreat cell.
            const int awayAngle = wrapAngle(getangle(
                observation.x - enemy.x, observation.y - enemy.y));
            const int retreatX = observation.x + mulscale30(Cos(awayAngle), 4096);
            const int retreatY = observation.y + mulscale30(Sin(awayAngle), 4096);
            event("combat_retreat", "mode=direct_away_no_cell");
            setGoal("COMBAT_RETREAT", enemy.sprite);
            return steerTo(retreatX, retreatY, false, false, -1,
                           observation.sector, kTraversalUnknown);
        }
        char detail[160];
        snprintf(detail, sizeof(detail), "enemy=%d cell=%d distance=%d->%d",
                 enemy.sprite, bestCell,
                 int(std::sqrt(double(currentEnemyDistance))),
                 int(std::sqrt(double(bestEnemyDistance))));
        event("combat_retreat", detail);
        setGoal("COMBAT_RETREAT", enemy.sprite);
        const NavCell &cell = navCells[bestCell];
        return navigateTo(cell.center.x, cell.center.y, observation.z,
                          observation.sector, -1, kTraversalUnknown);
    }

    bool interactionStillBusy(const InteractionMemory &memory) const
    {
        if (memory.target.wall >= 0 && inRange(memory.target.wall, 0, numwalls))
        {
            const int extra = wall[memory.target.wall].extra;
            if (extra > 0 && extra < kMaxXWalls && xwall[extra].busy != 0)
                return true;
        }
        const int sectorId = memory.target.sectorPush ? memory.target.to
            : memory.target.sectorPushCurrent ? memory.target.from : -1;
        if (inRange(sectorId, 0, numsectors))
        {
            const int extra = sector[sectorId].extra;
            if (extra > 0 && extra < kMaxXSectors && xsector[extra].busy != 0)
                return true;
        }
        return false;
    }

    GINPUT executeObjective()
    {
        if (!currentObjective.active)
            return GINPUT{};
        if (currentObjective.type == kObjectiveInteraction)
        {
            auto memory = interactions.find(currentObjective.interactionKey);
            if (memory == interactions.end() || !memory->second.observed)
            {
                invalidateObjective("interaction_not_known");
                return GINPUT{};
            }
            InteractionMemory &interaction = memory->second;
            if (interaction.engineAccepted && interaction.state == 2
                && interactionStillBusy(interaction)
                && observation.tick - interaction.lastActivationTick < 2 * kTicsPerSec)
            {
                const int key = interactionMemoryKey(interaction);
                if (lastInteractionWaitKey != key)
                {
                    lastInteractionWaitKey = key;
                    event("interaction_waiting_for_settle", "engine_accepted=1 source_busy=1");
                }
                return GINPUT{};
            }
            if (interaction.engineAccepted && interaction.state == 2)
            {
                lastInteractionWaitKey = -1;
                completeObjective("engine_accepted_and_settled");
                return GINPUT{};
            }
            if (interaction.state == 3)
            {
                invalidateObjective("interaction_rejected");
                return GINPUT{};
            }
            return steerInteraction(interaction);
        }
        if (currentObjective.type == kObjectiveFrontier)
        {
            const bool exactCrossing = observation.sector == currentObjective.targetSector
                && lastTransitionTick == observation.tick
                && lastTransitionFrom == currentObjective.sector
                && lastTransitionTo == currentObjective.targetSector;
            if (exactCrossing)
            {
                completeObjective("sector_transition");
                return GINPUT{};
            }
            if (portalPlanWall == currentObjective.wall && portalPlanAttempts >= 3)
            {
                const int edgeId = currentObjective.wall * 65536 + currentObjective.targetSector;
                for (const Portal &portal : observation.portals)
                    if (portal.wall == currentObjective.wall
                        && portal.to == currentObjective.targetSector)
                        localFailureSignatures[edgeId] = portalGeometrySignature(portal);
                retirePendingFrontier(currentObjective.wall, currentObjective.sector,
                                      currentObjective.targetSector);
                invalidateObjective("local_navigation_exhausted");
                return GINPUT{};
            }
            if (observation.sector == currentObjective.sector)
            {
                for (const Portal &portal : observation.portals)
                    if (portal.wall == currentObjective.wall
                        && portal.to == currentObjective.targetSector)
                        return steerPortal(portal);
                auto known = knownGraph.find(observation.sector);
                if (known != knownGraph.end())
                    for (const Portal &portal : known->second)
                        if (portal.wall == currentObjective.wall
                            && portal.to == currentObjective.targetSector)
                            return steerPortal(portal);
                retirePendingFrontier(currentObjective.wall, currentObjective.sector,
                                      currentObjective.targetSector);
                invalidateObjective("frontier_not_currently_observable");
                return GINPUT{};
            }
            if (observation.sector != currentObjective.sector)
            {
                Portal sourceRoute;
                if (findKnownRoute(currentObjective.sector, sourceRoute))
                    return steerPortal(sourceRoute);
                invalidateObjective("frontier_no_current_source_route");
                return GINPUT{};
            }
            invalidateObjective("frontier_source_state_changed");
            return GINPUT{};
        }
        if (currentObjective.type == kObjectivePickup
            || currentObjective.type == kObjectiveKey)
        {
            const ObjectKind kind = currentObjective.type == kObjectiveKey
                ? kObjectKey : kObjectPickup;
            const VisibleObject *object = nullptr;
            for (const VisibleObject &candidate : observation.objects)
            {
                if (candidate.sprite == currentObjective.id && candidate.kind == kind)
                {
                    object = &candidate;
                    break;
                }
            }
            if (!object)
            {
                auto remembered = objectMemory.find(currentObjective.id);
                if (remembered == objectMemory.end()
                    || !remembered->second.observed
                    || remembered->second.collected)
                {
                    completeObjective("pickup_confirmed_collected");
                    return GINPUT{};
                }
                if (!objectStillPresent(remembered->second))
                {
                    remembered->second.collected = true;
                    event("pickup_confirmed_collected", "source=sprite_removed_or_inventory");
                    completeObjective("pickup_confirmed_collected");
                    return GINPUT{};
                }
                // The observation is deliberately non-omniscient.  Keep
                // pursuing the last known pose while a corner or moving
                // wall temporarily occludes the pickup.
                selectedObject = remembered->second.object;
                object = &selectedObject;
            }
            if (object->sector != observation.sector)
            {
                Portal route;
                if (!findKnownRoute(object->sector, route))
                {
                    invalidateObjective("object_no_current_route");
                    return GINPUT{};
                }
                return steerPortal(route);
            }
            return navigateTo(object->x, object->y, object->z, object->sector,
                              object->sprite, kTraversalUnknown);
        }
        if (currentObjective.type == kObjectiveCombat)
        {
            const VisibleObject *enemy = nullptr;
            for (const VisibleObject &candidate : observation.objects)
                if (candidate.sprite == currentObjective.id && candidate.kind == kObjectEnemy)
                {
                    enemy = &candidate;
                    break;
                }
            if (!enemy)
            {
                completeObjective("threat_gone");
                return GINPUT{};
            }
            int rangedWeapon = kWeaponNone;
            if (rangedWeaponAvailable(rangedWeapon))
            {
                // A reachable enemy is not a reason to close to melee.  The
                // same visible target can be attacked while the player holds
                // the current position or strafes independently.
                return aimAndShoot(*enemy, rangedWeapon);
            }
            const int meleeRange = 1024;
            if (distance2(enemy->x, enemy->y, observation.x, observation.y)
                    <= meleeRange * meleeRange
                && directObjectReachable(*enemy))
                return navigateTo(enemy->x, enemy->y, enemy->z, enemy->sector,
                                  enemy->sprite, kTraversalUnknown, true);
            GINPUT retreat = retreatFromEnemy(*enemy);
            if (retreat.forward || retreat.strafe || retreat.q16turn || retreat.buttonFlags.shoot)
                return retreat;
            unreachableEnemies.insert(enemy->sprite);
            lastDamageTick = -1;
            event("threat_deferred", "reason=no_ranged_weapon_and_no_safe_retreat");
            invalidateObjective("threat_not_reachable");
        }
        return GINPUT{};
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
        targetBestDistance2 = targetLastDistance2;
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
        const bool horizontalProgress = currentDistance2 + 4096 < targetBestDistance2;
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
            targetBestDistance2 = std::min(targetBestDistance2, currentDistance2);
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

    MovementProbe probeMovement(int startX, int startY, int startZ, int startSector,
                                int targetX, int targetY, int targetSector, int tolerance) const
    {
        MovementProbe result;
        result.x = startX;
        result.y = startY;
        result.sector = startSector;
        if (!inRange(startSector, 0, numsectors) || !inRange(targetSector, 0, numsectors))
            return result;
        vec3_t position = { startX, startY, startZ };
        int16_t sectorNumber = int16_t(startSector);
        const int64_t dx = int64_t(targetX) - startX;
        const int64_t dy = int64_t(targetY) - startY;
        const int32_t xvect = int32_t(std::max<int64_t>(INT32_MIN + 1, std::min<int64_t>(INT32_MAX, dx << 14)));
        const int32_t yvect = int32_t(std::max<int64_t>(INT32_MIN + 1, std::min<int64_t>(INT32_MAX, dy << 14)));
        const int radius = gMe && gMe->pSprite ? (gMe->pSprite->clipdist << 2) : 128;
        int ceilingDistance = 0;
        int floorDistance = 0;
        playerCollisionDistances(ceilingDistance, floorDistance);
        result.hit = clipmove(&position, &sectorNumber, xvect, yvect, radius,
                              ceilingDistance, floorDistance, CLIPMASK0);
        result.x = position.x;
        result.y = position.y;
        result.sector = sectorNumber;
        if ((result.hit & 0xc000) == 0x8000)
            result.wall = result.hit & 0x3fff;
        const int remaining = distance2(position.x, position.y, targetX, targetY);
        result.reachable = sectorNumber == targetSector && remaining <= tolerance * tolerance;
        return result;
    }

    static int64_t navCross(const LocalWaypoint &a, const LocalWaypoint &b,
                            const LocalWaypoint &c)
    {
        return int64_t(b.x - a.x) * (c.y - a.y)
            - int64_t(b.y - a.y) * (c.x - a.x);
    }

    static bool navPointInTriangle(const LocalWaypoint &p, const LocalWaypoint *v)
    {
        const int64_t a = navCross(v[0], v[1], p);
        const int64_t b = navCross(v[1], v[2], p);
        const int64_t c = navCross(v[2], v[0], p);
        return (a >= 0 && b >= 0 && c >= 0)
            || (a <= 0 && b <= 0 && c <= 0);
    }

    int navGeometrySignature() const
    {
        int signature = 17;
        std::set<int> sectors = observedSectors;
        if (inRange(observation.sector, 0, numsectors))
            sectors.insert(observation.sector);
        for (int sectorId : sectors)
        {
            if (!inRange(sectorId, 0, numsectors))
                continue;
            const sectortype &sectorRecord = sector[sectorId];
            signature = signature * 31 + sectorId;
            signature = signature * 31 + sectorRecord.wallptr;
            signature = signature * 31 + sectorRecord.wallnum;
            signature = signature * 31 + sectorRecord.floorz;
            signature = signature * 31 + sectorRecord.ceilingz;
            for (int i = 0; i < sectorRecord.wallnum; ++i)
            {
                const int wallId = sectorRecord.wallptr + i;
                if (!inRange(wallId, 0, numwalls))
                    continue;
                const walltype &wallRecord = wall[wallId];
                signature = signature * 31 + wallRecord.x;
                signature = signature * 31 + wallRecord.y;
                signature = signature * 31 + wallRecord.point2;
                signature = signature * 31 + wallRecord.nextsector;
                signature = signature * 31 + wallRecord.cstat;
            }
        }
        return signature;
    }

    void addNavLink(int from, int to, NavEdgeMode mode, int wallId,
                    LocalWaypoint gateway = {}, bool hasGateway = false)
    {
        if (from < 0 || to < 0 || from == to)
            return;
        NavCell &cell = navCells[from];
        for (const NavLink &link : cell.links)
            if (link.target == to && link.wall == wallId)
                return;
        NavLink link;
        link.target = to;
        link.mode = mode;
        link.wall = wallId;
        link.gateway = gateway;
        link.hasGateway = hasGateway;
        cell.links.push_back(link);
    }

    int nearestNavCell(int sectorId, int x, int y) const
    {
        int best = -1;
        int bestDistance = INT32_MAX;
        for (const NavCell &cell : navCells)
        {
            if (cell.sector != sectorId)
                continue;
            if (navPointInTriangle({ x, y }, cell.vertex))
                return cell.id;
            const int currentDistance = distance2(x, y, cell.center.x, cell.center.y);
            if (currentDistance < bestDistance)
            {
                bestDistance = currentDistance;
                best = cell.id;
            }
        }
        return best;
    }

    void buildNavSector(int sectorId)
    {
        if (!inRange(sectorId, 0, numsectors))
            return;
        const sectortype &sectorRecord = sector[sectorId];
        std::set<int> sectorWalls;
        for (int i = 0; i < sectorRecord.wallnum; ++i)
        {
            const int wallId = sectorRecord.wallptr + i;
            if (inRange(wallId, 0, numwalls))
                sectorWalls.insert(wallId);
        }

        std::set<int> visitedWalls;
        std::vector<std::vector<LocalWaypoint>> loops;
        for (int start : sectorWalls)
        {
            if (visitedWalls.count(start))
                continue;
            std::vector<LocalWaypoint> loop;
            int current = start;
            std::set<int> loopWalls;
            while (sectorWalls.count(current) && !loopWalls.count(current))
            {
                loopWalls.insert(current);
                visitedWalls.insert(current);
                loop.push_back({ wall[current].x, wall[current].y });
                current = wall[current].point2;
            }
            if (current == start && loop.size() >= 3)
                loops.push_back(loop);
        }

        char audit[128];
        snprintf(audit, sizeof(audit), "sector=%d loops=%u", sectorId,
                 unsigned(loops.size()));
        event("nav_loop_audit", audit);

        for (size_t loopIndex = 0; loopIndex < loops.size(); ++loopIndex)
        {
            const std::vector<LocalWaypoint> &polygon = loops[loopIndex];
            int64_t area = 0;
            for (size_t i = 0; i < polygon.size(); ++i)
            {
                const LocalWaypoint &a = polygon[i];
                const LocalWaypoint &b = polygon[(i + 1) % polygon.size()];
                area += int64_t(a.x) * b.y - int64_t(b.x) * a.y;
            }
            const bool ccw = area > 0;
            std::vector<int> remaining;
            for (size_t i = 0; i < polygon.size(); ++i)
                remaining.push_back(int(i));
            const size_t guardLimit = polygon.size() * polygon.size();
            size_t guard = 0;
            auto addTriangle = [&](const LocalWaypoint triangle[3])
            {
                const LocalWaypoint center = {
                    (triangle[0].x + triangle[1].x + triangle[2].x) / 3,
                    (triangle[0].y + triangle[1].y + triangle[2].y) / 3 };
                // Build's point-in-sector test is the authority for rejecting
                // triangles whose centroid falls in an inner island/hole.
                if (inside(center.x, center.y, sectorId) != 1)
                    return;
                NavCell cell;
                cell.id = int(navCells.size());
                cell.sector = sectorId;
                for (int k = 0; k < 3; ++k)
                    cell.vertex[k] = triangle[k];
                cell.center = center;
                navCells.push_back(cell);
            };
            while (remaining.size() >= 3 && guard++ < guardLimit)
            {
                bool clipped = false;
                for (size_t i = 0; i < remaining.size(); ++i)
                {
                    const int previous = remaining[(i + remaining.size() - 1) % remaining.size()];
                    const int current = remaining[i];
                    const int next = remaining[(i + 1) % remaining.size()];
                    const int64_t turn = navCross(polygon[previous], polygon[current], polygon[next]);
                    if ((ccw && turn <= 0) || (!ccw && turn >= 0))
                        continue;
                    LocalWaypoint triangle[3] = { polygon[previous], polygon[current], polygon[next] };
                    bool containsVertex = false;
                    for (int other : remaining)
                    {
                        if (other == previous || other == current || other == next)
                            continue;
                        if (navPointInTriangle(polygon[other], triangle))
                        {
                            containsVertex = true;
                            break;
                        }
                    }
                    if (containsVertex)
                        continue;
                    addTriangle(triangle);
                    remaining.erase(remaining.begin() + i);
                    clipped = true;
                    break;
                }
                if (!clipped)
                    break;
            }
            if (remaining.size() >= 3)
            {
                const int base = remaining.front();
                for (size_t i = 1; i + 1 < remaining.size(); ++i)
                {
                    LocalWaypoint triangle[3] = { polygon[base], polygon[remaining[i]],
                                                  polygon[remaining[i + 1]] };
                    addTriangle(triangle);
                }
                event("nav_loop_fallback", "reason=ear_clip_incomplete");
            }
        }
    }

    NavEdgeMode navModeForPortal(const Portal &portal) const
    {
        if (portal.interactionAffordance && !portal.traversable)
            return kNavInteraction;
        if (portal.jumpable)
            return kNavJump;
        if (portal.crouchable)
            return kNavCrouch;
        if (portal.dropSafe)
            return kNavDrop;
        if (portal.floorDelta != 0)
            return kNavStep;
        if (portal.walkable)
            return kNavWalk;
        return kNavBlocked;
    }

    void ensureNavTopology()
    {
        const int signature = navGeometrySignature();
        if (signature == navTopologySignature && !navCells.empty())
            return;
        navTopologySignature = signature;
        navCells.clear();
        std::set<int> sectors = observedSectors;
        if (inRange(observation.sector, 0, numsectors))
            sectors.insert(observation.sector);
        for (int sectorId : sectors)
            buildNavSector(sectorId);

        for (size_t i = 0; i < navCells.size(); ++i)
            for (size_t j = i + 1; j < navCells.size(); ++j)
            {
                if (navCells[i].sector != navCells[j].sector)
                    continue;
                bool shared = false;
                LocalWaypoint gateway = {};
                for (int a = 0; a < 3 && !shared; ++a)
                    for (int b = 0; b < 3 && !shared; ++b)
                    {
                        const LocalWaypoint &a1 = navCells[i].vertex[a];
                        const LocalWaypoint &a2 = navCells[i].vertex[(a + 1) % 3];
                        const LocalWaypoint &b1 = navCells[j].vertex[b];
                        const LocalWaypoint &b2 = navCells[j].vertex[(b + 1) % 3];
                        shared = (a1.x == b2.x && a1.y == b2.y
                                  && a2.x == b1.x && a2.y == b1.y)
                            || (a1.x == b1.x && a1.y == b1.y
                                && a2.x == b2.x && a2.y == b2.y);
                        if (shared)
                        {
                            gateway = { (a1.x + a2.x) / 2,
                                        (a1.y + a2.y) / 2 };
                        }
                    }
                if (shared)
                {
                    addNavLink(int(i), int(j), kNavWalk, -1, gateway, true);
                    addNavLink(int(j), int(i), kNavWalk, -1, gateway, true);
                }
            }

        for (const auto &entry : knownGraph)
            for (const Portal &portal : entry.second)
            {
                if (!(portal.traversable || portal.jumpable)
                    || !observedSectors.count(portal.from)
                    || !observedSectors.count(portal.to))
                    continue;
                const int from = nearestNavCell(portal.from, portal.x, portal.y);
                const int to = nearestNavCell(portal.to, portal.x, portal.y);
                if (from >= 0 && to >= 0)
                {
                    const NavEdgeMode mode = navModeForPortal(portal);
                    addNavLink(from, to, mode, portal.wall,
                               { portal.x, portal.y }, true);
                }
            }

        int nextArea = 0;
        for (NavCell &cell : navCells)
            cell.walkArea = -1;
        for (NavCell &cell : navCells)
        {
            if (cell.walkArea >= 0)
                continue;
            std::deque<int> queue;
            queue.push_back(cell.id);
            cell.walkArea = nextArea;
            while (!queue.empty())
            {
                const int current = queue.front();
                queue.pop_front();
                for (const NavLink &link : navCells[current].links)
                {
                    if (link.mode != kNavWalk && link.mode != kNavStep)
                        continue;
                    if (navCells[link.target].walkArea < 0)
                    {
                        navCells[link.target].walkArea = nextArea;
                        queue.push_back(link.target);
                    }
                }
            }
            ++nextArea;
        }
        ++navTopologyRevision;
        char detail[128];
        snprintf(detail, sizeof(detail), "revision=%d cells=%u areas=%d sectors=%u",
                 navTopologyRevision, unsigned(navCells.size()), nextArea,
                 unsigned(sectors.size()));
        event("nav_topology_rebuilt", detail);
    }

    bool buildNavRoute(int targetX, int targetY, int targetSector, int signature)
    {
        if (targetSector != observation.sector)
            return false;
        ensureNavTopology();
        const int start = nearestNavCell(observation.sector, observation.x, observation.y);
        const int target = nearestNavCell(targetSector, targetX, targetY);
        if (start < 0 || target < 0 || start == target)
            return false;
        std::deque<int> queue;
        std::vector<int> parent(navCells.size(), -1);
        std::vector<bool> reached(navCells.size(), false);
        queue.push_back(start);
        reached[start] = true;
        while (!queue.empty() && !reached[target])
        {
            const int current = queue.front();
            queue.pop_front();
            for (const NavLink &link : navCells[current].links)
            {
                if (link.mode != kNavWalk && link.mode != kNavStep)
                    continue;
                if (!reached[link.target])
                {
                    reached[link.target] = true;
                    parent[link.target] = current;
                    queue.push_back(link.target);
                }
            }
        }
        if (!reached[target])
            return false;
        std::vector<int> cells;
        for (int cursor = target; cursor >= 0; cursor = parent[cursor])
        {
            cells.push_back(cursor);
            if (cursor == start)
                break;
        }
        if (cells.back() != start)
            return false;
        navRoute.clear();
        const int radius = gMe && gMe->pSprite ? (gMe->pSprite->clipdist << 2) : 128;
        for (auto it = cells.rbegin(); it != cells.rend(); ++it)
        {
            if (*it == start)
                continue;
            const int fromCell = *(it - 1);
            const int toCell = *it;
            LocalWaypoint waypoint = navCells[toCell].center;
            for (const NavLink &link : navCells[fromCell].links)
            {
                if (link.target != toCell)
                    continue;
                if (link.hasGateway)
                {
                    waypoint = link.gateway;
                    const int dx = navCells[toCell].center.x - waypoint.x;
                    const int dy = navCells[toCell].center.y - waypoint.y;
                    const int length = std::max(1, int(std::sqrt(double(dx * dx + dy * dy))));
                    const int inset = std::min(radius + 256, length / 2);
                    waypoint.x += dx * inset / length;
                    waypoint.y += dy * inset / length;
                }
                break;
            }
            if (navRoute.empty() || navRoute.back().x != waypoint.x
                || navRoute.back().y != waypoint.y)
                navRoute.push_back(waypoint);
        }
        navRouteIndex = 0;
        navRouteSignature = signature;
        navRouteTopologyRevision = navTopologyRevision;
        char detail[160];
        snprintf(detail, sizeof(detail), "cells=%u target=(%d,%d) area=%d",
                 unsigned(navRoute.size()), targetX, targetY, navCells[target].walkArea);
        event("nav_route_selected", detail);
        return !navRoute.empty();
    }

    bool collisionProbe(int startX, int startY, int startZ, int startSector,
                        int targetX, int targetY, int targetSector, int tolerance) const
    {
        return probeMovement(startX, startY, startZ, startSector,
                             targetX, targetY, targetSector, tolerance).reachable;
    }

    int navigationSignature(int x, int y, int z, int sector, int id) const
    {
        int signature = x * 31 + y;
        signature = signature * 31 + z;
        signature = signature * 31 + sector;
        signature = signature * 31 + id;
        return signature;
    }

    void resetNavigation()
    {
        navigationActive = false;
        navigationWaypointActive = false;
        navigationUsingDetour = false;
        navigationDetourWall = -1;
        navigationDetourDepth = 0;
        navigationRecentWalls.clear();
        navRoute.clear();
        navRouteIndex = 0;
        navRouteSignature = 0;
        navigationFailureSignature = 0;
        navigationFailureCount = 0;
        lastNavigationFailureTick = -1;
    }

    NavEdgeMode classifyNavigationBlock(int wallId, int x, int y) const
    {
        if (!inRange(wallId, 0, numwalls))
            return kNavBlocked;
        const walltype &wallRecord = wall[wallId];
        if (!inRange(wallRecord.point2, 0, numwalls))
            return kNavBlocked;
        if (!inRange(wallRecord.nextsector, 0, numsectors))
        {
            if (inRange(wallRecord.extra, 1, kMaxXWalls)
                && xwall[wallRecord.extra].triggerPush)
                return kNavInteraction;
            return kNavBlocked;
        }
        if (wallRecord.cstat & 1)
        {
            if (inRange(wallRecord.extra, 1, kMaxXWalls)
                && xwall[wallRecord.extra].triggerPush)
                return kNavInteraction;
            return kNavBlocked;
        }
        const int fromFloor = getflorzofslope(observation.sector, x, y);
        const int toFloor = getflorzofslope(wallRecord.nextsector, x, y);
        const int fromCeiling = getceilzofslope(observation.sector, x, y);
        const int toCeiling = getceilzofslope(wallRecord.nextsector, x, y);
        const int clearance = std::min(fromFloor - fromCeiling, toFloor - toCeiling);
        const int width = int(std::sqrt(double(distance2(
            wallRecord.x, wallRecord.y, wall[wallRecord.point2].x,
            wall[wallRecord.point2].y))));
        const int floorDelta = toFloor - fromFloor;
        const int body = playerBodyClearance();
        const int rise = playerJumpRiseLimit();
        bool mechanism = false;
        if (inRange(wallRecord.extra, 1, kMaxXWalls)
            && xwall[wallRecord.extra].triggerPush)
            mechanism = true;
        if (inRange(wallRecord.nextsector, 0, numsectors)
            && sector[wallRecord.nextsector].extra > 0
            && sector[wallRecord.nextsector].extra < kMaxXSectors
            && xsector[sector[wallRecord.nextsector].extra].Wallpush)
            mechanism = true;
        if (width >= kPlayerPassageWidth && clearance >= body
            && std::abs(floorDelta) <= kMaxWalkableStep)
            return floorDelta == 0 ? kNavWalk : kNavStep;
        if (width >= kPlayerPassageWidth && clearance >= body
            && !mechanism && floorDelta < 0 && -floorDelta <= rise)
            return kNavJump;
        if (width >= kPlayerPassageWidth && clearance >= body
            && floorDelta > 0 && floorDelta <= rise)
            return kNavDrop;
        if (inRange(wallRecord.extra, 1, kMaxXWalls)
            && xwall[wallRecord.extra].triggerPush)
            return kNavInteraction;
        return kNavBlocked;
    }

    bool chooseNavigationDetour(const MovementProbe &blocked)
    {
        if (blocked.wall < 0 || !inRange(blocked.wall, 0, numwalls)
            || navigationDetourDepth >= 4
            || navigationRecentWalls.count(blocked.wall))
            return false;

        const walltype &obstacle = wall[blocked.wall];
        if (!inRange(obstacle.point2, 0, numwalls))
            return false;
        const walltype &obstacleEnd = wall[obstacle.point2];
        const int radius = gMe && gMe->pSprite ? (gMe->pSprite->clipdist << 2) : 128;
        const int margin = std::max(512, radius + 256);
        const int dirs[][2] = {{ 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 },
                               { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 }};
        const int corners[][2] = {{ obstacle.x, obstacle.y },
                                  { obstacleEnd.x, obstacleEnd.y }};
        struct Candidate { int x; int y; bool finalReachable; int distance; };
        std::vector<Candidate> candidates;
        for (const auto &corner : corners)
        {
            for (const auto &dir : dirs)
            {
                const int norm = (dir[0] && dir[1]) ? 181 : 256;
                Candidate candidate;
                candidate.x = corner[0] + dir[0] * margin * 256 / norm;
                candidate.y = corner[1] + dir[1] * margin * 256 / norm;
                const MovementProbe fromPlayer = probeMovement(
                    observation.x, observation.y, observation.z, observation.sector,
                    candidate.x, candidate.y, observation.sector, std::max(256, radius));
                if (!fromPlayer.reachable)
                    continue;
                const MovementProbe toOriginal = probeMovement(
                    candidate.x, candidate.y, observation.z, observation.sector,
                    navigationOriginalX, navigationOriginalY, navigationOriginalSector,
                    std::max(256, radius));
                candidate.finalReachable = toOriginal.reachable;
                candidate.distance = distance2(candidate.x, candidate.y,
                                               navigationOriginalX, navigationOriginalY);
                candidates.push_back(candidate);
            }
        }
        if (candidates.empty())
            return false;
        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const Candidate &left, const Candidate &right)
                         {
                             if (left.finalReachable != right.finalReachable)
                                 return left.finalReachable > right.finalReachable;
                             return left.distance < right.distance;
                         });
        const Candidate &selected = candidates.front();
        navigationDetourWall = blocked.wall;
        navigationRecentWalls.insert(blocked.wall);
        navigationWaypointX = selected.x;
        navigationWaypointY = selected.y;
        navigationWaypointActive = true;
        navigationUsingDetour = true;
        ++navigationDetourDepth;
        char detail[192];
        snprintf(detail, sizeof(detail), "wall=%d waypoint=(%d,%d) depth=%d final_direct=%d",
                 blocked.wall, selected.x, selected.y, navigationDetourDepth,
                 selected.finalReachable ? 1 : 0);
        event("navigation_detour", detail);
        return true;
    }

    bool rememberCollisionInteraction(int wallIndex)
    {
        if (!inRange(wallIndex, 0, numwalls)
            || !inRange(observation.sector, 0, numsectors))
            return false;

        const walltype &wallRecord = wall[wallIndex];
        if (!inRange(wallRecord.point2, 0, numwalls))
            return false;
        if (!inRange(wallRecord.extra, 1, kMaxXWalls)
            || !xwall[wallRecord.extra].triggerPush)
            return false;

        const walltype &nextWall = wall[wallRecord.point2];
        InteractionCandidate candidate;
        candidate.kind = kInteractionWall;
        candidate.id = wallIndex;
        candidate.fromSector = observation.sector;
        candidate.targetSector = -1;
        candidate.x = (wallRecord.x + nextWall.x) / 2;
        candidate.y = (wallRecord.y + nextWall.y) / 2;
        candidate.z = observation.z;
        candidate.key = xwall[wallRecord.extra].key;
        candidate.locked = xwall[wallRecord.extra].locked != 0;
        candidate.reversible = true;
        candidate.target.wall = wallIndex;
        candidate.target.from = observation.sector;
        candidate.target.to = -1;
        candidate.target.x = candidate.x;
        candidate.target.y = candidate.y;
        candidate.target.wallPush = true;
        candidate.target.x1 = wallRecord.x;
        candidate.target.y1 = wallRecord.y;
        candidate.target.x2 = nextWall.x;
        candidate.target.y2 = nextWall.y;
        setInteractionGeometry(candidate.target);
        candidate.z = candidate.target.z;
        rememberInteraction(candidate);
        return true;
    }

    GINPUT navigateTo(int x, int y, int z, int targetSector, int targetId = -1,
                      TraversalCapability capability = kTraversalUnknown, bool shoot = false)
    {
        navigationActionableBlocker = -1;
        const int signature = navigationSignature(x, y, z, targetSector, targetId);
        if (!navigationActive || navigationOriginalSignature != signature)
        {
            resetNavigation();
            navigationActive = true;
            navigationOriginalX = x;
            navigationOriginalY = y;
            navigationOriginalZ = z;
            navigationOriginalSector = targetSector;
            navigationOriginalId = targetId;
            navigationOriginalSignature = signature;
        }

        const int radius = gMe && gMe->pSprite ? (gMe->pSprite->clipdist << 2) : 128;
        const int tolerance = std::max(256, radius);
        MovementProbe direct = probeMovement(observation.x, observation.y, observation.z,
                                             observation.sector, x, y, targetSector, tolerance);
        if (direct.reachable)
        {
            if (navigationUsingDetour)
                event("navigation_retarget_original");
            navigationWaypointActive = false;
            navigationUsingDetour = false;
            navigationDetourDepth = 0;
            navigationRecentWalls.clear();
            return steerTo(x, y, false, shoot, targetId, targetSector, capability);
        }

        if (!navRoute.empty() && navRouteSignature == signature
            && navRouteTopologyRevision == navTopologyRevision)
        {
            const LocalWaypoint &waypoint = navRoute[navRouteIndex];
            if (distance2(observation.x, observation.y, waypoint.x, waypoint.y)
                <= tolerance * tolerance)
            {
                ++navRouteIndex;
                if (navRouteIndex >= navRoute.size())
                    navRoute.clear();
                else
                    event("nav_retarget_original");
            }
            if (!navRoute.empty())
            {
                const LocalWaypoint &next = navRoute[navRouteIndex];
                const MovementProbe segment = probeMovement(
                    observation.x, observation.y, observation.z, observation.sector,
                    next.x, next.y, observation.sector, tolerance);
                if (segment.reachable)
                {
                    event("nav_cell_waypoint", "source=derived_triangle_cell");
                    return steerTo(next.x, next.y, false, shoot, targetId,
                                   observation.sector, capability);
                }
                event("nav_segment_failed", "source=derived_triangle_cell");
                navRoute.clear();
            }
        }

        if (navigationWaypointActive)
        {
            if (distance2(observation.x, observation.y, navigationWaypointX,
                          navigationWaypointY) <= tolerance * tolerance)
            {
                navigationWaypointActive = false;
                event("navigation_retarget_original");
                return steerTo(x, y, false, shoot, targetId, targetSector, capability);
            }
            const MovementProbe waypoint = probeMovement(
                observation.x, observation.y, observation.z, observation.sector,
                navigationWaypointX, navigationWaypointY, observation.sector, tolerance);
            if (!waypoint.reachable && waypoint.wall >= 0
                && waypoint.wall != navigationDetourWall)
            {
                navigationWaypointActive = false;
                navigationUsingDetour = false;
            }
            else
            {
                return steerTo(navigationWaypointX, navigationWaypointY, false, false,
                               targetId, observation.sector, capability);
            }
        }

        if (direct.wall >= 0)
        {
            const NavEdgeMode blockMode = classifyNavigationBlock(direct.wall, direct.x, direct.y);
            const int nextSector = inRange(direct.wall, 0, numwalls)
                ? wall[direct.wall].nextsector : -1;
            char classification[192];
            snprintf(classification, sizeof(classification),
                     "wall=%d mode=%s nextsector=%d stop=(%d,%d)", direct.wall,
                     navEdgeModeName(blockMode), nextSector, direct.x, direct.y);
            event("navigation_block_classified", classification);
            if (blockMode == kNavWalk || blockMode == kNavStep || blockMode == kNavDrop)
            {
                char detail[96];
                snprintf(detail, sizeof(detail), "wall=%d mode=%s", direct.wall,
                         navEdgeModeName(blockMode));
                event("nav_edge", detail);
                return steerTo(x, y, false, shoot, targetId, targetSector, capability);
            }
            if (blockMode == kNavJump)
            {
                event("nav_edge", "mode=JUMP");
                return steerTo(x, y, false, shoot, targetId, targetSector,
                               kTraversalJumpable);
            }
            bool actionable = false;
            for (const InteractionCandidate &candidate : observation.interactions)
            {
                if (candidate.kind == kInteractionWall && candidate.id == direct.wall)
                {
                    actionable = true;
                    break;
                }
            }
            char detail[160];
            snprintf(detail, sizeof(detail), "wall=%d target=%d actionable=%d stop=(%d,%d)",
                     direct.wall, targetId, actionable ? 1 : 0, direct.x, direct.y);
            event("navigation_blocked", detail);
            if (rememberCollisionInteraction(direct.wall))
                event("navigation_interaction_opportunity", "source=clipmove_collision_hit");
            ensureNavTopology();
            if (buildNavRoute(x, y, targetSector, signature))
            {
                const LocalWaypoint &waypoint = navRoute.front();
                return steerTo(waypoint.x, waypoint.y, false, false, targetId,
                               observation.sector, capability);
            }
            if (chooseNavigationDetour(direct))
                return steerTo(navigationWaypointX, navigationWaypointY, false, false,
                               targetId, observation.sector, capability);
        }
        if (navigationFailureSignature != signature)
        {
            navigationFailureSignature = signature;
            navigationFailureCount = 0;
        }
        ++navigationFailureCount;
        if (navigationFailureCount == 1)
            event("navigation_failed", "reason=no_bounded_collision_safe_detour");
        if (navigationFailureCount >= 12 && currentObjective.active
            && currentObjective.type != kObjectiveInteraction)
        {
            if (currentObjective.type == kObjectiveFrontier)
            {
                const int edgeId = currentObjective.wall * 65536
                    + currentObjective.targetSector;
                const Portal *failedPortal = findPortalForEdge(edgeId);
                if (failedPortal)
                {
                    recordEdgeFailure(*failedPortal, "local_portal_failed",
                                      "navigation_failed_bounded");
                    retirePendingFrontier(currentObjective.wall, currentObjective.sector,
                                          currentObjective.targetSector);
                }
            }
            event("navigation_failed_bounded", "reason=objective_temporarily_unreachable");
            invalidateObjective("navigation_failed_bounded");
        }
        return GINPUT{};
    }

    void preparePortalWaypoint(const Portal &portal)
    {
        portalWaypointWall = portal.wall;
        portalPlanWall = portal.wall;
        portalPlanSignature = portalGeometrySignature(portal);
        portalPlan.clear();
        portalPlanIndex = 0;
        portalWaypointActive = false;
        portalCrossingActive = false;
        portalCrossingDestinationValid = false;
        portalCrossingStartTick = -1;
        portalWaypointX = portal.x;
        portalWaypointY = portal.y;

        const int dx = portal.x2 - portal.x1;
        const int dy = portal.y2 - portal.y1;
        const int length = int(std::sqrt(double(int64_t(dx) * dx + int64_t(dy) * dy)));
        const int radius = gMe && gMe->pSprite ? (gMe->pSprite->clipdist << 2) : 128;
        const int margin = std::max(192, radius + 96);
        if (length <= margin * 2)
            return;

        // The crossing interval is the wall segment after removing one
        // player radius at either end.  Test several points, not just the
        // midpoint, because Build corners commonly block that line.
        const double tangentX = double(dx) / length;
        const double tangentY = double(dy) / length;
        const int safeHalf = length / 2 - margin;
        std::vector<LocalWaypoint> crossings;
        // Prefer the center of the receiving opening.  End samples are still
        // useful around ordinary corners, but slide-door mechanisms often
        // place a second blocking wall immediately beyond one endpoint.
        const int offsets[] = { 0, -safeHalf / 2, safeHalf / 2, -safeHalf, safeHalf };
        for (int offset : offsets)
        {
            LocalWaypoint candidate;
            candidate.x = portal.x + int(std::lround(tangentX * offset));
            candidate.y = portal.y + int(std::lround(tangentY * offset));
            crossings.push_back(candidate);
        }

        const int startTolerance = std::max(192, radius + 128);
        // Enter only a small body-clearance beyond the portal.  A deeper
        // target can land on the first internal sliding-wall segment of a
        // mechanism room even when the portal itself is clear.  The next
        // observation can then discover and use that room's mechanism.
        const int crossingDepth = std::max(radius + 64, 256);
        auto crossingDestinations = [&](const LocalWaypoint &crossing) {
            const int normalX = -(portal.y2 - portal.y1);
            const int normalY = portal.x2 - portal.x1;
            const int normalLength = std::max(1, int(std::sqrt(double(
                int64_t(normalX) * normalX + int64_t(normalY) * normalY))));
            std::vector<LocalWaypoint> destinations;
            destinations.push_back({
                crossing.x + normalX * crossingDepth / normalLength,
                crossing.y + normalY * crossingDepth / normalLength
            });
            destinations.push_back({
                crossing.x - normalX * crossingDepth / normalLength,
                crossing.y - normalY * crossingDepth / normalLength
            });
            return destinations;
        };
        for (const LocalWaypoint &crossing : crossings)
        {
            for (const LocalWaypoint &destination : crossingDestinations(crossing))
            {
                if (collisionProbe(observation.x, observation.y, observation.z, observation.sector,
                                   destination.x, destination.y, portal.to, startTolerance))
                {
                    portalPlan.push_back(crossing);
                    portalWaypointX = crossing.x;
                    portalWaypointY = crossing.y;
                    portalCrossingDestinationX = destination.x;
                    portalCrossingDestinationY = destination.y;
                    portalCrossingDestinationValid = true;
                    portalWaypointActive = true;
                    event("local_plan_created", "mode=direct collision_probe=clipmove nodes=1");
                    return;
                }
            }
        }

        // Construct a deliberately small local visibility graph.  Candidate
        // nodes are player-sized offsets around nearby wall corners.  Every
        // edge is validated by copied-state clipmove; cansee() is never used
        // as a body-clearance proof.
        std::vector<LocalWaypoint> nodes;
        nodes.push_back({ observation.x, observation.y });
        for (int i = 0; i < sector[observation.sector].wallnum; ++i)
        {
            const walltype &wallRecord = wall[sector[observation.sector].wallptr + i];
            const walltype &nextWall = wall[wallRecord.point2];
            const int corners[2][2] = {{ wallRecord.x, wallRecord.y }, { nextWall.x, nextWall.y }};
            for (const auto &corner : corners)
            {
                const int dirs[][2] = {{ 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 },
                                        { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 }};
                for (const auto &dir : dirs)
                {
                    const int norm = (dir[0] && dir[1]) ? 181 : 256;
                    LocalWaypoint node;
                    node.x = corner[0] + dir[0] * margin * 256 / norm;
                    node.y = corner[1] + dir[1] * margin * 256 / norm;
                    if (collisionProbe(observation.x, observation.y, observation.z, observation.sector,
                                       node.x, node.y, observation.sector, startTolerance))
                        nodes.push_back(node);
                    if (nodes.size() >= 25)
                        break;
                }
                if (nodes.size() >= 25)
                    break;
            }
            if (nodes.size() >= 25)
                break;
        }

        struct SearchNode { LocalWaypoint point; int parent = -1; };
        std::vector<SearchNode> search;
        search.push_back({ nodes.front(), -1 });
        std::deque<int> queue;
        queue.push_back(0);
        std::vector<bool> used(nodes.size(), false);
        used[0] = true;
        int goal = -1;
        while (!queue.empty() && goal < 0)
        {
            const int current = queue.front();
            queue.pop_front();
            const LocalWaypoint &from = search[current].point;
            for (size_t i = 1; i < nodes.size(); ++i)
            {
                if (used[i])
                    continue;
                if (!collisionProbe(from.x, from.y, observation.z, observation.sector,
                                     nodes[i].x, nodes[i].y, observation.sector, startTolerance))
                    continue;
                used[i] = true;
                search.push_back({ nodes[i], current });
                queue.push_back(int(search.size() - 1));
                for (const LocalWaypoint &crossing : crossings)
                {
                    for (const LocalWaypoint &destination : crossingDestinations(crossing))
                    {
                        if (collisionProbe(nodes[i].x, nodes[i].y, observation.z, observation.sector,
                                           destination.x, destination.y, portal.to, startTolerance))
                        {
                            search.push_back({ crossing, int(search.size() - 1) });
                            portalCrossingDestinationX = destination.x;
                            portalCrossingDestinationY = destination.y;
                            portalCrossingDestinationValid = true;
                            goal = int(search.size() - 1);
                            break;
                        }
                    }
                    if (goal >= 0)
                        break;
                }
                if (goal >= 0)
                    break;
            }
        }
        if (goal < 0)
        {
            // Some Build door rooms deliberately overlap their entrance
            // geometry, so copied clipmove can reject every straight probe
            // even though the receiving sector contains a valid central
            // point.  Keep this as a bounded last resort: topology still
            // comes from inside(), and the real player tick remains the
            // authority on whether the crossing succeeds.
            const int normalX = -(portal.y2 - portal.y1);
            const int normalY = portal.x2 - portal.x1;
            const int normalLength = std::max(1, int(std::sqrt(double(
                int64_t(normalX) * normalX + int64_t(normalY) * normalY))));
            for (int direction : { 1, -1 })
            {
                const int destinationX = portal.x + direction * normalX * crossingDepth / normalLength;
                const int destinationY = portal.y + direction * normalY * crossingDepth / normalLength;
                if (inRange(portal.to, 0, numsectors)
                    && inside(destinationX, destinationY, portal.to) == 1)
                {
                    portalWaypointX = portal.x;
                    portalWaypointY = portal.y;
                    portalCrossingDestinationX = destinationX;
                    portalCrossingDestinationY = destinationY;
                    portalCrossingDestinationValid = true;
                    event("local_plan_fallback", "reason=receiving_sector_inside central_crossing=1");
                    break;
                }
            }
            event("local_plan_failed", "reason=no_collision_safe_path alternatives=5 corner_nodes=24");
            return;
        }
        std::vector<LocalWaypoint> reverse;
        for (int cursor = goal; cursor >= 0; cursor = search[cursor].parent)
            reverse.push_back(search[cursor].point);
        for (auto it = reverse.rbegin(); it != reverse.rend(); ++it)
            if (portalPlan.empty() || it->x != portalPlan.back().x || it->y != portalPlan.back().y)
                portalPlan.push_back(*it);
        if (!portalPlan.empty())
        {
            // The first node is the copied current pose, not a movement goal.
            portalPlanIndex = portalPlan.size() > 1 ? 1 : 0;
            portalWaypointX = portalPlan.back().x;
            portalWaypointY = portalPlan.back().y;
            portalWaypointActive = true;
        }
        char detail[128];
        snprintf(detail, sizeof(detail), "mode=corner_graph collision_probe=clipmove nodes=%u waypoints=%u",
                 unsigned(nodes.size()), unsigned(portalPlan.size()));
        event("local_plan_created", detail);
    }

    GINPUT steerPortal(const Portal &portal)
    {
        const int bodyRadius = gMe && gMe->pSprite ? (gMe->pSprite->clipdist << 2) : 128;
        const int portalApproachTolerance = std::max(1024, bodyRadius + 1024);
        // Portal approach is now a specialization of the same local
        // topology used by arbitrary targets.  The old corner planner remains
        // below as a dynamic-geometry fallback, but do not ask it to steer a
        // straight line through a known same-sector NavCell route.
        if (!portalCrossingActive && portal.from == observation.sector
            && distance2(observation.x, observation.y, portal.x, portal.y)
                > portalApproachTolerance * portalApproachTolerance)
        {
            const MovementProbe sideProbe = probeMovement(
                observation.x, observation.y, observation.z, observation.sector,
                portal.x, portal.y, observation.sector, std::max(256, bodyRadius));
            if (!sideProbe.reachable)
            {
                const int routeSignature = navigationSignature(portal.x, portal.y,
                                                                portal.z, observation.sector,
                                                                portal.wall);
                if (navRouteSignature != routeSignature
                    || navRouteTopologyRevision != navTopologyRevision
                    || navRoute.empty())
                    buildNavRoute(portal.x, portal.y, observation.sector, routeSignature);
                if (!navRoute.empty() && navRouteSignature == routeSignature)
                    return navigateTo(portal.x, portal.y, portal.z, observation.sector,
                                      portal.wall, portal.capability);
            }
        }
        const int signature = portalGeometrySignature(portal);
        if (portalPlanWall != portal.wall || portalPlanSignature != signature)
        {
            portalPlanAttempts = 0;
            preparePortalWaypoint(portal);
        }

        const int radius = gMe && gMe->pSprite ? (gMe->pSprite->clipdist << 2) : 128;
        const int waypointTolerance = std::max(512, radius * 2);
        if (portalWaypointActive && portalPlanIndex < portalPlan.size()
            && distance2(observation.x, observation.y, portalPlan[portalPlanIndex].x,
                         portalPlan[portalPlanIndex].y) <= waypointTolerance * waypointTolerance)
        {
            ++portalPlanIndex;
            if (portalPlanIndex >= portalPlan.size())
                portalWaypointActive = false;
        }
        int targetX = portalWaypointActive ? portalPlan[portalPlanIndex].x : portal.x;
        int targetY = portalWaypointActive ? portalPlan[portalPlanIndex].y : portal.y;
        const bool intermediate = portalWaypointActive && portalPlanIndex + 1 < portalPlan.size();
        const bool crouch = portal.capability == kTraversalCrouchable;
        if (crouch && !crouchTargetActive)
        {
            event("crouch_started", "reason=low_portal");
            crouchTargetActive = true;
        }

        // A bad local plan gets bounded alternatives before the edge is
        // recorded as failed.  The edge itself remains eligible if its
        // geometry signature changes later.
        const bool crossingStuck = portalCrossingActive && portalCrossingStartTick >= 0
            && observation.tick - portalCrossingStartTick >= kMovementStuckTicks;
        if ((movementTargetActive && movementTargetGoal == currentGoal
             && movementTargetId == portal.wall
             && observation.tick - targetLastProgressTick >= kMovementStuckTicks)
            || crossingStuck)
        {
            if (portalPlanAttempts < 3)
            {
                ++portalPlanAttempts;
                char detail[96];
                snprintf(detail, sizeof(detail), "wall=%d attempt=%d reason=waypoint_stuck", portal.wall, portalPlanAttempts);
                event("local_plan_retry", detail);
                portalPlanWall = -1;
                movementTargetActive = false;
                preparePortalWaypoint(portal);
            }
            else
            {
                const int edgeId = portal.wall * 65536 + portal.to;
                if (portal.capability == kTraversalWalkable && portal.floorDelta < 0
                    && -portal.floorDelta <= portal.jumpRiseLimit
                    && jumpFallbackEdges.insert(edgeId).second)
                {
                    event("local_jump_fallback", "reason=walkable_collision_approaches_exhausted");
                    portalPlanWall = -1;
                    movementTargetActive = false;
                    return steerPortal(portal);
                }
                recordEdgeFailure(portal, "local_portal_failed", "bounded_collision_safe_approaches_exhausted");
                movementTargetActive = false;
            }
        }
        const int targetSector = intermediate ? observation.sector : portal.to;
        if (!intermediate && !portalWaypointActive
            && distance2(observation.x, observation.y, portal.x, portal.y)
                <= (radius + 1024) * (radius + 1024))
        {
            // Once the player-sized probe has reached the safe crossing
            // interval, aim through the portal rather than at the wall line.
            // This keeps forward motion normal to the actual crossing and
            // avoids oscillating on a corner when the midpoint is exactly on
            // the collision plane.
            if (!portalCrossingActive)
            {
                if (portalCrossingDestinationValid)
                {
                    portalCrossingX = portalCrossingDestinationX;
                    portalCrossingY = portalCrossingDestinationY;
                }
                else
                {
                    const int dxToPortal = portalWaypointX - observation.x;
                    const int dyToPortal = portalWaypointY - observation.y;
                    const int distance = std::max(1, int(std::sqrt(double(distance2(observation.x, observation.y,
                                                                                    portalWaypointX, portalWaypointY)))));
                    // The crossing target must be just inside the receiving
                    // sector.  Sending the player deep into a compact
                    // mechanism room can hit its first internal wall before
                    // the bot has had a chance to operate it.
                    const int crossingDepth = std::max(radius + 64, 256);
                    portalCrossingX = portalWaypointX + dxToPortal * crossingDepth / distance;
                    portalCrossingY = portalWaypointY + dyToPortal * crossingDepth / distance;
                }
                portalCrossingActive = true;
                portalCrossingStartTick = observation.tick;
                char detail[96];
                snprintf(detail, sizeof(detail), "wall=%d x=%d y=%d", portal.wall, portalCrossingX, portalCrossingY);
                event("portal_crossing_push", detail);
            }
            targetX = portalCrossingX;
            targetY = portalCrossingY;
        }
        // The final push through a portal is a specialization of the generic
        // navigator.  Once the safe crossing point has been selected, retain
        // the proven shallow Build movement semantics: a moving-door portal
        // can still report its own slide-wall as a clipmove obstacle while
        // the real player tick is already able to consume the opening.
        if (portalCrossingActive)
            return steerTo(targetX, targetY, false, false, portal.wall, targetSector,
                           portal.capability);
        return navigateTo(targetX, targetY, portal.z, targetSector, portal.wall, portal.capability);
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
            const Portal *portal = findPortalForEdge(movementTargetId * 65536 + movementTargetSector);
            if (portal)
                recordEdgeFailure(*portal, "frontier_failed", "jump_alternatives_exhausted");
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

        if (currentObjective.active)
        {
            if (urgentThreat() && currentObjective.type != kObjectiveCombat)
            {
                suspendedObjective = currentObjective;
                suspendedObjectiveActive = true;
                currentObjective = Objective{};
                event("objective_suspended", "reason=urgent_threat");
            }
            else
                return executeObjective();
        }
        if (suspendedObjectiveActive && !urgentThreat())
        {
            currentObjective = suspendedObjective;
            suspendedObjective = Objective{};
            suspendedObjectiveActive = false;
            event("objective_resumed");
            return executeObjective();
        }

        if (urgentThreat())
        {
            if (const VisibleObject *enemy = selectObject(kObjectEnemy))
            {
                Objective objective;
                objective.type = kObjectiveCombat;
                objective.id = enemy->sprite;
                objective.sector = enemy->sector;
                objective.targetSector = enemy->sector;
                selectObjective(objective, "COMBAT_THREAT");
                return executeObjective();
            }
        }

        if (const VisibleObject *key = selectLocalAreaObject(kObjectKey))
        {
            Objective objective;
            objective.type = kObjectiveKey;
            objective.id = key->sprite;
            objective.sector = key->sector;
            objective.targetSector = key->sector;
            selectObjective(objective, "COLLECT_LOCAL_KEY");
            return executeObjective();
        }
        if (const VisibleObject *pickup = selectLocalAreaObject(kObjectPickup))
        {
            Objective objective;
            objective.type = kObjectivePickup;
            objective.id = pickup->sprite;
            objective.sector = pickup->sector;
            objective.targetSector = pickup->sector;
            selectObjective(objective, "COLLECT_LOCAL_PICKUP");
            return executeObjective();
        }
        if (observation.exitHere)
        {
            setGoal("USE_OBSERVED_EXIT", observation.sector);
            GINPUT input = {};
            input.keyFlags.action = 1;
            return input;
        }
        if (InteractionMemory *interaction = selectNewInteraction(false))
        {
            Objective objective;
            objective.type = kObjectiveInteraction;
            objective.id = interaction->id;
            objective.sector = interaction->fromSector;
            objective.targetSector = interaction->targetSector;
            objective.x = interaction->x;
            objective.y = interaction->y;
            objective.z = interaction->z;
            objective.interactionKey = interactionMemoryKey(*interaction);
            selectObjective(objective, interaction->kind == kInteractionSprite
                ? "USE_NEW_SPRITE_INTERACTION"
                : interaction->kind == kInteractionSector
                ? "USE_NEW_SECTOR_INTERACTION" : "USE_NEW_WALL_INTERACTION");
            return executeObjective();
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
            Objective objective;
            objective.type = kObjectiveFrontier;
            objective.id = localPortal->wall;
            objective.sector = localPortal->from;
            objective.targetSector = localPortal->to;
            objective.wall = localPortal->wall;
            selectObjective(objective, "EXPLORE_LOCAL_PORTAL");
            return executeObjective();
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
            Objective objective;
            objective.type = kObjectiveFrontier;
            objective.id = portal->wall;
            objective.sector = portal->from;
            objective.targetSector = portal->to;
            objective.wall = portal->wall;
            selectObjective(objective, "EXPLORE_FRONTIER");
            return executeObjective();
        }
        if (PendingWork *pending = selectRemoteFrontier())
        {
            Objective objective;
            objective.type = kObjectiveFrontier;
            objective.id = pending->id;
            objective.sector = pending->sector;
            objective.targetSector = pending->edgeTarget;
            objective.wall = pending->id;
            selectObjective(objective, "EXPLORE_REMOTE_FRONTIER");
            return executeObjective();
        }

        // Exhausting current frontiers is a recovery transition, not a
        // terminal condition.  Reconsider one previously activated,
        // reversible interaction at a time before the bounded area search.
        if (!recoveryMode)
        {
            recoveryMode = true;
            event("exploration_recovery", "reason=no_frontier");
        }
        if (InteractionMemory *interaction = selectNewInteraction(true))
            return steerInteraction(*interaction);

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
    if (gMe->pSprite->extra > 0 && gMe->pSprite->extra < kMaxXSprites)
    {
        const int moveHit = gSpriteHit[gMe->pSprite->extra].hit;
        const int wallHit = (moveHit & 0xc000) == 0x8000 ? moveHit & 0x3fff : -1;
        if (wallHit != m_impl->lastEngineMoveHit)
        {
            m_impl->lastEngineMoveHit = wallHit;
            if (wallHit >= 0)
            {
                char detail[160];
                snprintf(detail, sizeof(detail), "wall=%d sector=%d x=%d y=%d hit=0x%x goal=%s target=%d",
                         wallHit, m_impl->observation.sector, int(gMe->pSprite->x), int(gMe->pSprite->y),
                         moveHit, m_impl->currentGoal.c_str(), m_impl->lastStateTarget);
                m_impl->event("engine_move_wall_hit", detail);
            }
        }
    }
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
