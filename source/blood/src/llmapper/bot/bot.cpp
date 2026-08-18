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
#include "db.h"
#include "demo.h"
#include "gameutil.h"
#include "globals.h"
#include "levels.h"
#include "network.h"
#include "player.h"
#include "triggers.h"

namespace
{
constexpr int kDefaultTimeoutSeconds = 30 * 60;
constexpr int kDefaultStallSeconds = 45;
constexpr int kObservationPeriod = 4;
constexpr int kTrajectoryPeriod = 8;

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
    int key = 0;
    bool visible = false;
    bool traversable = false;
    bool locked = false;
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
        if (current.extra > 0 && xsector[current.extra].Exit)
            result.exitHere = true;

        for (int i = 0; i < current.wallnum; ++i)
        {
            const int wallIndex = current.wallptr + i;
            const walltype &wallRecord = wall[wallIndex];
            if (!inRange(wallRecord.nextsector, 0, numsectors))
                continue;

            const walltype &nextWall = wall[wallRecord.point2];
            const int midX = (wallRecord.x + nextWall.x) / 2;
            const int midY = (wallRecord.y + nextWall.y) / 2;
            const int midZ = getflorzofslope(result.sector, midX, midY);
            const bool visible = cansee(result.x, result.y, result.z, result.sector,
                                         midX, midY, midZ, wallRecord.nextsector);

            Portal portal;
            portal.wall = wallIndex;
            portal.from = result.sector;
            portal.to = wallRecord.nextsector;
            portal.x = midX;
            portal.y = midY;
            portal.z = midZ;
            portal.visible = visible;
            portal.traversable = visible && !(wallRecord.cstat & 1);
            portal.locked = false;
            if (sector[portal.to].extra > 0)
            {
                const XSECTOR &extra = xsector[sector[portal.to].extra];
                portal.key = extra.Key;
                portal.locked = extra.locked != 0;
                if (extra.damageType != 0 || sector[portal.to].type == kSectorDamage)
                    portal.traversable = false;
            }

            if (!visible)
                continue;
            appendUnique(result.visibleSectors, portal.to);
            result.portals.push_back(portal);
        }
    }

    // The scan is over the engine's sprite list, but only objects passing the
    // authoritative cansee() test enter the bot's observation.
    for (int i = 0; i < kMaxSprites; ++i)
    {
        spritetype &candidate = sprite[i];
        if (candidate.statnum != kStatDude && candidate.statnum != kStatThing)
            continue;
        if (candidate.index == player->index || candidate.sectnum < 0 || candidate.extra <= 0)
            continue;
        if (candidate.statnum == kStatDude && (!isEnemyType(candidate.type) || xsprite[candidate.extra].health == 0))
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
        object.kind = candidate.statnum == kStatDude ? kObjectEnemy
                         : isKeyType(candidate.type) ? kObjectKey
                         : (xsprite[candidate.extra].Push || xsprite[candidate.extra].Vector) ? kObjectInteractive
                                                                                              : kObjectPickup;
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
        int id = -1;
        Portal portal;
        int attempts = 0;
        bool blocked = false;
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
    int lastProgressTick = 0;
    int lastSector = -1;
    int lastGoal = -1;
    int repeatedStateCount = 0;
    int lastStateTick = -1;
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
    std::set<int> seenEdges;
    std::set<int> visitedEdges;
    std::map<int, std::vector<Portal>> knownGraph;
    Portal routePortal;
    int selectedDoorId = -1;

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

    void trajectorySample()
    {
        if (!trajectory || !gMe || !gMe->pSprite)
            return;
        fprintf(trajectory, "{\"game_time\":%d,\"x\":%d,\"y\":%d,\"z\":%d,\"sector\":%d}\n",
                (gFrame * kTicsPerFrame) / kTicsPerSec,
                int(gMe->pSprite->x), int(gMe->pSprite->y), int(gMe->pSprite->z), int(gMe->pSprite->sectnum));
        fflush(trajectory);
    }

    void updateKnowledge()
    {
        observation = observeWorld();
        if (observation.sector < 0)
            return;
        const size_t oldSectors = observedSectors.size();
        observedSectors.insert(observation.visibleSectors.begin(), observation.visibleSectors.end());
        visitedSectors.insert(observation.sector);
        if (observedSectors.size() != oldSectors)
        {
            lastProgressTick = observation.tick;
            event("discovered_sector");
        }

        for (const Portal &portal : observation.portals)
        {
            const int edgeId = portal.wall * 65536 + portal.to;
            seenEdges.insert(edgeId);
            if (portal.traversable)
                knownGraph[portal.from].push_back(portal);
            if (portal.key || portal.locked || !portal.traversable)
            {
                DoorMemory &door = doors[portal.wall];
                if (door.id < 0)
                {
                    door.id = portal.wall;
                    door.portal = portal;
                    char detail[96];
                    snprintf(detail, sizeof(detail), "door=%d key=%d locked=%d", portal.wall, portal.key, portal.locked ? 1 : 0);
                    event("discovered_door", detail);
                }
                else
                    door.portal = portal;
            }
        }

        for (const VisibleObject &object : observation.objects)
        {
            if (object.kind == kObjectKey)
                knownKeys.insert(object.type - kItemKeyBase + 1);
            if (knownObjects.insert(object.sprite).second)
            {
                char detail[128];
                snprintf(detail, sizeof(detail), "sprite=%d kind=%s sector=%d type=%d",
                         object.sprite, objectName(object.kind), object.sector, object.type);
                event("observed_object", detail);
                lastProgressTick = observation.tick;
            }
        }

        for (int key = 1; key < 8; ++key)
        {
            if (gMe->hasKey[key] && knownKeys.insert(key).second)
            {
                char detail[64];
                snprintf(detail, sizeof(detail), "key=%d", key);
                event("acquired_key", detail);
                lastProgressTick = observation.tick;
            }
        }

        if (pendingUse >= 0 && pendingUseTick < observation.tick)
        {
            auto door = doors.find(pendingUse);
            if (door != doors.end() && door->second.portal.key && !hasKey(door->second.portal.key))
            {
                door->second.blocked = true;
                char detail[96];
                snprintf(detail, sizeof(detail), "door=%d key=%d", pendingUse, door->second.portal.key);
                event("blocked_key_required", detail);
            }
            pendingUse = -1;
        }
    }

    bool hasKey(int key) const
    {
        return key <= 0 || (gMe && gMe->hasKey[key]);
    }

    const Portal *selectPortal()
    {
        const Portal *best = nullptr;
        int bestDistance = INT32_MAX;
        for (const Portal &portal : observation.portals)
        {
            if (!portal.traversable || (portal.key && !hasKey(portal.key)))
                continue;
            const int edgeId = portal.wall * 65536 + portal.to;
            const int score = visitedEdges.count(edgeId) ? 100000000 : distance2(observation.x, observation.y, portal.x, portal.y);
            if (score < bestDistance)
            {
                best = &portal;
                bestDistance = score;
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

    const Portal *selectKnownDoor()
    {
        const Portal *best = nullptr;
        selectedDoorId = -1;
        int bestDistance = INT32_MAX;
        for (auto &entry : doors)
        {
            DoorMemory &door = entry.second;
            if (door.blocked || (door.portal.key && !hasKey(door.portal.key) && door.attempts > 0))
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

            routePortal = door.portal;
            int cursor = door.portal.from;
            while (cursor != observation.sector)
            {
                const Portal edge = parent[cursor];
                routePortal = edge;
                cursor = edge.from;
            }
            const int distance = distance2(observation.x, observation.y, routePortal.x, routePortal.y);
            if (distance < bestDistance)
            {
                best = &routePortal;
                bestDistance = distance;
                selectedDoorId = door.id;
            }
        }
        return best;
    }

    void setGoal(const char *goal)
    {
        if (currentGoal == goal)
            return;
        currentGoal = goal;
        char detail[160];
        snprintf(detail, sizeof(detail), "goal=%s", goal);
        event("goal_changed", detail);
    }

    GINPUT steerTo(int x, int y, bool use, bool shoot)
    {
        GINPUT input = {};
        const int targetAngle = getangle(x - observation.x, y - observation.y);
        const int delta = angleDelta(targetAngle, observation.angle);
        input.syncFlags.run = 1;
        input.q16turn = fix16_from_int(std::max(-96, std::min(96, delta)));
        if (std::abs(delta) < 96)
            input.forward = 2047;
        if (use && distance2(observation.x, observation.y, x, y) < 90000)
            input.keyFlags.action = 1;
        if (shoot)
            input.buttonFlags.shoot = 1;
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
            setGoal("COMBAT_VISIBLE_ENEMY");
            return steerTo(enemy->x, enemy->y, false, true);
        }
        if (const VisibleObject *key = selectObject(kObjectKey))
        {
            setGoal("COLLECT_VISIBLE_KEY");
            return steerTo(key->x, key->y, false, false);
        }
        if (const VisibleObject *interactive = selectObject(kObjectInteractive))
        {
            setGoal("USE_VISIBLE_INTERACTIVE");
            return steerTo(interactive->x, interactive->y, true, false);
        }
        if (const VisibleObject *pickup = selectObject(kObjectPickup))
        {
            setGoal("COLLECT_VISIBLE_PICKUP");
            return steerTo(pickup->x, pickup->y, false, false);
        }
        if (observation.exitHere)
        {
            setGoal("USE_OBSERVED_EXIT");
            GINPUT input = {};
            input.keyFlags.action = 1;
            return input;
        }
        if (const Portal *door = selectKnownDoor())
        {
            const bool atDoor = selectedDoorId == door->wall;
            const bool missingKey = atDoor && door->key && !hasKey(door->key);
            setGoal(missingKey ? "TRY_LOCKED_DOOR" : "REVISIT_LOCKED_DOOR");
            if (atDoor)
            {
                pendingUse = door->wall;
                pendingUseTick = observation.tick;
                ++doors[door->wall].attempts;
            }
            return steerTo(door->x, door->y, atDoor, false);
        }
        if (const Portal *portal = selectPortal())
        {
            setGoal("EXPLORE_FRONTIER");
            visitedEdges.insert(portal->wall * 65536 + portal->to);
            return steerTo(portal->x, portal->y, false, false);
        }

        // A short deterministic turn-and-walk probe resolves frontiers that
        // were not visible from the previous sample without consulting the
        // loaded map's complete topology.
        setGoal("SEARCH_CURRENT_AREA");
        idle.syncFlags.run = 1;
        idle.forward = 1024;
        idle.q16turn = fix16_from_int(24);
        return idle;
    }

    void detectStall()
    {
        if (observation.tick - lastProgressTick > stallSeconds * kTicsPerSec)
        {
            result = "STALLED";
            failureReason = "no meaningful world or knowledge progress";
            event("failure", failureReason.c_str());
            gQuitGame = true;
            return;
        }

        const int quantized = (observation.x >> 10) ^ ((observation.y >> 10) << 10) ^ (observation.sector << 20);
        if (quantized == lastSector && lastGoal == int(currentGoal.size()))
            ++repeatedStateCount;
        else
            repeatedStateCount = 0;
        lastSector = quantized;
        lastGoal = int(currentGoal.size());
        if (repeatedStateCount > 3 * kTicsPerSec)
        {
            result = "LOOP_DETECTED";
            failureReason = "repeated position/goal state";
            event("failure", failureReason.c_str());
            gQuitGame = true;
        }
    }

    void close(const char *reason)
    {
        if (!result.size())
            result = reason ? reason : "RUNTIME_ERROR";
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
    : m_impl(new Impl), m_enabled(false), m_fast(true)
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
