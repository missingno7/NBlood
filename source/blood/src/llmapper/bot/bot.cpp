//-------------------------------------------------------------------------
// Minimal autonomous LLMapper Blood bot.
//
// The active architecture is intentionally one loop:
//   observe -> choose nearest physically proven unresolved fact
//   -> reach it -> resolve it -> observe again.
//
// Build sectors/walls generate candidates. Blood's own cansee, ClipMove and
// ActionScanPreview provide positive physical evidence. A failed local
// search is reported as unknown; it is never stored as world impossibility.
//-------------------------------------------------------------------------
#include "bot.h"
#include "bot_input.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "build.h"
#include "../../actor.h"
#include "../../blood.h"
#include "../../common_game.h"
#include "../../db.h"
#include "../../demo.h"
#include "../../gameutil.h"
#include "../../globals.h"
#include "../../levels.h"
#include "../../player.h"
#include "../../trig.h"
#include "../../view.h"

namespace {

constexpr int kUseRange = 1024;
constexpr int kWaypointTolerance = 160;
constexpr int kUsePoseTolerance = 320;
constexpr int kMaximumLocalCandidates = 512;
int gPhysicalProbeCount = 0;

bool validSector(int sectorId)
{
    return sectorId >= 0 && sectorId < numsectors;
}

bool validWall(int wallId)
{
    return wallId >= 0 && wallId < numwalls;
}

bool validSprite(int spriteId)
{
    return spriteId >= 0 && spriteId < kMaxSprites
        && sprite[spriteId].statnum != kStatFree;
}

int64_t distanceSquared(int x1, int y1, int x2, int y2)
{
    const int64_t dx = int64_t(x2) - x1;
    const int64_t dy = int64_t(y2) - y1;
    return dx * dx + dy * dy;
}

int distanceTo(int x1, int y1, int x2, int y2)
{
    return int(std::sqrt(double(distanceSquared(x1, y1, x2, y2))));
}

int clampInt(int value, int low, int high)
{
    return std::max(low, std::min(high, value));
}

uint64_t mixHash(uint64_t value, uint64_t item)
{
    value ^= item + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
    return value;
}

struct BodyShape
{
    int radius = 128;
    int ceilingDistance = 0;
    int floorDistance = 0;
    int footOffset = 0;
    int height = 0;
};

BodyShape liveBodyShape()
{
    BodyShape result;
    if (!gMe || !gMe->pSprite)
        return result;
    int top = 0;
    int bottom = 0;
    GetSpriteExtents(gMe->pSprite, &top, &bottom);
    result.radius = gMe->pSprite->clipdist << 2;
    result.ceilingDistance = std::max(0, (gMe->pSprite->z - top) / 4);
    result.floorDistance = std::max(0, (bottom - gMe->pSprite->z) / 4);
    result.footOffset = bottom - gMe->pSprite->z;
    result.height = bottom - top;
    return result;
}

struct Pose
{
    int x = 0;
    int y = 0;
    int z = 0;
    int sector = -1;

    bool samePlace(const Pose &other, int tolerance = kWaypointTolerance) const
    {
        return sector == other.sector
            && distanceSquared(x, y, other.x, other.y)
                <= int64_t(tolerance) * tolerance;
    }
};

struct LineProbe
{
    bool reached = false;
    Pose end;
    int hit = 0;
};

bool engineLineHasContinuousSupport(const Pose &from, const Pose &to,
                                    const BodyShape &body)
{
    ++gPhysicalProbeCount;
    if (!gMe || !gMe->pSprite || !validSector(from.sector))
        return false;
    const int length = distanceTo(from.x, from.y, to.x, to.y);
    const int samples = std::max(1,
        (length + std::max(1, body.radius) - 1) / std::max(1, body.radius));
    int16_t sectorId = int16_t(from.sector);
    int supportZ = from.z + body.footOffset;
    spritetype *actor = gMe->pSprite;
    const int savedCstat = actor->cstat;
    actor->cstat &= ~257;
    for (int step = 1; step <= samples; ++step)
    {
        const int x = from.x + int(int64_t(to.x - from.x) * step / samples);
        const int y = from.y + int(int64_t(to.y - from.y) * step / samples);
        updatesectorz(x, y, supportZ - body.footOffset, &sectorId);
        if (!validSector(sectorId))
        {
            actor->cstat = int16_t(savedCstat);
            return false;
        }
        int ceilingZ = 0;
        int ceilingHit = 0;
        int floorZ = 0;
        int floorHit = 0;
        GetZRangeAtXYZ(x, y, supportZ - body.footOffset, sectorId,
            &ceilingZ, &ceilingHit, &floorZ, &floorHit,
            body.radius + 16, CLIPMASK0,
            PARALLAXCLIP_CEILING | PARALLAXCLIP_FLOOR);
        if (floorZ - ceilingZ
                < body.ceilingDistance + body.floorDistance
            || std::abs(floorZ - supportZ) > 1024)
        {
            actor->cstat = int16_t(savedCstat);
            return false;
        }
        supportZ = floorZ;
    }
    actor->cstat = int16_t(savedCstat);
    return sectorId == to.sector || inside(to.x, to.y, sectorId) == 1;
}

LineProbe engineWalkLine(const Pose &from, const Pose &to, int tolerance)
{
    ++gPhysicalProbeCount;
    LineProbe result;
    result.end = from;
    if (!gMe || !gMe->pSprite || !validSector(from.sector)
        || !validSector(to.sector))
        return result;

    const BodyShape body = liveBodyShape();
    spritetype *actor = gMe->pSprite;
    const int savedCstat = actor->cstat;
    actor->cstat &= ~257; // MoveDude removes the live actor from its own clip set.

    int x = from.x;
    int y = from.y;
    int z = from.z;
    int sectorId = from.sector;
    result.hit = int(ClipMove(
        &x, &y, &z, &sectorId, to.x - from.x, to.y - from.y,
        body.radius, body.ceilingDistance, body.floorDistance, CLIPMASK0));
    actor->cstat = int16_t(savedCstat);

    result.end = { x, y, z, sectorId };
    result.reached = validSector(sectorId)
        && distanceSquared(x, y, to.x, to.y)
            <= int64_t(tolerance) * tolerance
        && (sectorId == to.sector || inside(x, y, to.sector) == 1)
        && engineLineHasContinuousSupport(from, to, body);
    return result;
}

Pose poseOnFloor(int sectorId, int x, int y)
{
    const BodyShape body = liveBodyShape();
    Pose result;
    result.x = x;
    result.y = y;
    result.sector = sectorId;
    result.z = validSector(sectorId)
        ? getflorzofslope(sectorId, x, y) - body.footOffset : 0;
    return result;
}

bool appendUniquePose(std::vector<Pose> &poses, const Pose &candidate)
{
    if (!validSector(candidate.sector)
        || inside(candidate.x, candidate.y, candidate.sector) != 1)
        return false;
    for (const Pose &known : poses)
        if (known.samePlace(candidate, 64))
            return false;
    poses.push_back(candidate);
    return true;
}

int ownerSectorOfWall(int wallId)
{
    for (int sectorId = 0; sectorId < numsectors; ++sectorId)
    {
        const int first = sector[sectorId].wallptr;
        if (wallId >= first && wallId < first + sector[sectorId].wallnum)
            return sectorId;
    }
    return -1;
}

bool wallSurfaceVisibleFromPlayer(int wallId, int ownerSector)
{
    if (!gMe || !gMe->pSprite || !validWall(wallId)
        || !validSector(ownerSector) || !validWall(wall[wallId].point2))
        return false;
    const walltype &start = wall[wallId];
    const walltype &end = wall[start.point2];
    const int dx = end.x - start.x;
    const int dy = end.y - start.y;
    const int length = std::max(1, distanceTo(start.x, start.y, end.x, end.y));
    const int fractions[] = { 1, 2, 3 };
    for (int fraction : fractions)
    {
        const int surfaceX = start.x + int(int64_t(dx) * fraction / 4);
        const int surfaceY = start.y + int(int64_t(dy) * fraction / 4);
        for (int direction = -1; direction <= 1; direction += 2)
        {
            const int x = surfaceX + direction * -dy * 32 / length;
            const int y = surfaceY + direction * dx * 32 / length;
            if (inside(x, y, ownerSector) != 1)
                continue;
            const int ceiling = getceilzofslope(ownerSector, x, y);
            const int floor = getflorzofslope(ownerSector, x, y);
            const int z = clampInt(gMe->zView, ceiling + 64, floor - 64);
            if (cansee(gMe->pSprite->x, gMe->pSprite->y, gMe->zView,
                       gMe->pSprite->sectnum, x, y, z, ownerSector))
                return true;
        }
    }
    return false;
}

bool portalWitness(int wallId, int fromSector, Pose &source, Pose &target)
{
    if (!validWall(wallId) || !validSector(fromSector)
        || !validWall(wall[wallId].point2)
        || !validSector(wall[wallId].nextsector))
        return false;
    const walltype &start = wall[wallId];
    const walltype &end = wall[start.point2];
    const int toSector = start.nextsector;
    const int dx = end.x - start.x;
    const int dy = end.y - start.y;
    const int length = std::max(1, distanceTo(start.x, start.y, end.x, end.y));
    const BodyShape body = liveBodyShape();
    if (length < body.radius * 2)
        return false;

    std::vector<int> along;
    along.push_back(length / 2);
    along.push_back(body.radius);
    along.push_back(length - body.radius);
    along.push_back(length / 4);
    along.push_back(length * 3 / 4);
    for (int offset : along)
    {
        if (offset < body.radius || offset > length - body.radius)
            continue;
        const int wallX = start.x + int(int64_t(dx) * offset / length);
        const int wallY = start.y + int(int64_t(dy) * offset / length);
        const int insets[] = { body.radius + 64, body.radius, 32 };
        for (int inset : insets)
        {
            const int nx = int(int64_t(-dy) * inset / length);
            const int ny = int(int64_t(dx) * inset / length);
            for (int direction = -1; direction <= 1; direction += 2)
            {
                const int sourceX = wallX + direction * nx;
                const int sourceY = wallY + direction * ny;
                const int targetX = wallX - direction * nx;
                const int targetY = wallY - direction * ny;
                if (inside(sourceX, sourceY, fromSector) != 1
                    || inside(targetX, targetY, toSector) != 1)
                    continue;
                source = poseOnFloor(fromSector, sourceX, sourceY);
                target = poseOnFloor(toSector, targetX, targetY);
                if (std::abs((target.z + body.footOffset)
                             - (source.z + body.footOffset)) > 1024)
                    continue;
                if (engineWalkLine(source, target,
                                   std::max(48, body.radius / 2)).reached)
                    return true;
            }
        }
    }
    return false; // UNKNOWN, never cached as an impossible portal.
}

std::vector<Pose> localSearchPoints(const std::set<int> &sectors,
                                    const Pose &start, const Pose &goal)
{
    std::vector<Pose> result;
    appendUniquePose(result, start);
    appendUniquePose(result, goal);
    const BodyShape body = liveBodyShape();
    const int inset = body.radius + 96;
    for (int sectorId : sectors)
    {
        if (!validSector(sectorId))
            continue;
        const sectortype &region = sector[sectorId];
        for (int offset = 0; offset < region.wallnum; ++offset)
        {
            const int wallId = region.wallptr + offset;
            if (!validWall(wallId) || !validWall(wall[wallId].point2))
                continue;
            const walltype &a = wall[wallId];
            const walltype &b = wall[a.point2];
            const int dx = b.x - a.x;
            const int dy = b.y - a.y;
            const int length = std::max(1, distanceTo(a.x, a.y, b.x, b.y));
            int nx = int(int64_t(-dy) * inset / length);
            int ny = int(int64_t(dx) * inset / length);
            const int midX = (a.x + b.x) / 2;
            const int midY = (a.y + b.y) / 2;
            if (inside(midX + nx, midY + ny, sectorId) != 1)
            {
                nx = -nx;
                ny = -ny;
            }
            const int basesX[] = { a.x, midX, b.x };
            const int basesY[] = { a.y, midY, b.y };
            for (int i = 0; i < 3; ++i)
            {
                appendUniquePose(result, poseOnFloor(
                    sectorId, basesX[i] + nx, basesY[i] + ny));
                if (int(result.size()) >= kMaximumLocalCandidates)
                    return result;
            }
        }
    }
    return result;
}

bool findEngineBackedPath(const Pose &start, const Pose &goal,
                          const std::set<int> &allowedSectors,
                          std::vector<Pose> &path, int &cost)
{
    path.clear();
    cost = 0;
    if (engineWalkLine(start, goal, kWaypointTolerance).reached)
    {
        path.push_back(goal);
        cost = distanceTo(start.x, start.y, goal.x, goal.y);
        return true;
    }

    const std::vector<Pose> points = localSearchPoints(
        allowedSectors, start, goal);
    if (points.size() < 2)
        return false;
    const int count = int(points.size());
    const int64_t infinity = std::numeric_limits<int64_t>::max() / 4;
    std::vector<int64_t> distance(size_t(count), infinity);
    std::vector<int> parent(size_t(count), -1);
    using QueueEntry = std::pair<int64_t, int>;
    std::priority_queue<QueueEntry, std::vector<QueueEntry>,
                        std::greater<QueueEntry>> queue;
    distance[0] = 0;
    queue.push({ 0, 0 });
    while (!queue.empty())
    {
        const QueueEntry entry = queue.top();
        queue.pop();
        const int node = entry.second;
        if (entry.first != distance[size_t(node)])
            continue;
        if (node == 1)
            break;
        for (int next = 1; next < count; ++next)
        {
            if (next == node)
                continue;
            const int edgeCost = distanceTo(
                points[size_t(node)].x, points[size_t(node)].y,
                points[size_t(next)].x, points[size_t(next)].y);
            if (edgeCost == 0 || distance[size_t(node)] + edgeCost
                    >= distance[size_t(next)])
                continue;
            if (!engineWalkLine(points[size_t(node)], points[size_t(next)],
                                kWaypointTolerance).reached)
                continue;
            distance[size_t(next)] = distance[size_t(node)] + edgeCost;
            parent[size_t(next)] = node;
            queue.push({ distance[size_t(next)], next });
        }
    }
    if (distance[1] == infinity)
        return false; // Search did not find positive evidence: UNKNOWN.
    std::vector<Pose> reverse;
    for (int node = 1; node > 0; node = parent[size_t(node)])
        reverse.push_back(points[size_t(node)]);
    path.assign(reverse.rbegin(), reverse.rend());
    cost = int(std::min<int64_t>(INT32_MAX, distance[1]));
    return true;
}

enum class InteractionKind
{
    Wall,
    Sector,
    Sprite,
};

struct InteractionKey
{
    InteractionKind kind = InteractionKind::Wall;
    int id = -1;

    bool operator<(const InteractionKey &other) const
    {
        return std::tie(kind, id) < std::tie(other.kind, other.id);
    }
};

struct Interaction
{
    InteractionKey key;
    int fromSector = -1;
    int mapWall = -1;
    int x = 0;
    int y = 0;
    int targetZ = 0;
    int requiredKey = 0;
};

bool playerHasKey(int key)
{
    return key <= 0 || (gMe && key < int(sizeof(gMe->hasKey))
        && gMe->hasKey[key]);
}

bool interactionMatches(const Interaction &interaction,
                        int hit, int target)
{
    if (interaction.key.kind == InteractionKind::Wall)
        return hit == 0 && target == interaction.key.id;
    if (interaction.key.kind == InteractionKind::Sector)
        return hit == 6 && target == interaction.key.id;
    return hit == 3 && target == interaction.key.id;
}

int lookForTarget(int eyeZ, int targetZ, int horizontal)
{
    const double angle = std::atan2(double(eyeZ - targetZ),
        double(std::max(1, horizontal))) * 1024.0 / 3.14159265358979323846;
    return clampInt(int(std::lround(angle)), kLookDownLimit, kLookUpLimit);
}

bool previewUseFromPose(const Interaction &interaction, const Pose &pose,
                        int angle, int look)
{
    if (!gMe || !gMe->pSprite || !validSector(pose.sector))
        return false;
    spritetype *actor = gMe->pSprite;
    const spritetype savedActor = *actor;
    const fix16_t savedAngle = gMe->q16ang;
    const fix16_t savedLook = gMe->q16look;
    const fix16_t savedHoriz = gMe->q16horiz;
    const int savedSlope = gMe->slope;
    const int savedViewZ = gMe->zView;
    const HITINFO savedHit = gHitInfo;

    actor->x = pose.x;
    actor->y = pose.y;
    actor->z = pose.z;
    actor->sectnum = int16_t(pose.sector);
    actor->ang = int16_t(angle);
    gMe->q16ang = fix16_from_int(angle);
    gMe->q16look = fix16_from_int(look);
    gMe->q16horiz = fix16_from_float(
        100.f * tanf(float(look) * 3.14159265358979323846f / 1024.f));
    gMe->slope = (-fix16_to_int(gMe->q16horiz)) << 7;
    gMe->zView = pose.z
        - gMe->pPosture[gMe->lifeMode][gMe->posture].eyeAboveZ;
    int target = -1;
    int extra = -1;
    const int hit = ActionScanPreview(gMe, &target, &extra);

    *actor = savedActor;
    gMe->q16ang = savedAngle;
    gMe->q16look = savedLook;
    gMe->q16horiz = savedHoriz;
    gMe->slope = savedSlope;
    gMe->zView = savedViewZ;
    gHitInfo = savedHit;
    return interactionMatches(interaction, hit, target);
}

bool findUsePose(const Interaction &interaction, Pose &result,
                 int &resultAngle, int &resultLook)
{
    if (!validSector(interaction.fromSector))
        return false;
    std::vector<Pose> candidates;
    const int distances[] = { 704, 896, 512 };
    for (int distance : distances)
    {
        for (int angle = 0; angle < kAng360; angle += 128)
        {
            const int x = interaction.x - mulscale30(distance, Cos(angle));
            const int y = interaction.y - mulscale30(distance, Sin(angle));
            if (inside(x, y, interaction.fromSector) != 1)
                continue;
            appendUniquePose(candidates,
                poseOnFloor(interaction.fromSector, x, y));
        }
    }
    if (interaction.mapWall >= 0 && validWall(interaction.mapWall)
        && validWall(wall[interaction.mapWall].point2))
    {
        const walltype &a = wall[interaction.mapWall];
        const walltype &b = wall[a.point2];
        const int dx = b.x - a.x;
        const int dy = b.y - a.y;
        const int length = std::max(1, distanceTo(a.x, a.y, b.x, b.y));
        for (int fraction = 1; fraction <= 3; ++fraction)
        {
            const int onX = a.x + int(int64_t(dx) * fraction / 4);
            const int onY = a.y + int(int64_t(dy) * fraction / 4);
            for (int direction = -1; direction <= 1; direction += 2)
            {
                const int x = onX + direction * -dy * 704 / length;
                const int y = onY + direction * dx * 704 / length;
                if (inside(x, y, interaction.fromSector) == 1)
                    appendUniquePose(candidates,
                        poseOnFloor(interaction.fromSector, x, y));
            }
        }
    }

    for (const Pose &candidate : candidates)
    {
        const int angle = getangle(
            interaction.x - candidate.x, interaction.y - candidate.y);
        const int eyeZ = candidate.z
            - gMe->pPosture[gMe->lifeMode][gMe->posture].eyeAboveZ;
        const int look = lookForTarget(eyeZ, interaction.targetZ,
            distanceTo(candidate.x, candidate.y,
                       interaction.x, interaction.y));
        for (int adjustment = -96; adjustment <= 96; adjustment += 32)
        {
            const int adjustedLook = clampInt(
                look + adjustment, kLookDownLimit, kLookUpLimit);
            if (!previewUseFromPose(interaction, candidate,
                                    angle, adjustedLook))
                continue;
            result = candidate;
            resultAngle = angle;
            resultLook = adjustedLook;
            return true;
        }
    }
    return false;
}

enum class WorkKind
{
    Pickup,
    Use,
    ObserveWall,
};

struct Work
{
    WorkKind kind = WorkKind::ObserveWall;
    int id = -1;
    Pose destination;
    Interaction interaction;
    int desiredAngle = 0;
    int desiredLook = 0;
    std::vector<Pose> path;
    int cost = INT32_MAX;
};

const char *workName(WorkKind kind)
{
    switch (kind)
    {
    case WorkKind::Pickup: return "pickup";
    case WorkKind::Use: return "use";
    case WorkKind::ObserveWall: return "observe_wall";
    }
    return "unknown";
}

} // namespace

struct LLMapperBot::Impl
{
    FILE *telemetry = nullptr;
    FILE *trajectory = nullptr;
    std::string telemetryPath = "llmapper-bot.ndjson";
    std::string trajectoryPath = "llmapper-bot-trajectory.ndjson";
    std::string demoPath = "llmapper-bot.dem";
    int runtimeLimitSeconds = 300;
    std::string result;
    std::string failureReason;
    LLMapperPlayerState player;
    GINPUT issued = {};
    int lastTrajectoryFrame = -1;
    int lastSector = -1;
    std::set<int> enteredSectors;
    std::set<int> observedSectors;
    std::set<int> observedWalls;
    std::map<InteractionKey, Interaction> knownInteractions;
    std::set<InteractionKey> resolvedInteractions;
    bool hasActiveWork = false;
    Work activeWork;
    size_t pathIndex = 0;
    bool waitingForActionWorldChange = false;
    bool physicalStateInProgressReported = false;
    uint64_t actionWorldBefore = 0;
    int topologyBuilds = 0;
    int localSearches = 0;

    int gameTime() const
    {
        return (gFrame * kTicsPerFrame) / kTicsPerSec;
    }

    void event(const char *name, const char *detail = "")
    {
        if (!telemetry)
            return;
        std::fprintf(telemetry,
            "{\"type\":\"event\",\"game_time\":%d,\"tick\":%d,"
            "\"event\":\"%s\",\"detail\":\"%s\"}\n",
            gameTime(), gFrame * kTicsPerFrame, name, detail ? detail : "");
        std::fflush(telemetry);
    }

    void openFiles()
    {
        if (!telemetry && !telemetryPath.empty())
            telemetry = std::fopen(telemetryPath.c_str(), "wb");
        if (!trajectory && !trajectoryPath.empty())
            trajectory = std::fopen(trajectoryPath.c_str(), "wb");
    }

    LLMapperPlayerState readPlayerState() const
    {
        LLMapperPlayerState state;
        if (!gMe || !gMe->pSprite || !gMe->pXSprite)
            return state;
        state.x = gMe->pSprite->x;
        state.y = gMe->pSprite->y;
        state.z = gMe->pSprite->z;
        state.sector = gMe->pSprite->sectnum;
        state.angle = gMe->pSprite->ang;
        state.horizon = fix16_to_int(gMe->q16look);
        state.onGround = gMe->pXSprite->height == 0;
        state.crouched = gMe->posture == kPostureCrouch;
        state.alive = gMe->pXSprite->health > 0;
        return state;
    }

    Pose playerPose() const
    {
        return { player.x, player.y, player.z, player.sector };
    }

    uint64_t worldSignature() const
    {
        uint64_t hash = 1469598103934665603ULL;
        for (int i = 0; i < numsectors; ++i)
        {
            hash = mixHash(hash, uint32_t(sector[i].floorz));
            hash = mixHash(hash, uint32_t(sector[i].ceilingz));
            if (sector[i].extra > 0 && sector[i].extra < kMaxXSectors)
            {
                const XSECTOR &extra = xsector[sector[i].extra];
                hash = mixHash(hash, uint32_t(extra.state));
                hash = mixHash(hash, uint32_t(extra.busy));
                hash = mixHash(hash, uint32_t(extra.locked));
            }
        }
        for (int i = 0; i < numwalls; ++i)
        {
            hash = mixHash(hash, uint32_t(wall[i].cstat));
            if (wall[i].extra > 0 && wall[i].extra < kMaxXWalls)
            {
                const XWALL &extra = xwall[wall[i].extra];
                hash = mixHash(hash, uint32_t(extra.state));
                hash = mixHash(hash, uint32_t(extra.busy));
                hash = mixHash(hash, uint32_t(extra.locked));
            }
        }
        for (int key = 0; key < int(sizeof(gMe->hasKey)); ++key)
            hash = mixHash(hash, gMe->hasKey[key] ? uint64_t(key + 1) : 0);
        return hash;
    }

    bool worldStateInProgress() const
    {
        for (int i = 0; i < numsectors; ++i)
            if (sector[i].extra > 0 && sector[i].extra < kMaxXSectors
                && xsector[sector[i].extra].busy != 0
                && xsector[sector[i].extra].busy != 65536)
                return true;
        for (int i = 0; i < numwalls; ++i)
            if (wall[i].extra > 0 && wall[i].extra < kMaxXWalls
                && xwall[wall[i].extra].busy != 0
                && xwall[wall[i].extra].busy != 65536)
                return true;
        for (int i = 0; i < kMaxSprites; ++i)
            if (validSprite(i) && sprite[i].extra > 0
                && sprite[i].extra < kMaxXSprites
                && xsprite[sprite[i].extra].busy != 0
                && xsprite[sprite[i].extra].busy != 65536)
                return true;
        return false;
    }

    void observe()
    {
        player = readPlayerState();
        if (!validSector(player.sector))
            return;
        enteredSectors.insert(player.sector);
        if (lastSector != player.sector)
        {
            lastSector = player.sector;
            char detail[96];
            std::snprintf(detail, sizeof(detail),
                "sector=%d depth=0 branch=%d", player.sector, player.sector);
            event("sector_entered", detail);
        }

        for (int wallId = 0; wallId < numwalls; ++wallId)
        {
            const int owner = ownerSectorOfWall(wallId);
            if (owner < 0 || !wallSurfaceVisibleFromPlayer(wallId, owner))
                continue;
            observedWalls.insert(wallId);
            observedSectors.insert(owner);
            const walltype &record = wall[wallId];
            if (record.extra > 0 && record.extra < kMaxXWalls
                && xwall[record.extra].triggerPush)
            {
                Interaction interaction;
                interaction.key = { InteractionKind::Wall, wallId };
                interaction.fromSector = owner;
                interaction.mapWall = wallId;
                interaction.x = (record.x + wall[record.point2].x) / 2;
                interaction.y = (record.y + wall[record.point2].y) / 2;
                interaction.targetZ = (getceilzofslope(owner,
                    interaction.x, interaction.y) + getflorzofslope(owner,
                    interaction.x, interaction.y)) / 2;
                interaction.requiredKey = xwall[record.extra].key;
                knownInteractions[interaction.key] = interaction;
            }
            if (validSector(record.nextsector)
                && sector[record.nextsector].extra > 0
                && sector[record.nextsector].extra < kMaxXSectors
                && xsector[sector[record.nextsector].extra].Wallpush)
            {
                const XSECTOR &extra = xsector[sector[record.nextsector].extra];
                Interaction interaction;
                interaction.key = { InteractionKind::Sector,
                                    record.nextsector };
                interaction.fromSector = owner;
                interaction.mapWall = wallId;
                interaction.x = (record.x + wall[record.point2].x) / 2;
                interaction.y = (record.y + wall[record.point2].y) / 2;
                const int fromFloor = getflorzofslope(owner,
                    interaction.x, interaction.y);
                const int toFloor = getflorzofslope(record.nextsector,
                    interaction.x, interaction.y);
                const int fromCeiling = getceilzofslope(owner,
                    interaction.x, interaction.y);
                const int toCeiling = getceilzofslope(record.nextsector,
                    interaction.x, interaction.y);
                interaction.targetZ = fromFloor != toFloor
                    ? (fromFloor + toFloor) / 2
                    : (fromCeiling + toCeiling) / 2;
                interaction.requiredKey = extra.Key;
                knownInteractions[interaction.key] = interaction;
            }
        }

        const sectortype &current = sector[player.sector];
        if (current.extra > 0 && current.extra < kMaxXSectors
            && xsector[current.extra].Push)
        {
            Interaction interaction;
            interaction.key = { InteractionKind::Sector, player.sector };
            interaction.fromSector = player.sector;
            interaction.x = player.x;
            interaction.y = player.y;
            interaction.targetZ = player.z;
            interaction.requiredKey = xsector[current.extra].Key;
            knownInteractions[interaction.key] = interaction;
        }

        for (int spriteId = 0; spriteId < kMaxSprites; ++spriteId)
        {
            if (!validSprite(spriteId) || spriteId == gMe->pSprite->index)
                continue;
            spritetype &record = sprite[spriteId];
            if (!cansee(player.x, player.y, gMe->zView, player.sector,
                        record.x, record.y, record.z, record.sectnum))
                continue;
            observedSectors.insert(record.sectnum);
            if (record.extra <= 0 || record.extra >= kMaxXSprites
                || !xsprite[record.extra].Push)
                continue;
            Interaction interaction;
            interaction.key = { InteractionKind::Sprite, spriteId };
            interaction.fromSector = record.sectnum;
            interaction.x = record.x;
            interaction.y = record.y;
            int top = record.z;
            int bottom = record.z;
            GetSpriteExtents(&record, &top, &bottom);
            interaction.targetZ = (top + bottom) / 2;
            interaction.requiredKey = xsprite[record.extra].key;
            knownInteractions[interaction.key] = interaction;
        }

        if (waitingForActionWorldChange && worldSignature() != actionWorldBefore)
        {
            waitingForActionWorldChange = false;
            event("world_changed", "source=accepted_use");
        }
    }

    std::set<int> physicallyReachableSectors() const
    {
        std::set<int> reached;
        if (!validSector(player.sector))
            return reached;
        std::queue<int> pending;
        reached.insert(player.sector);
        pending.push(player.sector);
        while (!pending.empty())
        {
            const int from = pending.front();
            pending.pop();
            const sectortype &region = sector[from];
            for (int offset = 0; offset < region.wallnum; ++offset)
            {
                const int wallId = region.wallptr + offset;
                const int to = wall[wallId].nextsector;
                if (!validSector(to) || reached.count(to))
                    continue;
                Pose source;
                Pose target;
                if (!portalWitness(wallId, from, source, target))
                    continue;
                reached.insert(to);
                pending.push(to);
            }
        }
        return reached;
    }

    bool itemVisibleAndUseful(int spriteId) const
    {
        if (!validSprite(spriteId) || sprite[spriteId].statnum != kStatItem)
            return false;
        const spritetype &item = sprite[spriteId];
        return cansee(player.x, player.y, gMe->zView, player.sector,
                      item.x, item.y, item.z, item.sectnum)
            && playerCanBenefitFromPickup(gMe, &sprite[spriteId]);
    }

    Pose wallObservationPose(int wallId) const
    {
        const int owner = ownerSectorOfWall(wallId);
        if (!validWall(wallId) || !validSector(owner)
            || !validWall(wall[wallId].point2))
            return {};
        const walltype &a = wall[wallId];
        const walltype &b = wall[a.point2];
        const int midX = (a.x + b.x) / 2;
        const int midY = (a.y + b.y) / 2;
        const int length = std::max(1, distanceTo(a.x, a.y, b.x, b.y));
        const int inset = liveBodyShape().radius + 128;
        int nx = int(int64_t(-(b.y - a.y)) * inset / length);
        int ny = int(int64_t(b.x - a.x) * inset / length);
        if (inside(midX + nx, midY + ny, owner) != 1)
        {
            nx = -nx;
            ny = -ny;
        }
        return poseOnFloor(owner, midX + nx, midY + ny);
    }

    bool routeWork(Work &work, const std::set<int> &reachable)
    {
        ++localSearches;
        const Pose start = playerPose();
        std::set<int> allowed = reachable;
        allowed.insert(start.sector);
        allowed.insert(work.destination.sector);
        return findEngineBackedPath(start, work.destination,
                                    allowed, work.path, work.cost);
    }

    bool chooseWork(Work &selected)
    {
        const std::set<int> reachable = physicallyReachableSectors();
        std::vector<Work> candidates;

        for (int spriteId = 0; spriteId < kMaxSprites; ++spriteId)
        {
            if (!itemVisibleAndUseful(spriteId)
                || !reachable.count(sprite[spriteId].sectnum))
                continue;
            Work work;
            work.kind = WorkKind::Pickup;
            work.id = spriteId;
            work.destination = poseOnFloor(sprite[spriteId].sectnum,
                sprite[spriteId].x, sprite[spriteId].y);
            candidates.push_back(work);
        }

        for (const auto &entry : knownInteractions)
        {
            const Interaction &interaction = entry.second;
            if (resolvedInteractions.count(interaction.key)
                || !playerHasKey(interaction.requiredKey)
                || !reachable.count(interaction.fromSector))
                continue;
            Work work;
            work.kind = WorkKind::Use;
            work.id = interaction.key.id;
            work.interaction = interaction;
            if (!findUsePose(interaction, work.destination,
                             work.desiredAngle, work.desiredLook))
                continue; // Known action, execution domain still UNKNOWN.
            candidates.push_back(work);
        }

        for (int wallId = 0; wallId < numwalls; ++wallId)
        {
            const int owner = ownerSectorOfWall(wallId);
            if (observedWalls.count(wallId) || !reachable.count(owner))
                continue;
            Work work;
            work.kind = WorkKind::ObserveWall;
            work.id = wallId;
            work.destination = wallObservationPose(wallId);
            if (validSector(work.destination.sector))
                candidates.push_back(work);
        }

        bool found = false;
        for (Work &candidate : candidates)
        {
            if (!routeWork(candidate, reachable))
                continue;
            if (!found || candidate.cost < selected.cost
                || (candidate.cost == selected.cost
                    && std::tie(candidate.kind, candidate.id)
                        < std::tie(selected.kind, selected.id)))
            {
                selected = candidate;
                found = true;
            }
        }
        return found;
    }

    void selectIfNeeded()
    {
        if (hasActiveWork || waitingForActionWorldChange || !result.empty())
            return;
        Work selected;
        if (!chooseWork(selected))
        {
            if (worldStateInProgress())
            {
                if (!physicalStateInProgressReported)
                    event("physical_state_in_progress",
                          "action=observe_until_engine_state_changes");
                physicalStateInProgressReported = true;
                return;
            }
            result = "MODEL_CONTRADICTION";
            failureReason = "no physically proven unresolved possibility";
            event("model_contradiction", failureReason.c_str());
            gQuitGame = true;
            return;
        }
        activeWork = selected;
        physicalStateInProgressReported = false;
        hasActiveWork = true;
        pathIndex = 0;
        char detail[192];
        std::snprintf(detail, sizeof(detail),
            "kind=%s id=%d sector=%d at=(%d,%d,%d) cost=%d path=%u",
            workName(activeWork.kind), activeWork.id,
            activeWork.destination.sector, activeWork.destination.x,
            activeWork.destination.y, activeWork.destination.z, activeWork.cost,
            unsigned(activeWork.path.size()));
        event("work_selected", detail);
        for (size_t i = 0; i < activeWork.path.size(); ++i)
        {
            std::snprintf(detail, sizeof(detail),
                "work=%s:%d index=%u sector=%d at=(%d,%d,%d)",
                workName(activeWork.kind), activeWork.id, unsigned(i),
                activeWork.path[i].sector, activeWork.path[i].x,
                activeWork.path[i].y, activeWork.path[i].z);
            event("physical_path_step", detail);
        }
    }

    bool workStillUnresolved() const
    {
        if (!hasActiveWork)
            return false;
        if (activeWork.kind == WorkKind::Pickup)
            return itemVisibleAndUseful(activeWork.id);
        if (activeWork.kind == WorkKind::Use)
            return !resolvedInteractions.count(activeWork.interaction.key);
        return !observedWalls.count(activeWork.id);
    }

    GINPUT driveTo(const Pose &target)
    {
        llmapper::PhysicalCommand command;
        command.type = llmapper::PhysicalCommandType::Move;
        const int desired = getangle(target.x - player.x, target.y - player.y);
        const int delta = DANGLE(desired, player.angle);
        command.turn = fix16_from_int(clampInt(delta, -48, 48));
        const int remaining = distanceTo(player.x, player.y, target.x, target.y);
        const int magnitude = remaining < 512 ? 320 : 640;
        // Blood resolves forward/strafe in the live facing basis. Express
        // the desired world direction in that basis so turning does not bend
        // a certified straight segment into the wall beside it.
        command.forward = int16_t(mulscale30(magnitude, Cos(delta)));
        command.strafe = int16_t(-mulscale30(magnitude, Sin(delta)));
        return llmapper::commandToInput(command);
    }

    GINPUT alignAndUse()
    {
        int target = -1;
        int extra = -1;
        const int hit = ActionScanPreview(gMe, &target, &extra);
        if (interactionMatches(activeWork.interaction, hit, target))
        {
            actionWorldBefore = worldSignature();
            return llmapper::commandToInput(
                { llmapper::PhysicalCommandType::Use });
        }

        llmapper::PhysicalCommand command;
        command.type = llmapper::PhysicalCommandType::Move;
        const int angleDelta = DANGLE(activeWork.desiredAngle, player.angle);
        command.turn = fix16_from_int(clampInt(angleDelta, -48, 48));
        const int lookDelta = activeWork.desiredLook - player.horizon;
        command.look = fix16_from_int(clampInt(lookDelta / 8, -32, 32));
        return llmapper::commandToInput(command);
    }

    GINPUT decide()
    {
        observe();
        if (!player.alive || !result.empty())
            return {};
        if (hasActiveWork && !workStillUnresolved())
        {
            event("work_resolved", workName(activeWork.kind));
            hasActiveWork = false;
        }
        selectIfNeeded();
        if (!hasActiveWork || waitingForActionWorldChange || !result.empty())
            return {};

        while (pathIndex < activeWork.path.size()
               && playerPose().samePlace(activeWork.path[pathIndex],
                                         kWaypointTolerance))
            ++pathIndex;
        if (pathIndex < activeWork.path.size())
        {
            // A path is only positive physical evidence for the body shape
            // that was queried. Blood animation can change that live hull.
            // Revalidate the next segment; loss of the witness invalidates
            // this execution, but does not mark the destination impossible.
            if (!engineWalkLine(playerPose(), activeWork.path[pathIndex],
                                kWaypointTolerance).reached)
            {
                event("physical_path_invalidated",
                      "reason=live_engine_witness_changed");
                hasActiveWork = false;
                selectIfNeeded();
                return {};
            }
            return driveTo(activeWork.path[pathIndex]);
        }

        if (activeWork.kind == WorkKind::Use)
            return alignAndUse();
        return {};
    }

    void sampleTrajectory()
    {
        if (!trajectory || lastTrajectoryFrame == gFrame)
            return;
        lastTrajectoryFrame = gFrame;
        std::fprintf(trajectory,
            "{\"game_time\":%d,\"tick\":%d,\"x\":%d,\"y\":%d,"
            "\"z\":%d,\"sector\":%d,\"angle\":%d,\"look\":%d,"
            "\"on_ground\":%d,\"crouched\":%d,\"forward\":%d,"
            "\"strafe\":%d,\"turn\":%d,\"jump\":%d,\"use\":%d}\n",
            gameTime(), gFrame * kTicsPerFrame, player.x, player.y, player.z,
            player.sector, player.angle, player.horizon,
            player.onGround ? 1 : 0, player.crouched ? 1 : 0,
            int(issued.forward), int(issued.strafe), fix16_to_int(issued.q16turn),
            issued.buttonFlags.jump ? 1 : 0, issued.keyFlags.action ? 1 : 0);
        std::fflush(trajectory);
    }

    void close(const char *reason)
    {
        if (result.empty())
        {
            result = reason && *reason ? "RUNTIME_ERROR" : "STOPPED";
            failureReason = reason && *reason ? reason : "run stopped";
        }
        if (telemetry)
        {
            std::fprintf(telemetry,
                "{\"type\":\"summary\",\"result\":\"%s\","
                "\"failure_reason\":\"%s\",\"game_time\":%d,"
                "\"total_sectors\":%d,\"visited_sectors\":%u,"
                "\"observed_sectors\":%u,\"topology_rebuilds\":%d,"
                "\"local_searches\":%d,\"physical_probes\":%d}\n",
                result.c_str(), failureReason.c_str(), gameTime(), numsectors,
                unsigned(enteredSectors.size()), unsigned(observedSectors.size()),
                topologyBuilds, localSearches, gPhysicalProbeCount);
            std::fclose(telemetry);
            telemetry = nullptr;
        }
        if (trajectory)
        {
            std::fclose(trajectory);
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

void LLMapperBot::Enable(const char *telemetry, const char *trajectory,
                         const char *demo)
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
        m_impl->runtimeLimitSeconds = seconds;
}

void LLMapperBot::SetFast(bool fast) { m_fast = fast; }

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
    m_impl->event("run_started", "architecture=minimal_autonomous");
    if (!gDemo.at0 && !gDemo.at1 && !gDemo.Create(m_impl->demoPath.c_str()))
    {
        m_impl->result = "RUNTIME_ERROR";
        m_impl->failureReason = "could not create demo file";
        gQuitGame = true;
    }
}

GINPUT LLMapperBot::GetInput()
{
    if (!m_enabled)
        return {};
    m_impl->issued = m_impl->decide();
    return m_impl->issued;
}

void LLMapperBot::OnFrame()
{
    if (!m_enabled || !gGameStarted || !gMe || !gMe->pXSprite)
        return;
    m_impl->player = m_impl->readPlayerState();
    m_impl->sampleTrajectory();
    if ((gGameOptions.uGameFlags & kGameFlagContinuing)
        && m_impl->result.empty())
        OnLevelExit(kLevelExitNormal);
    if (!m_impl->player.alive && m_impl->result.empty())
    {
        m_impl->result = "DIED";
        m_impl->failureReason = "player health reached zero";
        m_impl->event("failure", m_impl->failureReason.c_str());
        gQuitGame = true;
    }
    if (m_impl->result.empty()
        && m_impl->gameTime() >= m_impl->runtimeLimitSeconds)
    {
        m_impl->result = "TIMEOUT";
        m_impl->failureReason = "external run time limit reached";
        m_impl->event("failure", m_impl->failureReason.c_str());
        gQuitGame = true;
    }
}

void LLMapperBot::OnActionResolved(int hit, int target, int, bool accepted,
                                   int)
{
    if (!m_enabled || !m_impl->hasActiveWork
        || m_impl->activeWork.kind != WorkKind::Use)
        return;
    char detail[128];
    std::snprintf(detail, sizeof(detail),
        "hit=%d target=%d accepted=%d", hit, target, accepted ? 1 : 0);
    m_impl->event("use_resolved", detail);
    if (!interactionMatches(m_impl->activeWork.interaction, hit, target)
        || !accepted)
    {
        m_impl->result = "MODEL_CONTRADICTION";
        m_impl->failureReason = "engine rejected a previewed USE action";
        m_impl->event("model_contradiction", m_impl->failureReason.c_str());
        gQuitGame = true;
        return;
    }
    m_impl->resolvedInteractions.insert(m_impl->activeWork.interaction.key);
    m_impl->hasActiveWork = false;
    m_impl->waitingForActionWorldChange = true;
}

void LLMapperBot::OnLevelExit(int exitType)
{
    if (!m_enabled || !m_impl->result.empty())
        return;
    m_impl->result = "COMPLETED";
    char detail[48];
    std::snprintf(detail, sizeof(detail), "exit_type=%d", exitType);
    m_impl->event("level_completed", detail);
    gQuitGame = true;
}

void LLMapperBot::DrawStatus()
{
    if (!m_enabled || !m_visible || !gGameStarted)
        return;
    char line[128];
    std::snprintf(line, sizeof(line), "MINBOT %d:%02d s%d %s",
        m_impl->gameTime() / 60, m_impl->gameTime() % 60,
        m_impl->player.sector,
        m_impl->hasActiveWork ? workName(m_impl->activeWork.kind) : "observe");
    viewDrawText(3, line, 2, 4, -128, 0, 0, true, 256);
}

void LLMapperBot::Finish(const char *reason)
{
    if (m_enabled && (m_impl->telemetry || m_impl->trajectory))
        m_impl->close(reason);
}

LLMapperBot gLLMapperBot;
