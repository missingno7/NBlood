#include "blood_interactions.h"

#include "fix16.h"

#include <algorithm>
#include <cmath>
#include <tuple>

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

// ActionScan accepts a hit at approxDist>>4 < 64, that is a planar reach of
// 1024 map units. Every candidate stance below stays inside the engine's own
// bound; none of these numbers is a reach rule of the bot's own invention.
constexpr int kEngineReach = 1024;

std::vector<int> gWallOwner;

int clampInt(int value, int low, int high)
{
    return std::max(low, std::min(high, value));
}

void refreshWallOwners()
{
    if (int(gWallOwner.size()) == numwalls && !gWallOwner.empty())
        return;
    gWallOwner.assign(size_t(std::max(0, int(numwalls))), -1);
    for (int sectorId = 0; sectorId < numsectors; ++sectorId)
    {
        const int first = sector[sectorId].wallptr;
        for (int offset = 0; offset < sector[sectorId].wallnum; ++offset)
            if (validWall(first + offset))
                gWallOwner[size_t(first + offset)] = sectorId;
    }
}

int ownerOf(int wallId)
{
    refreshWallOwners();
    return validWall(wallId) ? gWallOwner[size_t(wallId)] : -1;
}

bool surfaceVisible(int wallId, int ownerSector)
{
    if (!gMe || !validWall(wallId) || !validSector(ownerSector)
        || !validWall(wall[wallId].point2))
        return false;
    const walltype &start = wall[wallId];
    const walltype &end = wall[start.point2];
    const int dx = end.x - start.x;
    const int dy = end.y - start.y;
    const int length = std::max(1, planarDistance(start.x, start.y,
                                                  end.x, end.y));
    for (int fraction = 1; fraction <= 3; ++fraction)
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
            if (eyeCanSee(x, y, z, ownerSector))
                return true;
        }
    }
    return false;
}

bool objectVisible(int spriteId)
{
    if (!validSprite(spriteId) || !validSector(sprite[spriteId].sectnum))
        return false;
    spritetype &record = sprite[spriteId];
    int top = record.z;
    int bottom = record.z;
    GetSpriteExtents(&record, &top, &bottom);
    const int inset = std::min(64, std::max(0, (bottom - top) / 4));
    const int samples[] = { (top + bottom) / 2, top + inset, bottom - inset,
                            record.z };
    for (int z : samples)
        if (eyeCanSee(record.x, record.y, z, record.sectnum))
            return true;
    return false;
}

// A transcription of CheckPickUp: the same planar bound, the same vertical
// bound and the same visibility test the engine applies every frame. The
// numbers are Blood's, not a reach rule of the bot's own invention.
bool pickupResolvesFrom(int spriteId, const PhysicalPose &pose)
{
    if (!gMe || !gMe->pSprite || !validSprite(spriteId)
        || !validSector(pose.sector))
        return false;
    spritetype &item = sprite[spriteId];
    if (item.flags & 32)
        return false;
    const BodyShape body = liveBody();
    int actorTop = 0;
    int actorBottom = 0;
    GetSpriteExtents(gMe->pSprite, &actorTop, &actorBottom);
    const int headroom = gMe->pSprite->z - actorTop;
    const int z = pose.supportZ - body.footOffset;
    const int top = z - headroom;
    const int bottom = z + body.footOffset;

    const int dx = klabs(pose.x - item.x) >> 4;
    if (dx > 48)
        return false;
    const int dy = klabs(pose.y - item.y) >> 4;
    if (dy > 48)
        return false;
    int vertical = 0;
    if (item.z < top)
        vertical = (top - item.z) >> 8;
    else if (item.z > bottom)
        vertical = (item.z - bottom) >> 8;
    if (vertical > 32)
        return false;
    if (approxDist(dx, dy) > 48)
        return false;
    int itemTop = 0;
    int itemBottom = 0;
    GetSpriteExtents(&item, &itemTop, &itemBottom);
    return cansee(pose.x, pose.y, z, int16_t(pose.sector), item.x, item.y,
                  item.z, item.sectnum)
        || cansee(pose.x, pose.y, z, int16_t(pose.sector), item.x, item.y,
                  itemTop, item.sectnum)
        || cansee(pose.x, pose.y, z, int16_t(pose.sector), item.x, item.y,
                  itemBottom, item.sectnum);
}

bool playerHasKey(int key)
{
    return key <= 0
        || (gMe && key < int(sizeof(gMe->hasKey)) && gMe->hasKey[key]);
}

} // namespace

int interactionKind(const InteractionKey &key)
{
    if ((key.tag == kPickupTag || key.tag == 3) && validSprite(key.id))
        return sprite[key.id].type;
    if (key.tag == 6 && validSector(key.id))
        return sector[key.id].type;
    return -1;
}

int interactionChannel(const InteractionKey &key)
{
    switch (key.tag)
    {
    case 0:
        if (validWall(key.id) && wall[key.id].extra > 0
            && wall[key.id].extra < kMaxXWalls)
            return xwall[wall[key.id].extra].txID;
        return -1;
    case 3:
    case kPickupTag:
        if (validSprite(key.id) && sprite[key.id].extra > 0
            && sprite[key.id].extra < kMaxXSprites)
            return xsprite[sprite[key.id].extra].txID;
        return -1;
    case 6:
        if (validSector(key.id) && sector[key.id].extra > 0
            && sector[key.id].extra < kMaxXSectors)
            return xsector[sector[key.id].extra].txID;
        return -1;
    default:
        return -1;
    }
}

bool interactionExists(const InteractionKey &key)
{
    switch (key.tag)
    {
    case kPickupTag:
        return validSprite(key.id) && sprite[key.id].statnum == kStatItem
            && !(sprite[key.id].flags & 32);
    case 0:
        return validWall(key.id) && wall[key.id].extra > 0
            && wall[key.id].extra < kMaxXWalls
            && xwall[wall[key.id].extra].triggerPush;
    case 6:
        return validSector(key.id) && sector[key.id].extra > 0
            && sector[key.id].extra < kMaxXSectors;
    case 3:
        return validSprite(key.id) && sprite[key.id].extra > 0
            && sprite[key.id].extra < kMaxXSprites
            && xsprite[sprite[key.id].extra].Push;
    default:
        return false;
    }
}

bool interactionUnlocked(const InteractionKey &key)
{
    if (!interactionExists(key))
        return false;
    switch (key.tag)
    {
    case kPickupTag:
        // Nothing in Blood locks a pickup. It is there or it is gone.
        return true;
    case 0:
    {
        const XWALL &extra = xwall[wall[key.id].extra];
        return !extra.locked && !extra.isTriggered && playerHasKey(extra.key);
    }
    case 6:
    {
        const XSECTOR &extra = xsector[sector[key.id].extra];
        return !extra.locked && !extra.isTriggered && playerHasKey(extra.Key);
    }
    case 3:
    {
        const XSPRITE &extra = xsprite[sprite[key.id].extra];
        return !extra.locked && !extra.isTriggered && playerHasKey(extra.key);
    }
    default:
        return false;
    }
}

bool interactionMatches(const InteractionKey &key, int hit, int target)
{
    // A pickup is never what an action scan resolves to; it resolves by the
    // body arriving, which the mapper notices separately.
    return key.tag != kPickupTag && hit == key.tag && target == key.id;
}

void discoverInteractions(std::vector<InteractionRecord> &out)
{
    out.clear();
    if (!gMe || !gMe->pSprite)
        return;
    refreshWallOwners();

    for (int wallId = 0; wallId < numwalls; ++wallId)
    {
        const int owner = ownerOf(wallId);
        if (!validSector(owner) || !validWall(wall[wallId].point2))
            continue;
        const walltype &record = wall[wallId];
        const bool pushWall = record.extra > 0 && record.extra < kMaxXWalls
            && xwall[record.extra].triggerPush;
        const int behind = record.nextsector;
        const bool pushRegion = validSector(behind)
            && sector[behind].extra > 0 && sector[behind].extra < kMaxXSectors
            && xsector[sector[behind].extra].Wallpush;
        if (!pushWall && !pushRegion)
            continue;
        const bool seen = surfaceVisible(wallId, owner);

        const int midX = (record.x + wall[record.point2].x) / 2;
        const int midY = (record.y + wall[record.point2].y) / 2;
        if (pushWall)
        {
            InteractionRecord interaction;
            interaction.key = { 0, wallId };
            interaction.x = midX;
            interaction.y = midY;
            interaction.targetZ = (getceilzofslope(owner, midX, midY)
                + getflorzofslope(owner, midX, midY)) / 2;
            interaction.referenceSector = owner;
            interaction.facingWall = wallId;
            interaction.requiredKey = xwall[record.extra].key;
            interaction.visible = seen;
            out.push_back(interaction);
        }
        if (pushRegion)
        {
            InteractionRecord interaction;
            interaction.key = { 6, behind };
            interaction.x = midX;
            interaction.y = midY;
            const int nearFloor = getflorzofslope(owner, midX, midY);
            const int farFloor = getflorzofslope(behind, midX, midY);
            const int nearCeiling = getceilzofslope(owner, midX, midY);
            const int farCeiling = getceilzofslope(behind, midX, midY);
            interaction.targetZ = nearFloor != farFloor
                ? (nearFloor + farFloor) / 2
                : (nearCeiling + farCeiling) / 2;
            interaction.referenceSector = owner;
            interaction.facingWall = wallId;
            interaction.requiredKey = xsector[sector[behind].extra].Key;
            interaction.visible = seen;
            out.push_back(interaction);
        }
    }

    const int here = gMe->pSprite->sectnum;
    if (validSector(here) && sector[here].extra > 0
        && sector[here].extra < kMaxXSectors && xsector[sector[here].extra].Push)
    {
        InteractionRecord interaction;
        interaction.key = { 6, here };
        interaction.x = gMe->pSprite->x;
        interaction.y = gMe->pSprite->y;
        interaction.targetZ = gMe->pSprite->z;
        interaction.referenceSector = here;
        interaction.requiredKey = xsector[sector[here].extra].Key;
        interaction.visible = true; // the actor is standing in it
        out.push_back(interaction);
    }

    for (int spriteId = 0; spriteId < kMaxSprites; ++spriteId)
    {
        if (!validSprite(spriteId) || spriteId == gMe->pSprite->index)
            continue;
        spritetype &record = sprite[spriteId];
        if (record.extra <= 0 || record.extra >= kMaxXSprites
            || !xsprite[record.extra].Push)
            continue;
        int top = record.z;
        int bottom = record.z;
        GetSpriteExtents(&record, &top, &bottom);
        InteractionRecord interaction;
        interaction.key = { 3, spriteId };
        interaction.x = record.x;
        interaction.y = record.y;
        interaction.targetZ = (top + bottom) / 2;
        interaction.referenceSector = record.sectnum;
        // A wall-aligned sprite is flat on a surface and faces one way. The
        // engine has to trace to its front, so where to stand is not a
        // question to answer by trying every direction and hoping: it is the
        // direction the thing faces.
        if (record.cstat & 16)
            interaction.facingAngle = record.ang;
        interaction.requiredKey = xsprite[record.extra].key;
        interaction.visible = objectVisible(spriteId);
        out.push_back(interaction);
    }

    // Things that are taken rather than operated. Same record, same
    // execution-domain question; only the engine mechanics behind them
    // differ, and those stop here.
    for (int spriteId = headspritestat[kStatItem]; spriteId >= 0;
         spriteId = nextspritestat[spriteId])
    {
        if (!validSprite(spriteId) || (sprite[spriteId].flags & 32))
            continue;
        const spritetype &record = sprite[spriteId];
        InteractionRecord interaction;
        interaction.visible = objectVisible(spriteId);
        interaction.key = { kPickupTag, spriteId };
        interaction.kind = semantic::ActionKind::Collect;
        interaction.x = record.x;
        interaction.y = record.y;
        interaction.targetZ = record.z;
        interaction.referenceSector = record.sectnum;
        out.push_back(interaction);
    }
}

int aimAngle(const PhysicalPose &pose, int targetX, int targetY)
{
    return getangle(targetX - pose.x, targetY - pose.y);
}

// Which way to look to put something on the engine's own use ray.
//
// The two axes are not in the same units and the engine never pretends they
// are, so this is not an angle in any ordinary sense and cannot be found by
// handing a height and a distance to atan2. Following it through:
//
//   ActionScan traces with (Cos(ang)>>16, Sin(ang)>>16), a vector of length
//   1<<14, and with slope as the third component; HitScan then passes that
//   third component on as slope<<4; and hitscan advances x and z by the same
//   parametric step, so the ray reaches a point D away and dz down exactly
//   when (slope<<4) / (1<<14) == dz / D.
//
//   Blood builds slope out of look as -(100 * tan(look * pi / 1024)) << 7.
//
// Put together, tan(look * pi / 1024) = -dz / (kRayScale * D). Leaving out
// that scale -- comparing a z against an x as though a unit of each were the
// same length -- aims about two and a half times too steeply, which for
// anything much off level saturates at the limit of how far down a player
// can look. The bot then asks the world a question with its face in the
// floor. Twenty-eight of the twenty-nine ways to stand at AGTST18's exit
// switch were refused for this, and the twenty-ninth was accepted from a
// spot too close to a wall to stand in.
int aimLook(const PhysicalPose &pose, int targetX, int targetY, int targetZ)
{
    // (100 << 7 << 4) / (1 << 14): everything the engine puts between the
    // two axes, and nothing else.
    constexpr double kRayScale = double(100 << 11) / double(1 << 14);
    const BodyShape body = liveBody();
    const int eyeZ = pose.supportZ - body.footOffset - body.eyeAbove;
    const int horizontal = std::max(1,
        planarDistance(pose.x, pose.y, targetX, targetY));
    const double angle = std::atan2(double(eyeZ - targetZ),
                                    kRayScale * double(horizontal))
        * 1024.0 / 3.14159265358979323846;
    return clampInt(int(std::lround(angle)), kLookDownLimit, kLookUpLimit);
}

bool resolvesFrom(const InteractionKey &key, const PhysicalPose &pose,
                  int angle, int look)
{
    if (!gMe || !gMe->pSprite || !validSector(pose.sector))
        return false;
    if (key.tag == kPickupTag)
        return pickupResolvesFrom(key.id, pose);
    spritetype *actor = gMe->pSprite;
    const spritetype savedActor = *actor;
    const fix16_t savedAngle = gMe->q16ang;
    const fix16_t savedLook = gMe->q16look;
    const fix16_t savedHoriz = gMe->q16horiz;
    const int savedSlope = gMe->slope;
    const int savedViewZ = gMe->zView;
    const HITINFO savedHit = gHitInfo;

    const BodyShape body = liveBody();
    actor->x = pose.x;
    actor->y = pose.y;
    actor->z = pose.supportZ - body.footOffset;
    actor->sectnum = int16_t(pose.sector);
    actor->ang = int16_t(angle);
    gMe->q16ang = fix16_from_int(angle);
    gMe->q16look = fix16_from_int(look);
    gMe->q16horiz = fix16_from_float(
        100.f * tanf(float(look) * 3.14159265358979323846f / 1024.f));
    gMe->slope = (-fix16_to_int(gMe->q16horiz)) << 7;
    gMe->zView = actor->z - body.eyeAbove;
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
    return interactionMatches(key, hit, target);
}

bool aimFrom(const InteractionRecord &record, const PhysicalPose &pose,
             int &angle, int &look)
{
    if (!pose.valid || !gMe || !gMe->pSprite)
        return false;
    if (record.key.tag == kPickupTag)
        return false; // nothing is aimed at to be picked up
    if (planarDistance(pose.x, pose.y, record.x, record.y) >= kEngineReach)
        return false;
    const int facing = aimAngle(pose, record.x, record.y);
    const int base = aimLook(pose, record.x, record.y, record.targetZ);
    for (int adjustment = -96; adjustment <= 96; adjustment += 32)
    {
        const int candidate = clampInt(base + adjustment, kLookDownLimit,
                                       kLookUpLimit);
        if (!resolvesFrom(record.key, pose, facing, candidate))
            continue;
        angle = facing;
        look = candidate;
        return true;
    }
    return false;
}

void executionDomain(const InteractionRecord &record,
                     std::vector<ExecutionPose> &out, DomainAudit *audit)
{
    out.clear();
    if (audit)
        *audit = DomainAudit();
    if (!gMe || !gMe->pSprite)
        return;

    if (record.key.tag == kPickupTag)
    {
        // Standing on it is the first thing to try, because that is what a
        // player does. Everything else the engine would also accept lies
        // within its own reach of that, so the ring below stays inside it.
        std::vector<PhysicalPose> candidates;
        candidates.push_back(supportAt(record.x, record.y, record.targetZ,
                                       record.referenceSector));
        for (int distance = 256; distance <= 768; distance += 256)
            for (int angle = 0; angle < kAng360; angle += 256)
                candidates.push_back(supportAt(
                    record.x - mulscale30(distance, Cos(angle)),
                    record.y - mulscale30(distance, Sin(angle)),
                    record.targetZ, record.referenceSector));
        for (const PhysicalPose &stance : candidates)
        {
            if (!stance.valid || !pickupResolvesFrom(record.key.id, stance))
                continue;
            ExecutionPose accepted;
            accepted.pose = stance;
            accepted.angle = aimAngle(stance, record.x, record.y);
            accepted.look = aimLook(stance, record.x, record.y,
                                    record.targetZ);
            out.push_back(accepted);
            if (out.size() >= 8)
                break;
        }
        return;
    }

    std::vector<PhysicalPose> stances;
    auto addStance = [&](const PhysicalPose &pose) {
        if (!pose.valid)
            return;
        for (const PhysicalPose &known : stances)
            if (known.supportHit == pose.supportHit
                && planarDistanceSquared(known.x, known.y, pose.x, pose.y)
                    <= 64 * 64)
                return;
        stances.push_back(pose);
    };

    // Stand off the target at a spread of distances inside the engine's own
    // reach, all around it. Where the interaction was seen on a surface, the
    // stance normal to that surface is tried first, because that is the ray
    // the engine has to trace.
    const int distances[] = { 640, 384, 896 };
    // Straight out in front of the thing, and straight out behind it, at a
    // spread of distances. This is the whole of the answer for anything flat
    // on a surface, and without it such a thing is left to the ring below --
    // which walks around it at sixteen angles and keeps whichever of them
    // the engine happened to accept, wall or no wall. AGTST18's exit switch
    // came out of that with exactly one pose, a hundred units off the wall
    // beside it, which is inside the body's own width; the bot drove at it
    // for the rest of the level.
    if (record.facingAngle >= 0)
    {
        for (int distance : distances)
            for (int side = -1; side <= 1; side += 2)
            {
                const int reach = side * distance;
                addStance(supportAt(
                    record.x + mulscale30(reach, Cos(record.facingAngle)),
                    record.y + mulscale30(reach, Sin(record.facingAngle)),
                    record.targetZ, record.referenceSector));
                // And a little to either side of straight on, for a thing
                // whose front is not quite clear.
                for (int sway = -256; sway <= 256; sway += 512)
                    addStance(supportAt(
                        record.x + mulscale30(reach,
                            Cos(record.facingAngle + sway)),
                        record.y + mulscale30(reach,
                            Sin(record.facingAngle + sway)),
                        record.targetZ, record.referenceSector));
            }
    }
    if (validWall(record.facingWall)
        && validWall(wall[record.facingWall].point2))
    {
        const walltype &a = wall[record.facingWall];
        const walltype &b = wall[a.point2];
        const int dx = b.x - a.x;
        const int dy = b.y - a.y;
        const int length = std::max(1, planarDistance(a.x, a.y, b.x, b.y));
        for (int distance : distances)
            for (int fraction = 1; fraction <= 3; ++fraction)
            {
                const int onX = a.x + int(int64_t(dx) * fraction / 4);
                const int onY = a.y + int(int64_t(dy) * fraction / 4);
                for (int side = -1; side <= 1; side += 2)
                    addStance(supportAt(
                        onX + side * int(int64_t(-dy) * distance / length),
                        onY + side * int(int64_t(dx) * distance / length),
                        record.targetZ, record.referenceSector));
            }
    }
    for (int distance : distances)
        for (int angle = 0; angle < kAng360; angle += 128)
            addStance(supportAt(
                record.x - mulscale30(distance, Cos(angle)),
                record.y - mulscale30(distance, Sin(angle)),
                record.targetZ, record.referenceSector));

    // Nearest the actor first. This decides the order of the list and
    // nothing else -- every pose the engine accepts is kept, below.
    const int actorX = gMe->pSprite->x;
    const int actorY = gMe->pSprite->y;
    std::sort(stances.begin(), stances.end(),
        [&](const PhysicalPose &left, const PhysicalPose &right) {
            return std::make_tuple(
                       planarDistanceSquared(actorX, actorY, left.x, left.y),
                       left.x, left.y)
                < std::make_tuple(
                       planarDistanceSquared(actorX, actorY, right.x, right.y),
                       right.x, right.y);
        });

    if (audit)
        audit->candidates = int(stances.size());
    for (const PhysicalPose &stance : stances)
    {
        if (planarDistance(stance.x, stance.y, record.x, record.y)
            >= kEngineReach)
            continue;
        if (audit)
            ++audit->inReach;
        int angle = 0;
        int look = 0;
        if (aimFrom(record, stance, angle, look))
        {
            if (audit)
                ++audit->accepted;
            ExecutionPose accepted;
            accepted.pose = stance;
            accepted.angle = angle;
            accepted.look = look;
            out.push_back(accepted);
        }
    }
}

} // namespace bloodmap
