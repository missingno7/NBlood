#include "blood_terrain.h"

#include "triggers.h"

#include <algorithm>
#include <map>

#include "fix16.h"
#include "build.h"
#include "../../../actor.h"
#include "../../../blood.h"
#include "../../../common_game.h"
#include "../../../db.h"
#include "../../../globals.h"
#include "../../../player.h"
#include "../../../trig.h"
#include "blood_physics.h"

namespace bloodmap {

namespace {

using semantic::Loop;
using semantic::Plane;
using semantic::Vec2;

std::vector<uint64_t> gSectorHash;
int gObstructedSeams = 0;
struct ObstructedSeam { semantic::Vec2 from, to; int sprite, owner, behind; };
std::vector<ObstructedSeam> gObstructedWhere;

uint64_t mixHash(uint64_t value, uint64_t item)
{
    value ^= item + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
    return value;
}

// A height field fitted to what the engine reports at three points of the
// surface. A flat sector gives a flat plane; a sloped one gives its slope,
// with no special case anywhere above this line.
Plane planeFromSamples(int x1, int y1, int z1, int x2, int y2, int z2,
                       int x3, int y3, int z3)
{
    const int64_t det = int64_t(x2 - x1) * (y3 - y1)
        - int64_t(x3 - x1) * (y2 - y1);
    if (det == 0)
        return semantic::flatPlane(z1);
    const int64_t dz2 = int64_t(z2) - z1;
    const int64_t dz3 = int64_t(z3) - z1;
    Plane plane;
    plane.dzdx = (dz2 * (y3 - y1) - dz3 * (y2 - y1))
        * semantic::kPlaneScale / det;
    plane.dzdy = (dz3 * (x2 - x1) - dz2 * (x3 - x1))
        * semantic::kPlaneScale / det;
    plane.z0 = int64_t(z1) * semantic::kPlaneScale - plane.dzdx * x1
        - plane.dzdy * y1;
    return plane;
}

Plane surfacePlane(int sectorId, bool floor)
{
    const int first = sector[sectorId].wallptr;
    const int count = sector[sectorId].wallnum;
    if (count < 3)
        return semantic::flatPlane(floor ? sector[sectorId].floorz
                                         : sector[sectorId].ceilingz);
    const bool sloped = (floor ? sector[sectorId].floorstat
                               : sector[sectorId].ceilingstat) & 2;
    if (!sloped)
        return semantic::flatPlane(floor ? sector[sectorId].floorz
                                         : sector[sectorId].ceilingz);
    const walltype &a = wall[first];
    const walltype &b = wall[first + count / 3];
    const walltype &c = wall[first + (2 * count) / 3];
    auto height = [&](int x, int y) {
        return floor ? getflorzofslope(sectorId, x, y)
                     : getceilzofslope(sectorId, x, y);
    };
    return planeFromSamples(a.x, a.y, height(a.x, a.y), b.x, b.y,
                            height(b.x, b.y), c.x, c.y, height(c.x, c.y));
}

void sectorLoops(int sectorId, std::vector<Loop> &loops)
{
    loops.clear();
    const int first = sector[sectorId].wallptr;
    const int count = sector[sectorId].wallnum;
    std::vector<char> seen(size_t(std::max(0, count)), 0);
    for (int offset = 0; offset < count; ++offset)
    {
        if (seen[size_t(offset)])
            continue;
        Loop loop;
        int wallId = first + offset;
        int guard = count + 2;
        while (guard-- > 0)
        {
            const int index = wallId - first;
            if (index < 0 || index >= count || seen[size_t(index)])
                break;
            seen[size_t(index)] = 1;
            loop.push_back({ wall[wallId].x, wall[wallId].y });
            wallId = wall[wallId].point2;
            if (wallId == first + offset)
                break;
        }
        if (loop.size() >= 3)
            loops.push_back(loop);
    }
}

// A blocking sprite standing on the ground is a hole in it. Build clips a
// face sprite as a circle of its clip distance and a wall sprite as the line
// it is drawn on; both are represented here by the smallest rectangle that
// covers what the engine clips.
bool obstacleFootprint(int spriteId, Loop &out)
{
    const spritetype &record = sprite[spriteId];
    if (!(record.cstat & 1))
        return false;
    const int alignment = record.cstat & 48;
    if (alignment == 32)
        return false; // floor aligned: a surface, not an obstacle
    const int reach = std::max(64, int(record.clipdist) << 2);
    if (alignment == 16)
    {
        const int half = (tilesiz[record.picnum].x * record.xrepeat) / 8;
        const int cosang = Cos(record.ang) >> 16;
        const int sinang = Sin(record.ang) >> 16;
        // A wall sprite clips as its line; give that line the engine's own
        // clip distance as thickness so it is an area rather than nothing.
        const int dx = -sinang * half / 16384;
        const int dy = cosang * half / 16384;
        const int nx = cosang * reach / 16384;
        const int ny = sinang * reach / 16384;
        out = { { record.x - dx - nx, record.y - dy - ny },
                { record.x + dx - nx, record.y + dy - ny },
                { record.x + dx + nx, record.y + dy + ny },
                { record.x - dx + nx, record.y - dy + ny } };
        return true;
    }
    out = { { record.x - reach, record.y - reach },
            { record.x + reach, record.y - reach },
            { record.x + reach, record.y + reach },
            { record.x - reach, record.y + reach } };
    return true;
}

// A floor-aligned blocking sprite is a surface a body stands on. Its
// footprint is the rectangle Build draws and clips it as.
bool supportFootprint(int spriteId, Loop &out)
{
    const spritetype &record = sprite[spriteId];
    if (!(record.cstat & 1) || (record.cstat & 48) != 32)
        return false;
    const int halfX = (tilesiz[record.picnum].x * record.xrepeat) / 8;
    const int halfY = (tilesiz[record.picnum].y * record.yrepeat) / 8;
    if (halfX <= 0 || halfY <= 0)
        return false;
    const int cosang = Cos(record.ang) >> 16;
    const int sinang = Sin(record.ang) >> 16;
    auto corner = [&](int u, int v) {
        return Vec2{ record.x + (u * sinang - v * cosang) / 16384,
                     record.y + (v * sinang + u * cosang) / 16384 };
    };
    out = { corner(-halfX, -halfY), corner(halfX, -halfY),
            corner(halfX, halfY), corner(-halfX, halfY) };
    return true;
}

// How much room there is over a surface, in the only terms that change what
// can happen there. Blood gives Caleb one hull -- crouching lowers the view
// and nothing else -- so this engine never reports Crouching. The class is
// still what the world is described in, because whether a passage forces a
// crouch is a property of the passage.
semantic::ClearanceClass clearanceFor(int freeHeight, const BodyShape &body)
{
    if (freeHeight >= body.height)
        return semantic::ClearanceClass::Standing;
    if (!gMe)
        return semantic::ClearanceClass::None;
    const POSTURE &standing = gMe->pPosture[gMe->lifeMode][kPostureStand];
    const POSTURE &crouching = gMe->pPosture[gMe->lifeMode][kPostureCrouch];
    const int crouched = body.height
        - (standing.eyeAboveZ - crouching.eyeAboveZ);
    if (crouched > 0 && freeHeight >= crouched)
        return semantic::ClearanceClass::Crouching;
    return semantic::ClearanceClass::None;
}

semantic::Hazard hazardOf(int sectorId)
{
    semantic::Hazard hazard;
    const int extra = sector[sectorId].extra;
    if (extra <= 0 || extra >= kMaxXSectors)
        return hazard;
    const XSECTOR &record = xsector[extra];
    hazard.harmful = record.damageType != 0 || record.Depth != 0;
    hazard.submerged = record.Underwater != 0;
    // Whether this space is being moved through right now, asked of the
    // engine's own list of what it is animating. A sliding sector carries
    // its walls across the ground it stands on, and a body that walks in
    // while that is happening is somewhere a wall is going to be. There is
    // no clock in this: gBusy holds exactly the sectors mid-motion, and it
    // empties itself when they stop.
    // Here, or next door.
    //
    // Walls belong to two sectors and travel through both. A sliding sector
    // sweeps its wall into the room beside it, so the room beside it is the
    // dangerous place to be even though nothing about that room is moving --
    // which is exactly where the bot kept dying on AGTST18, once standing
    // still and once walking through. Asking only about this sector is
    // asking about the wrong side of the wall.
    //
    // This is a fact about right now: gBusy empties itself when the motion
    // stops, so the room stops being dangerous by itself.
    auto busy = [](int sector) {
        for (int index = 0; index < gBusyCount; ++index)
            if (gBusy[index].at0 == sector)
                return true;
        return false;
    };
    if (busy(sectorId))
        hazard.shifting = true;
    else
    {
        const int first = sector[sectorId].wallptr;
        for (int offset = 0; offset < sector[sectorId].wallnum; ++offset)
        {
            const int wallId = first + offset;
            if (!validWall(wallId))
                continue;
            const int behind = wall[wallId].nextsector;
            if (validSector(behind) && busy(behind))
            {
                hazard.shifting = true;
                break;
            }
        }
    }
    // And whether the floor itself travels. Blood's own condition, from the
    // place it applies it: a pan velocity, and the sector switched on or
    // mid-motion. A body standing on one of these is going somewhere with
    // no say in the matter, which is worth knowing before deciding that
    // standing still is a way to pass the time.
    hazard.carrying = record.panVel != 0
        && (record.panAlways || record.state || record.busy);
    return hazard;
}

// How much of an opening a solid thing covers.
//
// Not "is it near the opening" -- a thing hung over one end of a wide
// doorway leaves the rest of it open, and calling the whole thing shut is
// the same untruth as calling it open. What Build clips is a square for a
// face sprite and a plane for a wall sprite; both project onto the opening
// as an interval, and that interval is what is shut.
//
// Returns false when the thing does not reach the opening at all.
bool obstructedSpan(int spriteId, const semantic::Vec2 &from,
                    const semantic::Vec2 &to, double &firstOut,
                    double &lastOut)
{
    const spritetype &thing = sprite[spriteId];
    if (!(thing.cstat & 1))
        return false;
    const int alignment = thing.cstat & 48;
    if (alignment == 32)
        return false;   // floor aligned: a surface, not a barrier
    const int reach = std::max(64, int(thing.clipdist) << 2);
    // How far the thing extends along its own facing. A face sprite is a
    // square of its clip distance; a wall sprite is as wide as it is drawn.
    const int spread = alignment == 16
        ? std::max(reach, (tilesiz[thing.picnum].x * thing.xrepeat) / 8)
        : reach;

    const double dx = double(to.x) - from.x;
    const double dy = double(to.y) - from.y;
    const double length = std::sqrt(dx * dx + dy * dy);
    if (length < 1.0)
        return false;
    const double ux = dx / length;
    const double uy = dy / length;
    const double ox = double(thing.x) - from.x;
    const double oy = double(thing.y) - from.y;
    const double along = ox * ux + oy * uy;
    const double across = std::abs(ox * uy - oy * ux);
    if (across > reach)
        return false;   // it stands beside the opening, not in it
    const double first = std::max(0.0, along - spread);
    const double last = std::min(length, along + spread);
    if (last <= first)
        return false;
    firstOut = first / length;
    lastOut = last / length;
    return true;
}

// What can move this surface, or nothing.
//
// Not "does this sector have an XSECTOR". Most of them do not move: a sector
// carries extra data to be a trigger, to count things, to hurt whoever
// stands in it, to light itself. Blood moves a sector's floor only for the
// handful of types that are motion, and only those are a reason for the
// space either side of a join to be different space. Reading the record
// itself as movement splits one room in two because half of it happens to
// be a trigger.
uint64_t moverOf(int sectorId)
{
    if (sector[sectorId].extra <= 0)
        return 0;
    switch (sector[sectorId].type)
    {
    case kSectorZMotion:
    case kSectorZMotionSprite:
    case kSectorTeleport:
    case kSectorPath:
    case kSectorRotateStep:
    case kSectorSlideMarked:
    case kSectorRotateMarked:
    case kSectorSlide:
    case kSectorRotate:
        return uint64_t(sector[sectorId].extra);
    default:
        return 0;
    }
}

} // namespace


void extractTerrain(TerrainSnapshot &out)
{
    out.faces.clear();
    out.seams.clear();
    gObstructedWhere.clear();
    out.faceSector.clear();
    if (numsectors <= 0)
        return;

    const BodyShape body = liveBody();
    std::vector<int> faceOfSector(size_t(numsectors), -1);
    // Floor-aligned solid sprites are surfaces to stand on. They are also
    // ceilings for whatever is under them, which is a different fact and is
    // put back below.
    struct Overhead
    {
        Loop footprint;
        int underside = 0;
        size_t face = 0;
    };
    std::vector<Overhead> overhead;

    for (int sectorId = 0; sectorId < numsectors; ++sectorId)
    {
        std::vector<Loop> loops;
        sectorLoops(sectorId, loops);
        if (loops.empty())
            continue;
        // The outer boundary is the biggest loop, whatever order the engine
        // happens to store them in.
        size_t outer = 0;
        for (size_t i = 1; i < loops.size(); ++i)
            if (std::abs(semantic::signedDoubleArea(loops[i]))
                > std::abs(semantic::signedDoubleArea(loops[outer])))
                outer = i;
        terrain::SupportFace face;
        face.outline = loops[outer];
        for (size_t i = 0; i < loops.size(); ++i)
            if (i != outer)
                face.obstacles.push_back(loops[i]);
        face.support = surfacePlane(sectorId, true);
        face.ceiling = surfacePlane(sectorId, false);
        face.hazard = hazardOf(sectorId);
        face.supportTag = kSectorSupport + uint64_t(sectorId);
        face.stateTag = moverOf(sectorId);
        face.provenance = uint64_t(sectorId);
        // The free volume has to admit the body at all for this to be a
        // piece of standable world. A closed door is not a place.
        face.clearance = clearanceFor(
            sector[sectorId].floorz - sector[sectorId].ceilingz, body);
        faceOfSector[size_t(sectorId)] = int(out.faces.size());
        out.faces.push_back(face);
        out.faceSector.push_back(sectorId);
    }

    for (int spriteId = 0; spriteId < kMaxSprites; ++spriteId)
    {
        if (!validSprite(spriteId))
            continue;
        if (gMe && gMe->pSprite && spriteId == gMe->pSprite->index)
            continue;
        const spritetype &record = sprite[spriteId];
        if (!validSector(record.sectnum))
            continue;
        Loop footprint;
        if (supportFootprint(spriteId, footprint))
        {
            int top = record.z;
            int bottom = record.z;
            GetSpriteExtents(const_cast<spritetype *>(&record), &top, &bottom);
            terrain::SupportFace face;
            face.outline = footprint;
            face.support = semantic::flatPlane(top);
            face.ceiling = surfacePlane(record.sectnum, false);
            face.supportTag = kSpriteSupport + uint64_t(spriteId);
            face.stateTag = record.extra > 0 ? uint64_t(record.extra) : 0;
            face.provenance = 0x100000000ull | uint64_t(spriteId);
            face.clearance = clearanceFor(
                top - surfacePlane(record.sectnum, false).zAt(record.x,
                                                              record.y),
                body);
            Overhead above;
            above.footprint = footprint;
            above.underside = bottom;
            above.face = out.faces.size();
            overhead.push_back(above);
            out.faces.push_back(face);
            out.faceSector.push_back(record.sectnum);
            continue;
        }
        Loop obstacle;
        if (!obstacleFootprint(spriteId, obstacle)
            || faceOfSector[size_t(record.sectnum)] < 0)
            continue;
        // A solid object is a hole in the floor only where the body would
        // actually run into it. Build's own test is on the body's origin
        // against the object's top and bottom, allowing for the hull's floor
        // and ceiling distances -- so a thing hanging from the ceiling, or
        // standing on a ledge overhead, or lying in a pit below, is not a
        // hole in this floor at all.
        terrain::SupportFace &standing =
            out.faces[size_t(faceOfSector[size_t(record.sectnum)])];
        const int surface = standing.support.zAt(record.x, record.y);
        int top = record.z;
        int bottom = record.z;
        GetSpriteExtents(const_cast<spritetype *>(&record), &top, &bottom);
        const int origin = surface - body.footOffset;
        if (origin <= top - body.floorDistance
            || origin >= bottom + body.ceilingDistance)
            continue;
        // One straddling an authored edge is still a hole: the space it takes
        // up is the same space whichever side of that line its middle falls
        // on.
        standing.obstacles.push_back(obstacle);
    }

    // A surface low enough overhead is not a ceiling to duck under, it is a
    // floor the body cannot be on. Blood answers this with GetZRange, which
    // reports whatever is above as the top of the space; here it becomes a
    // hole in the floor below, which is the same statement in the model's own
    // terms and composes for any number of surfaces stacked over one room.
    for (const Overhead &above : overhead)
    {
        for (size_t index = 0; index < out.faces.size(); ++index)
        {
            if (index == above.face)
                continue;
            terrain::SupportFace &below = out.faces[index];
            const Vec2 middle = semantic::centroidOf(above.footprint);
            const int surface = below.support.zAt(middle.x, middle.y);
            if (surface <= above.underside)
                continue; // not above this floor at all
            if (surface - above.underside >= body.height)
                continue; // room enough to stand under it
            if (!semantic::loopsOverlap(below.outline, above.footprint))
                continue;
            below.obstacles.push_back(above.footprint);
        }
    }

    for (int wallId = 0; wallId < numwalls; ++wallId)
    {
        const walltype &record = wall[wallId];
        const int behind = record.nextsector;
        if (!validSector(behind) || !validWall(record.point2))
            continue;
        int owner = -1;
        for (int sectorId = 0; sectorId < numsectors; ++sectorId)
        {
            const int first = sector[sectorId].wallptr;
            if (wallId >= first && wallId < first + sector[sectorId].wallnum)
            {
                owner = sectorId;
                break;
            }
        }
        if (owner < 0 || owner > behind)
            continue; // each shared wall is reported once
        const int left = faceOfSector[size_t(owner)];
        const int right = faceOfSector[size_t(behind)];
        if (left < 0 || right < 0)
            continue;
        terrain::Seam seam;
        seam.left = size_t(left);
        seam.right = size_t(right);
        seam.from = { record.x, record.y };
        seam.to = { wall[record.point2].x, wall[record.point2].y };
        // Blood carries the blocking bit per side, so a wall solid from one
        // direction only is solid here: what matters is that the two spaces
        // are not one open piece. Reading a single side is how a masked wall
        // came to be dissolved into the space it divides.
        seam.solid = (record.cstat & 1) != 0;
        if (validWall(record.nextwall)
            && (wall[record.nextwall].cstat & 1) != 0)
            seam.solid = true;
        // And whatever is standing in it. A solid thing across an opening
        // shuts the opening as surely as a wall does, and Build clips it
        // whether or not anyone thought of it as a door. The opening is
        // still there and the two spaces are still next to each other, so
        // this is recorded apart from the wall's own blocking bit: it says
        // nothing goes through now, not that there is nothing here.
        if (seam.solid)
        {
            out.seams.push_back(seam);
            continue;
        }
        // Cut the opening where things stand in it. What is covered is shut
        // and what is not is open, and both are the same opening.
        struct Cover { double first, last; uint64_t thing; };
        std::vector<Cover> covers;
        for (int side = 0; side < 2; ++side)
        {
            const int sectorId = side == 0 ? owner : behind;
            if (!validSector(sectorId))
                continue;
            for (int spriteId = headspritesect[sectorId]; spriteId >= 0;
                 spriteId = nextspritesect[spriteId])
            {
                if (!validSprite(spriteId))
                    continue;
                if (sprite[spriteId].statnum == kStatDude
                    || sprite[spriteId].statnum == kStatProjectile)
                    continue;
                if (gMe && gMe->pSprite
                    && spriteId == gMe->pSprite->index)
                    continue;
                double first = 0.0;
                double last = 0.0;
                if (!obstructedSpan(spriteId, seam.from, seam.to, first,
                                    last))
                    continue;
                covers.push_back({ first, last,
                                   kSpriteSupport + uint64_t(spriteId) });
            }
        }
        if (covers.empty())
        {
            out.seams.push_back(seam);
            continue;
        }
        std::sort(covers.begin(), covers.end(),
                  [](const Cover &a, const Cover &b)
                  { return a.first < b.first; });
        auto pointAt = [&](double t) {
            return semantic::Vec2{
                seam.from.x + int(std::lround((seam.to.x - seam.from.x) * t)),
                seam.from.y + int(std::lround((seam.to.y - seam.from.y) * t)) };
        };
        double cursor = 0.0;
        for (const Cover &cover : covers)
        {
            const double first = std::max(cursor, cover.first);
            if (first >= cover.last)
                continue;
            if (first > cursor + 0.001)
            {
                terrain::Seam open = seam;
                open.from = pointAt(cursor);
                open.to = pointAt(first);
                out.seams.push_back(open);
            }
            terrain::Seam shut = seam;
            shut.from = pointAt(first);
            shut.to = pointAt(cover.last);
            shut.obstruction = cover.thing;
            out.seams.push_back(shut);
            ++gObstructedSeams;
            gObstructedWhere.push_back({ shut.from, shut.to,
                int(cover.thing - kSpriteSupport), owner, behind });
            cursor = cover.last;
        }
        if (cursor < 1.0 - 0.001)
        {
            terrain::Seam open = seam;
            open.from = pointAt(cursor);
            open.to = seam.to;
            out.seams.push_back(open);
        }
        continue;
    }
}

// Which sectors' standable geometry has changed since last time.
//
// Everything that can make a sector a different place: its own floor and
// ceiling, the state of the walls around it, and whatever solid things stand
// in it. One hash per sector, compared against the last one.
//
// The point of doing it per sector rather than per world is that a crate
// settling in a room the actor has never seen is not a reason to work the
// whole level out again. A single number for the whole map cannot tell the
// difference, and answering "has anything anywhere moved" with "yes" every
// few seconds is a full rebuild every few seconds.
void collectChangedSectors(std::vector<int> &out)
{
    out.clear();
    const size_t count = size_t(std::max(0, int(numsectors)));
    const bool first = gSectorHash.size() != count;
    if (first)
        gSectorHash.assign(count, 0);
    for (int i = 0; i < numsectors; ++i)
    {
        uint64_t hash = 1469598103934665603ULL;
        hash = mixHash(hash, uint32_t(sector[i].floorz));
        hash = mixHash(hash, uint32_t(sector[i].ceilingz));
        hash = mixHash(hash, uint32_t(sector[i].floorstat));
        hash = mixHash(hash, uint32_t(sector[i].ceilingstat));
        if (sector[i].extra > 0 && sector[i].extra < kMaxXSectors)
        {
            const XSECTOR &extra = xsector[sector[i].extra];
            hash = mixHash(hash, uint32_t(extra.state));
            hash = mixHash(hash, uint32_t(extra.locked));
        }
        const int firstWall = sector[i].wallptr;
        for (int w = firstWall; w < firstWall + sector[i].wallnum; ++w)
        {
            if (!validWall(w))
                continue;
            // Where the wall is, not only what it is.
            //
            // A sliding sector moves its walls and changes nothing else, so
            // a hash of states alone says the world is exactly as it was
            // while the room quietly reshapes itself. The bot then walks on
            // a map of where the walls used to be -- into one, on AGTST18,
            // which closed on it and killed it, twelve minutes in, at the
            // same second on every run.
            hash = mixHash(hash, uint32_t(wall[w].x));
            hash = mixHash(hash, uint32_t(wall[w].y));
            hash = mixHash(hash, uint32_t(wall[w].cstat));
            hash = mixHash(hash, uint32_t(uint16_t(wall[w].nextsector)));
            if (wall[w].extra > 0 && wall[w].extra < kMaxXWalls)
            {
                const XWALL &extra = xwall[wall[w].extra];
                hash = mixHash(hash, uint32_t(extra.state));
                hash = mixHash(hash, uint32_t(extra.locked));
            }
        }
        // Anything solid standing in it. Bodies are not terrain.
        for (int spriteId = headspritesect[i]; spriteId >= 0;
             spriteId = nextspritesect[spriteId])
        {
            if (!validSprite(spriteId) || !(sprite[spriteId].cstat & 1))
                continue;
            if (sprite[spriteId].statnum == kStatDude
                || sprite[spriteId].statnum == kStatProjectile
                || (gMe && gMe->pSprite && spriteId == gMe->pSprite->index))
                continue;
            hash = mixHash(hash, uint32_t(sprite[spriteId].x));
            hash = mixHash(hash, uint32_t(sprite[spriteId].y));
            hash = mixHash(hash, uint32_t(sprite[spriteId].z));
            hash = mixHash(hash, uint32_t(sprite[spriteId].cstat));
        }
        if (!first && gSectorHash[size_t(i)] != hash)
            out.push_back(i);
        gSectorHash[size_t(i)] = hash;
    }
}

bool worldSettling()
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

int obstructedSeams() { return gObstructedSeams; }

int obstructedSeamCount() { return int(gObstructedWhere.size()); }
void obstructedSeamAt(int index, int &x1, int &y1, int &x2, int &y2,
                      int &spriteId, int &owner, int &behind)
{
    const ObstructedSeam &seam = gObstructedWhere[size_t(index)];
    x1 = seam.from.x; y1 = seam.from.y; x2 = seam.to.x; y2 = seam.to.y;
    spriteId = seam.sprite; owner = seam.owner; behind = seam.behind;
}

uint64_t worldSignature()
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
    // Anything solid that can move is part of the world's state too: a
    // panel pushed aside opens a way that no sector's numbers record.
    for (int i = 0; i < kMaxSprites; ++i)
    {
        if (!validSprite(i) || !(sprite[i].cstat & 1))
            continue;
        // Bodies are not terrain. Only things that are part of the place
        // count here, or the world would be a different one every tick.
        if (sprite[i].statnum == kStatDude
            || sprite[i].statnum == kStatProjectile
            || (gMe && gMe->pSprite && i == gMe->pSprite->index))
            continue;
        hash = mixHash(hash, uint32_t(sprite[i].x));
        hash = mixHash(hash, uint32_t(sprite[i].y));
        hash = mixHash(hash, uint32_t(sprite[i].z));
        hash = mixHash(hash, uint32_t(sprite[i].cstat));
    }
    if (gMe)
        for (int key = 0; key < int(sizeof(gMe->hasKey)); ++key)
            hash = mixHash(hash, gMe->hasKey[key] ? uint64_t(key + 1) : 0);
    return hash;
}

} // namespace bloodmap
