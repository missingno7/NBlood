#include "bot_debug.h"

#include "triggers.h"

#include <cstdio>

#include "../../../blood.h"
#include "../../../common_game.h"
#include "../../../globals.h"
#include "../../../player.h"

namespace botdebug {

// Is the sector the body is standing in one the engine is animating right
// now, or does it neighbour one? Read straight off gBusy.
static int busyHere()
{
    if (!gMe || !gMe->pSprite)
        return -1;
    const int here = gMe->pSprite->sectnum;
    if (!bloodmap::validSector(here))
        return -1;
    for (int index = 0; index < gBusyCount; ++index)
        if (gBusy[index].at0 == here)
            return 1;
    const int first = sector[here].wallptr;
    for (int offset = 0; offset < sector[here].wallnum; ++offset)
    {
        const int wallId = first + offset;
        if (!bloodmap::validWall(wallId))
            continue;
        const int behind = wall[wallId].nextsector;
        if (!bloodmap::validSector(behind))
            continue;
        for (int index = 0; index < gBusyCount; ++index)
            if (gBusy[index].at0 == behind)
                return 2;
    }
    return gBusyCount > 0 ? 0 : -2;
}

const char *containerName(int tag)
{
    switch (tag)
    {
    case 0: return "wall";
    case 3: return "sprite";
    case 6: return "sector";
    case bloodmap::kPickupTag: return "item";
    default: return "unknown";
    }
}

void describeRegion(const bloodmap::WorldAdapter &adapter,
                    semantic::RegionId id, char *out, size_t size)
{
    bloodmap::RegionProvenance provenance;
    if (id == semantic::kNoId || !adapter.regionProvenance(id, provenance))
    {
        std::snprintf(out, size, "region=none");
        return;
    }
    char sectors[128] = "";
    size_t used = 0;
    for (size_t i = 0; i < provenance.sectors.size() && used + 8 < sizeof(sectors); ++i)
        used += size_t(std::snprintf(sectors + used, sizeof(sectors) - used,
            i ? ",%d" : "%d", provenance.sectors[i]));
    char sprites[64] = "";
    used = 0;
    for (size_t i = 0; i < provenance.sprites.size() && used + 8 < sizeof(sprites); ++i)
        used += size_t(std::snprintf(sprites + used, sizeof(sprites) - used,
            i ? ",%d" : "%d", provenance.sprites[i]));
    std::snprintf(out, size,
        "region=%u mover=%d corners=%d holes=%d barriers=%d "
        "blood_sectors=[%s] blood_sprites=[%s]",
        unsigned(id), provenance.mover, provenance.vertices, provenance.holes,
        provenance.barriers, sectors, sprites);
}

void describeFootprint(const semantic::Region &region, char *out,
                       size_t size)
{
    size_t used = 0;
    used += size_t(std::snprintf(out, size, "outline="));
    for (size_t i = 0; i < region.footprint.size() && used + 24 < size; ++i)
        used += size_t(std::snprintf(out + used, size - used, "%s%d,%d",
            i ? " " : "", region.footprint[i].x, region.footprint[i].y));
}

void describeAffordance(const bloodmap::WorldAdapter &adapter,
                        semantic::AffordanceId id, char *out, size_t size)
{
    bloodmap::AffordanceProvenance provenance;
    if (id == semantic::kNoId || !adapter.affordanceProvenance(id, provenance))
    {
        std::snprintf(out, size, "affordance=none");
        return;
    }
    std::snprintf(out, size, "affordance=%u blood_%s=%d type=%d needs_key=%d tx=%d where=(%d,%d,%d)",
        unsigned(id), containerName(provenance.tag), provenance.id,
        provenance.kind, provenance.requiredKey, provenance.channel,
        provenance.x, provenance.y, provenance.z);
}

void describeRejection(const bloodmap::RejectedSpace &space, char *out,
                       size_t size)
{
    std::snprintf(out, size,
        "blood_sector=%d free_height=%d support=%d ceiling=%d push=%d "
        "wall_push=%d on_enter=%d remote=%d locked=%d state=%d",
        space.sector, space.freeHeight, space.supportZ, space.ceilingZ,
        space.push ? 1 : 0, space.wallPush ? 1 : 0, space.onEnter ? 1 : 0,
        space.remote ? 1 : 0, space.locked ? 1 : 0, space.state);
}

void describeRelation(const bloodmap::WorldAdapter &adapter,
                      const bloodmap::CalebPhysics &physics,
                      const traversal::TraversalModel &traversal,
                      const semantic::SpatialRelation &relation,
                      char *out, size_t size)
{
    char where[160];
    describeRegion(adapter, relation.to, where, sizeof(where));
    // What the probe actually saw, so a wrong verdict can be read rather
    // than guessed at: where it aimed, where it ended, what held it up
    // there and what the far side's floor is at that point.
    const bloodmap::CalebPhysics::Evidence &saw =
        physics.evidenceFor(relation.id);
    std::snprintf(out, size,
        "relation=%u from_region=%u to_region=%u step=%d width=%d blocked=%d"
        " why=%s to_stances=%d to_pieces=%d aimed=(%d,%d) ended=(%d,%d)"
        " held_at=%d floor_at=%d in_sector=%d %s",
        unsigned(relation.id), unsigned(relation.from),
        unsigned(relation.to), relation.verticalStep,
        relation.gateway.width(), relation.blocked ? 1 : 0,
        bloodmap::CalebPhysics::refusalName(
            physics.refusalFor(relation.id)),
        traversal.stancesIn(relation.to), traversal.piecesOf(relation.to),
        saw.aimed.x, saw.aimed.y, saw.ended.x, saw.ended.y,
        saw.found, saw.wanted, saw.container, where);
}

void describeLocalFailure(const exec::LocalFailure &failure, char *out,
                          size_t size)
{
    std::snprintf(out, size,
        "region=%d from=(%d,%d) to=(%d,%d) from_inside=%d to_inside=%d "
        "from_free=%d to_free=%d nodes=%u walls=%u corners=%u holes=%u "
        "openings=%u",
        failure.region == semantic::kNoId ? -1 : int(failure.region),
        failure.from.x, failure.from.y, failure.to.x, failure.to.y,
        failure.fromInside ? 1 : 0, failure.toInside ? 1 : 0,
        failure.fromFree ? 1 : 0, failure.toFree ? 1 : 0,
        unsigned(failure.nodes), unsigned(failure.solid),
        unsigned(failure.corners), unsigned(failure.holes),
        unsigned(failure.openings));
}

void sampleTrajectory(FILE *out, int gameTime, int tick, const GINPUT &issued)
{
    if (!out || !gMe || !gMe->pSprite || !gMe->pXSprite)
        return;
    std::fprintf(out,
        "{\"game_time\":%d,\"tick\":%d,\"x\":%d,\"y\":%d,"
        "\"z\":%d,\"sector\":%d,\"angle\":%d,\"look\":%d,"
        "\"on_ground\":%d,\"crouched\":%d,\"forward\":%d,"
        "\"strafe\":%d,\"turn\":%d,\"jump\":%d,\"use\":%d,"
        "\"health\":%d,\"xvel\":%d,\"yvel\":%d,"
        "\"busy\":%d}\n",
        gameTime, tick, gMe->pSprite->x, gMe->pSprite->y, gMe->pSprite->z,
        int(gMe->pSprite->sectnum), int(gMe->pSprite->ang),
        fix16_to_int(gMe->q16look), gMe->pXSprite->height == 0 ? 1 : 0,
        gMe->posture == kPostureCrouch ? 1 : 0, int(issued.forward),
        int(issued.strafe), fix16_to_int(issued.q16turn),
        issued.buttonFlags.jump ? 1 : 0, issued.keyFlags.action ? 1 : 0,
        int(gMe->pXSprite->health), int(xvel[gMe->pSprite->index]),
        int(yvel[gMe->pSprite->index]), busyHere());
    std::fflush(out);
}

} // namespace botdebug
