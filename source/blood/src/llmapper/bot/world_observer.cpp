//-------------------------------------------------------------------------
// Blood observation and physics queries for the LLMapper bot.
//-------------------------------------------------------------------------
#include "world_observer.h"
#include "../../seq.h"

// Observed actor geometry, owned by this layer.
int gNextSupportId;
int gNextObjectId;
int gNextAffordanceId;
int gObservedCrouchClearance;
int gStandingClearance;
PlayerCollisionShape gObservedCrouchShape;

 SupportRef engineSupport(EngineHandleKind kind, int index)
{
    const EngineHandle handle = { kind, index };
    const auto found = gSupportIds.find(handle);
    if (found != gSupportIds.end())
        return found->second;
    const SupportRef id(gNextSupportId++);
    gSupportIds[handle] = id;
    gSupportHandles[id] = handle;
    return id;
}
 WorldObjectRef engineObject(EngineHandleKind kind, int index)
{
    const EngineHandle handle = { kind, index };
    const auto found = gObjectIds.find(handle);
    if (found != gObjectIds.end())
        return found->second;
    const WorldObjectRef id(gNextObjectId++);
    gObjectIds[handle] = id;
    gObjectHandles[id] = handle;
    return id;
}
llmapper::AffordanceId semanticAffordanceId(
    EngineHandleKind kind, int index, llmapper::ActionKind action)
{
    const EngineAffordanceHandle handle = { { kind, index }, action };
    const auto found = gAffordanceIds.find(handle);
    if (found != gAffordanceIds.end())
        return found->second;
    const llmapper::AffordanceId id(gNextAffordanceId++);
    gAffordanceIds[handle] = id;
    return id;
}
 int interactionOwnerSector(int observedTargetSector,
                                  int immediateReceiverSector)
{
    return observedTargetSector >= 0
        ? observedTargetSector : immediateReceiverSector;
}
llmapper::AffordanceId physicalAffordanceId(
    int wallId, int fromSector, int targetSector, bool wallPush,
    bool sectorPush, bool sectorPushCurrent, int causalReceiver,
    llmapper::ActionKind action)
{
    if (sectorPush && targetSector >= 0)
        return semanticAffordanceId(kEngineSector, targetSector, action);
    if (sectorPushCurrent && fromSector >= 0)
        return semanticAffordanceId(kEngineSector, fromSector, action);
    if (wallPush && targetSector >= 0)
        return semanticAffordanceId(kEngineSector, targetSector, action);
    if (wallPush && wallId >= 0)
        return semanticAffordanceId(kEngineWall, wallId, action);
    if (causalReceiver >= 0)
        return semanticAffordanceId(kEngineSector, causalReceiver, action);
    if (wallId >= 0)
        return semanticAffordanceId(kEngineWall, wallId, action);
    return llmapper::AffordanceId();
}
static EngineHandle engineSupportHandle(SupportRef id)
{
    const auto found = gSupportHandles.find(id);
    return found != gSupportHandles.end()
        ? found->second : EngineHandle{ kEngineSectorFloor, -1 };
}
 EngineHandleKind supportKind(SupportRef id)
{
    return engineSupportHandle(id).kind;
}
 int supportIndex(SupportRef id)
{
    return engineSupportHandle(id).index;
}
 EngineHandle engineObjectHandle(WorldObjectRef id)
{
    const auto found = gObjectHandles.find(id);
    return found != gObjectHandles.end()
        ? found->second : EngineHandle{ kEngineSector, -1 };
}
// Widest the player is.  An opening narrower than this cannot be walked
// through however inviting the geometry looks, so it must not be offered as
// a route -- the bot was planning between columns it could never fit past.
//
// A cheap geometric rejection only.  ClipMove stays the authority on whether
// the body actually gets through.
 int playerPassageWidth()
{
    return llmapper::capability::playerEnvelope().radius * 2;
}
// How far the pitchfork actually reaches, less the margin above.
 int meleeReach()
{
    return llmapper::capability::playerPitchforkReach() - kMeleeReachMargin;
}
// VECTORDATA::maxDist uses zero as the engine sentinel for an unbounded
// vector scan (actFireVector/VectorScan preserve that meaning).  Planner
// range comparisons need an ordinary upper bound, otherwise a Tommy gun's
// native zero rejects every non-zero firing pose.
 int plannerVectorReach(VECTOR_TYPE vectorType)
{
    const int engineReach = gVectorData[vectorType].maxDist;
    return engineReach > 0 ? engineReach : INT32_MAX;
}
// Health below which the run is in trouble: a quarter of what Caleb starts
// with, on Blood's own 16x scale rather than a restatement of it.
 int criticalHealth()
{
    return llmapper::capability::playerStartHealth() / 4;
}
 int wrapAngle(int angle)
{
    return angle & kAngMask;
}
 int angleDelta(int target, int current)
{
    return DANGLE(target, current);
}
 int distance2(int x1, int y1, int x2, int y2)
{
    const int64_t dx = int64_t(x2) - x1;
    const int64_t dy = int64_t(y2) - y1;
    return int(std::min<int64_t>(INT32_MAX, dx * dx + dy * dy));
}
 bool inRange(int value, int low, int high)
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
 bool validXSprite(int extra)
{
    return extra > 0 && extra < kMaxXSprites;
}
// How much height the player can gain by jumping.  Replayed from the
// engine's own impulse and gravity, and it follows Jump Boots, so there is
// exactly one account of Caleb's jump in the bot.
 int playerJumpRiseLimit()
{
    return llmapper::capability::playerJumpApex();
}
// How far the player may step down and still be able to get back up.  This
// is a reversibility question, not a survival one: a drop the bot cannot
// climb out of turns a wrong turn into the end of the run.
 int playerReversibleDrop()
{
    return playerJumpRiseLimit();
}
 int playerStandingClearance();
// The player's body as it is right this frame, mid-animation included.  Only
// the calibration asks this: everything the bot records about the world has
// to be stated against a body that does not change under it, or crouching
// once makes every doorway in the level appear to open and shut again.
 int playerCollisionClearance()
{
    const llmapper::capability::Envelope body = llmapper::capability::playerEnvelope();
    // playerProcess and MoveDude pass these quarter-scaled distances to
    // pushmove_old.  Sector Z values are compared against that collision
    // envelope; the sprite's full visual top-to-bottom height is not the
    // hull Blood uses for passage clearance.
    return std::max(1, body.ceilingDistance + body.floorDistance);
}
 int playerStandingClearance()
{
    return gStandingClearance > 0 ? gStandingClearance
                                  : playerCollisionClearance();
}
// Until the bot has actually crouched, it does not know how short crouching
// makes it.  The posture table's eyeAboveZ is a camera height, not a
// collision envelope, and guessing from it invents clearance the body may
// not have.  Unknown is reported as the standing envelope, which only ever
// makes the bot refuse a gap it might have fitted through.
 int playerCrouchClearance()
{
    const int standing = playerStandingClearance();
    if (gObservedCrouchClearance > 0)
        return std::min(standing, gObservedCrouchClearance);
    return standing;
}
 void playerCollisionDistances(int &ceilingDistance, int &floorDistance)
{
    const llmapper::capability::Envelope body = llmapper::capability::playerEnvelope();
    ceilingDistance = std::max(0, body.ceilingDistance);
    floorDistance = std::max(0, body.floorDistance);
}
 int lookAngleForTarget(int eyeZ, int targetZ, int horizontal)
{
    const double angle = std::atan2(double(eyeZ - targetZ),
                                    double(std::max(1, horizontal)))
        * 1024.0 / 3.14159265358979323846;
    return std::max(llmapper::capability::playerLookDownLimit(),
                    std::min(llmapper::capability::playerLookUpLimit(),
                             int(std::lround(angle))));
}
// Does this sector contain a floor-aligned sprite the player can stand on?
// Asked first because the answer is almost always no, and the surface query
// below is expensive: everywhere else the sector floor is the whole story
// and nothing changes.
static bool sectorHasFloorSprite(int sectorId)
{
    if (!inRange(sectorId, 0, numsectors))
        return false;
    for (int nSprite = headspritesect[sectorId]; nSprite >= 0;
         nSprite = nextspritesect[nSprite])
    {
        const spritetype &record = sprite[nSprite];
        if (!(record.cstat & CSTAT_SPRITE_BLOCK))
            continue;
        const int alignment = record.cstat & CSTAT_SPRITE_ALIGNMENT_MASK;
        if (alignment == CSTAT_SPRITE_ALIGNMENT_FLOOR
            || alignment == CSTAT_SPRITE_ALIGNMENT_SLOPE)
            return true;
    }
    return false;
}
// What the player would be standing on here, as the engine resolves it.
//
// A bridge made of floor sprites is a floor: reading the sector below it
// instead sees the pit it spans, so the crossing looks like a fatal drop and
// the route over it does not exist.  The probe is deliberately narrow -- the
// surface under this point, not any surface within arm's reach -- so the
// mesh does not place a cell in mid-air off the end of a plank.
static int standingFloorZ(int sectorId, int x, int y)
{
    const int sectorFloor = inRange(sectorId, 0, numsectors)
        ? getflorzofslope(sectorId, x, y) : 0;
    if (!sectorHasFloorSprite(sectorId))
        return sectorFloor;
    int ceilZ = 0;
    int ceilHit = 0;
    int floorZ = 0;
    int floorHit = 0;
    engineGetZRangeIgnoringActor(
        x, y, getceilzofslope(sectorId, x, y) + 1, sectorId,
        &ceilZ, &ceilHit, &floorZ, &floorHit, 4, CLIPMASK0, 0);
    if ((floorHit & 0xc000) == 0xc000 && floorZ < sectorFloor)
        return floorZ;
    return sectorFloor;
}
 PlayerCollisionShape livePlayerCollisionShape()
{
    PlayerCollisionShape shape;
    const llmapper::capability::Envelope body =
        llmapper::capability::playerEnvelope();
    if (!body.known || !gMe || !gMe->pSprite)
        return shape;
    shape.known = true;
    shape.radius = body.radius;
    shape.ceilingDistance = std::max(0, body.ceilingDistance);
    shape.floorDistance = std::max(0, body.floorDistance);
    shape.footOffset = body.bottom - gMe->pSprite->z;
    return shape;
}

PlayerCollisionCycle playerMovingCollisionCycle()
{
    PlayerCollisionCycle result;
    if (!gMe || !gMe->pSprite || !gMe->pXSprite
        || !gMe->pDudeInfo)
        return result;
    const int sequenceId = gMe->pDudeInfo->seqStartID + 8;
    DICTNODE *resource = gSysRes.Lookup(sequenceId, "SEQ");
    if (!resource)
        return result;
    Seq *sequence = static_cast<Seq *>(gSysRes.Lock(resource));
    if (!sequence)
        return result;
    if (memcmp(sequence->signature, "SEQ\x1a", 4) != 0
        || (sequence->version & 0xff00) != 0x300
        || sequence->nFrames <= 0 || sequence->nFrames > 4096)
    {
        gSysRes.Unlock(resource);
        return result;
    }

    result.ticksPerFrame = std::max(1, int(sequence->ticksPerFrame));
    result.looping = (sequence->flags & 1) != 0;
    const int scale = gMe->pXSprite->scale;
    int sampleYRepeat = gMe->pSprite->yrepeat;
    const int sampleZ = gMe->pSprite->z;
    result.frames.reserve(std::max(0, int(sequence->nFrames)));
    for (int index = 0; index < sequence->nFrames; ++index)
    {
        SEQFRAME &frame = sequence->frames[index];
        const int tile = seqGetTile(&frame);
        if (!inRange(tile, 0, MAXTILES))
            continue;
        if (frame.yrepeat)
            sampleYRepeat = scale
                ? ClipRange(mulscale8(frame.yrepeat, scale), 0, 255)
                : frame.yrepeat;
        const int height = tilesiz[tile].y;
        const int center = height / 2 + picanm[tile].yofs;
        const int top = sampleZ - (sampleYRepeat << 2) * center;
        const int bottom = sampleZ
            + (sampleYRepeat << 2) * (height - center);
        PlayerCollisionShape shape;
        shape.known = true;
        shape.radius = gMe->pSprite->clipdist << 2;
        shape.ceilingDistance = std::max(0, (sampleZ - top) / 4);
        shape.floorDistance = std::max(0, (bottom - sampleZ) / 4);
        shape.footOffset = bottom - sampleZ;
        result.frames.push_back(shape);
    }
    gSysRes.Unlock(resource);
    return result;
}

static llmapper::capability::GroundContactMotion groundContactMotion(
    int floorHit, int sectorId)
{
    llmapper::capability::GroundContactMotion motion;
    if ((floorHit & 0xc000) == 0xc000)
    {
        const int spriteId = floorHit & 0x3fff;
        if (inRange(spriteId, 0, kMaxSprites)
            && (sprite[spriteId].cstat & 0x30) == 0)
        {
            motion.radialSpriteResponse = true;
            motion.skipGroundDrag = true;
            motion.supportX = sprite[spriteId].x;
            motion.supportY = sprite[spriteId].y;
            return motion;
        }
    }
    if (inRange(sectorId, 0, numsectors))
    {
        const int extra = sector[sectorId].extra;
        motion.skipGroundDrag = extra > 0 && extra < kMaxXSectors
            && xsector[extra].Underwater;
    }
    return motion;
}

// Engine movement normally evaluates a sprite while that sprite is absent
// from its own collision set. A copied hypothetical player origin must do the
// same: otherwise the future path can collide with the actor's present-world
// sprite and turn a valid walk (or air arc) into false blocked evidence.
int engineClipMoveIgnoringActor(
    vec3_t &position, int16_t &sectorNumber,
    int32_t xvect, int32_t yvect, int radius,
    int ceilingDistance, int floorDistance, unsigned clipMask)
{
    spritetype *actor = gMe ? gMe->pSprite : nullptr;
    const int savedCstat = actor ? actor->cstat : 0;
    if (actor)
        // MoveDude removes both blocking identities (the exact engine mask
        // is 257) while moving a dude.  Hypothetical movement must make the
        // live actor equally absent: leaving the hitscan-blocking bit set can
        // make the copied future body collide with its present-world self.
        actor->cstat &= ~257;
    const int hit = clipmove(&position, &sectorNumber, xvect, yvect, radius,
                             ceilingDistance, floorDistance, clipMask);
    if (actor)
        actor->cstat = int16_t(savedCstat);
    return hit;
}

// A copied hypothetical player pose must be absent from its own collision
// set for both halves of a movement query, not just for ClipMove.
void engineGetZRangeIgnoringActor(
    int x, int y, int z, int sectorNumber,
    int *ceilingZ, int *ceilingHit, int *floorZ, int *floorHit,
    int radius, unsigned clipMask, int flags)
{
    spritetype *actor = gMe ? gMe->pSprite : nullptr;
    const int savedCstat = actor ? actor->cstat : 0;
    if (actor)
        // Match MoveDude's self-exclusion for the vertical half of the same
        // hypothetical pose query as well.  This is one collision contract,
        // not a planner exception for sprite supports.
        actor->cstat &= ~257;
    GetZRangeAtXYZ(x, y, z, sectorNumber,
                   ceilingZ, ceilingHit, floorZ, floorHit,
                   radius, clipMask, flags);
    if (actor)
        actor->cstat = int16_t(savedCstat);
}

PlayerGroundObservation enginePlayerGroundObservationAt(
    int x, int y, int z, int sectorNumber, int footOffset)
{
    PlayerGroundObservation result;
    if (!gMe || !gMe->pSprite || !inRange(sectorNumber, 0, numsectors))
        return result;

    int ceilingZ = 0, ceilingHit = 0;
    int floorHit = 0;
    const int clipRadius = gMe->pSprite->clipdist << 2;
    const int expandedRadius = clipRadius + 16;
    engineGetZRangeIgnoringActor(
        x, y, z, sectorNumber, &ceilingZ, &ceilingHit,
        &result.floorZ, &floorHit, expandedRadius, CLIPMASK0,
        PARALLAXCLIP_CEILING | PARALLAXCLIP_FLOOR);

    // MoveDude performs this second, parallax-clipped query for players at a
    // floor and conditionally keeps the first result.  Reproduce that choice
    // here so callers receive the same support Blood will use later in the
    // frame, without interpreting floor-hit bits above the observer.
    const int foot = footOffset >= 0
        ? footOffset : llmapper::capability::playerFootOffset();
    if (z + foot >= result.floorZ)
    {
        const int firstFloorZ = result.floorZ;
        const int firstFloorHit = floorHit;
        int parallaxFloorZ = 0, parallaxFloorHit = 0;
        engineGetZRangeIgnoringActor(
            x, y, z, sectorNumber, &ceilingZ, &ceilingHit,
            &parallaxFloorZ, &parallaxFloorHit, clipRadius, CLIPMASK0,
            PARALLAXCLIP_CEILING | PARALLAXCLIP_FLOOR);
        result.floorZ = parallaxFloorZ;
        floorHit = parallaxFloorHit;
        if (z + foot <= parallaxFloorZ
            && z - firstFloorZ < foot / 4)
        {
            result.floorZ = firstFloorZ;
            floorHit = firstFloorHit;
        }
    }

    result.motion = groundContactMotion(floorHit, sectorNumber);
    if ((floorHit & 0xc000) == 0xc000)
        result.support = engineSupport(kSupportSpriteFloor,
                                       floorHit & 0x3fff);
    else
        result.support = engineSupport(kSupportSectorFloor,
                                       (floorHit & 0xc000) == 0x4000
                                           ? floorHit & 0x3fff
                                           : sectorNumber);
    return result;
}

static std::vector<int> engineWallCoordinateFrame(int startRegion)
{
    std::vector<int> result;
    if (!inRange(startRegion, 0, numsectors))
        return result;
    std::vector<char> seen(size_t(numsectors), 0);
    std::deque<int> pending;
    pending.push_back(startRegion);
    seen[size_t(startRegion)] = 1;
    while (!pending.empty())
    {
        const int current = pending.front();
        pending.pop_front();
        result.push_back(current);
        const sectortype &record = sector[current];
        for (int index = 0; index < record.wallnum; ++index)
        {
            const int wallId = record.wallptr + index;
            if (!inRange(wallId, 0, numwalls))
                continue;
            const int neighbour = wall[wallId].nextsector;
            if (!inRange(neighbour, 0, numsectors)
                || seen[size_t(neighbour)])
                continue;
            seen[size_t(neighbour)] = 1;
            pending.push_back(neighbour);
        }
    }
    return result;
}

std::vector<EngineSpaceTranslation> engineSpaceTranslations()
{
    std::vector<EngineSpaceTranslation> result;
    for (int upperRegion = 0; upperRegion < numsectors; ++upperRegion)
    {
        const int upperSprite = gUpperLink[upperRegion];
        if (!inRange(upperSprite, 0, kMaxSprites))
            continue;
        const int upperType = sprite[upperSprite].type;
        // Water and goo links change posture/medium and are not ordinary
        // dry-space translations.  They need their own physical operation.
        if (upperType != kMarkerUpLink && upperType != kMarkerUpStack)
            continue;
        const int lowerSprite = sprite[upperSprite].owner;
        if (!inRange(lowerSprite, 0, kMaxSprites))
            continue;
        const int lowerType = sprite[lowerSprite].type;
        if (lowerType != kMarkerLowLink && lowerType != kMarkerLowStack)
            continue;
        const int lowerRegion = sprite[lowerSprite].sectnum;
        if (!inRange(lowerRegion, 0, numsectors))
            continue;

        EngineSpaceTranslation link;
        link.upperRegion = upperRegion;
        link.lowerRegion = lowerRegion;
        link.upperToLowerX = sprite[lowerSprite].x - sprite[upperSprite].x;
        link.upperToLowerY = sprite[lowerSprite].y - sprite[upperSprite].y;
        link.upperToLowerZ = sprite[lowerSprite].z - sprite[upperSprite].z;
        link.upperAnchorX = sprite[upperSprite].x;
        link.upperAnchorY = sprite[upperSprite].y;
        link.lowerAnchorX = sprite[lowerSprite].x;
        link.lowerAnchorY = sprite[lowerSprite].y;
        link.upperFrameRegions = engineWallCoordinateFrame(upperRegion);
        link.lowerFrameRegions = engineWallCoordinateFrame(lowerRegion);
        result.push_back(link);
    }
    return result;
}

MovementProbe probeMovement(
    int startX, int startY, int startZ, int startSector,
    int targetX, int targetY, int targetSector,
    int tolerance, bool crouched)
{
    MovementProbe result;
    result.x = startX;
    result.y = startY;
    result.sector = startSector;
    if (!inRange(startSector, 0, numsectors)
        || !inRange(targetSector, 0, numsectors))
        return result;

    vec3_t position = { startX, startY, startZ };
    int16_t sectorNumber = int16_t(startSector);
    const int64_t dx = int64_t(targetX) - startX;
    const int64_t dy = int64_t(targetY) - startY;
    const int32_t xvect = int32_t(std::max<int64_t>(
        INT32_MIN + 1, std::min<int64_t>(
            INT32_MAX, dx * (int64_t(1) << 14))));
    const int32_t yvect = int32_t(std::max<int64_t>(
        INT32_MIN + 1, std::min<int64_t>(
            INT32_MAX, dy * (int64_t(1) << 14))));
    const PlayerCollisionShape liveShape = livePlayerCollisionShape();
    const PlayerCollisionShape &shape = crouched && gObservedCrouchShape.known
        ? gObservedCrouchShape : liveShape;
    const int radius = shape.known ? shape.radius : 128;
    int ceilingDistance = 0;
    int floorDistance = 0;
    if (shape.known)
    {
        ceilingDistance = shape.ceilingDistance;
        floorDistance = shape.floorDistance;
    }
    else
        playerCollisionDistances(ceilingDistance, floorDistance);

    result.hit = engineClipMoveIgnoringActor(
        position, sectorNumber, xvect, yvect, radius,
        ceilingDistance, floorDistance, CLIPMASK0);
    result.x = position.x;
    result.y = position.y;
    result.sector = sectorNumber;
    if ((result.hit & 0xc000) == 0x8000)
        result.wall = result.hit & 0x3fff;
    else if ((result.hit & 0xc000) == 0xc000)
        result.sprite = result.hit & 0x3fff;

    const int remaining = distance2(position.x, position.y, targetX, targetY);
    bool equivalentTargetPose = sectorNumber == targetSector;
    if (!equivalentTargetPose && remaining <= tolerance * tolerance
        && inside(position.x, position.y, targetSector) == 1)
    {
        int liveCeiling = 0, liveCeilingHit = 0;
        int liveFloor = 0, liveFloorHit = 0;
        engineGetZRangeIgnoringActor(
            position.x, position.y, position.z, sectorNumber,
            &liveCeiling, &liveCeilingHit, &liveFloor, &liveFloorHit,
            radius, CLIPMASK0,
            PARALLAXCLIP_CEILING | PARALLAXCLIP_FLOOR);
        const int targetFloor = getflorzofslope(
            targetSector, position.x, position.y);
        equivalentTargetPose = std::abs(liveFloor - targetFloor)
            <= llmapper::capability::playerStepHeight() / 2;
    }
    result.reachable = equivalentTargetPose
        && remaining <= tolerance * tolerance;
    return result;
}
// Blood does not use q16look as a geometric weapon pitch. ProcessInput
// derives horiz=100*tan(look), then slope=-horiz*128, and VectorScan applies
// slope at 1/1024 per horizontal map unit. Account for that 12.5x scale for
// ranged activation; otherwise rounds hit the ceiling/floor before reaching
// a vertically displaced target. Keep interaction posing separate because
// ActionScan has its own short-range acquisition behavior.
 int vectorLookAngleForTarget(int eyeZ, int targetZ, int horizontal)
{
    const double angle = std::atan2(double(eyeZ - targetZ) * 8.0,
                                    double(std::max(1, horizontal)) * 100.0)
        * 1024.0 / 3.14159265358979323846;
    return std::max(llmapper::capability::playerLookDownLimit(),
                    std::min(llmapper::capability::playerLookUpLimit(),
                             int(std::lround(angle))));
}

bool engineVectorHitsTargetFromPose(
    int poseX, int poseY, int supportZ, int poseSector,
    int weaponAboveFloor, int aimX, int aimY, int aimZ,
    VECTOR_TYPE vectorType, int targetSprite, int targetWall,
    int alternateTargetWall)
{
    if (!gMe || !gMe->pSprite || !inRange(poseSector, 0, numsectors))
        return false;

    spritetype &origin = *gMe->pSprite;
    const spritetype savedOrigin = origin;
    const HITINFO savedHit = gHitInfo;
    origin.x = poseX;
    origin.y = poseY;
    origin.z = playerOriginAtSupport(supportZ);
    origin.sectnum = int16_t(poseSector);

    const int shotZ = supportZ + weaponAboveFloor;
    const int horizontal = std::max(1, int(std::sqrt(double(distance2(
        poseX, poseY, aimX, aimY)))));
    const int angle = getangle(aimX - poseX, aimY - poseY);
    const int look = vectorLookAngleForTarget(shotZ, aimZ, horizontal);
    int horiz = 0;
    if (VanillaMode())
    {
        if (look > 0)
            horiz = mulscale30(120, Sin(look << 3));
        else if (look < 0)
            horiz = mulscale30(180, Sin(look << 3));
    }
    else
        horiz = int(std::lround(100.f * tanf(
            float(look) * 3.14159265358979323846f / 1024.f)));
    const int slope = -horiz << 7;

    vec3_t adjustedOrigin = {};
    const int hit = !VanillaMode()
        ? VectorScanROR(&origin, 0, shotZ - origin.z,
                        Cos(angle) >> 16, Sin(angle) >> 16, slope,
                        gVectorData[vectorType].maxDist, 1, &adjustedOrigin)
        : VectorScan(&origin, 0, shotZ - origin.z,
                     Cos(angle) >> 16, Sin(angle) >> 16, slope,
                     gVectorData[vectorType].maxDist, 1);
    const int hitSprite = gHitInfo.hitsprite;
    const int hitWall = gHitInfo.hitwall;
    origin = savedOrigin;
    gHitInfo = savedHit;

    if (targetSprite >= 0)
        return hit == 3 && hitSprite == targetSprite;
    return (hit == 0 || hit == 4)
        && (hitWall == targetWall || hitWall == alternateTargetWall);
}

namespace {

struct TntTrajectoryState
{
    int x = 0;
    int y = 0;
    int z = 0;
    int sector = -1;
    int xvel = 0;
    int yvel = 0;
    int zvel = 0;
};

struct TntTrajectoryTrial
{
    int look = 0;
    int power = 0;
    int holdTicks = 0;
    int closestDistance = INT32_MAX;
    int closestFrame = -1;
    int nearFrames = 0;
    bool settledNear = false;
    TntTrajectoryState rest;
};

static int playerSlopeForLook(int look)
{
    if (VanillaMode())
    {
        const int horiz = look > 0
            ? mulscale30(120, Sin(look << 3))
            : look < 0 ? mulscale30(180, Sin(look << 3)) : 0;
        return -horiz << 7;
    }
    const double radians = double(look) * 3.14159265358979323846 / 1024.0;
    return -int(std::lround(100.0 * std::tan(radians))) << 7;
}

static int distanceToTargetVolume(
    const TntTrajectoryState &state, int targetX, int targetY,
    int targetTopZ, int targetBottomZ)
{
    const int low = std::min(targetTopZ, targetBottomZ);
    const int high = std::max(targetTopZ, targetBottomZ);
    const int nearestZ = std::max(low, std::min(high, state.z));
    const int64_t dx = int64_t(state.x) - targetX;
    const int64_t dy = int64_t(state.y) - targetY;
    // Radius damage compares Build Z at 1/16 of XY scale.
    const int64_t dz = (int64_t(state.z) - nearestZ) >> 4;
    return int(std::sqrt(double(dx * dx + dy * dy + dz * dz)));
}

static void applyThingAirDrag(TntTrajectoryState &state)
{
    int windX = 0;
    int windY = 0;
    if (inRange(state.sector, 0, numsectors))
    {
        const int extra = sector[state.sector].extra;
        if (extra > 0 && extra < kMaxXSectors)
        {
            const XSECTOR &record = xsector[extra];
            if (record.windVel && (record.windAlways || record.busy))
            {
                int speed = record.windVel << 12;
                if (!record.windAlways && record.busy)
                    speed = mulscale16(speed, record.busy);
                windX = mulscale30(speed, Cos(record.windAng));
                windY = mulscale30(speed, Sin(record.windAng));
            }
        }
    }
    state.xvel += mulscale16(windX - state.xvel, 128);
    state.yvel += mulscale16(windY - state.yvel, 128);
    state.zvel -= mulscale16(state.zvel, 128);
}

static int stepTntTrajectory(TntTrajectoryState &state, tspritetype &shape)
{
    if (!inRange(state.sector, 0, numsectors))
        return -1;
    const THINGINFO &info = thingInfo[kThingArmedTNTBundle - kThingBase];
    shape.x = state.x;
    shape.y = state.y;
    shape.z = state.z;
    shape.sectnum = int16_t(state.sector);

    applyThingAirDrag(state);
    int hit = 0;
    int top = 0, bottom = 0;
    GetSpriteExtents(&shape, &top, &bottom);
    if (state.xvel || state.yvel)
    {
        int nextSector = state.sector;
        hit = ClipMove(&state.x, &state.y, &state.z, &nextSector,
                       state.xvel >> 12, state.yvel >> 12,
                       shape.clipdist << 2,
                       (shape.z - top) / 4, (bottom - shape.z) / 4,
                       CLIPMASK0);
        state.sector = nextSector;
        if ((hit & 0xc000) == 0x8000)
            actWallBounceVector(&state.xvel, &state.yvel,
                                hit & 0x3fff, info.elastic);
    }
    if (!inRange(state.sector, 0, numsectors))
        return -1;
    if (state.zvel)
        state.z += state.zvel >> 8;

    int ceilingZ = 0, ceilingHit = 0;
    int floorZ = 0, floorHit = 0;
    GetZRangeAtXYZ(state.x, state.y, state.z, state.sector,
                   &ceilingZ, &ceilingHit, &floorZ, &floorHit,
                   shape.clipdist << 2, CLIPMASK0, 0);
    shape.x = state.x;
    shape.y = state.y;
    shape.z = state.z;
    shape.sectnum = int16_t(state.sector);
    GetSpriteExtents(&shape, &top, &bottom);
    if ((shape.flags & 2) && bottom < floorZ)
    {
        state.z += 455;
        state.zvel += 58254;
    }

    int linkedSector = state.sector;
    if (CheckLink(&state.x, &state.y, &state.z, &linkedSector))
    {
        state.sector = linkedSector;
        if (!inRange(state.sector, 0, numsectors))
            return -1;
        GetZRangeAtXYZ(state.x, state.y, state.z, state.sector,
                       &ceilingZ, &ceilingHit, &floorZ, &floorHit,
                       shape.clipdist << 2, CLIPMASK0, 0);
    }
    shape.x = state.x;
    shape.y = state.y;
    shape.z = state.z;
    shape.sectnum = int16_t(state.sector);
    GetSpriteExtents(&shape, &top, &bottom);
    if (bottom >= floorZ)
    {
        state.z += floorZ - bottom;
        int vertical = state.zvel - velFloor[state.sector];
        if (vertical > 0)
        {
            actFloorBounceVector(&state.xvel, &state.yvel, &vertical,
                                 state.sector, info.elastic);
            state.zvel = vertical;
            if (velFloor[state.sector] == 0
                && std::abs(state.zvel) < 0x10000)
                state.zvel = 0;
        }
    }
    if (top <= ceilingZ)
    {
        state.z += std::max(ceilingZ - top, 0);
        if (state.zvel < 0)
        {
            state.xvel = mulscale16(state.xvel, 0xc000);
            state.yvel = mulscale16(state.yvel, 0xc000);
            state.zvel = mulscale16(-state.zvel, 0x4000);
        }
    }
    if (bottom >= floorZ)
    {
        const int velocity = approxDist(state.xvel, state.yvel);
        const int clipped = std::min(velocity, 0x11111);
        if ((floorHit & 0xc000) == 0xc000)
        {
            const int support = floorHit & 0x3fff;
            if (inRange(support, 0, kMaxSprites)
                && (sprite[support].cstat & 0x30) == 0)
            {
                state.xvel += mulscale2(4, state.x - sprite[support].x);
                state.yvel += mulscale2(4, state.y - sprite[support].y);
            }
        }
        if (velocity > 0)
        {
            const int drag = divscale16(clipped, velocity);
            state.xvel -= mulscale16(drag, state.xvel);
            state.yvel -= mulscale16(drag, state.yvel);
        }
    }
    return hit;
}

static TntTrajectoryTrial traceTntTrajectory(
    int angle, int look, int holdTicks,
    int targetSprite,
    int targetX, int targetY, int targetTopZ, int targetBottomZ,
    int targetSector, int damagingRadius)
{
    TntTrajectoryTrial trial;
    trial.look = look;
    trial.holdTicks = holdTicks;
    trial.power = std::min(divscale16(holdTicks, 240), 65536);
    if (!gMe || !gMe->pSprite)
        return trial;

    const THINGINFO &info = thingInfo[kThingArmedTNTBundle - kThingBase];
    tspritetype shape = {};
    shape.picnum = info.picnum;
    shape.cstat = info.cstat;
    shape.clipdist = info.clipdist;
    shape.flags = info.flags;
    shape.xrepeat = info.xrepeat;
    shape.yrepeat = info.yrepeat;
    TntTrajectoryState state;
    state.x = gMe->pSprite->x
        + mulscale28(gMe->pSprite->clipdist, Cos(angle));
    state.y = gMe->pSprite->y
        + mulscale28(gMe->pSprite->clipdist, Sin(angle));
    state.z = gMe->zWeapon;
    state.sector = gMe->pSprite->sectnum;
    const int speed = mulscale16(trial.power, 0x177777) + 0x66666;
    state.xvel = mulscale30(speed, Cos(angle))
        + xvel[gMe->pSprite->index] / 2;
    state.yvel = mulscale30(speed, Sin(angle))
        + yvel[gMe->pSprite->index] / 2;
    state.zvel = mulscale14(speed, playerSlopeForLook(look) - 9460)
        + zvel[gMe->pSprite->index] / 2;

    int consecutiveNear = 0;
    int settledFrames = 0;
    for (int frame = 0; frame < 120; ++frame)
    {
        if (stepTntTrajectory(state, shape) < 0)
            break;
        const int distance = distanceToTargetVolume(
            state, targetX, targetY, targetTopZ, targetBottomZ);
        const int targetZ = std::max(
            std::min(state.z, std::max(targetTopZ, targetBottomZ)),
            std::min(targetTopZ, targetBottomZ));
        const bool affected = inRange(targetSprite, 0, kMaxSprites)
            && sprite[targetSprite].statnum < kMaxStatus
            ? CheckProximity(&sprite[targetSprite], state.x, state.y, state.z,
                             state.sector, std::max(1, damagingRadius >> 4))
            : inRange(targetSector, 0, numsectors)
                && cansee(state.x, state.y, state.z, state.sector,
                          targetX, targetY, targetZ, targetSector)
                && distance <= damagingRadius;
        if (distance < trial.closestDistance)
        {
            trial.closestDistance = distance;
            trial.closestFrame = frame;
        }
        if (affected)
        {
            ++consecutiveNear;
            trial.nearFrames = std::max(trial.nearFrames, consecutiveNear);
        }
        else
            consecutiveNear = 0;

        if (state.xvel == 0 && state.yvel == 0 && state.zvel == 0)
        {
            ++settledFrames;
            if (affected)
                trial.settledNear = true;
            if (settledFrames >= 8)
                break;
        }
        else
            settledFrames = 0;
    }
    trial.rest = state;
    return trial;
}

static bool betterTntTrial(const TntTrajectoryTrial &candidate,
                           const TntTrajectoryTrial &known,
                           int damagingRadius)
{
    const bool candidateValid = candidate.settledNear
        || candidate.nearFrames >= 4;
    const bool knownValid = known.settledNear || known.nearFrames >= 4;
    if (candidateValid != knownValid)
        return candidateValid;
    if (candidate.settledNear != known.settledNear)
        return candidate.settledNear;
    if (candidate.nearFrames != known.nearFrames)
        return candidate.nearFrames > known.nearFrames;
    const int target = damagingRadius / 3;
    const int candidateError = std::abs(candidate.closestDistance - target);
    const int knownError = std::abs(known.closestDistance - target);
    if (candidateError != knownError)
        return candidateError < knownError;
    return candidate.holdTicks < known.holdTicks;
}

} // namespace

TntThrowSolution engineSolveTntThrow(
    int targetSprite,
    int targetX, int targetY, int targetTopZ, int targetBottomZ,
    int targetSector, int damagingRadius)
{
    TntThrowSolution result;
    if (!gMe || !gMe->pSprite || damagingRadius <= 0)
        return result;
    const int angle = getangle(targetX - gMe->pSprite->x,
                               targetY - gMe->pSprite->y);
    TntTrajectoryTrial best;
    bool haveBest = false;
    auto consider = [&](int look, int holdTicks) {
        const TntTrajectoryTrial candidate = traceTntTrajectory(
            angle, look, holdTicks, targetSprite, targetX, targetY,
            targetTopZ, targetBottomZ, targetSector, damagingRadius);
        if (!haveBest || betterTntTrial(candidate, best, damagingRadius))
        {
            best = candidate;
            haveBest = true;
        }
    };

    const int lookLow = llmapper::capability::playerLookDownLimit();
    const int lookHigh = llmapper::capability::playerLookUpLimit();
    for (int look = lookLow; look <= lookHigh; look += 24)
    {
        for (int holdTicks = 0; holdTicks <= 240; holdTicks += 16)
            consider(look, holdTicks);
    }
    if (!haveBest)
        return result;

    const int coarseLook = best.look;
    const int coarseHold = best.holdTicks;
    for (int look = std::max(lookLow, coarseLook - 24);
         look <= std::min(lookHigh, coarseLook + 24); look += 4)
    {
        for (int holdTicks = std::max(0, coarseHold - 16);
             holdTicks <= std::min(240, coarseHold + 16); holdTicks += 4)
            consider(look, holdTicks);
    }

    result.valid = best.settledNear || best.nearFrames >= 4;
    result.angle = angle;
    result.look = best.look;
    result.power = best.power;
    result.holdTicks = best.holdTicks;
    result.closestDistance = best.closestDistance;
    result.closestFrame = best.closestFrame;
    result.restX = best.rest.x;
    result.restY = best.rest.y;
    result.restZ = best.rest.z;
    result.restSector = best.rest.sector;
    return result;
}
// GetSpriteExtents is Blood's visual/gameplay extent helper.  Build collision
// applies the tile y-offset through spriteheightofs(..., 1), which can differ
// by a small but decisive amount: probing the visual top may start just below
// the collision top and make getzrange report the sector floor.  Navigation
// needs the plane Caleb's body collides with.
static void spriteCollisionExtents(int spriteId, int &top, int &bottom)
{
    if (!inRange(spriteId, 0, kMaxSprites))
    {
        top = bottom = 0;
        return;
    }
    int height = 0;
    bottom = sprite[spriteId].z + spriteheightofs(spriteId, &height, 1);
    top = bottom - height;
}
// Enumerate every floor the engine can resolve at one XY, probing from just
// above each candidate plane.  One query from the sector ceiling only finds
// the topmost sprite and collapses stacked bridges; probing each known plane
// preserves every physically distinct support while still leaving GetZRange
// authoritative on footprint, one-sidedness, slope and clipping.  Build's
// getzrange treats the top extent of blocking face/wall sprites as a floor as
// well as floor-aligned sprites.  Use that same physical answer here instead
// of maintaining a narrower orientation-specific idea of support.
 std::vector<StandableSurface> standableSurfacesAt(
    int sectorId, int x, int y, int radius, int requiredClearance)
{
    std::vector<StandableSurface> result;
    if (!inRange(sectorId, 0, numsectors))
        return result;

    auto appendSurface = [&](const SupportRef &support, int surfaceZ,
                             int ceilingZ) {
        for (const StandableSurface &known : result)
            if (known.support == support && known.z == surfaceZ)
                return;
        StandableSurface surface;
        surface.support = support;
        surface.z = surfaceZ;
        surface.ceilingZ = ceilingZ;
        result.push_back(surface);
    };

    auto probe = [&](const SupportRef &support, int surfaceZ, int expectedHit) {
        int ceilingZ = 0, ceilingHit = 0, floorZ = 0, floorHit = 0;
        engineGetZRangeIgnoringActor(
            x, y, surfaceZ - 1, sectorId,
            &ceilingZ, &ceilingHit, &floorZ, &floorHit,
            radius, CLIPMASK0,
            PARALLAXCLIP_CEILING | PARALLAXCLIP_FLOOR);
        if (floorZ != surfaceZ)
            return;
        if (expectedHit >= 0 && floorHit != expectedHit)
            return;
        if (surfaceZ - ceilingZ < requiredClearance)
            return;
        appendSurface(support, surfaceZ, ceilingZ);
    };

    const int sectorFloor = getflorzofslope(sectorId, x, y);
    probe(engineSupport(kSupportSectorFloor, sectorId), sectorFloor, -1);
    for (int nSprite = headspritesect[sectorId]; nSprite >= 0;
         nSprite = nextspritesect[nSprite])
    {
        const spritetype &record = sprite[nSprite];
        if (gMe && gMe->pSprite && nSprite == gMe->pSprite->index)
            continue;
        if (!(record.cstat & CSTAT_SPRITE_BLOCK))
            continue;
        const int alignment = record.cstat & CSTAT_SPRITE_ALIGNMENT_MASK;
        int surfaceZ = 0;
        if (alignment == CSTAT_SPRITE_ALIGNMENT_FLOOR
            || alignment == CSTAT_SPRITE_ALIGNMENT_SLOPE)
            surfaceZ = spriteGetZOfSlope(uint16_t(nSprite), { x, y });
        else
        {
            int bottom = 0;
            spriteCollisionExtents(nSprite, surfaceZ, bottom);
        }
        probe(engineSupport(kSupportSpriteFloor, nSprite), surfaceZ, 0xc000 | nSprite);
    }
    return result;
}
// Ask NBlood which concrete collision surface owns the floor at a proposed
// player pose.  Navigation support identity is deliberately nothing more
// than this engine answer: artwork alignment and Build sector ownership do
// not grant or deny accessibility on their own.
 bool engineHasSupportAt(const SupportRef &support, int sectorId,
                               int x, int y, int surfaceZ, int radius)
{
    if (!inRange(sectorId, 0, numsectors) || supportIndex(support) < 0)
        return false;
    int ceilingZ = 0, ceilingHit = 0, floorZ = 0, floorHit = 0;
    engineGetZRangeIgnoringActor(
        x, y, surfaceZ - 1, sectorId,
        &ceilingZ, &ceilingHit, &floorZ, &floorHit,
        radius, CLIPMASK0,
        PARALLAXCLIP_CEILING | PARALLAXCLIP_FLOOR);
    const int expected = supportKind(support) == kSupportSpriteFloor
        ? (0xc000 | supportIndex(support)) : (0x4000 | supportIndex(support));
    return floorHit == expected && floorZ == surfaceZ;
}
// Resolve the physical plane owned by a support at one XY pose.  A support
// is not necessarily flat: sector floors and sloped floor sprites can both
// change Z while retaining the same collision identity.
 bool engineSupportZAt(const SupportRef &support, int x, int y,
                             int &surfaceZ)
{
    if (supportKind(support) == kSupportSectorFloor)
    {
        if (!inRange(supportIndex(support), 0, numsectors)
            || inside(x, y, supportIndex(support)) != 1)
            return false;
        surfaceZ = getflorzofslope(supportIndex(support), x, y);
        return true;
    }
    if (supportKind(support) != kSupportSpriteFloor
        || !inRange(supportIndex(support), 0, kMaxSprites)
        || sprite[supportIndex(support)].statnum >= kMaxStatus)
        return false;
    const spritetype &record = sprite[supportIndex(support)];
    const int alignment = record.cstat & CSTAT_SPRITE_ALIGNMENT_MASK;
    if (alignment == CSTAT_SPRITE_ALIGNMENT_FLOOR
        || alignment == CSTAT_SPRITE_ALIGNMENT_SLOPE)
        surfaceZ = spriteGetZOfSlope(uint16_t(supportIndex(support)), { x, y });
    else
    {
        int bottom = 0;
        spriteCollisionExtents(supportIndex(support), surfaceZ, bottom);
    }
    return true;
}
// How much horizontal error a concrete sprite-support pose tolerates before
// NBlood stops naming that same collision surface as the player's floor.
// This is intentionally sampled through GetZRange rather than reconstructed
// from artwork dimensions: floor sprites, slopes, wall sprites, voxels and
// ordinary blocking sprites all keep the exact collision semantics used by
// live movement. The result is route quality, not reachability; an edge pose
// remains in the graph with zero margin when it is the only physical option.
 int spriteSupportClearance(int sectorId, int spriteId, int x, int y,
                                  int radius)
{
    if (!inRange(sectorId, 0, numsectors)
        || !inRange(spriteId, 0, kMaxSprites))
        return 0;
    auto retained = [&](int px, int py) {
        int surfaceZ = 0;
        const spritetype &record = sprite[spriteId];
        const int alignment = record.cstat & CSTAT_SPRITE_ALIGNMENT_MASK;
        if (alignment == CSTAT_SPRITE_ALIGNMENT_FLOOR
            || alignment == CSTAT_SPRITE_ALIGNMENT_SLOPE)
            surfaceZ = spriteGetZOfSlope(uint16_t(spriteId), { px, py });
        else
        {
            int bottom = 0;
            spriteCollisionExtents(spriteId, surfaceZ, bottom);
        }
        return engineHasSupportAt(engineSupport(kSupportSpriteFloor, spriteId),
                                  sectorId, px, py, surfaceZ, radius);
    };

    static const int directions[8][2] = {
        { 256, 0 }, { -256, 0 }, { 0, 256 }, { 0, -256 },
        { 181, 181 }, { 181, -181 }, { -181, 181 }, { -181, -181 },
    };
    constexpr int maximum = 512;
    constexpr int stride = 64;
    int clearance = maximum;
    for (const auto &direction : directions)
    {
        int retainedDistance = 0;
        for (int distance = stride; distance <= maximum; distance += stride)
        {
            const int px = x + direction[0] * distance / 256;
            const int py = y + direction[1] * distance / 256;
            if (!retained(px, py))
                break;
            retainedDistance = distance;
        }
        clearance = std::min(clearance, retainedDistance);
    }
    return clearance;
}
 SupportRef currentPlayerSupport()
{
    if (!gMe || !gMe->pSprite)
        return SupportRef();
    int ceilingZ = 0, ceilingHit = 0, floorZ = 0, floorHit = 0;
    GetZRange(gMe->pSprite, &ceilingZ, &ceilingHit, &floorZ, &floorHit,
              (gMe->pSprite->clipdist << 2) + 16, CLIPMASK0,
              PARALLAXCLIP_CEILING | PARALLAXCLIP_FLOOR);
    if ((floorHit & 0xc000) == 0xc000)
        return engineSupport(kSupportSpriteFloor, floorHit & 0x3fff);
    // getzrange sweeps the whole player cylinder.  At a raised boundary the
    // floor which actually catches Caleb can therefore belong to an adjacent
    // sector while his origin (and pSprite->sectnum) is still on the lower
    // side.  floorHit is the collision authority in exactly that case.  Using
    // sectnum invented impossible states such as "sector-5 support at the
    // sector-2 floor height" and stranded routes on a pose already passed.
    if ((floorHit & 0xc000) == 0x4000)
        return engineSupport(kSupportSectorFloor, floorHit & 0x3fff);
    return engineSupport(kSupportSectorFloor, gMe->pSprite->sectnum);
}
// Sector-floor ids are collision metadata, not distinct navigation places.
// Thin and overlapping Build partitions can describe the same occupiable
// support plane while the engine legitimately keeps either sector number in
// the player sprite.  A sprite/voxel support retains exact collision identity;
// a sector floor pose is established when the player is physically inside
// that polygon on the expected plane.
 bool playerOccupiesSupportPose(const SupportRef &expected,
                                      int expectedZ)
{
    if (!gMe || !gMe->pSprite)
        return false;
    const int liveFloor = llmapper::capability::playerFloorZ();
    if (std::abs(liveFloor - expectedZ)
        > llmapper::capability::playerStepHeight() / 2)
        return false;
    // The floor hit returned for the player's whole collision cylinder owns
    // support even when the origin has already crossed an adjacent sector
    // boundary. Never weaken that direct engine answer with a point-in-polygon
    // test intended only for equivalent overlapping floor metadata.
    if (currentPlayerSupport() == expected)
        return true;
    if (supportKind(expected) == kSupportSpriteFloor)
        return false;
    if (supportKind(expected) != kSupportSectorFloor
        || !inRange(supportIndex(expected), 0, numsectors)
        || inside(gMe->pSprite->x, gMe->pSprite->y, supportIndex(expected)) != 1)
        return false;
    return std::abs(getflorzofslope(supportIndex(expected), gMe->pSprite->x,
                                    gMe->pSprite->y) - liveFloor)
        <= llmapper::capability::playerStepHeight() / 2;
}
// Exact environmental-contact rule used by actTouchFloor.  It is only ever
// true for the sector floor collision surface; a sprite floor hit does not
// call actTouchFloor and therefore must not inherit the sector's damage.
bool engineSupportDamagesPlayer(const SupportRef &support)
{
    if (supportKind(support) != kSupportSectorFloor
        || !inRange(supportIndex(support), 0, numsectors))
        return false;
    const sectortype &record = sector[supportIndex(support)];
    const XSECTOR *extra = record.extra > 0 && record.extra < kMaxXSectors
        ? &xsector[record.extra] : nullptr;
    bool damages = extra
        && (record.type == kSectorDamage || extra->damageType > 0);
#ifdef NOONE_EXTENSIONS
    if (gModernMap && damages && record.type == kSectorDamage && !extra->state)
        damages = false;
#endif
    return damages || tileGetSurfType(supportIndex(support), 0x4000) == kSurfLava;
}

bool engineRegionGeometryMoving(int sectorId)
{
    if (!inRange(sectorId, 0, numsectors))
        return false;
    const int sectorExtra = sector[sectorId].extra;
    if (inRange(sectorExtra, 1, kMaxXSectors)
        && (uint32_t(xsector[sectorExtra].busy) & 0xffffu) != 0)
        return true;
    const sectortype &record = sector[sectorId];
    for (int offset = 0; offset < record.wallnum; ++offset)
    {
        const int wallId = record.wallptr + offset;
        if (!inRange(wallId, 0, numwalls))
            continue;
        const int extra = wall[wallId].extra;
        if (inRange(extra, 1, kMaxXWalls)
            && (uint32_t(xwall[extra].busy) & 0xffffu) != 0)
            return true;
    }
    for (int spriteId = headspritesect[sectorId]; spriteId >= 0;
         spriteId = nextspritesect[spriteId])
        if ((!gMe || !gMe->pSprite || spriteId != gMe->pSprite->index)
            && (xvel[spriteId] != 0 || yvel[spriteId] != 0
                || zvel[spriteId] != 0))
            return true;
    return false;
}

int engineObjectStateSignature(WorldObjectRef object)
{
    const EngineHandle handle = engineObjectHandle(object);
    int signature = llmapper::mixHash(31, int(handle.kind));
    signature = llmapper::mixHash(signature, handle.index);
    if (handle.kind == kWorldSector && inRange(handle.index, 0, numsectors))
    {
        const sectortype &record = sector[handle.index];
        signature = llmapper::mixHash(signature, record.floorz);
        signature = llmapper::mixHash(signature, record.ceilingz);
        if (inRange(record.extra, 1, kMaxXSectors))
        {
            const XSECTOR &extra = xsector[record.extra];
            signature = llmapper::mixHash(signature, extra.state);
            signature = llmapper::mixHash(signature, extra.busy);
            signature = llmapper::mixHash(signature, extra.locked);
        }
    }
    else if (handle.kind == kWorldWall && inRange(handle.index, 0, numwalls))
    {
        const walltype &record = wall[handle.index];
        signature = llmapper::mixHash(signature, record.x);
        signature = llmapper::mixHash(signature, record.y);
        signature = llmapper::mixHash(signature, record.cstat);
        if (inRange(record.extra, 1, kMaxXWalls))
        {
            const XWALL &extra = xwall[record.extra];
            signature = llmapper::mixHash(signature, extra.state);
            signature = llmapper::mixHash(signature, extra.busy);
            signature = llmapper::mixHash(signature, extra.locked);
        }
    }
    else if (handle.kind == kWorldSprite
             && inRange(handle.index, 0, kMaxSprites))
    {
        const spritetype &record = sprite[handle.index];
        signature = llmapper::mixHash(signature, record.x);
        signature = llmapper::mixHash(signature, record.y);
        signature = llmapper::mixHash(signature, record.z);
        signature = llmapper::mixHash(signature, record.cstat);
        if (validXSprite(record.extra))
        {
            const XSPRITE &extra = xsprite[record.extra];
            signature = llmapper::mixHash(signature, extra.state);
            signature = llmapper::mixHash(signature, extra.busy);
            signature = llmapper::mixHash(signature, extra.locked);
        }
    }
    return signature;
}
 int64_t segmentDistance2(int x, int y, int x1, int y1, int x2, int y2)
{
    const int64_t dx = x2 - x1;
    const int64_t dy = y2 - y1;
    const int64_t length2 = dx * dx + dy * dy;
    if (length2 <= 0)
        return int64_t(x - x1) * (x - x1) + int64_t(y - y1) * (y - y1);
    int64_t t = ((int64_t(x - x1) * dx) + (int64_t(y - y1) * dy)) * 1024 / length2;
    if (t < 0)
        t = 0;
    if (t > 1024)
        t = 1024;
    const int px = int(x1 + dx * t / 1024);
    const int py = int(y1 + dy * t / 1024);
    return int64_t(x - px) * (x - px) + int64_t(y - py) * (y - py);
}
// The two ends of a wall-aligned sprite, as clipmove computes them.  A wall
// sprite is a line, not a post: a panel across a doorway is several hundred
// units wide, and treating it as a clipdist-sized circle at its centre
// misses everything but the middle.
//
// This deliberately does not call Blood's own GetSpriteExtents, which
// answers a slightly different question: it mirrors the x offset on
// CSTAT_SPRITE_YFLIP, while the engine's get_wallspr_dims -- the one
// clipmove actually clips with -- mirrors on CSTAT_SPRITE_XFLIP.  The bot is
// asking what the player's body will hit, so clipmove is the authority.
static void wallSpriteSpan(const spritetype &record, int &x1, int &y1,
                           int &x2, int &y2)
{
    const int span = tilesiz[record.picnum].x;
    const int offset = (record.cstat & CSTAT_SPRITE_XFLIP)
        ? -(picanm[record.picnum].xofs + record.xoffset)
        : (picanm[record.picnum].xofs + record.xoffset);
    const int dax = sintable[record.ang & kAngMask] * record.xrepeat;
    const int day = sintable[(record.ang + kAng90 + kAng180) & kAngMask] * record.xrepeat;
    const int anchor = (span >> 1) + offset;
    x1 = record.x - mulscale16(dax, anchor);
    y1 = record.y - mulscale16(day, anchor);
    x2 = x1 + mulscale16(dax, span);
    y2 = y1 + mulscale16(day, span);
}
// A sprite standing at this point that the player's body cannot pass.
// Returns its index, or -1.
//
// Only horizontal obstruction is asked about.  A floor-aligned sprite is a
// surface, not a barrier -- it is what a sprite bridge is made of -- and
// clipping against one walls off the very route it provides.
 int solidSpriteAt(int sectorId, int x, int y, int radius)
{
    if (!inRange(sectorId, 0, numsectors))
        return -1;
    const int floorZ = getflorzofslope(sectorId, x, y);
    const int headroom = playerCrouchClearance();
    for (int nSprite = headspritesect[sectorId]; nSprite >= 0;
         nSprite = nextspritesect[nSprite])
    {
        const spritetype &record = sprite[nSprite];
        if (!(record.cstat & CSTAT_SPRITE_BLOCK))
            continue;
        if (gMe && gMe->pSprite && nSprite == gMe->pSprite->index)
            continue;
        const int alignment = record.cstat & CSTAT_SPRITE_ALIGNMENT_MASK;
        if (alignment == CSTAT_SPRITE_ALIGNMENT_FLOOR
            || alignment == CSTAT_SPRITE_ALIGNMENT_SLOPE)
            continue;
        // Something hanging clear above head height is not in the way.
        int top = 0;
        int bottom = 0;
        spriteCollisionExtents(nSprite, top, bottom);
        if (bottom < floorZ - headroom || top > floorZ)
            continue;
        if (alignment == CSTAT_SPRITE_ALIGNMENT_WALL)
        {
            int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
            wallSpriteSpan(record, x1, y1, x2, y2);
            if (segmentDistance2(x, y, x1, y1, x2, y2) < int64_t(radius) * radius)
                return nSprite;
            continue;
        }
        const int64_t margin = int64_t(radius) + (record.clipdist << 2);
        if (distance2(x, y, record.x, record.y) < margin * margin)
            return nSprite;
    }
    return -1;
}
static int bodyRadiusOf()
{
    return llmapper::capability::playerEnvelope().radius;
}
// A NavCell stores the physical support plane under the player's feet.
// Convert that plane to the same sprite-origin Z used by live cansee() calls
// with the engine-derived sprite extents, never a guessed "eye offset".
 int playerOriginAtSupport(int supportZ)
{
    return supportZ - llmapper::capability::playerFootOffset();
}
// Ask the same collision resolver Blood uses after moving actors whether the
// player's complete hull can remain at this support pose.  A geometric
// distance-to-wall estimate is not equivalent around corners, narrow stairs,
// sprites, or overlapping sectors: pushmove is the authoritative predicate.
 bool enginePlayerPoseFits(int sectorId, int x, int y, int supportZ,
                                 bool crouched)
{
    if (!inRange(sectorId, 0, numsectors) || !gMe || !gMe->pSprite)
        return false;
    int32_t resolvedX = x;
    int32_t resolvedY = y;
    const PlayerCollisionShape liveShape = livePlayerCollisionShape();
    const PlayerCollisionShape &shape = crouched && gObservedCrouchShape.known
        ? gObservedCrouchShape : liveShape;
    if (!shape.known)
        return false;
    const int originZ = supportZ - shape.footOffset;
    int32_t resolvedZ = originZ;
    int16_t resolvedSector = int16_t(sectorId);
    int ceilingDistance = 0;
    int floorDistance = 0;
    ceilingDistance = shape.ceilingDistance;
    floorDistance = shape.floorDistance;
    spritetype *actor = gMe->pSprite;
    const int savedCstat = actor->cstat;
    // This is a hypothetical pose for the live actor. MoveDude removes the
    // actor's blocking identities while resolving its real movement; keeping
    // the present sprite in pushmove's collision set makes nearby future
    // poses collide with Caleb himself.
    actor->cstat &= ~257;
    const int result = pushmove_old(
        &resolvedX, &resolvedY, &resolvedZ, &resolvedSector,
        shape.radius,
        ceilingDistance, floorDistance, CLIPMASK0);
    actor->cstat = int16_t(savedCstat);
    return result >= 0 && resolvedX == x && resolvedY == y
        && resolvedZ == originZ
        && inRange(int(resolvedSector), 0, numsectors);
}
// Blood stores explosion radii in the EXPLOSION table in 1/16 map units;
// CheckProximity shifts XY deltas by four before comparing them.  Keep that
// conversion next to the engine table so navigation never grows a second,
// guessed set of TNT/barrel distances.
static int engineExplosionRadius(int explosionType)
{
    if (!inRange(explosionType, 0, kExplosionMax))
        return 0;
    return int(explodeInfo[explosionType].radius) << 4;
}
 int explosiveWeaponRadius(int weapon)
{
    switch (weapon)
    {
    case kWeaponTNT:
        return engineExplosionRadius(kExplosionStandard);
    case kWeaponNapalm:
        // The projectile and its secondary radius-damage path are both
        // engine-defined; the larger is the conservative deliberate-action
        // envelope.
        return std::max(engineExplosionRadius(kExplosionNapalm), 128 << 4);
    default:
        return 0;
    }
}
 int explosiveSpriteRadius(const spritetype &record)
{
    switch (record.type)
    {
    case kThingArmedTNTStick:
        return engineExplosionRadius(kExplosionSmall);
    case kThingArmedProxBomb:
    case kThingArmedRemoteBomb:
    case kThingArmedTNTBundle:
        return engineExplosionRadius(kExplosionStandard);
    case kThingTNTBarrel:
        return engineExplosionRadius(kExplosionLarge);
    default:
        return 0;
    }
}
 int safeExplosionSeparation(int damagingRadius)
{
    return damagingRadius > 0
        ? damagingRadius + bodyRadiusOf() + kExplosiveSafetyMargin : 0;
}
 void setInteractionGeometry(InteractionCandidate &interaction)
{
    EngineBoundaryObservation &portal = interaction.target;
    if (!inRange(portal.from, 0, numsectors))
        return;
    const int fromFloor = getflorzofslope(portal.from, portal.x, portal.y);
    const int fromCeiling = getceilzofslope(portal.from, portal.x, portal.y);
    int bottom = fromFloor;
    int top = fromCeiling;
    if (inRange(portal.to, 0, numsectors))
    {
        const int toFloor = getflorzofslope(portal.to, portal.x, portal.y);
        const int toCeiling = getceilzofslope(portal.to, portal.x, portal.y);
        // ActionScan resolves a two-sided wall only where hitscan actually
        // strikes its solid upper/lower band. The shared vertical overlap is
        // the opening and aiming there passes straight through it.
        if (toFloor != fromFloor)
        {
            top = std::min(fromFloor, toFloor);
            bottom = std::max(fromFloor, toFloor);
        }
        else if (toCeiling != fromCeiling)
        {
            top = std::min(fromCeiling, toCeiling);
            bottom = std::max(fromCeiling, toCeiling);
        }
    }
    if (bottom <= top)
    {
        bottom = fromFloor;
        top = fromCeiling;
    }
    interaction.targetBottomZ = bottom;
    interaction.targetTopZ = top;
    portal.z = (top + bottom) / 2;
}
// A usable sprite is aimed at by its own extents, not by the room it is in.
// Describing a low switch as if it filled the sector makes the bot stare
// straight ahead at mid-height and ActionScan never resolves it.
 void setSpriteInteractionGeometry(InteractionCandidate &interaction, int spriteIndex)
{
    EngineBoundaryObservation &portal = interaction.target;
    if (!inRange(spriteIndex, 0, kMaxSprites))
        return;
    spritetype &record = sprite[spriteIndex];
    int top = record.z;
    int bottom = record.z;
    GetSpriteExtents(&record, &top, &bottom);
    if (bottom < top)
        std::swap(top, bottom);
    interaction.targetTopZ = top;
    interaction.targetBottomZ = bottom;
    portal.z = record.z;
}
 const char *itemCategory(int type)
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
// Translate the engine's accepted damage classes into effects the planner
// can reason about.  This deliberately asks THINGINFO rather than naming
// wall-crack sprites: any live thing whose damage table accepts explosion
// damage exposes the same abstract opportunity.
 unsigned acceptedDamageEffects(const spritetype &record)
{
    if (record.statnum != kStatThing
        || record.type < kThingBase || record.type >= kThingMax
        || !validXSprite(record.extra) || xsprite[record.extra].health <= 0)
        return llmapper::kEffectNone;
    // Damageability alone is not a progression affordance (a thrown charge
    // is damageable too).  Destruction must either remove solid collision or
    // dispatch a mapper-authored world change.  This keeps volatile effect
    // producers available to a future satisfier search without mistaking
    // them for the blocked traversal itself.
    if (!(record.cstat & 1) && xsprite[record.extra].txID == 0)
        return llmapper::kEffectNone;
    const THINGINFO &info = thingInfo[record.type - kThingBase];
    unsigned effects = llmapper::kEffectNone;
    if (info.dmgControl[kDamageExplode] > 0)
        effects |= llmapper::kEffectExplosive;
    if (info.dmgControl[kDamageBullet] > 0)
        effects |= llmapper::kEffectBulletDamage;
    return effects;
}
static void appendUnique(std::vector<int> &values, int value)
{
    if (std::find(values.begin(), values.end(), value) == values.end())
        values.push_back(value);
}
Observation observeWorld()
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
    const int observerZ = gMe->zView;
    appendUnique(result.visibleSectors, result.sector);

    // Do not import Build's rendered-sector bitmap here. A sector is an
    // adapter query parameter, not a unit of visible space: one concave
    // sector can contain an occluded maze and many coplanar sectors can be a
    // single empty room. Far-side adapter regions are admitted below only by
    // a concrete, visible physical opening.

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
            const XSECTOR &state = xsector[current.extra];
            InteractionCandidate candidate;
            candidate.kind = kInteractionSector;
            candidate.id = result.sector;
            candidate.fromSector = result.sector;
            candidate.targetSector = result.sector;
            candidate.x = result.x;
            candidate.y = result.y;
            candidate.z = result.z;
            candidate.key = state.Key;
            candidate.reversible = true;
            candidate.target.from = result.sector;
            candidate.target.to = result.sector;
            candidate.target.x = result.x;
            candidate.target.y = result.y;
            candidate.target.sectorPushCurrent = true;
            setInteractionGeometry(candidate);
            candidate.z = candidate.target.z;
            result.interactions.push_back(candidate);
        }

        // Build ownership is not observation. A concave sector may wrap an
        // entire maze around the player, so enumerating its wall loop as
        // "local geometry" leaks doors and affordances through corners. Ask
        // Blood whether a point just inside this side of the wall is visible
        // from the player's actual pose. Multiple samples avoid making a
        // long, partly exposed wall depend on its midpoint.
        auto wallSurfaceVisible = [&](int wallIndex) {
            if (!inRange(wallIndex, 0, numwalls)
                || !inRange(wall[wallIndex].point2, 0, numwalls))
                return false;
            const walltype &start = wall[wallIndex];
            const walltype &end = wall[start.point2];
            const int dx = end.x - start.x;
            const int dy = end.y - start.y;
            const int length = std::max(1, int(std::sqrt(double(
                int64_t(dx) * dx + int64_t(dy) * dy))));
            const int normalX = -dy;
            const int normalY = dx;
            static const int kFractions[3] = { 1, 2, 3 };
            for (int fraction : kFractions)
            {
                const int surfaceX = start.x
                    + int(int64_t(dx) * fraction / 4);
                const int surfaceY = start.y
                    + int(int64_t(dy) * fraction / 4);
                for (int direction = -1; direction <= 1; direction += 2)
                {
                    // Stay on the observer's physical side. The inset is
                    // only to disambiguate a target exactly on a Build wall;
                    // it is not a navigation sampling interval.
                    const int targetX = surfaceX
                        + direction * normalX * 64 / length;
                    const int targetY = surfaceY
                        + direction * normalY * 64 / length;
                    if (inside(targetX, targetY, result.sector) != 1)
                        continue;
                    const int ceiling = getceilzofslope(
                        result.sector, targetX, targetY);
                    const int floor = standingFloorZ(
                        result.sector, targetX, targetY);
                    const int margin = std::min(256,
                        std::max(0, (floor - ceiling) / 4));
                    const int targetZ = std::max(ceiling + margin,
                        std::min(result.z, floor - margin));
                    if (cansee(result.x, result.y, observerZ, result.sector,
                               targetX, targetY, targetZ, result.sector))
                        return true;
                }
            }
            return false;
        };

        for (int i = 0; i < current.wallnum; ++i)
        {
            const int wallIndex = current.wallptr + i;
            const walltype &wallRecord = wall[wallIndex];
            const walltype &nextWall = wall[wallRecord.point2];
            const bool surfaceVisible = wallSurfaceVisible(wallIndex);
            if (!surfaceVisible)
                continue;
            if (!inRange(wallRecord.nextsector, 0, numsectors))
            {
                // XWALL triggerPush is usable without a nextsector.  It is
                // discovered from the reachable current sector, then the
                // real ActionScan preview remains the activation authority.
                if (wallRecord.extra > 0 && wallRecord.extra < kMaxXWalls
                    && xwall[wallRecord.extra].triggerPush)
                {
                    InteractionCandidate candidate;
                    candidate.kind = kInteractionWall;
                    candidate.id = wallIndex;
                    candidate.fromSector = result.sector;
                    candidate.x = (wallRecord.x + nextWall.x) / 2;
                    candidate.y = (wallRecord.y + nextWall.y) / 2;
                    candidate.z = result.z;
                    candidate.key = xwall[wallRecord.extra].key;
                    candidate.reversible = true;
                    candidate.target.wall = wallIndex;
                    candidate.target.from = result.sector;
                    candidate.target.to = -1;
            candidate.target.x = candidate.x;
            candidate.target.y = candidate.y;
            candidate.target.wallPush = true;
            candidate.dependencyChannel = xwall[wallRecord.extra].txID;
            candidate.target.x1 = wallRecord.x;
            candidate.target.y1 = wallRecord.y;
            candidate.target.x2 = nextWall.x;
            candidate.target.y2 = nextWall.y;
            setInteractionGeometry(candidate);
            candidate.z = candidate.target.z;
            result.interactions.push_back(candidate);
                }
                continue;
            }
            ++result.localPortalCount;

            const int midX = (wallRecord.x + nextWall.x) / 2;
            const int midY = (wallRecord.y + nextWall.y) / 2;
            // The surface the player would stand on at the threshold, which
            // is a bridge plank when one lies across the doorway.  Do not
            // query exactly on the shared wall: a floor sprite ending at the
            // boundary is deliberately ambiguous there and GetZRange can
            // return the pit below it.  Sample a short distance into each
            // sector, where the player's centre actually stands while
            // crossing the threshold.
            auto interiorSample = [&](int sectorId, int &sampleX, int &sampleY) {
                sampleX = midX;
                sampleY = midY;
                const int normalX = -(nextWall.y - wallRecord.y);
                const int normalY = nextWall.x - wallRecord.x;
                const int length = std::max(1, int(std::sqrt(double(
                    int64_t(normalX) * normalX + int64_t(normalY) * normalY))));
                const int radius = gMe && gMe->pSprite
                    ? (gMe->pSprite->clipdist << 2) : 128;
                // The player's centre cannot occupy the first clip-radius
                // band beside a wall.  Sampling shallower than that still
                // lands on the ambiguous bevel at the end of a bridge
                // sprite, especially on its one-sided edge.
                const int depth = std::max(384, std::min(512, radius * 3));
                for (int direction = -1; direction <= 1; direction += 2)
                {
                    const int x = midX + direction * normalX * depth / length;
                    const int y = midY + direction * normalY * depth / length;
                    if (inside(x, y, sectorId) == 1)
                    {
                        sampleX = x;
                        sampleY = y;
                        return;
                    }
                }
            };
            int fromSampleX = midX, fromSampleY = midY;
            int toSampleX = midX, toSampleY = midY;
            interiorSample(result.sector, fromSampleX, fromSampleY);
            interiorSample(wallRecord.nextsector, toSampleX, toSampleY);
            const int fromFloor = standingFloorZ(result.sector,
                                                  fromSampleX, fromSampleY);
            const int toFloor = standingFloorZ(wallRecord.nextsector,
                                                toSampleX, toSampleY);
            const int fromCeiling = getceilzofslope(result.sector,
                                                     fromSampleX, fromSampleY);
            const int toCeiling = getceilzofslope(wallRecord.nextsector,
                                                   toSampleX, toSampleY);
            const int openingWidth = int(std::sqrt(double(distance2(wallRecord.x, wallRecord.y, nextWall.x, nextWall.y))));
            const int clearance = std::min(fromFloor - fromCeiling, toFloor - toCeiling);
            const int floorDelta = toFloor - fromFloor;
            const int bodyClearance = playerStandingClearance();
            const int crouchClearance = playerCrouchClearance();
            const int jumpRiseLimit = playerJumpRiseLimit();
            const int dropLimit = jumpRiseLimit;
            const bool enoughWidth = openingWidth >= playerPassageWidth();
            const bool standingClearance = clearance >= bodyClearance;
            const bool crouchingClearance = clearance >= crouchClearance;
            const bool walkable = enoughWidth && standingClearance && std::abs(floorDelta) <= llmapper::capability::playerStepHeight();
            const bool crouchable = enoughWidth && !standingClearance && crouchingClearance
                && std::abs(floorDelta) <= llmapper::capability::playerStepHeight();
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
            const bool openingVisible = cansee(
                result.x, result.y, observerZ, result.sector,
                midX, midY, midZ, wallRecord.nextsector);

            EngineBoundaryObservation portal;
            bool portalHasInteraction = false;
            int interactionKey = 0;
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
            portal.visible = openingVisible;
            portal.openingWidth = openingWidth;
            portal.floorDelta = floorDelta;
            portal.clearance = clearance;
            // This boundary was admitted by wallSurfaceVisible above. Sector
            // membership alone never grants knowledge of the rest of the
            // containing polygon.
            portal.localGeometry = true;
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
            if (wallRecord.extra > 0 && wallRecord.extra < kMaxXWalls)
            {
                const XWALL &extra = xwall[wallRecord.extra];
                portal.wallPush = extra.triggerPush != 0;
                portal.shootable = extra.triggerVector != 0 && !extra.isTriggered;
                interactionKey = extra.key;
                portal.wallState = extra.state;
                portal.wallBusy = extra.busy;
            }
            if (sector[portal.to].extra > 0 && sector[portal.to].extra < kMaxXSectors)
            {
                const XSECTOR &extra = xsector[sector[portal.to].extra];
                if (!interactionKey)
                    interactionKey = extra.Key;
                portal.sectorPush = extra.Wallpush != 0;
                portal.sectorState = extra.state;
                portal.sectorBusy = extra.busy;
                // A damaging floor only hurts the player who touches it.
                // Blood calls actTouchFloor for a sector hit and not for a
                // sprite hit, so a bridge laid across the pit is a way over
                // it -- refusing the whole sector makes such a crossing
                // invisible.
                const bool touchesDamagingFloor =
                    standingFloorZ(portal.to, portal.x, portal.y)
                        >= getflorzofslope(portal.to, portal.x, portal.y);
                if ((extra.damageType != 0 || sector[portal.to].type == kSectorDamage)
                    && touchesDamagingFloor)
                {
                    portal.walkable = false;
                    portal.jumpable = false;
                    portal.crouchable = false;
                    portal.dropSafe = false;
                    portal.traversable = false;
                    portal.capability = kTraversalCurrentlyUnavailable;
                }
            }
            // A solid sprite standing in a doorway shuts it as surely as a
            // closed door does, and the wall geometry says nothing about it.
            // Without this a barricaded opening read as walkable, and the bot
            // shouldered it until the objective budget ran out instead of
            // treating it as an obstacle with something to be done about it.
            if (portal.traversable)
            {
                const int radius = gMe && gMe->pSprite
                    ? (gMe->pSprite->clipdist << 2) : 128;
                // Sample across the opening rather than only its midpoint.
                // A barricade often covers part of a doorway and leaves a
                // gap at one end; judging the whole boundary by its centre
                // either shuts a passable door or waves the bot at a solid
                // one.  Where a clear stretch exists, aim at the middle of
                // that stretch instead of at the middle of the wall.
                const int samples = 9;
                int clearFirst = -1;
                int clearLast = -1;
                int obstruction = -1;
                std::vector<int> operableObstructions;
                for (int step = 0; step < samples; ++step)
                {
                    const int px = portal.x1
                        + int(int64_t(portal.x2 - portal.x1) * step / (samples - 1));
                    const int py = portal.y1
                        + int(int64_t(portal.y2 - portal.y1) * step / (samples - 1));
                    int hit = solidSpriteAt(portal.from, px, py, radius);
                    if (hit < 0)
                        hit = solidSpriteAt(portal.to, px, py, radius);
                    if (hit >= 0)
                    {
                        obstruction = hit;
                        const spritetype &blocker = sprite[hit];
                        if (validXSprite(blocker.extra)
                            && (xsprite[blocker.extra].Push
                                || xsprite[blocker.extra].Vector))
                            appendUnique(operableObstructions, hit);
                        continue;
                    }
                    if (clearFirst < 0)
                        clearFirst = step;
                    clearLast = step;
                }
                if (clearFirst >= 0 && obstruction >= 0)
                {
                    const int middle = (clearFirst + clearLast) / 2;
                    portal.x = portal.x1
                        + int(int64_t(portal.x2 - portal.x1) * middle / (samples - 1));
                    portal.y = portal.y1
                        + int(int64_t(portal.y2 - portal.y1) * middle / (samples - 1));
                    obstruction = -1;
                }
                if (obstruction >= 0)
                {
                    portal.walkable = false;
                    portal.jumpable = false;
                    portal.crouchable = false;
                    portal.dropSafe = false;
                    portal.traversable = false;
                    portal.capability = kTraversalCurrentlyUnavailable;
                    portal.blockedBySprite = true;
                    portal.blockerSprite = obstruction;
                    // Blocked, but not necessarily hopeless: a barricade the
                    // player can operate is progression work, not a wall.
                    if (!operableObstructions.empty())
                        portalHasInteraction = true;
                }

                // A sprite barricade is operated as a sprite, not as the
                // otherwise ordinary wall portal it happens to cover.  Keep
                // the approach on the reachable side of the obstruction;
                // a long-range visibility observation of the same sprite in
                // the unopened destination sector must not make the useful
                // action look unreachable.
                if (portal.blockedBySprite)
                {
                    for (int blockerId : operableObstructions)
                    {
                        const spritetype &blocker = sprite[blockerId];
                        const XSPRITE &extra = xsprite[blocker.extra];
                        InteractionCandidate candidate;
                        candidate.kind = kInteractionSprite;
                        candidate.id = blockerId;
                        candidate.fromSector = result.sector;
                        candidate.targetSector = portal.to;
                        candidate.x = blocker.x;
                        candidate.y = blocker.y;
                        candidate.z = blocker.z;
                        candidate.reversible = true;
                        candidate.activationMode = extra.Vector
                            ? llmapper::kActivateVector : llmapper::kActivateUse;
                        if (candidate.activationMode == llmapper::kActivateUse)
                        {
                            candidate.key = extra.key;
                        }
                        candidate.target.wall = -1;
                        candidate.target.from = result.sector;
                        candidate.target.to = portal.to;
                        candidate.target.x = blocker.x;
                        candidate.target.y = blocker.y;
                        candidate.target.shootable = extra.Vector != 0;
                        candidate.dependencyChannel = extra.txID;
                        candidate.target.blockedBySprite = true;
                        candidate.target.blockerSprite = blockerId;
                        setSpriteInteractionGeometry(candidate, blockerId);
                        candidate.z = candidate.target.z;
                        result.interactions.push_back(candidate);
                    }
                }
            }
            if (sector[portal.from].extra > 0 && sector[portal.from].extra < kMaxXSectors)
                portal.sectorPushCurrent = xsector[sector[portal.from].extra].Push != 0;
            // This is construction state only. The concrete candidates below
            // are the semantic interaction records; boundary geometry does
            // not retain a second "has affordance" truth.
            portalHasInteraction = portalHasInteraction
                || portal.wallPush || portal.sectorPush
                || portal.sectorPushCurrent || portal.shootable;
            if (!portal.traversable && !portalHasInteraction)
            {
                portal.capability = kTraversalCurrentlyUnavailable;
            }
            if (portal.shootable)
            {
                InteractionCandidate candidate;
                candidate.kind = kInteractionWall;
                candidate.id = wallIndex;
                candidate.fromSector = result.sector;
                candidate.targetSector = portal.to;
                candidate.x = midX;
                candidate.y = midY;
                candidate.z = midZ;
                candidate.reversible = false;
                candidate.activationMode = llmapper::kActivateVector;
                candidate.target = portal;
                setInteractionGeometry(candidate);
                candidate.x = candidate.target.x;
                candidate.y = candidate.target.y;
                candidate.z = candidate.target.z;
                result.interactions.push_back(candidate);
            }
            else if (portalHasInteraction && !portal.blockedBySprite)
            {
                // A sprite obstruction makes the *route* actionable, not the
                // wall behind it.  The collision scan above has already
                // emitted the operable sprite as the concrete affordance, and
                // ActionScan will hit that sprite before it can ever name this
                // wall.  Recording both creates a phantom task which survives
                // after the sprite opens the route and later drags exploration
                // back to a surface the engine never considered usable.  If
                // the wall has an independent push affordance it will be
                // observed normally once the obstruction has moved away.
                InteractionCandidate candidate;
                candidate.kind = kInteractionWall;
                candidate.id = wallIndex;
                candidate.fromSector = result.sector;
                candidate.targetSector = portal.to;
                candidate.x = midX;
                candidate.y = midY;
                candidate.z = midZ;
                candidate.key = interactionKey;
                candidate.reversible = true;
                candidate.target = portal;
                setInteractionGeometry(candidate);
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
                        result.x, result.y, observerZ, result.sector,
                        adjacentX, adjacentY, observerZ, portal.to);
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
                    candidate.reversible = true;
                    candidate.target.wall = adjacentWallIndex;
                    candidate.target.from = result.sector;
                    candidate.target.to = -1;
                    candidate.target.x = adjacentX;
                    candidate.target.y = adjacentY;
                    candidate.target.wallPush = true;
                    candidate.dependencyChannel = xwall[adjacentWall.extra].txID;
                    candidate.target.x1 = adjacentWall.x;
                    candidate.target.y1 = adjacentWall.y;
                    candidate.target.x2 = adjacentNext.x;
                    candidate.target.y2 = adjacentNext.y;
                    setInteractionGeometry(candidate);
                    candidate.z = candidate.target.z;
                    result.interactions.push_back(candidate);
                }
            }

            if (openingVisible)
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
        const unsigned damageEffects = acceptedDamageEffects(candidate);
        // Whether the player can operate a sprite is a fact Blood records on
        // the sprite, not something to be inferred from its status list or
        // its type number.  A mapper is free to wire a plain decoration to a
        // channel, and one that answers to Use is a switch whatever tile it
        // wears; keying off the type whitelist alone left the bot walking
        // past hand-built mechanisms as though they were scenery.
        const bool operable = !isEnemy && validXSprite(candidate.extra)
            && (xsprite[candidate.extra].Push || xsprite[candidate.extra].Vector);
        const bool authoredDamageEffect = damageEffects != llmapper::kEffectNone
            && validXSprite(candidate.extra) && xsprite[candidate.extra].txID > 0;
        const bool vectorDamageReceiver = damageEffects != llmapper::kEffectNone
            && validXSprite(candidate.extra) && xsprite[candidate.extra].Vector
            && xsprite[candidate.extra].health > 0;
        const bool knownTraversalBlocker = damageEffects != llmapper::kEffectNone
            && std::any_of(result.interactions.begin(), result.interactions.end(),
                [&](const InteractionCandidate &interaction) {
                    return interaction.kind == kInteractionSprite
                        && interaction.id == i
                        && interaction.activationMode == llmapper::kActivateDamage;
                });
        const bool usefulDamage = authoredDamageEffect || knownTraversalBlocker;
        if (!isEnemy && !isItem && !isThing && !operable)
            continue;
        if (candidate.index == player->index || candidate.sectnum < 0)
            continue;
        if (isEnemy && (!isEnemyType(candidate.type) || !validXSprite(candidate.extra) || xsprite[candidate.extra].health == 0))
            continue;
        if (isItem && !operable && itemCategory(candidate.type) == nullptr)
            continue;
        if (isThing && !operable && !usefulDamage)
            continue;
        if (!cansee(result.x, result.y, observerZ, result.sector,
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
                         : (isItem && isKeyType(candidate.type)) ? kObjectKey
                         // Explicit mapper-authored Use/Vector semantics are
                         // more specific than inferred damageability.  A
                         // solid actuator may also have health; replacing
                         // its Vector contract with ExplosiveEffect would
                         // discard the actual satisfier the map supplied.
                         // Vector is the delivery mechanism for a
                         // health-bearing receiver, not proof that one hit
                         // completed it. Keep shooting until the receiver's
                         // authored effect changes the world or its collision
                         // body is actually removed.
                         : vectorDamageReceiver ? kObjectDamageable
                         : operable ? kObjectInteractive
                         : usefulDamage ? kObjectDamageable
                         : isThing ? kObjectInteractive
                         : kObjectPickup;
        object.acceptedEffects = damageEffects;
        result.objects.push_back(object);
        if ((object.kind == kObjectInteractive && validXSprite(candidate.extra)
             && (xsprite[candidate.extra].Push || xsprite[candidate.extra].Vector))
            || object.kind == kObjectDamageable)
        {
            InteractionCandidate interaction;
            interaction.kind = kInteractionSprite;
            interaction.id = i;
            // The sprite's containing sector is the stable approach region.
            // A sprite close enough to touch from across a doorway may use
            // the observer's side, but mere long-range visibility must not
            // continually rewrite the remembered approach to wherever the
            // player happens to be standing.
            // Vector activation is explicitly a remote interaction.  The
            // cansee() result above proves that the current physical layer
            // is a valid firing region even when the actuator belongs to a
            // different sector.  Requiring its containing sector first can
            // strand a shootable trigger behind the very transition it
            // controls (AGTST11).  Use remains local and keeps the stable
            // containing-sector approach unless it is actually in reach.
            const bool remoteVector = validXSprite(candidate.extra)
                && xsprite[candidate.extra].Vector;
            interaction.fromSector = remoteVector
                ? result.sector
                : distance2(result.x, result.y, candidate.x, candidate.y)
                    <= kActionScanRange * kActionScanRange
                ? result.sector : candidate.sectnum;
            interaction.targetSector = candidate.sectnum;
            interaction.x = candidate.x;
            interaction.y = candidate.y;
            interaction.z = candidate.z;
            interaction.reversible = object.kind != kObjectDamageable;
            interaction.activationMode = object.kind == kObjectDamageable
                ? llmapper::kActivateDamage
                : xsprite[candidate.extra].Vector
                    ? llmapper::kActivateVector : llmapper::kActivateUse;
            if (interaction.activationMode == llmapper::kActivateUse)
            {
                interaction.key = xsprite[candidate.extra].key;
            }
            interaction.requiredEffects = object.kind == kObjectDamageable
                ? object.acceptedEffects : unsigned(llmapper::kEffectNone);
            interaction.target.wall = -1;
            interaction.target.from = candidate.sectnum;
            interaction.target.to = candidate.sectnum;
            interaction.target.x = candidate.x;
            interaction.target.y = candidate.y;
            interaction.target.shootable = xsprite[candidate.extra].Vector != 0;
            interaction.dependencyChannel = xsprite[candidate.extra].txID;
            setInteractionGeometry(interaction);
            setSpriteInteractionGeometry(interaction, i);
            interaction.z = interaction.target.z;
            result.interactions.push_back(interaction);
        }
    }
    return result;
}

std::vector<InteractionCandidate> engineUseAffordancesAtCurrentPose()
{
    std::vector<InteractionCandidate> result;
    if (!gMe || !gMe->pSprite)
        return result;

    spritetype &actor = *gMe->pSprite;
    const int16_t savedSpriteAngle = actor.ang;
    const fix16_t savedAngle = gMe->q16ang;
    const fix16_t savedLook = gMe->q16look;
    const fix16_t savedHoriz = gMe->q16horiz;
    const int savedSlope = gMe->slope;
    const HITINFO savedHit = gHitInfo;

    gMe->q16look = 0;
    gMe->q16horiz = 0;
    gMe->slope = 0;
    auto retain = [&](const InteractionCandidate &candidate) {
        const auto known = std::find_if(
            result.begin(), result.end(),
            [&](const InteractionCandidate &other) {
                return other.kind == candidate.kind
                    && other.id == candidate.id
                    && other.target.wall == candidate.target.wall;
            });
        if (known == result.end())
            result.push_back(candidate);
    };

    // ActionScan is a ray. Thirty-two directions are finer than a player's
    // body radius at maximum Use range and keep this query bounded.
    for (int angle = 0; angle < 2048; angle += 64)
    {
        actor.ang = int16_t(angle);
        gMe->q16ang = fix16_from_int(angle);
        int target = -1;
        int extra = -1;
        const int hit = ActionScanPreview(gMe, &target, &extra);
        if (hit == 3 && inRange(target, 0, kMaxSprites)
            && validXSprite(sprite[target].extra))
        {
            const XSPRITE &state = xsprite[sprite[target].extra];
            InteractionCandidate candidate;
            candidate.kind = kInteractionSprite;
            candidate.id = target;
            candidate.fromSector = actor.sectnum;
            candidate.targetSector = sprite[target].sectnum;
            candidate.x = sprite[target].x;
            candidate.y = sprite[target].y;
            candidate.z = sprite[target].z;
            // Blood's real Use path applies the same key/lock gate to
            // sprites as it does to walls and sectors. Publish that known
            // precondition now; making execution discover it by rejection
            // falsely advertises a distant keyed action as executable.
            candidate.key = state.key;
            candidate.reversible = true;
            candidate.target.from = actor.sectnum;
            candidate.target.to = sprite[target].sectnum;
            candidate.target.x = candidate.x;
            candidate.target.y = candidate.y;
            candidate.dependencyChannel = state.txID;
            setSpriteInteractionGeometry(candidate, target);
            candidate.z = candidate.target.z;
            retain(candidate);
            continue;
        }
        if (hit == 0 && inRange(target, 0, numwalls)
            && inRange(wall[target].extra, 1, kMaxXWalls))
        {
            const walltype &record = wall[target];
            const walltype &end = wall[record.point2];
            const XWALL &state = xwall[record.extra];
            InteractionCandidate candidate;
            candidate.kind = kInteractionWall;
            candidate.id = target;
            candidate.fromSector = actor.sectnum;
            candidate.targetSector = record.nextsector;
            candidate.x = (record.x + end.x) / 2;
            candidate.y = (record.y + end.y) / 2;
            candidate.z = gHitInfo.hitz;
            candidate.key = state.key;
            candidate.reversible = true;
            candidate.target.wall = target;
            candidate.target.from = actor.sectnum;
            candidate.target.to = record.nextsector;
            candidate.target.x = candidate.x;
            candidate.target.y = candidate.y;
            candidate.target.wallPush = true;
            candidate.dependencyChannel = state.txID;
            candidate.target.x1 = record.x;
            candidate.target.y1 = record.y;
            candidate.target.x2 = end.x;
            candidate.target.y2 = end.y;
            setInteractionGeometry(candidate);
            candidate.z = candidate.target.z;
            retain(candidate);
            continue;
        }
        if (hit != 6 || !inRange(target, 0, numsectors)
            || !inRange(sector[target].extra, 1, kMaxXSectors))
            continue;
        const XSECTOR &state = xsector[sector[target].extra];
        InteractionCandidate candidate;
        candidate.id = target;
        candidate.fromSector = actor.sectnum;
        candidate.targetSector = target;
        candidate.x = gHitInfo.hitx;
        candidate.y = gHitInfo.hity;
        candidate.z = gHitInfo.hitz;
        candidate.key = state.Key;
        candidate.reversible = true;
        candidate.target.from = actor.sectnum;
        candidate.target.to = target;
        candidate.target.x = candidate.x;
        candidate.target.y = candidate.y;
        if (target == actor.sectnum || !inRange(gHitInfo.hitwall, 0, numwalls))
        {
            candidate.kind = kInteractionSector;
            candidate.target.sectorPushCurrent = true;
        }
        else
        {
            const int wallId = gHitInfo.hitwall;
            const walltype &record = wall[wallId];
            const walltype &end = wall[record.point2];
            candidate.kind = kInteractionWall;
            candidate.x = (record.x + end.x) / 2;
            candidate.y = (record.y + end.y) / 2;
            candidate.target.wall = wallId;
            candidate.target.sectorPush = true;
            candidate.target.x = candidate.x;
            candidate.target.y = candidate.y;
            candidate.target.x1 = record.x;
            candidate.target.y1 = record.y;
            candidate.target.x2 = end.x;
            candidate.target.y2 = end.y;
        }
        setInteractionGeometry(candidate);
        candidate.z = candidate.target.z;
        retain(candidate);
    }

    actor.ang = savedSpriteAngle;
    gMe->q16ang = savedAngle;
    gMe->q16look = savedLook;
    gMe->q16horiz = savedHoriz;
    gMe->slope = savedSlope;
    gHitInfo = savedHit;
    return result;
}

const char *objectName(ObjectKind kind)
{
    switch (kind)
    {
    case kObjectEnemy: return "enemy";
    case kObjectKey: return "key";
    case kObjectInteractive: return "interactive";
    case kObjectDamageable: return "damageable";
    default: return "pickup";
    }
}
