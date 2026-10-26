//-------------------------------------------------------------------------
/*
Copyright (C) 2022, 2023 - Jordon Moss (Aka. StrikerTheHedgefox)
Copyright (C) 2026 - NBlood developers

Bot navigation. The coarse route is a breadth-first search over sectors and
their portals, as in NetDuke32's DukeBot_GetNextWallForPath(). Which portals
can be crossed, and how, is decided from Blood's sector data and the player's
real collision hull; local movement is checked with the engine's own
ClipMove().

This file is part of NBlood.

NBlood is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License version 2
as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/
//-------------------------------------------------------------------------
#include "build.h"
#include "common_game.h"
#include "blood.h"
#include "db.h"
#include "gameutil.h"
#include "globals.h"
#include "levels.h"
#include "player.h"
#include "trig.h"
#include "warp.h"
#include "botai.h"

// Heights are in z units. A standing jump lifts Caleb a little over 72
// pixels (jump velocity 0xbaaaa against gravity 58254 per frame); on top of
// that comes the step ClipMove lets the body climb anyway.
#define kBotJumpLift    (64<<8)
#define kBotMaxDrop     (320<<8)
#define kBotSafeDrop    (192<<8)    // falls taken without a reason
#define kBotRouteAge    kBotFPS
#define kBotCheckFrames 12
#define kBotAvoidFrames (kBotFPS*12)

static short botQueue[kMaxSectors];
static short botParent[kMaxSectors];
static short botVia[kMaxSectors];
static uint8_t botFlags[kMaxSectors];
// Where each queue entry was entered, and whether it is a sprite we stand on
// above the sector's real floor (a bridge over a pit).
static vec2_t botEntry[kMaxSectors];
static int botEntryZ[kMaxSectors];
static uint8_t botElevated[kMaxSectors];
static char botVisited[(kMaxSectors+7)>>3];

struct BOTBODY
{
    int nRadius, nCeilDist, nFloorDist;
    int nHeight, nCrouchHeight;
    int nStep;      // highest floor step walked up without jumping
};

static void botGetBody(PLAYER *pPlayer, BOTBODY *pBody)
{
    spritetype *pSprite = pPlayer->pSprite;
    int top, bottom;
    botGetExtents(pPlayer, &top, &bottom);
    pBody->nRadius = pSprite->clipdist<<2;
    pBody->nCeilDist = (pSprite->z-top)/4;
    pBody->nFloorDist = (bottom-pSprite->z)/4;
    pBody->nHeight = bottom-top;
    pBody->nCrouchHeight = pBody->nHeight*5/8;
    // ClipMove refuses a step when the body's origin is below the new floor
    // minus the floor distance; the origin sits (bottom-z) above the feet.
    pBody->nStep = ClipLow((bottom-pSprite->z)-pBody->nFloorDist-256, 0);
}

void botNavReset(BOTINFO *pBot)
{
    BOTNAV *pNav = &pBot->nav;
    memset(pNav, 0, sizeof(BOTNAV));
    pNav->nRouteFrom = pNav->nRouteTo = -1;
    pNav->nNextSect = -1;
    pNav->nPortal = -1;
    for (int i = 0; i < ARRAY_SSIZE(pNav->nAvoidWall); i++)
        pNav->nAvoidWall[i] = -1;
}

static bool botAvoided(BOTNAV *pNav, int nWall, int nFrame)
{
    for (int i = 0; i < ARRAY_SSIZE(pNav->nAvoidWall); i++)
        if (pNav->nAvoidWall[i] == nWall && pNav->nAvoidUntil[i] > nFrame)
            return true;
    return false;
}

static void botAvoid(BOTINFO *pBot, int nWall, const char *pzWhy)
{
    BOTNAV *pNav = &pBot->nav;
    if (nWall < 0)
        return;
    pBot->nAvoids++;
    if (gBotDebug)
        OSD_Printf("bot %d: avoiding wall %d (sector %d -> %d, flags %d): %s\n", int(pBot-gBotInfo), nWall,
            sectorofwall(nWall), (int)wall[nWall].nextsector, pNav->nPortalFlags, pzWhy);
    int i = pNav->nAvoidHead++ % ARRAY_SSIZE(pNav->nAvoidWall);
    pNav->nAvoidWall[i] = nWall;
    pNav->nAvoidUntil[i] = pBot->nFrame+kBotAvoidFrames;
    // The other side of the same portal is the same opening.
    int nNext = wall[nWall].nextwall;
    if (nNext >= 0)
    {
        i = pNav->nAvoidHead++ % ARRAY_SSIZE(pNav->nAvoidWall);
        pNav->nAvoidWall[i] = nNext;
        pNav->nAvoidUntil[i] = pBot->nFrame+kBotAvoidFrames;
    }
    pNav->nRouteFrom = pNav->nRouteTo = -1;
}

static XSECTOR *botXSector(int nSector)
{
    int nXSector = sector[nSector].extra;
    if (nXSector <= 0 || nXSector >= kMaxXSectors)
        return NULL;
    return &xsector[nXSector];
}

static bool botIsUnderwater(int nSector)
{
    XSECTOR *pXSector = botXSector(nSector);
    return pXSector && pXSector->Underwater;
}

// Doors, lifts and the like.
static bool botSectorMoves(int nSector)
{
    XSECTOR *pXSector = botXSector(nSector);
    if (!pXSector || pXSector->locked)
        return false;
    switch (sector[nSector].type)
    {
    case kSectorZMotion:
    case kSectorZMotionSprite:
    case kSectorSlide:
    case kSectorSlideMarked:
    case kSectorRotate:
    case kSectorRotateMarked:
    case kSectorRotateStep:
        return true;
    }
    return false;
}

static bool botWallPushable(int nWall)
{
    if (nWall < 0)
        return false;
    int nXWall = wall[nWall].extra;
    return nXWall > 0 && nXWall < kMaxXWalls && xwall[nXWall].triggerPush && !xwall[nXWall].locked;
}

// Whether pressing action at this portal can set a door going. Blood has
// three ways: the door sector reacts to a push on its walls (Wallpush) or
// on itself (Push), or the wall carries its own push trigger aimed at it.
static bool botPortalOperable(int nFrom, int nWall, int nTo)
{
    if (!botSectorMoves(nTo) && !botSectorMoves(nFrom))
        return false;
    if (botWallPushable(nWall) || botWallPushable(wall[nWall].nextwall))
        return true;
    int nSectors[2] = { nTo, nFrom };
    for (int i = 0; i < 2; i++)
    {
        XSECTOR *pXSector = botXSector(nSectors[i]);
        if (pXSector && botSectorMoves(nSectors[i]) && (pXSector->Wallpush || pXSector->Push || pXSector->Enter))
            return true;
    }
    return false;
}

static bool botSectorMovesFloor(int nSector)
{
    XSECTOR *pXSector = botXSector(nSector);
    if (!pXSector)
        return false;
    if (sector[nSector].type != kSectorZMotion && sector[nSector].type != kSectorZMotionSprite)
        return false;
    return pXSector->onFloorZ != pXSector->offFloorZ;
}

// Highest the ceiling of a moving sector ever gets.
static int botOpenCeiling(int nSector, int nCeil)
{
    XSECTOR *pXSector = botXSector(nSector);
    if (!pXSector || (sector[nSector].type != kSectorZMotion && sector[nSector].type != kSectorZMotionSprite))
        return nCeil;
    return ClipHigh(nCeil, ClipHigh(pXSector->onCeilZ, pXSector->offCeilZ));
}

static int botLowestFloor(int nSector, int nFloor)
{
    XSECTOR *pXSector = botXSector(nSector);
    if (!pXSector || (sector[nSector].type != kSectorZMotion && sector[nSector].type != kSectorZMotionSprite))
        return nFloor;
    return ClipLow(nFloor, ClipLow(pXSector->onFloorZ, pXSector->offFloorZ));
}

// The surface the body would stand on at (x,y) coming from height nRefZ:
// the sector floor, or a blocking sprite above it such as a bridge. Only
// what is below jumping reach of nRefZ counts, not lamps or pillar tops.
static int botSupportFloor(const BOTBODY *pBody, int nSector, int x, int y, int nRefZ)
{
    int nCeil, nFloor;
    getzsofslope(nSector, x, y, &nCeil, &nFloor);
    int nCeilZ, nCeilHit, nFloorZ, nFloorHit;
    int z = ClipLow(nCeil+256, nRefZ-kBotJumpLift-pBody->nStep);
    if (z >= nFloor)
        return nFloor;
    for (int i = 0; i < 4; i++)
    {
        GetZRangeAtXYZ(x, y, z, nSector, &nCeilZ, &nCeilHit, &nFloorZ, &nFloorHit, pBody->nRadius, CLIPMASK0);
        // Someone's head is not a bridge: look below them.
        if ((nFloorHit & 0xc000) != 0xc000 || sprite[nFloorHit & 0x3fff].statnum != kStatDude)
            break;
        int top, bottom;
        GetSpriteExtents(&sprite[nFloorHit & 0x3fff], &top, &bottom);
        z = bottom+256;
        nFloorZ = nFloor;
        if (z >= nFloor)
            break;
    }
    return ClipHigh(nFloorZ, nFloor);
}

// Can a player walk through this portal, given the heights on both sides?
static int botPortalCheck(const BOTBODY *pBody, int nFrom, int nWall, int nTo,
    int nFromCeil, int nFromFloor, int nToCeil, int nToFloor, const char **pzWhy)
{
    int nFlags = 0;
    bool bWater = botIsUnderwater(nFrom) || botIsUnderwater(nTo);
    int nWindow = ClipHigh(nFromFloor, nToFloor)-ClipLow(nFromCeil, nToCeil);
    if (nWindow < pBody->nCrouchHeight)
    {
        *pzWhy = "opening too low";
        if (!botPortalOperable(nFrom, nWall, nTo))
            return -1;
        *pzWhy = "door does not open far enough";
        int nOpenWindow = ClipHigh(nFromFloor, nToFloor)-ClipLow(botOpenCeiling(nFrom, nFromCeil), botOpenCeiling(nTo, nToCeil));
        // A floor that drops (lowering bridge) also opens the way.
        nOpenWindow = ClipLow(nOpenWindow, ClipHigh(botLowestFloor(nFrom, nFromFloor), botLowestFloor(nTo, nToFloor))-ClipLow(nFromCeil, nToCeil));
        if (nOpenWindow < pBody->nCrouchHeight)
            return -1;
        nFlags |= kBotPortalDoor;
    }
    else if (nWindow < pBody->nHeight)
        nFlags |= kBotPortalCrouch;

    int nRise = nFromFloor-nToFloor;
    if (nRise > kBotJumpLift+pBody->nStep && !bWater)
    {
        *pzWhy = "too high to jump";
        if (!botSectorMovesFloor(nTo) && !botSectorMovesFloor(nFrom))
            return -1;
        nFlags |= kBotPortalLift;
    }
    else if (nRise > pBody->nStep)
        nFlags |= kBotPortalJump;
    *pzWhy = "deadly drop";
    if (-nRise > kBotMaxDrop && !bWater)
        return -1;
    if (-nRise > kBotSafeDrop && !bWater)
        nFlags |= kBotPortalDrop;
    *pzWhy = "";
    return nFlags;
}

static void botPortalNormal(const BOTBODY *pBody, int nWall, int *nx, int *ny)
{
    int dx = wall[wall[nWall].point2].x-wall[nWall].x, dy = wall[wall[nWall].point2].y-wall[nWall].y;
    int nLength = ClipLow(ksqrt(dx*dx+dy*dy), 1);
    // Out of the sector owning the wall.
    *nx = (int)((int64_t)dy*(pBody->nRadius+48)/nLength);
    *ny = (int)((int64_t)-dx*(pBody->nRadius+48)/nLength);
}

// A portal over a pit or onto a high ledge may still be crossed where a
// sprite carries the body. Look along the opening for such a place, the
// one nearest (nx,ny) when asked. Returns the flags for crossing there.
static int botPortalBridge(const BOTBODY *pBody, int nFrom, int nWall, int nTo, int nearX, int nearY, vec2_t *pCross, vec2_t *pApproach)
{
    walltype *pWall = &wall[nWall];
    int dx = wall[pWall->point2].x-pWall->x, dy = wall[pWall->point2].y-pWall->y;
    int ox, oy;
    botPortalNormal(pBody, nWall, &ox, &oy);
    int nBest = -1, nBestDist = INT32_MAX;
    for (int i = 1; i < 8; i++)
    {
        int px = pWall->x+dx*i/8, py = pWall->y+dy*i/8;
        int tx = px+ox, ty = py+oy, fx = px-ox, fy = py-oy;
        if (inside(tx, ty, nTo) != 1 || inside(fx, fy, nFrom) != 1)
            continue;
        int nFromCeil, nFromFloor, nToCeil, nToFloor;
        getzsofslope(nFrom, px, py, &nFromCeil, &nFromFloor);
        getzsofslope(nTo, px, py, &nToCeil, &nToFloor);
        int nRef = ClipHigh(nFromFloor, nToFloor);
        nFromFloor = botSupportFloor(pBody, nFrom, fx, fy, nRef);
        nToFloor = botSupportFloor(pBody, nTo, tx, ty, nRef);
        const char *pzWhy;
        int nFlags = botPortalCheck(pBody, nFrom, nWall, nTo, nFromCeil, nFromFloor, nToCeil, nToFloor, &pzWhy);
        if (nFlags < 0)
            continue;
        int nDist = approxDist(px-nearX, py-nearY);
        if (nDist < nBestDist)
        {
            nBestDist = nDist;
            nBest = nFlags|kBotPortalBridge;
            if (pCross)
            {
                pCross->x = tx;
                pCross->y = ty;
                pApproach->x = fx;
                pApproach->y = fy;
            }
        }
    }
    return nBest;
}

// Can a player walk through this portal, and what does it take? Returns -1
// when the portal is not usable right now or ever as far as we can tell.
static int botPortalFlags(const BOTBODY *pBody, int nFrom, int nWall, int nTo, const char **pzWhy = NULL)
{
    const char *pzDummy;
    if (!pzWhy)
        pzWhy = &pzDummy;
    walltype *pWall = &wall[nWall];
    *pzWhy = "blocking wall";
    if (pWall->cstat & 1)
        return -1;
    int x = (pWall->x+wall[pWall->point2].x)>>1;
    int y = (pWall->y+wall[pWall->point2].y)>>1;
    int nFromCeil, nFromFloor, nToCeil, nToFloor;
    getzsofslope(nFrom, x, y, &nFromCeil, &nFromFloor);
    getzsofslope(nTo, x, y, &nToCeil, &nToFloor);
    int nFlags = botPortalCheck(pBody, nFrom, nWall, nTo, nFromCeil, nFromFloor, nToCeil, nToFloor, pzWhy);
    // Sector floors say no or say "fall": a sprite may still carry us over.
    if (nFlags < 0 && (!strcmp(*pzWhy, "deadly drop") || !strcmp(*pzWhy, "too high to jump")))
    {
        int nBridge = botPortalBridge(pBody, nFrom, nWall, nTo, x, y, NULL, NULL);
        if (nBridge >= 0)
        {
            *pzWhy = "";
            return nBridge;
        }
    }
    else if (nFlags >= 0 && (nFlags & kBotPortalDrop))
    {
        int nBridge = botPortalBridge(pBody, nFrom, nWall, nTo, x, y, NULL, NULL);
        if (nBridge >= 0 && !(nBridge & kBotPortalDrop))
            return nBridge;
    }
    return nFlags;
}

static int botLinkDestination(int nLink)
{
    if (nLink >= 0 && sprite[nLink].owner >= 0 && sprite[nLink].owner < kMaxSprites)
        return sprite[sprite[nLink].owner].sectnum;
    return -1;
}

static int botTeleportDestination(int nSector)
{
    if (sector[nSector].type != kSectorTeleport)
        return -1;
    XSECTOR *pXSector = botXSector(nSector);
    if (!pXSector || pXSector->locked)
        return -1;
    int nMarker = pXSector->marker0;
    if (nMarker <= 0 || nMarker >= kMaxSprites || sprite[nMarker].statnum != kStatMarker)
        return -1;
    return sprite[nMarker].sectnum;
}

// Standing on a sprite well above the sector floor?
static bool botOnSprite(const BOTBODY *pBody, int nSector, int x, int y, int nRefZ, int *pZ)
{
    *pZ = botSupportFloor(pBody, nSector, x, y, nRefZ);
    return getflorzofslope(nSector, x, y)-*pZ > kBotSafeDrop;
}

// Ground all the way from a to b inside a sector, at about the same height.
static bool botSupportedPath(const BOTBODY *pBody, int nSector, vec2_t a, int nZ, vec2_t b)
{
    int nDist = approxDist(b.x-a.x, b.y-a.y);
    for (int nStep = 0; nStep <= nDist; nStep += 128)
    {
        int x = a.x+(int)((int64_t)(b.x-a.x)*nStep/ClipLow(nDist, 1));
        int y = a.y+(int)((int64_t)(b.y-a.y)*nStep/ClipLow(nDist, 1));
        if (botSupportFloor(pBody, nSector, x, y, nZ)-nZ > pBody->nStep)
            return false;
    }
    return true;
}

// Breadth-first search from nFrom to nTo. On success returns the index of the
// queue entry holding the first step and the number of steps. A sector only
// stood in on a bridge is left only where the bridge leads, so the search
// keeps track of where each sector was entered.
static int botSearch(BOTINFO *pBot, const BOTBODY *pBody, int nFrom, int nTo, int *pLength, const vec3_t *pStart = NULL)
{
    // nTo < 0 floods everything reachable (botVisited) and finds nothing.
    if ((unsigned)nFrom >= (unsigned)numsectors || nTo >= numsectors)
        return -1;
    BOTNAV *pNav = pBot ? &pBot->nav : NULL;
    memset(botVisited, 0, sizeof(botVisited));
    botQueue[0] = nFrom;
    botParent[0] = -1;
    botVia[0] = -1;
    botFlags[0] = 0;
    botElevated[0] = 0;
    if (pStart)
    {
        botEntry[0].x = pStart->x;
        botEntry[0].y = pStart->y;
        botElevated[0] = botOnSprite(pBody, nFrom, pStart->x, pStart->y, pStart->z, &botEntryZ[0]);
    }
    SetBitString(botVisited, nFrom);
    int nFound = -1;
    int nTail = 1;
    for (int nHead = 0; nHead < nTail && nFound < 0 && nTail < kMaxSectors-1; nHead++)
    {
        int nSector = botQueue[nHead];
        int nStart = sector[nSector].wallptr, nEnd = nStart+sector[nSector].wallnum;
        for (int nWall = nStart; nWall < nEnd && nTail < kMaxSectors-1; nWall++)
        {
            int nNext = wall[nWall].nextsector;
            if (nNext < 0)
                continue;
            if (pNav && botAvoided(pNav, nWall, pBot->nFrame))
                continue;
            int nFlags = botPortalFlags(pBody, nSector, nWall, nNext);
            if (nFlags < 0)
                continue;
            vec2_t cross;
            cross.x = (wall[nWall].x+wall[wall[nWall].point2].x)>>1;
            cross.y = (wall[nWall].y+wall[wall[nWall].point2].y)>>1;
            bool bElevated = false;
            if (botElevated[nHead] || (nFlags & kBotPortalBridge))
            {
                vec2_t approach;
                vec2_t aim = botElevated[nHead] ? botEntry[nHead] : cross;
                if (botPortalBridge(pBody, nSector, nWall, nNext, aim.x, aim.y, &cross, &approach) < 0)
                    continue;
                if (botElevated[nHead] && !botSupportedPath(pBody, nSector, botEntry[nHead], botEntryZ[nHead], approach))
                    continue;
                int nZ;
                int nRef = botElevated[nHead] ? botEntryZ[nHead] : getflorzofslope(nSector, approach.x, approach.y);
                bElevated = botOnSprite(pBody, nNext, cross.x, cross.y, nRef, &nZ);
                botEntryZ[nTail] = nZ;
            }
            if (bElevated)
            {
                // The same pit may hold several bridges: it is a new place
                // unless we already came onto this one.
                bool bSeen = false;
                for (int i = 1; i < nTail && !bSeen; i++)
                    bSeen = botQueue[i] == nNext && approxDist(botEntry[i].x-cross.x, botEntry[i].y-cross.y) < 1024;
                if (bSeen)
                    continue;
            }
            else if (TestBitString(botVisited, nNext))
                continue;
            SetBitString(botVisited, nNext);
            botQueue[nTail] = nNext;
            botParent[nTail] = nHead;
            botVia[nTail] = nWall;
            botFlags[nTail] = nFlags;
            botEntry[nTail] = cross;
            botElevated[nTail] = bElevated;
            if (nNext == nTo)
            {
                nFound = nTail++;
                break;
            }
            nTail++;
        }
        if (nFound >= 0)
            break;
        // Water surfaces, stacked sectors and teleporters connect sectors without walls.
        int nDests[3] = { botLinkDestination(gUpperLink[nSector]), botLinkDestination(gLowerLink[nSector]), botTeleportDestination(nSector) };
        for (int i = 0; i < 3 && nFound < 0; i++)
        {
            int nNext = nDests[i];
            if (nNext < 0 || nNext >= numsectors || TestBitString(botVisited, nNext) || nTail >= kMaxSectors-1)
                continue;
            SetBitString(botVisited, nNext);
            botQueue[nTail] = nNext;
            botParent[nTail] = nHead;
            botVia[nTail] = -1;
            botFlags[nTail] = kBotPortalLink;
            botElevated[nTail] = 0;
            if (nNext == nTo)
                nFound = nTail;
            nTail++;
        }
    }
    if (nFound < 0)
        return -1;
    int nLength = 0;
    int nStep = nFound;
    while (botParent[nStep] > 0)
    {
        nStep = botParent[nStep];
        nLength++;
    }
    if (pLength)
        *pLength = nLength+1;
    return nStep;
}

bool botNavReachable(PLAYER *pPlayer, int nSector)
{
    if (pPlayer->pSprite->sectnum == nSector)
        return true;
    BOTBODY body;
    botGetBody(pPlayer, &body);
    BOTINFO *pBot = &gBotInfo[pPlayer->nPlayer];
    spritetype *pSprite = pPlayer->pSprite;
    vec3_t start = { pSprite->x, pSprite->y, pSprite->z+(body.nFloorDist<<2) };
    return botSearch(pBot, &body, pSprite->sectnum, nSector, NULL, &start) >= 0;
}

// Point beyond the portal where the body fits, preferring the part of the
// opening nearest to us.
static void botPortalCrossing(const BOTBODY *pBody, int nWall, int x, int y, vec2_t *pCross, vec2_t *pApproach)
{
    walltype *pWall = &wall[nWall];
    walltype *pWall2 = &wall[pWall->point2];
    int x1 = pWall->x, y1 = pWall->y;
    int dx = pWall2->x-x1, dy = pWall2->y-y1;
    int nLength = ClipLow(ksqrt(dx*dx+dy*dy), 1);
    int nMargin = pBody->nRadius+32;
    int t;
    if (nLength <= nMargin*2)
        t = nLength>>1;
    else
    {
        int64_t dot = (int64_t)(x-x1)*dx+(int64_t)(y-y1)*dy;
        t = ClipRange((int)(dot/nLength), nMargin, nLength-nMargin);
    }
    int px = x1+(int)((int64_t)dx*t/nLength);
    int py = y1+(int)((int64_t)dy*t/nLength);
    // Normal pointing out of the sector that owns the wall.
    int nx = (int)((int64_t)dy*(pBody->nRadius+48)/nLength);
    int ny = (int)((int64_t)-dx*(pBody->nRadius+48)/nLength);
    pCross->x = px+nx;
    pCross->y = py+ny;
    pApproach->x = px-nx;
    pApproach->y = py-ny;
}

// Distance the body actually gets toward (dx,dy) when sliding from (x,y,z).
static int botClearDist(const BOTBODY *pBody, int x, int y, int z, int nSector, int dx, int dy, int *pHit = NULL)
{
    int nx = x, ny = y, nz = z, nSect = nSector;
    unsigned int nHit = ClipMove(&nx, &ny, &nz, &nSect, dx, dy, pBody->nRadius, pBody->nCeilDist, pBody->nFloorDist, CLIPMASK0);
    if (pHit)
        *pHit = nHit;
    int nLength = ClipLow(ksqrt(dx*dx+dy*dy), 1);
    int64_t dot = (int64_t)(nx-x)*dx+(int64_t)(ny-y)*dy;
    return (int)(dot/nLength);
}

int botNavProbe(PLAYER *pPlayer, int nAng, int nDist, int *pHit)
{
    BOTBODY body;
    botGetBody(pPlayer, &body);
    spritetype *pSprite = pPlayer->pSprite;
    int dx = mulscale30(nDist, Cos(nAng));
    int dy = mulscale30(nDist, Sin(nAng));
    return botClearDist(&body, pSprite->x, pSprite->y, pSprite->z, pSprite->sectnum, dx, dy, pHit);
}

static bool botStraightClear(const BOTBODY *pBody, int x, int y, int z, int nSector, int tx, int ty)
{
    int dx = tx-x, dy = ty-y;
    int nDist = ksqrt(dx*dx+dy*dy);
    if (nDist < 16)
        return true;
    if (botClearDist(pBody, x, y, z, nSector, dx, dy) < nDist-24)
        return false;
    // A sudden drop somewhere along the way? (Ramps and stairs are fine.)
    int nLast = z+(pBody->nFloorDist<<2);
    for (int nStep = 256; nStep < nDist; nStep += 256)
    {
        int sx = x+(int)((int64_t)dx*nStep/nDist), sy = y+(int)((int64_t)dy*nStep/nDist);
        short nSect = nSector;
        updatesector(sx, sy, &nSect);
        if (nSect < 0)
            return false;
        int nSupport = botSupportFloor(pBody, nSect, sx, sy, nLast);
        if (nSupport-nLast > kBotSafeDrop)
            return false;
        nLast = nSupport;
    }
    return true;
}

bool botNavOnBridge(PLAYER *pPlayer)
{
    BOTBODY body;
    botGetBody(pPlayer, &body);
    int nZ;
    spritetype *pSprite = pPlayer->pSprite;
    return botOnSprite(&body, pSprite->sectnum, pSprite->x, pSprite->y, pSprite->z+(body.nFloorDist<<2), &nZ);
}

bool botNavLedge(PLAYER *pPlayer, int nAng, int nDist)
{
    BOTBODY body;
    botGetBody(pPlayer, &body);
    spritetype *pSprite = pPlayer->pSprite;
    int nFeet = pSprite->z+(body.nFloorDist<<2);
    for (int nStep = nDist/2; nStep <= nDist; nStep += nDist/2)
    {
        int x = pSprite->x+mulscale30(nStep, Cos(nAng)), y = pSprite->y+mulscale30(nStep, Sin(nAng));
        short nSect = pSprite->sectnum;
        updatesector(x, y, &nSect);
        if (nSect < 0)
            continue;
        if (botSupportFloor(&body, nSect, x, y, nFeet)-nFeet > kBotSafeDrop)
            return true;
    }
    return false;
}

// Last check before moving: do not walk off an edge unless the route means to.
void botNavSafeMove(BOTINFO *pBot, PLAYER *pPlayer, BOTMOVE *pMove)
{
    if (!pMove->nSpeed || (!pMove->dx && !pMove->dy) || pPlayer->posture == kPostureSwim)
        return;
    if ((pBot->nav.nPortalFlags & kBotPortalDrop) && pBot->nav.nNextSect >= 0)
        return;
    int nSprite = pPlayer->nSprite;
    if (approxDist(xvel[nSprite], yvel[nSprite]) > 0x20000 && botNavLedge(pPlayer, getangle(xvel[nSprite], yvel[nSprite]), 448))
    {
        // Sliding toward an edge: push back against it.
        int nBack = getangle(-xvel[nSprite], -yvel[nSprite]);
        pMove->dx = Cos(nBack)>>16;
        pMove->dy = Sin(nBack)>>16;
        pMove->nSpeed = 2048;
        return;
    }
    int nAng = getangle(pMove->dx, pMove->dy);
    if (!botNavLedge(pPlayer, nAng, 448))
        return;
    for (int i = 1; i <= 4; i++)
    {
        int nTry = (nAng+((i+1)>>1)*256*((i&1) ? 1 : -1))&kAngMask;
        if (!botNavLedge(pPlayer, nTry, 448))
        {
            pMove->dx = Cos(nTry)>>16;
            pMove->dy = Sin(nTry)>>16;
            return;
        }
    }
    pMove->nSpeed = 0;
}

// The current sector may be concave or hold pillars. When the straight line
// to the target is blocked, walk to a corner of the sector from which the
// target can be reached.
static bool botCornerDetour(const BOTBODY *pBody, PLAYER *pPlayer, int tx, int ty, vec2_t *pVia)
{
    spritetype *pSprite = pPlayer->pSprite;
    int nSector = pSprite->sectnum;
    int nStart = sector[nSector].wallptr, nEnd = nStart+sector[nSector].wallnum;
    int nBest = INT32_MAX;
    int nOffset = pBody->nRadius+48;
    for (int nWall = nStart; nWall < nEnd; nWall++)
    {
        walltype *pWall = &wall[nWall];
        // Previous wall of the same loop.
        int nPrev = nWall;
        for (int i = nStart; i < nEnd; i++)
            if (wall[i].point2 == nWall)
            {
                nPrev = i;
                break;
            }
        int vx = pWall->x, vy = pWall->y;
        int ax = wall[pWall->point2].x-vx, ay = wall[pWall->point2].y-vy;
        int bx = vx-wall[nPrev].x, by = vy-wall[nPrev].y;
        int la = ClipLow(ksqrt(ax*ax+ay*ay), 1), lb = ClipLow(ksqrt(bx*bx+by*by), 1);
        // Inward normals of both walls, summed.
        int nx = -ay*1024/la-by*1024/lb;
        int ny = ax*1024/la+bx*1024/lb;
        int nl = ksqrt(nx*nx+ny*ny);
        if (nl < 64)
            continue;
        int px = vx+nx*nOffset/nl;
        int py = vy+ny*nOffset/nl;
        if (inside(px, py, nSector) != 1)
            continue;
        int nCost = approxDist(px-pSprite->x, py-pSprite->y)+approxDist(tx-px, ty-py);
        if (nCost >= nBest)
            continue;
        if (approxDist(px-pSprite->x, py-pSprite->y) < pBody->nRadius)
            continue;
        if (!botStraightClear(pBody, pSprite->x, pSprite->y, pSprite->z, nSector, px, py))
            continue;
        int z = getflorzofslope(nSector, px, py)-(pBody->nFloorDist<<2)-256;
        if (!botStraightClear(pBody, px, py, ClipHigh(z, pSprite->z), nSector, tx, ty))
            continue;
        nBest = nCost;
        pVia->x = px;
        pVia->y = py;
    }
    return nBest != INT32_MAX;
}

// Choose a heading toward (tx,ty) that the body can actually move along.
static int botSteer(BOTINFO *pBot, const BOTBODY *pBody, PLAYER *pPlayer, int tx, int ty, int ax, int ay)
{
    spritetype *pSprite = pPlayer->pSprite;
    BOTNAV *pNav = &pBot->nav;
    int nDirect = getangle(tx-pSprite->x, ty-pSprite->y);
    if (pNav->nSteerUntil > pBot->nFrame)
        return pNav->nSteerAng;
    if (botStraightClear(pBody, pSprite->x, pSprite->y, pSprite->z, pSprite->sectnum, ax, ay))
        return nDirect;
    vec2_t via;
    if (botCornerDetour(pBody, pPlayer, ax, ay, &via))
    {
        pNav->nSteerAng = getangle(via.x-pSprite->x, via.y-pSprite->y);
        pNav->nSteerUntil = pBot->nFrame+ClipRange(approxDist(via.x-pSprite->x, via.y-pSprite->y)/48, 3, 20);
        return pNav->nSteerAng;
    }
    // Feel around: the heading closest to the target that gets us somewhere.
    // On a bridge the headings are tried finer, and none may leave it.
    bool bBridge = botNavOnBridge(pPlayer);
    int nStep = bBridge ? 48 : 128;
    int nBestAng = nDirect, nBestScore = INT32_MIN;
    for (int i = 0; i <= (bBridge ? 20 : 12); i++)
    {
        int nDelta = ((i+1)>>1)*nStep*((i&1) ? 1 : -1);
        int nAng = (nDirect+nDelta)&kAngMask;
        if (botNavLedge(pPlayer, nAng, 384))
            continue;
        int nGain = botNavProbe(pPlayer, nAng, 384);
        int nScore = nGain*(4096+(Cos(nDelta)>>18))/4096-klabs(nDelta)/8;
        if (nScore > nBestScore)
        {
            nBestScore = nScore;
            nBestAng = nAng;
        }
    }
    pNav->nSteerAng = nBestAng;
    pNav->nSteerUntil = pBot->nFrame+6;
    return nBestAng;
}

static void botFindRoute(BOTINFO *pBot, const BOTBODY *pBody, int nFrom, int nTo, const vec3_t *pStart)
{
    BOTNAV *pNav = &pBot->nav;
    pNav->nRouteFrom = nFrom;
    pNav->nRouteTo = nTo;
    pNav->nRouteFrame = pBot->nFrame;
    pNav->nNextSect = -1;
    pNav->nPortal = -1;
    int nLength = 0;
    int nStep = botSearch(pBot, pBody, nFrom, nTo, &nLength, pStart);
    if (nStep < 0)
        return;
    pNav->nNextSect = botQueue[nStep];
    pNav->nPortal = botVia[nStep];
    pNav->nPortalFlags = botFlags[nStep];
    pNav->nRouteLen = nLength;
}

// Escape: something keeps us from moving. Try the obvious things a player
// would try, then pick another way.
static void botCheckProgress(BOTINFO *pBot, PLAYER *pPlayer, BOTMOVE *pMove)
{
    spritetype *pSprite = pPlayer->pSprite;
    BOTNAV *pNav = &pBot->nav;
    if (pBot->nFrame-pNav->nCheckFrame < kBotCheckFrames)
        return;
    int nMoved = approxDist(pSprite->x-pNav->checkPos.x, pSprite->y-pNav->checkPos.y);
    pNav->checkPos.x = pSprite->x;
    pNav->checkPos.y = pSprite->y;
    pNav->nCheckFrame = pBot->nFrame;
    if (nMoved > 160 || !pMove->nSpeed)
    {
        pNav->nStuck = 0;
        return;
    }
    pNav->nStuck++;
    switch (pNav->nStuck)
    {
    case 1:
        pMove->bJump = true;
        break;
    case 2:
        pMove->bAction = true;
        pNav->nSteerUntil = 0;
        break;
    case 3:
    case 5:
        pNav->nEscape = 10+botRandRange(pBot, 15);
        pNav->nEscapeAng = (getangle(pMove->dx, pMove->dy)+1024+botRandRange(pBot, 1024)-512)&kAngMask;
        break;
    case 4:
        pMove->bJump = true;
        pMove->bCrouch = false;
        break;
    default:
        // This portal does not work for us at the moment.
        botAvoid(pBot, pNav->nPortal, "no progress");
        pNav->nStuck = 0;
        break;
    }
}

// What the route search sees around a bot that cannot get out of a sector.
static void botNavDump(BOTINFO *pBot, PLAYER *pPlayer, const BOTBODY *pBody, int nGoal, const char *pzWhy)
{
    if (!gBotDebug)
        return;
    spritetype *pSprite = pPlayer->pSprite;
    BOTNAV *pNav = &pBot->nav;
    int nSector = pSprite->sectnum;
    sectortype *pSector = &sector[nSector];
    OSD_Printf("bot %d nav (%s): at %d,%d,%d sector %d (type %d floor %d ceil %d) goal %d next %d portal %d flags %d\n",
        pPlayer->nPlayer, pzWhy, (int)pSprite->x, (int)pSprite->y, (int)pSprite->z, nSector, (int)pSector->type,
        (int)pSector->floorz, (int)pSector->ceilingz, nGoal, pNav->nNextSect, pNav->nPortal, pNav->nPortalFlags);
    OSD_Printf("  body radius %d height %d crouch %d floordist %d ceildist %d\n",
        pBody->nRadius, pBody->nHeight, pBody->nCrouchHeight, pBody->nFloorDist, pBody->nCeilDist);
    for (int nWall = pSector->wallptr; nWall < pSector->wallptr+pSector->wallnum; nWall++)
    {
        int nNext = wall[nWall].nextsector;
        if (nNext < 0)
            continue;
        int x = (wall[nWall].x+wall[wall[nWall].point2].x)>>1;
        int y = (wall[nWall].y+wall[wall[nWall].point2].y)>>1;
        int nFromCeil, nFromFloor, nToCeil, nToFloor;
        getzsofslope(nSector, x, y, &nFromCeil, &nFromFloor);
        getzsofslope(nNext, x, y, &nToCeil, &nToFloor);
        OSD_Printf("  wall %d -> %d cstat %d flags %d rise %d window %d nexttype %d\n", nWall, nNext, (int)wall[nWall].cstat,
            botPortalFlags(pBody, nSector, nWall, nNext), nFromFloor-nToFloor,
            ClipHigh(nFromFloor, nToFloor)-ClipLow(nFromCeil, nToCeil), (int)sector[nNext].type);
    }
}

bool botNavMove(BOTINFO *pBot, PLAYER *pPlayer, int x, int y, int z, int nSector, BOTMOVE *pMove)
{
    spritetype *pSprite = pPlayer->pSprite;
    BOTNAV *pNav = &pBot->nav;
    BOTBODY body;
    botGetBody(pPlayer, &body);
    int nCur = pSprite->sectnum;
    pMove->nSpeed = 2048;

    if (pNav->nEscape > 0)
    {
        pNav->nEscape--;
        pMove->dx = Cos(pNav->nEscapeAng)>>16;
        pMove->dy = Sin(pNav->nEscapeAng)>>16;
        if (pNav->nEscape == 4)
            pMove->bJump = true;
        return true;
    }

    int tx = x, ty = y;     // where we head
    int ax = x, ay = y;     // last point inside the current sector
    int nFlags = 0;
    bool bLink = false;
    if (nSector >= 0 && nSector != nCur)
    {
        if (pNav->nRouteFrom != nCur || pNav->nRouteTo != nSector || pBot->nFrame-pNav->nRouteFrame > kBotRouteAge)
        {
            vec3_t start = { pSprite->x, pSprite->y, pSprite->z+(body.nFloorDist<<2) };
            botFindRoute(pBot, &body, nCur, nSector, &start);
        }
        if (pNav->nNextSect >= 0)
        {
            nFlags = pNav->nPortalFlags;
            if (pNav->nPortal >= 0)
            {
                vec2_t cross, approach;
                if (!(nFlags & kBotPortalBridge)
                    || botPortalBridge(&body, nCur, pNav->nPortal, pNav->nNextSect, pSprite->x, pSprite->y, &cross, &approach) < 0)
                    botPortalCrossing(&body, pNav->nPortal, pSprite->x, pSprite->y, &cross, &approach);
                pNav->crossing = cross;
                tx = cross.x;
                ty = cross.y;
                ax = approach.x;
                ay = approach.y;
            }
            else
            {
                // Link or teleporter: get into the middle of the sector.
                bLink = true;
                int cx = 0, cy = 0, nWalls = sector[nCur].wallnum, nStart = sector[nCur].wallptr;
                for (int i = 0; i < nWalls; i++)
                {
                    cx += wall[nStart+i].x;
                    cy += wall[nStart+i].y;
                }
                cx /= ClipLow(nWalls, 1);
                cy /= ClipLow(nWalls, 1);
                if (inside(cx, cy, nCur) != 1)
                {
                    cx = pSprite->x;
                    cy = pSprite->y;
                }
                tx = ax = cx;
                ty = ay = cy;
            }
            // Moving about without ever leaving the sector: this way does
            // not work for us right now.
            if (pNav->nStallSect != nCur)
            {
                pNav->nStallSect = nCur;
                pNav->nStallFrames = 0;
            }
            else if (++pNav->nStallFrames > kBotFPS*10)
            {
                botNavDump(pBot, pPlayer, &body, nSector, "stalled");
                botAvoid(pBot, pNav->nPortal, "stalled in sector");
                pNav->nStallFrames = 0;
            }
        }
        else
        {
            if (pNav->nRouteFrame == pBot->nFrame)
                botNavDump(pBot, pPlayer, &body, nSector, "no route");
            botCheckProgress(pBot, pPlayer, pMove);
            return false;
        }
    }

    int nDist = approxDist(tx-pSprite->x, ty-pSprite->y);
    int nAng = botSteer(pBot, &body, pPlayer, tx, ty, ax, ay);
    pMove->dx = Cos(nAng)>>16;
    pMove->dy = Sin(nAng)>>16;

    bool bSwimming = pPlayer->posture == kPostureSwim;
    if (bLink)
    {
        // Sink through a water surface or the floor of a stack, or rise
        // through the ceiling.
        if (gUpperLink[nCur] >= 0)
            pMove->bCrouch = true;
        else if (gLowerLink[nCur] >= 0)
            pMove->bJump = true;
        else
            pMove->bAction = (pBot->nFrame&15) == 0; // teleporter switch
        if (nDist < 256)
            pMove->nSpeed = 512;
    }
    else if (pNav->nPortal >= 0 && nSector != nCur)
    {
        walltype *pWall = &wall[pNav->nPortal];
        int nWallDist = approxDist(pSprite->x-((pWall->x+wall[pWall->point2].x)>>1), pSprite->y-((pWall->y+wall[pWall->point2].y)>>1));
        if ((nFlags & kBotPortalJump) && nWallDist < body.nRadius+320)
            pMove->bJump = true;
        if ((nFlags & kBotPortalCrouch) && nWallDist < body.nRadius+512)
            pMove->bCrouch = true;
        if (nFlags & (kBotPortalDoor|kBotPortalLift))
        {
            // Re-evaluate the live geometry: open now?
            int nNow = botPortalFlags(&body, nCur, pNav->nPortal, pNav->nNextSect);
            bool bOpen = nNow >= 0 && !(nNow & (kBotPortalDoor|kBotPortalLift));
            if (!bOpen && nWallDist < body.nRadius+900)
            {
                pMove->nFaceAng = getangle(tx-pSprite->x, ty-pSprite->y);
                pMove->nSpeed = nWallDist < body.nRadius+256 ? 0 : 1024;
                pMove->bAction = (pNav->nDoorWait % 20) == 0;
                pNav->nDoorWait++;
                pNav->checkPos.x = pSprite->x;
                pNav->checkPos.y = pSprite->y;
                pNav->nCheckFrame = pBot->nFrame;
                if (pNav->nDoorWait > kBotFPS*((nFlags & kBotPortalLift) ? 10 : 5))
                {
                    botAvoid(pBot, pNav->nPortal, "did not open");
                    pNav->nDoorWait = 0;
                }
                return true;
            }
            if (bOpen)
            {
                if (pNav->nDoorWait > 0)
                    pBot->nDoorsOpened++;
                pNav->nDoorWait = 0;
            }
            if (nNow >= 0 && (nNow & kBotPortalJump))
                pMove->bJump = nWallDist < body.nRadius+320;
        }
    }

    if (bSwimming && !bLink)
    {
        int nGoalZ = z;
        if (nSector != nCur && pNav->nNextSect >= 0)
            nGoalZ = getflorzofslope(pNav->nNextSect, tx, ty)-body.nHeight;
        if (nGoalZ < pSprite->z-(16<<8))
            pMove->bJump = true;
        else if (nGoalZ > pSprite->z+(16<<8))
            pMove->bCrouch = true;
    }

    botCheckProgress(pBot, pPlayer, pMove);
    return true;
}

// Somewhere worth a look: sectors holding pickups or start spots, chosen at
// random among those we can reach.
int botNavRoamSector(BOTINFO *pBot, PLAYER *pPlayer)
{
    for (int nTry = 0; nTry < 12; nTry++)
    {
        int nSector = -1;
        int nPick = botRandRange(pBot, 3);
        if (nPick == 0)
        {
            ZONE *pZone = &gStartZone[botRandRange(pBot, kMaxPlayers)];
            nSector = pZone->sectnum;
        }
        else if (nPick == 1)
        {
            int nCount = 0;
            for (int nSprite = headspritestat[kStatItem]; nSprite >= 0; nSprite = nextspritestat[nSprite])
                nCount++;
            if (nCount)
            {
                int n = botRandRange(pBot, nCount);
                for (int nSprite = headspritestat[kStatItem]; nSprite >= 0; nSprite = nextspritestat[nSprite])
                    if (!n--)
                    {
                        nSector = sprite[nSprite].sectnum;
                        break;
                    }
            }
        }
        if (nSector < 0)
            nSector = botRandRange(pBot, numsectors);
        if (nSector == pPlayer->pSprite->sectnum || (unsigned)nSector >= (unsigned)numsectors)
            continue;
        if (botNavReachable(pPlayer, nSector))
            return nSector;
    }
    return -1;
}

// bot_debug 2: which parts of the map the route search can reach from the
// start spots, and why it refuses the portals at the edge of that area.
void botNavReport(PLAYER *pPlayer)
{
    BOTBODY body;
    botGetBody(pPlayer, &body);
    static char reached[(kMaxSectors+7)>>3];
    memset(reached, 0, sizeof(reached));
    for (int i = 0; i < kMaxPlayers; i++)
    {
        int nStart = gStartZone[i].sectnum;
        if (nStart < 0 || nStart >= numsectors || TestBitString(reached, nStart))
            continue;
        botSearch(NULL, &body, nStart, -2, NULL);
        for (int j = 0; j < kMaxSectors; j++)
            if (TestBitString(botVisited, j))
                SetBitString(reached, j);
    }
    int nReached = 0;
    for (int i = 0; i < numsectors; i++)
        nReached += TestBitString(reached, i) ? 1 : 0;
    OSD_Printf("Bot nav report: %d of %d sectors reachable from the start spots (step %d, jump %d, height %d, crouch %d)\n",
        nReached, numsectors, body.nStep, kBotJumpLift+body.nStep, body.nHeight, body.nCrouchHeight);
    for (int nSector = 0; nSector < numsectors; nSector++)
    {
        int nDest[3] = { botLinkDestination(gUpperLink[nSector]), botLinkDestination(gLowerLink[nSector]), botTeleportDestination(nSector) };
        if (sector[nSector].type == kSectorTeleport || nDest[0] >= 0 || nDest[1] >= 0)
            OSD_Printf("  link sector %d (type %d): down %d up %d teleport %d\n", nSector, (int)sector[nSector].type, nDest[0], nDest[1], nDest[2]);
    }
    for (int nSector = 0; nSector < numsectors; nSector++)
    {
        for (int nWall = sector[nSector].wallptr; nWall < sector[nSector].wallptr+sector[nSector].wallnum; nWall++)
        {
            int nNext = wall[nWall].nextsector;
            if (nNext < 0)
                continue;
            int nFlags = botPortalFlags(&body, nSector, nWall, nNext);
            if (nFlags >= 0 && (nFlags & kBotPortalBridge))
                OSD_Printf("  bridge %d -> %d (wall %d, flags %d)\n", nSector, nNext, nWall, nFlags);
        }
    }
    // Traps: reachable sectors with no way back to the first start spot.
    static char trapped[(kMaxSectors+7)>>3];
    memset(trapped, 0, sizeof(trapped));
    int nHome = gStartZone[0].sectnum;
    for (int nSector = 0; nSector < numsectors; nSector++)
    {
        if (!TestBitString(reached, nSector) || nSector == nHome)
            continue;
        botSearch(NULL, &body, nSector, -2, NULL);
        if (!TestBitString(botVisited, nHome))
        {
            SetBitString(trapped, nSector);
            OSD_Printf("  trap sector %d (floor %d, type %d)\n", nSector, (int)sector[nSector].floorz, (int)sector[nSector].type);
        }
    }
    for (int nSector = 0; nSector < numsectors; nSector++)
    {
        if (!TestBitString(trapped, nSector))
            continue;
        for (int nWall = sector[nSector].wallptr; nWall < sector[nSector].wallptr+sector[nSector].wallnum; nWall++)
        {
            int nNext = wall[nWall].nextsector;
            if (nNext < 0 || TestBitString(trapped, nNext))
                continue;
            const char *pzWhy;
            botPortalFlags(&body, nSector, nWall, nNext, &pzWhy);
            int x = (wall[nWall].x+wall[wall[nWall].point2].x)>>1;
            int y = (wall[nWall].y+wall[wall[nWall].point2].y)>>1;
            OSD_Printf("  trap exit %d -> %d (wall %d at %d,%d, next type %d): %s, rise %d\n", nSector, nNext, nWall, x, y,
                (int)sector[nNext].type, pzWhy, getflorzofslope(nSector, x, y)-getflorzofslope(nNext, x, y));
        }
    }
    // And how the traps are entered.
    for (int nSector = 0; nSector < numsectors; nSector++)
    {
        if (!TestBitString(reached, nSector) || TestBitString(trapped, nSector))
            continue;
        for (int nWall = sector[nSector].wallptr; nWall < sector[nSector].wallptr+sector[nSector].wallnum; nWall++)
        {
            int nNext = wall[nWall].nextsector;
            if (nNext < 0 || !TestBitString(trapped, nNext))
                continue;
            const char *pzWhy;
            int nFlags = botPortalFlags(&body, nSector, nWall, nNext, &pzWhy);
            int x = (wall[nWall].x+wall[wall[nWall].point2].x)>>1;
            int y = (wall[nWall].y+wall[wall[nWall].point2].y)>>1;
            OSD_Printf("  trap entry %d -> %d (wall %d at %d,%d): flags %d %s, rise %d\n", nSector, nNext, nWall, x, y,
                nFlags, pzWhy, getflorzofslope(nSector, x, y)-getflorzofslope(nNext, x, y));
        }
    }
    for (int nSector = 0; nSector < numsectors; nSector++)
    {
        if (!TestBitString(reached, nSector))
            continue;
        for (int nWall = sector[nSector].wallptr; nWall < sector[nSector].wallptr+sector[nSector].wallnum; nWall++)
        {
            int nNext = wall[nWall].nextsector;
            if (nNext < 0 || TestBitString(reached, nNext))
                continue;
            const char *pzWhy;
            botPortalFlags(&body, nSector, nWall, nNext, &pzWhy);
            int x = (wall[nWall].x+wall[wall[nWall].point2].x)>>1;
            int y = (wall[nWall].y+wall[wall[nWall].point2].y)>>1;
            OSD_Printf("  sector %d -> %d (wall %d at %d,%d, next type %d): %s, rise %d\n", nSector, nNext, nWall, x, y,
                (int)sector[nNext].type, pzWhy, getflorzofslope(nSector, x, y)-getflorzofslope(nNext, x, y));
        }
    }
}

// bot_debug 3: draw the map from above as botnav.svg in the current
// directory. Grey: reachable from the start spots, orange: no way back
// from there, black: unreachable. Portals: green walkable, cyan bridge,
// red refused. Blocking sprites are drawn by their footprint, start spots blue.
void botNavDrawMap(PLAYER *pPlayer)
{
    BOTBODY body;
    botGetBody(pPlayer, &body);
    FILE *pFile = fopen("botnav.svg", "w");
    if (!pFile)
        return;
    int x0 = INT32_MAX, y0 = INT32_MAX, x1 = INT32_MIN, y1 = INT32_MIN;
    for (int i = 0; i < numwalls; i++)
    {
        x0 = ClipHigh(x0, wall[i].x); y0 = ClipHigh(y0, wall[i].y);
        x1 = ClipLow(x1, wall[i].x); y1 = ClipLow(y1, wall[i].y);
    }
    fprintf(pFile, "<svg xmlns='http://www.w3.org/2000/svg' viewBox='%d %d %d %d' width='2048' height='%d'>\n",
        x0-512, y0-512, x1-x0+1024, y1-y0+1024, (int)(2048.0*(y1-y0+1024)/(x1-x0+1024)));
    fprintf(pFile, "<rect x='%d' y='%d' width='%d' height='%d' fill='black'/>\n", x0-512, y0-512, x1-x0+1024, y1-y0+1024);
    static char reached[(kMaxSectors+7)>>3];
    memset(reached, 0, sizeof(reached));
    for (int i = 0; i < kMaxPlayers; i++)
    {
        int nStart = gStartZone[i].sectnum;
        if (nStart < 0 || nStart >= numsectors)
            continue;
        botSearch(NULL, &body, nStart, -2, NULL);
        for (int j = 0; j < kMaxSectors; j++)
            if (TestBitString(botVisited, j))
                SetBitString(reached, j);
    }
    for (int nSector = 0; nSector < numsectors; nSector++)
    {
        const char *pzFill = "#202020";
        if (TestBitString(reached, nSector))
        {
            botSearch(NULL, &body, nSector, -2, NULL);
            pzFill = TestBitString(botVisited, gStartZone[0].sectnum) ? "#606060" : "#b06000";
        }
        fprintf(pFile, "<path fill='%s' fill-rule='evenodd' d='", pzFill);
        int nStart = sector[nSector].wallptr, nEnd = nStart+sector[nSector].wallnum;
        int nLoop = nStart;
        for (int nWall = nStart; nWall < nEnd; nWall++)
        {
            fprintf(pFile, "%s%d %d ", nWall == nLoop ? "M" : "L", wall[nWall].x, wall[nWall].y);
            if (wall[nWall].point2 < nWall)
            {
                fprintf(pFile, "Z ");
                nLoop = nWall+1;
            }
        }
        fprintf(pFile, "'/>\n");
        int cx = 0, cy = 0;
        for (int nWall = nStart; nWall < nEnd; nWall++)
        {
            cx += wall[nWall].x;
            cy += wall[nWall].y;
        }
        fprintf(pFile, "<text x='%d' y='%d' font-size='160' fill='white'>%d</text>\n", cx/ClipLow(nEnd-nStart, 1), cy/ClipLow(nEnd-nStart, 1), nSector);
    }
    for (int nSector = 0; nSector < numsectors; nSector++)
    {
        for (int nWall = sector[nSector].wallptr; nWall < sector[nSector].wallptr+sector[nSector].wallnum; nWall++)
        {
            walltype *pWall = &wall[nWall], *pWall2 = &wall[pWall->point2];
            const char *pzColor = "white";
            if (pWall->nextsector >= 0)
            {
                int nFlags = botPortalFlags(&body, nSector, nWall, pWall->nextsector);
                pzColor = nFlags < 0 ? "red" : (nFlags & kBotPortalBridge) ? "cyan" : "#00c000";
            }
            // Offset each side of a portal a little so both directions show.
            int dx = pWall2->x-pWall->x, dy = pWall2->y-pWall->y, nLength = ClipLow(ksqrt(dx*dx+dy*dy), 1);
            int ox = -dy*24/nLength, oy = dx*24/nLength;
            fprintf(pFile, "<line x1='%d' y1='%d' x2='%d' y2='%d' stroke='%s' stroke-width='24'/>\n",
                pWall->x+ox, pWall->y+oy, pWall2->x+ox, pWall2->y+oy, pzColor);
        }
    }
    for (int nSprite = 0; nSprite < kMaxSprites; nSprite++)
    {
        spritetype *pSprite = &sprite[nSprite];
        if (pSprite->statnum >= kMaxStatus || !(pSprite->cstat & 1) || pSprite->statnum == kStatDude)
            continue;
        int nWidth = (tilesiz[pSprite->picnum].x*pSprite->xrepeat)>>2;
        int nDepth = (pSprite->cstat & 48) == 32 ? (tilesiz[pSprite->picnum].y*pSprite->yrepeat)>>2 : 32;
        fprintf(pFile, "<rect x='%d' y='%d' width='%d' height='%d' fill='%s' opacity='0.8' transform='rotate(%d %d %d)'/>\n",
            pSprite->x-nWidth/2, pSprite->y-nDepth/2, nWidth, nDepth, (pSprite->cstat & 48) == 32 ? "yellow" : "magenta",
            ((pSprite->cstat & 48) == 16 ? pSprite->ang*360/2048+90 : pSprite->ang*360/2048), (int)pSprite->x, (int)pSprite->y);
    }
    for (int i = 0; i < kMaxPlayers; i++)
        fprintf(pFile, "<circle cx='%d' cy='%d' r='192' fill='#4060ff'/>\n", gStartZone[i].x, gStartZone[i].y);
    fprintf(pFile, "</svg>\n");
    fclose(pFile);
    OSD_Printf("Wrote botnav.svg\n");
}
