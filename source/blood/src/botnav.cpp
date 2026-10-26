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

// Heights are in z units. A standing jump lifts Caleb's feet a little over
// 72 pixels (jump velocity 0xbaaaa against gravity 58254 per frame).
#define kBotJumpRise    (68<<8)
#define kBotMaxDrop     (320<<8)
#define kBotRouteAge    kBotFPS
#define kBotCheckFrames 12
#define kBotAvoidFrames (kBotFPS*12)

static short botQueue[kMaxSectors];
static short botParent[kMaxSectors];
static short botVia[kMaxSectors];
static uint8_t botFlags[kMaxSectors];
static char botVisited[(kMaxSectors+7)>>3];

struct BOTBODY
{
    int nRadius, nCeilDist, nFloorDist;
    int nHeight, nCrouchHeight;
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

static void botAvoid(BOTINFO *pBot, int nWall)
{
    BOTNAV *pNav = &pBot->nav;
    if (nWall < 0)
        return;
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

// A sector the player can set in motion: doors, lifts and the like.
static bool botSectorOperable(int nSector)
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
        return pXSector->Push || pXSector->Wallpush || pXSector->Enter;
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

// Can a player walk through this portal, and what does it take? Returns -1
// when the portal is not usable right now or ever as far as we can tell.
static int botPortalFlags(const BOTBODY *pBody, int nFrom, int nWall, int nTo)
{
    walltype *pWall = &wall[nWall];
    if (pWall->cstat & 1)
        return -1;
    int x = (pWall->x+wall[pWall->point2].x)>>1;
    int y = (pWall->y+wall[pWall->point2].y)>>1;
    int nFromCeil, nFromFloor, nToCeil, nToFloor;
    getzsofslope(nFrom, x, y, &nFromCeil, &nFromFloor);
    getzsofslope(nTo, x, y, &nToCeil, &nToFloor);

    int nFlags = 0;
    bool bWater = botIsUnderwater(nFrom) || botIsUnderwater(nTo);
    int nWindow = ClipHigh(nFromFloor, nToFloor)-ClipLow(nFromCeil, nToCeil);
    if (nWindow < pBody->nCrouchHeight)
    {
        if (!botSectorOperable(nTo) && !botSectorOperable(nFrom))
            return -1;
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
    if (nRise > kBotJumpRise && !bWater)
    {
        if (!botSectorMovesFloor(nTo) && !botSectorMovesFloor(nFrom))
            return -1;
        nFlags |= kBotPortalLift;
    }
    else if (nRise > pBody->nFloorDist)
        nFlags |= kBotPortalJump;
    if (-nRise > kBotMaxDrop && !bWater)
        return -1;
    return nFlags;
}

static int botLinkDestination(int nSector, int *pFlags)
{
    *pFlags = kBotPortalLink;
    int nLink = gUpperLink[nSector];
    if (nLink >= 0 && sprite[nLink].owner >= 0 && sprite[nLink].owner < kMaxSprites)
        return sprite[sprite[nLink].owner].sectnum;
    nLink = gLowerLink[nSector];
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

// Breadth-first search from nFrom to nTo. On success returns the index of the
// queue entry holding the first step and the number of steps.
static int botSearch(BOTINFO *pBot, const BOTBODY *pBody, int nFrom, int nTo, int *pLength)
{
    if ((unsigned)nFrom >= (unsigned)numsectors || (unsigned)nTo >= (unsigned)numsectors)
        return -1;
    BOTNAV *pNav = pBot ? &pBot->nav : NULL;
    memset(botVisited, 0, sizeof(botVisited));
    botQueue[0] = nFrom;
    botParent[0] = -1;
    botVia[0] = -1;
    botFlags[0] = 0;
    SetBitString(botVisited, nFrom);
    int nFound = -1;
    int nTail = 1;
    for (int nHead = 0; nHead < nTail && nFound < 0; nHead++)
    {
        int nSector = botQueue[nHead];
        int nStart = sector[nSector].wallptr, nEnd = nStart+sector[nSector].wallnum;
        for (int nWall = nStart; nWall < nEnd; nWall++)
        {
            int nNext = wall[nWall].nextsector;
            if (nNext < 0 || TestBitString(botVisited, nNext))
                continue;
            if (pNav && botAvoided(pNav, nWall, pBot->nFrame))
                continue;
            int nFlags = botPortalFlags(pBody, nSector, nWall, nNext);
            if (nFlags < 0)
                continue;
            SetBitString(botVisited, nNext);
            botQueue[nTail] = nNext;
            botParent[nTail] = nHead;
            botVia[nTail] = nWall;
            botFlags[nTail] = nFlags;
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
        int nLinkFlags;
        int nDests[2] = { botLinkDestination(nSector, &nLinkFlags), botTeleportDestination(nSector) };
        for (int i = 0; i < 2 && nFound < 0; i++)
        {
            int nNext = nDests[i];
            if (nNext < 0 || nNext >= numsectors || TestBitString(botVisited, nNext))
                continue;
            SetBitString(botVisited, nNext);
            botQueue[nTail] = nNext;
            botParent[nTail] = nHead;
            botVia[nTail] = -1;
            botFlags[nTail] = kBotPortalLink;
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
    return botSearch(pBot, &body, pPlayer->pSprite->sectnum, nSector, NULL) >= 0;
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
    return botClearDist(pBody, x, y, z, nSector, dx, dy) >= nDist-24;
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
    int nBestAng = nDirect, nBestScore = INT32_MIN;
    for (int i = 0; i <= 12; i++)
    {
        int nDelta = ((i+1)>>1)*128*((i&1) ? 1 : -1);
        int nAng = (nDirect+nDelta)&kAngMask;
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

static void botFindRoute(BOTINFO *pBot, const BOTBODY *pBody, int nFrom, int nTo)
{
    BOTNAV *pNav = &pBot->nav;
    pNav->nRouteFrom = nFrom;
    pNav->nRouteTo = nTo;
    pNav->nRouteFrame = pBot->nFrame;
    pNav->nNextSect = -1;
    pNav->nPortal = -1;
    int nLength = 0;
    int nStep = botSearch(pBot, pBody, nFrom, nTo, &nLength);
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
        botAvoid(pBot, pNav->nPortal);
        pNav->nStuck = 0;
        break;
    }
}

void botNavMove(BOTINFO *pBot, PLAYER *pPlayer, int x, int y, int z, int nSector, BOTMOVE *pMove)
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
        return;
    }

    int tx = x, ty = y;     // where we head
    int ax = x, ay = y;     // last point inside the current sector
    int nFlags = 0;
    bool bLink = false;
    if (nSector >= 0 && nSector != nCur)
    {
        if (pNav->nRouteFrom != nCur || pNav->nRouteTo != nSector || pBot->nFrame-pNav->nRouteFrame > kBotRouteAge)
            botFindRoute(pBot, &body, nCur, nSector);
        if (pNav->nNextSect >= 0)
        {
            nFlags = pNav->nPortalFlags;
            if (pNav->nPortal >= 0)
            {
                vec2_t cross, approach;
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
                    botAvoid(pBot, pNav->nPortal);
                    pNav->nDoorWait = 0;
                }
                return;
            }
            if (bOpen)
                pNav->nDoorWait = 0;
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
