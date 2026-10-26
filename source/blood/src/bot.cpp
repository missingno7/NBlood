//-------------------------------------------------------------------------
/*
Copyright (C) 1996, 2003 - 3D Realms Entertainment
Copyright (C) 2000, 2003 - Matt Saettler (EDuke Enhancements)
Copyright (C) 2004, 2007 - EDuke32 developers
Copyright (C) 2022, 2023 - Jordon Moss (Aka. StrikerTheHedgefox)
Copyright (C) 2026 - NBlood developers

BloodBath bots: player slots, lifecycle and per-frame input generation.
The structure of botThink() follows NetDuke32's DukeBot_GetInput()
(source/netduke32/src/dukebot.cpp at NetDuke32_v1.2.1,
f919616e0792a7af1f8006af5e606f0e26927673).

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
#include "compat.h"
#include "mmulti.h"
#include "osd.h"
#include "common_game.h"
#include "actor.h"
#include "blood.h"
#include "config.h"
#include "db.h"
#include "demo.h"
#include "endgame.h"
#include "gameutil.h"
#include "globals.h"
#include "levels.h"
#include "network.h"
#include "player.h"
#include "seq.h"
#include "trig.h"
#include "view.h"
#include "botai.h"

int32_t gBotCount = 0;
int32_t gBotSkill = kBotSkillDefault;
int32_t gBotAutopilot = 0;
int32_t gBotDebug = 0;
int32_t gBotTestTime = 0;
uint8_t gBotSlots = 0;

BOTINFO gBotInfo[kMaxPlayers];

const BOTSKILL gBotSkillInfo[kBotSkillMax+1] =
{
    // react    turn  err  fov autoaim strafe dodge memory      lead
    { 18, 30,    40,  90, 320,  2,      80,    30, kBotFPS*3,   64 }, // Still Kicking
    { 12, 22,    64,  60, 380,  2,     140,    80, kBotFPS*5,  128 }, // Pink On The Inside
    {  8, 15,    96,  36, 440,  1,     200,   130, kBotFPS*7,  180 }, // Lightly Broiled
    {  5, 10,   140,  20, 480,  1,     240,   190, kBotFPS*9,  230 }, // Well Done
    {  3,  6,   200,  10, 512,  1,     256,   240, kBotFPS*12, 256 }, // Extra Crispy
};

static const char *botNames[] =
{
    "Acolyte", "Fanatic", "Zealot", "Cheogh", "Shial", "Phantasm",
    "Gargoyle", "Butcher", "Hellhound", "Bloat", "Mime", "Bone Eel",
    "Gill Beast", "Cerberus", "Ophidian", "Nightwing",
};

struct BOTSTATS
{
    int nDeaths;
    int nSectors;
    int nEscapes;
    int nDistance;
    int lastX, lastY;
    char visited[(kMaxSectors+7)>>3];
};

static BOTSTATS botStats[kMaxPlayers];

uint32_t botRand(BOTINFO *pBot)
{
    // xorshift32: the bots' own stream, the game's RNG is never touched.
    uint32_t x = pBot->nRand;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return pBot->nRand = x;
}

int botRandRange(BOTINFO *pBot, int nRange)
{
    if (nRange <= 1)
        return 0;
    return (int)(botRand(pBot) % (uint32_t)nRange);
}

int botAngleDelta(int a, int b)
{
    return ((b-a+kAng180)&kAngMask)-kAng180;
}

void botGetExtents(PLAYER *pPlayer, int *pTop, int *pBottom)
{
    GetSpriteExtents(pPlayer->pSprite, pTop, pBottom);
}

int botPackStartInfo(void)
{
    int nCount = ClipRange(gBotCount, 0, kMaxPlayers-numplayers);
    if (gNetMode != NETWORK_NONE && gPacketMode == PACKETMODE_2)
        nCount = 0;
    if (!nCount)
        return 0;
    return nCount | (ClipRange(gBotSkill, kBotSkillMin, kBotSkillMax)<<4);
}

bool botStartInfoHasBots(int nInfo)
{
    return (nInfo & 15) != 0;
}

static void botRemoveFromConnectList(int nPlayer)
{
    if (connecthead == nPlayer)
    {
        connecthead = connectpoint2[nPlayer];
        return;
    }
    for (int p = connecthead; p >= 0; p = connectpoint2[p])
    {
        if (connectpoint2[p] == nPlayer)
        {
            connectpoint2[p] = connectpoint2[nPlayer];
            return;
        }
    }
}

static void botAppendToConnectList(int nPlayer)
{
    int nLast = connecthead;
    while (connectpoint2[nLast] >= 0)
        nLast = connectpoint2[nLast];
    connectpoint2[nLast] = nPlayer;
    connectpoint2[nPlayer] = -1;
}

static bool botInConnectList(int nPlayer)
{
    for (int p = connecthead; p >= 0; p = connectpoint2[p])
        if (p == nPlayer)
            return true;
    return false;
}

static int botConnectedCount(void)
{
    int nCount = 0;
    for (int p = connecthead; p >= 0; p = connectpoint2[p])
        nCount++;
    return nCount;
}

void botClearSlots(void)
{
    for (int p = 0; p < kMaxPlayers; p++)
    {
        if (botIsBot(p) && botInConnectList(p))
            botRemoveFromConnectList(p);
        gBotInfo[p].bActive = false;
    }
    gBotSlots = 0;
}

static void botInitSlot(int nPlayer, int nSkill, int nName)
{
    BOTINFO *pBot = &gBotInfo[nPlayer];
    memset(pBot, 0, sizeof(BOTINFO));
    pBot->bActive = true;
    pBot->nSkill = ClipRange(nSkill, kBotSkillMin, kBotSkillMax);
    pBot->nRand = 0x9e3779b9u ^ (gGameOptions.uMapCRC*2654435761u) ^ (uint32_t)(nPlayer*40503+1);
    if (!pBot->nRand)
        pBot->nRand = 1;
    pBot->nTarget = -1;
    pBot->nItem = -1;
    pBot->nLastItem = -1;
    pBot->nHurtBy = -1;
    pBot->nRoamSect = -1;
    botNavReset(pBot);
    if (nName >= 0)
    {
        PROFILE *pProfile = &gProfile[nPlayer];
        Bstrncpyz(pProfile->name, botNames[nName % ARRAY_SSIZE(botNames)], sizeof(pProfile->name));
        pProfile->nAutoAim = gBotSkillInfo[pBot->nSkill].nAutoAim;
        pProfile->nWeaponSwitch = 1;
        pProfile->skill = 2;
    }
    memset(&botStats[nPlayer], 0, sizeof(BOTSTATS));
}

void botSetupSlots(int nInfo)
{
    if (gDemo.at1)
    {
        // A demo carries its own connect list; its bots are plain inputs.
        gBotSlots = 0;
        return;
    }
    botClearSlots();
    int nCount = nInfo & 15;
    if (gGameOptions.nGameType < kGameTypeBloodBath)
        nCount = 0;
    nCount = ClipRange(nCount, 0, kMaxPlayers-botConnectedCount());
    for (int p = 0; p < kMaxPlayers && nCount > 0; p++)
    {
        if (botInConnectList(p))
            continue;
        botAppendToConnectList(p);
        gBotSlots |= 1<<p;
        nCount--;
    }
    gNetPlayers = gInitialNetPlayers = botConnectedCount();
}

void botLevelInit(void)
{
    int nInfo = gPacketStartGame.nBotInfo;
    int nSkill = (nInfo>>4)&7;
    int nName = (int)(gGameOptions.uMapCRC % ARRAY_SSIZE(botNames));
    for (int p = 0; p < kMaxPlayers; p++)
        gBotInfo[p].bActive = false;
    for (int p = connecthead; p >= 0; p = connectpoint2[p])
    {
        if (botIsBot(p))
            botInitSlot(p, nSkill, nName++);
    }
    if (gBotAutopilot && gGameOptions.nGameType >= kGameTypeBloodBath && !gDemo.at1)
        botInitSlot(myconnectindex, gBotSkill, -1);
    if (gBotSlots)
        OSD_Printf("Bots: %d in game, skill %d\n", gNetPlayers-numplayers, nSkill);
}

static void botTrackStats(int nPlayer)
{
    BOTSTATS *pStats = &botStats[nPlayer];
    spritetype *pSprite = gPlayer[nPlayer].pSprite;
    if (!TestBitString(pStats->visited, pSprite->sectnum))
    {
        SetBitString(pStats->visited, pSprite->sectnum);
        pStats->nSectors++;
    }
    int nStep = approxDist(pSprite->x-pStats->lastX, pSprite->y-pStats->lastY);
    if (nStep < 2048)
        pStats->nDistance += nStep;
    pStats->lastX = pSprite->x;
    pStats->lastY = pSprite->y;
}

static void botPrintStats(void)
{
    OSD_Printf("Bot stats after %d s:\n", gLevelTime/kBotFPS);
    for (int p = connecthead; p >= 0; p = connectpoint2[p])
    {
        if (!gBotInfo[p].bActive)
            continue;
        BOTSTATS *pStats = &botStats[p];
        PLAYER *pPlayer = &gPlayer[p];
        OSD_Printf("  %d %-10s frags %3d deaths %3d sectors %4d distance %7d escapes %3d weapon %d health %d\n",
            p, gProfile[p].name, pPlayer->fragCount, pStats->nDeaths, pStats->nSectors, pStats->nDistance,
            pStats->nEscapes, pPlayer->curWeapon, pPlayer->pXSprite->health>>4);
    }
}

// Look angle that produces a given aim slope (see ProcessInput: slope is
// -horiz<<7 and horiz is 100*tan(look)).
static fix16_t botLookForSlope(int nSlope)
{
    float fHoriz = -(float)nSlope/128.f;
    float fLook = atanf(fHoriz/100.f)*1024.f/fPI;
    return fix16_from_float(fclamp(fLook, -347.f, 289.f));
}

static void botThink(int nPlayer, GINPUT *pInput)
{
    BOTINFO *pBot = &gBotInfo[nPlayer];
    PLAYER *pPlayer = &gPlayer[nPlayer];
    memset(pInput, 0, sizeof(GINPUT));
    if (!pBot->bActive || !pPlayer->pSprite || !pPlayer->pXSprite)
        return;
    if (!gGameStarted || gPaused || gEndGameMgr.at0)
        return;
    const BOTSKILL *pSkill = &gBotSkillInfo[pBot->nSkill];
    spritetype *pSprite = pPlayer->pSprite;
    pBot->nFrame++;

    if (pPlayer->pXSprite->health <= 0)
    {
        // Respawn after a short, human-like pause.
        if (!pBot->nRespawnWait)
            botStats[nPlayer].nDeaths++;
        pBot->nRespawnWait++;
        if (pBot->nRespawnWait > kBotFPS*3 || (pBot->nRespawnWait > kBotFPS && botRandRange(pBot, 32) == 0))
            pInput->keyFlags.action = 1;
        return;
    }
    if (pBot->nRespawnWait)
    {
        pBot->nRespawnWait = 0;
        pBot->nTarget = -1;
        pBot->nItem = -1;
        pBot->nLastItem = -1;
        pBot->nRoamSect = -1;
        pBot->nFireFrames = 0;
        botNavReset(pBot);
        pBot->nLastHealth = pPlayer->pXSprite->health;
    }
    if (pSprite->sectnum < 0 || pSprite->sectnum >= numsectors)
        return;
    botTrackStats(nPlayer);

    botUpdateTargets(pBot, pPlayer);
    bool bNeedy = botNeedsItems(pPlayer);
    botUpdateItemGoal(pBot, pPlayer, bNeedy);

    BOTMOVE move;
    memset(&move, 0, sizeof(move));
    move.nFaceAng = -1;
    int nAimAng = -1, nAimSlope = 0;

    spritetype *pItem = pBot->nItem >= 0 ? &sprite[pBot->nItem] : NULL;
    int nItemDist = pItem ? approxDist(pItem->x-pSprite->x, pItem->y-pSprite->y) : INT32_MAX;
    bool bSeeItem = pItem && nItemDist < 8000
        && cansee(pSprite->x, pSprite->y, pPlayer->zView, pSprite->sectnum, pItem->x, pItem->y, pItem->z-(4<<8), pItem->sectnum);

    if (pBot->nTarget >= 0 && pBot->bTargetVisible)
    {
        botFight(pBot, pPlayer, pInput, &move, &nAimAng, &nAimSlope);
        spritetype *pTarget = gPlayer[pBot->nTarget].pSprite;
        int nTargetDist = approxDist(pTarget->x-pSprite->x, pTarget->y-pSprite->y);
        // Grab a pickup in reach rather than standing off (DukeBot's goForItem).
        if (pItem && bSeeItem && nItemDist < 4000 && nItemDist < (bNeedy ? nTargetDist*2 : nTargetDist/2))
        {
            BOTMOVE itemMove = move;
            botNavMove(pBot, pPlayer, pItem->x, pItem->y, pItem->z, pItem->sectnum, &itemMove);
            move.dx = itemMove.dx*4+move.dx/4;
            move.dy = itemMove.dy*4+move.dy/4;
            move.bJump |= itemMove.bJump;
            move.bCrouch = itemMove.bCrouch;
        }
        else if (pTarget->sectnum != pSprite->sectnum && botNavProbe(pPlayer, getangle(move.dx, move.dy), 384) < 160)
        {
            // Cannot get closer directly: route there instead.
            BOTMOVE navMove = move;
            botNavMove(pBot, pPlayer, pTarget->x, pTarget->y, pTarget->z, pTarget->sectnum, &navMove);
            move.dx = navMove.dx;
            move.dy = navMove.dy;
            move.bJump |= navMove.bJump;
            move.bCrouch = navMove.bCrouch;
            move.bAction = navMove.bAction;
        }
    }
    else
    {
        int gx, gy, gz, gs;
        bool bHunt = false;
        if (pItem && (bNeedy || pBot->nTarget < 0 || nItemDist < 3000))
        {
            gx = pItem->x;
            gy = pItem->y;
            gz = pItem->z;
            gs = pItem->sectnum;
        }
        else if (pBot->nTarget >= 0)
        {
            // Investigate where we last saw or heard them.
            vec3_t *pPos = &pBot->seenPos[pBot->nTarget];
            gx = pPos->x;
            gy = pPos->y;
            gz = pPos->z;
            gs = pBot->nSeenSect[pBot->nTarget];
            bHunt = true;
            if (approxDist(gx-pSprite->x, gy-pSprite->y) < 600)
            {
                pBot->nSeenFrame[pBot->nTarget] = 0;
                pBot->nTarget = -1;
            }
        }
        else
        {
            if (pBot->nRoamSect < 0 || pBot->nRoamUntil < pBot->nFrame || pSprite->sectnum == pBot->nRoamSect)
            {
                pBot->nRoamSect = botNavRoamSector(pBot, pPlayer);
                pBot->nRoamUntil = pBot->nFrame+kBotFPS*20;
            }
            gs = pBot->nRoamSect >= 0 ? pBot->nRoamSect : pSprite->sectnum;
            gx = wall[sector[gs].wallptr].x;
            gy = wall[sector[gs].wallptr].y;
            gz = sector[gs].floorz;
            // Head for the middle of the sector rather than a corner.
            int cx = 0, cy = 0, nWalls = sector[gs].wallnum;
            for (int i = 0; i < nWalls; i++)
            {
                cx += wall[sector[gs].wallptr+i].x;
                cy += wall[sector[gs].wallptr+i].y;
            }
            if (nWalls > 0 && inside(cx/nWalls, cy/nWalls, gs) == 1)
            {
                gx = cx/nWalls;
                gy = cy/nWalls;
            }
        }
        botNavMove(pBot, pPlayer, gx, gy, gz, gs, &move);
        if (bHunt && approxDist(gx-pSprite->x, gy-pSprite->y) < 5000 && move.nFaceAng < 0)
            move.nFaceAng = getangle(gx-pSprite->x, gy-pSprite->y);
        int nExpect = pBot->nTarget >= 0 ? approxDist(gx-pSprite->x, gy-pSprite->y) : 3000;
        botSelectWeapon(pBot, pPlayer, pInput, ClipRange(nExpect, 1000, 6000));
    }
    if (pItem)
    {
        if (!bSeeItem)
            pBot->nItemFail++;
        else if (nItemDist < 400)
            pBot->nItemFail += 8; // on top of it and still there: we cannot take it
    }
    if (pBot->nav.nEscape > 0 && pBot->nav.nEscape == 1)
        botStats[nPlayer].nEscapes++;

    botDodge(pBot, pPlayer, &move);
    botUseInventory(pBot, pPlayer, pInput);

    // Turn toward the enemy, or where we are going.
    int nWantAng = pSprite->ang;
    int nTurnSpeed = pSkill->nTurnSpeed;
    if (nAimAng >= 0)
        nWantAng = nAimAng;
    else if (move.nFaceAng >= 0)
        nWantAng = move.nFaceAng;
    else if (move.dx || move.dy)
    {
        nWantAng = getangle(move.dx, move.dy);
        nTurnSpeed = ClipLow(nTurnSpeed*2/3, 32);
    }
    int nTurn = ClipRange(botAngleDelta(pSprite->ang, nWantAng), -nTurnSpeed, nTurnSpeed);
    pInput->q16turn = fix16_from_int(nTurn);

    fix16_t q16Look = botLookForSlope(nAimAng >= 0 ? nAimSlope : 0);
    pInput->q16mlook = fix16_clamp((q16Look-pPlayer->q16look)>>3, -(F16(127)>>2), F16(127)>>2);

    if (move.nSpeed > 0 && (move.dx || move.dy))
    {
        // World direction to player-relative input (see ProcessInput).
        int nRel = (getangle(move.dx, move.dy)-pSprite->ang)&kAngMask;
        pInput->forward = mulscale30(move.nSpeed, Cos(nRel));
        pInput->strafe = -mulscale30(move.nSpeed, Sin(nRel));
        pInput->syncFlags.run = move.nSpeed > 1024;
    }
    if (pBot->nJumpWait > 0)
        pBot->nJumpWait--;
    else if (move.bJump)
    {
        pInput->buttonFlags.jump = 1;
        pBot->nJumpWait = pPlayer->posture == kPostureSwim ? 0 : 8;
    }
    if (pPlayer->posture == kPostureSwim && move.bJump)
        pInput->buttonFlags.jump = 1;
    pInput->buttonFlags.crouch = move.bCrouch && !pInput->buttonFlags.jump;
    if (move.bAction)
        pInput->keyFlags.action = 1;

    if (gBotDebug && (pBot->nFrame % (kBotFPS*5)) == 0)
    {
        OSD_Printf("bot %d %s: sect %d pos %d,%d hp %d wpn %d tgt %d%s item %d nav %d->%d stuck %d frags %d\n",
            nPlayer, gProfile[nPlayer].name, (int)pSprite->sectnum, (int)pSprite->x, (int)pSprite->y, pPlayer->pXSprite->health>>4,
            pPlayer->curWeapon, pBot->nTarget, pBot->bTargetVisible ? "(seen)" : "", pBot->nItem,
            (int)pSprite->sectnum, pBot->nav.nNextSect, pBot->nav.nStuck, pPlayer->fragCount);
    }
}

void botGenerateInputs(void)
{
    if (!gBotSlots || gDemo.at1 || myconnectindex != connecthead)
        return;
    for (int p = connecthead; p >= 0; p = connectpoint2[p])
    {
        if (!botIsBot(p))
            continue;
        while (gNetFifoHead[p] < gNetFifoHead[myconnectindex])
        {
            botThink(p, &gFifoInput[gNetFifoHead[p]&255][p]);
            gNetFifoHead[p]++;
        }
    }
    if (gBotTestTime > 0 && gGameStarted && gLevelTime >= gBotTestTime*kBotFPS)
    {
        botPrintStats();
        gBotTestTime = 0;
        gQuitGame = true;
    }
}

void botAutopilotInput(GINPUT *pInput)
{
    if (!gBotAutopilot || !gBotInfo[myconnectindex].bActive || gDemo.at1)
        return;
    KEYFLAGS keep = pInput->keyFlags;
    botThink(myconnectindex, pInput);
    pInput->keyFlags.quit |= keep.quit;
    pInput->keyFlags.pause |= keep.pause;
}

// Local games only: change the bot line-up while playing.
static bool botCanEditLocalGame(void)
{
    if (numplayers > 1 || gNetMode == NETWORK_CLIENT)
    {
        OSD_Printf("Bots can only be added or removed during a local game. Set bot_count before hosting.\n");
        return false;
    }
    if (!gGameStarted || gGameOptions.nGameType < kGameTypeBloodBath || gDemo.at0 || gDemo.at1)
    {
        OSD_Printf("Start a BloodBath or Teams game first (bot_count sets bots for the next game).\n");
        return false;
    }
    return true;
}

static int osdcmd_addbot(osdcmdptr_t parm)
{
    if (!botCanEditLocalGame())
        return OSDCMD_OK;
    int nSkill = parm->numparms > 0 ? ClipRange(atoi(parm->parms[0]), kBotSkillMin, kBotSkillMax) : gBotSkill;
    int nPlayer;
    for (nPlayer = 0; nPlayer < kMaxPlayers; nPlayer++)
        if (!botInConnectList(nPlayer))
            break;
    if (nPlayer >= kMaxPlayers)
    {
        OSD_Printf("No free player slot.\n");
        return OSDCMD_OK;
    }
    botAppendToConnectList(nPlayer);
    gBotSlots |= 1<<nPlayer;
    gNetPlayers = gInitialNetPlayers = botConnectedCount();
    for (int i = gNetFifoTail; i < gNetFifoHead[myconnectindex]; i++)
        memset(&gFifoInput[i&255][nPlayer], 0, sizeof(GINPUT));
    gNetFifoHead[nPlayer] = gNetFifoHead[myconnectindex];
    playerInit(nPlayer, 0);
    playerStart(nPlayer, 1);
    int nName = (int)((gGameOptions.uMapCRC+nPlayer*5) % ARRAY_SSIZE(botNames));
    botInitSlot(nPlayer, nSkill, nName);
    gBotCount = gNetPlayers-numplayers;
    gPacketStartGame.nBotInfo = botPackStartInfo();
    OSD_Printf("%s joined the game.\n", gProfile[nPlayer].name);
    return OSDCMD_OK;
}

static int osdcmd_removebot(osdcmdptr_t UNUSED(parm))
{
    UNREFERENCED_CONST_PARAMETER(parm);
    if (!botCanEditLocalGame())
        return OSDCMD_OK;
    int nPlayer = -1;
    for (int p = connecthead; p >= 0; p = connectpoint2[p])
        if (botIsBot(p))
            nPlayer = p;
    if (nPlayer < 0)
    {
        OSD_Printf("No bots in the game.\n");
        return OSDCMD_OK;
    }
    PLAYER *pPlayer = &gPlayer[nPlayer];
    char buffer[128];
    sprintf(buffer, "%s left the game with %d frags.", gProfile[nPlayer].name, pPlayer->fragCount);
    viewSetMessage(buffer);
    seqKill(3, pPlayer->pSprite->extra);
    actPostSprite(pPlayer->nSprite, kStatFree);
    if (gViewIndex == nPlayer)
    {
        gViewIndex = myconnectindex;
        gView = &gPlayer[gViewIndex];
    }
    botRemoveFromConnectList(nPlayer);
    gBotSlots &= ~(1<<nPlayer);
    gBotInfo[nPlayer].bActive = false;
    gNetPlayers = gInitialNetPlayers = botConnectedCount();
    gBotCount = gNetPlayers-numplayers;
    gPacketStartGame.nBotInfo = botPackStartInfo();
    return OSDCMD_OK;
}

static int osdcmd_botstats(osdcmdptr_t UNUSED(parm))
{
    UNREFERENCED_CONST_PARAMETER(parm);
    botPrintStats();
    return OSDCMD_OK;
}

void botInitCommands(void)
{
    static osdcvardata_t cvars_bot[] =
    {
        { "bot_count", "number of bots in the next BloodBath/Teams game (0-7)", (void *)&gBotCount, CVAR_INT|CVAR_NOSAVE, 0, kMaxPlayers-1 },
        { "bot_skill", "bot skill in the next game (0-4)", (void *)&gBotSkill, CVAR_INT|CVAR_NOSAVE, kBotSkillMin, kBotSkillMax },
        { "bot_autopilot", "debug: let bot AI control the local player too", (void *)&gBotAutopilot, CVAR_BOOL|CVAR_NOSAVE, 0, 1 },
        { "bot_debug", "print bot state to the console every few seconds", (void *)&gBotDebug, CVAR_BOOL|CVAR_NOSAVE, 0, 1 },
        { "bot_testtime", "debug: print bot stats and quit after this many seconds of play", (void *)&gBotTestTime, CVAR_INT|CVAR_NOSAVE, 0, 36000 },
    };
    for (auto &cv : cvars_bot)
        OSD_RegisterCvar(&cv, osdcmd_cvar_set);
    OSD_RegisterFunction("addbot", "addbot [skill]: adds a bot to the current local BloodBath game", osdcmd_addbot);
    OSD_RegisterFunction("removebot", "removebot: removes a bot from the current local game", osdcmd_removebot);
    OSD_RegisterFunction("botstats", "botstats: prints bot statistics", osdcmd_botstats);
}
