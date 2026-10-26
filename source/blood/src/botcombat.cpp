//-------------------------------------------------------------------------
/*
Copyright (C) 1996, 2003 - 3D Realms Entertainment
Copyright (C) 2000, 2003 - Matt Saettler (EDuke Enhancements)
Copyright (C) 2004, 2007 - EDuke32 developers
Copyright (C) 2022, 2023 - Jordon Moss (Aka. StrikerTheHedgefox)
Copyright (C) 2026 - NBlood developers

Bot combat: seeing enemies, choosing weapons, aiming, firing, dodging and
valuing pickups. Target acquisition, fight distance, strafing and projectile
dodging follow NetDuke32's DukeBot_GetInput() (NetDuke32_v1.2.1); the weapon
and item knowledge is Blood's own.

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
#include "mmulti.h"
#include "common_game.h"
#include "actor.h"
#include "blood.h"
#include "db.h"
#include "gameutil.h"
#include "globals.h"
#include "levels.h"
#include "player.h"
#include "trig.h"
#include "weapon.h"
#include "botai.h"

struct BOTWEAPON
{
    int nAmmo;          // ammo type, -1 = none needed
    int nMinAmmo;
    int nValue;         // how much we like it in its element
    int nMinRange;      // closer than this hurts us (splash)
    int nIdealMin, nIdealMax;
    int nMaxRange;
    int nSpeed;         // projectile distance per frame, 0 = hitscan/melee
    bool bSplash;
    bool bHold;         // hold the trigger and release to throw
};

// Blood's weapons, by kWeapon*. Ranges are in map units (a player is 256 wide).
static const BOTWEAPON botWeapons[kWeaponMax] =
{
    { -1,  0,   0,    0,    0,     0,     0,   0, false, false }, // none
    { -1,  0,  10,    0,    0,   700,   900,   0, false, false }, // pitchfork
    {  1,  1,  55,    0,  600,  6000, 14000, 768, false, false }, // flare gun
    {  2,  1,  85,    0,    0,  2600,  7000,   0, false, false }, // sawed-off
    {  3,  1,  70,    0,    0,  7000, 16000,   0, false, false }, // tommy gun
    {  4,  1,  90, 2000, 3200, 10000, 18000, 597,  true, false }, // napalm launcher
    {  5,  1,  60, 2200, 3000,  7000,  9000,   0,  true,  true }, // dynamite
    {  6, 20,  75,    0,    0,  1300,  1800, 273, false, false }, // aerosol can
    {  7,  1,  85,    0,    0,  9000, 16000, 683, false, false }, // tesla cannon
    {  8,  1,  75,    0,    0,  9000, 16000, 683, false, false }, // life leech
    {  9,  1,  20,    0,    0, 16000, 30000,   0, false, false }, // voodoo doll
    { 10,  1,   0,    0,    0,     0,     0,   0,  true,  true }, // proximity TNT
    { 11,  1,   0,    0,    0,     0,     0,   0,  true,  true }, // remote TNT
    { -1,  0,  60,    0,    0,   900,  1100,   0, false, false }, // beast claws
};

static bool botHasWeapon(PLAYER *pPlayer, int nWeapon)
{
    if (nWeapon <= kWeaponNone || nWeapon >= kWeaponMax || !pPlayer->hasWeapon[nWeapon])
        return false;
    const BOTWEAPON *pWeapon = &botWeapons[nWeapon];
    if (nWeapon == kWeaponLifeLeech)
        return pPlayer->ammoCount[pWeapon->nAmmo] >= pWeapon->nMinAmmo || pPlayer->pXSprite->health > (50<<4);
    return pWeapon->nAmmo < 0 || pPlayer->ammoCount[pWeapon->nAmmo] >= pWeapon->nMinAmmo;
}

static int botWeaponScore(PLAYER *pPlayer, int nWeapon, int nDist)
{
    if (!botHasWeapon(pPlayer, nWeapon))
        return 0;
    const BOTWEAPON *pWeapon = &botWeapons[nWeapon];
    if (pWeapon->nValue <= 0)
        return 0;
    if (pPlayer->isUnderwater && (nWeapon == kWeaponSprayCan || nWeapon == kWeaponTNT))
        return 0;
    if (pPlayer->lifeMode == kModeBeast)
        return nWeapon == kWeaponBeast ? 100 : 0;
    int nFactor;
    if (nDist < pWeapon->nMinRange)
        nFactor = 8;
    else if (nDist < pWeapon->nIdealMin)
        nFactor = 96+160*(nDist-pWeapon->nMinRange)/ClipLow(pWeapon->nIdealMin-pWeapon->nMinRange, 1);
    else if (nDist <= pWeapon->nIdealMax)
        nFactor = 256;
    else if (nDist <= pWeapon->nMaxRange)
        nFactor = 256-200*(nDist-pWeapon->nIdealMax)/ClipLow(pWeapon->nMaxRange-pWeapon->nIdealMax, 1);
    else
        nFactor = 24;
    int nScore = pWeapon->nValue*nFactor/256;
    if (pWeapon->nAmmo >= 0 && pPlayer->ammoCount[pWeapon->nAmmo] < pWeapon->nMinAmmo*4)
        nScore = nScore*2/3;
    if (powerupCheck(pPlayer, kPwUpTwoGuns) && (nWeapon == kWeaponFlare || nWeapon == kWeaponTommy || nWeapon == kWeaponNapalm))
        nScore += 15;
    return ClipLow(nScore, 1);
}

void botSelectWeapon(BOTINFO *pBot, PLAYER *pPlayer, GINPUT *pInput, int nDist)
{
    if (pBot->nWeaponWait > 0)
    {
        pBot->nWeaponWait--;
        return;
    }
    if (pPlayer->input.newWeapon || pPlayer->nextWeapon)
        return;
    // Never put away lit dynamite or a burning can.
    if ((pPlayer->curWeapon == kWeaponTNT && pPlayer->weaponState >= 4 && pPlayer->weaponState <= 6)
        || (pPlayer->curWeapon == kWeaponSprayCan && pPlayer->weaponState >= 5 && pPlayer->weaponState <= 7))
        return;
    int nBest = kWeaponNone, nBestScore = 0;
    for (int nWeapon = kWeaponPitchfork; nWeapon < kWeaponMax; nWeapon++)
    {
        int nScore = botWeaponScore(pPlayer, nWeapon, nDist);
        if (nScore > nBestScore)
        {
            nBest = nWeapon;
            nBestScore = nScore;
        }
    }
    if (nBest == kWeaponNone || nBest == pPlayer->curWeapon)
        return;
    int nCurrent = botWeaponScore(pPlayer, pPlayer->curWeapon, nDist);
    // Switching takes time; do it only for a real improvement.
    if (nCurrent > 0 && nBestScore < nCurrent+12)
        return;
    pInput->newWeapon = nBest;
    pBot->nWeaponWait = kBotFPS/2+botRandRange(pBot, kBotFPS/2);
}

bool botCanSeePlayer(PLAYER *pPlayer, PLAYER *pOther)
{
    spritetype *pSprite = pPlayer->pSprite;
    spritetype *pTarget = pOther->pSprite;
    int top, bottom;
    GetSpriteExtents(pTarget, &top, &bottom);
    int nHeight = bottom-top;
    int z = pPlayer->zView;
    for (int i = 1; i <= 3; i++)
    {
        if (cansee(pSprite->x, pSprite->y, z, pSprite->sectnum, pTarget->x, pTarget->y, top+nHeight*i/4, pTarget->sectnum))
            return true;
    }
    return false;
}

static bool botIsEnemy(PLAYER *pPlayer, PLAYER *pOther)
{
    if (pOther == pPlayer || !pOther->pSprite || !pOther->pXSprite || pOther->pXSprite->health <= 0)
        return false;
    if (gGameOptions.nGameType == kGameTypeTeams && pOther->teamId == pPlayer->teamId)
        return false;
    if (gGameOptions.nGameType == kGameTypeCoop)
        return false;
    return true;
}

void botUpdateTargets(BOTINFO *pBot, PLAYER *pPlayer)
{
    const BOTSKILL *pSkill = &gBotSkillInfo[pBot->nSkill];
    spritetype *pSprite = pPlayer->pSprite;
    int nHealth = pPlayer->pXSprite->health;
    bool bHurt = nHealth < pBot->nLastHealth;
    pBot->nLastHealth = nHealth;

    int nBest = -1, nBestScore = INT32_MAX;
    bool bCurrentVisible = false;
    for (int p = connecthead; p >= 0; p = connectpoint2[p])
    {
        PLAYER *pOther = &gPlayer[p];
        if (!botIsEnemy(pPlayer, pOther))
        {
            if (p != pPlayer->nPlayer)
                pBot->nSeenFrame[p] = 0;
            continue;
        }
        spritetype *pTarget = pOther->pSprite;
        int nDist = approxDist(pTarget->x-pSprite->x, pTarget->y-pSprite->y);
        int nDelta = klabs(botAngleDelta(pSprite->ang, getangle(pTarget->x-pSprite->x, pTarget->y-pSprite->y)));
        // Wide awareness close by, the view cone further away. Being shot
        // makes anyone look around.
        bool bInView = nDist < 1200 || nDelta <= pSkill->nFov || (bHurt && nDist < 12000);
        if (bInView && powerupCheck(pOther, kPwUpShadowCloak) && nDist > 2000)
            bInView = (botRand(pBot) & 7) == 0;
        bool bVisible = bInView && botCanSeePlayer(pPlayer, pOther);
        if (bVisible)
        {
            pBot->nSeenFrame[p] = pBot->nFrame;
            pBot->seenPos[p] = pTarget->xyz;
            pBot->nSeenSect[p] = pTarget->sectnum;
        }
        else if (nDist < 8000 && (pOther->flashEffect || pOther->visibility > 0) && pBot->nFrame-pBot->nSeenFrame[p] > 8)
        {
            // Heard a shot: roughly where it came from.
            pBot->nSeenFrame[p] = pBot->nFrame-pSkill->nMemory/2;
            pBot->seenPos[p].x = pTarget->x+botRandRange(pBot, 1025)-512;
            pBot->seenPos[p].y = pTarget->y+botRandRange(pBot, 1025)-512;
            pBot->seenPos[p].z = pTarget->z;
            pBot->nSeenSect[p] = pTarget->sectnum;
        }
        if (p == pBot->nTarget)
            bCurrentVisible = bVisible;
        if (!bVisible)
            continue;
        int nScore = nDist;
        if (p == pBot->nTarget)
            nScore = nScore/2;  // stick with who we are fighting
        if (p == pBot->nHurtBy && pBot->nFrame-pBot->nHurtFrame < kBotFPS*3)
            nScore = nScore/2;
        if (nScore < nBestScore)
        {
            nBestScore = nScore;
            nBest = p;
        }
    }
    if (bHurt && pPlayer->fraggerId < 0)
    {
        // Whoever is in sight most likely did it.
        if (nBest >= 0)
        {
            pBot->nHurtBy = nBest;
            pBot->nHurtFrame = pBot->nFrame;
        }
    }

    bool bWasVisible = pBot->bTargetVisible;
    int nOldTarget = pBot->nTarget;
    if (nBest >= 0)
    {
        pBot->nTarget = nBest;
        pBot->bTargetVisible = true;
    }
    else
    {
        pBot->bTargetVisible = false;
        // Keep hunting the last one we saw for a while.
        if (pBot->nTarget >= 0 && (!botIsEnemy(pPlayer, &gPlayer[pBot->nTarget]) || pBot->nFrame-pBot->nSeenFrame[pBot->nTarget] > pSkill->nMemory))
            pBot->nTarget = -1;
        if (pBot->nTarget < 0)
        {
            int nLatest = 0;
            for (int p = connecthead; p >= 0; p = connectpoint2[p])
            {
                if (p == pPlayer->nPlayer || !pBot->nSeenFrame[p] || !botIsEnemy(pPlayer, &gPlayer[p]))
                    continue;
                if (pBot->nFrame-pBot->nSeenFrame[p] <= pSkill->nMemory && pBot->nSeenFrame[p] > nLatest)
                {
                    nLatest = pBot->nSeenFrame[p];
                    pBot->nTarget = p;
                }
            }
        }
    }
    UNREFERENCED_PARAMETER(bCurrentVisible);
    if (pBot->bTargetVisible && (!bWasVisible || nOldTarget != pBot->nTarget))
        pBot->nReaction = pSkill->nReactMin+botRandRange(pBot, pSkill->nReactMax-pSkill->nReactMin+1);
}

// How much a pickup is worth to us right now. Mirrors the pickup rules in
// player.cpp closely enough that bots do not chase things they cannot take.
static int botItemScore(PLAYER *pPlayer, spritetype *pItem)
{
    int nType = pItem->type;
    int nHealth = pPlayer->pXSprite->health>>4;
    if (nType >= kItemWeaponBase && nType < kItemWeaponMax)
    {
        WEAPONITEMDATA *pData = &gWeaponItemData[nType-kItemWeaponBase];
        int nWeapon = pData->type;
        if (nWeapon <= kWeaponNone || nWeapon >= kWeaponMax)
            return 0;
        int nValue = botWeapons[nWeapon].nValue;
        if (!pPlayer->hasWeapon[nWeapon])
            return 40+nValue;
        if (gGameOptions.nWeaponSettings != 1 && pData->ammoType >= 0 && pPlayer->ammoCount[pData->ammoType] < gAmmoInfo[pData->ammoType].max)
            return nValue/3;
        return 0;
    }
    if (nType >= kItemAmmoBase && nType < kItemAmmoMax)
    {
        AMMOITEMDATA *pData = &gAmmoItemData[nType-kItemAmmoBase];
        int nAmmo = pData->type;
        if (nAmmo < 0 || nAmmo >= 12 || pPlayer->ammoCount[nAmmo] >= gAmmoInfo[nAmmo].max)
            return 0;
        int nWeapon = pData->weaponType;
        bool bUsable = nWeapon > kWeaponNone && nWeapon < kWeaponMax && pPlayer->hasWeapon[nWeapon];
        if (nAmmo == 5) // dynamite works without the weapon
            bUsable = true;
        if (!bUsable)
            return 3;
        int nValue = nWeapon < kWeaponMax ? botWeapons[nWeapon].nValue : 20;
        if (pPlayer->ammoCount[nAmmo] < gAmmoInfo[nAmmo].max/4)
            return nValue/2+10;
        return nValue/4;
    }
    if (nType >= kItemBase && nType < kItemMax)
    {
        int nItem = nType-kItemBase;
        switch (nType)
        {
        case kItemHealthMedPouch:
        case kItemHealthLifeEssense:
        case kItemHealthLifeSeed:
        case kItemHealthRedPotion:
        {
            int nMax = gPowerUpInfo[nItem].maxTime;
            if (nHealth >= nMax)
                return 0;
            int nGain = ClipHigh(gPowerUpInfo[nItem].bonusTime, nMax-nHealth);
            int nScore = 10+nGain;
            if (nHealth < 50)
                nScore *= 2;
            return nScore;
        }
        case kItemHealthDoctorBag:
            return pPlayer->packSlots[kPackMedKit].curAmount < 100 ? 25+(nHealth < 50 ? 40 : 0) : 0;
        case kItemArmorBasic:
        case kItemArmorBody:
        case kItemArmorFire:
        case kItemArmorSpirit:
        case kItemArmorSuper:
        {
            int nArmor = pPlayer->armor[0]+pPlayer->armor[1]+pPlayer->armor[2];
            if (nArmor >= 3*(200<<4))
                return 0;
            return nType == kItemArmorSuper ? 70 : 20+(3*(100<<4)-ClipHigh(nArmor, 3*(100<<4)))/(3<<4)/3;
        }
        case kItemDeathMask:
            return 110;
        case kItemTwoGuns:
            return 90;
        case kItemReflectShots:
            return 60;
        case kItemShadowCloak:
            return 45;
        case kItemJumpBoots:
            return pPlayer->packSlots[kPackJumpBoots].curAmount < 100 ? 12 : 0;
        case kItemCrystalBall:
        case kItemBeastVision:
            return 6;
        case kItemFeatherFall:
        case kItemGasMask:
            return 4;
        case kItemDivingSuit:
            return pPlayer->packSlots[kPackDivingSuit].curAmount < 100 ? 4 : 0;
        }
        return 0;
    }
    return 0;
}

bool botNeedsItems(PLAYER *pPlayer)
{
    if ((pPlayer->pXSprite->health>>4) <= 45)
        return true;
    int nWeapons = 0;
    for (int nWeapon = kWeaponFlare; nWeapon < kWeaponMax; nWeapon++)
        if (botWeapons[nWeapon].nValue > 30 && botHasWeapon(pPlayer, nWeapon))
            nWeapons++;
    return nWeapons == 0;
}

static bool botItemValid(int nSprite)
{
    if ((unsigned)nSprite >= kMaxSprites)
        return false;
    spritetype *pItem = &sprite[nSprite];
    return pItem->statnum == kStatItem && !(pItem->flags & 32) && !(pItem->cstat & 32768);
}

static bool botCanSeeItem(PLAYER *pPlayer, spritetype *pItem)
{
    spritetype *pSprite = pPlayer->pSprite;
    return cansee(pSprite->x, pSprite->y, pPlayer->zView, pSprite->sectnum, pItem->x, pItem->y, pItem->z-(4<<8), pItem->sectnum) != 0;
}

void botGiveUpItem(BOTINFO *pBot)
{
    if (pBot->nItem < 0)
        return;
    int i = pBot->nBadHead++ % ARRAY_SSIZE(pBot->nBadItem);
    pBot->nBadItem[i] = pBot->nItem;
    pBot->nBadUntil[i] = pBot->nFrame+kBotFPS*20;
    pBot->nItem = -1;
    pBot->nItemScore = 0;
    pBot->nItemFail = 0;
    pBot->nItemCooldown = kBotFPS/2;
}

void botUpdateItemGoal(BOTINFO *pBot, PLAYER *pPlayer, bool bNeedy)
{
    spritetype *pSprite = pPlayer->pSprite;
    if (pBot->nItemCooldown > 0)
        pBot->nItemCooldown--;
    if (pBot->nItem >= 0)
    {
        if (!botItemValid(pBot->nItem) || botItemScore(pPlayer, &sprite[pBot->nItem]) <= 0)
        {
            pBot->nItem = -1;
            pBot->nItemScore = 0;
            pBot->nItemFail = 0;
        }
        else if (pBot->nItemFail > kBotFPS*6)
            botGiveUpItem(pBot);
    }
    if (pBot->nItemCooldown > 0)
        return;
    pBot->nItemCooldown = kBotFPS;

    // Value over distance; things in sight count double, nearby things more.
    int nCand[4] = { -1, -1, -1, -1 };
    int nCandValue[4] = { 0, 0, 0, 0 };
    for (int nSprite = headspritestat[kStatItem]; nSprite >= 0; nSprite = nextspritestat[nSprite])
    {
        if (!botItemValid(nSprite))
            continue;
        bool bBad = false;
        for (int i = 0; i < ARRAY_SSIZE(pBot->nBadItem); i++)
            bBad |= pBot->nBadItem[i] == nSprite && pBot->nBadUntil[i] > pBot->nFrame;
        if (bBad)
            continue;
        spritetype *pItem = &sprite[nSprite];
        int nScore = botItemScore(pPlayer, pItem);
        if (nScore <= 0)
            continue;
        int nDist = approxDist(pItem->x-pSprite->x, pItem->y-pSprite->y)+(klabs(pItem->z-pSprite->z)>>4);
        int nValue = nScore*65536/(nDist+(bNeedy ? 3000 : 1500));
        if (nDist < 6000 && botCanSeeItem(pPlayer, pItem))
            nValue *= 2;
        for (int i = 0; i < 4; i++)
        {
            if (nValue > nCandValue[i])
            {
                for (int j = 3; j > i; j--)
                {
                    nCand[j] = nCand[j-1];
                    nCandValue[j] = nCandValue[j-1];
                }
                nCand[i] = nSprite;
                nCandValue[i] = nValue;
                break;
            }
        }
    }
    int nCurrentValue = 0;
    if (pBot->nItem >= 0)
    {
        spritetype *pItem = &sprite[pBot->nItem];
        int nDist = approxDist(pItem->x-pSprite->x, pItem->y-pSprite->y);
        nCurrentValue = botItemScore(pPlayer, pItem)*65536/(nDist+(bNeedy ? 3000 : 1500))*2;
    }
    for (int i = 0; i < 4; i++)
    {
        if (nCand[i] < 0 || nCandValue[i] <= nCurrentValue*5/4)
            break;
        if (nCand[i] == pBot->nItem)
            break;
        if (!botNavReachable(pPlayer, sprite[nCand[i]].sectnum))
            continue;
        pBot->nItem = nCand[i];
        pBot->nItemScore = nCandValue[i];
        pBot->nItemFail = 0;
        break;
    }
}

// Where an enemy will be when our shot arrives.
static void botLeadTarget(BOTINFO *pBot, PLAYER *pPlayer, spritetype *pTarget, int nSpeed, int *px, int *py)
{
    const BOTSKILL *pSkill = &gBotSkillInfo[pBot->nSkill];
    spritetype *pSprite = pPlayer->pSprite;
    *px = pTarget->x;
    *py = pTarget->y;
    if (nSpeed <= 0)
        return;
    int nDist = approxDist(pTarget->x-pSprite->x, pTarget->y-pSprite->y);
    int nFrames = ClipHigh(nDist/nSpeed, 45);
    int nTarget = pTarget->index;
    *px += mulscale8((xvel[nTarget]>>12)*nFrames, pSkill->nLead);
    *py += mulscale8((yvel[nTarget]>>12)*nFrames, pSkill->nLead);
}

void botFight(BOTINFO *pBot, PLAYER *pPlayer, GINPUT *pInput, BOTMOVE *pMove, int *pAimAng, int *pAimSlope)
{
    const BOTSKILL *pSkill = &gBotSkillInfo[pBot->nSkill];
    spritetype *pSprite = pPlayer->pSprite;
    PLAYER *pOther = &gPlayer[pBot->nTarget];
    spritetype *pTarget = pOther->pSprite;
    int nWeapon = pPlayer->curWeapon;
    const BOTWEAPON *pWeapon = &botWeapons[(unsigned)nWeapon < kWeaponMax ? nWeapon : 0];
    int nDist = ClipLow(approxDist(pTarget->x-pSprite->x, pTarget->y-pSprite->y), 1);

    botSelectWeapon(pBot, pPlayer, pInput, nDist);

    // Aim at the chest, ahead of a moving target for slow projectiles.
    int top, bottom;
    GetSpriteExtents(pTarget, &top, &bottom);
    int tx, ty;
    botLeadTarget(pBot, pPlayer, pTarget, pWeapon->nSpeed, &tx, &ty);
    int tz = top+(bottom-top)/3;
    int nAimDist = ClipLow(approxDist(tx-pSprite->x, ty-pSprite->y), 1);
    int nAimAng = getangle(tx-pSprite->x, ty-pSprite->y);
    int nAimSlope = divscale10(tz-pPlayer->zWeapon, nAimDist);
    if (nWeapon == kWeaponTNT)
        nAimSlope -= ClipHigh(nAimDist/12, 1400);   // lob it

    // Aim wobble, reduced the longer we track the same target.
    if (pBot->nAimErrUntil <= pBot->nFrame)
    {
        int nError = pSkill->nAimError;
        pBot->nAimErrAng = botRandRange(pBot, nError*2+1)-nError;
        pBot->nAimErrSlope = (botRandRange(pBot, nError*2+1)-nError)*8;
        pBot->nAimErrUntil = pBot->nFrame+8+botRandRange(pBot, 12);
    }
    nAimAng = (nAimAng+pBot->nAimErrAng)&kAngMask;
    nAimSlope += pBot->nAimErrSlope;
    *pAimAng = nAimAng;
    *pAimSlope = nAimSlope;

    if (pBot->nReaction > 0)
        pBot->nReaction--;

    // Fire when roughly on target. Blood's autoaim helps those who use it,
    // bots included, exactly as it helps players.
    int nTolerance = ClipRange(48000/nDist, 6, 96);
    if (gProfile[pPlayer->nPlayer].nAutoAim == 1 || (gProfile[pPlayer->nPlayer].nAutoAim == 2 && !pWeapon->nSpeed && !pWeapon->bHold))
        nTolerance = ClipLow(nTolerance, 40);
    int nOffTarget = klabs(botAngleDelta(pSprite->ang, nAimAng));
    bool bFire = pBot->nReaction <= 0 && nOffTarget <= nTolerance && pBot->nCeaseFire <= 0;
    if (pBot->nCeaseFire > 0)
        pBot->nCeaseFire--;
    if (nDist > pWeapon->nMaxRange+2000 && nWeapon != kWeaponPitchfork)
        bFire = bFire && (botRand(pBot) & 3) == 0;
    switch (nWeapon)
    {
    case kWeaponPitchfork:
    case kWeaponBeast:
        bFire = bFire && nDist < pWeapon->nMaxRange;
        break;
    case kWeaponSprayCan:
        bFire = bFire && nDist < pWeapon->nMaxRange;
        break;
    case kWeaponVoodoo:
        bFire = bFire && nOffTarget < 24;
        break;
    case kWeaponProxyTNT:
    case kWeaponRemoteTNT:
        bFire = false;
        break;
    }
    if (bFire && pWeapon->bSplash)
    {
        // Do not blow ourselves up on a wall or a nearby target.
        hitdata_t hit;
        vec3_t pos = { pSprite->x, pSprite->y, pPlayer->zWeapon };
        hitscan(&pos, pSprite->sectnum, Cos(nAimAng)>>16, Sin(nAimAng)>>16, nAimSlope<<4, &hit, CLIPMASK1);
        int nHitDist = approxDist(hit.x-pSprite->x, hit.y-pSprite->y);
        if (nHitDist < pWeapon->nMinRange || nDist < pWeapon->nMinRange)
            bFire = false;
    }

    if (pWeapon->bHold)
    {
        // Dynamite: hold to light and wind up, let go when strong enough.
        bool bLit = pPlayer->weaponState == 6;
        if (bLit)
        {
            int nWant = ClipRange((nDist-1500)*65536/7000, 0, 65536);
            pBot->nFireFrames++;
            pInput->buttonFlags.shoot = pPlayer->throwPower < nWant && pBot->nFireFrames < kBotFPS*2;
            if (!pInput->buttonFlags.shoot)
            {
                pBot->nFireFrames = 0;
                pBot->nCeaseFire = kBotFPS/2;
            }
        }
        else
        {
            pBot->nFireFrames = 0;
            pInput->buttonFlags.shoot = bFire;
        }
    }
    else if (bFire)
    {
        bool bAlt = false;
        switch (nWeapon)
        {
        case kWeaponShotgun:
            bAlt = nDist < 1600 && pPlayer->ammoCount[2] >= 2;
            break;
        case kWeaponTommy:
            bAlt = nDist < 2200 && pPlayer->ammoCount[3] > 50;
            break;
        case kWeaponTesla:
            bAlt = nDist < 5000 && pPlayer->ammoCount[7] >= 60 && (botRand(pBot) & 15) == 0;
            break;
        }
        if (bAlt)
            pInput->buttonFlags.shoot2 = 1;
        else
            pInput->buttonFlags.shoot = 1;
    }

    // Movement: keep the weapon's preferred distance and strafe around it
    // (DukeBot's fight position and sine-sum strafing).
    int nFightDist = (pWeapon->nIdealMin+pWeapon->nIdealMax)/2;
    if (nWeapon == kWeaponPitchfork || nWeapon == kWeaponBeast || nWeapon == kWeaponSprayCan)
        nFightDist = pWeapon->nIdealMax/2;
    nFightDist = ClipRange(nFightDist, 256, 6000);
    // Stay further back from someone high above us.
    nFightDist = ClipLow(nFightDist, klabs(pTarget->z-pSprite->z)>>4);
    int fx = pTarget->x+(int)((int64_t)(pSprite->x-pTarget->x)*nFightDist/nDist);
    int fy = pTarget->y+(int)((int64_t)(pSprite->y-pTarget->y)*nFightDist/nDist);
    int dx = fx-pSprite->x, dy = fy-pSprite->y;
    int nFightGap = approxDist(dx, dy);
    if (nFightGap > 192)
    {
        pMove->dx = dx*2048/nFightGap;
        pMove->dy = dy*2048/nFightGap;
    }

    int nStrafeClock = pBot->nFrame*kTicsPerFrame+pPlayer->nPlayer*13468;
    int nStrafe = Sin((nStrafeClock<<6)&kAngMask)>>16;
    nStrafe += Sin(((nStrafeClock+4245)<<5)&kAngMask)>>16;
    nStrafe += Sin(((nStrafeClock+6745)<<4)&kAngMask)>>16;
    nStrafe += Sin(((nStrafeClock+15685)<<3)&kAngMask)>>16;
    // Out of an opponent's line of fire.
    int nTheirDelta = botAngleDelta(pTarget->ang, getangle(pSprite->x-pTarget->x, pSprite->y-pTarget->y));
    if (klabs(nTheirDelta) < 256)
        nStrafe += nTheirDelta >= 0 ? 8192 : -8192;
    nStrafe = nStrafe*pSkill->nStrafe/256;
    int nSide = (nAimAng+512)&kAngMask;
    int sx = mulscale30(nStrafe, Cos(nSide))>>4;
    int sy = mulscale30(nStrafe, Sin(nSide))>>4;
    // Do not strafe into a wall.
    if (botNavProbe(pPlayer, getangle(sx, sy), 256) < 128)
    {
        sx = -sx;
        sy = -sy;
    }
    pMove->dx += sx;
    pMove->dy += sy;
    pMove->nSpeed = 2048;

    if (nDist > 600 && pPlayer->posture != kPostureSwim && pSkill->nStrafe > 128 && botRandRange(pBot, 200) == 0)
        pMove->bJump = true;
}

void botDodge(BOTINFO *pBot, PLAYER *pPlayer, BOTMOVE *pMove)
{
    const BOTSKILL *pSkill = &gBotSkillInfo[pBot->nSkill];
    spritetype *pSprite = pPlayer->pSprite;
    for (int nSprite = headspritestat[kStatProjectile]; nSprite >= 0; nSprite = nextspritestat[nSprite])
    {
        spritetype *pMissile = &sprite[nSprite];
        if (actSpriteOwnerToSpriteId(pMissile) == pSprite->index)
            continue;
        int vx = xvel[nSprite]>>12, vy = yvel[nSprite]>>12;
        int nSpeed = approxDist(vx, vy);
        if (nSpeed < 16)
            continue;
        int rx = pSprite->x-pMissile->x, ry = pSprite->y-pMissile->y;
        // Closest approach within the next second.
        int64_t dot = (int64_t)rx*vx+(int64_t)ry*vy;
        if (dot <= 0)
            continue;
        int t = (int)(dot/((int64_t)nSpeed*nSpeed));
        if (t > kBotFPS)
            continue;
        int cx = pMissile->x+vx*t-pSprite->x, cy = pMissile->y+vy*t-pSprite->y;
        if (approxDist(cx, cy) > 420)
            continue;
        // We only react to what we can see, and not always.
        if (klabs(botAngleDelta(pSprite->ang, getangle(-rx, -ry))) > gBotSkillInfo[pBot->nSkill].nFov+128)
            continue;
        if (((nSprite*2654435761u+(pBot->nFrame>>3)*40503u)>>24) > (unsigned)pSkill->nDodge)
            continue;
        // Step off its line.
        int nSide = (vx*ry-vy*rx) > 0 ? 512 : -512;
        int nAng = (getangle(vx, vy)+nSide)&kAngMask;
        pMove->dx += Cos(nAng)>>17;
        pMove->dy += Sin(nAng)>>17;
        pMove->nSpeed = 2048;
    }
    // Get away from lit dynamite and bombs lying around.
    for (int nSprite = headspritestat[kStatThing]; nSprite >= 0; nSprite = nextspritestat[nSprite])
    {
        spritetype *pThing = &sprite[nSprite];
        switch (pThing->type)
        {
        case kThingArmedTNTStick:
        case kThingArmedTNTBundle:
        case kThingArmedSpray:
        case kThingArmedProxBomb:
        case kThingArmedRemoteBomb:
        case kThingNapalmBall:
            break;
        default:
            continue;
        }
        int rx = pSprite->x-pThing->x, ry = pSprite->y-pThing->y;
        int nDist = approxDist(rx, ry);
        if (nDist > 1800 || klabs(pThing->z-pSprite->z) > (64<<8))
            continue;
        if (!cansee(pSprite->x, pSprite->y, pPlayer->zView, pSprite->sectnum, pThing->x, pThing->y, pThing->z, pThing->sectnum))
            continue;
        int nAway = getangle(rx, ry);
        pMove->dx = (pMove->dx>>1)+(Cos(nAway)>>17);
        pMove->dy = (pMove->dy>>1)+(Sin(nAway)>>17);
        pMove->nSpeed = 2048;
    }
}

void botUseInventory(BOTINFO *pBot, PLAYER *pPlayer, GINPUT *pInput)
{
    int nHealth = pPlayer->pXSprite->health>>4;
    if (nHealth < 60 && pPlayer->packSlots[kPackMedKit].curAmount > 0 && (pBot->nFrame & 15) == 0)
        pInput->useFlags.useMedKit = 1;
}
