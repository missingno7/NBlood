//-------------------------------------------------------------------------
/*
Copyright (C) 2022, 2023 - Jordon Moss (Aka. StrikerTheHedgefox)
Copyright (C) 2026 - NBlood developers

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

// Internal interface shared by bot.cpp, botnav.cpp and botcombat.cpp.
// Everything here only reads the game state; bots act exclusively through
// the GINPUT they produce.
#pragma once

#include "bot.h"
#include "player.h"

// Frames run at kTicsPerSec / kTicsPerFrame = 30 per second.
#define kBotFPS 30

enum
{
    // Portal requirements found by the route search.
    kBotPortalJump   = 1,
    kBotPortalCrouch = 2,
    kBotPortalDoor   = 4,  // closed now, may be opened by pressing action
    kBotPortalLink   = 8,  // ROR water/stack link or teleporter, no wall
    kBotPortalLift   = 16, // floor higher than a jump, but the sector moves
    kBotPortalBridge = 32, // only crossable on a sprite (bridge, ledge) in the way
    kBotPortalDrop   = 64, // a fall we take on purpose
};

struct BOTMOVE
{
    int dx, dy;     // desired world-space movement direction, 0,0 = stand
    int nSpeed;     // 0..2048, the input range of forward/strafe
    int nFaceAng;   // where to look when not aiming at an enemy, -1 = along movement
    bool bJump;
    bool bCrouch;
    bool bAction;
};

struct BOTNAV
{
    int nRouteFrom, nRouteTo;   // the cached route
    int nRouteFrame;
    int nNextSect;              // next sector of the route, -1 = none
    int nPortal;                // wall of the current sector leading there, -1 = link
    int nPortalFlags;
    int nRouteLen;
    vec2_t crossing;            // where the portal will be crossed
    vec2_t linkPos;             // for links: where the link is entered
    // Progress monitoring.
    vec2_t checkPos;
    int nCheckFrame;
    int nStuck;                 // consecutive progress checks that failed
    int nEscape;                // frames left of an escape manoeuvre
    int nEscapeAng;
    int nDoorWait;
    // Portals that recently failed. They are only avoided for a while; the
    // geometry can change (doors, lifts) and the failure may have been bad luck.
    int nAvoidWall[8];
    int nAvoidUntil[8];
    int nAvoidHead;
    // Local steering memory, keeps detours from flip-flopping.
    int nSteerAng;
    int nSteerUntil;
    // Time spent trying to leave the current sector along the route.
    int nStallSect;
    int nStallFrames;
};

struct BOTINFO
{
    bool bActive;
    uint32_t nRand;
    int nSkill;
    int nFrame;
    int nRespawnWait;

    // Enemies.
    int nTarget;                // player index, -1 = none
    int nReaction;              // frames until we may shoot the target
    bool bTargetVisible;
    int nSeenFrame[kMaxPlayers];   // bot frame the player was last seen (0 = never)
    vec3_t seenPos[kMaxPlayers];
    int nSeenSect[kMaxPlayers];
    int nLastHealth;
    int nHurtBy;
    int nHurtFrame;

    // Pickups.
    int nItem;
    int nItemScore;
    int nBadItem[4];            // pickups we failed to reach, ignored for a while
    int nBadUntil[4];
    int nBadHead;
    int nItemFail;
    int nItemCooldown;

    // Wandering when nobody is known.
    int nRoamSect;
    int nRoamUntil;

    // Aim imperfection.
    int nAimErrAng;
    int nAimErrSlope;
    int nAimErrUntil;

    // Weapons.
    int nWeaponWait;
    int nFireFrames;            // frames the fire button has been held
    int nCeaseFire;             // frames to keep the button released
    int nStrafeSign;
    int nStrafeUntil;
    int nJumpWait;
    int nAvoids;
    bool bAltFire;              // fire mode chosen for the current volley
    int nAltUntil;
    int nAltShots;
    int nDoorsOpened;
    bool bFalling;

    BOTNAV nav;
};

struct BOTSKILL
{
    int nReactMin, nReactMax;   // frames before firing at a newly seen enemy
    int nTurnSpeed;             // max angle change per frame (2048 = full turn)
    int nAimError;              // aim wobble amplitude, angle units
    int nFov;                   // half view cone, angle units
    int nAutoAim;               // gProfile autoaim mode
    int nStrafe;                // 0..256 how much to strafe in a fight
    int nDodge;                 // 0..256 chance to react to an incoming projectile
    int nMemory;                // frames an unseen enemy is still hunted
    int nLead;                  // 0..256 how well projectile shots are led
};

extern BOTINFO gBotInfo[kMaxPlayers];
extern const BOTSKILL gBotSkillInfo[kBotSkillMax+1];

uint32_t botRand(BOTINFO *pBot);
int botRandRange(BOTINFO *pBot, int nRange);
int botAngleDelta(int a, int b);
void botGetExtents(PLAYER *pPlayer, int *pTop, int *pBottom);

// botnav.cpp
void botNavReset(BOTINFO *pBot);
// Returns false when there is currently no known way to nSector.
bool botNavMove(BOTINFO *pBot, PLAYER *pPlayer, int x, int y, int z, int nSector, BOTMOVE *pMove);
int botNavProbe(PLAYER *pPlayer, int nAng, int nDist, int *pHit = NULL);
int botNavRoamSector(BOTINFO *pBot, PLAYER *pPlayer);
bool botNavReachable(PLAYER *pPlayer, int nSector);
// Would moving this way walk us off a ledge we did not mean to take?
bool botNavLedge(PLAYER *pPlayer, int nAng, int nDist);
bool botNavOnBridge(PLAYER *pPlayer);
void botNavSafeMove(BOTINFO *pBot, PLAYER *pPlayer, BOTMOVE *pMove);
void botNavReport(PLAYER *pPlayer);
void botNavDrawMap(PLAYER *pPlayer);

// botcombat.cpp
bool botCanSeePlayer(PLAYER *pPlayer, PLAYER *pOther);
void botUpdateTargets(BOTINFO *pBot, PLAYER *pPlayer);
void botUpdateItemGoal(BOTINFO *pBot, PLAYER *pPlayer, bool bNeedy);
void botGiveUpItem(BOTINFO *pBot);
bool botNeedsItems(PLAYER *pPlayer);
void botFight(BOTINFO *pBot, PLAYER *pPlayer, GINPUT *pInput, BOTMOVE *pMove, int *pAimAng, int *pAimSlope);
void botSelectWeapon(BOTINFO *pBot, PLAYER *pPlayer, GINPUT *pInput, int nDist);
void botDodge(BOTINFO *pBot, PLAYER *pPlayer, BOTMOVE *pMove);
void botUseInventory(BOTINFO *pBot, PLAYER *pPlayer, GINPUT *pInput);
