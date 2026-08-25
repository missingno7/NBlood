//-------------------------------------------------------------------------
// Bot intelligence.
//
// Of the things this bot can currently execute, which one does it want to
// do? That is the whole job. What is physically there is the world's answer,
// what the actor can do with it is the physics layer's answer, and how to
// carry it out is the executor's. This file must not be able to tell which
// engine it is planning for.
//-------------------------------------------------------------------------
#pragma once

#include <vector>

#include "../semantic/semantic_world.h"
#include "../traversal/traversal_model.h"

namespace planner {

enum class Intent
{
    None,
    GoTo,
    Approach,           // go and look at a way that cannot be gone through
    ExecuteAffordance,
};

enum class StallReason
{
    None,
    ActorHasNoRegion,
    NoKnownAction,
};

struct Diagnosis
{
    StallReason reason = StallReason::None;
    int knownRegions = 0;
    int reachableRegions = 0;
    int uncrossedGateways = 0;
    int unenteredRegions = 0;
    int uninspectedGateways = 0;
    int knownAffordances = 0;
    int affordancesUnreachable = 0;
    int affordancesAttemptedInertly = 0;
    // Physically real, but this bot has no executor for it. Reported so a
    // stall says what the world offered that could not be taken up.
    int possibleButUnexecutable = 0;
};

struct Decision
{
    Intent intent = Intent::None;
    semantic::RegionId destination = semantic::kNoId;
    semantic::AffordanceId affordance = semantic::kNoId;
    semantic::RelationId relation = semantic::kNoId;
    uint32_t option = 0;    // which of the affordance's execution options
    Diagnosis diagnosis;
    // Which rule offered this, for reading a run back.
    const char *why = "none";
};

// 1. prefer a way out of the known world that has not been taken yet: a
//    gateway the actor can drive, has never gone through, and which leads
//    somewhere it has never stood. Regions are large, so having entered one
//    is not the same as having explored from it -- the gateways are what
//    is left to find out about, and there is a finite number of them;
// 2. otherwise a way that is known, has never been gone through and cannot
//    be -- a shut door is a fact about the world, and standing at it is how
//    the actor finds out what would open it;
// 3. otherwise an unattempted reachable affordance;
// 4. otherwise an affordance whose last attempt opened something up, because
//    an interaction is allowed to be repeatable -- but one that moves the
//    world without widening it is not worth doing again;
// 5. otherwise report why there is nothing to do.
Decision choose(const semantic::SemanticWorld &world,
                const traversal::TraversalModel &traversal);

const char *stallName(StallReason reason);
const char *intentName(Intent intent);

} // namespace planner
