//-------------------------------------------------------------------------
// The one thing the executor is allowed to say to the body.
//
// A motor command is expressed in semantic terms: a place in space to move
// toward and an affordance to address. How that becomes a control input --
// which axis is forward, how momentum is cancelled, where exactly to look --
// belongs to the engine adapter, which is the only part that knows.
//-------------------------------------------------------------------------
#pragma once

#include "semantic_world.h"

namespace semantic {

struct MotorCommand
{
    bool move = false;
    Vec3 moveToward;

    // Address this affordance: aim at it and, when the engine confirms the
    // aim resolves to it, deliver the action.
    AffordanceId address = kNoId;
    // Which of the affordance's execution options is being used. The adapter
    // owns the pose behind it; this is how it is named.
    uint32_t option = 0;
    bool act = false;
};

} // namespace semantic
