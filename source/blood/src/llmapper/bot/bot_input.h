//-------------------------------------------------------------------------
// Minimal physical command vocabulary for the LLMapper bot.
//-------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <cstdint>

#include "compat.h"
#include "fix16.h"
#include "../../controls.h"

namespace llmapper {

enum class PhysicalCommandType
{
    Neutral,
    Move,
    Jump,
    Crouch,
    Use,
};

struct PhysicalCommand
{
    PhysicalCommandType type = PhysicalCommandType::Neutral;
    int16_t forward = 0;
    int16_t strafe = 0;
    fix16_t turn = 0;
    fix16_t look = 0;
};

inline GINPUT commandToInput(const PhysicalCommand &command)
{
    GINPUT input = {};
    switch (command.type)
    {
    case PhysicalCommandType::Move:
        input.forward = int16_t(std::max(-kMaxMoveInput,
            std::min(kMaxMoveInput, int(command.forward))));
        input.strafe = int16_t(std::max(-kMaxMoveInput,
            std::min(kMaxMoveInput, int(command.strafe))));
        input.q16turn = command.turn;
        input.q16mlook = command.look;
        break;
    case PhysicalCommandType::Jump:
        input.buttonFlags.jump = 1;
        break;
    case PhysicalCommandType::Crouch:
        input.buttonFlags.crouch = 1;
        break;
    case PhysicalCommandType::Use:
        input.keyFlags.action = 1;
        break;
    case PhysicalCommandType::Neutral:
        break;
    }
    return input;
}

} // namespace llmapper
