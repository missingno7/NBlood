#include "bot_input.h"

#include <cassert>
#include <cstring>

static bool allZero(const GINPUT &input)
{
    const unsigned char *bytes = reinterpret_cast<const unsigned char *>(&input);
    for (size_t i = 0; i < sizeof(input); ++i)
        if (bytes[i] != 0)
            return false;
    return true;
}

int main()
{
    using llmapper::PhysicalCommand;
    using llmapper::PhysicalCommandType;

    assert(allZero(llmapper::commandToInput({})));

    PhysicalCommand move;
    move.type = PhysicalCommandType::Move;
    move.forward = 900;
    move.strafe = -350;
    move.turn = fix16_from_int(17);
    move.look = fix16_from_int(-3);
    const GINPUT movement = llmapper::commandToInput(move);
    assert(movement.forward == 900);
    assert(movement.strafe == -350);
    assert(movement.q16turn == fix16_from_int(17));
    assert(movement.q16mlook == fix16_from_int(-3));
    assert(!movement.buttonFlags.byte && !movement.keyFlags.word);

    const GINPUT jump = llmapper::commandToInput(
        { PhysicalCommandType::Jump });
    assert(jump.buttonFlags.jump && !jump.buttonFlags.crouch);

    const GINPUT crouch = llmapper::commandToInput(
        { PhysicalCommandType::Crouch });
    assert(crouch.buttonFlags.crouch && !crouch.buttonFlags.jump);

    const GINPUT use = llmapper::commandToInput(
        { PhysicalCommandType::Use });
    assert(use.keyFlags.action && !use.buttonFlags.byte);
    return 0;
}
