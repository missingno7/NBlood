//-------------------------------------------------------------------------
// Drawing the semantic world into the game view.
//
// Purely diagnostic: it reads the model and paints it, and nothing it does
// is read back. Its job is to make a wrong world visible -- a region that
// covers ground it should not, a gateway the model thinks is walkable when
// it plainly is not, a piece of floor with no region on it at all.
//-------------------------------------------------------------------------
#pragma once

#include "fix16.h"
#include "../semantic/semantic_world.h"
#include "../traversal/traversal_model.h"

namespace botdebug {

struct Camera
{
    int x = 0;
    int y = 0;
    int z = 0;
    fix16_t angle = 0;
    fix16_t horizon = 0;
};

// Paints regions, the openings between them and what can be done from them.
void drawWorld(const Camera &camera, const semantic::SemanticWorld &world,
               const traversal::TraversalModel &traversal);

} // namespace botdebug
