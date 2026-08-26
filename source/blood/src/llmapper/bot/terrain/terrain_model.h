//-------------------------------------------------------------------------
// Turning physical support geometry into semantic Regions.
//
// The input is what the mapper found in the world: pieces of support surface
// and the seams between them. Seams that carry no physical meaning are
// dissolved, and what is left of each continuous piece of space becomes one
// Region -- however concave it is, however many things stand in it, and
// however many pieces the engine happened to store it in.
//
// Nothing here cuts a space up to make it easier to walk across. That is a
// question about a body, and no body appears in this file.
//
// This file knows about no engine. Its inputs are geometry and opaque tags,
// which is what lets the spatial invariants be tested directly.
//-------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <vector>

#include "../semantic/semantic_world.h"

namespace terrain {

// One piece of support surface as the mapper found it, before any semantic
// decision has been taken about it.
struct SupportFace
{
    semantic::Loop outline;
    std::vector<semantic::Loop> obstacles; // solid objects standing on it
    semantic::Plane support;
    semantic::Plane ceiling;
    semantic::Hazard hazard;
    semantic::ClearanceClass clearance = semantic::ClearanceClass::Standing;
    // Opaque to this file. `supportTag` identifies the physical thing that
    // holds a body up, `stateTag` the thing that can move it, `provenance`
    // is carried through for the debug layer and read by nothing else.
    uint64_t supportTag = 0;
    uint64_t stateTag = 0;
    uint64_t provenance = 0;
};

// Two faces meet along a segment of their boundary. `solid` means the world
// itself closes the join: a wall in space rather than a line in storage.
struct Seam
{
    size_t left = 0;
    size_t right = 0;
    semantic::Vec2 from;
    semantic::Vec2 to;
    // The world has no opening here at all: a wall. Two spaces either side
    // of one are two spaces and always will be, so this decides clustering.
    bool solid = false;
    // Something is standing in the opening as things are now. A different
    // fact, and a temporary one: the spaces are adjacent, the opening is
    // real, and nothing goes through it at the moment. It deliberately does
    // not decide clustering -- what is one space does not stop being one
    // because something was put in the doorway.
    //
    // Opaque here. Whoever supplied the seam knows what it names.
    uint64_t obstruction = 0;
};

struct BuiltRegion
{
    semantic::Region region;
    uint64_t key = 0;                  // stable identity, shape-derived
    // What can move this space, in whatever terms the caller supplied, or
    // zero. Opaque here; carried up so that the thing which moved can be
    // recognised after it has moved.
    uint64_t stateTag = 0;
    std::vector<uint64_t> provenance;  // debug only
};

struct BuildResult
{
    std::vector<BuiltRegion> regions;
    std::vector<semantic::SpatialRelation> relations; // ids left unset
    // Parallel to `relations`: what is standing in each one, or zero. Opaque
    // here, and only meaningful to whoever supplied the seams.
    std::vector<uint64_t> obstructions;
    int clusters = 0;
    int barriers = 0;   // solid seams that ended up inside a region
};

// faces + seams -> Regions + candidate SpatialRelations.
void build(const std::vector<SupportFace> &faces,
           const std::vector<Seam> &seams, BuildResult &out);

} // namespace terrain
