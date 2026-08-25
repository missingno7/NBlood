//-------------------------------------------------------------------------
// Reading physical support geometry out of Blood.
//
// This is the only file that turns sectors, walls and sprites into pieces of
// physical space. It makes no semantic decisions: it reports what surfaces
// exist, what free volume sits over them, what closes them off, and which
// engine object owns each one. Whether any of that amounts to one Region or
// five is decided afterwards, by shape, in terrain/.
//-------------------------------------------------------------------------
#pragma once

#include <vector>

#include "../terrain/terrain_model.h"

namespace bloodmap {

// Support provenance, in GetZRange's own encoding: a sector floor is tagged
// above 16384 and a solid sprite above 49152, exactly as florhit reports the
// surface holding a body up. Kept below the boundary; the debug layer is the
// only thing that ever decodes it.
constexpr uint64_t kSectorSupport = 16384;
constexpr uint64_t kSpriteSupport = 49152;

struct TerrainSnapshot
{
    std::vector<terrain::SupportFace> faces;
    std::vector<terrain::Seam> seams;
    // Parallel to faces: which engine object each came from, for provenance
    // and for deciding what a geometry change touched.
    std::vector<int> faceSector;
};

// Read the whole loaded world's support geometry.
void extractTerrain(TerrainSnapshot &out);

// Which region-owning engine objects moved since the last call.
void collectChangedSectors(std::vector<int> &out);

// True while something set in motion has not come to rest.
bool worldSettling();

// A hash of the engine state the model depends on; used only to notice that
// something changed, never interpreted.
uint64_t worldSignature();
// How many seams the last extraction found something standing in. Debug.
int obstructedSeams();
int obstructedSeamCount();
void obstructedSeamAt(int index, int &x1, int &y1, int &x2, int &y2,
                      int &spriteId, int &owner, int &behind);

} // namespace bloodmap
