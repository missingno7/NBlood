//-------------------------------------------------------------------------
// Provenance for humans.
//
// This module reads both sides of the boundary so that a log line can say
// "Region 17 [Blood: sector 12]". Nothing here is reachable from the
// planner or the executor, and nothing it computes is fed back into a
// decision: deleting the whole module would not change a single tick.
//-------------------------------------------------------------------------
#pragma once

#include <cstddef>
#include <cstdio>

#include "../blood/blood_world_adapter.h"
#include "../blood/caleb_physics.h"
#include "../exec/bot_executor.h"
#include "../semantic/semantic_world.h"

namespace botdebug {

void describeRegion(const bloodmap::WorldAdapter &adapter,
                    semantic::RegionId id, char *out, size_t size);

// The outline of a region, for reading a run back against the map.
void describeFootprint(const semantic::Region &region, char *out,
                       size_t size);

void describeAffordance(const bloodmap::WorldAdapter &adapter,
                        semantic::AffordanceId id, char *out, size_t size);

// A piece of the world the mapper found and did not make a place of, and
// what the engine says would bring it back.
void describeRejection(const bloodmap::RejectedSpace &space, char *out,
                       size_t size);

// Why a leg inside one region could not be worked out.
// One way out of the world the bot believes in, and why it stops there.
// The engine's own word for a kind of thing. Debug only.
const char *containerName(int tag);

void describeRelation(const bloodmap::WorldAdapter &adapter,
                      const bloodmap::CalebPhysics &physics,
                      const traversal::TraversalModel &traversal,
                      const semantic::SpatialRelation &relation,
                      char *out, size_t size);

void describeLocalFailure(const exec::LocalFailure &failure, char *out,
                          size_t size);

// One trajectory sample. Engine-shaped on purpose: the offline tools plot a
// run against the map it was run on. Nothing reads this back.
void sampleTrajectory(FILE *out, int gameTime, int tick,
                      const GINPUT &issued);

} // namespace botdebug
