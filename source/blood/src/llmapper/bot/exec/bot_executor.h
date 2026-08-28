//-------------------------------------------------------------------------
// Carrying out one chosen semantic action.
//
// The executor follows the goal it was given and never picks a different
// one. When the world stops supporting the goal it says so with a typed
// outcome and hands the decision back to the planner.
//
// This is also where concrete points in space come from. A Region is the
// planning unit; getting across one is a real navigation problem, solved
// here in the actor's configuration space, and none of its answers are
// stored anywhere.
//-------------------------------------------------------------------------
#pragma once

#include <vector>

#include "../nav/local_path.h"
#include "../planner/bot_planner.h"
#include "../semantic/actor_control.h"
#include "../semantic/semantic_world.h"
#include "../traversal/traversal_model.h"

namespace exec {

enum class Outcome
{
    Running,
    Succeeded,
    WorldChanged,
    Blocked,
    TargetUnavailable,
};

// Why a goal could not be carried on with. Reported, never recovered from.
enum class Block
{
    None,
    NoRoute,        // no chain of executable transitions leads there
    NoCrossing,     // the route names an opening with no verified crossing
    NoLocalPath,    // the body cannot get across this region to that point
    NoProgress,     // the body is being driven and is getting no nearer
    TargetGone,         // what was being done no longer exists
    TargetUnexecutable, // it exists, and the world no longer offers it
    OptionGone,         // the place it was to be done from is no longer one
    NotAccepted,        // the act was commanded and the world never took it
};

// Why a leg inside a region could not be worked out. Enough to tell a wrong
// free-space polygon from a wrong erosion from a wrong query, without
// guessing.
struct LocalFailure
{
    semantic::RegionId region = semantic::kNoId;
    semantic::Vec2 from;
    semantic::Vec2 to;
    bool fromInside = false;
    bool toInside = false;
    bool fromFree = false;
    bool toFree = false;
    size_t nodes = 0;
    size_t solid = 0;
    size_t corners = 0;
    size_t holes = 0;
    size_t openings = 0;
};

class Executor
{
public:
    void begin(const planner::Decision &decision,
               const semantic::SemanticWorld &world);
    void clear();
    bool active() const { return m_active; }

    planner::Intent intent() const { return m_intent; }
    semantic::RegionId destination() const { return m_destination; }
    semantic::AffordanceId affordance() const { return m_affordance; }
    uint32_t option() const { return m_option; }
    semantic::RelationId inspecting() const { return m_relation; }
    // The crossing the body was being driven at when it stopped getting
    // anywhere, if it was being driven at one.
    semantic::RelationId leg() const
    {
        return m_index < m_via.size() ? m_via[m_index] : semantic::kNoId;
    }
    semantic::RegionId waypoint() const { return m_waypoint; }
    size_t routeLength() const { return m_route.size(); }
    const std::vector<semantic::RegionId> &route() const { return m_route; }
    size_t routeIndex() const { return m_index; }
    // True on the tick the route was worked out afresh, because the body was
    // somewhere the old one did not pass through.
    bool rerouted() const { return m_rerouted; }
    Block block() const { return m_block; }
    // The concrete point last driven at, and how many legs of local path it
    // came from. Diagnostic only.
    const semantic::Vec2 &aim() const { return m_aim; }
    size_t legs() const { return m_path.size(); }
    // The steering points as they stand, for reading a stuck leg back.
    const std::vector<semantic::Vec2> &path() const { return m_path; }
    int remaining() const { return m_remaining; }

    const LocalFailure &lastFailure() const { return m_failure; }

    // The traversal model is not const here only because its map of a
    // region's free space is worked out once and shared: the executor asks
    // the same question the derivation asked, and must get the same answer.
    Outcome tick(const semantic::SemanticWorld &world,
                 traversal::TraversalModel &traversal,
                 const traversal::ActorProfile &profile,
                 semantic::MotorCommand &command);

private:
    bool follow(const semantic::SemanticWorld &world,
                const traversal::TraversalModel &traversal);
    // Is this leg getting anywhere? Answered against what the leg is for,
    // and timed by the speed this body has actually been managing, so there
    // is no clock in it and nothing to tune.
    bool losingGround(const semantic::Vec2 &at, size_t legsLeft,
                      int remaining, int radius);
    // Drive toward a point inside one region, around whatever is in the way.
    //
    // `beyond`, when given, is where the crossing at `to` was verified to
    // leave the body. It is driven instead of `to` on the last leg, once
    // nothing stands between the body and the opening: an opening is a line
    // between two spaces, and a body driven at a point on a line stops on
    // the line.
    bool steerInside(const semantic::SemanticWorld &world,
                     traversal::TraversalModel &navigator,
                     const semantic::Region &region,
                     const semantic::Vec2 &from, const semantic::Vec2 &to,
                     int radius, semantic::MotorCommand &command,
                     const semantic::Vec2 *beyond = nullptr);

    bool m_active = false;
    planner::Intent m_intent = planner::Intent::None;
    semantic::RegionId m_destination = semantic::kNoId;
    semantic::AffordanceId m_affordance = semantic::kNoId;
    uint32_t m_option = 0;
    semantic::RelationId m_relation = semantic::kNoId;
    semantic::RegionId m_waypoint = semantic::kNoId;
    Block m_block = Block::None;
    int m_attemptsAtStart = 0;
    size_t m_index = 0;
    bool m_rerouted = false;
    // What this leg is for, the nearest the body has come to it, and how
    // long since that got any better. Progress is getting nearer to the
    // thing being driven at; a body wedged against geometry the model
    // believes is open does not move at all, and a body shoved about moves a
    // great deal, and neither of those is progress.
    semantic::Vec2 m_legLast;
    size_t m_legsLeft = 0;
    int m_remaining = 0;
    int m_radiusUsed = 0;
    bool m_legKnown = false;
    int m_legClosest = 0;
    int m_legBest = 0;
    int m_bodyBest = 0;
    int64_t m_legStalled = 0;
    // Which stance of the act this goal set out for, said as the stance
    // rather than as its place in a list. The list is worked out again as
    // the world changes and a pose that was third can become first without
    // anything about it having changed.
    semantic::Vec3 m_optionAt;
    bool m_optionKnown = false;
    size_t m_domainAtStart = 0;
    bool m_leaveAtOnce = false;
    bool m_acted = false;
    semantic::Vec2 m_escapeTo;
    std::vector<semantic::RegionId> m_route;
    // The same route as pieces of free space. A Region can be more than one
    // piece and a route can pass through the same Region twice; without
    // this, arriving in that Region cannot be told from arriving in it the
    // second time, and the walk skips everything in between.
    std::vector<size_t> m_through;
    std::vector<semantic::RelationId> m_via;
    std::vector<semantic::Segment> m_openings;
    std::vector<semantic::Vec2> m_path;
    semantic::Vec2 m_aim;
    LocalFailure m_failure;
};

const char *outcomeName(Outcome outcome);
const char *blockName(Block block);

} // namespace exec
