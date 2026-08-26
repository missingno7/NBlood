#include "bot_executor.h"

#include <algorithm>

namespace exec {

using semantic::Affordance;
using semantic::MotorCommand;
using semantic::Region;
using semantic::RegionId;
using semantic::SemanticWorld;
using semantic::SpatialRelation;
using semantic::Vec2;
using semantic::kNoId;

void Executor::begin(const planner::Decision &decision,
                     const SemanticWorld &world)
{
    m_active = decision.intent != planner::Intent::None;
    m_intent = decision.intent;
    m_destination = decision.destination;
    m_affordance = decision.affordance;
    m_option = decision.option;
    m_relation = decision.relation;
    m_leaveAtOnce = decision.leaveAtOnce;
    m_escapeTo = decision.escapeTo;
    m_acted = false;
    m_waypoint = kNoId;
    m_block = Block::None;
    m_route.clear();
    m_through.clear();
    m_via.clear();
    m_index = 0;
    m_legKnown = false;
    m_bodyBest = 0;
    m_attemptsAtStart = 0;
    m_domainAtStart = 0;
    m_optionKnown = false;
    if (const Affordance *affordance = world.affordance(m_affordance))
    {
        m_attemptsAtStart = affordance->attempts;
        m_domainAtStart = affordance->domain.size();
        if (size_t(m_option) < affordance->domain.size())
        {
            m_optionAt = affordance->domain[m_option].at;
            m_optionKnown = true;
        }
    }
}

void Executor::clear()
{
    m_active = false;
    m_intent = planner::Intent::None;
    m_destination = kNoId;
    m_affordance = kNoId;
    m_relation = kNoId;
    m_waypoint = kNoId;
    m_route.clear();
    m_through.clear();
    m_via.clear();
    m_index = 0;
}

// Is this leg getting anywhere?
//
// Not "did the body move". A body wedged against geometry the model believes
// is open does not move at all, so a test that waits for movement to add up
// can never fire on the very case it is for. A body shoved along a wall
// moves a great deal and arrives nowhere.
//
// Nor is it "is the target any nearer": the way across a region goes round
// what is in it, and a body rounding a corner is walking away from where it
// is going for as long as the corner lasts. What shortens with every step
// taken to plan is what is left of the plan -- the local path from where the
// body is to where the leg ends.
//
// What counts as too long carries no clock: it is the time this body would
// need, at the best speed it has actually managed on this leg, to cover
// several of its own widths. A body that could have walked that far with
// nothing left of the plan going down is not on its way.
bool Executor::losingGround(const Vec2 &at, size_t legsLeft, int remaining,
                            int radius)
{
    const int width = std::max(1, radius);
    if (!m_legKnown)
    {
        m_legKnown = true;
        m_legLast = at;
        m_legsLeft = legsLeft;
        m_legClosest = remaining;
        m_legBest = 0;
        m_legStalled = 0;
        return false;
    }
    m_legBest = std::max(m_legBest, semantic::planarDistance(m_legLast, at));
    m_bodyBest = std::max(m_bodyBest, m_legBest);
    m_legLast = at;
    // Getting on with it is one of two things: a crossing behind the body
    // that was in front of it, or less of the current one left to walk than
    // has ever been left. Both are measured against the whole goal, because
    // a body taking turns at one opening changes leg constantly and would
    // otherwise be starting afresh every few ticks for ever.
    if (legsLeft < m_legsLeft)
    {
        m_legsLeft = legsLeft;
        m_legClosest = remaining;
        m_legStalled = 0;
        return false;
    }
    if (legsLeft > m_legsLeft)
    {
        // Went backwards. The count of crossings still to make only ever
        // ratchets down: a body taking turns at one opening would otherwise
        // undo its own high-water mark on the way back and count the next
        // trip out as progress again, for ever. What is left of a different
        // leg is not comparable with what was left of this one, so take the
        // new reading, but nothing here is progress.
        m_legClosest = remaining;
    }
    // Nearer by a whole body width, not merely nearer: what is left of the
    // plan jitters by a few units every tick as it is worked out afresh, and
    // counting that as progress is why a body going nowhere can look busy
    // indefinitely.
    else if (remaining + width < m_legClosest)
    {
        m_legClosest = remaining;
        m_legStalled = 0;
        return false;
    }
    ++m_legStalled;
    // How long is too long is a distance, not a count of ticks: several
    // dozen of this body's own widths. Turning it into ticks uses the speed
    // this body actually walks at -- the best it has managed, and at least
    // half of what it has managed anywhere on this goal, so a body that is
    // barely moving cannot buy itself unlimited time by barely moving.
    const int pace = std::max({ m_legBest, m_bodyBest / 2, 1 });
    return m_legStalled * pace > int64_t(width) * 48;
}

// Keep the route already in hand and only re-derive it when the actor is
// somewhere it does not pass through. Re-deriving a route to the same goal is
// not a new decision; the goal never changes here.
bool Executor::follow(const SemanticWorld &world,
                      const traversal::TraversalModel &traversal)
{
    const RegionId here = world.actor().region;
    const size_t piece = traversal.actorPlace();
    m_rerouted = false;
    size_t standing = m_route.size();
    // Which visit to this region the body is on, said in pieces of free
    // space rather than in region ids. A route across a ring-shaped ledge
    // and back along a bridge enters the same region at both ends; matching
    // on the region alone picks the later one and the walk starts at the far
    // end of a crossing it has not made.
    for (size_t index = m_route.size(); index-- > 0;)
        if (m_route[index] == here
            && (index >= m_through.size() || m_through[index] == piece))
        {
            standing = index;
            break;
        }
    if (standing == m_route.size())
        for (size_t index = m_route.size(); index-- > 0;)
            if (m_route[index] == here)
            {
                standing = index;
                break;
            }
    if (standing == m_route.size())
    {
        m_route.clear();
        m_through.clear();
        m_via.clear();
        m_index = 0;
        m_rerouted = true;
        // For an action, the way there is the way the planner priced: to the
        // very piece of free space the action is taken from, not merely to
        // its region. A region can be more than one place for a body with
        // width, and arriving in the wrong one is arriving nowhere.
        const bool found = m_intent == planner::Intent::ExecuteAffordance
            ? traversal.optionRoute(here, m_affordance, m_option,
                                    m_route, m_via, &m_through)
            : traversal.route(here, m_destination, m_route, m_via,
                              &m_through);
        if (!found || m_via.empty())
            return false;
        standing = 0;
    }
    // Where the body is on the route is where the body is. Going backwards
    // is a thing that happens -- a step taken too fast comes back down on
    // the near side -- and it has to be answered by crossing again, not by
    // driving the leg after the one the body is standing before. The two
    // sides of one opening no longer take turns because being somewhere now
    // means staying there until the body leaves it, which is where that
    // question belongs.
    m_index = standing;
    return m_index < m_via.size();
}

// One leg of walking: the region's own free space says which way round the
// things in it to go, and the first steering point that is actually somewhere
// else is what gets driven at.
bool Executor::steerInside(const SemanticWorld &world,
                           traversal::TraversalModel &traversal,
                           const Region &region, const Vec2 &from,
                           const Vec2 &to, int radius, MotorCommand &command,
                           const Vec2 *beyond)
{
    // The same walls the derivation used, so the executor cannot decide a
    // route exists where the model decided it does not.
    m_radiusUsed = radius;
    traversal.openingsFor(world, region.id, m_openings);
    const nav::LocalMap &map = traversal.navigator().mapFor(region,
                                                            m_openings,
                                                            radius);
    if (!map.path(from, to, m_path) || m_path.empty())
    {
        m_failure = LocalFailure();
        m_failure.region = region.id;
        m_failure.from = from;
        m_failure.to = to;
        m_failure.fromInside = semantic::pointInPolygon(region.shape(), from);
        m_failure.toInside = semantic::pointInPolygon(region.shape(), to);
        m_failure.fromFree = map.free(from);
        m_failure.toFree = map.free(to);
        m_failure.nodes = map.corners();
        m_failure.solid = map.solidEdges();
        m_failure.corners = region.footprint.size();
        m_failure.holes = region.holes.size();
        m_failure.openings = m_openings.size();
        return false;
    }
    // The furthest point on the route the body can already see.
    //
    // Taking the first one instead is right in spirit and wrong in practice.
    // The route is worked out afresh from wherever the body is, and the graph
    // connects that position to every node it can see, so the first hop is
    // routinely a node ten or twenty units away -- one it is already
    // standing on top of. Driving at a point ten units off at walking pace
    // overshoots it by thirty, and the next tick works out the same route
    // and drives back. In a room the body sails past and the next route no
    // longer mentions it; in a corridor there is nowhere to sail to, and it
    // rocks back and forth on the spot until the goal is given up on. Twelve
    // units short of a waypoint, for twenty-four thousand ticks.
    //
    // Seeing a point means the straight leg to it clears every wall by the
    // body's own width, which is the same test the route was built out of --
    // so skipping to it cannot cut a corner the route was going round. It is
    // the same route, minus the hops that were never going to move anything.
    size_t step = m_path.size();
    while (step-- > 0)
        if (map.clearBetween(from, m_path[step]))
            break;
    if (step >= m_path.size())
        step = 0;
    while (step < m_path.size() && m_path[step] == from)
        ++step;
    if (step >= m_path.size())
        return false;
    m_remaining = semantic::planarDistance(from, m_path[step]);
    for (size_t leg = step + 1; leg < m_path.size(); ++leg)
        m_remaining += semantic::planarDistance(m_path[leg - 1], m_path[leg]);
    m_aim = m_path[step];
    // Nothing left between the body and the opening: this leg is the
    // crossing itself, so drive it the way it was verified -- at the place
    // on the far side the body was left standing, not at the doorway.
    // Aim past the opening only once the body is at it.
    //
    // Driving at a point on the boundary stops the body on the line, so the
    // far side is what to drive at -- but only when the doorway is the next
    // thing, not while it is still most of a room away. The route to the
    // opening was checked and stays inside this space; the straight line to
    // a point beyond it was not, and from far enough back it cuts the corner
    // and leaves through somewhere else entirely. On AGTST8 that is a walk
    // out over the pit and a fall to the bottom of it, from a route that was
    // perfectly good right up to the last substitution.
    //
    // Near enough is within the body's own width of the opening: that is
    // exactly when aiming at the opening would stop the body short of going
    // through it.
    if (beyond && step + 1 == m_path.size()
        && semantic::planarDistance(from, m_path[step]) <= m_radiusUsed * 2)
        m_aim = *beyond;
    command.move = true;
    command.moveToward = { m_aim.x, m_aim.y,
                           region.support.zAt(m_aim.x, m_aim.y) };
    return true;
}

Outcome Executor::tick(const SemanticWorld &world,
                       traversal::TraversalModel &traversal,
                       const traversal::ActorProfile &profile,
                       MotorCommand &command)
{
    command = MotorCommand();
    m_waypoint = kNoId;
    m_block = Block::None;
    if (!m_active)
        return Outcome::Succeeded;

    const Region *goal = world.region(m_destination);
    if (!goal || !goal->exists)
        return Outcome::TargetUnavailable;

    const Affordance *affordance = nullptr;
    if (m_intent == planner::Intent::ExecuteAffordance)
    {
        affordance = world.affordance(m_affordance);
        // Did the thing this goal set out to do happen? That question comes
        // before every other one.
        //
        // Doing it is what changes the world, so the moment it lands, all
        // the tests below start failing: the door is triggered, so it is no
        // longer openable; the wall that was pushed no longer offers a place
        // to push it from. Asking those first reads success as the goal
        // falling apart -- "the way is gone", "the stance is gone" -- and
        // the act is done again from somewhere else. It only shows up once
        // one act has more than one surface, and then it shows up every
        // time.
        //
        // It is not finished while the thing is still moving, though.
        //
        // Pushing a door is not over when the push registers; it is over
        // when the door has swung. Ending the goal at the push means the
        // planner decides what to do next in a world where the door has not
        // opened yet -- so the way through it does not exist, and the best
        // thing on offer is whatever is furthest away. Then the door opens
        // behind the bot as it walks off across the level. A hundred and
        // twenty ticks early, once, is enough to send it somewhere else
        // entirely and never come back.
        if (affordance && affordance->attempts > m_attemptsAtStart)
            return affordance->settling || world.settling()
                ? Outcome::WorldChanged : Outcome::Succeeded;
        // The planner's preconditions are not re-derived here. These say
        // that the world has since changed under the goal, which is a fact
        // about the world and not a second opinion about it, so each one
        // says which fact.
        if (!affordance || !affordance->exists)
        {
            m_block = Block::TargetGone;
            return Outcome::TargetUnavailable;
        }
        if (!affordance->executable)
        {
            m_block = Block::TargetUnexecutable;
            return Outcome::TargetUnavailable;
        }
        // The stance this goal set out for, found again by where it is.
        //
        // Comparing the length of the list instead treats every other stance
        // of every other act as part of this goal: the world works out an
        // action's stances afresh whenever the geometry moves, a door
        // swinging two rooms away is enough, and the count comes back
        // different for reasons that have nothing to do with here. The goal
        // was then abandoned mid-walk and chosen again, over and over.
        if (!m_optionKnown)
        {
            m_block = Block::OptionGone;
            return Outcome::TargetUnavailable;
        }
        // Found by where it is on the ground, not by how high it is.
        //
        // A stance on a piece of geometry that moves goes up and down with
        // it, and acting on such a thing is the very thing that moves it --
        // so matching the height means the stance a goal set out for is
        // never there when it arrives, and the goal is abandoned by its own
        // success. The spot you stand on a lift is the same spot at every
        // height the lift has.
        bool stillThere = false;
        for (size_t option = 0; option < affordance->domain.size(); ++option)
            if (affordance->domain[option].at.x == m_optionAt.x
                && affordance->domain[option].at.y == m_optionAt.y)
            {
                m_option = uint32_t(option);
                m_optionAt = affordance->domain[option].at;
                stillThere = true;
                break;
            }
        if (!stillThere)
        {
            // The world no longer offers this act from there. That is news
            // about the world, not a second opinion about the goal.
            m_block = Block::OptionGone;
            return Outcome::TargetUnavailable;
        }
    }

    const semantic::ActorState &actor = world.actor();
    if (actor.region == kNoId)
    {
        m_block = Block::NoRoute;
        return Outcome::Blocked;
    }
    const Vec2 position = { actor.position.x, actor.position.y };
    const Region *here = world.region(actor.region);
    if (!here || !here->exists)
    {
        m_block = Block::NoRoute;
        return Outcome::Blocked;
    }

    // Whether the goal has been reached is asked before which region the body
    // is in, because the answer must not depend on that. A way to look at
    // lies on the boundary between two regions, so arriving at one puts the
    // body on the line: judging arrival by the region it is standing in
    // makes reaching the goal the same event as no longer being there.
    if (m_intent == planner::Intent::Approach)
    {
        const SpatialRelation *way = world.relation(m_relation);
        if (!way || !way->exists)
            return Outcome::TargetUnavailable;
        // Going to look at a way means going to a place this region has
        // beside it, not to a point on the line itself. A point on a
        // boundary belongs to both sides, so driving at one walks the body
        // out of the region it is crossing and the way back in is another
        // point on the same line -- which is a body taking turns at a
        // doorway for as long as the level lasts.
        const Vec2 on = way->gateway.midpoint();
        const Vec2 at = semantic::justInside(here->shape(), way->gateway.from,
                                             way->gateway.to, on,
                                             profile.radius);
        if (semantic::planarDistance(position, at) <= profile.radius * 2)
            return Outcome::Succeeded;
        // A way lies on the boundary of both the regions it joins, so which
        // of them the body is in says nothing about whether it can get to
        // it. If this region's own outline carries the way, walk at it from
        // here; only otherwise is there anywhere to route to.
        if (semantic::pointInPolygon(here->shape(), at))
        {
            if (!steerInside(world, traversal, *here, position, at,
                             profile.radius, command))
            {
                m_block = Block::NoLocalPath;
                return Outcome::Blocked;
            }
            if (losingGround(position, 0, m_remaining, profile.radius))
            {
                m_block = Block::NoProgress;
                return Outcome::Blocked;
            }
            return Outcome::Running;
        }
    }
    else if (m_intent == planner::Intent::ExecuteAffordance
             && size_t(m_option) < affordance->domain.size())
    {
        const semantic::Vec3 &at = affordance->domain[m_option].at;
        if (semantic::planarDistance(position, { at.x, at.y })
            <= profile.radius)
        {
            if (affordance->action == semantic::ActionKind::Use)
            {
                command.address = m_affordance;
                command.option = m_option;
                command.act = true;
                return Outcome::Running;
            }
            command.move = true;
            command.moveToward = at;
            return Outcome::Running;
        }
    }

    if (actor.region != m_destination)
    {
        if (!follow(world, traversal))
        {
            m_block = Block::NoRoute;
            return Outcome::Blocked;
        }
        const SpatialRelation *crossing = world.relation(m_via[m_index]);
        const Region *next = world.region(m_route[m_index + 1]);
        if (!crossing || !next || !next->exists)
        {
            m_block = Block::NoRoute;
            return Outcome::Blocked;
        }
        m_waypoint = next->id;

        // Where on the opening the physics layer established the crossing,
        // so what is driven is what was verified.
        Vec2 opening = crossing->gateway.midpoint();
        Vec2 arrival = next->interior;
        if (!traversal.crossingFor(m_via[m_index], opening, arrival))
        {
            m_block = Block::NoCrossing;
            return Outcome::Blocked;
        }
        if (!steerInside(world, traversal, *here, position, opening, profile.radius,
                         command, &arrival))
        {
            m_block = Block::NoLocalPath;
            return Outcome::Blocked;
        }
        if (losingGround(position, m_via.size() - m_index, m_remaining,
                         profile.radius))
        {
            m_block = Block::NoProgress;
            return Outcome::Blocked;
        }
        return Outcome::Running;
    }

    if (m_intent == planner::Intent::GoTo)
        return Outcome::Succeeded;

    if (m_intent == planner::Intent::Approach)
    {
        const SpatialRelation *way = world.relation(m_relation);
        if (!way || !way->exists)
            return Outcome::TargetUnavailable;
        const Vec2 at = semantic::justInside(here->shape(),
            way->gateway.from, way->gateway.to, way->gateway.midpoint(),
            profile.radius);
        if (!steerInside(world, traversal, *here, position, at, profile.radius,
                         command))
        {
            m_block = Block::NoLocalPath;
            return Outcome::Blocked;
        }
        return Outcome::Running;
    }

    // Done, and this was a stance that is not somewhere to still be. Go
    // where the derivation said there was still somewhere to be, and keep
    // going until the world has finished answering.
    if (m_acted && m_leaveAtOnce)
    {
        if (semantic::planarDistance(position, m_escapeTo) > profile.radius
            && steerInside(world, traversal, *here, position, m_escapeTo,
                           profile.radius, command))
            return Outcome::Running;
        m_leaveAtOnce = false;
    }

    // In the right space; now the act itself, which happens from a pose the
    // world accepts it from and not from the middle of the room.
    const semantic::Vec3 &at = affordance->domain[m_option].at;
    const Vec2 stance = { at.x, at.y };
    if (semantic::planarDistance(position, stance) > profile.radius)
    {
        if (!steerInside(world, traversal, *here, position, stance, profile.radius,
                         command))
        {
            m_block = Block::NoLocalPath;
            return Outcome::Blocked;
        }
        if (losingGround(position, 0, m_remaining, profile.radius))
        {
            m_block = Block::NoProgress;
            return Outcome::Blocked;
        }
        return Outcome::Running;
    }
    if (affordance->action == semantic::ActionKind::Use)
    {
        // Standing in the right place and asking. The world answers by
        // registering an attempt, and this goal ends when it does. If it
        // never does, the engine is refusing the act -- which is a fact
        // about the world, and the same watchdog that answers "this leg is
        // going nowhere" answers it, with nothing of the plan left to shrink.
        if (losingGround(position, 0, 0, profile.radius))
        {
            m_block = Block::NotAccepted;
            return Outcome::Blocked;
        }
        command.address = m_affordance;
        command.option = m_option;
        command.act = true;
        m_acted = true;
        return Outcome::Running;
    }
    // Collecting is done by being there. Keep closing on it until the world
    // says it has happened.
    command.move = true;
    command.moveToward = at;
    return Outcome::Running;
}

const char *outcomeName(Outcome outcome)
{
    switch (outcome)
    {
    case Outcome::Running: return "running";
    case Outcome::Succeeded: return "succeeded";
    case Outcome::WorldChanged: return "world_changed";
    case Outcome::Blocked: return "blocked";
    case Outcome::TargetUnavailable: return "target_unavailable";
    }
    return "unknown";
}

const char *blockName(Block block)
{
    switch (block)
    {
    case Block::None: return "none";
    case Block::NoRoute: return "no_route";
    case Block::NoCrossing: return "no_crossing";
    case Block::NoLocalPath: return "no_local_path";
    case Block::NoProgress: return "no_progress";
    case Block::TargetGone: return "target_gone";
    case Block::TargetUnexecutable: return "target_unexecutable";
    case Block::OptionGone: return "option_gone";
    case Block::NotAccepted: return "not_accepted";
    }
    return "unknown";
}

} // namespace exec
