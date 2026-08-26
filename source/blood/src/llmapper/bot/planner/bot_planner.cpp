#include "bot_planner.h"

#include <algorithm>

namespace planner {

using semantic::Affordance;
using semantic::AffordanceId;
using semantic::Region;
using semantic::RegionId;
using semantic::SemanticWorld;
using semantic::SpatialRelation;
using semantic::kNoId;
using traversal::TraversalModel;

namespace {

// The cheapest candidate so far. `tag` is whatever the caller is choosing
// between -- an affordance, or a way to go and look at.
// Is this thing standing in a way right now? A barrier that is in nothing's
// way is open, and opening it again means shutting it first.
bool standsInAWay(const SemanticWorld &world, semantic::AffordanceId thing)
{
    for (const SpatialRelation &relation : world.relations())
        if (relation.exists && relation.obstruction == thing)
            return true;
    return false;
}

struct Best
{
    RegionId region = kNoId;
    uint32_t tag = kNoId;
    uint32_t option = 0;
    // The way this act was chosen in order to open, when it was chosen for
    // one. Carried so that having tried it can be remembered.
    semantic::RelationId aimedAt = semantic::kNoId;
    int cost = 0;
    bool found = false;

    void offer(RegionId destination, uint32_t subject, int candidate,
               uint32_t which = 0,
               semantic::RelationId about = semantic::kNoId)
    {
        if (found && candidate >= cost)
            return;
        found = true;
        cost = candidate;
        region = destination;
        tag = subject;
        option = which;
        aimedAt = about;
    }
};

} // namespace

Decision choose(const SemanticWorld &world, const TraversalModel &traversal)
{
    Decision decision;
    Diagnosis &diagnosis = decision.diagnosis;
    diagnosis.knownRegions = int(world.liveRegionCount());
    diagnosis.knownAffordances = int(world.affordances().size());
    diagnosis.possibleButUnexecutable = traversal.knownButUnexecutable();

    const RegionId origin = world.actor().region;
    if (origin == kNoId || !world.region(origin))
    {
        diagnosis.reason = StallReason::ActorHasNoRegion;
        return decision;
    }

    std::vector<RegionId> reachable;
    traversal.reachableFrom(origin, reachable);
    diagnosis.reachableRegions = int(reachable.size());
    std::vector<char> isReachable(world.regions().size(), 0);
    for (RegionId id : reachable)
        if (size_t(id) < isReachable.size())
            isReachable[size_t(id)] = 1;

    // 1. a way out that has not been taken. The frontier is a gateway, not a
    //    place: a region can be entered once and still have most of what
    //    leads out of it unknown, and there is no way to be sure a large
    //    space has been seen from the inside except to go where it leads.
    Best frontier;
    for (const traversal::Transition &transition : traversal.transitions())
    {
        if (!transition.executable || transition.to == origin)
            continue;
        const SpatialRelation *relation = world.relation(transition.relation);
        const Region *target = world.region(transition.to);
        if (!relation || !relation->exists || relation->crossed)
            continue;
        if (!target || !target->exists)
            continue;
        ++diagnosis.uncrossedGateways;
        if (target->occupied)
            continue;
        const size_t slot = size_t(transition.to);
        if (slot >= isReachable.size() || !isReachable[slot])
            continue;
        ++diagnosis.unenteredRegions;
        int cost = 0;
        if (traversal.routeCost(origin, transition.to, cost))
            frontier.offer(transition.to, kNoId, cost);
    }

    // 2. something to do, preferring what has never been tried.
    Best untried;
    Best productive;
    for (const Affordance &affordance : world.affordances())
    {
        if (affordance.id == kNoId || !affordance.exists)
            continue;
        if (!affordance.executable || affordance.domain.empty())
        {
            ++diagnosis.affordancesUnreachable;
            continue;
        }
        // The domain is a set of places it can be done from. Which one is
        // used depends on where the actor is and what it can get to, which
        // is this layer's question and not the mapper's.
        size_t chosenOption = affordance.domain.size();
        int best = 0;
        int bestSurvival = 0;
        for (size_t option = 0; option < affordance.domain.size(); ++option)
        {
            const semantic::ExecutionOption &where = affordance.domain[option];
            if (where.region == kNoId)
                continue;
            // Not "can the region be reached" but "can this place in it be
            // stood in, from here". A body with width can be in a region and
            // still be walled off from part of it.
            int cost = 0;
            if (!traversal.optionCost(origin, affordance.id,
                                      uint32_t(option), cost))
                continue;
            // Nearer is better, but only among stances that leave the body
            // somewhere to be afterwards. A stance that is a hundred units
            // closer and inside the thing the act is about to move is not a
            // cheaper way of doing it; it is a different outcome.
            const int survival = int(traversal.survivalOf(affordance.id,
                                                          uint32_t(option)));
            if (chosenOption != affordance.domain.size()
                && survival > bestSurvival)
                continue;
            if (chosenOption == affordance.domain.size()
                || survival < bestSurvival || cost < best)
            {
                chosenOption = option;
                best = cost;
                bestSurvival = survival;
            }
        }
        if (chosenOption == affordance.domain.size())
        {
            ++diagnosis.affordancesUnreachable;
            continue;
        }
        const semantic::RegionId where =
            affordance.domain[chosenOption].region;
        // Nothing is done to a thing that is in the middle of doing
        // something. Its state is not the state the decision would be made
        // about.
        if (affordance.settling)
            continue;
        // Trying a thing to find out what it does is only worth a journey
        // while what it does is unknown. Where the world says this works the
        // same geometry as something already done, its effect is known and
        // there is nothing to find out: pressing it is only worth doing when
        // some other rule actually wants that geometry moved. This is why
        // the second control for one door is not a second discovery -- and
        // why undoing what was just done is not one either.
        if (world.effectKnown(affordance.id))
        {
            ++diagnosis.affordancesAttemptedInertly;
            continue;
        }
        if (affordance.attempts == 0)
            untried.offer(where, affordance.id, best, uint32_t(chosenOption));
        else if (affordance.lastAttemptOpenedWay
                 && !(affordance.barrier
                      && !standsInAWay(world, affordance.id)))
            productive.offer(where, affordance.id, best,
                             uint32_t(chosenOption));
        else
            ++diagnosis.affordancesAttemptedInertly;
    }

    // 3. a way that is known and cannot be gone through because something is
    //    standing in it. The world says what that something is, so there is
    //    no need to go and look: the thing in the way is the thing to do.
    Best shut;
    for (const SpatialRelation &relation : world.relations())
    {
        if (relation.id == kNoId || !relation.exists || relation.crossed)
            continue;
        if (relation.obstruction == kNoId)
            continue;
        // Only while it is actually shut. "Not walked through yet" is not
        // the same as "cannot be walked through": a door that has been
        // opened and not yet used is wide open, and pushing it again shuts
        // it. Standing next to one and pushing it every eighth tick for the
        // rest of the level is what that looks like from outside.
        bool passable = false;
        for (int mode = 0; mode < traversal::kModeCount && !passable; ++mode)
            passable = traversal.derived(relation.id, traversal::Mode(mode))
                && (traversal.executableModes()
                    & traversal::modeBit(traversal::Mode(mode))) != 0;
        if (passable)
            continue;
        const size_t slot = size_t(relation.from);
        if (slot >= isReachable.size() || !isReachable[slot])
            continue;
        const Affordance *thing = world.affordance(relation.obstruction);
        if (!thing || !thing->exists || !thing->executable)
            continue;
        // Not while it is still moving. What is in the way is on its way
        // somewhere, and pressing it again sends it back: this is the whole
        // of why a bot opens and shuts the same door all afternoon. There is
        // no timer here and nothing is being suppressed -- the thing is
        // simply not in a state to be acted on yet, and when it settles it
        // is offered again like anything else.
        if (thing->settling)
            continue;
        // Not if this same act has already been done to open this same way,
        // with this same thing standing in it in this same state. That
        // experiment has been run; running it again asks a question that has
        // been answered. It becomes worth doing again the moment what is in
        // the way changes, or changes state, which is the only thing that
        // could make the answer different.
        if (!world.worthTryingFor(thing->id, relation.id,
                                  world.blockerState(relation.id)))
            continue;
        for (size_t option = 0; option < thing->domain.size(); ++option)
        {
            int cost = 0;
            if (!traversal.optionCost(origin, thing->id, uint32_t(option),
                                      cost))
                continue;
            shut.offer(thing->domain[option].region, thing->id, cost,
                       uint32_t(option), relation.id);
        }
    }

    // 3b. a way that is known and cannot be gone through, with nothing
    //     standing in it -- but something, somewhere, that has been seen to
    //     change it before.
    //
    //     This is the same rule as the one above with the "standing in it"
    //     dropped. A curtain in a doorway and a switch across the room are
    //     the same kind of fact: the way is shut, and acting on that thing
    //     is what moves it. Only the first of the two can be seen by looking
    //     at the doorway, so the second is learned instead -- by having done
    //     it once and watched what moved. Nothing here knows about channels,
    //     switches, or doors; it knows that pressing that changed this.
    Best remembered;
    for (const SpatialRelation &relation : world.relations())
    {
        if (relation.id == kNoId || !relation.exists)
            continue;
        const size_t slot = size_t(relation.from);
        if (slot >= isReachable.size() || !isReachable[slot])
            continue;
        // Shut means this way cannot be gone through, not that the far
        // side is unreachable. A lift at the bottom of its travel is shut
        // even when the floor it serves can be walked to the long way round,
        // and the thing that moves it is still the thing to do.
        bool crossable = false;
        for (int mode = 0; mode < traversal::kModeCount && !crossable; ++mode)
            crossable = traversal.derived(relation.id, traversal::Mode(mode))
                && (traversal.executableModes()
                    & traversal::modeBit(traversal::Mode(mode))) != 0;
        if (crossable)
            continue;
        for (const Affordance &thing : world.affordances())
        {
            if (thing.id == kNoId || !thing.exists || !thing.executable
                || thing.settling)
                continue;
            if (!world.worthTrying(thing.id, relation.id))
                continue;
            // The same experiment memory rule 3a keeps. Knowing that this
            // act moves this way is a reason to do it again while the way
            // is shut; having already done it against this way, with this
            // same thing standing in it in this same state, and watched the
            // way stay shut, is a reason not to.
            if (!world.worthTryingFor(thing.id, relation.id,
                                      world.blockerState(relation.id)))
                continue;
            for (size_t option = 0; option < thing.domain.size(); ++option)
            {
                int cost = 0;
                if (!traversal.optionCost(origin, thing.id, uint32_t(option),
                                          cost))
                    continue;
                remembered.offer(thing.domain[option].region, thing.id,
                                 cost, uint32_t(option), relation.id);
            }
        }
    }

    // 3c. somewhere the body has not stood that can be reached by a plan
    //     which arranges the world on the way.
    //
    //     Every other rule reads the world as it is and asks what is worth
    //     doing in it. This one asks whether there is an order of walking
    //     and acting that opens something up, and offers the first step of
    //     it. That is a different question and it is the only one that finds
    //     a plan whose steps are each individually available but only work
    //     in one order -- which is what a level with two lifts is.
    //
    //     The search is in the traversal layer, over Places and whatever
    //     geometry the plan has had to commit to. Nothing here knows what
    //     the geometry is; it gets back "walk this" or "do that", and both
    //     are things it already knows how to want.
    Best arrangedAct;
    traversal::TraversalModel::PlanStep plan;
    if (traversal.planSomewhereNew(world, origin, plan, &diagnosis.plan))
    {
        ++diagnosis.reconfigureOffered;
        if (plan.act)
            arrangedAct.offer(plan.region, plan.affordance, plan.cost,
                              plan.option);
    }

    // 4. a way that is known and cannot be gone through, with nothing in it
    //    that can be done. What would open it is generally written on it, so
    //    the actor goes and looks -- once.
    Best closed;
    for (const SpatialRelation &relation : world.relations())
    {
        if (relation.id == kNoId || !relation.exists || relation.crossed
            || relation.inspected)
            continue;
        // Going to look at a way means walking to the way. One with no width
        // has nowhere to walk to: it is two spaces overlapping rather than
        // two spaces sharing an edge -- a walkway over a room -- and no
        // amount of standing near it turns it into a doorway. Offering them
        // as somewhere to go spends the run inspecting the fact that a floor
        // has a ceiling.
        if (relation.gateway.width() <= 0)
            continue;
        const size_t slot = size_t(relation.from);
        if (slot >= isReachable.size() || !isReachable[slot])
            continue;
        bool drivable = false;
        for (int mode = 0; mode < traversal::kModeCount && !drivable; ++mode)
            drivable = traversal.derived(relation.id, traversal::Mode(mode))
                && (traversal.executableModes()
                    & traversal::modeBit(traversal::Mode(mode))) != 0;
        if (drivable)
            continue;
        // Only if there is anything to learn by looking.
        //
        // The point of walking over to a way that cannot be gone through is
        // that what would open it is usually written on it, and the sweep
        // picks that up on arrival. If what is on the other side has already
        // been seen, there is nothing on the far side to find out and the
        // walk is purely a walk. On AGTST14 the last three goals of the run
        // were this: two of them to look at boundaries inside a hall the bot
        // had already been through eight times.
        const Region *beyond = world.region(relation.to);
        if (beyond && beyond->observed)
            continue;
        ++diagnosis.uninspectedGateways;
        int cost = 0;
        if (traversal.routeCost(origin, relation.from, cost))
            closed.offer(relation.from, relation.id, cost);
    }
    // Whatever is nearest, of everything worth doing.
    //
    // No ranking. The kinds are kinds, not priorities: opening what is shut,
    // trying what is untried, walking somewhere new and going to look at
    // something are all just things to do, and the one to do is the one you
    // are nearest. Ranking them is what makes a bot walk the length of the
    // level past a switch it will have to come back for.
    struct Candidate { const Best *best; Intent intent; const char *name; };
    const Candidate kinds[] = {
        { &shut,        Intent::ExecuteAffordance, "shut" },
        { &remembered,  Intent::ExecuteAffordance, "remembered" },

        { &untried,     Intent::ExecuteAffordance, "untried" },
        { &frontier,    Intent::GoTo,              "frontier" },
    };
    const Candidate *nearest = nullptr;
    for (const Candidate &kind : kinds)
    {
        if (!kind.best->found)
            continue;
        if (nearest && kind.best->cost >= nearest->best->cost)
            continue;
        nearest = &kind;
    }
    if (nearest)
    {
        decision.why = nearest->name;
        decision.intent = nearest->intent;
        decision.destination = nearest->best->region;
        if (nearest->intent == Intent::ExecuteAffordance)
        {
            decision.affordance = nearest->best->tag;
            decision.option = nearest->best->option;
            decision.relation = nearest->best->aimedAt;
            decision.leaveAtOnce = traversal.escapeFrom(decision.affordance,
                decision.option, decision.escapeTo);
        }
        return decision;
    }

    // Then arranging the world so that somewhere new can be walked to.
    //
    // Not a peer of the others, for the same reason looking at something is
    // not: it is what there is to do when there is nothing left to walk to.
    // Pricing it against walking does not work either -- an act costs almost
    // nothing to perform and a walk costs its whole length, so any plan that
    // presses something undercuts every plan that does not, and the bot ends
    // up riding a lift back and forth instead of stepping off it.
    //
    // Explore what can be reached; when that runs out, rearrange.
    if (arrangedAct.found)
    {
        decision.why = "planned_act";
        decision.intent = Intent::ExecuteAffordance;
        decision.destination = arrangedAct.region;
        decision.affordance = arrangedAct.tag;
        decision.option = arrangedAct.option;
        return decision;
    }

    // And only then, going to look at something.
    //
    // This is the one kind that is not a peer of the others. It accomplishes
    // nothing: it goes and looks at a way that cannot be gone through, on
    // the chance that what would open it is written on it. Let that compete
    // on distance and it wins constantly -- there is always an unreachable
    // ledge nearby -- and the bot spends the level inspecting walls while
    // the map stays shut. AGTST7 runs out of ideas in sixteen seconds.
    if (closed.found)
    {
        decision.why = "closed";
        decision.intent = Intent::Approach;
        decision.destination = closed.region;
        decision.relation = closed.tag;
        return decision;
    }

    // There is no fifth rule any more.
    //
    // There used to be: an act that had moved the world once was offered
    // again whenever nothing else was going. That is doing something because
    // it worked before rather than because it would help now, and it is the
    // whole of why the bot walks back across the level to press a switch it
    // has already pressed. Two of them, forty-three times each, in one run.
    //
    // Whether an act would help now is a question the model can answer, and
    // rule 3b above is where it answers it: this way is shut, that thing has
    // been seen to change this way, so go and change it. An act that moved
    // something once and is not known to move anything that is in the way
    // now has nothing to recommend it, and "the planner had nothing better"
    // is not a reason -- having nothing better is what having nothing to do
    // looks like, and saying so is more useful than walking in circles.
    diagnosis.reason = StallReason::NoKnownAction;
    return decision;
}

const char *stallName(StallReason reason)
{
    switch (reason)
    {
    case StallReason::None: return "none";
    case StallReason::ActorHasNoRegion: return "ACTOR_HAS_NO_REGION";
    case StallReason::NoKnownAction: return "NO_KNOWN_ACTION";
    }
    return "unknown";
}

const char *intentName(Intent intent)
{
    switch (intent)
    {
    case Intent::None: return "none";
    case Intent::GoTo: return "go_to";
    case Intent::Approach: return "approach";
    case Intent::ExecuteAffordance: return "execute_affordance";
    }
    return "unknown";
}

} // namespace planner
