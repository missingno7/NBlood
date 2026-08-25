//-------------------------------------------------------------------------
// LLMapper Blood bot: orchestration only.
//
//   Blood -> WorldAdapter -> SemanticWorld -.
//                                             >- TraversalModel -> Planner
//            CalebPhysics --------------------'                       |
//                                                                 Executor
//
// Four separate authorities: what is there, what the actor can do with it,
// what the bot wants, and how to carry it out. Every arrow is a file
// boundary and points one way. This file owns none of those jobs; it runs
// them in order, records what happened, and decides when a run is over.
//-------------------------------------------------------------------------
#include "bot.h"

#include <chrono>

#include <cstdio>
#include <string>
#include <vector>

#include "blood/blood_world_adapter.h"
#include "blood/caleb_physics.h"
#include "debug/bot_debug.h"
#include "debug/bot_overlay.h"
#include "exec/bot_executor.h"
#include "planner/bot_planner.h"
#include "semantic/semantic_world.h"
#include "traversal/traversal_model.h"

#include "../../blood.h"
#include "../../common_game.h"
#include "../../demo.h"
#include "../../globals.h"
#include "../../levels.h"
#include "../../player.h"
#include "../../view.h"

struct LLMapperBot::Impl
{
    FILE *telemetry = nullptr;
    FILE *trajectory = nullptr;
    std::string telemetryPath = "llmapper-bot.ndjson";
    std::string trajectoryPath = "llmapper-bot-trajectory.ndjson";
    std::string demoPath = "llmapper-bot.dem";
    // No limit unless one is asked for. A time limit is a thing the test
    // harness wants so a stuck run does not sit there all afternoon; it is
    // not a thing the game wants. Defaulting it to five minutes meant that
    // anyone who ran the bot the obvious way -- without the flag -- watched
    // it stop dead partway through any level that takes longer than that,
    // and AGTST18 takes twelve and a half minutes. Nothing said so; the run
    // just ended. tools/botrun.sh passes one explicitly, as it should.
    int runtimeLimitSeconds = 0;
    std::string result;
    std::string failureReason;

    bloodmap::WorldAdapter adapter;
    bloodmap::CalebPhysics physics;
    semantic::SemanticWorld world;
    traversal::TraversalModel traversal;
    planner::Decision decision;
    exec::Executor executor;

    GINPUT issued = {};
    bloodmap::AdapterCounters work;
    int lastTrajectoryFrame = -1;
    bool stallReported = false;
    bool reportedBody = false;
    semantic::RegionId reportedWaypoint = semantic::kNoId;
    semantic::Vec2 reportedAim;
    std::vector<uint64_t> reportedAffordances;
    bool modesConfigured = false;

    int gameTime() const { return (gFrame * kTicsPerFrame) / kTicsPerSec; }

    void event(const char *name, const char *detail = "")
    {
        if (!telemetry)
            return;
        std::fprintf(telemetry,
            "{\"type\":\"event\",\"game_time\":%d,\"tick\":%d,"
            "\"event\":\"%s\",\"detail\":\"%s\"}\n",
            gameTime(), gFrame * kTicsPerFrame, name, detail ? detail : "");
    }

    void openFiles()
    {
        if (!telemetry && !telemetryPath.empty())
        {
            telemetry = std::fopen(telemetryPath.c_str(), "wb");
            // A line at a time, flushed one at a time, is a system call per
            // event and a disk write per system call. The run ends by
            // writing its summary, and that flushes; a run that ends any
            // other way is a crash, and the last few lines of a crash are
            // not worth a syscall on every line of every run.
            if (telemetry)
                std::setvbuf(telemetry, nullptr, _IOFBF, 1 << 20);
        }
        if (!trajectory && !trajectoryPath.empty())
            trajectory = std::fopen(trajectoryPath.c_str(), "wb");
    }

    void reportStall(const planner::Diagnosis &diagnosis)
    {
        if (stallReported)
            return;
        stallReported = true;
        char detail[256];
        std::snprintf(detail, sizeof(detail),
            "reason=%s known_regions=%d reachable=%d uncrossed_gateways=%d "
            "unentered_regions=%d uninspected_gateways=%d affordances=%d "
            "unreachable=%d attempted_without_change=%d "
            "possible_but_unexecutable=%d",
            planner::stallName(diagnosis.reason), diagnosis.knownRegions,
            diagnosis.reachableRegions, diagnosis.uncrossedGateways,
            diagnosis.unenteredRegions, diagnosis.uninspectedGateways,
            diagnosis.knownAffordances,
            diagnosis.affordancesUnreachable,
            diagnosis.affordancesAttemptedInertly,
            diagnosis.possibleButUnexecutable);
        event("no_known_action", detail);
        result = "NO_KNOWN_ACTION";
        failureReason = "the model offers nothing further";
        gQuitGame = true;
    }

    // The trace the layers are meant to be debugged through: what the
    // mapper made of the world, and what the physics layer made of that.
    void dumpWorld()
    {
        char provenance[224];
        char detail[1024];
        for (const semantic::Region &region : world.regions())
        {
            if (!region.exists)
                continue;
            botdebug::describeRegion(adapter, region.id, provenance,
                                     sizeof(provenance));
            const semantic::Vec3 anchor = region.anchor();
            char outline[768];
            botdebug::describeFootprint(region, outline, sizeof(outline));
            std::snprintf(detail, sizeof(detail),
                "%s at=(%d,%d,%d) headroom=%d clearance=%s pieces=%d %s", provenance,
                anchor.x, anchor.y, anchor.z,
                region.clearanceAt({ anchor.x, anchor.y }),
                semantic::clearanceName(region.clearance),
                traversal.piecesOf(region.id), outline);
            event("region_mapped", detail);
        }
        for (const bloodmap::RejectedSpace &space : adapter.rejectedSpaces())
        {
            botdebug::describeRejection(space, detail, sizeof(detail));
            event("space_rejected", detail);
        }
        for (const semantic::SpatialRelation &relation : world.relations())
        {
            if (!relation.exists)
                continue;
            std::snprintf(detail, sizeof(detail),
                "relation=%u from=%u to=%u step=%d clearance=%d width=%d "
                "at=(%d,%d)-(%d,%d) blocked=%d walk=%d drop=%d why=%s "
                "shut_by=%d tag=%d stance=(%d,%d) aimed=(%d,%d) ended=(%d,%d) "
                "wanted=%d found=%d blood_container=%d stances=%d",
                unsigned(relation.id), unsigned(relation.from),
                unsigned(relation.to), relation.verticalStep,
                relation.clearance, relation.gateway.width(),
                relation.gateway.from.x, relation.gateway.from.y,
                relation.gateway.to.x, relation.gateway.to.y,
                relation.blocked ? 1 : 0,
                traversal.derived(relation.id, traversal::Mode::Walk) ? 1 : 0,
                traversal.derived(relation.id, traversal::Mode::Drop) ? 1 : 0,
                bloodmap::CalebPhysics::refusalName(
                    physics.refusalFor(relation.id)),
                relation.obstruction == semantic::kNoId
                    ? -1 : int(relation.obstruction),
                adapter.obstructionTag(relation.id),
                physics.evidenceFor(relation.id).at.x,
                physics.evidenceFor(relation.id).at.y,
                physics.evidenceFor(relation.id).aimed.x,
                physics.evidenceFor(relation.id).aimed.y,
                physics.evidenceFor(relation.id).ended.x,
                physics.evidenceFor(relation.id).ended.y,
                physics.evidenceFor(relation.id).wanted,
                physics.evidenceFor(relation.id).found,
                physics.evidenceFor(relation.id).container,
                traversal.stancesIn(relation.from));
            event("relation_mapped", detail);
        }
    }

    // Affordances change as the world does: an action may stop being
    // offered, or become reachable from somewhere new.
    // How much the world currently offers. Comparing this across an action
    // is what tells the model whether doing it achieved anything, without
    // anyone having to understand what the action was.
    int possibilities()
    {
        std::vector<semantic::RegionId> reachable;
        traversal.reachableFrom(world.actor().region, reachable);
        int total = int(reachable.size());
        for (const semantic::Affordance &affordance : world.affordances())
            if (affordance.exists && affordance.executable
                && !affordance.domain.empty())
                ++total;
        return total;
    }

    // Run the mapper without driving anything, so the overlay shows the
    // same world the bot would see while a person walks the level.
    void refreshModel()
    {
        Stopwatch observing;
        const semantic::WorldDelta delta = adapter.observe();
        world.apply(delta);
        physics.refresh();
        { const double spent = observing.stop();
          worstObserveMs = std::max(worstObserveMs, spent);
          totalObserveMs += spent; }
        Stopwatch deriving;
        traversal.update(world, physics);
        { const double spent = deriving.stop();
          worstTraversalMs = std::max(worstTraversalMs, spent);
          totalTraversalMs += spent; }
        work = adapter.counters();
    }

    void reportAffordances()
    {
        char provenance[192];
        char detail[288];
        for (const semantic::Affordance &affordance : world.affordances())
        {
            const uint64_t state = (uint64_t(affordance.exists) << 40)
                | (uint64_t(affordance.executable) << 41)
                | (uint64_t(affordance.observed) << 42)
                | uint64_t(affordance.domain.size());
            if (size_t(affordance.id) < reportedAffordances.size()
                && reportedAffordances[size_t(affordance.id)] == state)
                continue;
            if (reportedAffordances.size() <= size_t(affordance.id))
                reportedAffordances.resize(size_t(affordance.id) + 1, ~0ull);
            reportedAffordances[size_t(affordance.id)] = state;
            botdebug::describeAffordance(adapter, affordance.id, provenance,
                                         sizeof(provenance));
            std::snprintf(detail, sizeof(detail),
                "%s kind=%s exists=%d observed=%d executable=%d "
                "from_regions=%u first=%d at=(%d,%d,%d)",
                provenance, semantic::actionName(affordance.action),
                affordance.exists ? 1 : 0,
                affordance.observed ? 1 : 0, affordance.executable ? 1 : 0,
                unsigned(affordance.domain.size()),
                affordance.domain.empty()
                    ? -1 : int(affordance.domain.front().region),
                affordance.domain.empty() ? 0 : affordance.domain.front().at.x,
                affordance.domain.empty() ? 0 : affordance.domain.front().at.y,
                affordance.domain.empty()
                    ? 0 : affordance.domain.front().at.z);
            event("affordance_mapped", detail);
        }
    }

    void reportWaypoint()
    {
        if (executor.aim() != reportedAim)
        {
            reportedAim = executor.aim();
            char steering[160];
            std::snprintf(steering, sizeof(steering),
                "aim=(%d,%d) legs=%u from=(%d,%d) region=%d",
                reportedAim.x, reportedAim.y, unsigned(executor.legs()),
                world.actor().position.x, world.actor().position.y,
                world.actor().region == semantic::kNoId
                    ? -1 : int(world.actor().region));
            event("steering", steering);
        }
        if (executor.rerouted())
        {
            char chain[256];
            int used = 0;
            for (semantic::RegionId step : executor.route())
                used += std::snprintf(chain + used, sizeof(chain) - size_t(used),
                                      used ? ">%d" : "%d", int(step));
            char detail[320];
            std::snprintf(detail, sizeof(detail), "at=%d index=%u route=%s",
                world.actor().region == semantic::kNoId
                    ? -1 : int(world.actor().region),
                unsigned(executor.routeIndex()), chain);
            event("rerouted", detail);
        }
        if (executor.waypoint() == reportedWaypoint)
            return;
        reportedWaypoint = executor.waypoint();
        char provenance[192];
        char detail[256];
        botdebug::describeRegion(adapter, reportedWaypoint, provenance,
                                 sizeof(provenance));
        std::snprintf(detail, sizeof(detail), "route_left=%u %s",
            unsigned(executor.routeLength()), provenance);
        event("waypoint", detail);
    }

    // Ways out of where the actor is standing that the physics layer agreed
    // to and this model then withheld. When a bot says there is nothing to
    // do while standing next to a door, this is where the answer is.
    bool reportedDrops = false;
    void reportDroppedWays()
    {
        if (reportedDrops)
            return;
        reportedDrops = true;
        const semantic::RegionId here = world.actor().region;
        char detail[224];
        for (const traversal::DroppedCrossing &dropped
             : traversal.droppedCrossings())
        {
            const semantic::SpatialRelation *way =
                world.relation(dropped.relation);
            if (!way || way->from != here)
                continue;
            std::snprintf(detail, sizeof(detail),
                "relation=%u mode=%s to=%u leaves=%d arrives=%d "
                "from_stances=%d to_stances=%d from_pieces=%d to_pieces=%d",
                unsigned(dropped.relation),
                traversal::modeName(traversal::Mode(dropped.mode)),
                unsigned(way->to), dropped.leaves, dropped.arrives,
                dropped.fromStances, dropped.toStances,
                traversal.piecesOf(way->from), traversal.piecesOf(way->to));
            event("way_withheld", detail);
        }
    }

    // The edge of the world the bot believes in: every way out of somewhere
    // it can reach that leads somewhere it cannot, and what each of them was
    // refused for. If the bot is wrong about where it can go, the mistake is
    // one of these.
    // Is anything moving where the body is standing, or right next to it?
    // Walls arrive from the room next door, so the room next door counts.
    bool groundMovingNearby() const
    {
        const semantic::RegionId here = world.actor().region;
        if (here == semantic::kNoId)
            return false;
        const semantic::Region *space = world.region(here);
        if (space && space->exists && space->hazard.shifting)
            return true;
        for (const semantic::SpatialRelation &way : world.relations())
        {
            if (!way.exists || way.from != here)
                continue;
            const semantic::Region *beside = world.region(way.to);
            if (beside && beside->exists && beside->hazard.shifting)
                return true;
        }
        return false;
    }

    // Which ways the body can actually go through, one entry per relation.
    std::vector<char> crossableNow() const
    {
        std::vector<char> out(world.relations().size(), 0);
        for (const semantic::SpatialRelation &way : world.relations())
        {
            if (way.id == semantic::kNoId
                || size_t(way.id) >= out.size())
                continue;
            for (int mode = 0; mode < traversal::kModeCount; ++mode)
                if (traversal.derived(way.id, traversal::Mode(mode))
                    && (traversal.executableModes()
                        & traversal::modeBit(traversal::Mode(mode))) != 0)
                {
                    out[size_t(way.id)] = 1;
                    break;
                }
        }
        return out;
    }
    std::vector<char> crossableAtAttempt;

    int lastGroundFlags = -2;
    bool reportedEdge = false;
    void reportEdgeOfTheKnown()
    {
        if (reportedEdge)
            return;
        reportedEdge = true;
        std::vector<semantic::RegionId> reachable;
        traversal.reachableFrom(world.actor().region, reachable);
        std::vector<char> known(world.regions().size(), 0);
        for (semantic::RegionId id : reachable)
            if (size_t(id) < known.size())
                known[size_t(id)] = 1;
        int live = 0;
        for (const semantic::Region &region : world.regions())
            if (region.exists)
                ++live;
        char detail[224];
        std::snprintf(detail, sizeof(detail),
            "reachable=%u of %d regions", unsigned(reachable.size()), live);
        event("known_world", detail);
        for (const semantic::SpatialRelation &way : world.relations())
        {
            if (!way.exists || way.id == semantic::kNoId)
                continue;
            if (size_t(way.from) >= known.size() || !known[size_t(way.from)])
                continue;
            if (size_t(way.to) < known.size() && known[size_t(way.to)])
                continue;
            botdebug::describeRelation(adapter, physics, traversal, way,
                                       detail, sizeof(detail));
            event("edge_of_known", detail);
        }
        // Every way the physics agreed the body can walk that the model then
        // failed to give a Place at either end. Each one is a way out the
        // planner will never be offered, so each one is a false fact about
        // the world rather than a shortfall of effort.
        for (const traversal::DroppedCrossing &lost :
                 traversal.droppedCrossings())
        {
            if (lost.fromStances == 0 || lost.toStances == 0)
                continue;   // no space either side: honestly nowhere to go
            const semantic::SpatialRelation *way =
                world.relation(lost.relation);
            if (!way)
                continue;
            const semantic::Region *leaving = world.region(way->from);
            const semantic::Region *entering = world.region(way->to);
            std::snprintf(detail, sizeof(detail),
                "relation=%d from_region=%d to_region=%d leaves=%d arrives=%d"
                " step=%d depart=(%d,%d) depart_in=%d arrive=(%d,%d)"
                " arrive_in=%d",
                int(lost.relation), int(way->from), int(way->to),
                lost.leaves, lost.arrives, way->verticalStep,
                lost.departure.x, lost.departure.y,
                leaving ? int(semantic::pointInPolygon(leaving->shape(),
                                                   lost.departure)) : -1,
                lost.arrival.x, lost.arrival.y,
                entering ? int(semantic::pointInPolygon(entering->shape(),
                                                   lost.arrival)) : -1);
            event("orphan_crossing", detail);
        }
        // Every act the world says can be done that the bot never got to,
        // and what became of each pose the engine offered it from. An act
        // with poses but no Place is not out of reach -- it is being
        // withheld by this layer, which is a different thing entirely.
        for (const semantic::Affordance &thing : world.affordances())
        {
            if (thing.id == semantic::kNoId || !thing.exists
                || !thing.executable || thing.attempts > 0)
                continue;
            char poses[160];
            int written = 0;
            for (size_t option = 0; option < thing.domain.size()
                     && written < int(sizeof(poses)) - 24; ++option)
            {
                int cost = 0;
                const bool routable = traversal.optionCost(
                    world.actor().region, thing.id, uint32_t(option), cost);
                written += std::snprintf(poses + written,
                    sizeof(poses) - size_t(written), "%s%d:r%d/w%d/g%d",
                    option ? "," : "", int(thing.domain[option].region),
                    routable ? 1 : 0,
                    traversal.optionVerdict(thing.id, uint32_t(option)),
                    int(traversal.piecesOf(thing.domain[option].region)));
            }
            if (thing.domain.empty())
                std::snprintf(poses, sizeof(poses), "none");
            char at[160];
            int said = 0;
            for (size_t option = 0; option < thing.domain.size()
                     && said < int(sizeof(at)) - 20; ++option)
                said += std::snprintf(at + said, sizeof(at) - size_t(said),
                    "%s(%d,%d,%d)", option ? "," : "",
                    thing.domain[option].at.x, thing.domain[option].at.y,
                    thing.domain[option].at.z);
            if (thing.domain.empty())
                std::snprintf(at, sizeof(at), "none");
            int tried = 0;
            int inReach = 0;
            int accepted = 0;
            adapter.domainAudit(thing.id, tried, inReach, accepted);
            std::snprintf(detail, sizeof(detail),
                "affordance=%d options=%u tried=%d in_reach=%d accepted=%d"
                " where=%s at=%s",
                int(thing.id), unsigned(thing.domain.size()), tried, inReach,
                accepted, poses, at);
            event("act_withheld", detail);
        }
        // What each act has been seen to change, and whether the bot could
        // get to it to do so again.
        for (const semantic::Affordance &thing : world.affordances())
        {
            if (thing.id == semantic::kNoId || thing.affects.empty())
                continue;
            char ways[140];
            int said = 0;
            for (size_t i = 0; i < thing.affects.size()
                     && said < int(sizeof(ways)) - 8; ++i)
                said += std::snprintf(ways + said,
                    sizeof(ways) - size_t(said), "%s%d", i ? "," : "",
                    int(thing.affects[i]));
            bool routable = false;
            for (size_t option = 0; option < thing.domain.size(); ++option)
            {
                int cost = 0;
                if (traversal.optionCost(world.actor().region, thing.id,
                                         uint32_t(option), cost))
                    routable = true;
            }
            std::snprintf(detail, sizeof(detail),
                "affordance=%d exists=%d executable=%d settling=%d"
                " routable=%d changes=%u ways=%s",
                int(thing.id), thing.exists ? 1 : 0,
                thing.executable ? 1 : 0, thing.settling ? 1 : 0,
                routable ? 1 : 0, unsigned(thing.affects.size()), ways);
            event("act_causal", detail);
        }
    }

    // Everything the engine says about the things that stand in ways, and
    // what the model says about the ways they stand in. Emitted whenever any
    // of it changes, so what a Use actually does can be read off rather than
    // reasoned about.
    std::map<unsigned, std::string> blockerState;
    void reportBlockers()
    {
        char detail[320];
        for (const semantic::Affordance &thing : world.affordances())
        {
            if (!thing.exists || !thing.barrier)
                continue;
            bloodmap::WorldAdapter::ObjectState object;
            if (!adapter.objectState(thing.id, object))
                continue;
            int shuts = 0;
            int walkable = 0;
            for (const semantic::SpatialRelation &way : world.relations())
            {
                if (!way.exists)
                    continue;
                if (way.obstruction == thing.id)
                    ++shuts;
                if (way.obstruction == thing.id
                    && traversal.derived(way.id, traversal::Mode::Walk))
                    ++walkable;
            }
            std::snprintf(detail, sizeof(detail),
                "affordance=%u blood_%s=%d solid=%d state=%d tx=%d "
                "shuts=%d of_which_walkable=%d tried=%d opened_way=%d",
                unsigned(thing.id), botdebug::containerName(object.tag),
                object.id, object.solid ? 1 : 0, object.state,
                object.channel, shuts, walkable, thing.attempts,
                thing.lastAttemptOpenedWay ? 1 : 0);
            std::string &known = blockerState[unsigned(thing.id)];
            if (known == detail)
                continue;
            known = detail;
            event("blocker_state", detail);
        }
    }

    void announce(const planner::Decision &chosen)
    {
        char provenance[192];
        char detail[288];
        if (chosen.intent == planner::Intent::ExecuteAffordance)
        {
            botdebug::describeAffordance(adapter, chosen.affordance,
                                         provenance, sizeof(provenance));
            std::snprintf(detail, sizeof(detail), "intent=%s %s",
                planner::intentName(chosen.intent), provenance);
        }
        else if (chosen.intent == planner::Intent::Approach)
        {
            botdebug::describeRegion(adapter, chosen.destination, provenance,
                                     sizeof(provenance));
            std::snprintf(detail, sizeof(detail),
                "intent=%s relation=%u %s",
                planner::intentName(chosen.intent),
                unsigned(chosen.relation), provenance);
        }
        else
        {
            botdebug::describeRegion(adapter, chosen.destination, provenance,
                                     sizeof(provenance));
            std::snprintf(detail, sizeof(detail), "intent=%s %s",
                planner::intentName(chosen.intent), provenance);
        }
        {
            char labelled[288];
            std::snprintf(labelled, sizeof(labelled), "rule=%s %s",
                          chosen.why, detail);
            event("goal_chosen", labelled);
        }
    }

    // What one decision costs the game. The bot runs on the game's own
    // thread, so a tick it spends thinking is a tick the game does not draw.
    double worstTickMs = 0.0;
    double totalTickMs = 0.0;
    int ticksTimed = 0;
    int worstTickAt = 0;
    int slowTicks = 0;
    double worstObserveMs = 0.0;
    double totalObserveMs = 0.0;
    double totalTraversalMs = 0.0;
    double totalExecMs = 0.0;
    double worstTraversalMs = 0.0;
    double worstDecideMs = 0.0;

    struct Stopwatch
    {
        std::chrono::steady_clock::time_point began
            = std::chrono::steady_clock::now();
        double stop()
        {
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - began).count();
        }
    };

    GINPUT tick()
    {
        const auto began = std::chrono::steady_clock::now();
        const GINPUT answer = decide();
        const double spent = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - began).count();
        if (spent > worstTickMs)
        {
            worstTickMs = spent;
            worstTickAt = gFrame;
        }
        if (spent > 33.0)
            ++slowTicks;
        // A decision that took longer than several frames is a hitch the
        // player sees. Say when, and how long, so it can be chased.
        if (spent > 100.0)
        {
            char detail[96];
            std::snprintf(detail, sizeof(detail), "cost_ms=%.0f", spent);
            event("slow_decision", detail);
        }
        totalTickMs += spent;
        ++ticksTimed;
        return answer;
    }

    GINPUT decide()
    {
        if (!modesConfigured)
        {
            // What this bot can drive, which is a different fact from what
            // Caleb can physically do. Adding an executor changes this line
            // and nothing else in the model.
            //
            // Dropping needs no executor of its own: walking off the edge is
            // the whole of it, and the engine does the falling. What makes it
            // safe to select is that the physics layer only derives a drop
            // the body walks away from.
            traversal.setExecutableModes(
                traversal::modeBit(traversal::Mode::Walk)
                | traversal::modeBit(traversal::Mode::Drop));
            modesConfigured = true;
        }
        // World, then physics, then what the two of them make possible.
        Stopwatch observing;
        const semantic::WorldDelta delta = adapter.observe();
        world.apply(delta);
        physics.refresh();
        { const double spent = observing.stop();
          worstObserveMs = std::max(worstObserveMs, spent);
          totalObserveMs += spent; }
        Stopwatch deriving;
        traversal.update(world, physics);
        { const double spent = deriving.stop();
          worstTraversalMs = std::max(worstTraversalMs, spent);
          totalTraversalMs += spent; }
        // The engine may start tearing a finished level down before the run
        // is written out, so keep the tally here rather than reading it back
        // from an adapter that has already been reset for the next one.
        const int buildsBefore = work.terrainBuilds;
        work = adapter.counters();
        if (work.terrainBuilds != buildsBefore)
        {
            char detail[128];
            std::snprintf(detail, sizeof(detail),
                "faces=%d clusters=%d regions=%d relations=%u obstructed_seams=%d",
                work.facesIn, work.clusters, work.regionsBuilt,
                unsigned(world.liveRelationCount()),
                bloodmap::obstructedSeams());
            event("world_mapped", detail);
            {
                char moving[180];
                int said = 0;
                int count = 0;
                for (const semantic::Region &region : world.regions())
                {
                    if (!region.exists || !region.hazard.shifting)
                        continue;
                    ++count;
                    if (said < int(sizeof(moving)) - 12)
                        said += std::snprintf(moving + said,
                            sizeof(moving) - size_t(said), "%s%d",
                            said ? "," : "", int(region.id));
                }
                if (count > 0)
                {
                    std::snprintf(detail, sizeof(detail),
                        "count=%d regions=%s", count, moving);
                    event("ground_shifting", detail);
                }
            }
            for (int i = 0; i < bloodmap::obstructedSeamCount(); ++i)
            {
                int x1, y1, x2, y2, spriteId, owner, behind;
                bloodmap::obstructedSeamAt(i, x1, y1, x2, y2, spriteId,
                                           owner, behind);
                char line[160];
                std::snprintf(line, sizeof(line),
                    "at=(%d,%d)-(%d,%d) blood_sprite=%d between=%d,%d",
                    x1, y1, x2, y2, spriteId, owner, behind);
                event("seam_obstructed", line);
            }
            dumpWorld();
        }
        if (delta.resolvedAffordance != semantic::kNoId)
        {
            char provenance[192];
            botdebug::describeAffordance(adapter, delta.resolvedAffordance,
                                         provenance, sizeof(provenance));
            event("action_delivered", provenance);
            world.beginAttempt(delta.resolvedAffordance, possibilities());
            crossableAtAttempt = crossableNow();
        }
        // Whether the act opened anything up is measured once the world has
        // stopped moving, which is the right question for that and the wrong
        // one for whether to stand still (see below).
        if (world.attemptOpen() && !world.settling())
        {
            const semantic::AffordanceId id = world.openAttempt();
            // What that act did to the world, said in the only terms the
            // planner cares about: which ways can be gone through now that
            // could not before, and which can no longer be. Both directions
            // count -- a switch that shuts a corridor is the thing to press
            // to open it again -- and nothing else counts at all, which is
            // what keeps a door from being credited with the ledges in the
            // rooms it happens to border.
            const std::vector<char> after = crossableNow();
            const size_t common = std::min(after.size(),
                                           crossableAtAttempt.size());
            for (size_t index = 0; index < common; ++index)
                if (after[index] != crossableAtAttempt[index])
                    world.noteAffects(id, semantic::RelationId(index));
            crossableAtAttempt.clear();
            world.closeAttempt(possibilities());
            const semantic::Affordance *affordance = world.affordance(id);
            event("action_settled", affordance
                && affordance->lastAttemptOpenedWay
                    ? "opened_way=1" : "opened_way=0");
        }
        if (!reportedBody && physics.profile().revision)
        {
            reportedBody = true;
            const traversal::ActorProfile &shape = physics.profile();
            char detail[224];
            std::snprintf(detail, sizeof(detail),
                "radius=%d stand=%d crouch=%d step_up=%d walk_speed=%d "
                "jump=%d gravity=%d",
                shape.radius, shape.standHeight, shape.crouchHeight,
                shape.stepUp, shape.walkSpeed, shape.jumpImpulse,
                shape.gravity);
            event("actor_measured", detail);
        }
        if (!bloodmap::motionTrace().empty())
        {
            for (const std::string &line : bloodmap::motionTrace())
                event("motion_trace", line.c_str());
            bloodmap::clearMotionTrace();
        }
        reportAffordances();
        reportBlockers();
        if (!world.actor().alive || !result.empty())
            return GINPUT();
        // Wait for what this bot just did to finish, and only that. The
        // world moving somewhere else is not its business: a lift running in
        // a room it has never seen would otherwise stand it still for as
        // long as the lift takes, again and again, all run.
        //
        // And waiting means standing still. If the body is somewhere else
        // than it was while this bot is issuing no input at all, then it is
        // not standing still and this is not waiting -- something in the
        // world has hold of it, and the one thing that must not happen next
        // is another tick of nothing.
        //
        // On AGTST18 the bot pressed a switch and then gave no input for a
        // hundred and sixty ticks while the floor it was standing on carried
        // it six hundred units west into a wall, at the same second on every
        // run. Nothing in the geometry moved and nothing was blocked; the
        // bot simply called being carried "waiting" and held still for it.
        // Whatever is doing the carrying -- a floor that pans, a sector on a
        // path, a wall arriving -- the body's own position says it is
        // happening, and that is the whole of the test.
        // ... and not while the floor is carrying the body somewhere.
        //
        // Comparing where the body was a tick ago does not tell these apart
        // from ordinary walking: Blood keeps a body's momentum for a good
        // while after the input stops, so an honest wait right after a walk
        // looks exactly like being dragged. The engine has the fact outright
        // -- a sector with a pan velocity that is switched on moves whatever
        // stands on it -- so ask that instead of guessing from motion.
        const semantic::Region *underfoot =
            world.region(world.actor().region);
        const bool carried = underfoot && underfoot->exists
            && underfoot->hazard.carrying;
        {
            const int flags = underfoot && underfoot->exists
                ? (underfoot->hazard.harmful ? 4 : 0)
                    | (underfoot->hazard.shifting ? 2 : 0)
                    | (underfoot->hazard.carrying ? 1 : 0)
                : -1;
            if (flags != lastGroundFlags)
            {
                lastGroundFlags = flags;
                char note[96];
                std::snprintf(note, sizeof(note),
                    "region=%d harmful=%d shifting=%d carrying=%d",
                    world.actor().region == semantic::kNoId
                        ? -1 : int(world.actor().region),
                    flags > 0 && (flags & 4) ? 1 : 0,
                    flags > 0 && (flags & 2) ? 1 : 0,
                    flags > 0 && (flags & 1) ? 1 : 0);
                event("standing_on", note);
            }
        }
        // Nothing waits here any more.
        //
        // There used to be a stop: having acted, stand still until the world
        // settles. Every version of it was wrong in the same way. Waiting on
        // the world waits on anything at all moving anywhere; waiting on the
        // thing acted on waits as long as that thing takes, and a Blood
        // switch animates for the whole time the sector it drives is
        // travelling. On AGTST18 that is a hundred and seventy ticks of
        // standing perfectly still next to a switch that has just sent a
        // wall along the corridor the bot is standing in, and it ends the
        // same way every run.
        //
        // What the stop was really for was to stop the bot acting on a thing
        // in the middle of it doing something -- pressing a door again while
        // it is still swinging, which sends it back. That is a question
        // about the thing, and it is now answered where it belongs: the
        // planner does not offer an act whose object is still moving. There
        // is nothing left for standing still to accomplish, and standing
        // still was never free.

        semantic::MotorCommand command;
        if (executor.active())
        {
            const exec::Outcome outcome = executor.tick(world, traversal,
                physics.profile(), command);
            switch (outcome)
            {
            case exec::Outcome::Running:
                reportWaypoint();
                return adapter.toInput(command);
            case exec::Outcome::WorldChanged:
                // The act landed and the thing is still moving. Ordinarily
                // there is nothing to do but let it finish, and letting it
                // finish is worth doing: deciding what to do next before the
                // door has opened is deciding in a world that does not have
                // the doorway in it yet.
                //
                // Unless the moving geometry is here. A body waiting politely
                // in a corridor that a wall is travelling down is carried
                // into whatever is behind it and squashed, and it does not
                // have to be this room that is moving -- the wall arrives
                // from next door. So: wait, unless something adjacent is on
                // the move, in which case do anything else at all.
                if (!carried && !groundMovingNearby())
                    return GINPUT();
                break;
            case exec::Outcome::Succeeded:
            case exec::Outcome::Blocked:
            case exec::Outcome::TargetUnavailable:
            {
                // Going to look at a way that cannot be gone through is
                // answered by whatever happened: standing at it, or finding
                // there is no getting to it. Either way there is nothing
                // further to learn by setting off for it again.
                if (executor.intent() == planner::Intent::Approach)
                    world.noteInspected(executor.inspecting());
                char why[96];
                std::snprintf(why, sizeof(why),
                    "%s block=%s leg=%d region=%d",
                    exec::outcomeName(outcome),
                    exec::blockName(executor.block()),
                    executor.leg() == semantic::kNoId
                        ? -1 : int(executor.leg()),
                    world.actor().region == semantic::kNoId
                        ? -1 : int(world.actor().region));
                event("goal_finished", why);
                if (executor.block() == exec::Block::NoLocalPath
                    || executor.block() == exec::Block::NoProgress
                    || executor.block() == exec::Block::NoRoute)
                {
                    // The derivation offered something the executor could
                    // not drive. That disagreement is a fact about this
                    // actor in this world, so the model is told; it forgets
                    // again the moment either of them changes.
                    // What the executor learned is which leg it could not
                    // drive. Where it happened to be headed is a different
                    // fact, and refusing that instead leaves every other
                    // route through the same stuck leg on offer.
                    if (executor.leg() != semantic::kNoId)
                        traversal.refuseCrossing(executor.leg());
                    else if (executor.intent()
                        == planner::Intent::ExecuteAffordance)
                        traversal.refuseOption(executor.affordance(),
                                               executor.option());
                    else if (executor.intent() == planner::Intent::GoTo
                        && executor.destination() != semantic::kNoId
                        && executor.destination() != world.actor().region)
                        traversal.refuseRegion(executor.destination());
                    // Going to look at a way and not managing it is already
                    // written down: the way has been inspected, and will not
                    // be offered again. Refusing the region as well says the
                    // body cannot get to where it is standing, and then
                    // nothing anywhere can be routed to.
                    char where[256];
                    botdebug::describeLocalFailure(executor.lastFailure(),
                                                   where, sizeof(where));
                    event("undrivable", where);
                }
            }
                executor.clear();
                reportedWaypoint = semantic::kNoId;
                break;
            }
        }

        Stopwatch choosing;
        decision = planner::choose(world, traversal);
        worstDecideMs = std::max(worstDecideMs, choosing.stop());
        if (decision.intent == planner::Intent::None)
        {
            // Having nothing to do is only a conclusion once the mapper has
            // finished answering, and once the world has stopped moving.
            // This is not a retry: no action is being repeated, and nothing
            // is being waited out on a clock.
            //
            // The second half matters as much as the first. A corridor with
            // a wall travelling down it offers no way through while the wall
            // is in it, and asking "is there anything left to do" at that
            // instant gets the answer "no" about a moment rather than about
            // the level. On AGTST18 the bot ends its run sealed in a room
            // whose door is halfway open.
            if (!world.establishing() && !world.settling())
            {
                reportStall(decision.diagnosis);
                reportDroppedWays();
                reportEdgeOfTheKnown();
            }
            return GINPUT();
        }
        stallReported = false;
        announce(decision);
        executor.begin(decision, world);
        if (executor.tick(world, traversal, physics.profile(), command)
            == exec::Outcome::Running)
            return adapter.toInput(command);
        return GINPUT();
    }

    void sampleTrajectory()
    {
        // One flush per tick rather than one per event. What is at risk is
        // the last tick of a run that dies without closing its files, which
        // is a fair trade for not making a system call every time the bot
        // has something to say.
        if (telemetry)
            std::fflush(telemetry);
        if (!trajectory || lastTrajectoryFrame == gFrame)
            return;
        lastTrajectoryFrame = gFrame;
        botdebug::sampleTrajectory(trajectory, gameTime(),
                                   gFrame * kTicsPerFrame, issued);
    }

    void close(const char *reason)
    {
        if (result.empty())
        {
            result = reason && *reason ? "RUNTIME_ERROR" : "STOPPED";
            failureReason = reason && *reason ? reason : "run stopped";
        }
        if (telemetry)
        {
            // What the model still could not account for, however the run
            // ended. A run that stops on the clock has the same false facts
            // in it as one that stops for want of anything to do.
            reportEdgeOfTheKnown();
            int observed = 0;
            for (const semantic::Region &region : world.regions())
                if (region.exists && region.observed)
                    ++observed;
            std::fprintf(telemetry,
                "{\"type\":\"summary\",\"result\":\"%s\","
                "\"failure_reason\":\"%s\",\"game_time\":%d,"
                "\"total_sectors\":%d,\"faces\":%d,\"clusters\":%d,"
                "\"regions\":%u,\"observed_regions\":%d,\"relations\":%u,"
                "\"transitions\":%u,\"unexecutable_transitions\":%d,"
                "\"orphaned_crossings\":%d,"
                "\"uncrossed_relations\":%u,\"local_maps\":%d,"
                "\"places\":%u,\"split_regions\":%u,\"refusals\":%d,"
                "\"affordances\":%u,\"terrain_builds\":%d,"
                "\"traversal_evaluations\":%d,\"domain_queries\":%d,"
                "\"physical_probes\":%d,\"worst_tick_ms\":%.1f,"
                "\"mean_tick_ms\":%.2f,\"worst_observe_ms\":%.1f,"
                "\"worst_traversal_ms\":%.1f,\"worst_decide_ms\":%.1f,"
                "\"worst_standing_ms\":%.1f,\"worst_crossings_ms\":%.1f,"
                "\"worst_pieces_ms\":%.1f,\"worst_tick_at\":%d,"
                "\"slow_ticks\":%d,\"mean_observe_ms\":%.2f,"
                "\"mean_traversal_ms\":%.2f,\"mean_standing_ms\":%.2f,"
                "\"mean_crossings_ms\":%.2f,\"mean_pieces_ms\":%.2f,\"derivations\":%d,\"standing_maps\":%d,\"ticks_timed\":%d}\n",
                result.c_str(), failureReason.c_str(), gameTime(), numsectors,
                work.facesIn, work.clusters,
                unsigned(world.liveRegionCount()), observed,
                unsigned(world.liveRelationCount()),
                unsigned(traversal.transitions().size()),
                traversal.knownButUnexecutable(),
                traversal.orphanedCrossings(),
                unsigned(world.uncrossedRelationCount()),
                traversal.navigator().builds(),
                unsigned(traversal.places()),
                unsigned(traversal.splitRegions()), traversal.refusals(),
                unsigned(world.affordances().size()), work.terrainBuilds,
                traversal.totalEvaluations(), work.domainQueries,
                bloodmap::probeCount(), worstTickMs,
                ticksTimed ? totalTickMs / ticksTimed : 0.0,
                worstObserveMs, worstTraversalMs, worstDecideMs,
                traversal.worstStandingMs(), traversal.worstCrossingsMs(),
                traversal.worstPiecesMs(), worstTickAt, slowTicks,
                ticksTimed ? totalObserveMs / ticksTimed : 0.0,
                ticksTimed ? totalTraversalMs / ticksTimed : 0.0,
                ticksTimed ? traversal.totalStandingMs() / ticksTimed : 0.0,
                ticksTimed ? traversal.totalCrossingsMs() / ticksTimed : 0.0,
                ticksTimed ? traversal.totalPiecesMs() / ticksTimed : 0.0,
                traversal.derivations(), traversal.standingBuilds(), ticksTimed);
            std::fclose(telemetry);
            telemetry = nullptr;
        }
        if (trajectory)
        {
            std::fclose(trajectory);
            trajectory = nullptr;
        }
        if (gDemo.at0)
            gDemo.Close();
    }
};

LLMapperBot::LLMapperBot()
    : m_impl(new Impl), m_enabled(false), m_fast(true), m_visible(false),
      m_debug(false)
{
}

LLMapperBot::~LLMapperBot()
{
    Finish("RUNTIME_ERROR");
    delete m_impl;
}

void LLMapperBot::Enable(const char *telemetry, const char *trajectory,
                         const char *demo)
{
    m_enabled = true;
    if (telemetry && *telemetry)
        m_impl->telemetryPath = telemetry;
    if (trajectory && *trajectory)
        m_impl->trajectoryPath = trajectory;
    if (demo && *demo)
        m_impl->demoPath = demo;
}

void LLMapperBot::ConfigureTimeout(int seconds)
{
    if (seconds > 0)
        m_impl->runtimeLimitSeconds = seconds;
}

void LLMapperBot::SetFast(bool fast) { m_fast = fast; }

void LLMapperBot::SetVisible(bool visible)
{
    m_visible = visible;
    if (visible)
        m_fast = false;
}

void LLMapperBot::SetDebugOverlay(bool debug) { m_debug = debug; }

void LLMapperBot::DrawDebugOverlay(int cameraX, int cameraY, int cameraZ,
                                   fix16_t cameraAngle,
                                   fix16_t cameraHorizon)
{
    if (!m_debug || !gGameStarted)
        return;
    botdebug::Camera camera;
    camera.x = cameraX;
    camera.y = cameraY;
    camera.z = cameraZ;
    camera.angle = cameraAngle;
    camera.horizon = cameraHorizon;
    botdebug::drawWorld(camera, m_impl->world, m_impl->traversal);
}

void LLMapperBot::PrepareLaunch()
{
    if (!m_enabled)
        return;
    m_impl->openFiles();
    m_impl->adapter.reset();
    m_impl->event("run_started", "architecture=layered_semantic");
    if (!gDemo.at0 && !gDemo.at1 && !gDemo.Create(m_impl->demoPath.c_str()))
    {
        m_impl->result = "RUNTIME_ERROR";
        m_impl->failureReason = "could not create demo file";
        gQuitGame = true;
    }
}

GINPUT LLMapperBot::GetInput()
{
    if (!m_enabled)
        return GINPUT();
    m_impl->issued = m_impl->tick();
    return m_impl->issued;
}

void LLMapperBot::OnFrame()
{
    if (!gGameStarted)
        return;
    if (!m_enabled)
    {
        if (m_debug)
            m_impl->refreshModel();
        return;
    }
    m_impl->sampleTrajectory();
    if ((gGameOptions.uGameFlags & kGameFlagContinuing)
        && m_impl->result.empty())
        OnLevelExit(kLevelExitNormal);
    if (m_impl->world.liveRegionCount() && !m_impl->world.actor().alive
        && m_impl->result.empty())
    {
        m_impl->result = "DIED";
        m_impl->failureReason = "player health reached zero";
        m_impl->event("failure", m_impl->failureReason.c_str());
        gQuitGame = true;
    }
    if (m_impl->result.empty() && m_impl->runtimeLimitSeconds > 0
        && m_impl->gameTime() >= m_impl->runtimeLimitSeconds)
    {
        m_impl->result = "TIMEOUT";
        m_impl->failureReason = "external run time limit reached";
        m_impl->event("failure", m_impl->failureReason.c_str());
        gQuitGame = true;
    }
}

void LLMapperBot::OnActionResolved(int hit, int target, int, bool accepted,
                                   int)
{
    if (!m_enabled)
        return;
    m_impl->adapter.noteEngineAction(hit, target, accepted);
}

void LLMapperBot::OnLevelExit(int exitType)
{
    if (!m_enabled || !m_impl->result.empty())
        return;
    m_impl->result = "COMPLETED";
    char detail[48];
    std::snprintf(detail, sizeof(detail), "exit_type=%d", exitType);
    m_impl->event("level_completed", detail);
    gQuitGame = true;
}

void LLMapperBot::DrawStatus()
{
    if (!m_enabled || !m_visible || !gGameStarted)
        return;
    char line[128];
    std::snprintf(line, sizeof(line), "BOT %d:%02d r%u/%u %s",
        m_impl->gameTime() / 60, m_impl->gameTime() % 60,
        unsigned(m_impl->world.actor().region),
        unsigned(m_impl->world.liveRegionCount()),
        planner::intentName(m_impl->executor.intent()));
    viewDrawText(3, line, 2, 4, -128, 0, 0, true, 256);
}

void LLMapperBot::Finish(const char *reason)
{
    if (m_enabled && (m_impl->telemetry || m_impl->trajectory))
        m_impl->close(reason);
}

LLMapperBot gLLMapperBot;
