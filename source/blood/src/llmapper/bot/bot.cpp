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
#include <set>
#include <tuple>
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
    uint64_t reportedBody = 0;
    std::set<semantic::GeometryId> reportedGeometry;
    std::set<semantic::RelationId> reportedCondition;
    std::set<semantic::AffordanceId> reportedCommands;
    std::set<semantic::GeometryId> reportedWiring;
    std::string lastSealing;
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

    // The physics layer has to be able to put a piece of geometry into a
    // configuration to ask about it, and only the adapter knows which engine
    // handle each GeometryId came from.
    void connectGeometry()
    {
        physics.resolveGeometryWith(
            [this](semantic::GeometryId id) {
                return adapter.geometryTag(id);
            });
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
            "possible_but_unexecutable=%d reconfig=%d/%d/%d/%d",
            planner::stallName(diagnosis.reason), diagnosis.knownRegions,
            diagnosis.reachableRegions, diagnosis.uncrossedGateways,
            diagnosis.unenteredRegions, diagnosis.uninspectedGateways,
            diagnosis.knownAffordances,
            diagnosis.affordancesUnreachable,
            diagnosis.affordancesAttemptedInertly,
            diagnosis.possibleButUnexecutable,
            diagnosis.reconfigureLooked, diagnosis.reconfigureWouldOpen,
            diagnosis.reconfigureNoMover, diagnosis.reconfigureOffered);
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
            // Where it can be done from is part of what changed about it.
            // Counting only how many stances there are misses a lift whose
            // stances moved from one floor to another without changing in
            // number, which is exactly what a lift does.
            uint64_t where = 0;
            for (const semantic::ExecutionOption &option : affordance.domain)
                where = where * 1315423911u + uint64_t(option.region) + 1;
            const uint64_t state = (uint64_t(affordance.exists) << 40)
                | (uint64_t(affordance.executable) << 41)
                | (uint64_t(affordance.observed) << 42)
                | uint64_t(affordance.domain.size())
                | (where << 43);
            if (size_t(affordance.id) < reportedAffordances.size()
                && reportedAffordances[size_t(affordance.id)] == state)
                continue;
            if (reportedAffordances.size() <= size_t(affordance.id))
                reportedAffordances.resize(size_t(affordance.id) + 1, ~0ull);
            reportedAffordances[size_t(affordance.id)] = state;
            botdebug::describeAffordance(adapter, affordance.id, provenance,
                                         sizeof(provenance));
            char spots[120] = "none";
            {
                // Region and piece of free space together. Which region an
                // act can be done from does not say whether the body can
                // get to it: a region is not always one piece.
                std::set<std::pair<unsigned, unsigned>> regions;
                for (size_t option = 0; option < affordance.domain.size();
                     ++option)
                    regions.insert({
                        unsigned(affordance.domain[option].region),
                        unsigned(traversal.placeOfOption(affordance.id,
                                                     uint32_t(option))) });
                int said = 0;
                for (const auto &where : regions)
                {
                    if (said >= int(sizeof(spots)) - 12)
                        break;
                    said += std::snprintf(spots + said,
                        sizeof(spots) - size_t(said),
                        said ? ",r%u@%u" : "r%u@%u", where.first,
                        where.second);
                }
            }
            // Why each stance is or is not somewhere the body can be sent:
            // 0 placed, 1 never asked, 2 no room to stand, 3 no piece of
            // free space found for it.
            char verdicts[48];
            {
                int tally[4] = { 0, 0, 0, 0 };
                for (size_t option = 0; option < affordance.domain.size();
                     ++option)
                {
                    const int said = traversal.optionVerdict(affordance.id,
                                                             uint32_t(option));
                    if (said >= 0 && said < 4)
                        ++tally[said];
                }
                std::snprintf(verdicts, sizeof(verdicts),
                    "%d/%d/%d/%d", tally[0], tally[1], tally[2], tally[3]);
            }
            // How many stances leave the body somewhere to be once what
            // the act commands has arrived: remain/escape/unknown/unsafe.
            char lives[48];
            {
                int tally[4] = { 0, 0, 0, 0 };
                for (size_t option = 0; option < affordance.domain.size();
                     ++option)
                {
                    const int said = int(traversal.survivalOf(affordance.id,
                                                          uint32_t(option)));
                    if (said >= 0 && said < 4)
                        ++tally[said];
                }
                std::snprintf(lives, sizeof(lives), "%d/%d/%d/%d",
                    tally[0], tally[1], tally[2], tally[3]);
            }
            std::snprintf(detail, sizeof(detail),
                "%s kind=%s exists=%d observed=%d executable=%d "
                "why=%s lives=%s from_regions=%u first=%d in=[%s] at=(%d,%d,%d)",
                provenance, semantic::actionName(affordance.action),
                affordance.exists ? 1 : 0,
                affordance.observed ? 1 : 0, affordance.executable ? 1 : 0,
                verdicts, lives,
                unsigned(affordance.domain.size()),
                affordance.domain.empty()
                    ? -1 : int(affordance.domain.front().region),
                spots,
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
    semantic::RelationId aimedAtWay = semantic::kNoId;
    uint64_t blockerAtAttempt = 0;
    bool movingAtAttempt = false;
    int attemptFrame = 0;
    std::map<semantic::GeometryId, int> setOffAt;
    std::set<semantic::GeometryId> wasTravelling;
    std::vector<std::tuple<semantic::GeometryId, uint32_t, bool>>
        configurationAtAttempt;

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
        {
            // Where one Region came out as more than one piece of free
            // space, and which piece each way out of it belongs to. A route
            // that walks into a region and then cannot cross the way it went
            // there for is a disagreement about exactly this.
            for (const semantic::Region &region : world.regions())
            {
                if (!region.exists)
                    continue;
                std::vector<size_t> pieces;
                traversal.placesOf(region.id, pieces);
                if (pieces.size() < 2)
                    continue;
                int said = std::snprintf(detail, sizeof(detail),
                    "region=%u pieces=%u ways=", unsigned(region.id),
                    unsigned(pieces.size()));
                for (const semantic::SpatialRelation &way : world.relations())
                {
                    if (!way.exists || way.from != region.id
                        || said >= int(sizeof(detail)) - 16)
                        continue;
                    size_t leaves = 0;
                    size_t arrives = 0;
                    if (!traversal.placesAcross(way.id, leaves, arrives))
                        continue;
                    said += std::snprintf(detail + said,
                        sizeof(detail) - size_t(said), "%u@%u->%u,",
                        unsigned(way.id), unsigned(leaves),
                        unsigned(arrives));
                }
                event("place_split", detail);
            }
            // Which regions keep having their free space thrown away.
            std::vector<std::pair<int, int>> churn;
            for (const nav::Navigator::Entry &entry
                     : traversal.navigatorFor().entries())
                churn.push_back({ entry.rebuilds, int(entry.id) });
            std::sort(churn.rbegin(), churn.rend());
            for (size_t i = 0; i < churn.size() && i < 10; ++i)
            {
                if (churn[i].first < 2)
                    break;
                std::snprintf(detail, sizeof(detail),
                    "region=%d rebuilds=%d", churn[i].second, churn[i].first);
                event("map_churn", detail);
            }
        }
        // Every way out of the room the body is actually in, and what this
        // layer decided about each. When the answer is "nowhere to go", this
        // is the list that has to explain it.
        for (const semantic::SpatialRelation &way : world.relations())
        {
            if (!way.exists || way.id == semantic::kNoId
                || way.from != world.actor().region)
                continue;
            const traversal::TraversalModel::Verdict said =
                traversal.verdictFor(way.id, traversal::Mode::Walk);
            std::snprintf(detail, sizeof(detail),
                "relation=%u to=%u width=%d blocked=%d valid=%d exists=%d"
                " walk=%d executable=%d refused=%d leaves=%d arrives=%d"
                " from_stances=%d to_stances=%d routed=%d",
                unsigned(way.id), unsigned(way.to), way.gateway.width(),
                way.blocked ? 1 : 0, said.valid ? 1 : 0,
                said.exists ? 1 : 0, said.possible ? 1 : 0,
                said.executable ? 1 : 0, said.refused ? 1 : 0,
                said.leaves, said.arrives, int(said.from), int(said.to),
                said.routed ? 1 : 0);
            event("way_out_audit", detail);
        }
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
        if (!chosen.diagnosis.plan.empty())
        {
            char steps[224];
            int said = std::snprintf(steps, sizeof(steps), "from=%d on=%d ",
                world.actor().region == semantic::kNoId
                    ? -1 : int(world.actor().region),
                world.actor().supportedBy == semantic::kNoId
                    ? -1 : int(world.actor().supportedBy));
            for (const semantic::StatefulGeometry &piece : world.geometry())
            {
                if (said >= int(sizeof(steps)) - 24)
                    break;
                said += std::snprintf(steps + said,
                    sizeof(steps) - size_t(said), "G%u=%u%s ",
                    unsigned(piece.id), piece.state,
                    piece.moving ? "*" : "");
            }
            for (const traversal::TraversalModel::PlanStep &step
                     : chosen.diagnosis.plan)
            {
                if (said >= int(sizeof(steps)) - 28)
                    break;
                said += step.act
                    ? std::snprintf(steps + said, sizeof(steps) - size_t(said),
                        "| act %u@r%u ", unsigned(step.affordance),
                        unsigned(step.region))
                    : std::snprintf(steps + said, sizeof(steps) - size_t(said),
                        "| walk %u ", unsigned(step.crossing));
            }
            event("plan", steps);
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
            attemptFrame = gFrame;
            // Was anything it works still travelling when this was asked?
            //
            // These doors do not accept a use until they have finished, so a
            // press made a tick too early is a press the world drops. What
            // came of it is not "this act does nothing" -- it is that the
            // act was never made, and writing it down as an answer is how
            // the bot decides a way is shut that is merely busy.
            movingAtAttempt = false;
            if (const semantic::Affordance *doing =
                    world.affordance(world.openAttempt()))
                for (semantic::GeometryId which : doing->commands)
                {
                    const semantic::StatefulGeometry *piece =
                        world.geometryOf(which);
                    movingAtAttempt = movingAtAttempt
                        || (piece && piece->moving);
                }
            // Which way this act was chosen to open, and how what stands in
            // it stands at this moment, so that an act which changes nothing
            // is known to have changed nothing.
            aimedAtWay = executor.inspecting();
            if (aimedAtWay != semantic::kNoId)
            {
                bloodmap::WorldAdapter::ObjectState blocker;
                const semantic::SpatialRelation *way =
                    world.relation(aimedAtWay);
                uint64_t stamp = 0;
                if (way && way->obstruction != semantic::kNoId
                    && adapter.objectState(way->obstruction, blocker))
                    stamp = (uint64_t(uint32_t(blocker.state)) << 32)
                        ^ uint64_t(uint32_t(blocker.busy))
                        ^ (uint64_t(uint32_t(blocker.x)) << 8);
                world.noteBlockerState(aimedAtWay, stamp);
                blockerAtAttempt = stamp;
            }
            configurationAtAttempt.clear();
            for (const semantic::StatefulGeometry &piece : world.geometry())
                configurationAtAttempt.push_back(
                    { piece.id, piece.state, piece.moving });
        }
        // Whether the act opened anything up is measured once the world has
        // stopped moving, which is the right question for that and the wrong
        // one for whether to stand still (see below).
        // Whatever started moving, or finished somewhere else, while this
        // act was in the air is what the act moves.
        //
        // Watched every tick rather than only when the act settles: geometry
        // takes time to get going, and an act judged only at the end of its
        // settling gets superseded by the next one and its effect is never
        // attributed to anything. Nothing here asks what the geometry is.
        if (world.attemptOpen() && !configurationAtAttempt.empty())
        {
            const semantic::AffordanceId acting = world.openAttempt();
            for (const auto &was : configurationAtAttempt)
            {
                const semantic::StatefulGeometry *now =
                    world.geometryOf(std::get<0>(was));
                if (!now)
                    continue;
                const bool started = now->moving && !std::get<2>(was);
                // A thing already travelling was going to arrive whatever
                // happened next, so crediting its arrival to whichever act
                // was in flight is false: picking up an ammo box gets
                // recorded as the thing that works a lift, and that then
                // decides what the pickup is allowed to do and from where.
                //
                // Told apart by when the thing set off, not by whether it is
                // moving now. Blood resolves a use inside the tick it
                // happens, so something this act started is already moving
                // when the attempt is written down -- asking "was it still"
                // rejects the act's own doing and leaves AGTST8 at two
                // sectors. Asking "did it set off after I asked" separates
                // them exactly. Something never seen to set off is left as it
                // was: unknown is not evidence either way.
                auto began = setOffAt.find(std::get<0>(was));
                const bool mine = began == setOffAt.end()
                    || began->second >= attemptFrame;
                const bool arrived = now->state != std::get<1>(was) && mine;
                if (!started && !arrived)
                    continue;
                if (!world.moves(acting, std::get<0>(was)))
                {
                    char note[128];
                    std::snprintf(note, sizeof(note),
                        "affordance=%u moves=%u started=%d arrived=%d",
                        unsigned(acting), unsigned(std::get<0>(was)),
                        started ? 1 : 0, arrived ? 1 : 0);
                    event("geometry_moved_by", note);
                }
                world.noteMoves(acting, std::get<0>(was));
            }
        }
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
            configurationAtAttempt.clear();
            // Did it open the way it was chosen to open? If not, that is
            // an experiment run and answered.
            if (aimedAtWay != semantic::kNoId)
            {
                const std::vector<char> now = crossableNow();
                const bool opened = size_t(aimedAtWay) < now.size()
                    && now[size_t(aimedAtWay)];
                if (!opened && !movingAtAttempt)
                    world.noteFruitless(id, aimedAtWay, blockerAtAttempt);
                aimedAtWay = semantic::kNoId;
            }
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
        if (physics.profile().revision != reportedBody)
        {
            reportedBody = physics.profile().revision;
            const traversal::ActorProfile &shape = physics.profile();
            char detail[224];
            std::snprintf(detail, sizeof(detail),
                "radius=%d stand=%d crouch=%d step_up=%d walk_speed=%d "
                "jump=%d gravity=%d posture=%d life=%d",
                shape.radius, shape.standHeight, shape.crouchHeight,
                shape.stepUp, shape.walkSpeed, shape.jumpImpulse,
                shape.gravity, shape.posture, shape.lifeMode);
            event("actor_measured", detail);
        }
        // Every piece of geometry that rests in more than one configuration,
        // said once, as heights. Nothing here is called a door or a lift.
        for (const semantic::StatefulGeometry &piece : world.geometry())
        {
            if (reportedGeometry.count(piece.id))
                continue;
            reportedGeometry.insert(piece.id);
            char detail[224];
            int said = std::snprintf(detail, sizeof(detail),
                "geometry=%u states=%u at=", unsigned(piece.id),
                unsigned(piece.configurations.size()));
            for (size_t which = 0; which < piece.configurations.size()
                     && said < int(sizeof(detail)) - 32; ++which)
                said += std::snprintf(detail + said,
                    sizeof(detail) - size_t(said), "%s[%u]support=%d,ceiling=%d",
                    which ? " " : "", unsigned(which),
                    piece.configurations[which].supportZ,
                    piece.configurations[which].ceilingZ);
            event("geometry_states", detail);
        }
        // When each thing set off, so that what an act did can be told from
        // what was already happening. Watched every tick because it is the
        // only place the answer exists: by the time an act is settled the
        // world has long since replied.
        for (const semantic::StatefulGeometry &piece : world.geometry())
        {
            const bool was = wasTravelling.count(piece.id) != 0;
            if (piece.moving && !was)
                setOffAt[piece.id] = gFrame;
            if (piece.moving)
                wasTravelling.insert(piece.id);
            else
                wasTravelling.erase(piece.id);
        }
        for (const semantic::StatefulGeometry &piece : world.geometry())
        {
            if (reportedWiring.count(piece.id))
                continue;
            reportedWiring.insert(piece.id);
            char detail[96];
            std::snprintf(detail, sizeof(detail),
                "geometry=%u listens=%d described=%d", unsigned(piece.id),
                adapter.listensOn(piece.id),
                piece.configurationsDiffer ? 1 : 0);
            event("geometry_wiring", detail);
        }
        // What the world's own wiring says each act works, and whether the
        // model can describe where that geometry goes.
        for (const semantic::Affordance &thing : world.affordances())
        {
            if (thing.id == semantic::kNoId || thing.commands.empty()
                || reportedCommands.count(thing.id))
                continue;
            reportedCommands.insert(thing.id);
            char detail[224];
            int said = std::snprintf(detail, sizeof(detail),
                "affordance=%u works=", unsigned(thing.id));
            for (size_t i = 0; i < thing.commands.size()
                     && said < int(sizeof(detail)) - 16; ++i)
            {
                const semantic::StatefulGeometry *piece =
                    world.geometryOf(thing.commands[i]);
                said += std::snprintf(detail + said,
                    sizeof(detail) - size_t(said), "%s%u%s", i ? "," : "",
                    unsigned(thing.commands[i]),
                    piece && piece->configurationsDiffer ? "" : "?");
            }
            event("act_commands", detail);
        }
        {
            char detail[224];
            int said = std::snprintf(detail, sizeof(detail),
                "looked=%d skipped=%d sealed=", traversal.sealingLooked(),
                traversal.sealingSkipped());
            for (const auto &one : traversal.sealedActs())
            {
                if (said >= int(sizeof(detail)) - 16)
                    break;
                said += std::snprintf(detail + said,
                    sizeof(detail) - size_t(said), "%u@r%u,",
                    unsigned(one.first), unsigned(one.second));
            }
            if (lastSealing != std::string(detail))
            {
                lastSealing = detail;
                event("seal_check", detail);
                for (const auto &said : traversal.sealingAudit())
                {
                    std::snprintf(detail, sizeof(detail),
                        "affordance=%u room=%u ways=%d open_now=%d"
                        " open_after=%d", unsigned(said.thing),
                        unsigned(said.region), said.ways,
                        said.openNow ? 1 : 0, said.openAfter ? 1 : 0);
                    event("seal_one", detail);
                }
            }
        }
        // Every way whose walkability depends on a configuration, and which
        // configurations allow it. Derived by asking the engine with the
        // geometry posed each way -- not declared, and not named.
        for (const semantic::SpatialRelation &way : world.relations())
        {
            if (!way.exists || way.id == semantic::kNoId)
                continue;
            const semantic::GeometryId on = traversal.conditionOf(way.id);
            if (on == semantic::kNoId || reportedCondition.count(way.id))
                continue;
            const semantic::StatefulGeometry *piece = world.geometryOf(on);
            if (!piece)
                continue;
            reportedCondition.insert(way.id);
            char detail[224];
            int said = std::snprintf(detail, sizeof(detail),
                "relation=%u from=%u to=%u geometry=%u walk_in=",
                unsigned(way.id), unsigned(way.from), unsigned(way.to),
                unsigned(on));
            for (uint32_t which = 0;
                 which < uint32_t(piece->configurations.size())
                     && said < int(sizeof(detail)) - 12; ++which)
                said += std::snprintf(detail + said,
                    sizeof(detail) - size_t(said), "%s%u:%d",
                    which ? "," : "", which,
                    traversal.possibleInConfiguration(way.id, which,
                        traversal::Mode::Walk) ? 1 : 0);
            event("conditional_way", detail);
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
                char why[160];
                std::snprintf(why, sizeof(why),
                    "%s block=%s leg=%d region=%d dest=%d at_place=%u"
                    " act=%d opt=%u opt_place=%u places=%u",
                    exec::outcomeName(outcome),
                    exec::blockName(executor.block()),
                    executor.leg() == semantic::kNoId
                        ? -1 : int(executor.leg()),
                    world.actor().region == semantic::kNoId
                        ? -1 : int(world.actor().region),
                    executor.destination() == semantic::kNoId
                        ? -1 : int(executor.destination()),
                    unsigned(traversal.actorPlace()),
                    executor.affordance() == semantic::kNoId
                        ? -1 : int(executor.affordance()),
                    unsigned(executor.option()),
                    unsigned(traversal.placeOfOption(executor.affordance(),
                                                     executor.option())),
                    unsigned(traversal.places()));
                event("goal_finished", why);
                if (executor.block() == exec::Block::NoProgress)
                {
                    // Where it was, where it was heading, and the whole of
                    // the route it thought it was walking.
                    char stuck[240];
                    int said = std::snprintf(stuck, sizeof(stuck),
                        "at=(%d,%d) aim=(%d,%d) left=%d legs=%u path=",
                        world.actor().position.x, world.actor().position.y,
                        executor.aim().x, executor.aim().y,
                        executor.remaining(), unsigned(executor.legs()));
                    for (const semantic::Vec2 &point : executor.path())
                    {
                        if (said >= int(sizeof(stuck)) - 24)
                            break;
                        said += std::snprintf(stuck + said,
                            sizeof(stuck) - size_t(said), "(%d,%d)",
                            point.x, point.y);
                    }
                    event("went_nowhere", stuck);
                }
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
                "\"stale_by_crossing\":%d,\"stale_by_body\":%d,"
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
                traversal.staleByCrossing(), traversal.staleByBody(),
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
    m_impl->connectGeometry();
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
