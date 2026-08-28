//-------------------------------------------------------------------------
// The physics authority.
//
// One question, asked of the engine: given what the actor physically is
// right now, can it carry out this mode across this piece of world? The
// answer never comes from comparing a height difference to a number -- the
// numbers in the profile are there to be reported, not to decide.
//
// Picking up Jumping Boots changes what this object says. It changes nothing
// about the world, which is the point of the split.
//-------------------------------------------------------------------------
#pragma once

#include <functional>
#include <vector>

#include "../traversal/traversal_model.h"
#include "blood_physics.h"

namespace bloodmap {

class CalebPhysics : public traversal::PhysicsOracle
{
public:
    // Re-read the live player. The profile revision moves only when
    // something about the body actually changed.
    void refresh();

    const traversal::ActorProfile &profile() const override
    {
        return m_profile;
    }

    bool canStand(const semantic::Region &region,
                  const semantic::Vec2 &at) const override;

    bool canTraverse(traversal::Mode mode,
                     const semantic::SpatialRelation &relation,
                     const semantic::Region &from,
                     const semantic::Region &to,
                     const semantic::Vec2 *startFrom,
                     semantic::Vec2 &crossing,
                     semantic::Vec2 &arrival,
                     semantic::Vec2 &departure) const override;

    int queries() const { return m_queries; }

    // Why a crossing was refused. Diagnostic only: a relation the world
    // plainly offers and the model will not walk should be able to say
    // which of its own tests said no.
    enum class Refusal
    {
        None,
        NotOffered,      // blocked, or one side does not exist
        NoWidth,         // stacked surfaces, nothing to walk across
        NoRoom,          // no room for the body one width in on this side
        NoLanding,       // no room for the body one width in on the far side
        NoStance,        // the world does not hold the body up where it starts
        OpeningUnreached,// could not walk to the opening
        LandingUnreached,// could not walk through it
        OutsideTarget,   // ended up somewhere that is not the far side
        WrongSupport,    // ended up on something else holding it up
    };
    Refusal refusalFor(semantic::RelationId relation) const;
    static const char *refusalName(Refusal refusal);

    // The numbers behind the last refusal of one relation, so a wrong
    // verdict can be read rather than guessed at.
    struct Evidence
    {
        semantic::Vec2 at;      // where the stance was tried
        semantic::Vec2 aimed;   // the point past the opening it was pushed at
        semantic::Vec2 ended;   // where the engine left the body
        int wanted = 0;         // the surface the region says is there
        int found = 0;          // the surface the engine reported
        int container = -1;     // the engine's own answer for where that is
        bool located = false;
    };
    const Evidence &evidenceFor(semantic::RelationId relation) const;

    // What the engine said was standing in this way, in its own encoding,
    // or -1. A way that cannot be walked because a thing is in it is a
    // different fact from a way that cannot be walked because of its shape,
    // and only the first one has something that can be done about it.
    int obstacleFor(semantic::RelationId relation) const;

private:
    bool walkAcross(const semantic::SpatialRelation &relation,
                    const semantic::Region &from, const semantic::Region &to,
                    const semantic::Vec2 *startFrom,
                    semantic::Vec2 &crossing,
                    semantic::Vec2 &arrival,
                    semantic::Vec2 &departure) const;
    bool dropAcross(const semantic::SpatialRelation &relation,
                    const semantic::Region &from, const semantic::Region &to,
                    const semantic::Vec2 *startFrom,
                    semantic::Vec2 &crossing,
                    semantic::Vec2 &arrival,
                    semantic::Vec2 &departure) const;

    void note(semantic::RelationId relation, Refusal refusal) const;

    // How to turn a GeometryId back into the engine handle it was interned
    // from. Supplied by the adapter, which owns that mapping; this layer
    // must not keep its own copy of it.
    std::function<uint64_t(semantic::GeometryId)> m_geometryTag;
public:
    void resolveGeometryWith(std::function<uint64_t(semantic::GeometryId)> f)
    {
        m_geometryTag = std::move(f);
    }
    void canStandThrough(
        semantic::GeometryId geometry, uint32_t step, uint32_t steps,
        const std::vector<traversal::PhysicsOracle::Stance> &places,
        std::vector<char> &out) const override;
    void sweptThrough(
        semantic::GeometryId geometry, uint32_t step, uint32_t steps,
        const std::vector<traversal::PhysicsOracle::Stance> &places,
        std::vector<char> &out) const override;
    bool anyWayOut(
        const std::vector<std::pair<semantic::GeometryId, uint32_t>> &posed,
        const semantic::Region &from, const semantic::Vec2 &at,
        const std::vector<traversal::PhysicsOracle::WayOut> &ways)
        const override;
    bool canTraverseWith(semantic::GeometryId geometry, uint32_t configuration,
                         traversal::Mode mode,
                         const semantic::SpatialRelation &relation,
                         const semantic::Region &from,
                         const semantic::Region &to,
                         const semantic::Vec2 *startFrom,
                         semantic::Vec2 &crossing, semantic::Vec2 &arrival,
                         semantic::Vec2 &departure) const override;
private:
    traversal::ActorProfile m_profile;
    // What the body measured the first time it was seen in each posture, so
    // the walk animation cannot keep changing what the body is said to be.
    struct Measured { int key = -1; BodyShape shape; };
    mutable std::vector<Measured> m_measured;
    BodyShape stableBody() const;
    mutable int m_queries = 0;
    mutable std::vector<Refusal> m_refusal;
    mutable std::vector<Evidence> m_evidence;
    mutable std::vector<int> m_obstacle;
};

} // namespace bloodmap
