#include "bot_overlay.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "build.h"
#include "colmatch.h"
#include "../../../blood.h"
#include "../../../common_game.h"
#include "../../../globals.h"
#include "../../../player.h"
#include "../../../trig.h"
#include "../../../view.h"

namespace botdebug {

namespace {

using semantic::Region;
using semantic::Segment;
using semantic::SpatialRelation;
using semantic::Vec2;

// Far enough to see the room you are in and the ones next to it; drawing the
// whole map at once is unreadable and slow.
constexpr int kOverlayRange = 16384;
// Lines are drawn a little above the surface they describe so they do not
// fight with the floor texture for the same pixels.
constexpr int kLift = 128;

struct ScreenPoint
{
    int x = 0;
    int y = 0;
    double depth = 0;
    bool visible = false;
};

struct Projector
{
    double centerX = 0;
    double centerY = 0;
    double cosine = 0;
    double sine = 0;
    double scale = 0;
    Camera camera;

    ScreenPoint operator()(int x, int y, int z) const
    {
        ScreenPoint point;
        const double dx = double(x) - camera.x;
        const double dy = double(y) - camera.y;
        const double forward = dx * cosine + dy * sine;
        if (forward <= 32.0)
            return point;
        const double side = dy * cosine - dx * sine;
        point.x = int(std::lround(centerX + side * scale / forward));
        point.y = int(std::lround(centerY
            + double(z - camera.z) * scale / (forward * 16.0)));
        point.depth = forward;
        point.visible = point.x >= gViewX0 - 64 && point.x <= gViewX1 + 64
            && point.y >= gViewY0 - 64 && point.y <= gViewY1 + 64;
        return point;
    }
};

int64_t distanceSquared(int x1, int y1, int x2, int y2)
{
    const int64_t dx = int64_t(x2) - x1;
    const int64_t dy = int64_t(y2) - y1;
    return dx * dx + dy * dy;
}

void drawSegment(const Projector &project, int x1, int y1, int z1, int x2,
                 int y2, int z2, int colour)
{
    const ScreenPoint a = project(x1, y1, z1);
    const ScreenPoint b = project(x2, y2, z2);
    if (a.depth <= 0 || b.depth <= 0 || (!a.visible && !b.visible))
        return;
    renderDrawLine(a.x << 12, a.y << 12, b.x << 12, b.y << 12, char(colour));
}

void drawLabel(const Projector &project, int x, int y, int z,
               const char *text)
{
    const ScreenPoint point = project(x, y, z);
    if (!point.visible || point.depth <= 0)
        return;
    // viewDrawText works in Blood's aspect-corrected 320x200 space while the
    // projection above is in framebuffer pixels, so only the position is
    // converted.
    const int aspect = std::min(scale(ydim, 4, 3), xdim);
    viewDrawText(3, text, scale(point.x, 320, std::max(1, aspect)),
                 scale(point.y, 200, std::max(1, ydim)), -128, 0, 0, true,
                 256);
}

void drawLoop(const Projector &project, const semantic::Loop &loop,
              const semantic::Plane &support, int lift, int colour)
{
    for (size_t i = 0; i < loop.size(); ++i)
    {
        const Vec2 &a = loop[i];
        const Vec2 &b = loop[(i + 1) % loop.size()];
        drawSegment(project, a.x, a.y, support.zAt(a.x, a.y) - lift, b.x,
                    b.y, support.zAt(b.x, b.y) - lift, colour);
    }
}

} // namespace

void drawWorld(const Camera &camera, const semantic::SemanticWorld &world,
               const traversal::TraversalModel &traversal)
{
    if (!gGameStarted || gViewMode != 3 || world.regions().empty())
        return;

    const int viewWidth = std::max(1, gViewX1 - gViewX0 + 1);
    const int viewHeight = std::max(1, gViewY1 - gViewY0 + 1);
    Projector project;
    project.camera = camera;
    project.centerX = gViewX0 + viewWidth * 0.5;
    project.centerY = gViewY0
        + fix16_to_float(camera.horizon) * viewHeight / 200.0;
    const int angle = fix16_to_int(camera.angle) & kAngMask;
    project.cosine = double(Cos(angle)) / double(1 << 30);
    project.sine = double(Sin(angle)) / double(1 << 30);
    project.scale = viewWidth * 0.5 * 65536.0 / std::max(1, viewingrange);

    const int white = paletteGetClosestColor(255, 255, 255);
    const int green = paletteGetClosestColor(48, 255, 96);
    const int red = paletteGetClosestColor(255, 48, 48);
    const int amber = paletteGetClosestColor(255, 192, 32);
    const int violet = paletteGetClosestColor(220, 100, 255);
    const int grey = paletteGetClosestColor(140, 140, 140);

    auto withinRange = [&](const Vec2 &point) {
        return distanceSquared(camera.x, camera.y, point.x, point.y)
            <= int64_t(kOverlayRange) * kOverlayRange;
    };

    // Regions. One outline per continuous piece of space -- white where the
    // actor is standing, violet where it has been seen, amber where it has
    // not. A stretch of floor with no outline round it is the mapper missing
    // a place, and that is what this is for.
    for (const Region &region : world.regions())
    {
        if (!region.exists || region.footprint.size() < 3)
            continue;
        if (!withinRange(region.interior))
            continue;
        const int colour = region.id == world.actor().region ? white
            : region.observed ? violet : amber;
        drawLoop(project, region.footprint, region.support, kLift, colour);
        // What is standing in it, and what it wraps around. Both are things
        // to walk past, never places of their own.
        for (const semantic::Loop &hole : region.holes)
            drawLoop(project, hole, region.support, kLift * 2, red);
        for (const Segment &barrier : region.barriers)
            drawSegment(project, barrier.from.x, barrier.from.y,
                        region.support.zAt(barrier.from.x, barrier.from.y)
                            - kLift * 2,
                        barrier.to.x, barrier.to.y,
                        region.support.zAt(barrier.to.x, barrier.to.y)
                            - kLift * 2, red);
        const semantic::Vec3 anchor = region.anchor();
        char label[48];
        std::snprintf(label, sizeof(label), "%u", unsigned(region.id));
        drawLabel(project, anchor.x, anchor.y, anchor.z - kLift * 6, label);
    }

    // Openings. Green where the model says the body can walk it, amber where
    // something is physically possible but this bot cannot drive it, red
    // where the model says no, grey where it has already been through. A red
    // line across a doorway you can plainly walk through is the model being
    // wrong, and that is the point.
    for (const SpatialRelation &relation : world.relations())
    {
        if (!relation.exists || relation.gateway.width() <= 0)
            continue;
        if (!withinRange(relation.gateway.midpoint()))
            continue;
        const Region *from = world.region(relation.from);
        if (!from)
            continue;
        const bool walkable = traversal.derived(relation.id,
                                                traversal::Mode::Walk);
        bool other = false;
        for (int mode = 0; mode < traversal::kModeCount && !other; ++mode)
            if (mode != int(traversal::Mode::Walk))
                other = traversal.derived(relation.id, traversal::Mode(mode));
        int colour = walkable ? green : other ? amber : red;
        if (walkable && relation.crossed)
            colour = grey;
        const Vec2 &a = relation.gateway.from;
        const Vec2 &b = relation.gateway.to;
        const int lift = kLift * 3;
        drawSegment(project, a.x, a.y, from->support.zAt(a.x, a.y) - lift,
                    b.x, b.y, from->support.zAt(b.x, b.y) - lift, colour);
    }

    // What can be done, and from where.
    for (const semantic::Affordance &affordance : world.affordances())
    {
        if (!affordance.exists || !withinRange({ affordance.target.x,
                                          affordance.target.y }))
            continue;
        const int colour = affordance.executable ? green : red;
        const int span = 192;
        drawSegment(project, affordance.target.x - span, affordance.target.y,
                    affordance.target.z, affordance.target.x + span,
                    affordance.target.y, affordance.target.z, colour);
        drawSegment(project, affordance.target.x, affordance.target.y - span,
                    affordance.target.z, affordance.target.x,
                    affordance.target.y + span, affordance.target.z, colour);
        for (const semantic::ExecutionOption &option : affordance.domain)
            drawSegment(project, affordance.target.x, affordance.target.y,
                        affordance.target.z, option.at.x, option.at.y,
                        option.at.z - kLift, colour);
        char label[64];
        std::snprintf(label, sizeof(label), "%s %u%s",
                      semantic::actionName(affordance.action),
                      unsigned(affordance.id),
                      affordance.attempts ? "*" : "");
        drawLabel(project, affordance.target.x, affordance.target.y,
                  affordance.target.z, label);
    }
}

} // namespace botdebug
