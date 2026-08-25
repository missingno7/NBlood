#include "geometry.h"

#include <algorithm>
#include <cmath>
#include <map>

namespace semantic {

namespace {

uint64_t mixHash(uint64_t value, uint64_t item)
{
    value ^= item + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
    return value;
}

bool onSegment(const Vec2 &point, const Vec2 &a, const Vec2 &b)
{
    if (crossOf(a, b, point) != 0)
        return false;
    return point.x >= std::min(a.x, b.x) && point.x <= std::max(a.x, b.x)
        && point.y >= std::min(a.y, b.y) && point.y <= std::max(a.y, b.y);
}

int64_t gcdOf(int64_t a, int64_t b)
{
    a = a < 0 ? -a : a;
    b = b < 0 ? -b : b;
    while (b != 0)
    {
        const int64_t next = a % b;
        a = b;
        b = next;
    }
    return a == 0 ? 1 : a;
}

// The line a segment lies on, in a form two collinear segments always agree
// on: direction reduced to lowest terms with a fixed sign, plus the signed
// area of the origin against it.
struct LineKey
{
    int64_t dx = 0;
    int64_t dy = 0;
    int64_t offset = 0;

    bool operator<(const LineKey &other) const
    {
        if (dx != other.dx) return dx < other.dx;
        if (dy != other.dy) return dy < other.dy;
        return offset < other.offset;
    }
};

bool lineOf(const Segment &segment, LineKey &key)
{
    int64_t dx = int64_t(segment.to.x) - segment.from.x;
    int64_t dy = int64_t(segment.to.y) - segment.from.y;
    if (dx == 0 && dy == 0)
        return false;
    const int64_t divisor = gcdOf(dx, dy);
    dx /= divisor;
    dy /= divisor;
    if (dx < 0 || (dx == 0 && dy < 0))
    {
        dx = -dx;
        dy = -dy;
    }
    key.dx = dx;
    key.dy = dy;
    key.offset = dx * int64_t(segment.from.y) - dy * int64_t(segment.from.x);
    return true;
}

int64_t projectOnto(const LineKey &line, const Vec2 &point)
{
    return line.dx * int64_t(point.x) + line.dy * int64_t(point.y);
}

Vec2 unprojectFrom(const LineKey &line, const Vec2 &anchor, int64_t position)
{
    const int64_t base = projectOnto(line, anchor);
    const int64_t span = line.dx * line.dx + line.dy * line.dy;
    const int64_t steps = (position - base) / span;
    return { int(anchor.x + line.dx * steps), int(anchor.y + line.dy * steps) };
}

struct Interval
{
    int64_t low = 0;
    int64_t high = 0;
};

} // namespace

Plane flatPlane(int z)
{
    Plane plane;
    plane.z0 = int64_t(z) * kPlaneScale;
    return plane;
}

int64_t crossOf(const Vec2 &origin, const Vec2 &a, const Vec2 &b)
{
    return (int64_t(a.x) - origin.x) * (int64_t(b.y) - origin.y)
         - (int64_t(a.y) - origin.y) * (int64_t(b.x) - origin.x);
}

int64_t signedDoubleArea(const Loop &loop)
{
    int64_t total = 0;
    for (size_t i = 0; i < loop.size(); ++i)
    {
        const Vec2 &a = loop[i];
        const Vec2 &b = loop[(i + 1) % loop.size()];
        total += int64_t(a.x) * b.y - int64_t(b.x) * a.y;
    }
    return total;
}

void makeCounterClockwise(Loop &loop)
{
    if (signedDoubleArea(loop) < 0)
        std::reverse(loop.begin(), loop.end());
}

void removeCollinear(Loop &loop)
{
    bool changed = true;
    while (changed && loop.size() > 3)
    {
        changed = false;
        for (size_t i = 0; i < loop.size(); ++i)
        {
            const Vec2 &previous = loop[(i + loop.size() - 1) % loop.size()];
            const Vec2 &next = loop[(i + 1) % loop.size()];
            if (loop[i] == previous || loop[i] == next
                || crossOf(previous, loop[i], next) == 0)
            {
                loop.erase(loop.begin() + int(i));
                changed = true;
                break;
            }
        }
    }
}

bool pointInLoop(const Loop &loop, const Vec2 &point)
{
    if (loop.size() < 3)
        return false;
    bool inside = false;
    for (size_t i = 0, j = loop.size() - 1; i < loop.size(); j = i++)
    {
        if (onSegment(point, loop[j], loop[i]))
            return true;
        if ((loop[i].y > point.y) == (loop[j].y > point.y))
            continue;
        const double t = double(point.y - loop[i].y)
            / double(loop[j].y - loop[i].y);
        if (double(point.x) < double(loop[i].x)
                + t * double(loop[j].x - loop[i].x))
            inside = !inside;
    }
    return inside;
}

bool pointInPolygon(const Polygon &polygon, const Vec2 &point)
{
    if (!pointInLoop(polygon.outer, point))
        return false;
    for (const Loop &hole : polygon.holes)
        if (pointInLoop(hole, point))
            return false;
    return true;
}

Vec2 centroidOf(const Loop &loop)
{
    if (loop.empty())
        return {};
    const int64_t doubleArea = signedDoubleArea(loop);
    if (doubleArea == 0)
    {
        int64_t sx = 0;
        int64_t sy = 0;
        for (const Vec2 &point : loop)
        {
            sx += point.x;
            sy += point.y;
        }
        return { int(sx / int64_t(loop.size())),
                 int(sy / int64_t(loop.size())) };
    }
    double cx = 0.0;
    double cy = 0.0;
    for (size_t i = 0; i < loop.size(); ++i)
    {
        const Vec2 &a = loop[i];
        const Vec2 &b = loop[(i + 1) % loop.size()];
        const double factor = double(int64_t(a.x) * b.y - int64_t(b.x) * a.y);
        cx += double(a.x + b.x) * factor;
        cy += double(a.y + b.y) * factor;
    }
    const double scale = 1.0 / (3.0 * double(doubleArea));
    return { int(cx * scale), int(cy * scale) };
}

int planarDistance(const Vec2 &a, const Vec2 &b)
{
    const double dx = double(b.x) - a.x;
    const double dy = double(b.y) - a.y;
    return int(std::lround(std::sqrt(dx * dx + dy * dy)));
}

int planarDistance(const Vec3 &a, const Vec3 &b)
{
    const double dx = double(b.x) - a.x;
    const double dy = double(b.y) - a.y;
    return int(std::lround(std::sqrt(dx * dx + dy * dy)));
}

Vec2 closestPointOnSegment(const Vec2 &point, const Vec2 &a, const Vec2 &b)
{
    const double dx = double(b.x) - a.x;
    const double dy = double(b.y) - a.y;
    const double length = dx * dx + dy * dy;
    if (length < 1.0)
        return a;
    double t = (double(point.x - a.x) * dx + double(point.y - a.y) * dy)
        / length;
    t = std::max(0.0, std::min(1.0, t));
    return { int(std::lround(a.x + t * dx)), int(std::lround(a.y + t * dy)) };
}

int64_t pointSegmentDistanceSquared(const Vec2 &point, const Vec2 &a,
                                    const Vec2 &b)
{
    const double dx = double(b.x) - a.x;
    const double dy = double(b.y) - a.y;
    const double length = dx * dx + dy * dy;
    double t = 0.0;
    if (length >= 1.0)
    {
        t = (double(point.x - a.x) * dx + double(point.y - a.y) * dy) / length;
        t = std::max(0.0, std::min(1.0, t));
    }
    const double px = double(a.x) + t * dx - point.x;
    const double py = double(a.y) + t * dy - point.y;
    return int64_t(px * px + py * py);
}

bool segmentsProperlyIntersect(const Vec2 &a1, const Vec2 &a2, const Vec2 &b1,
                               const Vec2 &b2)
{
    const int64_t d1 = crossOf(a1, a2, b1);
    const int64_t d2 = crossOf(a1, a2, b2);
    const int64_t d3 = crossOf(b1, b2, a1);
    const int64_t d4 = crossOf(b1, b2, a2);
    return ((d1 > 0) != (d2 > 0)) && d1 != 0 && d2 != 0
        && ((d3 > 0) != (d4 > 0)) && d3 != 0 && d4 != 0;
}

bool pointInBox(const Vec2 &point, const Vec2 &centre, int half)
{
    return std::abs(point.x - centre.x) < half
        && std::abs(point.y - centre.y) < half;
}

bool segmentEntersBox(const Vec2 &a, const Vec2 &b, const Vec2 &centre,
                      int half)
{
    if (pointInBox(a, centre, half) || pointInBox(b, centre, half))
        return true;
    // Liang-Barsky against the square, in the segment's own parameter.
    double low = 0.0;
    double high = 1.0;
    const double delta[2] = { double(b.x) - a.x, double(b.y) - a.y };
    const double start[2] = { double(a.x), double(a.y) };
    const double middle[2] = { double(centre.x), double(centre.y) };
    for (int axis = 0; axis < 2; ++axis)
    {
        const double minimum = middle[axis] - half;
        const double maximum = middle[axis] + half;
        if (std::abs(delta[axis]) < 1e-9)
        {
            if (start[axis] <= minimum || start[axis] >= maximum)
                return false;
            continue;
        }
        double enter = (minimum - start[axis]) / delta[axis];
        double leave = (maximum - start[axis]) / delta[axis];
        if (enter > leave)
            std::swap(enter, leave);
        low = std::max(low, enter);
        high = std::min(high, leave);
        if (low >= high)
            return false;
    }
    return low < high;
}

int64_t segmentDistanceSquared(const Vec2 &a1, const Vec2 &a2, const Vec2 &b1,
                               const Vec2 &b2)
{
    if (segmentsProperlyIntersect(a1, a2, b1, b2))
        return 0;
    if (onSegment(b1, a1, a2) || onSegment(b2, a1, a2)
        || onSegment(a1, b1, b2) || onSegment(a2, b1, b2))
        return 0;
    int64_t best = pointSegmentDistanceSquared(a1, b1, b2);
    best = std::min(best, pointSegmentDistanceSquared(a2, b1, b2));
    best = std::min(best, pointSegmentDistanceSquared(b1, a1, a2));
    best = std::min(best, pointSegmentDistanceSquared(b2, a1, a2));
    return best;
}

Vec2 justInside(const Polygon &shape, const Vec2 &edgeFrom,
                const Vec2 &edgeTo, const Vec2 &at, int inset)
{
    const double dx = double(edgeTo.x) - edgeFrom.x;
    const double dy = double(edgeTo.y) - edgeFrom.y;
    const double width = std::sqrt(dx * dx + dy * dy);
    if (width < 1.0 || inset <= 0)
        return at;
    const double ux = -dy / width;
    const double uy = dx / width;
    // Which way is in is decided beside the point, not from the far end of
    // the shape: a shape that wraps around what is on the other side of this
    // edge has its middle on the wrong side of the edge's line.
    const Vec2 plus = { at.x + int(std::lround(ux * 4)),
                        at.y + int(std::lround(uy * 4)) };
    const Vec2 minus = { at.x - int(std::lround(ux * 4)),
                         at.y - int(std::lround(uy * 4)) };
    const bool towardPlus = pointInPolygon(shape, plus);
    const bool towardMinus = pointInPolygon(shape, minus);
    if (towardPlus == towardMinus)
        return at;
    const double sign = towardPlus ? 1.0 : -1.0;
    const Vec2 away = { at.x + int(std::lround(ux * sign * inset * 2)),
                        at.y + int(std::lround(uy * sign * inset * 2)) };
    const int room = depthInto(shape, at, away);
    const int depth = room >= inset ? inset : std::max(1, room / 2);
    return { at.x + int(std::lround(ux * sign * depth)),
             at.y + int(std::lround(uy * sign * depth)) };
}

int depthInto(const Polygon &polygon, const Vec2 &from, const Vec2 &to)
{
    const double dx = double(to.x) - from.x;
    const double dy = double(to.y) - from.y;
    const double span = std::sqrt(dx * dx + dy * dy);
    if (span < 1.0)
        return 0;
    double nearest = 1.0;
    auto scan = [&](const Loop &loop)
    {
        for (size_t i = 0; i < loop.size(); ++i)
        {
            const Vec2 &a = loop[i];
            const Vec2 &b = loop[(i + 1) % loop.size()];
            const double ex = double(b.x) - a.x;
            const double ey = double(b.y) - a.y;
            const double denominator = dx * ey - dy * ex;
            if (denominator == 0.0)
                continue;
            const double ox = double(a.x) - from.x;
            const double oy = double(a.y) - from.y;
            const double t = (ox * ey - oy * ex) / denominator;
            const double u = (ox * dy - oy * dx) / denominator;
            if (u < 0.0 || u > 1.0)
                continue;
            // A crossing at the start is the boundary this push begins on.
            if (t * span <= 1.0 || t > nearest)
                continue;
            nearest = t;
        }
    };
    scan(polygon.outer);
    for (const Loop &hole : polygon.holes)
        scan(hole);
    return int(nearest * span);
}

Vec2 representativePoint(const Polygon &polygon)
{
    if (polygon.outer.size() < 3)
        return polygon.outer.empty() ? Vec2{} : polygon.outer[0];

    std::vector<Segment> edges;
    std::vector<int> cuts;
    auto addLoop = [&](const Loop &loop) {
        for (size_t i = 0; i < loop.size(); ++i)
        {
            const Vec2 &a = loop[i];
            const Vec2 &b = loop[(i + 1) % loop.size()];
            cuts.push_back(a.y);
            if (a.y != b.y)
                edges.push_back({ a, b });
        }
    };
    addLoop(polygon.outer);
    for (const Loop &hole : polygon.holes)
        addLoop(hole);
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());

    Vec2 best = centroidOf(polygon.outer);
    int64_t bestWidth = pointInPolygon(polygon, best) ? 1 : -1;
    std::vector<double> crossings;
    for (size_t slab = 0; slab + 1 < cuts.size(); ++slab)
    {
        const double y = (double(cuts[slab]) + double(cuts[slab + 1])) * 0.5;
        crossings.clear();
        for (const Segment &edge : edges)
        {
            const double low = std::min(edge.from.y, edge.to.y);
            const double high = std::max(edge.from.y, edge.to.y);
            if (y < low || y > high)
                continue;
            const double t = (y - edge.from.y)
                / (double(edge.to.y) - edge.from.y);
            crossings.push_back(edge.from.x
                + t * (double(edge.to.x) - edge.from.x));
        }
        std::sort(crossings.begin(), crossings.end());
        for (size_t pair = 0; pair + 1 < crossings.size(); pair += 2)
        {
            const double width = crossings[pair + 1] - crossings[pair];
            if (int64_t(width) <= bestWidth)
                continue;
            const Vec2 candidate = {
                int(std::lround((crossings[pair] + crossings[pair + 1]) * 0.5)),
                int(std::lround(y)) };
            if (!pointInPolygon(polygon, candidate))
                continue;
            bestWidth = int64_t(width);
            best = candidate;
        }
    }
    return best;
}

void sharedBoundaries(const Loop &left, const Loop &right,
                      std::vector<Segment> &out)
{
    out.clear();
    std::vector<Segment> pieces;
    for (size_t i = 0; i < left.size(); ++i)
    {
        const Vec2 &a1 = left[i];
        const Vec2 &a2 = left[(i + 1) % left.size()];
        const int64_t dx = int64_t(a2.x) - a1.x;
        const int64_t dy = int64_t(a2.y) - a1.y;
        const int64_t span = dx * dx + dy * dy;
        if (span == 0)
            continue;
        for (size_t j = 0; j < right.size(); ++j)
        {
            const Vec2 &b1 = right[j];
            const Vec2 &b2 = right[(j + 1) % right.size()];
            if (crossOf(a1, a2, b1) != 0 || crossOf(a1, a2, b2) != 0)
                continue;
            auto project = [&](const Vec2 &point) {
                return (int64_t(point.x) - a1.x) * dx
                     + (int64_t(point.y) - a1.y) * dy;
            };
            const int64_t low = std::max<int64_t>(0,
                std::min(project(b1), project(b2)));
            const int64_t high = std::min<int64_t>(span,
                std::max(project(b1), project(b2)));
            if (high <= low)
                continue;
            pieces.push_back({ { int(a1.x + dx * low / span),
                                 int(a1.y + dy * low / span) },
                               { int(a1.x + dx * high / span),
                                 int(a1.y + dy * high / span) } });
        }
    }
    mergeCollinearSegments(pieces, out);
}

void mergeCollinearSegments(const std::vector<Segment> &in,
                            std::vector<Segment> &out)
{
    out.clear();
    std::map<LineKey, std::vector<Interval>> lines;
    std::map<LineKey, Vec2> anchors;
    for (const Segment &segment : in)
    {
        LineKey line;
        if (!lineOf(segment, line))
            continue;
        const int64_t a = projectOnto(line, segment.from);
        const int64_t b = projectOnto(line, segment.to);
        lines[line].push_back({ std::min(a, b), std::max(a, b) });
        if (anchors.find(line) == anchors.end())
            anchors[line] = segment.from;
    }
    for (auto &entry : lines)
    {
        std::vector<Interval> &spans = entry.second;
        std::sort(spans.begin(), spans.end(),
            [](const Interval &left, const Interval &right) {
                return left.low != right.low ? left.low < right.low
                                             : left.high < right.high;
            });
        Interval current = spans.front();
        const Vec2 &anchor = anchors[entry.first];
        for (size_t i = 1; i <= spans.size(); ++i)
        {
            if (i < spans.size() && spans[i].low <= current.high)
            {
                current.high = std::max(current.high, spans[i].high);
                continue;
            }
            out.push_back({ unprojectFrom(entry.first, anchor, current.low),
                            unprojectFrom(entry.first, anchor, current.high) });
            if (i < spans.size())
                current = spans[i];
        }
    }
}

void subtractCollinear(const std::vector<Segment> &in,
                       const std::vector<Segment> &cuts,
                       std::vector<Segment> &out)
{
    out.clear();
    std::map<LineKey, std::vector<Interval>> removals;
    for (const Segment &cut : cuts)
    {
        LineKey line;
        if (!lineOf(cut, line))
            continue;
        const int64_t a = projectOnto(line, cut.from);
        const int64_t b = projectOnto(line, cut.to);
        removals[line].push_back({ std::min(a, b), std::max(a, b) });
    }
    for (const Segment &segment : in)
    {
        LineKey line;
        if (!lineOf(segment, line))
            continue;
        auto found = removals.find(line);
        if (found == removals.end())
        {
            out.push_back(segment);
            continue;
        }
        const int64_t start = projectOnto(line, segment.from);
        const int64_t end = projectOnto(line, segment.to);
        const bool forward = start <= end;
        int64_t low = std::min(start, end);
        const int64_t high = std::max(start, end);
        std::vector<Interval> spans = found->second;
        std::sort(spans.begin(), spans.end(),
            [](const Interval &left, const Interval &right) {
                return left.low < right.low;
            });
        auto emit = [&](int64_t from, int64_t to) {
            if (to <= from)
                return;
            const Vec2 a = unprojectFrom(line, segment.from,
                                         forward ? from : to);
            const Vec2 b = unprojectFrom(line, segment.from,
                                         forward ? to : from);
            out.push_back({ a, b });
        };
        for (const Interval &span : spans)
        {
            if (span.high <= low)
                continue;
            if (span.low >= high)
                break;
            emit(low, std::min(span.low, high));
            low = std::max(low, span.high);
            if (low >= high)
                break;
        }
        emit(low, high);
    }
}

bool loopsOverlap(const Loop &left, const Loop &right)
{
    for (const Vec2 &point : left)
        if (pointInLoop(right, point))
            return true;
    for (const Vec2 &point : right)
        if (pointInLoop(left, point))
            return true;
    if (pointInLoop(right, centroidOf(left))
        || pointInLoop(left, centroidOf(right)))
        return true;
    for (size_t i = 0; i < left.size(); ++i)
        for (size_t j = 0; j < right.size(); ++j)
            if (segmentsProperlyIntersect(left[i],
                    left[(i + 1) % left.size()], right[j],
                    right[(j + 1) % right.size()]))
                return true;
    return false;
}

bool segmentCrossesLoop(const Vec2 &from, const Vec2 &to, const Loop &loop)
{
    if (loop.size() < 3)
        return false;
    if (pointInLoop(loop, from) || pointInLoop(loop, to))
        return true;
    const Vec2 middle = { (from.x + to.x) / 2, (from.y + to.y) / 2 };
    if (pointInLoop(loop, middle))
        return true;
    for (size_t i = 0; i < loop.size(); ++i)
        if (segmentsProperlyIntersect(from, to, loop[i],
                                      loop[(i + 1) % loop.size()]))
            return true;
    return false;
}

uint64_t shapeKey(const Loop &loop)
{
    Loop normalized = loop;
    removeCollinear(normalized);
    makeCounterClockwise(normalized);
    if (normalized.empty())
        return 0;
    // Start from the lexicographically smallest vertex so the same outline
    // hashes the same however it was wound or where its list began.
    size_t start = 0;
    for (size_t i = 1; i < normalized.size(); ++i)
        if (normalized[i].x < normalized[start].x
            || (normalized[i].x == normalized[start].x
                && normalized[i].y < normalized[start].y))
            start = i;
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < normalized.size(); ++i)
    {
        const Vec2 &point = normalized[(start + i) % normalized.size()];
        hash = mixHash(hash, uint64_t(uint32_t(point.x)));
        hash = mixHash(hash, uint64_t(uint32_t(point.y)));
    }
    return hash;
}

} // namespace semantic
