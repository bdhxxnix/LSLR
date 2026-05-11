#include "optimal_pla.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using V2 = std::pair<double, double>;  // (a=slope, b=intercept)

namespace {

constexpr double EPSILON_DBL = 1e-12;

// Build initial bounding box polygon in CCW order, scaled to the key range.
// For keys in [key_min, key_max] with n points:
//   slope a ≈ n/(key_max-key_min), so bound a generously.
std::vector<V2> initial_polygon(Key key_min, Key key_max, size_t n) {
    double kmin = static_cast<double>(key_min);
    double kmax = static_cast<double>(key_max);
    double key_span = kmax - kmin;
    if (key_span < 1.0) key_span = 1.0;

    // Estimated slope range.
    // Use a generous multiplier because local slope can be much steeper than global.
    double a_hi = 1e6 * static_cast<double>(n) / key_span;
    // Also ensure a_hi >= n (covers the case where all keys are in a tiny range)
    if (a_hi < static_cast<double>(n)) a_hi = static_cast<double>(n);
    double a_lo = -a_hi * 0.01;  // slightly negative for ε slack

    // Cap a_hi so a*x stays within double precision (2^53 ≈ 9e15)
    // a_hi * kmax should stay below ~1e15
    if (a_hi * kmax > 1e14) a_hi = 1e14 / kmax;
    if (a_lo < -1e9) a_lo = -1e9;
    if (a_hi > 1e9) a_hi = 1e9;

    // Intercept range
    double b_hi = static_cast<double>(n) + a_hi * kmax + 1e6;
    double b_lo = -a_hi * kmax - 1e6;
    if (b_lo > -1e9) b_lo = -1e9;
    if (b_hi < 1e9) b_hi = 1e9;

    return {
        {a_lo, b_lo},
        {a_hi, b_lo},
        {a_hi, b_hi},
        {a_lo, b_hi}
    };
}

// Clip convex polygon (CCW vertices) against half-plane: A*a + B*b <= C.
// Returns false if polygon becomes empty.
bool clip_polygon(std::vector<V2>& poly, double A, double B, double C) {
    if (poly.size() < 3) return false;

    std::vector<V2> out;
    out.reserve(poly.size() + 1);
    size_t n = poly.size();

    for (size_t i = 0; i < n; ++i) {
        const V2& v1 = poly[i];
        const V2& v2 = poly[(i + 1) % n];

        double d1 = A * v1.first + B * v1.second - C;
        double d2 = A * v2.first + B * v2.second - C;

        bool in1 = (d1 <= EPSILON_DBL);
        bool in2 = (d2 <= EPSILON_DBL);

        if (in1) {
            out.push_back(v1);
        }

        if (in1 != in2) {
            double t = d1 / (d1 - d2);
            V2 inter = {
                v1.first + t * (v2.first - v1.first),
                v1.second + t * (v2.second - v1.second)
            };
            out.push_back(inter);
        }
    }

    if (out.size() < 3) {
        poly = std::move(out);
        return false;
    }

    poly = std::move(out);
    return true;
}

// Pick a centroid point from the feasible polygon.
void polygon_centroid(const std::vector<V2>& poly, double& a, double& b) {
    double sa = 0.0, sb = 0.0;
    for (const auto& v : poly) {
        sa += v.first;
        sb += v.second;
    }
    a = sa / poly.size();
    b = sb / poly.size();
}

} // anonymous namespace

bool OptimalPLA::can_cover_single_segment(
    const std::vector<Key>& keys,
    size_t start_idx,
    size_t end_idx,
    size_t start_rank,
    double& out_slope,
    double& out_intercept) const
{
    if (end_idx <= start_idx) {
        out_slope = 0.0;
        out_intercept = static_cast<double>(start_rank);
        return true;
    }

    Key k_min = keys[start_idx];
    Key k_max = keys[end_idx];
    size_t seg_n = end_idx - start_idx + 1;
    auto poly = initial_polygon(k_min, k_max, seg_n);

    for (size_t i = start_idx; i <= end_idx; ++i) {
        double x = static_cast<double>(keys[i]);
        double y = static_cast<double>(start_rank + (i - start_idx));
        double eps = epsilon_;

        // Upper constraint: a*x + b <= y + eps
        if (!clip_polygon(poly, x, 1.0, y + eps)) return false;
        // Lower constraint: a*x + b >= y - eps
        if (!clip_polygon(poly, -x, -1.0, -y + eps)) return false;
    }

    polygon_centroid(poly, out_slope, out_intercept);
    return true;
}

size_t OptimalPLA::build_one_segment(
    const std::vector<Key>& keys,
    size_t start_idx,
    size_t start_rank,
    double& out_slope,
    double& out_intercept) const
{
    // Use full key range from the entire dataset for safe initial bounds
    Key k_all_min = keys.front();
    Key k_all_max = keys.back();
    size_t n_total = keys.size();
    auto poly = initial_polygon(k_all_min, k_all_max, n_total);
    size_t i = start_idx;

    for (; i < keys.size(); ++i) {
        double x = static_cast<double>(keys[i]);
        double y = static_cast<double>(start_rank + (i - start_idx));
        double eps = epsilon_;

        // Try adding point i
        auto saved = poly;
        bool ok = clip_polygon(poly, x, 1.0, y + eps);
        if (ok) ok = clip_polygon(poly, -x, -1.0, -y + eps);

        if (!ok) {
            // Point i cannot be added, segment ends at i-1
            poly = std::move(saved);
            break;
        }
    }

    // i is the first index that could NOT be added, or keys.size()
    size_t end_idx = (i > start_idx) ? i - 1 : start_idx;
    polygon_centroid(poly, out_slope, out_intercept);
    return end_idx;
}

std::vector<PLASegment> OptimalPLA::build(
    const std::vector<Key>& keys,
    size_t start_rank) const
{
    return build_range(keys, 0, keys.size(), start_rank);
}

std::vector<PLASegment> OptimalPLA::build_range(
    const std::vector<Key>& keys,
    size_t start_idx,
    size_t end_idx,
    size_t start_rank) const
{
    std::vector<PLASegment> segments;
    if (start_idx >= end_idx) return segments;

    size_t current = start_idx;
    size_t current_rank = start_rank;

    while (current < end_idx) {
        double slope, intercept;
        size_t seg_end = build_one_segment(keys, current, current_rank,
                                            slope, intercept);

        PLASegment seg;
        seg.start = current_rank;
        seg.end = current_rank + (seg_end - current);
        seg.first_key = keys[current];
        seg.last_key = keys[seg_end];
        seg.slope = slope;
        seg.intercept = intercept;
        segments.push_back(seg);

        size_t seg_len = seg_end - current + 1;
        current = seg_end + 1;
        current_rank += seg_len;
    }

    return segments;
}
