#pragma once

// OptimalPLA based on the PGM-Index implementation:
// https://github.com/gvinciguerra/PGM-index
//
// Licensed under the Apache License, Version 2.0.
// Copyright (c) 2018 Giorgio Vinciguerra.

#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace pgm::internal {

template<typename T>
using LargeSigned = typename std::conditional_t<std::is_floating_point_v<T>,
                                                long double,
                                                std::conditional_t<(sizeof(T) < 8), int64_t, __int128>>;

template<typename X, typename Y>
class OptimalPiecewiseLinearModel {
private:
    using SX = LargeSigned<X>;
    using SY = LargeSigned<Y>;

    struct Slope {
        SX dx{};
        SY dy{};

        bool operator<(const Slope &p) const { return dy * p.dx < dx * p.dy; }
        bool operator>(const Slope &p) const { return dy * p.dx > dx * p.dy; }
        bool operator==(const Slope &p) const { return dy * p.dx == dx * p.dy; }
        bool operator!=(const Slope &p) const { return dy * p.dx != dx * p.dy; }
        explicit operator long double() const { return dy / (long double) dx; }
    };

    struct Point {
        X x{};
        Y y{};

        Slope operator-(const Point &p) const { return {SX(x) - p.x, SY(y) - p.y}; }
    };

    const Y epsilon;
    std::vector<Point> lower;
    std::vector<Point> upper;
    X first_x = 0;
    X last_x = 0;
    size_t lower_start = 0;
    size_t upper_start = 0;
    size_t points_in_hull = 0;
    Point rectangle[4];

    auto cross(const Point &O, const Point &A, const Point &B) const {
        auto OA = A - O;
        auto OB = B - O;
        return OA.dx * OB.dy - OA.dy * OB.dx;
    }

public:

    class CanonicalSegment;

    explicit OptimalPiecewiseLinearModel(Y epsilon) : epsilon(epsilon), lower(), upper() {
        if (epsilon < 0)
            throw std::invalid_argument("epsilon cannot be negative");

        upper.reserve(1u << 16);
        lower.reserve(1u << 16);
    }

    bool add_point(const X &x, const Y &y) {
        if (points_in_hull > 0 && x <= last_x)
            throw std::logic_error("Points must be increasing by x.");

        last_x = x;
        auto max_y = std::numeric_limits<Y>::max();
        auto min_y = std::numeric_limits<Y>::lowest();
        Point p1{x, y >= max_y - epsilon ? max_y : y + epsilon};
        Point p2{x, y <= min_y + epsilon ? min_y : y - epsilon};

        if (points_in_hull == 0) {
            first_x = x;
            rectangle[0] = p1;
            rectangle[1] = p2;
            upper.clear();
            lower.clear();
            upper.push_back(p1);
            lower.push_back(p2);
            upper_start = lower_start = 0;
            ++points_in_hull;
            return true;
        }

        if (points_in_hull == 1) {
            rectangle[2] = p2;
            rectangle[3] = p1;
            upper.push_back(p1);
            lower.push_back(p2);
            ++points_in_hull;
            return true;
        }

        auto slope1 = rectangle[2] - rectangle[0];
        auto slope2 = rectangle[3] - rectangle[1];
        bool outside_line1 = p1 - rectangle[2] < slope1;
        bool outside_line2 = p2 - rectangle[3] > slope2;

        if (outside_line1 || outside_line2) {
            points_in_hull = 0;
            return false;
        }

        if (p1 - rectangle[1] < slope2) {
            auto min = lower[lower_start] - p1;
            auto min_i = lower_start;
            for (auto i = lower_start + 1; i < lower.size(); i++) {
                auto val = lower[i] - p1;
                if (val > min)
                    break;
                min = val;
                min_i = i;
            }

            rectangle[1] = lower[min_i];
            rectangle[3] = p1;
            lower_start = min_i;

            auto end = upper.size();
            for (; end >= upper_start + 2 && cross(upper[end - 2], upper[end - 1], p1) <= 0; --end)
                continue;
            upper.resize(end);
            upper.push_back(p1);
        }

        if (p2 - rectangle[0] > slope1) {
            auto max = upper[upper_start] - p2;
            auto max_i = upper_start;
            for (auto i = upper_start + 1; i < upper.size(); i++) {
                auto val = upper[i] - p2;
                if (val < max)
                    break;
                max = val;
                max_i = i;
            }

            rectangle[0] = upper[max_i];
            rectangle[2] = p2;
            upper_start = max_i;

            auto end = lower.size();
            for (; end >= lower_start + 2 && cross(lower[end - 2], lower[end - 1], p2) >= 0; --end)
                continue;
            lower.resize(end);
            lower.push_back(p2);
        }

        ++points_in_hull;
        return true;
    }

    CanonicalSegment get_segment() {
        if (points_in_hull == 1)
            return CanonicalSegment(rectangle[0], rectangle[1], first_x);
        return CanonicalSegment(rectangle, first_x);
    }

    void reset() {
        points_in_hull = 0;
        lower.clear();
        upper.clear();
    }
};

template<typename X, typename Y>
class OptimalPiecewiseLinearModel<X, Y>::CanonicalSegment {
    friend class OptimalPiecewiseLinearModel;

    Point rectangle[4];
    X first;

    CanonicalSegment(const Point &p0, const Point &p1, X first) : rectangle{p0, p1, p0, p1}, first(first) {};

    CanonicalSegment(const Point (&rectangle)[4], X first)
        : rectangle{rectangle[0], rectangle[1], rectangle[2], rectangle[3]}, first(first) {};

    bool one_point() const {
        return rectangle[0].x == rectangle[2].x && rectangle[0].y == rectangle[2].y
            && rectangle[1].x == rectangle[3].x && rectangle[1].y == rectangle[3].y;
    }

public:

    CanonicalSegment() = default;

    X get_first_x() const { return first; }

    std::pair<long double, long double> get_intersection() const {
        auto &p0 = rectangle[0];
        auto &p1 = rectangle[1];
        auto &p2 = rectangle[2];
        auto &p3 = rectangle[3];
        auto slope1 = p2 - p0;
        auto slope2 = p3 - p1;

        if (one_point() || slope1 == slope2)
            return {p0.x, p0.y};

        auto p0p1 = p1 - p0;
        auto a = slope1.dx * slope2.dy - slope1.dy * slope2.dx;
        auto b = (p0p1.dx * slope2.dy - p0p1.dy * slope2.dx) / static_cast<long double>(a);
        auto i_x = p0.x + b * slope1.dx;
        auto i_y = p0.y + b * slope1.dy;
        return {i_x, i_y};
    }

    std::pair<long double, SY> get_floating_point_segment(const X &origin) const {
        if (one_point())
            return {0, (rectangle[0].y + rectangle[1].y) / 2};

        if constexpr (std::is_integral_v<X> && std::is_integral_v<Y>) {
            auto slope = rectangle[3] - rectangle[1];
            auto intercept_n = slope.dy * (SX(origin) - rectangle[1].x);
            auto intercept_d = slope.dx;
            auto rounding_term = ((intercept_n < 0) ^ (intercept_d < 0) ? -1 : +1) * intercept_d / 2;
            auto intercept = (intercept_n + rounding_term) / intercept_d + rectangle[1].y;
            return {static_cast<long double>(slope), intercept};
        }

        auto[i_x, i_y] = get_intersection();
        auto[min_slope, max_slope] = get_slope_range();
        auto slope = (min_slope + max_slope) / 2.;
        auto intercept = i_y - (i_x - origin) * slope;
        return {slope, intercept};
    }

    std::pair<long double, long double> get_slope_range() const {
        if (one_point())
            return {0, 1};

        auto min_slope = static_cast<long double>(rectangle[2] - rectangle[0]);
        auto max_slope = static_cast<long double>(rectangle[3] - rectangle[1]);
        return {min_slope, max_slope};
    }
};

} // namespace pgm::internal


// Public adapter class maintaining the existing OptimalPLA interface.
class OptimalPLA {
public:
    explicit OptimalPLA(double epsilon) : epsilon_(epsilon) {}

    std::vector<PLASegment> build(
        const std::vector<Key>& keys,
        size_t start_rank = 0) const
    {
        return build_range(keys, 0, keys.size(), start_rank);
    }

    std::vector<PLASegment> build_range(
        const std::vector<Key>& keys,
        size_t start_idx,
        size_t end_idx,
        size_t start_rank) const;

    bool can_cover_single_segment(
        const std::vector<Key>& keys,
        size_t start_idx,
        size_t end_idx,
        size_t start_rank,
        double& out_slope,
        double& out_intercept) const;

    double epsilon() const { return epsilon_; }

private:
    double epsilon_;

    using OPLA = pgm::internal::OptimalPiecewiseLinearModel<Key, size_t>;

    PLASegment segment_from_canonical(
        const typename OPLA::CanonicalSegment& canon,
        Key first_key, Key last_key,
        size_t start_rank, size_t end_rank) const;
};

// --- Implementation ---

inline PLASegment OptimalPLA::segment_from_canonical(
    const typename OPLA::CanonicalSegment& canon,
    Key first_key, Key last_key,
    size_t start_rank, size_t end_rank) const
{
    PLASegment seg;
    seg.start = start_rank;
    seg.end = end_rank;
    seg.first_key = first_key;
    seg.last_key = last_key;

    // Use the mid-slope intersection method (PGM's floating-point branch).
    // The integer branch (get_floating_point_segment) picks the max-slope line
    // which requires rational arithmetic at prediction time for the epsilon
    // guarantee. The intersection-based mid-slope works correctly with
    // floating-point predictions.
    auto [min_slope, max_slope] = canon.get_slope_range();
    if (min_slope == 0.0L && max_slope == 1.0L) {
        // one_point segment: slope range is [0, 1], use slope=0
        seg.slope = 0.0L;
        seg.intercept = static_cast<long double>(start_rank);
    } else {
        seg.slope = (min_slope + max_slope) / 2.0L;
        auto [i_x, i_y] = canon.get_intersection();
        // Line passes through (i_x, i_y) with given slope.
        // Standard form: pred = slope * key + intercept
        //   slope * i_x + intercept = i_y  =>  intercept = i_y - slope * i_x
        seg.intercept = i_y - seg.slope * i_x;
    }

    return seg;
}

inline std::vector<PLASegment> OptimalPLA::build_range(
    const std::vector<Key>& keys,
    size_t start_idx,
    size_t end_idx,
    size_t start_rank) const
{
    std::vector<PLASegment> segments;
    if (start_idx >= end_idx) return segments;

    OPLA opt(static_cast<size_t>(epsilon_));
    size_t seg_start_idx = start_idx;
    size_t seg_start_rank = start_rank;

    for (size_t i = start_idx; i < end_idx; ++i) {
        if (!opt.add_point(keys[i], start_rank + (i - start_idx))) {
            // Point i cannot be added: segment covers [seg_start_idx, i-1]
            auto canon = opt.get_segment();
            segments.push_back(segment_from_canonical(
                canon,
                keys[seg_start_idx],
                keys[i - 1],
                seg_start_rank,
                start_rank + (i - 1 - start_idx)));

            // Start new segment with current point
            opt.reset();
            opt.add_point(keys[i], start_rank + (i - start_idx));
            seg_start_idx = i;
            seg_start_rank = start_rank + (i - start_idx);
        }
    }

    // Extract final segment
    auto canon = opt.get_segment();
    segments.push_back(segment_from_canonical(
        canon,
        keys[seg_start_idx],
        keys[end_idx - 1],
        seg_start_rank,
        start_rank + (end_idx - 1 - start_idx)));

    return segments;
}

inline bool OptimalPLA::can_cover_single_segment(
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

    OPLA opt(static_cast<size_t>(epsilon_));

    for (size_t i = start_idx; i <= end_idx; ++i) {
        if (!opt.add_point(keys[i], start_rank + (i - start_idx))) {
            return false;
        }
    }

    auto canon = opt.get_segment();
    auto [min_slope, max_slope] = canon.get_slope_range();
    if (min_slope == 0.0L && max_slope == 1.0L) {
        out_slope = 0.0;
        out_intercept = static_cast<double>(start_rank);
    } else {
        long double slope = (min_slope + max_slope) / 2.0L;
        auto [i_x, i_y] = canon.get_intersection();
        out_slope = static_cast<double>(slope);
        out_intercept = static_cast<double>(i_y - slope * i_x);
    }

    return true;
}
