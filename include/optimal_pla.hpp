#pragma once

#include "common.hpp"

// Builds optimal (minimum-segment) piecewise linear approximation.
// Given sorted keys and error bound epsilon, produces a list of segments
// such that for every key k_i at rank i, |f(k_i) - i| <= epsilon.
class OptimalPLA {
public:
    OptimalPLA(double epsilon) : epsilon_(epsilon) {}

    // Build PLA model for [keys] where rank of keys[i] = start_rank + i.
    std::vector<PLASegment> build(
        const std::vector<Key>& keys,
        size_t start_rank = 0) const;

    // Build PLA model for a specific range within keys.
    std::vector<PLASegment> build_range(
        const std::vector<Key>& keys,
        size_t start_idx,
        size_t end_idx,
        size_t start_rank) const;

    double epsilon() const { return epsilon_; }

    // Check if a set of points can be covered by a single epsilon-bounded line.
    // Uses incremental convex polygon clipping in (slope, intercept) space.
    bool can_cover_single_segment(
        const std::vector<Key>& keys,
        size_t start_idx,
        size_t end_idx,
        size_t start_rank,
        double& out_slope,
        double& out_intercept) const;

private:
    double epsilon_;

    // Build one segment greedily starting at start_idx, return end_idx (inclusive)
    // and the chosen (slope, intercept).
    size_t build_one_segment(
        const std::vector<Key>& keys,
        size_t start_idx,
        size_t start_rank,
        double& out_slope,
        double& out_intercept) const;
};
