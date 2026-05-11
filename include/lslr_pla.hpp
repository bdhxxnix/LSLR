#pragma once

#include "common.hpp"
#include "optimal_pla.hpp"

// Fenwick tree for point queries and range adds on segment offsets.
class FenwickTree {
public:
    explicit FenwickTree(size_t n) : tree_(n + 2, 0) {}
    void range_add(size_t l, size_t r, int64_t delta);
    int64_t point_query(size_t idx) const;
    size_t size() const { return tree_.size() - 2; }
    void clear();
private:
    std::vector<int64_t> tree_;
    void add_(size_t idx, int64_t delta);
    int64_t prefix_sum_(size_t idx) const;
};

// LSLR-PLA: Lazy-Shift Local-Repair Piecewise Linear Approximation.
// Supports in-place insertions with local repair instead of full rebuild.
class LSLR_PLA {
public:
    struct Config {
        double epsilon = 64.0;
        int window_radius = 1;   // c=0: affected only, c=1: +1 neighbor, etc.
        bool adaptive = false;    // enable adaptive window expansion
        int max_window_radius = 8; // cap on adaptive expansion
    };

    explicit LSLR_PLA(const Config& config);
    LSLR_PLA(double epsilon, int window_radius = 1, bool adaptive = false);

    // Build initial model from sorted keys.
    void build(const std::vector<Key>& keys);

    // Insert a single key, updating the model in-place.
    // Returns insertion statistics.
    InsertionRecord insert(Key x);

    // Query: predict rank for a key.
    size_t predict(Key x) const;

    // Accessors
    const std::vector<Key>& keys() const { return keys_; }
    const std::vector<PLASegment>& segments() const { return segments_; }
    size_t num_segments() const { return segments_.size(); }
    size_t num_keys() const { return keys_.size(); }
    double epsilon() const { return config_.epsilon; }
    int window_radius() const { return config_.window_radius; }
    bool is_adaptive() const { return config_.adaptive; }

    // Validate the model: returns (max_error, num_violations).
    std::pair<double, size_t> validate() const;

    // Get effective intercept for a segment (applying lazy shift).
    double effective_intercept(size_t seg_id) const;

private:
    Config config_;
    std::vector<Key> keys_;
    std::vector<PLASegment> segments_;
    std::vector<Key> segment_first_keys_;  // for binary search lookup
    FenwickTree offset_tree_;
    OptimalPLA pla_builder_;

    // Find segment containing a given key.
    size_t find_segment(Key x) const;

    // Find segment by rank position.
    size_t find_segment_by_rank(size_t rank) const;

    // Replace segments [left, right] with new_segments, updating all metadata.
    void replace_segments(size_t left, size_t right,
                          const std::vector<PLASegment>& new_segments);

    // Try to merge seg_a (left) with seg_b (right).
    // Returns true and fills merged if successful.
    bool try_merge(const PLASegment& seg_a, const PLASegment& seg_b,
                   PLASegment& merged) const;
};
