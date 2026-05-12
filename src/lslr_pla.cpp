#include "lslr_pla.hpp"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>

// --- FenwickTree ---

void FenwickTree::add_(size_t idx, int64_t delta) {
    idx += 1;
    while (idx < tree_.size()) {
        tree_[idx] += delta;
        idx += idx & -idx;
    }
}

int64_t FenwickTree::prefix_sum_(size_t idx) const {
    int64_t sum = 0;
    idx += 1;
    while (idx > 0) {
        sum += tree_[idx];
        idx -= idx & -idx;
    }
    return sum;
}

void FenwickTree::range_add(size_t l, size_t r, int64_t delta) {
    if (l > r || l >= tree_.size() - 1) return;
    r = std::min(r, tree_.size() - 2);
    add_(l, delta);
    add_(r + 1, -delta);
}

int64_t FenwickTree::point_query(size_t idx) const {
    return prefix_sum_(idx);
}

void FenwickTree::clear() {
    std::fill(tree_.begin(), tree_.end(), 0);
}

// --- LSLR_PLA ---

LSLR_PLA::LSLR_PLA(const Config& config)
    : config_(config)
    , offset_tree_(0)
    , pla_builder_(config.epsilon)
{}

LSLR_PLA::LSLR_PLA(double epsilon, int window_radius, bool adaptive)
    : offset_tree_(0)
    , pla_builder_(epsilon)
{
    config_.epsilon = epsilon;
    config_.window_radius = window_radius;
    config_.adaptive = adaptive;
}

void LSLR_PLA::build(const std::vector<Key>& keys) {
    keys_ = keys;
    segments_ = pla_builder_.build(keys_, 0);

    // Build segment first-key index
    segment_first_keys_.clear();
    segment_first_keys_.reserve(segments_.size());
    for (const auto& seg : segments_) {
        segment_first_keys_.push_back(seg.first_key);
    }

    offset_tree_ = FenwickTree(segments_.size());
}

size_t LSLR_PLA::find_segment(Key x) const {
    if (segment_first_keys_.empty()) return 0;
    auto it = std::upper_bound(segment_first_keys_.begin(),
                               segment_first_keys_.end(), x);
    if (it == segment_first_keys_.begin()) return 0;
    size_t idx = static_cast<size_t>(it - segment_first_keys_.begin() - 1);
    return std::min(idx, segments_.size() - 1);
}

size_t LSLR_PLA::find_segment_by_rank(size_t rank) const {
    // Key-based lookup: find the segment containing the key at this position.
    // If inserting at end, rank == keys_.size(), return last segment.
    if (rank >= keys_.size()) {
        return segments_.empty() ? 0 : segments_.size() - 1;
    }
    return find_segment(keys_[rank]);
}

double LSLR_PLA::effective_intercept(size_t seg_id) const {
    return segments_[seg_id].intercept +
           static_cast<double>(offset_tree_.point_query(seg_id));
}

size_t LSLR_PLA::predict(Key x) const {
    if (keys_.empty()) return 0;
    size_t seg_id = find_segment(x);
    const auto& seg = segments_[seg_id];
    long double pred = seg.slope * static_cast<long double>(x) + seg.intercept;
    if (pred < 0) pred = 0;
    if (pred > static_cast<long double>(keys_.size() - 1))
        pred = static_cast<long double>(keys_.size() - 1);
    return static_cast<size_t>(std::round(pred));
}

InsertionRecord LSLR_PLA::insert(Key x) {
    InsertionRecord rec;
    rec.key = x;
    Timer timer;

    // Step 1: Find insertion rank
    auto it = std::lower_bound(keys_.begin(), keys_.end(), x);
    size_t p = static_cast<size_t>(it - keys_.begin());

    // Handle empty case
    if (keys_.empty()) {
        rec.position = 0;
        rec.segment_id = 0;
        keys_.push_back(x);
        segments_.clear();
        segment_first_keys_.clear();
        PLASegment seg;
        seg.start = 0; seg.end = 0;
        seg.first_key = x; seg.last_key = x;
        seg.slope = 0; seg.intercept = 0;
        segments_.push_back(seg);
        segment_first_keys_.push_back(x);
        offset_tree_ = FenwickTree(1);
        rec.insert_time_us = timer.elapsed_us();
        return rec;
    }

    rec.position = p;

    // Step 2: Insert x into global array permanently.
    keys_.insert(it, x);

    // Step 3: Find affected segment q using key-based lookup on the inserted key.
    size_t q = find_segment(x);
    rec.segment_id = q;

    // Step 4: Update segment ranks and key ranges to reflect the insertion.
    // Segment q contains the insertion point — extend its end and possibly last_key.
    segments_[q].end += 1;
    if (x > segments_[q].last_key) {
        segments_[q].last_key = x;
    }
    // Segments after q are entirely after the insertion — shift start/end.
    for (size_t j = q + 1; j < segments_.size(); ++j) {
        segments_[j].start += 1;
        segments_[j].end += 1;
    }

    // Step 5: Apply intercept shift for segments entirely after q.
    for (size_t j = q + 1; j < segments_.size(); ++j) {
        segments_[j].intercept += 1.0;
    }

    // Step 6: Select local repair window around q
    int c = config_.window_radius;
    size_t left = (q >= static_cast<size_t>(c)) ? q - c : 0;
    size_t right = std::min(q + static_cast<size_t>(c), segments_.size() - 1);

    // Step 7: Extract keys in the window from the updated array
    Key key_L = segments_[left].first_key;
    Key key_R = segments_[right].last_key;

    auto it_L = std::lower_bound(keys_.begin(), keys_.end(), key_L);
    auto it_R = std::upper_bound(keys_.begin(), keys_.end(), key_R);
    size_t idx_L = static_cast<size_t>(it_L - keys_.begin());
    size_t idx_R = static_cast<size_t>(it_R - keys_.begin());
    if (idx_R > keys_.size()) idx_R = keys_.size();

    std::vector<Key> local_keys;
    local_keys.reserve(idx_R - idx_L + 1);
    for (size_t i = idx_L; i < idx_R; ++i) {
        local_keys.push_back(keys_[i]);
    }

    // Step 8: Rebuild OptimalPLA on local window
    std::vector<PLASegment> new_local_segments =
        pla_builder_.build_range(local_keys, 0, local_keys.size(), idx_L);

    rec.keys_repaired = local_keys.size();
    rec.segments_repaired = new_local_segments.size();

    // Step 9: Adaptive expansion if enabled
    if (config_.adaptive && !new_local_segments.empty()) {
        int max_iter = config_.max_window_radius;
        while (max_iter-- > 0) {
            bool did_expand = false;

            if (left > 0 && !new_local_segments.empty()) {
                PLASegment merged;
                if (try_merge(segments_[left - 1], new_local_segments.front(), merged)) {
                    left--;
                    key_L = segments_[left].first_key;
                    did_expand = true;
                }
            }

            if (right + 1 < segments_.size() && !new_local_segments.empty()) {
                PLASegment merged;
                if (try_merge(new_local_segments.back(), segments_[right + 1], merged)) {
                    right++;
                    key_R = segments_[right].last_key;
                    did_expand = true;
                }
            }

            if (!did_expand) break;

            rec.window_expanded = true;
            rec.window_expansions++;

            // Re-extract keys with expanded window
            it_L = std::lower_bound(keys_.begin(), keys_.end(), key_L);
            it_R = std::upper_bound(keys_.begin(), keys_.end(), key_R);
            idx_L = static_cast<size_t>(it_L - keys_.begin());
            idx_R = static_cast<size_t>(it_R - keys_.begin());
            if (idx_R > keys_.size()) idx_R = keys_.size();

            local_keys.clear();
            for (size_t i = idx_L; i < idx_R; ++i) {
                local_keys.push_back(keys_[i]);
            }
            new_local_segments = pla_builder_.build_range(
                local_keys, 0, local_keys.size(), idx_L);
        }
    }

    rec.keys_repaired = local_keys.size();
    rec.segments_repaired = new_local_segments.size();

    // Step 10: Replace old segments [left, right] with new ones
    replace_segments(left, right, new_local_segments);

    rec.insert_time_us = timer.elapsed_us();
    return rec;
}

void LSLR_PLA::replace_segments(size_t left, size_t right,
                                 const std::vector<PLASegment>& new_segments)
{
    size_t old_count = segments_.size();
    size_t removed_count = right - left + 1;
    size_t new_count = new_segments.size();
    size_t new_total = old_count - removed_count + new_count;

    std::vector<PLASegment> new_segs;
    new_segs.reserve(new_total);

    for (size_t i = 0; i < left; ++i) {
        new_segs.push_back(segments_[i]);
    }
    for (const auto& seg : new_segments) {
        new_segs.push_back(seg);
    }
    for (size_t i = right + 1; i < old_count; ++i) {
        new_segs.push_back(segments_[i]);
    }

    segments_ = std::move(new_segs);

    // Rebuild segment_first_keys
    segment_first_keys_.clear();
    segment_first_keys_.reserve(segments_.size());
    for (const auto& seg : segments_) {
        segment_first_keys_.push_back(seg.first_key);
    }
}

bool LSLR_PLA::try_merge(const PLASegment& seg_a, const PLASegment& seg_b,
                          PLASegment& merged) const
{
    // Find the key indices
    auto it_a = std::lower_bound(keys_.begin(), keys_.end(), seg_a.first_key);
    auto it_b = std::upper_bound(keys_.begin(), keys_.end(), seg_b.last_key);

    if (it_a == keys_.end() || it_b == keys_.begin()) return false;

    size_t start_idx = static_cast<size_t>(it_a - keys_.begin());
    size_t end_idx = static_cast<size_t>(it_b - keys_.begin());
    if (end_idx > keys_.size()) end_idx = keys_.size();
    if (end_idx <= start_idx) return false;

    // The rank of the first key is start_idx
    size_t start_rank = start_idx;

    // Build combined key list
    std::vector<Key> combined_keys;
    for (size_t i = start_idx; i < end_idx; ++i) {
        combined_keys.push_back(keys_[i]);
    }

    double slope, intercept;
    bool ok = pla_builder_.can_cover_single_segment(
        combined_keys, 0, combined_keys.size() - 1, start_rank, slope, intercept);

    if (ok) {
        merged.start = start_rank;
        merged.end = start_rank + combined_keys.size() - 1;
        merged.first_key = combined_keys.front();
        merged.last_key = combined_keys.back();
        merged.slope = slope;
        merged.intercept = intercept;
    }

    return ok;
}

std::pair<double, size_t> LSLR_PLA::validate() const {
    double max_err = 0.0;
    size_t violations = 0;

    for (size_t i = 0; i < keys_.size(); ++i) {
        size_t seg_id = find_segment(keys_[i]);
        const auto& seg = segments_[seg_id];
        long double pred = seg.slope * static_cast<long double>(keys_[i]) + seg.intercept;
        long double err = std::abs(pred - static_cast<long double>(i));
        if (err > max_err) max_err = err;
        if (err > config_.epsilon + 1e-9) {
            violations++;
        }
    }

    return {max_err, violations};
}
