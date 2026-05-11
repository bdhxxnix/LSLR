#include "baselines.hpp"
#include <cmath>

// --- FullRebuildEach ---

FullRebuildEach::FullRebuildEach(double epsilon)
    : epsilon_(epsilon), pla_builder_(epsilon) {}

void FullRebuildEach::build(const std::vector<Key>& keys) {
    keys_ = keys;
    segments_ = pla_builder_.build(keys_, 0);
}

size_t FullRebuildEach::find_segment(Key x) const {
    if (segments_.empty()) return 0;
    // Binary search on first_key
    size_t lo = 0, hi = segments_.size() - 1;
    while (lo < hi) {
        size_t mid = lo + (hi - lo + 1) / 2;
        if (segments_[mid].first_key <= x) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

InsertionRecord FullRebuildEach::insert(Key x) {
    InsertionRecord rec;
    rec.key = x;
    Timer timer;

    auto it = std::lower_bound(keys_.begin(), keys_.end(), x);
    rec.position = static_cast<size_t>(it - keys_.begin());
    keys_.insert(it, x);

    // Full rebuild
    segments_ = pla_builder_.build(keys_, 0);

    rec.insert_time_us = timer.elapsed_us();
    rec.keys_repaired = keys_.size();
    rec.segments_repaired = segments_.size();
    return rec;
}

size_t FullRebuildEach::predict(Key x) const {
    if (keys_.empty()) return 0;
    size_t seg_id = find_segment(x);
    const auto& seg = segments_[seg_id];
    double pred = seg.slope * static_cast<double>(x) + seg.intercept;
    if (pred < 0) pred = 0;
    if (pred > static_cast<double>(keys_.size() - 1))
        pred = static_cast<double>(keys_.size() - 1);
    return static_cast<size_t>(std::round(pred));
}

std::pair<double, size_t> FullRebuildEach::validate() const {
    double max_err = 0.0;
    size_t violations = 0;
    for (size_t i = 0; i < keys_.size(); ++i) {
        size_t seg_id = find_segment(keys_[i]);
        const auto& seg = segments_[seg_id];
        double pred = seg.slope * static_cast<double>(keys_[i]) + seg.intercept;
        double err = std::abs(pred - static_cast<double>(i));
        max_err = std::max(max_err, err);
        if (err > epsilon_ + 1e-9) violations++;
    }
    return {max_err, violations};
}

// --- FullRebuildFinal ---

FullRebuildFinal::FullRebuildFinal(double epsilon)
    : epsilon_(epsilon), pla_builder_(epsilon) {}

void FullRebuildFinal::build(const std::vector<Key>& keys) {
    keys_ = keys;
    segments_ = pla_builder_.build(keys_, 0);
}

InsertionRecord FullRebuildFinal::insert(Key x) {
    InsertionRecord rec;
    rec.key = x;
    Timer timer;

    auto it = std::lower_bound(keys_.begin(), keys_.end(), x);
    rec.position = static_cast<size_t>(it - keys_.begin());
    keys_.insert(it, x);

    // Don't rebuild model - just insert key
    rec.insert_time_us = timer.elapsed_us();
    return rec;
}

void FullRebuildFinal::finalize() {
    segments_ = pla_builder_.build(keys_, 0);
    finalized_ = true;
}

size_t FullRebuildFinal::predict(Key x) const {
    if (keys_.empty() || !finalized_) return 0;
    size_t lo = 0, hi = segments_.size() - 1;
    while (lo < hi) {
        size_t mid = lo + (hi - lo + 1) / 2;
        if (segments_[mid].first_key <= x) lo = mid;
        else hi = mid - 1;
    }
    const auto& seg = segments_[lo];
    double pred = seg.slope * static_cast<double>(x) + seg.intercept;
    if (pred < 0) pred = 0;
    if (pred > static_cast<double>(keys_.size() - 1))
        pred = static_cast<double>(keys_.size() - 1);
    return static_cast<size_t>(std::round(pred));
}

std::pair<double, size_t> FullRebuildFinal::validate() const {
    double max_err = 0.0;
    size_t violations = 0;
    for (size_t i = 0; i < keys_.size(); ++i) {
        size_t lo = 0, hi = segments_.size() - 1;
        while (lo < hi) {
            size_t mid = lo + (hi - lo + 1) / 2;
            if (segments_[mid].first_key <= keys_[i]) lo = mid;
            else hi = mid - 1;
        }
        const auto& seg = segments_[lo];
        double pred = seg.slope * static_cast<double>(keys_[i]) + seg.intercept;
        double err = std::abs(pred - static_cast<double>(i));
        max_err = std::max(max_err, err);
        if (err > epsilon_ + 1e-9) violations++;
    }
    return {max_err, violations};
}

// --- PeriodicRebuild ---

PeriodicRebuild::PeriodicRebuild(double epsilon, size_t rebuild_interval)
    : epsilon_(epsilon), rebuild_interval_(rebuild_interval),
      pla_builder_(epsilon) {}

void PeriodicRebuild::build(const std::vector<Key>& keys) {
    keys_ = keys;
    segments_ = pla_builder_.build(keys_, 0);
    inserts_since_rebuild_ = 0;
}

void PeriodicRebuild::rebuild() {
    segments_ = pla_builder_.build(keys_, 0);
    inserts_since_rebuild_ = 0;
}

size_t PeriodicRebuild::find_segment(Key x) const {
    if (segments_.empty()) return 0;
    size_t lo = 0, hi = segments_.size() - 1;
    while (lo < hi) {
        size_t mid = lo + (hi - lo + 1) / 2;
        if (segments_[mid].first_key <= x) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

InsertionRecord PeriodicRebuild::insert(Key x) {
    InsertionRecord rec;
    rec.key = x;
    Timer timer;

    auto it = std::lower_bound(keys_.begin(), keys_.end(), x);
    rec.position = static_cast<size_t>(it - keys_.begin());
    keys_.insert(it, x);

    inserts_since_rebuild_++;

    if (inserts_since_rebuild_ >= rebuild_interval_) {
        rebuild();
        rec.keys_repaired = keys_.size();
        rec.segments_repaired = segments_.size();
    } else {
        rec.keys_repaired = 0;
        rec.segments_repaired = 0;
    }

    rec.insert_time_us = timer.elapsed_us();
    return rec;
}

size_t PeriodicRebuild::predict(Key x) const {
    if (keys_.empty()) return 0;
    size_t seg_id = find_segment(x);
    const auto& seg = segments_[seg_id];
    double pred = seg.slope * static_cast<double>(x) + seg.intercept;
    if (pred < 0) pred = 0;
    if (pred > static_cast<double>(keys_.size() - 1))
        pred = static_cast<double>(keys_.size() - 1);
    return static_cast<size_t>(std::round(pred));
}

std::pair<double, size_t> PeriodicRebuild::validate() const {
    double max_err = 0.0;
    size_t violations = 0;
    for (size_t i = 0; i < keys_.size(); ++i) {
        size_t seg_id = find_segment(keys_[i]);
        const auto& seg = segments_[seg_id];
        double pred = seg.slope * static_cast<double>(keys_[i]) + seg.intercept;
        double err = std::abs(pred - static_cast<double>(i));
        max_err = std::max(max_err, err);
        if (err > epsilon_ + 1e-9) violations++;
    }
    return {max_err, violations};
}
