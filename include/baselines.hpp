#pragma once

#include "common.hpp"
#include "optimal_pla.hpp"

// Baseline 1: Full rebuild after each insertion.
class FullRebuildEach {
public:
    FullRebuildEach(double epsilon);

    void build(const std::vector<Key>& keys);
    InsertionRecord insert(Key x);
    size_t predict(Key x) const;
    size_t num_segments() const { return segments_.size(); }
    size_t num_keys() const { return keys_.size(); }
    const std::vector<PLASegment>& segments() const { return segments_; }
    std::pair<double, size_t> validate() const;

private:
    double epsilon_;
    std::vector<Key> keys_;
    std::vector<PLASegment> segments_;
    OptimalPLA pla_builder_;
    size_t find_segment(Key x) const;
};

// Baseline 2: Full rebuild once after all insertions (oracle for final quality).
class FullRebuildFinal {
public:
    FullRebuildFinal(double epsilon);

    void build(const std::vector<Key>& keys);
    InsertionRecord insert(Key x);  // just inserts key, no model update
    void finalize();  // rebuild model after all insertions
    size_t predict(Key x) const;
    size_t num_segments() const { return segments_.size(); }
    size_t num_keys() const { return keys_.size(); }
    const std::vector<PLASegment>& segments() const { return segments_; }
    std::pair<double, size_t> validate() const;

private:
    double epsilon_;
    std::vector<Key> keys_;
    std::vector<PLASegment> segments_;
    OptimalPLA pla_builder_;
    bool finalized_{false};
    size_t find_segment(Key x) const;
};

// Baseline 3: Periodic rebuild after every B insertions.
class PeriodicRebuild {
public:
    PeriodicRebuild(double epsilon, size_t rebuild_interval);

    void build(const std::vector<Key>& keys);
    InsertionRecord insert(Key x);
    size_t predict(Key x) const;
    size_t num_segments() const { return segments_.size(); }
    size_t num_keys() const { return keys_.size(); }
    const std::vector<PLASegment>& segments() const { return segments_; }
    std::pair<double, size_t> validate() const;

private:
    double epsilon_;
    size_t rebuild_interval_;
    size_t inserts_since_rebuild_{0};
    std::vector<Key> keys_;
    std::vector<PLASegment> segments_;
    OptimalPLA pla_builder_;
    size_t find_segment(Key x) const;
    void rebuild();
};
