#pragma once
// Wrapper around DynamicLearnedIndex (Sgelet et al.) to avoid name conflicts.
// Accepts epsilon at construction; dispatches to correct template instantiation.

#include <cstdint>
#include <cstddef>
#include <vector>
#include <utility>

class BaselineIndex {
public:
    explicit BaselineIndex(int epsilon);
    ~BaselineIndex();

    bool insert(int64_t key);
    bool find(int64_t key);
    int segment_count() const;
    size_t memory_bytes() const;
    std::vector<std::pair<int64_t, int64_t>> segment_ranges() const;

    int epsilon() const { return epsilon_; }

private:
    struct Impl;
    Impl* impl_;
    int epsilon_;
};
