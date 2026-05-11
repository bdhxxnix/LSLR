#pragma once

#include "common.hpp"

enum class Distribution {
    UNIFORM,
    NORMAL,
    LOGNORMAL,
    PIECEWISE,
    FB_200M     // SOSD fb_200M_uint64 dataset
};

enum class Workload {
    UNIFORM_RANDOM,     // A: uniform random insertions
    SAME_DISTRIBUTION,  // B: same distribution as original
    HOTSPOT,            // C: 80% insertions into 10% of key range
    APPEND,             // D: all keys larger than existing
    BOUNDARY            // E: insert near existing segment boundaries
};

class DataGenerator {
public:
    // Load all keys from a raw binary file (little-endian uint64).
    static std::vector<Key> load_binary_keys(const std::string& filepath);

    // Load a random sample of n keys from a raw binary file.
    // Uses reservoir sampling for uniform random sample without loading all keys.
    static std::vector<Key> load_sampled_keys(const std::string& filepath,
                                               size_t n, uint64_t seed = 42);

    // Load a contiguous range of keys [start_idx, start_idx + n) from binary file.
    static std::vector<Key> load_range_keys(const std::string& filepath,
                                             size_t start_idx, size_t n);

    // Generate sorted unique keys from a distribution.
    // key_range: upper bound for key values (keys in [0, key_range))
    static std::vector<Key> generate_keys(
        Distribution dist, size_t n, Key key_range = 0, uint64_t seed = 42);

    // Generate insertion keys according to a workload.
    // existing_keys: the current sorted keys (to avoid duplicates, append, boundary)
    // ratio: fraction of existing n to generate (e.g., 0.1 = 10%)
    // workload: insertion pattern
    // segments: existing PLA segments (needed for BOUNDARY workload)
    static std::vector<Key> generate_insertions(
        const std::vector<Key>& existing_keys,
        double ratio,
        Workload workload,
        const std::vector<PLASegment>& segments = {},
        uint64_t seed = 123);

private:
    static std::vector<Key> make_unique_sorted(std::vector<Key>& data);
};
