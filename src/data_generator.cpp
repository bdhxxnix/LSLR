#include "data_generator.hpp"
#include <random>
#include <algorithm>
#include <set>
#include <cmath>
#include <fstream>
#include <stdexcept>

// --- Binary dataset loading ---

std::vector<Key> DataGenerator::load_binary_keys(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open " + filepath);

    size_t file_size = static_cast<size_t>(f.tellg());
    size_t n = file_size / sizeof(Key);
    std::vector<Key> keys(n);

    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(keys.data()), static_cast<std::streamsize>(file_size));
    if (!f) throw std::runtime_error("Failed to read " + filepath);

    // Verify sorted order (fb_200M is pre-sorted)
    bool sorted = true;
    for (size_t i = 1; i < n; ++i) {
        if (keys[i] < keys[i-1]) { sorted = false; break; }
    }
    if (!sorted) {
        std::sort(keys.begin(), keys.end());
    }
    return keys;
}

std::vector<Key> DataGenerator::load_sampled_keys(const std::string& filepath,
                                                    size_t n, uint64_t seed) {
    std::ifstream f(filepath, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open " + filepath);

    size_t file_size = static_cast<size_t>(f.tellg());
    size_t total_n = file_size / sizeof(Key);

    if (n >= total_n) return load_binary_keys(filepath);

    std::mt19937_64 rng(seed);

    // Reservoir sampling for random sample
    std::vector<Key> reservoir;
    reservoir.reserve(n);
    f.seekg(0, std::ios::beg);

    // Read first n as initial reservoir
    for (size_t i = 0; i < n; ++i) {
        Key k;
        f.read(reinterpret_cast<char*>(&k), sizeof(Key));
        reservoir.push_back(k);
    }

    // Replace elements with decreasing probability
    std::uniform_int_distribution<size_t> dist(0, 0);  // placeholder
    for (size_t i = n; i < total_n; ++i) {
        Key k;
        f.read(reinterpret_cast<char*>(&k), sizeof(Key));
        size_t j = std::uniform_int_distribution<size_t>(0, i)(rng);
        if (j < n) {
            reservoir[j] = k;
        }
    }

    std::sort(reservoir.begin(), reservoir.end());
    return reservoir;
}

std::vector<Key> DataGenerator::load_range_keys(const std::string& filepath,
                                                  size_t start_idx, size_t n) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + filepath);

    f.seekg(static_cast<std::streamoff>(start_idx * sizeof(Key)), std::ios::beg);
    std::vector<Key> keys(n);
    f.read(reinterpret_cast<char*>(keys.data()), static_cast<std::streamsize>(n * sizeof(Key)));
    if (!f) throw std::runtime_error("Failed to read range from " + filepath);

    // Verify sorted
    bool sorted = true;
    for (size_t i = 1; i < n; ++i) {
        if (keys[i] < keys[i-1]) { sorted = false; break; }
    }
    if (!sorted) std::sort(keys.begin(), keys.end());
    return keys;
}

std::vector<Key> DataGenerator::make_unique_sorted(std::vector<Key>& data) {
    std::sort(data.begin(), data.end());
    data.erase(std::unique(data.begin(), data.end()), data.end());
    return data;
}

std::vector<Key> DataGenerator::generate_keys(
    Distribution dist, size_t n, Key key_range, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::vector<Key> keys;
    keys.reserve(n);

    switch (dist) {
    case Distribution::FB_200M: {
        // Load random sample from fb_200M_uint64 dataset
        const std::string fb_path = "/home/andy/Projects/Datasets/SOSD/fb_200M_uint64";
        keys = load_sampled_keys(fb_path, n, seed);
        break;
    }
    case Distribution::UNIFORM: {
        std::uniform_int_distribution<Key> uid(0, key_range - 1);
        std::set<Key> seen;
        while (keys.size() < n) {
            Key k = uid(rng);
            if (seen.insert(k).second) {
                keys.push_back(k);
            }
        }
        break;
    }
    case Distribution::NORMAL: {
        double mean = key_range / 2.0;
        double stddev = key_range / 6.0;
        std::normal_distribution<double> nd(mean, stddev);
        std::set<Key> seen;
        while (keys.size() < n) {
            double val = nd(rng);
            if (val < 0 || val >= static_cast<double>(key_range)) continue;
            Key k = static_cast<Key>(val);
            if (seen.insert(k).second) {
                keys.push_back(k);
            }
        }
        break;
    }
    case Distribution::LOGNORMAL: {
        // Lognormal with parameters such that most values are in [0, key_range)
        double mu = std::log(key_range / 2.0);
        double sigma = 1.5;
        std::lognormal_distribution<double> lnd(mu, sigma);
        std::set<Key> seen;
        while (keys.size() < n) {
            double val = lnd(rng);
            if (val >= static_cast<double>(key_range)) continue;
            Key k = static_cast<Key>(val);
            if (seen.insert(k).second) {
                keys.push_back(k);
            }
        }
        break;
    }
    case Distribution::PIECEWISE: {
        // Four regions with different characteristics
        // Region 1 [0, 25%]: sparse linear (uniform)
        // Region 2 [25%, 35%]: dense cluster (normal with small stddev)
        // Region 3 [35%, 75%]: medium density (uniform)
        // Region 4 [75%, 100%]: another dense cluster
        std::set<Key> seen;

        size_t n1 = n / 4;       // 25% in region 1 (sparse)
        size_t n2 = n / 3;       // 33% in region 2 (dense cluster)
        size_t n3 = n / 4;       // 25% in region 3 (medium)
        size_t n4 = n - n1 - n2 - n3; // remainder in region 4

        Key r1_start = 0, r1_end = key_range / 4;
        Key r2_start = r1_end, r2_end = key_range * 35 / 100;
        Key r3_start = r2_end, r3_end = key_range * 75 / 100;
        Key r4_start = r3_end, r4_end = key_range;

        std::uniform_int_distribution<Key> uid1(r1_start, r1_end - 1);
        while (keys.size() < n1) {
            Key k = uid1(rng);
            if (seen.insert(k).second) keys.push_back(k);
        }

        // Dense cluster in region 2
        double mean2 = (r2_start + r2_end) / 2.0;
        double stddev2 = (r2_end - r2_start) / 8.0;
        std::normal_distribution<double> nd2(mean2, stddev2);
        while (keys.size() < n1 + n2) {
            double val = nd2(rng);
            if (val < r2_start || val >= r2_end) continue;
            Key k = static_cast<Key>(val);
            if (seen.insert(k).second) keys.push_back(k);
        }

        std::uniform_int_distribution<Key> uid3(r3_start, r3_end - 1);
        while (keys.size() < n1 + n2 + n3) {
            Key k = uid3(rng);
            if (seen.insert(k).second) keys.push_back(k);
        }

        // Dense cluster in region 4
        double mean4 = (r4_start + r4_end) / 2.0;
        double stddev4 = (r4_end - r4_start) / 10.0;
        std::normal_distribution<double> nd4(mean4, stddev4);
        while (keys.size() < n) {
            double val = nd4(rng);
            if (val < r4_start || val >= r4_end) continue;
            Key k = static_cast<Key>(val);
            if (seen.insert(k).second) keys.push_back(k);
        }
        break;
    }
    }

    std::sort(keys.begin(), keys.end());
    return keys;
}

std::vector<Key> DataGenerator::generate_insertions(
    const std::vector<Key>& existing_keys,
    double ratio,
    Workload workload,
    const std::vector<PLASegment>& segments,
    uint64_t seed)
{
    std::mt19937_64 rng(seed);

    size_t n_existing = existing_keys.size();
    size_t n_insert = static_cast<size_t>(n_existing * ratio);
    if (n_insert == 0) n_insert = 1;

    std::set<Key> existing_set(existing_keys.begin(), existing_keys.end());
    std::vector<Key> insertions;
    insertions.reserve(n_insert);

    Key min_key = existing_keys.front();
    Key max_key = existing_keys.back();
    Key range = max_key - min_key + 1;

    std::set<Key> seen;

    auto generate_unique_key = [&](auto& dist_fn) {
        for (size_t attempts = 0; attempts < 10000; ++attempts) {
            Key k = dist_fn(rng);
            if (existing_set.count(k) == 0 && seen.count(k) == 0) {
                seen.insert(k);
                return k;
            }
        }
        // Fallback: generate very large key
        Key k = max_key + seen.size() + 1;
        while (existing_set.count(k) || seen.count(k)) {
            k++;
        }
        seen.insert(k);
        return k;
    };

    switch (workload) {
    case Workload::UNIFORM_RANDOM: {
        std::uniform_int_distribution<Key> uid(min_key, max_key);
        while (insertions.size() < n_insert) {
            Key k = generate_unique_key(uid);
            insertions.push_back(k);
        }
        break;
    }
    case Workload::SAME_DISTRIBUTION: {
        // Sample from empirical distribution: pick a random existing key
        // and generate nearby key
        std::uniform_int_distribution<size_t> idx_dist(0, n_existing - 1);
        std::exponential_distribution<double> exp_dist(1.0 / (range / n_existing));
        while (insertions.size() < n_insert) {
            size_t idx = idx_dist(rng);
            Key base = existing_keys[idx];
            double offset = exp_dist(rng);
            // Random sign
            if (rng() % 2 == 0 && offset <= static_cast<double>(base)) {
                Key k = static_cast<Key>(base - static_cast<Key>(offset));
                if (k < min_key) k = min_key;
                if (existing_set.count(k) == 0 && seen.insert(k).second) {
                    insertions.push_back(k);
                }
            } else {
                Key k = static_cast<Key>(base + static_cast<Key>(offset));
                if (k > max_key) k = max_key;
                if (existing_set.count(k) == 0 && seen.insert(k).second) {
                    insertions.push_back(k);
                }
            }
        }
        break;
    }
    case Workload::HOTSPOT: {
        // 80% into 10% of range — pick a random hotspot region
        Key hotspot_start = min_key + (rng() % static_cast<Key>(range * 0.9));
        Key hotspot_end = hotspot_start + static_cast<Key>(range * 0.1);
        if (hotspot_end > max_key) hotspot_end = max_key;

        size_t n_hotspot = n_insert * 80 / 100;
        size_t n_rest = n_insert - n_hotspot;

        std::uniform_int_distribution<Key> hotspot_dist(hotspot_start, hotspot_end);
        while (insertions.size() < n_hotspot) {
            Key k = generate_unique_key(hotspot_dist);
            insertions.push_back(k);
        }

        std::uniform_int_distribution<Key> rest_dist(min_key, max_key);
        while (insertions.size() < n_hotspot + n_rest) {
            Key k = generate_unique_key(rest_dist);
            insertions.push_back(k);
        }
        break;
    }
    case Workload::APPEND: {
        // All new keys > max_key
        Key start = max_key + 1;
        // Spread append keys evenly over some range
        Key append_range = range / 10;  // 10% of original range
        if (append_range < n_insert) append_range = n_insert * 2;
        std::uniform_int_distribution<Key> append_dist(start, start + append_range);
        while (insertions.size() < n_insert) {
            Key k = append_dist(rng);
            if (seen.insert(k).second) {
                insertions.push_back(k);
            }
        }
        break;
    }
    case Workload::BOUNDARY: {
        // Generate keys near segment boundaries.
        // Build a pool of candidate boundary-adjacent keys, sample from it.
        if (segments.empty()) {
            std::uniform_int_distribution<Key> uid(min_key, max_key);
            while (insertions.size() < n_insert) {
                Key k = generate_unique_key(uid);
                insertions.push_back(k);
            }
        } else {
            // Collect all possible boundary-adjacent keys
            std::vector<Key> boundary_pool;
            for (const auto& seg : segments) {
                Key bk = seg.last_key;
                // keys near this boundary (±50)
                for (int64_t off = -50; off <= 50; ++off) {
                    int64_t nk = static_cast<int64_t>(bk) + off;
                    if (nk >= static_cast<int64_t>(min_key) &&
                        nk <= static_cast<int64_t>(max_key)) {
                        Key k = static_cast<Key>(nk);
                        if (existing_set.count(k) == 0) {
                            boundary_pool.push_back(k);
                        }
                    }
                }
            }
            // Remove duplicates from pool
            std::sort(boundary_pool.begin(), boundary_pool.end());
            boundary_pool.erase(std::unique(boundary_pool.begin(), boundary_pool.end()),
                                boundary_pool.end());

            // Sample from pool (with shuffle), fall back to uniform if pool exhausted
            std::shuffle(boundary_pool.begin(), boundary_pool.end(), rng);

            size_t pool_idx = 0;
            std::uniform_int_distribution<Key> uid(min_key, max_key);

            while (insertions.size() < n_insert) {
                Key k;
                if (pool_idx < boundary_pool.size()) {
                    k = boundary_pool[pool_idx++];
                    if (seen.count(k) > 0) continue;
                } else {
                    k = generate_unique_key(uid);
                }
                seen.insert(k);
                insertions.push_back(k);
            }
        }
        break;
    }
    }

    std::sort(insertions.begin(), insertions.end());
    return insertions;
}
