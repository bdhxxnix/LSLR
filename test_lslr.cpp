#include "lslr_pla.hpp"
#include "data_generator.hpp"
#include <iostream>

int main() {
    // Test LSLR on a simple dataset
    auto keys = DataGenerator::generate_keys(Distribution::UNIFORM, 10000, 1000000, 42);

    std::cout << "Testing LSLR-PLA on " << keys.size() << " uniform keys\n";
    std::cout << "Key range: [" << keys.front() << ", " << keys.back() << "]\n";

    std::vector<std::pair<std::string, LSLR_PLA::Config>> configs = {
        {"LSLR-c0", {64.0, 0, false, 8}},
        {"LSLR-c1", {64.0, 1, false, 8}},
        {"LSLR-c2", {64.0, 2, false, 8}},
        {"LSLR-adaptive", {64.0, 1, true, 8}},
    };

    // Generate insertions (10% of n)
    OptimalPLA init_pla(64.0);
    auto init_segs = init_pla.build(keys, 0);
    std::cout << "Initial segments: " << init_segs.size() << "\n";

    auto insert_keys = DataGenerator::generate_insertions(
        keys, 0.10, Workload::UNIFORM_RANDOM, init_segs, 123);
    std::cout << "Insertions: " << insert_keys.size() << "\n\n";

    for (const auto& [name, cfg] : configs) {
        LSLR_PLA lslr(cfg);
        lslr.build(keys);

        // Insert all keys
        for (size_t i = 0; i < insert_keys.size(); ++i) {
            auto rec = lslr.insert(insert_keys[i]);
            if (i < 3 || i >= insert_keys.size() - 3) {
                std::cout << "  " << name << " insert " << i
                          << ": pos=" << rec.position
                          << " seg=" << rec.segment_id
                          << " keys_repaired=" << rec.keys_repaired
                          << " segs_repaired=" << rec.segments_repaired
                          << " time=" << rec.insert_time_us << "us\n";
            }
        }

        auto [max_err, violations] = lslr.validate();
        std::cout << "  " << name << " final: "
                  << lslr.num_segments() << " segments, "
                  << "max_err=" << max_err << ", "
                  << "violations=" << violations;
        if (violations > 0) {
            std::cout << " FAIL\n";
            return 1;
        }
        std::cout << " OK\n";

        // Test a few point queries
        for (size_t i = 0; i < 3; ++i) {
            size_t idx = i * lslr.num_keys() / 3;
            Key k = lslr.keys()[idx];
            size_t pred = lslr.predict(k);
            std::cout << "    query key=" << k << " actual_rank=" << idx
                      << " predicted=" << pred << "\n";
        }
        std::cout << "\n";
    }

    std::cout << "All LSLR tests passed!\n";
    return 0;
}
