#include "common.hpp"
#include "optimal_pla.hpp"
#include "lslr_pla.hpp"
#include "baselines.hpp"
#include "data_generator.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>

struct BenchResult {
    std::string method;
    // Correctness
    size_t initial_segments;
    size_t final_segments;
    size_t optimal_final_segments;
    double segment_overhead;
    double max_error;
    size_t violations;
    // Update cost
    double total_time_s;
    double avg_time_us;
    double p50_time_us;
    double p95_time_us;
    double p99_time_us;
    // Repair stats (LSLR only)
    double avg_keys_repaired;
    double avg_segs_repaired;
};

void print_separator(int width = 100) {
    std::cout << std::string(width, '-') << "\n";
}

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(100, '=') << "\n\n";
}

int main() {
    double epsilon = 64.0;
    size_t n_initial = 100000;
    double insert_ratio = 0.05;  // 5% insertions = 5000 keys
    size_t n_insert;
    uint64_t seed = 42;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Config: n=" << n_initial << ", eps=" << epsilon
              << ", insert_ratio=" << insert_ratio << "\n";

    // Load fb_200M dataset
    std::cout << "\nLoading fb_200M_uint64 dataset... " << std::flush;
    auto keys = DataGenerator::load_sampled_keys(
        "/home/andy/Projects/Datasets/SOSD/fb_200M_uint64", n_initial, seed);
    std::cout << keys.size() << " keys, range ["
              << keys.front() << ", " << keys.back() << "]\n";

    // Build initial model for segment count reference
    OptimalPLA pla(epsilon);
    auto init_segs = pla.build(keys, 0);
    std::cout << "Initial OptimalPLA: " << init_segs.size() << " segments\n";

    // Generate insertions
    auto insert_keys = DataGenerator::generate_insertions(
        keys, insert_ratio, Workload::UNIFORM_RANDOM, init_segs, 123);
    n_insert = insert_keys.size();
    std::cout << "Insertions: " << n_insert << " keys\n";

    // Compute oracle (FullRebuildFinal)
    std::vector<Key> final_keys = keys;
    for (auto k : insert_keys) final_keys.push_back(k);
    std::sort(final_keys.begin(), final_keys.end());
    size_t optimal_segments = pla.build(final_keys, 0).size();
    std::cout << "Optimal final segments (oracle): " << optimal_segments << "\n";

    std::vector<BenchResult> results;

    // ========================================
    // 1. FullRebuildEach
    // ========================================
    {
        std::cout << "\n[FullRebuildEach] Running... " << std::flush;
        FullRebuildEach fre(epsilon);
        fre.build(keys);

        std::vector<double> times;
        times.reserve(n_insert);
        Timer total;

        for (size_t i = 0; i < n_insert; ++i) {
            Timer t;
            fre.insert(insert_keys[i]);
            times.push_back(t.elapsed_us());
        }

        double total_s = total.elapsed_us() / 1e6;
        std::sort(times.begin(), times.end());

        auto [max_err, viol] = fre.validate();

        BenchResult r;
        r.method = "FullRebuildEach";
        r.initial_segments = init_segs.size();
        r.final_segments = fre.num_segments();
        r.optimal_final_segments = optimal_segments;
        r.segment_overhead = static_cast<double>(fre.num_segments()) / optimal_segments;
        r.max_error = max_err;
        r.violations = viol;
        r.total_time_s = total_s;
        r.avg_time_us = total_s / n_insert * 1e6;
        r.p50_time_us = percentile(times, 50);
        r.p95_time_us = percentile(times, 95);
        r.p99_time_us = percentile(times, 99);
        r.avg_keys_repaired = 0;
        r.avg_segs_repaired = 0;
        results.push_back(r);
        std::cout << "done (" << total_s << "s)\n";
    }

    // ========================================
    // 2. PeriodicRebuild
    // ========================================
    {
        size_t interval = std::max(size_t(1), n_initial / 100); // every 1%
        std::cout << "\n[PeriodicRebuild (every " << interval << " inserts)] Running... " << std::flush;
        PeriodicRebuild pr(epsilon, interval);
        pr.build(keys);

        std::vector<double> times;
        times.reserve(n_insert);
        Timer total;

        for (size_t i = 0; i < n_insert; ++i) {
            Timer t;
            pr.insert(insert_keys[i]);
            times.push_back(t.elapsed_us());
        }

        double total_s = total.elapsed_us() / 1e6;
        std::sort(times.begin(), times.end());

        auto [max_err, viol] = pr.validate();

        BenchResult r;
        r.method = "PeriodicRebuild";
        r.initial_segments = init_segs.size();
        r.final_segments = pr.num_segments();
        r.optimal_final_segments = optimal_segments;
        r.segment_overhead = static_cast<double>(pr.num_segments()) / optimal_segments;
        r.max_error = max_err;
        r.violations = viol;
        r.total_time_s = total_s;
        r.avg_time_us = total_s / n_insert * 1e6;
        r.p50_time_us = percentile(times, 50);
        r.p95_time_us = percentile(times, 95);
        r.p99_time_us = percentile(times, 99);
        r.avg_keys_repaired = 0;
        r.avg_segs_repaired = 0;
        results.push_back(r);
        std::cout << "done (" << total_s << "s)\n";
    }

    // ========================================
    // 3-6. LSLR variants
    // ========================================
    std::vector<std::pair<std::string, LSLR_PLA::Config>> lslr_configs = {
        {"LSLR-c0", {epsilon, 0, false, 8}},
        {"LSLR-c1", {epsilon, 1, false, 8}},
        {"LSLR-c2", {epsilon, 2, false, 8}},
        {"LSLR-adaptive", {epsilon, 1, true, 8}},
    };

    for (const auto& [name, cfg] : lslr_configs) {
        std::cout << "\n[" << name << "] Running... " << std::flush;
        LSLR_PLA lslr(cfg);
        lslr.build(keys);

        std::vector<double> times;
        times.reserve(n_insert);
        size_t total_keys_rep = 0, total_segs_rep = 0;
        Timer total;

        for (size_t i = 0; i < n_insert; ++i) {
            Timer t;
            auto rec = lslr.insert(insert_keys[i]);
            times.push_back(t.elapsed_us());
            total_keys_rep += rec.keys_repaired;
            total_segs_rep += rec.segments_repaired;
        }

        double total_s = total.elapsed_us() / 1e6;
        std::sort(times.begin(), times.end());

        auto [max_err, viol] = lslr.validate();

        BenchResult r;
        r.method = name;
        r.initial_segments = init_segs.size();
        r.final_segments = lslr.num_segments();
        r.optimal_final_segments = optimal_segments;
        r.segment_overhead = static_cast<double>(lslr.num_segments()) / optimal_segments;
        r.max_error = max_err;
        r.violations = viol;
        r.total_time_s = total_s;
        r.avg_time_us = total_s / n_insert * 1e6;
        r.p50_time_us = percentile(times, 50);
        r.p95_time_us = percentile(times, 95);
        r.p99_time_us = percentile(times, 99);
        r.avg_keys_repaired = static_cast<double>(total_keys_rep) / n_insert;
        r.avg_segs_repaired = static_cast<double>(total_segs_rep) / n_insert;
        results.push_back(r);
        std::cout << "done (" << total_s << "s, " << lslr.num_segments() << " segs)\n";
    }

    // ========================================
    // Print comparison tables
    // ========================================

    print_header("RESULTS: fb_200M_uint64 (100K keys, eps=64, 5% insertions)");

    // Table 1: Correctness
    std::cout << "TABLE 1: Correctness\n\n";
    std::cout << std::left
              << std::setw(22) << "Method"
              << std::setw(14) << "Final Segs"
              << std::setw(14) << "Optimal"
              << std::setw(14) << "Overhead"
              << std::setw(14) << "Max Error"
              << std::setw(14) << "Violations"
              << std::setw(14) << "Valid?"
              << "\n";
    print_separator(106);

    for (const auto& r : results) {
        bool valid = (r.max_error <= epsilon + 1e-9 && r.violations == 0);
        std::cout << std::left
                  << std::setw(22) << r.method
                  << std::setw(14) << r.final_segments
                  << std::setw(14) << r.optimal_final_segments
                  << std::setprecision(3)
                  << std::setw(14) << r.segment_overhead
                  << std::setprecision(1)
                  << std::setw(14) << r.max_error
                  << std::setw(14) << r.violations
                  << std::setw(14) << (valid ? "YES" : "NO")
                  << "\n";
    }

    // Table 2: Update cost
    std::cout << "\n\nTABLE 2: Update Cost\n\n";
    std::cout << std::left
              << std::setw(22) << "Method"
              << std::setw(14) << "Total(s)"
              << std::setw(14) << "Avg(us)"
              << std::setw(14) << "p50(us)"
              << std::setw(14) << "p95(us)"
              << std::setw(14) << "p99(us)"
              << std::setw(14) << "vs FBE"
              << "\n";
    print_separator(106);

    double fbe_time = 0;
    for (const auto& r : results) {
        if (r.method == "FullRebuildEach") { fbe_time = r.total_time_s; break; }
    }

    for (const auto& r : results) {
        double speedup = (fbe_time > 0 && r.total_time_s > 0)
            ? fbe_time / r.total_time_s : 0;
        std::cout << std::left
                  << std::setw(22) << r.method
                  << std::setprecision(2)
                  << std::setw(14) << r.total_time_s
                  << std::setprecision(1)
                  << std::setw(14) << r.avg_time_us
                  << std::setw(14) << r.p50_time_us
                  << std::setw(14) << r.p95_time_us
                  << std::setw(14) << r.p99_time_us
                  << std::setprecision(1)
                  << std::setw(14) << (std::to_string(speedup) + "x")
                  << "\n";
    }

    // Table 3: Latency distribution
    std::cout << "\n\nTABLE 3: Latency Stability (p99/p50 ratio)\n\n";
    std::cout << std::left
              << std::setw(22) << "Method"
              << std::setw(14) << "p50(us)"
              << std::setw(14) << "p99(us)"
              << std::setw(14) << "p99/p50"
              << std::setw(14) << "Tail Risk"
              << "\n";
    print_separator(78);

    for (const auto& r : results) {
        double ratio = (r.p50_time_us > 0) ? r.p99_time_us / r.p50_time_us : 0;
        std::string risk;
        if (ratio < 2) risk = "LOW";
        else if (ratio < 10) risk = "MEDIUM";
        else risk = "HIGH";
        std::cout << std::left
                  << std::setw(22) << r.method
                  << std::setprecision(1)
                  << std::setw(14) << r.p50_time_us
                  << std::setw(14) << r.p99_time_us
                  << std::setprecision(1)
                  << std::setw(14) << ratio
                  << std::setw(14) << risk
                  << "\n";
    }

    // Table 4: Repair statistics (LSLR only)
    std::cout << "\n\nTABLE 4: Local Repair Statistics (LSLR methods)\n\n";
    std::cout << std::left
              << std::setw(22) << "Method"
              << std::setw(18) << "Avg Keys Repaired"
              << std::setw(18) << "Avg Segs Repaired"
              << std::setw(18) << "% of Data"
              << "\n";
    print_separator(76);

    for (const auto& r : results) {
        if (r.avg_keys_repaired == 0) continue;
        double pct = r.avg_keys_repaired / (n_initial + n_insert) * 100;
        std::cout << std::left
                  << std::setw(22) << r.method
                  << std::setprecision(1)
                  << std::setw(18) << r.avg_keys_repaired
                  << std::setprecision(2)
                  << std::setw(18) << r.avg_segs_repaired
                  << std::setprecision(1)
                  << std::setw(18) << pct
                  << "\n";
    }

    std::cout << "\n\n";
    return 0;
}
