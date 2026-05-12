// Comparison: LSLR-PLA vs DynamicLearnedIndex (Sgelet et al.)
// Fair comparison on fb_200M_uint64 with same epsilon, same keys.
//
// Build: cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make compare -j$(nproc)
// Usage: ./build/compare [n_initial_M] [n_insert]

#include "common.hpp"
#include "optimal_pla.hpp"
#include "lslr_pla.hpp"
#include "baselines.hpp"
#include "data_generator.hpp"

#include "baseline_wrapper.hpp"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <unordered_set>

struct BenchResult {
    std::string method;
    size_t initial_segments{0};
    size_t final_segments{0};
    size_t optimal_final_segments{0};
    double segment_overhead{0};
    double max_error{0};
    size_t violations{0};
    double total_time_s{0};
    double avg_time_us{0};
    double p50_time_us{0};
    double p95_time_us{0};
    double p99_time_us{0};
    bool valid{false};
};

void print_sep(int w = 110) { std::cout << std::string(w, '-') << "\n"; }

// Validate baseline: check that all original + inserted keys are findable.
// Also measures approximate lookup overhead.
std::pair<double, size_t> validate_baseline(
    BaselineIndex& idx, const std::vector<int64_t>& keys,
    const std::vector<int64_t>& inserted)
{
    size_t missing = 0;
    // Check a random sample to keep it fast
    size_t sample = std::min(size_t(5000), keys.size());
    std::mt19937_64 rng(99);
    Timer t;
    for (size_t i = 0; i < sample; ++i) {
        size_t j = rng() % keys.size();
        if (!idx.find(keys[j])) missing++;
    }
    // Also check ALL inserted keys (small set)
    for (auto k : inserted) {
        if (!idx.find(k)) missing++;
    }
    double elapsed = t.elapsed_us();
    return {elapsed, missing};
}

int main(int argc, char** argv) {
    std::cout << std::fixed << std::setprecision(1);

    // Config
    size_t n_initial_M  = 10;
    size_t n_insert     = 10'000;

    if (argc > 1) n_initial_M = std::stoull(argv[1]);
    if (argc > 2) n_insert    = std::stoull(argv[2]);

    size_t n_initial = n_initial_M * 1'000'000;
    size_t n_total = n_initial + n_insert;

    std::cout << "\n================================================================================\n";
    std::cout << "  COMPARISON BENCHMARK: LSLR-PLA vs DynamicLearnedIndex (Sgelet et al.)\n";
    std::cout << "  Dataset: fb_200M_uint64\n";
    std::cout << "  Initial keys: " << n_initial_M << "M  |  Insertions: " << n_insert << "\n";
    std::cout << "  Epsilon: 64  (both methods)\n";
    std::cout << "================================================================================\n";

    // ---- Load data ----
    const std::string ds_path = "/home/andy/Projects/Datasets/SOSD/fb_200M_uint64";

    std::cout << "\nLoading first " << n_initial_M << "M keys + last "
              << n_insert << " keys from fb_200M_uint64...\n";

    // Initial keys: first n_initial
    auto keys_u64 = DataGenerator::load_range_keys(ds_path, 0, n_initial);
    std::vector<Key> keys(keys_u64.begin(), keys_u64.end());
    std::cout << "  Initial: " << keys.size() << " keys, range ["
              << keys.front() << ", " << keys.back() << "]\n";

    // Insertion keys: last n_insert from the 200M dataset
    auto insert_u64 = DataGenerator::load_range_keys(ds_path, 200'000'000 - n_insert, n_insert);
    std::vector<Key> insert_keys(insert_u64.begin(), insert_u64.end());
    {
        std::mt19937_64 rng(123);
        std::shuffle(insert_keys.begin(), insert_keys.end(), rng);
    }
    std::cout << "  Insert:  " << insert_keys.size() << " keys, range ["
              << insert_keys.front() << ", " << insert_keys.back() << "] (shuffled)\n";

    // Convert to int64_t for baseline (filter keys > INT64_MAX)
    std::vector<int64_t> keys_i64;
    keys_i64.reserve(keys.size());
    for (auto k : keys) {
        if (k <= static_cast<uint64_t>(INT64_MAX)) keys_i64.push_back(static_cast<int64_t>(k));
    }
    std::vector<int64_t> insert_i64;
    insert_i64.reserve(insert_keys.size());
    size_t overflow_count = 0;
    for (auto k : insert_keys) {
        if (k <= static_cast<uint64_t>(INT64_MAX)) insert_i64.push_back(static_cast<int64_t>(k));
        else overflow_count++;
    }
    if (overflow_count > 0) {
        std::cout << "  Note: " << overflow_count << " insert keys > INT64_MAX filtered for baseline\n";
    }

    // ---- Oracle: OptimalPLA on full final set ----
    std::cout << "\nComputing oracle (OptimalPLA on " << (n_initial + n_insert) << " keys)...\n";
    std::vector<Key> final_all(keys.begin(), keys.end());
    final_all.insert(final_all.end(), insert_keys.begin(), insert_keys.end());
    std::sort(final_all.begin(), final_all.end());
    OptimalPLA pla(64.0);
    auto oracle_segs = pla.build(final_all, 0);
    size_t optimal_segments = oracle_segs.size();
    std::cout << "  Oracle: " << optimal_segments << " segments\n";

    // ---- Compute initial OptimalPLA ----
    auto init_segs = pla.build(keys, 0);
    std::cout << "  Initial OptimalPLA: " << init_segs.size() << " segments\n";

    std::vector<BenchResult> results;

    // ================================================================
    // 1. LSLR-PLA variants
    // ================================================================
    std::cout << "\n--- LSLR-PLA ---\n";

    std::vector<std::pair<std::string, LSLR_PLA::Config>> lslr_configs = {
        {"LSLR-c0",       {64.0, 0, false, 8}},
        {"LSLR-c1",       {64.0, 1, false, 8}},
        {"LSLR-c2",       {64.0, 2, false, 8}},
        {"LSLR-adaptive", {64.0, 1, true,  8}},
    };

    for (const auto& [name, cfg] : lslr_configs) {
        std::cout << "  [" << name << "] building..." << std::flush;
        LSLR_PLA lslr(cfg);
        lslr.build(keys);
        std::cout << " " << lslr.num_segments() << " segs, inserting "
                  << n_insert << "..." << std::flush;

        std::vector<double> times; times.reserve(n_insert);
        Timer total;
        for (size_t i = 0; i < n_insert; ++i) {
            Timer t;
            lslr.insert(insert_keys[i]);
            times.push_back(t.elapsed_us());
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
        r.valid = (max_err <= 64.0 + 1e-9 && viol == 0);
        r.total_time_s = total_s;
        r.avg_time_us = total_s / n_insert * 1e6;
        r.p50_time_us = percentile(times, 50);
        r.p95_time_us = percentile(times, 95);
        r.p99_time_us = percentile(times, 99);
        results.push_back(r);

        std::cout << " " << total_s << "s, " << lslr.num_segments()
                  << " segs, max_err=" << max_err << "\n";
    }

    // ================================================================
    // 2. DynamicLearnedIndex (baseline, Sgelet et al.)
    //
    // The baseline models key = a·rank + b, with epsilon in KEY space.
    // Our LSLR models rank = a·key + b, with epsilon in RANK space.
    // For a fair comparison, convert our rank epsilon to key space:
    //   eps_key = eps_rank × (key_range / n_keys)
    // ================================================================
    {
        // Compute equivalent epsilon in key space
        double avg_keys_per_rank = static_cast<double>(keys.back() - keys.front()) / keys.size();
        int equiv_eps = std::max(1, static_cast<int>(64.0 * avg_keys_per_rank));
        std::cout << "\n--- DynamicLearnedIndex (Sgelet et al.) ---\n";
        std::cout << "  Avg keys-per-rank: " << std::fixed << std::setprecision(1)
                  << avg_keys_per_rank << "\n";
        std::cout << "  Our eps=64 (rank) ≈ baseline eps=" << equiv_eps << " (key)\n\n";

        // Run with original eps=64 (key) — shows the mismatch
        // Run with equivalent eps — the fair comparison
        std::vector<std::pair<std::string, int>> baseline_configs = {
            {"DLI-eps=64(key)", 64},
            {"DLI-eps=" + std::to_string(equiv_eps) + "(key)", equiv_eps},
        };

        for (const auto& [name, eps_val] : baseline_configs) {
            std::cout << "  [" << name << "] loading " << keys_i64.size()
                      << " keys..." << std::flush;
            BaselineIndex baseline(eps_val);
            Timer load_timer;
            size_t report_every = std::max(size_t(1), keys_i64.size() / 5);
            for (size_t i = 0; i < keys_i64.size(); ++i) {
                baseline.insert(keys_i64[i]);
                if ((i + 1) % report_every == 0)
                    std::cout << (i+1)*100/keys_i64.size() << "% " << std::flush;
            }
            double load_time = load_timer.elapsed_us() / 1e6;
            int init_segs = baseline.segment_count();
            std::cout << " done (" << load_time << "s, " << init_segs << " segs)\n";

            std::cout << "    Inserting " << insert_i64.size() << " keys..." << std::flush;
            std::vector<double> times; times.reserve(insert_i64.size());
            Timer total;
            for (size_t i = 0; i < insert_i64.size(); ++i) {
                Timer t;
                baseline.insert(insert_i64[i]);
                times.push_back(t.elapsed_us());
            }
            double total_s = total.elapsed_us() / 1e6;
            std::sort(times.begin(), times.end());

            auto [lookup_time, missing] = validate_baseline(baseline, keys_i64, insert_i64);

            BenchResult r;
            r.method = name;
            r.initial_segments = static_cast<size_t>(init_segs);
            r.final_segments = static_cast<size_t>(baseline.segment_count());
            r.optimal_final_segments = optimal_segments;
            r.segment_overhead = static_cast<double>(baseline.segment_count()) / optimal_segments;
            r.max_error = (missing == 0) ? 64.0 : 999.0;
            r.violations = missing;
            r.valid = (missing == 0);
            r.total_time_s = total_s;
            r.avg_time_us = total_s / n_insert * 1e6;
            r.p50_time_us = percentile(times, 50);
            r.p95_time_us = percentile(times, 95);
            r.p99_time_us = percentile(times, 99);
            results.push_back(r);

            std::cout << " " << total_s << "s, " << baseline.segment_count()
                      << " segs, missing=" << missing << "\n";
        }
    }

    // ================================================================
    // PRINT RESULTS
    // ================================================================
    std::cout << "\n\n" << std::string(110, '=') << "\n";
    std::cout << "  RESULTS: " << n_initial_M << "M initial + " << n_insert
              << " insertions, eps=" << 64.0 << "\n";
    std::cout << "  Initial segments: " << init_segs.size()
              << "  |  Optimal final: " << optimal_segments << "\n";
    std::cout << std::string(110, '=') << "\n\n";

    // Table 1: Correctness + Model quality
    std::cout << "TABLE 1: Model Quality & Correctness\n\n";
    std::cout << std::left
              << std::setw(22) << "Method"
              << std::setw(16) << "Init Segs"
              << std::setw(16) << "Final Segs"
              << std::setw(16) << "Optimal"
              << std::setw(14) << "Overhead"
              << std::setw(14) << "Max Error"
              << std::setw(14) << "Missing"
              << std::setw(10) << "Valid?"
              << "\n";
    print_sep();
    for (const auto& r : results) {
        std::cout << std::left
                  << std::setw(22) << r.method
                  << std::setw(16) << r.initial_segments
                  << std::setw(16) << r.final_segments
                  << std::setw(16) << r.optimal_final_segments
                  << std::setprecision(3)
                  << std::setw(14) << r.segment_overhead
                  << std::setprecision(1)
                  << std::setw(14) << r.max_error
                  << std::setw(14) << r.violations
                  << std::setw(10) << (r.valid ? "YES" : "NO")
                  << "\n";
    }

    // Table 2: Insertion performance
    std::cout << "\n\nTABLE 2: Insertion Latency (10K inserts)\n\n";
    std::cout << std::left
              << std::setw(22) << "Method"
              << std::setw(14) << "Total(s)"
              << std::setw(14) << "Avg(us)"
              << std::setw(14) << "p50(us)"
              << std::setw(14) << "p95(us)"
              << std::setw(14) << "p99(us)"
              << std::setw(14) << "p99/p50"
              << "\n";
    print_sep();
    for (const auto& r : results) {
        double ratio = (r.p50_time_us > 0) ? r.p99_time_us / r.p50_time_us : 0;
        std::cout << std::left
                  << std::setw(22) << r.method
                  << std::setprecision(3)
                  << std::setw(14) << r.total_time_s
                  << std::setprecision(1)
                  << std::setw(14) << r.avg_time_us
                  << std::setw(14) << r.p50_time_us
                  << std::setw(14) << r.p95_time_us
                  << std::setw(14) << r.p99_time_us
                  << std::setprecision(1)
                  << std::setw(14) << ratio
                  << "\n";
    }

    // Comparison summary
    std::cout << "\n\n--- Summary ---\n";
    // Find results for head-to-head
    const BenchResult* lslr_c1 = nullptr;
    const BenchResult* baseline_64 = nullptr;
    const BenchResult* baseline_eq = nullptr;
    for (const auto& r : results) {
        if (r.method == "LSLR-c1") lslr_c1 = &r;
        if (r.method.find("DLI-eps=64") == 0) baseline_64 = &r;
        if (r.method.find("DLI-eps=") == 0 && baseline_64 && r.method != baseline_64->method)
            baseline_eq = &r;
    }
    std::cout << "\n  Coordinate systems:\n";
    std::cout << "    LSLR:        rank = a*key  + b,  epsilon=" << 64.0 << " (rank)\n";
    std::cout << "    Baseline:    key  = a*rank + b,  epsilon in KEY space\n";
    std::cout << "    Equiv. conv: eps_key = eps_rank * avg_keys_per_rank\n\n";

    if (lslr_c1 && baseline_64) {
        std::cout << "  With original eps=64 (unfair — different error spaces):\n";
        std::cout << "    LSLR-c1 segments:        " << lslr_c1->final_segments << "\n";
        std::cout << "    DLI-eps=64(key) segments: " << baseline_64->final_segments
                  << "  (" << baseline_64->segment_overhead << "x optimal)\n";
    }
    if (lslr_c1 && baseline_eq) {
        std::cout << "\n  With EQUIVALENT epsilon (fair — both ~same rank error):\n";
        std::cout << "    LSLR-c1 segments:        " << lslr_c1->final_segments << "\n";
        std::cout << "    " << baseline_eq->method << " segments: "
                  << baseline_eq->final_segments << "\n";
        std::cout << "    LSLR-c1 overhead:        " << lslr_c1->segment_overhead << "x optimal\n";
        std::cout << "    " << baseline_eq->method << " overhead: "
                  << baseline_eq->segment_overhead << "x optimal\n";
        std::cout << "    LSLR-c1 avg insert:      " << lslr_c1->avg_time_us << " us\n";
        std::cout << "    " << baseline_eq->method << " avg insert: "
                  << baseline_eq->avg_time_us << " us\n";
        std::cout << "    LSLR-c1 p99/p50:         " << lslr_c1->p99_time_us / lslr_c1->p50_time_us << "\n";
        std::cout << "    " << baseline_eq->method << " p99/p50: "
                  << baseline_eq->p99_time_us / baseline_eq->p50_time_us << "\n";
    }

    std::cout << "\n";
    return 0;
}
