#include "common.hpp"
#include "optimal_pla.hpp"
#include "lslr_pla.hpp"
#include "baselines.hpp"
#include "data_generator.hpp"

#include <iostream>
#include <iomanip>
#include <algorithm>

struct BenchResult {
    std::string method;
    size_t initial_segments;
    size_t final_segments;
    size_t optimal_final_segments;
    double segment_overhead;
    double max_error;
    size_t violations;
    double total_time_s;
    double avg_time_us;
    double p50_time_us;
    double p95_time_us;
    double p99_time_us;
    double avg_keys_repaired;
    double avg_segs_repaired;
};

void print_separator(int w = 106) { std::cout << std::string(w, '-') << "\n"; }

int main(int argc, char** argv) {
    // --- Configurable parameters ---
    double epsilon          = 64.0;
    size_t n_initial        = 180'000'000;  // first 180M for training
    size_t n_insert         = 1'000'000;     // 1M insertions
    size_t progress_every   = 100'000;       // print progress every N inserts
    bool   run_fbe          = false;         // FullRebuildEach — impossibly slow at this scale

    // Allow command-line overrides
    if (argc > 1) epsilon     = std::stod(argv[1]);
    if (argc > 2) n_initial   = std::stoull(argv[2]);
    if (argc > 3) n_insert    = std::stoull(argv[3]);
    if (argc > 4) run_fbe     = (std::stoi(argv[4]) != 0);

    size_t n_total = n_initial + n_insert;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Config: n_initial=" << n_initial
              << ", n_insert=" << n_insert
              << ", eps=" << epsilon
              << ", run_fbe=" << (run_fbe ? "yes" : "no") << "\n\n";

    // ================================================================
    // Load full fb_200M dataset (1.6 GB, 200M uint64 keys)
    // ================================================================
    const std::string ds_path = "/home/andy/Projects/Datasets/SOSD/fb_200M_uint64";

    std::cout << "Loading first " << n_total << " keys from fb_200M_uint64..." << std::flush;
    auto all_keys = DataGenerator::load_range_keys(ds_path, 0, n_total);
    std::cout << " done (" << all_keys.size() << " keys, range ["
              << all_keys.front() << ", " << all_keys.back() << "])\n";

    // Split: first n_initial for training, remaining n_insert for insertion workload
    std::vector<Key> keys(all_keys.begin(), all_keys.begin() + static_cast<ptrdiff_t>(n_initial));
    std::vector<Key> insert_keys(all_keys.begin() + static_cast<ptrdiff_t>(n_initial),
                                  all_keys.end());

    // Shuffle insertion keys so they don't arrive sorted
    {
        std::mt19937_64 rng(123);
        std::shuffle(insert_keys.begin(), insert_keys.end(), rng);
    }

    std::cout << "Training keys: " << keys.size()
              << "  range [" << keys.front() << ", " << keys.back() << "]\n";
    std::cout << "Insert keys:   " << insert_keys.size()
              << "  range [" << insert_keys.front() << ", " << insert_keys.back()
              << "]  (shuffled)\n";

    // ================================================================
    // Build initial model (for reference segment count)
    // ================================================================
    OptimalPLA pla(epsilon);
    Timer init_timer;
    auto init_segs = pla.build(keys, 0);
    double init_time = init_timer.elapsed_us() / 1e6;
    std::cout << "\nInitial OptimalPLA: " << init_segs.size()
              << " segments  (built in " << init_time << "s)\n";

    // ================================================================
    // Oracle: FullRebuildFinal (build once on the complete final dataset)
    // ================================================================
    std::cout << "Computing oracle (FullRebuildFinal on all "
              << n_total << " keys)..." << std::flush;
    Timer oracle_timer;
    auto oracle_segs = pla.build(all_keys, 0);
    double oracle_time = oracle_timer.elapsed_us() / 1e6;
    size_t optimal_segments = oracle_segs.size();
    std::cout << " done: " << optimal_segments << " segments ("
              << oracle_time << "s)\n";

    // Verify oracle
    {
        double max_err = 0;
        for (size_t i = 0; i < all_keys.size(); ++i) {
            size_t lo = 0, hi = oracle_segs.size() - 1;
            while (lo < hi) {
                size_t mid = lo + (hi - lo + 1) / 2;
                if (oracle_segs[mid].first_key <= all_keys[i]) lo = mid;
                else hi = mid - 1;
            }
            long double pred = oracle_segs[lo].slope * static_cast<long double>(all_keys[i]) + oracle_segs[lo].intercept;
            long double err = std::abs(pred - static_cast<long double>(i));
            if (err > max_err) max_err = err;
        }
        std::cout << "  oracle max_err=" << max_err << "\n";
    }

    std::vector<BenchResult> results;

    // ================================================================
    // 1. FullRebuildEach  (only if explicitly enabled)
    // ================================================================
    if (run_fbe) {
        std::cout << "\n[FullRebuildEach] WARNING: extremely slow at this scale\n";
        size_t n_fbe = std::min(n_insert, size_t(100000));  // cap at 1000 inserts
        // std::cout << "  Running only " << n_fbe << " insertions..." << std::flush;
        FullRebuildEach fre(epsilon);
        fre.build(keys);
        std::vector<double> times;
        Timer total;
        for (size_t i = 0; i < n_fbe; ++i) {
            Timer t;
            fre.insert(insert_keys[i]);
            times.push_back(t.elapsed_us());
            if ((i+1) % 100 == 0) std::cout << (i+1)*100/n_fbe << "% " << std::flush;
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
        r.max_error = max_err; r.violations = viol;
        r.total_time_s = total_s;
        r.avg_time_us = total_s / n_fbe * 1e6;
        r.p50_time_us = percentile(times, 50);
        r.p95_time_us = percentile(times, 95);
        r.p99_time_us = percentile(times, 99);
        results.push_back(r);
        std::cout << " done (" << total_s << "s for " << n_fbe << " inserts, "
                  << "est full: " << (total_s / n_fbe * n_insert / 3600) << "h)\n";
    }

    // ================================================================
    // 2. PeriodicRebuild
    // ================================================================
    {
        size_t interval = std::max(size_t(1), n_insert / 10);  // 10 rebuilds total
        std::cout << "\n[PeriodicRebuild every " << interval << " inserts] "
                  << n_insert << " insertions..." << std::flush;
        PeriodicRebuild pr(epsilon, interval);
        pr.build(keys);
        std::vector<double> times; times.reserve(n_insert);
        Timer total;
        for (size_t i = 0; i < n_insert; ++i) {
            Timer t;
            pr.insert(insert_keys[i]);
            times.push_back(t.elapsed_us());
            if ((i+1) % progress_every == 0)
                std::cout << (i+1)*100/n_insert << "% " << std::flush;
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
        r.max_error = max_err; r.violations = viol;
        r.total_time_s = total_s;
        r.avg_time_us = total_s / n_insert * 1e6;
        r.p50_time_us = percentile(times, 50);
        r.p95_time_us = percentile(times, 95);
        r.p99_time_us = percentile(times, 99);
        results.push_back(r);
        std::cout << " done (" << total_s << "s)\n";
    }

    // ================================================================
    // 3-6. LSLR variants
    // ================================================================
    std::vector<std::pair<std::string, LSLR_PLA::Config>> lslr_configs = {
        {"LSLR-c0",        {epsilon, 0, false, 8}},
        {"LSLR-c1",        {epsilon, 1, false, 8}},
        {"LSLR-c2",        {epsilon, 2, false, 8}},
        {"LSLR-adaptive",  {epsilon, 1, true,  8}},
    };

    for (const auto& [name, cfg] : lslr_configs) {
        std::cout << "\n[" << name << "] " << n_insert << " insertions..." << std::flush;
        LSLR_PLA lslr(cfg);
        lslr.build(keys);
        std::vector<double> times; times.reserve(n_insert);
        size_t total_keys_rep = 0, total_segs_rep = 0;
        Timer total;
        for (size_t i = 0; i < n_insert; ++i) {
            Timer t;
            auto rec = lslr.insert(insert_keys[i]);
            times.push_back(t.elapsed_us());
            total_keys_rep += rec.keys_repaired;
            total_segs_rep += rec.segments_repaired;
            if ((i+1) % progress_every == 0)
                std::cout << (i+1)*100/n_insert << "% " << std::flush;
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
        r.max_error = max_err; r.violations = viol;
        r.total_time_s = total_s;
        r.avg_time_us = total_s / n_insert * 1e6;
        r.p50_time_us = percentile(times, 50);
        r.p95_time_us = percentile(times, 95);
        r.p99_time_us = percentile(times, 99);
        r.avg_keys_repaired = static_cast<double>(total_keys_rep) / n_insert;
        r.avg_segs_repaired = static_cast<double>(total_segs_rep) / n_insert;
        results.push_back(r);
        // Print intermediate result
        std::cout << " done (" << total_s << "s, " << lslr.num_segments()
                  << " segs, max_err=" << max_err << ", viol=" << viol << ")\n";
    }

    // ================================================================
    // Print tables
    // ================================================================

    std::cout << "\n\n" << std::string(106, '=') << "\n";
    std::cout << "  RESULTS: fb_200M_uint64 full (n_initial=" << n_initial
              << ", n_insert=" << n_insert << ", eps=" << epsilon << ")\n";
    std::cout << "  Initial segments: " << init_segs.size()
              << "  |  Optimal final: " << optimal_segments << "\n";
    std::cout << std::string(106, '=') << "\n\n";

    // TABLE 1: Correctness
    std::cout << "TABLE 1: Correctness\n\n";
    std::cout << std::left
              << std::setw(22) << "Method"
              << std::setw(16) << "Final Segs"
              << std::setw(16) << "Optimal"
              << std::setw(14) << "Overhead"
              << std::setw(14) << "Max Error"
              << std::setw(14) << "Violations"
              << std::setw(10) << "Valid?"
              << "\n";
    print_separator();
    for (const auto& r : results) {
        bool valid = (r.max_error <= epsilon + 1e-9 && r.violations == 0);
        std::cout << std::left
                  << std::setw(22) << r.method
                  << std::setw(16) << r.final_segments
                  << std::setw(16) << r.optimal_final_segments
                  << std::setprecision(3)
                  << std::setw(14) << r.segment_overhead
                  << std::setprecision(1)
                  << std::setw(14) << r.max_error
                  << std::setw(14) << r.violations
                  << std::setw(10) << (valid ? "YES" : "NO")
                  << "\n";
    }

    // TABLE 2: Update cost
    std::cout << "\n\nTABLE 2: Update Cost\n\n";
    std::cout << std::left
              << std::setw(22) << "Method"
              << std::setw(14) << "Total(s)"
              << std::setw(14) << "Avg(us)"
              << std::setw(14) << "p50(us)"
              << std::setw(14) << "p95(us)"
              << std::setw(14) << "p99(us)"
              << "\n";
    print_separator();
    for (const auto& r : results) {
        std::cout << std::left
                  << std::setw(22) << r.method
                  << std::setprecision(2)
                  << std::setw(14) << r.total_time_s
                  << std::setprecision(1)
                  << std::setw(14) << r.avg_time_us
                  << std::setw(14) << r.p50_time_us
                  << std::setw(14) << r.p95_time_us
                  << std::setw(14) << r.p99_time_us
                  << "\n";
    }

    // TABLE 3: Latency stability
    std::cout << "\n\nTABLE 3: Latency Stability (p99/p50 ratio)\n\n";
    std::cout << std::left
              << std::setw(22) << "Method"
              << std::setw(14) << "p50(us)"
              << std::setw(14) << "p99(us)"
              << std::setw(14) << "p99/p50"
              << "\n";
    print_separator(64);
    for (const auto& r : results) {
        double ratio = (r.p50_time_us > 0) ? r.p99_time_us / r.p50_time_us : 0;
        std::cout << std::left
                  << std::setw(22) << r.method
                  << std::setprecision(1)
                  << std::setw(14) << r.p50_time_us
                  << std::setw(14) << r.p99_time_us
                  << std::setprecision(1)
                  << std::setw(14) << ratio
                  << "\n";
    }

    // TABLE 4: Repair statistics (LSLR only)
    std::cout << "\n\nTABLE 4: Local Repair Statistics (LSLR methods)\n\n";
    std::cout << std::left
              << std::setw(22) << "Method"
              << std::setw(18) << "Avg Keys Repaired"
              << std::setw(18) << "Avg Segs Repaired"
              << std::setw(18) << "% of Initial Data"
              << "\n";
    print_separator(76);
    for (const auto& r : results) {
        if (r.avg_keys_repaired == 0) continue;
        double pct = r.avg_keys_repaired / n_initial * 100;
        std::cout << std::left
                  << std::setw(22) << r.method
                  << std::setprecision(1)
                  << std::setw(18) << r.avg_keys_repaired
                  << std::setprecision(2)
                  << std::setw(18) << r.avg_segs_repaired
                  << std::setprecision(2)
                  << std::setw(18) << pct
                  << "\n";
    }

    std::cout << "\n";
    return 0;
}
