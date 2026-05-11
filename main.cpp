#include "common.hpp"
#include "optimal_pla.hpp"
#include "lslr_pla.hpp"
#include "baselines.hpp"
#include "data_generator.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <map>

struct RunConfig {
    Distribution dist;
    std::string dist_name;
    size_t n;
    double epsilon;
    Workload workload;
    std::string workload_name;
    double insert_ratio = 0.10;
};

void print_header() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "  LSLR-PLA: Lazy-Shift Local-Repair Dynamic OptimalPLA\n";
    std::cout << "  Experiment Suite\n";
    std::cout << std::string(70, '=') << "\n\n";
}

std::vector<ExperimentResult> run_experiment(const RunConfig& cfg) {
    std::vector<ExperimentResult> results;

    Key key_range = (cfg.n < 1'000'000) ?
        static_cast<Key>(cfg.n) * 100 :
        static_cast<Key>(cfg.n) * 10;

    std::cout << "\n" << std::string(60, '-') << "\n";
    std::cout << "Dataset: " << cfg.dist_name << ", n=" << cfg.n
              << ", eps=" << cfg.epsilon
              << ", workload=" << cfg.workload_name
              << ", insert_ratio=" << cfg.insert_ratio << "\n";
    std::cout << std::string(60, '-') << "\n";

    // Generate initial data
    std::cout << "  Generating initial keys... " << std::flush;
    auto keys = DataGenerator::generate_keys(cfg.dist, cfg.n, key_range, 42);
    std::cout << "done (" << keys.size() << " keys, range ["
              << keys.front() << ", " << keys.back() << "])\n";

    // Build initial model (to get segments for boundary workload)
    OptimalPLA init_builder(cfg.epsilon);
    auto init_segments = init_builder.build(keys, 0);
    std::cout << "  Initial model: " << init_segments.size() << " segments\n";

    // Generate insertion workload
    std::cout << "  Generating insertion workload... " << std::flush;
    auto insert_keys = DataGenerator::generate_insertions(
        keys, cfg.insert_ratio, cfg.workload, init_segments, 123);
    std::cout << "done (" << insert_keys.size() << " keys)\n";

    size_t n_insert = insert_keys.size();

    // --- Oracle: Full-Rebuild-Final ---
    std::cout << "  Running Full-Rebuild-Final (oracle)... " << std::flush;
    {
        FullRebuildFinal oracle(cfg.epsilon);
        oracle.build(keys);
        Timer t;
        for (size_t i = 0; i < n_insert; ++i) {
            oracle.insert(insert_keys[i]);
        }
        oracle.finalize();
        double final_time = t.elapsed_us();

        ExperimentResult r;
        r.method_name = "FullRebuildFinal";
        r.dataset_name = cfg.dist_name;
        r.workload_name = cfg.workload_name;
        r.n_initial = cfg.n;
        r.n_inserted = n_insert;
        r.epsilon = cfg.epsilon;
        r.total_update_time_us = final_time;
        r.initial_segments = init_segments.size();
        r.final_segments = oracle.num_segments();
        r.optimal_final_segments = oracle.num_segments();
        r.segment_overhead = 1.0;
        auto [max_err, violations] = oracle.validate();
        r.max_error = max_err;
        r.violations = violations;
        results.push_back(r);
        std::cout << r.final_segments << " segments, "
                  << std::fixed << std::setprecision(1)
                  << r.total_update_time_us / 1e6 << "s\n";
    }

    size_t optimal_final_segments = results.back().final_segments;

    // --- Full-Rebuild-Each (only for a subset: small n, eps=64, uniform_random) ---
    if (cfg.n <= 200'000 && cfg.epsilon == 64.0 && cfg.workload_name == "uniform_random") {
        std::cout << "  Running Full-Rebuild-Each... " << std::flush;
        FullRebuildEach fre(cfg.epsilon);
        fre.build(keys);

        ExperimentResult r;
        r.method_name = "FullRebuildEach";
        r.dataset_name = cfg.dist_name;
        r.workload_name = cfg.workload_name;
        r.n_initial = cfg.n;
        r.n_inserted = n_insert;
        r.epsilon = cfg.epsilon;
        r.initial_segments = init_segments.size();

        std::vector<double> times;
        times.reserve(n_insert);

        Timer total_timer;
        for (size_t i = 0; i < n_insert; ++i) {
            Timer ins_timer;
            fre.insert(insert_keys[i]);
            double t_us = ins_timer.elapsed_us();
            times.push_back(t_us);

            if ((i + 1) % (n_insert / 5) == 0) {
                std::cout << (i + 1) * 100 / n_insert << "% " << std::flush;
            }
        }
        double total_time = total_timer.elapsed_us();

        r.total_update_time_us = total_time;
        std::sort(times.begin(), times.end());
        r.avg_insert_time_us = total_time / n_insert;
        r.p50_insert_time_us = percentile(times, 50);
        r.p95_insert_time_us = percentile(times, 95);
        r.p99_insert_time_us = percentile(times, 99);
        r.per_insert_times = times;
        r.final_segments = fre.num_segments();
        r.optimal_final_segments = optimal_final_segments;
        r.segment_overhead = static_cast<double>(fre.num_segments()) / optimal_final_segments;
        auto [max_err, violations] = fre.validate();
        r.max_error = max_err;
        r.violations = violations;

        results.push_back(r);
        std::cout << "done: " << std::fixed << std::setprecision(1)
                  << total_time / 1e6 << "s, "
                  << r.avg_insert_time_us << "us/insert\n";
    }

    // --- Periodic-Rebuild ---
    {
        size_t rebuild_every = std::max(size_t(1), cfg.n / 100);  // 1% of n
        std::cout << "  Running Periodic-Rebuild (every " << rebuild_every << ")... " << std::flush;
        PeriodicRebuild pr(cfg.epsilon, rebuild_every);
        pr.build(keys);

        ExperimentResult r;
        r.method_name = "PeriodicRebuild";
        r.dataset_name = cfg.dist_name;
        r.workload_name = cfg.workload_name;
        r.n_initial = cfg.n;
        r.n_inserted = n_insert;
        r.epsilon = cfg.epsilon;
        r.initial_segments = init_segments.size();

        std::vector<double> times;
        times.reserve(n_insert);

        Timer total_timer;
        for (size_t i = 0; i < n_insert; ++i) {
            Timer ins_timer;
            pr.insert(insert_keys[i]);
            times.push_back(ins_timer.elapsed_us());
        }
        double total_time = total_timer.elapsed_us();

        r.total_update_time_us = total_time;
        std::sort(times.begin(), times.end());
        r.avg_insert_time_us = total_time / n_insert;
        r.p50_insert_time_us = percentile(times, 50);
        r.p95_insert_time_us = percentile(times, 95);
        r.p99_insert_time_us = percentile(times, 99);
        r.per_insert_times = times;
        r.final_segments = pr.num_segments();
        r.optimal_final_segments = optimal_final_segments;
        r.segment_overhead = static_cast<double>(pr.num_segments()) / optimal_final_segments;
        auto [max_err, violations] = pr.validate();
        r.max_error = max_err;
        r.violations = violations;

        results.push_back(r);
        std::cout << "done: " << std::fixed << std::setprecision(1)
                  << total_time / 1e6 << "s, " << pr.num_segments() << " segments\n";
    }

    // --- LSLR variants ---
    std::vector<std::pair<std::string, LSLR_PLA::Config>> lslr_configs = {
        {"LSLR-c0", {cfg.epsilon, 0, false, 8}},
        {"LSLR-c1", {cfg.epsilon, 1, false, 8}},
        {"LSLR-c2", {cfg.epsilon, 2, false, 8}},
        {"LSLR-adaptive", {cfg.epsilon, 1, true, 8}},
    };

    for (const auto& [name, lslr_cfg] : lslr_configs) {
        std::cout << "  Running " << name << "... " << std::flush;
        LSLR_PLA lslr(lslr_cfg);
        lslr.build(keys);

        ExperimentResult r;
        r.method_name = name;
        r.dataset_name = cfg.dist_name;
        r.workload_name = cfg.workload_name;
        r.n_initial = cfg.n;
        r.n_inserted = n_insert;
        r.epsilon = cfg.epsilon;
        r.initial_segments = init_segments.size();

        std::vector<double> times;
        times.reserve(n_insert);
        size_t total_keys_repaired = 0;
        size_t total_segs_repaired = 0;
        size_t total_expansions = 0;

        Timer total_timer;
        for (size_t i = 0; i < n_insert; ++i) {
            auto rec = lslr.insert(insert_keys[i]);
            times.push_back(rec.insert_time_us);
            total_keys_repaired += rec.keys_repaired;
            total_segs_repaired += rec.segments_repaired;
            if (rec.window_expanded) total_expansions++;

            if ((i + 1) % (n_insert / 5) == 0) {
                std::cout << (i + 1) * 100 / n_insert << "% " << std::flush;
            }
        }
        double total_time = total_timer.elapsed_us();

        r.total_update_time_us = total_time;
        std::sort(times.begin(), times.end());
        r.avg_insert_time_us = total_time / n_insert;
        r.p50_insert_time_us = percentile(times, 50);
        r.p95_insert_time_us = percentile(times, 95);
        r.p99_insert_time_us = percentile(times, 99);
        r.per_insert_times = times;
        r.final_segments = lslr.num_segments();
        r.optimal_final_segments = optimal_final_segments;
        r.segment_overhead = static_cast<double>(lslr.num_segments()) / optimal_final_segments;
        auto [max_err, violations] = lslr.validate();
        r.max_error = max_err;
        r.violations = violations;
        r.avg_keys_repaired = static_cast<double>(total_keys_repaired) / n_insert;
        r.avg_segments_repaired = static_cast<double>(total_segs_repaired) / n_insert;
        r.expansion_rate = static_cast<double>(total_expansions) / n_insert;
        r.window_expansions = total_expansions;

        results.push_back(r);
        std::cout << "done: " << std::fixed << std::setprecision(1)
                  << total_time / 1e6 << "s, "
                  << lslr.num_segments() << " segs (overhead "
                  << std::setprecision(3) << r.segment_overhead << "x), "
                  << "avg " << std::setprecision(1) << r.avg_insert_time_us
                  << "us/insert\n";
    }

    return results;
}

void append_results_csv(const std::string& filename,
                         const std::vector<ExperimentResult>& results,
                         bool write_header = false) {
    std::ofstream f(filename, write_header ? std::ios::trunc : std::ios::app);
    if (!f) {
        std::cerr << "Cannot write to " << filename << "\n";
        return;
    }
    if (write_header && !results.empty()) {
        results[0].write_csv_header(f);
    }
    for (const auto& r : results) {
        r.write_csv_row(f);
    }
}

void write_results_csv(const std::string& filename,
                       const std::vector<ExperimentResult>& all_results) {
    std::ofstream f(filename);
    if (!f) {
        std::cerr << "Cannot write to " << filename << "\n";
        return;
    }
    if (all_results.empty()) return;
    all_results[0].write_csv_header(f);
    for (const auto& r : all_results) {
        r.write_csv_row(f);
    }
    std::cout << "\nResults written to " << filename << " ("
              << all_results.size() << " rows)\n";
}

void print_summary_table(const std::vector<ExperimentResult>& results) {
    // Group by dataset, workload, epsilon
    std::cout << "\n\n" << std::string(100, '=') << "\n";
    std::cout << "  SUMMARY TABLE\n";
    std::cout << std::string(100, '=') << "\n\n";

    // Header
    std::cout << std::left
              << std::setw(18) << "Method"
              << std::setw(12) << "Dataset"
              << std::setw(14) << "Workload"
              << std::setw(10) << "n"
              << std::setw(8) << "eps"
              << std::setw(12) << "Total(s)"
              << std::setw(12) << "Avg(us)"
              << std::setw(10) << "Segments"
              << std::setw(10) << "Overhead"
              << std::setw(10) << "MaxErr"
              << std::setw(10) << "Viols"
              << "\n";
    std::cout << std::string(120, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left
                  << std::setw(18) << r.method_name
                  << std::setw(12) << r.dataset_name
                  << std::setw(14) << r.workload_name
                  << std::setw(10) << r.n_initial
                  << std::setw(8) << r.epsilon
                  << std::fixed << std::setprecision(2)
                  << std::setw(12) << r.total_update_time_us / 1e6
                  << std::setprecision(1)
                  << std::setw(12) << r.avg_insert_time_us
                  << std::setw(10) << r.final_segments
                  << std::setprecision(3)
                  << std::setw(10) << r.segment_overhead
                  << std::setprecision(1)
                  << std::setw(10) << r.max_error
                  << std::setw(10) << r.violations
                  << "\n";
    }
}

int main() {
    print_header();

    std::vector<RunConfig> configs;

    // Datasets to test
    std::vector<std::pair<Distribution, std::string>> datasets = {
        {Distribution::FB_200M, "fb_200M"},
        {Distribution::UNIFORM, "uniform"},
        {Distribution::NORMAL, "normal"},
        {Distribution::LOGNORMAL, "lognormal"},
    };

    // Sizes
    std::vector<size_t> sizes = {100'000, 500'000};

    // Epsilons
    std::vector<double> epsilons = {64, 128, 256};

    // Workloads
    std::vector<std::pair<Workload, std::string>> workloads = {
        {Workload::UNIFORM_RANDOM, "uniform_random"},
        {Workload::SAME_DISTRIBUTION, "same_dist"},
        {Workload::APPEND, "append"},
        // hotspot and boundary are slower, skip for now
    };

    // Build config list
    for (const auto& [dist, dname] : datasets) {
        for (size_t n : sizes) {
            for (double eps : epsilons) {
                for (const auto& [wl, wname] : workloads) {
                    configs.push_back({dist, dname, n, eps, wl, wname});
                }
            }
        }
    }

    std::cout << "Total experiment configurations: " << configs.size() << "\n";
    std::cout << "  Datasets: " << datasets.size()
              << ", Sizes: " << sizes.size()
              << ", Epsilons: " << epsilons.size()
              << ", Workloads: " << workloads.size() << "\n";

    std::vector<ExperimentResult> all_results;
    all_results.reserve(configs.size() * 8);

    // Initialize CSV with header
    bool first_config = true;
    std::string csv_path = "results/experiment_results.csv";

    size_t config_idx = 0;
    for (const auto& cfg : configs) {
        config_idx++;
        std::cout << "\n[" << config_idx << "/" << configs.size() << "] ";
        auto res = run_experiment(cfg);
        all_results.insert(all_results.end(), res.begin(), res.end());

        // Incrementally write results to CSV
        append_results_csv(csv_path, res, first_config);
        first_config = false;
    }

    // Write full CSV at end (overwrites incremental, provides clean final version)
    write_results_csv(csv_path, all_results);

    // Print summary
    print_summary_table(all_results);

    std::cout << "\n\nDone! All results in results/experiment_results.csv\n";
    return 0;
}
