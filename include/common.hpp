#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <string>
#include <iostream>
#include <chrono>
#include <random>
#include <cassert>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <set>
#include <deque>

using Key = uint64_t;
using Rank = size_t;

constexpr double INF_SLOPE = 1e18;

struct Point {
    Key x;
    Rank y;
};

struct PLASegment {
    size_t start;       // start rank in logical array
    size_t end;         // end rank in logical array
    Key first_key;
    Key last_key;
    double slope;
    double intercept;

    // For lazy shift tracking
    int64_t lazy_shift{0};
};

struct InsertionRecord {
    Key key;
    size_t position;    // insertion rank
    size_t segment_id;  // which segment it fell into
    double insert_time_us;
    size_t keys_repaired;
    size_t segments_repaired;
    bool window_expanded{false};
    size_t window_expansions{0};
};

struct ExperimentResult {
    std::string method_name;
    std::string dataset_name;
    std::string workload_name;
    size_t n_initial;
    size_t n_inserted;
    double epsilon;

    // Timing
    double total_update_time_us{0};
    double avg_insert_time_us{0};
    double p50_insert_time_us{0};
    double p95_insert_time_us{0};
    double p99_insert_time_us{0};
    std::vector<double> per_insert_times;

    // Model quality
    size_t initial_segments{0};
    size_t final_segments{0};
    size_t optimal_final_segments{0};
    double segment_overhead{0};

    // Validity
    double max_error{0};
    size_t violations{0};

    // Local repair stats (LSLR only)
    double avg_keys_repaired{0};
    double avg_segments_repaired{0};
    double expansion_rate{0};
    size_t window_expansions{0};

    // Query performance
    double avg_query_time_us{0};
    double p95_query_time_us{0};
    double avg_search_range{0};

    void write_csv_header(std::ostream& os) const {
        os << "method,dataset,workload,n_initial,n_inserted,epsilon,"
           << "total_update_us,avg_insert_us,p50_insert_us,p95_insert_us,p99_insert_us,"
           << "initial_segments,final_segments,optimal_final_segments,segment_overhead,"
           << "max_error,violations,"
           << "avg_keys_repaired,avg_segments_repaired,expansion_rate,window_expansions,"
           << "avg_query_us,p95_query_us,avg_search_range\n";
    }

    void write_csv_row(std::ostream& os) const {
        os << method_name << "," << dataset_name << "," << workload_name << ","
           << n_initial << "," << n_inserted << "," << epsilon << ","
           << total_update_time_us << "," << avg_insert_time_us << ","
           << p50_insert_time_us << "," << p95_insert_time_us << "," << p99_insert_time_us << ","
           << initial_segments << "," << final_segments << "," << optimal_final_segments << ","
           << segment_overhead << ","
           << max_error << "," << violations << ","
           << avg_keys_repaired << "," << avg_segments_repaired << ","
           << expansion_rate << "," << window_expansions << ","
           << avg_query_time_us << "," << p95_query_time_us << "," << avg_search_range << "\n";
    }
};

// Timer utility
class Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start_;
public:
    Timer() : start_(Clock::now()) {}
    void reset() { start_ = Clock::now(); }
    double elapsed_us() const {
        return std::chrono::duration<double, std::micro>(Clock::now() - start_).count();
    }
};

// Compute percentile from sorted vector
inline double percentile(const std::vector<double>& sorted, double pct) {
    if (sorted.empty()) return 0;
    size_t idx = static_cast<size_t>(pct / 100.0 * (sorted.size() - 1));
    idx = std::min(idx, sorted.size() - 1);
    return sorted[idx];
}
