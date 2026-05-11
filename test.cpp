#include "optimal_pla.hpp"
#include <iostream>
#include <cassert>

int main() {
    // Test 1: Perfectly linear data (should give 1 segment for any epsilon >= 0)
    {
        std::vector<Key> keys;
        for (size_t i = 0; i < 1000; ++i) {
            keys.push_back(i * 100);  // keys at 0, 100, 200, ..., 99900
        }

        OptimalPLA pla(1.0);
        auto segs = pla.build(keys, 0);
        std::cout << "Test 1 (perfectly linear, 1000 keys, eps=1): "
                  << segs.size() << " segments" << std::endl;

        // Verify all errors are within bound
        for (size_t i = 0; i < keys.size(); ++i) {
            size_t seg_id = 0;
            for (size_t s = 0; s < segs.size(); ++s) {
                if (segs[s].first_key <= keys[i] && keys[i] <= segs[s].last_key) {
                    seg_id = s;
                    break;
                }
            }
            double pred = segs[seg_id].slope * keys[i] + segs[seg_id].intercept;
            double err = std::abs(pred - static_cast<double>(i));
            if (err > 1.0 + 1e-9) {
                std::cerr << "  ERROR at index " << i << ": key=" << keys[i]
                          << " pred=" << pred << " actual=" << i
                          << " err=" << err << "\n";
                return 1;
            }
        }
        std::cout << "  All errors <= 1.0, OK" << std::endl;
    }

    // Test 2: Random keys, check validity
    {
        std::mt19937_64 rng(42);
        std::uniform_int_distribution<uint64_t> dist(0, 10000000);
        std::set<uint64_t> seen;
        std::vector<Key> keys;
        while (keys.size() < 5000) {
            auto k = dist(rng);
            if (seen.insert(k).second) keys.push_back(k);
        }
        std::sort(keys.begin(), keys.end());

        std::vector<double> epsilons = {32.0, 64.0, 128.0};
        for (double eps : epsilons) {
            OptimalPLA pla(eps);
            auto segs = pla.build(keys, 0);
            std::cout << "Test 2 (random 5000 keys, eps=" << eps << "): "
                      << segs.size() << " segments" << std::endl;

            double max_err = 0;
            for (size_t i = 0; i < keys.size(); ++i) {
                // find segment
                size_t seg_id = 0;
                for (size_t s = 0; s < segs.size(); ++s) {
                    if (keys[i] >= segs[s].first_key && keys[i] <= segs[s].last_key) {
                        seg_id = s;
                        break;
                    }
                }
                double pred = segs[seg_id].slope * keys[i] + segs[seg_id].intercept;
                double err = std::abs(pred - static_cast<double>(i));
                max_err = std::max(max_err, err);
            }
            std::cout << "  max_err=" << max_err << " (bound=" << eps << ")";
            if (max_err <= eps + 1e-9) {
                std::cout << " OK" << std::endl;
            } else {
                std::cout << " FAIL" << std::endl;
                return 1;
            }
        }
    }

    // Test 3: Small number of equidistant points, verify single segment
    {
        std::vector<Key> keys = {100, 200, 300, 400, 500};
        OptimalPLA pla(10.0);
        auto segs = pla.build(keys, 0);
        std::cout << "Test 3 (5 equidistant keys, eps=10): "
                  << segs.size() << " segment(s)" << std::endl;
        if (segs.size() != 1) {
            std::cerr << "  Expected 1 segment!" << std::endl;
            return 1;
        }
        std::cout << "  slope=" << segs[0].slope << " intercept=" << segs[0].intercept << std::endl;
    }

    // Test 4: Two clusters far apart
    {
        std::vector<Key> keys;
        for (int i = 0; i < 100; ++i) keys.push_back(i * 10);       // cluster 1
        for (int i = 0; i < 100; ++i) keys.push_back(1000000 + i * 10); // cluster 2
        OptimalPLA pla(5.0);
        auto segs = pla.build(keys, 0);
        std::cout << "Test 4 (two clusters, eps=5): "
                  << segs.size() << " segment(s)" << std::endl;
        // Should be 2 segments (one per cluster) since the gap between clusters
        // causes very different slopes to be optimal
    }

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
