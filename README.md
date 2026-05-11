# LSLR-PLA: Lazy-Shift Local-Repair Dynamic OptimalPLA

Dynamic piecewise linear approximation for learned indexes under insertions.
Rather than rebuilding the entire PLA model after each key insertion, LSLR-PLA
lazily shifts segment intercepts and repairs only the local region around the
insertion point.

## Key Idea

```
Insert key x at rank p into segment q:
  1. All segments after q get intercept += 1  (lazy shift — their keys' ranks all increased by 1)
  2. Repair a local window [q−c, q+c] using OptimalPLA
  3. All other segments remain unchanged
```

The model stays **ε-valid** at all times (zero violations), and the local repair
cost is proportional to the window size, not the full dataset.

## Variants

| Variant | Window | Behavior |
|---------|--------|----------|
| LSLR-c0 | radius 0 | Repair only the affected segment. Fastest insert, but fragments (many more segments than optimal). |
| **LSLR-c1** | radius 1 | Repair affected ± 1 neighbor. **Best balance**: perfect quality (1.0x overhead), 3−4x faster than full rebuild. |
| LSLR-c2 | radius 2 | Repair affected ± 2 neighbors. Same quality as c1, slower inserts. |
| LSLR-adaptive | radius 1 + expansion | Like c1, but expands window if boundary segments can merge with outside neighbors. |

## Baselines

| Method | Description |
|--------|-------------|
| FullRebuildEach | Full OptimalPLA rebuild after every insertion. Correctness oracle, but O(u·n). |
| FullRebuildFinal | Build once on the final dataset after all insertions. Quality oracle. |
| PeriodicRebuild | Rebuild every B insertions (default: 1% of n). Fastest amortized, but stale between rebuilds. |

## Project Structure

```
dynamic_optimalPLA/
├── include/
│   ├── common.hpp          # Shared types, PLASegment, ExperimentResult, Timer
│   ├── optimal_pla.hpp     # OptimalPLA builder (greedy minimum-segment)
│   ├── lslr_pla.hpp        # LSLR-PLA dynamic index + FenwickTree
│   ├── baselines.hpp       # FullRebuildEach, FullRebuildFinal, PeriodicRebuild
│   └── data_generator.hpp  # Synthetic distributions + fb_200M binary loader
├── src/
│   ├── optimal_pla.cpp     # Core algorithm: polygon clipping in (slope, intercept) space
│   ├── lslr_pla.cpp        # Insert logic: lazy shift, local repair, segment replacement
│   ├── baselines.cpp       # Baseline implementations
│   └── data_generator.cpp  # Dataset generation + SOSD binary loading
├── main.cpp                # Full experiment matrix runner
├── benchmark.cpp           # Single-dataset head-to-head comparison (100K sample)
├── benchmark_full.cpp      # Full-scale benchmark (up to 200M keys)
├── test.cpp                # OptimalPLA unit tests
├── test_lslr.cpp           # LSLR-PLA unit tests
├── scripts/
│   └── plot_results.py     # Generate comparison tables from experiment CSV
├── CMakeLists.txt
└── results/                # Experiment output (CSV + console logs)
```

## Build

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

Requires: C++17, CMake 3.16+.

### Executables

| Binary | Purpose |
|--------|---------|
| `build/experiments` | Full config matrix across datasets/sizes/epsilons/workloads |
| `build/benchmark_full` | Single run on fb_200M at scale (configurable n, eps) |

## Quick Start

### Small-scale test (100K keys, ~30 seconds)

```bash
./build/benchmark_full 64 100000 5000
```

### Medium-scale test (1M keys, ~few minutes)

```bash
./build/benchmark_full 64 1000000 50000
```

### Full-scale experiment (180M train, 1M insert, ~hours)

```bash
./build/benchmark_full 64 180000000 1000000
```

Arguments: `benchmark_full [epsilon] [n_initial] [n_insert] [run_fbe]`

## Key Implementation Details

### OptimalPLA algorithm
Uses incremental convex polygon clipping in (slope, intercept) dual space.
Each point adds two half-plane constraints. The feasible region is the
intersection of all half-planes. When it becomes empty, a new segment starts.

The initial bounding box is **data-dependent** — scaled to the key range
to avoid double-precision loss (the original `1e15` bound caused `1e15+1 == 1e15`).

### Lazy shift
After inserting at rank p into segment q, all segments after q have their
keys' ranks increased by 1. We compensate by adding 1 to their stored intercept.
This is O(m) per insert (m = number of segments), which is acceptable since
m is typically small (< 1000 for reasonable ε).

### Local repair
Keys are extracted from the global array by key range (not rank, since ranks
become stale after insertions). The OptimalPLA is rebuilt only on this local
window. New segments replace the old window. Segment metadata (first_key,
last_key) is kept consistent.

### Segment lookup
Binary search on `first_key` values. Key-based (not rank-based) to avoid
staleness issues after insertions shift all ranks.

## Datasets

- **fb_200M_uint64** at `/home/andy/Projects/Datasets/SOSD/fb_200M_uint64`
  - 200 million sorted uint64 keys, 1.6 GB raw binary (little-endian)
  - Key range: ~1 to ~77 billion
  - Loaded via `DataGenerator::load_range_keys()` or `load_sampled_keys()`
- Synthetic: uniform, normal, lognormal, piecewise (generated on-the-fly)

## Key Bugs Found and Fixed

1. **Numerical precision**: `BIG=1e15` → data-dependent bounds. `1e15+1` loses the +1 in double precision.
2. **Stale segment ranks**: Switched from rank-based to key-based segment lookup after insertions shift all ranks.
3. **Stale Fenwick tree**: `predict()` queried an offset tree never resized after segment replacement. Fixed by using `seg.intercept` directly.
4. **Wrong segment for insertion**: `find_segment(keys_[p])` was called before insertion, finding the old key's segment. Fixed: insert first, then `find_segment(x)`.
5. **APPEND last_key**: When a key extends a segment's key range, `last_key` wasn't updated, so local repair missed the new key.
