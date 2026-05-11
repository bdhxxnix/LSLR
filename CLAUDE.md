# CLAUDE.md — Guidance for Claude Code

## Project overview

This is **LSLR-PLA**: a dynamic learned index that maintains an OptimalPLA
(piecewise linear approximation) model under key insertions without full rebuilds.

The core contribution is the **lazy-shift local-repair** strategy:
- Inserting a key at rank p only shifts ranks of keys after p by +1
- Segments entirely after the insertion point just need intercept += 1
- Only the local region around the insertion needs OptimalPLA rebuild

## Build

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

Three executables are built:
- `build/experiments` — full matrix across datasets/epsilons/workloads
- `build/benchmark_full` — single-run configurable benchmark on fb_200M
- `build/benchmark` — focused head-to-head on 100K sample (compiled separately)

## Architecture

```
include/
  common.hpp          — PLASegment, InsertionRecord, Timer, percentile()
  optimal_pla.hpp     — OptimalPLA builder class
  lslr_pla.hpp        — LSLR_PLA class + FenwickTree (unused, kept for reference)
  baselines.hpp       — FullRebuildEach, FullRebuildFinal, PeriodicRebuild
  data_generator.hpp  — Distribution enum, Workload enum, binary loaders

src/
  optimal_pla.cpp     — Polygon clipping in (a,b) space; data-dependent bounds
  lslr_pla.cpp        — insert(): lazy shift + local repair + segment replacement
  baselines.cpp       — Straightforward implementations
  data_generator.cpp  — Synthetic generators + SOSD binary I/O
```

### Core algorithm: `src/optimal_pla.cpp`

The OptimalPLA uses **incremental convex polygon clipping** in the dual space
of (slope a, intercept b). Each point (key x, rank y) with error bound ε adds
two half-plane constraints:

```
a·x + b ≤ y + ε   (upper)
a·x + b ≥ y − ε   (lower)
```

The feasible region is the intersection of all half-planes, maintained as a
convex polygon in CCW vertex order. Points are added greedily until the polygon
becomes empty, then a new segment starts. The segment's (a,b) is the polygon centroid.

**Critical detail**: the initial bounding box is data-dependent, computed from
the key range and count. This avoids double-precision loss (the original `1e15`
bound caused `1e15 + 1 == 1e15`). See `initial_polygon(kmin, kmax, n)`.

The greedy algorithm is NOT strictly optimal for minimum segment count — the
choice of centroid from the feasible polygon affects downstream segments. In
practice, LSLR-c1 often matches or beats the greedy from-scratch result.

### Lazy shift + local repair: `src/lslr_pla.cpp`

The `insert(x)` method:

1. Insert x into global sorted array (permanent)
2. `q = find_segment(x)` — key-based binary search on segment first_keys
3. `segments_[q].end += 1` and update `last_key` if x extends the range
4. For all j > q: `segments_[j].start++, .end++, .intercept += 1`
5. Select window [q−c, q+c] clamped to [0, m−1]
6. Extract keys by key range [segments[left].first_key, segments[right].last_key]
7. Rebuild OptimalPLA on local keys → `new_local_segments`
8. Adaptive expansion: try merge new boundary segments with outside neighbors
9. `replace_segments(left, right, new_local_segments)` — rebuild first_key index

Key-based lookup is used throughout (not rank-based) because ranks become stale
after insertions. Segment boundaries are identified by `first_key` / `last_key`
values, which are stable.

### Segment replacement

`replace_segments()` swaps out the window [left, right] with new segments and
rebuilds `segment_first_keys_` from scratch. It does NOT resize the Fenwick tree
(the tree is unused in the current simplified design — all intercept shifts are
applied directly).

## Running experiments

### Quick test (100K keys, ~30s)
```bash
./build/benchmark_full 64 100000 5000
```

### Full-scale (180M train, 1M insert, ~hours)
```bash
./build/benchmark_full 64 180000000 1000000
```

Arguments: `benchmark_full [epsilon] [n_initial] [n_insert] [run_fbe]`
- `run_fbe=1` enables FullRebuildEach (capped at 1000 inserts — each rebuild is O(n))

### Config matrix (main.cpp)
```bash
./build/experiments
```
Runs all dataset/epsilon/workload combinations. Writes incremental CSV to
`results/experiment_results.csv`. Can be stopped/resumed safely (CSV is
appended per-config).

## Key parameter guidance

- **ε (epsilon)**: Error bound in rank units. Larger ε → fewer segments → faster
  inserts but larger last-mile search range. For fb_200M at 100K keys, ε=64
  gives ~10 segments. At 200M keys, ε=64 gives thousands.
- **Window radius c**: c=0 is fastest but fragments badly (40−80x segment overhead).
  c=1 is the sweet spot (1.0x overhead, 3−4x speedup). c=2 does more work for
  no quality gain. Adaptive rarely expands.
- **Insert ratio**: 5−10% of n is standard for learned index benchmarks.

## Common issues

1. **Large ε with few segments**: When m is small (m ≤ 2c+1), the window covers
   all segments and LSLR degrades to near-full-rebuild cost. This is expected.
2. **BOUNDARY workload generator**: Was slow due to duplicate generation.
   Fixed by pre-computing the boundary-adjacent key pool, shuffling, and
   falling back to uniform when exhausted.
3. **APPEND workload**: LSLR-c0 achieves 1.0x overhead here because only the
   last segment is affected. The easiest case for all methods.
4. **Double precision in polygon clipping**: Always use data-dependent bounds.
   The `a_hi * kmax` product must stay below ~1e14 to avoid precision loss
   (2^53 ≈ 9e15). The current code caps `a_hi` at `1e14 / kmax`.
5. **Memory**: 200M keys = 1.6 GB. With segment structures, 2−4 GB total.

## What to modify for new datasets

Add a new case to `DataGenerator::generate_keys()` in `data_generator.cpp`:

```cpp
case Distribution::MY_DATA: {
    keys = load_range_keys("/path/to/file", 0, n);
    break;
}
```

Or call `load_range_keys()` / `load_sampled_keys()` directly for raw uint64
binary files. The format is little-endian uint64, pre-sorted.

## What NOT to do

- Don't use rank-based segment lookup — ranks become stale after any insertion
- Don't rely on the Fenwick tree — it's dead code kept for reference
- Don't use `effective_intercept()` — use `seg.intercept` directly
- Don't run FullRebuildEach on > 100K keys without capping inserts
- Don't use fixed `BIG` bounds for polygon initialization — use the
  data-dependent `initial_polygon(kmin, kmax, n)` function
