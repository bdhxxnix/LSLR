# LSLR-PLA: Analysis and Experimental Evaluation

## 1. OptimalPLA — Convex Hull Maintenance in the Original Space

Based on: *"Maximum error-bounded Piecewise Linear Representation for online
stream approximation"* (Xie et al., 2014), implemented in the PGM-index
(gvinciguerra/PGM-index).

### 1.1 Model and error bound

The OptimalPLA maps keys to ranks using linear segments:

\[\hat{y} = a \cdot x + b\]

where \(x\) is a key, \(a\) the slope, \(b\) the intercept, and \(\hat{y}\) the predicted
rank. For error bound \(\varepsilon\) (in rank units):

\[|\hat{y}_i - y_i| \leq \varepsilon \quad \forall i\]

### 1.2 Convex hull algorithm

Rather than polygon clipping in \((a, b)\) dual space, the PGM-index algorithm
works directly in \((x, y) = (key, rank)\) space. For each point \((x_i, y_i)\),
two \(\varepsilon\)-shifted points are created:

\[U_i = (x_i,\; y_i + \varepsilon), \quad L_i = (x_i,\; y_i - \varepsilon)\]

A line \((a, b)\) is feasible iff it passes above all \(L_i\) and below all \(U_i\):

\[L_{i,y} \leq a \cdot x_i + b \leq U_{i,y} \quad \forall i\]

The algorithm maintains two **convex hulls**:

- **Upper hull**: convex chain of \(U\) points (those that constrain the line from above)
- **Lower hull**: convex chain of \(L\) points (those that constrain the line from below)

#### Feasibility check (O(1))

The current feasible region is captured by a 4-point **rectangle**:

```
rectangle[0] — upper-left  (leftmost point of the upper constraining edge)
rectangle[1] — lower-left  (leftmost point of the lower constraining edge)
rectangle[2] — lower-right (rightmost point of the lower constraining edge)
rectangle[3] — upper-right (rightmost point of the upper constraining edge)
```

Two slopes define the feasible range:

- \(\text{slope}_1 = \text{rectangle}[2] - \text{rectangle}[0]\) — minimum allowed slope
- \(\text{slope}_2 = \text{rectangle}[3] - \text{rectangle}[1]\) — maximum allowed slope

A new point \((x, y)\) is feasible iff its shifted points fall within these bounds:

\[U - \text{rectangle}[2] \geq \text{slope}_1, \quad L - \text{rectangle}[3] \leq \text{slope}_2\]

If either condition fails, no single line can cover all points seen so far → the
segment ends and a new one begins.

#### Adding a point (O(1) amortized)

When the point passes the feasibility check, the rectangle and convex hulls are updated:

1. If \(U\) lies above the current upper-right constraint, walk the lower hull
   to find the new most-constraining point, update `rectangle[1]` and `rectangle[3]`,
   and prune non-convex points from the upper hull (those forming a right turn
   with the new point).
2. If \(L\) lies below the current lower-right constraint, walk the upper hull
   similarly, update `rectangle[0]` and `rectangle[2]`, and prune non-convex
   points from the lower hull.

Convexity is maintained using cross-product checks:

\[\text{cross}(A, B, C) = (B_x - A_x)(C_y - A_y) - (B_y - A_y)(C_x - A_x)\]

Points that form non-convex turns (cross ≤ 0 for upper hull, ≥ 0 for lower hull)
are removed.

#### Segment extraction

When the segment ends (or all points are processed), the segment parameters are
computed from the feasible region:

- **Slope**: midpoint of \([a_{\text{min}}, a_{\text{max}}]\), where
  \(a_{\text{min}} = \text{slope}_1\), \(a_{\text{max}} = \text{slope}_2\)
- **Intercept**: computed from the intersection point \((i_x, i_y)\) of the
  two diagonal lines of the rectangle, using \(b = i_y - a \cdot i_x\)

#### Arithmetic precision

All cross products use `LargeSigned<T>` — `__int128` for 64-bit types, `long double`
for floating-point — to avoid overflow in intermediate multiplications without
resorting to arbitrary precision.

---

## 2. LSLR-PLA — Lazy-Shift Local-Repair

### 2.1 Core algorithm

**Insert key \(x\) at rank \(p\) into segment \(q\):**

1. Insert \(x\) into the global sorted array (permanent)

2. `q = find_segment(x)` — key-based binary search on segment `first_key` values

3. `segments[q].end += 1` and update `last_key` if \(x\) extends the range

4. For all \(j > q\): `segments[j].start++, .end++, .intercept += 1` (lazy shift)

5. Select window \([q-c,\; q+c]\) clamped to \([0, m-1]\)

6. Extract keys by key range `[segments[left].first_key, segments[right].last_key]`

7. Rebuild OptimalPLA on local keys → `new_local_segments`

8. Adaptive expansion: try merging new boundary segments with outside neighbors;
   if merge fails and adaptive mode is on, expand window and retry

9. `replace_segments(left, right, new_local_segments)` — splice and rebuild `first_key` index

   ![ChatGPT Image 2026年5月13日 21_28_31](/home/andy/Downloads/ChatGPT Image 2026年5月13日 21_28_31.png)

### 2.2 Why lazy shift is correct

After inserting at rank \(p\), all keys after \(p\) shift by +1 in rank. For any
segment entirely after the insertion point, incrementing its intercept by +1 exactly
compensates for the +1 shift of all its keys. The segment's slope remains valid
because the relative positions of keys within the segment are unchanged.

### 2.3 Window radius

| Variant | Window | Behavior |
|---------|--------|----------|
| LSLR-c0 | \([q, q]\) | Only affected segment. Fast but ~4% fragmentation. |
| **LSLR-c1** | \([q-1, q+1]\) | +1 neighbor. Matches oracle (1.00× overhead). Best balance. |
| LSLR-c2 | \([q-2, q+2]\) | +2 neighbors. Same quality as c1, slower. |
| LSLR-adaptive | \([q-1, q+1]\) + expansion | Like c1 with adaptive expansion. Rarely expands. |

\(c=1\) is sufficient because a single insertion only changes the intercept, not the
slope, for keys after \(p\). The only segments whose composition can change are those
whose key ranges overlap with or are immediately adjacent to the insertion key.

### 2.4 Key design decisions

- **Key-based lookup**: binary search on `first_key`. Rank-based lookup becomes stale
  after any insertion.
- **Key-range extraction**: local repair uses key ranges, not ranks, to extract the
  window's keys.
- **Stable boundaries**: `first_key` / `last_key` values don't change after lazy shift.
- **Segment replacement**: old window is spliced out, new segments spliced in,
  `first_key` index rebuilt from scratch.

---

## 3. Baseline Methods

### 3.1 FullRebuildEach

Rebuilds the entire OptimalPLA from scratch after every insertion.

- **Quality**: always optimal (matches from-scratch build)
- **Cost per insert**: \(O(n)\) — rebuilds all \(n\) keys
- **Total cost for \(k\) inserts**: \(O(k \cdot n)\)
- **Use**: oracle for correctness; impractical beyond trivial sizes

### 3.2 FullRebuildFinal

Builds OptimalPLA once on the complete final dataset (initial + all insertions).

- **Use**: quality oracle — "what would a from-scratch build produce?"
- **Not a dynamic method**; included for segment count reference

### 3.3 PeriodicRebuild

Rebuilds the entire OptimalPLA every \(R\) insertions.

- **Between rebuilds**: \(O(1)\) per insert (just array insertion)
- **At rebuild**: \(O(n)\) for full OptimalPLA build
- **Total cost**: \(O(k + \frac{k}{R} \cdot n)\)
- **Tradeoff**: larger \(R\) → faster amortized inserts but more quality degradation
- **Latency**: bimodal — median ~0.3 μs but periodic spikes of ~25s at 180M scale

### 3.4 DynamicLearnedIndex (Sgelet et al.)

A header-only C++ library implementing a dynamic PGM-index with convex hull
maintenance. Reference: `github.com/Sgelet/DynamicLearnedIndex`.

#### 3.4.1 Architecture

The index has three layers:

1. **RankHullTree** — a rank-based AVL tree maintaining the convex hull of points
   in dual space for each segment
2. **LineTree** — an AVL tree of segments (LineNodes), each containing a RankHullTree
   and a fitted line segment
3. **Page storage** — keys stored in pages (~\(\varepsilon\) keys per page),
   organized by `key/epsilon` with doubly-linked traversal

#### 3.4.2 Coordinate system (critical difference)

| Aspect | LSLR-PLA / OptimalPLA | DynamicLearnedIndex |
|---|---|---|
| Model space | (key, rank): rank = a·key + b | (rank, key): key = a·rank + b |
| x-axis | key (non-uniform spacing) | rank (uniform spacing) |
| y-axis | rank | key |
| Error bound | \(\vert\text{rank} - \hat{\text{rank}}\vert \leq \varepsilon\) | \(\vert\text{key} - \hat{\text{key}}\vert \leq \varepsilon\) |
| Slope | ranks per key (density) | keys per rank |
| Epsilon units | **rank units** | **key units** |

This is the fundamental architectural difference. Epsilon means completely
different things in the two systems. For a fair comparison, epsilons must be
converted:

\[\varepsilon_{\text{key}} = \varepsilon_{\text{rank}} \times \frac{k_{\text{max}} - k_{\text{min}}}{n}\]

For fb_200M first 10M keys (\(k_{\text{max}} \approx 3.78 \times 10^9\)):
\(\varepsilon_{\text{key}} = 64 \times 378 = 24,\!178\).

#### 3.4.3 RankHullTree — dynamic convex hull

Each segment maintains a RankHullTree: an AVL tree where leaves are (rank, key)
points and internal nodes store **bridges** — the convex hull edges connecting
left and right subtrees.

- **insert(key)**: adds a leaf, triggers `onUpdate` which recomputes bridges via
  `findBridge()` — a binary walk comparing segment midpoints and slopes
- **findLine()**: searches for a line intersecting all bridge segments of the
  upper and lower hull envelopes. Walks the tree comparing bridge slopes and
  checking endpoint containment. When search intervals collapse, falls back to
  `findWitness()` for double-wedge detection.
- **findLine is conservative**: it returns `false` when specific bridge segments
  overlap in (rank, key) space (lines 779–780, 785–786, 805–806, 825–826 of
  RankHullTree.h). This is a **sufficient but not necessary** condition for
  global infeasibility — a feasible line may exist even when local bridges
  appear to intersect.

#### 3.4.4 LineTree — dynamic segment tree

An AVL tree of LineNodes. Each insert triggers:

1. **Into existing segment**: insert key into hull, call `findLine()`. If it
   succeeds, update counts. If it fails, **split** the hull at the insertion
   point, creating a new segment.
2. **Between segments**: try to join into left neighbor's hull, then right
   neighbor's hull. If both fail, create a singleton segment.
3. **Merge attempts**: try to merge affected segment with predecessor and
   successor. Merge succeeds only if `findLine()` on the combined hull returns
   true. If `findLine()` fails, the merge is **undone** — the conservative
   `findLine` prevents merges that should be feasible.

#### 3.4.5 Why DLI fragments with mismatched epsilon

With \(\varepsilon = 64\) in **key space** (equivalent to \(\varepsilon \approx 0.15\)
in rank space for fb_200M):

- Each segment can cover at most ~64 key units of error
- At ~378 keys per rank, 64 key units covers only ~0.17 ranks
- `findLine()` frequently fails after just 2–3 keys → 1.6M segments for 10M keys
- The conservative `findLine` compounds this: even when a line exists, bridge
  overlap causes early failure

With equivalent epsilon (\(\varepsilon = 24,\!178\) in key space):
- The error budget is 24,178 key units ≈ 64 ranks
- `findLine()` succeeds for much longer → 25K segments (near optimal)

#### 3.4.6 Other limitations

- **No bulk-load**: keys must be inserted individually. Loading 10M keys takes
  69–302 seconds (vs ~milliseconds for OptimalPLA::build).
- **Memory**: each key is an AVL node (~120 bytes). 10M keys ≈ 3 GB.
- **int64_t only**: uses signed arithmetic; cannot handle uint64 keys > INT64_MAX.
  fb_200M has 4 such keys in its last 10K.

### 3.5 Epsilon conversion for cross-method comparison

Since LSLR and DLI use epsilon in different spaces (rank vs key), comparing them
at the same numeric epsilon value is meaningless. A conversion is required.

#### 3.5.1 Global conversion formula

The relationship between rank error and key error at a given point depends on the
local slope — how many key units correspond to one rank unit:

\[\varepsilon_{\text{key}} = \varepsilon_{\text{rank}} \times \frac{\Delta k}{\Delta r}\]

where \(\frac{\Delta k}{\Delta r}\) is the local keys-per-rank. Since we don't know
the local slope a priori, we use the **global average**:

\[\bar{s} = \frac{k_{\text{max}} - k_{\text{min}}}{n}, \quad
\varepsilon_{\text{key}} = \varepsilon_{\text{rank}} \times \bar{s}\]

For fb_200M first 10M keys: \(\bar{s} = \frac{3.78 \times 10^9}{10^7} \approx 378\),
giving \(\varepsilon_{\text{key}} = 64 \times 378 = 24,\!178\).

#### 3.5.2 Local vs global slope — why the conversion is imperfect

The global average masks substantial local variation. Consider two regions of the
dataset at different densities:

| Region | Key range | Ranks | Local kpr | Equivalent ε_key |
|---|---|---|---|---|
| Dense (early fb_200M) | [1, 100,307] | 0–234 | 429 | 27,434 |
| Sparse (late fb_200M) | [7.73×10¹⁰, 7.73×10¹⁰] | — | ≫ 378 | ≫ 24,178 |
| **Global average** | [1, 3.78×10⁹] | 0–10M | **378** | **24,178** |

The global conversion \(\varepsilon_{\text{key}} = 24,\!178\) is:

- **Too tight in dense regions** (local kpr = 429): DLI would need ε = 27,434 to
  match our rank error budget, but receives only 24,178 — effectively a tighter
  constraint than intended. Segments are forced shorter than they could be.
- **Too loose in sparse regions** (local kpr ≫ 378): DLI receives more key-space
  budget than the equivalent rank error would grant, allowing longer segments
  than strictly fair.

The net effect depends on the data distribution. In fb_200M, early keys are
densely packed (many keys per key-range unit) and later keys are sparser. The
single global conversion cannot simultaneously match the local density everywhere.

#### 3.5.3 Why a perfect conversion is impossible

A truly fair epsilon would need to vary **per segment**, adapting to the local
keys-per-rank. But the local kpr is unknown until the segment is built — it's a
circular dependency. Furthermore, DLI and LSLR use different coordinate systems
not just for epsilon but for the entire feasibility test:

- **LSLR**: checks feasibility in (key, rank) space → convex hulls of
  \((x_i, y_i \pm \varepsilon)\) → slope = Δrank/Δkey
- **DLI**: checks feasibility in (rank, key) space → bridge segments of
  \((r_i, k_i \pm \varepsilon)\) → slope = Δkey/Δrank

Even with a per-segment epsilon conversion, the two algorithms would make
different choices about where to place segment boundaries because their
feasibility tests operate on different geometric objects (upper/lower convex
hulls of rank-shifted points vs bridge segments of key-shifted points).

#### 3.5.4 Practical consequence

The global conversion is the best achievable approximation for a fair comparison
without modifying either algorithm. The 3.4% segment count difference (DLI with
24,178 fewer segments than the greedy oracle) should be interpreted with this
caveat: part of the difference may come from the conversion being looser in some
regions, not purely from DLI's line-selection heuristic outperforming the greedy
mid-slope.

---


## 4. Experimental Comparison

### 4.1 Setup

- **Dataset**: fb_200M_uint64 (200M sorted uint64 keys)
- **Scale**: 10M initial keys + 10K insertions (last 10K of dataset, shuffled)
- **Epsilon**: \(\varepsilon = 64\) (rank space for LSLR)
- **Hardware**: Linux, GCC 11.5, `-O3 -march=native`

### 4.2 Segment quality — fair comparison

For the baseline, epsilon is converted to equivalent key-space units:
\(\varepsilon_{\text{key}} = 64 \times 378 = 24,\!178\).

| Method | Epsilon space | Initial Segments | Final Segments | Overhead vs Oracle |
|---|---|---|---|---|
| OptimalPLA (oracle) | 64 (rank) | — | 26,070 | 1.000× |
| LSLR-c0 | 64 (rank) | 26,044 | 27,191 | 1.043× |
| **LSLR-c1** | 64 (rank) | 26,044 | **26,070** | **1.000×** |
| LSLR-c2 | 64 (rank) | 26,044 | 26,070 | 1.000× |
| LSLR-adaptive | 64 (rank) | 26,044 | 26,070 | 1.000× |
| PeriodicRebuild | 64 (rank) | — | 26,070 | 1.000× |
| DLI (unfair) | 64 (key) | 1,619,171 | 1,621,093 | 62.2× |
| **DLI (equiv)** | 24,178 (key) | 25,136 | **25,180** | **0.966×** |

**Key findings:**

- With equivalent epsilon, **both LSLR-c1 and DLI achieve near-optimal segment
  counts** (1.00× and 0.97× overhead respectively).
- The 62× overhead was entirely an artifact of comparing different error spaces.
- DLI with equivalent epsilon slightly beats the greedy OptimalPLA (0.97×),
  confirming that the greedy centroid heuristic is not strictly optimal.
- LSLR-c0 shows 4% fragmentation from lacking neighbor context in rebuilds.

### 4.3 Insertion latency

| Method | Avg (μs) | p50 (μs) | p95 (μs) | p99 (μs) | p99/p50 |
|---|---|---|---|---|---|
| LSLR-c0 | 309 | 299 | 355 | 406 | 1.4 |
| **LSLR-c1** | 362 | 350 | 466 | 772 | 2.2 |
| LSLR-c2 | 410 | 389 | 538 | 839 | 2.2 |
| LSLR-adaptive | 365 | 353 | 419 | 466 | 1.3 |
| DLI-eps=64(key) | 25 | 15 | 75 | 106 | 6.9 |
| **DLI-eps=24178(key)** | 160 | 165 | 260 | 287 | 1.7 |

**Key findings:**

- DLI (equivalent ε) is ~2.3× faster per insert than LSLR-c1 (160 μs vs 362 μs)
- DLI with ε=64(key) is much faster (25 μs) but produces 62× segment overhead —
  it's doing very little work because segments are tiny
- LSLR tail latency is stable (p99/p50 ≤ 2.2)
- DLI equivalent-ε tail is also stable (p99/p50 = 1.7)

### 4.4 Load time and memory (10M keys)

| Method | Load time | Memory |
|---|---|---|
| OptimalPLA::build | < 0.1 s | ~200 MB |
| LSLR-PLA | < 0.1 s | ~200 MB |
| PeriodicRebuild | < 0.1 s | ~200 MB |
| DLI-eps=64 | 69 s | ~3.0 GB |
| DLI-eps=24178 | 302 s | ~3.0 GB |

DLI has no bulk-load — each key is inserted individually through the AVL tree
and hull maintenance. Larger epsilon makes hull operations more expensive
(longer segments → larger hulls → more bridge computations).

### 4.5 Segment size distribution (1M keys, ε-mismatched)

At ε=64 without conversion:

| Method | Segments | Avg keys/seg | p50 | p10 | Min | Max |
|---|---|---|---|---|---|---|
| OptimalPLA | 2,596 | 385 | 358 | 228 | 142 | 1,155 |
| DLI-eps=64(key) | 159,462 | 6.3 | **3** | **2** | 2 | 267 |

Half of DLI's segments contain 3 or fewer keys. The key-space constraint is
~378× tighter than the equivalent rank-space constraint, and the conservative
`findLine` prevents effective merging.

### 4.6 Repair statistics (LSLR methods, 10M keys)

| Method | Avg Keys Repaired | Avg Segs Repaired | % of Data |
|---|---|---|---|
| LSLR-c0 | 385 | 1.0 | 0.0038% |
| LSLR-c1 | 1,081 | 2.8 | 0.0108% |
| LSLR-c2 | 1,843 | 4.8 | 0.0184% |
| LSLR-adaptive | 1,091 | 2.9 | 0.0109% |

Local repair touches 0.004–0.018% of the data per insertion. LSLR-c1 repairs
~2.8 segments worth of keys, sufficient to maintain global optimality.

---

## 5. Algorithmic Comparison Summary

### 5.1 Feasibility testing

| | OptimalPLA / LSLR | DynamicLearnedIndex |
|---|---|---|
| **Space** | Original (x,y) = (key, rank) | Original (x,y) = (rank, key) |
| **Method** | Two convex hulls (U/L) + 4-point rectangle | Bridge walk through hull tree |
| **Test** | O(1) slope comparison against rectangle diagonals | Do bridge segments admit a separating line? |
| **Nature** | **Exact** — infeasible ⇔ no line passes between hulls | **Conservative** — bridge overlap ⇒ return false |
| **Arithmetic** | __int128 cross products (exact until final extraction) | BigQuotient rational arithmetic |
| **Result** | Maximum-length segments within greedy limits | Premature splits when bridges overlap locally |

### 5.2 Segment maintenance

| | LSLR-PLA | DynamicLearnedIndex |
|---|---|---|
| **Insert trigger** | Local OptimalPLA rebuild | Hull insert + findLine + split |
| **Affected region** | Window [q−c, q+c] | Current segment + neighbors |
| **Merge strategy** | Implicit in rebuild (polygon naturally extends) | Attempt join with neighbors; undo if findLine fails |
| **Quality preservation** | Rebuild ensures local optimality | findLine conservatism causes fragmentation; merges limited |

### 5.3 Method ranking

| Method | Segment Quality | Insert Speed | Load Speed | Memory | Tail Stability |
|---|---|---|---|---|---|
| FullRebuildEach | 1.00× | O(n) — impractical | Fast | Low | N/A |
| PeriodicRebuild | 1.00× | Fast amortized, O(n) spikes | Fast | Low | Poor (bimodal) |
| **LSLR-c1** | **1.00×** | **Good (362 μs)** | **Instant** | **Low** | **Good (2.2×)** |
| DLI (equiv ε) | 0.97× | Better (160 μs) | Very slow (302s) | High (15×) | Good (1.7×) |

## 6. Conclusions

1. **LSLR-c1 matches the oracle**: with window radius \(c=1\), segment count after
   10K insertions equals the from-scratch OptimalPLA build. The lazy-shift + local
   repair strategy is both correct and practical.

2. **Coordinate systems matter**: the DynamicLearnedIndex's (rank, key) orientation
   applies epsilon to key prediction error, not rank error. Without epsilon conversion
   (\(64 \to 24,\!178\)), the comparison is fundamentally unfair — the baseline appears
   62× worse when it's actually solving a 378× tighter problem. With equivalent
   epsilon, both methods achieve near-optimal segments.

3. **DLI insertions are faster but at a steep upfront cost**: DLI is ~2.3× faster
   per insert (160 μs vs 362 μs) but requires 5 minutes to load 10M keys (vs
   milliseconds for LSLR) and uses 15× more memory. This makes LSLR better for
   scenarios with pre-existing data; DLI better for pure incremental construction.

4. **PeriodicRebuild is simple but unreliable**: its bimodal latency (0.3 μs median
   vs 25s spikes at 180M) makes it unsuitable for latency-sensitive applications.

5. **LSLR-c0 isn't enough**: zero-radius window causes 4% segment fragmentation.
   The cost of \(c=1\) is modest and the quality gain is definitive.

6. **The greedy OptimalPLA is not strictly optimal**: DLI with equivalent epsilon
   produces 0.97× the greedy segment count, and LSLR-c1 at large scale can sometimes
   beat the oracle by 1–2 segments. Centroid choice matters.
