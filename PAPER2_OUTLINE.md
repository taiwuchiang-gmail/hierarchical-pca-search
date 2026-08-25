# Composable Exact Bounds: A Self-Configuring Hybrid of PCA and Triangle-Inequality Pruning

**Tai-Wu Chiang** — sequel to *A Hierarchical Pruning Algorithm for Fast, Exact
Nearest Neighbor Search in High-Dimensional Spaces* (Zenodo, 2026,
doi:10.5281/zenodo.21387359). Same venue style (ACM 2-column, ~3–4 pages) and
same philosophy: an engineering artifact with runnable code, honest about when
it does NOT help.

> **Status of this document:** section-by-section skeleton. Every claim is
> tagged with the evidence that already exists (`script -> measured number`)
> or marked **[GAP]** if an optional experiment would strengthen it.
> React to the structure before prose is written.

---

## Abstract (draft prose — react to framing)

Exact nearest-neighbor search can be accelerated by cheap lower bounds that
prune candidates without computing full distances. Our prior work used bounds
from truncated PCA projections, which are tight precisely when variance
concentrates in few dimensions. We observe that such bounds are one member of
a *family*: any region that provably contains a set of points yields an exact
lower bound via the query-to-region distance, and different region shapes are
tight for different data structures. We compose two complementary families —
PCA truncation (tight for linearly correlated data) and triangle-inequality
ball bounds around k-means centroids (tight for clustered data; exact for
*any* centers, so approximation quality affects only speed, never
correctness) — into a single exact cascade. Because neither family helps on
all data, the index *configures itself*: a three-tier diagnostic measures a
sample's eigenvalue decay, detects structure beyond the covariance with two
cheap statistics (excess kurtosis of PCA components and squared-dependence),
and settles the question with a measured trial of the real cascade. We report
results in hardware-independent work units — full-vector fetches, total bytes
touched (charging the pruning machinery itself), and resident RAM per vector —
because wall-clock proved regime-dependent: the same algorithm measured
4.7x/3.6x/2.7x/parity under changes of numeric type, machine load, and
implementation tuning, while its work counts never moved. On SIFT1M with
official queries, the composed cascade computes full distances for 0.06% of
the dataset and keeps 8-40 bytes per vector resident (vs 512 for a flat
scan); on clustered data with a flat eigenvalue spectrum the ball level
prunes to 3% before a single projection is computed (13.9x wall-clock vs
2.1x for PCA alone); on SIFT the ball level's bound work cancels its savings
— a verdict the diagnostic correctly predicts from a sample, protecting the
practitioner from building a useless level. Wall-clock appears as validation
of a bytes-touched cost model. Exactness is verified on every query. On data
with neither structure, the diagnostic correctly reports that no method helps.
All code, diagnostics, and experiments are public.

---

## 1. Introduction

- Recap the problem + Paper I's result in one paragraph (exact search,
  GEMINI-style lower-bound pruning, PCA cascade, 100% exactness).
- **The gap this paper fills:** bound tightness is a property of the *data's
  structure type*, and Paper I's bound covers only one type (linear
  correlation). Practitioners have no way to know, before integrating, which
  bound family — if any — their data supports.
- **Thesis:** exact bounds compose (the max of lower bounds is a lower
  bound), and the right composition can be *chosen by measurement* on a
  sample, cheaply, before any index is built.
- **Economic motivation (RAM residency):** an exhaustive flat scan requires
  the entire dataset resident in RAM — the most expensive byte in the
  machine (DRAM runs ~an order of magnitude above NVMe per byte, more under
  supply pressure). The pruning cascade needs only its metadata resident
  (8–40 B/vector vs 512 B/vector fp32: 12–64x less RAM); full vectors live
  on NVMe/disk and are fetched for <0.1% of candidates. At 1B vectors this
  is 8–40 GB of RAM versus 512 GB — a commodity box versus a specialty one,
  with exactness preserved.
- Contributions list (the paper's spine):
  1. **Containing-region principle** unifying exact pruning bounds; the
     ball/triangle-inequality bound as the clustered-data complement to the
     PCA bound. (§3)
  2. **A composed exact cascade** — ball level + PCA levels — with a
     radius re-probe that keeps the cascade as tight as PCA-only while
     entering with half the candidates. Storage overhead ~8 bytes/point.
     (§5)
  3. **A three-tier self-configuration funnel** — eigen curve, then
     kurt(PCA)/sqdep screening, then a measured trial: "the statistics
     nominate, the measurement decides." Includes honest negative verdicts.
     (§4)
  4. **Reproducible evidence in hardware-independent work units** (vector
     fetches, bytes touched, RAM residency, survivor cascades), with
     wall-clock demoted to validating a first-order cost model — plus
     negative controls and a sample-to-full-scale extrapolation check. (§6)

## 2. Background and Related Work

- Paper I in two paragraphs (Lemma: truncated-PCA distance <= true distance;
  dynamic Top-K probe; SIFT1M results).
- **Honest lineage** (position as composition-not-invention):
  - Triangle-inequality pruning: ball trees (Omohundro), Elkan's k-means
    acceleration, AESA/LAESA, VP-trees (shell bounds), GNAT.
  - Dimension-reduction bounds: GEMINI (Faloutsos et al.), VA-file.
  - Approximate cousins: FAISS IVF (same k-means geometry, no exactness),
    OPQ (per-cell rotation = the ellipsoid rung of the shape ladder).
  - Novelty claim, stated plainly: not the bounds themselves but (a) their
    exact composition in one cascade, (b) the measured self-configuration,
    (c) a diagnostic that also tells you when to walk away.

## 3. Theory: Composable Exact Bounds

- **3.1 The containing-region principle.** For any region R ⊇ cluster C:
  `||q − p|| ≥ dist(q, R)` for all p ∈ C. Bound families = shape families;
  table: ball / ellipsoid / slab / cone×annulus, with per-shape storage and
  bound-computation cost. (One paragraph noting the quantizer duality —
  VQ/OPQ/local-low-rank/gain-shape — as a remark, not a section.)
- **3.2 The ball bound.** `||q − p|| ≥ | ||q − c|| − d_p |` with stored
  `d_p = ||p − c||`. Two granularities: per-cluster radius `R_j` (prunes
  whole clusters — the out-of-core block-skip story) and per-point `d_p`
  (tighter; one gather + one subtraction per point at query time, zero
  distance computations).
- **3.3 Exactness is unconditional; quality is not.** The bound holds for
  ANY centers (triangle inequality) — k-means only makes it *tight*.
  Corollaries: stale centers stay correct; insertions need no re-clustering.
  Mirror of Paper I's "any orthogonal rotation is exact; PCA makes it tight."
- **3.4 Complementarity.** PCA bound tight ⇔ steep eigen decay (global,
  linear). Ball bound tight ⇔ small `d_p` vs. between-cluster gaps (local,
  multimodal) — and more generally low *intrinsic* dimension (wide spread of
  `d_p` and `||q−c||`). Each family covers the other's blind spot; max of
  bounds composes exactly.
  - Evidence anchor: synthetic 512-tight-clusters has flat spectrum
    (r95 = 111/128 dims) yet ball bounds prune to 7% before any PCA stage.

## 4. The Self-Configuration Funnel

- **Tier 1 — eigen curve** (from Paper I's diagnostic): rates the PCA levels.
- **Tier 2 — beyond-covariance statistics:** mean excess kurtosis of PCA
  components (marginal shape) and sqdep = mean |corr(z_i², z_j²)| (joint
  dependence PCA cannot remove; finite-sample noise floor ≈ sqrt(2/n)).
  Costs milliseconds; detects that structure *exists*, not whether it is
  *harvestable*.
- **Tier 3 — the measured trial:** fit the real hybrid on the sample, run
  the real cascade both ways, report measured gain; verdict ADD/SKIP.
- **Why tier 3 must be the decider** (a lesson section, with data — all
  numbers from the implementation-parity code):
  - **SIFT: tier 2 FIRES (sqdep 7x floor) yet the trial says SKIP (1.0x)**
    — real beyond-covariance structure exists but ball bounds cannot harvest
    it (only 50-59% pruned; the stage's cost cancels its savings).
  - **latent-8 Gaussian: tier 2 is CLEAN yet the trial says ADD (1.9x)** —
    low intrinsic dimension spreads d_p widely, so bounds bite (prune to
    23%) even with zero multimodality.
  - Statistics alone would get these two wrong in OPPOSITE directions.
  - latent-48 Gaussian: trial correctly SKIPs — and shows the ball level
    actively HURTS there (0.8x; 100% survivors = pure overhead).
  - 512 clusters: tier 2 fires (sqdep 6x floor), trial confirms (4.2x,
    prunes to 3%).
  - i.i.d. noise: everything correctly reports "nothing works" (PCA POOR,
    ball SKIP) — the honest-negative case.
- **The measured law** (across all datasets): the ball level's in-RAM value
  is a monotone function of its own pruning rate — 3% survivors -> 4-14x,
  15-23% -> ~2-3x, 50-59% -> tie, 100% -> slight loss. The trial measures
  exactly this quantity; no statistic substitutes for it.
- Evidence: `diagnose.py --demo` output (four regimes) + SIFT trial.

## 5. The Hybrid Algorithm

- **5.1 Offline:** PCA fit + k-means (k ≈ √N clamped to [64, 4096];
  MiniBatch). Stored per point: cluster id (int32) + d_p (float32) = 8 bytes;
  centroids k×d. Fit cost measured: 24s on SIFT1M (1M×128, k=1024).
- **5.2 Query pipeline (pseudocode + complexity):**
  1. k centroid distances (k·d ops).
  2. Per-point ball bounds for all N — one gather + |subtract| each.
  3. Probe: exact distances to the PROBE_K smallest bounds → initial radius.
  4. Ball pruning: `bound² < r²`.
  5. PCA cascade on survivors, **with a re-probe at the first PCA stage**
     (tighten r from the best 8D candidates before pruning).
  6. Full-D check on final survivors. Provably identical to brute force.
- **5.3 The re-probe ablation** (why it matters): without it the ball-seeded
  radius is loose and the cascade degrades (SIFT 8D survivors 27.2% vs
  14.7%); with it the cascade matches PCA-only stage-for-stage.
  Measured on loaded-machine runs: hybrid wall-clock nearly halved from this
  one fix (survivor cascade restored to PCA-only tightness; re-time under
  the 5-rep protocol for the paper).
- **5.4 Properties:** exact under arbitrary/stale centers; O(1) insertion.
- **5.5 Out-of-core layout (fragmentation analysis, measured 2026-08-23):**
  pruning fragments I/O only if naively laid out — the classic index-vs-scan
  crossover (600 scattered 512B reads lose to a full scan on seek-bound
  media). Measured on SIFT1M, k=1024 (~488KB blocks):
  (a) cluster-level bound alone skips too little — 40.4% of clusters
  survive with a 95th-percentile radius + 5% escape set (52.3% with max
  radius) → block-skipping is a ~2.5x I/O win at best;
  (b) BUT the final ~600 survivors concentrate in **~44 distinct clusters**
  (median 44/1024) → cluster-major layout converts the fetch into ~44
  contiguous ~488KB reads (~21 MB).
  → Two-tier architecture: (1) sequentially scan the tiny bound metadata
  (assign+d_p = 8 MB, optionally the 8-dim PCA sketch = 32 MB — 64x/16x
  smaller than the data; pruning DECISIONS never touch full vectors),
  (2) fetch survivors: batched random 512B reads on NVMe (~0.3 MB), or the
  ~44 cluster blocks on seek-bound media. Back-of-envelope: NVMe ~15 ms vs
  146 ms full scan (~10x); HDD ~0.55 s vs 3.4 s (~6x) — where the naive
  scattered layout would LOSE (~4.8 s). Same wisdom as VA-file / SPANN /
  DiskANN: scan small sketches sequentially, fetch survivors in layout-aware
  batches. **[GAP: wall-clock cold-cache I/O run — optional, design above]**

## 6. Experiments

All exactness-verified per query against brute force; all reproducible from
the public repo (`hybrid_search.py`, `diagnose.py`).

- **Metrics policy (state this up front, §6 opening paragraph):** primary
  claims are made in **hardware-independent work units**, because wall-clock
  proved regime-dependent (this work measured the same algorithm at
  4.7x/3.6x/2.7x/tie under dtype, load, and implementation-tuning changes,
  while work counts never moved). Primary metrics, in order: (1)
  full-vector fetches per query (out-of-core I/O currency), (2) total bytes
  touched per query INCLUDING all pruning metadata — work accounting must
  charge the bound machinery itself, or it repeats the SIFT ball-level
  mistake (2x fewer stage-1 candidates, wall-clock tie, because bound ops
  are work too), (3) resident RAM bytes/vector, (4) the survivor cascade
  (the algorithmic fingerprint; invariant through every confound). Wall-
  clock appears once, as VALIDATION of a first-order cost model
  (time ~ bytes/bandwidth + ops/throughput, FAISS as the tuned-constants
  reference) — so a reader plugs their machine's constants into our work
  numbers and predicts their own outcome. This extends the paper's thesis
  from data ("measure your data, then index") to hardware ("count the work,
  then predict the time").

- **6.0 Setup / hardware note:** Intel i9-9900K (8C/16T Coffee Lake,
  AVX2+FMA, no AVX-512), dual-channel DDR4; faiss-cpu 1.15 dispatching its
  AVX2 build (`OPTIMIZE DD AVX2`); NumPy/OpenBLAS at the X86_V3 (AVX2/FMA)
  tier — so baseline and FAISS use the SAME instruction set; FAISS's ~5.6x
  kernel advantage is fusion (single-pass, ~512 MB traffic/query) vs
  NumPy's temporaries (~3 passes, ~1.5-2 GB traffic), not SIMD presence.
  Timings are medians of 5 repetitions on a quiet machine, warm cache;
  **results are CPU-governor-insensitive** (balanced vs performance profile:
  all medians within 2%, ratios identical — cores turbo to 4.7 GHz under
  load either way and the hot loops are DRAM-bandwidth-bound). On AVX-512 /
  many-channel servers the FAISS scan advantage grows, so the in-RAM scoping
  stated here is conservative in FAISS's favor.

- **6.1 SIFT1M, official query set** (the headline table):

  | method | ms/query | speedup | survivors per stage |
  |---|---|---|---|
  Protocol: median of 5 repetitions, quiet machine, warm cache, exactness
  vs official ground truth verified every rep (`bench_faiss_5rep.py`),
  implementation parity between the compared paths (see the parity note
  below). Spreads <=2% except where noted.

  | brute (vectorized NumPy) | 167.3 | 1.0x | — |
  | PCA-only (Paper I) | 61.1 | 2.7x | 14.7 / 5.2 / 0.8 / 0.06 % |
  | hybrid | 62.2 | 2.7x | ball 50.6% → 14.6 / 5.1 / 0.7 / 0.05 % |

  **On SIFT the ball level is a wall-clock tie** — its 50% pruning saves
  about what the bound computation costs — exactly as the sample trial
  predicts (SKIP). The hybrid's SIFT-side value is therefore out-of-core /
  residency only (§5.5), and the wall-clock case for the ball level rests
  on clustered / low-intrinsic-dim data (§6.2).

  **Parity note (methodological, worth a paragraph):** two confounds were
  caught only by interrogating ratio changes across runs: (a) dtype regime
  — sklearn PCA silently yields float64, whose bandwidth-bound brute
  baseline is 1.86x costlier, inflating Paper I-style speedups (measured
  2x2: f64 3.6x vs f32 2.7x, same code); (b) implementation parity — an
  unnecessary full-array gather in the PCA-only path inflated the ball
  level's apparent contribution to 2.0x; fixing it revealed the tie.
  Recommendation baked into the paper: report survivor counts
  (implementation-independent) alongside wall-clock, and compare only
  equally-tuned paths.

  Note: PCA-only survivor cascade exactly reproduces Paper I's Fig. 1 —
  continuity between the papers.
- **6.2 Structure regimes (synthetic, n=200k/60k, parity code):** the
  verdict table — clustered (PCA-only 2.1x → hybrid **13.9x**; ball prunes
  to 3% before any projection, its bound needs no multiplies), latent-8
  (PCA-only 2.1x → hybrid 5.6x), latent-48 (ball 0.8x — hurts; SKIP),
  i.i.d. (all POOR). Negative results included deliberately.
- **6.3 Funnel extrapolation check:** the 100k-sample trial on SIFT returns
  1.0x -> verdict SKIP; the full-1M parity benchmark measured exactly that
  (61.1 vs 62.2 ms tie). The diagnostic predicts full-scale outcomes from a
  sample — including the negative verdict, which is its most valuable
  behavior (it prevents building a useless level).
- **6.4 Ablation:** re-probe on/off. Single loaded-machine runs showed
  156.6 -> 88.4 ms (survivors 27.2% -> 14.6% at 8D) **[re-measure under the
  5-rep protocol when writing — the survivor improvement is load-independent
  and already solid; the ms ratio needs the clean protocol]**.
- **6.5 FAISS-flat baseline [GAP CLOSED 2026-08-23 — scoping result]:**
  faiss-cpu 1.15, same 100 official queries, all methods 100/100 vs ground
  truth. Medians of 5 reps, quiet machine: faiss-flat wins in-RAM per-query
  (29.0 ms single-thread, 5.6 ms/query batched-100 on 16 threads) vs hybrid
  62.2 ms (= PCA-only 61.1 ms), numpy brute 167.3 ms — its SIMD kernel is
  ~5.6x NumPy at equal work. Frame as
  **work-rate vs work-reduction**: FAISS accelerates the full scan; the
  hybrid touches ~1,700x fewer full vectors (~30 MB metadata + 0.06% of
  vectors vs 512 MB/query). Claim scoped accordingly: the hybrid's regime is
  out-of-core / memory-constrained / expensive-distance settings, plus any
  implementation at native-code constants (bounds and kernels compose — the
  cascade could drive FAISS kernels on its survivor sets). Emphasize the
  **residency asymmetry**: a flat scan REQUIRES the full dataset resident in
  RAM (512 B/vector fp32); the cascade keeps 8–40 B/vector resident and
  pages full vectors from NVMe — same exactness, 12–64x less of the
  expensive resource. (Caveat to state: disk-based ANN systems — DiskANN,
  SPANN — also target this regime but are approximate; the exactness niche
  stands.) Also note: 16-thread per-query is SLOWER than 1-thread (37.7 vs
  29.0 ms) — threading pays only batched. Timing hygiene: RESOLVED — the
  5-rep quiet-machine protocol gives spreads <=2% on scan-bound methods;
  earlier ~2x same-day variance traced to background memory-bandwidth load
  (which slows the brute baseline most and therefore inflates pruner
  speedups — report quiet-machine medians only).
- **[GAP — optional]:** one more real dataset (GIST1M or a modern embedding
  set) to populate the verdict table with real-data diversity.
- **[GAP — optional]:** k sweep (centroid count vs gain), k-NN (top-k)
  generalization.

## 7. Limitations (own them explicitly, as Paper I did)

- High *intrinsic* dimensionality defeats every cheap bound (distance
  concentration flattens both the eigen curve and the d_p spread) — the
  funnel exists precisely to detect this and say "use ANN or brute force."
- Wall-clock numbers are in-RAM CPU with NumPy constants; the out-of-core
  I/O advantage (whole-cluster block skipping) is argued, not yet measured.
- 1-NN implemented; top-k is a straightforward but unimplemented extension
  (r = current k-th best).
- Sequel scope: exact search only; no claim against the ANN recall/speed
  frontier.

## 8. Conclusion + Artifact

One paragraph: bounds compose, data decides, code is public. Repo URL +
Zenodo DOI for this paper; cite Paper I.

## References (seed list)

Paper I (Chiang 2026, Zenodo) · Faloutsos et al. 1994 (GEMINI) · Weber et
al. 1998 (VA-file) · Jégou et al. 2011 (PQ) · Ge et al. 2013 (OPQ) ·
Elkan 2003 (triangle-inequality k-means) · Omohundro 1989 (ball trees) ·
Yianilos 1993 (VP-tree) · Brin 1995 (GNAT) · Malkov & Yashunin 2018 (HNSW) ·
Johnson et al. 2019 (FAISS).

---

## Claim → evidence map (for writing and for reviewers)

| Claim | Evidence | Where |
|---|---|---|
| Ball bound exact for any centers | triangle inequality, 2-line proof | §3.3 |
| SIFT1M: PCA-only 2.7x, ball level = tie (parity) | `bench_faiss_5rep.py` medians, 2026-08-23 | §6.1 |
| Work counts: pca-only 12.60M terms (10.2x), hybrid 8.67M + 1.0M bound ops, brute/faiss 128M | `count_ops` in hybrid_search.py via bench_faiss_5rep.py | §6.1 Table ops |
| dtype regime: f64 3.6x vs f32 2.7x, brute 1.86x costlier in f64 | 2x2 decomposition run, 2026-08-23 | §6.1 parity note |
| Survivor cascade = Paper I Fig. 1 | `hybrid_search.py sift`, survivor column | §6.1 |
| 13.9x on flat-spectrum clustered data (PCA-only 2.1x) | `hybrid_search.py demo`, parity code | §6.2 |
| Ball-value law: 3%->4-14x, 15-23%->2-3x, 50%->tie, 100%->0.8x | demos + SIFT trial, parity code | §4 |
| Re-probe: survivors 27.2% → 14.6% at 8D | before/after runs, same day | §6.4 |
| Tier-2 fires on clusters (sqdep 6x floor) | `diagnose.py` 512-cluster regime | §4 |
| Tier-3 must decide (latent-8 vs latent-48) | `diagnose.py --demo` | §4 |
| Sample trial extrapolates (1.7x → 2.1x) | diagnose 100k vs hybrid 1M | §6.3 |
| SIFT has beyond-covariance structure | sqdep 0.032 vs floor 0.0045 | §6.1 |
| Fit cost 24s / storage 8 B/point | SIFT fit log | §5.1 |

## Figures plan

1. **Fig 1:** the two bound geometries (schematic: PCA truncation vs ball;
   one blob-with-eigen-axes, one clustered-with-radii). New drawing.
2. **Fig 2:** funnel flowchart (tier 1→2→3, ADD/SKIP outcomes). New drawing.
3. **Fig 3:** SIFT1M survivors-per-stage, PCA-only vs hybrid (log-scale bars
   — style of Paper I Fig. 1). Data exists; regenerate.
4. **Fig 4:** the four-regime verdict table as a small-multiples chart, or
   keep as a table. Data exists.

## Open decisions for Tai

1. Venue: Zenodo (self-published, same as Paper I — no blockers) vs. a
   workshop/demo track (needs the FAISS baseline, likely k-NN too)?
2. Which GAPs to close before writing: FAISS baseline? second real dataset?
   k-NN? (My ranking: FAISS > second dataset > k-NN > out-of-core > k-sweep.)
3. Keep the quantizer-duality remark (§3.1) as one paragraph, or cut for
   focus? (It is the most conceptually interesting bit but not load-bearing.)
