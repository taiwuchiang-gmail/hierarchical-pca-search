---
title: 'hierarchical-pca-search: Exact nearest-neighbor search with composable lower bounds and a measured prunability diagnostic'
tags:
  - Python
  - nearest-neighbor search
  - exact search
  - PCA
  - branch and bound
  - similarity search
  - out-of-core
authors:
  - name: Tai-Wu Chiang
    orcid: 0009-0004-7062-9346
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 30 August 2026
bibliography: paper.bib
---

# Summary

`hierarchical-pca-search` finds the *exact* nearest neighbor of a query
vector in a large high-dimensional dataset, typically touching only a small
fraction of the data. It composes two families of mathematically exact lower
bounds on the Euclidean distance, each covering the other's blind spot:

1. **PCA truncation bounds** [@chiang2026hierarchical]: after rotating the
   data onto its principal axes, the distance computed on the first $k$
   coordinates is a lower bound on the true distance. A cascade of
   increasingly tight bounds ($8 \to 16 \to 32 \to 64$ dimensions) prunes
   candidates against a dynamically discovered search radius. These bounds
   are tight when variance concentrates in few directions — linearly
   correlated data.
2. **Ball (triangle-inequality) bounds** [@chiang2026composable]: for a
   point $p$ in a k-means cluster with centroid $c$ and stored offline
   distance $d_p = \lVert p-c \rVert$, the triangle inequality gives
   $\lVert q-p \rVert \ge \big|\, \lVert q-c \rVert - d_p \,\big|$. This
   costs one gather and one subtraction per point — no per-point distance
   computations — and is tight exactly where the PCA bound is loose:
   clustered, multimodal data. The bound is exact for *any* choice of
   centers; center quality affects speed, never correctness.

Because the maximum of lower bounds is a lower bound, the two levels compose
freely, and the result is provably identical to brute force — the test suite
verifies this per query, on every data regime, for every code path. On the
SIFT1M benchmark the PCA cascade evaluates 12.6 million per-dimension
distance terms per query versus the 128 million of any exhaustive scan — a
10.2$\times$ algorithmic work reduction with 100% exactness — while storing
only 8–40 bytes of resident metadata per vector against 512 bytes for the
raw vectors, which is what makes an out-of-core deployment attractive.

The package deliberately reports *hardware-independent work counts*
(`count_ops`) alongside wall-clock time: per-dimension subtract-square terms
and bound evaluations are counted exactly from the survivor statistics of
each query, so results are reproducible across machines and comparisons
against compiled baselines are fair by construction.

# Statement of need

Modern similarity-search libraries such as FAISS [@johnson2019billion] and
HNSW [@malkov2020efficient] are *approximate*: they trade recall for speed.
That trade is often right, but a persistent set of tasks requires exactness
— generating ground truth to evaluate those same ANN indexes, deduplication,
record linkage, and legal or medical retrieval where a missed true neighbor
is a correctness bug rather than a benchmark artifact. For exact search in
high dimensions the standard answer has been the optimized flat scan, since
classic space-partitioning trees degrade to worse-than-brute-force behavior
there [@weber1998quantitative]. A flat scan, however, cannot exploit the
structure most real embedding datasets actually have: strong correlation
(low intrinsic dimension) and cluster structure.

The second, less obvious need is *predictability*. Whether bound-based
pruning helps depends entirely on the data's correlation structure, so a
practitioner cannot know in advance whether integrating an index will pay
off. `diagnose.py` answers this in seconds on a sample of the user's own
vectors with a three-tier funnel: (1) the eigenvalue decay curve rates the
PCA levels; (2) kurtosis and squared-dependence statistics detect structure
beyond the covariance that a ball level could exploit; (3) a measured trial
of the real hybrid cascade on the sample delivers the final ADD/SKIP
verdict. The design principle — *statistics nominate, measurement decides* —
exists because the statistics are provably insufficient: on SIFT1M the
tier-2 statistics fire yet the measured ball level is a wall-clock tie,
while on low-intrinsic-dimension data the statistics are clean yet the
measured trial shows a real gain [@chiang2026composable].

The intended audience is researchers and engineers who need exact
nearest-neighbor results on correlated, high-dimensional data — in RAM or
out-of-core — and anyone building ANN evaluation pipelines who currently
pays for exhaustive ground-truth scans. The implementation is pure
NumPy [@harris2020array] (scikit-learn [@pedregosa2011scikit] optionally
accelerates the offline k-means fit), installs with `pip`, and reproduces
the published SIFT1M results [@jegou2011product] with one command.

# Functionality

- `HybridIndex` — the exact 1-NN index: `fit(X)` (PCA rotation + k-means +
  offline centroid distances), `query(q, use_ball=...)` (ball level plus PCA
  cascade, or PCA cascade alone), and `brute(q)` (the verification
  baseline).
- `diagnose.py` / `pca-diagnose` — the prunability diagnostic CLI for
  `.npy`/`.fvecs` files: eigen-decay report, measured cascade with per-query
  exactness verification, three-tier ball-level funnel, plain-language
  verdicts, optional plot.
- `count_ops` — exact, hardware-independent work accounting from survivor
  statistics.
- `benchmark.py` — one-command reproduction of the SIFT1M results, including
  dataset download.

# Acknowledgements

The SIFT1M dataset is provided by the TEXMEX corpus
[@jegou2011product]. Development was assisted by Anthropic's Claude;
all algorithms, measurements, and claims were designed and verified by the
author.

# References
