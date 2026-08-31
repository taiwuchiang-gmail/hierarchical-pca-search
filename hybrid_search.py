"""Exact hybrid nearest-neighbor search: ball (k-means) bound + PCA cascade.

Composes two families of exact lower bounds, each covering the other's blind
spot:

  PCA bound (the paper): dist of k-dim PCA truncations <= true dist.
      Tight when variance concentrates in few directions (linear correlation).
  Ball bound (triangle inequality): for point p in cluster with centroid c,
      ||q - p|| >= | ||q - c|| - d_p |,   d_p = ||p - c|| (stored offline).
      Tight when points huddle near centroids (clustered / multimodal data).

Both are unconditionally exact: bound quality only affects SPEED, never
correctness. The max of lower bounds is a lower bound, so the two levels
compose freely. Offline cost: PCA fit + k-means; per point we store one int32
(cluster id) and one float32 (d_p) -- ~8 bytes/point.

Query pipeline:
  1. k centroid distances (k << N full-D distances).
  2. Per-point ball bounds for ALL N points -- pure subtractions, no distance
     computations: |dq[assign] - d_p|.
  3. Probe: exact distances to the PROBE points with smallest ball bounds
     -> mathematically safe initial radius r (same trick as the paper's
     Top-50 probe, but seeded by ball bounds instead of 8-D distances).
  4. Ball pruning: keep points with ball bound < r.
  5. PCA cascade [8, 16, 32, 64] on the survivors (the paper's algorithm).
  6. Full-D check on the final survivors. Result provably == brute force.

Usage:
    python hybrid_search.py demo       # synthetic showcase: PCA-hostile
                                       # (many tight clusters) vs PCA-friendly
    python hybrid_search.py sift       # SIFT1M with the official query set,
                                       # brute vs PCA-only vs hybrid
"""

import argparse
import os
import time

import numpy as np

PROBE_K = 50


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def fit_kmeans(X, k, seed=0):
    """MiniBatch k-means (sklearn if present, else chunked numpy Lloyd)."""
    try:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=3,
                             max_iter=100, random_state=seed)
        assign = km.fit_predict(X)
        return km.cluster_centers_.astype(np.float32), assign.astype(np.int32)
    except ImportError:
        rng = np.random.default_rng(seed)
        C = X[rng.choice(len(X), k, replace=False)].astype(np.float32)
        assign = np.zeros(len(X), np.int32)
        for _ in range(20):
            for s in range(0, len(X), 100_000):
                blk = X[s:s + 100_000]
                d2 = ((blk ** 2).sum(1)[:, None] - 2 * blk @ C.T
                      + (C ** 2).sum(1))
                assign[s:s + 100_000] = d2.argmin(1)
            for j in range(k):
                m = assign == j
                if m.any():
                    C[j] = X[m].mean(0)
        return C, assign


def count_ops(stats, levels, d, n_clusters=0, use_ball=True, probe=PROBE_K):
    """Exact per-query work counts from the survivor statistics.

    Returns (terms, bound_ops). `terms` counts per-dimension (x-y)^2
    evaluations -- one full d-dim distance = d terms -- so an exhaustive
    scan is N*d terms by definition (which also gives the count for any
    compiled flat-scan library, no instrumentation required). `bound_ops`
    counts per-point ball-bound evaluations (one gather + one |subtract|
    each), charged separately because they are cheaper than a term but not
    free: work accounting must charge the pruning machinery itself.
    Both are hardware- and implementation-independent.
    """
    levels = list(levels)
    n = stats[0]
    if use_ball:
        want = 2 + len(levels)               # [N, ball, one per level]
        st = list(stats) + [stats[-1]] * (want - len(stats))
        terms = n_clusters * d + probe * d   # centroid dists + probe
        bound_ops = float(n)                 # one bound per point
        entering = st[1]                     # ball survivors enter stage 1
        for i, k_dims in enumerate(levels):
            terms += entering * k_dims
            if i == 0 and entering > probe:
                terms += probe * d           # radius re-probe
            entering = st[2 + i]
        terms += st[-1] * d                  # final full-distance check
    else:
        want = 1 + len(levels)               # [N, one per level]
        st = list(stats) + [stats[-1]] * (want - len(stats))
        terms = n * levels[0] + probe * d    # stage-1 scan doubles as probe
        bound_ops = 0.0
        entering = st[1]
        for i, k_dims in enumerate(levels[1:]):
            terms += entering * k_dims
            entering = st[2 + i]
        terms += st[-1] * d
    return float(terms), bound_ops


def _kth_smallest_unique(ev_idx, ev_d, k):
    """k-th smallest distance over DISTINCT evaluated points.

    The evaluated set can contain the same point twice (probe and re-probe
    overlap); the k-th order statistic of a multiset with duplicates is
    biased low, which would over-prune -- an exactness bug, not a speed
    issue. min() (the 1-NN special case) is duplicate-invariant; the k-th
    order statistic is not.
    """
    uniq, first = np.unique(ev_idx, return_index=True)
    ud = ev_d[first]
    return float(np.partition(ud, k - 1)[k - 1]), uniq, ud


class HybridIndex:
    """Exact k-NN index: ball bound level + hierarchical PCA cascade."""

    def __init__(self, levels=(8, 16, 32, 64), n_clusters=None):
        self.levels = list(levels)
        self.n_clusters = n_clusters

    def fit(self, X):
        X = X.astype(np.float32)
        n, d = X.shape
        self.levels = [k for k in self.levels if k < d]

        self.mean = X.mean(0)
        Xc = (X - self.mean).astype(np.float64)
        cov = (Xc.T @ Xc) / (n - 1)
        eigval, eigvec = np.linalg.eigh(cov)
        self.rot = eigvec[:, ::-1].astype(np.float32)     # descending variance
        self.X = ((X - self.mean) @ self.rot).astype(np.float32)

        k = self.n_clusters or int(np.clip(np.sqrt(n), 64, 4096))
        self.C, self.assign = fit_kmeans(self.X, k)
        diff = self.X - self.C[self.assign]
        self.d_p = np.sqrt((diff ** 2).sum(1)).astype(np.float32)
        self.R = np.zeros(k, np.float32)                  # cluster radii
        np.maximum.at(self.R, self.assign, self.d_p)
        return self

    def _transform(self, q):
        return ((q.astype(np.float32) - self.mean) @ self.rot)

    def query(self, q_raw, use_ball=True, k=1):
        """Exact k-NN. For k=1 returns (index, dist_sq, survivors-per-stage);
        for k>1 the first two are arrays sorted by distance.

        The pruning radius is the k-th smallest exact distance among the
        points evaluated so far (probe, re-probe): the k-th order statistic
        over a subset can only overestimate the true one, so any point whose
        lower bound reaches it provably cannot enter the k-NN set. Requires
        k < n (at least one prunable point).
        """
        q = self._transform(q_raw)
        X, N = self.X, len(self.X)
        if not 1 <= k < N:
            raise ValueError(f"k must be in [1, {N - 1}], got {k}")
        n_probe = min(max(PROBE_K, k), N - 1)
        stats = [N]

        if use_ball:
            dq = np.sqrt(((self.C - q) ** 2).sum(1))          # k centroid dists
            bound = np.abs(dq[self.assign] - self.d_p)        # N subtractions
            probe = np.argpartition(bound, n_probe)[:n_probe]
        else:
            d8 = ((X[:, :self.levels[0]] - q[:self.levels[0]]) ** 2).sum(1)
            probe = np.argpartition(d8, n_probe)[:n_probe]

        ev_idx = probe                                        # evaluated so far
        ev_d = ((X[probe] - q) ** 2).sum(1)
        radius, _, _ = _kth_smallest_unique(ev_idx, ev_d, k)  # k-th best

        if use_ball:
            cand = np.flatnonzero(bound * bound < radius)     # ball pruning
            stats.append(len(cand))
            start = 0
        else:
            # reuse the stage-1 distances already computed for the probe --
            # no full-array gather (implementation parity with the ball path)
            cand = np.flatnonzero(d8 < radius)
            stats.append(len(cand))
            start = 1

        for i, k_dims in enumerate(self.levels):
            if i < start or not len(cand):
                if not len(cand):
                    break
                continue
            dk = ((X[cand, :k_dims] - q[:k_dims]) ** 2).sum(1)
            if i == 0 and use_ball and len(cand) > n_probe:
                # Re-probe: the ball-seeded radius can be loose; tighten it
                # from the best first-stage candidates before pruning.
                top = np.argpartition(dk, n_probe)[:n_probe]
                ev_idx = np.concatenate([ev_idx, cand[top]])
                ev_d = np.concatenate([ev_d, ((X[cand[top]] - q) ** 2).sum(1)])
                radius, _, _ = _kth_smallest_unique(ev_idx, ev_d, k)
            cand = cand[dk < radius]
            stats.append(len(cand))

        if len(cand):
            ev_idx = np.concatenate([ev_idx, cand])
            ev_d = np.concatenate([ev_d, ((X[cand] - q) ** 2).sum(1)])

        # probe / re-probe / finalists may overlap: dedupe, take k smallest
        _, uniq, ud = _kth_smallest_unique(ev_idx, ev_d, k)
        sel = np.argpartition(ud, k - 1)[:k]
        order = np.argsort(ud[sel], kind="stable")
        out_idx, out_d = uniq[sel][order], ud[sel][order]
        if k == 1:
            return int(out_idx[0]), float(out_d[0]), stats
        return out_idx.astype(np.int64), out_d, stats

    def brute(self, q_raw, k=1):
        q = self._transform(q_raw)
        d2 = ((self.X - q) ** 2).sum(1)
        if k == 1:
            return int(d2.argmin()), float(d2.min())
        sel = np.argpartition(d2, k - 1)[:k]
        order = np.argsort(d2[sel], kind="stable")
        return sel[order].astype(np.int64), d2[sel][order]


def bench(X, queries, name, n_clusters=None):
    print(f"\n=== {name}: {len(X):,} x {X.shape[1]}, "
          f"{len(queries)} queries ===")
    t0 = time.perf_counter()
    idx = HybridIndex(n_clusters=n_clusters).fit(X)
    print(f"  offline fit (PCA + k-means, k={len(idx.C)}): "
          f"{time.perf_counter() - t0:.1f}s")

    times = {'brute': 0.0, 'pca-only': 0.0, 'hybrid': 0.0}
    stats = {'pca-only': [], 'hybrid': []}
    for q in queries:
        t0 = time.perf_counter(); b_idx, b_d = idx.brute(q)
        times['brute'] += time.perf_counter() - t0
        t0 = time.perf_counter(); p_idx, p_d, ps = idx.query(q, use_ball=False)
        times['pca-only'] += time.perf_counter() - t0
        t0 = time.perf_counter(); h_idx, h_d, hs = idx.query(q, use_ball=True)
        times['hybrid'] += time.perf_counter() - t0
        assert b_d == p_d == h_d, "exactness violated!"    # dist always unique
        stats['pca-only'].append(ps + [ps[-1]] * (len(idx.levels) + 2 - len(ps)))
        stats['hybrid'].append(hs + [hs[-1]] * (len(idx.levels) + 2 - len(hs)))

    n, nq = len(X), len(queries)
    print(f"  exactness: all {nq} queries match brute force")
    print(f"  {'method':>9} {'ms/query':>9} {'speedup':>8}   survivors/stage")
    for m in ('brute', 'pca-only', 'hybrid'):
        ms = times[m] / nq * 1000
        sp = times['brute'] / times[m]
        surv = ""
        if m in stats:
            s = np.mean(stats[m], axis=0)
            labels = (['ball'] if m == 'hybrid' else []) + \
                     [f"{k}D" for k in idx.levels]
            surv = "  ".join(f"{l}:{v / n * 100:.2f}%"
                             for l, v in zip(labels, s[1:]))
        print(f"  {m:>9} {ms:>9.1f} {sp:>7.1f}x   {surv}")


def demo():
    rng = np.random.default_rng(0)
    n, d = 200_000, 64

    # PCA-hostile: 512 tight clusters whose centers span ALL dimensions ->
    # flat eigen decay (PCA bound loose) but tiny radii (ball bound razor).
    centers = rng.normal(size=(512, d))
    lab = rng.integers(0, 512, n)
    clustered = (centers[lab] + 0.05 * rng.normal(size=(n, d))
                 ).astype(np.float32)

    # PCA-friendly: low-rank correlated Gaussian -> steep eigen decay
    # (PCA bound tight) but one big blob (ball bound useless).
    lowrank = (rng.normal(size=(n, 8)) @ rng.normal(size=(8, d))
               + 0.1 * rng.normal(size=(n, d))).astype(np.float32)

    for name, X in [("many tight clusters (PCA-hostile)", clustered),
                    ("low-rank Gaussian (PCA-friendly)", lowrank)]:
        qi = rng.choice(n, 100, replace=False)
        mask = np.ones(n, bool); mask[qi] = False
        bench(X[mask], X[qi], name)


def sift():
    base = fvecs_read('sift/sift_base.fvecs')
    queries = fvecs_read('sift/sift_query.fvecs')[:100]
    bench(base, queries, "SIFT1M (official queries)", n_clusters=1024)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('mode', nargs='?', default='demo',
                    choices=['demo', 'sift'],
                    help="'demo' (synthetic showcase) or 'sift' (SIFT1M)")
    args = ap.parse_args()
    if args.mode == 'sift':
        if not os.path.exists('sift'):
            raise SystemExit("run from the repo root after benchmark.py "
                             "has downloaded ./sift")
        sift()
    else:
        demo()
