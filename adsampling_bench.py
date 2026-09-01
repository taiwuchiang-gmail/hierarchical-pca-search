"""ADSampling baseline (Gao & Long, SIGMOD 2023) in the parity harness.

Faithful NumPy port of the reference DCO (adsampling_ref/src/adsampling.h):
random orthogonal rotation offline, then per candidate an incremental
partial squared distance over dimension chunks of delta_d, pruned by the
hypothesis test  partial >= r^2 * (i/D) * (1 + eps0/sqrt(i))^2.  Points
that survive all D dims yield exact distances; pruning is statistical,
so results carry a (small) failure probability -- recall is measured,
not guaranteed. This is the contrast with our deterministic bounds.

Parity notes vs the reference linear scan: candidates are processed in
blocks of BLOCK points; the pruning radius (k-th best exact distance)
updates between blocks, not between individual points. Work is counted
as dims touched per candidate (their tot_dimension) = per-dimension
(x-y)^2 term evaluations, the same currency as count_ops in
hybrid_search.py.

Usage: python adsampling_bench.py <dataset> [k] [eps0]
       dataset: texmex dir (sift, gist) or .npy file
"""

import sys
import time

import numpy as np

from hybrid_search import fvecs_read

DELTA_D = 32
BLOCK = 4096
N_QUERIES = 100


def load(name):
    if name.endswith(".npy"):
        X = np.load(name, mmap_mode="r")
        rng = np.random.default_rng(0)
        q_idx = rng.choice(len(X), N_QUERIES, replace=False)
        mask = np.ones(len(X), bool)
        mask[q_idx] = False
        return np.asarray(X[mask]), np.asarray(X[q_idx])
    base = fvecs_read(f"{name}/{name}_base.fvecs")
    return base, fvecs_read(f"{name}/{name}_query.fvecs")[:N_QUERIES]


def rotate(X, Q, chunk=100_000):
    out = np.empty_like(X)
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = X[i:i + chunk] @ Q
    return out


def ratios(D, eps0):
    """ratio(D, i) from adsampling.h for each checkpoint i."""
    i = np.arange(DELTA_D, D + DELTA_D, DELTA_D).clip(max=D).astype(np.float64)
    r = i / D * (1.0 + eps0 / np.sqrt(i)) ** 2
    r[i == D] = 1.0                       # exact at full dimensionality
    return i.astype(np.int64), r.astype(np.float32)


def query_scan(Xr, q, k, eps0, io):
    """ADSampling linear scan: returns (ids, exact sq dists) of the k-NN
    candidates it believes in; io accumulates dims touched."""
    n, D = Xr.shape
    ck_i, ck_r = ratios(D, eps0)
    best_d = np.full(k, np.inf, np.float32)
    best_i = np.full(k, -1, np.int64)
    r2 = np.float32(np.inf)
    for s in range(0, n, BLOCK):
        blk = Xr[s:s + BLOCK]
        m = len(blk)
        alive = np.arange(m)
        res = np.zeros(m, np.float32)
        for ci in range(len(ck_i)):
            lo = 0 if ci == 0 else int(ck_i[ci - 1])
            hi = int(ck_i[ci])
            d = blk[alive, lo:hi] - q[lo:hi]
            res[alive] += np.einsum("ij,ij->i", d, d)
            io[0] += len(alive) * (hi - lo)
            keep = res[alive] < r2 * ck_r[ci]
            alive = alive[keep]
            if not len(alive):
                break
        if len(alive):                    # survivors have exact distances
            alld = np.concatenate([best_d, res[alive]])
            alli = np.concatenate([best_i, s + alive])
            sel = np.argpartition(alld, k - 1)[:k]
            best_d, best_i = alld[sel], alli[sel]
            r2 = best_d.max()
    o = np.argsort(best_d, kind="stable")
    return best_i[o], best_d[o]


def main(name, k=1, eps0=2.1):
    base, queries = load(name)
    n, D = base.shape
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D), dtype=np.float32))
    t0 = time.perf_counter()
    Xr = rotate(base, Q)
    print(f"{name}: {n:,} x {D}, k={k}, eps0={eps0}, delta_d={DELTA_D}, "
          f"block={BLOCK}; rotate {time.perf_counter() - t0:.0f}s")

    norms = np.einsum("ij,ij->i", Xr, Xr)   # for the ground-truth check
    times, dims, rec = [], [], []
    for q in queries:
        qr = q @ Q
        io = [0]
        t0 = time.perf_counter()
        ids, dd = query_scan(Xr, qr, k, eps0, io)
        times.append(time.perf_counter() - t0)
        dims.append(io[0])
        # ground truth on the rotated data (orthogonal: same distances)
        d2 = norms - 2.0 * (Xr @ qr) + qr @ qr
        sel = np.sort(np.partition(d2, k - 1)[:k])
        # tie-tolerant recall: match on distances, not ids
        # rtol loose enough for f32 norms-identity vs incremental rounding
        rec.append(np.isclose(np.sort(dd), sel, rtol=1e-4).mean())

    terms = np.mean(dims)
    print(f"  ms/query (scan only): {np.median(times) * 1000:.1f}")
    print(f"  dims touched/query: {terms / 1e6:.2f}M terms "
          f"(brute = {n * D / 1e6:.1f}M; reduction {n * D / terms:.1f}x)")
    print(f"  recall@{k}: {np.mean(rec):.4f} "
          f"({int(np.sum(np.array(rec) < 1.0))}/{len(rec)} queries imperfect)")


if __name__ == "__main__":
    main(sys.argv[1],
         int(sys.argv[2]) if len(sys.argv) > 2 else 1,
         float(sys.argv[3]) if len(sys.argv) > 3 else 2.1)
