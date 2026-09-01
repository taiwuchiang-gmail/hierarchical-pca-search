"""Centroid-count sensitivity sweep on SIFT1M (VLDB plan task 5).

Re-measures the k-means k sweep whose data was lost with an old session:
for each k in {8, 64, 256, 1024, 4096}, fit the hybrid index at that
cluster count and record fit time, ball-survivor rate, hybrid query time,
and exactness vs brute force. PCA-only (no ball) is the k-independent
reference. 3 repetitions x 100 official queries, medians reported.

Usage: python ksweep.py
"""

import time

import numpy as np

from hybrid_search import HybridIndex, fvecs_read

KS = (8, 64, 256, 1024, 4096)
N_REPS = 3


def main():
    base = fvecs_read("sift/sift_base.fvecs")
    queries = fvecs_read("sift/sift_query.fvecs")[:100]
    n = len(base)

    idx0 = HybridIndex(n_clusters=KS[0]).fit(base)   # PCA shared; refit per k
    brute_d = [idx0.brute(q)[1] for q in queries]

    # pca-only reference (k-independent)
    reps = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        for q in queries:
            idx0.query(q, use_ball=False)
        reps.append((time.perf_counter() - t0) / len(queries) * 1000)
    print(f"pca-only reference: {np.median(reps):.1f} ms/query "
          f"(median of {N_REPS} reps)")

    print(f"\n{'k':>5} {'fit_s':>6} {'ball_surv%':>10} {'ms/query':>9} "
          f"  (exact on all queries, {N_REPS} reps)")
    for k in KS:
        t0 = time.perf_counter()
        idx = HybridIndex(n_clusters=k).fit(base)
        fit_s = time.perf_counter() - t0
        surv, rep_ms = [], []
        for rep in range(N_REPS):
            t0 = time.perf_counter()
            for qi, q in enumerate(queries):
                _, d, stats = idx.query(q, use_ball=True)
                if rep == 0:
                    assert d == brute_d[qi], f"k={k} inexact on q{qi}"
                    surv.append(stats[1])
            rep_ms.append((time.perf_counter() - t0) / len(queries) * 1000)
        print(f"{k:>5} {fit_s:>6.0f} {np.mean(surv) / n * 100:>10.2f} "
              f"{np.median(rep_ms):>9.1f}")


if __name__ == "__main__":
    main()
