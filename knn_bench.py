"""k-NN benchmark on SIFT1M: wall-clock, survivors, and work counts vs k.

Extends the 1-NN measurement (Paper II, Table 2) to k in {1, 10, 100}.
Exactness is asserted per query against brute force; k=100 is additionally
checked against the official ground truth (which ships exactly 100 NN).

Usage: python knn_bench.py [dataset] [levels]
    dataset: texmex dir (sift, gist) or a .npy file (100 rows held out as
             queries, no ground-truth check); default sift
    levels:  'auto' (eigen-curve selection) or e.g. 8,16,32,64 (default)
"""

import sys
import time

import numpy as np

from hybrid_search import PROBE_K, HybridIndex, count_ops, fvecs_read, ivecs_read

K_VALUES = (1, 10, 100)
N_QUERIES = 100


def run(name="sift", levels=(8, 16, 32, 64)):
    if name.endswith(".npy"):
        X = np.load(name, mmap_mode="r")     # 6 GB file: don't double-load
        rng = np.random.default_rng(0)
        q_idx = rng.choice(len(X), N_QUERIES, replace=False)
        mask = np.ones(len(X), bool)
        mask[q_idx] = False
        base, queries, gt = np.asarray(X[mask]), np.asarray(X[q_idx]), None
        del X
    else:
        base = fvecs_read(f"{name}/{name}_base.fvecs")
        queries = fvecs_read(f"{name}/{name}_query.fvecs")[:N_QUERIES]
        gt = ivecs_read(f"{name}/{name}_groundtruth.ivecs")[:N_QUERIES]
    n, d = base.shape

    t0 = time.perf_counter()
    idx = HybridIndex(levels=levels, n_clusters=1024).fit(base)
    print(f"{name.upper()}: {n:,} x {d}, {len(queries)} queries, "
          f"levels {idx.levels}; fit {time.perf_counter() - t0:.1f}s")
    if gt is None:
        del base                 # only needed for the ground-truth tie check

    for k in K_VALUES:
        probe = max(PROBE_K, k)
        times = {"brute": 0.0, "pca-only": 0.0, "hybrid": 0.0}
        ops = {"pca-only": [], "hybrid": []}
        surv = {"pca-only": [], "hybrid": []}
        for qi, q in enumerate(queries):
            t0 = time.perf_counter()
            b_idx, b_d = idx.brute(q, k=k)
            times["brute"] += time.perf_counter() - t0
            for method, use_ball in (("pca-only", False), ("hybrid", True)):
                t0 = time.perf_counter()
                h_idx, h_d, st = idx.query(q, use_ball=use_ball, k=k)
                times[method] += time.perf_counter() - t0
                if k == 1:
                    assert h_d == b_d, f"exactness violated ({method}, q{qi})"
                else:
                    assert np.array_equal(h_d, b_d), \
                        f"exactness violated ({method}, k={k}, q{qi})"
                    if gt is not None and k == gt.shape[1]:
                        # exact distance ties at the k-th boundary make the
                        # k-NN set non-unique (SIFT/GIST are integer-valued
                        # descriptors -- ties are real, e.g. SIFT q13):
                        # any disagreement with GT must sit at the boundary
                        diff = set(np.atleast_1d(h_idx).tolist()) \
                            ^ set(gt[qi].tolist())
                        if diff:
                            dd = ((base[sorted(diff)] - q) ** 2).sum(1)
                            kth = ((base[gt[qi]] - q) ** 2).sum(1).max()
                            assert np.allclose(dd, kth), \
                                f"ground-truth mismatch beyond ties (q{qi})"
                terms, bops = count_ops(st, idx.levels, d,
                                        n_clusters=len(idx.C),
                                        use_ball=use_ball, probe=probe)
                ops[method].append((terms, bops))
                surv[method].append(st[1] / n)

        nq = len(queries)
        print(f"\n  k={k} (exact on all {nq} queries)")
        print(f"  {'method':>9} {'ms/query':>9} {'speedup':>8} "
              f"{'Mterms':>8} {'Mbound':>7} {'stage1-surv':>12}")
        print(f"  {'brute':>9} {times['brute'] / nq * 1e3:>9.1f} "
              f"{'1.0x':>8} {n * d / 1e6:>8.2f} {0.0:>7.2f}")
        for m in ("pca-only", "hybrid"):
            terms = np.mean([o[0] for o in ops[m]])
            bops = np.mean([o[1] for o in ops[m]])
            print(f"  {m:>9} {times[m] / nq * 1e3:>9.1f} "
                  f"{times['brute'] / times[m]:>7.1f}x "
                  f"{terms / 1e6:>8.2f} {bops / 1e6:>7.2f} "
                  f"{np.mean(surv[m]) * 100:>11.2f}%")


if __name__ == "__main__":
    lv = sys.argv[2] if len(sys.argv) > 2 else "8,16,32,64"
    lv = "auto" if lv == "auto" else tuple(int(x) for x in lv.split(","))
    run(sys.argv[1] if len(sys.argv) > 1 else "sift", levels=lv)
