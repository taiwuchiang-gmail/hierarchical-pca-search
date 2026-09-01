"""Survivor locality: how many disk blocks would each query touch?

For each query, the cascade's full-D fetch set (probe + re-probe +
finalists) is mapped onto the k-means clusters. Under a cluster-major
on-disk layout, one touched cluster = one contiguous block read. This
predicts the cold-cache I/O outcome entirely from in-RAM quantities:

  bytes(full scan)      = n * d * 4
  bytes(point fetch)    = fetched pts * ceil to 4 KiB pages (orig. order,
                          assumes >= 1 page per point: scattered reads)
  bytes(block fetch)    = touched clusters * avg cluster bytes

Usage: python locality_stats.py <dataset> [levels]  (texmex dir or .npy)
"""

import sys
import time

import numpy as np

from hybrid_search import HybridIndex, fvecs_read

N_QUERIES = 100
PAGE = 4096


def run(name, levels=(8, 16, 32, 64), n_clusters=1024, k_nn=1):
    if name.endswith(".npy"):
        X = np.load(name, mmap_mode="r")
        rng = np.random.default_rng(0)
        q_idx = rng.choice(len(X), N_QUERIES, replace=False)
        mask = np.ones(len(X), bool)
        mask[q_idx] = False
        base, queries = np.asarray(X[mask]), np.asarray(X[q_idx])
        del X
    else:
        base = fvecs_read(f"{name}/{name}_base.fvecs")
        queries = fvecs_read(f"{name}/{name}_query.fvecs")[:N_QUERIES]

    n, d = base.shape
    t0 = time.perf_counter()
    idx = HybridIndex(levels=levels, n_clusters=n_clusters).fit(base)
    print(f"{name}: {n:,} x {d}, levels {idx.levels}, "
          f"{len(idx.C)} clusters; fit {time.perf_counter() - t0:.0f}s")

    vec_bytes = d * 4
    scan_mb = n * vec_bytes / 2**20
    csize = np.bincount(idx.assign, minlength=len(idx.C))   # cluster sizes
    fetched, clusters, blk_mb, pt_mb = [], [], [], []
    for q in queries:
        idx.query(q, k=k_nn)
        ids = idx.last_fetched
        cl = np.unique(idx.assign[ids])
        fetched.append(len(ids))
        clusters.append(len(cl))
        blk_mb.append(csize[cl].sum() * vec_bytes / 2**20)
        pt_mb.append(len(ids) * max(vec_bytes, PAGE) / 2**20)

    f, c = np.mean(fetched), np.mean(clusters)
    bm, pm = np.mean(blk_mb), np.mean(pt_mb)
    print(f"  full-D fetch set: {f:,.0f} pts ({f / n * 100:.2f}%) in "
          f"{c:.0f}/{len(idx.C)} clusters "
          f"(uniform placement would hit ~{len(idx.C) * (1 - (1 - 1 / len(idx.C)) ** f):.0f})")
    print(f"  bytes/query: full scan {scan_mb:,.0f} MB | "
          f"point-fetch (orig order, 4K pages) {pm:,.1f} MB | "
          f"block-fetch (cluster-major) {bm:,.1f} MB")
    print(f"  I/O reduction vs scan: point {scan_mb / pm:,.0f}x, "
          f"block {scan_mb / bm:,.0f}x; "
          f"reads/query: point {f:,.0f} small vs block {c:.0f} large")


if __name__ == "__main__":
    lv = sys.argv[2] if len(sys.argv) > 2 else "8,16,32,64"
    lv = "auto" if lv == "auto" else tuple(int(x) for x in lv.split(","))
    run(sys.argv[1], levels=lv)
