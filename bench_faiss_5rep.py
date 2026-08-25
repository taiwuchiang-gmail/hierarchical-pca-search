"""FAISS-flat vs numpy-brute vs PCA-only vs hybrid on SIFT1M, 5 repetitions.

Timing hygiene: both indexes fit once; one untimed warm-up pass; then each
configuration is timed over the 100 official queries, 5 repetitions; report
median ms/query with min-max spread. Ground truth verified every rep.
"""
import time

import numpy as np
import faiss

from hybrid_search import fvecs_read, ivecs_read, HybridIndex

REPS = 5

base = fvecs_read('sift/sift_base.fvecs')
queries = fvecs_read('sift/sift_query.fvecs')[:100]
gt = ivecs_read('sift/sift_groundtruth.ivecs')[:100, 0]

findex = faiss.IndexFlatL2(base.shape[1])
findex.add(base)
hidx = HybridIndex(n_clusters=1024)
t0 = time.perf_counter(); hidx.fit(base)
print(f"setup: faiss add + hybrid fit done ({time.perf_counter()-t0:.0f}s fit)")


def faiss_per_query(nthreads):
    faiss.omp_set_num_threads(nthreads)
    ids, dt = [], 0.0
    for q in queries:
        t0 = time.perf_counter()
        _, I = findex.search(q[None, :], 1)
        dt += time.perf_counter() - t0
        ids.append(I[0, 0])
    return dt, np.array(ids)


def faiss_batch(nthreads):
    faiss.omp_set_num_threads(nthreads)
    t0 = time.perf_counter()
    _, I = findex.search(queries, 1)
    return time.perf_counter() - t0, I[:, 0]


def ours(fn):
    ids, dt = [], 0.0
    for q in queries:
        t0 = time.perf_counter()
        i = fn(q)
        dt += time.perf_counter() - t0
        ids.append(i)
    return dt, np.array(ids)


CONFIGS = [
    ("faiss-flat 1thr, per-query", lambda: faiss_per_query(1)),
    ("faiss-flat 16thr, per-query", lambda: faiss_per_query(16)),
    ("faiss-flat 16thr, batch-100", lambda: faiss_batch(16)),
    ("numpy brute, per-query", lambda: ours(lambda q: hidx.brute(q)[0])),
    ("pca-only, per-query", lambda: ours(lambda q: hidx.query(q, use_ball=False)[0])),
    ("hybrid, per-query", lambda: ours(lambda q: hidx.query(q, use_ball=True)[0])),
]

# warm-up (untimed): touch every code path and page in all arrays
for name, fn in CONFIGS:
    fn()
print("warm-up pass done")

times = {name: [] for name, _ in CONFIGS}
gt_ok = {name: True for name, _ in CONFIGS}
for rep in range(REPS):
    for name, fn in CONFIGS:
        dt, ids = fn()
        times[name].append(dt / len(queries) * 1000)
        gt_ok[name] &= bool((ids == gt).all())
    print(f"rep {rep + 1}/{REPS} done")

print(f"\n{'method':>28} {'median':>8} {'min':>7} {'max':>7} {'GT':>5}")
med_brute = np.median(times["numpy brute, per-query"])
for name, _ in CONFIGS:
    t = np.array(times[name])
    print(f"{name:>28} {np.median(t):>8.1f} {t.min():>7.1f} {t.max():>7.1f} "
          f"{'ok' if gt_ok[name] else 'FAIL':>5}")
print(f"\nspeedups vs numpy-brute median ({med_brute:.1f} ms): "
      + ", ".join(f"{n.split(',')[0]}={med_brute/np.median(times[n]):.1f}x"
                  for n, _ in CONFIGS if n != 'numpy brute, per-query'))
