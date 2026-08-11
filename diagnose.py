"""Prunability diagnostic: will hierarchical PCA pruning help YOUR data?

Answers that question in seconds, on a sample of your vectors, BEFORE you
integrate anything:

  1. Fits PCA (numpy-only) on a sample and reports the eigenvalue decay --
     the "how correlated are my dimensions" signal. The k-D PCA distance is a
     lower bound whose average tightness equals the cumulative explained
     variance at k, so fast decay => tight bounds at low k => hard pruning.
  2. Runs the ACTUAL pruning cascade (same logic as benchmark.py, top-50
     probe included) on the sample and measures survivor counts, exactness,
     and wall-clock speedup vs vectorized brute force.
  3. Prints a plain verdict: strong / moderate / poor prunability.

Usage:
    python diagnose.py vectors.npy                 # rows = vectors
    python diagnose.py sift/sift_base.fvecs        # fvecs also supported
    python diagnose.py --demo                      # synthetic tour: high /
                                                   # medium / no correlation
Options:
    --levels 8,16,32,64    cascade dimensions (default; auto-clipped to d)
    --sample 100000        max vectors sampled from the file
    --queries 100          held-out sample rows used as queries
    --plot out.png         save eigen-decay + survivors chart (needs matplotlib)
"""

import argparse
import sys
import time

import numpy as np

PROBE_K = 50


# --- Data loading -----------------------------------------------------------

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def load_vectors(path):
    if path.endswith('.npy'):
        return np.load(path)
    if path.endswith('.fvecs'):
        return fvecs_read(path)
    raise ValueError(f"Unsupported file type: {path} (use .npy or .fvecs)")


# --- PCA (numpy-only, no sklearn dependency) --------------------------------

def pca_fit_transform(X):
    """Center X, rotate onto principal axes. Returns (X_pca, eigvals desc)."""
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / (len(Xc) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)          # ascending
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0, None)
    return Xc @ eigvecs[:, order], eigvals


# --- The pruning cascade (mirrors hierarchical_pca_1nn_l2 in benchmark.py) --

def cascade_1nn(base_pca, query_pca, levels):
    """Returns (best_idx, survivors-per-stage)."""
    n = len(base_pca)
    candidates = np.arange(n)
    survivors = []

    q8 = query_pca[:levels[0]]
    d8 = np.sum((base_pca[:, :levels[0]] - q8) ** 2, axis=1)
    probe = np.argpartition(d8, min(PROBE_K, n - 1))[:min(PROBE_K, n - 1)]
    exact = np.sum((base_pca[probe] - query_pca) ** 2, axis=1)
    best_dist = exact.min()
    best_idx = probe[exact.argmin()]

    for i, k in enumerate(levels):
        d = d8 if i == 0 else np.sum(
            (base_pca[candidates, :k] - query_pca[:k]) ** 2, axis=1)
        candidates = candidates[d < best_dist] if i else np.nonzero(d < best_dist)[0]
        survivors.append(len(candidates))
        if not len(candidates):
            break

    if len(candidates):
        final = np.sum((base_pca[candidates] - query_pca) ** 2, axis=1)
        if final.min() < best_dist:
            best_idx = candidates[final.argmin()]
    return best_idx, survivors + [len(candidates)]


# --- Diagnostic -------------------------------------------------------------

def diagnose(X, levels, n_queries=100, name="data", plot=None):
    n, d = X.shape
    levels = [k for k in levels if k < d]
    print(f"\n=== {name}:  {n:,} vectors, {d} dims,  cascade {levels} ===")
    if n < 2000:
        print("WARNING: <2000 vectors; results will be noisy.")

    X_pca, eigvals = pca_fit_transform(X.astype(np.float64))
    cum = np.cumsum(eigvals) / max(eigvals.sum(), 1e-30)

    print("\n  Eigenvalue decay (cumulative explained variance = bound tightness):")
    for k in levels:
        bar = '#' * int(round(cum[k - 1] * 40))
        print(f"    first {k:>4} dims: {cum[k - 1] * 100:5.1f}%  |{bar}")
    r95 = int(np.searchsorted(cum, 0.95) + 1)
    print(f"    dims for 95% variance: {r95} / {d}")

    n_queries = min(n_queries, n // 10)
    rng = np.random.default_rng(0)
    q_idx = rng.choice(n, n_queries, replace=False)
    mask = np.ones(n, bool)
    mask[q_idx] = False
    base, queries = X_pca[mask], X_pca[q_idx]

    t_h = t_b = 0.0
    all_surv, mismatches = [], 0
    for q in queries:
        t0 = time.perf_counter()
        h_idx, surv = cascade_1nn(base, q, levels)
        t_h += time.perf_counter() - t0
        t0 = time.perf_counter()
        b_idx = np.argmin(np.sum((base - q) ** 2, axis=1))
        t_b += time.perf_counter() - t0
        mismatches += (h_idx != b_idx)
        all_surv.append(surv + [surv[-1]] * (len(levels) + 1 - len(surv)))

    surv = np.mean(all_surv, axis=0)
    nb = len(base)
    print("\n  Measured pruning cascade (avg over "
          f"{n_queries} queries, base {nb:,}):")
    for k, s in zip(levels, surv):
        print(f"    after {k:>4}D: {s:>10,.0f} candidates  ({s / nb * 100:6.2f}%)")
    fullfrac = (surv[-1] + PROBE_K) / nb
    speed = t_b / max(t_h, 1e-9)
    print(f"    full-D fetches (probe+finalists): {fullfrac * 100:.2f}% of base")
    print(f"    exactness: {n_queries - mismatches}/{n_queries} match brute force")
    print(f"    wall-clock speedup vs vectorized brute force: {speed:.1f}x "
          f"(in-RAM sample; out-of-core I/O gain tracks the fetch %)")

    frac = surv[-1] / nb
    if frac < 0.005 and speed > 2:
        verdict = "STRONG  -- highly correlated data; pruning will pay off."
    elif frac < 0.05:
        verdict = "MODERATE -- useful gains, especially out-of-core."
    else:
        verdict = ("POOR    -- dimensions too uncorrelated (flat eigenvalue "
                   "decay); bounds are loose, use brute force / ANN instead.")
    print(f"\n  VERDICT: {verdict}")

    if plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
            a1.plot(np.arange(1, d + 1), cum * 100)
            a1.set(title='Cumulative explained variance',
                   xlabel='PCA components', ylabel='%', ylim=(0, 101))
            for k in levels:
                a1.axvline(k, ls=':', c='gray', lw=0.8)
            a2.bar([str(k) for k in levels],
                   np.maximum(surv[:len(levels)], 0.5),
                   color='#4C72B0', edgecolor='black')
            a2.set_yscale('log')
            a2.set(title='Avg surviving candidates', xlabel='after stage (dims)')
            fig.suptitle(f'Prunability: {name}')
            fig.tight_layout()
            import os
            os.makedirs(os.path.dirname(plot) or '.', exist_ok=True)
            fig.savefig(plot, dpi=200)
            print(f"  Saved plot -> {plot}")
        except ImportError:
            print("  (matplotlib not available; skipped plot)")
    return frac, speed


# --- Demo: three correlation regimes ----------------------------------------

def demo(levels):
    rng = np.random.default_rng(42)
    n, d = 60_000, 128
    print("Synthetic demo: same n and d, three correlation levels.\n"
          "Watch the verdict flip as correlation (eigen-decay) weakens.")
    strong = rng.normal(size=(n, 8)) @ rng.normal(size=(8, d)) \
        + 0.1 * rng.normal(size=(n, d))
    medium = rng.normal(size=(n, 48)) @ rng.normal(size=(48, d)) \
        + 0.5 * rng.normal(size=(n, d))
    poor = rng.normal(size=(n, d))
    diagnose(strong, levels, name="high correlation (latent dim 8)")
    diagnose(medium, levels, name="medium correlation (latent dim 48)")
    diagnose(poor, levels, name="no correlation (i.i.d. gaussian)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('data', nargs='?', help='.npy or .fvecs file of vectors')
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--levels', default='8,16,32,64')
    ap.add_argument('--sample', type=int, default=100_000)
    ap.add_argument('--queries', type=int, default=100)
    ap.add_argument('--plot', default=None)
    args = ap.parse_args()
    levels = sorted({int(k) for k in args.levels.split(',')})

    if args.demo:
        demo(levels)
        return
    if not args.data:
        ap.print_help()
        sys.exit(1)
    X = load_vectors(args.data)
    if len(X) > args.sample:
        X = X[np.random.default_rng(0).choice(len(X), args.sample, replace=False)]
    diagnose(X, levels, n_queries=args.queries, name=args.data, plot=args.plot)


if __name__ == '__main__':
    main()
