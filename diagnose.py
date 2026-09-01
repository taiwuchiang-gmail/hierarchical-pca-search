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
    --levels auto          cascade dimensions: picked from the eigen curve
                           (default) or explicit, e.g. --levels 8,16,32,64
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


LEVEL_TARGETS = (0.30, 0.50, 0.70, 0.85)


def pick_levels(cum, targets=LEVEL_TARGETS):
    """Cascade levels from the eigen curve: the dimension reaching each
    cumulative-variance target, rounded up to a multiple of 8 (min 8,
    capped at d//2, duplicates dropped). A fixed ladder like [8,16,32,64]
    is implicitly calibrated for d~128: on 1536-D OpenAI embeddings it
    stops at 4% of the dimensions and the funnel misreads the data as
    unprunable. Measured A/B vs hand-picked levels (100k samples):
    SIFT 8.2x vs 8.5x (tie), GIST 15.2x vs 9.3x, DBpedia-OpenAI 8.2x
    vs 7.6x -- the eigen curve nominates, the trial below still decides.
    """
    d = len(cum)
    cap = max(8, d // 2)
    levels = []
    for t in targets:
        k = int(np.searchsorted(cum, t) + 1)
        k = min(max(8, ((k + 7) // 8) * 8), cap)
        if not levels or k > levels[-1]:
            levels.append(k)
    return levels


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


# --- Beyond-covariance diagnostics (tier 2) ---------------------------------

def kurt_sqdep(Z):
    """Mean excess kurtosis of PCA components + mean |corr(z_i^2, z_j^2)|.

    kurt: marginal non-Gaussianity per component (Gaussian = 0).
    sqdep: joint dependence PCA cannot remove (noise floor ~ sqrt(2/n));
    values well above the floor mean structure exists beyond the covariance
    -- the kind a ball (k-means) level can potentially exploit.
    """
    d = Z - Z.mean(0)
    m2 = (d ** 2).mean(0)
    kurt = float(np.mean((d ** 4).mean(0) / (m2 ** 2 + 1e-12) - 3.0))
    Z2 = d ** 2
    Z2 = Z2 - Z2.mean(0)
    den = np.sqrt((Z2 ** 2).sum(0))
    den[den == 0] = 1
    C = (Z2.T @ Z2) / np.outer(den, den)
    sqdep = float(np.mean(np.abs(C[~np.eye(C.shape[1], dtype=bool)])))
    return kurt, sqdep


# --- Ball-level trial (tier 3): measure, don't predict ----------------------

def ball_level_trial(base_raw, queries_raw, levels):
    """Fit a HybridIndex on the sample and MEASURE what the ball (k-means)
    level adds over the PCA-only cascade. Returns dict of measured stats."""
    from hybrid_search import HybridIndex
    idx = HybridIndex(levels=levels).fit(base_raw)
    t_p = t_h = 0.0
    ball_surv = []
    for q in queries_raw:
        t0 = time.perf_counter()
        _, dp, _ = idx.query(q, use_ball=False)
        t_p += time.perf_counter() - t0
        t0 = time.perf_counter()
        _, dh, hs = idx.query(q, use_ball=True)
        t_h += time.perf_counter() - t0
        assert dp == dh, "exactness violated in ball trial!"
        ball_surv.append(hs[1] / hs[0])
    return dict(k=len(idx.C),
                ball_surv=float(np.mean(ball_surv)),
                ms_pca=t_p / len(queries_raw) * 1000,
                ms_hyb=t_h / len(queries_raw) * 1000)


# --- Diagnostic -------------------------------------------------------------

def diagnose(X, levels=None, n_queries=100, name="data", plot=None, ball=True):
    n, d = X.shape
    levels = [k for k in levels if k < d]
    print(f"\n=== {name}:  {n:,} vectors, {d} dims,  cascade {levels} ===")
    if n < 2000:
        print("WARNING: <2000 vectors; results will be noisy.")

    X_pca, eigvals = pca_fit_transform(X.astype(np.float64))
    cum = np.cumsum(eigvals) / max(eigvals.sum(), 1e-30)

    if levels is None:
        levels = pick_levels(cum)
        print(f"\n  Auto-selected cascade levels (eigen-curve targets "
              f"{[int(t * 100) for t in LEVEL_TARGETS]}% variance): {levels}")

    print("\n  Eigenvalue decay (cumulative explained variance = bound tightness):")
    for k in levels:
        bar = '#' * int(round(cum[k - 1] * 40))
        print(f"    first {k:>4} dims: {cum[k - 1] * 100:5.1f}%  |{bar}")
    r95 = int(np.searchsorted(cum, 0.95) + 1)
    print(f"    dims for 95% variance: {r95} / {d}")

    # Tier 2: structure BEYOND covariance (invisible to the curve above)
    kurt, sq = kurt_sqdep(X_pca)
    floor = (2.0 / n) ** 0.5
    beyond = sq > 2.5 * floor or abs(kurt) > 1.0
    print(f"\n  Beyond-covariance structure (invisible to the eigen curve):")
    print(f"    kurt(PCA) = {kurt:+.2f} (Gaussian = 0)   "
          f"sqdep = {sq:.4f} (noise floor ~ {floor:.4f})")
    print("    -> " + ("non-Gaussian structure detected -- a k-means/ball "
                       "level may exploit it" if beyond else
                       "none detected -- data is ~Gaussian given its "
                       "covariance"))

    n_queries = min(n_queries, n // 10)
    rng = np.random.default_rng(0)
    q_idx = rng.choice(n, n_queries, replace=False)
    mask = np.ones(n, bool)
    mask[q_idx] = False
    base, queries = X_pca[mask], X_pca[q_idx]
    base_raw, queries_raw = X[mask], X[q_idx]        # for the ball trial

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

    # Tier 3: measure what a ball (k-means) level actually adds. Statistics
    # nominate, measurement decides -- the trial runs the real hybrid cascade.
    ball_verdict = None
    if ball:
        try:
            bt = ball_level_trial(base_raw.astype(np.float32),
                                  queries_raw.astype(np.float32), levels)
            gain = bt['ms_pca'] / max(bt['ms_hyb'], 1e-9)
            print(f"\n  Ball-level trial (k-means k={bt['k']}, exact "
                  f"triangle-inequality bounds):")
            print(f"    ball stage prunes to {bt['ball_surv'] * 100:.2f}% "
                  f"before any PCA stage")
            print(f"    PCA-only {bt['ms_pca']:.1f} ms/query  vs  "
                  f"hybrid {bt['ms_hyb']:.1f} ms/query  "
                  f"({gain:.1f}x from the ball level)")
            if gain > 1.25:
                ball_verdict = (f"ADD -- measured {gain:.1f}x on top of the "
                                f"PCA cascade" +
                                (" (non-Gaussian structure harvested)"
                                 if beyond else
                                 " (cheap prefilter; bounds only moderately "
                                 "tight)"))
            else:
                ball_verdict = ("SKIP -- no measured gain; ball bounds too "
                                "loose on this data")
        except ImportError:
            print("\n  (hybrid_search.py not found -- ball-level trial "
                  "skipped)")

    # Verdict from the MEASURED work ratio, not the survivor fraction: the
    # same finalist % that is marginal at d=128 is a big win at d=960,
    # because the cascade's cost is early-stage terms + surv*d full fetches
    # while brute always pays n*d (GIST: 6.9% finalists = 9x fewer terms).
    frac = surv[-1] / nb
    terms = nb * levels[0] + PROBE_K * d + surv[-1] * d
    for s, lv in zip(surv[:-1], levels[1:]):
        terms += s * lv
    work = nb * d / terms
    print(f"\n  Work ratio (terms, hardware-independent): brute {nb * d / 1e6:.1f}M"
          f" / cascade {terms / 1e6:.1f}M = {work:.1f}x reduction")
    if work > 8 and speed > 2:
        verdict = f"STRONG  -- tight bounds; {work:.0f}x less work, pruning will pay off."
    elif work > 2:
        verdict = "MODERATE -- useful gains, especially out-of-core."
    else:
        verdict = ("POOR    -- bounds too loose (uncorrelated dimensions); "
                   "use brute force / ANN instead.")
    print(f"  VERDICT: PCA levels: {verdict}")
    if ball_verdict:
        print(f"           Ball level: {ball_verdict}")

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
    centers = rng.normal(size=(512, d))
    clustered = centers[rng.integers(0, 512, n)] \
        + 0.05 * rng.normal(size=(n, d))
    diagnose(strong, levels, name="high correlation (latent dim 8)")
    diagnose(medium, levels, name="medium correlation (latent dim 48)")
    diagnose(poor, levels, name="no correlation (i.i.d. gaussian)")
    diagnose(clustered, levels,
             name="512 tight clusters (flat spectrum, non-Gaussian)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('data', nargs='?', help='.npy or .fvecs file of vectors')
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--levels', default='auto',
                    help="'auto' (from the eigen curve) or e.g. 8,16,32,64")
    ap.add_argument('--sample', type=int, default=100_000)
    ap.add_argument('--queries', type=int, default=100)
    ap.add_argument('--plot', default=None)
    ap.add_argument('--no-ball', action='store_true',
                    help='skip the k-means ball-level trial')
    args = ap.parse_args()
    levels = None if args.levels == 'auto' \
        else sorted({int(k) for k in args.levels.split(',')})

    if args.demo:
        demo(levels)
        return
    if not args.data:
        ap.print_help()
        sys.exit(1)
    X = load_vectors(args.data)
    if len(X) > args.sample:
        X = X[np.random.default_rng(0).choice(len(X), args.sample, replace=False)]
    diagnose(X, levels, n_queries=args.queries, name=args.data,
             plot=args.plot, ball=not args.no_ball)


if __name__ == '__main__':
    main()
