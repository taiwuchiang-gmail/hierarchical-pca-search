import numpy as np


def make_regime(name, n=4000, d=32, seed=0):
    """Synthetic datasets covering the three regimes the bounds care about.

    clustered: many tight clusters, flat eigen-spectrum -- PCA bound loose,
               ball bound razor-sharp.
    lowrank:   low-rank correlated Gaussian -- PCA bound tight.
    iid:       isotropic Gaussian -- both bounds loose (worst case); pruning
               must still be EXACT, just not fast.
    neardup:   groups of 80 float32 near-duplicates (one coordinate of each
               copy nudged by one ulp) -- with few, broad clusters, |q-c| and
               d_p nearly cancel and only the rounding margin keeps the ball
               bound below the true distance.
    """
    rng = np.random.default_rng(seed)
    if name == "clustered":
        centers = rng.normal(size=(64, d))
        X = centers[rng.integers(0, 64, n)] + 0.05 * rng.normal(size=(n, d))
    elif name == "lowrank":
        X = (rng.normal(size=(n, 4)) @ rng.normal(size=(4, d))
             + 0.1 * rng.normal(size=(n, d)))
    elif name == "iid":
        X = rng.normal(size=(n, d))
    elif name == "neardup":
        per = 80
        centers = (3 * rng.normal(size=(n // per, d))).astype(np.float32)
        X = np.repeat(centers, per, axis=0)
        rows = np.arange(len(X))
        j = rng.integers(0, d, size=len(X))
        X[rows, j] = np.nextafter(X[rows, j], np.float32(np.inf))
        X[::per] = centers                      # one exact copy per group
    else:
        raise ValueError(name)
    return X.astype(np.float32)


def split_queries(X, n_queries=20):
    """Last n_queries rows are the query set, the rest is the base."""
    return X[:-n_queries], X[-n_queries:]
