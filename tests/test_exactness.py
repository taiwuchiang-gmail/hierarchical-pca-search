"""Exactness is the product's core invariant: every query path must return
the brute-force nearest neighbor, on every data regime, always. Bound
quality is allowed to affect speed only, never the result."""

import numpy as np
import pytest

from hybrid_search import HybridIndex
from diagnose import cascade_1nn, pca_fit_transform

from helpers import make_regime, split_queries

REGIMES = ["clustered", "lowrank", "iid"]


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("use_ball", [False, True])
def test_query_matches_brute_force(regime, use_ball):
    base, queries = split_queries(make_regime(regime))
    idx = HybridIndex().fit(base)
    for q in queries:
        b_idx, b_dist = idx.brute(q)
        h_idx, h_dist, stats = idx.query(q, use_ball=use_ball)
        assert h_dist == b_dist, (
            f"exactness violated on {regime} (use_ball={use_ball})")
        assert h_idx == b_idx
        assert stats[0] == len(base)


@pytest.mark.parametrize("use_ball", [False, True])
def test_levels_are_clipped_to_dimension(use_ball):
    # d=16 < default levels (8, 16, 32, 64): fit must clip to [8] and the
    # query must still be exact through the shortened cascade.
    base, queries = split_queries(make_regime("lowrank", d=16))
    idx = HybridIndex().fit(base)
    assert idx.levels == [8]
    for q in queries:
        _, b_dist = idx.brute(q)
        _, h_dist, _ = idx.query(q, use_ball=use_ball)
        assert h_dist == b_dist


@pytest.mark.parametrize("use_ball", [False, True])
def test_small_dataset(use_ball):
    # Barely more points than the probe size: exercises the probe and
    # k-means clamp (k=64 clusters on 180 base points) edge cases.
    base, queries = split_queries(make_regime("iid", n=200))
    idx = HybridIndex().fit(base)
    for q in queries:
        _, b_dist = idx.brute(q)
        _, h_dist, _ = idx.query(q, use_ball=use_ball)
        assert h_dist == b_dist


def test_explicit_n_clusters():
    base, queries = split_queries(make_regime("clustered"))
    idx = HybridIndex(n_clusters=32).fit(base)
    assert len(idx.C) == 32
    for q in queries:
        _, b_dist = idx.brute(q)
        _, h_dist, _ = idx.query(q, use_ball=True)
        assert h_dist == b_dist


def test_query_is_a_base_point():
    # Querying a base point gives a near-zero radius -- the tightest
    # possible pruning must still not dismiss the true neighbor.
    base, _ = split_queries(make_regime("lowrank"))
    idx = HybridIndex().fit(base)
    for i in (0, 137, len(base) - 1):
        b_idx, b_dist = idx.brute(base[i])
        for use_ball in (False, True):
            h_idx, h_dist, _ = idx.query(base[i], use_ball=use_ball)
            assert h_idx == b_idx == i
            assert h_dist == b_dist


@pytest.mark.parametrize("regime", REGIMES)
def test_cascade_1nn_matches_brute_force(regime):
    # diagnose.py runs its own PCA-only cascade; it must be exact too.
    X = make_regime(regime).astype(np.float64)
    X_pca, _ = pca_fit_transform(X)
    base, queries = split_queries(X_pca)
    for q in queries:
        b_idx = int(np.argmin(np.sum((base - q) ** 2, axis=1)))
        h_idx, survivors = cascade_1nn(base, q, [8, 16])
        assert int(h_idx) == b_idx
        assert survivors[0] <= len(base)
