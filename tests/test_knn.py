"""k-NN exactness: the k-th-best-evaluated radius generalizes the 1-NN
dynamic radius, and the result must equal brute-force k-NN -- same
invariant, k times over."""

import numpy as np
import pytest

from hybrid_search import HybridIndex

from helpers import make_regime, split_queries

REGIMES = ["clustered", "lowrank", "iid"]


def assert_knn_equal(idx, q, k, use_ball):
    b_idx, b_d = idx.brute(q, k=k)
    h_idx, h_d, stats = idx.query(q, use_ball=use_ball, k=k)
    # distances must match exactly; indices as sets (exact-tie order is
    # arbitrary, and float ties have measure zero on continuous data)
    assert np.array_equal(h_d, b_d), f"k-NN distances differ (k={k})"
    assert set(h_idx) == set(b_idx)
    assert len(h_idx) == k
    assert np.all(np.diff(h_d) >= 0)          # sorted by distance
    return stats


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("use_ball", [False, True])
@pytest.mark.parametrize("k", [5, 10, 50])
def test_knn_matches_brute_force(regime, use_ball, k):
    base, queries = split_queries(make_regime(regime))
    idx = HybridIndex().fit(base)
    for q in queries:
        assert_knn_equal(idx, q, k, use_ball)


@pytest.mark.parametrize("use_ball", [False, True])
def test_k_larger_than_probe_size(use_ball):
    # k=80 > PROBE_K=50: the probe must expand so the initial radius is
    # a valid k-th order statistic.
    base, queries = split_queries(make_regime("lowrank"))
    idx = HybridIndex().fit(base)
    for q in queries[:5]:
        assert_knn_equal(idx, q, 80, use_ball)


@pytest.mark.parametrize("use_ball", [False, True])
def test_knn_on_small_dataset(use_ball):
    # k close to n on a dataset barely larger than the probe.
    base, queries = split_queries(make_regime("iid", n=200))
    idx = HybridIndex().fit(base)
    for q in queries[:5]:
        for k in (10, len(base) - 1):
            assert_knn_equal(idx, q, k, use_ball)


def test_k1_keeps_scalar_return_type():
    base, queries = split_queries(make_regime("lowrank"))
    idx = HybridIndex().fit(base)
    h_idx, h_d, _ = idx.query(queries[0], k=1)
    assert isinstance(h_idx, int) and isinstance(h_d, float)
    b_idx, b_d = idx.brute(queries[0], k=1)
    assert isinstance(b_idx, int) and isinstance(b_d, float)
    assert (h_idx, h_d) == (b_idx, b_d)


def test_invalid_k_raises():
    base, queries = split_queries(make_regime("iid", n=200))
    idx = HybridIndex().fit(base)
    for bad in (0, len(base), len(base) + 5):
        with pytest.raises(ValueError, match="k must be"):
            idx.query(queries[0], k=bad)
