"""The three-tier funnel: eigen curve rates the PCA levels, statistics
nominate a ball level, a measured trial decides. Timing-based numbers
(speedups, ADD/SKIP thresholds) are deliberately NOT asserted -- they are
hardware-dependent; the deterministic quantities (survivor fractions,
exactness, verdict plumbing) are."""

import numpy as np

from diagnose import ball_level_trial, diagnose, kurt_sqdep

from helpers import make_regime

LEVELS = [8, 16]


def test_poor_verdict_on_uncorrelated_data(capsys):
    X = make_regime("iid", n=3000, d=32)
    frac, _ = diagnose(X, LEVELS, name="iid", ball=False)
    out = capsys.readouterr().out
    assert frac >= 0.05                      # loose bounds, many survivors
    assert "POOR" in out
    assert "100/100 match brute force" in out


def test_prunable_verdict_on_correlated_data(capsys):
    X = make_regime("lowrank", n=3000, d=32)
    frac, _ = diagnose(X, LEVELS, name="lowrank", ball=False)
    out = capsys.readouterr().out
    assert frac < 0.05                       # tight bounds, few survivors
    assert "POOR" not in out
    assert "100/100 match brute force" in out


def test_ball_verdict_is_reported(capsys):
    # With the trial enabled a ball verdict (ADD or SKIP -- the direction
    # is timing-dependent) must always be delivered.
    X = make_regime("clustered", n=3000, d=32)
    diagnose(X, LEVELS, name="clustered", ball=True)
    out = capsys.readouterr().out
    assert "Ball level:" in out
    assert ("ADD" in out) or ("SKIP" in out)


def test_ball_trial_prunes_hard_on_clustered_data():
    X = make_regime("clustered", n=4000, d=32)
    res = ball_level_trial(X[:-100], X[-100:], LEVELS)
    assert res["ball_surv"] < 0.15           # tiny radii -> razor bound


def test_ball_trial_is_loose_on_iid_data():
    X = make_regime("iid", n=4000, d=32)
    res = ball_level_trial(X[:-100], X[-100:], LEVELS)
    assert res["ball_surv"] > 0.5            # no cluster structure -> loose


def test_kurt_sqdep_gaussian_vs_clustered():
    # Tier-2 statistics: ~0 on Gaussian data, elevated on multimodal data.
    n = 4000
    gauss = make_regime("iid", n=n, d=16).astype(np.float64)
    clus = make_regime("clustered", n=n, d=16).astype(np.float64)
    floor = (2.0 / n) ** 0.5
    kurt_g, sq_g = kurt_sqdep(gauss)
    kurt_c, sq_c = kurt_sqdep(clus)
    assert abs(kurt_g) < 0.5 and sq_g < 2.5 * floor
    assert sq_c > 2.5 * floor or abs(kurt_c) > 1.0


def test_pick_levels_scales_with_the_eigen_curve():
    from diagnose import pick_levels
    # steep decay (85% variance inside the first dims): minimal ladder
    steep = 1.0 - 0.5 ** np.arange(1, 129)
    assert pick_levels(steep) == [8]
    # perfectly flat curve, d=128: targets land at 39/64/90/109, rounded
    # up to multiples of 8 and capped at d//2
    flat = np.arange(1, 129) / 128.0
    assert pick_levels(flat) == [40, 64]
    # properties on an arbitrary curve: increasing, multiples of 8, capped
    rng = np.random.default_rng(0)
    ev = np.sort(rng.random(960))[::-1]
    lv = pick_levels(np.cumsum(ev) / ev.sum())
    assert all(k % 8 == 0 and 8 <= k <= 480 for k in lv)
    assert lv == sorted(set(lv))


def test_hybrid_index_auto_levels_stays_exact():
    from hybrid_search import HybridIndex
    from helpers import split_queries
    base, queries = split_queries(make_regime("lowrank"))
    idx = HybridIndex(levels="auto").fit(base)
    assert idx.levels == sorted(set(idx.levels))     # concrete, increasing
    assert all(k < base.shape[1] for k in idx.levels)
    for q in queries[:10]:
        assert idx.query(q)[:2] == idx.brute(q)
