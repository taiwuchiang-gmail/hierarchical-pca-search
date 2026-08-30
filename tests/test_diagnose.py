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
