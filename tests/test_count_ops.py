"""count_ops turns survivor statistics into exact, hardware-independent
work counts (Paper II, Table 3). The values below are hand-computed from
the accounting rules in its docstring."""

from hybrid_search import HybridIndex, count_ops

from helpers import make_regime, split_queries


def test_pca_only_hand_computed():
    # N=1000, levels=[8,16], d=32, probe=50, survivors 100 -> 10.
    # stage 1 (doubles as probe): 1000*8 + 50*32 = 9600
    # stage 2: 100*16 = 1600; final full-D check: 10*32 = 320
    terms, bound_ops = count_ops([1000, 100, 10], [8, 16], 32,
                                 use_ball=False)
    assert terms == 9600 + 1600 + 320 == 11520
    assert bound_ops == 0


def test_hybrid_hand_computed_with_reprobe():
    # N=1000, k=20 clusters, levels=[8,16], d=32, ball survivors 200.
    # centroids+probe: 20*32 + 50*32 = 2240; stage 1: 200*8 = 1600;
    # re-probe (200 > 50): 50*32 = 1600; stage 2: 100*16 = 1600;
    # final: 10*32 = 320. Ball bounds: one per point.
    terms, bound_ops = count_ops([1000, 200, 100, 10], [8, 16], 32,
                                 n_clusters=20, use_ball=True)
    assert terms == 2240 + 1600 + 1600 + 1600 + 320 == 7360
    assert bound_ops == 1000


def test_hybrid_no_reprobe_when_ball_survivors_fit_in_probe():
    # Ball survivors (40) <= probe (50): the re-probe full distances are
    # NOT charged.
    terms, bound_ops = count_ops([1000, 40, 20, 10], [8, 16], 32,
                                 n_clusters=20, use_ball=True)
    assert terms == 2240 + 40 * 8 + 20 * 16 + 10 * 32 == 3200
    assert bound_ops == 1000


def test_short_stats_padded_with_last_value():
    # A cascade that empties early reports fewer stages; count_ops pads
    # with the final value. Empty-from-stage-1 costs just scan + probe.
    terms, _ = count_ops([1000, 0], [8, 16], 32, use_ball=False)
    assert terms == 1000 * 8 + 50 * 32 == 9600


def test_counts_from_real_query_never_exceed_flat_scan_on_prunable_data():
    # End-to-end sanity: on strongly correlated data the counted work of a
    # real query must be far below the N*d flat-scan definition.
    base, queries = split_queries(make_regime("lowrank", n=4000, d=32))
    idx = HybridIndex().fit(base)
    flat = len(base) * 32
    for q in queries:
        _, _, stats = idx.query(q, use_ball=False)
        terms, _ = count_ops(stats, idx.levels, 32, use_ball=False)
        assert terms < 0.5 * flat
