"""Kati_tryout chain on the replication (research log trials 6–7): the paired block bootstrap
(jkmp.evaluation.bootstrap) and the executed-score selection rules (jkmp.portfolio.executed). The executed-book
engine itself is checked against stored trial 2a in ``scripts/18_kati_chain.py verify``."""

import numpy as np
import pandas as pd
import pytest

from jkmp.evaluation.bootstrap import (circular_block_indices, executed_score, net_sharpe, p_better_sharpe,
                                       score_standard_error)
from jkmp.portfolio.executed import active_series, choose, executed_scores, one_se_pick


def test_circular_blocks_are_contiguous_mod_n():
    idx = circular_block_indices(10, 50, 3, np.random.default_rng(0))
    assert idx.shape == (50, 10)
    steps = np.diff(idx, axis=1) % 10
    assert np.isin(steps[:, :2], [1]).all()           # within the first block, consecutive months (mod n)
    with pytest.raises(ValueError):
        circular_block_indices(10, 5, 11, np.random.default_rng(0))


def test_p_better_sharpe_ties_and_clear_winner():
    rng = np.random.default_rng(1)
    r = rng.normal(0.01, 0.04, 240)
    same = p_better_sharpe(r, np.zeros(240), r, np.zeros(240), 2000, 6, 7)
    assert same["p_better"] == pytest.approx(0.5)
    better = p_better_sharpe(r + 0.02, np.zeros(240), r, np.zeros(240), 2000, 6, 7)
    assert better["p_better"] > 0.99 and better["diff"] == pytest.approx(net_sharpe(r + 0.02, 0 * r) - net_sharpe(r, 0 * r))


def test_executed_score_and_standard_error():
    rng = np.random.default_rng(2)
    net, gross = rng.normal(0.002, 0.01, 120), rng.normal(0.0025, 0.01, 120)
    assert executed_score(net, gross, 31.19) == pytest.approx(12 * (net.mean() - 0.5 * 31.19 * gross.var(ddof=1)))
    se = score_standard_error(net, gross, 31.19, 2000, 6, 3)
    assert 0 < se < 0.05 and se == score_standard_error(net, gross, 31.19, 2000, 6, 3)   # seeded


def test_active_series_and_scores_use_only_months_to_the_december():
    eom_ret = pd.date_range("2000-01-31", periods=24, freq="ME")
    books = {"eom_ret": eom_ret, "tc": np.zeros((24, 2)),
             "r": np.column_stack([np.full(24, 0.01), np.r_[np.full(12, 0.01), np.full(12, 0.5)]])}
    bench = pd.DataFrame({"eom_ret": eom_ret, "r": 0.005, "tc": 0.0})
    net, gross = active_series(books, bench)
    sc = executed_scores(books, net, gross, [pd.Timestamp("2000-12-31")], 10.0)
    assert sc.shape == (1, 2) and sc[0, 0] == pytest.approx(sc[0, 1])   # the 2001 jump is not seen in Dec 2000


def test_one_se_pick_prefers_simpler_within_one_se():
    table = pd.DataFrame({"cand": range(4), "tag": "t", "g": [0.13, 0.05, 0.13, 0.13], "p": [512, 64, 64, 8],
                          "l_no": [0, 1, 2, 3], "l": [1.0, 10.0, 100.0, 5.0]})
    rng = np.random.default_rng(4)
    net = rng.normal(0.003, 0.01, (60, 4)); gross = net.copy()
    scores = np.array([0.020, 0.0195, 0.0194, -0.5])
    k, info = one_se_pick(table, scores, net, gross, 10.0, 500, 6, 5)
    assert info["best"] == 0 and info["n_eligible"] == 3
    assert k == 2                          # p = 64 beats 512; among p = 64 the larger λ (100) wins; p = 8 is not eligible
    ch = choose(table, [k], [pd.Timestamp("2000-12-31")], [scores[k]])
    assert ch.loc[0, "year"] == 2001 and ch.loc[0, "p"] == 64 and np.isinf(ch.loc[0, "h"])
