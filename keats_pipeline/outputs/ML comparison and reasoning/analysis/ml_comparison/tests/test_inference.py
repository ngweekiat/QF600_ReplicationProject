"""Independent formula and edge-case checks for saved-output inference."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from analysis.ml_comparison.inference import (
    bootstrap_rank_stability, circular_block_indices, compare_pairs, holm_adjust, net_sharpe,
    newey_west_mean, sharpe_influence, utility_flows,
)


def _sample():
    dates = pd.date_range("2020-01-31", periods=36, freq="ME")
    r = np.random.default_rng(7).normal(.01, .025, len(dates))
    return pd.concat([
        pd.DataFrame({"dataset": "us", "strategy": "a", "eom_ret": dates, "r": r + .002, "tc": .001}),
        pd.DataFrame({"dataset": "us", "strategy": "b", "eom_ret": dates, "r": r, "tc": .002}),
    ], ignore_index=True)


def test_paper_probability_matches_r_formula_and_sample_variance_utility():
    pf = _sample()
    pf.loc[pf.strategy.eq("a"), "r"] *= 1.3
    a, b = [pf[pf.strategy.eq(s)] for s in ["a", "b"]]
    diff = ((a.r.to_numpy() - a.tc.to_numpy() - 5 * (a.r.to_numpy() - a.r.mean())**2)
            - (b.r.to_numpy() - b.tc.to_numpy() - 5 * (b.r.to_numpy() - b.r.mean())**2))
    expected_p = norm.cdf(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))
    expected_u = 12 * ((a.r - a.tc).mean() - 5*a.r.var(ddof=1)
                       - ((b.r - b.tc).mean() - 5*b.r.var(ddof=1)))
    result = compare_pairs(pf, bootstrap_reps=120).iloc[0]
    assert result.posterior_probability_a_gt_b == pytest.approx(expected_p)
    assert result.utility_diff_annual == pytest.approx(expected_u)
    assert result.sharpe_diff == pytest.approx(net_sharpe(a.r, a.tc) - net_sharpe(b.r, b.tc))


def test_newey_west_matches_explicit_bartlett_sum():
    x = np.array([.02, -.04, .03, .01, -.01, .015])
    centered = x - x.mean()
    lrv = np.dot(centered, centered) / len(x)
    for k in range(1, 3):
        lrv += 2 * (1 - k/3) * np.dot(centered[k:], centered[:-k]) / len(x)
    actual = newey_west_mean(x, lags=2)
    assert actual["se"] == pytest.approx(np.sqrt(lrv / len(x)))
    assert actual["p"] == pytest.approx(2*norm.sf(abs(x.mean()/actual["se"])))


def test_holm_adjustment_and_nan_family():
    np.testing.assert_allclose(holm_adjust(np.array([.01, .04, .03, np.nan])), [.03, .06, .06, np.nan])


def test_circular_blocks_and_paired_bootstrap_preserve_deterministic_spread():
    indices = circular_block_indices(17, 20, 4, np.random.default_rng(1))
    assert indices.shape == (20, 17)
    np.testing.assert_array_equal((indices[:, 1:4] - indices[:, :3]) % 17, np.ones((20, 3)))
    result = compare_pairs(_sample(), bootstrap_reps=200).iloc[0]
    assert result.utility_diff_annual == pytest.approx(.036)
    assert result.utility_boot_ci_low == pytest.approx(.036)
    assert result.utility_boot_ci_high == pytest.approx(.036)
    assert result.posterior_probability_a_gt_b == 1
    assert result.utility_hac_p == 0


def test_identical_portfolios_have_zero_difference_and_no_evidence():
    pf = _sample()
    pf.loc[pf.strategy.eq("a"), ["r", "tc"]] = pf.loc[pf.strategy.eq("b"), ["r", "tc"]].to_numpy()
    result = compare_pairs(pf, bootstrap_reps=80).iloc[0]
    assert result.posterior_probability_a_gt_b == .5
    assert result.utility_hac_p == 1
    assert result.sharpe_hac_p == 1
    assert result.utility_diff_annual == 0
    assert result.sharpe_diff == 0


def test_sharpe_influence_matches_numerical_delta_method():
    rng = np.random.default_rng(8)
    r, tc = rng.normal(.015, .03, 100), rng.uniform(0, .002, 100)
    net, variance = (r-tc).mean(), r.var(ddof=1)
    scores = sharpe_influence(r, tc)
    i, eps = 8, 1e-7
    vmoment = len(r)/(len(r)-1) * ((r[i]-r.mean())**2 - ((r-r.mean())**2).mean())
    numerical = (np.sqrt(12)*(net+eps*((r[i]-tc[i])-net))/np.sqrt(variance+eps*vmoment)
                 - np.sqrt(12)*net/np.sqrt(variance))/eps
    assert scores[i] == pytest.approx(numerical, rel=1e-5)
    assert scores.mean() == pytest.approx(0, abs=1e-14)


def test_explicit_missing_pair_counts_and_duplicate_or_calendar_gap_failures():
    pf = _sample()
    result = compare_pairs(pf.drop(index=[0]), bootstrap_reps=20).iloc[0]
    assert result.n_a_excluded == 0
    assert result.n_b_excluded == 1
    with pytest.raises(ValueError, match="duplicate"):
        compare_pairs(pd.concat([pf, pf.iloc[[0]]]), bootstrap_reps=20)
    with pytest.raises(ValueError, match="nonconsecutive"):
        compare_pairs(pf[pf.eom_ret.ne(pd.Timestamp("2021-05-31"))], bootstrap_reps=20)


def test_datasets_do_not_mix_and_results_repeat_with_seed():
    pf = _sample()
    combined = pd.concat([pf, pf.assign(dataset="eu")])
    one = compare_pairs(combined, bootstrap_reps=100)
    two = compare_pairs(combined, bootstrap_reps=100)
    pd.testing.assert_frame_equal(one, two)
    assert len(one) == 2
    assert set(one.dataset) == {"us", "eu"}


def test_constant_gross_return_skips_undefined_sharpe_and_keeps_utility():
    pf = _sample().assign(r=.01)
    result = compare_pairs(pf, bootstrap_reps=20).iloc[0]
    assert result.sharpe_status.startswith("SKIPPED")
    assert np.isnan(result.sharpe_diff)
    assert result.utility_diff_annual == pytest.approx(.012)


def test_corrected_flow_equals_table2_utility():
    r = np.array([.1, -.05, .02, .03])
    tc = np.array([.001, .002, .003, .004])
    assert utility_flows(r, tc, 10, True).mean() == pytest.approx((r-tc).mean() - 5*r.var(ddof=1))


def test_swapping_names_reverses_effect_ci_and_probability_without_changing_p():
    pf=_sample()
    pf.loc[pf.strategy.eq('a'),'r']*=1.4
    one=compare_pairs(pf,bootstrap_reps=100,seed=123).iloc[0]
    swapped=pf.assign(strategy=pf.strategy.map({'a':'b','b':'a'}))
    two=compare_pairs(swapped,bootstrap_reps=100,seed=123).iloc[0]
    assert one.posterior_probability_a_gt_b==pytest.approx(1-two.posterior_probability_a_gt_b)
    for name in ['utility','sharpe']:
        assert one[f'{name}_hac_p']==pytest.approx(two[f'{name}_hac_p'])
        assert one[f'{name}_boot_ci_low']==pytest.approx(-two[f'{name}_boot_ci_high'])
        assert one[f'{name}_boot_ci_high']==pytest.approx(-two[f'{name}_boot_ci_low'])


def test_missing_identifier_is_rejected_instead_of_silently_dropping_dataset():
    pf=_sample()
    pf.loc[0,'dataset']=None
    with pytest.raises(ValueError,match='missing dataset'):
        compare_pairs(pf,bootstrap_reps=20)


def test_joint_bootstrap_ordering_respects_deterministic_cost_dominance():
    # Identical gross paths; a has exactly 10 bp lower costs each month, so
    # paired resampling must always preserve both utility and Sharpe ordering.
    pf=_sample()
    pf.loc[pf.strategy.eq('a'),'r']=pf.loc[pf.strategy.eq('b'),'r'].to_numpy()
    ranks=bootstrap_rank_stability(pf,bootstrap_reps=100,block_length=6)
    a=ranks[ranks.strategy.eq('a')]
    b=ranks[ranks.strategy.eq('b')]
    assert a.observed_rank.eq(1).all()
    assert a.rank_ci_low.eq(1).all() and a.rank_ci_high.eq(1).all()
    assert a.bootstrap_probability_best_ties_split.eq(1).all()
    assert b.rank_ci_low.eq(2).all() and b.rank_ci_high.eq(2).all()
    assert b.bootstrap_probability_best_ties_split.eq(0).all()


def test_identical_strategy_bootstrap_rank_ties_split_best_probability():
    pf=_sample()
    pf.loc[pf.strategy.eq('a'),['r','tc']]=pf.loc[pf.strategy.eq('b'),['r','tc']].to_numpy()
    ranks=bootstrap_rank_stability(pf,bootstrap_reps=60)
    assert ranks.observed_rank.eq(1).all()
    assert ranks.rank_ci_high.eq(1).all()
    assert ranks.bootstrap_probability_best_ties_split.eq(.5).all()
    for _,g in ranks.groupby(['dataset','ranking_metric']):
        assert g.bootstrap_probability_best_ties_split.sum()==pytest.approx(1)
