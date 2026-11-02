"""Economic reporting checks distinguish additive paths from investor wealth."""
import numpy as np
import pandas as pd
import pytest

from analysis.ml_comparison.metrics import additive_path, paths_and_rolling, statistics


def monthly(net):
    return pd.DataFrame(dict(dataset='example', strategy='Portfolio ML',
        eom_ret=pd.date_range('2020-01-31', periods=len(net), freq='ME'),
        r=np.array(net)+.001, tc=.001, turnover=.1, inv=1., shorting=0.))


def test_additive_losses_include_initial_zero_and_recovery():
    # Two 10% losses accumulate to -20 pp, not the compounded -19%.
    path, dd = additive_path([-.1, -.1, .3, -.15])
    np.testing.assert_allclose(path, [-.1, -.2, .1, -.05], atol=1e-15)
    np.testing.assert_allclose(dd, [-.1, -.2, 0., -.15], atol=1e-15)
    stats = statistics(monthly([-.1, -.1, .3, -.15]))
    assert stats['max_drawdown_excess'] == pytest.approx(-.2)
    assert stats['cumulative_net_excess'] == pytest.approx(-.05)
    assert stats['path_convention'] == 'additive'
    assert 'cagr_excess' not in stats and 'calmar_excess' not in stats


def test_statistical_and_monthly_reports_agree_without_discarding_large_losses():
    frame = monthly([.2, -1.2, .4, -.1]).sample(frac=1, random_state=8)
    stats = statistics(frame)
    paths = paths_and_rolling(frame)
    np.testing.assert_allclose(paths.cumulative_net_excess, [.2, -1., -.6, -.7])
    np.testing.assert_allclose(paths.drawdown_excess, [0., -1.2, -.8, -.9])
    assert stats['max_drawdown_excess'] == pytest.approx(paths.drawdown_excess.min())
    assert stats['cumulative_net_excess'] == pytest.approx(paths.cumulative_net_excess.iloc[-1])
    assert stats['R_TC'] == pytest.approx(12*frame.eval('r-tc').mean())
    assert len(paths) == len(frame) and paths.drawdown_excess.notna().all()


def test_additive_active_path_is_sum_of_paired_monthly_differences():
    candidate = np.array([.02, -.03, .04, .01])
    market = np.array([.01, -.02, .03, .005])
    a, _ = additive_path(candidate)
    b, _ = additive_path(market)
    np.testing.assert_allclose(a-b, np.cumsum(candidate-market), atol=1e-15)
    with pytest.raises(ValueError, match='finite'):
        additive_path([0., np.nan])
