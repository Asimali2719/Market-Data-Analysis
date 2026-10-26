import numpy as np, pandas as pd, pytest
from market_analysis import data, metrics as m, portfolio as p


def test_max_drawdown_known():
    assert m.max_drawdown(pd.Series([0.10, -0.50, 0.20])) == pytest.approx(-0.5)


def test_annualized_return_constant():
    assert m.annualized_return(pd.Series([0.001] * 252)) == pytest.approx(1.001 ** 252 - 1)


def test_beta_of_levered_series_is_two():
    b = pd.Series(np.random.default_rng(0).normal(0, 0.01, 500))
    beta, alpha = m.beta_alpha(pd.DataFrame({"x": 2 * b}), b)
    assert beta["x"] == pytest.approx(2.0) and alpha["x"] == pytest.approx(0.0, abs=1e-9)


def test_var_cvar_ordering():
    var, cvar = m.var_cvar(pd.Series(np.random.default_rng(1).normal(0, 0.01, 2000)))
    assert cvar > var > 0


def test_cleaning(tmp_path):
    f = tmp_path / "p.csv"
    f.write_text("Date,A,B\n2024-01-03,10,20\n2024-01-02,10,20\n2024-01-04,0,21\n2024-01-04,11,22\n2024-01-05,12,\n2024-01-08,13,23\n")
    df, rep = data.load_prices(str(f))
    assert rep["duplicate_dates_removed"] == 1 and rep["was_sorted"] is False
    assert df.index.is_monotonic_increasing and not df.isna().any().any() and (df > 0).all().all()


def test_min_variance_two_uncorrelated_assets():
    w = p.min_variance(np.array([0.1, 0.05]), np.diag([0.04, 0.01]))
    assert w.sum() == pytest.approx(1) and w[0] == pytest.approx(0.2, abs=1e-4)  # inverse-variance weights


def test_weight_cap_respected():
    a = np.random.default_rng(2).normal(size=(400, 4)) * 0.01
    w = p.max_sharpe(np.array([0.3, 0.05, 0.04, 0.03]), np.cov(a.T) * 252, 0.02, cap=0.4)
    assert w.max() <= 0.4 + 1e-6 and w.sum() == pytest.approx(1)
