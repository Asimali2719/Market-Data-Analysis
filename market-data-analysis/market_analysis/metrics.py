"""Return and risk metrics. All functions take simple (arithmetic) daily returns."""
from __future__ import annotations
import numpy as np
import pandas as pd

PPY = 252


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def annualized_return(r, ppy: int = PPY):
    """Geometric (compound) annual growth rate."""
    return (1 + r).prod() ** (ppy / len(r)) - 1


def annualized_vol(r, ppy: int = PPY):
    return r.std(ddof=1) * np.sqrt(ppy)


def sharpe(r, rf: float = 0.0, ppy: int = PPY):
    return (r.mean() * ppy - rf) / annualized_vol(r, ppy)


def sortino(r, rf: float = 0.0, ppy: int = PPY):
    downside = np.sqrt((np.minimum(r - rf / ppy, 0) ** 2).mean() * ppy)
    return (r.mean() * ppy - rf) / downside


def drawdown(r):
    wealth = (1 + r).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(r):
    return drawdown(r).min()


def var_cvar(r, level: float = 0.95):
    """Historical one-day VaR and CVaR, reported as positive loss fractions."""
    q = r.quantile(1 - level)
    if isinstance(r, pd.Series):
        return -q, -r[r <= q].mean()
    return -q, -pd.Series({c: r[c][r[c] <= q[c]].mean() for c in r.columns})


def beta_alpha(r: pd.DataFrame, bench: pd.Series, rf: float = 0.0, ppy: int = PPY):
    """CAPM beta and annualised Jensen's alpha versus a benchmark."""
    var_b = bench.var(ddof=1)
    beta = r.apply(lambda c: c.cov(bench) / var_b)
    alpha = (r.mean() - rf / ppy - beta * (bench.mean() - rf / ppy)) * ppy
    return beta, alpha


def summary_table(r: pd.DataFrame, rf: float = 0.0, bench: str | None = None) -> pd.DataFrame:
    var95, cvar95 = var_cvar(r)
    t = pd.DataFrame({
        "ann_return": annualized_return(r), "ann_vol": annualized_vol(r), "sharpe": sharpe(r, rf),
        "sortino": sortino(r, rf), "max_drawdown": max_drawdown(r), "var_95_1d": var95, "cvar_95_1d": cvar95,
    })
    if bench:
        t["beta"], t["alpha"] = beta_alpha(r, r[bench], rf)
    return t
