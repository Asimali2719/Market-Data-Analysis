"""Mean-variance portfolio construction (long-only, optional weight cap)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def stats(w, mu, cov, rf=0.0):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    return ret, vol, (ret - rf) / vol


def _solve(objective, n, cap, extra=()):
    """SLSQP with progressively looser tolerance (tight tolerances can stall on flat objectives)."""
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}, *extra]
    for ftol in (1e-12, 1e-10, 1e-8):
        res = minimize(objective, np.full(n, 1 / n), method="SLSQP", bounds=[(0, cap)] * n, constraints=cons,
                       options={"maxiter": 500, "ftol": ftol})
        if res.success:
            return res.x
    raise RuntimeError(f"optimiser failed: {res.message}")


def min_variance(mu, cov, cap=1.0):
    return _solve(lambda w: w @ cov @ w, len(mu), cap)


def max_sharpe(mu, cov, rf=0.0, cap=1.0):
    return _solve(lambda w: -stats(w, mu, cov, rf)[2], len(mu), cap)


def efficient_frontier(mu, cov, cap=1.0, points=30):
    rows = []
    for target in np.linspace(float(mu.min()), float(mu.max()), points):
        try:
            w = _solve(lambda w: w @ cov @ w, len(mu), cap, [{"type": "eq", "fun": lambda w, t=target: w @ mu - t}])
        except RuntimeError:
            continue
        rows.append((target, float(np.sqrt(w @ cov @ w))))
    return pd.DataFrame(rows, columns=["ret", "vol"])


def random_portfolios(mu, cov, rf=0.0, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(len(mu)), n)
    ret = w @ mu
    vol = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))
    return pd.DataFrame({"ret": ret, "vol": vol, "sharpe": (ret - rf) / vol})
