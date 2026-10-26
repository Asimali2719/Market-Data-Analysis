"""Out-of-sample test: optimise on a training window, evaluate on the unseen remainder."""
from __future__ import annotations
import pandas as pd
from . import metrics as m, portfolio as p


def walk_forward(r: pd.DataFrame, assets, rf=0.03, train_frac=0.75, cap=0.4):
    split = int(len(r) * train_frac)
    train, test = r.iloc[:split][assets], r.iloc[split:][assets]
    mu, cov = train.mean() * m.PPY, train.cov() * m.PPY
    w_opt = pd.Series(p.max_sharpe(mu.values, cov.values, rf, cap), index=assets)
    w_eq = pd.Series(1 / len(assets), index=assets)
    series = {"Max-Sharpe (trained)": test @ w_opt, "Equal weight": test @ w_eq}
    rows = {k: {"ann_return": m.annualized_return(s), "ann_vol": m.annualized_vol(s), "sharpe": m.sharpe(s, rf),
                "max_drawdown": m.max_drawdown(s)} for k, s in series.items()}
    return pd.DataFrame(rows).T, w_opt, pd.DataFrame(series), r.index[split]
