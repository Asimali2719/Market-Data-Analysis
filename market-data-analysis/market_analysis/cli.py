"""Command line entry point: python -m market_analysis --prices data/prices_sample.csv"""
from __future__ import annotations
import argparse, json, os
import pandas as pd
from . import data, metrics as m, portfolio as p, backtest as bt, report as rp

FMT = {"ann_return": "{:.1%}", "ann_vol": "{:.1%}", "max_drawdown": "{:.1%}", "var_95_1d": "{:.2%}", "cvar_95_1d": "{:.2%}",
       "alpha": "{:.1%}", "beta": "{:.2f}", "sharpe": "{:.2f}", "sortino": "{:.2f}", "max_sharpe": "{:.1%}", "min_variance": "{:.1%}"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Market data analysis")
    ap.add_argument("--prices", default="data/prices_sample.csv")
    ap.add_argument("--benchmark", default="MKT")
    ap.add_argument("--rf", type=float, default=0.03, help="annual risk-free rate")
    ap.add_argument("--cap", type=float, default=0.40, help="max weight per asset")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--make-sample", action="store_true", help="(re)generate the synthetic sample file first")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.make_sample:
        os.makedirs(os.path.dirname(a.prices) or ".", exist_ok=True)
        data.generate_sample(a.prices)

    prices, rep = data.load_prices(a.prices)
    r = m.daily_returns(prices)
    assets = [c for c in r.columns if c != a.benchmark]

    table = m.summary_table(r, a.rf, a.benchmark)
    corr = r.corr()
    mu, cov = r[assets].mean() * m.PPY, r[assets].cov() * m.PPY
    w_ms = pd.Series(p.max_sharpe(mu.values, cov.values, a.rf, a.cap), index=assets)
    w_mv = pd.Series(p.min_variance(mu.values, cov.values, a.cap), index=assets)
    weights = pd.DataFrame({"max_sharpe": w_ms, "min_variance": w_mv})
    front = p.efficient_frontier(mu.values, cov.values, a.cap)
    cloud = p.random_portfolios(mu.values, cov.values, a.rf)
    bt_table, bt_w, curves, split = bt.walk_forward(r, assets, a.rf, cap=a.cap)

    table.to_csv(f"{a.out}/asset_stats.csv"); corr.to_csv(f"{a.out}/correlation.csv")
    weights.to_csv(f"{a.out}/weights.csv"); bt_table.to_csv(f"{a.out}/backtest.csv")
    json.dump(rep, open(f"{a.out}/cleaning_report.json", "w"), indent=2)
    rp.growth(prices, f"{a.out}/01_growth.png"); rp.drawdowns(r, [a.benchmark, "TECH", "ENERGY", "BOND"], f"{a.out}/02_drawdown.png")
    rp.correlation(r, f"{a.out}/03_correlation.png")
    st = lambda w: p.stats(w.values, mu.values, cov.values, a.rf)
    rs, rv = st(w_ms), st(w_mv)
    front = front[front.ret >= rv[0] - 1e-9]   # keep only the efficient (upper) branch
    rp.frontier(cloud, front, [("Max Sharpe", (rs[1], rs[0]), "*"), ("Min variance", (rv[1], rv[0]), "D")], f"{a.out}/04_frontier.png")
    rp.backtest(curves, split, f"{a.out}/05_backtest.png")

    md = f"""# Market data analysis: results

Data: {len(rep['assets'])} series, {rep['start']} to {rep['end']} ({rep['rows_clean']} clean trading days). Synthetic sample data.

## Data cleaning
Raw rows {rep['rows_raw']}; duplicate dates removed {rep['duplicate_dates_removed']}; non-positive prices {rep['non_positive_prices']};
missing cells before fill {rep['missing_cells_before']}; rows dropped as still incomplete {rep['rows_dropped_incomplete']}.

## Risk and return per asset (risk-free {a.rf:.1%})
{rp.md_table(table, FMT)}
## Optimal portfolios (max weight {a.cap:.0%} per asset)
{rp.md_table(weights, FMT)}
Max Sharpe: return {rs[0]:.1%}, volatility {rs[1]:.1%}, Sharpe {rs[2]:.2f}. Min variance: return {rv[0]:.1%}, volatility {rv[1]:.1%}, Sharpe {rv[2]:.2f}.

## Out-of-sample test (weights trained on data before {split.date()})
{rp.md_table(bt_table, FMT)}
In-sample Sharpe is optimistic by construction; the table above is the honest comparison.
"""
    open(f"{a.out}/summary.md", "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
