# Market Data Analysis

A Python pipeline that takes raw multi-asset price data and produces a cleaned dataset, return and risk metrics,
an efficient frontier, optimal long-only portfolios, and an out-of-sample test of those portfolios.

**Stack:** Python 3.11+, pandas, NumPy, SciPy, Matplotlib, pytest.

## Run it

```bash
pip install -r requirements.txt
python -m market_analysis --make-sample        # generates data/prices_sample.csv, then runs everything
python -m market_analysis --prices my_prices.csv --benchmark SPY --rf 0.04
python -m pytest                               # 7 tests
```

Input is a wide CSV: a `Date` column and one price column per asset. Results are written to `outputs/`.

## What it does

| Stage | Module | Detail |
|---|---|---|
| Clean | `data.py` | Parse dates, remove duplicate dates, sort, treat non-positive prices as missing, forward-fill gaps of up to 3 days, drop remaining incomplete rows, and log every action to `cleaning_report.json` |
| Measure | `metrics.py` | Compound annual return, volatility, Sharpe, Sortino, maximum drawdown, historical 95% VaR and CVaR, CAPM beta and alpha |
| Optimise | `portfolio.py` | Long-only maximum-Sharpe and minimum-variance portfolios with a per-asset weight cap, efficient frontier, random-portfolio cloud |
| Validate | `backtest.py` | Fit weights on the first 75% of history, evaluate on the unseen last 25% against equal weight |
| Report | `report.py`, `cli.py` | Five charts, CSV tables and `summary.md` |

## Results on the bundled sample (4 years, 6 assets plus a market index)

- Cleaning fixed 5 duplicate dates, 1 zero price, 60 missing cells and out-of-order rows without dropping any trading day.
- Maximum-Sharpe portfolio (40% cap): 9.4% return, 7.4% volatility, Sharpe 0.86; minimum variance: 6.5% return, 6.3% volatility.
- Out-of-sample (weights fitted before 2025): Sharpe 0.76 versus -0.11 for equal weight. See `outputs/summary.md` for the full tables and `outputs/*.png` for charts.

## Limitations (read before trusting the numbers)

- The sample file is **synthetic** (a one-factor model with a stress period and injected defects). Results illustrate the method, not real markets.
- Mean-variance optimisation is sensitive to estimated returns. The weight cap and the out-of-sample test exist to show that.
- No transaction costs, taxes or rebalancing are modelled; the backtest holds fixed weights.
- `data.fetch_yahoo()` (live download through `yfinance`) is included for convenience and was not run in the build environment.
