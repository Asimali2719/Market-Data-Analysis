# Market data analysis: results

Data: 7 series, 2022-01-03 to 2025-12-31 (1043 clean trading days). Synthetic sample data.

## Data cleaning
Raw rows 1048; duplicate dates removed 5; non-positive prices 1;
missing cells before fill 60; rows dropped as still incomplete 0.

## Risk and return per asset (risk-free 3.0%)
| | ann_return | ann_vol | sharpe | sortino | max_drawdown | var_95_1d | cvar_95_1d | beta | alpha |
|---|---|---|---|---|---|---|---|---|---|
| MKT | 1.7% | 15.7% | -0.01 | -0.01 | -23.3% | 1.57% | 2.18% | 1.00 | -0.0% |
| TECH | -10.3% | 26.1% | -0.40 | -0.55 | -60.9% | 2.68% | 3.46% | 1.36 | -10.3% |
| BANK | 13.5% | 21.7% | 0.55 | 0.80 | -21.9% | 2.10% | 2.91% | 1.09 | 12.1% |
| ENERGY | 3.1% | 22.8% | 0.12 | 0.16 | -37.6% | 2.27% | 2.90% | 0.83 | 2.7% |
| HEALTH | -1.5% | 16.4% | -0.19 | -0.27 | -37.0% | 1.62% | 2.07% | 0.76 | -3.0% |
| UTIL | 12.0% | 10.8% | 0.83 | 1.20 | -12.2% | 1.04% | 1.41% | 0.44 | 9.0% |
| BOND | 4.1% | 3.3% | 0.32 | 0.46 | -3.6% | 0.30% | 0.40% | -0.06 | 1.0% |

## Optimal portfolios (max weight 40% per asset)
| | max_sharpe | min_variance |
|---|---|---|
| TECH | 0.0% | 0.0% |
| BANK | 20.0% | 0.0% |
| ENERGY | 0.0% | 3.0% |
| HEALTH | 0.0% | 17.0% |
| UTIL | 40.0% | 40.0% |
| BOND | 40.0% | 40.0% |

Max Sharpe: return 9.4%, volatility 7.4%, Sharpe 0.86. Min variance: return 6.5%, volatility 6.3%, Sharpe 0.56.

## Out-of-sample test (weights trained on data before 2025-01-01)
| | ann_return | ann_vol | sharpe | max_drawdown |
|---|---|---|---|---|
| Max-Sharpe (trained) | 8.5% | 7.2% | 0.76 | -5.3% |
| Equal weight | 1.0% | 11.5% | -0.11 | -13.9% |

In-sample Sharpe is optimistic by construction; the table above is the honest comparison.
