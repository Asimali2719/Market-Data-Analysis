"""Loading, cleaning and generating price data."""
from __future__ import annotations
import numpy as np
import pandas as pd


def generate_sample(path: str, seed: int = 42, start: str = "2022-01-03", end: str = "2025-12-31") -> pd.DataFrame:
    """Write a deliberately messy synthetic price file (no market-data feed needed).

    Assets follow a one-factor model with a high-volatility stress regime. Defects are
    injected on purpose so the cleaning step has real work to do: missing cells,
    duplicated dates, a zero price and out-of-order rows.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    vol_mult = np.ones(n)
    vol_mult[400:470] = 2.2                      # stress regime
    mkt = rng.normal(0.0004, 0.009, n) * vol_mult
    spec = {  # name: (alpha, beta, idio vol)
        "TECH": (0.0003, 1.35, 0.009), "BANK": (0.0001, 1.10, 0.008), "ENERGY": (0.0002, 0.85, 0.012),
        "HEALTH": (0.0002, 0.70, 0.007), "UTIL": (0.0001, 0.45, 0.005), "BOND": (0.0001, -0.05, 0.002),
    }
    cols = {"MKT": mkt}
    for k, (a, b, s) in spec.items():
        cols[k] = a + b * mkt + rng.normal(0, s, n)
    prices = 100 * (1 + pd.DataFrame(cols, index=dates)).cumprod()
    prices.index.name = "Date"
    prices = prices.round(4)
    flat = prices.to_numpy(copy=True)
    holes = rng.random(flat.shape) < 0.008        # ~0.8% missing cells
    flat[holes] = np.nan
    flat[250, 2] = 0.0                              # bad print
    messy = pd.DataFrame(flat, index=prices.index, columns=prices.columns)
    messy = pd.concat([messy, messy.iloc[[10, 11, 12, 300, 301]]])   # duplicated dates
    messy = pd.concat([messy.iloc[500:700], messy.iloc[:500], messy.iloc[700:]])  # out of order
    messy.to_csv(path)
    return messy


def load_prices(path: str, max_gap: int = 3):
    """Load a wide price CSV (Date + one column per asset) and clean it.

    Returns (clean_prices, report). Steps: parse dates, drop duplicate dates (keep last),
    sort, treat non-positive prices as missing, forward-fill gaps up to ``max_gap`` days,
    then drop any rows still incomplete.
    """
    raw = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    report = {"rows_raw": len(raw), "assets": list(raw.columns)}
    df = raw[~raw.index.duplicated(keep="last")]
    report["duplicate_dates_removed"] = len(raw) - len(df)
    report["was_sorted"] = bool(df.index.is_monotonic_increasing)
    df = df.sort_index()
    bad = (df <= 0).sum().sum()
    df = df.mask(df <= 0)
    report["non_positive_prices"] = int(bad)
    report["missing_cells_before"] = int(df.isna().sum().sum())
    df = df.ffill(limit=max_gap)
    before = len(df)
    df = df.dropna()
    report["rows_dropped_incomplete"] = before - len(df)
    report["rows_clean"] = len(df)
    report["start"], report["end"] = str(df.index[0].date()), str(df.index[-1].date())
    return df, report


def fetch_yahoo(tickers, start, end):
    """Optional live download (needs `pip install yfinance` and internet access)."""
    import yfinance as yf
    return yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"].dropna()
