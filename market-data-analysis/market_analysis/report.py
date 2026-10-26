"""Charts and markdown helpers."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import metrics as m

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def growth(prices, path):
    (prices / prices.iloc[0] * 100).plot(figsize=(8, 4), lw=1.3, title="Growth of 100 invested")
    plt.ylabel("Value"); plt.tight_layout(); plt.savefig(path); plt.close()


def drawdowns(r, cols, path):
    m.drawdown(r[cols]).mul(100).plot(figsize=(8, 3.5), lw=1.1, title="Drawdown from previous peak (%)")
    plt.tight_layout(); plt.savefig(path); plt.close()


def correlation(r, path):
    c = r.corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(c, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c.columns)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, shrink=0.8); ax.set_title("Correlation of daily returns")
    plt.tight_layout(); plt.savefig(path); plt.close()


def frontier(cloud, front, points, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(cloud.vol * 100, cloud.ret * 100, c=cloud.sharpe, s=5, cmap="viridis", alpha=0.6)
    ax.plot(front.vol * 100, front.ret * 100, color="black", lw=1.6, label="Efficient frontier")
    for label, (v, r_), mk in points:
        ax.scatter(v * 100, r_ * 100, marker=mk, s=90, edgecolor="black", zorder=5, label=label)
    fig.colorbar(sc, label="Sharpe ratio"); ax.set_xlabel("Volatility (%)"); ax.set_ylabel("Return (%)")
    ax.set_title("Efficient frontier"); ax.legend(); plt.tight_layout(); plt.savefig(path); plt.close()


def backtest(curves, split_date, path):
    (1 + curves).cumprod().mul(100).plot(figsize=(8, 3.8), lw=1.4, title=f"Out-of-sample test (weights fixed on data before {split_date.date()})")
    plt.tight_layout(); plt.savefig(path); plt.close()


def md_table(df, fmt):
    head = "| | " + " | ".join(df.columns) + " |\n|---|" + "---|" * len(df.columns) + "\n"
    rows = "".join("| " + str(i) + " | " + " | ".join(fmt.get(c, "{:.3f}").format(v) for c, v in row.items()) + " |\n" for i, row in df.iterrows())
    return head + rows
