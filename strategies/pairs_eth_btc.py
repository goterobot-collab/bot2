#!/usr/bin/env python3
"""
Cointegration-based pairs trading: ETH vs BTC (1h).

Hypothesis: ETHUSD and BTCUSD are co-moving crypto majors. Their log-price
spread, after fitting a hedge ratio, is mean-reverting on short horizons.
When the spread (ETH cheap relative to BTC, z<<0) starts turning back up,
go LONG ETH. Exit when the spread normalises (|z|<0.5).

This is a stand-alone backtest — it is NOT registered into the main
STRATEGIES dict / grail_loop SPACES, because pairs trading needs two
synchronised assets and the framework is single-asset.

Run:
    python3 strategies/pairs_eth_btc.py
Outputs:
    results/pairs_eth_btc_report.md   (WR / PF / Ret / DD / trades)

Method:
    1) Inner-join ETH and BTC on timestamp.
    2) Hedge ratio  beta = OLS slope of log(ETH) on log(BTC)
       (one-shot in-sample for simplicity; easy to convert to rolling
       later if walk-forward leakage becomes a concern).
    3) spread_t = log(ETH_t) - beta * log(BTC_t)
       z_t      = (spread_t - SMA(spread,100)) / STD(spread,100)
    4) ENTRY (long ETH at next bar open):
           z[i-1] < -2.0  AND  z[i-1] > z[i-2]    (mean-reversion turn-up)
       EXIT:
           |z[i-1]| < 0.5  (spread back to normal)
       Short side is intentionally close-only — i.e. NOT taken — per spec
       (this codebase is long-only, no synthetic shorts).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest import load_csv, simulate, metrics_from_trades  # noqa: E402

DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "results"

# -- Strategy parameters ------------------------------------------------------
Z_WINDOW       = 100      # rolling window for z-score mean / std
Z_ENTRY        = -2.0     # long when z < this AND turning up
Z_EXIT         = 0.5      # exit when |z| < this
FEE            = 0.001    # 10 bps / side (matches backtest.py default)
SLIPPAGE       = 0.0005   # 5  bps / side
SL_ATR         = None     # spread-driven exit only — no ATR stop
TP_ATR         = None
TRAIL_ATR      = None
TIMEOUT        = 200      # safety: force-exit if spread never normalises (~8 days)


def hedge_ratio(log_eth: np.ndarray, log_btc: np.ndarray) -> tuple[float, float]:
    """OLS slope/intercept of log_eth = alpha + beta * log_btc."""
    x = np.column_stack([np.ones_like(log_btc), log_btc])
    coef, *_ = np.linalg.lstsq(x, log_eth, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    return alpha, beta


def build_signals(eth: pd.DataFrame, btc: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """Inner-join on timestamp, build entry/exit Series aligned to ETH bars.

    Returns (eth_aligned_df, entry_series, exit_series, info_dict)."""
    # Inner-join on timestamp index (both already UTC-indexed by load_csv)
    joined = eth.join(btc[["close"]].rename(columns={"close": "btc_close"}), how="inner")
    if len(joined) < Z_WINDOW + 50:
        raise RuntimeError(f"Not enough overlapping bars: {len(joined)}")

    log_eth = np.log(joined["close"].values)
    log_btc = np.log(joined["btc_close"].values)
    alpha, beta = hedge_ratio(log_eth, log_btc)

    spread = pd.Series(log_eth - beta * log_btc, index=joined.index)
    mu  = spread.rolling(Z_WINDOW).mean()
    sd  = spread.rolling(Z_WINDOW).std()
    z   = (spread - mu) / sd

    # Entry: z[t-1] < Z_ENTRY AND turning up (z[t-1] > z[t-2])
    # Using shift(1)/shift(2) so signal at bar t uses only data through bar t-1.
    z_prev  = z.shift(1)
    z_prev2 = z.shift(2)
    entry = (z_prev < Z_ENTRY) & (z_prev > z_prev2)
    # Exit when spread is back near zero
    exit_ = z_prev.abs() < Z_EXIT

    entry = entry.fillna(False).astype(bool)
    exit_ = exit_.fillna(False).astype(bool)

    info = {
        "alpha": alpha,
        "beta": beta,
        "n_overlap": len(joined),
        "n_entry_signals": int(entry.sum()),
        "n_exit_signals":  int(exit_.sum()),
        "first": str(joined.index[0]),
        "last":  str(joined.index[-1]),
    }
    return joined, entry, exit_, info


def main():
    eth_path = DATA_DIR / "ETHUSD_1h.csv"
    btc_path = DATA_DIR / "BTCUSD_1h.csv"
    if not eth_path.exists() or not btc_path.exists():
        sys.exit(f"Missing data: need {eth_path} and {btc_path}")

    eth = load_csv(eth_path)
    btc = load_csv(btc_path)

    joined, entry, exit_, info = build_signals(eth, btc)

    print(f"Overlap: {info['n_overlap']} bars  ({info['first']} -> {info['last']})")
    print(f"Hedge ratio: log(ETH) = {info['alpha']:.4f} + {info['beta']:.4f} * log(BTC)")
    print(f"Entry signals: {info['n_entry_signals']}   Exit signals: {info['n_exit_signals']}")

    # simulate() expects a df with open/high/low/close on the asset we are
    # trading (ETH). entry / exit must be aligned to the same index.
    eth_aligned = joined[["open", "high", "low", "close", "volume"]]

    trades = simulate(
        eth_aligned, entry, exit_,
        fee=FEE, slippage=SLIPPAGE,
        sl_atr=SL_ATR, tp_atr=TP_ATR,
        trail_atr=TRAIL_ATR, timeout=TIMEOUT,
    )
    m = metrics_from_trades("pairs_eth_btc", "ETHUSD", "1h", trades, eth_aligned)

    print(f"\nTrades={m.trades}  WR={m.wr}%  PF={m.profit_factor}  "
          f"Ret={m.total_return_pct}%  DD={m.max_drawdown_pct}%")
    print(f"Exits: {m.exits_by_reason}")

    # Markdown report
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pairs trading: ETH vs BTC (1h, cointegration)",
        "",
        "## Method",
        "- Inner-join ETHUSD and BTCUSD on hourly timestamp.",
        "- Hedge ratio via OLS: `log(ETH) = alpha + beta * log(BTC)` (in-sample, full history).",
        "- Spread = `log(ETH) - beta * log(BTC)`.",
        f"- Z-score: rolling {Z_WINDOW}-bar mean/std of spread.",
        f"- LONG ETH when `z[t-1] < {Z_ENTRY}` AND `z[t-1] > z[t-2]` (turning up).",
        f"- EXIT when `|z[t-1]| < {Z_EXIT}` OR after {TIMEOUT}-bar timeout.",
        "- Short side intentionally NOT taken (long-only framework).",
        "- Execution: next-bar open, fees 10bps/side, slippage 5bps/side.",
        "",
        "## Fit",
        f"- Overlapping bars: **{info['n_overlap']}** ({info['first']} -> {info['last']})",
        f"- alpha = {info['alpha']:.6f}",
        f"- beta  = {info['beta']:.6f}",
        f"- Raw entry signals: {info['n_entry_signals']}",
        f"- Raw exit signals:  {info['n_exit_signals']}",
        "",
        "## Backtest result (single run)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Trades | {m.trades} |",
        f"| Wins / Losses | {m.wins} / {m.losses} |",
        f"| Win rate (WR) | {m.wr}% |",
        f"| Profit factor (PF) | {m.profit_factor} |",
        f"| Total return | {m.total_return_pct}% |",
        f"| Max drawdown (DD) | {m.max_drawdown_pct}% |",
        f"| Avg win | {m.avg_win_pct}% |",
        f"| Avg loss | {m.avg_loss_pct}% |",
        f"| Exits by reason | {m.exits_by_reason} |",
        f"| Period | {m.first_bar} -> {m.last_bar} |",
        "",
        "## Caveats",
        "- Hedge ratio fitted in-sample on the full overlap (look-ahead in the",
        "  beta only, not in the z-score which is rolling). For a deployable",
        "  variant, refit beta on a rolling window or a walk-forward train slice.",
        "- Long-only: half of the cointegration edge (short ETH when z>>0) is",
        "  discarded. WR and trade count reflect long leg only.",
        "- Not registered in `grail_loop.py` SPACES — this is stand-alone.",
    ]
    report_path = OUT_DIR / "pairs_eth_btc_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
