"""Time-Series Momentum (TSMOM) with vol-targeting.

References:
    - Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum"
    - Bianchi & Babiak (2022) "Time-Series Momentum in Cryptocurrencies"

Logic:
    Entry (long-only):
        - 12-bar return (close.pct_change(mom_lb)) > 0
            => price has positive momentum over the lookback window
        - close > N-bar SMA (trend filter, lookback in {100,150,200})
            => structural uptrend confirmation
        - realized vol over `vol_len` bars is BELOW its rolling
          quantile (default 0.7 over `vol_win` bars)
            => "vol-targeting" surrogate. The original TSMOM scales the
               position by 1/sigma; this framework uses fixed equity, so
               instead we *gate* entries to low-vol regimes (which is what
               vol-targeting effectively does — it shrinks size when vol
               spikes; we just shrink to zero).

    Exit (signal):
        - 12-bar return turns negative (mom flips)
        - OR close < N-bar SMA (trend break)
        Standard ATR sl/tp/trail/timeout exits are layered on top by the
        framework's `simulate()`.

Builder signature matches the rest of grail_loop.py:
    sig_tsmom_voltarget(df, p) -> (entry: bool Series, exit: bool Series)
"""
from __future__ import annotations

import pandas as pd

from strategies.indicators import sma


def sig_tsmom_voltarget(df: pd.DataFrame, p: dict):
    c = df["close"]

    # 1) Momentum: N-bar simple return
    mom = c.pct_change(p["mom_lb"])
    mom_pos = mom > 0

    # 2) Trend filter: price above long SMA
    s = sma(c, p["sma_trend"])
    trend_up = c > s

    # 3) Vol-target gate: realized vol below its rolling quantile.
    # Realized vol = rolling std of simple returns over vol_len bars.
    # MOP-style scaling (1/sigma) collapses to a binary gate here because
    # the framework uses fixed equity per trade.
    rets = c.pct_change()
    rv = rets.rolling(p["vol_len"], min_periods=p["vol_len"]).std(ddof=0)
    rv_q = rv.rolling(p["vol_win"], min_periods=p["vol_win"]).quantile(p["vol_q"])
    low_vol = rv < rv_q

    entry = mom_pos & trend_up & low_vol

    # Signal exit: momentum flips negative OR trend break
    ex = (mom < 0) | (c < s)

    return entry.fillna(False), ex.fillna(False)


# ---------------------------------------------------------------------------
# Param grid (document & expose for SPACES registration)
# ---------------------------------------------------------------------------
# mom_lb     : momentum lookback (bars). 12 = canonical MOP. We give a small
#              spread so the search can adapt per timeframe.
# sma_trend  : trend SMA lookback. 100/150/200 per task spec, plus 250.
# vol_len    : realized-vol window (bars). 20 = canonical.
# vol_win    : rolling window for the vol quantile gate.
# vol_q      : quantile threshold; only enter when current vol < this quantile.
PARAMS_GRID = {
    "mom_lb":    [8, 12, 16, 24, 36, 48],
    "sma_trend": [100, 150, 200, 250],
    "vol_len":   [10, 14, 20, 30],
    "vol_win":   [200, 400, 800],
    "vol_q":     [0.5, 0.6, 0.7, 0.8, 0.9],
}

# Standard ATR-based exit grid; trail emphasized because trend-following
# benefits from letting winners run.
EXIT_GRID = {
    "sl_atr":    [None, 1.5, 2.0, 2.5, 3.0, 4.0],
    "tp_atr":    [None, 3.0, 4.0, 6.0, 8.0, 12.0],
    "trail_atr": [None, 2.0, 3.0, 4.0, 5.0, 6.0],
    "timeout":   [None, 24, 48, 72, 120, 168, 240],
}


# ---------------------------------------------------------------------------
# Integration snippet for grail_loop.SPACES
# ---------------------------------------------------------------------------
# Add to grail_loop.py:
#
#     from strategies.tsmom import sig_tsmom_voltarget, PARAMS_GRID as _TSMOM_P, EXIT_GRID as _TSMOM_E
#
#     SPACES["tsmom_voltarget"] = {"sig": sig_tsmom_voltarget, "params": _TSMOM_P, "exit": _TSMOM_E}
#
# (One line if you collapse the import.)


# ---------------------------------------------------------------------------
# Manual test entry point: count entries on an ETHUSD CSV.
# Run:  python3 -m strategies.tsmom
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backtest import load_csv

    csv = Path(__file__).resolve().parents[1] / "data" / "ETHUSD_1h.csv"
    df = load_csv(csv)

    test_p = {
        "mom_lb":    12,
        "sma_trend": 150,
        "vol_len":   20,
        "vol_win":   400,
        "vol_q":     0.7,
    }
    ent, ex = sig_tsmom_voltarget(df, test_p)
    print(f"CSV:           {csv.name}")
    print(f"Bars loaded:   {len(df):,}")
    print(f"Params:        {test_p}")
    print(f"Entry signals: {int(ent.sum()):,}")
    print(f"Exit signals:  {int(ex.sum()):,}")
    print(f"Entry rate:    {ent.mean()*100:.2f}% of bars")
