#!/usr/bin/env python3
"""
Grail Loop — random-search parameter sweep over top strategies.

Goal: find configurations hitting **WR>60% AND PnL>0 AND maxDD>-30% AND PF>1**
(= "grail") on real OHLCV. Each iteration samples one strategy + one random
parameter combo, builds signals, runs the same simulator as backtest.py, and
if it's a grail, appends to results/grails_loop.md.

Usage:
    python3 grail_loop.py                           # 500 iter, default filters
    python3 grail_loop.py --iter 5000 --seed 7
    python3 grail_loop.py --min-wr 0.65 --grail-dd 25
    python3 grail_loop.py --assets ETHUSD           # only ETH
    python3 grail_loop.py --report-every 100

Output (incremental, crash-safe):
    results/grails_loop.md     — grail configs (appended as found)
    results/grails_loop.jsonl  — every iteration's metrics (one JSON per line)
"""
from __future__ import annotations
import argparse
import json
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategies.indicators import (
    ema, sma, rsi, macd, bbands, stoch, supertrend, atr, crossover, crossunder,
)
from backtest import simulate, metrics_from_trades, load_csv

DATA_DIR = Path("data")
OUT_DIR = Path("results")


# ---------- parameterized signal builders -----------------------------------
def sig_rsi2_regime(df, p):
    c = df["close"]
    s = sma(c, p["sma_trend"])
    rising = s > s.shift(p["slope_bars"])
    r = rsi(c, p["rsi_len"])
    entry = rising & (c > s) & (r < p["rsi_buy"])
    ex = c > sma(c, p["exit_sma"])
    return entry.fillna(False), ex.fillna(False)


def sig_supertrend_atrstop(df, p):
    _, dirn = supertrend(df["high"], df["low"], df["close"], p["st_atr"], p["st_mult"])
    trend = ema(df["close"], p["ema_trend"])
    flip_up = (dirn == 1) & (dirn.shift(1) == -1)
    flip_dn = (dirn == -1) & (dirn.shift(1) == 1)
    entry = flip_up & (df["close"] > trend)
    return entry.fillna(False), flip_dn.fillna(False)


def sig_ema_cross_trend(df, p):
    c = df["close"]
    ef = ema(c, p["fast"])
    es = ema(c, p["slow"])
    et = ema(c, p["trend"])
    entry = crossover(ef, es) & (c > et)
    ex = crossunder(ef, es)
    return entry.fillna(False), ex.fillna(False)


def sig_macd_trend(df, p):
    c = df["close"]
    m, s, _ = macd(c, p["fast"], p["slow"], p["sig"])
    t = ema(c, p["trend"])
    entry = crossover(m, s) & (c > t)
    ex = crossunder(m, s)
    return entry.fillna(False), ex.fillna(False)


def sig_donchian_trail(df, p):
    c = df["close"]
    upper = df["high"].rolling(p["dc_len"], min_periods=p["dc_len"]).max().shift(1)
    entry = c > upper
    ex = pd.Series(False, index=c.index)
    return entry.fillna(False), ex


def sig_bb_trend_rejoin(df, p):
    """Mean-rev with trend filter: enter when close crosses lower BB AND close > EMA200."""
    c = df["close"]
    lo, mid, _ = bbands(c, p["bb_len"], p["bb_mult"])
    t = ema(c, p["trend"])
    entry = crossover(c, lo) & (c > t)
    ex = crossover(c, mid)
    return entry.fillna(False), ex.fillna(False)


def sig_keltner_squeeze(df, p):
    """BB inside Keltner = squeeze; enter on breakout up + trend filter."""
    c = df["close"]; h = df["high"]; l = df["low"]
    # BB
    bmid = sma(c, p["len"])
    bstd = c.rolling(p["len"], min_periods=p["len"]).std(ddof=0)
    bb_up = bmid + p["bb_mult"] * bstd
    bb_dn = bmid - p["bb_mult"] * bstd
    # Keltner
    kmid = ema(c, p["len"])
    a = atr(h, l, c, p["len"])
    kc_up = kmid + p["kc_mult"] * a
    kc_dn = kmid - p["kc_mult"] * a
    squeeze = (bb_up < kc_up) & (bb_dn > kc_dn)
    # squeeze released AND close breaking above recent range
    released = squeeze.shift(1) & (~squeeze)
    trend_ok = c > ema(c, p["trend"])
    entry = released & (c > bmid) & trend_ok
    ex = crossunder(c, bmid)
    return entry.fillna(False), ex.fillna(False)


def sig_zscore_revert(df, p):
    """Z-score mean reversion with trend filter."""
    c = df["close"]
    m = sma(c, p["z_len"])
    s = c.rolling(p["z_len"], min_periods=p["z_len"]).std(ddof=0)
    z = (c - m) / s.replace(0, 1e-9)
    t = ema(c, p["trend"])
    # enter when z crosses up from deep oversold AND above trend EMA
    entry = crossover(z, pd.Series(-p["z_enter"], index=c.index)) & (c > t)
    ex = crossover(z, pd.Series(0.0, index=c.index))
    return entry.fillna(False), ex.fillna(False)


def sig_psar_trend(df, p):
    """Breakout-of-N-bar-high with EMA trend and slope filter (PSAR-like)."""
    c = df["close"]; h = df["high"]; l = df["low"]
    breakout = c > h.rolling(p["hi_len"], min_periods=p["hi_len"]).max().shift(1)
    t = ema(c, p["trend"])
    entry = breakout & (c > t) & (t > t.shift(p["slope_bars"]))
    ex = c < l.rolling(p["exit_len"], min_periods=p["exit_len"]).min().shift(1)
    return entry.fillna(False), ex.fillna(False)


def sig_adx_pullback(df, p):
    """Trend-confirmed pullback: ADX>threshold + close pulls back to EMA + recovers."""
    c = df["close"]; h = df["high"]; l = df["low"]
    # ADX via DMI approximation inline
    up = h.diff()
    dn = -l.diff()
    import numpy as _np
    pdm = pd.Series(_np.where((up > dn) & (up > 0), up, 0.0), index=c.index)
    mdm = pd.Series(_np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index)
    a = atr(h, l, c, p["adx_len"])
    pdi = 100 * pdm.ewm(alpha=1.0/p["adx_len"], adjust=False, min_periods=p["adx_len"]).mean() / a
    mdi = 100 * mdm.ewm(alpha=1.0/p["adx_len"], adjust=False, min_periods=p["adx_len"]).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, _np.nan)
    adx = dx.ewm(alpha=1.0/p["adx_len"], adjust=False, min_periods=p["adx_len"]).mean()
    e = ema(c, p["ema_len"])
    touched = (l <= e).rolling(p["lookback"], min_periods=1).max().astype(bool)
    trend_up = adx > p["adx_min"]
    di_up = pdi > mdi
    recovery = (c > c.shift(1)) & (c > e)
    entry = trend_up & di_up & touched & recovery
    ex = crossunder(c, e)
    return entry.fillna(False), ex.fillna(False)


# ---------- parameter spaces -------------------------------------------------
# Each entry: builder + param grid + exit-config grid (sl/tp/trail/timeout)
SPACES = {
    "rsi2_regime": {
        "sig": sig_rsi2_regime,
        "params": {
            "rsi_len":    [2, 3, 4, 5],
            "rsi_buy":    [3, 5, 7, 8, 10, 12, 15, 18, 20, 25, 30],
            "sma_trend":  [75, 100, 125, 150, 175, 200, 225, 250, 300, 400],
            "slope_bars": [5, 10, 15, 20, 30, 40, 50, 65, 80, 100],
            "exit_sma":   [2, 3, 5, 8, 13, 21],
        },
        "exit": {
            "sl_atr":    [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
            "tp_atr":    [None, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0],
            "trail_atr": [None, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
            "timeout":   [None, 8, 12, 18, 24, 36, 48, 72, 96, 120, 168],
        },
    },
    "supertrend_atr": {
        "sig": sig_supertrend_atrstop,
        "params": {
            "st_atr":    [7, 10, 14, 20],
            "st_mult":   [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
            "ema_trend": [50, 100, 150, 200, 250],
        },
        "exit": {
            "sl_atr":    [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
            "tp_atr":    [None, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0],
            "trail_atr": [None, 2.0, 3.0, 4.0, 5.0],
            "timeout":   [None],
        },
    },
    "ema_cross_trend": {
        "sig": sig_ema_cross_trend,
        "params": {
            "fast":  [5, 9, 12, 21, 34],
            "slow":  [21, 34, 50, 89, 144],
            "trend": [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [None, 1.5, 2.0, 3.0],
            "tp_atr":    [None, 3.0, 4.0, 6.0, 8.0, 12.0],
            "trail_atr": [None, 2.0, 3.0, 4.0, 5.0],
            "timeout":   [None],
        },
    },
    "macd_trend": {
        "sig": sig_macd_trend,
        "params": {
            "fast":  [8, 12, 16, 20],
            "slow":  [20, 26, 34, 50],
            "sig":   [5, 9, 12, 16],
            "trend": [100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [None, 1.5, 2.0, 3.0],
            "tp_atr":    [None, 3.0, 4.0, 5.0, 8.0],
            "trail_atr": [None, 2.0, 3.0, 4.0],
            "timeout":   [None],
        },
    },
    "donchian_trail": {
        "sig": sig_donchian_trail,
        "params": {
            "dc_len": [10, 20, 30, 55, 80],
        },
        "exit": {
            "sl_atr":    [None, 2.0, 3.0],
            "tp_atr":    [None],
            "trail_atr": [2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
            "timeout":   [None],
        },
    },
    "bb_trend_rejoin": {
        "sig": sig_bb_trend_rejoin,
        "params": {
            "bb_len":  [8, 10, 14, 20, 24, 30, 40, 50, 60, 80, 100],
            "bb_mult": [1.2, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5],
            "trend":   [50, 75, 100, 125, 150, 175, 200, 250, 300],
        },
        "exit": {
            "sl_atr":    [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "tp_atr":    [None, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0],
            "trail_atr": [None, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
            "timeout":   [None, 12, 18, 24, 36, 48, 72, 96, 120, 168],
        },
    },
    "keltner_squeeze": {
        "sig": sig_keltner_squeeze,
        "params": {
            "len":     [14, 20, 30, 50],
            "bb_mult": [1.5, 2.0, 2.5],
            "kc_mult": [1.0, 1.5, 2.0, 2.5],
            "trend":   [50, 100, 150, 200, 300],
        },
        "exit": {
            "sl_atr":    [1.0, 1.5, 2.0, 3.0],
            "tp_atr":    [None, 2.0, 3.0, 4.0, 6.0],
            "trail_atr": [None, 2.0, 3.0, 4.0],
            "timeout":   [None, 24, 48, 72],
        },
    },
    "zscore_revert": {
        "sig": sig_zscore_revert,
        "params": {
            "z_len":   [10, 14, 20, 30, 50, 80],
            "z_enter": [1.5, 2.0, 2.5, 3.0],
            "trend":   [100, 150, 200, 250, 300],
        },
        "exit": {
            "sl_atr":    [1.0, 1.5, 2.0, 2.5, 3.0],
            "tp_atr":    [None, 2.0, 3.0, 4.0, 5.0],
            "trail_atr": [None, 2.0, 3.0, 4.0],
            "timeout":   [None, 24, 48, 72, 120],
        },
    },
    "psar_trend": {
        "sig": sig_psar_trend,
        "params": {
            "hi_len":     [10, 14, 20, 30, 40, 55],
            "exit_len":   [5, 8, 10, 15, 20],
            "trend":      [100, 150, 200, 300],
            "slope_bars": [10, 20, 30, 50],
        },
        "exit": {
            "sl_atr":    [None, 1.5, 2.0, 3.0],
            "tp_atr":    [None, 3.0, 5.0, 8.0],
            "trail_atr": [2.0, 3.0, 4.0, 5.0],
            "timeout":   [None],
        },
    },
    "adx_pullback": {
        "sig": sig_adx_pullback,
        "params": {
            "adx_len":  [10, 14, 20],
            "adx_min":  [20, 25, 30, 35, 40],
            "ema_len":  [13, 20, 34, 50],
            "lookback": [3, 5, 8, 12],
        },
        "exit": {
            "sl_atr":    [0.75, 1.0, 1.5, 2.0],
            "tp_atr":    [None, 2.0, 3.0, 4.0, 6.0],
            "trail_atr": [None, 2.0, 3.0],
            "timeout":   [None, 24, 48, 72],
        },
    },
}

# Plug TSMOM (registered out-of-band so the family table stays clean).
try:
    from strategies.tsmom import sig_tsmom_voltarget, PARAMS_GRID as _TSMOM_P, EXIT_GRID as _TSMOM_E
    SPACES["tsmom_voltarget"] = {"sig": sig_tsmom_voltarget, "params": _TSMOM_P, "exit": _TSMOM_E}
except Exception as _e:
    print(f"[warn] could not load tsmom: {_e}")

# Plug 6 new diversification families (ConnorsRSI, Williams %R, Ichimoku,
# Hull MA cross, Stochastic cross, Volatility breakout).
try:
    from strategies.new_families import SPACES_NEW
    SPACES.update(SPACES_NEW)
except Exception as _e:
    print(f"[warn] could not load new_families: {_e}")

# Plug round-2 families (CCI, CMF, Heikin-Ashi, Aroon, KAMA, Gap Fade).
try:
    from strategies.more_families import SPACES_MORE
    SPACES.update(SPACES_MORE)
except Exception as _e:
    print(f"[warn] could not load more_families: {_e}")

# Plug NOVEL families (microstructure + state-space + cycle analysis,
# specifically chosen to be absent from the user's V8 dashboard).
try:
    from strategies.novel_families import SPACES_NOVEL
    SPACES.update(SPACES_NOVEL)
except Exception as _e:
    print(f"[warn] could not load novel_families: {_e}")


@dataclass
class Grail:
    strategy: str
    asset: str
    tf: str
    params: dict
    exit_cfg: dict
    trades: int
    wr: float
    pf: float
    ret: float
    dd: float
    exits_by_reason: str


def sample(space: dict, rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in space.items()}


def validate_exits(e: dict, fam: str) -> bool:
    """Keep combinations that make sense."""
    if e.get("tp_atr") is not None and e.get("sl_atr") is not None:
        if e["tp_atr"] <= e["sl_atr"]:
            return False  # negative R:R
    if e.get("tp_atr") is None and e.get("trail_atr") is None and fam != "bb_trend_rejoin":
        # need at least one exit mechanism (besides signal)
        return True
    return True


def evaluate(df, fam, params, exit_cfg, fee, slippage, asset, tf):
    """Walk-forward: split data 50/50. A grail must pass filter in BOTH
    halves AND on full history. Returns dict with 3 metric sets or None."""
    builder = SPACES[fam]["sig"]
    try:
        ent, ex = builder(df, params)
    except Exception:
        return None

    def _eval(sub_df, sub_ent, sub_ex):
        trades = simulate(sub_df, sub_ent, sub_ex, fee=fee, slippage=slippage,
                          sl_atr=exit_cfg.get("sl_atr"),
                          tp_atr=exit_cfg.get("tp_atr"),
                          trail_atr=exit_cfg.get("trail_atr"),
                          timeout=exit_cfg.get("timeout"))
        return metrics_from_trades(fam, asset, tf, trades, sub_df)

    n = len(df)
    half = n // 2
    m_full = _eval(df, ent, ex)
    m1 = _eval(df.iloc[:half], ent.iloc[:half], ex.iloc[:half])
    m2 = _eval(df.iloc[half:], ent.iloc[half:], ex.iloc[half:])
    return {"full": m_full, "h1": m1, "h2": m2}


def write_grail_header(path: Path, min_wr, grail_dd, min_pf):
    if path.exists() and path.stat().st_size > 0:
        return
    path.write_text(
        f"# 🏆 GRAIL loop — found configs\n\n"
        f"- Filter: WR > {min_wr*100:.0f}% AND PnL > 0 AND maxDD > -{grail_dd}% AND PF > {min_pf}\n"
        f"- Trades > 5 (inherited from backtest.py)\n\n"
        "| # | Strategy | Asset | TF | Trades | WR % | PF | Ret % | DD % | Params | Exit cfg |\n"
        "|---|----------|-------|----|--------|------|-----|-------|------|--------|----------|\n"
    )


def append_grail(path: Path, idx: int, g: Grail):
    with path.open("a") as f:
        f.write(
            f"| {idx} | `{g.strategy}` | {g.asset} | {g.tf} | {g.trades} | "
            f"{g.wr} | {g.pf} | {g.ret} | {g.dd} | "
            f"`{json.dumps(g.params, separators=(',',':'))}` | "
            f"`{json.dumps({k:v for k,v in g.exit_cfg.items() if v is not None}, separators=(',',':'))}` |\n"
        )


def load_all(assets_filter: list[str] | None):
    out = {}
    for p in sorted(DATA_DIR.glob("*.csv")):
        stem = p.stem
        try:
            asset, tf = stem.rsplit("_", 1)
        except ValueError:
            continue
        if assets_filter and asset not in assets_filter:
            continue
        df = load_csv(p)
        if len(df) < 300:
            continue
        out[(asset, tf)] = df
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-wr", type=float, default=0.60)
    ap.add_argument("--min-trades", type=int, default=50,
                    help="Minimum trades over full history (v2 walk-forward default: 50; "
                         "each half must show >= min_trades/2)")
    ap.add_argument("--min-pf", type=float, default=1.0)
    ap.add_argument("--grail-dd", type=float, default=30.0)
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--slippage", type=float, default=0.0005)
    ap.add_argument("--assets", nargs="+", default=None)
    ap.add_argument("--tfs", nargs="+", default=None,
                    help="Restrict to specific timeframes (e.g. --tfs 1h 15m)")
    ap.add_argument("--report-every", type=int, default=50)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    grails_md = OUT_DIR / "grails_loop_wf.md"
    grails_jsonl = OUT_DIR / "grails_loop_wf.jsonl"
    write_grail_header(grails_md, args.min_wr, args.grail_dd, args.min_pf)
    jsonl_f = grails_jsonl.open("a")

    dfs = load_all(args.assets)
    if args.tfs:
        dfs = {k: v for k, v in dfs.items() if k[1] in args.tfs}
    if not dfs:
        print("No CSVs in data/. Run tools/convert_cryptopredictions.py first.")
        return
    print(f"Loaded {len(dfs)} (asset, tf) series: {list(dfs.keys())}")

    rng = random.Random(args.seed)
    families = list(SPACES.keys())
    # User instruction: rsi2_regime and bb_trend_rejoin are CONFIRMED (top
    # plateau). Stop concentrating on them; bias hard toward UNEXPLORED
    # families to find new edges.
    # User feedback: V8 dashboard already tests 24,431 grails across all
    # VWAP / BB / RSI / Stoch / Williams / CCI / ConnorsRSI / ADX / Keltner /
    # MACD / ATR-channel / MA-envelope / Rubber-band / TV_* families. So EVERY
    # family we had before is V8-covered. The only fresh search space is the
    # novel microstructure / state-space / cycle-analysis set.
    v8_covered = {
        # original 11
        "rsi2_regime", "bb_trend_rejoin", "zscore_revert", "supertrend_atr",
        "ema_cross_trend", "macd_trend", "donchian_trail", "keltner_squeeze",
        "psar_trend", "adx_pullback", "tsmom_voltarget",
        # round-1 diversification
        "connors_rsi", "williams_revert", "ichimoku", "hull_cross",
        "stoch_cross", "vol_breakout",
        # round-2 diversification
        "cci_revert", "cmf_pullback", "heikin_trend", "aroon_cross",
        "kama_trend", "gap_fade",
    }
    novel = {
        "hurst_regime", "kalman_residual", "ehlers_mama", "amihud_contrarian",
        "vrp_proxy", "coint_pair", "corwin_schultz", "kyle_lambda",
        "naked_poc", "bvc_ofi", "hawkes_burst",
    }
    family_weights = [0 if f in v8_covered else (10 if f in novel else 1) for f in families]

    stop = {"flag": False}
    def handler(*_): stop["flag"] = True; print("\n[interrupt] finishing current iter and exiting cleanly.")
    signal.signal(signal.SIGINT, handler)

    grail_count = 0
    t0 = time.time()
    best_ret = -1e9; best_row = None
    for i in range(1, args.iter + 1):
        if stop["flag"]:
            break
        fam = rng.choices(families, weights=family_weights, k=1)[0]
        params = sample(SPACES[fam]["params"], rng)
        exit_cfg = sample(SPACES[fam]["exit"], rng)
        if not validate_exits(exit_cfg, fam):
            continue
        # pick a random (asset, tf)
        asset, tf = rng.choice(list(dfs.keys()))
        df = dfs[(asset, tf)]
        res = evaluate(df, fam, params, exit_cfg, args.fee, args.slippage, asset, tf)
        if res is None:
            continue
        m = res["full"]; m1 = res["h1"]; m2 = res["h2"]
        row = {
            "iter": i, "strategy": fam, "asset": asset, "tf": tf,
            "params": params, "exit_cfg": exit_cfg,
            # full-history metrics
            "trades": m.trades, "wr": m.wr, "pf": m.profit_factor,
            "ret": m.total_return_pct, "dd": m.max_drawdown_pct,
            # walk-forward halves
            "h1_trades": m1.trades, "h1_wr": m1.wr, "h1_pf": m1.profit_factor,
            "h1_ret": m1.total_return_pct, "h1_dd": m1.max_drawdown_pct,
            "h2_trades": m2.trades, "h2_wr": m2.wr, "h2_pf": m2.profit_factor,
            "h2_ret": m2.total_return_pct, "h2_dd": m2.max_drawdown_pct,
        }
        jsonl_f.write(json.dumps(row) + "\n")

        # track best full-history return
        if m.total_return_pct > best_ret and m.trades > args.min_trades:
            best_ret = m.total_return_pct; best_row = row

        # strict walk-forward grail: must pass filter on full AND both halves.
        def passes(mm, min_tr):
            return (mm.trades > min_tr and mm.wr > args.min_wr * 100
                    and mm.total_return_pct > 0
                    and mm.max_drawdown_pct > -args.grail_dd
                    and mm.profit_factor > args.min_pf)
        # require min_trades on full, and at least min_trades/2 in each half
        half_min = max(10, args.min_trades // 2)
        is_grail = passes(m, args.min_trades) and passes(m1, half_min) and passes(m2, half_min)
        if is_grail:
            grail_count += 1
            g = Grail(strategy=fam, asset=asset, tf=tf, params=params, exit_cfg=exit_cfg,
                      trades=m.trades, wr=m.wr, pf=m.profit_factor,
                      ret=m.total_return_pct, dd=m.max_drawdown_pct,
                      exits_by_reason=m.exits_by_reason)
            append_grail(grails_md, grail_count, g)
            print(f"[iter {i}] 🏆 GRAIL #{grail_count}: {fam} {asset} {tf} "
                  f"WR={m.wr} PF={m.profit_factor} ret={m.total_return_pct}% DD={m.max_drawdown_pct}%")

        if i % args.report_every == 0:
            rate = i / max(1e-6, time.time() - t0)
            print(f"[iter {i}/{args.iter}] grails={grail_count}  rate={rate:.1f} it/s  "
                  f"best_ret_so_far={best_ret:.2f}%")

    jsonl_f.close()
    print(f"\nDone. {grail_count} grails in {i} iterations in {time.time()-t0:.1f}s.")
    if best_row:
        print(f"Best return seen: {best_row['strategy']} {best_row['asset']} {best_row['tf']} "
              f"ret={best_row['ret']}%  WR={best_row['wr']}%  PF={best_row['pf']}  DD={best_row['dd']}%")
        print(f"  params={best_row['params']}  exit={best_row['exit_cfg']}")


if __name__ == "__main__":
    main()
