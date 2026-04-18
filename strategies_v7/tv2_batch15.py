#!/usr/bin/env python3
"""TV2 BATCH 15 — 30 estrategias SMC + Order Blocks + Market Structure 2026-04-01

  FVG_Bullish            — Bullish FVG retest entry (v5, ~5000L)
  FVG_Bearish            — Bearish FVG retest entry (v5, ~5000L)
  FVG_Bidir              — Both FVG directions (v5, ~3000L)
  FVG_EMA                — FVG + EMA trend filter (v5, ~2000L)
  FVG_Volume             — FVG + volume confirmation (v5, ~1500L)
  OB_Bullish             — Bullish Order Block retest (v5, ~4000L)
  OB_Bearish             — Bearish Order Block retest (v5, ~4000L)
  OB_FVG_Combo           — OB + FVG alignment combo (v5, ~3000L)
  OB_ATR                 — Order Block + ATR sizing (v6, ~2000L)
  OB_RSI                 — Order Block + RSI confirmation (v5, ~1500L)
  BOS_Strategy           — Break of Structure long/short (v5, ~4500L)
  CHoCH_Strategy         — Change of Character reversal (v5, ~3500L)
  BOS_CHoCH_Combo        — BOS + CHoCH combined (v5, ~2500L)
  BOS_EMA                — BOS + EMA trend alignment (v5, ~2000L)
  BOS_Volume             — BOS + volume surge (v5, ~1500L)
  Liquidity_Sweep_Bull   — Bullish liquidity sweep reversal (v5, ~3000L)
  Liquidity_Sweep_Bear   — Bearish liquidity sweep reversal (v5, ~3000L)
  Liquidity_Sweep_Bidir  — Both sweep directions (v5, ~2500L)
  Liquidity_EMA          — Liquidity sweep + EMA filter (v5, ~1500L)
  Liquidity_RSI          — Liquidity sweep + RSI recovery (v5, ~1200L)
  Premium_Discount       — Premium/Discount 50% retracement zones (v5, ~2500L)
  Premium_Discount_RSI   — P/D zones + RSI filter (v5, ~1500L)
  Wyckoff_Spring         — Wyckoff Spring support break reversal (v5, ~2000L)
  Wyckoff_Upthrust       — Wyckoff Upthrust resistance break reversal (v5, ~2000L)
  Wyckoff_Bidir          — Both Spring and Upthrust (v5, ~1500L)
  Inside_Bar_Breakout    — Inside bar breakout long/short (v5, ~3000L)
  Inside_Bar_EMA         — Inside bar breakout + EMA direction (v5, ~1800L)
  NR7_Breakout           — Narrowest Range 7 breakout (v5, ~2000L)
  EQH_EQL_Sweep          — Equal Highs/Lows liquidity sweep (v5, ~2000L)
  SMC_Full               — Full SMC: FVG + OB + BOS aligned (v5, ~3500L)
"""

import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── helpers ───────────────────────────────────────────────────────────────────

def _ema(s, p):
    return s.ewm(span=int(p), adjust=False).mean()


def _rma(s, p):
    return s.ewm(alpha=1.0 / int(p), adjust=False).mean()


def _sma(s, p):
    return s.rolling(int(p), min_periods=1).mean()


def _atr(df, p):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return _rma(tr, p)


def _rsi(s, p):
    d = s.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    rs = _rma(g, p) / _rma(l, p).replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def _pivot_high(s, left, right):
    w = int(left) + int(right) + 1
    return (s.rolling(w, center=True).max() == s)


def _pivot_low(s, left, right):
    w = int(left) + int(right) + 1
    return (s.rolling(w, center=True).min() == s)


# ── 1. FVG_Bullish ────────────────────────────────────────────────────────────

def gen_FVG_Bullish(df, lookback=10, min_gap_pct=0.002, **kw):
    """Bullish FVG: high[i-2] < low[i]. Long when price retraces into the gap."""
    lookback     = int(lookback)
    min_gap_pct  = float(min_gap_pct)
    close        = df['close']
    high         = df['high']
    low          = df['low']
    n            = len(df)
    out          = pd.Series(0, index=df.index)

    # For each bar, track whether a bullish FVG was formed in the last `lookback` bars
    # Bullish FVG at bar j: high[j-2] < low[j]  (gap between j-2 top and j bottom)
    # Retest: current price enters zone [high[j-2], low[j]]
    for i in range(lookback + 2, n):
        for j in range(i - lookback, i):
            if j < 2:
                continue
            gap_top    = low.iloc[j]
            gap_bottom = high.iloc[j - 2]
            if gap_bottom >= gap_top:
                continue
            gap_size = (gap_top - gap_bottom) / gap_bottom
            if gap_size < min_gap_pct:
                continue
            # Retest: current close inside gap zone
            c = close.iloc[i]
            if gap_bottom <= c <= gap_top:
                out.iloc[i] = 1
                break
    return out


def space_FVG_Bullish(trial):
    return {
        'lookback':    trial.suggest_int('lookback',    3, 20),
        'min_gap_pct': trial.suggest_float('min_gap_pct', 0.001, 0.005),
    }


# ── 2. FVG_Bearish ────────────────────────────────────────────────────────────

def gen_FVG_Bearish(df, lookback=10, min_gap_pct=0.002, **kw):
    """Bearish FVG: low[i-2] > high[i]. Short when price rallies into the gap."""
    lookback    = int(lookback)
    min_gap_pct = float(min_gap_pct)
    close       = df['close']
    high        = df['high']
    low         = df['low']
    n           = len(df)
    out         = pd.Series(0, index=df.index)

    for i in range(lookback + 2, n):
        for j in range(i - lookback, i):
            if j < 2:
                continue
            gap_bottom = high.iloc[j]
            gap_top    = low.iloc[j - 2]
            if gap_top <= gap_bottom:
                continue
            gap_size = (gap_top - gap_bottom) / gap_bottom
            if gap_size < min_gap_pct:
                continue
            c = close.iloc[i]
            if gap_bottom <= c <= gap_top:
                out.iloc[i] = -1
                break
    return out


def space_FVG_Bearish(trial):
    return {
        'lookback':    trial.suggest_int('lookback',    3, 20),
        'min_gap_pct': trial.suggest_float('min_gap_pct', 0.001, 0.005),
    }


# ── 3. FVG_Bidir ──────────────────────────────────────────────────────────────

def gen_FVG_Bidir(df, lookback=10, min_gap_pct=0.002, **kw):
    """Both bullish and bearish FVG retest entries."""
    lookback    = int(lookback)
    min_gap_pct = float(min_gap_pct)
    close       = df['close']
    high        = df['high']
    low         = df['low']
    n           = len(df)
    out         = pd.Series(0, index=df.index)

    for i in range(lookback + 2, n):
        sig = 0
        for j in range(i - lookback, i):
            if j < 2:
                continue
            # Bullish FVG
            gap_top    = low.iloc[j]
            gap_bottom = high.iloc[j - 2]
            if gap_bottom < gap_top:
                gap_size = (gap_top - gap_bottom) / max(gap_bottom, 1e-9)
                if gap_size >= min_gap_pct:
                    c = close.iloc[i]
                    if gap_bottom <= c <= gap_top:
                        sig = 1
                        break
            # Bearish FVG
            gap_bottom2 = high.iloc[j]
            gap_top2    = low.iloc[j - 2]
            if gap_top2 > gap_bottom2:
                gap_size2 = (gap_top2 - gap_bottom2) / max(gap_bottom2, 1e-9)
                if gap_size2 >= min_gap_pct:
                    c = close.iloc[i]
                    if gap_bottom2 <= c <= gap_top2:
                        sig = -1
                        break
        out.iloc[i] = sig
    return out


def space_FVG_Bidir(trial):
    return {
        'lookback':    trial.suggest_int('lookback',    3, 20),
        'min_gap_pct': trial.suggest_float('min_gap_pct', 0.001, 0.005),
    }


# ── 4. FVG_EMA ────────────────────────────────────────────────────────────────

def gen_FVG_EMA(df, lookback=10, ema_p=50, min_gap_pct=0.002, **kw):
    """FVG + EMA trend filter: only trade FVGs in direction of trend."""
    lookback    = int(lookback)
    ema_p       = int(ema_p)
    min_gap_pct = float(min_gap_pct)
    close       = df['close']
    high        = df['high']
    low         = df['low']
    ema         = _ema(close, ema_p)
    n           = len(df)
    out         = pd.Series(0, index=df.index)

    for i in range(max(lookback + 2, ema_p), n):
        trend_up   = close.iloc[i] > ema.iloc[i]
        trend_down = close.iloc[i] < ema.iloc[i]
        for j in range(i - lookback, i):
            if j < 2:
                continue
            if trend_up:
                gap_top    = low.iloc[j]
                gap_bottom = high.iloc[j - 2]
                if gap_bottom < gap_top:
                    gap_size = (gap_top - gap_bottom) / max(gap_bottom, 1e-9)
                    if gap_size >= min_gap_pct and gap_bottom <= close.iloc[i] <= gap_top:
                        out.iloc[i] = 1
                        break
            elif trend_down:
                gap_bottom = high.iloc[j]
                gap_top    = low.iloc[j - 2]
                if gap_top > gap_bottom:
                    gap_size = (gap_top - gap_bottom) / max(gap_bottom, 1e-9)
                    if gap_size >= min_gap_pct and gap_bottom <= close.iloc[i] <= gap_top:
                        out.iloc[i] = -1
                        break
    return out


def space_FVG_EMA(trial):
    return {
        'lookback':    trial.suggest_int('lookback',    3, 20),
        'ema_p':       trial.suggest_int('ema_p',       20, 100),
        'min_gap_pct': trial.suggest_float('min_gap_pct', 0.001, 0.005),
    }


# ── 5. FVG_Volume ─────────────────────────────────────────────────────────────

def gen_FVG_Volume(df, lookback=10, vol_p=20, min_gap_pct=0.002, **kw):
    """Bullish FVG with volume confirmation (vol > vol_sma)."""
    lookback    = int(lookback)
    vol_p       = int(vol_p)
    min_gap_pct = float(min_gap_pct)
    close       = df['close']
    high        = df['high']
    low         = df['low']
    vol         = df['volume']
    vol_sma     = _sma(vol, vol_p)
    n           = len(df)
    out         = pd.Series(0, index=df.index)

    for i in range(max(lookback + 2, vol_p), n):
        vol_ok = vol.iloc[i] > vol_sma.iloc[i]
        if not vol_ok:
            continue
        for j in range(i - lookback, i):
            if j < 2:
                continue
            # Bullish FVG
            gap_top    = low.iloc[j]
            gap_bottom = high.iloc[j - 2]
            if gap_bottom < gap_top:
                gap_size = (gap_top - gap_bottom) / max(gap_bottom, 1e-9)
                if gap_size >= min_gap_pct and gap_bottom <= close.iloc[i] <= gap_top:
                    out.iloc[i] = 1
                    break
            # Bearish FVG
            gap_bottom2 = high.iloc[j]
            gap_top2    = low.iloc[j - 2]
            if gap_top2 > gap_bottom2:
                gap_size2 = (gap_top2 - gap_bottom2) / max(gap_bottom2, 1e-9)
                if gap_size2 >= min_gap_pct and gap_bottom2 <= close.iloc[i] <= gap_top2:
                    out.iloc[i] = -1
                    break
    return out


def space_FVG_Volume(trial):
    return {
        'lookback':    trial.suggest_int('lookback',    3, 20),
        'vol_p':       trial.suggest_int('vol_p',       20, 50),
        'min_gap_pct': trial.suggest_float('min_gap_pct', 0.001, 0.005),
    }


# ── 6. OB_Bullish ─────────────────────────────────────────────────────────────

def gen_OB_Bullish(df, atr_p=14, atr_mult=2.0, lookback=20, **kw):
    """Bullish OB: last bearish candle before strong bullish move.
    Long when price returns to OB zone."""
    atr_p    = int(atr_p)
    lookback = int(lookback)
    atr_mult = float(atr_mult)
    close    = df['close']
    open_    = df['open']
    high     = df['high']
    low      = df['low']
    atr      = _atr(df, atr_p)
    n        = len(df)
    out      = pd.Series(0, index=df.index)

    # Detect OB formation: bearish candle at j, then strong bullish move (close > open + atr_mult*atr)
    ob_top    = pd.Series(np.nan, index=df.index)
    ob_bottom = pd.Series(np.nan, index=df.index)

    for j in range(1, n - 1):
        is_bearish = close.iloc[j] < open_.iloc[j]
        if not is_bearish:
            continue
        # Check for strong bullish move in next bar
        move_up = close.iloc[j + 1] - open_.iloc[j + 1]
        if move_up >= atr_mult * atr.iloc[j + 1]:
            ob_top.iloc[j]    = high.iloc[j]
            ob_bottom.iloc[j] = low.iloc[j]

    for i in range(lookback + 2, n):
        for j in range(i - lookback, i):
            if pd.isna(ob_top.iloc[j]):
                continue
            c = close.iloc[i]
            if ob_bottom.iloc[j] <= c <= ob_top.iloc[j]:
                out.iloc[i] = 1
                break
    return out


def space_OB_Bullish(trial):
    return {
        'atr_p':    trial.suggest_int('atr_p',    10, 20),
        'atr_mult': trial.suggest_float('atr_mult', 1.5, 3.0),
        'lookback': trial.suggest_int('lookback', 10, 30),
    }


# ── 7. OB_Bearish ─────────────────────────────────────────────────────────────

def gen_OB_Bearish(df, atr_p=14, atr_mult=2.0, lookback=20, **kw):
    """Bearish OB: last bullish candle before strong bearish move.
    Short when price returns to OB zone."""
    atr_p    = int(atr_p)
    lookback = int(lookback)
    atr_mult = float(atr_mult)
    close    = df['close']
    open_    = df['open']
    high     = df['high']
    low      = df['low']
    atr      = _atr(df, atr_p)
    n        = len(df)
    out      = pd.Series(0, index=df.index)

    ob_top    = pd.Series(np.nan, index=df.index)
    ob_bottom = pd.Series(np.nan, index=df.index)

    for j in range(1, n - 1):
        is_bullish = close.iloc[j] > open_.iloc[j]
        if not is_bullish:
            continue
        move_down = open_.iloc[j + 1] - close.iloc[j + 1]
        if move_down >= atr_mult * atr.iloc[j + 1]:
            ob_top.iloc[j]    = high.iloc[j]
            ob_bottom.iloc[j] = low.iloc[j]

    for i in range(lookback + 2, n):
        for j in range(i - lookback, i):
            if pd.isna(ob_top.iloc[j]):
                continue
            c = close.iloc[i]
            if ob_bottom.iloc[j] <= c <= ob_top.iloc[j]:
                out.iloc[i] = -1
                break
    return out


def space_OB_Bearish(trial):
    return {
        'atr_p':    trial.suggest_int('atr_p',    10, 20),
        'atr_mult': trial.suggest_float('atr_mult', 1.5, 3.0),
        'lookback': trial.suggest_int('lookback', 10, 30),
    }


# ── 8. OB_FVG_Combo ───────────────────────────────────────────────────────────

def gen_OB_FVG_Combo(df, atr_p=14, atr_mult=2.0, gap_pct=0.002, lookback=20, **kw):
    """OB + FVG alignment: Long when bullish OB AND bullish FVG present nearby."""
    atr_p    = int(atr_p)
    lookback = int(lookback)
    atr_mult = float(atr_mult)
    gap_pct  = float(gap_pct)
    close    = df['close']
    open_    = df['open']
    high     = df['high']
    low      = df['low']
    atr      = _atr(df, atr_p)
    n        = len(df)
    out      = pd.Series(0, index=df.index)

    # Build OB zones
    bull_ob_top    = pd.Series(np.nan, index=df.index)
    bull_ob_bottom = pd.Series(np.nan, index=df.index)
    bear_ob_top    = pd.Series(np.nan, index=df.index)
    bear_ob_bottom = pd.Series(np.nan, index=df.index)

    for j in range(1, n - 1):
        if close.iloc[j] < open_.iloc[j]:
            move_up = close.iloc[j + 1] - open_.iloc[j + 1]
            if move_up >= atr_mult * atr.iloc[j + 1]:
                bull_ob_top.iloc[j]    = high.iloc[j]
                bull_ob_bottom.iloc[j] = low.iloc[j]
        else:
            move_down = open_.iloc[j + 1] - close.iloc[j + 1]
            if move_down >= atr_mult * atr.iloc[j + 1]:
                bear_ob_top.iloc[j]    = high.iloc[j]
                bear_ob_bottom.iloc[j] = low.iloc[j]

    for i in range(lookback + 2, n):
        c          = close.iloc[i]
        has_bull_ob = False
        has_bear_ob = False
        has_bull_fvg = False
        has_bear_fvg = False

        for j in range(i - lookback, i):
            if not pd.isna(bull_ob_top.iloc[j]):
                if bull_ob_bottom.iloc[j] <= c <= bull_ob_top.iloc[j]:
                    has_bull_ob = True
            if not pd.isna(bear_ob_top.iloc[j]):
                if bear_ob_bottom.iloc[j] <= c <= bear_ob_top.iloc[j]:
                    has_bear_ob = True
            if j >= 2:
                gt = low.iloc[j]
                gb = high.iloc[j - 2]
                if gb < gt and (gt - gb) / max(gb, 1e-9) >= gap_pct:
                    if gb <= c <= gt:
                        has_bull_fvg = True
                gb2 = high.iloc[j]
                gt2 = low.iloc[j - 2]
                if gt2 > gb2 and (gt2 - gb2) / max(gb2, 1e-9) >= gap_pct:
                    if gb2 <= c <= gt2:
                        has_bear_fvg = True

        if has_bull_ob and has_bull_fvg:
            out.iloc[i] = 1
        elif has_bear_ob and has_bear_fvg:
            out.iloc[i] = -1
    return out


def space_OB_FVG_Combo(trial):
    return {
        'atr_p':    trial.suggest_int('atr_p',    10, 20),
        'atr_mult': trial.suggest_float('atr_mult', 1.5, 3.0),
        'gap_pct':  trial.suggest_float('gap_pct',  0.001, 0.005),
        'lookback': trial.suggest_int('lookback', 10, 30),
    }


# ── 9. OB_ATR ─────────────────────────────────────────────────────────────────

def gen_OB_ATR(df, atr_p=14, ob_mult=1.5, retest_pct=0.005, lookback=20, **kw):
    """Order Block detected via ATR-sized moves; retest within retest_pct of OB zone."""
    atr_p      = int(atr_p)
    lookback   = int(lookback)
    ob_mult    = float(ob_mult)
    retest_pct = float(retest_pct)
    close      = df['close']
    open_      = df['open']
    high       = df['high']
    low        = df['low']
    atr        = _atr(df, atr_p)
    n          = len(df)
    out        = pd.Series(0, index=df.index)

    bull_ob = pd.Series(np.nan, index=df.index)
    bear_ob = pd.Series(np.nan, index=df.index)

    for j in range(1, n):
        candle_size = abs(close.iloc[j] - open_.iloc[j])
        if candle_size < ob_mult * atr.iloc[j]:
            continue
        if close.iloc[j] > open_.iloc[j]:
            # Strong bullish candle → mark previous bearish candle as bull OB
            if j >= 1 and close.iloc[j - 1] < open_.iloc[j - 1]:
                bull_ob.iloc[j - 1] = (high.iloc[j - 1] + low.iloc[j - 1]) / 2
        else:
            # Strong bearish candle → mark previous bullish candle as bear OB
            if j >= 1 and close.iloc[j - 1] > open_.iloc[j - 1]:
                bear_ob.iloc[j - 1] = (high.iloc[j - 1] + low.iloc[j - 1]) / 2

    for i in range(lookback + 2, n):
        c = close.iloc[i]
        for j in range(i - lookback, i):
            if not pd.isna(bull_ob.iloc[j]):
                zone_mid = bull_ob.iloc[j]
                if abs(c - zone_mid) / max(zone_mid, 1e-9) <= retest_pct:
                    out.iloc[i] = 1
                    break
            if not pd.isna(bear_ob.iloc[j]):
                zone_mid = bear_ob.iloc[j]
                if abs(c - zone_mid) / max(zone_mid, 1e-9) <= retest_pct:
                    out.iloc[i] = -1
                    break
    return out


def space_OB_ATR(trial):
    return {
        'atr_p':      trial.suggest_int('atr_p',      10, 20),
        'ob_mult':    trial.suggest_float('ob_mult',    1.0, 3.0),
        'retest_pct': trial.suggest_float('retest_pct', 0.001, 0.01),
        'lookback':   trial.suggest_int('lookback',   10, 30),
    }


# ── 10. OB_RSI ────────────────────────────────────────────────────────────────

def gen_OB_RSI(df, atr_p=14, atr_mult=2.0, rsi_p=14, lookback=20, **kw):
    """OB + RSI confirmation.
    Long: bullish OB retest AND RSI crossing up from below 50.
    Short: bearish OB retest AND RSI crossing down from above 50."""
    atr_p    = int(atr_p)
    lookback = int(lookback)
    atr_mult = float(atr_mult)
    rsi_p    = int(rsi_p)
    close    = df['close']
    open_    = df['open']
    high     = df['high']
    low      = df['low']
    atr      = _atr(df, atr_p)
    rsi      = _rsi(close, rsi_p)
    n        = len(df)
    out      = pd.Series(0, index=df.index)

    bull_ob_top    = pd.Series(np.nan, index=df.index)
    bull_ob_bottom = pd.Series(np.nan, index=df.index)
    bear_ob_top    = pd.Series(np.nan, index=df.index)
    bear_ob_bottom = pd.Series(np.nan, index=df.index)

    for j in range(1, n - 1):
        if close.iloc[j] < open_.iloc[j]:
            if close.iloc[j + 1] - open_.iloc[j + 1] >= atr_mult * atr.iloc[j + 1]:
                bull_ob_top.iloc[j]    = high.iloc[j]
                bull_ob_bottom.iloc[j] = low.iloc[j]
        else:
            if open_.iloc[j + 1] - close.iloc[j + 1] >= atr_mult * atr.iloc[j + 1]:
                bear_ob_top.iloc[j]    = high.iloc[j]
                bear_ob_bottom.iloc[j] = low.iloc[j]

    rsi_cross_up   = (rsi > 50) & (rsi.shift(1) <= 50)
    rsi_cross_down = (rsi < 50) & (rsi.shift(1) >= 50)

    for i in range(lookback + 2, n):
        c = close.iloc[i]
        for j in range(i - lookback, i):
            if not pd.isna(bull_ob_top.iloc[j]):
                if bull_ob_bottom.iloc[j] <= c <= bull_ob_top.iloc[j] and rsi_cross_up.iloc[i]:
                    out.iloc[i] = 1
                    break
            if not pd.isna(bear_ob_top.iloc[j]):
                if bear_ob_bottom.iloc[j] <= c <= bear_ob_top.iloc[j] and rsi_cross_down.iloc[i]:
                    out.iloc[i] = -1
                    break
    return out


def space_OB_RSI(trial):
    return {
        'atr_p':    trial.suggest_int('atr_p',    10, 20),
        'atr_mult': trial.suggest_float('atr_mult', 1.5, 3.0),
        'rsi_p':    trial.suggest_int('rsi_p',     7, 21),
        'lookback': trial.suggest_int('lookback', 10, 30),
    }


# ── 11. BOS_Strategy ──────────────────────────────────────────────────────────

def gen_BOS_Strategy(df, pivot_left=5, pivot_right=5, **kw):
    """Break of Structure: track pivot highs/lows and detect BOS.
    Bullish BOS: price breaks above last confirmed pivot high after making a LL.
    Bearish BOS: price breaks below last confirmed pivot low after making a HH."""
    pivot_left  = int(pivot_left)
    pivot_right = int(pivot_right)
    close       = df['close']
    high        = df['high']
    low         = df['low']
    n           = len(df)
    out         = pd.Series(0, index=df.index)

    ph = _pivot_high(high, pivot_left, pivot_right)
    pl = _pivot_low(low,   pivot_left, pivot_right)

    last_ph = np.nan
    last_pl = np.nan
    # Track structure: last_hh, last_ll
    last_hh = np.nan
    last_ll = np.nan
    structure = 0  # 1 = uptrend, -1 = downtrend

    for i in range(pivot_left + pivot_right, n):
        c = close.iloc[i]
        if ph.iloc[i]:
            last_ph = high.iloc[i]
        if pl.iloc[i]:
            last_pl = low.iloc[i]

        # Bullish BOS: close breaks above last_ph (previous swing high)
        if not np.isnan(last_ph) and c > last_ph and structure != 1:
            out.iloc[i] = 1
            structure   = 1
            last_hh     = last_ph
        # Bearish BOS: close breaks below last_pl (previous swing low)
        elif not np.isnan(last_pl) and c < last_pl and structure != -1:
            out.iloc[i] = -1
            structure   = -1
            last_ll     = last_pl
    return out


def space_BOS_Strategy(trial):
    return {
        'pivot_left':  trial.suggest_int('pivot_left',  3, 10),
        'pivot_right': trial.suggest_int('pivot_right', 3, 10),
    }


# ── 12. CHoCH_Strategy ────────────────────────────────────────────────────────

def gen_CHoCH_Strategy(df, pivot_left=5, pivot_right=5, lookback=20, **kw):
    """Change of Character: more conservative than BOS.
    Long: in downtrend, price breaks first swing high (CHoCH signal).
    Short: in uptrend, price breaks first swing low."""
    pivot_left  = int(pivot_left)
    pivot_right = int(pivot_right)
    lookback    = int(lookback)
    close       = df['close']
    high        = df['high']
    low         = df['low']
    n           = len(df)
    out         = pd.Series(0, index=df.index)

    ph = _pivot_high(high, pivot_left, pivot_right)
    pl = _pivot_low(low,   pivot_left, pivot_right)

    # Collect pivot timestamps
    ph_vals = []
    pl_vals = []
    trend   = 0  # 0=undefined, 1=up, -1=down

    for i in range(pivot_left + pivot_right, n):
        c = close.iloc[i]
        if ph.iloc[i]:
            ph_vals.append((i, high.iloc[i]))
        if pl.iloc[i]:
            pl_vals.append((i, low.iloc[i]))

        # Keep only pivots in lookback window
        ph_vals = [(idx, v) for (idx, v) in ph_vals if i - idx <= lookback]
        pl_vals = [(idx, v) for (idx, v) in pl_vals if i - idx <= lookback]

        if len(ph_vals) >= 2:
            # HH = uptrend
            if ph_vals[-1][1] > ph_vals[-2][1]:
                trend = 1
        if len(pl_vals) >= 2:
            # LL = downtrend
            if pl_vals[-1][1] < pl_vals[-2][1]:
                trend = -1

        # CHoCH long: in downtrend, break first swing high (lowest ph_val)
        if trend == -1 and len(ph_vals) > 0:
            first_ph = min(v for (_, v) in ph_vals)
            if c > first_ph:
                out.iloc[i] = 1
                trend = 0
        # CHoCH short: in uptrend, break first swing low
        elif trend == 1 and len(pl_vals) > 0:
            first_pl = max(v for (_, v) in pl_vals)
            if c < first_pl:
                out.iloc[i] = -1
                trend = 0
    return out


def space_CHoCH_Strategy(trial):
    return {
        'pivot_left':  trial.suggest_int('pivot_left',  3, 10),
        'pivot_right': trial.suggest_int('pivot_right', 3, 10),
        'lookback':    trial.suggest_int('lookback',   10, 30),
    }


# ── 13. BOS_CHoCH_Combo ───────────────────────────────────────────────────────

def gen_BOS_CHoCH_Combo(df, pivot_left=5, pivot_right=5, **kw):
    """Combined BOS + CHoCH. BOS = strong confirmation, CHoCH = early signal.
    Both produce long/short signals."""
    pivot_left  = int(pivot_left)
    pivot_right = int(pivot_right)
    close       = df['close']
    high        = df['high']
    low         = df['low']
    n           = len(df)
    out         = pd.Series(0, index=df.index)

    ph = _pivot_high(high, pivot_left, pivot_right)
    pl = _pivot_low(low,   pivot_left, pivot_right)

    last_ph    = np.nan
    last_pl    = np.nan
    prev_ph    = np.nan
    prev_pl    = np.nan
    structure  = 0

    for i in range(pivot_left + pivot_right, n):
        c = close.iloc[i]
        if ph.iloc[i]:
            prev_ph = last_ph
            last_ph = high.iloc[i]
        if pl.iloc[i]:
            prev_pl = last_pl
            last_pl = low.iloc[i]

        # BOS long
        if not np.isnan(last_ph) and c > last_ph and structure != 1:
            out.iloc[i] = 1
            structure   = 1
        # BOS short
        elif not np.isnan(last_pl) and c < last_pl and structure != -1:
            out.iloc[i] = -1
            structure   = -1
        # CHoCH long: downtrend breaking intermediate high
        elif structure == -1 and not np.isnan(prev_ph) and c > prev_ph:
            out.iloc[i] = 1
            structure   = 0
        # CHoCH short: uptrend breaking intermediate low
        elif structure == 1 and not np.isnan(prev_pl) and c < prev_pl:
            out.iloc[i] = -1
            structure   = 0
    return out


def space_BOS_CHoCH_Combo(trial):
    return {
        'pivot_left':  trial.suggest_int('pivot_left',  3, 10),
        'pivot_right': trial.suggest_int('pivot_right', 3, 10),
    }


# ── 14. BOS_EMA ───────────────────────────────────────────────────────────────

def gen_BOS_EMA(df, pivot_left=5, pivot_right=5, ema_p=100, **kw):
    """BOS + EMA trend alignment.
    Long: bullish BOS AND close > EMA. Short: bearish BOS AND close < EMA."""
    pivot_left  = int(pivot_left)
    pivot_right = int(pivot_right)
    ema_p       = int(ema_p)
    close       = df['close']
    high        = df['high']
    low         = df['low']
    ema         = _ema(close, ema_p)
    n           = len(df)
    out         = pd.Series(0, index=df.index)

    ph = _pivot_high(high, pivot_left, pivot_right)
    pl = _pivot_low(low,   pivot_left, pivot_right)

    last_ph   = np.nan
    last_pl   = np.nan
    structure = 0

    for i in range(max(pivot_left + pivot_right, ema_p), n):
        c      = close.iloc[i]
        e      = ema.iloc[i]
        if ph.iloc[i]:
            last_ph = high.iloc[i]
        if pl.iloc[i]:
            last_pl = low.iloc[i]

        if not np.isnan(last_ph) and c > last_ph and structure != 1 and c > e:
            out.iloc[i] = 1
            structure   = 1
        elif not np.isnan(last_pl) and c < last_pl and structure != -1 and c < e:
            out.iloc[i] = -1
            structure   = -1
    return out


def space_BOS_EMA(trial):
    return {
        'pivot_left':  trial.suggest_int('pivot_left',  3, 10),
        'pivot_right': trial.suggest_int('pivot_right', 3, 10),
        'ema_p':       trial.suggest_int('ema_p',       50, 200),
    }


# ── 15. BOS_Volume ────────────────────────────────────────────────────────────

def gen_BOS_Volume(df, pivot_left=5, pivot_right=5, vol_p=20, vol_mult=1.5, **kw):
    """BOS + volume surge confirmation.
    Long: bullish BOS AND volume > vol_mult * vol_sma."""
    pivot_left  = int(pivot_left)
    pivot_right = int(pivot_right)
    vol_p       = int(vol_p)
    vol_mult    = float(vol_mult)
    close       = df['close']
    high        = df['high']
    low         = df['low']
    vol         = df['volume']
    vol_sma     = _sma(vol, vol_p)
    n           = len(df)
    out         = pd.Series(0, index=df.index)

    ph = _pivot_high(high, pivot_left, pivot_right)
    pl = _pivot_low(low,   pivot_left, pivot_right)

    last_ph   = np.nan
    last_pl   = np.nan
    structure = 0

    for i in range(max(pivot_left + pivot_right, vol_p), n):
        c      = close.iloc[i]
        vol_ok = vol.iloc[i] > vol_mult * vol_sma.iloc[i]
        if ph.iloc[i]:
            last_ph = high.iloc[i]
        if pl.iloc[i]:
            last_pl = low.iloc[i]

        if not np.isnan(last_ph) and c > last_ph and structure != 1 and vol_ok:
            out.iloc[i] = 1
            structure   = 1
        elif not np.isnan(last_pl) and c < last_pl and structure != -1 and vol_ok:
            out.iloc[i] = -1
            structure   = -1
    return out


def space_BOS_Volume(trial):
    return {
        'pivot_left':  trial.suggest_int('pivot_left',  3, 10),
        'pivot_right': trial.suggest_int('pivot_right', 3, 10),
        'vol_p':       trial.suggest_int('vol_p',       20, 50),
        'vol_mult':    trial.suggest_float('vol_mult',   1.0, 3.0),
    }


# ── 16. Liquidity_Sweep_Bull ──────────────────────────────────────────────────

def gen_Liquidity_Sweep_Bull(df, swing_lookback=20, min_sweep_pct=0.002, **kw):
    """Bullish liquidity sweep: low sweeps below recent swing low then closes above it."""
    swing_lookback = int(swing_lookback)
    min_sweep_pct  = float(min_sweep_pct)
    close          = df['close']
    low            = df['low']
    n              = len(df)
    out            = pd.Series(0, index=df.index)

    for i in range(swing_lookback + 1, n):
        swing_low = low.iloc[i - swing_lookback: i].min()
        if low.iloc[i] < swing_low:
            sweep_depth = (swing_low - low.iloc[i]) / max(swing_low, 1e-9)
            if sweep_depth >= min_sweep_pct and close.iloc[i] > swing_low:
                out.iloc[i] = 1
    return out


def space_Liquidity_Sweep_Bull(trial):
    return {
        'swing_lookback': trial.suggest_int('swing_lookback', 10, 30),
        'min_sweep_pct':  trial.suggest_float('min_sweep_pct', 0.001, 0.005),
    }


# ── 17. Liquidity_Sweep_Bear ──────────────────────────────────────────────────

def gen_Liquidity_Sweep_Bear(df, swing_lookback=20, min_sweep_pct=0.002, **kw):
    """Bearish liquidity sweep: high sweeps above recent swing high then closes below it."""
    swing_lookback = int(swing_lookback)
    min_sweep_pct  = float(min_sweep_pct)
    close          = df['close']
    high           = df['high']
    n              = len(df)
    out            = pd.Series(0, index=df.index)

    for i in range(swing_lookback + 1, n):
        swing_high = high.iloc[i - swing_lookback: i].max()
        if high.iloc[i] > swing_high:
            sweep_height = (high.iloc[i] - swing_high) / max(swing_high, 1e-9)
            if sweep_height >= min_sweep_pct and close.iloc[i] < swing_high:
                out.iloc[i] = -1
    return out


def space_Liquidity_Sweep_Bear(trial):
    return {
        'swing_lookback': trial.suggest_int('swing_lookback', 10, 30),
        'min_sweep_pct':  trial.suggest_float('min_sweep_pct', 0.001, 0.005),
    }


# ── 18. Liquidity_Sweep_Bidir ─────────────────────────────────────────────────

def gen_Liquidity_Sweep_Bidir(df, swing_lookback=20, min_sweep_pct=0.002, **kw):
    """Both bullish and bearish liquidity sweeps."""
    swing_lookback = int(swing_lookback)
    min_sweep_pct  = float(min_sweep_pct)
    close          = df['close']
    high           = df['high']
    low            = df['low']
    n              = len(df)
    out            = pd.Series(0, index=df.index)

    for i in range(swing_lookback + 1, n):
        swing_low  = low.iloc[i - swing_lookback: i].min()
        swing_high = high.iloc[i - swing_lookback: i].max()
        c          = close.iloc[i]

        if low.iloc[i] < swing_low:
            depth = (swing_low - low.iloc[i]) / max(swing_low, 1e-9)
            if depth >= min_sweep_pct and c > swing_low:
                out.iloc[i] = 1
                continue
        if high.iloc[i] > swing_high:
            height = (high.iloc[i] - swing_high) / max(swing_high, 1e-9)
            if height >= min_sweep_pct and c < swing_high:
                out.iloc[i] = -1
    return out


def space_Liquidity_Sweep_Bidir(trial):
    return {
        'swing_lookback': trial.suggest_int('swing_lookback', 10, 30),
        'min_sweep_pct':  trial.suggest_float('min_sweep_pct', 0.001, 0.005),
    }


# ── 19. Liquidity_EMA ─────────────────────────────────────────────────────────

def gen_Liquidity_EMA(df, swing_lookback=20, min_sweep_pct=0.002, ema_p=100, **kw):
    """Liquidity sweep + EMA trend filter.
    Long: bull sweep AND close > EMA. Short: bear sweep AND close < EMA."""
    swing_lookback = int(swing_lookback)
    min_sweep_pct  = float(min_sweep_pct)
    ema_p          = int(ema_p)
    close          = df['close']
    high           = df['high']
    low            = df['low']
    ema            = _ema(close, ema_p)
    n              = len(df)
    out            = pd.Series(0, index=df.index)

    for i in range(max(swing_lookback + 1, ema_p), n):
        swing_low  = low.iloc[i - swing_lookback: i].min()
        swing_high = high.iloc[i - swing_lookback: i].max()
        c          = close.iloc[i]
        e          = ema.iloc[i]

        if low.iloc[i] < swing_low:
            depth = (swing_low - low.iloc[i]) / max(swing_low, 1e-9)
            if depth >= min_sweep_pct and c > swing_low and c > e:
                out.iloc[i] = 1
                continue
        if high.iloc[i] > swing_high:
            height = (high.iloc[i] - swing_high) / max(swing_high, 1e-9)
            if height >= min_sweep_pct and c < swing_high and c < e:
                out.iloc[i] = -1
    return out


def space_Liquidity_EMA(trial):
    return {
        'swing_lookback': trial.suggest_int('swing_lookback', 10, 30),
        'min_sweep_pct':  trial.suggest_float('min_sweep_pct', 0.001, 0.005),
        'ema_p':          trial.suggest_int('ema_p',           50, 200),
    }


# ── 20. Liquidity_RSI ─────────────────────────────────────────────────────────

def gen_Liquidity_RSI(df, swing_lookback=20, min_sweep_pct=0.002, rsi_p=14, **kw):
    """Liquidity sweep + RSI recovery.
    Long: bull sweep AND RSI turning up from < 40.
    Short: bear sweep AND RSI turning down from > 60."""
    swing_lookback = int(swing_lookback)
    min_sweep_pct  = float(min_sweep_pct)
    rsi_p          = int(rsi_p)
    close          = df['close']
    high           = df['high']
    low            = df['low']
    rsi            = _rsi(close, rsi_p)
    n              = len(df)
    out            = pd.Series(0, index=df.index)

    for i in range(max(swing_lookback + 1, rsi_p + 1), n):
        swing_low  = low.iloc[i - swing_lookback: i].min()
        swing_high = high.iloc[i - swing_lookback: i].max()
        c          = close.iloc[i]
        r          = rsi.iloc[i]
        r_prev     = rsi.iloc[i - 1]

        if low.iloc[i] < swing_low:
            depth = (swing_low - low.iloc[i]) / max(swing_low, 1e-9)
            if depth >= min_sweep_pct and c > swing_low and r > r_prev and r < 50:
                out.iloc[i] = 1
                continue
        if high.iloc[i] > swing_high:
            height = (high.iloc[i] - swing_high) / max(swing_high, 1e-9)
            if height >= min_sweep_pct and c < swing_high and r < r_prev and r > 50:
                out.iloc[i] = -1
    return out


def space_Liquidity_RSI(trial):
    return {
        'swing_lookback': trial.suggest_int('swing_lookback', 10, 30),
        'min_sweep_pct':  trial.suggest_float('min_sweep_pct', 0.001, 0.005),
        'rsi_p':          trial.suggest_int('rsi_p',           7, 21),
    }


# ── 21. Premium_Discount ──────────────────────────────────────────────────────

def gen_Premium_Discount(df, range_p=50, **kw):
    """Premium/Discount zones based on rolling range.
    Discount zone = lower 50% of range. Premium zone = upper 50%.
    Long: in discount with bullish candle. Short: in premium with bearish candle."""
    range_p = int(range_p)
    close   = df['close']
    open_   = df['open']
    high    = df['high']
    low     = df['low']
    rng_hi  = high.rolling(range_p, min_periods=range_p).max()
    rng_lo  = low.rolling(range_p,  min_periods=range_p).min()
    mid     = (rng_hi + rng_lo) / 2.0
    n       = len(df)
    out     = pd.Series(0, index=df.index)

    for i in range(range_p, n):
        c  = close.iloc[i]
        o  = open_.iloc[i]
        m  = mid.iloc[i]
        if pd.isna(m):
            continue
        bull_candle = c > o
        bear_candle = c < o
        if c < m and bull_candle:
            out.iloc[i] = 1
        elif c > m and bear_candle:
            out.iloc[i] = -1
    return out


def space_Premium_Discount(trial):
    return {'range_p': trial.suggest_int('range_p', 20, 100)}


# ── 22. Premium_Discount_RSI ──────────────────────────────────────────────────

def gen_Premium_Discount_RSI(df, range_p=50, rsi_p=14, **kw):
    """P/D zones + RSI. Long: in discount AND RSI < 40. Short: in premium AND RSI > 60."""
    range_p = int(range_p)
    rsi_p   = int(rsi_p)
    close   = df['close']
    high    = df['high']
    low     = df['low']
    rng_hi  = high.rolling(range_p, min_periods=range_p).max()
    rng_lo  = low.rolling(range_p,  min_periods=range_p).min()
    mid     = (rng_hi + rng_lo) / 2.0
    rsi     = _rsi(close, rsi_p)
    n       = len(df)
    out     = pd.Series(0, index=df.index)

    for i in range(max(range_p, rsi_p), n):
        c = close.iloc[i]
        m = mid.iloc[i]
        r = rsi.iloc[i]
        if pd.isna(m):
            continue
        if c < m and r < 40:
            out.iloc[i] = 1
        elif c > m and r > 60:
            out.iloc[i] = -1
    return out


def space_Premium_Discount_RSI(trial):
    return {
        'range_p': trial.suggest_int('range_p', 20, 100),
        'rsi_p':   trial.suggest_int('rsi_p',    7, 21),
    }


# ── 23. Wyckoff_Spring ────────────────────────────────────────────────────────

def gen_Wyckoff_Spring(df, sup_p=30, spring_pct=0.002, **kw):
    """Wyckoff Spring: price falls below support (rolling min) then reverses.
    Long: wick below support + close back above support."""
    sup_p      = int(sup_p)
    spring_pct = float(spring_pct)
    close      = df['close']
    low        = df['low']
    n          = len(df)
    out        = pd.Series(0, index=df.index)

    support = low.rolling(sup_p, min_periods=sup_p).min().shift(1)

    for i in range(sup_p + 1, n):
        sup = support.iloc[i]
        if pd.isna(sup):
            continue
        if low.iloc[i] < sup:
            depth = (sup - low.iloc[i]) / max(sup, 1e-9)
            if depth >= spring_pct and close.iloc[i] > sup:
                out.iloc[i] = 1
    return out


def space_Wyckoff_Spring(trial):
    return {
        'sup_p':      trial.suggest_int('sup_p',      20, 50),
        'spring_pct': trial.suggest_float('spring_pct', 0.001, 0.005),
    }


# ── 24. Wyckoff_Upthrust ──────────────────────────────────────────────────────

def gen_Wyckoff_Upthrust(df, res_p=30, thrust_pct=0.002, **kw):
    """Wyckoff Upthrust: price rises above resistance then falls back.
    Short: wick above resistance + close back below resistance."""
    res_p      = int(res_p)
    thrust_pct = float(thrust_pct)
    close      = df['close']
    high       = df['high']
    n          = len(df)
    out        = pd.Series(0, index=df.index)

    resistance = high.rolling(res_p, min_periods=res_p).max().shift(1)

    for i in range(res_p + 1, n):
        res = resistance.iloc[i]
        if pd.isna(res):
            continue
        if high.iloc[i] > res:
            height = (high.iloc[i] - res) / max(res, 1e-9)
            if height >= thrust_pct and close.iloc[i] < res:
                out.iloc[i] = -1
    return out


def space_Wyckoff_Upthrust(trial):
    return {
        'res_p':      trial.suggest_int('res_p',      20, 50),
        'thrust_pct': trial.suggest_float('thrust_pct', 0.001, 0.005),
    }


# ── 25. Wyckoff_Bidir ─────────────────────────────────────────────────────────

def gen_Wyckoff_Bidir(df, zone_p=30, swing_pct=0.002, **kw):
    """Both Wyckoff Spring (long) and Upthrust (short)."""
    zone_p    = int(zone_p)
    swing_pct = float(swing_pct)
    close     = df['close']
    high      = df['high']
    low       = df['low']
    n         = len(df)
    out       = pd.Series(0, index=df.index)

    support    = low.rolling(zone_p,  min_periods=zone_p).min().shift(1)
    resistance = high.rolling(zone_p, min_periods=zone_p).max().shift(1)

    for i in range(zone_p + 1, n):
        sup = support.iloc[i]
        res = resistance.iloc[i]
        c   = close.iloc[i]
        if pd.isna(sup) or pd.isna(res):
            continue
        # Spring
        if low.iloc[i] < sup:
            depth = (sup - low.iloc[i]) / max(sup, 1e-9)
            if depth >= swing_pct and c > sup:
                out.iloc[i] = 1
                continue
        # Upthrust
        if high.iloc[i] > res:
            height = (high.iloc[i] - res) / max(res, 1e-9)
            if height >= swing_pct and c < res:
                out.iloc[i] = -1
    return out


def space_Wyckoff_Bidir(trial):
    return {
        'zone_p':    trial.suggest_int('zone_p',    20, 50),
        'swing_pct': trial.suggest_float('swing_pct', 0.001, 0.005),
    }


# ── 26. Inside_Bar_Breakout ───────────────────────────────────────────────────

def gen_Inside_Bar_Breakout(df, consec_ib=1, vol_mult=1.5, **kw):
    """Inside bar breakout.
    Inside bar: high < prev high AND low > prev low.
    Signal: next bar breaks above mother bar high (long) or below mother bar low (short).
    Optional: require volume > vol_mult * avg volume on breakout."""
    consec_ib = int(consec_ib)
    vol_mult  = float(vol_mult)
    close     = df['close']
    high      = df['high']
    low       = df['low']
    vol       = df['volume']
    vol_sma   = _sma(vol, 20)
    n         = len(df)
    out       = pd.Series(0, index=df.index)

    for i in range(consec_ib + 1, n):
        # Check consec_ib consecutive inside bars ending at i-1
        is_ib_run = True
        for k in range(consec_ib):
            bar   = i - 1 - k
            prev  = bar - 1
            if prev < 0:
                is_ib_run = False
                break
            if not (high.iloc[bar] < high.iloc[prev] and low.iloc[bar] > low.iloc[prev]):
                is_ib_run = False
                break
        if not is_ib_run:
            continue

        # Mother bar = bar just before the IB run
        mother_idx = i - 1 - consec_ib
        if mother_idx < 0:
            continue
        mother_high = high.iloc[mother_idx]
        mother_low  = low.iloc[mother_idx]
        c           = close.iloc[i]
        vol_ok      = vol.iloc[i] > vol_mult * vol_sma.iloc[i] if not pd.isna(vol_sma.iloc[i]) else True

        if c > mother_high and vol_ok:
            out.iloc[i] = 1
        elif c < mother_low and vol_ok:
            out.iloc[i] = -1
    return out


def space_Inside_Bar_Breakout(trial):
    return {
        'consec_ib': trial.suggest_int('consec_ib', 1, 3),
        'vol_mult':  trial.suggest_float('vol_mult',  1.0, 2.5),
    }


# ── 27. Inside_Bar_EMA ────────────────────────────────────────────────────────

def gen_Inside_Bar_EMA(df, ema_p=50, vol_mult=1.5, **kw):
    """Inside bar breakout filtered by EMA direction.
    Long: IB breakout up AND close > EMA. Short: IB breakout down AND close < EMA."""
    ema_p    = int(ema_p)
    vol_mult = float(vol_mult)
    close    = df['close']
    high     = df['high']
    low      = df['low']
    vol      = df['volume']
    ema      = _ema(close, ema_p)
    vol_sma  = _sma(vol, 20)
    n        = len(df)
    out      = pd.Series(0, index=df.index)

    for i in range(ema_p + 1, n):
        prev = i - 1
        if prev < 1:
            continue
        is_ib = high.iloc[prev] < high.iloc[prev - 1] and low.iloc[prev] > low.iloc[prev - 1]
        if not is_ib:
            continue
        mother_high = high.iloc[prev - 1]
        mother_low  = low.iloc[prev - 1]
        c           = close.iloc[i]
        e           = ema.iloc[i]
        vol_ok      = vol.iloc[i] > vol_mult * vol_sma.iloc[i] if not pd.isna(vol_sma.iloc[i]) else True

        if c > mother_high and c > e and vol_ok:
            out.iloc[i] = 1
        elif c < mother_low and c < e and vol_ok:
            out.iloc[i] = -1
    return out


def space_Inside_Bar_EMA(trial):
    return {
        'ema_p':    trial.suggest_int('ema_p',    20, 100),
        'vol_mult': trial.suggest_float('vol_mult', 1.0, 2.5),
    }


# ── 28. NR7_Breakout ──────────────────────────────────────────────────────────

def gen_NR7_Breakout(df, nr_period=7, vol_mult=1.5, **kw):
    """NR7: narrowest range of last nr_period bars.
    Long: close > NR7 high. Short: close < NR7 low.
    Volume confirmation: vol > vol_mult * vol_sma."""
    nr_period = int(nr_period)
    vol_mult  = float(vol_mult)
    close     = df['close']
    high      = df['high']
    low       = df['low']
    vol       = df['volume']
    vol_sma   = _sma(vol, 20)
    bar_range = high - low
    n         = len(df)
    out       = pd.Series(0, index=df.index)

    for i in range(nr_period, n):
        # Check if current bar has narrowest range in last nr_period bars
        window_ranges = bar_range.iloc[i - nr_period + 1: i + 1]
        if bar_range.iloc[i] != window_ranges.min():
            continue
        # NR7 found at bar i; signal on bar i+1
        if i + 1 >= n:
            continue
        nr_high = high.iloc[i]
        nr_low  = low.iloc[i]
        c_next  = close.iloc[i + 1]
        vol_ok  = vol.iloc[i + 1] > vol_mult * vol_sma.iloc[i + 1] if not pd.isna(vol_sma.iloc[i + 1]) else True

        if c_next > nr_high and vol_ok:
            out.iloc[i + 1] = 1
        elif c_next < nr_low and vol_ok:
            out.iloc[i + 1] = -1
    return out


def space_NR7_Breakout(trial):
    return {
        'nr_period': trial.suggest_int('nr_period', 5, 10),
        'vol_mult':  trial.suggest_float('vol_mult',  1.0, 3.0),
    }


# ── 29. EQH_EQL_Sweep ────────────────────────────────────────────────────────

def gen_EQH_EQL_Sweep(df, tolerance=0.002, lookback=20, **kw):
    """Equal Highs/Lows liquidity pools.
    EQH: two recent highs within tolerance => liquidity pool above.
    Sweep above EQH then reversal = short.
    EQL: same for lows => sweep below then reversal = long."""
    tolerance = float(tolerance)
    lookback  = int(lookback)
    close     = df['close']
    high      = df['high']
    low       = df['low']
    n         = len(df)
    out       = pd.Series(0, index=df.index)

    for i in range(lookback + 1, n):
        c           = close.iloc[i]
        window_high = high.iloc[i - lookback: i]
        window_low  = low.iloc[i - lookback: i]

        # EQH: find two highs within tolerance of each other
        h_max   = window_high.max()
        h_vals  = window_high[abs(window_high - h_max) / max(h_max, 1e-9) <= tolerance]
        if len(h_vals) >= 2:
            eqh_level = h_vals.mean()
            if high.iloc[i] > eqh_level and c < eqh_level:
                out.iloc[i] = -1
                continue

        # EQL: find two lows within tolerance of each other
        l_min   = window_low.min()
        l_vals  = window_low[abs(window_low - l_min) / max(l_min, 1e-9) <= tolerance]
        if len(l_vals) >= 2:
            eql_level = l_vals.mean()
            if low.iloc[i] < eql_level and c > eql_level:
                out.iloc[i] = 1
    return out


def space_EQH_EQL_Sweep(trial):
    return {
        'tolerance': trial.suggest_float('tolerance', 0.001, 0.005),
        'lookback':  trial.suggest_int('lookback',   10, 30),
    }


# ── 30. SMC_Full ──────────────────────────────────────────────────────────────

def gen_SMC_Full(df, pivot_left=5, pivot_right=5, atr_p=14, gap_pct=0.002,
                 atr_mult=2.0, lookback=20, **kw):
    """Full SMC confluence: FVG + OB + BOS all aligned.
    Long: bullish BOS + bullish OB retest + bullish FVG retest (any two of three).
    Short: bearish BOS + bearish OB retest + bearish FVG retest (any two of three)."""
    pivot_left  = int(pivot_left)
    pivot_right = int(pivot_right)
    atr_p       = int(atr_p)
    lookback    = int(lookback)
    gap_pct     = float(gap_pct)
    atr_mult    = float(atr_mult)
    close       = df['close']
    open_       = df['open']
    high        = df['high']
    low         = df['low']
    atr         = _atr(df, atr_p)
    n           = len(df)
    out         = pd.Series(0, index=df.index)

    # BOS tracking
    ph = _pivot_high(high, pivot_left, pivot_right)
    pl = _pivot_low(low,   pivot_left, pivot_right)

    last_ph        = np.nan
    last_pl        = np.nan
    bos_bull_bars  = set()
    bos_bear_bars  = set()
    structure      = 0

    for i in range(pivot_left + pivot_right, n):
        if ph.iloc[i]:
            last_ph = high.iloc[i]
        if pl.iloc[i]:
            last_pl = low.iloc[i]
        c = close.iloc[i]
        if not np.isnan(last_ph) and c > last_ph and structure != 1:
            bos_bull_bars.add(i)
            structure = 1
        elif not np.isnan(last_pl) and c < last_pl and structure != -1:
            bos_bear_bars.add(i)
            structure = -1

    # OB zones
    bull_ob_top    = pd.Series(np.nan, index=df.index)
    bull_ob_bottom = pd.Series(np.nan, index=df.index)
    bear_ob_top    = pd.Series(np.nan, index=df.index)
    bear_ob_bottom = pd.Series(np.nan, index=df.index)

    for j in range(1, n - 1):
        if close.iloc[j] < open_.iloc[j]:
            if close.iloc[j + 1] - open_.iloc[j + 1] >= atr_mult * atr.iloc[j + 1]:
                bull_ob_top.iloc[j]    = high.iloc[j]
                bull_ob_bottom.iloc[j] = low.iloc[j]
        else:
            if open_.iloc[j + 1] - close.iloc[j + 1] >= atr_mult * atr.iloc[j + 1]:
                bear_ob_top.iloc[j]    = high.iloc[j]
                bear_ob_bottom.iloc[j] = low.iloc[j]

    for i in range(max(pivot_left + pivot_right, lookback + 2, atr_p), n):
        c         = close.iloc[i]
        score_l   = 0
        score_s   = 0

        # BOS signal in recent lookback
        for b in range(i - lookback, i + 1):
            if b in bos_bull_bars:
                score_l += 1
            if b in bos_bear_bars:
                score_s += 1

        # OB + FVG
        for j in range(i - lookback, i):
            if j < 2:
                continue
            if not pd.isna(bull_ob_top.iloc[j]) and bull_ob_bottom.iloc[j] <= c <= bull_ob_top.iloc[j]:
                score_l += 1
            if not pd.isna(bear_ob_top.iloc[j]) and bear_ob_bottom.iloc[j] <= c <= bear_ob_top.iloc[j]:
                score_s += 1
            # Bull FVG
            gt = low.iloc[j]
            gb = high.iloc[j - 2]
            if gb < gt and (gt - gb) / max(gb, 1e-9) >= gap_pct and gb <= c <= gt:
                score_l += 1
            # Bear FVG
            gb2 = high.iloc[j]
            gt2 = low.iloc[j - 2]
            if gt2 > gb2 and (gt2 - gb2) / max(gb2, 1e-9) >= gap_pct and gb2 <= c <= gt2:
                score_s += 1

        if score_l >= 2 and score_l > score_s:
            out.iloc[i] = 1
        elif score_s >= 2 and score_s > score_l:
            out.iloc[i] = -1
    return out


def space_SMC_Full(trial):
    return {
        'pivot_left':  trial.suggest_int('pivot_left',  3, 10),
        'pivot_right': trial.suggest_int('pivot_right', 3, 10),
        'atr_p':       trial.suggest_int('atr_p',       10, 20),
        'gap_pct':     trial.suggest_float('gap_pct',    0.001, 0.005),
        'atr_mult':    trial.suggest_float('atr_mult',   1.5, 3.0),
        'lookback':    trial.suggest_int('lookback',    10, 30),
    }


# ── STRATEGY_EXPORT ───────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'FVG_Bullish': {
        'gen':            gen_FVG_Bullish,
        'space':          space_FVG_Bullish,
        'default_params': {'lookback': 10, 'min_gap_pct': 0.002},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       5000,
            'description': 'Bullish Fair Value Gap: detects gap between candle[i-2].high and candle[i].low, enters long on retest.',
        },
    },
    'FVG_Bearish': {
        'gen':            gen_FVG_Bearish,
        'space':          space_FVG_Bearish,
        'default_params': {'lookback': 10, 'min_gap_pct': 0.002},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       5000,
            'description': 'Bearish Fair Value Gap: detects gap between candle[i-2].low and candle[i].high, enters short on retest.',
        },
    },
    'FVG_Bidir': {
        'gen':            gen_FVG_Bidir,
        'space':          space_FVG_Bidir,
        'default_params': {'lookback': 10, 'min_gap_pct': 0.002},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       3000,
            'description': 'Bidirectional Fair Value Gap: both bullish and bearish FVG retest entries.',
        },
    },
    'FVG_EMA': {
        'gen':            gen_FVG_EMA,
        'space':          space_FVG_EMA,
        'default_params': {'lookback': 10, 'ema_p': 50, 'min_gap_pct': 0.002},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       2000,
            'description': 'Fair Value Gap with EMA trend filter: only trade FVGs aligned with trend direction.',
        },
    },
    'FVG_Volume': {
        'gen':            gen_FVG_Volume,
        'space':          space_FVG_Volume,
        'default_params': {'lookback': 10, 'vol_p': 20, 'min_gap_pct': 0.002},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       1500,
            'description': 'Fair Value Gap with volume confirmation: FVG retest only when volume exceeds average.',
        },
    },
    'OB_Bullish': {
        'gen':            gen_OB_Bullish,
        'space':          space_OB_Bullish,
        'default_params': {'atr_p': 14, 'atr_mult': 2.0, 'lookback': 20},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       4000,
            'description': 'Bullish Order Block: last bearish candle before strong bullish impulse. Long on retest of OB zone.',
        },
    },
    'OB_Bearish': {
        'gen':            gen_OB_Bearish,
        'space':          space_OB_Bearish,
        'default_params': {'atr_p': 14, 'atr_mult': 2.0, 'lookback': 20},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       4000,
            'description': 'Bearish Order Block: last bullish candle before strong bearish impulse. Short on retest of OB zone.',
        },
    },
    'OB_FVG_Combo': {
        'gen':            gen_OB_FVG_Combo,
        'space':          space_OB_FVG_Combo,
        'default_params': {'atr_p': 14, 'atr_mult': 2.0, 'gap_pct': 0.002, 'lookback': 20},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       3000,
            'description': 'OB + FVG confluence: high-probability entry when both bullish OB and bullish FVG align at same price level.',
        },
    },
    'OB_ATR': {
        'gen':            gen_OB_ATR,
        'space':          space_OB_ATR,
        'default_params': {'atr_p': 14, 'ob_mult': 1.5, 'retest_pct': 0.005, 'lookback': 20},
        'info': {
            'source':      'TradingView',
            'version':     6,
            'likes':       2000,
            'description': 'Order Block detected via ATR-sized impulse candles with ATR-based retest tolerance.',
        },
    },
    'OB_RSI': {
        'gen':            gen_OB_RSI,
        'space':          space_OB_RSI,
        'default_params': {'atr_p': 14, 'atr_mult': 2.0, 'rsi_p': 14, 'lookback': 20},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       1500,
            'description': 'Order Block + RSI confirmation: OB retest confirmed by RSI momentum cross through 50.',
        },
    },
    'BOS_Strategy': {
        'gen':            gen_BOS_Strategy,
        'space':          space_BOS_Strategy,
        'default_params': {'pivot_left': 5, 'pivot_right': 5},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       4500,
            'description': 'Break of Structure: detects pivot highs/lows and signals when price breaks through the last confirmed swing.',
        },
    },
    'CHoCH_Strategy': {
        'gen':            gen_CHoCH_Strategy,
        'space':          space_CHoCH_Strategy,
        'default_params': {'pivot_left': 5, 'pivot_right': 5, 'lookback': 20},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       3500,
            'description': 'Change of Character: conservative trend-change signal when price breaks opposite swing in confirmed trend.',
        },
    },
    'BOS_CHoCH_Combo': {
        'gen':            gen_BOS_CHoCH_Combo,
        'space':          space_BOS_CHoCH_Combo,
        'default_params': {'pivot_left': 5, 'pivot_right': 5},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       2500,
            'description': 'Combined BOS + CHoCH: strong BOS for confirmation, early CHoCH for anticipation.',
        },
    },
    'BOS_EMA': {
        'gen':            gen_BOS_EMA,
        'space':          space_BOS_EMA,
        'default_params': {'pivot_left': 5, 'pivot_right': 5, 'ema_p': 100},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       2000,
            'description': 'BOS with EMA trend alignment: only take BOS signals in the direction of the EMA trend.',
        },
    },
    'BOS_Volume': {
        'gen':            gen_BOS_Volume,
        'space':          space_BOS_Volume,
        'default_params': {'pivot_left': 5, 'pivot_right': 5, 'vol_p': 20, 'vol_mult': 1.5},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       1500,
            'description': 'BOS + volume surge: only accept BOS when volume confirms the breakout.',
        },
    },
    'Liquidity_Sweep_Bull': {
        'gen':            gen_Liquidity_Sweep_Bull,
        'space':          space_Liquidity_Sweep_Bull,
        'default_params': {'swing_lookback': 20, 'min_sweep_pct': 0.002},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       3000,
            'description': 'Bullish liquidity sweep: price wicks below recent swing low then closes back above it (stop-hunt reversal).',
        },
    },
    'Liquidity_Sweep_Bear': {
        'gen':            gen_Liquidity_Sweep_Bear,
        'space':          space_Liquidity_Sweep_Bear,
        'default_params': {'swing_lookback': 20, 'min_sweep_pct': 0.002},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       3000,
            'description': 'Bearish liquidity sweep: price wicks above recent swing high then closes back below it.',
        },
    },
    'Liquidity_Sweep_Bidir': {
        'gen':            gen_Liquidity_Sweep_Bidir,
        'space':          space_Liquidity_Sweep_Bidir,
        'default_params': {'swing_lookback': 20, 'min_sweep_pct': 0.002},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       2500,
            'description': 'Bidirectional liquidity sweep: both bullish and bearish stop-hunt reversal patterns.',
        },
    },
    'Liquidity_EMA': {
        'gen':            gen_Liquidity_EMA,
        'space':          space_Liquidity_EMA,
        'default_params': {'swing_lookback': 20, 'min_sweep_pct': 0.002, 'ema_p': 100},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       1500,
            'description': 'Liquidity sweep filtered by EMA: only take sweep signals aligned with EMA trend direction.',
        },
    },
    'Liquidity_RSI': {
        'gen':            gen_Liquidity_RSI,
        'space':          space_Liquidity_RSI,
        'default_params': {'swing_lookback': 20, 'min_sweep_pct': 0.002, 'rsi_p': 14},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       1200,
            'description': 'Liquidity sweep + RSI momentum recovery: sweep confirmed by RSI turning in reversal direction.',
        },
    },
    'Premium_Discount': {
        'gen':            gen_Premium_Discount,
        'space':          space_Premium_Discount,
        'default_params': {'range_p': 50},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       2500,
            'description': 'Premium/Discount zones: long in discount (lower 50% of range) with bullish candle, short in premium with bearish candle.',
        },
    },
    'Premium_Discount_RSI': {
        'gen':            gen_Premium_Discount_RSI,
        'space':          space_Premium_Discount_RSI,
        'default_params': {'range_p': 50, 'rsi_p': 14},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       1500,
            'description': 'P/D zones + RSI: long in discount when RSI < 40, short in premium when RSI > 60.',
        },
    },
    'Wyckoff_Spring': {
        'gen':            gen_Wyckoff_Spring,
        'space':          space_Wyckoff_Spring,
        'default_params': {'sup_p': 30, 'spring_pct': 0.002},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       2000,
            'description': 'Wyckoff Spring: price breaks below rolling support with wick then reverses and closes above support.',
        },
    },
    'Wyckoff_Upthrust': {
        'gen':            gen_Wyckoff_Upthrust,
        'space':          space_Wyckoff_Upthrust,
        'default_params': {'res_p': 30, 'thrust_pct': 0.002},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       2000,
            'description': 'Wyckoff Upthrust: price breaks above rolling resistance with wick then reverses and closes below resistance.',
        },
    },
    'Wyckoff_Bidir': {
        'gen':            gen_Wyckoff_Bidir,
        'space':          space_Wyckoff_Bidir,
        'default_params': {'zone_p': 30, 'swing_pct': 0.002},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       1500,
            'description': 'Bidirectional Wyckoff: both Spring (long) and Upthrust (short) patterns.',
        },
    },
    'Inside_Bar_Breakout': {
        'gen':            gen_Inside_Bar_Breakout,
        'space':          space_Inside_Bar_Breakout,
        'default_params': {'consec_ib': 1, 'vol_mult': 1.5},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       3000,
            'description': 'Inside bar breakout: breakout above mother bar high (long) or below mother bar low (short) with volume confirmation.',
        },
    },
    'Inside_Bar_EMA': {
        'gen':            gen_Inside_Bar_EMA,
        'space':          space_Inside_Bar_EMA,
        'default_params': {'ema_p': 50, 'vol_mult': 1.5},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       1800,
            'description': 'Inside bar breakout filtered by EMA: only trade IB breakouts in the direction of EMA trend.',
        },
    },
    'NR7_Breakout': {
        'gen':            gen_NR7_Breakout,
        'space':          space_NR7_Breakout,
        'default_params': {'nr_period': 7, 'vol_mult': 1.5},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       2000,
            'description': 'NR7 Breakout: narrowest range of last 7 bars signals compression; breakout entry with volume filter.',
        },
    },
    'EQH_EQL_Sweep': {
        'gen':            gen_EQH_EQL_Sweep,
        'space':          space_EQH_EQL_Sweep,
        'default_params': {'tolerance': 0.002, 'lookback': 20},
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       2000,
            'description': 'Equal Highs/Lows sweep: detects liquidity pools at equal highs/lows, trades the false breakout reversal.',
        },
    },
    'SMC_Full': {
        'gen':            gen_SMC_Full,
        'space':          space_SMC_Full,
        'default_params': {
            'pivot_left':  5,
            'pivot_right': 5,
            'atr_p':       14,
            'gap_pct':     0.002,
            'atr_mult':    2.0,
            'lookback':    20,
        },
        'info': {
            'source':      'TradingView',
            'version':     5,
            'likes':       3500,
            'description': 'Full SMC confluence: enters only when at least 2 of 3 SMC factors align (FVG + OB + BOS) for maximum probability.',
        },
    },
}
