#!/usr/bin/env python3
"""TV2 BATCH 18 — 30 estrategias Candlestick + Harmonic + Price Action 2026-04-01

  Doji_Reversal            — Doji at trend extremes with EMA filter (v4, ~3000L)
  Engulfing_Pattern        — Bull/bear engulfing with trend + volume (v4, ~5000L)
  Morning_Evening_Star     — Three-candle reversal pattern (v4, ~2500L)
  Three_White_Soldiers     — Three consecutive directional candles (v4, ~2000L)
  Marubozu                 — Full-body no-wick candle (v4, ~2000L)
  Tweezer_Tops_Bottoms     — Twin high/low reversal (v5, ~1800L)
  Three_Inside_UpDown      — Harami confirmation pattern (v4, ~1500L)
  Piercing_DarkCloud       — 50%+ penetration reversal (v4, ~1500L)
  Spinning_Top_Reversal    — Small body at trend extremes (v4, ~1200L)
  Harami                   — Inside candle reversal (v4, ~1500L)
  Pin_Bar_Advanced         — Wick pin bar with volume filter (v5, ~4000L)
  Two_Bar_Reversal         — Large opposite two-candle reversal (v4, ~2000L)
  Outside_Bar              — Outside bar direction from close (v4, ~2000L)
  Kangaroo_Tail            — Extended lower shadow spike (v5, ~2500L)
  Fakey_Pattern            — Inside bar false breakout reverse (v5, ~2000L)
  ABCD_Pattern             — ABCD Fibonacci swing pattern (v5, ~3000L)
  Gartley_Pattern          — Gartley 222 harmonic (v5, ~2500L)
  Butterfly_Pattern        — Butterfly harmonic (v5, ~2000L)
  Bat_Pattern              — Bat harmonic (v5, ~1800L)
  Crab_Pattern             — Crab extreme harmonic (v5, ~1500L)
  Three_Line_Strike_Bull   — Three bulls then engulfing bear (counter-trend) (v4, ~2000L)
  Three_Line_Strike_Bear   — Three bears then engulfing bull (counter-trend) (v4, ~2000L)
  Pivot_Daily              — Classical pivot PP/R1/S1 rolling approx (v5, ~3000L)
  Camarilla_Pivots         — Camarilla R3/S3 breakout (v5, ~2000L)
  Woodie_Pivots            — Woodie pivot S1/R1 bounce (v4, ~1800L)
  Bull_Flag                — Pole + tight consolidation breakout (v5, ~3000L)
  Bear_Flag                — Pole down + consolidation breakdown (v5, ~3000L)
  Pennant                  — Converging range continuation (v5, ~2000L)
  Gap_Fill                 — Gap then fill direction (v4, ~2500L)
  Gap_Continuation         — Breakaway gap in trend (v4, ~2000L)
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
    d  = s.diff()
    g  = d.clip(lower=0)
    l  = (-d).clip(lower=0)
    rs = _rma(g, p) / _rma(l, p).replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def _body(df):
    return (df['close'] - df['open']).abs()


def _upper_wick(df):
    return df['high'] - df[['close', 'open']].max(axis=1)


def _lower_wick(df):
    return df[['close', 'open']].min(axis=1) - df['low']


def _candle_range(df):
    return (df['high'] - df['low']).replace(0, 1e-9)


def _pivot_high(s, left, right):
    """Rolling pivot high: True where s[i] is the highest over [i-left .. i+right]."""
    result = pd.Series(False, index=s.index)
    for i in range(left, len(s) - right):
        window = s.iloc[i - left: i + right + 1]
        if s.iloc[i] == window.max():
            result.iloc[i] = True
    return result


def _pivot_low(s, left, right):
    """Rolling pivot low: True where s[i] is the lowest over [i-left .. i+right]."""
    result = pd.Series(False, index=s.index)
    for i in range(left, len(s) - right):
        window = s.iloc[i - left: i + right + 1]
        if s.iloc[i] == window.min():
            result.iloc[i] = True
    return result


# ── 1. Doji_Reversal ─────────────────────────────────────────────────────────

def gen_Doji_Reversal(df, doji_thresh=0.1, ema_p=20, atr_p=14, **kw):
    doji_thresh = float(doji_thresh)
    ema_p       = int(ema_p)
    atr_p       = int(atr_p)
    atr         = _atr(df, atr_p)
    ema         = _ema(df['close'], ema_p)
    body        = _body(df)
    is_doji     = body < atr * doji_thresh
    ema_down    = ema < ema.shift(1)
    ema_up      = ema > ema.shift(1)
    long        = is_doji & ema_down & (df['close'] < ema)
    short       = is_doji & ema_up   & (df['close'] > ema)
    out         = pd.Series(0, index=df.index)
    out[long]   =  1
    out[short]  = -1
    return out.fillna(0).astype(int)


def space_Doji_Reversal(trial):
    return {
        'doji_thresh': trial.suggest_float('doji_thresh', 0.1, 0.3),
        'ema_p':       trial.suggest_int('ema_p', 14, 50),
        'atr_p':       trial.suggest_int('atr_p', 10, 20),
    }


# ── 2. Engulfing_Pattern ─────────────────────────────────────────────────────

def gen_Engulfing_Pattern(df, ema_p=20, vol_mult=1.2, **kw):
    ema_p    = int(ema_p)
    vol_mult = float(vol_mult)
    ema      = _ema(df['close'], ema_p)
    body     = _body(df)
    bull_c   = df['close'] > df['open']
    bear_c   = df['close'] < df['open']
    vol_avg  = _sma(df['volume'], 20) if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    vol_ok   = (df['volume'] >= vol_avg * vol_mult) if 'volume' in df.columns else pd.Series(True, index=df.index)
    # Bullish engulfing: prev bearish, current bullish, body[i] > body[i-1], in downtrend
    bull_eng = (
        bear_c.shift(1) & bull_c &
        (body > body.shift(1)) &
        (df['close'] < ema) &
        vol_ok
    )
    # Bearish engulfing: prev bullish, current bearish, body[i] > body[i-1], in uptrend
    bear_eng = (
        bull_c.shift(1) & bear_c &
        (body > body.shift(1)) &
        (df['close'] > ema) &
        vol_ok
    )
    out        = pd.Series(0, index=df.index)
    out[bull_eng] =  1
    out[bear_eng] = -1
    return out.fillna(0).astype(int)


def space_Engulfing_Pattern(trial):
    return {
        'ema_p':    trial.suggest_int('ema_p', 14, 50),
        'vol_mult': trial.suggest_float('vol_mult', 1.0, 2.0),
    }


# ── 3. Morning_Evening_Star ──────────────────────────────────────────────────

def gen_Morning_Evening_Star(df, big_body_pct=0.6, small_body_pct=0.2, ema_p=20, **kw):
    big_body_pct   = float(big_body_pct)
    small_body_pct = float(small_body_pct)
    ema_p          = int(ema_p)
    atr            = _atr(df, 14)
    body           = _body(df)
    big_body       = body > atr * big_body_pct
    small_body     = body < atr * small_body_pct
    bull_c         = df['close'] > df['open']
    bear_c         = df['close'] < df['open']
    ema            = _ema(df['close'], ema_p)
    # Morning star: big bear[i-2] + small body[i-1] + big bull[i], in downtrend
    morning = (
        bear_c.shift(2) & big_body.shift(2) &
        small_body.shift(1) &
        bull_c & big_body &
        (df['close'].shift(2) > ema.shift(2))
    )
    # Evening star: big bull[i-2] + small body[i-1] + big bear[i], in uptrend
    evening = (
        bull_c.shift(2) & big_body.shift(2) &
        small_body.shift(1) &
        bear_c & big_body &
        (df['close'].shift(2) < ema.shift(2))
    )
    out           = pd.Series(0, index=df.index)
    out[morning]  =  1
    out[evening]  = -1
    return out.fillna(0).astype(int)


def space_Morning_Evening_Star(trial):
    return {
        'big_body_pct':   trial.suggest_float('big_body_pct', 0.5, 0.8),
        'small_body_pct': trial.suggest_float('small_body_pct', 0.0, 0.3),
        'ema_p':          trial.suggest_int('ema_p', 14, 50),
    }


# ── 4. Three_White_Soldiers ──────────────────────────────────────────────────

def gen_Three_White_Soldiers(df, min_body_pct=0.5, ema_p=20, **kw):
    min_body_pct = float(min_body_pct)
    ema_p        = int(ema_p)
    atr          = _atr(df, 14)
    body         = _body(df)
    bull_c       = df['close'] > df['open']
    bear_c       = df['close'] < df['open']
    big          = body > atr * min_body_pct
    ema          = _ema(df['close'], ema_p)
    # Three white soldiers: 3 consecutive bullish big candles each closing higher
    three_bull = (
        bull_c & big &
        bull_c.shift(1) & big.shift(1) &
        bull_c.shift(2) & big.shift(2) &
        (df['close'] > df['close'].shift(1)) &
        (df['close'].shift(1) > df['close'].shift(2))
    )
    # Three black crows: 3 consecutive bearish big candles each closing lower
    three_bear = (
        bear_c & big &
        bear_c.shift(1) & big.shift(1) &
        bear_c.shift(2) & big.shift(2) &
        (df['close'] < df['close'].shift(1)) &
        (df['close'].shift(1) < df['close'].shift(2))
    )
    out             = pd.Series(0, index=df.index)
    out[three_bull] =  1
    out[three_bear] = -1
    return out.fillna(0).astype(int)


def space_Three_White_Soldiers(trial):
    return {
        'min_body_pct': trial.suggest_float('min_body_pct', 0.3, 0.7),
        'ema_p':        trial.suggest_int('ema_p', 14, 50),
    }


# ── 5. Marubozu ──────────────────────────────────────────────────────────────

def gen_Marubozu(df, wick_thresh=0.1, ema_p=20, **kw):
    wick_thresh = float(wick_thresh)
    ema_p       = int(ema_p)
    body        = _body(df)
    rng         = _candle_range(df)
    upper       = _upper_wick(df)
    lower       = _lower_wick(df)
    bull_c      = df['close'] > df['open']
    bear_c      = df['close'] < df['open']
    ema         = _ema(df['close'], ema_p)
    # Marubozu: body is almost all of the candle range; tiny wicks
    small_wicks = (upper < rng * wick_thresh) & (lower < rng * wick_thresh)
    bull_maru   = bull_c & small_wicks & (df['close'] > ema)
    bear_maru   = bear_c & small_wicks & (df['close'] < ema)
    out              = pd.Series(0, index=df.index)
    out[bull_maru]   =  1
    out[bear_maru]   = -1
    return out.fillna(0).astype(int)


def space_Marubozu(trial):
    return {
        'wick_thresh': trial.suggest_float('wick_thresh', 0.05, 0.2),
        'ema_p':       trial.suggest_int('ema_p', 14, 50),
    }


# ── 6. Tweezer_Tops_Bottoms ──────────────────────────────────────────────────

def gen_Tweezer_Tops_Bottoms(df, tolerance=0.002, trend_p=30, **kw):
    tolerance = float(tolerance)
    trend_p   = int(trend_p)
    ema       = _ema(df['close'], trend_p)
    atr       = _atr(df, 14)
    tol_price = atr * tolerance * 100  # tolerance as fraction of ATR
    # Tweezer bottom: two consecutive lows within tolerance, in downtrend
    low_match  = (df['low'] - df['low'].shift(1)).abs() < tol_price
    high_match = (df['high'] - df['high'].shift(1)).abs() < tol_price
    ema_down   = ema < ema.shift(1)
    ema_up     = ema > ema.shift(1)
    tweezer_bot = low_match  & ema_down & (df['close'] < ema)
    tweezer_top = high_match & ema_up   & (df['close'] > ema)
    out                  = pd.Series(0, index=df.index)
    out[tweezer_bot]     =  1
    out[tweezer_top]     = -1
    return out.fillna(0).astype(int)


def space_Tweezer_Tops_Bottoms(trial):
    return {
        'tolerance': trial.suggest_float('tolerance', 0.001, 0.005),
        'trend_p':   trial.suggest_int('trend_p', 20, 60),
    }


# ── 7. Three_Inside_UpDown ───────────────────────────────────────────────────

def gen_Three_Inside_UpDown(df, ema_p=20, **kw):
    ema_p  = int(ema_p)
    ema    = _ema(df['close'], ema_p)
    bull_c = df['close'] > df['open']
    bear_c = df['close'] < df['open']
    # Inside bar: current bar's range is fully inside previous bar
    inside_hi = df['high'].shift(1) < df['high'].shift(2)
    inside_lo = df['low'].shift(1)  > df['low'].shift(2)
    is_inside  = inside_hi & inside_lo
    # Three inside up: bearish[i-2] + inside bullish[i-1] + bullish[i] closing above bearish open
    three_in_up = (
        bear_c.shift(2) &
        is_inside &
        bull_c.shift(1) &
        bull_c &
        (df['close'] > df['open'].shift(2)) &
        (df['close'] < ema)
    )
    # Three inside down: bullish[i-2] + inside bearish[i-1] + bearish[i] closing below bullish open
    three_in_dn = (
        bull_c.shift(2) &
        is_inside &
        bear_c.shift(1) &
        bear_c &
        (df['close'] < df['open'].shift(2)) &
        (df['close'] > ema)
    )
    out              = pd.Series(0, index=df.index)
    out[three_in_up] =  1
    out[three_in_dn] = -1
    return out.fillna(0).astype(int)


def space_Three_Inside_UpDown(trial):
    return {'ema_p': trial.suggest_int('ema_p', 14, 50)}


# ── 8. Piercing_DarkCloud ────────────────────────────────────────────────────

def gen_Piercing_DarkCloud(df, pct=0.5, ema_p=20, **kw):
    pct   = float(pct)
    ema_p = int(ema_p)
    ema   = _ema(df['close'], ema_p)
    bull_c = df['close'] > df['open']
    bear_c = df['close'] < df['open']
    prev_mid = (df['open'].shift(1) + df['close'].shift(1)) / 2.0
    # Piercing line: prev bear candle + current bull opens below prev low and closes above midpoint
    piercing = (
        bear_c.shift(1) &
        bull_c &
        (df['open'] < df['close'].shift(1)) &
        (df['close'] > prev_mid) &
        (df['close'] < df['open'].shift(1)) &
        (df['close'] < ema)
    )
    # Dark cloud: prev bull candle + current bear opens above prev high and closes below midpoint
    dark_cloud = (
        bull_c.shift(1) &
        bear_c &
        (df['open'] > df['close'].shift(1)) &
        (df['close'] < prev_mid) &
        (df['close'] > df['open'].shift(1)) &
        (df['close'] > ema)
    )
    out              = pd.Series(0, index=df.index)
    out[piercing]    =  1
    out[dark_cloud]  = -1
    return out.fillna(0).astype(int)


def space_Piercing_DarkCloud(trial):
    return {
        'pct':   trial.suggest_float('pct', 0.5, 0.8),
        'ema_p': trial.suggest_int('ema_p', 14, 50),
    }


# ── 9. Spinning_Top_Reversal ─────────────────────────────────────────────────

def gen_Spinning_Top_Reversal(df, body_pct=0.2, ema_p=20, **kw):
    body_pct = float(body_pct)
    ema_p    = int(ema_p)
    rng      = _candle_range(df)
    body     = _body(df)
    upper    = _upper_wick(df)
    lower    = _lower_wick(df)
    ema      = _ema(df['close'], ema_p)
    ema_down = ema < ema.shift(2)
    ema_up   = ema > ema.shift(2)
    # Spinning top: small body, meaningful wicks on both sides
    is_spin   = (body < rng * body_pct) & (upper > rng * 0.2) & (lower > rng * 0.2)
    spin_long  = is_spin & ema_down & (df['close'] < ema)
    spin_short = is_spin & ema_up   & (df['close'] > ema)
    out              = pd.Series(0, index=df.index)
    out[spin_long]   =  1
    out[spin_short]  = -1
    return out.fillna(0).astype(int)


def space_Spinning_Top_Reversal(trial):
    return {
        'body_pct': trial.suggest_float('body_pct', 0.1, 0.3),
        'ema_p':    trial.suggest_int('ema_p', 14, 50),
    }


# ── 10. Harami ───────────────────────────────────────────────────────────────

def gen_Harami(df, inside_pct=0.5, ema_p=20, **kw):
    inside_pct = float(inside_pct)
    ema_p      = int(ema_p)
    ema        = _ema(df['close'], ema_p)
    body       = _body(df)
    bull_c     = df['close'] > df['open']
    bear_c     = df['close'] < df['open']
    # Harami: current body is inside and smaller than previous body
    inside     = (
        (df[['close', 'open']].max(axis=1) < df[['close', 'open']].max(axis=1).shift(1)) &
        (df[['close', 'open']].min(axis=1) > df[['close', 'open']].min(axis=1).shift(1)) &
        (body < body.shift(1) * inside_pct)
    )
    bull_harami = inside & bear_c.shift(1) & bull_c & (df['close'] < ema)
    bear_harami = inside & bull_c.shift(1) & bear_c & (df['close'] > ema)
    out               = pd.Series(0, index=df.index)
    out[bull_harami]  =  1
    out[bear_harami]  = -1
    return out.fillna(0).astype(int)


def space_Harami(trial):
    return {
        'inside_pct': trial.suggest_float('inside_pct', 0.3, 0.6),
        'ema_p':      trial.suggest_int('ema_p', 14, 50),
    }


# ── 11. Pin_Bar_Advanced ─────────────────────────────────────────────────────

def gen_Pin_Bar_Advanced(df, wick_ratio=2.0, ema_p=20, vol_mult=1.2, **kw):
    wick_ratio = float(wick_ratio)
    ema_p      = int(ema_p)
    vol_mult   = float(vol_mult)
    ema        = _ema(df['close'], ema_p)
    body       = _body(df).replace(0, 1e-9)
    upper      = _upper_wick(df)
    lower      = _lower_wick(df)
    ema_down   = ema < ema.shift(2)
    ema_up     = ema > ema.shift(2)
    vol_avg    = _sma(df['volume'], 20) if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    vol_ok     = (df['volume'] >= vol_avg * vol_mult) if 'volume' in df.columns else pd.Series(True, index=df.index)
    # Bullish pin: long lower wick, small upper wick
    bull_pin = (lower > body * wick_ratio) & (upper < body) & ema_down & vol_ok
    # Bearish pin: long upper wick, small lower wick
    bear_pin = (upper > body * wick_ratio) & (lower < body) & ema_up   & vol_ok
    out            = pd.Series(0, index=df.index)
    out[bull_pin]  =  1
    out[bear_pin]  = -1
    return out.fillna(0).astype(int)


def space_Pin_Bar_Advanced(trial):
    return {
        'wick_ratio': trial.suggest_float('wick_ratio', 1.5, 3.0),
        'ema_p':      trial.suggest_int('ema_p', 14, 50),
        'vol_mult':   trial.suggest_float('vol_mult', 0.8, 2.0),
    }


# ── 12. Two_Bar_Reversal ─────────────────────────────────────────────────────

def gen_Two_Bar_Reversal(df, size_pct=0.6, overlap_pct=0.5, **kw):
    size_pct    = float(size_pct)
    overlap_pct = float(overlap_pct)
    atr         = _atr(df, 14)
    body        = _body(df)
    bull_c      = df['close'] > df['open']
    bear_c      = df['close'] < df['open']
    big         = body > atr * size_pct
    # Bullish two-bar reversal: large bear then large bull with good overlap
    overlap_bull = df['close'] > (df['open'].shift(1) + df['close'].shift(1)) / 2.0 * overlap_pct
    bull_2br = bear_c.shift(1) & big.shift(1) & bull_c & big & overlap_bull
    # Bearish two-bar reversal: large bull then large bear with good overlap
    overlap_bear = df['close'] < (df['open'].shift(1) + df['close'].shift(1)) / 2.0 * (2 - overlap_pct)
    bear_2br = bull_c.shift(1) & big.shift(1) & bear_c & big & (df['close'] < df['close'].shift(1))
    out            = pd.Series(0, index=df.index)
    out[bull_2br]  =  1
    out[bear_2br]  = -1
    return out.fillna(0).astype(int)


def space_Two_Bar_Reversal(trial):
    return {
        'size_pct':    trial.suggest_float('size_pct', 0.5, 0.8),
        'overlap_pct': trial.suggest_float('overlap_pct', 0.5, 0.8),
    }


# ── 13. Outside_Bar ──────────────────────────────────────────────────────────

def gen_Outside_Bar(df, ema_p=20, **kw):
    ema_p = int(ema_p)
    ema   = _ema(df['close'], ema_p)
    # Outside bar: current high > prev high AND current low < prev low
    is_outside = (df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
    # Direction from close relative to prev close
    bull_ob = is_outside & (df['close'] > df['high'].shift(1))
    bear_ob = is_outside & (df['close'] < df['low'].shift(1))
    out          = pd.Series(0, index=df.index)
    out[bull_ob] =  1
    out[bear_ob] = -1
    return out.fillna(0).astype(int)


def space_Outside_Bar(trial):
    return {'ema_p': trial.suggest_int('ema_p', 14, 50)}


# ── 14. Kangaroo_Tail ────────────────────────────────────────────────────────

def gen_Kangaroo_Tail(df, tail_ratio=3.0, ema_p=20, **kw):
    tail_ratio = float(tail_ratio)
    ema_p      = int(ema_p)
    ema        = _ema(df['close'], ema_p)
    body       = _body(df).replace(0, 1e-9)
    lower      = _lower_wick(df)
    upper      = _upper_wick(df)
    rng        = _candle_range(df)
    ema_down   = ema < ema.shift(2)
    ema_up     = ema > ema.shift(2)
    # Kangaroo tail long: extended lower wick > tail_ratio*body, close near top, in downtrend
    bull_kt = (
        (lower > body * tail_ratio) &
        (df['close'] > df['open'] + rng * 0.5) &
        ema_down
    )
    # Kangaroo tail short: extended upper wick > tail_ratio*body, close near bottom, in uptrend
    bear_kt = (
        (upper > body * tail_ratio) &
        (df['close'] < df['open'] - rng * 0.5) &
        ema_up
    )
    out          = pd.Series(0, index=df.index)
    out[bull_kt] =  1
    out[bear_kt] = -1
    return out.fillna(0).astype(int)


def space_Kangaroo_Tail(trial):
    return {
        'tail_ratio': trial.suggest_float('tail_ratio', 2.0, 5.0),
        'ema_p':      trial.suggest_int('ema_p', 14, 50),
    }


# ── 15. Fakey_Pattern ────────────────────────────────────────────────────────

def gen_Fakey_Pattern(df, ema_p=30, **kw):
    ema_p = int(ema_p)
    ema   = _ema(df['close'], ema_p)
    # Inside bar: high[i-1] < high[i-2] AND low[i-1] > low[i-2]
    ib_high = df['high'].shift(1)
    ib_low  = df['low'].shift(1)
    prev_high = df['high'].shift(2)
    prev_low  = df['low'].shift(2)
    is_ib = (ib_high < prev_high) & (ib_low > prev_low)
    # Fakey bull: IB, false break below IB low, current close back above IB low
    false_break_down = (df['low'].shift(1) < prev_low) | (df['close'].shift(1) < ib_low)
    fakey_bull = is_ib.shift(1) & (df['low'] < ib_low.shift(1)) & (df['close'] > ib_low.shift(1))
    # Fakey bear: IB, false break above IB high, current close back below IB high
    fakey_bear = is_ib.shift(1) & (df['high'] > ib_high.shift(1)) & (df['close'] < ib_high.shift(1))
    out              = pd.Series(0, index=df.index)
    out[fakey_bull]  =  1
    out[fakey_bear]  = -1
    return out.fillna(0).astype(int)


def space_Fakey_Pattern(trial):
    return {'ema_p': trial.suggest_int('ema_p', 20, 60)}


# ── Harmonic helpers ──────────────────────────────────────────────────────────

def _get_last_pivots(df, pivot_left, pivot_right, n=5):
    """Return up to n most recent pivot high/low levels as (price, idx, is_high) tuples."""
    pivots = []
    ph = _pivot_high(df['high'], pivot_left, pivot_right)
    pl = _pivot_low(df['low'], pivot_left, pivot_right)
    for i in range(len(df) - 1, -1, -1):
        if ph.iloc[i]:
            pivots.append((df['high'].iloc[i], i, True))
        elif pl.iloc[i]:
            pivots.append((df['low'].iloc[i], i, False))
        if len(pivots) >= n:
            break
    return list(reversed(pivots))


def _check_fib(ratio, target, tol):
    return abs(ratio - target) <= tol


def _harmonic_signals(df, pivot_left, pivot_right, fib_tol,
                       ab_xa_range, bc_ab_range, cd_bc_range, d_xa_target):
    """
    Generic harmonic: detect XABCD using last 5 pivots.
    Returns (bull_signal_index, bear_signal_index) as boolean Series.
    """
    n    = len(df)
    bull = pd.Series(False, index=df.index)
    bear = pd.Series(False, index=df.index)
    ph   = _pivot_high(df['high'], pivot_left, pivot_right)
    pl   = _pivot_low(df['low'],   pivot_left, pivot_right)

    # Collect all pivot points in order
    all_pivots = []
    for i in range(n):
        if ph.iloc[i]:
            all_pivots.append((df['high'].iloc[i], i, True))
        elif pl.iloc[i]:
            all_pivots.append((df['low'].iloc[i], i, False))

    # Need at least 5 pivots alternating high-low-high-low-high (or reverse)
    for k in range(len(all_pivots) - 4):
        pts = all_pivots[k: k + 5]
        # Check alternating
        if not all(pts[i][2] != pts[i + 1][2] for i in range(4)):
            continue
        X, A, B, C, D = [p[0] for p in pts]
        D_idx = pts[4][1]
        if D_idx >= n:
            continue

        xa = abs(A - X)
        ab = abs(B - A)
        bc = abs(C - B)
        cd = abs(D - C)
        if xa < 1e-9 or ab < 1e-9 or bc < 1e-9:
            continue

        ab_xa = ab / xa
        bc_ab = bc / ab
        cd_bc = cd / bc if bc > 1e-9 else 0.0
        d_xa  = abs(D - X) / xa if xa > 1e-9 else 0.0

        fib_ok = (
            _check_fib(ab_xa, (ab_xa_range[0] + ab_xa_range[1]) / 2, fib_tol + (ab_xa_range[1] - ab_xa_range[0]) / 2) and
            _check_fib(bc_ab, (bc_ab_range[0] + bc_ab_range[1]) / 2, fib_tol + (bc_ab_range[1] - bc_ab_range[0]) / 2) and
            _check_fib(cd_bc, (cd_bc_range[0] + cd_bc_range[1]) / 2, fib_tol + (cd_bc_range[1] - cd_bc_range[0]) / 2) and
            _check_fib(d_xa,  d_xa_target, fib_tol)
        )
        if not fib_ok:
            continue

        # Bullish: X is high, A is low, B is high, C is low, D is low (below A)
        if not pts[0][2] and pts[2][2] and not pts[4][2]:  # X=low, A=high, B=low, C=high, D=low
            if D_idx < n:
                bull.iloc[D_idx] = True
        # Bearish: X is low, A is high, B is low, C is high, D is high
        if pts[0][2] and not pts[2][2] and pts[4][2]:  # X=high, A=low, B=high, C=low, D=high
            if D_idx < n:
                bear.iloc[D_idx] = True

    return bull, bear


# ── 16. ABCD_Pattern ─────────────────────────────────────────────────────────

def gen_ABCD_Pattern(df, fib_tol=0.1, pivot_left=5, pivot_right=5, **kw):
    fib_tol     = float(fib_tol)
    pivot_left  = int(pivot_left)
    pivot_right = int(pivot_right)
    ph = _pivot_high(df['high'], pivot_left, pivot_right)
    pl = _pivot_low(df['low'],   pivot_left, pivot_right)
    bull = pd.Series(False, index=df.index)
    bear = pd.Series(False, index=df.index)
    all_pivots = []
    for i in range(len(df)):
        if ph.iloc[i]:
            all_pivots.append((df['high'].iloc[i], i, True))
        elif pl.iloc[i]:
            all_pivots.append((df['low'].iloc[i], i, False))

    for k in range(len(all_pivots) - 3):
        pts = all_pivots[k: k + 4]
        if not all(pts[i][2] != pts[i + 1][2] for i in range(3)):
            continue
        A, B, C, D = [p[0] for p in pts]
        D_idx = pts[3][1]
        ab = abs(B - A)
        bc = abs(C - B)
        cd = abs(D - C)
        if ab < 1e-9:
            continue
        # ABCD: BC ~0.618*AB, CD ~AB (equality)
        bc_ab = bc / ab
        cd_ab = cd / ab
        if (_check_fib(bc_ab, 0.618, fib_tol) and _check_fib(cd_ab, 1.0, fib_tol)):
            # Bullish: A=high, B=low, C=high, D=low
            if pts[0][2] and not pts[3][2] and D_idx < len(df):
                bull.iloc[D_idx] = True
            # Bearish: A=low, B=high, C=low, D=high
            if not pts[0][2] and pts[3][2] and D_idx < len(df):
                bear.iloc[D_idx] = True

    out          = pd.Series(0, index=df.index)
    out[bull]    =  1
    out[bear]    = -1
    return out.fillna(0).astype(int)


def space_ABCD_Pattern(trial):
    return {
        'fib_tol':     trial.suggest_float('fib_tol', 0.05, 0.15),
        'pivot_left':  trial.suggest_int('pivot_left', 3, 8),
        'pivot_right': trial.suggest_int('pivot_right', 3, 8),
    }


# ── 17. Gartley_Pattern ──────────────────────────────────────────────────────

def gen_Gartley_Pattern(df, fib_tol=0.15, pivot_left=5, pivot_right=5, **kw):
    fib_tol     = float(fib_tol)
    pivot_left  = int(pivot_left)
    pivot_right = int(pivot_right)
    # Gartley: AB=0.618*XA, BC=0.382-0.886*AB, CD=1.272-1.618*BC, D=0.786*XA
    bull, bear = _harmonic_signals(
        df, pivot_left, pivot_right, fib_tol,
        ab_xa_range=(0.618, 0.618),
        bc_ab_range=(0.382, 0.886),
        cd_bc_range=(1.272, 1.618),
        d_xa_target=0.786,
    )
    out       = pd.Series(0, index=df.index)
    out[bull] =  1
    out[bear] = -1
    return out.fillna(0).astype(int)


def space_Gartley_Pattern(trial):
    return {
        'fib_tol':     trial.suggest_float('fib_tol', 0.1, 0.2),
        'pivot_left':  trial.suggest_int('pivot_left', 3, 8),
        'pivot_right': trial.suggest_int('pivot_right', 3, 8),
    }


# ── 18. Butterfly_Pattern ────────────────────────────────────────────────────

def gen_Butterfly_Pattern(df, fib_tol=0.15, pivot_left=5, pivot_right=5, **kw):
    fib_tol     = float(fib_tol)
    pivot_left  = int(pivot_left)
    pivot_right = int(pivot_right)
    # Butterfly: AB=0.786*XA, BC=0.382-0.886*AB, CD=1.618-2.618*BC, D=1.272*XA
    bull, bear = _harmonic_signals(
        df, pivot_left, pivot_right, fib_tol,
        ab_xa_range=(0.786, 0.786),
        bc_ab_range=(0.382, 0.886),
        cd_bc_range=(1.618, 2.618),
        d_xa_target=1.272,
    )
    out       = pd.Series(0, index=df.index)
    out[bull] =  1
    out[bear] = -1
    return out.fillna(0).astype(int)


def space_Butterfly_Pattern(trial):
    return {
        'fib_tol':     trial.suggest_float('fib_tol', 0.1, 0.2),
        'pivot_left':  trial.suggest_int('pivot_left', 3, 8),
        'pivot_right': trial.suggest_int('pivot_right', 3, 8),
    }


# ── 19. Bat_Pattern ──────────────────────────────────────────────────────────

def gen_Bat_Pattern(df, fib_tol=0.15, pivot_left=5, pivot_right=5, **kw):
    fib_tol     = float(fib_tol)
    pivot_left  = int(pivot_left)
    pivot_right = int(pivot_right)
    # Bat: AB=0.382-0.5*XA, BC=0.382-0.886*AB, CD=1.618-2.618*BC, D=0.886*XA
    bull, bear = _harmonic_signals(
        df, pivot_left, pivot_right, fib_tol,
        ab_xa_range=(0.382, 0.500),
        bc_ab_range=(0.382, 0.886),
        cd_bc_range=(1.618, 2.618),
        d_xa_target=0.886,
    )
    out       = pd.Series(0, index=df.index)
    out[bull] =  1
    out[bear] = -1
    return out.fillna(0).astype(int)


def space_Bat_Pattern(trial):
    return {
        'fib_tol':     trial.suggest_float('fib_tol', 0.1, 0.2),
        'pivot_left':  trial.suggest_int('pivot_left', 3, 8),
        'pivot_right': trial.suggest_int('pivot_right', 3, 8),
    }


# ── 20. Crab_Pattern ─────────────────────────────────────────────────────────

def gen_Crab_Pattern(df, fib_tol=0.15, pivot_left=5, pivot_right=5, **kw):
    fib_tol     = float(fib_tol)
    pivot_left  = int(pivot_left)
    pivot_right = int(pivot_right)
    # Crab: AB=0.382-0.618*XA, BC=0.382-0.886*AB, CD=2.618-3.618*BC, D=1.618*XA
    bull, bear = _harmonic_signals(
        df, pivot_left, pivot_right, fib_tol,
        ab_xa_range=(0.382, 0.618),
        bc_ab_range=(0.382, 0.886),
        cd_bc_range=(2.618, 3.618),
        d_xa_target=1.618,
    )
    out       = pd.Series(0, index=df.index)
    out[bull] =  1
    out[bear] = -1
    return out.fillna(0).astype(int)


def space_Crab_Pattern(trial):
    return {
        'fib_tol':     trial.suggest_float('fib_tol', 0.1, 0.2),
        'pivot_left':  trial.suggest_int('pivot_left', 3, 8),
        'pivot_right': trial.suggest_int('pivot_right', 3, 8),
    }


# ── 21. Three_Line_Strike_Bull ───────────────────────────────────────────────

def gen_Three_Line_Strike_Bull(df, ema_p=20, vol_mult=1.0, **kw):
    ema_p    = int(ema_p)
    vol_mult = float(vol_mult)
    ema      = _ema(df['close'], ema_p)
    body     = _body(df)
    atr      = _atr(df, 14)
    bull_c   = df['close'] > df['open']
    bear_c   = df['close'] < df['open']
    vol_avg  = _sma(df['volume'], 20) if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    vol_ok   = (df['volume'] >= vol_avg * vol_mult) if 'volume' in df.columns else pd.Series(True, index=df.index)
    # 3 consecutive bullish candles followed by 1 large bearish candle (the strike)
    three_bull_prev = (
        bull_c.shift(3) & bull_c.shift(2) & bull_c.shift(1) &
        (df['close'].shift(1) > df['close'].shift(2)) &
        (df['close'].shift(2) > df['close'].shift(3))
    )
    # Strike: bear candle engulfing all three
    strike_bear = (
        bear_c &
        (df['open'] >= df['close'].shift(1)) &
        (df['close'] <= df['open'].shift(3)) &
        vol_ok
    )
    # Counter-trend: buy after bearish strike in bull sequence
    bull_tls = three_bull_prev & strike_bear
    # 3 consecutive bearish + big bull strike
    three_bear_prev = (
        bear_c.shift(3) & bear_c.shift(2) & bear_c.shift(1) &
        (df['close'].shift(1) < df['close'].shift(2)) &
        (df['close'].shift(2) < df['close'].shift(3))
    )
    strike_bull = (
        bull_c &
        (df['open'] <= df['close'].shift(1)) &
        (df['close'] >= df['open'].shift(3)) &
        vol_ok
    )
    bear_tls = three_bear_prev & strike_bull
    out           = pd.Series(0, index=df.index)
    out[bull_tls] =  1
    out[bear_tls] = -1
    return out.fillna(0).astype(int)


def space_Three_Line_Strike_Bull(trial):
    return {
        'ema_p':    trial.suggest_int('ema_p', 14, 50),
        'vol_mult': trial.suggest_float('vol_mult', 0.8, 1.5),
    }


# ── 22. Three_Line_Strike_Bear ───────────────────────────────────────────────

def gen_Three_Line_Strike_Bear(df, ema_p=20, vol_mult=1.0, **kw):
    ema_p    = int(ema_p)
    vol_mult = float(vol_mult)
    ema      = _ema(df['close'], ema_p)
    bull_c   = df['close'] > df['open']
    bear_c   = df['close'] < df['open']
    vol_avg  = _sma(df['volume'], 20) if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    vol_ok   = (df['volume'] >= vol_avg * vol_mult) if 'volume' in df.columns else pd.Series(True, index=df.index)
    # Three bear candles + bull strike (counter-trend short)
    three_bear_prev = (
        bear_c.shift(3) & bear_c.shift(2) & bear_c.shift(1) &
        (df['close'].shift(1) < df['close'].shift(2)) &
        (df['close'].shift(2) < df['close'].shift(3))
    )
    strike_bull = (
        bull_c &
        (df['open'] <= df['close'].shift(1)) &
        (df['close'] >= df['open'].shift(3)) &
        vol_ok
    )
    bear_tls = three_bear_prev & strike_bull
    # Three bull candles + bear strike (counter-trend long)
    three_bull_prev = (
        bull_c.shift(3) & bull_c.shift(2) & bull_c.shift(1) &
        (df['close'].shift(1) > df['close'].shift(2)) &
        (df['close'].shift(2) > df['close'].shift(3))
    )
    strike_bear = (
        bear_c &
        (df['open'] >= df['close'].shift(1)) &
        (df['close'] <= df['open'].shift(3)) &
        vol_ok
    )
    bull_tls = three_bull_prev & strike_bear
    out           = pd.Series(0, index=df.index)
    out[bull_tls] =  1
    out[bear_tls] = -1
    return out.fillna(0).astype(int)


def space_Three_Line_Strike_Bear(trial):
    return {
        'ema_p':    trial.suggest_int('ema_p', 14, 50),
        'vol_mult': trial.suggest_float('vol_mult', 0.8, 1.5),
    }


# ── 23. Pivot_Daily ──────────────────────────────────────────────────────────

def gen_Pivot_Daily(df, pivot_p=24, **kw):
    pivot_p = int(pivot_p)
    H = df['high'].rolling(pivot_p).max()
    L = df['low'].rolling(pivot_p).min()
    C = df['close'].rolling(pivot_p).mean()
    PP = (H + L + C) / 3.0
    R1 = 2.0 * PP - L
    S1 = 2.0 * PP - H
    # Long: close crosses above PP; Short: close crosses below PP
    above_pp = df['close'] > PP
    below_pp = df['close'] < PP
    long  = above_pp & ~above_pp.shift(1).astype(bool).fillna(False)
    short = below_pp & ~below_pp.shift(1).astype(bool).fillna(False)
    out       = pd.Series(0, index=df.index)
    out[long] =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_Pivot_Daily(trial):
    return {'pivot_p': trial.suggest_int('pivot_p', 20, 30)}


# ── 24. Camarilla_Pivots ─────────────────────────────────────────────────────

def gen_Camarilla_Pivots(df, pivot_p=24, **kw):
    pivot_p = int(pivot_p)
    H = df['high'].rolling(pivot_p).max()
    L = df['low'].rolling(pivot_p).min()
    C = df['close']
    rng = H - L
    R3 = C + 1.1 * rng / 4.0
    S3 = C - 1.1 * rng / 4.0
    R4 = C + 1.1 * rng / 2.0
    S4 = C - 1.1 * rng / 2.0
    # Long: close breaks above R3; Short: close breaks below S3
    above_r3 = df['close'] > R3
    below_s3 = df['close'] < S3
    long  = above_r3 & ~above_r3.shift(1).astype(bool).fillna(False)
    short = below_s3 & ~below_s3.shift(1).astype(bool).fillna(False)
    out       = pd.Series(0, index=df.index)
    out[long] =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_Camarilla_Pivots(trial):
    return {'pivot_p': trial.suggest_int('pivot_p', 20, 30)}


# ── 25. Woodie_Pivots ────────────────────────────────────────────────────────

def gen_Woodie_Pivots(df, pivot_p=24, **kw):
    pivot_p = int(pivot_p)
    H   = df['high'].rolling(pivot_p).max()
    L   = df['low'].rolling(pivot_p).min()
    C   = df['close']
    PP  = (H + L + 2.0 * C) / 4.0
    R1  = 2.0 * PP - L
    S1  = 2.0 * PP - H
    R2  = PP + (H - L)
    S2  = PP - (H - L)
    # Long: bounce off S1 (close crosses above S1 from below)
    above_s1 = df['close'] > S1
    below_r1 = df['close'] < R1
    long  = above_s1 & ~above_s1.shift(1).astype(bool).fillna(False) & (df['close'] < PP)
    short = below_r1 & ~below_r1.shift(1).astype(bool).fillna(False) & (df['close'] > PP)
    out       = pd.Series(0, index=df.index)
    out[long] =  1
    out[short] = -1
    return out.fillna(0).astype(int)


def space_Woodie_Pivots(trial):
    return {'pivot_p': trial.suggest_int('pivot_p', 20, 30)}


# ── 26. Bull_Flag ────────────────────────────────────────────────────────────

def gen_Bull_Flag(df, pole_mult=2.0, flag_bars=5, vol_mult=1.5, **kw):
    pole_mult = float(pole_mult)
    flag_bars = int(flag_bars)
    vol_mult  = float(vol_mult)
    atr       = _atr(df, 14)
    close     = df['close']
    vol_avg   = _sma(df['volume'], 20) if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    # Pole: a big up move in the preceding bar
    pole_up   = close.shift(flag_bars) - close.shift(flag_bars + 3)
    pole_big  = pole_up > atr.shift(flag_bars) * pole_mult
    # Flag: low volatility in the last flag_bars bars
    flag_std  = close.rolling(flag_bars).std()
    flag_atr  = atr.rolling(flag_bars).mean()
    flag_tight = flag_std < flag_atr * 0.5
    # Breakout: close > highest close during flag
    flag_high = close.shift(1).rolling(flag_bars).max()
    breakout  = close > flag_high
    vol_break = (df['volume'] >= vol_avg * vol_mult) if 'volume' in df.columns else pd.Series(True, index=df.index)
    bull_flag = pole_big & flag_tight & breakout & vol_break
    # Bear flag: pole down + consolidation + breakdown
    pole_dn    = close.shift(flag_bars + 3) - close.shift(flag_bars)
    pole_big_d = pole_dn > atr.shift(flag_bars) * pole_mult
    flag_low   = close.shift(1).rolling(flag_bars).min()
    breakdown  = close < flag_low
    bear_flag  = pole_big_d & flag_tight & breakdown & vol_break
    out             = pd.Series(0, index=df.index)
    out[bull_flag]  =  1
    out[bear_flag]  = -1
    return out.fillna(0).astype(int)


def space_Bull_Flag(trial):
    return {
        'pole_mult': trial.suggest_float('pole_mult', 1.5, 3.0),
        'flag_bars': trial.suggest_int('flag_bars', 3, 8),
        'vol_mult':  trial.suggest_float('vol_mult', 1.2, 2.5),
    }


# ── 27. Bear_Flag ────────────────────────────────────────────────────────────

def gen_Bear_Flag(df, pole_mult=2.0, flag_bars=5, vol_mult=1.5, **kw):
    pole_mult = float(pole_mult)
    flag_bars = int(flag_bars)
    vol_mult  = float(vol_mult)
    atr       = _atr(df, 14)
    close     = df['close']
    vol_avg   = _sma(df['volume'], 20) if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    vol_break = (df['volume'] >= vol_avg * vol_mult) if 'volume' in df.columns else pd.Series(True, index=df.index)
    flag_std  = close.rolling(flag_bars).std()
    flag_atr  = atr.rolling(flag_bars).mean()
    flag_tight = flag_std < flag_atr * 0.5
    # Bear flag: big downward pole + tight consolidation + breakdown
    pole_dn    = close.shift(flag_bars + 3) - close.shift(flag_bars)
    pole_big   = pole_dn > atr.shift(flag_bars) * pole_mult
    flag_low   = close.shift(1).rolling(flag_bars).min()
    breakdown  = close < flag_low
    bear_flag  = pole_big & flag_tight & breakdown & vol_break
    # Bull flag (counter)
    pole_up    = close.shift(flag_bars) - close.shift(flag_bars + 3)
    pole_big_u = pole_up > atr.shift(flag_bars) * pole_mult
    flag_high  = close.shift(1).rolling(flag_bars).max()
    breakout   = close > flag_high
    bull_flag  = pole_big_u & flag_tight & breakout & vol_break
    out             = pd.Series(0, index=df.index)
    out[bull_flag]  =  1
    out[bear_flag]  = -1
    return out.fillna(0).astype(int)


def space_Bear_Flag(trial):
    return {
        'pole_mult': trial.suggest_float('pole_mult', 1.5, 3.0),
        'flag_bars': trial.suggest_int('flag_bars', 3, 8),
        'vol_mult':  trial.suggest_float('vol_mult', 1.2, 2.5),
    }


# ── 28. Pennant ──────────────────────────────────────────────────────────────

def gen_Pennant(df, pole_mult=2.0, pennant_bars=8, **kw):
    pole_mult    = float(pole_mult)
    pennant_bars = int(pennant_bars)
    atr          = _atr(df, 14)
    close        = df['close']
    # Pole: big move before pennant
    pole_up  = close.shift(pennant_bars) - close.shift(pennant_bars + 3)
    pole_dn  = close.shift(pennant_bars + 3) - close.shift(pennant_bars)
    pole_big_u = pole_up > atr.shift(pennant_bars) * pole_mult
    pole_big_d = pole_dn > atr.shift(pennant_bars) * pole_mult
    # Pennant: converging highs and lows (shrinking range)
    high_roll   = df['high'].rolling(pennant_bars)
    low_roll    = df['low'].rolling(pennant_bars)
    range_now   = high_roll.max() - low_roll.min()
    range_prev  = df['high'].rolling(pennant_bars).max().shift(pennant_bars // 2) - \
                  df['low'].rolling(pennant_bars).min().shift(pennant_bars // 2)
    converging  = range_now < range_prev * 0.8
    # Breakout
    prev_high   = df['high'].shift(1).rolling(pennant_bars).max()
    prev_low    = df['low'].shift(1).rolling(pennant_bars).min()
    bull_break  = close > prev_high
    bear_break  = close < prev_low
    bull_pennant = pole_big_u & converging & bull_break
    bear_pennant = pole_big_d & converging & bear_break
    out                 = pd.Series(0, index=df.index)
    out[bull_pennant]   =  1
    out[bear_pennant]   = -1
    return out.fillna(0).astype(int)


def space_Pennant(trial):
    return {
        'pole_mult':    trial.suggest_float('pole_mult', 1.5, 3.0),
        'pennant_bars': trial.suggest_int('pennant_bars', 5, 15),
    }


# ── 29. Gap_Fill ─────────────────────────────────────────────────────────────

def gen_Gap_Fill(df, min_gap_pct=0.005, fill_pct=0.5, **kw):
    min_gap_pct = float(min_gap_pct)
    fill_pct    = float(fill_pct)
    prev_close  = df['close'].shift(1)
    prev_high   = df['high'].shift(1)
    prev_low    = df['low'].shift(1)
    gap_up_size   = (df['open'] - prev_high) / prev_close.replace(0, 1e-9)
    gap_down_size = (prev_low - df['open'])  / prev_close.replace(0, 1e-9)
    gap_up   = gap_up_size   >= min_gap_pct
    gap_down = gap_down_size >= min_gap_pct
    # Gap fill long: gap down, price starts rising to fill gap (close > open significantly)
    fill_progress_down = (df['close'] - df['open']) / (prev_low - df['open']).replace(0, 1e-9)
    # Gap fill short: gap up, price starts filling gap downward
    fill_progress_up   = (df['open'] - df['close']) / (df['open'] - prev_high).replace(0, 1e-9)
    long_fill  = gap_down & (df['close'] > df['open']) & (fill_progress_down > fill_pct)
    short_fill = gap_up   & (df['close'] < df['open']) & (fill_progress_up   > fill_pct)
    out              = pd.Series(0, index=df.index)
    out[long_fill]   =  1
    out[short_fill]  = -1
    return out.fillna(0).astype(int)


def space_Gap_Fill(trial):
    return {
        'min_gap_pct': trial.suggest_float('min_gap_pct', 0.002, 0.01),
        'fill_pct':    trial.suggest_float('fill_pct', 0.3, 0.8),
    }


# ── 30. Gap_Continuation ─────────────────────────────────────────────────────

def gen_Gap_Continuation(df, gap_pct=0.005, ema_p=20, **kw):
    gap_pct = float(gap_pct)
    ema_p   = int(ema_p)
    ema     = _ema(df['close'], ema_p)
    prev_close = df['close'].shift(1)
    prev_high  = df['high'].shift(1)
    prev_low   = df['low'].shift(1)
    ema_up   = ema > ema.shift(2)
    ema_down = ema < ema.shift(2)
    gap_up_size   = (df['open'] - prev_high)  / prev_close.replace(0, 1e-9)
    gap_down_size = (prev_low  - df['open'])  / prev_close.replace(0, 1e-9)
    # Breakaway gap: gap in trend direction = continuation
    bull_gap = (gap_up_size >= gap_pct) & ema_up   & (df['close'] > df['open'])
    bear_gap = (gap_down_size >= gap_pct) & ema_down & (df['close'] < df['open'])
    out           = pd.Series(0, index=df.index)
    out[bull_gap] =  1
    out[bear_gap] = -1
    return out.fillna(0).astype(int)


def space_Gap_Continuation(trial):
    return {
        'gap_pct': trial.suggest_float('gap_pct', 0.002, 0.01),
        'ema_p':   trial.suggest_int('ema_p', 14, 50),
    }


# ── STRATEGY_EXPORT ───────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Doji_Reversal': {
        'gen': gen_Doji_Reversal,
        'space': space_Doji_Reversal,
        'default_params': {'doji_thresh': 0.1, 'ema_p': 20, 'atr_p': 14},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 3000,
            'description': 'Doji candle (body < atr*thresh) at trend extremes. Long: doji in downtrend below EMA. Short: doji in uptrend above EMA.',
        },
    },
    'Engulfing_Pattern': {
        'gen': gen_Engulfing_Pattern,
        'space': space_Engulfing_Pattern,
        'default_params': {'ema_p': 20, 'vol_mult': 1.2},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 5000,
            'description': 'Bull engulfing: bearish prev + bullish current with larger body + downtrend + volume confirm.',
        },
    },
    'Morning_Evening_Star': {
        'gen': gen_Morning_Evening_Star,
        'space': space_Morning_Evening_Star,
        'default_params': {'big_body_pct': 0.6, 'small_body_pct': 0.2, 'ema_p': 20},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2500,
            'description': 'Morning star: big bear + small body + big bull in downtrend. Evening star: opposite in uptrend.',
        },
    },
    'Three_White_Soldiers': {
        'gen': gen_Three_White_Soldiers,
        'space': space_Three_White_Soldiers,
        'default_params': {'min_body_pct': 0.5, 'ema_p': 20},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2000,
            'description': 'Three consecutive big bullish candles each closing higher. Three black crows: opposite. Entry after 3rd candle.',
        },
    },
    'Marubozu': {
        'gen': gen_Marubozu,
        'space': space_Marubozu,
        'default_params': {'wick_thresh': 0.1, 'ema_p': 20},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2000,
            'description': 'Full-body candle with tiny wicks (<wick_thresh of range). Trend confirmation with EMA.',
        },
    },
    'Tweezer_Tops_Bottoms': {
        'gen': gen_Tweezer_Tops_Bottoms,
        'space': space_Tweezer_Tops_Bottoms,
        'default_params': {'tolerance': 0.002, 'trend_p': 30},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1800,
            'description': 'Tweezer bottom: two matching lows in downtrend. Tweezer top: two matching highs in uptrend.',
        },
    },
    'Three_Inside_UpDown': {
        'gen': gen_Three_Inside_UpDown,
        'space': space_Three_Inside_UpDown,
        'default_params': {'ema_p': 20},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1500,
            'description': 'Three inside up: bearish + bullish inside + bullish close above bearish open. Opposite for three inside down.',
        },
    },
    'Piercing_DarkCloud': {
        'gen': gen_Piercing_DarkCloud,
        'space': space_Piercing_DarkCloud,
        'default_params': {'pct': 0.5, 'ema_p': 20},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1500,
            'description': 'Piercing line: bear then bull opening below, closing above midpoint. Dark cloud cover: opposite.',
        },
    },
    'Spinning_Top_Reversal': {
        'gen': gen_Spinning_Top_Reversal,
        'space': space_Spinning_Top_Reversal,
        'default_params': {'body_pct': 0.2, 'ema_p': 20},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1200,
            'description': 'Small body with wicks on both sides at trend extremes. Long at bottom, short at top.',
        },
    },
    'Harami': {
        'gen': gen_Harami,
        'space': space_Harami,
        'default_params': {'inside_pct': 0.5, 'ema_p': 20},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1500,
            'description': 'Large candle followed by small inside candle. Bull harami in downtrend = long signal.',
        },
    },
    'Pin_Bar_Advanced': {
        'gen': gen_Pin_Bar_Advanced,
        'space': space_Pin_Bar_Advanced,
        'default_params': {'wick_ratio': 2.0, 'ema_p': 20, 'vol_mult': 1.2},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 4000,
            'description': 'Pin bar with wick > wick_ratio*body. Bullish pin: long lower wick in downtrend + volume. Bearish pin: long upper wick.',
        },
    },
    'Two_Bar_Reversal': {
        'gen': gen_Two_Bar_Reversal,
        'space': space_Two_Bar_Reversal,
        'default_params': {'size_pct': 0.6, 'overlap_pct': 0.5},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2000,
            'description': 'Two large opposite candles with significant overlap. Bull: large bear then large bull.',
        },
    },
    'Outside_Bar': {
        'gen': gen_Outside_Bar,
        'space': space_Outside_Bar,
        'default_params': {'ema_p': 20},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2000,
            'description': 'Outside bar engulfs previous bar. Direction from close: above prev high = long, below prev low = short.',
        },
    },
    'Kangaroo_Tail': {
        'gen': gen_Kangaroo_Tail,
        'space': space_Kangaroo_Tail,
        'default_params': {'tail_ratio': 3.0, 'ema_p': 20},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2500,
            'description': 'Extended shadow spike (tail > tail_ratio*body) at trend extremes. Long: lower tail in downtrend.',
        },
    },
    'Fakey_Pattern': {
        'gen': gen_Fakey_Pattern,
        'space': space_Fakey_Pattern,
        'default_params': {'ema_p': 30},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2000,
            'description': 'Inside bar false breakout reversal. Long: IB formed, false break below, close recovers above IB low.',
        },
    },
    'ABCD_Pattern': {
        'gen': gen_ABCD_Pattern,
        'space': space_ABCD_Pattern,
        'default_params': {'fib_tol': 0.1, 'pivot_left': 5, 'pivot_right': 5},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 3000,
            'description': 'ABCD harmonic: BC=0.618*AB, CD=AB. Bullish ABCD completes at D pivot low.',
        },
    },
    'Gartley_Pattern': {
        'gen': gen_Gartley_Pattern,
        'space': space_Gartley_Pattern,
        'default_params': {'fib_tol': 0.15, 'pivot_left': 5, 'pivot_right': 5},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2500,
            'description': 'Gartley 222: AB=0.618*XA, BC=0.382-0.886*AB, CD=1.272-1.618*BC, D=0.786*XA.',
        },
    },
    'Butterfly_Pattern': {
        'gen': gen_Butterfly_Pattern,
        'space': space_Butterfly_Pattern,
        'default_params': {'fib_tol': 0.15, 'pivot_left': 5, 'pivot_right': 5},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2000,
            'description': 'Butterfly harmonic: AB=0.786*XA, BC=0.382-0.886*AB, CD=1.618-2.618*BC, D=1.272*XA.',
        },
    },
    'Bat_Pattern': {
        'gen': gen_Bat_Pattern,
        'space': space_Bat_Pattern,
        'default_params': {'fib_tol': 0.15, 'pivot_left': 5, 'pivot_right': 5},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1800,
            'description': 'Bat harmonic: AB=0.382-0.5*XA, BC=0.382-0.886*AB, CD=1.618-2.618*BC, D=0.886*XA.',
        },
    },
    'Crab_Pattern': {
        'gen': gen_Crab_Pattern,
        'space': space_Crab_Pattern,
        'default_params': {'fib_tol': 0.15, 'pivot_left': 5, 'pivot_right': 5},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 1500,
            'description': 'Crab harmonic (most extreme): AB=0.382-0.618*XA, CD=2.618-3.618*BC, D=1.618*XA.',
        },
    },
    'Three_Line_Strike_Bull': {
        'gen': gen_Three_Line_Strike_Bull,
        'space': space_Three_Line_Strike_Bull,
        'default_params': {'ema_p': 20, 'vol_mult': 1.0},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2000,
            'description': 'Three consecutive bull candles + bearish strike engulfing all three. Counter-trend buy on the strike.',
        },
    },
    'Three_Line_Strike_Bear': {
        'gen': gen_Three_Line_Strike_Bear,
        'space': space_Three_Line_Strike_Bear,
        'default_params': {'ema_p': 20, 'vol_mult': 1.0},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2000,
            'description': 'Three consecutive bear candles + bullish strike engulfing all three. Counter-trend short on the strike.',
        },
    },
    'Pivot_Daily': {
        'gen': gen_Pivot_Daily,
        'space': space_Pivot_Daily,
        'default_params': {'pivot_p': 24},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 3000,
            'description': 'Classical pivot PP=(H+L+C)/3 using rolling lookback. Long: close crosses above PP. Short: crosses below.',
        },
    },
    'Camarilla_Pivots': {
        'gen': gen_Camarilla_Pivots,
        'space': space_Camarilla_Pivots,
        'default_params': {'pivot_p': 24},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2000,
            'description': 'Camarilla R3=close+1.1*(H-L)/4, S3=close-1.1*(H-L)/4. Long: breakout above R3. Short: below S3.',
        },
    },
    'Woodie_Pivots': {
        'gen': gen_Woodie_Pivots,
        'space': space_Woodie_Pivots,
        'default_params': {'pivot_p': 24},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 1800,
            'description': 'Woodie PP=(H+L+2C)/4. Long: bounce off S1 below PP. Short: rejection from R1 above PP.',
        },
    },
    'Bull_Flag': {
        'gen': gen_Bull_Flag,
        'space': space_Bull_Flag,
        'default_params': {'pole_mult': 2.0, 'flag_bars': 5, 'vol_mult': 1.5},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 3000,
            'description': 'Bull flag: strong upward pole + tight low-vol consolidation + breakout with volume. Bearish counterpart included.',
        },
    },
    'Bear_Flag': {
        'gen': gen_Bear_Flag,
        'space': space_Bear_Flag,
        'default_params': {'pole_mult': 2.0, 'flag_bars': 5, 'vol_mult': 1.5},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 3000,
            'description': 'Bear flag: strong downward pole + tight consolidation + breakdown with volume.',
        },
    },
    'Pennant': {
        'gen': gen_Pennant,
        'space': space_Pennant,
        'default_params': {'pole_mult': 2.0, 'pennant_bars': 8},
        'info': {
            'source': 'TradingView',
            'version': 5,
            'likes': 2000,
            'description': 'Pennant: strong pole + converging highs/lows (shrinking range) + continuation breakout.',
        },
    },
    'Gap_Fill': {
        'gen': gen_Gap_Fill,
        'space': space_Gap_Fill,
        'default_params': {'min_gap_pct': 0.005, 'fill_pct': 0.5},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2500,
            'description': 'Price gaps up/down then begins to fill. Long: gap down + starts filling. Short: gap up + filling down.',
        },
    },
    'Gap_Continuation': {
        'gen': gen_Gap_Continuation,
        'space': space_Gap_Continuation,
        'default_params': {'gap_pct': 0.005, 'ema_p': 20},
        'info': {
            'source': 'TradingView',
            'version': 4,
            'likes': 2000,
            'description': 'Breakaway gap in trend direction = continuation. Long: gap up in uptrend. Short: gap down in downtrend.',
        },
    },
}
