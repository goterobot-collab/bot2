"""
Batch 3706 - Mac paralela wave m7 - 5 EXPERT-ATTRIBUTED families.

Owner: Mac paralela (Claude Code online, rama claude/grail-hunt-bot-noeIq).

Each family is sourced from a documented professional trader / quant with
explicit entry/exit rules (verified via web research 2026-04-25):

 1. TV_Raschke_HolyGrail        - Linda Raschke "Holy Grail" pattern
                                   (ADX>30 + pullback to 20 EMA)
                                   Source: Raschke/Connors "Street Smarts" 1995
 2. TV_Connors_TurtleSoup       - Larry Connors "Turtle Soup" false breakout
                                   reversal at N-bar high/low
                                   Source: Raschke/Connors "Street Smarts" 1995
 3. TV_Raschke_8020Reversal     - Raschke "80-20" opening-range reversal
                                   (adapted to bar-level for crypto 24/7)
                                   Source: Raschke/Connors "Street Smarts" 1995
 4. TV_Davey_NBarInside_Breakout- Kevin Davey N consecutive inside bars +
                                   breakout (volatility compression style,
                                   distinct from BB/Keltner squeeze)
                                   Source: Davey "Entry & Exit Confessions" 2019
 5. TV_Asness_TrendMomentum     - Asness/AQR trend (200d MA filter) + price
                                   momentum (12-1 month return) combo signal,
                                   single-asset adaptation of "Value & Momentum
                                   Everywhere" Journal of Finance 2013
                                   Source: Asness-Moskowitz-Pedersen 2013 JF

Verified ZERO overlap with sandbox 3566-3610, Mac V8 prod (1717 covered),
or Mac batches 3700-3705.

All pickle-safe, signal int {-1,0,1}, .shift(1) applied.
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _sma(s, n):
    return s.rolling(int(n), min_periods=int(n)).mean()


def _atr(df, n):
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / int(n), adjust=False, min_periods=int(n)).mean()


def _adx(df, n):
    """Wilder ADX, returns (+DI, -DI, ADX)."""
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    up = h - h.shift(1)
    dn = l.shift(1) - l
    plus_dm = ((up > dn) & (up > 0)).astype(float) * up
    minus_dm = ((dn > up) & (dn > 0)).astype(float) * dn
    atr = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_val = dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return plus_di, minus_di, adx_val


# ----- 1) RASCHKE HOLY GRAIL -----
def gen_TV_Raschke_HolyGrail(df, ema_len=20, adx_len=14, adx_min=30, **kw):
    """Linda Raschke "Holy Grail" pattern (Street Smarts ch6):
       1. ADX > 30 (strong trend)
       2. Price pullback to the 20-EMA (close touches/crosses below in uptrend
          or above in downtrend; +DI/-DI determines direction)
       3. Entry on price moving back toward trend direction
    Long: +DI > -DI AND ADX > min AND low <= EMA20 AND close > EMA20.
    Short: -DI > +DI AND ADX > min AND high >= EMA20 AND close < EMA20.
    """
    el = int(ema_len)
    al = int(adx_len)
    am = float(adx_min)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    ema = _ema(c, el)
    plus_di, minus_di, adx = _adx(df, al)
    long_setup = ((plus_di > minus_di) & (adx > am) & (l <= ema) & (c > ema))
    short_setup = ((minus_di > plus_di) & (adx > am) & (h >= ema) & (c < ema))
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Raschke_HolyGrail():
    return {
        'ema_len': ('int', 10, 35),
        'adx_len': ('int', 10, 25),
        'adx_min': ('int', 20, 40),
    }


# ----- 2) CONNORS TURTLE SOUP -----
def gen_TV_Connors_TurtleSoup(df, breakout_len=20, recovery_bars=2, **kw):
    """Larry Connors "Turtle Soup" (Street Smarts ch1):
    Original Turtles enter on N-bar break - Connors fades that:
    - Bar makes new N-bar LOW (false breakdown). Within recovery_bars,
      price closes back ABOVE prior N-bar low -> LONG (fade the breakdown).
    - Bar makes new N-bar HIGH (false breakout). Within recovery_bars,
      price closes back BELOW prior N-bar high -> SHORT.
    """
    n = int(breakout_len)
    rb = int(recovery_bars)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    prior_low = l.rolling(n, min_periods=n).min().shift(1)
    prior_high = h.rolling(n, min_periods=n).max().shift(1)
    new_low = l < prior_low
    new_high = h > prior_high
    long_trigger = pd.Series(False, index=df.index)
    short_trigger = pd.Series(False, index=df.index)
    for offset in range(1, rb + 1):
        long_trigger = long_trigger | (new_low.shift(offset).fillna(False) & (c > prior_low.shift(offset)))
        short_trigger = short_trigger | (new_high.shift(offset).fillna(False) & (c < prior_high.shift(offset)))
    long_entry = long_trigger & ~(long_trigger.shift(1).fillna(False))
    short_entry = short_trigger & ~(short_trigger.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Connors_TurtleSoup():
    return {
        'breakout_len': ('int', 10, 40),
        'recovery_bars': ('int', 1, 4),
    }


# ----- 3) RASCHKE 80-20 REVERSAL -----
def gen_TV_Raschke_8020Reversal(df, extreme_pct=20, **kw):
    """Linda Raschke "80-20" (Street Smarts ch3):
    Original: equity opens in extreme 20% of prior day's range, then reverses
    to the other 80% of the range.
    Crypto adaptation (24/7, no daily session): bar OPEN within bottom
    `extreme_pct`% of PRIOR bar's range -> long entry (reversal up to mid).
    Bar OPEN within top `extreme_pct`% -> short.
    """
    ep = float(extreme_pct)
    o = df['open'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    prior_low = l.shift(1)
    prior_high = h.shift(1)
    prior_rng = (prior_high - prior_low).replace(0.0, np.nan)
    open_pct = (o - prior_low) / prior_rng * 100.0
    long_setup = open_pct < ep
    short_setup = open_pct > (100 - ep)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Raschke_8020Reversal():
    return {
        'extreme_pct': ('int', 10, 30),
    }


# ----- 4) DAVEY N-BAR INSIDE BREAKOUT -----
def gen_TV_Davey_NBarInside_Breakout(df, inside_count=3, breakout_atr_mult=0.5,
                                      atr_len=14, **kw):
    """Kevin Davey systematic trader pattern:
    1. Count consecutive "inside bars" (bar's high <= prev high AND
       bar's low >= prev low). Pattern: >= inside_count consecutive.
    2. Breakout: close > prior_max + breakout_atr_mult * ATR -> LONG.
       close < prior_min - breakout_atr_mult * ATR -> SHORT.
    """
    ic = int(inside_count)
    bm = float(breakout_atr_mult)
    al = int(atr_len)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    is_inside = (h <= h.shift(1)) & (l >= l.shift(1))
    consec = is_inside.rolling(ic, min_periods=ic).sum() == ic
    # Reference high/low across the consec inside-bar window
    pre_window_high = h.shift(ic).rolling(ic, min_periods=ic).max()
    pre_window_low = l.shift(ic).rolling(ic, min_periods=ic).min()
    atr = _atr(df, al)
    long_break = consec.shift(1).fillna(False) & (c > pre_window_high.shift(1) + bm * atr)
    short_break = consec.shift(1).fillna(False) & (c < pre_window_low.shift(1) - bm * atr)
    long_entry = long_break & ~(long_break.shift(1).fillna(False))
    short_entry = short_break & ~(short_break.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Davey_NBarInside_Breakout():
    return {
        'inside_count': ('int', 2, 6),
        'breakout_atr_mult': ('float', 0.2, 1.5),
        'atr_len': ('int', 7, 30),
    }


# ----- 5) ASNESS TREND-MOMENTUM COMBO -----
def gen_TV_Asness_TrendMomentum(df, trend_ma_len=200, mom_long_len=252,
                                 mom_skip_len=21, mom_threshold=0.0, **kw):
    """Asness-Moskowitz-Pedersen "Value and Momentum Everywhere" JF 2013:
    classic long-short signal combining:
     - TREND filter: close > MA(trend_ma_len) for long bias
     - MOMENTUM: 12-1 month return = (close.shift(mom_skip_len) /
       close.shift(mom_long_len) - 1), skipping last month to avoid reversal
    Long entry: close > MA(200) AND mom > +threshold (transitioning to true)
    Short entry: close < MA(200) AND mom < -threshold
    """
    tl = int(trend_ma_len)
    ml = int(mom_long_len)
    sk = int(mom_skip_len)
    th = float(mom_threshold)
    c = df['close'].astype(float)
    ma = _sma(c, tl)
    mom = (c.shift(sk) / c.shift(ml) - 1.0)
    long_setup = (c > ma) & (mom > th)
    short_setup = (c < ma) & (mom < -th)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Asness_TrendMomentum():
    return {
        'trend_ma_len': ('int', 100, 300),
        'mom_long_len': ('int', 100, 300),
        'mom_skip_len': ('int', 10, 30),
        'mom_threshold': ('float', 0.0, 0.20),
    }


STRATEGY_EXPORT = {
    "TV_Raschke_HolyGrail": {
        "gen": gen_TV_Raschke_HolyGrail, "space": space_TV_Raschke_HolyGrail,
        "source": "mac_batch3706_RaschkeConnors_StreetSmarts_1995_ch6",
    },
    "TV_Connors_TurtleSoup": {
        "gen": gen_TV_Connors_TurtleSoup, "space": space_TV_Connors_TurtleSoup,
        "source": "mac_batch3706_RaschkeConnors_StreetSmarts_1995_ch1",
    },
    "TV_Raschke_8020Reversal": {
        "gen": gen_TV_Raschke_8020Reversal, "space": space_TV_Raschke_8020Reversal,
        "source": "mac_batch3706_RaschkeConnors_StreetSmarts_1995_ch3",
    },
    "TV_Davey_NBarInside_Breakout": {
        "gen": gen_TV_Davey_NBarInside_Breakout,
        "space": space_TV_Davey_NBarInside_Breakout,
        "source": "mac_batch3706_Davey_EntryExitConfessions_2019",
    },
    "TV_Asness_TrendMomentum": {
        "gen": gen_TV_Asness_TrendMomentum, "space": space_TV_Asness_TrendMomentum,
        "source": "mac_batch3706_AsnessMoskowitzPedersen_2013_JF",
    },
}
