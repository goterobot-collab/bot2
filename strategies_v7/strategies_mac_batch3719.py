"""
Batch 3719 - Mac paralela wave m20 - 5 candlestick / price-action patterns.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod (incl.
hammer_shooting_star), Mac 3700-3718.

 1. TV_Engulfing_Reversal      - bullish/bearish engulfing pattern
 2. TV_PinBar_Reversal         - long wick (>=N*body) rejection candle
 3. TV_Wide_Range_Climax       - single bar > N*ATR climax reversal
 4. TV_Three_White_Soldiers    - 3 strong same-direction bars (body > N*ATR)
 5. TV_Hikkake_FalseBreak      - inside-bar false breakout, fade direction
"""
import numpy as np
import pandas as pd


def _atr(df, n):
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / int(n), adjust=False, min_periods=int(n)).mean()


# ----- 1) ENGULFING REVERSAL -----
def gen_TV_Engulfing_Reversal(df, atr_len=14, body_atr_min=0.5, **kw):
    """Bullish engulfing: prev bar red (close<open), current bar green
    (close>open) AND current body fully engulfs prev body
    (open <= prev_close AND close >= prev_open).
    """
    al = int(atr_len)
    bm = float(body_atr_min)
    o = df['open'].astype(float)
    c = df['close'].astype(float)
    body = (c - o).abs()
    atr = _atr(df, al)
    big_body = body > bm * atr
    prev_o = o.shift(1)
    prev_c = c.shift(1)
    prev_red = prev_c < prev_o
    prev_green = prev_c > prev_o
    bull_eng = (c > o) & prev_red & (o <= prev_c) & (c >= prev_o) & big_body
    bear_eng = (c < o) & prev_green & (o >= prev_c) & (c <= prev_o) & big_body
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[bull_eng.shift(1).fillna(False).astype(bool)] = 1
    sig[bear_eng.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Engulfing_Reversal():
    return {
        'atr_len': ('int', 7, 30),
        'body_atr_min': ('float', 0.3, 1.5),
    }


# ----- 2) PIN BAR REVERSAL -----
def gen_TV_PinBar_Reversal(df, wick_to_body_ratio=2.0, atr_len=14, **kw):
    """Bullish pin bar: long lower wick, small body in upper third of bar.
       lower_wick = min(open,close) - low
       upper_wick = high - max(open,close)
       body = |close - open|
    Bullish: lower_wick >= ratio * body AND lower_wick > upper_wick.
    Bearish: upper_wick >= ratio * body AND upper_wick > lower_wick.
    """
    r = float(wick_to_body_ratio)
    al = int(atr_len)
    o = df['open'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    body = (c - o).abs().replace(0.0, np.nan)
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
    atr = _atr(df, al)
    valid_size = (h - l) > 0.5 * atr  # avoid tiny bars
    bullish_pin = (lower_wick >= r * body) & (lower_wick > upper_wick) & valid_size
    bearish_pin = (upper_wick >= r * body) & (upper_wick > lower_wick) & valid_size
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[bullish_pin.shift(1).fillna(False).astype(bool)] = 1
    sig[bearish_pin.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_PinBar_Reversal():
    return {
        'wick_to_body_ratio': ('float', 1.5, 4.0),
        'atr_len': ('int', 7, 30),
    }


# ----- 3) WIDE RANGE CLIMAX -----
def gen_TV_Wide_Range_Climax(df, atr_len=14, range_mult=2.5, **kw):
    """Single bar with range > range_mult * ATR = climax move.
    Direction = OPPOSITE of bar's close vs open (mean-revert climax).
    """
    al = int(atr_len)
    rm = float(range_mult)
    o = df['open'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    bar_range = h - l
    atr = _atr(df, al)
    climax = bar_range > rm * atr
    long_setup = climax & (c < o)  # big down bar -> long reversal
    short_setup = climax & (c > o)  # big up bar -> short reversal
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_setup.shift(1).fillna(False).astype(bool)] = 1
    sig[short_setup.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Wide_Range_Climax():
    return {
        'atr_len': ('int', 7, 30),
        'range_mult': ('float', 1.8, 4.0),
    }


# ----- 4) THREE WHITE SOLDIERS / BLACK CROWS -----
def gen_TV_Three_White_Soldiers(df, atr_len=14, body_atr_min=0.5, **kw):
    """Three white soldiers: 3 consecutive green bars where each opens
    within prior body and closes higher AND body > body_atr_min * ATR.
    Three black crows = symmetric.
    """
    al = int(atr_len)
    bm = float(body_atr_min)
    o = df['open'].astype(float)
    c = df['close'].astype(float)
    body = (c - o).abs()
    atr = _atr(df, al)
    green = c > o
    red = c < o
    big_body = body > bm * atr
    # Each bar opens within prior bar's body
    prev_o = o.shift(1)
    prev_c = c.shift(1)
    prev_low = pd.Series(np.minimum(prev_o.to_numpy(), prev_c.to_numpy()), index=o.index)
    prev_high = pd.Series(np.maximum(prev_o.to_numpy(), prev_c.to_numpy()), index=o.index)
    open_in_prev_body = (o > prev_low) & (o < prev_high)
    higher_close = c > c.shift(1)
    lower_close = c < c.shift(1)
    soldier = green & big_body & open_in_prev_body & higher_close
    crow = red & big_body & open_in_prev_body & lower_close
    three_soldiers = soldier & soldier.shift(1).fillna(False) & soldier.shift(2).fillna(False)
    three_crows = crow & crow.shift(1).fillna(False) & crow.shift(2).fillna(False)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[three_soldiers.shift(1).fillna(False).astype(bool)] = 1
    sig[three_crows.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Three_White_Soldiers():
    return {
        'atr_len': ('int', 7, 30),
        'body_atr_min': ('float', 0.3, 1.0),
    }


# ----- 5) HIKKAKE FALSE BREAK -----
def gen_TV_Hikkake_FalseBreak(df, **kw):
    """Hikkake (Daniel Chesler) - 4-bar pattern:
    Bar 1: any (mother bar)
    Bar 2: inside bar (high<=bar1.high AND low>=bar1.low)
    Bar 3: breaks above OR below bar2's high/low
    Bar 4-5: closes back inside bar2's range (false break) -> trade opposite
    Long entry: bar3 broke DOWN, then bar4 closes ABOVE bar2.low (bull hikkake)
    Short entry: bar3 broke UP, then bar4 closes BELOW bar2.high
    """
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    inside_2 = (h.shift(2) <= h.shift(3)) & (l.shift(2) >= l.shift(3))
    broke_down_3 = inside_2 & (l.shift(1) < l.shift(2))
    broke_up_3 = inside_2 & (h.shift(1) > h.shift(2))
    bull_hikkake = broke_down_3 & (c > l.shift(2))  # current closed back above
    bear_hikkake = broke_up_3 & (c < h.shift(2))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[bull_hikkake.shift(1).fillna(False).astype(bool)] = 1
    sig[bear_hikkake.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Hikkake_FalseBreak():
    # Pattern is rule-based; jitter via dummy param to give random search variation
    return {
        'noop_jitter': ('int', 1, 5),
    }


STRATEGY_EXPORT = {
    "TV_Engulfing_Reversal": {
        "gen": gen_TV_Engulfing_Reversal, "space": space_TV_Engulfing_Reversal,
        "source": "mac_batch3719_engulfing_reversal_pattern",
    },
    "TV_PinBar_Reversal": {
        "gen": gen_TV_PinBar_Reversal, "space": space_TV_PinBar_Reversal,
        "source": "mac_batch3719_pin_bar_long_wick_rejection",
    },
    "TV_Wide_Range_Climax": {
        "gen": gen_TV_Wide_Range_Climax, "space": space_TV_Wide_Range_Climax,
        "source": "mac_batch3719_wide_range_climax_reversal",
    },
    "TV_Three_White_Soldiers": {
        "gen": gen_TV_Three_White_Soldiers,
        "space": space_TV_Three_White_Soldiers,
        "source": "mac_batch3719_3_soldiers_3_crows",
    },
    "TV_Hikkake_FalseBreak": {
        "gen": gen_TV_Hikkake_FalseBreak, "space": space_TV_Hikkake_FalseBreak,
        "source": "mac_batch3719_Chesler_Hikkake_pattern",
    },
}
