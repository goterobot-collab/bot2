"""
Batch 3725 - Mac paralela wave m26 - 5 wizards III + crypto-specific + grammar.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3724.

 1. TV_Minervini_VCP        - Mark Minervini Volatility Contraction Pattern:
                               3+ progressively smaller bar ranges + breakout
                               Source: Minervini "Trade Like a Stock Market Wizard" 2013
 2. TV_RoundNumber_Magnet   - close approaches round-number psychological level
                               (1%, 5%, 10% increments of price magnitude) - NEW
 3. TV_Walton_ExtremeFade   - Stuart Walton-style fade at extreme RSI + ATR
                               (counter-trend at exhaustion)
 4. TV_Calendar_QuarterEnd  - end-of-quarter price action (UTC bar in last 3
                               days of Mar/Jun/Sep/Dec) - NEW (sandbox has
                               weekly/monthly/seasonality, not quarter-end specific)
 5. TV_BarBody_GrammarUUD   - 3-bar grammar: detect specific UUD/DDU/UDU/DUD
                               patterns at recent extremes (NEW)
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _sma(s, n):
    return s.rolling(int(n), min_periods=int(n)).mean()


def _rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(df, n):
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / int(n), adjust=False, min_periods=int(n)).mean()


# ----- 1) MINERVINI VCP -----
def gen_TV_Minervini_VCP(df, contract_bars=3, contract_ratio=0.7,
                          breakout_atr_mult=0.5, atr_len=14, **kw):
    """Volatility Contraction Pattern: 3+ consecutive bars where each bar's
    range < contract_ratio * prior bar's range. Then breakout above the
    pattern's highest high.
    """
    cb = int(contract_bars)
    cr = float(contract_ratio)
    bm = float(breakout_atr_mult)
    al = int(atr_len)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    bar_range = h - l
    contracting = bar_range < cr * bar_range.shift(1)
    contraction_streak = contracting.rolling(cb, min_periods=cb).sum() == cb
    pattern_high = h.shift(1).rolling(cb, min_periods=cb).max()
    pattern_low = l.shift(1).rolling(cb, min_periods=cb).min()
    atr = _atr(df, al)
    long_break = contraction_streak.shift(1).fillna(False) & (c > pattern_high + bm * atr)
    short_break = contraction_streak.shift(1).fillna(False) & (c < pattern_low - bm * atr)
    long_entry = long_break & ~(long_break.shift(1).fillna(False))
    short_entry = short_break & ~(short_break.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Minervini_VCP():
    return {
        'contract_bars': ('int', 2, 5),
        'contract_ratio': ('float', 0.5, 0.85),
        'breakout_atr_mult': ('float', 0.2, 1.5),
        'atr_len': ('int', 7, 30),
    }


# ----- 2) ROUND NUMBER MAGNET -----
def gen_TV_RoundNumber_Magnet(df, atr_len=14, near_atr_pct=30, **kw):
    """Round-number levels at order-of-magnitude steps:
    For price P, level_step = 10^floor(log10(P)) / 10
    e.g. P=2350 -> step=100; P=42 -> step=1; P=0.085 -> step=0.001
    Long entry: low approaches round level from above, close > level
    Short entry: high approaches round level from below, close < level
    near = within near_atr_pct% of ATR
    """
    al = int(atr_len)
    nap = float(near_atr_pct) / 100.0
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    log_price = np.log10(c.replace(0.0, np.nan))
    level_step = 10 ** (np.floor(log_price - 1))  # 10x smaller than magnitude
    nearest_level = (c / level_step).round() * level_step
    atr = _atr(df, al)
    near_threshold = nap * atr
    bounce_off_low = (l <= nearest_level + near_threshold) & (l >= nearest_level) & (c > nearest_level + near_threshold)
    bounce_off_high = (h >= nearest_level - near_threshold) & (h <= nearest_level) & (c < nearest_level - near_threshold)
    long_entry = bounce_off_low & ~(bounce_off_low.shift(1).fillna(False))
    short_entry = bounce_off_high & ~(bounce_off_high.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_RoundNumber_Magnet():
    return {
        'atr_len': ('int', 7, 30),
        'near_atr_pct': ('int', 10, 60),
    }


# ----- 3) WALTON EXTREME FADE -----
def gen_TV_Walton_ExtremeFade(df, rsi_len=14, rsi_extreme=85, atr_len=14,
                                atr_mult=2.0, **kw):
    """Stuart Walton fade: at extreme RSI (rsi >= extreme or <= 100-extreme),
    ALSO bar range >= atr_mult * ATR (climax).
    Fade the direction.
    """
    rl = int(rsi_len)
    re = float(rsi_extreme)
    al = int(atr_len)
    am = float(atr_mult)
    o = df['open'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    rsi = _rsi(c, rl)
    bar_range = h - l
    atr = _atr(df, al)
    big_range = bar_range > am * atr
    # Extreme overbought + climax up = SHORT (fade)
    short_setup = (rsi >= re) & big_range & (c > o)
    # Extreme oversold + climax down = LONG (fade)
    long_setup = (rsi <= (100 - re)) & big_range & (c < o)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Walton_ExtremeFade():
    return {
        'rsi_len': ('int', 7, 21),
        'rsi_extreme': ('int', 75, 92),
        'atr_len': ('int', 7, 25),
        'atr_mult': ('float', 1.5, 3.5),
    }


# ----- 4) CALENDAR QUARTER-END -----
def gen_TV_Calendar_QuarterEnd(df, days_window=3, ret_len=10, **kw):
    """End-of-quarter window: last `days_window` days of Mar/Jun/Sep/Dec.
    During window, fade recent N-day return (institutional rebalancing -> reversal).
    """
    dw = int(days_window)
    rl = int(ret_len)
    c = df['close'].astype(float)
    # df has DatetimeIndex via canary_runner load_candles
    idx = df.index
    is_qend_month = idx.month.isin([3, 6, 9, 12])
    days_in_month = idx.days_in_month
    is_last_days = (days_in_month - idx.day) < dw
    qend_window = pd.Series(is_qend_month & is_last_days, index=idx)
    ret = (c / c.shift(rl) - 1.0)
    # Fade direction during qend window
    long_setup = qend_window & (ret < -0.05)  # was down -> long fade
    short_setup = qend_window & (ret > 0.05)  # was up -> short fade
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Calendar_QuarterEnd():
    return {
        'days_window': ('int', 1, 7),
        'ret_len': ('int', 5, 30),
    }


# ----- 5) BAR BODY GRAMMAR (3-BAR PATTERNS) -----
def gen_TV_BarBody_GrammarUUD(df, win=20, **kw):
    """3-bar grammar coding: U if close>open, D if close<open, S if equal.
    At recent extreme (close near hh or ll over win):
       - "UUD" at top = potential reversal short
       - "DDU" at bottom = potential reversal long
       - "UDU" / "DUD" choppy = avoid (no signal)
    """
    w = int(win)
    o = df['open'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    is_u = c > o
    is_d = c < o
    # 3-bar pattern check
    bar3_u = is_u.shift(2).fillna(False)
    bar2_u = is_u.shift(1).fillna(False)
    bar1_u = is_u.fillna(False)
    bar3_d = is_d.shift(2).fillna(False)
    bar2_d = is_d.shift(1).fillna(False)
    bar1_d = is_d.fillna(False)
    pattern_uud = bar3_u & bar2_u & bar1_d
    pattern_ddu = bar3_d & bar2_d & bar1_u
    rolling_high = h.rolling(w, min_periods=w).max()
    rolling_low = l.rolling(w, min_periods=w).min()
    near_top = c > rolling_high * 0.98
    near_bottom = c < rolling_low * 1.02
    short_setup = pattern_uud & near_top
    long_setup = pattern_ddu & near_bottom
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_setup.shift(1).fillna(False).astype(bool)] = 1
    sig[short_setup.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_BarBody_GrammarUUD():
    return {
        'win': ('int', 10, 60),
    }


STRATEGY_EXPORT = {
    "TV_Minervini_VCP": {
        "gen": gen_TV_Minervini_VCP, "space": space_TV_Minervini_VCP,
        "source": "mac_batch3725_Minervini_VCP_2013",
    },
    "TV_RoundNumber_Magnet": {
        "gen": gen_TV_RoundNumber_Magnet, "space": space_TV_RoundNumber_Magnet,
        "source": "mac_batch3725_round_number_psychological_level",
    },
    "TV_Walton_ExtremeFade": {
        "gen": gen_TV_Walton_ExtremeFade, "space": space_TV_Walton_ExtremeFade,
        "source": "mac_batch3725_Walton_extreme_RSI_ATR_climax_fade",
    },
    "TV_Calendar_QuarterEnd": {
        "gen": gen_TV_Calendar_QuarterEnd, "space": space_TV_Calendar_QuarterEnd,
        "source": "mac_batch3725_quarter_end_institutional_fade",
    },
    "TV_BarBody_GrammarUUD": {
        "gen": gen_TV_BarBody_GrammarUUD, "space": space_TV_BarBody_GrammarUUD,
        "source": "mac_batch3725_3bar_grammar_UUD_DDU_at_extreme",
    },
}
