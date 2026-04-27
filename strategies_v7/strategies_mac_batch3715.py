"""
Batch 3715 - Mac paralela wave m16 - 5 microstructure / liquidity / pivot families.

All vectorized, low-param (1-3 each), pickle-safe, signal int {-1,0,1},
.shift(1). ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3714.

 1. TV_Roll_SpreadEstimator    - Roll 1984 effective spread proxy =
                                  2*sqrt(-cov(Δp_t, Δp_{t-1})). High spread
                                  -> avoid trade. Low spread -> momentum entry.
                                  Source: Roll JF 1984
 2. TV_Amihud_Illiquidity       - Amihud ILLIQ = |return| / dollar_volume.
                                  Cross threshold = liquidity regime change.
                                  Source: Amihud JFM 2002
 3. TV_Fibonacci_Retracement   - Last swing high/low + 38.2/61.8 retracement
                                  bounce entry (NEW, not in repo)
 4. TV_DeMark_TD9_Sequential    - Tom DeMark TD Sequential setup: 9 consec
                                  closes lower than 4-bar-ago close = exhaustion
                                  Source: DeMark "New Market Timing Techniques"
 5. TV_PriceImpact_Composite   - composite |return|/volume + ATR/close
                                  z-score regime
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


# ----- 1) ROLL SPREAD ESTIMATOR -----
def gen_TV_Roll_SpreadEstimator(df, win=30, low_pct=30, **kw):
    """Roll 1984: spread = 2 * sqrt(-cov(dp_t, dp_{t-1}))
    Vectorized: cov ≈ rolling mean(dp_t * dp_{t-1}) - mean(dp_t)*mean(dp_{t-1})
    Low spread (lower percentile) = good liquidity = trade with momentum sign.
    """
    w = int(win)
    lp = float(low_pct)
    c = df['close'].astype(float)
    dp = c.diff()
    dp_prev = dp.shift(1)
    mu1 = dp.rolling(w, min_periods=w).mean()
    mu2 = dp_prev.rolling(w, min_periods=w).mean()
    cov = (dp * dp_prev).rolling(w, min_periods=w).mean() - mu1 * mu2
    spread = 2 * np.sqrt(np.maximum(-cov, 0.0))  # zero out positive cov
    spread_rank = spread.rolling(w * 3, min_periods=w * 3).rank(pct=True) * 100
    low_spread = spread_rank < lp
    short_ret = dp.rolling(5, min_periods=5).sum()
    long_setup = low_spread & (short_ret > 0)
    short_setup = low_spread & (short_ret < 0)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Roll_SpreadEstimator():
    return {
        'win': ('int', 15, 60),
        'low_pct': ('int', 15, 45),
    }


# ----- 2) AMIHUD ILLIQUIDITY -----
def gen_TV_Amihud_Illiquidity(df, win=20, smooth=10, change_pct=50, **kw):
    """Amihud: ILLIQ = |return| / (price * volume). High ILLIQ = thin mkt.
    Use ILLIQ percentile rank: regime change cross = entry signal.
    Direction by recent return sign.
    """
    w = int(win)
    sm = int(smooth)
    cp = float(change_pct)
    c = df['close'].astype(float)
    v = df['volume'].astype(float).replace(0.0, np.nan)
    ret = c.pct_change()
    illiq = ret.abs() / (c * v)
    illiq_smooth = _sma(illiq, sm)
    illiq_rank = illiq_smooth.rolling(w * 5, min_periods=w * 3).rank(pct=True) * 100
    rank_change_up = (illiq_rank > cp) & (illiq_rank.shift(1) <= cp)
    rank_change_dn = (illiq_rank < (100 - cp)) & (illiq_rank.shift(1) >= (100 - cp))
    short_ret = ret.rolling(5, min_periods=5).sum()
    # When illiq rises (mkt thinning) -> mean revert
    # When illiq drops (liquidity returning) -> follow trend
    long_setup = (rank_change_dn & (short_ret > 0)) | (rank_change_up & (short_ret < 0))
    short_setup = (rank_change_dn & (short_ret < 0)) | (rank_change_up & (short_ret > 0))
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Amihud_Illiquidity():
    return {
        'win': ('int', 10, 50),
        'smooth': ('int', 5, 20),
        'change_pct': ('int', 40, 75),
    }


# ----- 3) FIBONACCI RETRACEMENT BOUNCE -----
def gen_TV_Fibonacci_Retracement(df, swing_len=30, fib_level=0.618, atr_tol=0.5, **kw):
    """Find recent swing high (highest high in window) and swing low.
    Fib level = swing_low + fib_level * (swing_high - swing_low).
    Long bounce: in uptrend, close pulls back to fib level (within atr_tol*ATR)
                 then closes above. Symmetric for short.
    """
    sl = int(swing_len)
    fl = float(fib_level)
    at = float(atr_tol)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    swing_high = h.rolling(sl, min_periods=sl).max()
    swing_low = l.rolling(sl, min_periods=sl).min()
    rng = swing_high - swing_low
    fib_long = swing_low + fl * rng
    fib_short = swing_high - fl * rng
    atr = _atr(df, 14)
    # Uptrend = close > middle of recent swing
    mid = (swing_high + swing_low) / 2.0
    in_uptrend = c > mid
    in_downtrend = c < mid
    near_fib_long = (c - fib_long).abs() < at * atr
    near_fib_short = (c - fib_short).abs() < at * atr
    long_bounce = in_uptrend & near_fib_long & (c > c.shift(1)) & (c.shift(1) <= fib_long.shift(1))
    short_bounce = in_downtrend & near_fib_short & (c < c.shift(1)) & (c.shift(1) >= fib_short.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_bounce.shift(1).fillna(False).astype(bool)] = 1
    sig[short_bounce.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Fibonacci_Retracement():
    return {
        'swing_len': ('int', 20, 80),
        'fib_level': ('float', 0.382, 0.786),
        'atr_tol': ('float', 0.2, 1.5),
    }


# ----- 4) DEMARK TD9 SEQUENTIAL -----
def gen_TV_DeMark_TD9_Sequential(df, lookback=4, count_target=9, **kw):
    """DeMark TD Sequential setup:
    - Buy setup: count consecutive bars where close < close[lookback bars ago].
      When count reaches 9 -> exhaustion -> potential reversal LONG.
    - Sell setup: 9 consecutive closes > close[lookback ago] -> SHORT.
    """
    lb = int(lookback)
    ct = int(count_target)
    c = df['close'].astype(float)
    is_lower = (c < c.shift(lb)).astype(int)
    is_higher = (c > c.shift(lb)).astype(int)
    # Streak of lower closes (resets on first non-lower)
    # Vectorized cumcount where condition resets
    grp_lower = (is_lower != is_lower.shift(1)).cumsum()
    streak_lower = is_lower.groupby(grp_lower).cumsum()
    grp_higher = (is_higher != is_higher.shift(1)).cumsum()
    streak_higher = is_higher.groupby(grp_higher).cumsum()
    long_trigger = (streak_lower == ct) & (is_lower == 1)
    short_trigger = (streak_higher == ct) & (is_higher == 1)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_trigger.shift(1).fillna(False).astype(bool)] = 1
    sig[short_trigger.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_DeMark_TD9_Sequential():
    return {
        'lookback': ('int', 3, 6),
        'count_target': ('int', 7, 13),
    }


# ----- 5) PRICE IMPACT COMPOSITE -----
def gen_TV_PriceImpact_Composite(df, win=20, z_thr=1.5, **kw):
    """Composite price impact = |return|/volume + ATR/close (both z-scored).
    Z-score sum > thr -> high impact bar -> often reverses.
    Direction = opposite of bar's close direction.
    """
    w = int(win)
    th = float(z_thr)
    c = df['close'].astype(float)
    v = df['volume'].astype(float).replace(0.0, np.nan)
    ret = c.pct_change()
    impact1 = ret.abs() / v
    atr = _atr(df, w)
    impact2 = atr / c
    z1 = (impact1 - impact1.rolling(w, min_periods=w).mean()) / impact1.rolling(w, min_periods=w).std(ddof=0).replace(0.0, np.nan)
    z2 = (impact2 - impact2.rolling(w, min_periods=w).mean()) / impact2.rolling(w, min_periods=w).std(ddof=0).replace(0.0, np.nan)
    composite = z1 + z2
    surge = (composite > 2 * th) & (composite.shift(1) <= 2 * th)
    long_setup = surge & (ret < 0)  # impact + down -> reversal up
    short_setup = surge & (ret > 0)  # impact + up -> reversal down
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_setup.shift(1).fillna(False).astype(bool)] = 1
    sig[short_setup.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_PriceImpact_Composite():
    return {
        'win': ('int', 10, 40),
        'z_thr': ('float', 1.0, 3.0),
    }


STRATEGY_EXPORT = {
    "TV_Roll_SpreadEstimator": {
        "gen": gen_TV_Roll_SpreadEstimator,
        "space": space_TV_Roll_SpreadEstimator,
        "source": "mac_batch3715_Roll_JF_1984",
    },
    "TV_Amihud_Illiquidity": {
        "gen": gen_TV_Amihud_Illiquidity,
        "space": space_TV_Amihud_Illiquidity,
        "source": "mac_batch3715_Amihud_JFM_2002",
    },
    "TV_Fibonacci_Retracement": {
        "gen": gen_TV_Fibonacci_Retracement,
        "space": space_TV_Fibonacci_Retracement,
        "source": "mac_batch3715_classical_Fibonacci",
    },
    "TV_DeMark_TD9_Sequential": {
        "gen": gen_TV_DeMark_TD9_Sequential,
        "space": space_TV_DeMark_TD9_Sequential,
        "source": "mac_batch3715_DeMark_TD_Sequential",
    },
    "TV_PriceImpact_Composite": {
        "gen": gen_TV_PriceImpact_Composite,
        "space": space_TV_PriceImpact_Composite,
        "source": "mac_batch3715_price_impact_zscore_composite",
    },
}
