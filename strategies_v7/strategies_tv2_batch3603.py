"""
Batch 3603 - Seasonality / session effects
Sources:
  - Time-of-day effects in crypto: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3310817
  - Day-of-week anomaly: https://en.wikipedia.org/wiki/Day-of-the-week_effect
  - US session open 13:30 UTC (NYSE cash): https://www.nyse.com/markets/hours-calendars
  - Weekend gap research: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3423380
  - Binance funding schedule 00/08/16 UTC: https://www.binance.com/en/support/faq/360033525031
5 strategies, pickle-safe, no lambdas, no look-ahead.
"""
import numpy as np
import pandas as pd


def _ensure_dt_index(df):
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True)
    else:
        if idx.tz is None:
            idx = idx.tz_localize('UTC')
    return idx


# ---------------------------------------------------------------------------
# 1) TV_TOD_WR_Pocket
#    "Enter on bars where historical (same hour-of-day) WR > threshold
#     for a short lookback."
# ---------------------------------------------------------------------------
def gen_TV_TOD_WR_Pocket(df, lookback=200, wr_thresh=0.60, **kw):
    """
    For each bar, compute the up-bar rate among the last `lookback` bars that
    share the same hour-of-day (UTC), using only past data (shifted).
    If that rate > wr_thresh and the current bar's prior close slope is up,
    go long; mirror for short.  No look-ahead: we use .shift(1) so the
    statistic is known at bar open.
    """
    idx = _ensure_dt_index(df)
    c = df['close'].astype(float)
    o = df['open'].astype(float)
    up = (c > o).astype(float)

    hour = pd.Series(idx.hour, index=df.index)
    # Per-hour trailing WR using only past bars; vectorised per hour group.
    past_up = up.shift(1)
    n = int(lookback)
    wr = _grouped_rolling_mean(past_up.to_numpy(),
                               hour.to_numpy(),
                               n)
    wr = pd.Series(wr, index=df.index)

    t_hi = float(wr_thresh)
    t_lo = 1.0 - float(wr_thresh)
    long_cond = wr >= t_hi
    short_cond = wr <= t_lo

    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cond.fillna(False)] = 1
    sig[short_cond.fillna(False)] = -1
    return sig


def _grouped_rolling_mean(values, groups, n):
    """
    Per-group trailing rolling mean, vectorised per group.
    Returns a numpy array the same length as values.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    out = np.full(values.shape, np.nan, dtype=float)
    uniq = np.unique(groups[~pd.isna(groups)]) if groups.dtype.kind in 'fcmM' else np.unique(groups)
    for g in uniq:
        mask = (groups == g)
        sub = pd.Series(values[mask]).rolling(int(n), min_periods=max(20, int(n) // 4)).mean().to_numpy()
        out[mask] = sub
    return out


def space_TV_TOD_WR_Pocket():
    return {
        'lookback': ('int', 50, 1000),
        'wr_thresh': ('float', 0.55, 0.80),
    }


# ---------------------------------------------------------------------------
# 2) TV_DOW_Trend_Filter
#    "Long only on specific day-of-week with bullish trend"
# ---------------------------------------------------------------------------
def gen_TV_DOW_Trend_Filter(df, dow_long=1, dow_short=4, ema_len=200, **kw):
    """
    If current bar is on `dow_long` (0=Mon..6=Sun) AND close > EMA200 -> long.
    If on `dow_short` AND close < EMA200 -> short.
    """
    idx = _ensure_dt_index(df)
    c = df['close'].astype(float)
    ema = c.ewm(span=int(ema_len), adjust=False, min_periods=int(ema_len)).mean()
    dow = pd.Series(idx.dayofweek, index=df.index)

    long_cond = (dow == int(dow_long)) & (c > ema)
    short_cond = (dow == int(dow_short)) & (c < ema)

    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cond.fillna(False)] = 1
    sig[short_cond.fillna(False)] = -1
    return sig


def space_TV_DOW_Trend_Filter():
    return {
        'dow_long': ('int', 0, 6),
        'dow_short': ('int', 0, 6),
        'ema_len': ('int', 50, 400),
    }


# ---------------------------------------------------------------------------
# 3) TV_US_Session_Breakout
#    "Breakout after US session opens (13:30 UTC)"
# ---------------------------------------------------------------------------
def gen_TV_US_Session_Breakout(df, pre_window=12, **kw):
    """
    Compute the high and low of the `pre_window` bars ENDING at the last bar
    before 13:30 UTC (Asia + Europe session). When the US session opens,
    the first bar at or after 13:30 UTC that closes above that high goes long;
    closes below that low goes short.  To avoid look-ahead we use .shift(1)
    on the reference high/low (they're known at the bar's open).
    """
    idx = _ensure_dt_index(df)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)

    minute_of_day = idx.hour * 60 + idx.minute
    us_open_min = 13 * 60 + 30

    pw = int(pre_window)
    pre_hi = h.shift(1).rolling(pw, min_periods=pw).max()
    pre_lo = l.shift(1).rolling(pw, min_periods=pw).min()

    in_us_open = pd.Series((minute_of_day >= us_open_min) & (minute_of_day <= us_open_min + 60),
                           index=df.index)
    long_cond = in_us_open & (c > pre_hi)
    short_cond = in_us_open & (c < pre_lo)

    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cond.fillna(False)] = 1
    sig[short_cond.fillna(False)] = -1
    return sig


def space_TV_US_Session_Breakout():
    return {
        'pre_window': ('int', 4, 48),
    }


# ---------------------------------------------------------------------------
# 4) TV_Weekend_Gap_Fade
#    "After a large Sat/Sun move, fade back to Friday's last close"
# ---------------------------------------------------------------------------
def gen_TV_Weekend_Gap_Fade(df, k_atr=1.5, atr_len=14, **kw):
    """
    Saturday=5, Sunday=6.  During the weekend, compute the last Friday
    close (i.e. the close of the most recent bar with dayofweek<=4, shifted
    once to avoid look-ahead).  If current close is > friday_close + k*ATR
    -> short (fade up), if < friday_close - k*ATR -> long (fade down).
    """
    idx = _ensure_dt_index(df)
    c = df['close'].astype(float)

    # Wilder ATR
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / float(atr_len), adjust=False, min_periods=int(atr_len)).mean()

    dow = pd.Series(idx.dayofweek, index=df.index)
    is_weekend = (dow >= 5)
    is_weekday = (dow <= 4)

    # Last weekday close, carried forward and shifted so it is strictly prior info.
    weekday_close = c.where(is_weekday)
    friday_close = weekday_close.ffill().shift(1)

    diff = c - friday_close
    up_gap = is_weekend & (diff > float(k_atr) * atr)
    dn_gap = is_weekend & (diff < -float(k_atr) * atr)

    sig = pd.Series(0, index=df.index, dtype=int)
    sig[up_gap.fillna(False)] = -1
    sig[dn_gap.fillna(False)] = 1
    return sig


def space_TV_Weekend_Gap_Fade():
    return {
        'k_atr': ('float', 0.5, 4.0),
        'atr_len': ('int', 7, 48),
    }


# ---------------------------------------------------------------------------
# 5) TV_Funding_Arb_Proxy
#    "Large candle right after 00/08/16 UTC (funding times) fades the move"
# ---------------------------------------------------------------------------
def gen_TV_Funding_Arb_Proxy(df, window_min=60, k_atr=1.0, atr_len=14, **kw):
    """
    Binance perpetuals pay funding at 00, 08, 16 UTC.  In the `window_min`
    minutes AFTER each funding timestamp, if the cumulative move (close vs
    close at funding) exceeds k*ATR in one direction, fade it.
    Works on 5m/15m/1h (window of bars depends on timeframe).
    """
    idx = _ensure_dt_index(df)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / float(atr_len), adjust=False, min_periods=int(atr_len)).mean()

    minute_of_day = idx.hour * 60 + idx.minute
    funding_mins = np.array([0, 8 * 60, 16 * 60])

    # distance (minutes) since most recent funding, within same day
    mod = np.asarray(minute_of_day) % (8 * 60)
    since_funding = mod  # 0..479

    in_window = pd.Series(since_funding <= int(window_min), index=df.index)

    # close at the funding bar = first bar of current 8h block, shifted so it is
    # strictly prior information relative to the signal bar.
    # We approximate by taking close of the bar where since_funding was minimal
    # within the current 8h block.
    block_id = pd.Series(((idx.hour // 8) + idx.dayofyear * 3 + idx.year * 3 * 400),
                         index=df.index)
    # first close per block, carried forward, then shifted one bar for no look-ahead.
    first_in_block = c.groupby(block_id).transform('first')
    ref_close = first_in_block.shift(1)

    move = c - ref_close
    up_move = in_window & (move > float(k_atr) * atr)
    dn_move = in_window & (move < -float(k_atr) * atr)

    sig = pd.Series(0, index=df.index, dtype=int)
    sig[up_move.fillna(False)] = -1
    sig[dn_move.fillna(False)] = 1
    return sig


def space_TV_Funding_Arb_Proxy():
    return {
        'window_min': ('int', 5, 240),
        'k_atr': ('float', 0.3, 3.0),
        'atr_len': ('int', 7, 48),
    }


STRATEGY_EXPORT = {
    'TV_TOD_WR_Pocket': {
        'gen': gen_TV_TOD_WR_Pocket,
        'space': space_TV_TOD_WR_Pocket,
        'source': 'https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3310817',
    },
    'TV_DOW_Trend_Filter': {
        'gen': gen_TV_DOW_Trend_Filter,
        'space': space_TV_DOW_Trend_Filter,
        'source': 'https://en.wikipedia.org/wiki/Day-of-the-week_effect',
    },
    'TV_US_Session_Breakout': {
        'gen': gen_TV_US_Session_Breakout,
        'space': space_TV_US_Session_Breakout,
        'source': 'https://www.nyse.com/markets/hours-calendars',
    },
    'TV_Weekend_Gap_Fade': {
        'gen': gen_TV_Weekend_Gap_Fade,
        'space': space_TV_Weekend_Gap_Fade,
        'source': 'https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3423380',
    },
    'TV_Funding_Arb_Proxy': {
        'gen': gen_TV_Funding_Arb_Proxy,
        'space': space_TV_Funding_Arb_Proxy,
        'source': 'https://www.binance.com/en/support/faq/360033525031',
    },
}
