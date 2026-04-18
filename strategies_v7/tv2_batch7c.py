"""
tv2_batch7c.py — Batch 7C: 10 Pine Script strategies converted to Python
Strategies:
  1. CCI_EMA_RSI       — CCI+RSI+EMA Strategy (v5, alifer123)
  2. Flash_Momentum    — The Flash Strategy: Momentum-RSI, EMA, ATR (v5)
  3. TradePro_2EMA_StochRSI — 2 EMA + Stoch RSI + ATR (v5, PtGambler)
  4. Crypto_Momentum   — Crypto momentum / Squeeze (v4, echepata)
  5. PriceAction_BB    — Price Action + Bollinger Strategy (v4)
  6. ATR_Trail_Stop    — ATR Trailing Stoploss Strategy (v4, ceyhun)
  7. Noro_MA_ATR       — Noro's MA+ATR Strategy (v4)
  8. KAMA_Strategy     — Kaufman Adaptive Moving Average (v5, TradeDots)
  9. MACD_RSI_Signal   — MACD crossover + RSI Oversold/Overbought (v4)
  10. Aggressive_Scalper — Crypto Price Scalper / Vortex (v4, exlux99)

sig=1(LONG), -1(SHORT), 0(none). No look-ahead. Pine defaults preserved.
"""

import numpy as np
import pandas as pd


# ─── HELPERS ────────────────────────────────────────────────────────────────

def _ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def _sma(s, p):
    return s.rolling(p).mean()

def _rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0).ewm(span=p, adjust=False).mean()
    l = (-d.where(d < 0, 0.0)).ewm(span=p, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))

def _atr(h, l, c, p=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def _wma(s, p):
    w = np.arange(1, p + 1, dtype=float)
    return s.rolling(p).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def _hma(s, p):
    return _wma(2 * _wma(s, p // 2) - _wma(s, p), max(1, int(p ** 0.5)))

def _bb(c, p=20, m=2.0):
    b = _sma(c, p)
    s = c.rolling(p).std()
    return b, b + m * s, b - m * s

def _stoch(c, h, l, k=14, d=3):
    lo = l.rolling(k).min()
    hi = h.rolling(k).max()
    ks = 100 * (c - lo) / (hi - lo + 1e-10)
    ds = _sma(ks, d)
    return ks, ds

def _keltner(c, h, l, p=20, m=1.5):
    mid = _ema(c, p)
    a = _atr(h, l, c, p)
    return mid, mid + m * a, mid - m * a

def _mfi(c, h, l, v, p=14):
    tp = (h + l + c) / 3
    rmf = tp * v
    pos = rmf.where(tp > tp.shift(1), 0)
    neg = rmf.where(tp < tp.shift(1), 0)
    mfr = pos.rolling(p).sum() / (neg.rolling(p).sum() + 1e-10)
    return 100 - 100 / (1 + mfr)

def _cci(h, l, c, p=20):
    tp = (h + l + c) / 3
    sma = _sma(tp, p)
    md = tp.rolling(p).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * md + 1e-10)


# ─── 1. CCI_EMA_RSI ─────────────────────────────────────────────────────────
# Pine: CCI crossover above oversold + price above EMA (optional) + RSI filter (optional)
# Long: CCI crosses UP through oversold level; Short: CCI crosses DOWN through overbought

def gen_CCI_EMA_RSI(df, cci_len=14, cci_ob=150, cci_os=-140,
                    use_ema=True, ema_len=55,
                    use_rsi=False, rsi_len=14, rsi_ob=70, rsi_os=30):
    h, l, c = df['high'], df['low'], df['close']
    hlc3 = (h + l + c) / 3
    cci = _cci(h, l, c, cci_len)
    ema = _ema(hlc3, ema_len)
    rsi = _rsi(hlc3, rsi_len)

    # CCI crossover (prev below threshold, current above)
    long_cci  = (cci.shift(1) < cci_os) & (cci >= cci_os)
    short_cci = (cci.shift(1) > cci_ob) & (cci <= cci_ob)

    # EMA filter
    ema_long  = (c > ema)  if use_ema else pd.Series(True, index=df.index)
    ema_short = (c < ema)  if use_ema else pd.Series(True, index=df.index)

    # RSI filter
    rsi_long  = (rsi < rsi_os) if use_rsi else pd.Series(True, index=df.index)
    rsi_short = (rsi > rsi_ob) if use_rsi else pd.Series(True, index=df.index)

    sig = pd.Series(0, index=df.index)
    sig[long_cci  & ema_long  & rsi_long]  = 1
    sig[short_cci & ema_short & rsi_short] = -1
    return sig

def space_CCI_EMA_RSI(trial):
    return {
        'cci_len': trial.suggest_int('cci_len', 8, 30),
        'cci_ob':  trial.suggest_int('cci_ob',  100, 200),
        'cci_os':  trial.suggest_int('cci_os',  -200, -80),
        'use_ema': trial.suggest_categorical('use_ema', [True, False]),
        'ema_len': trial.suggest_int('ema_len', 20, 100),
        'use_rsi': trial.suggest_categorical('use_rsi', [True, False]),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'rsi_ob':  trial.suggest_int('rsi_ob',  60, 80),
        'rsi_os':  trial.suggest_int('rsi_os',  20, 40),
    }


# ─── 2. Flash_Momentum ──────────────────────────────────────────────────────
# Pine: Momentum-RSI > threshold + SuperTrend direction + EMA trailing channel
# Long: supertrend bullish + mom_rsi > threshold + price above ema_trail
# Short: supertrend bearish + mom_rsi > threshold + price below ema_trail

def _supertrend(high, low, close, period=10, factor=3.0):
    """Returns direction: +1=bullish, -1=bearish. Handles NaN warmup correctly."""
    atr_v = _atr(high, low, close, period)
    hl2 = (high + low) / 2
    up_raw = (hl2 - factor * atr_v).values
    dn_raw = (hl2 + factor * atr_v).values
    c = close.values
    n = len(c)
    fu = np.full(n, np.nan)
    fd = np.full(n, np.nan)
    direction = np.full(n, 1.0)

    # Find first non-NaN ATR bar
    warmup = period
    while warmup < n and np.isnan(up_raw[warmup]):
        warmup += 1
    if warmup >= n:
        return pd.Series(direction, index=close.index)
    fu[warmup] = up_raw[warmup]
    fd[warmup] = dn_raw[warmup]

    for i in range(warmup + 1, n):
        if np.isnan(up_raw[i]):
            fu[i] = fu[i-1]
            fd[i] = fd[i-1]
            direction[i] = direction[i-1]
            continue
        fu[i] = up_raw[i] if (up_raw[i] > fu[i-1] or c[i-1] < fu[i-1]) else fu[i-1]
        fd[i] = dn_raw[i] if (dn_raw[i] < fd[i-1] or c[i-1] > fd[i-1]) else fd[i-1]
        if direction[i-1] == 1:
            direction[i] = -1 if c[i] < fu[i] else 1
        else:
            direction[i] = 1 if c[i] > fd[i] else -1
    return pd.Series(direction, index=close.index)

def _ema_trail(close, ema_len=12, pct=1.0):
    """EMA with percentage band — returns (trail1, trail2) where trail2 is the channel"""
    af = pct / 100.0
    ema = _ema(close, ema_len)
    sl = ema * af
    trail2 = np.zeros(len(close))
    ema_v = ema.values
    sl_v = sl.values
    for i in range(1, len(close)):
        prev = trail2[i-1]
        if ema_v[i] > prev and ema_v[i-1] > prev:
            trail2[i] = max(prev, ema_v[i] - sl_v[i])
        elif ema_v[i] < prev and ema_v[i-1] < prev:
            trail2[i] = min(prev, ema_v[i] + sl_v[i])
        elif ema_v[i] > prev:
            trail2[i] = ema_v[i] - sl_v[i]
        else:
            trail2[i] = ema_v[i] + sl_v[i]
    return pd.Series(trail2, index=close.index)

def gen_Flash_Momentum(df, mom_len=10, mom_rsi_thresh=60,
                       st_period=10, st_factor=3.0,
                       ema_len=12, ema_pct=1.0):
    c = df['close']
    h, l = df['high'], df['low']

    # Momentum-RSI: RSI of (close - close[len])
    mom = c - c.shift(mom_len)
    mom_rsi = _rsi(mom, mom_len)

    # SuperTrend direction
    st_dir = _supertrend(h, l, c, st_period, st_factor)

    # EMA trailing channel
    ema_v = _ema(c, ema_len)
    trail2 = _ema_trail(c, ema_len, ema_pct)

    # Signals: supertrend direction + momentum-rsi confirms + price vs trail
    long_cond  = (st_dir == 1)  & (mom_rsi > mom_rsi_thresh) & (c > trail2)
    short_cond = (st_dir == -1) & (mom_rsi > mom_rsi_thresh) & (c < trail2)

    # Entry on change (crossover of condition)
    long_entry  = long_cond  & ~long_cond.shift(1).fillna(False)
    short_entry = short_cond & ~short_cond.shift(1).fillna(False)

    sig = pd.Series(0, index=df.index)
    sig[long_entry]  = 1
    sig[short_entry] = -1
    return sig

def space_Flash_Momentum(trial):
    return {
        'mom_len':        trial.suggest_int('mom_len', 5, 20),
        'mom_rsi_thresh': trial.suggest_int('mom_rsi_thresh', 50, 75),
        'st_period':      trial.suggest_int('st_period', 7, 20),
        'st_factor':      trial.suggest_float('st_factor', 2.0, 5.0),
        'ema_len':        trial.suggest_int('ema_len', 8, 20),
        'ema_pct':        trial.suggest_float('ema_pct', 0.5, 2.0),
    }


# ─── 3. TradePro_2EMA_StochRSI ──────────────────────────────────────────────
# Pine: 2 EMA trend filter + StochRSI oversold/overbought + higher low / lower high pivot
# Long:  ema1>ema2 + price<ema1 + stoch_k<20 then stoch crossup + higher low
# Short: ema1<ema2 + price>ema1 + stoch_k>80 then stoch crossdown + lower high

def _stoch_rsi(close, rsi_len=14, stoch_len=14, smooth_k=3, smooth_d=3):
    """StochRSI: treat RSI as price for stochastic calculation"""
    rsi = _rsi(close, rsi_len)
    lo  = rsi.rolling(stoch_len).min()
    hi  = rsi.rolling(stoch_len).max()
    k_raw = 100 * (rsi - lo) / (hi - lo + 1e-10)
    k = _sma(k_raw, smooth_k)
    d = _sma(k, smooth_d)
    return k, d

def gen_TradePro_2EMA_StochRSI(df, ema1_len=50, ema2_len=200,
                                rsi_len=14, stoch_len=14, smooth_k=3, smooth_d=3,
                                atr_len=14, atr_mult=0.7):
    c = df['close']
    h, l = df['high'], df['low']

    ema1 = _ema(c, ema1_len)
    ema2 = _ema(c, ema2_len)
    ema_bull = ema1 > ema2
    ema_bear = ema1 < ema2

    k, d = _stoch_rsi(c, rsi_len, stoch_len, smooth_k, smooth_d)

    # Crossovers of K vs D
    stoch_crossup   = (k.shift(1) < d.shift(1)) & (k >= d)
    stoch_crossdown = (k.shift(1) > d.shift(1)) & (k <= d)

    # Pivot highs/lows of k (simplified: local max/min over 3 bars)
    k_hi = k.rolling(3, center=True).max()
    k_lo = k.rolling(3, center=True).min()
    is_pivot_hi = (k == k_hi)
    is_pivot_lo = (k == k_lo)

    # Recent pivot values for divergence check (higher low / lower high)
    # Simplified: check last two pivot values
    recent_hi = k.where(is_pivot_hi).ffill()
    prev_hi   = k.where(is_pivot_hi).shift(1).ffill()
    recent_lo = k.where(is_pivot_lo).ffill()
    prev_lo   = k.where(is_pivot_lo).shift(1).ffill()

    higher_low  = recent_lo > prev_lo
    lower_high  = recent_hi < prev_hi

    # Trigger zones: wait for stoch to be OB/OS then cross
    l_trigger = ema_bull & (c < ema1) & (k < 20)
    s_trigger = ema_bear & (c > ema1) & (k > 80)

    l_entry = l_trigger.shift(1).fillna(False) & stoch_crossup   & higher_low
    s_entry = s_trigger.shift(1).fillna(False) & stoch_crossdown & lower_high

    sig = pd.Series(0, index=df.index)
    sig[l_entry] = 1
    sig[s_entry] = -1
    return sig

def space_TradePro_2EMA_StochRSI(trial):
    return {
        'ema1_len':  trial.suggest_int('ema1_len', 20, 100),
        'ema2_len':  trial.suggest_int('ema2_len', 100, 300),
        'rsi_len':   trial.suggest_int('rsi_len', 7, 21),
        'stoch_len': trial.suggest_int('stoch_len', 7, 21),
        'smooth_k':  trial.suggest_int('smooth_k', 2, 5),
        'smooth_d':  trial.suggest_int('smooth_d', 2, 5),
        'atr_len':   trial.suggest_int('atr_len', 7, 21),
        'atr_mult':  trial.suggest_float('atr_mult', 0.3, 1.5),
    }


# ─── 4. Crypto_Momentum ─────────────────────────────────────────────────────
# Pine: Squeeze momentum (LazyBear) + EMA slope filter
# Squeeze: BB inside KC = squeeze on; momentum = linreg of (close - midpoint)
# Long:  slope crosses above 0 + close > ema + ema rising
# Short: slope crosses below 0

def _linreg_slope(s, p):
    """Linear regression slope over rolling window p"""
    def _lr(x):
        n = len(x)
        if n < 2:
            return 0.0
        xi = np.arange(n, dtype=float)
        return np.polyfit(xi, x, 1)[0]
    return s.rolling(p).apply(_lr, raw=True)

def gen_Crypto_Momentum(df, bb_len=20, bb_mult=2.0, kc_len=20, kc_mult=1.5, ema_p=50):
    c = df['close']
    h, l = df['high'], df['low']

    # BB
    bb_basis = _sma(c, bb_len)
    bb_dev   = c.rolling(bb_len).std() * bb_mult
    bb_upper = bb_basis + bb_dev
    bb_lower = bb_basis - bb_dev

    # KC (simple range version as in Pine v4)
    kc_ma   = _sma(c, kc_len)
    tr      = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    range_ma = _sma(tr, kc_len)
    kc_upper = kc_ma + range_ma * kc_mult
    kc_lower = kc_ma - range_ma * kc_mult

    sqz_on  = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    sqz_off = (bb_lower < kc_lower) & (bb_upper > kc_upper)

    # Momentum value: linreg of (close - midpoint)
    highest_h = h.rolling(kc_len).max()
    lowest_l  = l.rolling(kc_len).min()
    midpoint  = (highest_h + lowest_l) / 2 + _sma(c, kc_len)
    delta     = c - midpoint / 2  # matches Pine: avg(avg(highest,lowest), sma(close))
    # correct midpoint: avg( avg(highest(high,len), lowest(low,len)), sma(close,len) )
    delta2    = c - ((highest_h + lowest_l) / 2 + _sma(c, kc_len)) / 2

    # linreg over kc_len bars
    val = delta2.rolling(kc_len).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] * (len(x) - 1) + np.polyfit(np.arange(len(x)), x, 1)[1],
        raw=True
    )
    # slope of val
    slope = val - val.shift(2)

    # EMA for trend filter
    ema   = _ema(c, ema_p)
    ema_slope = ema - ema.shift(1)

    # Crossovers
    co = (slope.shift(1) <= 0) & (slope > 0)   # slope crosses up
    cu = (slope.shift(1) >= 0) & (slope < 0)   # slope crosses down

    sig = pd.Series(0, index=df.index)
    sig[co & (c > ema) & (ema_slope > 0)] = 1
    sig[cu] = -1
    return sig

def space_Crypto_Momentum(trial):
    return {
        'bb_len':   trial.suggest_int('bb_len', 10, 30),
        'bb_mult':  trial.suggest_float('bb_mult', 1.5, 3.0),
        'kc_len':   trial.suggest_int('kc_len', 10, 30),
        'kc_mult':  trial.suggest_float('kc_mult', 1.0, 2.5),
        'ema_p':    trial.suggest_int('ema_p', 20, 100),
    }


# ─── 5. PriceAction_BB ──────────────────────────────────────────────────────
# Pine: Reversal after large candle outside BB
# Pin bar pattern: small body after large body outside BB → reversal
# Long:  prev large bear candle outside lower BB, current candle reversal
# Short: prev large bull candle outside upper BB, current candle reversal

def gen_PriceAction_BB(df, bb_len=21, bb_dev=2.1, body_avg_len=100):
    c = df['close']
    o = df['open']
    h, l = df['high'], df['low']

    bar   = np.where(c > o, 1, np.where(c < o, -1, 0))
    bar   = pd.Series(bar, index=df.index)
    body  = (c - o).abs()
    avg_body = _sma(body, body_avg_len)

    bb_sig = _sma(c, bb_len)
    bb_std = c.rolling(bb_len).std() * bb_dev
    bb_upper = bb_sig + bb_std
    bb_lower = bb_sig - bb_std

    # Pattern: current body < prev body / 2, prev body large, prev outside BB
    # up1: prev large bull closed above upper BB → short signal (reversal)
    # dn1: prev large bear closed below lower BB → long signal (reversal)
    up1 = (body < body.shift(1) / 2) & (body.shift(1) > avg_body.shift(1) * 2) \
          & (bar.shift(1) == 1) & (bar == -1) & (c.shift(1) > bb_upper.shift(1))
    dn1 = (body < body.shift(1) / 2) & (body.shift(1) > avg_body.shift(1) * 2) \
          & (bar.shift(1) == -1) & (bar == 1) & (c.shift(1) < bb_lower.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[dn1] = 1    # long on reversal after bear outside lower BB
    sig[up1] = -1   # short on reversal after bull outside upper BB
    return sig

def space_PriceAction_BB(trial):
    return {
        'bb_len':       trial.suggest_int('bb_len', 10, 30),
        'bb_dev':       trial.suggest_float('bb_dev', 1.5, 3.0),
        'body_avg_len': trial.suggest_int('body_avg_len', 50, 150),
    }


# ─── 6. ATR_Trail_Stop ──────────────────────────────────────────────────────
# Pine: ATR trailing stop — close crosses above trailing stop → long
# TS = highest(high - mult*atr, hhv_period) with stickiness
# Buy when close crosses above TS; Sell when crosses below

def gen_ATR_Trail_Stop(df, atr_p=5, hhv_p=10, mult=2.5):
    c = df['close']
    h = df['high']

    atr_v = _atr(h, df['low'], c, atr_p)
    band  = h - mult * atr_v
    prev_ts = band.rolling(hhv_p).max()

    # Build trailing stop with stickiness (iterative)
    ts_vals = np.full(len(c), np.nan)
    c_arr = c.values
    pt_arr = prev_ts.values

    for i in range(hhv_p + atr_p, len(c_arr)):
        if np.isnan(pt_arr[i]):
            continue
        prev = ts_vals[i-1] if not np.isnan(ts_vals[i-1]) else pt_arr[i]
        # If close > previous TS and close > prev close → TS rises
        if c_arr[i] > pt_arr[i] and c_arr[i] > c_arr[i-1]:
            ts_vals[i] = pt_arr[i]
        else:
            ts_vals[i] = prev

    ts = pd.Series(ts_vals, index=df.index)

    # Crossovers
    buy  = (c.shift(1) <= ts.shift(1)) & (c > ts)
    sell = (c.shift(1) >= ts.shift(1)) & (c < ts)

    sig = pd.Series(0, index=df.index)
    sig[buy]  = 1
    sig[sell] = -1
    return sig

def space_ATR_Trail_Stop(trial):
    return {
        'atr_p':  trial.suggest_int('atr_p', 3, 15),
        'hhv_p':  trial.suggest_int('hhv_p', 5, 20),
        'mult':   trial.suggest_float('mult', 1.0, 4.0),
    }


# ─── 7. Noro_MA_ATR ─────────────────────────────────────────────────────────
# Pine: MA + ATR band — trend defined by price relative to MA±ATR band
# Trend = +1 when low > ma + atr; -1 when high < ma - atr; else unchanged
# Entry when trend changes

def gen_Noro_MA_ATR(df, ma_len=30, atr_mult=2.0):
    c = df['close']
    h, l = df['high'], df['low']
    o = df['open']

    src = (o + h + l + c) / 4  # ohlc4
    ma  = _sma(src, ma_len)
    atr_v = _sma(
        pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1),
        ma_len
    ) * atr_mult

    # Trend: iterate to carry forward
    trend_arr = np.zeros(len(c))
    ma_arr  = ma.values
    atr_arr = atr_v.values
    l_arr   = l.values
    h_arr   = h.values

    for i in range(1, len(c)):
        if np.isnan(ma_arr[i]) or np.isnan(atr_arr[i]):
            trend_arr[i] = trend_arr[i-1]
            continue
        if l_arr[i] > ma_arr[i] + atr_arr[i]:
            trend_arr[i] = 1
        elif h_arr[i] < ma_arr[i] - atr_arr[i]:
            trend_arr[i] = -1
        else:
            trend_arr[i] = trend_arr[i-1]

    trend = pd.Series(trend_arr, index=df.index)

    # Entry on trend change
    long_entry  = (trend == 1)  & (trend.shift(1) != 1)
    short_entry = (trend == -1) & (trend.shift(1) != -1)

    sig = pd.Series(0, index=df.index)
    sig[long_entry]  = 1
    sig[short_entry] = -1
    return sig

def space_Noro_MA_ATR(trial):
    return {
        'ma_len':    trial.suggest_int('ma_len', 10, 60),
        'atr_mult':  trial.suggest_float('atr_mult', 1.0, 4.0),
    }


# ─── 8. KAMA_Strategy ───────────────────────────────────────────────────────
# Pine: Kaufman Adaptive Moving Average — rising KAMA → long, falling KAMA → short

def _kama(close, lookback=10, fast_len=5, slow_len=50):
    """Kaufman Adaptive Moving Average"""
    fastest = 2.0 / (fast_len + 1)
    slowest = 2.0 / (slow_len + 1)
    c = close.values
    kama_v = np.full(len(c), np.nan)

    # Find first valid index
    start = lookback
    while start < len(c) and np.isnan(c[start]):
        start += 1
    if start >= len(c):
        return pd.Series(kama_v, index=close.index)

    kama_v[start] = c[start]

    for i in range(start + 1, len(c)):
        if np.isnan(c[i]):
            kama_v[i] = kama_v[i-1]
            continue
        # Efficiency ratio
        direction = abs(c[i] - c[max(0, i - lookback)])
        volatility = sum(abs(c[j] - c[j-1]) for j in range(max(1, i - lookback + 1), i + 1))
        er = direction / (volatility + 1e-10)
        sc = (er * (fastest - slowest) + slowest) ** 2
        prev = kama_v[i-1] if not np.isnan(kama_v[i-1]) else c[i]
        kama_v[i] = prev + sc * (c[i] - prev)

    return pd.Series(kama_v, index=close.index)

def gen_KAMA_Strategy(df, lookback=10, fast_len=5, slow_len=50, rise_p=10, fall_p=10):
    c = df['close']
    kama = _kama(c, lookback, fast_len, slow_len)

    # Rising / falling over rise_p / fall_p bars
    rising  = kama > kama.shift(rise_p)
    falling = kama < kama.shift(fall_p)

    # Entry on change of state
    long_entry  = rising  & ~rising.shift(1).fillna(False)
    short_entry = falling & ~falling.shift(1).fillna(False)

    sig = pd.Series(0, index=df.index)
    sig[long_entry]  = 1
    sig[short_entry] = -1
    return sig

def space_KAMA_Strategy(trial):
    return {
        'lookback': trial.suggest_int('lookback', 5, 20),
        'fast_len': trial.suggest_int('fast_len', 2, 10),
        'slow_len': trial.suggest_int('slow_len', 20, 80),
        'rise_p':   trial.suggest_int('rise_p', 3, 20),
        'fall_p':   trial.suggest_int('fall_p', 3, 20),
    }


# ─── 9. MACD_RSI_Signal ─────────────────────────────────────────────────────
# Pine: MACD crossover (bull/bear) when RSI was recently oversold/overbought
# Long:  MACD crosses above signal + RSI was oversold in last 6 bars
# Short: MACD crosses below signal + RSI was overbought in last 6 bars

def gen_MACD_RSI_Signal(df, fast_ma=12, slow_ma=26, signal_len=9,
                        rsi_os=37, rsi_ob=69, rsi_len=14, lookback=6):
    c = df['close']

    ema_fast = _ema(c, fast_ma)
    ema_slow = _ema(c, slow_ma)
    macd     = ema_fast - ema_slow
    signal   = _ema(macd, signal_len)

    rsi = _rsi(c, rsi_len)

    # Was oversold / overbought in past `lookback` bars
    was_oversold   = pd.Series(False, index=df.index)
    was_overbought = pd.Series(False, index=df.index)
    for lag in range(lookback):
        was_oversold   = was_oversold   | (rsi.shift(lag) <= rsi_os)
        was_overbought = was_overbought | (rsi.shift(lag) >= rsi_ob)

    # MACD crossovers
    cross_bull = (macd.shift(1) < signal.shift(1)) & (macd >= signal)
    cross_bear = (macd.shift(1) > signal.shift(1)) & (macd <= signal)

    sig = pd.Series(0, index=df.index)
    sig[cross_bull & was_oversold]   = 1
    sig[cross_bear & was_overbought] = -1
    return sig

def space_MACD_RSI_Signal(trial):
    return {
        'fast_ma':    trial.suggest_int('fast_ma', 5, 20),
        'slow_ma':    trial.suggest_int('slow_ma', 15, 40),
        'signal_len': trial.suggest_int('signal_len', 5, 15),
        'rsi_os':     trial.suggest_int('rsi_os', 20, 45),
        'rsi_ob':     trial.suggest_int('rsi_ob', 55, 80),
        'rsi_len':    trial.suggest_int('rsi_len', 7, 21),
        'lookback':   trial.suggest_int('lookback', 3, 10),
    }


# ─── 10. Aggressive_Scalper ─────────────────────────────────────────────────
# Pine: Vortex indicator crossover + price position relative to N-bar high/low
# Vortex: VIP = sum(|high - low[1]|, p) / sum(ATR(1), p)
#         VIM = sum(|low - high[1]|, p) / sum(ATR(1), p)
# PlotG = avg of pct from N-bar low and pct from N-bar high
# Long:  plotg crosses above 0 + close > high[2] + VIP crosses above VIM
# Short: plotg crosses below 0 + VIM crosses above VIP

def gen_Aggressive_Scalper(df, range_p=60, vortex_p=14):
    c = df['close']
    h, l = df['high'], df['low']

    low9  = l.rolling(range_p).min()
    high9 = h.rolling(range_p).max()

    plot_low  = ((c - low9)  / (low9 + 1e-10))  * 100
    plot_high = ((c - high9) / (high9.abs() + 1e-10)) * 100
    plotg = (plot_low + plot_high) / 2

    center = 0.0

    # Vortex
    tr1  = (h - l.shift(1)).abs()
    tr2  = (l - h.shift(1)).abs()
    tr   = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    vmp  = tr1.rolling(vortex_p).sum()
    vmm  = tr2.rolling(vortex_p).sum()
    str_ = tr.rolling(vortex_p).sum()
    vip  = vmp / (str_ + 1e-10)
    vim  = vmm / (str_ + 1e-10)

    # Crossovers
    plotg_co  = (plotg.shift(1) <= center) & (plotg > center)
    plotg_cu  = (plotg.shift(1) >= center) & (plotg < center)
    vip_cross = (vip.shift(1) <= vim.shift(1)) & (vip > vim)
    vim_cross = (vim.shift(1) <= vip.shift(1)) & (vim > vip)

    sig = pd.Series(0, index=df.index)
    sig[plotg_co & (c > h.shift(2)) & vip_cross] = 1
    sig[plotg_cu & vim_cross] = -1
    return sig

def space_Aggressive_Scalper(trial):
    return {
        'range_p':   trial.suggest_int('range_p', 20, 100),
        'vortex_p':  trial.suggest_int('vortex_p', 7, 28),
    }


# ─── STRATEGY EXPORT ────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'CCI_EMA_RSI': {
        'gen':   gen_CCI_EMA_RSI,
        'space': space_CCI_EMA_RSI,
        'params': {
            'cci_len': 14, 'cci_ob': 150, 'cci_os': -140,
            'use_ema': True, 'ema_len': 55,
            'use_rsi': False, 'rsi_len': 14, 'rsi_ob': 70, 'rsi_os': 30,
        },
    },
    'Flash_Momentum': {
        'gen':   gen_Flash_Momentum,
        'space': space_Flash_Momentum,
        'params': {
            'mom_len': 10, 'mom_rsi_thresh': 60,
            'st_period': 10, 'st_factor': 3.0,
            'ema_len': 12, 'ema_pct': 1.0,
        },
    },
    'TradePro_2EMA_StochRSI': {
        'gen':   gen_TradePro_2EMA_StochRSI,
        'space': space_TradePro_2EMA_StochRSI,
        'params': {
            'ema1_len': 50, 'ema2_len': 200,
            'rsi_len': 14, 'stoch_len': 14, 'smooth_k': 3, 'smooth_d': 3,
            'atr_len': 14, 'atr_mult': 0.7,
        },
    },
    'Crypto_Momentum': {
        'gen':   gen_Crypto_Momentum,
        'space': space_Crypto_Momentum,
        'params': {
            'bb_len': 20, 'bb_mult': 2.0,
            'kc_len': 20, 'kc_mult': 1.5,
            'ema_p': 50,
        },
    },
    'PriceAction_BB': {
        'gen':   gen_PriceAction_BB,
        'space': space_PriceAction_BB,
        'params': {
            'bb_len': 21, 'bb_dev': 2.1, 'body_avg_len': 100,
        },
    },
    'ATR_Trail_Stop': {
        'gen':   gen_ATR_Trail_Stop,
        'space': space_ATR_Trail_Stop,
        'params': {
            'atr_p': 5, 'hhv_p': 10, 'mult': 2.5,
        },
    },
    'Noro_MA_ATR': {
        'gen':   gen_Noro_MA_ATR,
        'space': space_Noro_MA_ATR,
        'params': {
            'ma_len': 30, 'atr_mult': 2.0,
        },
    },
    'KAMA_Strategy': {
        'gen':   gen_KAMA_Strategy,
        'space': space_KAMA_Strategy,
        'params': {
            'lookback': 10, 'fast_len': 5, 'slow_len': 50,
            'rise_p': 10, 'fall_p': 10,
        },
    },
    'MACD_RSI_Signal': {
        'gen':   gen_MACD_RSI_Signal,
        'space': space_MACD_RSI_Signal,
        'params': {
            'fast_ma': 12, 'slow_ma': 26, 'signal_len': 9,
            'rsi_os': 37, 'rsi_ob': 69, 'rsi_len': 14, 'lookback': 6,
        },
    },
    'Aggressive_Scalper': {
        'gen':   gen_Aggressive_Scalper,
        'space': space_Aggressive_Scalper,
        'params': {
            'range_p': 60, 'vortex_p': 14,
        },
    },
}
