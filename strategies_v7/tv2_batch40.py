#!/usr/bin/env python3
"""
TV2 BATCH 40 — Pine Script v4/v5/v6 strategies converted to Python
Sources: Alorse/pinescript-strategies, geraked/tradingview,
         harryguiacorn/TradingView-Proprietary-Indicators,
         AlbertoCuadra/algo_trading_weighted_strategy
"""

import pandas as pd
import numpy as np

# ─── Standard helpers ────────────────────────────────────────────────────────
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _sma(s, n): return s.rolling(n).mean()
def _rma(s, n): return s.ewm(alpha=1/n, adjust=False).mean()
def _atr(df, n):
    hi, lo, cl = df['high'], df['low'], df['close']
    tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    return _rma(tr, n)
def _rsi(s, n):
    d = s.diff()
    return 100 - 100/(1 + _rma(d.clip(lower=0), n) / _rma((-d).clip(lower=0), n))
def _wma(s, n):
    w = np.arange(1, n+1)
    return s.rolling(n).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
def _hma(s, n):
    return _wma(2*_wma(s, n//2) - _wma(s, n), int(np.sqrt(n)))
def _stoch(df, k, d):
    lo = df['low'].rolling(k).min()
    hi = df['high'].rolling(k).max()
    k_line = 100 * (df['close'] - lo) / (hi - lo + 1e-10)
    return k_line, k_line.rolling(d).mean()
def _supertrend(df, n, mult):
    atr = _atr(df, n)
    hl2 = (df['high'] + df['low']) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upper.iloc[i-1]: trend.iloc[i] = 1
        elif df['close'].iloc[i] < lower.iloc[i-1]: trend.iloc[i] = -1
        else: trend.iloc[i] = trend.iloc[i-1]
    return trend
def _dmi(df, n=14, lensig=14):
    """Returns (diplus, diminus, adx)"""
    hi, lo, cl = df['high'], df['low'], df['close']
    tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    up_move = hi.diff()
    dn_move = -lo.diff()
    dm_plus = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    dm_minus = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    atr_n = _rma(tr, n)
    di_plus = 100 * _rma(pd.Series(dm_plus, index=df.index), n) / (atr_n + 1e-10)
    di_minus = 100 * _rma(pd.Series(dm_minus, index=df.index), n) / (atr_n + 1e-10)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10)
    adx = _rma(dx, lensig)
    return di_plus, di_minus, adx
def _linreg(s, n):
    """Rolling linear regression endpoint (ta.linreg equivalent)"""
    def _lr(arr):
        if np.any(np.isnan(arr)): return np.nan
        x = np.arange(n)
        m, b = np.polyfit(x, arr, 1)
        return m * (n-1) + b
    return s.rolling(n).apply(_lr, raw=True)
def _zlsma(s, n):
    lsma = _linreg(s, n)
    lsma2 = _linreg(lsma, n)
    eq = lsma - lsma2
    return lsma + eq
def _crossover(a, b):
    """Series: True where a crosses over b (was below, now above)"""
    return (a > b) & (a.shift(1) <= b.shift(1))
def _crossunder(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))
def _macd(s, fast=12, slow=26, sig=9):
    fm = _ema(s, fast)
    sm = _ema(s, slow)
    m = fm - sm
    signal = _ema(m, sig)
    return m, signal, m - signal
def _stochrsi(s, rsi_len=14, stoch_len=14, k_smooth=3, d_smooth=3):
    rsi = _rsi(s, rsi_len)
    lo = rsi.rolling(stoch_len).min()
    hi = rsi.rolling(stoch_len).max()
    k = 100 * (rsi - lo) / (hi - lo + 1e-10)
    k = k.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d

# ─────────────────────────────────────────────────────────────────────────────
# 1. Two_EMA_RSI  (v4 — 2 EMA/SMA + RSI Strategy [Alorse])
# Entry: fast>slow + RSI 25-40 + green candle (shifted 1)  → LONG
#        fast<slow + RSI 60-75 + red candle (shifted 1)    → SHORT
# ─────────────────────────────────────────────────────────────────────────────
def gen_Two_EMA_RSI(df, fast=10, slow=30, tend=200, rsi_len=8, **kw):
    cl = df['close']
    fast_ma = _sma(cl, fast)
    slow_ma = _sma(cl, slow)
    rsi = _rsi(cl, rsi_len)
    long  = (fast_ma.shift(1) > slow_ma.shift(1)) & (rsi.shift(1) > 25) & (rsi.shift(1) < 40) & (cl.shift(1) > df['open'].shift(1))
    short = (fast_ma.shift(1) < slow_ma.shift(1)) & (rsi.shift(1) < 75) & (rsi.shift(1) > 60) & (cl.shift(1) < df['open'].shift(1))
    sig = pd.Series(0, index=df.index)
    sig[long]  =  1
    sig[short] = -1
    return sig

space_Two_EMA_RSI = {
    'fast':    ('int',   [5, 10, 20]),
    'slow':    ('int',   [20, 30, 50]),
    'rsi_len': ('int',   [6, 8, 14]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Three_EMA_Cross  (v4 — 3 EMA/SMA + Cross Strategy [Alorse])
# Entry: fast crossover slow, price above tend, green candle
# Exit: fast crossunder slow
# ─────────────────────────────────────────────────────────────────────────────
def gen_Three_EMA_Cross(df, fast=10, slow=20, tend=100, **kw):
    cl = df['close']
    fast_ma = _ema(cl, fast)
    slow_ma = _ema(cl, slow)
    tend_ma = _ema(cl, tend)
    co = _crossover(fast_ma, slow_ma)
    cu = _crossunder(fast_ma, slow_ma)
    long_entry  = co & (cl > tend_ma) & (cl > df['open'])
    short_entry = cu & (cl < tend_ma) & (cl < df['open'])
    sig = pd.Series(0, index=df.index)
    sig[long_entry]  =  1
    sig[short_entry] = -1
    return sig

space_Three_EMA_Cross = {
    'fast':  ('int', [5, 10, 20]),
    'slow':  ('int', [20, 30, 50]),
    'tend':  ('int', [100, 200]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. BB_Divergence_GH  (v4 — Bollinger Bands Divergence [Alorse])
# Long: close > upper && upper expanding && lower contracting && green candle
# Short: close < lower && lower contracting && upper expanding && red candle
# ─────────────────────────────────────────────────────────────────────────────
def gen_BB_Divergence_GH(df, length=20, mult=2.0, candleper=30, **kw):
    cl, hi, lo = df['close'], df['high'], df['low']
    op = df['open']
    basis = _sma(cl, length)
    dev   = mult * cl.rolling(length).std(ddof=0)
    upper = basis + dev
    lower = basis - dev
    buy  = (cl > upper) & (upper > upper.shift(1)) & (lower < lower.shift(1)) & (cl > op)
    sell = (cl < lower) & (lower < lower.shift(1)) & (upper > upper.shift(1)) & (cl < op)
    cp = candleper / 100
    candle_size = hi - lo
    buyzone  = hi  - candle_size * (1 - cp)
    sellzone = lo  + candle_size * (1 - cp)
    long_ok  = buy  & (buyzone  > upper)
    short_ok = sell & (sellzone < lower)
    sig = pd.Series(0, index=df.index)
    sig[long_ok]  =  1
    sig[short_ok] = -1
    return sig

space_BB_Divergence_GH = {
    'length':    ('int',   [14, 20, 30]),
    'mult':      ('float', [1.5, 2.0, 2.5]),
    'candleper': ('int',   [20, 30, 40]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. BB_Winner_Lite  (v4 — Bollinger Bands Winner LITE [Alorse])
# Long: low portion of candle body is below lower BB
# Short: high portion above upper BB
# ─────────────────────────────────────────────────────────────────────────────
def gen_BB_Winner_Lite(df, length=20, mult=2.0, candleper=30, **kw):
    cl, hi, lo = df['close'], df['high'], df['low']
    op = df['open']
    basis = _sma(cl, length)
    dev   = mult * cl.rolling(length).std(ddof=0)
    upper = basis + dev
    lower = basis - dev
    cp = candleper / 100
    candle_size = hi - lo
    buyzone  = cp * candle_size + lo
    sellzone = hi - cp * candle_size
    # additional filter: body doesn't already clear lower
    body_size = (cl - op).abs()
    bsBuy  = lo  + body_size * 0.6
    bsSell = hi  - body_size * 0.6
    buy  = (buyzone  < lower) & ~(bsBuy  < lower)
    sell = (sellzone > upper)
    sig = pd.Series(0, index=df.index)
    sig[buy]  =  1
    sig[sell] = -1
    return sig

space_BB_Winner_Lite = {
    'length':    ('int',   [14, 20, 30]),
    'mult':      ('float', [1.5, 2.0, 2.5]),
    'candleper': ('int',   [20, 30, 40]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. BB_Winner_Pro  (v5 — Bollinger Bands Winner PRO [Alorse])
# Like Lite but adds RSI filter + MA trend filter
# ─────────────────────────────────────────────────────────────────────────────
def gen_BB_Winner_Pro(df, length=20, mult=2.0, candleper=30,
                      rsi_len=14, rsi_above=45, rsi_below=55,
                      ma_len=200, **kw):
    cl, hi, lo = df['close'], df['high'], df['low']
    op = df['open']
    basis = _sma(cl, length)
    dev   = mult * cl.rolling(length).std(ddof=0)
    upper = basis + dev
    lower = basis - dev
    rsi = _rsi(cl, rsi_len)
    ma  = _ema(cl, ma_len)
    cp = candleper / 100
    candle_size = hi - lo
    buyzone  = cp * candle_size + lo
    sellzone = hi - cp * candle_size
    buy  = (buyzone < lower)  & (cl < op) & (rsi < rsi_above) & (cl > ma)
    sell = (sellzone > upper) & (cl > op) & (rsi > rsi_below)  & (cl < ma)
    sig = pd.Series(0, index=df.index)
    sig[buy]  =  1
    sig[sell] = -1
    return sig

space_BB_Winner_Pro = {
    'length':    ('int',   [14, 20, 30]),
    'mult':      ('float', [1.5, 2.0, 2.5]),
    'candleper': ('int',   [20, 30, 40]),
    'rsi_len':   ('int',   [10, 14, 20]),
    'ma_len':    ('int',   [100, 200, 300]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 6. BB_Aroon  (v5 — Bollinger Bands + Aroon [Alorse])
# Long: close crosses under lower BB AND AroonUP > 90
# Exit: close crosses under upper BB OR AroonUP < 70
# ─────────────────────────────────────────────────────────────────────────────
def gen_BB_Aroon(df, length=20, mult=2.0, aroon_len=288,
                 confirm=90, stop=70, **kw):
    cl, hi = df['close'], df['high']
    basis = _sma(cl, length)
    dev   = mult * cl.rolling(length).std(ddof=0)
    upper = basis + dev
    lower = basis - dev
    aroon_up = 100 * (hi.rolling(aroon_len+1).apply(lambda x: x.argmax(), raw=True)) / aroon_len
    bullish = _crossunder(cl, lower)
    sig = pd.Series(0, index=df.index)
    in_long = False
    for i in range(len(df)):
        if bullish.iloc[i] and aroon_up.iloc[i] > confirm:
            in_long = True
        if in_long:
            if cl.iloc[i] <= upper.iloc[i] and aroon_up.iloc[i] > stop:
                sig.iloc[i] = 1
            else:
                in_long = False
    return sig

space_BB_Aroon = {
    'length':    ('int', [14, 20, 30]),
    'mult':      ('float', [1.5, 2.0, 2.5]),
    'aroon_len': ('int', [100, 200, 288]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 7. Double_RSI  (v5 — Double RSI [Alorse])
# Long: RSI(14) crosses up 30 AND higher-TF RSI < 35
# Short: RSI(14) crosses down 70 AND higher-TF RSI > 65
# MTF simulated as RSI on resampled × 3
# ─────────────────────────────────────────────────────────────────────────────
def gen_Double_RSI(df, rsi_len=14, mtf_factor=3, **kw):
    cl = df['close']
    rsi  = _rsi(cl, rsi_len)
    # Simulate MTF: use longer period RSI as proxy
    rsi_mtf = _rsi(cl, rsi_len * mtf_factor)
    buy  = _crossover(rsi, pd.Series(30, index=df.index)) & (rsi_mtf < 35)
    sell = _crossunder(rsi, pd.Series(70, index=df.index)) & (rsi_mtf > 65)
    sig = pd.Series(0, index=df.index)
    sig[buy]  =  1
    sig[sell] = -1
    return sig

space_Double_RSI = {
    'rsi_len':    ('int', [10, 14, 20]),
    'mtf_factor': ('int', [2, 3, 4]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 8. EMA_Moving_Away  (v5 — EMA Moving away Strategy [Alorse])
# Long:  close <= ema*(1-pct/100) and candle % < SL
# Short: close >= ema*(1+pct/100) and candle % < SL + not 4 green
# ─────────────────────────────────────────────────────────────────────────────
def gen_EMA_Moving_Away(df, length=55, moving_away=2.0, sl_pct=2.0, **kw):
    cl, op = df['close'], df['open']
    ema = _ema(cl, length)
    lc_pct = ((cl - op) / op.abs()).abs() * 100
    last4green = ((cl > op).rolling(4).sum() == 4)
    long_entry  = (cl <= ema * (1 - moving_away/100)) & (lc_pct < sl_pct)
    short_entry = (cl >= ema * (1 + moving_away/100)) & (lc_pct < sl_pct) & (~last4green)
    sig = pd.Series(0, index=df.index)
    sig[long_entry]  =  1
    sig[short_entry] = -1
    return sig

space_EMA_Moving_Away = {
    'length':      ('int',   [20, 34, 55, 89]),
    'moving_away': ('float', [1.0, 1.5, 2.0, 3.0]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 9. Exceeded_Candle  (v4 — Exceeded candle [Alorse])
# Long: green candle surpasses prev red, close < BB basis
# ─────────────────────────────────────────────────────────────────────────────
def gen_Exceeded_Candle(df, length=20, mult=2.0, **kw):
    cl, op, hi, lo = df['close'], df['open'], df['high'], df['low']
    basis = _sma(cl, length)
    dev   = mult * cl.rolling(length).std(ddof=0)
    upper = basis + dev
    green_surpass = (cl.shift(1) < op.shift(1)) & (cl > op) & (cl > op.shift(1))
    red_surpass   = (cl.shift(1) > op.shift(1)) & (cl < op) & (cl < op.shift(1))
    last3red = (cl.shift(2) < op.shift(2)) & (cl.shift(3) < op.shift(3))
    buy_entry = green_surpass & (cl < basis) & (~last3red)
    buy_exit  = cl > upper
    sig = pd.Series(0, index=df.index)
    in_long = False
    for i in range(len(df)):
        if buy_entry.iloc[i]:
            in_long = True
        if buy_exit.iloc[i]:
            in_long = False
        if in_long:
            sig.iloc[i] = 1
    return sig

space_Exceeded_Candle = {
    'length': ('int',   [14, 20, 30]),
    'mult':   ('float', [1.5, 2.0, 2.5]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 10. Full_Candle  (v5 — Full Candle [Alorse])
# Long: green candle, close > EMA, shadow% <= threshold
# Short: red candle, close < EMA, shadow% <= threshold
# ─────────────────────────────────────────────────────────────────────────────
def gen_Full_Candle(df, ema_len=10, shadow_pct=5.0, **kw):
    cl, op, hi, lo = df['close'], df['open'], df['high'], df['low']
    ema = _ema(cl, ema_len)
    candle_size = hi - lo
    shadow_size = np.where(cl > op, hi - cl, cl - lo)
    shadow_size = pd.Series(shadow_size, index=df.index)
    s_pct = (shadow_size * 100) / (candle_size + 1e-10)
    long_ok  = (cl > op) & (cl > ema) & (s_pct <= shadow_pct)
    short_ok = (cl < op) & (cl < ema) & (s_pct <= shadow_pct)
    sig = pd.Series(0, index=df.index)
    sig[long_ok]  =  1
    sig[short_ok] = -1
    return sig

space_Full_Candle = {
    'ema_len':    ('int',   [5, 10, 20]),
    'shadow_pct': ('float', [3.0, 5.0, 10.0]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 11. MACD_RSI_GH  (v4 — MACD + RSI Strategy [Alorse])
# Long: MACD crosses over signal AND RSI was oversold in last N bars
# ─────────────────────────────────────────────────────────────────────────────
def gen_MACD_RSI_GH(df, fast=12, slow=26, sig_len=9, rsi_len=14,
                    rsi_back=5, rsi_dn=30, rsi_up=70, **kw):
    cl = df['close']
    macd, signal, _ = _macd(cl, fast, slow, sig_len)
    rsi = _rsi(cl, rsi_len)
    was_rsi_down = pd.Series(False, index=df.index)
    was_rsi_up   = pd.Series(False, index=df.index)
    for lag in range(1, rsi_back+1):
        was_rsi_down = was_rsi_down | (rsi.shift(lag) <= rsi_dn)
        was_rsi_up   = was_rsi_up   | (rsi.shift(lag) >= rsi_up)
    bull = _crossover(macd, signal) & was_rsi_down
    bear = _crossunder(macd, signal) & was_rsi_up
    sig = pd.Series(0, index=df.index)
    sig[bull] =  1
    sig[bear] = -1
    return sig

space_MACD_RSI_GH = {
    'fast':     ('int', [8, 12, 16]),
    'slow':     ('int', [20, 26, 34]),
    'rsi_len':  ('int', [10, 14, 20]),
    'rsi_dn':   ('int', [20, 25, 30]),
    'rsi_up':   ('int', [65, 70, 75]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 12. MACD_BB_RSI  (v5 — MACD + BB + RSI [Alorse])
# Long: MACD crossover + RSI<50 + close < BB basis
# Short: MACD crossunder + RSI>50 + close > BB basis
# ─────────────────────────────────────────────────────────────────────────────
def gen_MACD_BB_RSI(df, fast=12, slow=26, sig_len=9,
                    bb_len=20, bb_mult=2.0, rsi_len=14, **kw):
    cl = df['close']
    macd, signal, _ = _macd(cl, fast, slow, sig_len)
    basis = _sma(cl, bb_len)
    dev   = bb_mult * cl.rolling(bb_len).std(ddof=0)
    upper = basis + dev
    lower = basis - dev
    rsi = _rsi(cl, rsi_len)
    long_entry  = _crossover(macd, signal) & (rsi < 50) & (cl < basis)
    short_entry = _crossunder(macd, signal) & (rsi > 50) & (cl > basis)
    long_exit   = (rsi > 70) & (cl > upper)
    short_exit  = (rsi < 31) & (cl < lower)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if long_exit.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if short_exit.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_MACD_BB_RSI = {
    'fast':     ('int',   [8, 12, 16]),
    'slow':     ('int',   [20, 26, 34]),
    'bb_len':   ('int',   [14, 20, 30]),
    'bb_mult':  ('float', [1.5, 2.0, 2.5]),
    'rsi_len':  ('int',   [10, 14, 20]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 13. MACD_DMI  (v4 — MACD + DMI [Alorse])
# Long: MACD crossover + DI+ > DI-
# Exit: crossunder MACD + DI+ < DI- OR volstop triggered
# ─────────────────────────────────────────────────────────────────────────────
def gen_MACD_DMI(df, dmi_len=14, dmi_sig=14, fast=12, slow=26, sig_len=9, **kw):
    cl = df['close']
    macd, signal, _ = _macd(cl, fast, slow, sig_len)
    di_plus, di_minus, adx = _dmi(df, dmi_len, dmi_sig)
    long_entry = _crossover(macd, signal) & (di_plus > di_minus)
    long_exit  = _crossunder(macd, signal) & (di_plus < di_minus)
    sig = pd.Series(0, index=df.index)
    sig[long_entry] =  1
    sig[long_exit]  = 0  # will be processed below
    # stateful fill
    in_long = False
    out = pd.Series(0, index=df.index)
    for i in range(len(df)):
        if long_entry.iloc[i]: in_long = True
        if long_exit.iloc[i]:  in_long = False
        out.iloc[i] = 1 if in_long else 0
    return out

space_MACD_DMI = {
    'dmi_len': ('int', [10, 14, 20]),
    'fast':    ('int', [8, 12, 16]),
    'slow':    ('int', [20, 26, 34]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 14. MA_Cross_DMI  (v4 — MA Cross + DMI [Alorse])
# Long: fast MA crossover slow MA
# Short: fast MA crossunder slow MA
# ─────────────────────────────────────────────────────────────────────────────
def gen_MA_Cross_DMI(df, ma1_len=10, ma2_len=20, **kw):
    cl = df['close']
    ma1 = _ema(cl, ma1_len)
    ma2 = _ema(cl, ma2_len)
    long_entry  = _crossover(ma1, ma2)
    short_entry = _crossunder(ma1, ma2)
    sig = pd.Series(0, index=df.index)
    sig[long_entry]  =  1
    sig[short_entry] = -1
    return sig

space_MA_Cross_DMI = {
    'ma1_len': ('int', [5, 10, 20]),
    'ma2_len': ('int', [20, 30, 50]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 15. MEMA_BB_RSI  (v5 — MEMA + BB + RSI [Alorse])
# MTF EMA simulated with longer period
# Long: close > slow_ma AND low < lower_BB
# Short: close < slow_ma AND high > upper_BB AND RSI>50
# ─────────────────────────────────────────────────────────────────────────────
def gen_MEMA_BB_RSI(df, ma1=10, ma2=55, mtf_factor=5,
                    bb_len=20, bb_mult=2.0, rsi_len=14, rsi_exit=71, **kw):
    cl, hi, lo = df['close'], df['high'], df['low']
    # simulate MTF EMA
    ma_slow = _ema(cl, ma1 * mtf_factor)
    ma_fast = _ema(cl, ma2 * mtf_factor)
    basis = _sma(cl, bb_len)
    dev   = bb_mult * cl.rolling(bb_len).std(ddof=0)
    upper = basis + dev
    lower = basis - dev
    rsi = _rsi(cl, rsi_len)
    long_entry  = (cl > ma_slow) & (lo < lower)
    short_entry = (cl < ma_slow) & (hi > upper) & (rsi > 50)
    long_exit   = (rsi > rsi_exit) | (cl < ma_fast)
    short_exit  = (cl < lower) | (cl > ma_fast)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if long_exit.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if short_exit.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_MEMA_BB_RSI = {
    'ma1':        ('int',   [5, 10, 20]),
    'ma2':        ('int',   [30, 55, 80]),
    'bb_len':     ('int',   [14, 20, 30]),
    'bb_mult':    ('float', [1.5, 2.0, 2.5]),
    'rsi_exit':   ('int',   [65, 71, 80]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 16. Multi_BB  (v4 — Multi Bollinger Bands [Alorse])
# MTF BB: simulate higher TF as 3× factor
# Long: close < avg(lower, mtf_lower)
# Exit: close > upper
# ─────────────────────────────────────────────────────────────────────────────
def gen_Multi_BB(df, length=20, mult=2.0, mtf_factor=3, **kw):
    cl = df['close']
    basis = _sma(cl, length)
    dev   = mult * cl.rolling(length).std(ddof=0)
    upper = basis + dev
    lower = basis - dev
    # simulate MTF: same BB on larger window
    basis_m = _sma(cl, length * mtf_factor)
    dev_m   = mult * cl.rolling(length * mtf_factor).std(ddof=0)
    upper_m = basis_m + dev_m
    lower_m = basis_m - dev_m
    avg_lower = (lower + lower_m) / 2
    long_entry = cl < avg_lower
    long_exit  = cl > upper
    sig = pd.Series(0, index=df.index)
    in_long = False
    for i in range(len(df)):
        if long_entry.iloc[i]: in_long = True
        if long_exit.iloc[i]:  in_long = False
        sig.iloc[i] = 1 if in_long else 0
    return sig

space_Multi_BB = {
    'length':     ('int',   [10, 20, 30]),
    'mult':       ('float', [1.5, 2.0, 2.5]),
    'mtf_factor': ('int',   [2, 3, 4]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 17. Pin_Bar_Magic  (v4 — Pin Bar Magic v1 [Alorse])
# Long: uptrend (EMA fan) + bullish pin bar + pin pierces MA
# Short: downtrend + bearish pin bar + pin pierces MA
# ─────────────────────────────────────────────────────────────────────────────
def gen_Pin_Bar_Magic(df, slow=50, medm=18, fast=6, **kw):
    cl, op, hi, lo = df['close'], df['open'], df['high'], df['low']
    slowSMA = _sma(cl, slow)
    medmEMA = _ema(cl, medm)
    fastEMA = _ema(cl, fast)
    bull_pin = ((cl > op) & ((op - lo) > 0.66*(hi-lo))) | ((cl < op) & ((cl - lo) > 0.66*(hi-lo)))
    bear_pin = ((cl > op) & ((hi - cl) > 0.66*(hi-lo))) | ((cl < op) & ((hi - op) > 0.66*(hi-lo)))
    fan_up = (fastEMA > medmEMA) & (medmEMA > slowSMA)
    fan_dn = (fastEMA < medmEMA) & (medmEMA < slowSMA)
    bull_pierce = ((lo < fastEMA) & (op > fastEMA) & (cl > fastEMA)) | \
                  ((lo < medmEMA) & (op > medmEMA) & (cl > medmEMA)) | \
                  ((lo < slowSMA) & (op > slowSMA) & (cl > slowSMA))
    bear_pierce = ((hi > fastEMA) & (op < fastEMA) & (cl < fastEMA)) | \
                  ((hi > medmEMA) & (op < medmEMA) & (cl < medmEMA)) | \
                  ((hi > slowSMA) & (op < slowSMA) & (cl < slowSMA))
    long_entry  = fan_up & bull_pin & bull_pierce
    short_entry = fan_dn & bear_pin & bear_pierce
    long_exit   = _crossunder(fastEMA, medmEMA)
    short_exit  = _crossover(fastEMA, medmEMA)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if long_exit.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if short_exit.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_Pin_Bar_Magic = {
    'slow': ('int', [30, 50, 100]),
    'medm': ('int', [10, 18, 25]),
    'fast': ('int', [3, 6, 10]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 18. QQE_Signals  (v4 — QQE signals [Alorse])
# Full QQE implementation
# ─────────────────────────────────────────────────────────────────────────────
def gen_QQE_Signals(df, rsi_period=14, sf=5, qqe_factor=4.238, **kw):
    cl = df['close']
    wilders = rsi_period * 2 - 1
    rsi_val = _rsi(cl, rsi_period)
    rsi_ma  = _ema(rsi_val, sf)
    atr_rsi = rsi_ma.diff().abs()
    ma_atr  = _ema(atr_rsi, wilders)
    dar     = _ema(ma_atr, wilders) * qqe_factor
    longband  = pd.Series(0.0, index=df.index)
    shortband = pd.Series(0.0, index=df.index)
    trend     = pd.Series(0,   index=df.index)
    for i in range(1, len(df)):
        newlong  = rsi_ma.iloc[i] - dar.iloc[i]
        newshort = rsi_ma.iloc[i] + dar.iloc[i]
        longband.iloc[i]  = newlong  if (rsi_ma.iloc[i-1] <= longband.iloc[i-1]  or rsi_ma.iloc[i] <= longband.iloc[i-1])  else max(longband.iloc[i-1], newlong)
        shortband.iloc[i] = newshort if (rsi_ma.iloc[i-1] >= shortband.iloc[i-1] or rsi_ma.iloc[i] >= shortband.iloc[i-1]) else min(shortband.iloc[i-1], newshort)
        cross1 = (longband.iloc[i-1] != rsi_ma.iloc[i]) and (longband.iloc[i] == rsi_ma.iloc[i])
        if rsi_ma.iloc[i] > shortband.iloc[i-1]:
            trend.iloc[i] = 1
        elif cross1:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
    fast_tl = pd.Series(np.where(trend == 1, longband, shortband), index=df.index)
    qqe_long  = pd.Series(0, index=df.index)
    qqe_short = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        qqe_long.iloc[i]  = qqe_long.iloc[i-1]  + (1 if fast_tl.iloc[i] < rsi_ma.iloc[i] else -qqe_long.iloc[i-1])
        qqe_short.iloc[i] = qqe_short.iloc[i-1] + (1 if fast_tl.iloc[i] > rsi_ma.iloc[i] else -qqe_short.iloc[i-1])
    long_sig  = qqe_long == 1
    short_sig = qqe_short == 1
    sig = pd.Series(0, index=df.index)
    in_long = False
    for i in range(len(df)):
        if long_sig.iloc[i]: in_long = True
        if short_sig.iloc[i]: in_long = False
        sig.iloc[i] = 1 if in_long else 0
    return sig

space_QQE_Signals = {
    'rsi_period': ('int',   [10, 14, 20]),
    'sf':         ('int',   [3, 5, 8]),
    'qqe_factor': ('float', [2.618, 4.238, 6.0]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 19. RSI_1200  (v5 — RSI + 1200 Strategy [Alorse])
# Long: close>MTF_EMA + RSI crosses up oversold + close > ema*1.01
# MTF simulated with mtf_factor
# ─────────────────────────────────────────────────────────────────────────────
def gen_RSI_1200(df, rsi_len=14, ema_len=150, mtf_factor=2,
                 rsi_ob=72, rsi_os=28, **kw):
    cl, op = df['close'], df['open']
    rsi = _rsi(cl, rsi_len)
    ema = _ema(cl, ema_len * mtf_factor)
    slack_long  = cl > ema * 1.01
    slack_short = cl < ema * 0.99
    five_long   = (cl > ema) & (cl < ema*1.02) & (cl < op) & (cl.shift(1) < op.shift(1)) & (cl.shift(2) < op.shift(2))
    five_short  = (cl < ema) & (cl > ema*0.99) & (cl > op) & (cl.shift(1) > op.shift(1)) & (cl.shift(2) > op.shift(2))
    long_entry  = (cl > ema) & _crossover(rsi, pd.Series(rsi_os, index=df.index)) & slack_long
    short_entry = (cl < ema) & _crossunder(rsi, pd.Series(rsi_ob, index=df.index)) & slack_short
    long_exit   = (rsi > rsi_ob) | five_long
    short_exit  = (rsi < rsi_os) | five_short
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if long_exit.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if short_exit.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_RSI_1200 = {
    'rsi_len':   ('int', [10, 14, 20]),
    'ema_len':   ('int', [100, 150, 200]),
    'mtf_factor':('int', [1, 2, 3]),
    'rsi_ob':    ('int', [68, 72, 75]),
    'rsi_os':    ('int', [25, 28, 32]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 20. RSI_EMA  (v5 — RSI + EMA [Alorse])
# Long: RSI < oversold AND fast MA > slow MA
# Short: RSI > overbought AND fast MA > slow MA (momentum check)
# ─────────────────────────────────────────────────────────────────────────────
def gen_RSI_EMA(df, rsi_len=14, rsi_ob=70, rsi_os=30,
                ma1_len=150, ma2_len=600, **kw):
    cl = df['close']
    rsi = _rsi(cl, rsi_len)
    ma1 = _ema(cl, ma1_len)
    ma2 = _ema(cl, ma2_len)
    long_entry  = (rsi < rsi_os) & (ma1 > ma2)
    short_entry = (rsi > rsi_ob) & (ma1 > ma2)
    long_exit   = rsi > rsi_ob
    short_exit  = rsi < rsi_os
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if long_exit.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if short_exit.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_RSI_EMA = {
    'rsi_len':  ('int', [10, 14, 20]),
    'rsi_ob':   ('int', [65, 70, 75]),
    'rsi_os':   ('int', [25, 30, 35]),
    'ma1_len':  ('int', [100, 150, 200]),
    'ma2_len':  ('int', [400, 600, 800]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 21. ST_Simple  (v4 — Supertrend [Alorse])
# Long: supertrend flips to 1 (from -1)
# ─────────────────────────────────────────────────────────────────────────────
def gen_ST_Simple(df, periods=10, multiplier=3.7, **kw):
    trend = _supertrend(df, periods, multiplier)
    buy_signal  = (trend == 1) & (trend.shift(1) == -1)
    sell_signal = (trend == -1) & (trend.shift(1) == 1)
    sig = pd.Series(0, index=df.index)
    sig[buy_signal]  =  1
    sig[sell_signal] = -1
    return sig

space_ST_Simple = {
    'periods':    ('int',   [7, 10, 14]),
    'multiplier': ('float', [2.5, 3.0, 3.7, 4.5]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 22. ST_EMA_Rebound  (v5 — Supertrend + EMA rebound [Alorse])
# Long: ST flips up OR (in uptrend + close crossed back above EMA)
# Short: ST flips down OR (in downtrend + close crossed back below EMA)
# ─────────────────────────────────────────────────────────────────────────────
def gen_ST_EMA_Rebound(df, atr_period=10, factor=3.0, ema_len=20, **kw):
    trend = _supertrend(df, atr_period, factor)
    ema = _ema(df['close'], ema_len)
    cl = df['close']
    direction = trend  # 1=long, -1=short
    in_long  = direction < 0  # Pine convention: direction<0 means up
    in_short = direction > 0
    flip_long   = (direction.shift(1) > 0) & (direction < 0)  # -1 to 1 equivalent
    flip_short  = (direction.shift(1) < 0) & (direction > 0)
    rebound_long  = in_long  & (cl.shift(1) < ema.shift(1)) & (cl > ema)
    rebound_short = in_short & (cl.shift(1) > ema.shift(1)) & (cl < ema)
    long_entry  = flip_long  | rebound_long
    short_entry = flip_short | rebound_short
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if long_entry.iloc[i]:  in_pos = 1
        elif short_entry.iloc[i]: in_pos = -1
        if flip_short.iloc[i] and in_pos == 1:  in_pos = 0
        if flip_long.iloc[i]  and in_pos == -1: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_ST_EMA_Rebound = {
    'atr_period': ('int',   [7, 10, 14]),
    'factor':     ('float', [2.0, 3.0, 4.0]),
    'ema_len':    ('int',   [10, 20, 50]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 23. ST_RSI  (v5 — Supertrend + RSI Strategy [Alorse])
# Long: in long trend + RSI crosses up 50
# Short: in short trend + RSI crosses down 50
# ─────────────────────────────────────────────────────────────────────────────
def gen_ST_RSI(df, atr_period=10, factor=3.0, rsi_len=14,
               rsi_ob=72, rsi_os=28, **kw):
    trend = _supertrend(df, atr_period, factor)
    cl = df['close']
    rsi = _rsi(cl, rsi_len)
    in_long  = trend == 1
    in_short = trend == -1
    rsi50_cross_up   = (rsi.shift(1) < 50) & (rsi >= 50)
    rsi50_cross_down = (rsi.shift(1) > 50) & (rsi <= 50)
    long_entry  = in_long  & rsi50_cross_up
    short_entry = in_short & rsi50_cross_down
    long_exit   = (rsi > rsi_ob) | (trend == -1)
    short_exit  = (rsi < rsi_os) | (trend == 1)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if long_exit.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if short_exit.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_ST_RSI = {
    'atr_period': ('int',   [7, 10, 14]),
    'factor':     ('float', [2.0, 3.0, 4.0]),
    'rsi_len':    ('int',   [10, 14, 20]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 24. Tendency_EMA_RSI  (v5 — Trend EMA + RSI [Alorse])
# Long: EMA_A crosses over EMA_B, EMA_A > EMA_C, green candle
# Short: EMA_A crosses under EMA_B, EMA_A < EMA_C, red candle
# ─────────────────────────────────────────────────────────────────────────────
def gen_Tendency_EMA_RSI(df, len_a=10, len_b=20, len_c=100, rsi_len=14, **kw):
    cl, op = df['close'], df['open']
    ema_a = _ema(cl, len_a)
    ema_b = _ema(cl, len_b)
    ema_c = _ema(cl, len_c)
    rsi   = _rsi(cl, rsi_len)
    long_entry  = _crossover(ema_a, ema_b) & (ema_a > ema_c) & (cl > op)
    short_entry = _crossunder(ema_a, ema_b) & (ema_a < ema_c) & (cl < op)
    long_exit   = rsi > 70
    short_exit  = rsi < 30
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if long_exit.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if short_exit.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_Tendency_EMA_RSI = {
    'len_a':   ('int', [5, 10, 20]),
    'len_b':   ('int', [15, 20, 30]),
    'len_c':   ('int', [50, 100, 200]),
    'rsi_len': ('int', [10, 14, 20]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 25. MACD_Long_GH  (v5 — MACD Long Strategy [Alorse/Bunghole])
# Long: RSI was oversold in last 10 bars + MACD crossover
# Short: RSI was overbought in last 10 bars + MACD crossunder
# ─────────────────────────────────────────────────────────────────────────────
def gen_MACD_Long_GH(df, fast=12, slow=26, sig_len=9, rsi_len=14,
                     rsi_os=30, rsi_ob=70, lookback=10, **kw):
    cl = df['close']
    macd, signal, _ = _macd(cl, fast, slow, sig_len)
    rsi = _rsi(cl, rsi_len)
    was_os = pd.Series(False, index=df.index)
    was_ob = pd.Series(False, index=df.index)
    for lag in range(1, lookback+1):
        was_os = was_os | (rsi.shift(lag) <= rsi_os)
        was_ob = was_ob | (rsi.shift(lag) >= rsi_ob)
    buy  = was_os & _crossover(macd, signal)
    sell = was_ob & _crossunder(macd, signal)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if buy.iloc[i]:  in_pos = 1
            elif sell.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if sell.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if buy.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_MACD_Long_GH = {
    'fast':     ('int', [8, 12, 16]),
    'slow':     ('int', [20, 26, 34]),
    'rsi_os':   ('int', [20, 25, 30]),
    'rsi_ob':   ('int', [65, 70, 75]),
    'lookback': ('int', [5, 10, 15]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 26. Improvising_GH  (v5 — Improvising [Alorse])
# Long: engulf up + close>EMA + RSI<65 + MACD rising
# Short: engulf down + close<EMA + RSI>35 + MACD falling
# ─────────────────────────────────────────────────────────────────────────────
def gen_Improvising_GH(df, ema_len=10, rsi_len=14, fast=12, slow=26, **kw):
    cl, op = df['close'], df['open']
    ema  = _ema(cl, ema_len)
    rsi  = _rsi(cl, rsi_len)
    macd = _ema(cl, fast) - _ema(cl, slow)
    buy  = (cl.shift(1) < op.shift(1)) & (cl > op.shift(1))
    sell = (cl.shift(1) > op.shift(1)) & (cl < op.shift(1))
    long_entry  = buy  & (cl > ema) & (cl.shift(1) > ema.shift(1)) & (rsi < 65) & (macd > macd.shift(1))
    short_entry = sell & (cl < ema) & (cl.shift(1) < ema.shift(1)) & (rsi > 35) & (macd < macd.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[long_entry]  =  1
    sig[short_entry] = -1
    return sig

space_Improvising_GH = {
    'ema_len': ('int', [5, 10, 20]),
    'rsi_len': ('int', [10, 14, 20]),
    'fast':    ('int', [8, 12, 16]),
    'slow':    ('int', [20, 26, 34]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 27. Omar_MMR  (v5 — Omar MMR [Alorse])
# Long: close>EMA200 + EMA20>EMA50 + MACD crossover + RSI 29-70
# ─────────────────────────────────────────────────────────────────────────────
def gen_Omar_MMR(df, len_a=20, len_b=50, len_c=200,
                 macd_fast=12, macd_slow=26, macd_sig=9, rsi_len=14, **kw):
    cl = df['close']
    ema_a = _ema(cl, len_a)
    ema_b = _ema(cl, len_b)
    ema_c = _ema(cl, len_c)
    macd, signal, _ = _macd(cl, macd_fast, macd_slow, macd_sig)
    rsi = _rsi(cl, rsi_len)
    long_entry = (cl > ema_c) & (ema_a > ema_b) & _crossover(macd, signal) & (rsi > 29) & (rsi < 70)
    sig = pd.Series(0, index=df.index)
    sig[long_entry] = 1
    return sig

space_Omar_MMR = {
    'len_a':     ('int', [10, 20, 30]),
    'len_b':     ('int', [30, 50, 80]),
    'len_c':     ('int', [100, 200, 300]),
    'rsi_len':   ('int', [10, 14, 20]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 28. Omar_Edited_WF  (v5 — Williams Fractals + EMA [Alorse])
# Long: price above EMAs + down fractal forms + MACD bullish
# Short: price below EMAs + up fractal + MACD bearish
# ─────────────────────────────────────────────────────────────────────────────
def gen_Omar_Edited_WF(df, n=2, len_a=20, len_b=50, len_c=100, **kw):
    cl, hi, lo, op = df['close'], df['high'], df['low'], df['open']
    ema_a = _ema(cl, len_a)
    ema_b = _ema(cl, len_b)
    ema_c = _ema(cl, len_c)
    macd, signal, _ = _macd(cl, 12, 26, 9)
    # Williams fractal (offset -n: detected at n bars ago)
    def up_fractal(hi, n):
        result = pd.Series(False, index=hi.index)
        for i in range(n, len(hi)-n):
            if all(hi.iloc[i] > hi.iloc[i-j] for j in range(1, n+1)) and \
               all(hi.iloc[i] > hi.iloc[i+j] for j in range(1, n+1)):
                result.iloc[i] = True
        return result
    def dn_fractal(lo, n):
        result = pd.Series(False, index=lo.index)
        for i in range(n, len(lo)-n):
            if all(lo.iloc[i] < lo.iloc[i-j] for j in range(1, n+1)) and \
               all(lo.iloc[i] < lo.iloc[i+j] for j in range(1, n+1)):
                result.iloc[i] = True
        return result
    up_frac = up_fractal(hi, n).shift(n).fillna(False).infer_objects(copy=False)
    dn_frac = dn_fractal(lo, n).shift(n).fillna(False).infer_objects(copy=False)
    long_ok  = (cl > ema_a) & (cl > ema_b) & (cl > ema_c) & (cl > op)
    short_ok = (cl < ema_a) & (cl < ema_b) & (cl < ema_c) & (cl < op)
    long_entry  = long_ok  & dn_frac & (macd > signal)
    short_entry = short_ok & up_frac & (macd < signal)
    sig = pd.Series(0, index=df.index)
    sig[long_entry]  =  1
    sig[short_entry] = -1
    return sig

space_Omar_Edited_WF = {
    'n':     ('int', [1, 2, 3]),
    'len_a': ('int', [10, 20, 30]),
    'len_b': ('int', [30, 50, 80]),
    'len_c': ('int', [80, 100, 150]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 29. HA_SSL  (v5 — HA UnivLong&Short Futures [Alorse])
# Uses Heikin-Ashi approximation with SSL channel
# ─────────────────────────────────────────────────────────────────────────────
def gen_HA_SSL(df, sma_period=3, **kw):
    cl, op, hi, lo = df['close'], df['open'], df['high'], df['low']
    # Approximate HA candles
    ha_cl = (op + hi + lo + cl) / 4
    ha_op = pd.Series(index=df.index, dtype=float)
    ha_op.iloc[0] = (op.iloc[0] + cl.iloc[0]) / 2
    for i in range(1, len(df)):
        ha_op.iloc[i] = (ha_op.iloc[i-1] + ha_cl.iloc[i-1]) / 2
    ha_hi = pd.concat([hi, ha_op, ha_cl], axis=1).max(axis=1)
    ha_lo = pd.concat([lo, ha_op, ha_cl], axis=1).min(axis=1)
    sma_hi = _sma(ha_hi, sma_period)
    sma_lo = _sma(ha_lo, sma_period)
    # SSL
    hlv = pd.Series(0, index=df.index)
    for i in range(len(df)):
        if ha_cl.iloc[i] > sma_hi.iloc[i]:   hlv.iloc[i] = 1
        elif ha_cl.iloc[i] < sma_lo.iloc[i]: hlv.iloc[i] = -1
        else: hlv.iloc[i] = hlv.iloc[i-1] if i > 0 else 0
    ssl_up = pd.Series(np.where(hlv < 0, sma_lo, sma_hi), index=df.index)
    ssl_dn = pd.Series(np.where(hlv < 0, sma_hi, sma_lo), index=df.index)
    long_sig  = _crossover(ssl_up, ssl_dn)
    short_sig = _crossover(ssl_dn, ssl_up)
    sig = pd.Series(0, index=df.index)
    sig[long_sig]  =  1
    sig[short_sig] = -1
    return sig

space_HA_SSL = {
    'sma_period': ('int', [2, 3, 5, 8]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 30. MTF_RSI_GH  (v5 — MTF RSI [Alorse])
# Long: close>MA + RSI oversold
# Exit: RSI overbought or close < MA
# ─────────────────────────────────────────────────────────────────────────────
def gen_MTF_RSI_GH(df, rsi_len=14, rsi_ob=70, rsi_os=30,
                   ma_len=900, ma_type='ema', **kw):
    cl = df['close']
    rsi = _rsi(cl, rsi_len)
    ma  = _ema(cl, ma_len) if ma_type == 'ema' else _sma(cl, ma_len)
    long_entry = (cl > ma) & (rsi < rsi_os)
    long_exit  = (rsi > rsi_ob) | (cl < ma)
    sig = pd.Series(0, index=df.index)
    in_long = False
    for i in range(len(df)):
        if long_entry.iloc[i]: in_long = True
        if long_exit.iloc[i]:  in_long = False
        sig.iloc[i] = 1 if in_long else 0
    return sig

space_MTF_RSI_GH = {
    'rsi_len': ('int', [10, 14, 20]),
    'rsi_ob':  ('int', [65, 70, 75]),
    'rsi_os':  ('int', [25, 30, 35]),
    'ma_len':  ('int', [500, 700, 900]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 31. Stoch_RSI_Cross_EMA  (v4 — Stoch RSI Crossover + EMA [Trade Pro])
# Long: Stoch K crosses D (from below 60, above 10) + EMA fan up + close>EMA1
# Short: Stoch K crosses D (from above 40, below 95) + EMA fan down + close<EMA1
# ─────────────────────────────────────────────────────────────────────────────
def gen_Stoch_RSI_Cross_EMA(df, rsi_len=14, stoch_len=14, k_smooth=3, d_smooth=3,
                             ema1_len=8, ema2_len=14, ema3_len=50, **kw):
    cl = df['close']
    rsi = _rsi(cl, rsi_len)
    lo = rsi.rolling(stoch_len).min()
    hi = rsi.rolling(stoch_len).max()
    k = _sma(100 * (rsi - lo) / (hi - lo + 1e-10), k_smooth)
    d = _sma(k, d_smooth)
    ema1 = _ema(cl, ema1_len)
    ema2 = _ema(cl, ema2_len)
    ema3 = _ema(cl, ema3_len)
    crossup   = (k.shift(1) >= d.shift(1)) & (k.shift(2) <= d.shift(2)) & (k <= 60) & (k >= 10)
    crossdown = (k.shift(1) <= d.shift(1)) & (k.shift(2) >= d.shift(2)) & (k >= 40) & (k <= 95)
    barbuy  = crossup   & (ema1 > ema2) & (ema2 > ema3) & (cl > ema1)
    barsell = crossdown & (ema3 > ema2) & (ema2 > ema1) & (cl < ema1)
    sig = pd.Series(0, index=df.index)
    sig[barbuy]  =  1
    sig[barsell] = -1
    return sig

space_Stoch_RSI_Cross_EMA = {
    'rsi_len':    ('int', [10, 14, 20]),
    'stoch_len':  ('int', [10, 14, 20]),
    'ema1_len':   ('int', [5, 8, 13]),
    'ema2_len':   ('int', [10, 14, 21]),
    'ema3_len':   ('int', [30, 50, 100]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 32. Three_MAF  (v5 — 3MA and Fractals Strategy [geraked])
# Long: EMA1>EMA2>EMA3, low>EMA3, low<EMA1, down fractal
# Short: EMA3>EMA2>EMA1, high<EMA3, high>EMA1, up fractal
# ─────────────────────────────────────────────────────────────────────────────
def gen_Three_MAF(df, len1=20, len2=50, len3=100, frac_n=2, **kw):
    cl, hi, lo = df['close'], df['high'], df['low']
    ma1 = _ema(cl, len1)
    ma2 = _ema(cl, len2)
    ma3 = _ema(cl, len3)
    # Williams fractals
    def up_f(h, n):
        r = pd.Series(False, index=h.index)
        for i in range(n, len(h)-n):
            if all(h.iloc[i] > h.iloc[i-j] for j in range(1,n+1)) and \
               all(h.iloc[i] >= h.iloc[i+j] for j in range(1,n+1)):
                r.iloc[i] = True
        return r
    def dn_f(l, n):
        r = pd.Series(False, index=l.index)
        for i in range(n, len(l)-n):
            if all(l.iloc[i] < l.iloc[i-j] for j in range(1,n+1)) and \
               all(l.iloc[i] <= l.iloc[i+j] for j in range(1,n+1)):
                r.iloc[i] = True
        return r
    up_frac = up_f(hi, frac_n).shift(frac_n).fillna(False).infer_objects(copy=False)
    dn_frac = dn_f(lo, frac_n).shift(frac_n).fillna(False).infer_objects(copy=False)
    long_sig  = (ma1 > ma2) & (ma2 > ma3) & (lo > ma3) & (lo < ma1) & dn_frac
    short_sig = (ma3 > ma2) & (ma2 > ma1) & (hi < ma3) & (hi > ma1) & up_frac
    sig = pd.Series(0, index=df.index)
    sig[long_sig]  =  1
    sig[short_sig] = -1
    return sig

space_Three_MAF = {
    'len1':   ('int', [10, 20, 30]),
    'len2':   ('int', [30, 50, 80]),
    'len3':   ('int', [80, 100, 150]),
    'frac_n': ('int', [2, 3, 4]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 33. CEZLSMA  (v5 — Chandelier Exit & ZLSMA Strategy [geraked])
# Long: Chandelier Exit flips up + close > ZLSMA
# Short: flips down + close < ZLSMA
# ─────────────────────────────────────────────────────────────────────────────
def gen_CEZLSMA(df, ce_len=1, ce_mult=2, zlsma_len=50, **kw):
    cl = df['close']
    atr = ce_mult * _atr(df, ce_len)
    long_stop  = cl.rolling(ce_len).max() - atr
    short_stop = cl.rolling(ce_len).min() + atr
    # Track direction
    ce_dir = pd.Series(1, index=df.index)
    ls = long_stop.copy()
    ss = short_stop.copy()
    for i in range(1, len(df)):
        prev_ls = ls.iloc[i-1]
        prev_ss = ss.iloc[i-1]
        ls.iloc[i] = max(ls.iloc[i], prev_ls) if cl.iloc[i-1] > prev_ls else ls.iloc[i]
        ss.iloc[i] = min(ss.iloc[i], prev_ss) if cl.iloc[i-1] < prev_ss else ss.iloc[i]
        if cl.iloc[i] > prev_ss:   ce_dir.iloc[i] = 1
        elif cl.iloc[i] < prev_ls: ce_dir.iloc[i] = -1
        else: ce_dir.iloc[i] = ce_dir.iloc[i-1]
    buy_sig  = (ce_dir == 1) & (ce_dir.shift(1) == -1)
    sell_sig = (ce_dir == -1) & (ce_dir.shift(1) == 1)
    zlsma = _zlsma(cl, zlsma_len)
    long_close  = _crossunder(cl, zlsma)
    short_close = _crossover(cl, zlsma)
    long_entry  = buy_sig  & (cl > zlsma)
    short_entry = sell_sig & (cl < zlsma)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if long_close.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if short_close.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_CEZLSMA = {
    'ce_len':     ('int',   [1, 2, 3]),
    'ce_mult':    ('int',   [1, 2, 3]),
    'zlsma_len':  ('int',   [30, 50, 80]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 34. CP1  (v5 — Candlestick Pattern 1 [geraked])
# Long: two-bar pattern (green after red, higher close, lower low)
# Short: two-bar pattern (red after green)
# ─────────────────────────────────────────────────────────────────────────────
def gen_CP1(df, **kw):
    cl, op, hi, lo = df['close'], df['open'], df['high'], df['low']
    long_sig  = (cl > op) & (cl.shift(1) > op.shift(1)) & \
                (op.shift(1) < op) & (cl > cl.shift(1)) & \
                (lo < lo.shift(1)) & (lo < op.shift(1))
    short_sig = (cl < op) & (cl.shift(1) < op.shift(1)) & \
                (op.shift(1) > op) & (cl < cl.shift(1)) & \
                (hi > hi.shift(1)) & (hi > op.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[long_sig]  =  1
    sig[short_sig] = -1
    return sig

space_CP1 = {}

# ─────────────────────────────────────────────────────────────────────────────
# 35. LRCUTB  (v5 — Linear Regression Candles + UT Bot [geraked])
# Long: UT Bot buy signal + close > LinReg signal line
# Short: UT Bot sell signal + close < signal
# ─────────────────────────────────────────────────────────────────────────────
def gen_LRCUTB(df, lr_len=11, lr_sig_len=7, ut_key=2, ut_atr=1, **kw):
    cl, hi, lo, op = df['close'], df['high'], df['low'], df['open']
    bclose = _linreg(cl, lr_len)
    bopen  = _linreg(op, lr_len)
    signal = _sma(bclose, lr_sig_len)
    # UT Bot
    xATR  = _atr(df, ut_atr)
    nLoss = ut_key * xATR
    ts = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        prev = ts.iloc[i-1]
        if cl.iloc[i] > prev and cl.iloc[i-1] > prev:
            ts.iloc[i] = max(prev, cl.iloc[i] - nLoss.iloc[i])
        elif cl.iloc[i] < prev and cl.iloc[i-1] < prev:
            ts.iloc[i] = min(prev, cl.iloc[i] + nLoss.iloc[i])
        elif cl.iloc[i] > prev:
            ts.iloc[i] = cl.iloc[i] - nLoss.iloc[i]
        else:
            ts.iloc[i] = cl.iloc[i] + nLoss.iloc[i]
    pos = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if cl.iloc[i-1] < ts.iloc[i-1] and cl.iloc[i] > ts.iloc[i]:   pos.iloc[i] = 1
        elif cl.iloc[i-1] > ts.iloc[i-1] and cl.iloc[i] < ts.iloc[i]: pos.iloc[i] = -1
        else: pos.iloc[i] = pos.iloc[i-1]
    ema1  = _ema(cl, 1)
    above = _crossover(ema1, ts)
    below = _crossover(ts, ema1)
    buy  = (cl > ts) & above
    sell = (cl < ts) & below
    long_entry  = buy  & (cl > signal)
    short_entry = sell & (cl < signal)
    long_close  = (bclose.shift(1) > bopen.shift(1)) & (bclose < bopen)
    short_close = (bclose.shift(1) < bopen.shift(1)) & (bclose > bopen)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if long_close.iloc[i]: in_pos = 0
        elif in_pos == -1:
            if short_close.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_LRCUTB = {
    'lr_len':     ('int', [7, 11, 15]),
    'lr_sig_len': ('int', [5, 7, 10]),
    'ut_key':     ('int', [1, 2, 3]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 36. NWE_RSI_ASF  (v5 — Nadaraya-Watson Env + RSI + ATR SL [geraked])
# Long: low < lower_NWE, green candle, RSI was < 30
# Short: high > upper_NWE, red candle, RSI was > 70
# NWE approximated with rolling gaussian-weighted linreg
# ─────────────────────────────────────────────────────────────────────────────
def gen_NWE_RSI_ASF(df, h=8.0, mult=3.0, rsi_len=5, **kw):
    cl, hi, lo = df['close'], df['high'], df['low']
    # Simplified NWE: use EMA as center, stdev bands
    n = min(len(df), 500)
    nwe_center = cl.ewm(span=int(h * h), adjust=False).mean()
    mae = (cl - nwe_center).abs().rolling(n).mean() * mult
    upper = nwe_center + mae
    lower = nwe_center - mae
    rsi = _rsi(cl, rsi_len)
    long_cond  = (lo.shift(1) < lower.shift(1)) & (cl > df['open']) & (rsi.shift(1) < 30)
    short_cond = (hi.shift(1) > upper.shift(1)) & (cl < df['open']) & (rsi.shift(1) > 70)
    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    return sig

space_NWE_RSI_ASF = {
    'h':       ('float', [5.0, 8.0, 12.0]),
    'mult':    ('float', [2.0, 3.0, 4.0]),
    'rsi_len': ('int',   [3, 5, 8]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 37. STRG_BBForce  (v5 — STRG-BBForce [harryguiacorn])
# Long: all 3 BB components (upper, lower, basis) rise together
# Short: all 3 fall together
# ─────────────────────────────────────────────────────────────────────────────
def gen_STRG_BBForce(df, length=26, mult=2.0, **kw):
    cl = df['close']
    basis = _sma(cl, length)
    dev   = mult * cl.rolling(length).std(ddof=0)
    upper = basis + dev
    lower = basis - dev
    cond_up   = (upper > upper.shift(1)) & (lower > lower.shift(1)) & (basis > basis.shift(1))
    cond_down = (upper < upper.shift(1)) & (lower < lower.shift(1)) & (basis < basis.shift(1))
    # only signal on condition change (new mini signal)
    prev_cond = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if cond_up.iloc[i]:   prev_cond.iloc[i] = 1
        elif cond_down.iloc[i]: prev_cond.iloc[i] = -1
        else: prev_cond.iloc[i] = prev_cond.iloc[i-1]
    cond_mini = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if prev_cond.iloc[i] != prev_cond.iloc[i-1]:
            cond_mini.iloc[i] = prev_cond.iloc[i]
    long_sig  = cond_mini == 1
    short_sig = cond_mini == -1
    sig = pd.Series(0, index=df.index)
    sig[long_sig]  =  1
    sig[short_sig] = -1
    return sig

space_STRG_BBForce = {
    'length': ('int',   [14, 20, 26, 34]),
    'mult':   ('float', [1.5, 2.0, 2.5]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 38. STRG_HOLP  (v5 — STRG-HOLP [harryguiacorn])
# Long: new low in lookback, then close breaks above the high of the lowest bar
# Short: new high in lookback, then close breaks below the low of the highest bar
# ─────────────────────────────────────────────────────────────────────────────
def gen_STRG_HOLP(df, lookback=10, **kw):
    cl, hi, lo = df['close'], df['high'], df['low']
    bool_new_low  = pd.Series(False, index=df.index)
    bool_new_high = pd.Series(False, index=df.index)
    session_low  = pd.Series(np.nan, index=df.index)
    session_high = pd.Series(np.nan, index=df.index)
    long_entry  = pd.Series(False, index=df.index)
    short_entry = pd.Series(False, index=df.index)
    _new_low = False
    _new_high = False
    _slo = np.nan
    _shi = np.nan
    for i in range(lookback, len(df)):
        lo_lb = lo.iloc[max(0, i-lookback):i].min()
        hi_lb = hi.iloc[max(0, i-lookback):i].max()
        lo_lb_prev = lo.iloc[max(0, i-lookback-1):i-1].min()
        hi_lb_prev = hi.iloc[max(0, i-lookback-1):i-1].max()
        if lo.iloc[i] < lo_lb_prev:
            _slo = lo.iloc[i]
            _new_low = True
        if hi.iloc[i] > hi_lb_prev:
            _shi = hi.iloc[i]
            _new_high = True
        # find bar index of lowest in lookback
        lo_window = lo.iloc[max(0, i-lookback):i]
        hi_window = hi.iloc[max(0, i-lookback):i]
        last_lo_bar = lo_window.idxmin()
        last_hi_bar = hi_window.idxmax()
        if cl.iloc[i] > hi.loc[last_lo_bar] and _new_low:
            long_entry.iloc[i] = True
            _new_low = False
        if cl.iloc[i] < lo.loc[last_hi_bar] and _new_high:
            short_entry.iloc[i] = True
            _new_high = False
    sig = pd.Series(0, index=df.index)
    sig[long_entry]  =  1
    sig[short_entry] = -1
    return sig

space_STRG_HOLP = {
    'lookback': ('int', [5, 8, 10, 15]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 39. STRG_Kijun  (v5 — STRG-Kijun [harryguiacorn])
# Long: Kijun-Sen turns up (from flat/down to rising)
# Short: Kijun-Sen turns down
# ─────────────────────────────────────────────────────────────────────────────
def gen_STRG_Kijun(df, base_periods=26, **kw):
    hi, lo = df['high'], df['low']
    kijun = (hi.rolling(base_periods).max() + lo.rolling(base_periods).min()) / 2
    k_dir = np.sign(kijun.diff())
    direction = pd.Series(0, index=df.index)
    prev_dir = 0
    for i in range(len(df)):
        cur = int(k_dir.iloc[i]) if not np.isnan(k_dir.iloc[i]) else 0
        if cur != 0:
            direction.iloc[i] = cur if cur != prev_dir else 0
            prev_dir = cur
    long_entry  = direction == 1
    short_entry = direction == -1
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if long_entry.iloc[i]:  in_pos = 1
        elif short_entry.iloc[i]: in_pos = -1
        sig.iloc[i] = in_pos
    return sig

space_STRG_Kijun = {
    'base_periods': ('int', [13, 20, 26, 34]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 40. One_Bar_Pursuit  (v5 — STRG One Bar Pursuit [FxCloudTrader])
# Long: bearish bar followed by bullish bar
# Short: bullish bar followed by bearish bar
# Exit: opposite color candle
# ─────────────────────────────────────────────────────────────────────────────
def gen_One_Bar_Pursuit(df, **kw):
    cl, op = df['close'], df['open']
    bull = cl > op
    bear = cl < op
    long_entry  = bear.shift(1) & bull
    short_entry = bull.shift(1) & bear
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if long_entry.iloc[i]:  in_pos = 1
        elif short_entry.iloc[i]: in_pos = -1
        if in_pos == 1  and bear.iloc[i]: in_pos = 0
        if in_pos == -1 and bull.iloc[i]: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_One_Bar_Pursuit = {}

# ─────────────────────────────────────────────────────────────────────────────
# 41. Weighted_Strategy  (v5 — Acrypto Weighted Strategy [AlbertoCuadra])
# Combines: MACD cross + StochRSI + RSI + Supertrend + MA cross with weights
# ─────────────────────────────────────────────────────────────────────────────
def gen_Weighted_Strategy(df, macd_fast=16, macd_slow=36, srsi_len=14,
                           srsi_stoch=14, srsi_smooth=3, st_period=2,
                           st_mult=2.4, ma1_fast=46, ma2_slow=82, **kw):
    cl = df['close']
    src = (df['high'] + df['low']) / 2  # hl2
    # MACD
    macd, signal, _ = _macd(src, macd_fast, macd_slow, 9)
    macd_bull = _crossover(macd, signal)
    macd_bear = _crossunder(macd, signal)
    # StochRSI
    k, d = _stochrsi(cl, srsi_len, srsi_stoch, srsi_smooth, srsi_smooth)
    srsi_bull = k < 20
    srsi_bear = k > 80
    # RSI
    rsi = _rsi(cl, 14)
    rsi_bull = rsi < 30
    rsi_bear = rsi > 70
    # Supertrend
    st = _supertrend(df, st_period, st_mult)
    st_bull = st == 1
    st_bear = st == -1
    # MA Cross
    ma1 = _ema(cl, ma1_fast)
    ma2 = _ema(cl, ma2_slow)
    ma_bull = _crossover(ma1, ma2)
    ma_bear = _crossunder(ma1, ma2)
    # weighted sum
    long_score  = macd_bull.astype(int) + srsi_bull.astype(int) + rsi_bull.astype(int) + st_bull.astype(int) + ma_bull.astype(int)
    short_score = macd_bear.astype(int) + srsi_bear.astype(int) + rsi_bear.astype(int) + st_bear.astype(int) + ma_bear.astype(int)
    long_entry  = long_score  >= 2
    short_entry = short_score >= 2
    sig = pd.Series(0, index=df.index)
    sig[long_entry]  =  1
    sig[short_entry] = -1
    return sig

space_Weighted_Strategy = {
    'macd_fast':  ('int',   [12, 16, 20]),
    'macd_slow':  ('int',   [30, 36, 46]),
    'st_period':  ('int',   [2, 3, 5]),
    'st_mult':    ('float', [2.0, 2.4, 3.0]),
    'ma1_fast':   ('int',   [30, 46, 60]),
    'ma2_slow':   ('int',   [60, 82, 100]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 42. Heikin_Ashi_V2  (v4 — Heikin Ashi Strategy V2 [Alorse])
# Long: fast HA EMA crossover slow HA EMA
# Short: crossunder
# ─────────────────────────────────────────────────────────────────────────────
def gen_Heikin_Ashi_V2(df, fast_ema=1, slow_ema=30, **kw):
    cl, op, hi, lo = df['close'], df['open'], df['high'], df['low']
    # Approximate HA
    ha_cl = (op + hi + lo + cl) / 4
    ha_op = pd.Series(index=df.index, dtype=float)
    ha_op.iloc[0] = (op.iloc[0] + cl.iloc[0]) / 2
    for i in range(1, len(df)):
        ha_op.iloc[i] = (ha_op.iloc[i-1] + ha_cl.iloc[i-1]) / 2
    fma = _ema(ha_cl, fast_ema)
    sma = _ema(ha_cl, slow_ema)
    golong  = _crossover(fma, sma)
    goshort = _crossunder(fma, sma)
    sig = pd.Series(0, index=df.index)
    sig[golong]  =  1
    sig[goshort] = -1
    return sig

space_Heikin_Ashi_V2 = {
    'fast_ema': ('int', [1, 2, 3]),
    'slow_ema': ('int', [15, 30, 50]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 43. MTF_BB_GH  (v4 — MTF Bollinger Bands [Alorse])
# Long: close < avg(lower, mtf_lower)
# Exit: close > upper
# MTF simulated with 3× factor
# ─────────────────────────────────────────────────────────────────────────────
def gen_MTF_BB_GH(df, length=20, mult=2.0, mtf_factor=3, **kw):
    cl = df['close']
    basis = _sma(cl, length)
    dev   = mult * cl.rolling(length).std(ddof=0)
    upper = basis + dev
    lower = basis - dev
    basis_m = _sma(cl, length * mtf_factor)
    dev_m   = mult * cl.rolling(length * mtf_factor).std(ddof=0)
    lower_m = basis_m - dev_m
    upper_m = basis_m + dev_m
    avg_lower = (lower + lower_m) / 2
    buy  = cl < avg_lower
    exit = cl > upper
    sig = pd.Series(0, index=df.index)
    in_long = False
    for i in range(len(df)):
        if buy.iloc[i]:  in_long = True
        if exit.iloc[i]: in_long = False
        sig.iloc[i] = 1 if in_long else 0
    return sig

space_MTF_BB_GH = {
    'length':     ('int',   [10, 20, 30]),
    'mult':       ('float', [1.5, 2.0, 2.5]),
    'mtf_factor': ('int',   [2, 3, 4]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 44. MACD_Divergences  (v4 — MACD + Divergences [Alorse] / MacdNew)
# Long: MACD crosses up + close > EMA_fast > EMA_slow
# Short: MACD crosses down + close < EMA_fast < EMA_slow
# ─────────────────────────────────────────────────────────────────────────────
def gen_MACD_Divergences(df, fast=12, slow=26, sig_len=9,
                         ma_a=150, ma_b=600, **kw):
    cl = df['close']
    macd, signal, _ = _macd(cl, fast, slow, sig_len)
    ema_a = _ema(cl, ma_a)
    ema_b = _ema(cl, ma_b)
    bull = _crossover(macd, signal) & (ema_a > ema_b)
    bear = _crossunder(macd, signal) & (ema_a < ema_b)
    sig = pd.Series(0, index=df.index)
    sig[bull] =  1
    sig[bear] = -1
    return sig

space_MACD_Divergences = {
    'fast':   ('int', [8, 12, 16]),
    'slow':   ('int', [20, 26, 34]),
    'ma_a':   ('int', [100, 150, 200]),
    'ma_b':   ('int', [400, 600, 800]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 45. StochRSI_ST  (v4 — StochRSI + Supertrend [Alorse])
# Long: in up-trend (supertrend=1) + stochRSI oversold cross
# ─────────────────────────────────────────────────────────────────────────────
def gen_StochRSI_ST_GH(df, k_smooth=3, d_smooth=3, rsi_len=14, stoch_len=14,
                        atr_period=10, st_mult=3.0, **kw):
    cl = df['close']
    trend = _supertrend(df, atr_period, st_mult)
    k, d = _stochrsi(cl, rsi_len, stoch_len, k_smooth, d_smooth)
    # Long: trend up + K crosses over D from oversold
    long_entry  = (trend == 1) & _crossover(k, d) & (k < 30)
    short_entry = (trend == -1) & _crossunder(k, d) & (k > 70)
    sig = pd.Series(0, index=df.index)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0:
            if long_entry.iloc[i]:  in_pos = 1
            elif short_entry.iloc[i]: in_pos = -1
        elif in_pos == 1:
            if trend.iloc[i] == -1: in_pos = 0
        elif in_pos == -1:
            if trend.iloc[i] == 1: in_pos = 0
        sig.iloc[i] = in_pos
    return sig

space_StochRSI_ST_GH = {
    'k_smooth':   ('int',   [2, 3, 5]),
    'rsi_len':    ('int',   [10, 14, 20]),
    'atr_period': ('int',   [7, 10, 14]),
    'st_mult':    ('float', [2.0, 3.0, 4.0]),
}

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY EXPORT
# ─────────────────────────────────────────────────────────────────────────────
STRATEGY_EXPORT = {
    'Two_EMA_RSI':        {'gen': gen_Two_EMA_RSI,        'space': space_Two_EMA_RSI},
    'Three_EMA_Cross':    {'gen': gen_Three_EMA_Cross,    'space': space_Three_EMA_Cross},
    'BB_Divergence_GH':   {'gen': gen_BB_Divergence_GH,   'space': space_BB_Divergence_GH},
    'BB_Winner_Lite':     {'gen': gen_BB_Winner_Lite,      'space': space_BB_Winner_Lite},
    'BB_Winner_Pro':      {'gen': gen_BB_Winner_Pro,       'space': space_BB_Winner_Pro},
    'BB_Aroon':           {'gen': gen_BB_Aroon,            'space': space_BB_Aroon},
    'Double_RSI':         {'gen': gen_Double_RSI,          'space': space_Double_RSI},
    'EMA_Moving_Away':    {'gen': gen_EMA_Moving_Away,     'space': space_EMA_Moving_Away},
    'Exceeded_Candle':    {'gen': gen_Exceeded_Candle,     'space': space_Exceeded_Candle},
    'Full_Candle':        {'gen': gen_Full_Candle,         'space': space_Full_Candle},
    'MACD_RSI_GH':        {'gen': gen_MACD_RSI_GH,         'space': space_MACD_RSI_GH},
    'MACD_BB_RSI':        {'gen': gen_MACD_BB_RSI,         'space': space_MACD_BB_RSI},
    'MACD_DMI':           {'gen': gen_MACD_DMI,            'space': space_MACD_DMI},
    'MA_Cross_DMI':       {'gen': gen_MA_Cross_DMI,        'space': space_MA_Cross_DMI},
    'MEMA_BB_RSI':        {'gen': gen_MEMA_BB_RSI,         'space': space_MEMA_BB_RSI},
    'Multi_BB':           {'gen': gen_Multi_BB,            'space': space_Multi_BB},
    'Pin_Bar_Magic':      {'gen': gen_Pin_Bar_Magic,       'space': space_Pin_Bar_Magic},
    'QQE_Signals':        {'gen': gen_QQE_Signals,         'space': space_QQE_Signals},
    'RSI_1200':           {'gen': gen_RSI_1200,            'space': space_RSI_1200},
    'RSI_EMA':            {'gen': gen_RSI_EMA,             'space': space_RSI_EMA},
    'ST_Simple':          {'gen': gen_ST_Simple,           'space': space_ST_Simple},
    'ST_EMA_Rebound':     {'gen': gen_ST_EMA_Rebound,      'space': space_ST_EMA_Rebound},
    'ST_RSI':             {'gen': gen_ST_RSI,              'space': space_ST_RSI},
    'Tendency_EMA_RSI':   {'gen': gen_Tendency_EMA_RSI,    'space': space_Tendency_EMA_RSI},
    'MACD_Long_GH':       {'gen': gen_MACD_Long_GH,        'space': space_MACD_Long_GH},
    'Improvising_GH':     {'gen': gen_Improvising_GH,      'space': space_Improvising_GH},
    'Omar_MMR':           {'gen': gen_Omar_MMR,            'space': space_Omar_MMR},
    'Omar_Edited_WF':     {'gen': gen_Omar_Edited_WF,      'space': space_Omar_Edited_WF},
    'HA_SSL':             {'gen': gen_HA_SSL,              'space': space_HA_SSL},
    'MTF_RSI_GH':         {'gen': gen_MTF_RSI_GH,          'space': space_MTF_RSI_GH},
    'Stoch_RSI_Cross_EMA':{'gen': gen_Stoch_RSI_Cross_EMA,'space': space_Stoch_RSI_Cross_EMA},
    'Three_MAF':          {'gen': gen_Three_MAF,           'space': space_Three_MAF},
    'CEZLSMA':            {'gen': gen_CEZLSMA,             'space': space_CEZLSMA},
    'CP1':                {'gen': gen_CP1,                 'space': space_CP1},
    'LRCUTB':             {'gen': gen_LRCUTB,              'space': space_LRCUTB},
    'NWE_RSI_ASF':        {'gen': gen_NWE_RSI_ASF,         'space': space_NWE_RSI_ASF},
    'STRG_BBForce':       {'gen': gen_STRG_BBForce,        'space': space_STRG_BBForce},
    'STRG_HOLP':          {'gen': gen_STRG_HOLP,           'space': space_STRG_HOLP},
    'STRG_Kijun':         {'gen': gen_STRG_Kijun,          'space': space_STRG_Kijun},
    'One_Bar_Pursuit':    {'gen': gen_One_Bar_Pursuit,     'space': space_One_Bar_Pursuit},
    'Weighted_Strategy':  {'gen': gen_Weighted_Strategy,   'space': space_Weighted_Strategy},
    'Heikin_Ashi_V2':     {'gen': gen_Heikin_Ashi_V2,      'space': space_Heikin_Ashi_V2},
    'MTF_BB_GH':          {'gen': gen_MTF_BB_GH,           'space': space_MTF_BB_GH},
    'MACD_Divergences':   {'gen': gen_MACD_Divergences,    'space': space_MACD_Divergences},
    'StochRSI_ST_GH':     {'gen': gen_StochRSI_ST_GH,      'space': space_StochRSI_ST_GH},
}
