#!/usr/bin/env python3
"""
TV2 BATCH 41 — Freqtrade strategies adapted to Bot V7 format
Sources:
  - jilv220/BB_RPB_TSL (212 stars) — BB/RSI/EWO multi-signal strategy
  - mikedigriz/freqtrade-strategy-mikedigriz (39 stars) — HMA, CCI, RSI, FisherRSI
  - keithorange/FreqTradeCustomOrders (14 stars) — MA slope strategies
  - Rikj000/MoniGoMani (1025 stars) — weighted signal system
  - paulcpk/freqtrade-strategies-that-work (321 stars) — EMA/MACD/RSI with trend
  - froggleston/cryptofrog-strategies (183 stars) — multi-indicator BB/Stoch/MFI

Adaptation: freqtrade populate_buy_trend/populate_sell_trend → gen_NAME(df) returning
Series(0/1/-1) for neutral/long/short.

Total: 26 strategies
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
def _bb(s, n, std=2):
    m = _sma(s, n)
    sd = s.rolling(n).std()
    return m - std*sd, m, m + std*sd
def _stoch(df, k, d):
    lo = df['low'].rolling(k).min()
    hi = df['high'].rolling(k).max()
    k_line = 100*(df['close']-lo)/(hi-lo+1e-10)
    return k_line, k_line.rolling(d).mean()
def _macd(s, fast=12, slow=26, sig=9):
    m = _ema(s, fast) - _ema(s, slow)
    return m, _ema(m, sig), m - _ema(m, sig)
def _wma(s, n):
    w = np.arange(1, n+1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
def _hma(s, n):
    return _wma(2*_wma(s, n//2) - _wma(s, n), max(1, int(np.sqrt(n))))
def _crossover(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))
def _crossunder(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))
def _typical_price(df):
    return (df['high'] + df['low'] + df['close']) / 3.0
def _bb_typical(df, n=20, std=2):
    """Bollinger Bands on typical price (like qtpylib version)"""
    tp = _typical_price(df)
    m = _sma(tp, n)
    sd = tp.rolling(n).std()
    return m - std*sd, m, m + std*sd
def _williams_r(df, period=14):
    """Williams %R"""
    hh = df['high'].rolling(period).max()
    ll = df['low'].rolling(period).min()
    return -100 * (hh - df['close']) / (hh - ll + 1e-10)
def _cci(df, n=14):
    """Commodity Channel Index"""
    tp = _typical_price(df)
    m = _sma(tp, n)
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - m) / (0.015 * mad + 1e-10)
def _mfi(df, n=14):
    """Money Flow Index"""
    tp = _typical_price(df)
    raw_mf = tp * df['volume']
    pos_mf = raw_mf.where(tp > tp.shift(1), 0.0)
    neg_mf = raw_mf.where(tp < tp.shift(1), 0.0)
    mfr = pos_mf.rolling(n).sum() / (neg_mf.rolling(n).sum() + 1e-10)
    return 100 - 100 / (1 + mfr)
def _cmf(df, n=20):
    """Chaikin Money Flow"""
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
    mfv = mfv * df['volume']
    return mfv.rolling(n).sum() / (df['volume'].rolling(n).sum() + 1e-10)
def _ewo(df, fast=5, slow=35):
    """Elliott Wave Oscillator = (EMA_fast - EMA_slow) / low * 100"""
    return (_ema(df['close'], fast) - _ema(df['close'], slow)) / df['low'] * 100
def _fisher_rsi(s, n=14):
    """Fisher Transform of RSI"""
    r = _rsi(s, n)
    x = 0.1 * (r - 50)
    ex2 = np.exp(2 * x)
    return (ex2 - 1) / (ex2 + 1)
def _dmi(df, n=14):
    """Returns (diplus, diminus, adx)"""
    hi, lo, cl = df['high'], df['low'], df['close']
    tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    up_move = hi.diff()
    dn_move = -lo.diff()
    dm_plus = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=df.index)
    dm_minus = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=df.index)
    atr_n = _rma(tr, n)
    di_plus = 100 * _rma(dm_plus, n) / (atr_n + 1e-10)
    di_minus = 100 * _rma(dm_minus, n) / (atr_n + 1e-10)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10)
    adx = _rma(dx, n)
    return di_plus, di_minus, adx
def _stochrsi(s, rsi_n=14, stoch_n=14, k=3, d=3):
    """Stochastic RSI"""
    rsi = _rsi(s, rsi_n)
    rsi_min = rsi.rolling(stoch_n).min()
    rsi_max = rsi.rolling(stoch_n).max()
    srsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    k_line = srsi.rolling(k).mean()
    d_line = k_line.rolling(d).mean()
    return k_line, d_line
def _roc(s, n):
    """Rate of Change"""
    return 100 * (s - s.shift(n)) / (s.shift(n) + 1e-10)
def _sar(df, iaf=0.02, maxaf=0.2):
    """Parabolic SAR — simplified vectorised approximation"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    n = len(close)
    sar = np.full(n, np.nan)
    trend = 1  # 1=up, -1=down
    ep = low[0]
    af = iaf
    sar[0] = high[0]
    for i in range(1, n):
        prev_sar = sar[i-1]
        if trend == 1:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], low[i-1], low[i-2] if i > 1 else low[i-1])
            if low[i] < sar[i]:
                trend = -1
                sar[i] = ep
                ep = low[i]
                af = iaf
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + iaf, maxaf)
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
            if high[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = high[i]
                af = iaf
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + iaf, maxaf)
    return pd.Series(sar, index=df.index)
def _vwap(df):
    """Simple VWAP (resets at start of data)"""
    tp = _typical_price(df)
    return (tp * df['volume']).cumsum() / df['volume'].cumsum()
def _rmi(s, length=20, mom=5):
    """Relative Momentum Index"""
    d = s.diff(mom)
    gain = _rma(d.clip(lower=0), length)
    loss = _rma((-d).clip(lower=0), length)
    return 100 - 100 / (1 + gain / (loss + 1e-10))

# ─── 1. BB_RPB_TSL — Dip signal (BB + CCI + RMI + StochRSI) ─────────────────
def gen_BB_RPB_TSL(df, **p):
    """
    Source: jilv220/BB_RPB_TSL — 'dip' buy condition from BB_RPB_TSL.
    Buy: CCI oversold + RMI < 50 + StochRSI fastk < 30 + BB expansion
    Short: BB upper breakout + CCI overbought
    """
    cci = _cci(df, 25)
    rmi = _rmi(df['close'], 17, 5)
    srsi_k, _ = _stochrsi(df['close'], 14, 14, 3, 3)
    bb_lo, bb_mid, bb_hi = _bb_typical(df, 20, 2)
    bb_width = (bb_hi - bb_lo) / (bb_mid + 1e-10)

    long_cond = (
        (cci < -116) &
        (rmi < 49) &
        (srsi_k < 32) &
        (bb_width > 0.095)
    )
    short_cond = (
        (cci > 100) &
        (rmi > 65) &
        (df['close'] > bb_hi)
    )
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 2. BB_RPB_EWO — EWO + EMA dip ──────────────────────────────────────────
def gen_BB_RPB_EWO(df, **p):
    """
    Source: jilv220/BB_RPB_TSL — 'ewo' buy condition.
    Buy: EWO < -5 + RSI < 23 + close < EMA50 * 0.968 + close > EMA200 * 0.935
    Short: EWO > 5 + RSI > 77 + close > EMA50 * 1.05
    """
    ewo = _ewo(df, 5, 35)
    rsi = _rsi(df['close'], 14)
    rsi_fast = _rsi(df['close'], 4)
    ema50 = _ema(df['close'], 50)
    ema200 = _ema(df['close'], 200)

    long_cond = (
        (ewo < -5.001) &
        (rsi < 23) &
        (rsi_fast < 44) &
        (df['close'] < ema50 * 0.968) &
        (df['close'] > ema200 * 0.935)
    )
    short_cond = (
        (ewo > 5.0) &
        (rsi > 77) &
        (df['close'] > ema50 * 1.05)
    )
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 3. BB_RPB_CLUCHA — Bollinger lower band reversal ────────────────────────
def gen_BB_RPB_Clucha(df, **p):
    """
    Source: jilv220/BB_RPB_TSL — 'clucha' condition.
    Buy: BB delta large + close near lower band + close delta small + tail ratio
    Short: price above upper band + CMF negative
    """
    bb_lo, bb_mid, bb_hi = _bb_typical(df, 20, 2)
    bb_delta = bb_hi - bb_lo
    close_delta = df['close'].diff().abs()
    tail = df['low'] - bb_lo
    cmf = _cmf(df, 20)

    long_cond = (
        (bb_delta / df['close'] > 0.049) &
        (bb_delta.shift(1) / df['close'].shift(1) > 1.146 * bb_delta / df['close']) &
        (df['close'] - bb_lo < 0.018 * df['close']) &
        (close_delta / df['close'] < 0.017) &
        (df['close'] <= bb_lo + 0.01 * df['close'])
    )
    short_cond = (
        (df['close'] > bb_hi * 0.99) &
        (cmf < -0.046)
    )
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 4. BB_RPB_Deadfish — Deadfish sell pattern (reversal of consolidation) ──
def gen_BB_RPB_Deadfish(df, **p):
    """
    Source: jilv220/BB_RPB_TSL — 'r_deadfish' reverse deadfish condition.
    Buy: close above EMA * factor + narrow BB + CTI oversold + Williams R oversold
    Short: consolidation band break down
    """
    ema200 = _ema(df['close'], 200)
    bb_lo, bb_mid, bb_hi = _bb_typical(df, 20, 2)
    bb_width = (bb_hi - bb_lo) / (bb_mid + 1e-10)
    wr14 = _williams_r(df, 14)
    rsi = _rsi(df['close'], 14)
    # CTI approximation: correlation of close with linear trend
    def _cti_approx(s, n=20):
        x = np.arange(n, dtype=float)
        def corr_with_trend(arr):
            if np.any(np.isnan(arr)): return np.nan
            c = np.corrcoef(x, arr)
            return c[0, 1] if not np.isnan(c[0, 1]) else 0.0
        return s.rolling(n).apply(corr_with_trend, raw=True)
    cti = _cti_approx(df['close'], 20)

    long_cond = (
        (df['close'] > ema200 * 1.054) &
        (bb_width > 0.299) &
        (cti < -0.115) &
        (wr14 < -44.34)
    )
    short_cond = (
        (df['close'] < ema200 * 0.98) &
        (bb_width < 0.05) &
        (rsi > 60)
    )
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 5. BB_RPB_COFI — Cofi pattern: stochastic + EWO + ADX ──────────────────
def gen_BB_RPB_COFI(df, **p):
    """
    Source: jilv220/BB_RPB_TSL — 'cofi' condition.
    Buy: fastk < 39 + fastd < 28 + EWO high + EMA cofi filter
    Short: fastk > 80 + fastd > 80 + EWO low
    """
    stoch_k, stoch_d = _stoch(df, 5, 3)  # stoch fast: 5/3
    ewo = _ewo(df, 5, 35)
    ema50 = _ema(df['close'], 50)
    di_plus, di_minus, adx = _dmi(df, 14)

    long_cond = (
        (stoch_k < 39) &
        (stoch_d < 28) &
        (ewo > 8.594) &
        (df['close'] < ema50 * 1.147) &
        (adx > 13) &
        (di_plus > di_minus)
    )
    short_cond = (
        (stoch_k > 80) &
        (stoch_d > 80) &
        (ewo < -5.0) &
        (adx > 20)
    )
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 6. BB_RPB_SqzMom — Squeeze Momentum ─────────────────────────────────────
def gen_BB_RPB_SqzMom(df, **p):
    """
    Source: jilv220/BB_RPB_TSL — 'sqzmom' condition.
    Buy: close > EMA * 0.981 + EWO < -3.966 + Williams R < -45
    Short: close < EMA * 0.97 + EWO > 3.966 + Williams R > -5
    """
    ema200 = _ema(df['close'], 200)
    ewo = _ewo(df, 5, 35)
    wr14 = _williams_r(df, 14)

    long_cond = (
        (df['close'] > ema200 * 0.981) &
        (ewo < -3.966) &
        (wr14 < -45.068)
    )
    short_cond = (
        (df['close'] < ema200 * 0.97) &
        (ewo > 3.966) &
        (wr14 > -5.0)
    )
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 7. HMA_CrossSignal — Hull MA crossover (BuyOrDie + EasyInEasyOut) ───────
def gen_HMA_CrossSignal(df, **p):
    """
    Source: mikedigriz/BuyOrDie + EasyInEasyOut — HMA(20) crossover.
    Buy: close crosses above HMA20 (close_prev < hma_prev AND close_curr > hma_curr)
    Short: close crosses below HMA20
    """
    hma20 = _hma(df['close'], 20)
    close_prev = df['close'].shift(2)
    hma_prev = hma20.shift(2)
    close_curr = df['close'].shift(1)
    hma_curr = hma20.shift(1)

    long_cond = (close_curr > hma_curr) & (close_prev < hma_prev)
    short_cond = (close_curr < hma_curr) & (close_prev > hma_prev)

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 8. CCI_BB_Oversold — CCI + BB oversold reversal ────────────────────────
def gen_CCI_BB_Oversold(df, **p):
    """
    Source: mikedigriz/CCI_BB — CCI < -134 + close below BB lower band
    Short (added): CCI > 134 + close above BB upper band
    """
    cci = _cci(df, 14)
    bb_lo, _, bb_hi = _bb_typical(df, 20, 2)
    bb_lo1, _, bb_hi3 = _bb_typical(df, 20, 1), _bb(df['close'], 20, 3)[2], _bb(df['close'], 20, 1)[0]

    long_cond = (
        (cci <= -134) &
        (df['close'] < bb_lo)
    )
    short_cond = (
        (cci >= 134) &
        (df['close'] > bb_hi)
    )
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 9. FisherHull_Reversal — Fisher RSI + HMA + CCI ────────────────────────
def gen_FisherHull_Reversal(df, **p):
    """
    Source: mikedigriz/FisherHull — HMA trending down + CCI < -50 + Fisher RSI < -0.5
    Long: HMA falling + oversold (buy the dip)
    Short: HMA rising + overbought
    """
    hma14 = _hma(df['close'], 14)
    cci = _cci(df, 14)
    fisher = _fisher_rsi(df['close'], 14)

    hma_falling = hma14 < hma14.shift(1)
    hma_rising = hma14 > hma14.shift(1)

    long_cond = (
        hma_falling &
        (cci <= -50.0) &
        (fisher < -0.5) &
        (df['volume'] > 0)
    )
    short_cond = (
        hma_rising &
        (cci >= 100.0) &
        (fisher > 0.5) &
        (df['volume'] > 0)
    )
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 10. RSI_BB_Bounce — RSI + BB bounce ────────────────────────────────────
def gen_RSI_BB_Bounce(df, **p):
    """
    Source: mikedigriz/RSI_BB
    Buy: close < BB lower band (1 std)
    Short: RSI > 56 + close > BB upper (3 std)
    """
    rsi = _rsi(df['close'], 14)
    bb_lo1, _, _ = _bb_typical(df, 20, 1)
    _, _, bb_hi3 = _bb_typical(df, 20, 3)

    long_cond = df['close'] < bb_lo1
    short_cond = (rsi > 56) & (df['close'] > bb_hi3)

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 11. SmartMoney_CMF — CMF + MFI + EMA200 trend ─────────────────────────
def gen_SmartMoney_CMF(df, **p):
    """
    Source: mikedigriz/smart_money_strategy
    Buy: close < EMA200 + MFI < 35 + CMF < -0.07 (dip in downtrend reversal)
    Short: close > EMA200 + MFI > 70 + CMF > 0.20
    """
    ema200 = _ema(df['close'], 200)
    mfi = _mfi(df, 14)
    cmf = _cmf(df, 20)

    long_cond = (
        (df['close'] < ema200) &
        (mfi < 35) &
        (cmf < -0.07)
    )
    short_cond = (
        (df['close'] > ema200) &
        (mfi > 70) &
        (cmf > 0.20)
    )
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 12. MASlope_EMA_Trend — MA slope linear regression ─────────────────────
def gen_MASlope_EMA_Trend(df, **p):
    """
    Source: keithorange/MASlopeStrategy — slope of EMA as trend filter.
    Buy: EMA slope positive + close crosses above EMA
    Short: EMA slope negative + close crosses below EMA
    """
    period = p.get('period', 50)
    slope_period = p.get('slope_period', 14)
    ema = _ema(df['close'], period)

    # Linear regression slope over rolling window
    def _slope(arr):
        if np.any(np.isnan(arr)): return np.nan
        n = len(arr)
        x = np.arange(n, dtype=float)
        m, _ = np.polyfit(x, arr, 1)
        return m
    ma_slope = ema.rolling(slope_period).apply(_slope, raw=True)

    long_cond = (ma_slope > 0) & _crossover(df['close'], ema)
    short_cond = (ma_slope < 0) & _crossunder(df['close'], ema)

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 13. MASlope_HMA_Trend — HMA slope variant ──────────────────────────────
def gen_MASlope_HMA_Trend(df, **p):
    """
    Source: keithorange/MASlopeStrategy — HMA variant.
    Buy: HMA slope positive + price above HMA
    Short: HMA slope negative + price below HMA
    """
    period = p.get('period', 30)
    slope_period = p.get('slope_period', 10)
    hma = _hma(df['close'], period)

    def _slope(arr):
        if np.any(np.isnan(arr)): return np.nan
        n = len(arr)
        x = np.arange(n, dtype=float)
        m, _ = np.polyfit(x, arr, 1)
        return m
    hma_slope = hma.rolling(slope_period).apply(_slope, raw=True)

    long_cond = (hma_slope > 0) & (df['close'] > hma) & (df['close'].shift(1) <= hma.shift(1))
    short_cond = (hma_slope < 0) & (df['close'] < hma) & (df['close'].shift(1) >= hma.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 14. MoniGoMani_BB_EMA — Weighted BB + EMA golden cross ─────────────────
def gen_MoniGoMani_BB_EMA(df, **p):
    """
    Source: Rikj000/MoniGoMani — bollinger_bands + ema golden cross signals.
    Buy: close crosses above BB lower + EMA9 crosses above EMA50 + above EMA200
    Short: close crosses below BB upper + EMA9 crosses below EMA50
    """
    bb_lo, _, bb_hi = _bb_typical(df, 20, 2)
    ema9 = _ema(df['close'], 9)
    ema50 = _ema(df['close'], 50)
    ema200 = _ema(df['close'], 200)

    bb_cross_above_lower = _crossover(df['close'], bb_lo)
    bb_cross_below_upper = _crossunder(df['close'], bb_hi)
    ema_golden = _crossover(ema9, ema50)
    ema_death = _crossunder(ema9, ema50)

    long_cond = (
        bb_cross_above_lower &
        (df['close'] > ema200)
    ) | ema_golden
    short_cond = (
        bb_cross_below_upper &
        (df['close'] < ema200)
    ) | ema_death

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 15. MoniGoMani_MACD_RSI — MACD + RSI weighted ──────────────────────────
def gen_MoniGoMani_MACD_RSI(df, **p):
    """
    Source: Rikj000/MoniGoMani — macd + rsi signals.
    Buy: MACD above signal + RSI crosses above 30
    Short: MACD below signal + RSI crosses below 70
    """
    macd, macdsig, _ = _macd(df['close'], 12, 26, 9)
    rsi = _rsi(df['close'], 14)

    long_cond = (
        (macd > macdsig) &
        _crossover(rsi, pd.Series(30, index=df.index))
    )
    short_cond = (
        (macd < macdsig) &
        _crossunder(rsi, pd.Series(70, index=df.index))
    )
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 16. MoniGoMani_SMA_Cross — SMA golden/death cross ──────────────────────
def gen_MoniGoMani_SMA_Cross(df, **p):
    """
    Source: Rikj000/MoniGoMani — sma golden/death cross.
    Buy: SMA9 crosses above SMA50 + above SMA200 (long-term golden cross)
    Short: SMA9 crosses below SMA50 + below SMA200
    """
    sma9 = _sma(df['close'], 9)
    sma50 = _sma(df['close'], 50)
    sma200 = _sma(df['close'], 200)

    long_cond = _crossover(sma9, sma50) & (df['close'] > sma200)
    short_cond = _crossunder(sma9, sma50) & (df['close'] < sma200)

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 17. MoniGoMani_ADX_VWAP — ADX + VWAP ───────────────────────────────────
def gen_MoniGoMani_ADX_VWAP(df, **p):
    """
    Source: Rikj000/MoniGoMani — adx strong + vwap cross.
    Buy: ADX > 25 + DI+ > DI- + close crosses above VWAP
    Short: ADX > 25 + DI- > DI+ + close crosses below VWAP
    """
    di_plus, di_minus, adx = _dmi(df, 14)
    vwap = _vwap(df)

    long_cond = (
        (adx > 25) &
        (di_plus > di_minus) &
        _crossover(df['close'], vwap)
    )
    short_cond = (
        (adx > 25) &
        (di_minus > di_plus) &
        _crossunder(df['close'], vwap)
    )
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 18. DoubleEMA_Crossover_Trend — Double EMA cross + EMA200 trend ─────────
def gen_DoubleEMA_Crossover_Trend(df, **p):
    """
    Source: paulcpk/DoubleEMACrossoverWithTrend
    Buy: EMA9 crosses above EMA21 + price above EMA200
    Short: EMA9 crosses below EMA21 OR price below EMA200
    """
    ema9 = _ema(df['close'], 9)
    ema21 = _ema(df['close'], 21)
    ema200 = _ema(df['close'], 200)

    long_cond = (
        _crossover(ema9, ema21) &
        (df['low'] > ema200) &
        (df['volume'] > 0)
    )
    short_cond = (
        _crossunder(ema9, ema21) |
        (df['low'] < ema200)
    )
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 19. EMA_Price_Crossover_Threshold — EMA800 price crossover ──────────────
def gen_EMA_Price_Crossover_Threshold(df, **p):
    """
    Source: paulcpk/EMAPriceCrossoverWithThreshold
    Buy: close crosses above EMA200 (using 200 instead of 800 for crypto speed)
    Short: close crosses below EMA200 * 0.99 (threshold)
    """
    ema200 = _ema(df['close'], 200)
    threshold = ema200 * 0.99

    long_cond = (
        _crossover(df['close'], ema200) &
        (df['volume'] > 0)
    )
    short_cond = _crossunder(df['close'], threshold)

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 20. MACD_Crossover_Trend — MACD crossover with EMA100 trend ─────────────
def gen_MACD_Crossover_Trend(df, **p):
    """
    Source: paulcpk/MACDCrossoverWithTrend
    Buy: MACD < 0 + MACD crosses above signal + low > EMA100
    Short: MACD crosses below 0 OR low < EMA100
    """
    macd, macdsig, _ = _macd(df['close'], 12, 26, 9)
    ema100 = _ema(df['close'], 100)

    long_cond = (
        (macd < 0) &
        _crossover(macd, macdsig) &
        (df['low'] > ema100) &
        (df['volume'] > 0)
    )
    short_cond = (
        _crossunder(macd, pd.Series(0.0, index=df.index)) |
        (df['low'] < ema100)
    )
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 21. RSI_Directional_Trend — RSI(4) with EMA100 trend ───────────────────
def gen_RSI_Directional_Trend(df, **p):
    """
    Source: paulcpk/RSIDirectionalWithTrend
    Buy: RSI(4) crosses above 15 + price above EMA100
    Short: RSI(4) crosses above 85 OR price below EMA100
    """
    rsi4 = _rsi(df['close'], 4)
    ema100 = _ema(df['close'], 100)

    long_cond = (
        _crossover(rsi4, pd.Series(15.0, index=df.index)) &
        (df['low'] > ema100) &
        (df['volume'] > 0)
    )
    short_cond = (
        _crossover(rsi4, pd.Series(85.0, index=df.index)) |
        (df['low'] < ema100)
    )
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 22. RSI_Directional_Slow — RSI(10) + EMA200 slow trend ─────────────────
def gen_RSI_Directional_Slow(df, **p):
    """
    Source: paulcpk/RSIDirectionalWithTrendSlow
    Buy: RSI(10) crosses above 25 + price above EMA200
    Short: RSI(10) crosses below 20 OR price below EMA200
    """
    rsi10 = _rsi(df['close'], 10)
    ema200 = _ema(df['close'], 200)

    long_cond = (
        _crossover(rsi10, pd.Series(25.0, index=df.index)) &
        (df['low'] > ema200) &
        (df['volume'] > 0)
    )
    short_cond = (
        _crossunder(rsi10, pd.Series(20.0, index=df.index)) |
        (df['low'] < ema200)
    )
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 23. CryptoFrog_BB_Stoch — BB expansion + StochRSI + MFI (CryptoFrog) ───
def gen_CryptoFrog_BB_Stoch(df, **p):
    """
    Source: froggleston/CryptoFrog — BBW expansion + StochRSI + MFI.
    Buy: BB width expanding + StochRSI d < k (k < 30) + MFI < 20
    Short: close above BB upper + MFI > 80
    """
    bb_lo, bb_mid, bb_hi = _bb_typical(df, 20, 1)
    bb_width = (bb_hi - bb_lo) / (bb_mid + 1e-10)
    # BBW expansion: current width > max of last 4 (excl current)
    bbw_exp = bb_width > bb_width.shift(1).rolling(3).max()

    srsi_k, srsi_d = _stochrsi(df['close'], 14, 14, 3, 3)
    mfi = _mfi(df, 14)
    stoch_fast_k, stoch_fast_d = _stoch(df, 5, 3)

    long_cond = (
        bbw_exp &
        (srsi_d >= srsi_k) &
        (srsi_d < 30) &
        (mfi < 20) &
        (df['volume'] > 0)
    )
    short_cond = (
        (df['close'] > bb_hi) &
        (mfi > 80) &
        (df['volume'] > 0)
    )
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 24. CryptoFrog_SAR_Stoch — SAR + StochRSI undersold ────────────────────
def gen_CryptoFrog_SAR_Stoch(df, **p):
    """
    Source: froggleston/CryptoFrog — secondary buy in undersold regions.
    Buy: close < SAR + StochRSI d >= k (d < 30) + stoch_fastd < 23 + MFI < 30
    Short: close > SAR + StochRSI oversold + MFI > 75
    """
    sar = _sar(df)
    srsi_k, srsi_d = _stochrsi(df['close'], 14, 14, 3, 3)
    fast_k, fast_d = _stoch(df, 5, 3)
    mfi = _mfi(df, 14)

    long_cond = (
        (df['close'] < sar) &
        (srsi_d >= srsi_k) &
        (srsi_d < 30) &
        (fast_d > fast_k) &
        (fast_d < 23) &
        (mfi < 30) &
        (df['volume'] > 0)
    )
    short_cond = (
        (df['close'] > sar) &
        (srsi_k > 70) &
        (mfi > 75) &
        (df['volume'] > 0)
    )
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 25. CryptoFrog_DMI_BB — DMI + BB lower band ────────────────────────────
def gen_CryptoFrog_DMI_BB(df, **p):
    """
    Source: froggleston/CryptoFrog — dmi minus breakout at BB lower.
    Buy: DI- > 30 + DI- crosses above DI+ + close < BB lower
    Short: DI+ > 30 + DI+ crosses above DI- + close > BB upper
    """
    bb_lo, _, bb_hi = _bb_typical(df, 20, 1)
    di_plus, di_minus, adx = _dmi(df, 14)

    long_cond = (
        (di_minus > 30) &
        _crossover(di_minus, di_plus) &
        (df['close'] < bb_lo)
    )
    short_cond = (
        (di_plus > 30) &
        _crossover(di_plus, di_minus) &
        (df['close'] > bb_hi)
    )
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── 26. NFIX_BB_RPB_EWO2 — EWO2 high bullish signal ────────────────────────
def gen_NFIX_BB_RPB_EWO2(df, **p):
    """
    Source: jilv220/NFIX_BB_RPB — 'ewo_high_2' condition.
    Buy: EWO > 4.179 + RSI(fast) < 45 + RSI < 35 + close in EMA range
    Short: EWO < -4 + close > EMA50
    """
    ewo = _ewo(df, 5, 35)
    rsi = _rsi(df['close'], 14)
    rsi_fast = _rsi(df['close'], 4)
    ema50 = _ema(df['close'], 50)
    ema200 = _ema(df['close'], 200)

    long_cond = (
        (ewo > 4.179) &
        (rsi_fast < 45) &
        (rsi < 35) &
        (df['close'] > ema200 * 0.970) &
        (df['close'] < ema50 * 1.087)
    )
    short_cond = (
        (ewo < -4.0) &
        (df['close'] > ema50 * 1.03)
    )
    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig

# ─── Strategy export ─────────────────────────────────────────────────────────
STRATEGY_EXPORT = {
    # BB_RPB_TSL family (jilv220, 212 stars)
    'BB_RPB_TSL': {
        'gen': gen_BB_RPB_TSL,
        'params': {},
        'source': 'jilv220/BB_RPB_TSL',
    },
    'BB_RPB_EWO': {
        'gen': gen_BB_RPB_EWO,
        'params': {},
        'source': 'jilv220/BB_RPB_TSL',
    },
    'BB_RPB_Clucha': {
        'gen': gen_BB_RPB_Clucha,
        'params': {},
        'source': 'jilv220/BB_RPB_TSL',
    },
    'BB_RPB_Deadfish': {
        'gen': gen_BB_RPB_Deadfish,
        'params': {},
        'source': 'jilv220/BB_RPB_TSL',
    },
    'BB_RPB_COFI': {
        'gen': gen_BB_RPB_COFI,
        'params': {},
        'source': 'jilv220/BB_RPB_TSL',
    },
    'BB_RPB_SqzMom': {
        'gen': gen_BB_RPB_SqzMom,
        'params': {},
        'source': 'jilv220/BB_RPB_TSL',
    },
    'NFIX_BB_RPB_EWO2': {
        'gen': gen_NFIX_BB_RPB_EWO2,
        'params': {},
        'source': 'jilv220/NFIX_BB_RPB',
    },
    # mikedigriz family (39 stars)
    'HMA_CrossSignal': {
        'gen': gen_HMA_CrossSignal,
        'params': {},
        'source': 'mikedigriz/BuyOrDie+EasyInEasyOut',
    },
    'CCI_BB_Oversold': {
        'gen': gen_CCI_BB_Oversold,
        'params': {},
        'source': 'mikedigriz/CCI_BB',
    },
    'FisherHull_Reversal': {
        'gen': gen_FisherHull_Reversal,
        'params': {},
        'source': 'mikedigriz/FisherHull',
    },
    'RSI_BB_Bounce': {
        'gen': gen_RSI_BB_Bounce,
        'params': {},
        'source': 'mikedigriz/RSI_BB',
    },
    'SmartMoney_CMF': {
        'gen': gen_SmartMoney_CMF,
        'params': {},
        'source': 'mikedigriz/smart_money_strategy',
    },
    # keithorange family (14 stars)
    'MASlope_EMA_Trend': {
        'gen': gen_MASlope_EMA_Trend,
        'params': {'period': 50, 'slope_period': 14},
        'source': 'keithorange/MASlopeStrategy',
    },
    'MASlope_HMA_Trend': {
        'gen': gen_MASlope_HMA_Trend,
        'params': {'period': 30, 'slope_period': 10},
        'source': 'keithorange/MASlopeStrategy-HMA',
    },
    # MoniGoMani family (Rikj000, 1025 stars)
    'MoniGoMani_BB_EMA': {
        'gen': gen_MoniGoMani_BB_EMA,
        'params': {},
        'source': 'Rikj000/MoniGoMani',
    },
    'MoniGoMani_MACD_RSI': {
        'gen': gen_MoniGoMani_MACD_RSI,
        'params': {},
        'source': 'Rikj000/MoniGoMani',
    },
    'MoniGoMani_SMA_Cross': {
        'gen': gen_MoniGoMani_SMA_Cross,
        'params': {},
        'source': 'Rikj000/MoniGoMani',
    },
    'MoniGoMani_ADX_VWAP': {
        'gen': gen_MoniGoMani_ADX_VWAP,
        'params': {},
        'source': 'Rikj000/MoniGoMani',
    },
    # paulcpk family (321 stars)
    'DoubleEMA_Crossover_Trend': {
        'gen': gen_DoubleEMA_Crossover_Trend,
        'params': {},
        'source': 'paulcpk/DoubleEMACrossoverWithTrend',
    },
    'EMA_Price_Crossover_Threshold': {
        'gen': gen_EMA_Price_Crossover_Threshold,
        'params': {},
        'source': 'paulcpk/EMAPriceCrossoverWithThreshold',
    },
    'MACD_Crossover_Trend': {
        'gen': gen_MACD_Crossover_Trend,
        'params': {},
        'source': 'paulcpk/MACDCrossoverWithTrend',
    },
    'RSI_Directional_Trend': {
        'gen': gen_RSI_Directional_Trend,
        'params': {},
        'source': 'paulcpk/RSIDirectionalWithTrend',
    },
    'RSI_Directional_Slow': {
        'gen': gen_RSI_Directional_Slow,
        'params': {},
        'source': 'paulcpk/RSIDirectionalWithTrendSlow',
    },
    # CryptoFrog family (froggleston, 183 stars)
    'CryptoFrog_BB_Stoch': {
        'gen': gen_CryptoFrog_BB_Stoch,
        'params': {},
        'source': 'froggleston/CryptoFrog',
    },
    'CryptoFrog_SAR_Stoch': {
        'gen': gen_CryptoFrog_SAR_Stoch,
        'params': {},
        'source': 'froggleston/CryptoFrog',
    },
    'CryptoFrog_DMI_BB': {
        'gen': gen_CryptoFrog_DMI_BB,
        'params': {},
        'source': 'froggleston/CryptoFrog',
    },
}
