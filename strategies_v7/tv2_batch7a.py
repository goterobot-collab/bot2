#!/usr/bin/env python3
"""
TV2 Batch 7a — 10 estrategias Pine → Python
Likes totales: ~36,756

1. Buy_Sell_AO_Stoch_RSI_ATR  (12590 likes) — Pine v4 — AO+Stoch+RSI oversold/overbought
2. BB_Breakout                 (3821 likes)  — Pine v4 — Bollinger Bands breakout
3. Full_AllinOne_Risk          (3747 likes)  — Pine v4 — SAR+EMA+BB+MACD+RSI
4. Fibonacci_RSI               (3285 likes)  — Pine v4 — VWMA Fibonacci bands + RSI
5. Scalping_WilliamsR_MACD_SMA (3250 likes)  — Pine v5 — Williams %R + MACD + SMA
6. Keltner_Trend               (2390 likes)  — Pine v5 — Keltner + DMI/ADX trend
7. ATR_Trailing_Stop           (2283 likes)  — Pine v4 — Dual ATR trailing stop crossover
8. ATR_PSAR_QuantNomad         (1788 likes)  — Pine v4 — ATR-based Parabolic SAR
9. Turtle_Donchian_ATR         (1709 likes)  — Pine v4 — Donchian channel breakout + ATR SL
10. Two_TP_EMA_WMA              (1503 likes)  — Pine v4 — EMA/WMA crossover dual TP

Convención: sig = 1 (LONG), -1 (SHORT), 0 (sin señal)
Sin look-ahead: señales basadas en datos hasta la barra actual (shift(1) para prev).
"""

import pandas as pd
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

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


# ─── 1. BUY/SELL AO+STOCH+RSI+ATR ────────────────────────────────────────────

def gen_Buy_Sell_AO_Stoch_RSI_ATR(df, ao_fast=5, ao_slow=34,
                                   stoch_k=14, stoch_d=3, stoch_smooth=3,
                                   rsi_len=10, rsi_ob=70, rsi_os=30,
                                   stoch_ob=80, stoch_os=20):
    """
    Buy&Sell Strategy depends on AO+Stoch+RSI+ATR — Pine v4 (12590 likes)
    LONG:  Stoch K < stoch_os AND RSI < rsi_os AND AO > AO[1]  (rising momentum from oversold)
    SHORT: Stoch K > stoch_ob AND RSI > rsi_ob AND AO < AO[1]  (falling momentum from overbought)
    """
    close  = df['close']
    high   = df['high']
    low    = df['low']

    # Awesome Oscillator: SMA(hl2, fast) - SMA(hl2, slow)
    hl2 = (high + low) / 2
    ao  = _sma(hl2, ao_fast) - _sma(hl2, ao_slow)

    # Stochastic
    k_raw, _ = _stoch(close, high, low, stoch_k, stoch_d)
    k        = _sma(k_raw, stoch_d)     # Pine: sma(stoch(c,h,l,K), D)
    k_smooth = _sma(k, stoch_smooth)    # Pine uses k for signal; use smoothed k for entry

    # RSI
    rsi = _rsi(close, rsi_len)

    long_cond  = (k_smooth < stoch_os) & (rsi < rsi_os) & (ao > ao.shift(1))
    short_cond = (k_smooth > stoch_ob) & (rsi > rsi_ob) & (ao < ao.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig


def space_Buy_Sell_AO_Stoch_RSI_ATR(trial):
    return {
        'ao_fast':      trial.suggest_int('ao_fast',      3, 10),
        'ao_slow':      trial.suggest_int('ao_slow',      20, 50),
        'stoch_k':      trial.suggest_int('stoch_k',      5, 21),
        'stoch_d':      trial.suggest_int('stoch_d',      2, 5),
        'stoch_smooth': trial.suggest_int('stoch_smooth', 2, 5),
        'rsi_len':      trial.suggest_int('rsi_len',      5, 21),
        'rsi_ob':       trial.suggest_int('rsi_ob',       65, 80),
        'rsi_os':       trial.suggest_int('rsi_os',       20, 40),
        'stoch_ob':     trial.suggest_int('stoch_ob',     70, 90),
        'stoch_os':     trial.suggest_int('stoch_os',     10, 30),
    }


# ─── 2. BOLLINGER BANDS BREAKOUT ──────────────────────────────────────────────

def gen_BB_Breakout(df, bb_len=55, bb_mult=1.0):
    """
    Bollinger Bands Breakout Strategy — Pine v4 (3821 likes) by TradeChartist.
    Logic:
      - 'long'  condition: close > upper band  (price closes above upper BB)
      - 'short' condition: close < lower band  (price closes below lower BB)
      - L1 = barssince(long),  S1 = barssince(short)
      - longSignal  fires when L1 < S1 AND previous bar was NOT (L1 < S1)
        i.e., first bar where last-long is more recent than last-short
      - shortSignal fires when S1 < L1 AND previous bar was NOT (S1 < L1)
    This is equivalent to: the first bar AFTER a new long/short breakout.
    We simplify: track the "regime" (who was last) and fire on regime change.
    """
    close  = df['close']

    basis, upper, lower = _bb(close, bb_len, bb_mult)

    long_flag  = (close > upper).astype(int)
    short_flag = (close < lower).astype(int)

    # Track index of last long/short event
    n = len(df)
    last_long  = np.full(n, -1, dtype=int)
    last_short = np.full(n, -1, dtype=int)

    lf = long_flag.values
    sf = short_flag.values

    for i in range(n):
        last_long[i]  = i if lf[i] else (last_long[i-1]  if i > 0 else -1)
        last_short[i] = i if sf[i] else (last_short[i-1] if i > 0 else -1)

    # regime: True when long is more recent than short (L1 < S1 in Pine)
    regime = (last_long > last_short)

    sig = pd.Series(0, index=df.index)
    # longSignal: regime just turned True (first bar where long > short)
    long_signal  = regime & ~pd.Series(regime, index=df.index).shift(1).fillna(False)
    short_signal = ~regime & pd.Series(regime, index=df.index).shift(1).fillna(True)

    sig[long_signal]  = 1
    sig[short_signal] = -1
    return sig


def space_BB_Breakout(trial):
    return {
        'bb_len':  trial.suggest_int('bb_len',   20, 100),
        'bb_mult': trial.suggest_float('bb_mult', 0.5, 3.0, step=0.1),
    }


# ─── 3. FULL ALL-IN-ONE WITH RISK MANAGEMENT ──────────────────────────────────

def _psar_loop(close, high, low, start=0.02, increment=0.02, maximum=0.2):
    """
    Parabolic SAR implemented as a stateful loop (Pine v4 manual PSAR).
    Returns (psar, uptrend) arrays.
    """
    n = len(close)
    c = close.values
    h = high.values
    l = low.values

    psar      = np.zeros(n)
    uptrend   = np.zeros(n, dtype=bool)
    ep        = np.zeros(n)
    af_arr    = np.zeros(n)
    nextbar   = np.zeros(n)

    if n < 2:
        return psar, uptrend

    # Init bar 1 direction
    if c[1] > c[0]:
        uptrend[1] = True
        ep[1]      = h[1]
        psar[1]    = l[0] + start * (h[1] - l[0])
    else:
        uptrend[1] = False
        ep[1]      = l[1]
        psar[1]    = h[0] + start * (l[1] - h[0])

    af_arr[1]  = start
    nextbar[1] = psar[1] + af_arr[1] * (ep[1] - psar[1])

    for i in range(2, n):
        sar_i = nextbar[i - 1]
        up    = uptrend[i - 1]
        ep_i  = ep[i - 1]
        af_i  = af_arr[i - 1]

        first_trend_bar = False

        if up:
            if sar_i > l[i]:
                first_trend_bar = True
                up    = False
                sar_i = max(ep_i, h[i])
                ep_i  = l[i]
                af_i  = start
        else:
            if sar_i < h[i]:
                first_trend_bar = True
                up    = True
                sar_i = min(ep_i, l[i])
                ep_i  = h[i]
                af_i  = start

        if not first_trend_bar:
            if up:
                if h[i] > ep_i:
                    ep_i = h[i]
                    af_i = min(af_i + increment, maximum)
            else:
                if l[i] < ep_i:
                    ep_i = l[i]
                    af_i = min(af_i + increment, maximum)

        if up:
            sar_i = min(sar_i, l[i - 1])
            if i > 1:
                sar_i = min(sar_i, l[i - 2])
        else:
            sar_i = max(sar_i, h[i - 1])
            if i > 1:
                sar_i = max(sar_i, h[i - 2])

        psar[i]    = sar_i
        uptrend[i] = up
        ep[i]      = ep_i
        af_arr[i]  = af_i
        nextbar[i] = sar_i + af_i * (ep_i - sar_i)

    return pd.Series(psar, index=close.index), pd.Series(uptrend, index=close.index)


def gen_Full_AllinOne_Risk(df, rsi_len=5, rsi_os=23, rsi_ob=72,
                            macd_fast=12, macd_slow=26, macd_sig=9,
                            bb_len=17, bb_mult=2.0, ema_len=10,
                            sar_start=0.02, sar_inc=0.02, sar_max=0.2):
    """
    Full Strategy AllinOne with risk management — Pine v4 (3747 likes) by SoftKill21.
    Indicators: PSAR (uptrend bool), EMA crossover BB-basis, MACD histogram, RSI.
    LONG:  uptrend=True  AND EMA crossover above BB-basis AND MACD hist > 0 AND RSI > rsi_ob
    SHORT: uptrend=False AND EMA crossunder below BB-basis AND MACD hist < 0 AND RSI < rsi_os
    """
    close = df['close']
    high  = df['high']
    low   = df['low']

    # PSAR
    psar, uptrend = _psar_loop(close, high, low, sar_start, sar_inc, sar_max)

    # RSI
    rsi = _rsi(close, rsi_len)

    # MACD
    fast_ma  = _sma(close, macd_fast)
    slow_ma  = _sma(close, macd_slow)
    macd     = fast_ma - slow_ma
    signal   = _sma(macd, macd_sig)
    hist     = macd - signal

    # BB basis
    bb_basis = _sma(close, bb_len)

    # EMA
    ema_line = _ema(close, ema_len)

    # Crossover / crossunder of EMA vs BB basis
    ema_co = (ema_line.shift(1) < bb_basis.shift(1)) & (ema_line >= bb_basis)
    ema_cu = (ema_line.shift(1) > bb_basis.shift(1)) & (ema_line <= bb_basis)

    long_cond  = uptrend & ema_co & (hist > 0) & (rsi > rsi_ob)
    short_cond = (~uptrend) & ema_cu & (hist < 0) & (rsi < rsi_os)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig


def space_Full_AllinOne_Risk(trial):
    return {
        'rsi_len':   trial.suggest_int('rsi_len',   3, 14),
        'rsi_os':    trial.suggest_int('rsi_os',    15, 35),
        'rsi_ob':    trial.suggest_int('rsi_ob',    60, 85),
        'macd_fast': trial.suggest_int('macd_fast', 5, 20),
        'macd_slow': trial.suggest_int('macd_slow', 20, 40),
        'macd_sig':  trial.suggest_int('macd_sig',  5, 15),
        'bb_len':    trial.suggest_int('bb_len',    10, 30),
        'bb_mult':   trial.suggest_float('bb_mult', 1.0, 3.0, step=0.1),
        'ema_len':   trial.suggest_int('ema_len',   5, 20),
    }


# ─── 4. FIBONACCI + RSI STRATEGY ─────────────────────────────────────────────

def gen_Fibonacci_RSI(df, rsi_len=14, rsi_os=30, rsi_ob=70,
                      fib_len=200, fib_mult=3.0, fib_level=764):
    """
    Fibonacci + RSI Strategy — Pine v4 (3285 likes) by MohamedYAbdelaziz.
    Uses VWMA-based Fibonacci bands.
    Upper band: vwma(hlc3, fib_len) + mult * std * 0.001 * level   (fu764)
    Lower band: vwma(hlc3, fib_len) - mult * std * 0.001 * level   (fd764)
    Outer bands at ±1 std dev from vwma.

    LONG:  low < fd1 (outer lower) AND RSI crossover above rsi_os from prev bar
    SHORT: high > fu1 (outer upper) AND RSI crossunder below rsi_ob from prev bar

    Note: Pine uses vrsi[1] (previous bar RSI) for the cross check.
    """
    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df['volume']

    hlc3   = (high + low + close) / 3

    # VWMA: volume-weighted moving average
    def _vwma(src, p):
        return (src * volume).rolling(p).sum() / volume.rolling(p).sum()

    basis  = _vwma(hlc3, fib_len)
    dev    = fib_mult * hlc3.rolling(fib_len).std()

    fu764  = basis + 0.001 * fib_level * dev
    fd764  = basis - 0.001 * fib_level * dev
    fu1    = basis + dev
    fd1    = basis - dev

    # RSI — Pine uses vrsi[1] for cross (prev bar RSI vs threshold)
    vrsi   = _rsi(close, rsi_len)
    vrsi_p = vrsi.shift(1)   # vrsi[1] in Pine = previous bar

    # crossover(vrsi[1], overSold) → vrsi_p crosses above rsi_os
    rsi_co = (vrsi_p.shift(1) < rsi_os) & (vrsi_p >= rsi_os)
    # crossunder(vrsi[1], overBought) → vrsi_p crosses below rsi_ob
    rsi_cu = (vrsi_p.shift(1) > rsi_ob) & (vrsi_p <= rsi_ob)

    long_cond  = (low < fd1) & rsi_co
    short_cond = (high > fu1) & rsi_cu

    # Additional Pine filter: long entry requires close < fu764, short requires close > fd764
    long_cond  = long_cond  & (high < fu764)
    short_cond = short_cond & (low  > fd764)

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  = 1
    sig[short_cond] = -1
    return sig


def space_Fibonacci_RSI(trial):
    return {
        'rsi_len':   trial.suggest_int('rsi_len',   7, 21),
        'rsi_os':    trial.suggest_int('rsi_os',    20, 40),
        'rsi_ob':    trial.suggest_int('rsi_ob',    60, 80),
        'fib_len':   trial.suggest_int('fib_len',   100, 300),
        'fib_mult':  trial.suggest_float('fib_mult', 1.5, 5.0, step=0.5),
        'fib_level': trial.suggest_int('fib_level', 500, 900),
    }


# ─── 5. SCALPING WILLIAMS %R + MACD + SMA ─────────────────────────────────────

def gen_Scalping_WilliamsR_MACD_SMA(df, wr_len=140, wr_buy=-94, wr_sell=-6,
                                     wr_deact_buy=-40, wr_deact_sell=-60,
                                     macd_fast=24, macd_slow=52, macd_sig=9,
                                     sma_len=7):
    """
    Scalping Strategy with Williams %R, MACD, and SMA — Pine v5 (3250 likes).
    Williams %R: (highest_high - close) / (highest_high - lowest_low) * -100
    State machine:
      - buyActive  activated when WR crosses above wr_buy  AND close > SMA
      - buyActive  deactivated when WR crosses above wr_deact_buy
      - BUY signal:  buyActive AND MACD hist changes from negative to positive
      - Buy EXIT:   MACD hist declines (macd[i] < macd[i-1])
      - sellActive activated when WR crosses below wr_sell AND close < SMA
      - sellActive deactivated when WR crosses below wr_deact_sell
      - SELL signal: sellActive AND MACD hist changes from positive to negative
      - Sell EXIT:  MACD hist rises (macd[i] > macd[i-1])
    We return entry signals only (exits handled by bot's SL/TP framework).
    """
    close  = df['close']
    high   = df['high']
    low    = df['low']

    # Williams %R
    hh  = high.rolling(wr_len).max()
    ll  = low.rolling(wr_len).min()
    wr  = (hh - close) / (hh - ll + 1e-10) * -100

    # MACD
    macd_line   = _ema(close, macd_fast) - _ema(close, macd_slow)
    macd_signal = _ema(macd_line, macd_sig)
    macd_hist   = macd_line - macd_signal

    # SMA trend filter
    sma_line = _sma(close, sma_len)

    n = len(df)
    wr_v    = wr.values
    hist_v  = macd_hist.values
    cl_v    = close.values
    sma_v   = sma_line.values
    sig_arr = np.zeros(n)

    buy_active  = False
    sell_active = False

    for i in range(1, n):
        if np.isnan(wr_v[i]) or np.isnan(hist_v[i]) or np.isnan(sma_v[i]):
            continue

        # WR crossover wr_buy (WR goes from below wr_buy to above wr_buy)
        if wr_v[i - 1] < wr_buy <= wr_v[i] and cl_v[i] > sma_v[i]:
            buy_active = True
        # WR crossover wr_deact_buy — deactivate
        if wr_v[i - 1] < wr_deact_buy <= wr_v[i]:
            buy_active = False

        # Buy entry: buyActive AND MACD hist turned positive (was negative last bar)
        if buy_active and hist_v[i - 1] < 0 and hist_v[i] > 0:
            sig_arr[i] = 1

        # Buy exit: MACD hist declining → implicit via framework, but reset state
        if buy_active and hist_v[i] < hist_v[i - 1] and hist_v[i] > 0:
            buy_active = False

        # WR crossunder wr_sell (WR goes from above wr_sell to below wr_sell)
        if wr_v[i - 1] > wr_sell >= wr_v[i] and cl_v[i] < sma_v[i]:
            sell_active = True
        # WR crossunder wr_deact_sell — deactivate
        if wr_v[i - 1] > wr_deact_sell >= wr_v[i]:
            sell_active = False

        # Sell entry: sellActive AND MACD hist turned negative
        if sell_active and hist_v[i - 1] > 0 and hist_v[i] < 0:
            sig_arr[i] = -1

        # Sell exit: MACD hist rising → reset state
        if sell_active and hist_v[i] > hist_v[i - 1] and hist_v[i] < 0:
            sell_active = False

    return pd.Series(sig_arr, index=df.index)


def space_Scalping_WilliamsR_MACD_SMA(trial):
    return {
        'wr_len':          trial.suggest_int('wr_len',          50, 200),
        'wr_buy':          trial.suggest_int('wr_buy',          -99, -80),
        'wr_sell':         trial.suggest_int('wr_sell',         -20, -1),
        'wr_deact_buy':    trial.suggest_int('wr_deact_buy',    -60, -20),
        'wr_deact_sell':   trial.suggest_int('wr_deact_sell',   -80, -40),
        'macd_fast':       trial.suggest_int('macd_fast',       8, 30),
        'macd_slow':       trial.suggest_int('macd_slow',       30, 80),
        'macd_sig':        trial.suggest_int('macd_sig',        5, 15),
        'sma_len':         trial.suggest_int('sma_len',         3, 20),
    }


# ─── 6. KELTNER CHANNEL + TREND (DMI/ADX) ────────────────────────────────────

def _adx_dmi(high, low, close, di_len=14, adx_len=14):
    """
    Compute +DI, -DI, ADX using RMA (Wilder's smoothing = EMA with alpha=1/period).
    """
    up   = high.diff()
    down = -low.diff()

    # True Range
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low  - close.shift()).abs()], axis=1).max(axis=1)

    def _rma(s, p):
        return s.ewm(alpha=1.0 / p, min_periods=p, adjust=False).mean()

    tr_rma   = _rma(tr, di_len)
    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    plus_di  = 100 * _rma(plus_dm,  di_len) / (tr_rma + 1e-10)
    minus_di = 100 * _rma(minus_dm, di_len) / (tr_rma + 1e-10)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = _rma(dx, adx_len)

    return adx, plus_di, minus_di


def gen_Keltner_Trend(df, ribbon_period=46, kc_len=81, kc_mult=2.5,
                      di_len=19, adx_len=10, dmi_benchmark=27):
    """
    Keltner Channel + Trend Strategy — Pine v5 (2390 likes) by Wunderbit Trading.
    Indicators:
      - Ribbon: EMA vs SMA (ribbon_period) → UT = SMA < EMA (uptrend)
      - Keltner Channel: SMA ± rangema * mult  (rangema = SMA of TR)
      - DMI: +DI, -DI, ADX
    LONG:  UT AND open > KC_lower AND open < KC_upper
           AND close > KC_upper (breakout)
           AND +DI > -DI AND +DI > dmi_benchmark
    No SHORT entries (original strategy is long-only).
    """
    close  = df['close']
    high   = df['high']
    low    = df['low']
    opn    = df['open']

    # Ribbon trend
    ema_r = _ema(close, ribbon_period)
    sma_r = _sma(close, ribbon_period)
    ut    = sma_r < ema_r   # uptrend when SMA < EMA

    # Keltner Channel (SMA-based, range = SMA of TR)
    ma  = _sma(close, kc_len)
    tr  = pd.concat([high - low,
                     (high - close.shift()).abs(),
                     (low  - close.shift()).abs()], axis=1).max(axis=1)
    rangema = _sma(tr, kc_len)
    upper   = ma + rangema * kc_mult
    lower   = ma - rangema * kc_mult

    # DMI / ADX
    adx_val, plus_di, minus_di = _adx_dmi(high, low, close, di_len, adx_len)

    # Entry: open inside channel, close breaks above upper
    entry_long = (opn > lower) & (opn < upper) & (close > upper) \
                 & (plus_di > minus_di) & (plus_di > dmi_benchmark)

    long_cond = ut & entry_long

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    return sig


def space_Keltner_Trend(trial):
    return {
        'ribbon_period': trial.suggest_int('ribbon_period',  20, 80),
        'kc_len':        trial.suggest_int('kc_len',         30, 120),
        'kc_mult':       trial.suggest_float('kc_mult',      1.0, 4.0, step=0.25),
        'di_len':        trial.suggest_int('di_len',         10, 30),
        'adx_len':       trial.suggest_int('adx_len',        5, 20),
        'dmi_benchmark': trial.suggest_int('dmi_benchmark',  15, 40),
    }


# ─── 7. ATR TRAILING STOP (DUAL) ──────────────────────────────────────────────

def gen_ATR_Trailing_Stop(df, ap1=5, af1=0.5, ap2=10, af2=3.0):
    """
    ATR Trailing Stop Strategy by ceyhun — Pine v4 (2283 likes).
    Fast trail: Trail1 tracks price ± AF1*ATR(AP1)
    Slow trail: Trail2 tracks price ± AF2*ATR(AP2)
    BUY signal:  Trail1 crosses above Trail2  (crossover)
    SELL signal: Trail1 crosses below Trail2  (crossunder)
    """
    close = df['close']
    high  = df['high']
    low   = df['low']

    sl1 = af1 * _atr(high, low, close, ap1)
    sl2 = af2 * _atr(high, low, close, ap2)

    n   = len(df)
    c   = close.values
    s1  = sl1.values
    s2  = sl2.values
    t1  = np.zeros(n)
    t2  = np.zeros(n)

    for i in range(1, n):
        if np.isnan(s1[i]) or np.isnan(s2[i]):
            t1[i] = t1[i-1]
            t2[i] = t2[i-1]
            continue

        # Trail1
        if c[i] > t1[i-1] and c[i-1] > t1[i-1]:
            t1[i] = max(t1[i-1], c[i] - s1[i])
        elif c[i] < t1[i-1] and c[i-1] < t1[i-1]:
            t1[i] = min(t1[i-1], c[i] + s1[i])
        elif c[i] > t1[i-1]:
            t1[i] = c[i] - s1[i]
        else:
            t1[i] = c[i] + s1[i]

        # Trail2
        if c[i] > t2[i-1] and c[i-1] > t2[i-1]:
            t2[i] = max(t2[i-1], c[i] - s2[i])
        elif c[i] < t2[i-1] and c[i-1] < t2[i-1]:
            t2[i] = min(t2[i-1], c[i] + s2[i])
        elif c[i] > t2[i-1]:
            t2[i] = c[i] - s2[i]
        else:
            t2[i] = c[i] + s2[i]

    t1_s = pd.Series(t1, index=df.index)
    t2_s = pd.Series(t2, index=df.index)

    # crossover(Trail1, Trail2): Trail1 was below Trail2, now above
    buy  = (t1_s.shift(1) < t2_s.shift(1)) & (t1_s >= t2_s)
    sell = (t1_s.shift(1) > t2_s.shift(1)) & (t1_s <= t2_s)

    sig = pd.Series(0, index=df.index)
    sig[buy]  = 1
    sig[sell] = -1
    return sig


def space_ATR_Trailing_Stop(trial):
    return {
        'ap1': trial.suggest_int('ap1',     3, 15),
        'af1': trial.suggest_float('af1',   0.2, 2.0, step=0.1),
        'ap2': trial.suggest_int('ap2',     5, 25),
        'af2': trial.suggest_float('af2',   1.0, 6.0, step=0.5),
    }


# ─── 8. ATR PARABOLIC SAR (QUANTNOMAD) ────────────────────────────────────────

def gen_ATR_PSAR_QuantNomad(df, atr_length=14, start=0.02, increment=0.02,
                             maximum=0.2, entry_bars=1):
    """
    ATR Parabolic SAR Strategy [QuantNomad] — Pine v4 (1788 likes).
    Custom ATR-based PSAR where step = AF * ATR (not AF * price distance).
    trend_bars: positive = bullish bars count, negative = bearish bars count.
    LONG:  trend_bars == +entry_bars  (first N bars in new uptrend)
    SHORT: trend_bars == -entry_bars  (first N bars in new downtrend)
    """
    close  = df['close']
    high   = df['high']
    low    = df['low']

    atr_s  = _atr(high, low, close, atr_length).fillna(method='bfill')

    n = len(df)
    c = close.values
    h = high.values
    l = low.values
    a = atr_s.values

    psar        = np.zeros(n)
    trend_dir   = np.zeros(n, dtype=int)
    ep_arr      = np.zeros(n)
    af_arr      = np.zeros(n)
    trend_bars  = np.zeros(n, dtype=int)

    # Init from bar 0 based on bar 1 close vs open
    if n < 2:
        return pd.Series(0, index=df.index)

    opn0 = df['open'].values
    if c[1] > opn0[1]:
        trend_dir[0] = 1
        ep_arr[0]    = h[1]
        psar[0]      = l[1]
    else:
        trend_dir[0] = -1
        ep_arr[0]    = l[1]
        psar[0]      = h[1]
    af_arr[0]   = start
    trend_bars[0] = trend_dir[0]

    for i in range(1, n):
        td_prev  = trend_dir[i - 1]
        ep_prev  = ep_arr[i - 1]
        af_prev  = af_arr[i - 1]
        psar_prev = psar[i - 1]
        tb_prev  = trend_bars[i - 1]

        sar_long_to_short = (td_prev == 1)  and (c[i] <= psar_prev)
        sar_short_to_long = (td_prev == -1) and (c[i] >= psar_prev)
        trend_change      = sar_long_to_short or sar_short_to_long

        # trend_dir
        if sar_long_to_short:
            td = -1
        elif sar_short_to_long:
            td = 1
        else:
            td = td_prev
        trend_dir[i] = td

        # trend_bars
        if sar_long_to_short:
            tb = -1
        elif sar_short_to_long:
            tb = 1
        elif td == 1:
            tb = tb_prev + 1
        elif td == -1:
            tb = tb_prev - 1
        else:
            tb = tb_prev
        trend_bars[i] = tb

        # AF
        if trend_change:
            af_i = start
        elif (td == 1 and h[i] > ep_prev) or (td == -1 and l[i] < ep_prev):
            af_i = min(maximum, af_prev + increment)
        else:
            af_i = af_prev
        af_arr[i] = af_i

        # EP
        if trend_change and td == 1:
            ep_i = h[i]
        elif trend_change and td == -1:
            ep_i = l[i]
        elif td == 1:
            ep_i = max(ep_prev, h[i])
        else:
            ep_i = min(ep_prev, l[i])
        ep_arr[i] = ep_i

        # PSAR
        atr_i = a[i] if not np.isnan(a[i]) else a[max(0, i-1)]
        if trend_change:
            psar[i] = ep_prev
        elif td == 1:
            psar[i] = psar_prev + af_i * atr_i
        else:
            psar[i] = psar_prev - af_i * atr_i

    tb_s = pd.Series(trend_bars, index=df.index)

    sig = pd.Series(0, index=df.index)
    sig[tb_s ==  entry_bars] = 1
    sig[tb_s == -entry_bars] = -1
    return sig


def space_ATR_PSAR_QuantNomad(trial):
    return {
        'atr_length': trial.suggest_int('atr_length',   7, 21),
        'start':      trial.suggest_float('start',       0.01, 0.05, step=0.005),
        'increment':  trial.suggest_float('increment',   0.01, 0.05, step=0.005),
        'maximum':    trial.suggest_float('maximum',     0.1,  0.5,  step=0.05),
        'entry_bars': trial.suggest_int('entry_bars',    1, 5),
    }


# ─── 9. TURTLE TRADING — DONCHIAN + ATR ──────────────────────────────────────

def gen_Turtle_Donchian_ATR(df, enter_period=20, exit_period=10,
                             atrmult=2.0, atrperiod=20):
    """
    Turtle Trading Strategy (Donchian/ATR) — Pine v4 (1709 likes).
    LONG:  close >= highest(close, enter_period)[1]  (Donchian breakout above)
    LONG EXIT: close <= lowest(close, exit_period)[1] OR ATR stop hit
    SHORT: close <= lowest(close, enter_period)[1]
    SHORT EXIT: close >= highest(close, exit_period)[1] OR ATR stop hit

    Since the bot manages exits via SL/TP, we only return entry signals here.
    ATR stop (atrlower/atrupper) computed as EMA(close ± ATR*mult, 3).
    """
    close  = df['close']
    high   = df['high']
    low    = df['low']

    # ATR (EMA-based, as in Pine: ema(tr(true), period))
    tr      = pd.concat([high - low,
                         (high - close.shift()).abs(),
                         (low  - close.shift()).abs()], axis=1).max(axis=1)
    atr_e   = _ema(tr, atrperiod)
    atrstop = atrmult * atr_e

    # Donchian channels for entry and exit
    upper_enter = close.rolling(enter_period).max()
    lower_enter = close.rolling(enter_period).min()
    upper_exit  = close.rolling(exit_period).max()
    lower_exit  = close.rolling(exit_period).min()

    # ATR trailing levels (EMA of close ± atrstop, period=3)
    atrlower = _ema(close - atrstop, 3)
    atrupper = _ema(close + atrstop, 3)

    # Breakout entries with position tracking (Pine strategy.entry only fires when flat)
    ue = upper_enter.shift(1).values
    le = lower_enter.shift(1).values
    cl = close.values
    n  = len(df)
    sig_arr = np.zeros(n)
    pos = 0  # 0=flat, 1=long, -1=short

    for i in range(1, n):
        if np.isnan(ue[i]) or np.isnan(le[i]):
            continue
        break_up_i   = cl[i] >= ue[i]
        break_down_i = cl[i] <= le[i]
        exit_long_i  = cl[i] <= lower_exit.shift(1).values[i]
        exit_short_i = cl[i] >= upper_exit.shift(1).values[i]

        # Exit first
        if pos == 1 and (exit_long_i or (atrlower.values[i-1] and cl[i] <= atrlower.values[i-1])):
            pos = 0
        elif pos == -1 and (exit_short_i or (atrupper.values[i-1] and cl[i] >= atrupper.values[i-1])):
            pos = 0

        # New entry (only when flat)
        if pos == 0:
            if break_up_i:
                sig_arr[i] = 1
                pos = 1
            elif break_down_i:
                sig_arr[i] = -1
                pos = -1

    return pd.Series(sig_arr, index=df.index)


def space_Turtle_Donchian_ATR(trial):
    return {
        'enter_period': trial.suggest_int('enter_period', 10, 50),
        'exit_period':  trial.suggest_int('exit_period',  5, 30),
        'atrmult':      trial.suggest_float('atrmult',    1.0, 4.0, step=0.5),
        'atrperiod':    trial.suggest_int('atrperiod',    10, 30),
    }


# ─── 10. TWO TAKE PROFIT — EMA/WMA CROSSOVER ─────────────────────────────────

def gen_Two_TP_EMA_WMA(df, ema_period=10, wma_period=20):
    """
    Two Take Profit Strategy (FS ATR & PS) — Pine v4 (1503 likes).
    Entry: EMA(close, ema_period) crossover/crossunder WMA(close, wma_period)
    Long:  EMA crosses above WMA (bullish crossover)
    Short: EMA crosses below WMA (bearish crossunder)
    No position stacking: only enter when flat (mimics nz(strategy.position_size)==0).
    """
    close    = df['close']

    ema_line = _ema(close, ema_period)
    wma_line = _wma(close, wma_period)

    # Crossover: EMA above WMA (was below)
    long_co  = (ema_line.shift(1) < wma_line.shift(1)) & (ema_line >= wma_line)
    # Crossunder: EMA below WMA (was above)
    short_cu = (ema_line.shift(1) > wma_line.shift(1)) & (ema_line <= wma_line)

    sig = pd.Series(0, index=df.index)

    # Simulate no-stacking: only enter when not in position
    in_pos  = 0
    sig_arr = np.zeros(len(df))
    lc = long_co.values
    sc = short_cu.values

    for i in range(len(df)):
        if in_pos == 0:
            if lc[i]:
                sig_arr[i] = 1
                in_pos = 1
            elif sc[i]:
                sig_arr[i] = -1
                in_pos = -1
        else:
            # Exit on opposite signal
            if in_pos == 1 and sc[i]:
                sig_arr[i] = -1
                in_pos = -1
            elif in_pos == -1 and lc[i]:
                sig_arr[i] = 1
                in_pos = 1

    return pd.Series(sig_arr, index=df.index)


def space_Two_TP_EMA_WMA(trial):
    return {
        'ema_period': trial.suggest_int('ema_period', 5, 30),
        'wma_period': trial.suggest_int('wma_period', 10, 60),
    }


# ─── STRATEGY EXPORT ──────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'Buy_Sell_AO_Stoch_RSI_ATR': {
        'gen':   gen_Buy_Sell_AO_Stoch_RSI_ATR,
        'space': space_Buy_Sell_AO_Stoch_RSI_ATR,
        'default_params': {
            'ao_fast': 5, 'ao_slow': 34,
            'stoch_k': 14, 'stoch_d': 3, 'stoch_smooth': 3,
            'rsi_len': 10, 'rsi_ob': 70, 'rsi_os': 30,
            'stoch_ob': 80, 'stoch_os': 20,
        },
        'info': {'likes': 12590, 'version': 'v4',
                 'source': 'Buy_Sell_Strategy_depends_on_AO_Sto'},
    },
    'BB_Breakout': {
        'gen':   gen_BB_Breakout,
        'space': space_BB_Breakout,
        'default_params': {'bb_len': 55, 'bb_mult': 1.0},
        'info': {'likes': 3821, 'version': 'v4',
                 'source': 'Bollinger_Bands_Breakout_Strategy'},
    },
    'Full_AllinOne_Risk': {
        'gen':   gen_Full_AllinOne_Risk,
        'space': space_Full_AllinOne_Risk,
        'default_params': {
            'rsi_len': 5, 'rsi_os': 23, 'rsi_ob': 72,
            'macd_fast': 12, 'macd_slow': 26, 'macd_sig': 9,
            'bb_len': 17, 'bb_mult': 2.0, 'ema_len': 10,
            'sar_start': 0.02, 'sar_inc': 0.02, 'sar_max': 0.2,
        },
        'info': {'likes': 3747, 'version': 'v4',
                 'source': 'Full_strategy_AllinOne_with_risk_ma'},
    },
    'Fibonacci_RSI': {
        'gen':   gen_Fibonacci_RSI,
        'space': space_Fibonacci_RSI,
        'default_params': {
            'rsi_len': 14, 'rsi_os': 30, 'rsi_ob': 70,
            'fib_len': 200, 'fib_mult': 3.0, 'fib_level': 764,
        },
        'info': {'likes': 3285, 'version': 'v4',
                 'source': 'Fibonacci___RSI___Strategy'},
    },
    'Scalping_WilliamsR_MACD_SMA': {
        'gen':   gen_Scalping_WilliamsR_MACD_SMA,
        'space': space_Scalping_WilliamsR_MACD_SMA,
        'default_params': {
            'wr_len': 140, 'wr_buy': -94, 'wr_sell': -6,
            'wr_deact_buy': -40, 'wr_deact_sell': -60,
            'macd_fast': 24, 'macd_slow': 52, 'macd_sig': 9,
            'sma_len': 7,
        },
        'info': {'likes': 3250, 'version': 'v5',
                 'source': 'Scalping_with_Williams__R__MACD__an'},
    },
    'Keltner_Trend': {
        'gen':   gen_Keltner_Trend,
        'space': space_Keltner_Trend,
        'default_params': {
            'ribbon_period': 46, 'kc_len': 81, 'kc_mult': 2.5,
            'di_len': 19, 'adx_len': 10, 'dmi_benchmark': 27,
        },
        'info': {'likes': 2390, 'version': 'v5',
                 'source': 'Keltner_Channel___Trend_Based_Strat'},
    },
    'ATR_Trailing_Stop': {
        'gen':   gen_ATR_Trailing_Stop,
        'space': space_ATR_Trailing_Stop,
        'default_params': {'ap1': 5, 'af1': 0.5, 'ap2': 10, 'af2': 3.0},
        'info': {'likes': 2283, 'version': 'v4',
                 'source': 'ATR_Trailing_Stop_Strategy_by_ceyhu'},
    },
    'ATR_PSAR_QuantNomad': {
        'gen':   gen_ATR_PSAR_QuantNomad,
        'space': space_ATR_PSAR_QuantNomad,
        'default_params': {
            'atr_length': 14, 'start': 0.02, 'increment': 0.02,
            'maximum': 0.2, 'entry_bars': 1,
        },
        'info': {'likes': 1788, 'version': 'v4',
                 'source': 'ATR_Parabolic_SAR_Strategy__QuantNo'},
    },
    'Turtle_Donchian_ATR': {
        'gen':   gen_Turtle_Donchian_ATR,
        'space': space_Turtle_Donchian_ATR,
        'default_params': {
            'enter_period': 20, 'exit_period': 10,
            'atrmult': 2.0, 'atrperiod': 20,
        },
        'info': {'likes': 1709, 'version': 'v4',
                 'source': 'Turtle_trading_strategy__Donchian_A'},
    },
    'Two_TP_EMA_WMA': {
        'gen':   gen_Two_TP_EMA_WMA,
        'space': space_Two_TP_EMA_WMA,
        'default_params': {'ema_period': 10, 'wma_period': 20},
        'info': {'likes': 1503, 'version': 'v4',
                 'source': 'Two_Take_Profit_Strategy'},
    },
}
