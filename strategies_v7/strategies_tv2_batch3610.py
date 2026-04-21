"""
Batch 3610 - 5 Mac-approved families for H8 wave.

Mac ACK 2026-04-21 approved:
 - VolumeProfileDelta: order imbalance rolling delta (2-panel)
 - HeikinAshiTrend: HA candles trend regime (ANTI-REPAINT: uses shifted HA)
 - KAMA_Crossover: Kaufman Adaptive Moving Average cross vs price
 - IchimokuKumo_Breakout: price breaks above/below Kumo cloud
   (NOTE: 26/52 params ill-fitted for 5m - skip 5m in hunter_runner)
 - ElderTripleScreen: weekly trend + daily oscillator + intraday entry

All pickle-safe, no lambdas, no look-ahead (.shift(1) on signals).
Anti-repaint guard on Heikin Ashi: only use fully-closed HA bars.
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _atr(df, n):
    h, l, c = df['high'].astype(float), df['low'].astype(float), df['close'].astype(float)
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(int(n), min_periods=int(n)).mean()


# ----- 1) VOLUME PROFILE DELTA -----
def gen_TV_VolumeProfileDelta(df, delta_len=20, ema_len=40, thresh=0.35, **kw):
    """Order imbalance proxy: sign(close-open)*volume rolling sum vs its EMA.

    Entry long when delta EMA crosses above 0 with magnitude > thresh*|mean|.
    Approximates buy-sell imbalance without order book data.
    """
    c = df['close'].astype(float); o = df['open'].astype(float)
    v = df['volume'].astype(float)
    direction = np.sign(c - o)
    delta = (direction * v).rolling(int(delta_len), min_periods=int(delta_len)).sum()
    delta_ema = _ema(delta, ema_len)
    ref = delta.abs().rolling(int(ema_len), min_periods=int(ema_len)).mean().replace(0.0, np.nan)
    strong = delta_ema.abs() > float(thresh) * ref
    cross_up = (delta_ema > 0) & (delta_ema.shift(1) <= 0) & strong
    cross_dn = (delta_ema < 0) & (delta_ema.shift(1) >= 0) & strong
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False)] = 1
    sig[cross_dn.shift(1).fillna(False)] = -1
    return sig


def space_TV_VolumeProfileDelta():
    return {'delta_len': ('int', 10, 60),
            'ema_len': ('int', 20, 120),
            'thresh': ('float', 0.1, 0.8)}


# ----- 2) HEIKIN ASHI TREND (ANTI-REPAINT) -----
def gen_TV_HeikinAshiTrend(df, trend_len=5, atr_len=14, atr_mult=0.5, **kw):
    """Heikin Ashi regime with ANTI-REPAINT guard.

    HA candles computed on SHIFTED OHLC (t-1) so the bar at index t is based
    on fully-closed data. Signal further .shift(1) for next-bar entry.
    Long when N consecutive HA-green candles AND range > atr_mult*ATR.
    """
    o = df['open'].astype(float).shift(1)
    h = df['high'].astype(float).shift(1)
    l = df['low'].astype(float).shift(1)
    c = df['close'].astype(float).shift(1)
    ha_c = (o + h + l + c) / 4.0
    ha_o = (o.shift(1) + c.shift(1)) / 2.0
    # Seed first valid ha_o with (o[0]+c[0])/2 - but .shift already handles this
    ha_o = ha_o.fillna((o + c) / 2.0)
    ha_h = pd.concat([h, ha_o, ha_c], axis=1).max(axis=1)
    ha_l = pd.concat([l, ha_o, ha_c], axis=1).min(axis=1)
    green = (ha_c > ha_o)
    red = (ha_c < ha_o)
    n = int(trend_len)
    trend_green = green.rolling(n, min_periods=n).sum() >= n
    trend_red = red.rolling(n, min_periods=n).sum() >= n
    atr = _atr(df, atr_len)
    strong = (ha_h - ha_l) > float(atr_mult) * atr
    sig_long = trend_green & strong
    sig_short = trend_red & strong
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[sig_long.shift(1).fillna(False)] = 1
    sig[sig_short.shift(1).fillna(False)] = -1
    return sig


def space_TV_HeikinAshiTrend():
    return {'trend_len': ('int', 3, 10),
            'atr_len': ('int', 7, 30),
            'atr_mult': ('float', 0.2, 1.5)}


# ----- 3) KAUFMAN KAMA CROSSOVER -----
def gen_TV_KAMA_Crossover(df, er_len=10, fast=2, slow=30, **kw):
    """Kaufman Adaptive Moving Average: ER * (fast - slow) + slow smoothing.

    Signal: close crosses above/below KAMA.
    """
    c = df['close'].astype(float)
    n = int(er_len)
    change = (c - c.shift(n)).abs()
    vol = c.diff().abs().rolling(n, min_periods=n).sum()
    er = (change / vol.replace(0.0, np.nan)).fillna(0.0)
    fast_sc = 2.0 / (int(fast) + 1.0)
    slow_sc = 2.0 / (int(slow) + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Recursive KAMA
    kama = pd.Series(index=c.index, dtype=float)
    prev = c.iloc[n] if len(c) > n else float('nan')
    kama.iloc[:n + 1] = float('nan')
    for i in range(n + 1, len(c)):
        cur = prev + float(sc.iloc[i]) * (float(c.iloc[i]) - prev)
        kama.iloc[i] = cur
        prev = cur
    cross_up = (c > kama) & (c.shift(1) <= kama.shift(1))
    cross_dn = (c < kama) & (c.shift(1) >= kama.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[cross_up.shift(1).fillna(False)] = 1
    sig[cross_dn.shift(1).fillna(False)] = -1
    return sig


def space_TV_KAMA_Crossover():
    return {'er_len': ('int', 5, 40),
            'fast': ('int', 2, 5),
            'slow': ('int', 20, 60)}


# ----- 4) ICHIMOKU KUMO BREAKOUT -----
def gen_TV_IchimokuKumo_Breakout(df, tenkan=9, kijun=26, senkou=52, **kw):
    """Price breaks above senkou_a/b cloud (long) or below (short).

    NOTE: Mac flagged 26/52 ill-fitted for 5m. Hunter runner should skip 5m.
    """
    h = df['high'].astype(float); l = df['low'].astype(float); c = df['close'].astype(float)
    t = int(tenkan); k = int(kijun); s = int(senkou)
    ten = (h.rolling(t, min_periods=t).max() + l.rolling(t, min_periods=t).min()) / 2.0
    kij = (h.rolling(k, min_periods=k).max() + l.rolling(k, min_periods=k).min()) / 2.0
    senkou_a = ((ten + kij) / 2.0).shift(k)
    senkou_b = ((h.rolling(s, min_periods=s).max() + l.rolling(s, min_periods=s).min()) / 2.0).shift(k)
    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    break_up = (c > cloud_top) & (c.shift(1) <= cloud_top.shift(1))
    break_dn = (c < cloud_bot) & (c.shift(1) >= cloud_bot.shift(1))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[break_up.shift(1).fillna(False)] = 1
    sig[break_dn.shift(1).fillna(False)] = -1
    return sig


def space_TV_IchimokuKumo_Breakout():
    return {'tenkan': ('int', 6, 15),
            'kijun': ('int', 20, 40),
            'senkou': ('int', 40, 80)}


# ----- 5) ELDER TRIPLE SCREEN -----
def gen_TV_ElderTripleScreen(df, tide_len=100, wave_len=13, rsi_len=5,
                             rsi_buy=30.0, rsi_sell=70.0, **kw):
    """Elder Triple Screen proxy:

    - Screen 1 (tide): EMA slope on `tide_len` (long bias if slope > 0)
    - Screen 2 (wave): Force Index EMA `wave_len` (countertrend hints)
    - Screen 3 (entry): RSI `rsi_len` oversold (long) or overbought (short)
    """
    c = df['close'].astype(float); v = df['volume'].astype(float)
    tide = _ema(c, tide_len)
    tide_up = tide > tide.shift(1)
    tide_dn = tide < tide.shift(1)
    force = (c - c.shift(1)) * v
    fi = _ema(force, wave_len)
    wave_pullback_long = fi < 0
    wave_pullback_short = fi > 0
    # RSI
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / int(rsi_len), adjust=False, min_periods=int(rsi_len)).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / int(rsi_len), adjust=False, min_periods=int(rsi_len)).mean()
    rs = up / dn.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi_oversold = rsi < float(rsi_buy)
    rsi_overbought = rsi > float(rsi_sell)
    long_sig = tide_up & wave_pullback_long & rsi_oversold
    short_sig = tide_dn & wave_pullback_short & rsi_overbought
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_sig.shift(1).fillna(False)] = 1
    sig[short_sig.shift(1).fillna(False)] = -1
    return sig


def space_TV_ElderTripleScreen():
    return {'tide_len': ('int', 50, 200),
            'wave_len': ('int', 7, 30),
            'rsi_len': ('int', 3, 14),
            'rsi_buy': ('float', 20.0, 40.0),
            'rsi_sell': ('float', 60.0, 80.0)}


STRATEGY_EXPORT = {
    'TV_VolumeProfileDelta': {
        'gen': gen_TV_VolumeProfileDelta, 'space': space_TV_VolumeProfileDelta,
        'source': 'Order imbalance rolling delta (batch 3610)'},
    'TV_HeikinAshiTrend': {
        'gen': gen_TV_HeikinAshiTrend, 'space': space_TV_HeikinAshiTrend,
        'source': 'Heikin Ashi trend regime ANTI-REPAINT (batch 3610)'},
    'TV_KAMA_Crossover': {
        'gen': gen_TV_KAMA_Crossover, 'space': space_TV_KAMA_Crossover,
        'source': 'Kaufman Adaptive MA crossover (batch 3610)'},
    'TV_IchimokuKumo_Breakout': {
        'gen': gen_TV_IchimokuKumo_Breakout, 'space': space_TV_IchimokuKumo_Breakout,
        'source': 'Ichimoku Kumo cloud breakout NO_5m (batch 3610)'},
    'TV_ElderTripleScreen': {
        'gen': gen_TV_ElderTripleScreen, 'space': space_TV_ElderTripleScreen,
        'source': 'Elder Triple Screen (batch 3610)'},
}
