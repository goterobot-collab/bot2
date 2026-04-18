"""TV2 Batch 3598 — SMC/ICT + Quantitative/Kelly + Session/Wyckoff + Adaptive MA/Gann
               + Chart Patterns + Ichimoku + Divergence + ADX/DMI + Breakout/Elder
Source: HUNTER2 Batches 035 (24 strats) + 036 (25 strats)
Date: 2026-04-17
Total: 49 strategies
"""
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _ema(series, n):
    return series.ewm(span=n, adjust=False).mean()


def _atr(df, n=14):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _bb(series, n=20, k=2.0):
    mid = series.rolling(n).mean()
    std = series.rolling(n).std(ddof=0)
    return mid - k * std, mid, mid + k * std


def _macd(series, fast=12, slow=26, sig=9):
    m = _ema(series, fast) - _ema(series, slow)
    s = _ema(m, sig)
    return m, s


def _stoch(high, low, close, k_len=14, smooth_k=3):
    lo_k = low.rolling(k_len).min()
    hi_k = high.rolling(k_len).max()
    k = (close - lo_k) / (hi_k - lo_k + 1e-9) * 100
    return k.rolling(smooth_k).mean()


def _adx(df, n=14):
    high, low, close = df['high'], df['low'], df['close']
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = _atr(df, n)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.ewm(alpha=1/n, adjust=False).mean()
    return adx, plus_di, minus_di


def _swing_highs_lows(high, low, left=5, right=5):
    """Return boolean series for swing highs/lows."""
    sh = pd.Series(False, index=high.index)
    sl = pd.Series(False, index=low.index)
    for i in range(left, len(high) - right):
        if all(high.iloc[i] >= high.iloc[i - left:i]) and all(high.iloc[i] >= high.iloc[i + 1:i + right + 1]):
            sh.iloc[i] = True
        if all(low.iloc[i] <= low.iloc[i - left:i]) and all(low.iloc[i] <= low.iloc[i + 1:i + right + 1]):
            sl.iloc[i] = True
    return sh, sl


def _resample_htf(df, tf_minutes):
    """Resample 1h OHLCV to higher timeframe, shift(1) to avoid lookahead."""
    rule = f'{tf_minutes}min'
    htf = df.resample(rule).agg({'open': 'first', 'high': 'max', 'low': 'min',
                                   'close': 'last', 'volume': 'sum'}).dropna()
    return htf.shift(1).reindex(df.index, method='ffill')


def _kama(series, n=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average."""
    close = series.values
    kama = np.full(len(close), np.nan)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    for i in range(n, len(close)):
        direction = abs(close[i] - close[i - n])
        volatility = sum(abs(close[j] - close[j - 1]) for j in range(i - n + 1, i + 1))
        er = direction / volatility if volatility != 0 else 0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        prev = kama[i - 1] if not np.isnan(kama[i - 1]) else close[i - 1]
        kama[i] = prev + sc * (close[i] - prev)
    return pd.Series(kama, index=series.index)


def _ichimoku(df, tenkan=9, kijun=26, senkou_b=52):
    high, low = df['high'], df['low']
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    chikou = df['close'].shift(-kijun)
    return tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou


def _wavetrend(df, ch_len=10, avg_len=21):
    esa = _ema((df['high'] + df['low'] + df['close']) / 3, ch_len)
    d = _ema(((df['high'] + df['low'] + df['close']) / 3 - esa).abs(), ch_len)
    ci = ((df['high'] + df['low'] + df['close']) / 3 - esa) / (0.015 * d.replace(0, np.nan))
    wt1 = _ema(ci, avg_len)
    wt2 = wt1.rolling(4).mean()
    return wt1, wt2


def _ssl_channel(df, n=10):
    sma_high = df['high'].rolling(n).mean()
    sma_low = df['low'].rolling(n).mean()
    hlv = pd.Series(np.nan, index=df.index)
    for i in range(1, len(df)):
        if df['close'].iloc[i] > sma_high.iloc[i]:
            hlv.iloc[i] = 1
        elif df['close'].iloc[i] < sma_low.iloc[i]:
            hlv.iloc[i] = -1
        else:
            hlv.iloc[i] = hlv.iloc[i - 1]
    ssl_up = pd.Series(np.where(hlv < 0, sma_high, sma_low), index=df.index)
    ssl_dn = pd.Series(np.where(hlv < 0, sma_low, sma_high), index=df.index)
    return ssl_up, ssl_dn


# ---------------------------------------------------------------------------
# BATCH 035 — SMC/ICT (9)
# ---------------------------------------------------------------------------

def gen_TV_SMC_OB_FVG_DOE(df, ob_len=10, fvg_mult=0.5, atr_len=14, trend_len=50):
    """SMC Pro BTC ICT OB+FVG [DOE_Trade] — Order Blocks + Fair Value Gaps.
    Source: https://www.tradingview.com/script/QMvHkvdQ-SMC-Pro-BTC-ICT-Order-Blocks-FVG-DOE/
    Concept: Bullish OB = last down-candle before an up-move; FVG = gap between bars.
    """
    close, high, low, open_ = df['close'], df['high'], df['low'], df['open']
    atr = _atr(df, atr_len)
    trend = _ema(close, trend_len)

    # Bullish OB: bearish candle followed by strong bullish move
    bear_candle = close < open_
    bull_move = (close.shift(-1) - open_.shift(-1)) > atr * 0.5
    ob_bull = bear_candle & bull_move

    # Bearish OB: bullish candle followed by strong bearish move
    bull_candle = close > open_
    bear_move = (open_.shift(-1) - close.shift(-1)) > atr * 0.5
    ob_bear = bull_candle & bear_move

    # FVG: gap between bar[i-1].high and bar[i+1].low (bullish FVG)
    fvg_bull = low.shift(-1) > high.shift(1)
    fvg_bear = high.shift(-1) < low.shift(1)

    # Price revisiting OB zone
    ob_bull_zone = ob_bull.rolling(ob_len).max().astype(bool)
    ob_bear_zone = ob_bear.rolling(ob_len).max().astype(bool)

    sig = pd.Series(0, index=df.index)
    # Long: price in bullish OB zone + above trend + bullish FVG nearby
    sig[(ob_bull_zone) & (close > trend) & (fvg_bull.rolling(3).max().astype(bool))] = 1
    sig[(ob_bear_zone) & (close < trend) & (fvg_bear.rolling(3).max().astype(bool))] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_SMC_OB_FVG_DOE():
    return {
        'ob_len': ('int', 5, 20),
        'fvg_mult': ('float', 0.3, 1.0),
        'atr_len': ('int', 10, 20),
        'trend_len': ('int', 30, 80),
    }


def gen_TV_ES_MTF_SMC_Entry(df, htf_minutes=240, ob_len=8, conf_thresh=6, atr_len=14):
    """ES MTF SMC Entry System [Prototype1111] — Multi-TF confluence scoring.
    Source: https://www.tradingview.com/script/cVMaQmZg-ES-Multi-Timeframe-SMC-Entry-System/
    Concept: 10-point confluence score across trend/structure/OB/FVG/momentum.
    """
    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)
    htf = _resample_htf(df, htf_minutes)

    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    rsi = _rsi(close, 14)
    htf_close = htf['close'] if 'close' in htf.columns else _ema(close, 20)

    score = pd.Series(0.0, index=df.index)
    score += (close > ema20).astype(float)
    score += (ema20 > ema50).astype(float)
    score += (close > htf_close).astype(float)
    score += (rsi > 50).astype(float)
    score += (rsi.diff() > 0).astype(float)
    bear_can = (close < close.shift(1))
    ob_zone = bear_can.rolling(ob_len).max().astype(bool)
    score += ob_zone.astype(float)
    fvg = low.shift(-1) > high.shift(1)
    score += fvg.rolling(4).max().astype(float)
    vol_surge = df['volume'] > df['volume'].rolling(20).mean() * 1.2
    score += vol_surge.astype(float)
    score += (close > (high.rolling(20).max() + low.rolling(20).min()) / 2).astype(float)
    score += (close.pct_change(5) > 0).astype(float)

    sig = pd.Series(0, index=df.index)
    sig[score >= conf_thresh] = 1
    sig[score <= (10 - conf_thresh)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_ES_MTF_SMC_Entry():
    return {
        'htf_minutes': ('int', 60, 480),
        'ob_len': ('int', 4, 16),
        'conf_thresh': ('int', 5, 9),
        'atr_len': ('int', 10, 20),
    }


def gen_TV_SMC_Trap_FVG(df, fvg_len=3, trap_atr=1.0, atr_len=14, ema_len=50):
    """SMC Trap and FVG Strategy [relivewithros] — Liquidity trap + FVG fill.
    Source: https://www.tradingview.com/script/qX06Avws-SMC-Trap-and-FVG-Strategy/
    Concept: Price sweeps a level (trap), then FVG fill signals reversal.
    """
    close, high, low, open_ = df['close'], df['high'], df['low'], df['open']
    atr = _atr(df, atr_len)
    trend = _ema(close, ema_len)

    prev_high = high.shift(1)
    prev_low = low.shift(1)

    # Trap up: high sweeps prev high then closes below (bear trap)
    trap_bear = (high > prev_high) & (close < prev_high)
    # Trap down: low sweeps prev low then closes above (bull trap)
    trap_bull = (low < prev_low) & (close > prev_low)

    # FVG: imbalance zone
    fvg_bull = low.shift(-1) > high.shift(1)
    fvg_bear = high.shift(-1) < low.shift(1)

    sig = pd.Series(0, index=df.index)
    sig[(trap_bull.shift(1)) & (fvg_bull.rolling(fvg_len).max().astype(bool)) & (close > trend)] = 1
    sig[(trap_bear.shift(1)) & (fvg_bear.rolling(fvg_len).max().astype(bool)) & (close < trend)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_SMC_Trap_FVG():
    return {
        'fvg_len': ('int', 2, 8),
        'trap_atr': ('float', 0.5, 2.0),
        'atr_len': ('int', 10, 20),
        'ema_len': ('int', 30, 80),
    }


def gen_TV_SMC_Fractal_v3(df, fractal_left=5, fractal_right=5, atr_len=14,
                            session_start=8, session_end=20):
    """SMC Fractal Strategy v3 [JamolCooper] — BOS + Fractal OB + session filter.
    Source: https://www.tradingview.com/script/97G0VL40-SMC-Fractal-Strategy-Jamol-v3/
    Concept: Break of Structure using fractals, session NY+London filter.
    """
    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)

    sh, sl = _swing_highs_lows(high, low, fractal_left, fractal_right)

    # BOS: close breaks above last swing high (bullish BOS)
    last_sh = high.where(sh).ffill()
    last_sl = low.where(sl).ffill()

    bos_bull = close > last_sh.shift(1)
    bos_bear = close < last_sl.shift(1)

    # Session filter
    hour = df.index.hour
    in_session = (hour >= session_start) & (hour <= session_end)

    sig = pd.Series(0, index=df.index)
    sig[bos_bull & in_session] = 1
    sig[bos_bear & in_session] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_SMC_Fractal_v3():
    return {
        'fractal_left': ('int', 3, 10),
        'fractal_right': ('int', 3, 10),
        'atr_len': ('int', 10, 20),
        'session_start': ('int', 6, 10),
        'session_end': ('int', 16, 22),
    }


def gen_TV_LuxAlgo_SMC_Ultimate(df, ob_len=10, fvg_len=4, premium=0.7, discount=0.3,
                                  ema_len=200):
    """LuxAlgo SMC Pro Ultimate v6 [HAC19] — Premium/Discount zones + OB + FVG.
    Source: https://www.tradingview.com/script/4AzLMpYa-LuxAlgo-SMC-Pro-Ultimate-v6/
    Concept: Price action in premium (>70% of range) = potential short; discount = long.
    """
    close, high, low = df['close'], df['high'], df['low']
    trend = _ema(close, ema_len)

    # Range from recent swing
    rng_high = high.rolling(50).max()
    rng_low = low.rolling(50).min()
    rng = rng_high - rng_low + 1e-9
    pct_pos = (close - rng_low) / rng

    # OB zones
    bear_can = close < close.shift(1)
    bull_can = close > close.shift(1)
    ob_bull_zone = bear_can.rolling(ob_len).max().astype(bool)
    ob_bear_zone = bull_can.rolling(ob_len).max().astype(bool)

    # FVG
    fvg_bull = (low.shift(-1) > high.shift(1)).rolling(fvg_len).max().astype(bool)
    fvg_bear = (high.shift(-1) < low.shift(1)).rolling(fvg_len).max().astype(bool)

    sig = pd.Series(0, index=df.index)
    sig[(pct_pos < discount) & ob_bull_zone & fvg_bull & (close > trend)] = 1
    sig[(pct_pos > premium) & ob_bear_zone & fvg_bear & (close < trend)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_LuxAlgo_SMC_Ultimate():
    return {
        'ob_len': ('int', 5, 20),
        'fvg_len': ('int', 2, 8),
        'premium': ('float', 0.6, 0.85),
        'discount': ('float', 0.15, 0.4),
        'ema_len': ('int', 100, 300),
    }


def gen_TV_ICT_Killzones_Pivots(df, kz_start=7, kz_end=10, pivot_len=5, atr_len=14):
    """2N STRAT ICT Killzones Pivots [DevBullish] — ICT Kill Zone + Pivot entry.
    Source: https://www.tradingview.com/script/uGP5w2ki-2N-STRAT-ICT-Killzones-Pivots/
    Concept: Trade reversals at pivot S/R during ICT kill zones (London/NY open).
    """
    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)

    hour = df.index.hour
    in_kz = (hour >= kz_start) & (hour <= kz_end)

    sh, sl = _swing_highs_lows(high, low, pivot_len, pivot_len)
    piv_high = high.where(sh).ffill()
    piv_low = low.where(sl).ffill()

    # Bounce from pivot low in kill zone
    near_piv_low = (close - piv_low).abs() < atr * 0.5
    near_piv_high = (close - piv_high).abs() < atr * 0.5

    sig = pd.Series(0, index=df.index)
    sig[near_piv_low & in_kz & (close > close.shift(1))] = 1
    sig[near_piv_high & in_kz & (close < close.shift(1))] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_ICT_Killzones_Pivots():
    return {
        'kz_start': ('int', 6, 10),
        'kz_end': ('int', 9, 14),
        'pivot_len': ('int', 3, 10),
        'atr_len': ('int', 10, 20),
    }


def gen_TV_Liquidity_Sweep_Filter(df, sweep_len=20, atr_mult=0.5, atr_len=14, ema_len=50):
    """Liquidity Sweep Filter [AlgoAlpha x PineIndicators] — Liquidity sweep reversal.
    Source: https://www.tradingview.com/script/gx7267BR-Liquidity-Sweep-Filter-Strategy/
    Concept: Price sweeps above/below prior high/low then reverses = liquidity grab.
    """
    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)
    trend = _ema(close, ema_len)

    prior_high = high.rolling(sweep_len).max().shift(1)
    prior_low = low.rolling(sweep_len).min().shift(1)

    # Sweep high: high exceeds prior_high but close back below
    sweep_high = (high > prior_high) & (close < prior_high - atr * atr_mult)
    # Sweep low: low below prior_low but close back above
    sweep_low = (low < prior_low) & (close > prior_low + atr * atr_mult)

    sig = pd.Series(0, index=df.index)
    sig[sweep_low & (close > trend)] = 1
    sig[sweep_high & (close < trend)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Liquidity_Sweep_Filter():
    return {
        'sweep_len': ('int', 10, 40),
        'atr_mult': ('float', 0.3, 1.5),
        'atr_len': ('int', 10, 20),
        'ema_len': ('int', 30, 100),
    }


def gen_TV_SMC_Liquidity_Grab_Pro(df, grab_len=15, confirm_bars=2, atr_len=14, ema_len=50):
    """SMC Liquidity Grab Pro [Ericem] — Liquidity grab + confirmation candle.
    Source: https://www.tradingview.com/script/WZ4s1MRC-SMC-Liquidity-Grab-Pro/
    Concept: Sharp wick beyond S/R + reversal confirmation = institutional grab.
    """
    close, high, low, open_ = df['close'], df['high'], df['low'], df['open']
    atr = _atr(df, atr_len)
    trend = _ema(close, ema_len)

    level_high = high.rolling(grab_len).max().shift(1)
    level_low = low.rolling(grab_len).min().shift(1)

    wick_up = high - pd.concat([close, open_], axis=1).max(axis=1)
    wick_dn = pd.concat([close, open_], axis=1).min(axis=1) - low

    # Bearish grab: wick up exceeds level_high, body below level
    grab_bear = (high > level_high) & (wick_up > atr * 0.3) & (close < level_high)
    # Bullish grab: wick down below level_low, body above level
    grab_bull = (low < level_low) & (wick_dn > atr * 0.3) & (close > level_low)

    sig = pd.Series(0, index=df.index)
    sig[grab_bull.shift(confirm_bars) & (close > trend)] = 1
    sig[grab_bear.shift(confirm_bars) & (close < trend)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_SMC_Liquidity_Grab_Pro():
    return {
        'grab_len': ('int', 8, 30),
        'confirm_bars': ('int', 1, 4),
        'atr_len': ('int', 10, 20),
        'ema_len': ('int', 30, 100),
    }


def gen_TV_Casper_SMC_ORB_Retest(df, orb_bars=6, retest_atr=0.3, atr_len=14):
    """Casper SMC 5m ORB Retest — Opening Range Breakout with retest.
    Source: https://www.tradingview.com/script/muLbjEdA-Casper-SMC-5m-ORB-Retest/
    Concept: First N bars form opening range; breakout + retest = entry.
    """
    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)

    hour = df.index.hour
    minute = df.index.minute if hasattr(df.index, 'minute') else pd.Series(0, index=df.index)

    # ORB: high/low of first orb_bars of each day
    orb_high = pd.Series(np.nan, index=df.index)
    orb_low = pd.Series(np.nan, index=df.index)
    dates = df.index.normalize().unique()
    for d in dates:
        mask = df.index.normalize() == d
        day_idx = df.index[mask]
        if len(day_idx) >= orb_bars:
            orb_h = high[mask].iloc[:orb_bars].max()
            orb_l = low[mask].iloc[:orb_bars].min()
            orb_high[mask] = orb_h
            orb_low[mask] = orb_l

    orb_high = orb_high.ffill()
    orb_low = orb_low.ffill()

    # Breakout above ORB high
    broke_high = close > orb_high
    broke_low = close < orb_low

    # Retest: price comes back near the breakout level
    retest_bull = (close - orb_high).abs() < atr * retest_atr
    retest_bear = (close - orb_low).abs() < atr * retest_atr

    sig = pd.Series(0, index=df.index)
    sig[broke_high.shift(1) & retest_bull] = 1
    sig[broke_low.shift(1) & retest_bear] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Casper_SMC_ORB_Retest():
    return {
        'orb_bars': ('int', 3, 12),
        'retest_atr': ('float', 0.2, 0.8),
        'atr_len': ('int', 10, 20),
    }


# ---------------------------------------------------------------------------
# BATCH 035 — QUANTITATIVE / KELLY (3)
# ---------------------------------------------------------------------------

def gen_TV_Bayesian_Kelly(df, wr_window=20, kelly_mult=0.5, rsi_len=14, atr_len=14):
    """Bayesian Kelly Strategy [nasu_is_gaji] — Bayesian WR updating + Kelly sizing.
    Source: https://www.tradingview.com/script/MynxSWHC-Bayesian-Kelly-Strategy/
    Concept: EMA of rolling win rate as Bayesian estimate; enter when Kelly fraction positive.
    """
    close = df['close']
    atr = _atr(df, atr_len)
    rsi = _rsi(close, rsi_len)

    # Rolling win proxy: bars where close > prev close
    wins = (close > close.shift(1)).astype(float)
    wr_est = wins.ewm(span=wr_window, adjust=False).mean()  # Bayesian EMA update

    # Kelly fraction: f = wr - (1-wr) = 2*wr - 1
    kelly = 2 * wr_est - 1

    # Trend confirmation
    ema_fast = _ema(close, 10)
    ema_slow = _ema(close, 30)

    sig = pd.Series(0, index=df.index)
    sig[(kelly > kelly_mult * 0.1) & (ema_fast > ema_slow) & (rsi > 50)] = 1
    sig[(kelly < -kelly_mult * 0.1) & (ema_fast < ema_slow) & (rsi < 50)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Bayesian_Kelly():
    return {
        'wr_window': ('int', 10, 50),
        'kelly_mult': ('float', 0.2, 1.0),
        'rsi_len': ('int', 10, 21),
        'atr_len': ('int', 10, 20),
    }


def gen_TV_Kelly_Dynamic_Sizing(df, kelly_window=30, rsi_len=14, ema_fast=12, ema_slow=26):
    """Kelly Ratio Dynamic Sizing [CryptoRox] — Kelly criterion as signal filter.
    Source: https://www.tradingview.com/script/bFXf4IXh-Built-in-Kelly-ratio-for-dynamic-position-sizing/
    Concept: Kelly = WR - (1-WR)/RR; positive Kelly = trade direction of momentum.
    """
    close = df['close']
    rsi = _rsi(close, rsi_len)
    ef = _ema(close, ema_fast)
    es = _ema(close, ema_slow)

    wins = (close > close.shift(1)).astype(float)
    wr = wins.rolling(kelly_window).mean()
    avg_win = close.pct_change().clip(lower=0).rolling(kelly_window).mean()
    avg_loss = (-close.pct_change().clip(upper=0)).rolling(kelly_window).mean().replace(0, np.nan)
    rr = avg_win / avg_loss
    kelly = wr - (1 - wr) / rr.replace(0, np.nan)

    sig = pd.Series(0, index=df.index)
    sig[(kelly > 0) & (ef > es)] = 1
    sig[(kelly < 0) & (ef < es)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Kelly_Dynamic_Sizing():
    return {
        'kelly_window': ('int', 15, 60),
        'rsi_len': ('int', 10, 21),
        'ema_fast': ('int', 8, 20),
        'ema_slow': ('int', 20, 50),
    }


def gen_TV_Dual_Momentum(df, mom_fast=12, mom_slow=26, threshold=0.0, ema_len=50):
    """Dual Momentum Strategy [EdgeTools] — Relative + absolute momentum dual filter.
    Source: https://www.tradingview.com/script/wFRnnlQr/
    Concept: Crypto adaptation — fast mom vs slow mom as relative; abs mom vs 0.
    """
    close = df['close']
    trend = _ema(close, ema_len)

    # Absolute momentum
    abs_mom = close.pct_change(mom_slow) * 100
    # Relative momentum: fast vs slow ROC
    rel_mom = close.pct_change(mom_fast) - close.pct_change(mom_slow)

    sig = pd.Series(0, index=df.index)
    sig[(abs_mom > threshold) & (rel_mom > 0) & (close > trend)] = 1
    sig[(abs_mom < -threshold) & (rel_mom < 0) & (close < trend)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Dual_Momentum():
    return {
        'mom_fast': ('int', 5, 20),
        'mom_slow': ('int', 20, 60),
        'threshold': ('float', 0.0, 2.0),
        'ema_len': ('int', 30, 100),
    }


# ---------------------------------------------------------------------------
# BATCH 035 — MOMENTUM (2)
# ---------------------------------------------------------------------------

def gen_TV_Momentum_Strategy_REV(df, mom_len=14, rsi_len=14, ema_len=50, vol_mult=1.2):
    """Momentum Strategy [REV0LUTI0N] — Momentum + RSI + volume confirmation.
    Source: https://www.tradingview.com/script/XhCZbT4b-Momentum-Strategy/
    Concept: Rate-of-change momentum with RSI and volume surge filter.
    """
    close = df['close']
    vol = df['volume']

    mom = close.pct_change(mom_len) * 100
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, ema_len)
    vol_avg = vol.rolling(20).mean()

    sig = pd.Series(0, index=df.index)
    sig[(mom > 0) & (rsi > 50) & (close > trend) & (vol > vol_avg * vol_mult)] = 1
    sig[(mom < 0) & (rsi < 50) & (close < trend) & (vol > vol_avg * vol_mult)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Momentum_Strategy_REV():
    return {
        'mom_len': ('int', 8, 30),
        'rsi_len': ('int', 10, 21),
        'ema_len': ('int', 30, 100),
        'vol_mult': ('float', 1.0, 2.0),
    }


def gen_TV_Momentum_Long_REV(df, mom_len=14, rsi_ob=60, ema_fast=10, ema_slow=30):
    """Momentum Long Strategy [REV0LUTI0N] — Long-only momentum with RSI filter.
    Source: https://www.tradingview.com/script/gEIISPE5-Momentum-Long-Strategy/
    Concept: Long when positive momentum and RSI above threshold.
    """
    close = df['close']
    mom = close.pct_change(mom_len) * 100
    rsi = _rsi(close, 14)
    ef = _ema(close, ema_fast)
    es = _ema(close, ema_slow)

    sig = pd.Series(0, index=df.index)
    sig[(mom > 0) & (rsi > rsi_ob) & (ef > es)] = 1
    sig[(mom < 0) | (rsi < 50)] = -1  # exit
    # Keep only transitions
    sig = sig.where(sig != sig.shift(1), 0)
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Momentum_Long_REV():
    return {
        'mom_len': ('int', 8, 30),
        'rsi_ob': ('int', 55, 70),
        'ema_fast': ('int', 5, 20),
        'ema_slow': ('int', 20, 60),
    }


# ---------------------------------------------------------------------------
# BATCH 035 — VIX REGIME / SESSION / WYCKOFF (5)
# ---------------------------------------------------------------------------

def gen_TV_VIX_Regime(df, atr_fast=5, atr_slow=21, fear_mult=1.5, ema_len=50):
    """VIX Futures Spread Regime Strategy [jtonka] — ATR ratio as VIX proxy.
    Source: https://www.tradingview.com/script/pn4ymGUu-VIX-Futures-Spread-Strategy/
    Concept: ATR_fast/ATR_slow ratio as crypto fear proxy; trade trend in calm regime.
    """
    close = df['close']
    atr_f = _atr(df, atr_fast)
    atr_s = _atr(df, atr_slow)
    regime_ratio = atr_f / atr_s.replace(0, np.nan)
    calm = regime_ratio < fear_mult
    trend = _ema(close, ema_len)
    rsi = _rsi(close, 14)

    sig = pd.Series(0, index=df.index)
    sig[calm & (close > trend) & (rsi > 50)] = 1
    sig[calm & (close < trend) & (rsi < 50)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_VIX_Regime():
    return {
        'atr_fast': ('int', 3, 10),
        'atr_slow': ('int', 14, 30),
        'fear_mult': ('float', 1.2, 2.5),
        'ema_len': ('int', 30, 100),
    }


def gen_TV_Long_ORB_Pivot(df, orb_bars=4, pivot_len=5, atr_len=14):
    """Long-Only ORB with Pivot Points [VolumeVigilante] — ORB breakout above pivot.
    Source: https://www.tradingview.com/script/6kDE9bLA-Long-Only-Opening-Range-Breakout-ORB/
    Concept: ORB high breakout confirmed when price is above daily pivot level.
    """
    close, high, low, open_ = df['close'], df['high'], df['low'], df['open']

    # Daily pivot
    prev_high = high.resample('D').max().shift(1).reindex(df.index, method='ffill')
    prev_low = low.resample('D').min().shift(1).reindex(df.index, method='ffill')
    prev_close = close.resample('D').last().shift(1).reindex(df.index, method='ffill')
    pivot = (prev_high + prev_low + prev_close) / 3

    # ORB
    orb_high = pd.Series(np.nan, index=df.index)
    for d in df.index.normalize().unique():
        mask = df.index.normalize() == d
        day_idx = df.index[mask]
        if len(day_idx) >= orb_bars:
            orb_high[mask] = high[mask].iloc[:orb_bars].max()
    orb_high = orb_high.ffill()

    sig = pd.Series(0, index=df.index)
    sig[(close > orb_high.shift(1)) & (close > pivot)] = 1
    sig[close < pivot] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Long_ORB_Pivot():
    return {
        'orb_bars': ('int', 2, 8),
        'pivot_len': ('int', 3, 10),
        'atr_len': ('int', 10, 20),
    }


def gen_TV_Gold_Asia_Session(df, asia_start=0, asia_end=8, breakout_mult=0.5, atr_len=14):
    """Gold Asia Session Breakout [bradenstrock] — Asia range breakout.
    Source: https://www.tradingview.com/script/UNq4BtXc-Gold-Asia-Session-Breakout-Entries-Exits/
    Concept: Build range during Asia session; trade breakout in London/NY.
    """
    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)
    hour = df.index.hour

    asia_mask = (hour >= asia_start) & (hour < asia_end)
    asia_high = pd.Series(np.nan, index=df.index)
    asia_low = pd.Series(np.nan, index=df.index)
    for d in df.index.normalize().unique():
        day_mask = df.index.normalize() == d
        am = day_mask & asia_mask
        if am.sum() > 0:
            ah = high[am].max()
            al = low[am].min()
            asia_high[day_mask] = ah
            asia_low[day_mask] = al
    asia_high = asia_high.ffill()
    asia_low = asia_low.ffill()

    sig = pd.Series(0, index=df.index)
    sig[(close > asia_high + atr * breakout_mult) & (~asia_mask)] = 1
    sig[(close < asia_low - atr * breakout_mult) & (~asia_mask)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Gold_Asia_Session():
    return {
        'asia_start': ('int', 0, 3),
        'asia_end': ('int', 6, 10),
        'breakout_mult': ('float', 0.2, 1.5),
        'atr_len': ('int', 10, 20),
    }


def gen_TV_Pivot_Reversal_Backtest(df, pivot_left=4, pivot_right=4, atr_len=14):
    """Pivot Reversal with Backtest Range [QuantNomad] — Swing pivot reversal.
    Source: https://www.tradingview.com/script/fq8iiXQI-Pivot-Reversal-Strategy/
    Concept: Long on swing low pivot, short on swing high pivot.
    """
    close, high, low = df['close'], df['high'], df['low']

    sh, sl = _swing_highs_lows(high, low, pivot_left, pivot_right)

    sig = pd.Series(0, index=df.index)
    sig[sl] = 1   # Long on swing low (reversal up expected)
    sig[sh] = -1  # Short on swing high (reversal down expected)
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Pivot_Reversal_Backtest():
    return {
        'pivot_left': ('int', 2, 10),
        'pivot_right': ('int', 2, 10),
        'atr_len': ('int', 10, 20),
    }


def gen_TV_Wyckoff_Range(df, range_len=30, phase_thresh=0.3, atr_len=14, vol_mult=1.5):
    """Wyckoff Range Strategy [deperp] — Accumulation/Distribution phase detection.
    Source: https://www.tradingview.com/script/vQSBf9rh-Wyckoff-Range-Strategy/
    Concept: Detect tight range (low ATR relative to history) + spring/upthrust entry.
    """
    close, high, low = df['close'], df['high'], df['low']
    vol = df['volume']
    atr = _atr(df, atr_len)

    # Range: ATR compressed vs its longer average
    atr_long = atr.rolling(range_len).mean()
    in_range = atr < atr_long * phase_thresh

    # Spring (Wyckoff): low dips below range_low then recovers
    rng_low = low.rolling(range_len).min()
    rng_high = high.rolling(range_len).max()
    spring = (close.shift(1) < rng_low.shift(1)) & (close > rng_low)
    upthrust = (close.shift(1) > rng_high.shift(1)) & (close < rng_high)

    vol_surge = vol > vol.rolling(20).mean() * vol_mult

    sig = pd.Series(0, index=df.index)
    sig[spring & vol_surge] = 1
    sig[upthrust & vol_surge] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Wyckoff_Range():
    return {
        'range_len': ('int', 15, 60),
        'phase_thresh': ('float', 0.2, 0.6),
        'atr_len': ('int', 10, 20),
        'vol_mult': ('float', 1.2, 2.5),
    }


# ---------------------------------------------------------------------------
# BATCH 035 — ADAPTIVE MA / GANN (5)
# ---------------------------------------------------------------------------

def gen_TV_WaveTrend_Plus(df, ch_len=10, avg_len=21, ob_level=53, os_level=-53):
    """WaveTrend+ Strategy [SystemAlpha] — WaveTrend oscillator with OB/OS levels.
    Source: https://www.tradingview.com/script/bPJohkjs-WaveTrend-Strategy-SystemAlpha/
    Concept: WaveTrend cross at OB/OS levels for mean-reversion entries.
    """
    close = df['close']
    wt1, wt2 = _wavetrend(df, ch_len, avg_len)

    cross_up = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_dn = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[cross_up & (wt1 < os_level)] = 1
    sig[cross_dn & (wt1 > ob_level)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_WaveTrend_Plus():
    return {
        'ch_len': ('int', 6, 16),
        'avg_len': ('int', 10, 30),
        'ob_level': ('float', 40, 70),
        'os_level': ('float', -70, -40),
    }


def gen_TV_KAMA_Adaptive(df, kama_n=10, kama_fast=2, kama_slow=30, atr_len=14):
    """KAMA Adaptive MA Backtest [HPotter] — Kaufman Adaptive MA crossover.
    Source: https://www.tradingview.com/script/KVMbvcHV-Kaufman-Moving-Average-Adaptive-KAMA/
    Concept: KAMA adapts speed to market efficiency; cross with price signals trend.
    """
    close = df['close']
    kama = _kama(close, kama_n, kama_fast, kama_slow)
    atr = _atr(df, atr_len)

    sig = pd.Series(0, index=df.index)
    sig[close > kama + atr * 0.1] = 1
    sig[close < kama - atr * 0.1] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_KAMA_Adaptive():
    return {
        'kama_n': ('int', 5, 20),
        'kama_fast': ('int', 2, 5),
        'kama_slow': ('int', 20, 50),
        'atr_len': ('int', 10, 20),
    }


def gen_TV_Gann_Trend_Oscillator(df, gann_len=14, smooth=3, ema_len=50):
    """Gann Trend Oscillator Backtest [HPotter] — Gann price range oscillator.
    Source: https://www.tradingview.com/script/IxCdFCjE-Gann-Trend-Oscillator-Backtest/
    Concept: (H+L)/2 momentum oscillator smoothed — cross zero = trend change.
    """
    close, high, low = df['close'], df['high'], df['low']
    hl2 = (high + low) / 2
    gann_osc = hl2 - hl2.shift(gann_len)
    gann_smooth = gann_osc.rolling(smooth).mean()
    trend = _ema(close, ema_len)

    sig = pd.Series(0, index=df.index)
    sig[(gann_smooth > 0) & (gann_smooth.shift(1) <= 0) & (close > trend)] = 1
    sig[(gann_smooth < 0) & (gann_smooth.shift(1) >= 0) & (close < trend)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Gann_Trend_Oscillator():
    return {
        'gann_len': ('int', 5, 30),
        'smooth': ('int', 2, 8),
        'ema_len': ('int', 30, 100),
    }


def gen_TV_Combo_123_Gann(df, reversal_bars=3, gann_len=14, ema_len=50, atr_len=14):
    """Combo 123 Reversal + Gann Swing [HPotter] — 1-2-3 pattern + Gann filter.
    Source: https://www.tradingview.com/script/gAOca1TB-Combo-Backtest-123-Reversal-Gann-Swing/
    Concept: 1-2-3 reversal pattern confirmed by Gann oscillator direction.
    """
    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)
    trend = _ema(close, ema_len)
    hl2 = (high + low) / 2
    gann_osc = hl2 - hl2.shift(gann_len)

    # 1-2-3 bullish: lower low, then higher low, then breakout
    p1 = low.shift(reversal_bars * 2)
    p2 = low.shift(reversal_bars)
    p3 = low
    pattern_bull = (p2 < p1) & (p3 > p2) & (close > close.shift(reversal_bars))

    p1h = high.shift(reversal_bars * 2)
    p2h = high.shift(reversal_bars)
    p3h = high
    pattern_bear = (p2h > p1h) & (p3h < p2h) & (close < close.shift(reversal_bars))

    sig = pd.Series(0, index=df.index)
    sig[pattern_bull & (gann_osc > 0) & (close > trend)] = 1
    sig[pattern_bear & (gann_osc < 0) & (close < trend)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Combo_123_Gann():
    return {
        'reversal_bars': ('int', 2, 6),
        'gann_len': ('int', 5, 25),
        'ema_len': ('int', 30, 100),
        'atr_len': ('int', 10, 20),
    }


def gen_TV_Gann_Swing_MultiLayer(df, swing_len=5, layers=3, atr_len=14, ema_len=50):
    """Gann Swing 1-Bar Multi-Layer [AutomatedTradingAlgorithms] — Multi-layer Gann swing.
    Source: https://www.tradingview.com/script/hiduNmYl-Gann-Swing-Strategy-1-Bar-Multi-Layer/
    Concept: Multi-timeframe Gann swing: short/medium/long swing alignment for entry.
    """
    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)
    trend = _ema(close, ema_len)

    scores = pd.Series(0.0, index=df.index)
    for layer in range(1, layers + 1):
        n = swing_len * layer
        sh, sl = _swing_highs_lows(high, low, n, n)
        last_sh = high.where(sh).ffill()
        last_sl = low.where(sl).ffill()
        scores += (close > last_sh).astype(float)
        scores -= (close < last_sl).astype(float)

    sig = pd.Series(0, index=df.index)
    sig[(scores >= layers) & (close > trend)] = 1
    sig[(scores <= -layers) & (close < trend)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Gann_Swing_MultiLayer():
    return {
        'swing_len': ('int', 3, 10),
        'layers': ('int', 2, 4),
        'atr_len': ('int', 10, 20),
        'ema_len': ('int', 30, 100),
    }


# ---------------------------------------------------------------------------
# BATCH 036 — HARMONIC PATTERNS (4)
# ---------------------------------------------------------------------------

def gen_TV_Auto_Harmonic_Pattern(df, pivot_left=5, pivot_right=5, gartley_xab=0.618,
                                   tolerance=0.05, atr_len=14):
    """Auto Harmonic Pattern Backtester [Trendoscope] — Gartley/Bat/Butterfly/Crab.
    Source: https://www.tradingview.com/script/1BTKcWcE-Auto-Harmonic-Pattern-Backtester/
    Concept: XABCD pattern with Fibonacci ratio validation at each leg.
    """
    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)

    sh, sl = _swing_highs_lows(high, low, pivot_left, pivot_right)
    swing_h_vals = high.where(sh)
    swing_l_vals = low.where(sl)

    # Build simplified Gartley: X=swing_high[-3], A=swing_low[-2], B=swing_high[-1]
    sh_idx = swing_h_vals.dropna().index
    sl_idx = swing_l_vals.dropna().index

    sig = pd.Series(0, index=df.index)
    if len(sh_idx) >= 2 and len(sl_idx) >= 2:
        for i in range(2, min(len(sh_idx), len(sl_idx), len(df) - 1)):
            X = swing_h_vals[sh_idx[i - 2]] if i - 2 < len(sh_idx) else None
            A = swing_l_vals[sl_idx[i - 2]] if i - 2 < len(sl_idx) else None
            B = swing_h_vals[sh_idx[i - 1]] if i - 1 < len(sh_idx) else None
            if X is None or A is None or B is None:
                continue
            xa = X - A
            ab = B - A
            if xa == 0:
                continue
            xab_ratio = ab / xa
            if abs(xab_ratio - gartley_xab) < tolerance:
                # Bullish Gartley: D = A + 0.786*XA
                D_target = A + 0.786 * xa
                mask = (close - D_target).abs() < atr * 0.5
                sig[mask] = 1

    return sig.shift(1).fillna(0).astype(int)


def space_TV_Auto_Harmonic_Pattern():
    return {
        'pivot_left': ('int', 3, 10),
        'pivot_right': ('int', 3, 10),
        'gartley_xab': ('float', 0.5, 0.75),
        'tolerance': ('float', 0.02, 0.1),
        'atr_len': ('int', 10, 20),
    }


def gen_TV_ABCD_Pattern_Daveatt(df, pivot_len=5, ab_ratio_min=0.382, ab_ratio_max=0.886,
                                  atr_len=14, ema_len=50):
    """BEST ABCD Pattern Strategy Trailing SL+TP [Daveatt] — ABCD Fibonacci pattern.
    Source: https://www.tradingview.com/script/HkuqtyQ7-Daveatt-BEST-ABCD-Pattern-Strategy/
    Concept: AB=CD harmonic — BC retraces AB 38.2-88.6%; CD projects equal to AB.
    """
    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)
    trend = _ema(close, ema_len)
    sh, sl = _swing_highs_lows(high, low, pivot_len, pivot_len)

    sh_prices = high.where(sh).ffill()
    sl_prices = low.where(sl).ffill()
    sh_prev = high.where(sh).shift(1).ffill()
    sl_prev = low.where(sl).shift(1).ffill()

    ab_bull = sh_prices - sl_prev
    bc_bull = sh_prices - sl_prices
    ratio_bull = bc_bull / ab_bull.replace(0, np.nan)
    abcd_bull = (ratio_bull >= ab_ratio_min) & (ratio_bull <= ab_ratio_max)

    ab_bear = sl_prices - sh_prev
    bc_bear = sl_prices - sh_prices
    ratio_bear = bc_bear / ab_bear.replace(0, np.nan)
    abcd_bear = (ratio_bear >= ab_ratio_min) & (ratio_bear <= ab_ratio_max)

    sig = pd.Series(0, index=df.index)
    sig[abcd_bull & (close > trend)] = 1
    sig[abcd_bear & (close < trend)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_ABCD_Pattern_Daveatt():
    return {
        'pivot_len': ('int', 3, 10),
        'ab_ratio_min': ('float', 0.30, 0.50),
        'ab_ratio_max': ('float', 0.75, 0.95),
        'atr_len': ('int', 10, 20),
        'ema_len': ('int', 30, 100),
    }


def gen_TV_ABCD_Harmonic_BullBear(df, pivot_len=5, fib_ratio=0.618, atr_len=14):
    """ABCD Harmonic Pattern Strategy Bull+Bear [TagsTrading].
    Source: https://www.tradingview.com/script/6hnGTVTM-ABCD-Harmonic-Pattern-Strategy/
    Concept: ABCD with 61.8% Fibonacci retracement, both long and short setups.
    """
    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)
    sh, sl = _swing_highs_lows(high, low, pivot_len, pivot_len)

    pivot_h = high.where(sh).ffill()
    pivot_l = low.where(sl).ffill()
    prev_pivot_h = high.where(sh).ffill().shift(1)
    prev_pivot_l = low.where(sl).ffill().shift(1)

    # Bullish: A=swing_low, B=retracement high, C=swing_low2, D=projection
    ab = pivot_h - prev_pivot_l
    cd_target = pivot_l + ab
    near_d_bull = (close - cd_target).abs() < atr * 0.3
    retracement_ok_bull = ((pivot_h - pivot_l) / ab.replace(0, np.nan)).between(fib_ratio - 0.1, fib_ratio + 0.1)

    sig = pd.Series(0, index=df.index)
    sig[near_d_bull & retracement_ok_bull] = 1
    sig[(close < pivot_l - atr * 0.5)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_ABCD_Harmonic_BullBear():
    return {
        'pivot_len': ('int', 3, 10),
        'fib_ratio': ('float', 0.5, 0.786),
        'atr_len': ('int', 10, 20),
    }


def gen_TV_Gartley_222(df, pivot_len=5, atr_len=14):
    """Gartley 222 Strategy Final Full Version [TheWealthyInvestorbyDSAB].
    Source: https://www.tradingview.com/script/ZB3myP9g-Gartley-222-Strategy-Final-Full-Version/
    Concept: XAB=61.8%, AB=61.8%, XABCD=78.6% (fixed Gartley ratios as constants).
    """
    XAB_RATIO = 0.618
    AB_RATIO = 0.618
    XABCD_RATIO = 0.786
    TOL = 0.05

    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)
    sh, sl = _swing_highs_lows(high, low, pivot_len, pivot_len)

    pivot_h = high.where(sh).ffill()
    pivot_l = low.where(sl).ffill()
    ph_prev = high.where(sh).ffill().shift(1)
    pl_prev = low.where(sl).ffill().shift(1)

    XA = pivot_h - pl_prev
    AB = pivot_h - pivot_l
    xab_check = (AB / XA.replace(0, np.nan) - XAB_RATIO).abs() < TOL
    xabcd = (pivot_l / ph_prev.replace(0, np.nan) - (1 - XABCD_RATIO)).abs() < TOL * 2

    sig = pd.Series(0, index=df.index)
    sig[xab_check & xabcd & (close > pivot_l)] = 1
    sig[xab_check & xabcd & (close < pivot_h)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Gartley_222():
    return {
        'pivot_len': ('int', 3, 10),
        'atr_len': ('int', 10, 20),
    }


# ---------------------------------------------------------------------------
# BATCH 036 — CHART PATTERNS (3)
# ---------------------------------------------------------------------------

def gen_TV_Double_Top_Bottom(df, pivot_len=10, tolerance=0.02, atr_len=14, ema_len=50):
    """Double Top/Bottom Ultimate [Trendoscope] — Auto double top/bottom via pivots.
    Source: https://www.tradingview.com/script/a0vTLaS6-Double-Top-Bottom-Ultimate/
    Concept: Two equal swing highs/lows + neckline break = confirmed pattern.
    """
    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)
    trend = _ema(close, ema_len)
    sh, sl = _swing_highs_lows(high, low, pivot_len, pivot_len)

    pivot_h_vals = high.where(sh).dropna()
    pivot_l_vals = low.where(sl).dropna()

    sig = pd.Series(0, index=df.index)
    # Double top: two consecutive swing highs within tolerance
    for i in range(1, len(pivot_h_vals)):
        h1, h2 = pivot_h_vals.iloc[i - 1], pivot_h_vals.iloc[i]
        if abs(h1 - h2) / max(h1, h2) < tolerance:
            idx = pivot_h_vals.index[i]
            pos = df.index.get_loc(idx) if idx in df.index else -1
            if 0 <= pos < len(sig):
                sig.iloc[pos] = -1

    for i in range(1, len(pivot_l_vals)):
        l1, l2 = pivot_l_vals.iloc[i - 1], pivot_l_vals.iloc[i]
        if abs(l1 - l2) / max(l1, l2) < tolerance:
            idx = pivot_l_vals.index[i]
            pos = df.index.get_loc(idx) if idx in df.index else -1
            if 0 <= pos < len(sig):
                sig.iloc[pos] = 1

    return sig.shift(1).fillna(0).astype(int)


def space_TV_Double_Top_Bottom():
    return {
        'pivot_len': ('int', 5, 20),
        'tolerance': ('float', 0.01, 0.05),
        'atr_len': ('int', 10, 20),
        'ema_len': ('int', 30, 100),
    }


def gen_TV_Triangle_Breakout_EMA(df, lookback=20, breakout_atr=0.3, atr_len=14, ema_len=50):
    """Triangle Breakout Strategy with TP/SL + EMA Filter [Reanimate_].
    Source: https://www.tradingview.com/script/8I2NqR8A-Triangle-Breakout-Strategy/
    Concept: Converging highs + lows (triangle) breakout with EMA trend filter.
    """
    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)
    trend = _ema(close, ema_len)

    high_slope = high.rolling(lookback).max() - high.rolling(lookback // 2).max()
    low_slope = low.rolling(lookback).min() - low.rolling(lookback // 2).min()
    # Converging: high slope descending, low slope ascending
    converging = (high_slope < 0) & (low_slope > 0)

    recent_high = high.rolling(lookback).max()
    recent_low = low.rolling(lookback).min()

    sig = pd.Series(0, index=df.index)
    sig[converging & (close > recent_high - atr * breakout_atr) & (close > trend)] = 1
    sig[converging & (close < recent_low + atr * breakout_atr) & (close < trend)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Triangle_Breakout_EMA():
    return {
        'lookback': ('int', 10, 40),
        'breakout_atr': ('float', 0.1, 0.8),
        'atr_len': ('int', 10, 20),
        'ema_len': ('int', 30, 100),
    }


def gen_TV_PriceAction_Pattern_Breakout(df, pivot_len=5, channel_len=20, atr_len=14):
    """Price Action Pattern Breakout: Wedge+Triangle+Channel [Navixa].
    Source: https://www.tradingview.com/script/aXV6ElL1-Price-Action-Pattern-Breakout/
    Concept: Channel breakout with multiple pattern types via swing pivot regression.
    """
    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)

    ch_high = high.rolling(channel_len).max()
    ch_low = low.rolling(channel_len).min()
    ch_mid = (ch_high + ch_low) / 2

    # Wedge: channel narrowing
    ch_width = ch_high - ch_low
    narrowing = ch_width < ch_width.shift(channel_len // 2)

    sig = pd.Series(0, index=df.index)
    sig[narrowing & (close > ch_high.shift(1))] = 1
    sig[narrowing & (close < ch_low.shift(1))] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_PriceAction_Pattern_Breakout():
    return {
        'pivot_len': ('int', 3, 10),
        'channel_len': ('int', 10, 40),
        'atr_len': ('int', 10, 20),
    }


# ---------------------------------------------------------------------------
# BATCH 036 — ICHIMOKU (3)
# ---------------------------------------------------------------------------

def gen_TV_Ichimoku_HPotter(df, tenkan=9, kijun=26, senkou_b_len=52):
    """Ichimoku Backtest [HPotter] — Classic TK cross with cloud confirmation.
    Source: https://www.tradingview.com/script/tCqtJVeB-Ichimoku-Backtest/
    Concept: Long when Tenkan > Kijun AND price > cloud (Kumo).
    """
    close = df['close']
    t, k, sa, sb, _ = _ichimoku(df, tenkan, kijun, senkou_b_len)

    cloud_top = pd.concat([sa, sb], axis=1).max(axis=1)
    cloud_bot = pd.concat([sa, sb], axis=1).min(axis=1)

    tk_bull = (t > k) & (t.shift(1) <= k.shift(1))
    tk_bear = (t < k) & (t.shift(1) >= k.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[tk_bull & (close > cloud_top)] = 1
    sig[tk_bear & (close < cloud_bot)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Ichimoku_HPotter():
    return {
        'tenkan': ('int', 7, 15),
        'kijun': ('int', 20, 35),
        'senkou_b_len': ('int', 40, 65),
    }


def gen_TV_Ichimoku_TPSL_Cloud(df, tenkan=9, kijun=26, senkou_b_len=52, atr_mult=2.0):
    """Ichimoku Backtester with TP/SL + Cloud Confirmation [gregh2].
    Source: https://www.tradingview.com/script/fsYSuQGL-Ichimoku-Backtester-with-TP-SL/
    Concept: TK cross + cloud confirmation + ATR-based TP/SL management.
    """
    close = df['close']
    atr = _atr(df, 14)
    t, k, sa, sb, chikou = _ichimoku(df, tenkan, kijun, senkou_b_len)

    cloud_top = pd.concat([sa, sb], axis=1).max(axis=1)
    cloud_bot = pd.concat([sa, sb], axis=1).min(axis=1)

    bull_setup = (t > k) & (close > cloud_top) & (close > t) & (close > k)
    bear_setup = (t < k) & (close < cloud_bot) & (close < t) & (close < k)

    sig = pd.Series(0, index=df.index)
    sig[bull_setup & ~bull_setup.shift(1).astype(bool).fillna(False)] = 1
    sig[bear_setup & ~bear_setup.shift(1).astype(bool).fillna(False)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Ichimoku_TPSL_Cloud():
    return {
        'tenkan': ('int', 7, 15),
        'kijun': ('int', 20, 35),
        'senkou_b_len': ('int', 40, 65),
        'atr_mult': ('float', 1.0, 3.0),
    }


def gen_TV_Ichimoku_Long_Only(df, tenkan=9, kijun=26, senkou_b_len=52, chikou_lookback=26):
    """Ichimoku Cloud Strategy Long Only [Bitduke].
    Source: https://www.tradingview.com/script/xJZp0Pm1-Ichimoku-Cloud-Strategy-Long-Only/
    Concept: Long only — all Ichimoku signals bullish (TK cross + above cloud + chikou clear).
    """
    close = df['close']
    t, k, sa, sb, chikou = _ichimoku(df, tenkan, kijun, senkou_b_len)

    cloud_top = pd.concat([sa, sb], axis=1).max(axis=1)
    chikou_clear = close > close.shift(chikou_lookback)

    all_bull = (t > k) & (close > cloud_top) & chikou_clear

    sig = pd.Series(0, index=df.index)
    sig[all_bull & ~all_bull.shift(1).astype(bool).fillna(False)] = 1
    sig[~all_bull & all_bull.shift(1).astype(bool).fillna(False)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Ichimoku_Long_Only():
    return {
        'tenkan': ('int', 7, 15),
        'kijun': ('int', 20, 35),
        'senkou_b_len': ('int', 40, 65),
        'chikou_lookback': ('int', 20, 35),
    }


# ---------------------------------------------------------------------------
# BATCH 036 — DIVERGENCE (2)
# ---------------------------------------------------------------------------

def gen_TV_MACD_Divergence_MTF_EMA(df, macd_fast=12, macd_slow=26, macd_sig=9,
                                     htf_minutes=240, ema_len=50, div_len=14):
    """MACD Divergence + MTF EMA Reversal [daviddtech].
    Source: https://www.tradingview.com/script/mFSouWh2-MACD-Divergence-MTF-EMA-Reversal/
    Concept: MACD divergence (price vs MACD histogram) + HTF EMA trend filter.
    """
    close, low, high = df['close'], df['low'], df['high']
    m, ms = _macd(close, macd_fast, macd_slow, macd_sig)
    hist = m - ms
    trend = _ema(close, ema_len)

    htf = _resample_htf(df, htf_minutes)
    htf_ema = _ema(htf['close'], ema_len).reindex(df.index, method='ffill')

    # Bullish divergence: price lower low, MACD histogram higher low
    price_lower_low = low < low.shift(div_len)
    hist_higher_low = hist > hist.shift(div_len)
    bull_div = price_lower_low & hist_higher_low

    # Bearish divergence: price higher high, MACD histogram lower high
    price_higher_high = high > high.shift(div_len)
    hist_lower_high = hist < hist.shift(div_len)
    bear_div = price_higher_high & hist_lower_high

    sig = pd.Series(0, index=df.index)
    sig[bull_div & (close > htf_ema)] = 1
    sig[bear_div & (close < htf_ema)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_MACD_Divergence_MTF_EMA():
    return {
        'macd_fast': ('int', 8, 16),
        'macd_slow': ('int', 20, 32),
        'macd_sig': ('int', 6, 12),
        'htf_minutes': ('int', 60, 480),
        'ema_len': ('int', 30, 100),
        'div_len': ('int', 8, 21),
    }


def gen_TV_RSI_Divergence_Alifer(df, rsi_len=14, div_len=14, ob=70, os=30, ema_len=50):
    """RSI Divergence Strategy [AliferCrypto].
    Source: https://www.tradingview.com/script/Spg5RGk1-RSI-Divergence-Strategy-AliferCrypto/
    Concept: Classic RSI divergence — price vs RSI diverge at OB/OS extremes.
    """
    close, high, low = df['close'], df['high'], df['low']
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, ema_len)

    # Bullish: price new low but RSI higher low (at OS)
    price_ll = low < low.shift(div_len)
    rsi_hl = rsi > rsi.shift(div_len)
    bull_div = price_ll & rsi_hl & (rsi < os + 10)

    # Bearish: price new high but RSI lower high (at OB)
    price_hh = high > high.shift(div_len)
    rsi_lh = rsi < rsi.shift(div_len)
    bear_div = price_hh & rsi_lh & (rsi > ob - 10)

    sig = pd.Series(0, index=df.index)
    sig[bull_div] = 1
    sig[bear_div] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_RSI_Divergence_Alifer():
    return {
        'rsi_len': ('int', 10, 21),
        'div_len': ('int', 8, 21),
        'ob': ('float', 65, 80),
        'os': ('float', 20, 35),
        'ema_len': ('int', 30, 100),
    }


# ---------------------------------------------------------------------------
# BATCH 036 — ADX / DMI (3)
# ---------------------------------------------------------------------------

def gen_TV_ADX_DMI_Trend(df, adx_len=14, adx_thresh=25, ema_len=50):
    """ADX DMI Trend Strategy [millerrh] — ADX gate + DI cross entry.
    Source: https://www.tradingview.com/script/4aQD9gqq-ADX-DMI-Trend-Strategy/
    Concept: Enter long when DI+ > DI- AND ADX > threshold (trending market gate).
    """
    close = df['close']
    adx, di_plus, di_minus = _adx(df, adx_len)
    trend = _ema(close, ema_len)

    bull_cross = (di_plus > di_minus) & (di_plus.shift(1) <= di_minus.shift(1))
    bear_cross = (di_plus < di_minus) & (di_plus.shift(1) >= di_minus.shift(1))
    strong_trend = adx > adx_thresh

    sig = pd.Series(0, index=df.index)
    sig[bull_cross & strong_trend] = 1
    sig[bear_cross & strong_trend] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_ADX_DMI_Trend():
    return {
        'adx_len': ('int', 10, 21),
        'adx_thresh': ('float', 20, 35),
        'ema_len': ('int', 30, 100),
    }


def gen_TV_DMI_Toolbox(df, dmi_len=14, adx_thresh=20, di_diff_thresh=5, ema_len=50):
    """DMI Toolbox Strategy [Chart0bserver] — Multi-permutation DMI Optuna sweep.
    Source: https://www.tradingview.com/script/kgsU4SHu-DMI-Toolbox-Strategy/
    Concept: Expose multiple DMI params to Optuna for sweep; DI diff + ADX gate.
    """
    close = df['close']
    adx, di_plus, di_minus = _adx(df, dmi_len)
    trend = _ema(close, ema_len)

    di_diff = di_plus - di_minus
    strong = adx > adx_thresh

    sig = pd.Series(0, index=df.index)
    sig[(di_diff > di_diff_thresh) & strong & (close > trend)] = 1
    sig[(di_diff < -di_diff_thresh) & strong & (close < trend)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_DMI_Toolbox():
    return {
        'dmi_len': ('int', 8, 25),
        'adx_thresh': ('float', 15, 35),
        'di_diff_thresh': ('float', 2, 15),
        'ema_len': ('int', 30, 100),
    }


def gen_TV_DMI_ADX_Astropark(df, dmi_len=14, adx_thresh=25, rsi_len=14, ema_len=50):
    """DMI/ADX Strategy [astropark] — DMI with RSI momentum confirmation.
    Source: https://www.tradingview.com/script/PEGQed2H-astropark-DMI-ADX-strategy/
    Concept: DI cross + ADX trending + RSI momentum confirmation.
    """
    close = df['close']
    adx, di_plus, di_minus = _adx(df, dmi_len)
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, ema_len)

    sig = pd.Series(0, index=df.index)
    sig[(di_plus > di_minus) & (adx > adx_thresh) & (rsi > 50) & (close > trend)] = 1
    sig[(di_minus > di_plus) & (adx > adx_thresh) & (rsi < 50) & (close < trend)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_DMI_ADX_Astropark():
    return {
        'dmi_len': ('int', 8, 21),
        'adx_thresh': ('float', 18, 35),
        'rsi_len': ('int', 10, 21),
        'ema_len': ('int', 30, 100),
    }


# ---------------------------------------------------------------------------
# BATCH 036 — BREAKOUT / S&R (5)
# ---------------------------------------------------------------------------

def gen_TV_Donchian_Channel(df, don_len=20, atr_len=14, ema_len=50):
    """Donchian Channel Strategy [RafaelPiccolo] — Classic Donchian breakout.
    Source: https://www.tradingview.com/script/ZZ0T9Wc7-Donchian-Channel-Strategy/
    Concept: Long on close > upper Donchian, short on close < lower Donchian.
    """
    close, high, low = df['close'], df['high'], df['low']
    don_high = high.rolling(don_len).max()
    don_low = low.rolling(don_len).min()
    trend = _ema(close, ema_len)

    sig = pd.Series(0, index=df.index)
    sig[(close > don_high.shift(1)) & (close > trend)] = 1
    sig[(close < don_low.shift(1)) & (close < trend)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Donchian_Channel():
    return {
        'don_len': ('int', 10, 50),
        'atr_len': ('int', 10, 20),
        'ema_len': ('int', 30, 100),
    }


def gen_TV_Camarilla_Pivots(df, pivot_h8_mult=1.1/6, pivot_l8_mult=1.1/6, atr_len=14):
    """Camarilla Pivots Signal + Backtest [tso_trade] — Camarilla level bounces.
    Source: https://www.tradingview.com/script/93feo9Xv-Camarilla-Pivots-Signal/
    Concept: Camarilla H3/L3 for mean-reversion, H4/L4 for breakout.
    """
    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)

    prev_high = high.resample('D').max().shift(1).reindex(df.index, method='ffill')
    prev_low = low.resample('D').min().shift(1).reindex(df.index, method='ffill')
    prev_close = close.resample('D').last().shift(1).reindex(df.index, method='ffill')
    rng = prev_high - prev_low

    h3 = prev_close + rng * (1.1 / 6)
    l3 = prev_close - rng * (1.1 / 6)
    h4 = prev_close + rng * (1.1 / 4)
    l4 = prev_close - rng * (1.1 / 4)

    sig = pd.Series(0, index=df.index)
    # Mean-reversion at H3/L3
    sig[(close < l3) & (close > l3 - atr * 0.3)] = 1
    sig[(close > h3) & (close < h3 + atr * 0.3)] = -1
    # Breakout at H4/L4
    sig[close > h4] = 1
    sig[close < l4] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Camarilla_Pivots():
    return {
        'pivot_h8_mult': ('float', 0.15, 0.25),
        'pivot_l8_mult': ('float', 0.15, 0.25),
        'atr_len': ('int', 10, 20),
    }


def gen_TV_SR_Trendlines(df, pivot_len=10, sr_atr=0.3, atr_len=14, ema_len=50):
    """S/R + Trend Lines Backtest [tso_trade] — Dynamic S/R from swing pivots.
    Source: https://www.tradingview.com/script/MBNjzGn6-Support-and-Resistance-with-Trend-Lines/
    Concept: Bounce off dynamic S/R levels defined by recent swing highs/lows.
    """
    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)
    trend = _ema(close, ema_len)
    sh, sl = _swing_highs_lows(high, low, pivot_len, pivot_len)

    resist = high.where(sh).ffill()
    support = low.where(sl).ffill()

    near_support = (close - support).abs() < atr * sr_atr
    near_resist = (close - resist).abs() < atr * sr_atr

    sig = pd.Series(0, index=df.index)
    sig[near_support & (close > trend)] = 1
    sig[near_resist & (close < trend)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_SR_Trendlines():
    return {
        'pivot_len': ('int', 5, 20),
        'sr_atr': ('float', 0.2, 0.8),
        'atr_len': ('int', 10, 20),
        'ema_len': ('int', 30, 100),
    }


def gen_TV_ORB_SR_Backtest(df, orb_bars=4, atr_len=14, ema_len=50):
    """ORB S&R Strategy with Backtest [tso_trade] — ORB + pivot S/R confluence.
    Source: https://www.tradingview.com/script/MJCRcQ6g-Opening-Range-Breakout-S-R-Strategy/
    Concept: ORB breakout confirmed by position relative to daily pivot.
    """
    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)
    trend = _ema(close, ema_len)

    prev_h = high.resample('D').max().shift(1).reindex(df.index, method='ffill')
    prev_l = low.resample('D').min().shift(1).reindex(df.index, method='ffill')
    prev_c = close.resample('D').last().shift(1).reindex(df.index, method='ffill')
    pivot = (prev_h + prev_l + prev_c) / 3

    orb_high = pd.Series(np.nan, index=df.index)
    orb_low = pd.Series(np.nan, index=df.index)
    for d in df.index.normalize().unique():
        mask = df.index.normalize() == d
        day_idx = df.index[mask]
        if len(day_idx) >= orb_bars:
            orb_high[mask] = high[mask].iloc[:orb_bars].max()
            orb_low[mask] = low[mask].iloc[:orb_bars].min()
    orb_high = orb_high.ffill()
    orb_low = orb_low.ffill()

    sig = pd.Series(0, index=df.index)
    sig[(close > orb_high.shift(1)) & (close > pivot) & (close > trend)] = 1
    sig[(close < orb_low.shift(1)) & (close < pivot) & (close < trend)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_ORB_SR_Backtest():
    return {
        'orb_bars': ('int', 2, 8),
        'atr_len': ('int', 10, 20),
        'ema_len': ('int', 30, 100),
    }


def gen_TV_Cyatophilum_Intraday(df, range_bars=12, breakout_atr=0.5, atr_len=14, ema_len=50):
    """Cyatophilum Intraday Breakouts [cyatophilum] — Intraday range + breakout.
    Source: https://www.tradingview.com/script/hBAG8hFU-Cyatophilum-Intraday-Breakouts/
    Concept: Build intraday range for N bars, then trade range breakout.
    """
    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)
    trend = _ema(close, ema_len)

    rng_high = high.rolling(range_bars).max()
    rng_low = low.rolling(range_bars).min()

    sig = pd.Series(0, index=df.index)
    sig[(close > rng_high.shift(1) + atr * breakout_atr * 0.1) & (close > trend)] = 1
    sig[(close < rng_low.shift(1) - atr * breakout_atr * 0.1) & (close < trend)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Cyatophilum_Intraday():
    return {
        'range_bars': ('int', 6, 30),
        'breakout_atr': ('float', 0.2, 1.5),
        'atr_len': ('int', 10, 20),
        'ema_len': ('int', 30, 100),
    }


# ---------------------------------------------------------------------------
# BATCH 036 — FLAG / ELLIOTT / ELDER / SSL+WT (4)
# ---------------------------------------------------------------------------

def gen_TV_TrendGuard_Flag_Finder(df, trend_len=50, flag_bars=10, flag_atr=0.5, atr_len=14):
    """TrendGuard Flag Finder Strategy [PresentTrading] — Bull/bear flag breakout.
    Source: https://www.tradingview.com/script/qDPWh3KO-TrendGuard-Flag-Finder-Strategy/
    Concept: Strong trend leg (pole) + tight consolidation (flag) + breakout entry.
    """
    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)
    trend = _ema(close, trend_len)

    # Pole: strong directional move
    pole_up = close.pct_change(flag_bars) > atr.shift(flag_bars) / close.shift(flag_bars) * 2
    pole_dn = close.pct_change(flag_bars) < -atr.shift(flag_bars) / close.shift(flag_bars) * 2

    # Flag: ATR compression in recent bars
    recent_atr = atr
    avg_atr = atr.rolling(flag_bars * 3).mean()
    flag_forming = recent_atr < avg_atr * flag_atr

    rng_high = high.rolling(flag_bars).max()
    rng_low = low.rolling(flag_bars).min()

    sig = pd.Series(0, index=df.index)
    sig[pole_up.shift(flag_bars) & flag_forming & (close > rng_high.shift(1))] = 1
    sig[pole_dn.shift(flag_bars) & flag_forming & (close < rng_low.shift(1))] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_TrendGuard_Flag_Finder():
    return {
        'trend_len': ('int', 30, 100),
        'flag_bars': ('int', 5, 20),
        'flag_atr': ('float', 0.3, 0.9),
        'atr_len': ('int', 10, 20),
    }


def gen_TV_Elliott_Wave_Fib_Scalper(df, swing_len=5, fib_382=0.382, fib_618=0.618,
                                      atr_len=14, ema_len=50):
    """Elliott Wave Auto + Fibonacci Targets Scalper [shravanreddy0808].
    Source: https://www.tradingview.com/script/2zSvEJZ8-Elliott-Wave-Auto-Fib-Targets/
    Concept: Wave 2/4 retracement zones (38.2-61.8% Fib) as entry for wave 3/5.
    """
    close, high, low = df['close'], df['high'], df['low']
    atr = _atr(df, atr_len)
    trend = _ema(close, ema_len)
    sh, sl = _swing_highs_lows(high, low, swing_len, swing_len)

    wave_top = high.where(sh).ffill()
    wave_bot = low.where(sl).ffill()

    wave_range = wave_top - wave_bot
    fib_low = wave_bot + wave_range * fib_382
    fib_high = wave_bot + wave_range * fib_618

    in_retracement_bull = (close >= fib_low) & (close <= fib_high) & (close > trend)
    in_retracement_bear = (close <= wave_top - wave_range * fib_382) & \
                          (close >= wave_top - wave_range * fib_618) & (close < trend)

    sig = pd.Series(0, index=df.index)
    sig[in_retracement_bull & (close > close.shift(1))] = 1
    sig[in_retracement_bear & (close < close.shift(1))] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Elliott_Wave_Fib_Scalper():
    return {
        'swing_len': ('int', 3, 10),
        'fib_382': ('float', 0.30, 0.45),
        'fib_618': ('float', 0.55, 0.70),
        'atr_len': ('int', 10, 20),
        'ema_len': ('int', 30, 100),
    }


def gen_TV_Elder_Ray_Bear_Power(df, ema_len=13, bear_thresh=0.0, trend_len=50):
    """Elder Ray Bear Power Strategy Backtest [HPotter].
    Source: https://www.tradingview.com/script/7GdzAJRc-Elder-Ray-Bear-Power-Strategy/
    Concept: Bear Power = Low - EMA(13). Negative = bears control; cross zero = signal.
    """
    close, low, high = df['close'], df['low'], df['high']
    ema13 = _ema(close, ema_len)
    bear_power = low - ema13
    trend = _ema(close, trend_len)

    cross_up = (bear_power > bear_thresh) & (bear_power.shift(1) <= bear_thresh)
    cross_dn = (bear_power < bear_thresh) & (bear_power.shift(1) >= bear_thresh)

    sig = pd.Series(0, index=df.index)
    sig[cross_up & (close > trend)] = 1
    sig[cross_dn & (close < trend)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Elder_Ray_Bear_Power():
    return {
        'ema_len': ('int', 8, 21),
        'bear_thresh': ('float', -1.0, 1.0),
        'trend_len': ('int', 30, 100),
    }


def gen_TV_Elder_Ray_Bull_Power(df, ema_len=13, bull_thresh=0.0, trend_len=50):
    """Elder Ray Bull Power Strategy Backtest [HPotter].
    Source: https://www.tradingview.com/script/dfKRefJa-Elder-Ray-Bull-Power-Strategy/
    Concept: Bull Power = High - EMA(13). Positive = bulls control; cross zero = signal.
    """
    close, high = df['close'], df['high']
    ema13 = _ema(close, ema_len)
    bull_power = high - ema13
    trend = _ema(close, trend_len)

    cross_up = (bull_power > bull_thresh) & (bull_power.shift(1) <= bull_thresh)
    cross_dn = (bull_power < bull_thresh) & (bull_power.shift(1) >= bull_thresh)

    sig = pd.Series(0, index=df.index)
    sig[cross_up & (close > trend)] = 1
    sig[cross_dn & (close < trend)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_Elder_Ray_Bull_Power():
    return {
        'ema_len': ('int', 8, 21),
        'bull_thresh': ('float', -1.0, 1.0),
        'trend_len': ('int', 30, 100),
    }


def gen_TV_SSL_WaveTrend(df, ssl_len=10, wt_ch=10, wt_avg=21, ob=53, os=-53):
    """SSL + Wave Trend Strategy [kevinmck100] — SSL trend + WaveTrend oscillator.
    Source: https://www.tradingview.com/script/J0urw1QI-SSL-Wave-Trend-Strategy/
    Concept: SSL channel for trend direction, WaveTrend OB/OS for entry timing.
    """
    close = df['close']
    ssl_up, ssl_dn = _ssl_channel(df, ssl_len)
    wt1, wt2 = _wavetrend(df, wt_ch, wt_avg)

    ssl_bull = close > ssl_up
    ssl_bear = close < ssl_dn

    wt_cross_up = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    wt_cross_dn = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[ssl_bull & wt_cross_up & (wt1 < ob)] = 1
    sig[ssl_bear & wt_cross_dn & (wt1 > os)] = -1
    return sig.shift(1).fillna(0).astype(int)


def space_TV_SSL_WaveTrend():
    return {
        'ssl_len': ('int', 5, 20),
        'wt_ch': ('int', 6, 16),
        'wt_avg': ('int', 10, 30),
        'ob': ('float', 40, 70),
        'os': ('float', -70, -40),
    }


# ---------------------------------------------------------------------------
# STRATEGY EXPORT
# ---------------------------------------------------------------------------

STRATEGY_EXPORT = {
    # --- Batch 035: SMC/ICT ---
    'TV_SMC_OB_FVG_DOE': {
        'gen': gen_TV_SMC_OB_FVG_DOE,
        'space': space_TV_SMC_OB_FVG_DOE,
    },
    'TV_ES_MTF_SMC_Entry': {
        'gen': gen_TV_ES_MTF_SMC_Entry,
        'space': space_TV_ES_MTF_SMC_Entry,
    },
    'TV_SMC_Trap_FVG': {
        'gen': gen_TV_SMC_Trap_FVG,
        'space': space_TV_SMC_Trap_FVG,
    },
    'TV_SMC_Fractal_v3': {
        'gen': gen_TV_SMC_Fractal_v3,
        'space': space_TV_SMC_Fractal_v3,
    },
    'TV_LuxAlgo_SMC_Ultimate': {
        'gen': gen_TV_LuxAlgo_SMC_Ultimate,
        'space': space_TV_LuxAlgo_SMC_Ultimate,
    },
    'TV_ICT_Killzones_Pivots': {
        'gen': gen_TV_ICT_Killzones_Pivots,
        'space': space_TV_ICT_Killzones_Pivots,
    },
    'TV_Liquidity_Sweep_Filter': {
        'gen': gen_TV_Liquidity_Sweep_Filter,
        'space': space_TV_Liquidity_Sweep_Filter,
    },
    'TV_SMC_Liquidity_Grab_Pro': {
        'gen': gen_TV_SMC_Liquidity_Grab_Pro,
        'space': space_TV_SMC_Liquidity_Grab_Pro,
    },
    'TV_Casper_SMC_ORB_Retest': {
        'gen': gen_TV_Casper_SMC_ORB_Retest,
        'space': space_TV_Casper_SMC_ORB_Retest,
    },
    # --- Batch 035: Quantitative/Kelly ---
    'TV_Bayesian_Kelly': {
        'gen': gen_TV_Bayesian_Kelly,
        'space': space_TV_Bayesian_Kelly,
    },
    'TV_Kelly_Dynamic_Sizing': {
        'gen': gen_TV_Kelly_Dynamic_Sizing,
        'space': space_TV_Kelly_Dynamic_Sizing,
    },
    'TV_Dual_Momentum': {
        'gen': gen_TV_Dual_Momentum,
        'space': space_TV_Dual_Momentum,
    },
    # --- Batch 035: Momentum ---
    'TV_Momentum_Strategy_REV': {
        'gen': gen_TV_Momentum_Strategy_REV,
        'space': space_TV_Momentum_Strategy_REV,
    },
    'TV_Momentum_Long_REV': {
        'gen': gen_TV_Momentum_Long_REV,
        'space': space_TV_Momentum_Long_REV,
    },
    # --- Batch 035: VIX Regime / Session / Wyckoff ---
    'TV_VIX_Regime': {
        'gen': gen_TV_VIX_Regime,
        'space': space_TV_VIX_Regime,
    },
    'TV_Long_ORB_Pivot': {
        'gen': gen_TV_Long_ORB_Pivot,
        'space': space_TV_Long_ORB_Pivot,
    },
    'TV_Gold_Asia_Session': {
        'gen': gen_TV_Gold_Asia_Session,
        'space': space_TV_Gold_Asia_Session,
    },
    'TV_Pivot_Reversal_Backtest': {
        'gen': gen_TV_Pivot_Reversal_Backtest,
        'space': space_TV_Pivot_Reversal_Backtest,
    },
    'TV_Wyckoff_Range': {
        'gen': gen_TV_Wyckoff_Range,
        'space': space_TV_Wyckoff_Range,
    },
    # --- Batch 035: Adaptive MA / Gann ---
    'TV_WaveTrend_Plus': {
        'gen': gen_TV_WaveTrend_Plus,
        'space': space_TV_WaveTrend_Plus,
    },
    'TV_KAMA_Adaptive': {
        'gen': gen_TV_KAMA_Adaptive,
        'space': space_TV_KAMA_Adaptive,
    },
    'TV_Gann_Trend_Oscillator': {
        'gen': gen_TV_Gann_Trend_Oscillator,
        'space': space_TV_Gann_Trend_Oscillator,
    },
    'TV_Combo_123_Gann': {
        'gen': gen_TV_Combo_123_Gann,
        'space': space_TV_Combo_123_Gann,
    },
    'TV_Gann_Swing_MultiLayer': {
        'gen': gen_TV_Gann_Swing_MultiLayer,
        'space': space_TV_Gann_Swing_MultiLayer,
    },
    # --- Batch 036: Harmonic Patterns ---
    'TV_Auto_Harmonic_Pattern': {
        'gen': gen_TV_Auto_Harmonic_Pattern,
        'space': space_TV_Auto_Harmonic_Pattern,
    },
    'TV_ABCD_Pattern_Daveatt': {
        'gen': gen_TV_ABCD_Pattern_Daveatt,
        'space': space_TV_ABCD_Pattern_Daveatt,
    },
    'TV_ABCD_Harmonic_BullBear': {
        'gen': gen_TV_ABCD_Harmonic_BullBear,
        'space': space_TV_ABCD_Harmonic_BullBear,
    },
    'TV_Gartley_222': {
        'gen': gen_TV_Gartley_222,
        'space': space_TV_Gartley_222,
    },
    # --- Batch 036: Chart Patterns ---
    'TV_Double_Top_Bottom': {
        'gen': gen_TV_Double_Top_Bottom,
        'space': space_TV_Double_Top_Bottom,
    },
    'TV_Triangle_Breakout_EMA': {
        'gen': gen_TV_Triangle_Breakout_EMA,
        'space': space_TV_Triangle_Breakout_EMA,
    },
    'TV_PriceAction_Pattern_Breakout': {
        'gen': gen_TV_PriceAction_Pattern_Breakout,
        'space': space_TV_PriceAction_Pattern_Breakout,
    },
    # --- Batch 036: Ichimoku ---
    'TV_Ichimoku_HPotter': {
        'gen': gen_TV_Ichimoku_HPotter,
        'space': space_TV_Ichimoku_HPotter,
    },
    'TV_Ichimoku_TPSL_Cloud': {
        'gen': gen_TV_Ichimoku_TPSL_Cloud,
        'space': space_TV_Ichimoku_TPSL_Cloud,
    },
    'TV_Ichimoku_Long_Only': {
        'gen': gen_TV_Ichimoku_Long_Only,
        'space': space_TV_Ichimoku_Long_Only,
    },
    # --- Batch 036: Divergence ---
    'TV_MACD_Divergence_MTF_EMA': {
        'gen': gen_TV_MACD_Divergence_MTF_EMA,
        'space': space_TV_MACD_Divergence_MTF_EMA,
    },
    'TV_RSI_Divergence_Alifer': {
        'gen': gen_TV_RSI_Divergence_Alifer,
        'space': space_TV_RSI_Divergence_Alifer,
    },
    # --- Batch 036: ADX/DMI ---
    'TV_ADX_DMI_Trend': {
        'gen': gen_TV_ADX_DMI_Trend,
        'space': space_TV_ADX_DMI_Trend,
    },
    'TV_DMI_Toolbox': {
        'gen': gen_TV_DMI_Toolbox,
        'space': space_TV_DMI_Toolbox,
    },
    'TV_DMI_ADX_Astropark': {
        'gen': gen_TV_DMI_ADX_Astropark,
        'space': space_TV_DMI_ADX_Astropark,
    },
    # --- Batch 036: Breakout/S&R ---
    'TV_Donchian_Channel': {
        'gen': gen_TV_Donchian_Channel,
        'space': space_TV_Donchian_Channel,
    },
    'TV_Camarilla_Pivots': {
        'gen': gen_TV_Camarilla_Pivots,
        'space': space_TV_Camarilla_Pivots,
    },
    'TV_SR_Trendlines': {
        'gen': gen_TV_SR_Trendlines,
        'space': space_TV_SR_Trendlines,
    },
    'TV_ORB_SR_Backtest': {
        'gen': gen_TV_ORB_SR_Backtest,
        'space': space_TV_ORB_SR_Backtest,
    },
    'TV_Cyatophilum_Intraday': {
        'gen': gen_TV_Cyatophilum_Intraday,
        'space': space_TV_Cyatophilum_Intraday,
    },
    # --- Batch 036: Flag/Elliott/Elder/SSL ---
    'TV_TrendGuard_Flag_Finder': {
        'gen': gen_TV_TrendGuard_Flag_Finder,
        'space': space_TV_TrendGuard_Flag_Finder,
    },
    'TV_Elliott_Wave_Fib_Scalper': {
        'gen': gen_TV_Elliott_Wave_Fib_Scalper,
        'space': space_TV_Elliott_Wave_Fib_Scalper,
    },
    'TV_Elder_Ray_Bear_Power': {
        'gen': gen_TV_Elder_Ray_Bear_Power,
        'space': space_TV_Elder_Ray_Bear_Power,
    },
    'TV_Elder_Ray_Bull_Power': {
        'gen': gen_TV_Elder_Ray_Bull_Power,
        'space': space_TV_Elder_Ray_Bull_Power,
    },
    'TV_SSL_WaveTrend': {
        'gen': gen_TV_SSL_WaveTrend,
        'space': space_TV_SSL_WaveTrend,
    },
}
