"""TV2 Batch 3594 — Time Cycles + Seasonality + Ehlers Cycles + Composite Momentum (Batch 027)
Source: HUNTER2 Batch 027
Date: 2026-04-17
Strategies (25 total):
  Seasonality + Calendar Cycles (5): TV_TrueSeasonalPattern, TV_SeasonalityPresidentialCycle,
    TV_IntradaySeasonality, TV_SeasonalityForecast, TV_EnhancedSeasonalityBacktest
  Intraday Time Patterns (5): TV_MonthlyPerformanceStrategy, TV_SeasonalityChartingCycles,
    TV_SeasonalityMonthHighlight, TV_SeasonalTendency, TV_SeasonalityCustomInterval
  MESA + Dominant Cycles (5): TV_MESAAdaptiveEhlersFlow, TV_EhlersMESAAdaptiveMA,
    TV_DominantCycleLibrary, TV_EhlersAdaptiveCyberCycle, TV_EhlersSimpleCycleIndicator
  Ehlers Fisher Transform Variants (5): TV_InverseFisherAdaptiveStoch, TV_InverseFisherStoch,
    TV_AdaptiveFisherizedCMO, TV_AdaptiveFisherizedROC, TV_FisherTransformEhlersBacktest
  Composite Momentum Scoring (5): TV_MLEnhancedCompositeSignal, TV_CompositeBuySellScore,
    TV_StochRSI_RSI_MACD_Signals, TV_CombinedStrategyRSIADXSMA, TV_PeriodHighlighterPro
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
    high = df['high']
    low = df['low']
    close = df['close']
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = _atr(df, n)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) * 100
    return dx.ewm(alpha=1/n, adjust=False).mean(), plus_di, minus_di


def _cmo(series, n=14):
    diff = series.diff()
    up = diff.clip(lower=0).rolling(n).sum()
    dn = (-diff).clip(lower=0).rolling(n).sum()
    return 100 * (up - dn) / (up + dn + 1e-9)


def _roc(series, n=10):
    return (series / series.shift(n) - 1) * 100


def _fisher_transform(series, n=10):
    lo = series.rolling(n).min()
    hi = series.rolling(n).max()
    val = 2 * (series - lo) / (hi - lo + 1e-9) - 1
    val = val.clip(-0.999, 0.999)
    fisher = 0.5 * np.log((1 + val) / (1 - val))
    return fisher


def _ift(x):
    e2 = np.exp(2 * x)
    return (e2 - 1) / (e2 + 1)


def _dominant_cycle(close, min_period=10, max_period=48):
    """Simple dominant cycle estimate via autocorrelation peak."""
    n = len(close)
    dc = pd.Series(20.0, index=close.index)
    for i in range(max_period, n):
        seg = close.iloc[i - max_period:i].values
        seg = seg - seg.mean()
        best_period = max_period
        best_corr = -1.0
        for p in range(min_period, max_period + 1):
            if i - p < 0:
                continue
            seg_shifted = np.roll(seg, p)
            corr_val = float(np.corrcoef(seg, seg_shifted)[0, 1])
            if corr_val > best_corr:
                best_corr = corr_val
                best_period = p
        dc.iloc[i] = best_period
    return dc


def _mama_fama(close, fast_limit=0.5, slow_limit=0.05):
    """MESA Adaptive Moving Average (MAMA/FAMA) approximation."""
    n = len(close)
    mama = close.copy().astype(float)
    fama = close.copy().astype(float)
    smooth = pd.Series(0.0, index=close.index)
    detrender = pd.Series(0.0, index=close.index)
    period = pd.Series(0.0, index=close.index)
    phase = pd.Series(0.0, index=close.index)

    for i in range(6, n):
        s = (4 * close.iloc[i] + 3 * close.iloc[i-1] + 2 * close.iloc[i-2] + close.iloc[i-3]) / 10
        smooth.iloc[i] = s

        d = (0.0962 * smooth.iloc[i] + 0.5769 * smooth.iloc[i-2]
             - 0.5769 * smooth.iloc[i-4] - 0.0962 * smooth.iloc[i-6]) * (0.075 * period.iloc[i-1] + 0.54)
        detrender.iloc[i] = d

        q1 = (0.0962 * detrender.iloc[i] + 0.5769 * detrender.iloc[i-2]
              - 0.5769 * detrender.iloc[i-4] - 0.0962 * detrender.iloc[i-6]) * (0.075 * period.iloc[i-1] + 0.54)
        i1 = detrender.iloc[i-3] if i >= 3 else 0.0

        ji = (0.0962 * q1 + 0.5769 * (q1 if i < 2 else q1)
              ) if False else q1  # placeholder to avoid unused warning
        jq = i1

        i2 = i1 - jq * 0.2
        q2 = q1 + ji * 0.2
        re = i2 * (i2 if False else 0.2) + q2 * (q2 if False else 0.2)
        im_val = i2 * q2 - q2 * i2

        # Simplified: use ratio of Q1/I1 for phase
        if abs(i1) > 1e-9:
            ph = np.degrees(np.arctan(q1 / i1))
        else:
            ph = 0.0
        phase.iloc[i] = ph

        delta_phase = max(1.0, phase.iloc[i-1] - ph)
        alpha = max(slow_limit, min(fast_limit, fast_limit / delta_phase))

        mama.iloc[i] = alpha * close.iloc[i] + (1 - alpha) * mama.iloc[i-1]
        fama.iloc[i] = 0.5 * alpha * mama.iloc[i] + (1 - 0.5 * alpha) * fama.iloc[i-1]
        period.iloc[i] = 2 * np.pi / max(0.01, abs(delta_phase) * np.pi / 180) if delta_phase > 0 else 20.0

    return mama, fama


def _cyber_cycle(close, alpha=0.07):
    """Ehlers Adaptive Cyber Cycle approximation."""
    n = len(close)
    smooth = pd.Series(0.0, index=close.index)
    cycle = pd.Series(0.0, index=close.index)
    for i in range(4, n):
        s = (close.iloc[i] + 2 * close.iloc[i-1] + 2 * close.iloc[i-2] + close.iloc[i-3]) / 6
        smooth.iloc[i] = s
        c = ((1 - 0.5 * alpha) ** 2
             * (smooth.iloc[i] - 2 * smooth.iloc[i-1] + smooth.iloc[i-2])
             + 2 * (1 - alpha) * cycle.iloc[i-1]
             - (1 - alpha) ** 2 * cycle.iloc[i-2])
        cycle.iloc[i] = c
    return cycle


def _simple_cycle(close, alpha=0.07):
    """Ehlers Simple Cycle Indicator."""
    cycle = _cyber_cycle(close, alpha)
    signal = cycle.shift(1)
    return cycle, signal


# ---------------------------------------------------------------------------
# BATCH 027 — SEASONALITY + CALENDAR CYCLES (5)
# ---------------------------------------------------------------------------

def gen_TV_TrueSeasonalPattern(df, lookback_years=3, entry_month_offset=0,
                                exit_month_offset=1, trend_len=50,
                                rsi_len=14):
    """True Seasonal Pattern [TradeVizion] — bias from avg monthly returns over N years.
    Source: https://www.tradingview.com/script/SijvaWFx-True-Seasonal-Pattern-tradeviZion/
    Concept: compute avg return for each calendar month across lookback_years; buy months
    with historically positive bias, sell months with negative bias.
    """
    close = df['close']
    trend = _ema(close, trend_len)
    rsi = _rsi(close, rsi_len)

    # Monthly return bias
    month = df.index.month
    monthly_bias = pd.Series(0.0, index=df.index)
    for m in range(1, 13):
        mask = month == m
        if mask.sum() < 2:
            continue
        monthly_ret = close[mask].pct_change().rolling(lookback_years * 20, min_periods=5).mean()
        monthly_bias[mask] = monthly_ret.fillna(0)

    sig = pd.Series(0, index=df.index)
    long_cond = (monthly_bias > 0) & (close > trend) & (rsi > 50)
    short_cond = (monthly_bias < 0) & (close < trend) & (rsi < 50)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


def gen_TV_SeasonalityPresidentialCycle(df, pres_cycle_len=48, trend_len=50,
                                         rsi_len=14, momentum_len=20,
                                         adx_len=14):
    """Seasonality + Presidential Cycle [FullTimeTradingRu].
    Source: https://www.tradingview.com/script/psG4Vj0S-seasonality-and-presidential-cycle/
    Concept: 4-year presidential cycle phase (year 1=bearish, year 2=neutral,
    year 3=bullish, year 4=bullish) combined with monthly seasonality bias.
    """
    close = df['close']
    trend = _ema(close, trend_len)
    rsi = _rsi(close, rsi_len)
    adx_val, plus_di, minus_di = _adx(df, adx_len)

    # Presidential cycle phase (modulo 4 years from some epoch)
    year = df.index.year
    pres_phase = year % 4  # 0=election, 1=post-election, 2=midterm, 3=pre-election
    pres_bias = pd.Series(0.0, index=df.index)
    pres_bias[pres_phase == 0] = 0.5   # election year bullish
    pres_bias[pres_phase == 1] = -0.3  # post-election bearish
    pres_bias[pres_phase == 2] = 0.1   # midterm neutral
    pres_bias[pres_phase == 3] = 0.8   # pre-election very bullish

    # Monthly bias
    month = df.index.month
    monthly_bias = pd.Series(0.0, index=df.index)
    bullish_months = [4, 5, 10, 11, 12]
    bearish_months = [1, 6, 8, 9]
    monthly_bias[month.isin(bullish_months)] = 0.5
    monthly_bias[month.isin(bearish_months)] = -0.5

    composite = pres_bias + monthly_bias

    sig = pd.Series(0, index=df.index)
    long_cond = (composite > 0.5) & (close > trend) & (rsi > 45) & (adx_val > 20)
    short_cond = (composite < -0.3) & (close < trend) & (rsi < 55) & (adx_val > 20)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


def gen_TV_IntradaySeasonality(df, session_open_hour=9, session_close_hour=16,
                                trend_len=20, rsi_len=10, vol_mult=1.5,
                                morning_bias_hours=3):
    """Intraday Seasonality [kaaiii] — time-of-day bias + volume filter.
    Source: https://www.tradingview.com/script/OuD7PG2n-Intraday-Seasonality/
    Concept: buy in morning session (open to open+N hours) when volume is above average,
    and price is above short-term trend; fade at close of session.
    """
    close = df['close']
    volume = df['volume']
    trend = _ema(close, trend_len)
    rsi = _rsi(close, rsi_len)

    hour = df.index.hour
    vol_ma = volume.rolling(20).mean()
    vol_filter = volume > vol_mult * vol_ma

    morning_cond = (hour >= session_open_hour) & (hour < session_open_hour + morning_bias_hours)
    close_cond = (hour >= session_close_hour - 1)

    sig = pd.Series(0, index=df.index)
    long_cond = morning_cond & vol_filter & (close > trend) & (rsi > 45)
    short_cond = close_cond & (close < trend) & (rsi < 55)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


def gen_TV_SeasonalityForecast(df, forecast_len=5, trend_len=50,
                                rsi_len=14, vol_len=20, adx_len=14):
    """Seasonality Forecast [dsvaryts] — projected seasonal bias + trend confirmation.
    Source: https://www.tradingview.com/script/OhNg5Yk6-Seasonality-Forecast/
    Concept: rolling average of same-period returns across history projects a seasonal
    expected return; trade when forecast is positive/negative and trend confirms.
    """
    close = df['close']
    trend = _ema(close, trend_len)
    rsi = _rsi(close, rsi_len)
    adx_val, plus_di, minus_di = _adx(df, adx_len)

    # Seasonal forecast: rolling mean of N-bar forward return pattern
    ret = close.pct_change(forecast_len)
    seasonal = ret.rolling(52 * forecast_len, min_periods=10).mean()  # ~1 year of weekly cycles

    sig = pd.Series(0, index=df.index)
    long_cond = (seasonal > 0) & (close > trend) & (rsi > 50) & (plus_di > minus_di)
    short_cond = (seasonal < 0) & (close < trend) & (rsi < 50) & (minus_di > plus_di)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


def gen_TV_EnhancedSeasonalityBacktest(df, lookback=100, z_thresh=0.5,
                                        trend_len=50, rsi_len=14,
                                        vol_mult=1.2):
    """Enhanced Seasonality Trade Backtest [Tim234] — z-scored seasonal return.
    Source: https://www.tradingview.com/script/ztSwRLRk/
    Concept: compute z-score of same-day-of-year returns; enter when z-score exceeds
    threshold and trend/RSI confirm direction.
    """
    close = df['close']
    trend = _ema(close, trend_len)
    rsi = _rsi(close, rsi_len)
    volume = df['volume']
    vol_ma = volume.rolling(20).mean()

    doy = df.index.dayofyear
    ret = close.pct_change()
    doy_mean = pd.Series(np.nan, index=df.index)
    doy_std = pd.Series(np.nan, index=df.index)
    for d in range(1, 367):
        mask = doy == d
        if mask.sum() < 3:
            continue
        rolling_vals = ret[mask].rolling(lookback, min_periods=5).mean()
        rolling_std = ret[mask].rolling(lookback, min_periods=5).std(ddof=0)
        doy_mean[mask] = rolling_vals
        doy_std[mask] = rolling_std
    doy_std = doy_std.replace(0, np.nan)
    z_score = (ret - doy_mean) / doy_std

    sig = pd.Series(0, index=df.index)
    long_cond = (z_score > z_thresh) & (close > trend) & (rsi > 50) & (volume > vol_mult * vol_ma)
    short_cond = (z_score < -z_thresh) & (close < trend) & (rsi < 50) & (volume > vol_mult * vol_ma)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


# ---------------------------------------------------------------------------
# BATCH 027 — INTRADAY TIME PATTERNS (5)
# ---------------------------------------------------------------------------

def gen_TV_MonthlyPerformanceStrategy(df, lookback_months=12, trend_len=50,
                                       rsi_len=14, adx_thresh=20,
                                       vol_len=20):
    """Monthly Performance Strategy [exlux] — unlock seasonality via monthly returns.
    Source: https://www.tradingview.com/script/DR2MFR41-Unlock-the-Power-of-Seasonality-Monthly-Performance-Strategy/
    Concept: rank current month by historical average monthly return; buy top-N months,
    sell bottom-N months with trend confirmation.
    """
    close = df['close']
    trend = _ema(close, trend_len)
    rsi = _rsi(close, rsi_len)
    adx_val, plus_di, minus_di = _adx(df, adx_thresh)

    month = df.index.month
    monthly_avg = {}
    monthly_ret = close.resample('ME').last().pct_change() if hasattr(df.index, 'freq') else close.pct_change(20)
    for m in range(1, 13):
        mask = month == m
        monthly_avg[m] = close[mask].pct_change().mean()

    month_bias = df.index.month.map(lambda m: monthly_avg.get(m, 0.0))
    month_bias = pd.Series(month_bias, index=df.index)

    sig = pd.Series(0, index=df.index)
    long_cond = (month_bias > 0) & (close > trend) & (rsi > 50)
    short_cond = (month_bias < 0) & (close < trend) & (rsi < 50)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


def gen_TV_SeasonalityChartingCycles(df, cycle_len=20, trend_len=50,
                                      rsi_len=14, momentum_len=10,
                                      vol_mult=1.2):
    """Seasonality [ChartingCycles] — cyclic momentum bias with trend filter.
    Source: https://www.tradingview.com/script/i1TZiqVf-Seasonality/
    Concept: rolling cycle of fixed length; enter when cycle turns positive/negative
    with trend and momentum confirmation.
    """
    close = df['close']
    trend = _ema(close, trend_len)
    rsi = _rsi(close, rsi_len)
    volume = df['volume']
    vol_ma = volume.rolling(20).mean()

    cycle_phase = close.pct_change(cycle_len).rolling(cycle_len).mean()

    sig = pd.Series(0, index=df.index)
    long_cond = (cycle_phase > 0) & (close > trend) & (rsi > 48) & (volume > vol_mult * vol_ma)
    short_cond = (cycle_phase < 0) & (close < trend) & (rsi < 52) & (volume > vol_mult * vol_ma)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


def gen_TV_SeasonalityMonthHighlight(df, bullish_months_str=4, bearish_months_str=9,
                                      trend_len=50, rsi_len=14,
                                      adx_len=14):
    """Seasonality Month Highlight [twingall] — highlight calendar months by historical bias.
    Source: https://www.tradingview.com/script/8uBdh16P-Seasonality-Month-Highlight/
    Concept: months labeled bullish/bearish based on long-run average; trade with trend.
    """
    close = df['close']
    trend = _ema(close, trend_len)
    rsi = _rsi(close, rsi_len)
    adx_val, plus_di, minus_di = _adx(df, adx_len)

    month = df.index.month
    # Standard crypto seasonality: Q4 bullish, Q3 bearish
    bullish = {10, 11, 12, 1}
    bearish = {6, 7, 8, 9}

    bias = pd.Series(0, index=df.index)
    bias[month.isin(bullish)] = 1
    bias[month.isin(bearish)] = -1

    sig = pd.Series(0, index=df.index)
    long_cond = (bias == 1) & (close > trend) & (rsi > 48) & (adx_val > 18)
    short_cond = (bias == -1) & (close < trend) & (rsi < 52) & (adx_val > 18)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


def gen_TV_SeasonalTendency(df, lookback=200, quantile_thresh=0.6,
                             trend_len=50, rsi_len=14, vol_len=20):
    """Seasonal Tendency [fadi] — percentile-based seasonal entry filter.
    Source: https://www.tradingview.com/script/9h4Tj6sM-Seasonal-Tendency-fadi/
    Concept: rolling returns by day-of-year; trade when current period's historical
    return rank exceeds quantile threshold.
    """
    close = df['close']
    trend = _ema(close, trend_len)
    rsi = _rsi(close, rsi_len)

    ret = close.pct_change(5)  # weekly return
    ret_rank = ret.rolling(lookback, min_periods=20).rank(pct=True)

    sig = pd.Series(0, index=df.index)
    long_cond = (ret_rank > quantile_thresh) & (close > trend) & (rsi > 50)
    short_cond = (ret_rank < 1 - quantile_thresh) & (close < trend) & (rsi < 50)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


def gen_TV_SeasonalityCustomInterval(df, interval_bars=24, trend_len=50,
                                      rsi_len=14, vol_mult=1.3,
                                      lookback=50):
    """Seasonality with Custom Interval [TradersPod] — custom-length cycle seasonality.
    Source: https://www.tradingview.com/script/3vlvXQq1-Seasonality-with-Custom-Interval/
    Concept: compute rolling average return for each bar position within a custom cycle
    length; enter long/short based on expected return and trend filter.
    """
    close = df['close']
    trend = _ema(close, trend_len)
    rsi = _rsi(close, rsi_len)
    volume = df['volume']
    vol_ma = volume.rolling(20).mean()

    ret = close.pct_change()
    bar_in_cycle = pd.Series(range(len(close)), index=df.index) % interval_bars
    cycle_avg = pd.Series(np.nan, index=df.index)
    for b in range(interval_bars):
        mask = bar_in_cycle == b
        if mask.sum() < 3:
            continue
        cycle_avg[mask] = ret[mask].rolling(lookback, min_periods=3).mean()
    cycle_avg = cycle_avg.fillna(0)

    sig = pd.Series(0, index=df.index)
    long_cond = (cycle_avg > 0) & (close > trend) & (rsi > 48) & (volume > vol_mult * vol_ma)
    short_cond = (cycle_avg < 0) & (close < trend) & (rsi < 52) & (volume > vol_mult * vol_ma)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


# ---------------------------------------------------------------------------
# BATCH 027 — MESA + DOMINANT CYCLES (5)
# ---------------------------------------------------------------------------

def gen_TV_MESAAdaptiveEhlersFlow(df, fast_limit=0.5, slow_limit=0.05,
                                   trend_len=50, rsi_len=14,
                                   adx_len=14):
    """MESA Adaptive Ehlers Flow [AlphaNatt] — MAMA/FAMA staircase crossover.
    Source: https://www.tradingview.com/script/9SZSadPm-MESA-Adaptive-Ehlers-Flow-AlphaNatt/
    Concept: MAMA crosses above FAMA → long; below → short. Filtered by trend and ADX.
    """
    close = df['close']
    mama, fama = _mama_fama(close, fast_limit, slow_limit)
    trend = _ema(close, trend_len)
    rsi = _rsi(close, rsi_len)
    adx_val, plus_di, minus_di = _adx(df, adx_len)

    mama_cross_up = (mama > fama) & (mama.shift(1) <= fama.shift(1))
    mama_cross_dn = (mama < fama) & (mama.shift(1) >= fama.shift(1))

    sig = pd.Series(0, index=df.index)
    long_cond = mama_cross_up & (close > trend) & (adx_val > 20)
    short_cond = mama_cross_dn & (close < trend) & (adx_val > 20)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


def gen_TV_EhlersMESAAdaptiveMA(df, fast_limit=0.5, slow_limit=0.05,
                                  trend_len=50, rsi_len=14, vol_mult=1.2):
    """Ehlers MESA Adaptive Moving Average [LazyBear] — Hilbert Transform cycle MA.
    Source: https://www.tradingview.com/script/foQxLbU3-Ehlers-MESA-Adaptive-Moving-Average-LazyBear/
    Concept: MAMA/FAMA adaptive crossover with volume confirmation.
    """
    close = df['close']
    volume = df['volume']
    vol_ma = volume.rolling(20).mean()
    mama, fama = _mama_fama(close, fast_limit, slow_limit)
    rsi = _rsi(close, rsi_len)

    sig = pd.Series(0, index=df.index)
    long_cond = (mama > fama) & (rsi > 50) & (volume > vol_mult * vol_ma)
    short_cond = (mama < fama) & (rsi < 50) & (volume > vol_mult * vol_ma)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


def gen_TV_DominantCycleLibrary(df, min_period=10, max_period=48,
                                  trend_len=50, rsi_len=14, adx_len=14):
    """DominantCycle Library [lastguru] — autocorrelation dominant cycle detection.
    Source: https://www.tradingview.com/script/cY7DdxyZ-DominantCycle/
    Concept: detect dominant price cycle; go long at cycle trough, short at peak.
    """
    close = df['close']
    trend = _ema(close, trend_len)
    rsi = _rsi(close, rsi_len)
    adx_val, plus_di, minus_di = _adx(df, adx_len)

    dc = _dominant_cycle(close, min_period, max_period)
    dc_smooth = dc.rolling(5).mean()

    # Cycle position: sine approximation based on bar count modulo dominant period
    cycle_pos = pd.Series(0.0, index=close.index)
    bar_count = pd.Series(range(len(close)), index=close.index)
    for i in range(max_period, len(close)):
        p = max(1, int(dc_smooth.iloc[i]))
        phase_val = (bar_count.iloc[i] % p) / p * 2 * np.pi
        cycle_pos.iloc[i] = np.sin(phase_val)

    sig = pd.Series(0, index=df.index)
    # Long when cycle is turning up from trough (sin cross zero upward)
    long_cond = (cycle_pos > 0) & (cycle_pos.shift(1) <= 0) & (close > trend) & (adx_val > 18)
    short_cond = (cycle_pos < 0) & (cycle_pos.shift(1) >= 0) & (close < trend) & (adx_val > 18)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


def gen_TV_EhlersAdaptiveCyberCycle(df, alpha=0.07, overbought=0.5,
                                     oversold=-0.5, trend_len=50,
                                     rsi_len=14):
    """Ehlers Adaptive Cyber Cycle [LazyBear] — adaptive cycle oscillator.
    Source: https://www.tradingview.com/script/3lV1e3ci-Ehlers-Adaptive-Cyber-Cycle-Indicator-LazyBear/
    Concept: cycle crosses zero → trade; overbought/oversold levels for mean-reversion.
    """
    close = df['close']
    trend = _ema(close, trend_len)
    rsi = _rsi(close, rsi_len)

    cycle = _cyber_cycle(close, alpha)
    trigger = cycle.shift(1)

    sig = pd.Series(0, index=df.index)
    long_cond = (cycle > trigger) & (cycle < oversold) & (close > trend)
    short_cond = (cycle < trigger) & (cycle > overbought) & (close < trend)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


def gen_TV_EhlersSimpleCycleIndicator(df, alpha=0.07, trend_len=50,
                                       rsi_len=14, vol_mult=1.2,
                                       adx_len=14):
    """Ehlers Simple Cycle Indicator [LazyBear] — zero-cross cycle momentum.
    Source: https://www.tradingview.com/script/xQ4mP4kc-Ehlers-Simple-Cycle-Indicator-LazyBear/
    Concept: cycle crosses zero from below → long; crosses from above → short.
    """
    close = df['close']
    trend = _ema(close, trend_len)
    rsi = _rsi(close, rsi_len)
    volume = df['volume']
    vol_ma = volume.rolling(20).mean()

    cycle, signal = _simple_cycle(close, alpha)

    sig = pd.Series(0, index=df.index)
    long_cond = (cycle > 0) & (cycle.shift(1) <= 0) & (close > trend) & (volume > vol_mult * vol_ma)
    short_cond = (cycle < 0) & (cycle.shift(1) >= 0) & (close < trend) & (volume > vol_mult * vol_ma)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


# ---------------------------------------------------------------------------
# BATCH 027 — EHLERS FISHER TRANSFORM VARIANTS (5)
# ---------------------------------------------------------------------------

def gen_TV_InverseFisherAdaptiveStoch(df, stoch_len=14, smooth_k=3,
                                       ift_thresh=0.5, trend_len=50,
                                       rsi_len=14):
    """Inverse Fisher Transform Adaptive Stochastic [palitoj_endthen].
    Source: https://www.tradingview.com/script/uMTi9oYD-inverse-fisher-transform-adaptive-stochastic/
    Concept: apply IFT to stochastic → bounded [-1,+1]; cross ±threshold → trade.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    trend = _ema(close, trend_len)
    rsi = _rsi(close, rsi_len)

    stoch_k = _stoch(high, low, close, stoch_len, smooth_k)
    # Normalize stochastic to [-1, 1]
    stoch_norm = (stoch_k / 50 - 1).clip(-0.999, 0.999)
    ift_stoch = pd.Series(_ift(stoch_norm.values), index=df.index)

    sig = pd.Series(0, index=df.index)
    long_cond = (ift_stoch > -ift_thresh) & (ift_stoch.shift(1) <= -ift_thresh) & (close > trend)
    short_cond = (ift_stoch < ift_thresh) & (ift_stoch.shift(1) >= ift_thresh) & (close < trend)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


def gen_TV_InverseFisherStoch(df, stoch_len=14, smooth_k=3,
                               signal_len=9, trend_len=50,
                               rsi_len=14):
    """Inverse Fisher Transform Stochastic [KivancOzbilgic].
    Source: https://www.tradingview.com/script/WikDwOZC/
    Concept: IFT of stochastic crosses signal line for entry; trend filter.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    trend = _ema(close, trend_len)

    stoch_k = _stoch(high, low, close, stoch_len, smooth_k)
    stoch_norm = (stoch_k / 50 - 1).clip(-0.999, 0.999)
    ift_stoch = pd.Series(_ift(stoch_norm.values), index=df.index)
    ift_signal = _ema(ift_stoch, signal_len)

    sig = pd.Series(0, index=df.index)
    long_cond = (ift_stoch > ift_signal) & (ift_stoch.shift(1) <= ift_signal.shift(1)) & (close > trend)
    short_cond = (ift_stoch < ift_signal) & (ift_stoch.shift(1) >= ift_signal.shift(1)) & (close < trend)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


def gen_TV_AdaptiveFisherizedCMO(df, cmo_len=14, fisher_len=10,
                                   ift_thresh=0.3, trend_len=50,
                                   adx_len=14):
    """Adaptive Fisherized CMO [simwai] — IFT applied to Chande Momentum Oscillator.
    Source: https://www.tradingview.com/script/f9bz3PcY-Adaptive-Fisherized-CMO/
    Concept: CMO normalized and Fisher-transformed; IFT bounds it to [-1,+1].
    Cross of zero or threshold level triggers entry with ADX filter.
    """
    close = df['close']
    trend = _ema(close, trend_len)
    adx_val, plus_di, minus_di = _adx(df, adx_len)

    cmo = _cmo(close, cmo_len)
    cmo_norm = (cmo / 100).clip(-0.999, 0.999)
    fisher_cmo = _fisher_transform(close, fisher_len)
    ift_cmo = pd.Series(_ift(cmo_norm.values), index=df.index)

    sig = pd.Series(0, index=df.index)
    long_cond = (ift_cmo > ift_thresh) & (ift_cmo.shift(1) <= ift_thresh) & (close > trend) & (adx_val > 20)
    short_cond = (ift_cmo < -ift_thresh) & (ift_cmo.shift(1) >= -ift_thresh) & (close < trend) & (adx_val > 20)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


def gen_TV_AdaptiveFisherizedROC(df, roc_len=10, fisher_len=10,
                                   ift_thresh=0.2, trend_len=50,
                                   rsi_len=14):
    """Adaptive Fisherized ROC [simwai] — IFT applied to Rate of Change.
    Source: https://www.tradingview.com/script/Ufbx4DRV-Adaptive-Fisherized-ROC/
    Concept: ROC normalized to [-1,+1] and Fisher-transformed then IFT applied;
    trades cross of IFT threshold with trend confirmation.
    """
    close = df['close']
    trend = _ema(close, trend_len)
    rsi = _rsi(close, rsi_len)

    roc = _roc(close, roc_len)
    # Normalize ROC to [-1,+1] using rolling percentile
    roc_rank = roc.rolling(100, min_periods=10).rank(pct=True) * 2 - 1
    roc_norm = roc_rank.clip(-0.999, 0.999)
    ift_roc = pd.Series(_ift(roc_norm.values), index=df.index)

    sig = pd.Series(0, index=df.index)
    long_cond = (ift_roc > ift_thresh) & (ift_roc.shift(1) <= ift_thresh) & (close > trend) & (rsi > 50)
    short_cond = (ift_roc < -ift_thresh) & (ift_roc.shift(1) >= -ift_thresh) & (close < trend) & (rsi < 50)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


def gen_TV_FisherTransformEhlersBacktest(df, fisher_len=10, signal_len=1,
                                          trend_len=50, rsi_len=14,
                                          adx_len=14):
    """Fisher Transform Indicator by Ehlers Backtest v2.0 [HPotter].
    Source: https://www.tradingview.com/script/qhXwZvJ2-Fisher-Transform-Indicator-by-Ehlers-Backtest-v-2-0/
    Concept: Ehlers Fisher Transform crossover of signal line; bounded non-linear
    transformation amplifies turning points; ADX and RSI for quality filter.
    """
    close = df['close']
    trend = _ema(close, trend_len)
    rsi = _rsi(close, rsi_len)
    adx_val, plus_di, minus_di = _adx(df, adx_len)

    fisher = _fisher_transform(close, fisher_len)
    signal_line = fisher.shift(signal_len)

    sig = pd.Series(0, index=df.index)
    long_cond = (fisher > signal_line) & (fisher.shift(1) <= signal_line.shift(1)) & (close > trend) & (adx_val > 20)
    short_cond = (fisher < signal_line) & (fisher.shift(1) >= signal_line.shift(1)) & (close < trend) & (adx_val > 20)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


# ---------------------------------------------------------------------------
# BATCH 027 — COMPOSITE MOMENTUM SCORING (5)
# ---------------------------------------------------------------------------

def gen_TV_MLEnhancedCompositeSignal(df, rsi_len=14, macd_fast=12,
                                      macd_slow=26, macd_sig=9,
                                      adx_len=14):
    """ML-Enhanced Multi-Indicator Composite Signal [QuantCreative].
    Source: https://www.tradingview.com/script/vofMjW64-ML-Enhanced-Multi-Indicator-Composite-Signal/
    Concept: weighted composite of RSI, MACD, ADX, Stochastic, and Bollinger;
    adaptive weights via rolling correlation of each indicator with forward returns.
    """
    close = df['close']
    high = df['high']
    low = df['low']

    rsi = _rsi(close, rsi_len)
    macd_line, macd_signal = _macd(close, macd_fast, macd_slow, macd_sig)
    adx_val, plus_di, minus_di = _adx(df, adx_len)
    stoch_k = _stoch(high, low, close, rsi_len, 3)
    bb_lo, bb_mid, bb_hi = _bb(close, 20, 2.0)

    # Normalize each to [-1, +1]
    rsi_n = (rsi - 50) / 50
    macd_n = np.sign(macd_line - macd_signal)
    adx_n = np.sign(plus_di - minus_di) * (adx_val / 50).clip(0, 1)
    stoch_n = (stoch_k - 50) / 50
    bb_pos = (close - bb_mid) / (bb_hi - bb_lo + 1e-9)
    bb_n = -bb_pos  # mean-reversion: above upper = bearish

    # Forward return for adaptive weights
    fwd = close.pct_change(3).shift(-3)

    def _rolling_corr_weight(indicator, window=50):
        corr = indicator.rolling(window, min_periods=10).corr(fwd)
        return corr.abs().fillna(0.2)

    w_rsi = _rolling_corr_weight(rsi_n)
    w_macd = _rolling_corr_weight(pd.Series(macd_n, index=df.index))
    w_adx = _rolling_corr_weight(pd.Series(adx_n, index=df.index))
    w_stoch = _rolling_corr_weight(stoch_n)
    w_bb = _rolling_corr_weight(pd.Series(bb_n, index=df.index))

    w_total = w_rsi + w_macd + w_adx + w_stoch + w_bb + 1e-9
    composite = (w_rsi * rsi_n + w_macd * macd_n + w_adx * adx_n
                 + w_stoch * stoch_n + w_bb * bb_n) / w_total

    sig = pd.Series(0, index=df.index)
    sig[composite > 0.2] = 1
    sig[composite < -0.2] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


def gen_TV_CompositeBuySellScore(df, rsi_len=14, macd_fast=12,
                                  macd_slow=26, williams_len=14,
                                  score_thresh=30):
    """Composite Buy/Sell Score [-100 to +100] [LouisMont].
    Source: https://www.tradingview.com/script/DFgw9vm5-Composite-Buy-Sell-Score-100-to-100-by-LM/
    Concept: PPO + ADX directional + RSI + MACD histogram + Williams%R → composite
    score scaled to [-100, +100]; buy > thresh, sell < -thresh.
    """
    close = df['close']
    high = df['high']
    low = df['low']

    rsi = _rsi(close, rsi_len)
    macd_line, macd_signal = _macd(close, macd_fast, macd_slow)
    macd_hist = macd_line - macd_signal
    adx_val, plus_di, minus_di = _adx(df, 14)

    # PPO (Percentage Price Oscillator)
    ema_fast = _ema(close, macd_fast)
    ema_slow = _ema(close, macd_slow)
    ppo = (ema_fast - ema_slow) / ema_slow.replace(0, np.nan) * 100

    # Williams %R
    hi_n = high.rolling(williams_len).max()
    lo_n = low.rolling(williams_len).min()
    wr = -100 * (hi_n - close) / (hi_n - lo_n + 1e-9)

    # Score components (each ±20 points)
    s_rsi = (rsi - 50) * 0.4  # RSI 70→+8, 30→-8
    s_ppo = ppo.clip(-5, 5) * 4
    s_macd = np.sign(macd_hist) * 20
    s_adx = np.sign(plus_di - minus_di) * (adx_val / 50).clip(0, 1) * 20
    s_wr = (wr + 50) * 0.4  # WR -80→-12, -20→+12

    score = s_rsi + s_ppo + s_macd + s_adx + s_wr
    score = pd.Series(score, index=df.index) if not isinstance(score, pd.Series) else score

    sig = pd.Series(0, index=df.index)
    sig[score > score_thresh] = 1
    sig[score < -score_thresh] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


def gen_TV_StochRSI_RSI_MACD_Signals(df, rsi_len=14, stoch_len=14,
                                      smooth_k=3, smooth_d=3,
                                      macd_fast=12):
    """Stoch RSI + RSI + MACD Signals [Pedro-Elias].
    Source: https://www.tradingview.com/script/yZCjneHe-Stoch-RSI-and-RSI-Buy-Sell-Signals-with-MACD-Trend-Filter/
    Concept: StochRSI K > D crossover with RSI confirmation and MACD trend filter.
    """
    close = df['close']
    high = df['high']
    low = df['low']

    rsi = _rsi(close, rsi_len)
    # StochRSI
    rsi_lo = rsi.rolling(stoch_len).min()
    rsi_hi = rsi.rolling(stoch_len).max()
    stoch_rsi = (rsi - rsi_lo) / (rsi_hi - rsi_lo + 1e-9) * 100
    k = stoch_rsi.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()

    macd_line, macd_signal = _macd(close, macd_fast, 26, 9)
    macd_bull = macd_line > macd_signal

    sig = pd.Series(0, index=df.index)
    long_cond = (k > d) & (k.shift(1) <= d.shift(1)) & (k < 80) & (rsi > 40) & macd_bull
    short_cond = (k < d) & (k.shift(1) >= d.shift(1)) & (k > 20) & (rsi < 60) & ~macd_bull
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


def gen_TV_CombinedStrategyRSIADXSMA(df, rsi_len=14, adx_len=14,
                                      sma_len=20, rsi_ob=70,
                                      rsi_os=30):
    """Combined Strategy Trading Bot RSI ADX 20SMA [Mowisdom].
    Source: https://www.tradingview.com/script/7wrshs1q-Combined-Strategy-Trading-Bot-RSI-ADX-20SMA/
    Concept: RSI oversold/overbought + ADX trend strength + SMA trend direction.
    """
    close = df['close']
    rsi = _rsi(close, rsi_len)
    adx_val, plus_di, minus_di = _adx(df, adx_len)
    sma = close.rolling(sma_len).mean()

    sig = pd.Series(0, index=df.index)
    long_cond = (rsi < rsi_os) & (adx_val > 20) & (plus_di > minus_di) & (close > sma)
    short_cond = (rsi > rsi_ob) & (adx_val > 20) & (minus_di > plus_di) & (close < sma)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


def gen_TV_PeriodHighlighterPro(df, period_len=20, highlight_thresh=0.6,
                                  trend_len=50, rsi_len=14,
                                  vol_mult=1.2):
    """Period Highlighter Pro [mjarestad] — highlight high-momentum periods.
    Source: https://www.tradingview.com/script/kV6F4yel-Period-Highlighter-Pro/
    Concept: score each bar by composite momentum within a lookback period;
    enter when score exceeds threshold in trend direction.
    """
    close = df['close']
    volume = df['volume']
    trend = _ema(close, trend_len)
    rsi = _rsi(close, rsi_len)
    vol_ma = volume.rolling(20).mean()

    # Period score: fraction of bars with positive return in lookback
    ret = close.pct_change()
    pos_fraction = (ret > 0).rolling(period_len).mean()
    momentum = close.pct_change(period_len)

    sig = pd.Series(0, index=df.index)
    long_cond = (pos_fraction > highlight_thresh) & (momentum > 0) & (close > trend) & (rsi > 50) & (volume > vol_mult * vol_ma)
    short_cond = (pos_fraction < 1 - highlight_thresh) & (momentum < 0) & (close < trend) & (rsi < 50) & (volume > vol_mult * vol_ma)
    sig[long_cond] = 1
    sig[short_cond] = -1

    entry = sig.shift(1)
    sig_out = pd.Series(0, index=df.index)
    sig_out[entry == 1] = 1
    sig_out[entry == -1] = -1
    return sig_out


# ---------------------------------------------------------------------------
# SPACES
# ---------------------------------------------------------------------------

def _space_TV_TrueSeasonalPattern(trial):
    return {
        'lookback_years': trial.suggest_int('lookback_years', 2, 5),
        'entry_month_offset': trial.suggest_int('entry_month_offset', 0, 2),
        'exit_month_offset': trial.suggest_int('exit_month_offset', 1, 3),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
    }


def _space_TV_SeasonalityPresidentialCycle(trial):
    return {
        'pres_cycle_len': trial.suggest_int('pres_cycle_len', 36, 60),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'momentum_len': trial.suggest_int('momentum_len', 10, 30),
        'adx_len': trial.suggest_int('adx_len', 10, 20),
    }


def _space_TV_IntradaySeasonality(trial):
    return {
        'session_open_hour': trial.suggest_int('session_open_hour', 0, 12),
        'session_close_hour': trial.suggest_int('session_close_hour', 12, 23),
        'trend_len': trial.suggest_int('trend_len', 10, 50),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'vol_mult': trial.suggest_float('vol_mult', 1.0, 2.5),
    }


def _space_TV_SeasonalityForecast(trial):
    return {
        'forecast_len': trial.suggest_int('forecast_len', 3, 10),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'vol_len': trial.suggest_int('vol_len', 10, 30),
        'adx_len': trial.suggest_int('adx_len', 10, 20),
    }


def _space_TV_EnhancedSeasonalityBacktest(trial):
    return {
        'lookback': trial.suggest_int('lookback', 50, 200),
        'z_thresh': trial.suggest_float('z_thresh', 0.2, 1.5),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'vol_mult': trial.suggest_float('vol_mult', 1.0, 2.5),
    }


def _space_TV_MonthlyPerformanceStrategy(trial):
    return {
        'lookback_months': trial.suggest_int('lookback_months', 6, 24),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'adx_thresh': trial.suggest_int('adx_thresh', 10, 30),
        'vol_len': trial.suggest_int('vol_len', 10, 30),
    }


def _space_TV_SeasonalityChartingCycles(trial):
    return {
        'cycle_len': trial.suggest_int('cycle_len', 10, 40),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'momentum_len': trial.suggest_int('momentum_len', 5, 20),
        'vol_mult': trial.suggest_float('vol_mult', 1.0, 2.5),
    }


def _space_TV_SeasonalityMonthHighlight(trial):
    return {
        'bullish_months_str': trial.suggest_int('bullish_months_str', 1, 6),
        'bearish_months_str': trial.suggest_int('bearish_months_str', 6, 12),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'adx_len': trial.suggest_int('adx_len', 10, 20),
    }


def _space_TV_SeasonalTendency(trial):
    return {
        'lookback': trial.suggest_int('lookback', 100, 300),
        'quantile_thresh': trial.suggest_float('quantile_thresh', 0.5, 0.8),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'vol_len': trial.suggest_int('vol_len', 10, 30),
    }


def _space_TV_SeasonalityCustomInterval(trial):
    return {
        'interval_bars': trial.suggest_int('interval_bars', 12, 48),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'vol_mult': trial.suggest_float('vol_mult', 1.0, 2.5),
        'lookback': trial.suggest_int('lookback', 30, 100),
    }


def _space_TV_MESAAdaptiveEhlersFlow(trial):
    return {
        'fast_limit': trial.suggest_float('fast_limit', 0.3, 0.7),
        'slow_limit': trial.suggest_float('slow_limit', 0.01, 0.1),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'adx_len': trial.suggest_int('adx_len', 10, 20),
    }


def _space_TV_EhlersMESAAdaptiveMA(trial):
    return {
        'fast_limit': trial.suggest_float('fast_limit', 0.3, 0.7),
        'slow_limit': trial.suggest_float('slow_limit', 0.01, 0.1),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'vol_mult': trial.suggest_float('vol_mult', 1.0, 2.5),
    }


def _space_TV_DominantCycleLibrary(trial):
    return {
        'min_period': trial.suggest_int('min_period', 5, 20),
        'max_period': trial.suggest_int('max_period', 24, 60),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'adx_len': trial.suggest_int('adx_len', 10, 20),
    }


def _space_TV_EhlersAdaptiveCyberCycle(trial):
    return {
        'alpha': trial.suggest_float('alpha', 0.03, 0.15),
        'overbought': trial.suggest_float('overbought', 0.3, 0.7),
        'oversold': trial.suggest_float('oversold', -0.7, -0.3),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
    }


def _space_TV_EhlersSimpleCycleIndicator(trial):
    return {
        'alpha': trial.suggest_float('alpha', 0.03, 0.15),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'vol_mult': trial.suggest_float('vol_mult', 1.0, 2.5),
        'adx_len': trial.suggest_int('adx_len', 10, 20),
    }


def _space_TV_InverseFisherAdaptiveStoch(trial):
    return {
        'stoch_len': trial.suggest_int('stoch_len', 7, 21),
        'smooth_k': trial.suggest_int('smooth_k', 2, 5),
        'ift_thresh': trial.suggest_float('ift_thresh', 0.2, 0.8),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
    }


def _space_TV_InverseFisherStoch(trial):
    return {
        'stoch_len': trial.suggest_int('stoch_len', 7, 21),
        'smooth_k': trial.suggest_int('smooth_k', 2, 5),
        'signal_len': trial.suggest_int('signal_len', 5, 15),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
    }


def _space_TV_AdaptiveFisherizedCMO(trial):
    return {
        'cmo_len': trial.suggest_int('cmo_len', 7, 21),
        'fisher_len': trial.suggest_int('fisher_len', 5, 20),
        'ift_thresh': trial.suggest_float('ift_thresh', 0.1, 0.6),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'adx_len': trial.suggest_int('adx_len', 10, 20),
    }


def _space_TV_AdaptiveFisherizedROC(trial):
    return {
        'roc_len': trial.suggest_int('roc_len', 5, 20),
        'fisher_len': trial.suggest_int('fisher_len', 5, 20),
        'ift_thresh': trial.suggest_float('ift_thresh', 0.1, 0.5),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
    }


def _space_TV_FisherTransformEhlersBacktest(trial):
    return {
        'fisher_len': trial.suggest_int('fisher_len', 5, 20),
        'signal_len': trial.suggest_int('signal_len', 1, 5),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'adx_len': trial.suggest_int('adx_len', 10, 20),
    }


def _space_TV_MLEnhancedCompositeSignal(trial):
    return {
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'macd_fast': trial.suggest_int('macd_fast', 8, 16),
        'macd_slow': trial.suggest_int('macd_slow', 20, 32),
        'macd_sig': trial.suggest_int('macd_sig', 6, 12),
        'adx_len': trial.suggest_int('adx_len', 10, 20),
    }


def _space_TV_CompositeBuySellScore(trial):
    return {
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'macd_fast': trial.suggest_int('macd_fast', 8, 16),
        'macd_slow': trial.suggest_int('macd_slow', 20, 32),
        'williams_len': trial.suggest_int('williams_len', 7, 21),
        'score_thresh': trial.suggest_int('score_thresh', 15, 50),
    }


def _space_TV_StochRSI_RSI_MACD_Signals(trial):
    return {
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'stoch_len': trial.suggest_int('stoch_len', 7, 21),
        'smooth_k': trial.suggest_int('smooth_k', 2, 5),
        'smooth_d': trial.suggest_int('smooth_d', 2, 5),
        'macd_fast': trial.suggest_int('macd_fast', 8, 16),
    }


def _space_TV_CombinedStrategyRSIADXSMA(trial):
    return {
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'adx_len': trial.suggest_int('adx_len', 10, 20),
        'sma_len': trial.suggest_int('sma_len', 10, 50),
        'rsi_ob': trial.suggest_int('rsi_ob', 65, 80),
        'rsi_os': trial.suggest_int('rsi_os', 20, 35),
    }


def _space_TV_PeriodHighlighterPro(trial):
    return {
        'period_len': trial.suggest_int('period_len', 10, 40),
        'highlight_thresh': trial.suggest_float('highlight_thresh', 0.5, 0.8),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'vol_mult': trial.suggest_float('vol_mult', 1.0, 2.5),
    }


# ---------------------------------------------------------------------------
# BATCH028 — SR ZONE BREAKOUT (7)
# ---------------------------------------------------------------------------

def gen_TV_SRBreakout5Min(df, pivot_left=10, pivot_right=5, sr_thresh=0.001,
                           rsi_len=14, trend_len=50):
    """Support-Resistance Strategy 5Min TF [The_Algo_Engineer]
    Source: https://www.tradingview.com/script/KOfsPM9h-SUPPORT-RESISTANCE-STRATEGY-5MIN-TF/
    Concept: Pivot-based S/R levels; enter long on close above resistance,
    short on close below support, with RSI and trend filter.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, trend_len)

    ph = high.rolling(pivot_left + pivot_right + 1).max()
    pl = low.rolling(pivot_left + pivot_right + 1).min()
    res = ph.shift(pivot_right)
    sup = pl.shift(pivot_right)

    long_sig = (close > res * (1 + sr_thresh)) & (close > trend) & (rsi > 50)
    short_sig = (close < sup * (1 - sr_thresh)) & (close < trend) & (rsi < 50)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_SRBreakout5Min(trial):
    return {
        'pivot_left': trial.suggest_int('pivot_left', 5, 20),
        'pivot_right': trial.suggest_int('pivot_right', 3, 10),
        'sr_thresh': trial.suggest_float('sr_thresh', 0.0005, 0.005),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
    }


def gen_TV_SRBreakoutPmk(df, pivot_left=15, pivot_right=5, vol_mult=1.2,
                          rsi_len=14, atr_mult=1.5):
    """Support-Resistance Breakout [pmk07]
    Source: https://www.tradingview.com/script/hwHKYkE0-Support-Resistance-breakout/
    Concept: Pivot high/low breakout with volume confirmation and ATR-based buffer.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    vol = df['volume']
    atr = _atr(df, 14)
    rsi = _rsi(close, rsi_len)
    vol_avg = vol.rolling(20).mean()

    ph = high.rolling(pivot_left + pivot_right + 1).max().shift(pivot_right)
    pl = low.rolling(pivot_left + pivot_right + 1).min().shift(pivot_right)
    buf = atr * atr_mult * 0.1

    long_sig = (close > ph + buf) & (vol > vol_avg * vol_mult) & (rsi > 50)
    short_sig = (close < pl - buf) & (vol > vol_avg * vol_mult) & (rsi < 50)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_SRBreakoutPmk(trial):
    return {
        'pivot_left': trial.suggest_int('pivot_left', 8, 25),
        'pivot_right': trial.suggest_int('pivot_right', 3, 10),
        'vol_mult': trial.suggest_float('vol_mult', 0.8, 2.5),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'atr_mult': trial.suggest_float('atr_mult', 0.5, 3.0),
    }


def gen_TV_GQTVolumeSRZones(df, zone_len=20, vol_mult=1.5, rsi_len=14,
                              atr_mult=1.0, trend_len=50):
    """GQT GPT Volume-based S&R Zones V2 [Tuzi_GQT]
    Source: https://www.tradingview.com/script/WPCe698c-GQT-GPT-Volume-based-Support-Resistance-Zones-V2/
    Concept: High-volume candles define S/R zones; trade breakouts with trend filter.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    vol = df['volume']
    atr = _atr(df, 14)
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, trend_len)
    vol_avg = vol.rolling(zone_len).mean()

    hi_vol = vol > vol_avg * vol_mult
    zone_hi = high.where(hi_vol, np.nan).rolling(zone_len).max().ffill()
    zone_lo = low.where(hi_vol, np.nan).rolling(zone_len).min().ffill()
    buf = atr * atr_mult * 0.05

    long_sig = (close > zone_hi + buf) & (close > trend) & (rsi > 50)
    short_sig = (close < zone_lo - buf) & (close < trend) & (rsi < 50)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_GQTVolumeSRZones(trial):
    return {
        'zone_len': trial.suggest_int('zone_len', 10, 40),
        'vol_mult': trial.suggest_float('vol_mult', 1.0, 3.0),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'atr_mult': trial.suggest_float('atr_mult', 0.5, 2.0),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
    }


def gen_TV_DynamicBreakoutMaster(df, atr_len=14, atr_mult=1.5, pivot_left=10,
                                  pivot_right=5, rsi_len=14):
    """Dynamic Breakout Master [tradingbauhaus]
    Source: https://www.tradingview.com/script/mqpqHVvo-Dynamic-Breakout-Master-by-tradingbauhaus/
    Concept: Dynamic ATR-adjusted breakout bands from pivot highs/lows.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    atr = _atr(df, atr_len)
    rsi = _rsi(close, rsi_len)

    ph = high.rolling(pivot_left + pivot_right + 1).max().shift(pivot_right)
    pl = low.rolling(pivot_left + pivot_right + 1).min().shift(pivot_right)
    upper = ph + atr * atr_mult * 0.2
    lower = pl - atr * atr_mult * 0.2

    long_sig = (close > upper) & (rsi > 50) & (rsi < 80)
    short_sig = (close < lower) & (rsi < 50) & (rsi > 20)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_DynamicBreakoutMaster(trial):
    return {
        'atr_len': trial.suggest_int('atr_len', 7, 21),
        'atr_mult': trial.suggest_float('atr_mult', 0.5, 3.0),
        'pivot_left': trial.suggest_int('pivot_left', 5, 20),
        'pivot_right': trial.suggest_int('pivot_right', 3, 10),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
    }


def gen_TV_BreaksAndRetests(df, pivot_left=10, pivot_right=5, retest_atr=0.5,
                             rsi_len=14, trend_len=50):
    """Breaks and Retests [Free990]
    Source: https://www.tradingview.com/script/800ndgbX-Breaks-and-Retests-Free990/
    Concept: Break above resistance, then retest (pullback within ATR band) = entry.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    atr = _atr(df, 14)
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, trend_len)

    ph = high.rolling(pivot_left + pivot_right + 1).max().shift(pivot_right)
    pl = low.rolling(pivot_left + pivot_right + 1).min().shift(pivot_right)

    broke_res = (close.shift(1) > ph.shift(1)) & (close > trend)
    retest_zone_lo = ph - atr * retest_atr
    retest_long = broke_res & (close >= retest_zone_lo) & (close <= ph + atr * 0.2) & (rsi > 40)

    broke_sup = (close.shift(1) < pl.shift(1)) & (close < trend)
    retest_zone_hi = pl + atr * retest_atr
    retest_short = broke_sup & (close <= retest_zone_hi) & (close >= pl - atr * 0.2) & (rsi < 60)

    sig = pd.Series(0, index=df.index)
    sig[retest_long] = 1
    sig[retest_short] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_BreaksAndRetests(trial):
    return {
        'pivot_left': trial.suggest_int('pivot_left', 5, 20),
        'pivot_right': trial.suggest_int('pivot_right', 3, 10),
        'retest_atr': trial.suggest_float('retest_atr', 0.2, 1.5),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
    }


def gen_TV_DynamicSRPivot(df, pivot_left=12, pivot_right=4, atr_mult=1.0,
                           rsi_len=14, trend_len=50):
    """Dynamic SR Pivot Strategy [felipemiransan]
    Source: https://www.tradingview.com/script/OKq8iEzr/
    Concept: Dynamic S/R from recent pivots; buy on breakout with trend + RSI filter.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    atr = _atr(df, 14)
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, trend_len)

    ph = high.rolling(pivot_left + pivot_right + 1).max().shift(pivot_right)
    pl = low.rolling(pivot_left + pivot_right + 1).min().shift(pivot_right)
    buf = atr * atr_mult * 0.1

    long_sig = (close > ph + buf) & (close > trend) & (rsi > 45)
    short_sig = (close < pl - buf) & (close < trend) & (rsi < 55)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_DynamicSRPivot(trial):
    return {
        'pivot_left': trial.suggest_int('pivot_left', 5, 25),
        'pivot_right': trial.suggest_int('pivot_right', 2, 10),
        'atr_mult': trial.suggest_float('atr_mult', 0.3, 2.0),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
    }


def gen_TV_EMA920_SRBreakout(df, ema_fast=9, ema_slow=20, pivot_left=10,
                               pivot_right=5, rsi_len=14):
    """EMA 9/20 with SR Breakout [ankurb171991]
    Source: https://www.tradingview.com/script/5EGVP61U-EMA-9-20-with-Support-and-Resistance-Breakout/
    Concept: EMA 9/20 crossover confirms trend; pivot S/R breakout is the entry trigger.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    ema9 = _ema(close, ema_fast)
    ema20 = _ema(close, ema_slow)
    rsi = _rsi(close, rsi_len)

    ph = high.rolling(pivot_left + pivot_right + 1).max().shift(pivot_right)
    pl = low.rolling(pivot_left + pivot_right + 1).min().shift(pivot_right)

    bull_trend = ema9 > ema20
    long_sig = (close > ph) & bull_trend & (rsi > 50)
    short_sig = (close < pl) & ~bull_trend & (rsi < 50)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_EMA920_SRBreakout(trial):
    return {
        'ema_fast': trial.suggest_int('ema_fast', 5, 15),
        'ema_slow': trial.suggest_int('ema_slow', 15, 50),
        'pivot_left': trial.suggest_int('pivot_left', 5, 20),
        'pivot_right': trial.suggest_int('pivot_right', 2, 10),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
    }


# ---------------------------------------------------------------------------
# BATCH028 — PIVOT POINT SYSTEMS (8)
# ---------------------------------------------------------------------------

def gen_TV_PivotPointSuperTrend(df, ph_left=2, ph_right=2, atr_mult=3.0,
                                 atr_len=14, rsi_len=14):
    """Pivot Point SuperTrend Backtest [LonesomeTheBlue]
    Source: https://www.tradingview.com/script/DwdC6FT4-Pivot-Point-SuperTrend-Backtest/
    Concept: SuperTrend built from pivot highs/lows instead of simple HL2.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    rsi = _rsi(close, rsi_len)
    atr = _atr(df, atr_len)

    ph = high.rolling(ph_left + ph_right + 1).max().shift(ph_right)
    pl = low.rolling(ph_left + ph_right + 1).min().shift(ph_right)
    pivot_mid = (ph + pl) / 2

    upper_band = pivot_mid + atr_mult * atr
    lower_band = pivot_mid - atr_mult * atr

    trend = pd.Series(1, index=close.index)
    for i in range(1, len(close)):
        if close.iloc[i] > upper_band.iloc[i - 1]:
            trend.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]

    flip_up = (trend == 1) & (trend.shift(1) == -1)
    flip_dn = (trend == -1) & (trend.shift(1) == 1)

    sig = pd.Series(0, index=df.index)
    sig[flip_up & (rsi > 40)] = 1
    sig[flip_dn & (rsi < 60)] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_PivotPointSuperTrend(trial):
    return {
        'ph_left': trial.suggest_int('ph_left', 1, 5),
        'ph_right': trial.suggest_int('ph_right', 1, 5),
        'atr_mult': trial.suggest_float('atr_mult', 1.5, 6.0),
        'atr_len': trial.suggest_int('atr_len', 7, 21),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
    }


def gen_TV_CamarillaPivots(df, session_len=1440, h3_factor=1.0833,
                            l3_factor=1.0833, rsi_len=14, trend_len=50):
    """Camarilla Pivot Points Backtest [HPotter]
    Source: https://www.tradingview.com/script/pZshc5OY-Camarilla-Pivot-Points-Backtest/
    Concept: Camarilla H3/L3 levels from previous day range; trade mean-reversion at H3/L3.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, trend_len)

    prev_h = high.rolling(session_len, min_periods=1).max().shift(1)
    prev_l = low.rolling(session_len, min_periods=1).min().shift(1)
    prev_c = close.shift(1)
    rng = prev_h - prev_l

    h3 = prev_c + rng * (h3_factor - 1)
    l3 = prev_c - rng * (l3_factor - 1)

    long_sig = (close <= l3) & (close > trend.shift(1)) & (rsi < 40)
    short_sig = (close >= h3) & (close < trend.shift(1)) & (rsi > 60)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_CamarillaPivots(trial):
    return {
        'session_len': trial.suggest_categorical('session_len', [288, 1440]),
        'h3_factor': trial.suggest_float('h3_factor', 1.06, 1.12),
        'l3_factor': trial.suggest_float('l3_factor', 1.06, 1.12),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
    }


def gen_TV_CamarillaPivotsV2(df, session_len=1440, h4_factor=1.1,
                               l4_factor=1.1, rsi_len=14, atr_mult=1.0):
    """Camarilla Pivot Points V2 Backtest [HPotter]
    Source: https://www.tradingview.com/script/MXkclJVM-Camarilla-Pivot-Points-V2-Backtest/
    Concept: H4/L4 Camarilla levels — stronger than H3/L3; breakout above H4 = breakout trade.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    atr = _atr(df, 14)
    rsi = _rsi(close, rsi_len)

    prev_h = high.rolling(session_len, min_periods=1).max().shift(1)
    prev_l = low.rolling(session_len, min_periods=1).min().shift(1)
    prev_c = close.shift(1)
    rng = prev_h - prev_l

    h4 = prev_c + rng * (h4_factor - 1) * 1.5
    l4 = prev_c - rng * (l4_factor - 1) * 1.5
    buf = atr * atr_mult * 0.05

    long_sig = (close > h4 + buf) & (rsi > 50)
    short_sig = (close < l4 - buf) & (rsi < 50)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_CamarillaPivotsV2(trial):
    return {
        'session_len': trial.suggest_categorical('session_len', [288, 1440]),
        'h4_factor': trial.suggest_float('h4_factor', 1.07, 1.15),
        'l4_factor': trial.suggest_float('l4_factor', 1.07, 1.15),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'atr_mult': trial.suggest_float('atr_mult', 0.3, 2.0),
    }


def gen_TV_PivotPointV2(df, session_len=1440, r1_w=0.5, rsi_len=14,
                         trend_len=50, atr_mult=1.0):
    """Pivot Point V2 Backtest [HPotter]
    Source: https://www.tradingview.com/script/SIQgcWq2-Pivot-Point-V2-Backtest/
    Concept: Classic pivot point (HLC/3); R1/S1 as mean-reversion targets.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    atr = _atr(df, 14)
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, trend_len)

    prev_h = high.rolling(session_len, min_periods=1).max().shift(1)
    prev_l = low.rolling(session_len, min_periods=1).min().shift(1)
    prev_c = close.shift(1)
    pp = (prev_h + prev_l + prev_c) / 3
    r1 = pp + r1_w * (prev_h - prev_l)
    s1 = pp - r1_w * (prev_h - prev_l)
    buf = atr * atr_mult * 0.05

    long_sig = (close <= s1 + buf) & (close > trend.shift(1)) & (rsi < 45)
    short_sig = (close >= r1 - buf) & (close < trend.shift(1)) & (rsi > 55)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_PivotPointV2(trial):
    return {
        'session_len': trial.suggest_categorical('session_len', [288, 1440]),
        'r1_w': trial.suggest_float('r1_w', 0.3, 1.0),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'atr_mult': trial.suggest_float('atr_mult', 0.2, 2.0),
    }


def gen_TV_WoodiePivots(df, session_len=1440, rsi_len=14, trend_len=50,
                         vol_mult=1.0, atr_mult=1.0):
    """Woodie Pivot Points Backtest [HPotter]
    Source: https://www.tradingview.com/script/ZyNj4Ewl-Woodie-Pivot-Points-Backtest/
    Concept: Woodie pivot = (2*prev_close + prev_high + prev_low) / 4;
    trade bounces at R1/S1 with volume and RSI confirmation.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    vol = df['volume']
    atr = _atr(df, 14)
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, trend_len)
    vol_avg = vol.rolling(20).mean()

    prev_h = high.rolling(session_len, min_periods=1).max().shift(1)
    prev_l = low.rolling(session_len, min_periods=1).min().shift(1)
    prev_c = close.shift(1)
    wp = (2 * prev_c + prev_h + prev_l) / 4
    r1 = 2 * wp - prev_l
    s1 = 2 * wp - prev_h
    buf = atr * atr_mult * 0.05

    long_sig = (close <= s1 + buf) & (close > trend) & (rsi < 45) & (vol > vol_avg * vol_mult)
    short_sig = (close >= r1 - buf) & (close < trend) & (rsi > 55) & (vol > vol_avg * vol_mult)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_WoodiePivots(trial):
    return {
        'session_len': trial.suggest_categorical('session_len', [288, 1440]),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'vol_mult': trial.suggest_float('vol_mult', 0.5, 2.5),
        'atr_mult': trial.suggest_float('atr_mult', 0.2, 2.0),
    }


def gen_TV_DynamicPivot(df, pivot_left=5, pivot_right=5, rsi_len=14,
                         trend_len=50, atr_mult=1.0):
    """Dynamic Pivot Point Backtest [HPotter]
    Source: https://www.tradingview.com/script/MiFxDDNW-Dynamic-Pivot-Point-Backtest/
    Concept: Real-time pivot detection (not session-based); breakout above pivot high.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    atr = _atr(df, 14)
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, trend_len)

    ph = high.rolling(pivot_left + pivot_right + 1).max().shift(pivot_right)
    pl = low.rolling(pivot_left + pivot_right + 1).min().shift(pivot_right)
    buf = atr * atr_mult * 0.1

    long_sig = (close > ph + buf) & (close > trend) & (rsi > 50)
    short_sig = (close < pl - buf) & (close < trend) & (rsi < 50)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_DynamicPivot(trial):
    return {
        'pivot_left': trial.suggest_int('pivot_left', 3, 15),
        'pivot_right': trial.suggest_int('pivot_right', 2, 10),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'atr_mult': trial.suggest_float('atr_mult', 0.3, 2.0),
    }


def gen_TV_FloorPivots(df, session_len=1440, rsi_len=14, trend_len=50,
                        vol_mult=1.0, atr_mult=1.0):
    """Floor Pivot Points Backtest [HPotter]
    Source: https://www.tradingview.com/script/4UlDByOA-Floor-Pivot-Points-Backtest/
    Concept: Classic floor pivots (PP, R1/R2, S1/S2); bounce or break signals.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    vol = df['volume']
    atr = _atr(df, 14)
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, trend_len)
    vol_avg = vol.rolling(20).mean()

    prev_h = high.rolling(session_len, min_periods=1).max().shift(1)
    prev_l = low.rolling(session_len, min_periods=1).min().shift(1)
    prev_c = close.shift(1)
    pp = (prev_h + prev_l + prev_c) / 3
    r1 = 2 * pp - prev_l
    s1 = 2 * pp - prev_h
    buf = atr * atr_mult * 0.05

    long_sig = (close > r1 + buf) & (close > trend) & (rsi > 50) & (vol > vol_avg * vol_mult)
    short_sig = (close < s1 - buf) & (close < trend) & (rsi < 50) & (vol > vol_avg * vol_mult)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_FloorPivots(trial):
    return {
        'session_len': trial.suggest_categorical('session_len', [288, 1440]),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'vol_mult': trial.suggest_float('vol_mult', 0.5, 2.5),
        'atr_mult': trial.suggest_float('atr_mult', 0.2, 2.0),
    }


def gen_TV_PivotSuperTrendTrendFilter(df, ph_left=2, ph_right=2, atr_mult=3.0,
                                       trend_len=200, rsi_len=14):
    """Pivot Point SuperTrend + TrendFilter [Julien_Eche]
    Source: https://www.tradingview.com/script/WpJ5ym7u-Pivot-Point-SuperTrend-Strategy-TrendFilter/
    Concept: Pivot-based SuperTrend + HTF trend filter (SMA200) to avoid counter-trend trades.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    rsi = _rsi(close, rsi_len)
    atr = _atr(df, 14)
    sma_trend = close.rolling(trend_len).mean()

    ph = high.rolling(ph_left + ph_right + 1).max().shift(ph_right)
    pl = low.rolling(ph_left + ph_right + 1).min().shift(ph_right)
    pivot_mid = (ph + pl) / 2
    upper_band = pivot_mid + atr_mult * atr
    lower_band = pivot_mid - atr_mult * atr

    trend = pd.Series(1, index=close.index)
    for i in range(1, len(close)):
        if close.iloc[i] > upper_band.iloc[i - 1]:
            trend.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]

    flip_up = (trend == 1) & (trend.shift(1) == -1)
    flip_dn = (trend == -1) & (trend.shift(1) == 1)
    bull_bias = close > sma_trend
    bear_bias = close < sma_trend

    sig = pd.Series(0, index=df.index)
    sig[flip_up & bull_bias & (rsi > 40)] = 1
    sig[flip_dn & bear_bias & (rsi < 60)] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_PivotSuperTrendTrendFilter(trial):
    return {
        'ph_left': trial.suggest_int('ph_left', 1, 5),
        'ph_right': trial.suggest_int('ph_right', 1, 5),
        'atr_mult': trial.suggest_float('atr_mult', 1.5, 6.0),
        'trend_len': trial.suggest_int('trend_len', 100, 300),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
    }


# ---------------------------------------------------------------------------
# BATCH028 — SUPPLY AND DEMAND ZONES (5)
# ---------------------------------------------------------------------------

def gen_TV_SupplyDemandStrategy(df, zone_len=10, atr_mult=1.5, rsi_len=14,
                                  trend_len=50, vol_mult=1.0):
    """Supply and Demand Strategy [mohammadhasanmohammadi]
    Source: https://www.tradingview.com/script/gHhqbgsG-Supply-and-Demand-Strategy/
    Concept: Swing high/low zones define supply (sell) and demand (buy) areas;
    price re-entering zone = potential reversal entry.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    vol = df['volume']
    atr = _atr(df, 14)
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, trend_len)
    vol_avg = vol.rolling(20).mean()

    swing_h = high.rolling(zone_len * 2 + 1).max().shift(zone_len)
    swing_l = low.rolling(zone_len * 2 + 1).min().shift(zone_len)
    demand_top = swing_l + atr * atr_mult * 0.2
    supply_bot = swing_h - atr * atr_mult * 0.2

    long_sig = (close >= swing_l) & (close <= demand_top) & (close > trend) & (rsi < 50) & (vol >= vol_avg * vol_mult)
    short_sig = (close <= swing_h) & (close >= supply_bot) & (close < trend) & (rsi > 50) & (vol >= vol_avg * vol_mult)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_SupplyDemandStrategy(trial):
    return {
        'zone_len': trial.suggest_int('zone_len', 5, 25),
        'atr_mult': trial.suggest_float('atr_mult', 0.5, 3.0),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'vol_mult': trial.suggest_float('vol_mult', 0.5, 2.0),
    }


def gen_TV_SupplyDemandEnhanced(df, zone_len=10, vol_mult=1.5, rsi_len=14,
                                  atr_mult=1.0, trend_len=50):
    """Supply and Demand Zones Enhanced Signals [IJAlgo]
    Source: https://www.tradingview.com/script/nQcfcgN4-Supply-and-Demand-Zones-with-Enhanced-Signals/
    Concept: Enhanced SD zones with volume spike detection; high-volume candle defines zone base.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    vol = df['volume']
    atr = _atr(df, 14)
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, trend_len)
    vol_avg = vol.rolling(zone_len).mean()

    hi_vol_bar = vol > vol_avg * vol_mult
    demand_base = low.where(hi_vol_bar, np.nan).rolling(zone_len * 2).min().ffill()
    supply_base = high.where(hi_vol_bar, np.nan).rolling(zone_len * 2).max().ffill()
    buf = atr * atr_mult * 0.1

    long_sig = (close >= demand_base - buf) & (close <= demand_base + buf * 3) & (close > trend) & (rsi < 55)
    short_sig = (close <= supply_base + buf) & (close >= supply_base - buf * 3) & (close < trend) & (rsi > 45)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_SupplyDemandEnhanced(trial):
    return {
        'zone_len': trial.suggest_int('zone_len', 5, 25),
        'vol_mult': trial.suggest_float('vol_mult', 1.0, 3.0),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'atr_mult': trial.suggest_float('atr_mult', 0.3, 2.0),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
    }


def gen_TV_SupplyDemandCleanV6(df, zone_len=15, atr_mult=1.0, rsi_len=14,
                                  trend_len=50, vol_mult=1.2):
    """Supply and Demand Zones Clean v6 [CaseInTheMoney]
    Source: https://www.tradingview.com/script/VDv1AnDA-Supply-and-Demand-Zones-Clean-v6/
    Concept: Clean SD zone detection using candle body vs wick ratio to identify base candles.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    op = df['open']
    vol = df['volume']
    atr = _atr(df, 14)
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, trend_len)
    vol_avg = vol.rolling(20).mean()

    body = (close - op).abs()
    wick = (high - low)
    base_candle = (body < wick * 0.5) & (vol > vol_avg * vol_mult)

    demand_zone = low.where(base_candle, np.nan).rolling(zone_len).min().ffill()
    supply_zone = high.where(base_candle, np.nan).rolling(zone_len).max().ffill()
    buf = atr * atr_mult * 0.1

    long_sig = (close >= demand_zone - buf) & (close <= demand_zone + buf * 2) & (close > trend) & (rsi < 55)
    short_sig = (close <= supply_zone + buf) & (close >= supply_zone - buf * 2) & (close < trend) & (rsi > 45)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_SupplyDemandCleanV6(trial):
    return {
        'zone_len': trial.suggest_int('zone_len', 8, 30),
        'atr_mult': trial.suggest_float('atr_mult', 0.3, 2.0),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'vol_mult': trial.suggest_float('vol_mult', 0.8, 2.5),
    }


def gen_TV_SupplyDemandPro(df, zone_len=20, atr_mult=1.0, vol_mult=1.5,
                             rsi_len=14, trend_len=50):
    """Supply Demand Zones Pro [ProjectSyndicate]
    Source: https://www.tradingview.com/script/h0jxhmgn-Supply-Demand-Zones-Pro/
    Concept: Pro-level SD zones using consecutive bullish/bearish candles to define base.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    op = df['open']
    vol = df['volume']
    atr = _atr(df, 14)
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, trend_len)
    vol_avg = vol.rolling(zone_len).mean()

    bull_candle = close > op
    bear_candle = close < op
    base_bull = bull_candle & (vol > vol_avg * vol_mult)
    base_bear = bear_candle & (vol > vol_avg * vol_mult)

    demand_lo = low.where(base_bull, np.nan).rolling(zone_len).min().ffill()
    demand_hi = close.where(base_bull, np.nan).rolling(zone_len).min().ffill()
    supply_hi = high.where(base_bear, np.nan).rolling(zone_len).max().ffill()
    supply_lo = close.where(base_bear, np.nan).rolling(zone_len).max().ffill()
    buf = atr * atr_mult * 0.05

    long_sig = (close >= demand_lo - buf) & (close <= demand_hi + buf) & (close > trend) & (rsi < 55)
    short_sig = (close <= supply_hi + buf) & (close >= supply_lo - buf) & (close < trend) & (rsi > 45)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_SupplyDemandPro(trial):
    return {
        'zone_len': trial.suggest_int('zone_len', 10, 40),
        'atr_mult': trial.suggest_float('atr_mult', 0.3, 2.0),
        'vol_mult': trial.suggest_float('vol_mult', 1.0, 3.0),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
    }


def gen_TV_AGProSupplyDemand(df, zone_len=10, swing_len=5, atr_mult=1.5,
                               rsi_len=14, trend_len=50):
    """AG Pro Auto Supply and Demand Zones [AGProLabs]
    Source: https://www.tradingview.com/script/rcbLhPjF-AG-Pro-Auto-Supply-Demand-Zones-AGPro-Series/
    Concept: Auto-detected SD zones based on swing structure; price re-entry at zone = signal.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    atr = _atr(df, 14)
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, trend_len)

    swing_h = high.rolling(swing_len * 2 + 1).max().shift(swing_len)
    swing_l = low.rolling(swing_len * 2 + 1).min().shift(swing_len)
    demand_base = swing_l.rolling(zone_len).min()
    supply_base = swing_h.rolling(zone_len).max()
    buf = atr * atr_mult * 0.1

    long_sig = (close >= demand_base - buf) & (close <= demand_base + buf * 2) & (close > trend) & (rsi < 55)
    short_sig = (close <= supply_base + buf) & (close >= supply_base - buf * 2) & (close < trend) & (rsi > 45)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_AGProSupplyDemand(trial):
    return {
        'zone_len': trial.suggest_int('zone_len', 5, 25),
        'swing_len': trial.suggest_int('swing_len', 3, 15),
        'atr_mult': trial.suggest_float('atr_mult', 0.5, 3.0),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
    }


# ---------------------------------------------------------------------------
# BATCH028 — MARKET STRUCTURE BOS/CHoCH (5)
# ---------------------------------------------------------------------------

def gen_TV_MarketStructureBOSCHoCH(df, swing_len=5, atr_mult=1.0, rsi_len=14,
                                    trend_len=50, vol_mult=1.0):
    """Market Structure BOS/CHoCH/FVG/OB [nephew_sam_]
    Source: https://www.tradingview.com/script/PmIP7HAZ-Market-Structure-BOS-CHOCH-MSB-FVG-OB-BB-Nephew-Sam/
    Concept: Break of Structure (BOS) = continuation signal; Change of Character (CHoCH) = reversal.
    Long on CHoCH up (prev down trend now creates HH), short on CHoCH down.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    vol = df['volume']
    atr = _atr(df, 14)
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, trend_len)
    vol_avg = vol.rolling(20).mean()

    swing_h = high.rolling(swing_len * 2 + 1).max().shift(swing_len)
    swing_l = low.rolling(swing_len * 2 + 1).min().shift(swing_len)

    hh = swing_h > swing_h.shift(swing_len)
    ll = swing_l < swing_l.shift(swing_len)
    hl = swing_l > swing_l.shift(swing_len)
    lh = swing_h < swing_h.shift(swing_len)

    choch_up = hl & lh.shift(1)
    choch_dn = lh & hl.shift(1)
    bos_up = hh & close > swing_h.shift(1)
    bos_dn = ll & close < swing_l.shift(1)

    buf = atr * atr_mult * 0.1
    long_sig = (choch_up | bos_up) & (close > trend) & (rsi > 45) & (vol > vol_avg * vol_mult) & (close > swing_h.shift(1) - buf)
    short_sig = (choch_dn | bos_dn) & (close < trend) & (rsi < 55) & (vol > vol_avg * vol_mult) & (close < swing_l.shift(1) + buf)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_MarketStructureBOSCHoCH(trial):
    return {
        'swing_len': trial.suggest_int('swing_len', 3, 15),
        'atr_mult': trial.suggest_float('atr_mult', 0.3, 2.0),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'vol_mult': trial.suggest_float('vol_mult', 0.5, 2.5),
    }


def gen_TV_CHoCHBOSFractal(df, swing_len=5, fractal_left=3, rsi_len=14,
                             trend_len=50, atr_mult=1.0):
    """Market Structure CHoCH/BOS Fractal [LuxAlgo]
    Source: https://www.tradingview.com/script/ZpHqSrBK-Market-Structure-CHoCH-BOS-Fractal-LuxAlgo/
    Concept: Fractal-based pivot detection for BOS/CHoCH identification.
    Fractal high = bar is highest in fractal_left bars on both sides.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    atr = _atr(df, 14)
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, trend_len)

    win = fractal_left * 2 + 1
    frac_h = high.rolling(win).max().shift(fractal_left)
    frac_l = low.rolling(win).min().shift(fractal_left)

    prev_frac_h = frac_h.shift(swing_len)
    prev_frac_l = frac_l.shift(swing_len)

    bos_up = (close > frac_h.shift(1)) & (frac_h > prev_frac_h)
    bos_dn = (close < frac_l.shift(1)) & (frac_l < prev_frac_l)
    choch_up = (close > frac_h.shift(1)) & (frac_l > prev_frac_l)
    choch_dn = (close < frac_l.shift(1)) & (frac_h < prev_frac_h)

    buf = atr * atr_mult * 0.05
    long_sig = (bos_up | choch_up) & (close > trend) & (rsi > 45)
    short_sig = (bos_dn | choch_dn) & (close < trend) & (rsi < 55)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_CHoCHBOSFractal(trial):
    return {
        'swing_len': trial.suggest_int('swing_len', 3, 15),
        'fractal_left': trial.suggest_int('fractal_left', 2, 7),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'atr_mult': trial.suggest_float('atr_mult', 0.2, 2.0),
    }


def gen_TV_MTFBreakOfStructure(df, swing_len=10, atr_mult=1.0, rsi_len=14,
                                 trend_len=50, vol_mult=1.0):
    """MTF Break of Structure + MSS [Lenny_Kiruthu]
    Source: https://www.tradingview.com/script/1LrVbh1I-MTF-Break-of-Structure-BOS-Market-Structure-Shift-MSS/
    Concept: Multi-timeframe BOS — uses longer swing_len to capture higher-timeframe structure breaks.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    vol = df['volume']
    atr = _atr(df, 14)
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, trend_len)
    vol_avg = vol.rolling(20).mean()

    swing_h = high.rolling(swing_len * 2 + 1).max().shift(swing_len)
    swing_l = low.rolling(swing_len * 2 + 1).min().shift(swing_len)

    bos_up = (close > swing_h.shift(1)) & (close > trend)
    bos_dn = (close < swing_l.shift(1)) & (close < trend)
    mss_up = bos_up & (swing_h > swing_h.shift(swing_len))
    mss_dn = bos_dn & (swing_l < swing_l.shift(swing_len))

    buf = atr * atr_mult * 0.1
    long_sig = (mss_up) & (rsi > 45) & (vol > vol_avg * vol_mult)
    short_sig = (mss_dn) & (rsi < 55) & (vol > vol_avg * vol_mult)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_MTFBreakOfStructure(trial):
    return {
        'swing_len': trial.suggest_int('swing_len', 5, 25),
        'atr_mult': trial.suggest_float('atr_mult', 0.3, 2.0),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'vol_mult': trial.suggest_float('vol_mult', 0.5, 2.5),
    }


def gen_TV_DrawOnLiquidity(df, swing_len=10, liq_mult=1.5, rsi_len=14,
                             trend_len=50, atr_mult=1.0):
    """Draw on Liquidity [PhenLabs]
    Source: https://www.tradingview.com/script/b66E7VZK-Draw-on-Liquidity-PhenLabs/
    Concept: Liquidity pools at swing highs/lows; price sweeps (false breakout) then reverses.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    atr = _atr(df, 14)
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, trend_len)

    swing_h = high.rolling(swing_len * 2 + 1).max().shift(swing_len)
    swing_l = low.rolling(swing_len * 2 + 1).min().shift(swing_len)
    liq_h = swing_h * liq_mult / (liq_mult - 0.001)
    liq_l = swing_l * (liq_mult - 0.001) / liq_mult
    buf = atr * atr_mult * 0.2

    liquidity_sweep_up = (high > swing_h.shift(1)) & (close < swing_h.shift(1) + buf)
    liquidity_sweep_dn = (low < swing_l.shift(1)) & (close > swing_l.shift(1) - buf)

    long_sig = liquidity_sweep_dn & (close > trend) & (rsi < 55)
    short_sig = liquidity_sweep_up & (close < trend) & (rsi > 45)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_DrawOnLiquidity(trial):
    return {
        'swing_len': trial.suggest_int('swing_len', 5, 20),
        'liq_mult': trial.suggest_float('liq_mult', 1.001, 2.0),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'atr_mult': trial.suggest_float('atr_mult', 0.1, 1.5),
    }


def gen_TV_MarketStructureZigZagBOS(df, swing_len=5, zz_dev=0.05, rsi_len=14,
                                     trend_len=50, atr_mult=1.0):
    """Market Structure ZigZag + BOS + OB [The_Forex_Steward]
    Source: https://www.tradingview.com/script/NXgQEbZi-Market-Structure-ZigZag-Break-of-Structure-Order-Blocks/
    Concept: ZigZag-based market structure; BOS confirmed when close exceeds last ZZ pivot.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    atr = _atr(df, 14)
    rsi = _rsi(close, rsi_len)
    trend = _ema(close, trend_len)

    swing_h = high.rolling(swing_len * 2 + 1).max().shift(swing_len)
    swing_l = low.rolling(swing_len * 2 + 1).min().shift(swing_len)

    dev_h = swing_h * (1 - zz_dev)
    dev_l = swing_l * (1 + zz_dev)

    bos_up = (close > swing_h.shift(1)) & (close > dev_h)
    bos_dn = (close < swing_l.shift(1)) & (close < dev_l)

    long_sig = bos_up & (close > trend) & (rsi > 45)
    short_sig = bos_dn & (close < trend) & (rsi < 55)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig.shift(1).fillna(0).astype(int)


def _space_TV_MarketStructureZigZagBOS(trial):
    return {
        'swing_len': trial.suggest_int('swing_len', 3, 15),
        'zz_dev': trial.suggest_float('zz_dev', 0.01, 0.15),
        'rsi_len': trial.suggest_int('rsi_len', 7, 21),
        'trend_len': trial.suggest_int('trend_len', 20, 100),
        'atr_mult': trial.suggest_float('atr_mult', 0.2, 2.0),
    }


# ---------------------------------------------------------------------------
# STRATEGY_EXPORT
# ---------------------------------------------------------------------------

STRATEGY_EXPORT = {
    'TV_TrueSeasonalPattern': {
        'gen': gen_TV_TrueSeasonalPattern,
        'space': _space_TV_TrueSeasonalPattern,
        'source': 'https://www.tradingview.com/script/SijvaWFx-True-Seasonal-Pattern-tradeviZion/',
        'default_params': {'lookback_years': 3, 'entry_month_offset': 0, 'exit_month_offset': 1, 'trend_len': 50, 'rsi_len': 14},
    },
    'TV_SeasonalityPresidentialCycle': {
        'gen': gen_TV_SeasonalityPresidentialCycle,
        'space': _space_TV_SeasonalityPresidentialCycle,
        'source': 'https://www.tradingview.com/script/psG4Vj0S-seasonality-and-presidential-cycle/',
        'default_params': {'pres_cycle_len': 48, 'trend_len': 50, 'rsi_len': 14, 'momentum_len': 20, 'adx_len': 14},
    },
    'TV_IntradaySeasonality': {
        'gen': gen_TV_IntradaySeasonality,
        'space': _space_TV_IntradaySeasonality,
        'source': 'https://www.tradingview.com/script/OuD7PG2n-Intraday-Seasonality/',
        'default_params': {'session_open_hour': 9, 'session_close_hour': 16, 'trend_len': 20, 'rsi_len': 10, 'vol_mult': 1.5},
    },
    'TV_SeasonalityForecast': {
        'gen': gen_TV_SeasonalityForecast,
        'space': _space_TV_SeasonalityForecast,
        'source': 'https://www.tradingview.com/script/OhNg5Yk6-Seasonality-Forecast/',
        'default_params': {'forecast_len': 5, 'trend_len': 50, 'rsi_len': 14, 'vol_len': 20, 'adx_len': 14},
    },
    'TV_EnhancedSeasonalityBacktest': {
        'gen': gen_TV_EnhancedSeasonalityBacktest,
        'space': _space_TV_EnhancedSeasonalityBacktest,
        'source': 'https://www.tradingview.com/script/ztSwRLRk/',
        'default_params': {'lookback': 100, 'z_thresh': 0.5, 'trend_len': 50, 'rsi_len': 14, 'vol_mult': 1.2},
    },
    'TV_MonthlyPerformanceStrategy': {
        'gen': gen_TV_MonthlyPerformanceStrategy,
        'space': _space_TV_MonthlyPerformanceStrategy,
        'source': 'https://www.tradingview.com/script/DR2MFR41-Unlock-the-Power-of-Seasonality-Monthly-Performance-Strategy/',
        'default_params': {'lookback_months': 12, 'trend_len': 50, 'rsi_len': 14, 'adx_thresh': 20, 'vol_len': 20},
    },
    'TV_SeasonalityChartingCycles': {
        'gen': gen_TV_SeasonalityChartingCycles,
        'space': _space_TV_SeasonalityChartingCycles,
        'source': 'https://www.tradingview.com/script/i1TZiqVf-Seasonality/',
        'default_params': {'cycle_len': 20, 'trend_len': 50, 'rsi_len': 14, 'momentum_len': 10, 'vol_mult': 1.2},
    },
    'TV_SeasonalityMonthHighlight': {
        'gen': gen_TV_SeasonalityMonthHighlight,
        'space': _space_TV_SeasonalityMonthHighlight,
        'source': 'https://www.tradingview.com/script/8uBdh16P-Seasonality-Month-Highlight/',
        'default_params': {'bullish_months_str': 4, 'bearish_months_str': 9, 'trend_len': 50, 'rsi_len': 14, 'adx_len': 14},
    },
    'TV_SeasonalTendency': {
        'gen': gen_TV_SeasonalTendency,
        'space': _space_TV_SeasonalTendency,
        'source': 'https://www.tradingview.com/script/9h4Tj6sM-Seasonal-Tendency-fadi/',
        'default_params': {'lookback': 200, 'quantile_thresh': 0.6, 'trend_len': 50, 'rsi_len': 14, 'vol_len': 20},
    },
    'TV_SeasonalityCustomInterval': {
        'gen': gen_TV_SeasonalityCustomInterval,
        'space': _space_TV_SeasonalityCustomInterval,
        'source': 'https://www.tradingview.com/script/3vlvXQq1-Seasonality-with-Custom-Interval/',
        'default_params': {'interval_bars': 24, 'trend_len': 50, 'rsi_len': 14, 'vol_mult': 1.3, 'lookback': 50},
    },
    'TV_MESAAdaptiveEhlersFlow': {
        'gen': gen_TV_MESAAdaptiveEhlersFlow,
        'space': _space_TV_MESAAdaptiveEhlersFlow,
        'source': 'https://www.tradingview.com/script/9SZSadPm-MESA-Adaptive-Ehlers-Flow-AlphaNatt/',
        'default_params': {'fast_limit': 0.5, 'slow_limit': 0.05, 'trend_len': 50, 'rsi_len': 14, 'adx_len': 14},
    },
    'TV_EhlersMESAAdaptiveMA': {
        'gen': gen_TV_EhlersMESAAdaptiveMA,
        'space': _space_TV_EhlersMESAAdaptiveMA,
        'source': 'https://www.tradingview.com/script/foQxLbU3-Ehlers-MESA-Adaptive-Moving-Average-LazyBear/',
        'default_params': {'fast_limit': 0.5, 'slow_limit': 0.05, 'trend_len': 50, 'rsi_len': 14, 'vol_mult': 1.2},
    },
    'TV_DominantCycleLibrary': {
        'gen': gen_TV_DominantCycleLibrary,
        'space': _space_TV_DominantCycleLibrary,
        'source': 'https://www.tradingview.com/script/cY7DdxyZ-DominantCycle/',
        'default_params': {'min_period': 10, 'max_period': 48, 'trend_len': 50, 'rsi_len': 14, 'adx_len': 14},
    },
    'TV_EhlersAdaptiveCyberCycle': {
        'gen': gen_TV_EhlersAdaptiveCyberCycle,
        'space': _space_TV_EhlersAdaptiveCyberCycle,
        'source': 'https://www.tradingview.com/script/3lV1e3ci-Ehlers-Adaptive-Cyber-Cycle-Indicator-LazyBear/',
        'default_params': {'alpha': 0.07, 'overbought': 0.5, 'oversold': -0.5, 'trend_len': 50, 'rsi_len': 14},
    },
    'TV_EhlersSimpleCycleIndicator': {
        'gen': gen_TV_EhlersSimpleCycleIndicator,
        'space': _space_TV_EhlersSimpleCycleIndicator,
        'source': 'https://www.tradingview.com/script/xQ4mP4kc-Ehlers-Simple-Cycle-Indicator-LazyBear/',
        'default_params': {'alpha': 0.07, 'trend_len': 50, 'rsi_len': 14, 'vol_mult': 1.2, 'adx_len': 14},
    },
    'TV_InverseFisherAdaptiveStoch': {
        'gen': gen_TV_InverseFisherAdaptiveStoch,
        'space': _space_TV_InverseFisherAdaptiveStoch,
        'source': 'https://www.tradingview.com/script/uMTi9oYD-inverse-fisher-transform-adaptive-stochastic/',
        'default_params': {'stoch_len': 14, 'smooth_k': 3, 'ift_thresh': 0.5, 'trend_len': 50, 'rsi_len': 14},
    },
    'TV_InverseFisherStoch': {
        'gen': gen_TV_InverseFisherStoch,
        'space': _space_TV_InverseFisherStoch,
        'source': 'https://www.tradingview.com/script/WikDwOZC/',
        'default_params': {'stoch_len': 14, 'smooth_k': 3, 'signal_len': 9, 'trend_len': 50, 'rsi_len': 14},
    },
    'TV_AdaptiveFisherizedCMO': {
        'gen': gen_TV_AdaptiveFisherizedCMO,
        'space': _space_TV_AdaptiveFisherizedCMO,
        'source': 'https://www.tradingview.com/script/f9bz3PcY-Adaptive-Fisherized-CMO/',
        'default_params': {'cmo_len': 14, 'fisher_len': 10, 'ift_thresh': 0.3, 'trend_len': 50, 'adx_len': 14},
    },
    'TV_AdaptiveFisherizedROC': {
        'gen': gen_TV_AdaptiveFisherizedROC,
        'space': _space_TV_AdaptiveFisherizedROC,
        'source': 'https://www.tradingview.com/script/Ufbx4DRV-Adaptive-Fisherized-ROC/',
        'default_params': {'roc_len': 10, 'fisher_len': 10, 'ift_thresh': 0.2, 'trend_len': 50, 'rsi_len': 14},
    },
    'TV_FisherTransformEhlersBacktest': {
        'gen': gen_TV_FisherTransformEhlersBacktest,
        'space': _space_TV_FisherTransformEhlersBacktest,
        'source': 'https://www.tradingview.com/script/qhXwZvJ2-Fisher-Transform-Indicator-by-Ehlers-Backtest-v-2-0/',
        'default_params': {'fisher_len': 10, 'signal_len': 1, 'trend_len': 50, 'rsi_len': 14, 'adx_len': 14},
    },
    'TV_MLEnhancedCompositeSignal': {
        'gen': gen_TV_MLEnhancedCompositeSignal,
        'space': _space_TV_MLEnhancedCompositeSignal,
        'source': 'https://www.tradingview.com/script/vofMjW64-ML-Enhanced-Multi-Indicator-Composite-Signal/',
        'default_params': {'rsi_len': 14, 'macd_fast': 12, 'macd_slow': 26, 'macd_sig': 9, 'adx_len': 14},
    },
    'TV_CompositeBuySellScore': {
        'gen': gen_TV_CompositeBuySellScore,
        'space': _space_TV_CompositeBuySellScore,
        'source': 'https://www.tradingview.com/script/DFgw9vm5-Composite-Buy-Sell-Score-100-to-100-by-LM/',
        'default_params': {'rsi_len': 14, 'macd_fast': 12, 'macd_slow': 26, 'williams_len': 14, 'score_thresh': 30},
    },
    'TV_StochRSI_RSI_MACD_Signals': {
        'gen': gen_TV_StochRSI_RSI_MACD_Signals,
        'space': _space_TV_StochRSI_RSI_MACD_Signals,
        'source': 'https://www.tradingview.com/script/yZCjneHe-Stoch-RSI-and-RSI-Buy-Sell-Signals-with-MACD-Trend-Filter/',
        'default_params': {'rsi_len': 14, 'stoch_len': 14, 'smooth_k': 3, 'smooth_d': 3, 'macd_fast': 12},
    },
    'TV_CombinedStrategyRSIADXSMA': {
        'gen': gen_TV_CombinedStrategyRSIADXSMA,
        'space': _space_TV_CombinedStrategyRSIADXSMA,
        'source': 'https://www.tradingview.com/script/7wrshs1q-Combined-Strategy-Trading-Bot-RSI-ADX-20SMA/',
        'default_params': {'rsi_len': 14, 'adx_len': 14, 'sma_len': 20, 'rsi_ob': 70, 'rsi_os': 30},
    },
    'TV_PeriodHighlighterPro': {
        'gen': gen_TV_PeriodHighlighterPro,
        'space': _space_TV_PeriodHighlighterPro,
        'source': 'https://www.tradingview.com/script/kV6F4yel-Period-Highlighter-Pro/',
        'default_params': {'period_len': 20, 'highlight_thresh': 0.6, 'trend_len': 50, 'rsi_len': 14, 'vol_mult': 1.2},
    },
    # ---- BATCH028: SR ZONE BREAKOUT (7) ----
    'TV_SRBreakout5Min': {
        'gen': gen_TV_SRBreakout5Min,
        'space': _space_TV_SRBreakout5Min,
        'source': 'https://www.tradingview.com/script/KOfsPM9h-SUPPORT-RESISTANCE-STRATEGY-5MIN-TF/',
        'default_params': {'pivot_left': 10, 'pivot_right': 5, 'sr_thresh': 0.001, 'rsi_len': 14, 'trend_len': 50},
    },
    'TV_SRBreakoutPmk': {
        'gen': gen_TV_SRBreakoutPmk,
        'space': _space_TV_SRBreakoutPmk,
        'source': 'https://www.tradingview.com/script/hwHKYkE0-Support-Resistance-breakout/',
        'default_params': {'pivot_left': 15, 'pivot_right': 5, 'vol_mult': 1.2, 'rsi_len': 14, 'atr_mult': 1.5},
    },
    'TV_GQTVolumeSRZones': {
        'gen': gen_TV_GQTVolumeSRZones,
        'space': _space_TV_GQTVolumeSRZones,
        'source': 'https://www.tradingview.com/script/WPCe698c-GQT-GPT-Volume-based-Support-Resistance-Zones-V2/',
        'default_params': {'zone_len': 20, 'vol_mult': 1.5, 'rsi_len': 14, 'atr_mult': 1.0, 'trend_len': 50},
    },
    'TV_DynamicBreakoutMaster': {
        'gen': gen_TV_DynamicBreakoutMaster,
        'space': _space_TV_DynamicBreakoutMaster,
        'source': 'https://www.tradingview.com/script/mqpqHVvo-Dynamic-Breakout-Master-by-tradingbauhaus/',
        'default_params': {'atr_len': 14, 'atr_mult': 1.5, 'pivot_left': 10, 'pivot_right': 5, 'rsi_len': 14},
    },
    'TV_BreaksAndRetests': {
        'gen': gen_TV_BreaksAndRetests,
        'space': _space_TV_BreaksAndRetests,
        'source': 'https://www.tradingview.com/script/800ndgbX-Breaks-and-Retests-Free990/',
        'default_params': {'pivot_left': 10, 'pivot_right': 5, 'retest_atr': 0.5, 'rsi_len': 14, 'trend_len': 50},
    },
    'TV_DynamicSRPivot': {
        'gen': gen_TV_DynamicSRPivot,
        'space': _space_TV_DynamicSRPivot,
        'source': 'https://www.tradingview.com/script/OKq8iEzr/',
        'default_params': {'pivot_left': 12, 'pivot_right': 4, 'atr_mult': 1.0, 'rsi_len': 14, 'trend_len': 50},
    },
    'TV_EMA920_SRBreakout': {
        'gen': gen_TV_EMA920_SRBreakout,
        'space': _space_TV_EMA920_SRBreakout,
        'source': 'https://www.tradingview.com/script/5EGVP61U-EMA-9-20-with-Support-and-Resistance-Breakout/',
        'default_params': {'ema_fast': 9, 'ema_slow': 20, 'pivot_left': 10, 'pivot_right': 5, 'rsi_len': 14},
    },
    # ---- BATCH028: PIVOT POINT SYSTEMS (8) ----
    'TV_PivotPointSuperTrend': {
        'gen': gen_TV_PivotPointSuperTrend,
        'space': _space_TV_PivotPointSuperTrend,
        'source': 'https://www.tradingview.com/script/DwdC6FT4-Pivot-Point-SuperTrend-Backtest/',
        'default_params': {'ph_left': 2, 'ph_right': 2, 'atr_mult': 3.0, 'atr_len': 14, 'rsi_len': 14},
    },
    'TV_CamarillaPivots': {
        'gen': gen_TV_CamarillaPivots,
        'space': _space_TV_CamarillaPivots,
        'source': 'https://www.tradingview.com/script/pZshc5OY-Camarilla-Pivot-Points-Backtest/',
        'default_params': {'session_len': 1440, 'h3_factor': 1.0833, 'l3_factor': 1.0833, 'rsi_len': 14, 'trend_len': 50},
    },
    'TV_CamarillaPivotsV2': {
        'gen': gen_TV_CamarillaPivotsV2,
        'space': _space_TV_CamarillaPivotsV2,
        'source': 'https://www.tradingview.com/script/MXkclJVM-Camarilla-Pivot-Points-V2-Backtest/',
        'default_params': {'session_len': 1440, 'h4_factor': 1.1, 'l4_factor': 1.1, 'rsi_len': 14, 'atr_mult': 1.0},
    },
    'TV_PivotPointV2': {
        'gen': gen_TV_PivotPointV2,
        'space': _space_TV_PivotPointV2,
        'source': 'https://www.tradingview.com/script/SIQgcWq2-Pivot-Point-V2-Backtest/',
        'default_params': {'session_len': 1440, 'r1_w': 0.5, 'rsi_len': 14, 'trend_len': 50, 'atr_mult': 1.0},
    },
    'TV_WoodiePivots': {
        'gen': gen_TV_WoodiePivots,
        'space': _space_TV_WoodiePivots,
        'source': 'https://www.tradingview.com/script/ZyNj4Ewl-Woodie-Pivot-Points-Backtest/',
        'default_params': {'session_len': 1440, 'rsi_len': 14, 'trend_len': 50, 'vol_mult': 1.0, 'atr_mult': 1.0},
    },
    'TV_DynamicPivot': {
        'gen': gen_TV_DynamicPivot,
        'space': _space_TV_DynamicPivot,
        'source': 'https://www.tradingview.com/script/MiFxDDNW-Dynamic-Pivot-Point-Backtest/',
        'default_params': {'pivot_left': 5, 'pivot_right': 5, 'rsi_len': 14, 'trend_len': 50, 'atr_mult': 1.0},
    },
    'TV_FloorPivots': {
        'gen': gen_TV_FloorPivots,
        'space': _space_TV_FloorPivots,
        'source': 'https://www.tradingview.com/script/4UlDByOA-Floor-Pivot-Points-Backtest/',
        'default_params': {'session_len': 1440, 'rsi_len': 14, 'trend_len': 50, 'vol_mult': 1.0, 'atr_mult': 1.0},
    },
    'TV_PivotSuperTrendTrendFilter': {
        'gen': gen_TV_PivotSuperTrendTrendFilter,
        'space': _space_TV_PivotSuperTrendTrendFilter,
        'source': 'https://www.tradingview.com/script/WpJ5ym7u-Pivot-Point-SuperTrend-Strategy-TrendFilter/',
        'default_params': {'ph_left': 2, 'ph_right': 2, 'atr_mult': 3.0, 'trend_len': 200, 'rsi_len': 14},
    },
    # ---- BATCH028: SUPPLY AND DEMAND ZONES (5) ----
    'TV_SupplyDemandStrategy': {
        'gen': gen_TV_SupplyDemandStrategy,
        'space': _space_TV_SupplyDemandStrategy,
        'source': 'https://www.tradingview.com/script/gHhqbgsG-Supply-and-Demand-Strategy/',
        'default_params': {'zone_len': 10, 'atr_mult': 1.5, 'rsi_len': 14, 'trend_len': 50, 'vol_mult': 1.0},
    },
    'TV_SupplyDemandEnhanced': {
        'gen': gen_TV_SupplyDemandEnhanced,
        'space': _space_TV_SupplyDemandEnhanced,
        'source': 'https://www.tradingview.com/script/nQcfcgN4-Supply-and-Demand-Zones-with-Enhanced-Signals/',
        'default_params': {'zone_len': 10, 'vol_mult': 1.5, 'rsi_len': 14, 'atr_mult': 1.0, 'trend_len': 50},
    },
    'TV_SupplyDemandCleanV6': {
        'gen': gen_TV_SupplyDemandCleanV6,
        'space': _space_TV_SupplyDemandCleanV6,
        'source': 'https://www.tradingview.com/script/VDv1AnDA-Supply-and-Demand-Zones-Clean-v6/',
        'default_params': {'zone_len': 15, 'atr_mult': 1.0, 'rsi_len': 14, 'trend_len': 50, 'vol_mult': 1.2},
    },
    'TV_SupplyDemandPro': {
        'gen': gen_TV_SupplyDemandPro,
        'space': _space_TV_SupplyDemandPro,
        'source': 'https://www.tradingview.com/script/h0jxhmgn-Supply-Demand-Zones-Pro/',
        'default_params': {'zone_len': 20, 'atr_mult': 1.0, 'vol_mult': 1.5, 'rsi_len': 14, 'trend_len': 50},
    },
    'TV_AGProSupplyDemand': {
        'gen': gen_TV_AGProSupplyDemand,
        'space': _space_TV_AGProSupplyDemand,
        'source': 'https://www.tradingview.com/script/rcbLhPjF-AG-Pro-Auto-Supply-Demand-Zones-AGPro-Series/',
        'default_params': {'zone_len': 10, 'swing_len': 5, 'atr_mult': 1.5, 'rsi_len': 14, 'trend_len': 50},
    },
    # ---- BATCH028: MARKET STRUCTURE BOS/CHoCH (5) ----
    'TV_MarketStructureBOSCHoCH': {
        'gen': gen_TV_MarketStructureBOSCHoCH,
        'space': _space_TV_MarketStructureBOSCHoCH,
        'source': 'https://www.tradingview.com/script/PmIP7HAZ-Market-Structure-BOS-CHOCH-MSB-FVG-OB-BB-Nephew-Sam/',
        'default_params': {'swing_len': 5, 'atr_mult': 1.0, 'rsi_len': 14, 'trend_len': 50, 'vol_mult': 1.0},
    },
    'TV_CHoCHBOSFractal': {
        'gen': gen_TV_CHoCHBOSFractal,
        'space': _space_TV_CHoCHBOSFractal,
        'source': 'https://www.tradingview.com/script/ZpHqSrBK-Market-Structure-CHoCH-BOS-Fractal-LuxAlgo/',
        'default_params': {'swing_len': 5, 'fractal_left': 3, 'rsi_len': 14, 'trend_len': 50, 'atr_mult': 1.0},
    },
    'TV_MTFBreakOfStructure': {
        'gen': gen_TV_MTFBreakOfStructure,
        'space': _space_TV_MTFBreakOfStructure,
        'source': 'https://www.tradingview.com/script/1LrVbh1I-MTF-Break-of-Structure-BOS-Market-Structure-Shift-MSS/',
        'default_params': {'swing_len': 10, 'atr_mult': 1.0, 'rsi_len': 14, 'trend_len': 50, 'vol_mult': 1.0},
    },
    'TV_DrawOnLiquidity': {
        'gen': gen_TV_DrawOnLiquidity,
        'space': _space_TV_DrawOnLiquidity,
        'source': 'https://www.tradingview.com/script/b66E7VZK-Draw-on-Liquidity-PhenLabs/',
        'default_params': {'swing_len': 10, 'liq_mult': 1.5, 'rsi_len': 14, 'trend_len': 50, 'atr_mult': 1.0},
    },
    'TV_MarketStructureZigZagBOS': {
        'gen': gen_TV_MarketStructureZigZagBOS,
        'space': _space_TV_MarketStructureZigZagBOS,
        'source': 'https://www.tradingview.com/script/NXgQEbZi-Market-Structure-ZigZag-Break-of-Structure-Order-Blocks/',
        'default_params': {'swing_len': 5, 'zz_dev': 0.05, 'rsi_len': 14, 'trend_len': 50, 'atr_mult': 1.0},
    },
}
