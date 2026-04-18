"""
Batch 704 — Multi-Timeframe via Resampling
TV_HTF_EMA_Gate, TV_HTF_RSI_Confirm, TV_Daily_Close_Signal, TV_Weekly_MACD
"""
import numpy as np
import pandas as pd


def _704_ema(s, n):
    return s.ewm(alpha=2/(n+1), adjust=False).mean()

def _704_rma(s, n):
    return s.ewm(alpha=1/n, adjust=False).mean()

def _704_atr(h, lo, c, n=14):
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    return _704_rma(tr, n)

def _704_resample_ffill(c, ts_ms, freq_str):
    """
    Resample close series to higher timeframe and forward-fill back to original index.
    ts_ms: numpy array of timestamps in milliseconds (Unix ms)
    freq_str: pandas offset string e.g. '4h', '1D', '1W'
    Returns a Series aligned to c.index.
    """
    idx = pd.to_datetime(ts_ms, unit='ms', utc=True)
    df_r = pd.Series(c.values, index=idx, name='close')
    htf = df_r.resample(freq_str, label='left', closed='left').last().dropna()
    # Reindex to original index and forward-fill
    htf_aligned = htf.reindex(idx, method='ffill')
    htf_aligned.index = c.index
    return htf_aligned


# ─── 1. TV_HTF_EMA_Gate ──────────────────────────────────────────────────────

def gen_TV_HTF_EMA_Gate(df, htf_hours=4, ema_len=50, rsi_len=14, rsi_os=35, **kw):
    """
    Resample to htf_hours bars and compute EMA on HTF close.
    Use HTF EMA as a trend gate:
    Long only when HTF close > HTF EMA (bullish bias) AND current-bar RSI is oversold.
    Short only when HTF close < HTF EMA AND RSI is overbought.
    """
    c  = df['close'].astype(float)
    h  = df['high'].astype(float)
    lo = df['low'].astype(float)
    n  = len(c)

    ts_ms = pd.to_numeric(df['ts'], errors='coerce').values

    # HTF EMA (computed on HTF bars, then aligned back to original)
    htf_close = _704_resample_ffill(c, ts_ms, f'{htf_hours}h')
    htf_ema   = _704_ema(htf_close, ema_len)

    # Current TF RSI (Wilder)
    delta = c.diff()
    ag    = _704_rma(delta.clip(lower=0), rsi_len)
    al    = _704_rma((-delta).clip(lower=0), rsi_len)
    rsi   = 100 - 100 / (1 + ag / (al + 1e-10))

    rsi_ob = 100 - rsi_os  # Overbought mirror

    warmup = max(ema_len * htf_hours + htf_hours, rsi_len) + 10
    sig    = np.zeros(n, dtype=int)
    pos    = 0

    for i in range(warmup, n):
        htf_c = htf_close.iloc[i]
        htf_e = htf_ema.iloc[i]
        ri    = rsi.iloc[i]
        ci    = c.iloc[i]

        if any(np.isnan(x) for x in [htf_c, htf_e, ri]):
            sig[i] = pos
            continue

        htf_bull = htf_c > htf_e
        htf_bear = htf_c < htf_e

        if pos == 0:
            if htf_bull and ri <= rsi_os:
                pos = 1; sig[i] = 1
            elif htf_bear and ri >= rsi_ob:
                pos = -1; sig[i] = -1
        elif pos == 1:
            if ri > 60 or htf_bear:
                pos = 0; sig[i] = 0
            else:
                sig[i] = 1
        elif pos == -1:
            if ri < 40 or htf_bull:
                pos = 0; sig[i] = 0
            else:
                sig[i] = -1

    return pd.Series(sig, index=df.index).shift(1, fill_value=0).astype(int)


def space_TV_HTF_EMA_Gate(trial):
    return {
        'htf_hours': trial.suggest_int('htf_hours', 1, 6),
        'ema_len':   trial.suggest_int('ema_len', 20, 100),
        'rsi_len':   trial.suggest_int('rsi_len', 7, 21),
        'rsi_os':    trial.suggest_int('rsi_os', 25, 45),
    }


# ─── 2. TV_HTF_RSI_Confirm ───────────────────────────────────────────────────

def gen_TV_HTF_RSI_Confirm(df, htf_hours=4, rsi_len=14, rsi_os=40, ema_trend=100,
                             **kw):
    """
    Dual RSI confirmation: both current TF and HTF RSI must be oversold to go long.
    Both must be overbought to go short.
    Exits when either TF RSI crosses back to neutral.
    """
    c  = df['close'].astype(float)
    n  = len(c)

    ts_ms = pd.to_numeric(df['ts'], errors='coerce').values

    # Current-TF RSI
    delta = c.diff()
    ag    = _704_rma(delta.clip(lower=0), rsi_len)
    al    = _704_rma((-delta).clip(lower=0), rsi_len)
    rsi   = 100 - 100 / (1 + ag / (al + 1e-10))

    # HTF RSI: compute on resampled close
    htf_close = _704_resample_ffill(c, ts_ms, f'{htf_hours}h')
    htf_delta = htf_close.diff()
    htf_ag    = _704_rma(htf_delta.clip(lower=0), rsi_len)
    htf_al    = _704_rma((-htf_delta).clip(lower=0), rsi_len)
    htf_rsi   = 100 - 100 / (1 + htf_ag / (htf_al + 1e-10))

    ema_t  = _704_ema(c, ema_trend)
    rsi_ob = 100 - rsi_os

    warmup = max(rsi_len * htf_hours + htf_hours, ema_trend) + 10
    sig    = np.zeros(n, dtype=int)
    pos    = 0

    for i in range(warmup, n):
        ri    = rsi.iloc[i]
        htf_r = htf_rsi.iloc[i]
        ci    = c.iloc[i]

        if any(np.isnan(x) for x in [ri, htf_r]):
            sig[i] = pos
            continue

        above_trend = ci > ema_t.iloc[i]
        below_trend = ci < ema_t.iloc[i]

        both_os = ri <= rsi_os and htf_r <= rsi_os + 5
        both_ob = ri >= rsi_ob and htf_r >= rsi_ob - 5

        if pos == 0:
            if both_os and above_trend:
                pos = 1; sig[i] = 1
            elif both_ob and below_trend:
                pos = -1; sig[i] = -1
        elif pos == 1:
            if ri > 55 or below_trend:
                pos = 0; sig[i] = 0
            else:
                sig[i] = 1
        elif pos == -1:
            if ri < 45 or above_trend:
                pos = 0; sig[i] = 0
            else:
                sig[i] = -1

    return pd.Series(sig, index=df.index).shift(1, fill_value=0).astype(int)


def space_TV_HTF_RSI_Confirm(trial):
    return {
        'htf_hours': trial.suggest_int('htf_hours', 2, 8),
        'rsi_len':   trial.suggest_int('rsi_len', 7, 21),
        'rsi_os':    trial.suggest_int('rsi_os', 30, 50),
        'ema_trend': trial.suggest_int('ema_trend', 50, 200),
    }


# ─── 3. TV_Daily_Close_Signal ────────────────────────────────────────────────

def gen_TV_Daily_Close_Signal(df, daily_ema=10, ema_trend=100, rsi_len=14,
                               rsi_os=40, **kw):
    """
    Resample to daily bars.
    Compute EMA of daily close. Bias = bullish if daily close > daily EMA.
    On lower TF: only take longs when daily bias is bullish AND RSI is oversold.
    Only take shorts when daily bias is bearish AND RSI is overbought.
    """
    c  = df['close'].astype(float)
    n  = len(c)

    ts_ms = pd.to_numeric(df['ts'], errors='coerce').values

    # Daily close resampled and EMA
    daily_close = _704_resample_ffill(c, ts_ms, '1D')
    daily_ema_s = _704_ema(daily_close, daily_ema)

    # Current TF RSI
    delta = c.diff()
    ag    = _704_rma(delta.clip(lower=0), rsi_len)
    al    = _704_rma((-delta).clip(lower=0), rsi_len)
    rsi   = 100 - 100 / (1 + ag / (al + 1e-10))

    ema_t  = _704_ema(c, ema_trend)
    rsi_ob = 100 - rsi_os

    warmup = max(daily_ema * 24 + 24, ema_trend, rsi_len) + 10
    sig    = np.zeros(n, dtype=int)
    pos    = 0

    for i in range(warmup, n):
        dc   = daily_close.iloc[i]
        de   = daily_ema_s.iloc[i]
        ri   = rsi.iloc[i]
        ci   = c.iloc[i]

        if any(np.isnan(x) for x in [dc, de, ri]):
            sig[i] = pos
            continue

        daily_bull = dc > de
        daily_bear = dc < de
        above_ema  = ci > ema_t.iloc[i]
        below_ema  = ci < ema_t.iloc[i]

        if pos == 0:
            if daily_bull and above_ema and ri <= rsi_os:
                pos = 1; sig[i] = 1
            elif daily_bear and below_ema and ri >= rsi_ob:
                pos = -1; sig[i] = -1
        elif pos == 1:
            if not daily_bull or ri > 60:
                pos = 0; sig[i] = 0
            else:
                sig[i] = 1
        elif pos == -1:
            if not daily_bear or ri < 40:
                pos = 0; sig[i] = 0
            else:
                sig[i] = -1

    return pd.Series(sig, index=df.index).shift(1, fill_value=0).astype(int)


def space_TV_Daily_Close_Signal(trial):
    return {
        'daily_ema': trial.suggest_int('daily_ema', 5, 20),
        'ema_trend': trial.suggest_int('ema_trend', 50, 200),
        'rsi_len':   trial.suggest_int('rsi_len', 7, 21),
        'rsi_os':    trial.suggest_int('rsi_os', 30, 50),
    }


# ─── 4. TV_Weekly_MACD ───────────────────────────────────────────────────────

def gen_TV_Weekly_MACD(df, macd_fast=12, macd_slow=26, macd_sig=9,
                        ema_trend=100, **kw):
    """
    Weekly MACD from resampled weekly close.
    MACD histogram direction on weekly = bias for all intraweek trades.
    Positive weekly MACD histogram → long bias (only take longs intraweek).
    Negative → short bias.
    Exit when weekly MACD histogram flips direction.
    """
    c  = df['close'].astype(float)
    n  = len(c)

    ts_ms = pd.to_numeric(df['ts'], errors='coerce').values

    # Weekly close
    weekly_close = _704_resample_ffill(c, ts_ms, '1W')

    # Weekly MACD
    wk_fast = _704_ema(weekly_close, macd_fast)
    wk_slow = _704_ema(weekly_close, macd_slow)
    wk_macd = wk_fast - wk_slow
    wk_sig  = _704_ema(wk_macd, macd_sig)
    wk_hist = wk_macd - wk_sig

    ema_t = _704_ema(c, ema_trend)

    warmup = max((macd_slow + macd_sig) * 7 + 7, ema_trend) + 10
    sig    = np.zeros(n, dtype=int)
    pos    = 0

    for i in range(warmup, n):
        hist_i = wk_hist.iloc[i]
        hist_p = wk_hist.iloc[i - 1]
        ci     = c.iloc[i]

        if any(np.isnan(x) for x in [hist_i, hist_p]):
            sig[i] = pos
            continue

        above_trend = ci > ema_t.iloc[i]
        below_trend = ci < ema_t.iloc[i]

        wk_bull = hist_i > 0
        wk_bear = hist_i < 0
        hist_rising  = hist_i > hist_p
        hist_falling = hist_i < hist_p

        if pos == 0:
            if wk_bull and hist_rising and above_trend:
                pos = 1; sig[i] = 1
            elif wk_bear and hist_falling and below_trend:
                pos = -1; sig[i] = -1
        elif pos == 1:
            if wk_bear or (wk_bull and hist_falling and hist_i < hist_p * 0.5):
                pos = 0; sig[i] = 0
            else:
                sig[i] = 1
        elif pos == -1:
            if wk_bull or (wk_bear and hist_rising and hist_i > hist_p * 0.5):
                pos = 0; sig[i] = 0
            else:
                sig[i] = -1

    return pd.Series(sig, index=df.index).shift(1, fill_value=0).astype(int)


def space_TV_Weekly_MACD(trial):
    return {
        'macd_fast': trial.suggest_int('macd_fast', 8, 20),
        'macd_slow': trial.suggest_int('macd_slow', 18, 40),
        'macd_sig':  trial.suggest_int('macd_sig', 5, 14),
        'ema_trend': trial.suggest_int('ema_trend', 50, 200),
    }


# ─── Export ──────────────────────────────────────────────────────────────────

STRATEGY_EXPORT = {
    'TV_HTF_EMA_Gate':        {'gen': gen_TV_HTF_EMA_Gate,        'space': space_TV_HTF_EMA_Gate},
    'TV_HTF_RSI_Confirm':     {'gen': gen_TV_HTF_RSI_Confirm,     'space': space_TV_HTF_RSI_Confirm},
    'TV_Daily_Close_Signal':  {'gen': gen_TV_Daily_Close_Signal,  'space': space_TV_Daily_Close_Signal},
    'TV_Weekly_MACD':         {'gen': gen_TV_Weekly_MACD,         'space': space_TV_Weekly_MACD},
}
