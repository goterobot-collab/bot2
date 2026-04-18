"""
tv2_batch45.py — 20 Pine Script strategies converted to Python
Sources: hasnocool, getupandCROW, LouisLetcher, jamesbachini repos
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (RMA) — same as Pine ta.rma."""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = _rma(gain, n)
    avg_loss = _rma(loss, n)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return _rma(tr, n)

def _bb(close: pd.Series, n: int = 20, mult: float = 2.0):
    basis = _sma(close, n)
    dev = close.rolling(n).std(ddof=0)
    return basis, basis + mult * dev, basis - mult * dev

def _kc(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 20, mult: float = 1.5):
    ma = _ema(c, n)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_ma = _ema(tr, n)
    return ma, ma + mult * atr_ma, ma - mult * atr_ma

def _crossover(s1: pd.Series, s2: pd.Series) -> pd.Series:
    return (s1 > s2) & (s1.shift(1) <= s2.shift(1))

def _crossunder(s1: pd.Series, s2: pd.Series) -> pd.Series:
    return (s1 < s2) & (s1.shift(1) >= s2.shift(1))

def _wma(s: pd.Series, n: int) -> pd.Series:
    weights = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def _alma(s: pd.Series, n: int, offset: float = 0.85, sigma: int = 6) -> pd.Series:
    """Arnaud Legoux Moving Average."""
    m = offset * (n - 1)
    s2 = sigma * sigma
    weights = np.array([np.exp(-((i - m) ** 2) / (2 * s2)) for i in range(n)])
    weights /= weights.sum()
    return s.rolling(n).apply(lambda x: np.dot(x, weights), raw=True)

def _adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14):
    up_move = h.diff()
    down_move = l.diff().mul(-1)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=h.index)
    minus_dm = pd.Series(minus_dm, index=h.index)
    atr = _atr(h, l, c, n)
    plus_di = 100 * _rma(plus_dm, n) / atr
    minus_di = 100 * _rma(minus_dm, n) / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx_val = _rma(dx.fillna(0), n)
    return adx_val, plus_di, minus_di

def _mfi(h: pd.Series, l: pd.Series, c: pd.Series, vol: pd.Series, n: int = 14) -> pd.Series:
    tp = (h + l + c) / 3
    raw_mf = tp * vol
    pos_mf = raw_mf.where(tp > tp.shift(1), 0.0)
    neg_mf = raw_mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos_mf.rolling(n).sum()
    neg_sum = neg_mf.rolling(n).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - 100 / (1 + mfr)

def _supertrend(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 7, mult: float = 3.0):
    """Returns supertrend line and trend direction (+1 up, -1 down)."""
    src = (h + l) / 2
    atr = _atr(h, l, c, period)
    upper_band = src + mult * atr
    lower_band = src - mult * atr

    trend = pd.Series(np.ones(len(c)), index=c.index)
    tup = lower_band.copy()
    tdn = upper_band.copy()

    for i in range(1, len(c)):
        prev_tup = tup.iloc[i - 1]
        prev_tdn = tdn.iloc[i - 1]
        prev_close = c.iloc[i - 1]

        tup.iloc[i] = lower_band.iloc[i] if lower_band.iloc[i] > prev_tup or prev_close < prev_tup else prev_tup
        tdn.iloc[i] = upper_band.iloc[i] if upper_band.iloc[i] < prev_tdn or prev_close > prev_tdn else prev_tdn

        if trend.iloc[i - 1] == -1 and c.iloc[i] > tdn.iloc[i - 1]:
            trend.iloc[i] = 1
        elif trend.iloc[i - 1] == 1 and c.iloc[i] < tup.iloc[i - 1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]

    st_line = tup.where(trend == 1, tdn)
    return st_line, trend

def _stoch_rsi(close: pd.Series, rsi_len: int = 14, stoch_len: int = 14,
               smooth_k: int = 3, smooth_d: int = 3):
    rsi_val = _rsi(close, rsi_len)
    lowest_rsi = rsi_val.rolling(stoch_len).min()
    highest_rsi = rsi_val.rolling(stoch_len).max()
    k_raw = 100 * (rsi_val - lowest_rsi) / (highest_rsi - lowest_rsi).replace(0, np.nan)
    k = _sma(k_raw, smooth_k)
    d = _sma(k, smooth_d)
    return k, d

def _linreg(s: pd.Series, n: int) -> pd.Series:
    """Linear regression value (same as Pine ta.linreg with offset=0)."""
    return s.rolling(n).apply(
        lambda x: np.polyval(np.polyfit(np.arange(len(x)), x, 1), len(x) - 1),
        raw=True,
    )

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ===========================================================================
# 1. AlphaTrend Strategy
# ===========================================================================

def gen_AlphaTrend(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    AlphaTrend: MFI/RSI-based adaptive trend line.
    Entry: crossover(AlphaTrend, AlphaTrend[2]) = long
           crossunder(AlphaTrend, AlphaTrend[2]) = short
    """
    coeff = params.get("coeff", 1.0)
    period = params.get("period", 14)
    use_rsi = params.get("use_rsi", False)

    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    atr = _sma(pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1), period)
    up_t = l - atr * coeff
    dn_t = h + atr * coeff

    hlc3 = (h + l + c) / 3
    if use_rsi:
        indicator = _rsi(c, period)
    else:
        indicator = _mfi(h, l, c, v, period)

    at = pd.Series(np.nan, index=c.index)
    for i in range(len(c)):
        prev = at.iloc[i - 1] if i > 0 else 0.0
        if pd.isna(prev):
            prev = 0.0
        if indicator.iloc[i] >= 50:
            at.iloc[i] = max(up_t.iloc[i], prev) if up_t.iloc[i] < prev else up_t.iloc[i]
        else:
            at.iloc[i] = min(dn_t.iloc[i], prev) if dn_t.iloc[i] > prev else dn_t.iloc[i]

    at_lag2 = at.shift(2)
    buy_sig = _crossover(at, at_lag2)
    sell_sig = _crossunder(at, at_lag2)

    signal = pd.Series(0, index=c.index)
    signal[buy_sig] = 1
    signal[sell_sig] = -1
    return signal


def space_AlphaTrend(trial) -> dict:
    return {
        "coeff": trial.suggest_float("coeff", 0.5, 3.0, step=0.1),
        "period": trial.suggest_int("period", 7, 30),
        "use_rsi": trial.suggest_categorical("use_rsi", [False, True]),
    }


# ===========================================================================
# 2. BB Keltner Squeeze Strategy
# ===========================================================================

def gen_BBKeltnerSqueeze(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    BB+KC Squeeze: when BB inside KC → squeeze. Exit squeeze with momentum direction.
    Long: squeeze ends, close > BB basis. Short: squeeze ends, close < BB basis.
    """
    length = params.get("length", 20)
    bb_mult = params.get("bb_mult", 2.0)
    kc_mult = params.get("kc_mult", 1.5)

    c = df["close"]
    basis, bb_upper, bb_lower = _bb(c, length, bb_mult)
    _, kc_upper, kc_lower = _kc(df["high"], df["low"], c, length, kc_mult)

    squeeze = (bb_upper <= kc_upper) | (bb_lower >= kc_lower)

    # midc: 0=squeeze, 1=above basis, 2=below basis
    midc = pd.Series(0, index=c.index)
    midc[~squeeze & (c > basis)] = 1
    midc[~squeeze & (c <= basis)] = 2

    # Long entry: transition from squeeze (0) to above basis (1)
    long_entry = (midc.shift(1) == 0) & (midc == 1)
    # Short entry: transition from squeeze (0) to below basis (2)
    short_entry = (midc.shift(1) == 0) & (midc == 2)
    # Exit: direction changed
    exit_sig = midc != midc.shift(1)

    signal = pd.Series(0, index=c.index)
    signal[long_entry] = 1
    signal[short_entry] = -1
    return signal


def space_BBKeltnerSqueeze(trial) -> dict:
    return {
        "length": trial.suggest_int("length", 10, 40),
        "bb_mult": trial.suggest_float("bb_mult", 1.5, 3.0, step=0.5),
        "kc_mult": trial.suggest_float("kc_mult", 1.0, 2.5, step=0.5),
    }


# ===========================================================================
# 3. ALMA Cross Strategy
# ===========================================================================

def gen_ALMACross(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Fast ALMA crosses above/below slow ALMA, filtered by volume oscillator > 0.
    """
    fast_len = params.get("fast_len", 60)
    slow_len = params.get("slow_len", 120)
    alma_offset = params.get("alma_offset", 0.85)
    alma_sigma = params.get("alma_sigma", 6)
    vol_fast = params.get("vol_fast", 5)
    vol_slow = params.get("vol_slow", 10)

    c = df["close"]
    v = df["volume"]

    alma_fast = _alma(c, fast_len, alma_offset, alma_sigma)
    alma_slow = _alma(c, slow_len, alma_offset, alma_sigma)

    vol_osc = 100 * (_ema(v, vol_fast) - _ema(v, vol_slow)) / _ema(v, vol_slow)

    buy = _crossover(alma_fast, alma_slow) & (vol_osc > 0)
    sell = _crossunder(alma_fast, alma_slow) & (vol_osc > 0)

    signal = pd.Series(0, index=c.index)
    signal[buy] = 1
    signal[sell] = -1
    return signal


def space_ALMACross(trial) -> dict:
    return {
        "fast_len": trial.suggest_int("fast_len", 20, 80, step=10),
        "slow_len": trial.suggest_int("slow_len", 80, 200, step=20),
        "alma_offset": trial.suggest_float("alma_offset", 0.5, 0.95, step=0.05),
        "alma_sigma": trial.suggest_int("alma_sigma", 3, 10),
        "vol_fast": trial.suggest_int("vol_fast", 3, 10),
        "vol_slow": trial.suggest_int("vol_slow", 5, 20),
    }


# ===========================================================================
# 4. ATR Mean Reversion Strategy V1
# ===========================================================================

def gen_ATRMeanReversion(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Long only: buy when low < (open - ATR), exit when high > EMA or above ATR band.
    Adapted to both long and short for crypto.
    """
    period = params.get("period", 10)
    sl_mult = params.get("sl_mult", 1.5)

    h, l, c, o = df["high"], df["low"], df["close"], df["open"]
    atr = _atr(h, l, c, period)
    mean = _ema(c, period)

    buy_cond = l < (o - atr)
    sell_cond = (h > mean) & (c < l.shift(1)) | (h > o + atr)

    # Short: price pokes above upper band → reversion down
    short_cond = h > (o + atr)
    short_exit = (l < mean) & (c > h.shift(1)) | (l < o - atr)

    signal = pd.Series(0, index=c.index)
    signal[buy_cond] = 1
    signal[short_cond] = -1
    return signal


def space_ATRMeanReversion(trial) -> dict:
    return {
        "period": trial.suggest_int("period", 5, 25),
        "sl_mult": trial.suggest_float("sl_mult", 1.0, 3.0, step=0.25),
    }


# ===========================================================================
# 5. Adaptive Price Channel Strategy
# ===========================================================================

def gen_AdaptivePriceChannel(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    ADX-filtered channel. Sideways: trade mean reversion. Trending: trade breakout.
    """
    length = params.get("length", 20)
    atr_mult = params.get("atr_mult", 3.2)
    adx_threshold = params.get("adx_threshold", 25)

    h, l, c = df["high"], df["low"], df["close"]
    hh = h.rolling(length).max()
    ll = l.rolling(length).min()
    atr = _atr(h, l, c, length)
    adx_val, plus_di, minus_di = _adx(h, l, c, length)

    signal = pd.Series(0, index=c.index)

    # Sideways market (ADX < threshold): mean reversion
    sideways = adx_val < adx_threshold
    long_sw = sideways & (c > hh - atr_mult * atr)
    short_sw = sideways & (c < ll + atr_mult * atr)

    # Trending (ADX >= threshold): breakout direction
    bull_trend = (adx_val >= adx_threshold) & (plus_di > minus_di)
    bear_trend = (adx_val >= adx_threshold) & (plus_di < minus_di)
    long_tr = bull_trend & (c > hh - atr_mult * atr)
    short_tr = bear_trend & (c < ll + atr_mult * atr)

    signal[long_sw | long_tr] = 1
    signal[short_sw | short_tr] = -1
    return signal


def space_AdaptivePriceChannel(trial) -> dict:
    return {
        "length": trial.suggest_int("length", 10, 40),
        "atr_mult": trial.suggest_float("atr_mult", 1.5, 5.0, step=0.5),
        "adx_threshold": trial.suggest_int("adx_threshold", 15, 40, step=5),
    }


# ===========================================================================
# 6. Athena Momentum Squeeze
# ===========================================================================

def gen_AthenaMomentumSqueeze(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Squeeze momentum (BB inside KC) + linreg momentum value.
    Entry after squeeze releases with momentum direction, filtered by SMA200.
    """
    bb_len = params.get("bb_len", 20)
    bb_mult = params.get("bb_mult", 2.0)
    kc_len = params.get("kc_len", 20)
    kc_mult = params.get("kc_mult", 1.5)
    ma_len = params.get("ma_len", 200)
    sqz_filter = params.get("sqz_filter", 6)

    h, l, c = df["high"], df["low"], df["close"]
    basis, bb_upper, bb_lower = _bb(c, bb_len, bb_mult)
    _, kc_upper, kc_lower = _kc(h, l, c, kc_len, kc_mult)

    sqz_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    sqz_off = (bb_lower < kc_lower) & (bb_upper > kc_upper)
    no_sqz = ~sqz_on & ~sqz_off

    # Squeeze momentum value
    hh = h.rolling(kc_len).max()
    ll = l.rolling(kc_len).min()
    src_mod = c - ((hh + ll) / 2 + _sma(c, kc_len)) / 2
    sqz_val = _linreg(src_mod, kc_len)

    bar_light_green = (sqz_val > 0) & (sqz_val > sqz_val.shift(1))
    bar_dark_green = (sqz_val > 0) & ~(sqz_val > sqz_val.shift(1))
    bar_light_red = (sqz_val <= 0) & (sqz_val < sqz_val.shift(1))
    bar_dark_red = (sqz_val <= 0) & ~(sqz_val < sqz_val.shift(1))

    circle_light_blue = ~no_sqz & ~sqz_on  # squeeze just released
    dark_blue_cnt = sqz_on.astype(int)
    # count consecutive dark-blue (squeeze on) bars
    cnt = pd.Series(0, index=c.index)
    for i in range(1, len(c)):
        cnt.iloc[i] = cnt.iloc[i - 1] + 1 if sqz_on.iloc[i] else 0

    ma_val = _sma(c, ma_len)
    ma_long = c > ma_val
    ma_short = c < ma_val

    long_cond = bar_light_green & circle_light_blue & (cnt.shift(1) >= sqz_filter) & ma_long
    short_cond = bar_light_red & circle_light_blue & (cnt.shift(1) >= sqz_filter) & ma_short

    # Exit: dark-green (long side) or dark-red (short side)
    signal = pd.Series(0, index=c.index)
    signal[long_cond] = 1
    signal[short_cond] = -1
    return signal


def space_AthenaMomentumSqueeze(trial) -> dict:
    return {
        "bb_len": trial.suggest_int("bb_len", 10, 30),
        "bb_mult": trial.suggest_float("bb_mult", 1.5, 3.0, step=0.5),
        "kc_len": trial.suggest_int("kc_len", 10, 30),
        "kc_mult": trial.suggest_float("kc_mult", 1.0, 2.5, step=0.5),
        "ma_len": trial.suggest_int("ma_len", 50, 300, step=50),
        "sqz_filter": trial.suggest_int("sqz_filter", 2, 12),
    }


# ===========================================================================
# 7. QFL Mean Reversion (3C QFL)
# ===========================================================================

def gen_QFLMeanReversal(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    QFL base support/resistance: fractal highs/lows with volume confirmation.
    Buy when price drops below fractal-low base by pct. Sell above fractal-high base.
    """
    vol_period = params.get("vol_period", 60)
    pct_threshold = params.get("pct_threshold", 0.5)
    max_age = params.get("max_age", 10)

    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    vam = _sma(v, vol_period)

    # Fractal: low[3] is local min if low[3]<low[4]<low[5] and low[2]>low[3] and low[1]>low[2]
    down = (
        (l.shift(3) < l.shift(4)) & (l.shift(4) < l.shift(5)) &
        (l.shift(2) > l.shift(3)) & (l.shift(1) > l.shift(2)) &
        (v.shift(3) > vam.shift(3))
    )
    up = (
        (h.shift(3) > h.shift(4)) & (h.shift(4) > h.shift(5)) &
        (h.shift(2) < h.shift(3)) & (h.shift(1) < h.shift(2)) &
        (v.shift(3) > vam.shift(3))
    )

    # Track most recent fractal levels
    fdown = pd.Series(np.nan, index=c.index)
    fup = pd.Series(np.nan, index=c.index)
    for i in range(len(c)):
        fdown.iloc[i] = l.iloc[i - 3] if down.iloc[i] else (fdown.iloc[i - 1] if i > 0 else np.nan)
        fup.iloc[i] = h.iloc[i - 3] if up.iloc[i] else (fup.iloc[i - 1] if i > 0 else np.nan)

    # Age of base (bars since last fractal update)
    age_down = pd.Series(0, index=c.index)
    age_up = pd.Series(0, index=c.index)
    for i in range(1, len(c)):
        age_down.iloc[i] = 0 if down.iloc[i] else age_down.iloc[i - 1] + 1
        age_up.iloc[i] = 0 if up.iloc[i] else age_up.iloc[i - 1] + 1

    age_ok_down = (max_age == 0) | (age_down < max_age)
    age_ok_up = (max_age == 0) | (age_up < max_age)

    buy = (100 * (c / fdown.fillna(method="ffill")) < 100 - pct_threshold) & age_ok_down
    sell = (100 * (c / fup.fillna(method="ffill")) > 100 + pct_threshold) & age_ok_up

    signal = pd.Series(0, index=c.index)
    signal[buy] = 1
    signal[sell] = -1
    return signal


def space_QFLMeanReversal(trial) -> dict:
    return {
        "vol_period": trial.suggest_int("vol_period", 20, 100, step=10),
        "pct_threshold": trial.suggest_float("pct_threshold", 0.2, 2.0, step=0.1),
        "max_age": trial.suggest_int("max_age", 0, 20),
    }


# ===========================================================================
# 8. VADER Directional Energy Ratio (DEB)
# ===========================================================================

def gen_VADER_DEB(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    VADER: volume-weighted directional energy ratio.
    Long when all 3 conditions: red sentiment, red energy bar, negative signal → atBottom.
    Short when all 3: green sentiment, green energy bar, positive signal → atTop.
    """
    length = params.get("length", 10)
    der_avg = params.get("der_avg", 5)
    smooth = params.get("smooth", 3)
    senti_len = params.get("senti_len", 20)
    vlookbk = params.get("vlookbk", 20)

    c = df["close"]
    v = df["volume"].fillna(0)

    # Relative volume normalization
    v_min = v.rolling(vlookbk).min()
    v_max = v.rolling(vlookbk).max()
    vola = (v - v_min) / (v_max - v_min + 1e-10)  # 0..1

    # VADER core
    r2 = (df["high"].rolling(2).max() - df["low"].rolling(2).min()) / 2
    sr = c.diff() / r2.replace(0, np.nan)
    rsr = sr.clip(-1, 1).fillna(0)
    c_val = (rsr * vola).fillna(0)
    c_plus = c_val.clip(lower=0)
    c_minus = (-c_val).clip(lower=0)

    avg_vola = _wma(vola, length)
    dem = _wma(c_plus, length) / avg_vola.replace(0, np.nan)
    sup = _wma(c_minus, length) / avg_vola.replace(0, np.nan)
    adp = 100 * _wma(dem.fillna(0), der_avg)
    asp = 100 * _wma(sup.fillna(0), der_avg)
    anp = adp - asp
    anp_s = _wma(anp, smooth)

    s_adp = 100 * _wma(dem.fillna(0), senti_len)
    s_asp = 100 * _wma(sup.fillna(0), senti_len)
    v_senti = _wma(s_adp - s_asp, smooth)

    s_up = v_senti >= 0
    sflag_up = v_senti.diff() > 0
    up = anp_s >= 0

    # atBottom: all red
    at_bottom = (~s_up & sflag_up) & (adp < asp) & ~up
    # atTop: all green
    at_top = (s_up & sflag_up) & (adp > asp) & up

    signal = pd.Series(0, index=c.index)
    signal[at_bottom] = 1   # long (reversal up from bottom)
    signal[at_top] = -1     # short (reversal down from top)
    return signal


def space_VADER_DEB(trial) -> dict:
    return {
        "length": trial.suggest_int("length", 5, 20),
        "der_avg": trial.suggest_int("der_avg", 3, 10),
        "smooth": trial.suggest_int("smooth", 2, 7),
        "senti_len": trial.suggest_int("senti_len", 10, 40),
        "vlookbk": trial.suggest_int("vlookbk", 10, 40),
    }


# ===========================================================================
# 9. Bayesian SuperTrend Swing Pivot
# ===========================================================================

def gen_BayesianSuperTrend(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Bayesian RSI probability estimate for up/down moves, combined with SuperTrend.
    Long: Bayesian probability crossover its SMA. Short: crossunder.
    """
    st_factor = params.get("st_factor", 3)
    st_atr = params.get("st_atr", 7)
    period = params.get("period", 30)
    thresh = params.get("thresh", 1.003)
    look_range = params.get("look_range", 7)
    rsi_jump = params.get("rsi_jump", 8)
    bayes_smooth = params.get("bayes_smooth", 3)
    rsi_period = params.get("rsi_period", 14)

    c = df["close"]
    h, l = df["high"], df["low"]
    rsi_val = _rsi(c, rsi_period)

    # Bayesian probability
    bayes = pd.Series(50.0, index=c.index)
    for i in range(period, len(c)):
        countup = 1; countdn = 1
        countup2 = 1; countup3 = 1
        for j in range(1, period + 1):
            idx = i - j
            ref_idx = idx - look_range
            if ref_idx < 0:
                continue
            ratio = c.iloc[idx] / c.iloc[ref_idx] if c.iloc[ref_idx] != 0 else 1.0
            if ratio > thresh:
                countup += 1
            else:
                countdn += 1
            # with RSI confirmation
            rsi_now = rsi_val.iloc[idx] if not pd.isna(rsi_val.iloc[idx]) else 50
            rsi_ref = rsi_val.iloc[ref_idx] if ref_idx >= 0 and not pd.isna(rsi_val.iloc[ref_idx]) else 50
            if ratio > thresh and rsi_now > rsi_ref + rsi_jump:
                countup2 += 1
            else:
                countup3 += 1

        total = countup + countdn
        p_up = countup / total
        p_dn = countdn / total
        p2 = countup2 / period
        p3 = countup3 / period
        denom = p_up * p2 + p_dn * p3
        bayes.iloc[i] = ((p_up * p2) / denom * 100) if denom > 0 else 50.0

    sn1 = _sma(bayes, bayes_smooth)
    long_cond = _crossover(bayes, sn1)
    short_cond = _crossunder(bayes, sn1)

    # SuperTrend filter
    _, trend = _supertrend(h, l, c, st_atr, st_factor)

    signal = pd.Series(0, index=c.index)
    signal[long_cond & (trend == 1)] = 1
    signal[short_cond & (trend == -1)] = -1
    return signal


def space_BayesianSuperTrend(trial) -> dict:
    return {
        "st_factor": trial.suggest_int("st_factor", 2, 5),
        "st_atr": trial.suggest_int("st_atr", 5, 14),
        "period": trial.suggest_int("period", 15, 50),
        "thresh": trial.suggest_float("thresh", 1.001, 1.01, step=0.001),
        "look_range": trial.suggest_int("look_range", 3, 14),
        "rsi_jump": trial.suggest_int("rsi_jump", 4, 15),
        "bayes_smooth": trial.suggest_int("bayes_smooth", 2, 7),
        "rsi_period": trial.suggest_int("rsi_period", 7, 21),
    }


# ===========================================================================
# 10. BB + EMA + RSI + ADX Reversal/Breakout
# ===========================================================================

def gen_BBEMARSIADXReversal(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Dual mode strategy:
    Reversal: price touches BB lower + bullish reversal candle + ADX low + price above EMA.
    Breakout: price above BB upper + RSI high + ADX high + price above EMA.
    """
    ema1_len = params.get("ema1_len", 200)
    ema2_len = params.get("ema2_len", 100)
    bb_len = params.get("bb_len", 20)
    bb_mult = params.get("bb_mult", 2.0)
    atr_len = params.get("atr_len", 14)
    adx_len = params.get("adx_len", 14)
    adx_ranging = params.get("adx_ranging", 30)
    adx_trending = params.get("adx_trending", 50)
    rsi_len = params.get("rsi_len", 14)
    rsi_trend_long_min = params.get("rsi_trend_long_min", 65)
    rsi_trend_short_max = params.get("rsi_trend_short_max", 35)

    h, l, c, o = df["high"], df["low"], df["close"], df["open"]
    ema1 = _ema(c, ema1_len)
    ema2 = _ema(c, ema2_len)
    basis, bb_upper, bb_lower = _bb(c, bb_len, bb_mult)
    rsi_val = _rsi(c, rsi_len)
    adx_val, plus_di, minus_di = _adx(h, l, c, adx_len)
    atr = _atr(h, l, c, atr_len)

    ema_long = (c > ema1) & (c > ema2)
    ema_short = (c < ema1) & (c < ema2)
    reverse_candle_long = (c.shift(1) < o.shift(1)) & (c > o)
    reverse_candle_short = (c.shift(1) > o.shift(1)) & (c < o)

    # Ranging: ADX < ranging threshold
    ranging_adx = adx_val < adx_ranging
    trending_adx = adx_val > adx_trending

    # Price below lower band (in last 2 bars)
    price_below_lower = (c <= bb_lower) | (c.shift(1) <= bb_lower.shift(1))
    price_above_upper = (c >= bb_upper) | (c.shift(1) >= bb_upper.shift(1))

    # Reversal entries
    cross_long = price_below_lower & reverse_candle_long & ranging_adx & ema_long
    cross_short = price_above_upper & reverse_candle_short & ranging_adx & ema_short

    # Breakout entries
    trend_long = (c >= bb_upper) & ema_long & (rsi_val > rsi_trend_long_min) & trending_adx
    trend_short = (c <= bb_lower) & ema_short & (rsi_val < rsi_trend_short_max) & trending_adx

    signal = pd.Series(0, index=c.index)
    signal[cross_long | trend_long] = 1
    signal[cross_short | trend_short] = -1
    return signal


def space_BBEMARSIADXReversal(trial) -> dict:
    return {
        "ema1_len": trial.suggest_int("ema1_len", 100, 300, step=50),
        "ema2_len": trial.suggest_int("ema2_len", 50, 200, step=50),
        "bb_len": trial.suggest_int("bb_len", 10, 30),
        "bb_mult": trial.suggest_float("bb_mult", 1.5, 3.0, step=0.5),
        "adx_len": trial.suggest_int("adx_len", 7, 21),
        "adx_ranging": trial.suggest_int("adx_ranging", 20, 40, step=5),
        "adx_trending": trial.suggest_int("adx_trending", 40, 70, step=5),
        "rsi_len": trial.suggest_int("rsi_len", 7, 21),
        "rsi_trend_long_min": trial.suggest_int("rsi_trend_long_min", 55, 75),
        "rsi_trend_short_max": trial.suggest_int("rsi_trend_short_max", 25, 45),
        "atr_len": trial.suggest_int("atr_len", 7, 21),
    }


# ===========================================================================
# 11. Bullish Engulfing + RSI Exit
# ===========================================================================

def gen_BullishEngulfing(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Enter long on bullish engulfing pattern. Exit when RSI > threshold.
    Short: bearish engulfing. Exit when RSI < (100 - threshold).
    """
    rsi_len = params.get("rsi_len", 2)
    rsi_exit = params.get("rsi_exit", 90)

    c, o = df["close"], df["open"]
    rsi_val = _rsi(c, rsi_len)

    # Bullish engulfing: prev bearish, current close > prev open, current open < prev close
    bull_eng = (c.shift(1) < o.shift(1)) & (c > o.shift(1)) & (o < c.shift(1))
    # Bearish engulfing
    bear_eng = (c.shift(1) > o.shift(1)) & (c < o.shift(1)) & (o > c.shift(1))

    rsi_exit_long = rsi_val > rsi_exit
    rsi_exit_short = rsi_val < (100 - rsi_exit)

    signal = pd.Series(0, index=c.index)
    # We signal entry only (exit managed by backtest engine)
    signal[bull_eng] = 1
    signal[bear_eng] = -1
    # Override with exit signals (neutral)
    signal[rsi_exit_long & (signal.shift(1) == 1)] = 0
    return signal


def space_BullishEngulfing(trial) -> dict:
    return {
        "rsi_len": trial.suggest_int("rsi_len", 2, 14),
        "rsi_exit": trial.suggest_int("rsi_exit", 70, 95, step=5),
    }


# ===========================================================================
# 12. Inside Day + RSI
# ===========================================================================

def gen_InsideDay(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Inside bar (today's high-low inside yesterday's) → long entry.
    Exit when RSI > overbought threshold.
    """
    rsi_len = params.get("rsi_len", 5)
    ob_level = params.get("ob_level", 80)

    h, l, c = df["high"], df["low"], df["close"]
    rsi_val = _rsi(c, rsi_len)

    is_inside = (h.shift(1) > h) & (l.shift(1) < l)
    rsi_ob = rsi_val > ob_level
    rsi_os = rsi_val < (100 - ob_level)

    signal = pd.Series(0, index=c.index)
    signal[is_inside] = 1       # Long on inside bar
    signal[is_inside & (c < c.shift(1))] = -1  # Short inside bar with down close
    return signal


def space_InsideDay(trial) -> dict:
    return {
        "rsi_len": trial.suggest_int("rsi_len", 2, 14),
        "ob_level": trial.suggest_int("ob_level", 70, 90, step=5),
    }


# ===========================================================================
# 13. Stan Weinstein Stage 2 Breakout (adapted, no request.security)
# ===========================================================================

def gen_StanWeinstein(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Stage 2 breakout: price above MA, breaks above recent high,
    volume above MA. RS proxy: relative strength vs own MA trend.
    Exit: price crosses below MA.
    """
    rs_period = params.get("rs_period", 50)
    vol_ma_len = params.get("vol_ma_len", 5)
    price_ma_len = params.get("price_ma_len", 30)
    highest_lookback = params.get("highest_lookback", 52)

    c, v = df["close"], df["volume"]
    price_ma = _sma(c, price_ma_len)
    vol_ma = _sma(v, vol_ma_len)

    # RS proxy: ratio of price now vs price rs_period bars ago
    rs_value = c / c.shift(rs_period) - 1

    # Highest high over lookback (excluding current bar like Pine [1])
    highest_high = df["high"].shift(1).rolling(highest_lookback).max()

    long_entry = (c > price_ma) & (rs_value > 0) & (v > vol_ma) & (c > highest_high)
    long_exit = c < price_ma

    signal = pd.Series(0, index=c.index)
    signal[long_entry] = 1
    signal[long_exit] = -1
    return signal


def space_StanWeinstein(trial) -> dict:
    return {
        "rs_period": trial.suggest_int("rs_period", 20, 100, step=10),
        "vol_ma_len": trial.suggest_int("vol_ma_len", 3, 15),
        "price_ma_len": trial.suggest_int("price_ma_len", 15, 60, step=5),
        "highest_lookback": trial.suggest_int("highest_lookback", 20, 100, step=10),
    }


# ===========================================================================
# 14. Breakout Sniper (N-bar high/low breakout)
# ===========================================================================

def gen_BreakoutSniper(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Enter long when price breaks above N-bar high. Short on N-bar low breakdown.
    Exit after hold_period bars.
    """
    lookback = params.get("lookback", 100)
    hold = params.get("hold", 30)

    h, l, c = df["high"], df["low"], df["close"]
    highest_high = h.rolling(lookback).max()
    lowest_low = l.rolling(lookback).min()

    breakout = h >= highest_high.shift(1)
    breakdown = l <= lowest_low.shift(1)

    signal = pd.Series(0, index=c.index)
    signal[breakout] = 1
    signal[breakdown] = -1

    # Apply hold period: exit after hold bars
    bars_since_breakout = pd.Series(np.inf, index=c.index)
    bars_since_breakdown = pd.Series(np.inf, index=c.index)
    cnt_l = 0; cnt_s = 0
    for i in range(len(c)):
        if breakout.iloc[i]:
            cnt_l = 0
        else:
            cnt_l += 1
        if breakdown.iloc[i]:
            cnt_s = 0
        else:
            cnt_s += 1
        bars_since_breakout.iloc[i] = cnt_l
        bars_since_breakdown.iloc[i] = cnt_s

    # Force exit signal after hold period (let engine handle; just signal 0)
    signal[bars_since_breakout > hold] = 0
    signal[bars_since_breakdown > hold] = 0
    return signal


def space_BreakoutSniper(trial) -> dict:
    return {
        "lookback": trial.suggest_int("lookback", 20, 200, step=20),
        "hold": trial.suggest_int("hold", 10, 60, step=5),
    }


# ===========================================================================
# 15. Price Channels (EMA + ATR)
# ===========================================================================

def gen_PriceChannels(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    EMA ± ATR*mult channels. Long on crossover above support (ema - atr*mult).
    Exit after hold bars or when price exceeds resistance.
    """
    ema_len = params.get("ema_len", 21)
    atr_mult = params.get("atr_mult", 2.0)
    hold = params.get("hold", 30)

    h, l, c = df["high"], df["low"], df["close"]
    ma = _ema(c, ema_len)
    atr = _atr(h, l, c, 14)
    support = ma - atr * atr_mult
    resistance = ma + atr * atr_mult

    long_entry = _crossover(c, support)
    long_exit_hold = pd.Series(False, index=c.index)
    long_exit_over = _crossover(c, resistance)

    # count bars since entry
    cnt = 0
    in_trade = False
    signal = pd.Series(0, index=c.index)
    for i in range(len(c)):
        if long_entry.iloc[i]:
            signal.iloc[i] = 1
            in_trade = True
            cnt = 0
        elif in_trade:
            cnt += 1
            if cnt > hold or long_exit_over.iloc[i]:
                signal.iloc[i] = 0
                in_trade = False
    return signal


def space_PriceChannels(trial) -> dict:
    return {
        "ema_len": trial.suggest_int("ema_len", 10, 50, step=5),
        "atr_mult": trial.suggest_float("atr_mult", 1.0, 4.0, step=0.5),
        "hold": trial.suggest_int("hold", 10, 60, step=5),
    }


# ===========================================================================
# 16. 2-Tier Ichimoku (simplified, no multi-TF)
# ===========================================================================

def gen_Ichimoku2Tier(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Ichimoku Cloud breakout. Long when price breaks above Kumo (cloud).
    Short when price breaks below Kumo.
    Confirmation: Tenkan > Kijun for long, Tenkan < Kijun for short.
    """
    tenkan_len = params.get("tenkan_len", 9)
    kijun_len = params.get("kijun_len", 26)
    senkou_b_len = params.get("senkou_b_len", 52)

    h, l, c = df["high"], df["low"], df["close"]

    tenkan = (h.rolling(tenkan_len).max() + l.rolling(tenkan_len).min()) / 2
    kijun = (h.rolling(kijun_len).max() + l.rolling(kijun_len).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(kijun_len)
    senkou_b = ((h.rolling(senkou_b_len).max() + l.rolling(senkou_b_len).min()) / 2).shift(kijun_len)

    kumo_top = senkou_a.combine(senkou_b, max)
    kumo_bot = senkou_a.combine(senkou_b, min)

    above_kumo = c > kumo_top
    below_kumo = c < kumo_bot
    bull_cross = tenkan > kijun
    bear_cross = tenkan < kijun

    long_entry = _crossover(c, kumo_top) & bull_cross
    short_entry = _crossunder(c, kumo_bot) & bear_cross

    signal = pd.Series(0, index=c.index)
    signal[long_entry] = 1
    signal[short_entry] = -1
    return signal


def space_Ichimoku2Tier(trial) -> dict:
    return {
        "tenkan_len": trial.suggest_int("tenkan_len", 7, 15),
        "kijun_len": trial.suggest_int("kijun_len", 20, 35),
        "senkou_b_len": trial.suggest_int("senkou_b_len", 40, 70, step=5),
    }


# ===========================================================================
# 17. Altered OBV on MACD
# ===========================================================================

def gen_AlteredOBVMACD(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Altered OBV (direction based on open/close relationship) run through MACD.
    Long: MACD crosses above signal and MACD is rising.
    Short: MACD falls below signal and is falling.
    """
    fast_len = params.get("fast_len", 12)
    slow_len = params.get("slow_len", 26)
    signal_len = params.get("signal_len", 9)

    c, o, v = df["close"], df["open"], df["volume"]

    # Altered OBV: if (close < prev_close and open < close) or close > prev_close → +1, else -1
    chng = pd.Series(0, index=c.index)
    chng[(c < c.shift(1)) & (o < c)] = 1
    chng[c > c.shift(1)] = 1
    chng[(c >= c.shift(1)) & ~(c > c.shift(1))] = -1
    chng = chng.where(chng != 0, -1)
    obvalt = (np.sign(chng) * v).cumsum()

    macd_line = _ema(obvalt, fast_len) - _ema(obvalt, slow_len)
    sig_line = _ema(macd_line, signal_len)

    macd_rising = macd_line > macd_line.shift(1)
    macd_falling = macd_line < macd_line.shift(1)
    macd_long = _crossover(macd_line, sig_line)
    macd_below_sig = macd_line < sig_line
    low_falling = df["low"] < df["low"].shift(1)

    long_cond = macd_long & macd_rising
    short_cond = macd_falling & macd_below_sig & low_falling

    signal = pd.Series(0, index=c.index)
    signal[long_cond] = 1
    signal[short_cond] = -1
    return signal


def space_AlteredOBVMACD(trial) -> dict:
    return {
        "fast_len": trial.suggest_int("fast_len", 6, 20),
        "slow_len": trial.suggest_int("slow_len", 20, 40),
        "signal_len": trial.suggest_int("signal_len", 5, 15),
    }


# ===========================================================================
# 18. Auto Fib Golden Pocket Band
# ===========================================================================

def gen_AutoFibGoldenPocket(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Three Fibonacci band layers (strong/middle/sideways) using dynamic high/low range.
    Buy: price touches and bounces above the EMA of fib level.
    Sell: price touches and drops below the EMA of fib level.
    """
    strong_lookback = params.get("strong_lookback", 400)
    strong_ema_len = params.get("strong_ema_len", 120)
    strong_fib_bull = params.get("strong_fib_bull", 0.236)
    middle_lookback = params.get("middle_lookback", 900)
    middle_ema_len = params.get("middle_ema_len", 400)
    middle_fib_bull = params.get("middle_fib_bull", 0.618)

    h, l, c = df["high"], df["low"], df["close"]

    def fib_ema_level(lookback, fib_lvl, ema_len):
        ph = h.rolling(lookback).max()
        pl = l.rolling(lookback).min()
        fib_price = (ph - pl) * (1 - fib_lvl) + pl
        return _ema(fib_price, ema_len)

    strong_ema = fib_ema_level(strong_lookback, strong_fib_bull, strong_ema_len)
    middle_ema = fib_ema_level(middle_lookback, middle_fib_bull, middle_ema_len)

    # Buy: low touched below fib EMA (within last bar) and now price is above
    strong_buy = (l.shift(1) < strong_ema.shift(1)) & (c > strong_ema)
    middle_buy = (l.shift(1) < middle_ema.shift(1)) & (c > middle_ema)

    strong_sell = (h.shift(1) > strong_ema.shift(1)) & (c < strong_ema)
    middle_sell = (h.shift(1) > middle_ema.shift(1)) & (c < middle_ema)

    signal = pd.Series(0, index=c.index)
    signal[strong_buy | middle_buy] = 1
    signal[strong_sell | middle_sell] = -1
    return signal


def space_AutoFibGoldenPocket(trial) -> dict:
    return {
        "strong_lookback": trial.suggest_int("strong_lookback", 100, 600, step=100),
        "strong_ema_len": trial.suggest_int("strong_ema_len", 50, 200, step=50),
        "strong_fib_bull": trial.suggest_float("strong_fib_bull", 0.236, 0.786, step=0.118),
        "middle_lookback": trial.suggest_int("middle_lookback", 300, 1200, step=300),
        "middle_ema_len": trial.suggest_int("middle_ema_len", 100, 500, step=100),
        "middle_fib_bull": trial.suggest_float("middle_fib_bull", 0.382, 0.886, step=0.118),
    }


# ===========================================================================
# 19. 365-Day High Breakout (adapted to 100-bar)
# ===========================================================================

def gen_HighBreakout365(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Breakout above N-bar high (adapting 365-day/250-bar to configurable lookback).
    Filtered by MA alignment (50 > 150 > 200) and relative strength vs own trend.
    ATR trailing stop exit.
    """
    lookback = params.get("lookback", 100)
    ma_fast = params.get("ma_fast", 50)
    ma_mid = params.get("ma_mid", 150)
    ma_slow = params.get("ma_slow", 200)
    rs_period = params.get("rs_period", 50)
    atr_period = params.get("atr_period", 14)
    atr_mult = params.get("atr_mult", 3.5)

    h, l, c = df["high"], df["low"], df["close"]

    ma50 = _sma(c, ma_fast)
    ma150 = _sma(c, ma_mid)
    ma200 = _sma(c, ma_slow)

    rs = c / c.shift(rs_period) - 1  # simple RS proxy

    highest = h.shift(1).rolling(lookback).max()
    ma_aligned = (ma50 > ma150) & (ma150 > ma200)
    is_buy = (c > highest) & (rs > 0) & ma_aligned & (c > ma50) & (c > ma200)

    # ATR trailing stop
    atr = _atr(h, l, c, atr_period)
    n_loss = atr_mult * atr

    ts = c.copy()
    for i in range(1, len(c)):
        prev_ts = ts.iloc[i - 1]
        prev_c = c.iloc[i - 1]
        cur_c = c.iloc[i]
        cur_loss = n_loss.iloc[i]
        if cur_c > prev_ts and prev_c > prev_ts:
            ts.iloc[i] = max(prev_ts, cur_c - cur_loss)
        elif cur_c < prev_ts and prev_c < prev_ts:
            ts.iloc[i] = min(prev_ts, cur_c + cur_loss)
        elif cur_c > prev_ts:
            ts.iloc[i] = cur_c - cur_loss
        else:
            ts.iloc[i] = cur_c + cur_loss

    buy_close = c < ts.shift(1)

    signal = pd.Series(0, index=c.index)
    signal[is_buy] = 1
    signal[buy_close & ~is_buy] = 0  # exit handled by engine
    return signal


def space_HighBreakout365(trial) -> dict:
    return {
        "lookback": trial.suggest_int("lookback", 50, 250, step=25),
        "ma_fast": trial.suggest_int("ma_fast", 30, 70, step=10),
        "ma_mid": trial.suggest_int("ma_mid", 100, 200, step=25),
        "ma_slow": trial.suggest_int("ma_slow", 150, 250, step=25),
        "rs_period": trial.suggest_int("rs_period", 20, 80, step=10),
        "atr_period": trial.suggest_int("atr_period", 7, 21),
        "atr_mult": trial.suggest_float("atr_mult", 2.0, 5.0, step=0.5),
    }


# ===========================================================================
# 20. 3x SuperTrend Confluence
# ===========================================================================

def gen_TripleSuperTrend(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Three SuperTrend lines (fast/medium/slow) + StochRSI + EMA200 filter.
    Long: EMA uptrend, StochRSI crossover below threshold, ≥1 ST bullish.
    Short: EMA downtrend, StochRSI crossunder above threshold, ≥1 ST bearish.
    """
    ema_len = params.get("ema_len", 200)
    st_slow_len = params.get("st_slow_len", 12)
    st_slow_mult = params.get("st_slow_mult", 3)
    st_med_len = params.get("st_med_len", 11)
    st_med_mult = params.get("st_med_mult", 2)
    st_fast_len = params.get("st_fast_len", 10)
    st_fast_mult = params.get("st_fast_mult", 1)
    srsi_len_rsi = params.get("srsi_len_rsi", 14)
    srsi_len_stoch = params.get("srsi_len_stoch", 14)
    srsi_smooth_k = params.get("srsi_smooth_k", 3)
    srsi_smooth_d = params.get("srsi_smooth_d", 3)
    srsi_thresh = params.get("srsi_thresh", 20)

    h, l, c = df["high"], df["low"], df["close"]

    ema_val = _ema(c, ema_len)
    ema_trend = (c > ema_val).map({True: 1, False: -1})

    _, trend_slow = _supertrend(h, l, c, st_slow_len, st_slow_mult)
    _, trend_med = _supertrend(h, l, c, st_med_len, st_med_mult)
    _, trend_fast = _supertrend(h, l, c, st_fast_len, st_fast_mult)

    st_count = trend_slow + trend_med + trend_fast  # range -3..+3

    k, d = _stoch_rsi(c, srsi_len_rsi, srsi_len_stoch, srsi_smooth_k, srsi_smooth_d)

    stoch_long = (k < srsi_thresh) & _crossover(k, d)
    stoch_short = (k > (100 - srsi_thresh)) & _crossunder(k, d)

    long_cond = (ema_trend > 0) & stoch_long & (st_count >= 1)
    short_cond = (ema_trend < 0) & stoch_short & (st_count <= -1)

    signal = pd.Series(0, index=c.index)
    signal[long_cond] = 1
    signal[short_cond] = -1
    return signal


def space_TripleSuperTrend(trial) -> dict:
    return {
        "ema_len": trial.suggest_int("ema_len", 100, 300, step=50),
        "st_slow_len": trial.suggest_int("st_slow_len", 8, 20),
        "st_slow_mult": trial.suggest_int("st_slow_mult", 2, 5),
        "st_med_len": trial.suggest_int("st_med_len", 7, 15),
        "st_med_mult": trial.suggest_int("st_med_mult", 1, 4),
        "st_fast_len": trial.suggest_int("st_fast_len", 5, 12),
        "st_fast_mult": trial.suggest_int("st_fast_mult", 1, 3),
        "srsi_len_rsi": trial.suggest_int("srsi_len_rsi", 7, 21),
        "srsi_len_stoch": trial.suggest_int("srsi_len_stoch", 7, 21),
        "srsi_smooth_k": trial.suggest_int("srsi_smooth_k", 2, 5),
        "srsi_smooth_d": trial.suggest_int("srsi_smooth_d", 2, 5),
        "srsi_thresh": trial.suggest_int("srsi_thresh", 10, 30),
    }


# ===========================================================================
# STRATEGY_EXPORT
# ===========================================================================

STRATEGY_EXPORT = {
    "AlphaTrend": {"gen": gen_AlphaTrend, "space": space_AlphaTrend},
    "BBKeltnerSqueeze": {"gen": gen_BBKeltnerSqueeze, "space": space_BBKeltnerSqueeze},
    "ALMACross": {"gen": gen_ALMACross, "space": space_ALMACross},
    "ATRMeanReversion": {"gen": gen_ATRMeanReversion, "space": space_ATRMeanReversion},
    "AdaptivePriceChannel": {"gen": gen_AdaptivePriceChannel, "space": space_AdaptivePriceChannel},
    "AthenaMomentumSqueeze": {"gen": gen_AthenaMomentumSqueeze, "space": space_AthenaMomentumSqueeze},
    "QFLMeanReversal": {"gen": gen_QFLMeanReversal, "space": space_QFLMeanReversal},
    "VADER_DEB": {"gen": gen_VADER_DEB, "space": space_VADER_DEB},
    "BayesianSuperTrend": {"gen": gen_BayesianSuperTrend, "space": space_BayesianSuperTrend},
    "BBEMARSIADXReversal": {"gen": gen_BBEMARSIADXReversal, "space": space_BBEMARSIADXReversal},
    "BullishEngulfing": {"gen": gen_BullishEngulfing, "space": space_BullishEngulfing},
    "InsideDay": {"gen": gen_InsideDay, "space": space_InsideDay},
    "StanWeinstein": {"gen": gen_StanWeinstein, "space": space_StanWeinstein},
    "BreakoutSniper": {"gen": gen_BreakoutSniper, "space": space_BreakoutSniper},
    "PriceChannels": {"gen": gen_PriceChannels, "space": space_PriceChannels},
    "Ichimoku2Tier": {"gen": gen_Ichimoku2Tier, "space": space_Ichimoku2Tier},
    "AlteredOBVMACD": {"gen": gen_AlteredOBVMACD, "space": space_AlteredOBVMACD},
    "AutoFibGoldenPocket": {"gen": gen_AutoFibGoldenPocket, "space": space_AutoFibGoldenPocket},
    "HighBreakout365": {"gen": gen_HighBreakout365, "space": space_HighBreakout365},
    "TripleSuperTrend": {"gen": gen_TripleSuperTrend, "space": space_TripleSuperTrend},
}
