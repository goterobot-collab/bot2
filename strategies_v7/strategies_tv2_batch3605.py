"""
Batch 3605 - Ehlers cycle indicators
Sources: Ehlers J. (2001-2013) MAMA/FAMA, Sine wave, Fisher transform, Super Smoother, Trend Mode
5 strategies, pickle-safe, no lambdas, no look-ahead.
"""
import numpy as np
import pandas as pd


def _hilbert_mama_fama(c, fast_limit=0.5, slow_limit=0.05):
    c = np.asarray(c, dtype=float)
    n = c.size
    mama = np.full(n, np.nan, dtype=float)
    fama = np.full(n, np.nan, dtype=float)
    if n < 7:
        return mama, fama
    smooth = np.zeros(n); detrend = np.zeros(n)
    I1 = np.zeros(n); Q1 = np.zeros(n)
    jI = np.zeros(n); jQ = np.zeros(n)
    I2 = np.zeros(n); Q2 = np.zeros(n)
    Re = np.zeros(n); Im = np.zeros(n)
    period = np.zeros(n); smp = np.zeros(n); phase = np.zeros(n)
    fl, sl = float(fast_limit), float(slow_limit)
    for i in range(n):
        if i < 6:
            continue
        smooth[i] = (4*c[i] + 3*c[i-1] + 2*c[i-2] + c[i-3]) / 10.0
        detrend[i] = ((0.0962*smooth[i] + 0.5769*smooth[i-2]
                       - 0.5769*smooth[i-4] - 0.0962*smooth[i-6])
                      * (0.075*period[i-1] + 0.54))
        Q1[i] = ((0.0962*detrend[i] + 0.5769*detrend[i-2]
                  - 0.5769*detrend[i-4] - 0.0962*detrend[i-6])
                 * (0.075*period[i-1] + 0.54))
        I1[i] = detrend[i-3]
        jI[i] = ((0.0962*I1[i] + 0.5769*I1[i-2]
                  - 0.5769*I1[i-4] - 0.0962*I1[i-6])
                 * (0.075*period[i-1] + 0.54))
        jQ[i] = ((0.0962*Q1[i] + 0.5769*Q1[i-2]
                  - 0.5769*Q1[i-4] - 0.0962*Q1[i-6])
                 * (0.075*period[i-1] + 0.54))
        I2_raw = I1[i] - jQ[i]
        Q2_raw = Q1[i] + jI[i]
        I2[i] = 0.2*I2_raw + 0.8*I2[i-1]
        Q2[i] = 0.2*Q2_raw + 0.8*Q2[i-1]
        Re[i] = 0.2*(I2[i]*I2[i-1] + Q2[i]*Q2[i-1]) + 0.8*Re[i-1]
        Im[i] = 0.2*(I2[i]*Q2[i-1] - Q2[i]*I2[i-1]) + 0.8*Im[i-1]
        p = period[i-1]
        if Im[i] != 0.0 and Re[i] != 0.0:
            p = 2*np.pi / np.arctan2(Im[i], Re[i])
        p = min(max(p, 6.0), 50.0)
        if period[i-1] > 0:
            p = min(max(p, 0.67*period[i-1]), 1.5*period[i-1])
        period[i] = 0.2*p + 0.8*period[i-1]
        smp[i] = 0.33*period[i] + 0.67*smp[i-1]
        phase[i] = (180.0/np.pi)*np.arctan(Q1[i]/I1[i]) if I1[i] != 0.0 else phase[i-1]
        delta = max(phase[i-1] - phase[i], 1.0)
        alpha = min(max(fl/delta, sl), fl)
        if i == 6:
            mama[i] = c[i]; fama[i] = c[i]
        else:
            mama[i] = alpha*c[i] + (1-alpha)*mama[i-1]
            fama[i] = 0.5*alpha*mama[i] + (1 - 0.5*alpha)*fama[i-1]
    return mama, fama


def gen_TV_Ehlers_MESA_Cross(df, fast_limit=0.5, slow_limit=0.05, **kw):
    c = df['close'].astype(float).to_numpy()
    mama, fama = _hilbert_mama_fama(c, float(fast_limit), float(slow_limit))
    mama = pd.Series(mama, index=df.index)
    fama = pd.Series(fama, index=df.index)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[((mama > fama) & (mama.shift(1) <= fama.shift(1))).fillna(False)] = 1
    sig[((mama < fama) & (mama.shift(1) >= fama.shift(1))).fillna(False)] = -1
    return sig


def space_TV_Ehlers_MESA_Cross():
    return {'fast_limit': ('float', 0.1, 0.9), 'slow_limit': ('float', 0.01, 0.2)}


def _ehlers_sine(c, cycle_len=10):
    c = np.asarray(c, dtype=float)
    n = c.size
    sine = np.full(n, np.nan, dtype=float)
    lead = np.full(n, np.nan, dtype=float)
    cl = int(cycle_len)
    if n < 6 or cl < 3:
        return sine, lead
    smooth = np.zeros(n)
    for i in range(3, n):
        smooth[i] = (4*c[i] + 3*c[i-1] + 2*c[i-2] + c[i-3]) / 10.0
    for i in range(cl, n):
        I = smooth[i] - smooth[i - cl // 2]
        q_lag = max(1, cl // 4)
        Q = smooth[i] - smooth[i - q_lag]
        if np.hypot(I, Q) > 0:
            phase = np.arctan2(Q, I)
            sine[i] = np.sin(phase)
            lead[i] = np.sin(phase + np.pi / 4)
    return sine, lead


def gen_TV_Ehlers_Sine_Wave(df, cycle_len=10, **kw):
    c = df['close'].astype(float).to_numpy()
    s, ls = _ehlers_sine(c, int(cycle_len))
    s = pd.Series(s, index=df.index); ls = pd.Series(ls, index=df.index)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[((s > ls) & (s.shift(1) <= ls.shift(1))).fillna(False)] = 1
    sig[((s < ls) & (s.shift(1) >= ls.shift(1))).fillna(False)] = -1
    return sig


def space_TV_Ehlers_Sine_Wave():
    return {'cycle_len': ('int', 5, 40)}


def gen_TV_Ehlers_Fisher_Transform(df, fisher_len=10, **kw):
    h = df['high'].astype(float); l = df['low'].astype(float)
    med = (h + l) / 2.0
    n = int(fisher_len)
    hi = med.rolling(n, min_periods=n).max()
    lo = med.rolling(n, min_periods=n).min()
    rng = (hi - lo).replace(0.0, np.nan)
    raw = (2.0 * (med - lo) / rng - 1.0).clip(lower=-0.999, upper=0.999)
    vals = raw.to_numpy(dtype=float)
    m = vals.size
    xs = np.zeros(m); fisher = np.zeros(m)
    first = True
    for i in range(m):
        if not np.isfinite(vals[i]):
            xs[i] = 0.0 if first else xs[i-1]
            fisher[i] = 0.0 if first else fisher[i-1]
            continue
        if first:
            xs[i] = vals[i]
            fisher[i] = 0.5 * np.log((1 + xs[i]) / (1 - xs[i]))
            first = False
        else:
            xs[i] = min(max(0.33*vals[i] + 0.67*xs[i-1], -0.999), 0.999)
            fisher[i] = 0.5*np.log((1+xs[i])/(1-xs[i])) + 0.5*fisher[i-1]
    f = pd.Series(fisher, index=df.index)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[((f > 0) & (f.shift(1) <= 0)).fillna(False)] = 1
    sig[((f < 0) & (f.shift(1) >= 0)).fillna(False)] = -1
    return sig


def space_TV_Ehlers_Fisher_Transform():
    return {'fisher_len': ('int', 5, 40)}


def _super_smoother(c, length=20):
    c = np.asarray(c, dtype=float)
    n = c.size
    out = np.full(n, np.nan, dtype=float)
    if n < 3 or length < 2:
        return out
    a1 = np.exp(-1.414 * np.pi / float(length))
    b1 = 2.0 * a1 * np.cos(1.414 * np.pi / float(length))
    c2, c3 = b1, -a1 * a1
    c1 = 1.0 - c2 - c3
    out[0], out[1] = c[0], c[1]
    for i in range(2, n):
        out[i] = c1*(c[i]+c[i-1])/2.0 + c2*out[i-1] + c3*out[i-2]
    return out


def gen_TV_Ehlers_Super_Smoother(df, length=20, ref_len=50, **kw):
    c = df['close'].astype(float).to_numpy()
    ss = pd.Series(_super_smoother(c, int(length)), index=df.index)
    rising = ss > ss.shift(int(ref_len))
    falling = ss < ss.shift(int(ref_len))
    price = df['close'].astype(float)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[((price > ss) & (price.shift(1) <= ss.shift(1)) & rising).fillna(False)] = 1
    sig[((price < ss) & (price.shift(1) >= ss.shift(1)) & falling).fillna(False)] = -1
    return sig


def space_TV_Ehlers_Super_Smoother():
    return {'length': ('int', 5, 80), 'ref_len': ('int', 5, 200)}


def gen_TV_Ehlers_Trend_Mode(df, ss_fast=10, ss_slow=40, std_len=50, std_mult=1.0, **kw):
    c = df['close'].astype(float).to_numpy()
    ss_f = pd.Series(_super_smoother(c, int(ss_fast)), index=df.index)
    ss_s = pd.Series(_super_smoother(c, int(ss_slow)), index=df.index)
    diff = ss_f - ss_s
    sd = ss_f.rolling(int(std_len), min_periods=int(std_len)).std(ddof=0)
    in_trend = diff.abs() > float(std_mult) * sd
    price = df['close'].astype(float)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[(in_trend & (diff > 0) & (price > ss_f)).fillna(False)] = 1
    sig[(in_trend & (diff < 0) & (price < ss_f)).fillna(False)] = -1
    return sig


def space_TV_Ehlers_Trend_Mode():
    return {'ss_fast': ('int', 5, 30), 'ss_slow': ('int', 20, 120),
            'std_len': ('int', 20, 200), 'std_mult': ('float', 0.3, 3.0)}


STRATEGY_EXPORT = {
    'TV_Ehlers_MESA_Cross': {'gen': gen_TV_Ehlers_MESA_Cross, 'space': space_TV_Ehlers_MESA_Cross,
                              'source': 'https://www.mesasoftware.com/papers/MAMA.pdf'},
    'TV_Ehlers_Sine_Wave': {'gen': gen_TV_Ehlers_Sine_Wave, 'space': space_TV_Ehlers_Sine_Wave,
                             'source': 'https://www.mesasoftware.com/papers/TheSinewaveIndicator.pdf'},
    'TV_Ehlers_Fisher_Transform': {'gen': gen_TV_Ehlers_Fisher_Transform,
                                    'space': space_TV_Ehlers_Fisher_Transform,
                                    'source': 'https://www.mesasoftware.com/papers/UsingTheFisherTransform.pdf'},
    'TV_Ehlers_Super_Smoother': {'gen': gen_TV_Ehlers_Super_Smoother, 'space': space_TV_Ehlers_Super_Smoother,
                                  'source': 'https://www.mesasoftware.com/papers/PredictiveIndicators.pdf'},
    'TV_Ehlers_Trend_Mode': {'gen': gen_TV_Ehlers_Trend_Mode, 'space': space_TV_Ehlers_Trend_Mode,
                              'source': 'https://www.mesasoftware.com/papers/TrendModes.pdf'},
}
