"""
Batch 3606 - Machine-learning-lite (closed-form, no sklearn at runtime)
Sources: Linear regression channel, Nadaraya-Watson kernel, Avellaneda-Lee stat arb,
         Box-Jenkins AR(1), Grinold-Kahn Information Coefficient
5 strategies, pickle-safe, no lambdas, no look-ahead.
"""
import numpy as np
import pandas as pd


def _rolling_ols_line(y, n):
    y = y.astype(float)
    idx = np.arange(int(n), dtype=float)
    xm = idx.mean()
    ss_x = float(((idx - xm) ** 2).sum())

    def _slope(v):
        if np.isnan(v).any(): return np.nan
        return float(((idx - xm) * (v - v.mean())).sum() / ss_x)

    def _intercept(v):
        if np.isnan(v).any(): return np.nan
        slope = ((idx - xm) * (v - v.mean())).sum() / ss_x
        return float(v.mean() - slope * xm)

    def _rstd(v):
        if np.isnan(v).any(): return np.nan
        slope = ((idx - xm) * (v - v.mean())).sum() / ss_x
        intercept = v.mean() - slope * xm
        resid = v - (slope * idx + intercept)
        return float(np.sqrt((resid ** 2).mean()))

    def _fit_end(v):
        if np.isnan(v).any(): return np.nan
        slope = ((idx - xm) * (v - v.mean())).sum() / ss_x
        intercept = v.mean() - slope * xm
        return float(slope * (len(idx) - 1) + intercept)

    r = y.rolling(int(n), min_periods=int(n))
    return r.apply(_slope, raw=True), r.apply(_intercept, raw=True), \
           r.apply(_rstd, raw=True), r.apply(_fit_end, raw=True)


def gen_TV_Linear_Regression_Channel(df, lr_len=60, k_std=2.0, **kw):
    c = df['close'].astype(float)
    _, _, rstd, fit_end = _rolling_ols_line(c, int(lr_len))
    upper = (fit_end + float(k_std) * rstd).shift(1)
    lower = (fit_end - float(k_std) * rstd).shift(1)
    below = c < lower
    above = c > upper
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[below.fillna(False)] = 1
    sig[above.fillna(False)] = -1
    return sig


def space_TV_Linear_Regression_Channel():
    return {'lr_len': ('int', 20, 200), 'k_std': ('float', 1.0, 3.5)}


def _kernel_forecast(ret_hist, bandwidth):
    ret_hist = np.asarray(ret_hist, dtype=float)
    n = ret_hist.size
    if n < 5 or not np.isfinite(ret_hist).all():
        return np.nan
    x = ret_hist[:-1]
    y = ret_hist[1:]
    x_cur = ret_hist[-1]
    h = float(bandwidth)
    if h <= 0: return np.nan
    w = np.exp(-0.5 * ((x - x_cur) / h) ** 2)
    ws = w.sum()
    if ws <= 0: return np.nan
    return float((w * y).sum() / ws)


def gen_TV_Kernel_Ridge_Score(df, hist_len=80, bandwidth=0.01, thresh=0.0005, **kw):
    c = df['close'].astype(float)
    r = np.log(c.replace(0.0, np.nan)).diff()
    vals = r.to_numpy(dtype=float)
    n = vals.size
    out = np.full(n, np.nan, dtype=float)
    w = int(hist_len)
    for i in range(w, n):
        out[i] = _kernel_forecast(vals[i-w: i], float(bandwidth))
    fc = pd.Series(out, index=df.index).shift(1)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[(fc > float(thresh)).fillna(False)] = 1
    sig[(fc < -float(thresh)).fillna(False)] = -1
    return sig


def space_TV_Kernel_Ridge_Score():
    return {'hist_len': ('int', 30, 300), 'bandwidth': ('float', 0.001, 0.05),
            'thresh': ('float', 0.0001, 0.005)}


def gen_TV_OLS_Residual_Fade(df, ema_fast=50, ema_slow=200, z_len=100, z_thresh=2.0, **kw):
    c = df['close'].astype(float)
    ema_f = c.ewm(span=int(ema_fast), adjust=False, min_periods=int(ema_fast)).mean()
    ema_s = c.ewm(span=int(ema_slow), adjust=False, min_periods=int(ema_slow)).mean()
    w = int(z_len)
    cv = c.to_numpy(dtype=float); ef = ema_f.to_numpy(dtype=float); es = ema_s.to_numpy(dtype=float)
    n = cv.size
    resid = np.full(n, np.nan, dtype=float)
    for i in range(w, n):
        X = np.column_stack([ef[i-w: i], es[i-w: i]])
        y = cv[i-w: i]
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            continue
        XtX = X.T @ X
        det = XtX[0,0]*XtX[1,1] - XtX[0,1]*XtX[1,0]
        if abs(det) < 1e-12: continue
        inv = np.array([[XtX[1,1], -XtX[0,1]], [-XtX[1,0], XtX[0,0]]]) / det
        beta = inv @ (X.T @ y)
        resid[i] = cv[i] - (beta[0]*ef[i] + beta[1]*es[i])
    resid_s = pd.Series(resid, index=df.index)
    mu = resid_s.rolling(w, min_periods=w).mean()
    sd = resid_s.rolling(w, min_periods=w).std(ddof=0)
    z = ((resid_s - mu) / sd.replace(0.0, np.nan)).shift(1)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[(z <= -float(z_thresh)).fillna(False)] = 1
    sig[(z >= float(z_thresh)).fillna(False)] = -1
    return sig


def space_TV_OLS_Residual_Fade():
    return {'ema_fast': ('int', 10, 100), 'ema_slow': ('int', 50, 400),
            'z_len': ('int', 30, 300), 'z_thresh': ('float', 1.0, 4.0)}


def _rolling_ar1_phi(r, n):
    r = np.asarray(r, dtype=float)
    m = r.size
    out = np.full(m, np.nan, dtype=float)
    if m < n + 2: return out
    for i in range(n + 1, m):
        y = r[i-n: i]
        x = r[i-n-1: i-1]
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            continue
        xm = x.mean(); ym = y.mean()
        denom = float(((x - xm) ** 2).sum())
        if denom <= 0: continue
        out[i] = float(((x - xm) * (y - ym)).sum() / denom)
    return out


def gen_TV_AR1_Forecast(df, ar_len=50, thresh=0.0003, **kw):
    c = df['close'].astype(float)
    r = np.log(c.replace(0.0, np.nan)).diff()
    phi = _rolling_ar1_phi(r.to_numpy(dtype=float), int(ar_len))
    phi_s = pd.Series(phi, index=df.index)
    forecast = (phi_s * r).shift(1)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[(forecast > float(thresh)).fillna(False)] = 1
    sig[(forecast < -float(thresh)).fillna(False)] = -1
    return sig


def space_TV_AR1_Forecast():
    return {'ar_len': ('int', 20, 300), 'thresh': ('float', 0.00005, 0.005)}


def _rank(a):
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, a.size + 1, dtype=float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros_like(counts, dtype=float)
        np.add.at(sums, inv, ranks)
        avg = sums / counts
        ranks = avg[inv]
    return ranks


def _rolling_spearman(x, y, n):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = x.size
    out = np.full(m, np.nan, dtype=float)
    for i in range(n - 1, m):
        xs = x[i-n+1: i+1]; ys = y[i-n+1: i+1]
        if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
            continue
        rx = _rank(xs); ry = _rank(ys)
        rxm, rym = rx.mean(), ry.mean()
        num = ((rx - rxm) * (ry - rym)).sum()
        den = np.sqrt(((rx - rxm) ** 2).sum() * ((ry - rym) ** 2).sum())
        if den <= 0: continue
        out[i] = float(num / den)
    return out


def gen_TV_Information_Coefficient(df, ic_len=50, ic_thresh=0.2, feat_smooth=5, **kw):
    c = df['close'].astype(float); v = df['volume'].astype(float)
    r = np.log(c.replace(0.0, np.nan)).diff()
    logv = np.log(v.replace(0.0, np.nan))
    feat = logv.ewm(span=int(feat_smooth), adjust=False, min_periods=int(feat_smooth)).mean()
    r_past = r.shift(1).to_numpy(dtype=float)
    f_past = feat.shift(1).to_numpy(dtype=float)
    ic = _rolling_spearman(f_past, r_past, int(ic_len))
    ic_s = pd.Series(ic, index=df.index).shift(1)
    z = ((feat - feat.rolling(int(ic_len), min_periods=int(ic_len)).mean())
         / feat.rolling(int(ic_len), min_periods=int(ic_len)).std(ddof=0).replace(0.0, np.nan)).shift(1)
    good = ic_s.abs() >= float(ic_thresh)
    pos_ic = good & (ic_s > 0)
    neg_ic = good & (ic_s < 0)
    long_cond = (pos_ic & (z > 0)) | (neg_ic & (z < 0))
    short_cond = (pos_ic & (z < 0)) | (neg_ic & (z > 0))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_cond.fillna(False)] = 1
    sig[short_cond.fillna(False)] = -1
    return sig


def space_TV_Information_Coefficient():
    return {'ic_len': ('int', 20, 200), 'ic_thresh': ('float', 0.05, 0.5),
            'feat_smooth': ('int', 1, 30)}


STRATEGY_EXPORT = {
    'TV_Linear_Regression_Channel': {'gen': gen_TV_Linear_Regression_Channel,
                                      'space': space_TV_Linear_Regression_Channel,
                                      'source': 'https://www.tradingview.com/support/solutions/43000501985-linear-regression/'},
    'TV_Kernel_Ridge_Score': {'gen': gen_TV_Kernel_Ridge_Score, 'space': space_TV_Kernel_Ridge_Score,
                               'source': 'https://en.wikipedia.org/wiki/Kernel_regression'},
    'TV_OLS_Residual_Fade': {'gen': gen_TV_OLS_Residual_Fade, 'space': space_TV_OLS_Residual_Fade,
                              'source': 'https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1153505'},
    'TV_AR1_Forecast': {'gen': gen_TV_AR1_Forecast, 'space': space_TV_AR1_Forecast,
                         'source': 'https://en.wikipedia.org/wiki/Autoregressive_model'},
    'TV_Information_Coefficient': {'gen': gen_TV_Information_Coefficient,
                                    'space': space_TV_Information_Coefficient,
                                    'source': 'https://en.wikipedia.org/wiki/Information_coefficient'},
}
