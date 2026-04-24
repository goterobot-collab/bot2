"""
Batch 3704 - Mac paralela wave m5 - 5 academic-paper-based families.

Owner: Mac paralela (Claude Code online, rama claude/grail-hunt-bot-noeIq).

Each family derives its rule from a specific published academic paper.
All checked for ZERO overlap with sandbox batches 3566-3610, Mac V8 prod,
and Mac batches 3700-3703.

 1. TV_LiuTsyvinski_MultiHorizon_TSMOM  (Liu-Tsyvinski 2021 RFS / 2022 JF)
    - Time-series momentum with 1-4 week formation (adapted to 7/14/21/28d)
    - Paper: "Risks and Returns of Cryptocurrency" RFS 2021;
             "Common Risk Factors in Cryptocurrency" JF 2022

 2. TV_Detzel_MA_Ensemble              (Detzel et al. 2018 "Bitcoin:
    Predictability and Profitability via Technical Analysis")
    - 5-to-100 day MA ensemble voting (5,20,50,100,200)

 3. TV_FracassiKogan_PureMomentum      (Fracassi-Kogan SSRN 4138685
    "Pure Momentum in Cryptocurrency Markets")
    - Past-30d return percentile of own history

 4. TV_Dobrynskaya_MomentumReversal    (Dobrynskaya HSE conference paper
    "Cryptocurrency Momentum and Reversal")
    - Short-term reversal (5d) within medium-term momentum (20d) regime

 5. TV_Cafferata_Disposition           (Cafferata-Patacca-Tramontana 2025
    "Disposition effect and its outcome on endogenous price fluctuations"
    Decisions in Economics and Finance 2025)
    - Consecutive-direction count + RSI overshoot = disposition-driven reversal

All pickle-safe, no lambdas, signal integer in {-1, 0, 1}, .shift(1) applied.
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _sma(s, n):
    return s.rolling(int(n), min_periods=int(n)).mean()


def _rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


# ----- 1) LIU-TSYVINSKI 1-4 WEEK TSMOM -----
def gen_TV_LiuTsyvinski_MultiHorizon_TSMOM(df, h1=7, h2=14, h3=21, h4=28,
                                            vote_threshold=3, **kw):
    """Liu-Tsyvinski 2021 RFS / 2022 JF: sort cryptos by 1-to-4 week returns,
    winner-minus-loser yields ~3% weekly excess return. Adapted to single-asset
    time-series momentum: compute sign(ROC) at 4 horizons (7/14/21/28 days),
    count how many horizons agree. If >= vote_threshold agree long, enter long.
    Symmetric for short.
    """
    c = df['close'].astype(float)
    h1, h2, h3, h4 = int(h1), int(h2), int(h3), int(h4)
    vt = int(vote_threshold)
    r1 = (c / c.shift(h1) - 1.0)
    r2 = (c / c.shift(h2) - 1.0)
    r3 = (c / c.shift(h3) - 1.0)
    r4 = (c / c.shift(h4) - 1.0)
    longs = ((r1 > 0).astype(int) + (r2 > 0).astype(int)
             + (r3 > 0).astype(int) + (r4 > 0).astype(int))
    shorts = ((r1 < 0).astype(int) + (r2 < 0).astype(int)
              + (r3 < 0).astype(int) + (r4 < 0).astype(int))
    long_trigger = (longs >= vt) & (longs.shift(1) < vt)
    short_trigger = (shorts >= vt) & (shorts.shift(1) < vt)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_trigger.shift(1).fillna(False).astype(bool)] = 1
    sig[short_trigger.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_LiuTsyvinski_MultiHorizon_TSMOM():
    return {
        'h1': ('int', 5, 10),
        'h2': ('int', 10, 18),
        'h3': ('int', 18, 25),
        'h4': ('int', 25, 35),
        'vote_threshold': ('int', 2, 4),
    }


# ----- 2) DETZEL 5-TO-100 DAY MA ENSEMBLE -----
def gen_TV_Detzel_MA_Ensemble(df, ma_short1=5, ma_short2=20, ma_mid=50,
                                ma_long=100, ma_very_long=200,
                                vote_threshold=3, **kw):
    """Detzel et al. 2018: 5-to-100 day moving average rules collectively
    generate alpha for Bitcoin. Ensemble: count how many "price > MA(n)"
    signals are bullish across {5, 20, 50, 100, 200}. If >= vote_threshold
    bullish votes, enter long; <= (5 - vote_threshold) bearish, enter short.
    """
    c = df['close'].astype(float)
    mas = [_sma(c, int(n)) for n in (ma_short1, ma_short2, ma_mid, ma_long, ma_very_long)]
    bullish_count = sum(((c > m).astype(int) for m in mas), start=pd.Series(0, index=df.index))
    vt = int(vote_threshold)
    long_trigger = (bullish_count >= vt) & (bullish_count.shift(1) < vt)
    short_trigger = (bullish_count <= (5 - vt)) & (bullish_count.shift(1) > (5 - vt))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_trigger.shift(1).fillna(False).astype(bool)] = 1
    sig[short_trigger.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Detzel_MA_Ensemble():
    return {
        'ma_short1': ('int', 3, 10),
        'ma_short2': ('int', 15, 25),
        'ma_mid': ('int', 40, 60),
        'ma_long': ('int', 80, 120),
        'ma_very_long': ('int', 150, 250),
        'vote_threshold': ('int', 3, 5),
    }


# ----- 3) FRACASSI-KOGAN PURE MOMENTUM -----
def gen_TV_FracassiKogan_PureMomentum(df, form_len=30, lookback_pct=120,
                                        pct_long=80, pct_short=20, **kw):
    """Fracassi-Kogan SSRN 4138685: past-N-day return ranked against rolling
    window of own history. If today's formation return in top pct_long
    percentile of last lookback_pct days -> LONG; bottom pct_short -> SHORT.
    """
    c = df['close'].astype(float)
    fl = int(form_len)
    lp = int(lookback_pct)
    pl = float(pct_long)
    ps = float(pct_short)
    form_ret = (c / c.shift(fl) - 1.0)
    # rolling percentile rank
    def _pct_rank(x):
        a = x.to_numpy(dtype=float)
        if np.all(np.isnan(a)):
            return np.nan
        current = a[-1]
        if np.isnan(current):
            return np.nan
        valid = a[~np.isnan(a)]
        if len(valid) == 0:
            return np.nan
        return 100.0 * (valid < current).sum() / len(valid)
    pct_rank = form_ret.rolling(lp, min_periods=max(10, lp // 3)).apply(_pct_rank, raw=False)
    long_trigger = (pct_rank >= pl) & (pct_rank.shift(1) < pl)
    short_trigger = (pct_rank <= ps) & (pct_rank.shift(1) > ps)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_trigger.shift(1).fillna(False).astype(bool)] = 1
    sig[short_trigger.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_FracassiKogan_PureMomentum():
    return {
        'form_len': ('int', 14, 60),
        'lookback_pct': ('int', 60, 250),
        'pct_long': ('int', 70, 95),
        'pct_short': ('int', 5, 30),
    }


# ----- 4) DOBRYNSKAYA SHORT-REVERSAL / MEDIUM-MOMENTUM -----
def gen_TV_Dobrynskaya_MomentumReversal(df, short_len=5, med_len=20,
                                          short_rev_pct=5.0, **kw):
    """Dobrynskaya (HSE): cryptocurrencies exhibit short-term reversal AND
    medium-term momentum. Rule: if past short_len-day return < -short_rev_pct
    AND past med_len-day return > 0 -> LONG (mean-revert within uptrend).
    Symmetric: short_len return > +short_rev_pct AND med_len return < 0
    -> SHORT (mean-revert within downtrend).
    """
    c = df['close'].astype(float)
    sl = int(short_len)
    ml = int(med_len)
    srp = float(short_rev_pct) / 100.0
    r_short = (c / c.shift(sl) - 1.0)
    r_med = (c / c.shift(ml) - 1.0)
    long_trigger = (r_short < -srp) & (r_med > 0)
    short_trigger = (r_short > srp) & (r_med < 0)
    # Only trigger on new entry (not while already true)
    long_entry = long_trigger & ~(long_trigger.shift(1).fillna(False))
    short_entry = short_trigger & ~(short_trigger.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Dobrynskaya_MomentumReversal():
    return {
        'short_len': ('int', 2, 10),
        'med_len': ('int', 14, 40),
        'short_rev_pct': ('float', 2.0, 12.0),
    }


# ----- 5) CAFFERATA DISPOSITION EFFECT -----
def gen_TV_Cafferata_Disposition(df, consec_len=5, rsi_len=14,
                                   rsi_ob=70, rsi_os=30, **kw):
    """Cafferata-Patacca-Tramontana 2025: disposition effect causes
    asymmetric hold periods - investors hold losers too long and sell winners
    too fast. Proxy rule: after N consecutive days of same-direction closes
    AND RSI in extreme territory, price will revert.
    Long entry: consec_len consecutive DOWN closes AND RSI < rsi_os.
    Short entry: consec_len consecutive UP closes AND RSI > rsi_ob.
    """
    c = df['close'].astype(float)
    cl = int(consec_len)
    rl = int(rsi_len)
    ob = float(rsi_ob)
    os = float(rsi_os)
    up_day = c > c.shift(1)
    dn_day = c < c.shift(1)
    up_streak = up_day.rolling(cl, min_periods=cl).sum() == cl
    dn_streak = dn_day.rolling(cl, min_periods=cl).sum() == cl
    r = _rsi(c, rl)
    long_trigger = dn_streak & (r < os)
    short_trigger = up_streak & (r > ob)
    # Only trigger on new entry (not while already true)
    long_entry = long_trigger & ~(long_trigger.shift(1).fillna(False))
    short_entry = short_trigger & ~(short_trigger.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Cafferata_Disposition():
    return {
        'consec_len': ('int', 3, 8),
        'rsi_len': ('int', 7, 21),
        'rsi_ob': ('int', 65, 85),
        'rsi_os': ('int', 15, 35),
    }


STRATEGY_EXPORT = {
    "TV_LiuTsyvinski_MultiHorizon_TSMOM": {
        "gen": gen_TV_LiuTsyvinski_MultiHorizon_TSMOM,
        "space": space_TV_LiuTsyvinski_MultiHorizon_TSMOM,
        "source": "mac_batch3704_LiuTsyvinski2021RFS_2022JF",
    },
    "TV_Detzel_MA_Ensemble": {
        "gen": gen_TV_Detzel_MA_Ensemble,
        "space": space_TV_Detzel_MA_Ensemble,
        "source": "mac_batch3704_Detzel2018_JFM",
    },
    "TV_FracassiKogan_PureMomentum": {
        "gen": gen_TV_FracassiKogan_PureMomentum,
        "space": space_TV_FracassiKogan_PureMomentum,
        "source": "mac_batch3704_FracassiKogan_SSRN4138685",
    },
    "TV_Dobrynskaya_MomentumReversal": {
        "gen": gen_TV_Dobrynskaya_MomentumReversal,
        "space": space_TV_Dobrynskaya_MomentumReversal,
        "source": "mac_batch3704_Dobrynskaya_HSE",
    },
    "TV_Cafferata_Disposition": {
        "gen": gen_TV_Cafferata_Disposition,
        "space": space_TV_Cafferata_Disposition,
        "source": "mac_batch3704_CafferataPatacca Tramontana2025_DEF",
    },
}
