"""
Batch 3711 - Mac paralela wave m12 - 5 Market Wizards / classic systematic.

Each family attributed to a documented professional trader / system:

 1. TV_PaulTudorJones_200MA_Bias - PTJ rule: never long below 200d MA,
                                    never short above. Asymmetric.
 2. TV_Druckenmiller_RegimeBet   - Stan Druckenmiller multi-TF regime
                                    alignment (200MA + 50MA + momentum)
 3. TV_TurtleSystem_55Day        - Richard Dennis Turtle System #2:
                                    55-day Donchian breakout (slower than
                                    sandbox's TV_Donchian_Channel param range)
 4. TV_O_Neil_NewHigh_Volume     - William O'Neil CANSLIM-style: new
                                    N-period high WITH volume surge
                                    (NOT in repo - novel)
 5. TV_Livermore_PivotalPoint    - Jesse Livermore "pivotal point" =
                                    multi-bar HH/LL break with extension
                                    (DISTINCT from sandbox Pivot variants
                                    which are Floor/Camarilla style)

ZERO overlap with sandbox 3566-3610, Mac V8 prod, Mac 3700-3710.
Low-param (1-3 each), pickle-safe, signal int {-1,0,1}, .shift(1).
"""
import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def _sma(s, n):
    return s.rolling(int(n), min_periods=int(n)).mean()


# ----- 1) PAUL TUDOR JONES 200 MA ASYMMETRIC -----
def gen_TV_PaulTudorJones_200MA_Bias(df, ma_len=200, mom_len=20, **kw):
    """PTJ "never long below 200, never short above" asymmetric rule:
    - Long entry: close > MA(200) AND momentum(20d) > 0 AND fresh transition
    - Short entry: close < MA(200) AND momentum(20d) < 0 AND fresh transition
    No reverse trades when on wrong side of 200MA.
    """
    ml = int(ma_len)
    mol = int(mom_len)
    c = df['close'].astype(float)
    ma = _sma(c, ml)
    mom = (c / c.shift(mol) - 1.0)
    long_setup = (c > ma) & (mom > 0)
    short_setup = (c < ma) & (mom < 0)
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_PaulTudorJones_200MA_Bias():
    return {
        'ma_len': ('int', 100, 250),
        'mom_len': ('int', 10, 40),
    }


# ----- 2) DRUCKENMILLER REGIME BET -----
def gen_TV_Druckenmiller_RegimeBet(df, ma_long=200, ma_mid=50, mom_len=60,
                                     mom_pct=70, **kw):
    """Stan Druckenmiller "concentrated bet on regime alignment":
    Long: close > MA(200) AND close > MA(50) AND momentum(60d) percentile
          rank in last 252 days > mom_pct (i.e. top tail).
    Short: close < MA(200) AND close < MA(50) AND momentum(60d) pct < (100-mom_pct)
    """
    ml = int(ma_long)
    mm = int(ma_mid)
    mol = int(mom_len)
    mp = float(mom_pct)
    c = df['close'].astype(float)
    ma_l = _sma(c, ml)
    ma_m = _sma(c, mm)
    mom = (c / c.shift(mol) - 1.0)
    pct_rank = mom.rolling(252, min_periods=60).rank(pct=True) * 100
    long_setup = (c > ma_l) & (c > ma_m) & (pct_rank > mp)
    short_setup = (c < ma_l) & (c < ma_m) & (pct_rank < (100 - mp))
    long_entry = long_setup & ~(long_setup.shift(1).fillna(False))
    short_entry = short_setup & ~(short_setup.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Druckenmiller_RegimeBet():
    return {
        'ma_long': ('int', 100, 250),
        'ma_mid': ('int', 30, 80),
        'mom_len': ('int', 30, 100),
        'mom_pct': ('int', 60, 90),
    }


# ----- 3) TURTLE SYSTEM #2 (55-DAY) -----
def gen_TV_TurtleSystem_55Day(df, breakout_len=55, exit_len=20, **kw):
    """Richard Dennis Turtle System #2:
    Long: close > N-day high (entry breakout, default 55).
    Short: close < N-day low. Exits handled by signal flip / SL in engine.
    Slower than sandbox TV_Donchian_Channel typical 20-day param.
    """
    bl = int(breakout_len)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    entry_high = h.rolling(bl, min_periods=bl).max().shift(1)
    entry_low = l.rolling(bl, min_periods=bl).min().shift(1)
    long_break = (c > entry_high)
    short_break = (c < entry_low)
    long_entry = long_break & ~(long_break.shift(1).fillna(False))
    short_entry = short_break & ~(short_break.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_TurtleSystem_55Day():
    return {
        'breakout_len': ('int', 40, 100),
        'exit_len': ('int', 10, 35),
    }


# ----- 4) WILLIAM O'NEIL CANSLIM NEW HIGH + VOLUME -----
def gen_TV_O_Neil_NewHigh_Volume(df, hi_len=50, vol_mult=1.5, vol_ma=20, **kw):
    """O'Neil pattern: stock breaks to new N-period high WITH volume above
    vol_mult * SMA(volume, vol_ma). Symmetric short on new low + vol.
    """
    hl = int(hi_len)
    vm = float(vol_mult)
    vma = int(vol_ma)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    new_high = c > h.rolling(hl, min_periods=hl).max().shift(1)
    new_low = c < l.rolling(hl, min_periods=hl).min().shift(1)
    vol_threshold = _sma(v, vma) * vm
    big_vol = v > vol_threshold
    long_trigger = new_high & big_vol
    short_trigger = new_low & big_vol
    long_entry = long_trigger & ~(long_trigger.shift(1).fillna(False))
    short_entry = short_trigger & ~(short_trigger.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_O_Neil_NewHigh_Volume():
    return {
        'hi_len': ('int', 30, 100),
        'vol_mult': ('float', 1.2, 3.0),
        'vol_ma': ('int', 10, 40),
    }


# ----- 5) LIVERMORE PIVOTAL POINT -----
def gen_TV_Livermore_PivotalPoint(df, pivot_lookback=10, extension_pct=2.0, **kw):
    """Jesse Livermore "pivotal point": multi-bar high or low becomes
    a key level. Confirmed reversal when price extends extension_pct
    beyond the point. Distinct from sandbox Floor/Camarilla pivots which
    are session-based round-number computations.
    """
    pl = int(pivot_lookback)
    ep = float(extension_pct) / 100.0
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    # Local pivot: bar's high is highest in window centered on the bar
    # Approximate with prior pl bars max/min vs current bar
    rolling_high = h.rolling(pl, min_periods=pl).max()
    rolling_low = l.rolling(pl, min_periods=pl).min()
    is_pivot_high = (h.shift(pl // 2) == rolling_high.shift(pl // 2))
    is_pivot_low = (l.shift(pl // 2) == rolling_low.shift(pl // 2))
    last_pivot_high = h.shift(pl // 2).where(is_pivot_high).ffill()
    last_pivot_low = l.shift(pl // 2).where(is_pivot_low).ffill()
    long_break = c > last_pivot_high * (1 + ep)
    short_break = c < last_pivot_low * (1 - ep)
    long_entry = long_break & ~(long_break.shift(1).fillna(False))
    short_entry = short_break & ~(short_break.shift(1).fillna(False))
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_entry.shift(1).fillna(False).astype(bool)] = 1
    sig[short_entry.shift(1).fillna(False).astype(bool)] = -1
    return sig


def space_TV_Livermore_PivotalPoint():
    return {
        'pivot_lookback': ('int', 5, 25),
        'extension_pct': ('float', 0.5, 5.0),
    }


STRATEGY_EXPORT = {
    "TV_PaulTudorJones_200MA_Bias": {
        "gen": gen_TV_PaulTudorJones_200MA_Bias,
        "space": space_TV_PaulTudorJones_200MA_Bias,
        "source": "mac_batch3711_PTJ_200MA_asymmetric",
    },
    "TV_Druckenmiller_RegimeBet": {
        "gen": gen_TV_Druckenmiller_RegimeBet,
        "space": space_TV_Druckenmiller_RegimeBet,
        "source": "mac_batch3711_Druckenmiller_concentrated_regime",
    },
    "TV_TurtleSystem_55Day": {
        "gen": gen_TV_TurtleSystem_55Day,
        "space": space_TV_TurtleSystem_55Day,
        "source": "mac_batch3711_RichardDennis_Turtle_System2",
    },
    "TV_O_Neil_NewHigh_Volume": {
        "gen": gen_TV_O_Neil_NewHigh_Volume,
        "space": space_TV_O_Neil_NewHigh_Volume,
        "source": "mac_batch3711_W_ONeil_CANSLIM_HighVolBreakout",
    },
    "TV_Livermore_PivotalPoint": {
        "gen": gen_TV_Livermore_PivotalPoint,
        "space": space_TV_Livermore_PivotalPoint,
        "source": "mac_batch3711_JesseLivermore_PivotalPoint",
    },
}
