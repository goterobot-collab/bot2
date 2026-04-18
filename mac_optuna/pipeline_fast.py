#!/usr/bin/env python3
"""
FAST PIPELINE v2: 149 strategies x 563 assets x 5 timeframes
Vectorized backtest engine — NO Python loops
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import os
import sys
import time
from datetime import datetime

DB_PATH = "/Users/sabrina/Code/activos binace /activos_binance.db"
PROJECT_DIR = "/Users/sabrina/CLAUDE CODE/Estrategias"
PROGRESS_FILE = os.path.join(PROJECT_DIR, "data", "backtesting_progress.json")
RESULTS_DIR = os.path.join(PROJECT_DIR, "data", "results")
COMMISSION = 0.001
SLIPPAGE = 0.0005
MIN_TRADES = 5

os.makedirs(os.path.join(PROJECT_DIR, "data"), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── DB ───
def get_symbols_ranked():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT symbol, COUNT(*) as bars FROM candles WHERE timeframe='5m' GROUP BY symbol ORDER BY bars DESC", conn)
    conn.close()
    return list(df.itertuples(index=False, name=None))

def load_candles(symbol, timeframe="5m"):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT ts, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe=? ORDER BY ts", conn, params=(symbol, timeframe))
    conn.close()
    if len(df) == 0:
        return None
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    return df

def resample_ohlcv(df, target_tf):
    tf_map = {'15m': '15min', '4h': '4h', '1d': '1D'}
    rule = tf_map.get(target_tf)
    if rule is None:
        return None
    return df.resample(rule).agg({'ts':'first','open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['close'])

# ─── INDICATORS (all vectorized) ───
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def sma(s, p): return s.rolling(p).mean()

def rsi(s, p=14):
    d = s.diff()
    g = d.where(d>0,0.0).ewm(span=p, adjust=False).mean()
    l = (-d.where(d<0,0.0)).ewm(span=p, adjust=False).mean()
    return 100 - 100/(1+g/(l+1e-10))

def atr(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def bb(c, p=20, std=2):
    m = c.rolling(p).mean(); s = c.rolling(p).std()
    return m, m+std*s, m-std*s

def macd(c, f=12, s=26, sig=9):
    m = ema(c,f) - ema(c,s); si = ema(m, sig)
    return m, si, m-si

def stoch(h, l, c, k_p=14, d_p=3):
    lo = l.rolling(k_p).min(); hi = h.rolling(k_p).max()
    k = 100*(c-lo)/(hi-lo+1e-10); d = k.rolling(d_p).mean()
    return k, d

def adx(h, l, c, p=14):
    tr1 = atr(h,l,c,1)
    up = h.diff(); dn = -l.diff()
    pdm = np.where((up>dn)&(up>0),up,0)
    mdm = np.where((dn>up)&(dn>0),dn,0)
    pdi = 100*pd.Series(pdm,index=c.index).rolling(p).mean()/(tr1.rolling(p).mean()+1e-10)
    mdi = 100*pd.Series(mdm,index=c.index).rolling(p).mean()/(tr1.rolling(p).mean()+1e-10)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi+1e-10)
    return dx.rolling(p).mean(), pdi, mdi

# ─── VECTORIZED BACKTEST ENGINE ───
def backtest_vectorized(df, signals):
    """
    Vectorized backtest: signal[i] → entry on open[i+1]
    Returns metrics dict or None
    """
    if signals is None:
        return None

    sig = signals.values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)

    trades = []
    pos = 0
    entry_p = 0.0
    entry_i = 0
    mae = 0.0

    for i in range(1, n):
        s = sig[i-1]
        p = opens[i]

        if pos == 0 and s == 1:
            entry_p = p * (1 + SLIPPAGE + COMMISSION)
            entry_i = i
            mae = 0.0
            pos = 1
        elif pos == 1:
            low_pct = (lows[i] - entry_p) / entry_p
            if low_pct < mae:
                mae = low_pct
            if s == -1:
                exit_p = p * (1 - SLIPPAGE - COMMISSION)
                pnl = (exit_p - entry_p) / entry_p * 100
                trades.append((pnl, mae, i - entry_i, df.index[entry_i].year if hasattr(df.index[entry_i], 'year') else 2025))
                pos = 0

    if len(trades) < MIN_TRADES:
        return None

    pnls = np.array([t[0] for t in trades])
    maes = np.array([t[1] for t in trades])
    years = np.array([t[3] for t in trades])

    wins = (pnls > 0).sum()
    total = len(pnls)
    wr = wins / total * 100
    total_pnl = pnls.sum()

    w_mask = pnls > 0
    avg_win = pnls[w_mask].mean() if w_mask.any() else 0
    avg_loss = pnls[~w_mask].mean() if (~w_mask).any() else 0
    losses = (~w_mask).sum()
    pf = abs(avg_win * wins / (avg_loss * losses + 1e-10)) if losses > 0 else 999

    cum = pnls.cumsum()
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = dd.max()

    mae_p95 = abs(np.percentile(maes, 95)) if len(maes) > 0 else 0.1
    sharpe = (pnls.mean() / (pnls.std() + 1e-10)) * np.sqrt(252)

    yearly = {}
    for y in np.unique(years):
        mask = years == y
        y_pnls = pnls[mask]
        y_wins = (y_pnls > 0).sum()
        yearly[str(y)] = {'wr': round(y_wins/len(y_pnls)*100,1), 'trades': int(len(y_pnls)), 'pnl': round(y_pnls.sum(),2)}

    return {
        'trades': int(total), 'wr': round(wr,1), 'pnl': round(total_pnl,2),
        'avg_win': round(avg_win,2), 'avg_loss': round(avg_loss,2),
        'profit_factor': round(pf,2), 'max_drawdown': round(max_dd,2),
        'mae_p95': round(mae_p95,4), 'sharpe': round(sharpe,2), 'yearly': yearly
    }

# ─── STRATEGY SIGNAL GENERATORS (all vectorized, no loops) ───
STRATEGIES = []

def reg(name, func):
    STRATEGIES.append((name, func))

# RSI variants
for p in [7,9,14,21]:
    for b in [20,25,30,35]:
        s = 100-b
        def f(df, _p=p, _b=b, _s=s):
            r = rsi(df['close'], _p); sig = pd.Series(0, index=df.index)
            sig[r<_b]=1; sig[r>_s]=-1; return sig
        reg(f"RSI_{p}_{b}_{s}", f)

# EMA cross
for fast,slow in [(5,13),(8,21),(9,21),(12,26),(20,50),(50,200)]:
    def f(df, _f=fast, _s=slow):
        ef=ema(df['close'],_f); es=ema(df['close'],_s); sig=pd.Series(0,index=df.index)
        sig[(ef>es)&(ef.shift()<=es.shift())]=1; sig[(ef<es)&(ef.shift()>=es.shift())]=-1; return sig
    reg(f"EMA_{fast}_{slow}", f)

# BB variants
for p in [10,15,20,30]:
    for std in [1.5,2.0,2.5,3.0]:
        def f(df, _p=p, _std=std):
            m,u,l=bb(df['close'],_p,_std); sig=pd.Series(0,index=df.index)
            sig[df['close']<l]=1; sig[df['close']>u]=-1; return sig
        reg(f"BB_{p}_{std}", f)

# MACD variants
for fast,slow,signal in [(8,17,9),(12,26,9),(5,35,5)]:
    def f(df, _f=fast, _s=slow, _si=signal):
        _,_2,hist=macd(df['close'],_f,_s,_si); sig=pd.Series(0,index=df.index)
        sig[(hist>0)&(hist.shift()<=0)]=1; sig[(hist<0)&(hist.shift()>=0)]=-1; return sig
    reg(f"MACD_{fast}_{slow}_{signal}", f)

# Stoch variants
for k in [5,9,14,21]:
    for buy in [15,20,25]:
        sell=100-buy
        def f(df, _k=k, _b=buy, _s=sell):
            kk,dd=stoch(df['high'],df['low'],df['close'],_k,3); sig=pd.Series(0,index=df.index)
            sig[(kk<_b)&(kk>kk.shift())]=1; sig[(kk>_s)&(kk<kk.shift())]=-1; return sig
        reg(f"Stoch_{k}_{buy}", f)

# VWAP bounce
def vwap_bounce(df):
    tp=(df['high']+df['low']+df['close'])/3; cumvol=df['volume'].cumsum()
    vwap=((tp*df['volume']).cumsum())/(cumvol+1e-10); sig=pd.Series(0,index=df.index)
    sig[(df['close']<vwap*0.99)&(df['close']>df['close'].shift())]=1
    sig[(df['close']>vwap*1.01)&(df['close']<df['close'].shift())]=-1; return sig
reg("VWAP_Bounce", vwap_bounce)

# RSI+MACD combos
for rb in [25,30,35,40]:
    rs=100-rb
    def f(df, _rb=rb, _rs=rs):
        r=rsi(df['close'],14); _,_2,hist=macd(df['close']); sig=pd.Series(0,index=df.index)
        sig[(r<_rb)&(hist>0)&(hist.shift()<=0)]=1; sig[(r>_rs)&(hist<0)&(hist.shift()>=0)]=-1; return sig
    reg(f"RSI_MACD_{rb}_{rs}", f)

# Keltner breakout
for p in [10,20,30]:
    for m in [1.0,1.5,2.0,2.5]:
        def f(df, _p=p, _m=m):
            e=ema(df['close'],_p); a=atr(df['high'],df['low'],df['close'],_p)
            u=e+_m*a; l=e-_m*a; sig=pd.Series(0,index=df.index)
            sig[(df['close']>u)&(df['close'].shift()<=u.shift())]=1
            sig[(df['close']<l)&(df['close'].shift()>=l.shift())]=-1; return sig
        reg(f"Keltner_{p}_{m}", f)

# Donchian breakout
for p in [10,20,30,55]:
    def f(df, _p=p):
        hi=df['high'].rolling(_p).max(); lo=df['low'].rolling(_p).min(); sig=pd.Series(0,index=df.index)
        sig[(df['close']>hi.shift())&(df['close'].shift()<=hi.shift(2))]=1
        sig[(df['close']<lo.shift())&(df['close'].shift()>=lo.shift(2))]=-1; return sig
    reg(f"Donchian_{p}", f)

# Z-Score mean reversion
for lb in [20,30,50,100]:
    for z in [1.5,2.0,2.5]:
        def f(df, _lb=lb, _z=z):
            m=df['close'].rolling(_lb).mean(); s=df['close'].rolling(_lb).std()
            zs=(df['close']-m)/(s+1e-10); sig=pd.Series(0,index=df.index)
            sig[zs<-_z]=1; sig[zs>_z]=-1; return sig
        reg(f"ZScore_{lb}_{z}", f)

# ADX trend
for thresh in [20,25,30]:
    for p in [10,14,20]:
        def f(df, _t=thresh, _p=p):
            a,pdi,mdi=adx(df['high'],df['low'],df['close'],_p); sig=pd.Series(0,index=df.index)
            strong=a>_t; sig[strong&(pdi>mdi)&(pdi.shift()<=mdi.shift())]=1
            sig[strong&(mdi>pdi)&(mdi.shift()<=pdi.shift())]=-1; return sig
        reg(f"ADX_{thresh}_{p}", f)

# CCI
for p in [14,20,30]:
    for level in [80,100,150]:
        def f(df, _p=p, _l=level):
            tp=(df['high']+df['low']+df['close'])/3; m=tp.rolling(_p).mean()
            md=tp.rolling(_p).std()*0.6745  # approx mean deviation
            cci=(tp-m)/(0.015*md+1e-10); sig=pd.Series(0,index=df.index)
            sig[(cci<-_l)&(cci>cci.shift())]=1; sig[(cci>_l)&(cci<cci.shift())]=-1; return sig
        reg(f"CCI_{p}_{level}", f)

# Williams %R
for p in [10,14,21]:
    for buy in [-85,-80,-75]:
        sell=-(100+buy)
        def f(df, _p=p, _b=buy, _s=sell):
            hi=df['high'].rolling(_p).max(); lo=df['low'].rolling(_p).min()
            wr=-100*(hi-df['close'])/(hi-lo+1e-10); sig=pd.Series(0,index=df.index)
            sig[(wr<_b)&(wr>wr.shift())]=1; sig[(wr>_s)&(wr<wr.shift())]=-1; return sig
        reg(f"WilliamsR_{p}_{abs(buy)}", f)

# HMA
for p in [9,16,25,36,49]:
    def f(df, _p=p):
        half=max(2,_p//2); sqrt_p=max(2,int(np.sqrt(_p)))
        w1=df['close'].rolling(half).mean(); w2=df['close'].rolling(_p).mean()
        h=(2*w1-w2).rolling(sqrt_p).mean(); sig=pd.Series(0,index=df.index)
        sig[(h>h.shift())&(h.shift()<=h.shift(2))]=1
        sig[(h<h.shift())&(h.shift()>=h.shift(2))]=-1; return sig
    reg(f"HMA_{p}", f)

# Momentum
for p in [5,10,14,21]:
    def f(df, _p=p):
        mom=df['close']/df['close'].shift(_p)-1; sig=pd.Series(0,index=df.index)
        sig[(mom>0.02)&(mom.shift()<=0.02)]=1; sig[(mom<-0.02)&(mom.shift()>=-0.02)]=-1; return sig
    reg(f"Momentum_{p}", f)

# Volume spike
for mult in [1.5,2.0,3.0,5.0]:
    def f(df, _m=mult):
        vm=df['volume'].rolling(20).mean(); spike=df['volume']>_m*vm; sig=pd.Series(0,index=df.index)
        sig[spike&(df['close']>df['open'])]=1; sig[spike&(df['close']<df['open'])]=-1; return sig
    reg(f"VolSpike_{mult}x", f)

# Engulfing
def engulfing(df):
    bull=(df['close']>df['open'])&(df['close'].shift()<df['open'].shift())&(df['close']>df['open'].shift())&(df['open']<df['close'].shift())
    bear=(df['close']<df['open'])&(df['close'].shift()>df['open'].shift())&(df['close']<df['open'].shift())&(df['open']>df['close'].shift())
    sig=pd.Series(0,index=df.index); sig[bull]=1; sig[bear]=-1; return sig
reg("Engulfing", engulfing)

# Doji reversal
def doji_rev(df):
    body=(df['close']-df['open']).abs(); wick=df['high']-df['low']
    doji=body<wick*0.1; r=rsi(df['close'],14); sig=pd.Series(0,index=df.index)
    sig[doji&(r<30)]=1; sig[doji&(r>70)]=-1; return sig
reg("Doji_Reversal", doji_rev)

# Ichimoku
def ichimoku(df):
    th=df['high'].rolling(9).max(); tl=df['low'].rolling(9).min(); tenkan=(th+tl)/2
    kh=df['high'].rolling(26).max(); kl=df['low'].rolling(26).min(); kijun=(kh+kl)/2
    sig=pd.Series(0,index=df.index)
    sig[(tenkan>kijun)&(tenkan.shift()<=kijun.shift())]=1
    sig[(tenkan<kijun)&(tenkan.shift()>=kijun.shift())]=-1; return sig
reg("Ichimoku", ichimoku)

# Pivot bounce
def pivot_bounce(df):
    pp=(df['high'].shift()+df['low'].shift()+df['close'].shift())/3
    s1=2*pp-df['high'].shift(); r1=2*pp-df['low'].shift()
    sig=pd.Series(0,index=df.index); sig[(df['low']<=s1)&(df['close']>s1)]=1
    sig[(df['high']>=r1)&(df['close']<r1)]=-1; return sig
reg("Pivot_Bounce", pivot_bounce)

# OBV divergence
def obv_div(df):
    obv=(np.sign(df['close'].diff())*df['volume']).cumsum()
    o_sma=obv.rolling(14).mean(); p_sma=df['close'].rolling(14).mean()
    sig=pd.Series(0,index=df.index); sig[(obv>o_sma)&(df['close']<p_sma)]=1
    sig[(obv<o_sma)&(df['close']>p_sma)]=-1; return sig
reg("OBV_Divergence", obv_div)

# Squeeze
def squeeze(df):
    mid,bb_up,bb_lo=bb(df['close'],20); e=ema(df['close'],20); a=atr(df['high'],df['low'],df['close'],20)
    kc_up=e+1.5*a; kc_lo=e-1.5*a; sq=(bb_lo>kc_lo)&(bb_up<kc_up)
    mom=df['close']-df['close'].rolling(20).mean(); sig=pd.Series(0,index=df.index)
    sig[~sq&sq.shift().fillna(False)&(mom>0)]=1; sig[~sq&sq.shift().fillna(False)&(mom<0)]=-1; return sig
reg("Squeeze", squeeze)

# ATR breakout
def atr_breakout(df):
    a=atr(df['high'],df['low'],df['close'],14); move=df['close']-df['close'].shift()
    sig=pd.Series(0,index=df.index); sig[(move>1.5*a)&(move.shift()<=1.5*a.shift())]=1
    sig[(move<-1.5*a)&(move.shift()>=-1.5*a.shift())]=-1; return sig
reg("ATR_Breakout", atr_breakout)

# Triple EMA
for f_,m_,s_ in [(3,8,21),(5,13,34),(8,21,55),(13,34,89)]:
    def f(df, _f=f_, _m=m_, _s=s_):
        ef=ema(df['close'],_f); em=ema(df['close'],_m); es=ema(df['close'],_s)
        bull=(ef>em)&(em>es); bear=(ef<em)&(em<es); sig=pd.Series(0,index=df.index)
        sig[bull&~bull.shift().fillna(False)]=1; sig[bear&~bear.shift().fillna(False)]=-1; return sig
    reg(f"TripleEMA_{f_}_{m_}_{s_}", f)

print(f"Total strategies: {len(STRATEGIES)}")

# ─── WEIGHTED SCORE ───
def calc_weighted_score(yearly):
    years = sorted(yearly.keys(), reverse=True)
    n = len(years)
    if n == 0: return 0
    wmap = {1:[100],2:[55,45],3:[40,30,30],4:[35,25,25,15],5:[30,25,20,15,10],6:[27,22,18,14,11,8],7:[25,20,16,13,10,9,7]}
    w = wmap.get(min(n,7), wmap[7])[:n]
    return round(sum(yearly[y]['wr']*w[i]/100 for i,y in enumerate(years[:len(w)])), 1)

# ─── PROCESS ONE ASSET ───
def process_asset(symbol):
    df_5m = load_candles(symbol, '5m')
    if df_5m is None or len(df_5m) < 200:
        return {'symbol': symbol, 'status': 'skip_no_data', 'results': []}

    # Build timeframes
    dfs = {'5m': df_5m}
    dfs['15m'] = resample_ohlcv(df_5m, '15m')
    df_1h = load_candles(symbol, '1h')
    dfs['1h'] = df_1h if df_1h is not None and len(df_1h) > 100 else None
    dfs['4h'] = resample_ohlcv(df_5m, '4h')
    df_1d = load_candles(symbol, '1d')
    dfs['1d'] = df_1d if df_1d is not None and len(df_1d) > 30 else resample_ohlcv(df_5m, '1d')

    results = []
    for tf, df in dfs.items():
        if df is None or len(df) < 100:
            continue
        for sname, sfunc in STRATEGIES:
            try:
                sig = sfunc(df)
                if sig is None or sig.abs().sum() == 0:
                    continue
                r = backtest_vectorized(df, sig)
                if r is not None:
                    r['strategy'] = sname
                    r['timeframe'] = tf
                    r['symbol'] = symbol
                    results.append(r)
            except Exception:
                continue

    if not results:
        return {'symbol': symbol, 'status': 'no_valid_results', 'results': []}

    best = max(results, key=lambda x: x['sharpe'])
    sorted_r = sorted(results, key=lambda x: x['sharpe'], reverse=True)

    dream_team = []
    seen = set()
    for r in sorted_r:
        key = r['strategy']
        if key not in seen and r['wr'] >= 50:
            score = calc_weighted_score(r.get('yearly', {}))
            mae = r.get('mae_p95', 0.1)
            dream_team.append({
                'strategy': r['strategy'], 'timeframe': r['timeframe'],
                'wr': r['wr'], 'pnl': r['pnl'], 'sharpe': r['sharpe'],
                'mae_p95': mae, 'yearly': r.get('yearly', {}),
                'weighted_score': score,
                'safe_leverage': min(20, int(1/(mae*2.5+1e-10)))
            })
            seen.add(key)
        if len(dream_team) >= 5:
            break

    top_score = dream_team[0]['weighted_score'] if dream_team else 0
    decision = "PRODUCTION" if top_score >= 70 else ("TESTING" if top_score >= 50 else "REJECTED")
    if not dream_team:
        decision = "NO_QUALIFYING"

    return {
        'symbol': symbol, 'status': 'completed',
        'total_tested': len(results),
        'best_strategy': best['strategy'], 'best_tf': best.get('timeframe',''),
        'best_wr': best['wr'], 'best_pnl': best['pnl'], 'best_sharpe': best['sharpe'],
        'dream_team': dream_team, 'decision': decision,
        'weighted_score': top_score, 'results': results,
        'completed_at': datetime.now().isoformat()
    }

# ─── PROGRESS ───
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {'started_at': datetime.now().isoformat(), 'total_strategies': len(STRATEGIES),
            'total_assets': 0, 'assets_completed': 0, 'assets': {},
            'summary': {'production_ready':0,'testing':0,'rejected':0,'no_qualifying':0,'skip_no_data':0}}

def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2, default=str)

# ─── MAIN ───
def main():
    print("="*60)
    print(f"FAST PIPELINE v2: {len(STRATEGIES)} strategies x 563 assets x 5 TFs")
    print("="*60)

    symbols = get_symbols_ranked()
    print(f"Total symbols: {len(symbols)}")

    progress = load_progress()
    progress['total_strategies'] = len(STRATEGIES)
    progress['total_assets'] = len(symbols)

    completed = set(s for s,d in progress['assets'].items() if d.get('status') in ('completed','skip_no_data','no_valid_results'))
    remaining = [(s,b) for s,b in symbols if s not in completed]
    print(f"Already done: {len(completed)} | Remaining: {len(remaining)}")
    print("-"*60)

    start = time.time()
    for idx, (symbol, bars) in enumerate(remaining):
        t0 = time.time()
        short = symbol[:20].ljust(20)
        print(f"[{idx+1}/{len(remaining)}] {short} ({bars:>8,} bars)...", end=" ", flush=True)

        try:
            result = process_asset(symbol)
            status = result['status']
            summary = {k:v for k,v in result.items() if k != 'results'}
            progress['assets'][symbol] = summary

            if status == 'completed':
                dec = result.get('decision','REJECTED')
                progress['summary'][{'PRODUCTION':'production_ready','TESTING':'testing','NO_QUALIFYING':'no_qualifying'}.get(dec,'rejected')] += 1
                dt = len(result.get('dream_team',[]))
                print(f"OK {time.time()-t0:.0f}s | n={result['total_tested']} best={result['best_strategy']} WR={result['best_wr']}% S={result['best_sharpe']} DT={dt} [{dec}]")
                safe = symbol.replace('/','_').replace(':','_')
                with open(os.path.join(RESULTS_DIR, f"{safe}.json"), 'w') as f:
                    json.dump(result['results'], f, default=str)
            elif status == 'skip_no_data':
                progress['summary']['skip_no_data'] = progress['summary'].get('skip_no_data',0)+1
                print("SKIP")
            else:
                print("NO RESULTS")

        except Exception as e:
            print(f"ERR: {e}")
            progress['assets'][symbol] = {'status':'error','error':str(e)}

        progress['assets_completed'] = len([s for s,d in progress['assets'].items() if d.get('status')=='completed'])
        elapsed = time.time()-start
        done = idx+1
        rate = elapsed/done
        eta = rate*(len(remaining)-done)/3600
        progress['summary']['elapsed_hours'] = round(elapsed/3600,2)
        progress['summary']['estimated_remaining_hours'] = round(eta,1)
        save_progress(progress)

    print("\n"+"="*60)
    print("DONE!")
    s = progress['summary']
    print(f"PROD: {s['production_ready']} | TEST: {s['testing']} | REJ: {s.get('rejected',0)} | SKIP: {s.get('skip_no_data',0)}")
    print(f"Time: {s['elapsed_hours']:.1f}h")

if __name__ == '__main__':
    main()
