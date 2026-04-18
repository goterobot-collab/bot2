#!/usr/bin/env python3
"""
OPTUNA FULL v2: TODAS las estrategias × TODOS los assets
Objetivo: encontrar el SANTO GRIAL por cada activo
- Optimiza cada estrategia al máximo para cada asset
- Walk-forward 70/30 (train desde origen, test últimos 30%)
- Solo guarda WR > 70% Y PnL+ en TEST
- Usa multiprocessing para máxima velocidad
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import warnings
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

warnings.filterwarnings('ignore')
multiprocessing.set_start_method('spawn', force=True)

DB_PATH = "/Users/sabrina/Code/activos binace /activos_binance.db"
PROJECT_DIR = "/Users/sabrina/CLAUDE CODE/Estrategias"
PROGRESS_FILE = os.path.join(PROJECT_DIR, "data", "optuna_v2_progress.json")
GRAIL_FILE = os.path.join(PROJECT_DIR, "data", "santo_grial.json")
N_TRIALS = 20
COMMISSION = 0.001
SLIPPAGE = 0.0005
MIN_TRADES = 5
MAX_WORKERS = 8

os.makedirs(os.path.join(PROJECT_DIR, "data"), exist_ok=True)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("Installing optuna..."); os.system("pip3 install optuna -q")
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── DB ───
def load_candles(symbol, tf="5m"):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT ts,open,high,low,close,volume FROM candles WHERE symbol=? AND timeframe=? ORDER BY ts",
                      conn, params=(symbol, tf))
    conn.close()
    if len(df) == 0: return None
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    return df

def resample(df, target):
    m = {'15m':'15min','4h':'4h','1d':'1D'}
    r = m.get(target)
    if not r: return None
    return df.resample(r).agg({'ts':'first','open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['close'])

def get_symbols_ranked():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT symbol, COUNT(*) as bars FROM candles WHERE timeframe='5m' GROUP BY symbol ORDER BY bars DESC", conn)
    conn.close()
    return list(df.itertuples(index=False, name=None))

# ─── INDICATORS ───
def ema(s,p): return s.ewm(span=p,adjust=False).mean()
def rsi(s,p=14):
    d=s.diff(); g=d.where(d>0,0.).ewm(span=p,adjust=False).mean()
    l=(-d.where(d<0,0.)).ewm(span=p,adjust=False).mean()
    return 100-100/(1+g/(l+1e-10))
def atr(h,l,c,p=14):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(p).mean()
def bb(c,p=20,std=2):
    m=c.rolling(p).mean(); s=c.rolling(p).std(); return m,m+std*s,m-std*s
def macd(c,f=12,s=26,sig=9):
    m=ema(c,f)-ema(c,s); si=ema(m,sig); return m,si,m-si
def stoch(h,l,c,kp=14,dp=3):
    lo=l.rolling(kp).min(); hi=h.rolling(kp).max()
    k=100*(c-lo)/(hi-lo+1e-10); return k,k.rolling(dp).mean()
def adx_calc(h,l,c,p=14):
    tr1=atr(h,l,c,1); up=h.diff(); dn=-l.diff()
    pdm=np.where((up>dn)&(up>0),up,0); mdm=np.where((dn>up)&(dn>0),dn,0)
    pdi=100*pd.Series(pdm,index=c.index).rolling(p).mean()/(tr1.rolling(p).mean()+1e-10)
    mdi=100*pd.Series(mdm,index=c.index).rolling(p).mean()/(tr1.rolling(p).mean()+1e-10)
    dx=100*(pdi-mdi).abs()/(pdi+mdi+1e-10); return dx.rolling(p).mean(),pdi,mdi

# ─── BACKTEST ENGINE ───
def backtest(df, signals):
    if signals is None: return None
    sig=signals.values; opens=df['open'].values; highs=df['high'].values
    lows=df['low'].values; closes=df['close'].values; n=len(df)
    trades=[]; pos=0; ep=0.; ei=0; mae=0.
    for i in range(1,n):
        s=sig[i-1]; p=opens[i]
        if pos==0 and s==1:
            ep=p*(1+SLIPPAGE+COMMISSION); ei=i; mae=0.; pos=1
        elif pos==1:
            lp=(lows[i]-ep)/ep
            if lp<mae: mae=lp
            if s==-1:
                xp=p*(1-SLIPPAGE-COMMISSION)
                pnl=(xp-ep)/ep*100
                yr=df.index[ei].year if hasattr(df.index[ei],'year') else 2025
                trades.append((pnl,mae,i-ei,yr))
                pos=0
    if len(trades)<MIN_TRADES: return None
    pnls=np.array([t[0] for t in trades]); maes=np.array([t[1] for t in trades])
    years=np.array([t[3] for t in trades])
    wins=(pnls>0).sum(); total=len(pnls); wr=wins/total*100
    total_pnl=pnls.sum()
    wm=pnls>0; avg_w=pnls[wm].mean() if wm.any() else 0
    avg_l=pnls[~wm].mean() if (~wm).any() else 0
    losses=(~wm).sum()
    pf=abs(avg_w*wins/(avg_l*losses+1e-10)) if losses>0 else 999
    cum=pnls.cumsum(); peak=np.maximum.accumulate(cum); dd=(peak-cum).max()
    mae95=abs(np.percentile(maes,95)) if len(maes)>0 else 0.1
    sharpe=(pnls.mean()/(pnls.std()+1e-10))*np.sqrt(252)
    yearly={}
    for y in np.unique(years):
        m=years==y; yp=pnls[m]; yw=(yp>0).sum()
        yearly[str(y)]={'wr':round(yw/len(yp)*100,1),'trades':int(len(yp)),'pnl':round(yp.sum(),2)}
    return {'trades':int(total),'wr':round(wr,1),'pnl':round(total_pnl,2),
            'profit_factor':round(pf,2),'max_drawdown':round(dd,2),
            'mae_p95':round(mae95,4),'sharpe':round(sharpe,2),'yearly':yearly,
            'avg_win':round(avg_w,2),'avg_loss':round(avg_l,2)}

# ─── ALL STRATEGY TYPES WITH OPTUNA SEARCH SPACES ───
STRATEGY_TYPES = {}

def reg(name, gen_func, space_func):
    STRATEGY_TYPES[name] = {'gen': gen_func, 'space': space_func}

# RSI
def gen_rsi(df, period, buy, sell):
    r=rsi(df['close'],period); sig=pd.Series(0,index=df.index); sig[r<buy]=1; sig[r>sell]=-1; return sig
reg('RSI', gen_rsi, lambda t: {'period':t.suggest_int('period',5,30),'buy':t.suggest_int('buy',15,40),'sell':t.suggest_int('sell',60,85)})

# Bollinger Bands
def gen_bb(df, period, std_mult):
    m,u,l=bb(df['close'],period,std_mult); sig=pd.Series(0,index=df.index); sig[df['close']<l]=1; sig[df['close']>u]=-1; return sig
reg('BB', gen_bb, lambda t: {'period':t.suggest_int('period',8,50),'std_mult':t.suggest_float('std_mult',1.0,4.0,step=0.25)})

# EMA Cross
def gen_ema(df, fast, slow):
    ef=ema(df['close'],fast); es=ema(df['close'],slow); sig=pd.Series(0,index=df.index)
    sig[(ef>es)&(ef.shift()<=es.shift())]=1; sig[(ef<es)&(ef.shift()>=es.shift())]=-1; return sig
reg('EMA', gen_ema, lambda t: {'fast':t.suggest_int('fast',3,30),'slow':t.suggest_int('slow',15,250)})

# MACD
def gen_macd(df, fast, slow, signal):
    _,_2,hist=macd(df['close'],fast,slow,signal); sig=pd.Series(0,index=df.index)
    sig[(hist>0)&(hist.shift()<=0)]=1; sig[(hist<0)&(hist.shift()>=0)]=-1; return sig
reg('MACD', gen_macd, lambda t: {'fast':t.suggest_int('fast',4,18),'slow':t.suggest_int('slow',18,45),'signal':t.suggest_int('signal',4,18)})

# Stochastic
def gen_stoch(df, k_period, buy, sell):
    k,d=stoch(df['high'],df['low'],df['close'],k_period,3); sig=pd.Series(0,index=df.index)
    sig[(k<buy)&(k>k.shift())]=1; sig[(k>sell)&(k<k.shift())]=-1; return sig
reg('Stoch', gen_stoch, lambda t: {'k_period':t.suggest_int('k_period',3,28),'buy':t.suggest_int('buy',8,35),'sell':t.suggest_int('sell',65,92)})

# Keltner Channel
def gen_keltner(df, period, mult):
    e=ema(df['close'],period); a=atr(df['high'],df['low'],df['close'],period)
    u=e+mult*a; l=e-mult*a; sig=pd.Series(0,index=df.index)
    sig[(df['close']>u)&(df['close'].shift()<=u.shift())]=1
    sig[(df['close']<l)&(df['close'].shift()>=l.shift())]=-1; return sig
reg('Keltner', gen_keltner, lambda t: {'period':t.suggest_int('period',6,45),'mult':t.suggest_float('mult',0.5,3.5,step=0.25)})

# Z-Score Mean Reversion
def gen_zscore(df, lookback, z_thresh):
    m=df['close'].rolling(lookback).mean(); s=df['close'].rolling(lookback).std()
    z=(df['close']-m)/(s+1e-10); sig=pd.Series(0,index=df.index)
    sig[z<-z_thresh]=1; sig[z>z_thresh]=-1; return sig
reg('ZScore', gen_zscore, lambda t: {'lookback':t.suggest_int('lookback',8,150),'z_thresh':t.suggest_float('z_thresh',0.8,3.5,step=0.1)})

# CCI
def gen_cci(df, period, level):
    tp=(df['high']+df['low']+df['close'])/3; m=tp.rolling(period).mean()
    md=tp.rolling(period).std()*0.6745; cci=(tp-m)/(0.015*md+1e-10); sig=pd.Series(0,index=df.index)
    sig[(cci<-level)&(cci>cci.shift())]=1; sig[(cci>level)&(cci<cci.shift())]=-1; return sig
reg('CCI', gen_cci, lambda t: {'period':t.suggest_int('period',8,45),'level':t.suggest_int('level',50,250)})

# Williams %R
def gen_williams(df, period, buy_level):
    hi=df['high'].rolling(period).max(); lo=df['low'].rolling(period).min()
    wr=-100*(hi-df['close'])/(hi-lo+1e-10); sell=-(100+buy_level); sig=pd.Series(0,index=df.index)
    sig[(wr<buy_level)&(wr>wr.shift())]=1; sig[(wr>sell)&(wr<wr.shift())]=-1; return sig
reg('WilliamsR', gen_williams, lambda t: {'period':t.suggest_int('period',5,30),'buy_level':t.suggest_int('buy_level',-95,-65)})

# Donchian Breakout
def gen_donchian(df, period):
    hi=df['high'].rolling(period).max(); lo=df['low'].rolling(period).min(); sig=pd.Series(0,index=df.index)
    sig[(df['close']>hi.shift())&(df['close'].shift()<=hi.shift(2))]=1
    sig[(df['close']<lo.shift())&(df['close'].shift()>=lo.shift(2))]=-1; return sig
reg('Donchian', gen_donchian, lambda t: {'period':t.suggest_int('period',5,80)})

# ADX Trend
def gen_adx(df, period, thresh):
    a,pdi,mdi=adx_calc(df['high'],df['low'],df['close'],period); sig=pd.Series(0,index=df.index)
    strong=a>thresh; sig[strong&(pdi>mdi)&(pdi.shift()<=mdi.shift())]=1
    sig[strong&(mdi>pdi)&(mdi.shift()<=pdi.shift())]=-1; return sig
reg('ADX', gen_adx, lambda t: {'period':t.suggest_int('period',6,28),'thresh':t.suggest_int('thresh',12,40)})

# Momentum
def gen_momentum(df, period, threshold):
    mom=df['close']/df['close'].shift(period)-1; sig=pd.Series(0,index=df.index)
    sig[(mom>threshold)&(mom.shift()<=threshold)]=1
    sig[(mom<-threshold)&(mom.shift()>=-threshold)]=-1; return sig
reg('Momentum', gen_momentum, lambda t: {'period':t.suggest_int('period',3,35),'threshold':t.suggest_float('threshold',0.003,0.08,step=0.002)})

# HMA
def gen_hma(df, period):
    half=max(2,period//2); sqp=max(2,int(np.sqrt(period)))
    w1=df['close'].rolling(half).mean(); w2=df['close'].rolling(period).mean()
    h=(2*w1-w2).rolling(sqp).mean(); sig=pd.Series(0,index=df.index)
    sig[(h>h.shift())&(h.shift()<=h.shift(2))]=1
    sig[(h<h.shift())&(h.shift()>=h.shift(2))]=-1; return sig
reg('HMA', gen_hma, lambda t: {'period':t.suggest_int('period',4,70)})

# RSI + MACD Combo
def gen_rsi_macd(df, rsi_buy, rsi_sell):
    r=rsi(df['close'],14); _,_2,hist=macd(df['close']); sig=pd.Series(0,index=df.index)
    sig[(r<rsi_buy)&(hist>0)&(hist.shift()<=0)]=1
    sig[(r>rsi_sell)&(hist<0)&(hist.shift()>=0)]=-1; return sig
reg('RSI_MACD', gen_rsi_macd, lambda t: {'rsi_buy':t.suggest_int('rsi_buy',18,48),'rsi_sell':t.suggest_int('rsi_sell',52,82)})

# VWAP Bounce
def gen_vwap(df, dist_pct):
    tp=(df['high']+df['low']+df['close'])/3; cumvol=df['volume'].cumsum()
    vwap=((tp*df['volume']).cumsum())/(cumvol+1e-10); sig=pd.Series(0,index=df.index)
    sig[(df['close']<vwap*(1-dist_pct))&(df['close']>df['close'].shift())]=1
    sig[(df['close']>vwap*(1+dist_pct))&(df['close']<df['close'].shift())]=-1; return sig
reg('VWAP', gen_vwap, lambda t: {'dist_pct':t.suggest_float('dist_pct',0.003,0.05,step=0.002)})

# Volume Spike
def gen_volspike(df, vol_mult):
    vm=df['volume'].rolling(20).mean(); spike=df['volume']>vol_mult*vm; sig=pd.Series(0,index=df.index)
    sig[spike&(df['close']>df['open'])]=1; sig[spike&(df['close']<df['open'])]=-1; return sig
reg('VolSpike', gen_volspike, lambda t: {'vol_mult':t.suggest_float('vol_mult',1.2,6.0,step=0.2)})

# Engulfing + RSI filter
def gen_engulfing(df, rsi_filter):
    bull=(df['close']>df['open'])&(df['close'].shift()<df['open'].shift())&(df['close']>df['open'].shift())&(df['open']<df['close'].shift())
    bear=(df['close']<df['open'])&(df['close'].shift()>df['open'].shift())&(df['close']<df['open'].shift())&(df['open']>df['close'].shift())
    r=rsi(df['close'],14); sig=pd.Series(0,index=df.index)
    sig[bull&(r<rsi_filter)]=1; sig[bear&(r>(100-rsi_filter))]=-1; return sig
reg('Engulfing', gen_engulfing, lambda t: {'rsi_filter':t.suggest_int('rsi_filter',25,55)})

# Ichimoku
def gen_ichimoku(df, tenkan_p, kijun_p):
    th=df['high'].rolling(tenkan_p).max(); tl=df['low'].rolling(tenkan_p).min(); tenkan=(th+tl)/2
    kh=df['high'].rolling(kijun_p).max(); kl=df['low'].rolling(kijun_p).min(); kijun=(kh+kl)/2
    sig=pd.Series(0,index=df.index)
    sig[(tenkan>kijun)&(tenkan.shift()<=kijun.shift())]=1
    sig[(tenkan<kijun)&(tenkan.shift()>=kijun.shift())]=-1; return sig
reg('Ichimoku', gen_ichimoku, lambda t: {'tenkan_p':t.suggest_int('tenkan_p',5,20),'kijun_p':t.suggest_int('kijun_p',15,52)})

# Squeeze Momentum
def gen_squeeze(df, bb_period, kc_mult):
    mid,bb_up,bb_lo=bb(df['close'],bb_period); e=ema(df['close'],bb_period)
    a=atr(df['high'],df['low'],df['close'],bb_period)
    kc_up=e+kc_mult*a; kc_lo=e-kc_mult*a; sq=(bb_lo>kc_lo)&(bb_up<kc_up)
    mom=df['close']-df['close'].rolling(bb_period).mean(); sig=pd.Series(0,index=df.index)
    sig[~sq&sq.shift().fillna(False)&(mom>0)]=1; sig[~sq&sq.shift().fillna(False)&(mom<0)]=-1; return sig
reg('Squeeze', gen_squeeze, lambda t: {'bb_period':t.suggest_int('bb_period',10,35),'kc_mult':t.suggest_float('kc_mult',1.0,2.5,step=0.25)})

# Triple EMA
def gen_triple_ema(df, fast, mid, slow):
    ef=ema(df['close'],fast); em=ema(df['close'],mid); es=ema(df['close'],slow)
    bull=(ef>em)&(em>es); bear=(ef<em)&(em<es); sig=pd.Series(0,index=df.index)
    sig[bull&~bull.shift().fillna(False)]=1; sig[bear&~bear.shift().fillna(False)]=-1; return sig
reg('TripleEMA', gen_triple_ema, lambda t: {'fast':t.suggest_int('fast',2,12),'mid':t.suggest_int('mid',8,30),'slow':t.suggest_int('slow',20,100)})

# Pivot Bounce
def gen_pivot(df, atr_filter):
    pp=(df['high'].shift()+df['low'].shift()+df['close'].shift())/3
    s1=2*pp-df['high'].shift(); r1=2*pp-df['low'].shift()
    a=atr(df['high'],df['low'],df['close'],14); sig=pd.Series(0,index=df.index)
    sig[(df['low']<=s1)&(df['close']>s1)&(a>a.rolling(20).mean()*atr_filter)]=1
    sig[(df['high']>=r1)&(df['close']<r1)]=1; return sig
reg('Pivot', gen_pivot, lambda t: {'atr_filter':t.suggest_float('atr_filter',0.5,2.0,step=0.1)})

# OBV Divergence
def gen_obv(df, period):
    obv=(np.sign(df['close'].diff())*df['volume']).cumsum()
    o_sma=obv.rolling(period).mean(); p_sma=df['close'].rolling(period).mean()
    sig=pd.Series(0,index=df.index); sig[(obv>o_sma)&(df['close']<p_sma)]=1
    sig[(obv<o_sma)&(df['close']>p_sma)]=-1; return sig
reg('OBV', gen_obv, lambda t: {'period':t.suggest_int('period',8,40)})

print(f"Total strategy types: {len(STRATEGY_TYPES)}")
STRAT_NAMES = list(STRATEGY_TYPES.keys())

# ─── WEIGHTED SCORE ───
def calc_score(yearly):
    years=sorted(yearly.keys(),reverse=True); n=len(years)
    if n==0: return 0
    wm={1:[100],2:[55,45],3:[40,30,30],4:[35,25,25,15],5:[30,25,20,15,10],6:[27,22,18,14,11,8],7:[25,20,16,13,10,9,7]}
    w=wm.get(min(n,7),wm[7])[:n]
    return round(sum(yearly[y]['wr']*w[i]/100 for i,y in enumerate(years[:len(w)])),1)

# ─── OPTIMIZE ONE STRATEGY FOR ONE ASSET+TF ───
def optimize_one(df_train, df_test, df_full, stype):
    """Returns grail dict or None"""
    info = STRATEGY_TYPES[stype]
    gen = info['gen']
    space = info['space']

    def objective(trial):
        params = space(trial)
        try:
            sig = gen(df_train, **params)
            r = backtest(df_train, sig)
            if r is None: return -999
            # Maximize composite: WR most important + some Sharpe
            return r['wr'] * 0.7 + min(r['sharpe'], 50) * 0.3
        except:
            return -999

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, timeout=60)

    if study.best_value <= 0:
        return None

    best_params = study.best_params

    # Validate on TEST
    try:
        test_sig = gen(df_test, **best_params)
        test_r = backtest(df_test, test_sig)
        if test_r is None: return None

        train_sig = gen(df_train, **best_params)
        train_r = backtest(df_train, train_sig)
        if train_r is None: return None

        full_sig = gen(df_full, **best_params)
        full_r = backtest(df_full, full_sig)
    except:
        return None

    # SANTO GRIAL CHECK: WR > 70% AND PnL > 0 on TEST
    if test_r['wr'] < 70 or test_r['pnl'] <= 0:
        return None

    # Anti-overfit: gap > 30% = suspicious
    wr_diff = abs(train_r['wr'] - test_r['wr'])
    if wr_diff > 30:
        return None

    mae = test_r.get('mae_p95', 0.1)
    safe_lev = min(20, max(1, int(1 / (mae * 2.5 + 1e-10))))

    return {
        'strategy': stype,
        'best_params': best_params,
        'train': {'wr': train_r['wr'], 'pnl': train_r['pnl'], 'sharpe': train_r['sharpe'],
                  'trades': train_r['trades'], 'pf': train_r['profit_factor']},
        'test': {'wr': test_r['wr'], 'pnl': test_r['pnl'], 'sharpe': test_r['sharpe'],
                 'trades': test_r['trades'], 'pf': test_r['profit_factor'],
                 'yearly': test_r.get('yearly', {}), 'mae_p95': mae,
                 'max_dd': test_r['max_drawdown']},
        'full': {'wr': full_r['wr'], 'pnl': full_r['pnl'], 'sharpe': full_r['sharpe'],
                 'trades': full_r['trades']} if full_r else None,
        'wr_diff': round(wr_diff, 1),
        'weighted_score': calc_score(test_r.get('yearly', {})),
        'safe_leverage': safe_lev,
    }

# ─── PROCESS ONE ASSET: ALL STRATEGIES × ALL TIMEFRAMES ───
def process_asset(symbol):
    """Find the SANTO GRIAL for this asset"""
    df_5m = load_candles(symbol, '5m')
    if df_5m is None or len(df_5m) < 500:
        return symbol, 'skip', [], 0

    # Build timeframes
    dfs = {'5m': df_5m}
    dfs['15m'] = resample(df_5m, '15m')
    df_1h = load_candles(symbol, '1h')
    dfs['1h'] = df_1h if df_1h is not None and len(df_1h) > 100 else None
    dfs['4h'] = resample(df_5m, '4h')
    df_1d = load_candles(symbol, '1d')
    dfs['1d'] = df_1d if df_1d is not None and len(df_1d) > 30 else resample(df_5m, '1d')

    grails = []
    tested = 0

    for tf, df in dfs.items():
        if df is None or len(df) < 200:
            continue

        # Walk-forward split: 70% train / 30% test
        split = int(len(df) * 0.7)
        train = df.iloc[:split].copy()
        test = df.iloc[split:].copy()

        if len(train) < 100 or len(test) < 50:
            continue

        for stype in STRAT_NAMES:
            tested += 1
            try:
                result = optimize_one(train, test, df, stype)
                if result is not None:
                    result['timeframe'] = tf
                    result['symbol'] = symbol
                    grails.append(result)
            except:
                continue

    # Sort by test WR × Sharpe
    grails.sort(key=lambda x: x['test']['wr'] * 0.6 + min(x['test']['sharpe'], 50) * 0.4, reverse=True)

    return symbol, 'completed', grails, tested

# ─── MAIN (PARALELO) ───
def main():
    print("=" * 70)
    print(f"SANTO GRIAL OPTIMIZER v2 PARALELO: {len(STRATEGY_TYPES)} types × 563 assets × 5 TFs")
    print(f"Optuna {N_TRIALS} trials | Walk-forward 70/30 | WR>70% + PnL+ only")
    print(f"Workers: {MAX_WORKERS} | CPUs: {multiprocessing.cpu_count()}")
    print("=" * 70)

    symbols = get_symbols_ranked()
    print(f"Total symbols: {len(symbols)}")

    # Load progress
    progress = {'completed': {}, 'grails': [], 'stats': {'tested': 0, 'grails_found': 0, 'assets_with_grail': 0}}
    if os.path.exists(PROGRESS_FILE):
        progress = json.load(open(PROGRESS_FILE))
    completed = set(progress.get('completed', {}).keys())

    remaining = [(s, b) for s, b in symbols if s not in completed]
    print(f"Already done: {len(completed)} | Remaining: {len(remaining)}")
    print(f"Expected: {len(remaining) * len(STRATEGY_TYPES) * 5} optimizations")
    print("-" * 70)

    start = time.time()
    total_grails = len(progress.get('grails', []))
    total_tested = progress.get('stats', {}).get('tested', 0)
    assets_with = progress.get('stats', {}).get('assets_with_grail', 0)
    processed = 0

    # ── PARALLEL EXECUTION ──
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all remaining assets
        futures = {}
        for symbol, bars in remaining:
            fut = executor.submit(process_asset, symbol)
            futures[fut] = symbol

        for fut in as_completed(futures):
            symbol = futures[fut]
            processed += 1
            short = symbol[:25].ljust(25)

            try:
                sym, status, grails, tested = fut.result(timeout=600)
            except Exception as e:
                progress['completed'][symbol] = {'status': 'error', 'error': str(e)[:100]}
                print(f"[{processed}/{len(remaining)}] {short} ⚠️ ERROR: {str(e)[:80]}")
                continue

            total_tested += tested

            if status == 'skip':
                progress['completed'][symbol] = {'status': 'skip'}
            elif grails:
                total_grails += len(grails)
                assets_with += 1
                progress['completed'][symbol] = {
                    'status': 'grail_found',
                    'count': len(grails),
                    'best_wr': grails[0]['test']['wr'],
                    'best_strategy': grails[0]['strategy'],
                    'best_tf': grails[0]['timeframe'],
                }
                progress['grails'].extend(grails)

                best = grails[0]
                elapsed_m = (time.time() - start) / 60
                rate = processed / elapsed_m if elapsed_m > 0 else 1
                eta_m = (len(remaining) - processed) / rate if rate > 0 else 0
                print(f"[{processed}/{len(remaining)}] {short} 🏆 {len(grails)} grails | "
                      f"{best['strategy']}@{best['timeframe']} "
                      f"WR={best['test']['wr']:.0f}% PnL={best['test']['pnl']:.0f}% "
                      f"Lev={best['safe_leverage']}x | "
                      f"Total={total_grails} grails | ETA={eta_m:.0f}min")
            else:
                progress['completed'][symbol] = {'status': 'no_grail', 'tested': tested}
                if processed % 20 == 0:
                    elapsed_m = (time.time() - start) / 60
                    rate = processed / elapsed_m if elapsed_m > 0 else 1
                    eta_m = (len(remaining) - processed) / rate if rate > 0 else 0
                    print(f"[{processed}/{len(remaining)}] Progress: {total_grails} grails | "
                          f"{rate:.1f} assets/min | ETA={eta_m:.0f}min")

            # Save progress every 5 assets
            if processed % 5 == 0:
                progress['stats'] = {
                    'tested': total_tested,
                    'grails_found': total_grails,
                    'assets_with_grail': assets_with,
                    'assets_done': len(progress['completed']),
                    'assets_total': len(symbols),
                    'elapsed_hours': round((time.time() - start) / 3600, 2),
                }
                with open(PROGRESS_FILE, 'w') as f:
                    json.dump(progress, f, default=str)
                with open(GRAIL_FILE, 'w') as f:
                    json.dump(progress['grails'], f, indent=1, default=str)

    # Final save
    progress['stats'] = {
        'tested': total_tested,
        'grails_found': total_grails,
        'assets_with_grail': assets_with,
        'assets_done': len(progress['completed']),
        'assets_total': len(symbols),
        'elapsed_hours': round((time.time() - start) / 3600, 2),
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, default=str)
    with open(GRAIL_FILE, 'w') as f:
        json.dump(progress['grails'], f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("SANTO GRIAL SEARCH COMPLETE")
    print(f"Total tested: {total_tested:,}")
    print(f"Total grails found: {total_grails}")
    print(f"Assets with at least 1 grail: {assets_with}")
    print(f"Elapsed: {progress['stats']['elapsed_hours']:.1f}h")

    # Top 20
    grails_sorted = sorted(progress['grails'], key=lambda x: x['test']['wr'], reverse=True)
    print(f"\nTOP 20 SANTO GRIALES:")
    for g in grails_sorted[:20]:
        print(f"  {g['symbol'][:22]:22s} {g['strategy']:12s} @{g['timeframe']:3s} "
              f"Test={g['test']['wr']:.1f}% PnL={g['test']['pnl']:.1f}% "
              f"S={g['test']['sharpe']:.1f} Lev={g['safe_leverage']}x "
              f"params={json.dumps(g['best_params'])}")

if __name__ == '__main__':
    main()
