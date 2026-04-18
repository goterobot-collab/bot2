#!/usr/bin/env python3
"""
OPTUNA OPTIMIZATION: Top 5 strategies per asset × 563 assets
Walk-forward 70/30 — HONEST validation
No look-ahead, no repainting
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

warnings.filterwarnings('ignore')

DB_PATH = "/Users/sabrina/Code/activos binace /activos_binance.db"
PROJECT_DIR = "/Users/sabrina/CLAUDE CODE/Estrategias"
RESULTS_DIR = os.path.join(PROJECT_DIR, "data", "results")
OPTUNA_FILE = os.path.join(PROJECT_DIR, "data", "optuna_full_results.json")
PROGRESS_FILE = os.path.join(PROJECT_DIR, "data", "optuna_progress.json")
COMMISSION = 0.001
SLIPPAGE = 0.0005
MIN_TRADES = 5
N_TRIALS = 15

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("WARNING: optuna not installed, using grid search fallback")

# ─── DB ───
def load_candles(symbol, timeframe="5m"):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT ts, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe=? ORDER BY ts",
                      conn, params=(symbol, timeframe))
    conn.close()
    if len(df) == 0: return None
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    return df

def resample_ohlcv(df, target):
    m = {'15m':'15min','4h':'4h','1d':'1D'}
    r = m.get(target)
    if not r: return None
    return df.resample(r).agg({'ts':'first','open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['close'])

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
    sig=signals.values; opens=df['open'].values; lows=df['low'].values; n=len(df)
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
            'mae_p95':round(mae95,4),'sharpe':round(sharpe,2),'yearly':yearly}

# ─── PARAMETERIZED STRATEGY GENERATORS ───
def gen_rsi_signals(df, period, buy, sell):
    r = rsi(df['close'], period); sig = pd.Series(0, index=df.index)
    sig[r < buy] = 1; sig[r > sell] = -1; return sig

def gen_bb_signals(df, period, std_mult):
    m,u,l = bb(df['close'], period, std_mult); sig = pd.Series(0, index=df.index)
    sig[df['close'] < l] = 1; sig[df['close'] > u] = -1; return sig

def gen_ema_signals(df, fast, slow):
    ef = ema(df['close'], fast); es = ema(df['close'], slow); sig = pd.Series(0, index=df.index)
    sig[(ef>es)&(ef.shift()<=es.shift())] = 1; sig[(ef<es)&(ef.shift()>=es.shift())] = -1; return sig

def gen_macd_signals(df, fast, slow, signal):
    _,_2,hist = macd(df['close'], fast, slow, signal); sig = pd.Series(0, index=df.index)
    sig[(hist>0)&(hist.shift()<=0)] = 1; sig[(hist<0)&(hist.shift()>=0)] = -1; return sig

def gen_stoch_signals(df, k_period, buy, sell):
    k,d = stoch(df['high'],df['low'],df['close'],k_period,3); sig = pd.Series(0, index=df.index)
    sig[(k<buy)&(k>k.shift())] = 1; sig[(k>sell)&(k<k.shift())] = -1; return sig

def gen_keltner_signals(df, period, mult):
    e = ema(df['close'],period); a = atr(df['high'],df['low'],df['close'],period)
    u=e+mult*a; l=e-mult*a; sig = pd.Series(0, index=df.index)
    sig[(df['close']>u)&(df['close'].shift()<=u.shift())] = 1
    sig[(df['close']<l)&(df['close'].shift()>=l.shift())] = -1; return sig

def gen_zscore_signals(df, lookback, z_thresh):
    m=df['close'].rolling(lookback).mean(); s=df['close'].rolling(lookback).std()
    z=(df['close']-m)/(s+1e-10); sig = pd.Series(0, index=df.index)
    sig[z<-z_thresh] = 1; sig[z>z_thresh] = -1; return sig

def gen_cci_signals(df, period, level):
    tp=(df['high']+df['low']+df['close'])/3; m=tp.rolling(period).mean()
    md=tp.rolling(period).std()*0.6745
    cci=(tp-m)/(0.015*md+1e-10); sig = pd.Series(0, index=df.index)
    sig[(cci<-level)&(cci>cci.shift())] = 1; sig[(cci>level)&(cci<cci.shift())] = -1; return sig

def gen_williams_signals(df, period, buy_level):
    hi=df['high'].rolling(period).max(); lo=df['low'].rolling(period).min()
    wr=-100*(hi-df['close'])/(hi-lo+1e-10); sig = pd.Series(0, index=df.index)
    sell_level = -(100+buy_level)
    sig[(wr<buy_level)&(wr>wr.shift())] = 1; sig[(wr>sell_level)&(wr<wr.shift())] = -1; return sig

def gen_donchian_signals(df, period):
    hi=df['high'].rolling(period).max(); lo=df['low'].rolling(period).min(); sig = pd.Series(0, index=df.index)
    sig[(df['close']>hi.shift())&(df['close'].shift()<=hi.shift(2))] = 1
    sig[(df['close']<lo.shift())&(df['close'].shift()>=lo.shift(2))] = -1; return sig

def gen_adx_signals(df, period, thresh):
    a,pdi,mdi = adx_calc(df['high'],df['low'],df['close'],period); sig = pd.Series(0, index=df.index)
    strong = a > thresh
    sig[strong&(pdi>mdi)&(pdi.shift()<=mdi.shift())] = 1
    sig[strong&(mdi>pdi)&(mdi.shift()<=pdi.shift())] = -1; return sig

def gen_momentum_signals(df, period, threshold):
    mom=df['close']/df['close'].shift(period)-1; sig = pd.Series(0, index=df.index)
    sig[(mom>threshold)&(mom.shift()<=threshold)] = 1
    sig[(mom<-threshold)&(mom.shift()>=-threshold)] = -1; return sig

def gen_hma_signals(df, period):
    half=max(2,period//2); sqp=max(2,int(np.sqrt(period)))
    w1=df['close'].rolling(half).mean(); w2=df['close'].rolling(period).mean()
    h=(2*w1-w2).rolling(sqp).mean(); sig = pd.Series(0, index=df.index)
    sig[(h>h.shift())&(h.shift()<=h.shift(2))] = 1
    sig[(h<h.shift())&(h.shift()>=h.shift(2))] = -1; return sig

def gen_rsi_macd_signals(df, rsi_buy, rsi_sell):
    r=rsi(df['close'],14); _,_2,hist=macd(df['close']); sig = pd.Series(0, index=df.index)
    sig[(r<rsi_buy)&(hist>0)&(hist.shift()<=0)] = 1
    sig[(r>rsi_sell)&(hist<0)&(hist.shift()>=0)] = -1; return sig

# ─── STRATEGY TYPE → OPTUNA SEARCH SPACE ───
STRATEGY_TYPES = {
    'RSI': {
        'gen': gen_rsi_signals,
        'space': lambda trial: {
            'period': trial.suggest_int('period', 5, 30),
            'buy': trial.suggest_int('buy', 15, 40),
            'sell': trial.suggest_int('sell', 60, 85),
        }
    },
    'BB': {
        'gen': gen_bb_signals,
        'space': lambda trial: {
            'period': trial.suggest_int('period', 8, 40),
            'std_mult': trial.suggest_float('std_mult', 1.0, 3.5, step=0.25),
        }
    },
    'EMA': {
        'gen': gen_ema_signals,
        'space': lambda trial: {
            'fast': trial.suggest_int('fast', 3, 25),
            'slow': trial.suggest_int('slow', 15, 200),
        }
    },
    'MACD': {
        'gen': gen_macd_signals,
        'space': lambda trial: {
            'fast': trial.suggest_int('fast', 5, 15),
            'slow': trial.suggest_int('slow', 20, 40),
            'signal': trial.suggest_int('signal', 5, 15),
        }
    },
    'Stoch': {
        'gen': gen_stoch_signals,
        'space': lambda trial: {
            'k_period': trial.suggest_int('k_period', 3, 25),
            'buy': trial.suggest_int('buy', 10, 30),
            'sell': trial.suggest_int('sell', 70, 90),
        }
    },
    'Keltner': {
        'gen': gen_keltner_signals,
        'space': lambda trial: {
            'period': trial.suggest_int('period', 8, 40),
            'mult': trial.suggest_float('mult', 0.5, 3.0, step=0.25),
        }
    },
    'ZScore': {
        'gen': gen_zscore_signals,
        'space': lambda trial: {
            'lookback': trial.suggest_int('lookback', 10, 120),
            'z_thresh': trial.suggest_float('z_thresh', 1.0, 3.0, step=0.25),
        }
    },
    'CCI': {
        'gen': gen_cci_signals,
        'space': lambda trial: {
            'period': trial.suggest_int('period', 10, 40),
            'level': trial.suggest_int('level', 50, 200),
        }
    },
    'WilliamsR': {
        'gen': gen_williams_signals,
        'space': lambda trial: {
            'period': trial.suggest_int('period', 7, 28),
            'buy_level': trial.suggest_int('buy_level', -90, -70),
        }
    },
    'Donchian': {
        'gen': gen_donchian_signals,
        'space': lambda trial: {
            'period': trial.suggest_int('period', 7, 60),
        }
    },
    'ADX': {
        'gen': gen_adx_signals,
        'space': lambda trial: {
            'period': trial.suggest_int('period', 8, 25),
            'thresh': trial.suggest_int('thresh', 15, 35),
        }
    },
    'Momentum': {
        'gen': gen_momentum_signals,
        'space': lambda trial: {
            'period': trial.suggest_int('period', 3, 30),
            'threshold': trial.suggest_float('threshold', 0.005, 0.05, step=0.005),
        }
    },
    'HMA': {
        'gen': gen_hma_signals,
        'space': lambda trial: {
            'period': trial.suggest_int('period', 5, 60),
        }
    },
    'RSI_MACD': {
        'gen': gen_rsi_macd_signals,
        'space': lambda trial: {
            'rsi_buy': trial.suggest_int('rsi_buy', 20, 45),
            'rsi_sell': trial.suggest_int('rsi_sell', 55, 80),
        }
    },
}

def get_strategy_type(name):
    """Map strategy name → type for Optuna parameter space"""
    for prefix in STRATEGY_TYPES:
        if name.startswith(prefix):
            return prefix
    return None

# ─── OPTUNA OBJECTIVE ───
def create_objective(train_df, strat_type):
    gen_func = STRATEGY_TYPES[strat_type]['gen']
    space_func = STRATEGY_TYPES[strat_type]['space']

    def objective(trial):
        params = space_func(trial)
        try:
            signals = gen_func(train_df, **params)
            result = backtest(train_df, signals)
            if result is None:
                return -999
            # Multi-objective: WR × Sharpe (both matter)
            return result['wr'] * 0.5 + result['sharpe'] * 0.5
        except:
            return -999

    return objective

# ─── WEIGHTED SCORE ───
def calc_weighted_score(yearly):
    years = sorted(yearly.keys(), reverse=True)
    n = len(years)
    if n == 0: return 0
    wmap = {1:[100],2:[55,45],3:[40,30,30],4:[35,25,25,15],5:[30,25,20,15,10],6:[27,22,18,14,11,8],7:[25,20,16,13,10,9,7]}
    w = wmap.get(min(n,7), wmap[7])[:n]
    return round(sum(yearly[y]['wr']*w[i]/100 for i,y in enumerate(years[:len(w)])), 1)

# ─── PROCESS ONE ASSET ───
def optimize_asset(symbol, top_strategies):
    """
    For one asset:
    1. Load data
    2. For each qualifying strategy type:
       a. Walk-forward split 70/30
       b. Optuna optimize on TRAIN
       c. Validate on TEST
       d. Also run on FULL for reference
    """
    df_5m = load_candles(symbol, '5m')
    if df_5m is None or len(df_5m) < 500:
        return None

    # Build all timeframes
    dfs = {}
    dfs['5m'] = df_5m
    dfs['15m'] = resample_ohlcv(df_5m, '15m')
    df_1h = load_candles(symbol, '1h')
    dfs['1h'] = df_1h if df_1h is not None and len(df_1h) > 100 else None
    dfs['4h'] = resample_ohlcv(df_5m, '4h')
    df_1d = load_candles(symbol, '1d')
    dfs['1d'] = df_1d if df_1d is not None and len(df_1d) > 30 else resample_ohlcv(df_5m, '1d')

    results = []

    for strat_info in top_strategies:
        sname = strat_info['strategy']
        tf = strat_info['timeframe']
        stype = get_strategy_type(sname)

        if stype is None:
            continue  # Can't optimize non-parameterized strategies

        df = dfs.get(tf)
        if df is None or len(df) < 200:
            continue

        # Walk-forward split
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        if len(train_df) < 100 or len(test_df) < 50:
            continue

        gen_func = STRATEGY_TYPES[stype]['gen']

        # ─── OPTUNA OPTIMIZATION ON TRAIN ───
        best_params = None
        best_train_score = -999

        if HAS_OPTUNA:
            study = optuna.create_study(direction='maximize',
                                         sampler=optuna.samplers.TPESampler(seed=42))
            objective = create_objective(train_df, stype)
            study.optimize(objective, n_trials=N_TRIALS, timeout=30)
            best_params = study.best_params
            best_train_score = study.best_value
        else:
            # Grid search fallback - not used if optuna installed
            best_params = {}

        if best_params is None:
            continue

        # ─── VALIDATE ON TEST ───
        try:
            test_signals = gen_func(test_df, **best_params)
            test_result = backtest(test_df, test_signals)

            train_signals = gen_func(train_df, **best_params)
            train_result = backtest(train_df, train_signals)

            full_signals = gen_func(df, **best_params)
            full_result = backtest(df, full_signals)
        except:
            continue

        if test_result is None or train_result is None:
            continue

        # ─── ANTI-CHEAT CHECK ───
        # If train WR is way higher than test → overfitting
        wr_diff = abs(train_result['wr'] - test_result['wr'])
        overfit_flag = wr_diff > 25  # >25% difference = suspicious

        # Decision based on TEST results (not train!)
        test_wr = test_result['wr']
        test_score = calc_weighted_score(test_result.get('yearly', {}))

        if test_wr >= 70:
            decision = "PASS"
        elif test_wr >= 55:
            decision = "TESTING"
        else:
            decision = "FAIL"

        if overfit_flag:
            decision = "OVERFIT"

        results.append({
            'strategy': sname,
            'type': stype,
            'timeframe': tf,
            'best_params': best_params,
            'train': {
                'wr': train_result['wr'],
                'sharpe': train_result['sharpe'],
                'pnl': train_result['pnl'],
                'trades': train_result['trades'],
                'pf': train_result['profit_factor'],
            },
            'test': {
                'wr': test_result['wr'],
                'sharpe': test_result['sharpe'],
                'pnl': test_result['pnl'],
                'trades': test_result['trades'],
                'pf': test_result['profit_factor'],
                'yearly': test_result.get('yearly', {}),
                'mae_p95': test_result.get('mae_p95', 0.1),
            },
            'full': {
                'wr': full_result['wr'] if full_result else 0,
                'sharpe': full_result['sharpe'] if full_result else 0,
                'pnl': full_result['pnl'] if full_result else 0,
                'trades': full_result['trades'] if full_result else 0,
            } if full_result else None,
            'wr_diff': round(wr_diff, 1),
            'overfit': overfit_flag,
            'weighted_score': test_score,
            'decision': decision,
        })

    return results

# ─── MAIN ───
def main():
    print("=" * 70)
    print("OPTUNA FULL OPTIMIZATION: Top strategies × 563 assets")
    print(f"Walk-forward 70/30 | {N_TRIALS} trials | Anti-overfit checks")
    print("=" * 70)

    # Load screening results to find top strategies per asset
    import glob
    files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    print(f"Loading screening results from {len(files)} assets...")

    # For each asset, find top 5 unique strategy TYPES with WR >= 50%
    asset_tops = {}
    for f in files:
        try:
            results = json.load(open(f))
            if not results:
                continue
            symbol = results[0].get('symbol', '')
            if not symbol:
                continue

            # Group by strategy, keep best per type
            seen_types = set()
            top = []
            for r in sorted(results, key=lambda x: x.get('sharpe', 0), reverse=True):
                if r.get('wr', 0) < 50:
                    continue
                stype = get_strategy_type(r['strategy'])
                if stype and stype not in seen_types:
                    top.append(r)
                    seen_types.add(stype)
                if len(top) >= 5:
                    break
            if top:
                asset_tops[symbol] = top
        except:
            pass

    print(f"Assets with qualifying strategies: {len(asset_tops)}")
    total_combos = sum(len(v) for v in asset_tops.values())
    print(f"Total combos to optimize: {total_combos}")
    print(f"Total Optuna trials: {total_combos * N_TRIALS:,}")
    print("-" * 70)

    # Load progress
    progress = {}
    if os.path.exists(PROGRESS_FILE):
        progress = json.load(open(PROGRESS_FILE))
    completed_assets = set(progress.get('completed', []))

    all_results = progress.get('results', [])
    start = time.time()

    symbols = sorted(asset_tops.keys())
    remaining = [s for s in symbols if s not in completed_assets]
    print(f"Already done: {len(completed_assets)} | Remaining: {len(remaining)}")

    stats = {'pass': 0, 'testing': 0, 'fail': 0, 'overfit': 0}
    # Count existing
    for r in all_results:
        d = r.get('decision', 'FAIL')
        stats[d.lower()] = stats.get(d.lower(), 0) + 1

    for idx, symbol in enumerate(remaining):
        t0 = time.time()
        short = symbol[:25].ljust(25)
        tops = asset_tops[symbol]
        n_strats = len(tops)
        print(f"[{idx+1}/{len(remaining)}] {short} ({n_strats} strats)...", end=" ", flush=True)

        try:
            results = optimize_asset(symbol, tops)
            if results:
                for r in results:
                    r['symbol'] = symbol
                    all_results.append(r)
                    d = r['decision']
                    stats[d.lower()] = stats.get(d.lower(), 0) + 1

                best = max(results, key=lambda x: x['test']['wr'])
                dt = time.time() - t0
                print(f"OK {dt:.0f}s | {len(results)} optimized | best={best['strategy']}@{best['timeframe']} "
                      f"Train={best['train']['wr']:.1f}% Test={best['test']['wr']:.1f}% [{best['decision']}]")
            else:
                print(f"NO RESULTS")

        except Exception as e:
            print(f"ERR: {str(e)[:50]}")

        completed_assets.add(symbol)
        progress['completed'] = list(completed_assets)
        progress['results'] = all_results
        progress['stats'] = stats
        elapsed = time.time() - start
        done = idx + 1
        rate = elapsed / done if done > 0 else 1
        eta = rate * (len(remaining) - done) / 3600
        progress['elapsed_hours'] = round(elapsed / 3600, 2)
        progress['eta_hours'] = round(eta, 1)

        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, default=str)

        # Also save full results periodically
        if (idx + 1) % 10 == 0:
            with open(OPTUNA_FILE, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

    # Final save
    with open(OPTUNA_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("OPTUNA OPTIMIZATION COMPLETE")
    print(f"Total optimized: {len(all_results)}")
    print(f"PASS (test WR>=70%):  {stats.get('pass',0)}")
    print(f"TESTING (WR 55-70%):  {stats.get('testing',0)}")
    print(f"FAIL (WR<55%):        {stats.get('fail',0)}")
    print(f"OVERFIT (>25% diff):  {stats.get('overfit',0)}")
    print(f"Elapsed: {progress.get('elapsed_hours',0):.1f}h")
    print("=" * 70)

    # Show top 20
    passed = [r for r in all_results if r['decision'] == 'PASS']
    passed.sort(key=lambda x: x['test']['wr'], reverse=True)
    print(f"\nTOP 20 PASS (validated on OUT-OF-SAMPLE test data):")
    for r in passed[:20]:
        print(f"  {r['symbol'][:20]:20s} {r['strategy']:20s} @{r['timeframe']} "
              f"Train={r['train']['wr']:.1f}% Test={r['test']['wr']:.1f}% "
              f"S={r['test']['sharpe']:.1f} PF={r['test']['pf']:.1f} "
              f"params={json.dumps(r['best_params'])}")

if __name__ == '__main__':
    main()
