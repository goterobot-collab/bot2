#!/usr/bin/env python3
"""
FACTORY RUNNER: Ejecuta 200+ estrategias × 563 assets × 5 TFs
Reanudable. Guarda en factory_progress.json y factory_grails.json
NO toca los griales existentes.
"""
import sqlite3, pandas as pd, numpy as np, json, os, sys, time, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

DB_PATH = "/Users/sabrina/Code/activos binace /activos_binance.db"
PROJECT = "/Users/sabrina/CLAUDE CODE/Estrategias"
PROGRESS_FILE = os.path.join(PROJECT, "data", "factory_progress.json")
GRAIL_FILE = os.path.join(PROJECT, "data", "factory_grails.json")
N_TRIALS = 25; COMMISSION = 0.001; SLIPPAGE = 0.0005; MIN_TRADES = 15
os.makedirs(os.path.join(PROJECT, "data"), exist_ok=True)

import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, PROJECT)

# Try to load factory strategies
try:
    from strategy_factory import FACTORY_STRATS
    print(f"Factory strategies loaded: {len(FACTORY_STRATS)}")
except ImportError:
    print("ERROR: strategy_factory.py not ready yet. Waiting...")
    sys.exit(1)

def load_candles(symbol, tf="5m"):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT ts,open,high,low,close,volume FROM candles WHERE symbol=? AND timeframe=? ORDER BY ts", conn, params=(symbol, tf))
    conn.close()
    if len(df)==0: return None
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    for c in ['open','high','low','close','volume']: df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    return df

def resample(df, t):
    m={'15m':'15min','4h':'4h','1d':'1D'}; r=m.get(t)
    if not r: return None
    return df.resample(r).agg({'ts':'first','open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['close'])

def backtest(df, signals):
    if signals is None: return None
    sig=signals.values; opens=df['open'].values; lows=df['low'].values; n=len(df)
    trades=[]; pos=0; ep=0.; ei=0; mae=0.
    for i in range(1,n):
        s=sig[i-1]; p=opens[i]
        if pos==0 and s==1: ep=p*(1+SLIPPAGE+COMMISSION); ei=i; mae=0.; pos=1
        elif pos==1:
            lp=(lows[i]-ep)/ep
            if lp<mae: mae=lp
            if s==-1:
                xp=p*(1-SLIPPAGE-COMMISSION); pnl=(xp-ep)/ep*100
                yr=df.index[ei].year if hasattr(df.index[ei],'year') else 2025
                dur = (df.index[i]-df.index[ei]).total_seconds()/3600
                trades.append((pnl,mae,i-ei,yr,dur)); pos=0
    if len(trades)<5: return None
    pnls=np.array([t[0] for t in trades]); maes=np.array([t[1] for t in trades])
    years=np.array([t[3] for t in trades]); durs=np.array([t[4] for t in trades])
    wins=(pnls>0).sum(); total=len(pnls); wr=wins/total*100
    wm=pnls>0; avg_w=pnls[wm].mean() if wm.any() else 0; avg_l=pnls[~wm].mean() if (~wm).any() else 0
    losses=(~wm).sum(); pf=abs(avg_w*wins/(avg_l*losses+1e-10)) if losses>0 else 999
    cum=pnls.cumsum(); peak=np.maximum.accumulate(cum); dd=(peak-cum).max()
    mae95=abs(np.percentile(maes,95)) if len(maes)>0 else 0.1
    sharpe=(pnls.mean()/(pnls.std()+1e-10))*np.sqrt(252)
    yearly={}
    for y in np.unique(years):
        m=years==y; yp=pnls[m]; yw=(yp>0).sum()
        yearly[str(y)]={'wr':round(yw/len(yp)*100,1),'trades':int(len(yp)),'pnl':round(yp.sum(),2)}
    return {'trades':int(total),'wr':round(wr,1),'pnl':round(pnls.sum(),2),'profit_factor':round(pf,2),
            'max_drawdown':round(dd,2),'mae_p95':round(mae95,4),'sharpe':round(sharpe,2),'yearly':yearly,
            'dur_avg':round(np.mean(durs),1),'dur_p95':round(np.percentile(durs,95),1),'dur_max':round(np.max(durs),1)}

def get_symbols():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT symbol, COUNT(*) as bars FROM candles WHERE timeframe='5m' GROUP BY symbol ORDER BY bars DESC", conn)
    conn.close()
    return list(df['symbol'])

def main():
    symbols = get_symbols()
    strat_names = list(FACTORY_STRATS.keys())
    print(f"Symbols: {len(symbols)} | Strategies: {len(strat_names)}")
    print(f"Total combos: {len(symbols) * len(strat_names) * 5} (× {N_TRIALS} trials)")
    print("="*70)

    progress = {'completed':[], 'grails':[], 'stats':{}}
    if os.path.exists(PROGRESS_FILE):
        progress = json.load(open(PROGRESS_FILE))
    done_set = set(progress.get('completed',[])); all_grails = progress.get('grails',[])
    remaining = [s for s in symbols if s not in done_set]
    print(f"Done: {len(done_set)} | Remaining: {len(remaining)}")

    start = time.time()
    for idx, symbol in enumerate(remaining):
        t0 = time.time()
        df_5m = load_candles(symbol, '5m')
        if df_5m is None or len(df_5m)<500:
            done_set.add(symbol); continue

        dfs = {'5m': df_5m, '15m': resample(df_5m,'15m'), '4h': resample(df_5m,'4h')}
        df_1h = load_candles(symbol, '1h')
        dfs['1h'] = df_1h if df_1h is not None and len(df_1h)>100 else None
        df_1d = load_candles(symbol, '1d')
        dfs['1d'] = df_1d if df_1d is not None and len(df_1d)>30 else resample(df_5m,'1d')

        sym_grails = []
        for tf, df in dfs.items():
            if df is None or len(df)<200: continue
            split=int(len(df)*0.7); train=df.iloc[:split].copy(); test_df=df.iloc[split:].copy()
            if len(train)<100 or len(test_df)<50: continue

            for sname in strat_names:
                info = FACTORY_STRATS[sname]; gen=info['gen']; space=info['space']
                try:
                    def obj(trial):
                        params = space(trial)
                        try:
                            sig = gen(train, **params)
                            r = backtest(train, sig)
                            if r is None: return -999
                            return r['wr']*0.7 + min(r['sharpe'],50)*0.3
                        except: return -999
                    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
                    study.optimize(obj, n_trials=N_TRIALS, timeout=45)
                    if study.best_value <= 0: continue
                    bp = study.best_params

                    test_sig=gen(test_df,**bp); test_r=backtest(test_df,test_sig)
                    train_sig=gen(train,**bp); train_r=backtest(train,train_sig)
                    full_sig=gen(df,**bp); full_r=backtest(df,full_sig)
                    if not test_r or not train_r or not full_r: continue
                    if test_r['wr']<70 or test_r['pnl']<=0: continue
                    if full_r['trades']<MIN_TRADES: continue
                    wr_diff=abs(train_r['wr']-test_r['wr'])
                    if wr_diff>30: continue

                    mae=test_r.get('mae_p95',0.1)
                    grail = {
                        'strategy':sname,'symbol':symbol,'timeframe':tf,'best_params':bp,
                        'train':{'wr':train_r['wr']},
                        'test':{'wr':test_r['wr'],'pnl':test_r['pnl'],'sharpe':test_r['sharpe'],'trades':test_r['trades'],'pf':test_r['profit_factor'],'yearly':test_r.get('yearly',{}),'mae_p95':mae,'max_dd':test_r['max_drawdown']},
                        'full':{'wr':full_r['wr'],'pnl':full_r['pnl'],'sharpe':full_r['sharpe'],'trades':full_r['trades'],'pf':full_r['profit_factor'],'max_dd':full_r['max_drawdown']},
                        'duration':{'avg_h':full_r.get('dur_avg',0),'p95_h':full_r.get('dur_p95',0),'max_h':full_r.get('dur_max',0)},
                        'wr_diff':round(wr_diff,1),'safe_leverage':min(20,int(1/(mae*2.5+1e-10))),
                    }
                    sym_grails.append(grail); all_grails.append(grail)
                except: continue

        done_set.add(symbol); dt=time.time()-t0; elapsed=time.time()-start
        done_n=idx+1; rate=elapsed/done_n; eta=rate*(len(remaining)-done_n)/3600

        if sym_grails:
            best=max(sym_grails,key=lambda g:g['full']['wr'])
            print(f"[{done_n}/{len(remaining)}] {symbol[:22]:22s} 🏆 {len(sym_grails)} | {best['strategy'][:20]}@{best['timeframe']} Full={best['full']['trades']}T WR={best['full']['wr']:.1f}% | {dt:.0f}s")
        else:
            print(f"[{done_n}/{len(remaining)}] {symbol[:22]:22s} — | {dt:.0f}s")

        progress['completed']=list(done_set); progress['grails']=all_grails
        progress['stats']={'grails':len(all_grails),'done':len(done_set),'total':len(symbols),'elapsed_h':round(elapsed/3600,2),'eta_h':round(eta,1),'strategies':len(strat_names)}
        with open(PROGRESS_FILE,'w') as f: json.dump(progress,f,default=str)
        if done_n%5==0 or sym_grails:
            with open(GRAIL_FILE,'w') as f: json.dump(all_grails,f,indent=2,default=str)

    with open(GRAIL_FILE,'w') as f: json.dump(all_grails,f,indent=2,default=str)
    print(f"\nFACTORY DONE! Grails: {len(all_grails)}")

if __name__=='__main__':
    main()
