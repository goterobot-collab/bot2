#!/usr/bin/env python3
"""
OPTUNA ROUND 2: 52 new strategy types × 562 assets
Same smart approach: only test combos where screening showed WR >= 50%
But for round 2 we don't have screening data, so we do a QUICK screen first
"""
import sqlite3, pandas as pd, numpy as np, json, os, sys, time, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

DB_PATH = "/Users/sabrina/Code/activos binace /activos_binance.db"
PROJECT_DIR = "/Users/sabrina/CLAUDE CODE/Estrategias"
PROGRESS_FILE = os.path.join(PROJECT_DIR, "data", "optuna_r2_progress.json")
N_TRIALS = 25
COMMISSION = 0.001; SLIPPAGE = 0.0005; MIN_TRADES_FULL = 15
os.makedirs(os.path.join(PROJECT_DIR, "data"), exist_ok=True)

import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, PROJECT_DIR)
from strategies_round2 import STRATEGY_TYPES_R2

print(f"Round 2 strategies: {len(STRATEGY_TYPES_R2)}")

# ─── DB ───
def load_candles(symbol, tf="5m"):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT ts,open,high,low,close,volume FROM candles WHERE symbol=? AND timeframe=? ORDER BY ts", conn, params=(symbol, tf))
    conn.close()
    if len(df) == 0: return None
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    for c in ['open','high','low','close','volume']: df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    return df

def resample(df, target):
    m = {'15m':'15min','4h':'4h','1d':'1D'}
    r = m.get(target)
    if not r: return None
    return df.resample(r).agg({'ts':'first','open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['close'])

def get_symbols():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT symbol, COUNT(*) as bars FROM candles WHERE timeframe='5m' GROUP BY symbol ORDER BY bars DESC", conn)
    conn.close()
    return list(df.itertuples(index=False, name=None))

# ─── BACKTEST ───
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
                trades.append((pnl,mae,i-ei,yr)); pos=0
    if len(trades)<5: return None
    pnls=np.array([t[0] for t in trades]); maes=np.array([t[1] for t in trades])
    years=np.array([t[3] for t in trades])
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
            'max_drawdown':round(dd,2),'mae_p95':round(mae95,4),'sharpe':round(sharpe,2),'yearly':yearly}

def calc_score(yearly):
    years=sorted(yearly.keys(),reverse=True); n=len(years)
    if n==0: return 0
    wm={1:[100],2:[55,45],3:[40,30,30],4:[35,25,25,15],5:[30,25,20,15,10],6:[27,22,18,14,11,8],7:[25,20,16,13,10,9,7]}
    w=wm.get(min(n,7),wm[7])[:n]
    return round(sum(yearly[y]['wr']*w[i]/100 for i,y in enumerate(years[:len(w)])),1)

# ─── MAIN ───
def main():
    symbols = get_symbols()
    print(f"Symbols: {len(symbols)} | Strategy types: {len(STRATEGY_TYPES_R2)}")
    print(f"Quick screen + Optuna {N_TRIALS} trials per promising combo")
    print("="*70)

    progress = {'completed': [], 'grails': [], 'stats': {}}
    if os.path.exists(PROGRESS_FILE):
        progress = json.load(open(PROGRESS_FILE))
    done_set = set(progress.get('completed', []))
    all_grails = progress.get('grails', [])
    remaining = [s for s,_ in symbols if s not in done_set]
    print(f"Done: {len(done_set)} | Remaining: {len(remaining)}")

    start = time.time()
    strat_names = list(STRATEGY_TYPES_R2.keys())

    for idx, symbol in enumerate(remaining):
        t0 = time.time()
        short = symbol[:25].ljust(25)

        df_5m = load_candles(symbol, '5m')
        if df_5m is None or len(df_5m) < 500:
            done_set.add(symbol); continue

        dfs = {'5m': df_5m, '15m': resample(df_5m,'15m'), '4h': resample(df_5m,'4h')}
        df_1h = load_candles(symbol, '1h')
        dfs['1h'] = df_1h if df_1h is not None and len(df_1h)>100 else None
        df_1d = load_candles(symbol, '1d')
        dfs['1d'] = df_1d if df_1d is not None and len(df_1d)>30 else resample(df_5m,'1d')

        sym_grails = []
        for tf, df in dfs.items():
            if df is None or len(df) < 200: continue
            split = int(len(df)*0.7)
            train = df.iloc[:split].copy(); test = df.iloc[split:].copy()
            if len(train)<100 or len(test)<50: continue

            for sname in strat_names:
                info = STRATEGY_TYPES_R2[sname]
                gen = info['gen']; space = info['space']

                # Quick screen: run with default-ish params
                try:
                    # First try a quick backtest to see if this strategy has ANY signal
                    test_trial = optuna.trial.FixedTrial({'dummy': 0})
                    # Just run 1 optuna trial as quick screen
                    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
                    def obj(trial):
                        params = space(trial)
                        try:
                            sig = gen(train, **params)
                            r = backtest(train, sig)
                            if r is None: return -999
                            return r['wr']*0.7 + min(r['sharpe'],50)*0.3
                        except: return -999
                    study.optimize(obj, n_trials=N_TRIALS, timeout=45)
                    if study.best_value <= 0: continue
                    bp = study.best_params

                    test_sig = gen(test, **bp); test_r = backtest(test, test_sig)
                    train_sig = gen(train, **bp); train_r = backtest(train, train_sig)
                    full_sig = gen(df, **bp); full_r = backtest(df, full_sig)
                    if not test_r or not train_r or not full_r: continue
                    if test_r['wr'] < 70 or test_r['pnl'] <= 0: continue
                    if full_r['trades'] < MIN_TRADES_FULL: continue
                    wr_diff = abs(train_r['wr'] - test_r['wr'])
                    if wr_diff > 30: continue

                    mae = test_r.get('mae_p95',0.1)
                    grail = {
                        'strategy': sname, 'symbol': symbol, 'timeframe': tf, 'best_params': bp,
                        'train': {'wr':train_r['wr'],'pnl':train_r['pnl'],'sharpe':train_r['sharpe'],'trades':train_r['trades'],'pf':train_r['profit_factor']},
                        'test': {'wr':test_r['wr'],'pnl':test_r['pnl'],'sharpe':test_r['sharpe'],'trades':test_r['trades'],'pf':test_r['profit_factor'],'yearly':test_r.get('yearly',{}),'mae_p95':mae,'max_dd':test_r['max_drawdown']},
                        'full': {'wr':full_r['wr'],'pnl':full_r['pnl'],'sharpe':full_r['sharpe'],'trades':full_r['trades'],'pf':full_r['profit_factor'],'max_dd':full_r['max_drawdown']},
                        'wr_diff': round(wr_diff,1), 'weighted_score': calc_score(test_r.get('yearly',{})),
                        'safe_leverage': min(20, int(1/(mae*2.5+1e-10))),
                    }
                    sym_grails.append(grail); all_grails.append(grail)
                except: continue

        done_set.add(symbol)
        dt = time.time()-t0; elapsed = time.time()-start; done_n = idx+1
        rate = elapsed/done_n; eta = rate*(len(remaining)-done_n)/3600

        if sym_grails:
            best = max(sym_grails, key=lambda g: g['full']['wr'])
            print(f"[{done_n}/{len(remaining)}] {short} 🏆 {len(sym_grails)} grails {dt:.0f}s | {best['strategy']}@{best['timeframe']} Full={best['full']['trades']}T WR={best['full']['wr']:.1f}%")
        else:
            print(f"[{done_n}/{len(remaining)}] {short} — {dt:.0f}s")

        progress['completed'] = list(done_set); progress['grails'] = all_grails
        progress['stats'] = {'grails':len(all_grails),'done':len(done_set),'total':len(symbols),'elapsed_h':round(elapsed/3600,2),'eta_h':round(eta,1)}
        with open(PROGRESS_FILE,'w') as f: json.dump(progress, f, default=str)

    print(f"\nDONE! Grails: {len(all_grails)}")

if __name__ == '__main__':
    main()
