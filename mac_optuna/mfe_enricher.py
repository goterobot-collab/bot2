#!/usr/bin/env python3
"""
MFE ENRICHER: Re-backtestea cada grail y extrae MFE (Maximum Favorable Excursion)
- Lee santo_grial.json (R1) y santo_grial_r2.json (R2 si existe)
- Corre backtest con params exactos del grail
- Calcula MFE per-trade (máximo PnL no realizado antes de cerrar)
- Agrega campos: mfe_p50, mfe_p75, mfe_p95, tp_empirico, sl_empirico, n_trades_test
- Guarda en santo_grial_enriched.json
- Progress reanudable
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

DB_PATH = "/Users/sabrina/Code/activos binace /activos_binance.db"
PROJECT_DIR = "/Users/sabrina/CLAUDE CODE/Estrategias"
GRAIL_R1 = os.path.join(PROJECT_DIR, "data", "santo_grial.json")
GRAIL_R2 = os.path.join(PROJECT_DIR, "data", "santo_grial_r2.json")
OUTPUT_FILE = os.path.join(PROJECT_DIR, "data", "santo_grial_enriched.json")
PROGRESS_FILE = os.path.join(PROJECT_DIR, "data", "mfe_progress.json")
COMMISSION = 0.001
SLIPPAGE = 0.0005
MAX_WORKERS = 6

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

# ─── INDICATORS (same as optuna_full_v2) ───
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
def macd_ind(c,f=12,s=26,sig=9):
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

# ─── BACKTEST WITH MFE TRACKING ───
def backtest_with_mfe(df, signals):
    """Backtest que trackea MAE y MFE per-trade"""
    if signals is None: return None
    sig = signals.values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)

    trades = []
    pos = 0; ep = 0.; ei = 0; mae = 0.; mfe = 0.

    for i in range(1, n):
        s = sig[i-1]; p = opens[i]
        if pos == 0 and s == 1:
            ep = p * (1 + SLIPPAGE + COMMISSION)
            ei = i; mae = 0.; mfe = 0.; pos = 1
        elif pos == 1:
            # MAE = worst unrealized loss
            low_pct = (lows[i] - ep) / ep
            if low_pct < mae: mae = low_pct
            # MFE = best unrealized gain
            high_pct = (highs[i] - ep) / ep
            if high_pct > mfe: mfe = high_pct

            if s == -1:
                xp = p * (1 - SLIPPAGE - COMMISSION)
                pnl = (xp - ep) / ep * 100
                trades.append({
                    'pnl': pnl,
                    'mae': mae * 100,  # as percentage
                    'mfe': mfe * 100,  # as percentage
                    'bars': i - ei,
                    'won': pnl > 0
                })
                pos = 0

    return trades

# ─── STRATEGY SIGNAL GENERATORS ───
# R1 strategies (22 types)
def gen_rsi(df, period, buy, sell):
    r=rsi(df['close'],period); sig=pd.Series(0,index=df.index); sig[r<buy]=1; sig[r>sell]=-1; return sig

def gen_bb(df, period, std_mult):
    m,u,l=bb(df['close'],period,std_mult); sig=pd.Series(0,index=df.index); sig[df['close']<l]=1; sig[df['close']>u]=-1; return sig

def gen_ema(df, fast, slow):
    ef=ema(df['close'],fast); es=ema(df['close'],slow); sig=pd.Series(0,index=df.index)
    sig[(ef>es)&(ef.shift()<=es.shift())]=1; sig[(ef<es)&(ef.shift()>=es.shift())]=-1; return sig

def gen_macd(df, fast, slow, signal):
    m,s,h=macd_ind(df['close'],fast,slow,signal); sig=pd.Series(0,index=df.index)
    sig[(m>s)&(m.shift()<=s.shift())]=1; sig[(m<s)&(m.shift()>=s.shift())]=-1; return sig

def gen_stoch(df, k_period, d_period, buy, sell):
    k,d=stoch(df['high'],df['low'],df['close'],k_period,d_period); sig=pd.Series(0,index=df.index)
    sig[(k<buy)&(k>d)&(k.shift()<=d.shift())]=1; sig[(k>sell)&(k<d)&(k.shift()>=d.shift())]=-1; return sig

def gen_keltner(df, period, mult):
    m=ema(df['close'],period); a=atr(df['high'],df['low'],df['close'],period); u=m+mult*a; l=m-mult*a
    sig=pd.Series(0,index=df.index); sig[df['close']<l]=1; sig[df['close']>u]=-1; return sig

def gen_zscore(df, lookback, z_thresh):
    m=df['close'].rolling(lookback).mean(); s=df['close'].rolling(lookback).std(); z=(df['close']-m)/(s+1e-10)
    sig=pd.Series(0,index=df.index); sig[z<-z_thresh]=1; sig[z>z_thresh]=-1; return sig

def gen_cci(df, period, buy, sell):
    tp=(df['high']+df['low']+df['close'])/3; m=tp.rolling(period).mean(); d=tp.rolling(period).apply(lambda x: np.mean(np.abs(x-x.mean())),raw=True)
    c=((tp-m)/(0.015*d+1e-10)); sig=pd.Series(0,index=df.index); sig[c<buy]=1; sig[c>sell]=-1; return sig

def gen_williamsr(df, period, buy, sell):
    hh=df['high'].rolling(period).max(); ll=df['low'].rolling(period).min()
    wr=-100*(hh-df['close'])/(hh-ll+1e-10); sig=pd.Series(0,index=df.index); sig[wr<buy]=1; sig[wr>sell]=-1; return sig

def gen_donchian(df, period, offset):
    hh=df['high'].rolling(period).max(); ll=df['low'].rolling(period).min()
    sig=pd.Series(0,index=df.index); sig[df['close']>hh.shift(offset)]=1; sig[df['close']<ll.shift(offset)]=-1; return sig

def gen_adx(df, period, threshold):
    a,pdi,mdi=adx_calc(df['high'],df['low'],df['close'],period); sig=pd.Series(0,index=df.index)
    sig[(a>threshold)&(pdi>mdi)]=1; sig[(a>threshold)&(mdi>pdi)]=-1; return sig

def gen_momentum(df, period, threshold):
    m=df['close']/df['close'].shift(period)*100-100; sig=pd.Series(0,index=df.index)
    sig[m>threshold]=1; sig[m<-threshold]=-1; return sig

def gen_hma(df, period):
    h1=ema(df['close'],period//2); h2=ema(df['close'],period); hma=ema(2*h1-h2,int(period**0.5))
    sig=pd.Series(0,index=df.index); sig[(hma>hma.shift())&(hma.shift()<=hma.shift(2))]=1
    sig[(hma<hma.shift())&(hma.shift()>=hma.shift(2))]=-1; return sig

def gen_rsi_macd(df, rsi_period, rsi_buy, macd_fast, macd_slow, macd_sig):
    r=rsi(df['close'],rsi_period); m,s,h=macd_ind(df['close'],macd_fast,macd_slow,macd_sig)
    sig=pd.Series(0,index=df.index); sig[(r<rsi_buy)&(m>s)]=1; sig[(r>100-rsi_buy)&(m<s)]=-1; return sig

def gen_vwap(df, period, std_mult):
    tp=(df['high']+df['low']+df['close'])/3; v=df['volume']
    cv=(tp*v).rolling(period).sum(); sv=v.rolling(period).sum(); vw=cv/(sv+1e-10)
    s2=((tp-vw)**2*v).rolling(period).sum(); vstd=(s2/(sv+1e-10))**0.5
    sig=pd.Series(0,index=df.index); sig[df['close']<vw-std_mult*vstd]=1; sig[df['close']>vw+std_mult*vstd]=-1; return sig

def gen_volspike(df, vol_period, vol_mult):
    vm=df['volume'].rolling(vol_period).mean(); sig=pd.Series(0,index=df.index)
    spike=df['volume']>vm*vol_mult; green=df['close']>df['open']; red=df['close']<df['open']
    sig[spike&green]=1; sig[spike&red]=-1; return sig

def gen_engulfing(df, body_ratio):
    b=df['close']-df['open']; ab=b.abs(); pb=df['close'].shift()-df['open'].shift(); apb=pb.abs()
    sig=pd.Series(0,index=df.index)
    bull=(b>0)&(pb<0)&(ab>apb*body_ratio); bear=(b<0)&(pb>0)&(ab>apb*body_ratio)
    sig[bull]=1; sig[bear]=-1; return sig

def gen_ichimoku(df, tenkan, kijun, senkou_b):
    th=df['high'].rolling(tenkan).max(); tl=df['low'].rolling(tenkan).min(); t=(th+tl)/2
    kh=df['high'].rolling(kijun).max(); kl=df['low'].rolling(kijun).min(); k=(kh+kl)/2
    sig=pd.Series(0,index=df.index); sig[(t>k)&(t.shift()<=k.shift())]=1; sig[(t<k)&(t.shift()>=k.shift())]=-1; return sig

def gen_squeeze(df, bb_period, bb_std, kc_period, kc_mult):
    bm,bu,bl=bb(df['close'],bb_period,bb_std); a=atr(df['high'],df['low'],df['close'],kc_period)
    km=ema(df['close'],kc_period); ku=km+kc_mult*a; kl=km-kc_mult*a
    squeeze=(bl>kl)&(bu<ku); mom=df['close']-df['close'].rolling(bb_period).mean()
    sig=pd.Series(0,index=df.index); sig[squeeze&(mom>0)&(mom.shift()<=0)]=1; sig[squeeze&(mom<0)&(mom.shift()>=0)]=-1; return sig

def gen_tripleema(df, fast, mid, slow):
    ef=ema(df['close'],fast); em=ema(df['close'],mid); es=ema(df['close'],slow)
    sig=pd.Series(0,index=df.index); sig[(ef>em)&(em>es)&(ef.shift()<=em.shift())]=1
    sig[(ef<em)&(em<es)&(ef.shift()>=em.shift())]=-1; return sig

def gen_pivot(df, lookback):
    hh=df['high'].rolling(lookback).max(); ll=df['low'].rolling(lookback).min(); pp=(hh+ll+df['close'])/3
    r1=2*pp-ll; s1=2*pp-hh; sig=pd.Series(0,index=df.index); sig[df['close']<s1]=1; sig[df['close']>r1]=-1; return sig

def gen_obv(df, period, signal):
    o=(df['volume']*np.sign(df['close'].diff())).cumsum(); sig_line=ema(o,signal); osm=ema(o,period)
    sig=pd.Series(0,index=df.index); sig[(o>osm)&(o.shift()<=osm.shift())]=1; sig[(o<osm)&(o.shift()>=osm.shift())]=-1; return sig

# Strategy dispatcher
STRATEGY_GENERATORS = {
    'RSI': gen_rsi, 'BB': gen_bb, 'EMA': gen_ema, 'MACD': gen_macd,
    'Stoch': gen_stoch, 'Keltner': gen_keltner, 'ZScore': gen_zscore,
    'CCI': gen_cci, 'WilliamsR': gen_williamsr, 'Donchian': gen_donchian,
    'ADX': gen_adx, 'Momentum': gen_momentum, 'HMA': gen_hma,
    'RSI_MACD': gen_rsi_macd, 'VWAP': gen_vwap, 'VolSpike': gen_volspike,
    'Engulfing': gen_engulfing, 'Ichimoku': gen_ichimoku, 'Squeeze': gen_squeeze,
    'TripleEMA': gen_tripleema, 'Pivot': gen_pivot, 'OBV': gen_obv
}

# ─── R2 STRATEGIES (import if available) ───
try:
    from strategies_round2 import STRATEGY_TYPES_R2
    for name, spec in STRATEGY_TYPES_R2.items():
        if name not in STRATEGY_GENERATORS:
            STRATEGY_GENERATORS[name] = spec['gen']
    print(f"  R2 strategies loaded: {len(STRATEGY_TYPES_R2)} types")
except ImportError:
    print("  R2 strategies not available, skipping")

# ─── PROCESS ONE GRAIL ───
def process_grail(grail_idx, grail):
    """Re-run backtest for a grail with MFE tracking, on TEST portion only"""
    symbol = grail['symbol']
    strategy = grail['strategy']
    tf = grail['timeframe']
    params = grail['best_params']

    if strategy not in STRATEGY_GENERATORS:
        return grail_idx, None, f"Unknown strategy: {strategy}"

    gen_func = STRATEGY_GENERATORS[strategy]

    # Load data
    if tf in ('5m', '1h', '1d'):
        df = load_candles(symbol, tf)
    else:
        df5 = load_candles(symbol, '5m')
        if df5 is None: return grail_idx, None, "No 5m data"
        df = resample(df5, tf)

    if df is None or len(df) < 100:
        return grail_idx, None, "Insufficient data"

    # Walk-forward split: same as optuna — 70% train, 30% test
    split = int(len(df) * 0.7)
    test_df = df.iloc[split:]

    if len(test_df) < 50:
        return grail_idx, None, "Test too short"

    # Generate signals with exact grail params
    try:
        signals = gen_func(test_df, **params)
    except Exception as e:
        return grail_idx, None, f"Signal gen error: {e}"

    # Run backtest with MFE
    trades = backtest_with_mfe(test_df, signals)

    if not trades or len(trades) < 3:
        return grail_idx, None, "Too few trades in test"

    # Extract MFE/MAE stats
    mfes = [t['mfe'] for t in trades]
    maes = [abs(t['mae']) for t in trades]  # positive values
    pnls = [t['pnl'] for t in trades]
    bars = [t['bars'] for t in trades]
    wins = [t for t in trades if t['won']]

    # MFE of WINNING trades (for TP calculation)
    win_mfes = [t['mfe'] for t in wins] if wins else mfes
    # MAE of WINNING trades (for SL calculation)
    win_maes = [abs(t['mae']) for t in wins] if wins else maes

    mfe_p50 = float(np.percentile(win_mfes, 50)) if win_mfes else 0
    mfe_p75 = float(np.percentile(win_mfes, 75)) if win_mfes else 0
    mfe_p95 = float(np.percentile(win_mfes, 95)) if win_mfes else 0
    mae_p95 = float(np.percentile(win_maes, 95)) if win_maes else 0

    # Empirical TP/SL
    tp_empirico = round(mfe_p75 * 0.8, 2)  # capture 80% of P75 MFE
    sl_empirico = round(mae_p95 * 1.5, 2)  # MAE P95 × 1.5

    # Hard caps
    sl_empirico = max(2.0, min(sl_empirico, 40.0))
    tp_empirico = max(1.0, tp_empirico)

    # Validate SL vs leverage
    safe_lev = grail.get('safe_leverage', 1)
    max_sl_for_lev = 80.0 / max(safe_lev, 1)  # 80% of liquidation point
    if sl_empirico > max_sl_for_lev:
        sl_empirico = round(max_sl_for_lev, 2)

    # Duration
    dur_p95 = float(np.percentile(bars, 95))

    enrichment = {
        'mfe_p50': round(mfe_p50, 2),
        'mfe_p75': round(mfe_p75, 2),
        'mfe_p95': round(mfe_p95, 2),
        'mae_p95_recalc': round(mae_p95, 2),
        'tp_empirico': tp_empirico,
        'sl_empirico': sl_empirico,
        'rr_ratio': round(tp_empirico / sl_empirico, 2) if sl_empirico > 0 else 0,
        'n_trades_test': len(trades),
        'n_wins_test': len(wins),
        'avg_mfe_win': round(np.mean(win_mfes), 2) if win_mfes else 0,
        'avg_mae_win': round(np.mean(win_maes), 2) if win_maes else 0,
        'max_mfe': round(max(mfes), 2),
        'dur_p95_bars': round(dur_p95, 0),
        'max_duration_bars': round(dur_p95 * 1.5, 0),
    }

    return grail_idx, enrichment, None

# ─── MAIN ───
def main():
    print("=" * 60)
    print("MFE ENRICHER — Adding empirical TP/SL to grails")
    print("=" * 60)

    # Load grails
    grails = []
    if os.path.exists(GRAIL_R1):
        r1 = json.load(open(GRAIL_R1))
        print(f"R1 grails: {len(r1)}")
        grails.extend(r1)

    if os.path.exists(GRAIL_R2):
        r2 = json.load(open(GRAIL_R2))
        print(f"R2 grails: {len(r2)}")
        grails.extend(r2)

    print(f"Total grails to enrich: {len(grails)}")

    # Load progress
    completed = set()
    if os.path.exists(PROGRESS_FILE):
        prog = json.load(open(PROGRESS_FILE))
        completed = set(prog.get('completed', []))
        print(f"Already completed: {len(completed)}")

    # Load existing output
    enriched = []
    if os.path.exists(OUTPUT_FILE):
        enriched = json.load(open(OUTPUT_FILE))
        print(f"Existing enriched: {len(enriched)}")
    else:
        enriched = [None] * len(grails)

    # Ensure enriched list matches grails length
    if len(enriched) < len(grails):
        enriched.extend([None] * (len(grails) - len(enriched)))

    # Filter pending
    pending = [(i, g) for i, g in enumerate(grails) if i not in completed]
    print(f"Pending: {len(pending)}")

    if not pending:
        print("All done!")
        return

    t0 = time.time()
    done = 0
    errors = 0

    # Process in parallel
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_grail, i, g): i for i, g in pending}

        for future in as_completed(futures):
            try:
                grail_idx, enrichment, error = future.result(timeout=120)
            except Exception as e:
                grail_idx = futures[future]
                enrichment = None
                error = str(e)

            if enrichment:
                # Merge enrichment into grail
                merged = {**grails[grail_idx], **enrichment}
                enriched[grail_idx] = merged
            else:
                # Keep original grail without MFE
                enriched[grail_idx] = grails[grail_idx]
                if error:
                    errors += 1

            completed.add(grail_idx)
            done += 1

            # Save progress every 50
            if done % 50 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(pending) - done) / rate / 60 if rate > 0 else 0
                print(f"  [{done}/{len(pending)}] {rate:.1f}/s | errors: {errors} | ETA: {eta:.0f}min")

                # Save progress
                json.dump({'completed': list(completed)}, open(PROGRESS_FILE, 'w'))
                # Save enriched (filter Nones)
                valid = [e for e in enriched if e is not None]
                json.dump(valid, open(OUTPUT_FILE, 'w'), indent=1)

    # Final save
    valid = [e for e in enriched if e is not None]
    json.dump(valid, open(OUTPUT_FILE, 'w'), indent=1)
    json.dump({'completed': list(completed)}, open(PROGRESS_FILE, 'w'))

    elapsed = time.time() - t0
    with_mfe = len([e for e in valid if 'mfe_p50' in e])

    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed/60:.1f} min")
    print(f"Total enriched: {len(valid)} | With MFE: {with_mfe} | Errors: {errors}")
    print(f"Output: {OUTPUT_FILE}")

    # Quick stats
    mfes = [e['mfe_p75'] for e in valid if 'mfe_p75' in e and e['mfe_p75'] > 0]
    tps = [e['tp_empirico'] for e in valid if 'tp_empirico' in e]
    sls = [e['sl_empirico'] for e in valid if 'sl_empirico' in e]
    rrs = [e['rr_ratio'] for e in valid if 'rr_ratio' in e and e['rr_ratio'] > 0]

    if mfes:
        print(f"\nMFE P75 stats: min={min(mfes):.1f}% med={np.median(mfes):.1f}% max={max(mfes):.1f}%")
    if tps:
        print(f"TP empirico: min={min(tps):.1f}% med={np.median(tps):.1f}% max={max(tps):.1f}%")
    if sls:
        print(f"SL empirico: min={min(sls):.1f}% med={np.median(sls):.1f}% max={max(sls):.1f}%")
    if rrs:
        print(f"R:R ratio: min={min(rrs):.2f} med={np.median(rrs):.2f} max={max(rrs):.2f}")
        good_rr = len([r for r in rrs if r >= 1.0])
        print(f"R:R >= 1.0: {good_rr}/{len(rrs)} ({good_rr/len(rrs)*100:.0f}%)")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
