#!/usr/bin/env python3
"""
OPTUNA SMART: Solo combos prometedores (WR>=60% en screening)
12,291 combos × 30 trials = ~1 hora
Busca el MÁXIMO de cada estrategia para cada activo en cada TF
"""

import sqlite3, pandas as pd, numpy as np, json, os, sys, time, glob, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

DB_PATH = "/Users/sabrina/Code/activos binace /activos_binance.db"
PROJECT_DIR = "/Users/sabrina/CLAUDE CODE/Estrategias"
PROGRESS_FILE = os.path.join(PROJECT_DIR, "data", "optuna_smart_progress.json")
GRAIL_FILE = os.path.join(PROJECT_DIR, "data", "santo_grial_v2.json")
RESULTS_DIR = os.path.join(PROJECT_DIR, "data", "results")
N_TRIALS = 30
COMMISSION = 0.001
SLIPPAGE = 0.0005
MIN_TRADES_FULL = 15  # Minimum trades in FULL history
os.makedirs(os.path.join(PROJECT_DIR, "data"), exist_ok=True)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── DB ───
def load_candles(symbol, tf="5m"):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT ts,open,high,low,close,volume FROM candles WHERE symbol=? AND timeframe=? ORDER BY ts", conn, params=(symbol, tf))
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
def macd_calc(c,f=12,s=26,sig=9):
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

# ─── BACKTEST ───
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
                xp=p*(1-SLIPPAGE-COMMISSION); pnl=(xp-ep)/ep*100
                yr=df.index[ei].year if hasattr(df.index[ei],'year') else 2025
                trades.append((pnl,mae,i-ei,yr)); pos=0
    if len(trades)<5: return None
    pnls=np.array([t[0] for t in trades]); maes=np.array([t[1] for t in trades])
    years=np.array([t[3] for t in trades])
    wins=(pnls>0).sum(); total=len(pnls); wr=wins/total*100
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
    return {'trades':int(total),'wr':round(wr,1),'pnl':round(pnls.sum(),2),
            'profit_factor':round(pf,2),'max_drawdown':round(dd,2),
            'mae_p95':round(mae95,4),'sharpe':round(sharpe,2),'yearly':yearly}

# ─── ALL STRATEGY TYPES ───
STYPES = {}
def reg(name, gen, space):
    STYPES[name] = {'gen': gen, 'space': space}

# RSI
reg('RSI', lambda df,period,buy,sell: (lambda r,sig: (sig.__setitem__(r<buy,1), sig.__setitem__(r>sell,-1), sig)[-1])(rsi(df['close'],period), pd.Series(0,index=df.index)),
    lambda t: {'period':t.suggest_int('period',5,30),'buy':t.suggest_int('buy',15,40),'sell':t.suggest_int('sell',60,85)})

# BB
reg('BB', lambda df,period,std_mult: (lambda m,u,l,sig: (sig.__setitem__(df['close']<l,1), sig.__setitem__(df['close']>u,-1), sig)[-1])(*bb(df['close'],period,std_mult), pd.Series(0,index=df.index)),
    lambda t: {'period':t.suggest_int('period',8,50),'std_mult':t.suggest_float('std_mult',1.0,4.0,step=0.25)})

# EMA
def gen_ema(df, fast, slow):
    ef=ema(df['close'],fast); es=ema(df['close'],slow); sig=pd.Series(0,index=df.index)
    sig[(ef>es)&(ef.shift()<=es.shift())]=1; sig[(ef<es)&(ef.shift()>=es.shift())]=-1; return sig
reg('EMA', gen_ema, lambda t: {'fast':t.suggest_int('fast',3,30),'slow':t.suggest_int('slow',15,250)})

# MACD
def gen_macd(df, fast, slow, signal):
    _,_2,hist=macd_calc(df['close'],fast,slow,signal); sig=pd.Series(0,index=df.index)
    sig[(hist>0)&(hist.shift()<=0)]=1; sig[(hist<0)&(hist.shift()>=0)]=-1; return sig
reg('MACD', gen_macd, lambda t: {'fast':t.suggest_int('fast',4,18),'slow':t.suggest_int('slow',18,45),'signal':t.suggest_int('signal',4,18)})

# Stoch
def gen_stoch(df, k_period, buy, sell):
    k,d=stoch(df['high'],df['low'],df['close'],k_period,3); sig=pd.Series(0,index=df.index)
    sig[(k<buy)&(k>k.shift())]=1; sig[(k>sell)&(k<k.shift())]=-1; return sig
reg('Stoch', gen_stoch, lambda t: {'k_period':t.suggest_int('k_period',3,28),'buy':t.suggest_int('buy',8,35),'sell':t.suggest_int('sell',65,92)})

# Keltner
def gen_keltner(df, period, mult):
    e=ema(df['close'],period); a=atr(df['high'],df['low'],df['close'],period)
    u=e+mult*a; l=e-mult*a; sig=pd.Series(0,index=df.index)
    sig[(df['close']>u)&(df['close'].shift()<=u.shift())]=1
    sig[(df['close']<l)&(df['close'].shift()>=l.shift())]=-1; return sig
reg('Keltner', gen_keltner, lambda t: {'period':t.suggest_int('period',6,45),'mult':t.suggest_float('mult',0.5,3.5,step=0.25)})

# ZScore
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

# WilliamsR
def gen_williams(df, period, buy_level):
    hi=df['high'].rolling(period).max(); lo=df['low'].rolling(period).min()
    wr=-100*(hi-df['close'])/(hi-lo+1e-10); sell=-(100+buy_level); sig=pd.Series(0,index=df.index)
    sig[(wr<buy_level)&(wr>wr.shift())]=1; sig[(wr>sell)&(wr<wr.shift())]=-1; return sig
reg('WilliamsR', gen_williams, lambda t: {'period':t.suggest_int('period',5,30),'buy_level':t.suggest_int('buy_level',-95,-65)})

# Donchian
def gen_donchian(df, period):
    hi=df['high'].rolling(period).max(); lo=df['low'].rolling(period).min(); sig=pd.Series(0,index=df.index)
    sig[(df['close']>hi.shift())&(df['close'].shift()<=hi.shift(2))]=1
    sig[(df['close']<lo.shift())&(df['close'].shift()>=lo.shift(2))]=-1; return sig
reg('Donchian', gen_donchian, lambda t: {'period':t.suggest_int('period',5,80)})

# ADX
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
    sig[(h>h.shift())&(h.shift()<=h.shift(2))]=1; sig[(h<h.shift())&(h.shift()>=h.shift(2))]=-1; return sig
reg('HMA', gen_hma, lambda t: {'period':t.suggest_int('period',4,70)})

# RSI_MACD
def gen_rsi_macd(df, rsi_buy, rsi_sell):
    r=rsi(df['close'],14); _,_2,hist=macd_calc(df['close']); sig=pd.Series(0,index=df.index)
    sig[(r<rsi_buy)&(hist>0)&(hist.shift()<=0)]=1; sig[(r>rsi_sell)&(hist<0)&(hist.shift()>=0)]=-1; return sig
reg('RSI_MACD', gen_rsi_macd, lambda t: {'rsi_buy':t.suggest_int('rsi_buy',18,48),'rsi_sell':t.suggest_int('rsi_sell',52,82)})

# VWAP
def gen_vwap(df, dist_pct):
    tp=(df['high']+df['low']+df['close'])/3; cumvol=df['volume'].cumsum()
    vwap=((tp*df['volume']).cumsum())/(cumvol+1e-10); sig=pd.Series(0,index=df.index)
    sig[(df['close']<vwap*(1-dist_pct))&(df['close']>df['close'].shift())]=1
    sig[(df['close']>vwap*(1+dist_pct))&(df['close']<df['close'].shift())]=-1; return sig
reg('VWAP', gen_vwap, lambda t: {'dist_pct':t.suggest_float('dist_pct',0.002,0.06,step=0.001)})

# VolSpike
def gen_volspike(df, vol_mult):
    vm=df['volume'].rolling(20).mean(); spike=df['volume']>vol_mult*vm; sig=pd.Series(0,index=df.index)
    sig[spike&(df['close']>df['open'])]=1; sig[spike&(df['close']<df['open'])]=-1; return sig
reg('VolSpike', gen_volspike, lambda t: {'vol_mult':t.suggest_float('vol_mult',1.2,6.0,step=0.2)})

# Engulfing
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

# Squeeze
def gen_squeeze(df, bb_period, kc_mult):
    mid,bb_up,bb_lo=bb(df['close'],bb_period); e=ema(df['close'],bb_period)
    a=atr(df['high'],df['low'],df['close'],bb_period)
    kc_up=e+kc_mult*a; kc_lo=e-kc_mult*a; sq=(bb_lo>kc_lo)&(bb_up<kc_up)
    mom=df['close']-df['close'].rolling(bb_period).mean(); sig=pd.Series(0,index=df.index)
    sig[~sq&sq.shift().fillna(False)&(mom>0)]=1; sig[~sq&sq.shift().fillna(False)&(mom<0)]=-1; return sig
reg('Squeeze', gen_squeeze, lambda t: {'bb_period':t.suggest_int('bb_period',10,35),'kc_mult':t.suggest_float('kc_mult',1.0,2.5,step=0.25)})

# TripleEMA
def gen_triple_ema(df, fast, mid, slow):
    ef=ema(df['close'],fast); em=ema(df['close'],mid); es=ema(df['close'],slow)
    bull=(ef>em)&(em>es); bear=(ef<em)&(em<es); sig=pd.Series(0,index=df.index)
    sig[bull&~bull.shift().fillna(False)]=1; sig[bear&~bear.shift().fillna(False)]=-1; return sig
reg('TripleEMA', gen_triple_ema, lambda t: {'fast':t.suggest_int('fast',2,12),'mid':t.suggest_int('mid',8,30),'slow':t.suggest_int('slow',20,100)})

# Pivot
def gen_pivot(df, atr_filter):
    pp=(df['high'].shift()+df['low'].shift()+df['close'].shift())/3
    s1=2*pp-df['high'].shift(); r1=2*pp-df['low'].shift()
    sig=pd.Series(0,index=df.index)
    sig[(df['low']<=s1)&(df['close']>s1)]=1; sig[(df['high']>=r1)&(df['close']<r1)]=-1; return sig
reg('Pivot', gen_pivot, lambda t: {'atr_filter':t.suggest_float('atr_filter',0.5,2.0,step=0.1)})

# OBV
def gen_obv(df, period):
    obv=(np.sign(df['close'].diff())*df['volume']).cumsum()
    o_sma=obv.rolling(period).mean(); p_sma=df['close'].rolling(period).mean()
    sig=pd.Series(0,index=df.index); sig[(obv>o_sma)&(df['close']<p_sma)]=1
    sig[(obv<o_sma)&(df['close']>p_sma)]=-1; return sig
reg('OBV', gen_obv, lambda t: {'period':t.suggest_int('period',8,40)})

# Doji
def gen_doji(df, rsi_level):
    body=(df['close']-df['open']).abs(); wick=df['high']-df['low']
    doji=body<wick*0.1; r=rsi(df['close'],14); sig=pd.Series(0,index=df.index)
    sig[doji&(r<rsi_level)]=1; sig[doji&(r>(100-rsi_level))]=-1; return sig
reg('Doji', gen_doji, lambda t: {'rsi_level':t.suggest_int('rsi_level',20,45)})

# ATR breakout
def gen_atr_break(df, mult):
    a=atr(df['high'],df['low'],df['close'],14); move=df['close']-df['close'].shift()
    sig=pd.Series(0,index=df.index); sig[(move>mult*a)&(move.shift()<=mult*a.shift())]=1
    sig[(move<-mult*a)&(move.shift()>=-mult*a.shift())]=-1; return sig
reg('ATR', gen_atr_break, lambda t: {'mult':t.suggest_float('mult',0.8,3.0,step=0.2)})

print(f"Total strategy types: {len(STYPES)}")

# ─── WEIGHTED SCORE ───
def calc_score(yearly):
    years=sorted(yearly.keys(),reverse=True); n=len(years)
    if n==0: return 0
    wm={1:[100],2:[55,45],3:[40,30,30],4:[35,25,25,15],5:[30,25,20,15,10],6:[27,22,18,14,11,8],7:[25,20,16,13,10,9,7]}
    w=wm.get(min(n,7),wm[7])[:n]
    return round(sum(yearly[y]['wr']*w[i]/100 for i,y in enumerate(years[:len(w)])),1)

# ─── BUILD PROMISING COMBOS FROM SCREENING ───
def load_promising_combos():
    """Load all strategy_type × symbol × tf combos with WR >= 60% from screening"""
    files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    combos = {}  # (stype, symbol, tf) → screening_wr
    for f in files:
        try:
            for r in json.load(open(f)):
                if r.get('wr', 0) >= 50:
                    prefix = r['strategy'].split('_')[0]
                    if prefix in STYPES:
                        key = (prefix, r['symbol'], r['timeframe'])
                        if key not in combos or r['wr'] > combos[key]:
                            combos[key] = r['wr']
        except:
            pass
    return combos

# ─── MAIN ───
def main():
    combos = load_promising_combos()
    print(f"Promising combos: {len(combos):,}")

    # Group by symbol for efficient data loading
    by_symbol = {}
    for (stype, symbol, tf), wr in combos.items():
        by_symbol.setdefault(symbol, []).append((stype, tf, wr))

    symbols_sorted = sorted(by_symbol.keys(), key=lambda s: -len(by_symbol[s]))
    print(f"Unique symbols: {len(symbols_sorted)}")
    print(f"Optuna trials: {len(combos) * N_TRIALS:,}")
    print("=" * 70)

    # Load progress
    progress = {'completed_symbols': [], 'grails': [], 'stats': {}}
    if os.path.exists(PROGRESS_FILE):
        progress = json.load(open(PROGRESS_FILE))
    done_symbols = set(progress.get('completed_symbols', []))
    all_grails = progress.get('grails', [])

    remaining = [s for s in symbols_sorted if s not in done_symbols]
    print(f"Already done: {len(done_symbols)} | Remaining: {len(remaining)}")

    start = time.time()
    new_grails = 0

    for idx, symbol in enumerate(remaining):
        t0 = time.time()
        stypes_tfs = by_symbol[symbol]
        short = symbol[:25].ljust(25)

        # Load data once
        df_5m = load_candles(symbol, '5m')
        if df_5m is None or len(df_5m) < 200:
            done_symbols.add(symbol)
            progress['completed_symbols'] = list(done_symbols)
            continue

        dfs = {'5m': df_5m}
        dfs['15m'] = resample(df_5m, '15m')
        df_1h = load_candles(symbol, '1h')
        dfs['1h'] = df_1h if df_1h is not None and len(df_1h) > 100 else None
        dfs['4h'] = resample(df_5m, '4h')
        df_1d = load_candles(symbol, '1d')
        dfs['1d'] = df_1d if df_1d is not None and len(df_1d) > 30 else resample(df_5m, '1d')

        symbol_grails = []

        for stype, tf, screening_wr in stypes_tfs:
            df = dfs.get(tf)
            if df is None or len(df) < 200:
                continue

            split = int(len(df) * 0.7)
            train = df.iloc[:split].copy()
            test = df.iloc[split:].copy()
            if len(train) < 100 or len(test) < 50:
                continue

            info = STYPES[stype]
            gen = info['gen']
            space = info['space']

            # Optuna on TRAIN
            def objective(trial):
                params = space(trial)
                try:
                    sig = gen(train, **params)
                    r = backtest(train, sig)
                    if r is None: return -999
                    return r['wr'] * 0.7 + min(r['sharpe'], 50) * 0.3
                except:
                    return -999

            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=N_TRIALS, timeout=60)
            if study.best_value <= 0:
                continue

            bp = study.best_params

            # Validate
            try:
                test_sig = gen(test, **bp)
                test_r = backtest(test, test_sig)
                train_sig = gen(train, **bp)
                train_r = backtest(train, train_sig)
                full_sig = gen(df, **bp)
                full_r = backtest(df, full_sig)
            except:
                continue

            if test_r is None or train_r is None or full_r is None:
                continue

            # GRAIL CHECK: WR>70% test + PnL>0 test + enough trades
            if test_r['wr'] < 70 or test_r['pnl'] <= 0:
                continue
            if full_r['trades'] < MIN_TRADES_FULL:
                continue

            wr_diff = abs(train_r['wr'] - test_r['wr'])
            if wr_diff > 30:
                continue

            mae = test_r.get('mae_p95', 0.1)
            grail = {
                'strategy': stype, 'symbol': symbol, 'timeframe': tf,
                'best_params': bp,
                'train': {'wr': train_r['wr'], 'pnl': train_r['pnl'], 'sharpe': train_r['sharpe'], 'trades': train_r['trades'], 'pf': train_r['profit_factor']},
                'test': {'wr': test_r['wr'], 'pnl': test_r['pnl'], 'sharpe': test_r['sharpe'], 'trades': test_r['trades'], 'pf': test_r['profit_factor'], 'yearly': test_r.get('yearly',{}), 'mae_p95': mae, 'max_dd': test_r['max_drawdown']},
                'full': {'wr': full_r['wr'], 'pnl': full_r['pnl'], 'sharpe': full_r['sharpe'], 'trades': full_r['trades'], 'pf': full_r['profit_factor'], 'max_dd': full_r['max_drawdown']},
                'wr_diff': round(wr_diff, 1),
                'weighted_score': calc_score(test_r.get('yearly', {})),
                'safe_leverage': min(20, max(1, int(1 / (mae * 2.5 + 1e-10)))),
            }
            symbol_grails.append(grail)
            all_grails.append(grail)
            new_grails += 1

        done_symbols.add(symbol)
        dt = time.time() - t0
        elapsed = time.time() - start
        done_count = idx + 1
        rate = elapsed / done_count
        eta = rate * (len(remaining) - done_count) / 3600

        if symbol_grails:
            best = max(symbol_grails, key=lambda g: g['full']['wr'])
            print(f"[{done_count}/{len(remaining)}] {short} 🏆 {len(symbol_grails)} grails {dt:.0f}s | "
                  f"BEST: {best['strategy']}@{best['timeframe']} Full={best['full']['trades']}T WR={best['full']['wr']:.1f}% PnL={best['full']['pnl']:.1f}%")
        else:
            print(f"[{done_count}/{len(remaining)}] {short} — {len(stypes_tfs)} tested {dt:.0f}s")

        # Save
        progress['completed_symbols'] = list(done_symbols)
        progress['grails'] = all_grails
        progress['stats'] = {
            'total_grails': len(all_grails),
            'new_grails': new_grails,
            'symbols_done': len(done_symbols),
            'symbols_total': len(symbols_sorted),
            'elapsed_hours': round(elapsed/3600, 2),
            'eta_hours': round(eta, 1),
        }
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, default=str)
        if (done_count % 10 == 0) or symbol_grails:
            with open(GRAIL_FILE, 'w') as f:
                json.dump(all_grails, f, indent=2, default=str)

    # Final
    with open(GRAIL_FILE, 'w') as f:
        json.dump(all_grails, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"DONE! Grails: {len(all_grails)} | New: {new_grails}")
    print(f"Elapsed: {progress['stats']['elapsed_hours']:.1f}h")

if __name__ == '__main__':
    main()
