#!/usr/bin/env python3
"""
FAST R3 WORKER — Optimized for 411 strategies.
- Batch2 strategies (with params): Optuna 10 trials, 15s timeout
- Batch1-6 strategies (no params): Single backtest, no Optuna
Usage: python3 optuna_worker_r3_fast.py <worker_id> <total_workers>
"""
import sqlite3, pandas as pd, numpy as np, json, os, sys, time, warnings, inspect
from datetime import datetime
import filelock

warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

DB_PATH = "/Users/sabrina/Code/activos binace /activos_binance.db"
DB_PATH_OKX = "/Users/sabrina/Code/BOT/okx_bot/data/ohlcv_market.db"
PROJECT_DIR = "/Users/sabrina/CLAUDE CODE/Estrategias"
COMMISSION = 0.001
SLIPPAGE = 0.0005
MIN_TRADES = 5
MIN_TRADES_FULL = 15
N_TRIALS = 10
TIMEOUT_PER_STRAT = 15

WORKER_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
TOTAL_WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 1

PROGRESS_FILE = os.path.join(PROJECT_DIR, "data", "optuna_r3_progress.json")
LOCK_FILE = PROGRESS_FILE + ".lock"
USE_TFS = ['5m', '1h', '1d']

sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "strategies_code"))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─── Load strategies separated by type ───
def load_strategies():
    with_params = {}  # Optuna-able
    no_params = {}    # Direct backtest only

    # batch2_strategies.py — signal-returning functions WITH params
    try:
        import batch2_strategies as b2
        skip = {'ema', 'sma', 'rsi', 'atr', 'vwap', 'donchian_high', 'donchian_low',
                'linreg_slope', 'hurst_exponent', 'np', 'pd', 'stats'}
        for name in dir(b2):
            obj = getattr(b2, name)
            if callable(obj) and not name.startswith('_') and name not in skip:
                sig = inspect.signature(obj)
                params = [(p.name, p.default) for p in sig.parameters.values() if p.name != 'df']

                def make_gen(f):
                    def gen(df, **kw):
                        try: return f(df, **kw)
                        except: return None
                    return gen

                def make_space(param_list):
                    def space(trial):
                        p = {}
                        for pname, default in param_list:
                            if default is inspect.Parameter.empty:
                                default = 14
                            if isinstance(default, int):
                                p[pname] = trial.suggest_int(pname, max(2, default//3), default*3)
                            elif isinstance(default, float):
                                p[pname] = trial.suggest_float(pname, max(0.01, default/3), default*3)
                        return p
                    return space

                if params:  # Has tunable params
                    with_params[name] = {'gen': make_gen(obj), 'space': make_space(params)}
                else:
                    no_params[name] = {'gen': make_gen(obj)}
        print(f"BATCH2: {len(with_params)} with params, {len(no_params)} without")
    except Exception as e:
        print(f"[W{WORKER_ID}] batch2 error: {e}")

    # batch1-7: monkey-patch simulate to capture signals (NO params)
    for batch_num in range(1, 8):
        try:
            mod = __import__(f"strategies_batch{batch_num}")
            dict_name = f"BATCH{batch_num}_STRATEGIES"
            batch_dict = getattr(mod, dict_name, None)
            if not batch_dict or not isinstance(batch_dict, dict):
                continue

            count = 0
            for sname, func in batch_dict.items():
                key = f"B{batch_num}_{sname}"
                if key in no_params or key in with_params:
                    continue

                def make_gen(f, m):
                    def gen(df, **kw):
                        try:
                            orig = m.simulate
                            captured = {}
                            def cap(d, signals, max_hold=None):
                                captured['s'] = signals
                                return [], [1.0]
                            m.simulate = cap
                            try: f(df)
                            finally: m.simulate = orig
                            s = captured.get('s')
                            if s is not None:
                                return s if isinstance(s, pd.Series) else pd.Series(s, index=df.index[:len(s)])
                        except:
                            pass
                        return None
                    return gen

                no_params[key] = {'gen': make_gen(func, mod)}
                count += 1
            if count:
                print(f"BATCH{batch_num}: {count} strategies (no params)")
        except Exception as e:
            print(f"[W{WORKER_ID}] batch{batch_num} error: {e}")

    return with_params, no_params


# ─── DB ───
MAX_ROWS = {'5m': 50000, '1h': 8000, '1d': 2000}  # ~6mo 5m, ~1yr 1h, ~5yr 1d

def load_candles(symbol, tf="5m"):
    limit = MAX_ROWS.get(tf, 50000)
    # Try main DB first, fallback to OKX DB
    for db_path in [DB_PATH, DB_PATH_OKX]:
        if not os.path.exists(db_path):
            continue
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(
            f"SELECT ts,open,high,low,close,volume FROM candles WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT {limit}",
            conn, params=(symbol, tf))
        conn.close()
        if len(df) > 0:
            df = df.sort_values('ts')
            break
    else:
        return None
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
    return df.resample(r).agg({
        'ts':'first','open':'first','high':'max','low':'min','close':'last','volume':'sum'
    }).dropna(subset=['close'])

def get_symbols():
    all_syms = set()
    for db_path in [DB_PATH, DB_PATH_OKX]:
        if not os.path.exists(db_path):
            continue
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(
            "SELECT symbol, COUNT(*) as cnt FROM candles WHERE timeframe='5m' GROUP BY symbol ORDER BY cnt DESC", conn)
        conn.close()
        all_syms.update(df['symbol'].tolist())
    return sorted(all_syms)


# ─── BACKTEST (vectorized where possible) ───
def backtest(df, signals):
    if signals is None: return None
    if len(signals) > len(df):
        signals = signals.iloc[:len(df)]
    elif len(signals) < len(df):
        return None
    sig = np.asarray(signals, dtype=np.float64)
    opens = df['open'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    n = len(df)
    # Use numba-style loop but in pure numpy where possible
    trades = []; pos = 0; ep = 0.; ei = 0; mae = 0.
    for i in range(1, n):
        s = sig[i-1]; p = opens[i]
        if pos == 0 and s == 1:
            ep = p*(1+SLIPPAGE+COMMISSION); ei = i; mae = 0.; pos = 1
        elif pos == 1:
            lp = (lows[i]-ep)/ep
            if lp < mae: mae = lp
            if s == -1:
                xp = p*(1-SLIPPAGE-COMMISSION); pnl = (xp-ep)/ep*100
                yr = df.index[ei].year if hasattr(df.index[ei],'year') else 2025
                trades.append((pnl, mae, i-ei, yr)); pos = 0
    if len(trades) < MIN_TRADES: return None
    pnls = np.array([t[0] for t in trades]); maes = np.array([t[1] for t in trades])
    years = np.array([t[3] for t in trades])
    wins = (pnls>0).sum(); total = len(pnls); wr = wins/total*100
    wm = pnls>0
    avg_w = pnls[wm].mean() if wm.any() else 0
    avg_l = pnls[~wm].mean() if (~wm).any() else 0
    losses_n = (~wm).sum()
    pf = abs(avg_w*wins/(avg_l*losses_n+1e-10)) if losses_n > 0 else 999
    cum = pnls.cumsum(); peak = np.maximum.accumulate(cum); dd = (peak-cum).max()
    mae95 = abs(np.percentile(maes, 95)) if len(maes) > 0 else 0.1
    sharpe = (pnls.mean()/(pnls.std()+1e-10))*np.sqrt(252)
    yearly = {}
    for y in np.unique(years):
        m = years==y; yp = pnls[m]; yw = (yp>0).sum()
        yearly[str(y)] = {'wr':round(yw/len(yp)*100,1),'trades':int(len(yp)),'pnl':round(yp.sum(),2)}
    return {'trades':int(total),'wr':round(wr,1),'pnl':round(pnls.sum(),2),'profit_factor':round(pf,2),
            'max_drawdown':round(dd,2),'mae_p95':round(mae95,4),'sharpe':round(sharpe,2),'yearly':yearly}

def calc_score(yearly):
    years = sorted(yearly.keys(), reverse=True); n = len(years)
    if n == 0: return 0
    wm = {1:[100],2:[55,45],3:[40,30,30],4:[35,25,25,15],5:[30,25,20,15,10],6:[27,22,18,14,11,8],7:[25,20,16,13,10,9,7]}
    w = wm.get(min(n,7), wm[7])[:n]
    return round(sum(yearly[y]['wr']*w[i]/100 for i,y in enumerate(years[:len(w)])),1)


# ─── PROGRESS ───
def load_progress():
    try:
        lock = filelock.FileLock(LOCK_FILE, timeout=10)
        with lock:
            if os.path.exists(PROGRESS_FILE):
                return json.load(open(PROGRESS_FILE))
    except:
        if os.path.exists(PROGRESS_FILE):
            return json.load(open(PROGRESS_FILE))
    return {'completed': [], 'grails': [], 'stats': {}}

def save_progress(done_set, all_grails, stats):
    try:
        lock = filelock.FileLock(LOCK_FILE, timeout=10)
        with lock:
            existing = {'completed': [], 'grails': []}
            if os.path.exists(PROGRESS_FILE):
                existing = json.load(open(PROGRESS_FILE))
            merged_done = set(existing.get('completed', [])) | done_set
            seen = set()
            merged_grails = []
            for g in existing.get('grails', []) + all_grails:
                key = (g['strategy'], g['symbol'], g['timeframe'])
                if key not in seen:
                    seen.add(key)
                    merged_grails.append(g)
            progress = {
                'completed': list(merged_done),
                'grails': merged_grails,
                'stats': {**stats, 'done': len(merged_done), 'grails': len(merged_grails)},
            }
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(progress, f, default=str)
    except Exception as e:
        print(f"[W{WORKER_ID}] Save error: {e}")


def make_grail(sname, symbol, tf, bp, train_r, test_r, full_r, wr_diff):
    mae = test_r.get('mae_p95', 0.1)
    return {
        'strategy': sname, 'symbol': symbol, 'timeframe': tf,
        'best_params': bp,
        'train': {'wr':train_r['wr'],'pnl':train_r['pnl'],'sharpe':train_r['sharpe'],'trades':train_r['trades'],'pf':train_r['profit_factor']},
        'test': {'wr':test_r['wr'],'pnl':test_r['pnl'],'sharpe':test_r['sharpe'],'trades':test_r['trades'],'pf':test_r['profit_factor'],'yearly':test_r.get('yearly',{}),'mae_p95':mae,'max_dd':test_r['max_drawdown']},
        'full': {'wr':full_r['wr'],'pnl':full_r['pnl'],'sharpe':full_r['sharpe'],'trades':full_r['trades'],'pf':full_r['profit_factor'],'max_dd':full_r['max_drawdown']},
        'wr_diff': round(wr_diff, 1),
        'weighted_score': calc_score(test_r.get('yearly', {})),
        'safe_leverage': min(20, int(1/(mae*2.5+1e-10))),
        'round': 'r3',
    }


def validate_walkforward(gen, df, train, test, bp, sname, symbol, tf):
    """Run walk-forward validation. Returns grail dict or None."""
    try:
        test_sig = gen(test, **bp)
        test_r = backtest(test, test_sig)
        if not test_r or test_r['wr'] < 70 or test_r['pnl'] <= 0:
            return None

        train_sig = gen(train, **bp)
        train_r = backtest(train, train_sig)
        if not train_r:
            return None

        full_sig = gen(df, **bp)
        full_r = backtest(df, full_sig)
        if not full_r or full_r['trades'] < MIN_TRADES_FULL:
            return None

        wr_diff = abs(train_r['wr'] - test_r['wr'])
        if wr_diff > 30:
            return None

        return make_grail(sname, symbol, tf, bp, train_r, test_r, full_r, wr_diff)
    except:
        return None


# ─── MAIN ───
def main():
    with_params, no_params = load_strategies()
    total_strats = len(with_params) + len(no_params)
    print(f"[W{WORKER_ID}/{TOTAL_WORKERS}] R3 FAST | {len(with_params)} with params + {len(no_params)} no params = {total_strats}")

    all_symbols = get_symbols()
    progress = load_progress()
    done_set = set(progress.get('completed', []))

    my_symbols = [s for i, s in enumerate(all_symbols) if i % TOTAL_WORKERS == WORKER_ID]
    remaining = [s for s in my_symbols if s not in done_set]
    print(f"[W{WORKER_ID}] Symbols: {len(my_symbols)} total, {len(remaining)} remaining")

    start = time.time()
    my_grails = []

    for idx, symbol in enumerate(remaining):
        t0 = time.time()

        df_5m = load_candles(symbol, '5m')
        if df_5m is None or len(df_5m) < 500:
            done_set.add(symbol)
            continue

        dfs = {'5m': df_5m}
        df_1h = load_candles(symbol, '1h')
        dfs['1h'] = df_1h if df_1h is not None and len(df_1h) > 100 else None
        df_1d = load_candles(symbol, '1d')
        dfs['1d'] = df_1d if df_1d is not None and len(df_1d) > 30 else resample(df_5m, '1d')

        sym_grails = []

        for tf, df in dfs.items():
            if df is None or len(df) < 200:
                continue
            split = int(len(df) * 0.7)
            train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
            if len(train) < 100 or len(test) < 50:
                continue

            # Phase 1: No-params strategies (FAST — single backtest each)
            for sname, info in no_params.items():
                gen = info['gen']
                try:
                    sig = gen(train)
                    r = backtest(train, sig)
                    if r is None or r['wr'] < 50:
                        continue
                    grail = validate_walkforward(gen, df, train, test, {}, sname, symbol, tf)
                    if grail:
                        sym_grails.append(grail)
                        my_grails.append(grail)
                except:
                    continue

            # Phase 2: With-params strategies (Optuna)
            for sname, info in with_params.items():
                gen, space = info['gen'], info['space']
                try:
                    # Quick check first
                    try:
                        quick_sig = gen(train)
                        quick_r = backtest(train, quick_sig)
                        if quick_r and quick_r['wr'] < 35:
                            continue
                    except:
                        pass

                    study = optuna.create_study(direction='maximize',
                                                sampler=optuna.samplers.TPESampler(seed=42))

                    def obj(trial, _g=gen, _s=space, _t=train):
                        params = _s(trial)
                        try:
                            sig = _g(_t, **params)
                            r = backtest(_t, sig)
                            if r is None: return -999
                            return r['wr']*0.7 + min(r['sharpe'],50)*0.3
                        except:
                            return -999

                    study.optimize(obj, n_trials=N_TRIALS, timeout=TIMEOUT_PER_STRAT)
                    if study.best_value <= 0:
                        continue
                    bp = study.best_params

                    grail = validate_walkforward(gen, df, train, test, bp, sname, symbol, tf)
                    if grail:
                        sym_grails.append(grail)
                        my_grails.append(grail)
                except:
                    continue

        done_set.add(symbol)
        dt = time.time() - t0
        elapsed = time.time() - start
        rate = elapsed / (idx + 1)
        eta = rate * (len(remaining) - idx - 1) / 3600

        if sym_grails:
            best = max(sym_grails, key=lambda g: g['full']['wr'])
            print(f"[W{WORKER_ID}][{idx+1}/{len(remaining)}] {symbol[:25]:25s} "
                  f"🏆 {len(sym_grails)} grails | {best['strategy']}@{best['timeframe']} "
                  f"WR={best['full']['wr']:.1f}% | {dt:.0f}s ETA={eta:.1f}h")
        else:
            if (idx + 1) % 5 == 0 or dt > 60:
                print(f"[W{WORKER_ID}][{idx+1}/{len(remaining)}] {symbol[:25]:25s} "
                      f"0 grails | {dt:.0f}s ETA={eta:.1f}h | total={len(my_grails)}")

        # Save every 2 symbols
        if (idx + 1) % 2 == 0 or sym_grails:
            save_progress(done_set, my_grails, {
                'worker': WORKER_ID, 'elapsed_h': round(elapsed/3600, 2), 'eta_h': round(eta, 1),
            })

    save_progress(done_set, my_grails, {
        'worker': WORKER_ID, 'finished': True,
        'elapsed_h': round((time.time()-start)/3600, 2),
    })
    print(f"\n[W{WORKER_ID}] DONE! My grails: {len(my_grails)}")


if __name__ == '__main__':
    main()
