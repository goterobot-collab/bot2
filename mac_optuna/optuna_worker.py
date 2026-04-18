#!/usr/bin/env python3
"""
OPTUNA WORKER — Single-process worker for R2/R3.
Launch 8 instances each processing 1/8 of symbols.
Usage: python3 optuna_worker.py <round> <worker_id> <total_workers>
Example: python3 optuna_worker.py r2 0 8
"""
import sqlite3, pandas as pd, numpy as np, json, os, sys, time, warnings
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

ROUND = sys.argv[1] if len(sys.argv) > 1 else 'r2'
WORKER_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 0
TOTAL_WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
TOP_N_SYMBOLS = int(sys.argv[4]) if len(sys.argv) > 4 else 0  # 0 = all

# R3 is 8x more strategies → need faster settings
if ROUND == 'r4_batch':
    N_TRIALS = 20          # More trials than R3 — batch strategies now have real search spaces
    TIMEOUT_PER_STRAT = 30
    USE_TFS = ['5m', '1h', '1d']
elif ROUND == 'r3':
    N_TRIALS = 10
    TIMEOUT_PER_STRAT = 15
    USE_TFS = ['5m', '1h', '1d']  # skip 15m, 4h (resampled, slow)
elif ROUND == 'pine':
    N_TRIALS = 25           # Full optimization for PINE strategies
    TIMEOUT_PER_STRAT = 45
    USE_TFS = ['5m', '15m', '1h', '4h']
else:
    N_TRIALS = 25
    TIMEOUT_PER_STRAT = 45
    USE_TFS = ['5m', '15m', '1h', '4h', '1d']

PROGRESS_FILE = os.path.join(PROJECT_DIR, "data", f"optuna_{ROUND}_progress.json")
LOCK_FILE = PROGRESS_FILE + ".lock"

sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "strategies_code"))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─── Load strategies ───
def load_strategies():
    if ROUND == 'r2':
        from strategies_round2 import STRATEGY_TYPES_R2
        return STRATEGY_TYPES_R2
    elif ROUND == 'r4_batch':
        # R4: ONLY batch strategies with wrappers + search spaces (50 strategies)
        strats = {}
        try:
            from batch_search_spaces import BATCH_SEARCH_SPACES, make_space_func
            from strategies_code.batch_param_wrappers import BATCH_WRAPPERS
            for key, wrapper_fn in BATCH_WRAPPERS.items():
                if key in BATCH_SEARCH_SPACES:
                    strats[key] = {'gen': wrapper_fn, 'space': make_space_func(key)}
            print(f"[W{WORKER_ID}] R4_BATCH: Loaded {len(strats)} batch strategies with Optuna search spaces")
        except Exception as e:
            print(f"[W{WORKER_ID}] R4_BATCH load error: {e}")
        return strats
    elif ROUND == 'pine':
        # PINE: strategies from strategies_new_winners.py (11 Pine→Python conversions)
        strats = {}
        try:
            _pine_path = '/Users/sabrina/CLAUDE CODE/BOT V7/strategies'
            if _pine_path not in sys.path:
                sys.path.insert(0, _pine_path)
            from strategies_new_winners import NEW_WINNER_STRATS
            strats.update(NEW_WINNER_STRATS)
            print(f"[W{WORKER_ID}] PINE: Loaded {len(strats)} Pine→Python strategies")
        except Exception as e:
            print(f"[W{WORKER_ID}] PINE load error: {e}")
        return strats
    else:
        return _load_r3()

def _load_r3():
    import inspect
    strats = {}

    # batch2_strategies.py — signal-returning functions
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

                strats[name] = {'gen': make_gen(obj), 'space': make_space(params)}
    except Exception as e:
        print(f"[W{WORKER_ID}] batch2 error: {e}")

    # batch1-6: monkey-patch simulate to capture signals
    for batch_num in range(1, 7):
        try:
            mod = __import__(f"strategies_batch{batch_num}")
            dict_name = f"BATCH{batch_num}_STRATEGIES"
            batch_dict = getattr(mod, dict_name, None)
            if not batch_dict or not isinstance(batch_dict, dict):
                continue

            for sname, func in batch_dict.items():
                key = f"B{batch_num}_{sname}"
                if key in strats:
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

                strats[key] = {'gen': make_gen(func, mod), 'space': lambda trial: {}}
        except Exception as e:
            print(f"[W{WORKER_ID}] batch{batch_num} error: {e}")

    # ─── Upgrade batch strategies with external search spaces + wrappers ───
    # For strategies that have BATCH_SEARCH_SPACES definitions AND wrapper
    # functions, replace the monkey-patched generator (which ignores **kw)
    # with the clean wrapper that accepts Optuna params directly.
    n_upgraded = 0
    try:
        from batch_search_spaces import BATCH_SEARCH_SPACES, make_space_func
        from strategies_code.batch_param_wrappers import BATCH_WRAPPERS

        for key, wrapper_fn in BATCH_WRAPPERS.items():
            if key in strats and key in BATCH_SEARCH_SPACES:
                strats[key]['gen'] = wrapper_fn   # Clean wrapper: (df, **params) -> signals
                strats[key]['space'] = make_space_func(key)  # Optuna search space
                n_upgraded += 1

        # For strategies WITH search space but WITHOUT wrapper yet,
        # at least give them the search space (monkey-patch gen still works
        # for default params, Optuna trials will use default since gen ignores **kw)
        for key in BATCH_SEARCH_SPACES:
            if key in strats and key not in BATCH_WRAPPERS:
                strats[key]['space'] = make_space_func(key)

        print(f"[W{WORKER_ID}] Batch upgrade: {n_upgraded} wrappers, "
              f"{len(BATCH_SEARCH_SPACES)} search spaces applied")
    except ImportError as e:
        print(f"[W{WORKER_ID}] Batch search spaces not available: {e}")
    except Exception as e:
        print(f"[W{WORKER_ID}] Batch upgrade error: {e}")

    return strats


# ─── DB ───
def load_candles(symbol, tf="5m"):
    for db_path in [DB_PATH, DB_PATH_OKX]:
        if not os.path.exists(db_path):
            continue
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(
            "SELECT ts,open,high,low,close,volume FROM candles WHERE symbol=? AND timeframe=? ORDER BY ts",
            conn, params=(symbol, tf))
        conn.close()
        if len(df) > 0:
            break
    else:
        return None
    if len(df) == 0: return None
    try:
        df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
        df.dropna(subset=['ts'], inplace=True)
        if len(df) == 0: return None
        df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('datetime', inplace=True)
    except (ValueError, OverflowError) as e:
        print(f"[W{WORKER_ID}] ⚠️ Corrupt data for {symbol}/{tf}: {e}")
        return None
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
    all_syms = []
    for db_path in [DB_PATH, DB_PATH_OKX]:
        if not os.path.exists(db_path):
            continue
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(
            "SELECT symbol FROM candles WHERE timeframe='5m' GROUP BY symbol ORDER BY COUNT(*) DESC", conn)
        conn.close()
        all_syms.extend(df['symbol'].tolist())
    # Deduplicate preserving order (most data first)
    seen = set()
    unique = []
    for s in all_syms:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    # Apply top-N filter if specified
    if TOP_N_SYMBOLS > 0:
        unique = unique[:TOP_N_SYMBOLS]
    return unique


# ─── BACKTEST ───
def backtest(df, signals):
    if signals is None: return None
    sig = signals.values; opens = df['open'].values; lows = df['low'].values; n = len(df)
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


# ─── THREAD-SAFE PROGRESS ───
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
            # Merge with existing progress (other workers may have updated)
            existing = {'completed': [], 'grails': []}
            if os.path.exists(PROGRESS_FILE):
                existing = json.load(open(PROGRESS_FILE))
            merged_done = set(existing.get('completed', [])) | done_set
            # Deduplicate grails by (strategy, symbol, timeframe)
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


# ─── MAIN WORKER ───
def main():
    strategies = load_strategies()
    all_symbols = get_symbols()
    progress = load_progress()
    done_set = set(progress.get('completed', []))

    # Split symbols across workers
    my_symbols = [s for i, s in enumerate(all_symbols) if i % TOTAL_WORKERS == WORKER_ID]
    remaining = [s for s in my_symbols if s not in done_set]

    print(f"[W{WORKER_ID}/{TOTAL_WORKERS}] {ROUND.upper()} | Strategies: {len(strategies)} | "
          f"My symbols: {len(my_symbols)} | Remaining: {len(remaining)}")

    start = time.time()
    my_grails = []

    for idx, symbol in enumerate(remaining):
        t0 = time.time()

        df_5m = load_candles(symbol, '5m')
        if df_5m is None or len(df_5m) < 500:
            done_set.add(symbol)
            continue

        dfs = {'5m': df_5m}
        if '15m' in USE_TFS:
            dfs['15m'] = resample(df_5m, '15m')
        if '4h' in USE_TFS:
            dfs['4h'] = resample(df_5m, '4h')
        if '1h' in USE_TFS:
            df_1h = load_candles(symbol, '1h')
            dfs['1h'] = df_1h if df_1h is not None and len(df_1h) > 100 else None
        if '1d' in USE_TFS:
            df_1d = load_candles(symbol, '1d')
            dfs['1d'] = df_1d if df_1d is not None and len(df_1d) > 30 else resample(df_5m, '1d')

        sym_grails = []
        for tf, df in dfs.items():
            if df is None or len(df) < 200: continue
            # Temporal split: test = 2024-2026 (out-of-sample reciente)
            # Fallback a 70/30 si no hay suficientes datos en alguna partición
            cutoff = pd.Timestamp('2024-01-01')
            train_tmp = df[df.index < cutoff].copy()
            test_tmp  = df[df.index >= cutoff].copy()
            if len(train_tmp) >= 200 and len(test_tmp) >= 50:
                train, test = train_tmp, test_tmp
            else:
                split = int(len(df) * 0.7)
                train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
            if len(train) < 100 or len(test) < 50: continue

            for sname, info in strategies.items():
                gen, space = info['gen'], info['space']
                try:
                    study = optuna.create_study(direction='maximize',
                                                sampler=optuna.samplers.TPESampler(seed=42))

                    def obj(trial, _g=gen, _s=space, _t=train):
                        params = _s(trial)
                        try:
                            sig = _g(_t, **params)
                            r = backtest(_t, sig)
                            if r is None: return -999
                            return r['wr']*0.7 + min(r['sharpe'],50)*0.3
                        except: return -999

                    # Check if has params
                    has_params = False
                    try:
                        tp = space(optuna.trial.FixedTrial({}))
                        has_params = bool(tp)
                    except:
                        has_params = True

                    if has_params:
                        # Quick check: default params on train first
                        try:
                            quick_sig = gen(train)
                            quick_r = backtest(train, quick_sig)
                            if quick_r and quick_r['wr'] < 35:
                                continue  # hopeless, skip Optuna
                        except:
                            pass
                        study.optimize(obj, n_trials=N_TRIALS, timeout=TIMEOUT_PER_STRAT)
                        if study.best_value <= 0: continue
                        bp = study.best_params
                    else:
                        bp = {}
                        sig_c = gen(train, **bp)
                        r_c = backtest(train, sig_c)
                        if r_c is None or r_c['wr'] < 50: continue

                    # Walk-forward validation
                    test_sig = gen(test, **bp); test_r = backtest(test, test_sig)
                    train_sig = gen(train, **bp); train_r = backtest(train, train_sig)
                    full_sig = gen(df, **bp); full_r = backtest(df, full_sig)
                    if not test_r or not train_r or not full_r: continue
                    if test_r['wr'] < 70 or test_r['pnl'] <= 0: continue
                    if full_r['trades'] < MIN_TRADES_FULL: continue
                    wr_diff = abs(train_r['wr'] - test_r['wr'])
                    if wr_diff > 30: continue

                    mae = test_r.get('mae_p95', 0.1)
                    grail = {
                        'strategy': sname, 'symbol': symbol, 'timeframe': tf,
                        'best_params': bp,
                        'train': {'wr':train_r['wr'],'pnl':train_r['pnl'],'sharpe':train_r['sharpe'],'trades':train_r['trades'],'pf':train_r['profit_factor']},
                        'test': {'wr':test_r['wr'],'pnl':test_r['pnl'],'sharpe':test_r['sharpe'],'trades':test_r['trades'],'pf':test_r['profit_factor'],'yearly':test_r.get('yearly',{}),'mae_p95':mae,'max_dd':test_r['max_drawdown']},
                        'full': {'wr':full_r['wr'],'pnl':full_r['pnl'],'sharpe':full_r['sharpe'],'trades':full_r['trades'],'pf':full_r['profit_factor'],'max_dd':full_r['max_drawdown']},
                        'wr_diff': round(wr_diff,1),
                        'weighted_score': calc_score(test_r.get('yearly',{})),
                        'safe_leverage': min(20, int(1/(mae*2.5+1e-10))),
                        'round': ROUND,
                    }
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
        elif (idx + 1) % 10 == 0:
            print(f"[W{WORKER_ID}][{idx+1}/{len(remaining)}] grails={len(my_grails)} ETA={eta:.1f}h")

        # Save every 3 symbols
        if (idx + 1) % 3 == 0 or sym_grails:
            save_progress(done_set, my_grails, {
                'worker': WORKER_ID, 'elapsed_h': round(elapsed/3600, 2), 'eta_h': round(eta, 1),
            })

    # Final save
    save_progress(done_set, my_grails, {
        'worker': WORKER_ID, 'finished': True,
        'elapsed_h': round((time.time()-start)/3600, 2),
    })
    print(f"\n[W{WORKER_ID}] DONE! My grails: {len(my_grails)}")


if __name__ == '__main__':
    main()
