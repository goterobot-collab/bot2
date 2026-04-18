#!/usr/bin/env python3
"""
OPTUNA TURBO — R2 (52 types) + R3 (300+ types from bot_engine)
Optimizado para Mac M5: 10 cores, 24GB RAM
8 workers paralelos, progreso reanudable, griales consolidados.
"""
import sqlite3, pandas as pd, numpy as np, json, os, sys, time, warnings, traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass  # already set
warnings.filterwarnings('ignore')

# ─── CONFIG ───
DB_PATH = "/Users/sabrina/Code/activos binace /activos_binance.db"
PROJECT_DIR = "/Users/sabrina/CLAUDE CODE/Estrategias"
N_TRIALS = 25
COMMISSION = 0.001
SLIPPAGE = 0.0005
MIN_TRADES = 5
MIN_TRADES_FULL = 15
MAX_WORKERS = 8  # 10 cores - 2 for OS/dashboard
TIMEOUT_PER_STRAT = 45  # seconds per optuna study

# Which round to run? Set via CLI: python3 optuna_turbo.py r2|r3|both
ROUND = sys.argv[1] if len(sys.argv) > 1 else 'both'

os.makedirs(os.path.join(PROJECT_DIR, "data"), exist_ok=True)
sys.path.insert(0, PROJECT_DIR)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── LOAD STRATEGIES ───
def load_r2_strategies():
    """52 strategies from strategies_round2.py"""
    from strategies_round2 import STRATEGY_TYPES_R2
    return STRATEGY_TYPES_R2

def load_r3_strategies():
    """300+ strategies from bot_engine batch files, adapted to Optuna interface.
    Returns dict: {name: {'gen': func(df, **params)->signals, 'space': func(trial)->params}}
    """
    strats = {}
    sys.path.insert(0, os.path.join(PROJECT_DIR, "strategies_code"))

    # batch2_strategies.py: functions that return signals directly
    try:
        import batch2_strategies as b2
        # Find all strategy functions (they take df + params, return signals)
        for name in dir(b2):
            obj = getattr(b2, name)
            if callable(obj) and not name.startswith('_') and name not in (
                'ema', 'sma', 'rsi', 'atr', 'vwap', 'donchian_high', 'donchian_low',
                'linreg_slope', 'hurst_exponent', 'np', 'pd', 'stats'
            ):
                # Create a wrapper that adds Optuna param space
                strats[name] = _make_signal_strategy(name, obj)
    except Exception as e:
        print(f"[WARN] batch2_strategies: {e}")

    # strategies_batch1-6: functions that call simulate() internally
    # We need to extract just the signal generation part
    for batch_num in range(1, 7):
        try:
            mod_name = f"strategies_batch{batch_num}"
            mod = __import__(mod_name)

            # Get the batch dict
            dict_name = f"BATCH{batch_num}_STRATEGIES"
            batch_dict = getattr(mod, dict_name, None)
            if batch_dict and isinstance(batch_dict, dict):
                for sname, func in batch_dict.items():
                    if callable(func) and sname not in strats:
                        strats[f"B{batch_num}_{sname}"] = _make_batch_strategy(sname, func, mod)
            else:
                # Fallback: find all s_* functions
                for name in dir(mod):
                    if name.startswith('s_') and callable(getattr(mod, name)):
                        func = getattr(mod, name)
                        key = f"B{batch_num}_{name}"
                        if key not in strats:
                            strats[key] = _make_batch_strategy(name, func, mod)
        except Exception as e:
            print(f"[WARN] batch{batch_num}: {e}")

    return strats

def _make_signal_strategy(name, func):
    """Wrap a signal-returning function for Optuna.
    These functions already return pd.Series of signals.
    We auto-detect their parameters and create a generic space."""
    import inspect
    sig = inspect.signature(func)
    params = [p for p in sig.parameters if p != 'df']

    def gen(df, **kwargs):
        try:
            return func(df, **{k: v for k, v in kwargs.items() if k in [p.name for p in sig.parameters.values()]})
        except:
            return None

    def space(trial):
        p = {}
        for pname in params:
            param = sig.parameters[pname]
            default = param.default if param.default != inspect.Parameter.empty else 14
            if isinstance(default, int):
                lo, hi = max(2, default // 3), default * 3
                p[pname] = trial.suggest_int(pname, lo, hi)
            elif isinstance(default, float):
                lo, hi = max(0.01, default / 3), default * 3
                p[pname] = trial.suggest_float(pname, lo, hi)
            # skip non-numeric params
        return p

    return {'gen': gen, 'space': space}

def _make_batch_strategy(name, func, mod):
    """Wrap a batch strategy (returns trades,equity) to extract signals.
    We run it and reconstruct the signal series from the trades."""

    def gen(df, **kwargs):
        """Run the batch strategy but intercept signals before simulate()."""
        try:
            # Most batch strategies: compute indicators → create signal series → simulate(df, sig)
            # We monkey-patch simulate to capture the signals
            original_simulate = getattr(mod, 'simulate', None)
            captured = {}

            def capture_simulate(df_in, signals, max_hold=None):
                captured['signals'] = signals
                return [], [1.0]  # dummy return

            if original_simulate:
                mod.simulate = capture_simulate
                try:
                    func(df)
                finally:
                    mod.simulate = original_simulate

                if 'signals' in captured:
                    sig = captured['signals']
                    if isinstance(sig, pd.Series):
                        return sig
                    return pd.Series(sig, index=df.index[:len(sig)])
            return None
        except:
            return None

    def space(trial):
        # Batch strategies don't have Optuna params — they use fixed params
        # We add a generic "sensitivity" modifier
        return {}

    return {'gen': gen, 'space': space}


# ─── DB ───
def load_candles(symbol, tf="5m"):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT ts,open,high,low,close,volume FROM candles WHERE symbol=? AND timeframe=? ORDER BY ts",
        conn, params=(symbol, tf)
    )
    conn.close()
    if len(df) == 0:
        return None
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    return df

def resample(df, target):
    m = {'15m': '15min', '4h': '4h', '1d': '1D'}
    r = m.get(target)
    if not r:
        return None
    return df.resample(r).agg({
        'ts': 'first', 'open': 'first', 'high': 'max',
        'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna(subset=['close'])

def get_symbols():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT symbol, COUNT(*) as bars FROM candles WHERE timeframe='5m' GROUP BY symbol ORDER BY bars DESC",
        conn
    )
    conn.close()
    return list(df['symbol'])


# ─── BACKTEST ───
def backtest(df, signals):
    if signals is None:
        return None
    sig = signals.values
    opens = df['open'].values
    lows = df['low'].values
    n = len(df)
    trades = []
    pos = 0
    ep = 0.0
    ei = 0
    mae = 0.0
    for i in range(1, n):
        s = sig[i - 1]
        p = opens[i]
        if pos == 0 and s == 1:
            ep = p * (1 + SLIPPAGE + COMMISSION)
            ei = i
            mae = 0.0
            pos = 1
        elif pos == 1:
            lp = (lows[i] - ep) / ep
            if lp < mae:
                mae = lp
            if s == -1:
                xp = p * (1 - SLIPPAGE - COMMISSION)
                pnl = (xp - ep) / ep * 100
                yr = df.index[ei].year if hasattr(df.index[ei], 'year') else 2025
                trades.append((pnl, mae, i - ei, yr))
                pos = 0
    if len(trades) < MIN_TRADES:
        return None
    pnls = np.array([t[0] for t in trades])
    maes = np.array([t[1] for t in trades])
    years = np.array([t[3] for t in trades])
    wins = (pnls > 0).sum()
    total = len(pnls)
    wr = wins / total * 100
    wm = pnls > 0
    avg_w = pnls[wm].mean() if wm.any() else 0
    avg_l = pnls[~wm].mean() if (~wm).any() else 0
    losses_n = (~wm).sum()
    pf = abs(avg_w * wins / (avg_l * losses_n + 1e-10)) if losses_n > 0 else 999
    cum = pnls.cumsum()
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum).max()
    mae95 = abs(np.percentile(maes, 95)) if len(maes) > 0 else 0.1
    sharpe = (pnls.mean() / (pnls.std() + 1e-10)) * np.sqrt(252)
    yearly = {}
    for y in np.unique(years):
        m = years == y
        yp = pnls[m]
        yw = (yp > 0).sum()
        yearly[str(y)] = {'wr': round(yw / len(yp) * 100, 1), 'trades': int(len(yp)), 'pnl': round(yp.sum(), 2)}
    return {
        'trades': int(total), 'wr': round(wr, 1), 'pnl': round(pnls.sum(), 2),
        'profit_factor': round(pf, 2), 'max_drawdown': round(dd, 2),
        'mae_p95': round(mae95, 4), 'sharpe': round(sharpe, 2), 'yearly': yearly
    }


def calc_score(yearly):
    years = sorted(yearly.keys(), reverse=True)
    n = len(years)
    if n == 0:
        return 0
    wm = {1: [100], 2: [55, 45], 3: [40, 30, 30], 4: [35, 25, 25, 15],
          5: [30, 25, 20, 15, 10], 6: [27, 22, 18, 14, 11, 8], 7: [25, 20, 16, 13, 10, 9, 7]}
    w = wm.get(min(n, 7), wm[7])[:n]
    return round(sum(yearly[y]['wr'] * w[i] / 100 for i, y in enumerate(years[:len(w)])), 1)


# ─── WORKER: process one symbol across all strategies ───
def process_symbol(args):
    """Worker function: optimize all strategies for one symbol."""
    symbol, strat_dict_serialized, round_name = args

    # Re-import strategies in worker process
    sys.path.insert(0, PROJECT_DIR)
    sys.path.insert(0, os.path.join(PROJECT_DIR, "strategies_code"))

    if round_name == 'r2':
        from strategies_round2 import STRATEGY_TYPES_R2
        strategies = STRATEGY_TYPES_R2
    else:
        # R3: reload in worker
        strategies = _load_r3_in_worker()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Load candles
    df_5m = load_candles(symbol, '5m')
    if df_5m is None or len(df_5m) < 500:
        return symbol, []

    dfs = {'5m': df_5m, '15m': resample(df_5m, '15m'), '4h': resample(df_5m, '4h')}
    df_1h = load_candles(symbol, '1h')
    dfs['1h'] = df_1h if df_1h is not None and len(df_1h) > 100 else None
    df_1d = load_candles(symbol, '1d')
    dfs['1d'] = df_1d if df_1d is not None and len(df_1d) > 30 else resample(df_5m, '1d')

    sym_grails = []

    for tf, df in dfs.items():
        if df is None or len(df) < 200:
            continue
        split = int(len(df) * 0.7)
        train = df.iloc[:split].copy()
        test = df.iloc[split:].copy()
        if len(train) < 100 or len(test) < 50:
            continue

        for sname, info in strategies.items():
            gen = info['gen']
            space = info['space']
            try:
                study = optuna.create_study(
                    direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=42)
                )

                def obj(trial, _gen=gen, _space=space, _train=train):
                    params = _space(trial)
                    try:
                        sig = _gen(_train, **params)
                        r = backtest(_train, sig)
                        if r is None:
                            return -999
                        return r['wr'] * 0.7 + min(r['sharpe'], 50) * 0.3
                    except:
                        return -999

                # Check if strategy has params to optimize
                has_params = True
                try:
                    test_p = space(optuna.trial.FixedTrial({}))
                    if not test_p:
                        has_params = False
                except:
                    has_params = True  # assume it needs trial.suggest_*

                if has_params:
                    study.optimize(obj, n_trials=N_TRIALS, timeout=TIMEOUT_PER_STRAT)
                    if study.best_value <= 0:
                        continue
                    bp = study.best_params
                else:
                    # No params to optimize — just run once with defaults
                    bp = {}
                    sig_check = gen(train, **bp)
                    r_check = backtest(train, sig_check)
                    if r_check is None or r_check['wr'] < 50:
                        continue

                # Validate on test set
                test_sig = gen(test, **bp)
                test_r = backtest(test, test_sig)
                train_sig = gen(train, **bp)
                train_r = backtest(train, train_sig)
                full_sig = gen(df, **bp)
                full_r = backtest(df, full_sig)

                if not test_r or not train_r or not full_r:
                    continue
                if test_r['wr'] < 70 or test_r['pnl'] <= 0:
                    continue
                if full_r['trades'] < MIN_TRADES_FULL:
                    continue
                wr_diff = abs(train_r['wr'] - test_r['wr'])
                if wr_diff > 30:
                    continue  # OVERFIT

                mae = test_r.get('mae_p95', 0.1)
                grail = {
                    'strategy': sname, 'symbol': symbol, 'timeframe': tf,
                    'best_params': bp,
                    'train': {'wr': train_r['wr'], 'pnl': train_r['pnl'], 'sharpe': train_r['sharpe'],
                              'trades': train_r['trades'], 'pf': train_r['profit_factor']},
                    'test': {'wr': test_r['wr'], 'pnl': test_r['pnl'], 'sharpe': test_r['sharpe'],
                             'trades': test_r['trades'], 'pf': test_r['profit_factor'],
                             'yearly': test_r.get('yearly', {}), 'mae_p95': mae,
                             'max_dd': test_r['max_drawdown']},
                    'full': {'wr': full_r['wr'], 'pnl': full_r['pnl'], 'sharpe': full_r['sharpe'],
                             'trades': full_r['trades'], 'pf': full_r['profit_factor'],
                             'max_dd': full_r['max_drawdown']},
                    'wr_diff': round(wr_diff, 1),
                    'weighted_score': calc_score(test_r.get('yearly', {})),
                    'safe_leverage': min(20, int(1 / (mae * 2.5 + 1e-10))),
                    'round': round_name,
                }
                sym_grails.append(grail)
            except:
                continue

    return symbol, sym_grails


def _load_r3_in_worker():
    """Load R3 strategies inside worker process."""
    strats = {}
    try:
        import batch2_strategies as b2
        import inspect
        for name in dir(b2):
            obj = getattr(b2, name)
            if callable(obj) and not name.startswith('_') and name not in (
                'ema', 'sma', 'rsi', 'atr', 'vwap', 'donchian_high', 'donchian_low',
                'linreg_slope', 'hurst_exponent', 'np', 'pd', 'stats'
            ):
                sig = inspect.signature(obj)
                params = [p for p in sig.parameters if p != 'df']

                def make_gen(f, s):
                    def gen(df, **kwargs):
                        try:
                            return f(df, **{k: v for k, v in kwargs.items()
                                           if k in [p.name for p in s.parameters.values()]})
                        except:
                            return None
                    return gen

                def make_space(s, pnames):
                    def space(trial_or_dict):
                        if isinstance(trial_or_dict, dict):
                            return {}
                        p = {}
                        for pname in pnames:
                            param = s.parameters[pname]
                            default = param.default if param.default != inspect.Parameter.empty else 14
                            if isinstance(default, int):
                                lo, hi = max(2, default // 3), default * 3
                                p[pname] = trial_or_dict.suggest_int(pname, lo, hi)
                            elif isinstance(default, float):
                                lo, hi = max(0.01, default / 3), default * 3
                                p[pname] = trial_or_dict.suggest_float(pname, lo, hi)
                        return p
                    return space

                strats[name] = {
                    'gen': make_gen(obj, sig),
                    'space': make_space(sig, params)
                }
    except Exception as e:
        print(f"[R3 Worker] batch2 load error: {e}")

    # batch1-6: extract signals via simulate monkey-patch
    for batch_num in range(1, 7):
        try:
            mod = __import__(f"strategies_batch{batch_num}")
            dict_name = f"BATCH{batch_num}_STRATEGIES"
            batch_dict = getattr(mod, dict_name, None)

            if batch_dict and isinstance(batch_dict, dict):
                items = list(batch_dict.items())
            else:
                items = [(n, getattr(mod, n)) for n in dir(mod)
                         if n.startswith('s_') and callable(getattr(mod, n))]

            for sname, func in items:
                key = f"B{batch_num}_{sname}"
                if key in strats:
                    continue

                def make_batch_gen(f, m):
                    def gen(df, **kwargs):
                        try:
                            original = getattr(m, 'simulate', None)
                            captured = {}
                            def cap(df_in, signals, max_hold=None):
                                captured['sig'] = signals
                                return [], [1.0]
                            if original:
                                m.simulate = cap
                                try:
                                    f(df)
                                finally:
                                    m.simulate = original
                                if 'sig' in captured:
                                    s = captured['sig']
                                    return s if isinstance(s, pd.Series) else pd.Series(s, index=df.index[:len(s)])
                            return None
                        except:
                            return None
                    return gen

                strats[key] = {
                    'gen': make_batch_gen(func, mod),
                    'space': lambda trial_or_dict: {}  # no params for batch strategies
                }
        except Exception as e:
            print(f"[R3 Worker] batch{batch_num}: {e}")

    return strats


# ─── MAIN ───
def run_round(round_name, progress_file):
    """Run one round with multiprocessing."""
    symbols = get_symbols()

    # Load progress
    progress = {'completed': [], 'grails': [], 'stats': {}}
    if os.path.exists(progress_file):
        progress = json.load(open(progress_file))
    done_set = set(progress.get('completed', []))
    all_grails = progress.get('grails', [])
    remaining = [s for s in symbols if s not in done_set]

    # Quick count of strategies
    if round_name == 'r2':
        from strategies_round2 import STRATEGY_TYPES_R2
        n_strats = len(STRATEGY_TYPES_R2)
    else:
        n_strats = '300+'

    print(f"\n{'='*70}")
    print(f"OPTUNA TURBO — Round {round_name.upper()}")
    print(f"Strategies: {n_strats} | Symbols: {len(remaining)}/{len(symbols)} remaining")
    print(f"Workers: {MAX_WORKERS} | CPUs: {multiprocessing.cpu_count()} | Trials: {N_TRIALS}")
    print(f"Grails so far: {len(all_grails)}")
    print(f"{'='*70}\n")

    start = time.time()
    tasks = [(s, None, round_name) for s in remaining]
    completed = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_symbol, t): t[0] for t in tasks}

        for future in as_completed(futures):
            symbol = futures[future]
            completed += 1
            try:
                sym, grails = future.result(timeout=600)
                done_set.add(sym)
                all_grails.extend(grails)

                elapsed = time.time() - start
                rate = elapsed / completed
                eta = rate * (len(remaining) - completed) / 3600

                if grails:
                    best = max(grails, key=lambda g: g['full']['wr'])
                    print(
                        f"[{completed}/{len(remaining)}] {sym[:25]:25s} "
                        f"🏆 {len(grails)} grails | {best['strategy']}@{best['timeframe']} "
                        f"WR={best['full']['wr']:.1f}% | ETA {eta:.1f}h"
                    )
                else:
                    if completed % 20 == 0:
                        print(f"[{completed}/{len(remaining)}] ... grails={len(all_grails)} ETA={eta:.1f}h")

                # Save progress every 5 symbols
                if completed % 5 == 0 or grails:
                    progress['completed'] = list(done_set)
                    progress['grails'] = all_grails
                    progress['stats'] = {
                        'grails': len(all_grails), 'done': len(done_set),
                        'total': len(symbols), 'elapsed_h': round(elapsed / 3600, 2),
                        'eta_h': round(eta, 1), 'round': round_name,
                    }
                    with open(progress_file, 'w') as f:
                        json.dump(progress, f, default=str)

            except Exception as e:
                print(f"[{completed}/{len(remaining)}] {symbol[:25]:25s} ERROR: {e}")
                done_set.add(symbol)

    # Final save
    progress['completed'] = list(done_set)
    progress['grails'] = all_grails
    elapsed = time.time() - start
    progress['stats'] = {
        'grails': len(all_grails), 'done': len(done_set),
        'total': len(symbols), 'elapsed_h': round(elapsed / 3600, 2),
        'round': round_name, 'finished': True,
    }
    with open(progress_file, 'w') as f:
        json.dump(progress, f, default=str)

    print(f"\n{'='*70}")
    print(f"ROUND {round_name.upper()} COMPLETE!")
    print(f"Grails: {len(all_grails)} | Time: {elapsed/3600:.1f}h")
    print(f"{'='*70}\n")
    return all_grails


def main():
    print(f"╔{'═'*68}╗")
    print(f"║  OPTUNA TURBO — Mac M5 × 10 cores × 24GB RAM                      ║")
    print(f"║  R2: 52 strategies | R3: 300+ strategies | 563 assets              ║")
    print(f"║  Workers: {MAX_WORKERS} | Trials: {N_TRIALS} per combo                            ║")
    print(f"╚{'═'*68}╝")

    r2_file = os.path.join(PROJECT_DIR, "data", "optuna_r2_progress.json")
    r3_file = os.path.join(PROJECT_DIR, "data", "optuna_r3_progress.json")

    all_grails = []

    if ROUND in ('r2', 'both'):
        grails = run_round('r2', r2_file)
        all_grails.extend(grails)

    if ROUND in ('r3', 'both'):
        grails = run_round('r3', r3_file)
        all_grails.extend(grails)

    # Consolidate all grails (R1 + R2 + R3)
    r1_file = os.path.join(PROJECT_DIR, "data", "santo_grial.json")
    if os.path.exists(r1_file):
        r1_grails = json.load(open(r1_file))
        if isinstance(r1_grails, list):
            all_grails = r1_grails + all_grails
        elif isinstance(r1_grails, dict) and 'grails' in r1_grails:
            all_grails = r1_grails['grails'] + all_grails

    # Save consolidated
    consolidated_file = os.path.join(PROJECT_DIR, "data", "santo_grial_all.json")
    with open(consolidated_file, 'w') as f:
        json.dump(all_grails, f, default=str)

    print(f"\n🏆 TOTAL GRAILS (R1+R2+R3): {len(all_grails)}")
    print(f"Saved to: {consolidated_file}")


if __name__ == '__main__':
    main()
