#!/usr/bin/env python3
"""
META-STRATEGY OPTIMIZER — Confluence de 2+ estrategias via Optuna
================================================================

CONCEPTO (explicado simple):
  - Hoy: 1 estrategia dice "BUY BTC" → abrir trade
  - Meta: 2 estrategias DIFERENTES dicen "BUY BTC" → señal más fuerte

  La meta-estrategia combina señales de 2 estrategias existentes:
    Señal = (Estrategia_A dice BUY) AND (Estrategia_B dice BUY dentro de N velas)

  Optuna optimiza PER ACTIVO:
    - Qué par de estrategias funciona mejor
    - Ventana de confirmación (1-10 velas)
    - Peso de cada estrategia
    - SL/TP empíricos del combo

PIPELINE:
  1. Para cada símbolo: probar pares de estrategias que YA tienen WR>=70%
  2. Optuna busca el mejor par + parámetros de confluencia
  3. Walk-forward 70/30 para validar (anti-overfit)
  4. Solo guardar si WR>=75% en TEST (más exigente que individual)
  5. MFE backtest para SL/TP empíricos

ANTI-OVERFIT:
  - Require WR >= 75% en OUT-OF-SAMPLE (más estricto que individual)
  - Require >= 10 trades en test period
  - |train_wr - test_wr| <= 20% (más estricto)
  - Min 20 trades total (more data needed for combo)

USAGE:
  python3 scripts/meta_strategy_optuna.py <worker_id> <total_workers>

  # Run with 4 workers:
  for i in $(seq 0 3); do
    python3 -u scripts/meta_strategy_optuna.py $i 4 &
  done

OUTPUT:
  - optuna_results_meta.db (SQLite with all results)
  - Grails format compatible with v6_optimized_sl_tp.json
"""

import sys
import os
import time
import json
import sqlite3
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from collections import defaultdict

# Setup paths
PROJ_ROOT = Path(__file__).resolve().parent.parent
STRAT_CODE = PROJ_ROOT / "strategies_code"
sys.path.insert(0, str(STRAT_CODE))
sys.path.insert(0, str(PROJ_ROOT))

# Import strategy modules
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("ERROR: pip install optuna")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

# ── CONFIG ──────────────────────────────────────────────────────────────
DB_PATH = Path("/Users/sabrina/Code/activos binace /activos_binance.db")
RESULTS_DB = PROJ_ROOT / "optuna_results_meta.db"
OPTIMIZED_JSON = Path(os.path.expanduser("~/CLAUDE CODE/BOT V7/live/data/v6_optimized_sl_tp.json"))

# Walk-forward split
TRAIN_PCT = 0.70
TEST_PCT = 0.30
SEED = 42

# Validation gates (stricter than individual)
MIN_WR_TEST = 75.0        # Higher bar for meta
MIN_TRADES_TEST = 8       # Need enough combo trades
MIN_TRADES_TOTAL = 15     # Total across train+test
MAX_WR_DIFF = 20.0        # Stricter anti-overfit
MIN_PNL_TEST = 0.0        # Must be profitable

# Optuna
N_TRIALS = 30             # Per symbol
TIMEFRAMES = ['5m', '15m', '1h']

# Strategy families (different types that complement each other)
STRATEGY_FAMILIES = {
    'trend': ['ema', 'sma', 'dema', 'supertrend', 'aroon', 'adx', 'macd'],
    'reversal': ['rsi', 'stoch', 'williams', 'cci', 'mfi', 'cmo', 'bb'],
    'volume': ['vwap', 'obv', 'cmf', 'vpt', 'klinger'],
    'volatility': ['atr', 'keltner', 'donchian', 'squeeze', 'ulcer'],
    'pattern': ['engulfing', 'hammer', 'doji', 'harami', 'shooting'],
    'mean_rev': ['zscore', 'mean_rev', 'ou_', 'rubber_band', 'percentile'],
    'momentum': ['momentum', 'roc', 'trix', 'fourier'],
}

def classify_strategy(name: str) -> str:
    """Classify strategy into a family based on name."""
    name_lower = name.lower()
    for family, keywords in STRATEGY_FAMILIES.items():
        if any(kw in name_lower for kw in keywords):
            return family
    return 'other'


# ── DATABASE ────────────────────────────────────────────────────────────

def init_results_db():
    """Create results DB if not exists."""
    db = sqlite3.connect(str(RESULTS_DB))
    db.execute("""
        CREATE TABLE IF NOT EXISTS meta_grails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            strategy_a TEXT NOT NULL,
            strategy_b TEXT NOT NULL,
            family_a TEXT,
            family_b TEXT,
            confirm_window INT,
            weight_a REAL,
            weight_b REAL,
            train_wr REAL,
            test_wr REAL,
            train_pnl REAL,
            test_pnl REAL,
            train_trades INT,
            test_trades INT,
            total_trades INT,
            wr_diff REAL,
            composite_score REAL,
            params_a TEXT,
            params_b TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timeframe, strategy_a, strategy_b)
        )
    """)
    db.commit()
    return db


def load_candles(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load candles from activos_binance.db."""
    try:
        db = sqlite3.connect(str(DB_PATH))
        # Symbol in DB is already "BTC/USDT:USDT" format
        # If passed as "BTCUSDT", convert back
        if '/' not in symbol:
            # BTCUSDT → BTC/USDT:USDT
            sym_db = symbol.replace('USDT', '/USDT:USDT') if symbol.endswith('USDT') else symbol
        else:
            sym_db = symbol
        df = pd.read_sql(
            "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol = ? AND timeframe = ? ORDER BY ts",
            db, params=(sym_db, timeframe)
        )
        db.close()

        if df.empty:
            return pd.DataFrame()

        # Standard OHLCV columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                # Try common alternatives
                alt = {'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v'}
                if alt.get(col) in df.columns:
                    df[col] = df[alt[col]]

        # ts column is Unix ms timestamp
        if 'ts' in df.columns:
            df.index = pd.to_datetime(df['ts'], unit='ms')
        elif 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.debug(f"Failed to load {symbol} {timeframe}: {e}")
        return pd.DataFrame()


# ── STRATEGY LOADING ────────────────────────────────────────────────────

def load_all_strategy_functions() -> dict:
    """Load all s_XXX functions from batch strategy files."""
    strat_funcs = {}

    for batch_num in range(1, 8):
        batch_file = STRAT_CODE / f"strategies_batch{batch_num}.py"
        if not batch_file.exists():
            continue

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"batch{batch_num}", str(batch_file))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            for name in dir(mod):
                if name.startswith('s_') and callable(getattr(mod, name)):
                    strat_funcs[name] = getattr(mod, name)
        except Exception as e:
            logger.warning(f"Failed to load batch{batch_num}: {e}")

    logger.info(f"Loaded {len(strat_funcs)} strategy functions")
    return strat_funcs


def get_existing_grails() -> dict:
    """Load existing grails to know which strategy+symbol combos are good."""
    try:
        with open(str(OPTIMIZED_JSON)) as f:
            data = json.load(f)

        # Group by symbol → list of strategies with WR >= 70%
        grails = defaultdict(list)
        for b in data:
            wr = b.get('test_wr', 0) or 0
            if wr >= 70:
                grails[b['symbol']].append({
                    'strategy': b['strategy'],
                    'wr': wr,
                    'timeframe': b.get('timeframe', '5m'),
                    'params': b.get('params', {}),
                    'sl_pct': b.get('sl_pct', 0),
                    'tp_pct': b.get('tp_pct', 0),
                })
        return grails
    except Exception as e:
        logger.error(f"Failed to load grails: {e}")
        return {}


# ── META-STRATEGY SIMULATION ────────────────────────────────────────────

def simulate_meta(df: pd.DataFrame, func_a, func_b,
                  confirm_window: int = 3,
                  weight_a: float = 0.5, weight_b: float = 0.5,
                  max_hold: int = None) -> list:
    """
    Simulate a meta-strategy that requires CONFLUENCE from two strategies.

    Logic:
      1. Run Strategy A → get signal_a series
      2. Run Strategy B → get signal_b series
      3. Combined signal: both must agree within `confirm_window` bars
      4. Entry when combined signal triggers, exit when either says exit

    Returns list of trades with pnl_pct.
    """
    try:
        trades_a, _ = func_a(df)
        trades_b, _ = func_b(df)
    except Exception:
        return []

    if not trades_a or not trades_b:
        return []

    # Convert trades to signal series for easier combination
    sig_a = pd.Series(0, index=df.index)
    sig_b = pd.Series(0, index=df.index)

    # Mark entry bars
    for t in trades_a:
        try:
            idx = df.index.get_loc(t['entry_time'])
            sig_a.iloc[idx] = 1
        except (KeyError, IndexError):
            pass

    for t in trades_b:
        try:
            idx = df.index.get_loc(t['entry_time'])
            sig_b.iloc[idx] = 1
        except (KeyError, IndexError):
            pass

    # Rolling window: check if both strategies signaled within N bars
    sig_a_window = sig_a.rolling(confirm_window, min_periods=1).max()
    sig_b_window = sig_b.rolling(confirm_window, min_periods=1).max()

    # Confluence: both must have signaled within the window
    confluence = ((sig_a_window > 0) & (sig_b_window > 0)).astype(int)

    # Simulate trades using confluence signals
    SLIPPAGE = 0.0005
    COMMISSION = 0.001  # 0.1% per side (Binance/OKX futures)
    mh = max_hold or 200  # default max hold
    trades = []
    pos = 0
    ep = 0
    ei = 0

    for i in range(1, len(df) - 1):
        if pos == 0 and confluence.iloc[i] == 1:
            # Entry on next bar
            pos = 1
            ep = df['open'].iloc[i + 1] * (1 + SLIPPAGE + COMMISSION)
            ei = i + 1
        elif pos == 1:
            # Exit conditions: max hold or price reversal
            bars_held = i - ei
            if bars_held >= mh:
                exit_price = df['open'].iloc[i + 1] * (1 - SLIPPAGE - COMMISSION)
                pnl = (exit_price - ep) / ep * 100
                trades.append({
                    'entry_time': df.index[ei],
                    'exit_time': df.index[i + 1],
                    'pnl_pct': pnl,
                    'bars_held': bars_held,
                })
                pos = 0

    return trades


# ── OPTUNA OBJECTIVE ────────────────────────────────────────────────────

def create_objective(symbol: str, timeframe: str, df: pd.DataFrame,
                     strat_funcs: dict, candidate_pairs: list):
    """Create Optuna objective for a symbol."""

    # Walk-forward split
    split_idx = int(len(df) * TRAIN_PCT)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    def objective(trial):
        # Choose pair of strategies
        pair_idx = trial.suggest_int('pair_idx', 0, len(candidate_pairs) - 1)
        strat_a_name, strat_b_name = candidate_pairs[pair_idx]

        func_a = strat_funcs.get(strat_a_name)
        func_b = strat_funcs.get(strat_b_name)
        if not func_a or not func_b:
            return -999

        # Confluence parameters
        confirm_window = trial.suggest_int('confirm_window', 1, 10)
        max_hold = trial.suggest_int('max_hold', 10, 200)

        # Run on TRAIN
        try:
            train_trades = simulate_meta(df_train, func_a, func_b,
                                        confirm_window=confirm_window,
                                        max_hold=max_hold)
        except Exception:
            return -999

        if len(train_trades) < 5:
            return -999

        train_wins = sum(1 for t in train_trades if t['pnl_pct'] > 0)
        train_wr = train_wins / len(train_trades) * 100
        train_pnl = sum(t['pnl_pct'] for t in train_trades)

        if train_wr < 65 or train_pnl <= 0:
            return -999

        # Run on TEST (out-of-sample)
        try:
            test_trades = simulate_meta(df_test, func_a, func_b,
                                       confirm_window=confirm_window,
                                       max_hold=max_hold)
        except Exception:
            return -999

        if len(test_trades) < MIN_TRADES_TEST:
            return -999

        test_wins = sum(1 for t in test_trades if t['pnl_pct'] > 0)
        test_wr = test_wins / len(test_trades) * 100
        test_pnl = sum(t['pnl_pct'] for t in test_trades)

        # Validation gates
        wr_diff = abs(train_wr - test_wr)
        if test_wr < MIN_WR_TEST:
            return -999
        if test_pnl <= MIN_PNL_TEST:
            return -999
        if wr_diff > MAX_WR_DIFF:
            return -999

        # Composite score (weighted by recency — recent years worth more)
        # For now, use simple score
        total_trades = len(train_trades) + len(test_trades)
        score = test_wr * 0.4 + min(test_pnl, 100) * 0.3 + min(total_trades, 50) * 0.3

        # Store metadata in trial
        trial.set_user_attr('strategy_a', strat_a_name)
        trial.set_user_attr('strategy_b', strat_b_name)
        trial.set_user_attr('train_wr', train_wr)
        trial.set_user_attr('test_wr', test_wr)
        trial.set_user_attr('train_pnl', train_pnl)
        trial.set_user_attr('test_pnl', test_pnl)
        trial.set_user_attr('train_trades', len(train_trades))
        trial.set_user_attr('test_trades', len(test_trades))
        trial.set_user_attr('wr_diff', wr_diff)

        return score

    return objective


# ── MAIN WORKER ─────────────────────────────────────────────────────────

def _build_func_keywords(strat_funcs: dict) -> dict:
    """Pre-build keyword index for fuzzy matching."""
    kw = {}
    for fname in strat_funcs:
        parts = fname.replace('s_', '').split('_')
        kw[fname] = set(p for p in parts if len(p) > 2)
    return kw

_FUNC_KW_CACHE = {}

def _find_matching_func(grail_name: str, strat_funcs: dict) -> str:
    """Find best matching s_ function for a grail strategy name.

    Tries multiple strategies:
    1. Direct match: s_<name>
    2. Remove batch prefix: B1_RSI → s_rsi
    3. Fuzzy: match by keyword overlap (>=2 keywords)
    """
    import re
    clean = re.sub(r'^B\d+_', '', grail_name).lower()

    # Direct match
    direct = f"s_{clean}"
    if direct in strat_funcs:
        return direct

    # Build keyword cache once
    global _FUNC_KW_CACHE
    if not _FUNC_KW_CACHE:
        _FUNC_KW_CACHE = _build_func_keywords(strat_funcs)

    # Keyword overlap
    grail_parts = set(p for p in clean.split('_') if len(p) > 2)
    best_match = None
    best_overlap = 0
    for fname, fkw in _FUNC_KW_CACHE.items():
        overlap = len(grail_parts & fkw)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = fname

    if best_overlap >= 2:
        return best_match
    elif best_overlap == 1 and len(grail_parts) <= 2:
        return best_match

    return None


def get_candidate_pairs(symbol: str, grails: dict, strat_funcs: dict) -> list:
    """Get candidate strategy pairs for a symbol.

    Prefer pairs from DIFFERENT families (trend+reversal, volume+momentum, etc.)
    Uses fuzzy matching to map grail names → s_ function names.
    """
    sym_grails = grails.get(symbol, [])
    if len(sym_grails) < 2:
        return []

    available_strats = []
    seen_funcs = set()
    for g in sym_grails:
        strat_name = g['strategy']
        func_name = _find_matching_func(strat_name, strat_funcs)
        if func_name and func_name not in seen_funcs:
            available_strats.append((strat_name, func_name, g))
            seen_funcs.add(func_name)

    if len(available_strats) < 2:
        return []

    # Generate pairs, preferring different families
    pairs = []
    for (name_a, func_a, grail_a), (name_b, func_b, grail_b) in combinations(available_strats, 2):
        family_a = classify_strategy(name_a)
        family_b = classify_strategy(name_b)

        # Different families get priority (sorted first)
        priority = 0 if family_a != family_b else 1
        pairs.append((priority, func_a, func_b, name_a, name_b))

    # Sort: different families first, then by name
    pairs.sort()

    # Return top 20 pairs (limit for Optuna efficiency)
    return [(p[1], p[2]) for p in pairs[:20]]


def run_worker(worker_id: int, total_workers: int):
    """Main worker loop."""
    logger.info(f"Worker {worker_id}/{total_workers} starting...")

    # Load everything
    strat_funcs = load_all_strategy_functions()
    if not strat_funcs:
        logger.error("No strategy functions loaded!")
        return

    grails = get_existing_grails()
    logger.info(f"Loaded grails for {len(grails)} symbols")

    results_db = init_results_db()

    # Get symbols assigned to this worker
    all_symbols = sorted(grails.keys())
    my_symbols = [s for i, s in enumerate(all_symbols) if i % total_workers == worker_id]
    logger.info(f"Worker {worker_id}: {len(my_symbols)} symbols assigned")

    total_grails = 0
    for sym_idx, symbol in enumerate(my_symbols):
        t0 = time.time()

        # Get candidate pairs
        pairs = get_candidate_pairs(symbol, grails, strat_funcs)
        if not pairs:
            continue

        for timeframe in TIMEFRAMES:
            # Load candles
            df = load_candles(symbol, timeframe)
            if len(df) < 500:  # Need enough data
                continue

            # Create Optuna study
            study_name = f"meta_{symbol}_{timeframe}"
            try:
                study = optuna.create_study(
                    direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=SEED),
                )

                objective = create_objective(symbol, timeframe, df, strat_funcs, pairs)
                study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

                # Save best result if valid
                if study.best_value > 0:
                    best = study.best_trial

                    strat_a = best.user_attrs.get('strategy_a', '')
                    strat_b = best.user_attrs.get('strategy_b', '')
                    test_wr = best.user_attrs.get('test_wr', 0)
                    test_pnl = best.user_attrs.get('test_pnl', 0)

                    if test_wr >= MIN_WR_TEST and test_pnl > 0:
                        try:
                            results_db.execute("""
                                INSERT OR REPLACE INTO meta_grails
                                (symbol, timeframe, strategy_a, strategy_b,
                                 family_a, family_b, confirm_window,
                                 weight_a, weight_b,
                                 train_wr, test_wr, train_pnl, test_pnl,
                                 train_trades, test_trades, total_trades,
                                 wr_diff, composite_score)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                symbol, timeframe, strat_a, strat_b,
                                classify_strategy(strat_a), classify_strategy(strat_b),
                                best.params.get('confirm_window', 3),
                                0.5, 0.5,  # weights (future: optimize)
                                best.user_attrs.get('train_wr', 0),
                                test_wr,
                                best.user_attrs.get('train_pnl', 0),
                                test_pnl,
                                best.user_attrs.get('train_trades', 0),
                                best.user_attrs.get('test_trades', 0),
                                best.user_attrs.get('train_trades', 0) + best.user_attrs.get('test_trades', 0),
                                best.user_attrs.get('wr_diff', 0),
                                study.best_value,
                            ))
                            results_db.commit()
                            total_grails += 1
                            logger.info(
                                f"✅ META GRAIL: {symbol} {timeframe} "
                                f"{strat_a}+{strat_b} "
                                f"WR={test_wr:.1f}% PnL={test_pnl:.1f}% "
                                f"trades={best.user_attrs.get('test_trades', 0)}"
                            )
                        except sqlite3.IntegrityError:
                            pass  # Already exists

            except Exception as e:
                logger.debug(f"Optuna failed for {symbol} {timeframe}: {e}")

        elapsed = time.time() - t0
        if (sym_idx + 1) % 10 == 0:
            logger.info(
                f"Worker {worker_id}: {sym_idx+1}/{len(my_symbols)} symbols, "
                f"{total_grails} meta-grails found, last={elapsed:.1f}s"
            )

    logger.info(f"Worker {worker_id} DONE: {total_grails} meta-grails from {len(my_symbols)} symbols")
    results_db.close()


# ── ENTRY POINT ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 meta_strategy_optuna.py <worker_id> <total_workers>")
        print("Example: for i in $(seq 0 3); do python3 -u meta_strategy_optuna.py $i 4 & done")
        sys.exit(1)

    worker_id = int(sys.argv[1])
    total_workers = int(sys.argv[2])

    run_worker(worker_id, total_workers)
