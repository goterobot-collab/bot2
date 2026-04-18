#!/usr/bin/env python3
"""
FORENSIC BACKTEST ENGINE — Motor de backtesting independiente con trazabilidad total
====================================================================================
Toma params de Optuna → corre 100% de la historia → registra CADA trade con detalle forense.
Genera dashboard HTML auto-contenido con equity curve, drawdown, MAE/MFE, y trade log.

REGLA CLAVE: Este backtest es 100% independiente de Optuna.
  - Entry al OPEN de la vela siguiente a la señal (como en real)
  - Fees reales: 0.1% comisión + 0.05% slippage por lado
  - SL protectivo desde el grail
  - CERO contacto con el optimizador

Usage:
  python3 scripts/forensic_backtest.py                              # Top 20 grails
  python3 scripts/forensic_backtest.py --input data/grails.json     # Custom grails
  python3 scripts/forensic_backtest.py --all                        # Todos los grails
  python3 scripts/forensic_backtest.py --symbol BAN --strategy Adaptive_ZScore  # Uno solo
"""

import json
import sqlite3
import os
import sys
import argparse
import numpy as np
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
from collections import Counter

# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════
PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "strategies_code"))

_HETZNER_DB = "/home/ubuntu/candles_db/activos_binance.db"
_MAC_DB = "/Users/sabrina/CLAUDE CODE/data/activos_binance.db"
DEFAULT_DB = _HETZNER_DB if os.path.exists(_HETZNER_DB) else _MAC_DB
DB_PATH = os.environ.get('BACKTEST_DB_PATH', DEFAULT_DB)

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS — Fees REALES del exchange
# ═══════════════════════════════════════════════════════════════════════
COMMISSION = 0.001     # 0.1% per side (Binance taker)

# SLIPPAGE DINAMICO POR LIQUIDEZ (2026-04-10)
# Basado en investigacion: micro-caps tienen spread 0.1-0.5% vs 0.02% en BTC
# Sources: Bitget Academy, sandx.ai architecture, forensic analysis interno
SLIPPAGE_BY_VOLUME = {
    'HIGH':  0.0002,   # >$10M/h vol — BTC, SOL, XRP (spread ~0.02%)
    'MED':   0.0005,   # $1-10M/h — LTC, XMR, D, KERNEL (spread ~0.05%)
    'LOW':   0.0010,   # $100K-1M/h — NAORIS, BAN, BEL, SKYAI (spread ~0.10%)
    'MICRO': 0.0015,   # <$100K/h — SFP, B2, RLC, XVG, ALCH (spread ~0.15%)
}

# Volume thresholds (USD/hour average)
VOL_THRESHOLDS = {
    'HIGH':  10_000_000,
    'MED':   1_000_000,
    'LOW':   100_000,
    'MICRO': 0,
}

# Default (fallback if volume unknown)
SLIPPAGE = 0.0005      # 0.05% default

# TP_CAP_BY_TIMEFRAME — MUST match production signal_factory.py
# FIX 2026-04-11: Cerebro flagged as URGENTE — forensic without caps inflates PnL
# vs production which caps TP. Without this, CONFIRMED grails may underperform.
TP_CAP_BY_TIMEFRAME = {
    '5m':  0.08,   # 8%
    '15m': 0.10,   # 10%
    '1h':  0.12,   # 12%
    '4h':  0.18,   # 18%
    '1d':  0.30,   # 30%
}
COST_PER_SIDE = COMMISSION + SLIPPAGE  # default 0.15%


def get_liquidity_tier(avg_volume_usd):
    """Classify symbol liquidity by average hourly volume in USD."""
    if avg_volume_usd > 10_000_000:
        return 'HIGH'
    elif avg_volume_usd > 1_000_000:
        return 'MED'
    elif avg_volume_usd > 100_000:
        return 'LOW'
    return 'MICRO'


def get_slippage_for_symbol(symbol, db_path=None):
    """Get realistic slippage estimate based on recent volume."""
    if db_path is None:
        db_path = DB_PATH
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        conn.execute("PRAGMA query_only=ON")
        row = conn.execute("""
            SELECT AVG(volume * close) as avg_vol
            FROM candles WHERE symbol=? AND timeframe='1h'
            AND ts > (SELECT MAX(ts) - 30*86400000 FROM candles WHERE symbol=? AND timeframe='1h')
        """, (symbol, symbol)).fetchone()
        conn.close()
        if row and row[0]:
            tier = get_liquidity_tier(row[0])
            return SLIPPAGE_BY_VOLUME[tier], tier, row[0]
    except Exception:
        pass
    return SLIPPAGE, 'UNKNOWN', 0

# ═══════════════════════════════════════════════════════════════════════
# GATES CONSENSO FORENSE — Aprobados por Sabrina 2026-04-10
# ═══════════════════════════════════════════════════════════════════════
# Gates DUROS — SINCRONIZADOS con forensic_gate.py (Regla 24, 2026-04-11)
# ANTES los gaps usaban la dirección equivocada (solo bloqueaban cuando Optuna
# subestimaba), lo que dejaba pasar 254 INFLATED con gap Optuna→Real > 10pp.
# Ahora alineados con los 6 gates de forensic_gate.py:
#   G2: real_wr >= 65%        (era 60% — subido)
#   G3: gap_pos <= 10pp       (era solo gap NEGATIVO — añadido positivo)
#   G4: n_trades >= 20        (igual)
#   G5: pnl > 0               (igual)
GATE_PNL_MIN = 0          # G5: PnL <= 0 → FUERA
GATE_MIN_TRADES = 20      # G4: trades < 20 → FUERA
GATE_MIN_WR = 65.0        # G2: real_WR < 65% → FUERA (era 60%, subido Regla 24)
GATE_MAX_DD = 90.0        # DD > 90% → FUERA
# Gate AJUSTADO (absorbe fees reales Binance):
GATE_MIN_PF = 1.15        # PF < 1.15 → FUERA
# Gate GAP — G3: gap POSITIVO (Optuna sobreestimó) es el riesgo real
# gap = optuna_wr - real_wr. Si >10pp → Optuna mentió, el bot vive de un WR irreal
GATE_GAP_POS_MAX = 10.0   # G3: gap > 10pp → INFLATED → FUERA (alineado con forensic_gate G3)
GATE_GAP_HARD = -20.0     # gap < -20pp → FUERA siempre (Optuna subestimó mucho)
GATE_GAP_SOFT = -15.0     # gap < -15pp + PF < 1.5 → FUERA
GATE_GAP_PF_THRESHOLD = 1.5  # PF threshold for gap soft gate

# ═══════════════════════════════════════════════════════════════════════
# RISK SCORE — 3 capas (PAR 60% + STRATEGY 25% + ASSET 15%)
# Regla de Oro: per activo x per estrategia, NUNCA genérico
# ═══════════════════════════════════════════════════════════════════════
RISK_WEIGHT_PAR = 0.60
RISK_WEIGHT_STRATEGY = 0.25
RISK_WEIGHT_ASSET = 0.15

SIZING_TABLE = [
    (30,  'Full size ($15-20)',  'Entran PRIMERO'),
    (45,  'Standard ($10-12)',   'Entran PRIMERO'),
    (55,  'Reduced ($7-10)',     'Entran SEGUNDO'),
    (65,  'Minimum ($5-7)',      'Entran SEGUNDO'),
    (75,  'Micro ($3-5)',        'Entran ULTIMOS'),
    (100, 'Shadow only ($2 max)','Shadow hasta validar'),
]

# ═══════════════════════════════════════════════════════════════════════
# CANDLE LOADING
# ═══════════════════════════════════════════════════════════════════════
def load_candles(symbol, tf):
    base_tf = {'15m': '5m', '4h': '1h', '1d': '1h'}.get(tf, tf)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA cache_size=-64000")
    df = pd.read_sql(
        "SELECT ts,open,high,low,close,volume FROM candles "
        "WHERE symbol=? AND timeframe=? ORDER BY ts",
        conn, params=(symbol, base_tf))
    conn.close()
    if len(df) == 0:
        return None
    df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
    df.dropna(subset=['ts'], inplace=True)
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    if tf != base_tf:
        m = {'15m': '15min', '4h': '4h', '1d': '1D'}
        r = m.get(tf)
        if r:
            df = df.resample(r).agg({
                'ts': 'first', 'open': 'first', 'high': 'max',
                'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna(subset=['close'])
    return df if len(df) >= 50 else None


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY LOADING (same as optuna_fullhistory.py)
# ═══════════════════════════════════════════════════════════════════════
_STRATEGIES = None

def load_all_strategies():
    global _STRATEGIES
    if _STRATEGIES is not None:
        return _STRATEGIES
    strats = {}
    try:
        from strategies_round2 import STRATEGY_TYPES_R2
        strats.update(STRATEGY_TYPES_R2)
    except Exception: pass
    try:
        from strategies_round3 import STRATEGY_TYPES_R3
        strats.update(STRATEGY_TYPES_R3)
    except Exception: pass
    try:
        from batch_search_spaces import BATCH_SEARCH_SPACES, make_space_func
        from strategies_code.batch_param_wrappers import BATCH_WRAPPERS
        for key, wrapper_fn in BATCH_WRAPPERS.items():
            if key in BATCH_SEARCH_SPACES:
                strats[key] = {'gen': wrapper_fn}
    except Exception: pass
    tv2_dir = os.path.join(PROJECT_DIR, "strategies_tv2_batches")
    if os.path.isdir(tv2_dir) and tv2_dir not in sys.path:
        sys.path.insert(0, tv2_dir)
    for batch_num in range(100, 2500):
        try:
            mod = __import__(f"strategies_tv2_batch{batch_num}")
            if hasattr(mod, 'STRATEGY_EXPORT'):
                strats.update(mod.STRATEGY_EXPORT)
        except ImportError: pass
        except Exception: pass
    for pine_path in ['/Users/sabrina/CLAUDE CODE/BOT V7/strategies',
                      '/home/ubuntu/bot-v7/strategies']:
        try:
            if os.path.isdir(pine_path) and pine_path not in sys.path:
                sys.path.insert(0, pine_path)
            from strategies_new_winners import NEW_WINNER_STRATS
            strats.update(NEW_WINNER_STRATS)
            break
        except Exception: pass
    try:
        from strategy_factory import FACTORY_STRATS
        for name, entry in FACTORY_STRATS.items():
            if name not in strats:
                strats[name] = entry
    except Exception: pass
    _STRATEGIES = strats
    return strats


# ═══════════════════════════════════════════════════════════════════════
# FORENSIC BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════
def forensic_backtest(df, gen_fn, params, sl_pct=0.40, leverage=1, symbol=None, timeframe=None,
                      trade_size_usd=10.0):
    """
    Backtest forense V2 — LONG + SHORT + sizing fijo + funding rates.

    Mejoras V2 (2026-04-12):
      - LONG y SHORT: signal=+1 entra LONG, signal=-1 entra SHORT
      - Sizing fijo: cada trade usa $trade_size_usd (default $10), no compounding
      - Funding rates: 0.01%/8h descontado del PnL de posiciones abiertas
      - Direction-aware: SL/TP/MAE/MFE invertidos para SHORTs

    Reglas de ejecución (como en producción real):
      - Entry: al OPEN de la vela SIGUIENTE a la señal
      - Exit por señal opuesta: al OPEN de la vela siguiente
      - Exit por SL: cuando price adverso toca nivel SL
      - Fees: comisión + slippage DINAMICO por liquidez del activo

    Returns: dict con trades[], equity_curve, metrics
    """
    # Get signal function
    if isinstance(gen_fn, dict) and 'gen' in gen_fn:
        fn = gen_fn['gen']
    elif callable(gen_fn):
        fn = gen_fn
    else:
        return None

    try:
        signals = fn(df, **params)
    except Exception as e:
        return {'error': str(e), 'trades': []}

    if signals is None or len(signals) == 0:
        return {'error': 'No signals generated', 'trades': []}

    # Convert signals to numpy for fast indexing
    if hasattr(signals, 'values'):
        sig_vals = signals.values
    else:
        sig_vals = np.array(signals)

    opens = df['open'].values
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    dates = df.index

    # Dynamic slippage based on symbol liquidity
    if symbol:
        slip, liq_tier, avg_vol = get_slippage_for_symbol(symbol)
    else:
        slip, liq_tier, avg_vol = SLIPPAGE, 'UNKNOWN', 0
    cost_per_side = COMMISSION + slip

    # Funding rate: ~0.01% per 8h (Binance average)
    FUNDING_RATE_PER_8H = 0.0001
    # Estimate bar duration in hours for funding calc
    _tf_hours = {'5m': 5/60, '15m': 0.25, '1h': 1, '4h': 4, '1d': 24}
    bar_hours = _tf_hours.get(timeframe, 1)

    # TP cap per timeframe (matches production signal_factory.py)
    _tp_cap = TP_CAP_BY_TIMEFRAME.get(timeframe, 1.0) if timeframe else 1.0

    trades = []
    balance = 10000.0
    equity = [balance]
    in_trade = False
    direction = None
    peak_equity = balance
    max_drawdown = 0

    for i in range(2, len(sig_vals)):
        prev_sig = sig_vals[i - 1]

        if not in_trade:
            # ── ENTRY: signal=+1 → LONG, signal=-1 → SHORT ──
            if prev_sig == 1 or prev_sig == -1:
                direction = 'LONG' if prev_sig == 1 else 'SHORT'
                entry_price = opens[i]
                if entry_price <= 0:
                    continue
                entry_cost = entry_price * cost_per_side
                entry_date = str(dates[i])[:19]
                entry_bar = i
                worst_mae = 0
                best_mfe = 0
                in_trade = True

        elif in_trade:
            bar_low = lows[i]
            bar_high = highs[i]

            # ── MAE/MFE direction-aware ──
            if direction == 'LONG':
                mae_pct = (entry_price - bar_low) / entry_price
                mfe_pct = (bar_high - entry_price) / entry_price
            else:  # SHORT
                mae_pct = (bar_high - entry_price) / entry_price
                mfe_pct = (entry_price - bar_low) / entry_price

            worst_mae = max(worst_mae, mae_pct)
            best_mfe = max(best_mfe, mfe_pct)

            # ── CHECK SL HIT ──
            sl_hit = False
            if direction == 'LONG' and mae_pct >= sl_pct:
                exit_price = entry_price * (1 - sl_pct)
                sl_hit = True
            elif direction == 'SHORT' and mae_pct >= sl_pct:
                exit_price = entry_price * (1 + sl_pct)
                sl_hit = True

            if sl_hit:
                # PnL calculation
                if direction == 'LONG':
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price

                # Fees (entry + exit)
                fee_pct = cost_per_side * 2
                pnl_pct -= fee_pct

                # Funding cost
                duration_bars = i - entry_bar
                hours_held = duration_bars * bar_hours
                funding_cost = FUNDING_RATE_PER_8H * (hours_held / 8)
                pnl_pct -= funding_cost

                # Fixed sizing: PnL in dollars based on trade_size_usd
                pnl_dollar = trade_size_usd * pnl_pct * leverage
                balance += pnl_dollar

                trades.append({
                    'id': len(trades) + 1,
                    'entry_date': entry_date,
                    'exit_date': str(dates[i])[:19],
                    'entry_price': round(entry_price, 8),
                    'exit_price': round(exit_price, 8),
                    'direction': direction,
                    'pnl_pct': round(pnl_pct * 100, 3),
                    'pnl_dollar': round(pnl_dollar, 2),
                    'duration_bars': duration_bars,
                    'exit_type': 'SL_HIT',
                    'mae_pct': round(worst_mae * 100, 2),
                    'mfe_pct': round(best_mfe * 100, 2),
                    'win': False,
                    'fee_total': round(fee_pct * 100, 3),
                    'funding_cost_pct': round(funding_cost * 100, 4),
                    'balance_after': round(balance, 2),
                })

                equity.append(balance)
                peak_equity = max(peak_equity, balance)
                dd = (peak_equity - balance) / peak_equity if peak_equity > 0 else 0
                max_drawdown = max(max_drawdown, dd)
                in_trade = False
                direction = None
                continue

            # ── CHECK TP_CAP HIT ──
            if mfe_pct >= _tp_cap:
                if direction == 'LONG':
                    exit_price = entry_price * (1 + _tp_cap)
                else:
                    exit_price = entry_price * (1 - _tp_cap)

                if direction == 'LONG':
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price

                fee_pct = cost_per_side * 2
                pnl_pct -= fee_pct

                duration_bars = i - entry_bar
                hours_held = duration_bars * bar_hours
                funding_cost = FUNDING_RATE_PER_8H * (hours_held / 8)
                pnl_pct -= funding_cost

                pnl_dollar = trade_size_usd * pnl_pct * leverage
                balance += pnl_dollar

                trades.append({
                    'id': len(trades) + 1,
                    'entry_date': entry_date,
                    'exit_date': str(dates[i])[:19],
                    'entry_price': round(entry_price, 8),
                    'exit_price': round(exit_price, 8),
                    'direction': direction,
                    'pnl_pct': round(pnl_pct * 100, 3),
                    'pnl_dollar': round(pnl_dollar, 2),
                    'duration_bars': duration_bars,
                    'exit_type': 'TP_CAP',
                    'mae_pct': round(worst_mae * 100, 2),
                    'mfe_pct': round(best_mfe * 100, 2),
                    'win': pnl_pct > 0,
                    'fee_total': round(fee_pct * 100, 3),
                    'funding_cost_pct': round(funding_cost * 100, 4),
                    'balance_after': round(balance, 2),
                })

                equity.append(balance)
                peak_equity = max(peak_equity, balance)
                dd = (peak_equity - balance) / peak_equity if peak_equity > 0 else 0
                max_drawdown = max(max_drawdown, dd)
                in_trade = False
                direction = None
                continue

            # ── CHECK SIGNAL EXIT ──
            # LONG exits on signal=-1 or 0; SHORT exits on signal=+1 or 0
            signal_exit = False
            if direction == 'LONG' and prev_sig in (-1, 0):
                signal_exit = True
            elif direction == 'SHORT' and prev_sig in (1, 0):
                signal_exit = True

            if signal_exit:
                exit_bar = min(i + 1, len(opens) - 1)
                exit_price = opens[exit_bar] if exit_bar < len(opens) else closes[i]

                if direction == 'LONG':
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price

                fee_pct = cost_per_side * 2
                pnl_pct -= fee_pct

                duration_bars = i - entry_bar
                hours_held = duration_bars * bar_hours
                funding_cost = FUNDING_RATE_PER_8H * (hours_held / 8)
                pnl_pct -= funding_cost

                pnl_dollar = trade_size_usd * pnl_pct * leverage
                balance += pnl_dollar

                trades.append({
                    'id': len(trades) + 1,
                    'entry_date': entry_date,
                    'exit_date': str(dates[i])[:19],
                    'entry_price': round(entry_price, 8),
                    'exit_price': round(exit_price, 8),
                    'direction': direction,
                    'pnl_pct': round(pnl_pct * 100, 3),
                    'pnl_dollar': round(pnl_dollar, 2),
                    'duration_bars': duration_bars,
                    'exit_type': 'SIGNAL',
                    'mae_pct': round(worst_mae * 100, 2),
                    'mfe_pct': round(best_mfe * 100, 2),
                    'win': pnl_pct > 0,
                    'fee_total': round(fee_pct * 100, 3),
                    'funding_cost_pct': round(funding_cost * 100, 4),
                    'balance_after': round(balance, 2),
                })

                equity.append(balance)
                peak_equity = max(peak_equity, balance)
                dd = (peak_equity - balance) / peak_equity if peak_equity > 0 else 0
                max_drawdown = max(max_drawdown, dd)
                in_trade = False
                direction = None

    if len(trades) == 0:
        return {'error': 'No trades executed', 'trades': []}

    # ── CALCULATE METRICS ──
    wins = [t for t in trades if t['win']]
    losses = [t for t in trades if not t['win']]
    pnls = [t['pnl_pct'] for t in trades]
    longs = [t for t in trades if t['direction'] == 'LONG']
    shorts = [t for t in trades if t['direction'] == 'SHORT']

    # Sharpe (annualized, crypto trades 365 days)
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(365)
    else:
        sharpe = 0

    # Profit factor
    gross_profit = sum(t['pnl_pct'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl_pct'] for t in losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

    # Consecutive wins/losses
    max_consec_wins = max_consec_losses = current_streak = 0
    is_winning = None
    for t in trades:
        if t['win']:
            if is_winning:
                current_streak += 1
            else:
                current_streak = 1
                is_winning = True
            max_consec_wins = max(max_consec_wins, current_streak)
        else:
            if not is_winning and is_winning is not None:
                current_streak += 1
            else:
                current_streak = 1
                is_winning = False
            max_consec_losses = max(max_consec_losses, current_streak)

    # Funding total
    total_funding = sum(t.get('funding_cost_pct', 0) for t in trades)

    # ── WR PER HOUR UTC (para evaluar wr_low_hour_gate y hard_block_hour per-combo) ──
    hour_stats = {}
    for t in trades:
        try:
            h = int(t['entry_date'][11:13])  # extract hour from "2024-01-15 03:00:00"
        except (ValueError, IndexError):
            continue
        if h not in hour_stats:
            hour_stats[h] = {'wins': 0, 'losses': 0, 'pnl': 0.0}
        if t['win']:
            hour_stats[h]['wins'] += 1
        else:
            hour_stats[h]['losses'] += 1
        hour_stats[h]['pnl'] += t['pnl_pct']

    wr_per_hour = {}
    for h, s in hour_stats.items():
        total_h = s['wins'] + s['losses']
        wr_per_hour[h] = {
            'trades': total_h,
            'wr': round(s['wins'] / total_h * 100, 1),
            'pnl': round(s['pnl'], 2),
        }

    metrics = {
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round(len(wins) / len(trades) * 100, 1),
        'total_pnl_pct': round(sum(pnls), 2),
        'avg_pnl_pct': round(np.mean(pnls), 3),
        'avg_win_pct': round(np.mean([t['pnl_pct'] for t in wins]), 3) if wins else 0,
        'avg_loss_pct': round(np.mean([t['pnl_pct'] for t in losses]), 3) if losses else 0,
        'max_drawdown_pct': round(max_drawdown * 100, 2),
        'sharpe': round(sharpe, 2),
        'profit_factor': round(profit_factor, 2),
        'avg_mae_pct': round(np.mean([t['mae_pct'] for t in trades]), 2),
        'avg_mfe_pct': round(np.mean([t['mfe_pct'] for t in trades]), 2),
        'avg_duration_bars': round(np.mean([t['duration_bars'] for t in trades]), 1),
        'sl_hits': sum(1 for t in trades if t['exit_type'] == 'SL_HIT'),
        'signal_exits': sum(1 for t in trades if t['exit_type'] == 'SIGNAL'),
        'tp_cap_hits': sum(1 for t in trades if t['exit_type'] == 'TP_CAP'),
        'max_consec_wins': max_consec_wins,
        'max_consec_losses': max_consec_losses,
        'final_equity': round(balance, 2),
        'total_fees_pct': round(sum(t['fee_total'] for t in trades), 2),
        'total_funding_pct': round(total_funding, 4),
        'long_trades': len(longs),
        'short_trades': len(shorts),
        'long_wr': round(sum(1 for t in longs if t['win']) / max(1, len(longs)) * 100, 1),
        'short_wr': round(sum(1 for t in shorts if t['win']) / max(1, len(shorts)) * 100, 1),
        'trade_size_usd': trade_size_usd,
        'liquidity_tier': liq_tier,
        'slippage_pct': round(slip * 100, 3),
        'avg_volume_usd': round(avg_vol),
        'cost_per_side_pct': round(cost_per_side * 100, 3),
        # Regla 24 canonical — persistido para gate V8 fee_rate (decimal round-trip)
        'fee_rate': round(cost_per_side * 2, 6),           # 0.0030 = 0.30% round-trip
        'fee_rate_per_side': round(cost_per_side, 6),      # 0.0015 = 0.15% por lado
        'commission_per_side': round(COMMISSION, 6),       # 0.001  = 0.10% fee
        'slippage_per_side': round(slip, 6),               # 0.0005 default, dinámico por liquidez
        'wr_per_hour': wr_per_hour,
    }

    # Drawdown curve
    dd_curve = []
    peak = equity[0]
    for eq in equity:
        peak = max(peak, eq)
        dd_curve.append(round((peak - eq) / peak * 100, 2))

    return {
        'trades': trades,
        'equity_curve': [round(e, 2) for e in equity],
        'drawdown_curve': dd_curve,
        'metrics': metrics,
    }


# ═══════════════════════════════════════════════════════════════════════
# HTML DASHBOARD GENERATOR
# ═══════════════════════════════════════════════════════════════════════
def generate_dashboard_html(all_results, output_path):
    """Genera un HTML auto-contenido con el dashboard forense."""

    # Prepare summary data
    summary_rows = []
    for r in all_results:
        if 'error' in r and not r.get('trades'):
            continue
        m = r['metrics']
        summary_rows.append({
            'strategy': r['strategy'],
            'symbol': r['symbol'].replace('/USDT:USDT', ''),
            'timeframe': r['timeframe'],
            'optuna_wr': r.get('optuna_wr', 0),
            'real_wr': m['win_rate'],
            'gap': round(m['win_rate'] - r.get('optuna_wr', 0), 1),
            'trades': m['total_trades'],
            'pnl': m['total_pnl_pct'],
            'sharpe': m['sharpe'],
            'max_dd': m['max_drawdown_pct'],
            'profit_factor': m['profit_factor'],
            'avg_mae': m['avg_mae_pct'],
            'avg_mfe': m['avg_mfe_pct'],
            'sl_hits': m['sl_hits'],
        })

    # JSON data for charts
    data_json = json.dumps({
        'summary': summary_rows,
        'details': {f"{r['strategy']}|{r['symbol']}|{r['timeframe']}": {
            'trades': r.get('trades', []),
            'equity': r.get('equity_curve', []),
            'drawdown': r.get('drawdown_curve', []),
            'metrics': r.get('metrics', {}),
        } for r in all_results if r.get('trades')},
    }, default=str)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Bot V7 — Backtest Forense</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{
  background: #0a0a0a;
  color: #e0e0e0;
  font-family: "JetBrains Mono", "Courier New", monospace;
  min-height: 100vh;
}}
#header {{
  position: fixed; top: 0; left: 0; right: 0;
  background: rgba(10,10,10,0.95);
  border-bottom: 1px solid #00ff88;
  padding: 12px 24px;
  z-index: 100;
  display: flex; align-items: center; justify-content: space-between;
  backdrop-filter: blur(8px);
}}
#header h1 {{
  font-size: 16px; color: #00ff88;
  letter-spacing: 2px; font-weight: normal;
}}
#header .subtitle {{ font-size: 11px; color: #888; margin-top: 2px; }}
.container {{ margin-top: 70px; padding: 20px; max-width: 1400px; margin-left: auto; margin-right: auto; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }}
.card {{
  background: #111; border: 1px solid #222; border-radius: 8px;
  padding: 16px; text-align: center;
}}
.card .value {{ font-size: 28px; color: #00ff88; font-weight: bold; }}
.card .label {{ font-size: 10px; color: #888; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }}
.card.danger .value {{ color: #ff4444; }}
.card.warning .value {{ color: #ffaa00; }}
.section {{ margin-bottom: 32px; }}
.section h2 {{ color: #00ff88; font-size: 14px; margin-bottom: 12px; letter-spacing: 1px; border-bottom: 1px solid #222; padding-bottom: 8px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
th {{ background: #111; color: #00ff88; padding: 8px 6px; text-align: left; border-bottom: 1px solid #333; position: sticky; top: 60px; }}
td {{ padding: 6px; border-bottom: 1px solid #1a1a1a; }}
tr:hover td {{ background: #0d1f0d; }}
tr.selected td {{ background: #0a2a0a; border-left: 2px solid #00ff88; }}
.wr-good {{ color: #00ff88; }}
.wr-ok {{ color: #ffaa00; }}
.wr-bad {{ color: #ff4444; }}
.gap-good {{ color: #00ff88; }}
.gap-bad {{ color: #ff4444; }}
.charts-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }}
.chart-box {{ background: #111; border: 1px solid #222; border-radius: 8px; padding: 16px; }}
.chart-box h3 {{ color: #00aaff; font-size: 11px; margin-bottom: 8px; }}
canvas {{ max-height: 250px; }}
#trade-log {{ max-height: 400px; overflow-y: auto; }}
.btn {{
  background: transparent; border: 1px solid #00ff88; color: #00ff88;
  padding: 6px 14px; font-family: inherit; font-size: 11px;
  cursor: pointer; margin: 2px; transition: all 0.2s; border-radius: 4px;
}}
.btn:hover, .btn.active {{ background: #00ff88; color: #0a0a0a; }}
.filter-bar {{ margin-bottom: 16px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
.filter-bar label {{ color: #888; font-size: 10px; }}
.detail-panel {{ display: none; margin-top: 24px; }}
.detail-panel.visible {{ display: block; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; margin-bottom: 16px; }}
.metric {{ background: #0a0a0a; border: 1px solid #1a1a1a; padding: 10px; border-radius: 4px; text-align: center; }}
.metric .val {{ font-size: 18px; color: #00ff88; }}
.metric .lbl {{ font-size: 9px; color: #666; margin-top: 2px; }}
</style>
</head>
<body>

<div id="header">
  <div>
    <h1>BACKTEST FORENSE</h1>
    <div class="subtitle">Trades reales acelerados — Entry al OPEN — Fees 0.30% RT — {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
  </div>
</div>

<div class="container">
  <!-- Summary Cards -->
  <div class="cards" id="summary-cards"></div>

  <!-- Comparison Table -->
  <div class="section">
    <h2>OPTUNA vs REALIDAD — Click en una fila para ver detalle</h2>
    <table id="comparison-table">
      <thead>
        <tr>
          <th>Strategy</th><th>Symbol</th><th>TF</th>
          <th>Optuna WR</th><th>Real WR</th><th>Gap</th>
          <th>Trades</th><th>PnL%</th><th>Sharpe</th>
          <th>Max DD</th><th>PF</th><th>Avg MAE</th><th>SL Hits</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>

  <!-- Detail Panel -->
  <div class="detail-panel" id="detail-panel">
    <div class="section">
      <h2 id="detail-title">DETALLE</h2>
      <div class="metric-grid" id="detail-metrics"></div>
    </div>
    <div class="charts-row">
      <div class="chart-box">
        <h3>EQUITY CURVE ($10K inicial)</h3>
        <canvas id="equity-chart"></canvas>
      </div>
      <div class="chart-box">
        <h3>DRAWDOWN</h3>
        <canvas id="dd-chart"></canvas>
      </div>
    </div>
    <div class="charts-row">
      <div class="chart-box">
        <h3>MAE vs MFE (cada punto = 1 trade)</h3>
        <canvas id="mae-mfe-chart"></canvas>
      </div>
      <div class="chart-box">
        <h3>PnL POR TRADE</h3>
        <canvas id="pnl-chart"></canvas>
      </div>
    </div>
    <div class="section" id="trade-log">
      <h2>TRADE LOG</h2>
      <table id="trades-table">
        <thead>
          <tr>
            <th>#</th><th>Entry</th><th>Exit</th><th>Exit Type</th>
            <th>PnL%</th><th>PnL$</th><th>MAE%</th><th>MFE%</th>
            <th>Bars</th><th>Fee%</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const DATA = {data_json};

// ── Render summary cards ──
const summary = DATA.summary;
const totalGrails = summary.length;
const avgGap = summary.reduce((s,r) => s + r.gap, 0) / totalGrails;
const confirmed = summary.filter(r => Math.abs(r.gap) <= 5).length;
const inflated = summary.filter(r => r.gap < -5).length;
const avgRealWR = summary.reduce((s,r) => s + r.real_wr, 0) / totalGrails;

document.getElementById('summary-cards').innerHTML = `
  <div class="card"><div class="value">${{totalGrails}}</div><div class="label">Grails testeados</div></div>
  <div class="card"><div class="value">${{avgRealWR.toFixed(1)}}%</div><div class="label">WR Real Promedio</div></div>
  <div class="card ${{avgGap < -5 ? 'danger' : avgGap < -2 ? 'warning' : ''}}"><div class="value">${{avgGap > 0 ? '+' : ''}}${{avgGap.toFixed(1)}}pp</div><div class="label">Gap Promedio</div></div>
  <div class="card"><div class="value">${{confirmed}}</div><div class="label">Confirmados (gap &lt;5pp)</div></div>
  <div class="card danger"><div class="value">${{inflated}}</div><div class="label">Inflados (gap &gt;5pp)</div></div>
`;

// ── Render comparison table ──
const tbody = document.querySelector('#comparison-table tbody');
summary.sort((a,b) => a.gap - b.gap);
summary.forEach((r, idx) => {{
  const wrClass = r.real_wr >= 70 ? 'wr-good' : r.real_wr >= 60 ? 'wr-ok' : 'wr-bad';
  const gapClass = Math.abs(r.gap) <= 5 ? 'gap-good' : 'gap-bad';
  const tr = document.createElement('tr');
  tr.dataset.key = `${{r.strategy}}|${{r.symbol}}/USDT:USDT|${{r.timeframe}}`;
  tr.innerHTML = `
    <td>${{r.strategy}}</td><td>${{r.symbol}}</td><td>${{r.timeframe}}</td>
    <td>${{r.optuna_wr.toFixed(1)}}%</td>
    <td class="${{wrClass}}">${{r.real_wr.toFixed(1)}}%</td>
    <td class="${{gapClass}}">${{r.gap > 0 ? '+' : ''}}${{r.gap.toFixed(1)}}pp</td>
    <td>${{r.trades}}</td>
    <td style="color:${{r.pnl >= 0 ? '#00ff88' : '#ff4444'}}">${{r.pnl.toFixed(1)}}%</td>
    <td>${{r.sharpe.toFixed(1)}}</td>
    <td>${{r.max_dd.toFixed(1)}}%</td>
    <td>${{r.profit_factor.toFixed(2)}}</td>
    <td>${{r.avg_mae.toFixed(1)}}%</td>
    <td>${{r.sl_hits}}</td>
  `;
  tr.style.cursor = 'pointer';
  tr.onclick = () => showDetail(tr.dataset.key, tr);
  tbody.appendChild(tr);
}});

// ── Charts ──
let equityChart, ddChart, maeMfeChart, pnlChart;

function showDetail(key, rowEl) {{
  document.querySelectorAll('#comparison-table tr').forEach(r => r.classList.remove('selected'));
  if (rowEl) rowEl.classList.add('selected');

  const detail = DATA.details[key];
  if (!detail) return;

  const panel = document.getElementById('detail-panel');
  panel.classList.add('visible');

  const parts = key.split('|');
  document.getElementById('detail-title').textContent = `${{parts[0]}} x ${{parts[1].replace('/USDT:USDT','')}} ${{parts[2]}}`;

  const m = detail.metrics;
  document.getElementById('detail-metrics').innerHTML = `
    <div class="metric"><div class="val">${{m.win_rate}}%</div><div class="lbl">Win Rate</div></div>
    <div class="metric"><div class="val">${{m.total_trades}}</div><div class="lbl">Trades</div></div>
    <div class="metric"><div class="val">${{m.total_pnl_pct}}%</div><div class="lbl">PnL Total</div></div>
    <div class="metric"><div class="val">${{m.sharpe}}</div><div class="lbl">Sharpe</div></div>
    <div class="metric"><div class="val">${{m.max_drawdown_pct}}%</div><div class="lbl">Max DD</div></div>
    <div class="metric"><div class="val">${{m.profit_factor}}</div><div class="lbl">Profit Factor</div></div>
    <div class="metric"><div class="val">${{m.avg_mae_pct}}%</div><div class="lbl">Avg MAE</div></div>
    <div class="metric"><div class="val">${{m.avg_mfe_pct}}%</div><div class="lbl">Avg MFE</div></div>
    <div class="metric"><div class="val">${{m.avg_duration_bars}}</div><div class="lbl">Avg Duration</div></div>
    <div class="metric"><div class="val">${{m.sl_hits}}</div><div class="lbl">SL Hits</div></div>
    <div class="metric"><div class="val">${{m.max_consec_wins}}</div><div class="lbl">Max Consec Wins</div></div>
    <div class="metric"><div class="val">${{m.max_consec_losses}}</div><div class="lbl">Max Consec Losses</div></div>
  `;

  // Equity chart
  if (equityChart) equityChart.destroy();
  equityChart = new Chart(document.getElementById('equity-chart'), {{
    type: 'line',
    data: {{
      labels: detail.equity.map((_, i) => i),
      datasets: [{{ data: detail.equity, borderColor: '#00ff88', borderWidth: 1.5, pointRadius: 0, fill: false }}]
    }},
    options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ display: false }}, y: {{ grid: {{ color: '#1a1a1a' }}, ticks: {{ color: '#888' }} }} }} }}
  }});

  // Drawdown chart
  if (ddChart) ddChart.destroy();
  ddChart = new Chart(document.getElementById('dd-chart'), {{
    type: 'line',
    data: {{
      labels: detail.drawdown.map((_, i) => i),
      datasets: [{{ data: detail.drawdown.map(d => -d), borderColor: '#ff4444', borderWidth: 1.5, pointRadius: 0, fill: true, backgroundColor: 'rgba(255,68,68,0.1)' }}]
    }},
    options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ display: false }}, y: {{ grid: {{ color: '#1a1a1a' }}, ticks: {{ color: '#888' }} }} }} }}
  }});

  // MAE vs MFE scatter
  if (maeMfeChart) maeMfeChart.destroy();
  const wins = detail.trades.filter(t => t.win);
  const loss = detail.trades.filter(t => !t.win);
  maeMfeChart = new Chart(document.getElementById('mae-mfe-chart'), {{
    type: 'scatter',
    data: {{
      datasets: [
        {{ label: 'Wins', data: wins.map(t => ({{ x: t.mae_pct, y: t.mfe_pct }})), backgroundColor: '#00ff88', pointRadius: 4 }},
        {{ label: 'Losses', data: loss.map(t => ({{ x: t.mae_pct, y: t.mfe_pct }})), backgroundColor: '#ff4444', pointRadius: 4 }},
      ]
    }},
    options: {{ responsive: true, scales: {{ x: {{ title: {{ display: true, text: 'MAE %', color: '#888' }}, grid: {{ color: '#1a1a1a' }}, ticks: {{ color: '#888' }} }}, y: {{ title: {{ display: true, text: 'MFE %', color: '#888' }}, grid: {{ color: '#1a1a1a' }}, ticks: {{ color: '#888' }} }} }}, plugins: {{ legend: {{ labels: {{ color: '#888' }} }} }} }}
  }});

  // PnL per trade
  if (pnlChart) pnlChart.destroy();
  pnlChart = new Chart(document.getElementById('pnl-chart'), {{
    type: 'bar',
    data: {{
      labels: detail.trades.map(t => t.id),
      datasets: [{{ data: detail.trades.map(t => t.pnl_pct), backgroundColor: detail.trades.map(t => t.win ? '#00ff88' : '#ff4444') }}]
    }},
    options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ display: false }}, y: {{ grid: {{ color: '#1a1a1a' }}, ticks: {{ color: '#888' }} }} }} }}
  }});

  // Trade log
  const ttbody = document.querySelector('#trades-table tbody');
  ttbody.innerHTML = '';
  detail.trades.forEach(t => {{
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${{t.id}}</td>
      <td>${{t.entry_date.slice(0,10)}}</td>
      <td>${{t.exit_date.slice(0,10)}}</td>
      <td style="color:${{t.exit_type === 'SL_HIT' ? '#ff4444' : '#00aaff'}}">${{t.exit_type}}</td>
      <td style="color:${{t.win ? '#00ff88' : '#ff4444'}}">${{t.pnl_pct > 0 ? '+' : ''}}${{t.pnl_pct.toFixed(2)}}%</td>
      <td style="color:${{t.pnl_dollar >= 0 ? '#00ff88' : '#ff4444'}}">$${{t.pnl_dollar.toFixed(2)}}</td>
      <td>${{t.mae_pct.toFixed(1)}}%</td>
      <td>${{t.mfe_pct.toFixed(1)}}%</td>
      <td>${{t.duration_bars}}</td>
      <td>${{t.fee_total.toFixed(2)}}%</td>
    `;
    ttbody.appendChild(tr);
  }});

  panel.scrollIntoView({{ behavior: 'smooth' }});
}}
</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"\n  Dashboard saved to: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# CONSENSUS GATES — Aplica gates aprobados por Sabrina 2026-04-10
# ═══════════════════════════════════════════════════════════════════════
def apply_consensus_gates(result):
    """
    Aplica los 6 gates del consenso forense a un resultado de backtest.
    Returns: (approved: bool, tier: str, reasons: list[str], tags: list[str])
      - tier: 'TIER_1' (produccion directa), 'TIER_2' (monitoring_required), 'BLOCKED'
    """
    m = result['metrics']
    # gap = optuna_wr - real_wr
    # Positivo → Optuna sobreestimó (riesgo real: el bot vive de WR irreal)
    # Negativo → Optuna subestimó (buena señal)
    optuna_wr = result.get('optuna_wr', 0)
    real_wr = m['win_rate']
    # Si optuna_wr=0 (no guardado), usar real_wr como fallback (gap=0, no penalizar)
    if optuna_wr and optuna_wr > 0:
        gap = optuna_wr - real_wr
    else:
        gap = 0.0
    reasons = []
    tags = []

    # GATES DUROS — alineados con forensic_gate.py Regla 24
    # G5: PnL
    if m.get('total_pnl_pct', m.get('total_return_pct', 0)) <= 0:
        reasons.append(f"PnL <= 0 ({m.get('total_pnl_pct', 0):+.1f}%)")
    # G4: trades mínimos
    if m['total_trades'] < GATE_MIN_TRADES:
        reasons.append(f"trades < {GATE_MIN_TRADES} (n={m['total_trades']})")
    # G2: WR mínimo
    if real_wr < GATE_MIN_WR:
        reasons.append(f"WR < {GATE_MIN_WR}% ({real_wr:.1f}%)")
    if m['max_drawdown_pct'] > GATE_MAX_DD:
        reasons.append(f"DD > {GATE_MAX_DD}% ({m['max_drawdown_pct']:.1f}%)")

    # GATE AJUSTADO — fees reales
    if m['profit_factor'] < GATE_MIN_PF:
        reasons.append(f"PF < {GATE_MIN_PF} ({m['profit_factor']:.2f})")

    # G3: GAP POSITIVO — Optuna sobreestimó (el peligro real)
    # ANTES este gate no existía → dejaba pasar 254 INFLATED
    if gap > GATE_GAP_POS_MAX:
        reasons.append(f"gap_optuna_real={gap:.1f}pp > {GATE_GAP_POS_MAX}pp "
                       f"(optuna={optuna_wr:.1f}% real={real_wr:.1f}%) — INFLATED")

    # Gate gap negativo (Optuna subestimó mucho — datos dudosos)
    if gap < GATE_GAP_HARD:
        reasons.append(f"gap < {GATE_GAP_HARD}pp ({gap:+.1f}pp)")
    elif gap < GATE_GAP_SOFT:
        if m['profit_factor'] < GATE_GAP_PF_THRESHOLD:
            reasons.append(f"gap < {GATE_GAP_SOFT}pp ({gap:+.1f}pp) + PF < {GATE_GAP_PF_THRESHOLD}")
        else:
            tags.append('monitoring_required')

    if reasons:
        return False, 'BLOCKED', reasons, tags

    tier = 'TIER_2' if 'monitoring_required' in tags else 'TIER_1'
    return True, tier, [], tags


# ═══════════════════════════════════════════════════════════════════════
# RISK SCORE 3 CAPAS — per activo x per estrategia (Regla de Oro)
# ═══════════════════════════════════════════════════════════════════════
def _normalize(val, low, high, invert=False):
    """Normaliza val a 0-100. invert=True → valor alto = riesgo bajo."""
    if high == low:
        return 50
    score = max(0, min(100, (val - low) / (high - low) * 100))
    return 100 - score if invert else score


def compute_risk_scores(approved_results):
    """
    Calcula risk score por 3 capas para cada grail aprobado.
    CAPA 1 (60%): El PAR específico (strategy x symbol)
    CAPA 2 (25%): La ESTRATEGIA en todos sus activos
    CAPA 3 (15%): El ACTIVO con todas las estrategias
    Returns: list of dicts con risk_score, par_risk, strat_risk, asset_risk, sizing
    """
    if not approved_results:
        return []

    # ── Pre-calcular agregados por ESTRATEGIA ──
    strat_stats = {}
    for r in approved_results:
        s = r['strategy']
        m = r['metrics']
        if s not in strat_stats:
            strat_stats[s] = {'wrs': [], 'pfs': [], 'pnls': []}
        strat_stats[s]['wrs'].append(m['win_rate'])
        strat_stats[s]['pfs'].append(m['profit_factor'])
        strat_stats[s]['pnls'].append(m.get('total_pnl_pct', m.get('total_return_pct', 0)))

    for s in strat_stats:
        d = strat_stats[s]
        d['avg_wr'] = np.mean(d['wrs'])
        d['avg_pf'] = np.mean(d['pfs'])
        d['pct_losing'] = sum(1 for p in d['pnls'] if p <= 0) / len(d['pnls']) * 100

    # ── Pre-calcular agregados por ACTIVO ──
    asset_stats = {}
    for r in approved_results:
        sym = r['symbol']
        m = r['metrics']
        if sym not in asset_stats:
            asset_stats[sym] = {'wrs': [], 'dds': [], 'pnls': []}
        asset_stats[sym]['wrs'].append(m['win_rate'])
        asset_stats[sym]['dds'].append(m['max_drawdown_pct'])
        asset_stats[sym]['pnls'].append(m.get('total_pnl_pct', m.get('total_return_pct', 0)))

    for a in asset_stats:
        d = asset_stats[a]
        d['avg_wr'] = np.mean(d['wrs'])
        d['avg_dd'] = np.mean(d['dds'])
        d['pct_losing'] = sum(1 for p in d['pnls'] if p <= 0) / len(d['pnls']) * 100

    # ── Rangos globales para normalización ──
    all_dd = [r['metrics']['max_drawdown_pct'] for r in approved_results]
    all_pf = [r['metrics']['profit_factor'] for r in approved_results]
    all_wr = [r['metrics']['win_rate'] for r in approved_results]
    all_sharpe = [r['metrics']['sharpe'] for r in approved_results]
    all_n = [r['metrics']['total_trades'] for r in approved_results]
    all_gaps = [r['metrics']['win_rate'] - r.get('optuna_wr', r['metrics']['win_rate'])
                for r in approved_results]

    dd_lo, dd_hi = min(all_dd), max(all_dd)
    pf_lo, pf_hi = min(all_pf), max(all_pf)
    wr_lo, wr_hi = min(all_wr), max(all_wr)
    sh_lo, sh_hi = min(all_sharpe), max(all_sharpe)
    n_lo, n_hi = min(all_n), max(all_n)
    gap_lo, gap_hi = min(all_gaps), max(all_gaps)

    scored = []
    for r in approved_results:
        m = r['metrics']
        gap = m['win_rate'] - r.get('optuna_wr', m['win_rate'])

        # ── CAPA 1: El PAR específico (60% del peso) ──
        par_dd   = _normalize(m['max_drawdown_pct'], dd_lo, dd_hi, invert=False)  # más DD = más riesgo
        par_pf   = _normalize(m['profit_factor'], pf_lo, pf_hi, invert=True)  # más PF = menos riesgo
        par_wr   = _normalize(m['win_rate'], wr_lo, wr_hi, invert=True)       # más WR = menos riesgo
        par_sh   = _normalize(m['sharpe'], sh_lo, sh_hi, invert=True)         # más Sharpe = menos riesgo
        par_gap  = _normalize(gap, gap_lo, gap_hi, invert=True)              # gap más positivo = menos riesgo
        par_n    = _normalize(m['total_trades'], n_lo, n_hi, invert=True)     # más trades = menos riesgo

        par_risk = (par_dd * 0.30 + par_pf * 0.25 + par_wr * 0.15 +
                    par_sh * 0.10 + par_gap * 0.10 + par_n * 0.10)

        # ── CAPA 2: La ESTRATEGIA en todos sus activos (25% del peso) ──
        ss = strat_stats[r['strategy']]
        strat_wr = _normalize(ss['avg_wr'], 55, 85, invert=True)
        strat_pf = _normalize(ss['avg_pf'], 1.0, 4.0, invert=True)
        strat_loss = _normalize(ss['pct_losing'], 0, 60, invert=False)  # más % perdedor = más riesgo
        strat_risk = strat_wr * 0.40 + strat_pf * 0.30 + strat_loss * 0.30

        # ── CAPA 3: El ACTIVO con todas las estrategias (15% del peso) ──
        aa = asset_stats[r['symbol']]
        asset_wr = _normalize(aa['avg_wr'], 55, 85, invert=True)
        asset_dd = _normalize(aa['avg_dd'], 10, 80, invert=False)     # más DD promedio = más riesgo
        asset_loss = _normalize(aa['pct_losing'], 0, 60, invert=False)
        asset_risk = asset_wr * 0.35 + asset_dd * 0.35 + asset_loss * 0.30

        # ── RISK SCORE FINAL ──
        risk_total = (par_risk * RISK_WEIGHT_PAR +
                      strat_risk * RISK_WEIGHT_STRATEGY +
                      asset_risk * RISK_WEIGHT_ASSET)

        # ── SIZING ──
        sizing = 'Unknown'
        fase = 'Unknown'
        for threshold, sz, ph in SIZING_TABLE:
            if risk_total < threshold:
                sizing = sz
                fase = ph
                break

        scored.append({
            'strategy': r['strategy'],
            'symbol': r['symbol'],
            'timeframe': r.get('timeframe', '?'),
            'risk_score': round(risk_total, 1),
            'par_risk': round(par_risk, 1),
            'strat_risk': round(strat_risk, 1),
            'asset_risk': round(asset_risk, 1),
            'sizing': sizing,
            'fase': fase,
            'win_rate': m['win_rate'],
            'pnl_pct': m.get('total_pnl_pct', m.get('total_return_pct', 0)),
            'profit_factor': m['profit_factor'],
            'max_drawdown': m['max_drawdown_pct'],
            'sharpe': m['sharpe'],
            'total_trades': m['total_trades'],
            'gap': round(gap, 1),
            'tier': r.get('_tier', 'TIER_1'),
            'tags': r.get('_tags', []),
        })

    scored.sort(key=lambda x: x['risk_score'])
    return scored


def get_sizing_for_risk(risk_score):
    """Devuelve sizing y fase para un risk score dado."""
    for threshold, sizing, fase in SIZING_TABLE:
        if risk_score < threshold:
            return sizing, fase
    return 'Shadow only ($2 max)', 'Shadow hasta validar'


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Forensic Backtest Engine')
    parser.add_argument('--input', type=str, help='Path to grails JSON')
    parser.add_argument('--all', action='store_true', help='Test all fullhistory grails')
    parser.add_argument('--strategy', type=str, help='Filter by strategy name')
    parser.add_argument('--symbol', type=str, help='Filter by symbol')
    parser.add_argument('--limit', type=int, default=20, help='Max grails to test')
    parser.add_argument('--output', type=str, default=None, help='Output HTML path')
    parser.add_argument('--db', type=str, default=None, help='Database path')
    parser.add_argument('--workers', type=int, default=1, help='Parallel workers (default 1)')
    parser.add_argument('--trade-size', type=float, default=10.0, help='Fixed trade size USD (default 10)')
    parser.add_argument('--no-merge', action='store_true', help='Do NOT merge with existing results (avoids loading 36GB file)')
    parser.add_argument('--json-output', type=str, default=None, help='Output JSON path (overrides default)')
    args = parser.parse_args()

    global DB_PATH
    if args.db:
        DB_PATH = args.db

    # Load grails
    if args.input:
        with open(args.input) as f:
            grails = json.load(f)
    else:
        # Load both batches
        grails = []
        for fname in ['grails_fullhistory_FINAL.json', 'grails_fullhistory_BATCH2.json']:
            path = os.path.join(DATA_DIR, fname)
            if os.path.exists(path):
                with open(path) as f:
                    grails.extend(json.load(f))

    # Filters
    if args.strategy:
        grails = [g for g in grails if args.strategy.lower() in g['strategy'].lower()]
    if args.symbol:
        grails = [g for g in grails if args.symbol.upper() in g['symbol'].upper()]
    if not args.all:
        grails = grails[:args.limit]

    print(f"FORENSIC BACKTEST ENGINE V2")
    print(f"  Grails: {len(grails)}")
    print(f"  DB: {DB_PATH}")
    print(f"  Fees: {COST_PER_SIDE*200:.1f}% round-trip")
    print(f"  Entry: OPEN next bar")
    print(f"  Sizing: ${args.trade_size}/trade (fixed)")
    print(f"  Workers: {args.workers}")

    # Load strategies
    strategies = load_all_strategies()
    print(f"  Strategies loaded: {len(strategies)}")

    # ── Worker function for parallel processing ──
    def _run_one_grail(task):
        idx, g, total = task
        strat = g['strategy']
        symbol = g['symbol']
        tf = g['timeframe']
        params = g.get('best_params', g.get('params', {}))
        sl = g.get('sl', 0.40)
        lev = g.get('leverage', 1)
        optuna_wr = g.get('full_wr', 0)
        sym = symbol.replace('/USDT:USDT', '')

        gen_fn = strategies.get(strat)
        if not gen_fn:
            return None, f"  [{idx+1}/{total}] {strat} x {sym}: strategy not found"

        df = load_candles(symbol, tf)
        if df is None:
            return None, f"  [{idx+1}/{total}] {strat} x {sym}: no candles"

        result = forensic_backtest(df, gen_fn, params, sl_pct=sl, leverage=lev,
                                   symbol=symbol, timeframe=tf, trade_size_usd=args.trade_size)

        if result is None or not result.get('trades'):
            err = result.get('error', 'no trades') if result else 'None returned'
            return None, f"  [{idx+1}/{total}] {strat} x {sym}: {err}"

        m = result['metrics']
        gap = m['win_rate'] - optuna_wr
        gap_str = f"{gap:+.1f}pp"
        icon = "OK" if abs(gap) <= 5 else ("WARN" if abs(gap) <= 10 else "FAIL")

        result['strategy'] = strat
        result['symbol'] = symbol
        result['timeframe'] = tf
        result['optuna_wr'] = optuna_wr
        result['params'] = params

        msg = (f"  [{idx+1}/{total}] {icon} {strat:<28} {sym:<8} {tf:>3} | "
               f"Optuna:{optuna_wr:.1f}% Real:{m['win_rate']:.1f}% Gap:{gap_str} | "
               f"n={m['total_trades']} L:{m['long_trades']} S:{m['short_trades']} "
               f"PnL={m['total_pnl_pct']:+.1f}% Sharpe={m['sharpe']:.1f}")
        return result, msg

    # ── Run: parallel if workers>1, sequential otherwise ──
    all_results = []
    tasks = [(i, g, len(grails)) for i, g in enumerate(grails)]

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        print(f"\n  Running {len(tasks)} grails with {args.workers} threads...")
        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for result, msg in pool.map(_run_one_grail, tasks):
                completed += 1
                print(msg)
                if result:
                    all_results.append(result)
                if completed % 50 == 0:
                    print(f"  --- Progress: {completed}/{len(tasks)} done, {len(all_results)} valid ---")
    else:
        for task in tasks:
            result, msg = _run_one_grail(task)
            print(msg)
            if result:
                all_results.append(result)

    # ═══════════════════════════════════════════════════════════════════
    # CONSENSUS GATES — Clasificar cada grail
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  GATES CONSENSO FORENSE")
    print(f"{'='*70}")

    approved = []
    blocked = []
    for r in all_results:
        ok, tier, reasons, tags = apply_consensus_gates(r)
        r['_approved'] = ok
        r['_tier'] = tier
        r['_block_reasons'] = reasons
        r['_tags'] = tags
        if ok:
            approved.append(r)
        else:
            blocked.append(r)

    sym_fn = lambda r: r['symbol'].replace('/USDT:USDT', '')
    print(f"\n  ✅ APROBADOS: {len(approved)}/{len(all_results)} ({len(approved)*100//max(1,len(all_results))}%)")
    tier1 = [r for r in approved if r['_tier'] == 'TIER_1']
    tier2 = [r for r in approved if r['_tier'] == 'TIER_2']
    if tier1:
        print(f"     Tier 1 (producción directa): {len(tier1)}")
    if tier2:
        print(f"     Tier 2 (monitoring_required): {len(tier2)}")

    print(f"\n  ❌ BLOQUEADOS: {len(blocked)}")
    reason_counts = Counter()
    for r in blocked:
        for reason in r['_block_reasons']:
            key = reason.split('(')[0].strip()
            reason_counts[key] += 1
    for reason, count in reason_counts.most_common():
        print(f"     {reason}: {count}")

    # ═══════════════════════════════════════════════════════════════════
    # RISK SCORE 3 CAPAS — Solo para aprobados
    # ═══════════════════════════════════════════════════════════════════
    risk_scored = []
    if approved:
        risk_scored = compute_risk_scores(approved)

        print(f"\n{'='*70}")
        print(f"  RISK SCORE — ORDEN DE ENTRADA AL EXCHANGE")
        print(f"  (3 capas: PAR 60% + STRATEGY 25% + ASSET 15%)")
        print(f"{'='*70}")
        print(f"  {'#':>3} {'Risk':>5} {'PAR':>4} {'STR':>4} {'AST':>4} {'Strategy':<28} {'Symbol':<8} {'TF':>3} {'WR':>6} {'PnL%':>8} {'PF':>5} {'DD%':>5} {'Sizing'}")
        print(f"  {'─'*3} {'─'*5} {'─'*4} {'─'*4} {'─'*4} {'─'*28} {'─'*8} {'─'*3} {'─'*6} {'─'*8} {'─'*5} {'─'*5} {'─'*20}")

        for i, s in enumerate(risk_scored, 1):
            sym = s['symbol'].replace('/USDT:USDT', '')
            tag = ' 📋' if 'monitoring_required' in s.get('tags', []) else ''
            print(f"  {i:>3} {s['risk_score']:>5.1f} {s['par_risk']:>4.0f} {s['strat_risk']:>4.0f} {s['asset_risk']:>4.0f} "
                  f"{s['strategy']:<28} {sym:<8} {s['timeframe']:>3} {s['win_rate']:>5.1f}% "
                  f"{s['pnl_pct']:>+7.0f}% {s['profit_factor']:>5.2f} {s['max_drawdown']:>4.0f}% "
                  f"{s['sizing']}{tag}")

    # Save JSON results (con gates + risk)
    if args.json_output:
        json_path = args.json_output
    else:
        json_path = os.path.join(DATA_DIR, f'forensic_v2_results_{datetime.now().strftime("%Y%m%d_%H%M")}.json')

    # Enriquecer resultados con gates info
    for r in all_results:
        r['gate_approved'] = r.pop('_approved', False)
        r['gate_tier'] = r.pop('_tier', 'UNKNOWN')
        r['gate_block_reasons'] = r.pop('_block_reasons', [])
        r['gate_tags'] = r.pop('_tags', [])

    # MERGE con resultados anteriores (skip if --no-merge or file >1GB)
    existing_results = []
    if not args.no_merge:
        merge_path = os.path.join(DATA_DIR, 'forensic_backtest_results.json')
        if os.path.exists(merge_path):
            file_size = os.path.getsize(merge_path)
            if file_size > 1_000_000_000:  # >1GB
                print(f"\n  ⚠ Skipping merge: existing results file is {file_size/1e9:.1f}GB (too large)")
                print(f"  Results saved to NEW file: {json_path}")
            else:
                try:
                    with open(merge_path) as f:
                        prev = json.load(f)
                    if isinstance(prev, dict) and 'results' in prev:
                        existing_results = prev['results']
                    elif isinstance(prev, list):
                        existing_results = prev
                except (json.JSONDecodeError, IOError):
                    pass

    # Merge: new results override existing by key
    merged = {}
    for r in existing_results:
        key = f"{r.get('strategy','')}|{r.get('symbol','')}|{r.get('timeframe','')}"
        merged[key] = r
    for r in all_results:
        key = f"{r.get('strategy','')}|{r.get('symbol','')}|{r.get('timeframe','')}"
        merged[key] = r  # New results take priority

    merged_results = list(merged.values())
    merged_approved = [r for r in merged_results if r.get('gate_approved') is True]
    merged_blocked = [r for r in merged_results if r.get('gate_approved') is False]

    # Merge risk scores (only if not --no-merge)
    all_risk_scored = risk_scored
    if not args.no_merge:
        existing_risk = []
        risk_path_check = os.path.join(DATA_DIR, 'forensic_risk_scores.json')
        if os.path.exists(risk_path_check):
            try:
                file_size = os.path.getsize(risk_path_check)
                if file_size < 100_000_000:  # <100MB
                    with open(risk_path_check) as f:
                        existing_risk = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        merged_risk = {}
        for rs in existing_risk:
            key = f"{rs.get('strategy','')}|{rs.get('symbol','')}|{rs.get('timeframe','')}"
            merged_risk[key] = rs
        for rs in risk_scored:
            key = f"{rs.get('strategy','')}|{rs.get('symbol','')}|{rs.get('timeframe','')}"
            merged_risk[key] = rs
        all_risk_scored = list(merged_risk.values())

    output_data = {
        'timestamp': datetime.now().isoformat(),
        'total_tested': len(merged_results),
        'total_approved': len(merged_approved),
        'total_blocked': len(merged_blocked),
        'results': merged_results,
        'risk_scores': all_risk_scored,
        'gates_config': {
            'pnl_min': GATE_PNL_MIN,
            'min_trades': GATE_MIN_TRADES,
            'min_wr': GATE_MIN_WR,
            'max_dd': GATE_MAX_DD,
            'min_pf': GATE_MIN_PF,
            'gap_hard': GATE_GAP_HARD,
            'gap_soft': GATE_GAP_SOFT,
        },
    }
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\n  Results JSON: {json_path} (merged: {len(existing_results)} existing + {len(all_results)} new = {len(merged_results)} total)")

    # Save risk scores separately for easy consumption (merged)
    if all_risk_scored:
        risk_path = os.path.join(DATA_DIR, 'forensic_risk_scores.json')
        with open(risk_path, 'w') as f:
            json.dump(all_risk_scored, f, indent=2, default=str)
        print(f"  Risk scores: {risk_path} ({len(all_risk_scored)} total)")

    # Generate HTML dashboard (from merged results)
    html_path = args.output or os.path.join(
        str(Path(PROJECT_DIR).parent), 'BOT V7', 'docs', 'maestro_v7', 'backtest_forense.html')
    generate_dashboard_html(merged_results, html_path)

    # Summary
    if all_results:
        gaps = [r['metrics']['win_rate'] - r['optuna_wr'] for r in all_results]
        confirmed = sum(1 for g in gaps if abs(g) <= 5)
        inflated = sum(1 for g in gaps if g < -5)
        approved_pnl = sum(r['metrics'].get('total_pnl_pct', 0) for r in approved)
        blocked_pnl = sum(r['metrics'].get('total_pnl_pct', 0) for r in blocked)

        print(f"\n{'='*70}")
        print(f"  RESUMEN FORENSE COMPLETO")
        print(f"{'='*70}")
        print(f"  Total testeados:    {len(all_results)}")
        print(f"  Aprobados:          {len(approved)} ({len(approved)*100//max(1,len(all_results))}%) | PnL total: {approved_pnl:+,.0f}%")
        print(f"  Bloqueados:         {len(blocked)} | PnL total: {blocked_pnl:+,.0f}%")
        print(f"  Gap promedio:       {np.mean(gaps):+.1f}pp (Optuna→Real)")
        print(f"  CONFIRMED (±5pp):   {confirmed}")
        print(f"  INFLATED (>10pp):   {inflated}")
        if risk_scored:
            top3 = risk_scored[:3]
            print(f"\n  🏆 TOP 3 MAS SEGUROS:")
            for i, t in enumerate(top3, 1):
                sym = t['symbol'].replace('/USDT:USDT', '')
                print(f"     {i}. {t['strategy']} x {sym} {t['timeframe']} | Risk={t['risk_score']:.1f} | {t['sizing']}")
        print(f"\n  Dashboard: {html_path}")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()
