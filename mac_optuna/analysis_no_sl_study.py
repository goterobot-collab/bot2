#!/usr/bin/env python3
"""
ANÁLISIS COMPREHENSIVO: ¿Qué pasa si NO usamos Stop Loss?
=========================================================
Compara 4 estrategias de salida usando datos REALES de candles.
Usa los 17 grails de optuna_v7_progress.json.
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN
# ============================================================
DB_PATH = "/Users/sabrina/CLAUDE CODE/data/activos_binance.db"
PROGRESS_PATH = "/Users/sabrina/CLAUDE CODE/Estrategias/data/optuna_v7_progress.json"
MAX_DURATION_BARS = 30
EMA_PERIOD = 20
RSI_OB = 70  # RSI overbought (exit long)
RSI_OS = 30  # RSI oversold (exit short)
WIDE_SL_MULT = 2.0
ENTRY_SPACING = 20  # bars between entries to avoid overlap

# ============================================================
# CARGAR GRAILS
# ============================================================
with open(PROGRESS_PATH) as f:
    data = json.load(f)

grails = data['grails']
print(f"{'='*80}")
print(f"ANÁLISIS: ¿Qué pasa si NO usamos Stop Loss?")
print(f"{'='*80}")
print(f"Total grails a analizar: {len(grails)}")
print(f"Fuente: optuna_v7_progress.json")
print(f"Base de datos: activos_binance.db")
print()

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def compute_ema(closes, period):
    """EMA usando pandas para velocidad."""
    s = pd.Series(closes)
    return s.ewm(span=period, adjust=False).mean().values

def compute_rsi(closes, period=14):
    """RSI clásico."""
    s = pd.Series(closes)
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def load_candles(symbol, timeframe):
    """Carga candles de la DB."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT ts, open, high, low, close, volume
        FROM candles
        WHERE symbol = ? AND timeframe = ?
        ORDER BY ts ASC
    """
    df = pd.read_sql(query, conn, params=[symbol, timeframe])
    conn.close()
    return df

def resample_to_4h(df_5m):
    """Resamplea candles de 5m a 4h."""
    df = df_5m.copy()
    # Ensure ts is numeric and clean
    df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
    df = df.dropna(subset=['ts'])
    df['ts'] = df['ts'].astype(int)
    # Filter reasonable timestamps (2017-2027)
    df = df[(df['ts'] > 1483228800000) & (df['ts'] < 1798761600000)]
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df = df.set_index('datetime')

    ohlcv = df.resample('4h').agg({
        'ts': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return ohlcv.reset_index(drop=True)


# ============================================================
# SIMULACIÓN DE LAS 4 ESTRATEGIAS
# ============================================================

def simulate_strategy_1(entry_idx, df, sl, tp, direction='long'):
    """Strategy 1: Fixed SL + TP (current system)."""
    entry_price = df['close'].iloc[entry_idx]

    for i in range(entry_idx + 1, min(entry_idx + 200, len(df))):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        close_i = df['close'].iloc[i]

        if direction == 'long':
            # Check SL first (conservative)
            if low <= entry_price * (1 - sl):
                pnl = -sl
                return pnl, i - entry_idx, 'SL', entry_price * (1 - sl)
            # Check TP
            if high >= entry_price * (1 + tp):
                pnl = tp
                return pnl, i - entry_idx, 'TP', entry_price * (1 + tp)
        else:  # short
            if high >= entry_price * (1 + sl):
                pnl = -sl
                return pnl, i - entry_idx, 'SL', entry_price * (1 + sl)
            if low <= entry_price * (1 - tp):
                pnl = tp
                return pnl, i - entry_idx, 'TP', entry_price * (1 - tp)

    # Max bars reached without exit
    final_close = df['close'].iloc[min(entry_idx + 199, len(df) - 1)]
    if direction == 'long':
        pnl = (final_close - entry_price) / entry_price
    else:
        pnl = (entry_price - final_close) / entry_price
    return pnl, 200, 'TIMEOUT', final_close


def simulate_strategy_2(entry_idx, df, tp, max_dur, direction='long'):
    """Strategy 2: NO SL, solo TP + Max Duration."""
    entry_price = df['close'].iloc[entry_idx]
    max_mae = 0  # Maximum adverse excursion

    for i in range(entry_idx + 1, min(entry_idx + max_dur + 1, len(df))):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        close_i = df['close'].iloc[i]

        if direction == 'long':
            mae_this = (entry_price - low) / entry_price
            max_mae = max(max_mae, mae_this)
            if high >= entry_price * (1 + tp):
                return tp, i - entry_idx, 'TP', entry_price * (1 + tp), max_mae
        else:
            mae_this = (high - entry_price) / entry_price
            max_mae = max(max_mae, mae_this)
            if low <= entry_price * (1 - tp):
                return tp, i - entry_idx, 'TP', entry_price * (1 - tp), max_mae

    # Max duration reached - exit at close
    exit_idx = min(entry_idx + max_dur, len(df) - 1)
    final_close = df['close'].iloc[exit_idx]
    if direction == 'long':
        pnl = (final_close - entry_price) / entry_price
        mae_final = (entry_price - df['low'].iloc[entry_idx+1:exit_idx+1].min()) / entry_price if exit_idx > entry_idx else 0
    else:
        pnl = (entry_price - final_close) / entry_price
        mae_final = (df['high'].iloc[entry_idx+1:exit_idx+1].max() - entry_price) / entry_price if exit_idx > entry_idx else 0
    max_mae = max(max_mae, mae_final if mae_final > 0 else 0)
    return pnl, exit_idx - entry_idx, 'MAX_DUR', final_close, max_mae


def simulate_strategy_3(entry_idx, df, tp, ema_values, direction='long'):
    """Strategy 3: NO SL, TP + EMA signal reversal exit."""
    entry_price = df['close'].iloc[entry_idx]
    max_mae = 0

    # Don't check EMA exit in first 3 bars (give signal time)
    for i in range(entry_idx + 1, min(entry_idx + 200, len(df))):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        close_i = df['close'].iloc[i]

        if direction == 'long':
            mae_this = (entry_price - low) / entry_price
            max_mae = max(max_mae, mae_this)
            # Check TP
            if high >= entry_price * (1 + tp):
                return tp, i - entry_idx, 'TP', entry_price * (1 + tp), max_mae
            # Check EMA exit (after 3 bars grace)
            if i >= entry_idx + 3 and close_i < ema_values[i]:
                pnl = (close_i - entry_price) / entry_price
                return pnl, i - entry_idx, 'EMA_EXIT', close_i, max_mae
        else:
            mae_this = (high - entry_price) / entry_price
            max_mae = max(max_mae, mae_this)
            if low <= entry_price * (1 - tp):
                return tp, i - entry_idx, 'TP', entry_price * (1 - tp), max_mae
            if i >= entry_idx + 3 and close_i > ema_values[i]:
                pnl = (entry_price - close_i) / entry_price
                return pnl, i - entry_idx, 'EMA_EXIT', close_i, max_mae

    # Fallback
    exit_idx = min(entry_idx + 199, len(df) - 1)
    final_close = df['close'].iloc[exit_idx]
    if direction == 'long':
        pnl = (final_close - entry_price) / entry_price
    else:
        pnl = (entry_price - final_close) / entry_price
    return pnl, 200, 'TIMEOUT', final_close, max_mae


def simulate_strategy_4(entry_idx, df, sl_wide, tp, direction='long'):
    """Strategy 4: WIDE SL (2x) + TP."""
    entry_price = df['close'].iloc[entry_idx]
    max_mae = 0

    for i in range(entry_idx + 1, min(entry_idx + 200, len(df))):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]

        if direction == 'long':
            mae_this = (entry_price - low) / entry_price
            max_mae = max(max_mae, mae_this)
            if low <= entry_price * (1 - sl_wide):
                return -sl_wide, i - entry_idx, 'SL_WIDE', entry_price * (1 - sl_wide), max_mae
            if high >= entry_price * (1 + tp):
                return tp, i - entry_idx, 'TP', entry_price * (1 + tp), max_mae
        else:
            mae_this = (high - entry_price) / entry_price
            max_mae = max(max_mae, mae_this)
            if high >= entry_price * (1 + sl_wide):
                return -sl_wide, i - entry_idx, 'SL_WIDE', entry_price * (1 + sl_wide), max_mae
            if low <= entry_price * (1 - tp):
                return tp, i - entry_idx, 'TP', entry_price * (1 - tp), max_mae

    exit_idx = min(entry_idx + 199, len(df) - 1)
    final_close = df['close'].iloc[exit_idx]
    if direction == 'long':
        pnl = (final_close - entry_price) / entry_price
    else:
        pnl = (entry_price - final_close) / entry_price
    return pnl, 200, 'TIMEOUT', final_close, max_mae


def analyze_sl_recovery(entry_indices, df, sl, tp, direction='long'):
    """¿Cuántos trades con SL hit habrían sido rentables si se hubieran mantenido?"""
    sl_hit_count = 0
    would_have_recovered = 0
    recovery_bars = []

    for entry_idx in entry_indices:
        entry_price = df['close'].iloc[entry_idx]
        sl_hit = False
        sl_bar = None

        # First check if SL gets hit
        for i in range(entry_idx + 1, min(entry_idx + 200, len(df))):
            if direction == 'long':
                if df['low'].iloc[i] <= entry_price * (1 - sl):
                    sl_hit = True
                    sl_bar = i
                    break
                if df['high'].iloc[i] >= entry_price * (1 + tp):
                    break  # TP hit first, not relevant
            else:
                if df['high'].iloc[i] >= entry_price * (1 + sl):
                    sl_hit = True
                    sl_bar = i
                    break
                if df['low'].iloc[i] <= entry_price * (1 - tp):
                    break

        if sl_hit and sl_bar is not None:
            sl_hit_count += 1
            # Would this trade have recovered to TP if held?
            for j in range(sl_bar + 1, min(entry_idx + 200, len(df))):
                if direction == 'long':
                    if df['high'].iloc[j] >= entry_price * (1 + tp):
                        would_have_recovered += 1
                        recovery_bars.append(j - sl_bar)
                        break
                else:
                    if df['low'].iloc[j] <= entry_price * (1 - tp):
                        would_have_recovered += 1
                        recovery_bars.append(j - sl_bar)
                        break

    return sl_hit_count, would_have_recovered, recovery_bars


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

all_results = []
all_recovery_data = []

print(f"{'='*80}")
print(f"PROCESANDO {len(grails)} GRAILS CON DATOS REALES")
print(f"{'='*80}")
print()

for g_idx, grail in enumerate(grails):
    symbol = grail['symbol']
    strategy = grail['strategy']
    tf = grail['timeframe']
    sl = grail['sl']
    tp = grail['tp']
    direction = grail.get('direction', 'long')
    if direction is None:
        direction = 'long'  # Default

    leverage_detail = grail.get('leverage_detail', {})
    mae_worst = leverage_detail.get('mae_worst', 0)
    mae_p99 = leverage_detail.get('mae_p99', 0)
    mae_p95 = leverage_detail.get('mae_p95', 0)
    test_wr = grail.get('test_wr', 0)
    g1_wr = grail.get('gates', {}).get('G1_WR', {}).get('value', 0)

    print(f"--- Grail {g_idx+1}/{len(grails)}: {strategy} | {symbol} | {tf} ---")
    print(f"    SL={sl:.4f} ({sl*100:.2f}%), TP={tp:.4f} ({tp*100:.2f}%), Dir={direction}")
    print(f"    MAE worst={mae_worst:.4f}, MAE p99={mae_p99:.4f}, MAE p95={mae_p95:.4f}")

    # Load candles
    df_raw = load_candles(symbol, '5m')
    if len(df_raw) == 0:
        print(f"    ⚠ No hay datos para {symbol} en 5m. Saltando.")
        continue

    # Resample if needed
    if tf == '4h':
        df = resample_to_4h(df_raw)
    elif tf == '1h':
        df_raw['ts'] = pd.to_numeric(df_raw['ts'], errors='coerce')
        df_raw = df_raw.dropna(subset=['ts'])
        df_raw['ts'] = df_raw['ts'].astype(int)
        df_raw = df_raw[(df_raw['ts'] > 1483228800000) & (df_raw['ts'] < 1798761600000)]
        df_raw['datetime'] = pd.to_datetime(df_raw['ts'], unit='ms', utc=True)
        df_raw = df_raw.set_index('datetime')
        df = df_raw.resample('1h').agg({
            'ts': 'first', 'open': 'first', 'high': 'max',
            'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().reset_index(drop=True)
    else:
        df = df_raw  # 5m directly

    print(f"    Candles disponibles: {len(df)} ({tf})")

    if len(df) < 200:
        print(f"    ⚠ Datos insuficientes. Saltando.")
        continue

    # Compute EMA for Strategy 3
    ema_values = compute_ema(df['close'].values, EMA_PERIOD)

    # Generate entry points (every ENTRY_SPACING bars, starting from bar 50 for indicator warmup)
    entry_indices = list(range(50, len(df) - 200, ENTRY_SPACING))
    n_entries = len(entry_indices)
    print(f"    Entries simuladas: {n_entries}")

    if n_entries < 10:
        print(f"    ⚠ Menos de 10 entries. Saltando.")
        continue

    # Run all 4 strategies
    results_s1 = []  # Strategy 1: Fixed SL+TP
    results_s2 = []  # Strategy 2: No SL, TP + MaxDur
    results_s3 = []  # Strategy 3: No SL, TP + EMA exit
    results_s4 = []  # Strategy 4: Wide SL + TP

    for entry_idx in entry_indices:
        # S1: Fixed SL + TP
        pnl1, dur1, exit1, _ = simulate_strategy_1(entry_idx, df, sl, tp, direction)
        results_s1.append({'pnl': pnl1, 'duration': dur1, 'exit_type': exit1})

        # S2: No SL, TP + Max Duration
        pnl2, dur2, exit2, _, mae2 = simulate_strategy_2(entry_idx, df, tp, MAX_DURATION_BARS, direction)
        results_s2.append({'pnl': pnl2, 'duration': dur2, 'exit_type': exit2, 'mae': mae2})

        # S3: No SL, TP + EMA exit
        pnl3, dur3, exit3, _, mae3 = simulate_strategy_3(entry_idx, df, tp, ema_values, direction)
        results_s3.append({'pnl': pnl3, 'duration': dur3, 'exit_type': exit3, 'mae': mae3})

        # S4: Wide SL (2x) + TP
        pnl4, dur4, exit4, _, mae4 = simulate_strategy_4(entry_idx, df, sl * WIDE_SL_MULT, tp, direction)
        results_s4.append({'pnl': pnl4, 'duration': dur4, 'exit_type': exit4, 'mae': mae4})

    # SL Recovery analysis
    sl_hits, recovered, recovery_bars = analyze_sl_recovery(entry_indices, df, sl, tp, direction)

    # Compute metrics for each strategy
    def compute_metrics(results, name):
        pnls = [r['pnl'] for r in results]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        durations = [r['duration'] for r in results]
        maes = [r.get('mae', 0) for r in results]

        wr = len(wins) / len(pnls) * 100 if pnls else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        rr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        total_pnl = sum(pnls)
        pnl_per_100 = total_pnl / len(pnls) * 100 if pnls else 0
        worst_trade = min(pnls) if pnls else 0
        best_trade = max(pnls) if pnls else 0
        max_mae = max(maes) if maes else 0
        avg_dur = np.mean(durations) if durations else 0

        # Max consecutive losses
        max_consec = 0
        current_consec = 0
        for p in pnls:
            if p <= 0:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0

        max_dd_potential = max_consec * abs(avg_loss) if avg_loss != 0 else 0

        # At what leverage would worst trade liquidate?
        liq_leverage = 1.0 / abs(worst_trade) if worst_trade < 0 else float('inf')

        # Expected PnL with 2x leverage
        pnl_2x = sum(p * 2 for p in pnls) / len(pnls) * 100 if pnls else 0

        # Exit type distribution
        exit_types = defaultdict(int)
        for r in results:
            exit_types[r['exit_type']] += 1

        return {
            'name': name,
            'n_trades': len(pnls),
            'wr': wr,
            'avg_win': avg_win * 100,  # as percentage
            'avg_loss': avg_loss * 100,
            'rr': rr,
            'total_pnl_pct': total_pnl * 100,
            'pnl_per_100': pnl_per_100,
            'worst_trade_pct': worst_trade * 100,
            'best_trade_pct': best_trade * 100,
            'max_mae_pct': max_mae * 100,
            'avg_duration': avg_dur,
            'max_consec_losses': max_consec,
            'max_dd_potential_pct': max_dd_potential * 100,
            'liq_leverage': liq_leverage,
            'pnl_2x_per_100': pnl_2x,
            'exit_types': dict(exit_types),
        }

    m1 = compute_metrics(results_s1, 'S1: SL+TP Fijo')
    m2 = compute_metrics(results_s2, 'S2: Sin SL, TP+MaxDur')
    m3 = compute_metrics(results_s3, 'S3: Sin SL, TP+EMA Exit')
    m4 = compute_metrics(results_s4, 'S4: SL Ancho (2x)+TP')

    result_row = {
        'grail': f"{strategy}|{symbol}",
        'symbol': symbol,
        'strategy': strategy,
        'tf': tf,
        'sl': sl,
        'tp': tp,
        'direction': direction,
        'test_wr': test_wr,
        'g1_wr': g1_wr,
        'mae_worst': mae_worst,
        'n_entries': n_entries,
        'S1': m1,
        'S2': m2,
        'S3': m3,
        'S4': m4,
        'sl_hits': sl_hits,
        'sl_recovered': recovered,
        'recovery_pct': (recovered / sl_hits * 100) if sl_hits > 0 else 0,
        'recovery_bars_avg': np.mean(recovery_bars) if recovery_bars else 0,
    }
    all_results.append(result_row)

    # Print quick summary for this grail
    print(f"    RESULTADOS (n={n_entries} trades):")
    print(f"    {'Estrategia':<25} {'WR%':>6} {'AvgWin%':>8} {'AvgLoss%':>9} {'R:R':>5} {'PnL/100':>8} {'WorstTr%':>9} {'LiqLev':>7}")
    print(f"    {'-'*80}")
    for m in [m1, m2, m3, m4]:
        liq_str = f"{m['liq_leverage']:.1f}x" if m['liq_leverage'] < 100 else "INF"
        print(f"    {m['name']:<25} {m['wr']:>5.1f}% {m['avg_win']:>7.2f}% {m['avg_loss']:>8.2f}% {m['rr']:>5.2f} {m['pnl_per_100']:>7.2f}% {m['worst_trade_pct']:>8.2f}% {liq_str:>7}")

    if sl_hits > 0:
        print(f"    SL Recovery: {sl_hits} SL hits, {recovered} habrían sido rentables ({result_row['recovery_pct']:.1f}%)")
        if recovery_bars:
            print(f"    Promedio de barras para recovery: {np.mean(recovery_bars):.1f}")
    print()


# ============================================================
# ANÁLISIS AGREGADO
# ============================================================

if not all_results:
    print("ERROR: No hay resultados para analizar.")
    exit(1)

print(f"\n{'='*80}")
print(f"TABLA COMPARATIVA AGREGADA - TODOS LOS GRAILS")
print(f"{'='*80}")
print()

# Aggregate across all grails
agg = {'S1': [], 'S2': [], 'S3': [], 'S4': []}
for r in all_results:
    for sk in ['S1', 'S2', 'S3', 'S4']:
        agg[sk].append(r[sk])

header = f"{'Métrica':<35} {'S1:SL+TP':>12} {'S2:NoSL+Dur':>12} {'S3:NoSL+EMA':>12} {'S4:SL2x+TP':>12}"
print(header)
print("-" * len(header))

metrics_to_show = [
    ('WR Promedio (%)', 'wr', '{:.1f}'),
    ('Avg Win (%)', 'avg_win', '{:.2f}'),
    ('Avg Loss (%)', 'avg_loss', '{:.2f}'),
    ('R:R Promedio', 'rr', '{:.2f}'),
    ('PnL por 100 trades (%)', 'pnl_per_100', '{:.2f}'),
    ('Peor Trade Individual (%)', 'worst_trade_pct', '{:.2f}'),
    ('Mejor Trade Individual (%)', 'best_trade_pct', '{:.2f}'),
    ('MAE Máximo (%)', 'max_mae_pct', '{:.2f}'),
    ('Duración Promedio (bars)', 'avg_duration', '{:.1f}'),
    ('Máx Pérdidas Consecutivas', 'max_consec_losses', '{:.0f}'),
    ('DD Potencial Máx (%)', 'max_dd_potential_pct', '{:.2f}'),
    ('Leverage de Liquidación', 'liq_leverage', '{:.1f}x'),
    ('PnL con 2x Lev/100 trades (%)', 'pnl_2x_per_100', '{:.2f}'),
]

for label, key, fmt in metrics_to_show:
    vals = {}
    for sk in ['S1', 'S2', 'S3', 'S4']:
        values = [m[key] for m in agg[sk]]
        if key == 'liq_leverage':
            # For leverage, use the minimum (worst case)
            vals[sk] = min(values)
        elif key in ['worst_trade_pct', 'max_consec_losses', 'max_dd_potential_pct', 'max_mae_pct']:
            # For "worst" metrics, use the actual worst
            if key in ['worst_trade_pct']:
                vals[sk] = min(values)
            else:
                vals[sk] = max(values)
        else:
            vals[sk] = np.mean(values)

    formatted = f"{label:<35}"
    for sk in ['S1', 'S2', 'S3', 'S4']:
        if key == 'liq_leverage':
            if vals[sk] > 100:
                formatted += f"{'INF':>12}"
            else:
                formatted += f"{fmt.format(vals[sk]):>12}"
        else:
            formatted += f"{fmt.format(vals[sk]):>12}"
    print(formatted)

# ============================================================
# ANÁLISIS DE RECUPERACIÓN DE SL
# ============================================================

print(f"\n{'='*80}")
print(f"ANÁLISIS: ¿Cuántos trades con SL hit habrían sido rentables si se mantuvieran?")
print(f"{'='*80}")
print()
print(f"{'Grail':<40} {'SL Hits':>8} {'Recuperados':>12} {'% Recup':>8} {'Bars Avg':>9}")
print("-" * 80)

total_sl = 0
total_recovered = 0
for r in all_results:
    total_sl += r['sl_hits']
    total_recovered += r['sl_recovered']
    bars_str = f"{r['recovery_bars_avg']:.1f}" if r['recovery_bars_avg'] > 0 else "N/A"
    print(f"{r['grail']:<40} {r['sl_hits']:>8} {r['sl_recovered']:>12} {r['recovery_pct']:>7.1f}% {bars_str:>9}")

if total_sl > 0:
    print("-" * 80)
    print(f"{'TOTAL':<40} {total_sl:>8} {total_recovered:>12} {total_recovered/total_sl*100:>7.1f}%")
print()

# ============================================================
# ANÁLISIS POR SÍMBOLO
# ============================================================

print(f"\n{'='*80}")
print(f"COMPARATIVA POR SÍMBOLO")
print(f"{'='*80}")
print()

symbols_seen = {}
for r in all_results:
    sym = r['symbol']
    if sym not in symbols_seen:
        symbols_seen[sym] = {'S1': [], 'S2': [], 'S3': [], 'S4': []}
    for sk in ['S1', 'S2', 'S3', 'S4']:
        symbols_seen[sym][sk].append(r[sk])

for sym, strats in symbols_seen.items():
    print(f"\n  {sym}")
    print(f"  {'Estrategia':<25} {'WR%':>6} {'PnL/100':>9} {'WorstTr%':>9} {'LiqLev':>8}")
    print(f"  {'-'*60}")
    for sk, label in [('S1', 'S1: SL+TP Fijo'), ('S2', 'S2: Sin SL+MaxDur'), ('S3', 'S3: Sin SL+EMA'), ('S4', 'S4: SL 2x+TP')]:
        wr = np.mean([m['wr'] for m in strats[sk]])
        pnl = np.mean([m['pnl_per_100'] for m in strats[sk]])
        worst = min([m['worst_trade_pct'] for m in strats[sk]])
        liq = min([m['liq_leverage'] for m in strats[sk]])
        liq_str = f"{liq:.1f}x" if liq < 100 else "INF"
        print(f"  {label:<25} {wr:>5.1f}% {pnl:>8.2f}% {worst:>8.2f}% {liq_str:>8}")

# ============================================================
# DISTRIBUCIÓN DE EXITS
# ============================================================

print(f"\n{'='*80}")
print(f"DISTRIBUCIÓN DE TIPOS DE SALIDA (% del total)")
print(f"{'='*80}")
print()

for sk, label in [('S1', 'S1: SL+TP Fijo'), ('S2', 'S2: Sin SL, TP+MaxDur'),
                   ('S3', 'S3: Sin SL, TP+EMA'), ('S4', 'S4: SL Ancho 2x')]:
    exit_totals = defaultdict(int)
    total_trades = 0
    for r in all_results:
        for et, cnt in r[sk]['exit_types'].items():
            exit_totals[et] += cnt
            total_trades += cnt

    print(f"  {label}:")
    for et, cnt in sorted(exit_totals.items(), key=lambda x: -x[1]):
        print(f"    {et:<15} {cnt:>6} ({cnt/total_trades*100:.1f}%)")
    print()


# ============================================================
# ANÁLISIS DE TAIL RISK - LA CLAVE
# ============================================================

print(f"\n{'='*80}")
print(f"ANÁLISIS DE TAIL RISK (Riesgo de Cola) - EL PUNTO CLAVE")
print(f"{'='*80}")
print()

print("La pregunta clave: Sin SL, ¿cuál es el peor escenario REAL con los datos históricos?")
print()

for sk, label in [('S1', 'S1: SL+TP Fijo'), ('S2', 'S2: Sin SL+MaxDur'), ('S3', 'S3: Sin SL+EMA'), ('S4', 'S4: SL 2x')]:
    all_worst = [r[sk]['worst_trade_pct'] for r in all_results]
    all_liq = [r[sk]['liq_leverage'] for r in all_results]
    all_dd = [r[sk]['max_dd_potential_pct'] for r in all_results]

    worst_ever = min(all_worst)
    liq_min = min(all_liq)
    dd_max = max(all_dd)

    print(f"  {label}:")
    print(f"    Peor trade individual:       {worst_ever:>8.2f}%")
    liq_str = f"{liq_min:.1f}x" if liq_min < 100 else "INF"
    print(f"    Leverage para liquidación:    {liq_str:>8}")
    print(f"    DD potencial máx (consec):   {dd_max:>8.2f}%")

    # With 2x, 3x, 5x leverage
    for lev in [2, 3, 5]:
        worst_lev = worst_ever * lev
        survived = "SOBREVIVE" if abs(worst_lev) < 100 else "LIQUIDADO"
        print(f"    Con {lev}x leverage:            {worst_lev:>8.2f}% → {survived}")
    print()


# ============================================================
# CONCLUSIÓN FINAL
# ============================================================

print(f"\n{'='*80}")
print(f"CONCLUSIÓN BASADA EN DATOS REALES")
print(f"{'='*80}")
print()

# Compare average PnL
avg_pnl_s1 = np.mean([r['S1']['pnl_per_100'] for r in all_results])
avg_pnl_s2 = np.mean([r['S2']['pnl_per_100'] for r in all_results])
avg_pnl_s3 = np.mean([r['S3']['pnl_per_100'] for r in all_results])
avg_pnl_s4 = np.mean([r['S4']['pnl_per_100'] for r in all_results])

worst_s1 = min([r['S1']['worst_trade_pct'] for r in all_results])
worst_s2 = min([r['S2']['worst_trade_pct'] for r in all_results])
worst_s3 = min([r['S3']['worst_trade_pct'] for r in all_results])
worst_s4 = min([r['S4']['worst_trade_pct'] for r in all_results])

avg_wr_s1 = np.mean([r['S1']['wr'] for r in all_results])
avg_wr_s2 = np.mean([r['S2']['wr'] for r in all_results])
avg_wr_s3 = np.mean([r['S3']['wr'] for r in all_results])
avg_wr_s4 = np.mean([r['S4']['wr'] for r in all_results])

print(f"1. PROMEDIO: PnL por 100 trades")
print(f"   S1 (SL+TP Fijo):     {avg_pnl_s1:>8.2f}%")
print(f"   S2 (Sin SL+MaxDur):  {avg_pnl_s2:>8.2f}%")
print(f"   S3 (Sin SL+EMA):     {avg_pnl_s3:>8.2f}%")
print(f"   S4 (SL 2x+TP):       {avg_pnl_s4:>8.2f}%")
print()

print(f"2. WIN RATE promedio")
print(f"   S1: {avg_wr_s1:.1f}%  |  S2: {avg_wr_s2:.1f}%  |  S3: {avg_wr_s3:.1f}%  |  S4: {avg_wr_s4:.1f}%")
print()

print(f"3. PEOR TRADE INDIVIDUAL (tail risk)")
print(f"   S1: {worst_s1:.2f}%  |  S2: {worst_s2:.2f}%  |  S3: {worst_s3:.2f}%  |  S4: {worst_s4:.2f}%")
print()

# SL Recovery summary
if total_sl > 0:
    pct_rec = total_recovered / total_sl * 100
    print(f"4. SL RECOVERY: De {total_sl} trades donde se activó el SL,")
    print(f"   {total_recovered} ({pct_rec:.1f}%) habrían sido eventualmente rentables si se mantenían.")
    if pct_rec > 50:
        print(f"   --> DATO IMPORTANTE: Más de la mitad de los SL cortan ganadores prematuramente.")
        print(f"   --> PERO: sin SL, el peor trade fue {worst_s2:.2f}% vs {worst_s1:.2f}% con SL.")
    else:
        print(f"   --> El SL está bien calibrado: la mayoría de los SL protegen de pérdidas reales.")
print()

# Key insight calculation
risk_ratio_s1 = abs(worst_s1) / avg_pnl_s1 if avg_pnl_s1 > 0 else float('inf')
risk_ratio_s2 = abs(worst_s2) / avg_pnl_s2 if avg_pnl_s2 > 0 else float('inf')

print(f"5. RATIO RIESGO/RECOMPENSA EXTREMO (WorstTrade / AvgPnL100)")
print(f"   S1: {risk_ratio_s1:.1f}x  →  Necesitas {risk_ratio_s1:.0f} trades promedio para recuperar 1 peor trade")
print(f"   S2: {risk_ratio_s2:.1f}x  →  Necesitas {risk_ratio_s2:.0f} trades promedio para recuperar 1 peor trade")
print()

# The verdict
print(f"{'='*80}")
print(f"VEREDICTO FINAL")
print(f"{'='*80}")
print()

better_pnl = avg_pnl_s2 > avg_pnl_s1
worse_tail = abs(worst_s2) > abs(worst_s1) * 1.5

if better_pnl and worse_tail:
    print("SIN SL: Mayor PnL promedio PERO tail risk significativamente peor.")
    print("La pregunta NO es si el promedio mejora (sí mejora),")
    print("sino si SOBREVIVES al peor trade con leverage.")
    print()
    print(f"Con SL:  peor trade = {worst_s1:.2f}%  →  Con 3x lev = {worst_s1*3:.2f}%")
    print(f"Sin SL:  peor trade = {worst_s2:.2f}%  →  Con 3x lev = {worst_s2*3:.2f}%")
    liq_3x_s2 = abs(worst_s2 * 3) >= 100
    if liq_3x_s2:
        print(f"  --> Con 3x leverage y sin SL: LIQUIDACIÓN en el peor escenario")
    else:
        print(f"  --> Con 3x leverage: {abs(worst_s2*3):.1f}% de pérdida en peor caso")
elif better_pnl and not worse_tail:
    print("SIN SL: Mayor PnL promedio Y tail risk controlado.")
    print("En este caso específico, el SL actual puede estar siendo demasiado ajustado.")
    print("Recomendación: Considerar WIDE SL (2x) como safety net.")
elif not better_pnl:
    print("CON SL: El SL actual ya es óptimo - sin SL el PnL empeora.")
    print("El SL no solo protege sino que MEJORA la rentabilidad.")

print()
print("RECOMENDACIÓN PRÁCTICA:")
print("- NUNCA operar sin SL con leverage > 1x")
print("- Si el SL corta muchos ganadores (recovery > 50%): ensanchar SL")
print("- Wide SL (2x) como safety net: captura el edge de 'menos exits prematuros'")
print("  mientras mantiene protección contra eventos extremos")
print("- La mejor combinación: buenos entries + SL amplio + TP dinámico")

print(f"\n{'='*80}")
print(f"FIN DEL ANÁLISIS")
print(f"{'='*80}")
