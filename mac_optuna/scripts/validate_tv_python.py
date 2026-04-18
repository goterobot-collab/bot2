#!/usr/bin/env python3
"""
validate_tv_python.py
=====================
Framework de validación TV vs Python para los 84 grails TV.

Compara:
  1. WR "TV style" — señales puras sin SL/TP externo (como TV reporta)
  2. WR "Optuna style" — con SL/TP empírico (como V7 usa)
  3. Gap entre ambos → diagnóstico de causa (repainting / datos / comisiones)

Diagnóstico automático de repainting:
  - Si gap > 15pp → SOSPECHA de repainting
  - Verifica .shift(1) al final vs indicadores calculados con barra actual

Uso:
  python3 validate_tv_python.py                        # Top 10 TV grails
  python3 validate_tv_python.py --all                  # Los 84 grails
  python3 validate_tv_python.py --strategy TV_JokerTrailing --symbol CYS/USDT:USDT --tf 1h
  python3 validate_tv_python.py --top 10               # Top N por WR
  python3 validate_tv_python.py --export report.csv    # Exportar resultados
"""
import sys
import os
import json
import sqlite3
import warnings
import argparse
import importlib
import importlib.util
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# ─── PATHS ───────────────────────────────────────────────────────────────────
SCRIPT_DIR       = Path(__file__).resolve().parent
ESTRATEGIAS_DIR  = SCRIPT_DIR.parent
BATCHES_DIR      = ESTRATEGIAS_DIR / "strategies_tv2_batches"
DATA_DIR         = ESTRATEGIAS_DIR / "data"
GRAILS_FILE      = DATA_DIR / "optuna_v7_progress.json"
DB_PATH          = Path("/Users/sabrina/CLAUDE CODE/data/activos_binance.db")
OUTPUT_DIR       = DATA_DIR / "tv_validation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Agregar batches al path
sys.path.insert(0, str(ESTRATEGIAS_DIR))
sys.path.insert(0, str(BATCHES_DIR))

# ─── CONSTANTES ──────────────────────────────────────────────────────────────
COMMISSION     = 0.001   # 0.1% por lado (Binance futures taker)
SLIPPAGE       = 0.0005  # 0.05%
GAP_THRESHOLD  = 15.0    # pp — sospecha de repainting si gap > esto
MIN_TRADES_TV  = 5       # mínimo trades para comparación válida

# ─── LOAD GRAILS ─────────────────────────────────────────────────────────────

def load_tv_grails(top_n=None):
    """Carga los 84 grails TV de optuna_v7_progress.json."""
    if not GRAILS_FILE.exists():
        raise FileNotFoundError(f"No se encontró {GRAILS_FILE}")
    with open(GRAILS_FILE) as f:
        data = json.load(f)
    all_grails = data.get('grails', [])
    tv_grails = [g for g in all_grails if g.get('strategy', '').startswith('TV_')]
    # Ordenar por test_wr descendente
    tv_grails.sort(key=lambda x: x.get('test_wr', 0), reverse=True)
    if top_n:
        tv_grails = tv_grails[:top_n]
    return tv_grails


def load_single_grail(strategy, symbol, tf):
    """Carga un grail específico."""
    with open(GRAILS_FILE) as f:
        data = json.load(f)
    all_grails = data.get('grails', [])
    for g in all_grails:
        if (g.get('strategy') == strategy and
                g.get('symbol') == symbol and
                g.get('timeframe') == tf):
            return g
    return None


# ─── CANDLE LOADER ───────────────────────────────────────────────────────────

def load_candles(symbol, tf):
    """Carga candles de activos_binance.db."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB no encontrada: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql(
        "SELECT ts, open, high, low, close, volume FROM candles "
        "WHERE symbol=? AND timeframe=? ORDER BY ts",
        conn, params=(symbol, tf))
    conn.close()
    if len(df) < 50:
        return None
    df['ts_raw'] = pd.to_numeric(df['ts'], errors='coerce')
    df.dropna(subset=['ts_raw'], inplace=True)
    df['datetime'] = pd.to_datetime(df['ts_raw'], unit='ms')
    df.set_index('datetime', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    return df


def resample_df(df, target_tf):
    """Resamplea 5m → 15m / 4h / 1d."""
    tf_map = {'15m': '15min', '4h': '4h', '1d': '1D'}
    rule = tf_map.get(target_tf)
    if not rule:
        return None
    res = df.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna(subset=['close'])
    return res if len(res) >= 50 else None


def get_candles(symbol, tf):
    """Obtiene candles para cualquier TF soportado."""
    if tf in ('5m', '1h'):
        return load_candles(symbol, tf)
    elif tf in ('15m', '4h', '1d'):
        base = load_candles(symbol, '5m')
        if base is None:
            return None
        return resample_df(base, tf)
    return None


# ─── STRATEGY LOADER ─────────────────────────────────────────────────────────

_strategy_cache = {}

def get_strategy_fn(strategy_name):
    """
    Busca la función gen_<X> para una estrategia TV en los batches.
    Cachea el resultado para evitar re-imports.
    """
    if strategy_name in _strategy_cache:
        return _strategy_cache[strategy_name]

    # Buscar en todos los batches 80-165
    for batch_file in sorted(BATCHES_DIR.glob("strategies_tv2_batch*.py")):
        if '.bak' in batch_file.name:
            continue
        spec = importlib.util.spec_from_file_location(
            batch_file.stem, str(batch_file))
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            continue
        # Buscar en STRATEGY_EXPORT del módulo
        export = getattr(mod, 'STRATEGY_EXPORT', {})
        if strategy_name in export:
            fn = export[strategy_name].get('gen')
            _strategy_cache[strategy_name] = fn
            return fn
    # No encontrado
    _strategy_cache[strategy_name] = None
    return None


# ─── BACKTEST ENGINES ────────────────────────────────────────────────────────

def backtest_tv_style(df, signals):
    """
    Simula trades AL ESTILO TV:
    - Sin comisiones (TV default = 0)
    - Sin slippage (TV default = 0)
    - Entry en OPEN de la barra siguiente a la señal (TV default: fill_on_bar_close=False)
    - Exit: primera señal de salida (no SL/TP externo)

    Devuelve dict con WR, n_trades, pnl_raw (sin costos).
    """
    if signals is None or len(signals) == 0:
        return {'wr': 0.0, 'n_trades': 0, 'pnl': 0.0, 'trades': []}

    sig = signals.values
    opens   = df['open'].values
    highs   = df['high'].values
    lows    = df['low'].values
    closes  = df['close'].values
    n       = len(df)

    trades = []
    in_trade = False
    entry_price = 0.0
    entry_bar   = 0
    side        = 0  # 1=long, -1=short

    for i in range(1, n):
        if not in_trade:
            prev_sig = sig[i - 1]
            if prev_sig == 1 and side != 1:
                # Entrada LONG: open de la barra actual
                in_trade    = True
                side        = 1
                entry_price = opens[i]
                entry_bar   = i
            elif prev_sig == -1 and side != -1:
                # Entrada SHORT: open de la barra actual
                in_trade    = True
                side        = -1
                entry_price = opens[i]
                entry_bar   = i
        else:
            # Verificar señal de salida
            current_sig = sig[i]
            exit_now = False
            if side == 1 and current_sig != 1:
                exit_now = True
            elif side == -1 and current_sig != -1:
                exit_now = True

            if exit_now:
                exit_price = opens[i]  # TV: fill on bar open
                if side == 1:
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price * 100
                trades.append({
                    'side':         'LONG' if side == 1 else 'SHORT',
                    'entry_bar':    entry_bar,
                    'exit_bar':     i,
                    'entry_price':  entry_price,
                    'exit_price':   exit_price,
                    'pnl_pct':      pnl_pct,
                    'win':          pnl_pct > 0,
                    'hold_bars':    i - entry_bar,
                })
                in_trade = False
                side     = 0

    if not trades:
        return {'wr': 0.0, 'n_trades': 0, 'pnl': 0.0, 'trades': []}

    n_wins  = sum(1 for t in trades if t['win'])
    n_total = len(trades)
    wr      = n_wins / n_total * 100
    pnl     = sum(t['pnl_pct'] for t in trades)
    return {'wr': round(wr, 1), 'n_trades': n_total, 'pnl': round(pnl, 2), 'trades': trades}


def backtest_with_sl_tp(df, signals, sl_pct, tp_pct, max_dur_bars=None):
    """
    Simula trades con SL/TP empírico (estilo Optuna V7):
    - Comisión 0.1% por lado
    - Slippage 0.05%
    - Entry en OPEN de la barra siguiente
    - Exit: SL hit / TP hit / señal contraria / max_dur

    Devuelve dict con WR, n_trades, pnl_neto.
    """
    if signals is None or len(signals) == 0:
        return {'wr': 0.0, 'n_trades': 0, 'pnl': 0.0}

    sig     = signals.values
    opens   = df['open'].values
    highs   = df['high'].values
    lows    = df['low'].values
    n       = len(df)

    total_cost = (COMMISSION + SLIPPAGE) * 2  # round-trip

    trades  = []
    in_trade = False
    entry_price = 0.0
    entry_bar   = 0
    side        = 0

    for i in range(1, n):
        if not in_trade:
            prev_sig = sig[i - 1]
            if prev_sig == 1 and side != 1:
                in_trade    = True
                side        = 1
                entry_price = opens[i] * (1 + SLIPPAGE)
                entry_bar   = i
            elif prev_sig == -1 and side != -1:
                in_trade    = True
                side        = -1
                entry_price = opens[i] * (1 - SLIPPAGE)
                entry_bar   = i
        else:
            # Calcular precio de SL y TP
            if side == 1:
                sl_price = entry_price * (1 - sl_pct)
                tp_price = entry_price * (1 + tp_pct)
                # Intrabar: TV asume open→high→low→close si open > mid(h,l), else open→low→high→close
                if opens[i] > (highs[i] + lows[i]) / 2:
                    sl_hit = lows[i] <= sl_price
                    tp_hit = highs[i] >= tp_price
                else:
                    sl_hit = lows[i] <= sl_price
                    tp_hit = highs[i] >= tp_price

                if sl_hit or tp_hit:
                    exit_price = tp_price if tp_hit else sl_price
                    pnl_raw    = (exit_price - entry_price) / entry_price
                    pnl_net    = (pnl_raw - total_cost) * 100
                    trades.append({'win': pnl_net > 0, 'pnl_net': pnl_net, 'reason': 'TP' if tp_hit else 'SL'})
                    in_trade = False; side = 0
                    continue

            elif side == -1:
                sl_price = entry_price * (1 + sl_pct)
                tp_price = entry_price * (1 - tp_pct)
                if opens[i] > (highs[i] + lows[i]) / 2:
                    sl_hit = highs[i] >= sl_price
                    tp_hit = lows[i] <= tp_price
                else:
                    sl_hit = highs[i] >= sl_price
                    tp_hit = lows[i] <= tp_price

                if sl_hit or tp_hit:
                    exit_price = tp_price if tp_hit else sl_price
                    pnl_raw    = (entry_price - exit_price) / entry_price
                    pnl_net    = (pnl_raw - total_cost) * 100
                    trades.append({'win': pnl_net > 0, 'pnl_net': pnl_net, 'reason': 'TP' if tp_hit else 'SL'})
                    in_trade = False; side = 0
                    continue

            # Max duration
            if max_dur_bars and (i - entry_bar) >= max_dur_bars:
                exit_price = opens[i]
                if side == 1:
                    pnl_raw = (exit_price - entry_price) / entry_price
                else:
                    pnl_raw = (entry_price - exit_price) / entry_price
                pnl_net = (pnl_raw - total_cost) * 100
                trades.append({'win': pnl_net > 0, 'pnl_net': pnl_net, 'reason': 'MAX_DUR'})
                in_trade = False; side = 0
                continue

            # Señal contraria
            current_sig = sig[i]
            if (side == 1 and current_sig == -1) or (side == -1 and current_sig == 1):
                exit_price = opens[i]
                if side == 1:
                    pnl_raw = (exit_price - entry_price) / entry_price
                else:
                    pnl_raw = (entry_price - exit_price) / entry_price
                pnl_net = (pnl_raw - total_cost) * 100
                trades.append({'win': pnl_net > 0, 'pnl_net': pnl_net, 'reason': 'REV'})
                in_trade = False; side = 0

    if not trades:
        return {'wr': 0.0, 'n_trades': 0, 'pnl': 0.0}

    n_wins  = sum(1 for t in trades if t['win'])
    n_total = len(trades)
    wr      = n_wins / n_total * 100
    pnl     = sum(t['pnl_net'] for t in trades)
    return {'wr': round(wr, 1), 'n_trades': n_total, 'pnl': round(pnl, 2)}


# ─── REPAINTING DIAGNOSIS ─────────────────────────────────────────────────────

def diagnose_repainting_risk(strategy_name, grail):
    """
    Heurística de riesgo de repainting basada en:
    1. Tipo de indicadores usados (ATR, BB rolling = OK; HTF security() = HIGH RISK)
    2. Gap entre train_wr y test_wr del CPCV
    3. WR histórico vs WR reciente (drift)

    Retorna: ('LOW' | 'MEDIUM' | 'HIGH', explicación)
    """
    risk_factors = []
    risk_level   = 'LOW'

    # Factor 1: ¿El grail viene de solo una ventana CV?
    cv_windows = grail.get('cv_windows', [])
    if len(cv_windows) == 1:
        risk_factors.append("Solo 1 ventana CV (WF_70_30) — validación débil")
        risk_level = 'MEDIUM'

    # Factor 2: Gap WR entre ventanas CPCV
    wr_gaps = [abs(w.get('wr_gap', 0)) for w in cv_windows]
    if any(g > 30 for g in wr_gaps):
        risk_factors.append(f"Gap train/test > 30pp en alguna ventana (gaps: {wr_gaps})")
        risk_level = 'HIGH'
    elif any(g > 20 for g in wr_gaps):
        risk_factors.append(f"Gap train/test > 20pp (gaps: {wr_gaps})")
        if risk_level == 'LOW':
            risk_level = 'MEDIUM'

    # Factor 3: WR muy alto con pocos trades → overfitting
    full = grail.get('full', {})
    total_trades = full.get('trades', 0)
    test_wr = grail.get('test_wr', 0)
    if test_wr > 85 and total_trades < 20:
        risk_factors.append(f"WR={test_wr}% con solo {total_trades} trades totales")
        risk_level = 'HIGH'

    # Factor 4: RR bajo (TP < SL) — puede inflar WR artificialmente
    sl = grail.get('sl', 1)
    tp = grail.get('tp', 1)
    rr = tp / sl if sl > 0 else 0
    if rr < 0.5:
        risk_factors.append(f"RR={rr:.2f} — TP/SL bajo, WR inflado por trade asimétrico")
        if risk_level == 'LOW':
            risk_level = 'MEDIUM'

    # Factor 5: Estrategias con lookback variable o HTF son más propensas
    htf_indicators = ['Ichimoku', 'VWAP_Fibo', 'AllInOne', 'FlawlessVictory', 'LorentzianKNN']
    if any(ind in strategy_name for ind in htf_indicators):
        risk_factors.append(f"Indicador complejo/HTF — riesgo inherente de lookahead")
        if risk_level == 'LOW':
            risk_level = 'MEDIUM'

    if not risk_factors:
        risk_factors.append("Sin factores de riesgo detectados")

    return risk_level, '; '.join(risk_factors)


# ─── CORE VALIDATION ─────────────────────────────────────────────────────────

def validate_single(grail):
    """
    Valida UN grail TV:
    1. Carga candles
    2. Genera señales con params del grail
    3. Backtest TV style (sin SL/TP)
    4. Backtest con SL/TP del grail
    5. Compara vs WR reportado en Optuna
    6. Diagnóstico
    """
    strategy = grail['strategy']
    symbol   = grail['symbol']
    tf       = grail['timeframe']
    params   = grail.get('best_params', {})
    optuna_wr = grail.get('test_wr', 0)
    sl_pct   = grail.get('sl', 0.03)
    tp_pct   = grail.get('tp', 0.02)
    max_dur  = grail.get('max_dur_bars', None)

    result = {
        'strategy':       strategy,
        'symbol':         symbol,
        'timeframe':      tf,
        'optuna_wr':      optuna_wr,
        'optuna_sl':      round(sl_pct * 100, 2),
        'optuna_tp':      round(tp_pct * 100, 2),
        'tv_style_wr':    None,
        'tv_style_trades': None,
        'sl_tp_wr':       None,
        'sl_tp_trades':   None,
        'gap_tv_vs_optuna': None,
        'gap_tv_vs_sltp':   None,
        'repainting_risk':  None,
        'repainting_detail': None,
        'status':          'OK',
        'error':           None,
    }

    # Cargar candles
    df = get_candles(symbol, tf)
    if df is None or len(df) < 100:
        result['status'] = 'NO_DATA'
        result['error']  = f"Datos insuficientes para {symbol}/{tf}"
        return result

    # Cargar función generadora
    gen_fn = get_strategy_fn(strategy)
    if gen_fn is None:
        result['status'] = 'NO_GEN_FN'
        result['error']  = f"No se encontró gen_fn para {strategy}"
        return result

    # Generar señales con best_params del grail
    try:
        signals = gen_fn(df, **params)
    except Exception as e:
        result['status'] = 'GEN_ERROR'
        result['error']  = f"Error al generar señales: {e}"
        return result

    if signals is None:
        result['status'] = 'NULL_SIGNALS'
        result['error']  = "gen_fn retornó None"
        return result

    # ── Backtest 1: TV Style (sin SL/TP, sin comisiones) ────────────────────
    tv_res = backtest_tv_style(df, signals)
    result['tv_style_wr']     = tv_res['wr']
    result['tv_style_trades'] = tv_res['n_trades']

    # ── Backtest 2: Con SL/TP del grail ─────────────────────────────────────
    sltp_res = backtest_with_sl_tp(df, signals, sl_pct, tp_pct, max_dur)
    result['sl_tp_wr']     = sltp_res['wr']
    result['sl_tp_trades'] = sltp_res['n_trades']

    # ── Gaps ─────────────────────────────────────────────────────────────────
    if tv_res['n_trades'] >= MIN_TRADES_TV:
        result['gap_tv_vs_optuna'] = round(tv_res['wr'] - optuna_wr, 1)
        result['gap_tv_vs_sltp']   = round(tv_res['wr'] - sltp_res['wr'], 1) if sltp_res['n_trades'] > 0 else None

    # ── Diagnóstico de repainting ────────────────────────────────────────────
    risk, detail = diagnose_repainting_risk(strategy, grail)
    result['repainting_risk']   = risk
    result['repainting_detail'] = detail

    # ── Flag de sospecha ─────────────────────────────────────────────────────
    if result['gap_tv_vs_sltp'] is not None and abs(result['gap_tv_vs_sltp']) > GAP_THRESHOLD:
        result['status'] = 'REPAINTING_SUSPECT'

    return result


# ─── REPORT ──────────────────────────────────────────────────────────────────

def print_report(results):
    """Imprime reporte formateado en consola."""
    print("\n" + "=" * 110)
    print("VALIDACIÓN TV vs PYTHON — FRAMEWORK DE PARIDAD")
    print(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 110)

    # Header
    hdr = (f"{'Estrategia':<38} {'Símbolo':<20} {'TF':<5} "
           f"{'TV%':>7} {'TV_n':>6} {'SL/TP%':>7} {'SL_n':>6} "
           f"{'Optuna%':>8} {'Gap_SL':>8} {'Risk':<8} {'Status'}")
    print(hdr)
    print("-" * 110)

    ok_count     = 0
    suspect_count = 0
    error_count  = 0

    for r in results:
        tv_wr   = f"{r['tv_style_wr']:.1f}" if r['tv_style_wr'] is not None else "N/A"
        tv_n    = str(r['tv_style_trades'] or "N/A")
        sltp_wr = f"{r['sl_tp_wr']:.1f}" if r['sl_tp_wr'] is not None else "N/A"
        sltp_n  = str(r['sl_tp_trades'] or "N/A")
        opt_wr  = f"{r['optuna_wr']:.1f}"
        gap     = f"{r['gap_tv_vs_sltp']:+.1f}" if r['gap_tv_vs_sltp'] is not None else "N/A"
        risk    = r['repainting_risk'] or "N/A"
        status  = r['status']

        # Colorear por status
        flag = ""
        if status == 'REPAINTING_SUSPECT':
            flag = " ⚠"
            suspect_count += 1
        elif status == 'OK':
            ok_count += 1
        else:
            flag = " ✗"
            error_count += 1

        line = (f"{r['strategy']:<38} {r['symbol']:<20} {r['timeframe']:<5} "
                f"{tv_wr:>7} {tv_n:>6} {sltp_wr:>7} {sltp_n:>6} "
                f"{opt_wr:>8} {gap:>8} {risk:<8} {status}{flag}")
        print(line)

    print("=" * 110)
    print(f"RESUMEN: {len(results)} validados | {ok_count} OK | {suspect_count} sospechosos | {error_count} errores")
    print()

    # Sospechosos — detalle
    suspects = [r for r in results if r['status'] == 'REPAINTING_SUSPECT']
    if suspects:
        print("── SOSPECHOSOS DE REPAINTING ─────────────────────────────────────")
        for r in suspects:
            print(f"\n  {r['strategy']} × {r['symbol']} × {r['timeframe']}")
            print(f"    TV style WR:  {r['tv_style_wr']}% ({r['tv_style_trades']} trades)")
            print(f"    SL/TP WR:     {r['sl_tp_wr']}% ({r['sl_tp_trades']} trades)")
            print(f"    Optuna WR:    {r['optuna_wr']}%")
            print(f"    Gap TV-SL/TP: {r['gap_tv_vs_sltp']:+.1f}pp")
            print(f"    Risk:         {r['repainting_risk']}")
            print(f"    Detalle:      {r['repainting_detail']}")
        print()


def save_results(results, export_path=None):
    """Guarda resultados como JSON y opcionalmente como CSV."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    # JSON
    json_path = OUTPUT_DIR / f"validation_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'generated': datetime.now().isoformat(),
            'n_total':   len(results),
            'results':   results,
        }, f, indent=2)
    print(f"Guardado: {json_path}")

    # CSV
    if export_path:
        df = pd.DataFrame(results)
        df.to_csv(export_path, index=False)
        print(f"CSV exportado: {export_path}")

    return json_path


# ─── ANÁLISIS GLOBAL ─────────────────────────────────────────────────────────

def analyze_global_gap(results):
    """
    Análisis estadístico del gap TV vs Python.
    Identifica patrones por estrategia y símbolo.
    """
    valid = [r for r in results if r['gap_tv_vs_sltp'] is not None]
    if not valid:
        print("No hay datos válidos para análisis global.")
        return

    gaps = [r['gap_tv_vs_sltp'] for r in valid]
    print("\n── ANÁLISIS GLOBAL DE GAPS (TV_WR - SL/TP_WR) ──────────────────")
    print(f"  Media:    {np.mean(gaps):+.1f}pp")
    print(f"  Mediana:  {np.median(gaps):+.1f}pp")
    print(f"  Std:      {np.std(gaps):.1f}pp")
    print(f"  Min/Max:  {min(gaps):+.1f} / {max(gaps):+.1f}pp")
    print(f"  >15pp (sospechosos): {sum(1 for g in gaps if abs(g) > 15)} / {len(gaps)}")
    print()

    # Por estrategia
    strat_gaps = {}
    for r in valid:
        strat = r['strategy']
        strat_gaps.setdefault(strat, []).append(r['gap_tv_vs_sltp'])

    print("  Gap promedio por estrategia:")
    for strat, g_list in sorted(strat_gaps.items(), key=lambda x: abs(np.mean(x[1])), reverse=True)[:10]:
        print(f"    {strat:<40} avg_gap={np.mean(g_list):+.1f}pp  n={len(g_list)}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Validación TV vs Python')
    parser.add_argument('--all',          action='store_true',  help='Validar todos los 84 grails TV')
    parser.add_argument('--top',          type=int, default=10, help='Top N grails por WR (default: 10)')
    parser.add_argument('--strategy',     type=str,             help='Estrategia específica')
    parser.add_argument('--symbol',       type=str,             help='Símbolo específico')
    parser.add_argument('--tf',           type=str,             help='Timeframe específico')
    parser.add_argument('--export',       type=str,             help='Exportar CSV a este path')
    parser.add_argument('--no-save',      action='store_true',  help='No guardar resultados en disco')
    args = parser.parse_args()

    # ── Selección de grails ──────────────────────────────────────────────────
    if args.strategy and args.symbol and args.tf:
        grail = load_single_grail(args.strategy, args.symbol, args.tf)
        if grail is None:
            print(f"ERROR: No se encontró grail para {args.strategy} × {args.symbol} × {args.tf}")
            sys.exit(1)
        grails = [grail]
    elif args.all:
        grails = load_tv_grails()
        print(f"Cargados {len(grails)} grails TV para validación completa.")
    else:
        top_n  = args.top
        grails = load_tv_grails(top_n=top_n)
        print(f"Validando TOP {len(grails)} grails TV por WR.")

    if not grails:
        print("No se encontraron grails TV. Verificar GRAILS_FILE.")
        sys.exit(1)

    # ── Validar ─────────────────────────────────────────────────────────────
    print(f"\nValidando {len(grails)} grails...")
    results = []
    for i, grail in enumerate(grails, 1):
        strat  = grail['strategy']
        sym    = grail['symbol']
        tf     = grail['timeframe']
        print(f"  [{i:02d}/{len(grails)}] {strat:<38} {sym:<25} {tf}...", end='', flush=True)
        r = validate_single(grail)
        results.append(r)
        status_str = r['status']
        if r['tv_style_wr'] is not None:
            status_str += f" TV={r['tv_style_wr']:.1f}%"
        print(f" {status_str}")

    # ── Report ───────────────────────────────────────────────────────────────
    print_report(results)
    analyze_global_gap(results)

    # ── Guardar ──────────────────────────────────────────────────────────────
    if not args.no_save:
        save_results(results, export_path=args.export)


if __name__ == '__main__':
    main()
