#!/usr/bin/env python3
"""
validate_tv_python.py — Diagnóstico de brechas TradingView → Python

Compara la implementación Python contra lo que TV debería producir para
una estrategia dada en un activo/TF específico.

Corre la estrategia en dos modos:
  A) "TV mode": sin SL/TP externo, con exits naturales de la estrategia
  B) "Optuna mode": con SL/TP empírico calibrado por MAE/MFE
Luego reporta las brechas.

Uso:
  python3 validate_tv_python.py --strategy TV_BB_MeanRev_Simple --symbol BEAT/USDT:USDT --tf 1h
  python3 validate_tv_python.py --strategy TV_VWAP_Bounce_Cross --symbol CYS/USDT:USDT --tf 5m
  python3 validate_tv_python.py --all-top                          # Corre los top 5 grails
"""

import argparse
import sqlite3
import sys
import os
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
DB_PATH = "/Users/sabrina/CLAUDE CODE/data/activos_binance.db"
BATCHES_DIR = "/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches"
sys.path.insert(0, BATCHES_DIR)

COMMISSION = 0.001
SLIPPAGE   = 0.0005
COST       = COMMISSION + SLIPPAGE

# ── Top grails a validar (estrategia × activo × TF) ───────────────────────────
TOP_GRAILS = [
    ("TV_BB_MeanRev_Simple",    "BEAT/USDT:USDT",  "1h"),
    ("TV_VWAP_Bounce_Cross",    "BEAT/USDT:USDT",  "5m"),
    ("TV_VWAP_Bounce_Cross",    "CYS/USDT:USDT",   "5m"),
    ("TV_Spike_Reversion_70",   "DEGO/USDT:USDT",  "1h"),
    ("TV_StochRSI_SuperTrend",  "TRX/USDT:USDT",   "5m"),
]

# ── Carga de datos ──────────────────────────────────────────────────────────────

def load_candles(symbol, tf):
    """Carga velas de la DB. Filtra timestamps corruptos (pre-2010)."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA query_only=ON")
    df = pd.read_sql(
        "SELECT ts,open,high,low,close,volume FROM candles "
        "WHERE symbol=? AND timeframe=? ORDER BY ts",
        conn, params=(symbol, tf))
    conn.close()
    if len(df) == 0:
        return None
    df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
    # FIX BRECHA 6: filtrar timestamps corruptos (antes de 2010-01-01)
    df = df[df['ts'] > 1_262_304_000_000]   # 2010-01-01 en ms
    df.dropna(subset=['ts'], inplace=True)
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    return df


# ── Backtests ───────────────────────────────────────────────────────────────────

def backtest_tv_mode(df, signals):
    """
    Backtest 'TV mode': sin SL/TP externo.
    Replica cómo TV contaría trades: cierra cuando la señal cambia.
    Entrada: OPEN de la barra siguiente a la señal.
    """
    sig = signals.values
    opens = df['open'].values
    n = min(len(sig), len(opens))
    trades = []
    pos = 0; ep = 0.; ei = 0

    for i in range(1, n):
        s = int(sig[i - 1])
        p = opens[i]

        if pos == 0:
            if s == 1:
                ep = p * (1 + COST); ei = i; pos = 1
            elif s == -1:
                ep = p * (1 - COST); ei = i; pos = -1

        elif pos == 1:
            if s == -1 or s == 0:
                xp = p * (1 - COST)
                pnl = (xp - ep) / ep
                trades.append({'pnl': pnl, 'dur_bars': i - ei,
                                'win': pnl > 0, 'exit': 'SIG'})
                pos = 0
                if s == -1:
                    ep = p * (1 - COST); ei = i; pos = -1

        elif pos == -1:
            if s == 1 or s == 0:
                xp = p * (1 + COST)
                pnl = (ep - xp) / ep
                trades.append({'pnl': pnl, 'dur_bars': i - ei,
                                'win': pnl > 0, 'exit': 'SIG'})
                pos = 0
                if s == 1:
                    ep = p * (1 + COST); ei = i; pos = 1

    return trades


def backtest_with_sl_tp(df, signals, sl_pct, tp_pct, max_dur_bars=None):
    """Backtest con SL/TP externo (modo Optuna)."""
    sig = signals.values
    opens = df['open'].values
    highs = df['high'].values
    lows  = df['low'].values
    n = len(df)
    trades = []
    pos = 0; ep = 0.; ei = 0

    for i in range(1, n):
        s = sig[i - 1]; p = opens[i]
        if pos == 0:
            if s == 1:
                ep = p * (1 + COST); ei = i; pos = 1
            elif s == -1:
                ep = p * (1 - COST); ei = i; pos = -1

        elif pos == 1:
            if (lows[i] - ep) / ep <= -sl_pct:
                xp = ep * (1 - sl_pct)
                trades.append({'pnl': (xp - ep) / ep, 'dur_bars': i - ei,
                                'win': False, 'exit': 'SL'})
                pos = 0; continue
            if (highs[i] - ep) / ep >= tp_pct:
                xp = ep * (1 + tp_pct)
                trades.append({'pnl': (xp - ep) / ep, 'dur_bars': i - ei,
                                'win': True, 'exit': 'TP'})
                pos = 0; continue
            if max_dur_bars and (i - ei) >= max_dur_bars:
                xp = p * (1 - COST)
                trades.append({'pnl': (xp - ep) / ep, 'dur_bars': i - ei,
                                'win': (xp - ep) / ep > 0, 'exit': 'DUR'})
                pos = 0; continue
            if s == -1 or s == 0:
                xp = p * (1 - COST)
                trades.append({'pnl': (xp - ep) / ep, 'dur_bars': i - ei,
                                'win': (xp - ep) / ep > 0, 'exit': 'SIG'})
                pos = 0

        elif pos == -1:
            if (ep - highs[i]) / ep <= -sl_pct:
                xp = ep * (1 - sl_pct)
                trades.append({'pnl': (xp - ep) / ep, 'dur_bars': i - ei,
                                'win': False, 'exit': 'SL'})
                pos = 0; continue
            if (ep - lows[i]) / ep >= tp_pct:
                xp = ep * (1 + tp_pct)
                trades.append({'pnl': (xp - ep) / ep, 'dur_bars': i - ei,
                                'win': True, 'exit': 'TP'})
                pos = 0; continue
            if max_dur_bars and (i - ei) >= max_dur_bars:
                xp = p * (1 + COST)
                trades.append({'pnl': (ep - xp) / ep, 'dur_bars': i - ei,
                                'win': (ep - xp) / ep > 0, 'exit': 'DUR'})
                pos = 0; continue
            if s == 1 or s == 0:
                xp = p * (1 + COST)
                trades.append({'pnl': (ep - xp) / ep, 'dur_bars': i - ei,
                                'win': (ep - xp) / ep > 0, 'exit': 'SIG'})
                pos = 0

    return trades


# ── Métricas ────────────────────────────────────────────────────────────────────

def metrics(trades, label=""):
    if not trades:
        return {"label": label, "trades": 0, "wr": 0, "pnl": 0}
    pnls = np.array([t['pnl'] for t in trades])
    wins = (pnls > 0).sum()
    exits = {}
    for t in trades:
        exits[t.get('exit', '?')] = exits.get(t.get('exit', '?'), 0) + 1
    durs = np.array([t['dur_bars'] for t in trades])
    return {
        "label":     label,
        "trades":    len(trades),
        "wr":        round(wins / len(trades) * 100, 1),
        "pnl_pct":   round(pnls.sum() * 100, 2),
        "avg_dur":   round(durs.mean(), 1),
        "p95_dur":   int(np.percentile(durs, 95)),
        "mae_p95":   round(np.percentile([abs(t.get('pnl', 0)) for t in trades if t['pnl'] < 0] or [0], 95), 4),
        "exits":     exits,
    }


# ── Diagnóstico de brechas específicas ────────────────────────────────────────

def diagnose_rsi_impl(df):
    """Brecha 1: RSI ewm vs RMA Wilder."""
    c = df['close']
    period = 14
    delta = c.diff()

    # Python batch163 style (ewm)
    gain_ewm = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    loss_ewm = (-delta.clip(upper=0)).ewm(com=period-1, adjust=False).mean()
    rs_ewm = np.where(loss_ewm == 0, 100.0, gain_ewm / loss_ewm)
    rsi_ewm = pd.Series(100.0 - 100.0 / (1.0 + rs_ewm), index=df.index)

    # batch175 / Pine style (RMA, seed=mean)
    def rma(src_a, n):
        out = np.full(len(src_a), np.nan)
        src_clean = src_a[~np.isnan(src_a)]
        if len(src_clean) < n:
            return out
        start = next(i for i in range(len(src_a)) if not np.isnan(src_a[i]))
        out[start + n - 1] = np.nanmean(src_a[start:start+n])
        alpha = 1.0 / n
        for i in range(start + n, len(src_a)):
            out[i] = alpha * src_a[i] + (1 - alpha) * out[i-1]
        return out

    g_arr = delta.clip(lower=0).values
    l_arr = (-delta.clip(upper=0)).values
    gain_rma = rma(g_arr, period)
    loss_rma = rma(l_arr, period)
    rs_rma = np.where(loss_rma > 0, gain_rma / loss_rma, 100.0)
    rsi_rma = pd.Series(100.0 - 100.0 / (1.0 + rs_rma), index=df.index)

    mask = ~(rsi_ewm.isna() | rsi_rma.isna())
    diff = (rsi_ewm - rsi_rma).abs()[mask]
    return {
        "brecha": "RSI ewm vs RMA",
        "mean_diff_points": round(diff.mean(), 2),
        "max_diff_points":  round(diff.max(), 2),
        "p95_diff_points":  round(diff.quantile(0.95), 2),
        "early_bars_diff":  round(diff.iloc[:50].mean(), 2) if len(diff) >= 50 else None,
        "conclusion": "ewm y RMA convergen pero los primeros 50 bars difieren significativamente"
    }


def diagnose_bb_std(df, bb_period=20):
    """Brecha 2: BB std ddof=1 vs ddof=0."""
    c = df['close']
    std_sample = c.rolling(bb_period).std()       # ddof=1 (pandas default)
    std_pop    = c.rolling(bb_period).std(ddof=0) # ddof=0 (Pine)
    ratio = (std_sample / std_pop).dropna()
    return {
        "brecha": "BB std ddof=1 vs ddof=0",
        "mean_ratio": round(ratio.mean(), 5),
        "expected_ratio": round(np.sqrt(bb_period / (bb_period - 1)), 5),
        "effect_on_band_width": f"+{(ratio.mean()-1)*100:.2f}% wider than Pine",
        "signals_impact": "Python genera menos señales BB que TV (bandas más anchas)"
    }


def diagnose_vwap_reset(df):
    """Brecha 3: rolling VWAP vs daily reset."""
    c = df['close']
    v = df['volume']
    h = df['high']
    l = df['low']
    hlc3 = (h + l + c) / 3

    # Rolling VWAP (Python batch163)
    vwap_len = 20
    vwma_n = (hlc3 * v).rolling(vwap_len).sum()
    vwma_d = v.rolling(vwap_len).sum()
    vwap_rolling = np.where(vwma_d > 0, vwma_n / vwma_d, c.values)
    vwap_rolling = pd.Series(vwap_rolling, index=df.index)

    # Daily reset VWAP (como Pine)
    vwap_daily = np.full(len(df), np.nan)
    cum_vol, cum_hv = 0.0, 0.0
    for i, (idx, row) in enumerate(df.iterrows()):
        is_new_day = (i == 0) or (idx.date() != df.index[i-1].date())
        if is_new_day:
            cum_vol = max(row['volume'], 1e-10)
            cum_hv  = hlc3.iloc[i] * cum_vol
        else:
            cum_vol += row['volume']
            cum_hv  += hlc3.iloc[i] * row['volume']
        vwap_daily[i] = cum_hv / cum_vol if cum_vol > 0 else row['close']
    vwap_daily = pd.Series(vwap_daily, index=df.index)

    # Diferencia relativa
    mask = ~(vwap_rolling.isna() | vwap_daily.isna()) & (vwap_daily != 0)
    diff_pct = ((vwap_rolling - vwap_daily).abs() / vwap_daily * 100)[mask]
    return {
        "brecha": "VWAP rolling vs daily reset",
        "mean_diff_pct": round(diff_pct.mean(), 3),
        "max_diff_pct":  round(diff_pct.max(), 3),
        "p95_diff_pct":  round(diff_pct.quantile(0.95), 3),
        "conclusion": "Diferencia sistemática — señales de bounce/cross en posiciones distintas"
    }


def diagnose_sl_tp_impact(df, signals, sl_pct, tp_pct):
    """Brecha 4: cuántos trades TV ganarían pero Python pierde por SL."""
    trades_tv  = backtest_tv_mode(df, signals)
    trades_opt = backtest_with_sl_tp(df, signals, sl_pct, tp_pct)
    m_tv  = metrics(trades_tv,  "TV mode (sin SL/TP)")
    m_opt = metrics(trades_opt, f"Optuna mode (SL={sl_pct:.0%} TP={tp_pct:.0%})")

    # Trades que TV gana pero Optuna no (por SL hit)
    sl_hits = sum(1 for t in trades_opt if t.get('exit') == 'SL')
    sl_pct_of_total = sl_hits / len(trades_opt) * 100 if trades_opt else 0

    return {
        "brecha": "SL/TP externo vs exit natural",
        "tv_mode":    m_tv,
        "optuna_mode": m_opt,
        "wr_gap":     round(m_tv['wr'] - m_opt['wr'], 1),
        "sl_hits":    sl_hits,
        "sl_pct":     round(sl_pct_of_total, 1),
        "conclusion": f"WR gap: {round(m_tv['wr'] - m_opt['wr'], 1)}pp. {sl_hits} trades ({sl_pct_of_total:.1f}%) cerrados por SL en Python pero continuarían en TV."
    }


# ── Runner principal ───────────────────────────────────────────────────────────

def run_validation(strategy_name, symbol, tf, sl_pct=0.05, tp_pct=0.08):
    print(f"\n{'='*70}")
    print(f"VALIDACION: {strategy_name} × {symbol} × {tf}")
    print(f"{'='*70}")

    # Cargar datos
    df = load_candles(symbol, tf)
    if df is None or len(df) < 200:
        print(f"  ERROR: Sin datos suficientes para {symbol} {tf}")
        return

    print(f"  Datos: {len(df)} velas | {df.index[0].date()} → {df.index[-1].date()}")

    # Cargar estrategia dinámicamente
    import importlib, glob
    gen_fn = None
    for batch_file in sorted(glob.glob(os.path.join(BATCHES_DIR, "strategies_tv2_batch*.py"))):
        try:
            mod_name = os.path.basename(batch_file).replace('.py', '')
            spec = importlib.util.spec_from_file_location(mod_name, batch_file)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, 'STRATEGY_EXPORT') and strategy_name in mod.STRATEGY_EXPORT:
                gen_fn = mod.STRATEGY_EXPORT[strategy_name]['gen']
                print(f"  Estrategia cargada desde: {os.path.basename(batch_file)}")
                break
        except Exception as e:
            continue

    if gen_fn is None:
        print(f"  ERROR: No se encontró {strategy_name} en ningún batch")
        return

    # Generar señales
    try:
        signals = gen_fn(df)
    except Exception as e:
        print(f"  ERROR generando señales: {e}")
        return

    n_long  = (signals == 1).sum()
    n_short = (signals == -1).sum()
    n_total = n_long + n_short
    print(f"  Señales: {n_total} total ({n_long} long, {n_short} short)")

    # ── DIAGNÓSTICO BRECHA 1: RSI ──────────────────────────────────────────
    if 'RSI' in strategy_name or 'StochRSI' in strategy_name or 'ST_RSI' in strategy_name:
        r = diagnose_rsi_impl(df)
        print(f"\n  [BRECHA 1] RSI ewm vs RMA:")
        print(f"    Diff media: {r['mean_diff_points']} puntos RSI")
        print(f"    Diff primeras 50 barras: {r['early_bars_diff']} puntos RSI")
        print(f"    P95 diff: {r['p95_diff_points']} puntos RSI")
        if r['mean_diff_points'] > 5:
            print(f"    >> IMPACTO SIGNIFICATIVO en señales")

    # ── DIAGNÓSTICO BRECHA 2: BB ───────────────────────────────────────────
    if 'BB' in strategy_name:
        r = diagnose_bb_std(df)
        print(f"\n  [BRECHA 2] BB std ddof:")
        print(f"    Python bandas {r['effect_on_band_width']}")
        print(f"    {r['signals_impact']}")

    # ── DIAGNÓSTICO BRECHA 3: VWAP ────────────────────────────────────────
    if 'VWAP' in strategy_name:
        r = diagnose_vwap_reset(df)
        print(f"\n  [BRECHA 3] VWAP rolling vs daily reset:")
        print(f"    Diff media: {r['mean_diff_pct']:.3f}% | Max: {r['max_diff_pct']:.3f}% | P95: {r['p95_diff_pct']:.3f}%")
        if r['mean_diff_pct'] > 0.5:
            print(f"    >> IMPACTO SIGNIFICATIVO en niveles de bounce/cross")

    # ── DIAGNÓSTICO BRECHA 4: SL/TP ───────────────────────────────────────
    r4 = diagnose_sl_tp_impact(df, signals, sl_pct, tp_pct)
    print(f"\n  [BRECHA 4] SL/TP externo vs exit natural:")
    print(f"    TV mode  : {r4['tv_mode']['trades']} trades | WR={r4['tv_mode']['wr']}% | PnL={r4['tv_mode']['pnl_pct']}%")
    print(f"    Optuna   : {r4['optuna_mode']['trades']} trades | WR={r4['optuna_mode']['wr']}% | PnL={r4['optuna_mode']['pnl_pct']}%")
    print(f"    WR gap   : {r4['wr_gap']}pp | SL hits: {r4['sl_hits']} ({r4['sl_pct']}%)")
    avg_dur_tv = r4['tv_mode'].get('avg_dur', 0)
    avg_dur_opt = r4['optuna_mode'].get('avg_dur', 0)
    print(f"    Dur media: TV={avg_dur_tv} bars | Optuna={avg_dur_opt} bars")

    # Veredicto
    print(f"\n  VEREDICTO:")
    wr_gap = r4['wr_gap']
    if wr_gap > 15:
        print(f"    >> BRECHA CRITICA: WR gap {wr_gap}pp — SL/TP destruye el edge de esta estrategia")
        print(f"       Esta estrategia necesita exits propios, no SL/TP externo")
    elif wr_gap > 5:
        print(f"    >> BRECHA MEDIA: WR gap {wr_gap}pp — SL demasiado ajustado o TP demasiado amplio")
        print(f"       Calibrar SL/TP con MFE/MAE específicos de esta estrategia")
    else:
        print(f"    >> BRECHA BAJA: WR gap {wr_gap}pp — SL/TP razonablemente calibrado")

    return r4


# ── Fixes sugeridos ────────────────────────────────────────────────────────────

def print_fixes():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FIXES PROPUESTOS POR BRECHA                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║ [FIX 1] RSI — Estandarizar a RMA Wilder en todos los batches                ║
║   Dónde: strategies_tv2_batch163.py → _rsi_calc()                           ║
║   Qué cambiar:                                                               ║
║     # ANTES (batch163):                                                      ║
║     gain = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()       ║
║     # DESPUÉS (igual a batch175 _rma175):                                   ║
║     gain = pd.Series(_rma175(delta.clip(lower=0).values, period), ...)      ║
║                                                                              ║
║ [FIX 2] BB std — Usar ddof=0 para igualar Pine                              ║
║   Dónde: strategies_tv2_batch163.py → gen_TV_BB_MeanRev_Simple()            ║
║     # ANTES:                                                                 ║
║     bb_dev = c.rolling(bb_period).std() * bb_std                            ║
║     # DESPUÉS:                                                               ║
║     bb_dev = c.rolling(bb_period).std(ddof=0) * bb_std                      ║
║                                                                              ║
║ [FIX 3] VWAP — Usar daily reset en lugar de rolling window                  ║
║   Dónde: strategies_tv2_batch163.py → gen_TV_VWAP_Bounce_Cross()            ║
║     Reemplazar el rolling VWAP por _vwap_daily_reset() (función nueva)       ║
║     Ver implementación más abajo.                                            ║
║                                                                              ║
║ [FIX 4] SL/TP — No aplicar SL externo a estrategias con exit propio         ║
║   Dónde: optuna_v7.py → backtest_with_sl_tp()                               ║
║   Idea: agregar flag use_strategy_exit=True en STRATEGY_EXPORT              ║
║     Si use_strategy_exit=True → calibrar SL/TP solo como safety net (3×MAE) ║
║     no como exit primario                                                    ║
║                                                                              ║
║ [FIX 5] Timestamps corruptos — Filtrar en load_candles()                    ║
║   Dónde: optuna_v7.py → load_candles()                                      ║
║     Agregar: df = df[df['ts'] > 1_262_304_000_000]  # post-2010             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


# ── Fix 3: Función VWAP con daily reset ───────────────────────────────────────

def _vwap_daily_reset(df):
    """
    VWAP con reset diario (00:00 UTC).
    Equivalente a ta.vwap en Pine Script (reset en cada sesión).
    Retorna pd.Series alineada al índice del df.

    IMPORTANTE: Pine usa 'session' que en crypto = día calendar UTC.
    Para futuros: también día calendar UTC (no hay sesión de mercado fija).
    """
    c = df['close']
    v = df['volume']
    h = df['high']
    l = df['low']
    hlc3 = (h + l + c) / 3

    vwap = np.full(len(df), np.nan)
    cum_vol, cum_hv = 0.0, 0.0

    for i in range(len(df)):
        idx = df.index[i]
        is_new_day = (i == 0) or (idx.date() != df.index[i-1].date())
        if is_new_day:
            cum_vol = max(v.iloc[i], 1e-10)
            cum_hv  = hlc3.iloc[i] * cum_vol
        else:
            cum_vol += v.iloc[i]
            cum_hv  += hlc3.iloc[i] * v.iloc[i]
        vwap[i] = cum_hv / cum_vol if cum_vol > 0 else c.iloc[i]

    return pd.Series(vwap, index=df.index)


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Valida brechas TV → Python")
    parser.add_argument('--strategy', type=str, default=None)
    parser.add_argument('--symbol',   type=str, default='BEAT/USDT:USDT')
    parser.add_argument('--tf',       type=str, default='1h')
    parser.add_argument('--sl',       type=float, default=0.05,
                        help='SL para modo Optuna (default 5%%)')
    parser.add_argument('--tp',       type=float, default=0.08,
                        help='TP para modo Optuna (default 8%%)')
    parser.add_argument('--all-top',  action='store_true',
                        help='Corre los top 5 grails')
    parser.add_argument('--fixes',    action='store_true',
                        help='Muestra los fixes propuestos')
    args = parser.parse_args()

    if args.fixes:
        print_fixes()
        return

    if args.all_top:
        print("\nCORRIENDO TOP GRAILS...\n")
        for strat, sym, tf in TOP_GRAILS:
            run_validation(strat, sym, tf, args.sl, args.tp)
        print_fixes()
        return

    if args.strategy:
        run_validation(args.strategy, args.symbol, args.tf, args.sl, args.tp)
        print_fixes()
    else:
        parser.print_help()
        print("\nEjemplo:")
        print("  python3 validate_tv_python.py --strategy TV_BB_MeanRev_Simple --symbol BEAT/USDT:USDT --tf 1h")
        print("  python3 validate_tv_python.py --all-top")
        print("  python3 validate_tv_python.py --fixes")


if __name__ == '__main__':
    main()
