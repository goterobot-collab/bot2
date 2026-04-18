#!/usr/bin/env python3
"""
RISK MANAGER — Gestión de riesgo por BOT y por ACTIVO
Regla de oro: NO PERDER PLATA

Calcula para cada grail:
1. Kelly Criterion → % óptimo del capital por trade
2. Stop Loss dinámico → basado en MAE P95 del backtest
3. Max Drawdown Kill Switch → apaga bot si pierde X%
4. Correlación entre activos → no sobreexponer al mismo movimiento
5. Position sizing → cuánto meter en cada trade
6. Risk Score → nota de 1-10 de seguridad del bot
"""
import json, os, sys, warnings
import numpy as np
import pandas as pd
import sqlite3
from collections import defaultdict

warnings.filterwarnings('ignore')

PROJECT = "/Users/sabrina/CLAUDE CODE/Estrategias"
DB_PATH = "/Users/sabrina/Code/activos binace /activos_binance.db"

# ═══════════════════════════════════════════════════════════
# 1. KELLY CRITERION — Cuánto arriesgar por trade
# ═══════════════════════════════════════════════════════════
def kelly_criterion(wr_pct, profit_factor):
    """
    Kelly % = W - (1-W)/R
    W = win rate (0-1)
    R = ratio avg_win / avg_loss = profit_factor

    Usamos HALF KELLY (más conservador, reduce volatilidad 50%)
    """
    w = wr_pct / 100.0
    r = max(profit_factor, 1.01)  # evitar division por 0

    kelly_full = w - (1 - w) / r
    kelly_half = kelly_full / 2  # HALF KELLY = más seguro

    # Limits: nunca más del 5% por trade, nunca negativo
    kelly_safe = max(0, min(kelly_half, 0.05))

    return {
        'kelly_full_pct': round(kelly_full * 100, 2),
        'kelly_half_pct': round(kelly_half * 100, 2),
        'kelly_safe_pct': round(kelly_safe * 100, 2),  # ← USAR ESTE
        'recommendation': classify_kelly(kelly_safe)
    }

def classify_kelly(k):
    if k <= 0: return "❌ NO OPERAR — Kelly negativo"
    if k < 0.005: return "⚠️ Muy bajo — skip o mínimo"
    if k < 0.01: return "🟡 Conservador — 0.5-1% del capital"
    if k < 0.02: return "🟢 Normal — 1-2% del capital"
    if k < 0.03: return "🟢 Bueno — 2-3% del capital"
    if k < 0.05: return "🔵 Agresivo — 3-5% del capital"
    return "🔵 Máximo — 5% del capital (tope)"


# ═══════════════════════════════════════════════════════════
# 2. STOP LOSS DINÁMICO — Basado en MAE del backtest
# ═══════════════════════════════════════════════════════════
def dynamic_stop_loss(mae_p95, max_dd_pct, avg_pnl_per_trade=None):
    """
    MAE P95 = máxima excursión adversa en el percentil 95
    = cuánto baja el trade ANTES de ganar

    Stop loss = MAE_P95 × 1.5 (margen de seguridad)
    Si el trade baja MÁS que esto → algo cambió → salir
    """
    if mae_p95 <= 0:
        mae_p95 = 0.005  # default 0.5%

    # Stop loss basado en MAE
    sl_from_mae = mae_p95 * 1.5  # 50% de margen sobre P95

    # Stop loss máximo per-bot (rango 2%-40%, no cap fijo)
    sl = max(0.02, min(0.40, sl_from_mae))

    # Take profit sugerido (ratio risk:reward mínimo 1:2)
    tp_suggested = sl * 2.5

    return {
        'stop_loss_pct': round(sl * 100, 3),       # % desde entrada
        'take_profit_pct': round(tp_suggested * 100, 3),
        'mae_p95_pct': round(mae_p95 * 100, 3),
        'risk_reward_ratio': round(tp_suggested / sl, 1) if sl > 0 else 0,
        'sl_type': 'MAE-based dynamic',
        'note': f"Si el trade baja más de {sl*100:.2f}% → CERRAR"
    }


# ═══════════════════════════════════════════════════════════
# 3. MAX DRAWDOWN KILL SWITCH — Apagar bot si pierde mucho
# ═══════════════════════════════════════════════════════════
def kill_switch(max_dd_backtest, capital_pct=100):
    """
    Si el bot pierde más que su peor drawdown histórico × 1.3 → APAGAR

    Niveles:
    - ALERTA: drawdown = 50% del max histórico
    - WARNING: drawdown = 80% del max histórico
    - KILL: drawdown = 130% del max histórico (nunca pasó esto → algo roto)
    """
    dd = max(max_dd_backtest, 1)  # mínimo 1%

    alert_level = dd * 0.5    # 50% del peor caso → vigilar
    warning_level = dd * 0.8  # 80% del peor caso → reducir posición
    kill_level = dd * 1.3     # 130% del peor caso → APAGAR

    # Límites absolutos (nunca perder más del 15% del capital asignado al bot)
    kill_absolute = min(kill_level, 15.0)

    return {
        'alert_dd_pct': round(alert_level, 1),     # 🟡 Vigilar
        'warning_dd_pct': round(warning_level, 1),  # 🟠 Reducir posición 50%
        'kill_dd_pct': round(kill_absolute, 1),      # 🔴 APAGAR BOT
        'max_dd_backtest': round(dd, 1),
        'action_plan': {
            f'DD > {alert_level:.0f}%': '🟡 Monitorear cada hora',
            f'DD > {warning_level:.0f}%': '🟠 Reducir posición al 50%',
            f'DD > {kill_absolute:.0f}%': '🔴 APAGAR BOT — revisar manualmente',
        }
    }


# ═══════════════════════════════════════════════════════════
# 4. CORRELACIÓN ENTRE ACTIVOS — No sobreexponer
# ═══════════════════════════════════════════════════════════
def analyze_correlations(symbols, timeframe='1d', top_n=20):
    """
    Calcula correlación de retornos entre activos.
    Si 5 bots operan activos correlacionados (BTC, ETH, BNB) →
    cuando BTC cae, TODOS pierden → riesgo concentrado

    Solución: limitar exposición a grupos correlacionados
    """
    conn = sqlite3.connect(DB_PATH)

    # Cargar closes diarios de los símbolos
    returns_dict = {}
    for sym in symbols[:top_n]:
        try:
            df = pd.read_sql(
                "SELECT ts, close FROM candles WHERE symbol=? AND timeframe=? ORDER BY ts",
                conn, params=(sym, timeframe)
            )
            if len(df) < 30:
                continue
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['return'] = df['close'].pct_change()
            returns_dict[sym] = df['return'].values[1:]  # quitar primer NaN
        except:
            continue

    conn.close()

    if len(returns_dict) < 2:
        return {'error': 'Not enough data'}

    # Alinear longitudes
    min_len = min(len(v) for v in returns_dict.values())
    aligned = {k: v[-min_len:] for k, v in returns_dict.items()}

    # Matriz de correlación
    names = list(aligned.keys())
    matrix = np.corrcoef([aligned[n] for n in names])

    # Encontrar pares altamente correlacionados (> 0.7)
    high_corr_pairs = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            corr = matrix[i][j]
            if abs(corr) > 0.7:
                high_corr_pairs.append({
                    'asset1': names[i],
                    'asset2': names[j],
                    'correlation': round(corr, 3)
                })

    # Grupos de activos correlacionados
    groups = build_correlation_groups(names, matrix, threshold=0.7)

    return {
        'total_assets': len(names),
        'high_corr_pairs': sorted(high_corr_pairs, key=lambda x: -abs(x['correlation'])),
        'correlation_groups': groups,
        'rule': 'No más del 20% del capital en un mismo grupo correlacionado'
    }

def build_correlation_groups(names, matrix, threshold=0.7):
    """Agrupa activos correlacionados"""
    visited = set()
    groups = []

    for i in range(len(names)):
        if names[i] in visited:
            continue
        group = [names[i]]
        visited.add(names[i])
        for j in range(len(names)):
            if j != i and names[j] not in visited and abs(matrix[i][j]) > threshold:
                group.append(names[j])
                visited.add(names[j])
        if len(group) > 1:
            groups.append(group)

    return groups


# ═══════════════════════════════════════════════════════════
# 5. POSITION SIZING — Cuánto meter en cada trade
# ═══════════════════════════════════════════════════════════
def position_sizing(capital, kelly_safe_pct, stop_loss_pct, leverage=1):
    """
    Calcula el tamaño de posición.

    Regla: nunca perder más de kelly_safe% del capital en un trade.

    position_size = (capital × risk_per_trade) / stop_loss
    Con leverage: position_size × leverage
    """
    risk_amount = capital * (kelly_safe_pct / 100)  # $ que podés perder

    if stop_loss_pct <= 0:
        stop_loss_pct = 1  # default 1%

    # Tamaño de posición sin leverage
    position_base = risk_amount / (stop_loss_pct / 100)

    # Con leverage
    margin_required = position_base / leverage

    # No superar el 25% del capital en una sola posición
    max_position = capital * 0.25 * leverage
    position_final = min(position_base, max_position)
    margin_final = position_final / leverage

    return {
        'capital': capital,
        'risk_per_trade_usd': round(risk_amount, 2),
        'position_size_usd': round(position_final, 2),
        'margin_required_usd': round(margin_final, 2),
        'leverage': leverage,
        'max_loss_usd': round(risk_amount, 2),
        'max_loss_pct': round(kelly_safe_pct, 2),
    }


# ═══════════════════════════════════════════════════════════
# 6. RISK SCORE — Nota de seguridad 1-10
# ═══════════════════════════════════════════════════════════
def risk_score(grail):
    """
    Calcula un score de riesgo de 1 (peligroso) a 10 (muy seguro).

    Factores:
    - WR alto en test → más seguro
    - Muchos trades → más confiable (sample size)
    - DD bajo → menos riesgo
    - PF alto → gana más de lo que pierde
    - WR diff bajo → no es overfit
    - Sharpe alto → retorno ajustado por riesgo
    """
    test = grail.get('test', {})
    full = grail.get('full', {})

    scores = []

    # WR en test (70-100 → 3-10)
    wr = test.get('wr', 70)
    s_wr = min(10, max(1, (wr - 60) / 4))
    scores.append(('win_rate', s_wr, 0.25))

    # Trades (15-100+ → 3-10)
    trades = full.get('trades', 15)
    s_trades = min(10, max(1, trades / 10))
    scores.append(('sample_size', s_trades, 0.20))

    # Max DD (bajo = mejor) (0-50 → 10-1)
    dd = full.get('max_dd', 20)
    s_dd = max(1, 10 - dd / 5)
    scores.append(('drawdown', s_dd, 0.20))

    # Profit Factor (1-5+ → 3-10)
    pf = full.get('pf', test.get('pf', 1))
    s_pf = min(10, max(1, pf * 2))
    scores.append(('profit_factor', s_pf, 0.15))

    # WR diff (bajo = mejor) (0-25 → 10-3)
    diff = grail.get('wr_diff', 10)
    s_diff = max(1, 10 - diff / 3)
    scores.append(('consistency', s_diff, 0.10))

    # Sharpe (alto = mejor)
    sharpe = full.get('sharpe', test.get('sharpe', 0))
    s_sharpe = min(10, max(1, sharpe / 3))
    scores.append(('sharpe', s_sharpe, 0.10))

    # Weighted score
    total = sum(s * w for _, s, w in scores)

    # Classify
    if total >= 8: grade = 'A+'
    elif total >= 7: grade = 'A'
    elif total >= 6: grade = 'B+'
    elif total >= 5: grade = 'B'
    elif total >= 4: grade = 'C'
    else: grade = 'D'

    return {
        'risk_score': round(total, 1),
        'grade': grade,
        'components': {name: round(s, 1) for name, s, _ in scores},
        'safe_to_trade': total >= 5,
        'confidence': 'HIGH' if total >= 7 else 'MEDIUM' if total >= 5 else 'LOW'
    }


# ═══════════════════════════════════════════════════════════
# 7. RISK PROFILE COMPLETO POR GRAIL
# ═══════════════════════════════════════════════════════════
def full_risk_profile(grail, capital=100):
    """Genera el perfil de riesgo COMPLETO de un grail"""
    test = grail.get('test', {})
    full = grail.get('full', {})

    wr = test.get('wr', 70)
    pf = full.get('pf', test.get('pf', 1.5))
    mae = test.get('mae_p95', 0.01)
    dd = full.get('max_dd', 20)

    # 1. Kelly
    k = kelly_criterion(wr, pf)

    # 2. Stop Loss
    sl = dynamic_stop_loss(mae, dd)

    # 3. Kill Switch
    ks = kill_switch(dd)

    # 4. Risk Score
    rs = risk_score(grail)

    # 5. Safe leverage (basado en MAE)
    safe_lev = min(20, max(1, int(1 / (mae * 2.5 + 1e-10))))

    # 6. Position sizing
    ps = position_sizing(capital, k['kelly_safe_pct'], sl['stop_loss_pct'], leverage=min(safe_lev, 5))

    # 7. Duration alerts
    dur = grail.get('duration', {})
    dur_alerts = {
        'avg_hours': dur.get('avg_h', 0),
        'p95_hours': dur.get('p95_h', 0),
        'max_hours': dur.get('max_h', 0),
        'alert_at_hours': dur.get('p95_h', 48),  # alertar en P95
        'force_close_hours': dur.get('max_h', 168),  # cerrar en max histórico
    }

    return {
        'grail_id': f"{grail['strategy']}_{grail['symbol']}_{grail['timeframe']}",
        'strategy': grail['strategy'],
        'symbol': grail['symbol'],
        'timeframe': grail['timeframe'],
        'kelly': k,
        'stop_loss': sl,
        'kill_switch': ks,
        'risk_score': rs,
        'position_sizing': ps,
        'safe_leverage': safe_lev,
        'duration_alerts': dur_alerts,
        'summary': {
            'risk_grade': rs['grade'],
            'risk_score': rs['risk_score'],
            'risk_per_trade': f"{k['kelly_safe_pct']:.1f}%",
            'stop_loss': f"{sl['stop_loss_pct']:.2f}%",
            'kill_at_dd': f"{ks['kill_dd_pct']:.0f}%",
            'max_leverage': safe_lev,
            'safe_to_trade': rs['safe_to_trade'],
        }
    }


# ═══════════════════════════════════════════════════════════
# MAIN — Analizar todos los grails
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("🛡️  RISK MANAGER — Análisis de riesgo por bot y por activo")
    print("=" * 70)

    # Load grails
    r1 = json.load(open(os.path.join(PROJECT, 'data/santo_grial_clean.json')))
    r1_elite = [g for g in r1 if g['full']['trades']>=20 and g['test']['wr']>=70
                and g.get('wr_diff',0)<25 and g['full']['pnl']>0]

    turbo = []
    for i in range(8):
        f = os.path.join(PROJECT, f'data/worker_{i}_progress.json')
        if os.path.exists(f):
            p = json.load(open(f))
            turbo.extend(p.get('grails', []))
    turbo_elite = [g for g in turbo if g.get('test',{}).get('wr',0)>=70
                   and g.get('full',{}).get('pnl',0)>0 and g.get('full',{}).get('trades',0)>=15
                   and g.get('wr_diff',99)<25]

    all_grails = r1_elite + turbo_elite
    print(f"\nTotal grails a analizar: {len(all_grails)}")

    # Capital base
    CAPITAL = float(sys.argv[1]) if len(sys.argv) > 1 else 100
    print(f"Capital base: ${CAPITAL}")

    # Analyze each grail
    results = []
    grades = defaultdict(int)

    for g in all_grails:
        profile = full_risk_profile(g, capital=CAPITAL)
        results.append(profile)
        grades[profile['risk_score']['grade']] += 1

    # Sort by risk score (best first)
    results.sort(key=lambda r: -r['risk_score']['risk_score'])

    # Summary
    print(f"\n{'='*70}")
    print("📊 RESUMEN DE RIESGO")
    print(f"{'='*70}")

    for grade in ['A+', 'A', 'B+', 'B', 'C', 'D']:
        n = grades.get(grade, 0)
        if n > 0:
            bar = '█' * min(50, n // 5)
            print(f"  {grade:3s}: {n:5d} bots {bar}")

    safe = sum(1 for r in results if r['risk_score']['safe_to_trade'])
    print(f"\n✅ SAFE TO TRADE: {safe} / {len(results)}")
    print(f"❌ NO OPERAR:     {len(results) - safe} / {len(results)}")

    # Top 20 safest
    print(f"\n{'='*70}")
    print("🏆 TOP 20 BOTS MÁS SEGUROS")
    print(f"{'='*70}")
    print(f"{'#':>3} {'Score':>5} {'Grade':>5} {'Strategy':25s} {'Symbol':22s} {'TF':>4} {'WR%':>5} {'Kelly%':>6} {'SL%':>6} {'KillDD':>6} {'Lev':>3}")
    print("-" * 120)

    for i, r in enumerate(results[:20]):
        g = [g for g in all_grails if f"{g['strategy']}_{g['symbol']}_{g['timeframe']}" == r['grail_id']][0] if False else None
        print(f"{i+1:3d} {r['risk_score']['risk_score']:5.1f} {r['risk_score']['grade']:>5s} "
              f"{r['strategy'][:25]:25s} {r['symbol'][:22]:22s} {r['timeframe']:>4s} "
              f"{r['summary']['risk_per_trade']:>6s} "
              f"{r['summary']['stop_loss']:>6s} {r['summary']['kill_at_dd']:>6s} {r['safe_leverage']:>3d}")

    # Risk by asset (aggregate)
    print(f"\n{'='*70}")
    print("📈 RIESGO POR ACTIVO (top 20 con más bots)")
    print(f"{'='*70}")

    asset_risk = defaultdict(list)
    for r in results:
        asset_risk[r['symbol']].append(r['risk_score']['risk_score'])

    asset_summary = []
    for sym, scores in asset_risk.items():
        asset_summary.append({
            'symbol': sym,
            'bots': len(scores),
            'avg_risk': np.mean(scores),
            'min_risk': np.min(scores),
            'max_risk': np.max(scores),
        })

    asset_summary.sort(key=lambda x: (-x['bots'], -x['avg_risk']))
    print(f"{'Symbol':22s} {'Bots':>5} {'Avg Score':>9} {'Min':>5} {'Max':>5}")
    print("-" * 50)
    for a in asset_summary[:20]:
        print(f"{a['symbol'][:22]:22s} {a['bots']:5d} {a['avg_risk']:9.1f} {a['min_risk']:5.1f} {a['max_risk']:5.1f}")

    # Correlation analysis (top 20 most traded assets)
    print(f"\n{'='*70}")
    print("🔗 CORRELACIÓN ENTRE ACTIVOS")
    print(f"{'='*70}")

    top_symbols = [a['symbol'] for a in asset_summary[:20]]
    corr = analyze_correlations(top_symbols)

    if 'correlation_groups' in corr:
        print(f"\nGrupos correlacionados (corr > 0.7):")
        for i, group in enumerate(corr['correlation_groups']):
            print(f"  Grupo {i+1}: {', '.join(g[:15] for g in group)}")
            print(f"    ⚠️  Máximo 20% del capital total en este grupo")

        if corr['high_corr_pairs']:
            print(f"\nPares más correlacionados:")
            for p in corr['high_corr_pairs'][:10]:
                print(f"  {p['asset1'][:15]:15s} ↔ {p['asset2'][:15]:15s} = {p['correlation']:.3f}")

    # Portfolio rules
    print(f"\n{'='*70}")
    print("📜 REGLAS DE ORO PARA NO PERDER PLATA")
    print(f"{'='*70}")
    print(f"""
    1. NUNCA arriesgar más del 2% del capital en UN trade
    2. NUNCA más del 20% del capital en activos correlacionados
    3. MÁXIMO 5-10 bots activos simultáneamente
    4. Si un bot llega a DD kill level → APAGAR inmediatamente
    5. Si 3+ bots en DD warning → REDUCIR todo al 25%
    6. Paper trading 2 semanas ANTES de meter plata real
    7. Empezar con leverage 2-3x máximo (no importa lo que diga Kelly)
    8. Reinvertir ganancias SOLO después de 1 mes rentable
    9. Guardar 20% de ganancias fuera del exchange (take profit del portfolio)
    10. NUNCA meter plata que no puedas perder

    Con ${CAPITAL} de capital:
    - Riesgo máximo por trade: ${CAPITAL * 0.02:.2f} (2%)
    - Máximo en un grupo correlacionado: ${CAPITAL * 0.20:.2f}
    - Ganancia meta mensual (realista): ${CAPITAL * 0.10:.2f} - ${CAPITAL * 0.20:.2f} (10-20%)
    """)

    # Save results
    output = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'capital': CAPITAL,
        'total_grails': len(all_grails),
        'safe_to_trade': safe,
        'grade_distribution': dict(grades),
        'risk_profiles': results[:500],  # top 500
        'correlation_groups': corr.get('correlation_groups', []),
        'high_corr_pairs': corr.get('high_corr_pairs', [])[:30],
        'golden_rules': {
            'max_risk_per_trade_pct': 2,
            'max_correlated_exposure_pct': 20,
            'max_simultaneous_bots': 10,
            'paper_trading_weeks': 2,
            'max_initial_leverage': 3,
            'profit_reserve_pct': 20,
        }
    }

    out_file = os.path.join(PROJECT, 'data/risk_profiles.json')
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 Guardado en: {out_file}")

    # Asset-level risk summary
    asset_output = []
    for a in asset_summary:
        bots_for_asset = [r for r in results if r['symbol'] == a['symbol']]
        best_bot = bots_for_asset[0] if bots_for_asset else None
        asset_output.append({
            'symbol': a['symbol'],
            'total_bots': a['bots'],
            'avg_risk_score': round(a['avg_risk'], 1),
            'best_bot': {
                'strategy': best_bot['strategy'] if best_bot else None,
                'timeframe': best_bot['timeframe'] if best_bot else None,
                'risk_grade': best_bot['risk_score']['grade'] if best_bot else None,
                'kelly_pct': best_bot['kelly']['kelly_safe_pct'] if best_bot else None,
                'stop_loss_pct': best_bot['stop_loss']['stop_loss_pct'] if best_bot else None,
            } if best_bot else None
        })

    asset_file = os.path.join(PROJECT, 'data/risk_by_asset.json')
    with open(asset_file, 'w') as f:
        json.dump(asset_output, f, indent=2, default=str)
    print(f"💾 Risk por activo: {asset_file}")


if __name__ == '__main__':
    main()
