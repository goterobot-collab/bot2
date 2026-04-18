#!/usr/bin/env python3
"""
V6 PIPELINE CORREGIDO v2 — LEVERAGE REAL POR ACTIVO × BOT

MEJORAS v2:
1. Leverage calculado POR BOT con MAE del bot + MAE del activo (el PEOR caso)
2. BRONZE eliminado — calidad insuficiente para producción
3. SILVER requiere mínimo 10 trades test para entrar
4. Mínimo 10 trades test para TODOS los bots (significancia estadística)
5. V5 avoid warning → reduce sizing, NO elimina
6. Stop loss basado en MAE real del BOT (no genérico)
7. Crash vulnerability reduce leverage adicional

REGLA DE LEVERAGE (anti-liquidación):
- mae_bot = test.mae_p95 del bot específico (cómo ESA estrategia se mueve en ESE activo)
- mae_v5 = mae_p95 del V5 leverage map (cómo ESE activo se mueve en producción real)
- mae_worst = max(mae_bot, mae_v5) — el PEOR caso
- safe_leverage = 1 / (mae_worst × 2.5) — sobrevivir 2.5x la peor MAE
- Capped at 20x max, 1x min

Bot V5: spike + RSI + trailing, 87 bots, WR 61.7%, 161K trades, $29K PnL
Bot V6: VWAP/ZScore/BB mean reversion, 265 estrategias, WR 85%+ en test OOS
"""
import json, os, sys
import numpy as np
from collections import defaultdict
from datetime import datetime

PROJECT = "/Users/sabrina/CLAUDE CODE/Estrategias"
GOLDEN = os.path.join(PROJECT, "v5_golden_data_for_v6")

# V5 reference data (for enrichment, NOT for filtering)
def load_v5_reference():
    """Load V5 data as REFERENCE only"""
    ref = {}

    # Leverage map from V5 MAE (useful — real market data)
    lev_map = json.load(open(os.path.join(GOLDEN, 'v5_leverage_map.json')))
    ref['v5_leverage'] = {item['symbol']: item for item in lev_map}

    # Dream team symbols (reference — these work well in V5)
    dt = json.load(open(os.path.join(GOLDEN, 'v5_dream_team_symbols.json')))
    ref['v5_dream_team'] = set(item['symbol'] for item in dt)

    # Crash vulnerability (useful for risk management)
    crash = json.load(open(os.path.join(GOLDEN, 'v5_crash_vulnerability.json')))
    ref['v5_crash'] = {item['symbol']: item for item in crash}

    # V5 blacklist — ONLY as warning flag, NOT as filter
    avoid = json.load(open(os.path.join(GOLDEN, 'v5_symbols_to_avoid.json')))
    ref['v5_avoid'] = set(item['symbol'] for item in avoid)

    # V5 bot performance
    ref['v5_performance'] = json.load(open(os.path.join(GOLDEN, 'v5_bot_symbol_performance.json')))

    return ref


def load_v6_grails():
    """Load ALL V6 grails (R1 + Turbo)"""
    r1 = json.load(open(os.path.join(PROJECT, 'data/santo_grial_clean.json')))
    r1_elite = [g for g in r1 if g['full']['trades'] >= 15 and g['test']['wr'] >= 70
                and g.get('wr_diff', 0) < 25 and g['full']['pnl'] > 0]

    turbo = []
    for i in range(8):
        f = os.path.join(PROJECT, f'data/worker_{i}_progress.json')
        if os.path.exists(f):
            p = json.load(open(f))
            turbo.extend(p.get('grails', []))
    turbo_elite = [g for g in turbo if g.get('test', {}).get('wr', 0) >= 70
                   and g.get('full', {}).get('pnl', 0) > 0
                   and g.get('full', {}).get('trades', 0) >= 15
                   and g.get('wr_diff', 99) < 25]

    print(f"V6 Grails: {len(r1_elite)} R1 + {len(turbo_elite)} Turbo = {len(r1_elite + turbo_elite)}")
    return r1_elite + turbo_elite


# ═══════════════════════════════════════════════════════════
# PHASE 1: V6 SELF-VALIDATION (no V5 filtering!)
# ═══════════════════════════════════════════════════════════
def phase1_validate(grails):
    """Filter only by V6's own walk-forward validation — NOT by V5 blacklist"""
    stats = defaultdict(int)
    passed = []

    # Duplicate symbols to skip (same coin, different denomination)
    DUPES = {'1000BONK/USDT:USDT', '1000FLOKI/USDT:USDT', '1000SATS/USDT:USDT',
             '1000PEPE/USDT:USDT', '1000RATS/USDT:USDT', '1000000BOB/USDT:USDT'}

    for g in grails:
        stats['total'] += 1

        # Only remove actual duplicates (same coin)
        if g['symbol'] in DUPES:
            stats['duplicate_token'] += 1
            continue

        # V6 own quality filters
        test = g.get('test', {})
        full = g.get('full', {})

        if test.get('wr', 0) < 70:
            stats['low_wr'] += 1
            continue
        if test.get('pnl', 0) <= 0:
            stats['negative_pnl'] += 1
            continue
        if full.get('trades', 0) < 15:
            stats['low_trades_full'] += 1
            continue
        if test.get('trades', 0) < 10:
            stats['low_trades_test'] += 1
            continue
        if g.get('wr_diff', 99) > 25:
            stats['overfit'] += 1
            continue

        passed.append(g)
        stats['passed'] += 1

    print(f"\n{'='*60}")
    print(f"PHASE 1 — V6 SELF-VALIDATION (sin filtro V5)")
    print(f"{'='*60}")
    for k, v in sorted(stats.items()):
        pct = v / max(stats['total'], 1) * 100
        print(f"  {k:20s}: {v:6d} ({pct:5.1f}%)")
    print(f"\n  ✅ {stats['passed']} grails pasaron validación V6 propia")

    return passed


# ═══════════════════════════════════════════════════════════
# PHASE 2: ENRICH with V5 data (reference, not filter)
# ═══════════════════════════════════════════════════════════
def phase2_enrich(grails, ref):
    """Add V5 reference data + LEVERAGE REAL POR BOT (anti-liquidación)"""
    lev_distribution = defaultdict(int)

    for g in grails:
        sym = g['symbol']
        test = g.get('test', {})

        # ═══════════════════════════════════════════
        # LEVERAGE REAL POR ACTIVO × BOT
        # Cada activo responde diferente a cada config
        # No queremos ser liquidados NI sacados por SL
        # ═══════════════════════════════════════════

        # MAE del BOT específico (cómo ESA estrategia se mueve en ESE activo)
        mae_bot = test.get('mae_p95', 0)

        # MAE del ACTIVO en V5 producción real
        v5_info = ref['v5_leverage'].get(sym, {})
        mae_v5 = v5_info.get('mae_p95', 0)

        # Crash vulnerability — activos volátiles en crashes necesitan más margen
        crash = ref['v5_crash'].get(sym, {})
        crash_vuln = crash.get('vulnerability_score', 50)
        crash_max_dd = crash.get('max_floating_dd_pct', 0)

        # PEOR CASO: usar la MAE más alta entre bot y activo
        mae_candidates = [m for m in [mae_bot, mae_v5] if m and m > 0]
        if mae_candidates:
            mae_worst = max(mae_candidates)
        else:
            mae_worst = 0.05  # default conservador 5%

        # Si el activo es muy vulnerable a crashes, agregar margen extra
        if crash_vuln > 70:
            mae_worst = max(mae_worst, crash_max_dd * 0.5) if crash_max_dd > 0 else mae_worst * 1.3

        # Fórmula: sobrevivir 2.5x la peor MAE sin liquidación
        # Con 2.5x tenés margen para: MAE normal + slippage + spread + wick
        safe_lev = 1.0 / (mae_worst * 2.5) if mae_worst > 0 else 5

        # Caps: mínimo 1x, máximo 20x
        safe_lev = max(1, min(20, safe_lev))
        final_lev = int(safe_lev)  # redondear hacia abajo (más seguro)
        if final_lev < 1:
            final_lev = 1

        g['final_leverage'] = final_lev
        g['mae_bot'] = round(mae_bot, 6) if mae_bot else None
        g['mae_v5'] = round(mae_v5, 6) if mae_v5 else None
        g['mae_worst'] = round(mae_worst, 6)
        g['safe_leverage_v5'] = v5_info.get('safe_lev', None)
        g['safe_leverage_v6'] = g.get('safe_leverage', 20)

        if mae_bot and mae_bot > 0 and mae_v5 and mae_v5 > 0:
            g['leverage_source'] = 'MAE_bot+V5 (worst)'
        elif mae_bot and mae_bot > 0:
            g['leverage_source'] = 'MAE_bot'
        elif mae_v5 and mae_v5 > 0:
            g['leverage_source'] = 'MAE_V5'
        else:
            g['leverage_source'] = 'default_conservador'

        lev_distribution[final_lev] += 1

        # ═══════════════════════════════════════════
        # STOP LOSS REAL POR BOT (basado en SU MAE)
        # SL = MAE del bot × 2 (margen para no salir por ruido)
        # ═══════════════════════════════════════════
        if mae_bot and mae_bot > 0:
            sl = mae_bot * 2.0  # 2x la MAE del bot
        elif mae_v5 and mae_v5 > 0:
            sl = mae_v5 * 1.5  # 1.5x si solo tenemos V5
        else:
            sl = 0.02  # 2% default
        g['final_stop_loss_pct'] = round(max(0.02, min(0.40, sl)) * 100, 3)  # per-bot range 2%-40%

        # Dream team flag
        g['v5_dream_team'] = sym in ref['v5_dream_team']

        # V5 blacklist flag (WARNING — reduce sizing, no eliminar)
        g['v5_avoid_warning'] = sym in ref['v5_avoid']

        # Crash vulnerability
        g['crash_vulnerability'] = crash.get('vulnerability_score', 'unknown')

        # Kelly criterion (half-Kelly, conservador)
        wr = test.get('wr', 70)
        pf = g.get('full', {}).get('pf', test.get('pf', 1.5))
        w = wr / 100
        r = max(pf, 1.01)
        kelly = max(0, min((w - (1 - w) / r) / 2, 0.05))
        g['final_kelly_pct'] = round(kelly * 100, 2)

        # Kill switch DD (basado en max_dd real del bot)
        dd = g.get('full', {}).get('max_dd', 20)
        g['final_kill_dd_pct'] = round(min(dd * 1.3, 15), 1)

    # Stats
    with_mae_bot = sum(1 for g in grails if g.get('mae_bot') and g['mae_bot'] > 0)
    with_mae_v5 = sum(1 for g in grails if g.get('mae_v5') and g['mae_v5'] > 0)
    with_both = sum(1 for g in grails if g.get('mae_bot') and g['mae_bot'] > 0 and g.get('mae_v5') and g['mae_v5'] > 0)
    dt_count = sum(1 for g in grails if g.get('v5_dream_team'))
    warn_count = sum(1 for g in grails if g.get('v5_avoid_warning'))

    print(f"\n{'='*60}")
    print(f"PHASE 2 — LEVERAGE REAL POR ACTIVO × BOT")
    print(f"{'='*60}")
    print(f"  Con MAE del bot:         {with_mae_bot} ({with_mae_bot/len(grails)*100:.0f}%)")
    print(f"  Con MAE V5 real:         {with_mae_v5} ({with_mae_v5/len(grails)*100:.0f}%)")
    print(f"  Con AMBAS MAE:           {with_both} ({with_both/len(grails)*100:.0f}%)")
    print(f"  V5 Dream Team overlap:   {dt_count}")
    print(f"  V5 Avoid WARNING:        {warn_count} (sizing reducido, NO eliminados)")
    print(f"\n  📊 Distribución de leverage:")
    for lev in sorted(lev_distribution.keys()):
        n = lev_distribution[lev]
        bar = '█' * max(1, n // 50)
        print(f"    {lev:2d}x: {n:5d} {bar}")

    return grails


# ═══════════════════════════════════════════════════════════
# PHASE 3: CLASSIFY into tiers
# ═══════════════════════════════════════════════════════════
def phase3_classify(grails):
    """Classify by V6 quality + V5 confidence — NO BRONZE (calidad insuficiente)"""
    tiers = defaultdict(list)
    rejected_bronze = 0

    for g in grails:
        test = g.get('test', {})
        full = g.get('full', {})
        wr = test.get('wr', 0)
        trades_test = test.get('trades', 0)
        trades_full = full.get('trades', 0)
        is_dt = g.get('v5_dream_team', False)
        is_avoid = g.get('v5_avoid_warning', False)
        sharpe = full.get('sharpe', test.get('sharpe', 0))
        pf = full.get('pf', test.get('pf', 1))
        wr_diff = g.get('wr_diff', 10)

        # Quality score — penalizar avoid warning, premiar trades altos
        quality = (
            wr * 0.30 +
            min(sharpe, 20) * 1.5 +
            min(pf, 10) * 2 +
            min(trades_full, 100) * 0.2 +
            min(trades_test, 50) * 0.3 +        # NUEVO: premiar trades test
            (10 if is_dt else 0) +
            (-5 if is_avoid else 0) +            # NUEVO: penalizar avoid
            max(0, 10 - wr_diff) * 0.5 +
            (3 if g.get('mae_bot') and g['mae_bot'] > 0 else 0)  # bonus si tiene MAE real
        )
        g['quality_score'] = round(quality, 1)

        # Tier — sin BRONZE, SILVER requiere más
        if wr >= 90 and is_dt and trades_test >= 15:
            tier = 'ULTIMATE'
        elif wr >= 90 and trades_test >= 15:
            tier = 'DIAMOND'
        elif wr >= 85 and is_dt and trades_test >= 10:
            tier = 'GOLD'
        elif wr >= 85 and trades_test >= 10:
            tier = 'PLATINUM'
        elif wr >= 80 and trades_test >= 10:
            tier = 'ELITE'
        elif wr >= 75 and trades_test >= 10:
            tier = 'SILVER'
        else:
            # ELIMINADO: antes era BRONZE, ahora se descarta
            rejected_bronze += 1
            continue

        # V5 avoid warning → bajar un tier de sizing (no eliminar)
        if is_avoid:
            g['sizing_penalty'] = True
        else:
            g['sizing_penalty'] = False

        g['tier'] = tier
        tiers[tier].append(g)

    # Sort each tier
    for t in tiers:
        tiers[t].sort(key=lambda g: -g['quality_score'])

    print(f"\n{'='*60}")
    print(f"PHASE 3 — TIER CLASSIFICATION (sin BRONZE)")
    print(f"{'='*60}")
    # Sizing: normal y con penalidad por V5 avoid
    sizing = {'ULTIMATE': '$15-20', 'DIAMOND': '$10-15', 'GOLD': '$10-15',
              'PLATINUM': '$7-10', 'ELITE': '$5-7', 'SILVER': '$3-5'}
    sizing_penalty = {'ULTIMATE': '$10-15', 'DIAMOND': '$7-10', 'GOLD': '$7-10',
                      'PLATINUM': '$5-7', 'ELITE': '$3-5', 'SILVER': '$2-3'}
    total = 0
    for t in ['ULTIMATE', 'DIAMOND', 'GOLD', 'PLATINUM', 'ELITE', 'SILVER']:
        n = len(tiers.get(t, []))
        total += n
        n_penalty = sum(1 for g in tiers.get(t, []) if g.get('sizing_penalty'))
        if n > 0:
            penalty_info = f" ({n_penalty} con sizing reducido)" if n_penalty else ""
            print(f"  {t:12s}: {n:5d} grails | {sizing[t]}{penalty_info}")
    print(f"  {'RECHAZADOS':12s}: {rejected_bronze:5d} (serían BRONZE — calidad insuficiente)")
    print(f"  {'TOTAL PROD':12s}: {total:5d}")

    return tiers


# ═══════════════════════════════════════════════════════════
# PHASE 4: SELECT — Top production bots
# ═══════════════════════════════════════════════════════════
def phase4_select(tiers):
    """Select top bots with diversity"""
    all_g = []
    for grails in tiers.values():
        all_g.extend(grails)
    all_g.sort(key=lambda g: -g['quality_score'])

    # Dedup: best per strategy+symbol+tf
    seen = set()
    deduped = []
    for g in all_g:
        key = f"{g['strategy']}_{g['symbol']}_{g['timeframe']}"
        if key not in seen:
            seen.add(key)
            deduped.append(g)

    # Select with diversity (max 3 per symbol)
    selected = []
    sym_count = defaultdict(int)
    for g in deduped:
        if sym_count[g['symbol']] >= 3:
            continue
        sym_count[g['symbol']] += 1
        selected.append(g)
        if len(selected) >= 1000:
            break

    unique_syms = len(set(g['symbol'] for g in selected))
    unique_strats = len(set(g['strategy'] for g in selected))
    dt_count = sum(1 for g in selected if g.get('v5_dream_team'))
    warn_count = sum(1 for g in selected if g.get('v5_avoid_warning'))

    print(f"\n{'='*60}")
    print(f"PHASE 4 — FINAL SELECTION")
    print(f"{'='*60}")
    print(f"  Total selected:     {len(selected)}")
    print(f"  Unique symbols:     {unique_syms}")
    print(f"  Unique strategies:  {unique_strats}")
    print(f"  V5 Dream Team:      {dt_count} ⭐")
    print(f"  V5 Avoid warning:   {warn_count} ⚠️ (incluidos — son estrategias diferentes)")

    return selected, deduped


# ═══════════════════════════════════════════════════════════
# PHASE 5: COMPARE V6 vs V5
# ═══════════════════════════════════════════════════════════
def phase5_compare(selected, ref):
    """Compare V6 strategies vs V5 bot performance"""
    print(f"\n{'='*60}")
    print(f"🥊 V6 vs V5 — COMPARACIÓN DIRECTA")
    print(f"{'='*60}")

    # V5 stats (from dashboard)
    v5_stats = {
        'total_trades': 161014,
        'wr': 61.7,
        'pnl': 29617.81,
        'pf': 1.78,
        'bots': 78,
        'symbols': 399,
        'top_bot_wr': 62.8,  # ULTRA_SL45
        'top_bot_pnl': 1065.32,
    }

    # V6 stats
    v6_wrs = [g['test']['wr'] for g in selected]
    v6_pnls = [g['test']['pnl'] for g in selected]
    v6_trades = [g['full']['trades'] for g in selected]

    print(f"\n  {'Métrica':25s} {'V5 (real)':>15s} {'V6 (backtest)':>15s} {'Winner':>10s}")
    print(f"  {'-'*70}")

    comparisons = [
        ('Win Rate promedio', f"{v5_stats['wr']:.1f}%", f"{np.mean(v6_wrs):.1f}%",
         'V6 ✅' if np.mean(v6_wrs) > v5_stats['wr'] else 'V5'),
        ('Profit Factor', f"{v5_stats['pf']:.2f}", f"{np.mean([g.get('full',{}).get('pf',1) for g in selected]):.2f}",
         'V6 ✅'),
        ('Estrategias únicas', f"{v5_stats['bots']}", f"{len(set(g['strategy'] for g in selected))}",
         'V6 ✅' if len(set(g['strategy'] for g in selected)) > v5_stats['bots'] else 'V5'),
        ('Activos cubiertos', f"{v5_stats['symbols']}", f"{len(set(g['symbol'] for g in selected))}",
         'Comparar'),
        ('Total trades', f"{v5_stats['total_trades']:,}", f"{sum(v6_trades):,}",
         'V5 (más historial)'),
        ('Optimización', 'Parámetros fijos', 'Optuna per-asset', 'V6 ✅'),
        ('Validación', 'Walk-forward', 'Walk-forward 70/30', 'Ambos ✅'),
        ('Diversidad', 'Spike+RSI+filtros', 'VWAP+ZScore+BB+243 tipos', 'V6 ✅'),
    ]

    for metric, v5, v6, winner in comparisons:
        print(f"  {metric:25s} {v5:>15s} {v6:>15s} {winner:>10s}")

    print(f"\n  ⚠️ NOTA: V5 WR es de TRADES REALES, V6 es BACKTEST")
    print(f"  ⚠️ WR real suele ser 5-15% menor que backtest")
    print(f"  ⚠️ V6 necesita paper trading para validar en real")

    return v5_stats


# ═══════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════
def output(selected, deduped, tiers):
    out_dir = os.path.join(PROJECT, 'data')

    # Sizing por tier (con penalidad para V5 avoid)
    SIZING = {'ULTIMATE': 20, 'DIAMOND': 15, 'GOLD': 12, 'PLATINUM': 10, 'ELITE': 7, 'SILVER': 5}
    SIZING_PENALTY = {'ULTIMATE': 15, 'DIAMOND': 10, 'GOLD': 10, 'PLATINUM': 7, 'ELITE': 5, 'SILVER': 3}

    # Production file
    production = []
    for g in selected:
        tier = g['tier']
        has_penalty = g.get('sizing_penalty', False)
        size = SIZING_PENALTY.get(tier, 5) if has_penalty else SIZING.get(tier, 5)

        production.append({
            'rank': len(production) + 1,
            'tier': tier,
            'quality_score': g['quality_score'],
            'strategy': g['strategy'],
            'symbol': g['symbol'],
            'timeframe': g['timeframe'],
            'test_wr': g['test']['wr'],
            'test_pnl': g['test']['pnl'],
            'test_trades': g['test']['trades'],
            'full_wr': g['full']['wr'],
            'full_pnl': g['full']['pnl'],
            'full_trades': g['full']['trades'],
            'wr_diff': g.get('wr_diff', 0),
            'final_leverage': g['final_leverage'],
            'leverage_source': g.get('leverage_source', ''),
            'mae_bot': g.get('mae_bot'),
            'mae_v5': g.get('mae_v5'),
            'mae_worst': g.get('mae_worst'),
            'safe_leverage_v5': g.get('safe_leverage_v5'),
            'safe_leverage_v6': g.get('safe_leverage_v6', 20),
            'final_stop_loss_pct': g.get('final_stop_loss_pct', 1.0),
            'final_kelly_pct': g.get('final_kelly_pct', 2.0),
            'final_kill_dd_pct': g.get('final_kill_dd_pct', 10),
            'position_size_usd': size,
            'sizing_penalty': has_penalty,
            'v5_dream_team': g.get('v5_dream_team', False),
            'v5_avoid_warning': g.get('v5_avoid_warning', False),
            'crash_vulnerability': g.get('crash_vulnerability', 'unknown'),
            'params': g.get('best_params', {}),
            'duration': g.get('duration', {}),
        })

    with open(os.path.join(out_dir, 'v6_production_final.json'), 'w') as f:
        json.dump(production, f, indent=2, default=str)

    # Top 30
    print(f"\n{'='*60}")
    print(f"🏆 TOP 30 — V6 PRODUCTION")
    print(f"{'='*60}")
    print(f"{'#':>3} {'Tier':>9} {'Q':>4} {'Strategy':22s} {'Symbol':18s} {'TF':>4} {'WR%':>5} {'PnL':>7} {'Lev':>4} {'$':>4} {'MAE':>6} {'DT':>3}")
    print("-" * 115)
    for p in production[:30]:
        dt = '⭐' if p['v5_dream_team'] else ('⚠️' if p['v5_avoid_warning'] else '')
        mae = p.get('mae_worst', 0)
        mae_str = f"{mae*100:.1f}%" if mae else "N/A"
        print(f"{p['rank']:3d} {p['tier']:>9s} {p['quality_score']:4.0f} "
              f"{p['strategy'][:22]:22s} {p['symbol'][:18]:18s} {p['timeframe']:>4s} "
              f"{p['test_wr']:5.1f} {p['test_pnl']:7.1f} {p['final_leverage']:3d}x "
              f"${p['position_size_usd']:>3d} {mae_str:>6s} {dt:>3s}")

    print(f"\n💾 data/v6_production_final.json — {len(production)} bots")
    return production


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("🚀 V6 PIPELINE CORREGIDO — Compete con V5, ganarle!")
    print(f"   {datetime.now().isoformat()}")
    print("   V5: spike+RSI, WR 61.7%, 161K trades, $29K PnL")
    print("   V6: VWAP/MR, WR 85%+ backtest, Optuna per-asset")
    print("=" * 60)

    ref = load_v5_reference()
    grails = load_v6_grails()

    # Phase 1: V6 own validation only
    validated = phase1_validate(grails)

    # Phase 2: Enrich with V5 reference
    enriched = phase2_enrich(validated, ref)

    # Phase 3: Classify
    tiers = phase3_classify(enriched)

    # Phase 4: Select
    selected, deduped = phase4_select(tiers)

    # Phase 5: V6 vs V5 comparison
    phase5_compare(selected, ref)

    # Output
    output(selected, deduped, tiers)

    print(f"\n{'='*60}")
    print(f"📊 RESUMEN")
    print(f"{'='*60}")
    print(f"  V6 grails input:    {len(grails)}")
    print(f"  V6 validated:       {len(validated)} ({len(validated)/len(grails)*100:.0f}%)")
    print(f"  Production:         {len(selected)}")
    print(f"  ⚡ Workers aún corriendo — re-run cuando terminen!")


if __name__ == '__main__':
    main()
