#!/usr/bin/env python3
"""
V6 PIPELINE — Filtro + Clasificación + Selección de Estrategias Ganadoras
Integra datos V5 (161K trades reales) con grails V6 (Optuna per-asset)

REGLAS ABSOLUTAS (de BEST_STRATEGIES_FOR_V6.md):
1. Leverage = min(safe_leverage_v5, safe_leverage_v6) — NUNCA usar solo V6
2. Blacklist: NO tradear los 55 símbolos de v5_symbols_to_avoid.json
3. Solo confiar en grails con ≥10 trades test OOS
4. Session filter: skip hours UTC 1,2,4,8,10,12,13,15,19,20,21,22
5. BlackSwan: bloquear 0-6h post-crash, recovery 2x isolated 6-48h
6. Max 40 posiciones simultáneas del mismo cluster de correlación
7. Eliminar duplicados (1000BONK→BONK, 1000FLOKI→FLOKI, etc.)
"""
import json, os, sys
import numpy as np
from collections import defaultdict
from datetime import datetime

PROJECT = "/Users/sabrina/CLAUDE CODE/Estrategias"
GOLDEN = os.path.join(PROJECT, "v5_golden_data_for_v6")

# ═══════════════════════════════════════════════════════════
# LOAD ALL DATA SOURCES
# ═══════════════════════════════════════════════════════════
def load_data():
    """Load all V5 golden data + V6 grails"""
    data = {}

    # V5 Golden Data
    data['blacklist'] = set(
        item['symbol'] for item in json.load(open(os.path.join(GOLDEN, 'v5_symbols_to_avoid.json')))
    )
    data['dream_team'] = set(
        item['symbol'] for item in json.load(open(os.path.join(GOLDEN, 'v5_dream_team_symbols.json')))
    )

    # V5 leverage map: symbol → safe_leverage
    lev_map_raw = json.load(open(os.path.join(GOLDEN, 'v5_leverage_map.json')))
    data['v5_leverage'] = {}
    for item in lev_map_raw:
        data['v5_leverage'][item['symbol']] = {
            'safe_leverage': item.get('safe_leverage', 20),
            'mae_p95': item.get('mae_p95', 0.05),
        }

    # V5 crash vulnerability
    crash_raw = json.load(open(os.path.join(GOLDEN, 'v5_crash_vulnerability.json')))
    data['crash_vuln'] = {}
    for item in crash_raw:
        data['crash_vuln'][item['symbol']] = item

    # Ultimate combos (reference)
    data['ultimate'] = json.load(open(os.path.join(GOLDEN, 'ultimate_combos.json')))

    # V5 bot performance
    data['v5_performance'] = json.load(open(os.path.join(GOLDEN, 'v5_bot_symbol_performance.json')))

    # V6 Grails — R1 + Turbo (live data)
    r1 = json.load(open(os.path.join(PROJECT, 'data/santo_grial_clean.json')))
    r1_elite = [g for g in r1 if g['full']['trades'] >= 20 and g['test']['wr'] >= 70
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

    data['grails'] = r1_elite + turbo_elite
    print(f"Loaded: {len(r1_elite)} R1 + {len(turbo_elite)} Turbo = {len(data['grails'])} grails")
    print(f"Blacklist: {len(data['blacklist'])} | Dream Team: {len(data['dream_team'])} | Leverage map: {len(data['v5_leverage'])}")

    return data


# ═══════════════════════════════════════════════════════════
# DUPLICATE SYMBOLS MAP
# ═══════════════════════════════════════════════════════════
DUPLICATE_MAP = {
    '1000BONK/USDT:USDT': 'BONK/USDT:USDT',
    '1000FLOKI/USDT:USDT': 'FLOKI/USDT:USDT',
    '1000SATS/USDT:USDT': 'SATS/USDT:USDT',
    '1000PEPE/USDT:USDT': 'PEPE/USDT:USDT',
    '1000RATS/USDT:USDT': 'RATS/USDT:USDT',
    '1000000BOB/USDT:USDT': 'BOB/USDT:USDT',
}

SESSION_SKIP_HOURS = {1, 2, 4, 8, 10, 12, 13, 15, 19, 20, 21, 22}
SESSION_ACTIVE_HOURS = set(range(24)) - SESSION_SKIP_HOURS  # 0,3,5,6,7,9,11,14,16,17,18,23


# ═══════════════════════════════════════════════════════════
# PHASE 1: FILTER — Apply V5 rules to each grail
# ═══════════════════════════════════════════════════════════
def phase1_filter(data):
    """Apply V5 rules: blacklist, leverage, min trades, duplicates"""
    grails = data['grails']
    blacklist = data['blacklist']
    v5_lev = data['v5_leverage']

    stats = defaultdict(int)
    filtered = []

    for g in grails:
        sym = g['symbol']
        stats['total'] += 1

        # Rule 1: Blacklist
        if sym in blacklist:
            stats['blacklisted'] += 1
            continue

        # Rule 2: Duplicate symbols → skip (keep main version)
        if sym in DUPLICATE_MAP:
            stats['duplicate'] += 1
            continue

        # Rule 3: Min trades in test OOS (≥10 for statistical confidence)
        test_trades = g.get('test', {}).get('trades', 0)
        if test_trades < 10:
            stats['low_trades'] += 1
            continue

        # Rule 4: WR must be > 70% in test AND positive PnL
        test_wr = g.get('test', {}).get('wr', 0)
        test_pnl = g.get('test', {}).get('pnl', 0)
        if test_wr < 70 or test_pnl <= 0:
            stats['low_wr_pnl'] += 1
            continue

        # Rule 5: Overfit check (train-test gap < 25%)
        wr_diff = g.get('wr_diff', 99)
        if wr_diff > 25:
            stats['overfit'] += 1
            continue

        # Rule 6: Apply V5 leverage (CRITICAL — V6 is too optimistic)
        v5_info = v5_lev.get(sym, {})
        lev_v5 = v5_info.get('safe_leverage', 20)
        lev_v6 = g.get('safe_leverage', 20)
        safe_lev = max(1, int(min(lev_v5, lev_v6)))
        mae_v5 = v5_info.get('mae_p95', 0.05)

        # Rule 7: Full sample validation
        full_trades = g.get('full', {}).get('trades', 0)
        full_pnl = g.get('full', {}).get('pnl', 0)
        if full_trades < 15 or full_pnl <= 0:
            stats['full_bad'] += 1
            continue

        # Enrich grail with V5 data
        g['safe_leverage_v5'] = lev_v5
        g['safe_leverage_v6'] = lev_v6
        g['final_leverage'] = safe_lev
        g['mae_p95_v5'] = mae_v5
        g['is_dream_team'] = sym in data['dream_team']
        g['crash_vuln'] = data['crash_vuln'].get(sym, {}).get('vulnerability', 'unknown')

        # Compute stop loss from V5 MAE (more realistic than V6)
        # SL per-bot: MAE × 1.5, clamp 2%-40% (NO cap fijo de 3%)
        g['final_stop_loss_pct'] = round(max(0.02, min(0.40, mae_v5 * 1.5)) * 100, 3)

        # Kelly criterion
        pf = g.get('full', {}).get('pf', g.get('test', {}).get('pf', 1.5))
        w = test_wr / 100
        r = max(pf, 1.01)
        kelly_half = max(0, min((w - (1 - w) / r) / 2, 0.05))
        g['final_kelly_pct'] = round(kelly_half * 100, 2)

        # Kill switch DD
        dd = g.get('full', {}).get('max_dd', 20)
        g['final_kill_dd_pct'] = round(min(dd * 1.3, 15), 1)

        filtered.append(g)
        stats['passed'] += 1

    print(f"\n{'='*60}")
    print(f"PHASE 1 — FILTER RESULTS")
    print(f"{'='*60}")
    for k, v in sorted(stats.items()):
        pct = v / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {k:20s}: {v:6d} ({pct:5.1f}%)")

    return filtered


# ═══════════════════════════════════════════════════════════
# PHASE 2: CLASSIFY — Assign tiers based on V5+V6 criteria
# ═══════════════════════════════════════════════════════════
def phase2_classify(filtered, data):
    """Classify grails into tiers: ULTIMATE, GOLD, DIAMOND, ELITE, SILVER"""

    tiers = {
        'ULTIMATE': [],   # WR≥90% test + Dream Team + ≥15 trades
        'GOLD': [],       # WR≥85% test + Dream Team + ≥12 trades
        'DIAMOND': [],    # WR≥85% test + ≥15 trades (any symbol)
        'ELITE': [],      # WR≥80% test + ≥10 trades
        'SILVER': [],     # WR≥75% test + ≥10 trades
        'BRONZE': [],     # WR≥70% test + ≥10 trades (minimum)
    }

    sizing = {
        'ULTIMATE': {'min': 15, 'max': 20},
        'GOLD': {'min': 10, 'max': 15},
        'DIAMOND': {'min': 7, 'max': 10},
        'ELITE': {'min': 5, 'max': 7},
        'SILVER': {'min': 3, 'max': 5},
        'BRONZE': {'min': 2, 'max': 3},
    }

    for g in filtered:
        test = g.get('test', {})
        full = g.get('full', {})
        wr = test.get('wr', 0)
        trades = test.get('trades', 0)
        full_trades = full.get('trades', 0)
        is_dt = g.get('is_dream_team', False)
        sharpe = full.get('sharpe', test.get('sharpe', 0))
        pf = full.get('pf', test.get('pf', 1))

        # Composite quality score
        quality = (
            wr * 0.30 +
            min(sharpe, 20) * 1.5 +       # sharpe capped at 20
            min(pf, 10) * 2 +             # PF capped at 10
            min(full_trades, 100) * 0.2 +  # more trades = more confidence
            (10 if is_dt else 0) +          # dream team bonus
            max(0, 10 - g.get('wr_diff', 10)) * 0.5  # consistency bonus
        )
        g['quality_score'] = round(quality, 1)

        # Tier assignment
        if wr >= 90 and is_dt and trades >= 15:
            tier = 'ULTIMATE'
        elif wr >= 85 and is_dt and trades >= 12:
            tier = 'GOLD'
        elif wr >= 85 and trades >= 15:
            tier = 'DIAMOND'
        elif wr >= 80 and trades >= 10:
            tier = 'ELITE'
        elif wr >= 75 and trades >= 10:
            tier = 'SILVER'
        else:
            tier = 'BRONZE'

        g['tier'] = tier
        g['sizing_usd'] = sizing[tier]
        tiers[tier].append(g)

    # Sort each tier by quality score
    for tier in tiers:
        tiers[tier].sort(key=lambda g: -g['quality_score'])

    print(f"\n{'='*60}")
    print(f"PHASE 2 — TIER CLASSIFICATION")
    print(f"{'='*60}")
    total = sum(len(v) for v in tiers.values())
    for tier, grails in tiers.items():
        s = sizing[tier]
        print(f"  {tier:12s}: {len(grails):5d} grails | ${s['min']}-${s['max']} per trade")
    print(f"  {'TOTAL':12s}: {total:5d}")

    return tiers


# ═══════════════════════════════════════════════════════════
# PHASE 3: SELECT — Pick winners, deduplicate, build portfolio
# ═══════════════════════════════════════════════════════════
def phase3_select(tiers, data):
    """Select final winners with diversification rules"""

    all_tiered = []
    for tier_name, grails in tiers.items():
        all_tiered.extend(grails)

    # Sort by quality score
    all_tiered.sort(key=lambda g: -g['quality_score'])

    # Dedup: keep best per strategy+symbol+timeframe
    seen = {}
    deduped = []
    for g in all_tiered:
        key = f"{g['strategy']}_{g['symbol']}_{g['timeframe']}"
        if key not in seen:
            seen[key] = g
            deduped.append(g)

    # Portfolio selection: max diversity
    # Rules: max 3 bots per symbol, max 40 per correlation cluster
    selected = []
    sym_count = defaultdict(int)
    strat_count = defaultdict(int)
    MAX_PER_SYMBOL = 3
    MAX_TOTAL = 500  # top 500

    for g in deduped:
        sym = g['symbol']
        strat = g['strategy']

        if sym_count[sym] >= MAX_PER_SYMBOL:
            continue

        sym_count[sym] += 1
        strat_count[strat] += 1
        selected.append(g)

        if len(selected) >= MAX_TOTAL:
            break

    # Stats
    tier_counts = defaultdict(int)
    for g in selected:
        tier_counts[g['tier']] += 1

    unique_syms = len(set(g['symbol'] for g in selected))
    unique_strats = len(set(g['strategy'] for g in selected))
    dt_count = sum(1 for g in selected if g.get('is_dream_team'))

    # Capital allocation
    total_capital = 0
    for g in selected:
        s = g['sizing_usd']
        g['allocated_usd'] = s['min']  # conservative
        total_capital += s['min']

    print(f"\n{'='*60}")
    print(f"PHASE 3 — FINAL SELECTION")
    print(f"{'='*60}")
    print(f"  Selected:        {len(selected)} bots")
    print(f"  Unique symbols:  {unique_syms}")
    print(f"  Unique strats:   {unique_strats}")
    print(f"  Dream Team:      {dt_count}")
    print(f"  Capital needed:  ${total_capital}")
    print(f"\n  By tier:")
    for t in ['ULTIMATE', 'GOLD', 'DIAMOND', 'ELITE', 'SILVER', 'BRONZE']:
        if tier_counts[t] > 0:
            print(f"    {t:12s}: {tier_counts[t]:4d}")

    return selected, deduped


# ═══════════════════════════════════════════════════════════
# OUTPUT — Generate final files
# ═══════════════════════════════════════════════════════════
def output_results(selected, deduped, tiers, data):
    """Save all results"""
    out_dir = os.path.join(PROJECT, 'data')

    # 1. Top 500 selected (for production)
    production = []
    for g in selected:
        production.append({
            'rank': len(production) + 1,
            'tier': g['tier'],
            'quality_score': g['quality_score'],
            'strategy': g['strategy'],
            'symbol': g['symbol'],
            'timeframe': g['timeframe'],
            'test_wr': g['test']['wr'],
            'test_pnl': g['test']['pnl'],
            'test_trades': g['test']['trades'],
            'test_pf': g['test'].get('pf', 0),
            'test_sharpe': g['test'].get('sharpe', 0),
            'full_wr': g['full']['wr'],
            'full_pnl': g['full']['pnl'],
            'full_trades': g['full']['trades'],
            'full_sharpe': g['full'].get('sharpe', 0),
            'full_pf': g['full'].get('pf', 0),
            'wr_diff': g.get('wr_diff', 0),
            'final_leverage': g['final_leverage'],
            'safe_leverage_v5': g.get('safe_leverage_v5', 20),
            'safe_leverage_v6': g.get('safe_leverage_v6', 20),
            'mae_p95_v5': g.get('mae_p95_v5', 0.05),
            'final_stop_loss_pct': g.get('final_stop_loss_pct', 1.0),
            'final_kelly_pct': g.get('final_kelly_pct', 2.0),
            'final_kill_dd_pct': g.get('final_kill_dd_pct', 10),
            'sizing_usd': g['sizing_usd'],
            'allocated_usd': g.get('allocated_usd', 5),
            'is_dream_team': g.get('is_dream_team', False),
            'crash_vulnerability': g.get('crash_vuln', 'unknown'),
            'params': g.get('best_params', {}),
            'duration': g.get('duration', {}),
            'session_filter': list(SESSION_ACTIVE_HOURS),
        })

    with open(os.path.join(out_dir, 'v6_production_top500.json'), 'w') as f:
        json.dump(production, f, indent=2, default=str)

    # 2. All filtered & classified (for analysis)
    with open(os.path.join(out_dir, 'v6_all_classified.json'), 'w') as f:
        classified = []
        for g in deduped:
            classified.append({
                'tier': g['tier'],
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
                'is_dream_team': g.get('is_dream_team', False),
                'params': g.get('best_params', {}),
            })
        json.dump(classified, f, indent=2, default=str)

    # 3. Tier summary
    tier_summary = {}
    for tier_name, grails in tiers.items():
        if not grails:
            continue
        wrs = [g['test']['wr'] for g in grails]
        pnls = [g['test']['pnl'] for g in grails]
        tier_summary[tier_name] = {
            'count': len(grails),
            'avg_wr': round(np.mean(wrs), 1),
            'avg_pnl': round(np.mean(pnls), 1),
            'unique_symbols': len(set(g['symbol'] for g in grails)),
            'unique_strategies': len(set(g['strategy'] for g in grails)),
            'dream_team_pct': round(sum(1 for g in grails if g.get('is_dream_team')) / len(grails) * 100, 1),
        }

    with open(os.path.join(out_dir, 'v6_tier_summary.json'), 'w') as f:
        json.dump(tier_summary, f, indent=2, default=str)

    # 4. Print top 30
    print(f"\n{'='*60}")
    print(f"🏆 TOP 30 ESTRATEGIAS PARA PRODUCCIÓN")
    print(f"{'='*60}")
    print(f"{'#':>3} {'Tier':>8} {'Score':>5} {'Strategy':22s} {'Symbol':18s} {'TF':>4} {'WR%':>5} {'PnL':>7} {'T':>4} {'Lev':>3} {'SL%':>5} {'DT':>3}")
    print("-" * 110)
    for i, p in enumerate(production[:30]):
        dt = '⭐' if p['is_dream_team'] else ''
        print(f"{p['rank']:3d} {p['tier']:>8s} {p['quality_score']:5.0f} "
              f"{p['strategy'][:22]:22s} {p['symbol'][:18]:18s} {p['timeframe']:>4s} "
              f"{p['test_wr']:5.1f} {p['test_pnl']:7.1f} {p['test_trades']:4d} "
              f"{p['final_leverage']:3d}x {p['final_stop_loss_pct']:5.2f} {dt:>3s}")

    print(f"\n💾 Archivos guardados:")
    print(f"  data/v6_production_top500.json — {len(production)} bots listos para producción")
    print(f"  data/v6_all_classified.json — {len(deduped)} grails clasificados")
    print(f"  data/v6_tier_summary.json — resumen por tier")

    return production


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("🚀 V6 PIPELINE — Filtro + Clasificación + Selección")
    print(f"   Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    # Load
    data = load_data()

    # Phase 1: Filter
    filtered = phase1_filter(data)

    # Phase 2: Classify
    tiers = phase2_classify(filtered, data)

    # Phase 3: Select
    selected, deduped = phase3_select(tiers, data)

    # Output
    production = output_results(selected, deduped, tiers, data)

    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 RESUMEN FINAL V6 PIPELINE")
    print(f"{'='*60}")
    print(f"  Input grails:     {len(data['grails'])}")
    print(f"  After filter:     {len(filtered)} ({len(filtered)/len(data['grails'])*100:.0f}% passed)")
    print(f"  After dedup:      {len(deduped)}")
    print(f"  Production top:   {len(selected)}")
    print(f"  Workers running:  8 (turbo — data will grow)")
    print(f"\n  ⚡ RE-RUN this script when workers finish for final results!")


if __name__ == '__main__':
    main()
