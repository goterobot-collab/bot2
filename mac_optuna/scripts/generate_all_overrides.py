#!/usr/bin/env python3
"""
GENERADOR DE OVERRIDES UNIFICADO
=================================
Toma resultados de forensic V2 (producción + Hetzner) y genera:
1. hour_overrides.json — combos que ganan en horas bloqueadas
2. rvol_overrides actualizado — combos que ganan con RVOL bajo

Input: forensic V2 JSONs con wr_per_hour
Output: BOT V7/live/data/hour_overrides.json (hot-reload sin restart)

Regla 29: TODOS los filtros per-activo × per-estrategia, NUNCA genéricos.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════
BOT_V7_DIR = Path("/Users/sabrina/CLAUDE CODE/BOT V7")
ESTRATEGIAS_DIR = Path("/Users/sabrina/CLAUDE CODE/Estrategias")
HOUR_OVERRIDES_PATH = BOT_V7_DIR / "live" / "data" / "hour_overrides.json"
RVOL_OVERRIDES_PATH = BOT_V7_DIR / "live" / "data" / "rvol_overrides.json"

# ═══════════════════════════════════════════════════════════════════════
# BLOCKED HOURS (same as signal_factory.py)
# ═══════════════════════════════════════════════════════════════════════
HARD_BLOCK_HOURS = {5}
LOW_WR_HOURS = {3, 4, 6, 7, 8, 19, 20}

# ═══════════════════════════════════════════════════════════════════════
# GATES — minimum criteria for override
# ═══════════════════════════════════════════════════════════════════════
MIN_TRADES_PER_HOUR = 8       # Minimum trades in that hour to trust WR
MIN_WR_FOR_OVERRIDE = 65.0    # Minimum WR to bypass hour block
MIN_PNL_FOR_OVERRIDE = 0.0    # Must be PnL positive in that hour
MIN_TOTAL_TRADES = 25         # Minimum total trades for the grail (agreed with Sabrina)
MIN_TOTAL_WR = 55.0           # Minimum overall WR for the grail


def load_forensic_results(*paths):
    """Load and merge multiple forensic V2 result files."""
    all_results = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"  SKIP (not found): {p}")
            continue
        data = json.loads(p.read_text())
        if isinstance(data, dict) and 'results' in data:
            results = data['results']
        elif isinstance(data, list):
            results = data
        else:
            print(f"  SKIP (unknown format): {p}")
            continue

        # Only include results that have wr_per_hour
        with_wph = [r for r in results if r.get('metrics', {}).get('wr_per_hour')]
        without_wph = len(results) - len(with_wph)

        print(f"  Loaded {p.name}: {len(results)} results ({len(with_wph)} with wr_per_hour, {without_wph} without)")
        all_results.extend(with_wph)

    return all_results


def generate_hour_overrides(results):
    """Generate hour_overrides.json from forensic V2 results with wr_per_hour."""
    overrides = {}
    stats = {
        'total_combos_analyzed': 0,
        'combos_with_overrides': 0,
        'total_hours_bypassed': 0,
        'hard_block_bypasses': 0,
        'low_wr_bypasses': 0,
    }

    blocked_hours = HARD_BLOCK_HOURS | LOW_WR_HOURS

    for r in results:
        strat = r.get('strategy', '')
        sym = r.get('symbol', '')
        tf = r.get('timeframe', '')
        metrics = r.get('metrics', {})
        wr_per_hour = metrics.get('wr_per_hour', {})
        total_trades = metrics.get('total_trades', 0)
        total_wr = metrics.get('win_rate', 0)

        if not strat or not sym or not tf:
            continue
        if total_trades < MIN_TOTAL_TRADES:
            continue
        if total_wr < MIN_TOTAL_WR:
            continue

        stats['total_combos_analyzed'] += 1
        combo_key = f"{strat}|{sym}|{tf}"
        allowed_hours = []

        for hour_str, hour_data in wr_per_hour.items():
            hour = int(hour_str)

            # Only consider hours that are currently blocked
            if hour not in blocked_hours:
                continue

            trades_h = hour_data.get('trades', 0)
            wr_h = hour_data.get('wr', 0)
            pnl_h = hour_data.get('pnl', 0)

            # Gate: enough trades + profitable in this hour
            if trades_h >= MIN_TRADES_PER_HOUR and wr_h >= MIN_WR_FOR_OVERRIDE and pnl_h >= MIN_PNL_FOR_OVERRIDE:
                allowed_hours.append({
                    'hour': hour,
                    'wr': wr_h,
                    'trades': trades_h,
                    'pnl': round(pnl_h, 2)
                })

                if hour in HARD_BLOCK_HOURS:
                    stats['hard_block_bypasses'] += 1
                else:
                    stats['low_wr_bypasses'] += 1

        if allowed_hours:
            overrides[combo_key] = {
                'allowed_hours': sorted(allowed_hours, key=lambda x: x['hour']),
                'total_wr': round(total_wr, 1),
                'total_trades': total_trades,
                'source': 'forensic_v2'
            }
            stats['combos_with_overrides'] += 1
            stats['total_hours_bypassed'] += len(allowed_hours)

    return overrides, stats


def main():
    print("=" * 70)
    print("GENERADOR DE OVERRIDES UNIFICADO — Forensic V2")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # ── Load all forensic V2 results ──
    print("Loading forensic V2 results...")
    forensic_files = [
        ESTRATEGIAS_DIR / "data" / "forensic_v2_ALL_1548_production.json",
        ESTRATEGIAS_DIR / "data" / "forensic_v2_hetzner_23K_LIGHT.json",  # 34MB vs 7.6GB
    ]

    results = load_forensic_results(*forensic_files)
    print(f"\nTotal results with wr_per_hour: {len(results)}")

    if not results:
        print("ERROR: No results with wr_per_hour found!")
        sys.exit(1)

    # ── Generate hour overrides ──
    print("\n" + "─" * 50)
    print("GENERATING HOUR OVERRIDES")
    print(f"Gates: trades/hour >= {MIN_TRADES_PER_HOUR}, WR >= {MIN_WR_FOR_OVERRIDE}%, PnL >= {MIN_PNL_FOR_OVERRIDE}")
    print(f"Blocked hours: hard_block={HARD_BLOCK_HOURS}, low_wr={LOW_WR_HOURS}")

    overrides, stats = generate_hour_overrides(results)

    # ── Save hour_overrides.json ──
    output = {
        'generated_at': datetime.now().isoformat(),
        'generated_by': 'generate_all_overrides.py',
        'sources': [str(f) for f in forensic_files],
        'gates': {
            'min_trades_per_hour': MIN_TRADES_PER_HOUR,
            'min_wr': MIN_WR_FOR_OVERRIDE,
            'min_pnl': MIN_PNL_FOR_OVERRIDE,
            'min_total_trades': MIN_TOTAL_TRADES,
            'min_total_wr': MIN_TOTAL_WR,
        },
        'stats': stats,
        'overrides': overrides
    }

    # Backup existing
    if HOUR_OVERRIDES_PATH.exists():
        bak = HOUR_OVERRIDES_PATH.with_suffix(f'.json.bak_{datetime.now().strftime("%H%M")}')
        HOUR_OVERRIDES_PATH.rename(bak)
        print(f"\nBackup: {bak}")

    HOUR_OVERRIDES_PATH.write_text(json.dumps(output, indent=1))
    print(f"Saved: {HOUR_OVERRIDES_PATH}")
    print(f"Size: {HOUR_OVERRIDES_PATH.stat().st_size / 1024:.1f} KB")

    # ── Report ──
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Combos analyzed:      {stats['total_combos_analyzed']}")
    print(f"Combos with overrides: {stats['combos_with_overrides']}")
    print(f"Total hour bypasses:   {stats['total_hours_bypassed']}")
    print(f"  hard_block (h5):     {stats['hard_block_bypasses']}")
    print(f"  low_wr (h3-8,19-20): {stats['low_wr_bypasses']}")

    # ── Sample overrides ──
    if overrides:
        print(f"\nSample overrides (first 5):")
        for i, (k, v) in enumerate(list(overrides.items())[:5]):
            hours = [ah['hour'] for ah in v['allowed_hours']]
            print(f"  {k}: hours={hours} total_wr={v['total_wr']}% trades={v['total_trades']}")

    print("\nDONE — hour_overrides.json will hot-reload on next signal_factory cycle")


if __name__ == '__main__':
    main()
