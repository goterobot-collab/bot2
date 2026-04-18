#!/usr/bin/env python3
"""
FASE 1 — Classify 57K grails into tiers A/B/C/MICRO/DISCARD.

Based on candle coverage in activos_binance.db and smoking gun detection.
Runs in ~2 minutes on Mac.

Tiers:
  A:      ≥730d + ≥100 trades + no smoking gun  → full backtest FIRST
  B:      ≥365d + ≥50 trades + no smoking gun   → full backtest SECOND
  C:      ≥180d + ≥30 trades                    → resume Optuna warm-start
  MICRO:  <180d + WR≥90% + ≥20 trades + PnL+   → shadow-only tracking
  DISCARD: everything else                       → archive (never delete)

Usage:
  python3 scripts/classify_grails_by_coverage.py
  python3 scripts/classify_grails_by_coverage.py --grails data/grails_master.json
"""

import json
import sqlite3
import os
import sys
import argparse
from datetime import datetime
from collections import defaultdict

# Paths
DEFAULT_GRAILS = "/Users/sabrina/CLAUDE CODE/Estrategias/data/grails_master.json"
DB_PATH = "/Users/sabrina/CLAUDE CODE/data/activos_binance.db"
OUTPUT_DIR = "/Users/sabrina/CLAUDE CODE/Estrategias/data"


def get_candle_coverage(db_path):
    """Query days of candle history per (symbol, timeframe)."""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA query_only=ON")
    rows = conn.execute("""
        SELECT symbol, timeframe,
               COUNT(*) as n_candles,
               (MAX(ts) - MIN(ts)) / 86400000.0 as days,
               MIN(ts) as first_ts,
               MAX(ts) as last_ts
        FROM candles
        GROUP BY symbol, timeframe
    """).fetchall()
    conn.close()

    coverage = {}
    for sym, tf, n_candles, days, first_ts, last_ts in rows:
        coverage[(sym, tf)] = {
            'candles': n_candles,
            'days': round(days, 1),
            'first_ts': first_ts,
            'last_ts': last_ts,
        }
    return coverage


def classify_grail(g, coverage):
    """Classify a single grail into a tier."""
    sym = g.get('symbol', '')
    tf = g.get('timeframe', '')
    test_trades = g.get('test_trades', 0) or 0
    train_wr = g.get('train_wr', 0) or 0
    test_wr = g.get('test_wr', 0) or 0
    test_pnl = g.get('test_pnl', 0) or 0

    # For 15m/4h: check coverage of base TF (5m for 15m, 1h for 4h)
    # because these are resampled from base TFs
    base_tf = tf
    if tf == '15m':
        base_tf = '5m'
    elif tf == '4h':
        base_tf = '1h'

    cov = coverage.get((sym, base_tf), {'days': 0, 'candles': 0})
    days = cov.get('days', 0)

    # Smoking gun: test WR > train WR + 5pp = data snooping indicator
    smoking_gun = test_wr > train_wr + 5

    # Classification logic
    if days >= 730 and test_trades >= 100 and not smoking_gun:
        return 'A', days, smoking_gun
    elif days >= 365 and test_trades >= 50 and not smoking_gun:
        return 'B', days, smoking_gun
    elif days >= 180 and test_trades >= 30:
        return 'C', days, smoking_gun
    elif days < 180 and test_wr >= 90 and test_trades >= 20 and test_pnl > 0:
        return 'MICRO', days, smoking_gun
    else:
        return 'DISCARD', days, smoking_gun


def main():
    parser = argparse.ArgumentParser(description='FASE 1: Classify grails by coverage')
    parser.add_argument('--grails', default=DEFAULT_GRAILS, help='Path to grails_master.json')
    parser.add_argument('--db', default=DB_PATH, help='Path to activos_binance.db')
    parser.add_argument('--out-dir', default=OUTPUT_DIR, help='Output directory')
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"FASE 1 — CLASSIFY GRAILS BY COVERAGE")
    print(f"{'='*60}")
    print(f"Grails:  {args.grails}")
    print(f"DB:      {args.db}")
    print(f"Output:  {args.out_dir}")
    print()

    # 1. Load grails
    print("Loading grails_master.json...", end=' ', flush=True)
    with open(args.grails) as f:
        grails = json.load(f)
    print(f"{len(grails):,} grails loaded")

    # 2. Get candle coverage
    print("Querying candle coverage...", end=' ', flush=True)
    coverage = get_candle_coverage(args.db)
    print(f"{len(coverage)} (symbol, tf) pairs found")

    # 3. Classify each grail
    print("Classifying...", end=' ', flush=True)
    tiers = defaultdict(list)
    smoking_gun_count = 0
    strategy_stats = defaultdict(lambda: defaultdict(int))

    for g in grails:
        tier, days, smoking_gun = classify_grail(g, coverage)
        g['tier'] = tier
        g['candle_days'] = round(days, 1)
        g['smoking_gun'] = smoking_gun
        tiers[tier].append(g)

        if smoking_gun:
            smoking_gun_count += 1

        strategy_stats[g.get('strategy', 'unknown')][tier] += 1

    print("DONE")
    print()

    # 4. Save tier files
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{'Tier':<10} {'Count':>8} {'%':>7}  File")
    print(f"{'-'*10} {'-'*8} {'-'*7}  {'-'*40}")

    total = len(grails)
    for tier_name in ['A', 'B', 'C', 'MICRO', 'DISCARD']:
        items = tiers.get(tier_name, [])
        pct = len(items) / total * 100 if total > 0 else 0
        filename = f"grails_tier_{tier_name}.json"
        filepath = os.path.join(args.out_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(items, f, indent=2, default=str)

        print(f"{tier_name:<10} {len(items):>8,} {pct:>6.1f}%  {filepath}")

    # 5. Summary stats
    print()
    print(f"{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total grails:          {total:,}")
    print(f"Smoking gun detected:  {smoking_gun_count:,} ({smoking_gun_count/total*100:.1f}%)")
    print(f"Tier A (backtest 1st): {len(tiers['A']):,}")
    print(f"Tier B (backtest 2nd): {len(tiers['B']):,}")
    print(f"  → Total for FASE 2:  {len(tiers['A']) + len(tiers['B']):,}")
    print(f"Tier C (resume Optuna):{len(tiers['C']):,}")
    print(f"Tier MICRO (shadow):   {len(tiers['MICRO']):,}")
    print(f"Tier DISCARD:          {len(tiers['DISCARD']):,}")
    print()

    # 6. Top strategies in tier A+B
    ab_strategies = defaultdict(int)
    for g in tiers['A'] + tiers['B']:
        ab_strategies[g['strategy']] += 1

    if ab_strategies:
        print("Top strategies in Tier A+B:")
        for strat, count in sorted(ab_strategies.items(), key=lambda x: -x[1])[:15]:
            print(f"  {strat:<30} {count:>4} grails")

    # 7. TF distribution in A+B
    tf_dist = defaultdict(int)
    for g in tiers['A'] + tiers['B']:
        tf_dist[g['timeframe']] += 1
    print()
    print("Timeframe distribution (Tier A+B):")
    for tf, count in sorted(tf_dist.items()):
        print(f"  {tf:<6} {count:>4} grails")

    # 8. Save classification report
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_grails': total,
        'smoking_gun_count': smoking_gun_count,
        'tiers': {t: len(items) for t, items in tiers.items()},
        'top_ab_strategies': dict(sorted(ab_strategies.items(), key=lambda x: -x[1])[:20]),
        'tf_distribution_ab': dict(tf_dist),
    }
    report_path = os.path.join(args.out_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: {report_path}")

    print(f"\n{'='*60}")
    print(f"FASE 1 COMPLETE — Ready for FASE 2 (full_historical_backtest.py)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
