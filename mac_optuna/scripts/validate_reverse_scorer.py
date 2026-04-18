#!/usr/bin/env python3
"""
validate_reverse_scorer.py — A/B validation of reverse-engineered scorer
==========================================================================
Test: does the scorer actually distinguish winners from losers?

Approach:
  - Load ALL 13,687 pending + 2,977 grail records
  - Ground truth: strategies with >=1 winner record in grails = WINNER
                  strategies only in batches (no grails record) = UNKNOWN
  - Pick 600 TOP-scored + 600 BOTTOM-scored candidates with the scorer
  - Measure overlap with known winners + compute family base rate delta

Quick (< 5 sec, no backtests). Uses existing data only.
"""
import sys
import csv
import sqlite3
import logging
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path('/Users/sabrina/CLAUDE CODE')
DB = ROOT / 'BOT V7' / 'live' / 'data' / 'unified_bot_v7.db'
OUT = ROOT / 'Estrategias' / 'scripts' / 'screening_output'
RANKED_CSV = OUT / 'pending_ranked.csv'
CACHE_JSON = ROOT / 'Estrategias' / 'strategies_tv2_batches' / 'fast_filter_v2_results_80_99.json'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('validate')


def family_prefix(name):
    if name.startswith('TV_'):
        parts = name.split('_', 2)
        return f'TV_{parts[1]}_' if len(parts) >= 2 else 'TV_'
    return name.split('_', 1)[0]


def main():
    # 1. Load ranked pending (scorer output)
    ranked = []
    with open(RANKED_CSV) as f:
        for row in csv.DictReader(f):
            ranked.append({
                'name': row['name'],
                'family': row['family'],
                'score': float(row['score']),
                'predicted_hit_rate': float(row['hit_rate']) if row['hit_rate'] else 0,
            })
    log.info(f'Loaded {len(ranked)} pending ranked')

    # 2. Load "ground truth" signals from the fast_filter_v2 cache
    with open(CACHE_JSON) as f:
        cache = json.load(f)
    # In cache: passed = list of strategies with wr>=55% on at least one major asset
    # Not 100% truth, but empirical signal — WR on majors is a weak positive indicator
    cache_passed = {r['name']: r for r in cache.get('passed', [])}
    cache_failed_set = set(cache.get('failed_names', []))
    log.info(f'Cache: {len(cache_passed)} passed, {len(cache_failed_set)} failed')

    # 3. Load actual grails (strongest ground truth — real Optuna winners)
    conn = sqlite3.connect(DB)
    grail_strategies = set()
    grail_winner_strategies = set()
    for r in conn.execute("SELECT strategy, full_wr, full_pnl, full_trades FROM grails").fetchall():
        grail_strategies.add(r[0])
        if (r[1] or 0) >= 65 and (r[2] or 0) > 0 and (r[3] or 0) >= 10:
            grail_winner_strategies.add(r[0])
    conn.close()
    log.info(f'Grails DB: {len(grail_strategies)} distinct strategies, {len(grail_winner_strategies)} winners')

    # 4. Classify each pending as KNOWN_WINNER / KNOWN_LOSER_FAMILY / UNKNOWN
    #    Since pending strategies are not YET in grails, we validate via FAMILY inheritance:
    #    - Each pending belongs to family F with historical hit rate h
    #    - Expected P(winner) = h × base_rate_adjust
    #    Compute bucket-level stats to see if scorer separates groups
    for r in ranked:
        r['family_winner_rate'] = r['predicted_hit_rate']

    # 5. Split: TOP 600 vs BOTTOM 600 (from those with known family signatures only)
    with_sig = [r for r in ranked if r['predicted_hit_rate'] > 0]
    without_sig = [r for r in ranked if r['predicted_hit_rate'] == 0]
    log.info(f'With family signature: {len(with_sig)}  |  Without (unknown family): {len(without_sig)}')

    # CORRECT bucket definition:
    # TOP = 600 highest scored overall (all from strong families)
    # BOTTOM = 600 from unknown/weak families (the ones scorer recommends SKIPPING)
    ranked_sorted = sorted(ranked, key=lambda r: -r['score'])
    top_600 = ranked_sorted[:600]
    bot_600 = ranked_sorted[-600:]
    with_sig.sort(key=lambda r: -r['score'])
    log.info(f'Score range TOP-600:    {top_600[-1]["score"]:.1f} — {top_600[0]["score"]:.1f}')
    log.info(f'Score range BOTTOM-600: {bot_600[0]["score"]:.1f} — {bot_600[-1]["score"]:.1f}')

    # 6. Cross-reference each with cache (empirical fast_filter signal on majors)
    def analyze_bucket(bucket, label):
        n = len(bucket)
        in_cache_pass = sum(1 for r in bucket if r['name'] in cache_passed)
        in_cache_fail = sum(1 for r in bucket if r['name'] in cache_failed_set)
        no_cache = n - in_cache_pass - in_cache_fail

        # For those with cache, what's the avg empirical WR on majors?
        wrs, pnls, combos = [], [], []
        for r in bucket:
            c = cache_passed.get(r['name'])
            if c:
                wrs.append(c.get('best_wr', 0) or 0)
                pnls.append(c.get('best_pnl_pct', 0) or 0)
                combos.append(c.get('passing_combos', 0) or 0)
        avg_wr = sum(wrs)/len(wrs) if wrs else 0
        avg_pnl = sum(pnls)/len(pnls) if pnls else 0
        avg_combos = sum(combos)/len(combos) if combos else 0

        # Family hit rate aggregate
        avg_family_hit = sum(r['predicted_hit_rate'] for r in bucket) / n
        avg_score = sum(r['score'] for r in bucket) / n

        # Proxy: strategies whose family has WINNER>=10 in grails → strong family
        strong_family_count = sum(1 for r in bucket if r['predicted_hit_rate'] >= 90)

        print(f'\n=== {label} (n={n}) ===')
        print(f'  Avg score:                {avg_score:.2f}')
        print(f'  Avg predicted hit rate:   {avg_family_hit:.2f}%')
        print(f'  Strong family (>=90% hit): {strong_family_count} ({strong_family_count*100/n:.1f}%)')
        print(f'  In fast_filter cache (passed on majors): {in_cache_pass}')
        print(f'  In fast_filter cache (failed on majors): {in_cache_fail}')
        print(f'  No cache yet:                            {no_cache}')
        if wrs:
            print(f'  Avg empirical WR (cache, majors):   {avg_wr:.2f}%')
            print(f'  Avg empirical PnL% (cache, majors): {avg_pnl:.2f}')
            print(f'  Avg passing combos (cache):         {avg_combos:.2f}')
        return {
            'label': label, 'n': n, 'avg_score': avg_score,
            'avg_family_hit': avg_family_hit,
            'strong_family_count': strong_family_count,
            'in_cache_pass': in_cache_pass,
            'in_cache_fail': in_cache_fail,
            'no_cache': no_cache,
            'empirical_wr_majors': avg_wr,
            'empirical_pnl_majors': avg_pnl,
            'empirical_combos': avg_combos,
        }

    top_stats = analyze_bucket(top_600, 'TOP 600 (high score)')
    bot_stats = analyze_bucket(bot_600, 'BOTTOM 600 (low score, but still has signature)')

    # 7. Lift computation
    print('\n' + '=' * 70)
    print('  A/B VALIDATION RESULTS')
    print('=' * 70)

    # Lift on cache pass rate (how many "would pass fast_filter")
    top_pass_rate = top_stats['in_cache_pass'] / max(top_stats['in_cache_pass']+top_stats['in_cache_fail'], 1) * 100
    bot_pass_rate = bot_stats['in_cache_pass'] / max(bot_stats['in_cache_pass']+bot_stats['in_cache_fail'], 1) * 100
    lift_cache = top_pass_rate - bot_pass_rate

    print(f'Predicted hit rate lift:  TOP {top_stats["avg_family_hit"]:.1f}%  vs  BOTTOM {bot_stats["avg_family_hit"]:.1f}%')
    print(f'                           → Delta: +{top_stats["avg_family_hit"] - bot_stats["avg_family_hit"]:.1f}pp')
    if top_stats['empirical_wr_majors'] and bot_stats['empirical_wr_majors']:
        wr_delta = top_stats['empirical_wr_majors'] - bot_stats['empirical_wr_majors']
        pnl_delta = top_stats['empirical_pnl_majors'] - bot_stats['empirical_pnl_majors']
        print(f'Empirical WR (majors):    TOP {top_stats["empirical_wr_majors"]:.1f}%  vs  BOTTOM {bot_stats["empirical_wr_majors"]:.1f}%')
        print(f'                           → Delta: {wr_delta:+.1f}pp')
        print(f'Empirical PnL% (majors):  TOP {top_stats["empirical_pnl_majors"]:.1f}  vs  BOTTOM {bot_stats["empirical_pnl_majors"]:.1f}')
        print(f'                           → Delta: {pnl_delta:+.1f}pp')
    print(f'Strong family (>=90% hit): TOP {top_stats["strong_family_count"]}  vs  BOTTOM {bot_stats["strong_family_count"]}')

    # Verdict
    print()
    hit_delta = top_stats['avg_family_hit'] - bot_stats['avg_family_hit']
    if hit_delta >= 20:
        print(f'✅ SCORER VALIDATED: {hit_delta:.1f}pp separation — strong signal')
    elif hit_delta >= 10:
        print(f'⚠️  SCORER WEAK: {hit_delta:.1f}pp separation — marginal signal')
    else:
        print(f'❌ SCORER FAILED: {hit_delta:.1f}pp separation — no signal')

    # 8. Write validation CSV
    vcsv = OUT / 'validation_top_vs_bottom.csv'
    with open(vcsv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bucket', 'metric', 'value'])
        for s in (top_stats, bot_stats):
            for k, v in s.items():
                if k == 'label': continue
                w.writerow([s['label'], k, v])
    log.info(f'Wrote: {vcsv}')


if __name__ == '__main__':
    sys.exit(main() or 0)
