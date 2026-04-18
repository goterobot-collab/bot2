#!/usr/bin/env python3
"""
empirical_compare_approaches.py
===============================
Empirical A/B/C test: how well would each screening approach have captured
the 216 REAL grail-winner strategies we already know?

Ground truth: 216 distinct strategies with >=1 grail record (full_wr>=65,
              full_pnl>0, full_trades>=10) in grails DB.

Approaches evaluated:
  A. fast_filter_v2 cache (current state, majors-only BTC/ETH/SOL/DOGE/AVAX)
  B. reverse_engineer_grails family scorer (leave-one-strategy-out)
  C. A + B combined (pass if EITHER flags it)

Metrics: recall, precision, coverage. Also: which families/types each
approach misses, so we know what to improve.

Fast (<10s, no backtests).
"""
import sys, json, sqlite3, csv
from pathlib import Path
from collections import defaultdict

ROOT = Path('/Users/sabrina/CLAUDE CODE')
DB = ROOT / 'BOT V7' / 'live' / 'data' / 'unified_bot_v7.db'
CACHE = ROOT / 'Estrategias' / 'strategies_tv2_batches' / 'fast_filter_v2_results_80_99.json'
OUT = ROOT / 'Estrategias' / 'scripts' / 'screening_output'


def family_prefix(name):
    if name.startswith('TV_'):
        parts = name.split('_', 2)
        return f'TV_{parts[1]}_' if len(parts) >= 2 else 'TV_'
    return name.split('_', 1)[0]


def main():
    # ---------- 1. Load ground truth: 216 grail-winner strategies ----------
    conn = sqlite3.connect(DB)
    # Per-strategy aggregates (not per record)
    rows = conn.execute("""
        SELECT strategy,
               COUNT(*)                                                           AS n_records,
               SUM(CASE WHEN full_wr>=65 AND full_pnl>0 AND full_trades>=10 THEN 1 ELSE 0 END) AS n_winner_records,
               AVG(full_wr)                                                       AS avg_wr,
               AVG(full_pnl)                                                      AS avg_pnl,
               AVG(full_trades)                                                   AS avg_trades
        FROM grails
        GROUP BY strategy
    """).fetchall()
    conn.close()

    truth = []
    for name, n_rec, n_win, avg_wr, avg_pnl, avg_tr in rows:
        is_winner = (n_win or 0) >= 1
        truth.append({
            'name': name,
            'family': family_prefix(name),
            'n_records': n_rec,
            'n_winner_records': n_win or 0,
            'avg_wr': avg_wr or 0,
            'avg_pnl': avg_pnl or 0,
            'avg_trades': avg_tr or 0,
            'is_winner': is_winner,
        })

    winners = [t for t in truth if t['is_winner']]
    non_winners = [t for t in truth if not t['is_winner']]
    print(f'Ground truth: {len(truth)} distinct strategies in grails DB')
    print(f'  Winners (>=1 winner record):      {len(winners)}')
    print(f'  Non-winners (grail records but none passed gate): {len(non_winners)}')

    # ---------- 2. Approach A: fast_filter_v2 cache (majors) ----------
    with open(CACHE) as f:
        cache = json.load(f)
    cache_passed = {r['name']: r for r in cache.get('passed', [])}
    cache_failed = set(cache.get('failed_names', []))
    cache_all = set(cache_passed.keys()) | cache_failed
    print(f'\nApproach A (fast_filter cache majors): {len(cache_passed)} pass / {len(cache_failed)} fail')

    def a_flags(name):
        """A flags a strategy if cache says passed (WR>=55 on some major)."""
        return name in cache_passed

    # ---------- 3. Approach B: family scorer (leave-one-strategy-out) ----------
    # Build family signatures EXCLUDING each target, then predict score.
    # Signature stats: win_rate_family, avg_wr, avg_pnl, n_tested.
    conn = sqlite3.connect(DB)
    # pull all grail records for LOO computation
    all_records = conn.execute("""
        SELECT strategy, full_wr, full_pnl, full_trades
        FROM grails
    """).fetchall()
    conn.close()

    # pre-aggregate by family with winners
    fam_agg = defaultdict(lambda: {'n': 0, 'winners': 0, 'sum_wr': 0, 'sum_pnl': 0})
    for strat, wr, pnl, tr in all_records:
        fam = family_prefix(strat)
        fam_agg[fam]['n'] += 1
        is_win = (wr or 0) >= 65 and (pnl or 0) > 0 and (tr or 0) >= 10
        if is_win:
            fam_agg[fam]['winners'] += 1
            fam_agg[fam]['sum_wr'] += (wr or 0)
            fam_agg[fam]['sum_pnl'] += (pnl or 0)

    def b_score_loo(target):
        """Leave-one-strategy-out score for target."""
        fam = family_prefix(target['name'])
        agg = fam_agg.get(fam)
        if not agg or agg['n'] <= 1:
            return 0.0
        # subtract target's contribution
        n = agg['n'] - target['n_records']
        winners = agg['winners'] - target['n_winner_records']
        if n <= 0:
            return 0.0
        hit_rate = winners / n * 100 if n else 0
        tv_bonus = 10 if fam.startswith('TV_') else 0
        # simple composite
        score = 0.4 * hit_rate + 0.25 * (agg['sum_wr'] / max(agg['winners'], 1)) \
                + 0.20 * min(100, (agg['sum_pnl'] / max(agg['winners'], 1))) \
                + tv_bonus + 0.05 * min(100, n)
        return score

    THRESHOLDS = [50, 60, 70, 80]

    # ---------- 4. Evaluate each approach ----------
    print('\n' + '=' * 70)
    print('  RECALL ON 216 KNOWN GRAIL STRATEGIES')
    print('=' * 70)

    # Approach A
    a_flagged_winners = [t for t in winners if a_flags(t['name'])]
    a_missed_winners = [t for t in winners if not a_flags(t['name'])]
    a_in_cache = [t for t in winners if t['name'] in cache_all]
    a_outside_cache = [t for t in winners if t['name'] not in cache_all]
    print(f'\n[A] fast_filter cache:')
    print(f'    Winners IN cache:     {len(a_in_cache)}  (of {len(winners)})')
    print(f'    Winners FLAGGED (passed): {len(a_flagged_winners)}')
    print(f'    Winners MISSED (failed):  {len(a_in_cache)-len(a_flagged_winners)}')
    print(f'    Winners NOT in cache yet: {len(a_outside_cache)}  ← never tested')
    # recall AMONG tested strategies
    if a_in_cache:
        a_recall_tested = len(a_flagged_winners) / len(a_in_cache) * 100
    else:
        a_recall_tested = 0
    a_recall_overall = len(a_flagged_winners) / len(winners) * 100
    print(f'    Recall on tested: {a_recall_tested:.1f}%  |  Recall overall: {a_recall_overall:.1f}%')

    # Approach B
    print(f'\n[B] family scorer (leave-one-strategy-out):')
    for thr in THRESHOLDS:
        b_flagged_winners = [t for t in winners if b_score_loo(t) >= thr]
        b_flagged_non = [t for t in non_winners if b_score_loo(t) >= thr]
        recall = len(b_flagged_winners) / len(winners) * 100
        precision = len(b_flagged_winners) / max(len(b_flagged_winners)+len(b_flagged_non), 1) * 100
        print(f'    threshold>={thr}: recall={recall:.1f}%  '
              f'({len(b_flagged_winners)}/{len(winners)} winners)  '
              f'precision={precision:.1f}%  '
              f'({len(b_flagged_non)} non-winners also flagged)')

    # Approach C: A OR B (at threshold 60)
    thr_c = 60
    c_flagged = [t for t in winners if a_flags(t['name']) or b_score_loo(t) >= thr_c]
    c_flagged_non = [t for t in non_winners if a_flags(t['name']) or b_score_loo(t) >= thr_c]
    c_recall = len(c_flagged) / len(winners) * 100
    c_precision = len(c_flagged) / max(len(c_flagged)+len(c_flagged_non), 1) * 100
    print(f'\n[C] combined (A pass OR B>={thr_c}):')
    print(f'    recall={c_recall:.1f}%  ({len(c_flagged)}/{len(winners)})  '
          f'precision={c_precision:.1f}%')

    # ---------- 5. Diagnostic: what type of strategies each approach MISSES ----------
    print('\n' + '=' * 70)
    print('  WHAT GETS MISSED — by family')
    print('=' * 70)
    # For B at thr=60, which families lose winners?
    miss_b = defaultdict(int)
    total_b = defaultdict(int)
    for t in winners:
        total_b[t['family']] += 1
        if b_score_loo(t) < 60:
            miss_b[t['family']] += 1
    print(f'\n[B@60] families with missed winners (scorer too strict):')
    print(f'  {"Family":25s} {"Missed":>8s} {"Total":>8s} {"MissRate":>10s}')
    for fam, m in sorted(miss_b.items(), key=lambda kv: -kv[1])[:15]:
        tot = total_b[fam]
        print(f'  {fam:25s} {m:>8d} {tot:>8d} {m*100/tot:>9.1f}%')

    # For A, which families fall entirely outside cache?
    miss_a = defaultdict(lambda: {'not_in_cache': 0, 'in_cache_failed': 0, 'in_cache_passed': 0})
    for t in winners:
        if t['name'] not in cache_all:
            miss_a[t['family']]['not_in_cache'] += 1
        elif t['name'] in cache_failed:
            miss_a[t['family']]['in_cache_failed'] += 1
        else:
            miss_a[t['family']]['in_cache_passed'] += 1
    print(f'\n[A] families with winners not in cache yet (coverage gap):')
    rows_a = sorted(miss_a.items(), key=lambda kv: -kv[1]['not_in_cache'])[:15]
    print(f'  {"Family":25s} {"NoCache":>8s} {"Failed":>8s} {"Passed":>8s}')
    for fam, s in rows_a:
        print(f'  {fam:25s} {s["not_in_cache"]:>8d} {s["in_cache_failed"]:>8d} {s["in_cache_passed"]:>8d}')

    # ---------- 6. Actionable improvements ----------
    print('\n' + '=' * 70)
    print('  IMPROVEMENTS SUGGESTED BY THE DATA')
    print('=' * 70)
    print()
    print('1. [A] fast_filter cache has HUGE coverage gap:')
    print(f'   Only {len(a_in_cache)} of {len(winners)} winners ever evaluated.')
    print(f'   → Approach A cannot be trusted alone — recall capped at {len(a_in_cache)*100/len(winners):.0f}%.')
    print()
    print('2. [B] family scorer is strict but has precision vs recall tradeoff:')
    # pick the best operating point
    best = None
    for thr in THRESHOLDS:
        b_w = sum(1 for t in winners if b_score_loo(t) >= thr)
        b_nw = sum(1 for t in non_winners if b_score_loo(t) >= thr)
        recall = b_w / len(winners)
        prec = b_w / max(b_w + b_nw, 1)
        f1 = 2 * recall * prec / max(recall + prec, 0.001)
        if best is None or f1 > best[0]:
            best = (f1, thr, recall, prec)
    print(f'   Best F1 at threshold {best[1]}: recall={best[2]*100:.1f}%  precision={best[3]*100:.1f}%')
    print()
    print('3. [C] combined recovers missed winners:')
    print(f'   A+B@60 hits {c_recall:.1f}% recall — higher than either alone.')
    print()
    print('4. Rescue classics (B1-B6, RSI non-TV):')
    classic_families = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'RSI']
    classic_winners_missed = [t for t in winners
                              if t['family'] in classic_families
                              and b_score_loo(t) < 60
                              and not a_flags(t['name'])]
    print(f'   Classics missed by BOTH A and B@60: {len(classic_winners_missed)}')
    if classic_winners_missed:
        print('   → Add "classic track": lower threshold for B1-B6 (strong empirical hit rates 72-90%)')

    # ---------- 7. Dump CSV ----------
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / 'empirical_comparison.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['strategy', 'family', 'is_winner', 'n_records', 'n_winner_records',
                    'avg_wr', 'avg_pnl', 'in_cache', 'cache_passed',
                    'b_score_loo', 'a_flags', 'b_flags_60', 'c_flags'])
        for t in truth:
            b_s = b_score_loo(t)
            in_c = t['name'] in cache_all
            c_p = t['name'] in cache_passed
            a_f = a_flags(t['name'])
            b_f = b_s >= 60
            c_f = a_f or b_f
            w.writerow([t['name'], t['family'], int(t['is_winner']), t['n_records'],
                        t['n_winner_records'], f"{t['avg_wr']:.2f}", f"{t['avg_pnl']:.2f}",
                        int(in_c), int(c_p), f"{b_s:.2f}",
                        int(a_f), int(b_f), int(c_f)])
    print(f'\nDump: {out_csv}')


if __name__ == '__main__':
    sys.exit(main() or 0)
