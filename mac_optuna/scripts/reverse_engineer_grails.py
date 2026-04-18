#!/usr/bin/env python3
"""
reverse_engineer_grails.py — Learn from winners, predict the rest
==================================================================
Take the 2,682 grails that actually WORKED (WR>=65% + PnL>0 + trades>=10)
and reverse-engineer a family-signature scorer to prioritize the 12,337
pending strategies WITHOUT running any backtest.

Empirical training set (from unified_bot_v7.db):
  - 1,876 TV_ grails + 806 classic grails = 2,682 winners
  - 216 distinct strategies, 2,977 total grail records

Reverse-engineered features:
  - family_prefix    (e.g. TV_RSI_, TV_VWAP_, B4, B6)
  - n_grails_same_family  (how many grails exist for this family)
  - family_hit_rate       (winners / tested_in_grails)
  - family_avg_wr
  - family_avg_pnl
  - family_avg_rr
  - family_best_tfs       (which TFs work historically)
  - family_best_symbols   (which symbols pay)

Score = weighted composite of family signature.
Rank all 12,337 pending strategies. Top K goes to Optuna.

NO BACKTESTING. NO OPTUNA. Pure pattern matching — runs in <2 seconds.

Output:
  - pending_ranked.csv        (all 12,337, ranked by predicted_score)
  - pending_top_K.csv         (top K to send to Optuna first)
  - priority_symbol_tf.csv    (per-candidate recommended sym×TF test matrix)
  - reverse_report.md         (executive summary + family leaderboard)
"""
import os
import sys
import re
import csv
import json
import glob
import sqlite3
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

ROOT = Path('/Users/sabrina/CLAUDE CODE')
DB = ROOT / 'BOT V7' / 'live' / 'data' / 'unified_bot_v7.db'
BATCH_DIR = ROOT / 'Estrategias' / 'strategies_tv2_batches'
OUT_DIR = ROOT / 'Estrategias' / 'scripts' / 'screening_output'

# ─── EMPIRICAL GATES (reverse-engineered from grails table) ─────────────
WINNER_FILTER = "full_wr>=65 AND full_pnl>0 AND full_trades>=10"


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%H:%M:%S')
    return logging.getLogger('reverse_engineer')


def family_prefix(name):
    """Extract family: TV_* uses 2 tokens, else 1 token.
    Examples:
      TV_RSI_SlowFast_MA_Cross  -> TV_RSI_
      TV_VWAP_BB_Squeeze        -> TV_VWAP_
      B4_Regime_BB_Width        -> B4
      RSI_Dynamic               -> RSI
    """
    if name.startswith('TV_'):
        parts = name.split('_', 2)
        if len(parts) >= 2:
            return f'TV_{parts[1]}_'
        return 'TV_'
    # Non-TV: first token (B4, RSI, MeanRev, ULTRA, etc.)
    return name.split('_', 1)[0]


def learn_family_signatures(log):
    """Compute per-family stats from the grails DB."""
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    # All grails records (both winners and losers)
    rows = conn.execute("""
        SELECT strategy, symbol, timeframe, full_wr, full_pnl, rr_ratio, full_trades,
               (full_wr>=65 AND full_pnl>0 AND full_trades>=10) as is_winner
        FROM grails
    """).fetchall()
    log.info(f'Loaded {len(rows)} grail records from DB')

    fam = defaultdict(lambda: {
        'n': 0, 'winners': 0, 'sum_wr': 0, 'sum_pnl': 0, 'sum_rr': 0,
        'sum_trades': 0, 'tfs': defaultdict(int), 'syms': defaultdict(int),
        'win_tfs': defaultdict(int), 'win_syms': defaultdict(int),
    })

    for r in rows:
        name = r['strategy']
        f = family_prefix(name)
        s = fam[f]
        s['n'] += 1
        s['sum_wr'] += r['full_wr'] or 0
        s['sum_pnl'] += r['full_pnl'] or 0
        s['sum_rr'] += r['rr_ratio'] or 0
        s['sum_trades'] += r['full_trades'] or 0
        s['tfs'][r['timeframe']] += 1
        s['syms'][r['symbol']] += 1
        if r['is_winner']:
            s['winners'] += 1
            s['win_tfs'][r['timeframe']] += 1
            s['win_syms'][r['symbol']] += 1

    # Flatten to dict of summaries
    sig = {}
    for f, s in fam.items():
        n = s['n']
        sig[f] = {
            'family': f,
            'n_tested': n,
            'n_winners': s['winners'],
            'hit_rate': round(s['winners'] / n * 100, 2) if n else 0,
            'avg_wr': round(s['sum_wr'] / n, 2) if n else 0,
            'avg_pnl': round(s['sum_pnl'] / n, 2) if n else 0,
            'avg_rr': round(s['sum_rr'] / n, 3) if n else 0,
            'avg_trades': round(s['sum_trades'] / n, 1) if n else 0,
            'top_tfs': sorted(s['win_tfs'].items(), key=lambda x: -x[1])[:3],
            'top_syms': sorted(s['win_syms'].items(), key=lambda x: -x[1])[:10],
        }

    conn.close()
    log.info(f'Learned signatures for {len(sig)} families')
    return sig


def predict_score(name, sig, global_stats):
    """Predict score 0-100 based on family signature.
    Heuristic:
      40% family hit rate (historical P(winner|family))
      25% avg_pnl (magnitude of edge)
      20% avg_wr    (reliability)
      10% TV bonus  (empirically 2× more grails)
      5%  sample_size confidence (sqrt of n_tested, capped)
    """
    f = family_prefix(name)
    s = sig.get(f)
    if s is None:
        # Unknown family — cold start, give global base rate
        return {
            'family': f,
            'score': 10.0,  # low default
            'hit_rate': 0,
            'avg_wr': 0, 'avg_pnl': 0, 'avg_rr': 0,
            'n_tested': 0, 'n_winners': 0,
            'reason': 'unknown_family',
            'recommended_tfs': ['1h', '4h'],
            'recommended_syms': global_stats['top_syms'][:5],
        }

    # Components
    hit_comp = 0.40 * s['hit_rate']                                    # 0-40
    pnl_comp = 0.25 * min(100, max(0, s['avg_pnl'] / 3))               # pnl 300% -> 100
    wr_comp = 0.20 * max(0, (s['avg_wr'] - 50) * 2)                    # wr 50->0, 100->100
    tv_bonus = 10 if name.startswith('TV_') else 0
    import math
    size_conf = 5 * min(1.0, math.sqrt(s['n_tested']) / 10)             # n=100 -> full 5 pts

    score = round(hit_comp + pnl_comp + wr_comp + tv_bonus + size_conf, 2)

    tfs = [t for t, _ in s['top_tfs']] or ['1h', '4h']
    syms = [sym for sym, _ in s['top_syms']] or global_stats['top_syms'][:5]

    return {
        'family': f,
        'score': score,
        'hit_rate': s['hit_rate'],
        'avg_wr': s['avg_wr'],
        'avg_pnl': s['avg_pnl'],
        'avg_rr': s['avg_rr'],
        'n_tested': s['n_tested'],
        'n_winners': s['n_winners'],
        'reason': 'family_match',
        'recommended_tfs': tfs,
        'recommended_syms': syms[:5],
    }


def extract_pending_names(log, cached_names):
    """Parse all strategies_tv2_batch*.py to find strategies NOT in cache/production."""
    files = sorted(glob.glob(str(BATCH_DIR / 'strategies_tv2_batch*.py')))
    log.info(f'Scanning {len(files)} batch files')
    all_names = {}
    for bf in files:
        try:
            with open(bf) as fh:
                src = fh.read()
            m = re.search(r"STRATEGY_EXPORT\s*=\s*\{(.*?)\n\}\s*(?:#|$)", src, re.DOTALL)
            if not m:
                continue
            for km in re.finditer(r"['\"]([A-Za-z0-9_]+)['\"]\s*:", m.group(1)):
                all_names.setdefault(km.group(1), os.path.basename(bf))
        except Exception as e:
            log.debug(f'{bf}: {e}')
    pending = {n: b for n, b in all_names.items() if n not in cached_names}
    log.info(f'Total batch strategies: {len(all_names)} | already-grailed: {len(all_names) - len(pending)} | pending: {len(pending)}')
    return pending


def global_top(sig):
    """Aggregate global top TFs and symbols from winners across all families."""
    tfs = defaultdict(int)
    syms = defaultdict(int)
    for s in sig.values():
        for t, n in s['top_tfs']:
            tfs[t] += n
        for sym, n in s['top_syms']:
            syms[sym] += n
    return {
        'top_tfs': [t for t, _ in sorted(tfs.items(), key=lambda x: -x[1])[:4]],
        'top_syms': [sym for sym, _ in sorted(syms.items(), key=lambda x: -x[1])[:15]],
    }


def load_already_grailed(log):
    """Names that already appear in grails table — don't re-test."""
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT DISTINCT strategy FROM grails").fetchall()
    conn.close()
    names = set(r[0] for r in rows)
    log.info(f'Strategies already in grails table: {len(names)}')
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top-k', type=int, default=500, help='Top K to write as priority queue')
    ap.add_argument('--min-score', type=float, default=50.0)
    ap.add_argument('--output-dir', default=str(OUT_DIR))
    args = ap.parse_args()

    log = setup_logging()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = datetime.now()

    # 1. Learn from historical grails
    sig = learn_family_signatures(log)
    global_stats = global_top(sig)
    log.info(f"Global top TFs: {global_stats['top_tfs']}")
    log.info(f"Global top 5 syms: {global_stats['top_syms'][:5]}")

    # 2. Get pending
    already = load_already_grailed(log)
    pending = extract_pending_names(log, already)

    # 3. Score each pending
    rows = []
    for name, batch in pending.items():
        p = predict_score(name, sig, global_stats)
        rows.append({
            'name': name,
            'batch': batch,
            **p,
            'recommended_tfs_csv': '|'.join(p['recommended_tfs']),
            'recommended_syms_csv': '|'.join(p['recommended_syms']),
        })

    rows.sort(key=lambda r: -r['score'])

    # 4. Write outputs
    all_csv = out / 'pending_ranked.csv'
    top_csv = out / 'pending_top_K.csv'
    prio_csv = out / 'priority_symbol_tf.csv'
    report = out / 'reverse_report.md'

    fieldnames = ['name', 'batch', 'family', 'score', 'hit_rate', 'avg_wr',
                  'avg_pnl', 'avg_rr', 'n_tested', 'n_winners', 'reason',
                  'recommended_tfs_csv', 'recommended_syms_csv']

    with open(all_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    top = [r for r in rows if r['score'] >= args.min_score][:args.top_k]
    with open(top_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(top)

    # 5. Priority sym×TF matrix (per candidate)
    with open(prio_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'symbol', 'timeframe', 'predicted_score'])
        for r in top:
            for sym in r['recommended_syms'][:5]:
                for tf in r['recommended_tfs'][:2]:
                    w.writerow([r['name'], sym, tf, r['score']])

    # 6. Report
    elapsed = (datetime.now() - t0).total_seconds()
    fam_lb = sorted(sig.items(), key=lambda kv: -kv[1]['n_winners'])[:20]

    score_buckets = defaultdict(int)
    for r in rows:
        s = r['score']
        if s >= 80: score_buckets['80-100'] += 1
        elif s >= 70: score_buckets['70-80'] += 1
        elif s >= 60: score_buckets['60-70'] += 1
        elif s >= 50: score_buckets['50-60'] += 1
        elif s >= 30: score_buckets['30-50'] += 1
        else: score_buckets['<30'] += 1

    with open(report, 'w') as f:
        f.write(f'# Reverse-Engineered Grail Prediction Report\n\n')
        f.write(f'**Generated**: {datetime.now(timezone.utc).isoformat()}  \n')
        f.write(f'**Runtime**: {elapsed:.2f}s  \n\n')
        f.write(f'## Training set\n\n')
        winners = sum(s['n_winners'] for s in sig.values())
        tested = sum(s['n_tested'] for s in sig.values())
        f.write(f'- Grail records in DB: {tested}\n')
        f.write(f'- Winners (WR>=65 + PnL>0 + trades>=10): {winners}\n')
        f.write(f'- Families learned: {len(sig)}\n\n')
        f.write(f'## Pending strategies analyzed\n\n')
        f.write(f'- Total pending: {len(pending)}\n')
        f.write(f'- Already in grails table (skipped): {len(already)}\n\n')
        f.write(f'## Score distribution\n\n')
        for k in ['80-100', '70-80', '60-70', '50-60', '30-50', '<30']:
            f.write(f'- {k}: {score_buckets[k]}\n')
        f.write(f'\n## Top K selected (score>={args.min_score}): {len(top)}\n\n')
        f.write(f'## Family leaderboard (by winners)\n\n')
        f.write(f'| Family | Tested | Winners | Hit% | AvgWR | AvgPnL | AvgRR |\n')
        f.write(f'|---|---|---|---|---|---|---|\n')
        for fam_name, s in fam_lb:
            f.write(f"| `{fam_name}` | {s['n_tested']} | {s['n_winners']} | {s['hit_rate']}% | {s['avg_wr']} | {s['avg_pnl']} | {s['avg_rr']} |\n")
        f.write(f'\n## Top 20 pending candidates\n\n')
        f.write(f'| Rank | Name | Family | Score | Family hit% | Family AvgWR |\n')
        f.write(f'|---|---|---|---|---|---|\n')
        for i, r in enumerate(top[:20], 1):
            f.write(f"| {i} | `{r['name']}` | `{r['family']}` | {r['score']} | {r['hit_rate']}% | {r['avg_wr']} |\n")
        f.write(f'\n## Next step\n\n')
        f.write(f'```bash\n')
        f.write(f'# Send top K with recommended sym×TF to Optuna\n')
        f.write(f'python3 optuna_v7.py --combos-file {prio_csv} --workers 8\n')
        f.write(f'```\n')
        f.write(f'\nExpected yield (extrapolating base rates):\n')
        avg_hit = sum(r['hit_rate'] for r in top) / max(len(top), 1)
        f.write(f'- Top {len(top)} × avg family hit rate {avg_hit:.1f}% = **~{int(len(top)*avg_hit/100)} new grails expected**\n')
        f.write(f'- Optuna cost: {len(top)} × ~5 sym × 2 TF × 0.75 min = **~{len(top)*5*2*0.75/60:.0f} hours Hetzner 8 workers**\n')

    log.info(f'Wrote: {all_csv}')
    log.info(f'Wrote: {top_csv} ({len(top)} rows)')
    log.info(f'Wrote: {prio_csv}')
    log.info(f'Wrote: {report}')
    log.info(f'DONE in {elapsed:.2f}s')


if __name__ == '__main__':
    sys.exit(main() or 0)
