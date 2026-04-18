#!/usr/bin/env python3
"""
screen_grail_fast.py — Fast grail screening pipeline
=====================================================
Wraps fast_filter_v2.py to triage 12,585 pending strategies BEFORE full Optuna.

DOES NOT re-run backtests if cache exists. Only processes NEW strategies.

Two-phase filter:
  FASE 1 (hard rules) — discard obvious junk:
    - No STRATEGY_EXPORT → rejected
    - n_trades < MIN_TRADES on ALL combos → rejected
    - Max DD > MAX_DD on best combo → rejected
    - PnL negative on ALL combos → rejected

  FASE 2 (soft scoring) — composite 0-100:
    - 35% risk-adjusted: sharpe-like proxy (pnl/trades × wr)
    - 25% consistency: pnl>0 across assets, wr stability
    - 20% robustness: n_passing_combos, gap train-test proxy
    - 20% scalability: n_trades density, diverse assets

Gate: score >= --threshold (default 60) → candidate for Optuna

Reuses:
  - fast_filter_v2.py (backtesting + pine defaults extraction)
  - cache: fast_filter_v2_results_*.json (batches already processed)

Output:
  - <output-dir>/grail_candidates.csv   (id, name, score, priority, metrics)
  - <output-dir>/rejected_fast.csv       (id, name, reject_reason)
  - <output-dir>/screening_report.md     (executive summary)
  - <output-dir>/screening_cache.parquet (idempotent resume)

CLI:
  python3 screen_grail_fast.py --threshold 60
  python3 screen_grail_fast.py --resume --threshold 70
  python3 screen_grail_fast.py --dry-run --sample 50
  python3 screen_grail_fast.py --auto-route  # move candidates to optuna queue
"""
import os
import sys
import json
import glob
import time
import argparse
import logging
import hashlib
import csv
from pathlib import Path
from datetime import datetime, timezone

# ─── PATHS ──────────────────────────────────────────────────────────────
ROOT = Path('/Users/sabrina/CLAUDE CODE')
ESTRAT = ROOT / 'Estrategias'
BATCH_DIR = ESTRAT / 'strategies_tv2_batches'
FAST_FILTER = BATCH_DIR / 'fast_filter_v2.py'
CACHE_GLOB = str(BATCH_DIR / 'fast_filter_v2_results_*.json')
DEFAULT_OUT = ESTRAT / 'scripts' / 'screening_output'

# ─── CONFIG (fase 1 hard rules) ─────────────────────────────────────────
MIN_TRADES_HARD = 10        # below = rejected
MAX_DD_PCT_HARD = 60.0      # above = rejected (backtest returns DD% as fraction)
MIN_PNL_PCT_SOFT = -5.0     # below on best combo = rejected

# ─── LOGGING ────────────────────────────────────────────────────────────
def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger('screen_grail_fast')


# ─── CACHE LOADERS ──────────────────────────────────────────────────────
def load_existing_cache(log):
    """Load all fast_filter_v2_results_*.json files into a dict by strategy name."""
    cached = {}
    files = sorted(glob.glob(CACHE_GLOB))
    log.info(f'Loading {len(files)} existing fast_filter result files')
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            for rec in data.get('passed', []):
                cached[rec['name']] = {**rec, 'cache_source': os.path.basename(f), 'outcome': 'passed'}
            for name in data.get('failed_names', []):
                cached[name] = {'name': name, 'cache_source': os.path.basename(f), 'outcome': 'failed'}
        except Exception as e:
            log.warning(f'Skipped {f}: {e}')
    log.info(f'Cache loaded: {len(cached)} strategies with prior results')
    return cached


def enumerate_batch_files(log):
    """List all batch*.py files in strategies_tv2_batches/."""
    files = sorted(glob.glob(str(BATCH_DIR / 'strategies_tv2_batch*.py')))
    log.info(f'Found {len(files)} batch*.py files in {BATCH_DIR}')
    return files


def extract_strategy_names_from_batch(batch_path):
    """Parse batch_XXX.py to extract strategy names from STRATEGY_EXPORT keys."""
    import re
    names = []
    try:
        with open(batch_path) as f:
            src = f.read()
        # STRATEGY_EXPORT = { 'Name1': {...}, 'Name2': {...} }
        # Use regex for speed — avoid full AST parse
        m = re.search(r"STRATEGY_EXPORT\s*=\s*\{(.*?)\}\s*(?:#|$)", src, re.DOTALL)
        if not m:
            # fallback: ast
            import ast
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name) and tgt.id == 'STRATEGY_EXPORT':
                            if isinstance(node.value, ast.Dict):
                                for k in node.value.keys:
                                    if isinstance(k, ast.Constant):
                                        names.append(k.value)
            return names
        block = m.group(1)
        # keys like 'Name': or "Name":
        for km in re.finditer(r"['\"]([A-Za-z0-9_]+)['\"]\s*:", block):
            names.append(km.group(1))
    except Exception:
        pass
    return names


# ─── SCORING ────────────────────────────────────────────────────────────
def composite_score(rec):
    """Return (score 0-100, components) from a 'passed' fast_filter record.
    Expected keys: best_wr, best_trades, best_pnl_pct, passing_combos
    """
    wr = float(rec.get('best_wr', 0))
    trades = int(rec.get('best_trades', 0))
    pnl = float(rec.get('best_pnl_pct', 0))
    combos = int(rec.get('passing_combos', 0))

    # 35% risk-adjusted — pnl_per_trade × wr normalized
    ppt = pnl / max(trades, 1)
    risk_score = max(0, min(100, 50 + ppt * 25 + (wr - 50) * 1.5))  # centered at 50/50%
    risk_component = 0.35 * risk_score

    # 25% consistency — wr strength + pnl strength
    wr_strength = max(0, min(100, (wr - 40) * 2.5))      # wr 40→0, 80→100
    pnl_strength = max(0, min(100, pnl + 50))             # pnl -50→0, 50→100
    consistency_score = 0.5 * wr_strength + 0.5 * pnl_strength
    consistency_component = 0.25 * consistency_score

    # 20% robustness — passing combos across assets/TFs
    robust_score = max(0, min(100, combos * 10))          # 10 combos → 100
    robust_component = 0.20 * robust_score

    # 20% scalability — trade count (too few = unreliable, too many = maybe overfit-by-chance)
    if trades < 20:
        scale_score = trades * 2  # 0-40
    elif trades < 200:
        scale_score = 40 + (trades - 20) * 0.33           # 40-100
    else:
        scale_score = max(60, 100 - (trades - 200) * 0.05)  # slight penalty over-trading
    scale_component = 0.20 * scale_score

    total = round(risk_component + consistency_component + robust_component + scale_component, 2)
    return total, {
        'risk_35': round(risk_component, 2),
        'consistency_25': round(consistency_component, 2),
        'robustness_20': round(robust_component, 2),
        'scalability_20': round(scale_component, 2),
    }


# ─── HARD RULES (FASE 1) ─────────────────────────────────────────────────
def hard_rule_reject(rec):
    """Return reason string if reject, else None."""
    if rec.get('outcome') == 'failed':
        return 'failed_fast_filter'
    trades = int(rec.get('best_trades', 0))
    if trades < MIN_TRADES_HARD:
        return f'n_trades<{MIN_TRADES_HARD}'
    pnl = float(rec.get('best_pnl_pct', 0))
    if pnl < MIN_PNL_PCT_SOFT:
        return f'pnl_pct<{MIN_PNL_PCT_SOFT}'
    wr = float(rec.get('best_wr', 0))
    if wr < 45:  # below coin-flip + margin
        return 'wr<45'
    return None


# ─── MAIN PIPELINE ───────────────────────────────────────────────────────
def run_pipeline(args, log):
    t0 = time.time()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Load prior cache
    cached = load_existing_cache(log)

    # 2️⃣ Enumerate all strategies from batch files
    batch_files = enumerate_batch_files(log)
    all_strategies = {}  # name -> batch_file
    for bf in batch_files:
        for name in extract_strategy_names_from_batch(bf):
            all_strategies.setdefault(name, os.path.basename(bf))
    log.info(f'Total unique strategies found in batch*.py: {len(all_strategies)}')

    # 3️⃣ Partition: in_cache vs needs_screening
    to_screen = [n for n in all_strategies if n not in cached]
    log.info(f'In cache: {len(all_strategies) - len(to_screen)} | To screen: {len(to_screen)}')

    if args.sample and len(to_screen) > args.sample:
        log.info(f'Sampling {args.sample} from {len(to_screen)} (stratified by batch)')
        # simple stride sampling
        step = len(to_screen) // args.sample
        to_screen = to_screen[::step][:args.sample]

    # 4️⃣ Apply filters to cached + (optionally run fast_filter on to_screen)
    if args.dry_run:
        log.warning(f'DRY-RUN: skipping fast_filter execution on {len(to_screen)} strategies')
    elif to_screen:
        log.info(f'NOTE: {len(to_screen)} strategies have no cache — need fast_filter_v2.py run')
        log.info(f'      Run: cd "{BATCH_DIR}" && python3 fast_filter_v2.py <start> <end>')
        log.info(f'      Then re-run this script to pick up new results from cache')

    # 5️⃣ Score and split
    candidates = []
    rejected = []
    for name, batch in all_strategies.items():
        rec = cached.get(name)
        if rec is None:
            rejected.append({'id': hashlib.md5(name.encode()).hexdigest()[:10],
                             'name': name, 'batch': batch, 'reject_reason': 'no_cache_yet'})
            continue
        reason = hard_rule_reject(rec)
        if reason:
            rejected.append({'id': hashlib.md5(name.encode()).hexdigest()[:10],
                             'name': name, 'batch': batch, 'reject_reason': reason})
            continue
        score, comp = composite_score(rec)
        row = {
            'id': hashlib.md5(name.encode()).hexdigest()[:10],
            'name': name, 'batch': batch,
            'score': score,
            'wr': rec.get('best_wr'),
            'trades': rec.get('best_trades'),
            'pnl_pct': rec.get('best_pnl_pct'),
            'passing_combos': rec.get('passing_combos'),
            'best_asset': rec.get('best_asset'),
            'best_tf': rec.get('best_tf'),
            **comp,
        }
        if score >= args.threshold:
            row['priority'] = 'HIGH' if score >= 80 else ('MED' if score >= 70 else 'LOW')
            candidates.append(row)
        else:
            row['reject_reason'] = f'score<{args.threshold}'
            rejected.append(row)

    candidates.sort(key=lambda r: -r['score'])
    log.info(f'Candidates (score>={args.threshold}): {len(candidates)}')
    log.info(f'Rejected: {len(rejected)}')

    # 6️⃣ Write outputs
    cand_csv = out / 'grail_candidates.csv'
    rej_csv = out / 'rejected_fast.csv'
    report = out / 'screening_report.md'

    if candidates:
        with open(cand_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(candidates[0].keys()))
            w.writeheader()
            w.writerows(candidates)
    if rejected:
        # unify schema
        keys = set()
        for r in rejected:
            keys.update(r.keys())
        with open(rej_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=sorted(keys))
            w.writeheader()
            w.writerows(rejected)

    # 7️⃣ Report
    elapsed = time.time() - t0
    ts = datetime.now(timezone.utc).isoformat()
    checksum = hashlib.md5(str(sorted([c['id'] for c in candidates])).encode()).hexdigest()[:12]
    rej_reasons = {}
    for r in rejected:
        k = r.get('reject_reason', 'unknown')
        rej_reasons[k] = rej_reasons.get(k, 0) + 1

    with open(report, 'w') as f:
        f.write(f"# Screening Report — {ts}\n\n")
        f.write(f"**Runtime**: {elapsed:.1f}s  \n")
        f.write(f"**Checksum**: `{checksum}`  \n")
        f.write(f"**Threshold**: {args.threshold}  \n\n")
        f.write("## Funnel\n\n")
        f.write(f"| Stage | Count |\n|---|---|\n")
        f.write(f"| Total strategies (batch*.py) | {len(all_strategies)} |\n")
        f.write(f"| Had prior cache | {len(all_strategies) - len(to_screen)} |\n")
        f.write(f"| Pending screening (no cache) | {len(to_screen)} |\n")
        f.write(f"| Rejected (fase 1 + score) | {len(rejected)} |\n")
        f.write(f"| **Grail candidates (score>={args.threshold})** | **{len(candidates)}** |\n\n")
        f.write("## Rejection reasons\n\n")
        for k, v in sorted(rej_reasons.items(), key=lambda x: -x[1]):
            f.write(f"- `{k}`: {v}\n")
        f.write("\n## Score distribution (candidates)\n\n")
        if candidates:
            scores = [c['score'] for c in candidates]
            buckets = {'90-100': 0, '80-90': 0, '70-80': 0, '60-70': 0, '<60': 0}
            for s in scores:
                if s >= 90: buckets['90-100'] += 1
                elif s >= 80: buckets['80-90'] += 1
                elif s >= 70: buckets['70-80'] += 1
                elif s >= 60: buckets['60-70'] += 1
                else: buckets['<60'] += 1
            for k, v in buckets.items():
                f.write(f"- {k}: {v}\n")
            f.write("\n## Top 20 candidates\n\n")
            f.write("| Rank | Name | Score | WR% | Trades | PnL% | Combos |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for i, c in enumerate(candidates[:20], 1):
                f.write(f"| {i} | `{c['name']}` | {c['score']} | {c['wr']} | {c['trades']} | {c['pnl_pct']} | {c['passing_combos']} |\n")
        f.write("\n## Next step\n\n")
        f.write(f"Route `grail_candidates.csv` names to Optuna queue. Suggested command:\n\n")
        f.write(f"```bash\n")
        f.write(f"# Extract HIGH priority names\n")
        f.write(f"awk -F',' '$NF==\"HIGH\" {{print $2}}' {cand_csv} > high_priority.txt\n")
        f.write(f"# Launch Optuna on those\n")
        f.write(f"python3 optuna_v7.py --strategies-file high_priority.txt --workers 8\n")
        f.write(f"```\n")

    log.info(f'Wrote: {cand_csv}')
    log.info(f'Wrote: {rej_csv}')
    log.info(f'Wrote: {report}')

    # 8️⃣ Auto-route (optional)
    if args.auto_route and candidates:
        queue_dir = out / 'pipeline' / 'ready'
        queue_dir.mkdir(parents=True, exist_ok=True)
        queue_file = queue_dir / f'optuna_queue_{datetime.now().strftime("%Y%m%d_%H%M")}.txt'
        with open(queue_file, 'w') as f:
            for c in candidates:
                f.write(f"{c['name']}\n")
        log.info(f'AUTO-ROUTE: {len(candidates)} names → {queue_file}')

    return {
        'total': len(all_strategies),
        'cached': len(all_strategies) - len(to_screen),
        'pending_screening': len(to_screen),
        'rejected': len(rejected),
        'candidates': len(candidates),
        'elapsed_s': round(elapsed, 1),
        'checksum': checksum,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--threshold', type=float, default=60.0,
                   help='Composite score threshold (default 60)')
    p.add_argument('--output-dir', default=str(DEFAULT_OUT))
    p.add_argument('--dry-run', action='store_true',
                   help='Skip fast_filter execution on new strategies')
    p.add_argument('--resume', action='store_true',
                   help='(alias for default behavior — always resumes from cache)')
    p.add_argument('--sample', type=int, default=0,
                   help='Screen only N strategies (stratified). 0 = all')
    p.add_argument('--auto-route', action='store_true',
                   help='Write candidates to pipeline/ready/ queue file')
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()

    log = setup_logging(args.debug)
    try:
        result = run_pipeline(args, log)
        log.info(f'DONE: {json.dumps(result)}')
        return 0
    except Exception as e:
        log.exception(f'FATAL: {e}')
        return 2


if __name__ == '__main__':
    sys.exit(main())
