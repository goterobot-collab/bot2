#!/usr/bin/env python3
"""
Inject V4 validated grails into v6_optimized_sl_tp.json
========================================================
FBI 3/3 APPROVED (2026-04-10) with conditions:
  - 55 existing bots: update params directly
  - 17 new bots: observation_mode=true (shadow until 20 trades)
  - 4 DD>200% outliers: force SILVER sizing ($5)
  - NAORIS: max 4 active, rest in shadow
  - Backup mandatory before edit
  - validate_json.py pre and post

Usage:
  python3 scripts/inject_v4_grails.py --dry-run        # Preview changes
  python3 scripts/inject_v4_grails.py --apply           # Apply changes
"""
import json
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
V4_VALIDATED = ROOT / "data" / "grails_v4_revalidated_validated.json"
PROD_JSON = Path("/Users/sabrina/CLAUDE CODE/BOT V7/live/data/v6_optimized_sl_tp.json")
VALIDATE_SCRIPT = Path("/Users/sabrina/CLAUDE CODE/BOT V7/scripts/validate_json.py")

# FBI conditions
MAX_NAORIS_ACTIVE = 4
DD_OUTLIER_THRESHOLD = 200  # %
SILVER_SIZE = 5.0

# Grails with DD > 200% (identified in audit)
DD_OUTLIERS = {
    "BB_Double|Q/USDT:USDT|1h",
    "B6_Rubber_Band_EMA50|AGT/USDT:USDT|1h",
    "ZScore_VWAP_RSI|Q/USDT:USDT|1h",
    "Combo_VWAP_Stoch|NAORIS/USDT:USDT|1h",
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Preview only')
    parser.add_argument('--apply', action='store_true', help='Apply changes')
    args = parser.parse_args()

    if not args.dry_run and not args.apply:
        print("Usage: --dry-run or --apply")
        sys.exit(1)

    # Load V4 validated grails
    with open(V4_VALIDATED) as f:
        v4_grails = json.load(f)
    print(f"V4 validated grails: {len(v4_grails)}")

    # Load production JSON
    with open(PROD_JSON) as f:
        prod_bots = json.load(f)
    print(f"Production bots: {len(prod_bots)}")

    # Build index of production bots
    prod_index = {}
    for i, bot in enumerate(prod_bots):
        key = f"{bot.get('strategy','')}|{bot.get('symbol','')}|{bot.get('timeframe','')}"
        prod_index[key] = i

    # Classify V4 grails
    updates = []  # existing bots to update
    new_bots = []  # new bots to add
    naoris_count = 0

    for g in v4_grails:
        key = f"{g['strategy']}|{g['symbol']}|{g['timeframe']}"
        is_dd_outlier = key in DD_OUTLIERS or g.get('full_max_dd', 0) > DD_OUTLIER_THRESHOLD
        is_naoris = 'NAORIS' in g.get('symbol', '')

        if key in prod_index:
            updates.append((key, g, is_dd_outlier, is_naoris))
        else:
            new_bots.append((key, g, is_dd_outlier, is_naoris))

        if is_naoris:
            naoris_count += 1

    print(f"\nUpdates (existing): {len(updates)}")
    print(f"New bots: {len(new_bots)}")
    print(f"DD outliers: {sum(1 for _,g,dd,_ in updates+new_bots if dd)}")
    print(f"NAORIS total: {naoris_count} (max {MAX_NAORIS_ACTIVE} active)")

    if args.dry_run:
        print("\n=== DRY RUN — Changes preview ===\n")

        print("UPDATES (params change):")
        for key, g, dd, nao in updates:
            sym = g['symbol'].replace('/USDT:USDT','')
            flags = []
            if dd: flags.append("DD_OUTLIER→SILVER")
            if nao: flags.append("NAORIS")
            print(f"  {g['strategy']:<25} × {sym:<12} {g['timeframe']} WR={g.get('full_wr',0):.1f}% {' '.join(flags)}")

        print(f"\nNEW BOTS (shadow mode):")
        for key, g, dd, nao in new_bots:
            sym = g['symbol'].replace('/USDT:USDT','')
            flags = ["SHADOW"]
            if dd: flags.append("DD_OUTLIER→SILVER")
            if nao: flags.append("NAORIS")
            print(f"  {g['strategy']:<25} × {sym:<12} {g['timeframe']} WR={g.get('full_wr',0):.1f}% {' '.join(flags)}")

        print(f"\n=== DRY RUN COMPLETE — use --apply to execute ===")
        return

    # === APPLY ===
    print("\n=== APPLYING CHANGES ===")

    # 1. Backup
    ts = datetime.now().strftime('%H%M')
    backup = str(PROD_JSON) + f'.bak_{ts}'
    shutil.copy2(PROD_JSON, backup)
    print(f"Backup: {backup}")

    # 2. Update existing bots
    naoris_active = 0
    for key, g, is_dd_outlier, is_naoris in updates:
        idx = prod_index[key]
        bot = prod_bots[idx]

        # Update params from V4 optimization
        if g.get('best_params'):
            bot['params'] = g['best_params']

        # Update SL/TP from V3 re-validation
        if g.get('sl'):
            bot['sl_pct'] = g['sl']
        if g.get('leverage'):
            bot['leverage'] = g['leverage']

        # DD outlier → force SILVER
        if is_dd_outlier:
            bot['size_usd'] = SILVER_SIZE
            bot['tier'] = 'SILVER'

        # NAORIS cap
        if is_naoris:
            naoris_active += 1
            if naoris_active > MAX_NAORIS_ACTIVE:
                bot['suspended'] = True
                bot['suspend_reason'] = 'fbi_naoris_cap_4'

        # Mark as V4 updated
        if 'prod_stats' not in bot:
            bot['prod_stats'] = {}
        bot['prod_stats']['v4_updated'] = datetime.now().isoformat()
        bot['prod_stats']['v4_wr'] = g.get('full_wr', 0)

    print(f"  Updated {len(updates)} existing bots")

    # 3. Add new bots (shadow mode)
    for key, g, is_dd_outlier, is_naoris in new_bots:
        new_bot = {
            'strategy': g['strategy'],
            'symbol': g['symbol'],
            'timeframe': g['timeframe'],
            'params': g.get('best_params', {}),
            'sl_pct': g.get('sl', 0.40),
            'tp_pct': 0,  # signal-exit, no TP
            'leverage': g.get('leverage', 1),
            'tier': 'SILVER' if is_dd_outlier else 'ELITE',
            'size_usd': SILVER_SIZE if is_dd_outlier else 7.0,
            'max_dur_h': 720,
            'test_wr': g.get('full_wr', 0),
            'test_pnl': g.get('full_pnl', 0),
            'test_trades': g.get('full_trades', 0),
            'composite_score': 1.0,
            'round': 'V4_fullhistory',
            'suspended': False,
            'sl_tp_method': 'signal_exit_v3',
            'prod_stats': {
                'observation_mode': True,
                'obs_reason': 'pipeline_v4_new_bot_fbi_approved',
                'v4_updated': datetime.now().isoformat(),
                'v4_wr': g.get('full_wr', 0),
            },
        }

        # NAORIS cap
        if is_naoris:
            naoris_active += 1
            if naoris_active > MAX_NAORIS_ACTIVE:
                new_bot['suspended'] = True
                new_bot['suspend_reason'] = 'fbi_naoris_cap_4'

        prod_bots.append(new_bot)

    print(f"  Added {len(new_bots)} new bots (shadow mode)")
    print(f"  NAORIS: {naoris_active} total, {min(naoris_active, MAX_NAORIS_ACTIVE)} active")

    # 4. Save
    with open(PROD_JSON, 'w') as f:
        json.dump(prod_bots, f, indent=2)
    print(f"  Saved: {PROD_JSON}")
    print(f"  Total bots: {len(prod_bots)}")

    # 5. Validate
    if VALIDATE_SCRIPT.exists():
        print("\n  Running validate_json.py...")
        os.system(f'python3 "{VALIDATE_SCRIPT}"')
    else:
        print(f"  WARNING: validate_json.py not found at {VALIDATE_SCRIPT}")

    print(f"\n=== INJECTION COMPLETE ===")
    print(f"  Updated: {len(updates)}")
    print(f"  New (shadow): {len(new_bots)}")
    print(f"  DD outliers → SILVER: {sum(1 for _,_,dd,_ in updates+new_bots if dd)}")
    print(f"  NAORIS capped at {MAX_NAORIS_ACTIVE}")
    print(f"  Backup: {backup}")


if __name__ == '__main__':
    main()
