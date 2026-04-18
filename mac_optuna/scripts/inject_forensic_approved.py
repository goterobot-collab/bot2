#!/usr/bin/env python3
"""
Inject forensic-approved grails into production JSON.
Reads forensic_summary.json, cross-references grails_master.json for params,
finds approved grails not yet in production, and adds them.

Usage:
    python3 scripts/inject_forensic_approved.py --dry-run   # Preview only
    python3 scripts/inject_forensic_approved.py              # Actually inject
"""
import json, os, sys, shutil
from datetime import datetime
from pathlib import Path

FORENSIC_SUMMARY = Path("/Users/sabrina/CLAUDE CODE/Estrategias/data/forensic_summary.json")
GRAILS_MASTER = Path("/Users/sabrina/CLAUDE CODE/Estrategias/data/grails_master.json")
PROD_JSON = Path("/Users/sabrina/CLAUDE CODE/BOT V7/live/data/v6_optimized_sl_tp.json")

# ── CAPA 1: ForensicGate (Regla 24) ─────────────────────────────────────────
_BOT_V7_SCRIPTS = Path("/Users/sabrina/CLAUDE CODE/BOT V7/scripts")
sys.path.insert(0, str(_BOT_V7_SCRIPTS))
try:
    from forensic_gate import ForensicGate, check_forensic_compliance
    _FORENSIC_GATE_AVAILABLE = True
except ImportError:
    _FORENSIC_GATE_AVAILABLE = False
    print("⚠️  [WARN] forensic_gate.py no disponible — CAPA 1 deshabilitada")


def norm_sym(s):
    """Normalize symbol: 'XXX/USDT:USDT' → 'XXX', 'XXX' → 'XXX'"""
    return s.split('/')[0] if '/' in s else s


def full_sym(s):
    """Ensure full symbol format: 'XXX' → 'XXX/USDT:USDT'"""
    return s if '/' in s else f"{s}/USDT:USDT"


def main():
    dry_run = '--dry-run' in sys.argv

    # Load forensic summary (3MB, fast)
    print(f"Loading forensic summary: {FORENSIC_SUMMARY}")
    with open(FORENSIC_SUMMARY) as f:
        forensic = json.load(f)
    results = forensic.get('results', [])

    # Load grails_master for params (30MB)
    print(f"Loading grails master: {GRAILS_MASTER}")
    with open(GRAILS_MASTER) as f:
        grails = json.load(f)

    # Build grails lookup by normalized key
    gm_map = {}
    for g in grails:
        key = f"{g['strategy']}|{norm_sym(g['symbol'])}|{g['timeframe']}"
        gm_map[key] = g

    # Load production JSON
    print(f"Loading production JSON: {PROD_JSON}")
    with open(PROD_JSON) as f:
        prod_bots = json.load(f)

    # Build production key set (normalize symbols for matching)
    prod_keys = set()
    for b in prod_bots:
        key = f"{b.get('strategy','')}|{norm_sym(b.get('symbol',''))}|{b.get('timeframe','')}"
        prod_keys.add(key)

    # Find approved grails not in production, with params from grails_master
    new_bots = []
    skipped_no_params = 0
    skipped_in_prod = 0

    for r in results:
        if not r.get('gate_approved'):
            continue

        norm_key = f"{r['strategy']}|{norm_sym(r['symbol'])}|{r['timeframe']}"

        if norm_key in prod_keys:
            skipped_in_prod += 1
            continue

        # Get params from grails_master
        grail = gm_map.get(norm_key)
        if not grail:
            skipped_no_params += 1
            continue

        params = grail.get('best_params', {})
        is_tv = r['strategy'].startswith('TV_')
        symbol_full = full_sym(r['symbol'])

        # Match leverage/SL to existing production bots for same symbol
        # If MAE data missing (safe_leverage=20, mae_p95=0), use conservative defaults
        safe_lev = grail.get('safe_leverage', 1)
        mae = grail.get('test_mae_p95', 0)
        if mae <= 0 or safe_lev >= 20:
            # No MAE data — use production-consistent conservative defaults
            sl_pct = 0.40  # 40% SL (matches existing prod bots)
            leverage = 1    # 1x leverage (matches existing prod bots)
        else:
            sl_pct = min(0.40, max(0.02, mae * 1.5))
            leverage = min(safe_lev, 20)

        # TP: ensure R:R >= 0.50 (Gate 8)
        tp_pct = max(sl_pct * 0.50, sl_pct * 0.80)

        bot = {
            'strategy': r['strategy'],
            'symbol': symbol_full,
            'timeframe': r['timeframe'],
            'params': params,
            'sl_pct': round(sl_pct, 4),
            'tp_pct': round(tp_pct, 4),
            'sl_tp_method': 'optuna_v7_empirical',
            'leverage': leverage,
            'suspended': is_tv,
            'observation_mode': is_tv,
            'test_trades': r.get('n_trades', 0),
            'test_wr': r.get('real_wr', 0),
            'forensic': {
                'real_wr': r.get('real_wr', 0),
                'optuna_wr': r.get('optuna_wr', 0),
                'gap_pp': r.get('gap_pp', 0),
                'total_trades': r.get('n_trades', 0),
                'pnl_pct': r.get('pnl_pct', 0),
                'profit_factor': r.get('profit_factor', 0),
                'max_drawdown': r.get('max_dd_pct', 0),
                'sharpe': r.get('sharpe', 0),
                'tier': r.get('gate_tier', 'UNKNOWN'),
                'tested_date': datetime.now().strftime('%Y-%m-%d'),
                'round': 'FORENSIC_BATCH_V3',
            }
        }
        # ── CAPA 1: FORENSIC GATE — Verificar ANTES de aceptar el bot ──────
        if _FORENSIC_GATE_AVAILABLE:
            ok, reason = check_forensic_compliance(bot)
            if not ok:
                print(f"  ⛔ FORENSIC GATE BLOQUEÓ: {reason}")
                continue
        # ────────────────────────────────────────────────────────────────────

        new_bots.append(bot)
        prod_keys.add(norm_key)

    active_new = [b for b in new_bots if not b['suspended']]
    shadow_new = [b for b in new_bots if b['suspended']]

    print(f"\n=== INJECT FORENSIC APPROVED ===")
    print(f"Forensic results:    {len(results)}")
    print(f"Approved total:      {sum(1 for r in results if r.get('gate_approved'))}")
    print(f"Already in prod:     {skipped_in_prod}")
    print(f"No params found:     {skipped_no_params}")
    print(f"New to inject:       {len(new_bots)} ({len(active_new)} active + {len(shadow_new)} shadow)")
    print(f"Production current:  {len(prod_bots)}")

    if new_bots:
        print(f"\nNew bots to inject:")
        for b in sorted(new_bots, key=lambda x: x['forensic']['pnl_pct'], reverse=True):
            tag = "SHADOW" if b['suspended'] else "ACTIVE"
            print(f"  [{tag}] {b['strategy']:30s} x {b['symbol']:20s} {b['timeframe']:3s} | "
                  f"WR={b['forensic']['real_wr']:.1f}% PnL={b['forensic']['pnl_pct']:+.1f}% "
                  f"PF={b['forensic']['profit_factor']:.2f} n={b['forensic']['total_trades']} "
                  f"SL={b['sl_pct']:.2%} TP={b['tp_pct']:.2%} Lev={b['leverage']}x")

    if dry_run:
        print(f"\n[DRY RUN] No changes made. Remove --dry-run to inject.")
        return

    if not new_bots:
        print(f"\nNo new bots to inject.")
        return

    # Backup (Regla 14)
    backup = str(PROD_JSON) + f".bak_{datetime.now().strftime('%H%M')}"
    shutil.copy2(PROD_JSON, backup)
    print(f"\nBackup: {backup}")

    # Inject
    prod_bots.extend(new_bots)

    # Atomic write
    import tempfile
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=PROD_JSON.parent, suffix='.tmp', delete=False) as tmp:
            json.dump(prod_bots, tmp, indent=2, ensure_ascii=False)
            tmp_path = tmp.name
        os.replace(tmp_path, PROD_JSON)
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    print(f"Injected: {len(new_bots)} bots -> total {len(prod_bots)}")

    # Validate (Regla 14)
    import subprocess
    validate = subprocess.run(
        ['python3', '/Users/sabrina/CLAUDE CODE/BOT V7/scripts/validate_json.py'],
        capture_output=True, text=True
    )
    if validate.returncode == 0:
        print(f"validate_json.py: PASS")
    else:
        print(f"validate_json.py: FAIL")
        print(validate.stdout[-500:] if validate.stdout else "")
        print(validate.stderr[-500:] if validate.stderr else "")


if __name__ == '__main__':
    main()
