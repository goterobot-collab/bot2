#!/usr/bin/env python3
"""
inject_v8_real_to_production.py — Inject V8 REAL candidates to production.

Lee v8_real_candidates_*.json (output de hetzner_pipeline_v8.py) e inyecta
los candidatos que pasaron los 7 gates V8 al JSON de producción.

Convergencia 3/3 A/A/A. Aprobado Sabrina 2026-04-15.

CARACTERÍSTICAS V8 REAL vs V7.5 survivors:
  - source='v8_real' (vs 'hetzner_23K_V8_gates')
  - forensic incluye v8_gates_score, gates_breakdown, v8_profile
  - status='OBS' explícito + obs_deadline (72h UTC)
  - Sizing: SIZING_MAP Opción A + REGLA MAX HARD
  - Tier: DIAMOND si PF≥3 & Sharpe≥8, else GOLD
  - round='V8_REAL_T1'

RED DE SEGURIDAD:
  - TODOS los V8 REAL entran en observation_mode=True
  - Graduación a live solo si shadow WR≥65% y PnL+ en 72h
  - MAX HARD: nunca bajar sizing de bot existente

DEFAULT: DRY-RUN. Requiere --apply para escribir.

Usage:
  # DRY-RUN (default):
  python3 scripts/inject_v8_real_to_production.py \
      --input data/v8_real_candidates_20260415_1800.json

  # APPLY:
  python3 scripts/inject_v8_real_to_production.py \
      --input data/v8_real_candidates_20260415_1800.json \
      --apply

Ruta V8 T+2 (después de esto):
  1. Bot V7 restart para cargar los V8_REAL
  2. 72h observation → graduación
  3. Si ≥80% graduan → broadcast V8_REAL_LIVE
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path("/Users/sabrina/CLAUDE CODE")
PROD_JSON = ROOT / "BOT V7/live/data/v6_optimized_sl_tp.json"
REPORT_DIR = ROOT / "Estrategias/data/empirical_analysis_20260415"

# SIZING_MAP canonical (Opción A — idem upgrade_sizing_v8.py)
SIZING_MAP = {
    "Full size ($15-20)": 17.0,
    "Standard ($10-12)": 11.0,
    "Reduced ($7-10)": 8.0,
    "Minimum ($5-7)": 6.0,
}

OBS_DEADLINE_HOURS = 72
DEFAULT_MAX_INJECTIONS = 500  # safety limit per run


def tier_of(pf: float, sharpe: float) -> str:
    """DIAMOND si PF>=3 & Sharpe>=8, else GOLD."""
    try:
        if (pf or 0) >= 3.0 and (sharpe or 0) >= 8.0:
            return "DIAMOND"
    except (TypeError, ValueError):
        pass
    return "GOLD"


def sizing_from_risk(profit_factor: float, sharpe: float, dd_pct: float) -> tuple[str, float]:
    """
    Asigna sizing label + size_usd a partir de métricas forensic.

    Heurística canónica (Opción A):
      - Full size   ($17): PF≥2.5 & Sharpe≥5 & DD≤15
      - Standard    ($11): PF≥2.0 & Sharpe≥3 & DD≤20
      - Reduced      ($8): PF≥1.5 & Sharpe≥2 & DD≤30
      - Minimum      ($6): resto que pasa gates
    """
    pf = profit_factor or 0
    s = sharpe or 0
    dd = abs(dd_pct or 0)

    if pf >= 2.5 and s >= 5.0 and dd <= 15.0:
        return ("Full size ($15-20)", SIZING_MAP["Full size ($15-20)"])
    if pf >= 2.0 and s >= 3.0 and dd <= 20.0:
        return ("Standard ($10-12)", SIZING_MAP["Standard ($10-12)"])
    if pf >= 1.5 and s >= 2.0 and dd <= 30.0:
        return ("Reduced ($7-10)", SIZING_MAP["Reduced ($7-10)"])
    return ("Minimum ($5-7)", SIZING_MAP["Minimum ($5-7)"])


def max_dur_of(tf: str) -> int:
    return {"4h": 720, "1d": 720, "1h": 360, "5m": 72, "15m": 144}.get(tf, 720)


VALID_V8_PROFILES = {'default', 'relaxed_v75plus'}
CANONICAL_FEE_RATE = 0.0030  # Regla 24 — round-trip


def _extract_v8_profile(candidate: dict, v8r: dict) -> str:
    """Profile puede venir en top-level 'v8_profile' o en v8_result.config_profile.
    Fallback defensivo a 'default' (aceptado por gold_rule_validator)."""
    p = candidate.get('v8_profile') or v8r.get('config_profile')
    return p if p in VALID_V8_PROFILES else 'default'


def build_forensic_meta(
    candidate: dict,
    now_iso: str,
    *,
    forensic_suggested_lower: bool = False,
    size_max_hard_applied: bool = False,
    forensic_suggested_size: float | None = None,
) -> dict:
    """
    Construye dict forensic V8 REAL para trazabilidad total.

    Args:
        candidate: V8 candidate (con forensic_metrics + v8_result)
        now_iso: ISO timestamp de validación
        forensic_suggested_lower: True si forensic recomendó sizing menor al actual
        size_max_hard_applied: True si la regla MAX HARD rescató el sizing mayor
        forensic_suggested_size: valor USD sugerido (audit trail)
    """
    m = candidate.get('forensic_metrics', {})
    v8r = candidate.get('v8_result', {}) or {}
    v8_gates = v8r.get('gates', {}) or {}

    sizing_label, _ = sizing_from_risk(
        m.get('profit_factor', 0),
        m.get('sharpe', 0),
        m.get('max_drawdown_pct', m.get('max_dd_pct', 0)),
    )

    # Regla 24 canónica: si el candidato no persistió fee o lo hizo mal, impone 0.30%
    fee = m.get('fee_rate')
    if fee is None or abs(float(fee) - CANONICAL_FEE_RATE) > 1e-6:
        fee = CANONICAL_FEE_RATE

    return {
        'validated_at': now_iso,
        'source': 'v8_real',
        'v8_profile': _extract_v8_profile(candidate, v8r),
        'real_wr': m.get('win_rate'),
        'real_pnl_pct': m.get('total_pnl_pct'),
        'profit_factor': m.get('profit_factor'),
        'sharpe': m.get('sharpe'),
        'total_trades': m.get('total_trades'),
        'max_dd': m.get('max_drawdown_pct', m.get('max_dd_pct')),
        'long_wr': m.get('long_wr'),
        'short_wr': m.get('short_wr'),
        'avg_mae_pct': m.get('avg_mae_pct'),
        'avg_mfe_pct': m.get('avg_mfe_pct'),
        'fee_rate': fee,  # Regla 24 canónica (0.0030 round-trip)
        'sizing': sizing_label,
        # MAX HARD audit flags (Regla MAX HARD + gold_rule_validator runtime guard)
        'forensic_suggested_lower': bool(forensic_suggested_lower),
        'size_max_hard_applied': bool(size_max_hard_applied),
        'forensic_suggested_size_usd': forensic_suggested_size,
        # Gates breakdown
        'v8_gates_score': v8r.get('score'),
        'v8_gates_pass': v8r.get('pass'),
        'v8_gates_failed': v8r.get('failed_gates') or [],
        'v8_gate_wilson_lcb': (v8_gates.get('wilson_lcb') or {}).get('value'),
        'v8_gate_dsr': (v8_gates.get('dsr') or {}).get('value'),
        'v8_gate_hurst': (v8_gates.get('hurst') or {}).get('value'),
        'v8_gate_regime_coverage': (v8_gates.get('regime_coverage') or {}).get('value'),
        'v8_gate_direction': {
            'wr_long': (v8_gates.get('wr_per_direction') or v8_gates.get('direction_wr') or {}).get('long_wr'),
            'wr_short': (v8_gates.get('wr_per_direction') or v8_gates.get('direction_wr') or {}).get('short_wr'),
            'direction_mix': (v8_gates.get('wr_per_direction') or v8_gates.get('direction_wr') or {}).get('direction_mix'),
        },
        'gate_tier': 'V8_REAL_T1',
        'gate_tags': ['v8_real', 'v8_gates_passed'],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Inject V8 REAL candidates to production JSON',
    )
    parser.add_argument('--input', type=str, required=True,
                        help='v8_real_candidates_*.json from hetzner_pipeline_v8.py')
    parser.add_argument('--apply', action='store_true',
                        help='Escribir cambios. DRY-RUN sin esto.')
    parser.add_argument('--max-injections', type=int, default=DEFAULT_MAX_INJECTIONS,
                        help=f'Safety limit per run (default {DEFAULT_MAX_INJECTIONS})')
    args = parser.parse_args()

    print("=" * 72)
    header = "INJECT V8 REAL — APPLY" if args.apply else "INJECT V8 REAL — DRY-RUN"
    print(f"  {header}")
    print("=" * 72)

    # ─── Load inputs ───
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}")
        return 1
    with input_path.open() as f:
        candidates = json.load(f)
    if not isinstance(candidates, list):
        print(f"ERROR: input must be a list (got {type(candidates).__name__})")
        return 1
    print(f"\n[1/6] V8 REAL candidates loaded: {len(candidates)}")

    if not PROD_JSON.exists():
        print(f"ERROR: PROD_JSON missing: {PROD_JSON}")
        return 1
    with PROD_JSON.open() as f:
        prod = json.load(f)
    print(f"      Production bots:            {len(prod)}")

    # ─── Backup if apply ───
    if args.apply:
        backup_ts = datetime.utcnow().strftime('%Y%m%d_%H%M')
        backup = PROD_JSON.parent / f"v6_optimized_sl_tp.json.bak_v8real_inject_{backup_ts}"
        backup.write_text(PROD_JSON.read_text())
        print(f"      Backup:                     {backup.name}")

    # ─── Classify ───
    prod_idx = {}
    for i, b in enumerate(prod):
        k = (b.get('strategy'), b.get('symbol'), b.get('timeframe'))
        prod_idx[k] = (i, b)

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    obs_deadline_dt = now + timedelta(hours=OBS_DEADLINE_HOURS)
    obs_deadline_iso = obs_deadline_dt.isoformat()
    obs_deadline_ts = obs_deadline_dt.timestamp()

    to_inject = []
    already_v8_real = []
    already_v7_or_legacy = []

    for c in candidates:
        if c.get('status') != 'V8_PASS':
            continue  # defensivo, candidates list ya debería ser sólo V8_PASS
        key = (c.get('strategy'), c.get('symbol'), c.get('timeframe'))
        if key in prod_idx:
            _, bot = prod_idx[key]
            source = (bot.get('forensic') or {}).get('source')
            if source == 'v8_real':
                already_v8_real.append(key)
            else:
                already_v7_or_legacy.append((key, c, bot))
        else:
            to_inject.append(c)

    print(f"\n[2/6] Classification:")
    print(f"      To inject (new V8 REAL):    {len(to_inject)}")
    print(f"      Already V8 REAL (skip):     {len(already_v8_real)}")
    print(f"      V7.5 or legacy (upgrade):   {len(already_v7_or_legacy)}")

    # Safety limit
    if len(to_inject) > args.max_injections:
        print(f"\n  WARNING: {len(to_inject)} > max-injections={args.max_injections}. Truncating.")
        to_inject = to_inject[:args.max_injections]

    # ─── Build new V8 REAL bots ───
    print(f"\n[3/6] Building {len(to_inject)} new V8 REAL bots (observation_mode)...")
    new_bots = []
    for c in to_inject:
        forensic = build_forensic_meta(c, now_iso)
        pf = c['forensic_metrics'].get('profit_factor', 0)
        sharpe = c['forensic_metrics'].get('sharpe', 0)
        dd = c['forensic_metrics'].get('max_drawdown_pct', 0)
        sizing_label, size_usd = sizing_from_risk(pf, sharpe, dd)
        tier = tier_of(pf, sharpe)

        bot = {
            'strategy': c['strategy'],
            'symbol': c['symbol'],
            'timeframe': c['timeframe'],
            'params': c.get('best_params') or {},
            'sl_pct': float(c.get('sl', 0.40)),
            'tp_pct': 0.40,
            'leverage': int(c.get('leverage', 1)),
            'tier': tier,
            'size_usd': size_usd,
            'max_dur_h': max_dur_of(c['timeframe']),
            'test_wr': forensic['real_wr'],
            'test_pnl': forensic['real_pnl_pct'],
            'test_trades': forensic['total_trades'],
            'composite_score': round((pf or 0), 2),
            'round': 'V8_REAL_T1',
            'sl_tp_method': 'optuna_v7_empirical + forensic_v2 + v8_gates',
            'suspended': False,
            'observation_mode': True,
            'status': 'OBS',
            'status_set_at': now_iso,
            'obs_deadline': obs_deadline_ts,
            'obs_deadline_iso': obs_deadline_iso,
            'suspend_reason': None,
            'forensic': forensic,
        }
        new_bots.append(bot)

    if new_bots:
        sample = new_bots[0]
        print(f"      Sample: {sample['strategy']} x {sample['symbol'].replace('/USDT:USDT','')} "
              f"{sample['timeframe']} | size=${sample['size_usd']} tier={sample['tier']} "
              f"v8_score={sample['forensic']['v8_gates_score']:.2f}")

    # ─── MAX HARD upgrade of existing V7.5/legacy ───
    print(f"\n[4/6] MAX HARD upgrade on {len(already_v7_or_legacy)} existing bots...")
    upgraded = 0
    max_hard_raised = 0
    max_hard_kept = 0
    for (key, c, bot) in already_v7_or_legacy:
        pf = c['forensic_metrics'].get('profit_factor', 0)
        sharpe = c['forensic_metrics'].get('sharpe', 0)
        dd = c['forensic_metrics'].get('max_drawdown_pct', c['forensic_metrics'].get('max_dd_pct', 0))
        sizing_label, size_forensic = sizing_from_risk(pf, sharpe, dd)
        tier = tier_of(pf, sharpe)

        old_size = float(bot.get('size_usd', 10.0))
        new_size = max(old_size, size_forensic)
        forensic_suggested_lower = size_forensic < old_size
        size_max_hard_applied = forensic_suggested_lower  # flag cuando MAX HARD rescata

        # Update sizing (MAX HARD — never downgrade)
        bot['size_usd'] = new_size
        bot['tier'] = tier

        # Merge V8 REAL forensic metadata (preserve existing source)
        existing_forensic = bot.get('forensic', {}) or {}
        v8_forensic = build_forensic_meta(
            c, now_iso,
            forensic_suggested_lower=forensic_suggested_lower,
            size_max_hard_applied=size_max_hard_applied,
            forensic_suggested_size=size_forensic,
        )
        # Do NOT overwrite source (keep v7.5 as-is, add v8_real_also flag)
        existing_forensic['v8_real_also_passed'] = True
        existing_forensic['v8_real_score'] = v8_forensic['v8_gates_score']
        existing_forensic['v8_real_gates'] = v8_forensic['v8_gates_failed']
        existing_forensic['v8_real_validated_at'] = now_iso
        existing_forensic['v8_real_profile'] = v8_forensic['v8_profile']
        # MAX HARD audit flags (consumed by gold_rule_validator)
        existing_forensic['forensic_suggested_lower'] = forensic_suggested_lower
        existing_forensic['size_max_hard_applied'] = size_max_hard_applied
        existing_forensic['forensic_suggested_size_usd'] = size_forensic
        if forensic_suggested_lower:
            max_hard_kept += 1
        else:
            max_hard_raised += 1
        bot['forensic'] = existing_forensic
        upgraded += 1
    print(f"      Upgraded: {upgraded} (raised: {max_hard_raised}, kept higher: {max_hard_kept})")

    # ─── Apply or DRY-RUN ───
    print(f"\n[5/6] {'APPLY' if args.apply else 'DRY-RUN'}...")
    if args.apply:
        for b in new_bots:
            prod.append(b)
        with PROD_JSON.open('w') as f:
            json.dump(prod, f, indent=2)
        print(f"      Written:                    {PROD_JSON}")
        print(f"      New bot count:              {len(prod)} (+{len(new_bots)})")
    else:
        print(f"      DRY-RUN — no changes written.")
        print(f"      Would add {len(new_bots)} new V8 REAL bots")
        print(f"      Would upgrade {upgraded} existing bots (MAX HARD)")

    # ─── Report ───
    print(f"\n[6/6] Writing report...")
    report = {
        'timestamp': now_iso,
        'mode': 'APPLY' if args.apply else 'DRY_RUN',
        'input_file': str(input_path),
        'prod_bots_before': len(prod) - (len(new_bots) if args.apply else 0),
        'prod_bots_after': len(prod) if args.apply else (len(prod) + len(new_bots)),
        'candidates_total': len(candidates),
        'candidates_v8_pass': sum(1 for c in candidates if c.get('status') == 'V8_PASS'),
        'injected_new': len(new_bots),
        'already_v8_real_skipped': len(already_v8_real),
        'upgraded_existing': upgraded,
        'max_hard_raised': max_hard_raised,
        'max_hard_kept_higher': max_hard_kept,
        'sizing_map': SIZING_MAP,
        'obs_deadline_hours': OBS_DEADLINE_HOURS,
        'obs_deadline_iso': obs_deadline_iso,
        'rules_applied': ['R23', 'R24', 'R26', 'R27', 'MAX_HARD', 'ADENDUM_OPCION_A', 'V8_GATES'],
        'samples_injected': [
            {
                'strategy': b['strategy'],
                'symbol': b['symbol'],
                'timeframe': b['timeframe'],
                'size_usd': b['size_usd'],
                'tier': b['tier'],
                'v8_score': b['forensic']['v8_gates_score'],
                'wr': b['forensic']['real_wr'],
            }
            for b in new_bots[:10]
        ],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"v8_real_inject_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json"
    with report_path.open('w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"      Report:                     {report_path}")

    print("\n" + "=" * 72)
    if args.apply:
        print(f"DONE — APPLY — run `python3 BOT\\ V7/scripts/validate_json.py` next (R14)")
    else:
        print(f"DONE — DRY-RUN — use --apply to write")
    print("=" * 72)
    return 0


if __name__ == '__main__':
    sys.exit(main())
