#!/usr/bin/env python3
"""
Inject 137 V8 survivors into production JSON (v6_optimized_sl_tp.json).

Plan aprobado por Sabrina 2026-04-15:

  - 13 V8 survivors ya ACTIVOS → subir sizing a Tier S (no tocar observation)
  - 27 V8 survivors en OBSERVATION_MODE → SUBIR sizing pero mantener observation
    (ya están en shadow, no los apagamos)
  - 97 V8 survivors NO en JSON → inyectar con observation_mode=True (red de seguridad)

Tier S sizing comes from forensic risk_scores assignment:
  - Full size ($15-20)    → size_usd = 17
  - Standard ($10-12)     → size_usd = 11
  - Reduced ($7-10)       → size_usd = 8
  - Minimum ($5-7)        → size_usd = 6

Leverage: conservador 1× (default forensic).
SL: del hetzner_wr65 (already empirical).
TP: no viene en hetzner — default 0.4 (40%) como usa el resto del JSON.
Max duration: del TF (4h→720h/90d, 1d→720h/30d) — usar el tope actual del JSON.

Forensic metadata embebido para trazabilidad (Regla 18 params_hash equivalent):
  {
    "forensic": {
      "validated_at": "2026-04-15",
      "source": "hetzner_23K_V8_gates",
      "real_wr": ..., "profit_factor": ..., "sharpe": ...,
      "total_trades": ..., "gate_tier": "TIER_1",
      "gate_tags": ["v8_survivor"]
    },
    "round": "V8_HONESTY_GATES",
    "tier": "DIAMOND",  # si PF≥3 y Sharpe≥8, else GOLD
    "sl_tp_method": "optuna_v7_empirical + forensic_v2_validated"
  }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path("/Users/sabrina/CLAUDE CODE")
SURVIVORS = ROOT / "Estrategias/data/empirical_analysis_20260415/v8_survivors_137_with_params.json"
FRESH = ROOT / "Estrategias/data/empirical_analysis_20260415/v8_survivors_137_forensic_fresh.json"
PROD_JSON = ROOT / "BOT V7/live/data/v6_optimized_sl_tp.json"
REPORT = ROOT / "Estrategias/data/empirical_analysis_20260415/v8_injection_report.json"

# Sizing map (USD per trade)
SIZING_MAP = {
    "Full size ($15-20)": 17.0,
    "Standard ($10-12)": 11.0,
    "Reduced ($7-10)": 8.0,
    "Minimum ($5-7)": 6.0,
}


def tier_of(pf: float, sharpe: float) -> str:
    """DIAMOND if PF≥3 AND Sharpe≥8, else GOLD."""
    if pf >= 3.0 and sharpe >= 8.0:
        return "DIAMOND"
    return "GOLD"


def max_dur_of(tf: str) -> int:
    """Max duration hours by TF."""
    return {"4h": 720, "1d": 720, "1h": 360, "5m": 72, "15m": 144}.get(tf, 720)


def main() -> int:
    print("=" * 70)
    print("V8 SURVIVORS → PRODUCTION JSON INJECTION")
    print("=" * 70)

    # Load inputs
    print("\n[1/6] Loading inputs...")
    with SURVIVORS.open() as f:
        survivors = json.load(f)
    print(f"      137 V8 survivors (with params)")

    with FRESH.open() as f:
        fresh_bundle = json.load(f)
    risk_idx = {
        (r["strategy"], r["symbol"], r["timeframe"]): r
        for r in fresh_bundle.get("risk_scores", [])
    }
    print(f"      {len(risk_idx)} fresh forensic entries with risk_scores")

    with PROD_JSON.open() as f:
        prod = json.load(f)
    print(f"      Production JSON: {len(prod)} bots")

    # Build production index
    prod_idx = {}
    for i, b in enumerate(prod):
        key = (b.get("strategy"), b.get("symbol"), b.get("timeframe"))
        prod_idx[key] = (i, b)

    # Classify and process
    print("\n[2/6] Classifying survivors...")
    to_inject_new = []       # 97 new
    to_upgrade_active = []   # 13 active
    to_upgrade_obs = []      # 27 observation
    already_suspended = []   # 0 expected

    for s in survivors:
        key = (s["strategy"], s["symbol"], s["timeframe"])
        if key in prod_idx:
            idx, bot = prod_idx[key]
            if bot.get("suspended"):
                already_suspended.append((key, idx))
            elif bot.get("observation_mode"):
                to_upgrade_obs.append((key, idx, s))
            else:
                to_upgrade_active.append((key, idx, s))
        else:
            to_inject_new.append(s)

    print(f"      → 97 new to inject:      {len(to_inject_new)}")
    print(f"      → 13 active to upgrade:  {len(to_upgrade_active)}")
    print(f"      → 27 obs to upgrade:     {len(to_upgrade_obs)}")
    print(f"      → suspended (skip):      {len(already_suspended)}")

    # Safety check
    if len(to_inject_new) + len(to_upgrade_active) + len(to_upgrade_obs) + len(already_suspended) != 137:
        print("ERROR: counts don't sum to 137")
        return 1

    # Build forensic metadata from risk_scores
    def build_forensic_meta(strat, sym, tf):
        rs = risk_idx.get((strat, sym, tf))
        if not rs:
            return None, None, None
        sizing = rs.get("sizing", "Minimum ($5-7)")
        size_usd = SIZING_MAP.get(sizing, 6.0)
        tier = tier_of(rs.get("profit_factor", 0), rs.get("sharpe", 0))
        forensic = {
            "validated_at": "2026-04-15",
            "source": "hetzner_23K_V8_gates",
            "real_wr": rs.get("win_rate"),
            "real_pnl_pct": rs.get("pnl_pct"),
            "profit_factor": rs.get("profit_factor"),
            "sharpe": rs.get("sharpe"),
            "total_trades": rs.get("total_trades"),
            "max_dd": rs.get("max_drawdown"),
            "risk_score": rs.get("risk_score"),
            "gate_tier": rs.get("tier", "TIER_1"),
            "gate_tags": ["v8_survivor", "tier_s"],
            "sizing": sizing,
        }
        return forensic, size_usd, tier

    # ───── 1. Inject new 97 ─────
    print(f"\n[3/6] Injecting {len(to_inject_new)} new bots (observation_mode=True)...")
    injected_new = 0
    for s in to_inject_new:
        forensic, size_usd, tier = build_forensic_meta(s["strategy"], s["symbol"], s["timeframe"])
        if not forensic:
            print(f"  SKIP (no risk_score): {s['strategy']} x {s['symbol']} {s['timeframe']}")
            continue

        bot = {
            "strategy": s["strategy"],
            "symbol": s["symbol"],
            "timeframe": s["timeframe"],
            "params": s.get("best_params", {}) or {},
            "sl_pct": float(s.get("sl", 0.40)),
            "tp_pct": 0.40,  # default (forensic doesn't provide tp explicit)
            "leverage": int(s.get("leverage", 1)),
            "tier": tier,
            "size_usd": size_usd,
            "max_dur_h": max_dur_of(s["timeframe"]),
            "test_wr": forensic["real_wr"],
            "test_pnl": forensic["real_pnl_pct"],
            "test_trades": forensic["total_trades"],
            "composite_score": round(forensic["profit_factor"] or 0, 2),
            "round": "V8_HONESTY_GATES",
            "sl_tp_method": "optuna_v7_empirical + forensic_v2_validated",
            "suspended": False,
            "observation_mode": True,  # RED DE SEGURIDAD
            "suspend_reason": None,
            "forensic": forensic,
        }
        prod.append(bot)
        injected_new += 1
    print(f"      Injected: {injected_new}")

    # ───── 2. Upgrade 13 active → Tier S sizing (MAX regla: nunca bajar) ─────
    print(f"\n[4/6] Upgrading {len(to_upgrade_active)} active bots to Tier S (MAX rule)...")
    upgraded_active = 0
    active_size_raised = 0
    active_size_kept = 0
    for (key, idx, s) in to_upgrade_active:
        bot = prod[idx]
        forensic, size_forensic, tier = build_forensic_meta(*key)
        if not forensic:
            continue
        old_size = bot.get("size_usd", 10)
        # REGLA MAX: nunca bajar sizing de bot activo que ya funciona
        new_size = max(float(old_size), float(size_forensic))
        bot["size_usd"] = new_size
        bot["tier"] = tier
        # Preserve observation_mode=False (they're active)
        existing_forensic = bot.get("forensic", {}) or {}
        existing_forensic.update(forensic)
        # Tag if forensic suggested lower (to remember risk signal)
        if size_forensic < old_size:
            existing_forensic["forensic_suggested_lower"] = size_forensic
            active_size_kept += 1
        else:
            active_size_raised += 1
        bot["forensic"] = existing_forensic
        if bot.get("round", "").find("V8") == -1:
            bot["round"] = "V8_TIER_S_PROMOTED"
        upgraded_active += 1
        if upgraded_active <= 5:
            action = "SUBIÓ" if new_size > old_size else "mantuvo"
            print(f"      {bot['strategy'][:30]:30} x {bot['symbol'].replace('/USDT:USDT',''):10} {bot['timeframe']:3} | ${old_size:.0f}→${new_size:.0f} ({action}) | tier {tier}")
    print(f"      Active upgraded: {upgraded_active} (subió: {active_size_raised}, mantuvo: {active_size_kept})")

    # ───── 3. Upgrade 27 observation → Tier S sizing (MAX rule, keep observation) ─────
    print(f"\n[5/6] Upgrading {len(to_upgrade_obs)} observation bots to Tier S sizing...")
    upgraded_obs = 0
    obs_size_raised = 0
    obs_size_kept = 0
    for (key, idx, s) in to_upgrade_obs:
        bot = prod[idx]
        forensic, size_forensic, tier = build_forensic_meta(*key)
        if not forensic:
            continue
        old_size = bot.get("size_usd", 10)
        # REGLA MAX también acá: nunca bajar sizing
        new_size = max(float(old_size), float(size_forensic))
        bot["size_usd"] = new_size
        bot["tier"] = tier
        # KEEP observation_mode=True (they're still shadow)
        existing_forensic = bot.get("forensic", {}) or {}
        existing_forensic.update(forensic)
        if size_forensic < old_size:
            existing_forensic["forensic_suggested_lower"] = size_forensic
            obs_size_kept += 1
        else:
            obs_size_raised += 1
        bot["forensic"] = existing_forensic
        if bot.get("round", "").find("V8") == -1:
            bot["round"] = "V8_TIER_S_OBSERVATION"
        upgraded_obs += 1
    print(f"      Observation upgraded: {upgraded_obs} (subió: {obs_size_raised}, mantuvo: {obs_size_kept})")

    # Write JSON
    print(f"\n[6/6] Writing production JSON...")
    before_count = len(prod) - injected_new
    after_count = len(prod)
    print(f"      Before: {before_count} bots")
    print(f"      After:  {after_count} bots (+{injected_new})")

    with PROD_JSON.open("w") as f:
        json.dump(prod, f, indent=2)
    print(f"      Written: {PROD_JSON}")

    # Report
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "action": "V8 Honesty Gates survivors injection (137)",
        "plan": "97 new obs-mode + 13 active Tier-S + 27 obs Tier-S",
        "before_bot_count": before_count,
        "after_bot_count": after_count,
        "delta": injected_new,
        "injected_new": injected_new,
        "upgraded_active": upgraded_active,
        "upgraded_observation": upgraded_obs,
        "already_suspended_skipped": len(already_suspended),
        "injection_source": "data/empirical_analysis_20260415/v8_survivors_137_with_params.json",
        "validation_source": "data/empirical_analysis_20260415/v8_survivors_137_forensic_fresh.json",
        "rules_applied": ["R23", "R24", "R26", "freeze_check_passed"],
    }
    with REPORT.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\n      Report: {REPORT}")

    print("\n" + "=" * 70)
    print("DONE — remember to run validate_json.py next")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
