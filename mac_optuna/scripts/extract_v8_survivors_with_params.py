#!/usr/bin/env python3
"""
Extract V8-survivors (the 137 grails that passed V8 Honesty Gates) with their
params, so we can re-run forensic_backtest.py FRESH and validate they're not
dataset artifacts.

V8 Honesty Gates:
  - gate_approved = True (from original forensic, Regla 24)
  - TF in {4h, 1d}           (G11)
  - profit_factor >= 2.0     (G12)
  - sharpe >= 2.0            (G13)
  - total_trades >= 50       (G15)

We match survivors against hetzner_wr65_for_forensic_v2.json which has
best_params/sl/leverage that were used to produce the original forensic.

Output format = forensic_backtest.py --input expected format:
  [{strategy, symbol, timeframe, best_params, sl, leverage, full_wr/optuna_wr}, ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/Users/sabrina/CLAUDE CODE/Estrategias")
LIGHT = ROOT / "data/forensic_v2_hetzner_23K_LIGHT.json"
WR65 = ROOT / "data/hetzner_wr65_for_forensic_v2.json"
OUT_DIR = ROOT / "data/empirical_analysis_20260415"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "v8_survivors_137_with_params.json"
REPORT_FILE = OUT_DIR / "v8_survivors_extraction_report.json"

# V8 Honesty Gates
V8_ALLOWED_TFS = {"4h", "1d"}
V8_MIN_PF = 2.0
V8_MIN_SHARPE = 2.0
V8_MIN_TRADES = 50


def passes_v8(record: dict) -> bool:
    """Record is a LIGHT result dict with gate_approved + metrics."""
    if not record.get("gate_approved"):
        return False
    if record.get("timeframe") not in V8_ALLOWED_TFS:
        return False
    m = record.get("metrics", {}) or {}
    if m.get("profit_factor", 0) < V8_MIN_PF:
        return False
    if m.get("sharpe", 0) < V8_MIN_SHARPE:
        return False
    if m.get("total_trades", 0) < V8_MIN_TRADES:
        return False
    return True


def key_of(strategy: str, symbol: str, tf: str) -> str:
    return f"{strategy}|{symbol}|{tf}"


def main() -> int:
    print(f"[1/4] Load LIGHT dataset: {LIGHT.name}")
    with LIGHT.open() as f:
        light = json.load(f)
    results = light.get("results", [])
    print(f"      {len(results):,} records total")
    print(f"      {sum(1 for r in results if r.get('gate_approved')):,} gate_approved (Regla 24 pass)")

    print(f"[2/4] Filter by V8 Honesty Gates (TF+PF+Sharpe+trades)...")
    survivors = [r for r in results if passes_v8(r)]
    print(f"      {len(survivors):,} V8 survivors")

    if not survivors:
        print("ERROR: 0 V8 survivors - aborting")
        return 1

    print(f"[3/4] Load hetzner_wr65 params: {WR65.name}")
    with WR65.open() as f:
        wr65 = json.load(f)
    print(f"      {len(wr65):,} entries (with best_params)")

    wr65_idx = {}
    for row in wr65:
        k = key_of(row.get("strategy"), row.get("symbol"), row.get("timeframe"))
        wr65_idx[k] = row
    print(f"      built index of {len(wr65_idx):,} (strategy|symbol|tf) keys")

    print(f"[4/4] Match V8 survivors against wr65 index for params...")
    matched = []
    missing = []

    for s in survivors:
        k = key_of(s.get("strategy"), s.get("symbol"), s.get("timeframe"))
        params_row = wr65_idx.get(k)
        if not params_row:
            missing.append({
                "strategy": s.get("strategy"),
                "symbol": s.get("symbol"),
                "timeframe": s.get("timeframe"),
            })
            continue

        # Build input record for forensic_backtest.py
        entry = {
            "strategy": s["strategy"],
            "symbol": s["symbol"],
            "timeframe": s["timeframe"],
            "best_params": params_row.get("best_params") or {},
            "sl": params_row.get("sl", 0.40),
            "leverage": params_row.get("leverage", 1),
            "optuna_wr": params_row.get("full_wr") or params_row.get("test_wr") or s.get("optuna_wr", 0),
            # Keep original forensic metrics for after-comparison
            "_original_forensic": {
                "win_rate": s["metrics"].get("win_rate"),
                "profit_factor": s["metrics"].get("profit_factor"),
                "sharpe": s["metrics"].get("sharpe"),
                "total_trades": s["metrics"].get("total_trades"),
                "total_pnl_pct": s["metrics"].get("total_pnl_pct"),
                "max_drawdown_pct": s["metrics"].get("max_drawdown_pct"),
            },
        }
        matched.append(entry)

    print(f"      matched: {len(matched):,}")
    print(f"      missing params: {len(missing):,}")

    with OUT_FILE.open("w") as f:
        json.dump(matched, f, indent=2)
    print(f"\n[OK] Wrote {OUT_FILE}")
    print(f"     {len(matched):,} survivors with params ready for forensic_backtest.py")

    report = {
        "timestamp": "2026-04-15",
        "total_light_records": len(results),
        "v8_survivors_count": len(survivors),
        "matched_with_params": len(matched),
        "missing_params": len(missing),
        "missing_sample": missing[:10],
        "v8_gates": {
            "allowed_tfs": sorted(V8_ALLOWED_TFS),
            "min_pf": V8_MIN_PF,
            "min_sharpe": V8_MIN_SHARPE,
            "min_trades": V8_MIN_TRADES,
        },
    }
    with REPORT_FILE.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"     Report: {REPORT_FILE}")

    # Quick distribution
    if matched:
        tfs = {}
        for m in matched:
            tfs[m["timeframe"]] = tfs.get(m["timeframe"], 0) + 1
        print(f"\nTF distribution of matched survivors:")
        for tf, n in sorted(tfs.items()):
            print(f"     {tf}: {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
