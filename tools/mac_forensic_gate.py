#!/usr/bin/env python3
"""
Mac forensic gate (gate 1 of 4) on wave m1 candidates.

Uses canary_runner.run_forensic with:
  - commission=0.0015 (0.15%/side), slippage=0.0 (implicit in commission)
  - entry at next-bar open (built-in)
  - SL=0.40, TP cap per TF (built-in)
  - funding deduction (built-in)
  - full history via load_candles

Outputs results/mac_m1_forensic_gate1.json with per-grail forensic_wr,
gap_pp, pnl_pct, max_trade_pct_of_pnl, n_clean, and PASS/FAIL vs gate:
  real_wr >= 70, gap_pp <= 5, pnl_pct > 0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "strategies_v7"))

from canary_runner import load_candles, run_forensic
import strategies_mac_batch3700 as batch


CANDIDATES = [
    {"strategy": "TV_TRIX_Cross", "symbol": "ETH",  "tf": "1d",
     "params": {"trix_len": 13, "sig_len": 17, "require_zero": 1},
     "optuna_wr": 77.8, "optuna_trades": 18, "optuna_pf": 7.659,
     "optuna_pnl_pct": 72.18},
    {"strategy": "TV_TRIX_Cross", "symbol": "XRP",  "tf": "1d",
     "params": {"trix_len": 9, "sig_len": 9, "require_zero": 1},
     "optuna_wr": 80.0, "optuna_trades": 15, "optuna_pf": 23.529,
     "optuna_pnl_pct": 113.45},
    {"strategy": "TV_TRIX_Cross", "symbol": "INJ",  "tf": "1d",
     "params": {"trix_len": 27, "sig_len": 20, "require_zero": 0},
     "optuna_wr": 78.6, "optuna_trades": 14, "optuna_pf": 2.62,
     "optuna_pnl_pct": 20.91},
    {"strategy": "TV_TRIX_Cross", "symbol": "NEAR", "tf": "1d",
     "params": {"trix_len": 24, "sig_len": 10, "require_zero": 1},
     "optuna_wr": 71.4, "optuna_trades": 14, "optuna_pf": 5.138,
     "optuna_pnl_pct": 48.79},
]

SOURCE_TF = {"1h": "1h", "4h": "1h", "1d": "1h", "5m": "5m", "15m": "5m"}

GATE_MIN_WR = 70.0
GATE_MAX_GAP_PP = 5.0
GATE_MIN_PNL = 0.0

# Dynamic min trades per R34
def dynamic_min_trades(wr):
    if wr >= 100: return 2
    if wr >= 96:  return 3
    if wr >= 90:  return 4
    if wr >= 80:  return 8
    if wr >= 70:  return 8
    return 999


def run_gate1():
    gen_map = {k: v["gen"] for k, v in batch.STRATEGY_EXPORT.items()}
    results = []
    for c in CANDIDATES:
        print(f"\n=== {c['strategy']} {c['symbol']} {c['tf']} ===")
        print(f"  optuna: WR={c['optuna_wr']}% n={c['optuna_trades']} PF={c['optuna_pf']:.2f} PnL={c['optuna_pnl_pct']:.1f}%")
        df = load_candles(c["symbol"], SOURCE_TF[c["tf"]], c["tf"])
        print(f"  candles: {len(df):,} bars  range={df.index.min()} -> {df.index.max()}")

        gen_fn = gen_map[c["strategy"]]
        r = run_forensic(df, gen_fn, c["params"],
                         sl_pct=0.40, timeframe=c["tf"],
                         commission=0.0015, slippage=0.0)

        if r.get("wr") is None:
            print(f"  FORENSIC_FAIL: {r.get('error')}")
            results.append({**c, "status": "FORENSIC_RUN_FAIL", "error": r.get("error")})
            continue

        forensic_wr = r["wr"]
        forensic_n = r["trades"]
        forensic_pnl = r["total_pnl_pct"]
        gap_pp = round(c["optuna_wr"] - forensic_wr, 2)

        pnls = [t["pnl_pct"] * 100 for t in r["trade_list"]]
        positive = [p for p in pnls if p > 0]
        if positive:
            max_trade = max(positive)
            sum_pos = sum(positive)
            max_trade_pct = round(100.0 * max_trade / sum_pos, 1) if sum_pos > 0 else None
        else:
            max_trade_pct = None
        dyn_min = dynamic_min_trades(forensic_wr)

        pass_wr = forensic_wr >= GATE_MIN_WR
        pass_gap = gap_pp <= GATE_MAX_GAP_PP
        pass_pnl = forensic_pnl > GATE_MIN_PNL
        pass_n = forensic_n >= dyn_min
        status = "PASS" if (pass_wr and pass_gap and pass_pnl and pass_n) else "FAIL"
        reasons = []
        if not pass_wr:  reasons.append(f"real_wr<70 ({forensic_wr}%)")
        if not pass_gap: reasons.append(f"gap>5pp ({gap_pp}pp)")
        if not pass_pnl: reasons.append(f"pnl<=0 ({forensic_pnl}%)")
        if not pass_n:   reasons.append(f"n<dynmin ({forensic_n}<{dyn_min})")

        print(f"  forensic: WR={forensic_wr}% n={forensic_n} PnL={forensic_pnl:.1f}% "
              f"gap={gap_pp:+.1f}pp max_trade_pct={max_trade_pct}%")
        print(f"  R24 gate 1 status: {status}" + (f" (reasons: {', '.join(reasons)})" if reasons else ""))

        results.append({
            **c,
            "forensic_wr": forensic_wr,
            "forensic_n_trades": forensic_n,
            "forensic_pnl_pct": forensic_pnl,
            "gap_pp": gap_pp,
            "max_trade_pct_of_pnl": max_trade_pct,
            "dynamic_min_trades": dyn_min,
            "fees_bps_per_side": 15,
            "engine": "canary_runner.run_forensic v2",
            "gate1_status": status,
            "gate1_reasons": reasons,
            "trade_list_summary": {
                "count": len(pnls),
                "pnls_pct": [round(p, 3) for p in pnls],
            },
        })

    out = ROOT / "results" / "mac_m1_forensic_gate1.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    n_pass = sum(1 for r in results if r.get("gate1_status") == "PASS")
    print(f"\nGate 1 summary: {n_pass}/{len(results)} PASS")
    return results


if __name__ == "__main__":
    run_gate1()
