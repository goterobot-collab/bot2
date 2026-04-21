#!/usr/bin/env python3
"""MED-3: Holdout OOS 20% sobre top-10 score del V8 master.

SINGLE-SHOT test — re-run contamina el OOS. Aplicar Regla 24 gate al
resultado OOS.

Método:
  1. Cargar top-10 grails por score desde V8 master.
  2. Localizar params en progress.json.
  3. Para cada grail: backtest sobre el último 20% del history.
  4. Aplicar Regla 24: WR ≥ 65%, gap ≤ 10pp vs WR stored, PnL > 0,
     n_clean ≥ dynamic_min.

Output: results/holdout_top10_20260421.json
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "strategies_v7"))

from canary_runner import load_candles, backtest_signal_exit
import importlib

SRC_TF = {"5m": "5m", "15m": "5m", "1h": "1h", "4h": "1h", "1d": "1h"}
HOLDOUT_FRAC = 0.20


def dyn_min(wr):
    if wr >= 100: return 2
    if wr >= 90:  return 4
    if wr >= 80:  return 8
    if wr >= 70:  return 8
    if wr >= 65:  return 10
    return 999


def find_strategy_spec(name):
    for f in sorted(os.listdir(ROOT / "strategies_v7")):
        if not f.endswith(".py") or f.startswith("__"): continue
        try:
            mod = importlib.import_module(f[:-3])
            if hasattr(mod, "STRATEGY_EXPORT") and name in mod.STRATEGY_EXPORT:
                return mod.STRATEGY_EXPORT[name]
        except Exception:
            continue
    return None


def find_params(strategy, symbol, tf):
    best = None
    for fn in sorted(os.listdir(ROOT / "results")):
        if "progress" not in fn:
            continue
        try:
            d = json.loads((ROOT / "results" / fn).read_text())
        except Exception:
            continue
        for s in d.get("shortlist", []):
            if not s.get("passes"):
                continue
            if (s["strategy"], s["symbol"], s["tf"]) != (strategy, symbol, tf):
                continue
            if best is None or s["metrics"].get("wr", 0) > best["metrics"].get("wr", 0):
                best = s
    return best


def holdout_eval(gen_fn, params, df):
    n = len(df)
    cut = int(n * (1 - HOLDOUT_FRAC))
    sub = df.iloc[cut:].reset_index(drop=True)
    return backtest_signal_exit(sub, gen_fn, params, sl_pct=0.40)


def main():
    v8 = json.loads((ROOT / "coordination" / "SANDBOX_FINAL_MASTER_V8_GRAILS.json").read_text())
    # Top-10 by score
    top10 = sorted(v8["grails"], key=lambda g: -g.get("score", 0))[:10]
    print(f"TOP-10 by score (single-shot holdout):")
    results = []
    survivors = 0
    for i, g in enumerate(top10, 1):
        row = {"rank": i, "strategy": g["strategy"], "symbol": g["symbol"],
               "tf": g["tf"], "stored_wr": g["wr"], "stored_trades": g["trades"],
               "stored_pf": g["pf"], "stored_score": g["score"], "tier": g["tier"]}
        spec = find_strategy_spec(g["strategy"])
        if spec is None:
            row["error"] = "strategy not found"; results.append(row); continue
        entry = find_params(g["strategy"], g["symbol"], g["tf"])
        if entry is None:
            row["error"] = "params not found"; results.append(row); continue
        row["params"] = entry["params"]
        try:
            df = load_candles(g["symbol"], SRC_TF[g["tf"]], g["tf"])
        except Exception as e:
            row["error"] = f"load_candles: {str(e)[:60]}"; results.append(row); continue

        try:
            ho = holdout_eval(spec["gen"], entry["params"], df)
        except Exception as e:
            row["error"] = f"holdout: {str(e)[:60]}"; results.append(row); continue

        ho_wr = ho.get("wr")
        ho_pnl = ho.get("total_pnl_pct", 0)
        ho_n = ho.get("trades", 0)
        ho_pf = ho.get("profit_factor")
        row["holdout_wr"] = ho_wr
        row["holdout_pnl_pct"] = ho_pnl
        row["holdout_trades"] = ho_n
        row["holdout_pf"] = ho_pf

        # Regla 24 gate
        survives = True
        reasons = []
        if ho_wr is None or ho_wr < 65:
            survives = False; reasons.append(f"ho_wr={ho_wr}")
        gap = abs((g["wr"] or 0) - (ho_wr or 0))
        row["gap_wr"] = gap
        if gap > 10:
            survives = False; reasons.append(f"gap={gap:.1f}pp")
        if ho_pnl <= 0:
            survives = False; reasons.append(f"pnl={ho_pnl:.2f}")
        min_t = dyn_min(ho_wr or 0)
        if ho_n < min_t:
            survives = False; reasons.append(f"n={ho_n}<min{min_t}")

        row["survives_regla24"] = survives
        row["reject_reasons"] = reasons if not survives else []
        if survives:
            survivors += 1
            print(f"  [{i}] SURVIVE  {g['strategy']:30s} {g['symbol']:6s} {g['tf']:3s}  "
                  f"HO: WR={ho_wr:.1f}% n={ho_n} PnL={ho_pnl:.1f}% gap={gap:.1f}pp")
        else:
            print(f"  [{i}] FAIL     {g['strategy']:30s} {g['symbol']:6s} {g['tf']:3s}  "
                  f"[{', '.join(reasons[:3])}]")
        results.append(row)

    out = ROOT / "results" / "holdout_top10_20260421.json"
    out.write_text(json.dumps({
        "_doc": "MED-3 Holdout OOS 20% top-10 score + Regla 24",
        "_generated_at": "2026-04-21",
        "_holdout_frac": HOLDOUT_FRAC,
        "_n_targets": len(top10),
        "_n_survivors": survivors,
        "results": results,
    }, indent=2, default=str))
    print(f"\nDONE: {survivors}/{len(top10)} survivors | wrote {out.name}")


if __name__ == "__main__":
    main()
