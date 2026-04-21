#!/usr/bin/env python3
"""HIGH-1: MC 500 perms + Regla 24 gate sobre los 41 C-tier sin mc_p_value.

Backtest engine: canary_runner.backtest_signal_exit (COST=0.30% round-trip,
entry next-bar open). Esto es el equivalente "forensic" en sandbox.

Gate (Mac's Regla 24):
  - mc_p_value < 0.05
  - forensic_wr_real >= 65%
  - (optuna_wr - forensic_wr) <= 10pp (aqui optuna_wr = stored wr, forensic_wr = re-backtest)
  - forensic_pnl_neto_fees > 0
  - n_clean_trades >= dynamic_min_trades(wr)

Output: results/mc_c_tier_41_upgrades_20260421.json
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "strategies_v7"))

from canary_runner import load_candles, backtest_signal_exit
import importlib

SRC_TF = {"5m": "5m", "15m": "5m", "1h": "1h", "4h": "1h", "1d": "1h"}
N_PERM = 500


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
    """Scan progress.json shortlists for passing entry with best WR."""
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


def mc_test(gen_fn, params, df, n_perm=500):
    r = backtest_signal_exit(df, gen_fn, params, sl_pct=0.40)
    if r.get("trades", 0) < 10:
        return None, r
    obs_total = r["total_pnl_pct"]
    n_trades = r["trades"]
    avg_win = r["avg_win_pct"]
    avg_loss = r["avg_loss_pct"]
    p_win = r["wins"] / n_trades
    rng = np.random.default_rng(42)
    perm_totals = []
    for _ in range(n_perm):
        signs = rng.random(n_trades) < p_win
        pnls = np.where(signs, avg_win, avg_loss)
        perm_totals.append(float(pnls.sum()))
    perm_arr = np.array(perm_totals)
    p_value = float((perm_arr >= obs_total).mean())
    return {"obs_total_pnl": obs_total, "p_value": p_value,
            "n_perm": n_perm, "n_trades": n_trades}, r


def main():
    v8 = json.loads((ROOT / "coordination" / "SANDBOX_FINAL_MASTER_V8_GRAILS.json").read_text())
    targets = [g for g in v8["grails"] if g.get("tier") == "C" and g.get("mc_p_value") is None]
    print(f"Targets: {len(targets)} C-tier sin mc_p_value")
    results = []
    upgrades = 0
    for i, g in enumerate(targets, 1):
        key = (g["strategy"], g["symbol"], g["tf"])
        row = {"strategy": g["strategy"], "symbol": g["symbol"], "tf": g["tf"],
               "stored_wr": g["wr"], "stored_trades": g["trades"], "stored_pf": g["pf"]}
        spec = find_strategy_spec(g["strategy"])
        if spec is None:
            row["error"] = "strategy not found"
            results.append(row); continue
        shortlist_entry = find_params(*key)
        if shortlist_entry is None:
            row["error"] = "params not found in any progress.json"
            results.append(row); continue
        params = shortlist_entry["params"]
        row["params"] = params
        try:
            df = load_candles(g["symbol"], SRC_TF[g["tf"]], g["tf"])
        except Exception as e:
            row["error"] = f"load_candles: {str(e)[:60]}"
            results.append(row); continue
        try:
            mc_res, backtest_res = mc_test(spec["gen"], params, df, N_PERM)
        except Exception as e:
            row["error"] = f"mc_test: {str(e)[:60]}"
            results.append(row); continue

        forensic_wr = backtest_res.get("wr")
        forensic_pnl = backtest_res.get("total_pnl_pct", 0)
        forensic_n = backtest_res.get("trades", 0)
        row["forensic_wr"] = forensic_wr
        row["forensic_pnl_pct"] = forensic_pnl
        row["forensic_trades"] = forensic_n
        row["forensic_pf"] = backtest_res.get("profit_factor")
        row["mc"] = mc_res

        # Regla 24 gate
        gate_pass = True
        reasons = []
        if mc_res is None:
            gate_pass = False; reasons.append(f"n_trades<10 ({forensic_n})")
        else:
            if mc_res["p_value"] >= 0.05:
                gate_pass = False; reasons.append(f"mc_p={mc_res['p_value']:.3f}>=0.05")
        if forensic_wr is None or forensic_wr < 65:
            gate_pass = False; reasons.append(f"forensic_wr={forensic_wr}")
        gap = abs((g["wr"] or 0) - (forensic_wr or 0))
        row["gap_wr"] = gap
        if gap > 10:
            gate_pass = False; reasons.append(f"gap={gap:.1f}pp>10")
        if forensic_pnl <= 0:
            gate_pass = False; reasons.append(f"pnl={forensic_pnl:.2f}<=0")
        min_t = dyn_min(forensic_wr or 0)
        if forensic_n < min_t:
            gate_pass = False; reasons.append(f"n={forensic_n}<min{min_t}")

        row["upgrade_to_B"] = gate_pass
        row["reject_reasons"] = reasons if not gate_pass else []
        if gate_pass:
            upgrades += 1
            print(f"  [{i}/{len(targets)}] UPGRADE {g['strategy']} {g['symbol']} {g['tf']} "
                  f"MC_p={mc_res['p_value']:.3f} WR={forensic_wr:.1f}% n={forensic_n}")
        else:
            print(f"  [{i}/{len(targets)}] reject {g['strategy']} {g['symbol']} {g['tf']} "
                  f"[{', '.join(reasons[:3])}]")
        results.append(row)

    out = ROOT / "results" / "mc_c_tier_41_upgrades_20260421.json"
    out.write_text(json.dumps({
        "_doc": "HIGH-1 MC + Regla 24 gate sobre C-tier sin mc_p_value",
        "_generated_at": "2026-04-21",
        "_n_targets": len(targets),
        "_n_upgrades": upgrades,
        "_n_errors": sum(1 for r in results if "error" in r),
        "results": results,
    }, indent=2, default=str))
    print(f"\nDONE: {upgrades}/{len(targets)} upgrades | wrote {out.name}")


if __name__ == "__main__":
    main()
