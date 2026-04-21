#!/usr/bin/env python3
"""Block bootstrap MC sobre C-tier (Mac-approved replacement).

Mac ACK 2026-04-21: "usa block bootstrap, no i.i.d. per-trade.
block_size = max(5, min(10, int(sqrt(n_trades)))); n_boot = 5000 (o 10000)."

Reemplaza `mc_c_tier_upgrade.py` (sign-shuffle i.i.d. muy conservador).
Usa la lista real de per-trade PnLs del backtest y muestrea bloques
contiguos para preservar autocorrelación; luego permuta signos por bloque
para generar nula (PnL esperado bajo hipótesis de skill=0).

Gate (Regla 24):
  - mc_p_value < 0.05
  - forensic_wr_real >= 65%
  - |stored_wr - forensic_wr| <= 10pp
  - forensic_pnl_neto > 0
  - n_clean_trades >= dynamic_min

Output: results/mc_block_bootstrap_c_tier_<date>.json
"""
from __future__ import annotations
import json
import os
import sys
from datetime import date
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "strategies_v7"))

from canary_runner import load_candles, backtest_signal_exit
import importlib

SRC_TF = {"5m": "5m", "15m": "5m", "1h": "1h", "4h": "1h", "1d": "1h"}
N_BOOT = 5000


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


def block_bootstrap_mc(pnls, n_boot=N_BOOT, seed=42):
    """Block bootstrap + sign permutation test.

    Mac formula: block_size = max(5, min(10, int(sqrt(n)))).
    Null hypothesis: skill = 0 (random sign flips preserve serial correlation).
    p-value: fraction of bootstrap total_pnl >= observed.
    """
    pnls = np.array(pnls, dtype=float)
    n = len(pnls)
    if n < 10:
        return None
    block_size = max(5, min(10, int(np.sqrt(n))))
    observed = float(pnls.sum())
    rng = np.random.default_rng(seed)
    count_extreme = 0
    n_blocks_needed = n // block_size + 1
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks_needed)
        sample = np.concatenate([pnls[s:s + block_size] for s in starts])[:n]
        # Permute signs block-wise (preserves within-block serial structure)
        sign_blocks = rng.choice([-1, 1], size=n_blocks_needed)
        signed = np.concatenate([sign_blocks[i] * sample[i * block_size:(i + 1) * block_size]
                                 for i in range(n_blocks_needed)])[:n]
        if float(signed.sum()) >= observed:
            count_extreme += 1
    return {"p_value": count_extreme / n_boot, "n_boot": n_boot,
            "block_size": block_size, "observed_pnl": observed}


def main():
    v8 = json.loads((ROOT / "coordination" / "SANDBOX_FINAL_MASTER_V8_GRAILS.json").read_text())
    targets = [g for g in v8["grails"] if g.get("tier") == "C" and g.get("mc_p_value") is None]
    print(f"Targets: {len(targets)} C-tier sin mc_p_value (block bootstrap test)")
    results = []
    upgrades = 0
    for i, g in enumerate(targets, 1):
        key = (g["strategy"], g["symbol"], g["tf"])
        row = {"strategy": g["strategy"], "symbol": g["symbol"], "tf": g["tf"],
               "stored_wr": g["wr"], "stored_trades": g["trades"], "stored_pf": g["pf"]}
        spec = find_strategy_spec(g["strategy"])
        if spec is None:
            row["error"] = "strategy not found"; results.append(row); continue
        entry = find_params(*key)
        if entry is None:
            row["error"] = "params not found"; results.append(row); continue
        row["params"] = entry["params"]
        try:
            df = load_candles(g["symbol"], SRC_TF[g["tf"]], g["tf"])
        except Exception as e:
            row["error"] = f"load_candles: {str(e)[:60]}"; results.append(row); continue
        try:
            bt = backtest_signal_exit(df, spec["gen"], entry["params"], sl_pct=0.40)
        except Exception as e:
            row["error"] = f"backtest: {str(e)[:60]}"; results.append(row); continue

        pnls = bt.get("pnls_pct", [])
        if not pnls:
            row["error"] = "no pnls"; results.append(row); continue

        forensic_wr = bt.get("wr"); forensic_pnl = bt.get("total_pnl_pct", 0)
        forensic_n = bt.get("trades", 0)
        row.update({"forensic_wr": forensic_wr, "forensic_pnl_pct": forensic_pnl,
                    "forensic_trades": forensic_n,
                    "forensic_pf": bt.get("profit_factor")})
        mc = block_bootstrap_mc(pnls)
        if mc is None:
            row["error"] = f"n<10 ({forensic_n})"; results.append(row); continue
        row["mc_block_bootstrap"] = mc

        # Regla 24 gate
        gate = True; reasons = []
        if mc["p_value"] >= 0.05:
            gate = False; reasons.append(f"mc_p={mc['p_value']:.3f}")
        if forensic_wr is None or forensic_wr < 65:
            gate = False; reasons.append(f"wr={forensic_wr}")
        gap = abs((g["wr"] or 0) - (forensic_wr or 0))
        row["gap_wr"] = gap
        if gap > 10:
            gate = False; reasons.append(f"gap={gap:.1f}pp")
        if forensic_pnl <= 0:
            gate = False; reasons.append(f"pnl={forensic_pnl:.2f}")
        min_t = dyn_min(forensic_wr or 0)
        if forensic_n < min_t:
            gate = False; reasons.append(f"n={forensic_n}<min{min_t}")

        row["upgrade_to_B"] = gate
        row["reject_reasons"] = reasons if not gate else []
        if gate:
            upgrades += 1
            print(f"  [{i}/{len(targets)}] UPGRADE {g['strategy'][:22]:22s} "
                  f"{g['symbol']:6s} {g['tf']:3s} p={mc['p_value']:.3f} "
                  f"WR={forensic_wr:.1f}% n={forensic_n} block={mc['block_size']}")
        else:
            print(f"  [{i}/{len(targets)}] reject  {g['strategy'][:22]:22s} "
                  f"{g['symbol']:6s} {g['tf']:3s} [{', '.join(reasons[:3])}]")
        results.append(row)

    today = date.today().isoformat().replace("-", "")
    out = ROOT / "results" / f"mc_block_bootstrap_c_tier_{today}.json"
    out.write_text(json.dumps({
        "_doc": "C-tier MC block bootstrap (Mac-approved replacement for sign-shuffle)",
        "_method": "block_size=max(5,min(10,sqrt(n))) n_boot=5000 Regla24",
        "_generated_at": today,
        "_n_targets": len(targets),
        "_n_upgrades": upgrades,
        "results": results,
    }, indent=2, default=str))
    print(f"\nDONE: {upgrades}/{len(targets)} upgrades B-tier | wrote {out.name}")


if __name__ == "__main__":
    main()
