#!/usr/bin/env python3
"""
Gate 2 (plateau) on Mac wave m1 survivors of gate 1.

For each grail: jitter numeric params by [-20%, -10%, 0, +10%, +20%],
re-run run_forensic, count neighbors passing a LOOSE gate:
    WR >= 65, n >= 10, PnL > 0

Plateau PASS: >= 60% of neighbor combos pass.

Categorical params (require_zero) are fixed at Optuna's choice (not jitterable).
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "strategies_v7"))

from canary_runner import load_candles, run_forensic
import strategies_mac_batch3700 as batch


SOURCE_TF = {"1h": "1h", "4h": "1h", "1d": "1h"}
JITTERS = [-0.20, -0.10, 0.0, 0.10, 0.20]
LOOSE_WR = 65.0
LOOSE_N = 10
LOOSE_PNL = 0.0
PLATEAU_THR = 0.60

# param categorical: keep fixed (not jitterable)
CATEGORICAL_PARAMS = {"require_zero", "mode"}


def jitter_param(val, space_spec):
    """Return a list of jittered candidate values.

    space_spec is a tuple like ('int', lo, hi) or ('float', lo, hi).
    For int, round and clip to [lo, hi] and dedup.
    For float, clip to [lo, hi].
    """
    kind, lo, hi = space_spec[0], space_spec[1], space_spec[2]
    out = set()
    for j in JITTERS:
        v = val * (1.0 + j)
        if kind == "int":
            vi = int(round(v))
            vi = max(int(lo), min(int(hi), vi))
            out.add(vi)
        else:
            v = max(float(lo), min(float(hi), v))
            out.add(round(v, 6))
    return sorted(out)


def run_plateau(grail):
    """Return dict with plateau_pct, neighbors_pass/total, and per-combo detail."""
    strat_name = grail["strategy"]
    sym = grail["symbol"]
    tf = grail["tf"]
    params = dict(grail["params"])
    spec = batch.STRATEGY_EXPORT[strat_name]
    space = spec["space"]()
    gen_fn = spec["gen"]

    df = load_candles(sym, SOURCE_TF[tf], tf)

    # build jittered grid
    axes = {}
    for pname, pval in params.items():
        if pname in CATEGORICAL_PARAMS:
            axes[pname] = [pval]
            continue
        if pname not in space:
            axes[pname] = [pval]
            continue
        axes[pname] = jitter_param(pval, space[pname])

    keys = list(axes.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[axes[k] for k in keys])]
    # remove center (exact match to grail params) to measure neighbors
    combos = [c for c in combos if c != params]

    neighbors = []
    for c in combos:
        r = run_forensic(df, gen_fn, c, sl_pct=0.40, timeframe=tf,
                         commission=0.0015, slippage=0.0)
        wr = r.get("wr")
        trades = r.get("trades", 0) or 0
        pnl = r.get("total_pnl_pct", 0.0) or 0.0
        if wr is None or trades == 0:
            passes = False
        else:
            passes = (wr >= LOOSE_WR and trades >= LOOSE_N and pnl > LOOSE_PNL)
        neighbors.append({
            "params": c, "wr": wr, "trades": trades, "pnl_pct": pnl,
            "passes_loose": bool(passes),
        })
    pass_n = sum(1 for n in neighbors if n["passes_loose"])
    total = len(neighbors)
    pct = (pass_n / total * 100.0) if total else 0.0
    return {
        "strategy": strat_name, "symbol": sym, "tf": tf,
        "grail_params": params,
        "n_neighbors": total,
        "n_neighbors_passing": pass_n,
        "plateau_pct_passing": round(pct, 1),
        "plateau_threshold_pct": PLATEAU_THR * 100,
        "gate2_status": "PASS" if pct >= PLATEAU_THR * 100 else "FAIL",
        "neighbors": neighbors,
    }


def main():
    # Read gate 1 results
    gate1 = json.loads((ROOT / "results" / "mac_m1_forensic_gate1.json").read_text())

    # Eligible for gate 2: gate1 PASS AND concentration risk <= 40%
    survivors = []
    for g in gate1:
        if g.get("gate1_status") != "PASS":
            print(f"[skip] {g['strategy']} {g['symbol']} {g['tf']}: gate1={g.get('gate1_status')}")
            continue
        mt = g.get("max_trade_pct_of_pnl")
        if mt is not None and mt > 40.0:
            print(f"[reject concentration] {g['strategy']} {g['symbol']} {g['tf']}: max_trade={mt}%")
            continue
        survivors.append(g)

    print(f"\nGate 2 input: {len(survivors)} survivors\n")

    results = []
    for g in survivors:
        print(f"=== plateau {g['strategy']} {g['symbol']} {g['tf']} ===")
        r = run_plateau(g)
        print(f"  neighbors passing: {r['n_neighbors_passing']}/{r['n_neighbors']} "
              f"= {r['plateau_pct_passing']}%  threshold={r['plateau_threshold_pct']}%  "
              f"status={r['gate2_status']}")
        r.update({k: v for k, v in g.items() if k not in r})
        results.append(r)

    out = ROOT / "results" / "mac_m1_plateau_gate2.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
