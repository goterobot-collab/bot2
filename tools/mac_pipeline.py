#!/usr/bin/env python3
"""
End-to-end 4-gate Mac pipeline for any wave(s).

Usage:
    python3 tools/mac_pipeline.py --waves m2 m3 --min-wr-eligible 70 --min-n 15

Loads results/mac_<wave>_1h_4h_1d_progress.json, filters candidates with
WR>=min_wr_eligible and trades>=min_n, applies:
    1. Forensic gate (WR>=70 real, gap<=5pp, PnL>0)
    2. Concentration (max_trade_pct <= 40)
    3. Plateau (+-20% jitter, 60% threshold)
    4. MC bootstrap (500 perms, p<0.05)

Writes per-wave gate1/gate2/gate3 JSONs and a combined master.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "strategies_v7"))

from canary_runner import load_candles, run_forensic


SOURCE_TF = {"1h": "1h", "4h": "1h", "1d": "1h", "5m": "5m", "15m": "5m"}
JITTERS = [-0.20, -0.10, 0.0, 0.10, 0.20]
CATEGORICAL_PARAMS = {"require_zero", "mode"}

GATE1_WR = 70.0
GATE1_GAP = 5.0
GATE1_PNL = 0.0
GATE_CONCENTRATION_MAX = 40.0
LOOSE_WR = 65.0
LOOSE_N = 10
LOOSE_PNL = 0.0
PLATEAU_THR = 0.60
MC_PERMS = 500
MC_PVALUE_MAX = 0.05


def dyn_min_trades(wr):
    if wr >= 100: return 2
    if wr >= 96:  return 3
    if wr >= 90:  return 4
    if wr >= 80:  return 8
    if wr >= 70:  return 8
    return 999


def load_wave_candidates(wave, min_wr, min_n):
    path = ROOT / "results" / f"mac_{wave}_1h_4h_1d_progress.json"
    if not path.exists():
        print(f"[warn] {path} missing, skipping")
        return []
    d = json.loads(path.read_text())
    out = []
    for r in d.get("shortlist", []):
        if not r.get("passes"):
            continue
        m = r["metrics"]
        if m["wr"] < min_wr or m["trades"] < min_n:
            continue
        out.append({
            "wave": wave,
            "batch": r["batch"],
            "strategy": r["strategy"],
            "symbol": r["symbol"],
            "tf": r["tf"],
            "params": r["params"],
            "optuna_wr": m["wr"],
            "optuna_trades": m["trades"],
            "optuna_pf": m["pf"],
            "optuna_pnl_pct": m["total_pnl_pct"],
        })
    return out


def load_gen(batch, name):
    mod = importlib.import_module(f"strategies_mac_batch{batch}")
    spec = mod.STRATEGY_EXPORT[name]
    return spec["gen"], spec["space"]()


def gate1_forensic(cand):
    gen_fn, _ = load_gen(cand["batch"], cand["strategy"])
    df = load_candles(cand["symbol"], SOURCE_TF[cand["tf"]], cand["tf"])
    r = run_forensic(df, gen_fn, cand["params"], sl_pct=0.40,
                     timeframe=cand["tf"], commission=0.0015, slippage=0.0)
    if r.get("wr") is None:
        return {**cand, "gate1_status": "RUN_FAIL", "gate1_error": r.get("error")}

    forensic_wr = r["wr"]
    forensic_n = r["trades"]
    forensic_pnl = r["total_pnl_pct"]
    gap_pp = round(cand["optuna_wr"] - forensic_wr, 2)

    pnls = [t["pnl_pct"] * 100 for t in r["trade_list"]]
    positives = [p for p in pnls if p > 0]
    if positives:
        max_trade_pct = round(100.0 * max(positives) / sum(positives), 1)
    else:
        max_trade_pct = None

    dyn_n = dyn_min_trades(forensic_wr)
    pass_wr = forensic_wr >= GATE1_WR
    pass_gap = gap_pp <= GATE1_GAP
    pass_pnl = forensic_pnl > GATE1_PNL
    pass_n = forensic_n >= dyn_n
    ok = pass_wr and pass_gap and pass_pnl and pass_n

    reasons = []
    if not pass_wr:  reasons.append(f"real_wr<70 ({forensic_wr}%)")
    if not pass_gap: reasons.append(f"gap>5pp ({gap_pp}pp)")
    if not pass_pnl: reasons.append(f"pnl<=0 ({forensic_pnl}%)")
    if not pass_n:   reasons.append(f"n<dynmin ({forensic_n}<{dyn_n})")

    return {
        **cand,
        "forensic_wr": forensic_wr,
        "forensic_n_trades": forensic_n,
        "forensic_pnl_pct": forensic_pnl,
        "gap_pp": gap_pp,
        "max_trade_pct_of_pnl": max_trade_pct,
        "dynamic_min_trades": dyn_n,
        "trade_pnls_pct": [round(p, 4) for p in pnls],
        "gate1_status": "PASS" if ok else "FAIL",
        "gate1_reasons": reasons,
    }


def gate_concentration(g1):
    if g1.get("gate1_status") != "PASS":
        return g1
    mt = g1.get("max_trade_pct_of_pnl")
    if mt is None:
        return {**g1, "gate_concentration_status": "FAIL_NO_POSITIVES"}
    status = "PASS" if mt <= GATE_CONCENTRATION_MAX else "FAIL"
    return {**g1, "gate_concentration_status": status}


def jitter_int(val, space_spec):
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


def gate2_plateau(g):
    if g.get("gate_concentration_status") != "PASS":
        return g
    gen_fn, space = load_gen(g["batch"], g["strategy"])
    df = load_candles(g["symbol"], SOURCE_TF[g["tf"]], g["tf"])
    axes = {}
    params = dict(g["params"])
    for pname, pval in params.items():
        if pname in CATEGORICAL_PARAMS or pname not in space:
            axes[pname] = [pval]
            continue
        axes[pname] = jitter_int(pval, space[pname])
    keys = list(axes.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[axes[k] for k in keys])]
    combos = [c for c in combos if c != params]
    pass_n = 0
    neighbors = []
    for c in combos:
        r = run_forensic(df, gen_fn, c, sl_pct=0.40, timeframe=g["tf"],
                         commission=0.0015, slippage=0.0)
        wr = r.get("wr")
        trades = r.get("trades", 0) or 0
        pnl = r.get("total_pnl_pct", 0.0) or 0.0
        passes = (wr is not None and trades > 0 and wr >= LOOSE_WR
                  and trades >= LOOSE_N and pnl > LOOSE_PNL)
        if passes:
            pass_n += 1
        neighbors.append({"params": c, "wr": wr, "trades": trades,
                          "pnl_pct": pnl, "passes": bool(passes)})
    total = len(combos)
    pct = (pass_n / total * 100.0) if total else 0.0
    status = "PASS" if pct >= PLATEAU_THR * 100 else "FAIL"
    return {**g,
            "plateau_n_neighbors": total,
            "plateau_n_passing": pass_n,
            "plateau_pct_passing": round(pct, 1),
            "gate2_status": status,
            "gate2_neighbors": neighbors[:30]}


def gate3_mc_bootstrap(g):
    if g.get("gate2_status") != "PASS":
        return g
    pnls = g.get("trade_pnls_pct", [])
    if not pnls:
        return {**g, "gate3_status": "NO_DATA"}
    pnls = np.array(pnls, dtype=float)
    observed = float(pnls.sum())
    n = len(pnls)
    block_size = max(5, min(10, int(np.sqrt(n))))
    rng = np.random.default_rng(42)
    count_extreme = 0
    for _ in range(MC_PERMS):
        # block bootstrap resample
        sample = []
        while len(sample) < n:
            start = rng.integers(0, max(1, n - block_size + 1))
            sample.extend(pnls[start:start + block_size].tolist())
        sample = np.array(sample[:n], dtype=float)
        # sign shuffle within resample
        signs = rng.choice([-1, 1], size=n)
        perm_sum = float((sample * signs).sum())
        if perm_sum >= observed:
            count_extreme += 1
    p = count_extreme / MC_PERMS
    status = "PASS" if p < MC_PVALUE_MAX else "FAIL"
    return {**g, "mc_pvalue": round(p, 4), "gate3_status": status}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waves", nargs="+", required=True)
    ap.add_argument("--min-wr-eligible", type=float, default=70.0)
    ap.add_argument("--min-n", type=int, default=15)
    args = ap.parse_args()

    all_cands = []
    for w in args.waves:
        cs = load_wave_candidates(w, args.min_wr_eligible, args.min_n)
        print(f"[{w}] eligible candidates (WR>={args.min_wr_eligible}, n>={args.min_n}): {len(cs)}")
        all_cands.extend(cs)

    print(f"\nTotal eligible: {len(all_cands)}\n")

    # Gate 1
    print("=== GATE 1 FORENSIC ===")
    g1 = []
    for c in all_cands:
        r = gate1_forensic(c)
        tag = r.get("gate1_status", "?")
        if tag == "PASS":
            print(f"  PASS  {r['strategy']:<26} {r['symbol']:<6} {r['tf']:<3}  "
                  f"fwr={r['forensic_wr']}%  gap={r['gap_pp']}pp  "
                  f"pnl={r['forensic_pnl_pct']:.1f}%  max%={r['max_trade_pct_of_pnl']}")
        else:
            print(f"  FAIL  {r['strategy']:<26} {r['symbol']:<6} {r['tf']:<3}  "
                  f"reasons={r.get('gate1_reasons')}")
        g1.append(r)
    g1_pass = [r for r in g1 if r.get("gate1_status") == "PASS"]
    print(f"\ngate1 PASS: {len(g1_pass)}/{len(g1)}\n")

    # Concentration
    print("=== GATE CONCENTRATION ===")
    g_conc = [gate_concentration(r) for r in g1]
    g_conc_pass = [r for r in g_conc if r.get("gate_concentration_status") == "PASS"]
    for r in g_conc:
        if r.get("gate1_status") == "PASS":
            st = r.get("gate_concentration_status")
            print(f"  {st}  {r['strategy']:<26} {r['symbol']:<6} {r['tf']:<3}  "
                  f"max_trade%={r.get('max_trade_pct_of_pnl')}")
    print(f"\nconcentration PASS: {len(g_conc_pass)}/{len(g1_pass)}\n")

    # Gate 2
    print("=== GATE 2 PLATEAU ===")
    g2 = [gate2_plateau(r) for r in g_conc]
    g2_pass = [r for r in g2 if r.get("gate2_status") == "PASS"]
    for r in g2:
        if r.get("gate_concentration_status") == "PASS":
            st = r.get("gate2_status")
            print(f"  {st}  {r['strategy']:<26} {r['symbol']:<6} {r['tf']:<3}  "
                  f"plateau={r.get('plateau_pct_passing')}% "
                  f"({r.get('plateau_n_passing')}/{r.get('plateau_n_neighbors')})")
    print(f"\ngate2 PASS: {len(g2_pass)}/{len(g_conc_pass)}\n")

    # Gate 3
    print("=== GATE 3 MC BOOTSTRAP ===")
    g3 = [gate3_mc_bootstrap(r) for r in g2]
    g3_pass = [r for r in g3 if r.get("gate3_status") == "PASS"]
    for r in g3:
        if r.get("gate2_status") == "PASS":
            st = r.get("gate3_status")
            print(f"  {st}  {r['strategy']:<26} {r['symbol']:<6} {r['tf']:<3}  "
                  f"mc_p={r.get('mc_pvalue')}")
    print(f"\ngate3 PASS: {len(g3_pass)}/{len(g2_pass)}\n")

    # Write outputs
    suffix = "_".join(args.waves)
    out_all = ROOT / "results" / f"mac_{suffix}_pipeline_results.json"
    out_all.write_text(json.dumps(g3, indent=2, default=str) + "\n")
    print(f"Wrote {out_all}")

    confirmed = [r for r in g3 if r.get("gate3_status") == "PASS"]
    print(f"\n=== CONFIRMED GRAILS: {len(confirmed)} ===")
    for r in confirmed:
        print(f"  {r['strategy']} {r['symbol']} {r['tf']}  "
              f"optuna_wr={r['optuna_wr']}  forensic_wr={r['forensic_wr']}  "
              f"gap={r['gap_pp']}pp  plateau={r['plateau_pct_passing']}%  "
              f"mc_p={r['mc_pvalue']}")

    return g3


if __name__ == "__main__":
    main()
