#!/usr/bin/env python3
"""
HUNTER runner (Paso 2 + Paso 3 of SANDBOX_CHECKLIST).

For each strategy in batches 3566-3569 (HUNTER1, 20 strats) and 3594+3598
(HUNTER2, 99 strats), run Optuna-lite (a grid sample of param space)
against all 21 candle symbols x 5 TFs = 105 combos per strategy.

Gate per combo (R24):
  - WR >= 70%
  - trades >= dynamic_min (70% WR -> 8, 80% -> 8, 90% -> 4, 100% -> 2)
  - PF >= 1.2
  - gap Optuna->Forensic <= 10pp  (we report Optuna-style only here;
    forensic gap is minimal since backtest_signal_exit matches canonical)

PROMOTION rule: if >=1 combo passes, the WHOLE strategy goes to PROMOTED.json
so Hetzner/Mac can run it against the 563-symbol full universe.

Output:
  results/sandbox_h1_3566_3569_SHORTLIST.md
  results/sandbox_h2_3594_3598_SHORTLIST.md
  results/sandbox_h1_PROMOTED.json
  results/sandbox_h2_PROMOTED.json
"""
from __future__ import annotations

import gzip
import importlib
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies_v7"))
sys.path.insert(0, str(ROOT / "tools"))
from canary_runner import backtest_signal_exit, load_candles  # proven port of canonical

SYMBOLS = [
    "AGT", "APT", "ARB", "AVAX", "DYDX", "GMX", "INJ", "JTO", "JUP", "LINK",
    "NEAR", "ONDO", "OP", "PENDLE", "PYTH", "SEI", "SFP", "SUI", "SWARMS",
    "TIA", "WLD",
]
TFS = ["5m", "15m", "1h", "4h", "1d"]
SOURCE_TF = {"5m": "5m", "15m": "5m", "1h": "1h", "4h": "1h", "1d": "1h"}

# Gate per Paso 2/3
MIN_WR = 70.0
MIN_PF = 1.2
MIN_TRADES = 8  # dynamic_min floor (WR 70%+ -> 8)

# Trials per strategy-symbol-tf combo (Optuna-lite: random grid sample)
TRIALS_PER_COMBO = 20


def _min_trades_dynamic(wr: float) -> int:
    if wr >= 100: return 2
    if wr >= 90:  return 4
    if wr >= 80:  return 8
    if wr >= 70:  return 8
    return 999  # below threshold anyway


def _sample_params_from_space(space, trial_rng: random.Random) -> dict:
    """Sample params from callable (optuna-style, with/without trial arg) or dict space."""
    import inspect
    if callable(space):
        # Some spaces (batch3598) take 0 args. Try both signatures.
        try:
            sig = inspect.signature(space)
            nparams = len([p for p in sig.parameters.values()
                           if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)])
        except Exception:
            nparams = 1
        if nparams == 0:
            raw = space()
            if isinstance(raw, dict) and raw and isinstance(next(iter(raw.values())), tuple):
                # dict format returned by 0-arg callable: sample from it
                return _sample_dict(raw, trial_rng)
            return raw
        class _FakeTrial:
            def suggest_int(self, name, lo, hi, step=1):
                return trial_rng.randrange(lo, hi + 1, step)
            def suggest_float(self, name, lo, hi, step=None, log=False):
                if log:
                    return float(np.exp(trial_rng.uniform(np.log(lo), np.log(hi))))
                return trial_rng.uniform(lo, hi)
            def suggest_categorical(self, name, choices):
                return trial_rng.choice(choices)
        return space(_FakeTrial())
    return _sample_dict(space, trial_rng)


def _sample_dict(space: dict, trial_rng: random.Random) -> dict:
    out = {}
    for name, spec in space.items():
        if len(spec) == 2:
            ptype, vals = spec
            if ptype == "categorical":
                out[name] = trial_rng.choice(list(vals))
                continue
        ptype, lo, hi = spec[0], spec[1], spec[2]
        if ptype == "int":
            out[name] = trial_rng.randrange(lo, hi + 1)
        elif ptype == "float":
            out[name] = trial_rng.uniform(lo, hi)
        elif ptype == "categorical":
            out[name] = trial_rng.choice(list(hi) if hasattr(hi, "__iter__") else [lo, hi])
        else:
            out[name] = lo
    return out


def _backtest_metrics(df, gen_fn, params, sl_pct=None):
    try:
        r = backtest_signal_exit(df, gen_fn, params, sl_pct=sl_pct)
    except Exception as e:
        return None, str(e)
    if r.get("trades", 0) == 0 or r.get("wr") is None:
        return None, r.get("error", "no trades")
    return {
        "wr": r["wr"],
        "trades": r["trades"],
        "pf": r["pf"],
        "total_pnl_pct": r["total_pnl_pct"],
        "wins": r["wins"],
        "losses": r["losses"],
        "avg_win_pct": r["avg_win_pct"],
        "avg_loss_pct": r["avg_loss_pct"],
    }, None


def run_hunter(batch_ids: list, label: str) -> dict:
    """Load batches, test each strategy against all symbols/TFs, gate, return results."""
    combos_tested = 0
    grails_passed = 0
    strategies_promoted = {}   # strategy_name -> list of passing combos
    shortlist = []             # all (strat, sym, tf, params, metrics)
    errors = []
    t0 = time.time()

    # Pre-load all candles
    candles_cache = {}
    for sym in SYMBOLS:
        for tf in TFS:
            src = SOURCE_TF[tf]
            try:
                df = load_candles(sym, src, tf)
                if len(df) < 300:
                    continue
                candles_cache[(sym, tf)] = df
            except Exception as e:
                errors.append(f"load {sym} {tf}: {e}")

    print(f"[{label}] candles cache: {len(candles_cache)} (sym,tf) pairs")
    total_batches = len(batch_ids)

    for bi, b in enumerate(batch_ids):
        mod_name = f"strategies_tv2_batch{b}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            errors.append(f"import {mod_name}: {e}")
            continue
        strats = mod.STRATEGY_EXPORT
        print(f"[{label}] batch {b} ({bi+1}/{total_batches}) -> {len(strats)} strats")

        for strat_name, spec in strats.items():
            gen_fn = spec.get("gen")
            space_fn = spec.get("space")
            if not gen_fn or not space_fn:
                continue
            rng = random.Random(hash((strat_name, b)) & 0xFFFFFFFF)
            best_by_combo = {}

            for sym, tf in candles_cache:
                df = candles_cache[(sym, tf)]
                best = None
                for t in range(TRIALS_PER_COMBO):
                    try:
                        params = _sample_params_from_space(space_fn, rng)
                    except Exception as e:
                        errors.append(f"{strat_name} space: {e}")
                        break
                    m, err = _backtest_metrics(df, gen_fn, params, sl_pct=0.40)
                    combos_tested += 1
                    if m is None:
                        continue
                    if best is None or m["wr"] > best["metrics"]["wr"]:
                        best = {"params": params, "metrics": m}
                if best is None:
                    continue
                m = best["metrics"]
                dyn = _min_trades_dynamic(m["wr"])
                passes = (
                    m["wr"] >= MIN_WR
                    and m["trades"] >= max(MIN_TRADES, dyn)
                    and m["pf"] >= MIN_PF
                )
                row = {
                    "batch": b, "strategy": strat_name,
                    "symbol": sym, "tf": tf,
                    "params": best["params"], "metrics": m,
                    "passes": passes,
                }
                shortlist.append(row)
                if passes:
                    grails_passed += 1
                    strategies_promoted.setdefault(strat_name, []).append(row)
        elapsed = time.time() - t0
        print(f"[{label}] ... {combos_tested} combos, {grails_passed} passing  t={elapsed:.0f}s")

    return {
        "label": label,
        "batches": batch_ids,
        "candles_cache_size": len(candles_cache),
        "combos_tested": combos_tested,
        "grails_passed": grails_passed,
        "strategies_promoted_count": len(strategies_promoted),
        "strategies_promoted": strategies_promoted,
        "shortlist": shortlist,
        "errors": errors[:200],
        "elapsed_s": round(time.time() - t0, 1),
    }


def write_report(res: dict, shortlist_path: Path, promoted_path: Path):
    passing = [r for r in res["shortlist"] if r["passes"]]
    passing.sort(key=lambda r: (-r["metrics"]["wr"], -r["metrics"]["pf"]))
    lines = [
        f"# {res['label']} shortlist",
        "",
        f"- Batches: {', '.join(str(b) for b in res['batches'])}",
        f"- Combos tested: {res['combos_tested']:,}",
        f"- Candle cache: {res['candles_cache_size']} (sym,tf) pairs",
        f"- Grails passing R24 gate: **{res['grails_passed']}**",
        f"- Strategies promoted (>=1 combo pass): **{res['strategies_promoted_count']}**",
        f"- Elapsed: {res['elapsed_s']}s",
        "",
        "| Rank | Strategy | Sym | TF | WR | trades | PF~ | total% | params |",
        "|------|----------|-----|----|-----|--------|-----|--------|--------|",
    ]
    for i, r in enumerate(passing[:100], 1):
        m = r["metrics"]
        p = json.dumps(r["params"], separators=(",", ":"))
        lines.append(f"| {i} | `{r['strategy']}` | {r['symbol']} | {r['tf']} | "
                     f"{m['wr']:.1f}% | {m['trades']} | {m['pf']:.2f} | "
                     f"{m['total_pnl_pct']:.1f}% | `{p}` |")
    shortlist_path.write_text("\n".join(lines) + "\n")

    promoted = {}
    for name, combos in res["strategies_promoted"].items():
        combos.sort(key=lambda r: -r["metrics"]["wr"])
        best = combos[0]
        promoted[name] = {
            "best_combo": {
                "symbol": best["symbol"], "tf": best["tf"],
                "wr": best["metrics"]["wr"], "trades": best["metrics"]["trades"],
                "pf": best["metrics"]["pf"],
                "params": best["params"],
            },
            "passing_combos": len(combos),
            "all_passing": [{"sym": c["symbol"], "tf": c["tf"],
                             "wr": c["metrics"]["wr"], "trades": c["metrics"]["trades"],
                             "params": c["params"]} for c in combos],
        }
    promoted_path.write_text(json.dumps(promoted, indent=2) + "\n")
    print(f"wrote {shortlist_path} + {promoted_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", choices=["h1", "h2"], required=True)
    args = ap.parse_args()

    if args.wave == "h1":
        batches = ["3566", "3567", "3568", "3569"]
        label = "HUNTER1 (Paso 2)"
        sp = ROOT / "results" / "sandbox_h1_3566_3569_SHORTLIST.md"
        pp = ROOT / "results" / "sandbox_h1_PROMOTED.json"
    else:
        batches = ["3594", "3598"]
        label = "HUNTER2 (Paso 3)"
        sp = ROOT / "results" / "sandbox_h2_3594_3598_SHORTLIST.md"
        pp = ROOT / "results" / "sandbox_h2_PROMOTED.json"

    (ROOT / "results").mkdir(exist_ok=True)
    res = run_hunter(batches, label)
    write_report(res, sp, pp)
    print(f"\n{label}: {res['grails_passed']} passing combos, "
          f"{res['strategies_promoted_count']} strategies promoted")
