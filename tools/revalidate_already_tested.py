#!/usr/bin/env python3
"""ETAPA 7: Re-validate 38 already_tested grails on sandbox candles.

For each grail in coordination/already_tested_grails.json:
  1. Map symbol to sandbox candle (strip /USDT:USDT suffix)
  2. Load candles
  3. Run backtest_signal_exit with exact params + SL from grail
  4. Compare WR/trades observed vs expected
  5. Flag gap > 3pp as "DIVERGED" (worth Mac review)

Output: results/sandbox_already_tested_antioverfit.md
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "mac_optuna"))
sys.path.insert(0, str(ROOT / "strategies_v7"))
from canary_runner import backtest_signal_exit, load_candles

CANDLES = ROOT / "data" / "candles"
SRC_TF = {"5m": "5m", "15m": "5m", "1h": "1h", "4h": "1h", "1d": "1h"}


def resolve_strategy(name):
    """Find gen function across mac_optuna + strategies_v7."""
    # Map known strategies to their impl
    from canary_runner import get_strategy_fn
    g = get_strategy_fn(name)
    if g:
        return g
    # Try strategy_factory
    try:
        import strategy_factory
        STRAT = getattr(strategy_factory, "FACTORY_STRATS", None) or \
                getattr(strategy_factory, "STRATEGIES", {})
        if name in STRAT:
            spec = STRAT[name]
            return spec[0] if isinstance(spec, tuple) else spec.get("gen")
    except Exception:
        pass
    # Try grail_dual_validator registry
    try:
        import grail_dual_validator as gdv
        if hasattr(gdv, "_STRAT_REGISTRY") and name in gdv._STRAT_REGISTRY:
            return gdv._STRAT_REGISTRY[name]
    except Exception:
        pass
    # Try scanning all tv2_batch* files
    import importlib, os
    for f in sorted(os.listdir(ROOT / "strategies_v7")):
        if not f.endswith(".py") or f.startswith("__"): continue
        try:
            mod = importlib.import_module(f[:-3])
            if hasattr(mod, "STRATEGY_EXPORT") and name in mod.STRATEGY_EXPORT:
                return mod.STRATEGY_EXPORT[name]["gen"]
        except Exception:
            continue
    return None


def main():
    path = ROOT / "coordination" / "already_tested_grails.json"
    data = json.load(open(path))
    out = []
    missing_impl = []
    missing_data = []

    for g in data:
        strat = g.get("strategy")
        sym_raw = g.get("symbol", "")
        # strip /USDT:USDT -> raw sym  (e.g., "SFP/USDT:USDT" -> "SFP")
        sym = sym_raw.split("/")[0]
        tf = g.get("timeframe")
        params = g.get("best_params", {})
        sl = g.get("sl", 0.4)
        exp_wr = g.get("full_wr")
        exp_tr = g.get("full_trades")

        gen_fn = resolve_strategy(strat)
        if gen_fn is None:
            missing_impl.append(strat)
            out.append({**g, "observed_wr": None, "gap_pp": None,
                        "status": "IMPL_MISSING"})
            continue
        try:
            df = load_candles(sym, SRC_TF[tf], tf)
        except Exception as e:
            missing_data.append(f"{sym}_{tf}")
            out.append({**g, "observed_wr": None, "gap_pp": None,
                        "status": f"DATA_MISSING: {sym}_{tf}"})
            continue
        try:
            r = backtest_signal_exit(df, gen_fn, params, sl_pct=sl)
        except Exception as e:
            out.append({**g, "observed_wr": None, "gap_pp": None,
                        "status": f"RUN_FAIL: {str(e)[:60]}"})
            continue
        wr_obs = r.get("wr")
        if wr_obs is None:
            out.append({**g, "observed_wr": None, "gap_pp": None,
                        "status": "NO_TRADES"})
            continue
        gap = wr_obs - exp_wr
        status = "MATCH" if abs(gap) <= 3.0 else "DIVERGED" if abs(gap) <= 10 else "WRONG_IMPL"
        out.append({**g, "observed_wr": wr_obs, "observed_trades": r.get("trades"),
                    "gap_pp": round(gap, 1), "status": status})

    # Summary
    by_status = {}
    for o in out:
        by_status[o["status"]] = by_status.get(o["status"], 0) + 1
    print(f"38 grails revalidated, status: {by_status}")

    md = [
        "# Re-validation of 38 already_tested grails",
        "",
        f"- Total: {len(out)}",
        f"- Status breakdown: {by_status}",
        f"- Missing impl: {len(missing_impl)} ({missing_impl[:5]}...)" if missing_impl else "",
        f"- Missing data: {len(set(missing_data))}" if missing_data else "",
        "",
        "| Strategy | Symbol | TF | Expected WR | Observed WR | Gap | Status |",
        "|----------|--------|----|-------------|-------------|-----|--------|",
    ]
    for o in sorted(out, key=lambda x: abs(x.get("gap_pp") or 999)):
        wr_e = o.get("full_wr", "N/A")
        wr_o = o.get("observed_wr", "N/A")
        gap = o.get("gap_pp", "N/A")
        md.append(f"| `{o['strategy']}` | {o['symbol']} | {o['timeframe']} | "
                  f"{wr_e}% | {wr_o if wr_o else 'N/A'}% | {gap}pp | {o['status']} |")
    (ROOT / "results" / "sandbox_already_tested_antioverfit.md").write_text(
        "\n".join(l for l in md if l) + "\n")
    (ROOT / "results" / "sandbox_already_tested_antioverfit.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"Wrote results/sandbox_already_tested_antioverfit.md/json")


if __name__ == "__main__":
    main()
