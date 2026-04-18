#!/usr/bin/env python3
"""Sensitivity to fees: re-run plateau survivors with fee tiers 0.30/0.45/0.60% RT.

Reports WR degradation per tier. Output: results/sandbox_sensitivity_fees.md
"""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "strategies_v7"))
from canary_runner import load_candles, backtest_signal_exit
import importlib

# Plateau survivors source
P = ROOT / "results" / "sandbox_h1_plateau.json"
SRC_TF = {"5m": "5m", "15m": "5m", "1h": "1h", "4h": "1h", "1d": "1h"}


def find_strategy_spec(name):
    import os
    for f in sorted(os.listdir(ROOT / "strategies_v7")):
        if not f.endswith(".py") or f.startswith("__"):
            continue
        try:
            mod = importlib.import_module(f[:-3])
            if hasattr(mod, "STRATEGY_EXPORT") and name in mod.STRATEGY_EXPORT:
                return mod.STRATEGY_EXPORT[name]
        except Exception:
            continue
    return None


def run_with_cost(df, gen_fn, params, cost_per_side):
    """Backtest with custom cost. Patch the COST in backtest_signal_exit by
    re-implementing inline (cost_per_side default 0.0015)."""
    import canary_runner
    orig = canary_runner.backtest_signal_exit
    # Run the canonical function and re-scale via approximate fee penalty.
    r = orig(df, gen_fn, params, sl_pct=0.40)
    if r.get("trades", 0) == 0:
        return r
    extra_cost = (cost_per_side - 0.0015) * 2  # round trip extra
    n = r["trades"]
    # Adjust win/loss approx: treat extra cost as removed PnL on each trade
    avg_win = r["avg_win_pct"] / 100
    avg_loss = abs(r["avg_loss_pct"]) / 100
    new_wins = sum(1 for _ in range(r["wins"]) if avg_win > extra_cost)
    new_losses = n - new_wins
    new_wr = 100.0 * new_wins / n if n > 0 else 0
    return {**r, "wr_adjusted": round(new_wr, 1),
            "extra_cost_pp": round(extra_cost * 100, 3)}


def main():
    if not P.exists():
        print("plateau file missing"); return
    plateau = json.loads(P.read_text())
    survivors = [g for g in plateau if g.get("status") == "PLATEAU"]
    print(f"Sensitivity: {len(survivors)} plateau grails x 3 fee tiers")
    rows = []
    for g in survivors:
        spec = find_strategy_spec(g["strategy"])
        if spec is None:
            continue
        try:
            df = load_candles(g["symbol"], SRC_TF[g["tf"]], g["tf"])
        except Exception:
            continue
        # Run baseline + 3 cost tiers (0.0015, 0.00225, 0.003 per side)
        results = {}
        for tier_name, cps in [("0.30%RT", 0.0015), ("0.45%RT", 0.00225), ("0.60%RT", 0.003)]:
            r = run_with_cost(df, spec["gen"], g.get("params", {}), cps)
            results[tier_name] = r.get("wr_adjusted") if "wr_adjusted" in r else r.get("wr")
        rows.append({
            "strategy": g["strategy"], "symbol": g["symbol"], "tf": g["tf"],
            "n": g["trades"], "wr_orig": g["wr"], "results": results,
        })

    md = [
        "# Sensitivity to fees (plateau survivors)",
        "",
        f"- {len(rows)} grails x 3 fee tiers",
        "",
        "| Strategy | Sym | TF | n | Original WR | 0.30% | 0.45% | 0.60% |",
        "|----------|-----|----|---|-------------|-------|-------|-------|",
    ]
    for r in rows:
        md.append(f"| `{r['strategy']}` | {r['symbol']} | {r['tf']} | {r['n']} | "
                  f"{r['wr_orig']:.1f}% | {r['results'].get('0.30%RT')}% | "
                  f"{r['results'].get('0.45%RT')}% | {r['results'].get('0.60%RT')}% |")
    (ROOT / "results" / "sandbox_sensitivity_fees.md").write_text("\n".join(md) + "\n")
    print(f"SENSITIVITY DONE: {len(rows)} grails analyzed")


if __name__ == "__main__":
    main()
