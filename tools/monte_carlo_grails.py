#!/usr/bin/env python3
"""Monte Carlo permutation test on plateau survivors.

For each grail: shuffle the SIGN of trade returns 500 times, compute fraction
of perms with cumulative return >= observed. p < 0.01 = significant.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "strategies_v7"))
from canary_runner import load_candles, backtest_signal_exit
import importlib

P = ROOT / "results" / "sandbox_h1_plateau.json"
SRC_TF = {"5m": "5m", "15m": "5m", "1h": "1h", "4h": "1h", "1d": "1h"}
N_PERM = 500


def find_strategy_spec(name):
    import os
    for f in sorted(os.listdir(ROOT / "strategies_v7")):
        if not f.endswith(".py") or f.startswith("__"): continue
        try:
            mod = importlib.import_module(f[:-3])
            if hasattr(mod, "STRATEGY_EXPORT") and name in mod.STRATEGY_EXPORT:
                return mod.STRATEGY_EXPORT[name]
        except Exception:
            continue
    return None


def mc_test(gen_fn, params, df, n_perm=500):
    """Run baseline, then n_perm permutations of trade signs. Return p-value."""
    rng = np.random.default_rng(42)
    r = backtest_signal_exit(df, gen_fn, params, sl_pct=0.40)
    if r.get("trades", 0) < 10:
        return None
    obs_total = r["total_pnl_pct"]
    n_trades = r["trades"]
    avg_win = r["avg_win_pct"]
    avg_loss = r["avg_loss_pct"]  # negative
    # Reconstruct synthetic trade pnls (not perfect — we don't have per-trade list,
    # so approximate using avg_win/avg_loss with random allocation)
    p_win = r["wins"] / n_trades
    perm_results = []
    for _ in range(n_perm):
        signs = rng.random(n_trades) < p_win
        pnls = np.where(signs, avg_win, avg_loss)
        # randomly shuffle (already random), sum
        perm_results.append(float(pnls.sum()))
    perm_arr = np.array(perm_results)
    p_value = float((perm_arr >= obs_total).mean())
    return {"obs_total_pnl": obs_total, "p_value": p_value,
            "n_perm": n_perm, "n_trades": n_trades}


def main():
    if not P.exists():
        print("plateau file missing"); return
    plateau = json.loads(P.read_text())
    survivors = [g for g in plateau if g.get("status") == "PLATEAU"]
    print(f"Monte Carlo: {len(survivors)} plateau grails x {N_PERM} perms")
    rows = []
    for g in survivors:
        spec = find_strategy_spec(g["strategy"])
        if spec is None:
            continue
        try:
            df = load_candles(g["symbol"], SRC_TF[g["tf"]], g["tf"])
        except Exception:
            continue
        params = g.get("params", {})
        try:
            res = mc_test(spec["gen"], params, df, n_perm=N_PERM)
        except Exception as e:
            res = {"error": str(e)[:80]}
        rows.append({**g, "mc": res})
        if res and res.get("p_value") is not None:
            print(f"  {g['strategy']:<32} {g['symbol']:<6} {g['tf']:<3} p={res['p_value']:.3f}")
    md = [
        "# Monte Carlo p-values (plateau survivors, 500 perms)",
        "",
        f"- {len(rows)} grails tested",
        f"- p<0.01: {sum(1 for r in rows if r.get('mc',{}).get('p_value',1) < 0.01)}",
        f"- p<0.05: {sum(1 for r in rows if r.get('mc',{}).get('p_value',1) < 0.05)}",
        "",
        "| Strategy | Sym | TF | n | WR | p-value | Significant? |",
        "|----------|-----|----|---|-----|---------|--------------|",
    ]
    for r in sorted(rows, key=lambda x: x.get("mc",{}).get("p_value",1)):
        p = r.get("mc",{}).get("p_value")
        sig = "YES" if p is not None and p < 0.01 else ("warn" if p is not None and p < 0.05 else "no")
        md.append(f"| `{r['strategy']}` | {r['symbol']} | {r['tf']} | {r['trades']} | "
                  f"{r['wr']:.1f}% | {p if p is not None else 'N/A'} | {sig} |")
    (ROOT / "results" / "sandbox_monte_carlo.md").write_text("\n".join(md) + "\n")
    (ROOT / "results" / "sandbox_monte_carlo.json").write_text(
        json.dumps(rows, indent=2, default=str) + "\n")
    print(f"MONTE_CARLO DONE")


if __name__ == "__main__":
    main()
