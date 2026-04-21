#!/usr/bin/env python3
"""HIGH-2: PBO + DSR per-grail sobre los 19 B-tier con plateau.

Per-grail PBO (simplified): compute WR on first-half (h1) and second-half (h2)
of the price series. If best trial's test rel-rank < 0.5, flag overfit.
Single-grail PBO is binary; full PBO fórmula needs a trial pool, así que
aprovecho el plateau neighbor set (k=25) como el "pool de trials" implícito.

Per-grail DSR: Bernoulli payoff from (wr, pf, trades). N_trials = pool size.

Gate (Mac's):
  PBO <= 0.50 AND DSR > 0  ->  upgrade C/B -> A

Output: results/pbo_dsr_b_tier_19_20260421.json
"""
from __future__ import annotations
import json
import math
import os
import sys
from pathlib import Path
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "strategies_v7"))

from canary_runner import load_candles, backtest_signal_exit
import importlib

SRC_TF = {"5m": "5m", "15m": "5m", "1h": "1h", "4h": "1h", "1d": "1h"}


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


def bernoulli_moments(wr_pct, pf):
    w = float(np.clip(wr_pct / 100.0, 1e-6, 1 - 1e-6))
    p = float(np.clip(pf, 1e-6, None))
    b = p * (1 - w) / w
    mu = w * b - (1 - w)
    sigma = math.sqrt(w * (1 - w)) * (b + 1)
    skew = (1 - 2 * w) / math.sqrt(w * (1 - w))
    kurt_raw = (1.0 / (w * (1 - w)) - 6.0) + 3.0
    return mu, sigma, skew, kurt_raw


def sharpe_proxy(wr_pct, pf, trades):
    mu, sigma, _, _ = bernoulli_moments(wr_pct, pf)
    sr_t = mu / sigma if sigma > 0 else 0.0
    return sr_t * math.sqrt(max(trades, 1))


def dsr_score(sr, trades, skew, kurt_raw, n_trials):
    """DSR per Bailey & LdP 2014."""
    if n_trials < 2:
        e_max = 0.0
    else:
        em = 0.5772156649015329
        z1 = norm.ppf(1 - 1.0 / n_trials)
        z2 = norm.ppf(1 - 1.0 / (n_trials * math.e))
        e_max = (1 - em) * z1 + em * z2
    denom = math.sqrt(max(1 - skew * sr + (kurt_raw - 1) / 4.0 * sr * sr, 1e-9))
    T = max(trades - 1, 1)
    z = (sr - e_max) * math.sqrt(T) / denom
    # DSR as z-value. Mac's gate is DSR > 0 (positive z), so we return z directly.
    return z, e_max


def pbo_halves(gen_fn, params, df):
    """Split series in half, compute WR on each, return (wr1, wr2, n1, n2, overfit)."""
    n = len(df)
    mid = n // 2
    df1, df2 = df.iloc[:mid].reset_index(drop=True), df.iloc[mid:].reset_index(drop=True)
    r1 = backtest_signal_exit(df1, gen_fn, params, sl_pct=0.40)
    r2 = backtest_signal_exit(df2, gen_fn, params, sl_pct=0.40)
    wr1 = r1.get("wr") if r1.get("trades", 0) >= 5 else None
    wr2 = r2.get("wr") if r2.get("trades", 0) >= 5 else None
    n1 = r1.get("trades", 0); n2 = r2.get("trades", 0)
    # Overfit if big WR gap between halves (>= 20pp)
    if wr1 is None or wr2 is None:
        overfit = None  # insufficient data
    else:
        gap = abs(wr1 - wr2)
        overfit = 1.0 if gap >= 20 else 0.0
    return {"wr1": wr1, "wr2": wr2, "n1": n1, "n2": n2,
            "pbo_per_grail": overfit, "gap_wr": abs(wr1-wr2) if (wr1 is not None and wr2 is not None) else None}


def main():
    v8 = json.loads((ROOT / "coordination" / "SANDBOX_FINAL_MASTER_V8_GRAILS.json").read_text())
    # B-tier with plateau
    b_tier = [g for g in v8["grails"] if g.get("tier") == "B" and g.get("plateau_confirmed")]
    pool = json.loads((ROOT / "results" / "sandbox_master_ranking.json").read_text())
    n_trials_pool = len(pool)
    print(f"Targets B-tier plateau: {len(b_tier)} | pool size N_trials={n_trials_pool}")

    results = []
    upgrades = 0
    for i, g in enumerate(b_tier, 1):
        row = {"strategy": g["strategy"], "symbol": g["symbol"], "tf": g["tf"],
               "stored_wr": g["wr"], "stored_trades": g["trades"], "stored_pf": g["pf"]}
        spec = find_strategy_spec(g["strategy"])
        if spec is None:
            row["error"] = "strategy not found"; results.append(row); continue
        entry = find_params(g["strategy"], g["symbol"], g["tf"])
        if entry is None:
            row["error"] = "params not found"; results.append(row); continue
        params = entry["params"]
        row["params"] = params
        try:
            df = load_candles(g["symbol"], SRC_TF[g["tf"]], g["tf"])
        except Exception as e:
            row["error"] = f"load_candles: {str(e)[:60]}"; results.append(row); continue

        # PBO per-grail via halves
        try:
            pbo_info = pbo_halves(spec["gen"], params, df)
        except Exception as e:
            row["error"] = f"pbo: {str(e)[:60]}"; results.append(row); continue
        row["pbo"] = pbo_info

        # DSR
        sr = sharpe_proxy(g["wr"], g["pf"], g["trades"])
        _, _, skew, kurt = bernoulli_moments(g["wr"], g["pf"])
        dsr_z, e_max = dsr_score(sr, g["trades"], skew, kurt, n_trials_pool)
        row["sharpe_proxy"] = sr
        row["dsr_z"] = dsr_z
        row["e_max_sr"] = e_max
        row["bernoulli_skew"] = skew
        row["bernoulli_kurt_raw"] = kurt

        # Gate: PBO <= 0.50 AND DSR > 0
        pbo_val = pbo_info.get("pbo_per_grail")
        gate_pass = True
        reasons = []
        if pbo_val is None:
            gate_pass = False; reasons.append("pbo_halves insufficient data")
        elif pbo_val > 0.50:
            gate_pass = False; reasons.append(f"pbo={pbo_val:.2f}>0.50 (gap_wr={pbo_info['gap_wr']:.1f}pp)")
        if dsr_z <= 0:
            gate_pass = False; reasons.append(f"dsr_z={dsr_z:.3f}<=0")

        row["upgrade_to_A"] = gate_pass
        row["reject_reasons"] = reasons if not gate_pass else []
        if gate_pass:
            upgrades += 1
            print(f"  [{i}/{len(b_tier)}] UPGRADE→A  {g['strategy']} {g['symbol']} {g['tf']}  "
                  f"pbo={pbo_val} dsr_z={dsr_z:.2f}  gap_wr={pbo_info.get('gap_wr'):.1f}pp")
        else:
            print(f"  [{i}/{len(b_tier)}] hold B     {g['strategy']:28s} {g['symbol']:6s} {g['tf']:3s}  "
                  f"[{', '.join(reasons[:2])}]")
        results.append(row)

    out = ROOT / "results" / "pbo_dsr_b_tier_19_20260421.json"
    out.write_text(json.dumps({
        "_doc": "HIGH-2 PBO halves + DSR Bernoulli gate sobre B-tier plateau",
        "_generated_at": "2026-04-21",
        "_n_targets": len(b_tier),
        "_n_upgrades_to_A": upgrades,
        "_n_pool_trials": n_trials_pool,
        "_gate": "PBO per-grail (WR gap between halves < 20pp) AND DSR z > 0",
        "results": results,
    }, indent=2, default=str))
    print(f"\nDONE: {upgrades}/{len(b_tier)} upgrades to A | wrote {out.name}")


if __name__ == "__main__":
    main()
