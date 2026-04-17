#!/usr/bin/env python3
"""
Monte Carlo permutation test (Aronson, "Evidence-Based Technical Analysis").

For each top-5 plateau grail:
  1. Build the real entry boolean series via the strategy's signal builder.
  2. Run the same simulator -> observed total return.
  3. Generate N=500 random permutations of entry positions (preserve total
     entry count; shuffle which timestamps fire). Exit signal is rebuilt by
     the strategy (deterministic from price), but exits via SL/TP/trail/
     timeout still apply on each shuffled entry.
  4. Re-simulate -> null distribution of returns.
  5. p-value = (1 + #{null >= observed}) / (1 + N).
  6. Significant if p < 0.01.

Output: results/monte_carlo_report.md
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grail_loop import SPACES  # noqa: E402
from backtest import simulate, metrics_from_trades, load_csv  # noqa: E402

DATA = ROOT / "data" / "ETHUSD_1h.csv"
OUT = ROOT / "results" / "monte_carlo_report.md"
N_PERM = 500
N_PROCS = 4
FEE = 0.0005
SLIPPAGE = 0.0002
ASSET = "ETHUSD"
TF = "1h"

# Top-5 plateau grails (rsi2_regime ETHUSD 1h) from results/plateau_report.md
GRAILS = [
    {"rank": 1, "strategy": "rsi2_regime",
     "params": {"rsi_len": 2, "rsi_buy": 3, "sma_trend": 100,
                "slope_bars": 5, "exit_sma": 8},
     "exit": {"sl_atr": 3.5, "tp_atr": 4.0, "trail_atr": 5.0}},
    {"rank": 2, "strategy": "rsi2_regime",
     "params": {"rsi_len": 5, "rsi_buy": 18, "sma_trend": 125,
                "slope_bars": 80, "exit_sma": 13},
     "exit": {"sl_atr": 5.0, "tp_atr": 7.0, "timeout": 48}},
    {"rank": 3, "strategy": "rsi2_regime",
     "params": {"rsi_len": 5, "rsi_buy": 20, "sma_trend": 75,
                "slope_bars": 80, "exit_sma": 8},
     "exit": {"sl_atr": 5.0, "timeout": 96}},
    {"rank": 4, "strategy": "rsi2_regime",
     "params": {"rsi_len": 2, "rsi_buy": 7, "sma_trend": 75,
                "slope_bars": 100, "exit_sma": 13},
     "exit": {"sl_atr": 4.0, "tp_atr": 10.0, "timeout": 36}},
    {"rank": 5, "strategy": "rsi2_regime",
     "params": {"rsi_len": 5, "rsi_buy": 20, "sma_trend": 75,
                "slope_bars": 5, "exit_sma": 8},
     "exit": {"sl_atr": 3.0, "tp_atr": 4.0, "timeout": 18}},
]


def build_signals(df, fam, params):
    return SPACES[fam]["sig"](df, params)


def run_sim(df, ent, ex, exit_cfg):
    trades = simulate(df, ent, ex, fee=FEE, slippage=SLIPPAGE,
                      sl_atr=exit_cfg.get("sl_atr"),
                      tp_atr=exit_cfg.get("tp_atr"),
                      trail_atr=exit_cfg.get("trail_atr"),
                      timeout=exit_cfg.get("timeout"))
    m = metrics_from_trades("x", ASSET, TF, trades, df)
    return m.total_return_pct, m.trades, m.wr, m.profit_factor


# Globals filled by pool initializer (avoids pickling df per task).
_DF = None
_EX = None
_EXIT_CFG = None
_VALID_IDX = None
_N_ENTRIES = None


def _init_worker(df, ex, exit_cfg, valid_idx, n_entries):
    global _DF, _EX, _EXIT_CFG, _VALID_IDX, _N_ENTRIES
    _DF = df
    _EX = ex
    _EXIT_CFG = exit_cfg
    _VALID_IDX = valid_idx
    _N_ENTRIES = n_entries


def _perm_one(seed):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(_VALID_IDX, size=_N_ENTRIES, replace=False)
    arr = np.zeros(len(_DF), dtype=bool)
    arr[chosen] = True
    ent = pd.Series(arr, index=_DF.index)
    ret, *_ = run_sim(_DF, ent, _EX, _EXIT_CFG)
    return ret


def mc_test(df, grail):
    fam = grail["strategy"]
    params = grail["params"]
    exit_cfg = grail["exit"]
    ent, ex = build_signals(df, fam, params)
    obs_ret, obs_trades, obs_wr, obs_pf = run_sim(df, ent, ex, exit_cfg)

    # Skip warm-up region so indicators/ATR are defined before any entry.
    warmup = max(params.get("sma_trend", 0), params.get("slope_bars", 0),
                 params.get("rsi_len", 0), params.get("exit_sma", 0), 14)
    valid_idx = np.arange(warmup, len(df))
    n_entries = int(ent.values.sum())
    if n_entries == 0 or n_entries > len(valid_idx):
        return {"observed": obs_ret, "trades": obs_trades, "wr": obs_wr,
                "pf": obs_pf, "p_value": float("nan"), "null": []}

    t0 = time.time()
    seeds = list(range(1, N_PERM + 1))
    with Pool(processes=N_PROCS,
              initializer=_init_worker,
              initargs=(df, ex, exit_cfg, valid_idx, n_entries)) as pool:
        null_rets = pool.map(_perm_one, seeds, chunksize=8)
    elapsed = time.time() - t0

    null = np.array(null_rets, dtype=float)
    ge = int(np.sum(null >= obs_ret))
    p = (1 + ge) / (1 + N_PERM)
    return {"observed": obs_ret, "trades": obs_trades, "wr": obs_wr,
            "pf": obs_pf, "n_entries": n_entries, "p_value": p,
            "null_mean": float(null.mean()), "null_std": float(null.std()),
            "null_max": float(null.max()), "null_pct95": float(np.percentile(null, 95)),
            "elapsed_s": elapsed, "ge_count": ge}


def main():
    df = load_csv(DATA)
    print(f"Loaded {len(df)} bars from {DATA.name}")

    rows = []
    for g in GRAILS:
        print(f"\n[grail #{g['rank']}] {g['strategy']} params={g['params']}")
        r = mc_test(df, g)
        r["rank"] = g["rank"]
        r["params"] = g["params"]
        r["exit"] = g["exit"]
        rows.append(r)
        print(f"  observed_ret={r['observed']:.2f}%  n_entries={r.get('n_entries')}  "
              f"null_mean={r['null_mean']:.2f}%  null_max={r['null_max']:.2f}%  "
              f"p={r['p_value']:.4f}  ({r['elapsed_s']:.1f}s)")

    lines = [
        "# Monte Carlo permutation test (Aronson EBTA)",
        "",
        f"- N permutations: **{N_PERM}** per grail",
        f"- Asset/TF: **{ASSET} {TF}** (real OHLCV, {len(df)} bars)",
        f"- Procedure: shuffle entry-bar positions (preserve total entry count); "
        "rebuild exit signal deterministically from price; re-simulate.",
        "- p-value = (1 + #{null_ret >= observed_ret}) / (1 + N)",
        "- Significant if p < 0.01",
        "",
        "| Rank | Strategy | Trades | Entries | Obs Ret % | Null Mean % | Null 95th % | Null Max % | p-value | Significant |",
        "|------|----------|--------|---------|-----------|-------------|-------------|------------|---------|-------------|",
    ]
    for r in rows:
        sig = "YES" if r["p_value"] < 0.01 else "no"
        lines.append(
            f"| {r['rank']} | `rsi2_regime` | {r['trades']} | {r.get('n_entries','-')} | "
            f"{r['observed']:.2f} | {r['null_mean']:.2f} | {r['null_pct95']:.2f} | "
            f"{r['null_max']:.2f} | {r['p_value']:.4f} | {sig} |"
        )
    lines += ["", "## Configurations tested", ""]
    for r in rows:
        lines.append(
            f"- **#{r['rank']}** params=`{json.dumps(r['params'], separators=(',',':'))}` "
            f"exit=`{json.dumps(r['exit'], separators=(',',':'))}` -> p={r['p_value']:.4f}"
        )

    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT}")
    n_sig = sum(1 for r in rows if r["p_value"] < 0.01)
    print(f"{n_sig}/{len(rows)} grails significant at p<0.01")


if __name__ == "__main__":
    main()
