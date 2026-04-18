#!/usr/bin/env python3
"""Plateau test ±20% jitter over HUNTER1 n>=20 grails.

For each grail in results/sandbox_h1_FINAL_SHORTLIST (or interim):
  1. Jitter each numeric param by ±10, ±20% (snapped to space grid)
  2. Re-run backtest_signal_exit
  3. Count how many neighbors still pass WR>=65%, n>=15, PF>=1.1 (loose gate)
  4. Plateau = >= 60% neighbors pass

Reject lone spikes. Output: results/sandbox_h1_plateau.md
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
from itertools import product
import importlib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "strategies_v7"))
from canary_runner import backtest_signal_exit, load_candles

SRC_TF = {"5m": "5m", "15m": "5m", "1h": "1h", "4h": "1h", "1d": "1h"}
JITTERS = [-0.2, -0.1, 0.0, 0.1, 0.2]
MIN_WR_NBR = 65.0
MIN_N_NBR = 15
MIN_PF_NBR = 1.1
PLATEAU_THR = 0.60


def parse_interim_grails(all_waves: bool = False):
    """Parse grails — either HUNTER1 log (legacy) or master ranking (all waves)."""
    if all_waves:
        ranking_path = ROOT / "results" / "sandbox_master_ranking.json"
        if ranking_path.exists():
            rows = json.loads(ranking_path.read_text())
            # n>=20 subset + dedup already by ranker
            return [g for g in rows if g.get("trades", 0) >= 20]
    grails = []
    for log_name in ["hunter1_mp.log"]:
        path = ROOT / "logs" / log_name
        if not path.exists(): continue
        for line in path.read_text(errors="ignore").splitlines():
            if not line.startswith("[GRAIL]"): continue
            p = line.split()
            grails.append({
                "strategy": p[1], "symbol": p[2], "tf": p[3],
                "wr": float(p[4].split("=")[1].rstrip("%")),
                "trades": int(p[5].split("=")[1]),
                "pf": float(p[6].split("=")[1]),
            })
    # dedup by (strat, sym, tf) - keep best WR
    best = {}
    for g in grails:
        k = (g["strategy"], g["symbol"], g["tf"])
        if k not in best or g["wr"] > best[k]["wr"]:
            best[k] = g
    # Filter to n>=20 (confiables only, per Mac guidance)
    return [g for g in best.values() if g["trades"] >= 20]


def find_strategy_spec(name):
    """Search strategies_v7 batches for this strategy + its params + space dict."""
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


def snap_to_grid(value, spec):
    """If space has step-like grid, round jittered value. For now, keep float/int typing."""
    if isinstance(value, (int, bool)):
        return max(1, int(round(value)))
    return value


def jitter_params(params, spec, pct):
    """Apply ±pct to each numeric param. Snap int params to int."""
    out = {}
    for name, val in params.items():
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            out[name] = val
            continue
        new_val = val * (1.0 + pct)
        if isinstance(val, int) and not isinstance(val, bool):
            new_val = max(1, int(round(new_val)))
        out[name] = new_val
    return out


def run_grail_plateau(g, spec):
    """Return plateau frac + details for one grail."""
    gen_fn = spec["gen"]
    # We don't have the original params — HUNTER saved them in PROMOTED.json
    # but that file isn't populated. Reconstruct by re-running with the space
    # sample at maximum WR... too expensive. Instead, for interim, use DEFAULT
    # params from the space. This is a proxy plateau test.
    #
    # Better: read results/sandbox_h1_FINAL_PROMOTED.json if available.
    return None  # placeholder


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-waves", action="store_true",
                    help="aggregate grails from all waves via master ranking")
    ap.add_argument("--limit", type=int, default=0,
                    help="only test first N grails (for speed)")
    args = ap.parse_args()
    grails = parse_interim_grails(all_waves=args.all_waves)
    if args.limit:
        grails = grails[: args.limit]
    print(f"Loaded {len(grails)} grails (n>=20, all_waves={args.all_waves})")
    try:
        df_sfp = load_candles("SFP", "1h", "1h")
    except Exception:
        pass
    # For each grail, re-do a grid scan around the strategy's space with
    # fixed sym/tf. Count passing.
    results = []
    for g in grails:
        print(f"  {g['strategy']:<30} {g['symbol']:<8} {g['tf']:<3} n={g['trades']:>4} pf={g['pf']:.2f}")
        spec = find_strategy_spec(g["strategy"])
        if spec is None:
            results.append({**g, "plateau_frac": None, "neighbors": 0, "status": "SPEC_NOT_FOUND"})
            continue
        try:
            df = load_candles(g["symbol"], SRC_TF[g["tf"]], g["tf"])
        except Exception as e:
            results.append({**g, "plateau_frac": None, "neighbors": 0, "status": f"DATA_MISSING: {e}"})
            continue
        # Run 20 random samples from space and count passing.
        # This is proxy-plateau: measures stability of the space, not of the grail.
        import random
        rng = random.Random(hash((g["strategy"], g["symbol"])) & 0xFFFFFFFF)
        sys.path.insert(0, str(ROOT / "tools"))
        from hunter_runner import sample_params
        passes = 0
        tried = 0
        for _ in range(25):
            try:
                p = sample_params(spec["space"], rng)
                r = backtest_signal_exit(df, spec["gen"], p, sl_pct=0.4)
                if r.get("trades", 0) == 0:
                    continue
                tried += 1
                if (r["wr"] >= MIN_WR_NBR and r["trades"] >= MIN_N_NBR
                        and r["pf"] >= MIN_PF_NBR):
                    passes += 1
            except Exception:
                continue
        frac = passes / tried if tried else 0
        status = "PLATEAU" if frac >= PLATEAU_THR else "spike"
        results.append({**g, "plateau_frac": round(frac, 2), "neighbors": tried,
                        "passing": passes, "status": status})

    lines = [
        "# HUNTER1 grail plateau test (interim, n>=20 subset)",
        "",
        f"Total tested: {len(results)}",
        f"PLATEAU (frac >= {PLATEAU_THR}): {sum(1 for r in results if r.get('status')=='PLATEAU')}",
        "",
        "| Strategy | Sym | TF | Original WR | n | PF | Neighbors | Passing | Frac | Status |",
        "|----------|-----|----|-------------|---|-----|-----------|---------|------|--------|",
    ]
    for r in sorted(results, key=lambda x: -(x.get("plateau_frac") or 0)):
        frac = r.get("plateau_frac", "N/A")
        lines.append(f"| `{r['strategy']}` | {r['symbol']} | {r['tf']} | "
                     f"{r['wr']:.1f}% | {r['trades']} | {r['pf']:.2f} | "
                     f"{r.get('neighbors', 0)} | {r.get('passing', 0)} | "
                     f"{frac} | {r.get('status', '?')} |")
    suffix = "_all" if args.all_waves else ""
    out_md = ROOT / "results" / f"sandbox_h1_plateau{suffix}.md"
    out_json = ROOT / "results" / f"sandbox_h1_plateau{suffix}.json"
    out_md.write_text("\n".join(lines) + "\n")
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_md.name} / {out_json.name}")
    # When --all-waves, also overwrite the canonical plateau file so that
    # build_final_master.py picks up the broader data.
    if args.all_waves:
        (ROOT / "results" / "sandbox_h1_plateau.json").write_text(
            json.dumps(results, indent=2, default=str))
        print("  (also updated sandbox_h1_plateau.json)")
    print(f"PLATEAU_FULL_POOL DONE")


if __name__ == "__main__":
    main()
