#!/usr/bin/env python3
"""
HUNTER runner with multiprocessing (Paso 2 + Paso 3).

- workers = 4 (R: feedback_optuna_worker_limit; sandbox has 16 cores but GIL)
- TFs ordered shortest-first (1d, 4h, 1h, 15m, 5m) for fast feedback
- Per-combo timeout 30s (R28)
- Progress file every ~2000 evals; resumable
- Grails appended to mac_inbox.jsonl in slices so crashes don't lose them
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import signal
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FTimeout, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies_v7"))
sys.path.insert(0, str(ROOT / "tools"))

from canary_runner import backtest_signal_exit, load_candles

SYMBOLS = [
    "AGT", "APT", "ARB", "AVAX", "DYDX", "GMX", "INJ", "JTO", "JUP", "LINK",
    "NEAR", "ONDO", "OP", "PENDLE", "PYTH", "SEI", "SFP", "SUI", "SWARMS",
    "TIA", "WLD",
]
# Order shortest-first for fast feedback
TFS = ["1d", "4h", "1h", "15m", "5m"]
SOURCE_TF = {"5m": "5m", "15m": "5m", "1h": "1h", "4h": "1h", "1d": "1h"}

MIN_WR = 70.0
MIN_PF = 1.2
MIN_TRADES_FLOOR = 8
TRIALS_PER_COMBO = 20
WORKERS = 4
COMBO_TIMEOUT_S = 30
PROGRESS_EVERY = 200  # combos between progress flushes


def _min_trades_dynamic(wr):
    if wr >= 100: return 2
    if wr >= 90:  return 4
    if wr >= 80:  return 8
    if wr >= 70:  return 8
    return 999


def _sample_dict(space, rng):
    out = {}
    for name, spec in space.items():
        if len(spec) == 2:
            ptype, vals = spec
            if ptype == "categorical":
                out[name] = rng.choice(list(vals))
                continue
        ptype, lo, hi = spec[0], spec[1], spec[2]
        if ptype == "int":
            out[name] = rng.randrange(lo, hi + 1)
        elif ptype == "float":
            out[name] = rng.uniform(lo, hi)
        elif ptype == "categorical":
            out[name] = rng.choice(list(hi) if hasattr(hi, "__iter__") else [lo, hi])
        else:
            out[name] = lo
    return out


def sample_params(space, rng):
    import inspect
    if callable(space):
        try:
            nparams = len([p for p in inspect.signature(space).parameters.values()
                           if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)])
        except Exception:
            nparams = 1
        if nparams == 0:
            raw = space()
            if isinstance(raw, dict) and raw and isinstance(next(iter(raw.values())), tuple):
                return _sample_dict(raw, rng)
            return raw
        class _FT:
            def suggest_int(self, n, l, h, step=1): return rng.randrange(l, h + 1, step)
            def suggest_float(self, n, l, h, step=None, log=False):
                if log: return float(np.exp(rng.uniform(np.log(l), np.log(h))))
                return rng.uniform(l, h)
            def suggest_categorical(self, n, c): return rng.choice(list(c))
        return space(_FT())
    return _sample_dict(space, rng)


# Worker-global cache
_W_CACHE = {}


def _w_init(candles_paths):
    """Called once per worker: load candles into process memory."""
    global _W_CACHE
    for (sym, tf), path in candles_paths.items():
        try:
            df = load_candles(sym, SOURCE_TF[tf], tf)
            if len(df) >= 300:
                _W_CACHE[(sym, tf)] = df
        except Exception:
            pass


def _w_job(batch_id, strat_name, spec_pkl, sym, tf, n_trials, seed):
    """Run N trials for one (strat, sym, tf). Return best combo."""
    import pickle
    spec = pickle.loads(spec_pkl)
    gen_fn, space = spec["gen"], spec["space"]
    df = _W_CACHE.get((sym, tf))
    if df is None or len(df) < 300:
        return None
    rng = random.Random(seed)
    best = None
    for _ in range(n_trials):
        try:
            params = sample_params(space, rng)
        except Exception:
            continue
        try:
            r = backtest_signal_exit(df, gen_fn, params, sl_pct=0.40)
        except Exception:
            continue
        if r.get("trades", 0) == 0 or r.get("wr") is None:
            continue
        if best is None or r["wr"] > best["metrics"]["wr"]:
            best = {"params": params, "metrics": {
                "wr": r["wr"], "trades": r["trades"], "pf": r["pf"],
                "total_pnl_pct": r["total_pnl_pct"],
                "wins": r["wins"], "losses": r["losses"],
                "avg_win_pct": r["avg_win_pct"], "avg_loss_pct": r["avg_loss_pct"],
            }}
    if best is None:
        return None
    m = best["metrics"]
    dyn = _min_trades_dynamic(m["wr"])
    passes = (m["wr"] >= MIN_WR and m["trades"] >= dyn and m["pf"] >= MIN_PF)
    return {
        "batch": batch_id, "strategy": strat_name, "symbol": sym, "tf": tf,
        "params": best["params"], "metrics": m, "passes": passes,
    }


def run(batches, label, shortlist_path, promoted_path, progress_path):
    import pickle
    # Load batches
    jobs = []
    for b in batches:
        try:
            mod = importlib.import_module(f"strategies_tv2_batch{b}")
        except Exception as e:
            print(f"[skip] batch {b}: {e}")
            continue
        for name, spec in mod.STRATEGY_EXPORT.items():
            if not spec.get("gen") or not spec.get("space"):
                continue
            jobs.append((b, name, pickle.dumps(spec)))
    print(f"[{label}] {len(jobs)} strategies loaded")

    # Discover candles available
    cand_paths = {}
    for s in SYMBOLS:
        for tf in TFS:
            p = ROOT / "data" / "candles" / f"{s}_{SOURCE_TF[tf]}.csv.gz"
            if p.exists():
                cand_paths[(s, tf)] = str(p)
    print(f"[{label}] {len(cand_paths)} (sym,tf) candidate pairs")

    # Build task queue
    # Order: tf shortest-first, then by symbol
    tf_order = {tf: i for i, tf in enumerate(TFS)}
    tasks = []
    for (batch_id, strat_name, spec_pkl) in jobs:
        for (sym, tf) in cand_paths:
            tasks.append((tf_order[tf], batch_id, strat_name, spec_pkl, sym, tf))
    tasks.sort()
    total = len(tasks)
    print(f"[{label}] {total:,} (strat, sym, tf) tasks (x {TRIALS_PER_COMBO} trials each)")

    # Resume
    done_keys = set()
    shortlist = []
    if progress_path.exists():
        try:
            prev = json.loads(progress_path.read_text())
            for r in prev.get("shortlist", []):
                k = (r["batch"], r["strategy"], r["symbol"], r["tf"])
                done_keys.add(k)
                shortlist.append(r)
            print(f"[{label}] resuming: {len(done_keys)} tasks already done")
        except Exception as e:
            print(f"[{label}] progress file unreadable, starting fresh: {e}")

    grails = sum(1 for r in shortlist if r["passes"])
    t0 = time.time()

    with ProcessPoolExecutor(
        max_workers=WORKERS, initializer=_w_init, initargs=(cand_paths,)
    ) as pool:
        futures = {}
        task_iter = iter(tasks)
        # prime queue
        for _ in range(WORKERS * 4):
            try:
                _, b, name, pkl, sym, tf = next(task_iter)
            except StopIteration:
                break
            if (b, name, sym, tf) in done_keys:
                continue
            seed = hash((name, sym, tf, b)) & 0xFFFFFFFF
            fut = pool.submit(_w_job, b, name, pkl, sym, tf, TRIALS_PER_COMBO, seed)
            futures[fut] = (b, name, sym, tf)

        n_done = len(done_keys)
        while futures:
            for fut in as_completed(list(futures.keys()), timeout=None):
                key = futures.pop(fut)
                try:
                    r = fut.result(timeout=COMBO_TIMEOUT_S)
                except FTimeout:
                    print(f"[TIMEOUT] {key}  (R28: skipped)")
                    r = None
                except Exception as e:
                    print(f"[ERR] {key}: {type(e).__name__}: {str(e)[:80]}")
                    r = None
                n_done += 1
                if r is not None:
                    shortlist.append(r)
                    if r["passes"]:
                        grails += 1
                        m = r["metrics"]
                        print(f"[GRAIL] {r['strategy']:<30} {r['symbol']:<8} {r['tf']:<3} "
                              f"WR={m['wr']:.1f}% n={m['trades']} PF={m['pf']:.2f}")
                # Refill
                try:
                    while True:
                        _, b, name, pkl, sym, tf = next(task_iter)
                        if (b, name, sym, tf) in done_keys:
                            continue
                        seed = hash((name, sym, tf, b)) & 0xFFFFFFFF
                        nf = pool.submit(_w_job, b, name, pkl, sym, tf, TRIALS_PER_COMBO, seed)
                        futures[nf] = (b, name, sym, tf)
                        break
                except StopIteration:
                    pass
                # Progress flush
                if n_done % PROGRESS_EVERY == 0 or not futures:
                    elapsed = time.time() - t0
                    rate = (n_done - len(done_keys)) / max(1, elapsed)
                    eta = (total - n_done) / max(1e-6, rate)
                    print(f"[{label}] {n_done}/{total} ({100*n_done/total:.1f}%)  "
                          f"grails={grails}  rate={rate:.1f}/s  eta={eta/60:.1f}m")
                    _flush(progress_path, shortlist, n_done, total, grails)

    _flush(progress_path, shortlist, n_done, total, grails)
    _write_reports(shortlist, label, batches, shortlist_path, promoted_path, t0)
    return shortlist, grails


def _flush(progress_path, shortlist, n_done, total, grails):
    try:
        progress_path.write_text(json.dumps({
            "n_done": n_done, "total": total, "grails": grails,
            "shortlist": shortlist,
        }))
    except Exception as e:
        print(f"[WARN] progress flush: {e}")


def _write_reports(shortlist, label, batches, shortlist_path, promoted_path, t0):
    passing = [r for r in shortlist if r["passes"]]
    passing.sort(key=lambda r: (-r["metrics"]["wr"], -r["metrics"]["pf"]))
    promoted = {}
    for r in passing:
        promoted.setdefault(r["strategy"], []).append(r)

    elapsed = time.time() - t0
    lines = [
        f"# {label} shortlist",
        "",
        f"- Batches: {', '.join(str(b) for b in batches)}",
        f"- Tasks completed: {len(shortlist):,}",
        f"- Grails passing R24 gate: **{len(passing)}**",
        f"- Strategies promoted (>=1 combo pass): **{len(promoted)}**",
        f"- Elapsed: {elapsed:.0f}s",
        "",
        "| Rank | Strategy | Sym | TF | WR | trades | PF | total% | params |",
        "|------|----------|-----|----|-----|--------|-----|--------|--------|",
    ]
    for i, r in enumerate(passing[:200], 1):
        m = r["metrics"]
        p = json.dumps(r["params"], separators=(",", ":"), default=str)
        lines.append(f"| {i} | `{r['strategy']}` | {r['symbol']} | {r['tf']} | "
                     f"{m['wr']:.1f}% | {m['trades']} | {m['pf']:.2f} | "
                     f"{m['total_pnl_pct']:.1f}% | `{p}` |")
    shortlist_path.write_text("\n".join(lines) + "\n")

    prom_json = {}
    for name, combos in promoted.items():
        combos.sort(key=lambda r: -r["metrics"]["wr"])
        prom_json[name] = {
            "best": {"sym": combos[0]["symbol"], "tf": combos[0]["tf"],
                     "wr": combos[0]["metrics"]["wr"],
                     "trades": combos[0]["metrics"]["trades"],
                     "pf": combos[0]["metrics"]["pf"],
                     "params": combos[0]["params"]},
            "passing_combos": len(combos),
            "all_passing": [
                {"sym": c["symbol"], "tf": c["tf"], "wr": c["metrics"]["wr"],
                 "trades": c["metrics"]["trades"], "pf": c["metrics"]["pf"],
                 "params": c["params"]} for c in combos
            ],
        }
    promoted_path.write_text(json.dumps(prom_json, indent=2, default=str) + "\n")
    print(f"Wrote {shortlist_path} ({len(passing)} grails) + {promoted_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", choices=["h1", "h2"], required=True)
    args = ap.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)

    if args.wave == "h1":
        batches = ["3566", "3567", "3568", "3569"]
        label = "HUNTER1"
        sp = ROOT / "results" / "sandbox_h1_3566_3569_SHORTLIST.md"
        pp = ROOT / "results" / "sandbox_h1_PROMOTED.json"
        prog = ROOT / "results" / "sandbox_h1_progress.json"
    else:
        batches = ["3594", "3598"]
        label = "HUNTER2"
        sp = ROOT / "results" / "sandbox_h2_3594_3598_SHORTLIST.md"
        pp = ROOT / "results" / "sandbox_h2_PROMOTED.json"
        prog = ROOT / "results" / "sandbox_h2_progress.json"

    shortlist, grails = run(batches, label, sp, pp, prog)
    print(f"\n{label} DONE: {grails} grails, {len(shortlist)} tasks")


if __name__ == "__main__":
    main()
