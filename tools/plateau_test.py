#!/usr/bin/env python3
"""
Parameter-plateau stability test.

For each grail in results/top_grails.md, jitter its numeric params and
exit-cfg values by +/-20% (one step at a time and combinations) and
re-evaluate. A grail is on a "plateau" if at least 75% of neighbors also
pass the walk-forward grail filter.

Lone spikes (neighbors collapse) are likely overfits -> reject.

Output: results/plateau_report.md
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grail_loop import SPACES, evaluate, load_all

SRC = Path("results/grails_loop_wf.jsonl")
OUT = Path("results/plateau_full_pool.md")
JITTERS = [-0.2, -0.1, 0.0, 0.1, 0.2]  # +/-20% in 10% steps
PLATEAU_THR = 0.75  # fraction of neighbors that must pass


def _nearest(value, grid):
    """Snap a jittered numeric value to the nearest grid option."""
    grid_num = [g for g in grid if g is not None]
    if not grid_num:
        return value
    return min(grid_num, key=lambda g: abs(g - value))


def _jitter_numeric(base: dict, grid: dict, pct: float) -> dict:
    """Apply a single +/-pct jitter to every numeric key, snap to grid."""
    out = {}
    for k, v in base.items():
        if v is None or not isinstance(v, (int, float)):
            out[k] = v
            continue
        if k not in grid:
            out[k] = v
            continue
        new_val = v * (1.0 + pct)
        if isinstance(v, int):
            new_val = int(round(new_val))
        out[k] = _nearest(new_val, grid[k])
    return out


def _passes(m, min_tr=50) -> bool:
    return (m.trades > min_tr
            and m.wr > 60
            and m.total_return_pct > 0
            and m.max_drawdown_pct > -30
            and m.profit_factor > 1)


def _passes_wf(result) -> bool:
    if result is None:
        return False
    if not _passes(result["full"], 50):
        return False
    if not _passes(result["h1"], 20):
        return False
    if not _passes(result["h2"], 20):
        return False
    return True


def load_grails():
    """Read unique (strategy, asset, params, exit) rows from WF jsonl."""
    seen = {}
    for line in SRC.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # only use those that already passed WF in the loop
        if "h1_wr" not in r:
            continue
        if not (r["wr"] > 60 and r["ret"] > 0 and r["pf"] > 1
                and r["dd"] > -30 and r["trades"] >= 50):
            continue
        if not (r["h1_wr"] > 60 and r["h2_wr"] > 60
                and r["h1_ret"] > 0 and r["h2_ret"] > 0):
            continue
        params_k = json.dumps(r["params"], sort_keys=True)
        exit_k = json.dumps({k: v for k, v in r["exit_cfg"].items()
                             if v is not None}, sort_keys=True)
        key = (r["strategy"], r["asset"], r["tf"], params_k, exit_k)
        if key not in seen:
            seen[key] = r
    return list(seen.values())


def neighbors_of(r):
    """Yield (params, exit_cfg) tuples for +/-20% jitter variations."""
    fam = r["strategy"]
    params = r["params"]
    exit_cfg = {k: v for k, v in r["exit_cfg"].items() if v is not None}
    pg = SPACES[fam]["params"]
    eg = SPACES[fam]["exit"]
    # single-axis neighbors: jitter params, jitter exit, jitter both
    for p_pct, e_pct in product(JITTERS, JITTERS):
        if p_pct == 0.0 and e_pct == 0.0:
            continue  # skip the baseline
        new_p = _jitter_numeric(params, pg, p_pct)
        new_e = _jitter_numeric(exit_cfg, eg, e_pct)
        yield new_p, new_e


def main():
    grails = load_grails()
    print(f"Loaded {len(grails)} unique WF grails to plateau-test")
    dfs = load_all(None)
    print(f"Loaded {len(dfs)} (asset, tf) datasets")

    results = []
    for i, r in enumerate(grails, 1):
        key = (r["asset"], r["tf"])
        if key not in dfs:
            continue
        df = dfs[key]
        total = 0
        passed = 0
        for new_p, new_e in neighbors_of(r):
            # dedupe: if jitter collapsed to baseline, skip
            if (new_p == r["params"]
                    and new_e == {k: v for k, v in r["exit_cfg"].items()
                                  if v is not None}):
                continue
            total += 1
            try:
                res = evaluate(df, r["strategy"], new_p, new_e,
                               fee=0.0005, slippage=0.0002,
                               asset=r["asset"], tf=r["tf"])
            except Exception:
                continue
            if _passes_wf(res):
                passed += 1
        frac = passed / total if total else 0.0
        plateau = frac >= PLATEAU_THR
        results.append({
            "strategy": r["strategy"], "asset": r["asset"], "tf": r["tf"],
            "params": r["params"], "exit_cfg": r["exit_cfg"],
            "wr": r["wr"], "pf": r["pf"], "ret": r["ret"], "dd": r["dd"],
            "trades": r["trades"],
            "neighbors": total, "passed": passed, "frac": frac,
            "plateau": plateau,
        })
        print(f"  [{i}/{len(grails)}] {r['strategy']:<18} {r['asset']:<8} "
              f"neighbors={total:>3} passed={passed:>3} frac={frac:.2f} "
              f"{'PLATEAU' if plateau else 'SPIKE'}")

    results.sort(key=lambda x: (-x["frac"], -x["ret"]))
    lines = [
        "# Parameter-plateau stability report",
        "",
        f"- Input: {len(grails)} unique WF grails",
        f"- Jitter: +/-20% in 10% steps on every numeric param and exit-cfg value",
        f"- Plateau threshold: >= {int(PLATEAU_THR * 100)}% of neighbors still pass WF filter",
        "",
        "| Rank | Strategy | Asset | TF | Trades | WR | PF | Ret% | DD% | "
        "Neighbors | Passed | Frac | Verdict |",
        "|------|----------|-------|----|--------|----|----|------|-----|"
        "-----------|--------|------|---------|",
    ]
    for i, x in enumerate(results, 1):
        lines.append(
            f"| {i} | `{x['strategy']}` | {x['asset']} | {x['tf']} | "
            f"{x['trades']} | {x['wr']} | {x['pf']} | {x['ret']} | {x['dd']} | "
            f"{x['neighbors']} | {x['passed']} | {x['frac']:.2f} | "
            f"{'PLATEAU' if x['plateau'] else 'spike'} |"
        )
    lines.append("")
    lines.append("## Survivors (plateau)")
    lines.append("")
    for x in results:
        if not x["plateau"]:
            continue
        lines.append(
            f"- **{x['strategy']}** {x['asset']} {x['tf']} | "
            f"params=`{json.dumps(x['params'], separators=(',', ':'))}` | "
            f"exit=`{json.dumps({k: v for k, v in x['exit_cfg'].items() if v is not None}, separators=(',', ':'))}` | "
            f"frac={x['frac']:.2f}"
        )

    OUT.write_text("\n".join(lines) + "\n")
    n_plateau = sum(1 for x in results if x["plateau"])
    print(f"\nWrote {OUT}: {n_plateau}/{len(results)} grails pass plateau test")


if __name__ == "__main__":
    main()
