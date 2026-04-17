#!/usr/bin/env python3
"""
Read results/grails_loop.jsonl, dedup on (strategy, asset, tf, params, exit_cfg),
rank by a composite score, write results/top_grails.md.

Composite score = ret_pct * pf * (trades>=8 ? 1 : trades/8) * (dd>-15 ? 1 : 0.6)
...which penalizes very few-trade luck and deeper drawdowns.
"""
import json
from pathlib import Path

SRC = Path("results/grails_loop_wf.jsonl")  # walk-forward v2
OUT = Path("results/top_grails.md")


def count_by_family(d: dict) -> str:
    fams = {}
    for k in d:
        fams[k[0]] = fams.get(k[0], 0) + 1
    return ", ".join(f"{k}={v}" for k, v in sorted(fams.items(), key=lambda x: -x[1]))


def score(r):
    """Walk-forward-aware score. Require grail filter on full + both halves.
    Reward consistency between halves (penalize if one half is much worse).
    """
    ret = r["ret"]; pf = r["pf"]; dd = r["dd"]; t = r["trades"]; wr = r["wr"]
    if ret <= 0 or pf <= 1 or wr <= 60 or dd <= -30 or t < 50:
        return -1e9
    # walk-forward constraints (if present in the row)
    if "h1_wr" in r:
        if r["h1_wr"] <= 60 or r["h2_wr"] <= 60: return -1e9
        if r["h1_ret"] <= 0 or r["h2_ret"] <= 0: return -1e9
        if r["h1_pf"] <= 1 or r["h2_pf"] <= 1:   return -1e9
        if r["h1_dd"] <= -30 or r["h2_dd"] <= -30: return -1e9
        if r["h1_trades"] < 20 or r["h2_trades"] < 20: return -1e9
    pf_capped = min(pf, 5.0)
    trade_pen = min(1.0, 0.4 + 0.6 * (t - 50) / (200 - 50)) if t >= 50 else 0.0
    dd_pen = 1.0 if dd > -10 else (0.8 if dd > -20 else 0.6)
    # consistency bonus: reward when h1 and h2 WR are close
    cons = 1.0
    if "h1_wr" in r:
        gap = abs(r["h1_wr"] - r["h2_wr"])
        cons = 1.0 if gap < 10 else (0.85 if gap < 20 else 0.7)
    return ret * pf_capped * trade_pen * dd_pen * cons


def main():
    seen = {}
    seen_cfg = {}  # dedup by (strategy, params, exit) ignoring asset/tf
    n_total = 0
    n_grail = 0
    for line in SRC.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        n_total += 1
        if score(r) <= 0:
            continue
        params_k = json.dumps(r["params"], sort_keys=True)
        exit_k = json.dumps({k: v for k, v in r["exit_cfg"].items() if v is not None}, sort_keys=True)
        key = (r["strategy"], r["asset"], r["tf"], params_k, exit_k)
        cfg_key = (r["strategy"], params_k, exit_k)
        if key not in seen or score(r) > score(seen[key]):
            seen[key] = r
            n_grail += 1
        if cfg_key not in seen_cfg or score(r) > score(seen_cfg[cfg_key]):
            seen_cfg[cfg_key] = r
    rows = sorted(seen.values(), key=score, reverse=True)
    n_unique_cfg = len(seen_cfg)

    lines = [
        "# 🏆 TOP GRAILS (deduplicated & ranked)",
        "",
        f"- Source: `results/grails_loop.jsonl` ({n_total} total evaluations)",
        f"- Unique **(strategy, asset, params, exit)** combos: **{len(rows)}**",
        f"- Unique **strategy configs** (asset-agnostic): **{n_unique_cfg}**",
        f"- Family breakdown (asset-agnostic configs): {count_by_family(seen_cfg)}",
        "- Score = `ret × PF × trade-penalty × DD-penalty` (rewards profit, consistency, low DD)",
        "- Filter: WR>60% AND PnL>0 AND maxDD>-30% AND PF>1 AND trades>5",
        "",
        "| Rank | Score | Strategy | Asset | TF | Trades | WR % | PF | Ret % | DD % | Params | Exit |",
        "|------|-------|----------|-------|----|--------|------|-----|-------|------|--------|------|",
    ]
    for i, r in enumerate(rows, 1):
        exit_nice = {k: v for k, v in r["exit_cfg"].items() if v is not None}
        sc = score(r)
        lines.append(
            f"| {i} | {sc:.2f} | `{r['strategy']}` | {r['asset']} | {r['tf']} | "
            f"{r['trades']} | {r['wr']} | {r['pf']} | {r['ret']} | {r['dd']} | "
            f"`{json.dumps(r['params'], separators=(',',':'))}` | "
            f"`{json.dumps(exit_nice, separators=(',',':'))}` |"
        )
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT} with {len(rows)} unique grails (from {n_total} evals).")
    if rows:
        print("Top 5:")
        for r in rows[:5]:
            print(f"  {r['strategy']:<18} {r['asset']:<8} WR={r['wr']:>5} "
                  f"PF={r['pf']:>6} ret={r['ret']:>7}% DD={r['dd']:>6}% trades={r['trades']}")


if __name__ == "__main__":
    main()
