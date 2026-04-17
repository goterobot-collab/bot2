#!/usr/bin/env python3
"""
Read results/grails_loop.jsonl, dedup on (strategy, asset, tf, params, exit_cfg),
rank by a composite score, write results/top_grails.md.

Composite score = ret_pct * pf * (trades>=8 ? 1 : trades/8) * (dd>-15 ? 1 : 0.6)
...which penalizes very few-trade luck and deeper drawdowns.
"""
import json
from pathlib import Path

SRC = Path("results/grails_loop.jsonl")
OUT = Path("results/top_grails.md")


def score(r):
    """Composite score that penalizes small-sample luck and deep drawdowns.
    PF capped at 5.0 so a 6-trade WR=100%/PF=999 can't dominate a 50-trade
    WR=75%/PF=2.5 that's more likely to hold out-of-sample."""
    ret = r["ret"]; pf = r["pf"]; dd = r["dd"]; t = r["trades"]; wr = r["wr"]
    if ret <= 0 or pf <= 1 or wr <= 60 or dd <= -30 or t <= 5:
        return -1e9
    pf_capped = min(pf, 5.0)
    # sigmoid-ish trade count penalty: 0.4 at 6 trades, 0.9 at 30, 1.0 at 100+
    trade_pen = min(1.0, 0.4 + 0.6 * (t - 6) / (100 - 6)) if t >= 6 else 0.0
    dd_pen = 1.0 if dd > -10 else (0.8 if dd > -20 else 0.6)
    return ret * pf_capped * trade_pen * dd_pen


def main():
    seen = {}
    n_total = 0
    n_grail = 0
    for line in SRC.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        n_total += 1
        if score(r) <= 0:
            continue
        key = (r["strategy"], r["asset"], r["tf"],
               json.dumps(r["params"], sort_keys=True),
               json.dumps({k: v for k, v in r["exit_cfg"].items() if v is not None}, sort_keys=True))
        if key not in seen or score(r) > score(seen[key]):
            seen[key] = r
            n_grail += 1
    rows = sorted(seen.values(), key=score, reverse=True)

    lines = [
        "# 🏆 TOP GRAILS (deduplicated & ranked)",
        "",
        f"- Source: `results/grails_loop.jsonl` ({n_total} total evaluations)",
        f"- Unique grail configs: **{len(rows)}**",
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
