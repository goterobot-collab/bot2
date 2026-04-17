#!/usr/bin/env python3
"""Locked 20% OOS holdout test.
Re-evaluates each unique WF grail from results/grails_loop_wf.jsonl on the
final 20% of its asset's CSV. Reports which still satisfy the grail filter.
Run ONCE per holdout cut. Re-running after any retune contaminates the OOS.
See results/locked_oos_proposal.md for the full rationale.
Output: results/holdout_report.md + results/holdout_results.jsonl
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grail_loop import SPACES, load_all
from backtest import simulate, metrics_from_trades

SRC = Path("results/grails_loop_wf.jsonl")
OUT_MD = Path("results/holdout_report.md")
OUT_JSONL = Path("results/holdout_results.jsonl")
HOLDOUT_FRAC, MIN_TR_HO = 0.20, 10
FEE, SLIPPAGE = 0.001, 0.0005


def _grail(wr, ret, pf, dd, tr, min_tr):
    return wr > 60 and ret > 0 and pf > 1 and dd > -30 and tr >= min_tr


def passes(m):
    return _grail(m.wr, m.total_return_pct, m.profit_factor,
                  m.max_drawdown_pct, m.trades, MIN_TR_HO + 1)


def passes_in_loop(r):
    """Reproduce grail_loop's WF filter to recover the ~1059 inputs."""
    if "h1_wr" not in r:
        return False
    return (_grail(r["wr"], r["ret"], r["pf"], r["dd"], r["trades"], 50)
            and _grail(r["h1_wr"], r["h1_ret"], r["h1_pf"], r["h1_dd"],
                       r["h1_trades"], 20)
            and _grail(r["h2_wr"], r["h2_ret"], r["h2_pf"], r["h2_dd"],
                       r["h2_trades"], 20))


def load_unique_grails():
    seen = {}
    for line in SRC.open():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not passes_in_loop(r):
            continue
        pk = json.dumps(r["params"], sort_keys=True)
        ek = json.dumps({k: v for k, v in r["exit_cfg"].items() if v is not None}, sort_keys=True)
        seen.setdefault((r["strategy"], r["asset"], r["tf"], pk, ek), r)
    return list(seen.values())


def eval_holdout(df, fam, params, exit_cfg, asset, tf):
    try:
        ent, ex = SPACES[fam]["sig"](df, params)
    except Exception:
        return None
    cut = int(len(df) * (1.0 - HOLDOUT_FRAC))
    sub = df.iloc[cut:]
    trades = simulate(sub, ent.iloc[cut:], ex.iloc[cut:], fee=FEE,
                      slippage=SLIPPAGE, sl_atr=exit_cfg.get("sl_atr"),
                      tp_atr=exit_cfg.get("tp_atr"),
                      trail_atr=exit_cfg.get("trail_atr"),
                      timeout=exit_cfg.get("timeout"))
    return metrics_from_trades(fam, asset, tf, trades, sub)


def main():
    grails, dfs = load_unique_grails(), load_all(None)
    print(f"{len(grails)} unique WF grails | {len(dfs)} datasets | "
          f"holdout=last {int(HOLDOUT_FRAC*100)}%")
    rows, survivors, jf = [], 0, OUT_JSONL.open("w")
    for i, r in enumerate(grails, 1):
        if (r["asset"], r["tf"]) not in dfs:
            continue
        ec = {k: v for k, v in r["exit_cfg"].items() if v is not None}
        m = eval_holdout(dfs[(r["asset"], r["tf"])], r["strategy"],
                         r["params"], ec, r["asset"], r["tf"])
        if m is None:
            continue
        ok = passes(m); survivors += int(ok)
        row = {"strategy": r["strategy"], "asset": r["asset"], "tf": r["tf"],
               "params": r["params"], "exit_cfg": ec, "is_wr": r["wr"],
               "is_ret": r["ret"], "is_pf": r["pf"], "ho_trades": m.trades,
               "ho_wr": m.wr, "ho_pf": m.profit_factor,
               "ho_ret": m.total_return_pct, "ho_dd": m.max_drawdown_pct,
               "survives": ok}
        jf.write(json.dumps(row) + "\n"); rows.append(row)
        if i % 50 == 0 or i == len(grails):
            print(f"  [{i}/{len(grails)}] survivors={survivors}")
    jf.close()

    rows.sort(key=lambda x: (-int(x["survives"]), -x["ho_ret"]))
    pct = 100.0 * survivors / max(1, len(rows))
    hdr = "| # | Strategy | Asset | TF | IS WR | IS Ret | HO Tr | HO WR | HO PF | HO Ret | HO DD | Survives |\n|-|-|-|-|-|-|-|-|-|-|-|-|"
    body = "\n".join(
        f"| {i} | `{x['strategy']}` | {x['asset']} | {x['tf']} | {x['is_wr']}"
        f" | {x['is_ret']} | {x['ho_trades']} | {x['ho_wr']} | {x['ho_pf']}"
        f" | {x['ho_ret']} | {x['ho_dd']} |"
        f" {'YES' if x['survives'] else 'no'} |"
        for i, x in enumerate(rows, 1))
    OUT_MD.write_text(
        f"# Locked 20% OOS holdout report\n\n"
        f"- Inputs: {len(grails)} unique v2 WF grails\n"
        f"- Holdout: final {int(HOLDOUT_FRAC*100)}% of each asset CSV\n"
        f"- Filter: WR>60 AND PnL>0 AND DD>-30 AND PF>1 AND trades>{MIN_TR_HO}\n"
        f"- **Survivors: {survivors}/{len(rows)} ({pct:.1f}%)**\n\n"
        f"{hdr}\n{body}\n")
    print(f"Wrote {OUT_MD} and {OUT_JSONL}")


if __name__ == "__main__":
    main()
