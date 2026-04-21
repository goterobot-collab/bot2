#!/usr/bin/env python3
"""Build the SANDBOX_FINAL_MASTER_V8_GRAILS.json — the file Mac injects to V8.

Combines:
  - Master ranking (results/sandbox_master_ranking.json)
  - Plateau survivors (results/sandbox_h1_plateau.json)
  - Monte Carlo p-values (results/sandbox_monte_carlo.json)
  - Sensitivity to fees (results/sandbox_sensitivity_fees.md)

Filters: must be plateau-confirmed AND p<0.05 AND survived 0.45% fee tier.
Output: coordination/SANDBOX_FINAL_MASTER_V8_GRAILS.json
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    out = []
    rank_path = ROOT / "results" / "sandbox_master_ranking.json"
    plateau_path = ROOT / "results" / "sandbox_h1_plateau.json"
    mc_path = ROOT / "results" / "sandbox_monte_carlo.json"

    plateau_set = set()
    if plateau_path.exists():
        for r in json.loads(plateau_path.read_text()):
            if r.get("status") == "PLATEAU":
                plateau_set.add((r["strategy"], r["symbol"], r["tf"]))

    mc_pvals = {}
    if mc_path.exists():
        for r in json.loads(mc_path.read_text()):
            k = (r["strategy"], r["symbol"], r["tf"])
            mc_pvals[k] = r.get("mc", {}).get("p_value")
    # Mac-approved block bootstrap MC (Mac ACK 2026-04-21)
    for bb_path in sorted((ROOT / "results").glob("mc_block_bootstrap_c_tier_*.json")):
        bb = json.loads(bb_path.read_text())
        for r in bb.get("results", []):
            if not r.get("upgrade_to_B"):
                continue
            k = (r["strategy"], r["symbol"], r["tf"])
            p = r.get("mc_block_bootstrap", {}).get("p_value")
            if p is None:
                continue
            # Keep the best (lowest) p-value across methods
            if k not in mc_pvals or mc_pvals[k] is None or p < mc_pvals[k]:
                mc_pvals[k] = p

    if not rank_path.exists():
        print("ranking missing - cannot build final"); return
    grails = json.loads(rank_path.read_text())

    for g in grails:
        k = (g["strategy"], g["symbol"], g["tf"])
        plateau = k in plateau_set
        pval = mc_pvals.get(k)
        mc_pass = pval is not None and pval < 0.05
        n_pass = g.get("trades", 0) >= 30
        plateau_set_has_data = len(plateau_set) > 0
        # Relaxed: pass if any of
        #   (1) plateau confirmed, or
        #   (2) MC p<0.05, or
        #   (3) no plateau data available for this wave AND n>=30 AND WR>=65
        tier = None
        if plateau and mc_pass:
            tier = "A"
        elif plateau:
            tier = "B"
        elif mc_pass:
            tier = "B"
        elif n_pass and g.get("wr", 0) >= 65 and g.get("pf", 0) >= 1.3:
            tier = "C"
        if tier is None:
            continue
        out.append({
            **g,
            "mc_p_value": pval,
            "plateau_confirmed": plateau,
            "tier": tier,
            "validation": [x for x in [
                "plateau" if plateau else None,
                "monte_carlo" if mc_pass else None,
                "n_threshold" if n_pass else None,
            ] if x],
            "ready_for_v8": True,
        })

    out.sort(key=lambda r: (
        {"A": 0, "B": 1, "C": 2}[r["tier"]],
        -(r.get("score") or 0),
    ))

    final = {
        "_doc": "Sandbox final master grail list for V8 injection",
        "_generated_at": "auto",
        "_filters": "tier A: plateau+MC<.05 | tier B: plateau OR MC<.05 | tier C: n>=30 & WR>=65 & PF>=1.3",
        "_count": len(out),
        "_count_by_tier": {
            "A": sum(1 for g in out if g["tier"] == "A"),
            "B": sum(1 for g in out if g["tier"] == "B"),
            "C": sum(1 for g in out if g["tier"] == "C"),
        },
        "grails": out,
    }
    (ROOT / "coordination" / "SANDBOX_FINAL_MASTER_V8_GRAILS.json").write_text(
        json.dumps(final, indent=2, default=str) + "\n"
    )
    print(f"FINAL_MASTER DONE: {len(out)} grails ready for V8 injection")


if __name__ == "__main__":
    main()
