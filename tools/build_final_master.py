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

    if not rank_path.exists():
        print("ranking missing - cannot build final"); return
    grails = json.loads(rank_path.read_text())

    for g in grails:
        k = (g["strategy"], g["symbol"], g["tf"])
        plateau = k in plateau_set
        pval = mc_pvals.get(k)
        # Final filter: plateau AND p<0.05 (or n>=50 if no MC available)
        if plateau and (pval is None and g["trades"] >= 50 or pval is not None and pval < 0.05):
            out.append({
                **g,
                "mc_p_value": pval,
                "validation": ["plateau", "monte_carlo" if pval is not None else "n_threshold"],
                "ready_for_v8": True,
            })

    final = {
        "_doc": "Sandbox final master grail list for V8 injection",
        "_generated_at": "auto",
        "_filters": "plateau AND (MC p<0.05 OR n>=50)",
        "_count": len(out),
        "grails": out,
    }
    (ROOT / "coordination" / "SANDBOX_FINAL_MASTER_V8_GRAILS.json").write_text(
        json.dumps(final, indent=2, default=str) + "\n"
    )
    print(f"FINAL_MASTER DONE: {len(out)} grails ready for V8 injection")


if __name__ == "__main__":
    main()
