#!/usr/bin/env python3
"""Composite ranker over all grails found across HUNTER waves + plateau survivors.

Score = WR × log(1+trades) × PF × (1 + plateau_bonus) where plateau_bonus = 0.5 if grail
is in plateau survivors, else 0.

Output: results/sandbox_master_ranking.md + .json
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent

def parse_grails_from_log(log_path: Path):
    grails = []
    if not log_path.exists():
        return grails
    for line in log_path.read_text(errors="ignore").splitlines():
        if not line.startswith("[GRAIL]"):
            continue
        p = line.split()
        try:
            grails.append({
                "strategy": p[1], "symbol": p[2], "tf": p[3],
                "wr": float(p[4].split("=")[1].rstrip("%")),
                "trades": int(p[5].split("=")[1]),
                "pf": float(p[6].split("=")[1]),
                "source": log_path.name,
            })
        except (IndexError, ValueError):
            continue
    return grails


def score(g, plateau_set):
    pf = min(g["pf"], 10.0)  # cap PF outliers
    base = g["wr"] * math.log1p(g["trades"]) * pf
    bonus = 0.5 if (g["strategy"], g["symbol"], g["tf"]) in plateau_set else 0
    return round(base * (1 + bonus), 2)


def parse_grails_from_progress(prog_path: Path):
    """Parse passing grails from a hunter progress.json shortlist."""
    grails = []
    if not prog_path.exists():
        return grails
    try:
        d = json.loads(prog_path.read_text())
    except Exception:
        return grails
    for s in d.get("shortlist", []):
        if not s.get("passes"):
            continue
        m = s.get("metrics", {})
        try:
            grails.append({
                "strategy": s["strategy"], "symbol": s["symbol"], "tf": s["tf"],
                "wr": float(m["wr"]),
                "trades": int(m["trades"]),
                "pf": float(m["pf"]),
                "source": prog_path.name,
            })
        except (KeyError, ValueError, TypeError):
            continue
    return grails


def main():
    all_grails = []
    for log_name in ("hunter1_mp.log", "hunter1_5m15m.log",
                      "hunter2_5m15m.log", "hunter2_high.log",
                      "hunter3_5m15m.log", "hunter3_high.log"):
        all_grails.extend(parse_grails_from_log(ROOT / "logs" / log_name))
    # Also include grails from progress.json shortlists (resumable hunters
    # write per-200-task heartbeats; logs may be rotated on restart).
    for prog_name in (
        "sandbox_h1_5m_15m_progress.json",
        "sandbox_h2_5m_15m_progress.json",
        "sandbox_h2_1h_4h_1d_progress.json",
        "sandbox_h3_5m_15m_progress.json",
        "sandbox_h3_1h_4h_1d_progress.json",
    ):
        all_grails.extend(parse_grails_from_progress(ROOT / "results" / prog_name))

    # dedup (strategy, symbol, tf) -> best WR
    best = {}
    for g in all_grails:
        k = (g["strategy"], g["symbol"], g["tf"])
        if k not in best or g["wr"] > best[k]["wr"]:
            best[k] = g
    grails = list(best.values())

    # plateau survivors
    plateau_set = set()
    p = ROOT / "results" / "sandbox_h1_plateau.json"
    if p.exists():
        try:
            data = json.loads(p.read_text())
            for r in data:
                if r.get("status") == "PLATEAU":
                    plateau_set.add((r["strategy"], r["symbol"], r["tf"]))
        except Exception:
            pass

    for g in grails:
        g["plateau"] = (g["strategy"], g["symbol"], g["tf"]) in plateau_set
        g["score"] = score(g, plateau_set)

    grails.sort(key=lambda x: -x["score"])

    lines = [
        "# Sandbox Master Ranking (all HUNTER waves + plateau bonus)",
        "",
        f"- Total unique grails: {len(grails)}",
        f"- Plateau-confirmed: {sum(1 for g in grails if g['plateau'])}",
        "",
        "Score = WR × log(1+trades) × min(PF,10) × (1.5 if plateau else 1)",
        "",
        "| # | Strategy | Sym | TF | WR | n | PF | Plateau | Score |",
        "|---|----------|-----|----|-----|---|-----|---------|-------|",
    ]
    for i, g in enumerate(grails[:200], 1):
        lines.append(f"| {i} | `{g['strategy']}` | {g['symbol']} | {g['tf']} | "
                     f"{g['wr']:.1f}% | {g['trades']} | {g['pf']:.2f} | "
                     f"{'Y' if g['plateau'] else ''} | {g['score']:.0f} |")
    (ROOT / "results" / "sandbox_master_ranking.md").write_text("\n".join(lines) + "\n")
    (ROOT / "results" / "sandbox_master_ranking.json").write_text(
        json.dumps(grails, indent=2, default=str) + "\n"
    )
    print(f"RANKER DONE: {len(grails)} grails ranked, top {min(200,len(grails))} written")


if __name__ == "__main__":
    main()
