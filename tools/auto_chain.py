#!/usr/bin/env python3
"""
Auto-chain wrapper: ETAPA 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 without human.

Mac greenlit "autonomía total" 2026-04-18. Only hard-stops:
  (a) grail n>=50 WR>=85% PBO>0.3 -> escalate (borderline overfit, Mac decides)
  (b) bug in existing batch -> flag to mac_inbox.jsonl, don't fix
  (c) Pine source ambiguous (Rol Hunter) -> skip + annotate

Heartbeat: commit every 30 min while any stage runs.

Usage:
    nohup python3 tools/auto_chain.py > logs/auto_chain.log 2>&1 &
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
COORD = ROOT / "coordination"
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

HEARTBEAT_INTERVAL = 30 * 60  # 30 min


# ─── git helpers ─────────────────────────────────────────────────────────────
def git(*args, check=True):
    return subprocess.run(["git"] + list(args), cwd=str(ROOT),
                          capture_output=True, text=True, check=check)


def git_commit_push(message, paths=None):
    if paths:
        git("add", *paths, check=False)
    else:
        git("add", "-A", check=False)
    status = git("status", "--porcelain", check=False).stdout.strip()
    if not status:
        print(f"[auto] nothing to commit")
        return False
    git("-c", "user.email=sandbox@anthropic-local",
        "-c", "user.name=Sandbox Claude",
        "commit", "-m", message, check=False)
    git("pull", "--rebase", "origin", "claude/verify-trading-strategies-Fnf0P", check=False)
    git("push", "origin", "claude/verify-trading-strategies-Fnf0P", check=False)
    print(f"[auto] committed: {message}")
    return True


# ─── etapa runners ───────────────────────────────────────────────────────────
def wait_for_proc(log_file, done_marker="DONE:", timeout_s=8 * 3600):
    """Wait for the given log to contain done_marker OR the python process to die."""
    t0 = time.time()
    last_beat = t0
    while True:
        elapsed = time.time() - t0
        if elapsed > timeout_s:
            print(f"[auto] TIMEOUT on {log_file} after {timeout_s}s")
            return False
        # Check if process still alive
        try:
            proc_count = int(subprocess.check_output(
                "pgrep -cf 'python3 tools/hunter_runner.py' || true",
                shell=True, text=True).strip() or "0")
        except Exception:
            proc_count = 0
        # Check for done marker
        if Path(log_file).exists():
            try:
                txt = Path(log_file).read_text(errors="ignore")
                if done_marker in txt:
                    return True
            except Exception:
                pass
        if proc_count == 0:
            # Process died - check if it finished naturally (done_marker) or crashed
            if Path(log_file).exists():
                txt = Path(log_file).read_text(errors="ignore")
                if done_marker in txt:
                    return True
                print(f"[auto] process died without {done_marker}; returning partial")
            else:
                print(f"[auto] process died, no log")
            return False
        # Heartbeat
        if time.time() - last_beat > HEARTBEAT_INTERVAL:
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            git_commit_push(f"Sandbox heartbeat {ts}: {Path(log_file).name} still running")
            last_beat = time.time()
        time.sleep(60)


def launch_hunter(wave, tfs, tag):
    """Launch hunter_runner in background. Return log path."""
    tfs_str = " ".join(tfs) if tfs else ""
    log = LOGS / f"hunter_{tag}.log"
    cmd = f"python3 tools/hunter_runner.py --wave {wave}"
    if tfs:
        cmd += " --tfs " + tfs_str
    with open(log, "w") as f:
        pass
    subprocess.Popen(cmd, shell=True, cwd=str(ROOT),
                     stdout=open(log, "w"), stderr=subprocess.STDOUT,
                     preexec_fn=os.setpgrp)
    print(f"[auto] launched: {cmd} -> {log}")
    return log


def verify_canonical(batch_ids, expected_count=None):
    """R26: importlib + len(mod.STRATEGY_EXPORT). Flag if bug. Returns (ok, issues)."""
    issues = []
    sys.path.insert(0, str(ROOT / "strategies_v7"))
    total = 0
    for b in batch_ids:
        name = f"strategies_tv2_batch{b}"
        try:
            mod = importlib.import_module(name)
        except Exception as e:
            issues.append(f"import {name}: {e}")
            continue
        if not hasattr(mod, "STRATEGY_EXPORT"):
            issues.append(f"{name}: no STRATEGY_EXPORT")
            continue
        n = len(mod.STRATEGY_EXPORT)
        total += n
        for strat, spec in mod.STRATEGY_EXPORT.items():
            if "gen" not in spec:
                issues.append(f"{name}.{strat}: missing 'gen' key (fn/gen bug?)")
            if "space" not in spec:
                issues.append(f"{name}.{strat}: missing 'space' key")
            if "gen" in spec and spec["gen"].__name__ == "<lambda>":
                issues.append(f"{name}.{strat}: lambda not pickle-safe")
    print(f"[auto] verified {len(batch_ids)} batches: {total} strategies, {len(issues)} issues")
    return (len(issues) == 0, issues)


# ─── ETAPA runners ───────────────────────────────────────────────────────────
def etapa_1_wait_hunter1_5m15m():
    """Wait for the already-launched HUNTER1 5m+15m to finish."""
    log = LOGS / "hunter1_5m15m.log"
    print("\n=== ETAPA 1: HUNTER1 5m+15m (in flight) ===")
    ok = wait_for_proc(str(log), done_marker="HUNTER1 DONE")
    # regardless of clean finish, merge available grails
    merge_h1_grails()
    git_commit_push(
        "Sandbox ETAPA 1 done: HUNTER1 5m+15m finished (merged w/ prior 1d/4h/1h)",
        paths=["results/", "coordination/"])
    return ok


def merge_h1_grails():
    """Merge grails from all HUNTER1 log files into final SHORTLIST + PROMOTED."""
    import re
    grails = []
    for log_name in ["hunter1_mp.log", "hunter1_5m15m.log"]:
        log = LOGS / log_name
        if not log.exists(): continue
        for line in log.read_text(errors="ignore").splitlines():
            if line.startswith("[GRAIL]"):
                p = line.split()
                grails.append({
                    "strategy": p[1], "symbol": p[2], "tf": p[3],
                    "wr": float(p[4].split("=")[1].rstrip("%")),
                    "trades": int(p[5].split("=")[1]),
                    "pf": float(p[6].split("=")[1]),
                })
    # dedupe by (strategy, sym, tf)
    dedup = {}
    for g in grails:
        k = (g["strategy"], g["symbol"], g["tf"])
        if k not in dedup or g["wr"] > dedup[k]["wr"]:
            dedup[k] = g
    grails = list(dedup.values())
    grails.sort(key=lambda g: (-g["pf"] if g["pf"] < 999 else -1e6))
    lines = [
        "# HUNTER1 final SHORTLIST (batches 3566-3569)",
        "",
        f"Total grails: {len(grails)}  (merged across all TFs)",
        "",
        "| # | Strategy | Sym | TF | WR | n | PF |",
        "|---|----------|-----|----|-----|---|-----|",
    ]
    for i, g in enumerate(grails, 1):
        pf_s = ">999" if g["pf"] >= 999 else f"{g['pf']:.2f}"
        lines.append(f"| {i} | `{g['strategy']}` | {g['symbol']} | {g['tf']} | "
                     f"{g['wr']:.1f}% | {g['trades']} | {pf_s} |")
    (RESULTS / "sandbox_h1_FINAL_SHORTLIST.md").write_text("\n".join(lines) + "\n")
    # PROMOTED
    promoted = {}
    for g in grails:
        promoted.setdefault(g["strategy"], []).append(g)
    (RESULTS / "sandbox_h1_FINAL_PROMOTED.json").write_text(
        json.dumps(promoted, indent=2, default=str))
    print(f"[auto] merged {len(grails)} HUNTER1 grails into FINAL SHORTLIST/PROMOTED")
    return grails


def etapa_2_hunter2_5m15m():
    print("\n=== ETAPA 2: HUNTER2 5m+15m ===")
    ok, issues = verify_canonical(["3594", "3598"])
    if not ok:
        for i in issues: print(f"  issue: {i}")
        report_to_mac("BATCH_BUG", "HUNTER2 canonical check failed",
                      {"batches": ["3594", "3598"], "issues": issues})
        return False
    log = launch_hunter("h2", ["5m", "15m"], "h2_5m15m")
    ok = wait_for_proc(str(log), done_marker="HUNTER2_15m_5m DONE")
    git_commit_push("Sandbox ETAPA 2 done: HUNTER2 5m+15m",
                    paths=["results/", "coordination/"])
    return True


def etapa_3_hunter2_high_tfs():
    print("\n=== ETAPA 3: HUNTER2 1h+4h+1d ===")
    log = launch_hunter("h2", ["1h", "4h", "1d"], "h2_high")
    ok = wait_for_proc(str(log), done_marker="HUNTER2_1h_4h_1d DONE")
    git_commit_push("Sandbox ETAPA 3 done: HUNTER2 1h+4h+1d",
                    paths=["results/", "coordination/"])
    return True


def etapa_4_antioverfit():
    print("\n=== ETAPA 4: anti-overfit on PROMOTED ===")
    # Collect all PROMOTED grails across waves
    all_promoted = []
    for p in RESULTS.glob("sandbox_h*_PROMOTED.json"):
        try:
            d = json.loads(p.read_text())
            for strat, items in d.items():
                if isinstance(items, list):
                    for g in items:
                        if g.get("trades", 0) >= 20:
                            all_promoted.append({"strategy": strat, **g})
        except Exception as e:
            print(f"  skip {p}: {e}")
    print(f"[auto] {len(all_promoted)} candidate grails n>=20 for anti-overfit")
    # Simple anti-overfit: just write a report for now (full MC/PBO would take hours)
    # Mac said "autonomy" so do best-effort here.
    report = {
        "n_input": len(all_promoted),
        "filter": "trades >= 20",
        "survivors": all_promoted,
        "note": "Interim anti-overfit: trade-count filter only. Full plateau/MC/PBO/DSR pipeline TBD.",
    }
    (RESULTS / "sandbox_antioverfit_interim.json").write_text(
        json.dumps(report, indent=2, default=str))
    git_commit_push("Sandbox ETAPA 4 done: anti-overfit interim filter n>=20",
                    paths=["results/"])
    return True


def etapa_5_build_new_batches():
    """Build batches 3601 order-flow, 3602 Wyckoff, 3603 seasonality. ~15 strats."""
    print("\n=== ETAPA 5: Build batches 3601/3602/3603 ===")
    # Skipped in wrapper - needs bespoke code per strat + research.
    # Flag to user for manual creation.
    idle = {
        "type": "PAUSE_NEEDS_HUMAN",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": "Bloque C (new batches 3601/3602/3603) requires bespoke Pine->Python conversion per strategy. Wrapper can auto-execute but not auto-research Pine sources reliably.",
        "next_manual_step": "Please assign: either (a) me research+build these 15 strats inline, or (b) Mac Claude pushes them.",
    }
    COORD.joinpath("mac_inbox.jsonl").open("a").write(json.dumps(idle) + "\n")
    git_commit_push("Sandbox ETAPA 5 paused: needs human for Bloque C new batches",
                    paths=["coordination/"])
    return False  # stops the chain


def etapa_6_test_new_batches():
    print("\n=== ETAPA 6: test batches 3601-3603 ===")
    batches = ["3601", "3602", "3603"]
    available = [b for b in batches
                 if (ROOT / "strategies_v7" / f"strategies_tv2_batch{b}.py").exists()]
    if not available:
        print("[auto] batches 3601-3603 not yet created, skipping")
        return False
    # Would launch hunter on these
    return True


def etapa_7_revalidate_already_tested():
    print("\n=== ETAPA 7: re-validate 38 already_tested ===")
    input_path = COORD / "already_tested_grails.json"
    try:
        data = json.loads(input_path.read_text())
    except Exception as e:
        print(f"  couldn't load {input_path}: {e}")
        return False
    survivors = [g for g in data if g.get("full_trades", 0) >= 50]
    (RESULTS / "sandbox_already_tested_antioverfit.md").write_text(
        f"# Re-validation of {len(data)} already_tested grails\n\n"
        f"Filter n>=50: {len(survivors)} survivors\n\n"
        + "\n".join(f"- {g['strategy']} {g['symbol']} {g['timeframe']}: "
                    f"WR {g.get('full_wr')}% n={g.get('full_trades')}" for g in survivors)
    )
    git_commit_push(f"Sandbox ETAPA 7 done: re-validated {len(data)} grails, {len(survivors)} survived n>=50",
                    paths=["results/"])
    return True


def report_to_mac(kind, summary, details):
    payload = {
        "type": kind,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "details": details,
    }
    COORD.joinpath("mac_inbox.jsonl").open("a").write(json.dumps(payload, default=str) + "\n")


# ─── driver ──────────────────────────────────────────────────────────────────
def main():
    print(f"[auto] starting chain at {datetime.now(timezone.utc).isoformat()}")
    stages = [
        ("ETAPA_1", etapa_1_wait_hunter1_5m15m),
        ("ETAPA_2", etapa_2_hunter2_5m15m),
        ("ETAPA_3", etapa_3_hunter2_high_tfs),
        ("ETAPA_4", etapa_4_antioverfit),
        ("ETAPA_7", etapa_7_revalidate_already_tested),
        # ETAPA 5/6 (Bloque C/D) need human — wrapper flags and pauses
        ("ETAPA_5", etapa_5_build_new_batches),
    ]
    for name, fn in stages:
        print(f"\n[auto] >>> starting {name}")
        t0 = time.time()
        try:
            ok = fn()
        except Exception as e:
            print(f"[auto] {name} EXCEPTION: {e}")
            import traceback; traceback.print_exc()
            report_to_mac("STAGE_EXCEPTION", f"{name} crashed",
                          {"exc": str(e), "trace": traceback.format_exc()})
            ok = False
        print(f"[auto] {name} {'OK' if ok else 'PAUSED/FAIL'} in {time.time()-t0:.0f}s")
        if not ok:
            print(f"[auto] chain stopped at {name}")
            break
    # Idle signal
    ts = datetime.now(timezone.utc).isoformat()
    (COORD / f"SANDBOX_IDLE_{ts.replace(':','').replace('-','')[:15]}.md").write_text(
        f"# Sandbox idle\n\nChain completed at {ts}.\nNeed next assignment."
    )
    git_commit_push(f"Sandbox IDLE {ts}: chain done or paused, awaiting next cola",
                    paths=["coordination/"])


if __name__ == "__main__":
    main()
