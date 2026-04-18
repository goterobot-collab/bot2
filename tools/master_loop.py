#!/usr/bin/env python3
"""Master loop: drives the 4-day queue. Persists state across crashes/turns.

On invocation:
  1. Load coordination/sandbox_state.json
  2. Find tasks whose status is 'running' but process is dead -> mark as
     'stalled' (will be resumed) or 'done' (if log shows DONE)
  3. Find first 'pending' task; check if its prerequisites are 'done'
  4. Launch up to N parallel running tasks (independent ones)
  5. Wait until current cohort finishes or timeout (default 30 min per call)
  6. Update state, commit, exit
  7. Cron / next conversation turn re-invokes -> picks up next batch

Usage:
    # one-shot drive (commits between tasks):
    python3 tools/master_loop.py

    # show current status only:
    python3 tools/master_loop.py --status

    # force restart a stalled task:
    python3 tools/master_loop.py --restart HUNTER1_5m_15m
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "coordination" / "sandbox_state.json"
HEARTBEAT_INTERVAL = 5 * 60        # 5 min
SLEEP_TICK = 30                    # check every 30s
MAX_CONVERSATION_BUDGET_S = 25 * 60  # exit after 25 min per turn


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text())


def save_state(s: dict):
    s["_updated_at"] = now_iso()
    STATE_PATH.write_text(json.dumps(s, indent=2, default=str) + "\n")


def proc_alive(cmd_pattern: str) -> int:
    """Return number of running processes matching the pattern."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-cf", cmd_pattern], text=True
        ).strip()
        return int(out or 0)
    except subprocess.CalledProcessError:
        return 0


def is_done_marker_in_log(log_path: Path, label: str) -> bool:
    """Check if log contains ' DONE:' (HUNTER) or task-specific marker."""
    if not log_path.exists():
        return False
    try:
        txt = log_path.read_text(errors="ignore")
    except Exception:
        return False
    return f"{label} DONE" in txt or " DONE:" in txt or "FINAL_DONE" in txt


def reconcile_running(state: dict):
    """Mark stalled tasks (running but no process)."""
    for name, t in state["tasks"].items():
        if t.get("status") != "running":
            continue
        cmd = t.get("cmd", "")
        log = ROOT / t.get("log_file", "")
        # special-case for in-flight HUNTER1 5m/15m
        if name == "HUNTER1_5m_15m":
            cmd_match = "python3 tools/hunter_runner.py --wave h1 --tfs 15m 5m"
        else:
            cmd_match = cmd.split(">", 1)[0].strip() if cmd else name
        alive = proc_alive(cmd_match) if cmd_match else 0
        # check DONE marker first
        label = name.split("_")[0] if name.startswith("HUNTER") else name
        if is_done_marker_in_log(log, label):
            t["status"] = "done"
            t["finished_at"] = now_iso()
            print(f"[reconcile] {name} -> done (DONE marker in log)")
        elif alive == 0:
            t["status"] = "stalled"
            t["last_check"] = now_iso()
            print(f"[reconcile] {name} -> stalled (no process)")


def launch(name: str, state: dict) -> bool:
    t = state["tasks"][name]
    cmd = t.get("cmd")
    if not cmd:
        print(f"[launch] {name}: no cmd defined, skipping")
        t["status"] = "skipped"
        return False
    log_file = ROOT / t["log_file"]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    full_cmd = f"nohup {cmd} > {log_file} 2>&1 &"
    subprocess.Popen(full_cmd, shell=True, cwd=str(ROOT),
                     preexec_fn=os.setpgrp)
    t["status"] = "running"
    t["started_at"] = now_iso()
    t["last_check"] = now_iso()
    print(f"[launch] {name}: {cmd}")
    return True


def restart_stalled(name: str, state: dict):
    t = state["tasks"][name]
    if t.get("status") != "stalled":
        return
    t["status"] = "pending"  # so launch picks it up
    print(f"[restart] {name}: marked pending")


def count_running(state: dict) -> int:
    return sum(1 for t in state["tasks"].values() if t.get("status") == "running")


def next_pending(state: dict) -> str | None:
    for name in state["_priority_queue"]:
        t = state["tasks"].get(name)
        if t and t.get("status") == "pending":
            return name
    # fallback: any pending task by insertion order
    for name, t in state["tasks"].items():
        if t.get("status") == "pending":
            return name
    return None


def commit_progress(message: str):
    try:
        subprocess.run(
            ["git", "add", "coordination/sandbox_state.json", "results/", "logs/"],
            cwd=str(ROOT), capture_output=True
        )
        subprocess.run(
            ["git", "-c", "user.email=sandbox@anthropic-local",
             "-c", "user.name=Sandbox Loop",
             "commit", "-m", message],
            cwd=str(ROOT), capture_output=True
        )
        subprocess.run(
            ["git", "pull", "--rebase", "origin",
             "claude/verify-trading-strategies-Fnf0P"],
            cwd=str(ROOT), capture_output=True
        )
        subprocess.run(
            ["git", "push", "origin", "claude/verify-trading-strategies-Fnf0P"],
            cwd=str(ROOT), capture_output=True
        )
        print(f"[commit] {message}")
    except Exception as e:
        print(f"[commit] error: {e}")


def status_table(state: dict):
    print(f"=== sandbox_state @ {state.get('_updated_at')} ===")
    by = {}
    for name, t in state["tasks"].items():
        by.setdefault(t.get("status", "?"), []).append(name)
    for st, names in sorted(by.items()):
        print(f"  [{st}] {len(names)}")
        for n in names:
            print(f"    - {n}")


def drive_one_pass(state: dict):
    """One iteration: reconcile, then launch up to MAX_PARALLEL."""
    reconcile_running(state)
    max_par = state.get("_max_parallel_workers", 4)
    while count_running(state) < max_par:
        nxt = next_pending(state)
        if nxt is None:
            break
        if not launch(nxt, state):
            # mark skipped to avoid infinite loop
            state["tasks"][nxt]["status"] = "skipped"
            continue
        save_state(state)
    save_state(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--restart", help="task name to restart")
    ap.add_argument("--once", action="store_true",
                    help="single drive pass then exit")
    args = ap.parse_args()

    state = load_state()

    if args.status:
        status_table(state)
        return

    if args.restart:
        restart_stalled(args.restart, state)
        save_state(state)
        return

    print(f"[loop] start  budget={MAX_CONVERSATION_BUDGET_S}s")
    t0 = time.time()
    last_commit = t0
    last_advance = t0

    while True:
        elapsed = time.time() - t0
        if elapsed > MAX_CONVERSATION_BUDGET_S:
            print("[loop] conversation budget elapsed, exiting cleanly")
            break

        drive_one_pass(state)

        # progress commit every HEARTBEAT_INTERVAL
        if time.time() - last_commit > HEARTBEAT_INTERVAL:
            commit_progress(f"sandbox loop heartbeat {now_iso()}")
            last_commit = time.time()

        if args.once:
            break

        # if nothing pending and nothing running, we're idle — write SANDBOX_IDLE
        if count_running(state) == 0 and next_pending(state) is None:
            ts = now_iso().replace(":", "").replace("-", "")[:15]
            (ROOT / "coordination" / f"SANDBOX_IDLE_{ts}.md").write_text(
                f"# Sandbox idle @ {now_iso()}\n\nAll tasks done or skipped. "
                f"Awaiting new cola.\n"
            )
            commit_progress(f"sandbox IDLE {ts}: queue exhausted")
            print("[loop] queue exhausted, exiting")
            break

        time.sleep(SLEEP_TICK)

    commit_progress(f"sandbox loop pause {now_iso()}: turn budget done")


if __name__ == "__main__":
    main()
