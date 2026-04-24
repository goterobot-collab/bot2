#!/usr/bin/env python3
"""
Atomic update of coordination/MAC_V8_COVERED_DEDUP_20260421.json via flock.

Mandated by MAC_PARALLEL_SESSION_INSTRUCTIONS.md section 3. Either session
calls this BEFORE persisting its V8 so the dedup set stays consistent.

Usage:
    python3 tools/update_dedup.py --add-source results/mac_m1_1h_4h_1d_PROMOTED.json \\
        --tag MAC_PARALLEL_V8 --session mac

The --add-source JSON must follow the schema of the hunter's promoted_json:
    {strat_name: {best: {sym, tf, wr, trades, pf, params}, all_passing: [...]}}
"""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEDUP = ROOT / "coordination" / "MAC_V8_COVERED_DEDUP_20260421.json"
LOCK = ROOT / "coordination" / "DEDUP.lock"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add-source", required=True,
                    help="Path to promoted JSON from mac_hunter_runner.py")
    ap.add_argument("--tag", default="MAC_PARALLEL_V8",
                    help="Source tag stored in each new entry")
    ap.add_argument("--session", choices=["mac", "sandbox"], default="mac",
                    help="Which subset to append to (v8_mac_covered or sandbox_v8_covered)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_path = Path(args.add_source)
    if not src_path.is_absolute():
        src_path = ROOT / src_path
    if not src_path.exists():
        print(f"ERROR: source file not found: {src_path}", file=sys.stderr)
        sys.exit(2)

    promoted = json.loads(src_path.read_text())
    new_entries = []
    for strat, rec in promoted.items():
        for combo in rec.get("all_passing", []):
            new_entries.append({
                "strategy": strat,
                "symbol": combo.get("sym"),
                "tf": combo.get("tf"),
                "source": args.tag,
                "tier": "MAC_PARALLEL",
                "test_wr": combo.get("wr"),
            })
    print(f"Parsed {len(new_entries)} new entries from {src_path.name}")

    if args.dry_run:
        for e in new_entries[:10]:
            print(f"  + {e['strategy']} | {e['symbol']} | {e['tf']} | WR={e['test_wr']}")
        print("(dry-run: NOT modifying dedup file)")
        return

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "w") as lk:
        fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
        try:
            if DEDUP.exists():
                current = json.loads(DEDUP.read_text())
            else:
                current = {
                    "_doc": "Dedup list",
                    "v8_mac_covered": [],
                    "sandbox_v8_covered": [],
                }
            key = "v8_mac_covered" if args.session == "mac" else "sandbox_v8_covered"
            existing_set = set()
            for t in current.get("v8_mac_covered", []) + current.get("sandbox_v8_covered", []):
                existing_set.add((t.get("strategy"), t.get("symbol"), t.get("tf")))

            added = 0
            for e in new_entries:
                k = (e["strategy"], e["symbol"], e["tf"])
                if k in existing_set:
                    continue
                current.setdefault(key, []).append(e)
                existing_set.add(k)
                added += 1

            current["_last_update"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            current["_last_update_by"] = args.tag
            current["_total_covered"] = (len(current.get("v8_mac_covered", []))
                                         + len(current.get("sandbox_v8_covered", [])))
            DEDUP.write_text(json.dumps(current, indent=2) + "\n")
            print(f"Added {added} new entries to {key} (total covered: {current['_total_covered']})")
        finally:
            fcntl.flock(lk.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
