#!/usr/bin/env python3
"""
Regla MAX HARD Enforcer (Condición FBI V8 #4).

Purpose:
  When V8 upgrades sizing, NEVER downgrade.
  Only UPGRADE or NO_CHANGE. DOWNGRADE → BLOCKED, final = current.

FBI Approval: 3/3 (CEREBRO + AUDITOR + TRADE_LOSS), 2026-04-15.
See: BUZON/FBI_V8_SYNTHESIS_20260415_PARA_SIGNAL_AUDITOR.md L30
See: BUZON/SPECS_V8_DELIVERABLES_20260415.md §8
See: BUZON/SPEC_REGLA_MAX_HARD_V8_20260415.md

Reconcile rules (COORD decision #4, 2026-04-15):
  - MAX rule applies regardless of current 'source' field
    (forensic / default / hand_tuned / unknown all treated same)
  - If current size_usd missing → treat as upgrade (any proposed ≥1 passes)
  - If bot NOT in current JSON → INSERT (MAX does not block new bots)
  - Invalid proposed size (non-numeric, negative, >50) → INVALID_INPUT (no write)

Usage:
  # Dry-run (default, safe)
  python3 regla_max_enforcer.py \\
    --json live/data/v6_optimized_sl_tp.json \\
    --upgrade-plan logs/v8_upgrade_plan.json \\
    --output logs/v8_upgrade_applied.json

  # Apply (requires env guard):
  REGLA_MAX_APPLY_CONFIRMED=1 python3 regla_max_enforcer.py \\
    --json ... --upgrade-plan ... --output ... --apply

Exit codes:
  0 = success
  1 = input validation failed
  2 = postcondition failed (some bot downgraded)
  3 = --apply requested without env guard
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---- Canonical tier map ----
# FBI-approved 3/3 (2026-04-15): tier-based sizing in USD
# DIAMOND=$15, GOLD=$12, PLATINUM=$10, ELITE=$7, SILVER=$5
TIER_SIZES: dict[str, int] = {
    "DIAMOND": 15,
    "GOLD": 12,
    "PLATINUM": 10,
    "ELITE": 7,
    "SILVER": 5,
}

# ---- Actions ----
ACTION_UPGRADE = "UPGRADE"
ACTION_NO_CHANGE = "NO_CHANGE"
ACTION_DOWNGRADE_BLOCKED = "DOWNGRADE_BLOCKED"
ACTION_INSERT = "INSERT"
ACTION_INVALID_INPUT = "INVALID_INPUT"

# ---- Limits ----
MIN_SIZE_USD = 1
MAX_SIZE_USD = 50


# ---- Helpers ----

def make_key(bot: dict) -> str:
    """Generate canonical bot_key from strategy|symbol|timeframe."""
    return f"{bot.get('strategy', '?')}|{bot.get('symbol', '?')}|{bot.get('timeframe', '?')}"


def _validate_plan_entry(entry: dict) -> tuple[bool, str]:
    """Return (is_valid, reason_if_invalid)."""
    key = entry.get("bot_key")
    if not key:
        return False, "missing bot_key"
    size = entry.get("new_size_usd")
    if size is None or not isinstance(size, (int, float)) or isinstance(size, bool):
        return False, "proposed size missing or non-numeric"
    if size < MIN_SIZE_USD or size > MAX_SIZE_USD:
        return False, f"proposed size {size} out of range [{MIN_SIZE_USD}, {MAX_SIZE_USD}]"
    return True, ""


# ---- Core logic ----

def enforce_max(
    current_bots: list[dict],
    upgrade_plan: list[dict],
) -> dict[str, Any]:
    """Pure function: apply Regla MAX HARD. No I/O, no mutations on inputs.

    Args:
        current_bots: list of bot dicts from v6_optimized_sl_tp.json
        upgrade_plan: list of {bot_key, new_tier, new_size_usd}

    Returns:
        dict with counts + per-bot details
    """
    by_key: dict[str, dict] = {}
    for b in current_bots:
        k = b.get("bot_key") or make_key(b)
        by_key[k] = b

    counts = {
        ACTION_UPGRADE: 0,
        ACTION_NO_CHANGE: 0,
        ACTION_DOWNGRADE_BLOCKED: 0,
        ACTION_INSERT: 0,
        ACTION_INVALID_INPUT: 0,
    }
    details: list[dict] = []

    for entry in upgrade_plan:
        ok, reason = _validate_plan_entry(entry)
        if not ok:
            details.append({
                "bot": entry.get("bot_key"),
                "action": ACTION_INVALID_INPUT,
                "reason": reason,
            })
            counts[ACTION_INVALID_INPUT] += 1
            continue

        bot_key = entry["bot_key"]
        proposed_size = entry["new_size_usd"]
        proposed_tier = entry.get("new_tier")

        current = by_key.get(bot_key)

        if current is None:
            details.append({
                "bot": bot_key,
                "current": None,
                "proposed": {"tier": proposed_tier, "size_usd": proposed_size},
                "final": {"tier": proposed_tier, "size_usd": proposed_size},
                "action": ACTION_INSERT,
            })
            counts[ACTION_INSERT] += 1
            continue

        current_size = current.get("size_usd")
        current_tier = current.get("tier")
        current_source = current.get("source", "unknown")

        if current_size is None:
            details.append({
                "bot": bot_key,
                "current": {"tier": current_tier, "size_usd": None, "source": current_source},
                "proposed": {"tier": proposed_tier, "size_usd": proposed_size},
                "final": {"tier": proposed_tier, "size_usd": proposed_size},
                "action": ACTION_UPGRADE,
                "reason": "current size_usd missing",
            })
            counts[ACTION_UPGRADE] += 1
            continue

        # Core MAX rule
        if proposed_size > current_size:
            action = ACTION_UPGRADE
            final = {"tier": proposed_tier, "size_usd": proposed_size}
        elif proposed_size < current_size:
            action = ACTION_DOWNGRADE_BLOCKED
            final = {"tier": current_tier, "size_usd": current_size}
        else:
            action = ACTION_NO_CHANGE
            final = {"tier": current_tier, "size_usd": current_size}

        details.append({
            "bot": bot_key,
            "current": {"tier": current_tier, "size_usd": current_size, "source": current_source},
            "proposed": {"tier": proposed_tier, "size_usd": proposed_size},
            "final": final,
            "action": action,
        })
        counts[action] += 1

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_bots_in_plan": len(upgrade_plan),
        "upgrades_applied": counts[ACTION_UPGRADE],
        "downgrades_blocked": counts[ACTION_DOWNGRADE_BLOCKED],
        "no_change": counts[ACTION_NO_CHANGE],
        "inserts": counts[ACTION_INSERT],
        "invalid_inputs": counts[ACTION_INVALID_INPUT],
        "details": details,
    }


def validate_postconditions(results: dict[str, Any]) -> list[str]:
    """Verify invariants after enforcement. Returns list of errors (empty → OK)."""
    errors: list[str] = []
    for r in results.get("details", []):
        action = r.get("action")
        current = (r.get("current") or {}).get("size_usd")
        final = (r.get("final") or {}).get("size_usd")

        if action == ACTION_UPGRADE:
            if current is not None and final is not None and final <= current:
                errors.append(
                    f"BUG: {r.get('bot')} marked UPGRADE but final={final} <= current={current}"
                )
        elif action == ACTION_DOWNGRADE_BLOCKED:
            if current != final:
                errors.append(
                    f"BUG: {r.get('bot')} DOWNGRADE_BLOCKED but final={final} != current={current}"
                )
        elif action == ACTION_NO_CHANGE:
            if current != final:
                errors.append(
                    f"BUG: {r.get('bot')} NO_CHANGE but final={final} != current={current}"
                )
    return errors


# ---- I/O ----

def load_current_bots(path: Path) -> list[dict]:
    """Load bots list from v6_optimized_sl_tp.json (supports dict or list)."""
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "bots" in data and isinstance(data["bots"], list):
            return data["bots"]
        # dict keyed by bot_key
        out = []
        for k, v in data.items():
            if isinstance(v, dict):
                if "bot_key" not in v:
                    v = dict(v)
                    v["bot_key"] = k
                out.append(v)
        return out
    raise ValueError(f"Unrecognized JSON schema in {path}")


def load_upgrade_plan(path: Path) -> list[dict]:
    """Load upgrade plan (list or {upgrades: [...]} or {plan: [...]})."""
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("upgrades", "plan", "bots"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"Unrecognized upgrade plan schema in {path}")


# ---- CLI ----

def main() -> int:
    parser = argparse.ArgumentParser(description="Regla MAX HARD Enforcer (FBI V8 #4)")
    parser.add_argument("--json", required=True, help="Current v6_optimized_sl_tp.json")
    parser.add_argument("--upgrade-plan", required=True, help="Upgrade plan JSON")
    parser.add_argument("--output", required=True, help="Output applied JSON")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Dry-run mode (default). No destructive writes.")
    parser.add_argument("--apply", action="store_true",
                        help="Apply. Requires REGLA_MAX_APPLY_CONFIRMED=1 env var.")
    args = parser.parse_args()

    if args.apply:
        if os.environ.get("REGLA_MAX_APPLY_CONFIRMED") != "1":
            print(
                "ERROR: --apply requires env REGLA_MAX_APPLY_CONFIRMED=1 "
                "(defense in depth, FBI V8).",
                file=sys.stderr,
            )
            return 3
        args.dry_run = False

    current_path = Path(args.json)
    plan_path = Path(args.upgrade_plan)
    out_path = Path(args.output)

    if not current_path.exists():
        print(f"ERROR: {current_path} not found", file=sys.stderr)
        return 1
    if not plan_path.exists():
        print(f"ERROR: {plan_path} not found", file=sys.stderr)
        return 1

    print(f"[1/4] Loading current JSON: {current_path.name}")
    current_bots = load_current_bots(current_path)
    print(f"      {len(current_bots)} bots in current JSON")

    print(f"[2/4] Loading upgrade plan: {plan_path.name}")
    plan = load_upgrade_plan(plan_path)
    print(f"      {len(plan)} entries in upgrade plan")

    print("[3/4] Applying Regla MAX HARD...")
    results = enforce_max(current_bots, plan)

    print("[4/4] Validating postconditions...")
    errors = validate_postconditions(results)
    if errors:
        for e in errors[:10]:
            print(f"  POSTCOND FAILED: {e}", file=sys.stderr)
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print()
    print("=== SUMMARY ===")
    print(f"  UPGRADES applied:   {results['upgrades_applied']}")
    print(f"  DOWNGRADES blocked: {results['downgrades_blocked']}")
    print(f"  NO_CHANGE:          {results['no_change']}")
    print(f"  INSERTS:            {results['inserts']}")
    print(f"  INVALID inputs:     {results['invalid_inputs']}")
    print(f"  Output: {out_path}")
    print(f"  Mode: {'APPLIED' if args.apply else 'DRY-RUN'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
