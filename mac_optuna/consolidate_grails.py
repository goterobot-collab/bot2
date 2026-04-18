#!/usr/bin/env python3
"""
Consolidate R1 + R2 Optuna grails into a master file with ranking.
Creates:
  - grails_master.json   (all grails ranked by composite score)
  - dream_teams.json     (top 5 strategies per symbol)
  - leverage_map_v6.json (safe leverage per symbol from best grail)
"""

import json
import os
from collections import defaultdict

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + "/data"

def load_grails(path, round_label):
    with open(path) as f:
        data = json.load(f)
    grails = data.get("grails", [])
    for g in grails:
        g["round"] = round_label
    return grails

def composite_score(g):
    """Composite: test_wr*0.4 + sharpe*0.3 + pf*0.2 + (1-dd)*0.1"""
    test = g.get("test", {})
    wr = test.get("wr", 0) / 100.0  # normalize to 0-1
    sharpe = min(test.get("sharpe", 0), 50) / 50.0
    pf = min(test.get("pf", 0), 10) / 10.0
    dd = min(test.get("max_dd", 0), 50) / 50.0
    return wr * 0.4 + sharpe * 0.3 + pf * 0.2 + (1 - dd) * 0.1

def main():
    # 1. Load ALL progress files (glob-based — reads all 32 files, not just 3)
    import glob as _glob
    _progress_files = sorted(_glob.glob(f"{DATA_DIR}/*progress*.json"))
    all_grails = []
    for _pf in _progress_files:
        _label = os.path.basename(_pf).replace("optuna_", "").replace("_progress.json", "")
        _loaded = load_grails(_pf, _label)
        if _loaded:
            print(f"  {_label}: {len(_loaded)} grails")
            all_grails.extend(_loaded)
    print(f"Loaded: {len(all_grails)} total grails from {len(_progress_files)} files")

    # 2. Deduplicate by (strategy, symbol, timeframe) — keep higher test WR
    dedup = {}
    dups_found = 0
    for g in all_grails:
        key = (g["strategy"], g["symbol"], g["timeframe"])
        if key in dedup:
            dups_found += 1
            existing_wr = dedup[key].get("test", {}).get("wr", 0)
            new_wr = g.get("test", {}).get("wr", 0)
            if new_wr > existing_wr:
                dedup[key] = g
        else:
            dedup[key] = g

    grails = list(dedup.values())
    print(f"After dedup: {len(grails)} grails ({dups_found} duplicates removed)")

    # 3. Compute composite score and sort
    for g in grails:
        g["_composite"] = composite_score(g)

    grails.sort(key=lambda g: g["_composite"], reverse=True)

    # 4. Build master records
    master = []
    for rank, g in enumerate(grails, 1):
        test = g.get("test", {})
        train = g.get("train", {})
        master.append({
            "rank": rank,
            "strategy": g["strategy"],
            "symbol": g["symbol"],
            "timeframe": g["timeframe"],
            "round": g["round"],
            "best_params": g.get("best_params", {}),
            "test_wr": test.get("wr", 0),
            "test_pnl": test.get("pnl", 0),
            "test_sharpe": test.get("sharpe", 0),
            "test_pf": test.get("pf", 0),
            "test_trades": test.get("trades", 0),
            "test_mae_p95": test.get("mae_p95", 0),
            "test_max_dd": test.get("max_dd", 0),
            "train_wr": train.get("wr", 0),
            "train_pnl": train.get("pnl", 0),
            "wr_diff": g.get("wr_diff", 0),
            "safe_leverage": g.get("safe_leverage", 1),
            "composite_score": round(g["_composite"], 6),
            "weighted_score": g.get("weighted_score", 0),
        })

    # Save master
    master_path = f"{DATA_DIR}/grails_master.json"
    with open(master_path, "w") as f:
        json.dump(master, f, indent=2)
    print(f"Saved: {master_path} ({len(master)} grails)")

    # 5. Dream teams: top 5 per symbol
    by_symbol = defaultdict(list)
    for m in master:
        by_symbol[m["symbol"]].append(m)

    dream_teams = {}
    for sym, entries in sorted(by_symbol.items()):
        top5 = entries[:5]  # already sorted by composite
        dream_teams[sym] = [{
            "rank": e["rank"],
            "strategy": e["strategy"],
            "timeframe": e["timeframe"],
            "test_wr": e["test_wr"],
            "composite_score": e["composite_score"],
            "safe_leverage": e["safe_leverage"],
        } for e in top5]

    dt_path = f"{DATA_DIR}/dream_teams.json"
    with open(dt_path, "w") as f:
        json.dump(dream_teams, f, indent=2)
    print(f"Saved: {dt_path} ({len(dream_teams)} symbols)")

    # 6. Leverage map: best grail per symbol
    leverage_map = {}
    for sym, entries in sorted(by_symbol.items()):
        best = entries[0]
        leverage_map[sym] = {
            "leverage": best["safe_leverage"],
            "mae_p95": best["test_mae_p95"],
            "strategy": best["strategy"],
        }

    lev_path = f"{DATA_DIR}/leverage_map_v6.json"
    with open(lev_path, "w") as f:
        json.dump(leverage_map, f, indent=2)
    print(f"Saved: {lev_path} ({len(leverage_map)} symbols)")

    # 7. Summary stats
    print("\n" + "=" * 60)
    print("SUMMARY STATS")
    print("=" * 60)

    r1_count = sum(1 for m in master if m["round"] == "R1")
    r2_count = sum(1 for m in master if m["round"] == "R2")
    r3_count = sum(1 for m in master if m["round"] == "R3")
    print(f"  R1 grails: {r1_count}")
    print(f"  R2 grails: {r2_count}")
    print(f"  R3 grails: {r3_count}")
    print(f"  Total:     {len(master)}")
    print(f"  Symbols:   {len(by_symbol)}")

    strategies = defaultdict(int)
    for m in master:
        strategies[m["strategy"]] += 1
    print(f"  Strategies: {len(strategies)}")

    wrs = [m["test_wr"] for m in master]
    print(f"\n  Test WR:  min={min(wrs):.1f}%  avg={sum(wrs)/len(wrs):.1f}%  max={max(wrs):.1f}%")

    scores = [m["composite_score"] for m in master]
    print(f"  Score:    min={min(scores):.4f}  avg={sum(scores)/len(scores):.4f}  max={max(scores):.4f}")

    levs = [m["safe_leverage"] for m in master]
    print(f"  Leverage: min={min(levs)}  avg={sum(levs)/len(levs):.1f}  max={max(levs)}")

    # Top 10
    print(f"\n  TOP 10 GRAILS:")
    print(f"  {'Rank':>4} {'Strategy':<20} {'Symbol':<22} {'TF':<4} {'WR':>6} {'Score':>8} {'Lev':>4} {'Rd'}")
    for m in master[:10]:
        print(f"  {m['rank']:>4} {m['strategy']:<20} {m['symbol']:<22} {m['timeframe']:<4} {m['test_wr']:>5.1f}% {m['composite_score']:>8.4f} {m['safe_leverage']:>4} {m['round']}")

    # Top strategies by count
    print(f"\n  TOP 10 STRATEGIES (by grail count):")
    for strat, cnt in sorted(strategies.items(), key=lambda x: -x[1])[:10]:
        avg_wr = sum(m["test_wr"] for m in master if m["strategy"] == strat) / cnt
        print(f"    {strat:<25} {cnt:>4} grails  avg WR={avg_wr:.1f}%")

    # Timeframe distribution
    tf_counts = defaultdict(int)
    for m in master:
        tf_counts[m["timeframe"]] += 1
    print(f"\n  TIMEFRAME DISTRIBUTION:")
    for tf, cnt in sorted(tf_counts.items(), key=lambda x: -x[1]):
        print(f"    {tf:<6} {cnt:>4} grails ({cnt/len(master)*100:.1f}%)")


if __name__ == "__main__":
    main()
