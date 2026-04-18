#!/usr/bin/env python3
"""
V6 Validation Proof — Generates hard evidence that the V6 strategy system works.
Separated from V5 pipeline. Read-only: does NOT modify any existing files.
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
V6_PATH = os.path.join(BASE, "data", "v6_production_final.json")
GRAILS_PATH = os.path.join(BASE, "data", "combined_grails.json")
OUTPUT_PATH = os.path.join(BASE, "data", "v6_validation_proof.json")

# ── helpers ──────────────────────────────────────────────────────────────────

def avg(lst):
    return round(sum(lst) / len(lst), 2) if lst else 0.0

def median(lst):
    if not lst:
        return 0.0
    s = sorted(lst)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return round((s[n // 2 - 1] + s[n // 2]) / 2, 2)

def pct(part, whole):
    return round(part / whole * 100, 1) if whole else 0.0

def load_json(path):
    with open(path) as f:
        return json.load(f)

def hr(char="═", width=80):
    print(char * width)

def section(title):
    print()
    hr()
    print(f"  {title}")
    hr()

# ── tier ordering ────────────────────────────────────────────────────────────

TIER_ORDER = ["ULTIMATE", "DIAMOND", "PLATINUM", "GOLD", "ELITE", "SILVER", "BRONZE"]

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n🔬  V6 VALIDATION PROOF — Crypto & Stock Strategy Analyzer")
    print(f"    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    hr("─")

    # ── Load data ────────────────────────────────────────────────────────────
    v6 = load_json(V6_PATH)
    grails = load_json(GRAILS_PATH)
    print(f"\n  V6 production bots loaded: {len(v6)}")
    print(f"  Combined grails loaded:    {len(grails)}")

    # Build grail lookup: (strategy, symbol, timeframe) -> grail
    grail_map = {}
    for g in grails:
        key = (g["strategy"], g["symbol"], g["timeframe"])
        grail_map[key] = g

    evidence = {
        "generated": datetime.now().isoformat(),
        "v6_bots": len(v6),
        "grails_total": len(grails),
    }

    # ═══════════════════════════════════════════════════════════════════════
    # 1. TIER BREAKDOWN
    # ═══════════════════════════════════════════════════════════════════════
    section("1. TIER BREAKDOWN")

    tier_data = defaultdict(list)
    for bot in v6:
        tier_data[bot["tier"]].append(bot)

    tier_stats = {}
    for tier in TIER_ORDER:
        bots = tier_data.get(tier, [])
        if not bots:
            continue

        symbols = set(b["symbol"] for b in bots)
        strategies = set(b["strategy"] for b in bots)
        test_wrs = [b["test_wr"] for b in bots if b["test_wr"] is not None]
        test_pnls = [b["test_pnl"] for b in bots if b["test_pnl"] is not None]
        full_wrs = [b["full_wr"] for b in bots if b["full_wr"] is not None]
        full_pnls = [b["full_pnl"] for b in bots if b["full_pnl"] is not None]
        trades = [b["full_trades"] for b in bots if b["full_trades"] is not None]
        test_trades = [b["test_trades"] for b in bots if b["test_trades"] is not None]
        wr_diffs = [b["wr_diff"] for b in bots if b["wr_diff"] is not None]

        # Sharpe from grails
        sharpes = []
        for b in bots:
            key = (b["strategy"], b["symbol"], b["timeframe"])
            g = grail_map.get(key)
            if g and "sharpe" in g.get("test", {}):
                sharpes.append(g["test"]["sharpe"])

        stats = {
            "count": len(bots),
            "unique_symbols": len(symbols),
            "unique_strategies": len(strategies),
            "strategies_list": sorted(strategies),
            "test_wr_avg": avg(test_wrs),
            "test_wr_median": median(test_wrs),
            "test_pnl_avg": avg(test_pnls),
            "full_wr_avg": avg(full_wrs),
            "full_pnl_avg": avg(full_pnls),
            "sharpe_avg": avg(sharpes),
            "trades_min": min(trades) if trades else 0,
            "trades_avg": avg(trades),
            "trades_max": max(trades) if trades else 0,
            "test_trades_min": min(test_trades) if test_trades else 0,
            "test_trades_avg": avg(test_trades),
            "test_trades_max": max(test_trades) if test_trades else 0,
            "wr_diff_avg": avg(wr_diffs),
        }
        tier_stats[tier] = stats

        print(f"\n  ▸ {tier} ({stats['count']} bots)")
        print(f"    Symbols: {stats['unique_symbols']}  |  Strategies: {stats['unique_strategies']} {stats['strategies_list']}")
        print(f"    Test WR:  avg={stats['test_wr_avg']}%  median={stats['test_wr_median']}%")
        print(f"    Test PnL: avg={stats['test_pnl_avg']}%")
        print(f"    Full WR:  avg={stats['full_wr_avg']}%  |  Full PnL: avg={stats['full_pnl_avg']}%")
        print(f"    Sharpe:   avg={stats['sharpe_avg']}")
        print(f"    Full trades:  min={stats['trades_min']}  avg={stats['trades_avg']}  max={stats['trades_max']}")
        print(f"    Test trades:  min={stats['test_trades_min']}  avg={stats['test_trades_avg']}  max={stats['test_trades_max']}")
        print(f"    WR diff (full-test): avg={stats['wr_diff_avg']}%")

    evidence["tier_stats"] = tier_stats

    # ═══════════════════════════════════════════════════════════════════════
    # 2. TEMPORAL CONSISTENCY PROOF (top 50 bots)
    # ═══════════════════════════════════════════════════════════════════════
    section("2. TEMPORAL CONSISTENCY PROOF (Top 50 bots)")

    # Sort by quality_score desc
    top50 = sorted(v6, key=lambda b: b.get("quality_score", 0), reverse=True)[:50]

    consistency_results = []
    multi_year_pass = 0
    single_year_warn = 0

    for bot in top50:
        key = (bot["strategy"], bot["symbol"], bot["timeframe"])
        g = grail_map.get(key)
        yearly = g["test"]["yearly"] if g and "yearly" in g.get("test", {}) else {}

        years_above_70 = []
        years_data = {}
        for year, yd in sorted(yearly.items()):
            wr = yd.get("wr", 0)
            trades = yd.get("trades", 0)
            pnl = yd.get("pnl", 0)
            years_data[year] = {"wr": wr, "trades": trades, "pnl": pnl}
            if wr >= 70 and trades >= 2:
                years_above_70.append(year)

        is_consistent = len(years_above_70) >= 2
        if is_consistent:
            multi_year_pass += 1
            verdict = "CONSISTENT"
        elif len(yearly) <= 1:
            single_year_warn += 1
            verdict = "LIMITED_DATA"
        else:
            single_year_warn += 1
            verdict = "OVERFIT_WARNING"

        entry = {
            "rank": bot["rank"],
            "tier": bot["tier"],
            "strategy": bot["strategy"],
            "symbol": bot["symbol"],
            "timeframe": bot["timeframe"],
            "quality_score": bot["quality_score"],
            "yearly": years_data,
            "years_wr_70+": years_above_70,
            "verdict": verdict,
        }
        consistency_results.append(entry)

    print(f"\n  Multi-year consistent (WR>=70% in 2+ years): {multi_year_pass}/50 ({pct(multi_year_pass, 50)}%)")
    print(f"  Single-year / limited data warnings:          {single_year_warn}/50 ({pct(single_year_warn, 50)}%)")

    print(f"\n  {'Rank':<6}{'Tier':<12}{'Strategy':<15}{'Symbol':<25}{'TF':<6}{'Years WR>70%':<20}{'Verdict'}")
    print("  " + "─" * 95)
    for r in consistency_results[:25]:
        yrs = ", ".join(r["years_wr_70+"]) if r["years_wr_70+"] else "—"
        print(f"  {r['rank']:<6}{r['tier']:<12}{r['strategy']:<15}{r['symbol']:<25}{r['timeframe']:<6}{yrs:<20}{r['verdict']}")

    if len(consistency_results) > 25:
        print(f"  ... ({len(consistency_results) - 25} more)")

    evidence["temporal_consistency"] = {
        "top50_multi_year_pass": multi_year_pass,
        "top50_single_year_warn": single_year_warn,
        "pass_rate": pct(multi_year_pass, 50),
        "details": consistency_results,
    }

    # ═══════════════════════════════════════════════════════════════════════
    # 3. WALK-FORWARD PROOF (train vs test gap)
    # ═══════════════════════════════════════════════════════════════════════
    section("3. WALK-FORWARD PROOF (Train vs Test WR gap)")

    wf_by_tier = {}
    for tier in TIER_ORDER:
        bots = tier_data.get(tier, [])
        if not bots:
            continue
        gaps = []
        train_wrs = []
        test_wrs_list = []
        for b in bots:
            key = (b["strategy"], b["symbol"], b["timeframe"])
            g = grail_map.get(key)
            if g:
                tw = g["train"]["wr"]
                tsw = g["test"]["wr"]
                gap = tw - tsw
                gaps.append(gap)
                train_wrs.append(tw)
                test_wrs_list.append(tsw)

        if not gaps:
            continue

        avg_gap = avg(gaps)
        med_gap = median(gaps)
        max_gap = round(max(gaps), 1)
        min_gap = round(min(gaps), 1)
        overfit_count = sum(1 for g in gaps if g > 25)

        if avg_gap < 15:
            verdict = "ROBUSTO"
        elif avg_gap < 25:
            verdict = "ACEPTABLE"
        else:
            verdict = "OVERFIT_WARNING"

        wf_stats = {
            "matched_bots": len(gaps),
            "train_wr_avg": avg(train_wrs),
            "test_wr_avg": avg(test_wrs_list),
            "gap_avg": avg_gap,
            "gap_median": med_gap,
            "gap_max": max_gap,
            "gap_min": min_gap,
            "overfit_count_gt25": overfit_count,
            "verdict": verdict,
        }
        wf_by_tier[tier] = wf_stats

        sym = "✅" if verdict == "ROBUSTO" else ("⚠️" if verdict == "ACEPTABLE" else "❌")
        print(f"\n  ▸ {tier} ({len(gaps)} bots matched to grails)")
        print(f"    Train WR avg: {avg(train_wrs)}%  |  Test WR avg: {avg(test_wrs_list)}%")
        print(f"    Gap avg: {avg_gap}%  |  median: {med_gap}%  |  range: [{min_gap}%, {max_gap}%]")
        print(f"    Bots con gap > 25%: {overfit_count}/{len(gaps)}")
        print(f"    Veredicto: {sym} {verdict}")

    evidence["walk_forward"] = wf_by_tier

    # ═══════════════════════════════════════════════════════════════════════
    # 4. DIVERSIFICATION PROOF
    # ═══════════════════════════════════════════════════════════════════════
    section("4. DIVERSIFICATION PROOF")

    all_symbols = set(b["symbol"] for b in v6)
    all_strategies = set(b["strategy"] for b in v6)
    all_tfs = set(b["timeframe"] for b in v6)

    # categorize symbols
    crypto_symbols = [s for s in all_symbols if "/" in s or "-SWAP" in s or "-USDT" in s]
    stock_symbols = [s for s in all_symbols if s not in crypto_symbols]

    # Strategy frequency
    strat_counts = defaultdict(int)
    for b in v6:
        strat_counts[b["strategy"]] += 1
    strat_sorted = sorted(strat_counts.items(), key=lambda x: -x[1])

    # Symbol frequency
    sym_counts = defaultdict(int)
    for b in v6:
        sym_counts[b["symbol"]] += 1
    sym_sorted = sorted(sym_counts.items(), key=lambda x: -x[1])

    # Timeframe frequency
    tf_counts = defaultdict(int)
    for b in v6:
        tf_counts[b["timeframe"]] += 1
    tf_sorted = sorted(tf_counts.items(), key=lambda x: -x[1])

    # Coverage: how many symbols have 3+ strategies
    symbols_multi_strat = sum(1 for s, c in sym_counts.items() if c >= 3)

    diversification = {
        "unique_symbols": len(all_symbols),
        "crypto_symbols": len(crypto_symbols),
        "stock_symbols": len(stock_symbols),
        "unique_strategies": len(all_strategies),
        "unique_timeframes": len(all_tfs),
        "timeframes": dict(tf_sorted),
        "top_strategies": dict(strat_sorted[:15]),
        "top_symbols": dict(sym_sorted[:15]),
        "symbols_with_3plus_strategies": symbols_multi_strat,
    }

    print(f"\n  Activos cubiertos:     {len(all_symbols)} ({len(crypto_symbols)} crypto, {len(stock_symbols)} stock)")
    print(f"  Estrategias distintas: {len(all_strategies)}")
    print(f"  Timeframes:            {len(all_tfs)} → {dict(tf_sorted)}")
    print(f"  Symbols con 3+ strats: {symbols_multi_strat}")
    print(f"\n  Top 10 estrategias por uso:")
    for name, cnt in strat_sorted[:10]:
        bar = "█" * (cnt // 5)
        print(f"    {name:<20} {cnt:>4} bots  {bar}")
    print(f"\n  Top 10 symbols por cobertura:")
    for name, cnt in sym_sorted[:10]:
        print(f"    {name:<30} {cnt:>3} bots")

    evidence["diversification"] = diversification

    # ═══════════════════════════════════════════════════════════════════════
    # 5. V6 vs V5 COMPARISON
    # ═══════════════════════════════════════════════════════════════════════
    section("5. V6 vs V5 COMPARISON")

    # V5 reference values (hardcoded from user)
    v5_ref = {
        "wr_avg": 61.7,
        "pf_avg": 1.78,
        "bots": 78,
        "symbols": 399,
    }

    # V6 equivalents
    v6_test_wrs = [b["test_wr"] for b in v6 if b["test_wr"] is not None]
    v6_full_wrs = [b["full_wr"] for b in v6 if b["full_wr"] is not None]

    # PF from grails
    v6_pfs = []
    for b in v6:
        key = (b["strategy"], b["symbol"], b["timeframe"])
        g = grail_map.get(key)
        if g and "pf" in g.get("test", {}):
            v6_pfs.append(g["test"]["pf"])

    v6_stats = {
        "test_wr_avg": avg(v6_test_wrs),
        "full_wr_avg": avg(v6_full_wrs),
        "pf_avg": avg(v6_pfs),
        "bots": len(v6),
        "symbols": len(all_symbols),
    }

    comparison = {
        "v5": v5_ref,
        "v6": v6_stats,
        "improvements": {},
    }

    def comp(label, v6_val, v5_val, higher_is_better=True):
        diff = v6_val - v5_val
        pct_chg = round(diff / v5_val * 100, 1) if v5_val else 0
        win = (diff > 0) == higher_is_better
        sym = "▲" if win else "▼"
        comparison["improvements"][label] = {
            "v5": v5_val, "v6": v6_val, "diff": round(diff, 2),
            "pct_change": pct_chg, "v6_wins": win
        }
        return f"  {label:<25} V5: {v5_val:<10} V6: {v6_val:<10} {sym} {'+' if diff >= 0 else ''}{round(diff, 2)} ({'+' if pct_chg >= 0 else ''}{pct_chg}%)"

    print()
    print(comp("Test WR avg (%)", v6_stats["test_wr_avg"], v5_ref["wr_avg"]))
    print(comp("Profit Factor avg", v6_stats["pf_avg"], v5_ref["pf_avg"]))
    print(comp("Total bots", v6_stats["bots"], v5_ref["bots"]))
    print(comp("Unique symbols", v6_stats["symbols"], v5_ref["symbols"]))

    v6_wins = sum(1 for v in comparison["improvements"].values() if v["v6_wins"])
    total_metrics = len(comparison["improvements"])
    print(f"\n  Resultado: V6 gana en {v6_wins}/{total_metrics} metricas")

    evidence["comparison_v6_v5"] = comparison

    # ═══════════════════════════════════════════════════════════════════════
    # 6. QUALITY DISTRIBUTION
    # ═══════════════════════════════════════════════════════════════════════
    section("6. QUALITY SCORE DISTRIBUTION")

    scores = sorted([b["quality_score"] for b in v6 if b.get("quality_score")], reverse=True)
    brackets = [(90, 100), (80, 90), (70, 80), (60, 70), (50, 60), (0, 50)]
    score_dist = {}
    for lo, hi in brackets:
        cnt = sum(1 for s in scores if lo <= s < hi) if hi < 100 else sum(1 for s in scores if lo <= s <= hi)
        label = f"{lo}-{hi}"
        score_dist[label] = cnt
        bar = "█" * (cnt // 5)
        print(f"  {label:<10} {cnt:>4} bots  {bar}")

    print(f"\n  Total: {len(scores)} bots  |  Avg score: {avg(scores)}  |  Median: {median(scores)}")

    evidence["quality_distribution"] = {
        "brackets": score_dist,
        "avg": avg(scores),
        "median": median(scores),
        "top10": scores[:10],
    }

    # ═══════════════════════════════════════════════════════════════════════
    # 7. RISK METRICS
    # ═══════════════════════════════════════════════════════════════════════
    section("7. RISK METRICS SUMMARY")

    leverages = [b["final_leverage"] for b in v6 if b.get("final_leverage")]
    stop_losses = [b["final_stop_loss_pct"] for b in v6 if b.get("final_stop_loss_pct")]
    kellys = [b["final_kelly_pct"] for b in v6 if b.get("final_kelly_pct")]
    kill_dds = [b["final_kill_dd_pct"] for b in v6 if b.get("final_kill_dd_pct")]

    risk = {
        "leverage": {"avg": avg(leverages), "median": median(leverages), "max": max(leverages) if leverages else 0},
        "stop_loss": {"avg": avg(stop_losses), "median": median(stop_losses)},
        "kelly": {"avg": avg(kellys), "median": median(kellys)},
        "kill_dd": {"avg": avg(kill_dds), "median": median(kill_dds)},
    }

    print(f"  Leverage:    avg={risk['leverage']['avg']}x  median={risk['leverage']['median']}x  max={risk['leverage']['max']}x")
    print(f"  Stop Loss:   avg={risk['stop_loss']['avg']}%  median={risk['stop_loss']['median']}%")
    print(f"  Kelly Size:  avg={risk['kelly']['avg']}%  median={risk['kelly']['median']}%")
    print(f"  Kill DD:     avg={risk['kill_dd']['avg']}%  median={risk['kill_dd']['median']}%")

    evidence["risk_metrics"] = risk

    # ═══════════════════════════════════════════════════════════════════════
    # 8. OVERALL VERDICT
    # ═══════════════════════════════════════════════════════════════════════
    section("8. OVERALL VERDICT")

    checks = []

    # Check 1: WR
    global_test_wr = avg(v6_test_wrs)
    c1 = global_test_wr >= 70
    checks.append(("Test WR avg >= 70%", global_test_wr, c1))

    # Check 2: Walk-forward robustness
    all_gaps = []
    for tier, wf in wf_by_tier.items():
        all_gaps.extend([wf["gap_avg"]])
    global_gap = avg(all_gaps)
    c2 = global_gap < 20
    checks.append(("Walk-forward gap avg < 20%", global_gap, c2))

    # Check 3: Diversification
    c3 = len(all_symbols) >= 100
    checks.append(("100+ unique symbols", len(all_symbols), c3))

    # Check 4: Strategy diversity
    c4 = len(all_strategies) >= 10
    checks.append(("10+ unique strategies", len(all_strategies), c4))

    # Check 5: Scale
    c5 = len(v6) >= 500
    checks.append(("500+ production bots", len(v6), c5))

    # Check 6: Multi-year consistency
    c6 = multi_year_pass >= 25
    checks.append(("50%+ top50 multi-year consistent", multi_year_pass, c6))

    passed = sum(1 for _, _, c in checks if c)

    for label, val, ok in checks:
        sym = "PASS" if ok else "FAIL"
        print(f"  [{sym}] {label:<45} (valor: {val})")

    print(f"\n  Resultado: {passed}/{len(checks)} checks pasaron")
    if passed == len(checks):
        print("  ╔══════════════════════════════════════════╗")
        print("  ║  SISTEMA V6 VALIDADO — TODAS LAS PRUEBAS PASAN  ║")
        print("  ╚══════════════════════════════════════════╝")
    elif passed >= len(checks) - 1:
        print("  → Sistema V6 MAYORMENTE validado (1 check fallido)")
    else:
        print(f"  → Sistema V6 necesita revision ({len(checks) - passed} checks fallidos)")

    evidence["overall_verdict"] = {
        "checks": [{"label": l, "value": v, "passed": p} for l, v, p in checks],
        "passed": passed,
        "total": len(checks),
        "final_verdict": "VALIDATED" if passed == len(checks) else ("MOSTLY_VALID" if passed >= len(checks) - 1 else "NEEDS_REVIEW"),
    }

    # ═══════════════════════════════════════════════════════════════════════
    # SAVE EVIDENCE
    # ═══════════════════════════════════════════════════════════════════════
    with open(OUTPUT_PATH, "w") as f:
        json.dump(evidence, f, indent=2, default=str)
    print(f"\n  Evidencia guardada en: {OUTPUT_PATH}")
    print(f"  Tamaño: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")
    hr("─")
    print()


if __name__ == "__main__":
    main()
