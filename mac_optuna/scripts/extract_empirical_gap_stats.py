#!/usr/bin/env python3
"""
Extrae estadisticas empiricas del forensic V2 (23,693 grails) para:
1. Input empirico a /consejo-ia (multi-IA)
2. Input empirico a WebSearch
3. Baseline del experimento V7 vs V8

Foco: real_wr (forensic) vs optuna_wr (prometido). Gap = optuna_wr - real_wr.
Regla 24: gap>10pp = INFLATED. Gate prod: real_wr>=65% AND gap<=10pp.

Output:
- CSV con stats por segmento
- Markdown con tablas listas para pegar en queries
- JSON con top-line summary
"""
import json
import statistics
from pathlib import Path
from collections import defaultdict

FORENSIC = Path("/Users/sabrina/CLAUDE CODE/Estrategias/data/forensic_v2_hetzner_23K_LIGHT.json")
OUT_DIR = Path("/Users/sabrina/CLAUDE CODE/Estrategias/data/empirical_analysis_20260415")
OUT_DIR.mkdir(exist_ok=True, parents=True)


def bucket_wr(wr):
    if wr >= 90: return "90-100"
    if wr >= 80: return "80-90"
    if wr >= 70: return "70-80"
    if wr >= 65: return "65-70"
    if wr >= 60: return "60-65"
    return "<60"

def bucket_pf(pf):
    if pf is None or pf != pf: return "NA"
    if pf >= 3.0: return ">=3.0"
    if pf >= 2.0: return "2.0-3.0"
    if pf >= 1.5: return "1.5-2.0"
    if pf >= 1.0: return "1.0-1.5"
    return "<1.0"

def bucket_sharpe(s):
    if s is None or s != s: return "NA"
    if s >= 3.0: return ">=3.0"
    if s >= 2.0: return "2.0-3.0"
    if s >= 1.0: return "1.0-2.0"
    if s >= 0.5: return "0.5-1.0"
    if s >= 0.0: return "0.0-0.5"
    return "<0.0"

def pct(num, denom):
    return (num / denom * 100) if denom else 0.0

def quantile(values, q):
    if not values: return None
    s = sorted(values)
    idx = int(len(s) * q)
    idx = max(0, min(idx, len(s) - 1))
    return s[idx]


def main():
    print(f"Reading {FORENSIC}...")
    with open(FORENSIC) as f:
        data = json.load(f)

    results = data["results"]
    print(f"Total records: {len(results):,}")

    # Enriquecer con gap y filtrar sample insuficiente
    # Regla 24: min 100 trades para que el forensic sea estadisticamente valido
    rows = []
    skipped_low_n = 0
    for r in results:
        m = r.get("metrics", {})
        n = m.get("total_trades", 0) or 0
        if n < 30:  # bajamos a 30 para no perder demasiado; flageamos ≥100 aparte
            skipped_low_n += 1
            continue
        optuna_wr = r.get("optuna_wr")
        real_wr = m.get("win_rate")
        if optuna_wr is None or real_wr is None:
            continue
        gap = optuna_wr - real_wr
        rows.append({
            "strategy": r.get("strategy"),
            "symbol": r.get("symbol"),
            "tf": r.get("timeframe"),
            "optuna_wr": optuna_wr,
            "real_wr": real_wr,
            "gap": gap,
            "abs_gap": abs(gap),
            "n_trades": n,
            "pnl_pct": m.get("total_pnl_pct"),
            "sharpe": m.get("sharpe"),
            "pf": m.get("profit_factor"),
            "liquidity": m.get("liquidity_tier"),
            "long_wr": m.get("long_wr"),
            "short_wr": m.get("short_wr"),
            "long_trades": m.get("long_trades", 0),
            "short_trades": m.get("short_trades", 0),
            "avg_mae": m.get("avg_mae_pct"),
            "avg_mfe": m.get("avg_mfe_pct"),
            "fees_pct": m.get("total_fees_pct"),
            "sl_hits": m.get("sl_hits", 0),
            "signal_exits": m.get("signal_exits", 0),
        })

    print(f"Usable (>=30 trades): {len(rows):,}  |  Skipped (<30 trades): {skipped_low_n:,}")

    # === TOP LINE ===
    gaps = [r["gap"] for r in rows]
    real_wrs = [r["real_wr"] for r in rows]
    pnls = [r["pnl_pct"] for r in rows if r["pnl_pct"] is not None]

    topline = {
        "total_records_forensic": len(results),
        "usable_min_30_trades": len(rows),
        "gap_median_pp": statistics.median(gaps),
        "gap_mean_pp": statistics.mean(gaps),
        "gap_stdev_pp": statistics.stdev(gaps),
        "gap_p10_pp": quantile(gaps, 0.10),
        "gap_p25_pp": quantile(gaps, 0.25),
        "gap_p75_pp": quantile(gaps, 0.75),
        "gap_p90_pp": quantile(gaps, 0.90),
        "real_wr_median_pct": statistics.median(real_wrs),
        "real_wr_mean_pct": statistics.mean(real_wrs),
        "pct_gap_le_5pp": pct(sum(1 for g in gaps if abs(g) <= 5), len(gaps)),
        "pct_gap_le_10pp": pct(sum(1 for g in gaps if abs(g) <= 10), len(gaps)),
        "pct_real_wr_ge_65": pct(sum(1 for w in real_wrs if w >= 65), len(real_wrs)),
        "pct_real_wr_ge_70": pct(sum(1 for w in real_wrs if w >= 70), len(real_wrs)),
        "pct_pnl_positive": pct(sum(1 for p in pnls if p > 0), len(pnls)),
        "pct_both_gates_pass": pct(sum(1 for r in rows if abs(r["gap"]) <= 10 and r["real_wr"] >= 65 and (r["pnl_pct"] or 0) > 0), len(rows)),
    }

    # === SEGMENTOS ===
    def stats_for_subset(subset):
        if not subset:
            return None
        gs = [r["gap"] for r in subset]
        rws = [r["real_wr"] for r in subset]
        ps = [r["pnl_pct"] for r in subset if r["pnl_pct"] is not None]
        pass_ct = sum(1 for r in subset if abs(r["gap"]) <= 10 and r["real_wr"] >= 65 and (r["pnl_pct"] or 0) > 0)
        return {
            "n": len(subset),
            "gap_median": round(statistics.median(gs), 2),
            "gap_mean": round(statistics.mean(gs), 2),
            "real_wr_median": round(statistics.median(rws), 2),
            "pct_honest_10pp": round(pct(sum(1 for g in gs if abs(g) <= 10), len(gs)), 1),
            "pct_honest_5pp": round(pct(sum(1 for g in gs if abs(g) <= 5), len(gs)), 1),
            "pct_real_wr_ge_65": round(pct(sum(1 for w in rws if w >= 65), len(rws)), 1),
            "pct_pnl_pos": round(pct(sum(1 for p in ps if p > 0), len(ps)), 1),
            "pct_passes_all_gates": round(pct(pass_ct, len(subset)), 1),
        }

    # Por TF
    by_tf = {}
    for tf in ["5m", "15m", "1h", "4h", "1d"]:
        subset = [r for r in rows if r["tf"] == tf]
        by_tf[tf] = stats_for_subset(subset)

    # Por bucket optuna_wr
    by_opt_wr = {}
    for b in ["65-70", "70-80", "80-90", "90-100"]:
        subset = [r for r in rows if bucket_wr(r["optuna_wr"]) == b]
        by_opt_wr[b] = stats_for_subset(subset)

    # Por bucket PF
    by_pf = {}
    for b in ["<1.0", "1.0-1.5", "1.5-2.0", "2.0-3.0", ">=3.0"]:
        subset = [r for r in rows if bucket_pf(r["pf"]) == b]
        by_pf[b] = stats_for_subset(subset)

    # Por bucket Sharpe
    by_sharpe = {}
    for b in ["<0.0", "0.0-0.5", "0.5-1.0", "1.0-2.0", "2.0-3.0", ">=3.0"]:
        subset = [r for r in rows if bucket_sharpe(r["sharpe"]) == b]
        by_sharpe[b] = stats_for_subset(subset)

    # Por liquidity
    by_liq = {}
    for liq in sorted({r["liquidity"] for r in rows if r["liquidity"]}):
        subset = [r for r in rows if r["liquidity"] == liq]
        by_liq[liq] = stats_for_subset(subset)

    # Cross: TF x optuna_wr bucket
    cross_tf_wr = {}
    for tf in ["5m", "15m", "1h", "4h", "1d"]:
        cross_tf_wr[tf] = {}
        for b in ["65-70", "70-80", "80-90", "90-100"]:
            subset = [r for r in rows if r["tf"] == tf and bucket_wr(r["optuna_wr"]) == b]
            cross_tf_wr[tf][b] = stats_for_subset(subset)

    # Combined Gate empirico (propuesto para V8): real_wr >=65 + pf>=2 + sharpe>=2 + tf in [4h,1d]
    combined_b = [r for r in rows if (r["pf"] or 0) >= 2.0 and (r["sharpe"] or -99) >= 2.0 and r["tf"] in ("4h", "1d")]
    gate_b_stats = stats_for_subset(combined_b)

    # Baseline: todos
    baseline = stats_for_subset(rows)

    # === OUTPUT ===
    out_json = {
        "source": str(FORENSIC),
        "generated_at": "2026-04-15",
        "topline": topline,
        "baseline_all": baseline,
        "by_timeframe": by_tf,
        "by_optuna_wr_bucket": by_opt_wr,
        "by_profit_factor_bucket": by_pf,
        "by_sharpe_bucket": by_sharpe,
        "by_liquidity_tier": by_liq,
        "cross_tf_x_optuna_wr": cross_tf_wr,
        "gate_b_candidate": {
            "criteria": "pf>=2.0 AND sharpe>=2.0 AND tf in (4h,1d)",
            "stats": gate_b_stats,
        },
        "notes": {
            "real_wr": "forensic win_rate (full history, fees 0.30% RT)",
            "optuna_wr": "optuna test WR (30% window)",
            "gap": "optuna_wr - real_wr (positive = inflated)",
            "r24_gate": "abs(gap) <= 10pp AND real_wr >= 65 AND pnl > 0",
        },
    }

    json_out = OUT_DIR / "empirical_gap_stats.json"
    with open(json_out, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n[OK] JSON -> {json_out}")

    # === MARKDOWN para IAs ===
    md = []
    md.append("# Empirical Gap Stats — Forensic V2 (23,693 grails, Hetzner scan)\n")
    md.append(f"**Fuente**: {FORENSIC.name}")
    md.append(f"**Generado**: 2026-04-15")
    md.append(f"**Usable (>=30 trades forensic)**: {len(rows):,} de {len(results):,}")
    md.append("")
    md.append("## Topline (baseline)")
    md.append(f"- Gap mediana: **{topline['gap_median_pp']:.1f}pp** (optuna_wr prometido - real_wr realidad)")
    md.append(f"- Gap media: {topline['gap_mean_pp']:.1f}pp (stdev {topline['gap_stdev_pp']:.1f})")
    md.append(f"- Gap p10/p25/p75/p90: {topline['gap_p10_pp']:.1f} / {topline['gap_p25_pp']:.1f} / {topline['gap_p75_pp']:.1f} / {topline['gap_p90_pp']:.1f}")
    md.append(f"- real_wr mediana: **{topline['real_wr_median_pct']:.1f}%**")
    md.append(f"- % con abs(gap) <= 5pp: {topline['pct_gap_le_5pp']:.1f}%")
    md.append(f"- % con abs(gap) <= 10pp: {topline['pct_gap_le_10pp']:.1f}%")
    md.append(f"- % con real_wr >= 65%: {topline['pct_real_wr_ge_65']:.1f}%")
    md.append(f"- % con PnL forensic > 0: {topline['pct_pnl_positive']:.1f}%")
    md.append(f"- % PASA R24 (gap<=10 + real_wr>=65 + PnL>0): **{topline['pct_both_gates_pass']:.1f}%**")
    md.append("")

    def emit_table(title, d, key_label):
        md.append(f"## {title}")
        md.append(f"| {key_label} | n | gap_med | real_wr_med | %honest_10pp | %honest_5pp | %real_wr>=65 | %PnL>0 | %PASA_R24 |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for k, v in d.items():
            if v is None:
                md.append(f"| {k} | 0 | - | - | - | - | - | - | - |")
            else:
                md.append(f"| {k} | {v['n']} | {v['gap_median']:+.1f} | {v['real_wr_median']:.1f} | {v['pct_honest_10pp']:.1f}% | {v['pct_honest_5pp']:.1f}% | {v['pct_real_wr_ge_65']:.1f}% | {v['pct_pnl_pos']:.1f}% | {v['pct_passes_all_gates']:.1f}% |")
        md.append("")

    emit_table("Por Timeframe", by_tf, "TF")
    emit_table("Por bucket de Optuna WR (prometido)", by_opt_wr, "optuna_wr")
    emit_table("Por Profit Factor (real)", by_pf, "PF")
    emit_table("Por Sharpe (real)", by_sharpe, "Sharpe")
    emit_table("Por Liquidity Tier", by_liq, "liquidity")

    # Cross TF x optuna_wr
    md.append("## Cross: TF × optuna_wr bucket (% PASA R24)")
    md.append("| TF | 65-70 | 70-80 | 80-90 | 90-100 |")
    md.append("|---|---|---|---|---|")
    for tf in ["5m", "15m", "1h", "4h", "1d"]:
        row = [tf]
        for b in ["65-70", "70-80", "80-90", "90-100"]:
            v = cross_tf_wr[tf][b]
            if v is None:
                row.append("n=0")
            else:
                row.append(f"{v['pct_passes_all_gates']:.1f}% (n={v['n']})")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    # Gate B candidato
    md.append("## Candidato Gate V8 (COMBINED_B)")
    md.append(f"**Criterio**: `pf>=2.0 AND sharpe>=2.0 AND tf in (4h, 1d)`")
    if gate_b_stats:
        md.append(f"- n: {gate_b_stats['n']}")
        md.append(f"- gap mediana: {gate_b_stats['gap_median']:+.1f}pp (vs baseline {topline['gap_median_pp']:+.1f}pp)")
        md.append(f"- real_wr mediana: {gate_b_stats['real_wr_median']:.1f}%")
        md.append(f"- % PASA R24: **{gate_b_stats['pct_passes_all_gates']:.1f}%** (vs baseline {topline['pct_both_gates_pass']:.1f}%)")
    md.append("")

    md.append("## Notas")
    md.append("- **real_wr**: forensic win_rate, entry al OPEN de vela siguiente, fees 0.30% round-trip, historia completa.")
    md.append("- **optuna_wr**: WR reportado por Optuna en test window (30% data).")
    md.append("- **gap positivo**: Optuna prometió más de lo que existe en la realidad.")
    md.append("- **R24**: `abs(gap) <= 10pp AND real_wr >= 65 AND PnL_forensic > 0`.")
    md.append("- **15m TF ausente del scan Hetzner** (0 de 23,693 records). Regla 30 fuerza su inclusión.")
    md.append("")

    md_out = OUT_DIR / "empirical_gap_stats.md"
    with open(md_out, "w") as f:
        f.write("\n".join(md))
    print(f"[OK] Markdown -> {md_out}")

    # Print top-line a stdout
    print("\n=== TOPLINE ===")
    for k, v in topline.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    print("\n=== TF breakdown (% PASA R24) ===")
    for tf, v in by_tf.items():
        if v:
            print(f"  {tf}: n={v['n']:,} gap_med={v['gap_median']:+.1f}pp  real_wr_med={v['real_wr_median']:.1f}%  %PASA={v['pct_passes_all_gates']:.1f}%")
        else:
            print(f"  {tf}: n=0 (ausente)")
    print("\n=== PF breakdown (% PASA R24) ===")
    for b, v in by_pf.items():
        if v:
            print(f"  PF {b}: n={v['n']:,}  %PASA={v['pct_passes_all_gates']:.1f}%")
    print("\n=== Gate V8 candidato (COMBINED_B) ===")
    if gate_b_stats:
        print(f"  n={gate_b_stats['n']}  gap_med={gate_b_stats['gap_median']:+.1f}pp  %PASA={gate_b_stats['pct_passes_all_gates']:.1f}%")
        print(f"  Baseline %PASA={topline['pct_both_gates_pass']:.1f}%  ->  Gate B %PASA={gate_b_stats['pct_passes_all_gates']:.1f}%")


if __name__ == "__main__":
    main()
