#!/usr/bin/env python3
"""
Quick V7 vs V8 Test — Smoke test retrospectivo de 10 minutos.

Aplica gates V7 (baseline) y gates V8 (Honesty) al MISMO universo de 23,131 grails
ya computados con forensic backtest. Es retrospectivo (no OOS) pero permite ver
el delta V7→V8 de forma honesta.

LIMITACIONES:
- NO es out-of-sample. Los gates V8 fueron derivados observando este mismo dataset.
- NO prueba robustez a regime shift ni generalización temporal.
- SÍ muestra: si aplicáramos V8 a los grails que V7 encontró, qué pasaría.

Criterios A+B+C PRE-REGISTRADOS (antes de ver resultados):
- A: mean |gap| V8 <= 5pp AND V8 < V7 por >= 3pp
- B: % pasa R24 en V8 >= 35%
- C: V8 PnL_median >= 0.9 * V7 PnL_median

Uso:
    python3 scripts/quick_v7_vs_v8_test.py
"""

import json
import statistics
import sys
from pathlib import Path
from datetime import datetime

# ─── Config ──────────────────────────────────────────────────────────────

DATASET = Path("/Users/sabrina/CLAUDE CODE/Estrategias/data/forensic_v2_hetzner_23K_LIGHT.json")
OUTPUT_DIR = Path("/Users/sabrina/CLAUDE CODE/Estrategias/data/empirical_analysis_20260415")
OUTPUT_JSON = OUTPUT_DIR / "v7_vs_v8_smoke_test.json"
OUTPUT_MD = OUTPUT_DIR / "v7_vs_v8_smoke_test.md"

# ⚠️ SOLO LOS APROBADOS V7 (~1,344 que pasaron a produccion)
# Filtramos por gate_approved=True. Los rechazados (22K) no son relevantes para
# la pregunta "de los que estan operando HOY, cuantos sobrevivirian V8?"
ONLY_APPROVED = True

# Criterios A+B+C PRE-REGISTRADOS
CRITERION_A_MAX_GAP_V8 = 5.0       # mean |gap| V8 debe ser <= 5pp
CRITERION_A_MIN_DELTA = 3.0        # V8 debe ser < V7 por >= 3pp
CRITERION_B_MIN_PASS_RATE = 35.0   # % pasa R24 en V8 debe ser >= 35%
CRITERION_C_MIN_PNL_RATIO = 0.9    # V8 PnL_median >= 0.9 * V7

# Gates V7 (baseline) — lo que V7 hace hoy
V7_MIN_OPTUNA_WR = 65.0   # Gate V7 tipico: optuna_wr >= 65%

# Gates V8 (Honesty) — propuesto
V8_ALLOWED_TFS = {"4h", "1d"}    # G11: TF filter (soft seria penalty, pero aqui test hard)
V8_MIN_PF = 2.0                   # G12: PF forensic >= 2.0
V8_MIN_SHARPE = 2.0               # G13: Sharpe forensic >= 2.0
V8_MIN_TRADES = 50                # G15: total_trades forensic >= 50
# G14 (gap <= 10pp) NO se usa como gate (seria circular) — se reporta como output

# Regla 24 (gate de honestidad final)
R24_MAX_GAP = 10.0      # |gap| <= 10pp
R24_MIN_REAL_WR = 65.0  # real_wr >= 65%
# R24 tambien exige PnL > 0 (se aplica abajo)

# Minimo de trades forenses para considerar grail "evaluado" (igual que stats previas)
MIN_FORENSIC_TRADES = 30

# ─── Carga ───────────────────────────────────────────────────────────────

def load_grails():
    """Carga grails del JSON forense V2 light."""
    if not DATASET.exists():
        print(f"ERROR: No existe {DATASET}", file=sys.stderr)
        sys.exit(1)

    with open(DATASET, "r") as f:
        data = json.load(f)

    # Estructura: {results: [{strategy, symbol, timeframe, optuna_wr, metrics: {...}}]}
    results = data.get("results", [])
    print(f"Cargados {len(results):,} records del dataset")

    # Filtrar: grails con metrics completos y >= MIN_FORENSIC_TRADES
    # Y (opcional) solo los aprobados por V7 (gate_approved=True)
    usable = []
    approved_count = 0
    for r in results:
        metrics = r.get("metrics") or {}
        n_trades = metrics.get("total_trades") or 0
        if n_trades < MIN_FORENSIC_TRADES:
            continue
        # Necesitamos optuna_wr, real_wr, PF, Sharpe, PnL
        if r.get("optuna_wr") is None:
            continue
        if metrics.get("win_rate") is None:
            continue
        if r.get("gate_approved"):
            approved_count += 1
        # Si ONLY_APPROVED=True, filtrar a solo los aprobados V7
        if ONLY_APPROVED and not r.get("gate_approved"):
            continue
        usable.append(r)

    filter_tag = "SOLO APROBADOS V7 (gate_approved=True)" if ONLY_APPROVED else "TODOS"
    print(f"Usables (>= {MIN_FORENSIC_TRADES} trades, {filter_tag}): {len(usable):,}")
    print(f"  (aprobados V7 en dataset completo: {approved_count:,})")
    return usable

# ─── Extraccion de campos ────────────────────────────────────────────────

def extract_fields(grail):
    """Extrae campos normalizados de un grail."""
    m = grail.get("metrics") or {}
    optuna_wr = float(grail.get("optuna_wr") or 0)  # viene en % (0-100)
    real_wr = float(m.get("win_rate") or 0)          # viene en 0-1
    # Normalizar real_wr a 0-100 si viene en 0-1
    if real_wr <= 1.0:
        real_wr = real_wr * 100.0
    # Optuna_wr si viene en 0-1 tambien normalizar
    if optuna_wr <= 1.0:
        optuna_wr = optuna_wr * 100.0

    return {
        "strategy": grail.get("strategy"),
        "symbol": grail.get("symbol"),
        "timeframe": grail.get("timeframe"),
        "optuna_wr": optuna_wr,
        "real_wr": real_wr,
        "gap": optuna_wr - real_wr,
        "pnl_pct": float(m.get("total_pnl_pct") or 0),
        "pf": float(m.get("profit_factor") or 0),
        "sharpe": float(m.get("sharpe") or 0),
        "trades": int(m.get("total_trades") or 0),
    }

# ─── Gates ───────────────────────────────────────────────────────────────

def passes_v7(g):
    """V7 baseline.
    Si ONLY_APPROVED=True: ya vienen pre-filtrados por gate V7 real → todos pasan.
    Si ONLY_APPROVED=False: aplicar gate optuna_wr >= umbral como proxy V7.
    """
    if ONLY_APPROVED:
        return True  # ya son los aprobados V7 en produccion
    return g["optuna_wr"] >= V7_MIN_OPTUNA_WR

def passes_v8(g):
    """V8 Honesty Gates (excluye G14 circular)"""
    if g["timeframe"] not in V8_ALLOWED_TFS:
        return False
    if g["pf"] < V8_MIN_PF:
        return False
    if g["sharpe"] < V8_MIN_SHARPE:
        return False
    if g["trades"] < V8_MIN_TRADES:
        return False
    return True

def passes_r24(g):
    """Regla 24: |gap| <= 10pp AND real_wr >= 65% AND PnL > 0"""
    if abs(g["gap"]) > R24_MAX_GAP:
        return False
    if g["real_wr"] < R24_MIN_REAL_WR:
        return False
    if g["pnl_pct"] <= 0:
        return False
    return True

# ─── Metricas ────────────────────────────────────────────────────────────

def median_safe(xs):
    return statistics.median(xs) if xs else 0.0

def mean_safe(xs):
    return statistics.mean(xs) if xs else 0.0

def pct(num, den):
    return (100.0 * num / den) if den > 0 else 0.0

def compute_metrics(label, grails):
    """Computa metricas agregadas para un set de grails."""
    n = len(grails)
    if n == 0:
        return {
            "label": label, "n": 0,
            "gap_median": 0, "gap_mean": 0, "abs_gap_mean": 0,
            "real_wr_median": 0, "pnl_median": 0, "pnl_mean": 0,
            "pf_median": 0, "sharpe_median": 0,
            "pass_r24": 0, "pct_pass_r24": 0.0,
        }

    gaps = [g["gap"] for g in grails]
    abs_gaps = [abs(g["gap"]) for g in grails]
    real_wrs = [g["real_wr"] for g in grails]
    pnls = [g["pnl_pct"] for g in grails]
    pfs = [g["pf"] for g in grails]
    sharpes = [g["sharpe"] for g in grails]

    r24_count = sum(1 for g in grails if passes_r24(g))

    # Desglose por TF
    tf_dist = {}
    for g in grails:
        tf = g["timeframe"]
        tf_dist[tf] = tf_dist.get(tf, 0) + 1

    return {
        "label": label,
        "n": n,
        "gap_median": round(median_safe(gaps), 2),
        "gap_mean": round(mean_safe(gaps), 2),
        "abs_gap_mean": round(mean_safe(abs_gaps), 2),
        "real_wr_median": round(median_safe(real_wrs), 2),
        "pnl_median": round(median_safe(pnls), 4),
        "pnl_mean": round(mean_safe(pnls), 4),
        "pf_median": round(median_safe(pfs), 3),
        "sharpe_median": round(median_safe(sharpes), 3),
        "pass_r24": r24_count,
        "pct_pass_r24": round(pct(r24_count, n), 2),
        "tf_distribution": tf_dist,
    }

# ─── Criterios A/B/C ─────────────────────────────────────────────────────

def evaluate_criteria(v7_m, v8_m):
    """Evalua criterios A+B+C pre-registrados."""
    # A: mean |gap| V8 <= 5pp AND V8 < V7 por >= 3pp
    crit_a_absgap = v8_m["abs_gap_mean"] <= CRITERION_A_MAX_GAP_V8
    crit_a_delta = (v7_m["abs_gap_mean"] - v8_m["abs_gap_mean"]) >= CRITERION_A_MIN_DELTA
    crit_a = crit_a_absgap and crit_a_delta

    # B: % pasa R24 V8 >= 35%
    crit_b = v8_m["pct_pass_r24"] >= CRITERION_B_MIN_PASS_RATE

    # C: V8 PnL_median >= 0.9 * V7 PnL_median
    v7_pnl = v7_m["pnl_median"]
    v8_pnl = v8_m["pnl_median"]
    if v7_pnl <= 0:
        # Si V7 median es <=0, C no aplica estrictamente — V8 gana por default si >= V7
        crit_c = v8_pnl >= v7_pnl
        crit_c_note = f"(V7 PnL median <= 0: aceptar si V8 >= V7. V8={v8_pnl}, V7={v7_pnl})"
    else:
        crit_c = v8_pnl >= CRITERION_C_MIN_PNL_RATIO * v7_pnl
        crit_c_note = f"(V8_pnl={v8_pnl} vs 0.9*V7={round(0.9*v7_pnl, 4)})"

    return {
        "A_abs_gap_le_5pp": {
            "pass": crit_a_absgap,
            "v8_abs_gap_mean": v8_m["abs_gap_mean"],
            "threshold": CRITERION_A_MAX_GAP_V8,
        },
        "A_delta_ge_3pp": {
            "pass": crit_a_delta,
            "delta": round(v7_m["abs_gap_mean"] - v8_m["abs_gap_mean"], 2),
            "threshold": CRITERION_A_MIN_DELTA,
        },
        "A_final": crit_a,
        "B_pass_rate_ge_35pct": {
            "pass": crit_b,
            "v8_pct_pass_r24": v8_m["pct_pass_r24"],
            "threshold": CRITERION_B_MIN_PASS_RATE,
        },
        "B_final": crit_b,
        "C_pnl_non_inferior": {
            "pass": crit_c,
            "note": crit_c_note,
        },
        "C_final": crit_c,
        "ALL_PASS": crit_a and crit_b and crit_c,
    }

# ─── Main ────────────────────────────────────────────────────────────────

def main():
    t0 = datetime.now()
    print("=" * 70)
    print("QUICK V7 vs V8 SMOKE TEST — retrospectivo")
    print("=" * 70)
    print(f"Fecha: {t0.isoformat()}")
    print(f"Dataset: {DATASET.name}")
    print()
    print("Criterios PRE-REGISTRADOS:")
    print(f"  A: mean |gap| V8 <= {CRITERION_A_MAX_GAP_V8}pp AND delta V7-V8 >= {CRITERION_A_MIN_DELTA}pp")
    print(f"  B: % pasa R24 en V8 >= {CRITERION_B_MIN_PASS_RATE}%")
    print(f"  C: V8 PnL_median >= {CRITERION_C_MIN_PNL_RATIO} * V7 PnL_median")
    print()
    print("Gates V7 (baseline): optuna_wr >= 65%")
    print(f"Gates V8 (Honesty):")
    print(f"  G11: TF in {V8_ALLOWED_TFS}")
    print(f"  G12: PF_forensic >= {V8_MIN_PF}")
    print(f"  G13: Sharpe_forensic >= {V8_MIN_SHARPE}")
    print(f"  G15: trades >= {V8_MIN_TRADES}")
    print("  (G14 gap <= 10pp se excluye — seria circular; se reporta como output)")
    print()

    # 1) Cargar
    grails_raw = load_grails()
    grails = [extract_fields(g) for g in grails_raw]

    # 2) Universo total (baseline absoluto = ningun gate)
    universe_m = compute_metrics("UNIVERSE (no gates)", grails)

    # 3) V7 baseline
    v7_grails = [g for g in grails if passes_v7(g)]
    v7_label = "V7 (aprobados = en produccion hoy)" if ONLY_APPROVED else "V7 (optuna_wr>=65%)"
    v7_m = compute_metrics(v7_label, v7_grails)

    # 4) V8 Honesty
    v8_grails = [g for g in grails if passes_v8(g)]
    v8_m = compute_metrics("V8 (Honesty Gates)", v8_grails)

    # 5) Criterios
    criteria = evaluate_criteria(v7_m, v8_m)

    # 6) Output console
    print("─" * 70)
    print("RESULTADOS")
    print("─" * 70)

    def print_set(m):
        print(f"\n[{m['label']}]")
        print(f"  n                : {m['n']:,}")
        print(f"  gap_median       : {m['gap_median']:+.2f}pp")
        print(f"  gap_mean         : {m['gap_mean']:+.2f}pp")
        print(f"  abs_gap_mean     : {m['abs_gap_mean']:+.2f}pp")
        print(f"  real_wr_median   : {m['real_wr_median']:.2f}%")
        print(f"  pnl_median (%)   : {m['pnl_median']:+.4f}")
        print(f"  pnl_mean (%)     : {m['pnl_mean']:+.4f}")
        print(f"  pf_median        : {m['pf_median']:.3f}")
        print(f"  sharpe_median    : {m['sharpe_median']:.3f}")
        print(f"  pasa R24         : {m['pass_r24']:,} ({m['pct_pass_r24']:.2f}%)")
        if m.get("tf_distribution"):
            tfs = ", ".join(f"{k}:{v}" for k, v in sorted(m["tf_distribution"].items()))
            print(f"  TF distribution  : {tfs}")

    print_set(universe_m)
    print_set(v7_m)
    print_set(v8_m)

    print()
    print("─" * 70)
    print("CRITERIOS PRE-REGISTRADOS")
    print("─" * 70)
    def ok(b): return "PASS" if b else "FAIL"

    print(f"\n  A.1 |gap| V8 <= 5pp            : {ok(criteria['A_abs_gap_le_5pp']['pass'])}"
          f"  (V8={criteria['A_abs_gap_le_5pp']['v8_abs_gap_mean']}pp, threshold=5.0)")
    print(f"  A.2 delta V7-V8 >= 3pp         : {ok(criteria['A_delta_ge_3pp']['pass'])}"
          f"  (delta={criteria['A_delta_ge_3pp']['delta']}pp, threshold=3.0)")
    print(f"  A (AMBOS) FINAL                : {ok(criteria['A_final'])}")
    print()
    print(f"  B. %pasa R24 V8 >= 35%         : {ok(criteria['B_pass_rate_ge_35pct']['pass'])}"
          f"  (V8={criteria['B_pass_rate_ge_35pct']['v8_pct_pass_r24']}%, threshold=35%)")
    print()
    print(f"  C. V8 PnL_median no-inferior   : {ok(criteria['C_pnl_non_inferior']['pass'])}"
          f"  {criteria['C_pnl_non_inferior']['note']}")
    print()
    print("=" * 70)
    print(f"VEREDICTO TOTAL A+B+C: {ok(criteria['ALL_PASS'])}")
    print("=" * 70)

    # 7) Guardar JSON
    output = {
        "timestamp": t0.isoformat(),
        "dataset": str(DATASET),
        "min_forensic_trades": MIN_FORENSIC_TRADES,
        "pre_registered_criteria": {
            "A_max_abs_gap_v8_pp": CRITERION_A_MAX_GAP_V8,
            "A_min_delta_pp": CRITERION_A_MIN_DELTA,
            "B_min_pass_rate_pct": CRITERION_B_MIN_PASS_RATE,
            "C_min_pnl_ratio": CRITERION_C_MIN_PNL_RATIO,
        },
        "v7_gates": {"min_optuna_wr": V7_MIN_OPTUNA_WR},
        "v8_gates": {
            "allowed_tfs": sorted(V8_ALLOWED_TFS),
            "min_pf": V8_MIN_PF,
            "min_sharpe": V8_MIN_SHARPE,
            "min_trades": V8_MIN_TRADES,
            "notes": "G14 (gap <= 10pp) excluido de gate (circular); reportado como output",
        },
        "universe_metrics": universe_m,
        "v7_metrics": v7_m,
        "v8_metrics": v8_m,
        "criteria_evaluation": criteria,
        "caveats": [
            "Retrospectivo en el mismo dataset usado para derivar V8 gates",
            "NO es OOS — no prueba generalizacion temporal",
            "Validacion formal requiere holdout temporal Abr-May 2026",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nGuardado: {OUTPUT_JSON}")

    # 8) Guardar Markdown
    md = []
    md.append(f"# V7 vs V8 Smoke Test — {t0.strftime('%Y-%m-%d %H:%M')}")
    md.append("")
    md.append("**Tipo**: Retrospectivo (mismo universo, no OOS).")
    md.append(f"**Dataset**: `{DATASET.name}` ({len(grails):,} grails usables con >= {MIN_FORENSIC_TRADES} trades)")
    md.append("")
    md.append("## Criterios pre-registrados")
    md.append(f"- **A**: `mean |gap| V8 <= {CRITERION_A_MAX_GAP_V8}pp AND V7-V8 >= {CRITERION_A_MIN_DELTA}pp`")
    md.append(f"- **B**: `% pasa R24 V8 >= {CRITERION_B_MIN_PASS_RATE}%`")
    md.append(f"- **C**: `V8 PnL_median >= {CRITERION_C_MIN_PNL_RATIO} * V7`")
    md.append("")
    md.append("## Gates")
    md.append(f"- **V7 baseline**: `optuna_wr >= {V7_MIN_OPTUNA_WR}%`")
    md.append(f"- **V8 Honesty**: TF in {sorted(V8_ALLOWED_TFS)} AND PF>={V8_MIN_PF} AND Sharpe>={V8_MIN_SHARPE} AND trades>={V8_MIN_TRADES}")
    md.append("")
    md.append("## Resultados")
    md.append("")
    md.append("| Set | n | gap_med | abs_gap_mean | real_wr_med | PnL_med | PF_med | %pasa_R24 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in [universe_m, v7_m, v8_m]:
        md.append(f"| {m['label']} | {m['n']:,} | {m['gap_median']:+.2f}pp | {m['abs_gap_mean']:+.2f}pp "
                  f"| {m['real_wr_median']:.2f}% | {m['pnl_median']:+.4f}% | {m['pf_median']:.3f} "
                  f"| {m['pct_pass_r24']:.2f}% |")
    md.append("")
    md.append("## Evaluacion de criterios")
    md.append("")
    md.append(f"- **A.1** `|gap| V8 <= 5pp`: **{ok(criteria['A_abs_gap_le_5pp']['pass'])}** "
              f"(V8={criteria['A_abs_gap_le_5pp']['v8_abs_gap_mean']}pp)")
    md.append(f"- **A.2** `delta V7-V8 >= 3pp`: **{ok(criteria['A_delta_ge_3pp']['pass'])}** "
              f"(delta={criteria['A_delta_ge_3pp']['delta']}pp)")
    md.append(f"- **A final**: **{ok(criteria['A_final'])}**")
    md.append(f"- **B** `%pasa R24 V8 >= 35%`: **{ok(criteria['B_pass_rate_ge_35pct']['pass'])}** "
              f"(V8={criteria['B_pass_rate_ge_35pct']['v8_pct_pass_r24']}%)")
    md.append(f"- **C** `V8 PnL >= 0.9*V7`: **{ok(criteria['C_pnl_non_inferior']['pass'])}** "
              f"{criteria['C_pnl_non_inferior']['note']}")
    md.append("")
    md.append(f"### VEREDICTO A+B+C: **{ok(criteria['ALL_PASS'])}**")
    md.append("")
    md.append("## Caveats (importante)")
    md.append("- Es retrospectivo: los gates V8 se derivaron observando ESTE dataset.")
    md.append("- NO es OOS: riesgo de circular si migramos a produccion solo con esto.")
    md.append("- Validacion formal requiere holdout temporal (datos Abr 15 - May 15 no usados en derivacion).")
    md.append("- Este smoke test dice: 'si V7 hubiera tenido V8, habria filtrado asi'. No dice si V8 generaliza.")

    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(md))
    print(f"Guardado: {OUTPUT_MD}")

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nTiempo total: {elapsed:.2f}s")

    # Exit code: 0 si pasa A+B+C, 1 si falla al menos uno
    sys.exit(0 if criteria["ALL_PASS"] else 1)

if __name__ == "__main__":
    main()
