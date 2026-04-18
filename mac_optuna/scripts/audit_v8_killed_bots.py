#!/usr/bin/env python3
"""
Audit V8 Killed Bots — Los 5m+1h que V8 mataria estan performando en produccion?

Cruza:
  - Dataset forensic: forensic_v2_hetzner_23K_LIGHT.json (1,288 aprobados V7)
  - DB unificada: BOT V7/live/data/unified_bot_v7.db (trades reales produccion)

Output: lista de combos (strategy, symbol, tf) que V8 eliminaria + metricas reales
(WR clean, PnL real, n_trades, ultimos 7/30 dias).

Veredicto:
  - Si >50% estan ganando en prod → V8 es demasiado agresivo (perdemos $$$)
  - Si >50% estan perdiendo en prod → V8 tiene razon (eliminar ahorra $$$)
  - Si no tienen trades → pendiente (aun no activados o sin senales)

Uso:
    python3 scripts/audit_v8_killed_bots.py
"""

import json
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta
import statistics

# ─── Config ──────────────────────────────────────────────────────────────

DATASET = Path("/Users/sabrina/CLAUDE CODE/Estrategias/data/forensic_v2_hetzner_23K_LIGHT.json")
DB_UNIFIED = Path("/Users/sabrina/CLAUDE CODE/BOT V7/live/data/unified_bot_v7.db")
DB_TRADES_V7 = Path("/Users/sabrina/CLAUDE CODE/BOT V7/live/data/trades_v7.db")

OUTPUT_DIR = Path("/Users/sabrina/CLAUDE CODE/Estrategias/data/empirical_analysis_20260415")
OUTPUT_JSON = OUTPUT_DIR / "v8_killed_bots_audit.json"
OUTPUT_MD = OUTPUT_DIR / "v8_killed_bots_audit.md"

# V8 gates (mismo que el smoke test)
V8_ALLOWED_TFS = {"4h", "1d"}
V8_MIN_PF = 2.0
V8_MIN_SHARPE = 2.0
V8_MIN_TRADES_FORENSIC = 50

MIN_FORENSIC_TRADES = 30

# ─── Helpers ─────────────────────────────────────────────────────────────

def norm_pct(v):
    """Convierte 0-1 a 0-100 si hace falta."""
    v = float(v or 0)
    return v * 100 if v <= 1.0 else v

def killed_by_v8(g):
    """True si el grail era aprobado V7 pero V8 lo mataria."""
    m = g.get("metrics") or {}
    tf = g.get("timeframe")
    pf = float(m.get("profit_factor") or 0)
    sharpe = float(m.get("sharpe") or 0)
    trades = int(m.get("total_trades") or 0)
    if tf not in V8_ALLOWED_TFS:
        return True, "tf_filter"
    if pf < V8_MIN_PF:
        return True, "pf_low"
    if sharpe < V8_MIN_SHARPE:
        return True, "sharpe_low"
    if trades < V8_MIN_TRADES_FORENSIC:
        return True, "few_forensic_trades"
    return False, None

# ─── 1. Cargar aprobados V7 ──────────────────────────────────────────────

def load_approved_grails():
    print(f"Cargando {DATASET.name}...")
    with open(DATASET, "r") as f:
        data = json.load(f)
    results = data.get("results", [])

    approved = []
    for r in results:
        if not r.get("gate_approved"):
            continue
        m = r.get("metrics") or {}
        if (m.get("total_trades") or 0) < MIN_FORENSIC_TRADES:
            continue
        if r.get("optuna_wr") is None:
            continue
        approved.append(r)
    print(f"Aprobados V7 con >= {MIN_FORENSIC_TRADES} trades forenses: {len(approved):,}")
    return approved

# ─── 2. Clasificar killed vs survived ────────────────────────────────────

def classify(approved):
    killed = []
    survived = []
    for g in approved:
        is_killed, reason = killed_by_v8(g)
        entry = {
            "strategy": g.get("strategy"),
            "symbol": g.get("symbol"),
            "timeframe": g.get("timeframe"),
            "optuna_wr": norm_pct(g.get("optuna_wr")),
            "real_wr": norm_pct((g.get("metrics") or {}).get("win_rate")),
            "pnl_pct_forensic": float((g.get("metrics") or {}).get("total_pnl_pct") or 0),
            "pf_forensic": float((g.get("metrics") or {}).get("profit_factor") or 0),
            "sharpe_forensic": float((g.get("metrics") or {}).get("sharpe") or 0),
            "forensic_trades": int((g.get("metrics") or {}).get("total_trades") or 0),
            "kill_reason": reason,
        }
        if is_killed:
            killed.append(entry)
        else:
            survived.append(entry)
    return killed, survived

# ─── 3. Cruzar con produccion real ───────────────────────────────────────

def detect_trades_schema(db_path):
    """Detecta que columnas tiene la tabla de trades."""
    if not db_path.exists():
        return None, None
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = con.cursor()
        # Buscar tabla de trades
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        # Priorizar tabla 'trades'
        table = None
        for candidate in ["trades", "virtual_trades", "executed_trades"]:
            if candidate in tables:
                table = candidate
                break
        if table is None:
            con.close()
            return None, None
        cur.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in cur.fetchall()]
        con.close()
        return table, cols
    except Exception as e:
        print(f"ERROR leyendo {db_path}: {e}")
        return None, None

def symbol_base(symbol_raw):
    """De 'BTC/USDT:USDT' → 'BTC'. De 'BTC' → 'BTC'."""
    if not symbol_raw:
        return ""
    s = str(symbol_raw).upper()
    # Extraer antes del /
    if "/" in s:
        s = s.split("/")[0]
    return s

def extract_strategy_from_bot(bot_name, symbol_raw):
    """De bot_name='B5_VWAP_2Std_CYS' + symbol='CYS/USDT:USDT' → strategy='B5_VWAP_2Std'."""
    if not bot_name:
        return None
    base = symbol_base(symbol_raw)
    suffix = f"_{base}"
    if base and bot_name.upper().endswith(suffix):
        return bot_name[:-len(suffix)]
    return bot_name  # fallback: asumir bot_name es la strategy completa

def get_production_metrics_v2(killed, db_path):
    """Version mejorada: detecta tabla 'trades' o 'virtual_trades', extrae strategy desde bot_name."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]

    # Elegir tabla
    table = None
    for candidate in ["trades", "virtual_trades"]:
        if candidate in tables:
            table = candidate
            break
    if not table:
        con.close()
        return {}

    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    has_infra = "is_infra_noise" in cols
    has_notes = "notes" in cols
    pnl_col = "pnl_usd" if "pnl_usd" in cols else ("pnl" if "pnl" in cols else None)
    if not pnl_col:
        con.close()
        return {}

    # Filtro infra
    where_parts = []
    if has_infra:
        where_parts.append("is_infra_noise = 0")
    if has_notes:
        where_parts.append(
            "(notes IS NULL OR notes = '' OR (notes NOT LIKE '%POSITION_GONE%' "
            "AND notes NOT LIKE '%FORCE_CLOSE%' AND notes NOT LIKE '%timeout%' "
            "AND notes NOT LIKE '%LOSS_CAP_RUNTIME%'))"
        )
    # status closed solo (excluir open y skipped)
    if "status" in cols:
        where_parts.append("status = 'closed'")

    where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    sql = f"SELECT bot_name, symbol, timeframe, {pnl_col} FROM {table} {where_sql}"
    try:
        cur.execute(sql)
        rows = cur.fetchall()
    except Exception as e:
        print(f"    ERROR: {e}")
        con.close()
        return {}
    con.close()

    print(f"    Trades clean cerrados: {len(rows):,}")

    # Agrupar por key
    groups = {}
    for bot_name, symbol_raw, tf, pnl in rows:
        strat = extract_strategy_from_bot(bot_name, symbol_raw)
        if not strat:
            continue
        key = (strat, symbol_raw, tf)
        groups.setdefault(key, []).append(float(pnl or 0))

    prod_by_combo = {}
    for entry in killed:
        key = (entry["strategy"], entry["symbol"], entry["timeframe"])
        pnls = groups.get(key)
        if pnls:
            n = len(pnls)
            wins = sum(1 for p in pnls if p > 0)
            losses = sum(1 for p in pnls if p < 0)
            total_pnl = sum(pnls)
            wr = (wins / n * 100) if n > 0 else 0
            prod_by_combo[key] = {
                "trades_clean": n,
                "wins": wins,
                "losses": losses,
                "wr_clean": round(wr, 2),
                "pnl_usd_total": round(total_pnl, 4),
                "avg_pnl": round(total_pnl / n, 4) if n > 0 else 0,
            }
        else:
            prod_by_combo[key] = {
                "trades_clean": 0,
                "wins": 0,
                "losses": 0,
                "wr_clean": 0,
                "pnl_usd_total": 0,
                "avg_pnl": 0,
            }
    return prod_by_combo


def get_production_metrics(killed, db_path):
    """Para cada killed combo, cruza con DB de produccion (bot_name-based).
    Carga todos los trades una vez, agrupa por (strategy, symbol_base, tf) en Python.
    """
    if not db_path.exists():
        print(f"  DB no existe: {db_path}")
        return {}

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.cursor()

    # Verificar columnas
    cur.execute("PRAGMA table_info(trades)")
    cols = [r[1] for r in cur.fetchall()]
    has_infra = "is_infra_noise" in cols
    has_notes = "notes" in cols
    has_pnl = "pnl_usd" in cols or "pnl" in cols
    pnl_col = "pnl_usd" if "pnl_usd" in cols else "pnl"

    if not has_pnl:
        print(f"  FAIL: no hay columna pnl en trades")
        con.close()
        return {}

    # Cargar todos los trades con infra filter
    where_parts = []
    if has_infra:
        where_parts.append("is_infra_noise = 0")
    else:
        # Heuristica con notes si existe
        if has_notes:
            where_parts.append(
                "(notes IS NULL OR (notes NOT LIKE '%POSITION_GONE%' "
                "AND notes NOT LIKE '%FORCE_CLOSE%' AND notes NOT LIKE '%timeout%' "
                "AND notes NOT LIKE '%LOSS_CAP_RUNTIME%'))"
            )
    where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    sql = f"SELECT bot_name, symbol, timeframe, {pnl_col} FROM trades {where_sql}"
    print(f"  SQL: {sql}")
    try:
        cur.execute(sql)
        rows = cur.fetchall()
    except Exception as e:
        print(f"  ERROR: {e}")
        con.close()
        return {}
    con.close()

    print(f"  Trades clean cargados: {len(rows):,}")

    # Agrupar por (strategy, symbol_full, tf)
    groups = {}
    for bot_name, symbol_raw, tf, pnl in rows:
        strat = extract_strategy_from_bot(bot_name, symbol_raw)
        if not strat:
            continue
        # Key en forensic: symbol = 'BTC/USDT:USDT' full → mantenemos asi
        key = (strat, symbol_raw, tf)
        groups.setdefault(key, []).append(float(pnl or 0))

    # Debug: cuántos grupos tenemos
    print(f"  Grupos unicos (strategy, symbol, tf): {len(groups):,}")

    # Mapear los killed por su key
    prod_by_combo = {}
    for entry in killed:
        key = (entry["strategy"], entry["symbol"], entry["timeframe"])
        pnls = groups.get(key)
        if pnls:
            n = len(pnls)
            wins = sum(1 for p in pnls if p > 0)
            losses = sum(1 for p in pnls if p < 0)
            total_pnl = sum(pnls)
            wr = (wins / n * 100) if n > 0 else 0
            prod_by_combo[key] = {
                "trades_clean": n,
                "wins": wins,
                "losses": losses,
                "wr_clean": round(wr, 2),
                "pnl_usd_total": round(total_pnl, 4),
                "avg_pnl": round(total_pnl / n, 4) if n > 0 else 0,
            }
        else:
            prod_by_combo[key] = {
                "trades_clean": 0,
                "wins": 0,
                "losses": 0,
                "wr_clean": 0,
                "pnl_usd_total": 0,
                "avg_pnl": 0,
            }

    return prod_by_combo

# ─── 4. Main ─────────────────────────────────────────────────────────────

def main():
    t0 = datetime.now()
    print("=" * 70)
    print("AUDIT V8 KILLED BOTS — performan en produccion?")
    print("=" * 70)
    print(f"Fecha: {t0.isoformat()}")
    print()

    # 1. Cargar aprobados V7
    approved = load_approved_grails()

    # 2. Clasificar
    killed, survived = classify(approved)
    print(f"\nV7 aprobados: {len(approved):,}")
    print(f"  V8 killed   : {len(killed):,}")
    print(f"  V8 survived : {len(survived):,}")

    # Desglose por razon
    reasons = {}
    for k in killed:
        r = k["kill_reason"]
        reasons[r] = reasons.get(r, 0) + 1
    print(f"  Razones de kill: {reasons}")

    # Desglose por TF
    killed_by_tf = {}
    for k in killed:
        tf = k["timeframe"]
        killed_by_tf[tf] = killed_by_tf.get(tf, 0) + 1
    print(f"  Killed por TF  : {killed_by_tf}")

    # 3. Cruzar con DB unificada + DB trades_v7 (merge ambas)
    print(f"\n--- Cruzando con DBs de produccion ---")
    prod = {}
    db_prod_list = []
    for db_path in [DB_UNIFIED, DB_TRADES_V7]:
        if not db_path.exists():
            continue
        print(f"DB: {db_path.name}")
        sub = get_production_metrics_v2(killed, db_path)
        # Merge: si un combo aparece en ambos, sumar trades/pnls
        for key, stats in sub.items():
            if key not in prod or prod[key]["trades_clean"] == 0:
                prod[key] = stats
            elif stats["trades_clean"] > 0:
                # merge
                prod[key]["trades_clean"] += stats["trades_clean"]
                prod[key]["wins"] += stats["wins"]
                prod[key]["losses"] += stats["losses"]
                prod[key]["pnl_usd_total"] = round(prod[key]["pnl_usd_total"] + stats["pnl_usd_total"], 4)
                prod[key]["wr_clean"] = round(prod[key]["wins"] / prod[key]["trades_clean"] * 100, 2) if prod[key]["trades_clean"] > 0 else 0
                prod[key]["avg_pnl"] = round(prod[key]["pnl_usd_total"] / prod[key]["trades_clean"], 4) if prod[key]["trades_clean"] > 0 else 0
        db_prod_list.append(db_path.name)
    db_prod = ", ".join(db_prod_list) if db_prod_list else None

    # 4. Clasificar killed por performance real
    with_trades = []
    no_trades = []
    for k in killed:
        key = (k["strategy"], k["symbol"], k["timeframe"])
        p = prod.get(key) or {"trades_clean": 0, "wr_clean": 0, "pnl_usd_total": 0}
        k["production"] = p
        if p["trades_clean"] > 0:
            with_trades.append(k)
        else:
            no_trades.append(k)

    # Metricas de los killed con trades
    if with_trades:
        winning = [k for k in with_trades if k["production"]["pnl_usd_total"] > 0]
        losing = [k for k in with_trades if k["production"]["pnl_usd_total"] < 0]
        neutral = [k for k in with_trades if k["production"]["pnl_usd_total"] == 0]
        total_pnl = sum(k["production"]["pnl_usd_total"] for k in with_trades)
        total_trades = sum(k["production"]["trades_clean"] for k in with_trades)
        avg_wr = statistics.mean(k["production"]["wr_clean"] for k in with_trades)
    else:
        winning = losing = neutral = []
        total_pnl = total_trades = avg_wr = 0

    print(f"\n--- Killed con trades reales ---")
    print(f"  Con trades    : {len(with_trades):,}")
    print(f"  Sin trades    : {len(no_trades):,}")
    if with_trades:
        print(f"  Ganando ($$$) : {len(winning):,}  (pnl total = +${sum(k['production']['pnl_usd_total'] for k in winning):.2f})")
        print(f"  Perdiendo     : {len(losing):,}  (pnl total = -${abs(sum(k['production']['pnl_usd_total'] for k in losing)):.2f})")
        print(f"  Neutral (0)   : {len(neutral):,}")
        print(f"  PnL total killed: {'+' if total_pnl>=0 else ''}{total_pnl:.2f} USD")
        print(f"  Trades totales  : {total_trades:,}")
        print(f"  WR promedio     : {avg_wr:.2f}%")

    # Top 10 ganadores y top 10 perdedores de los killed
    if with_trades:
        top_winners = sorted(winning, key=lambda k: -k["production"]["pnl_usd_total"])[:10]
        top_losers = sorted(losing, key=lambda k: k["production"]["pnl_usd_total"])[:10]

        print(f"\n--- TOP 10 WINNERS que V8 mataria ---")
        print(f"{'Strategy':<25} {'Symbol':<12} {'TF':<4} {'Trades':>7} {'WR%':>6} {'PnL':>10} {'PF_for':>7}")
        for k in top_winners:
            p = k["production"]
            print(f"{k['strategy'][:24]:<25} {k['symbol'][:11]:<12} {k['timeframe']:<4} "
                  f"{p['trades_clean']:>7} {p['wr_clean']:>6.1f} {p['pnl_usd_total']:>+10.2f} {k['pf_forensic']:>7.2f}")

        print(f"\n--- TOP 10 LOSERS que V8 matara (justificado) ---")
        print(f"{'Strategy':<25} {'Symbol':<12} {'TF':<4} {'Trades':>7} {'WR%':>6} {'PnL':>10} {'PF_for':>7}")
        for k in top_losers:
            p = k["production"]
            print(f"{k['strategy'][:24]:<25} {k['symbol'][:11]:<12} {k['timeframe']:<4} "
                  f"{p['trades_clean']:>7} {p['wr_clean']:>6.1f} {p['pnl_usd_total']:>+10.2f} {k['pf_forensic']:>7.2f}")

    # 5. ANALISIS FORENSIC (SIEMPRE — mas robusto que los pocos trades de produccion)
    print(f"\n--- Analisis forense de los killed (historia completa) ---")
    forensic_winning = [k for k in killed if k["pnl_pct_forensic"] > 0]
    forensic_losing = [k for k in killed if k["pnl_pct_forensic"] < 0]
    forensic_flat = [k for k in killed if k["pnl_pct_forensic"] == 0]
    sum_forensic_pnl = sum(k["pnl_pct_forensic"] for k in killed)
    median_forensic_pnl = statistics.median(k["pnl_pct_forensic"] for k in killed)

    print(f"  Killed forensic PnL>0 : {len(forensic_winning):,} ({len(forensic_winning)/len(killed)*100:.1f}%)")
    print(f"  Killed forensic PnL<0 : {len(forensic_losing):,} ({len(forensic_losing)/len(killed)*100:.1f}%)")
    print(f"  Killed forensic PnL=0 : {len(forensic_flat):,}")
    print(f"  Sum PnL_pct forensic  : {sum_forensic_pnl:+.2f}%")
    print(f"  Median PnL_pct forensic: {median_forensic_pnl:+.4f}%")

    # Breakdown por kill_reason
    print(f"\n  Por razon de kill:")
    for reason in ["pf_low", "sharpe_low", "tf_filter", "few_forensic_trades"]:
        r_killed = [k for k in killed if k["kill_reason"] == reason]
        if not r_killed:
            continue
        r_win = [k for k in r_killed if k["pnl_pct_forensic"] > 0]
        r_pnl = sum(k["pnl_pct_forensic"] for k in r_killed)
        r_median = statistics.median(k["pnl_pct_forensic"] for k in r_killed)
        print(f"    {reason:25s}: n={len(r_killed):>4} | ganadores={len(r_win):>4} "
              f"({len(r_win)/len(r_killed)*100:5.1f}%) | "
              f"sum PnL={r_pnl:+9.2f}% | median PnL={r_median:+7.4f}%")

    # 6. VEREDICTO
    print(f"\n{'=' * 70}")
    print("VEREDICTO")
    print(f"{'=' * 70}")

    # Priorizar datos reales si hay suficientes (>=30 combos con trades)
    if len(with_trades) >= 30:
        if total_pnl > 0 and len(winning) > len(losing):
            verdict_source = "PRODUCCION REAL"
            verdict = (f"V8 ES DEMASIADO AGRESIVO: de {len(with_trades)} killed con trades reales, "
                       f"{len(winning)} ganan ({len(winning)/len(with_trades)*100:.0f}%) y "
                       f"PnL total = +${total_pnl:.2f}. Matar estos cuesta dinero.")
        elif total_pnl < 0 and len(losing) > len(winning):
            verdict_source = "PRODUCCION REAL"
            verdict = (f"V8 TIENE RAZON: de {len(with_trades)} killed con trades reales, "
                       f"{len(losing)} pierden ({len(losing)/len(with_trades)*100:.0f}%) y "
                       f"PnL total = ${total_pnl:.2f}. Matar estos ahorra dinero.")
        else:
            verdict_source = "PRODUCCION REAL (mixto)"
            verdict = (f"RESULTADO MIXTO prod: {len(winning)} ganan vs {len(losing)} pierden. "
                       f"PnL neto = {'+' if total_pnl>=0 else ''}{total_pnl:.2f}.")
    else:
        # Fallback: forensic (historia completa, ~100 trades por combo)
        verdict_source = f"FORENSIC (prod tiene solo {len(with_trades)} combos con trades, insuficiente)"
        pct_losing = len(forensic_losing) / len(killed) * 100
        pct_winning = len(forensic_winning) / len(killed) * 100
        if sum_forensic_pnl < 0 and pct_losing > 50:
            verdict = (f"V8 TIENE RAZON (forensic): {pct_losing:.1f}% de los killed "
                       f"pierden plata en historia completa ({len(forensic_losing)} de {len(killed)}). "
                       f"Sum PnL forensic = {sum_forensic_pnl:+.2f}%. "
                       f"Median PnL = {median_forensic_pnl:+.4f}%.")
        elif sum_forensic_pnl > 0 and pct_winning > 50:
            verdict = (f"V8 ES DEMASIADO AGRESIVO (forensic): {pct_winning:.1f}% de los killed "
                       f"ganan plata en historia completa. Sum PnL = +{sum_forensic_pnl:.2f}%. "
                       f"Median PnL = {median_forensic_pnl:+.4f}%. Perderiamos edge.")
        else:
            verdict = (f"MIXTO (forensic): {pct_winning:.1f}% ganan / {pct_losing:.1f}% pierden. "
                       f"Sum PnL = {sum_forensic_pnl:+.2f}%. Decision caso por caso.")

    print(f"[Fuente: {verdict_source}]")
    print(verdict)

    # 6. Guardar
    output = {
        "timestamp": t0.isoformat(),
        "dataset": str(DATASET),
        "db_production": db_prod,
        "v8_gates": {
            "allowed_tfs": sorted(V8_ALLOWED_TFS),
            "min_pf": V8_MIN_PF,
            "min_sharpe": V8_MIN_SHARPE,
            "min_trades_forensic": V8_MIN_TRADES_FORENSIC,
        },
        "counts": {
            "v7_approved": len(approved),
            "v8_killed": len(killed),
            "v8_survived": len(survived),
            "killed_with_prod_trades": len(with_trades),
            "killed_no_prod_trades": len(no_trades),
            "killed_winning_in_prod": len(winning),
            "killed_losing_in_prod": len(losing),
            "killed_neutral_in_prod": len(neutral),
        },
        "kill_reasons": reasons,
        "killed_by_tf": killed_by_tf,
        "total_prod_pnl_killed": round(total_pnl, 2),
        "total_prod_trades_killed": total_trades,
        "avg_wr_killed": round(avg_wr, 2) if avg_wr else 0,
        "top_winners_killed": top_winners if with_trades else [],
        "top_losers_killed": top_losers if with_trades else [],
        "verdict": verdict,
        "verdict_source": verdict_source,
        "forensic_summary": {
            "winning_pnl_gt_0": len(forensic_winning),
            "losing_pnl_lt_0": len(forensic_losing),
            "flat_pnl_eq_0": len(forensic_flat),
            "sum_pnl_pct": round(sum_forensic_pnl, 2),
            "median_pnl_pct": round(median_forensic_pnl, 4),
        },
        "all_killed": killed,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nGuardado: {OUTPUT_JSON}")

    # Markdown resumen
    md = [f"# Audit V8 Killed Bots — {t0.strftime('%Y-%m-%d %H:%M')}"]
    md.append(f"\n**DB produccion**: `{db_prod}`")
    md.append(f"\n## Counts")
    md.append(f"- V7 aprobados: {len(approved):,}")
    md.append(f"- V8 killed: {len(killed):,}  ({len(killed)/len(approved)*100:.1f}%)")
    md.append(f"- V8 survived: {len(survived):,}")
    md.append(f"\n### Razones de kill")
    for r, n in sorted(reasons.items(), key=lambda x: -x[1]):
        md.append(f"- `{r}`: {n}")
    md.append(f"\n### Killed por TF")
    for tf, n in sorted(killed_by_tf.items()):
        md.append(f"- `{tf}`: {n}")
    md.append(f"\n## Performance en produccion (killed)")
    md.append(f"- Con trades reales: {len(with_trades):,}")
    md.append(f"- Sin trades       : {len(no_trades):,}")
    if with_trades:
        md.append(f"- Ganando (PnL+)   : {len(winning):,}")
        md.append(f"- Perdiendo (PnL-) : {len(losing):,}")
        md.append(f"- Neutral (PnL=0)  : {len(neutral):,}")
        md.append(f"- **PnL total killed**: `{'+' if total_pnl>=0 else ''}{total_pnl:.2f} USD`")
        md.append(f"- Trades totales   : {total_trades:,}")
        md.append(f"- WR promedio      : {avg_wr:.2f}%")
    md.append(f"\n## VEREDICTO")
    md.append(f"\n> {verdict}")

    if with_trades and winning:
        md.append(f"\n### Top 10 winners que V8 mataria")
        md.append(f"| Strategy | Symbol | TF | Trades | WR% | PnL | PF_for |")
        md.append(f"|---|---|---|---:|---:|---:|---:|")
        for k in top_winners:
            p = k["production"]
            md.append(f"| {k['strategy']} | {k['symbol']} | {k['timeframe']} | "
                      f"{p['trades_clean']} | {p['wr_clean']:.1f} | {p['pnl_usd_total']:+.2f} | {k['pf_forensic']:.2f} |")

    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(md))
    print(f"Guardado: {OUTPUT_MD}")

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nTiempo total: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
