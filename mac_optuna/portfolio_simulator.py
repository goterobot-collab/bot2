#!/usr/bin/env python3
"""
Portfolio Simulator — $100 Investment Plan using Grail Strategies
Reads combined_grails.json and creates a realistic investment plan.
"""

import json
import math
import statistics
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent / "data"
GRAILS_FILE = DATA_DIR / "combined_grails.json"
OUTPUT_FILE = DATA_DIR / "portfolio_plan.json"

INITIAL_CAPITAL = 100.0
MAX_BOTS = 10
MAX_ALLOC_PCT = 0.15  # Kelly cap 15%
MAX_LEVERAGE = 3       # Conservative max
COMMISSION = 0.001     # 0.1% per side
SLIPPAGE = 0.0005      # 0.05%
TRADING_COST = (COMMISSION * 2) + SLIPPAGE  # round trip


def load_grails():
    with open(GRAILS_FILE) as f:
        return json.load(f)


def compute_risk_score(g):
    """Compute a 0-10 risk score based on multiple factors."""
    score = 0

    # 1. Test WR (max 2.5 pts) — higher is safer
    wr = g["test"]["wr"]
    score += min(2.5, (wr - 70) / 12)  # 70->0, 100->2.5

    # 2. Low wr_diff (max 2 pts) — small gap = less overfit
    wr_diff = g.get("wr_diff", 30)
    score += max(0, 2 - wr_diff / 10)  # 0->2, 20->0

    # 3. Full Sharpe (max 2 pts)
    sharpe = g["full"].get("sharpe", 0)
    score += min(2, sharpe / 5)  # 0->0, 10->2

    # 4. Profit factor (max 1.5 pts)
    pf = g["full"].get("pf", 1)
    score += min(1.5, (pf - 1) / 4)  # 1->0, 7->1.5

    # 5. Trades count — enough sample size (max 1 pt)
    trades = g["full"].get("trades", 0)
    score += min(1, trades / 50)  # 50+ trades -> 1

    # 6. Low max drawdown (max 1 pt)
    dd = g["full"].get("max_dd", 50)
    score += max(0, 1 - dd / 30)  # 0->1, 30->0

    return round(min(10, score), 2)


def estimate_monthly_trades(g):
    """Estimate trades per month based on timeframe and historical data."""
    tf = g["timeframe"]
    total_trades = g["full"]["trades"]

    # Estimate period length based on data (2019-2026 ~ 84 months max)
    # But most strategies have varying data lengths; approximate from duration
    dur = g.get("duration", {})
    avg_h = dur.get("avg_h", 24)

    # Rough: total trades / estimated months of data
    # Use 24 months as rough estimate for test+train period
    tf_months = {"5m": 12, "15m": 18, "1h": 24, "4h": 36, "1d": 60}
    est_months = tf_months.get(tf, 24)

    return max(1, total_trades / est_months)


def kelly_fraction(wr_pct, avg_win_loss_ratio):
    """Kelly criterion: f* = (p*b - q) / b where p=win%, b=win/loss ratio, q=1-p"""
    p = wr_pct / 100
    q = 1 - p
    b = avg_win_loss_ratio
    if b <= 0:
        return 0
    kelly = (p * b - q) / b
    return max(0, min(kelly, MAX_ALLOC_PCT))


def estimate_avg_win_loss(g):
    """Estimate avg win/loss ratio from PnL, WR, and trades."""
    wr = g["test"]["wr"] / 100
    pnl = g["test"]["pnl"] / 100  # as fraction
    trades = max(g["test"].get("trades", 10), 5)

    wins = max(1, round(trades * wr))
    losses = max(1, trades - wins)

    # avg_pnl_per_trade = pnl / trades (as fraction of capital)
    avg_pnl = pnl / trades

    # Decompose: wins * avg_win - losses * avg_loss = total_pnl
    # Assume avg_loss ~ mae_p95 or estimate from max_dd
    mae = g["test"].get("mae_p95", 0.01)
    avg_loss = max(mae, 0.005)

    total_loss = losses * avg_loss
    total_win_pnl = pnl + total_loss
    avg_win = total_win_pnl / wins if wins > 0 else avg_loss

    ratio = avg_win / avg_loss if avg_loss > 0 else 2
    return max(0.5, min(ratio, 10))


# ── ASSET CORRELATION GROUPS ──
# Group correlated assets to ensure diversification
CORRELATION_GROUPS = {
    "BTC": ["BTC"],
    "ETH": ["ETH"],
    "LARGE_CAP": ["SOL", "BNB", "XRP", "ADA", "AVAX", "DOT", "MATIC", "LINK"],
    "MEME": ["DOGE", "SHIB", "PEPE", "FLOKI", "WIF", "BONK"],
    "DEFI": ["UNI", "AAVE", "MKR", "COMP", "CRV", "SUSHI", "SNX"],
    "L2": ["ARB", "OP", "STRK", "MANTA", "BLAST"],
    "AI": ["FET", "RNDR", "AGIX", "TAO", "WLD"],
    "GAMING": ["AXS", "SAND", "MANA", "GALA", "IMX"],
}


def get_asset_group(symbol):
    """Return correlation group for an asset."""
    sym_upper = symbol.upper()
    for group, assets in CORRELATION_GROUPS.items():
        for a in assets:
            if a in sym_upper:
                return group
    return f"OTHER_{symbol[:4]}"  # Unique group for uncategorized


def select_top_bots(grails):
    """Select top 10 diverse, safe bots."""
    # Score all grails
    scored = []
    for g in grails:
        risk = compute_risk_score(g)
        if risk < 6:
            continue
        # Prefer higher trades (more reliable)
        if g["full"]["trades"] < 20:
            continue
        # Prefer reasonable test PnL (not too low)
        if g["test"]["pnl"] < 5:
            continue
        scored.append({**g, "_risk_score": risk})

    # Sort by composite: risk_score * 0.4 + normalized(sharpe) * 0.3 + normalized(test_pnl) * 0.3
    max_sharpe = max(s["full"]["sharpe"] for s in scored) if scored else 1
    max_pnl = max(s["test"]["pnl"] for s in scored) if scored else 1

    for s in scored:
        composite = (
            s["_risk_score"] / 10 * 0.4 +
            min(s["full"]["sharpe"] / max_sharpe, 1) * 0.3 +
            min(s["test"]["pnl"] / max_pnl, 1) * 0.3
        )
        s["_composite"] = round(composite, 4)

    scored.sort(key=lambda x: x["_composite"], reverse=True)

    # Greedy selection: no repeated asset, max 1 per correlation group
    selected = []
    used_symbols = set()
    used_groups = set()
    used_strategies = set()  # Diversity in strategy type too

    for s in scored:
        sym = s["symbol"]
        group = get_asset_group(sym)
        strat = s["strategy"]

        # Skip if asset already used
        if sym in used_symbols:
            continue
        # Skip if correlation group already used (allow max 2 from OTHER)
        if group in used_groups and not group.startswith("OTHER"):
            continue
        # Allow max 2 of same strategy type
        strat_count = sum(1 for sel in selected if sel["strategy"] == strat)
        if strat_count >= 2:
            continue

        selected.append(s)
        used_symbols.add(sym)
        used_groups.add(group)
        used_strategies.add(strat)

        if len(selected) >= MAX_BOTS:
            break

    return selected


def allocate_capital(bots, total=INITIAL_CAPITAL):
    """Allocate capital using Kelly criterion, capped at 15% per bot."""
    kellys = []
    for b in bots:
        wl_ratio = estimate_avg_win_loss(b)
        kf = kelly_fraction(b["test"]["wr"], wl_ratio)
        kellys.append(max(0.05, kf))  # Floor at 5%

    # Normalize to sum to 1
    total_k = sum(kellys)
    weights = [k / total_k for k in kellys]

    # Cap at MAX_ALLOC_PCT and redistribute excess
    capped = [min(w, MAX_ALLOC_PCT) for w in weights]
    excess = 1 - sum(capped)
    if excess > 0.01:
        uncapped = [i for i, w in enumerate(weights) if w < MAX_ALLOC_PCT]
        if uncapped:
            per_bot = excess / len(uncapped)
            for i in uncapped:
                capped[i] = min(capped[i] + per_bot, MAX_ALLOC_PCT)

    # Final normalize
    total_c = sum(capped)
    capped = [c / total_c for c in capped]

    allocations = []
    for i, b in enumerate(bots):
        lev = min(b.get("safe_leverage", 2), MAX_LEVERAGE)
        alloc_usd = round(total * capped[i], 2)
        allocations.append({
            "weight": round(capped[i], 4),
            "capital_usd": alloc_usd,
            "leverage": lev,
            "effective_capital": round(alloc_usd * lev, 2),
        })

    return allocations


def simulate_scenario(bots, allocations, wr_penalty=0, failed_bot_idx=None):
    """
    Simulate monthly returns for a scenario.
    wr_penalty: percentage points to subtract from WR (e.g., 10 = 10pp less)
    failed_bot_idx: index of bot that fails completely (returns -max_dd)
    """
    monthly_returns = []
    bot_details = []

    for i, (bot, alloc) in enumerate(zip(bots, allocations)):
        if i == failed_bot_idx:
            # Bot fails: lose max_dd worth of allocated capital
            dd = bot["full"].get("max_dd", 20) / 100
            monthly_ret = -alloc["capital_usd"] * dd * alloc["leverage"]
            bot_details.append({
                "symbol": bot["symbol"],
                "strategy": bot["strategy"],
                "status": "FAILED",
                "monthly_return_usd": round(monthly_ret, 2),
            })
            monthly_returns.append(monthly_ret)
            continue

        wr = max(50, bot["test"]["wr"] - wr_penalty)
        trades_month = estimate_monthly_trades(bot)

        # Expected PnL per trade (simplified model)
        wl_ratio = estimate_avg_win_loss(bot)
        mae = bot["test"].get("mae_p95", 0.01)
        avg_loss = max(mae, 0.005)
        avg_win = avg_loss * wl_ratio

        # Expected return per trade after costs
        p = wr / 100
        exp_per_trade = p * avg_win - (1 - p) * avg_loss - TRADING_COST

        # Monthly return on effective capital
        monthly_pnl_pct = exp_per_trade * trades_month
        monthly_ret = alloc["effective_capital"] * monthly_pnl_pct

        bot_details.append({
            "symbol": bot["symbol"],
            "strategy": bot["strategy"],
            "adj_wr": round(wr, 1),
            "trades_month": round(trades_month, 1),
            "exp_per_trade_pct": round(exp_per_trade * 100, 3),
            "monthly_return_usd": round(monthly_ret, 2),
            "monthly_return_pct": round(monthly_pnl_pct * 100, 2),
        })
        monthly_returns.append(monthly_ret)

    total_monthly = sum(monthly_returns)
    total_monthly_pct = total_monthly / INITIAL_CAPITAL * 100

    # Max DD estimate: worst single bot DD * its weight * leverage
    max_dds = []
    for i, (bot, alloc) in enumerate(zip(bots, allocations)):
        dd = bot["full"].get("max_dd", 20) / 100
        portfolio_dd = dd * alloc["weight"] * alloc["leverage"]
        max_dds.append(portfolio_dd)
    portfolio_max_dd = sum(sorted(max_dds, reverse=True)[:3])  # Top 3 correlated

    # Time to targets (compound monthly)
    targets = {
        "double_200": None,
        "1000": None,
        "10000": None,
        "100000": None,
    }

    if total_monthly_pct > 0:
        monthly_rate = total_monthly_pct / 100
        for label, target in [("double_200", 200), ("1000", 1000), ("10000", 10000), ("100000", 100000)]:
            if monthly_rate > 0:
                months = math.log(target / INITIAL_CAPITAL) / math.log(1 + monthly_rate)
                targets[label] = round(months, 1)

    return {
        "monthly_return_usd": round(total_monthly, 2),
        "monthly_return_pct": round(total_monthly_pct, 2),
        "annual_return_pct": round((1 + total_monthly_pct / 100) ** 12 * 100 - 100, 1),
        "portfolio_max_dd_pct": round(portfolio_max_dd * 100, 1),
        "months_to_double": targets["double_200"],
        "months_to_1000": targets["1000"],
        "months_to_10000": targets["10000"],
        "months_to_100000": targets["100000"],
        "bots": bot_details,
    }


def build_management_rules():
    return {
        "add_capital": {
            "rule": "Reinvertir ganancias al alcanzar +50% del capital asignado a cada bot",
            "detail": "Cuando un bot acumula $X >= 1.5x su asignación inicial, reinvertir el exceso en el mismo bot o redistribuir"
        },
        "stop_bot": {
            "rule": "Apagar bot si WR cae debajo de 55% en 20+ trades reales o DD > 2x backtest DD",
            "detail": "Monitorear WR rolling de 20 trades. Si WR < 55% o drawdown real supera 2x el max_dd del backtest, pausar y re-evaluar"
        },
        "scale_leverage": {
            "rule": "Escalar leverage solo después de 50+ trades reales con WR >= backtest WR - 5pp",
            "conditions": [
                "Mínimo 50 trades reales ejecutados",
                "WR real >= WR backtest - 5 puntos porcentuales",
                "Max DD real <= Max DD backtest",
                "Escalar de 2x a 3x primero, luego 3x a 5x si se mantiene 100+ trades"
            ]
        },
        "rebalance": {
            "rule": "Rebalancear mensualmente según performance",
            "detail": "Redistribuir capital de bots underperforming a los que mantienen WR"
        },
        "emergency_stop": {
            "rule": "Si portfolio cae -30% del capital total, apagar TODOS los bots",
            "detail": "Circuit breaker global. Revisar todas las estrategias antes de reiniciar"
        }
    }


def main():
    print("=" * 70)
    print("  PORTFOLIO SIMULATOR — $100 Grail Investment Plan")
    print("=" * 70)

    grails = load_grails()
    print(f"\nGrails cargados: {len(grails)}")

    # ── STEP 1: Select top bots ──
    bots = select_top_bots(grails)
    print(f"Bots seleccionados: {len(bots)}")

    # ── STEP 2: Allocate capital ──
    allocations = allocate_capital(bots)

    # ── STEP 3: Simulate scenarios ──
    optimistic = simulate_scenario(bots, allocations, wr_penalty=0)
    realistic = simulate_scenario(bots, allocations, wr_penalty=10)

    # Pessimistic: -20% WR + worst bot fails
    # Find the bot with highest allocation to be the "failed" one
    worst_idx = max(range(len(allocations)), key=lambda i: allocations[i]["capital_usd"])
    pessimistic = simulate_scenario(bots, allocations, wr_penalty=20, failed_bot_idx=worst_idx)

    # ── STEP 4: Management rules ──
    rules = build_management_rules()

    # ── BUILD OUTPUT ──
    bot_summaries = []
    for i, (bot, alloc) in enumerate(zip(bots, allocations)):
        bot_summaries.append({
            "rank": i + 1,
            "strategy": bot["strategy"],
            "symbol": bot["symbol"],
            "timeframe": bot["timeframe"],
            "test_wr": bot["test"]["wr"],
            "test_pnl": bot["test"]["pnl"],
            "full_sharpe": bot["full"]["sharpe"],
            "full_pf": bot["full"]["pf"],
            "full_trades": bot["full"]["trades"],
            "full_max_dd": bot["full"]["max_dd"],
            "wr_diff": bot.get("wr_diff", 0),
            "risk_score": bot["_risk_score"],
            "composite_score": bot["_composite"],
            "allocation_pct": round(alloc["weight"] * 100, 1),
            "capital_usd": alloc["capital_usd"],
            "leverage": alloc["leverage"],
            "effective_capital_usd": alloc["effective_capital"],
            "best_params": bot.get("best_params", {}),
        })

    plan = {
        "generated_at": datetime.now().isoformat(),
        "initial_capital": INITIAL_CAPITAL,
        "num_bots": len(bots),
        "max_leverage": MAX_LEVERAGE,
        "trading_costs": {
            "commission_per_side": COMMISSION,
            "slippage": SLIPPAGE,
            "round_trip": TRADING_COST,
        },
        "selected_bots": bot_summaries,
        "scenarios": {
            "optimistic": {
                "description": "Todos los bots rinden como en test (WR = test WR)",
                **optimistic,
            },
            "realistic": {
                "description": "WR baja 10pp vs test (degradación normal backtest→real)",
                **realistic,
            },
            "pessimistic": {
                "description": "WR baja 20pp + 1 bot falla completamente (pierde max_dd)",
                **pessimistic,
            },
        },
        "management_rules": rules,
    }

    # ── SAVE ──
    with open(OUTPUT_FILE, "w") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(f"\nPlan guardado en: {OUTPUT_FILE}")

    # ── PRINT SUMMARY ──
    print("\n" + "=" * 70)
    print("  TOP 10 BOTS SELECCIONADOS")
    print("=" * 70)
    print(f"{'#':<3} {'Estrategia':<22} {'Activo':<20} {'TF':<5} {'WR%':<6} {'Risk':<5} {'$USD':<7} {'Lev':<4} {'$Efect':<8}")
    print("-" * 70)
    for b in bot_summaries:
        print(f"{b['rank']:<3} {b['strategy']:<22} {b['symbol']:<20} {b['timeframe']:<5} "
              f"{b['test_wr']:<6.1f} {b['risk_score']:<5.1f} ${b['capital_usd']:<6.2f} "
              f"{b['leverage']}x    ${b['effective_capital_usd']:<7.2f}")

    print(f"\n{'Capital total:':<35} ${INITIAL_CAPITAL:.2f}")
    print(f"{'Capital efectivo (con leverage):':<35} ${sum(a['effective_capital'] for a in allocations):.2f}")

    print("\n" + "=" * 70)
    print("  ESCENARIOS DE RETORNO")
    print("=" * 70)

    for name, scenario in [("OPTIMISTA", optimistic), ("REALISTA", realistic), ("PESIMISTA", pessimistic)]:
        print(f"\n{'─' * 35}")
        print(f"  {name}")
        print(f"{'─' * 35}")
        print(f"  Retorno mensual:      ${scenario['monthly_return_usd']:>8.2f}  ({scenario['monthly_return_pct']:>+.1f}%)")
        print(f"  Retorno anual:        {scenario['annual_return_pct']:>+8.1f}%")
        print(f"  Max Drawdown portfolio: {scenario['portfolio_max_dd_pct']:>6.1f}%")
        print(f"  Meses para duplicar:  {scenario['months_to_double'] or 'N/A':>8}")
        print(f"  Meses para $1,000:    {scenario['months_to_1000'] or 'N/A':>8}")
        print(f"  Meses para $10,000:   {scenario['months_to_10000'] or 'N/A':>8}")
        print(f"  Meses para $100,000:  {scenario['months_to_100000'] or 'N/A':>8}")

    print("\n" + "=" * 70)
    print("  REGLAS DE GESTIÓN")
    print("=" * 70)
    for key, rule in rules.items():
        print(f"\n  [{key.upper()}]")
        print(f"  {rule['rule']}")
        if "detail" in rule:
            print(f"  → {rule['detail']}")
        if "conditions" in rule:
            for c in rule["conditions"]:
                print(f"    • {c}")

    print("\n" + "=" * 70)
    print("  DISCLAIMER")
    print("=" * 70)
    print("  Los retornos están basados en backtests históricos.")
    print("  El rendimiento pasado NO garantiza resultados futuros.")
    print("  Siempre usar dinero que puedas permitirte perder.")
    print("=" * 70)


if __name__ == "__main__":
    main()
