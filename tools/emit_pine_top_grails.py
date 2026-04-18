#!/usr/bin/env python3
"""Emit Pine v5 strategy code for top 20 ranked grails.

Reads results/sandbox_master_ranking.json (top 20).
Output: pine_sources/grails_auto/<i>_<strategy>_<sym>_<tf>.pine
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RANK = ROOT / "results" / "sandbox_master_ranking.json"
OUT = ROOT / "pine_sources" / "grails_auto"
OUT.mkdir(parents=True, exist_ok=True)

PINE_TEMPLATE = '''//@version=5
// Auto-emitted from sandbox_master_ranking #{rank}
// Strategy: {strategy}  Sym: {sym}  TF: {tf}
// Observed: WR {wr}%  trades {n}  PF {pf}  Plateau: {plateau}
// Score: {score}
//
// NOTE: This is a TEMPLATE. The signal logic must be ported manually from
// strategies_v7/ Python to Pine. The risk wrapper below is canonical.

strategy("{strategy} {sym} {tf}", overlay=true,
     initial_capital=10000,
     default_qty_type=strategy.percent_of_equity,
     default_qty_value=100,
     commission_type=strategy.commission.percent,
     commission_value=0.1,
     slippage=2,
     process_orders_on_close=false)

// === RISK MANAGEMENT (canonical V8 wrapper) ==============================
riskPct      = input.float(1.0, "Risk per trade %", minval=0.1, maxval=5.0)
maxDDHalt    = input.float(20.0, "Halt at equity DD%")
slPct        = input.float(40.0, "Stop loss %", minval=1.0)
useRegime    = input.bool(true, "Daily EMA200 regime filter")

// Daily regime
dailyEma200 = request.security(syminfo.tickerid, "D", ta.ema(close, 200))
macroBull   = not useRegime or close > dailyEma200

// Equity halt
equityPeak = ta.max(strategy.equity)
curDD      = (equityPeak - strategy.equity) / equityPeak * 100
halted     = curDD > maxDDHalt

// === SIGNAL LOGIC (PORT FROM PYTHON HERE) ================================
// TODO: implement {strategy} signal in Pine (see strategies_v7/strategies_tv2_batch*.py)
longSignal  = false  // placeholder
shortSignal = false  // placeholder

// === ENTRY / EXIT ========================================================
canTrade = macroBull and not halted

if longSignal and canTrade and strategy.position_size == 0
    strategy.entry("L", strategy.long)
    strategy.exit("L_sl", "L", stop = strategy.position_avg_price * (1 - slPct/100))

if shortSignal and canTrade and strategy.position_size == 0
    strategy.entry("S", strategy.short)
    strategy.exit("S_sl", "S", stop = strategy.position_avg_price * (1 + slPct/100))

bgcolor(halted ? color.new(color.red, 80) : na, title="HALTED")
bgcolor(not macroBull ? color.new(color.gray, 92) : na, title="Macro bear")
'''


def main():
    if not RANK.exists():
        print("ranking file missing")
        return
    grails = json.loads(RANK.read_text())[:20]
    for i, g in enumerate(grails, 1):
        fname = f"{i:02d}_{g['strategy']}_{g['symbol']}_{g['tf']}.pine"
        body = PINE_TEMPLATE.format(
            rank=i, strategy=g['strategy'], sym=g['symbol'], tf=g['tf'],
            wr=g['wr'], n=g['trades'], pf=g['pf'],
            plateau="YES" if g.get('plateau') else "no",
            score=g.get('score', 0),
        )
        (OUT / fname).write_text(body)
    print(f"PINE_EMITTER DONE: emitted {len(grails)} pine files to {OUT}")


if __name__ == "__main__":
    main()
