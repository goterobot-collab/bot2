#!/usr/bin/env python3
"""
Emit a ready-to-paste Pine v5 strategy file for each top grail found by
grail_loop.py. Reads results/top_grails.md (ranked & deduped), writes one
.pine per grail into pine_sources/grails/.

Usage:
    python3 tools/emit_pine.py                     # top 50
    python3 tools/emit_pine.py --top 200
    python3 tools/emit_pine.py --input results/grails_loop.jsonl --top 100

The generated Pine files have exact hard-coded parameters + a date-range
filter matching the dataset the backtest ran on, so you can paste into
TradingView and see the same behavior (modulo TV's own execution model).
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path


OUT_DIR = Path("pine_sources/grails")
HEADER = """//@version=5
// AUTO-GENERATED FROM GRAIL LOOP  (tools/emit_pine.py)
// Strategy family: {family}
// Asset: {asset}   TF: {tf}
// In-sample metrics:  WR {wr}%  PF {pf}  ret {ret}%  DD {dd}%  trades {trades}
// Params: {params}
// Exit:   {exit_cfg}
strategy("{title}",
     overlay=true,
     initial_capital=10000,
     default_qty_type=strategy.percent_of_equity,
     default_qty_value=100,
     commission_type=strategy.commission.percent,
     commission_value=0.1,
     slippage=2,
     process_orders_on_close=false)
"""


def date_gate(asset: str) -> str:
    """Match the in-sample window of the dataset used."""
    if asset.startswith("ETH"):
        start = "timestamp(2018, 08, 02, 0, 0)"
    else:
        start = "timestamp(2021, 09, 29, 0, 0)"
    return (f"useWindow = input.bool(true, \"Apply in-sample window\")\n"
            f"startTs = {start}\n"
            f"inWindow = not useWindow or time >= startTs\n")


def emit_exit_block(exit_cfg: dict) -> str:
    """Return Pine lines that add hard SL/TP + trailing stop + timeout."""
    lines = []
    atr_needed = any(exit_cfg.get(k) for k in ("sl_atr", "tp_atr", "trail_atr"))
    if atr_needed:
        lines.append("atrV = ta.atr(14)")
    sl = exit_cfg.get("sl_atr")
    tp = exit_cfg.get("tp_atr")
    trail = exit_cfg.get("trail_atr")
    tmo = exit_cfg.get("timeout")

    # entry hook
    entry_extra = []
    if sl or tp:
        sl_str = f"strategy.position_avg_price - {sl}*atrV" if sl else "na"
        tp_str = f"strategy.position_avg_price + {tp}*atrV" if tp else "na"
        entry_extra.append(
            f'strategy.exit("sltp", "long", stop={sl_str}, limit={tp_str})'
        )
    if trail:
        entry_extra.append(f'strategy.exit("trail", "long", trail_price=high - {trail}*atrV, trail_offset=0)')

    if tmo:
        lines.append("var int entryBar = na")
        lines.append("if longCond and strategy.position_size == 0")
        lines.append("    strategy.entry(\"long\", strategy.long)")
        lines.append("    entryBar := bar_index")
        for e in entry_extra:
            lines.append(f"    {e}")
        lines.append(f"if strategy.position_size > 0 and not na(entryBar) and bar_index - entryBar >= {tmo}")
        lines.append("    strategy.close(\"long\")")
        lines.append("    entryBar := na")
    else:
        lines.append("if longCond and strategy.position_size == 0")
        lines.append("    strategy.entry(\"long\", strategy.long)")
        for e in entry_extra:
            lines.append(f"    {e}")
    return "\n".join(lines)


def emit_bb_trend_rejoin(p, e):
    return f"""
bbLen = {p['bb_len']}
bbMult = {p['bb_mult']}
trendLen = {p['trend']}

basis = ta.sma(close, bbLen)
dev = bbMult * ta.stdev(close, bbLen)
lower = basis - dev
upper = basis + dev
t = ta.ema(close, trendLen)

longCond = inWindow and ta.crossover(close, lower) and close > t
exitCond = ta.crossover(close, basis)

{emit_exit_block(e)}

if strategy.position_size > 0 and exitCond
    strategy.close("long")

plot(basis, color=color.orange)
plot(lower, color=color.green)
plot(upper, color=color.red)
plot(t,     color=color.gray, title="Trend EMA")
"""


def emit_rsi2_regime(p, e):
    return f"""
rsiLen   = {p['rsi_len']}
rsiBuy   = {p['rsi_buy']}
smaTrend = {p['sma_trend']}
slopeBars = {p['slope_bars']}
exitSma   = {p['exit_sma']}

slow = ta.sma(close, smaTrend)
fast = ta.sma(close, exitSma)
r    = ta.rsi(close, rsiLen)
rising = slow > slow[slopeBars]

longCond = inWindow and rising and close > slow and r < rsiBuy
exitCond = close > fast

{emit_exit_block(e)}

if strategy.position_size > 0 and exitCond
    strategy.close("long")

plot(slow, color=color.orange)
plot(fast, color=color.blue)
"""


def emit_supertrend_atr(p, e):
    return f"""
stAtr    = {p['st_atr']}
stMult   = {p['st_mult']}
emaTrend = {p['ema_trend']}

[_st, dir] = ta.supertrend(stMult, stAtr)
trend = ta.ema(close, emaTrend)
flipUp = dir == -1 and dir[1] == 1
flipDn = dir == 1 and dir[1] == -1

longCond = inWindow and flipUp and close > trend

{emit_exit_block(e)}

if strategy.position_size > 0 and flipDn
    strategy.close("long")

plot(trend, color=color.orange, title="Trend EMA")
"""


def emit_ema_cross_trend(p, e):
    return f"""
fastLen = {p['fast']}
slowLen = {p['slow']}
trendLen = {p['trend']}

ef = ta.ema(close, fastLen)
es = ta.ema(close, slowLen)
et = ta.ema(close, trendLen)

longCond = inWindow and ta.crossover(ef, es) and close > et
exitCond = ta.crossunder(ef, es)

{emit_exit_block(e)}

if strategy.position_size > 0 and exitCond
    strategy.close("long")

plot(ef, color=color.blue)
plot(es, color=color.orange)
plot(et, color=color.gray)
"""


def emit_macd_trend(p, e):
    return f"""
fastLen = {p['fast']}
slowLen = {p['slow']}
sigLen  = {p['sig']}
trendLen = {p['trend']}

[m, s, _h] = ta.macd(close, fastLen, slowLen, sigLen)
t = ta.ema(close, trendLen)

longCond = inWindow and ta.crossover(m, s) and close > t
exitCond = ta.crossunder(m, s)

{emit_exit_block(e)}

if strategy.position_size > 0 and exitCond
    strategy.close("long")

plot(t, color=color.orange, title="Trend EMA")
"""


def emit_donchian_trail(p, e):
    return f"""
dcLen = {p['dc_len']}

upper = ta.highest(high, dcLen)[1]
longCond = inWindow and close > upper
exitCond = false

{emit_exit_block(e)}

plot(upper, color=color.teal, title="DC upper")
"""


def emit_keltner_squeeze(p, e):
    return f"""
bbLen = {p['len']}
bbMult = {p['bb_mult']}
kcMult = {p['kc_mult']}
trendLen = {p['trend']}

bbMid = ta.sma(close, bbLen)
bbStd = ta.stdev(close, bbLen)
bbUp = bbMid + bbMult * bbStd
bbDn = bbMid - bbMult * bbStd
kcMid = ta.ema(close, bbLen)
aK = ta.atr(bbLen)
kcUp = kcMid + kcMult * aK
kcDn = kcMid - kcMult * aK
squeeze = bbUp < kcUp and bbDn > kcDn
released = squeeze[1] and not squeeze
t = ta.ema(close, trendLen)

longCond = inWindow and released and close > bbMid and close > t
exitCond = ta.crossunder(close, bbMid)

{emit_exit_block(e)}

if strategy.position_size > 0 and exitCond
    strategy.close("long")

plot(bbMid, color=color.orange)
plot(t,     color=color.gray, title="Trend EMA")
"""


def emit_zscore_revert(p, e):
    return f"""
zLen    = {p['z_len']}
zEnter  = {p['z_enter']}
trendLen = {p['trend']}

m  = ta.sma(close, zLen)
sd = ta.stdev(close, zLen)
z  = (close - m) / math.max(sd, 0.0000001)
t  = ta.ema(close, trendLen)

longCond = inWindow and ta.crossover(z, -zEnter) and close > t
exitCond = ta.crossover(z, 0.0)

{emit_exit_block(e)}

if strategy.position_size > 0 and exitCond
    strategy.close("long")

plot(t, color=color.orange, title="Trend EMA")
"""


def emit_psar_trend(p, e):
    return f"""
hiLen  = {p['hi_len']}
exitLen = {p['exit_len']}
trendLen = {p['trend']}
slopeBars = {p['slope_bars']}

priorHigh = ta.highest(high, hiLen)[1]
priorLow  = ta.lowest(low,  exitLen)[1]
t = ta.ema(close, trendLen)
trendUp = t > t[slopeBars]

longCond = inWindow and close > priorHigh and close > t and trendUp
exitCond = close < priorLow

{emit_exit_block(e)}

if strategy.position_size > 0 and exitCond
    strategy.close("long")

plot(t, color=color.orange, title="Trend EMA")
"""


def emit_adx_pullback(p, e):
    return f"""
adxLen   = {p['adx_len']}
adxMin   = {p['adx_min']}
emaLen   = {p['ema_len']}
lookback = {p['lookback']}

[di_p, di_m, adx] = ta.dmi(adxLen, adxLen)
eP = ta.ema(close, emaLen)
touched = ta.lowest(low, lookback) <= eP
recovery = close > close[1] and close > eP

longCond = inWindow and adx > adxMin and di_p > di_m and touched and recovery
exitCond = ta.crossunder(close, eP)

{emit_exit_block(e)}

if strategy.position_size > 0 and exitCond
    strategy.close("long")

plot(eP, color=color.orange, title="EMA pullback")
"""


EMITTERS = {
    "bb_trend_rejoin":  emit_bb_trend_rejoin,
    "rsi2_regime":      emit_rsi2_regime,
    "supertrend_atr":   emit_supertrend_atr,
    "ema_cross_trend":  emit_ema_cross_trend,
    "macd_trend":       emit_macd_trend,
    "donchian_trail":   emit_donchian_trail,
    "keltner_squeeze":  emit_keltner_squeeze,
    "zscore_revert":    emit_zscore_revert,
    "psar_trend":       emit_psar_trend,
    "adx_pullback":     emit_adx_pullback,
}


def parse_top_grails(path: Path):
    """Parse the markdown table in top_grails.md. Returns list of dicts."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith("| ") or "---" in line or "Rank" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 11:
            continue
        try:
            rank = int(cells[0])
        except ValueError:
            continue
        score = float(cells[1])
        strategy = cells[2].strip("`")
        asset = cells[3]
        tf = cells[4]
        trades = int(cells[5])
        wr = float(cells[6])
        pf = float(cells[7])
        ret = float(cells[8])
        dd = float(cells[9])
        params_str = cells[10].strip("`")
        exit_str = cells[11].strip("`")
        try:
            params = json.loads(params_str)
            exit_cfg = json.loads(exit_str) if exit_str else {}
        except json.JSONDecodeError:
            continue
        rows.append({
            "rank": rank, "score": score, "strategy": strategy, "asset": asset,
            "tf": tf, "trades": trades, "wr": wr, "pf": pf, "ret": ret, "dd": dd,
            "params": params, "exit_cfg": exit_cfg,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/top_grails.md")
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # wipe previous
    for p in OUT_DIR.glob("*.pine"):
        p.unlink()

    rows = parse_top_grails(Path(args.input))
    rows = rows[: args.top]
    if not rows:
        print(f"No grails parsed from {args.input}")
        return

    index = [
        "# Top grails — Pine files index",
        "",
        "| Rank | File | Strategy | Asset | WR % | PF | Ret % | DD % | Trades |",
        "|------|------|----------|-------|------|-----|-------|------|--------|",
    ]
    for r in rows:
        emit = EMITTERS.get(r["strategy"])
        if emit is None:
            continue
        title = f"GRAIL #{r['rank']:03d} {r['strategy']} {r['asset']} {r['tf']}"
        header = HEADER.format(
            family=r["strategy"], asset=r["asset"], tf=r["tf"],
            wr=r["wr"], pf=r["pf"], ret=r["ret"], dd=r["dd"], trades=r["trades"],
            params=json.dumps(r["params"]), exit_cfg=json.dumps(r["exit_cfg"]),
            title=title,
        )
        body = emit(r["params"], r["exit_cfg"])
        code = header + "\n" + date_gate(r["asset"]) + "\n" + body

        fname = f"grail_{r['rank']:03d}_{r['strategy']}_{r['asset']}_{r['tf']}.pine"
        (OUT_DIR / fname).write_text(code)
        index.append(
            f"| {r['rank']} | `{fname}` | `{r['strategy']}` | {r['asset']} | "
            f"{r['wr']} | {r['pf']} | {r['ret']} | {r['dd']} | {r['trades']} |"
        )
    (OUT_DIR / "INDEX.md").write_text("\n".join(index) + "\n")
    print(f"Wrote {len(rows)} Pine files to {OUT_DIR}/")
    print(f"Index: {OUT_DIR}/INDEX.md")


if __name__ == "__main__":
    main()
