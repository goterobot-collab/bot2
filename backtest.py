#!/usr/bin/env python3
"""
Backtest every strategy × asset × timeframe against CSVs in data/.

Supports per-strategy hard SL/TP (in ATR multiples of entry), ATR trailing
stop, and bar-count timeout. Entries execute at next-bar open; exits check
SL/TP/trail on each bar using bar high/low (stop-at-price, not stop-at-close).

Output:
    results/summary.csv          — every combo, all metrics
    results/summary.md           — qualifying combos sorted by WR desc
    results/grails.md            — strict grail filter: WR>60 AND return>0 AND maxDD<30
    results/by_wr/NN_*.md        — per-combo detail with trade log

Usage:
    python3 backtest.py
    python3 backtest.py --min-wr 0.60 --min-trades 5 --grail-dd 30
"""
from __future__ import annotations
import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from strategies import STRATEGIES
from strategies.library import SOURCES
from strategies.indicators import atr as atr_ind

DATA_DIR = Path("data")
OUT_DIR = Path("results")


@dataclass
class Trade:
    entry_time: int
    entry_price: float
    exit_time: int
    exit_price: float
    bars_held: int
    ret_pct: float
    exit_reason: str


@dataclass
class Metrics:
    strategy: str
    asset: str
    tf: str
    trades: int
    wins: int
    losses: int
    wr: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    total_return_pct: float
    max_drawdown_pct: float
    first_bar: str
    last_bar: str
    exits_by_reason: str


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def simulate(df: pd.DataFrame, entry: pd.Series, exit_sig: pd.Series,
             fee: float, slippage: float,
             sl_atr: float | None, tp_atr: float | None,
             trail_atr: float | None, timeout: int | None) -> list[Trade]:
    """Long-only, one position at a time, next-bar-open entry. Hard SL/TP
    checked on each bar using low/high. Trailing stop updates on each bar."""
    e = entry.shift(1).fillna(False).values
    x_sig = exit_sig.shift(1).fillna(False).values
    op = df["open"].values
    hi = df["high"].values
    lo = df["low"].values
    cl = df["close"].values
    a = atr_ind(df["high"], df["low"], df["close"], 14).values
    ts = ((df.index - pd.Timestamp("1970-01-01", tz="UTC"))
          .total_seconds().astype("int64").values)
    n = len(df)

    trades: list[Trade] = []
    pos = False
    ep = 0.0; et = 0; ebar = 0; atr_at_entry = 0.0
    hard_sl = None; hard_tp = None; trail_stop = None

    for i in range(n):
        if not pos:
            if e[i]:
                ep = op[i] * (1.0 + slippage)
                et = int(ts[i]); ebar = i
                atr_at_entry = a[i] if not np.isnan(a[i]) else 0.0
                hard_sl = ep - sl_atr * atr_at_entry if sl_atr and atr_at_entry > 0 else None
                hard_tp = ep + tp_atr * atr_at_entry if tp_atr and atr_at_entry > 0 else None
                trail_stop = (ep - trail_atr * atr_at_entry) if trail_atr and atr_at_entry > 0 else None
                pos = True
            continue

        # in position: update trailing stop based on current bar
        if trail_atr and atr_at_entry > 0:
            cand = hi[i] - trail_atr * atr_at_entry
            trail_stop = max(trail_stop, cand) if trail_stop is not None else cand

        reason = None; xp = None
        # priority: hard SL hit intra-bar → trail stop hit → hard TP → signal exit → timeout → end-of-data
        if hard_sl is not None and lo[i] <= hard_sl:
            xp = hard_sl * (1.0 - slippage); reason = "sl"
        elif trail_stop is not None and lo[i] <= trail_stop:
            xp = trail_stop * (1.0 - slippage); reason = "trail"
        elif hard_tp is not None and hi[i] >= hard_tp:
            xp = hard_tp * (1.0 - slippage); reason = "tp"
        elif x_sig[i]:
            xp = op[i] * (1.0 - slippage); reason = "signal"
        elif timeout is not None and (i - ebar) >= timeout:
            xp = op[i] * (1.0 - slippage); reason = "timeout"
        elif i == n - 1:
            xp = op[i] * (1.0 - slippage); reason = "eod"

        if xp is not None:
            gross = xp / ep - 1.0
            net = gross - 2 * fee
            trades.append(Trade(et, ep, int(ts[i]), xp, i - ebar, net * 100.0, reason))
            pos = False; hard_sl = hard_tp = trail_stop = None
    return trades


def metrics_from_trades(name: str, asset: str, tf: str, trades: list[Trade],
                        df: pd.DataFrame) -> Metrics:
    if not trades:
        return Metrics(name, asset, tf, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       str(df.index[0]) if len(df) else "",
                       str(df.index[-1]) if len(df) else "", "")
    rets = np.array([t.ret_pct for t in trades])
    wins = rets[rets > 0]; losses = rets[rets <= 0]
    wr = len(wins) / len(rets)
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = -losses.sum() if len(losses) else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    eq = np.cumprod(1.0 + rets / 100.0)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    return Metrics(
        strategy=name, asset=asset, tf=tf,
        trades=len(rets), wins=int(len(wins)), losses=int(len(losses)),
        wr=round(wr * 100, 2),
        avg_win_pct=round(wins.mean(), 3) if len(wins) else 0.0,
        avg_loss_pct=round(losses.mean(), 3) if len(losses) else 0.0,
        profit_factor=round(pf, 3) if pf != float("inf") else 999.0,
        total_return_pct=round((eq[-1] - 1.0) * 100, 2),
        max_drawdown_pct=round(dd.min() * 100, 2),
        first_bar=str(df.index[0]),
        last_bar=str(df.index[-1]),
        exits_by_reason=",".join(f"{k}:{v}" for k, v in sorted(reasons.items())),
    )


def run_all(min_wr: float, min_trades: int, grail_dd: float,
            fee: float, slippage: float):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_wr_dir = OUT_DIR / "by_wr"
    if by_wr_dir.exists():
        for p in by_wr_dir.iterdir():
            if p.is_file():
                p.unlink()
    by_wr_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[Metrics] = []
    trades_index: dict[str, list[Trade]] = {}

    csvs = sorted(DATA_DIR.glob("*.csv"))
    if not csvs:
        print(f"No CSVs in {DATA_DIR}. Run fetch_data.py or tools/convert_cryptopredictions.py first.")
        return

    for csv_path in csvs:
        stem = csv_path.stem
        try:
            asset, tf = stem.rsplit("_", 1)
        except ValueError:
            print(f"Skipping {csv_path.name}: bad stem"); continue
        df = load_csv(csv_path)
        if len(df) < 250:
            print(f"Skipping {csv_path.name}: only {len(df)} bars"); continue
        for sname, cfg in STRATEGIES.items():
            try:
                ent, ex = cfg["fn"](df)
            except Exception as e:
                print(f"  {sname} on {asset} {tf}: error {e}"); continue
            trades = simulate(df, ent, ex, fee=fee, slippage=slippage,
                              sl_atr=cfg.get("sl_atr"),
                              tp_atr=cfg.get("tp_atr"),
                              trail_atr=cfg.get("trail_atr"),
                              timeout=cfg.get("timeout"))
            m = metrics_from_trades(sname, asset, tf, trades, df)
            all_metrics.append(m)
            trades_index[f"{sname}|{asset}|{tf}"] = trades
            print(f"  {sname:<22} {asset:<10} {tf:<4} "
                  f"trades={m.trades:<4} WR={m.wr:>6.2f}%  PF={m.profit_factor:<6} "
                  f"ret={m.total_return_pct:>8}%  DD={m.max_drawdown_pct:>7}%")

    # summary.csv (all combos)
    with (OUT_DIR / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(all_metrics[0]).keys()))
        w.writeheader()
        for m in all_metrics:
            w.writerow(asdict(m))

    # qualifying = WR>min_wr AND trades>min_trades
    qual = [m for m in all_metrics
            if m.trades > min_trades and m.wr > min_wr * 100]
    qual.sort(key=lambda x: (-x.wr, -x.profit_factor))

    losers = sum(1 for m in qual if m.total_return_pct <= 0)
    lines = [
        "# Backtest Summary — qualifying combos",
        "",
        f"- Filter: WR > {min_wr*100:.0f}% AND trades > {min_trades}",
        f"- Fees: {fee*100:.3f}%/side, slippage: {slippage*100:.3f}%/side",
        f"- Execution: next-bar open, long-only, one position at a time",
        f"- Qualifying combos: **{len(qual)}** / {len(all_metrics)}",
    ]
    if losers:
        lines.append(f"- ⚠ {losers} of {len(qual)} qualifying rows have negative total return.")
    lines += [
        "",
        "| # | Strategy | Asset | TF | Trades | WR % | PF | Total Ret % | Max DD % | Exits |",
        "|---|----------|-------|----|--------|------|-----|-------------|----------|-------|",
    ]
    for i, m in enumerate(qual, 1):
        lines.append(f"| {i} | `{m.strategy}` | {m.asset} | {m.tf} | {m.trades} | "
                     f"{m.wr} | {m.profit_factor} | {m.total_return_pct} | {m.max_drawdown_pct} | {m.exits_by_reason} |")
    if not qual:
        lines.append("\n_No combo met WR>{}% and trades>{}._".format(int(min_wr*100), min_trades))
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")

    # GRAILS: WR>min_wr AND return>0 AND maxDD > -grail_dd (less drawdown)
    grails = [m for m in qual if m.total_return_pct > 0
              and m.max_drawdown_pct > -grail_dd
              and m.profit_factor > 1.0]
    grails.sort(key=lambda x: (-x.wr, -x.profit_factor))
    glines = [
        "# 🏆 GRAILS — WR>{}% AND PnL>0 AND maxDD>-{}% AND PF>1".format(int(min_wr*100), grail_dd),
        "",
        f"- Found: **{len(grails)}** / {len(all_metrics)} total combos.",
        "- All numbers measured on real OHLCV committed to GitHub "
        "(alimohammadiamirhossein/CryptoPredictions).",
        "",
        "| # | Strategy | Asset | TF | Trades | WR % | PF | Return % | Max DD % | Exits |",
        "|---|----------|-------|----|--------|------|-----|----------|----------|-------|",
    ]
    for i, m in enumerate(grails, 1):
        glines.append(f"| {i} | `{m.strategy}` | {m.asset} | {m.tf} | {m.trades} | "
                      f"{m.wr} | {m.profit_factor} | {m.total_return_pct} | "
                      f"{m.max_drawdown_pct} | {m.exits_by_reason} |")
    if not grails:
        glines.append("\n_No strategy hit all three filters simultaneously. "
                      "Either WR<60, or PnL≤0, or maxDD exceeded threshold._")
    (OUT_DIR / "grails.md").write_text("\n".join(glines) + "\n")

    # per-combo files in by_wr/ (use qual list, grail rows highlighted)
    grail_ids = {(m.strategy, m.asset, m.tf) for m in grails}
    for i, m in enumerate(qual, 1):
        src = SOURCES.get(m.strategy, {})
        is_grail = (m.strategy, m.asset, m.tf) in grail_ids
        fname = f"{i:02d}_{m.strategy}_{m.asset}_{m.tf}_WR{m.wr:.1f}.md"
        trades = trades_index[f"{m.strategy}|{m.asset}|{m.tf}"]
        content = [
            f"# {'🏆 ' if is_grail else ''}{m.strategy} — {m.asset} {m.tf}",
            "",
            f"- **Win rate:** {m.wr}%",
            f"- **Trades:** {m.trades} (wins {m.wins}, losses {m.losses})",
            f"- **Profit factor:** {m.profit_factor}",
            f"- **Total return:** {m.total_return_pct}%",
            f"- **Max drawdown:** {m.max_drawdown_pct}%",
            f"- **Avg win:** {m.avg_win_pct}% | **Avg loss:** {m.avg_loss_pct}%",
            f"- **Exits by reason:** {m.exits_by_reason}",
            f"- **Period:** {m.first_bar} → {m.last_bar}",
            "",
            "## Strategy",
            f"- **Name:** {src.get('name','?')}",
            f"- **Author/origin:** {src.get('author','?')}",
            f"- **Logic:** {src.get('logic','?')}",
            f"- **Pine source:** `{src.get('pine_ref','?')}`",
            "",
            "## Trade log (first 20)",
            "| # | Entry time | Entry px | Exit time | Exit px | Bars | Ret % | Exit |",
            "|---|------------|----------|-----------|---------|------|-------|------|",
        ]
        for j, t in enumerate(trades[:20], 1):
            content.append(
                f"| {j} | {pd.to_datetime(t.entry_time, unit='s', utc=True)} | "
                f"{t.entry_price:.6g} | {pd.to_datetime(t.exit_time, unit='s', utc=True)} | "
                f"{t.exit_price:.6g} | {t.bars_held} | {t.ret_pct:.3f} | {t.exit_reason} |"
            )
        if len(trades) > 20:
            content.append(f"\n_...{len(trades)-20} more trades omitted_")
        (by_wr_dir / fname).write_text("\n".join(content) + "\n")

    (OUT_DIR / "summary.json").write_text(
        json.dumps({"qualifying": [asdict(m) for m in qual],
                    "grails":     [asdict(m) for m in grails]}, indent=2) + "\n"
    )
    print(f"\n{len(qual)} qualifying / {len(grails)} GRAILS. See {OUT_DIR}/grails.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-wr", type=float, default=0.60)
    ap.add_argument("--min-trades", type=int, default=5)
    ap.add_argument("--grail-dd", type=float, default=30.0,
                    help="Max acceptable drawdown magnitude (percent)")
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--slippage", type=float, default=0.0005)
    args = ap.parse_args()
    run_all(args.min_wr, args.min_trades, args.grail_dd, args.fee, args.slippage)


if __name__ == "__main__":
    main()
