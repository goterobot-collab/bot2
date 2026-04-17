#!/usr/bin/env python3
"""
Backtest every strategy × asset × timeframe against CSVs in data/.

Output:
    results/summary.csv              — every combo, all metrics
    results/summary.md               — qualifying combos (WR>WR_MIN, trades>TRADE_MIN)
                                        ordered WR descending
    results/by_wr/NN_...md           — one file per qualifying combo

Usage:
    python3 backtest.py
    python3 backtest.py --min-wr 0.60 --min-trades 5
    python3 backtest.py --fee 0.001 --slippage 0.0005
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

DATA_DIR = Path("data")
OUT_DIR = Path("results")


@dataclass
class Trade:
    entry_time: int
    entry_price: float
    exit_time: int
    exit_price: float
    bars_held: int
    ret_pct: float  # net of fees+slippage


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


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def simulate(df: pd.DataFrame, entry: pd.Series, exit_: pd.Series,
             fee: float, slippage: float) -> list[Trade]:
    """Long-only, one position at a time, next-bar execution."""
    # shift signals so we act on next bar's open
    e = entry.shift(1).fillna(False).values
    x = exit_.shift(1).fillna(False).values
    op = df["open"].values
    ts = df.index.astype("int64") // 10**9
    n = len(df)
    trades: list[Trade] = []
    pos = False
    ep = 0.0
    et = 0
    ebar = 0
    for i in range(n):
        if not pos and e[i]:
            ep = op[i] * (1.0 + slippage)
            et = int(ts[i])
            ebar = i
            pos = True
        elif pos and (x[i] or i == n - 1):
            xp = op[i] * (1.0 - slippage)
            gross = xp / ep - 1.0
            net = gross - 2 * fee  # entry + exit fee
            trades.append(Trade(et, ep, int(ts[i]), xp, i - ebar, net * 100.0))
            pos = False
    return trades


def metrics_from_trades(name: str, asset: str, tf: str, trades: list[Trade],
                        df: pd.DataFrame) -> Metrics:
    if not trades:
        return Metrics(name, asset, tf, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       str(df.index[0]) if len(df) else "",
                       str(df.index[-1]) if len(df) else "")
    rets = np.array([t.ret_pct for t in trades])
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    wr = len(wins) / len(rets)
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = -losses.sum() if len(losses) else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    # equity curve (compounded, 1.0 = flat)
    eq = np.cumprod(1.0 + rets / 100.0)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
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
    )


def run_all(min_wr: float, min_trades: int, fee: float, slippage: float):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_wr_dir = OUT_DIR / "by_wr"
    # wipe prior per-combo files so stale runs don't pollute the folder
    if by_wr_dir.exists():
        for p in by_wr_dir.iterdir():
            if p.is_file():
                p.unlink()
    by_wr_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[Metrics] = []
    trades_index: dict[str, list[Trade]] = {}

    csvs = sorted(DATA_DIR.glob("*.csv"))
    if not csvs:
        print(f"No CSVs in {DATA_DIR}. Run fetch_data.py first.")
        return

    for csv_path in csvs:
        stem = csv_path.stem  # e.g. ETHUSDT_1h
        try:
            asset, tf = stem.rsplit("_", 1)
        except ValueError:
            print(f"Skipping {csv_path.name}: bad stem")
            continue
        df = load_csv(csv_path)
        if len(df) < 250:
            print(f"Skipping {csv_path.name}: only {len(df)} bars")
            continue
        for sname, fn in STRATEGIES.items():
            try:
                ent, ex = fn(df)
            except Exception as e:
                print(f"  {sname} on {asset} {tf}: error {e}")
                continue
            trades = simulate(df, ent, ex, fee=fee, slippage=slippage)
            m = metrics_from_trades(sname, asset, tf, trades, df)
            all_metrics.append(m)
            trades_index[f"{sname}|{asset}|{tf}"] = trades
            print(f"  {sname:<22} {asset:<10} {tf:<4} "
                  f"trades={m.trades:<4} WR={m.wr:>6.2f}%  PF={m.profit_factor}  "
                  f"ret={m.total_return_pct}%  maxDD={m.max_drawdown_pct}%")

    # write summary.csv (all combos)
    csv_out = OUT_DIR / "summary.csv"
    with csv_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(all_metrics[0]).keys()))
        w.writeheader()
        for m in all_metrics:
            w.writerow(asdict(m))

    # qualifying: WR>min_wr AND trades>min_trades AND total_return_pct>0
    qual = [m for m in all_metrics
            if m.trades > min_trades and m.wr > min_wr * 100 and m.total_return_pct > 0]
    qual.sort(key=lambda x: (-x.wr, -x.profit_factor))

    # summary.md
    md = OUT_DIR / "summary.md"
    lines = [
        "# Backtest Summary — qualifying combos",
        "",
        f"- Filter: WR > {min_wr*100:.0f}% AND trades > {min_trades} AND total return > 0",
        f"- Fees: {fee*100:.3f}%/side, slippage: {slippage*100:.3f}%/side",
        f"- Execution: next-bar open, long-only, one position at a time",
        f"- Qualifying combos: **{len(qual)}** / {len(all_metrics)}",
        "",
        "| # | Strategy | Asset | TF | Trades | WR % | PF | Total Ret % | Max DD % |",
        "|---|----------|-------|----|--------|------|-----|-------------|----------|",
    ]
    for i, m in enumerate(qual, 1):
        lines.append(f"| {i} | `{m.strategy}` | {m.asset} | {m.tf} | {m.trades} | "
                     f"{m.wr} | {m.profit_factor} | {m.total_return_pct} | {m.max_drawdown_pct} |")
    if not qual:
        lines.append("")
        lines.append("_No combo met the criteria._")
    md.write_text("\n".join(lines) + "\n")

    # per-combo files in by_wr/
    for i, m in enumerate(qual, 1):
        src = SOURCES.get(m.strategy, {})
        fname = f"{i:02d}_{m.strategy}_{m.asset}_{m.tf}_WR{m.wr:.1f}.md"
        path = by_wr_dir / fname
        trades = trades_index[f"{m.strategy}|{m.asset}|{m.tf}"]
        content = [
            f"# {m.strategy} — {m.asset} {m.tf}",
            "",
            f"- **Win rate:** {m.wr}%",
            f"- **Trades:** {m.trades} (wins {m.wins}, losses {m.losses})",
            f"- **Profit factor:** {m.profit_factor}",
            f"- **Total return:** {m.total_return_pct}%",
            f"- **Max drawdown:** {m.max_drawdown_pct}%",
            f"- **Avg win:** {m.avg_win_pct}% | **Avg loss:** {m.avg_loss_pct}%",
            f"- **Period:** {m.first_bar} → {m.last_bar}",
            "",
            "## Strategy",
            f"- **Name:** {src.get('name','?')}",
            f"- **Author/origin:** {src.get('author','?')}",
            f"- **Logic:** {src.get('logic','?')}",
            f"- **Pine source:** `{src.get('pine_ref','?')}`",
            "",
            "## Trade log (first 20)",
            "| # | Entry time | Entry px | Exit time | Exit px | Bars | Ret % |",
            "|---|------------|----------|-----------|---------|------|-------|",
        ]
        for j, t in enumerate(trades[:20], 1):
            content.append(
                f"| {j} | {pd.to_datetime(t.entry_time, unit='s', utc=True)} | "
                f"{t.entry_price:.6g} | {pd.to_datetime(t.exit_time, unit='s', utc=True)} | "
                f"{t.exit_price:.6g} | {t.bars_held} | {t.ret_pct:.3f} |"
            )
        if len(trades) > 20:
            content.append(f"\n_...{len(trades)-20} more trades omitted_")
        path.write_text("\n".join(content) + "\n")

    # JSON index for programmatic access
    (OUT_DIR / "summary.json").write_text(
        json.dumps([asdict(m) for m in qual], indent=2) + "\n"
    )
    print(f"\n{len(qual)} qualifying combos written to {OUT_DIR}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-wr", type=float, default=0.60,
                    help="Minimum win rate (fraction, default 0.60)")
    ap.add_argument("--min-trades", type=int, default=5,
                    help="Minimum trades across full history (default 5)")
    ap.add_argument("--fee", type=float, default=0.001,
                    help="Per-side taker fee (default 0.001 = 0.1%)")
    ap.add_argument("--slippage", type=float, default=0.0005,
                    help="Per-side slippage (default 0.0005 = 0.05%)")
    args = ap.parse_args()
    run_all(args.min_wr, args.min_trades, args.fee, args.slippage)


if __name__ == "__main__":
    main()
