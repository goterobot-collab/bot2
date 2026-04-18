#!/usr/bin/env python3
"""
Canary runner for Paso 1 of the SANDBOX_CHECKLIST.

Reproduces 3 known grails from already_tested_grails.json to verify the
sandbox pipeline (fees, candles, backtest engine) matches Mac/Hetzner.

Gate: 3/3 canaries within +/-3pp of expected WR => pipeline OK.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mac_optuna"))
sys.path.insert(0, str(ROOT / "mac_optuna" / "scripts"))

# Canary definitions (from SANDBOX_HANDOFF.md + already_tested_grails.json)
CANARIES = [
    {
        "id": "C1",
        "strategy": "B5_MA_Envelope_3",
        "symbol_v8": "SFP/USDT:USDT",
        "candle_sym": "SFP",
        "tf": "1d",
        "source_tf": "1h",   # resample from 1h (no native 1d in repo)
        "params": {"ema_period": 50, "envelope_pct": 0.03389162051698195},
        "sl_pct": 0.40,
        "leverage": 1,
        "expected_wr": 85.2,
        "expected_trades": 210,
    },
    {
        "id": "C2",
        "strategy": "VWAP_Double",
        "symbol_v8": "AGT/USDT:USDT",
        "candle_sym": "AGT",
        "tf": "1h",
        "source_tf": "1h",
        "params": {"dist_pct": 0.0757608409089568, "session_len": 99},
        "sl_pct": 0.40,
        "leverage": 1,
        "expected_wr": 82.2,
        "expected_trades": 253,
    },
    {
        "id": "C3",
        "strategy": "TV_Daily_Close_Signal",
        "symbol_v8": "SWARMS/USDT:USDT",
        "candle_sym": "SWARMS",
        "tf": "15m",
        "source_tf": "5m",
        "params": {"daily_ema": 10, "ema_trend": 193, "rsi_len": 17, "rsi_os": 42},
        "sl_pct": 0.40,
        "leverage": 1,
        "expected_wr": 79.3,
        "expected_trades": 208,
    },
]


def load_candles(sym: str, src_tf: str, target_tf: str) -> pd.DataFrame:
    """Load gzipped CSV, resample if needed, return df with DatetimeIndex.

    Keeps the `ts` column since some strategies (e.g. batch704 TV_*) need it.
    """
    path = ROOT / "data" / "candles" / f"{sym}_{src_tf}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    try:
        df = pd.read_csv(path, compression="gzip")
    except (UnicodeDecodeError, pd.errors.ParserError):
        # Some candle files have a trailing corrupt row with binary bytes.
        # Read raw, drop bad lines.
        import gzip, io
        with gzip.open(path, "rb") as f:
            raw = f.read()
        clean = raw.decode("utf-8", errors="ignore")
        df = pd.read_csv(io.StringIO(clean), on_bad_lines="skip")
    df = df.dropna(subset=["ts", "open", "high", "low", "close"])
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("dt").sort_index()
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    df["ts"] = df["ts"].astype("int64")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    if src_tf == target_tf:
        return df
    rule = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}[target_tf]
    r = df.resample(rule).agg({
        "ts": "first", "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    r["ts"] = r["ts"].astype("int64")
    return r


def get_strategy_fn(name: str):
    """Return the `gen` function for a canary strategy.

    The Mac codebase has strategy implementations split between:
      - mac_optuna/strategy_factory.py (VWAP_Double, ...)
      - mac_optuna/grail_dual_validator.py (B5_MA_Envelope_3, ...)
    C3 (TV_Daily_Close_Signal) is NOT in the repo - return None so caller
    reports as CANARY_MISSING_IMPL.
    """
    if name == "B5_MA_Envelope_3":
        # Canonical per Mac Claude + empirical match (C1 SFP 1d WR=85.2%, trades=210,
        # sl_hits=12 EXACT):
        #   - EMA (span), NOT SMA
        #   - THRESHOLD signal (not crossover): long every bar below lower band,
        #     short every bar above upper band
        def gen(df, ema_period=20, envelope_pct=0.03):
            ma = df["close"].ewm(span=ema_period, adjust=False).mean()
            upper = ma * (1 + envelope_pct)
            lower = ma * (1 - envelope_pct)
            sig = pd.Series(0, index=df.index)
            sig[df["close"] < lower] = 1
            sig[df["close"] > upper] = -1
            return sig
        return gen
    if name == "VWAP_Double":
        from strategy_factory import gen_vwap_double
        return gen_vwap_double
    if name == "TV_Daily_Close_Signal":
        sys.path.insert(0, str(ROOT / "strategies_v7"))
        from strategies_tv2_batch704 import gen_TV_Daily_Close_Signal
        return gen_TV_Daily_Close_Signal
    return None


def backtest_signal_exit(df: pd.DataFrame, gen_fn, params: dict,
                          sl_pct: float = None):
    """Port of mac_optuna/scripts/optuna_fullhistory.py::backtest_signal_exit.

    COST = 0.0015 (0.15% per side / 0.30% round-trip).
    Entry at next-bar open inflated by (1+COST) long, deflated long on exit.
    Signal=-1 while long (or signal=0) triggers exit; if signal=-1 also flips
    to short immediately on same bar. Optional SL: if sl_pct is set, MAE
    reaching sl_pct closes the trade at entry*(1+/-sl_pct).
    """
    COST = 0.0015
    try:
        signals = gen_fn(df, **params)
    except Exception as e:
        return {"error": f"gen_fn raised: {e}", "wr": None, "trades": []}
    if signals is None or len(signals) == 0:
        return {"error": "no signals", "wr": None, "trades": []}
    # batches 3566-3569 return a DataFrame with a 'signal' column; extract it
    if isinstance(signals, pd.DataFrame):
        if "signal" in signals.columns:
            signals = signals["signal"]
        elif "direction" in signals.columns:
            signals = signals["direction"]
        else:
            return {"error": "DataFrame lacks signal/direction column", "wr": None, "trades": []}
    sig = signals.values if hasattr(signals, "values") else np.array(signals)
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    n = min(len(sig), len(opens))
    trades = []
    pos = 0
    ep = 0.0
    ei = 0
    sl_hits = 0
    sig_exits = 0
    for i in range(1, n):
        s = int(sig[i - 1])
        p = opens[i]
        if pos == 0:
            if s == 1:
                ep = p * (1 + COST); ei = i; pos = 1
            elif s == -1:
                ep = p * (1 - COST); ei = i; pos = -1
        elif pos == 1:
            # Check SL first (intrabar)
            if sl_pct is not None:
                mae = (ep - lows[i]) / ep
                if mae >= sl_pct:
                    xp = ep * (1 - sl_pct)
                    pnl = (xp - ep) / ep - COST
                    trades.append({"pnl": pnl, "dur_bars": i - ei, "win": pnl > 0,
                                   "dir": "LONG", "exit": "SL"})
                    sl_hits += 1
                    pos = 0
                    continue
            if s == -1 or s == 0:
                xp = p * (1 - COST)
                pnl = (xp - ep) / ep
                trades.append({"pnl": pnl, "dur_bars": i - ei, "win": pnl > 0,
                               "dir": "LONG", "exit": "SIG"})
                sig_exits += 1
                pos = 0
                if s == -1:
                    ep = p * (1 - COST); ei = i; pos = -1
        elif pos == -1:
            if sl_pct is not None:
                mae = (highs[i] - ep) / ep
                if mae >= sl_pct:
                    xp = ep * (1 + sl_pct)
                    pnl = (ep - xp) / ep - COST
                    trades.append({"pnl": pnl, "dur_bars": i - ei, "win": pnl > 0,
                                   "dir": "SHORT", "exit": "SL"})
                    sl_hits += 1
                    pos = 0
                    continue
            if s == 1 or s == 0:
                xp = p * (1 + COST)
                pnl = (ep - xp) / ep
                trades.append({"pnl": pnl, "dur_bars": i - ei, "win": pnl > 0,
                               "dir": "SHORT", "exit": "SIG"})
                sig_exits += 1
                pos = 0
                if s == 1:
                    ep = p * (1 + COST); ei = i; pos = 1
    if not trades:
        return {"wr": None, "trades": 0, "total_pnl_pct": 0, "error": "no trades"}
    wins = sum(1 for t in trades if t["win"])
    wr = 100.0 * wins / len(trades)
    pnls = [t["pnl"] for t in trades]
    total_pnl = sum(pnls) * 100
    win_sum = sum(p for p in pnls if p > 0)
    loss_sum = -sum(p for p in pnls if p <= 0)
    pf = (win_sum / loss_sum) if loss_sum > 0 else (999.0 if win_sum > 0 else 0.0)
    avg_win = (win_sum / wins * 100) if wins > 0 else 0.0
    avg_loss = (-loss_sum / (len(trades) - wins) * 100) if (len(trades) - wins) > 0 else 0.0
    return {
        "wr": round(wr, 1),
        "trades": len(trades),
        "total_pnl_pct": round(total_pnl, 2),
        "wins": wins,
        "losses": len(trades) - wins,
        "sl_hits": sl_hits,
        "sig_exits": sig_exits,
        "pf": round(pf, 3),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
    }


def run_forensic(df: pd.DataFrame, gen_fn, params: dict, sl_pct: float = 0.40,
                 timeframe: str = "1h", commission: float = 0.001,
                 slippage: float = 0.001):
    """Lightweight forensic backtest matching mac_optuna logic (no DB).

    Replicates forensic_backtest(gen_fn, params) with:
    - Entry at next-bar open
    - Exit on signal flip / SL hit / TP cap per TF
    - Fees: commission + slippage per side
    - LONG + SHORT
    - Sizing: fixed $10 per trade (matches Mac forensic)
    """
    TP_CAP = {"5m": 0.08, "15m": 0.10, "1h": 0.12, "4h": 0.18, "1d": 0.30}
    tp_cap = TP_CAP.get(timeframe, 1.0)
    cost_per_side = commission + slippage
    FUNDING_RATE_PER_8H = 0.0001
    bar_hours = {"5m": 5/60, "15m": 0.25, "1h": 1, "4h": 4, "1d": 24}.get(timeframe, 1)

    try:
        signals = gen_fn(df, **params)
    except Exception as e:
        return {"error": f"gen_fn raised: {e}", "trades": [], "wr": None}
    if signals is None or len(signals) == 0:
        return {"error": "no signals", "trades": [], "wr": None}
    sig = signals.values if hasattr(signals, "values") else np.array(signals)
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values

    trades = []
    in_trade = False
    direction = None
    entry_price = 0.0
    entry_bar = 0

    for i in range(2, len(sig)):
        prev_sig = sig[i - 1]
        if not in_trade:
            if prev_sig == 1 or prev_sig == -1:
                direction = "LONG" if prev_sig == 1 else "SHORT"
                entry_price = opens[i]
                if entry_price <= 0:
                    continue
                entry_bar = i
                in_trade = True
        else:
            bar_low = lows[i]
            bar_high = highs[i]
            if direction == "LONG":
                mae = (entry_price - bar_low) / entry_price
                mfe = (bar_high - entry_price) / entry_price
                sl_hit = mae >= sl_pct
                tp_hit = mfe >= tp_cap
            else:
                mae = (bar_high - entry_price) / entry_price
                mfe = (entry_price - bar_low) / entry_price
                sl_hit = mae >= sl_pct
                tp_hit = mfe >= tp_cap

            exit_price = None
            exit_type = None
            if sl_hit:
                if direction == "LONG":
                    exit_price = entry_price * (1 - sl_pct)
                else:
                    exit_price = entry_price * (1 + sl_pct)
                exit_type = "SL"
            elif tp_hit:
                if direction == "LONG":
                    exit_price = entry_price * (1 + tp_cap)
                else:
                    exit_price = entry_price * (1 - tp_cap)
                exit_type = "TP_CAP"
            else:
                # SIGNAL exit: LONG exits when prev_sig in (-1, 0); SHORT exits when prev_sig in (1, 0)
                # (matches mac_optuna/scripts/forensic_backtest.py lines 458-468)
                signal_exit = False
                if direction == "LONG" and prev_sig in (-1, 0):
                    signal_exit = True
                elif direction == "SHORT" and prev_sig in (1, 0):
                    signal_exit = True
                if signal_exit:
                    exit_bar = min(i + 1, len(opens) - 1)
                    exit_price = opens[exit_bar]
                    exit_type = "SIGNAL"
            if exit_price is not None:
                if direction == "LONG":
                    gross_pct = (exit_price - entry_price) / entry_price
                else:
                    gross_pct = (entry_price - exit_price) / entry_price
                fee_pct = 2 * cost_per_side
                funding_pct = FUNDING_RATE_PER_8H * ((i - entry_bar) * bar_hours / 8)
                net_pct = gross_pct - fee_pct - funding_pct
                trades.append({
                    "dir": direction, "pnl_pct": net_pct,
                    "duration": i - entry_bar, "exit_type": exit_type,
                })
                in_trade = False

    if not trades:
        return {"error": "no trades", "wr": None, "trades": []}
    wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    wr = 100.0 * wins / len(trades)
    total_pnl = sum(t["pnl_pct"] * 100 for t in trades)  # percentage points
    return {
        "wr": round(wr, 1),
        "trades": len(trades),
        "total_pnl_pct": round(total_pnl, 2),
        "wins": wins,
        "losses": len(trades) - wins,
        "trade_list": trades,
    }


def main():
    ROOT_OUT = ROOT / "results"
    ROOT_OUT.mkdir(exist_ok=True)
    report_path = ROOT_OUT / "sandbox_canary_report.md"

    results = []
    missing = []
    for c in CANARIES:
        print(f"\n=== {c['id']} {c['strategy']} {c['candle_sym']} {c['tf']} ===")
        gen_fn = get_strategy_fn(c["strategy"])
        if gen_fn is None:
            print(f"  IMPL_MISSING: {c['strategy']} not in mac_optuna/ repo")
            missing.append(c["id"])
            results.append({**c, "wr_observed": None, "gap_pp": None,
                            "status": "IMPL_MISSING"})
            continue
        try:
            df = load_candles(c["candle_sym"], c["source_tf"], c["tf"])
        except Exception as e:
            print(f"  DATA_MISSING: {e}")
            results.append({**c, "wr_observed": None, "gap_pp": None,
                            "status": f"DATA_MISSING: {e}"})
            continue
        print(f"  candles: {len(df):,} rows  range={df.index.min()} -> {df.index.max()}")
        # full_wr source: backtest_signal_exit with SL from grail
        r = backtest_signal_exit(df, gen_fn, c["params"], sl_pct=c["sl_pct"])
        if r.get("wr") is None:
            print(f"  RUN_FAILED: {r.get('error')}")
            results.append({**c, "wr_observed": None, "gap_pp": None,
                            "status": f"RUN_FAILED: {r.get('error')}"})
            continue
        gap = r["wr"] - c["expected_wr"]
        status = "PASS" if abs(gap) <= 3.0 else "FAIL"
        print(f"  observed WR={r['wr']}% (expected {c['expected_wr']}% -> gap {gap:+.1f}pp) "
              f"trades={r['trades']} (expected {c['expected_trades']}) status={status}")
        results.append({
            **c, "wr_observed": r["wr"], "trades_observed": r["trades"],
            "total_pnl_pct_observed": r["total_pnl_pct"],
            "gap_pp": round(gap, 1), "status": status,
        })

    # write md report
    lines = [
        "# Sandbox Canary Report (Paso 1)",
        "",
        f"Generated: {pd.Timestamp.utcnow().isoformat()}",
        "",
        "| ID | Strategy | Sym | TF | Expected WR | Observed WR | Gap | Status |",
        "|----|----------|-----|----|-------------|-------------|-----|--------|",
    ]
    for r in results:
        wr_s = f"{r['wr_observed']}%" if r['wr_observed'] is not None else "N/A"
        gap_s = f"{r['gap_pp']:+.1f}pp" if r['gap_pp'] is not None else "N/A"
        lines.append(
            f"| {r['id']} | {r['strategy']} | {r['candle_sym']} | {r['tf']} | "
            f"{r['expected_wr']:.1f}% | {wr_s} | {gap_s} | {r['status']} |"
        )
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    lines.extend(["", f"**Pass**: {n_pass}/{len(results)}  (gate: 3/3)", ""])
    if missing:
        lines.append(f"**Missing impl**: {', '.join(missing)} — not in `mac_optuna/` or `strategies_v7/`")

    report_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {report_path}")
    print(f"Gate: {n_pass}/{len(results)} canaries pass")
    return results, missing


if __name__ == "__main__":
    main()
