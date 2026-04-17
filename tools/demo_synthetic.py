#!/usr/bin/env python3
"""
Demo: run the full backtest pipeline on SYNTHETIC random-walk data, one CSV
per "asset" (ADA_SYNTH / DOGE_SYNTH / INJ_SYNTH / ETH_SYNTH). This proves the
pipeline compiles and produces a sorted results folder — but the WR numbers
are MEANINGLESS. Real numbers require running fetch_data.py on a host that
can reach Binance, then backtest.py against the real CSVs.

Usage:
    python3 tools/demo_synthetic.py
    python3 backtest.py --min-wr 0.0 --min-trades 1
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("data")
OUT.mkdir(parents=True, exist_ok=True)
PAIRS = ["ADA_SYNTH", "DOGE_SYNTH", "INJ_SYNTH", "ETH_SYNTH"]
TFS   = {"5m": "5min", "15m": "15min", "1h": "1h"}
SEED_BASE = 42

def synth(n: int, seed: int, drift: float, vol: float) -> np.ndarray:
    r = np.random.default_rng(seed)
    rets = r.normal(drift, vol, n)
    return np.exp(np.cumsum(rets)) * 100.0

def make(pair: str, tf: str, n: int, seed: int):
    idx = pd.date_range("2024-04-17", periods=n, freq=TFS[tf], tz="UTC")
    close = synth(n, seed, drift=0.00008, vol=0.008)
    r = np.random.default_rng(seed + 1)
    df = pd.DataFrame({
        "timestamp": (idx.astype("int64") // 10**9).astype(int),
        "open":   close * (1 + r.normal(0, 0.001, n)),
        "high":   close * (1 + np.abs(r.normal(0, 0.002, n))),
        "low":    close * (1 - np.abs(r.normal(0, 0.002, n))),
        "close":  close,
        "volume": r.integers(100, 10000, n).astype(float),
    })
    path = OUT / f"{pair}_{tf}.csv"
    df.to_csv(path, index=False)
    return path, len(df)

if __name__ == "__main__":
    for i, pair in enumerate(PAIRS):
        for j, tf in enumerate(TFS):
            p, n = make(pair, tf, n=3000, seed=SEED_BASE + i * 10 + j)
            print(f"synthetic  {p}  ({n} bars)")
    print("\nNOW RUN:  python3 backtest.py --min-wr 0.0 --min-trades 1")
    print("(numbers are fake — just to show output format).")
