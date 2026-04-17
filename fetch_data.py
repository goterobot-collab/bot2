#!/usr/bin/env python3
"""
Fetch full-history OHLCV for ADA/DOGE/INJ/ETH × 5m/15m/1h from Binance spot
public REST (no API key). Stores as data/{SYMBOL}_{TF}.csv.

Usage:
    python3 fetch_data.py                 # default: all pairs × all TFs
    python3 fetch_data.py --pairs ETHUSDT --tfs 1h
    python3 fetch_data.py --start 2021-01-01
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import requests

BASE = "https://api.binance.com/api/v3/klines"
DEFAULT_PAIRS = ["ADAUSDT", "DOGEUSDT", "INJUSDT", "ETHUSDT"]
DEFAULT_TFS = ["5m", "15m", "1h"]
TF_MS = {"5m": 5 * 60_000, "15m": 15 * 60_000, "1h": 60 * 60_000}
LIMIT = 1000  # Binance per-request max


def _ts_ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_klines(pair: str, tf: str, start_ms: int, end_ms: int) -> list[list]:
    """Returns list of klines: [open_time, o, h, l, c, v, close_time, qav, n_trades, tbbav, tbqav, _]"""
    out = []
    cur = start_ms
    step = TF_MS[tf] * LIMIT
    while cur < end_ms:
        params = {
            "symbol": pair,
            "interval": tf,
            "startTime": cur,
            "endTime": min(cur + step, end_ms),
            "limit": LIMIT,
        }
        for attempt in range(5):
            try:
                r = requests.get(BASE, params=params, timeout=15)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
                print(f"  retry {attempt+1}: {e}", file=sys.stderr)
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        last_open = batch[-1][0]
        if last_open + TF_MS[tf] >= end_ms or len(batch) < LIMIT:
            cur = last_open + TF_MS[tf]
            if len(batch) < LIMIT:
                break
        else:
            cur = last_open + TF_MS[tf]
        time.sleep(0.12)  # be polite
    return out


def save_csv(klines: list[list], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for k in klines:
            ts = int(k[0]) // 1000  # seconds UTC
            w.writerow([ts, k[1], k[2], k[3], k[4], k[5]])
    return len(klines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    ap.add_argument("--tfs", nargs="+", default=DEFAULT_TFS)
    ap.add_argument("--start", default="2020-01-01", help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: now)")
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    start_ms = _ts_ms(args.start)
    end_ms = _ts_ms(args.end) if args.end else int(time.time() * 1000)
    outdir = Path(args.outdir)

    for pair in args.pairs:
        for tf in args.tfs:
            dest = outdir / f"{pair}_{tf}.csv"
            print(f"[{pair} {tf}] fetching {args.start} → {args.end or 'now'} → {dest}")
            try:
                kl = fetch_klines(pair, tf, start_ms, end_ms)
            except Exception as e:
                print(f"  FAILED: {e}", file=sys.stderr)
                continue
            n = save_csv(kl, dest)
            print(f"  saved {n} bars")


if __name__ == "__main__":
    main()
