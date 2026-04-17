#!/usr/bin/env python3
"""
Fetch 5m / 15m / 1h OHLCV from Binance public REST endpoint (no API key).

Runs on any host that can reach https://api.binance.com (the Anthropic
sandbox gets 403; a Hetzner VPS works fine).

Usage:
    python3 tools/fetch_binance_tfs.py                        # default pairs
    python3 tools/fetch_binance_tfs.py --pairs BTCUSDT ETHUSDT --tfs 5m 15m 1h
    python3 tools/fetch_binance_tfs.py --years 3              # 3y of history
    python3 tools/fetch_binance_tfs.py --update               # append last 7d only

Writes to data/<PAIR>_<tf>.csv in the same schema as
tools/convert_cryptopredictions.py:

    timestamp,open,high,low,close,volume
    <unix_seconds_UTC>,...

Appending is idempotent - duplicate timestamps are dropped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BINANCE = "https://api.binance.com/api/v3/klines"
LIMIT = 1000  # max bars per request

# Binance pair -> our asset stem. Binance uses USDT, we store as USD.
PAIR_MAP = {
    "BTCUSDT": "BTCUSD",
    "ETHUSDT": "ETHUSD",
    "ADAUSDT": "ADAUSD",
    "DOGEUSDT": "DOGEUSD",
    "SOLUSDT": "SOLUSD",
    "BNBUSDT": "BNBUSD",
    "XRPUSDT": "XRPUSD",
    "LINKUSDT": "LINKUSD",
    "AVAXUSDT": "AVAXUSD",
    "MATICUSDT": "MATICUSD",
    "LTCUSDT": "LTCUSD",
}

# Minutes per TF for paging math.
TF_MIN = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


def _http_get(url: str) -> list:
    req = urllib.request.Request(url, headers={"User-Agent": "grail-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def fetch_pair_tf(pair: str, tf: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch OHLCV for [start_ms, end_ms] paging through 1000-bar blocks."""
    rows = []
    cur = start_ms
    while cur < end_ms:
        params = urllib.parse.urlencode({
            "symbol": pair, "interval": tf,
            "startTime": cur, "endTime": end_ms, "limit": LIMIT,
        })
        url = f"{BINANCE}?{params}"
        try:
            data = _http_get(url)
        except Exception as e:
            print(f"    [err] {pair} {tf} @ {cur}: {e}", file=sys.stderr)
            time.sleep(2.0)
            continue
        if not data:
            break
        rows.extend(data)
        # next cursor = open-time of last bar + one tf step
        last_open = data[-1][0]
        cur = last_open + TF_MIN[tf] * 60_000
        # Binance rate limit: keep under 2400 weight/minute
        time.sleep(0.15)
        if len(data) < LIMIT:
            # reached tail
            break
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=[
        "ot", "open", "high", "low", "close", "volume",
        "ct", "qv", "ntrades", "tbbav", "tbqav", "ignore",
    ])
    df["timestamp"] = (df["ot"] // 1000).astype(int)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype({
        "open": float, "high": float, "low": float, "close": float, "volume": float,
    })
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def merge_write(out_path: Path, new_df: pd.DataFrame):
    if not new_df.empty and out_path.exists():
        old = pd.read_csv(out_path)
        merged = pd.concat([old, new_df]).drop_duplicates("timestamp")\
            .sort_values("timestamp").reset_index(drop=True)
    else:
        merged = new_df
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    return len(merged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+",
                    default=["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOGEUSDT"])
    ap.add_argument("--tfs", nargs="+", default=["5m", "15m", "1h"])
    ap.add_argument("--years", type=float, default=2.0,
                    help="How many years of history to fetch (default 2y)")
    ap.add_argument("--update", action="store_true",
                    help="Only fetch last 7 days (append to existing CSV)")
    ap.add_argument("--out", default="data", help="Output directory")
    args = ap.parse_args()

    now_ms = int(time.time() * 1000)
    if args.update:
        start_ms = now_ms - 7 * 24 * 60 * 60 * 1000
    else:
        start_ms = now_ms - int(args.years * 365.25 * 24 * 60 * 60 * 1000)

    out_dir = Path(args.out)
    for pair in args.pairs:
        asset = PAIR_MAP.get(pair, pair.replace("USDT", "USD"))
        for tf in args.tfs:
            if tf not in TF_MIN:
                print(f"[skip] unknown tf {tf}")
                continue
            print(f"[{asset} {tf}] fetch since {time.strftime('%Y-%m-%d', time.gmtime(start_ms/1000))}")
            df = fetch_pair_tf(pair, tf, start_ms, now_ms)
            out = out_dir / f"{asset}_{tf}.csv"
            n = merge_write(out, df)
            print(f"    wrote {out} -> {n} rows (fetched {len(df)} new)")


if __name__ == "__main__":
    main()
