#!/usr/bin/env python3
"""Fetch Bitfinex 1m OHLCV from Zombie-3000/Bitfinex-historical-data and
resample to 5m + 1h for ETH, LTC, XRP, BCH. Saves as data/candles/<SYM>_<TF>.csv.gz
matching Mac's schema: ts (ms), open, high, low, close, volume.

Column order in source is: timestamp_ms, open, CLOSE, HIGH, LOW, volume
(close before high — NOT standard OHLC).

No header. Per-year merged.csv, 2013-2019 for LTC, 2016-2019 for ETH, 2017-2019 for XRP.
"""
from __future__ import annotations
import gzip
import io
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path("/home/user/bot2")
OUT = ROOT / "data" / "candles"
OUT.mkdir(exist_ok=True)

SYMBOLS = {
    "ETH": ("ETHUSD", range(2016, 2020)),
    "LTC": ("LTCUSD", range(2013, 2020)),
    "XRP": ("XRPUSD", range(2017, 2020)),
}


def fetch_year(pair: str, year: int) -> pd.DataFrame:
    url = f"https://raw.githubusercontent.com/Zombie-3000/Bitfinex-historical-data/master/{pair}/Candles_1m/{year}/merged.csv"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "fetch/1"}),
            timeout=60,
        ) as r:
            data = r.read()
    except Exception as e:
        print(f"  {pair}/{year}: fetch fail {e}")
        return pd.DataFrame()
    # source columns: ts_ms, open, close, high, low, volume (no header)
    df = pd.read_csv(
        io.BytesIO(data),
        header=None,
        names=["ts", "open", "close", "high", "low", "volume"],
    )
    # reorder to standard OHLC
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    return df


def main():
    for sym, (pair, years) in SYMBOLS.items():
        print(f"\n=== {sym} ({pair}) years {list(years)} ===")
        parts = []
        for y in years:
            df = fetch_year(pair, y)
            if not df.empty:
                parts.append(df)
                print(f"  {y}: {len(df):,} rows")
        if not parts:
            print(f"  SKIP {sym}: no data")
            continue
        full = pd.concat(parts, ignore_index=True)
        full = full.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
        print(f"  Combined: {len(full):,} 1m rows, range {full.ts.min()} -> {full.ts.max()}")
        full["dt"] = pd.to_datetime(full["ts"], unit="ms", utc=True)
        idx = full.set_index("dt").sort_index()
        # resample to 5m and 1h (matching existing candle schema)
        for tf, rule in [("5m", "5min"), ("1h", "1h")]:
            out = idx.resample(rule).agg({
                "ts": "first",
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()
            out["ts"] = out["ts"].astype("int64")
            path = OUT / f"{sym}_{tf}.csv.gz"
            out_df = out.reset_index(drop=True)[["ts", "open", "high", "low", "close", "volume"]]
            out_df.to_csv(path, index=False, compression="gzip")
            print(f"    wrote {path}: {len(out_df):,} bars")


if __name__ == "__main__":
    main()
