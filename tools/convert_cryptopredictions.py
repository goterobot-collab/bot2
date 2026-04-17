#!/usr/bin/env python3
"""
Convert CSVs from alimohammadiamirhossein/CryptoPredictions (GitHub) into the
schema this repo's backtest.py expects: timestamp,open,high,low,close,volume
(timestamp = unix seconds UTC).

Why this exists: the execution sandbox for this project cannot reach exchange
APIs (Binance/Bybit/etc are all blocked at 403). codeload.github.com and
raw.githubusercontent.com ARE reachable, so we use a research repo that
committed real hourly OHLCV CSVs to GitHub as the data source.

No INJ data exists in that dataset (listing is too recent). 5m/15m are not
available either — only 1h and 1d. For 5m/15m/INJ you need fetch_data.py
on a host with Binance reachability.

Usage:
    python3 tools/convert_cryptopredictions.py               # default pairs
    python3 tools/convert_cryptopredictions.py ADAUSD ETHUSD
"""
from __future__ import annotations
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

ARCHIVE_URL = "https://codeload.github.com/alimohammadiamirhossein/CryptoPredictions/tar.gz/refs/heads/main"
DEFAULT_PAIRS = ["ADAUSD", "DOGEUSD", "ETHUSD"]


def fetch_archive() -> tarfile.TarFile:
    print(f"downloading {ARCHIVE_URL} ...", flush=True)
    req = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as r:
        buf = io.BytesIO(r.read())
    return tarfile.open(fileobj=buf, mode="r:gz")


def extract_pair(tar: tarfile.TarFile, pair: str) -> pd.DataFrame:
    name = f"CryptoPredictions-main/data/{pair}-1h-data.csv"
    mem = tar.getmember(name)
    f = tar.extractfile(mem)
    if f is None:
        raise FileNotFoundError(name)
    df = pd.read_csv(f)
    df = df.dropna(subset=["open", "high", "low", "close"])
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().astype("int64")
    return df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp").drop_duplicates("timestamp")


def main(pairs: list[str]):
    out = Path("data")
    out.mkdir(exist_ok=True)
    tar = fetch_archive()
    try:
        for pair in pairs:
            try:
                df = extract_pair(tar, pair)
            except KeyError:
                print(f"{pair}: not in dataset (available set is limited)")
                continue
            dest = out / f"{pair}_1h.csv"
            df.to_csv(dest, index=False)
            first = pd.to_datetime(df.timestamp.iloc[0], unit="s", utc=True)
            last  = pd.to_datetime(df.timestamp.iloc[-1], unit="s", utc=True)
            print(f"{dest}: {len(df)} bars  {first} → {last}")
    finally:
        tar.close()


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_PAIRS)
