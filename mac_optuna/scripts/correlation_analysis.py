#!/usr/bin/env python3
"""
Correlation Analysis of Grail Assets
Analyzes which grail assets move together (correlated) vs independently.
Uses 1h candles resampled to daily closes since most crypto assets lack 1d data.
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

# ── Config ──
GRAILS_PATH = Path("/Users/sabrina/CLAUDE CODE/Estrategias/data/combined_grails.json")
DB_PATH = Path("/Users/sabrina/Code/activos binace /activos_binance.db")
OUTPUT_PATH = Path("/Users/sabrina/CLAUDE CODE/Estrategias/data/correlation_groups.json")
TOP_N = 50
CORR_THRESHOLD = 0.7
MAX_CAPITAL_PCT = 20

# ── 1. Load grails and count per asset ──
print("=" * 70)
print("CORRELATION ANALYSIS OF GRAIL ASSETS")
print("=" * 70)

with open(GRAILS_PATH) as f:
    grails = json.load(f)

print(f"\nTotal grails: {len(grails)}")

# Count grails per symbol
symbol_counts = Counter(g["symbol"] for g in grails)
top_symbols_raw = [s for s, _ in symbol_counts.most_common(TOP_N)]

print(f"Unique symbols with grails: {len(symbol_counts)}")
print(f"\nTop {TOP_N} symbols by grail count:")
for i, (sym, cnt) in enumerate(symbol_counts.most_common(TOP_N)):
    print(f"  {i+1:3d}. {sym:<30s} {cnt:4d} grails")

# ── 2. Check which symbols exist in DB with 1h data ──
conn = sqlite3.connect(str(DB_PATH))

db_symbols_1h = set(
    pd.read_sql("SELECT DISTINCT symbol FROM candles WHERE timeframe='1h'", conn)["symbol"].tolist()
)
print(f"\nDB has {len(db_symbols_1h)} symbols with 1h data")

# Filter to symbols that exist in DB
top_symbols = [s for s in top_symbols_raw if s in db_symbols_1h]
missing = [s for s in top_symbols_raw if s not in db_symbols_1h]

print(f"Matched {len(top_symbols)}/{len(top_symbols_raw)} top grail symbols in DB (1h)")
if missing:
    print(f"Missing from DB: {missing}")

# ── 3. Load 1h closes and resample to daily ──
print("\nLoading 1h candles for top symbols (this may take a moment)...")

placeholders = ",".join(["?"] * len(top_symbols))
query = f"""
SELECT symbol, ts, close
FROM candles
WHERE timeframe = '1h' AND symbol IN ({placeholders})
ORDER BY symbol, ts
"""

df = pd.read_sql(query, conn, params=top_symbols)
conn.close()

print(f"Loaded {len(df):,} hourly candles for {df['symbol'].nunique()} symbols")

# Convert ts (ms) to datetime and resample to daily closes
df["date"] = pd.to_datetime(df["ts"], unit="ms")
df = df.set_index("date")

# Get daily close = last hourly close of each day
daily_closes = df.groupby([df.index.date, "symbol"])["close"].last().reset_index()
daily_closes.columns = ["date", "symbol", "close"]
daily_closes["date"] = pd.to_datetime(daily_closes["date"])

pivot = daily_closes.pivot_table(index="date", columns="symbol", values="close")

# Need at least 90 days of data
min_days = 90
valid_cols = pivot.columns[pivot.count() >= min_days]
pivot = pivot[valid_cols]
print(f"Symbols with >= {min_days} days of data: {len(valid_cols)}")

# ── 4. Calculate correlation of daily returns ──
print("Calculating daily returns and correlation matrix...")

returns = pivot.pct_change().dropna(how="all")
# Drop symbols with too many NaNs (need at least 50% of days)
returns = returns.dropna(axis=1, thresh=int(len(returns) * 0.5))

corr_matrix = returns.corr()
n = corr_matrix.shape[0]
print(f"Correlation matrix: {n} x {n}")

# ── 5. Group assets with correlation > threshold ──
print(f"\nGrouping assets with correlation > {CORR_THRESHOLD}...")

symbols_in_matrix = list(corr_matrix.columns)
visited = set()
groups = []

for i, sym_i in enumerate(symbols_in_matrix):
    if sym_i in visited:
        continue
    # Find all symbols correlated with sym_i
    group = [sym_i]
    visited.add(sym_i)
    for j, sym_j in enumerate(symbols_in_matrix):
        if sym_j in visited:
            continue
        if corr_matrix.loc[sym_i, sym_j] > CORR_THRESHOLD:
            group.append(sym_j)
            visited.add(sym_j)
    if len(group) > 1:
        # Calculate average pairwise correlation within group
        group_corrs = []
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                group_corrs.append(corr_matrix.loc[group[a], group[b]])
        avg_corr = float(np.mean(group_corrs))

        grail_count = sum(symbol_counts.get(g, 0) for g in group)

        groups.append({
            "id": len(groups) + 1,
            "assets": group,
            "num_assets": len(group),
            "total_grails": grail_count,
            "avg_correlation": round(avg_corr, 3),
            "max_capital_pct": MAX_CAPITAL_PCT
        })

# Uncorrelated assets (not in any group)
grouped = set()
for g in groups:
    grouped.update(g["assets"])
uncorrelated = [s for s in symbols_in_matrix if s not in grouped]

# Sort groups by size descending
groups.sort(key=lambda g: g["num_assets"], reverse=True)
for i, g in enumerate(groups):
    g["id"] = i + 1

# ── 6. Build top-20 matrix for output ──
top20_syms = [s for s, _ in symbol_counts.most_common(20) if s in corr_matrix.columns][:20]

if top20_syms:
    matrix_top20 = corr_matrix.loc[top20_syms, top20_syms].values.tolist()
    matrix_top20 = [[round(v, 3) if not np.isnan(v) else None for v in row] for row in matrix_top20]
else:
    matrix_top20 = []

# ── 7. Save results ──
output = {
    "metadata": {
        "total_grails": len(grails),
        "unique_symbols_with_grails": len(symbol_counts),
        "analyzed_symbols": len(symbols_in_matrix),
        "correlation_threshold": CORR_THRESHOLD,
        "max_capital_per_group_pct": MAX_CAPITAL_PCT,
        "data_source": "1h candles resampled to daily closes"
    },
    "groups": groups,
    "uncorrelated_assets": uncorrelated,
    "matrix_top20_labels": top20_syms,
    "matrix_top20": matrix_top20,
    "capital_rules": [
        f"No more than {MAX_CAPITAL_PCT}% of capital in any single correlated group",
        "Uncorrelated assets can each receive up to 10% of capital independently",
        "Diversify across at least 3 different correlation groups",
        "If a group has > 5 assets, pick the top 2-3 by grail quality"
    ]
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {OUTPUT_PATH}")

# ── 8. Summary ──
print("\n" + "=" * 70)
print("SUMMARY: CORRELATED GROUPS")
print("=" * 70)

for g in groups:
    print(f"\n--- Group {g['id']} ({g['num_assets']} assets, avg corr: {g['avg_correlation']:.3f}) ---")
    print(f"    Max capital allocation: {g['max_capital_pct']}%")
    print(f"    Total grails in group: {g['total_grails']}")
    for a in g["assets"]:
        cnt = symbol_counts.get(a, 0)
        print(f"      {a:<30s} ({cnt} grails)")

print(f"\n{'=' * 70}")
print(f"INDEPENDENT ASSETS (correlation <= {CORR_THRESHOLD} with all others)")
print(f"{'=' * 70}")
print(f"Total: {len(uncorrelated)} assets")
for a in uncorrelated:
    cnt = symbol_counts.get(a, 0)
    print(f"  {a:<30s} ({cnt} grails)")

print(f"\n{'=' * 70}")
print("CAPITAL ALLOCATION RULES")
print(f"{'=' * 70}")
for rule in output["capital_rules"]:
    print(f"  * {rule}")

# ── 9. Top correlation pairs ──
print(f"\n{'=' * 70}")
print("TOP 20 MOST CORRELATED PAIRS")
print(f"{'=' * 70}")

pairs = []
for i in range(len(symbols_in_matrix)):
    for j in range(i + 1, len(symbols_in_matrix)):
        s1 = symbols_in_matrix[i]
        s2 = symbols_in_matrix[j]
        c = corr_matrix.loc[s1, s2]
        if not np.isnan(c):
            pairs.append((s1, s2, c))

pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for a, b, c in pairs[:20]:
    print(f"  {a:<25s} <-> {b:<25s}  corr={c:.3f}")

# Most negatively correlated (hedging opportunities)
neg_pairs = [p for p in pairs if p[2] < -0.1]
if neg_pairs:
    neg_pairs.sort(key=lambda x: x[2])
    print(f"\n{'=' * 70}")
    print("NEGATIVELY CORRELATED PAIRS (hedging opportunities)")
    print(f"{'=' * 70}")
    for a, b, c in neg_pairs[:10]:
        print(f"  {a:<25s} <-> {b:<25s}  corr={c:.3f}")
else:
    print("\nNo significantly negatively correlated pairs found.")

print(f"\n{'=' * 70}")
print("RISK INSIGHT")
print(f"{'=' * 70}")
total_in_groups = sum(g["num_assets"] for g in groups)
print(f"  {total_in_groups}/{len(symbols_in_matrix)} assets are in correlated groups")
print(f"  {len(uncorrelated)}/{len(symbols_in_matrix)} assets move independently")
if groups:
    biggest = groups[0]
    print(f"  Largest group has {biggest['num_assets']} assets (avg corr {biggest['avg_correlation']:.3f})")
    print(f"  -> Trading all of them is like trading {biggest['num_assets']}x the same asset!")

print(f"\nDone! Output: {OUTPUT_PATH}")
