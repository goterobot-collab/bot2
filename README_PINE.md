# TradingView Backtest Workflow

## 1. Load a strategy

1. Open TradingView, any chart.
2. Open the **Pine Editor** (bottom panel → "Pine Editor").
3. Click **Open** → **New blank strategy** (or just paste over).
4. Paste the contents of one `.pine` file from `pine_sources/`.
5. Click **Save** (give it any name).
6. Click **Add to chart**.

## 2. Run the 1-year backtest on each asset × TF

For each of the 8 strategies:

| Symbol      | Exchange notation on TV     |
|-------------|-----------------------------|
| ADA         | `BINANCE:ADAUSDT`           |
| DOGE        | `BINANCE:DOGEUSDT`          |
| INJ         | `BINANCE:INJUSDT`           |
| ETH         | `BINANCE:ETHUSDT`           |

For each symbol, cycle through timeframes **5m**, **15m**, **1h**.

The `.pine` scripts default to a **365-day** backtest window via the `bars_back`
input. You can widen it in the strategy dialog; leaving it at 365 matches
"at least 1 year" as requested.

After each run, open the **Strategy Tester** tab at the bottom → **Performance
Summary** and record:

- **Percent Profitable** → WR %
- **Total Closed Trades** → trades
- **Profit Factor** → PF
- **Net Profit %** → return
- **Max Drawdown %** → max DD

## 3. Fill in `results_template.md`

The template has one row per strategy × asset × TF (8 × 4 × 3 = 96 rows).
Enter the numbers from Strategy Tester, then **sort by WR descending** and
**filter WR>60% AND trades>5** — that's your final list.

## 4. (Optional) Cross-check with Python

```bash
python3 fetch_data.py              # requires outbound access to Binance
python3 backtest.py                # writes results/ folder
```

The Python `backtest.py` uses the exact same entry/exit logic and applies the
same filter (WR>60%, trades>5), sorting by WR desc. Numbers won't be identical
to TradingView (different execution-model conventions) but should be in the
same ballpark.

## 5. Notes on the filter

- "More than 5 trades in the asset's full history" — the 365-day window is
  long enough that any strategy generating <5 trades in that span is effectively
  inactive on that TF and gets auto-rejected.
- On very low-TF high-activity strategies (5m), you may see 200+ trades. Those
  stress-test the edge and fees/slippage matter most.
- On 1h, 5–20 trades per year is normal — watch out for sample-size traps.
  The filter `trades>5` is the floor from your spec, not a recommendation.
