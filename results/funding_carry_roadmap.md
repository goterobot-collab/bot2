# Funding-Rate Carry Strategy — Implementation Roadmap

Cash-and-carry on crypto perps: hold spot, short the perp (invert if funding<0), harvest funding delta-neutral. Soska/Christin et al. (CMU 2021) reported >70% positive-funding WR and BTC Sharpes 7–13 on 2020–H1-2021 data.

## 1. Data needed

| Exchange | Endpoint | Cadence | Hist depth | Limits |
|---|---|---|---|---|
| Binance USD-M | `GET /fapi/v1/fundingRate` | 8h (some 4h since 2023) | since 2019-09 | 500 req/5min/IP, `limit=1000` |
| Bybit v5 | `GET /v5/market/funding/history` | 8h (some 1h/4h) | since 2020 | 200/call, ~120 req/s/UID |
| OKX v5 | `GET /api/v5/public/funding-rate-history` | 8h (some 4h) | rolling **3 mo** REST | 10 req/2 s/IP |

Backfill via Binance Vision daily ZIPs (`data.binance.vision/futures/um/daily/fundingRate/`); OKX needs CoinAPI/Coinglass for >3 mo. Pair with spot OHLCV at funding cadence. Store as `data/funding_<exch>_<sym>.csv` (`ts, fundingRate, markPrice, nextFundingTime`).

## 2. Signal

Per funding tick `t`: `f_t` = funding rate (1 bp/8h ≈ 10.95% APR). Threshold `X` (default ±1 bp).
- `f_t > +X`: **short perp + long spot** (collect from longs)
- `f_t < -X`: **long perp + short spot** (collect from shorts)
- `|f_t| ≤ X`: flat

EMA(3) of `f_t` to debounce; hysteresis (enter 1.5 bp, exit 0.5 bp) to cut churn.

## 3. Capital structure

- Spot leg: 1.00× notional. Use Binance Portfolio Margin so spot collateralises perp.
- Perp leg: 5–10× margin → 10–20% notional in margin. Cap at 3× to survive 30% adverse basis.
- Effective gross leverage 1× delta-neutral; capital efficiency ≈ 0.85–0.95×.
- Fees: maker spot + maker perp ≈ 4 bp round-trip Binance VIP0; carry must clear this per holding interval.

## 4. Risks

- **Funding flips** (e.g. post-ETF Jan 2024) invert sign in one tick → friction loss. Mitigate with hysteresis + sized rebalances.
- **Basis blowout**: LUNA (May 2022), FTX (Nov 2022), USDC depeg (Mar 2023) drove perp-spot bases ±5–15%. MTM loss on the short perp can liquidate even though terminal PnL = 0. Mitigate: low perp leverage, cross-margin, USDT+USDC collateral mix.
- **Counterparty**: exchange insolvency wipes both legs. Cap per-venue notional.
- **ADL/socialised loss** on Bybit/OKX inverse contracts; **stablecoin depeg** of collateral.
- **Carry compression**: arXiv 2510.14435 (Oct 2025) shows Sharpe 6.45 (2020–25) → 4.06 (2024) → **negative (2025)** as the trade crowds.

## 5. Implementation plan (5 steps into `grail_loop.py`)

1. **Fetcher** — new `/home/user/bot2/fetch_funding.py`: `fetch_binance_funding(symbol, start, end) -> pd.DataFrame`, `fetch_bybit_funding(...)`. Output `data/funding_<exch>_<sym>.csv` (`ts, rate, mark`). Reuse rate-limit/backoff from `fetch_data.py`.
2. **Signal builder** — add `sig_funding_carry(df, p)` to `strategies/library.py` (import in `grail_loop.py` next to `sig_rsi2_regime`). `df` = merged spot+funding. Params `enter_bps, exit_bps, ema_n, min_apr`. Returns `(entry, exit)` plus a `side` series (+1 short-perp, –1 long-perp).
3. **PnL engine** — fork `backtest.simulate` as `simulate_carry`: accrue `funding_pnl = -side * notional * rate` each funding ts; subtract `fees = 2 * taker_bp * turnover`. Spot leg = zero-PnL hedge; basis MTM modelled to stress liquidation.
4. **Grail filter** — register `funding_carry` in `STRATEGIES` dict in `grail_loop.py` with grid `{enter_bps:[0.5,1,1.5,2], exit_bps:[0.1,0.3,0.5], ema_n:[1,3,5]}`. Run `--min-wr 0.70 --min-pf 1.5` matching Soska 2021.
5. **Live wiring** — `tools/funding_monitor.py` cron (1h): pull latest funding, evaluate signal, write `results/carry_targets.json` (`{symbol, side, notional_usd, ts}`). Execution adapter out-of-scope.

## 6. Expected metrics

- **Soska/Christin et al. (CMU 2021)**: BTC carry Sharpe 7–13, H1-2021; >70% positive-funding interval WR; ~8% APR funding return at <1% vol.
- **He/Manela/Ross — arXiv 2212.06888 v5 (2024)**: large Sharpes survive top retail Binance fee tier 2020–23.
- **arXiv 2510.14435 (Oct 2025)**: 2020–25 Sharpe 6.45 → 4.06 (2024) → **negative (2025)**.
- Realistic 2026 target: WR 60–65%, net APR 4–8% on deployed capital, max DD 8–12% (basis-spike driven).

Sources: [CMU 2021 Carry Trade](https://www.andrew.cmu.edu/user/azj/files/CarryTrade.v1.0.pdf) · [arXiv 2212.06888](https://arxiv.org/abs/2212.06888) · [arXiv 2510.14435](https://arxiv.org/html/2510.14435v2) · [arXiv 2506.08573](https://arxiv.org/abs/2506.08573) · [BIS WP 1087](https://www.bis.org/publ/work1087.pdf) · [Binance funding](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History) · [Bybit v5](https://bybit-exchange.github.io/docs/v5/market/history-fund-rate) · [OKX v5](https://www.okx.com/docs-v5/en/) · [ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S2096720925000818)
