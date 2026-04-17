# Pine v5 Grail Strategy Audit

Audited: grail_001, grail_010, grail_020, grail_030, grail_040 (representative of 51).
All five share the same emit template, so most issues are systemic.

## Critical

### C1. Broken trailing stop (`trail_price=high - N*atr`)
- Files: grail_001 L42, grail_010 L41, grail_020 L43, grail_030 L42, grail_040 L42 (every file).
- Bug: `strategy.exit("trail", trail_price=high - N*atrV, trail_offset=0)` is evaluated once per bar against the live `high`, so the stop oscillates and does not "trail" the highest-high since entry. With `trail_offset=0` it also fires the instant price touches the line; on the entry bar it can immediately stop out (since `high - N*atr` is computed off the current bar's high, not the running max).
- Fix: use `trail_points` (in mintick units) anchored to highest-high tracking, e.g.
  `strategy.exit("trail", "long", trail_points = N*atrV / syminfo.mintick, trail_offset = N*atrV / syminfo.mintick)` — or maintain a `var float peak` and call `strategy.exit(..., stop = peak - N*atrV)`.

### C2. Duplicate-ID `strategy.exit("sltp"...)` plus `strategy.exit("trail"...)` attached to same entry
- Files: all five (lines 41-42 / 40-41 / 42-43, etc).
- Bug: two separate exit orders both target entry id `"long"` with different IDs ("sltp", "trail"). In Pine v5 each `strategy.exit` posts an OCA-bracket bound to the entry; with two brackets the SL/TP bracket and the trail bracket are NOT mutually cancelling. When TP fires, the trail bracket remains live and can re-close (or, more commonly, leave a phantom stop that fires on the next entry's quantity == 0 and is silently dropped, masking real exits in the report).
- Fix: combine into one `strategy.exit("x", "long", stop=..., limit=..., trail_points=..., trail_offset=...)` so all legs are OCA siblings.

## Important

### I1. `default_qty_value=100` (100% of equity) compounding with no risk cap
- Files: all (L11-12).
- Bug: every trade risks 100% of equity, plus the ATR stop distance varies — actual $ risk per trade is uncontrolled. Combined with the broken trail (C1), reported in-sample DDs (e.g. -15.8% on grail_030) are not reproducible.
- Fix: switch to fixed-fraction risk sizing: compute qty from `(equity * riskPct) / (stopDistance)` and pass via `qty=` on `strategy.entry`, or use `strategy.percent_of_equity` with a much smaller value (e.g. 10) and document leverage.

### I2. `process_orders_on_close=false` mismatches backtest assumption
- Files: all (L16).
- Bug: signals computed on bar close (`ta.crossover(close,...)`) but orders execute on the NEXT bar's open. The header in-sample stats almost certainly were generated assuming same-bar fills. This produces a systematic 1-bar slippage versus the JSON metrics.
- Fix: set `process_orders_on_close=true` to match the research harness (or re-run IS metrics with next-bar fills).

### I3. grail_040 missing TP but header omits `tp_atr`
- File: grail_040 L41 uses `limit=na` while header (L7) lists no `tp_atr` — header self-consistent, but the variant relies entirely on the (broken) trail + 24-bar timeout. Combined with C1 this strategy effectively has only the hard SL + timeout exits.
- Fix: same as C1; once trail is correct this design is intentional.

### I4. Timeout exit can race the same-bar entry exits
- Files: all (e.g. grail_001 L43-45).
- Bug: `strategy.close("long")` inside the same `if`-block that just entered (when `bar_index - entryBar >= timeout` fires on a new entry bar where entryBar was just set, `bar_index - entryBar == 0`, so safe — but on re-entry after a flat bar, entryBar may be stale `na` only on first run; the `var int entryBar = na` initialisation is fine). However, after `strategy.close` the `entryBar := na` reset happens only in the timeout branch, NOT in the `exitCond` branch (L47-48). After an exitCond close, entryBar still holds the old bar index; if a new entry fires N bars later the stale value is overwritten — safe — but if the entry-block guard `strategy.position_size == 0` is true and a new entry fires, the timeout check still uses the new entryBar, so OK. Net: no bug, but `entryBar := na` should also be set in the exitCond branch for clarity.
- Fix: add `entryBar := na` after L48 close.

## Nit

### N1. Hardcoded params, no `input.*`
- All files L23-27 (and equivalents) hardcode `bbLen`, `bbMult`, `trendLen`, ATR multiples and timeouts. The only `input` is `useWindow`. Makes TradingView re-tuning impossible without editing source.
- Fix: wrap each numeric param in `input.int` / `input.float` with the JSON value as default.

### N2. ATR length 14 hardcoded, not in header JSON
- All files L36/35/37. Header lists `sl_atr`, `tp_atr` multipliers but not the ATR length itself; backtest reproducibility requires this be documented or made an input.

### N3. `slippage=2` is in ticks, not bps — for ETHUSD 1h that's ~$0.02, negligible vs realistic crypto slippage (5-10 bps). Likely understates costs.

### N4. `useWindow` start timestamp `2018-08-02` hardcoded across all files; should be an `input.time`.

### N5. No `request.security` / `lookahead_on` usage anywhere — no repaint risk on that axis. Good.

## Summary
Two systemic critical bugs (C1, C2) affect every file and likely invalidate the reported in-sample metrics. Fixing the emit template in `tools/emit_pine.py` (referenced in file headers) will repair all 51 strategies at once.
