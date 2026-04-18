# Sandbox Canary Report (Paso 1)

Generated: 2026-04-18T01:00:46.781572+00:00

| ID | Strategy | Sym | TF | Expected WR | Observed WR | Gap | Status |
|----|----------|-----|----|-------------|-------------|-----|--------|
| C1 | B5_MA_Envelope_3 | SFP | 1d | 85.2% | 85.2% | +0.0pp | PASS |
| C2 | VWAP_Double | AGT | 1h | 82.2% | 82.2% | +0.0pp | PASS |
| C3 | TV_Daily_Close_Signal | SWARMS | 15m | 79.3% | N/A | N/A | IMPL_MISSING |

**Pass**: 2/3  (gate: 3/3)

**Missing impl**: C3 — not in `mac_optuna/` or `strategies_v7/`
