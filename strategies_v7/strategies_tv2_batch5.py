#!/usr/bin/env python3
"""
TV2 BATCH 5 — 18 nuevas estrategias convertidas 2026-03-31.

Estrategias:
  EMA_Cross_V2    — 5 EMAs on open, volume (Pine v4, 307L) BTC/4h: 558 señales
  Aroon_Strategy  — Aroon oscillator crossover (Pine v4, 394L) BTC/4h: 45
  Ichimoku_V2     — Ichimoku cloud LONG ONLY (Pine v4, 117L) BTC/4h: 277
  Kaufman_AMA     — KAMA adaptive MA cross (Pine v5, 916L) BTC/4h: 161
  Fractal_Break   — Fractal breakout + ATR (Pine v5, 1303L) BTC/4h: 796
  Percent_B       — EMA% Channel + BB (Pine v4, 109L) BTC/4h: 48
  Liquidity_Sweep — Institutional liquidity sweep (Pine v5, 30L) BTC/4h: 104
  Fib_Strategy    — Fibonacci retracement cross (Pine v5, 158L) BTC/4h: 892
  Vol_Spread      — Fourier VSA simplified (Pine v5, 417L) BTC/4h: 1536
  High_WR_Crypto  — HMA + EMA + RSI scalp (Pine v6, 28L) BTC/4h: 407
  Heikin_Ashi_V2  — Heikin Ashi LONG ONLY (Pine v4, 796L) BTC/4h: 811
  Supertrend_V2   — Supertrend EMA Vol LONG (Pine v5, 164L) BTC/4h: 1016
  Multi_Confirm   — Multi-band quantile (Pine v6, 113L) BTC/4h: 172
  QuantNomad_V2   — UT Bot ATR trailing (Pine v4, 5422L) BTC/4h: 1404
  Adaptive_RSI    — T3 + PSAR adaptive (Pine v6, 2109L) BTC/4h: 1024
  Dynamic_Support — Dynamic S/R pivots (Pine v6, 387L) BTC/4h: 246
  TPSL_Strategy   — 3EMA + SMA200 + RSI (Pine v4, 2351L) BTC/4h: 109
  Trailing_TP     — SMA cross trailing (Pine v4, 1684L) BTC/4h: 315

Uso:
  from strategies_tv2_batch5 import TV2_BATCH5_STRATS
  strategies.update(TV2_BATCH5_STRATS)
"""

TV2_BATCH5_STRATS = {}

_modules = [
    ('tv2_batch5a', ['EMA_Cross_V2', 'Aroon_Strategy', 'Ichimoku_V2', 'Kaufman_AMA', 'Fractal_Break', 'Percent_B']),
    ('tv2_batch5b', ['Liquidity_Sweep', 'Fib_Strategy', 'Vol_Spread', 'High_WR_Crypto', 'Heikin_Ashi_V2', 'Supertrend_V2']),
    ('tv2_batch5c', ['Multi_Confirm', 'QuantNomad_V2', 'Adaptive_RSI', 'Dynamic_Support', 'TPSL_Strategy', 'Trailing_TP']),
]

import importlib, sys, os
_strat_dir = os.path.dirname(os.path.abspath(__file__))
if _strat_dir not in sys.path:
    sys.path.insert(0, _strat_dir)

_loaded = 0
_failed = []
for _mod_name, _strat_keys in _modules:
    try:
        _mod = importlib.import_module(_mod_name)
        _export = getattr(_mod, 'STRATEGY_EXPORT', {})
        TV2_BATCH5_STRATS.update(_export)
        _loaded += 1
    except Exception as _e:
        _failed.append((_mod_name, str(_e)))

if _failed:
    import warnings
    for _m, _err in _failed:
        warnings.warn(f"tv2_batch5: failed to load {_m}: {_err}")
