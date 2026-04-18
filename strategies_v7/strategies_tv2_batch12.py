#!/usr/bin/env python3
"""
TV2 BATCH 12 — 30 estrategias 2026-03-31

Batch 12a (10): Top popularity — PMax, Hull Suite, AO+Stoch, Flawless Victory,
  RSI Divergence, Twin OTT, UT Bot, AlphaTrend, OTT Strategy, SSL Channel

Batch 12b (10): Mid-tier — VWAP Fibo, WaveTrend, Hammers Stars, Triple EMA ATR,
  CRYPTO 3EMA, Double AI SuperTrend, RSI Adaptive T3 SAR, Hulk Scalper,
  Quasimodo, SuperTrend MTF

Batch 12c (10): Lower-tier — Grid Spot, Fibonacci TR, Trend Reversal, Pivot Range,
  RSI OB/OS Div, Reversal Bot, Stoch BB, MACD Bidir, Renko EMA, Swing Failure
"""

TV2_BATCH12_STRATS = {}

import importlib, sys, os

_strat_dir = os.path.dirname(os.path.abspath(__file__))
if _strat_dir not in sys.path:
    sys.path.insert(0, _strat_dir)

for _mod_name in ['tv2_batch12a', 'tv2_batch12b', 'tv2_batch12c']:
    try:
        _mod = importlib.import_module(_mod_name)
        TV2_BATCH12_STRATS.update(getattr(_mod, 'STRATEGY_EXPORT', {}))
    except Exception as _e:
        import warnings
        warnings.warn(f"tv2_batch12: failed to load {_mod_name}: {_e}")
