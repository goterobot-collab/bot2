#!/usr/bin/env python3
"""
TV2 BATCH 11 — 27 estrategias 2026-03-31

Batch 11a (9): S/R + Hull + Renko + NW
  Iron_SR_Auto, SR_Trend, Adaptive_Hull_72s, Hull_Swing_SEASIDE, HA_PSAR_QN,
  NW_Envelope, Renko_Strategy, Renko_V2, PSAR_Close_QN

Batch 11b (9): Elder/DEMA/Chop/Pivot/CMO
  Elder_Ray_Bull, DEMA_ATR_Dashboard, SR_5min_Intraday, Turtle_Triple_EMA,
  Triple_DEMA_8_20_63, Chop_DMI_PSAR, Chande_Momentum, Pivot_Of_Pivot, SR_Ceyhun

Batch 11c (9): Renko/Turtle/Pivot/Aroon/Range
  Renko_SMT_ATR, Turtle_System_Eugene, Renko_Emulator_OCC, Renko_Intraday,
  Pivot_Reversal_RSI, Aroon_Strategy, Ichimoku_RSI_NoOff,
  SuperTrend_Crossover, Range_Filter_Strat
"""

TV2_BATCH11_STRATS = {}

import importlib, sys, os

_strat_dir = os.path.dirname(os.path.abspath(__file__))
if _strat_dir not in sys.path:
    sys.path.insert(0, _strat_dir)

for _mod_name in ['tv2_batch11a', 'tv2_batch11b', 'tv2_batch11c']:
    try:
        _mod = importlib.import_module(_mod_name)
        TV2_BATCH11_STRATS.update(getattr(_mod, 'STRATEGY_EXPORT', {}))
    except Exception as _e:
        import warnings
        warnings.warn(f"tv2_batch11: failed to load {_mod_name}: {_e}")
