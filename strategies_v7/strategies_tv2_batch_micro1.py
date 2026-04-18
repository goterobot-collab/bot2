#!/usr/bin/env python3
"""
TV2 BATCH MICRO1 wrapper — importa estrategias de tv2_batch_micro1.py

Uso:
  from strategies_tv2_batch_micro1 import TV2_BATCH_MICRO1_STRATS
  strategies.update(TV2_BATCH_MICRO1_STRATS)

Estrategias: CVD_MeanRev, OrderFlow_Imbalance, VolAbsorption,
             LiquiditySweep, TickMomentum
"""

import importlib
import sys
import os

TV2_BATCH_MICRO1_STRATS = {}

_strat_dir = os.path.dirname(os.path.abspath(__file__))
if _strat_dir not in sys.path:
    sys.path.insert(0, _strat_dir)

_failed = []
try:
    _mod = importlib.import_module('tv2_batch_micro1')
    _export = getattr(_mod, 'STRATEGY_EXPORT', {})
    TV2_BATCH_MICRO1_STRATS.update(_export)
except Exception as _e:
    _failed.append(('tv2_batch_micro1', str(_e)))

if _failed:
    import warnings
    for _m, _err in _failed:
        warnings.warn(f"strategies_tv2_batch_micro1: failed to load {_m}: {_err}")
