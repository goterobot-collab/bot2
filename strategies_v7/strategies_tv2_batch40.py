#!/usr/bin/env python3
"""TV2 BATCH 40 wrapper — estrategias Pine GitHub v4/v5/v6"""
TV2_BATCH40_STRATS = {}
import importlib, sys, os
_d = os.path.dirname(os.path.abspath(__file__))
if _d not in sys.path: sys.path.insert(0, _d)
try:
    _mod = importlib.import_module('tv2_batch40')
    TV2_BATCH40_STRATS.update(getattr(_mod, 'STRATEGY_EXPORT', {}))
except Exception as _e:
    import warnings; warnings.warn(f"tv2_batch40: {_e}")
