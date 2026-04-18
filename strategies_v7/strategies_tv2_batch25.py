#!/usr/bin/env python3
"""TV2 BATCH 25 wrapper"""
TV2_BATCH25_STRATS = {}
import importlib, sys, os
_d = os.path.dirname(os.path.abspath(__file__))
if _d not in sys.path: sys.path.insert(0, _d)
try:
    _m = importlib.import_module('tv2_batch25')
    TV2_BATCH25_STRATS.update(getattr(_m, 'STRATEGY_EXPORT', {}))
except Exception as _e:
    import warnings; warnings.warn(f"batch25: {_e}")
