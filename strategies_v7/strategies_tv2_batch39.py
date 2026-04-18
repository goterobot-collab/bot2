#!/usr/bin/env python3
"""TV2 BATCH 39 wrapper — 26 estrategias TV validadas WR>=60% Pine v4/v5/v6"""
TV2_BATCH39_STRATS = {}
import importlib, sys, os
_d = os.path.dirname(os.path.abspath(__file__))
if _d not in sys.path: sys.path.insert(0, _d)
try:
    _mod = importlib.import_module('tv2_batch39')
    TV2_BATCH39_STRATS.update(getattr(_mod, 'STRATEGY_EXPORT', {}))
except Exception as _e:
    import warnings; warnings.warn(f"tv2_batch39: {_e}")
