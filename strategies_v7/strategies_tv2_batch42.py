#!/usr/bin/env python3
"""TV2 BATCH 42 wrapper — estrategias Pine v4/v5/v6 (30 nuevas categorías)"""
TV2_BATCH42_STRATS = {}
import importlib, sys, os
_d = os.path.dirname(os.path.abspath(__file__))
if _d not in sys.path: sys.path.insert(0, _d)
try:
    _mod = importlib.import_module('tv2_batch42')
    TV2_BATCH42_STRATS.update(getattr(_mod, 'STRATEGY_EXPORT', {}))
except Exception as _e:
    import warnings; warnings.warn(f"tv2_batch42: {_e}")
