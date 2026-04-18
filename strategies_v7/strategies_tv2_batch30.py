#!/usr/bin/env python3
TV2_BATCH30_STRATS = {}
import importlib, sys, os
_d = os.path.dirname(os.path.abspath(__file__))
if _d not in sys.path: sys.path.insert(0, _d)
try:
    _m = importlib.import_module('tv2_batch30')
    TV2_BATCH30_STRATS.update(getattr(_m, 'STRATEGY_EXPORT', {}))
except Exception as _e:
    import warnings; warnings.warn(f"batch30: {_e}")
