#!/usr/bin/env python3
"""TV2 BATCH 38 wrapper — ML-Inspired + Advanced Pattern Recognition (34 strategies)"""

TV2_BATCH38_STRATS = {}

import importlib, sys, os

_strat_dir = os.path.dirname(os.path.abspath(__file__))
if _strat_dir not in sys.path:
    sys.path.insert(0, _strat_dir)

try:
    _mod = importlib.import_module('tv2_batch38')
    TV2_BATCH38_STRATS.update(getattr(_mod, 'STRATEGY_EXPORT', {}))
except Exception as _e:
    import warnings
    warnings.warn(f"tv2_batch38: failed to load tv2_batch38: {_e}")
