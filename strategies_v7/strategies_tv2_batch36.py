#!/usr/bin/env python3
"""TV2 BATCH 36 wrapper — Squeeze Momentum + Divergence variants (30 strategies)"""

TV2_BATCH36_STRATS = {}

import importlib, sys, os

_strat_dir = os.path.dirname(os.path.abspath(__file__))
if _strat_dir not in sys.path:
    sys.path.insert(0, _strat_dir)

try:
    _mod = importlib.import_module('tv2_batch36')
    TV2_BATCH36_STRATS.update(getattr(_mod, 'STRATEGY_EXPORT', {}))
except Exception as _e:
    import warnings
    warnings.warn(f"tv2_batch36: failed to load tv2_batch36: {_e}")
