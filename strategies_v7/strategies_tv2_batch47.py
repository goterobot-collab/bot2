#!/usr/bin/env python3
"""TV2 BATCH 47 wrapper — Top 10 TradingView Public Strategies (18 total)"""

TV2_BATCH47_STRATS = {}

import importlib, sys, os

_strat_dir = os.path.dirname(os.path.abspath(__file__))
if _strat_dir not in sys.path:
    sys.path.insert(0, _strat_dir)

try:
    _mod = importlib.import_module('tv2_batch47')
    TV2_BATCH47_STRATS.update(getattr(_mod, 'STRATEGY_EXPORT', {}))
except Exception as _e:
    import warnings
    warnings.warn(f"tv2_batch47: failed to load tv2_batch47: {_e}")
