#!/usr/bin/env python3
"""TV2 BATCH 55 wrapper — Estrategias GitHub Pine Script v5"""

TV2_BATCH55_STRATS = {}

import importlib, sys, os

_strat_dir = os.path.dirname(os.path.abspath(__file__))
if _strat_dir not in sys.path:
    sys.path.insert(0, _strat_dir)

try:
    _mod = importlib.import_module('tv2_batch55')
    TV2_BATCH55_STRATS.update(getattr(_mod, 'STRATEGY_EXPORT', {}))
except Exception as _e:
    import warnings
    warnings.warn(f"tv2_batch55: failed to load: {_e}")
