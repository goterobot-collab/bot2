#!/usr/bin/env python3
"""
TV2 BATCH 79 wrapper — importa estrategias de tv2_batch79.py

Uso:
  from strategies_tv2_batch79 import TV2_BATCH79_STRATS
  strategies.update(TV2_BATCH79_STRATS)
"""

import importlib
import sys
import os

TV2_BATCH79_STRATS = {}

_strat_dir = os.path.dirname(os.path.abspath(__file__))
if _strat_dir not in sys.path:
    sys.path.insert(0, _strat_dir)

_failed = []
try:
    _mod = importlib.import_module('tv2_batch79')
    _export = getattr(_mod, 'STRATEGY_EXPORT', {})
    TV2_BATCH79_STRATS.update(_export)
except Exception as _e:
    _failed.append(('tv2_batch79', str(_e)))

if _failed:
    import warnings
    for _m, _err in _failed:
        warnings.warn(f"strategies_tv2_batch79: failed to load {_m}: {_err}")
