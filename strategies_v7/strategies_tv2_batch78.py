#!/usr/bin/env python3
"""
TV2 BATCH 78 wrapper — importa estrategias de tv2_batch78.py

Uso:
  from strategies_tv2_batch78 import TV2_BATCH78_STRATS
  strategies.update(TV2_BATCH78_STRATS)
"""

import importlib
import sys
import os

TV2_BATCH78_STRATS = {}

_strat_dir = os.path.dirname(os.path.abspath(__file__))
if _strat_dir not in sys.path:
    sys.path.insert(0, _strat_dir)

_failed = []
try:
    _mod = importlib.import_module('tv2_batch78')
    _export = getattr(_mod, 'STRATEGY_EXPORT', {})
    TV2_BATCH78_STRATS.update(_export)
except Exception as _e:
    _failed.append(('tv2_batch78', str(_e)))

if _failed:
    import warnings
    for _m, _err in _failed:
        warnings.warn(f"strategies_tv2_batch78: failed to load {_m}: {_err}")
