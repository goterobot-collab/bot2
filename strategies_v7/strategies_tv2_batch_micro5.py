#!/usr/bin/env python3
"""
TV2 BATCH MICRO5 wrapper — importa estrategias de tv2_batch_micro5.py

Uso:
  from strategies_tv2_batch_micro5 import TV2_BATCH_MICRO5_STRATS
  strategies.update(TV2_BATCH_MICRO5_STRATS)
"""

import importlib
import sys
import os

TV2_BATCH_MICRO5_STRATS = {}

_strat_dir = os.path.dirname(os.path.abspath(__file__))
if _strat_dir not in sys.path:
    sys.path.insert(0, _strat_dir)

_failed = []
try:
    _mod = importlib.import_module('tv2_batch_micro5')
    _export = getattr(_mod, 'STRATEGY_EXPORT', {})
    TV2_BATCH_MICRO5_STRATS.update(_export)
except Exception as _e:
    _failed.append(('tv2_batch_micro5', str(_e)))

if _failed:
    import warnings
    for _name, _err in _failed:
        warnings.warn(f'Failed to load {_name}: {_err}')
