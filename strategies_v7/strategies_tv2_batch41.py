#!/usr/bin/env python3
"""
TV2 BATCH 41 wrapper — Freqtrade-adapted strategies
26 strategies from: BB_RPB_TSL, mikedigriz, keithorange, MoniGoMani, paulcpk, CryptoFrog
"""

TV2_BATCH41_STRATS = {}

import importlib, sys, os
_d = os.path.dirname(os.path.abspath(__file__))
if _d not in sys.path: sys.path.insert(0, _d)
try:
    _mod = importlib.import_module('tv2_batch41')
    TV2_BATCH41_STRATS.update(getattr(_mod, 'STRATEGY_EXPORT', {}))
except Exception as _e:
    import warnings; warnings.warn(f"tv2_batch41: {_e}")
