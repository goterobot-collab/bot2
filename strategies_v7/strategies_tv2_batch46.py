#!/usr/bin/env python3
"""TV2 BATCH 46 wrapper — VWAPPullback/AlphaTrend/BBRSIBOV/BalancePowerHA/Ichimoku/BBofVWAP/
   BBFib618/SmoothedHA/TripleRSI/ATRRSIv2/StochRSIMFIEMA/TripleSupertrend/ARRVWAPIntraday/
   ADXv2/EightDayRun/FourEMAVol/FibsMarket/BTSAR/VADER/AndeanScalping (20 estrategias)"""

TV2_BATCH46_STRATS = {}

import importlib, sys, os

_strat_dir = os.path.dirname(os.path.abspath(__file__))
if _strat_dir not in sys.path:
    sys.path.insert(0, _strat_dir)

try:
    _mod = importlib.import_module('tv2_batch46')
    TV2_BATCH46_STRATS.update(getattr(_mod, 'STRATEGY_EXPORT', {}))
except Exception as _e:
    import warnings
    warnings.warn(f"tv2_batch46: failed to load tv2_batch46: {_e}")
