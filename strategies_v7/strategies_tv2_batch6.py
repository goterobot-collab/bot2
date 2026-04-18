#!/usr/bin/env python3
"""
TV2 BATCH 6 — 16 estrategias de alta popularidad (TV likes > 1500).

Batch 6a (8 estrategias):
  RSI_Divergence       — RSI pivot divergence (v4, 9172L)
  Full_CRYPTO_pack     — EMA+MACD+RSI+OBV (v4, 4859L)
  Price_Volume_Breakout — Price+Volume dual breakout (v5, 4771L)
  HMA_72s_Adaptive     — Adaptive HMA+ (v4, 4552L)
  Hull_MA_Swing_Trader — HMA swing on open (v4, 3733L)
  EVWMA_VWAP_MACD      — EVWMA vs VWAP MACD (v4, 3575L)
  QuantNomad_HA_PSAR   — Heikin Ashi + PSAR (v4, 3188L)
  WaveTrend_MFI_EMA    — Cipher B+ WaveTrend (v4, 2451L)

Batch 6b (8 estrategias):
  MACD_ATR_Strategy    — Fast MACD (3,5,2) + ribbon (v5, 2344L)
  ATR_Dual_Trail       — Dual ATR trailing crossover (v4, 2283L)
  Swing_Hull_RSI_EMA   — HMA pullback + EMA (v5, 2056L)
  ORB_Heikin_Ashi      — HA candles + volume surge (v5, 2005L)
  Squeeze_Momentum     — LazyBear Squeeze Momentum (v5, 1917L)
  Ichimoku_RSI_NoOffset — Ichimoku no-offset + RSI (v4, 1799L)
  Triple_SuperTrend    — 3x SuperTrend + EMA200 (v4, 1609L)
  ZigZag_RSI           — RSI zigzag crossover (v4, 1497L)
"""

TV2_BATCH6_STRATS = {}

import importlib, sys, os
_strat_dir = os.path.dirname(os.path.abspath(__file__))
if _strat_dir not in sys.path:
    sys.path.insert(0, _strat_dir)

for _mod_name in ['tv2_batch6a', 'tv2_batch6b']:
    try:
        _mod = importlib.import_module(_mod_name)
        TV2_BATCH6_STRATS.update(getattr(_mod, 'STRATEGY_EXPORT', {}))
    except Exception as _e:
        import warnings
        warnings.warn(f"tv2_batch6: failed to load {_mod_name}: {_e}")
