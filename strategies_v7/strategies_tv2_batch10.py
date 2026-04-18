#!/usr/bin/env python3
"""
TV2 BATCH 10 — 31 estrategias 2026-03-31

Batch 10a (9): SuperTrend variants
  SuperTrend_Classic, Pivot_SuperTrend, AI_SuperTrend_Pivot, SuperTrend_ATR_TSL,
  HA_SuperTrend, SuperTrend_CCI, AI_Volume_SuperTrend, Triple_SuperTrend, ATR_GOD_4ST

Batch 10b (10): VWAP/MACD/Volume
  VWAP_Trendfollow, Price_Volume_Breakout, Full_Crypto_Pack, MACD_Reloaded,
  MACD_ATR_Trail, MACD_BB_RSI_Alorse, Scalping_Crypto_Stocks, Keltner_ETH_DMI,
  Squeeze_Momentum_TPSL, VWAP_Momentum_Pullback

Batch 10c (12): Fibonacci/Divergence/LR/Others
  Fibonacci_RSI_VWMA, Price_Divergence_Multi, CipherB_WaveTrend, ATR_Trail_Dual,
  Crypto_Scalper_SAR, Bitcoin_Momentum_EMA, Gaussian_Channel, Linear_Regression_Bo,
  Swing_Hull_T3, HighLow_Channel_Bo, Linear_Regression_Pearson, Donchian_HH_LL
"""

TV2_BATCH10_STRATS = {}

import importlib, sys, os

_strat_dir = os.path.dirname(os.path.abspath(__file__))
if _strat_dir not in sys.path:
    sys.path.insert(0, _strat_dir)

for _mod_name in ['tv2_batch10a', 'tv2_batch10b', 'tv2_batch10c']:
    try:
        _mod = importlib.import_module(_mod_name)
        TV2_BATCH10_STRATS.update(getattr(_mod, 'STRATEGY_EXPORT', {}))
    except Exception as _e:
        import warnings
        warnings.warn(f"tv2_batch10: failed to load {_mod_name}: {_e}")
