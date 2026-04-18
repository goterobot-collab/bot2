#!/usr/bin/env python3
"""
TV2 BATCH 7 — 30 estrategias convertidas 2026-03-31.

Batch 7a (10 estrategias):
  Buy_Sell_AO_Stoch_RSI_ATR  — AO + Stoch + RSI + ATR (v5)
  BB_Breakout                — Bollinger Band breakout (v4)
  Full_AllinOne_Risk         — RSI+BB+ATR all-in-one (v4)
  Fibonacci_RSI              — Fib retracement + RSI (v4)
  Scalping_WilliamsR_MACD_SMA — Williams %R + MACD scalp (v4)
  Keltner_Trend              — Keltner ribbon trend (v5)
  ATR_Trailing_Stop          — ATR trailing PSAR (v5)
  ATR_PSAR_QuantNomad        — ATR + Parabolic SAR (v4)
  Turtle_Donchian_ATR        — Turtle Donchian + ATR (v4)
  Two_TP_EMA_WMA             — Two TP EMA+WMA (v4)

Batch 7b (10 estrategias):
  DEMA_ATR                   — DEMA crossover + ATR (v4)
  Triple_EMA_Turtle          — Triple EMA turtle (v4)
  BB_Enhanced                — Enhanced BB (v4)
  Triple_EMA_QQE             — Triple EMA + QQE (v4)
  Hull_T3_Swing              — Hull + T3 swing (v5)
  ATR_Stop_Multiple          — ATR stop multiple (v4)
  ATR_MA_Trend               — ATR + MA trend (v5)
  Crypto_Squeeze             — Crypto squeeze (v5)
  Commas_Bollinger           — 3Commas Bollinger (v4)
  EMA_RSI_ADX                — EMA + RSI + ADX (v6)

Batch 7c (10 estrategias):
  CCI_EMA_RSI                — CCI + EMA + RSI (v4)
  Flash_Momentum             — Flash momentum (v5)
  TradePro_2EMA_StochRSI     — 2EMA + Stoch RSI (v4)
  Crypto_Momentum            — BB + KC squeeze momentum (v5)
  PriceAction_BB             — Price action BB (v4)
  ATR_Trail_Stop             — ATR trail stop (v4)
  Noro_MA_ATR                — Noro MA + ATR (v4)
  KAMA_Strategy              — KAMA adaptive (v4)
  MACD_RSI_Signal            — MACD + RSI signal (v4)
  Aggressive_Scalper         — Aggressive scalper (v5)
"""

TV2_BATCH7_STRATS = {}

import importlib, sys, os
_strat_dir = os.path.dirname(os.path.abspath(__file__))
if _strat_dir not in sys.path:
    sys.path.insert(0, _strat_dir)

for _mod_name in ['tv2_batch7a', 'tv2_batch7b', 'tv2_batch7c']:
    try:
        _mod = importlib.import_module(_mod_name)
        TV2_BATCH7_STRATS.update(getattr(_mod, 'STRATEGY_EXPORT', {}))
    except Exception as _e:
        import warnings
        warnings.warn(f"tv2_batch7: failed to load {_mod_name}: {_e}")
