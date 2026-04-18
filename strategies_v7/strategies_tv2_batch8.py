#!/usr/bin/env python3
"""
TV2 BATCH 8 — 27 estrategias convertidas 2026-03-31.

Batch 8a (9 estrategias):
  Trendline_Fib_ST           — Multi Fibonacci Supertrend (v6, 2511L)
  BB_Breakout_V2             — BB breakout [Luca] (v5, 2048L)
  Ichimoku_NoOffset_V2       — Ichimoku no-offset no-repaint (v4, 1799L)
  MA_BB_RSI                  — MA+BB+RSI bounce (v5, 1550L)
  Donchian_Free_Bot          — Donchian channel bot (v4, 1336L)
  Fractal_Breakout_KL        — Fractal breakout + ATR (v5, 1304L)
  Order_Block_FVG            — Order Block + FVG (v6, 1301L)
  Fibonacci_Trend_Reversal   — Fib retracement reversal (v6, 1054L)
  Fractal_Breakout_Simple    — Simple fractal breakout (v4, 1051L)

Batch 8b (9 estrategias):
  Optimized_HA               — Heikin Ashi + EMA filter (v5, 950L)
  KAMA_TradeDots             — KAMA adaptive MA cross (v5, 916L)
  CCI_EMA_ATR                — CCI+EMA with ATR TP/SL (v5, 887L)
  BB_Pending_Alerts          — BB cross signals (v5, 875L)
  Flash_Strategy_ATR         — Momentum+RSI+EMA+ATR (v5, 871L)
  TradePro_2EMA_Stoch_ATR    — 2EMA + StochRSI + ATR (v5, 843L)
  Bollinger_Stop             — BB stop reversal (v5, 836L)
  Crypto_Momentum_V4         — MACD+RSI momentum (v4, 827L)
  PriceAction_BB_V2          — Engulfing + BB (v4, 819L)

Batch 8c (9 estrategias):
  Ichimoku_QQE               — Ichimoku + QQE (v4, 800L)
  HA_PriceAction_Long        — Heikin Ashi LONG ONLY (v4, 796L)
  HighLow_Channel_Swing      — Hi/Lo channel + EMA (v4, 751L)
  ATR_Trailing_SL            — ATR trailing stop (v4, 745L)
  Fractal_Proximity_Scalp    — Fractal proximity + MA (v5, 738L)
  BB_RSI_MACD                — BB+RSI+MACD triple (v4, 728L)
  Noro_MA_ATR_V2             — Noro MA+ATR v2 (v4, 707L)
  KAMA_TradeDots_V2          — KAMA + EMA trend filter (v5, 704L)
  EMA_SMA_RSI_MACD           — EMA+SMA+RSI+MACD (v4, 693L)
"""

TV2_BATCH8_STRATS = {}

import importlib, sys, os
_strat_dir = os.path.dirname(os.path.abspath(__file__))
if _strat_dir not in sys.path:
    sys.path.insert(0, _strat_dir)

for _mod_name in ['tv2_batch8a', 'tv2_batch8b', 'tv2_batch8c']:
    try:
        _mod = importlib.import_module(_mod_name)
        TV2_BATCH8_STRATS.update(getattr(_mod, 'STRATEGY_EXPORT', {}))
    except Exception as _e:
        import warnings
        warnings.warn(f"tv2_batch8: failed to load {_mod_name}: {_e}")
