#!/usr/bin/env python3
"""
TV2 BATCH 3 — Wrapper para las 27 estrategias convertidas por TV V2 agent.
Pine v4/v5/v6 → Python, validadas en BTC/USDT:USDT 2020-2026.

Estrategias incluidas (15 originales):
  Barking_Rat_Lite      — FVG + RSI + EMA band (Pine v5, TV WR=80%)
  Squeeze_Breakout_BB_KC — BB dentro de KC squeeze (Pine v4, TV WR=81.8%)
  3EMA_Stoch_RSI_ATR    — StochRSI + triple EMA (Pine v4, TV WR=77.8%)
  HTF_MACD_MFI          — MACD + MFI con HTF filter (Pine v4, TV WR=62.5%)
  Adaptive_KDJ_MTF      — KDJ oscillator MTF (Pine v5, TV WR=85.7%)
  RSI_Professional      — RSI crossover + EMA (Pine v4, TV WR=85.7%)
  3Commas_DCA           — MA crossover DCA long-only (Pine v5, TV WR=66.4%)
  3Commas_Bot           — ATR trailing + trend (Pine v5, TV WR=61.5%)
  Golden_Triangle       — Bull flag variant (Pine v4, TV WR=93.8% / 13 trades)
  Advanced_Position_Mgmt — Multi-condición entry (Pine v5, TV WR=60.6%)
  Entry_Fragger         — PVSRA entry signals (Pine v5, TV WR=60.6%)
  CoGrid_Management     — Grid breakout entry (Pine v4)
  alphatrend            — AlphaTrend ATR+MFI dinámico (Pine v5, © KivancOzbilgic)
  smc_strategy          — SMC Order Blocks + BoS (Pine v5)
  ha_supertrend         — Heikin Ashi Supertrend (Pine v5, © jordanfray)

Nuevas estrategias batch4 (12 añadidas 2026-03-31):
  Parabolic_RSI         — SAR sobre RSI (Pine v6, likes=328) BTC/4h: 1800 señales
  VWAP_RSI              — RSI sobre VWAP rolling (Pine v4, likes=1507) BTC/4h: 374
  Donchian              — Breakout Donchian con doble confirm (Pine v4, likes=342) BTC/4h: 985
  Hull_MA               — Adaptive HMA+ slope+charged (Pine v4, likes=4552) BTC/4h: 155
  DEMA_Cross            — EMA doble crossover en open (Pine v4, likes=307) BTC/4h: 550
  FVG_Strat             — Fair Value Gap + session (Pine v6, likes=22) BTC/4h: 2454
  Divergence_RSI        — WaveTrend + MFI + EMA pullback (Pine v4, likes=2451) BTC/4h: 11
  Momentum_Crypto       — Squeeze Momentum (LazyBear) (Pine v4, likes=827) BTC/4h: 728
  ATR_Trailing          — Dual ATR trailing crossover (Pine v4, likes=2282) BTC/4h: 346
  MFI_Strategy          — MFI extreme + candle body (Pine v4, likes=287) BTC/4h: 639
  OBV_Strategy          — OBV oscillator EMA cross (Pine v4, likes=812) BTC/4h: 631
  ZigZag_RSI            — ZigZag RSI trend flip (Pine v4, likes=1497) BTC/4h: 205

Uso en Optuna:
  from strategies_tv2_batch3 import TV2_BATCH3_STRATS
  strategies.update(TV2_BATCH3_STRATS)

Actualizado: 2026-03-31 — batch4: 12 nuevas estrategias (27 total)
"""

TV2_BATCH3_STRATS = {}

_modules = [
    ('tv2_barking_rat',      'Barking_Rat_Lite'),
    ('tv2_squeeze_bb_kc',    'Squeeze_Breakout_BB_KC'),
    ('tv2_3ema_stoch_rsi',   '3EMA_Stoch_RSI_ATR'),
    ('tv2_htf_macd_mfi',     'HTF_MACD_MFI'),
    ('tv2_adaptive_kdj',     'Adaptive_KDJ_MTF'),
    ('tv2_rsi_professional', 'RSI_Professional'),
    ('tv2_3commas_dca',      '3Commas_DCA'),
    ('tv2_3commas_bot',      '3Commas_Bot'),
    ('tv2_golden_triangle',  'Golden_Triangle'),
    ('tv2_advanced_pos_mgmt','Advanced_Position_Mgmt'),
    ('tv2_entry_fragger',    'Entry_Fragger'),
    ('tv2_cogrid',           'CoGrid_Management'),
    # --- Nuevas 2026-03-31 (batch4) ---
    ('tv2_alphatrend',       'alphatrend'),
    ('tv2_smc_strategy',     'smc_strategy'),
    ('tv2_ha_supertrend',    'ha_supertrend'),
    # --- Nuevas batch4 TV V2 2026-03-31 ---
    ('tv2_parabolic_rsi',    'Parabolic_RSI'),
    ('tv2_vwap_rsi',         'VWAP_RSI'),
    ('tv2_donchian',         'Donchian'),
    ('tv2_hull_ma',          'Hull_MA'),
    ('tv2_dema_cross',       'DEMA_Cross'),
    ('tv2_fvg_strat',        'FVG_Strat'),
    ('tv2_divergence_rsi',   'Divergence_RSI'),
    ('tv2_momentum_crypto',  'Momentum_Crypto'),
    ('tv2_atr_trailing',     'ATR_Trailing'),
    ('tv2_mfi_strategy',     'MFI_Strategy'),
    ('tv2_obv_strategy',     'OBV_Strategy'),
    ('tv2_zigzag_rsi',       'ZigZag_RSI'),
]

import importlib, sys, os
_strat_dir = os.path.dirname(os.path.abspath(__file__))
if _strat_dir not in sys.path:
    sys.path.insert(0, _strat_dir)

_loaded = 0
_failed = []
for _mod_name, _strat_key in _modules:
    try:
        _mod = importlib.import_module(_mod_name)
        _export = getattr(_mod, 'STRATEGY_EXPORT', {})
        TV2_BATCH3_STRATS.update(_export)
        _loaded += 1
    except Exception as _e:
        _failed.append((_mod_name, str(_e)))

if _failed:
    import warnings
    for _m, _err in _failed:
        warnings.warn(f"tv2_batch3: failed to load {_m}: {_err}")
