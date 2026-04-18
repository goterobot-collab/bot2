#!/usr/bin/env python3
"""
External Optuna search spaces for batch strategies (B1-B6).
These strategies have hardcoded params that need per-asset optimization.
Generated from analysis of batch strategy source code.

Usage in optuna_worker.py:
    from batch_search_spaces import BATCH_SEARCH_SPACES, make_space_func
    if strategy_name in BATCH_SEARCH_SPACES:
        space_fn = make_space_func(strategy_name)
        params = space_fn(trial)
"""

# Each entry maps strategy_name -> dict of param_name -> (type, low, high)
# type is 'int' or 'float'
# For optuna: trial.suggest_int(name, low, high) or trial.suggest_float(name, low, high)
#
# Search space design:
#   - Periods: default/2 to default*2 (clamped to sensible min/max)
#   - Thresholds (RSI, Stoch, etc.): wider range around default
#   - Distance percentages: 0.3x to 3x of default
#   - Multipliers (std dev, ATR mult): 0.5x to 2x of default

BATCH_SEARCH_SPACES = {

    # =========================================================================
    # BATCH 2 STRATEGIES
    # =========================================================================

    # 1. B2_VWAP_Bounce (341 bots) — s_vwap_bounce
    # Default: rsi(14), close < vwap*0.998 (dist=0.002), rsi<40 buy, close > vwap*1.002, rsi>60 sell
    'B2_VWAP_Bounce': {
        'rsi_period': ('int', 7, 28),
        'vwap_buy_dist': ('float', 0.001, 0.006),
        'vwap_sell_dist': ('float', 0.001, 0.006),
        'rsi_buy_thresh': ('int', 25, 50),
        'rsi_sell_thresh': ('int', 55, 75),
    },

    # 7. B2_Rubber_Band (172 bots) — s_rubber_band
    # Default: ema(20), dist < -4% buy, dist > 0% sell
    'B2_Rubber_Band': {
        'ema_period': ('int', 10, 40),
        'buy_dist_pct': ('float', -8.0, -2.0),
        'sell_dist_pct': ('float', -1.0, 3.0),
    },

    # 9. B2_MeanRev_EMA21 (119 bots) — s_mean_rev_ema21
    # Default: ema(21), pct < -3% buy, pct > 3% sell
    'B2_MeanRev_EMA21': {
        'ema_period': ('int', 10, 42),
        'buy_pct': ('float', -6.0, -1.5),
        'sell_pct': ('float', 1.5, 6.0),
    },

    # 10. B2_PctRank_Reversal (108 bots) — s_pct_rank_reversal
    # Default: percent_rank(100), pr<10 buy, pr>90 sell
    'B2_PctRank_Reversal': {
        'lookback': ('int', 50, 200),
        'buy_thresh': ('int', 3, 20),
        'sell_thresh': ('int', 80, 97),
    },

    # 17. B2_Oversold_5pct (75 bots) — s_oversold_bounce
    # Default: 3-bar drop < -5% buy, 3-bar gain > 3% sell
    'B2_Oversold_5pct': {
        'lookback_bars': ('int', 2, 6),
        'buy_drop_pct': ('float', -10.0, -3.0),
        'sell_gain_pct': ('float', 1.5, 6.0),
    },

    # 18. B2_BB_50_2 (71 bots) — s_bb_50_2
    # Default: bollinger_bands(50, 2)
    'B2_BB_50_2': {
        'bb_period': ('int', 25, 100),
        'bb_std': ('float', 1.0, 3.0),
    },

    # 23. B2_Keltner_Bounce (66 bots) — s_keltner_bounce
    # Default: keltner_channels(20, 2)
    'B2_Keltner_Bounce': {
        'kc_period': ('int', 10, 40),
        'kc_mult': ('float', 1.0, 3.0),
    },

    # 25. B2_MeanRev_SMA20 (59 bots) — s_mean_rev_sma20
    # Default: sma(20), pct < -5% buy, pct > 5% sell
    'B2_MeanRev_SMA20': {
        'sma_period': ('int', 10, 40),
        'buy_pct': ('float', -10.0, -2.5),
        'sell_pct': ('float', 2.5, 10.0),
    },

    # 33. B2_BB_MFI (50 bots) — s_bb_mfi_combo
    # Default: bb(20,2), mfi(14), mfi<20 buy, mfi>80 sell
    'B2_BB_MFI': {
        'bb_period': ('int', 10, 40),
        'bb_std': ('float', 1.0, 3.0),
        'mfi_period': ('int', 7, 28),
        'mfi_buy_thresh': ('int', 10, 30),
        'mfi_sell_thresh': ('int', 70, 90),
    },

    # 35. B2_BB_1Std (49 bots) — s_bb_1std
    # Default: bollinger_bands(20, 1)
    'B2_BB_1Std': {
        'bb_period': ('int', 10, 40),
        'bb_std': ('float', 0.5, 2.0),
    },

    # 38. B2_BB_RSI (48 bots) — s_bb_rsi_combo
    # Default: bb(20,2), rsi(14), rsi<30 buy, rsi>70 sell
    'B2_BB_RSI': {
        'bb_period': ('int', 10, 40),
        'bb_std': ('float', 1.0, 3.0),
        'rsi_period': ('int', 7, 28),
        'rsi_buy_thresh': ('int', 15, 40),
        'rsi_sell_thresh': ('int', 60, 85),
    },

    # 42. B2_LinReg_MeanRev (40 bots) — s_linear_reg_mean_rev
    # Default: linear_regression(20), close < lr*0.97 buy, close > lr*1.03 sell
    'B2_LinReg_MeanRev': {
        'lr_period': ('int', 10, 40),
        'buy_dist': ('float', 0.01, 0.06),
        'sell_dist': ('float', 0.01, 0.06),
    },

    # 48. B2_ZScore_1_5 (39 bots) — s_zscore_1_5
    # Default: zscore(20), z < -1.5 buy, z > 1.5 sell
    'B2_ZScore_1_5': {
        'zscore_period': ('int', 10, 40),
        'buy_zscore': ('float', -3.0, -1.0),
        'sell_zscore': ('float', 1.0, 3.0),
    },

    # 50. B2_BB_Bounce (37 bots) — s_bb_bounce
    # Default: bollinger_bands(20, 2)
    'B2_BB_Bounce': {
        'bb_period': ('int', 10, 40),
        'bb_std': ('float', 1.0, 3.0),
    },

    # =========================================================================
    # BATCH 3 STRATEGIES
    # =========================================================================

    # 21. B3_ATR_Channel (67 bots) — s_atr_channel
    # Default: ema(20), atr(14), bands = ema +/- 2*atr
    'B3_ATR_Channel': {
        'ema_period': ('int', 10, 40),
        'atr_period': ('int', 7, 28),
        'atr_mult': ('float', 1.0, 4.0),
    },

    # 31. B3_Hammer_Buy (52 bots) — s_hammer_buy
    # Default: rsi(14), rsi<40 buy, rsi>70 sell
    'B3_Hammer_Buy': {
        'rsi_period': ('int', 7, 28),
        'rsi_buy_thresh': ('int', 25, 50),
        'rsi_sell_thresh': ('int', 60, 85),
    },

    # 32. B3_Multi_Osc (50 bots) — s_multi_osc_confirm
    # Default: rsi(14)<30, stoch(14,3)<20, cci(20)<-100, mfi(14)<20
    'B3_Multi_Osc': {
        'rsi_period': ('int', 7, 28),
        'rsi_buy_thresh': ('int', 20, 40),
        'stoch_period': ('int', 7, 28),
        'stoch_smooth': ('int', 2, 5),
        'stoch_buy_thresh': ('int', 10, 30),
        'cci_period': ('int', 10, 40),
        'cci_buy_thresh': ('int', -200, -50),
        'mfi_period': ('int', 7, 28),
        'mfi_buy_thresh': ('int', 10, 30),
        'rsi_sell_thresh': ('int', 60, 85),
        'stoch_sell_thresh': ('int', 70, 90),
        'cci_sell_thresh': ('int', 50, 200),
    },

    # 36. B3_RSI_Stoch_BB (48 bots) — s_rsi_stoch_bb
    # Default: rsi(14)<30, stoch(14,3)<20, bb(20,2)
    'B3_RSI_Stoch_BB': {
        'rsi_period': ('int', 7, 28),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 85),
        'stoch_period': ('int', 7, 28),
        'stoch_smooth': ('int', 2, 5),
        'stoch_buy_thresh': ('int', 10, 30),
        'stoch_sell_thresh': ('int', 70, 90),
        'bb_period': ('int', 10, 40),
        'bb_std': ('float', 1.0, 3.0),
    },

    # 39. B3_VWAP_RSI_MACD (46 bots) — s_vwap_rsi_macd
    # Default: vwap dist 0.002, rsi(14)<35 buy, rsi>65 sell, macd cross
    'B3_VWAP_RSI_MACD': {
        'vwap_buy_dist': ('float', 0.001, 0.006),
        'vwap_sell_dist': ('float', 0.001, 0.006),
        'rsi_period': ('int', 7, 28),
        'rsi_buy_thresh': ('int', 20, 45),
        'rsi_sell_thresh': ('int', 55, 80),
    },

    # 40. B3_Harami_Reversal (41 bots) — s_harami_reversal
    # Default: rsi(14)<40 buy, rsi>70 sell, body ratio 0.5
    'B3_Harami_Reversal': {
        'rsi_period': ('int', 7, 28),
        'rsi_buy_thresh': ('int', 25, 50),
        'rsi_sell_thresh': ('int', 60, 85),
        'body_ratio': ('float', 0.3, 0.7),
    },

    # 45. B3_BB_ADX_RSI (39 bots) — s_bb_adx_rsi
    # Default: bb(20,2), adx(14)<25 range, rsi(14)<30 buy, rsi>70 sell
    'B3_BB_ADX_RSI': {
        'bb_period': ('int', 10, 40),
        'bb_std': ('float', 1.0, 3.0),
        'adx_period': ('int', 7, 28),
        'adx_range_thresh': ('int', 15, 35),
        'rsi_period': ('int', 7, 28),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 85),
    },

    # 47. B3_BB_SuperTrend (39 bots) — s_bb_supertrend
    # Default: bb(20,2), supertrend(10,3)
    'B3_BB_SuperTrend': {
        'bb_period': ('int', 10, 40),
        'bb_std': ('float', 1.0, 3.0),
        'st_period': ('int', 5, 20),
        'st_mult': ('float', 1.5, 5.0),
    },

    # =========================================================================
    # BATCH 4 STRATEGIES
    # =========================================================================

    # 6. B4_Price_EMA_Dist_3 (177 bots) — s_price_ema_distance
    # Default: ema(50), dist < -3% buy, dist > 3% sell
    'B4_Price_EMA_Dist_3': {
        'ema_period': ('int', 25, 100),
        'buy_dist_pct': ('float', -6.0, -1.5),
        'sell_dist_pct': ('float', 1.5, 6.0),
    },

    # 12. B4_HVol_MeanRev (108 bots) — s_hvol_mean_rev
    # Default: historical_volatility(20), quantile(0.9) over 50 bars, rsi(14)<30 buy, rsi>60 sell
    'B4_HVol_MeanRev': {
        'hv_period': ('int', 10, 40),
        'hv_lookback': ('int', 25, 100),
        'hv_quantile': ('float', 0.8, 0.95),
        'rsi_period': ('int', 7, 28),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 50, 75),
    },

    # 24. B4_Regime_BB_Width (63 bots) — s_regime_bb_width
    # Default: bb_width(20,2), rolling(100) percentile, squeeze<10%, overext>90%
    'B4_Regime_BB_Width': {
        'bb_period': ('int', 10, 40),
        'bb_std': ('float', 1.0, 3.0),
        'percentile_lookback': ('int', 50, 200),
        'squeeze_pctile': ('int', 5, 20),
        'overext_pctile': ('int', 80, 95),
    },

    # 26. B4_Oversold_3pct (57 bots) — s_oversold_3pct_1bar
    # Default: 1-bar drop < -3% buy, 1-bar gain > 2% sell
    'B4_Oversold_3pct': {
        'buy_drop_pct': ('float', -6.0, -1.5),
        'sell_gain_pct': ('float', 1.0, 4.0),
    },

    # 28. B4_BB_RSI_Stoch_Vol (54 bots) — s_bb_rsi_stoch_vol
    # Default: bb(20,2), rsi(14)<30, stoch(14,3)<20, volume>sma(volume,20)
    'B4_BB_RSI_Stoch_Vol': {
        'bb_period': ('int', 10, 40),
        'bb_std': ('float', 1.0, 3.0),
        'rsi_period': ('int', 7, 28),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 85),
        'stoch_period': ('int', 7, 28),
        'stoch_smooth': ('int', 2, 5),
        'stoch_buy_thresh': ('int', 10, 30),
        'vol_sma_period': ('int', 10, 40),
    },

    # 34. B4_Two_Bar_Reversal (49 bots) — s_two_bar_reversal
    # Default: rsi(14)<40 buy condition, rsi>70 sell
    'B4_Two_Bar_Reversal': {
        'rsi_period': ('int', 7, 28),
        'rsi_buy_thresh': ('int', 25, 50),
        'rsi_sell_thresh': ('int', 60, 85),
    },

    # 37. B4_Key_Reversal (48 bots) — s_key_reversal
    # Default: rsi(14)<40 buy, rsi>65 sell
    'B4_Key_Reversal': {
        'rsi_period': ('int', 7, 28),
        'rsi_buy_thresh': ('int', 25, 50),
        'rsi_sell_thresh': ('int', 55, 80),
    },

    # 41. B4_Oversold_7pct (41 bots) — s_oversold_7pct_3bar
    # Default: 3-bar drop < -7% buy, 3-bar gain > 3% sell
    'B4_Oversold_7pct': {
        'lookback_bars': ('int', 2, 6),
        'buy_drop_pct': ('float', -14.0, -4.0),
        'sell_gain_pct': ('float', 1.5, 6.0),
    },

    # 43. B4_RSI_25_45_55 (40 bots) — s_rsi_25_45_55
    # Default: rsi(25), rsi<45 buy, rsi>55 sell
    'B4_RSI_25_45_55': {
        'rsi_period': ('int', 12, 50),
        'rsi_buy_thresh': ('int', 30, 50),
        'rsi_sell_thresh': ('int', 50, 70),
    },

    # 46. B4_Spring_Wyckoff (39 bots) — s_spring_pattern
    # Default: low rolling(20).min(), break below + recovery
    'B4_Spring_Wyckoff': {
        'lookback': ('int', 10, 40),
    },

    # =========================================================================
    # BATCH 5 STRATEGIES
    # =========================================================================

    # 11. B5_MA_Envelope_3 (108 bots) — s_ma_envelope
    # Default: ema(20), upper=ema*1.03, lower=ema*0.97
    'B5_MA_Envelope_3': {
        'ema_period': ('int', 10, 40),
        'envelope_pct': ('float', 0.015, 0.06),
    },

    # 13. B5_Percentile_5 (105 bots) — s_percentile_5
    # Default: rolling(100), pct<5 buy, pct>80 sell
    'B5_Percentile_5': {
        'lookback': ('int', 50, 200),
        'buy_pctile': ('int', 2, 10),
        'sell_pctile': ('int', 70, 95),
    },

    # 19. B5_MeanRev_2Std (70 bots) — s_mean_rev_std
    # Default: sma(50), 2*std below buy, 1*std above sell
    'B5_MeanRev_2Std': {
        'sma_period': ('int', 25, 100),
        'buy_std_mult': ('float', 1.0, 3.0),
        'sell_std_mult': ('float', 0.5, 2.0),
    },

    # 30. B5_MA_Envelope_5 (52 bots) — s_ma_envelope_5pct
    # Default: ema(20), upper=ema*1.05, lower=ema*0.95
    'B5_MA_Envelope_5': {
        'ema_period': ('int', 10, 40),
        'envelope_pct': ('float', 0.025, 0.10),
    },

    # =========================================================================
    # BATCH 6 STRATEGIES
    # =========================================================================

    # 2. B6_VWAP_EMA (302 bots) — s_vwap_ema_combo
    # Default: vwap, ema(21), rsi(14), rsi<35 buy, rsi>60 sell
    'B6_VWAP_EMA': {
        'ema_period': ('int', 10, 42),
        'rsi_period': ('int', 7, 28),
        'rsi_buy_thresh': ('int', 20, 45),
        'rsi_sell_thresh': ('int', 50, 75),
    },

    # 3. B6_VWAP_BB (294 bots) — s_vwap_bb
    # Default: vwap, bb(20,2)
    'B6_VWAP_BB': {
        'bb_period': ('int', 10, 40),
        'bb_std': ('float', 1.0, 3.0),
    },

    # 4. B6_VWAP_Stoch (239 bots) — s_vwap_stoch
    # Default: vwap dist 0.002, stoch(14,3), k<20 buy, k>80 sell
    'B6_VWAP_Stoch': {
        'vwap_buy_dist': ('float', 0.001, 0.006),
        'vwap_sell_dist': ('float', 0.001, 0.006),
        'stoch_period': ('int', 7, 28),
        'stoch_smooth': ('int', 2, 5),
        'stoch_buy_thresh': ('int', 10, 30),
        'stoch_sell_thresh': ('int', 70, 90),
    },

    # 5. B6_Rubber_Band_3 (204 bots) — s_rubber_band_3
    # Default: ema(20), dist < -3% buy, dist > 0% sell
    'B6_Rubber_Band_3': {
        'ema_period': ('int', 10, 40),
        'buy_dist_pct': ('float', -6.0, -1.5),
        'sell_dist_pct': ('float', -1.0, 2.0),
    },

    # 8. B6_Rubber_Band_EMA50 (159 bots) — s_rubber_band_ema50
    # Default: ema(50), dist < -5% & rsi(14)<35 buy, dist>0 | rsi>65 sell
    'B6_Rubber_Band_EMA50': {
        'ema_period': ('int', 25, 100),
        'buy_dist_pct': ('float', -10.0, -2.5),
        'rsi_period': ('int', 7, 28),
        'rsi_buy_thresh': ('int', 20, 45),
        'sell_dist_pct': ('float', -1.0, 3.0),
        'rsi_sell_thresh': ('int', 55, 80),
    },

    # 14. B6_Rubber_Band_5 (95 bots) — s_rubber_band_5
    # Default: ema(20), dist < -5% buy, dist > 1% sell
    'B6_Rubber_Band_5': {
        'ema_period': ('int', 10, 40),
        'buy_dist_pct': ('float', -10.0, -2.5),
        'sell_dist_pct': ('float', 0.0, 3.0),
    },

    # 15. B6_Oversold_Extreme (88 bots) — s_oversold_extreme
    # Default: rsi(7)<15, 3 red candles, volume>1.5*sma(vol,20), rsi>55 sell
    'B6_Oversold_Extreme': {
        'rsi_period': ('int', 3, 14),
        'rsi_buy_thresh': ('int', 8, 25),
        'rsi_sell_thresh': ('int', 40, 70),
        'vol_sma_period': ('int', 10, 40),
        'vol_mult': ('float', 1.0, 3.0),
    },

    # 16. B6_Rubber_Band_7 (78 bots) — s_rubber_band_7
    # Default: ema(50), dist < -7% buy, dist > 2% sell
    'B6_Rubber_Band_7': {
        'ema_period': ('int', 25, 100),
        'buy_dist_pct': ('float', -14.0, -3.5),
        'sell_dist_pct': ('float', 0.5, 5.0),
    },

    # 20. B6_Midnight_Rev (68 bots) — s_midnight_reversal
    # Default: hour==0, rsi(7)<30 buy, rsi>65 sell
    'B6_Midnight_Rev': {
        'rsi_period': ('int', 3, 14),
        'rsi_buy_thresh': ('int', 15, 40),
        'rsi_sell_thresh': ('int', 55, 80),
    },

    # 22. B6_Oversold_Combo_Heavy (66 bots) — s_oversold_combo_heavy
    # Default: rsi(14)<25, bb(20,2) lower, vwap, stoch(14,3)<15, rsi>60 sell
    'B6_Oversold_Combo_Heavy': {
        'rsi_period': ('int', 7, 28),
        'rsi_buy_thresh': ('int', 15, 35),
        'rsi_sell_thresh': ('int', 50, 75),
        'bb_period': ('int', 10, 40),
        'bb_std': ('float', 1.0, 3.0),
        'stoch_period': ('int', 7, 28),
        'stoch_smooth': ('int', 2, 5),
        'stoch_buy_thresh': ('int', 5, 25),
    },

    # 27. B6_Vol_Climax (55 bots) — s_volume_climax
    # Default: vol rolling(20).max(), rsi(14)<35 buy, rsi>65 sell
    'B6_Vol_Climax': {
        'vol_lookback': ('int', 10, 40),
        'rsi_period': ('int', 7, 28),
        'rsi_buy_thresh': ('int', 20, 45),
        'rsi_sell_thresh': ('int', 55, 80),
    },

    # 29. B6_Oversold_Combo_Medium (52 bots) — s_oversold_combo_medium
    # Default: rsi(14)<30, bb(20,2) lower, volume>sma(vol,20), rsi>65 sell
    'B6_Oversold_Combo_Medium': {
        'rsi_period': ('int', 7, 28),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 55, 80),
        'bb_period': ('int', 10, 40),
        'bb_std': ('float', 1.0, 3.0),
        'vol_sma_period': ('int', 10, 40),
    },

    # 49. B6_VWAP_SuperTrend (38 bots) — s_vwap_supertrend
    # Default: vwap dist 0.002, supertrend(10,3), rsi(14)<40 buy, rsi>70 sell
    'B6_VWAP_SuperTrend': {
        'vwap_buy_dist': ('float', 0.001, 0.006),
        'st_period': ('int', 5, 20),
        'st_mult': ('float', 1.5, 5.0),
        'rsi_period': ('int', 7, 28),
        'rsi_buy_thresh': ('int', 25, 50),
        'rsi_sell_thresh': ('int', 60, 85),
    },

    # =========================================================================
    # BATCH 1 STRATEGIES
    # =========================================================================

    # 44. B1_CCI_100 (40 bots) — s_cci_100
    # Default: cci(20), cci < -100 buy, cci > 100 sell
    'B1_CCI_100': {
        'cci_period': ('int', 10, 40),
        'cci_buy_thresh': ('int', -200, -50),
        'cci_sell_thresh': ('int', 50, 200),
    },

    # =========================================================================
    # NEW BATCH STRATEGIES (100 additions)
    # =========================================================================

    # --- B4 ---

    # B4_ST_Stoch_BB (30 bots) — SuperTrend + Stoch + BB combo
    'B4_ST_Stoch_BB': {
        'st_period': ('int', 7, 14),
        'st_mult': ('float', 1.5, 4.0),
        'stoch_period': ('int', 7, 21),
        'stoch_smooth': ('int', 2, 5),
        'stoch_buy_thresh': ('int', 15, 30),
        'stoch_sell_thresh': ('int', 70, 85),
        'bb_period': ('int', 10, 30),
        'bb_std': ('float', 1.0, 3.0),
    },

    # B4_Price_EMA_Dist_7 (19 bots) — ema distance 7%
    'B4_Price_EMA_Dist_7': {
        'ema_period': ('int', 25, 100),
        'buy_dist_pct': ('float', -14.0, -3.5),
        'sell_dist_pct': ('float', 3.5, 14.0),
    },

    # B4_Exhaustion_Bar (17 bots) — exhaustion bar pattern + RSI
    'B4_Exhaustion_Bar': {
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
        'body_ratio': ('float', 0.3, 0.8),
        'vol_mult': ('float', 1.2, 3.0),
    },

    # B4_Dual_RSI (15 bots) — two RSI periods confirm
    'B4_Dual_RSI': {
        'rsi_fast_period': ('int', 3, 9),
        'rsi_slow_period': ('int', 14, 28),
        'rsi_fast_buy': ('int', 10, 30),
        'rsi_slow_buy': ('int', 25, 45),
        'rsi_fast_sell': ('int', 70, 90),
        'rsi_slow_sell': ('int', 55, 75),
    },

    # B4_Overbought_Rev (15 bots) — overbought reversal short
    'B4_Overbought_Rev': {
        'rsi_period': ('int', 7, 21),
        'rsi_overbought': ('int', 70, 90),
        'rsi_exit': ('int', 40, 60),
        'stoch_period': ('int', 7, 21),
        'stoch_overbought': ('int', 75, 90),
    },

    # B4_RSI_9_35_65 (14 bots) — RSI(9) with 35/65 thresholds
    'B4_RSI_9_35_65': {
        'rsi_period': ('int', 5, 18),
        'rsi_buy_thresh': ('int', 20, 45),
        'rsi_sell_thresh': ('int', 55, 80),
    },

    # B4_RSI_5_25_75 (9 bots) — RSI(5) with 25/75 thresholds
    'B4_RSI_5_25_75': {
        'rsi_period': ('int', 3, 10),
        'rsi_buy_thresh': ('int', 15, 35),
        'rsi_sell_thresh': ('int', 65, 85),
    },

    # B4_RSI_3_Cumulative (7 bots) — cumulative RSI(3)
    'B4_RSI_3_Cumulative': {
        'rsi_period': ('int', 2, 7),
        'cum_bars': ('int', 2, 5),
        'cum_buy_thresh': ('int', 20, 60),
        'cum_sell_thresh': ('int', 140, 280),
    },

    # B4_Seven_Day_Losing (5 bots) — 7-day losing streak bounce
    'B4_Seven_Day_Losing': {
        'streak_length': ('int', 5, 9),
        'bounce_pct': ('float', 1.0, 5.0),
    },

    # B4_Day_Of_Week (5 bots) — day-of-week seasonal
    'B4_Day_Of_Week': {
        'buy_day': ('int', 0, 6),
        'sell_day': ('int', 0, 6),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 25, 45),
    },

    # B4_ADX_BB_RSI (5 bots) — ADX + BB + RSI triple confirm
    'B4_ADX_BB_RSI': {
        'adx_period': ('int', 10, 20),
        'adx_thresh': ('int', 20, 35),
        'bb_period': ('int', 10, 30),
        'bb_std': ('float', 1.0, 3.0),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
    },

    # B4_Pivot_S3 (5 bots) — pivot S3 bounce
    'B4_Pivot_S3': {
        'bounce_pct': ('float', 0.1, 1.5),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 15, 35),
    },

    # B4_Six_Day_Losing (4 bots) — 6-day losing streak
    'B4_Six_Day_Losing': {
        'streak_length': ('int', 4, 8),
        'bounce_pct': ('float', 1.0, 5.0),
    },

    # B4_Pivot_Central (2 bots) — pivot central bounce
    'B4_Pivot_Central': {
        'bounce_pct': ('float', 0.1, 1.0),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 30, 50),
    },

    # B4_Regime_Slope (1 bot) — regime by slope direction
    'B4_Regime_Slope': {
        'slope_period': ('int', 10, 30),
        'slope_thresh': ('float', -0.5, -0.05),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
    },

    # B4_Regime_ATR (1 bot) — regime by ATR expansion
    'B4_Regime_ATR': {
        'atr_period': ('int', 7, 21),
        'atr_lookback': ('int', 20, 60),
        'atr_quantile': ('float', 0.7, 0.95),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
    },

    # B4_Thrust_Pattern (1 bot) — thrust/breadth pattern
    'B4_Thrust_Pattern': {
        'lookback': ('int', 5, 15),
        'thrust_pct': ('float', 2.0, 8.0),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 25, 45),
    },

    # B4_Hour_Of_Day (1 bot) — hour-of-day seasonal
    'B4_Hour_Of_Day': {
        'buy_hour': ('int', 0, 23),
        'sell_hour': ('int', 0, 23),
        'rsi_period': ('int', 7, 14),
        'rsi_buy_thresh': ('int', 25, 45),
    },

    # B4_Oversold_15pct (1 bot) — 15% oversold bounce
    'B4_Oversold_15pct': {
        'lookback_bars': ('int', 2, 6),
        'buy_drop_pct': ('float', -25.0, -8.0),
        'sell_gain_pct': ('float', 2.0, 10.0),
    },

    # --- B2 ---

    # B2_BB_PctB (28 bots) — BB %B indicator
    'B2_BB_PctB': {
        'bb_period': ('int', 10, 30),
        'bb_std': ('float', 1.0, 3.0),
        'pctb_buy': ('float', 0.0, 0.15),
        'pctb_sell': ('float', 0.85, 1.0),
    },

    # B2_ZScore_2 (28 bots) — z-score with 2.0 threshold
    'B2_ZScore_2': {
        'zscore_period': ('int', 10, 40),
        'buy_zscore': ('float', -3.5, -1.5),
        'sell_zscore': ('float', 1.5, 3.5),
    },

    # B2_BB_Stoch (25 bots) — BB + Stochastic combo
    'B2_BB_Stoch': {
        'bb_period': ('int', 10, 30),
        'bb_std': ('float', 1.0, 3.0),
        'stoch_period': ('int', 7, 21),
        'stoch_smooth': ('int', 2, 5),
        'stoch_buy_thresh': ('int', 10, 30),
        'stoch_sell_thresh': ('int', 70, 90),
    },

    # B2_Close_To_Low (17 bots) — close near daily low
    'B2_Close_To_Low': {
        'lookback': ('int', 10, 30),
        'close_low_pct': ('float', 0.5, 3.0),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
    },

    # B2_BB_3Std (16 bots) — BB with 3 std deviations
    'B2_BB_3Std': {
        'bb_period': ('int', 10, 40),
        'bb_std': ('float', 2.0, 4.0),
    },

    # B2_Oversold_10pct (16 bots) — 10% oversold bounce
    'B2_Oversold_10pct': {
        'lookback_bars': ('int', 2, 6),
        'buy_drop_pct': ('float', -18.0, -5.0),
        'sell_gain_pct': ('float', 2.0, 8.0),
    },

    # B2_Five_Day_Losing (4 bots) — 5-day losing streak
    'B2_Five_Day_Losing': {
        'streak_length': ('int', 3, 7),
        'bounce_pct': ('float', 1.0, 5.0),
    },

    # B2_Pivot_S2_Bounce (3 bots) — pivot S2 bounce
    'B2_Pivot_S2_Bounce': {
        'bounce_pct': ('float', 0.1, 1.5),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 15, 35),
    },

    # B2_Four_Day_Losing (1 bot) — 4-day losing streak
    'B2_Four_Day_Losing': {
        'streak_length': ('int', 3, 6),
        'bounce_pct': ('float', 1.0, 4.0),
    },

    # B2_NVI_Strategy (1 bot) — Negative Volume Index
    'B2_NVI_Strategy': {
        'nvi_sma_period': ('int', 50, 200),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 25, 45),
        'rsi_sell_thresh': ('int', 55, 75),
    },

    # B2_Three_Day_Losing (1 bot) — 3-day losing streak
    'B2_Three_Day_Losing': {
        'streak_length': ('int', 2, 5),
        'bounce_pct': ('float', 0.5, 3.0),
    },

    # --- B1 ---

    # B1_RSI_Divergence (26 bots) — RSI divergence detection
    'B1_RSI_Divergence': {
        'rsi_period': ('int', 7, 21),
        'lookback': ('int', 10, 30),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
    },

    # B1_Williams_90 (26 bots) — Williams %R with -90/-10
    'B1_Williams_90': {
        'williams_period': ('int', 7, 21),
        'oversold': ('int', -95, -80),
        'overbought': ('int', -20, -5),
    },

    # B1_RSI_21_40_60 (24 bots) — RSI(21) with 40/60 thresholds
    'B1_RSI_21_40_60': {
        'rsi_period': ('int', 10, 42),
        'rsi_buy_thresh': ('int', 25, 50),
        'rsi_sell_thresh': ('int', 50, 75),
    },

    # B1_CCI_200 (23 bots) — CCI with 200 threshold
    'B1_CCI_200': {
        'cci_period': ('int', 10, 40),
        'cci_buy_thresh': ('int', -300, -100),
        'cci_sell_thresh': ('int', 100, 300),
    },

    # B1_Williams_80 (22 bots) — Williams %R with -80/-20
    'B1_Williams_80': {
        'williams_period': ('int', 7, 21),
        'oversold': ('int', -90, -70),
        'overbought': ('int', -30, -10),
    },

    # B1_RSI_14_30_70 (21 bots) — standard RSI(14) 30/70
    'B1_RSI_14_30_70': {
        'rsi_period': ('int', 7, 28),
        'rsi_buy_thresh': ('int', 15, 40),
        'rsi_sell_thresh': ('int', 60, 85),
    },

    # B1_Stoch_14_3 (21 bots) — Stochastic(14,3) standard
    'B1_Stoch_14_3': {
        'stoch_period': ('int', 7, 28),
        'stoch_smooth': ('int', 2, 5),
        'stoch_buy_thresh': ('int', 10, 30),
        'stoch_sell_thresh': ('int', 70, 90),
    },

    # B1_RSI_Range (21 bots) — RSI range-bound mean reversion
    'B1_RSI_Range': {
        'rsi_period': ('int', 7, 28),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
    },

    # B1_MFI_20_80 (20 bots) — MFI(14) with 20/80
    'B1_MFI_20_80': {
        'mfi_period': ('int', 7, 28),
        'mfi_buy_thresh': ('int', 10, 30),
        'mfi_sell_thresh': ('int', 70, 90),
    },

    # B1_RSI_7_20_80 (20 bots) — RSI(7) with 20/80
    'B1_RSI_7_20_80': {
        'rsi_period': ('int', 3, 14),
        'rsi_buy_thresh': ('int', 10, 30),
        'rsi_sell_thresh': ('int', 70, 90),
    },

    # B1_Stoch_21_7 (16 bots) — Stochastic(21,7)
    'B1_Stoch_21_7': {
        'stoch_period': ('int', 14, 35),
        'stoch_smooth': ('int', 3, 10),
        'stoch_buy_thresh': ('int', 10, 30),
        'stoch_sell_thresh': ('int', 70, 90),
    },

    # B1_Stoch_5_3 (14 bots) — fast Stochastic(5,3)
    'B1_Stoch_5_3': {
        'stoch_period': ('int', 3, 10),
        'stoch_smooth': ('int', 2, 5),
        'stoch_buy_thresh': ('int', 10, 30),
        'stoch_sell_thresh': ('int', 70, 90),
    },

    # B1_MFI_10_90 (10 bots) — MFI with extreme 10/90
    'B1_MFI_10_90': {
        'mfi_period': ('int', 7, 21),
        'mfi_buy_thresh': ('int', 5, 20),
        'mfi_sell_thresh': ('int', 80, 95),
    },

    # B1_RSI2_SMA200 (6 bots) — RSI(2) + SMA(200) filter
    'B1_RSI2_SMA200': {
        'rsi_period': ('int', 2, 5),
        'sma_period': ('int', 100, 250),
        'rsi_buy_thresh': ('int', 3, 15),
        'rsi_sell_thresh': ('int', 85, 97),
    },

    # B1_MACD_RSI (6 bots) — MACD + RSI confirm
    'B1_MACD_RSI': {
        'macd_fast': ('int', 8, 16),
        'macd_slow': ('int', 20, 30),
        'macd_signal': ('int', 7, 12),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
    },

    # B1_UO_30_70 (5 bots) — Ultimate Oscillator 30/70
    'B1_UO_30_70': {
        'uo_short': ('int', 5, 10),
        'uo_medium': ('int', 10, 18),
        'uo_long': ('int', 20, 35),
        'uo_buy_thresh': ('int', 20, 40),
        'uo_sell_thresh': ('int', 60, 80),
    },

    # B1_StochRSI_14 (4 bots) — StochRSI(14)
    'B1_StochRSI_14': {
        'rsi_period': ('int', 7, 21),
        'stoch_period': ('int', 7, 21),
        'stoch_buy_thresh': ('float', 0.05, 0.25),
        'stoch_sell_thresh': ('float', 0.75, 0.95),
    },

    # B1_SuperTrend_RSI (3 bots) — SuperTrend + RSI
    'B1_SuperTrend_RSI': {
        'st_period': ('int', 7, 14),
        'st_mult': ('float', 1.5, 4.0),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
    },

    # B1_RSI_2_5_95 (3 bots) — RSI(2) extreme 5/95
    'B1_RSI_2_5_95': {
        'rsi_period': ('int', 2, 5),
        'rsi_buy_thresh': ('int', 2, 10),
        'rsi_sell_thresh': ('int', 90, 98),
    },

    # B1_RSI_2_10_90 (3 bots) — RSI(2) with 10/90
    'B1_RSI_2_10_90': {
        'rsi_period': ('int', 2, 5),
        'rsi_buy_thresh': ('int', 5, 15),
        'rsi_sell_thresh': ('int', 85, 95),
    },

    # B1_Stoch_Double (3 bots) — double stochastic confirm
    'B1_Stoch_Double': {
        'stoch_fast_period': ('int', 3, 10),
        'stoch_slow_period': ('int', 10, 21),
        'stoch_smooth': ('int', 2, 5),
        'stoch_buy_thresh': ('int', 10, 25),
        'stoch_sell_thresh': ('int', 75, 90),
    },

    # B1_Mass_Index (2 bots) — mass index reversal
    'B1_Mass_Index': {
        'ema_period': ('int', 7, 14),
        'mass_period': ('int', 20, 30),
        'mass_thresh': ('float', 25.0, 28.0),
    },

    # B1_StochRSI_MACD (1 bot) — StochRSI + MACD
    'B1_StochRSI_MACD': {
        'rsi_period': ('int', 7, 21),
        'stoch_period': ('int', 7, 21),
        'stoch_buy_thresh': ('float', 0.05, 0.25),
        'stoch_sell_thresh': ('float', 0.75, 0.95),
        'macd_fast': ('int', 8, 16),
        'macd_slow': ('int', 20, 30),
        'macd_signal': ('int', 7, 12),
    },

    # B1_SMA_20_50 (1 bot) — SMA crossover 20/50
    'B1_SMA_20_50': {
        'sma_fast': ('int', 10, 30),
        'sma_slow': ('int', 35, 70),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 30, 50),
    },

    # B1_MACD_Hist_Reversal (1 bot) — MACD histogram reversal
    'B1_MACD_Hist_Reversal': {
        'macd_fast': ('int', 8, 16),
        'macd_slow': ('int', 20, 30),
        'macd_signal': ('int', 7, 12),
        'hist_bars': ('int', 2, 5),
    },

    # B1_SuperTrend_7_2 (1 bot) — SuperTrend(7,2)
    'B1_SuperTrend_7_2': {
        'st_period': ('int', 5, 14),
        'st_mult': ('float', 1.0, 3.5),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 25, 45),
    },

    # --- B6 ---

    # B6_Scalp_Doji (26 bots) — doji scalp reversal
    'B6_Scalp_Doji': {
        'body_ratio': ('float', 0.02, 0.15),
        'rsi_period': ('int', 5, 14),
        'rsi_buy_thresh': ('int', 15, 35),
        'rsi_sell_thresh': ('int', 55, 80),
    },

    # B6_Oversold_Combo_Light (24 bots) — light oversold combo
    'B6_Oversold_Combo_Light': {
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 55, 80),
        'bb_period': ('int', 10, 30),
        'bb_std': ('float', 1.0, 3.0),
    },

    # B6_VWAP_Extreme (20 bots) — extreme VWAP deviation
    'B6_VWAP_Extreme': {
        'vwap_buy_dist': ('float', 0.003, 0.015),
        'vwap_sell_dist': ('float', 0.003, 0.015),
        'rsi_period': ('int', 5, 14),
        'rsi_buy_thresh': ('int', 10, 30),
    },

    # B6_Downtrend_Rally (18 bots) — downtrend bounce
    'B6_Downtrend_Rally': {
        'ema_period': ('int', 15, 50),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 55, 75),
        'bounce_pct': ('float', 1.0, 5.0),
    },

    # B6_Scalp_BB_7 (17 bots) — BB(7) scalping
    'B6_Scalp_BB_7': {
        'bb_period': ('int', 5, 14),
        'bb_std': ('float', 1.0, 2.5),
        'rsi_period': ('int', 3, 10),
        'rsi_buy_thresh': ('int', 15, 35),
    },

    # B6_Vol_Divergence (15 bots) — volume divergence
    'B6_Vol_Divergence': {
        'vol_sma_period': ('int', 10, 30),
        'vol_mult': ('float', 1.2, 3.0),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
        'lookback': ('int', 5, 15),
    },

    # B6_Scalp_VWAP_RSI (15 bots) — VWAP + RSI scalping
    'B6_Scalp_VWAP_RSI': {
        'vwap_buy_dist': ('float', 0.001, 0.008),
        'rsi_period': ('int', 3, 10),
        'rsi_buy_thresh': ('int', 15, 35),
        'rsi_sell_thresh': ('int', 55, 80),
    },

    # B6_Trend_Dip_RSI (13 bots) — trend dip buy RSI confirm
    'B6_Trend_Dip_RSI': {
        'ema_period': ('int', 20, 60),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 25, 45),
        'rsi_sell_thresh': ('int', 60, 80),
        'dip_pct': ('float', 1.0, 5.0),
    },

    # B6_Trend_Dip_BB (9 bots) — trend dip buy BB confirm
    'B6_Trend_Dip_BB': {
        'ema_period': ('int', 20, 60),
        'bb_period': ('int', 10, 30),
        'bb_std': ('float', 1.0, 3.0),
        'dip_pct': ('float', 1.0, 5.0),
    },

    # B6_Asian_Session (8 bots) — Asian session reversal
    'B6_Asian_Session': {
        'rsi_period': ('int', 5, 14),
        'rsi_buy_thresh': ('int', 15, 35),
        'rsi_sell_thresh': ('int', 60, 80),
        'session_start_hour': ('int', 0, 4),
        'session_end_hour': ('int', 6, 10),
    },

    # B6_Weekend_Effect (4 bots) — weekend effect
    'B6_Weekend_Effect': {
        'rsi_period': ('int', 7, 14),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
    },

    # --- B5 ---

    # B5_Fair_Value_Gap (25 bots) — FVG detection
    'B5_Fair_Value_Gap': {
        'lookback': ('int', 5, 20),
        'min_gap_pct': ('float', 0.3, 2.0),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 45),
    },

    # B5_Variance_Ratio (25 bots) — variance ratio test
    'B5_Variance_Ratio': {
        'short_period': ('int', 5, 15),
        'long_period': ('int', 20, 60),
        'vr_buy_thresh': ('float', 0.3, 0.8),
        'vr_sell_thresh': ('float', 1.2, 2.0),
    },

    # B5_VWAP_2Std (24 bots) — VWAP 2-std band bounce
    'B5_VWAP_2Std': {
        'vwap_std_mult': ('float', 1.0, 3.0),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
    },

    # B5_Normalized_Composite (24 bots) — normalized composite score
    'B5_Normalized_Composite': {
        'rsi_period': ('int', 7, 21),
        'rsi_weight': ('float', 0.1, 0.5),
        'stoch_period': ('int', 7, 21),
        'stoch_weight': ('float', 0.1, 0.5),
        'bb_period': ('int', 10, 30),
        'composite_buy_thresh': ('float', -0.8, -0.3),
        'composite_sell_thresh': ('float', 0.3, 0.8),
    },

    # B5_Body_Ratio (24 bots) — body/range ratio analysis
    'B5_Body_Ratio': {
        'body_ratio_thresh': ('float', 0.1, 0.4),
        'lookback': ('int', 3, 10),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
    },

    # B5_LogReturn_Extreme (22 bots) — log return extremes
    'B5_LogReturn_Extreme': {
        'lookback': ('int', 10, 30),
        'buy_pctile': ('int', 1, 8),
        'sell_pctile': ('int', 92, 99),
    },

    # B5_Wick_Rejection (21 bots) — wick rejection pattern
    'B5_Wick_Rejection': {
        'wick_ratio': ('float', 1.5, 4.0),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
    },

    # B5_ZScore_3 (16 bots) — z-score with 3.0 threshold
    'B5_ZScore_3': {
        'zscore_period': ('int', 10, 40),
        'buy_zscore': ('float', -4.0, -2.0),
        'sell_zscore': ('float', 2.0, 4.0),
    },

    # B5_Multi_Feature_Score (15 bots) — multi-feature composite
    'B5_Multi_Feature_Score': {
        'rsi_period': ('int', 7, 21),
        'bb_period': ('int', 10, 30),
        'stoch_period': ('int', 7, 21),
        'score_buy_thresh': ('float', -0.8, -0.3),
        'score_sell_thresh': ('float', 0.3, 0.8),
    },

    # B5_IBS_Strategy (9 bots) — Internal Bar Strength
    'B5_IBS_Strategy': {
        'ibs_buy_thresh': ('float', 0.05, 0.25),
        'ibs_sell_thresh': ('float', 0.75, 0.95),
        'lookback': ('int', 1, 5),
    },

    # B5_Weighted_Score (8 bots) — weighted indicator score
    'B5_Weighted_Score': {
        'rsi_period': ('int', 7, 21),
        'rsi_weight': ('float', 0.1, 0.5),
        'stoch_period': ('int', 7, 21),
        'stoch_weight': ('float', 0.1, 0.5),
        'score_buy_thresh': ('float', -0.8, -0.3),
        'score_sell_thresh': ('float', 0.3, 0.8),
    },

    # --- B3 ---

    # B3_ShootingStar (25 bots) — shooting star pattern + RSI
    'B3_ShootingStar': {
        'wick_ratio': ('float', 1.5, 4.0),
        'rsi_period': ('int', 7, 21),
        'rsi_sell_thresh': ('int', 60, 85),
        'rsi_exit': ('int', 30, 50),
    },

    # B3_Engulfing_RSI (22 bots) — engulfing candle + RSI
    'B3_Engulfing_RSI': {
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
    },

    # B3_HA_RSI (22 bots) — Heikin Ashi + RSI
    'B3_HA_RSI': {
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
        'ha_confirm_bars': ('int', 1, 4),
    },

    # B3_Double_Bottom (21 bots) — double bottom pattern
    'B3_Double_Bottom': {
        'lookback': ('int', 10, 40),
        'tolerance_pct': ('float', 0.5, 3.0),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 25, 45),
    },

    # B3_Scalp_RSI7 (20 bots) — RSI(7) scalping
    'B3_Scalp_RSI7': {
        'rsi_period': ('int', 3, 10),
        'rsi_buy_thresh': ('int', 10, 30),
        'rsi_sell_thresh': ('int', 65, 85),
    },

    # B3_Doji_Reversal (20 bots) — doji pattern reversal
    'B3_Doji_Reversal': {
        'body_ratio': ('float', 0.02, 0.15),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
    },

    # B3_Scalp_RSI_Stoch (19 bots) — RSI + Stoch scalping
    'B3_Scalp_RSI_Stoch': {
        'rsi_period': ('int', 3, 10),
        'rsi_buy_thresh': ('int', 10, 30),
        'rsi_sell_thresh': ('int', 65, 85),
        'stoch_period': ('int', 3, 10),
        'stoch_smooth': ('int', 2, 4),
        'stoch_buy_thresh': ('int', 10, 25),
        'stoch_sell_thresh': ('int', 75, 90),
    },

    # B3_Scalp_Stoch5 (15 bots) — fast Stoch(5) scalping
    'B3_Scalp_Stoch5': {
        'stoch_period': ('int', 3, 8),
        'stoch_smooth': ('int', 2, 4),
        'stoch_buy_thresh': ('int', 5, 20),
        'stoch_sell_thresh': ('int', 80, 95),
    },

    # B3_Scalp_BB_Tight (14 bots) — tight BB scalping
    'B3_Scalp_BB_Tight': {
        'bb_period': ('int', 5, 15),
        'bb_std': ('float', 0.5, 2.0),
        'rsi_period': ('int', 3, 10),
        'rsi_buy_thresh': ('int', 15, 35),
    },

    # B3_Keltner_RSI_MACD (11 bots) — Keltner + RSI + MACD
    'B3_Keltner_RSI_MACD': {
        'kc_period': ('int', 10, 30),
        'kc_mult': ('float', 1.0, 3.0),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
        'macd_fast': ('int', 8, 16),
        'macd_slow': ('int', 20, 30),
        'macd_signal': ('int', 7, 12),
    },

    # B3_OBV_RSI (8 bots) — On-Balance Volume + RSI
    'B3_OBV_RSI': {
        'obv_sma_period': ('int', 10, 30),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
    },

    # B3_Support_Resistance (7 bots) — S/R level bounce
    'B3_Support_Resistance': {
        'lookback': ('int', 20, 60),
        'tolerance_pct': ('float', 0.3, 2.0),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 25, 45),
    },

    # B3_MACD_StochRSI (4 bots) — MACD + StochRSI
    'B3_MACD_StochRSI': {
        'macd_fast': ('int', 8, 16),
        'macd_slow': ('int', 20, 30),
        'macd_signal': ('int', 7, 12),
        'rsi_period': ('int', 7, 21),
        'stoch_period': ('int', 7, 21),
        'stoch_buy_thresh': ('float', 0.05, 0.25),
        'stoch_sell_thresh': ('float', 0.75, 0.95),
    },

    # B3_Scalp_RSI3 (4 bots) — ultra-fast RSI(3) scalp
    'B3_Scalp_RSI3': {
        'rsi_period': ('int', 2, 5),
        'rsi_buy_thresh': ('int', 5, 20),
        'rsi_sell_thresh': ('int', 75, 95),
    },

    # B3_Triple_Confirm (1 bot) — triple indicator confirm
    'B3_Triple_Confirm': {
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
        'stoch_period': ('int', 7, 21),
        'stoch_buy_thresh': ('int', 10, 30),
        'bb_period': ('int', 10, 30),
        'bb_std': ('float', 1.0, 3.0),
    },

    # B3_Momentum_Surge (1 bot) — momentum surge entry
    'B3_Momentum_Surge': {
        'mom_period': ('int', 7, 21),
        'mom_thresh': ('float', 1.5, 5.0),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 30, 50),
    },

    # B3_EMA_RSI_Vol (1 bot) — EMA + RSI + Volume
    'B3_EMA_RSI_Vol': {
        'ema_period': ('int', 10, 30),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
        'vol_sma_period': ('int', 10, 30),
        'vol_mult': ('float', 1.0, 2.5),
    },

    # B3_Trend_Pullback (1 bot) — trend pullback entry
    'B3_Trend_Pullback': {
        'ema_fast': ('int', 10, 25),
        'ema_slow': ('int', 30, 60),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 30, 50),
        'pullback_pct': ('float', 1.0, 4.0),
    },

    # B3_MorningStar (1 bot) — morning star pattern
    'B3_MorningStar': {
        'body_ratio': ('float', 0.3, 0.7),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
    },

    # --- Non-batch prefix strategies ---

    # macd_rsi_divergence (21 bots) — MACD + RSI divergence
    'macd_rsi_divergence': {
        'macd_fast': ('int', 8, 16),
        'macd_slow': ('int', 20, 30),
        'macd_signal': ('int', 7, 12),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
        'divergence_lookback': ('int', 5, 15),
    },

    # morning_evening_star (4 bots) — morning/evening star pattern
    'morning_evening_star': {
        'body_ratio': ('float', 0.3, 0.7),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
    },

    # hammer_shooting_star (1 bot) — hammer + shooting star pattern
    'hammer_shooting_star': {
        'wick_ratio': ('float', 1.5, 4.0),
        'body_ratio': ('float', 0.1, 0.4),
        'rsi_period': ('int', 7, 21),
        'rsi_buy_thresh': ('int', 20, 40),
        'rsi_sell_thresh': ('int', 60, 80),
    },
}


def make_space_func(strategy_name):
    """Create an Optuna space function for a batch strategy.

    Returns a callable that takes an Optuna trial and returns a dict of
    suggested parameter values.

    Usage:
        space_fn = make_space_func('B2_VWAP_Bounce')
        params = space_fn(trial)  # {'rsi_period': 14, 'vwap_buy_dist': 0.003, ...}
    """
    params = BATCH_SEARCH_SPACES.get(strategy_name)
    if not params:
        return lambda trial: {}

    def space(trial):
        result = {}
        for name, spec in params.items():
            if spec[0] == 'int':
                result[name] = trial.suggest_int(name, spec[1], spec[2])
            elif spec[0] == 'float':
                result[name] = trial.suggest_float(name, spec[1], spec[2])
        return result
    return space


def get_default_params(strategy_name):
    """Return the hardcoded default params for a batch strategy.

    Useful for seeding the first Optuna trial with the original values.
    """
    defaults = {
        'B2_VWAP_Bounce': {'rsi_period': 14, 'vwap_buy_dist': 0.002, 'vwap_sell_dist': 0.002, 'rsi_buy_thresh': 40, 'rsi_sell_thresh': 60},
        'B6_VWAP_EMA': {'ema_period': 21, 'rsi_period': 14, 'rsi_buy_thresh': 35, 'rsi_sell_thresh': 60},
        'B6_VWAP_BB': {'bb_period': 20, 'bb_std': 2.0},
        'B6_VWAP_Stoch': {'vwap_buy_dist': 0.002, 'vwap_sell_dist': 0.002, 'stoch_period': 14, 'stoch_smooth': 3, 'stoch_buy_thresh': 20, 'stoch_sell_thresh': 80},
        'B6_Rubber_Band_3': {'ema_period': 20, 'buy_dist_pct': -3.0, 'sell_dist_pct': 0.0},
        'B4_Price_EMA_Dist_3': {'ema_period': 50, 'buy_dist_pct': -3.0, 'sell_dist_pct': 3.0},
        'B2_Rubber_Band': {'ema_period': 20, 'buy_dist_pct': -4.0, 'sell_dist_pct': 0.0},
        'B6_Rubber_Band_EMA50': {'ema_period': 50, 'buy_dist_pct': -5.0, 'rsi_period': 14, 'rsi_buy_thresh': 35, 'sell_dist_pct': 0.0, 'rsi_sell_thresh': 65},
        'B2_MeanRev_EMA21': {'ema_period': 21, 'buy_pct': -3.0, 'sell_pct': 3.0},
        'B2_PctRank_Reversal': {'lookback': 100, 'buy_thresh': 10, 'sell_thresh': 90},
        'B5_MA_Envelope_3': {'ema_period': 20, 'envelope_pct': 0.03},
        'B4_HVol_MeanRev': {'hv_period': 20, 'hv_lookback': 50, 'hv_quantile': 0.9, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 60},
        'B5_Percentile_5': {'lookback': 100, 'buy_pctile': 5, 'sell_pctile': 80},
        'B6_Rubber_Band_5': {'ema_period': 20, 'buy_dist_pct': -5.0, 'sell_dist_pct': 1.0},
        'B6_Oversold_Extreme': {'rsi_period': 7, 'rsi_buy_thresh': 15, 'rsi_sell_thresh': 55, 'vol_sma_period': 20, 'vol_mult': 1.5},
        'B6_Rubber_Band_7': {'ema_period': 50, 'buy_dist_pct': -7.0, 'sell_dist_pct': 2.0},
        'B2_Oversold_5pct': {'lookback_bars': 3, 'buy_drop_pct': -5.0, 'sell_gain_pct': 3.0},
        'B2_BB_50_2': {'bb_period': 50, 'bb_std': 2.0},
        'B5_MeanRev_2Std': {'sma_period': 50, 'buy_std_mult': 2.0, 'sell_std_mult': 1.0},
        'B6_Midnight_Rev': {'rsi_period': 7, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 65},
        'B3_ATR_Channel': {'ema_period': 20, 'atr_period': 14, 'atr_mult': 2.0},
        'B6_Oversold_Combo_Heavy': {'rsi_period': 14, 'rsi_buy_thresh': 25, 'rsi_sell_thresh': 60, 'bb_period': 20, 'bb_std': 2.0, 'stoch_period': 14, 'stoch_smooth': 3, 'stoch_buy_thresh': 15},
        'B2_Keltner_Bounce': {'kc_period': 20, 'kc_mult': 2.0},
        'B4_Regime_BB_Width': {'bb_period': 20, 'bb_std': 2.0, 'percentile_lookback': 100, 'squeeze_pctile': 10, 'overext_pctile': 90},
        'B2_MeanRev_SMA20': {'sma_period': 20, 'buy_pct': -5.0, 'sell_pct': 5.0},
        'B4_Oversold_3pct': {'buy_drop_pct': -3.0, 'sell_gain_pct': 2.0},
        'B6_Vol_Climax': {'vol_lookback': 20, 'rsi_period': 14, 'rsi_buy_thresh': 35, 'rsi_sell_thresh': 65},
        'B4_BB_RSI_Stoch_Vol': {'bb_period': 20, 'bb_std': 2.0, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70, 'stoch_period': 14, 'stoch_smooth': 3, 'stoch_buy_thresh': 20, 'vol_sma_period': 20},
        'B6_Oversold_Combo_Medium': {'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 65, 'bb_period': 20, 'bb_std': 2.0, 'vol_sma_period': 20},
        'B5_MA_Envelope_5': {'ema_period': 20, 'envelope_pct': 0.05},
        'B3_Hammer_Buy': {'rsi_period': 14, 'rsi_buy_thresh': 40, 'rsi_sell_thresh': 70},
        'B3_Multi_Osc': {'rsi_period': 14, 'rsi_buy_thresh': 30, 'stoch_period': 14, 'stoch_smooth': 3, 'stoch_buy_thresh': 20, 'cci_period': 20, 'cci_buy_thresh': -100, 'mfi_period': 14, 'mfi_buy_thresh': 20, 'rsi_sell_thresh': 70, 'stoch_sell_thresh': 80, 'cci_sell_thresh': 100},
        'B2_BB_MFI': {'bb_period': 20, 'bb_std': 2.0, 'mfi_period': 14, 'mfi_buy_thresh': 20, 'mfi_sell_thresh': 80},
        'B4_Two_Bar_Reversal': {'rsi_period': 14, 'rsi_buy_thresh': 40, 'rsi_sell_thresh': 70},
        'B2_BB_1Std': {'bb_period': 20, 'bb_std': 1.0},
        'B3_RSI_Stoch_BB': {'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70, 'stoch_period': 14, 'stoch_smooth': 3, 'stoch_buy_thresh': 20, 'stoch_sell_thresh': 80, 'bb_period': 20, 'bb_std': 2.0},
        'B4_Key_Reversal': {'rsi_period': 14, 'rsi_buy_thresh': 40, 'rsi_sell_thresh': 65},
        'B2_BB_RSI': {'bb_period': 20, 'bb_std': 2.0, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70},
        'B3_VWAP_RSI_MACD': {'vwap_buy_dist': 0.002, 'vwap_sell_dist': 0.002, 'rsi_period': 14, 'rsi_buy_thresh': 35, 'rsi_sell_thresh': 65},
        'B3_Harami_Reversal': {'rsi_period': 14, 'rsi_buy_thresh': 40, 'rsi_sell_thresh': 70, 'body_ratio': 0.5},
        'B4_Oversold_7pct': {'lookback_bars': 3, 'buy_drop_pct': -7.0, 'sell_gain_pct': 3.0},
        'B2_LinReg_MeanRev': {'lr_period': 20, 'buy_dist': 0.03, 'sell_dist': 0.03},
        'B4_RSI_25_45_55': {'rsi_period': 25, 'rsi_buy_thresh': 45, 'rsi_sell_thresh': 55},
        'B1_CCI_100': {'cci_period': 20, 'cci_buy_thresh': -100, 'cci_sell_thresh': 100},
        'B3_BB_ADX_RSI': {'bb_period': 20, 'bb_std': 2.0, 'adx_period': 14, 'adx_range_thresh': 25, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70},
        'B4_Spring_Wyckoff': {'lookback': 20},
        'B3_BB_SuperTrend': {'bb_period': 20, 'bb_std': 2.0, 'st_period': 10, 'st_mult': 3.0},
        'B2_ZScore_1_5': {'zscore_period': 20, 'buy_zscore': -1.5, 'sell_zscore': 1.5},
        'B6_VWAP_SuperTrend': {'vwap_buy_dist': 0.002, 'st_period': 10, 'st_mult': 3.0, 'rsi_period': 14, 'rsi_buy_thresh': 40, 'rsi_sell_thresh': 70},
        'B2_BB_Bounce': {'bb_period': 20, 'bb_std': 2.0},

        # --- NEW 100 strategies defaults ---

        # B4
        'B4_ST_Stoch_BB': {'st_period': 10, 'st_mult': 3.0, 'stoch_period': 14, 'stoch_smooth': 3, 'stoch_buy_thresh': 20, 'stoch_sell_thresh': 80, 'bb_period': 20, 'bb_std': 2.0},
        'B4_Price_EMA_Dist_7': {'ema_period': 50, 'buy_dist_pct': -7.0, 'sell_dist_pct': 7.0},
        'B4_Exhaustion_Bar': {'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70, 'body_ratio': 0.5, 'vol_mult': 2.0},
        'B4_Dual_RSI': {'rsi_fast_period': 5, 'rsi_slow_period': 21, 'rsi_fast_buy': 20, 'rsi_slow_buy': 35, 'rsi_fast_sell': 80, 'rsi_slow_sell': 65},
        'B4_Overbought_Rev': {'rsi_period': 14, 'rsi_overbought': 80, 'rsi_exit': 50, 'stoch_period': 14, 'stoch_overbought': 80},
        'B4_RSI_9_35_65': {'rsi_period': 9, 'rsi_buy_thresh': 35, 'rsi_sell_thresh': 65},
        'B4_RSI_5_25_75': {'rsi_period': 5, 'rsi_buy_thresh': 25, 'rsi_sell_thresh': 75},
        'B4_RSI_3_Cumulative': {'rsi_period': 3, 'cum_bars': 3, 'cum_buy_thresh': 40, 'cum_sell_thresh': 210},
        'B4_Seven_Day_Losing': {'streak_length': 7, 'bounce_pct': 2.0},
        'B4_Day_Of_Week': {'buy_day': 1, 'sell_day': 4, 'rsi_period': 14, 'rsi_buy_thresh': 35},
        'B4_ADX_BB_RSI': {'adx_period': 14, 'adx_thresh': 25, 'bb_period': 20, 'bb_std': 2.0, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70},
        'B4_Pivot_S3': {'bounce_pct': 0.5, 'rsi_period': 14, 'rsi_buy_thresh': 25},
        'B4_Six_Day_Losing': {'streak_length': 6, 'bounce_pct': 2.0},
        'B4_Pivot_Central': {'bounce_pct': 0.3, 'rsi_period': 14, 'rsi_buy_thresh': 40},
        'B4_Regime_Slope': {'slope_period': 20, 'slope_thresh': -0.2, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70},
        'B4_Regime_ATR': {'atr_period': 14, 'atr_lookback': 40, 'atr_quantile': 0.85, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70},
        'B4_Thrust_Pattern': {'lookback': 10, 'thrust_pct': 5.0, 'rsi_period': 14, 'rsi_buy_thresh': 35},
        'B4_Hour_Of_Day': {'buy_hour': 4, 'sell_hour': 16, 'rsi_period': 10, 'rsi_buy_thresh': 35},
        'B4_Oversold_15pct': {'lookback_bars': 3, 'buy_drop_pct': -15.0, 'sell_gain_pct': 5.0},

        # B2
        'B2_BB_PctB': {'bb_period': 20, 'bb_std': 2.0, 'pctb_buy': 0.05, 'pctb_sell': 0.95},
        'B2_ZScore_2': {'zscore_period': 20, 'buy_zscore': -2.0, 'sell_zscore': 2.0},
        'B2_BB_Stoch': {'bb_period': 20, 'bb_std': 2.0, 'stoch_period': 14, 'stoch_smooth': 3, 'stoch_buy_thresh': 20, 'stoch_sell_thresh': 80},
        'B2_Close_To_Low': {'lookback': 20, 'close_low_pct': 1.0, 'rsi_period': 14, 'rsi_buy_thresh': 30},
        'B2_BB_3Std': {'bb_period': 20, 'bb_std': 3.0},
        'B2_Oversold_10pct': {'lookback_bars': 3, 'buy_drop_pct': -10.0, 'sell_gain_pct': 4.0},
        'B2_Five_Day_Losing': {'streak_length': 5, 'bounce_pct': 2.0},
        'B2_Pivot_S2_Bounce': {'bounce_pct': 0.5, 'rsi_period': 14, 'rsi_buy_thresh': 25},
        'B2_Four_Day_Losing': {'streak_length': 4, 'bounce_pct': 2.0},
        'B2_NVI_Strategy': {'nvi_sma_period': 100, 'rsi_period': 14, 'rsi_buy_thresh': 35, 'rsi_sell_thresh': 65},
        'B2_Three_Day_Losing': {'streak_length': 3, 'bounce_pct': 1.5},

        # B1
        'B1_RSI_Divergence': {'rsi_period': 14, 'lookback': 20, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70},
        'B1_Williams_90': {'williams_period': 14, 'oversold': -90, 'overbought': -10},
        'B1_RSI_21_40_60': {'rsi_period': 21, 'rsi_buy_thresh': 40, 'rsi_sell_thresh': 60},
        'B1_CCI_200': {'cci_period': 20, 'cci_buy_thresh': -200, 'cci_sell_thresh': 200},
        'B1_Williams_80': {'williams_period': 14, 'oversold': -80, 'overbought': -20},
        'B1_RSI_14_30_70': {'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70},
        'B1_Stoch_14_3': {'stoch_period': 14, 'stoch_smooth': 3, 'stoch_buy_thresh': 20, 'stoch_sell_thresh': 80},
        'B1_RSI_Range': {'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70},
        'B1_MFI_20_80': {'mfi_period': 14, 'mfi_buy_thresh': 20, 'mfi_sell_thresh': 80},
        'B1_RSI_7_20_80': {'rsi_period': 7, 'rsi_buy_thresh': 20, 'rsi_sell_thresh': 80},
        'B1_Stoch_21_7': {'stoch_period': 21, 'stoch_smooth': 7, 'stoch_buy_thresh': 20, 'stoch_sell_thresh': 80},
        'B1_Stoch_5_3': {'stoch_period': 5, 'stoch_smooth': 3, 'stoch_buy_thresh': 20, 'stoch_sell_thresh': 80},
        'B1_MFI_10_90': {'mfi_period': 14, 'mfi_buy_thresh': 10, 'mfi_sell_thresh': 90},
        'B1_RSI2_SMA200': {'rsi_period': 2, 'sma_period': 200, 'rsi_buy_thresh': 5, 'rsi_sell_thresh': 95},
        'B1_MACD_RSI': {'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70},
        'B1_UO_30_70': {'uo_short': 7, 'uo_medium': 14, 'uo_long': 28, 'uo_buy_thresh': 30, 'uo_sell_thresh': 70},
        'B1_StochRSI_14': {'rsi_period': 14, 'stoch_period': 14, 'stoch_buy_thresh': 0.1, 'stoch_sell_thresh': 0.9},
        'B1_SuperTrend_RSI': {'st_period': 10, 'st_mult': 3.0, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70},
        'B1_RSI_2_5_95': {'rsi_period': 2, 'rsi_buy_thresh': 5, 'rsi_sell_thresh': 95},
        'B1_RSI_2_10_90': {'rsi_period': 2, 'rsi_buy_thresh': 10, 'rsi_sell_thresh': 90},
        'B1_Stoch_Double': {'stoch_fast_period': 5, 'stoch_slow_period': 14, 'stoch_smooth': 3, 'stoch_buy_thresh': 15, 'stoch_sell_thresh': 85},
        'B1_Mass_Index': {'ema_period': 9, 'mass_period': 25, 'mass_thresh': 27.0},
        'B1_StochRSI_MACD': {'rsi_period': 14, 'stoch_period': 14, 'stoch_buy_thresh': 0.1, 'stoch_sell_thresh': 0.9, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9},
        'B1_SMA_20_50': {'sma_fast': 20, 'sma_slow': 50, 'rsi_period': 14, 'rsi_buy_thresh': 40},
        'B1_MACD_Hist_Reversal': {'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'hist_bars': 3},
        'B1_SuperTrend_7_2': {'st_period': 7, 'st_mult': 2.0, 'rsi_period': 14, 'rsi_buy_thresh': 35},

        # B6
        'B6_Scalp_Doji': {'body_ratio': 0.05, 'rsi_period': 7, 'rsi_buy_thresh': 25, 'rsi_sell_thresh': 70},
        'B6_Oversold_Combo_Light': {'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 65, 'bb_period': 20, 'bb_std': 2.0},
        'B6_VWAP_Extreme': {'vwap_buy_dist': 0.008, 'vwap_sell_dist': 0.008, 'rsi_period': 7, 'rsi_buy_thresh': 20},
        'B6_Downtrend_Rally': {'ema_period': 30, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 65, 'bounce_pct': 2.5},
        'B6_Scalp_BB_7': {'bb_period': 7, 'bb_std': 1.5, 'rsi_period': 5, 'rsi_buy_thresh': 25},
        'B6_Vol_Divergence': {'vol_sma_period': 20, 'vol_mult': 2.0, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70, 'lookback': 10},
        'B6_Scalp_VWAP_RSI': {'vwap_buy_dist': 0.003, 'rsi_period': 5, 'rsi_buy_thresh': 25, 'rsi_sell_thresh': 70},
        'B6_Trend_Dip_RSI': {'ema_period': 40, 'rsi_period': 14, 'rsi_buy_thresh': 35, 'rsi_sell_thresh': 70, 'dip_pct': 2.5},
        'B6_Trend_Dip_BB': {'ema_period': 40, 'bb_period': 20, 'bb_std': 2.0, 'dip_pct': 2.5},
        'B6_Asian_Session': {'rsi_period': 7, 'rsi_buy_thresh': 25, 'rsi_sell_thresh': 70, 'session_start_hour': 1, 'session_end_hour': 8},
        'B6_Weekend_Effect': {'rsi_period': 10, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70},

        # B5
        'B5_Fair_Value_Gap': {'lookback': 10, 'min_gap_pct': 0.8, 'rsi_period': 14, 'rsi_buy_thresh': 35},
        'B5_Variance_Ratio': {'short_period': 10, 'long_period': 40, 'vr_buy_thresh': 0.5, 'vr_sell_thresh': 1.5},
        'B5_VWAP_2Std': {'vwap_std_mult': 2.0, 'rsi_period': 14, 'rsi_buy_thresh': 30},
        'B5_Normalized_Composite': {'rsi_period': 14, 'rsi_weight': 0.3, 'stoch_period': 14, 'stoch_weight': 0.3, 'bb_period': 20, 'composite_buy_thresh': -0.5, 'composite_sell_thresh': 0.5},
        'B5_Body_Ratio': {'body_ratio_thresh': 0.2, 'lookback': 5, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70},
        'B5_LogReturn_Extreme': {'lookback': 20, 'buy_pctile': 3, 'sell_pctile': 97},
        'B5_Wick_Rejection': {'wick_ratio': 2.5, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70},
        'B5_ZScore_3': {'zscore_period': 20, 'buy_zscore': -3.0, 'sell_zscore': 3.0},
        'B5_Multi_Feature_Score': {'rsi_period': 14, 'bb_period': 20, 'stoch_period': 14, 'score_buy_thresh': -0.5, 'score_sell_thresh': 0.5},
        'B5_IBS_Strategy': {'ibs_buy_thresh': 0.1, 'ibs_sell_thresh': 0.9, 'lookback': 1},
        'B5_Weighted_Score': {'rsi_period': 14, 'rsi_weight': 0.3, 'stoch_period': 14, 'stoch_weight': 0.3, 'score_buy_thresh': -0.5, 'score_sell_thresh': 0.5},

        # B3
        'B3_ShootingStar': {'wick_ratio': 2.5, 'rsi_period': 14, 'rsi_sell_thresh': 70, 'rsi_exit': 40},
        'B3_Engulfing_RSI': {'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70},
        'B3_HA_RSI': {'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70, 'ha_confirm_bars': 2},
        'B3_Double_Bottom': {'lookback': 20, 'tolerance_pct': 1.5, 'rsi_period': 14, 'rsi_buy_thresh': 35},
        'B3_Scalp_RSI7': {'rsi_period': 7, 'rsi_buy_thresh': 20, 'rsi_sell_thresh': 75},
        'B3_Doji_Reversal': {'body_ratio': 0.05, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70},
        'B3_Scalp_RSI_Stoch': {'rsi_period': 7, 'rsi_buy_thresh': 20, 'rsi_sell_thresh': 75, 'stoch_period': 5, 'stoch_smooth': 3, 'stoch_buy_thresh': 15, 'stoch_sell_thresh': 85},
        'B3_Scalp_Stoch5': {'stoch_period': 5, 'stoch_smooth': 3, 'stoch_buy_thresh': 10, 'stoch_sell_thresh': 90},
        'B3_Scalp_BB_Tight': {'bb_period': 10, 'bb_std': 1.0, 'rsi_period': 5, 'rsi_buy_thresh': 25},
        'B3_Keltner_RSI_MACD': {'kc_period': 20, 'kc_mult': 2.0, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9},
        'B3_OBV_RSI': {'obv_sma_period': 20, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70},
        'B3_Support_Resistance': {'lookback': 40, 'tolerance_pct': 1.0, 'rsi_period': 14, 'rsi_buy_thresh': 35},
        'B3_MACD_StochRSI': {'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'rsi_period': 14, 'stoch_period': 14, 'stoch_buy_thresh': 0.1, 'stoch_sell_thresh': 0.9},
        'B3_Scalp_RSI3': {'rsi_period': 3, 'rsi_buy_thresh': 10, 'rsi_sell_thresh': 85},
        'B3_Triple_Confirm': {'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70, 'stoch_period': 14, 'stoch_buy_thresh': 20, 'bb_period': 20, 'bb_std': 2.0},
        'B3_Momentum_Surge': {'mom_period': 14, 'mom_thresh': 3.0, 'rsi_period': 14, 'rsi_buy_thresh': 40},
        'B3_EMA_RSI_Vol': {'ema_period': 20, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70, 'vol_sma_period': 20, 'vol_mult': 1.5},
        'B3_Trend_Pullback': {'ema_fast': 15, 'ema_slow': 45, 'rsi_period': 14, 'rsi_buy_thresh': 40, 'pullback_pct': 2.0},
        'B3_MorningStar': {'body_ratio': 0.5, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70},

        # Non-batch prefix
        'macd_rsi_divergence': {'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70, 'divergence_lookback': 10},
        'morning_evening_star': {'body_ratio': 0.5, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70},
        'hammer_shooting_star': {'wick_ratio': 2.5, 'body_ratio': 0.2, 'rsi_period': 14, 'rsi_buy_thresh': 30, 'rsi_sell_thresh': 70},
    }
    return defaults.get(strategy_name, {})


# Convenience: list of all covered strategy names
ALL_BATCH_STRATEGIES = sorted(BATCH_SEARCH_SPACES.keys())

if __name__ == '__main__':
    print(f"Batch search spaces defined for {len(BATCH_SEARCH_SPACES)} strategies:")
    for name in ALL_BATCH_STRATEGIES:
        n_params = len(BATCH_SEARCH_SPACES[name])
        print(f"  {name}: {n_params} params")
