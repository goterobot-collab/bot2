#!/usr/bin/env python3
"""
strategies_batch_loader.py — Carga TODAS las estrategias de BATCH 1-7
=====================================================================
Las estrategias Optuna R3 usan nombres con prefijo B{n}_ que mapean
a funciones en strategies_batch[1-7].py del directorio Estrategias.

También incluye alias para nombres sin prefijo (VWAP, RSI, etc.)
que mapean a estrategias equivalentes en batch2_strategies.py.

Total: ~500+ estrategias batch + ~50 alias = cobertura completa.
"""
import sys
import os
import logging

logger = logging.getLogger(__name__)

BATCH_CODE_DIR = os.path.expanduser(
    "~/CLAUDE CODE/Estrategias/strategies_code"
)

# ── Name aliases: bot names → batch2_strategies names ──────────────
# For strategies referenced in bots without B-prefix
STRATEGY_ALIASES = {
    "VWAP": "VWMA",
    "triple_barrier_strategy": "Triple_Barrier",
    "RSI": "RSI_EMA_Filter",
    "ZScore": "MeanRev_ZScore",
    "EMA": "EMA_Ribbon",
    "price_action_sr": "Price_Action_SR",
    "BB": "BB_Squeeze_Breakout",
    "ou_mean_reversion": "OU_MeanReversion",
    "Stoch": "Stochastic_RSI",
    "WilliamsR": "Williams_R",
    "gjr_garch_strategy": "GJR_GARCH",
    "mean_reversion_zscore": "MeanRev_ZScore",
    "cmo_strategy": "CMO",
    "fourier_strategy": "Fourier_Analysis",
    "mfi_strategy": "MFI",
    "cointegration_strategy": "Cointegration_ZScore",
    "williams_r_strategy": "Williams_R",
    "percent_r_reversal": "Percent_R_Reversal",
    "macd_rsi_divergence": "MACD_RSI_Divergence",
    "td_sequential_simplified": "TD_Sequential",
    "betting_against_beta": "Betting_Against_Beta",
    "jump_diffusion_strategy": "Jump_Diffusion",
    "stochastic_rsi_strategy": "Stochastic_RSI",
    "connors_rsi_full": "Connors_RSI_Full",
    "schaff_trend_cycle": "Schaff_Trend_Cycle",
    "ADX": "ADX_Trend",
    "VolSpike": "Vol_Regime",
    "ehlers_modified_stochastic": "Ehlers_ModStoch",
    "rsi_bb_strategy": "RSI_BB",
    "Momentum": "Momentum_Clenow",
    "bb_squeeze_breakout": "BB_Squeeze_Breakout",
    "volume_breakout_strategy": "Volume_Breakout",
    "hammer_shooting_star": "Hammer_ShootingStar",
    "morning_evening_star": "Morning_Evening_Star",
    "chaikin_money_flow": "Chaikin_MF",
    "macd_histogram_divergence": "MACD_Histogram_Div",
    "trend_following_cta": "CTA_Trend",
    "pin_bar_strategy": "Pin_Bar",
    "momentum_factor_clenow": "Momentum_Clenow",
    "adx_trend_strategy": "ADX_Trend",
    "triple_ema_crossover": "Triple_EMA_Cross",
    "rsi_macd_bb_combo": "RSI_MACD_BB",
    "CCI": "CCI",
    "Donchian": "Keltner_Breakout",
    "HMA": "Hull_MA",
    "Ichimoku": "Ichimoku",
    "Keltner": "Keltner_Breakout",
    "Engulfing": "Engulfing_Pattern",
    "Squeeze": "Squeeze_Momentum",
    "TripleEMA": "Triple_EMA_Cross",
    "OBV": "OBV",
    "Doji": "Doji_Reversal",
    "Pivot": "Pivot_S2_Bounce",
    "ATR": "ATR_Channel",
    # TV strategies: map TV_ prefix to registered name
    "TV_Spike_Reversion_70": "Spike_Reversion_70",
}


def _signal_only_simulate(df, signals, max_hold=None):
    """Replacement for simulate() that returns the signal Series directly.

    Batch strategies (1-6) call simulate(df, sig) and return (trades, equity).
    signal_factory needs the raw signal Series (0/1/-1), not backtested trades.
    This patch makes simulate() return just the signals so _check_signal() works.
    """
    return signals


def _wrap_gen(func):
    """Wrap a strategy function into {'gen': func} format for STRATS dict."""
    return {"gen": func}


def load_batch_strategies() -> dict:
    """Load all batch strategies and return dict compatible with STRATS.

    Returns: {strategy_name: {'gen': function}, ...}
    """
    if not os.path.isdir(BATCH_CODE_DIR):
        logger.warning(f"Batch strategies dir not found: {BATCH_CODE_DIR}")
        return {}

    sys.path.insert(0, BATCH_CODE_DIR)
    result = {}

    # ── Load BATCH 1-6 with B{n}_ prefix ──────────────────────────
    # Monkey-patch simulate() in batch1 so ALL batch functions return
    # the raw signal Series instead of (trades, equity) tuples.
    # This works because batch functions look up 'simulate' in their
    # module __globals__ at call time, not at import time.
    patched_modules = set()
    for batch_num in range(1, 7):
        try:
            mod = __import__(f"strategies_batch{batch_num}")
            # Patch simulate() to return signals instead of (trades, equity)
            if hasattr(mod, 'simulate') and id(mod) not in patched_modules:
                mod.simulate = _signal_only_simulate
                patched_modules.add(id(mod))
            batch_dict = getattr(mod, f"BATCH{batch_num}_STRATEGIES", {})
            for name, func in batch_dict.items():
                prefixed = f"B{batch_num}_{name}"
                result[prefixed] = _wrap_gen(func)
                # Also register without prefix (for enriched grails names)
                if name not in result:
                    result[name] = _wrap_gen(func)
        except Exception as e:
            logger.warning(f"Failed to load BATCH{batch_num}: {e}")

    # ── Load BATCH 7 (special case) ─────────────────────────────────
    # Batch7 wraps batch2_strategies functions with signal_to_trades()
    # which returns (trades, equity). We use the UNWRAPPED originals
    # from batch2_strategies._ALL instead, which return signal Series.
    try:
        mod7 = __import__("strategies_batch7")
        # _ALL contains the original batch2 functions (return pd.Series)
        b7_originals = getattr(mod7, '_ALL', {})
        b7_keys = getattr(mod7, '_NEW_KEYS', [])
        b7_count = 0
        for name in b7_keys:
            if name in b7_originals:
                prefixed = f"B7_{name}"
                result[prefixed] = _wrap_gen(b7_originals[name])
                if name not in result:
                    result[name] = _wrap_gen(b7_originals[name])
                b7_count += 1
        logger.info(f"BATCH7 loaded: {b7_count} strategies (unwrapped originals)")
    except Exception as e:
        logger.warning(f"Failed to load BATCH7: {e}")

    # ── Load extended batch2 strategies ────────────────────────────
    ext_strategies = {}
    try:
        import batch2_strategies as b2ext
        ext_strategies = getattr(b2ext, "BATCH2_STRATEGIES", {})
        for name, func in ext_strategies.items():
            if name not in result:
                result[name] = _wrap_gen(func)
    except Exception as e:
        logger.warning(f"Failed to load batch2_strategies: {e}")

    # ── Apply aliases (bot names → batch2 names) ──────────────────
    alias_count = 0
    for alias, target in STRATEGY_ALIASES.items():
        if alias in result:
            continue  # Already loaded
        if target in ext_strategies:
            result[alias] = _wrap_gen(ext_strategies[target])
            alias_count += 1
        elif target in result:
            result[alias] = result[target]
            alias_count += 1
        else:
            logger.debug(f"Alias target not found: {alias} → {target}")

    # ── Load Round 2 strategies (R2 Optuna — 52 strategies) ───────
    # These are NOT in any batch file — they live in strategies_round2.py.
    # Without this block, 666 R2 bots (22 strategy types) show as "unknown"
    # in signal_factory._check_signal() and are silently skipped every cycle.
    # Affected: Zscore_Volume (107), Ultimate_Osc (87), Ulcer_Index (86),
    # BB_RSI_MACD (83), DEMA_Cross (60), Regression_Channel (57), TEMA_Cross (45),
    # Stoch_RSI (37), Renko_Trend (29), Harami (20), Chaikin_Vol (12),
    # Morning_Star (10), Hammer_Star (8), Choppiness (8), Three_Soldiers (5),
    # Keltner_RSI (3), CMF (3), Ichimoku_RSI (2), Historical_Vol (1),
    # ADX_EMA (1), Heikin_Ashi (1), BB_Width (1).
    r2_count = 0
    try:
        # strategies_round2.py is in the same directory as this file (BOT V7/strategies/)
        r2_dir = os.path.dirname(os.path.abspath(__file__))
        if r2_dir not in sys.path:
            sys.path.insert(0, r2_dir)
        from strategies_round2 import STRATEGY_TYPES_R2
        for name, entry in STRATEGY_TYPES_R2.items():
            if name not in result:
                # R2 entries are already {'gen': func, 'space': func} — use as-is
                result[name] = entry
                r2_count += 1
        logger.info(f"Round2 strategies registered: {r2_count} new entries")
    except Exception as e:
        logger.warning(f"Failed to load strategies_round2: {e}")

    logger.info(
        f"Batch strategies loaded: {len(result)} total "
        f"(7 batches + {alias_count} aliases + {r2_count} R2)"
    )
    return result
