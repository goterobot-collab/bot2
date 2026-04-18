"""
GRAIL DUAL VALIDATOR — Validación Dual de Estrategias
======================================================
Implementa el flujo de 4 pasos:
  Paso 1: Filtrar grails (test_wr>80%, n_trades>20)
  Paso 2: Agente A — Full History Backtest (réplica de optuna_v7, sin SL/TP)
  Paso 3: Agente B — Auditor Externo (SL/TP empírico, slippage noise, CIEGO)
  Paso 4: Decisión (Producción / Shadow / Descartar)

Uso:
  python3 grail_dual_validator.py                    # procesa top grails
  python3 grail_dual_validator.py --symbol AGT/USDT:USDT
  python3 grail_dual_validator.py --strategy RSI_VWAP
  python3 grail_dual_validator.py --min-wr 80 --min-trades 20 --top 50
  python3 grail_dual_validator.py --symbol BEAT/USDT:USDT --strategy B5_Momentum

Resultados: data/dual_validation_results.json  +  data/dual_validation_log.csv
"""

import sys, os, json, sqlite3, argparse, time
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(ROOT, "activos_binance.db")
GRAILS_PATH = os.path.join(ROOT, "data", "grails_master.json")
OUT_JSON    = os.path.join(ROOT, "data", "dual_validation_results.json")
OUT_CSV     = os.path.join(ROOT, "data", "dual_validation_log.csv")

# ─── Parámetros del validador ─────────────────────────────────────────────────
SLIPPAGE   = 0.0005
COMMISSION = 0.0004     # 0.02% × 2 puntas
MAX_BARS   = 72         # duración máxima en bars (igual que optuna_v7)
SL_MIN, SL_MAX = 0.02, 0.40
TP_MIN, TP_MAX = 0.01, 0.50

# Umbrales de decisión (Paso 4)
WR_PRODUCCION_A = 75.0  # Agente A mínimo para producción
WR_PRODUCCION_B = 70.0  # Agente B mínimo para producción
WR_SHADOW_MIN   = 50.0  # Por debajo de esto → descartar
SLIPPAGE_NOISE  = 0.001 # 0.1% desplazamiento de precio para test de robustez

# ─── Helpers comunes ──────────────────────────────────────────────────────────
def rsi(close, n=14):
    d = close.diff()
    g = d.clip(lower=0); l = (-d).clip(lower=0)
    rs = g.ewm(alpha=1/n, adjust=False).mean() / (l.ewm(alpha=1/n, adjust=False).mean() + 1e-10)
    return 100 - 100 / (1 + rs)

def vwap_daily(df):
    tp = (df['high'] + df['low'] + df['close']) / 3
    dates = df.index.date
    return (tp * df['volume']).groupby(dates).cumsum() / (df['volume'].groupby(dates).cumsum() + 1e-10)

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def stoch_k(high, low, close, k=14):
    return 100 * (close - low.rolling(k).min()) / (high.rolling(k).max() - low.rolling(k).min() + 1e-10)

def obv(close, volume):
    return (close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * volume).cumsum()

def zscore(close, lookback=50):
    m = close.rolling(lookback).mean()
    s = close.rolling(lookback).std()
    return (close - m) / (s + 1e-10)

def bb_bands(close, n=20, std=2.0):
    m = close.rolling(n).mean()
    s = close.rolling(n).std()
    return m, m + std * s, m - std * s

# ─── Implementaciones independientes de estrategias (Agente B) ───────────────
# IMPORTANTE: estas funciones NO importan de strategy_factory.py
# Son implementaciones ciegas para auditoría independiente.

STRATEGY_IMPLS = {}

def _reg(name, fn):
    STRATEGY_IMPLS[name] = fn

def _rsi_vwap(df, rsi_buy=30, dist_pct=0.01):
    r = rsi(df['close'], 14); vw = vwap_daily(df)
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (df['close'] < vw * (1 - dist_pct))] = 1
    sig[(r > 100 - rsi_buy) & (df['close'] > vw * (1 + dist_pct))] = -1
    return sig
_reg('RSI_VWAP', _rsi_vwap)

def _vwap_stoch(df, dist_pct=0.01, k_period=14, stoch_buy=20, stoch_sell=80):
    vw = vwap_daily(df); k = stoch_k(df['high'], df['low'], df['close'], k_period)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (k < stoch_buy)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (k > stoch_sell)] = -1
    return sig
_reg('VWAP_Stoch', _vwap_stoch)
_reg('VWAP_Stoch_Combo', _vwap_stoch)

def _vwap_obv(df, dist_pct=0.01, obv_period=20):
    vw = vwap_daily(df); o = obv(df['close'], df['volume']); os = o.rolling(obv_period).mean()
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (o > os)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (o < os)] = -1
    return sig
_reg('VWAP_OBV', _vwap_obv)

def _vwap_rsi(df, dist_pct=0.01, rsi_buy=30):
    vw = vwap_daily(df); r = rsi(df['close'], 14)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < vw * (1 - dist_pct)) & (r < rsi_buy)] = 1
    sig[(df['close'] > vw * (1 + dist_pct)) & (r > 100 - rsi_buy)] = -1
    return sig
_reg('VWAP_RSI', _vwap_rsi)
_reg('RSI_VWAP_Combo', _vwap_rsi)

def _vwap_basic(df, dist_pct=0.01):
    vw = vwap_daily(df)
    sig = pd.Series(0, index=df.index)
    sig[df['close'] < vw * (1 - dist_pct)] = 1
    sig[df['close'] > vw * (1 + dist_pct)] = -1
    return sig
_reg('VWAP', _vwap_basic)
_reg('VWAP_Basic', _vwap_basic)

def _zscore_dual(df, short_lb=20, long_lb=100, z_thresh=2.0):
    z_s = zscore(df['close'], short_lb); z_l = zscore(df['close'], long_lb)
    sig = pd.Series(0, index=df.index)
    sig[(z_s < -z_thresh) & (z_l < -z_thresh)] = 1
    sig[(z_s > z_thresh) & (z_l > z_thresh)] = -1
    return sig
_reg('ZScore_Dual', _zscore_dual)

def _rsi_divergence(df, rsi_period=14, lookback=14, rsi_thresh_low=30, rsi_thresh_high=70):
    r = rsi(df['close'], rsi_period)
    price_low  = df['close'].rolling(lookback).min()
    price_high = df['close'].rolling(lookback).max()
    rsi_low    = r.rolling(lookback).min()
    rsi_high   = r.rolling(lookback).max()
    sig = pd.Series(0, index=df.index)
    # Bullish divergence: precio new low pero RSI no hace new low
    bull = (df['close'] <= price_low.shift()) & (r > rsi_low.shift()) & (r < rsi_thresh_low + 10)
    bear = (df['close'] >= price_high.shift()) & (r < rsi_high.shift()) & (r > rsi_thresh_high - 10)
    sig[bull] = 1; sig[bear] = -1
    return sig
_reg('RSI_Divergence', _rsi_divergence)

def _bb_mean_rev(df, bb_period=20, bb_std=2.0):
    _, upper, lower = bb_bands(df['close'], bb_period, bb_std)
    sig = pd.Series(0, index=df.index)
    sig[df['close'] < lower] = 1
    sig[df['close'] > upper] = -1
    return sig
_reg('BB_MeanRev', _bb_mean_rev)
_reg('TV_BB_MeanRev_Simple', _bb_mean_rev)

def _ema_cross(df, fast=9, slow=21):
    e_fast = ema(df['close'], fast); e_slow = ema(df['close'], slow)
    sig = pd.Series(0, index=df.index)
    sig[(e_fast > e_slow) & (e_fast.shift() <= e_slow.shift())] = 1
    sig[(e_fast < e_slow) & (e_fast.shift() >= e_slow.shift())] = -1
    return sig
_reg('EMA', _ema_cross)
_reg('EMA_Cross', _ema_cross)

# ─── Helpers adicionales (para B5/B6) ────────────────────────────────────────

def sma(s, n):
    return s.rolling(n).mean()

def rolling_vwap(df, n=20):
    """VWAP rodante de período n (igual que vwap_calc en strategies_round3)."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    return (tp * df['volume']).rolling(n).sum() / (df['volume'].rolling(n).sum() + 1e-10)

def mfi(df, n=14):
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    d = tp.diff()
    pmf = mf.where(d > 0, 0).rolling(n).sum()
    nmf = mf.where(d <= 0, 0).rolling(n).sum()
    return 100 - 100 / (1 + pmf / (nmf + 1e-10))

def williams_r(df, n=14):
    hh = df['high'].rolling(n).max()
    ll = df['low'].rolling(n).min()
    return -100 * (hh - df['close']) / (hh - ll + 1e-10)

def ultimate_osc(df, p1=7, p2=14, p3=28):
    prev_c = df['close'].shift()
    tr = pd.concat([df['high'] - df['low'],
                    (df['high'] - prev_c).abs(),
                    (df['low'] - prev_c).abs()], axis=1).max(axis=1)
    bp = df['close'] - pd.concat([df['low'], prev_c], axis=1).min(axis=1)
    avg1 = bp.rolling(p1).sum() / (tr.rolling(p1).sum() + 1e-10)
    avg2 = bp.rolling(p2).sum() / (tr.rolling(p2).sum() + 1e-10)
    avg3 = bp.rolling(p3).sum() / (tr.rolling(p3).sum() + 1e-10)
    return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

def stoch_dp(df, kp=14, dp=3):
    lo = df['low'].rolling(kp).min()
    hi = df['high'].rolling(kp).max()
    k = 100 * (df['close'] - lo) / (hi - lo + 1e-10)
    return k, k.rolling(dp).mean()

# ─── B5 Strategies ────────────────────────────────────────────────────────────

def _b5_ma_envelope(df, ma_p=20, pct=3.0):
    """MA Envelope: compra cuando cruza por debajo de -pct%, vende al cruzar +pct%."""
    ma = sma(df['close'], ma_p)
    upper = ma * (1 + pct / 100)
    lower = ma * (1 - pct / 100)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (df['close'].shift() >= lower.shift())] = 1
    sig[(df['close'] > upper) & (df['close'].shift() <= upper.shift())] = -1
    return sig
_reg('B5_MA_Envelope_3', _b5_ma_envelope)
_reg('B5_MA_Envelope_5', _b5_ma_envelope)

def _b5_meanrev_2std(df, period=20):
    """Z-score 2 desvíos: compra z<-2, vende z>2."""
    m = df['close'].rolling(period).mean()
    s = df['close'].rolling(period).std()
    z = (df['close'] - m) / (s + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(z < -2) & (z.shift() >= -2)] = 1
    sig[(z > 2) & (z.shift() <= 2)] = -1
    return sig
_reg('B5_MeanRev_2Std', _b5_meanrev_2std)

def _b5_percentile(df, period=40, buy_pct=5.0, sell_pct=95.0):
    """Percentile rank rodante: compra <buy_pct%, vende >sell_pct%."""
    prank = df['close'].rolling(period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=True)
    sig = pd.Series(0, index=df.index)
    sig[(prank < buy_pct) & (prank.shift() >= buy_pct)] = 1
    sig[(prank > sell_pct) & (prank.shift() <= sell_pct)] = -1
    return sig
_reg('B5_Percentile_5', _b5_percentile)

def _b5_vwap_2std(df, vwap_p=20):
    """VWAP ±2σ envelope."""
    vw = rolling_vwap(df, vwap_p)
    dev = df['close'] - vw
    std = dev.rolling(vwap_p).std()
    upper = vw + 2 * std
    lower = vw - 2 * std
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower) & (df['close'].shift() >= lower.shift())] = 1
    sig[(df['close'] > upper) & (df['close'].shift() <= upper.shift())] = -1
    return sig
_reg('B5_VWAP_2Std', _b5_vwap_2std)

# ─── B6 Strategies ────────────────────────────────────────────────────────────

def _b6_rubber_band(df, ema_p=20, threshold=3.0):
    """Rubber Band: cruza por debajo de -threshold% del EMA = buy."""
    e = ema(df['close'], ema_p)
    dev = (df['close'] - e) / (e + 1e-10) * 100
    sig = pd.Series(0, index=df.index)
    sig[(dev < -threshold) & (dev.shift() >= -threshold)] = 1
    sig[(dev > threshold) & (dev.shift() <= threshold)] = -1
    return sig
_reg('B6_Rubber_Band_3',    _b6_rubber_band)
_reg('B6_Rubber_Band_5',    _b6_rubber_band)
_reg('B6_Rubber_Band_7',    _b6_rubber_band)
_reg('B6_Rubber_Band_EMA50', _b6_rubber_band)

def _b6_vwap_ema(df, vwap_p=20, ema_p=21):
    """VWAP+EMA trend: compra al romper por encima de ambos."""
    vw = rolling_vwap(df, vwap_p)
    e  = ema(df['close'], ema_p)
    sig = pd.Series(0, index=df.index)
    above_both = (df['close'] > vw) & (df['close'] > e)
    was_below  = (df['close'].shift() <= vw.shift()) | (df['close'].shift() <= e.shift())
    sig[above_both & was_below] = 1
    below_both = (df['close'] < vw) & (df['close'] < e)
    was_above  = (df['close'].shift() >= vw.shift()) | (df['close'].shift() >= e.shift())
    sig[below_both & was_above] = -1
    return sig
_reg('B6_VWAP_EMA', _b6_vwap_ema)

def _b6_vwap_bb(df, vwap_p=20, bb_p=20, bb_std=2.0):
    """VWAP+BB: precio sobre VWAP pero bajo la banda inferior (dip en uptrend)."""
    vw = rolling_vwap(df, vwap_p)
    _, upper, lower = bb_bands(df['close'], bb_p, bb_std)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] > vw) & (df['close'] < lower)] = 1
    sig[(df['close'] < vw) & (df['close'] > upper)] = -1
    return sig
_reg('B6_VWAP_BB', _b6_vwap_bb)

def _b6_vwap_stoch(df, vwap_p=20, kp=14, dp=3, stoch_buy=20, stoch_sell=80):
    """VWAP+Stoch: precio sobre VWAP y estocástico sobrevendido."""
    vw = rolling_vwap(df, vwap_p)
    k, _ = stoch_dp(df, kp, dp)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] > vw) & (k < stoch_buy)] = 1
    sig[(df['close'] < vw) & (k > stoch_sell)] = -1
    return sig
_reg('B6_VWAP_Stoch', _b6_vwap_stoch)

def _b6_vwap_extreme(df, vwap_p=20, std_mult=2.0):
    """VWAP extreme deviation: desvío > std_mult×σ del VWAP."""
    vw = rolling_vwap(df, vwap_p)
    dev = df['close'] - vw
    std = dev.rolling(vwap_p).std()
    sig = pd.Series(0, index=df.index)
    sig[(dev < -std_mult * std) & (dev.shift() >= -std_mult * std.shift())] = 1
    sig[(dev > std_mult * std) & (dev.shift() <= std_mult * std.shift())] = -1
    return sig
_reg('B6_VWAP_Extreme', _b6_vwap_extreme)

def _b6_oversold_extreme(df, rsi_p=7, rsi_buy=15.0, uo_buy=30.0, willi_buy=-90.0):
    """Triple oversold: RSI + UO + Williams_R todos extremos."""
    r  = rsi(df['close'], rsi_p)
    uo = ultimate_osc(df)
    wr = williams_r(df)
    sig = pd.Series(0, index=df.index)
    sig[(r < rsi_buy) & (uo < uo_buy) & (wr < willi_buy)] = 1
    sig[(r > 100 - rsi_buy) & (uo > 100 - uo_buy) & (wr > 100 + willi_buy)] = -1
    return sig
_reg('B6_Oversold_Extreme', _b6_oversold_extreme)

def _b6_oversold_combo(df, rsi_p=14, rsi_buy=25.0, rsi_sell=60.0,
                        kp=14, dp=3, stoch_buy=20.0, stoch_sell=80.0,
                        mfi_p=14, mfi_buy=20.0, mfi_sell=80.0, min_signals=3):
    """Multi-indicator oversold: RSI + Stoch + MFI, compra si min_signals confirman."""
    r    = rsi(df['close'], rsi_p)
    k, _ = stoch_dp(df, kp, dp)
    m    = mfi(df, mfi_p)
    buy_score  = (r < rsi_buy).astype(int) + (k < stoch_buy).astype(int) + (m < mfi_buy).astype(int)
    sell_score = (r > rsi_sell).astype(int) + (k > stoch_sell).astype(int) + (m > mfi_sell).astype(int)
    sig = pd.Series(0, index=df.index)
    sig[buy_score  >= min_signals] = 1
    sig[sell_score >= min_signals] = -1
    return sig

_reg('B6_Oversold_Combo_Heavy',  lambda df, **p: _b6_oversold_combo(df, min_signals=3, **{k: v for k, v in p.items()}))
_reg('B6_Oversold_Combo_Medium', lambda df, **p: _b6_oversold_combo(df, min_signals=2, **{k: v for k, v in p.items()}))
_reg('B6_Oversold_Combo_Light',  lambda df, **p: _b6_oversold_combo(df, min_signals=1, **{k: v for k, v in p.items()}))

def _b6_vol_climax(df, vol_mult=2.0, rsi_p=14, rsi_buy=35.0, rsi_sell=65.0):
    """Volume climax + RSI: volumen extremo + sobrevendido/sobrecomprado."""
    vol_avg = sma(df['volume'], 20)
    climax  = df['volume'] > vol_avg * vol_mult
    r = rsi(df['close'], rsi_p)
    sig = pd.Series(0, index=df.index)
    sig[climax & (df['close'] < df['open']) & (r < rsi_buy)]  = 1
    sig[climax & (df['close'] > df['open']) & (r > rsi_sell)] = -1
    return sig
_reg('B6_Vol_Climax', _b6_vol_climax)

def _b6_midnight_rev(df, hour_window=2, rsi_p=7, rsi_buy=30.0, rsi_sell=70.0):
    """Midnight reversal: primera hora UTC + RSI sobrevendido/comprado."""
    if hasattr(df.index, 'hour'):
        near_midnight = pd.Series(df.index.hour, index=df.index) < hour_window
    else:
        near_midnight = pd.Series(True, index=df.index)
    r = rsi(df['close'], rsi_p)
    sig = pd.Series(0, index=df.index)
    sig[near_midnight & (r < rsi_buy)]  = 1
    sig[near_midnight & (r > rsi_sell)] = -1
    return sig
_reg('B6_Midnight_Rev', _b6_midnight_rev)

def _generic_fallback(df, **kwargs):
    """Para estrategias no implementadas: señal nula (fuerza rechazo)."""
    return pd.Series(0, index=df.index)

# ─── Carga de datos ───────────────────────────────────────────────────────────
_candle_cache = {}

def load_candles(symbol, tf):
    key = (symbol, tf)
    if key in _candle_cache:
        return _candle_cache[key]
    db = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT CAST(ts AS INTEGER) as ts, open, high, low, close, volume "
        "FROM candles WHERE symbol=? AND timeframe=? AND typeof(ts)='integer' ORDER BY ts",
        db, params=(symbol, tf))
    db.close()
    if len(df) == 0:
        return None
    df['ts'] = pd.to_datetime(df['ts'].astype('int64'), unit='ms')
    df = df.set_index('ts')
    _candle_cache[key] = df
    return df

# ─── AGENTE A: Réplica de optuna_v7 backtest_raw ─────────────────────────────

def agent_a_optuna_replica(df, sig):
    """
    Réplica exacta de backtest_raw de optuna_v7:
    - Sin SL/TP
    - Sale en el próximo signal opuesto o max_bars
    - Este es el WR que optuna_v7 reporta en test_wr
    """
    cost = SLIPPAGE + COMMISSION
    opens = df['open'].values; highs = df['high'].values; lows = df['low'].values
    sigs = sig.values
    trades = []; in_t = False; ep = 0; direction = 0; entry_i = 0

    for i in range(len(sigs) - 1):
        if not in_t:
            if sigs[i] != 0:
                direction = sigs[i]
                ep = opens[i + 1] * (1 + cost * direction)
                in_t = True; entry_i = i
        else:
            dur = i - entry_i
            if sigs[i] != 0 and sigs[i] != direction:
                # Señal opuesta → salir
                xp = opens[i + 1] * (1 - cost * direction)
                pnl = direction * (xp - ep) / ep
                trades.append({'pnl': pnl, 'exit': 'signal', 'dur': dur})
                in_t = False
                # Iniciar nueva posición
                direction = sigs[i]
                ep = opens[i + 1] * (1 + cost * direction)
                in_t = True; entry_i = i
            elif dur >= MAX_BARS:
                xp = opens[i + 1] * (1 - cost * direction)
                pnl = direction * (xp - ep) / ep
                trades.append({'pnl': pnl, 'exit': 'maxbar', 'dur': dur})
                in_t = False

    if not trades:
        return None

    wins = [t for t in trades if t['pnl'] > 0]
    return {
        'n': len(trades),
        'wr': len(wins) / len(trades) * 100,
        'pnl': sum(t['pnl'] for t in trades) * 100,
        'agent': 'A_optuna_replica'
    }

# ─── AGENTE B: Auditor Externo (SL/TP empírico, CIEGO) ───────────────────────

def _calibrate_sl_tp(df_train, sig_train):
    """P90(MAE) × 2.0 para SL, P90(MFE winners) × 0.85 para TP — igual que optuna_v7.2"""
    cost = SLIPPAGE + COMMISSION
    opens = df_train['open'].values; highs = df_train['high'].values; lows = df_train['low'].values
    sigs = sig_train.values
    all_maes = []; wmfes = []; in_t = False; ep = 0; d = 0; ei = 0

    for i in range(len(sigs) - 1):
        if not in_t:
            if sigs[i] != 0:
                d = sigs[i]; ep = opens[i + 1] * (1 + cost * d); in_t = True; ei = i
        else:
            mt = []; mf = []
            for j in range(ei + 1, min(ei + MAX_BARS + 1, len(sigs) - 1)):
                if d == 1:
                    mt.append(min(0, (lows[j] - ep) / ep))
                    mf.append(max(0, (highs[j] - ep) / ep))
                else:
                    mt.append(min(0, (ep - highs[j]) / ep))
                    mf.append(max(0, (ep - lows[j]) / ep))
            if mt: all_maes.append(abs(min(mt)))
            if mf and max(mf) > 0: wmfes.append(max(mf))
            in_t = False
            if sigs[i] != 0:
                d = sigs[i]; ep = opens[i + 1] * (1 + cost * d); in_t = True; ei = i

    sl = np.clip(np.percentile(all_maes, 90) * 2.0, SL_MIN, SL_MAX) if len(all_maes) >= 5 else 0.10
    tp = np.clip(np.percentile(wmfes, 90) * 0.85, TP_MIN, TP_MAX) if len(wmfes) >= 5 else sl * 1.5
    return sl, tp

def _run_sl_tp_backtest(df, sig, sl, tp):
    """Backtest realista con SL/TP fijo."""
    cost = SLIPPAGE + COMMISSION
    opens = df['open'].values; highs = df['high'].values; lows = df['low'].values
    sigs = sig.values
    trades = []; in_t = False; ep = 0; d = 0; ei = 0

    for i in range(len(sigs) - 1):
        if not in_t:
            if sigs[i] != 0:
                d = sigs[i]; ep = opens[i + 1] * (1 + cost * d); in_t = True; ei = i
        else:
            h, l = highs[i], lows[i]; closed = False; pnl = 0
            if d == 1:
                if   (l - ep) / ep <= -sl:   pnl = -sl - cost; closed = True
                elif (h - ep) / ep >= tp:    pnl =  tp - cost; closed = True
                elif i - ei >= MAX_BARS:     pnl = (opens[i+1] * (1-cost) - ep) / ep; closed = True
            else:
                if   (ep - h) / ep <= -sl:   pnl = -sl - cost; closed = True
                elif (ep - l) / ep >= tp:    pnl =  tp - cost; closed = True
                elif i - ei >= MAX_BARS:     pnl = (ep - opens[i+1] * (1+cost)) / ep; closed = True
            if closed:
                trades.append({'pnl': pnl}); in_t = False

    return trades

def _normalize_params_for_agent_b(strategy_name, params):
    """
    Traduce nombres de params del formato viejo (batch_search_spaces.py)
    al formato nuevo (strategies_round3.py) para las implementaciones de Agent B.
    Si params está vacío, se usan los defaults de cada función.
    """
    if not params:
        return {}

    p = dict(params)  # copia para no mutar

    # Rubber Band: ema_period → ema_p, buy/sell_dist_pct → threshold (valor abs promedio)
    if 'ema_period' in p and 'ema_p' not in p:
        p['ema_p'] = p.pop('ema_period')
    if 'buy_dist_pct' in p and 'threshold' not in p:
        buy_d = abs(p.pop('buy_dist_pct', 3.0))
        sell_d = abs(p.pop('sell_dist_pct', buy_d))
        p['threshold'] = (buy_d + sell_d) / 2

    # RSI params
    if 'rsi_period' in p and 'rsi_p' not in p:
        p['rsi_p'] = p.pop('rsi_period')
    if 'rsi_buy_thresh' in p and 'rsi_buy' not in p:
        p['rsi_buy'] = p.pop('rsi_buy_thresh')
    if 'rsi_sell_thresh' in p and 'rsi_sell' not in p:
        p['rsi_sell'] = p.pop('rsi_sell_thresh')

    # VWAP Stoch: vwap_buy_dist / vwap_sell_dist → vwap_p (approximate), stoch → kp/dp
    if 'stoch_period' in p and 'kp' not in p:
        p['kp'] = p.pop('stoch_period')
    if 'stoch_smooth' in p and 'dp' not in p:
        p['dp'] = p.pop('stoch_smooth')
    if 'stoch_buy_thresh' in p and 'stoch_buy' not in p:
        p['stoch_buy'] = p.pop('stoch_buy_thresh')
    if 'stoch_sell_thresh' in p and 'stoch_sell' not in p:
        p['stoch_sell'] = p.pop('stoch_sell_thresh')
    # vwap dist params: drop (use vwap_p default)
    p.pop('vwap_buy_dist', None)
    p.pop('vwap_sell_dist', None)

    # BB params
    if 'bb_period' in p and 'bb_p' not in p:
        p['bb_p'] = p.pop('bb_period')

    # Remove unknown keys that would cause TypeError
    # (keep only what the function accepts via try/except in caller)
    return p


def agent_b_auditor(df, strategy_name, params, noise_pct=SLIPPAGE_NOISE):
    """
    Agente Auditor CIEGO:
    1. No recibe resultados de Optuna
    2. Implementa estrategia de forma independiente
    3. Calibra SL/TP desde train 70%
    4. Prueba con slippage noise (robustez)
    """
    # Obtener implementación ciega
    gen_fn = STRATEGY_IMPLS.get(strategy_name, _generic_fallback)
    if gen_fn is _generic_fallback:
        return {
            'n': 0, 'wr': 0, 'pnl': 0, 'sl': 0, 'tp': 0,
            'noise_wr': 0, 'agent': 'B_auditor',
            'note': f'estrategia {strategy_name} no implementada en auditor — requiere agregar'
        }

    # Normalizar nombres de params (viejo formato → nuevo formato)
    norm_params = _normalize_params_for_agent_b(strategy_name, params)

    # Generar señales en dataset completo (con defaults si params vacíos)
    try:
        sig = gen_fn(df, **norm_params)
    except TypeError:
        sig = gen_fn(df)  # usar defaults

    # Calibrar SL/TP desde primer 70% (período train)
    split = int(len(df) * 0.70)
    sl, tp = _calibrate_sl_tp(df.iloc[:split], sig.iloc[:split])

    # Backtest en dataset COMPLETO
    trades = _run_sl_tp_backtest(df, sig, sl, tp)
    if not trades or len(trades) < 5:
        return {'n': len(trades) if trades else 0, 'wr': 0, 'pnl': 0,
                'sl': sl, 'tp': tp, 'noise_wr': 0, 'agent': 'B_auditor',
                'note': 'insuficientes trades'}

    wins = [t for t in trades if t['pnl'] > 0]
    wr = len(wins) / len(trades) * 100
    pnl_sum = sum(t['pnl'] for t in trades) * 100

    # ── Test de robustez: desplazamiento de precios ±0.1% ──────────────────
    df_noisy = df.copy()
    df_noisy['open']  *= (1 + noise_pct)
    df_noisy['high']  *= (1 + noise_pct)
    df_noisy['low']   *= (1 + noise_pct)
    df_noisy['close'] *= (1 + noise_pct)
    try:
        sig_noisy = gen_fn(df_noisy, **norm_params)
    except TypeError:
        sig_noisy = gen_fn(df_noisy)
    trades_noisy = _run_sl_tp_backtest(df_noisy, sig_noisy, sl, tp)
    wins_noisy = [t for t in trades_noisy if t['pnl'] > 0] if trades_noisy else []
    noise_wr = len(wins_noisy) / len(trades_noisy) * 100 if trades_noisy else 0

    # Max drawdown
    equity = np.cumsum([t['pnl'] for t in trades])
    max_dd = float(np.min(equity - np.maximum.accumulate(equity))) * 100

    return {
        'n': len(trades),
        'wr': round(wr, 1),
        'pnl': round(pnl_sum, 2),
        'sl': round(sl * 100, 2),
        'tp': round(tp * 100, 2),
        'max_dd': round(max_dd, 2),
        'noise_wr': round(noise_wr, 1),
        'agent': 'B_auditor',
        'note': ''
    }

# ─── PASO 4: Decisión ─────────────────────────────────────────────────────────

def decide(a_wr, b_wr, b_noise_wr, b_n):
    """
    PRODUCCIÓN:   A>=75% AND B>=70% AND noise_wr>=60%
    SHADOW:       uno cumple, el otro entre 50-70%
    DESCARTAR:    alguno < 50% o noise_wr muy inestable
    """
    a_pass = a_wr >= WR_PRODUCCION_A
    b_pass = b_wr >= WR_PRODUCCION_B
    noise_ok = b_noise_wr >= 60.0
    b_viable = b_wr >= WR_SHADOW_MIN
    b_n_ok = b_n >= 20

    if a_pass and b_pass and noise_ok and b_n_ok:
        return 'PRODUCCION', f'A={a_wr:.1f}%✓ B={b_wr:.1f}%✓ noise={b_noise_wr:.1f}%✓ n={b_n}✓'
    elif (a_pass or b_pass) and b_viable:
        reason = f'A={a_wr:.1f}% {"✓" if a_pass else "·"} B={b_wr:.1f}% {"✓" if b_pass else "·"} noise={b_noise_wr:.1f}% n={b_n}'
        return 'SHADOW_20', reason
    else:
        return 'DESCARTAR', f'A={a_wr:.1f}% B={b_wr:.1f}% noise={b_noise_wr:.1f}% n={b_n}'

# ─── Pipeline principal ───────────────────────────────────────────────────────

def validate_grail(grail, verbose=True):
    """Valida un único grail con el proceso dual de 4 pasos."""
    strat  = grail['strategy']
    symbol = grail['symbol']
    tf     = grail['timeframe']
    params = grail.get('best_params', {})
    optuna_reported_wr = grail.get('test_wr', 0)

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  {strat} × {symbol.split('/')[0]} × {tf}")
        print(f"  Optuna reportó: WR={optuna_reported_wr:.1f}% | n_trades={grail.get('test_trades',0)}")
        print(f"  Params: {params}")

    # Cargar candles
    df = load_candles(symbol, tf)
    if df is None or len(df) < 500:
        return {'decision': 'SKIP', 'reason': 'sin datos suficientes'}

    # ── AGENTE A: Réplica optuna_v7 (sin SL/TP) ───────────────────────────────
    sig_a = None
    try:
        sys.path.insert(0, ROOT)
        # 1) Intentar strategy_factory (clásicas)
        from strategy_factory import REGISTRY as SF_REGISTRY
        if strat in SF_REGISTRY:
            sig_a = SF_REGISTRY[strat]['gen'](df, **params)
    except Exception:
        pass

    if sig_a is None:
        try:
            # 2) Intentar strategies_round3 (B5/B6 y otras)
            from strategies_round3 import STRATEGY_TYPES_R3
            if strat in STRATEGY_TYPES_R3:
                sig_a = STRATEGY_TYPES_R3[strat]['gen'](df, **params)
        except Exception:
            pass

    if sig_a is None:
        # 3) Fallback a implementación interna (usa defaults si params vacíos)
        gen_fn_a = STRATEGY_IMPLS.get(strat, _generic_fallback)
        try:
            sig_a = gen_fn_a(df, **params)
        except TypeError:
            sig_a = gen_fn_a(df)  # defaults only

    res_a = agent_a_optuna_replica(df, sig_a)
    if res_a is None:
        res_a = {'n': 0, 'wr': 0, 'pnl': 0}

    # ── AGENTE B: Auditor externo (con SL/TP, CIEGO) ──────────────────────────
    res_b = agent_b_auditor(df, strat, params)

    # ── GAP analysis ──────────────────────────────────────────────────────────
    gap_optuna_vs_a = optuna_reported_wr - res_a['wr']
    gap_a_vs_b      = res_a['wr'] - res_b['wr']

    # ── PASO 4: Decisión ──────────────────────────────────────────────────────
    decision, reason = decide(res_a['wr'], res_b['wr'], res_b.get('noise_wr', 0), res_b['n'])

    result = {
        'strategy': strat,
        'symbol': symbol,
        'timeframe': tf,
        'params': params,
        'optuna_wr': optuna_reported_wr,
        'optuna_n_trades': grail.get('test_trades', 0),
        'agent_a': {
            'wr': round(res_a['wr'], 1),
            'n': res_a['n'],
            'pnl_pct': round(res_a['pnl'], 2),
            'engine': 'backtest_raw (sin SL/TP)',
        },
        'agent_b': {
            'wr': round(res_b['wr'], 1),
            'n': res_b['n'],
            'pnl_pct': res_b.get('pnl', 0),
            'sl_pct': res_b.get('sl', 0),
            'tp_pct': res_b.get('tp', 0),
            'max_dd_pct': res_b.get('max_dd', 0),
            'noise_wr': res_b.get('noise_wr', 0),
            'engine': 'backtest_sl_tp + slippage_noise',
            'note': res_b.get('note', ''),
        },
        'gaps': {
            'optuna_vs_A': round(gap_optuna_vs_a, 1),
            'A_vs_B': round(gap_a_vs_b, 1),
        },
        'decision': decision,
        'reason': reason,
        'full_history_bars': len(df),
        'full_history_from': str(df.index[0].date()),
        'full_history_to': str(df.index[-1].date()),
        'validated_at': datetime.utcnow().isoformat(),
    }

    if verbose:
        icon = {'PRODUCCION': '🟢', 'SHADOW_20': '🟡', 'DESCARTAR': '🔴'}.get(decision, '⚪')
        print(f"  Agente A (réplica optuna):  WR={res_a['wr']:.1f}% | n={res_a['n']} | gap vs optuna={gap_optuna_vs_a:+.1f}pp")
        print(f"  Agente B (auditor externo): WR={res_b['wr']:.1f}% | n={res_b['n']} | SL={res_b.get('sl',0):.1f}% TP={res_b.get('tp',0):.1f}% | noise_WR={res_b.get('noise_wr',0):.1f}%")
        print(f"  Gap A→B: {gap_a_vs_b:+.1f}pp | Gap optuna→A: {gap_optuna_vs_a:+.1f}pp")
        print(f"  {icon} DECISIÓN: {decision} — {reason}")

    return result

def run_pipeline(filters=None, top_n=50, verbose=True):
    """Pipeline completo: carga grails → filtra → valida → guarda resultados."""
    # Paso 1: Cargar y filtrar grails
    with open(GRAILS_PATH) as f:
        all_grails = json.load(f)

    min_wr     = (filters or {}).get('min_wr', 80.0)
    min_trades = (filters or {}).get('min_trades', 20)
    symbol_f   = (filters or {}).get('symbol')
    strategy_f = (filters or {}).get('strategy')

    candidates = [g for g in all_grails
                  if g.get('test_wr', 0) >= min_wr
                  and g.get('test_trades', 0) >= min_trades
                  and 'triple_barrier' not in g.get('strategy', '')]

    if symbol_f:
        candidates = [g for g in candidates if symbol_f in g['symbol']]
    if strategy_f:
        candidates = [g for g in candidates if strategy_f in g['strategy']]

    # Ordenar: primero los más prometedores (train_wr alto = menos overfit)
    candidates.sort(key=lambda x: -(x.get('train_wr', 0) * 0.6 + x.get('test_wr', 0) * 0.4))
    candidates = candidates[:top_n]

    print(f"\n{'═'*70}")
    print(f"  GRAIL DUAL VALIDATOR — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Filtros: test_wr>={min_wr}%, n_trades>={min_trades}, top={top_n}")
    print(f"  Candidatos seleccionados: {len(candidates)}")
    print(f"{'═'*70}")

    results = []
    stats = defaultdict(int)

    for i, grail in enumerate(candidates):
        if verbose:
            print(f"\n[{i+1}/{len(candidates)}]", end='')
        try:
            res = validate_grail(grail, verbose=verbose)
            results.append(res)
            stats[res['decision']] += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            stats['ERROR'] += 1
            results.append({'strategy': grail['strategy'], 'symbol': grail['symbol'],
                            'timeframe': grail['timeframe'], 'params': grail.get('best_params',{}),
                            'optuna_wr': grail.get('test_wr',0), 'optuna_n_trades': grail.get('test_trades',0),
                            'agent_a': {'wr':0,'n':0,'pnl_pct':0}, 'agent_b': {'wr':0,'n':0,'pnl_pct':0,'sl':0,'tp':0,'max_dd_pct':0,'noise_wr':0,'note':str(e)},
                            'gaps': {'optuna_vs_A':0,'A_vs_B':0}, 'decision':'ERROR', 'reason':str(e),
                            'full_history_bars':0,'full_history_from':'','full_history_to':'','validated_at':datetime.utcnow().isoformat()})

    # ── Guardar resultados ────────────────────────────────────────────────────
    with open(OUT_JSON, 'w') as f:
        json.dump({'run_at': datetime.utcnow().isoformat(), 'filters': filters,
                   'results': results, 'summary': dict(stats)}, f, indent=2)

    # CSV para análisis rápido
    rows = []
    for r in results:
        rows.append({
            'strategy': r['strategy'], 'symbol': r['symbol'], 'timeframe': r['timeframe'],
            'optuna_wr': r['optuna_wr'], 'agent_a_wr': r['agent_a']['wr'],
            'agent_b_wr': r['agent_b']['wr'], 'noise_wr': r['agent_b']['noise_wr'],
            'gap_optuna_A': r['gaps']['optuna_vs_A'], 'gap_A_B': r['gaps']['A_vs_B'],
            'decision': r['decision'], 'n_real': r['agent_b']['n'],
            'max_dd': r['agent_b']['max_dd_pct'],
        })
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    # ── Resumen final ─────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  RESUMEN FINAL — {len(results)} grails validados")
    print(f"{'═'*70}")
    print(f"  🟢 PRODUCCION:  {stats.get('PRODUCCION', 0):3d}  ({stats.get('PRODUCCION',0)*100//max(1,len(results))}%)")
    print(f"  🟡 SHADOW_20:   {stats.get('SHADOW_20', 0):3d}  ({stats.get('SHADOW_20',0)*100//max(1,len(results))}%)")
    print(f"  🔴 DESCARTAR:   {stats.get('DESCARTAR', 0):3d}  ({stats.get('DESCARTAR',0)*100//max(1,len(results))}%)")
    if stats.get('ERROR'):
        print(f"  ⚠️  ERRORES:     {stats.get('ERROR', 0):3d}")

    if results:
        prod_grails = [r for r in results if r['decision'] == 'PRODUCCION']
        if prod_grails:
            print(f"\n  GRAILS LISTOS PARA PRODUCCIÓN:")
            for r in prod_grails:
                print(f"    ✓ {r['strategy']:20s} × {r['symbol'].split('/')[0]:8s} | "
                      f"A={r['agent_a']['wr']:.1f}% B={r['agent_b']['wr']:.1f}% n={r['agent_b']['n']}")

        avg_gap = np.mean([r['gaps']['optuna_vs_A'] for r in results])
        avg_a_b = np.mean([r['gaps']['A_vs_B'] for r in results])
        print(f"\n  Gap promedio Optuna→A: {avg_gap:+.1f}pp (calibración de sesgo)")
        print(f"  Gap promedio A→B:      {avg_a_b:+.1f}pp (impacto de SL/TP realista)")

    print(f"\n  Resultados: {OUT_JSON}")
    print(f"  CSV:        {OUT_CSV}")
    return results

# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grail Dual Validator')
    parser.add_argument('--symbol',     help='Filtrar por símbolo (ej: AGT/USDT:USDT)')
    parser.add_argument('--strategy',   help='Filtrar por estrategia (ej: RSI_VWAP)')
    parser.add_argument('--min-wr',     type=float, default=80.0, help='WR mínimo Optuna (default: 80)')
    parser.add_argument('--min-trades', type=int,   default=20,   help='N trades mínimo (default: 20)')
    parser.add_argument('--top',        type=int,   default=50,   help='Máximo grails a validar (default: 50)')
    parser.add_argument('--quiet',      action='store_true',       help='Sin output detallado')
    args = parser.parse_args()

    filters = {'min_wr': args.min_wr, 'min_trades': args.min_trades}
    if args.symbol:   filters['symbol']   = args.symbol
    if args.strategy: filters['strategy'] = args.strategy

    run_pipeline(filters=filters, top_n=args.top, verbose=not args.quiet)
