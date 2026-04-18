#!/usr/bin/env python3
"""
TV→Python WR Validator
======================
Valida que las estrategias de TradingView (W60) producen el mismo WR en Python.
Usa los MISMOS activos y temporalidades que TV reportó.

Criterio: |WR_python - WR_tv| ≤ 10pp = PASS
          10pp < gap ≤ 25pp          = WARNING
          gap > 25pp                  = FAIL
          no impl Python              = NO_IMPL

Uso:
    python3 validate_tv_vs_python.py
    python3 validate_tv_vs_python.py --strategy TV_FlawlessVictory
    python3 validate_tv_vs_python.py --limit 20 --workers 4
"""
import sys, os, json, sqlite3, time, re, glob, importlib.util, struct
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import argparse

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = '/Users/sabrina/CLAUDE CODE'
DB_PATH     = '/Users/sabrina/Code/activos binace /activos_binance.db'
W60_JSON    = f'{BASE_DIR}/trading view/tv_backtest_results/W60_CORRECTO_PARA_PYTHON.json'
BATCH_DIR   = f'{BASE_DIR}/Estrategias/strategies_tv2_batches'
RESULTS_DIR = f'{BASE_DIR}/Estrategias/data/tv_validation'
os.makedirs(RESULTS_DIR, exist_ok=True)

COMMISSION  = 0.001   # 0.1% per side (standard)
MAX_HOLD    = 50      # max bars to hold position (opposite signal or max)
ATR_SL_MULT = 2.0    # ATR multiplier for SL fallback
ATR_TP_MULT = 3.0    # ATR multiplier for TP fallback (1.5:1 R:R)
MIN_SIG_DENSITY = 0.05  # if signal_rate < 5% of bars, use ATR SL/TP exit instead
GAP_PASS    = 10.0    # pp threshold for PASS
GAP_WARN    = 25.0    # pp threshold for WARNING

# ── Name Mapping: TV original name → Python key ───────────────────────────
# Hard-coded high-confidence matches + verified from batch docstrings
TV_NAME_TO_PY = {
    # Direct / very clear matches
    'Flawless Victory Strategy - 15min BTC Machine Lear': 'TV_FlawlessVictory',
    'Flawless Victory Strategy':                          'TV_FlawlessVictory',
    'UT Bot Alerts':                                       'TV_UTBot',
    'UT Bot Strategy':                                     'TV_UTBot',
    'Larry Connors RSI 3 Strategy':                        'TV_ConnorsRSI3',
    "Noro's RSI Strategy":                                 'TV_ConnorsRSI3',
    'BTFD strategy [3min]':                                'TV_BTFD',
    'TradingGroundhog - Strategy & Wavetrend V2':          'TV_WaveTrend',
    'GAVAD - Selling after a Strong Moviment':             'TV_GAVAD',
    '2Mars - MA / BB / SuperTrend':                        'TV_2Mars',
    '2Mars strategy [OKX]':                                'TV_2Mars',
    'Open Close Cross Strategy R5 revised by JustUncleL': 'TV_OpenCloseCross',
    'ABCD Harmonic Pattern Strategy (Bull + Bear) ':       'TV_ABCD',
    'ABCD Harmonic Pattern Strategy (Bull + Bear)':        'TV_ABCD',
    'MASU+ Institutional v4.5+':                          'TV_MASUInst',
    'Smart Money Bot [MTF Confluence Edition]':            'TV_SmartMoneyMTF',
    'Uptrick X PineIndicators: Z-Score Flow Strategy':    'TV_ZScoreFlow',
    'Bollinger Bands + RSI Double Strategy (by ChartArt)':'TV_DoubleBollinger',
    'Double Bollinger Bands Strategy with Signals (By R': 'TV_DoubleBollinger',
    'AMRS_LongOnly_PartTimer':                            'TV_AMRSLongOnly',
    'ORB SESSIONS':                                        'TV_ORBSessions',
    'Heiken Ashi & Super Trend':                          'TV_HeikenAshiMTF',
    "Noro's Locomotive Strategy v1.0":                    'TV_NoroDonchian',
    "Noro's Distance Strategy v1.0":                      'TV_NoroDonchian',
    "Noro's Stochastic Strategy v1.0":                    'TV_NoroDonchian',
    "Kifier's MFI/STOCH Hidden Divergence/Trend Follower":'TV_KifierMFI',
    'Dskyz (DAFE) MAtrix with ATR-Powered Precision':     'TV_DskyzDAFE',
    'Outside Bar Strategy % (Alessio)':                   'TV_BigBarStrat',
    'HURST Channel Strategy':                             'TV_DonchianIdea',
    'K-TREND Strategy':                                   'TV_TrendCatcher',
    'Pivot Fib 4H — EA':                                  'TV_PivotSuperTrend',
    'BB%B Strat':                                         'TV_BBKeltnerSqueeze',
    'Enhanced Opening Range Strategy':                    'TV_ORBSessions',
    'Stochastic Z-Score Oscillator Strategy [Quantrader]': 'TV_ZScoreFlow',
    '3 Down, 3 Up Strategy':                              'TV_3CandleStrike',
    # 'Williams %R Strategy' → now in Batch183 as TV_WilliamsRSt
    'MACD + RSI TS':                                      'TV_MACDDivergence',
    'Larry Connors RSI 3 Strategy':                       'TV_ConnorsRSI3',
    'CS Basic Scripts - Stochastic Special (Strategy)':   'TV_Stoch_Momentum',
    'SMI Ergodic Oscillator Backtest ver.2':              'TV_Combo123Ergodic',
    'Bollinger band & Volume based strategy V2':          'TV_BBPB_OBV',
    'RSI Divergence Indicator strategy':                  'TV_RSI_OB_OS',
    'TradingGroundhog - Strategy & Fractal V1':           'TV_GroundhogFractal',
    'D-Bot Alpha RSI Breakout Strategy':                  'TV_DBotAlphaRSI',
    'RSI Fibonacci Reversal Strategy (Intraday Hunter)':  'TV_RSIFibReversal',
    # 'Multi-Timeframe RSI Grid Strategy with ATR': SKIP — grid/martingale strategy (3 TF RSIs + pyramiding),
    #   incompatible with simple signal gen functions. TV_MultiSupertrend mapping was WRONG (40pp gap).
    'BTCUSD 1D Trend Strategy [Gemini]':                  'TV_TrendCatcher',
    'ADX + EMA channel, trend pullback':                  'TV_ADX_Regime',
    'adx efi 50 ema channel, trend pullback':             'TV_ADX_Regime',
    'EMA Crossover (Short Focus with Trailing Stop)':     'TV_DblEMACrossVol',
    'Normalized MACD (v420) strategy':                    'TV_MACDDivergence',
    'Hull Moving Average and Daily Candle Crossover':     'TV_DoubleSMA',
    'Short Selling EMA Cross (By Coinrule)':              'TV_DblEMACrossVol',
    'Bitcoin trend RVI and Ema':                          'TV_GetTrend',
    'Keltner Channel Strategy by Kevin Davey':            'TV_KeltnerKevDav',
    'Bolinger strategy v1 open':                          'TV_AdvancedBB',
    'Heiken Ashi & Super Trend':                          'TV_HeikenAshi_SuperTrend2',
    'TradingGroundhog - Strategy & Fractal V1':           'TV_GroundhogFractal',
    # Batch176 — conversiones 2026-04-05
    'ChopFlow ATR Scalp Strategy':                        'TV_ChopFlow_ATR',
    'BTC Trading Robot':                                  'TV_BTC_TradingRobot',
    'Trend #4 - ATR+EMA channel':                         'TV_ATR_EMA_Trend4',
    'TRIN (Arms Index) Trading Strategy':                 'TV_TRIN_Index',
    'BULL Whale Finder + BTC 1h':                         'TV_BullWhale',
    # Batch177 — conversiones 2026-04-05
    '3kilos BTC 15m':                                     'TV_3kilosBTC',
    'Donchian x WMA Crossover (2025 Only, Adjustable TP': 'TV_DonchianWMA',
    # Batch179 — conversiones 2026-04-05
    'VIDYA ProTrend Multi-Tier Profit':                   'TV_VIDYATrend',
    'SMA Offset Strategy':                                'TV_SMAOffset',
    # Batch180 — conversiones 2026-04-05
    'EMA SCALPEUR + RSi - SHORT':                         'TV_EMAScalpeur',
    'Consecutive Weekly Up w/o doji':                     'TV_ConsecWeeklyUp',
    # Batch181 — conversiones 2026-04-05
    'AVG Stochastic Strategy [M30 Backtesting]':          'TV_AVGStoch',
    'RSITrendStrategy':                                   'TV_RSITrend',
    'Strategia RSI semplice':                             'TV_StrategiaRSI',
    'Rob Booker Reversal Tabs Strategy':                  'TV_RobBooker',
    # Batch182 — conversiones 2026-04-05
    'Best TradingView Strategy - For NASDAQ and DOW30 an':'TV_BestTVStrategy',
    'Best TradingView Strategy':                          'TV_BestTVStrategy',
    'I11L - Meanreverter 4h':                            'TV_I11LMeanRev',
    '3x Supertrend and Stoch RSI':                       'TV_3xSupertrend',
    # Batch183 — conversiones 2026-04-05
    'Williams %R Strategy':                              'TV_WilliamsRSt',
    '5212 EMA Strategy':                                 'TV_5212EMA',
    'CCI High Performance long only':                    'TV_CCIHighPerf',
    # Batch184 — conversiones 2026-04-05
    'Mean Reversion and Trendfollowing':                 'TV_MeanRevTrend',
    'Impulse Strategy Signals V2':                       'TV_ImpulseSignals',
    'Turn of the Month Strategy on Steroids':            'TV_TurnOfMonth',
    # Batch185 — conversiones 2026-04-05
    'RSI BREAKOUT SIGNALS':                              'TV_RSIBreakoutSig',
    'Simple RSI Strategy - Rule Based Higher Timeframe': 'TV_SimpleRSIHTF',
    'Simple RSI Strategy - Rule Based Higher Timeframe Trading': 'TV_SimpleRSIHTF',
    'Mean Reversion Mirror':                             'TV_MeanRevMirror',
    # Batch255 — conversiones 2026-04-05
    'VWAP Pullback + RSI Confirmation':                  'TV_VWAPPullbackRSI',
    'ADR AsianLS':                                       'TV_ADRAsianLS',
    "MAC's V6 final":                                    'TV_MACsV6',
    # Batch256 — conversiones 2026-04-05
    'Mean Reversion V-F':                                'TV_MeanReversion_VF2',
    'Zazzamira 50-25-25 Trend System':                   'TV_Zazzamira_Trend',
    # GAVAD ya mapeado arriba → TV_GAVAD (batch132); TV_GAVAD2 es reimpl con prefijo TV_*
    # 'GAVAD - Selling after a Strong Moviment':         'TV_GAVAD2',  # alias de TV_GAVAD
    # Batch257 — conversiones 2026-04-05
    'MNQ 5m V5 AGG Dual Mode Trend Continuation Strategy': 'TV_MNQ_DualModeTrend',
    'Moving Stop-Loss mechanism + alerts to MT4/MT5':    'TV_MovingStopLoss',
    'Trade Manager + MOST RSI':                          'TV_MOSTRsi',
    # Batch258 — conversiones 2026-04-05
    'VWAP Breakout NY Open Only ':                       'TV_VWAPBreakoutNY',
    'VWAP Breakout NY Open Only':                        'TV_VWAPBreakoutNY',
    "Noro's Locomotive Strategy v1.0":                   'TV_NoroLocomotive',
    '[Anwar] : BTC Trend Strategy V2 (Fixed + ADX Working)': 'TV_AnwarBTCTrend',

    # Batch259 — conversiones 2026-04-05
    'NASDAQ 100 Peak Hours Strategy':                        'TV_NASDAQPeakHours',
    'iD EMARSI on Chart':                                    'TV_EMARSI_iD',
    'Gold Pro Strategy':                                     'TV_GoldPro',
    'NORN WEAVE | FEHU':                                     'TV_NornWeave',

    # Batch260 — conversiones 2026-04-05
    'Konigs | Bollinger Band Mean Reversion (Session Filter)': 'TV_KonigsBBMeanRev',
    'macd+stoch':                                            'TV_MACDStoch',
    'Cheat Code- Example 1; Short-Term; Follow the Trend ':  'TV_CheatCodeTrend',
    'Cheat Code- Example 1; Short-Term; Follow the Trend':   'TV_CheatCodeTrend',
    'Moving Average Displaced Envelope Backtest':            'TV_MADisplacedEnv',

    # Batch261 — conversiones 2026-04-05
    'TrueBlock bot ':                                        'TV_TrueBlock',
    'TrueBlock bot':                                         'TV_TrueBlock',
    'Singles AI Pro Engine':                                 'TV_SinglesAI',
    'Buy&Sell Strategy depends on AO+Stoch+RSI+ATR by SerdarYILMAZ': 'TV_AOStochRSI',
    'BTC Re-Entry Alpha 1H':                                 'TV_BTCReEntry',

    # Batch262 — conversiones 2026-04-05
    "Noro's Distance Strategy v1.0":                         'TV_NoroDistance',
    "Noro's Stochastic Strategy v1.0":                       'TV_NoroStoch',
    'Qullamagi EMA Breakout Autotrade (Crypto Futures L+S)': 'TV_QullamagiEMA',
    'DAKELAX-XRPUSDT Bollinger Band Strategy for Tradebotler': 'TV_DAKELAXBB',

    # Batch263 — conversiones 2026-04-05
    'R3 ETF Strategy':                                       'TV_R3ETF',
    'Short In Downtrend Below MA100 (Coinrule)':             'TV_ShortDowntrend',
    'BB%/MFI/RSI':                                           'TV_BBMFIRsi',
    'LowFinder_PyraMider_V2':                                'TV_LowFinderPyra',

    # Batch264 — conversiones 2026-04-05
    'TTP Intelligent Accumulator':                           'TV_TTPAccumulator',
    'EMA and Dow Theory Strategies V2  DOGE Current Optimum Value': 'TV_EMADowTheory',
    'TX_Smart_Cross_Session_TrendFollow':                    'TV_TxSmartCross',
    'Hyper Insight MA Strategy [Universal]':                 'TV_HyperInsightMA',

    # Batch265 — conversiones 2026-04-05
    'Vandan V2':                                             'TV_VandanV2',
    'RSI Buy & Sell Trading Script':                         'TV_RSIBuySell',
    'Dynamic Support and Resistance Pivot Strategy ':        'TV_DynamicSRPivot',
    'Dynamic Support and Resistance Pivot Strategy':         'TV_DynamicSRPivot',
    'Double Bollinger Bands Strategy with Signals (By Rolwin)': 'TV_DoubleBB',

    # Batch266 — conversiones 2026-04-05
    '3 x EMA + Stochastic RSI + ATR':                        'TV_3EMAStochATR',
    'TPS Short Strategy by Larry Conners':                   'TV_TPSConnors',
    'Ultimate Oscillator Trading Strategy':                  'TV_UltimateOsc',
    'Stochastic Z-Score Oscillator Strategy [TradeDots]':    'TV_StochZScore',

    # Batch267 — conversiones 2026-04-05
    'ICT Gap Strategy [Swing SL + Sessions]':                'TV_ICTGap',
    'RSI OverTrend Strategy (by Marcoweb) v1.0':             'TV_RSIOverTrend',
    'HatiKO Envelopes':                                      'TV_HatiKOEnv',
    'CCI Level Zone':                                        'TV_CCIZone',

    # Batch268 — conversiones 2026-04-05
    'REVERSALS':                                             'TV_Reversals',
    'Gold/Spread Algo':                                      'TV_GoldSpread',
    'Liquid Pulse':                                          'TV_LiquidPulse',
    'Maximized Scalping On Trend (by Coinrule)':             'TV_MaxScalpTrend',

    # Batch269 — conversiones 2026-04-05
    'Donchian x WMA Crossover (2025 Only, Adjustable TP, Real OHLC)': 'TV_DonchianWMA',
    'Bollinger Bands + RSI Double Strategy (by SlumdogTrader) ': 'TV_BBRSIDouble',
    'Bollinger Bands + RSI Double Strategy (by SlumdogTrader)':  'TV_BBRSIDouble',
    'VFI  strategy [based on VFI indicator published by  UTS]':  'TV_VFI',
    'Best TradingView Strategy - For NASDAQ and DOW30 and other Index': 'TV_BestTVIndex',

    # Batch270 — conversiones 2026-04-05
    'Trade Entry Detector, Wick to Body Ratio ':                 'TV_WickBodyRatio',
    'Trade Entry Detector, Wick to Body Ratio':                  'TV_WickBodyRatio',
    'Flawless Victory Strategy - 15min BTC Machine Learning Strategy': 'TV_FlawlessVic',
    'PowerZone Trading Strategy':                                'TV_PowerZone',
    'RSI Strategy':                                              'TV_RSIStrategy',

    # Batch271 — conversiones 2026-04-05
    'Divergence for Many Indicators v4 ST':                      'TV_DivSuperTrend',
    'Fibonacci Counter-Trend Trading':                           'TV_FibCounter',
    'MACD Bull Crossover and RSI Oversold 5 Candles Ago-Long Strategy': 'TV_MACDRSILag',
    'RSI TrueLevel Strategy':                                    'TV_RSITrueLevel',

    # Batch272 — conversiones 2026-04-05
    'Bollinger Bands And Aroon Scalping (by Coinrule)':          'TV_BBAroon',
    'Heatmap MACD Strategy - Pineconnector (Dynamic Alerts)':    'TV_HeatmapMACD',
    'Classic Nacked Z-Score Arbitrage':                          'TV_ZScoreArb',
    'Solana 4H RSI->MACD — Counter-Trend By Tetrad':             'TV_SolRSIMACD',

    # Batch273 — conversiones 2026-04-05
    'Optimized Range + RSI + MACD + Fibonacci + Risk Mgmt':      'TV_OptRangeFib',
    'ETH Momentum Breakout Strategy [geektrade.online]':          'TV_ETHMomentum',
    'WLD 4H Auto Entry Confirmed':                               'TV_WLD4H',
    'Strategy Template':                                         'TV_StratTemplate',

    # Batch274 — conversiones 2026-04-05
    'Warm_Up':                                                   'TV_WarmUp',
    'The Impeccable by zyberal':                                 'TV_Impeccable',
    'FibSniper Pro: SMC Edition':                                'TV_FibSniperSMC',
    'VIP':                                                       'TV_VIP',

    # Batch275 — conversiones 2026-04-05
    'Dump Reversal Peak Trail v2':                               'TV_DumpReversal',
    'WJ Strategy B':                                             'TV_WJStratB',
    'WJ BUY Strategy':                                           'TV_WJStratB',
    '[VJ]Thor for MFI':                                          'TV_VJThorMFI',
    'NQ Scalping Strategy (EMA + RSI + ATR) [Entries/Exits]':   'TV_NQScalp',
    'NQ Scalping Strategy (EMA + RSI + ATR)':                   'TV_NQScalp',

    # Batch276 — conversiones 2026-04-05
    'Big Runner':                                                'TV_BigRunner',
    'Ergodic CSI Backtest':                                      'TV_ErgodicCSI',
    'Momentum Upon Clearing Candle Wick':                        'TV_MomentumWick',

    # Batch277 — conversiones 2026-04-05
    '8 Whittle Down':                                            'TV_WhittleDown',
}

# ── Symbol mapping ────────────────────────────────────────────────────────────
def tv_to_db_symbol(tv_sym: str) -> str:
    """BTCUSDT → BTC/USDT:USDT"""
    base = tv_sym.replace('USDT', '')
    return f'{base}/USDT:USDT'

# ── TF resampling ─────────────────────────────────────────────────────────────
def resample_tf(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample 1h→4h or 5m→15m from base candles"""
    rules = {'4h': '4h', '15m': '15min', '1h': '1h', '5m': '5min', '1d': '1D'}
    rule = rules.get(target_tf)
    if rule is None:
        return df
    resampled = df.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    return resampled

# ── Load candles ──────────────────────────────────────────────────────────────
_candle_cache = {}

def load_candles(db_sym: str, tf: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """Load candles from DB, resample if needed (4h/15m)"""
    key = (db_sym, tf)
    if key in _candle_cache:
        return _candle_cache[key]

    base_tf = '1h' if tf == '4h' else ('5m' if tf == '15m' else tf)

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            'SELECT CAST(ts AS INTEGER) as ts, open, high, low, close, volume '
            'FROM candles WHERE symbol=? AND timeframe=? ORDER BY ts',
            conn, params=(db_sym, base_tf)
        )
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df.set_index('ts', inplace=True)
    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})

    if tf in ('4h', '15m'):
        df = resample_tf(df, tf)

    _candle_cache[key] = df
    return df

# ── Strategy loader ───────────────────────────────────────────────────────────
_py_strats = None

def load_python_strategies() -> dict:
    """Load all STRATEGY_EXPORT from batches 100+"""
    global _py_strats
    if _py_strats is not None:
        return _py_strats

    strats = {}
    files = sorted(glob.glob(f'{BATCH_DIR}/strategies_tv2_batch[0-9]*.py'))
    files = [f for f in files if 'bak' not in f]

    for fpath in files:
        m = re.search(r'batch(\d+)', fpath)
        if not m or int(m.group(1)) < 100:
            continue  # skip EMA stubs (batches 80-99)
        bname = fpath.split('/')[-1].replace('.py', '')
        try:
            spec = importlib.util.spec_from_file_location(bname, fpath)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            exports = getattr(mod, 'STRATEGY_EXPORT', {})
            strats.update(exports)
        except Exception:
            pass

    _py_strats = strats
    return strats

# ── Backtest ──────────────────────────────────────────────────────────────────
def _atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Compute ATR array"""
    high = df['high'].values
    low  = df['low'].values
    close = df['close'].values
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low  - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.zeros(len(tr))
    atr[:period] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def backtest_signals(df: pd.DataFrame, gen_func, params=None, max_hold: int = MAX_HOLD):
    """
    Run gen_func on df with given params, then simulate:

    Auto-detect exit mode:
    - Dense signals (≥5% of bars): hold until opposite signal OR max_hold bars
    - Sparse signals (<5% of bars): use ATR SL/TP (closer to TV strategy.exit() behavior)

    Returns: {'wr': float, 'trades': int, 'pnl': float, 'n_long': int, 'n_short': int, 'exit_mode': str}
    """
    if df.empty or len(df) < 50:
        return None

    params = params or {}
    try:
        signals = gen_func(df, **params)
    except Exception as e:
        return {'error': str(e)}

    if signals is None:
        return None

    sig = signals.values if hasattr(signals, 'values') else np.array(signals)
    close = df['close'].values
    high  = df['high'].values
    low   = df['low'].values
    n = len(close)

    # Detect signal density
    n_signals = np.sum(sig != 0)
    sig_density = n_signals / n
    use_atr_exit = (sig_density < MIN_SIG_DENSITY)
    exit_mode = 'atr_sl_tp' if use_atr_exit else 'opposite_signal'

    atr_vals = _atr(df) if use_atr_exit else None

    trades = []
    pos = 0          # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_bar = 0
    sl_price = 0.0
    tp_price = 0.0
    prev_s = 0

    for i in range(1, n):
        s = int(sig[i]) if i < len(sig) else 0

        if pos != 0:
            bars_held = i - entry_bar

            # Priority 1: ATR SL/TP hit (only for sparse-signal ATR mode)
            if use_atr_exit:
                if pos == 1 and low[i] <= sl_price:
                    pnl = (sl_price / entry_price - 1.0) - COMMISSION * 2
                    trades.append({'pnl': pnl, 'dir': pos, 'exit': 'sl'})
                    pos = 0; prev_s = s; continue
                elif pos == 1 and high[i] >= tp_price:
                    pnl = (tp_price / entry_price - 1.0) - COMMISSION * 2
                    trades.append({'pnl': pnl, 'dir': pos, 'exit': 'tp'})
                    pos = 0; prev_s = s; continue
                elif pos == -1 and high[i] >= sl_price:
                    pnl = (entry_price / sl_price - 1.0) - COMMISSION * 2
                    trades.append({'pnl': pnl, 'dir': pos, 'exit': 'sl'})
                    pos = 0; prev_s = s; continue
                elif pos == -1 and low[i] <= tp_price:
                    pnl = (entry_price / tp_price - 1.0) - COMMISSION * 2
                    trades.append({'pnl': pnl, 'dir': pos, 'exit': 'tp'})
                    pos = 0; prev_s = s; continue

            # Priority 2: Explicit exit (sig returns to 0 from non-zero = state-based exit)
            #             OR opposite signal
            sig_exit   = (prev_s != 0 and s == 0)
            opp_exit   = (pos == 1 and s == -1) or (pos == -1 and s == 1)
            time_exit  = (bars_held >= max_hold)

            if sig_exit or opp_exit or time_exit:
                if pos == 1:
                    pnl = (close[i] / entry_price - 1.0) - COMMISSION * 2
                else:
                    pnl = (entry_price / close[i] - 1.0) - COMMISSION * 2
                exit_label = 'sig0' if sig_exit else ('opp' if opp_exit else 'time')
                trades.append({'pnl': pnl, 'dir': pos, 'exit': exit_label})
                pos = 0

        prev_s = s

        if pos == 0 and s in (1, -1):
            pos = s
            entry_price = close[i]
            entry_bar = i
            if use_atr_exit and atr_vals is not None:
                atr_val = atr_vals[i]
                if pos == 1:
                    sl_price = entry_price - ATR_SL_MULT * atr_val
                    tp_price = entry_price + ATR_TP_MULT * atr_val
                else:
                    sl_price = entry_price + ATR_SL_MULT * atr_val
                    tp_price = entry_price - ATR_TP_MULT * atr_val

    if len(trades) < 5:
        return None

    wins  = sum(1 for t in trades if t['pnl'] > 0)
    wr    = wins / len(trades) * 100
    pnl   = sum(t['pnl'] for t in trades) * 100
    n_long  = sum(1 for t in trades if t['dir'] == 1)
    n_short = sum(1 for t in trades if t['dir'] == -1)

    return {
        'wr':        round(wr, 1),
        'trades':    len(trades),
        'pnl':       round(pnl, 2),
        'n_long':    n_long,
        'n_short':   n_short,
        'exit_mode': exit_mode,
        'sig_density': round(sig_density * 100, 1),
    }

# ── Name matcher ──────────────────────────────────────────────────────────────
def find_py_key(tv_name: str, py_strats: dict):
    """Look up Python key for a TV strategy name"""
    # 1. Hard-coded mapping (highest confidence)
    if tv_name in TV_NAME_TO_PY:
        key = TV_NAME_TO_PY[tv_name]
        if key in py_strats:
            return key

    # 2. Partial match (TV name STARTS with or CONTAINS the hard-coded key)
    for tv_known, py_key in TV_NAME_TO_PY.items():
        if tv_name.startswith(tv_known[:20]) or tv_known.startswith(tv_name[:20]):
            if py_key in py_strats:
                return py_key

    # 3. Try "TV_" + CamelCase from TV name (e.g. "Flawless Victory" → TV_FlawlessVictory)
    clean = re.sub(r'[^a-zA-Z0-9 ]', '', tv_name).title()
    candidate = 'TV_' + clean.replace(' ', '')[:20]
    for key in py_strats:
        if key.lower() == candidate.lower():
            return key

    # 4. Token overlap fuzzy match
    def tokenize(s):
        return set(re.sub(r'[^a-z0-9]', ' ', s.lower()).split())

    tv_tok = tokenize(tv_name)
    best_score, best_key = 0, None

    for py_key in py_strats:
        py_clean = re.sub(r'([A-Z])', r' \1', py_key.replace('TV_', '')).strip()
        py_tok   = tokenize(py_clean)
        inter    = tv_tok & py_tok
        score    = len(inter) / max(len(tv_tok | py_tok), 1)
        if score > best_score:
            best_score, best_key = score, py_key

    if best_score >= 0.30:
        return best_key

    return None

# ── Per-combo validation ───────────────────────────────────────────────────────
def validate_combo(args):
    """Validate one (tv_strategy, py_key, combo) — safe for subprocess"""
    tv_name, py_key, combo, db_path = args

    tv_sym = combo['symbol']
    tf     = combo['tf']
    tv_wr  = combo['wr']

    db_sym = tv_to_db_symbol(tv_sym)

    # Reload strategy (needed in subprocess)
    strats = load_python_strategies()
    if py_key not in strats:
        return {'tv_name': tv_name, 'py_key': py_key, 'tv_sym': tv_sym, 'tf': tf,
                'tv_wr': tv_wr, 'status': 'NO_IMPL', 'error': f'{py_key} not in strats'}

    gen_info = strats[py_key]
    gen_func = gen_info['gen'] if isinstance(gen_info, dict) else gen_info

    # Load candles
    df = load_candles(db_sym, tf, db_path)
    if df.empty:
        return {'tv_name': tv_name, 'py_key': py_key, 'tv_sym': tv_sym, 'tf': tf,
                'tv_wr': tv_wr, 'status': 'NO_DATA', 'error': f'No candles for {db_sym} {tf}'}

    # Backtest
    result = backtest_signals(df, gen_func)

    if result is None:
        return {'tv_name': tv_name, 'py_key': py_key, 'tv_sym': tv_sym, 'tf': tf,
                'tv_wr': tv_wr, 'status': 'FEW_TRADES', 'error': 'Not enough trades'}

    if 'error' in result:
        return {'tv_name': tv_name, 'py_key': py_key, 'tv_sym': tv_sym, 'tf': tf,
                'tv_wr': tv_wr, 'status': 'ERROR', 'error': result['error']}

    py_wr = result['wr']
    gap   = abs(py_wr - tv_wr)

    if gap <= GAP_PASS:
        status = 'PASS'
    elif gap <= GAP_WARN:
        status = 'WARNING'
    else:
        status = 'FAIL'

    return {
        'tv_name':  tv_name,
        'py_key':   py_key,
        'tv_sym':   tv_sym,
        'tf':       tf,
        'tv_wr':    tv_wr,
        'py_wr':    py_wr,
        'gap':      round(gap, 1),
        'status':   status,
        'trades':   result['trades'],
        'pnl':      result['pnl'],
        'n_long':   result['n_long'],
        'n_short':  result['n_short'],
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default=None, help='Only validate this py_key')
    parser.add_argument('--limit',    type=int, default=0, help='Max W60 strategies to process')
    parser.add_argument('--workers',  type=int, default=4)
    args = parser.parse_args()

    print(f'\n{"="*70}')
    print(f'  TV→Python WR Validator  [{datetime.now().strftime("%Y-%m-%d %H:%M")}]')
    print(f'{"="*70}\n')

    # Load W60
    with open(W60_JSON) as f:
        w60 = json.load(f)
    w60_strats = w60['strategies']
    if args.limit:
        w60_strats = w60_strats[:args.limit]

    # Load Python strategies
    print('Cargando estrategias Python (batches 100+)...')
    py_strats = load_python_strategies()
    print(f'  {len(py_strats)} estrategias Python disponibles\n')

    # Build validation tasks
    tasks = []
    no_impl = []
    matched_strats = set()

    for tv in w60_strats:
        tv_name = tv['strategy']
        py_key  = find_py_key(tv_name, py_strats) if not args.strategy else (
            args.strategy if args.strategy in py_strats else None
        )

        if py_key is None:
            no_impl.append(tv_name)
            continue

        matched_strats.add(tv_name)
        for combo in tv['combos']:
            if combo.get('wr', 0) >= 60:  # only validate combos that TV says WR>=60%
                tasks.append((tv_name, py_key, combo, DB_PATH))

    print(f'W60 strategies: {len(w60_strats)}')
    print(f'  Con Python impl: {len(matched_strats)}')
    print(f'  Sin Python impl: {len(no_impl)}')
    print(f'  Combos a validar: {len(tasks)}\n')

    if not tasks:
        print('No hay combos para validar.')
        return

    # Run validation
    results = []
    t0 = time.time()

    # Run single-threaded (gen functions need module state)
    print('Corriendo backtests...')
    for i, task in enumerate(tasks):
        r = validate_combo(task)
        results.append(r)
        if (i+1) % 10 == 0 or (i+1) == len(tasks):
            elapsed = time.time() - t0
            print(f'  [{i+1}/{len(tasks)}] {elapsed:.1f}s')

    # ── Report ────────────────────────────────────────────────────────────────
    pass_list    = [r for r in results if r.get('status') == 'PASS']
    warn_list    = [r for r in results if r.get('status') == 'WARNING']
    fail_list    = [r for r in results if r.get('status') == 'FAIL']
    error_list   = [r for r in results if r.get('status') in ('ERROR', 'FEW_TRADES', 'NO_DATA')]

    print(f'\n{"="*70}')
    print(f'  RESULTADOS VALIDACIÓN')
    print(f'{"="*70}')
    print(f'  Total combos:  {len(results)}')
    print(f'  PASS  (≤{GAP_PASS:g}pp):  {len(pass_list):3d}  ({len(pass_list)/len(results)*100:.0f}%)')
    print(f'  WARN  (≤{GAP_WARN:g}pp):  {len(warn_list):3d}  ({len(warn_list)/len(results)*100:.0f}%)')
    print(f'  FAIL  (>{GAP_WARN:g}pp):  {len(fail_list):3d}  ({len(fail_list)/len(results)*100:.0f}%)')
    print(f'  ERROR:        {len(error_list):3d}')
    print(f'\n  Sin impl Python: {len(no_impl)} estrategias')

    # Per-strategy summary
    print(f'\n{"─"*70}')
    print(f'  DETALLE POR ESTRATEGIA')
    print(f'{"─"*70}')

    # Group by tv_name
    from collections import defaultdict
    by_strat = defaultdict(list)
    for r in results:
        by_strat[r['tv_name']].append(r)

    rows = []
    for tv_name, combos in sorted(by_strat.items()):
        py_key = combos[0].get('py_key', '?')
        n_pass  = sum(1 for c in combos if c.get('status') == 'PASS')
        n_total = len(combos)
        avg_gap = np.mean([c.get('gap', 0) for c in combos if 'gap' in c]) if any('gap' in c for c in combos) else None
        tv_wrs  = [c.get('tv_wr', 0) for c in combos]
        py_wrs  = [c.get('py_wr', 0) for c in combos if 'py_wr' in c]
        status  = 'PASS' if n_pass == n_total else ('PARTIAL' if n_pass > 0 else 'FAIL')
        rows.append({
            'tv_name': tv_name[:45],
            'py_key':  py_key,
            'n_pass':  n_pass,
            'n_total': n_total,
            'avg_gap': avg_gap,
            'avg_tv_wr': round(np.mean(tv_wrs), 1) if tv_wrs else 0,
            'avg_py_wr': round(np.mean(py_wrs), 1) if py_wrs else 0,
            'status':  status,
        })

    rows.sort(key=lambda x: (x['status'] != 'PASS', -(x['avg_tv_wr'] or 0)))

    for r in rows:
        gap_str = f"gap={r['avg_gap']:.1f}pp" if r['avg_gap'] is not None else 'n/a'
        print(f"  [{r['status']:7s}] {r['tv_name']:45s} | {r['py_key']:25s} | "
              f"TV={r['avg_tv_wr']:5.1f}% PY={r['avg_py_wr']:5.1f}% {gap_str} "
              f"({r['n_pass']}/{r['n_total']} combos)")

    # Detail on FAILs
    if fail_list:
        print(f'\n{"─"*70}')
        print('  FAILS — Análisis causa raíz:')
        print(f'{"─"*70}')
        for r in sorted(fail_list, key=lambda x: -x.get('gap', 0)):
            print(f"  {r['tv_name'][:40]:40s} | {r['py_key']:25s} | "
                  f"{r['tv_sym']:15s} {r['tf']:4s} | "
                  f"TV={r['tv_wr']:5.1f}% PY={r.get('py_wr',0):5.1f}% gap={r.get('gap',0):.1f}pp")
            # Root cause hints
            if r.get('py_wr', 0) == 0 or r.get('trades', 0) < 10:
                print(f"    → Causa: pocos trades ({r.get('trades',0)}) — señales no funcionan en este activo")
            elif r.get('py_wr', 100) < 40:
                print(f"    → Causa: WR Python muy bajo — posible lógica invertida o parámetros defaults malos")
            elif r.get('gap', 0) > 30:
                print(f"    → Causa: gap grande — verificar que señales de entrada/salida son equivalentes a TV")

    # Strategies without Python impl
    if no_impl:
        print(f'\n{"─"*70}')
        print(f'  {len(no_impl)} ESTRATEGIAS SIN IMPLEMENTACIÓN PYTHON:')
        print(f'{"─"*70}')
        for tv in w60_strats:
            if tv['strategy'] in no_impl:
                print(f"  WR={tv['best_wr']:5.1f}%  {tv['strategy'][:65]}")

    # Save results
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_path = f'{RESULTS_DIR}/validation_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'generated': datetime.now().isoformat(),
            'total_w60': len(w60_strats),
            'matched':   len(matched_strats),
            'no_impl':   len(no_impl),
            'combos_validated': len(results),
            'pass':   len(pass_list),
            'warn':   len(warn_list),
            'fail':   len(fail_list),
            'errors': len(error_list),
            'no_impl_list': no_impl,
            'results': results,
        }, f, indent=2)

    print(f'\n  Guardado en: {out_path}')
    print(f'  Tiempo total: {time.time()-t0:.1f}s')
    print()

if __name__ == '__main__':
    main()
