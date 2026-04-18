#!/usr/bin/env python3
"""
EXTRACTOR DE TRADES INDIVIDUALES — 1000 Bots de Producción V6
Formato compatible con V5: entry, exit, PnL, duración, tipo cierre, MAE

Output: data/v6_all_trades.json
"""
import sqlite3, pandas as pd, numpy as np, json, os, sys, time, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

DB_PATH = "/Users/sabrina/Code/activos binace /activos_binance.db"
PROJECT = "/Users/sabrina/CLAUDE CODE/Estrategias"
COMM = 0.001
SLIP = 0.0005

sys.path.insert(0, PROJECT)

# Load strategies (same as turbo_worker but without starting the worker)
STRATS = {}
try:
    from strategy_factory import FACTORY_STRATS
    STRATS.update(FACTORY_STRATS)
except Exception as e:
    print(f"Warning loading factory: {e}")
try:
    from strategies_round2 import STRATEGY_TYPES_R2
    STRATS.update(STRATEGY_TYPES_R2)
except Exception as e:
    print(f"Warning loading round2: {e}")

def load_candles(symbol, tf="5m"):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT ts,open,high,low,close,volume FROM candles WHERE symbol=? AND timeframe=? ORDER BY ts", conn, params=(symbol, tf))
        if len(df) == 0:
            df = pd.read_sql(f"SELECT ts,open,high,low,close,volume FROM candles_{tf} WHERE symbol=? ORDER BY ts", conn, params=[symbol])
        if len(df) < 100: return None
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        for c in ['open','high','low','close','volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna(subset=['close'])
    except: return None
    finally: conn.close()

def resample(df, t):
    m = {'15m':'15min','4h':'4h','1d':'1D'}
    r = m.get(t)
    if not r: return None
    return df.resample(r).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['close'])

def extract_trades_detailed(df, signals, bot):
    """Extract each trade with REAL stop loss, take profit, and max duration.

    Protections (same as production bot):
    1. Stop Loss: final_stop_loss_pct from MAE (mae_worst × 2)
    2. Max Duration: p95_h × 1.5 — forced exit if exceeded
    3. Emergency Stop: -50% drawdown = immediate exit (anti-liquidation)
    """
    if signals is None: return []
    sig = signals.values; opens = df['open'].values
    highs = df['high'].values; lows = df['low'].values; closes = df['close'].values
    n = len(df)
    trades = []; pos = 0; ep = 0.; ei = 0; mae = 0.; mfe = 0.; pnl_acum = 0.
    lev = bot.get('final_leverage', 5); size = bot.get('position_size_usd', 10)

    # Risk parameters from production config
    sl_pct = bot.get('final_stop_loss_pct', 0)
    if sl_pct <= 0:
        # Fallback: use mae_worst × 2 as stop loss
        mae_worst = bot.get('mae_worst', 0.10)
        sl_pct = mae_worst * 2.0
    sl_pct = max(sl_pct, 0.02)  # minimum 2% SL
    sl_pct = min(sl_pct, 0.40)  # maximum 40% SL (never let it go beyond)

    # Max duration from bot's historical p95
    dur_data = bot.get('duration', {})
    p95_h = dur_data.get('p95_h', 0) if isinstance(dur_data, dict) else 0
    max_hours = p95_h * 1.5 if p95_h > 0 else 720  # default 30 days max
    max_hours = min(max_hours, 720)  # HARD CAP: never more than 30 days

    EMERGENCY_STOP = -0.50  # -50% = emergency exit (anti-liquidation)

    def close_trade(i, exit_price, exit_type_label):
        nonlocal pos, pnl_acum
        xp = exit_price * (1 - SLIP - COMM)
        pnl_pct = (xp - ep) / ep * 100
        pnl_usd = pnl_pct / 100 * size * lev
        pnl_acum += pnl_usd
        dur_h = (df.index[i] - df.index[ei]).total_seconds() / 3600
        dur_str = f"{int(dur_h)}h {int((dur_h % 1) * 60)}m"
        trades.append({
            'bot': bot['strategy'],
            'symbol': bot['symbol'].replace('/USDT:USDT', ''),
            'symbol_full': bot['symbol'],
            'timeframe': bot['timeframe'],
            'tier': bot.get('tier', '?'),
            'side': 'LONG',
            'entry_price': round(float(ep / (1+SLIP+COMM)), 8),
            'entry_time': str(df.index[ei])[:19],
            'exit_price': round(float(exit_price), 8),
            'exit_time': str(df.index[i])[:19],
            'pnl_usd': round(pnl_usd, 2),
            'pnl_pct': round(pnl_pct, 2),
            'size': size,
            'exit_type': exit_type_label,
            'leverage': lev,
            'duration_hours': round(dur_h, 1),
            'duration_str': dur_str,
            'mae_pct': round(mae * 100, 2),
            'mfe_pct': round(mfe * 100, 2),
            'pnl_acum': round(pnl_acum, 2),
            'result': 'WIN' if pnl_pct > 0 else 'LOSS',
            'year': df.index[ei].year,
        })
        pos = 0

    for i in range(1, n):
        s = sig[i-1]; p = opens[i]
        if pos == 0 and s == 1:
            ep = p * (1 + SLIP + COMM); ei = i; mae = 0.; mfe = 0.; pos = 1
        elif pos == 1:
            lp = (lows[i] - ep) / ep
            hp = (highs[i] - ep) / ep
            if lp < mae: mae = lp
            if hp > mfe: mfe = hp

            # Current drawdown from entry
            current_dd = (closes[i] - ep) / ep
            dur_h = (df.index[i] - df.index[ei]).total_seconds() / 3600

            # CHECK 1: Emergency stop (-50% drawdown)
            if lp <= EMERGENCY_STOP:
                # Exit at the emergency stop price (intrabar)
                emergency_price = ep * (1 + EMERGENCY_STOP)
                close_trade(i, emergency_price, 'emergency')
            # CHECK 2: Stop Loss hit (intrabar low touches SL)
            elif lp <= -sl_pct:
                sl_price = ep * (1 - sl_pct)
                close_trade(i, sl_price, 'sl')
            # CHECK 3: Max duration exceeded
            elif dur_h >= max_hours:
                close_trade(i, closes[i], 'timeout')
            # CHECK 4: Strategy signal to exit
            elif s == -1:
                close_trade(i, p, 'signal')
            # Otherwise: still in trade, continue

    return trades

def main():
    print("=" * 60)
    print("📋 EXTRACTOR DE TRADES V6 — Formato V5")
    print("=" * 60)

    prod = json.load(open(os.path.join(PROJECT, 'data/v6_production_final.json')))
    print(f"Bots: {len(prod)}")

    # Progress
    progress_file = os.path.join(PROJECT, 'data/extract_progress.json')
    done_keys = set(); all_trades = []
    if os.path.exists(progress_file):
        pg = json.load(open(progress_file))
        done_keys = set(pg.get('done', []))
        all_trades = pg.get('trades', [])
        print(f"Resuming: {len(done_keys)} done, {len(all_trades)} trades")

    # Group by symbol for efficient loading
    by_symbol = defaultdict(list)
    for bot in prod:
        by_symbol[bot['symbol']].append(bot)

    t0 = time.time(); processed = 0; errors = 0
    for sym_idx, (symbol, bots) in enumerate(by_symbol.items()):
        # Skip if all bots for this symbol are done
        pending = [b for b in bots if f"{b['strategy']}|{symbol}|{b['timeframe']}" not in done_keys]
        if not pending: continue

        # Load all TFs needed
        tfs_needed = set(b['timeframe'] for b in pending)
        df_5m = load_candles(symbol, '5m') if any(t in ('5m','15m','4h') for t in tfs_needed) else None
        dfs = {}
        if '5m' in tfs_needed and df_5m is not None: dfs['5m'] = df_5m
        if '15m' in tfs_needed and df_5m is not None: dfs['15m'] = resample(df_5m, '15m')
        if '4h' in tfs_needed and df_5m is not None: dfs['4h'] = resample(df_5m, '4h')
        if '1h' in tfs_needed: dfs['1h'] = load_candles(symbol, '1h')
        if '1d' in tfs_needed:
            d1d = load_candles(symbol, '1d')
            dfs['1d'] = d1d if d1d is not None else (resample(df_5m, '1d') if df_5m is not None else None)

        for bot in pending:
            key = f"{bot['strategy']}|{symbol}|{bot['timeframe']}"
            tf = bot['timeframe']
            strat_name = bot['strategy']
            params = bot.get('params', {})
            df = dfs.get(tf)

            if df is None or len(df) < 200 or strat_name not in STRATS:
                done_keys.add(key); errors += 1; continue

            try:
                gen = STRATS[strat_name]['gen']
                sp = int(len(df) * 0.7)
                test_df = df.iloc[sp:].copy()
                signals = gen(test_df, **params)
                trades = extract_trades_detailed(test_df, signals, bot)
                all_trades.extend(trades)
                processed += 1
            except:
                errors += 1

            done_keys.add(key)

        if (sym_idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{sym_idx+1}/{len(by_symbol)}] {len(all_trades)} trades | {processed} bots | {elapsed:.0f}s")

        # Save progress every 50 symbols
        if (sym_idx + 1) % 50 == 0:
            with open(progress_file, 'w') as f:
                json.dump({'done': list(done_keys), 'trades': all_trades}, f)

    # Sort by entry_time descending
    all_trades.sort(key=lambda t: t.get('entry_time', ''), reverse=True)
    for i, t in enumerate(all_trades):
        t['trade_id'] = len(all_trades) - i

    # Stats
    wins = sum(1 for t in all_trades if t['result'] == 'WIN')
    losses = len(all_trades) - wins
    total_pnl = sum(t['pnl_usd'] for t in all_trades)
    wr = wins / max(len(all_trades), 1) * 100

    output = {
        'total_trades': len(all_trades),
        'wins': wins, 'losses': losses,
        'win_rate': round(wr, 1),
        'total_pnl_usd': round(total_pnl, 2),
        'avg_pnl_usd': round(total_pnl / max(len(all_trades), 1), 4),
        'total_bots': len(prod),
        'bots_processed': processed + len(done_keys),
        'errors': errors,
        'note': 'BACKTEST walk-forward OOS (test 30%), NO live trading',
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'trades': all_trades,
    }

    out_path = os.path.join(PROJECT, 'data/v6_all_trades.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=1)

    # Cleanup progress
    with open(progress_file, 'w') as f:
        json.dump({'done': list(done_keys), 'trades': all_trades}, f)

    print(f"\n{'='*60}")
    print(f"✅ EXTRACCIÓN COMPLETADA")
    print(f"{'='*60}")
    print(f"  Total trades:  {len(all_trades):,}")
    print(f"  Wins:          {wins:,} ({wr:.1f}%)")
    print(f"  Losses:        {losses:,}")
    print(f"  Total PnL:     ${total_pnl:,.2f}")
    print(f"  Bots:          {processed} OK | {errors} errors")
    print(f"  💾 {out_path} ({os.path.getsize(out_path)//1024} KB)")

if __name__ == '__main__':
    main()
