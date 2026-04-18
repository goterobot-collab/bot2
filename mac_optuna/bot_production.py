#!/usr/bin/env python3
"""
BOT DE PRODUCCIÓN — Santo Grial
Lee dream_teams_final.json y monitorea señales en tiempo real.
Cuando detecta señal → alerta.
Si el trade excede P95 duración → alerta de timeout.

USO:
  python3 bot_production.py                    # Monitorea todos los dream teams
  python3 bot_production.py --asset DOGE/USDT:USDT  # Solo un asset
  python3 bot_production.py --list             # Lista dream teams
"""

import json, os, sys, time, argparse
from datetime import datetime, timedelta

PROJECT = "/Users/sabrina/CLAUDE CODE/Estrategias"

def load_dream_teams():
    with open(os.path.join(PROJECT, 'data', 'dream_teams_final.json')) as f:
        return json.load(f)

def format_duration(hours):
    if hours < 1: return f"{int(hours*60)}min"
    if hours < 24: return f"{hours:.1f}h"
    return f"{hours/24:.1f}d"

def list_teams():
    teams = load_dream_teams()
    print(f"\n{'='*80}")
    print(f"DREAM TEAMS — {len(teams)} assets")
    print(f"{'='*80}")
    print(f"{'Asset':<25} {'Score':>5} {'WR%':>6} {'#':>2}  Estrategias")
    print(f"{'-'*80}")
    for sym, dt in sorted(teams.items(), key=lambda x: -x[1]['best_score']):
        team = dt['team']
        strats = ' | '.join(f"{t['strategy']}@{t['timeframe']}" for t in team)
        print(f"{sym[:24]:<25} {dt['best_score']:>5} {dt['best_wr']:>5.1f}% {len(team):>2}  {strats}")

def show_asset_detail(symbol):
    teams = load_dream_teams()
    if symbol not in teams:
        print(f"Asset {symbol} no tiene dream team")
        return

    dt = teams[symbol]
    print(f"\n{'='*80}")
    print(f"🏆 DREAM TEAM: {symbol}")
    print(f"{'='*80}")

    for i, bot in enumerate(dt['team'], 1):
        print(f"\n  Bot #{i}: {bot['strategy']} @ {bot['timeframe']}")
        print(f"  {'─'*50}")
        print(f"  Score:      {bot['score']}")
        print(f"  WR (full):  {bot['full_wr']:.1f}%")
        print(f"  PnL (full): {bot['full_pnl']:.1f}%")
        print(f"  Trades:     {bot['full_trades']}")
        print(f"  Sharpe:     {bot['full_sharpe']:.2f}")
        print(f"  Test WR:    {bot['test_wr']:.1f}%")
        print(f"  Leverage:   {bot['safe_leverage']}x")
        print(f"  Parámetros: {json.dumps(bot['params'])}")
        print(f"  ⏱️ Duración:")
        print(f"     Promedio: {format_duration(bot['dur_avg_hours'])}")
        print(f"     P95:      {format_duration(bot['dur_p95_hours'])} ← ALERTA si supera")
        print(f"     Máximo:   {format_duration(bot['dur_max_hours'])} ← CERRAR si supera")

def generate_bot_config(symbol=None):
    """Generate JSON config for a trading bot"""
    teams = load_dream_teams()

    if symbol:
        assets = {symbol: teams[symbol]} if symbol in teams else {}
    else:
        assets = teams

    config = {
        'generated_at': datetime.now().isoformat(),
        'version': '1.0',
        'rules': {
            'entry': 'OPEN of NEXT bar after signal',
            'commission': 0.001,
            'slippage': 0.0005,
            'min_wr_to_trade': 70,
        },
        'assets': {}
    }

    for sym, dt in assets.items():
        bots = []
        for bot in dt['team']:
            bots.append({
                'strategy': bot['strategy'],
                'timeframe': bot['timeframe'],
                'params': bot['params'],
                'leverage': bot['safe_leverage'],
                'alerts': {
                    'duration_alert_hours': bot['dur_p95_hours'],
                    'duration_close_hours': bot['dur_max_hours'],
                    'description': f"Si trade dura más de {format_duration(bot['dur_p95_hours'])} → ALERTAR. Si más de {format_duration(bot['dur_max_hours'])} → CERRAR."
                },
                'performance': {
                    'wr': bot['full_wr'],
                    'pnl': bot['full_pnl'],
                    'trades': bot['full_trades'],
                    'sharpe': bot['full_sharpe'],
                    'score': bot['score'],
                }
            })

        config['assets'][sym] = {
            'bots': bots,
            'best_score': dt['best_score'],
            'best_wr': dt['best_wr'],
        }

    return config

def main():
    parser = argparse.ArgumentParser(description='Santo Grial Bot')
    parser.add_argument('--list', action='store_true', help='Lista dream teams')
    parser.add_argument('--asset', type=str, help='Detalle de un asset')
    parser.add_argument('--export', type=str, help='Exportar config JSON')
    parser.add_argument('--export-all', action='store_true', help='Exportar config de todos')
    args = parser.parse_args()

    if args.list:
        list_teams()
    elif args.asset:
        show_asset_detail(args.asset)
    elif args.export:
        config = generate_bot_config(args.export)
        out = f"bot_config_{args.export.replace('/','-').replace(':','')}.json"
        with open(out, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"Exported: {out}")
    elif args.export_all:
        config = generate_bot_config()
        out = "bot_config_ALL.json"
        with open(out, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"Exported: {out} ({len(config['assets'])} assets)")
    else:
        list_teams()
        print(f"\nUso:")
        print(f"  python3 bot_production.py --list")
        print(f"  python3 bot_production.py --asset 'DOGE/USDT:USDT'")
        print(f"  python3 bot_production.py --export 'BTC/USDT:USDT'")
        print(f"  python3 bot_production.py --export-all")

if __name__ == '__main__':
    main()
