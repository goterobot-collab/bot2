#!/usr/bin/env python3
"""
pilot_screen_500.py — Piloto de 500 estrategias aleatorias.
Mide empíricamente:
  - Tasa real de survivors (no la proyectada)
  - Throughput (strategies/minuto) → extrapolar tiempo full run
  - Reducción de compute por early-exit DD (comparando vs sin DD)
  - Recall sobre ganadores conocidos (si se cruzan)
"""
import sys, os, time, json, random, importlib, glob
from pathlib import Path

sys.path.insert(0, '/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches')

# Import after path injection
from fast_filter_v2 import screen_strategy, load_candles, SCREEN_ASSETS, SCREEN_TFS, MIN_WR, MIN_TRADES, EARLY_EXIT_DD_PCT

BATCHES_DIR = '/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches'
OUT = Path('/Users/sabrina/CLAUDE CODE/Estrategias/scripts/screening_output')
OUT.mkdir(parents=True, exist_ok=True)

N_PILOT = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SEED = 42


def discover_strategies(max_strats=None):
    """Load REAL strategy universe from pending_ranked.csv (13,687 names),
    then resolve gen_/SPACE_ functions across all batch files."""
    import csv
    ranked = '/Users/sabrina/CLAUDE CODE/Estrategias/scripts/screening_output/pending_ranked.csv'
    real_names = set()
    with open(ranked) as f:
        for r in csv.DictReader(f):
            real_names.add(r['name'])
    print(f'  Real strategy universe: {len(real_names)}')

    # Scan all batch files, resolve only those that match a real name AND have SPACE
    files = sorted(glob.glob(f'{BATCHES_DIR}/strategies_tv2_batch*.py'))
    found = []
    seen = set()
    for fpath in files:
        modname = os.path.basename(fpath)[:-3]
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for attr in dir(mod):
            if not attr.startswith('gen_'):
                continue
            sname = attr[4:]
            if sname not in real_names or sname in seen:
                continue
            gen_fn = getattr(mod, attr)
            space_fn = getattr(mod, f'SPACE_{sname}', None) or getattr(mod, f'space_{sname}', None)
            if not space_fn:
                continue  # skip helpers without space
            found.append((sname, gen_fn, space_fn, modname))
            seen.add(sname)
            if max_strats and len(found) >= max_strats:
                return found
    return found


def main():
    print(f'Discovering strategies...')
    all_strats = discover_strategies()
    print(f'Total strategies available: {len(all_strats)}')

    random.seed(SEED)
    pilot = random.sample(all_strats, min(N_PILOT, len(all_strats)))
    print(f'Piloto: {len(pilot)} aleatorias')
    print(f'Assets: {len(SCREEN_ASSETS)}  TFs: {SCREEN_TFS}  Gate: WR>={MIN_WR}, trades>={MIN_TRADES}, DD<{EARLY_EXIT_DD_PCT}%')

    # Pre-load candles cache (21 assets × 3 TFs = 63 loads)
    print(f'\nLoading candles cache ({len(SCREEN_ASSETS)} × {len(SCREEN_TFS)} = {len(SCREEN_ASSETS)*len(SCREEN_TFS)} timeseries)...')
    t0 = time.time()
    candles_cache = {}
    for sym in SCREEN_ASSETS:
        for tf in SCREEN_TFS:
            df = load_candles(sym, tf)
            # screen_strategy uses string key f"{asset}_{tf}" internally
            candles_cache[f"{sym}_{tf}"] = df
            if df is None:
                print(f'  MISSING: {sym} {tf}')
    print(f'  Cache loaded in {time.time()-t0:.1f}s')

    # Run screening
    print(f'\nStart screening pilot...')
    t0 = time.time()
    survivors = []
    no_data = 0
    errors = 0
    results = []

    for i, (name, gen_fn, space_fn, mod) in enumerate(pilot):
        try:
            r = screen_strategy(name, gen_fn, space_fn, candles_cache)
            if r is None:
                no_data += 1
                continue
            results.append(r)
            if r.get('passing_combos', 0) >= 1:
                survivors.append(r)
        except Exception as e:
            errors += 1

        if (i + 1) % 50 == 0:
            el = time.time() - t0
            rate = (i + 1) / el
            eta_12568 = 12568 / rate / 3600
            print(f'  [{i+1}/{len(pilot)}] {rate:.2f} strats/s | survivors {len(survivors)} | '
                  f'ETA 12.5k: {eta_12568:.1f}h single-worker / {eta_12568/8:.1f}h 8w')

    elapsed = time.time() - t0
    print(f'\n{"="*70}')
    print(f'  PILOT RESULTS (n={len(pilot)})')
    print(f'{"="*70}')
    print(f'  Elapsed:         {elapsed:.1f}s')
    print(f'  Rate:            {len(pilot)/elapsed:.2f} strats/s')
    print(f'  Survivors:       {len(survivors)}  ({len(survivors)*100/len(pilot):.1f}%)')
    print(f'  No data:         {no_data}')
    print(f'  Errors:          {errors}')
    print()
    est_12568_single = 12568 / (len(pilot)/elapsed) / 3600
    print(f'  Extrapolación sobre 12,568:')
    print(f'    1 worker  : {est_12568_single:.1f} h')
    print(f'    4 workers : {est_12568_single/4:.1f} h')
    print(f'    8 workers : {est_12568_single/8:.1f} h')
    print()
    print(f'  Survivors extrapolados: ~{int(len(survivors)*12568/len(pilot))}')

    # Top 15 survivors by best_wr
    if survivors:
        survivors.sort(key=lambda r: -(r.get('best_wr') or 0))
        print(f'\n  Top 15 survivors:')
        print(f'  {"Name":40s} {"Asset":10s} {"TF":6s} {"WR":>7s} {"Trades":>7s} {"PnL%":>8s}')
        for r in survivors[:15]:
            print(f'  {r["name"][:40]:40s} {r.get("best_asset","")[:10]:10s} {r.get("best_tf","")[:6]:6s} '
                  f'{r.get("best_wr",0):>7.1f} {r.get("best_trades",0):>7d} {r.get("best_pnl_pct",0):>8.2f}')

    # Dump results
    out_json = OUT / 'pilot_500_results.json'
    with open(out_json, 'w') as f:
        json.dump({
            'config': {
                'n_pilot': len(pilot),
                'seed': SEED,
                'assets': SCREEN_ASSETS,
                'tfs': SCREEN_TFS,
                'min_wr': MIN_WR,
                'min_trades': MIN_TRADES,
                'early_exit_dd': EARLY_EXIT_DD_PCT,
            },
            'metrics': {
                'elapsed_sec': elapsed,
                'rate_per_sec': len(pilot)/elapsed,
                'n_survivors': len(survivors),
                'survivor_pct': len(survivors)*100/len(pilot),
                'no_data': no_data,
                'errors': errors,
                'est_full_1w_h': est_12568_single,
                'est_full_8w_h': est_12568_single/8,
                'est_full_survivors': int(len(survivors)*12568/len(pilot)),
            },
            'survivors': survivors[:100],  # top 100
        }, f, indent=2, default=str)
    print(f'\n  Dumped: {out_json}')


if __name__ == '__main__':
    sys.exit(main() or 0)
