#!/usr/bin/env python3
"""
parallel_screen_all.py — Full 8,901 strategies across 8 Mac cores.

Approach:
  - Each worker process imports its own batch modules (no pickling of functions)
  - Work unit = strategy name + module name (pickleable)
  - Worker loads candles cache ONCE in initializer, reuses across tasks
  - Writes results as JSON per worker, then merges at the end

Throughput target: ~8× the single-worker rate (0.03 → ~0.24 strats/s → 10h for 8,901).
Overnight-friendly.
"""
import sys, os, time, json, random, importlib, glob, csv
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

BATCHES_DIR = '/Users/sabrina/CLAUDE CODE/Estrategias/strategies_tv2_batches'
sys.path.insert(0, BATCHES_DIR)

OUT = Path('/Users/sabrina/CLAUDE CODE/Estrategias/scripts/screening_output')
OUT.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = OUT / 'parallel_screen_ALL.json'
PROGRESS_JSON = OUT / 'parallel_screen_progress.json'
LOG_FILE = OUT / 'parallel_screen.log'

N_WORKERS = int(os.environ.get('WORKERS', 8))
CHUNK_SIZE = 50  # strats per chunk dispatched to each worker

# Per-worker lazy globals
_WORKER_CACHE = {}
_WORKER_MODULES = {}


def _worker_init():
    """Initializer: each worker loads candles cache ONCE."""
    import warnings
    warnings.filterwarnings('ignore')
    from fast_filter_v2 import load_candles, SCREEN_ASSETS, SCREEN_TFS

    t0 = time.time()
    cache = {}
    for sym in SCREEN_ASSETS:
        for tf in SCREEN_TFS:
            df = load_candles(sym, tf)
            if df is not None:
                cache[f"{sym}_{tf}"] = df
    _WORKER_CACHE['candles'] = cache
    _WORKER_CACHE['init_time'] = time.time() - t0
    # Log from worker
    pid = os.getpid()
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] Worker {pid} cache ready ({len(cache)}/{len(SCREEN_ASSETS)*len(SCREEN_TFS)}) in {_WORKER_CACHE['init_time']:.1f}s\n")


def _resolve(modname, strat_name):
    """Import module lazily (cached) and resolve gen_/SPACE_ functions."""
    if modname not in _WORKER_MODULES:
        try:
            _WORKER_MODULES[modname] = importlib.import_module(modname)
        except Exception as e:
            _WORKER_MODULES[modname] = None
    mod = _WORKER_MODULES[modname]
    if mod is None:
        return None, None
    gen_fn = getattr(mod, f'gen_{strat_name}', None)
    space_fn = getattr(mod, f'SPACE_{strat_name}', None) or getattr(mod, f'space_{strat_name}', None)
    return gen_fn, space_fn


def _screen_batch(batch):
    """Screen a list of (name, modname) tuples. Returns list of results."""
    from fast_filter_v2 import screen_strategy
    cache = _WORKER_CACHE.get('candles', {})
    out = []
    for name, modname in batch:
        try:
            gen_fn, space_fn = _resolve(modname, name)
            if not gen_fn or not space_fn:
                out.append({'name': name, 'status': 'unresolved'})
                continue
            r = screen_strategy(name, gen_fn, space_fn, cache)
            if r is None:
                out.append({'name': name, 'status': 'no_data'})
            else:
                # Keep light record
                r['status'] = 'ok'
                out.append(r)
        except Exception as e:
            out.append({'name': name, 'status': 'error', 'error': str(e)[:200]})
    return out


def discover_strategies():
    """Load REAL strategy universe + resolve modules once in main process."""
    ranked = '/Users/sabrina/CLAUDE CODE/Estrategias/scripts/screening_output/pending_ranked.csv'
    real_names = set()
    with open(ranked) as f:
        for r in csv.DictReader(f):
            real_names.add(r['name'])
    print(f'Real strategy universe: {len(real_names)}', flush=True)

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
            space_fn = getattr(mod, f'SPACE_{sname}', None) or getattr(mod, f'space_{sname}', None)
            if not space_fn:
                continue
            found.append((sname, modname))
            seen.add(sname)
    print(f'Resolvable (gen_ + SPACE_): {len(found)}', flush=True)
    return found


def main():
    t_start = time.time()
    with open(LOG_FILE, 'w') as f:
        f.write(f"=== parallel_screen_all.py START {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"WORKERS={N_WORKERS} CHUNK_SIZE={CHUNK_SIZE}\n")

    print(f'Discovering strategies (main process)...', flush=True)
    all_strats = discover_strategies()
    random.seed(42)
    random.shuffle(all_strats)  # distribute load uniformly across modules

    # Chunk the work
    chunks = [all_strats[i:i+CHUNK_SIZE] for i in range(0, len(all_strats), CHUNK_SIZE)]
    total = len(all_strats)
    print(f'Total strategies: {total}  |  Chunks: {len(chunks)}  |  Workers: {N_WORKERS}', flush=True)

    results = []
    survivors = []
    errors = 0
    no_data = 0
    unresolved = 0
    done = 0

    with Pool(processes=N_WORKERS, initializer=_worker_init) as pool:
        for i, batch_results in enumerate(pool.imap_unordered(_screen_batch, chunks, chunksize=1)):
            for r in batch_results:
                status = r.get('status')
                if status == 'error':
                    errors += 1
                elif status == 'no_data':
                    no_data += 1
                elif status == 'unresolved':
                    unresolved += 1
                else:
                    results.append(r)
                    if r.get('passing_combos', 0) >= 1:
                        survivors.append(r)
            done += len(batch_results)
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta_h = (total - done) / rate / 3600 if rate > 0 else 0

            # Progress snapshot every chunk
            prog = {
                'done': done, 'total': total,
                'survivors': len(survivors),
                'errors': errors, 'no_data': no_data, 'unresolved': unresolved,
                'rate_per_sec': round(rate, 3),
                'elapsed_min': round(elapsed/60, 1),
                'eta_hours': round(eta_h, 2),
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            with open(PROGRESS_JSON, 'w') as f:
                json.dump(prog, f, indent=2)

            # Print + log every 10 chunks (= 500 strats)
            if (i + 1) % 10 == 0 or i == 0:
                msg = (f"[{done}/{total}] {rate:.2f}/s  surv={len(survivors)} "
                       f"({len(survivors)*100/max(done,1):.1f}%)  err={errors}  "
                       f"no_data={no_data}  unres={unresolved}  ETA {eta_h:.1f}h")
                print(msg, flush=True)
                with open(LOG_FILE, 'a') as f:
                    f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

            # Periodic dump (every 20 chunks = 1000 strats)
            if (i + 1) % 20 == 0:
                with open(RESULTS_JSON, 'w') as f:
                    json.dump({
                        'progress': prog,
                        'survivors': survivors,
                        'all_ok_count': len(results),
                    }, f, indent=2, default=str)

    # Final dump
    elapsed = time.time() - t_start
    final = {
        'config': {'workers': N_WORKERS, 'chunk_size': CHUNK_SIZE},
        'metrics': {
            'elapsed_h': round(elapsed/3600, 2),
            'rate_per_sec': round(total/elapsed, 3),
            'total': total,
            'survivors': len(survivors),
            'survivor_pct': round(len(survivors)*100/total, 2),
            'errors': errors,
            'no_data': no_data,
            'unresolved': unresolved,
        },
        'survivors': survivors,  # full list
    }
    with open(RESULTS_JSON, 'w') as f:
        json.dump(final, f, indent=2, default=str)

    msg = (f"\n=== DONE in {elapsed/3600:.2f}h === "
           f"{total} screened, {len(survivors)} survivors "
           f"({len(survivors)*100/total:.1f}%)")
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')


if __name__ == '__main__':
    main()
