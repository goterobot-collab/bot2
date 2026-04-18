#!/usr/bin/env python3
"""
Loader for TV2 strategies in Optuna discovery pipeline
Use this to integrate all 1,896 strategies into optuna_v4
"""

import sys
from pathlib import Path

# Add strategies_tv2_batches to path
STRATEGIES_DIR = Path(__file__).parent.parent / "strategies_tv2_batches"
sys.path.insert(0, str(STRATEGIES_DIR))


def load_all_tv2_strategies():
    """Load all TV2 strategies from batches 80-3324+"""
    all_strategies = {}
    errors = []

    # Load batches 80-99 + 3300+ (extended range for newer batches)
    for batch_num in list(range(80, 100)) + list(range(3300, 3330)):
        module_name = f"strategies_tv2_batch{batch_num}"
        try:
            mod = __import__(module_name)
            batch_export = mod.STRATEGY_EXPORT
            all_strategies.update(batch_export)
            print(f"[OK] {module_name}: {len(batch_export)} strategies")
        except Exception as e:
            errors.append((module_name, str(e)))
            print(f"[ERROR] {module_name}: {e}")

    total = len(all_strategies)
    print(f"\n[+] Total strategies loaded: {total}")
    if errors:
        print(f"[!] Errors: {len(errors)}")
        for module, err in errors:
            print(f"    {module}: {err}")

    return all_strategies, errors


def get_tv2_strategy_names():
    """Return list of all strategy names"""
    strategies, _ = load_all_tv2_strategies()
    return sorted(strategies.keys())


def get_tv2_strategy_space(strategy_name):
    """Get parameter space for a specific strategy"""
    strategies, _ = load_all_tv2_strategies()
    if strategy_name not in strategies:
        raise ValueError(f"Strategy '{strategy_name}' not found")

    gen_func, space_func = strategies[strategy_name]
    return space_func()


def execute_tv2_strategy(strategy_name, df, **params):
    """Execute a TV2 strategy on a dataframe"""
    strategies, _ = load_all_tv2_strategies()
    if strategy_name not in strategies:
        raise ValueError(f"Strategy '{strategy_name}' not found")

    gen_func, space_func = strategies[strategy_name]
    return gen_func(df, **params)


if __name__ == '__main__':
    print("="*70)
    print("TV2 STRATEGIES LOADER (for Optuna)")
    print("="*70)

    # Load and summarize
    strategies, errors = load_all_tv2_strategies()

    print(f"\nSummary:")
    print(f"  Total: {len(strategies)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Status: {'✓ READY' if not errors else '✗ ERRORS'}")

    # Sample a few
    if strategies:
        print(f"\nSample strategies (first 5):")
        for name in sorted(strategies.keys())[:5]:
            gen_func, space_func = strategies[name]
            space = space_func()
            params_str = ", ".join([f"{k}: ({v[0]},{v[1]},{v[2]})" for k, v in space.items()])
            print(f"  {name:50s} | {params_str}")
