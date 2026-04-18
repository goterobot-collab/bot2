"""Wrapper batch9: carga tv2_batch9a + tv2_batch9b + tv2_batch9c."""
TV2_BATCH9_STRATS = {}
import importlib, sys, os

_strat_dir = os.path.dirname(os.path.abspath(__file__))
if _strat_dir not in sys.path:
    sys.path.insert(0, _strat_dir)

for _mod_name in ['tv2_batch9a', 'tv2_batch9b', 'tv2_batch9c']:
    try:
        _mod = importlib.import_module(_mod_name)
        TV2_BATCH9_STRATS.update(getattr(_mod, 'STRATEGY_EXPORT', {}))
    except Exception as _e:
        import warnings
        warnings.warn(f"tv2_batch9: failed to load {_mod_name}: {_e}")
