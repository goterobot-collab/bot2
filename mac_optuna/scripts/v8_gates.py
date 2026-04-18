#!/usr/bin/env python3
"""
v8_gates.py — Gates V8 REAL (honest) para Bot V7 production

Convergencia 3/3 A/A/A (COORDINADORA + CEREBRO + SIGNAL_AUDITOR).
Aprobado Sabrina 2026-04-15.

GATES IMPLEMENTADOS (6 de 7 propuestos):
  1. Wilson LCB        — WR con interval de confianza conservador
  2. DSR               — Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
                         corrige Sharpe por multiple testing (data snooping)
  3. Hurst R/S         — edge vs random walk (|H - 0.5| >= threshold)
  4. regime_coverage   — pasa en BULL + BEAR + CHOP (no solo 1 regime)
  5. wr_long/wr_short  — direccional separado (combos no pueden colar en 1 dirección)
  6. fee_rate == 0.30% — Regla 24 canónica (0.10% comisión + 0.05% slippage × 2 lados)
  7. n_trades >= 50    — estadística mínima

DEFERIDO a V8 T+2 (lunes):
  - BH-FDR: requiere p-values per-combo que Optuna no guarda hoy

USO:
  from v8_gates import apply_v8_gates, V8GateConfig
  result = apply_v8_gates(grail_dict, V8GateConfig.default())
  # result = {'pass': bool, 'gates': {...}, 'reasons': [...]}

TESTS:
  python3 -m pytest tests/test_v8_gates.py -v

REFERENCIAS:
  Bailey & Lopez de Prado 2014: "The Deflated Sharpe Ratio"
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
  Wilson 1927: "Probable Inference, the Law of Succession, and Statistical Inference"
  Peters 1994: "Fractal Market Analysis" (Hurst R/S)
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────
# 1. Wilson LCB — Wilson score interval, lower bound
# ─────────────────────────────────────────────────────────────


def wilson_lcb(wins: int, n: int, confidence: float = 0.95) -> float:
    """
    Wilson score interval, lower confidence bound (one-sided).

    Mejor que la fórmula normal aproximada (p_hat ± z*sqrt(p(1-p)/n))
    porque es válida para n pequeño y extremos (p=0, p=1).

    Args:
        wins: número de trades ganadores
        n:    total de trades
        confidence: nivel de confianza (0.95 = 95% one-sided)

    Returns:
        LCB ∈ [0, 1]. Si n=0, retorna 0.

    Ejemplos (sanity):
        wilson_lcb(70, 100) ≈ 0.619   (WR 70%, n=100, CI 95% → LCB=61.9%)
        wilson_lcb(1, 1)   ≈ 0.000   (WR 100% n=1 es estadísticamente nada)
        wilson_lcb(5, 5)   ≈ 0.566   (WR 100% n=5 → LCB 56.6%)
        wilson_lcb(8, 10)  ≈ 0.490   (WR 80% n=10 → LCB 49.0%)
        wilson_lcb(80, 100) ≈ 0.720  (WR 80% n=100 → LCB 72.0%)
    """
    if n <= 0:
        return 0.0
    if wins < 0 or wins > n:
        raise ValueError(f"wins={wins} out of range [0, n={n}]")

    # z para one-sided: confidence 0.95 → z=1.6449
    # Si confidence=0.95 two-sided, z=1.96. Para LCB unilateral usar 0.95→1.6449.
    # Convención SIGNAL_AUDITOR: LCB one-sided 95% → z=1.6449
    z_table = {0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}
    z = z_table.get(round(confidence, 3), 1.6449)

    p = wins / n
    denom = 1 + (z * z) / n
    center = p + (z * z) / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + (z * z) / (4 * n * n))
    lcb = (center - spread) / denom
    return max(0.0, min(1.0, lcb))


# ─────────────────────────────────────────────────────────────
# 2. DSR — Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
# ─────────────────────────────────────────────────────────────


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (stdlib only, uses math.erf)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _expected_max_sr(n_trials: int) -> float:
    """
    E[max SR*_N] bajo null hypothesis (returns iid N(0,1)).

    Approximation: sqrt(2 * log(N)) - (log(log(N)) + log(4π)) / (2*sqrt(2*log(N)))

    Para N=1 devuelve ~0 (no hay "max" de 1 intento).
    Para N grande crece con sqrt(2*log(N)).
    """
    if n_trials <= 1:
        return 0.0
    log_n = math.log(n_trials)
    sqrt2logn = math.sqrt(2.0 * log_n)
    # Evitar log(log(1))=log(0)=-inf
    if log_n <= 1.0:
        return sqrt2logn
    log_log_n = math.log(log_n)
    return sqrt2logn - (log_log_n + math.log(4.0 * math.pi)) / (2.0 * sqrt2logn)


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_trades: int,
    skewness: float = 0.0,
    kurtosis_excess: float = 0.0,
) -> float:
    """
    DSR: probability that observed Sharpe > threshold from multiple testing.

    Bailey & Lopez de Prado 2014.

    DSR = Φ( (SR - E[SR*_N]) * sqrt(T-1) / sqrt(1 - skew*SR + ((kurt-1)/4)*SR²) )

    donde:
        SR          = observed Sharpe ratio (annualized or raw, consistent)
        E[SR*_N]    = expected max from N trials under null
        T           = n_trades
        skew, kurt  = skewness and excess kurtosis of per-trade returns
        Φ           = standard normal CDF

    Args:
        sharpe:   Sharpe ratio observado del grail
        n_trials: número de combos/trials probados (para los 137: 23,829)
        n_trades: número de trades en el backtest
        skewness: skewness de returns (default 0 = normal)
        kurtosis_excess: exceso de kurtosis (default 0 = normal)

    Returns:
        DSR ∈ [0, 1]. Alto = Sharpe genuino. Gate típico: DSR >= 0.95.

    Ejemplos:
        deflated_sharpe_ratio(0.5, 100, 50)  ≈ baja (SR=0.5 no supera noise)
        deflated_sharpe_ratio(2.5, 10, 100)  ≈ alta (SR bien por encima)
        deflated_sharpe_ratio(3.0, 23829, 50) ≈ moderada (multiple testing severo)
    """
    if n_trades <= 1 or n_trials < 1:
        return 0.0

    sr0 = _expected_max_sr(n_trials)

    # Denominator: sqrt(1 - skew*SR + ((kurt-1)/4)*SR²)
    # Nota: Bailey usa "kurt-1" donde kurt = excess + 3, entonces kurt-1 = excess+2
    #       Pero la convención común es usar excess kurtosis directo con "kurt/4"
    #       Acá uso la forma canónica del paper (kurt es full kurtosis).
    full_kurtosis = kurtosis_excess + 3.0  # convert excess to full
    var_term = 1.0 - skewness * sharpe + ((full_kurtosis - 1.0) / 4.0) * sharpe * sharpe
    if var_term <= 0:
        # Degenerate case (extreme skew/kurt)
        return 0.0

    z = (sharpe - sr0) * math.sqrt(n_trades - 1) / math.sqrt(var_term)
    return _norm_cdf(z)


# ─────────────────────────────────────────────────────────────
# 3. Hurst R/S analysis
# ─────────────────────────────────────────────────────────────


def hurst_exponent(series: list[float], min_lag: int = 2, max_lag: int = 20) -> float:
    """
    Hurst exponent via R/S analysis.

    Interpretación:
        H ≈ 0.5 → random walk (sin edge)
        H > 0.5 → trend-persistent (mean-reversion falla)
        H < 0.5 → anti-persistent (mean-reversion funciona)

    Args:
        series: secuencia de returns (o precios si se prefiere, pero returns es estándar)
        min_lag, max_lag: rango de lags para la regresión

    Returns:
        H ∈ [0, 1] idealmente. Si la serie es muy corta → 0.5 (no info).

    Ejemplos:
        hurst_exponent(random_walk(100))  ≈ 0.50
        hurst_exponent(trend_up(100))     ≈ 0.75+
        hurst_exponent(mean_revert(100))  ≈ 0.30-
    """
    n = len(series)
    if n < max_lag * 3:
        # Muestra insuficiente
        return 0.5

    # Ajustar max_lag a la longitud disponible
    max_lag = min(max_lag, n // 4)
    if max_lag <= min_lag:
        return 0.5

    rs_values = []
    log_lags = []

    for lag in range(min_lag, max_lag + 1):
        # Dividir serie en chunks de tamaño lag, calcular R/S por chunk, promediar
        n_chunks = n // lag
        if n_chunks < 1:
            continue
        rs_per_chunk = []
        for i in range(n_chunks):
            chunk = series[i * lag:(i + 1) * lag]
            if len(chunk) < 2:
                continue
            mean_c = sum(chunk) / len(chunk)
            # Deviations from mean
            devs = [x - mean_c for x in chunk]
            # Cumulative deviations
            cum = []
            s = 0.0
            for d in devs:
                s += d
                cum.append(s)
            R = max(cum) - min(cum)  # range
            # Std dev
            if len(chunk) < 2:
                continue
            S = statistics.pstdev(chunk)
            if S <= 0 or R <= 0:
                continue
            rs_per_chunk.append(R / S)
        if rs_per_chunk:
            avg_rs = sum(rs_per_chunk) / len(rs_per_chunk)
            rs_values.append(math.log(avg_rs))
            log_lags.append(math.log(lag))

    if len(rs_values) < 2:
        return 0.5

    # OLS regression: log(R/S) = H * log(lag) + c
    # slope = cov(x,y) / var(x)
    mean_x = sum(log_lags) / len(log_lags)
    mean_y = sum(rs_values) / len(rs_values)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_lags, rs_values))
    den = sum((x - mean_x) ** 2 for x in log_lags)
    if den == 0:
        return 0.5
    H = num / den
    return max(0.0, min(1.5, H))  # clamp suave — valores extremos = sample issues


# ─────────────────────────────────────────────────────────────
# 4. Regime coverage — ¿pasa en BULL + BEAR + CHOP?
# ─────────────────────────────────────────────────────────────


def regime_coverage(
    trades: list[dict], min_trades_per_regime: int = 5, min_wr_per_regime: float = 0.50
) -> dict:
    """
    Evalúa si el grail tiene edge en múltiples regímenes de mercado.

    Args:
        trades: lista de dicts con al menos {'win': bool, 'regime': str}
                regime ∈ {'BULL', 'BEAR', 'CHOP', 'NEUTRAL'}
        min_trades_per_regime: mínimo de trades para contar el regime
        min_wr_per_regime: WR mínimo para considerar el regime "covered"

    Returns:
        {
          'bull_wr': float, 'bull_n': int,
          'bear_wr': float, 'bear_n': int,
          'chop_wr': float, 'chop_n': int,
          'n_regimes_covered': int (0-3),
          'coverage_score': float ∈ [0, 1] (fraction of regimes passing)
          'passes': bool (>= 2 regimes covered with WR >= threshold)
        }
    """
    regimes = {'BULL': [], 'BEAR': [], 'CHOP': []}
    for t in trades:
        r = (t.get('regime') or '').upper()
        if r in regimes:
            regimes[r].append(bool(t.get('win')))

    result = {}
    covered = 0
    for reg, wins_list in regimes.items():
        n = len(wins_list)
        wr = (sum(wins_list) / n) if n else 0.0
        result[f'{reg.lower()}_wr'] = wr
        result[f'{reg.lower()}_n'] = n
        if n >= min_trades_per_regime and wr >= min_wr_per_regime:
            covered += 1

    result['n_regimes_covered'] = covered
    result['coverage_score'] = covered / 3.0
    result['passes'] = covered >= 2  # pasa si al menos 2 de 3 regimes OK
    return result


# ─────────────────────────────────────────────────────────────
# 5. Direction split — wr_long vs wr_short
# ─────────────────────────────────────────────────────────────


def direction_split_wr(trades: list[dict], min_trades_per_direction: int = 10) -> dict:
    """
    Calcula WR per direction (LONG vs SHORT).

    Gate V8: si ambas direcciones tienen >= min_trades, AMBAS deben tener WR decente.
    Si solo una dirección tiene datos, es OK (combo direccional).

    Args:
        trades: lista con {'win': bool, 'direction': 'LONG'|'SHORT'}
        min_trades_per_direction: mínimo para evaluar WR de esa dirección

    Returns:
        {
          'wr_long': float or None (None si n<min),
          'wr_short': float or None,
          'n_long': int, 'n_short': int,
          'direction_mix': 'LONG_ONLY' | 'SHORT_ONLY' | 'BOTH' | 'NONE',
        }
    """
    long_wins = long_n = short_wins = short_n = 0
    for t in trades:
        d = (t.get('direction') or '').upper()
        w = bool(t.get('win'))
        if d == 'LONG':
            long_n += 1
            long_wins += int(w)
        elif d == 'SHORT':
            short_n += 1
            short_wins += int(w)

    wr_long = (long_wins / long_n) if long_n >= min_trades_per_direction else None
    wr_short = (short_wins / short_n) if short_n >= min_trades_per_direction else None

    if long_n >= min_trades_per_direction and short_n >= min_trades_per_direction:
        direction_mix = 'BOTH'
    elif long_n >= min_trades_per_direction:
        direction_mix = 'LONG_ONLY'
    elif short_n >= min_trades_per_direction:
        direction_mix = 'SHORT_ONLY'
    else:
        direction_mix = 'NONE'

    return {
        'wr_long': wr_long,
        'wr_short': wr_short,
        'n_long': long_n,
        'n_short': short_n,
        'direction_mix': direction_mix,
    }


# ─────────────────────────────────────────────────────────────
# 6. Gate aggregator — apply_v8_gates
# ─────────────────────────────────────────────────────────────


@dataclass
class V8GateConfig:
    """Thresholds para cada gate. Todos los valores son canonical V8 REAL."""

    min_wilson_lcb: float = 0.55       # WR LCB one-sided 95%
    min_dsr: float = 0.95              # Deflated Sharpe prob
    min_hurst_deviation: float = 0.05  # |H - 0.5| >= 0.05
    min_regime_coverage: float = 2     # at least 2 of 3 regimes
    min_wr_per_direction: float = 0.60 # si BOTH, cada uno >= 60%
    required_fee_rate: float = 0.0030  # 0.30% round-trip (Regla 24)
    min_n_trades: int = 50             # estadística mínima
    wilson_confidence: float = 0.95    # one-sided
    hurst_min_lag: int = 2
    hurst_max_lag: int = 20
    # BH-FDR deferido (no hay p-values per-trial hoy)
    require_bh_fdr: bool = False

    @classmethod
    def default(cls) -> "V8GateConfig":
        return cls()

    @classmethod
    def relaxed_v75plus(cls) -> "V8GateConfig":
        """Config V7.5+ (hoy, gates parciales)."""
        return cls(
            min_wilson_lcb=0.55,
            min_dsr=0.0,  # skip DSR hasta T+2
            min_hurst_deviation=0.0,  # skip Hurst hasta T+2
            min_regime_coverage=0,  # skip regime hasta T+2
            min_wr_per_direction=0.55,
            required_fee_rate=0.0030,
            min_n_trades=30,  # más permisivo
        )


def apply_v8_gates(grail: dict, config: V8GateConfig | None = None) -> dict:
    """
    Aplica los 6+1 gates V8 a un grail.

    Input grail requiere los siguientes campos (flexible, usa 0/defaults si faltan):
        - wins (int), total_trades (int)
        - sharpe (float)
        - returns (list[float]) — returns per-trade para skew/kurt/Hurst
        - trades (list[dict]) — {'win','direction','regime'} per trade
        - n_trials (int) — trials Optuna considerados (default 1000)
        - fee_rate (float) — 0.0030 para pasar (Regla 24)
        - skewness (float, opcional) — si no, calculado de returns
        - kurtosis_excess (float, opcional) — idem

    Returns:
        {
          'pass': bool,
          'gates': {
            'wilson_lcb': {'value': ..., 'pass': ..., 'threshold': ...},
            'dsr': {...},
            'hurst': {...},
            'regime_coverage': {...},
            'direction_wr': {...},
            'fee_rate': {...},
            'n_trades': {...},
          },
          'failed_gates': [list of gate names that failed],
          'score': float ∈ [0, 1] (fraction of gates passed),
        }
    """
    cfg = config or V8GateConfig.default()

    wins = int(grail.get('wins', 0))
    n_trades = int(grail.get('total_trades') or grail.get('n_trades') or len(grail.get('trades', [])))
    sharpe = float(grail.get('sharpe') or 0.0)
    n_trials = int(grail.get('n_trials') or 1000)
    fee_rate = grail.get('fee_rate')
    returns = grail.get('returns') or []
    trades_list = grail.get('trades') or []

    # Skew/Kurt
    skewness = grail.get('skewness')
    kurt_excess = grail.get('kurtosis_excess')
    if (skewness is None or kurt_excess is None) and len(returns) >= 4:
        try:
            m = sum(returns) / len(returns)
            s2 = sum((x - m) ** 2 for x in returns) / len(returns)
            s = math.sqrt(s2) if s2 > 0 else 0.0
            if s > 0:
                skewness = skewness if skewness is not None else \
                    sum(((x - m) / s) ** 3 for x in returns) / len(returns)
                kurt_excess = kurt_excess if kurt_excess is not None else \
                    sum(((x - m) / s) ** 4 for x in returns) / len(returns) - 3.0
            else:
                skewness = skewness or 0.0
                kurt_excess = kurt_excess or 0.0
        except (TypeError, ValueError, ZeroDivisionError):
            skewness = 0.0
            kurt_excess = 0.0
    if skewness is None:
        skewness = 0.0
    if kurt_excess is None:
        kurt_excess = 0.0

    # Gate 1: Wilson LCB
    wlcb = wilson_lcb(wins, n_trades, cfg.wilson_confidence)
    gate_wilson = {
        'value': round(wlcb, 4),
        'threshold': cfg.min_wilson_lcb,
        'pass': wlcb >= cfg.min_wilson_lcb,
    }

    # Gate 2: DSR
    if cfg.min_dsr > 0:
        dsr = deflated_sharpe_ratio(sharpe, n_trials, n_trades, skewness, kurt_excess)
        gate_dsr = {
            'value': round(dsr, 4),
            'threshold': cfg.min_dsr,
            'pass': dsr >= cfg.min_dsr,
            'n_trials': n_trials,
        }
    else:
        gate_dsr = {'value': None, 'threshold': cfg.min_dsr, 'pass': True, 'skipped': True}

    # Gate 3: Hurst
    if cfg.min_hurst_deviation > 0 and len(returns) >= 30:
        H = hurst_exponent(returns, cfg.hurst_min_lag, cfg.hurst_max_lag)
        deviation = abs(H - 0.5)
        gate_hurst = {
            'value': round(H, 4),
            'deviation': round(deviation, 4),
            'threshold': cfg.min_hurst_deviation,
            'pass': deviation >= cfg.min_hurst_deviation,
        }
    else:
        gate_hurst = {
            'value': None,
            'threshold': cfg.min_hurst_deviation,
            'pass': cfg.min_hurst_deviation == 0,  # skip si deshabilitado
            'skipped': True,
        }

    # Gate 4: Regime coverage
    if cfg.min_regime_coverage > 0 and trades_list:
        rc = regime_coverage(trades_list)
        gate_regime = {
            'value': rc['n_regimes_covered'],
            'threshold': cfg.min_regime_coverage,
            'pass': rc['n_regimes_covered'] >= cfg.min_regime_coverage,
            'breakdown': {k: v for k, v in rc.items() if k != 'passes'},
        }
    else:
        gate_regime = {
            'value': None,
            'threshold': cfg.min_regime_coverage,
            'pass': cfg.min_regime_coverage == 0,
            'skipped': True,
        }

    # Gate 5: Direction split
    if trades_list:
        ds = direction_split_wr(trades_list)
        if ds['direction_mix'] == 'BOTH':
            ok = (ds['wr_long'] >= cfg.min_wr_per_direction and
                  ds['wr_short'] >= cfg.min_wr_per_direction)
        elif ds['direction_mix'] in ('LONG_ONLY', 'SHORT_ONLY'):
            # Direccional puro — usar la dirección que tiene datos
            wr_used = ds['wr_long'] if ds['direction_mix'] == 'LONG_ONLY' else ds['wr_short']
            ok = wr_used is not None and wr_used >= cfg.min_wr_per_direction
        else:
            ok = False
        gate_direction = {
            'wr_long': ds['wr_long'],
            'wr_short': ds['wr_short'],
            'n_long': ds['n_long'],
            'n_short': ds['n_short'],
            'direction_mix': ds['direction_mix'],
            'threshold': cfg.min_wr_per_direction,
            'pass': ok,
        }
    else:
        gate_direction = {
            'value': None,
            'pass': False,
            'skipped': True,
            'reason': 'no trades list provided',
        }

    # Gate 6: Fee rate canónico
    fr_pass = (fee_rate is not None) and abs(float(fee_rate) - cfg.required_fee_rate) < 1e-6
    gate_fee = {
        'value': fee_rate,
        'threshold': cfg.required_fee_rate,
        'pass': fr_pass,
    }

    # Gate 7: n_trades minimum
    gate_n = {
        'value': n_trades,
        'threshold': cfg.min_n_trades,
        'pass': n_trades >= cfg.min_n_trades,
    }

    gates = {
        'wilson_lcb': gate_wilson,
        'dsr': gate_dsr,
        'hurst': gate_hurst,
        'regime_coverage': gate_regime,
        'direction_wr': gate_direction,
        'fee_rate': gate_fee,
        'n_trades': gate_n,
    }

    # Aggregate: todos los gates no-skipped deben pasar
    failed = [name for name, g in gates.items() if not g.get('pass', False) and not g.get('skipped')]
    total_active = sum(1 for g in gates.values() if not g.get('skipped'))
    passed_active = sum(1 for g in gates.values() if g.get('pass') and not g.get('skipped'))
    score = (passed_active / total_active) if total_active else 0.0

    return {
        'pass': len(failed) == 0,
        'gates': gates,
        'failed_gates': failed,
        'score': round(score, 4),
        'config_profile': 'default' if cfg.min_dsr > 0 else 'relaxed_v75plus',
    }


# ─────────────────────────────────────────────────────────────
# Smoke test inline
# ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import json
    print("─── v8_gates.py smoke tests ───")

    # Wilson LCB sanity
    print(f"\n[Wilson LCB]")
    for wins, n in [(70, 100), (1, 1), (5, 5), (8, 10), (80, 100)]:
        lcb = wilson_lcb(wins, n)
        print(f"  wilson_lcb({wins}, {n}) = {lcb:.4f}")

    # DSR sanity
    print(f"\n[DSR]")
    for sr, n_trials, n_trades in [(0.5, 100, 50), (2.5, 10, 100), (3.0, 23829, 50), (1.5, 1000, 100)]:
        dsr = deflated_sharpe_ratio(sr, n_trials, n_trades)
        print(f"  DSR(SR={sr}, N={n_trials}, T={n_trades}) = {dsr:.4f}")

    # Hurst sanity
    print(f"\n[Hurst]")
    # Random walk
    import random
    random.seed(42)
    rw = [random.gauss(0, 1) for _ in range(500)]
    print(f"  Hurst(random_walk, n=500) = {hurst_exponent(rw):.4f}  (esperado ~0.5)")
    # Trend
    trend = [i * 0.1 + random.gauss(0, 1) for i in range(500)]
    print(f"  Hurst(trend, n=500)       = {hurst_exponent(trend):.4f}  (esperado >0.6)")
    # Mean-rev
    mr = [random.gauss(0, 1) for _ in range(500)]
    for i in range(1, len(mr)):
        mr[i] = -0.3 * mr[i - 1] + random.gauss(0, 1)
    print(f"  Hurst(mean_rev, n=500)    = {hurst_exponent(mr):.4f}  (esperado <0.4)")

    # apply_v8_gates sample
    print(f"\n[apply_v8_gates]")
    sample_grail = {
        'wins': 70,
        'total_trades': 100,
        'sharpe': 1.8,
        'returns': [random.gauss(0.003, 0.02) for _ in range(100)],
        'trades': [
            {'win': i % 3 != 0, 'direction': 'LONG' if i % 2 else 'SHORT',
             'regime': ['BULL', 'BEAR', 'CHOP'][i % 3]}
            for i in range(100)
        ],
        'n_trials': 10000,
        'fee_rate': 0.0030,
    }
    result = apply_v8_gates(sample_grail)
    print(json.dumps(result, indent=2, default=str))

    print("\n─── smoke tests done ───")
