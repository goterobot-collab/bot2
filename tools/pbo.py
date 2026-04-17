"""Probability of Backtest Overfitting (PBO) via simplified 2-fold CSCV.

Bailey, Borwein, Lopez de Prado, Zhu (2014). Uses h1_ret / h2_ret as the two
walk-forward periods. For each fold (train=h1/test=h2 and train=h2/test=h1),
pick the in-sample best trial and check whether its out-of-sample relative
rank is below the median. PBO is the fraction of folds where it is.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/home/user/bot2/results/grails_loop_wf.jsonl")
OUT = Path("/home/user/bot2/results/pbo_report.md")


def _key(row: dict) -> str:
    """Stable trial id from (strategy, asset, tf, params, exit_cfg)."""
    return json.dumps(
        [row.get("strategy"), row.get("asset"), row.get("tf"),
         row.get("params"), row.get("exit_cfg")],
        sort_keys=True, separators=(",", ":"),
    )


def load_trials(path: Path) -> pd.DataFrame:
    keys, h1, h2 = [], [], []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            a, b = r.get("h1_ret"), r.get("h2_ret")
            if a is None or b is None:
                continue
            try:
                af, bf = float(a), float(b)
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(af) and math.isfinite(bf)):
                continue
            keys.append(_key(r))
            h1.append(af)
            h2.append(bf)
    df = pd.DataFrame({"key": keys, "h1": h1, "h2": h2})
    # Aggregate duplicate trials by mean (rare, but possible across iters).
    df = df.groupby("key", as_index=False).agg({"h1": "mean", "h2": "mean"})
    return df


def relative_rank(values: np.ndarray, idx: int) -> float:
    """Return rank of values[idx] in [0,1) using average-rank ties."""
    s = pd.Series(values)
    r = s.rank(method="average").to_numpy()
    return float((r[idx] - 1) / (len(values) - 1)) if len(values) > 1 else 0.5


def cscv_2fold(df: pd.DataFrame) -> dict:
    h1 = df["h1"].to_numpy()
    h2 = df["h2"].to_numpy()
    n = len(df)

    # Fold A: train on h1, test on h2.
    a_train_best = int(np.argmax(h1))
    a_test_rel = relative_rank(h2, a_train_best)
    a_logit = math.log(a_test_rel / (1 - a_test_rel)) if 0 < a_test_rel < 1 else float("nan")

    # Fold B: train on h2, test on h1.
    b_train_best = int(np.argmax(h2))
    b_test_rel = relative_rank(h1, b_train_best)
    b_logit = math.log(b_test_rel / (1 - b_test_rel)) if 0 < b_test_rel < 1 else float("nan")

    overfit_a = a_test_rel < 0.5
    overfit_b = b_test_rel < 0.5
    pbo = (int(overfit_a) + int(overfit_b)) / 2.0

    return {
        "n_trials": n,
        "fold_a": {"train_best_idx": a_train_best, "test_rel_rank": a_test_rel,
                   "logit": a_logit, "overfit": overfit_a,
                   "train_ret": float(h1[a_train_best]),
                   "test_ret": float(h2[a_train_best])},
        "fold_b": {"train_best_idx": b_train_best, "test_rel_rank": b_test_rel,
                   "logit": b_logit, "overfit": overfit_b,
                   "train_ret": float(h2[b_train_best]),
                   "test_ret": float(h1[b_train_best])},
        "pbo": pbo,
    }


def render_report(res: dict) -> str:
    pbo = res["pbo"]
    if pbo < 0.5:
        verdict = "Healthy: in-sample winners tend to remain above-median out of sample."
    elif pbo > 0.5:
        verdict = "Overfit-dominated: in-sample winners tend to land below-median out of sample."
    else:
        verdict = "Borderline: in-sample winners are no better than random out of sample."
    fa, fb = res["fold_a"], res["fold_b"]
    lines = [
        "# Probability of Backtest Overfitting (PBO)",
        "",
        "Method: simplified 2-fold CSCV (Bailey, Borwein, Lopez de Prado, Zhu 2014)",
        "using h1_ret / h2_ret as the two walk-forward periods.",
        "",
        f"- Unique trials: **{res['n_trials']:,}**",
        f"- **PBO = {pbo:.3f}**",
        "",
        f"_{verdict}_",
        "",
        "## Fold detail",
        "",
        "| Fold | Train | Test | Best train return | Test return | Test rel-rank | Logit | Overfit? |",
        "|------|-------|------|------------------:|------------:|--------------:|------:|:--------:|",
        f"| A | h1 | h2 | {fa['train_ret']:.2f} | {fa['test_ret']:.2f} | {fa['test_rel_rank']:.4f} | {fa['logit']:.3f} | {'YES' if fa['overfit'] else 'no'} |",
        f"| B | h2 | h1 | {fb['train_ret']:.2f} | {fb['test_ret']:.2f} | {fb['test_rel_rank']:.4f} | {fb['logit']:.3f} | {'YES' if fb['overfit'] else 'no'} |",
        "",
        "Note: with only two walk-forward halves the CSCV combinatorial set has size 2,",
        "so PBO is quantized to {0.0, 0.5, 1.0}. Treat as a coarse sanity check.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    print(f"Loading {SRC} ...")
    df = load_trials(SRC)
    print(f"Loaded {len(df):,} unique trials with both h1_ret and h2_ret.")
    res = cscv_2fold(df)
    OUT.write_text(render_report(res))
    print(f"PBO = {res['pbo']:.3f}  (n_trials={res['n_trials']:,})")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
