#!/usr/bin/env python3
"""Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

Reference: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551

Reads a JSONL of trial evaluations with at least the columns
{strategy, asset, tf, params, exit_cfg, trades, wr, pf, ret, dd},
computes a Sharpe-ratio proxy + the Deflated Sharpe Ratio (DSR) for
every unique trial and writes the top-20-by-DSR table to a Markdown
report.

Assumptions / heuristics (the JSONL has no per-trade return stream,
only summary stats, so we model trades as a Bernoulli payoff process):

1. Sharpe proxy from win-rate w and profit factor p.  Payoff ratio
   b = p*(1-w)/w gives  mu = w*b - (1-w),  sigma = sqrt(w(1-w))*(b+1).
   Trial Sharpe = (mu/sigma)*sqrt(trades).  Same construction Bailey &
   López de Prado use in "The Sharpe Ratio Efficient Frontier".
2. Skew gamma3 = (1-2w)/sqrt(w(1-w)),  raw kurt gamma4 = excess+3
   with excess = 1/(w(1-w)) - 6.  Both come from the Bernoulli payoff.
3. N (number of independent trials) = unique
   (strategy, asset, tf, params, exit_cfg) tuples in the file.
4. T (effective sample size in DSR) = trade count of the trial.
5. Pass = DSR > 0.95.

Stdlib + numpy + pandas + scipy.stats only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

JSONL = Path("/home/user/bot2/results/grails_loop_wf.jsonl")
REPORT = Path("/home/user/bot2/results/dsr_report.md")
MIN_TRADES = 30  # below this the moments are unstable


def load_trials(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows)
    df["params_key"] = df["params"].apply(lambda d: json.dumps(d, sort_keys=True))
    df["exit_key"] = df["exit_cfg"].apply(lambda d: json.dumps(d, sort_keys=True))
    df["trial_id"] = (df["strategy"].astype(str) + "|" + df["asset"].astype(str) + "|"
                      + df["tf"].astype(str) + "|" + df["params_key"] + "|" + df["exit_key"])
    return df


def bernoulli_moments(wr_pct, pf):
    w = np.clip(np.asarray(wr_pct) / 100.0, 1e-6, 1 - 1e-6)
    p = np.clip(np.asarray(pf), 1e-6, None)
    b = p * (1 - w) / w
    mu = w * b - (1 - w)
    sigma = np.sqrt(w * (1 - w)) * (b + 1)
    skew = (1 - 2 * w) / np.sqrt(w * (1 - w))
    kurt_raw = (1.0 / (w * (1 - w)) - 6.0) + 3.0
    return mu, sigma, skew, kurt_raw


def sharpe_proxy(wr_pct, pf, trades):
    mu, sigma, _, _ = bernoulli_moments(wr_pct, pf)
    sr_t = np.where(sigma > 0, mu / sigma, 0.0)
    return sr_t * np.sqrt(np.asarray(trades))


def deflated_sharpe(sr, trades, skew, kurt_raw, n_trials):
    """DSR per Bailey & López de Prado 2014 (eq. 9)."""
    sr = np.asarray(sr, float); T = np.asarray(trades, float)
    g3 = np.asarray(skew, float); g4 = np.asarray(kurt_raw, float)
    if n_trials < 2:
        e_max = 0.0
    else:
        em = 0.5772156649015329  # Euler-Mascheroni
        z1 = norm.ppf(1 - 1.0 / n_trials)
        z2 = norm.ppf(1 - 1.0 / (n_trials * math.e))
        e_max = (1 - em) * z1 + em * z2  # full B&LdP expected-max-SR estimator
    denom = np.sqrt(np.maximum(1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2, 1e-9))
    z = (sr - e_max) * np.sqrt(np.maximum(T - 1, 1)) / denom
    return norm.cdf(z), e_max


def md_table(df: pd.DataFrame, cols) -> str:
    head = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join("---" for _ in cols) + " |\n"
    body_rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(f"{v:.4f}")
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            else:
                cells.append(str(v))
        body_rows.append("| " + " | ".join(cells) + " |")
    return head + sep + "\n".join(body_rows) + "\n"


def main() -> None:
    df = load_trials(JSONL)
    print(f"loaded {len(df):,} rows")
    df = df.sort_values("pf", ascending=False).drop_duplicates("trial_id")
    n_trials = df["trial_id"].nunique()
    print(f"unique trials: {n_trials:,}")
    df = df[df["trades"].fillna(0) >= MIN_TRADES].copy()
    print(f"after MIN_TRADES={MIN_TRADES}: {len(df):,}")

    df["sr"] = sharpe_proxy(df["wr"], df["pf"], df["trades"])
    _, _, skew, kurt = bernoulli_moments(df["wr"].to_numpy(), df["pf"].to_numpy())
    df["skew"] = skew; df["kurt"] = kurt
    dsr, e_max = deflated_sharpe(df["sr"].to_numpy(), df["trades"].to_numpy(),
                                 skew, kurt, n_trials)
    df["dsr"] = dsr
    df["pass"] = dsr > 0.95

    top = df.sort_values("dsr", ascending=False).head(20)
    cols = ["strategy", "asset", "tf", "trades", "wr", "pf", "ret", "dd", "sr", "dsr", "pass"]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w") as fh:
        fh.write("# Deflated Sharpe Ratio Report\n\n")
        fh.write("Bailey & López de Prado (2014). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551\n\n")
        fh.write(f"- Source: `{JSONL}`\n")
        fh.write(f"- Unique trials (N): **{n_trials:,}**\n")
        fh.write(f"- E[max SR] under H0: **{e_max:.4f}**\n")
        fh.write(f"- Min trades filter: {MIN_TRADES}\n")
        fh.write(f"- Trials passing DSR > 0.95: **{int(df['pass'].sum()):,}** "
                 f"({100*df['pass'].mean():.2f}%)\n\n")
        fh.write("## Heuristic SR proxy\n\nBernoulli payoff with win-rate w and PF p: "
                 "b = p(1-w)/w, mu = wb-(1-w), sigma = sqrt(w(1-w))(b+1). "
                 "Trial SR = (mu/sigma)*sqrt(trades). Skew/kurt from same model.\n\n")
        fh.write("## Top 20 by DSR\n\n" + md_table(top, cols) + "\n")
        fh.write("## Top 20 - parameter detail\n\n")
        for _, r in top.iterrows():
            fh.write(f"- **{r['strategy']} / {r['asset']} {r['tf']}** — "
                     f"DSR={r['dsr']:.4f} SR={r['sr']:.3f} trades={int(r['trades'])} "
                     f"PF={r['pf']:.3f} ret={r['ret']:.2f}%\n")
            fh.write(f"  - params: `{r['params_key']}`\n  - exit:   `{r['exit_key']}`\n")
    print(f"wrote {REPORT}")
    print("\nTop 5 by DSR:")
    print(top[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
