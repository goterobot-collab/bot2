#!/usr/bin/env python3
"""Triple-Barrier + Meta-Labeling wrapper (Lopez de Prado, AFML ch 3).

Wraps the plateau-#1 grail (rsi2_regime, ETHUSD 1h) with a binary
classifier that decides "act / skip" for each primary entry signal.

Pipeline:
  1. Generate primary entries via the existing signal builder + simulator.
  2. Triple-barrier label each entry: profit (+1) / stop (-1) / timeout (0).
  3. Build per-entry features (no look-ahead).
  4. Train LightGBM (fallback: sklearn LogisticRegression) on first 70% of
     trades; binary target = "trade ended net-positive".
  5. Filter test trades by P(profit) > 0.6 and recompute WR / PF / DD.
  6. Compare baseline vs meta-filtered, write results/meta_label_report.md.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest import load_csv, simulate, Trade  # noqa: E402
from grail_loop import sig_rsi2_regime  # noqa: E402
from strategies.indicators import sma, rsi, atr  # noqa: E402

ASSET, TF = "ETHUSD", "1h"
PARAMS = {"rsi_len": 2, "rsi_buy": 3, "sma_trend": 100,
          "slope_bars": 5, "exit_sma": 8}
EXIT_CFG = {"sl_atr": 3.5, "tp_atr": 4.0, "trail_atr": 5.0, "timeout": None}
FEE, SLIPPAGE = 0.001, 0.0005
TRAIN_FRAC, P_THRESHOLD = 0.7, 0.6

OUT_REPORT = ROOT / "results" / "meta_label_report.md"
DATA_CSV = ROOT / "data" / f"{ASSET}_{TF}.csv"


def metrics(trades: list[Trade]) -> dict:
    if not trades:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "ret": 0.0, "dd": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0}
    rets = np.array([t.ret_pct for t in trades])
    wins, losses = rets[rets > 0], rets[rets <= 0]
    gw = wins.sum() if len(wins) else 0.0
    gl = -losses.sum() if len(losses) else 0.0
    pf = gw / gl if gl > 0 else float("inf")
    eq = np.cumprod(1.0 + rets / 100.0)
    dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min() * 100
    return {"n": len(rets), "wr": len(wins) / len(rets) * 100, "pf": pf,
            "ret": (eq[-1] - 1.0) * 100, "dd": dd,
            "avg_win": wins.mean() if len(wins) else 0.0,
            "avg_loss": losses.mean() if len(losses) else 0.0}


def build_features(df: pd.DataFrame, entry_idxs: list[int]) -> pd.DataFrame:
    """Per-entry features from data observable at the entry bar (no leak)."""
    c, h, l = df["close"], df["high"], df["low"]
    a14, a50 = atr(h, l, c, 14), atr(h, l, c, 50)
    s100, s20 = sma(c, 100), sma(c, 20)
    ret1 = c.pct_change(1)
    cols = {
        "atr_ratio":   a14 / a50,
        "rsi14":       rsi(c, 14),
        "rsi2":        rsi(c, 2),
        "sma_slope":   (s100 - s100.shift(5)) / s100,
        "dist_sma100": (c - s100) / s100,
        "dist_sma20":  (c - s20) / s20,
        "vol_regime":  ret1.rolling(20).std() / ret1.rolling(100).std(),
        "ret1":        ret1,
        "ret5":        c.pct_change(5),
        "ret20":       c.pct_change(20),
    }
    rows = []
    for i in entry_idxs:
        ts = df.index[i]
        row = {k: v.iat[i] for k, v in cols.items()}
        row["bar_of_day"] = ts.hour
        row["day_of_week"] = ts.dayofweek
        rows.append(row)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def triple_barrier_label(t: Trade) -> int:
    if t.exit_reason == "tp":
        return 1
    if t.exit_reason in ("sl", "trail"):
        return -1
    return 0  # timeout / signal / eod


def get_classifier():
    try:
        from lightgbm import LGBMClassifier
        return (LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                               min_child_samples=10, subsample=0.8,
                               colsample_bytree=0.8, random_state=42, verbose=-1),
                "LightGBM")
    except Exception:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        return (Pipeline([("sc", StandardScaler()),
                          ("lr", LogisticRegression(max_iter=1000,
                                                    class_weight="balanced",
                                                    random_state=42))]),
                "LogisticRegression(sklearn)")


def main():
    df = load_csv(DATA_CSV)
    print(f"Loaded {len(df)} bars: {df.index[0]} -> {df.index[-1]}")

    entry_sig, exit_sig = sig_rsi2_regime(df, PARAMS)
    trades = simulate(df, entry_sig, exit_sig, fee=FEE, slippage=SLIPPAGE,
                      sl_atr=EXIT_CFG["sl_atr"], tp_atr=EXIT_CFG["tp_atr"],
                      trail_atr=EXIT_CFG["trail_atr"],
                      timeout=EXIT_CFG["timeout"])
    print(f"Primary signals -> {len(trades)} executed trades")

    # entry_time is the timestamp of the bar that fills (next bar after signal).
    ts_secs = ((df.index - pd.Timestamp("1970-01-01", tz="UTC"))
               .total_seconds().astype("int64"))
    ts_to_bar = {int(t): i for i, t in enumerate(ts_secs)}
    entry_idxs = [ts_to_bar[t.entry_time] for t in trades]

    tb = [triple_barrier_label(t) for t in trades]
    n_tp, n_sl, n_to = tb.count(1), tb.count(-1), tb.count(0)
    print(f"Triple-barrier: profit={n_tp}, stop={n_sl}, timeout={n_to}")

    # AFML meta-label: side already chosen by primary -> classify go/no-go.
    y = np.array([1 if t.ret_pct > 0 else 0 for t in trades])
    feats = build_features(df, entry_idxs)
    print(f"Feature matrix: {feats.shape}")

    n_train = int(len(trades) * TRAIN_FRAC)
    if n_train < 20 or len(trades) - n_train < 10:
        raise SystemExit("Not enough trades for train/test split.")
    X_tr, X_te = feats.iloc[:n_train].values, feats.iloc[n_train:].values
    y_tr, y_te = y[:n_train], y[n_train:]
    test_trades = trades[n_train:]

    clf, clf_name = get_classifier()
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    print(f"Trained {clf_name}: train n={len(y_tr)} pos={y_tr.sum()} | "
          f"test n={len(y_te)} pos={y_te.sum()}")

    keep = proba > P_THRESHOLD
    meta_test = [t for t, k in zip(test_trades, keep) if k]
    m_full = metrics(trades)
    m_base = metrics(test_trades)
    m_meta = metrics(meta_test)

    deciles = pd.qcut(proba, q=min(5, len(proba)), duplicates="drop")
    cal = (pd.DataFrame({"p": proba, "y": y_te, "bin": deciles})
             .groupby("bin", observed=True)
             .agg(n=("y", "size"), avg_p=("p", "mean"), hit=("y", "mean")))

    def fmt(m):
        return (f"| {m['n']} | {m['wr']:.2f} | {m['pf']:.3f} | {m['ret']:.2f} "
                f"| {m['dd']:.2f} | {m['avg_win']:.3f} | {m['avg_loss']:.3f} |")

    d_wr = m_meta["wr"] - m_base["wr"]
    d_pf = m_meta["pf"] - m_base["pf"]
    d_dd = m_meta["dd"] - m_base["dd"]  # positive = shallower DD
    n_drop = m_base["n"] - m_meta["n"]

    if m_meta["n"] == 0:
        verdict = "**FILTER REJECTS ALL TEST TRADES** - threshold too strict."
    elif d_wr > 1 and d_pf > 0 and d_dd >= -1:
        verdict = ("**MODEST IMPROVEMENT**: meta-filter raises WR and PF "
                   "without materially worsening DD on the holdout.")
    elif d_wr > 0 and d_pf > 0:
        verdict = ("**BORDERLINE**: small directional gain on WR/PF that "
                   "could easily be noise at this sample size.")
    else:
        verdict = ("**NO IMPROVEMENT**: meta-filter does NOT improve the "
                   "holdout metrics. Consistent with AFML's warning that "
                   "meta-labeling only helps when the primary leaves "
                   "exploitable structure in the residuals.")

    lines = [
        "# Triple-Barrier + Meta-Labeling Report", "",
        f"- Primary: `rsi2_regime` {ASSET} {TF}",
        f"- Params: `{json.dumps(PARAMS)}`",
        f"- Exit cfg: `{json.dumps(EXIT_CFG)}`",
        f"- Fees/slip: {FEE*100:.3f}% / {SLIPPAGE*100:.3f}% per side",
        f"- Train/test split: first {int(TRAIN_FRAC*100)}% / last "
        f"{100-int(TRAIN_FRAC*100)}% of trades (chronological)",
        f"- Classifier: **{clf_name}**",
        f"- Meta-label keep threshold: P(profit) > {P_THRESHOLD}", "",
        "## Triple-barrier outcomes (full sample)", "",
        f"- Profit barrier hit (+1): **{n_tp}**",
        f"- Stop / trail barrier hit (-1): **{n_sl}**",
        f"- Timeout / signal exit (0): **{n_to}**", "",
        "## Comparison", "",
        "| Set | Trades | WR % | PF | Total Ret % | Max DD % | Avg Win % | Avg Loss % |",
        "|---|---|---|---|---|---|---|---|",
        f"| Primary (full sample) {fmt(m_full)}",
        f"| Baseline (test 30%, no filter) {fmt(m_base)}",
        f"| Meta-filtered (test 30%, p>{P_THRESHOLD}) {fmt(m_meta)}", "",
        f"- Trades filtered out: **{n_drop}** of {m_base['n']} "
        f"({n_drop / max(m_base['n'],1) * 100:.1f}%)",
        f"- DeltaWR = **{d_wr:+.2f} pp**, DeltaPF = **{d_pf:+.3f}**, "
        f"DeltaDD = **{d_dd:+.2f} pp** (positive = shallower DD)", "",
        "## Classifier calibration on holdout (5 buckets of P(profit))", "",
        "| Bucket | n | avg P(profit) | actual hit rate |",
        "|---|---|---|---|",
    ]
    for bucket, row in cal.iterrows():
        lines.append(f"| {bucket} | {int(row['n'])} | "
                     f"{row['avg_p']:.3f} | {row['hit']:.3f} |")

    lines += [
        "", "## Verdict", "", verdict, "",
        "## Notes", "",
        "- Meta-label binary target = `1 if trade.ret_pct > 0 else 0` "
        "(after fees), applied AFTER triple-barrier exit. The triple-barrier "
        "outcome is reported separately for context; the AFML formulation "
        "collapses TP -> 1 and SL/TO -> 0, but here a vertical-barrier exit "
        "can still be net-profitable, so we use realised PnL sign as the target.",
        "- Features use only data observable at the entry bar (no look-ahead).",
        "- Train/test is chronological (no shuffling) - leakage-safe.",
        "- A single train/test split on a few dozen test trades is "
        "high-variance; bigger sample (more assets / longer history) is needed "
        "for a definitive call.",
        "",
    ]
    OUT_REPORT.write_text("\n".join(lines))
    print(f"Wrote {OUT_REPORT}")
    print(f"Baseline test:  n={m_base['n']} WR={m_base['wr']:.2f} "
          f"PF={m_base['pf']:.3f} DD={m_base['dd']:.2f}")
    print(f"Meta-filtered:  n={m_meta['n']} WR={m_meta['wr']:.2f} "
          f"PF={m_meta['pf']:.3f} DD={m_meta['dd']:.2f}")


if __name__ == "__main__":
    main()
