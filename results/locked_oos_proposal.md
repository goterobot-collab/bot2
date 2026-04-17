# Locked 20% OOS Holdout — Upgrade Proposal

Status: **proposal only** — `grail_loop.py` is not modified (workers running).
Companion script: `tools/holdout_test.py` (replays existing 1059 grails on the locked tail).

Source of priority: `results/ANTI_OVERFIT_RESEARCH.md` §"Priority upgrades" item 4 —
*"FINAL: locked 20% OOS holdout — only touched after all other filters pass."*

---

## 1. Diff plan for `grail_loop.py`

Two surgical changes. All line numbers refer to the file at the time of writing
(see `Read` snapshot of `/home/user/bot2/grail_loop.py`).

### Change A — replace `evaluate()` (current lines 350-372)

The current function does a 50/50 walk-forward (`half = n // 2`) and returns
`{"full", "h1", "h2"}`. We split 40/40/20: train (first 40%), validate (next 40%),
holdout (final 20%, **never read inside the loop**).

```python
def evaluate(df, fam, params, exit_cfg, fee, slippage, asset, tf):
    """Walk-forward 40/40/20. Loop only reads train+validate (h1+h2).
    The final 20% is the LOCKED HOLDOUT — never evaluated here.
    Use tools/holdout_test.py post-hoc to score survivors on it ONCE."""
    builder = SPACES[fam]["sig"]
    try:
        ent, ex = builder(df, params)
    except Exception:
        return None

    def _eval(sub_df, sub_ent, sub_ex):
        trades = simulate(sub_df, sub_ent, sub_ex, fee=fee, slippage=slippage,
                          sl_atr=exit_cfg.get("sl_atr"),
                          tp_atr=exit_cfg.get("tp_atr"),
                          trail_atr=exit_cfg.get("trail_atr"),
                          timeout=exit_cfg.get("timeout"))
        return metrics_from_trades(fam, asset, tf, trades, sub_df)

    n = len(df)
    i_train_end    = int(n * 0.40)            # 0   .. 40%
    i_validate_end = int(n * 0.80)            # 40% .. 80%
    # 80%..100% is HOLDOUT — DO NOT touch in this function.

    train_df    = df.iloc[:i_train_end]
    validate_df = df.iloc[i_train_end:i_validate_end]
    is_df       = df.iloc[:i_validate_end]    # train+validate combined ("in-sample")

    return {
        # `full` is renamed to `is` (in-sample = first 80%) so callers stop
        # seeing holdout returns by accident.
        "is": _eval(is_df,       ent.iloc[:i_validate_end], ex.iloc[:i_validate_end]),
        "h1": _eval(train_df,    ent.iloc[:i_train_end],    ex.iloc[:i_train_end]),
        "h2": _eval(validate_df, ent.iloc[i_train_end:i_validate_end],
                                  ex.iloc[i_train_end:i_validate_end]),
    }
```

### Change B — update `passes()` callsite (current lines 474-501)

The main loop currently reads `res["full"]` (line 474) and computes `is_grail`
from full+h1+h2. Change those references to the new `"is"` key, keep the same
filter, and store an `is_*` block in the row instead of full-history numbers.

Replace lines 474-501 with:

```python
        m = res["is"]; m1 = res["h1"]; m2 = res["h2"]
        row = {
            "iter": i, "strategy": fam, "asset": asset, "tf": tf,
            "params": params, "exit_cfg": exit_cfg,
            # in-sample (train+validate, first 80%) metrics
            "trades": m.trades, "wr": m.wr, "pf": m.profit_factor,
            "ret": m.total_return_pct, "dd": m.max_drawdown_pct,
            # walk-forward halves (train=h1, validate=h2)
            "h1_trades": m1.trades, "h1_wr": m1.wr, "h1_pf": m1.profit_factor,
            "h1_ret": m1.total_return_pct, "h1_dd": m1.max_drawdown_pct,
            "h2_trades": m2.trades, "h2_wr": m2.wr, "h2_pf": m2.profit_factor,
            "h2_ret": m2.total_return_pct, "h2_dd": m2.max_drawdown_pct,
            "split": "40_40_20",   # marker so consumers can distinguish files
        }
        jsonl_f.write(json.dumps(row) + "\n")

        if m.total_return_pct > best_ret and m.trades > args.min_trades:
            best_ret = m.total_return_pct; best_row = row

        def passes(mm, min_tr):
            return (mm.trades > min_tr and mm.wr > args.min_wr * 100
                    and mm.total_return_pct > 0
                    and mm.max_drawdown_pct > -args.grail_dd
                    and mm.profit_factor > args.min_pf)
        # Each half is now ~40% of history (was 50%). Trade counts will roughly
        # halve again — adjust the half-min floor or expect fewer grails.
        half_min = max(10, int(args.min_trades * 0.4))
        is_grail = passes(m, args.min_trades) and passes(m1, half_min) and passes(m2, half_min)
```

Also rename the output paths around lines 431-432 so v3 results don't overwrite
the v2 pool of 1059 (which we still need for the migration in section 2):

```python
    grails_md    = OUT_DIR / "grails_loop_wf_v3.md"
    grails_jsonl = OUT_DIR / "grails_loop_wf_v3.jsonl"
```

### Lines that DO NOT change

Argparse (415-428), signal handler (453-455), main loop scaffolding outside
the slice above, and the `Grail` dataclass — all stay identical. The walk-forward
*shape* of the loop is unchanged; only the slice boundaries and the key name move.

### Why this is safe vs. running workers

Workers hold the *current* `grail_loop.py` bytecode for the life of the process,
so editing the file doesn't disturb them. But mixing v2 and v3 rows into
`grails_loop_wf.jsonl` would corrupt downstream tools (`plateau_test.py`,
`pbo.py`). New filenames (`_v3.jsonl`) avoid that.

---

## 2. Migration strategy — score existing 1059 grails on the holdout

Re-running 2M iterations is wasteful. Each iteration recomputes signals from
scratch on the *full* history; we only need each unique config evaluated **once**
on the **last 20%** of its asset's CSV.

### Plan

1. Read `results/grails_loop_wf.jsonl` (~2.0M rows).
2. Filter to rows that pass the in-loop walk-forward grail filter
   (`wr>60 ∧ ret>0 ∧ pf>1 ∧ dd>-30 ∧ trades>=50` and same on h1+h2). This
   reproduces the 1059 grails the user counts.
3. Deduplicate by `(strategy, asset, tf, params, exit_cfg)` — the loop samples
   with replacement, so the same config can be a "grail" in multiple rows.
4. For each unique grail, slice `df.iloc[int(n*0.80):]` and call the **same**
   `evaluate`-style core (build signals on the slice, simulate, metrics).
5. Apply the **same** grail filter on the holdout. Emit a markdown report
   (`results/holdout_report.md`) and a jsonl side-table.

### Cost estimate

- 1059 grails × 1 evaluation each ≈ 1059 simulator calls.
- Plateau test runs ~24 neighbors × 1059 = ~25k sim calls in ~30 minutes.
- Holdout test is ~25× cheaper → minutes, not hours.

### Files written by the migration

| File | Purpose |
|------|---------|
| `results/holdout_report.md` | Human-readable: which grails survive |
| `results/holdout_results.jsonl` | Machine-readable per-grail metrics on tail |

The script (`tools/holdout_test.py`, attached) does **not** mutate
`grails_loop_wf.jsonl` — it only reads it. Safe to run alongside workers.

---

## 3. Acceptance criteria

Predicted survival, anchored on the plateau test (`plateau_report.md`:
**13/115 = 11.3%** of unique WF grails pass plateau).

| Outcome | Survivors / 1059 | Interpretation |
|---------|------------------|----------------|
| Best plausible | ~150 (14%) | Plateau survival is upper bound; holdout is independent so could be slightly higher if regime is similar |
| **Expected** | **~110-160 (10-15%)** | Tracks the plateau base rate; consistent with WF being a partial guard against overfitting |
| Concerning | < 50 (5%) | Indicates the in-sample 40/40 WF is itself overfit to the search; revisit selection bias |
| Catastrophic | < 10 (1%) | Search is finding 1h indicator noise (matches Anti-Overfit research, archetype #10 rated C) |
| Suspicious | > 300 (28%) | Holdout regime probably benign (e.g., bull continuation); does NOT validate the strategies |

**Pass bar for the whole pipeline:**
- a grail must (a) pass WF in-loop, (b) pass plateau (≥75% neighbors), (c) pass
  holdout with the same WR/PF/DD/return filter and trades ≥ 10 on the ~20% slice.
- We expect maybe **5-20 grails total** to survive all three. That is the
  realistic deployment shortlist.

---

## 4. Risks — keeping the holdout actually held-out

The whole value of a locked OOS is that it sees the strategy **once**. Any
reuse leaks the holdout into selection and makes its p-value meaningless
(López de Prado's "test-set looks at you" failure mode).

### Risks and mitigations

1. **Iterating after seeing holdout result.** If the user observes that
   `bb_trend_rejoin ETH 1h` failed holdout and tweaks the search to avoid that
   region, the holdout has been used for selection. **Mitigation:** treat
   holdout as a single-shot accept/reject. If any change is made to
   `grail_loop.py`, `plateau_test.py`, the param spaces, or the grail filter,
   the holdout slice must be re-cut (e.g., shift to a *different* untouched
   period or wait for genuinely new data).

2. **Implicit reuse via plateau / PBO.** If `plateau_test.py` ever uses the
   last 20%, plateau success becomes pre-tested holdout. **Mitigation:** the
   `evaluate()` change in §1 hides the holdout from any caller of `evaluate()`;
   only `tools/holdout_test.py` slices the tail explicitly.

3. **Dataset extension.** Pulling fresh CSV bars later moves the 80% boundary
   *forward*, potentially exposing previously-holdout bars to the search loop.
   **Mitigation:** when extending data, freeze the old 20% boundary as a
   timestamp (e.g., `HOLDOUT_START = "2024-06-01"`) and configure the 40/40/20
   relative split to round to that boundary. Or maintain `holdout_until.csv`
   per asset that records the cutoff used the day the holdout was scored.

4. **Multi-asset cherry-picking.** Even if one asset's holdout is sacred,
   running the test on 3 assets and reporting only the best is selection on
   the holdout. **Mitigation:** report all-assets-or-none and apply a
   Bonferroni correction (~3 assets × N grails) to the holdout p-value.

5. **Worker race.** A worker process started before the diff may still write
   v2-shaped rows to `grails_loop_wf.jsonl` after we write v3 to
   `grails_loop_wf_v3.jsonl`. The migration script in §2 only ever reads the
   v2 file — it doesn't care if v2 keeps appending — and consumers downstream
   know the split via the `split` field added in Change B.

6. **The 1059 themselves were selected on the same data the new "in-sample"
   covers.** This is fine — that's the whole point of locking holdout: the
   1059 are the candidates *coming out of* in-sample selection; the holdout
   gives an honest first-look estimate of OOS WR / PF on each.

---

## Summary

- Two-spot diff to `grail_loop.py`: shrink train/validate halves to 40/40,
  hide the 80–100% slice from `evaluate()`, write to a new v3 jsonl.
- New `tools/holdout_test.py` migrates the existing 1059 v2-grails onto the
  locked tail in minutes (no 2M-eval re-run).
- Expect ~10-15% (≈110-160) to clear the holdout filter; expect ~5-20 to
  clear holdout *and* plateau.
- Do not look at the holdout twice. After it speaks, the next iteration of
  the search needs a different held-out slice.
