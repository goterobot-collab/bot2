# Coordination — 3-way node bridge

GitHub-as-bridge for Mac ↔ Hetzner ↔ Sandbox (Anthropic).

## Node roles

| Node | What it does | What it does NOT do |
|------|--------------|---------------------|
| **Mac** | V8 dev, inject production-ready grails into live bots, orchestrate the other two | Doesn't run heavy CPU loops (too noisy) |
| **Hetzner** (CX32 or similar) | Production backtest queue: Optuna sweeps, forensic backtests, V7/V8 bots batch jobs, 17GB Binance DB | Doesn't design new strategies |
| **Sandbox** (Anthropic) | Grail hunt (11 novel families, see `strategies/novel_families.py`), anti-overfit pipeline (plateau / MC / PBO / DSR) | Can't reach Binance/exchange APIs — only codeload.github.com + raw.githubusercontent.com + HuggingFace |

## Files in this directory

- `queue_claims.json` — who's working on what. Every node updates its own `nodes.<node>` section when starting/finishing a batch.
- `hetzner_progress.json` — progress of Hetzner-side batches (populated by Hetzner).
- `sandbox_progress.json` — progress of Sandbox-side batches (populated by Sandbox).
- `mac_inbox.jsonl` — grails that survived all pipelines, ready for the Mac to inject into V8 live bots.
- `mac_outbox.jsonl` — grails from Mac to be validated by Sandbox (anti-overfit) or re-run by Hetzner.

## Protocol: claiming work

Before starting a batch, a node does:

1. `git pull origin claude/verify-trading-strategies-Fnf0P`
2. Read `coordination/queue_claims.json`
3. Check if another node has an active claim on the same batch
4. If not, write the claim under `nodes.<me>.claim`, set `updated_at` to UTC ISO
5. `git commit -m "<node>: claim batch <name> slice <range>"`
6. `git push`
7. Start the work

If two nodes race and both push, the second push gets rejected; loser
pulls, re-checks, takes an unclaimed slice, pushes again.

Stale claims (`updated_at` older than `_max_stale_hours`) may be reclaimed
by others.

## Protocol: publishing results

When a node finishes a slice:

1. Write results to `results/<batch>_<node>_<ts>.jsonl` (or `.jsonl.gz` if > 50 MB)
2. Clear own claim (`nodes.<me>.claim = null`)
3. Append a line to the inbox of the next consumer
   (e.g. Hetzner → `mac_inbox.jsonl` if grail survived forensic;
    Sandbox → `mac_inbox.jsonl` if grail survived anti-overfit)
4. Commit + push

## What NOT to push to this repo

- `activos_binance.db` (17 GB SQLite) — keep local to Hetzner, share via
  `ccxt` on-demand fetch or a dedicated LFS-backed repo
- `v8_bots.json` (live production config) — never leave Mac
- API keys, `.env`, webhook secrets
- Parquet / pickle files from pandas; stick to JSONL so all three nodes
  can parse without dependency mismatches

## Minimum schema for a grail row

All three nodes agree on this JSONL schema:

```json
{
  "strategy":   "connors_rsi",
  "symbol":     "ETHUSDT",
  "tf":         "15m",
  "params":     {"...": "..."},
  "exit":       {"sl_atr": 2.0, "tp_atr": 3.0, "trail_atr": null, "timeout": 48},
  "trades":     237,
  "wr":         68.3,
  "pf":         1.42,
  "ret_pct":    84.1,
  "dd_pct":    -18.2,
  "sharpe":     1.85,
  "verdict":    "CONFIRMED",
  "source":     "hetzner|sandbox|mac",
  "pipeline":   ["grail_filter", "walk_forward", "plateau", "monte_carlo"],
  "start_ts":   1704067200,
  "end_ts":     1776000000,
  "fees_rt":    0.003,
  "trades_list": [[1704067200, 1704070800, 2.3], ["..."]]
}
```

`pipeline` is the ordered list of stages this grail has passed. The Mac
only injects into V8 grails whose `pipeline` includes at least
`["grail_filter", "walk_forward", "plateau", "monte_carlo"]`.

## Git workflow

All three nodes work on the same branch: `claude/verify-trading-strategies-Fnf0P`.

- Frequent small commits > one giant push
- Always `git pull --rebase` before pushing
- If rebase conflicts on `queue_claims.json`: take the more recent `updated_at` as the winner
- No force-push, no rewriting history
