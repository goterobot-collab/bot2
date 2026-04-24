#!/usr/bin/env bash
# Mac paralela infinite-ish loop: iterates through mac batches, runs hunter
# + 4-gate pipeline + commits. Stops when no new batch files appear OR
# a sentinel file coordination/MAC_INFINITE_STOP exists.
#
# Usage (from repo root):
#   nohup bash tools/mac_infinite_loop.sh > logs/mac_infinite_loop.log 2>&1 &
#
# Design:
# - Each "iteration" processes ONE batch (3700..)
# - For batch N: if results/mac_m{i}_1h_4h_1d_SHORTLIST.md exists, skip (done)
# - Otherwise: hunt wave m{i} with WR>=60 -> pipeline 4-gate -> commit -> push
# - Where m{i} maps to batch 3700+i-1 (m1=3700, m5=3704, m10=3709, ...)
#
# Safety:
# - NEVER pushes -f, NEVER touches sandbox_* or tv2_batch* files
# - Respects MAC_V8_COVERED_DEDUP via the pipeline tools
# - Exits if no new batch file is detected (prevents infinite sleep)

set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
STOP_SENTINEL="$ROOT/coordination/MAC_INFINITE_STOP"
MIN_WR=60
MAX_NO_NEW_BATCH_SLEEPS=3
NO_NEW_COUNT=0

ts() { date +"%Y-%m-%d %H:%M:%S"; }

log() { echo "[$(ts)] $*"; }

check_stop() {
    if [[ -f "$STOP_SENTINEL" ]]; then
        log "STOP sentinel found at $STOP_SENTINEL. Exiting."
        exit 0
    fi
}

# Discover latest batch file -> pick next wave number
iteration=0
while true; do
    check_stop
    iteration=$((iteration + 1))

    # List batch files sorted
    batch_files=( $(ls "$ROOT"/strategies_v7/strategies_mac_batch37*.py 2>/dev/null | sort) )
    if [[ ${#batch_files[@]} -eq 0 ]]; then
        log "No mac batch files found. Waiting for a batch file to be created..."
        NO_NEW_COUNT=$((NO_NEW_COUNT + 1))
        if [[ $NO_NEW_COUNT -ge $MAX_NO_NEW_BATCH_SLEEPS ]]; then
            log "Waited $NO_NEW_COUNT idle cycles - exiting (batch queue empty)."
            exit 0
        fi
        sleep 60
        continue
    fi

    processed_any=0

    for batch_file in "${batch_files[@]}"; do
        check_stop
        batch_num=$(basename "$batch_file" | sed -E 's/strategies_mac_batch([0-9]+)\.py/\1/')
        wave_idx=$((batch_num - 3700 + 1))
        wave="m${wave_idx}"

        shortlist="$ROOT/results/mac_${wave}_1h_4h_1d_SHORTLIST.md"
        pipeline_out="$ROOT/results/mac_${wave}_pipeline_results.json"

        if [[ -f "$shortlist" && -f "$pipeline_out" ]]; then
            continue  # already fully processed
        fi

        log "=== iteration $iteration: processing wave $wave (batch $batch_num) ==="

        # 1. Hunter (skip if shortlist exists)
        if [[ ! -f "$shortlist" ]]; then
            log "  [hunter] python3 tools/mac_hunter_runner.py --wave $wave --tfs 1h 4h 1d --min-wr $MIN_WR"
            python3 tools/mac_hunter_runner.py --wave "$wave" --tfs 1h 4h 1d --min-wr "$MIN_WR" \
                >> "logs/mac_${wave}_1h_4h_1d.log" 2>&1 || {
                    log "  [hunter] FAILED for wave $wave"
                    continue
                }
            log "  [hunter] done."
        else
            log "  [hunter] shortlist exists, skipping"
        fi

        # 2. Pipeline 4-gate
        if [[ ! -f "$pipeline_out" ]]; then
            log "  [pipeline] python3 tools/mac_pipeline.py --waves $wave --min-wr-eligible 65 --min-n 15"
            python3 tools/mac_pipeline.py --waves "$wave" --min-wr-eligible 65 --min-n 15 \
                >> "logs/mac_${wave}_pipeline.log" 2>&1 || {
                    log "  [pipeline] FAILED for wave $wave"
                    continue
                }
            log "  [pipeline] done."
        fi

        # 3. Commit + push
        cd "$ROOT"
        if ! git diff --quiet || [[ -n "$(git status --porcelain)" ]]; then
            git pull --rebase origin claude/grail-hunt-bot-noeIq > /dev/null 2>&1 || true

            # Stage only mac domain files
            git add "strategies_v7/strategies_mac_batch${batch_num}.py" 2>/dev/null || true
            git add "results/mac_${wave}_1h_4h_1d_SHORTLIST.md" 2>/dev/null || true
            git add "results/mac_${wave}_1h_4h_1d_PROMOTED.json" 2>/dev/null || true
            git add "results/mac_${wave}_1h_4h_1d_progress.json" 2>/dev/null || true
            git add "results/mac_${wave}_pipeline_results.json" 2>/dev/null || true

            # Safety - refuse to commit if any sandbox_* or tv2_batch* staged
            if git diff --cached --name-only | grep -qE "(sandbox_|strategies_tv2_batch)"; then
                log "  [commit] SAFETY ABORT: cross-domain file staged"
                git reset
                continue
            fi

            staged=$(git diff --cached --name-only)
            if [[ -z "$staged" ]]; then
                log "  [commit] nothing to commit for $wave"
            else
                grails=$(grep -c "^| [0-9]" "$shortlist" 2>/dev/null || echo 0)
                confirmed=$(python3 -c "
import json
try:
    d = json.load(open('$pipeline_out'))
    n = sum(1 for r in d if r.get('gate3_status') == 'PASS')
    print(n)
except Exception:
    print('?')
" 2>/dev/null)

                git commit -m "mac-loop iter $iteration: wave $wave raw=$grails CONFIRMED=$confirmed" > /dev/null 2>&1 && {
                    git push -u origin claude/grail-hunt-bot-noeIq > /dev/null 2>&1 && {
                        log "  [commit+push] wave $wave: raw=$grails CONFIRMED=$confirmed"
                    } || log "  [push] FAILED for wave $wave"
                } || log "  [commit] FAILED for wave $wave"
            fi
        fi

        processed_any=1
    done

    if [[ $processed_any -eq 0 ]]; then
        log "No unprocessed batches this iteration."
        NO_NEW_COUNT=$((NO_NEW_COUNT + 1))
        if [[ $NO_NEW_COUNT -ge $MAX_NO_NEW_BATCH_SLEEPS ]]; then
            log "All batches processed. Queue empty - exiting gracefully."
            log "To resume: add a new strategies_v7/strategies_mac_batch37<NN>.py and re-launch."
            exit 0
        fi
        log "Sleeping 60s before re-scan (idle cycle $NO_NEW_COUNT/$MAX_NO_NEW_BATCH_SLEEPS)"
        sleep 60
    else
        NO_NEW_COUNT=0
    fi
done
