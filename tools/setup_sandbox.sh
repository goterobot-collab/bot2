#!/usr/bin/env bash
# Setup a Hetzner (or any Ubuntu/Debian VPS) as a grail-hunt sandbox.
#
# Run as root or with sudo on a fresh Ubuntu 22.04+ / Debian 12+ VPS.
#
# What it does:
#   1. Installs system deps (python3, pip, git, tmux, htop)
#   2. Clones this repo (you provide GH token + branch)
#   3. Installs Python requirements
#   4. Fetches OHLCV from codeload.github.com
#   5. Generates 4h/1d resamples
#   6. Launches N workers in a tmux session (N = cores - 1)
#   7. Starts a committer cron that pushes results every 30 minutes
#
# Usage (from your Hetzner VPS shell, as root):
#   export GH_TOKEN=ghp_xxx           # needs repo push access
#   export GH_REPO=goterobot-collab/bot2
#   export GH_BRANCH=claude/verify-trading-strategies-Fnf0P
#   curl -fsSL https://raw.githubusercontent.com/$GH_REPO/$GH_BRANCH/tools/setup_sandbox.sh | bash
#
# After running:
#   tmux attach -t grail         # see all 12-ish workers
#   tail -f /root/bot2/logs/*.log
#   cat /root/bot2/results/grails_4h_1d.md
#
# To stop:
#   systemctl stop grail-commit.timer
#   tmux kill-session -t grail
set -euo pipefail

: "${GH_TOKEN:?need GH_TOKEN env var (ghp_xxx or github_pat_xxx)}"
: "${GH_REPO:=goterobot-collab/bot2}"
: "${GH_BRANCH:=claude/verify-trading-strategies-Fnf0P}"
: "${WORK_DIR:=/root/bot2}"
: "${COMMIT_EVERY_MIN:=30}"

echo "== Grail Sandbox Setup =="
echo "Repo:   $GH_REPO"
echo "Branch: $GH_BRANCH"
echo "Cores:  $(nproc)"
echo "RAM:    $(free -h | awk '/^Mem:/{print $2}')"

# ---- 1. system deps --------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git tmux htop ca-certificates curl

# ---- 2. clone repo ---------------------------------------------------------
if [ -d "$WORK_DIR/.git" ]; then
    echo "[info] $WORK_DIR already a repo; pulling"
    cd "$WORK_DIR"
    git remote set-url origin "https://${GH_TOKEN}@github.com/${GH_REPO}.git"
    git fetch origin "$GH_BRANCH"
    git checkout "$GH_BRANCH"
    git pull origin "$GH_BRANCH" --rebase
else
    git clone --branch "$GH_BRANCH" \
        "https://${GH_TOKEN}@github.com/${GH_REPO}.git" "$WORK_DIR"
    cd "$WORK_DIR"
fi

# ---- 3. python deps --------------------------------------------------------
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# ---- 4. fetch data ---------------------------------------------------------
cd "$WORK_DIR"
if [ ! -s data/ETHUSD_1h.csv ]; then
    echo "[info] fetching OHLCV from codeload.github.com"
    python3 tools/convert_cryptopredictions.py ADAUSD DOGEUSD ETHUSD BTCUSD || true
fi

# ---- 5. generate 4h/1d resamples ------------------------------------------
python3 - <<'PY'
import pandas as pd
from pathlib import Path
for p in sorted(Path("data").glob("*_1h.csv")):
    asset = p.stem.replace("_1h","")
    df = pd.read_csv(p)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("dt").sort_index()
    for tf, rule in [("4h","4h"), ("1d","1D")]:
        out = Path(f"data/{asset}_{tf}.csv")
        if out.exists(): continue
        r = df.resample(rule).agg({"timestamp":"first","open":"first",
            "high":"max","low":"min","close":"last","volume":"sum"}).dropna()
        r.reset_index(drop=True).to_csv(out, index=False)
        print(f"  wrote {out} ({len(r)} bars)")
PY

# ---- 6. launch workers in tmux --------------------------------------------
CORES=$(nproc)
N_WORKERS=$(( CORES > 1 ? CORES - 1 : 1 ))
echo "[info] launching $N_WORKERS workers in tmux session 'grail'"

mkdir -p "$WORK_DIR/logs"
tmux kill-session -t grail 2>/dev/null || true
tmux new-session -d -s grail -n w1 "cd $WORK_DIR && python3 grail_loop.py --iter 500000 --seed 100001 --report-every 50000 2>&1 | tee logs/hetz_w1.log"
for i in $(seq 2 $N_WORKERS); do
    tmux new-window -t grail -n "w$i" "cd $WORK_DIR && python3 grail_loop.py --iter 500000 --seed $((100000+i)) --report-every 50000 2>&1 | tee logs/hetz_w$i.log"
done

# ---- 7. committer service --------------------------------------------------
cat > /usr/local/bin/grail-commit.sh <<EOF
#!/usr/bin/env bash
set -e
cd $WORK_DIR
git add results/*.md results/*.json 2>/dev/null || true
if ! git diff --cached --quiet; then
    git -c user.email=hetz-sandbox@local \
        -c user.name="hetz-sandbox" \
        commit -m "hetz: auto-snapshot \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    git push origin $GH_BRANCH || true
fi
EOF
chmod +x /usr/local/bin/grail-commit.sh

cat > /etc/systemd/system/grail-commit.service <<EOF
[Unit]
Description=Push grail results snapshot to GitHub
[Service]
Type=oneshot
ExecStart=/usr/local/bin/grail-commit.sh
EOF

cat > /etc/systemd/system/grail-commit.timer <<EOF
[Unit]
Description=Auto-commit grail results every ${COMMIT_EVERY_MIN}m
[Timer]
OnBootSec=5min
OnUnitActiveSec=${COMMIT_EVERY_MIN}min
Unit=grail-commit.service
[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now grail-commit.timer

echo
echo "== DONE =="
echo "Attach workers:  tmux attach -t grail   (ctrl-b n/p to switch, ctrl-b d to detach)"
echo "Logs:            tail -f $WORK_DIR/logs/hetz_w*.log"
echo "Auto-commit:     systemctl status grail-commit.timer"
echo "Results pushed every ${COMMIT_EVERY_MIN}m to $GH_REPO / $GH_BRANCH"
