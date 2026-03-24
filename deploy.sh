#!/usr/bin/env bash
# deploy.sh — Full pipeline: rsync code → setup → run 3 seeds → rsync results back.
#
# Usage:
#   ./deploy.sh <ssh_addr> <ssh_port>
#   e.g. ./deploy.sh ssh6.vast.ai 12345
#
# VPS fallback for results (fill in if you have one):
#   VPS_HOST=mc@vps.yourdomain.com ./deploy.sh ...
set -euo pipefail

SSH_ADDR="${1:?Usage: ./deploy.sh <ssh_addr> <ssh_port>}"
SSH_PORT="${2:?Usage: ./deploy.sh <ssh_addr> <ssh_port>}"
VPS_HOST="${VPS_HOST:-}"

SSH_KEY="$HOME/.ssh/id_rsa"
SSH_OPTS="-i $SSH_KEY -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15"
REMOTE="root@$SSH_ADDR"
REMOTE_DIR="/root/jellyfih"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================="
echo " jellyfih deploy"
echo " Target: $REMOTE:$SSH_PORT"
echo " Local:  $LOCAL_DIR"
echo " $(date)"
echo "========================================="

# 1. Upload code
echo ""
echo "--- [1/4] Uploading code ---"
rsync -avz --exclude='.git' --exclude='output/' --exclude='.venv/' \
    -e "ssh $SSH_OPTS" \
    "$LOCAL_DIR/" \
    "$REMOTE:$REMOTE_DIR/"

# 2. Run setup
echo ""
echo "--- [2/4] Running setup ---"
ssh $SSH_OPTS "$REMOTE" "bash $REMOTE_DIR/setup_cloud.sh"

# 3. Queue experiments (runs in background via nohup so SSH disconnect is safe)
echo ""
echo "--- [3/4] Launching experiments (nohup, detached) ---"
LOCAL_IP=$(curl -s https://api.ipify.org 2>/dev/null || echo "")

ssh $SSH_OPTS "$REMOTE" bash << ENDSSH
    export PATH="\$HOME/.local/bin:\$PATH"
    cd $REMOTE_DIR
    mkdir -p logs

    # Write the LOCAL_HOST for rsync-back into run_experiments.sh env
    export LOCAL_HOST="${LOCAL_IP:+mc@$LOCAL_IP}"
    export VPS_HOST="$VPS_HOST"
    export REMOTE_DEST="$LOCAL_DIR/output/cloud"

    nohup bash run_experiments.sh > logs/run_experiments.log 2>&1 &
    BGPID=\$!
    echo "Experiments running as PID \$BGPID"
    echo \$BGPID > logs/experiments.pid
    echo "Tail logs with: ssh $SSH_OPTS $REMOTE tail -f $REMOTE_DIR/logs/run_experiments.log"
ENDSSH

echo ""
echo "--- [4/4] Done ---"
echo ""
echo "Experiments are running in the background on the instance."
echo "Monitor progress:"
echo "  ssh $SSH_OPTS $REMOTE tail -f $REMOTE_DIR/logs/run_experiments.log"
echo ""
echo "When complete, results land in output/cloud/ here (or on VPS if local unreachable)."
echo "To manually pull results early:"
echo "  rsync -avz -e 'ssh $SSH_OPTS' $REMOTE:$REMOTE_DIR/output/ ./output/cloud/"
echo ""
echo "To destroy instance when done (check vast.ai dashboard for ID):"
echo "  vastai destroy instance <INSTANCE_ID>"
