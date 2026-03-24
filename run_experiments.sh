#!/usr/bin/env bash
# run_experiments.sh — Queue 3 CMA-ES runs sequentially, then rsync output back.
#
# Rsync priority:
#   1. LOCAL_HOST  — your laptop/desktop (if reachable over SSH)
#   2. VPS_HOST    — fallback VPS (always attempted if LOCAL_HOST fails)
#
# Fill in the hosts below, or override via env vars:
#   LOCAL_HOST=user@1.2.3.4 VPS_HOST=user@vps.example.com ./run_experiments.sh
set -euo pipefail

LOCAL_HOST="${LOCAL_HOST:-}"          # e.g. mc@192.168.1.50  or your public IP
VPS_HOST="${VPS_HOST:-}"              # e.g. mc@vps.yourdomain.com
REMOTE_DEST="${REMOTE_DEST:-~/projects/jellyfih/output/cloud}"   # path on remote

SEEDS=(42 137 999)
REPO_DIR="$HOME/jellyfih"
export PATH="$HOME/.local/bin:$PATH"

cd "$REPO_DIR"

echo "========================================"
echo " jellyfih — 3-seed experiment queue"
echo " Seeds: ${SEEDS[*]}"
echo " $(date)"
echo "========================================"

# --- Run experiments ---
for SEED in "${SEEDS[@]}"; do
    RUN_ID="seed_${SEED}"
    LOG_DIR="output/${RUN_ID}"
    mkdir -p "$LOG_DIR"

    echo ""
    echo "--- Starting seed=$SEED  run-id=$RUN_ID  $(date) ---"

    uv run python evolve.py \
        --gens 50 \
        --seed "$SEED" \
        --run-id "$RUN_ID" \
        2>&1 | tee "$LOG_DIR/evolve.log"

    echo "--- Finished seed=$SEED  $(date) ---"
done

echo ""
echo "========================================"
echo " All 3 runs complete. Syncing output..."
echo "========================================"

sync_to() {
    local HOST="$1"
    echo "Attempting rsync to $HOST:$REMOTE_DEST ..."
    if rsync -avz --timeout=30 \
        -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10" \
        "$REPO_DIR/output/" \
        "$HOST:$REMOTE_DEST/"; then
        echo "rsync to $HOST succeeded."
        return 0
    else
        echo "rsync to $HOST failed."
        return 1
    fi
}

SYNCED=false

if [ -n "$LOCAL_HOST" ]; then
    if sync_to "$LOCAL_HOST"; then
        SYNCED=true
    fi
fi

if [ "$SYNCED" = false ] && [ -n "$VPS_HOST" ]; then
    echo "Local unreachable — trying VPS fallback..."
    if sync_to "$VPS_HOST"; then
        SYNCED=true
    fi
fi

if [ "$SYNCED" = false ]; then
    echo ""
    echo "WARNING: Could not rsync to any remote host."
    echo "Output is safe in $REPO_DIR/output/"
    echo "To rsync manually:"
    echo "  rsync -avz <HOST>:$REPO_DIR/output/ ./output/cloud/"
fi

echo ""
echo "Done. $(date)"
