#!/usr/bin/env bash
# sync_results.sh — Pull output from a remote instance to output/cloud/<label>/
#
# Usage:
#   ./sync_results.sh <ssh_addr> <ssh_port> [label]
#
#   label defaults to the hostname portion of ssh_addr
#
# Examples:
#   ./sync_results.sh 217.171.200.22 59022 3080x4
#   ./sync_results.sh ssh3.vast.ai    26458 4090
#
# Can be run repeatedly — rsync only transfers new/changed files.

set -euo pipefail

SSH_ADDR="${1:?Usage: ./sync_results.sh <ssh_addr> <ssh_port> [label]}"
SSH_PORT="${2:?Usage: ./sync_results.sh <ssh_addr> <ssh_port> [label]}"
LABEL="${3:-${SSH_ADDR%%.*}}"

SSH_KEY="$HOME/.ssh/id_rsa"
SSH_OPTS="-i $SSH_KEY -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15"
REMOTE="root@$SSH_ADDR"
REMOTE_DIR="/root/jellyfih/output"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)/output/cloud/$LABEL"

mkdir -p "$LOCAL_DIR"

echo "Syncing $REMOTE:$REMOTE_DIR/ → $LOCAL_DIR/"
rsync -avz --progress \
    -e "ssh $SSH_OPTS" \
    "$REMOTE:$REMOTE_DIR/" \
    "$LOCAL_DIR/"

echo ""
echo "Done. Files in $LOCAL_DIR:"
ls -lh "$LOCAL_DIR"
