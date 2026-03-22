#!/usr/bin/env bash
# deploy/bootstrap.sh — runs ON the vast.ai instance after SSH in.
#
# Usage: bash bootstrap.sh <GITHUB_TOKEN> [BRANCH]
#
# GITHUB_TOKEN: classic token or fine-grained with repo read access.
#   Create at: https://github.com/settings/tokens
#
# Example:
#   bash bootstrap.sh ghp_xxxxxxxxxxxxxxxxxxxx march

set -euo pipefail

GITHUB_TOKEN="${1:?Usage: bootstrap.sh <GITHUB_TOKEN> [BRANCH]}"
BRANCH="${2:-march}"
REPO="c-m-x-r/jellyfih"
WORKDIR="/root/jellyfih"
N_GENS=50

echo "======================================================"
echo " Jellyfih vast.ai bootstrap"
echo " branch: $BRANCH  |  gens: $N_GENS"
echo "======================================================"

echo ""
echo "=== [1/6] System deps ==="
apt-get update -qq
apt-get install -y -qq ffmpeg tmux rsync curl git python3 python3-pip
echo "  OK"

echo ""
echo "=== [2/6] Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"' >> /root/.bashrc
uv --version
echo "  OK"

echo ""
echo "=== [3/6] Cloning repo ==="
mkdir -p "$WORKDIR"
cd "$WORKDIR"
if [ -d "$WORKDIR/.git" ]; then
    echo "  Repo already exists — pulling latest"
    git remote set-url origin "https://${GITHUB_TOKEN}@github.com/${REPO}.git"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo "  Initialising git in existing directory (preserving output/)"
    git init
    git remote add origin "https://${GITHUB_TOKEN}@github.com/${REPO}.git"
    git fetch origin "$BRANCH"
    git checkout -b "$BRANCH" --track "origin/$BRANCH"
fi
echo "  OK — $(git log --oneline -1)"

echo ""
echo "=== [4/6] Installing Python deps ==="
cd "$WORKDIR"
uv sync
echo "  OK"

echo ""
echo "=== [5/6] Checking GPU and checkpoint ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Fix: CUDA 12.x images ship a compat libcuda.so (e.g. 530) but vast.ai hosts
# run newer drivers (e.g. 580). Taichi needs an unversioned libcuda.so that
# resolves to the *host* driver, not the compat stub. Create the symlink if missing.
LIBCUDA_DIR="/lib/x86_64-linux-gnu"
if [ ! -e "$LIBCUDA_DIR/libcuda.so" ] && [ -e "$LIBCUDA_DIR/libcuda.so.1" ]; then
    echo "  Fixing libcuda.so symlink (driver/compat mismatch workaround)"
    ln -sf "$LIBCUDA_DIR/libcuda.so.1" "$LIBCUDA_DIR/libcuda.so"
    ldconfig
fi

mkdir -p "$WORKDIR/output"
if [ -f "$WORKDIR/output/checkpoint.pkl" ]; then
    echo "  checkpoint.pkl found — run will RESUME from last checkpoint"
else
    echo "  No checkpoint — starting fresh from gen 0"
fi

echo ""
echo "=== [6/6] Starting evolve.py ==="
# Write tmux config (WSL clip.exe binding omitted — not available on container)
cat > /root/.tmux.conf << 'TMUXCONF'
# --- General Settings ---
set -g mouse on               # Enable mouse mode
set -g base-index 1           # Start windows at 1
setw -g pane-base-index 1     # Start panes at 1
set -g renumber-windows on    # Automatically renumber windows
set -g status-interval 5      # Update status bar more often

# --- Key Remaps ---
unbind C-b
set -g prefix `
bind ` send-prefix

bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"

# --- Selection ---
setw -g mode-keys vi

# --- Automatic Layout ---
set-hook -g session-created 'split-window -h; split-window -v; select-pane -t 1'
TMUXCONF

# vast.ai wraps SSH in its own tmux session (ssh_tmux).
# If we're already inside tmux, use TMUX='' to allow nesting; otherwise create normally.
CMD="cd $WORKDIR && export PATH=\"\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH\" && uv run python evolve.py --no-thermal --gens $N_GENS 2>&1 | tee output/run.log"

tmux kill-session -t evo 2>/dev/null || true
TMUX='' tmux new-session -d -s evo "$CMD"

echo ""
echo "======================================================"
echo " Bootstrap complete!"
echo ""
echo " Switch to run:  tmux switch -t evo   (if inside vast.ai ssh_tmux)"
echo " Attach direct:  tmux attach -t evo   (if SSHing fresh)"
echo " Follow log:     tail -f $WORKDIR/output/run.log"
echo " GPU status:     watch -n2 nvidia-smi"
echo " Progress:       tail -1 $WORKDIR/output/evolution_log.csv"
echo "======================================================"
