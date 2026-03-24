#!/usr/bin/env bash
# setup_cloud.sh — Bootstrap jellyfih on a fresh Ubuntu cloud GPU instance.
# Expects code to already be present at ~/jellyfih (uploaded via rsync by deploy.sh).
# Run once after provisioning. Safe to re-run.
set -euo pipefail

REPO_DIR="$HOME/jellyfih"

echo "=== jellyfih cloud setup ==="

# 1. System deps
sudo apt-get update -qq
sudo apt-get install -y -qq rsync libx11-6 libxext6 libxi6

# 2. uv
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

cd "$REPO_DIR"

# 4. Install Python deps (including simulation extras: taichi, cma, opencv)
uv sync --extra simulation

# 5. Smoke test — confirm Taichi sees the GPU
uv run python -c "
import taichi as ti
ti.init(arch=ti.cuda)
print('GPU OK:', ti.lang.impl.get_runtime().prog.config().arch)
"

echo ""
echo "=== Setup complete. Run ./run_experiments.sh to start. ==="
