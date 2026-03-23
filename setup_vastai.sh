#!/usr/bin/env bash
# setup_vastai.sh — provision a Vast.ai GPU instance for jellys experiments
# Run once after SSH-ing in: bash setup_vastai.sh
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
error() { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ── 1. CUDA check ────────────────────────────────────────────────────────────
info "Checking CUDA..."
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found — is this a GPU instance?"
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' || true)
info "CUDA version: ${CUDA_VERSION:-unknown}"

# ── 2. System packages ────────────────────────────────────────────────────────
info "Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    git curl wget unzip \
    ffmpeg \
    libgl1 libglib2.0-0 \
    libgomp1 \
    python3-dev build-essential \
    screen tmux \
    2>/dev/null
info "System packages installed."

# ── 3. uv ────────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv available in current shell
    export PATH="$HOME/.local/bin:$PATH"
    export PATH="$HOME/.cargo/bin:$PATH"
else
    info "uv already installed: $(uv --version)"
fi

# Ensure uv is on PATH for the rest of the script
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv &>/dev/null || error "uv install failed — PATH: $PATH"

# ── 4. Locate project ─────────────────────────────────────────────────────────
# Default: assume script lives in the project root (uploaded via scp/rsync)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"

if [[ ! -f "$PROJECT_DIR/pyproject.toml" ]]; then
    error "pyproject.toml not found in $PROJECT_DIR. Upload the project first:\n  rsync -avz jellys/ root@<instance-ip>:~/jellys/"
fi
info "Project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# ── 5. Python environment ─────────────────────────────────────────────────────
info "Installing Python dependencies (this may take a few minutes)..."
uv sync --extra all

# ── 6. Fix libcuda.so symlink (Taichi looks for unversioned name) ─────────────
# Vast.ai images expose libcuda.so.1 but not the bare libcuda.so Taichi needs
LIBCUDA_PATH=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF}' | head -1)
if [[ -n "$LIBCUDA_PATH" ]]; then
    LIBCUDA_DIR=$(dirname "$LIBCUDA_PATH")
    if [[ ! -f "$LIBCUDA_DIR/libcuda.so" ]]; then
        ln -sf "$LIBCUDA_PATH" "$LIBCUDA_DIR/libcuda.so"
        ldconfig
        info "Created libcuda.so symlink → $LIBCUDA_PATH"
    else
        info "libcuda.so already present."
    fi
else
    warn "libcuda.so.1 not found via ldconfig — Taichi may fall back to CPU."
fi

# ── 7. Verify Taichi + CUDA ───────────────────────────────────────────────────
info "Verifying Taichi CUDA backend..."
# Must write to a real file — Taichi's kernel introspection can't parse stdin/heredoc
cat > /tmp/taichi_check.py <<'PYEOF'
import taichi as ti
ti.init(arch=ti.cuda)

@ti.kernel
def test():
    x = ti.cast(1.0, ti.f32)
    print("Taichi CUDA OK — value:", x + 1.0)

test()
print("Taichi version:", ti.__version__)
PYEOF
uv run python /tmp/taichi_check.py
[[ $? -eq 0 ]] && info "Taichi CUDA verified." || error "Taichi CUDA failed — check libcuda.so symlink."

# ── 8. Output directory ───────────────────────────────────────────────────────
mkdir -p "$PROJECT_DIR/output"
info "Output directory ready: $PROJECT_DIR/output"

# ── 8. Summary ────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Environment ready. Quick-start commands:${NC}"
echo -e "${GREEN}════════════════════════════════════════════${NC}"
echo ""
echo "  # Quick smoke test (5 generations, ~6 min)"
echo "  uv run python evolve.py --gens 5"
echo ""
echo "  # Full run (50 generations, ~60 min)"
echo "  uv run python evolve.py --gens 50"
echo ""
echo "  # Resume after interruption (auto-detects checkpoint)"
echo "  uv run python evolve.py --gens 50"
echo ""
echo "  # Run in background with tmux"
echo "  tmux new -s jellys 'uv run python evolve.py --gens 50 | tee output/run.log'"
echo ""
echo "  # Aurelia reference baseline"
echo "  uv run python evolve.py --aurelia"
echo ""
