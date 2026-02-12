# Jellyfih Installation Guide

## Two Installation Modes

### 🌐 Web Viewer Only (Simple - Works on Termux/Android)

**Use pip directly:**
```bash
cd web/
./install.sh
python app.py
```

**Dependencies:**
- flask
- numpy
- scipy
- matplotlib
- pillow

**What you get:**
- Interactive genome viewer
- Morphology visualization
- Statistics panel
- All presets (Default, Aurelia, Random)

**What you DON'T need:**
- ❌ uv
- ❌ Taichi (GPU simulation)
- ❌ opencv-python
- ❌ CMA-ES

---

### 🚀 Full Simulation Suite (Advanced - Requires x86_64 + GPU)

**Use uv with optional dependencies:**
```bash
# On x86_64 Linux/Mac with GPU
uv sync --extra all
uv run python evolve.py
```

**Additional dependencies:**
- taichi (GPU MPM simulation)
- opencv-python (video rendering)
- imageio (video export)
- cma (evolutionary optimization)

**What you get:**
- Everything from web viewer
- GPU-accelerated MPM simulation
- Evolutionary optimization (CMA-ES)
- Video rendering
- Batch population runs

---

## Platform Support

### ✅ Web Viewer (pip)
- ✅ Termux (Android ARM64)
- ✅ Linux (x86_64, ARM64)
- ✅ macOS (Intel, Apple Silicon)
- ✅ Windows (WSL, native)

### ⚠️ Full Suite (uv)
- ✅ Linux x86_64 (CUDA or CPU)
- ✅ macOS x86_64/arm64
- ❌ Termux (Taichi requires x86_64, opencv not available)
- ⚠️ Windows (WSL recommended)

---

## Troubleshooting

### uv can't find Python
```bash
# Create venv with system Python
uv venv --python $(which python)

# Or use pip instead
pip install flask numpy scipy matplotlib pillow
```

### opencv-python won't install
```bash
# Web viewer doesn't need it!
# Only required for full simulation suite

# If needed for simulation:
pkg install opencv  # Termux
apt install python3-opencv  # Debian/Ubuntu
brew install opencv  # macOS
```

### Pillow build fails
```bash
# Install image libraries
pkg install libjpeg-turbo libpng  # Termux
apt install libjpeg-dev libpng-dev  # Debian/Ubuntu
brew install jpeg libpng  # macOS
```

### Import errors in web viewer
```bash
# Check what's installed
python web/check_deps.py

# Reinstall missing packages
cd web/
./install.sh
```

---

## Quick Reference

| Component | Install Method | Platform |
|-----------|---------------|----------|
| Web Viewer | `pip install flask numpy scipy matplotlib pillow` | All |
| Full Suite | `uv sync --extra all` | x86_64 only |
| Morphology Generator | `pip install numpy scipy` | All |
| GPU Simulation | `uv sync --extra simulation` | x86_64 + CUDA |

---

## Development Setup

### For Web Viewer Development
```bash
# Clone repo
git clone <repo> jellyfih
cd jellyfih

# Install web viewer deps
cd web/
./install.sh

# Run tests
python test_api.py

# Start dev server
python app.py
```

### For Full Simulation Development
```bash
# Clone repo
git clone <repo> jellyfih
cd jellyfih

# Install with uv
uv sync --extra all

# Run evolution
uv run python evolve.py --gens 5

# View results
uv run python evolve.py --view
```

---

## Summary

**Just want to explore morphologies?**
→ Use `pip` + web viewer (works everywhere)

**Want to run evolutionary optimization?**
→ Use `uv` + full suite (x86_64 only)

**Working on Termux/Android?**
→ Use `pip` + web viewer (perfect for on-the-go!)

**Flying at 30,000 feet with Starlink?**
→ Use `pip` + web viewer (tested at altitude ✈️)
