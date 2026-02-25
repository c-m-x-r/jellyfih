# Beyond Jelly: Evolutionary Soft Robotic Jellyfish

<img width="1432" height="1235" alt="Screenshot 2026-02-24 220055" src="https://github.com/user-attachments/assets/81f5cb3b-f274-4c53-977a-f27e51b2a7ad" />

GPU-accelerated evolutionary optimization of soft robotic jellyfish morphologies using CMA-ES and 2D Material Point Method (MPM) simulation.

## Overview

This project explores whether strict biomimetic copying of natural jellyfish shapes is suboptimal for soft robots carrying instrumented payloads. By leveraging evolutionary computation (CMA-ES) within a GPU-accelerated Taichi simulation, we aim to discover novel bell morphologies specifically optimized for payload-carrying applications.

<img width="1139" height="1281" alt="Screenshot 2026-02-24 220202" src="https://github.com/user-attachments/assets/24f17ad6-8eb4-4262-beb1-c4bec5769942" />

### Hypothesis
Evolutionary strategies applied within GPU-accelerated simulation will converge upon novel bell 
morphologies that:
- Differ significantly from biological baselines when subjected to centralized payloads
- Exhibit higher propulsive efficiency compared to standard biomimetic designs
- Demonstrate improved station-keeping stability

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CMA-ES Optimizer                       │
│               (evolve.py, popsize=16, 9 genes)              │
└─────────────────────┬───────────────────────────────────────┘
                      │ Genome Vector (9 floats)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Genotype-Phenotype Mapping                  │
│                     (make_jelly.py)                          │
│  • Bezier curve bell shape (6 params)                       │
│  • Variable thickness profile (3 params)                    │
│  • Muscle layer, mesoglea collar, transverse bridge         │
│  • Radial symmetry mirroring                                │
└─────────────────────┬───────────────────────────────────────┘
                      │ Particle positions + materials
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tank Assembly Stage                       │
│                  (fill_tank in make_jelly.py)                │
│  • Background water generation (lattice grid)               │
│  • KDTree boolean subtraction (carve robot from fluid)      │
│  • Padding to fixed 80K particle count                      │
└─────────────────────┬───────────────────────────────────────┘
                      │ Fixed-size particle arrays (CPU → GPU)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  MPM Simulation Engine                       │
│                     (mpm_sim.py)                             │
│  • 16 parallel instances on CUDA                            │
│  • 128×128 grid, 80K particles per instance                 │
│  • Materials: Water(0), Jelly(1), Payload(2), Muscle(3)     │
│  • Pulsed active stress actuation on muscle tissue          │
│  • GPU-side payload CoM tracking (fitness_buffer)           │
└─────────────────────┬───────────────────────────────────────┘
                      │ Payload displacement + stability (16 floats)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Fitness Evaluation                        │
│  • Vertical payload CoM displacement                        │
│  • Smooth quadratic lateral stability penalty               │
│  • Metabolic cost normalization (muscle particle count)     │
│  • Boundary-stuck and payload-loss detection                │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
jellyfih/
├── mpm_sim.py          # GPU MPM engine + renderer + fitness kernels
├── make_jelly.py       # Genotype-phenotype mapping + tank filler
├── evolve.py           # CMA-ES evolutionary loop + visualization
├── pyproject.toml      # Project dependencies
├── CLAUDE.md           # AI assistant project instructions
├── README.md           # This file
└── output/             # Evolution results (generated)
    ├── evolution_log.csv
    ├── best_genomes.json
    ├── checkpoint.pkl
    └── view_*.mp4
```

## Installation

```bash
# Clone repository
git clone <repo-url>
cd jellyfih

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Requirements
- Python 3.10+
- CUDA-capable GPU
- Taichi 1.7.0+
- CMA-ES 4.0+

## Usage

```bash
# Quick 5-generation test run (~6 min)
uv run python evolve.py --gens 5

# Full 50-generation evolution (~60 min)
uv run python evolve.py

# Resume from checkpoint (automatic)
uv run python evolve.py --gens 50

# Render best genomes as 4x4 grid video
# Rows = generations, columns = lime|green|turquoise|cyan
uv run python evolve.py --view

# Render a specific generation
uv run python evolve.py --view --gen 3

# Test morphology generator standalone
uv run python make_jelly.py
```

## Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Actuation | Pulsed active stress | Isotropic pressure on muscle particles |
| Fitness | Displacement / cost | Proxy for Cost of Transport |
| Resolution | 128x128 grid | 80K particles, quality=1 |
| Payload | 0.08 x 0.05 | Material 2, 2.5x density |
| Boundaries | Damped sides, clamped walls | Damping layer = grid/20 |
| CMA-ES | lambda=16, sigma=0.1 | Population matches GPU batch size |
| Sim Duration | 3 cycles (60K steps) | dt=5e-5, freq=1Hz |
| Spawn | [0.5, 0.4] | Centered, 40% up |

## Genome Encoding

9-dimensional vector controlling bell morphology via cubic Bezier curves:

| Index | Parameter | Description |
|-------|-----------|-------------|
| 0-1 | cp1_x, cp1_y | Control Point 1 (curve shaping) |
| 2-3 | cp2_x, cp2_y | Control Point 2 (curve shaping) |
| 4-5 | end_x, end_y | Tip position (bell extent) |
| 6 | t_base | Thickness at payload connection |
| 7 | t_mid | Thickness at bell middle |
| 8 | t_tip | Thickness at bell tip |

## Materials

| ID | Material | Properties |
|----|----------|------------|
| 0 | Water | Fluid, zero shear modulus |
| 1 | Jelly | Hyperelastic soft body (mesoglea) |
| 2 | Payload | Near-rigid, high density (2.5x) |
| 3 | Muscle | Soft body + pulsed active stress |
| -1 | Dead | Padding particles (skipped in kernels) |

## Outputs

| File | Description |
|------|-------------|
| `evolution_log.csv` | All genomes with fitness, displacement, drift, muscle count, validity per generation |
| `best_genomes.json` | Best genome per generation for replay |
| `checkpoint.pkl` | CMA-ES state for crash recovery |
| `view_*.mp4` | Rendered 4x4 grid videos (column-color-coded) |

## Performance

Benchmarked on CUDA (16 parallel instances, 80K particles each):

| Metric | Value |
|--------|-------|
| Substep throughput | ~1.2 ms/step |
| Per-generation time | ~74s (72s sim + 2s CPU) |
| 50-generation run | ~62 minutes |
| GPU-CPU transfer | 16x5 floats/generation |

## Current Status

### Implemented
- [x] MPM simulation engine with 16 parallel GPU instances
- [x] Bezier-curve morphology generator with muscle layer
- [x] Tank filler with KDTree boolean subtraction
- [x] Pulsed active stress actuation (asymmetric waveform)
- [x] GPU-side payload CoM fitness evaluation
- [x] CMA-ES evolutionary loop with bounds handling
- [x] Checkpoint/resume for crash recovery
- [x] Full CSV + JSON logging
- [x] HDR particle splatting renderer
- [x] 4x4 grid visualization with per-column color coding
- [x] Zero-actuation baseline validation
- [x] Boundary-stuck payload detection

### TODO
- [ ] Full Cost of Transport fitness (GPU energy tracking)
- [ ] Aurelia aurita baseline genome
- [ ] Adaptive resolution (128 -> 256 grid transition)
- [ ] Genome heatmap visualization
- [ ] Automatic per-generation video export

## References

1. Gemmell et al. "Passive energy recapture in jellyfish" PNAS 2013
2. Hansen, N. "The CMA Evolution Strategy: A Tutorial" 2016
3. Hu et al. "Taichi: High-performance computation" ACM TOG 2019

## License

MIT
