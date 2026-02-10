# Beyond Jelly: Evolutionary Soft Robotic Jellyfish

GPU-accelerated evolutionary optimization of soft robotic jellyfish morphologies using CMA-ES and 2D Material Point Method (MPM) simulation.

## Overview

This project explores whether strict biomimetic copying of natural jellyfish shapes is suboptimal for soft robots carrying instrumented payloads. By leveraging evolutionary computation (CMA-ES) within a GPU-accelerated Taichi simulation, we aim to discover novel bell morphologies specifically optimized for payload-carrying applications.

### Hypothesis

Evolutionary strategies applied within GPU-accelerated simulation will converge upon novel bell morphologies that:
- Differ significantly from biological baselines when subjected to centralized payloads
- Exhibit higher propulsive efficiency compared to standard biomimetic designs
- Demonstrate improved station-keeping stability

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CMA-ES Optimizer                        │
│                    (Black-box Evolution)                     │
└─────────────────────┬───────────────────────────────────────┘
                      │ Genome Vector (9 genes)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Genotype-Phenotype Mapping                  │
│                     (make_jelly.py)                          │
│  • Bezier curve encoding (6 control point params)           │
│  • Variable thickness profile (3 params)                     │
│  • Radial symmetry mirroring                                 │
└─────────────────────┬───────────────────────────────────────┘
                      │ Particle positions + materials
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tank Assembly Stage                       │
│  • Background water generation                               │
│  • Boolean subtraction (carve robot from fluid)              │
│  • Padding to fixed particle count                           │
└─────────────────────┬───────────────────────────────────────┘
                      │ Complete particle set
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  MPM Simulation Engine                       │
│                     (mpm_sim.py)                             │
│  • 16 parallel instances on GPU                              │
│  • 256×256 grid resolution                                   │
│  • Materials: Water(0), Jelly(1), Payload(2)                 │
│  • Universal sinusoidal actuation                            │
└─────────────────────┬───────────────────────────────────────┘
                      │ Fitness score (Cost of Transport)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Fitness Evaluation                        │
│  • Track payload displacement                                │
│  • Calculate Cost of Transport                               │
│  • Compare against biomimetic baseline                       │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
jellyfih/
├── mpm_sim.py          # GPU-accelerated MPM simulation (Taichi)
├── make_jelly.py       # Genotype-phenotype mapping
├── pyproject.toml      # Project dependencies
├── README.md           # This file
└── CLAUDE.md           # AI assistant instructions
```

## Configuration

Based on requirements analysis:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Actuation | Force-based | Radial forces on jelly particles |
| Fitness | Cost of Transport | Energy / (mass × distance) |
| Resolution | Adaptive | 16K/128×128 → 36K/256×256 |
| Payload | Fixed | Same for all morphologies |
| Boundaries | Hybrid | Damping on sides, outflow on bottom |
| Baseline | Aurelia aurita | Moon jelly profile as control |
| CMA-ES | λ=16, 100 generations | Matches GPU batch size |
| Sim Duration | 3-5 cycles | Per fitness evaluation |
| Frequency | 1.0 Hz | Biological mid-range |

## Current Status

### Implemented
- [x] MPM simulation engine with parallel instances
- [x] Bezier-curve based morphology generator
- [x] Basic material differentiation (water/jelly/payload)
- [x] Video export pipeline

### Integration Challenges (In Progress)
- [ ] Tank Filler: Background water generation with boolean subtraction
- [ ] Force-based actuation kernel for bell contraction
- [ ] Fixed-size particle padding for GPU compatibility
- [ ] Fitness evaluation loop for CMA-ES
- [ ] Hybrid boundary conditions (damping + outflow)
- [ ] Aurelia aurita baseline genome
- [ ] Full CSV logging (all genomes, fitness, sigma)
- [ ] Genome heatmap visualization

## Genome Encoding

9-dimensional vector controlling bell morphology:

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
| 2 | Payload | Near-rigid, high density |

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

## Usage

```bash
# Visualize random morphologies
uv run python make_jelly.py

# Run MPM simulation benchmark
uv run python mpm_sim.py
```

## Outputs

The system produces the following artifacts in `output/`:

| File | Description |
|------|-------------|
| `evolution_log.csv` | All genomes, fitness values, sigma per generation |
| `best_genomes.json` | Best genome trajectory for replay |
| `videos/gen_N_best.mp4` | Video of best individual per generation |
| `genome_heatmap.png` | Visualization of gene importance |

## Adaptive Resolution Strategy

| Phase | Generations | Particles | Grid | Purpose |
|-------|-------------|-----------|------|---------|
| Exploration | 0-70 | 16K | 128×128 | Fast broad search |
| Refinement | 71-100 | 36K | 256×256 | Fine-tuning with accurate vortices |

## Performance Metrics

Current benchmark on GPU (16 parallel instances):
- Low-res: 16K particles, 128×128 grid (~4x faster)
- High-res: 36K particles, 256×256 grid
- Timestep: 5×10⁻⁵ s
- Target: Real-time or faster for evolutionary feasibility

## References

1. Gemmell et al. "Passive energy recapture in jellyfish" PNAS 2013
2. Hansen, N. "The CMA Evolution Strategy: A Tutorial" 2016
3. Hu et al. "Taichi: High-performance computation" ACM TOG 2019

## License

MIT
