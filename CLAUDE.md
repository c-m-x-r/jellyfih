# Jellyfih Project Instructions

## Project Overview

Evolutionary optimization of soft robotic jellyfish morphologies using CMA-ES and 2D MPM simulation in Taichi. The goal is to discover bell shapes optimized for carrying instrumented payloads, potentially diverging from natural biomimetic designs.

## Current Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Actuation | Raised cosine pulse | 20/80 asymmetry, isotropic pressure on muscle (Mat 3) |
| Fitness | Staying power (altitude) | final_y * stability, no cost ratio |
| Resolution | 128x128 grid, 80K particles | quality=1, single phase for now |
| Payload | 0.08 x 0.05 normalized units | Material 2, 2.5x density, 0.44x gravity (slightly neg. buoyant) |
| Boundaries | Damped sides, clamped top/bottom | n_grid/20 damping layer width |
| CMA-ES | lambda=16, sigma=0.1, 50 gens | Population matches GPU batch size |
| Sim Duration | 3 actuation cycles (60K steps) | Per fitness evaluation |
| Frequency | 1.0 Hz | Biological mid-range |
| Spawn | [0.5, 0.7] | Centered, 70% up from bottom |

## Architecture

Three main components, fully integrated:

1. **mpm_sim.py** - GPU-accelerated MPM simulation engine
   - 16 parallel instances on CUDA
   - Materials: Water(0), Jelly(1), Payload(2), Muscle(3)
   - Fixed tensor allocation (16 x 80,000 particles)
   - Raised cosine actuation waveform (20% contraction, 80% relaxation)
   - GPU-side fitness evaluation via `fitness_buffer` field
   - Headless batch runner (`run_batch_headless`)
   - HDR particle splatting renderer with per-instance color tinting

2. **make_jelly.py** - Genotype-phenotype mapping
   - 9-gene Bezier curve encoding for bell shape
   - Structural features: bell, muscle layer, mesoglea collar, transverse bridge
   - `fill_tank()`: water generation, KDTree boolean subtraction, dead-particle padding
   - Returns muscle particle count for CPU-side stats
   - Includes `AURELIA_GENOME`: hand-designed moon jelly reference

3. **evolve.py** - Evolutionary optimizer (CMA-ES)
   - `--gens N`: set generation count
   - `--view`: render best genomes as 4x4 grid video
   - `--view --gen N`: render specific generation
   - `--aurelia`: evaluate moon jelly reference genome + render video
   - Zero-actuation baseline check at startup
   - Checkpoint/resume via pickle (every 5 gens)
   - Full CSV logging + JSON best-genome history

## Fitness Evaluation

**Staying power metric**: Payload is slightly negatively buoyant (sinks without active swimming). Fitness rewards altitude maintenance:
```
drift = |final_x - init_x|
stability = 1 / (1 + 50 * drift^2)
fitness = final_y * stability
```

No cost ratio — the physics provides implicit cost. More jelly mass = more gravitational drag. Muscle mass is needed for thrust but has neutral buoyancy. The fitness landscape naturally rewards morphologies that balance thrust, drag, and structural efficiency.

GPU-side: `compute_payload_stats()` kernel averages all Material 2 particle positions via `ti.atomic_add`. Two separate kernels (clear + compute) to avoid Taichi race conditions.

**Boundary-stuck detection**: Payload CoM at y > 0.93 or y < 0.01 flags instance as invalid. The 0.93 threshold prevents ceiling-riding exploits.

## Material System

| ID | Material | Properties | Role |
|----|----------|------------|------|
| 0 | Water | Fluid, mu=0, lambda=4000 | Background medium |
| 1 | Jelly | Hyperelastic, E=0.7e3, nu=0.3 | Passive bell structure |
| 2 | Payload | Near-rigid, E=4e4, 2.5x density, 0.44x gravity | Instrumented cargo (slightly neg. buoyant) |
| 3 | Muscle | Same elasticity as jelly + active stress | Actuation tissue |
| -1 | Dead | Skipped in all kernels | Padding to fixed count |

## Genome Encoding

9-dimensional vector controlling bell morphology:

| Index | Parameter | Bounds | Description |
|-------|-----------|--------|-------------|
| 0 | cp1_x | [0.0, 0.25] | Control Point 1 x-offset |
| 1 | cp1_y | [-0.15, 0.15] | Control Point 1 y-offset |
| 2 | cp2_x | [0.0, 0.3] | Control Point 2 x-offset |
| 3 | cp2_y | [-0.2, 0.15] | Control Point 2 y-offset |
| 4 | end_x | [0.05, 0.35] | Bell tip x-extent |
| 5 | end_y | [-0.45, -0.03] | Bell tip y-extent |
| 6 | t_base | [0.025, 0.08] | Thickness at payload |
| 7 | t_mid | [0.025, 0.1] | Thickness at bell middle |
| 8 | t_tip | [0.01, 0.04] | Thickness at bell tip |

CMA-ES uses built-in bounds handling (not hard clipping) to preserve covariance estimation.

## Code Conventions

- Taichi kernels for all GPU operations
- NumPy for CPU-side assembly
- `ti.field` for GPU state, `ti.atomic_add` for parallel reductions
- Separate clear/compute kernels to avoid race conditions
- `load_particles()` resets per-particle state; no separate reset kernel needed
- Grid cleared every substep in `substep()` kernel

## Usage

```bash
# Evolve for 5 generations (quick test, ~6 min)
uv run python evolve.py --gens 5

# Full 50-generation run (~60 min)
uv run python evolve.py

# Resume from checkpoint (automatic if output/checkpoint.pkl exists)
uv run python evolve.py --gens 50

# Render best genomes as 4x4 video (rows=generations, columns=color-coded)
uv run python evolve.py --view

# Render specific generation
uv run python evolve.py --view --gen 3

# Evaluate Aurelia aurita reference genome (baseline comparison)
uv run python evolve.py --aurelia

# Test morphology generator (matplotlib plot)
uv run python make_jelly.py
uv run python make_jelly.py --aurelia  # Moon jelly reference

# Fluid dynamics test visualization
uv run python fluid_test.py
```

## Files

| File | Purpose |
|------|---------|
| mpm_sim.py | MPM physics engine + renderer + fitness kernels |
| make_jelly.py | Morphology generator + tank filler + Aurelia reference |
| evolve.py | CMA-ES evolutionary loop + visualization + Aurelia eval |
| fluid_test.py | Fluid dynamics test visualization (oscillating paddle) |
| pyproject.toml | Dependencies (taichi, numpy, scipy, cma, imageio, etc.) |

## Output Files

All outputs in `output/` directory:

| File | Description |
|------|-------------|
| `evolution_log.csv` | Every individual: generation, genome[0-8], fitness, final_y, displacement, drift, muscle_count, valid, sigma |
| `best_genomes.json` | Best genome per generation (list of {generation, genome, fitness, displacement, sigma}) |
| `checkpoint.pkl` | CMA-ES state for crash recovery (every 5 gens) |
| `view_*.mp4` | Rendered videos from --view mode |
| `view_aurelia.mp4` | Aurelia aurita reference genome video |
| `fluid_test.mp4` | Fluid dynamics test visualization |

## Performance

Benchmarked on CUDA (quality=1, 128x128 grid, 80K particles, 16 instances):
- Substep throughput: ~1.2 ms/step (all 16 instances in parallel)
- Per-generation: ~72s simulation + ~2s CPU phenotype generation
- 50-generation run: ~62 minutes total
- CPU-GPU transfer: only 16x5 floats per generation (fitness results)

## Known Issues / TODO

1. **Gravity hack**: Payload uses 0.44x gravity to approximate slight negative buoyancy (needs calibration justification for paper)
2. **Mirror overlap**: Particles at x=0 duplicated by symmetry mirroring (density spike at midline)
3. **No adaptive resolution**: Phase 1/2 transition (128->256 grid) not implemented
4. **No genome heatmap**: Post-evolution gene importance visualization not implemented
5. **No per-generation video**: Only manual --view mode, no automatic video per generation
6. **No GPU energy tracking**: Full Cost of Transport metric deferred
