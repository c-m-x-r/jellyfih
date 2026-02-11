# Jellyfih Project Instructions

## Project Overview

Evolutionary optimization of soft robotic jellyfish morphologies using CMA-ES and 2D MPM simulation in Taichi. The goal is to discover bell shapes optimized for carrying instrumented payloads, potentially diverging from natural biomimetic designs.

## Current Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Actuation | Pulsed active stress | Isotropic pressure on muscle (Mat 3) particles |
| Fitness | Displacement / Cost proxy | See Fitness section below |
| Resolution | 128x128 grid, 80K particles | quality=1, single phase for now |
| Payload | 0.08 x 0.05 normalized units | Material 2, 2.5x density |
| Boundaries | Damped sides, clamped top/bottom | n_grid/20 damping layer width |
| CMA-ES | lambda=16, sigma=0.1, 50 gens | Population matches GPU batch size |
| Sim Duration | 3 actuation cycles (60K steps) | Per fitness evaluation |
| Frequency | 1.0 Hz | Biological mid-range |
| Spawn | [0.5, 0.4] | Centered, 40% up from bottom |

## Architecture

Three main components, fully integrated:

1. **mpm_sim.py** - GPU-accelerated MPM simulation engine
   - 16 parallel instances on CUDA
   - Materials: Water(0), Jelly(1), Payload(2), Muscle(3)
   - Fixed tensor allocation (16 x 80,000 particles)
   - Pulsed active stress actuation (asymmetric ramp-up/decay waveform)
   - GPU-side fitness evaluation via `fitness_buffer` field
   - Headless batch runner (`run_batch_headless`)
   - HDR particle splatting renderer with per-instance color tinting

2. **make_jelly.py** - Genotype-phenotype mapping
   - 9-gene Bezier curve encoding for bell shape
   - Structural features: bell, muscle layer, mesoglea collar, transverse bridge
   - `fill_tank()`: water generation, KDTree boolean subtraction, dead-particle padding
   - Returns muscle particle count for CPU-side cost calculation

3. **evolve.py** - Evolutionary optimizer (CMA-ES)
   - `--gens N`: set generation count
   - `--view`: render best genomes as 4x4 grid video
   - `--view --gen N`: render specific generation
   - Zero-actuation baseline check at startup
   - Checkpoint/resume via pickle (every 5 gens)
   - Full CSV logging + JSON best-genome history

## Fitness Evaluation

**Current proxy metric** (not full Cost of Transport):
```
displacement = final_payload_CoM_y - initial_payload_CoM_y
stability = 1 / (1 + 10 * lateral_drift^2)
cost = (muscle_count / 500) + 0.1
fitness = displacement * stability / cost
```

GPU-side: `compute_payload_stats()` kernel averages all Material 2 particle positions via `ti.atomic_add`. Two separate kernels (clear + compute) to avoid Taichi race conditions.

CPU-side: muscle count from `fill_tank()` stats, stability and cost normalization.

**Deferred**: Full Cost of Transport requires per-step energy accumulation in the GPU kernel. Current proxy correlates (more muscle = more energy, less displacement = worse) but is not identical.

**Boundary-stuck detection**: Payload CoM at y > 0.99 or y < 0.01 flags instance as invalid.

## Material System

| ID | Material | Properties | Role |
|----|----------|------------|------|
| 0 | Water | Fluid, mu=0, lambda=4000 | Background medium |
| 1 | Jelly | Hyperelastic, E=0.1e4, nu=0.45 | Passive bell structure |
| 2 | Payload | Near-rigid, E=4e4, 2.5x density | Instrumented cargo |
| 3 | Muscle | Same elasticity as jelly + active stress | Actuation tissue |
| -1 | Dead | Skipped in all kernels | Padding to fixed count |

## Genome Encoding

9-dimensional vector controlling bell morphology:

| Index | Parameter | Bounds | Description |
|-------|-----------|--------|-------------|
| 0 | cp1_x | [0.0, 0.2] | Control Point 1 x-offset |
| 1 | cp1_y | [-0.1, 0.1] | Control Point 1 y-offset |
| 2 | cp2_x | [0.0, 0.25] | Control Point 2 x-offset |
| 3 | cp2_y | [-0.15, 0.1] | Control Point 2 y-offset |
| 4 | end_x | [0.05, 0.3] | Bell tip x-extent |
| 5 | end_y | [-0.4, -0.05] | Bell tip y-extent |
| 6 | t_base | [0.01, 0.08] | Thickness at payload |
| 7 | t_mid | [0.01, 0.1] | Thickness at bell middle |
| 8 | t_tip | [0.003, 0.03] | Thickness at bell tip |

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

# Test morphology generator (matplotlib plot)
uv run python make_jelly.py
```

## Files

| File | Purpose |
|------|---------|
| mpm_sim.py | MPM physics engine + renderer + fitness kernels |
| make_jelly.py | Morphology generator + tank filler |
| evolve.py | CMA-ES evolutionary loop + visualization |
| pyproject.toml | Dependencies (taichi, numpy, scipy, cma, imageio, etc.) |

## Output Files

All outputs in `output/` directory:

| File | Description |
|------|-------------|
| `evolution_log.csv` | Every individual: generation, genome[0-8], fitness, displacement, drift, muscle_count, valid, sigma |
| `best_genomes.json` | Best genome per generation (list of {generation, genome, fitness, displacement, sigma}) |
| `checkpoint.pkl` | CMA-ES state for crash recovery (every 5 gens) |
| `view_*.mp4` | Rendered videos from --view mode |

## Performance

Benchmarked on CUDA (quality=1, 128x128 grid, 80K particles, 16 instances):
- Substep throughput: ~1.2 ms/step (all 16 instances in parallel)
- Per-generation: ~72s simulation + ~2s CPU phenotype generation
- 50-generation run: ~62 minutes total
- CPU-GPU transfer: only 16x5 floats per generation (fitness results)

## Known Issues / TODO

1. **Fitness proxy**: Using displacement/cost, not full Cost of Transport (needs GPU energy tracking)
2. **Gravity asymmetry**: Payload has 0.2x gravity (intentional buoyancy model, but needs baseline validation)
3. **Actuation residual**: Decay waveform asymptotically approaches zero, never fully resets
4. **Mirror overlap**: Particles at x=0 duplicated by symmetry mirroring (density spike at midline)
5. **No Aurelia aurita baseline**: Control genome not yet defined
6. **No adaptive resolution**: Phase 1/2 transition (128->256 grid) not implemented
7. **No genome heatmap**: Post-evolution gene importance visualization not implemented
8. **No per-generation video**: Only manual --view mode, no automatic video per generation
