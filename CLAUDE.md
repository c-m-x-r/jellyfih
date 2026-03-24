# Jellyfih Project Instructions

## Project Overview

Evolutionary optimization of soft robotic jellyfish morphologies using CMA-ES and 2D MPM simulation in Taichi. The goal is to discover bell shapes optimized for carrying instrumented payloads, potentially diverging from natural biomimetic designs.

## Current Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Actuation | Raised cosine pulse | 20% contraction / 40% relaxation / 40% refractory, tangent-aligned (Mat 3), strength=500 |
| Fitness | Efficiency metric | displacement / sqrt(muscle_count / 500); muscle floor ≥200; dynamic invalid penalty |
| Resolution | 128x128 grid, 80K particles | quality=1, single phase for now |
| Payload | 0.08 x 0.05 normalized units | Material 2, density via `instance_payload_density` (default 2.5×), full gravity |
| Boundaries | Damped sides, clamped top/bottom | n_grid/20 damping layer width |
| CMA-ES | lambda=16, sigma=0.1, 50 gens | Population matches GPU batch size |
| Sim Duration | 3 actuation cycles (60K steps) | Per fitness evaluation |
| Frequency | 1.0 Hz | Biological mid-range |
| Spawn | [0.5, 0.55] | Vertically centred; deepest bell tip (end_y=-0.45) reaches y≈0.10, safely above floor |

## Architecture

Three main components, fully integrated:

1. **mpm_sim.py** - GPU-accelerated MPM simulation engine
   - 16 simulation instances batched on one GPU
   - Materials: Water(0), Jelly(1), Payload(2), Muscle(3)
   - Fixed tensor allocation (16 x 80,000 particles)
   - Raised cosine actuation waveform (20% contraction / 40% relaxation / 40% refractory), tangent-aligned stress
   - Per-instance actuation field (`instance_actuation`) enables parameter sweeps across batch
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

**Efficiency metric**: Rewards thrust relative to muscle mass:
```
displacement = final_y - init_y
fitness = displacement / sqrt(muscle_count / 500)
```

This penalises bloated muscle mass — a shape that moves twice as far with the same muscle scores identically to one that uses half the muscle for the same displacement. The 500 normalisation factor keeps the metric near unity for typical genomes.

**Validity checks** (instance marked invalid, assigned `worst_valid_fitness - 1`):
- Payload CoM at y > 0.93 (ceiling-riding exploit)
- Payload CoM at y < 0.01 (floor contact)
- muscle_count < 200 (degenerate/empty morphology)

Drift penalty (`- 1.0 * drift`) is implemented but currently commented out in `evolve.py`.

GPU-side: `compute_payload_stats()` kernel averages all Material 2 particle positions via `ti.atomic_add`. Two separate kernels (clear + compute) to avoid Taichi race conditions.

## Material System

| ID | Material | Properties | Role |
|----|----------|------------|------|
| 0 | Water | Fluid, mu=0, lambda=100000 | Background medium |
| 1 | Jelly | Hyperelastic, E=0.7e3, nu=0.3 | Passive bell structure |
| 2 | Payload | Near-rigid, E=2e5, density=`instance_payload_density` (default 2.5×), full gravity | Instrumented cargo; density is per-instance configurable (2.5 ≈ LiPo/PCB) |
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

# Render all individuals from a specific generation (each cell = different individual)
uv run python evolve.py --view --gen 3

# Evaluate Aurelia aurita reference genome (baseline comparison)
uv run python evolve.py --aurelia

# Test morphology generator (matplotlib plot)
uv run python make_jelly.py
uv run python make_jelly.py --aurelia             # Moon jelly reference
uv run python make_jelly.py --aurelia --no-payload  # Payloadless (symmetry test)

# Fluid dynamics test visualization
uv run python helpers/fluid_test.py

# Payload sink baseline (no jellyfish, shows payload sinking under gravity)
uv run python helpers/payload_sink.py

# CAD export: generate STL files from a genome
uv run python helpers/make_cad.py --aurelia           # Aurelia reference
uv run python helpers/make_cad.py --gen 5             # Best genome from generation 5
uv run python helpers/make_cad.py --diameter 120      # Physical scale in mm

# Side-by-side comparison video (Aurelia vs Gen 0 vs Gen N)
uv run python helpers/make_comparison.py

# Actuation strength sweep across GPU batch
uv run python tune_actuation.py

# Quick single-genome preview — 1 instance = full 1024x1024 view (~4-5s)
uv run python helpers/view_single.py --aurelia                   # Aurelia reference, full-res
uv run python helpers/view_single.py --gen 5                     # Best from gen 5
uv run python helpers/view_single.py --aurelia --flow            # With vorticity overlay
uv run python helpers/view_single.py --aurelia --palette web     # Web palette
uv run python helpers/view_single.py --aurelia --palette random  # Two contrasting colours
uv run python helpers/view_single.py --aurelia --no-payload      # Payloadless symmetry
uv run python helpers/view_single.py --gen 3 --instances 16     # 4x4 colour-variant grid

# View all individuals from a generation (each GPU instance = different individual)
# Handles any generation size; shows top N by fitness with clear count messaging
uv run python helpers/view_generation.py --gen 5
uv run python helpers/view_generation.py --gen 5 --include-invalid
uv run python helpers/view_generation.py --gen 5 --palette random  # per-cell random colours
uv run python helpers/view_generation.py --gen 5 --palette web

# Web viewer (morphology explorer + evolutionary history + interactive designer)
cd web && python app.py   # http://localhost:5000
# /        — genome sliders, evolution history, convergence plots, click-drag Bezier
# /custom  — interactive designer, colour picker, shared aquarium, fitness prediction
```

## Files

| File | Purpose |
|------|---------|
| mpm_sim.py | MPM physics engine + renderer + fitness kernels |
| make_jelly.py | Morphology generator + tank filler + Aurelia reference |
| evolve.py | CMA-ES evolutionary loop + visualization + Aurelia eval |
| run_population.py | Batch population runner with CV2 rendering |
| tune_actuation.py | Per-instance actuation strength sweep across GPU batch |
| web/ | Flask web viewer: genome sliders, evolution history, convergence plots, click-drag Bezier; `/custom` interactive designer with shared aquarium |
| pyproject.toml | Dependencies (taichi, numpy, scipy, cma, imageio, trimesh, flask, etc.) |
| **helpers/** | Utility scripts (not required for core evolution loop) |
| helpers/fluid_test.py | Fluid dynamics test visualization (oscillating paddle) |
| helpers/make_cad.py | CAD export: genome → STL (extruded cross-section + revolved solid) |
| helpers/make_comparison.py | Side-by-side comparison video: Aurelia vs Gen 0 vs Gen N |
| helpers/payload_sink.py | Baseline demo: payload sinking without jellyfish (buoyancy check) |
| helpers/view_single.py | Quick single-genome render (~4.5s); `--aurelia`, `--gen N`, `--flow`, `--no-payload` |
| helpers/view_generation.py | View all individuals from one generation; each GPU instance = different morphology |

## Output Files

All outputs in `output/` directory:

| File | Description |
|------|-------------|
| `evolution_log.csv` | Every individual: generation, genome[0-8], fitness, efficiency, displacement, drift, muscle_count, valid, sigma |
| `best_genomes.json` | Best genome per generation (list of {generation, genome, fitness, efficiency, displacement, sigma}) |
| `checkpoint.pkl` | CMA-ES state for crash recovery (every 5 gens) |
| `view_*.mp4` | Rendered videos from --view mode |
| `view_aurelia.mp4` | Aurelia aurita reference genome video |
| `fluid_test.mp4` | Fluid dynamics test visualization |

## Performance

Benchmarked on a single CUDA GPU (quality=1, 128x128 grid, 80K particles, 16 instances batched):
- Substep throughput: ~1.2 ms/step (all 16 instances in one batch)
- Per-generation (λ=16): ~72s simulation + ~2s CPU phenotype generation
- 50-generation run (λ=16): ~62 minutes total
- Per-generation (λ=48, cloud): ~244s
- CPU-GPU transfer: λ×5 floats per generation (fitness results)
- GPU is SM-compute bound (~100% SM, ~6% VRAM at λ=16); two processes time-slice rather than parallelise — scale via larger λ in a single process

## Known Issues / TODO

1. **No buoyancy model**: MPM has no hydrostatic buoyancy. Payload uses full gravity (2.5x density) and sinks; jellyfish must generate active thrust to lift it.
2. **Mirror overlap**: Particles at x=0 duplicated by symmetry mirroring (density spike at midline)
3. **No adaptive resolution**: Phase 1/2 transition (128->256 grid) not implemented
4. **No genome heatmap**: Post-evolution gene importance visualization not implemented
5. **No per-generation video**: Only manual --view mode, no automatic video per generation
6. **No GPU energy tracking**: Full Cost of Transport metric deferred
7. **Drift penalty disabled**: `- 1.0 * drift` term in `compute_fitness()` is commented out; lateral stability not currently penalized
8. **Water lambda calibration**: lambda=100000 stiffens the fluid significantly — empirically stable but not physically justified
