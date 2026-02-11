# Beyond Jelly: Evolutionary Soft Robotic Jellyfish

GPU-accelerated evolutionary optimization of soft robotic jellyfish morphologies using CMA-ES and 2D Material Point Method (MPM) simulation in Taichi.

## Overview

This project explores whether strict biomimetic copying of natural jellyfish shapes is suboptimal for soft robots carrying instrumented payloads (batteries, sensors, comms modules). By leveraging evolutionary computation (CMA-ES) within a GPU-accelerated Taichi simulation, we aim to discover novel bell morphologies specifically optimized for payload-carrying applications that may diverge significantly from natural forms.

### Hypothesis

An evolutionary strategy applied within a GPU-accelerated simulation environment will converge upon novel bell morphologies that differ significantly from biological baselines when subjected to heavy, centralized payloads. These evolved morphologies will exhibit lower Cost of Transport compared to standard biomimetic designs (Aurelia aurita) when carrying identical instrumented payloads.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CMA-ES Optimizer                       │
│              (lambda=16, sigma=0.1, 100 gens)               │
└─────────────────────┬───────────────────────────────────────┘
                      │ Genome Vector (9 genes)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Genotype-Phenotype Mapping                  │
│                     (make_jelly.py)                          │
│  - Cubic Bezier curve encoding (6 control point params)     │
│  - Variable thickness profile (3 params: base/mid/tip)      │
│  - Bilateral symmetry via mirroring                         │
│  - Payload block (0.15 x 0.1 normalized units)              │
└─────────────────────┬───────────────────────────────────────┘
                      │ Robot particle positions + materials
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tank Assembly Stage                       │
│                  (fill_tank in make_jelly.py)                │
│  - Lattice-based water generation (dx/2 spacing)            │
│  - KDTree boolean subtraction (carve robot from fluid)      │
│  - Padding to fixed n_particles with dead particles [-1,-1] │
└─────────────────────┬───────────────────────────────────────┘
                      │ Fixed-size particle arrays
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  MPM Simulation Engine                       │
│                     (mpm_sim.py)                             │
│  - 16 parallel instances on GPU (CUDA)                      │
│  - 128x128 grid (quality=1), 70K particles per instance     │
│  - MLS-MPM with Neo-Hookean + fluid constitutive models     │
│  - Materials: Water(0), Jelly(1), Payload(2)                │
│  - GPU-side rendering to frame buffer                       │
└─────────────────────┬───────────────────────────────────────┘
                      │ Simulation video / fitness metrics
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               Population Runner (run_population.py)         │
│  - Generates N random genomes (one per instance)            │
│  - Loads all instances to GPU in batch                      │
│  - Runs parallel simulation + streaming video render        │
│  - Morphology preview grid (matplotlib)                     │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
jellyfih/
├── mpm_sim.py          # MPM physics engine (Taichi kernels, P2G/G2P, rendering)
├── make_jelly.py       # Genotype-phenotype mapping + tank assembly
├── run_population.py   # Batch population runner (parallel morphology evaluation)
├── proposal.txt        # SYDE 537 project proposal (academic context)
├── pyproject.toml      # Dependencies (taichi, numpy, scipy, opencv, matplotlib)
├── CLAUDE.md           # Development instructions
└── README.md           # This file
```

## Current Status

### Implemented (Phase 1-2)
- [x] MLS-MPM simulation engine with 16 parallel GPU instances
- [x] Cubic Bezier genotype-phenotype mapping (9-gene encoding)
- [x] Three-material system: Water (fluid), Jelly (hyperelastic), Payload (snow-like plasticity)
- [x] Tank filler with lattice-based water generation + KDTree boolean subtraction
- [x] Fixed-size particle padding for GPU tensor compatibility
- [x] GPU-side particle rendering to frame buffer (no history buffer needed)
- [x] Streaming video export pipeline (OpenCV mp4v)
- [x] Population runner with per-instance random genomes
- [x] Morphology preview grid visualization
- [x] Side-wall velocity damping (lateral wave absorption)
- [x] Hydrostatic warmup phase (200 substeps before recording)
- [x] Boundary clamping (all four walls)

### Not Yet Implemented
- [ ] Force-based actuation kernel (sinusoidal radial contraction/expansion)
- [ ] CMA-ES optimization loop integration
- [ ] Fitness evaluation (Cost of Transport calculation)
- [ ] Payload centroid tracking
- [ ] Energy accounting (kinetic + elastic strain)
- [ ] Aurelia aurita baseline genome (biomimetic control)
- [ ] Hybrid boundary conditions (bottom outflow instead of hard wall)
- [ ] Adaptive resolution switching (128 -> 256 grid at generation 70)
- [ ] Evolution logging (CSV with all genomes, fitness, sigma)
- [ ] Best genome JSON export
- [ ] Per-generation video of best individual
- [ ] Genome heatmap visualization

### Known Issues
- `load_particles` has a temporary mass override (`p_mass = p_vol * p_rho * 0.5` for robot particles) that shadows the module-level constant -- this is a debugging artifact
- Payload uses snow-like plasticity model (material 2) rather than true rigid body
- No actuation means jellyfish bodies are passive -- they sink and deform under gravity but don't swim
- Water fills entire domain (no free surface) -- jellyfish are submerged in a closed box
- `.gitignore` excludes `*.txt` which means `proposal.txt` is not tracked

## Simulation Parameters

| Parameter | Current Value | Notes |
|-----------|---------------|-------|
| n_instances | 16 | Parallel GPU simulations |
| n_particles | 70,000 | Per instance |
| n_grid | 128 | Grid resolution (quality=1) |
| dt | 5e-5 s | Timestep |
| E (Young's modulus) | 1000 Pa | Jelly stiffness |
| nu (Poisson's ratio) | 0.2 | |
| water_lambda | 4000 | Bulk modulus for incompressibility |
| gravity | 10.0 m/s^2 | Downward |
| warmup_steps | 200 | Hydrostatic equilibration |
| frames | 400 | Recording length |
| substeps_per_frame | 50 | Physics steps per rendered frame |

## Genome Encoding

9-dimensional vector controlling bell morphology via cubic Bezier curve:

| Index | Parameter | Range | Description |
|-------|-----------|-------|-------------|
| 0 | cp1_x | [0, 0.15] | Control Point 1 horizontal offset |
| 1 | cp1_y | [-0.05, 0.05] | Control Point 1 vertical offset |
| 2 | cp2_x | [0.05, 0.2] | Control Point 2 horizontal offset |
| 3 | cp2_y | [-0.1, 0.05] | Control Point 2 vertical offset |
| 4 | end_x | [0.1, 0.25] | Bell tip horizontal extent |
| 5 | end_y | [-0.3, -0.1] | Bell tip vertical extent |
| 6 | t_base | [0.02, 0.06] | Thickness at payload connection |
| 7 | t_mid | [0.02, 0.08] | Thickness at bell middle |
| 8 | t_tip | [0.005, 0.02] | Thickness at bell tip |

## Materials

| ID | Material | Constitutive Model | Key Properties |
|----|----------|-------------------|----------------|
| 0 | Water | Fluid (mu=0, high lambda) | Zero shear, bulk modulus 4000 |
| 1 | Jelly | Neo-Hookean hyperelastic | E=1000 Pa, h=0.3 (soft) |
| 2 | Payload | Snow-like plasticity | Clamped singular values |
| -1 | Dead | Skipped | Padding particles at [-1,-1] |

## Installation

```bash
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
# Run single-genome simulation (all 16 instances identical)
uv run python mpm_sim.py

# Run population of random morphologies (16 unique genomes)
uv run python run_population.py
```

## Planned Outputs

When the CMA-ES loop is implemented, the system will produce:

| File | Description |
|------|-------------|
| `output/evolution_log.csv` | All genomes, fitness, sigma per generation |
| `output/best_genomes.json` | Best genome per generation for replay |
| `output/videos/gen_N_best.mp4` | Video of best individual per generation |
| `output/genome_heatmap.png` | Gene importance across evolution |

## Adaptive Resolution Strategy (Planned)

| Phase | Generations | Particles | Grid | Purpose |
|-------|-------------|-----------|------|---------|
| Exploration | 0-70 | 16K | 128x128 | Fast broad search |
| Refinement | 71-100 | 36K | 256x256 | Fine-tuning with accurate vortices |

Transition: Re-evaluate top 20% of population at high-res before continuing.

## References

1. Gemmell et al. "Passive energy recapture in jellyfish contributes to propulsive advantage over other metazoans" PNAS 2013
2. Almubarak et al. "KryptoJelly: A jellyfish robot with SMA actuators" SPIE 2024
3. Li et al. "Development of a bionic jellyfish robot for collecting polymetallic nodules" Ocean Engineering 2023
4. Xu et al. "Field testing of biohybrid robotic jellyfish to demonstrate enhanced swimming speeds" Biomimetics 2020
5. Strgar & Krieger "Accelerated co-design of robots through morphological pretraining" NeurIPS 2024
6. Hansen, N. "The CMA Evolution Strategy: A Tutorial" 2016
7. Hu et al. "Taichi: High-performance computation on spatially sparse data structures" ACM TOG 2019

## License

MIT
