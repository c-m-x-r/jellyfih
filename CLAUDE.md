# Jellyfih Project Instructions

## Project Overview

Evolutionary optimization of soft robotic jellyfish morphologies using CMA-ES and 2D MPM simulation in Taichi. The goal is to discover bell shapes optimized for carrying instrumented payloads, potentially diverging from natural biomimetic designs.

## Finalized Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Actuation | Force-based | Radial forces on jelly particles |
| Fitness | Cost of Transport | Energy / (mass × distance) |
| Resolution | Adaptive | Low: 16K/128×128, High: 36K/256×256 |
| Payload | Fixed | 0.15 × 0.1 normalized units |
| Boundaries | Hybrid | Damping on sides, outflow on bottom |
| Baseline | Aurelia aurita | Moon jelly profile as control genome |
| CMA-ES | λ=16, σ=0.1, 100 gens | Population matches GPU batch size |
| Sim Duration | 3-5 actuation cycles | Per fitness evaluation |
| Frequency | 1.0 Hz | Biological mid-range |

## Architecture

Two main components requiring integration:

1. **mpm_sim.py** - GPU-accelerated MPM simulation
   - 16 parallel instances for batch evaluation
   - Materials: Water(0), Jelly(1), Payload(2)
   - Fixed tensor allocation (n_instances × n_particles)

2. **make_jelly.py** - Genotype-phenotype mapping
   - 9-gene Bezier curve encoding
   - Variable particle count output (incompatible with GPU tensors)
   - Outputs only robot particles (no background water)

## Critical Integration Tasks

### Tank Filler Algorithm
The morphology generator outputs robot particles only. Before simulation:
1. Generate uniform water particle grid
2. Use scipy.spatial.KDTree to carve out robot shape (boolean subtraction)
3. Pad to fixed n_particles count with "dead" particles at [-1, -1]

### Force-Based Actuation
Add actuation kernel (apply BEFORE substep):
```python
@ti.kernel
def apply_actuation(t: float, frequency: float, amplitude: float):
    phase = t * frequency * 2 * 3.14159
    contraction = ti.sin(phase) * amplitude
    for m, p in x:
        if material[m, p] == 1:  # Jelly
            # Calculate direction to centroid
            to_center = centroid[m] - x[m, p]
            # Apply radial force (positive = contract, negative = expand)
            v[m, p] += contraction * to_center.normalized() * dt
```

### Hybrid Boundary Conditions
Modify grid operations in substep():
- Sides (x<damping_width, x>1-damping_width): exponential velocity damping
- Bottom (y<3): allow outflow (remove clamping)
- Top: keep clamped to prevent escape

### Fitness Evaluation
Track Material 2 (payload) centroid displacement:
- `Y_displacement = Y_final - Y_initial`
- `Cost_of_Transport = Energy / (Mass × Y_displacement)`
- Energy = sum of kinetic + elastic strain energy over simulation

## Code Conventions

- Use Taichi kernels for GPU operations
- NumPy for CPU-side assembly
- Prefer `ti.field` over Python lists in hot paths
- Material IDs: 0=Water, 1=Jelly, 2=Payload

## Testing

```bash
# Verify morphology generator
uv run python make_jelly.py

# Benchmark simulation performance
uv run python mpm_sim.py
```

## Key Constraints

- Fixed particle tensor size (GPU requirement)
- 256×256 grid resolution baseline
- dt = 5×10⁻⁵ s for stability
- Closed-box boundaries (need damping layers for open ocean)

## Files

| File | Purpose |
|------|---------|
| mpm_sim.py | MPM physics engine |
| make_jelly.py | Morphology generator |
| pyproject.toml | Dependencies |

## Output Requirements

All outputs stored in `output/` directory:

1. **evolution_log.csv** - Full logging of ALL genomes
   - Columns: generation, individual, genome[0-8], fitness, sigma
   - Every individual from every generation

2. **best_genomes.json** - Best genome per generation
   - For replay and trajectory analysis

3. **videos/gen_{N}_best.mp4** - Video of best individual
   - One per generation showing morphology and swimming

4. **genome_heatmap.png** - Final visualization
   - Shows which genes changed most across evolution

## Adaptive Resolution Strategy

**Phase 1 (Exploration):** generations 0-70
- 16K particles, 128×128 grid
- Fast iteration for broad search

**Phase 2 (Refinement):** generations 71-100
- 36K particles, 256×256 grid
- Better vortex resolution for fine-tuning

Transition: Re-evaluate top 20% of population at high-res before continuing.

## Known Issues

1. Variable particle count incompatible with fixed GPU tensors
2. No background water in morphology output
3. Missing actuation logic
4. Boundary conditions cause wake recirculation
5. No Aurelia aurita baseline genome defined yet
