# Known Problems, Limitations, and Open Questions

This document tracks scientific and engineering issues with the simulation, fitness function, and
evolutionary methodology. Organized by severity and actionability.

---

## 1. Fluid Model Does Not Represent Water

**Severity: Critical (invalidates hydrodynamic claims)**

The "water" (Material 0) uses `mu=0, lambda=3000` — an inviscid, weakly compressible fluid.
This is physically closer to a compressible gas than to water.

Consequences:
- No viscous drag: the optimizer never learns to minimize drag or exploit boundary layer effects.
- No vortex ring formation: real jellyfish propulsion depends on starting/stopping vortex rings
  during bell contraction and refill suction during relaxation. This mechanism is entirely absent.
- No hydrostatic equilibrium: the fluid does not establish a pressure gradient under gravity before
  the simulation starts. Objects do not experience Archimedes buoyancy. Everything sinks unless
  it actively generates thrust. **This is the primary cause of the observed behavior where most
  morphologies — including the Aurelia baseline — sink despite being intended as neutral-density.**
- Compressibility artifacts: at the grid resolution used, pressure waves travel at ~55 normalized
  units/s. Actuation pulses may generate nonphysical shock-like pressure fronts.

**Observed consequence**: Evolution converges toward brute-force heavy-contraction morphologies
(large, powerful muscles) rather than hydrodynamically efficient shapes. Rankings may still be
internally consistent within the simulation, but the discovered morphologies have no guarantee
of hydrodynamic relevance in real water.

**Mitigation (without full rewrite)**: Add viscosity (`mu > 0` for water). This is a one-line
change but requires recalibrating material parameters and re-running all baselines. Does not fix
the vortex ring problem but adds drag-based selection pressure.

**Long-term fix**: Coupled MPM-LBM or MPM with proper incompressible fluid formulation.

---

## 2. Gravity Hack for Payload Buoyancy Is Incorrect and Inconsistent

**Severity: High**

The documentation (CLAUDE.md) states payload gravity is `0.44x`. The code (`mpm_sim.py:212`)
applies `gravity * 0.8`. These are different.

More fundamentally, all particles use the same `p_mass`. The claimed "2.5x density" for the
payload does not appear to be implemented in particle masses. If particle spacing is uniform,
the payload and water particles have identical mass, and `gravity * 0.8` makes the payload
*lighter* than water — slightly positively buoyant, not negatively buoyant.

Buoyancy should emerge from fluid pressure gradients, not from per-material gravity tuning.
Manually reducing gravity decouples the force from the actual fluid state.

**Proposed resolution**:
- Remove the gravity multiplier hack entirely.
- Model the payload as **neutrally buoyant with elevated inertia**: same gravity as water,
  but particle mass set to 2.5× `p_mass`. This is physically realistic — real AUV payloads
  are designed for neutral buoyancy; the hard constraint is the inertial mass of electronics
  and batteries, not gravity.
- The jellyfish must then accelerate a heavy-but-neutrally-buoyant payload upward, which is
  a cleaner and more defensible physical setup.
- Update `p_mass` assignment in `mpm_sim.py` to use `payload_mass = 2.5 * p_mass` for Material 2.

**Is neutrally buoyant payload realistic?** Yes. Standard AUV/underwater robot design targets
neutral buoyancy for the payload housing by adjusting geometry or adding syntactic foam. The
science question becomes: "can evolved morphologies efficiently accelerate a high-inertia
payload upward against a resisting fluid?" — which is cleaner than the current setup.

---

## 3. Fitness Function: Boundary Saturation with Longer Evaluations

**Severity: High**

Spawn position: y=0.7. Boundary stuck-detection threshold: y=0.93. This gives only 0.23
normalized units of vertical travel before triggering the ceiling penalty. Good swimmers
already approach this limit within 3 cycles. Doubling simulation time to 6 cycles makes
the saturation worse, not better.

**Proposed fix — velocity-based fitness**:

```python
# Measure CoM position at start of each cycle
# fitness = mean per-cycle rise over cycles 3–N (after transient settles)
fitness = mean(delta_y_cycle_k for k in range(3, n_cycles))
```

Advantages:
- Independent of absolute spawn position — boundary can never saturate fitness
- Measures sustained locomotion, not one-shot thrust
- Naturally averages over multiple cycles, reducing noise from initial transients
- Spawn can be lowered to y=0.4–0.5 to give more room while running 6+ cycles
- The stuck-detection threshold becomes less critical

**Alternative quick fix**: Lower spawn to y=0.4 while keeping displacement-based fitness.
This doubles available headroom without changing the fitness definition.

---

## 4. Single-Sample Fitness Evaluation (High Variance)

**Severity: High**

Each genome is evaluated exactly once. MPM at 128×128 resolution with compressible fluid
has significant numerical sensitivity. Two evaluations of the same genome can produce
different fitness values due to floating-point grid discretization in `fill_tank`.

This means CMA-ES is trying to estimate gradient direction from noisy rankings. The
covariance matrix update is unreliable when fitness differences between individuals are
smaller than the noise floor.

**Proposed fix**: Run each genome multiple times and average.

GPU budget options (16 slots fixed):
- **8 individuals × 2 trials**: keeps population above the CMA-ES minimum (~10 for n=9)
  while halving per-genome variance. Recommended baseline.
- **4 individuals × 4 trials**: reduces variance further but population is too small for
  reliable CMA-ES covariance estimation in 9 dimensions. Not recommended.
- **16 individuals × 1 trial + top-k re-evaluation**: run all 16 unique individuals,
  then use remaining capacity to re-evaluate the 4 highest-scoring individuals 3 more times
  and average. Costs 12 extra slots worth of time per generation but focuses replication
  where it matters most (at the selection boundary).

**Impact on runtime**: 8×2 approach does not change wall-clock time per generation (same
16 GPU slots running in parallel). It reduces effective population to 8 unique individuals
per CMA-ES step. May require more generations to converge but each step is more reliable.

---

## 5. Drift Penalty Is Disabled

**Severity: Medium (documented decision, not a bug)**

The lateral drift penalty (`- 1.0 * drift`) in `compute_fitness()` is commented out.
Current fitness is purely `displacement = final_y - init_y`.

**Justification for keeping it disabled**: Lateral tipping physically reduces vertical
displacement (converts vertical thrust to horizontal). The environment itself penalizes
drift implicitly through the fitness metric. Observed evolution of predominantly symmetric
morphologies supports this reasoning.

**Remaining concern**: A morphology that tilts 45° but maintains altitude by contracting
harder could still score well. Monitor evolved populations for asymmetric high-scorers.
If asymmetric morphologies begin dominating, re-enable the drift penalty.

**Resolution**: Keep disabled. Document the physical justification. Add a diagnostic that
logs the fraction of high-fitness individuals with significant drift (|drift| > 0.05).

---

## 6. Actuation Model Is Not Physically Realizable

**Severity: High for sim-to-real transfer; Medium for in-simulation comparisons**

Muscle actuation applies isotropic pressure (`stress += I * contractile_pressure * J`).
This means the muscle contracts equally in all spatial directions simultaneously — no
real actuator does this. Real jellyfish subumbrellar muscle is circumferentially oriented.
Pneumatic actuators expand/contract along a cavity axis. DEAs have specific polarization
directions.

Consequences:
- Overestimates thrust by allowing "squeezing" in directions the muscle physically cannot.
- Morphologies evolved for this actuation model may not be fabricable with any real actuator.
- The optimal fiber orientation (a key design variable in real soft robots) is absent from
  the genome.

**Short-term mitigation**: Add muscle fiber direction as genome parameters (2 additional
genes per muscle region: orientation angle + anisotropy ratio). This does not make the
actuation more physically accurate but makes it more flexible and allows the optimizer
to discover directional actuation strategies.

**Long-term fix**: Commit to a specific actuator technology (pneumatic chamber, cable tendon,
DEA) and implement its constitutive model. This would also inform fabrication constraints.

---

## 7. No Cost of Transport Metric

**Severity: Medium**

Current fitness maximizes upward displacement. A morphology that contracts maximally on
every step will beat an efficient swimmer if it generates more raw thrust. The simulation
has no concept of metabolic cost.

**Proposed proxy CoT** (internally consistent even with simplified physics):

```
proxy_CoT = total_muscle_work / (total_mass × vertical_displacement)
total_muscle_work = Σ_t Σ_particles (activation(t) × |active_stress| × |strain_rate| × dV × dt)
```

All quantities are available during simulation. The absolute value has no physical unit
mapping, but the *relative ranking* between morphologies is meaningful within this framework.
This would penalize the heavy-contraction strategy and select for morphologies that achieve
height gain with less actuation effort.

**Framing caveat**: Results should be presented as "relative efficiency within the simulation
framework" not as "absolute energy consumption." The proxy CoT is a dimensionless comparative
metric, not a joule count.

---

## 8. Resolution Too Coarse for Thin Structures

**Severity: Medium**

Grid resolution: 128×128. Grid spacing: dx = 1/128 ≈ 0.0078 normalized units.
Minimum genome bell tip thickness: `t_tip = 0.01` = ~1.3 grid cells.

MPM requires 3–4 cells across a structural feature for reliable stress transfer. Thin bell
tips are below this minimum. The optimizer may be penalizing thin tips not because they're
biomechanically inferior but because they're numerically underresolved.

**Fix**: Enforce minimum thickness in genome bounds: `t_tip >= 0.025` (≈3 cells).
Or run a convergence study at 256×256 to check whether thin-tip morphologies change ranking.

---

## 9. Midline Mirror Overlap

**Severity: Medium**

Symmetric morphology generation mirrors particle positions across x=0. Particles at x=0
are duplicated, creating a density spike along the bell's axis of symmetry. This artificial
stiffness acts as a structural seam.

Effect on evolution: unknown, but it may bias toward morphologies that don't stress the
midline, or may provide artificial structural rigidity that helps jellyfish maintain shape.

**Fix**: Deduplicate particles within a small tolerance (e.g., `|x| < dx/2`) during
`generate_phenotype()`.

---

## 10. 2D Simulation Cannot Capture Vortex Ring Propulsion

**Severity: High for scientific claims; Low for current goals**

Jellyfish are axisymmetric 3D structures. Their propulsion relies on 3D vortex ring
formation during bell contraction and a trailing vortex that provides a pressure wake
for the refill stroke. 2D cross-section simulation fundamentally cannot represent this.

In 2D, the "jellyfish" is effectively an infinite cylinder contracting laterally — a very
different hydrodynamic regime from a bell-shaped body of revolution.

**Short-term framing**: All results should be qualified as "2D cross-sectional morphological
trends." Claims about 3D propulsion performance cannot be made.

**Long-term path**: Axisymmetric 2D formulation (r-z coordinates) is a major improvement
over full-2D at manageable computational cost. True 3D MPM would be needed for full fidelity
but is orders of magnitude more expensive.

---

## 11. No Convergence Validation

**Severity: Medium**

The simulation has been run at a single resolution (128×128) and a single population/generation
count. There is no evidence that:
- Fitness rankings are resolution-independent (vs. 256×256)
- 50 generations at popsize=16 represents convergence (fitness plateau not confirmed)
- The discovered morphologies are global optima rather than local convergents

**Minimum required before trusting results**:
1. Run one experiment at 256×256 and compare fitness rankings of top morphologies
2. Plot fitness-vs-generation curve and confirm plateau before declaring convergence
3. Run CMA-ES from 3 independent random initializations and compare final best genomes

---

## Summary Table

| # | Issue | Severity | Actionability | Priority |
|---|-------|----------|--------------|----------|
| 1 | Inviscid fluid (no drag, no buoyancy) | Critical | Hard | Long-term |
| 2 | Gravity hack inconsistency / wrong buoyancy model | High | Easy | Fix now |
| 3 | Ceiling saturation with longer evals | High | Easy | Fix now |
| 4 | Single-sample evaluation / CMA-ES noise | High | Medium | Soon |
| 5 | Drift penalty disabled | Medium | Documented decision | Monitor |
| 6 | Isotropic / unrealizable actuation | High | Hard | Long-term |
| 7 | No Cost of Transport metric | Medium | Medium | Soon |
| 8 | Thin structures underresolved | Medium | Easy | Soon |
| 9 | Midline mirror overlap | Medium | Easy | Soon |
| 10 | 2D cannot capture vortex rings | High | Very hard | Acknowledge |
| 11 | No convergence validation | Medium | Medium | Soon |
