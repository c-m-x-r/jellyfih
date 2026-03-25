# Jellyfih Experiment Log

---

## Experiment 1 — Baseline Morphology Search

**Date:** 2026-03-24
**Hardware:** 1× RTX 4090 (sequential, seed 42) + 4× RTX 3080 (parallel, 4 seeds)
**Instance:** vast.ai — 4090 @ $0.30/hr destroyed after run; 4× 3080 @ $0.275/hr

### Configuration

| Parameter | Value |
|-----------|-------|
| Genome | 9D (shape + thickness only) |
| end_y bounds | [−0.30, −0.03] — tips constrained downward |
| Population (λ) | 16 |
| Generations | 50 |
| Steps/eval | 150,000 (3 cycles @ 1 Hz) |
| Fitness | `displacement / sqrt(muscle_count / 500)` |
| Spawn | [0.5, 0.40] |
| Actuation | Fixed 20/40/40 contraction/relaxation/refractory |
| Gravity | 10.0, payload 2.5× density, no buoyancy |

### Data

| Run | Seed | Best fitness | Gen converged | Notes |
|-----|------|-------------|---------------|-------|
| 4090-s42 | 42 | **0.536** (gen 42) | ~gen 40 | Full 50 gens |
| 3080-R0 | — | ~0.52 | ~gen 10 | |
| 3080-R1 | — | ~0.53 | ~gen 10 | |
| 3080-R2 | — | **0.535** (gen 41) | ~gen 15 | Only run to cross 0.53 |
| 3080-R3 | — | ~0.52 | ~gen 10 | cp1_x outlier, cond# 51 |

**Output:** `output/cloud/4090/seed_42/`, `output/cloud/3080x4/`
**Videos:** `view_gen{0,12,24,36,42}.mp4`, `view_gen{36,42}_long.mp4` (600 frames, 15 cycles)

### Results

**All runs converge within 10 generations** to a common attractor — a wide, nearly-flat bell with tips pressing against the upper end_y bound (−0.03). Cross-run genome correlation ~0.96.

**Locked genes** (CV < 0.15): `end_x` (0.277), `t_base` (0.054), `t_mid` (0.045). These are structural constants — evolution found one wall thickness solution and never deviated.

**Pressing against bounds:**
- `end_y` at 88% of [−0.30, −0.03] → tips want to be flatter/curl upward beyond −0.03
- `cp1_y` at 14% of [−0.15, +0.15] → first control point pulled downward, creating outward flare

**Dominant covariance coupling:** `cp2_x ↔ end_y` (−0.26 to −0.36): wider bell → shallower tip. This encodes the outward-flaring morphology.

**Ceiling exploit confirmed:** Payload reaches y ≈ 0.88 within 3 cycles from spawn y = 0.40. Fitness plateaus because ceiling (validity check at 0.93) caps raw displacement. Later generations optimize efficiency (muscle count) not displacement.

**Fluid dynamics analysis** (`helpers/fluid_analysis.py`):
- Wake not dissipated at next stroke: 35% of peak momentum remains at refractory end
- Vorticity *rises* during first 150ms of refractory (vortex ring still rolling up)
- Ceiling impact (top_flux) small but real (~1.3% of domain momentum at peak)
- Floor damping working correctly (bottom_flux ≈ 0)

### Discussion

The evolved morphology is a **wide, flat-tipped bell that flares outward**. The upstroke provides lift; the recovery (expansion) generates drag, which the outward flare partially decouples laterally. This is non-biomimetic: a real *Aurelia aurita* cannot have outward-curving bell tips due to radial symmetry constraints — but that's the point.

The 20/40/40 actuation timing fires into an undissipated vortex wake every cycle (35% residual momentum). The jellyfish cannot exploit this because timing is fixed. Evolved shapes may be locally optimal *for this fixed timing* but globally suboptimal.

**R3 (cond# 51, pressing against cp1_x and end_y bounds):** this run found a wider-swept bell on a different ridge. Its poor convergence (σ=0.086 at gen 49) suggests it was still exploring when the run ended, and may be hitting a gene bound constraint.

### Next Steps → See Experiment 2

---

## Experiment 2 — Timing-Free Morphology + Cup Bells

**Date:** 2026-03-24 (planned)
**Hardware:** 4× RTX 3080 (4 parallel seeds, CUDA_VISIBLE_DEVICES=0/1/2/3)
**Instance:** vast.ai 4× 3080 @ $0.275/hr, ~3 hrs, ~$0.85

### Hypothesis

1. **end_y upper bound relaxation** (→ +0.10): morphologies press against the flat bound; allowing positive end_y will unlock cup-shaped bells with upward-curling tips, fundamentally different propulsion geometry.

2. **Evolved timing** (genes 9–10): the fixed 20/40/40 waveform fires into its own wake (35% residual momentum at refractory end). Freeing contraction_frac and refractory_frac should evolve toward jet-mode (short sharp stroke + long coast) or discover constructive resonance with the wake.

3. **Fitness function with active_frac**: `displacement / sqrt(muscle_count × (1 − refractory_frac) / 500)`. A jellyfish with 70% refractory pays only 30% of the muscle cost vs one firing 60% of the time at equal particle count. This correctly rewards energy-efficient propulsion.

4. **λ=32**: doubles population, better sampling of the now 11D search space. At ~212 sec/gen × 50 gens = ~3 hrs per seed, all 4 run simultaneously.

### Configuration

| Parameter | Value | Change from Exp 1 |
|-----------|-------|-------------------|
| Genome | **11D** (shape + thickness + timing) | +2 genes |
| end_y bounds | **[−0.30, +0.10]** | upper relaxed from −0.03 |
| Gene 9: contraction_frac | [0.05, 0.40], **init 0.20** | new |
| Gene 10: refractory_frac | [0.20, 0.75], **init 0.40** | new |
| Population (λ) | **32** | doubled |
| Generations | 50 | same |
| Steps/eval | 150,000 | same |
| Fitness | `displacement / sqrt(muscle_count × (1−refractory) / 500)` | updated |
| Ceiling validity | 0.93 (unchanged) | same |
| Actuation | Per-instance from genome genes 9–10 | now evolved |

**Timing init rationale:** Start timing genes at known-good defaults (0.20/0.40) rather than random midpoint, so CMA-ES explores timing from a viable baseline. Random timing init risks spending early gens finding a workable waveform before improving shape.

### Expected outcomes

- **End_y > 0:** cup-shaped morphologies. Bell tips curl upward — possible vortex-trapping geometry. Qualitatively different to all Exp 1 morphologies.
- **Jet mode convergence:** refractory_frac → ~0.65–0.70, contraction_frac → ~0.10–0.15. Short powerful stroke, long coast in contracted position. The current 35% residual wake should drop dramatically.
- **Locked genes may shift:** t_base/t_mid/end_x may unlock once the timing axis is free, because the coupling between shape and timing could reveal trade-offs.
- **Higher peak fitness:** with full wake dissipation, each stroke operates in clean fluid — expected 10–20% improvement in displacement.

### Run commands (on 4× 3080 instance)

```bash
cd /root/jellyfih
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 JELLY_INSTANCES=32 \
  uv run python evolve.py --gens 50 --seed 42   --run-id exp2_s42   > logs/exp2_s42.log  2>&1 &
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 JELLY_INSTANCES=32 \
  uv run python evolve.py --gens 50 --seed 137  --run-id exp2_s137  > logs/exp2_s137.log 2>&1 &
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2 JELLY_INSTANCES=32 \
  uv run python evolve.py --gens 50 --seed 999  --run-id exp2_s999  > logs/exp2_s999.log 2>&1 &
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=3 JELLY_INSTANCES=32 \
  uv run python evolve.py --gens 50 --seed 2024 --run-id exp2_s2024 > logs/exp2_s2024.log 2>&1 &
```

### Analysis plan

Post-run, in addition to standard convergence plots:
1. **Timing gene trajectories:** plot contraction_frac and refractory_frac over generations — does the population converge on jet mode (high refractory) or resonance (low refractory)?
2. **end_y distribution:** do any runs find positive end_y? What fitness do cup morphologies achieve vs flat?
3. **Re-run fluid analysis** on best genome from Exp 2 — compare residual wake at refractory end to Exp 1's 35%.
4. **Render timing sweep:** `helpers/tune_actuation.py`-style sweep over contraction/refractory pairs to visualize the timing fitness landscape.
5. **Covariance coupling:** does the timing↔shape coupling emerge in the covariance matrix? (e.g. short contraction ↔ wide bell?)

### Success criteria

| Criterion | Threshold | Interpretation |
|-----------|-----------|----------------|
| Best fitness improvement | > 0.55 | timing + shape unlocked something Exp 1 could not reach |
| Jet mode discovered | refractory_frac > 0.60 in ≥2 runs | wake dissipation hypothesis confirmed |
| Cup morphology viable | any run with end_y > 0.02 in top 5 individuals | upward-curling tips competitive |
| Residual wake reduction | < 20% of peak momentum at refractory end | fluid analysis on Exp 2 best |
| Cross-run convergence | genome correlation > 0.85 (same attractor) or < 0.6 (multiple basins) | determines whether timing opened new landscape |

### Failure modes to watch

- **Timing genes ignored (converge to 0.20/0.40 default):** shape evolution dominates and timing doesn't matter for the ceiling-capped fitness. If this happens, the fitness function needs to explicitly reward sustained efficiency over longer sims.
- **Invalids spike early:** if many random timing combinations produce invalid morphologies (muscle_count < 200 due to poorly-timed loads), may need to reduce initial sigma on timing genes (e.g. `sigma0=0.05` for genes 9–10 only). CMA-ES doesn't support per-gene sigma but can be worked around with gene scaling.
- **Ceiling still hit immediately:** if λ=32 still saturates at y≈0.88 within 10 gens, the fitness metric is fundamentally ceiling-bound regardless of timing. Should then switch to average velocity metric.
- **cond# blows up:** if condition number > 500 by gen 30, the 11D landscape has collapsed to a 1D ridge and further gens are wasted. Stop early and analyse the dominant eigenvector.

### Open questions for future experiments

1. **Frequency as a gene:** currently 1 Hz fixed. Adding actuation_freq ∈ [0.3, 3.0 Hz] would allow evolution to find resonant frequencies with the tank geometry. Risk: highly coupled with timing genes.

2. **Longer evaluation (6+ cycles):** the ceiling exploit saturates at 3 cycles. 6 cycles at 150K steps each = 900K steps/eval (~12 min on 3080 per generation). Too slow for λ=32. Possible compromise: 4 cycles at 80K steps, medium-fast.

3. **Multi-objective:** Pareto front over (displacement, muscle_cost). Would explicitly separate "reach the ceiling" from "do it efficiently" without requiring a scalarised fitness.

4. **2.5D simulation:** the 2D model overestimates thrust (no out-of-plane leakage) and underestimates drag (infinite Re). A 3D axisymmetric (quasi-2D revolve) would be closer to real Aurelia without full 3D cost.

5. **Physical validation target:** goal is a 120mm diameter bell carrying a ~30g sensor pack (approximately 2.5× water density at that scale). Need to verify the normalised payload density corresponds correctly given the physical scaling (1 unit ≈ 48 cm).

---

---

## Experiment 3 — Raw Displacement + Frequency Gene

**Date:** 2026-03-25
**Hardware:** 4× RTX 3080 Ti, 1 seed (s42), ~5.1 hrs
**Status:** ✅ Complete 50/50 gens

**Best fitness:** +0.838 displacement (gen 43). 90% convergence at gen 16.

**Key results:**
- Fully converged: sigma=0.101, cond#=62.9, all 11 genes locked (CV < 0.15)
- **Same morphological attractor as Exp 1/2**: end_x=0.348 (pressing upper 0.35), t_base=0.079 (pressing upper 0.08). Shape is physics-determined, not fitness-determined.
- **Contraction pressing upper bound**: contraction_frac=0.550 (upper=0.60). Without efficiency penalty, evolution maximises firing fraction. Muscle count: 674 vs Exp 6's 472.
- **freq_mult settled at 0.90**: slightly below 1 Hz. Counter-intuitive — even without a frequency penalty, evolution chose slightly slower than baseline, suggesting a minimum vortex recovery time requirement.
- Beats Exp 2 raw displacement (~0.61–0.73) with fewer gens; confirms displacement headroom was there all along, not unlocked by timing genes specifically.

---

## Experiment 5 — Axisymmetric MPM

**Date:** 2026-03-25
**Hardware:** 4× RTX 3080 Ti, 1 seed (s42), ~1.8 hrs (131s/gen — 2.8× faster than Cartesian due to half particles)
**Status:** ✅ Complete 50/50 gens — **DID NOT CONVERGE**

**Best fitness:** +1.274 displacement (gen 33, axisym coordinates).

**Key results:**
- Numerically stable all 50 gens — no NaN explosions, axis BC working correctly
- **Did not converge**: sigma=0.41, cond#=21.2 at gen 49. Axisym landscape is much flatter than Cartesian (Exp 3 cond#=62.9). Would need more gens or higher lambda.
- **Different attractor from Cartesian**: cp2_y = +0.114 (positive, vs Cartesian −0.19); end_x=0.266 (not pressing upper bound); t_base=0.055 (thinner). Axisym geometry genuinely changes optimal bell shape.
- **freq_mult at 1.143** (above baseline): opposite direction from Exp 6 (0.50 Hz). Cylindrical pressure recovery may slightly favour higher firing rate.
- Several genes unlocked (cp1_y, t_mid, contraction_frac CV > 0.15) — landscape too broad for single seed at 50 gens.
- Vortex ring visualisation (vs Cartesian dipoles) not yet rendered.

---

## Experiment 6 — Efficiency Control (New Genome, Tall Tank)

**Date:** 2026-03-25
**Hardware:** 4× RTX 3080 Ti, 1 seed (s137), ~5.2 hrs
**Status:** ✅ Complete 50/50 gens — **New efficiency record**

**Best fitness:** 1.327 efficiency (gen 48). 90% convergence at gen 19.

**Key results:**
- **New record: 1.327** — beats Exp 2's 1.179 by 12.5%. Tall tank (no ceiling) + freq_mult encoding both contributed.
- Tightest convergence of all experiments: sigma=0.087, cond#=47.7, all genes locked.
- **freq_mult pressing lower bound (0.505/0.50 Hz)**: efficiency fitness drives evolution to pulse as infrequently as possible. Lower bound should be extended to 0.25–0.30 Hz in future experiments.
- **contraction_frac=0.293**: much lower than Exp 3's 0.550. Efficiency pressure keeps contractions short; frequency pressure keeps cycles long.
- Same morphological attractor as Exp 1/2/3 confirmed — shape is fully physics-determined.
- Genome change (freq_mult replacing refractory_frac) is transparent: Exp 6 reproduced the Exp 2 attractor.

---

## Experiment 4 — Payload Effect (Partial)

**Date:** 2026-03-25
**Hardware:** 4× RTX 3080 Ti, seed 999, 2/50 gens (checkpoint saved)
**Status:** 🔖 Incomplete — GPU reassigned mid-run

**Brief:** Validity bug found and fixed (body CoM tracking when no material-2 particles). GPU 2 banned (broken fan). Ran 2 gens on GPU 3 after Exp 5 freed it, then killed to avoid leaving 2 GPUs idle. Gen 2 best body CoM displacement +0.878 (healthy early-gen value). Resume from checkpoint for full results.

---

## Physical Model Assumptions (reference)

See memory files:
- `~/.claude/projects/-home-mc-projects-jellyfih/memory/physics_water.md` — inviscid fluid, ~7× slow speed of sound, no viscosity
- `~/.claude/projects/-home-mc-projects-jellyfih/memory/physics_payload.md` — near-rigid 2.5× density, no Archimedes buoyancy, Δy=−0.089/3-cycle neutral sink
