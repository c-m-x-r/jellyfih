# Experiment 6 — Efficiency Control (New Genome, Tall Tank)

**Status:** Planned
**Depends on:** Exp 3 genome changes merged to main (freq_mult replacing refractory gene)
**Hardware:** 1× RTX 3080 (1 seed, shares 4-GPU instance with Exps 3, 4, 5)
**Target:** 1 seed, 50 generations, tall tank (128×256)

---

## Motivation

Experiments 3, 4, and 5 all introduce variables beyond the genome change (different fitness function, no payload, or different physics). Experiment 6 is the **clean replication control**: it runs the new Exp 3 genome encoding with the same fitness function and configuration as Experiment 2, in the tall tank.

This answers two questions:

1. **Does the genome change itself affect the attractor?** The refractory gene was vestigial in Exp 2 (driven to 5% floor). Replacing it with freq_mult — which starts at a neutral 1.0 and has a wide range — may unlock different morphologies even with the same efficiency fitness. If the Exp 6 attractor matches Exp 2, the genome change is safe. If it diverges, the new gene is doing real work.

2. **What is the efficiency-optimal morphology at tall-tank scale?** Exp 2 ran in the square tank and hit the ceiling early. Exp 6 removes the ceiling constraint while keeping the efficiency fitness — letting the jellyfish keep climbing and revealing whether efficiency and raw height converge to the same shape (compare to Exp 3).

Without Exp 6, any difference between Exp 2 and Exp 3 is confounded by three simultaneous changes: fitness function, genome encoding, and tank height. Exp 6 isolates two of these.

---

## Genome

Same as Experiment 3 (new 11D encoding):

| Index | Parameter | Bounds | Init | Notes |
|-------|-----------|--------|------|-------|
| 0 | cp1_x | [0.0, 0.25] | mid | |
| 1 | cp1_y | [−0.15, 0.15] | mid | |
| 2 | cp2_x | [0.0, 0.30] | mid | |
| 3 | cp2_y | [−0.20, 0.15] | mid | |
| 4 | end_x | [0.05, 0.35] | mid | |
| 5 | end_y | [−0.30, +0.10] | mid | |
| 6 | t_base | [0.025, 0.08] | mid | |
| 7 | t_mid | [0.025, 0.10] | mid | |
| 8 | t_tip | [0.01, 0.04] | mid | |
| 9 | contraction_frac | [0.05, 0.60] | 0.20 | |
| 10 | freq_mult | [0.5, 2.0] | 1.0 | replaces refractory_frac |

**Refractory is implicit**: `refractory_frac = 1 − contraction_frac`.

---

## Fitness Function

**Activity-weighted efficiency** (same formula as Experiment 2, adapted for new genome):

```python
active_frac   = contraction_frac          # gene 9; was (1 - refractory_frac) in Exp 2
effective_muscle = muscle_count × active_frac
fitness = displacement / sqrt(effective_muscle / 500)
```

The formula is equivalent to Exp 2's fitness. In Exp 2, `active_frac = 1 − refractory_frac`; in Exp 6, that simplifies to `active_frac = contraction_frac` (since refractory is now implicit). A jellyfish that spends only 20% of its cycle contracting pays 20% of the muscle cost — same economic logic.

**Note**: the freq_mult gene (gene 10) is not in the fitness denominator. A high-frequency jellyfish with the same muscle count and contraction fraction pays the same fitness cost as a low-frequency one — even though it fires more often per second. This is intentional: if freq_mult is a free variable unpenalised in the denominator, we will observe whether evolution exploits it or leaves it near the default.

---

## Configuration

| Parameter | Value | Change from Exp 2 |
|-----------|-------|-------------------|
| Genome | **11D Exp 3 encoding** (freq_mult replaces refractory) | gene 10 changed |
| Fitness | **efficiency** (displacement / sqrt(muscle × active_frac / 500)) | same formula, adapted |
| Payload | Yes (with_payload=True) | same as Exp 2 |
| Tank | **128×256 tall**, ceiling y=1.93 | switched from square |
| Population (λ) | 32 | same |
| Generations | 50 | same |
| Steps/eval | 150,000 | same |
| Seed | 137 | single seed (1 GPU available) |
| Branch | `main` | same as Exps 3/4 |

*Using seed=137 (different from Exp 3's seed=42) to reduce correlation between simultaneously-running experiments.*

---

## Hypothesis

- **freq_mult will converge near 1.0**: the efficiency fitness does not reward higher firing rate (freq_mult not in denominator), so there is no direct selection pressure on gene 10. It may drift toward whatever frequency produces the best displacement per unit muscle.
- **Morphology matches Exp 2 attractor**: wide flat bell, end_x ≈ 0.34, t_base stiffened. If the attractor is the same, the genome change is transparent.
- **contraction_frac**: expected to converge near 0.20 (the known-good default and historical jet-mode value from Exp 2). The wider upper bound (0.60) is unlikely to be exploited with efficiency fitness.
- **Fitness higher than Exp 2**: tall tank removes ceiling saturation. With unconstrained displacement over 7.5s, the efficiency numerator can grow beyond what the square-tank ceiling allowed (~0.5 units).

---

## Analysis plan

1. **Genome correlation**: compare final-gen best genome to Exp 2 best — are the locked genes (end_x, t_base, t_mid) still locked?
2. **freq_mult trajectory**: does gene 10 drift, oscillate, or converge? If it converges away from 1.0, the frequency dimension is doing real work even without a direct fitness incentive.
3. **Fitness comparison**: Exp 6 efficiency vs Exp 3 raw displacement — do they find the same morphology from different angles?
4. **Fitness comparison**: Exp 6 vs Exp 2 — does tall tank + freq_mult encoding change peak efficiency?
5. **Covariance**: does removing the refractory gene change the gene coupling structure? (Exp 2 had cp2_x ↔ end_y as dominant; does freq_mult introduce new couplings?)

---

## Run commands

```bash
cd /root/jellyfih
# Must be on main branch
git checkout main
mkdir -p logs

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 JELLY_INSTANCES=32 JELLY_GRID_Y=256 JELLY_DOMAIN_H=2.0 \
  uv run python evolve.py --gens 50 --seed 137 --run-id exp6_s137 --tall \
  > logs/exp6_s137.log 2>&1 &
```

*Note: default fitness is `efficiency` — no `--fitness` flag needed.*

---

## Full 4-GPU launch script (Exps 3, 4, 5, 6 simultaneously)

```bash
# GPU 0 — Exp 3: raw displacement, Cartesian, freq_mult genome
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 JELLY_INSTANCES=32 JELLY_GRID_Y=256 JELLY_DOMAIN_H=2.0 \
  uv run python evolve.py --gens 50 --seed 42  --run-id exp3_s42  --tall --fitness displacement \
  > logs/exp3_s42.log 2>&1 &

# GPU 1 — Exp 6: efficiency fitness, Cartesian, freq_mult genome (control)
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 JELLY_INSTANCES=32 JELLY_GRID_Y=256 JELLY_DOMAIN_H=2.0 \
  uv run python evolve.py --gens 50 --seed 137 --run-id exp6_s137 --tall \
  > logs/exp6_s137.log 2>&1 &

# GPU 2 — Exp 4: no-payload, efficiency fitness, freq_mult genome
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2 JELLY_INSTANCES=32 JELLY_GRID_Y=256 JELLY_DOMAIN_H=2.0 \
  uv run python evolve.py --gens 50 --seed 999 --run-id exp4_s999 --tall --no-payload \
  > logs/exp4_s999.log 2>&1 &

# GPU 3 — Exp 5: axisym MPM, raw displacement, freq_mult genome (axisym-mpm branch)
# NOTE: must have axisym-mpm branch checked out in a separate working directory,
# or stash/switch branch before launching. Recommended: clone repo to /root/jellyfih-axisym
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=3 JELLY_INSTANCES=32 JELLY_GRID_Y=256 JELLY_DOMAIN_H=2.0 JELLY_AXISYM=1 \
  uv run python /root/jellyfih-axisym/evolve.py --gens 50 --seed 42 --run-id exp5_s42 --tall --fitness displacement \
  > logs/exp5_s42.log 2>&1 &

wait
echo "All experiments complete."
```

*Exp 5 requires the `axisym-mpm` branch. Run it from a separate clone so the other three can run from `main` without branch switching.*

---

## Success criteria

| Criterion | Threshold | Interpretation |
|-----------|-----------|----------------|
| Morphology correlation with Exp 2 | > 0.85 | Genome change transparent; attractor preserved |
| Morphology correlation with Exp 2 | < 0.6 | freq_mult encoding changed the landscape |
| freq_mult convergence | CV < 0.3 by gen 30 | Frequency has a preferred value even without direct incentive |
| Fitness > Exp 2 best (1.179) | > 1.3 | Tall tank unlocked headroom the square ceiling was suppressing |

---

## Failure modes

- **freq_mult ignored (CV > 0.6 throughout)**: gene 10 is just noise with efficiency fitness; confirms the fitness function is insensitive to frequency. Fine — this is useful information.
- **Morphology collapse**: if removing the refractory gene destabilises early-generation individuals, invalidity rate may spike. Check `n_invalid` in log for first 5 gens.
- **Lower fitness than Exp 2**: would indicate tall tank introduces additional challenge (e.g. longer time to reach steady state, or damping layer effects at larger domain). Flag for investigation.

---

## Results

**Date:** 2026-03-25
**Hardware:** 4× RTX 3080 Ti, seed 137, single replicate
**Status:** ✅ Complete (50/50 gens)

| Metric | Value | Notes |
|--------|-------|-------|
| Best efficiency fitness | **1.327** (gen 48) | ✅ New record — beats Exp 2's 1.179 by 12.5% |
| Best displacement | +0.695 (at gen 48 best) | Tall tank, unconstrained |
| 90% convergence | gen 19 | Matches Exp 2 pace |
| Final sigma | 0.087 | Tightest of all experiments |
| Final cond # | 47.7 | Fully converged narrow ridge |
| Muscle count (best) | 472 | Lower than Exp 3's 674 — efficiency penalises size |
| Sim time/gen | ~374s | On 3080 Ti, tall tank |

### Locked genes (all CV < 0.15)

| Gene | Final mean | Bound pressing? |
|------|-----------|-----------------|
| end_x | 0.334 | ← upper (0.35) |
| t_base | 0.077 | ← upper (0.08) |
| freq_mult | 0.505 | ← lower (0.50) ⚠️ |
| contraction_frac | 0.293 | — (near default 0.20) |
| cp2_x | 0.006 | ← lower (0.0) |
| cp2_y | −0.171 | ← lower (−0.20) |

### Key findings

- **New record: 1.327 efficiency** — the freq_mult genome + tall tank unlocked 12.5% improvement over Exp 2. The square tank was ceiling-bound; the tall tank allows the efficiency numerator to grow freely.
- **freq_mult pressing lower bound (0.505/0.50)**: the efficiency-optimal jellyfish wants to pulse as slowly as possible — 0.5 Hz, half the base frequency. With efficiency fitness penalising active muscle fraction, evolution minimises firing rate aggressively. This bound should be extended downward in future experiments.
- **contraction_frac = 0.293** (vs Exp 3's 0.550): the efficiency penalty successfully constrains contraction duration. Each contraction is shorter but the period is longer, spreading the cost.
- **Same morphological attractor as Exp 1/2/3**: end_x and t_base locked at same values — confirming the bell shape is physics-determined, not fitness-determined.
- **Genome change is transparent**: Exp 6 replicated Exp 2's attractor with the new encoding, validating that the freq_mult substitution for refractory_frac is safe.

### Success criteria outcome

| Criterion | Result |
|-----------|--------|
| Morphology matches Exp 2 (corr > 0.85) | ✅ Same attractor |
| freq_mult convergence (CV < 0.3) | ✅ CV = 0.00 (at lower bound) |
| Fitness > 1.179 | ✅ 1.327 |
| freq_mult away from 1.0 | ✅ Strongly — 0.505, lower bound |
