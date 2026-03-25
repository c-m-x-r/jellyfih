# Experiment 3 — Raw Displacement + Frequency Gene

**Status:** Planned
**Depends on:** Genome change (relaxation gene → frequency gene), refractory bound extension
**Hardware:** 4× RTX 3080 or better (tall tank is slower — see timing table)
**Target:** 4 parallel seeds, 50 generations, tall tank (128×256)

---

## Motivation

Experiment 2 revealed two things that motivate a redesign:

1. **The relaxation gene is vestigial.** Both successful runs evolved it to the 5% minimum floor. The waveform was already a near-symmetric half-arch followed by a long coast. Removing it and collapsing to a clean 2-phase model matches what evolution found and is biologically more accurate (fast twitch, passive elastic recoil).

2. **Frequency is fixed at 1 Hz, but it shouldn't be.** The cost-of-transport dynamics depend on frequency: at 2 Hz the jellyfish fires twice as often, changing how the wake evolves and how much energy is expended per unit of displacement. There may be resonant frequencies where successive pulses constructively reinforce the wake vortex dipole rather than fighting it. This is a genuinely open question.

3. **The activity-weighted fitness rewarded efficiency but suppressed raw performance.** s2024's higher raw displacement (0.728 vs 0.61) was penalised. Experiment 3 asks: what morphology simply goes *highest*, unconstrained by how much muscle work it takes?

---

## Genome Redesign: 11D (relaxation removed, frequency added)

The 3-phase waveform (contraction / relaxation / refractory) collapses to 2-phase:

```
f(t) = 0.5 × (1 − cos(π × t / c))     for t ∈ [0, c]    ← raised half-arch, 0→1→0
f(t) = 0                                for t ∈ [c, 1/f]  ← refractory (passive recoil)
```

Where `c = contraction_frac / f` (contraction window in seconds), `f = base_freq × freq_mult`.

**The relaxation gene (old gene 10 refractory_frac) is replaced by a frequency multiplier.**

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
| 9 | contraction_frac | [0.05, **0.60**] | **0.20** | fraction of cycle with nonzero activation |
| 10 | freq_mult | [**0.5**, **2.0**] | **1.0** | multiplier on 1 Hz base; range 0.5–2.0 Hz |

**Refractory is now implicit**: `refractory_frac = 1 − contraction_frac`. No gene needed.

**contraction_frac upper bound extended to 0.60** (from 0.40): refractory lower bound is implicitly 0.40. High freq_mult + high contraction_frac = continuous firing at high frequency — a physiologically extreme but valid search direction.

### Frequency coupling with fitness

The **effective active fraction per unit time** is:
```
active_rate = contraction_frac × freq_mult
```

At 2 Hz, 20% contraction: fires 40% of total time.
At 0.5 Hz, 40% contraction: fires 20% of total time.

The sim runs for a fixed number of steps (150,000 = 3s at dt=2e-5). High freq_mult means more pulses per run. This couples frequency and morphology in the fitness landscape — a high-frequency jellyfish with a large muscle will score very differently from a low-frequency efficient one.

---

## Fitness Function

**Raw displacement — no efficiency penalty:**

```python
fitness = min(final_y, ceiling) - init_y
```

Simple. Every metre of upward travel counts equally regardless of how much muscle it took. This directly answers: *what design goes highest?*

**Why this is interesting vs Exp 2:**
- Expected to discover larger muscles (no penalty for muscle count)
- Expected to find shorter refractory / higher frequency (firing more is free)
- May converge to different morphology than Exp 2 (wider/stronger bell?)
- Will demonstrate whether the Exp 2 attractor was shaped by efficiency pressure or by the physics of the problem

**No ceiling cap penalty.** Use **tall tank** (128×256, ceiling at y=1.93) to prevent premature saturation. With 3s @ 1Hz the payload travels at most ~0.7 units in Exp 2 — well within the tall tank domain.

---

## Configuration

| Parameter | Value | Change from Exp 2 |
|-----------|-------|-------------------|
| Genome | **11D** — shape (6) + thickness (3) + contraction (1) + freq_mult (1) | relaxation → freq_mult |
| Gene 9: contraction_frac | [0.05, 0.60], init 0.20 | upper bound extended |
| Gene 10: freq_mult | [0.5, 2.0], init 1.0 | **new** |
| refractory_frac | implicit: `1 − contraction_frac` | removed as gene |
| Fitness | **`displacement`** (raw) | simplified |
| Tank | **128×256 (tall)**, ceiling y=1.93 | switched from square |
| Population (λ) | 32 | same |
| Generations | 50 | same |
| Steps/eval | 150,000 | same |

---

## Expected outcomes

- **Higher muscle counts**: no muscle-count penalty → bell grows heavier
- **Frequency convergence**: expect discovery of a preferred band, possibly 1.2–1.8 Hz based on vortex advection timescale from Exp 1 fluid analysis (35% residual wake at 1 Hz — a slightly higher frequency might fire just as the previous vortex dipole has advected clear of the bell)
- **Morphology may diverge from Exp 2 attractor** if efficiency pressure was driving the locked gene values
- **contraction_frac**: may stay moderate (0.20–0.35) — very long contractions may reduce net displacement per cycle
- **Multiple attractors likely**: the frequency-morphology coupling creates a richer landscape than Exp 2

---

## Analysis plan

1. **Frequency trajectories**: plot freq_mult per generation — does it converge or stay diffuse?
2. **Attractor comparison**: compare final morphology genes to Exp 2 — same or different locked genes?
3. **Frequency × contraction heatmap**: fitness as a function of g9 × g10 in final generation — is there a ridge?
4. **Wake timing**: fluid analysis on best genome — does the evolved frequency better exploit wake residuals?
5. **Condition number**: does adding frequency gene increase or decrease the condition number trajectory?

---

## Run commands

```bash
# Requires: genome updated in make_jelly.py, evolve.py, mpm_sim.py
# Tank: JELLY_GRID_Y=256 JELLY_DOMAIN_H=2.0 for tall tank
cd /root/jellyfih
mkdir -p logs

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 JELLY_INSTANCES=32 JELLY_GRID_Y=256 JELLY_DOMAIN_H=2.0 \
  uv run python evolve.py --gens 50 --seed 42   --run-id exp3_s42   > logs/exp3_s42.log  2>&1 &
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 JELLY_INSTANCES=32 JELLY_GRID_Y=256 JELLY_DOMAIN_H=2.0 \
  uv run python evolve.py --gens 50 --seed 137  --run-id exp3_s137  > logs/exp3_s137.log 2>&1 &
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2 JELLY_INSTANCES=32 JELLY_GRID_Y=256 JELLY_DOMAIN_H=2.0 \
  uv run python evolve.py --gens 50 --seed 999  --run-id exp3_s999  > logs/exp3_s999.log 2>&1 &
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=3 JELLY_INSTANCES=32 JELLY_GRID_Y=256 JELLY_DOMAIN_H=2.0 \
  uv run python evolve.py --gens 50 --seed 2024 --run-id exp3_s2024 > logs/exp3_s2024.log 2>&1 &
```

*Note: tall tank is ~2.3× slower than square — expect ~480s/gen on 3080, ~6.7 hrs for 50 gens.*

---

## Success criteria

| Criterion | Threshold | Interpretation |
|-----------|-----------|----------------|
| Raw displacement > Exp 2 best | > 0.63 | Unconstrained fitness finds better raw performance |
| Frequency convergence | CV(freq_mult) < 0.3 in ≥2 runs | A preferred frequency range exists |
| Morphology shift | end_x or t_mid CV > 0.15 cross-run | Efficiency pressure was shaping Exp 2 attractor |
| Freq × contraction coupling | cov(g9, g10) > 0.3 | Frequency and contraction interact |

---

## Failure modes

- **Ceiling reached even in tall tank**: if raw displacement > 1.5 units, extend steps or use 3-cycle window
- **Frequency rail to 2.0**: if all runs max freq_mult, extend upper bound to 3.0 Hz
- **s999 crash repeat**: stagger launch by 5s to avoid CUDA init race condition

---

## Results

**Date:** 2026-03-25
**Hardware:** 4× RTX 3080 Ti, seed 42, single replicate
**Status:** ✅ Complete (50/50 gens)

| Metric | Value | Notes |
|--------|-------|-------|
| Best displacement | **+0.838** (gen 43) | ✅ Exceeds >0.63 threshold |
| Final-gen best | +0.816 | Slight late-gen noise |
| 90% convergence | gen 16 | Fast — attractor found early |
| Final sigma | 0.101 | Tight — fully converged |
| Final cond # | 62.9 | Extremely narrow ridge |
| Muscle count (best) | 674 | High — no penalty for size |
| Sim time/gen | ~368s | On 3080 Ti, tall tank |

### Locked genes (all CV < 0.15 — fully converged)

| Gene | Final mean | Bound pressing? |
|------|-----------|-----------------|
| end_x | 0.348 | ← upper (0.35) |
| t_base | 0.079 | ← upper (0.08) |
| contraction_frac | 0.550 | ← upper (0.60) |
| freq_mult | 0.900 | — |
| cp1_x | 0.005 | ← lower (0.0) |
| cp2_y | −0.195 | ← lower (−0.20) |

### Key findings

- **Same morphological attractor as Exp 1/2**: wide bell, end_x at upper bound, thick base — efficiency pressure was not shaping the Exp 2 morphology. The bell shape is intrinsic to the physics, not the cost function.
- **Contraction pressing upper bound (0.55/0.60)**: with no muscle-count penalty, evolution maximises firing fraction. The jellyfish wants to contract 55% of each cycle.
- **freq_mult settled at 0.90** (slightly sub-1 Hz): counter-intuitive — without frequency being penalised, evolution chose slightly *slower* than baseline. Likely the vortex wake requires a minimum recovery time regardless of muscle cost.
- **674 muscles** vs Exp 6's 472 — confirms efficiency pressure was suppressing muscle growth in Exp 2.

### Success criteria outcome

| Criterion | Result |
|-----------|--------|
| Displacement > 0.63 | ✅ 0.838 |
| Frequency convergence (CV < 0.3) | ✅ CV = 0.03 |
| Morphology shift from Exp 2 | ❌ Same attractor (end_x, t_base locked identically) |
| Freq × contraction coupling | Not measured (single seed) |
