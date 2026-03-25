# Experiment 4 — Payload Effect on Morphology

**Status:** Planned
**Depends on:** Exp 2 complete (provides baseline attractor for comparison)
**Hardware:** 4× RTX 3080 (same as Exp 2)
**Target:** 4 parallel seeds, 50 generations, tall tank

---

## Motivation

The Exp 1 and Exp 2 attractor — a wide, flat-tipped bell with locked `end_x` ≈ 0.34 — emerged in simulations that included a payload (0.08 × 0.05, density 2.5× water) hanging below the bell. The payload has two effects:

1. **Structural**: the bell must support and accelerate a near-rigid object that sinks at Δy = −0.089/3-cycle run without thrust. The spawn geometry places the payload CoM below the bell's elastic anchor.
2. **Dynamical**: the payload adds inertia to the system. The bell must generate enough upward impulse to overcome payload gravity at every stroke.

**Question: is the attractor a property of the jellyfish body plan, or is it shaped by the payload?**

If the attractor is intrinsic — if a payloadless jellyfish evolves the same wide bell — then the shape is driven by hydrodynamic efficiency in 2D MPM. If the attractor *shifts* (narrower bell, thinner walls, different timing), then the payload is a meaningful design constraint and its properties (size, density, position) are worth varying in future experiments.

This experiment is **half set up already**: `fill_tank()` accepts `with_payload=False` and `evolve.py` passes this through to all fitness evaluations. Only config changes are needed.

---

## Configuration

| Parameter | Value | Change from Exp 2 |
|-----------|-------|-------------------|
| Genome | **11D** (same as Exp 2 — or Exp 3 if genome redesign is merged first) | same |
| Payload | **None** | `with_payload=False` |
| Fitness | `displacement / sqrt(muscle_count × (1−refractory) / 500)` | same as Exp 2 |
| Gravity | 10.0 (unchanged — water still has inertia) | same |
| Tank | **128×256 (tall)** | switched from square |
| Population (λ) | 32 | same |
| Generations | 50 | same |
| Steps/eval | 150,000 | same |
| Spawn | [0.5, 0.40] (jellyfish only — no payload spawn) | adjusted |

*Note: if the Exp 3 genome redesign (relaxation → frequency gene) is implemented first, run Exp 4 with the Exp 3 genome for cleaner comparison. Otherwise run with the Exp 2 genome.*

---

## Hypothesis

The wide-bell attractor **partially persists** but with measurable shifts:

- `end_x` may relax slightly from 0.34 — without payload mass to accelerate, the optimal bell may be slightly narrower
- Timing genes may shift — without the payload's sink rate to overcome, the jellyfish needs less total impulse, potentially favouring shorter contractions or lower frequency
- `t_base` (thickness at payload connection) may decrease — this was likely stiffened by payload load; without payload it should thin
- Fitness values will be higher than Exp 2 because the denominator (effective muscle) stays the same but displacement is no longer fighting payload gravity

---

## What "payload effect" means concretely

The payload sink contributes a displacement handicap of −0.089 per 3-cycle run (measured from `helpers/payload_sink.py`). The jellyfish must generate > +0.089 net upward displacement just to break even. In Exp 2, the best displacement was ~0.61–0.73 units — the payload penalised this by approximately 0.09/7.5 cycles ≈ 0.012 units/cycle × 7.5 = ~0.09 units total. A small but real effect.

More important than displacement is the **structural coupling**: the bell anchor point is the payload attachment. The payload's near-rigid stiffness (E=2×10⁵, 286× jelly) creates a hard boundary condition at the bell manubrium. Without it, the bell's central region is free — it may buckle, narrow, or take a qualitatively different shape.

---

## Analysis plan

1. **Attractor comparison**: overlay final-generation best genomes from Exp 2 and Exp 4 — which genes shift, which are locked?
2. **Fitness ceiling**: does removing the payload increase peak fitness? By how much?
3. **Timing shift**: does jet-mode (high refractory) still win without payload to lift?
4. **Morphology render**: render Exp 4 best genome *with* payload reinserted — does it still function, or does the payload destabilise a shape optimised without it?
5. **Displacement decomposition**: with no payload gravity to fight, how much of Exp 2's displacement was "propulsion" vs "payload compensation"?

---

## Run commands

```bash
# Requires: --no-payload flag wired into evolve.py (or config flag)
# Currently fill_tank accepts with_payload=False — needs evolve.py CLI flag
cd /root/jellyfih
mkdir -p logs

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 JELLY_INSTANCES=32 JELLY_GRID_Y=256 JELLY_DOMAIN_H=2.0 \
  uv run python evolve.py --gens 50 --seed 42   --run-id exp4_s42   --no-payload > logs/exp4_s42.log  2>&1 &
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 JELLY_INSTANCES=32 JELLY_GRID_Y=256 JELLY_DOMAIN_H=2.0 \
  uv run python evolve.py --gens 50 --seed 137  --run-id exp4_s137  --no-payload > logs/exp4_s137.log 2>&1 &
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2 JELLY_INSTANCES=32 JELLY_GRID_Y=256 JELLY_DOMAIN_H=2.0 \
  uv run python evolve.py --gens 50 --seed 999  --run-id exp4_s999  --no-payload > logs/exp4_s999.log 2>&1 &
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=3 JELLY_INSTANCES=32 JELLY_GRID_Y=256 JELLY_DOMAIN_H=2.0 \
  uv run python evolve.py --gens 50 --seed 2024 --run-id exp4_s2024 --no-payload > logs/exp4_s2024.log 2>&1 &
```

*Prerequisite: add `--no-payload` flag to `evolve.py`.*

---

## Success criteria

| Criterion | Threshold | Interpretation |
|-----------|-----------|----------------|
| Attractor preserved | genome correlation Exp2↔Exp4 > 0.85 | Payload has minimal effect on optimal shape |
| Attractor shifted | genome correlation Exp2↔Exp4 < 0.6 | Payload fundamentally shapes the design |
| Fitness increase | Exp 4 best > Exp 2 best × 1.1 | Payload was a meaningful performance penalty |
| t_base reduction | Δt_base > 0.01 | Bell base no longer stiffened by payload load |

---

## Failure modes

- **Morphology collapse**: without payload anchor, bell may fold inward or generate degenerate shapes. May need validity check on bell symmetry.
- **Trivial swimmer**: payloadless bell may evolve to an arbitrarily thin membrane with huge muscle fraction — ensure muscle_count < 200 invalidity check still applies.

---

## Results

**Date:** 2026-03-25
**Hardware:** 4× RTX 3080 Ti, seed 999
**Status:** 🔖 Incomplete — 2/50 gens (checkpoint saved, GPU reassigned mid-run)

### Run history

Experiment 4 had a troubled run:
1. **First attempt (GPU 2, 3070 cluster):** crashed gen 0 — bug: `run_batch_headless` returned `valid=0` for all instances when payload count=0 (body CoM y=0 failed the `y > 0.01` gate). Fixed by adding `compute_body_stats()` fallback kernel.
2. **Second attempt (GPU 2, 3080 Ti):** cancelled before launch — GPU 2 had a broken cooling fan; banned from use.
3. **Third attempt (GPU 3, queued after Exp 5):** launched successfully. Gen 0 anomalously slow (2004s) — JIT compilation + possible brief GPU contention during Exp 5 transition. Gen 1-2 nominal (~760s/gen). Killed at gen 2 to avoid leaving 2 GPUs idle while waiting for full 50-gen completion.

### Partial results (2 gens)

| Metric | Gen 0 | Gen 1 | Gen 2 |
|--------|-------|-------|-------|
| Best displacement (body CoM) | +0.548 | +0.843 | +0.878 |
| Avg displacement | +0.069 | +0.246 | +0.285 |
| Invalid | 12/32 | 6/32 | 3/32 |
| Sim/gen | 2004s* | 759s | 761s |

*Gen 0 slow: JIT + possible contention

### Observations
- Body CoM tracking fix is working — invalids decreasing normally (12→6→3)
- Gen 2 best displacement +0.878 is competitive with Exp 3's early gens — no obvious penalty from missing payload
- Checkpoint `checkpoint_exp4_s999.pkl` saved at gen 2 for future resume

### Next steps
Resume with: `uv run python evolve.py --gens 50 --seed 999 --run-id exp4_s999 --tall --no-payload`
(CMA-ES will auto-resume from checkpoint)
