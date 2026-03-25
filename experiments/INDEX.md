# Jellyfih Experiment Index

Evolutionary optimisation of soft-robotic jellyfish morphologies for upward payload transport.
MPM simulation (Taichi), CMA-ES, 2D.

---

## Experiments

| # | File | Status | Key question | Best fitness |
|---|------|--------|-------------|-------------|
| 1 | [EXPERIMENT_1.md](EXPERIMENT_1.md) | ✅ Complete | What morphology emerges from shape-only search? | 0.536 (efficiency) |
| 2 | [EXPERIMENT_2.md](EXPERIMENT_2.md) | ✅ Complete (19/50 gens) | Does evolved timing unlock better performance? | 1.179 (efficiency) |
| 3 | [EXPERIMENT_3.md](EXPERIMENT_3.md) | ✅ Complete (50/50 gens, s42) | Raw displacement + frequency as gene | **0.838** (displacement) |
| 4 | [EXPERIMENT_4.md](EXPERIMENT_4.md) | 🔖 Checkpoint (2/50 gens) | Does payload shape the morphology attractor? | — (resume needed) |
| 5 | [EXPERIMENT_5.md](EXPERIMENT_5.md) | ✅ Complete (50/50 gens, s42) | Does axisymmetric physics change the optimal morphology? | **1.274** disp (axisym) |
| 6 | [EXPERIMENT_6.md](EXPERIMENT_6.md) | ✅ Complete (50/50 gens, s137) | Control: new genome + efficiency fitness | **1.327** (efficiency) 🏆 |

---

## Genome evolution

| Exp | Genes | Key change |
|-----|-------|------------|
| 1 | 9D | Shape (6) + thickness (3), fixed timing |
| 2 | 11D | + contraction_frac + refractory_frac |
| 3 | 11D | relaxation removed → **freq_mult** replaces refractory gene |
| 4 | 11D | Same as Exp 3, `with_payload=False` |
| 5 | 11D | Same as Exp 3, **axisymmetric MPM** (`JELLY_AXISYM=1`) |
| 6 | 11D | Same as Exp 3 genome, **efficiency fitness** (Exp 2 formula) — control |

---

## Default configuration (as of Exp 2+)

| Parameter | Value |
|-----------|-------|
| Population λ | 32 |
| Tank | 128×256 tall (JELLY_GRID_Y=256, JELLY_DOMAIN_H=2.0) |
| Steps/eval | 150,000 (7.5s at dt=2e-5) |
| Seeds (standard) | 42, 137, 999, 2024 |
| Hardware target | 4× RTX 3080, one seed per GPU |
| Time/gen (tall tank, λ=32) | ~480s |
| Time total (50 gens) | ~6.7 hrs |

---

## Key findings to date

- **Morphology attractor**: wide flat-tipped bell, `end_x` ≈ 0.34, `t_base` ≈ 0.077, robust across all seeds and experiments — **physics-determined, not fitness-determined** (Exp 3 confirmed: same attractor with raw displacement fitness)
- **Jet mode**: evolution converges to high refractory (~65–75%), minimal relaxation (5% floor), long coast (Exp 2). With freq_mult encoding (Exp 3/6): contraction_frac locks, not refractory.
- **Relaxation gene vestigial**: removed in Exp 3+ genome redesign
- **Cost of transport trade-off**: demonstrated explicitly in Exp 2 (s2024 higher displacement, lower fitness)
- **Ceiling exploit**: resolved by tall tank (128×256); Exps 3/5/6 unconstrained
- **Cup bells not found**: relaxing end_y upper bound to +0.10 did not unlock cup morphologies (Exp 2); end_y locks at −0.128 to −0.254 across experiments
- **freq_mult diverges by fitness function** (Exp 3 vs 6): displacement fitness → contraction_frac presses upper (0.55), freq_mult ≈ 0.90; efficiency fitness → freq_mult presses lower bound (0.50 Hz), short contractions (0.29). Lower bound should be extended.
- **New efficiency record: 1.327** (Exp 6, s137) — 12.5% above Exp 2's 1.179; tall tank + freq_mult encoding both contributed
- **Axisym finds different attractor** (Exp 5): cp2_y sign flips to positive, end_x unlocked, thinner base — axisym physics genuinely changes the optimal bell shape. Landscape much broader (sigma=0.41 at gen 49, not converged)
- **Exp 4 (no-payload)**: validity bug fixed; partial run saved at checkpoint gen 2 — resume pending

---

## Code commits by experiment

| Exp | Commit | Description |
|-----|--------|-------------|
| 1 | `3c5ad12` | Rendering overhaul, morphology fixes, physics corrections |
| 2 | `39a3cbe` | Add Exp 2 genome: timing genes, cup bells, activity-weighted fitness |
| 2 (tall tank support) | `8b74f17` | add evolve compatibility for tall tank |
| current HEAD | `0d62bd8` | bugfix |
| 3/4/6 | `7e913d4` | Add --no-payload flag |
| 3/4/6 | `e65a202` | Add --fitness flag (displacement mode) |
| 3/4/6 | `74c7b81` | Per-instance frequency field (instance_freq) |
| 3/4/6 | `3faf72b` | Genome redesign: refractory_frac → freq_mult |
| 5 | `57d7081` | Axisymmetric MPM toggle (axisym-mpm branch) |

---

## Other documentation

| File | Location | Contents |
|------|----------|---------|
| EXPERIMENT_LOG.md | storage/ | Original combined log (Exp 1 + 2 planning notes) |
| DESIGN_CRITIQUE.md | storage/ | Physics model limitations and design decisions |
| CLOUD_TEST_PLAN.md | storage/ | vast.ai deployment runbook |
| PROBLEMS.md | storage/ | Known issues and workarounds |
| SESSION_2026-03-24.md | storage/ | Session notes from initial cloud runs |
