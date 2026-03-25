# Experiment 5 — Axisymmetric MPM Trial

**Status:** Planned
**Depends on:** `axisym-mpm` branch (not yet merged to main); Exp 3 results for Cartesian comparison baseline
**Hardware:** 1× RTX 3080 (1 seed, shares 4-GPU instance with Exps 3, 4, 6)
**Target:** 1 seed, 50 generations, tall tank (128×256), `JELLY_AXISYM=1`

---

## Motivation

The 2D Cartesian simulation models each jellyfish as an infinite slab — the bell and jet extend without bound in the out-of-plane direction. A real *Aurelia aurita* is radially symmetric, so the correct 2D reduction is axisymmetric (r, z): each simulation point represents an annular ring. This changes three things:

1. **Pressure recovery**: in Cartesian 2D, vortex dipoles advect indefinitely. In axisym, vortex rings decay as energy spreads into the growing ring circumference — a physically correct energy sink that Cartesian 2D lacks.

2. **Thrust scaling**: a ring-shaped jet exhausts radially, and the impulse per stroke scales with ring area (πr²) not width. A narrow bell creates a thin high-velocity ring; a wide bell creates a slow fat ring. This changes the optimal bell radius in a way Cartesian geometry cannot capture.

3. **Volume conservation**: in Cartesian, the bell compresses an infinite-depth water column. In axisym, the compressed volume is finite (inside the bell). The restoring pressure from the enclosed water volume is physically different.

**Primary question**: does the morphology attractor shift when the physics are physically correct? Is the wide flat bell still optimal, or does the correct 3D-equivalent geometry prefer a different shape?

This is a **validation and discovery experiment**. Results should be interpreted alongside the Cartesian Exp 3 as a pair.

---

## Physics changes (axisym branch)

The `axisym-mpm` branch implements axisymmetric MPM via the `JELLY_AXISYM=1` environment toggle. Changes are applied at compile time via `ti.static(axisym)` — zero overhead in Cartesian mode.

| Component | Cartesian (default) | Axisymmetric (`JELLY_AXISYM=1`) |
|-----------|---------------------|---------------------------------|
| P2G mass | `weight × mass` | `weight × mass × r_p` (annular ring weighting) |
| Momentum | `weight × mass × v` | `weight × mass × r_p × v` |
| Grid correction | — | `(σ_rr − σ_θθ)/r` hoop body force |
| Left BC | wall (v_r ≥ 0) | axis: v_r = 0 at i=0 |
| Bell geometry | mirrored about x=0.5 | right half only (r ≥ 0), spawn at r=0 |
| Payload | centred at x=0.5 | right half [0, W/2], CoM at r=W/4 |
| Particle count | ~80K (full bell) | ~40K (half bell, no mirror) |

The hoop stress `σ_θθ` is computed from the circumferential stretch `λ_θ = r / r_ref` (radial position relative to spawn position). For water (μ=0), `σ_rr = σ_θθ` and the correction vanishes automatically.

---

## Genome

Same as Experiment 3 (new 11D encoding with freq_mult):

| Index | Parameter | Bounds | Init |
|-------|-----------|--------|------|
| 0–8 | shape + thickness | same as Exp 3 | same |
| 9 | contraction_frac | [0.05, 0.60] | 0.20 |
| 10 | freq_mult | [0.5, 2.0] | 1.0 |

---

## Fitness Function

**Raw displacement** (same as Exp 3, for direct comparison):

```python
fitness = min(final_y, ceiling) - init_y
```

This lets morphology differences between Exp 3 (Cartesian) and Exp 5 (axisym) be attributed to physics, not fitness function.

---

## Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Genome | 11D (Exp 3 encoding) | same as Exp 3/6 |
| Fitness | **displacement** (raw) | same as Exp 3 |
| Tank | **128×256 tall**, ceiling y=1.93 | same as Exp 3/6 |
| Physics | **JELLY_AXISYM=1** | axisymmetric r-z |
| Particle count | ~40K (right half only) | ~2× faster P2G than Cartesian |
| Population (λ) | 32 | same |
| Generations | 50 | same |
| Steps/eval | 150,000 | same |
| Seed | 42 | single seed (1 GPU available) |
| Branch | **`axisym-mpm`** | not yet merged to main |

*Time estimate: axisym P2G is ~2× faster per particle (half the particles), but hoop correction adds a pass. Net expected time similar to Cartesian tall tank (~480s/gen). 50 gens ≈ 6.7 hrs.*

---

## Expected outcomes

- **Morphology shift possible**: without the slab-geometry over-thrust, the optimal bell may be narrower or deeper — the r² area scaling rewards wide bells differently than the linear width scaling in Cartesian
- **Vortex ring decay**: higher-frequency waveforms may be penalised as vortex rings dissipate faster in axisym — freq_mult may converge lower than Exp 3
- **Displacement lower than Exp 3**: axisym pressure recovery and ring decay are energy sinks not present in Cartesian; absolute displacement values will be lower
- **Axis instability risk**: particles near r=0 have r_p → 0, giving near-zero effective mass in the P2G; the bell tip near the axis may need validity checks

---

## Validation protocol (must pass before results are trusted)

These checks should be run before the full 50-generation run:

1. **Smoke test** (2K steps): `JELLY_AXISYM=1 JELLY_INSTANCES=1 uv run python helpers/view_single.py --aurelia --steps 2000` — kernel must compile and run without NaN
2. **Symmetry test**: run Cartesian with a manually symmetric genome → run axisym on same genome; net upward displacement should be in the same order of magnitude (within 3×)
3. **Sink test**: `JELLY_AXISYM=1 JELLY_INSTANCES=1 uv run python helpers/payload_sink.py` — payload should still sink in axisym; verify sign and rough magnitude
4. **Vortex structure**: render vorticity overlay — expect single-sign vortex rings (CCW behind bell) rather than Cartesian dipole pairs
5. **CFL check**: verify no NaN explosion in first 1K steps at full λ=32

*If validation fails, do not proceed to full run. Document failure mode and halt.*

---

## Analysis plan

1. **Morphology comparison**: overlay final-gen best genomes from Exp 3 (Cartesian) and Exp 5 (axisym) — which genes shift?
2. **Displacement ratio**: Exp 5 / Exp 3 displacement — quantifies the Cartesian over-thrust artefact
3. **Frequency gene**: does axisym favour different freq_mult than Cartesian? If lower, ring decay is penalising high-frequency firing
4. **Wake structure**: compare vorticity videos — does axisym show vortex ring geometry vs Cartesian dipoles?
5. **Condition number**: does the tighter axisym physics collapse the landscape faster or slower?

---

## Run commands

```bash
# Must be on axisym-mpm branch
cd /root/jellyfih
git checkout axisym-mpm
mkdir -p logs

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=3 JELLY_INSTANCES=32 JELLY_GRID_Y=256 JELLY_DOMAIN_H=2.0 JELLY_AXISYM=1 \
  uv run python evolve.py --gens 50 --seed 42 --fitness displacement --run-id exp5_s42 --tall \
  > logs/exp5_s42.log 2>&1 &
```

*Note: other GPUs (0, 1, 2) run Exps 3, 4, 6 from the `main` branch simultaneously.*

---

## Success criteria

| Criterion | Threshold | Interpretation |
|-----------|-----------|----------------|
| No NaN / explosion | 0 crashes in 50 gens | Axisym physics numerically stable |
| Displacement positive | best > 0.1 | Jellyfish swimming upward in axisym domain |
| Morphology measurably different from Exp 3 | any gene Δ > 0.05 | Axisym physics influences optimal shape |
| Vortex ring structure | visible single-sign rings in vorticity video | Correct axisym fluid topology |

---

## Failure modes

- **NaN explosion near axis**: r_p → 0 gives near-zero mass; may need to clamp r_ref and r_p to dx/2 minimum (already partially guarded with `ti.max(r_p, dx * 0.05)`)
- **Hoop correction too large**: if the (σ_rr − σ_θθ)/r force is poorly scaled, it may dominate the dynamics; watch for runaway radial oscillations in early generations
- **Degenerate bell**: without mirror symmetry the bell is now a right-half profile; degenerate shapes that exploit the axis BC (very narrow bells pressed against r=0) may emerge as a spurious fitness mode
- **Performance too slow**: if axisym overhead pushes time/gen > 700s, reduce steps to 100K for this run only

---

## Results

**Date:** 2026-03-25
**Hardware:** 4× RTX 3080 Ti, seed 42, single replicate
**Status:** ✅ Complete (50/50 gens)

| Metric | Value | Notes |
|--------|-------|-------|
| Best displacement | **+1.274** (gen 33) | In axisym (r,z) coordinates — not directly comparable to Cartesian |
| Final-gen best | +1.174 | |
| 90% convergence | gen 2 | Misleading — landscape is broad, not sharply peaked |
| Final sigma | 0.410 | Still exploring — did NOT converge |
| Final cond # | 21.2 | Broader landscape than Cartesian (Exp 3: 62.9) |
| Muscle count (best) | 288 | ~half of Exp 3 (674) — right-half profile only |
| Sim time/gen | ~131s | ~2.8× faster than Cartesian (half particles) |
| Total invalids | 192/1600 (12%) | Acceptable |

### Gene convergence (many NOT locked)

| Gene | Final mean | Locked? | Notes |
|------|-----------|---------|-------|
| cp1_x | +0.126 | ✅ CV=0.15 | |
| cp1_y | −0.051 | ❌ CV=0.18 | Unlocked — bell curvature still explored |
| cp2_x | +0.246 | ✅ CV=0.14 | |
| cp2_y | **+0.114** | ✅ CV=0.10 | **Positive** — opposite sign to Cartesian (−0.19) |
| end_x | +0.266 | ❌ CV=0.15 | Not pressing upper bound (Cartesian did) |
| end_y | −0.254 | ✅ CV=0.09 | Deeper cup than Cartesian (−0.128) |
| t_base | +0.055 | ✅ CV=0.12 | Thinner than Cartesian (0.079) |
| t_mid | +0.050 | ❌ CV=0.18 | Unlocked |
| contraction | +0.289 | ❌ CV=0.19 | Unlocked — near default |
| freq_mult | **+1.143** | ✅ CV=0.15 | Slightly above baseline — opposite to Exp 6 |

### Key findings

- **Different attractor from Cartesian**: cp2_y is positive (+0.114) vs Cartesian's negative (−0.19). end_x not pressing upper bound. The axisym physics genuinely changes the optimal bell geometry. This is the experiment's central result.
- **Not converged at 50 gens**: sigma=0.41, many genes unlocked. The axisym landscape is significantly flatter than Cartesian — more degrees of freedom remain viable. Would benefit from more generations or higher lambda.
- **freq_mult at 1.14** (slightly above 1 Hz): opposite direction from Exp 6 (0.50 Hz) and opposite from Exp 3 (0.90 Hz). The correct cylindrical pressure recovery may make higher-frequency firing slightly more efficient.
- **Thinner bell base (t_base=0.055)** vs Cartesian's 0.077-0.079: axisym geometry changes the structural loading on the bell manubrium.
- **Validation criteria**: no NaN explosions, numerically stable for 50 gens — axisym physics implementation is robust. Formal Cartesian/axisym symmetry test and vortex ring visualisation still pending.

### Success criteria outcome

| Criterion | Result |
|-----------|--------|
| No NaN / explosion | ✅ Stable all 50 gens |
| Displacement positive | ✅ +1.274 |
| Morphology different from Exp 3 | ✅ cp2_y sign flip, end_x unlocked, thinner base |
| Vortex ring structure (visual) | ⏳ Not yet rendered |
