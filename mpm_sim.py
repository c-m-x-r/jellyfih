import os
import taichi as ti
import numpy as np
import math

# --- ENGINE CONFIGURATION ---
ti.set_logging_level(ti.ERROR)
ti.init(arch=ti.cuda)  # CUDA required

# Simulation Constants
n_instances = int(os.environ.get('JELLY_INSTANCES', '16'))
quality = 1  # 1=low-res (128 grid), 2=high-res (256 grid)
n_particles = int(os.environ.get('JELLY_PARTICLES', '80000'))
n_grid = 128 * quality
n_grid_y = int(os.environ.get('JELLY_GRID_Y', str(n_grid)))  # rows (y-axis); default = square
domain_height = float(os.environ.get('JELLY_DOMAIN_H', '1.0'))  # physical domain height
axisym = os.environ.get('JELLY_AXISYM', '0') == '1'  # axisymmetric (r,z) MPM — off by default
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 2e-5 / quality
p_vol = (dx * 0.5) ** 2
p_rho = 1
p_mass = p_vol * p_rho

# Material Constants
E_base = 0.1e4
nu_base = 0.2
mu_0_base = E_base / (2 * (1 + nu_base))
lambda_0_base = E_base * nu_base / ((1 + nu_base) * (1 - 2 * nu_base))

# Jelly/Muscle Properties (Soft)
E_jelly = 0.7e3  
nu_jelly = 0.3 
mu_jelly = E_jelly / (2 * (1 + nu_jelly))
lambda_jelly = E_jelly * nu_jelly / ((1 + nu_jelly) * (1 - 2 * nu_jelly))

# Payload Properties
# CFL check: dt * c_p / dx < 1, where c_p = sqrt((lambda+2mu)/rho)
# With E=2e5, nu=0.2, density_mult=2.5: c_p ≈ 298, dt*c_p/dx ≈ 0.76  ✓
# Higher density_mult lowers c_p, giving more headroom (heavier = safer CFL).
# Physical scale: 1 normalized unit ≈ 48 cm (Aurelia ~25 cm bell).
# Density mapping (water = 1.0 ≡ 1000 kg/m³):
#   1.9 = PCB/FR4,  2.5 = LiPo battery,  2.7 = aluminium,  7.8 = steel
E_payload = 2.0e5
nu_payload = 0.2
mu_payload = E_payload / (2 * (1 + nu_payload))
lambda_payload = E_payload * nu_payload / ((1 + nu_payload) * (1 - 2 * nu_payload))

water_lambda = 100000.0
gravity = 10.0

# Actuation (Pulsed Active Stress)
actuation_freq = 1  # Hz
actuation_strength = 500.0

# Actuation waveform phases (fraction of cycle)
ACT_CONTRACTION_END = 0.2   # 20% contraction (raised cosine ramp up)
ACT_RELAXATION_END  = 0.6   # 40% relaxation  (raised cosine ramp down)
                             # 40% refractory  (zero activation, bell settles)

# --- RENDERING PALETTES ---
# Three rendering modes are supported:
#   'abyss':  Dark background, HDR accumulation with Reinhard tone mapping.
#             Water particles are velocity-direction colourised (flow & vortex streets visible).
#             instance_hue controls jelly hue; instance_muscle_hue controls muscle hue.
#             Optional vorticity overlay: call render_vorticity_overlay() before tone_map_and_encode().
#   'web':    White background, flat material colours matching the web UI frontend.
#             Use clear_frame_buffer_white() + render_flat_pass() per material in WEB_PALETTE.
#   'custom': Like abyss, but with manually set instance_hue[m] / instance_muscle_hue[m] per instance.
#             For two contrasting colours set e.g. hue1 and (hue1 + 0.5) % 1.0 (complementary pair).
WEB_PALETTE = [
    (0, 0.933, 0.949, 0.957),   # Water:   rgb(238, 242, 244) — light blue-gray
    (1, 0.678, 0.882, 0.875),   # Jelly:   rgb(173, 225, 223) — teal
    (3, 0.976, 0.757, 0.765),   # Muscle:  rgb(249, 193, 195) — coral
    (2, 0.961, 0.824, 0.576),   # Payload: rgb(245, 210, 147) — orange
]

# Rendering
video_res = 1024
grid_side = int(math.ceil(math.sqrt(n_instances)))
res_sub = video_res // grid_side
visual_glow = ti.field(dtype=float, shape=(n_instances, n_particles))

# --- GPU MEMORY ALLOCATION ---
print(f"Allocating {n_instances} instances with {n_particles} particles each..."
      + (" [AXISYM mode: JELLY_AXISYM=1]" if axisym else ""))
x = ti.Vector.field(2, dtype=float, shape=(n_instances, n_particles))
v = ti.Vector.field(2, dtype=float, shape=(n_instances, n_particles))
C = ti.Matrix.field(2, 2, dtype=float, shape=(n_instances, n_particles))
F = ti.Matrix.field(2, 2, dtype=float, shape=(n_instances, n_particles))
material = ti.field(dtype=int, shape=(n_instances, n_particles))
Jp = ti.field(dtype=float, shape=(n_instances, n_particles))
grid_v = ti.Vector.field(2, dtype=float, shape=(n_instances, n_grid, n_grid_y))
grid_m = ti.field(dtype=float, shape=(n_instances, n_grid, n_grid_y))

sim_time = ti.field(dtype=float, shape=())
frame_buffer = ti.field(dtype=float, shape=(video_res, video_res, 3))

# Water rendering mode:
#   0 = ghost (near-invisible, clean canvas for vorticity overlay)
#   1 = plain blue, speed-brightness
#   2 = rainbow angle-colourised
water_angle_color = ti.field(dtype=int, shape=())

# Vorticity scratch buffer: normalised curl per grid cell, in (-1, 1)
# Filled by compute_vorticity_grid(); consumed by render_vorticity_rdbu / render_vorticity_hueshift
vort_grid = ti.field(dtype=float, shape=(n_instances, n_grid, n_grid_y))

# Fitness evaluation buffer: [sum_y, count, sum_x] per instance
fitness_buffer = ti.field(dtype=float, shape=(n_instances, 3))

# Per-particle fiber direction for tangent-aligned muscle actuation
fiber_dir = ti.Vector.field(2, dtype=float, shape=(n_instances, n_particles))

# Per-instance actuation strength (allows different strengths per instance for tuning)
# Defaults to actuation_strength; override per-instance for parameter sweeps
instance_actuation = ti.field(dtype=float, shape=(n_instances,))

# Per-instance pulse timing (genome gene 9)
# contraction_frac: fraction of period spent contracting (default 0.20)
# refractory is implicit: 1 - contraction_frac (2-phase waveform; no relaxation ramp)
instance_act_contraction = ti.field(dtype=float, shape=(n_instances,))

# Per-instance frequency multiplier (gene 10 in Experiment 3+; default 1.0 = no change)
# Scales actuation frequency relative to the global actuation_freq constant.
instance_freq = ti.field(dtype=float, shape=(n_instances,))

# Axisymmetric MPM fields (always allocated; only written/read when axisym=True via JELLY_AXISYM=1)
# r_ref:           reference radial position at spawn, for hoop-stretch calculation
# grid_stress_rr:  per-node mass-weighted Kirchhoff σ_rr accumulation (P2G scratch)
# grid_stress_hoop: per-node mass-weighted Kirchhoff σ_θθ accumulation (P2G scratch)
r_ref            = ti.field(dtype=float, shape=(n_instances, n_particles))
grid_stress_rr   = ti.field(dtype=float, shape=(n_instances, n_grid, n_grid_y))
grid_stress_hoop = ti.field(dtype=float, shape=(n_instances, n_grid, n_grid_y))

# Per-instance rendering hue (default 0.55 = blue-cyan, matching original look)
instance_hue = ti.field(dtype=float, shape=(n_instances,))
# Per-instance muscle hue (default offset by 0.15 from jelly hue for contrast)
instance_muscle_hue = ti.field(dtype=float, shape=(n_instances,))
# Per-instance payload density multiplier relative to water (1.0 ≡ 1000 kg/m³).
# Default 2.5 ≈ LiPo battery / mixed electronics assembly.
# CFL remains satisfied up to ~10× water at E=2e5.
instance_payload_density = ti.field(dtype=float, shape=(n_instances,))
for _i in range(n_instances):
    instance_hue[_i] = 0.55
    instance_muscle_hue[_i] = (0.55 + 0.15) % 1.0
    instance_actuation[_i] = actuation_strength
    instance_payload_density[_i] = 2.5
    instance_act_contraction[_i] = ACT_CONTRACTION_END        # 0.20
    instance_freq[_i] = 1.0

# --- PHYSICS KERNELS ---

@ti.kernel
def substep():
    """Main Physics Step (P2G -> Grid -> G2P)."""
    # 0. Update Time & Pulse
    current_time = sim_time[None]

    # 1. Reset Grid
    for m, i, j in grid_m:
        grid_v[m, i, j] = [0, 0]
        grid_m[m, i, j] = 0
        if ti.static(axisym):
            grid_stress_rr[m, i, j] = 0.0
            grid_stress_hoop[m, i, j] = 0.0

    # 2. P2G
    for m, p in x:
        if material[m, p] < 0: continue

        # Per-instance phase: frequency multiplied per-instance so each jellyfish
        # can oscillate at a different rate (gene 10 in Exp 3+).
        inst_period = 1.0 / (actuation_freq * instance_freq[m])
        phase = (current_time % inst_period) / inst_period

        # 2-phase waveform: half-cosine arch during contraction, then zero (refractory)
        c_end = instance_act_contraction[m]
        activation = 0.0
        if phase < c_end:
            activation = 0.5 * (1.0 - ti.cos(phase / c_end * 3.14159265))
        # else: refractory — activation stays 0 (implicit: 1 - contraction_frac)

        base = (x[m, p] * inv_dx - 0.5).cast(int)
        if base[0] < 0 or base[1] < 0 or base[0] >= n_grid - 2 or base[1] >= n_grid_y - 2: continue
        
        fx = x[m, p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        
        F[m, p] = (ti.Matrix.identity(float, 2) + dt * C[m, p]) @ F[m, p]

        # Material Logic
        mu, la = mu_0_base, lambda_0_base
        current_mass = p_mass  # Base mass

        if material[m, p] == 1 or material[m, p] == 3:  # Jelly or Muscle
            mu, la = mu_jelly, lambda_jelly
        
        elif material[m, p] == 2:  # Payload
            mu, la = mu_payload, lambda_payload
            current_mass *= instance_payload_density[m]
            
        elif material[m, p] == 0:  # Water
            mu, la = 0.0, water_lambda

        # SVD & Plasticity
        U, sig, V = ti.svd(F[m, p])
        for d in ti.static(range(2)):
            sig[d, d] = ti.max(sig[d, d], 1e-6)

        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if material[m, p] == 2:  # Payload Plasticity
                # Strict clamping prevents "spring winding" explosion
                new_sig = ti.min(ti.max(sig[d, d], 1 - 5e-3), 1 + 5e-3)
            Jp[m, p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig

        if material[m, p] == 0:
            F[m, p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        elif material[m, p] == 2:
            F[m, p] = U @ sig @ V.transpose()
            
        # Stress Calculation
        stress = 2 * mu * (F[m, p] - U @ V.transpose()) @ F[m, p].transpose() + \
                 ti.Matrix.identity(float, 2) * la * J * (J - 1)
        
        # Active Muscle Stress (tangent-aligned: contracts along bell wall, not isotropically)
        if material[m, p] == 3:
            contractile_pressure = instance_actuation[m] * activation
            fd = fiber_dir[m, p]
            stress += fd.outer_product(fd) * contractile_pressure * J

        # Axisymmetric hoop-stress correction: compute before scaling.
        # kirch_rr_val  = Kirchhoff σ_rr (accumulated on grid for hoop force term)
        # kirch_hoop_val = Kirchhoff σ_θθ from circumferential stretch λ_θ = r / r_ref
        # For water (mu=0): both equal la*J*(J-1), so correction vanishes automatically.
        # For solid: linearised hoop: 2μ(λ_θ-1) + λJ(J-1).
        kirch_rr_val = 0.0
        kirch_hoop_val = 0.0
        r_p_axisym = 0.0
        if ti.static(axisym):
            kirch_rr_val = stress[0, 0]
            r_p_axisym = x[m, p][0]
            r_ref_p = ti.max(r_ref[m, p], dx * 0.1)
            lambda_theta = ti.max(r_p_axisym, dx * 0.05) / r_ref_p
            kirch_hoop_val = 2.0 * mu * (lambda_theta - 1.0) + la * J * (J - 1.0)

        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress

        # Affine momentum transfer (APIC)
        # Uses current_mass to properly weight the heavy payload
        affine = stress + current_mass * C[m, p]

        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            if ti.static(axisym):
                # Annular mass weighting: each particle represents a ring of radius r_p.
                # Scale momentum and mass by r_p so grid normalization gives
                # correctly r-weighted velocity averages.
                grid_v[m, base + offset] += weight * (current_mass * r_p_axisym * v[m, p] + affine @ dpos)
                grid_m[m, base + offset] += weight * current_mass * r_p_axisym
                # Accumulate mass-weighted Kirchhoff stresses for hoop correction in grid update
                grid_stress_rr[m, base + offset]   += weight * current_mass * kirch_rr_val
                grid_stress_hoop[m, base + offset] += weight * current_mass * kirch_hoop_val
            else:
                grid_v[m, base + offset] += weight * (current_mass * v[m, p] + affine @ dpos)
                grid_m[m, base + offset] += weight * current_mass

    # 3. Grid Operations
    for m, i, j in grid_m:
        if grid_m[m, i, j] > 0:
            grid_v[m, i, j] /= grid_m[m, i, j]
            
            # STABILIZATION: Global Damping
            # Bleeds off excess energy from numerical errors (1% drag)
            #grid_v[m, i, j] *= 0.99998
            
            # Boundary Damping (all four sides)
            damp_cells_x = n_grid // 20
            damp_cells_y = n_grid_y // 20
            damp = 1.0
            if i < damp_cells_x: damp *= 0.95 + 0.05 * i / damp_cells_x
            if i > n_grid - damp_cells_x: damp *= 0.95 + 0.05 * (n_grid - i) / damp_cells_x
            if j < damp_cells_y: damp *= 0.95 + 0.05 * j / damp_cells_y
            if j > n_grid_y - damp_cells_y: damp *= 0.95 + 0.05 * (n_grid_y - j) / damp_cells_y
            grid_v[m, i, j] *= damp

            # Hoop stress geometric correction (axisym only).
            # The r-momentum equation in cylindrical coords has an extra (σ_rr - σ_θθ)/r term.
            # Apply it as a post-normalisation velocity correction.
            if ti.static(axisym):
                r_node = (i + 0.5) * dx  # cell-centre radius; avoids exact i=0 division
                if r_node > dx:          # skip axis cell (i=0) — correction handled by BC below
                    hoop_diff = (grid_stress_rr[m, i, j] - grid_stress_hoop[m, i, j]) / grid_m[m, i, j]
                    grid_v[m, i, j][0] += dt * hoop_diff / r_node

            # Wall Collisions / Axis BC
            if ti.static(axisym):
                # Axis of symmetry at i=0: enforce zero radial velocity
                if i == 0: grid_v[m, i, j][0] = 0.0
            else:
                if i < 3 and grid_v[m, i, j][0] < 0: grid_v[m, i, j][0] = 0
            if i > n_grid - 3 and grid_v[m, i, j][0] > 0: grid_v[m, i, j][0] = 0
            if j < 3 and grid_v[m, i, j][1] < 0: grid_v[m, i, j][1] = 0
            if j > n_grid_y - 3 and grid_v[m, i, j][1] > 0: grid_v[m, i, j][1] = 0

    # 4. G2P
    for m, p in x:
        if material[m, p] < 0: continue

        base = (x[m, p] * inv_dx - 0.5).cast(int)
        if base[0] >= 0 and base[1] >= 0 and base[0] < n_grid - 2 and base[1] < n_grid_y - 2:
            fx = x[m, p] * inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(float, 2)
            new_C = ti.Matrix.zero(float, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):
                dpos = ti.Vector([i, j]).cast(float) - fx
                g_v = grid_v[m, base + ti.Vector([i, j])]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
            v[m, p], C[m, p] = new_v, new_C

        v[m, p][1] -= dt * gravity  # uniform gravity for all active particles
        x[m, p] += dt * v[m, p]
        
        x[m, p][0] = ti.max(ti.min(x[m, p][0], 0.999), 0.001)
        x[m, p][1] = ti.max(ti.min(x[m, p][1], domain_height - 0.001), 0.001)
    
    #5. Glowing Effect?
    for m, p in x:
        # Slow decay of the visual effect
        visual_glow[m, p] = ti.max(visual_glow[m, p] * 0.9, ti.abs(1.0 - Jp[m, p]))

    sim_time[None] += dt


##Rendering Pipeline

@ti.func
def hsv2rgb(h: float, s: float, v: float) -> ti.types.vector(3, float):
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r, g, b = 0.0, 0.0, 0.0
    if i % 6 == 0: r, g, b = v, t, p
    elif i % 6 == 1: r, g, b = q, v, p
    elif i % 6 == 2: r, g, b = p, v, t
    elif i % 6 == 3: r, g, b = p, q, v
    elif i % 6 == 4: r, g, b = t, p, v
    elif i % 6 == 5: r, g, b = v, p, q
    return ti.Vector([r, g, b])

@ti.kernel
def clear_frame_buffer():
    for i, j, c in frame_buffer:
        frame_buffer[i, j, c] = 0.0  # Clear to Black

@ti.kernel
def tone_map_and_encode():
    """Converts HDR accumulator buffer to sRGB for display."""
    exposure = 2.5
    for i, j in ti.ndrange(video_res, video_res):
        hdr = ti.Vector([frame_buffer[i, j, 0], frame_buffer[i, j, 1], frame_buffer[i, j, 2]])
        
        # Reinhard Tone Mapping
        mapped = (hdr * exposure) / (hdr * exposure + 1.0)
        mapped = mapped / (mapped + 0.15)
        
        # Gamma Correction
        mapped = ti.pow(mapped, 1.0 / 2.2)
        
        frame_buffer[i, j, 0] = mapped[0]
        frame_buffer[i, j, 1] = mapped[1]
        frame_buffer[i, j, 2] = mapped[2]

@ti.kernel
def render_frame_abyss(p_res_sub: int, p_grid_side: int, radius: float):
    """
    'Deep Sea Abyss' Renderer with Grid-Based Light Blue Gradient.
    """
    # Re-calculate phase for visual sync (per-instance to match substep())
    current_time_r = sim_time[None]

    grid_norm_factor = float(p_grid_side - 1) if p_grid_side > 1 else 1.0

    for m, p in x:
        pos = x[m, p]
        mat = material[m, p]
        if mat < 0 or pos[0] < 0: continue

        # Per-instance activation for muscle colour sync (matches 2-phase substep() waveform)
        inst_period_r = 1.0 / (actuation_freq * instance_freq[m])
        phase = (current_time_r % inst_period_r) / inst_period_r
        c_end_r = instance_act_contraction[m]
        activation = 0.0
        if phase < c_end_r:
            activation = 0.5 * (1.0 - ti.cos(phase / c_end_r * 3.14159265))

        # --- LIGHT CALCULATION ---
        vel = v[m, p].norm()
        stress = 1.0 - Jp[m, p]

        color = ti.Vector([0.0, 0.0, 0.0])
        intensity = 0.0

        row_idx, col_idx = m // p_grid_side, m % p_grid_side
        norm_x = float(col_idx) / grid_norm_factor
        norm_y = float(row_idx) / grid_norm_factor

        if mat == 0: # WATER
            wmode = water_angle_color[None]
            if wmode == 0:
                # Ghost: barely-there neutral so vorticity overlay reads cleanly
                if vel > 0.02:
                    color = ti.Vector([0.15, 0.15, 0.18])
                    intensity = 0.008
            elif wmode == 1:
                # Plain: fixed cool hue, speed → brightness
                if vel > 0.02:
                    brightness = ti.min(vel / 2.0, 1.0)
                    color = hsv2rgb(0.60, 0.55, brightness)
                    intensity = 0.025 + 0.075 * brightness
            else:
                # Rainbow: full hue wheel mapped to velocity direction
                if vel > 0.02:
                    brightness = ti.min(vel / 2.0, 1.0)
                    vx_n = v[m, p][0]
                    vy_n = v[m, p][1]
                    flow_angle = ti.atan2(vy_n, vx_n)
                    hue = (flow_angle / (2.0 * 3.14159265) + 0.5) % 1.0
                    color = hsv2rgb(hue, 0.8, brightness)
                    intensity = 0.025 + 0.075 * brightness

        elif mat == 1: # JELLY
            hue = instance_hue[m]
            sat = 0.4 + (norm_y * 0.4)
            base_col = hsv2rgb(hue, sat, 0.9)
            glow = ti.abs(stress) * 2.0 + 0.1
            color = base_col + ti.Vector([glow, glow, glow])
            intensity = 0.05 + (glow * 0.2)

        elif mat == 3: # MUSCLE (Visually syncs with activation; uses separate muscle hue)
            hue = instance_muscle_hue[m]
            color = hsv2rgb(hue, 0.2, 1.0) + ti.Vector([activation, activation, activation])
            intensity = 0.4 + (activation * 0.4)
            
        elif mat == 2: # PAYLOAD
            color = ti.Vector([1.0, 0.2, 0.0]) 
            intensity = 0.8

        # --- SPLATTING ---
        if intensity > 0.001:
            row, col = m // p_grid_side, m % p_grid_side
            center_x = (pos[0] / domain_height + col) * p_res_sub
            center_y = ((domain_height - pos[1]) / domain_height + row) * p_res_sub
            
            draw_r = radius * 1.5
            low_x, high_x = int(center_x - draw_r), int(center_x + draw_r)
            low_y, high_y = int(center_y - draw_r), int(center_y + draw_r)

            for px in range(low_x, high_x + 1):
                for py in range(low_y, high_y + 1):
                    if 0 <= px < video_res and 0 <= py < video_res:
                        dist_sq = (px - center_x)**2 + (py - center_y)**2
                        dist_norm = dist_sq / (draw_r**2)
                        
                        if dist_norm < 1.0:
                            falloff = (1.0 - dist_norm)**4
                            val = color * intensity * falloff
                            frame_buffer[py, px, 0] += val[0]
                            frame_buffer[py, px, 1] += val[1]
                            frame_buffer[py, px, 2] += val[2]
                        

@ti.kernel
def clear_frame_buffer_white():
    for i, j, c in frame_buffer:
        frame_buffer[i, j, c] = 0.98  # Clear to #FAFAFA


@ti.kernel
def render_flat_pass(p_res_sub: int, p_grid_side: int, radius: float,
                     target_mat: int, cr: float, cg: float, cb: float):
    """Flat-color renderer matching the web frontend palette.
    Renders one material at a time for correct layering (water under jelly etc.)."""
    for m, p in x:
        pos = x[m, p]
        mat = material[m, p]
        if mat != target_mat or pos[0] < 0:
            continue

        row, col = m // p_grid_side, m % p_grid_side
        center_x = (pos[0] / domain_height + col) * p_res_sub
        center_y = ((domain_height - pos[1]) / domain_height + row) * p_res_sub

        draw_r = radius
        low_x = int(center_x - draw_r)
        high_x = int(center_x + draw_r)
        low_y = int(center_y - draw_r)
        high_y = int(center_y + draw_r)

        for px in range(low_x, high_x + 1):
            for py in range(low_y, high_y + 1):
                if 0 <= px < video_res and 0 <= py < video_res:
                    dist_sq = (px - center_x)**2 + (py - center_y)**2
                    if dist_sq <= draw_r * draw_r:
                        frame_buffer[py, px, 0] = cr
                        frame_buffer[py, px, 1] = cg
                        frame_buffer[py, px, 2] = cb


@ti.kernel
def render_vorticity_overlay(p_res_sub: int, p_grid_side: int, vort_scale: float):
    """
    Grid-based 2D vorticity overlay. Computes curl(v) = dvx/dy - dvy/dx per cell
    and paints coloured squares additively onto frame_buffer.

    Call AFTER render_frame_abyss() but BEFORE tone_map_and_encode().
    Warm colours (orange/red) = CCW rotation; cool colours (blue) = CW rotation.

    vort_scale: sensitivity multiplier. Recommended starting values:
        0.001  — strong pulses (actuation_strength >= 500)
        0.003  — weak pulses or early-generation morphologies
    """
    for m, i, j in grid_v:
        if i < 1 or i >= n_grid - 1 or j < 1 or j >= n_grid_y - 1:
            continue
        if grid_m[m, i, j] <= 0.0:
            continue
        # Finite-difference curl on the MPM grid
        dvx_dy = (grid_v[m, i, j + 1][0] - grid_v[m, i, j - 1][0]) / (2.0 * dx)
        dvy_dx = (grid_v[m, i + 1, j][1] - grid_v[m, i - 1, j][1]) / (2.0 * dx)
        curl = dvx_dy - dvy_dx
        vort_intensity = ti.min(ti.abs(curl) * vort_scale, 1.0)
        if vort_intensity < 0.01:
            continue
        if curl > 0:
            col = ti.Vector([1.0, 0.4, 0.0]) * vort_intensity   # warm: CCW
        else:
            col = ti.Vector([0.0, 0.4, 1.0]) * vort_intensity   # cool: CW
        row, c = m // p_grid_side, m % p_grid_side
        cell_x = int((float(i) / n_grid / domain_height + c) * p_res_sub)
        cell_y = int(((domain_height - float(j) / n_grid) / domain_height + row) * p_res_sub)
        cell_px = ti.max(1, p_res_sub // n_grid)
        for px in range(cell_x - cell_px // 2, cell_x + cell_px // 2 + 1):
            for py in range(cell_y - cell_px // 2, cell_y + cell_px // 2 + 1):
                if 0 <= px < video_res and 0 <= py < video_res:
                    frame_buffer[py, px, 0] += col[0] * 0.5
                    frame_buffer[py, px, 1] += col[1] * 0.5
                    frame_buffer[py, px, 2] += col[2] * 0.5


@ti.kernel
def compute_vorticity_grid(vort_scale: float):
    """
    Pre-compute normalised vorticity into vort_grid.
    curl = dvx/dy - dvy/dx per cell; tanh-mapped to (-1, 1).
    Call once per frame before render_vorticity_rdbu or render_vorticity_hueshift.
    """
    for m, i, j in vort_grid:
        vort_grid[m, i, j] = 0.0
    for m, i, j in grid_v:
        if i < 1 or i >= n_grid - 1 or j < 1 or j >= n_grid_y - 1:
            continue
        if grid_m[m, i, j] <= 0.0:
            continue
        dvx_dy = (grid_v[m, i, j + 1][0] - grid_v[m, i, j - 1][0]) / (2.0 * dx)
        dvy_dx = (grid_v[m, i + 1, j][1] - grid_v[m, i - 1, j][1]) / (2.0 * dx)
        curl = dvx_dy - dvy_dx
        vort_grid[m, i, j] = ti.tanh(curl * vort_scale)


@ti.kernel
def render_vorticity_rdbu(p_res_sub: int, p_grid_side: int, alpha_max: float):
    """
    Post-tonemapping RdBu overlay.  Call AFTER tone_map_and_encode().

    Maps vort_grid → red (+) / blue (−) and alpha-composites over the LDR frame_buffer.
    alpha = alpha_max * |vort|  so near-zero cells are fully transparent.

    alpha_max: maximum opacity at |vort|=1.  Recommended 0.55–0.70.
    """
    for px, py in ti.ndrange(video_res, video_res):
        # Map pixel back to the instance and grid cell it belongs to
        col  = px * p_grid_side // video_res
        row  = py * p_grid_side // video_res
        m    = row * p_grid_side + col
        if m >= n_instances:
            continue

        # Inverse of: px = (i/n_grid/domain_height + col) * p_res_sub
        #             py = (1 - j/n_grid_y + row) * p_res_sub
        sub_x = (px - col * p_res_sub) / float(p_res_sub)       # = i / n_grid / domain_height
        sub_y = 1.0 - (py - row * p_res_sub) / float(p_res_sub) # = j / n_grid_y

        gi = int(sub_x * n_grid * domain_height)
        gj = int(sub_y * n_grid_y)
        if gi < 0 or gi >= n_grid or gj < 0 or gj >= n_grid_y:
            continue

        vort = vort_grid[m, gi, gj]
        mag  = ti.abs(vort)
        if mag < 0.02:
            continue

        alpha = alpha_max * mag
        # Red (CCW +) blends toward (0.84, 0.10, 0.11); Blue (CW −) toward (0.13, 0.31, 0.80)
        t = 0.5 + 0.5 * ti.math.sign(vort)   # 1.0 if CCW, 0.0 if CW
        col_r = t * 0.84 + (1.0 - t) * 0.13
        col_g = t * 0.10 + (1.0 - t) * 0.31
        col_b = t * 0.11 + (1.0 - t) * 0.80

        frame_buffer[py, px, 0] = (1.0 - alpha) * frame_buffer[py, px, 0] + alpha * col_r
        frame_buffer[py, px, 1] = (1.0 - alpha) * frame_buffer[py, px, 1] + alpha * col_g
        frame_buffer[py, px, 2] = (1.0 - alpha) * frame_buffer[py, px, 2] + alpha * col_b


@ti.kernel
def render_vorticity_hueshift(p_res_sub: int, p_grid_side: int, strength: float):
    """
    Post-tonemapping hue-shift overlay.  Call AFTER tone_map_and_encode().

    Blends each pixel toward red (CCW) or blue (CW) proportional to |vort|,
    leaving the underlying particle colour intact in irrotational regions.
    Subtler than RdBu: the jellyfish and water retain their identity while
    vortex structures become warm/cool tinted.

    strength: blend fraction at |vort|=1.  Recommended 0.45–0.65.
    """
    for px, py in ti.ndrange(video_res, video_res):
        col  = px * p_grid_side // video_res
        row  = py * p_grid_side // video_res
        m    = row * p_grid_side + col
        if m >= n_instances:
            continue

        sub_x = (px - col * p_res_sub) / float(p_res_sub)
        sub_y = 1.0 - (py - row * p_res_sub) / float(p_res_sub)

        gi = int(sub_x * n_grid * domain_height)
        gj = int(sub_y * n_grid_y)
        if gi < 0 or gi >= n_grid or gj < 0 or gj >= n_grid_y:
            continue

        vort = vort_grid[m, gi, gj]
        mag  = ti.abs(vort)
        if mag < 0.02:
            continue

        alpha = strength * mag
        # Warm red (CCW +): (0.90, 0.18, 0.15); cool blue (CW −): (0.15, 0.35, 0.90)
        t = 0.5 + 0.5 * ti.math.sign(vort)
        col_r = t * 0.90 + (1.0 - t) * 0.15
        col_g = t * 0.18 + (1.0 - t) * 0.35
        col_b = t * 0.15 + (1.0 - t) * 0.90

        frame_buffer[py, px, 0] = (1.0 - alpha) * frame_buffer[py, px, 0] + alpha * col_r
        frame_buffer[py, px, 1] = (1.0 - alpha) * frame_buffer[py, px, 1] + alpha * col_g
        frame_buffer[py, px, 2] = (1.0 - alpha) * frame_buffer[py, px, 2] + alpha * col_b


@ti.kernel
def render_vorticity_white(p_res_sub: int, p_grid_side: int, alpha_max: float):
    """
    White-background vorticity-only render.  Clears frame_buffer to white then
    paints RdBu cells from vort_grid (must be pre-computed).  No particles rendered.
    Call standalone — does not require tone_map_and_encode().
    """
    # Clear to white
    for i, j, c in frame_buffer:
        frame_buffer[i, j, c] = 1.0

    # Paint vorticity
    for px, py in ti.ndrange(video_res, video_res):
        col  = px * p_grid_side // video_res
        row  = py * p_grid_side // video_res
        m    = row * p_grid_side + col
        if m >= n_instances:
            continue

        sub_x = (px - col * p_res_sub) / float(p_res_sub)
        sub_y = 1.0 - (py - row * p_res_sub) / float(p_res_sub)

        gi = int(sub_x * n_grid * domain_height)
        gj = int(sub_y * n_grid_y)
        if gi < 0 or gi >= n_grid or gj < 0 or gj >= n_grid_y:
            continue

        vort = vort_grid[m, gi, gj]
        mag  = ti.abs(vort)
        if mag < 0.02:
            continue

        alpha = alpha_max * mag
        # Red (CCW +): (0.84, 0.10, 0.11); Blue (CW −): (0.13, 0.31, 0.80)
        t     = 0.5 + 0.5 * ti.math.sign(vort)
        col_r = t * 0.84 + (1.0 - t) * 0.13
        col_g = t * 0.10 + (1.0 - t) * 0.31
        col_b = t * 0.11 + (1.0 - t) * 0.80

        # Alpha-composite over white (1,1,1)
        frame_buffer[py, px, 0] = (1.0 - alpha) + alpha * col_r
        frame_buffer[py, px, 1] = (1.0 - alpha) + alpha * col_g
        frame_buffer[py, px, 2] = (1.0 - alpha) + alpha * col_b


@ti.kernel
def _load_particles_kernel(instance: int, pos_np: ti.types.ndarray(),
                           mat_np: ti.types.ndarray(), fiber_np: ti.types.ndarray()):
    """Reset a specific instance with new particle data."""
    for p in range(n_particles):
        x[instance, p] = [pos_np[p, 0], pos_np[p, 1]]
        material[instance, p] = mat_np[p]
        v[instance, p] = [0.0, 0.0]
        F[instance, p] = ti.Matrix.identity(float, 2)
        C[instance, p] = ti.Matrix.zero(float, 2, 2)
        Jp[instance, p] = 1.0
        fiber_dir[instance, p] = [fiber_np[p, 0], fiber_np[p, 1]]
        r_ref[instance, p] = pos_np[p, 0]  # reference radial position for axisym hoop stretch


def load_particles(instance, pos_np, mat_np, fiber_np=None):
    """Reset a specific instance. fiber_np: (n_particles, 2) unit tangent vectors for muscle;
    defaults to (0, 1) for all particles if not provided."""
    if fiber_np is None:
        fiber_np = np.zeros((n_particles, 2), dtype=np.float32)
        fiber_np[:, 1] = 1.0
    _load_particles_kernel(instance, pos_np, mat_np, fiber_np)

# --- FITNESS EVALUATION KERNELS ---

@ti.kernel
def clear_fitness_buffer():
    """Clear fitness buffer. MUST be called before compute_payload_stats."""
    for i, j in fitness_buffer:
        fitness_buffer[i, j] = 0.0

@ti.kernel
def compute_payload_stats():
    """Accumulate payload (Material 2) positions for CoM calculation."""
    for i, p in x:
        if material[i, p] == 2:
            ti.atomic_add(fitness_buffer[i, 0], x[i, p][1])  # Sum Y
            ti.atomic_add(fitness_buffer[i, 1], 1.0)          # Count
            ti.atomic_add(fitness_buffer[i, 2], x[i, p][0])  # Sum X

def get_payload_stats():
    """Compute payload center of mass. Returns (n_instances, 3) array: [com_y, com_x, count]."""
    clear_fitness_buffer()
    compute_payload_stats()
    ti.sync()
    raw = fitness_buffer.to_numpy()
    stats = np.zeros((n_instances, 3))
    for i in range(n_instances):
        count = raw[i, 1]
        if count > 0:
            stats[i, 0] = raw[i, 0] / count  # CoM Y
            stats[i, 1] = raw[i, 2] / count  # CoM X
            stats[i, 2] = count
    return stats

def run_batch_headless(steps):
    """
    Run simulation headlessly for all instances.
    Returns (n_instances, 5): [init_y, init_x, final_y, final_x, valid]
    """
    sim_time[None] = 0.0

    # Capture initial payload positions
    initial = get_payload_stats()

    # Run physics
    for _ in range(steps):
        substep()

    # Capture final payload positions
    final = get_payload_stats()

    # Assemble results
    results = np.zeros((n_instances, 5))
    for i in range(n_instances):
        if initial[i, 2] > 0 and final[i, 2] > 0:
            results[i, 0] = initial[i, 0]  # init CoM Y
            results[i, 1] = initial[i, 1]  # init CoM X
            results[i, 2] = final[i, 0]    # final CoM Y
            results[i, 3] = final[i, 1]    # final CoM X
            # Only invalidate if payload sinks to floor (genuine failure)
            if final[i, 0] < 0.01:
                results[i, 4] = 0.0  # Invalid: payload lost to floor
            else:
                results[i, 4] = 1.0  # Valid
        else:
            results[i, 4] = 0.0  # Invalid: payload lost
    return results

