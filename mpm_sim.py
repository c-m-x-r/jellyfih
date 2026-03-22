import taichi as ti
import numpy as np
import math

# --- ENGINE CONFIGURATION ---
ti.set_logging_level(ti.ERROR)
ti.init(arch=ti.cuda)  # CUDA required

# Simulation Constants
n_instances = 16
quality = 1  # 1=low-res (128 grid), 2=high-res (256 grid)
n_particles = 80000
n_grid = 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 5e-5 / quality
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

# Payload Properties (Heavy & Stiff)
# Reduced E from 2e5 to 4e4 to satisfy CFL: c = sqrt(E/rho)
# We will compensate by increasing density (rho) in the kernel
E_payload = 4.0e4 
nu_payload = 0.2
mu_payload = E_payload / (2 * (1 + nu_payload))
lambda_payload = E_payload * nu_payload / ((1 + nu_payload) * (1 - 2 * nu_payload))

water_lambda = 3000.0
water_mu = 50.0   # Water bulk stiffness (not viscosity). Provides pressure-based drag via elastic stiffness.
                  # Higher values slow particle displacement; ≥150 makes water gel-like and unphysical.
grid_drag = 0.0   # Brinkman drag per grid node per step (0 = disabled). Overshoots easily; use water_mu instead.
water_visc = 0.0  # Kinematic viscosity for Laplacian grid diffusion (m²/s in sim units).
                  # CFL limit: water_visc ≤ dx²/(4·dt) = (1/128)²/(4×5e-5) ≈ 0.305.
                  # 0.0 = disabled (no extra kernel launch). Tune after calibrating water_mu.
payload_gravity_factor = 0.80  # Gravity scaling for payload. Net downward = 2.5×0.80 - 1.0 = 1.0× (slight neg. buoyancy).
                                # Physical basis: AUV neutrally buoyant by design; 2.5× inertial mass drives selection.
gravity = 10.0

# Actuation (Pulsed Active Stress)
actuation_freq = 1  # Hz
actuation_strength = 10000.0 

# Rendering
video_res = 1024
grid_side = int(math.ceil(math.sqrt(n_instances)))
res_sub = video_res // grid_side

# --- GPU MEMORY ALLOCATION ---
print(f"Allocating {n_instances} instances with {n_particles} particles each...")
x = ti.Vector.field(2, dtype=float, shape=(n_instances, n_particles))
v = ti.Vector.field(2, dtype=float, shape=(n_instances, n_particles))
C = ti.Matrix.field(2, 2, dtype=float, shape=(n_instances, n_particles))
F = ti.Matrix.field(2, 2, dtype=float, shape=(n_instances, n_particles))
material = ti.field(dtype=int, shape=(n_instances, n_particles))
Jp = ti.field(dtype=float, shape=(n_instances, n_particles))
grid_v = ti.Vector.field(2, dtype=float, shape=(n_instances, n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_instances, n_grid, n_grid))
grid_v_lap = ti.Vector.field(2, dtype=float, shape=(n_instances, n_grid, n_grid))  # Laplacian diffusion double-buffer

sim_time = ti.field(dtype=float, shape=())
frame_buffer = ti.field(dtype=float, shape=(video_res, video_res, 3))

# Fitness evaluation buffer: [sum_y, count, sum_x] per instance
fitness_buffer = ti.field(dtype=float, shape=(n_instances, 3))

# Circumferential fiber directions for anisotropic actuation (Mat 3)
fiber_dir = ti.Vector.field(2, dtype=float, shape=(n_instances, n_particles))
use_anisotropic_actuation = True   # Inward-normal anisotropic actuation (hoop stress projection)

# Per-instance rendering hue (default 0.55 = blue-cyan, matching original look)
instance_hue = ti.field(dtype=float, shape=(n_instances,))
for _i in range(n_instances):
    instance_hue[_i] = 0.55

# --- PHYSICS KERNELS ---

@ti.kernel
def _substep_p2g():
    """Grid reset + Particle-to-Grid scatter."""
    # Compute activation waveform from current sim time
    current_time = sim_time[None]
    period = 1.0 / actuation_freq
    phase = (current_time % period) / period

    # Raised cosine waveform: 20% contraction, 80% relaxation (bio-inspired asymmetry)
    activation = 0.0
    if phase < 0.2:
        activation = 0.5 * (1.0 - ti.cos(phase / 0.2 * 3.14159265))
    elif phase < 1.0:
        activation = 0.5 * (1.0 + ti.cos((phase - 0.2) / 0.8 * 3.14159265))

    # 1. Reset Grid (sequential before P2G within this kernel)
    for m, i, j in grid_m:
        grid_v[m, i, j] = [0, 0]
        grid_m[m, i, j] = 0

    # 2. P2G
    for m, p in x:
        if material[m, p] < 0: continue

        base = (x[m, p] * inv_dx - 0.5).cast(int)
        if base[0] < 0 or base[1] < 0 or base[0] >= n_grid - 2 or base[1] >= n_grid - 2: continue

        fx = x[m, p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        F[m, p] = (ti.Matrix.identity(float, 2) + dt * C[m, p]) @ F[m, p]

        # Material properties
        current_mass = p_mass
        if material[m, p] == 2:
            current_mass *= 2.5

        stress = ti.Matrix.zero(float, 2, 2)

        if material[m, p] == 0:
            # --- Water fast path: skip SVD ---
            # F starts as scalar*I each step (reset below), so U@V^T ≈ I (error O(dt·C) ≈ 5e-3).
            # Stress is isotropic: purely volumetric (water_lambda) + isotropic shear (water_mu).
            J = F[m, p].determinant()
            J = ti.max(J, 1e-6)
            s = ti.sqrt(J)
            stress = ((2.0 * water_mu * (s - 1.0) * s) + (water_lambda * J * (J - 1.0))) * \
                     ti.Matrix.identity(float, 2)
            F[m, p] = ti.Matrix.identity(float, 2) * s   # reset to scalar matrix

        else:
            # --- SVD path: jelly (1), payload (2), muscle (3) ---
            mu = mu_0_base
            la = lambda_0_base
            if material[m, p] == 1 or material[m, p] == 3:
                mu, la = mu_jelly, lambda_jelly
            elif material[m, p] == 2:
                mu, la = mu_payload, lambda_payload

            U, sig, V = ti.svd(F[m, p])
            for d in ti.static(range(2)):
                sig[d, d] = ti.max(sig[d, d], 1e-6)

            J = 1.0
            for d in ti.static(range(2)):
                new_sig = sig[d, d]
                if material[m, p] == 2:  # Payload plasticity: strict clamping prevents spring-winding
                    new_sig = ti.min(ti.max(sig[d, d], 1 - 5e-3), 1 + 5e-3)
                Jp[m, p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig

            if material[m, p] == 2:
                F[m, p] = U @ sig @ V.transpose()

            stress = 2 * mu * (F[m, p] - U @ V.transpose()) @ F[m, p].transpose() + \
                     ti.Matrix.identity(float, 2) * la * J * (J - 1)

            # Active Muscle Stress
            if material[m, p] == 3:
                contractile_pressure = actuation_strength * activation
                if ti.static(use_anisotropic_actuation):
                    fd = fiber_dir[m, p].normalized()
                    stress += fd.outer_product(fd) * contractile_pressure * J
                else:
                    stress += ti.Matrix.identity(float, 2) * contractile_pressure * J

        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress

        # APIC affine momentum transfer
        affine = stress + current_mass * C[m, p]

        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[m, base + offset] += weight * (current_mass * v[m, p] + affine @ dpos)
            grid_m[m, base + offset] += weight * current_mass


@ti.kernel
def _substep_grid_ops():
    """Grid velocity normalization, damping, and boundary conditions."""
    for m, i, j in grid_m:
        if grid_m[m, i, j] > 0:
            grid_v[m, i, j] /= grid_m[m, i, j]

            # Brinkman drag (disabled: grid_drag=0). Tunable if needed.
            if ti.static(grid_drag > 0):
                grid_v[m, i, j] *= (1.0 - grid_drag)

            # Boundary damping layer
            damp_cells = n_grid // 20
            damp = 1.0
            if i < damp_cells: damp *= 0.95 + 0.05 * i / damp_cells
            if i > n_grid - damp_cells: damp *= 0.95 + 0.05 * (n_grid - i) / damp_cells
            grid_v[m, i, j] *= damp

            # Wall collisions
            if i < 3 and grid_v[m, i, j][0] < 0: grid_v[m, i, j][0] = 0
            if i > n_grid - 3 and grid_v[m, i, j][0] > 0: grid_v[m, i, j][0] = 0
            if j < 3 and grid_v[m, i, j][1] < 0: grid_v[m, i, j][1] = 0
            if j > n_grid - 3 and grid_v[m, i, j][1] > 0: grid_v[m, i, j][1] = 0


@ti.kernel
def _substep_g2p():
    """Grid-to-Particle gather, gravity, position update, time advance."""
    for m, p in x:
        if material[m, p] < 0: continue

        base = (x[m, p] * inv_dx - 0.5).cast(int)
        if base[0] >= 0 and base[1] >= 0 and base[0] < n_grid - 2 and base[1] < n_grid - 2:
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

        g_factor = payload_gravity_factor if material[m, p] == 2 else 1.0
        v[m, p][1] -= dt * gravity * g_factor
        x[m, p] += dt * v[m, p]

        for d in ti.static(range(2)):
            x[m, p][d] = ti.max(ti.min(x[m, p][d], 0.999), 0.001)

    sim_time[None] += dt


@ti.kernel
def _substep_viscosity(visc: float):
    """Laplacian grid diffusion: reads grid_v, writes grid_v_lap.
    5-point stencil: ν·dt·∇²v added to each cell. Boundary cells clamp to nearest valid neighbor.
    visc is passed at call time so the validation script can sweep values without recompile.
    CFL: visc ≤ dx²/(4·dt) ≈ 0.305."""
    for m, i, j in grid_v:
        if grid_m[m, i, j] <= 0.0:
            grid_v_lap[m, i, j] = [0.0, 0.0]
        else:
            v_c  = grid_v[m, i,                          j                         ]
            v_px = grid_v[m, ti.min(i + 1, n_grid - 1),  j                         ]
            v_mx = grid_v[m, ti.max(i - 1, 0),           j                         ]
            v_py = grid_v[m, i,                          ti.min(j + 1, n_grid - 1) ]
            v_my = grid_v[m, i,                          ti.max(j - 1, 0)          ]
            lap  = (v_px + v_mx + v_py + v_my - 4.0 * v_c) * inv_dx * inv_dx
            grid_v_lap[m, i, j] = v_c + visc * dt * lap


@ti.kernel
def _substep_copy_lap():
    """Copy Laplacian-diffused velocities back to grid_v for G2P to read."""
    for m, i, j in grid_v:
        grid_v[m, i, j] = grid_v_lap[m, i, j]


def substep():
    """Main physics step: P2G → grid ops → [viscosity] → G2P.
    All kernels queued on CUDA stream; no CPU sync between them."""
    _substep_p2g()
    _substep_grid_ops()
    if water_visc > 0.0:
        _substep_viscosity(water_visc)
        _substep_copy_lap()
    _substep_g2p()


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
    # Re-calculate Pulse for visual sync in the renderer
    period = 1.0 / actuation_freq
    phase = (sim_time[None] % period) / period
    activation = 0.0
    if phase < 0.2:
        activation = 0.5 * (1.0 - ti.cos(phase / 0.2 * 3.14159265))
    elif phase < 1.0:
        activation = 0.5 * (1.0 + ti.cos((phase - 0.2) / 0.8 * 3.14159265))
    
    grid_norm_factor = float(p_grid_side - 1) if p_grid_side > 1 else 1.0

    for m, p in x:
        pos = x[m, p]
        mat = material[m, p]
        if mat < 0 or pos[0] < 0: continue

        # --- LIGHT CALCULATION ---
        vel = v[m, p].norm()
        stress = 1.0 - Jp[m, p] 
        
        color = ti.Vector([0.0, 0.0, 0.0])
        intensity = 0.0
        
        row_idx, col_idx = m // p_grid_side, m % p_grid_side
        norm_x = float(col_idx) / grid_norm_factor
        norm_y = float(row_idx) / grid_norm_factor

        if mat == 0: # WATER
            if vel > 0.5:
                color = ti.Vector([0.1, 0.2, 0.8])
                intensity = 0.015 * (vel / 10.0)

        elif mat == 1: # JELLY
            hue = instance_hue[m]
            sat = 0.4 + (norm_y * 0.4)
            base_col = hsv2rgb(hue, sat, 0.9)
            glow = ti.abs(stress) * 2.0 + 0.1
            color = base_col + ti.Vector([glow, glow, glow])
            intensity = 0.05 + (glow * 0.2)

        elif mat == 3: # MUSCLE (Visually syncs with activation)
            hue = instance_hue[m]
            color = hsv2rgb(hue, 0.2, 1.0) + ti.Vector([activation, activation, activation])
            intensity = 0.4 + (activation * 0.4)
            
        elif mat == 2: # PAYLOAD
            color = ti.Vector([1.0, 0.2, 0.0]) 
            intensity = 0.8

        # --- SPLATTING ---
        if intensity > 0.001:
            row, col = m // p_grid_side, m % p_grid_side
            center_x = (pos[0] + col) * p_res_sub
            center_y = ((1.0 - pos[1]) + row) * p_res_sub
            
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
        center_x = (pos[0] + col) * p_res_sub
        center_y = ((1.0 - pos[1]) + row) * p_res_sub

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
def _load_particles_kernel(instance: int, pos_np: ti.types.ndarray(), mat_np: ti.types.ndarray()):
    """Reset a specific instance with new particle data."""
    for p in range(n_particles):
        x[instance, p] = [pos_np[p, 0], pos_np[p, 1]]
        material[instance, p] = mat_np[p]
        v[instance, p] = [0.0, 0.0]
        F[instance, p] = ti.Matrix.identity(float, 2)
        C[instance, p] = ti.Matrix.zero(float, 2, 2)
        Jp[instance, p] = 1.0


@ti.kernel
def _load_fiber_kernel(instance: int, fiber_np: ti.types.ndarray()):
    """Load per-particle fiber directions for anisotropic actuation."""
    for p in range(n_particles):
        fiber_dir[instance, p] = [fiber_np[p, 0], fiber_np[p, 1]]


@ti.kernel
def _load_fiber_default(instance: int):
    """Fill fiber directions with default [1, 0] (unused placeholder)."""
    for p in range(n_particles):
        fiber_dir[instance, p] = [1.0, 0.0]


def load_particles(instance, pos_np, mat_np, fiber_np=None):
    """Python dispatcher: load particles and optionally fiber directions."""
    _load_particles_kernel(instance, pos_np, mat_np)
    if fiber_np is not None:
        _load_fiber_kernel(instance, fiber_np)
    else:
        _load_fiber_default(instance)

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
            # Check for boundary-stuck payload (ceiling/floor)
            if final[i, 0] > 0.93 or final[i, 0] < 0.01:
                results[i, 4] = 0.0  # Invalid: stuck at boundary
            else:
                results[i, 4] = 1.0  # Valid
        else:
            results[i, 4] = 0.0  # Invalid: payload lost
    return results

