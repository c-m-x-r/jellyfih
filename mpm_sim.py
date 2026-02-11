import taichi as ti
import time
import numpy as np
import cv2
import math

ti.set_logging_level(ti.ERROR)
ti.init(arch=ti.cuda)  # CUDA required for atomic_add performance

# --- SIMULATION SETTINGS ---
n_instances = 16
quality = 1  # 1=low-res (128 grid), 2=high-res (256 grid)
n_particles = 70000
n_grid = 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 5e-5 / quality
p_vol = (dx * 0.5) ** 2
p_rho = 1
p_mass = p_vol * p_rho

# --- MATERIAL SETTINGS ---
E_base = 0.1e4
nu_base = 0.2
mu_0_base = E_base / (2 * (1 + nu_base))
lambda_0_base = E_base * nu_base / ((1 + nu_base) * (1 - 2 * nu_base))

# Jelly/Muscle Elasticity
E_jelly = 0.1e4  
nu_jelly = 0.45 
mu_jelly = E_jelly / (2 * (1 + nu_jelly))
lambda_jelly = E_jelly * nu_jelly / ((1 + nu_jelly) * (1 - 2 * nu_jelly))

water_lambda = 4000.0
gravity = 10.0

# --- ACTUATION SETTINGS ---
actuation_freq = 2.0   # Hz
actuation_strength = 40.0 

# --- RECORDING SETTINGS ---
frames = 100
substeps_per_frame = 50
warmup_steps = 200

# --- RENDERING SETTINGS ---
grid_side = int(math.ceil(math.sqrt(n_instances)))
video_res = 1024
res_sub = video_res // grid_side

print(f"Allocating {n_instances} instances with {n_particles} particles each...")

# Fields
x = ti.Vector.field(2, dtype=float, shape=(n_instances, n_particles))
v = ti.Vector.field(2, dtype=float, shape=(n_instances, n_particles))
C = ti.Matrix.field(2, 2, dtype=float, shape=(n_instances, n_particles))
F = ti.Matrix.field(2, 2, dtype=float, shape=(n_instances, n_particles))
material = ti.field(dtype=int, shape=(n_instances, n_particles))
Jp = ti.field(dtype=float, shape=(n_instances, n_particles))
grid_v = ti.Vector.field(2, dtype=float, shape=(n_instances, n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_instances, n_grid, n_grid))

sim_time = ti.field(dtype=float, shape=()) 

# Render Buffer (HDR - can exceed 1.0)
frame_buffer = ti.field(dtype=float, shape=(video_res, video_res, 3))


@ti.kernel
def substep():
    # 0. Update Time & Actuation Phase
    current_time = sim_time[None]
    actuation_phase = ti.math.sin(current_time * actuation_freq * 2 * math.pi)
    contraction_factor = ti.max(0.0, actuation_phase) 

    # 1. Reset Grid
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

        # Material Logic
        mu, la = mu_0_base, lambda_0_base
        h = ti.exp(10 * (1.0 - Jp[m, p]))
        
        if material[m, p] == 1:  # Jelly
            mu, la = mu_jelly * 0.3, lambda_jelly * 0.3
            
        elif material[m, p] == 3: # Muscle
            mu, la = mu_jelly * 0.3, lambda_jelly * 0.3
            # Active Contraction
            dist_x = 0.5 - x[m, p][0]
            dir_x = 1.0 if dist_x > 0 else -1.0
            v[m, p][0] += dir_x * actuation_strength * contraction_factor * dt

        elif material[m, p] == 0:  # Water
            mu, la = 0.0, water_lambda

        # SVD & Plasticity
        U, sig, V = ti.svd(F[m, p])
        for d in ti.static(range(2)):
            sig[d, d] = ti.max(sig[d, d], 1e-6)

        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if material[m, p] == 2:  # Payload
                new_sig = ti.min(ti.max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)
            Jp[m, p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig

        if material[m, p] == 0:
            F[m, p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        elif material[m, p] == 2:
            F[m, p] = U @ sig @ V.transpose()
            
        stress = 2 * mu * (F[m, p] - U @ V.transpose()) @ F[m, p].transpose() + \
                 ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[m, p]
        
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[m, base + offset] += weight * (p_mass * v[m, p] + affine @ dpos)
            grid_m[m, base + offset] += weight * p_mass

    # 3. Grid Operations
    for m, i, j in grid_m:
        if grid_m[m, i, j] > 0:
            grid_v[m, i, j] /= grid_m[m, i, j]
            
            # Boundary Damping
            damp_cells = n_grid // 20
            damp = 1.0
            if i < damp_cells: damp *= 0.95 + 0.05 * i / damp_cells
            if i > n_grid - damp_cells: damp *= 0.95 + 0.05 * (n_grid - i) / damp_cells
            grid_v[m, i, j] *= damp

            # Wall Collisions
            if i < 3 and grid_v[m, i, j][0] < 0: grid_v[m, i, j][0] = 0
            if i > n_grid - 3 and grid_v[m, i, j][0] > 0: grid_v[m, i, j][0] = 0
            if j < 3 and grid_v[m, i, j][1] < 0: grid_v[m, i, j][1] = 0
            if j > n_grid - 3 and grid_v[m, i, j][1] > 0: grid_v[m, i, j][1] = 0

    # 4. G2P
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

        v[m, p][1] -= dt * gravity
        x[m, p] += dt * v[m, p]
        
        for d in ti.static(range(2)):
            x[m, p][d] = ti.max(ti.min(x[m, p][d], 0.999), 0.001)

    sim_time[None] += dt


# --- RENDERING PIPELINE (BIOLUMINESCENCE) ---

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
    Uses atomic_add to accumulate light from particles.
    """
    # Actuation pulse for visual sync
    pulse = ti.sin(sim_time[None] * actuation_freq * 2 * math.pi)
    
    # Pre-calculate normalization factor for the gradient
    # Avoid division by zero if grid_side is 1
    grid_norm_factor = float(p_grid_side - 1) if p_grid_side > 1 else 1.0

    for m, p in x:
        pos = x[m, p]
        mat = material[m, p]
        if mat < 0 or pos[0] < 0: continue

        # --- LIGHT CALCULATION ---
        vel = v[m, p].norm()
        stress = 1.0 - Jp[m, p] # >0 means compressed
        
        color = ti.Vector([0.0, 0.0, 0.0])
        intensity = 0.0
        
        # Calculate instance position in the 2D grid for the gradient
        row_idx, col_idx = m // p_grid_side, m % p_grid_side
        norm_x = float(col_idx) / grid_norm_factor
        norm_y = float(row_idx) / grid_norm_factor

        if mat == 0: # WATER (Sea Smoke)
            if vel > 0.5:
                color = ti.Vector([0.1, 0.2, 0.8]) # Deep Indigo
                intensity = 0.015 * (vel / 10.0)
                
        elif mat == 1: # JELLY (Bioluminescent Skin Gradient)
            # Hue: Shifts from Cyan (0.55) to Azure (0.65) across columns
            # Saturation: Shifts from Pale (0.4) to Vivid (0.8) across rows
            hue = 0.55 + (norm_x * 0.1)
            sat = 0.4 + (norm_y * 0.4)
            base_col = hsv2rgb(hue, sat, 0.9)
            
            # Glow on compression (stress)
            glow = ti.max(0.0, stress * 5.0)
            color = base_col + ti.Vector([glow, glow, glow])
            intensity = 0.05 + (glow * 0.2)
            
        elif mat == 3: # MUSCLE (High Energy)
            # Sync hue with skin, but keep it high-value/low-saturation for a "core" look
            hue = 0.55 + (norm_x * 0.1)
            activity = ti.max(0.0, pulse)
            color = hsv2rgb(hue, 0.2, 1.0) + ti.Vector([activity, activity, activity])
            intensity = 0.4 + (activity * 0.4)
            
        elif mat == 2: # PAYLOAD (Black Box)
            color = ti.Vector([1.0, 0.2, 0.0]) # High-contrast Orange
            intensity = 0.8

        # --- SPLATTING (Atomic Accumulation) ---
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
                            
                            # Atomic add to the HDR frame_buffer
                            frame_buffer[py, px, 0] += val[0]
                            frame_buffer[py, px, 1] += val[1]
                            frame_buffer[py, px, 2] += val[2]
                        

@ti.kernel
def load_particles(instance: int, pos_np: ti.types.ndarray(), mat_np: ti.types.ndarray()):
    for p in range(n_particles):
        x[instance, p] = [pos_np[p, 0], pos_np[p, 1]]
        material[instance, p] = mat_np[p]
        v[instance, p] = [0.0, 0.0]
        F[instance, p] = ti.Matrix.identity(float, 2)
        C[instance, p] = ti.Matrix.zero(float, 2, 2)
        Jp[instance, p] = 1.0


def main():
    from make_jelly import fill_tank, random_genome

    sim_time[None] = 0.0

    genome = random_genome()
    # Pass grid_res to match physics!
    positions, materials, info = fill_tank(genome, n_particles, grid_res=int(n_grid/quality))
    print(f"Tank: {info['n_robot']} robot + {info['n_water']} water + {info['n_dead']} dead")

    for m in range(n_instances):
        load_particles(m, positions, materials)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('biolum_sim.mp4', fourcc, 30.0, (video_res, video_res))

    print(f"Warmup: {warmup_steps} substeps...")
    for _ in range(warmup_steps):
        substep()
    ti.sync()

    print("Simulating + rendering...")
    
    for i in range(frames):
        for _ in range(substeps_per_frame):
            substep()

        # New Render Pipeline
        clear_frame_buffer()                 # 1. Clear to Black
        render_frame_abyss(res_sub, grid_side, 4) # 2. Accumulate Light
        tone_map_and_encode()                # 3. Tone Map (HDR -> sRGB)
        
        frame_np = frame_buffer.to_numpy()
        frame_u8 = (np.clip(frame_np, 0.0, 1.0) * 255).astype(np.uint8)
        out.write(cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR))

        if i % 10 == 0:
            print(f"Frame {i}/{frames}")

    ti.sync()
    out.release()
    print("Done.")

if __name__ == "__main__":
    main()