import taichi as ti
import time
import numpy as np
import cv2
import math

ti.init(arch=ti.gpu)  # Use ti.cuda or ti.gpu

# --- SIMULATION SETTINGS ---
n_instances = 16
quality = 1  # 1=low-res (128 grid), 2=high-res (256 grid)
n_particles = 70000
n_grid = 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 5e-5 / quality
p_vol = (dx * 0.5) ** 2

# Material densities
rho_water = 1.0
rho_jelly = 1.05   # Slightly denser than water
rho_payload = 2.0  # Heavy instrumented package

# Elasticity parameters
E = 0.1e4  # Base Young's modulus
nu = 0.2   # Base Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
water_lambda = 2000.0
gravity = 10.0

# Jelly-specific Lame parameters (nu=0.45, nearly incompressible)
jelly_nu = 0.45
jelly_h = 0.3
jelly_mu = E / (2 * (1 + jelly_nu)) * jelly_h
jelly_la = E * jelly_nu / ((1 + jelly_nu) * (1 - 2 * jelly_nu)) * jelly_h

# --- ACTUATION SETTINGS ---
actuation_frequency = 1.0   # Hz
actuation_amplitude = 5.0   # Actuation strength

# --- RECORDING SETTINGS ---
frames = 400
substeps_per_frame = 50
warmup_steps = 2000

# --- RENDERING SETTINGS ---
grid_side = int(math.ceil(math.sqrt(n_instances)))
video_res = 1024
res_sub = video_res // grid_side

print(f"Allocating {n_instances} instances with {n_particles} particles each...")
print(f"Total Particles: {n_instances * n_particles:,}")
print(f"Grid: {n_grid}x{n_grid}, approx PPC: {n_particles / (0.81 * n_grid**2):.1f}")

# --- PHYSICS FIELDS ---
x = ti.Vector.field(2, dtype=float, shape=(n_instances, n_particles))
v = ti.Vector.field(2, dtype=float, shape=(n_instances, n_particles))
C = ti.Matrix.field(2, 2, dtype=float, shape=(n_instances, n_particles))
F = ti.Matrix.field(2, 2, dtype=float, shape=(n_instances, n_particles))
material = ti.field(dtype=int, shape=(n_instances, n_particles))
Jp = ti.field(dtype=float, shape=(n_instances, n_particles))
p_mass_field = ti.field(dtype=float, shape=(n_instances, n_particles))
grid_v = ti.Vector.field(2, dtype=float, shape=(n_instances, n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_instances, n_grid, n_grid))

# Actuation & Time fields
jelly_centroid = ti.Vector.field(2, dtype=float, shape=(n_instances,))
jelly_count = ti.field(dtype=int, shape=(n_instances,))
sim_time_field = ti.field(dtype=float, shape=())  # 0-D field to store time on GPU

# Render buffer
frame_buffer = ti.field(dtype=float, shape=(video_res, video_res, 3))


@ti.kernel
def compute_jelly_centroid():
    """Compute centroid of jelly particles per instance via atomic reduction."""
    for m in range(n_instances):
        jelly_centroid[m] = [0.0, 0.0]
        jelly_count[m] = 0
    
    for m, p in x:
        if material[m, p] == 1:
            ti.atomic_add(jelly_centroid[m], x[m, p])
            ti.atomic_add(jelly_count[m], 1)
            
    for m in range(n_instances):
        if jelly_count[m] > 0:
            jelly_centroid[m] /= float(jelly_count[m])


@ti.kernel
def substep():
    # --- 0. Update Time & Actuation ---
    # We read and increment time on the GPU to avoid CPU synchronization overhead
    current_time = sim_time_field[None]
    phase = current_time * actuation_frequency * 2.0 * 3.14159
    contraction = ti.max(ti.sin(phase), 0.0) * actuation_amplitude
    
    # Apply actuation forces
    for m, p in x:
        if material[m, p] == 1:
            to_center = jelly_centroid[m] - x[m, p]
            dist = to_center.norm()
            if dist > 1e-6:
                v[m, p] += contraction * (to_center / dist) * dt

    # --- 1. Reset Grid ---
    for m, i, j in grid_m:
        grid_v[m, i, j] = [0, 0]
        grid_m[m, i, j] = 0

    # --- 2. P2G ---
    for m, p in x:
        if material[m, p] < 0:
            continue

        base = (x[m, p] * inv_dx - 0.5).cast(int)
        if base[0] < 0 or base[1] < 0 or base[0] >= n_grid - 2 or base[1] >= n_grid - 2:
            continue
        
        fx = x[m, p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        F[m, p] = (ti.Matrix.identity(float, 2) + dt * C[m, p]) @ F[m, p]

        # Material parameters
        h = ti.exp(10 * (1.0 - Jp[m, p]))
        mu = mu_0 * h
        la = lambda_0 * h

        if material[m, p] == 0:     # Water
            mu = 0.0
            la = water_lambda
        elif material[m, p] == 1:   # Jelly
            mu = jelly_mu
            la = jelly_la
        elif material[m, p] == 2:   # Payload
            h = 10.0
            mu = mu_0 * h
            la = lambda_0 * h

        U, sig, V = ti.svd(F[m, p])
        for d in ti.static(range(2)):
            sig[d, d] = ti.max(sig[d, d], 1e-6)

        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            Jp[m, p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig

        if material[m, p] == 0:
            F[m, p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
            
        stress = 2 * mu * (F[m, p] - U @ V.transpose()) @ F[m, p].transpose() + \
                 ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass_field[m, p] * C[m, p]
        
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[m, base + offset] += weight * (p_mass_field[m, p] * v[m, p] + affine @ dpos)
            grid_m[m, base + offset] += weight * p_mass_field[m, p]

    # --- 3. Grid Operations ---
    for m, i, j in grid_m:
        if grid_m[m, i, j] > 0:
            grid_v[m, i, j] /= grid_m[m, i, j]

            # Damping
            damp_cells = n_grid // 20
            damp = 1.0
            if i < damp_cells:
                damp *= 0.95 + 0.05 * i / damp_cells
            if i > n_grid - damp_cells:
                damp *= 0.95 + 0.05 * (n_grid - i) / damp_cells
            grid_v[m, i, j] *= damp

            # Boundaries
            if i < 3 and grid_v[m, i, j][0] < 0: grid_v[m, i, j][0] = 0
            if i > n_grid - 3 and grid_v[m, i, j][0] > 0: grid_v[m, i, j][0] = 0
            if j < 3 and grid_v[m, i, j][1] < 0: grid_v[m, i, j][1] = 0
            if j > n_grid - 3 and grid_v[m, i, j][1] > 0: grid_v[m, i, j][1] = 0

    # --- 4. G2P ---
    for m, p in x:
        if material[m, p] < 0:
            continue

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

    # --- Increment Time Locally ---
    sim_time_field[None] += dt


@ti.kernel
def clear_frame_buffer():
    for i, j, c in frame_buffer:
        frame_buffer[i, j, c] = 1.0


@ti.kernel
def render_frame(p_res_sub: int, p_grid_side: int, radius: float):
    for m, p in x:
        pos = x[m, p]
        mat = material[m, p]
        
        if mat < 0 or pos[0] < 0:
            continue

        row, col = m // p_grid_side, m % p_grid_side
        center_x = (pos[0] + col) * p_res_sub
        center_y = ((1.0 - pos[1]) + row) * p_res_sub

        low_x, high_x = int(center_x - radius), int(center_x + radius)
        low_y, high_y = int(center_y - radius), int(center_y + radius)

        for px in range(low_x, high_x + 1):
            for py in range(low_y, high_y + 1):
                if 0 <= px < video_res and 0 <= py < video_res:
                    dist_sq = (px - center_x)**2 + (py - center_y)**2
                    if dist_sq <= radius**2:
                        if mat == 0:    # Water: Blue
                            frame_buffer[py, px, 0] = 0.2
                            frame_buffer[py, px, 1] = 0.5
                            frame_buffer[py, px, 2] = 0.9
                        elif mat == 1:  # Jelly: Cyan
                            frame_buffer[py, px, 0] = 0.1
                            frame_buffer[py, px, 1] = 0.9
                            frame_buffer[py, px, 2] = 0.9
                        elif mat == 2:  # Payload: Dark red
                            frame_buffer[py, px, 0] = 0.8
                            frame_buffer[py, px, 1] = 0.2
                            frame_buffer[py, px, 2] = 0.2


@ti.kernel
def load_particles(instance: int, pos_np: ti.types.ndarray(), mat_np: ti.types.ndarray()):
    for p in range(n_particles):
        x[instance, p] = [pos_np[p, 0], pos_np[p, 1]]
        material[instance, p] = mat_np[p]

        if mat_np[p] == 0:       # Water
            p_mass_field[instance, p] = p_vol * rho_water
        elif mat_np[p] == 1:     # Jelly
            p_mass_field[instance, p] = p_vol * rho_jelly
        elif mat_np[p] == 2:     # Payload
            p_mass_field[instance, p] = p_vol * rho_payload
        else:
            p_mass_field[instance, p] = 0.0

        v[instance, p] = [0.0, 0.0]
        F[instance, p] = ti.Matrix.identity(float, 2)
        C[instance, p] = ti.Matrix.zero(float, 2, 2)
        Jp[instance, p] = 1.0


def main():
    from make_jelly import fill_tank, random_genome

    # 1. Generate & Load
    genome = random_genome()
    positions, materials, info = fill_tank(genome, n_particles)
    print(f"Tank: {info['n_robot']} robot + {info['n_water']} water + {info['n_dead']} dead")

    for m in range(n_instances):
        load_particles(m, positions, materials)

    # 2. Setup Video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('gpu_rendered_sim.mp4', fourcc, 30.0, (video_res, video_res))

    # 3. Warmup
    # We use the field to track time, avoiding Python args
    sim_time_field[None] = 0.0
    print(f"Warmup: {warmup_steps} substeps...")
    
    for i in range(warmup_steps):
        if i % 100 == 0:
            compute_jelly_centroid()
        substep()
    ti.sync()

    # 4. Main Loop
    print("Simulating + rendering (streaming)...")
    start_wall_time = time.time()

    for i in range(frames):
        # Update centroid once per frame (expensive-ish due to atomics)
        compute_jelly_centroid()
        
        # Burst fire substeps (fast, no arguments passed)
        for _ in range(substeps_per_frame):
            substep()

        # Render
        clear_frame_buffer()
        render_frame(res_sub, grid_side, 1)
        
        # Export
        frame_np = frame_buffer.to_numpy()
        frame_u8 = (np.clip(frame_np, 0.0, 1.0) * 255).astype(np.uint8)
        out.write(cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR))

        if i % 10 == 0:
            print(f"Frame {i}/{frames}")

    ti.sync()
    out.release()

    # Metrics
    end_wall_time = time.time()
    total_time = end_wall_time - start_wall_time
    fps = frames / total_time

    print(f"\n{'='*40}")
    print(f"SIMULATION PERFORMANCE REPORT")
    print(f"{'-'*40}")
    print(f"Total Time:            {total_time:.4f} s")
    print(f"Frames/s:              {fps:.2f}")
    print(f"Frames/s/instance:     {fps / n_instances:.4f}")
    print(f"{'='*40}")
    print(f"Video saved: gpu_rendered_sim.mp4")


if __name__ == "__main__":
    main()