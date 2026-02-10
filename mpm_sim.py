import taichi as ti
import time
import numpy as np
import cv2
import math

ti.set_logging_level(ti.ERROR)
ti.init(arch=ti.cuda)  # Use ti.cuda or ti.gpu when GPU available

# --- SIMULATION SETTINGS ---
# n_instances: CMA-ES population size. Minimum ~10 for 9-gene genome.
# Fewer instances = more particles per sim at same VRAM cost.
n_instances = 16
quality = 1  # 1=low-res (128 grid), 2=high-res (256 grid)
n_particles = 70000
n_grid = 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 5e-5 / quality  # Smaller timestep for stability with stiffer fluid
p_vol = (dx * 0.5) ** 2
p_rho = 1
p_mass = p_vol * p_rho
E, nu = 0.1e4, 0.2
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
water_lambda = 4000.0  # Much higher bulk modulus for water to resist compression
gravity = 10.0  # Uniform gravity — creates hydrostatic pressure that fills voids

# --- RECORDING SETTINGS ---
frames = 400  # Frames to record
substeps_per_frame = 50  # Physics steps per frame
warmup_steps = 200  # Let hydrostatic pressure equilibrate before recording

# --- RENDERING SETTINGS ---
grid_side = int(math.ceil(math.sqrt(n_instances)))
video_res = 1024
res_sub = video_res // grid_side  # Pixels per instance tile

print(f"Allocating {n_instances} instances with {n_particles} particles each...")
print(f"Total Particles: {n_instances * n_particles:,}")
print(f"Grid: {n_grid}x{n_grid}, approx PPC: {n_particles / (0.81 * n_grid**2):.1f}")

# Physics Fields
x = ti.Vector.field(2, dtype=float, shape=(n_instances, n_particles))
v = ti.Vector.field(2, dtype=float, shape=(n_instances, n_particles))
C = ti.Matrix.field(2, 2, dtype=float, shape=(n_instances, n_particles))
F = ti.Matrix.field(2, 2, dtype=float, shape=(n_instances, n_particles))
material = ti.field(dtype=int, shape=(n_instances, n_particles))
Jp = ti.field(dtype=float, shape=(n_instances, n_particles))
grid_v = ti.Vector.field(2, dtype=float, shape=(n_instances, n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_instances, n_grid, n_grid))

# Single-frame render buffer (~12 MB instead of ~6.5 GB for history + video buffers)
frame_buffer = ti.field(dtype=float, shape=(video_res, video_res, 3))


@ti.kernel
def substep():
    # 1. Reset Grid
    for m, i, j in grid_m:
        grid_v[m, i, j] = [0, 0]
        grid_m[m, i, j] = 0

    # 2. P2G
    for m, p in x:
        # Skip dead/invalid particles
        if material[m, p] < 0:
            continue
        if x[m, p][0] < 0.01 or x[m, p][0] > 0.99:
            continue
        if x[m, p][1] < 0.01 or x[m, p][1] > 0.99:
            continue

        base = (x[m, p] * inv_dx - 0.5).cast(int)
        fx = x[m, p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        F[m, p] = (ti.Matrix.identity(float, 2) + dt * C[m, p]) @ F[m, p]

        # Material hardening (from original mpm88)
        h = ti.exp(10 * (1.0 - Jp[m, p]))
        if material[m, p] == 1:  # Jelly: softer
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if material[m, p] == 0:  # Water: no shear, high bulk modulus
            mu = 0.0
            la = water_lambda  # Use much higher bulk modulus for incompressibility

        U, sig, V = ti.svd(F[m, p])
        for d in ti.static(range(2)):
            sig[d, d] = ti.max(sig[d, d], 1e-6)

        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if material[m, p] == 2:  # Payload: snow-like plasticity
                new_sig = ti.min(ti.max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)
            Jp[m, p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig

        if material[m, p] == 0:
            F[m, p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        elif material[m, p] == 2:
            F[m, p] = U @ sig @ V.transpose()
        stress = 2 * mu * (F[m, p] - U @ V.transpose()) @ F[m, p].transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1)
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

            # Side-only damping (left/right walls absorb lateral waves)
            # No top/bottom damping — bottom must support hydrostatic pressure,
            # damping there fights gravity and causes particles to sag into the wall
            damp_cells = n_grid // 20
            damp = 1.0
            if i < damp_cells:
                damp *= 0.95 + 0.05 * i / damp_cells
            if i > n_grid - damp_cells:
                damp *= 0.95 + 0.05 * (n_grid - i) / damp_cells
            grid_v[m, i, j] *= damp

            # Hard boundary conditions (solid walls)
            if i < 3 and grid_v[m, i, j][0] < 0: grid_v[m, i, j][0] = 0
            if i > n_grid - 3 and grid_v[m, i, j][0] > 0: grid_v[m, i, j][0] = 0
            if j < 3 and grid_v[m, i, j][1] < 0: grid_v[m, i, j][1] = 0
            if j > n_grid - 3 and grid_v[m, i, j][1] > 0: grid_v[m, i, j][1] = 0

    # 4. G2P
    for m, p in x:
        # Skip dead/invalid particles (need margin from edges for stencil)
        if material[m, p] < 0:
            continue
        if x[m, p][0] < 0.03 or x[m, p][0] > 0.97:
            continue
        if x[m, p][1] < 0.03 or x[m, p][1] > 0.97:
            continue

        base = (x[m, p] * inv_dx - 0.5).cast(int)
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

        # Gravity on all active particles — hydrostatic pressure fills voids
        if material[m, p] >= 0:
            v[m, p][1] -= dt * gravity

        x[m, p] += dt * v[m, p]


@ti.kernel
def clear_frame_buffer():
    for i, j, c in frame_buffer:
        frame_buffer[i, j, c] = 1.0


@ti.kernel
def render_frame(p_res_sub: int, p_grid_side: int, radius: float):
    """Render current particle state directly from x[] to frame_buffer (GPU-only)."""
    for m, p in x:
        pos = x[m, p]
        mat = material[m, p]

        # Skip dead particles (material -1 or position at -1,-1)
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
                        if mat == 0:  # Water: Blue
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
    """Load particle data from numpy arrays into Taichi fields for one instance."""
    for p in range(n_particles):
        x[instance, p] = [pos_np[p, 0], pos_np[p, 1]]
        material[instance, p] = mat_np[p]
        v[instance, p] = [0.0, 0.0]
        F[instance, p] = ti.Matrix.identity(float, 2)
        C[instance, p] = ti.Matrix.zero(float, 2, 2)
        Jp[instance, p] = 1.0


def main():
    from make_jelly import fill_tank, random_genome

    # Fill tank with water + random jellyfish morphology
    genome = random_genome()
    positions, materials, info = fill_tank(genome, n_particles)
    print(f"Tank: {info['n_robot']} robot + {info['n_water']} water + {info['n_dead']} dead")

    # Load identical initial state into all instances
    for m in range(n_instances):
        load_particles(m, positions, materials)

    # Set up streaming video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('gpu_rendered_sim.mp4', fourcc, 30.0, (video_res, video_res))

    # Warmup: let hydrostatic pressure equilibrate
    # Sound speed = sqrt(lambda/rho) ≈ 110, domain traversal ≈ 160 steps
    print(f"Warmup: {warmup_steps} substeps...")
    for _ in range(warmup_steps):
        substep()
    ti.sync()

    print("Simulating + rendering (streaming)...")
    start_wall_time = time.time()

    for i in range(frames):
        for _ in range(substeps_per_frame):
            substep()

        # Render on GPU, transfer single frame (~12 MB) to CPU, encode
        clear_frame_buffer()
        render_frame(res_sub, grid_side, 1)
        frame_np = frame_buffer.to_numpy()
        frame_u8 = (np.clip(frame_np, 0.0, 1.0) * 255).astype(np.uint8)
        out.write(cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR))

        if i % 10 == 0:
            print(f"Frame {i}/{frames}")

    ti.sync()
    out.release()

    # Performance metrics
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
