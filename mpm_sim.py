import taichi as ti
import time
import numpy as np # Needed for the final export
import cv2 

ti.set_logging_level(ti.ERROR)
# Use CPU for testing when GPU unavailable, fallback to Vulkan/GPU when available
ti.init(arch=ti.cuda)  # Use ti.cuda or ti.gpu when GPU available

# --- SIMULATION SETTINGS ---
# quality=1 for fast iteration, quality=2 for detailed vortex capture
n_instances = 16
quality = 1  # 1=low-res (128 grid), 2=high-res (256 grid)
n_particles = 36000 * quality**2
n_grid = 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 5e-5 / quality  # Smaller timestep for stability with stiffer fluid
p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho
E, nu = 0.1e4, 0.2
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
water_lambda = 12000.0  # Much higher bulk modulus for water to resist compression
gravity = 1.0  # Uniform gravity — creates hydrostatic pressure that fills voids
particle_bound = 3 * dx  # Stencil safety zone, matches wall boundary

# --- RECORDING SETTINGS ---
frames = 400  # Frames to record
substeps_per_frame = 50  # Physics steps per frame
frame_buffer = ti.field(dtype=float, shape=(1024, 1024, 3))  # Single frame: ~12MB vs ~5GB
print(f"Allocating {n_instances} instances with {n_particles} particles each...")
print(f"Total Particles: {n_instances * n_particles:,}")

# Physics Fields
x = ti.Vector.field(2, dtype=float, shape=(n_instances, n_particles))
v = ti.Vector.field(2, dtype=float, shape=(n_instances, n_particles))
C = ti.Matrix.field(2, 2, dtype=float, shape=(n_instances, n_particles))
F = ti.Matrix.field(2, 2, dtype=float, shape=(n_instances, n_particles))
material = ti.field(dtype=int, shape=(n_instances, n_particles))
Jp = ti.field(dtype=float, shape=(n_instances, n_particles))
grid_v = ti.Vector.field(2, dtype=float, shape=(n_instances, n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_instances, n_grid, n_grid))

# --- HISTORY BUFFER (The VRAM Tape) ---
# Shape: [Time, Instance, Particle]
history_x = ti.Vector.field(2, dtype=float, shape=(frames, n_instances, n_particles))



@ti.kernel
def substep():
    # ... (Your existing physics code here - NO CHANGES NEEDED) ...
    # 1. Reset Grid
    for m, i, j in grid_m:
        grid_v[m, i, j] = [0, 0]
        grid_m[m, i, j] = 0

    # 2. P2G
    for m, p in x:
        # Skip dead/invalid particles
        if material[m, p] < 0:
            continue
        if x[m, p][0] < particle_bound or x[m, p][0] > 1.0 - particle_bound:
            continue
        if x[m, p][1] < particle_bound or x[m, p][1] > 1.0 - particle_bound:
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

            # Rigid wall boundaries (no sponge damping — closed tank)
            if i < 3 and grid_v[m, i, j][0] < 0: grid_v[m, i, j][0] = 0
            if i > n_grid - 3 and grid_v[m, i, j][0] > 0: grid_v[m, i, j][0] = 0
            if j < 3 and grid_v[m, i, j][1] < 0: grid_v[m, i, j][1] = 0
            if j > n_grid - 3 and grid_v[m, i, j][1] > 0: grid_v[m, i, j][1] = 0

    # 4. G2P
    for m, p in x:
        # Skip dead/invalid particles (need margin from edges for stencil)
        if material[m, p] < 0:
            continue
        if x[m, p][0] < particle_bound or x[m, p][0] > 1.0 - particle_bound:
            continue
        if x[m, p][1] < particle_bound or x[m, p][1] > 1.0 - particle_bound:
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
def initialize():
    for m, i in x:
        # Instance-based layout
        if i < n_particles // 2:
            # Bottom Layer: "Water"
            x[m, i] = [ti.random() * 0.4 + 0.3, ti.random() * 0.2 + 0.05]
            material[m, i] = 0 # Material 0 (Water)
            v[m, i] = [0, 0]
        else:
            # Top Layer: "Dense Fluid"
            x[m, i] = [ti.random() * 0.4 + 0.3, ti.random() * 0.2 + 0.4]
            material[m, i] = 1 # Material 1 (Dense)
            v[m, i] = [0, 1.0] # Initial downward punch
        
        F[m, i] = ti.Matrix.identity(float, 2)
        Jp[m, i] = 1

# --- RECORDING KERNEL ---
@ti.kernel
def record_frame(f: int):
    # This copies data from the 'x' field to 'history_x'
    # It stays entirely on the GPU VRAM.
    for m, p in x:
        history_x[f, m, p] = x[m, p]

@ti.kernel
def render_single_frame(f: int, res_sub: int, grid_side: int, radius: float):
    """Render one frame from history_x into frame_buffer (stays on GPU)."""
    # Clear to white
    for i, j in ti.ndrange(1024, 1024):
        frame_buffer[i, j, 0] = 1.0
        frame_buffer[i, j, 1] = 1.0
        frame_buffer[i, j, 2] = 1.0
    # Paint particles
    for m, p in ti.ndrange(n_instances, n_particles):
        pos = history_x[f, m, p]
        mat = material[m, p]
        if mat < 0 or pos[0] < 0:
            continue
        row, col = m // grid_side, m % grid_side
        center_x = (pos[0] + col) * res_sub
        center_y = ((1.0 - pos[1]) + row) * res_sub
        low_x, high_x = int(center_x - radius), int(center_x + radius)
        low_y, high_y = int(center_y - radius), int(center_y + radius)
        for px in range(low_x, high_x + 1):
            for py in range(low_y, high_y + 1):
                if 0 <= px < 1024 and 0 <= py < 1024:
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
    initialize()
    print("Starting simulation loop (Recording to VRAM)...")
    
    # Warmup
    for _ in range(10): substep()
    ti.sync()
    
    start_wall_time = time.time()
    
    for i in range(frames):
        for _ in range(substeps_per_frame):
            substep()
        record_frame(i)
        if i % 10 == 0:
            print(f"Frame {i}/{frames} recorded.")

    ti.sync()
    
# Caculate and Display Timing Metrics
    end_wall_time = time.time()
    total_time = end_wall_time - start_wall_time

    fps = frames / total_time
    fps_per_sim = fps / n_instances
    fps_per_particle = fps / (n_instances * n_particles)

    print("\n" + "="*40)
    print(f"SIMULATION PERFORMANCE REPORT")
    print("-" * 40)
    print(f"Total Simulation Time: {total_time:.4f} s")
    print(f"Final frames/s:        {fps:.2f} fps")
    print(f"frames/s/simulation:   {fps_per_sim:.4f} fps/sim")
    print(f"fps/particle:          {fps_per_particle:.10f} fps/p")
    print("="*40 + "\n")


# Render per-frame on GPU, transfer only 12MB/frame instead of 5GB bulk
    print("GPU Rendering + encoding video...")
    grid_side = int(np.ceil(np.sqrt(n_instances)))
    res_sub = 1024 // grid_side

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('gpu_rendered_sim.mp4', fourcc, 30.0, (1024, 1024))

    for f in range(frames):
        render_single_frame(f, res_sub, grid_side, 1.0)
        frame_np = frame_buffer.to_numpy()  # ~12MB per transfer
        frame_u8 = (np.clip(frame_np, 0.0, 1.0) * 255).astype(np.uint8)
        out.write(cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR))
        if f % 50 == 0:
            print(f"  Rendered frame {f}/{frames}")

    out.release()
    print("Video saved!")

if __name__ == "__main__":
    main()