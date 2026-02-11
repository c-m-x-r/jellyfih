"""
Fluid test visualization: oscillating paddle in water tank.

Demonstrates MPM fluid dynamics and the renderer's ability to visualize
water movement, vortex formation, and pressure waves. Useful for paper
figures showing simulation fidelity.

Usage:
    uv run python fluid_test.py              # Render fluid test video
    uv run python fluid_test.py --steps 80000  # Longer simulation
"""

import argparse
import numpy as np
import os
import imageio.v3 as iio
import taichi as ti
import mpm_sim as sim

OUTPUT_DIR = "output"

# Paddle configuration
PADDLE_WIDTH = 0.01
PADDLE_HEIGHT = 0.15
PADDLE_CENTER_Y = 0.5
PADDLE_AMPLITUDE = 0.12   # horizontal oscillation amplitude
PADDLE_FREQ = 1.5         # Hz


def generate_fluid_test(max_particles, grid_res=128):
    """
    Generate a water tank with a thin paddle at center.
    The paddle is made of stiff jelly (Material 1) so it shows in the renderer.
    """
    spacing = 1.0 / (grid_res * 2.0)
    margin = spacing * 3

    # 1. Generate paddle particles at center
    px = np.arange(0.5 - PADDLE_WIDTH / 2, 0.5 + PADDLE_WIDTH / 2, spacing)
    py = np.arange(PADDLE_CENTER_Y - PADDLE_HEIGHT / 2,
                   PADDLE_CENTER_Y + PADDLE_HEIGHT / 2, spacing)
    pgx, pgy = np.meshgrid(px, py)
    paddle_pos = np.vstack([pgx.ravel(), pgy.ravel()]).T
    n_paddle = len(paddle_pos)

    # 2. Generate water grid
    wx = np.arange(margin, 1.0 - margin, spacing)
    wy = np.arange(margin, 1.0 - margin, spacing)
    wgx, wgy = np.meshgrid(wx, wy)
    water_candidates = np.vstack([wgx.ravel(), wgy.ravel()]).T

    # 3. Remove water that overlaps with paddle
    from scipy.spatial import cKDTree
    tree = cKDTree(paddle_pos)
    distances, _ = tree.query(water_candidates, k=1)
    water_pos = water_candidates[distances > 0.005]
    n_water = len(water_pos)

    if n_paddle + n_water > max_particles:
        n_water = max_particles - n_paddle

    # 4. Allocate
    positions = np.full((max_particles, 2), -1.0, dtype=np.float32)
    materials = np.full(max_particles, -1, dtype=np.int32)

    positions[:n_paddle] = paddle_pos
    materials[:n_paddle] = 1  # Jelly (visible, elastic)

    positions[n_paddle:n_paddle + n_water] = water_pos[:n_water]
    materials[n_paddle:n_paddle + n_water] = 0  # Water

    print(f"  Paddle: {n_paddle} particles, Water: {n_water} particles")
    return positions, materials, n_paddle


def run_fluid_test(steps=60000, render_every=200, fps=30):
    """Run fluid test with oscillating paddle and render video."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fluid dynamics test visualization")
    print("-" * 40)

    # Load into all 16 instances (same scene)
    positions, materials, n_paddle = generate_fluid_test(
        sim.n_particles, grid_res=int(sim.n_grid)
    )

    for i in range(sim.n_instances):
        sim.load_particles(i, positions, materials)
        sim.instance_hue[i] = 0.55  # Blue-cyan

    ti.sync()

    # Render
    video_path = os.path.join(OUTPUT_DIR, "fluid_test.mp4")
    total_frames = steps // render_every
    radius = max(1.0, sim.res_sub / sim.n_grid * 0.6)

    print(f"  Rendering {total_frames} frames ({steps} steps)...")
    sim.sim_time[None] = 0.0
    frames = []

    # Store initial paddle positions for oscillation
    paddle_init_x = positions[:n_paddle, 0].copy()

    for step in range(steps):
        # Oscillate paddle particles before each substep
        t = sim.sim_time[None]
        offset_x = PADDLE_AMPLITUDE * np.sin(2 * np.pi * PADDLE_FREQ * t)

        # Update paddle positions on GPU by applying velocity
        # (The substep will integrate, so we set velocity to achieve oscillation)
        target_vx = PADDLE_AMPLITUDE * 2 * np.pi * PADDLE_FREQ * np.cos(
            2 * np.pi * PADDLE_FREQ * t
        )

        # Apply paddle velocity via a kernel would be ideal, but for simplicity
        # we update every N steps to avoid overhead
        if step % 10 == 0:
            set_paddle_velocity(n_paddle, float(target_vx))

        sim.substep()

        if step % render_every == 0:
            sim.clear_frame_buffer()
            sim.render_frame_abyss(sim.res_sub, sim.grid_side, radius)
            sim.tone_map_and_encode()
            ti.sync()

            img = sim.frame_buffer.to_numpy()
            frames.append((np.clip(img, 0, 1) * 255).astype(np.uint8))

            frame_num = step // render_every
            if frame_num % 50 == 0:
                print(f"  Frame {frame_num}/{total_frames} "
                      f"(t={sim.sim_time[None]:.3f}s)")

    print(f"  Writing video to {video_path}...")
    iio.imwrite(video_path, frames, fps=fps)
    print(f"  Done! {len(frames)} frames written.")


@ti.kernel
def set_paddle_velocity(n_paddle: int, target_vx: float):
    """Set horizontal velocity on paddle particles (Material 1) for all instances."""
    for m in range(sim.n_instances):
        for p in range(n_paddle):
            if sim.material[m, p] == 1:
                sim.v[m, p][0] = target_vx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fluid test visualization")
    parser.add_argument('--steps', type=int, default=60000,
                        help="Simulation steps (default: 60000)")
    args = parser.parse_args()

    run_fluid_test(steps=args.steps)
