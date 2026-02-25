"""
Payload sink demo: shows the payload under its own weight
without a jellyfish to carry it.

Loads the same scene into all 16 GPU instances (matching the
evolution pipeline) so rendering cost and GPU occupancy are
identical to a real evaluation — making this a fair baseline.

Usage:
    uv run python payload_sink.py                # Default 3-cycle sim
    uv run python payload_sink.py --steps 80000  # Longer run
"""

import argparse
import numpy as np
import os
import imageio.v3 as iio
import taichi as ti
from scipy.spatial import cKDTree
import mpm_sim as sim
from make_jelly import PAYLOAD_WIDTH, PAYLOAD_HEIGHT, DEFAULT_SPAWN

OUTPUT_DIR = "output"

# Web palette from evolve.py — material colors matching the web frontend
WEB_PALETTE = [
    (0, 0.933, 0.949, 0.957),  # Water
    (1, 0.678, 0.882, 0.875),  # Jelly (unused here)
    (3, 0.976, 0.757, 0.765),  # Muscle (unused here)
    (2, 0.961, 0.824, 0.576),  # Payload
]


def generate_payload_tank(max_particles, grid_res=128, spawn_offset=None):
    """
    Generate a water tank with only the payload block — no jellyfish.
    """
    if spawn_offset is None:
        spawn_offset = DEFAULT_SPAWN

    spacing = 1.0 / (grid_res * 2.0)
    margin = spacing * 3
    raster_res = grid_res * 2

    # 1. Payload particles (Material 2)
    px = np.linspace(-PAYLOAD_WIDTH / 2, PAYLOAD_WIDTH / 2,
                     int(PAYLOAD_WIDTH * raster_res))
    py = np.linspace(0, PAYLOAD_HEIGHT, int(PAYLOAD_HEIGHT * raster_res))
    pgx, pgy = np.meshgrid(px, py)
    payload_pos = np.vstack([pgx.ravel(), pgy.ravel()]).T + spawn_offset
    n_payload = len(payload_pos)

    # 2. Water grid
    wx = np.arange(margin, 1.0 - margin, spacing)
    wy = np.arange(margin, 1.0 - margin, spacing)
    wgx, wgy = np.meshgrid(wx, wy)
    water_candidates = np.vstack([wgx.ravel(), wgy.ravel()]).T

    # 3. Remove water overlapping payload
    tree = cKDTree(payload_pos)
    distances, _ = tree.query(water_candidates, k=1)
    water_pos = water_candidates[distances > 0.005]
    n_water = len(water_pos)

    if n_payload + n_water > max_particles:
        n_water = max_particles - n_payload

    # 4. Allocate fixed-size arrays
    positions = np.full((max_particles, 2), -1.0, dtype=np.float32)
    materials = np.full(max_particles, -1, dtype=np.int32)

    positions[:n_payload] = payload_pos
    materials[:n_payload] = 2  # Payload

    positions[n_payload:n_payload + n_water] = water_pos[:n_water]
    materials[n_payload:n_payload + n_water] = 0  # Water

    print(f"  Payload: {n_payload} particles at y={spawn_offset[1]:.2f}")
    print(f"  Water:   {n_water} particles")
    return positions, materials


def run_payload_sink(steps=60000, render_every=200, fps=30):
    """Run the payload-only simulation and render a video."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Payload sink demo — no jellyfish")
    print("-" * 40)

    positions, materials = generate_payload_tank(
        sim.n_particles, grid_res=int(sim.n_grid)
    )

    # Load same scene into ALL 16 instances — matches evolution pipeline
    # so GPU occupancy, grid pressure, and rendering cost are identical.
    for i in range(sim.n_instances):
        sim.load_particles(i, positions, materials)
    ti.sync()

    # Standard 4x4 grid rendering (same as evolve.py --view)
    radius = max(1.0, sim.res_sub / sim.n_grid * 0.6)

    video_path = os.path.join(OUTPUT_DIR, "payload_sink.mp4")
    total_frames = steps // render_every
    print(f"  Rendering {total_frames} frames ({steps} steps)...")

    sim.sim_time[None] = 0.0
    frames = []

    # Track payload CoM over time (instance 0)
    trajectory_y = []

    for step in range(steps):
        sim.substep()

        if step % render_every == 0:
            # Record payload position
            stats = sim.get_payload_stats()
            com_y = stats[0, 0]
            trajectory_y.append(com_y)

            # Render with web palette (flat colors, white background)
            sim.clear_frame_buffer_white()
            for mat_id, r, g, b in WEB_PALETTE:
                sim.render_flat_pass(sim.res_sub, sim.grid_side, radius,
                                     mat_id, r, g, b)
            ti.sync()

            img = sim.frame_buffer.to_numpy()
            frames.append((np.clip(img, 0, 1) * 255).astype(np.uint8))

            frame_num = step // render_every
            if frame_num % 50 == 0:
                print(f"  Frame {frame_num}/{total_frames}  "
                      f"t={sim.sim_time[None]:.3f}s  "
                      f"payload y={com_y:.4f}")

    # Final stats
    if len(trajectory_y) >= 2:
        y_start = trajectory_y[0]
        y_end = trajectory_y[-1]
        delta = y_end - y_start
        direction = "rose" if delta > 0 else "sank"
        print(f"\n  Payload {direction} from y={y_start:.4f} to y={y_end:.4f}")
        print(f"  Total change: {delta:+.4f} normalized units")

    print(f"\n  Writing video to {video_path}...")
    iio.imwrite(video_path, frames, fps=fps)
    print(f"  Done! {len(frames)} frames written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Payload sink demo — payload without jellyfish"
    )
    parser.add_argument('--steps', type=int, default=60000,
                        help="Simulation steps (default: 20000, ~1 actuation cycle)")
    args = parser.parse_args()

    run_payload_sink(steps=args.steps)
