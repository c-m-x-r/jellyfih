"""
Payload sink demo — shows the payload sinking under gravity without a jellyfish.

Drops the payload at the centre of the domain (0.5, 0.5) by default so the
buoyancy/gravity balance is unambiguous. Runs a single GPU instance for speed.

Usage:
    uv run python helpers/payload_sink.py              # 1 instance, centred, 60 K steps
    uv run python helpers/payload_sink.py --steps 20000
    uv run python helpers/payload_sink.py --spawn 0.5 0.4   # custom spawn
"""

# JELLY_INSTANCES must be set before Taichi initialises.
import os as _os
import sys as _sys
import argparse as _ap

_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument("--instances", type=int, default=1)
_pre_args, _ = _pre.parse_known_args()
_os.environ.setdefault("JELLY_INSTANCES", str(_pre_args.instances))

import argparse
import numpy as np

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import imageio.v3 as iio
import taichi as ti
from scipy.spatial import cKDTree
import mpm_sim as sim
from mpm_sim import WEB_PALETTE
from make_jelly import PAYLOAD_WIDTH, PAYLOAD_HEIGHT

OUTPUT_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "output")
RENDER_EVERY = 500


def generate_payload_tank(max_particles, grid_res=128, spawn_offset=None):
    """Water tank with only the payload block — no jellyfish."""
    if spawn_offset is None:
        spawn_offset = np.array([0.5, 0.5])  # centre of domain

    spacing = 1.0 / (grid_res * 2.0)
    margin = spacing * 3
    raster_res = grid_res * 2

    px = np.linspace(-PAYLOAD_WIDTH / 2, PAYLOAD_WIDTH / 2,
                     int(PAYLOAD_WIDTH * raster_res))
    py = np.linspace(0, PAYLOAD_HEIGHT, int(PAYLOAD_HEIGHT * raster_res))
    pgx, pgy = np.meshgrid(px, py)
    payload_pos = np.vstack([pgx.ravel(), pgy.ravel()]).T + spawn_offset
    n_payload = len(payload_pos)

    wx = np.arange(margin, 1.0 - margin, spacing)
    wy = np.arange(margin, 1.0 - margin, spacing)
    wgx, wgy = np.meshgrid(wx, wy)
    water_candidates = np.vstack([wgx.ravel(), wgy.ravel()]).T

    tree = cKDTree(payload_pos)
    distances, _ = tree.query(water_candidates, k=1)
    water_pos = water_candidates[distances > 0.005]
    n_water = min(len(water_pos), max_particles - n_payload)

    positions = np.full((max_particles, 2), -1.0, dtype=np.float32)
    materials = np.full(max_particles, -1, dtype=np.int32)

    positions[:n_payload] = payload_pos
    materials[:n_payload] = 2

    positions[n_payload:n_payload + n_water] = water_pos[:n_water]
    materials[n_payload:n_payload + n_water] = 0

    print(f"  Payload: {n_payload} particles at y={spawn_offset[1]:.2f}")
    print(f"  Water:   {n_water} particles  |  instances: {sim.n_instances}")
    return positions, materials


def run_payload_sink(steps=60000, render_every=RENDER_EVERY, fps=30,
                     spawn_offset=None):
    """Run the payload-only simulation and render a video."""
    _os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Payload sink — no jellyfish")
    print("-" * 40)

    positions, materials = generate_payload_tank(
        sim.n_particles, grid_res=int(sim.n_grid), spawn_offset=spawn_offset
    )
    fibers = np.zeros((sim.n_particles, 2), dtype=np.float32)
    fibers[:, 1] = 1.0

    for i in range(sim.n_instances):
        sim.load_particles(i, positions, materials, fibers)
    ti.sync()

    radius = max(1.0, sim.res_sub / sim.n_grid * 0.6)
    video_path = _os.path.join(OUTPUT_DIR, "payload_sink.mp4")
    total_frames = steps // render_every
    print(f"  Rendering {total_frames} frames ({steps} steps)...")

    sim.sim_time[None] = 0.0
    frames = []
    trajectory_y = []

    for step in range(steps):
        sim.substep()

        if step % render_every == 0:
            stats = sim.get_payload_stats()
            com_y = stats[0, 0]
            trajectory_y.append(com_y)

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
                      f"t={sim.sim_time[None]:.3f}s  payload y={com_y:.4f}")

    if len(trajectory_y) >= 2:
        y_start, y_end = trajectory_y[0], trajectory_y[-1]
        delta = y_end - y_start
        direction = "rose" if delta > 0 else "sank"
        print(f"\n  Payload {direction}: y={y_start:.4f} → {y_end:.4f}  "
              f"(Δ={delta:+.4f})")

    print(f"\n  Writing {len(frames)} frames to {video_path}...")
    iio.imwrite(video_path, frames, fps=fps)
    print("  Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Payload sink demo — payload without jellyfish"
    )
    parser.add_argument("--steps", type=int, default=60000,
                        help="Simulation steps (default: 60000 = 3 cycles)")
    parser.add_argument("--spawn", type=float, nargs=2, default=[0.5, 0.5],
                        metavar=("X", "Y"),
                        help="Spawn position (default: 0.5 0.5 = centre)")
    parser.add_argument("--instances", type=int, default=_pre_args.instances,
                        help="GPU instances (default: 1)")
    args = parser.parse_args()

    run_payload_sink(steps=args.steps,
                     spawn_offset=np.array(args.spawn))
